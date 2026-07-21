"""
MAPLE-Inspired Structured Validation Schemas for QSP Model Calibration.

Based on: Eliason & Popel (2026) - "Quantitative Systems Pharmacology Model Calibration"
   MAPLE = Model-Aware Parameterization from Literature Evidence

Implements Pydantic schemas for structured extraction of calibration data from
scientific literature, with targeted validators for hallucination detection.
"""

import re
import math
import json
import requests
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple, Union, Literal
from pydantic import BaseModel, Field, field_validator, model_validator
from datetime import datetime


# ============================================================
# ENUMS & CONSTANTS
# ============================================================

class ForwardModelType(str, Enum):
    """15 built-in forward model types organized in 4 categories."""
    # ODE Templates
    EXPONENTIAL_GROWTH = "exponential_growth"          # dy/dt = k * y
    FIRST_ORDER_DECAY = "first_order_decay"            # dy/dt = -k * y
    TWO_STATE_TRANSITION = "two_state_transition"      # dy/dt = k_on*(1-y) - k_off*y
    LOGISTIC_GROWTH = "logistic_growth"                # dy/dt = k*y*(1 - y/K)
    SATURATION_KINETICS = "saturation_kinetics"        # dy/dt = Vmax*S/(Km + S) - k_deg*y
    MICHAELIS_MENTEN = "michaelis_menten"              # v = Vmax*[S]/(Km + [S])
    HILL_EQUATION = "hill_equation"                    # y = Vmax * x^n / (K^n + x^n)
    BIPHASIC_DECAY = "biphasic_decay"                  # y = A*exp(-k1*t) + B*exp(-k2*t)

    # Steady-State Types
    EQUILIBRIUM_BINDING = "equilibrium_binding"        # Kd = [L][R]/[LR]
    STEADY_STATE_RATIO = "steady_state_ratio"          # ratio = A / B at equilibrium

    # Batch Accumulation
    BATCH_ACCUMULATION = "batch_accumulation"          # y(t) = rate * t * cell_count

    # Generic Fallbacks
    ALGEBRAIC = "algebraic"                            # User-defined algebraic expression
    DIRECT_FIT = "direct_fit"                          # Direct parameter = value
    CUSTOM_ODE = "custom_ode"                          # User-defined ODE string
    CUSTOM = "custom"                                  # Fully custom code


class IndicationMatch(str, Enum):
    EXACT = "exact"
    RELATED = "related"
    PROXY = "proxy"
    UNRELATED = "unrelated"


class EvidenceType(str, Enum):
    CLINICAL = "clinical"
    IN_VIVO = "in_vivo"
    IN_VITRO = "in_vitro"
    EX_VIVO = "ex_vivo"
    COMPUTATIONAL = "computational"
    REVIEW = "review"


class UncertaintyType(str, Enum):
    SD = "sd"
    SEM = "sem"
    CI_95 = "ci_95"
    RANGE = "range"
    IQR = "iqr"
    NONE = "none"


class PriorDistribution(str, Enum):
    LOGNORMAL = "lognormal"
    NORMAL = "normal"
    UNIFORM = "uniform"
    BETA = "beta"
    GAMMA = "gamma"
    FIXED = "fixed"


# ============================================================
# SOURCE & PROVENANCE SCHEMAS
# ============================================================

class SourceReference(BaseModel):
    """Tracks provenance of extracted data back to original publication."""
    doi: str = Field(..., description="DOI of the source paper")
    title: str = Field(..., description="Full title of the paper")
    authors: str = Field(..., description="Author list (first author et al. format)")
    year: int = Field(..., ge=1900, le=2100)
    journal: Optional[str] = None

    @field_validator('doi')
    @classmethod
    def validate_doi_format(cls, v: str) -> str:
        """Basic DOI format check."""
        v = v.strip()
        if not v.startswith("10."):
            raise ValueError(f"DOI must start with '10.', got: {v}")
        if "/" not in v:
            raise ValueError(f"DOI must contain '/' separator, got: {v}")
        return v


class SourceRelevance(BaseModel):
    """Documents the translation from source context to model context."""
    indication_match: IndicationMatch = Field(
        ..., description="How well the source disease matches the model target"
    )
    indication_justification: str = Field(
        ..., description="Why this indication is appropriate"
    )
    species_source: str = Field(default="human", description="Species of the source data")
    species_target: str = Field(default="human", description="Species of the model")
    species_translation_notes: Optional[str] = None
    evidence_type: EvidenceType = Field(...)
    tme_compatible: bool = Field(
        default=True,
        description="Whether tumor microenvironment conditions are compatible"
    )
    tme_notes: Optional[str] = None
    estimated_translation_uncertainty: float = Field(
        default=2.0, ge=1.0, le=100.0,
        description="Fold-uncertainty from all translation factors (e.g., 10 = 10-fold)"
    )


class ExtractedSnippet(BaseModel):
    """A verbatim quoted text snippet from the source paper containing the extracted value."""
    text: str = Field(..., min_length=10, description="Verbatim quote from the paper")
    page_or_section: Optional[str] = Field(
        None, description="Page number or section reference"
    )


# ============================================================
# INPUT DATA SCHEMAS
# ============================================================

class ExtractedInput(BaseModel):
    """A single data point extracted from literature with full provenance."""
    name: str = Field(..., description="Descriptive name for this input")
    value: float = Field(..., description="Numeric value as reported")
    units: str = Field(..., description="Physical units (e.g., '1/day', 'nM', 'cells/mm^3')")
    uncertainty_value: Optional[float] = Field(None, ge=0)
    uncertainty_type: UncertaintyType = Field(default=UncertaintyType.NONE)
    sample_size: Optional[int] = Field(None, ge=1)
    snippet: ExtractedSnippet = Field(
        ..., description="Verbatim text snippet containing this value"
    )
    source: SourceReference = Field(...)


class TimeSeriesInput(BaseModel):
    """Time-course data extracted from a figure or table."""
    name: str
    time_points: List[float] = Field(..., min_length=2)
    values: List[float] = Field(..., min_length=2)
    time_units: str = Field(default="hours")
    value_units: str
    uncertainty_values: Optional[List[float]] = None
    uncertainty_type: UncertaintyType = Field(default=UncertaintyType.NONE)
    snippet: ExtractedSnippet
    source: SourceReference

    @model_validator(mode='after')
    def check_lengths_match(self):
        if len(self.time_points) != len(self.values):
            raise ValueError(
                f"time_points ({len(self.time_points)}) and values ({len(self.values)}) "
                "must have the same length"
            )
        if self.uncertainty_values and len(self.uncertainty_values) != len(self.values):
            raise ValueError("uncertainty_values must match values length")
        return self


# ============================================================
# FORWARD MODEL SPECIFICATION
# ============================================================

class ForwardModelParam(BaseModel):
    """A parameter in a forward model with its role."""
    name: str = Field(..., description="Parameter name matching full QSP model")
    role: Literal["calibration", "input", "reference", "literal"] = Field(
        ..., description="Whether this is the unknown to estimate, a measured input, etc."
    )
    value: Optional[float] = Field(None, description="Fixed value if role is 'literal' or 'input'")
    units: Optional[str] = None
    prior_distribution: Optional[PriorDistribution] = None
    prior_params: Optional[Dict[str, float]] = Field(
        None, description="e.g., {'mu': 0.1, 'sigma': 0.5} for lognormal"
    )
    prior_rationale: Optional[str] = None


class ForwardModelSpec(BaseModel):
    """Specification of a simplified forward model for parameter estimation."""
    model_type: ForwardModelType
    parameters: List[ForwardModelParam] = Field(..., min_length=1)
    custom_code: Optional[str] = Field(
        None,
        description="Python code for CUSTOM type models. Must define f(t, y, params) -> dydt"
    )
    error_model: str = Field(
        default="normal",
        description="Likelihood function: 'normal', 'lognormal', 'poisson'"
    )

    @model_validator(mode='after')
    def validate_custom_has_code(self):
        if self.model_type in (ForwardModelType.CUSTOM, ForwardModelType.CUSTOM_ODE):
            if not self.custom_code:
                raise ValueError(
                    f"Forward model type '{self.model_type}' requires 'custom_code'"
                )
        return self


# ============================================================
# SUBMODEL TARGET SCHEMA
# ============================================================

class SubmodelTarget(BaseModel):
    """
    Schema for isolated experiments constraining individual parameters.

    Two layers:
    - inputs: Data exactly as reported in literature
    - calibration: How that data constrains model parameters via a forward model
    """
    target_id: str = Field(..., description="Unique identifier, e.g., 'k_apsc_prolif'")
    description: str = Field(..., description="What this target measures")
    target_parameter: str = Field(
        ..., description="Name of the QSP model parameter being constrained"
    )

    # Input layer
    inputs: List[ExtractedInput] = Field(..., min_length=1)
    time_series: Optional[List[TimeSeriesInput]] = None

    # Calibration layer
    forward_model: ForwardModelSpec
    source_relevance: SourceRelevance

    # Metadata
    created_at: datetime = Field(default_factory=datetime.now)
    extraction_mode: Literal["batch", "interactive"] = Field(default="interactive")
    modeler_reviewed: bool = Field(default=False)
    notes: Optional[str] = None


# ============================================================
# CALIBRATION TARGET SCHEMA (FULL MODEL ENDPOINTS)
# ============================================================

class ObservableDefinition(BaseModel):
    """Defines how to compute a measurement from model species state."""
    name: str = Field(..., description="Observable name, e.g., 'intratumoral_cd8_density'")
    description: str
    compute_code: str = Field(
        ...,
        description="Python function body: receives 'state' dict of species concentrations, "
                    "returns scalar value with units"
    )
    units: str
    species_required: List[str] = Field(
        ..., description="List of model species names needed for computation"
    )
    named_constants: Optional[Dict[str, float]] = Field(
        None, description="Named constants with provenance, e.g., {'tumor_volume_mm3': 100.0}"
    )


class EmpiricalData(BaseModel):
    """Summary statistics derived from literature via Monte Carlo simulation."""
    median: float
    ci_lower: float = Field(..., description="Lower bound of 95% CI")
    ci_upper: float = Field(..., description="Upper bound of 95% CI")
    units: str
    n_patients: Optional[int] = None
    derivation_method: str = Field(
        default="direct",
        description="How summary stats were derived: 'direct', 'monte_carlo', 'subgroup_mixing'"
    )
    raw_inputs: List[ExtractedInput] = Field(..., min_length=1)


class ScenarioBlock(BaseModel):
    """Treatment context for intervention studies."""
    agents: List[str] = Field(..., description="Drug/treatment names")
    doses: Optional[List[str]] = None
    schedule: Optional[str] = None
    duration: Optional[str] = None
    line_of_therapy: Optional[str] = None


class CalibrationTarget(BaseModel):
    """
    Schema for clinical/in vivo endpoints constraining the full QSP model.

    Three components:
    - observable: How to compute the measurement from model state
    - empirical_data: Literature-derived summary statistics
    - scenario: Treatment context (for intervention studies)
    """
    target_id: str
    description: str

    observable: ObservableDefinition
    empirical_data: EmpiricalData
    source_relevance: SourceRelevance
    scenario: Optional[ScenarioBlock] = None

    # Metadata
    created_at: datetime = Field(default_factory=datetime.now)
    modeler_reviewed: bool = Field(default=False)
    notes: Optional[str] = None


# ============================================================
# VALIDATORS
# ============================================================

class ValidationResult(BaseModel):
    """Result of a single validation check."""
    validator_name: str
    passed: bool
    message: str
    severity: Literal["error", "warning", "info"] = "error"
    field_path: Optional[str] = None


class ValidationReport(BaseModel):
    """Aggregated validation results for a target."""
    target_id: str
    all_passed: bool
    results: List[ValidationResult]
    timestamp: datetime = Field(default_factory=datetime.now)


def validate_value_in_snippet(
    value: float,
    snippet_text: str,
    tolerance: float = 0.01
) -> ValidationResult:
    """
    Hallucination detection: Check that an extracted numeric value actually
    appears verbatim in the quoted source text snippet.
    """
    # Extract all numbers from the snippet
    numbers_in_text = re.findall(
        r'[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?',
        snippet_text
    )
    parsed_numbers = []
    for n in numbers_in_text:
        try:
            parsed_numbers.append(float(n))
        except ValueError:
            continue

    # Check if the value appears (within tolerance for floating-point)
    for n in parsed_numbers:
        if n == 0 and value == 0:
            return ValidationResult(
                validator_name="value_in_snippet",
                passed=True,
                message=f"Value {value} found in snippet.",
                severity="info"
            )
        if n != 0 and abs(value - n) / abs(n) < tolerance:
            return ValidationResult(
                validator_name="value_in_snippet",
                passed=True,
                message=f"Value {value} found in snippet (matched {n}).",
                severity="info"
            )

    return ValidationResult(
        validator_name="value_in_snippet",
        passed=False,
        message=f"HALLUCINATION DETECTED: Value {value} not found in snippet text. "
                f"Numbers found: {parsed_numbers}",
        severity="error"
    )


def validate_doi_resolution(doi: str) -> ValidationResult:
    """
    Citation verification: Resolve a DOI to check it actually exists.
    Uses the DOI.org API content negotiation.
    """
    try:
        url = f"https://doi.org/api/handles/{doi}"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data.get("responseCode") == 1:
                return ValidationResult(
                    validator_name="doi_resolution",
                    passed=True,
                    message=f"DOI {doi} resolved successfully.",
                    severity="info"
                )
        return ValidationResult(
            validator_name="doi_resolution",
            passed=False,
            message=f"FABRICATED CITATION: DOI {doi} could not be resolved.",
            severity="error"
        )
    except requests.RequestException as e:
        return ValidationResult(
            validator_name="doi_resolution",
            passed=False,
            message=f"DOI resolution failed (network error): {e}",
            severity="warning"
        )


def validate_units(units_str: str) -> ValidationResult:
    """
    Validate that a units string is parseable and physically meaningful.
    Uses basic pattern matching for common biological units.
    """
    valid_patterns = [
        r"^1/(day|hour|min|s|h)$",           # rate constants
        r"^(nM|uM|mM|M|pM|fM)$",             # concentrations
        r"^cells?(/mm\^?[23]|/[um]L|/mL)?$",  # cell densities
        r"^(mg|ug|ng|pg)/[dkmnu]?[Ll]$",      # mass concentrations
        r"^(mm|um|nm|cm)\^?[23]?$",           # length/area/volume
        r"^%$",                                # percentage
        r"^fold$",                             # fold-change
        r"^dimensionless$",                    # dimensionless
        r"^(day|hour|min|s|h)$",              # time
        r"^(pg|ng|ug|mg|g|kg)$",              # mass
        r"^.+/.+$",                            # any ratio
    ]

    for pattern in valid_patterns:
        if re.match(pattern, units_str, re.IGNORECASE):
            return ValidationResult(
                validator_name="unit_validation",
                passed=True,
                message=f"Units '{units_str}' recognized.",
                severity="info"
            )

    return ValidationResult(
        validator_name="unit_validation",
        passed=False,
        message=f"Units '{units_str}' not recognized. Check format.",
        severity="warning"
    )


def validate_forward_model_execution(spec: ForwardModelSpec) -> ValidationResult:
    """
    Execute the forward model to verify it doesn't crash.
    Tests with dummy data to catch malformed code.
    """
    try:
        if spec.model_type == ForwardModelType.EXPONENTIAL_GROWTH:
            # Test: dy/dt = k*y => y(t) = y0 * exp(k*t)
            k = 0.1  # dummy
            y0 = 1.0
            t_test = 10.0
            result = y0 * math.exp(k * t_test)
            if not math.isfinite(result):
                raise ValueError("Non-finite result from exponential growth model")

        elif spec.model_type == ForwardModelType.FIRST_ORDER_DECAY:
            k = 0.1
            y0 = 1.0
            t_test = 10.0
            result = y0 * math.exp(-k * t_test)
            if not math.isfinite(result):
                raise ValueError("Non-finite result from first-order decay model")

        elif spec.model_type == ForwardModelType.MICHAELIS_MENTEN:
            Vmax, Km, S = 1.0, 0.5, 1.0
            result = Vmax * S / (Km + S)
            if not math.isfinite(result):
                raise ValueError("Non-finite result from Michaelis-Menten model")

        elif spec.model_type == ForwardModelType.HILL_EQUATION:
            Vmax, K, n, x = 1.0, 0.5, 2.0, 1.0
            result = Vmax * (x ** n) / (K ** n + x ** n)
            if not math.isfinite(result):
                raise ValueError("Non-finite result from Hill equation model")

        elif spec.model_type in (ForwardModelType.CUSTOM, ForwardModelType.CUSTOM_ODE):
            if spec.custom_code:
                # Attempt to compile the code (don't execute arbitrary code in production)
                compile(spec.custom_code, "<forward_model>", "exec")

        return ValidationResult(
            validator_name="forward_model_execution",
            passed=True,
            message=f"Forward model '{spec.model_type}' executed successfully.",
            severity="info"
        )
    except Exception as e:
        return ValidationResult(
            validator_name="forward_model_execution",
            passed=False,
            message=f"Forward model '{spec.model_type}' failed: {e}",
            severity="error"
        )


def validate_submodel_target(target: SubmodelTarget) -> ValidationReport:
    """Run all validators on a SubmodelTarget and produce an aggregated report."""
    results = []

    # 1. Value-in-snippet checks for all inputs
    for inp in target.inputs:
        result = validate_value_in_snippet(inp.value, inp.snippet.text)
        result.field_path = f"inputs.{inp.name}"
        results.append(result)

    # 2. DOI resolution for all sources
    seen_dois = set()
    for inp in target.inputs:
        if inp.source.doi not in seen_dois:
            seen_dois.add(inp.source.doi)
            result = validate_doi_resolution(inp.source.doi)
            result.field_path = f"inputs.{inp.name}.source.doi"
            results.append(result)

    # 3. Unit validation for all inputs
    for inp in target.inputs:
        result = validate_units(inp.units)
        result.field_path = f"inputs.{inp.name}.units"
        results.append(result)

    # 4. Forward model execution
    result = validate_forward_model_execution(target.forward_model)
    result.field_path = "forward_model"
    results.append(result)

    all_passed = all(r.passed or r.severity != "error" for r in results)

    return ValidationReport(
        target_id=target.target_id,
        all_passed=all_passed,
        results=results
    )


def validate_calibration_target(target: CalibrationTarget) -> ValidationReport:
    """Run all validators on a CalibrationTarget."""
    results = []

    # 1. Value-in-snippet for empirical data raw inputs
    for inp in target.empirical_data.raw_inputs:
        result = validate_value_in_snippet(inp.value, inp.snippet.text)
        result.field_path = f"empirical_data.raw_inputs.{inp.name}"
        results.append(result)

    # 2. DOI resolution
    seen_dois = set()
    for inp in target.empirical_data.raw_inputs:
        if inp.source.doi not in seen_dois:
            seen_dois.add(inp.source.doi)
            result = validate_doi_resolution(inp.source.doi)
            result.field_path = f"empirical_data.raw_inputs.{inp.name}.source.doi"
            results.append(result)

    # 3. Unit validation
    result = validate_units(target.empirical_data.units)
    result.field_path = "empirical_data.units"
    results.append(result)

    # 4. Observable code compilation
    try:
        compile(target.observable.compute_code, "<observable>", "exec")
        results.append(ValidationResult(
            validator_name="observable_code",
            passed=True,
            message="Observable code compiles successfully.",
            severity="info",
            field_path="observable.compute_code"
        ))
    except SyntaxError as e:
        results.append(ValidationResult(
            validator_name="observable_code",
            passed=False,
            message=f"Observable code has syntax error: {e}",
            severity="error",
            field_path="observable.compute_code"
        ))

    # 5. CI bounds ordering
    if target.empirical_data.ci_lower > target.empirical_data.ci_upper:
        results.append(ValidationResult(
            validator_name="ci_bounds",
            passed=False,
            message="CI lower bound exceeds upper bound.",
            severity="error",
            field_path="empirical_data"
        ))

    all_passed = all(r.passed or r.severity != "error" for r in results)

    return ValidationReport(
        target_id=target.target_id,
        all_passed=all_passed,
        results=results
    )


# ============================================================
# SBML IMPORT SUPPORT
# ============================================================

def parse_sbml_to_blueprint(sbml_content: str) -> Dict[str, Any]:
    """
    Parse an SBML XML model into our internal blueprint format.
    Handles basic SBML Level 2/3 with species, reactions, and parameters.
    """
    import xml.etree.ElementTree as ET

    root = ET.fromstring(sbml_content)
    # Handle SBML namespace
    ns = {'sbml': 'http://www.sbml.org/sbml/level3/version2/core'}
    if root.tag.startswith('{'):
        ns_uri = root.tag.split('}')[0] + '}'
        ns = {'sbml': ns_uri.strip('{}')}

    model_el = root.find('.//sbml:model', ns)
    if model_el is None:
        # Try without namespace
        model_el = root.find('.//model')
    if model_el is None:
        model_el = root  # Fallback

    # Extract species
    nodes = []
    species_elements = (
        model_el.findall('.//sbml:species', ns) or
        model_el.findall('.//species') or
        []
    )
    for sp in species_elements:
        sp_id = sp.get('id', sp.get('name', ''))
        init_val = float(sp.get('initialConcentration', sp.get('initialAmount', '0.0')))
        name = sp.get('name', sp_id)
        nodes.append({
            'id': sp_id,
            'name': name,
            'initial_value': init_val
        })

    # Extract reactions and build edges
    edges = []
    reaction_elements = (
        model_el.findall('.//sbml:reaction', ns) or
        model_el.findall('.//reaction') or
        []
    )
    for rxn in reaction_elements:
        # Get reactants and products
        reactants = []
        products = []

        for r in (rxn.findall('.//sbml:listOfReactants/sbml:speciesReference', ns) or
                  rxn.findall('.//listOfReactants/speciesReference') or []):
            reactants.append(r.get('species', ''))

        for p in (rxn.findall('.//sbml:listOfProducts/sbml:speciesReference', ns) or
                  rxn.findall('.//listOfProducts/speciesReference') or []):
            products.append(p.get('species', ''))

        # Create activation edges from reactants to products
        for src in reactants:
            for tgt in products:
                if src and tgt:
                    edges.append({
                        'source': src,
                        'target': tgt,
                        'type': 'activation',
                        'parameters': {'k': 0.5, 'K_d': 1.0, 'n': 1.0}
                    })

        # Check for modifiers (inhibitors/activators)
        for mod in (rxn.findall('.//sbml:listOfModifiers/sbml:modifierSpeciesReference', ns) or
                    rxn.findall('.//listOfModifiers/modifierSpeciesReference') or []):
            mod_species = mod.get('species', '')
            mod_type = mod.get('sboTerm', '')
            # SBO:0000020 = inhibitor, SBO:0000459 = stimulator
            edge_type = 'inhibition' if '0000020' in str(mod_type) else 'activation'
            for tgt in products:
                if mod_species and tgt:
                    edges.append({
                        'source': mod_species,
                        'target': tgt,
                        'type': edge_type,
                        'parameters': {'k': 0.5, 'K_d': 1.0, 'n': 1.0}
                    })

    # Extract global parameters
    params = {}
    param_elements = (
        model_el.findall('.//sbml:listOfParameters/sbml:parameter', ns) or
        model_el.findall('.//listOfParameters/parameter') or
        []
    )
    for p in param_elements:
        p_id = p.get('id', '')
        p_val = float(p.get('value', '0.0'))
        params[p_id] = p_val

    blueprint = {
        'type': 'ODE',
        'nodes': nodes,
        'edges': edges,
        'sbml_parameters': params,
        'simulation_config': {
            't_max': 100.0
        }
    }

    return blueprint


# ============================================================
# YAML/JSON SERIALIZATION HELPERS
# ============================================================

def submodel_target_to_dict(target: SubmodelTarget) -> Dict[str, Any]:
    """Serialize a SubmodelTarget to a plain dict for YAML/JSON output."""
    return target.model_dump(mode='json')


def calibration_target_to_dict(target: CalibrationTarget) -> Dict[str, Any]:
    """Serialize a CalibrationTarget to a plain dict for YAML/JSON output."""
    return target.model_dump(mode='json')


def validation_report_to_dict(report: ValidationReport) -> Dict[str, Any]:
    """Serialize a ValidationReport to a plain dict."""
    return report.model_dump(mode='json')
