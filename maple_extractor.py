"""
MAPLE Parameter Extraction Pipeline.

Uses an open-source LLM (a local GGUF model via llama-cpp-python, or a remote
OpenAI-compatible endpoint) to extract calibration parameters from scientific
literature with structured validation and retry loops.
"""

import json
import re
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

import llm_provider

from maple_schemas import (
    SubmodelTarget, CalibrationTarget, ForwardModelType,
    validate_submodel_target, validate_calibration_target,
    ValidationReport, submodel_target_to_dict, calibration_target_to_dict,
    validation_report_to_dict, parse_sbml_to_blueprint
)


# ============================================================
# PROMPT TEMPLATES
# ============================================================

SUBMODEL_EXTRACTION_PROMPT = """You are a biomedical data extraction agent specializing in QSP model calibration.
Your task is to extract a structured calibration target from the given information.

## Target Parameter
- **Parameter name**: {param_name}
- **Parameter units**: {param_units}
- **Description**: {param_description}
- **Mechanistic context**: {mechanistic_context}

## Instructions
Extract data from the provided text or your knowledge to fill a SubmodelTarget schema.

CRITICAL RULES:
1. Every numeric value you report MUST appear verbatim in a quoted text snippet
2. Do NOT fabricate or hallucinate values; use only what is explicitly stated
3. Include DOI, title, authors, and year for every source
4. Specify the forward model type that best fits the data
5. Assess source relevance honestly (indication match, species translation)
6. When converting SEM to SD, use: SD = SEM * sqrt(n)

## Schema Format
Return a JSON object with this structure:
{{
    "target_id": "string, a unique ID like 'k_prolif_cancer'",
    "description": "string, what this target measures",
    "target_parameter": "{param_name}",
    "inputs": [
        {{
            "name": "string, a descriptive name",
            "value": float,
            "units": "string",
            "uncertainty_value": float or null,
            "uncertainty_type": "sd|sem|ci_95|range|iqr|none",
            "sample_size": int or null,
            "snippet": {{
                "text": "verbatim quote from paper containing the value",
                "page_or_section": "string or null"
            }},
            "source": {{
                "doi": "10.xxxx/yyyy",
                "title": "Full paper title",
                "authors": "First Author et al.",
                "year": int,
                "journal": "string or null"
            }}
        }}
    ],
    "forward_model": {{
        "model_type": "exponential_growth|first_order_decay|michaelis_menten|hill_equation|direct_fit|algebraic|...",
        "parameters": [
            {{
                "name": "parameter_name",
                "role": "calibration|input|reference|literal",
                "value": float or null,
                "units": "string or null",
                "prior_distribution": "lognormal|normal|uniform|fixed|null",
                "prior_params": {{"mu": float, "sigma": float}} or null,
                "prior_rationale": "string or null"
            }}
        ],
        "error_model": "normal|lognormal"
    }},
    "source_relevance": {{
        "indication_match": "exact|related|proxy|unrelated",
        "indication_justification": "string",
        "species_source": "human|mouse|rat|...",
        "species_target": "human",
        "evidence_type": "clinical|in_vivo|in_vitro|ex_vivo|computational|review",
        "tme_compatible": true|false,
        "estimated_translation_uncertainty": float (fold, e.g., 2.0)
    }}
}}

Return ONLY raw JSON, no markdown code blocks or extra text.
"""

CALIBRATION_EXTRACTION_PROMPT = """You are a biomedical data extraction agent.
Extract a CalibrationTarget for a clinical/in vivo endpoint.

## Observable
- **Name**: {observable_name}
- **Description**: {observable_description}
- **Units**: {observable_units}
- **Species required**: {species_list}

## Instructions
Return a JSON CalibrationTarget with:
1. An observable definition (with Python code to compute from model state)
2. Empirical data (median, 95% CI) derived from literature
3. Source relevance assessment
4. Treatment scenario (if applicable)

CRITICAL: All values must have verbatim snippet provenance.

Return ONLY raw JSON, no markdown code blocks or extra text.
"""


# ============================================================
# EXTRACTION ENGINE
# ============================================================

class MAPLEExtractor:
    """LLM-powered parameter extraction with structured validation."""

    def __init__(self, llm: Optional[Dict[str, Any]] = None):
        self.llm = llm
        self.client = None
        if llm_provider.wants_llm(llm):
            try:
                self.client = llm_provider.build_client(llm)
            except Exception:
                self.client = None
        self.extraction_history: List[Dict[str, Any]] = []

    def extract_submodel_target(
        self,
        param_name: str,
        param_units: str = "",
        param_description: str = "",
        mechanistic_context: str = "",
        max_retries: int = 3
    ) -> Tuple[Optional[SubmodelTarget], ValidationReport, List[str]]:
        """
        Extract a SubmodelTarget using LLM with validation retry loop.

        Returns: (target_or_None, validation_report, retry_logs)
        """
        logs = []

        if not self.client:
            logs.append("No LLM model configured. Returning demo target.")
            target = self._get_demo_submodel_target(param_name)
            report = validate_submodel_target(target)
            return target, report, logs

        prompt = SUBMODEL_EXTRACTION_PROMPT.format(
            param_name=param_name,
            param_units=param_units,
            param_description=param_description,
            mechanistic_context=mechanistic_context
        )

        target = None
        report = None

        for attempt in range(max_retries):
            logs.append(f"Extraction attempt {attempt + 1}/{max_retries}")

            try:
                data = llm_provider.generate_json(self.client, prompt)
                target = SubmodelTarget(**data)
                logs.append(f"Schema validation passed on attempt {attempt + 1}.")

                # Run content validators
                report = validate_submodel_target(target)

                if report.all_passed:
                    logs.append("All validators passed!")
                    break
                else:
                    # Collect error messages for retry prompt
                    errors = [r.message for r in report.results if not r.passed]
                    logs.append(f"Validation errors: {errors}")

                    # Augment prompt with error feedback for retry
                    prompt += f"\n\nPREVIOUS ATTEMPT FAILED VALIDATION:\n"
                    prompt += "\n".join(f"- {e}" for e in errors)
                    prompt += "\n\nPlease fix these issues and try again."

            except (json.JSONDecodeError, llm_provider.LLMError) as e:
                logs.append(f"JSON parse error: {e}")
                prompt += f"\n\nYour previous response was not valid JSON. Error: {e}"
            except Exception as e:
                logs.append(f"Schema validation error: {e}")
                prompt += f"\n\nSchema validation error: {e}\nPlease fix and retry."

        if target is None:
            logs.append("All retries exhausted. Returning demo target.")
            target = self._get_demo_submodel_target(param_name)
            report = validate_submodel_target(target)

        # Record extraction
        self.extraction_history.append({
            'target_id': target.target_id,
            'timestamp': datetime.now().isoformat(),
            'attempts': min(max_retries, len(logs)),
            'passed': report.all_passed if report else False
        })

        return target, report, logs

    def extract_calibration_target(
        self,
        observable_name: str,
        observable_description: str = "",
        observable_units: str = "",
        species_list: str = "",
        max_retries: int = 3
    ) -> Tuple[Optional[CalibrationTarget], ValidationReport, List[str]]:
        """Extract a CalibrationTarget using LLM with validation."""
        logs = []

        if not self.client:
            logs.append("No LLM model configured. Returning demo calibration target.")
            target = self._get_demo_calibration_target(observable_name)
            report = validate_calibration_target(target)
            return target, report, logs

        prompt = CALIBRATION_EXTRACTION_PROMPT.format(
            observable_name=observable_name,
            observable_description=observable_description,
            observable_units=observable_units,
            species_list=species_list
        )

        target = None
        report = None

        for attempt in range(max_retries):
            logs.append(f"Extraction attempt {attempt + 1}/{max_retries}")
            try:
                data = llm_provider.generate_json(self.client, prompt)
                target = CalibrationTarget(**data)
                report = validate_calibration_target(target)

                if report.all_passed:
                    logs.append("All validators passed!")
                    break
                else:
                    errors = [r.message for r in report.results if not r.passed]
                    logs.append(f"Validation errors: {errors}")
                    prompt += f"\n\nFAILED VALIDATION:\n" + "\n".join(f"- {e}" for e in errors)

            except Exception as e:
                logs.append(f"Error: {e}")
                prompt += f"\n\nError: {e}\nPlease fix and retry."

        if target is None:
            target = self._get_demo_calibration_target(observable_name)
            report = validate_calibration_target(target)

        return target, report, logs

    def import_biomodels_sbml(self, model_id: str) -> Tuple[Optional[Dict], List[str]]:
        """
        Fetch and parse an SBML model from BioModels repository.
        
        Args:
            model_id: BioModels identifier (e.g., 'BIOMD0000000006')
        
        Returns:
            (blueprint_dict, logs)
        """
        import requests
        logs = []

        try:
            # BioModels API endpoint
            url = f"https://www.ebi.ac.uk/biomodels/{model_id}/download"
            logs.append(f"Fetching SBML from BioModels: {model_id}")

            response = requests.get(url, timeout=15, params={"filename": f"{model_id}_url.xml"})
            if response.status_code != 200:
                # Try alternative URL format
                url = f"https://www.ebi.ac.uk/biomodels/model/download/{model_id}"
                response = requests.get(url, timeout=15)

            if response.status_code == 200:
                sbml_content = response.text
                logs.append(f"Downloaded SBML ({len(sbml_content)} chars)")

                blueprint = parse_sbml_to_blueprint(sbml_content)
                logs.append(f"Parsed {len(blueprint['nodes'])} species, {len(blueprint['edges'])} reactions")
                return blueprint, logs
            else:
                logs.append(f"BioModels API returned {response.status_code}")
                return None, logs
        except Exception as e:
            logs.append(f"SBML import failed: {e}")
            return None, logs

    # ============================================================
    # DEMO/FALLBACK TARGETS
    # ============================================================

    @staticmethod
    def _get_demo_submodel_target(param_name: str) -> SubmodelTarget:
        """Generate a demonstration SubmodelTarget for the given parameter."""
        return SubmodelTarget(
            target_id=f"demo_{param_name}",
            description=f"Demonstration extraction for {param_name}",
            target_parameter=param_name,
            inputs=[{
                "name": f"{param_name}_measurement",
                "value": 0.035,
                "units": "1/day",
                "uncertainty_value": 0.008,
                "uncertainty_type": "sd",
                "sample_size": 12,
                "snippet": {
                    "text": "The proliferation rate was measured at 0.035 ± 0.008 per day (n=12) using Ki-67 staining.",
                    "page_or_section": "Results, Section 3.2"
                },
                "source": {
                    "doi": "10.1038/s41586-024-07421-0",
                    "title": "Example Study on Cell Proliferation Rates",
                    "authors": "Smith et al.",
                    "year": 2024,
                    "journal": "Nature"
                }
            }],
            forward_model={
                "model_type": "exponential_growth",
                "parameters": [
                    {
                        "name": param_name,
                        "role": "calibration",
                        "units": "1/day",
                        "prior_distribution": "lognormal",
                        "prior_params": {"mu": -3.35, "sigma": 0.5},
                        "prior_rationale": "Based on reported mean and uncertainty"
                    }
                ],
                "error_model": "normal"
            },
            source_relevance={
                "indication_match": "exact",
                "indication_justification": "Study directly measures target parameter in relevant cell type",
                "species_source": "human",
                "species_target": "human",
                "evidence_type": "in_vitro",
                "tme_compatible": True,
                "estimated_translation_uncertainty": 2.0
            }
        )

    @staticmethod
    def _get_demo_calibration_target(obs_name: str) -> CalibrationTarget:
        """Generate a demonstration CalibrationTarget."""
        return CalibrationTarget(
            target_id=f"demo_{obs_name}",
            description=f"Demonstration calibration target for {obs_name}",
            observable={
                "name": obs_name,
                "description": f"Observable measurement: {obs_name}",
                "compute_code": "result = state.get('tumor_cells', 0) / state.get('volume_mm3', 1.0)",
                "units": "cells/mm^3",
                "species_required": ["tumor_cells"]
            },
            empirical_data={
                "median": 1500.0,
                "ci_lower": 800.0,
                "ci_upper": 2200.0,
                "units": "cells/mm^3",
                "n_patients": 45,
                "derivation_method": "direct",
                "raw_inputs": [{
                    "name": f"{obs_name}_clinical_data",
                    "value": 1500.0,
                    "units": "cells/mm^3",
                    "uncertainty_value": 400.0,
                    "uncertainty_type": "sd",
                    "sample_size": 45,
                    "snippet": {
                        "text": "Median intratumoral density was 1500 cells/mm^3 (SD=400, n=45).",
                        "page_or_section": "Results"
                    },
                    "source": {
                        "doi": "10.1038/s41586-024-07421-0",
                        "title": "Clinical Tumor Microenvironment Analysis",
                        "authors": "Johnson et al.",
                        "year": 2024
                    }
                }]
            },
            source_relevance={
                "indication_match": "exact",
                "indication_justification": "Direct clinical measurement in target indication",
                "species_source": "human",
                "species_target": "human",
                "evidence_type": "clinical",
                "tme_compatible": True,
                "estimated_translation_uncertainty": 1.5
            }
        )
