import os

# Load a local .env (AWS credentials, Bedrock config, etc.) into the process
# environment at startup so boto3's default credential chain and the Bedrock
# defaults pick them up — no keys ever need to be typed into the app UI.
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
except Exception:
    pass  # dotenv optional; real env vars / ~/.aws still work without it

import traceback
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError
from typing import Dict, Any, List, Optional

import db_interface
from simulation_engine import ODEModel, solve_pde, explore_parameter_space
import agent
import llm_provider

app = FastAPI(title="BioSimulateAI - Biological Modeling & Simulation Copilot")

# Enable CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Safety net: no endpoint may ever crash the client with a bare 500 ----------
# Any exception an endpoint does not catch itself is turned into a structured JSON
# body {"error": "...", "detail": "..."} the frontend can read and display, instead
# of an opaque "Internal Server Error". HTTPException keeps its own status/handler.
@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    # Provider and SDK exception text can contain account, endpoint, or credential
    # diagnostics. Log the exception type and stack locations, but never its message.
    print(f"Unhandled {type(exc).__name__} on {request.url.path}")
    traceback.print_tb(exc.__traceback__)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": "The server could not complete this request.",
        },
    )


# API Pydantic schemas
class ParseRequest(BaseModel):
    text: str
    llm: Optional[Dict[str, Any]] = None
    api_key: Optional[str] = None  # deprecated (Gemini); ignored

class CompileRequest(BaseModel):
    blueprint: Dict[str, Any]

class ExtractEquationsRequest(BaseModel):
    image: str                       # base64 string, optionally a data: URL
    format: Optional[str] = None     # png / jpeg / ... (inferred from a data URL if omitted)
    llm: Optional[Dict[str, Any]] = None

class EquationsRequest(BaseModel):
    equations: str                   # user-written ODE system (dX/dt = ...)

class SimulateRequest(BaseModel):
    blueprint: Dict[str, Any]
    custom_params: Optional[Dict[str, float]] = None
    # PDE runs break symmetry with random noise. The seed is explicit so the field
    # a researcher looks at is the same field /api/evaluate grades; pass a different
    # value to inspect another noise realisation.
    seed: Optional[int] = 0

class OptimizeRequest(BaseModel):
    blueprint: Dict[str, Any]
    target_data: Dict[str, List[float]]
    target_times: List[float]
    params_to_fit: List[str]

class RefineRequest(BaseModel):
    blueprint: Dict[str, Any]
    simulation_results: Dict[str, Any]
    targets: List[Dict[str, Any]]
    llm: Optional[Dict[str, Any]] = None
    api_key: Optional[str] = None  # deprecated (Gemini); ignored

class StringRequest(BaseModel):
    proteins: List[str]
    species: Optional[int] = 9606

class ABMSimulateRequest(BaseModel):
    blueprint: Dict[str, Any]

class ABMBlueprintRequest(BaseModel):
    text: str
    api_key: Optional[str] = None

class MAPLEExtractRequest(BaseModel):
    param_name: str
    param_units: Optional[str] = ""
    param_description: Optional[str] = ""
    mechanistic_context: Optional[str] = ""
    llm: Optional[Dict[str, Any]] = None
    api_key: Optional[str] = None  # deprecated (Gemini); ignored

class MAPLEValidateRequest(BaseModel):
    target_data: Dict[str, Any]
    target_type: str = "submodel"  # "submodel" or "calibration"

class SBMLImportRequest(BaseModel):
    model_id: str

class MultiscaleRequest(BaseModel):
    abm_blueprint: Dict[str, Any]
    ode_blueprint: Optional[Dict[str, Any]] = None
    coupling_rules: Optional[List[Dict[str, Any]]] = None
    num_mcs: Optional[int] = 100
    save_every: Optional[int] = 10

class SensitivityRequest(BaseModel):
    blueprint: Dict[str, Any]
    target_species: str
    param_names: List[str]

class OmniPathRequest(BaseModel):
    proteins: List[str]
    organism: Optional[int] = 9606

class SampleRequest(BaseModel):
    blueprint: Dict[str, Any]
    param_bounds: Dict[str, Dict[str, float]]   # name -> {"min": x, "max": y}
    n_samples: Optional[int] = 64
    method: Optional[str] = "lhs"               # "lhs" | "sobol" | "grid" | "random"
    target_species: Optional[str] = None
    seed: Optional[int] = 0


# --- Existing Endpoints ---

@app.post("/api/blueprint")
def generate_blueprint(req: ParseRequest):
    """Generates structured blueprint from natural language text."""
    try:
        blueprint = agent.parse_biological_text(req.text, req.llm)
        return blueprint
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/equations-to-model")
def equations_to_model(req: EquationsRequest):
    """Build a runnable blueprint directly from a user-written ODE system, with no
    LLM — the model is exactly what the user typed."""
    text = (req.equations or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Write at least one equation, e.g. dX/dt = k - X.")
    return agent.build_blueprint_from_equations(text)


@app.post("/api/extract-equations")
def extract_equations(req: ExtractEquationsRequest):
    """Read hand-written / printed equations from an uploaded photo and return a
    compiled, validated blueprint (same schema as /api/blueprint)."""
    import base64, re as _re
    data = (req.image or "").strip()
    fmt = req.format
    m = _re.match(r"^data:image/([a-zA-Z0-9.+-]+);base64,(.*)$", data, _re.DOTALL)
    if m:
        fmt = fmt or m.group(1)
        data = m.group(2)
    try:
        image_bytes = base64.b64decode(data, validate=False)
    except Exception:
        raise HTTPException(status_code=400, detail="The uploaded image could not be decoded.")
    if not image_bytes:
        raise HTTPException(status_code=400, detail="The uploaded image was empty.")
    if len(image_bytes) > 12 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image is too large (max 12 MB). Please use a smaller photo.")
    try:
        return agent.parse_image_to_blueprint(image_bytes, fmt or "png", req.llm)
    except ValueError as e:
        # Known, user-actionable problems (no engine, bad image, unreadable) -> 400
        raise HTTPException(status_code=400, detail=str(e))
    except llm_provider.LLMError:
        # Provider response bodies can contain account or credential diagnostics.
        raise HTTPException(
            status_code=502,
            detail=("The AI model could not read this image. Check the configured "
                    "Bedrock credential and model access, then try again."),
        )
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"Equation extraction failed ({type(e).__name__}). Please try another image.",
        )


@app.post("/api/compile")
def compile_blueprint(req: CompileRequest):
    """Compile a blueprint into display equations after structural validation."""
    try:
        validation_errors = req.blueprint.get("validation_errors")
        if validation_errors:
            raise HTTPException(
                status_code=400,
                detail="This model has unresolved validation errors and cannot be compiled.",
            )

        bp_type = str(req.blueprint.get("type", "ODE")).upper()
        if bp_type == "PDE":
            status, message = agent.validate_blueprint(req.blueprint)
            if status == "broken":
                raise HTTPException(status_code=400, detail=message)

            reactions = req.blueprint.get("spatial", {}).get("reactions", {})
            diffusions = req.blueprint.get("spatial", {}).get("diffusion", {})
            latex_eqs = {}
            for name, formula in reactions.items():
                D_coeff = diffusions.get(name, 0.1)
                latex_eqs[name] = f"\\frac{{\\partial {name}}}{{\\partial t}} = {D_coeff} \\nabla^2 {name} + {formula}"
            return {"equations": latex_eqs, "equations_verbose": latex_eqs}

        model = ODEModel(req.blueprint)
        latex_eqs = model.get_equations_latex()
        latex_eqs_verbose = model.get_equations_latex(verbose=True)
        return {
            "equations": latex_eqs,
            "equations_verbose": latex_eqs_verbose,
            "parameters": model.params_dict,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/simulate")
def simulate_blueprint(req: SimulateRequest):
    """Runs ODE or PDE simulation on compiled model."""
    try:
        bp = req.blueprint
        # A blueprint that already failed validation cannot be integrated; say so
        # directly instead of letting a SymPy SyntaxError surface as a 500.
        verrs = bp.get("validation_errors")
        if verrs:
            raise HTTPException(status_code=400,
                                detail="This model has unresolved problems: " + " ".join(map(str, verrs)))
        bp_type = bp.get("type", "ODE")
        
        if bp_type == "PDE":
            # The same solve the target evaluator uses (agent.simulate_pde_blueprint):
            # identical initial conditions and seed, so the field shown here is the
            # field /api/evaluate grades. Two separate solve paths would let the
            # displayed pattern and the graded numbers drift apart.
            result = agent.simulate_pde_blueprint(
                bp, seed=(0 if req.seed is None else int(req.seed)), save_every=5
            )
            return result
        else:
            model = ODEModel(bp)
            config = bp.get("simulation_config", {})
            t_max = config.get("t_max", 50.0)

            # Resolution scales with the time horizon so fast dynamics (oscillations,
            # sharp fold-change transients) are captured, not aliased by a coarse 100-point grid.
            num_points = int(min(5000, max(300, t_max * 15)))
            result = model.simulate(t_max, num_points=num_points, custom_params=req.custom_params)
            return result
    except HTTPException:
        raise                       # keep the precise 400 above, don't mask it as a 500
    except agent.SpatialTargetError as e:
        # An unusable spatial blueprint (no reactions, non-numeric horizon) is the
        # caller's input problem, not a server fault.
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/optimize")
def optimize_parameters(req: OptimizeRequest):
    """Fits model parameters to target data curves."""
    try:
        model = ODEModel(req.blueprint)
        fitted, loss = model.fit_parameters_to_target(
            target_data=req.target_data,
            target_times=req.target_times,
            params_to_fit=req.params_to_fit
        )
        return {
            "fitted_parameters": fitted,
            "loss": loss
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/refine")
def refine_model_endpoint(req: RefineRequest):
    """Closed loop feedback to evaluate targets and refine blueprint."""
    try:
        refined_bp, logs, success = agent.refine_model(
            blueprint=req.blueprint,
            simulation_results=req.simulation_results,
            targets=req.targets,
            llm=req.llm
        )
        return {
            "blueprint": refined_bp,
            "logs": logs,
            "success": success
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- Database integrations API ---

@app.get("/api/reactome/search")
def search_reactome(q: str):
    """Searches Reactome database for pathway IDs."""
    return db_interface.search_reactome_pathways(q)

@app.get("/api/reactome/reactions")
def get_reactome_reactions(pathway_id: str):
    """Fetches reactions contained in Reactome pathway."""
    return db_interface.get_reactome_pathway_reactions(pathway_id)

@app.post("/api/string/network")
def get_string_network(req: StringRequest):
    """Fetches interactions from STRING DB."""
    return db_interface.get_string_network(req.proteins, req.species)


# --- NEW: OmniPath API ---

@app.post("/api/omnipath/interactions")
def get_omnipath_interactions(req: OmniPathRequest):
    """Fetches protein interactions from OmniPath with directionality."""
    return db_interface.search_omnipath_interactions(req.proteins, req.organism)


# --- NEW: SIGNOR API ---

@app.get("/api/signor/search")
def search_signor(q: str):
    """Searches SIGNOR for curated signaling pathway relationships."""
    return db_interface.search_signor_pathway(q)


# --- NEW: BioModels API ---

@app.get("/api/biomodels/search")
def search_biomodels(q: str):
    """Searches BioModels repository for curated SBML models."""
    return db_interface.search_biomodels(q)

@app.post("/api/biomodels/import")
def import_biomodel(req: SBMLImportRequest):
    """Imports an SBML model from BioModels and converts to blueprint."""
    try:
        from maple_extractor import MAPLEExtractor
        extractor = MAPLEExtractor()
        blueprint, logs = extractor.import_biomodels_sbml(req.model_id)
        if blueprint:
            return {"blueprint": blueprint, "logs": logs}
        else:
            raise HTTPException(status_code=404, detail=f"Model {req.model_id} not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- NEW: ABM Endpoints ---

@app.get("/api/abm/presets")
def list_abm_presets():
    """List available ABM preset models."""
    from abm_blueprints import list_abm_presets as _list
    return _list()

@app.get("/api/abm/preset/{name}")
def get_abm_preset(name: str):
    """Get a specific ABM preset blueprint."""
    from abm_blueprints import get_abm_preset as _get
    try:
        return _get(name)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.post("/api/abm/simulate")
def simulate_abm(req: ABMSimulateRequest):
    """Run a Cellular Potts Model ABM simulation."""
    try:
        from abm_engine import build_cpm_from_blueprint
        cpm = build_cpm_from_blueprint(req.blueprint)
        config = req.blueprint.get("simulation_config", {})
        result = cpm.simulate(
            num_mcs=config.get("num_mcs", 500),
            save_every=config.get("save_every", 10)
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- NEW: MAPLE Calibration Endpoints ---

@app.post("/api/maple/extract")
def maple_extract(req: MAPLEExtractRequest):
    """Run MAPLE parameter extraction using LLM with structured validation."""
    try:
        from maple_extractor import MAPLEExtractor
        from maple_schemas import submodel_target_to_dict, validation_report_to_dict
        
        extractor = MAPLEExtractor(llm=req.llm)
        target, report, logs = extractor.extract_submodel_target(
            param_name=req.param_name,
            param_units=req.param_units or "",
            param_description=req.param_description or "",
            mechanistic_context=req.mechanistic_context or ""
        )
        return {
            "target": submodel_target_to_dict(target) if target else None,
            "validation": validation_report_to_dict(report) if report else None,
            "logs": logs
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/maple/validate")
def maple_validate(req: MAPLEValidateRequest):
    """Validate a calibration target YAML/JSON against MAPLE schemas."""
    try:
        from maple_schemas import (
            SubmodelTarget, CalibrationTarget,
            validate_submodel_target, validate_calibration_target,
            validation_report_to_dict
        )
        
        if req.target_type == "submodel":
            target = SubmodelTarget(**req.target_data)
            report = validate_submodel_target(target)
        else:
            target = CalibrationTarget(**req.target_data)
            report = validate_calibration_target(target)

        return validation_report_to_dict(report)
    except ValidationError as e:
        # A malformed target is the caller's input, not a server fault. Report the
        # field errors so the UI can point at them; 500 hid this as an outage.
        raise HTTPException(status_code=422, detail={
            "message": "The calibration target does not match the MAPLE schema.",
            "errors": [
                {"field": ".".join(str(p) for p in err.get("loc", ())), "problem": err.get("msg", "")}
                for err in e.errors()
            ][:20],
        })
    except (TypeError, KeyError) as e:
        raise HTTPException(
            status_code=422,
            detail=f"The calibration target is not a valid MAPLE object ({type(e).__name__}).",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- NEW: Multi-scale Simulation ---

@app.post("/api/multiscale/simulate")
def simulate_multiscale(req: MultiscaleRequest):
    """Run coupled ODE-ABM-PDE multi-scale simulation."""
    try:
        from multiscale import MultiscaleSimulator
        sim = MultiscaleSimulator()
        sim.configure(
            abm_blueprint=req.abm_blueprint,
            ode_blueprint=req.ode_blueprint,
            coupling_rules=req.coupling_rules
        )
        result = sim.simulate(
            num_mcs=req.num_mcs or 100,
            save_every=req.save_every or 10
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- NEW: Multi-condition target evaluation (oscillation/bistability/fold-change) ---

class EvaluateRequest(BaseModel):
    blueprint: Dict[str, Any]
    targets: List[Dict[str, Any]]
    custom_params: Optional[Dict[str, float]] = None

@app.get("/api/llm/env")
def llm_env():
    """Report whether the server environment (.env / env vars / AWS profile) is
    pre-configured for Bedrock, so the UI can auto-select it with no key entry.
    Never returns any credential value — only booleans and non-secret config."""
    return {
        "bedrock_env_ready": llm_provider.bedrock_env_ready(),
        "model": llm_provider.BEDROCK_DEFAULT_MODEL,
        "region": llm_provider.BEDROCK_DEFAULT_REGION,
        "engine_default": (os.getenv("LLM_ENGINE", "").strip().lower() or None),
    }


@app.post("/api/evaluate")
def evaluate_targets(req: EvaluateRequest):
    """Evaluate targets against the model's ACTUAL solution.

    A PDE blueprint is graded on its spatial field (each target's reduction, e.g. the
    final-frame spatial mean, is named in the result), a target type with no defensible
    field-level meaning is REFUSED rather than graded on a proxy, and any target whose
    acceptance window cannot fail is reported in `warnings` instead of quietly passing.
    Refused targets never count towards `met_count`.
    """
    try:
        t_max = float(req.blueprint.get("simulation_config", {}).get("t_max", 50.0))
        met_count, results = agent.evaluate_targets_on_blueprint(
            req.blueprint, req.targets, t_max=t_max, custom_params=req.custom_params
        )
        is_pde = str(req.blueprint.get("type", "ODE")).upper() == "PDE"
        return {
            "met_count": met_count,
            "total": len(req.targets),
            "results": results,
            # Unfalsifiable-target warnings are surfaced, never applied: a tolerance
            # is the researcher's choice, so it is reported rather than rewritten.
            "warnings": [r["warning"] for r in results if r.get("warning")],
            "refused_count": sum(1 for r in results if r.get("refused")),
            "graded_on": ("the spatial PDE field" if is_pde else "the ODE trajectory"),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- NEW: Open-source LLM management (local GGUF models) ---

class LLMDownloadRequest(BaseModel):
    model: str

@app.get("/api/llm/models")
def llm_models():
    """List available open-source models and their download status."""
    return llm_provider.list_models()

@app.post("/api/llm/download")
def llm_download(req: LLMDownloadRequest):
    """Start downloading a model's weights from HuggingFace (runs in background)."""
    try:
        return llm_provider.start_download(req.model)
    except llm_provider.LLMError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/llm/status")
def llm_status(model: str):
    """Poll download status for a model."""
    return llm_provider.download_status(model)


# --- NEW: Parameter-Space Exploration (Latin Hypercube Sampling) ---

@app.post("/api/sample")
def sample_parameter_space(req: SampleRequest):
    """Sample the parameter space (Latin Hypercube by default) and summarise outputs."""
    try:
        result = explore_parameter_space(
            blueprint=req.blueprint,
            param_bounds=req.param_bounds,
            n_samples=req.n_samples or 64,
            method=req.method or "lhs",
            target_species=req.target_species,
            seed=req.seed or 0,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- NEW: Sensitivity Analysis ---

@app.post("/api/sensitivity")
def run_sensitivity(req: SensitivityRequest):
    """Run local sensitivity analysis for parameter importance ranking."""
    try:
        from multiscale import local_sensitivity_analysis
        result = local_sensitivity_analysis(
            blueprint=req.blueprint,
            target_species=req.target_species,
            param_names=req.param_names
        )
        return {"sensitivities": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Geometry / topology / mesh / model / conditions / approaches / runs
#
# The preparation workflow (domain -> topology -> mesh -> model -> conditions) is
# shared, then ONE approach is selected, compiled and run. Every endpoint here is
# stateless except the run registry: the project document is passed in and returned,
# so the frontend owns project state and a reload cannot leave a half-applied stage.
# =============================================================================

import approach_base
import boundary_conditions as bc_module
import geometry as geometry_module
import meshing as meshing_module
import pde_model as pde_model_module
import project_schema as project_module
import topology as topology_module
from pde_solver_1d import solve_1d
from run_manager import RUNS, RunStateError


class ProjectRequest(BaseModel):
    project: Dict[str, Any]
    approach: Optional[str] = None


class ProjectTextRequest(BaseModel):
    text: str


class DomainRequest(BaseModel):
    domain: Dict[str, Any]


class TopologyRequest(BaseModel):
    topology: Dict[str, Any]
    domain: Optional[Dict[str, Any]] = None


class MeshRequest(BaseModel):
    domain: Dict[str, Any]
    settings: Optional[Dict[str, Any]] = None


class ModelRequest(BaseModel):
    model: Dict[str, Any]
    domain: Optional[Dict[str, Any]] = None


class ConditionsRequest(BaseModel):
    conditions: List[Dict[str, Any]]
    domain: Dict[str, Any]
    fields: List[str]


class Solve1DRequest(BaseModel):
    domain: Dict[str, Any]
    model: Dict[str, Any]
    field: Optional[str] = None
    conditions: Optional[List[Dict[str, Any]]] = None
    mesh_settings: Optional[Dict[str, Any]] = None
    dt: Optional[float] = None


class CreateRunRequest(BaseModel):
    project: Dict[str, Any]
    approach: Optional[str] = None
    seed: Optional[int] = None
    start: bool = True


def _issue_payload(issues) -> Dict[str, Any]:
    errors = [i for i in issues if i.severity == "error"]
    return {
        "valid": not errors,
        "error_count": len(errors),
        "warning_count": len([i for i in issues if i.severity == "warning"]),
        "issues": geometry_module.issues_to_dicts(issues),
    }


# --- approaches -------------------------------------------------------------
@app.get("/api/approaches")
def list_approaches():
    """Capabilities for all three approaches, including unavailable ones.

    An unavailable approach is listed WITH its reason rather than hidden, so the UI
    can disable it honestly instead of pretending it does not exist.
    """
    return {"approaches": approach_base.list_approaches(),
            "load_errors": approach_base.load_errors()}


@app.post("/api/approaches/{approach_id}/export")
def export_approach_configuration(approach_id: str, req: ProjectRequest):
    try:
        adapter = approach_base.get_approach(approach_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    try:
        return adapter.export_configuration(req.project)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"{type(e).__name__}: {e}")


# --- project ----------------------------------------------------------------
@app.post("/api/project/new")
def create_project(req: DomainRequest):
    issues = geometry_module.validate_domain(req.domain)
    if not geometry_module.is_valid(issues):
        raise HTTPException(status_code=400, detail=_issue_payload(issues))
    return project_module.new_project(domain=req.domain)


@app.post("/api/project/validate")
def validate_project(req: ProjectRequest):
    try:
        return project_module.validate_project(req.project, req.approach)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"{type(e).__name__}: {e}")


@app.post("/api/project/import")
def import_project(req: ProjectTextRequest):
    """Import a project document, migrating an older schema forward."""
    try:
        project = project_module.from_json(req.text)
    except project_module.ProjectVersionError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"project": project,
            "migrations_applied": project.get("migrations_applied", []),
            "validation": project_module.validate_project(project)}


@app.post("/api/project/export")
def export_project(req: ProjectRequest):
    return {"files": project_module.export_project_bundle(req.project)}


@app.post("/api/project/export/cc3d")
def export_project_cc3d(req: ProjectRequest):
    """Export a runnable CompuCell3D project. Available whether or not CC3D exists."""
    try:
        return {"files": project_module.export_cc3d_package(req.project)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"{type(e).__name__}: {e}")


# --- stage 1: geometry ------------------------------------------------------
@app.post("/api/geometry/validate")
def validate_geometry(req: DomainRequest):
    issues = geometry_module.validate_domain(req.domain)
    payload = _issue_payload(issues)
    if payload["valid"]:
        payload.update({
            "dimension": geometry_module.domain_dimension(req.domain),
            "bounds": list(geometry_module.domain_bounds(req.domain)),
            "measure": geometry_module.domain_measure(req.domain),
            "summary": geometry_module.describe_domain(req.domain),
        })
    return payload


# --- stage 2: topology -----------------------------------------------------
@app.post("/api/topology/validate")
def validate_topology(req: TopologyRequest):
    issues = topology_module.validate_topology(req.topology, req.domain)
    payload = _issue_payload(issues)
    payload["stats"] = topology_module.topology_stats(req.topology)
    return payload


@app.post("/api/topology/cycles")
def find_topology_cycles(req: TopologyRequest):
    """Candidate loops the user can convert into cells. Bounded by design."""
    cycles = topology_module.find_cycles(req.topology)
    return {"cycles": cycles, "count": len(cycles),
            "max_length": topology_module.MAX_CYCLE_LENGTH,
            "capped": len(cycles) >= topology_module.MAX_CYCLES}


@app.post("/api/topology/export")
def export_topology(req: TopologyRequest):
    return {"files": {
        "nodes.csv": topology_module.nodes_to_csv(req.topology),
        "edges.csv": topology_module.edges_to_csv(req.topology),
        "cells.csv": topology_module.cells_to_csv(req.topology),
    }}


# --- stage 3: mesh ---------------------------------------------------------
@app.post("/api/mesh/generate")
def generate_mesh(req: MeshRequest):
    issues = geometry_module.validate_domain(req.domain)
    if not geometry_module.is_valid(issues):
        raise HTTPException(status_code=400, detail=_issue_payload(issues))
    try:
        mesh = meshing_module.generate_mesh(req.domain, req.settings or {})
    except (ValueError, RuntimeError) as e:
        # Actionable: a spacing that is too fine, or unstructured meshing without SciPy.
        raise HTTPException(status_code=400, detail=str(e))
    validation = meshing_module.validate_mesh(mesh, req.domain)
    return {"mesh": mesh, "validation": _issue_payload(validation)}


# --- stage 4: model --------------------------------------------------------
@app.get("/api/model/presets")
def list_model_presets():
    return {"presets": [{"name": name, **{k: v for k, v in
                                          pde_model_module.get_preset(name).items()
                                          if k in ("label", "description")}}
                        for name in pde_model_module.preset_names()]}


@app.get("/api/model/preset/{name}")
def get_model_preset(name: str):
    try:
        return pde_model_module.get_preset(name)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/api/model/validate")
def validate_model(req: ModelRequest):
    issues = pde_model_module.validate_model(req.model, req.domain)
    payload = _issue_payload(issues)
    summaries = []
    for field in (req.model.get("fields") or []):
        if not isinstance(field, dict) or not field.get("name"):
            continue
        try:
            summaries.append({
                "field": field["name"],
                "latex": pde_model_module.equation_latex(field, req.model.get("parameters")),
                "summary": pde_model_module.plain_language_summary(
                    field, req.model.get("parameters")),
            })
        except Exception:
            # A summary is a convenience; a bad expression is already reported above.
            continue
    payload["fields"] = summaries
    return payload


# --- stage 5: conditions ---------------------------------------------------
@app.post("/api/conditions/validate")
def validate_conditions(req: ConditionsRequest):
    issues = bc_module.validate_conditions(req.conditions, req.domain, req.fields)
    payload = _issue_payload(issues)
    payload["description"] = bc_module.describe_conditions(req.conditions, req.domain,
                                                           req.fields)
    return payload


# --- 1D PDE execution ------------------------------------------------------
@app.post("/api/pde/solve1d")
def solve_pde_1d(req: Solve1DRequest):
    """Solve one field on a 1D interval with the assigned boundary conditions."""
    domain_issues = geometry_module.validate_domain(req.domain)
    if not geometry_module.is_valid(domain_issues):
        raise HTTPException(status_code=400, detail=_issue_payload(domain_issues))
    if geometry_module.domain_dimension(req.domain) != 1:
        raise HTTPException(status_code=400,
                            detail="This endpoint solves a 1D interval domain.")

    try:
        mesh = meshing_module.mesh_1d(
            req.domain,
            element_count=(req.mesh_settings or {}).get("element_count"),
            target_spacing=(req.mesh_settings or {}).get("target_spacing"),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    field_name = req.field or (req.model.get("fields") or [{}])[0].get("name")
    if not field_name:
        raise HTTPException(status_code=400, detail="The model defines no fields to solve.")
    try:
        compiled = pde_model_module.compile_field(req.model, field_name, req.domain)
    except (ValueError, KeyError, pde_model_module.ExpressionError) as e:
        raise HTTPException(status_code=400, detail=str(e))

    field_names = [str(f) for f in (req.fields if hasattr(req, "fields") else [])] or \
        [str(f.get("name")) for f in (req.model.get("fields") or []) if isinstance(f, dict)]
    condition_issues = bc_module.validate_conditions(req.conditions or [], req.domain,
                                                     field_names)
    if not geometry_module.is_valid(condition_issues):
        raise HTTPException(status_code=400, detail=_issue_payload(condition_issues))

    import numpy as _np
    x = _np.array([node[0] for node in mesh["nodes"]], dtype=float)
    zeros = _np.zeros_like(x)
    initial = _np.asarray(compiled["initial"](x, zeros), dtype=float)
    resolved = bc_module.resolve_1d_conditions(req.conditions or [], field_name)

    dx = float(mesh["spacing"])
    diffusion = float(compiled["diffusion"])
    velocity = float(compiled["advection"]["vx"])
    safe_dt = 0.4 * min(
        (dx * dx / (2.0 * diffusion)) if diffusion > 0 else float("inf"),
        (dx / abs(velocity)) if abs(velocity) > 0 else float("inf"),
    )
    dt = float(req.dt) if req.dt else (safe_dt if _np.isfinite(safe_dt) else 0.01)
    duration = max(1e-9, compiled["t_end"] - compiled["t_start"])
    save_every = max(1, int(round(compiled["output_interval"] / dt)))

    def reaction(u, xs, t):
        return compiled["reaction"](u, xs, _np.zeros_like(xs), t)

    def source(xs, t):
        return compiled["source"](xs, _np.zeros_like(xs), t)

    try:
        result = solve_1d(
            x, initial, diffusion=diffusion, t_max=duration, dt=dt,
            reaction=reaction, source=source, advection=velocity,
            left=resolved["left"], right=resolved["right"],
            periodic=resolved["periodic"], save_every=save_every,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    result.update({
        "field": field_name,
        "units": compiled["units"],
        "latex": compiled["latex"],
        "summary": compiled["summary"],
        "mesh": {"node_count": len(x), "spacing": dx},
    })
    return result


# --- stage 6-10: runs ------------------------------------------------------
@app.post("/api/runs")
def create_run(req: CreateRunRequest):
    """Validate, compile and (by default) start a run of the selected approach."""
    try:
        record = RUNS.create_run(req.project, req.approach, seed=req.seed)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if req.start and record.state == "ready":
        try:
            RUNS.start(record.run_id)
        except RunStateError as e:
            raise HTTPException(status_code=409, detail=str(e))
    # A run that failed validation or compilation is returned with that state and a
    # 200: the request itself succeeded, and the state is the answer.
    return RUNS.status(record.run_id)


@app.get("/api/runs")
def list_runs(limit: int = 50):
    return {"runs": RUNS.list_runs(limit=limit)}


@app.get("/api/runs/{run_id}")
def get_run(run_id: str):
    try:
        return RUNS.status(run_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/api/runs/{run_id}/results")
def get_run_results(run_id: str):
    try:
        return {"run_id": run_id, "results": RUNS.results(run_id)}
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RunStateError as e:
        # 409: the run exists but is not in a state that has results.
        raise HTTPException(status_code=409, detail=str(e))


@app.get("/api/runs/{run_id}/logs")
def get_run_logs(run_id: str, limit: int = 500):
    try:
        return {"run_id": run_id, "logs": RUNS.logs(run_id, limit=limit)}
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/api/runs/{run_id}/start")
def start_run(run_id: str):
    try:
        RUNS.start(run_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RunStateError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return RUNS.status(run_id)


@app.post("/api/runs/{run_id}/pause")
def pause_run(run_id: str):
    try:
        RUNS.pause(run_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RunStateError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return RUNS.status(run_id)


@app.post("/api/runs/{run_id}/resume")
def resume_run(run_id: str):
    try:
        RUNS.resume(run_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RunStateError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return RUNS.status(run_id)


@app.post("/api/runs/{run_id}/cancel")
def cancel_run(run_id: str):
    try:
        RUNS.cancel(run_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RunStateError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return RUNS.status(run_id)


# --- stage 11-12: result exports -------------------------------------------
@app.get("/api/runs/{run_id}/export/cells")
def export_run_cells(run_id: str, frame: Optional[int] = None):
    try:
        results = RUNS.results(run_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RunStateError as e:
        raise HTTPException(status_code=409, detail=str(e))
    try:
        return {"filename": f"{run_id}_cells.csv",
                "csv": project_module.export_cells_csv(results or {}, frame)}
    except IndexError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/runs/{run_id}/export/timeseries")
def export_run_timeseries(run_id: str):
    try:
        results = RUNS.results(run_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RunStateError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return {"filename": f"{run_id}_timeseries.csv",
            "csv": project_module.export_timeseries_csv(results or {})}


@app.get("/api/runs/{run_id}/export/metadata")
def export_run_metadata_endpoint(run_id: str):
    try:
        record = RUNS.get(run_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return project_module.export_run_metadata(record)


# Serve Static files - must be loaded after api routes
os.makedirs("static", exist_ok=True)

class NoCacheStaticFiles(StaticFiles):
    """Serve static assets with no-cache headers so edits to app.js/index.html/style.css
    are picked up on the next reload instead of being served from the browser cache."""
    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response

app.mount("/", NoCacheStaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
