import os

# Load a local .env (AWS credentials, Bedrock config, etc.) into the process
# environment at startup so boto3's default credential chain and the Bedrock
# defaults pick them up — no keys ever need to be typed into the app UI.
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
except Exception:
    pass  # dotenv optional; real env vars / ~/.aws still work without it

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
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

# API Pydantic schemas
class ParseRequest(BaseModel):
    text: str
    llm: Optional[Dict[str, Any]] = None
    api_key: Optional[str] = None  # deprecated (Gemini); ignored

class CompileRequest(BaseModel):
    blueprint: Dict[str, Any]

class SimulateRequest(BaseModel):
    blueprint: Dict[str, Any]
    custom_params: Optional[Dict[str, float]] = None

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

@app.post("/api/compile")
def compile_blueprint(req: CompileRequest):
    """Compiles blueprint into ODE system, returning LaTeX equations."""
    try:
        bp_type = req.blueprint.get("type", "ODE")
        if bp_type == "PDE":
            # For PDEs, we just return equations based on reaction expressions
            reactions = req.blueprint.get("spatial", {}).get("reactions", {})
            diffusions = req.blueprint.get("spatial", {}).get("diffusion", {})
            latex_eqs = {}
            for name, formula in reactions.items():
                D_coeff = diffusions.get(name, 0.1)
                latex_eqs[name] = f"\\frac{{\\partial {name}}}{{\\partial t}} = {D_coeff} \\nabla^2 {name} + {formula}"
            return {"equations": latex_eqs}
        else:
            # ODE compile
            model = ODEModel(req.blueprint)
            latex_eqs = model.get_equations_latex()
            latex_eqs_verbose = model.get_equations_latex(verbose=True)
            # Also return parameter names and defaults
            return {
                "equations": latex_eqs,
                "equations_verbose": latex_eqs_verbose,
                "parameters": model.params_dict
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/simulate")
def simulate_blueprint(req: SimulateRequest):
    """Runs ODE or PDE simulation on compiled model."""
    try:
        bp = req.blueprint
        bp_type = bp.get("type", "ODE")
        
        if bp_type == "PDE":
            spatial = bp.get("spatial", {})
            reactions = spatial.get("reactions", {})
            
            # Initial conditions
            initial_conditions = {}
            for node in bp.get("nodes", []):
                nid = node["id"]
                initial_conditions[nid] = {
                    "type": "random_noise",
                    "base_value": node.get("initial_value", 1.0),
                    "noise_amplitude": 0.05
                }
                
            config = bp.get("simulation_config", {})
            t_max = config.get("t_max", 100.0)
            dt = config.get("dt", 0.1)
            
            result = solve_pde(
                spatial_config=spatial,
                reaction_formulas=reactions,
                initial_conditions=initial_conditions,
                t_max=t_max,
                dt=dt,
                save_every=5
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
    """Evaluate targets on a blueprint, running the extra sims that bistability/fold-change need."""
    try:
        t_max = float(req.blueprint.get("simulation_config", {}).get("t_max", 50.0))
        met_count, results = agent.evaluate_targets_on_blueprint(
            req.blueprint, req.targets, t_max=t_max, custom_params=req.custom_params
        )
        return {"met_count": met_count, "total": len(req.targets), "results": results}
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
