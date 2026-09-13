import os

# Load a local .env (AWS credentials, Bedrock config, etc.) into the process
# environment at startup so boto3's default credential chain and the Bedrock
# defaults pick them up — no keys ever need to be typed into the app UI.
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
except Exception:
    pass  # dotenv optional; real env vars / ~/.aws still work without it

import contextlib
import math
import time
import traceback
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError
from typing import Dict, Any, List, NoReturn, Optional

import sympy
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
    # num_mcs has NO default: it is the size of the run, and defaulting it to 100
    # is what let a 16-byte body buy a 100-step simulation over a 100x100 lattice.
    # save_every only sets the reporting cadence, so it keeps its default.
    num_mcs: Optional[int] = None
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


# =============================================================================
# Request validation and execution budgets
#
# Every simulation endpoint below hands caller-supplied numbers to an unbounded
# numerical kernel. Four failure modes were measured on the live routes and are
# closed here, at the edge, before any solver is entered:
#
#   * NO EXECUTION BUDGET. One /api/simulate request with odes {"X": "X**X**X"}
#     and t_max=1e9 returned no bytes in 120 s and held the endpoint. A horizon
#     bound alone does not fix that -- see _ode_execution_budget.
#   * A FULL DEFAULT RUN FOR AN EMPTY BODY. POST {"blueprint": {}} to
#     /api/abm/simulate answered 200 with 5,621,290 bytes: a 100x100 lattice over
#     500 Monte Carlo steps, from a 16-byte unauthenticated request.
#   * NONSENSE NUMERICS REPORTED AS SUCCESS. t_max=-10 integrated backwards in
#     time, dt=-0.1 integrated nothing, num_mcs=-5 returned a single frame -- all
#     200, all indistinguishable downstream from a real run.
#   * BARE PYTHON REPRS AS 500s. A node with no id surfaced as
#     500 {"detail": "'id'"}, odes {"X": "1/0"} as 500 {"detail":
#     "'ComplexInfinity'"} -- nothing a caller can act on, for ordinary bad input.
#
# The rule applied throughout: a request whose numbers cannot describe a run is
# REFUSED and told which field and which value was wrong. Truncating, reversing
# or defaulting the run and reporting it as complete is the worse outcome,
# because nothing downstream can tell such a result from a real one.
# =============================================================================

def _env_positive_float(name: str, default: float) -> float:
    try:
        value = float(os.getenv(name, "") or default)
    except (TypeError, ValueError):
        return default
    return value if value > 0 and math.isfinite(value) else default


# Wall-clock ceiling for the integration inside ONE request.
SIMULATE_BUDGET_SECONDS = _env_positive_float("BIOSIM_SIMULATE_BUDGET_SECONDS", 20.0)
# Largest integration horizon accepted. Far past anything this app's models need;
# the point is that the horizon is BOUNDED, not that 1e6 is biologically special.
MAX_T_MAX = _env_positive_float("BIOSIM_MAX_T_MAX", 1.0e6)
# Explicit-scheme step ceilings. Both PDE solvers materialise one Python list
# entry per step (`step_sizes = [dt] * n_full`), so t_max/dt is a MEMORY bound as
# well as a time bound: t_end=1e9 at the stability-limited dt raised MemoryError
# inside /api/pde/solve1d and came back as an opaque 500.
MAX_PDE_STEPS = 2_000_000
MAX_MONTE_CARLO_STEPS = 100_000
MAX_SAMPLES = 2000
# The strategies simulation_engine._unit_samples actually implements. Anything
# else used to fall through to plain Monte-Carlo while the response echoed the
# name the caller sent, so "telepathy" was reported as the method used.
SAMPLING_METHODS = ("lhs", "latin", "latin_hypercube", "sobol", "quasi",
                    "grid", "factorial", "random")


def _reject(detail: Any) -> NoReturn:
    """422: the request was understood and is unusable. Never a 500."""
    raise HTTPException(status_code=422, detail=detail)


def _finite_number(field: str, value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        _reject(f"{field} must be a number; got {value!r}.")
    if not math.isfinite(number):
        _reject(f"{field} must be a finite number; got {value!r}.")
    return number


def _positive_number(field: str, value: Any, reason: str = "") -> float:
    number = _finite_number(field, value)
    if number <= 0:
        _reject(f"{field} must be greater than zero; got {number:g}."
                + (f" {reason}" if reason else ""))
    return number


def _positive_int(field: str, value: Any, reason: str = "") -> int:
    number = _finite_number(field, value)
    if number != int(number):
        _reject(f"{field} must be a whole number; got {value!r}.")
    if int(number) <= 0:
        _reject(f"{field} must be at least 1; got {int(number)}."
                + (f" {reason}" if reason else ""))
    return int(number)


class ExecutionBudgetExceeded(Exception):
    """One request's integration ran past its wall-clock budget."""

    def __init__(self, budget_seconds: float, t_max: float):
        self.budget_seconds = budget_seconds
        self.t_max = t_max
        super().__init__(f"execution budget of {budget_seconds:g}s exceeded")


@contextlib.contextmanager
def _ode_execution_budget(model, t_max: float, budget_seconds: Optional[float] = None):
    """Enforce a wall-clock deadline INSIDE the integration, not around it.

    A horizon bound is not an execution bound. LSODA takes as many internal steps
    as the right-hand side demands, so a stiff or explosive system spends unbounded
    time on a horizon that looks modest -- measured with odes {"X": "X**X**X"},
    which produced no bytes in 120 s. Checking the clock before or after the call
    cannot help, because there is nothing to check until the call returns.

    ODEModel integrates exclusively through `_f_lambdified`, so wrapping that one
    callable puts the deadline on the solver's own inner loop, which is the only
    place that can interrupt it. The exception propagates out of SciPy's LSODA
    wrapper (verified), and the original callable is restored on the way out, so
    the model object is left exactly as it was found.
    """
    budget = SIMULATE_BUDGET_SECONDS if budget_seconds is None else budget_seconds
    original = model._f_lambdified
    deadline = time.monotonic() + budget

    def guarded(t, y, params):
        if time.monotonic() > deadline:
            raise ExecutionBudgetExceeded(budget, t_max)
        return original(t, y, params)

    model._f_lambdified = guarded
    try:
        yield
    finally:
        model._f_lambdified = original


def _budget_exceeded(exc: ExecutionBudgetExceeded) -> HTTPException:
    """400 naming the limit and the horizon that blew it.

    Deliberately not a truncated 200: a horizon the solver never reached, reported
    as a completed run, cannot be told apart from a real result by anything
    downstream -- including the closed-loop refinement that grades these runs.
    """
    return HTTPException(
        status_code=400,
        detail=(
            f"This model exceeded the server's {exc.budget_seconds:g}-second execution "
            f"budget for a single request while integrating to t_max={exc.t_max:g}. "
            f"The run was ABANDONED, not shortened. Reduce t_max, soften the "
            f"stiffness of the equations, or run this model offline."
        ),
    )


def _validated_horizon(config: Dict[str, Any], default: float,
                       field: str = "simulation_config.t_max") -> float:
    """A t_max that is positive, finite and inside the server's bound."""
    supplied = config.get("t_max", default) if isinstance(config, dict) else default
    t_max = _positive_number(
        field, supplied,
        "A negative horizon integrates backwards in time and a zero horizon "
        "integrates nothing; either one is reported as a successful run.")
    if t_max > MAX_T_MAX:
        raise HTTPException(
            status_code=400,
            detail=(f"{field}={t_max:g} exceeds the server limit of {MAX_T_MAX:g}. "
                    f"The horizon is bounded so one request cannot occupy the "
                    f"endpoint indefinitely; the limit is not silently applied, "
                    f"because a shortened run reported as complete is worse than "
                    f"this refusal."),
        )
    return t_max


def _validated_ode_model(blueprint: Dict[str, Any], where: str = "blueprint"):
    """Build an ODEModel, turning every malformed-input failure into a 422.

    ODEModel indexes `node["id"]` directly and lambdifies whatever SymPy produced,
    so a node with no id surfaced as 500 {"detail": "'id'"} and odes {"X": "1/0"}
    as 500 {"detail": "'ComplexInfinity'"}. Both are ordinary bad input and both
    are named here, with the offending index or species.
    """
    nodes = blueprint.get("nodes") if isinstance(blueprint, dict) else None
    if not isinstance(nodes, list) or not nodes:
        _reject(f"{where}.nodes must be a non-empty list of species objects, each "
                f"with an 'id'.")
    seen: Dict[str, int] = {}
    for index, node in enumerate(nodes):
        if not isinstance(node, dict):
            _reject(f"{where}.nodes[{index}] must be an object with an 'id'; got "
                    f"{type(node).__name__}.")
        node_id = node.get("id")
        if node_id is None or str(node_id).strip() == "":
            _reject(f"{where}.nodes[{index}] has no 'id'. Every species needs one, "
                    f"because the id IS the state variable in the equations.")
        if str(node_id) in seen:
            _reject(f"{where}.nodes[{index}] repeats the id {str(node_id)!r}, already "
                    f"used by nodes[{seen[str(node_id)]}].")
        seen[str(node_id)] = index

    odes = blueprint.get("odes") or {}
    if odes and not isinstance(odes, dict):
        _reject(f"{where}.odes must be an object mapping a species id to its rate "
                f"expression; got {type(odes).__name__}.")
    for species, expression in (odes.items() if isinstance(odes, dict) else ()):
        try:
            parsed = sympy.sympify(str(expression))
        except (sympy.SympifyError, SyntaxError, TypeError, AttributeError) as exc:
            _reject(f"{where}.odes[{species!r}] is not a readable expression "
                    f"({type(exc).__name__}): {expression!r}.")
        if parsed.has(sympy.zoo) or parsed.has(sympy.oo) or parsed.has(sympy.nan):
            _reject(f"{where}.odes[{species!r}] = {expression!r} is not finite: it "
                    f"evaluates to {parsed}, which is a division by zero. Such a "
                    f"rate law cannot be compiled or integrated.")

    try:
        return ODEModel(blueprint)
    except HTTPException:
        raise
    except KeyError as exc:
        missing = exc.args[0] if exc.args else "a required field"
        _reject(f"{where} is missing {missing!r}, which compiling the equations "
                f"requires.")
    except (TypeError, ValueError, AttributeError) as exc:
        _reject(f"{where} could not be compiled into an ODE system "
                f"({type(exc).__name__}): {exc}")


def _blueprint_species(blueprint: Dict[str, Any]) -> List[str]:
    """Every species name this blueprint can be graded on.

    Nodes first (the ODE state), then spatial reaction keys (the PDE fields), so a
    PDE blueprint that declares its fields only under `spatial.reactions` is still
    matched.
    """
    names: List[str] = []
    for node in (blueprint.get("nodes") or []):
        if isinstance(node, dict) and node.get("id") is not None:
            names.append(str(node["id"]))
    spatial = blueprint.get("spatial") or {}
    for key in (spatial.get("reactions") or {}):
        if str(key) not in names:
            names.append(str(key))
    return names


def _validated_pde_config(blueprint: Dict[str, Any]) -> Dict[str, float]:
    """t_max, dt and grid dimensions that can actually describe a spatial run."""
    spatial = blueprint.get("spatial") or {}
    config = blueprint.get("simulation_config") or {}
    t_max = _validated_horizon(config, 100.0)
    dt = _positive_number(
        "simulation_config.dt", config.get("dt", 0.1),
        "A non-positive dt takes no time step at all: the solver returned a single "
        "frame at t=0 and reported the whole horizon as solved.")

    for key, read_as in (("nx", "x_grid"), ("ny", "y_grid")):
        if key in spatial:
            _reject(f"spatial.{key} is not a field the PDE solver reads, so it was "
                    f"silently ignored and the grid fell back to the 50x50 default. "
                    f"Use spatial.{read_as} instead (got spatial.{key}="
                    f"{spatial[key]!r}).")
    for key in ("x_grid", "y_grid"):
        if key in spatial:
            _positive_int(f"spatial.{key}", spatial[key],
                          "A grid needs at least one cell along each axis.")

    steps = math.floor(t_max / dt)
    if steps > MAX_PDE_STEPS:
        raise HTTPException(
            status_code=400,
            detail=(f"t_max={t_max:g} with dt={dt:g} needs {steps:,} explicit time "
                    f"steps, above the server limit of {MAX_PDE_STEPS:,}. Raise dt "
                    f"or lower t_max; the horizon is not shortened silently."),
        )
    return {"t_max": t_max, "dt": dt, "steps": steps}


def _missing_lattice_fields(blueprint: Any, where: str) -> List[str]:
    """The fields without which there is no ABM to run.

    POST {"blueprint": {}} used to answer 200 with a 100x100 lattice over 500
    Monte Carlo steps: a success reported for input nobody supplied, and an
    amplification vector -- a handful of concurrent 16-byte bodies saturate the
    process. Requiring these makes the caller state the size of the run.
    """
    if not isinstance(blueprint, dict):
        return [f"{where} must be an object describing the model, "
                f"not {type(blueprint).__name__}."]
    missing: List[str] = []
    grid = blueprint.get("grid")
    if not isinstance(grid, dict) or grid.get("width") is None or grid.get("height") is None:
        missing.append(f"{where}.grid.width and {where}.grid.height - the lattice size")
    cell_types = blueprint.get("cell_types")
    if not isinstance(cell_types, list) or not cell_types:
        missing.append(f"{where}.cell_types - at least one cell type, each with a type_id")
    else:
        for index, entry in enumerate(cell_types):
            if not isinstance(entry, dict) or entry.get("type_id") is None:
                missing.append(f"{where}.cell_types[{index}].type_id")
    if not blueprint.get("initial_config"):
        missing.append(f"{where}.initial_config - where the cells start; without it "
                       f"the lattice is empty and every step is wasted")
    return missing


def _validated_lattice(blueprint: Dict[str, Any], where: str) -> None:
    """Grid dimensions of a lattice model, after its required fields are present."""
    grid = blueprint.get("grid") or {}
    _positive_int(f"{where}.grid.width", grid.get("width"))
    _positive_int(f"{where}.grid.height", grid.get("height"))


def _validated_monte_carlo(num_mcs: Any, save_every: Any, where: str) -> Dict[str, int]:
    """num_mcs / save_every that describe a run instead of an empty answer."""
    steps = _positive_int(
        f"{where}.num_mcs", num_mcs,
        "A non-positive step count ran zero Monte Carlo steps and returned a "
        "single frame at t=0, reported as a completed simulation.")
    if steps > MAX_MONTE_CARLO_STEPS:
        raise HTTPException(
            status_code=400,
            detail=(f"{where}.num_mcs={steps:,} exceeds the server limit of "
                    f"{MAX_MONTE_CARLO_STEPS:,} Monte Carlo steps for one request."),
        )
    every = _positive_int(f"{where}.save_every", 10 if save_every is None else save_every)
    return {"num_mcs": steps, "save_every": every}


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

        model = _validated_ode_model(req.blueprint)
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
            # Validate the numerics BEFORE the solver sees them: t_max=-10 used to
            # come back 200 with a single frame at t=0, dt=-0.1 the same, and a
            # grid key the solver does not read fell back to 50x50 in silence.
            _validated_pde_config(bp)
            # The same solve the target evaluator uses (agent.simulate_pde_blueprint):
            # identical initial conditions and seed, so the field shown here is the
            # field /api/evaluate grades. Two separate solve paths would let the
            # displayed pattern and the graded numbers drift apart.
            result = agent.simulate_pde_blueprint(
                bp, seed=(0 if req.seed is None else int(req.seed)), save_every=5
            )
            return result
        else:
            model = _validated_ode_model(bp)
            config = bp.get("simulation_config", {})
            t_max = _validated_horizon(config, 50.0)

            # Resolution scales with the time horizon so fast dynamics (oscillations,
            # sharp fold-change transients) are captured, not aliased by a coarse 100-point grid.
            num_points = int(min(5000, max(300, t_max * 15)))
            # num_points was already capped, but the CAP IS NOT A BUDGET: LSODA's
            # internal step count is driven by the equations, not by the output
            # grid, so a stiff right-hand side runs unbounded between two output
            # points. The deadline lives inside the integration for that reason.
            with _ode_execution_budget(model, t_max):
                result = model.simulate(t_max, num_points=num_points,
                                        custom_params=req.custom_params)
            return result
    except HTTPException:
        raise                       # keep the precise 400/422 above, don't mask it as a 500
    except ExecutionBudgetExceeded as e:
        raise _budget_exceeded(e)
    except agent.SpatialTargetError as e:
        # An unusable spatial blueprint (no reactions, non-numeric horizon) is the
        # caller's input problem, not a server fault.
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/optimize")
def optimize_parameters(req: OptimizeRequest):
    """Fits model parameters to target data curves."""
    model = _validated_ode_model(req.blueprint)

    # fit_parameters_to_target indexes params_dict[p] directly, so an unknown name
    # surfaced as 500 {"detail": "'does_not_exist'"}. Name it, and say what the
    # model does have -- the caller cannot guess the generated parameter names.
    known = sorted(model.params_dict.keys())
    unknown = [p for p in (req.params_to_fit or []) if p not in model.params_dict]
    if unknown:
        _reject(f"These parameters do not exist in this model: "
                f"{', '.join(repr(p) for p in unknown)}. "
                f"Available parameters: {', '.join(known) if known else '(none)'}.")
    if not req.target_times:
        _reject("target_times must contain at least one time point to fit against.")
    for index, moment in enumerate(req.target_times):
        _finite_number(f"target_times[{index}]", moment)
    for species, curve in (req.target_data or {}).items():
        if len(curve) != len(req.target_times):
            _reject(f"target_data[{species!r}] has {len(curve)} points but "
                    f"target_times has {len(req.target_times)}; they must match.")

    try:
        t_max = max(float(t) for t in req.target_times)
        # The fit calls simulate once per least-squares evaluation, so the same
        # in-integration deadline applies -- otherwise one bad model turns a fit
        # into the unbounded request /api/simulate was just protected from.
        with _ode_execution_budget(model, t_max):
            fitted, loss = model.fit_parameters_to_target(
                target_data=req.target_data,
                target_times=req.target_times,
                params_to_fit=req.params_to_fit
            )
        return {
            "fitted_parameters": fitted,
            "loss": loss
        }
    except HTTPException:
        raise
    except ExecutionBudgetExceeded as e:
        raise _budget_exceeded(e)
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
    # An empty body used to produce a 100x100 lattice over 500 Monte Carlo steps
    # and 5,621,290 bytes of JSON. Nothing in that request said what to simulate,
    # so there is nothing to report as a success -- and a 16-byte body that costs
    # the process 14 seconds is an amplification vector on an unauthenticated route.
    missing = _missing_lattice_fields(req.blueprint, "blueprint")
    if missing:
        _reject({
            "message": "This request does not describe a simulation, so none was run.",
            "missing": missing,
        })
    _validated_lattice(req.blueprint, "blueprint")
    config = req.blueprint.get("simulation_config") or {}
    bounds = _validated_monte_carlo(config.get("num_mcs", 500), config.get("save_every", 10),
                                    "blueprint.simulation_config")

    try:
        from abm_engine import build_cpm_from_blueprint
        cpm = build_cpm_from_blueprint(req.blueprint)
        result = cpm.simulate(
            num_mcs=bounds["num_mcs"],
            save_every=bounds["save_every"]
        )
        return result
    except HTTPException:
        raise
    except KeyError as e:
        _reject(f"The ABM blueprint is missing {e.args[0]!r}, which building the "
                f"Cellular Potts model requires.")
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
    # Same defect as /api/abm/simulate: POST {"blueprint": {}} answered 200 with
    # 1,212,558 bytes off a 100x100 default lattice. num_mcs is required here too,
    # because on this route the step count is a REQUEST field, not part of the
    # blueprint -- so an omitted num_mcs silently bought 100 Monte Carlo steps.
    missing = _missing_lattice_fields(req.abm_blueprint, "abm_blueprint")
    if req.num_mcs is None:
        missing.append("num_mcs - how many Monte Carlo steps to run")
    if missing:
        _reject({
            "message": "This request does not describe a simulation, so none was run.",
            "missing": missing,
        })
    _validated_lattice(req.abm_blueprint, "abm_blueprint")
    bounds = _validated_monte_carlo(req.num_mcs, req.save_every, "request")
    if req.ode_blueprint:
        _validated_ode_model(req.ode_blueprint, "ode_blueprint")

    try:
        from multiscale import MultiscaleSimulator
        sim = MultiscaleSimulator()
        sim.configure(
            abm_blueprint=req.abm_blueprint,
            ode_blueprint=req.ode_blueprint,
            coupling_rules=req.coupling_rules
        )
        result = sim.simulate(
            num_mcs=bounds["num_mcs"],
            save_every=bounds["save_every"]
        )
        return result
    except HTTPException:
        raise
    except KeyError as e:
        _reject(f"The multi-scale request is missing {e.args[0]!r}, which building "
                f"the coupled model requires.")
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
        # An unknown SPECIES used to be graded, not refused: the result came back
        # met=false with detail "No simulation data", which never names the species
        # and reads like a transient failure rather than a typo. Name it here, so a
        # refinement loop cannot mistake an unevaluated target for a failed one.
        # (An unknown METRIC is already refused with met=false by
        # agent.evaluate_targets_on_blueprint and never counted towards met_count.)
        known = _blueprint_species(req.blueprint)
        for index, target in enumerate(req.targets or []):
            if not isinstance(target, dict):
                _reject(f"targets[{index}] must be an object describing one target.")
            species = target.get("species")
            if species is None or str(species).strip() == "":
                continue
            if str(species) not in known:
                _reject(f"targets[{index}] names the species {str(species)!r}, which "
                        f"this model does not contain, so it cannot be evaluated. "
                        f"This model's species: "
                        f"{', '.join(known) if known else '(none)'}.")

        t_max = _validated_horizon(req.blueprint.get("simulation_config", {}) or {}, 50.0)
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
    except HTTPException:
        raise
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
    # Three silent rewrites lived here, each reported as a successful run:
    # method "telepathy" fell through to Monte-Carlo while the response echoed
    # "telepathy" as the method used; n_samples=-5 was clamped to 1 and reported as
    # n_requested=1; and min=5.0 max=0.1 was swapped behind the caller's back. A
    # sampling design the caller did not ask for invalidates everything downstream
    # of it, so each is refused instead.
    method = (req.method or "lhs").strip().lower()
    if method not in SAMPLING_METHODS:
        _reject(f"method {req.method!r} is not a sampling strategy this server "
                f"implements, so no sampling was done. Accepted: "
                f"{', '.join(SAMPLING_METHODS)}.")
    n_samples = _positive_int(
        "n_samples", 64 if req.n_samples is None else req.n_samples,
        "A non-positive count was clamped to 1 and then reported back as "
        "n_requested=1, so the response described a design nobody asked for.")
    if n_samples > MAX_SAMPLES:
        _reject(f"n_samples={n_samples} exceeds the server limit of {MAX_SAMPLES}; "
                f"the request was refused rather than silently reduced.")
    if not req.param_bounds:
        _reject("param_bounds must name at least one parameter with a min and a max.")
    for name, bound in req.param_bounds.items():
        if not isinstance(bound, dict) or "min" not in bound or "max" not in bound:
            _reject(f"param_bounds[{name!r}] needs both a 'min' and a 'max'.")
        low = _finite_number(f"param_bounds[{name!r}].min", bound["min"])
        high = _finite_number(f"param_bounds[{name!r}].max", bound["max"])
        if low >= high:
            _reject(f"param_bounds[{name!r}] has min={low:g} and max={high:g}. The "
                    f"minimum must be below the maximum; the pair is not swapped for "
                    f"you, because a reversed bound usually means the two values "
                    f"were entered against the wrong fields.")

    try:
        result = explore_parameter_space(
            blueprint=req.blueprint,
            param_bounds=req.param_bounds,
            n_samples=n_samples,
            method=method,
            target_species=req.target_species,
            seed=req.seed or 0,
        )
        return result
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- NEW: Sensitivity Analysis ---

@app.post("/api/sensitivity")
def run_sensitivity(req: SensitivityRequest):
    """Run local sensitivity analysis for parameter importance ranking."""
    # local_sensitivity_analysis returns 0.0 for anything it cannot compute: an
    # unknown parameter, and an unknown target species (which makes EVERY entry
    # 0.0). 0.0 is exactly what a genuinely insensitive parameter scores, so the
    # answer was indistinguishable from a real one. /api/sample already refuses the
    # unknown-parameter case with 400 and that message; this route now agrees.
    model = _validated_ode_model(req.blueprint)
    known_species = sorted(model.node_ids)
    if req.target_species not in model.node_ids:
        _reject(f"target_species {req.target_species!r} is not a species in this "
                f"model, so no sensitivity could be computed for any parameter. "
                f"This model's species: "
                f"{', '.join(known_species) if known_species else '(none)'}.")
    if not req.param_names:
        _reject("param_names must name at least one parameter to analyse.")
    unknown = [p for p in req.param_names if p not in model.params_dict]
    if len(unknown) == len(req.param_names):
        raise HTTPException(status_code=400,
                            detail="None of the requested parameters exist in this model.")
    if unknown:
        _reject(f"These parameters do not exist in this model: "
                f"{', '.join(repr(p) for p in unknown)}. They would each have been "
                f"reported as a sensitivity of 0.0, which is what a genuinely "
                f"insensitive parameter scores. Available parameters: "
                f"{', '.join(sorted(model.params_dict))}.")

    try:
        from multiscale import local_sensitivity_analysis
        t_max = _validated_horizon(req.blueprint.get("simulation_config", {}) or {}, 50.0)
        with _ode_execution_budget(model, t_max):
            result = local_sensitivity_analysis(
                blueprint=req.blueprint,
                target_species=req.target_species,
                param_names=req.param_names,
                t_max=t_max
            )
        return {"sensitivities": result}
    except HTTPException:
        raise
    except ExecutionBudgetExceeded as e:
        raise _budget_exceeded(e)
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
def _validated_topology(req: "TopologyRequest") -> Dict[str, Any]:
    """Refuse a malformed topology with 422 before any consumer walks it.

    topology_module.validate_topology already reports exactly these problems --
    `nodes_not_list`, `edges_not_list`, and members that are strings instead of
    objects. /api/topology/cycles and /api/topology/export simply never called it,
    so `{"nodes": "x", "edges": "y"}` reached adjacency() and nodes_to_csv() and
    came out as an opaque 500 from the global handler.
    """
    issues = topology_module.validate_topology(req.topology, req.domain)
    if not geometry_module.is_valid(issues):
        _reject(_issue_payload(issues))
    return _issue_payload(issues)


@app.post("/api/topology/validate")
def validate_topology(req: TopologyRequest):
    issues = topology_module.validate_topology(req.topology, req.domain)
    payload = _issue_payload(issues)
    try:
        payload["stats"] = topology_module.topology_stats(req.topology)
    except (AttributeError, TypeError, KeyError, ValueError):
        # The issues above are the answer for a malformed topology, and this route
        # exists to REPORT that rather than fail. Statistics over members that are
        # not objects are not computable, so they are omitted and said to be
        # omitted -- a crash here would hide the diagnosis the caller asked for.
        payload["stats"] = None
        payload["stats_unavailable"] = ("Statistics need nodes, edges and cells to be "
                                       "lists of objects; see the issues above.")
    return payload


@app.post("/api/topology/cycles")
def find_topology_cycles(req: TopologyRequest):
    """Candidate loops the user can convert into cells. Bounded by design."""
    _validated_topology(req)
    cycles = topology_module.find_cycles(req.topology)
    return {"cycles": cycles, "count": len(cycles),
            "max_length": topology_module.MAX_CYCLE_LENGTH,
            "capped": len(cycles) >= topology_module.MAX_CYCLES}


@app.post("/api/topology/export")
def export_topology(req: TopologyRequest):
    _validated_topology(req)
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
    if req.dt is not None:
        # dt=-1.0 used to return 200 with steps=0 and a single frame at t=0, while
        # the summary in the same payload said "Solved from t=0 to t=10". Nothing
        # was solved; the run is refused rather than described inaccurately.
        _positive_number("dt", req.dt,
                         "A non-positive time step takes no step at all, so the "
                         "solver returns the initial condition and reports the "
                         "whole horizon as solved.")
    dt = float(req.dt) if req.dt else (safe_dt if _np.isfinite(safe_dt) else 0.01)
    # A reversed or empty window (t_end <= t_start) is already refused with a 400 by
    # compile_field above, so `duration` here is always positive.
    duration = max(1e-9, compiled["t_end"] - compiled["t_start"])
    interval = _positive_number("model.fields[].output_interval",
                                compiled["output_interval"])

    # solve_1d builds one Python list entry per step (`step_sizes = [dt] * n_full`),
    # so the step count is a memory bound as well as a time bound: t_end=1e9 with
    # output_interval=1e8 needs ~2e12 entries at the stability-limited dt and came
    # back as an opaque 500 from a MemoryError. Refuse it, naming both values.
    steps = math.floor(duration / dt)
    if steps > MAX_PDE_STEPS:
        _reject(f"Solving from t={compiled['t_start']:g} to t={compiled['t_end']:g} at "
                f"dt={dt:g} needs {steps:,} explicit time steps, above the server "
                f"limit of {MAX_PDE_STEPS:,}. Shorten the time window, raise dt, or "
                f"coarsen the mesh (dt is limited to {safe_dt:.4g} by stability at "
                f"the current spacing dx={dx:g}). The window is not truncated for "
                f"you, because a shortened run reported as complete is worse.")
    save_every = max(1, int(round(interval / dt)))

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
