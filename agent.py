import re
import json
import time
import numpy as np
import scipy.optimize
from typing import Dict, Any, List, Tuple, Optional

import llm_provider
from simulation_engine import ODEModel

# DPO-style preference history: track (original, revised, outcome) tuples
refinement_history: List[Dict[str, Any]] = []

# System instruction shared by the LLM-backed generation calls.
_SYSTEM_COMPILER = (
    "You are a precise biological model compiler. You output ONLY raw JSON that "
    "matches the requested schema, with no prose, markdown, or code fences. "
    "You prefer explicit mechanistic rate laws (mass-conserving fluxes) over generic "
    "influence graphs whenever the description implies rates, transport, or conservation."
)

# Prompt template for natural-language -> blueprint compilation. Kept as a plain
# string with a <<<DESCRIPTION>>> placeholder (not an f-string) so the many literal
# JSON braces need no escaping. The key design goal: for MECHANISTIC descriptions the
# model must emit the custom-kinetics format (odes + fluxes + parameters) rather than
# defaulting to a generic Hill graph, and it must never promote a constant into a species.
_COMPILER_PROMPT = r"""You are a biological model compiler. Translate the natural-language description below into a structured JSON blueprint for simulation.

STEP 1 - VALIDATION. Act as a biological/physical consistency checker. If the description is physically impossible or self-contradictory (e.g. a membrane protein diffusing freely, mass created from nothing, a reaction that drives a concentration negative), do NOT build a model: return {"validation_errors": ["..."]} only. Missing detail is NOT a reason to refuse: a description with no rates, no parameters and no compartments is a normal, supported input - build the QUALITATIVE form of it in STEP 2 instead of refusing. Reserve validation_errors for a description that CONTRADICTS itself or physics.

STEP 2 - CHOOSE THE REPRESENTATION.
- MECHANISTIC (emit "odes" + "fluxes" + "parameters"): use this whenever the text describes explicit rate processes - entry/influx at a rate, pumping/transport between compartments or pools, release, leak, removal/extrusion, production/decay, or a rate that is "activated by"/"gated by" some quantity. This produces an exact mass-conserving rate-law model.
- QUALITATIVE (emit "edges"): use for a bare influence sketch ("A activates B, B inhibits C") with no rates, compartments, stimuli, or conservation. This is the CORRECT answer for such a description - emit the nodes and edges and stop. Do NOT refuse it, and do NOT invent an upstream stimulus or rate law to turn it into a mechanistic model. Give the most UPSTREAM species (one nothing else activates) a non-zero initial_value such as 1.0 and leave the downstream ones at 0.0: with every species at zero the graph is an exact fixed point and nothing can happen.
- SPATIAL / PDE (emit "type":"PDE" with a "spatial" block): use when the text mentions diffusion, spatial patterns, or reaction-diffusion (Turing).
Prefer MECHANISTIC when the text gives you any rate process to work from; use QUALITATIVE when it gives you none. Words like enters, pumped, released, leaks, removed, extruded, at a (constant) rate, stimulus, store/compartment/pool => MECHANISTIC.

STEP 3 - HARD RULES for the MECHANISTIC form (violating these makes the model wrong). They constrain the rate laws you write; they are never a reason to refuse a description or to add species it does not mention:
R1. A constant input rate, an external reservoir treated as fixed, or an applied stimulus level is a PARAMETER (a number), NOT a species. Never create a node or ODE for it, and never give it a degradation/decay term. ("Calcium enters from outside at a constant rate, increased by a stimulus" => two parameters like v0 and v1*beta in an influx term - NOT species named EXTERNAL or STIMULUS.)
R2. CONSERVED TRANSPORT. When material moves from pool X to pool Y (pump, release, transport, leak), model it as ONE named flux that appears with a MINUS sign in dX/dt and a PLUS sign in dY/dt. Never model transport as an "activation" edge, and never let transport create or destroy material.
R3. RATE GATING / FEEDBACK. When a flux is "strongly activated by"/"gated by" a species S, multiply that flux by a Hill term, e.g. S**n/(K**n + S**n). Do NOT add a separate edge for this - it belongs inside the flux expression.
R4. REMOVAL from the system (extrusion out of the cell, efflux, degradation) is a linear loss term (- k*X) inside that species' OWN ode. It is NOT an inhibition edge to another node.
R5. Only create a node for a quantity that genuinely changes over time and is described dynamically.
R6. DIMENSIONAL CONSISTENCY. Every term of every d/dt must be a rate (amount per time). Never subtract a bare count from a concentration or add quantities of different kinds. Loss terms must be proportional to the species being lost (- k*X), so no concentration can go negative.
R7. DISTINCT CONSTANTS. Use a separate named rate constant for each distinct process. Never reuse one symbol (e.g. k_act) for two different rates (association AND removal). Give each its own parameter.
R8. DRIVERS/STIMULI. A quantity that is imposed/held from outside (an applied Ca level, a tetanus, a clamp) is a PARAMETER, or - if it must vary in time - a smooth algebraic function of t (e.g. a logistic step 1/(1+exp(-s*(t-t0)))). NEVER write a piecewise if(...) and never give an imposed driver its own accumulation ODE.
R9. ODES AND SPECIES MUST MATCH EXACTLY. Every entry in "odes" must be a species in "nodes", and every species in "nodes" must have exactly one entry in "odes". Never write an ODE for something that is not a declared species; never declare a species that has no ODE (that is a dead node - make it a parameter instead). Do not leave a species defined with d/dt = 0 that nothing else uses.
R10. THE SYSTEM MUST BE ABLE TO LEAVE ITS INITIAL STATE. If production of a species is autocatalytic (proportional to an already-active/phosphorylated form that starts at zero), you MUST also include a separate INITIATION or basal term that does not depend on that form, so the first bit can appear. Otherwise the all-zero state is a permanent fixed point and nothing ever happens.
R11. EVERY SPECIES MUST ACTUALLY MOVE. Each species you declare must visibly change during the simulation. It is NOT enough that some species move: check each one. A species whose only source term is proportional to itself (dRAS/dt = k*RAS/(1+RAS) - ...) or to another species that starts at zero stays pinned at zero forever, and every species downstream of it does too - the plot is a flat line at 0 and reports nothing. Drive the first species of any chain from something that is already non-zero at t=0.
R12. ONE SPECIES PER NAMED ENTITY. If the description names ERK, model ERK - do NOT emit ERK plus a separate ERK_act/ERKa/pERK twin and put the dynamics in the twin, because the species the researcher named would then be a flat unused pool. If an inactive/active split is genuinely required, make the interconversion two-way so BOTH forms vary. Never leave the named species constant.
R13. BOUNDED MAGNITUDES. Concentrations must stay within a few orders of magnitude of the initial values and parameters you chose. A species that grows exponentially to thousands of times the model's own scale is wrong, and it is almost always caused by adding one flux to two different species without subtracting it anywhere (material out of nothing), or by an unsaturated self-production term. Every named flux that MOVES material must appear with a minus sign in the source ODE and a plus sign in the destination ODE.
R14. MODEL ONLY WHAT IS DESCRIBED. Do not elaborate the description into the textbook pathway it reminds you of. "RAS activates RAF. RAF activates MEK." is a THREE-species model: do NOT add EGF, EGFR, ERK, or any other upstream or downstream step the user did not mention. Extra species change every trajectory in the model and appear to the user as species they asked for. Introduce an unnamed intermediate only when the described mechanism cannot be written without it.

SCHEMA (include only the relevant fields):
{
  "validation_errors": [],
  "type": "ODE",
  "nodes": [ {"id": "SHORTID", "name": "full name", "initial_value": 0.0} ],
  "parameters": {"name": number},                       // constants: influx rates, rate constants, thresholds K, Hill exponents n, stimulus levels
  "fluxes": {"flux_name": "expression"},                // MECHANISTIC: named shared rate laws over species ids + parameters
  "odes": {"SPECIES_ID": "d[species]/dt expression"},   // MECHANISTIC: may reference flux names, species, parameters
  "edges": [ {"source":"ID","target":"ID","type":"activation|inhibition","parameters":{"k":0.5,"K_d":1.0,"n":2.0}} ],  // QUALITATIVE mode only
  "spatial": {"x_grid":50,"y_grid":50,"dx":1.0,"dy":1.0,"diffusion":{"ID":0.1},"reactions":{"ID":"algebraic expression in species ids"}},  // PDE only
  "simulation_config": {"t_max": 50.0, "dt": 0.1}
}
EXPRESSIONS: use + - * / ** and functions like exp(); refer to species by id and constants by parameter name. A named flux may appear in several odes (that is how conservation is expressed).

FORMAT EXAMPLE (shows the mechanistic pattern - do NOT copy these values or this system):
Description: "Substance S is supplied at a constant rate and converted to product P; the conversion is switched on by P itself; P is then removed from the system."
{
  "type": "ODE",
  "nodes": [ {"id":"S","name":"Substrate","initial_value":1.0}, {"id":"P","name":"Product","initial_value":0.0} ],
  "parameters": {"vin":1.0, "vmax":5.0, "Kp":1.0, "nP":4.0, "kout":1.0},
  "fluxes": {"conv":"vmax*(P**nP/(Kp**nP + P**nP))*S"},
  "odes": {"S":"vin - conv", "P":"conv - kout*P"},
  "simulation_config": {"t_max":50.0}
}
Here the constant supply "vin" is a PARAMETER (not a species), the conversion is a shared flux gated by P via a Hill term, and removal is "- kout*P". Nothing was invented for "constant rate" or "the system".

Biological Description:
"<<<DESCRIPTION>>>"

Return ONLY the raw JSON, no markdown, no backticks, no comments."""


# ==========================================
# 1. BIOLOGICAL TEXT PARSER (STAGE 1)
# ==========================================

def _coerce_id(val: Any) -> Optional[str]:
    """Best-effort turn an LLM 'id' into a clean string. Handles the common
    malformations: an id given as a nested dict ({"id": "ERK"} / {"name": "ERK"})
    or as a number. Returns None if nothing usable can be extracted."""
    if isinstance(val, dict):
        val = val.get("id") or val.get("name") or val.get("species")
    if val is None:
        return None
    s = str(val).strip()
    return s or None


def _coerce_number(val: Any, default: float = 1.0) -> float:
    """Turn an LLM parameter/initial value into a float. Handles values wrapped as
    objects ({"value": 0.5, "units": "1/s"}) and non-numeric junk, which otherwise
    reach SymPy as a dict and crash the compiler ('name should be a string, not dict')."""
    if isinstance(val, dict):
        for k in ("value", "val", "default", "mean", "initial_value"):
            if k in val:
                val = val[k]
                break
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def sanitize_blueprint(bp: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize an LLM blueprint so it can never crash the ODE compiler or the graph
    renderer, for BOTH model modes (generic edges AND custom-kinetics odes):

      * node ids / parameter values are coerced to clean strings / floats, so a
        value the model wrapped in an object (or a mistyped id) can't reach SymPy as
        a dict. This is what previously made a slightly-off model burn all 3 repair
        rounds and fall back to the rule-based parser.
      * in generic (edges) mode, every edge endpoint is auto-declared as a node if
        the model referenced a compartment it forgot to declare.
    """
    if not isinstance(bp, dict):
        return bp

    # --- normalize "type": the schema defines ODE and PDE. Models invent other values
    # ("qualitative", "influence graph", "ode"), and an unrecognised type would reach the
    # API and the frontend as a third mode nothing knows how to render.
    raw_type = bp.get("type")
    if raw_type is not None:
        bp["type"] = "PDE" if str(raw_type).strip().upper() == "PDE" else "ODE"

    # --- normalize nodes: each becomes {"id": <str>, "initial_value": <float>, ...} ---
    nodes = bp.get("nodes")
    if isinstance(nodes, list):
        norm_nodes, seen = [], set()
        for n in nodes:
            if isinstance(n, dict):
                nid = _coerce_id(n.get("id") or n.get("name") or n.get("species"))
            elif isinstance(n, str):
                nid = _coerce_id(n)
            else:
                nid = None
            if not nid or nid in seen:
                continue
            seen.add(nid)
            node = dict(n) if isinstance(n, dict) else {}
            node["id"] = nid
            for k in ("initial_value", "initial", "value"):
                if k in node:
                    node["initial_value"] = _coerce_number(node.get(k), 0.0)
                    break
            norm_nodes.append(node)
        bp["nodes"] = nodes = norm_nodes

    # --- normalize parameters: {str: float} ---
    # Models also emit this block as a LIST of objects ([{"name": "k", "value": 0.5}]),
    # which reaches the compiler as a list and crashes it with
    # "'list' object has no attribute 'items'". Coerce that shape instead of dropping it,
    # so a model that is otherwise fine is not thrown away over its parameter container.
    params = bp.get("parameters")
    if isinstance(params, list):
        rebuilt: Dict[str, Any] = {}
        for item in params:
            if not isinstance(item, dict):
                continue
            name = _coerce_id(item.get("name") or item.get("id") or item.get("parameter")
                              or item.get("symbol"))
            if not name:
                continue
            rebuilt[name] = item.get("value", item.get("default", item.get("val", 1.0)))
        bp["parameters"] = params = rebuilt
    if isinstance(params, dict):
        norm = {}
        for k, v in params.items():
            key = str(k).strip()
            if key:
                norm[key] = _coerce_number(v, 1.0)
        bp["parameters"] = norm
    elif params is not None:
        bp["parameters"] = {}

    # --- normalize fluxes / odes: {str: str} (drop non-scalar expression values) ---
    for key in ("fluxes", "odes"):
        d = bp.get(key)
        if isinstance(d, dict):
            nd = {}
            for k, v in d.items():
                kk = str(k).strip()
                if kk and not isinstance(v, (dict, list)):
                    nd[kk] = str(v)
            bp[key] = nd

    # --- generic (edges) mode: auto-declare orphan edge endpoints ---
    edges = bp.get("edges")
    if isinstance(edges, list) and isinstance(bp.get("nodes"), list):
        nodes = bp["nodes"]
        node_ids = {n.get("id") for n in nodes if isinstance(n, dict) and n.get("id")}
        for e in edges:
            if not isinstance(e, dict):
                continue
            # Per-edge parameters are read with .get(), so a list or a scalar here is an
            # AttributeError in the compiler. Anything that is not a {name: number} map
            # becomes an empty map, which makes the compiler use its Hill defaults.
            ep = e.get("parameters")
            if isinstance(ep, list):
                rebuilt = {}
                for item in ep:
                    if isinstance(item, dict):
                        name = _coerce_id(item.get("name") or item.get("id"))
                        if name:
                            rebuilt[name] = _coerce_number(
                                item.get("value", item.get("default", 1.0)), 1.0)
                e["parameters"] = rebuilt
            elif isinstance(ep, dict):
                e["parameters"] = {str(k).strip(): _coerce_number(v, 1.0)
                                   for k, v in ep.items() if str(k).strip()}
            elif ep is not None:
                e["parameters"] = {}
            for key in ("source", "target"):
                nid = _coerce_id(e.get(key))
                if nid:
                    e[key] = nid
                    if nid not in node_ids:
                        nodes.append({"id": nid, "name": nid, "initial_value": 0.0})
                        node_ids.add(nid)
    return bp


# Prompt used to ask the model to REPAIR a blueprint that failed compile/simulation.
_REPAIR_PROMPT = r"""The JSON model you produced for this description does NOT work. Fix it and return corrected JSON only.

Description:
"<<<DESCRIPTION>>>"

Your previous JSON:
<<<PREVIOUS>>>

Defect found when compiling/simulating it:
<<<ERROR>>>

Return ONLY the corrected raw JSON (same schema: type, nodes, parameters, fluxes, odes, simulation_config). Keep the mechanism and mass conservation - just fix the defect. Enforce: constants/stimuli are PARAMETERS (never species); every term of each d/dt is a rate; a distinct rate constant per process (never reuse one symbol for two rates); no species may go negative (loss terms proportional to the species); a time-varying driver is a parameter or a smooth function of t, never a piecewise if(); the odes keys and the species set must be identical (no ODE for a non-species, no species without an ODE, no dead d/dt=0 species); the system must be able to leave its initial state (if production is autocatalytic in a form that starts at zero, add a separate initiation/basal term so it is not frozen); EVERY species must visibly change (a species whose only source is proportional to itself, or to another species that starts at zero, stays pinned at zero forever and so does everything downstream of it); ONE species per entity the description names (no flat named pool beside an _act/a/p twin that carries all the dynamics); and every flux that MOVES material must appear with a minus sign in the source ODE and a plus sign in the destination ODE, so nothing grows without bound."""


def _validation_error_messages(raw: Any) -> List[str]:
    """Normalize model-supplied validation errors without trusting their shape."""
    values = raw if isinstance(raw, (list, tuple, set)) else [raw]
    return [str(value).strip() for value in values if value is not None and str(value).strip()]


def _is_nonblocking_sparse_refusal(errors: List[str]) -> bool:
    """True only for refusals that mean "not enough kinetics", not physical invalidity.

    Bare qualitative graphs are valid input because the deterministic Hill compiler can
    supply generic dynamics. Every other refusal is preserved for the user rather than
    silently compiling a model the LLM judged contradictory or impossible.

    This has to read FREE-FORM model prose, so it cannot be a list of exact phrases.
    Measured against Bedrock (mistral.mistral-large-3-675b-instruct, us-east-2), the
    input "MEK activates ERK." was refused with

        "Description is a bare influence sketch with no rates, compartments, or
         mechanistic details. ... A qualitative influence graph is the only valid
         representation."

    which matched none of the original phrase list, so the most basic input a researcher
    can type came back as a hard validation error with NO model - even though a bare
    influence sketch is exactly what the deterministic Hill compiler exists to handle.
    The test is therefore: does the refusal complain about MISSING information, and does
    it stop short of naming a concrete contradiction?
    """
    if not errors:
        return False

    # A concrete physical/logical contradiction. Each marker must describe THE SYSTEM
    # being wrong, never the modelling being under-determined: a bare "impossible" is
    # deliberately absent, because a sparse refusal routinely says it is "impossible to
    # construct a mass-conserving model" from a description with no rates in it, and
    # reading that as a contradiction is what hard-failed "MEK activates ERK."
    blocking_markers = (
        "contradict", "creates mass", "mass from nothing", "mass out of nothing",
        "violates conservation", "violates mass", "violates the conservation",
        "conservation of mass is violated", "not mass-conserving because",
        "cannot be negative", "goes negative", "negative concentration",
        "concentration negative", "drives the concentration",
        "mutually exclusive", "logically inconsistent", "self-inconsistent",
        "self-contradictory", "physically impossible", "biologically impossible",
        "thermodynamically impossible", "physically unrealizable", "unphysical",
    )
    # A complaint about ABSENT detail. A fixed phrase list cannot keep up with model
    # prose - three runs of the same input produced "no mechanistic details", "lacks
    # mechanistic detail" and "without any mechanistic details" - so the real test is
    # structural: a negator sitting close to a word for the kind of information that is
    # missing. The explicit phrases below only cover wordings with no negator at all.
    negation_window = re.compile(
        r"\b(no|not|non|none|without|lack|lacks|lacking|missing|absent|omits?|"
        r"insufficient|inadequate|underdetermined|under-determined|only|bare|sparse|"
        r"unspecified|ambiguous|unclear|more|additional|further)\b"
        r"[^.;]{0,60}?"
        r"\b(rate|rates|kinetic|kinetics|mechanistic|mechanism|mechanisms|parameter|"
        r"parameters|quantitative|numeric|numerical|number|value|values|compartment|"
        r"compartments|conservation|constant|constants|detail|details|information|"
        r"initial|stoichiometr\w*|dynamics|transport|process|processes|specific\w*)\b",
        re.IGNORECASE)
    sparse_phrases = (
        "too sparse", "sparse", "only qualitative", "qualitative description",
        "qualitative influence", "influence sketch", "influence graph",
        "influence statement", "influence diagram", "qualitative model",
        "qualitative representation", "cannot determine", "cannot be determined",
        "cannot tell", "does not specify", "does not say", "provides no",
        "is not specified", "unable to determine",
    )
    for error in errors:
        lowered = error.lower()
        if any(marker in lowered for marker in blocking_markers):
            return False
        if not (negation_window.search(lowered)
                or any(phrase in lowered for phrase in sparse_phrases)):
            return False
    return True


def _pde_blueprint_status(bp: Dict[str, Any]) -> Tuple[str, str]:
    """Schema-check and smoke-test a PDE blueprint on a tiny grid."""
    nodes = bp.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        return "broken", "No PDE species (nodes) were produced."

    node_ids = [str(node.get("id", "")).strip() for node in nodes if isinstance(node, dict)]
    if len(node_ids) != len(nodes) or any(not node_id for node_id in node_ids):
        return "broken", "Every PDE species must have a non-empty string id."
    if len(set(node_ids)) != len(node_ids):
        return "broken", "PDE species ids must be unique."

    spatial = bp.get("spatial")
    if not isinstance(spatial, dict):
        return "broken", "A PDE blueprint needs a spatial configuration."
    reactions = spatial.get("reactions")
    if not isinstance(reactions, dict) or not reactions:
        return "broken", "A PDE blueprint needs one reaction expression per species."
    if any(not isinstance(name, str) or not isinstance(formula, str) or not formula.strip()
           for name, formula in reactions.items()):
        return "broken", "Every PDE reaction must map a species id to a non-empty expression."

    node_set, reaction_set = set(node_ids), set(reactions)
    orphan = reaction_set - node_set
    missing = node_set - reaction_set
    if orphan:
        return "broken", f"PDE reactions are defined for undeclared species {sorted(orphan)}."
    if missing:
        return "broken", f"PDE species {sorted(missing)} have no reaction expression."

    try:
        nx, ny = int(spatial.get("x_grid", 50)), int(spatial.get("y_grid", 50))
        dx, dy = float(spatial.get("dx", 1.0)), float(spatial.get("dy", 1.0))
        config = bp.get("simulation_config") or {}
        t_max, dt = float(config.get("t_max", 100.0)), float(config.get("dt", 0.1))
    except (TypeError, ValueError, OverflowError):
        return "broken", "PDE grid, spacing, time, and step values must be numeric."
    numeric_values = (dx, dy, t_max, dt)
    if nx < 3 or ny < 3:
        return "broken", "PDE grids must be at least 3 by 3."
    if not all(np.isfinite(value) and value > 0 for value in numeric_values):
        return "broken", "PDE spacing, simulation time, and time step must be positive finite values."
    if t_max < dt:
        return "broken", "PDE simulation time must be at least one time step."

    diffusions = spatial.get("diffusion", {})
    if not isinstance(diffusions, dict):
        return "broken", "PDE diffusion coefficients must be a species-to-number mapping."
    unknown_diffusions = set(diffusions) - node_set
    if unknown_diffusions:
        return "broken", f"Diffusion coefficients reference undeclared species {sorted(unknown_diffusions)}."
    try:
        diffusion_values = {name: float(diffusions.get(name, 0.1)) for name in node_ids}
    except (TypeError, ValueError, OverflowError):
        return "broken", "PDE diffusion coefficients must be numeric."
    if any(not np.isfinite(value) or value < 0 for value in diffusion_values.values()):
        return "broken", "PDE diffusion coefficients must be non-negative finite values."

    node_map = {node_id: node for node_id, node in zip(node_ids, nodes)}
    initial_conditions: Dict[str, Any] = {}
    try:
        for name in node_ids:
            base = float(node_map[name].get("initial_value", 1.0))
            if not np.isfinite(base):
                raise ValueError
            initial_conditions[name] = {"type": "uniform", "base_value": base}
    except (TypeError, ValueError, OverflowError):
        return "broken", "PDE initial values must be finite numbers."

    # Compile every reaction and execute only 1-3 steps on at most a 6x6 grid.
    # Comparisons avoid evaluating t_max / dt, which can overflow for extreme but
    # individually finite values.
    smoke_steps = 3 if dt <= t_max / 3.0 else (2 if dt <= t_max / 2.0 else 1)
    smoke_tmax = smoke_steps * dt
    smoke_spatial = dict(spatial)
    smoke_spatial.update({
        "x_grid": min(nx, 6), "y_grid": min(ny, 6),
        "dx": dx, "dy": dy, "diffusion": diffusion_values,
    })
    try:
        from simulation_engine import solve_pde
        with np.errstate(all="ignore"):
            result = solve_pde(
                smoke_spatial, reactions, initial_conditions,
                t_max=smoke_tmax, dt=dt, save_every=1,
            )
    except Exception as e:
        return "broken", f"The PDE reaction system does not compile or run: {e}."

    times = np.asarray(result.get("t", []), dtype=float)
    if times.size == 0 or not np.all(np.isfinite(times)) or times[-1] < smoke_tmax * 0.99:
        return "broken", "The PDE smoke test did not reach its requested end time."
    for name in node_ids:
        values = np.asarray((result.get("species") or {}).get(name, []), dtype=float)
        if values.size == 0 or not np.all(np.isfinite(values)):
            return "broken", f"PDE species '{name}' produced non-finite values during the smoke test."
    return "ok", ""


def _blueprint_status(bp: Dict[str, Any]) -> Tuple[str, str]:
    """Compile and smoke-test an ODE or PDE blueprint.

    Returns ``(status, message)`` where status is ``ok``, ``imperfect`` (an ODE
    runs but dips negative), or ``broken`` (invalid schema, compile/integration
    failure, or non-finite values).
    """
    if not isinstance(bp, dict):
        return "broken", "Blueprint is not a JSON object."
    if bp.get("validation_errors"):
        return "ok", ""                      # a deliberate model refusal is valid output
    if str(bp.get("type", "ODE")).upper() == "PDE":
        return _pde_blueprint_status(bp)
    if not bp.get("nodes"):
        return "broken", "No species (nodes) were produced."

    # Defect 12a: in custom-kinetics mode the ODE keys and the species set must be identical.
    # An ODE for a non-species is silently dropped by the compiler; a species with no ODE just
    # sits constant. Either one means the odes and nodes disagree (the phantom-node class of bug).
    odes = bp.get("odes")
    if isinstance(odes, dict) and odes:
        node_ids_set = {n.get("id") for n in bp.get("nodes", []) if isinstance(n, dict) and n.get("id")}
        orphan = set(odes.keys()) - node_ids_set
        if orphan:
            return "broken", (f"ODE(s) are defined for {sorted(orphan)} but those are not in the species (nodes) "
                              f"list, so their dynamics are silently dropped. Every ODE key must be a declared species.")
        missing = node_ids_set - set(odes.keys())
        if missing:
            return "broken", (f"Species {sorted(missing)} have no ODE, so they sit constant. Give every species its "
                              f"own d/dt, or make it a parameter instead of a species.")

    try:
        model = ODEModel(bp)
    except Exception as e:
        return "broken", (f"The model does not compile: {e}. Every symbol used in odes/fluxes must be a "
                          f"species id or a declared parameter, and expressions must be valid arithmetic.")

    # Defect 12b: a species whose derivative is identically zero AND is referenced by no other
    # equation is a dead species (e.g. a declared Ca_CaM with dCa_CaM/dt = 0, used nowhere).
    try:
        for sid in model.node_ids:
            expr = model.deriv_exprs.get(sid)
            if expr is not None and expr.is_zero:
                sym = model.vars.get(sid)
                used = any(sym in model.deriv_exprs[o].free_symbols
                           for o in model.node_ids if o != sid)
                if not used:
                    return "broken", (f"Species '{sid}' never changes (d/dt = 0) and is referenced by no other "
                                      f"equation - it is a dead species. Remove it, or give it a dynamical role.")
    except Exception:
        pass
    try:
        t_max = float((bp.get("simulation_config") or {}).get("t_max", 50.0) or 50.0)
        sim_tmax = min(max(t_max, 1.0), 200.0)
        res = model.simulate(sim_tmax, num_points=120)
    except Exception as e:
        return "broken", f"The model compiles but the simulation crashes: {e}."
    # Integration must reach the end time; a short trajectory means the solver bailed
    # (stiff system or finite-time blow-up from runaway positive feedback).
    tvals = res.get("t") or []
    if not tvals or float(tvals[-1]) < sim_tmax * 0.9:
        return "broken", (f"The integration failed before the end time (stopped at t={float(tvals[-1]) if tvals else 0:.3g} "
                          f"of {sim_tmax:.3g}) - the system is stiff or blows up. Add saturation to any runaway positive "
                          f"feedback and keep all rates bounded.")
    worst_neg = 0.0
    moved = False
    for sid, vals in (res.get("species") or {}).items():
        arr = np.asarray(vals, dtype=float)
        if arr.size == 0 or not np.all(np.isfinite(arr)):
            return "broken", (f"Species '{sid}' produced non-finite (NaN/Inf) values - the rate laws are unstable or "
                              f"dimensionally inconsistent (e.g. subtracting a subunit count from a concentration, or "
                              f"reusing one rate constant for two processes). Make every term of each d/dt a proper rate.")
        worst_neg = min(worst_neg, float(np.nanmin(arr)))
        span = float(np.nanmax(arr) - np.nanmin(arr))
        scale = max(abs(float(arr[0])), abs(float(np.nanmean(arr))), 1e-9)
        if span > 1e-6 and (span / scale) > 1e-3:
            moved = True

    # Defect 11: the model must actually evolve, not sit frozen at its initial condition.
    # (A model where every production flux is proportional to a species that starts at zero has
    #  the initial state as an exact fixed point: it "runs clean" but does nothing.)
    if not moved:
        return "broken", ("The model stays frozen at its initial state - no species changes over the whole "
                          "simulation. Usually every production flux is proportional to a species that starts at "
                          "zero, making the start an exact fixed point. Add an initiation or basal input term (like "
                          "the paper's v1) so the system can leave its starting state.")
    if worst_neg < -1e-3:
        return "imperfect", (f"A species dips negative (min {worst_neg:.3g}); a concentration cannot be negative. "
                             f"Make each loss term proportional to the species it removes.")
    return "ok", ""


def validate_blueprint(bp: Dict[str, Any]) -> Tuple[str, str]:
    """Public validation entry point used by API compilation and generation."""
    return _blueprint_status(bp)


# =============================================================================
# LLM-OUTPUT INTEGRITY AUDIT  (applied to LLM output only)
# =============================================================================
# _blueprint_status() answers "does this compile and run?". For LLM output that is
# not enough: the two worst defects measured against real AWS Bedrock models BOTH
# compile, integrate to t_max, stay finite and stay non-negative, so they were
# graded "ok" and handed to the researcher as validated models.
#
#  1. DEAD NAMED POOL. For an EGF -> EGFR -> RAS -> RAF -> MEK -> ERK cascade, the
#     model emitted  egfr_ras = k_egfr_ras*RAS/(1 + RAS)  as the ONLY source term of
#     RAS. RAS starts at 0, so dRAS/dt is 0 for all time, and RAF/MEK/ERK downstream
#     of it are identically 0 as well: FOUR of the six species the text named were
#     flat lines at exactly 0.0 while EGF and EGFR evolved normally.
#     _blueprint_status passed it because its "did anything move?" test is a single
#     OR over all species - one moving species exonerates every other one. A target
#     placed on ERK then measures a flat zero, and a flat trace is indistinguishable
#     from a real result. This is the same failure mode as the historical
#     ERK / ERK_act split, where the species the researcher named is a flat pool
#     beside a dynamic twin that carries all the actual dynamics.
#
#  2. UNBOUNDED GROWTH FROM MASS CREATION. The same cascade was written as
#     dEGFR/dt = -egf_egfr + egfr_ras  while ALSO writing  dRAS/dt = +egfr_ras:
#     one flux added to two species with a minus sign nowhere, i.e. material out of
#     nothing, and autocatalytic in EGFR. EGFR reached 10226.8 from an initial value
#     of 1.0 (RAS 5113, RAF 5083) and was still climbing at t_max - yet the run was
#     finite and non-negative, so this too was graded "ok".
#
# The audit therefore adds what a compile-and-run smoke test structurally cannot see:
# per-species dynamics, model-relative magnitude, and flux sign conservation. It runs
# ONLY on LLM output, so the deterministic parser and a researcher's own hand-written
# equations keep their existing (looser) contract.

# A species is a runaway when it exceeds this multiple of the model's OWN scale
# (largest initial value or parameter magnitude) and is still growing at t_max.
_RUNAWAY_FACTOR = 1.0e3


def _llm_model_scale(bp: Dict[str, Any]) -> float:
    """The model's own level scale: the largest initial value or parameter magnitude.
    Judging magnitude relative to this (never against an absolute number) keeps a
    model whose natural units are large from being called a runaway."""
    vals = [1.0]
    for node in bp.get("nodes") or []:
        if isinstance(node, dict):
            try:
                vals.append(abs(float(node.get("initial_value", 0.0) or 0.0)))
            except (TypeError, ValueError):
                pass
    params = bp.get("parameters")
    if isinstance(params, dict):
        for v in params.values():
            try:
                vals.append(abs(float(v)))
            except (TypeError, ValueError):
                pass
    return max(vals)


def _llm_active_twin(sid: str, candidates: List[str], is_flat) -> Optional[str]:
    """Return a DYNAMIC species that is clearly the activated twin of ``sid``
    (ERK -> ERKa / ERK_act / pERK / ppERK), or None. Matched loosely on the shape
    "another species' name plus a short suffix" rather than against a fixed list of
    spellings, because the suffix a model picks varies from run to run - 'a' and
    '_act' have both been observed for the same description."""
    for other in candidates:
        if other == sid or is_flat(other):
            continue
        if _llm_twin_base(other, [sid, other]) == sid:
            return other
    return None


def _llm_flux_conservation_error(bp: Dict[str, Any]) -> Optional[str]:
    """Detect a named flux that ADDS material to two or more species while removing it
    from none - material out of nothing.

    A real conversion or transport flux always carries a matching minus sign: A -> B + C
    is -1 in dA/dt and +1 in each of dB/dt and dC/dt. A flux that is positive in several
    ODEs and negative in none has no source. A flux whose expression uses NO species (a
    constant supply term) is exempt, because a constant influx may legitimately feed
    several pools at once.
    """
    import sympy as sp
    fluxes, odes = bp.get("fluxes"), bp.get("odes")
    if not (isinstance(fluxes, dict) and fluxes and isinstance(odes, dict) and odes):
        return None
    species = {n.get("id") for n in bp.get("nodes") or []
               if isinstance(n, dict) and n.get("id")}

    parsed: Dict[str, Any] = {}
    for sid, expr in odes.items():
        try:
            parsed[sid] = sp.sympify(str(expr))
        except Exception:
            return None          # unparsable: the compile check in _blueprint_status owns it

    for fname, fexpr in fluxes.items():
        try:
            fsym = sp.Symbol(str(fname))
            body = sp.sympify(str(fexpr))
        except Exception:
            continue
        if not ({str(s) for s in body.free_symbols} & species):
            continue             # parameter-only supply: a legitimate shared source
        plus, minus = [], []
        for sid, expr in parsed.items():
            if fsym not in expr.free_symbols:
                continue
            try:
                d = sp.diff(expr, fsym)
                val = float(d.subs({s: 1.0 for s in d.free_symbols}))
            except Exception:
                continue
            if val > 1e-12:
                plus.append(sid)
            elif val < -1e-12:
                minus.append(sid)
        if len(plus) >= 2 and not minus:
            return (f"Flux '{fname}' is ADDED to {sorted(plus)} and subtracted from nothing, so it "
                    f"creates material out of nothing (it depends on a species, so it is not a "
                    f"constant supply term). The same flux must appear with a MINUS sign in the ODE "
                    f"of the pool it comes FROM and a PLUS sign in the pool it enters.")
    return None


def _llm_model_integrity(bp: Dict[str, Any]) -> Tuple[str, str]:
    """Audit an LLM blueprint that already compiles and runs.

    Returns ``(status, message)`` with the same vocabulary as ``_blueprint_status``:
    ``ok``, ``imperfect`` (usable but flagged to the user), or ``broken`` (send back
    for repair, and never present as a validated model).
    """
    if not isinstance(bp, dict) or bp.get("validation_errors"):
        return "ok", ""
    if str(bp.get("type", "ODE")).upper() == "PDE":
        return "ok", ""                       # PDE fields are graded by _pde_blueprint_status
    if not bp.get("nodes"):
        return "ok", ""

    conservation = _llm_flux_conservation_error(bp)

    try:
        model = ODEModel(bp)
        t_max = float((bp.get("simulation_config") or {}).get("t_max", 50.0) or 50.0)
        sim_tmax = min(max(t_max, 1.0), 200.0)
        res = model.simulate(sim_tmax, num_points=200)
        traces = {sid: np.asarray(vals, dtype=float)
                  for sid, vals in (res.get("species") or {}).items()}
    except Exception:
        # A model that will not compile or run is already _blueprint_status's verdict.
        return ("broken", conservation) if conservation else ("ok", "")
    if not traces:
        return ("broken", conservation) if conservation else ("ok", "")

    def _flat(sid: str) -> bool:
        a = traces.get(sid)
        if a is None or a.size == 0 or not np.all(np.isfinite(a)):
            return True
        span = float(np.nanmax(a) - np.nanmin(a))
        local = max(abs(float(a[0])), abs(float(np.nanmean(a))), 1e-9)
        return not (span > 1e-6 and (span / local) > 1e-3)

    ids = list(traces.keys())
    flat = sorted(sid for sid in ids if _flat(sid))
    moving = [sid for sid in ids if sid not in flat]

    # Nothing moved at all -> already reported by _blueprint_status; do not double-report.
    if not moving:
        return ("broken", conservation) if conservation else ("ok", "")

    # --- Defect 1a: a declared species that is identically zero for the whole run. ---
    dead = [sid for sid in flat
            if float(np.nanmax(np.abs(traces[sid]))) <= 1e-9]
    if dead:
        return "broken", (
            f"Species {dead} never leave zero: they are flat at exactly 0.0 for the whole "
            f"simulation while {moving} evolve, so the model reports nothing about them. "
            f"A named species that stays at zero is an unused pool, and a flat trace cannot be "
            f"told apart from a real result. Usually the only production term for the first such "
            f"species is proportional to that species itself (or to another species that starts "
            f"at zero), which makes zero a permanent fixed point - give it a source term that does "
            f"NOT depend on it, or drive it from a species that is already non-zero at t=0."
            + (f" Also: {conservation}" if conservation else ""))

    # --- Defect 1b: a flat pool beside a dynamic activated twin (ERK vs ERKa). ---
    # _llm_resolve_active_twins() runs before this and COLLAPSES the pair whenever the
    # flat pool is a pure decoy, so anything reaching here is a pool that other rate
    # laws genuinely use. That split may be real, but it must never be silent: grade it
    # imperfect so the reason travels to the user in _llm_notice.
    twins = [(sid, _llm_active_twin(sid, ids, _flat)) for sid in flat]
    twins = [(sid, twin) for sid, twin in twins if twin]
    if twins:
        orphans = [(sid, twin) for sid, twin in twins
                   if not _llm_symbol_used_elsewhere(bp, sid)]
        if orphans:
            pairs = ", ".join(f"'{sid}' (flat) beside '{twin}' (dynamic)" for sid, twin in orphans)
            return "broken", (
                f"The model split a species into an inert pool and an active twin: {pairs}. The "
                f"species the description names must be the one that carries the dynamics; as "
                f"written, a target placed on it would measure a flat line. Emit ONE species per "
                f"named entity, or make the interconversion two-way so the named species varies."
                + (f" Also: {conservation}" if conservation else ""))
        pairs = "; ".join(
            f"'{sid}' is constant while '{twin}' carries the dynamics - read '{twin}'"
            for sid, twin in twins)
        return "imperfect", (
            f"The model uses an inactive/active split: {pairs}. "
            + (conservation or ""))

    # --- Defect 2: unbounded growth (mass creation / unsaturated autocatalysis). ---
    scale = _llm_model_scale(bp)
    for sid, a in traces.items():
        hi = float(np.nanmax(np.abs(a)))
        if hi <= _RUNAWAY_FACTOR * scale:
            continue
        tail = a[max(1, int(0.9 * a.size)):]
        still_growing = (tail.size >= 2 and
                         float(tail[-1] - tail[0]) > 1e-6 * max(1.0, abs(float(tail[-1]))))
        if still_growing:
            return "broken", (
                f"Species '{sid}' grows without bound: it reaches {hi:.4g}, which is "
                f"{hi / scale:.0f}x this model's own scale ({scale:.4g}, the largest initial value "
                f"or parameter), and it is still increasing at t_max, so the trajectory has no "
                f"physical meaning. "
                + (conservation or
                   "Add saturation to the responsible term, or subtract the growth flux from the "
                   "pool it consumes so material is conserved."))

    if conservation:
        return "broken", conservation

    # --- Flat but non-zero and with no active twin: usable, but the user must be told. ---
    if flat:
        return "imperfect", (
            f"Species {flat} never change over the whole simulation (they sit at their initial "
            f"value). Any target placed on them would measure a constant.")
    return "ok", ""


# Gene / protein-like tokens a description names explicitly. Deliberately narrow:
# an all-caps or mixed-caps token with at least one letter and no lowercase-only form
# (ERK, MEK, EGFR, MDM2, G6P, p53, LacI, TetR), which is how researchers write species.
_NAMED_TOKEN_RE = re.compile(r"\b(?:[A-Z][A-Za-z]*\d+[A-Za-z\d]*|[A-Z]{2,}\d*|[a-z]\d{2,})\b")
_NAMED_TOKEN_STOPWORDS = {
    "ODE", "PDE", "JSON", "DNA", "RNA", "MRNA", "ATP", "ADP", "GTP", "GDP", "NAD",
    "AND", "OR", "NOT", "THE", "A", "AN", "IF", "IS", "ARE", "IT", "ITS", "BY", "AT",
    "TO", "OF", "IN", "ON", "FOR", "WITH", "THEN", "ALSO", "BOTH", "EACH", "OTHER",
}


def _llm_missing_named_species(text: str, bp: Dict[str, Any]) -> List[str]:
    """Tokens the description names explicitly that appear NOWHERE in the model.

    Heuristic and advisory only: it never changes the blueprint's status, it only
    warns the researcher (and hints the repair round) when the thing they asked about
    is absent. A missing name is how "the model silently dropped ERK" reaches a user
    who then reads a plot of something else.
    """
    if not isinstance(bp, dict) or bp.get("validation_errors"):
        return []
    tokens = []
    for tok in _NAMED_TOKEN_RE.findall(text or ""):
        if tok.upper() in _NAMED_TOKEN_STOPWORDS or len(tok) < 2:
            continue
        if tok not in tokens:
            tokens.append(tok)
    if not tokens:
        return []

    haystack = []
    for node in bp.get("nodes") or []:
        if isinstance(node, dict):
            haystack += [str(node.get("id", "")), str(node.get("name", ""))]
    for key in ("parameters", "fluxes", "odes"):
        d = bp.get(key)
        if isinstance(d, dict):
            haystack += [str(k) for k in d.keys()]
            haystack += [str(v) for v in d.values() if not isinstance(v, (dict, list))]
    blob = " ".join(haystack).lower()

    missing = [tok for tok in tokens if tok.lower() not in blob]
    return missing


def _llm_text_tokens(text: str) -> List[str]:
    """Lower-cased alphanumeric chunks of the description, for support matching."""
    return [t for t in re.split(r"[^A-Za-z0-9]+", (text or "").lower()) if t]


def _llm_unrequested_species(text: str, bp: Dict[str, Any]) -> List[str]:
    """Species the model introduced that the description gives NO basis for.

    Measured defect: for the description "RAS activates RAF. RAF activates MEK." a model
    returned an entire textbook cascade - EGF, EGFR, RAS, RAF, MEK, ERK plus an 'a'
    twin of each - so a researcher modelling three species got a model of eleven. The
    two extra kinases are not an implementation detail: they change every trajectory in
    the model and appear in the target-species dropdown as if the researcher had asked
    for them.

    A species counts as SUPPORTED when its id or its display name shares a word stem
    with the description, or when it is the active twin / binary complex of a supported
    species (RASa from RAS; EGF_EGFR from EGF and EGFR). Advisory: an LLM may legitimately
    name an intermediate the text only implies (hexokinase products, a phosphorylated
    form), so this drives a repair instruction and a user-visible notice, never a refusal.
    """
    if not isinstance(bp, dict) or bp.get("validation_errors"):
        return []
    nodes = [n for n in (bp.get("nodes") or []) if isinstance(n, dict) and n.get("id")]
    if not nodes:
        return []
    words = _llm_text_tokens(text)
    if not words:
        return []

    def _supported_by_text(label: str) -> bool:
        for chunk in _llm_text_tokens(label):
            if len(chunk) < 2:
                continue
            for w in words:
                if chunk == w or (len(chunk) >= 3 and chunk in w) or (len(w) >= 3 and w in chunk):
                    return True
        return False

    ids = [str(n["id"]) for n in nodes]
    supported = set()
    for node in nodes:
        sid = str(node["id"])
        if _supported_by_text(sid) or _supported_by_text(str(node.get("name") or "")):
            supported.add(sid)

    # Derived forms of a supported species are supported too: an active twin, and a
    # complex whose id is built out of two or more supported ids (EGF_EGFR, Ca_CaM).
    for _ in range(3):
        grew = False
        for sid in ids:
            if sid in supported:
                continue
            base = _llm_twin_base(sid, ids)
            if base and base in supported:
                supported.add(sid)
                grew = True
                continue
            parts = [p for p in re.split(r"[^A-Za-z0-9]+", sid) if p]
            if len(parts) >= 2 and all(
                    any(p.lower() == other.lower() or p.lower() in other.lower()
                        for other in supported) for p in parts):
                supported.add(sid)
                grew = True
        if not grew:
            break
    return [sid for sid in ids if sid not in supported]


# Short tokens that mark an "activated"/"modified" form. Matched with ANY separator and
# at either end, so no single spelling is hard-coded: 'a' and '_act' have both been
# observed from the same model for the same description. The set matters, though - a
# free-for-all "base plus <=5 letters" match makes EGFR look like an active form of EGF,
# which would merge two genuinely different proteins.
_ACTIVATION_TOKENS = frozenset({
    "a", "act", "acti", "active", "activ", "activated", "on", "star",
    "p", "pp", "ppp", "phos", "phospho", "phosph", "phosphorylated",
    "i", "inact", "inactive", "free", "bound", "total", "tot", "cyt", "nuc",
})


def _llm_twin_base(twin: str, candidates: List[str]) -> Optional[str]:
    """If ``twin`` is another species' name plus a short activation token (as a suffix or
    a prefix), return that other species. The LONGEST matching candidate wins, so 'EGFRa'
    resolves to 'EGFR' rather than to 'EGF' when both are present."""
    low = twin.lower()
    for other in sorted((c for c in candidates if c != twin), key=len, reverse=True):
        o = other.lower()
        if len(low) <= len(o):
            continue
        if low.startswith(o):
            tail = low[len(o):].lstrip("_- ")
            if tail in _ACTIVATION_TOKENS:
                return other
        if low.endswith(o):
            head = low[:len(low) - len(o)].rstrip("_- ")
            if head in _ACTIVATION_TOKENS:
                return other
    return None


def _llm_symbol_used_elsewhere(bp: Dict[str, Any], sid: str) -> bool:
    """True when ``sid`` appears in any rate law other than its own d/dt - i.e. the pool
    genuinely feeds or gates something, rather than sitting there as a decoy."""
    pattern = re.compile(r"\b" + re.escape(sid) + r"\b")
    fluxes = bp.get("fluxes")
    if isinstance(fluxes, dict):
        for expr in fluxes.values():
            if isinstance(expr, str) and pattern.search(expr):
                return True
    odes = bp.get("odes")
    if isinstance(odes, dict):
        for key, expr in odes.items():
            if key != sid and isinstance(expr, str) and pattern.search(expr):
                return True
    edges = bp.get("edges")
    if isinstance(edges, list):
        for e in edges:
            if isinstance(e, dict) and sid in (e.get("source"), e.get("target")):
                return True
    return False


def _llm_rename_species(bp: Dict[str, Any], old: str, new: str) -> None:
    """Rename a species everywhere in the blueprint, on word boundaries."""
    pattern = re.compile(r"\b" + re.escape(old) + r"\b")
    for node in bp.get("nodes") or []:
        if isinstance(node, dict) and str(node.get("id")) == old:
            node["id"] = new
            if not node.get("name") or str(node.get("name")) == old:
                node["name"] = new
    odes = bp.get("odes")
    if isinstance(odes, dict):
        bp["odes"] = {(new if k == old else k):
                      (pattern.sub(new, v) if isinstance(v, str) else v)
                      for k, v in odes.items()}
    fluxes = bp.get("fluxes")
    if isinstance(fluxes, dict):
        bp["fluxes"] = {k: (pattern.sub(new, v) if isinstance(v, str) else v)
                        for k, v in fluxes.items()}
    for e in bp.get("edges") or []:
        if isinstance(e, dict):
            for key in ("source", "target"):
                if e.get(key) == old:
                    e[key] = new


def _llm_resolve_active_twins(bp: Dict[str, Any]) -> List[str]:
    """Repair the active-form duplication, in place, and return user-facing notes.

    The defect: the model emits both the species the researcher named (RAF) and an
    activated twin (RAFa / RAF_act / pRAF), puts every bit of the dynamics in the twin,
    and leaves the named species a flat pool. A target set on RAF then measures a
    constant, and a flat trace is indistinguishable from a real result.

    Two outcomes, chosen per pair on evidence rather than by preference:

      (a) COLLAPSE - the flat named pool is referenced by no other rate law, so it is a
          pure decoy. Delete it and rename the dynamic twin to the researcher's name.
          The model then contains exactly the species the description named. This is
          the default because it removes the ambiguity instead of documenting it.

      (b) DISCLOSE - the flat pool IS used by another rate law, so the inactive/active
          split is part of the mechanism and deleting it would change the model. Keep
          both and say plainly, in the notice, that the split happened and which species
          carries the dynamics (that is also the species the target evaluator retargets
          onto, so the readout and the disclosure agree).
    """
    notes: List[str] = []
    if not isinstance(bp, dict) or bp.get("validation_errors"):
        return notes
    if str(bp.get("type", "ODE")).upper() == "PDE":
        return notes
    nodes = [n for n in (bp.get("nodes") or []) if isinstance(n, dict) and n.get("id")]
    if len(nodes) < 2:
        return notes

    try:
        model = ODEModel(bp)
        t_max = float((bp.get("simulation_config") or {}).get("t_max", 50.0) or 50.0)
        res = model.simulate(min(max(t_max, 1.0), 200.0), num_points=200)
        traces = {sid: np.asarray(v, dtype=float)
                  for sid, v in (res.get("species") or {}).items()}
    except Exception:
        return notes                       # cannot judge without a trajectory
    if not traces:
        return notes

    def _flat(sid: str) -> bool:
        a = traces.get(sid)
        if a is None or a.size == 0 or not np.all(np.isfinite(a)):
            return True
        span = float(np.nanmax(a) - np.nanmin(a))
        local = max(abs(float(a[0])), abs(float(np.nanmean(a))), 1e-9)
        return not (span > 1e-6 and (span / local) > 1e-3)

    ids = [str(n["id"]) for n in nodes]
    pairs = []
    for twin in ids:
        base = _llm_twin_base(twin, ids)
        if base and _flat(base) and not _flat(twin):
            pairs.append((base, twin))

    for base, twin in pairs:
        if base not in [str(n["id"]) for n in bp.get("nodes") or [] if isinstance(n, dict)]:
            continue                       # already handled by an earlier pair
        span_twin = float(np.nanmax(traces[twin]) - np.nanmin(traces[twin]))
        if _llm_symbol_used_elsewhere(bp, base):
            # (b) the split is load-bearing: keep it, but never leave it undisclosed.
            notes.append(
                f"The AI split '{base}' into an inactive pool and an active form '{twin}'. "
                f"'{base}' stays constant at {float(traces[base][0]):.4g} for the whole run, so "
                f"'{twin}' is the species that carries the dynamics (it varies over "
                f"{span_twin:.4g}). Results are being measured on '{twin}': a target on "
                f"'{base}' would measure a flat line.")
            continue
        # (a) the named pool is a decoy: nothing reads it. Collapse the pair.
        bp["nodes"] = [n for n in bp.get("nodes") or []
                       if not (isinstance(n, dict) and str(n.get("id")) == base)]
        odes = bp.get("odes")
        if isinstance(odes, dict):
            odes.pop(base, None)
        _llm_rename_species(bp, twin, base)
        notes.append(
            f"The AI produced both '{base}' and an active twin '{twin}', with '{base}' left "
            f"as an unused flat pool. '{base}' was referenced by no rate law, so the two were "
            f"merged: the dynamics now live on '{base}', the name your description used.")
    return notes


def _llm_seed_qualitative_sources(bp: Dict[str, Any]) -> List[str]:
    """Give an edges-mode influence graph a non-zero starting point, in place.

    Measured defect: asked for the qualitative form of "MEK activates ERK.", Bedrock
    returned the correct graph but with EVERY initial value at 0.0. The Hill activation
    term is proportional to the source species, so at the all-zero state every derivative
    is zero: the model is an exact fixed point, the smoke test correctly calls it frozen,
    and three repair rounds are then spent throwing away a graph whose only flaw was its
    initial condition. Seed the upstream drivers (a species that is a source but never a
    target) so the cascade can run; a pure cycle gets its first source seeded instead.
    """
    notes: List[str] = []
    if not isinstance(bp, dict) or bp.get("validation_errors"):
        return notes
    if str(bp.get("type", "ODE")).upper() == "PDE":
        return notes
    if bp.get("odes"):
        return notes                       # custom-kinetics mode owns its own initials
    edges = bp.get("edges")
    nodes = [n for n in (bp.get("nodes") or []) if isinstance(n, dict) and n.get("id")]
    if not (isinstance(edges, list) and edges and nodes):
        return notes
    try:
        if any(abs(float(n.get("initial_value", 0.0) or 0.0)) > 1e-12 for n in nodes):
            return notes                   # something already starts non-zero
    except (TypeError, ValueError):
        return notes

    sources = [e.get("source") for e in edges if isinstance(e, dict) and e.get("source")]
    targets = {e.get("target") for e in edges if isinstance(e, dict) and e.get("target")}
    drivers = [s for s in dict.fromkeys(sources) if s not in targets]
    if not drivers:
        drivers = [sources[0]]             # every node is downstream of something: a cycle
    seeded = []
    for node in nodes:
        if str(node.get("id")) in drivers:
            node["initial_value"] = 1.0
            seeded.append(str(node.get("id")))
    if seeded:
        notes.append(
            "The AI produced an influence graph with every species starting at zero, which "
            "cannot evolve, so " + ", ".join(seeded) +
            " " + ("were" if len(seeded) > 1 else "was") + " set to 1.0 as the upstream "
            "input. Adjust the initial values in the Model Summary if that is not the "
            "starting state you meant.")
    return notes


def _llm_coverage_notices(text: str, bp: Dict[str, Any]) -> List[str]:
    """User-facing warnings about the gap between what the description named and what the
    model contains, in both directions: entities dropped, and species invented."""
    notes: List[str] = []
    missing = _llm_missing_named_species(text, bp)
    if missing:
        notes.append(
            "Warning: your description names " + ", ".join(missing) +
            " but the model contains no species, parameter, or rate law for " +
            ("them" if len(missing) > 1 else "it") +
            ". Check the Model Summary before reading any result as being about " +
            ("those" if len(missing) > 1 else "that") + ".")
    extra = _llm_unrequested_species(text, bp)
    if extra:
        notes.append(
            "Warning: the AI added " + ", ".join(extra) +
            ", which your description does not mention. " +
            ("These species affect" if len(extra) > 1 else "This species affects") +
            " every trajectory in the model - delete " +
            ("them" if len(extra) > 1 else "it") +
            " in the Model Summary if you did not want a larger pathway than you described.")
    return notes


def _llm_coverage_repair_hint(text: str, bp: Dict[str, Any]) -> str:
    """Extra repair instructions naming exactly what the model dropped or invented."""
    hint = ""
    missing = _llm_missing_named_species(text, bp)
    if missing:
        hint += ("\n\nAlso: the description explicitly names " + ", ".join(missing) +
                 ", but your model has no species or rate law for " +
                 ("them" if len(missing) > 1 else "it") +
                 ". Represent every entity the description names.")
    extra = _llm_unrequested_species(text, bp)
    if extra:
        hint += ("\n\nAlso: you added " + ", ".join(extra) + ", which the description never "
                 "mentions. Model ONLY what the description states - do not elaborate it into a "
                 "larger textbook pathway, and do not add upstream or downstream steps the user "
                 "did not ask for.")
    return hint


def parse_biological_text(text: str, llm: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Translates a natural language description into a structured biological blueprint.
    Uses an open-source LLM (local GGUF or a remote OpenAI-compatible endpoint) when
    one is configured; otherwise falls back to the deterministic regex parser.

    The LLM output is put through a compile+simulate validation loop: if the model
    won't run (crashes, NaN, negative concentrations, dimensional nonsense) the error
    is fed back to the model to self-repair, up to a few rounds, before falling back.
    """
    if llm_provider.wants_llm(llm):
        try:
            client = llm_provider.build_client(llm)

            prompt = _COMPILER_PROMPT.replace("<<<DESCRIPTION>>>", text)
            data = llm_provider.generate_json(client, prompt, system=_SYSTEM_COMPILER)

            MAX_ATTEMPTS = 3                 # 1 initial + up to 2 self-repairs
            best_effort = None               # last blueprint that at least runs (imperfect)
            best_effort_reason = ""          # why that one was only imperfect
            best_effort_notes: List[str] = []
            last_error = ""
            declined_as_qualitative = False  # the LLM judged the text a bare sketch
            for attempt in range(MAX_ATTEMPTS):
                errors = _validation_error_messages(
                    data.get("validation_errors") if isinstance(data, dict) else None
                )
                if errors:
                    if _is_nonblocking_sparse_refusal(errors):
                        # A bare influence graph is valid input; let the deterministic
                        # compiler supply generic kinetics if the LLM supplied no nodes.
                        data = dict(data)
                        data.pop("validation_errors", None)
                        declined_as_qualitative = True
                    else:
                        # Physical contradictions and impossible models are deliberate
                        # refusals from STEP 1 of the compiler prompt. Never erase them.
                        return {"validation_errors": errors}

                has_nodes = (isinstance(data, dict) and isinstance(data.get("nodes"), list)
                             and len(data["nodes"]) > 0)
                if not has_nodes:
                    last_error = "the model returned no usable species"
                    break                    # -> deterministic fallback below
                bp = sanitize_blueprint(data)
                # Repair the two defects that are deterministically fixable, BEFORE
                # grading: an influence graph frozen at all-zero initial values, and the
                # active-form duplication where the named species is a flat twin.
                notes = _llm_seed_qualitative_sources(bp)
                notes += _llm_resolve_active_twins(bp)
                status, err = _blueprint_status(bp)
                # A blueprint that compiles and runs can still be scientifically empty
                # (named species flat at zero) or physically meaningless (unbounded
                # growth from mass creation). Both grade "ok" on a compile-and-run smoke
                # test, so audit every runnable LLM blueprint before accepting it.
                if status in ("ok", "imperfect"):
                    integrity, integrity_err = _llm_model_integrity(bp)
                    if integrity == "broken":
                        status, err = "broken", integrity_err
                    elif integrity == "imperfect":
                        status = "imperfect"
                        err = f"{err} {integrity_err}".strip() if err else integrity_err
                if status == "ok":
                    notices = list(notes)
                    if attempt > 0:
                        notices.append(f"Model self-repaired and validated after {attempt} fix round(s).")
                    notices.extend(_llm_coverage_notices(text, bp))
                    if notices:
                        bp["_llm_notice"] = " ".join(notices)
                    return bp
                if status == "imperfect":
                    best_effort = bp          # runs but flawed; keep as a fallback candidate
                    best_effort_reason = err
                    best_effort_notes = list(notes)
                last_error = err
                if attempt < MAX_ATTEMPTS - 1:
                    repair = (_REPAIR_PROMPT
                              .replace("<<<DESCRIPTION>>>", text)
                              .replace("<<<PREVIOUS>>>", json.dumps(data)[:4000])
                              .replace("<<<ERROR>>>", err))
                    repair += _llm_coverage_repair_hint(text, bp)
                    data = llm_provider.generate_json(client, repair, system=_SYSTEM_COMPILER)

            # No fully-valid model after all rounds: prefer a model that at least RUNS
            # (flagged), else build one deterministically from the description.
            if best_effort is not None:
                best_effort.pop("validation_errors", None)
                best_effort["_llm_notice"] = " ".join(best_effort_notes + [
                    f"Model runs but did not fully validate: {best_effort_reason or last_error}"]
                    + _llm_coverage_notices(text, best_effort))
                return best_effort
            fallback = sanitize_blueprint(rule_based_parse(text))
            if fallback.get("validation_errors"):
                fallback["_llm_notice"] = (
                    f"The AI could not build a model ({last_error}) and the deterministic parser "
                    f"needs more detail, so no model was built.")
            elif declined_as_qualitative:
                # Not a defect: the AI judged the description a bare influence sketch, which
                # the deterministic Hill compiler is designed to handle. Say what happened
                # rather than raising an alarm about a normal, supported input.
                fallback["_llm_notice"] = (
                    "The AI judged your description a qualitative influence sketch with no rate "
                    "information, so it was compiled deterministically with generic Hill kinetics. "
                    "The species and interactions are yours; the rate constants are placeholders - "
                    "add rates to the description, or use closed-loop refinement, to make them real.")
            else:
                # Every AI attempt was rejected outright. Say so instead of presenting the
                # deterministic model as though the AI had produced it - a silent swap is how
                # a researcher ends up trusting a model nobody claimed to have built.
                fallback["_llm_notice"] = (
                    f"The AI output was REJECTED as unusable ({last_error}). What you see is the "
                    f"deterministic parser's model of your description, not the AI's - review it "
                    f"before trusting any result, and refine it with closed-loop refinement.")
            return fallback
        except Exception as e:
            # Never expose provider response text or credential diagnostics to the UI/log.
            print(f"LLM parsing failed ({type(e).__name__}); falling back to rule-based parser.")
            fallback = sanitize_blueprint(rule_based_parse(text))
            fallback["_llm_notice"] = "The configured AI model was unavailable; used the rule-based parser."
            return fallback

    return sanitize_blueprint(rule_based_parse(text))


_IMAGE_TRANSCRIBE_PROMPT = """You are reading a photo of a hand-written or printed biological model.
Transcribe EXACTLY what is written, into plain text, so it can be compiled into a simulation.

Include, if present:
- every differential equation, written like "dX/dt = ..." using the exact variable names shown,
- every species/variable and its initial value,
- every parameter and its numeric value,
- any reaction or interaction that is described in words.

Rules: transcribe ONLY what is actually in the image. Do not invent species, terms, or numbers.
Keep the mathematical structure faithful (Hill terms, fractions, products, sums). Output ONLY the
transcription (the equations and the lists of species/parameters), with no commentary."""


def parse_image_to_blueprint(image_bytes: bytes, fmt: str,
                             llm: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Read hand-written equations from a photo and compile them into a validated
    blueprint, reusing the same generate->validate->repair pipeline as typed text.
    The returned blueprint carries a '_transcription' field so the UI can show what
    the model read from the image."""
    if not llm_provider.wants_llm(llm):
        raise ValueError("Configure an AI engine (AWS Bedrock) before reading equations from an image.")
    client = llm_provider.build_client(llm)
    if not hasattr(client, "transcribe_image"):
        raise ValueError("The selected engine cannot read images. Switch to the AWS Bedrock engine for photo-to-equation.")
    transcription = client.transcribe_image(image_bytes, fmt, _IMAGE_TRANSCRIBE_PROMPT)
    if not (transcription or "").strip():
        raise ValueError("No equations could be read from the image. Try a sharper, better-lit photo.")
    bp = parse_biological_text(transcription, llm)
    if isinstance(bp, dict):
        bp["_transcription"] = transcription
    return bp


def build_blueprint_from_equations(text: str) -> Dict[str, Any]:
    """Deterministically turn a user-written ODE system into a runnable blueprint,
    with NO LLM in the loop, so the model is EXACTLY what the user typed. Accepted
    lines (any order, one per line, ';' also separates; '#' starts a comment):

        dX/dt = <expr>        or   X' = <expr>          -> ODE for species X
        name = <expression>                            -> auxiliary flux (uses species)
        name = <number>                                -> parameter value
        X = <number>          or   X starts at <number> -> initial value of species X

    '^' is accepted for powers. Undeclared symbols used in the equations become
    parameters (default 1.0). The result is sanitized and compile-checked; any
    problem is returned under 'validation_errors' so the UI can show it."""
    import sympy as sp
    raw_lines = re.split(r"[\n;]+", text or "")
    odes: Dict[str, str] = {}
    aux: Dict[str, str] = {}
    params: Dict[str, float] = {}
    initials: Dict[str, float] = {}
    assignments: List[Tuple[str, str]] = []

    for ln in raw_lines:
        ln = ln.split("#")[0].strip()
        if not ln:
            continue
        m = re.match(r"^d\s*([A-Za-z_]\w*)\s*/\s*d\s*t\s*=\s*(.+)$", ln, re.IGNORECASE)
        if not m:
            m = re.match(r"^([A-Za-z_]\w*)\s*'\s*=\s*(.+)$", ln)      # X' = ...
        if m:
            odes[m.group(1)] = m.group(2).strip()
            continue
        m = re.match(r"^([A-Za-z_]\w*)\s+(?:starts?\s+at|initial(?:\s+value)?(?:\s+of)?)\s+([-+0-9.eE]+)$", ln, re.IGNORECASE)
        if m:
            try:
                initials[m.group(1)] = float(m.group(2))
            except ValueError:
                pass
            continue
        # A line can carry several comma-separated assignments (e.g. "v0=1, v1=7.3").
        # Split only on commas that begin a new "name =" assignment, so commas inside
        # an expression (function args) are left intact.
        for seg in re.split(r",(?=\s*[A-Za-z_]\w*\s*=)", ln):
            seg = seg.strip()
            mm = re.match(r"^([A-Za-z_]\w*)\s*=\s*(.+)$", seg)
            if mm:
                assignments.append((mm.group(1), mm.group(2).strip()))

    species = set(odes.keys())
    for lhs, rhs in assignments:
        try:
            val = float(rhs)
            is_num = True
        except ValueError:
            is_num = False
        if lhs in species:
            if is_num:
                initials[lhs] = val
            # a species with an expression assignment is already defined by its ODE
        elif is_num:
            params[lhs] = val
        else:
            aux[lhs] = rhs

    # Normalize '^' -> '**' so SymPy reads powers, not bitwise-xor.
    odes = {k: v.replace("^", "**") for k, v in odes.items()}
    aux = {k: v.replace("^", "**") for k, v in aux.items()}

    # Any symbol used but not a species / flux / declared parameter becomes a
    # parameter (default 1.0), so it shows up in the editable table.
    known = species | set(aux.keys()) | {"t"}
    for expr in list(odes.values()) + list(aux.values()):
        try:
            e = sp.sympify(expr)
            for s in e.free_symbols:
                name = str(s)
                if name not in known and name not in params:
                    params[name] = 1.0
        except Exception:
            pass  # unparsable expr is caught by the compile check below

    nodes = [{"id": s, "initial_value": float(initials.get(s, 0.0))} for s in sorted(species)]
    bp: Dict[str, Any] = {
        "type": "ODE",
        "nodes": nodes,
        "parameters": params,
        "fluxes": aux,
        "odes": odes,
        "simulation_config": {"t_max": 50.0},
    }
    bp = sanitize_blueprint(bp)

    if not odes:
        bp["validation_errors"] = ["No differential equations found. Write at least one line like 'dX/dt = ...'."]
        return bp
    status, err = _blueprint_status(bp)
    if status == "broken":
        bp["validation_errors"] = [err]
    return bp


def rule_based_parse(text: str) -> Dict[str, Any]:
    """
    Fallback parser using regular expressions and keywords.
    """
    text_lower = text.lower()
    
    # Stopwords that should never be treated as species/node names.
    #
    # These are ENGLISH FUNCTION WORDS only. This set used to also contain PROTEIN,
    # MOLECULE, RECEPTOR, ENZYME, FACTOR and COMPLEX -- every one of which is a species
    # name a biologist actually writes. "LIGAND activates RECEPTOR. RECEPTOR activates
    # KINASE. KINASE activates TARGET." lost RECEPTOR entirely, which broke the chain, so
    # KINASE and TARGET came out completely FLAT and only LIGAND moved. A four-step
    # cascade -- the most basic thing anyone types into this box -- silently produced a
    # model missing a step, with no warning anywhere.
    #
    # The intent was to catch generic usage ("the protein activates X"). Dropping the
    # word was the wrong remedy: a missing node breaks every downstream species
    # invisibly, whereas modelling a generically-named species is at worst inelegant and
    # is exactly what the researcher described. Anything rejected here is now also
    # REPORTED if a relation referenced it, so a lost species can never be silent.
    STOPWORDS = {
        "IT", "ITSELF", "THEM", "THIS", "THAT", "WHICH", "THE", "AND", "OR",
        "TO", "IS", "ARE", "WAS", "WERE", "IN", "ON", "AT", "BY", "OF",
        "A", "AN", "ITS", "THEN", "ALSO", "BOTH", "EACH", "WITH", "FROM",
        "NOT", "BUT", "IF", "SO", "AS", "BE", "HAS", "HAD", "HAVE",
        "SELF", "DOES", "DO", "DID", "WILL", "CAN", "MAY", "SHOULD",
        # Adverbs the PDE branch reads as diffusion-rate hints, not species.
        "SLOWLY", "QUICKLY", "RAPIDLY", "FAST", "SLOW",
    }

    #: Names rejected by is_valid_species that a relation actually referred to. A
    #: dropped endpoint means the described chain is broken, which the caller must
    #: surface rather than returning a quietly incomplete model.
    rejected_in_relations = set()

    def is_valid_species(name: str) -> bool:
        """Check if a name is a valid biological species identifier."""
        raw = (name or "").strip()
        upper = raw.upper()
        # Must contain at least one letter
        if not raw or not any(c.isalpha() for c in raw):
            return False
        # Single-letter species are legitimate and common in textbook models
        # ("A activates B", "X inhibits Y"). Accept them when written in caps, which
        # distinguishes a species from the English article "a".
        if len(raw) == 1:
            return raw.isupper()
        if upper in STOPWORDS:
            rejected_in_relations.add(upper)
            return False
        return True
    
    # Adverbs and connectives that sit BETWEEN the subject and its verb, which pushes
    # the real subject out of the capture slot: "INSULIN in turn lowers GLUCOSE" parsed
    # as TURN -| GLUCOSE, and "P53 also drives production of MDM2" as ALSO -> MDM2. The
    # species was replaced by an English word and the real relation was lost. These are
    # removed from the copy of the sentence used for RELATION matching only -- the
    # original text still feeds the PDE/diffusion-rate detection, which reads words like
    # "slowly" as meaningful.
    interjections = re.compile(
        r"\b(?:in\s+turn|then|also|subsequently|in\s+addition|additionally|however|"
        r"therefore|thus|consequently|meanwhile|finally|next|again|further|furthermore|"
        r"moreover|indeed|actually|effectively|directly|indirectly|strongly|weakly|"
        r"rapidly|slowly|quickly|steadily|gradually|immediately|eventually|typically|"
        r"normally|usually|often|sometimes|always|only|just|both|either)\b",
        re.IGNORECASE)

    def _scrub(text: str) -> str:
        """Drop interjections and collapse the whitespace they leave behind."""
        return re.sub(r"\s{2,}", " ", interjections.sub(" ", text)).strip()

    def _first_valid_species(sentence: str):
        """The first token in the sentence that could be a species.

        Used only to recover the SUBJECT of a passive clause that a conjunction has
        pushed out of reach: "P53 is produced steadily and is degraded by MDM2" matches
        `<word> is degraded by <word>` with the first word being "and", which is not a
        species. The subject of such a sentence is its opening species, so falling back
        to it recovers MDM2 -| P53 instead of discarding the clause. Returns None when
        the sentence opens with nothing usable, and the relation is then skipped.
        """
        for token in re.findall(r"\w+", sentence):
            upper = token.upper()
            if is_valid_species(upper) and not token.replace(".", "").isdigit():
                return upper
        return None

    def _orient(match, is_reversed: bool, sentence: str):
        """Resolve one regex match into (source, target), or (None, None) to skip.

        `is_reversed` is set for passive voice, where the sentence names the target
        first: "ERK is phosphorylated by MEK" is the edge MEK -> ERK, and emitting it
        the other way round would invert the biology while looking perfectly sane.
        """
        first, second = str(match[0]).upper(), str(match[1]).upper()
        if is_reversed:
            target, source = first, second
            if not is_valid_species(target):
                # A conjunction stole the subject slot -- recover it from the sentence.
                recovered = _first_valid_species(sentence)
                if recovered is None:
                    return None, None
                target = recovered
        else:
            source, target = first, second
        if not is_valid_species(source) or not is_valid_species(target):
            return None, None
        return source, target

    # 1. Detect PDE vs ODE
    is_pde = any(w in text_lower for w in ["pde", "reaction-diffusion", "turing", "spatial", "pattern", "diffusion"])
    
    # Default structures
    if is_pde:
        # SPATIAL BRANCH. This used to return a hard-coded Gierer-Meinhardt
        # activator-inhibitor model -- species U and V, reactions "U**2/V - U + 0.02"
        # and "U**2 - V" -- for ANY text containing one of the trigger words above,
        # DISCARDING every species the researcher named, and with an EMPTY notice.
        # Measured before this fix:
        #
        #   "Nodal and Lefty set up a spatial pattern in the zebrafish embryo."  -> U, V
        #   "Sonic hedgehog diffuses from the notochord to pattern the neural tube." -> U, V
        #   "PIP2 diffusion in the membrane is slow."                            -> U, V
        #
        # It would then form a pattern and the shipped target would pass, so the
        # researcher had a working spatial model of a system they had not described.
        # The ODE branch below was explicitly fixed to refuse rather than fabricate, and
        # the comment there argues that a fabricated result with a disappearing
        # disclaimer is worse than a refusal. This branch was still doing exactly what
        # that comment condemns, and worse, because it did not even warn.
        #
        # Now: use the species the text actually names, and say plainly that the
        # KINETICS are an assumption rather than something derived from the description.
        # Which tokens are the SPECIES? A shape heuristic alone is not enough: the
        # shipped Turing text reads "A reaction-diffusion system containing Activator U
        # and Inhibitor V", where the species are U and V while "A", "Activator" and
        # "Inhibitor" are an article and two descriptors. Picking the first two
        # capitalised tokens chose A and ACTIVATOR.
        #
        # A species is REFERRED TO REPEATEDLY, because the description goes on to say
        # what it does: U appears four times ("Activator U", "U activates", "U starts
        # at", "U diffuses"), while "Activator" appears once. So rank by frequency and
        # break ties by first appearance. A single-letter name additionally has to occur
        # at least twice, which is what keeps the English article "A" out while still
        # admitting the textbook U and V.
        counts = {}
        order = {}
        for index, token in enumerate(re.findall(r"\b[A-Za-z][A-Za-z0-9_-]*\b", text)):
            upper = token.upper()
            if not is_valid_species(upper):
                continue
            looks_named = (token.isupper() or any(c.isdigit() for c in token)
                           or (token[:1].isupper() and len(token) > 2))
            if not looks_named:
                continue
            counts[upper] = counts.get(upper, 0) + 1
            order.setdefault(upper, index)

        spatial_names = [name for name in counts
                         if len(name) > 1 or counts[name] >= 2]
        spatial_names.sort(key=lambda name: (-counts[name], order[name]))

        if len(spatial_names) < 2:
            return {
                "type": "PDE",
                "nodes": [],
                "edges": [],
                "validation_errors": [
                    "A spatial model needs at least two named species that interact.",
                ],
                "_llm_notice": (
                    "That description mentions space or diffusion, but it does not name "
                    "two species clearly enough to build a reaction-diffusion model, so "
                    "none was built. No canned model was substituted. Name the species "
                    "and how they act on each other -- for example 'A activates B and "
                    "diffuses slowly; B inhibits A and diffuses quickly' -- or load the "
                    "Turing preset for a worked example."
                ),
            }

        activator, inhibitor = spatial_names[0], spatial_names[1]
        return {
            "type": "PDE",
            "nodes": [
                {"id": activator, "name": f"{activator} (activator)", "initial_value": 1.0},
                {"id": inhibitor, "name": f"{inhibitor} (inhibitor)", "initial_value": 1.0}
            ],
            "edges": [
                {"source": activator, "target": activator, "type": "activation", "parameters": {"k": 1.0, "K_d": 1.0, "n": 2.0}},
                {"source": activator, "target": inhibitor, "type": "activation", "parameters": {"k": 1.0, "K_d": 1.0, "n": 2.0}},
                {"source": inhibitor, "target": activator, "type": "inhibition", "parameters": {"k": 1.0, "K_d": 1.0, "n": 2.0}}
            ],
            "spatial": {
                "x_grid": 50,
                "y_grid": 50,
                "dx": 1.0,
                "dy": 1.0,
                "diffusion": {
                    activator: 0.05,
                    inhibitor: 1.0
                },
                "reactions": {
                    activator: f"{activator}**2 / {inhibitor} - {activator} + 0.02",
                    inhibitor: f"{activator}**2 - {inhibitor}"
                }
            },
            "simulation_config": {
                "t_max": 200.0,
                "dt": 0.1
            },
            "_llm_notice": (
                f"This is the Gierer-Meinhardt activator-inhibitor model, with "
                f"{activator} taken as the activator and {inhibitor} as the inhibitor. "
                f"THE RATE LAWS AND DIFFUSION CONSTANTS ARE AN ASSUMPTION, not something "
                f"read from your description: the reactions are "
                f"{activator}**2/{inhibitor} - {activator} + 0.02 and "
                f"{activator}**2 - {inhibitor}, with D={activator} 0.05 and "
                f"D={inhibitor} 1.0 (the inhibitor must diffuse faster for a Turing "
                f"pattern). Any species beyond the first two were not used. Edit the "
                f"reactions and diffusion constants on the PDE-model stage to describe "
                f"your own system."
            ),
        }
    
    # Otherwise assume ODE Signaling Pathway
    # Extract entities and relations using regex
    # Split on '.' EXCEPT when it sits between two digits. The original
    # `re.split(r'[.!?\n]', text)` turned "GENB starts at 0.5." into "GENB starts at 0"
    # + "5", so EVERY decimal in the description was silently truncated to its integer
    # part -- 0.5 and 0.25 became 0.0 and 2.5 became 2.0, leaving only x.0 values
    # apparently intact. Three of the four text presets were therefore simulated at
    # initial conditions their own text does not specify.
    # A period qualifies as a sentence end when it is NOT preceded by a digit, or NOT
    # followed by one -- which keeps "0.5" whole while still ending "at 1.0." correctly.
    sentences = re.split(r'(?<!\d)\.|\.(?!\d)|[!?\n]', text)
    sentences = [s for s in sentences if s]
    nodes = {}
    edges = []
    
    # Compound sentence pattern: "X binds to Y and activates Y/it"
    # A NOMINALISATION between the verb and the species names a PROCESS, not a species.
    # "GLUCOSE stimulates release of INSULIN" used to parse as GLUCOSE -> RELEASE, with
    # INSULIN dropped entirely and no warning: a two-species model with a fabricated
    # species, both traces flat, presented as a successful parse. Skipping the
    # nominalisation is what makes the real target reachable.
    nominalisation = (
        r"(?:(?:the|a|an)\s+)?"
        r"(?:release|production|expression|synthesis|secretion|transcription|"
        r"translation|degradation|breakdown|accumulation|activation|inactivation|"
        r"phosphorylation|dephosphorylation|formation|cleavage|activity|levels?|"
        r"amount|concentration|abundance)\s+of\s+"
    )
    # A species slot: an optional nominalisation, optional article, then the name.
    sp = rf"(?:{nominalisation})?(?:(?:the|a|an)\s+)?(\w+)"

    compound_activation_patterns = [
        r"(\w+)\s+(?:binds\s+to|binds)\s+(\w+)\s+and\s+(?:activates|stimulates|triggers|induces)\s+(?:it|itself|\w+)",
    ]

    # Each entry is (pattern, reversed). `reversed` means the FIRST group is the target
    # and the second is the source, which is how passive voice reads: "ERK is
    # phosphorylated by MEK" is MEK -> ERK. Passive voice is standard in biological
    # writing and previously produced NO species at all -- the parser refused the whole
    # description, which is honest but useless.
    activation_patterns = [
        # Active voice.
        (rf"(\w+)\s+(?:activates|activate|stimulates|stimulate|triggers|trigger|"
         rf"induces|induce|phosphorylates|phosphorylate|upregulates|upregulate|"
         rf"up-regulates|drives|drive|causes|cause|produces|produce|switches\s+on|"
         rf"turns\s+on|catalyses|catalyzes|promotes|promote|increases|increase|"
         rf"enhances|enhance|amplifies|amplify|recruits|recruit|expresses|express|"
         rf"secretes|secrete|releases|release)\s+{sp}", False),
        (rf"(\w+)\s+(?:leads\s+to|results\s+in|gives\s+rise\s+to|feeds\s+into)\s+{sp}", False),
        (rf"(\w+)\s+(?:binds\s+to|binds)\s+{sp}", False),
        # Passive voice -- target first, source after "by".
        (rf"(\w+)\s+(?:is|are|gets|becomes)\s+(?:activated|phosphorylated|stimulated|"
         rf"induced|upregulated|up-regulated|switched\s+on|turned\s+on|driven|produced|"
         rf"promoted|increased|enhanced|expressed|secreted|released|recruited)\s+by\s+{sp}", True),
        # Arrow notation, which biologists write constantly and the parser ignored.
        (rf"(\w+)\s*(?:-+>|=+>|\u2192)\s*{sp}", False),
    ]
    inhibition_patterns = [
        (rf"(\w+)\s+(?:inhibits|inhibit|blocks|block|suppresses|suppress|represses|"
         rf"repress|dephosphorylates|dephosphorylate|downregulates|downregulate|"
         rf"down-regulates|degrades|degrade|sequesters|sequester|inactivates|"
         rf"inactivate|antagonises|antagonizes|decreases|decrease|reduces|reduce|"
         rf"lowers|lower|switches\s+off|turns\s+off|removes|remove|"
         rf"consumes|consume)\s+{sp}", False),
        (rf"(\w+)\s+(?:is|are|gets|becomes)\s+(?:inhibited|blocked|suppressed|"
         rf"repressed|dephosphorylated|downregulated|down-regulated|degraded|"
         rf"sequestered|inactivated|switched\s+off|turned\s+off|removed|"
         rf"consumed|antagonised|antagonized)\s+by\s+{sp}", True),
        (rf"(\w+)\s*(?:-+\|+|\u22a3)\s*{sp}", False),
    ]
    initial_patterns = [
        r"(?:initial\s+)?(\w+)\s+(?:starts\s+at|is)\s+([0-9.]+)",
        r"([0-9.]+)\s+units?\s+of\s+(\w+)"
    ]
    
    # Species whose initial value the description states outright. Anything else got a
    # placeholder, and a placeholder must never be mistaken for the researcher's intent.
    explicit_initials = set()

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        # Parse initial values
        matched_init = False
        for pattern in initial_patterns:
            matches = re.findall(pattern, sentence, re.IGNORECASE)
            for m in matches:
                matched_init = True
                # The character class [0-9.] can capture a stray leading/trailing dot
                # (e.g. "1.0." if a sentence boundary was ambiguous). Strip it rather
                # than raising ValueError out of the parser.
                raw_a, raw_b = str(m[0]).strip("."), str(m[1]).strip(".")
                if not raw_a or not raw_b:
                    continue
                try:
                    if raw_a.replace('.', '').isdigit():
                        val, name = float(raw_a), raw_b.upper()
                    else:
                        name, val = raw_a.upper(), float(raw_b)
                except ValueError:
                    continue
                if is_valid_species(name):
                    nodes[name] = {"id": name, "name": f"{name} molecule", "initial_value": val}
                    explicit_initials.add(name)
        
        # First try compound patterns (e.g. "X binds to Y and activates it")
        relation_text = _scrub(sentence)
        compound_matched = False
        for pattern in compound_activation_patterns:
            matches = re.findall(pattern, relation_text, re.IGNORECASE)
            for m in matches:
                src, tgt = m[0].upper(), m[1].upper()
                if is_valid_species(src) and is_valid_species(tgt):
                    compound_matched = True
                    if src not in nodes:
                        nodes[src] = {"id": src, "name": f"{src} protein", "initial_value": 0.0}
                    if tgt not in nodes:
                        nodes[tgt] = {"id": tgt, "name": f"{tgt} protein", "initial_value": 0.0}
                    edges.append({
                        "source": src,
                        "target": tgt,
                        "type": "activation",
                        "parameters": {"k": 0.5, "K_d": 1.0, "n": 2.0}
                    })
        
        # Skip individual activation/inhibition matching if compound already handled
        if compound_matched:
            continue
                
        # Parse activations
        for pattern, is_reversed in activation_patterns:
            for m in re.findall(pattern, relation_text, re.IGNORECASE):
                src, tgt = _orient(m, is_reversed, sentence)
                if src is None:
                    continue
                # SELF-EDGES ARE KEPT. This used to continue, so
                # "CAMKII activates CAMKII." produced ZERO edges, an empty notice, and
                # the model dCAMKII/dt = -0.1*CAMKII -- pure decay presented as
                # autocatalysis. Autoregulation is not an edge case: it is the core motif
                # of every bistable switch and most oscillators, and the repo's own
                # verify_language_to_model.py [5/5] claimed to verify self-activation
                # while passing on that decay.
                #
                # The Hill compiler already expresses it correctly with no change: for a
                # self-activation the term k*X**n/(Kd**n + X**n) is added to X's own
                # production, which IS an autocatalytic term, and for a self-inhibition
                # the factor Kd**n/(Kd**n + X**n) multiplies it, which IS autorepression.
                # Dropping the edge was discarding a capability the engine already had.
                if src not in nodes:
                    nodes[src] = {"id": src, "name": f"{src} protein", "initial_value": 0.0}
                if tgt not in nodes:
                    nodes[tgt] = {"id": tgt, "name": f"{tgt} protein", "initial_value": 0.0}
                edges.append({
                    "source": src,
                    "target": tgt,
                    "type": "activation",
                    "parameters": {"k": 0.5, "K_d": 1.0, "n": 2.0}
                })
                
        # Parse inhibitions
        for pattern, is_reversed in inhibition_patterns:
            for m in re.findall(pattern, relation_text, re.IGNORECASE):
                src, tgt = _orient(m, is_reversed, sentence)
                if src is None:
                    continue
                # SELF-EDGES ARE KEPT. This used to continue, so
                # "CAMKII activates CAMKII." produced ZERO edges, an empty notice, and
                # the model dCAMKII/dt = -0.1*CAMKII -- pure decay presented as
                # autocatalysis. Autoregulation is not an edge case: it is the core motif
                # of every bistable switch and most oscillators, and the repo's own
                # verify_language_to_model.py [5/5] claimed to verify self-activation
                # while passing on that decay.
                #
                # The Hill compiler already expresses it correctly with no change: for a
                # self-activation the term k*X**n/(Kd**n + X**n) is added to X's own
                # production, which IS an autocatalytic term, and for a self-inhibition
                # the factor Kd**n/(Kd**n + X**n) multiplies it, which IS autorepression.
                # Dropping the edge was discarding a capability the engine already had.
                if src not in nodes:
                    nodes[src] = {"id": src, "name": f"{src} protein", "initial_value": 0.0}
                if tgt not in nodes:
                    nodes[tgt] = {"id": tgt, "name": f"{tgt} protein", "initial_value": 0.0}
                edges.append({
                    "source": src,
                    "target": tgt,
                    "type": "inhibition",
                    "parameters": {"k": 0.5, "K_d": 1.0, "n": 2.0}
                })

    # No species could be identified. DO NOT substitute a canned network.
    #
    # This used to return a complete, runnable EGF/EGFR/RAS/RAF/MEK/ERK cascade with a
    # 5-second warning toast. "The weather is nice today and I like cells." produced a
    # six-species MAPK model. Once the toast expired the researcher was looking at a
    # plausible, fully simulatable network that had nothing to do with what they typed
    # -- and could compile it, run it, and export it as their own model. A fabricated
    # result with a disappearing disclaimer is worse than a refusal, because the
    # refusal is still true five seconds later.
    #
    # Returning an empty model with a validation error means compile fails, the stage
    # reports why, and the user is told what to write instead.
    if not nodes:
        return {
            "type": "ODE",
            "nodes": [],
            "edges": [],
            "simulation_config": {"t_max": 50.0},
            "validation_errors": [
                "No biological species could be identified in that description."
            ],
            "_llm_notice": (
                "No species could be identified in that description, so no model was "
                "built. Name the species and how they act on each other -- for example "
                "'EGF activates EGFR. EGFR activates ERK. ERK inhibits EGFR.' -- or use "
                "the Write-equations tab. To start from a worked example instead, click "
                "the EGF/EGFR preset."
            ),
        }

    # DRIVE THE MODEL. Every species found through a relation gets initial_value 0.0,
    # so a description that names no starting amounts -- which is most prose -- compiled
    # to a model sitting entirely at zero. Each rate law is a Hill term in its
    # regulators, and a Hill term of zero is zero, so nothing ever moved: the researcher
    # got a structurally correct cascade whose every trace was a flat line at the
    # origin, which reads as "the model does nothing" rather than "nothing started it".
    #
    # A species with no incoming edge has no upstream regulator in the description, so it
    # IS the input the researcher is describing. Giving those a unit starting amount is
    # what makes the described behaviour visible. Species that are downstream legitimately
    # start empty, and any value the description stated is never touched.
    # A species can only RISE if something activates it. So the test is not "has any
    # incoming edge" but "has an incoming ACTIVATION": a species that is only ever
    # inhibited has no production term at all, and starting it at zero makes the
    # description unrealisable -- the inhibition acts on nothing and the trace is a flat
    # line at the origin. This also covers a closed feedback loop such as
    # "GLUCOSE stimulates INSULIN. INSULIN lowers GLUCOSE.", where every species has an
    # incoming edge, nothing was ever seeded, and the whole loop sat at zero.
    if nodes and edges:
        activated = {str(edge.get("target")) for edge in edges
                     if edge.get("type") == "activation"}
        seeded = []
        for node_id, node in nodes.items():
            if node_id in activated or node_id in explicit_initials:
                continue
            if float(node.get("initial_value") or 0.0) == 0.0:
                node["initial_value"] = 1.0
                seeded.append(node_id)
        if seeded:
            _seed_note = (
                "No starting amount was given for %s, and nothing in your description "
                "activates them, so each was set to 1.0 -- a species that is only "
                "inhibited (or never regulated) cannot rise from zero, and the whole "
                "model would read as inert. Set your own values in the "
                "initial-conditions panel."
                % ", ".join(sorted(seeded))
            )
        else:
            _seed_note = ""
    else:
        _seed_note = ""

    blueprint = {
        "type": "ODE",
        "nodes": list(nodes.values()),
        "edges": edges,
        "simulation_config": {
            "t_max": 50.0
        }
    }

    # A relation that named something we refused to treat as a species leaves a HOLE in
    # the chain: the edge is absent, everything downstream of it never gets driven, and
    # those species simulate as flat lines. That looked like a working model with boring
    # dynamics. Say so instead.
    if rejected_in_relations:
        dropped = ", ".join(sorted(rejected_in_relations))
        blueprint["_llm_notice"] = (
            f"These words were read as ordinary English rather than as species and were "
            f"left out of the model: {dropped}. Any step of your description that used "
            f"them is missing, so species downstream of it may not change at all. Rename "
            f"them to something specific (for example 'EGFR' instead of 'receptor') and "
            f"compile again."
        )

    if _seed_note:
        existing = blueprint.get("_llm_notice", "")
        blueprint["_llm_notice"] = (existing + " " if existing else "") + _seed_note

    # Species named in a relation but never reachable from a driver will simulate flat.
    # Report them, because a flat trace is indistinguishable from a modelling choice.
    if nodes and edges:
        driven = {str(edge.get("target")) for edge in edges}
        drivers = {str(edge.get("source")) for edge in edges}
        orphans = sorted(set(nodes) - driven - drivers)
        if orphans:
            existing = blueprint.get("_llm_notice", "")
            blueprint["_llm_notice"] = (existing + " " if existing else "") + (
                f"These species take part in no interaction, so they will stay at their "
                f"initial value: {', '.join(orphans)}."
            )

    return blueprint

def get_default_egfr_blueprint() -> Dict[str, Any]:
    return {
        "type": "ODE",
        "nodes": [
            {"id": "EGF", "name": "Epidermal Growth Factor", "initial_value": 10.0},
            {"id": "EGFR", "name": "EGF Receptor", "initial_value": 1.0},
            {"id": "RAS", "name": "RAS GTPase", "initial_value": 1.0},
            {"id": "RAF", "name": "RAF Kinase", "initial_value": 1.0},
            {"id": "MEK", "name": "MEK Kinase", "initial_value": 0.0},
            {"id": "ERK", "name": "ERK Kinase", "initial_value": 0.0}
        ],
        "edges": [
            {"source": "EGF", "target": "EGFR", "type": "activation", "parameters": {"k": 0.8, "K_d": 2.0, "n": 1.0}},
            {"source": "EGFR", "target": "RAS", "type": "activation", "parameters": {"k": 0.6, "K_d": 1.5, "n": 1.0}},
            {"source": "RAS", "target": "RAF", "type": "activation", "parameters": {"k": 0.7, "K_d": 1.2, "n": 1.0}},
            {"source": "RAF", "target": "MEK", "type": "activation", "parameters": {"k": 0.5, "K_d": 1.0, "n": 1.5}},
            {"source": "MEK", "target": "ERK", "type": "activation", "parameters": {"k": 0.5, "K_d": 1.0, "n": 1.5}},
            {"source": "ERK", "target": "EGFR", "type": "inhibition", "parameters": {"k": 0.8, "K_d": 0.8, "n": 2.0}}
        ],
        "simulation_config": {
            "t_max": 60.0
        }
    }


# ==========================================
# 2. CLOSED-LOOP FEEDBACK AGENT (STAGE 3)
# ==========================================

# --- Oscillation detection ------------------------------------------------
# Classic systems-biology behaviours (sustained oscillations, ...) can't be
# expressed with peak/steady-state targets, so the closed-loop optimizer could
# never drive a model toward them. These detectors let "oscillation" be both a
# UI target and an optimizer objective.

def _relative_amplitude(seg: np.ndarray) -> float:
    seg = np.asarray(seg, dtype=float)
    if seg.size == 0:
        return 0.0
    mean = abs(float(seg.mean())) + 1e-6
    return float(seg.max() - seg.min()) / mean


def sustained_oscillation_amplitude(y) -> float:
    """
    Relative amplitude that PERSISTS into the final quarter of the trajectory.
    ~0 for anything settling to a steady state; finite for a limit cycle.
    (Numerical ripple can't fake this — it's normalised by the signal level.)
    """
    y = np.asarray(y, dtype=float)
    if y.size < 8 or not np.all(np.isfinite(y)):
        return 0.0
    L = len(y)
    q3, q4 = y[L // 2:3 * L // 4], y[3 * L // 4:]
    return min(_relative_amplitude(q3), _relative_amplitude(q4))


def _prominent_peaks(seg: np.ndarray, frac: float = 0.05) -> int:
    seg = np.asarray(seg, dtype=float)
    if seg.size < 3 or not np.all(np.isfinite(seg)):
        return 0
    rng = float(seg.max() - seg.min())
    if rng < 1e-9:
        return 0
    thresh = frac * rng
    return sum(1 for i in range(1, len(seg) - 1)
               if seg[i - 1] < seg[i] > seg[i + 1]
               and (seg[i] - min(seg[i - 1], seg[i + 1])) > thresh)


def count_sustained_peaks(y) -> int:
    """Prominent local maxima in the second half (sustained oscillation)."""
    y = np.asarray(y, dtype=float)
    return _prominent_peaks(y[len(y) // 2:]) if y.size >= 6 else 0


def count_all_peaks(y) -> int:
    """Prominent local maxima over the whole trajectory (incl. damped transients)."""
    y = np.asarray(y, dtype=float)
    return _prominent_peaks(y, frac=0.03)


class TargetMetric:
    def __init__(self, target_def: Dict[str, Any]):
        self.species = target_def.get("species")
        self.metric_type = target_def.get("type")  # "peak_time", "peak_value", "decay_ratio", "steady_state"
        self.min_val = target_def.get("min")
        self.max_val = target_def.get("max")
        self.expected_val = target_def.get("value")
        self.tolerance = target_def.get("tolerance", 0.1)
        # Fold-change targets specify a distinct input and output species.
        self.input_species = target_def.get("input")
        self.output_species = target_def.get("output")

    def evaluate(self, t: List[float], y: List[float]) -> Tuple[bool, str]:
        t_arr = np.array(t)
        y_arr = np.array(y)
        
        if len(y_arr) == 0:
            return False, "No simulation data"
            
        if self.metric_type == "peak_time":
            peak_idx = np.argmax(y_arr)
            peak_t = t_arr[peak_idx]
            
            if self.min_val is not None and peak_t < self.min_val:
                return False, f"Peak time for {self.species} was too early: {peak_t:.2f} min (target: {self.min_val}-{self.max_val})"
            if self.max_val is not None and peak_t > self.max_val:
                return False, f"Peak time for {self.species} was too late: {peak_t:.2f} min (target: {self.min_val}-{self.max_val})"
            return True, f"Peak time {peak_t:.2f} min met requirements."
            
        elif self.metric_type == "peak_value":
            peak_val = np.max(y_arr)
            if self.min_val is not None and peak_val < self.min_val:
                return False, f"Peak value for {self.species} was too low: {peak_val:.2f} (target: >= {self.min_val})"
            if self.max_val is not None and peak_val > self.max_val:
                return False, f"Peak value for {self.species} was too high: {peak_val:.2f} (target: <= {self.max_val})"
            return True, f"Peak value {peak_val:.2f} met requirements."
            
        elif self.metric_type == "decay_ratio":
            # Evaluates if the signal goes down after peak. Value = final_val / peak_val
            peak_val = np.max(y_arr)
            final_val = y_arr[-1]
            if peak_val == 0:
                return False, f"No activation detected for {self.species}."
            ratio = final_val / peak_val
            if self.max_val is not None and ratio > self.max_val:
                return False, f"Signal decay for {self.species} is insufficient. Final concentration is {ratio*100:.1f}% of peak (target: < {self.max_val*100:.1f}%)"
            return True, f"Signal decay ratio {ratio:.2f} met requirements."
            
        elif self.metric_type == "steady_state":
            final_val = y_arr[-1]
            if self.expected_val is not None:
                diff = abs(final_val - self.expected_val)
                if diff > self.tolerance:
                    return False, f"Steady state of {self.species} was {final_val:.2f} (target: {self.expected_val} +/- {self.tolerance})"
            return True, f"Steady state {final_val:.2f} met requirements."

        elif self.metric_type == "oscillation":
            amp = sustained_oscillation_amplitude(y_arr)
            peaks = count_sustained_peaks(y_arr)
            min_amp = self.min_val if self.min_val is not None else 0.2
            if amp >= min_amp and peaks >= 2:
                return True, f"Sustained oscillation in {self.species}: {peaks} peaks, relative amplitude {amp:.2f} (>= {min_amp})."
            return False, (f"{self.species} is not oscillating: {peaks} sustained peaks, "
                           f"relative amplitude {amp:.2f} (target: sustained, amplitude >= {min_amp}).")

        return False, f"Target type '{self.metric_type}' is not a known metric, so it was not evaluated."

# ==========================================
# 2a. SPATIAL (PDE) TARGET EVALUATION
# ==========================================
# A PDE blueprint's solution is a FIELD u(x, y, t) on a grid, not one trajectory.
# Grading it through ODEModel is not a coarse approximation of the PDE, it is a
# DIFFERENT MODEL: ODEModel reads only nodes/edges/odes and never looks at
# blueprint["spatial"], so for the Turing preset it integrated a generic Hill
# influence graph with no diffusion and reported ITS final value (0.46) as the
# "steady state" of a field whose real final frame spans 0.028 to 2.89 with a
# spatial mean of 0.59. The number graded was not the PDE solution at all.
#
# So every scalar target needs an EXPLICIT, documented reduction from the field to
# one number, and any target type with no unambiguous reduction is REFUSED BY NAME
# rather than graded against a proxy. Reporting a proxy under the researcher's
# target label is worse than refusing: it looks like evidence.

# metric type -> exactly what number is measured on the field.
SPATIAL_TARGET_REDUCTIONS: Dict[str, str] = {
    # The defensible default for "the steady state of U" on a field: the bulk
    # level, which is what conservation constrains and which settles even when the
    # pattern itself stays strongly heterogeneous.
    "steady_state": "spatial mean over the grid at the final frame",
    # The largest concentration reached anywhere in the domain at any time - one
    # well-defined number, and the usual reading of "peak value" for a field.
    "peak_value": "maximum over the whole grid and all frames",
    # Decay of the BULK level. Named explicitly in the report because a Turing
    # pattern can hold a flat bulk level while local peaks keep sharpening.
    "decay_ratio": "final spatial mean divided by the largest spatial mean reached",
    # Genuinely spatial: how much structure the final field carries. This is the
    # honest target for "did a pattern form?", which a mean cannot express.
    "spatial_variance": "spatial standard deviation over the grid at the final frame",
}

# Target types with no single defensible field-level number.
SPATIAL_TARGET_REFUSALS: Dict[str, str] = {
    "peak_time": ("every grid point peaks at a different time, and in a "
                  "symmetry-breaking pattern which point peaks first is set by the "
                  "initial noise, so a field has no single peak time. Use a "
                  "'steady_state', 'peak_value' or 'spatial_variance' target, or "
                  "name one probe location to grade a trajectory at."),
    "oscillation": ("a field can oscillate locally while its bulk level stays flat "
                    "(a travelling wave), and can hold a static pattern while single "
                    "points look noisy, so an amplitude measured on the spatial mean "
                    "is not the same claim as 'this system oscillates'."),
    "bistability": ("this needs the same model re-run from a low and a high start; "
                    "for a field those two branches are whole initial-condition "
                    "fields, which this evaluator does not construct."),
    "fold_change": ("this needs the same model re-run at two input levels, which for "
                    "a field means two whole stimulus fields; this evaluator does "
                    "not construct them."),
}


class SpatialTargetError(ValueError):
    """A target that cannot be honestly evaluated against a spatial field."""


def pde_initial_conditions(blueprint: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Initial-condition spec for a PDE blueprint.

    Shared by /api/simulate and the target evaluator ON PURPOSE: if each built its
    own ICs, the evaluator would grade a field the researcher never saw. The small
    random perturbation is what breaks symmetry - started exactly on its homogeneous
    fixed point, a Turing model forms no pattern at all.
    """
    ics: Dict[str, Dict[str, Any]] = {}
    for node in blueprint.get("nodes", []) or []:
        nid = (node or {}).get("id")
        if not nid:
            continue
        try:
            base = float(node.get("initial_value", 1.0))
        except (TypeError, ValueError):
            base = 1.0
        ics[nid] = {"type": "random_noise", "base_value": base, "noise_amplitude": 0.05}
    return ics


def simulate_pde_blueprint(blueprint: Dict[str, Any], seed: int = 0,
                           save_every: int = 5) -> Dict[str, Any]:
    """Solve a PDE blueprint's reaction-diffusion system, reproducibly.

    solve_pde draws its symmetry-breaking noise from NumPy's legacy global RNG, so
    the seed is set here and the previous global state restored afterwards. Without
    that, evaluating the same blueprint twice grades two different fields and the
    reported numbers cannot be reproduced.
    """
    from simulation_engine import solve_pde

    spatial = blueprint.get("spatial") or {}
    reactions = spatial.get("reactions") or {}
    if not reactions:
        raise SpatialTargetError(
            "This PDE blueprint declares no spatial reactions, so it has no field to grade."
        )
    config = blueprint.get("simulation_config") or {}
    try:
        t_max = float(config.get("t_max", 100.0))
        dt = float(config.get("dt", 0.1))
    except (TypeError, ValueError):
        raise SpatialTargetError("PDE t_max and dt must be numbers.")

    rng_state = np.random.get_state()
    try:
        np.random.seed(int(seed))
        return solve_pde(
            spatial_config=spatial,
            reaction_formulas=reactions,
            initial_conditions=pde_initial_conditions(blueprint),
            t_max=t_max,
            dt=dt,
            save_every=save_every,
        )
    finally:
        np.random.set_state(rng_state)


def spatial_field_series(pde_result: Dict[str, Any], species: Optional[str]) -> np.ndarray:
    """A species' field as a (frames, Nx, Ny) array. Raises if it has no field."""
    fields = pde_result.get("species") or {}
    frames = fields.get(species) if species else None
    if not frames:
        available = ", ".join(sorted(fields)) or "none"
        raise SpatialTargetError(
            f"The PDE solution contains no field for species '{species}' "
            f"(species solved: {available}), so no target on it can be graded."
        )
    field = np.asarray(frames, dtype=float)
    if field.ndim == 2:                     # 1-D domain: (frames, Nx)
        field = field[:, :, None]
    if field.ndim != 3 or field.size == 0:
        raise SpatialTargetError(
            f"The solution returned for '{species}' is not a grid time series "
            f"(array shape {tuple(field.shape)})."
        )
    return field


def reduce_spatial_field(field: np.ndarray, metric_type: Optional[str],
                         species: Optional[str]) -> Tuple[float, str]:
    """Reduce a field time series to the ONE number a scalar target grades.

    Returns (value, description of what was measured). Raises SpatialTargetError for
    any target type with no unambiguous field-level meaning.
    """
    if metric_type in SPATIAL_TARGET_REFUSALS:
        raise SpatialTargetError(
            f"A '{metric_type}' target on '{species}' cannot be evaluated against a "
            f"spatial field: {SPATIAL_TARGET_REFUSALS[metric_type]} "
            f"It was NOT graded, because grading a substitute number under this "
            f"target's name would misreport what was tested."
        )
    if metric_type not in SPATIAL_TARGET_REDUCTIONS:
        raise SpatialTargetError(
            f"Target type '{metric_type}' on '{species}' has no defined meaning for a "
            f"spatial field. Field targets supported: "
            f"{', '.join(sorted(SPATIAL_TARGET_REDUCTIONS))}."
        )
    if not np.all(np.isfinite(field)):
        raise SpatialTargetError(
            f"The '{species}' field contains non-finite values - the PDE run diverged, "
            f"so no target can be graded on it."
        )

    frame_means = field.reshape(field.shape[0], -1).mean(axis=1)
    description = SPATIAL_TARGET_REDUCTIONS[metric_type]
    if metric_type == "steady_state":
        return float(frame_means[-1]), description
    if metric_type == "peak_value":
        return float(field.max()), description
    if metric_type == "spatial_variance":
        return float(field[-1].std()), description
    peak = float(frame_means.max())          # decay_ratio
    if peak <= 1e-12:
        raise SpatialTargetError(
            f"The '{species}' field never rises above zero, so a decay ratio is undefined."
        )
    return float(frame_means[-1]) / peak, description


def target_falsifiability_warning(target: Dict[str, Any]) -> Optional[str]:
    """Warn when a target's acceptance window is so wide that it cannot fail.

    A test that accepts every plausible value is not a test. This is REPORTED, never
    silently corrected: the shipped Turing preset asks for steady_state 1.0 with
    tolerance 10.0, which accepts anything in [-9, 11]. Changing the researcher's
    tolerance behind their back would hide the problem; naming it does not.
    """
    tgt = target or {}
    metric_type = tgt.get("type")
    species = tgt.get("species") or "this species"
    lo, hi, expected = tgt.get("min"), tgt.get("max"), tgt.get("value")

    if expected is not None:
        try:
            value = float(expected)
            tol = float(tgt.get("tolerance", 0.1))
        except (TypeError, ValueError):
            return None
        if tol >= max(abs(value), 1e-12):
            return (f"The '{metric_type}' target on {species} is UNFALSIFIABLE: tolerance "
                    f"{tol:g} is at least as large as the target value {value:g}, so it "
                    f"accepts anything in [{value - tol:g}, {value + tol:g}] - including "
                    f"zero and the opposite sign. It cannot fail, so it tests nothing. "
                    f"Tighten the tolerance to make this target a real test.")
        return None

    if metric_type == "decay_ratio" and hi is not None:
        try:
            cap = float(hi)
        except (TypeError, ValueError):
            return None
        if cap >= 1.0:
            return (f"The 'decay_ratio' target on {species} is UNFALSIFIABLE: a final "
                    f"value can never exceed the peak it is divided by, so a maximum of "
                    f"{cap:g} (>= 1) is satisfied by every possible run.")
        return None

    if metric_type in ("peak_time", "peak_value", "decay_ratio", "spatial_variance") \
            and lo is None and hi is None:
        return (f"The '{metric_type}' target on {species} is UNFALSIFIABLE: it sets "
                f"neither a min nor a max, so every value passes.")
    return None


def _grade_scalar_against_target(metric: "TargetMetric", value: float) -> Tuple[bool, str]:
    """Compare ONE measured number against a target's window (value+tolerance, or min/max)."""
    if metric.expected_val is not None:
        expected = float(metric.expected_val)
        tol = float(metric.tolerance if metric.tolerance is not None else 0.1)
        window = f"{expected:g} +/- {tol:g}"
        if abs(value - expected) > tol:
            return False, f"Measured {value:.4g}, outside the target {window}."
        return True, f"Measured {value:.4g}, within the target {window}."

    lo = None if metric.min_val is None else float(metric.min_val)
    hi = None if metric.max_val is None else float(metric.max_val)
    if lo is not None and value < lo:
        return False, f"Measured {value:.4g}, below the target minimum {lo:g}."
    if hi is not None and value > hi:
        return False, f"Measured {value:.4g}, above the target maximum {hi:g}."
    if lo is None and hi is None:
        return True, (f"Measured {value:.4g}, but the target sets no bound "
                      f"(no value, min or max), so nothing was actually tested.")
    bounds = " and ".join(part for part in
                          (f">= {lo:g}" if lo is not None else "",
                           f"<= {hi:g}" if hi is not None else "") if part)
    return True, f"Measured {value:.4g}, within the target ({bounds})."


def evaluate_targets_on_pde_blueprint(blueprint: Dict[str, Any],
                                      targets: List[Dict[str, Any]],
                                      seed: int = 0):
    """Grade targets against the ACTUAL spatial solution of a PDE blueprint.

    Returns (met_count, results). A refused target is never counted as met: a target
    that could not be evaluated has not been satisfied.
    """
    targets = list(targets or [])
    results: List[Dict[str, Any]] = []

    try:
        pde = simulate_pde_blueprint(blueprint, seed=seed)
    except SpatialTargetError as e:
        detail = str(e)
    except Exception as e:
        detail = (f"The PDE solution could not be computed, so no target was graded: "
                  f"{type(e).__name__}: {e}")
    else:
        detail = None

    if detail is not None:
        for tgt in targets:
            results.append({
                "species": (tgt or {}).get("species"),
                "type": (tgt or {}).get("type"),
                "met": False, "refused": True, "spatial": True, "detail": detail,
                "warning": target_falsifiability_warning(tgt),
            })
        return 0, results

    grid = f"{pde.get('x_size', '?')}x{pde.get('y_size', '?')}"
    n_frames = len(pde.get("t") or [])
    met_count = 0
    for tgt in targets:
        tgt = tgt or {}
        metric_type, species = tgt.get("type"), tgt.get("species")
        entry: Dict[str, Any] = {
            "species": species, "type": metric_type, "spatial": True,
            "warning": target_falsifiability_warning(tgt),
        }
        try:
            field = spatial_field_series(pde, species)
            value, description = reduce_spatial_field(field, metric_type, species)
        except SpatialTargetError as e:
            entry.update({"met": False, "refused": True, "detail": str(e)})
            results.append(entry)
            continue

        ok, verdict = _grade_scalar_against_target(TargetMetric(tgt), value)
        final = field[-1]
        entry.update({
            "met": bool(ok), "refused": False,
            "observable": description, "value": value,
            "detail": (f"{verdict} Graded on the {description} of {species} = {value:.4g}, "
                       f"from the {grid} PDE field over {n_frames} saved frames "
                       f"(that final field spans {float(final.min()):.4g} to "
                       f"{float(final.max()):.4g}, spatial std {float(final.std()):.4g})."),
        })
        results.append(entry)
        if ok:
            met_count += 1
    return met_count, results

# ==========================================
# 2b. NUMERICAL TARGET OPTIMIZER
# ==========================================
# The closed-loop feedback used to nudge parameters with fixed heuristic
# multipliers (and did nothing for steady-state targets), which rarely converged.
# This replaces that with an actual optimizer that fits the tunable parameters to
# satisfy the target constraints.

def _param_bounds(pname: str) -> Tuple[float, float]:
    if pname.startswith("deg_"):
        return (0.001, 3.0)
    if pname.startswith("kdrv_"):
        # Decay rate of an injected transient drive: sets WHEN the pulse peaks.
        # Must reach fast rates so an early (few-minute) peak is reachable.
        return (0.0, 3.0)
    if pname.startswith("syn_") or pname.startswith("drv_"):
        return (0.0, 5.0)
    if pname.endswith("_Kd"):
        return (0.05, 10.0)
    if pname.endswith("_n"):
        # Hill coefficient — steep nonlinearity is what enables oscillation/bistability.
        return (1.0, 8.0)
    if pname.endswith("_k"):
        # Activation strength — a wide range is needed to reach oscillatory regimes.
        return (0.0, 12.0)
    return (0.0, 12.0)


def _interior_bounds(lo, hi, frac: float = 0.06):
    """Shrink an acceptance window slightly for the OPTIMIZER's loss only.

    Without this the loss reaches exactly 0 on the boundary, so the search happily
    settles at e.g. peak_value 1.01 against a "<= 1.0" target: the loss says solved
    while the boolean checker says failed, and the loop stalls reporting "so close".
    Aiming at the interior means loss 0 implies comfortably met. The user-facing
    pass/fail test (TargetMetric.evaluate) is untouched — this only steers search."""
    if lo is not None and hi is not None and hi > lo:
        m = frac * (hi - lo)
        return lo + m, hi - m
    if lo is not None:
        return lo + frac * max(abs(lo), 1.0), hi
    if hi is not None:
        return lo, hi - frac * max(abs(hi), 1.0)
    return lo, hi


def _target_violation(metric: "TargetMetric", t: np.ndarray, y: np.ndarray) -> float:
    """Continuous, non-negative distance from meeting a single target (0 = met)."""
    if y.size == 0:
        return 10.0
    if not np.all(np.isfinite(y)):
        return 1e3
    mt = metric.metric_type
    if mt == "peak_time":
        pt = float(t[int(np.argmax(y))])
        lo, hi = _interior_bounds(metric.min_val, metric.max_val)
        v = 0.0
        if lo is not None and pt < lo:
            v += (lo - pt) / max(abs(lo), 1.0)
        if hi is not None and pt > hi:
            v += (pt - hi) / max(abs(hi), 1.0)
        return v
    if mt == "peak_value":
        pv = float(np.max(y))
        lo, hi = _interior_bounds(metric.min_val, metric.max_val)
        v = 0.0
        if lo is not None and pv < lo:
            v += (lo - pv) / max(abs(lo), 1.0)
        if hi is not None and pv > hi:
            v += (pv - hi) / max(abs(hi), 1.0)
        return v
    if mt == "decay_ratio":
        pv = float(np.max(y))
        fv = float(y[-1])
        if pv <= 1e-9:
            return 1.0
        cap = metric.max_val if metric.max_val is not None else 1.0
        cap_eff = cap * 0.9                      # aim below the cap, not exactly at it
        return max(0.0, (fv / pv) - cap_eff) / max(cap, 0.05)
    if mt == "steady_state":
        fv = float(y[-1])
        val = metric.expected_val if metric.expected_val is not None else 0.0
        tol = (metric.tolerance or 0.1) * 0.8    # land inside the tolerance band
        return max(0.0, abs(fv - val) - tol) / max(abs(val), 1.0)
    if mt == "oscillation":
        sustained = sustained_oscillation_amplitude(y)
        peaks_late = count_sustained_peaks(y)
        min_amp = metric.min_val if metric.min_val is not None else 0.2
        if sustained >= min_amp and peaks_late >= 2:
            return 0.0
        # A genuine limit cycle needs BOTH sufficient sustained amplitude AND
        # multiple periods in the window. Penalise each separately: a high-amplitude
        # SLOW transient (big swing, 0 repeating peaks) must not read as "met".
        amp_gap = max(0.0, min_amp - sustained) / max(min_amp, 0.05)
        peak_gap = max(0, 2 - peaks_late) * 0.5
        return amp_gap + peak_gap
    return 0.0


def _total_target_loss(sim_results: Dict[str, Any], targets: List[Dict[str, Any]]) -> float:
    t = np.array(sim_results.get("t", []))
    species = sim_results.get("species", {})
    if t.size == 0:
        return 1e6
    total = 0.0
    for tgt in targets:
        m = TargetMetric(tgt)
        total += _target_violation(m, t, np.array(species.get(m.species, [])))
    return total


def _all_targets_met(sim_results: Dict[str, Any], targets: List[Dict[str, Any]]) -> bool:
    t = sim_results.get("t", [])
    species = sim_results.get("species", {})
    for tgt in targets:
        m = TargetMetric(tgt)
        ok, _ = m.evaluate(t, species.get(m.species, []))
        if not ok:
            return False
    return True


# ==========================================
# 2c. MULTI-CONDITION BEHAVIOURS (bistability, fold-change)
# ==========================================
# These can't be judged from one trajectory: bistability needs runs from a LOW
# and a HIGH start; fold-change needs runs at two input levels. They take the
# compiled model + a parameter vector and run the extra simulations themselves.

# Most parameters a large compiled model exposes do not measurably move the targeted
# read-out, and every extra dimension multiplies the global search's cost per
# generation. Above this many tunable parameters, screen and search only the ones
# that matter.
MAX_SEARCH_DIMS = 12

SINGLE_TRAJECTORY_TYPES = {
    "peak_time", "peak_value", "decay_ratio", "steady_state", "oscillation",
}


def _final_stable(y: np.ndarray) -> bool:
    """Has the trajectory actually SETTLED, or is it still moving slowly?

    The spread of the tail alone is not enough. A slowly decaying transient has a
    small spread and a CONSISTENT direction, and the old test accepted it: with
    y[-1]=9.05 the tolerance was 0.02*9.05+0.01 = 0.191, which passed an observed tail
    drift of 0.136 that was monotone decreasing, not noise. That is how a model whose
    high branch decays 9.05 -> 0.0026 between t=100 and t=10000 was reported as having
    a second stable state.
    """
    y = np.asarray(y, dtype=float)
    if y.size < 10 or not np.all(np.isfinite(y)):
        return False
    tail = y[int(0.85 * len(y)):]
    if tail.size < 3:
        return False
    scale = max(abs(float(y[-1])), 1.0)
    spread = float(tail.max() - tail.min())
    if spread > 0.02 * scale + 0.01:
        return False
    # Direction test: a fixed point's tail wanders within noise; a transient's tail
    # moves one way. Reject a monotone tail whose NET change is not negligible.
    diffs = np.diff(tail)
    monotone = bool(np.all(diffs <= 0.0)) or bool(np.all(diffs >= 0.0))
    net = abs(float(tail[-1] - tail[0]))
    if monotone and net > 1e-3 * scale:
        return False
    return True


def bistability_states(model: "ODEModel", species: str, t_max: float,
                       cp: Optional[Dict[str, float]] = None,
                       high_level: float = 10.0,
                       confirm_factor: float = 10.0):
    """Settle from a LOW (0) and a HIGH start; return (low_final, high_final, both_stable).

    Two attractors are confirmed at a LONGER horizon, not just at ``t_max``. Sampling
    only t_max cannot distinguish a second stable state from a transient that has not
    finished relaxing, and a preset whose horizon is shorter than its relaxation time
    therefore reported bistability it does not have. ``confirm_factor`` re-runs the
    high branch for 10x longer and requires the separation to SURVIVE.
    """
    lo = model.simulate(t_max, num_points=150, custom_params=cp, custom_initial={species: 0.0})
    hi = model.simulate(t_max, num_points=150, custom_params=cp, custom_initial={species: high_level})
    ylo, yhi = lo["species"].get(species, []), hi["species"].get(species, [])
    if not ylo or not yhi:
        return 0.0, 0.0, False
    stable = _final_stable(np.array(ylo)) and _final_stable(np.array(yhi))

    if stable and confirm_factor and confirm_factor > 1.0:
        # The decisive check: does the gap persist when both are given far longer?
        near = abs(float(yhi[-1]) - float(ylo[-1]))
        try:
            long_t = float(t_max) * float(confirm_factor)
            lo2 = model.simulate(long_t, num_points=200, custom_params=cp,
                                 custom_initial={species: 0.0})
            hi2 = model.simulate(long_t, num_points=200, custom_params=cp,
                                 custom_initial={species: high_level})
            ylo2 = lo2["species"].get(species, [])
            yhi2 = hi2["species"].get(species, [])
            if ylo2 and yhi2:
                far = abs(float(yhi2[-1]) - float(ylo2[-1]))
                # Collapsing to under a fifth of the short-horizon gap means the "high
                # state" was a decaying transient converging on the low one.
                if near > 0 and far < 0.2 * near:
                    stable = False
        except Exception:
            # A failed confirmation must not upgrade a doubtful result to confident.
            stable = False
    return float(ylo[-1]), float(yhi[-1]), stable


def bistability_separation(model, species, t_max, cp=None) -> float:
    lo, hi, stable = bistability_states(model, species, t_max, cp)
    if not stable:
        return 0.0
    return abs(hi - lo) / max(abs(hi), abs(lo), 1.0)


def bistability_violation(model, species, t_max, cp=None, min_sep=0.5) -> float:
    sep = bistability_separation(model, species, t_max, cp)
    return 0.0 if sep >= min_sep else (min_sep - sep) / max(min_sep, 0.05)


def fold_change_response(model, input_species, output_species, t_max, cp=None,
                         base_level=1.0, fold=3.0):
    """
    Response of the output to a `fold`-change of the input, measured at two
    absolute input levels (base and 4x base). Returns (resp1, resp2) where each
    is the relative overshoot of the output above its own pre-stimulus baseline.
    """
    def response(input_start):
        # Pre-stimulus: input at input_start. Stimulus: multiply input by `fold`.
        stim = model.simulate(t_max, num_points=150, custom_params=cp,
                               custom_initial={input_species: input_start * fold})
        y = np.asarray(stim["species"].get(output_species, []), dtype=float)
        if y.size == 0:
            return 0.0
        base = float(y[0]) if abs(y[0]) > 1e-6 else 1e-6
        return (float(y.max()) - base) / abs(base)   # relative peak response

    return response(base_level), response(base_level * 4.0)


def fold_change_violation(model, target, t_max, cp=None) -> float:
    inp = target.get("input") or target.get("species")
    out = target.get("output") or target.get("species")
    tol = target.get("max", 0.15)
    try:
        r1, r2 = fold_change_response(model, inp, out, t_max, cp)
    except Exception:
        return 2.0
    if max(abs(r1), abs(r2)) < 0.05:
        return 1.0   # no response at all -> not fold-change detection
    rel_diff = abs(r1 - r2) / (max(abs(r1), abs(r2)) + 1e-6)
    return 0.0 if rel_diff <= tol else (rel_diff - tol)


def _remap_flat_targets(model: "ODEModel", targets: List[Dict[str, Any]], t_max: float,
                        cp: Optional[Dict[str, float]] = None):
    """A peak/steady/decay target on a species that stays constant cannot be met by
    tuning. This happens when an LLM model splits a protein into inactive + active forms
    (ERK vs ERK_act): the target names ERK, but the read-out that actually moves is
    ERK_act. Retarget to the dynamic active form when one clearly exists.
    Returns (adjusted_targets, notes)."""
    try:
        res = model.simulate(min(max(t_max, 1.0), 200.0), num_points=120, custom_params=cp)
    except Exception:
        return targets, []
    species = res.get("species", {})
    node_ids = list(getattr(model, "node_ids", []))

    def is_flat(sp):
        a = np.asarray(species.get(sp, []), dtype=float)
        if a.size == 0 or not np.all(np.isfinite(a)):
            return True
        return float(a.max() - a.min()) < 1e-3 * (abs(float(a.max())) + 1.0)

    def peaks_early(sp):  # peak is at the very start -> only decreases (inactive form)
        a = np.asarray(species.get(sp, []), dtype=float)
        return a.size >= 3 and int(np.argmax(a)) <= 1

    def active_form(sp):
        # "<name>a" / "<name>A" (ERK -> ERKa, RAS -> RASa) is the convention the
        # compiler most often produces for an activated form, alongside the
        # _act/_p/_phos family. Missing it left targets pointing at the INACTIVE
        # pool, which only decays from its initial value, so "peak between 5 and 15
        # min" could never be satisfied no matter how the rates were tuned.
        variants = [sp + "a", sp + "A", sp + "_a", sp + "_act", sp + "_active",
                    "p" + sp, sp + "_p", sp + "p", sp + "_phos", "active_" + sp,
                    sp + "_star", sp + "_on",
                    # doubly-phosphorylated forms, standard in MAPK-cascade models
                    "pp" + sp, sp + "_pp", sp + "pp", "active" + sp]
        for c in node_ids:
            if c != sp and c in variants and not is_flat(c):
                return c
        # looser: a dynamic species whose name is sp plus an activation-ish suffix
        for c in node_ids:
            if c == sp or is_flat(c):
                continue
            if re.match(r"^" + re.escape(sp) + r"[_ ]?(a|act|active|p|pp|phos|star|on)$", c, re.IGNORECASE):
                return c
        return None

    adjusted, notes = [], []
    for t in targets:
        t = dict(t or {})
        sp, ty = t.get("species"), t.get("type")
        # A constant species can never meet a peak/steady target. A species whose
        # maximum is its INITIAL value (an inactive pool that only drains, e.g. ERK
        # when the model also has ERKa) is just as wrong for any peak target: its
        # peak_time is always 0 and its peak_value merely reports the initial
        # condition, so such a target "passes" without measuring a response at all.
        # In every case retarget to the active form when a clearly-named one exists.
        needs_active = sp and (is_flat(sp)
                               or (ty in ("peak_time", "peak_value") and peaks_early(sp)))
        if ty in SINGLE_TRAJECTORY_TYPES and needs_active:
            alt = active_form(sp)
            if alt:
                notes.append(f"'{sp}' does not vary the right way here; retargeting to its active form '{alt}'.")
                t["species"] = alt
        adjusted.append(t)
    return adjusted, notes


def evaluate_targets_on_blueprint(blueprint: Dict[str, Any],
                                  targets: List[Dict[str, Any]],
                                  t_max: Optional[float] = None,
                                  custom_params: Optional[Dict[str, float]] = None):
    """
    Evaluate ALL targets on a blueprint, running multi-condition simulations for
    bistability/fold-change. Returns (met_count, results) where results is a list
    of {species, type, met, detail} plus, where applicable, a falsifiability
    `warning`, a `refused` flag, and for spatial models the `observable` that was
    measured and its `value`.
    """
    # A PDE blueprint's solution is a spatial field, and ODEModel cannot see
    # blueprint["spatial"] at all - it would integrate the nodes/edges influence
    # graph instead (no diffusion, no spatial reaction terms) and report that
    # unrelated model's final value under the researcher's target name.
    if str((blueprint or {}).get("type", "ODE")).upper() == "PDE":
        return evaluate_targets_on_pde_blueprint(blueprint, targets)

    try:
        model = ODEModel(blueprint)
    except Exception as e:
        return 0, [{"met": False, "refused": True,
                    "detail": f"Could not compile model: {e}"} for _ in targets]
    if t_max is None:
        t_max = float(blueprint.get("simulation_config", {}).get("t_max", 50.0))
    cp = custom_params or None

    targets, _ = _remap_flat_targets(model, targets, t_max, cp)  # inactive-form -> active-form
    single = [t for t in targets if (t or {}).get("type") in SINGLE_TRAJECTORY_TYPES]
    base_res = model.simulate(t_max, num_points=200, custom_params=cp) if single else None

    results, met_count = [], 0
    for tgt in targets:
        ty = (tgt or {}).get("type")
        sp = (tgt or {}).get("species")
        refused = False
        if ty in SINGLE_TRAJECTORY_TYPES:
            m = TargetMetric(tgt)
            ok, detail = m.evaluate(base_res["t"], base_res["species"].get(sp, []))
        elif ty == "bistability":
            lo, hi, stable = bistability_states(model, sp, t_max, cp)
            sep = (abs(hi - lo) / max(abs(hi), abs(lo), 1.0)) if stable else 0.0
            min_sep = tgt.get("min", 0.5)
            ok = stable and sep >= min_sep
            detail = (f"Bistable: low start -> {lo:.2f}, high start -> {hi:.2f} "
                      f"(separation {sep:.2f} >= {min_sep})." if ok else
                      f"Not bistable: low->{lo:.2f}, high->{hi:.2f}, separation {sep:.2f} "
                      f"(target >= {min_sep}{'' if stable else ', not settled'}).")
        elif ty == "fold_change":
            inp = tgt.get("input") or sp
            out = tgt.get("output") or sp
            tol = tgt.get("max", 0.15)
            try:
                r1, r2 = fold_change_response(model, inp, out, t_max)
                if max(abs(r1), abs(r2)) < 0.05:
                    ok, detail = False, f"No {out} response to the {inp} step."
                else:
                    rel = abs(r1 - r2) / (max(abs(r1), abs(r2)) + 1e-6)
                    ok = rel <= tol
                    detail = (f"Fold-change response {out}: {r1:.2f} vs {r2:.2f} at 4x input "
                              f"(mismatch {rel*100:.0f}% <= {tol*100:.0f}%)." if ok else
                              f"Response depends on absolute level: {r1:.2f} vs {r2:.2f} "
                              f"(mismatch {rel*100:.0f}%, target <= {tol*100:.0f}%).")
            except Exception as e:
                ok, detail = False, f"Fold-change test failed: {e}"
        else:
            # An unrecognised target type used to return met=True ("Unknown metric"),
            # which quietly credited a target nothing had measured.
            ok, refused = False, True
            detail = (f"Target type '{ty}' on '{sp}' is not a metric this evaluator knows, "
                      f"so it was NOT evaluated and cannot count as met.")
        results.append({"species": sp, "type": ty, "met": bool(ok), "refused": refused,
                        "detail": detail,
                        "warning": target_falsifiability_warning(tgt)})
        if ok:
            met_count += 1
    return met_count, results


def _apply_params_to_blueprint(blueprint: Dict[str, Any], tunable: List[str],
                               values: List[float]) -> Dict[str, Any]:
    """Write optimized parameter values back onto the blueprint so they persist.

    Two model modes are handled:
      * custom-kinetics (blueprint has 'odes'): every rate constant lives in the
        'parameters' dict, which is exactly what ODEModel reads. Writing tuned
        values there is what makes the closed-loop actually change the model. This
        is REQUIRED — without it the optimizer's solution is silently discarded and
        the refinement loop stalls (targets never met even though a fit was found).
      * generic (edges) model: parameter names map onto per-node degradation/
        synthesis and per-edge k/K_d/n.
    """
    bp = json.loads(json.dumps(blueprint))
    vals = {p: round(float(v), 5) for p, v in zip(tunable, values)}

    # --- custom-kinetics: persist into the parameters dict the compiler reads ---
    if bp.get("odes"):
        params = bp.get("parameters")
        if not isinstance(params, dict):
            params = {}
            bp["parameters"] = params
        for p, val in vals.items():
            params[p] = val
        return bp

    # --- generic (edges) model: map names onto nodes/edges -----------------------
    node_by_id = {n["id"]: n for n in bp.get("nodes", [])}

    def find_edge(src, tgt, etype):
        for e in bp.get("edges", []):
            if (e.get("source") == src and e.get("target") == tgt
                    and e.get("type", "activation") == etype):
                return e
        return None

    for p, val in vals.items():
        if p.startswith("deg_"):
            n = node_by_id.get(p[4:])
            if n is not None:
                n["degradation"] = val
        elif p.startswith("syn_"):
            n = node_by_id.get(p[4:])
            if n is not None:
                n["synthesis"] = val
        else:
            m = re.match(r"^(act|inh)_(.+)_to_(.+)_(k|Kd|n)$", p)
            if m:
                kind, src, tgt, suf = m.groups()
                etype = "activation" if kind == "act" else "inhibition"
                e = find_edge(src, tgt, etype)
                if e is not None:
                    e.setdefault("parameters", {})
                    key = {"k": "k", "Kd": "K_d", "n": "n"}[suf]
                    e["parameters"][key] = val
    return bp


def _maybe_add_feedback(blueprint: Dict[str, Any],
                        failed_targets: List[Tuple["TargetMetric", str]],
                        logs: List[str]) -> Dict[str, Any]:
    """
    Add the topology a target needs but the model lacks:
      * decay_ratio  -> a negative-feedback edge (if none exists at all)
      * oscillation  -> a DELAYED negative-feedback loop (a limit cycle needs
                        positive/production feedback + delayed inhibition)
    """
    bp = json.loads(json.dumps(blueprint))
    # Custom-kinetics models carry their dynamics in odes/fluxes, not in an edge
    # list, so these edge-based feedback tricks do not apply (the compiler would
    # ignore any edge we added). Leave the topology alone and let the numerical
    # optimizer tune the rate constants instead.
    if bp.get("odes"):
        logs.append("Custom-kinetics model: tuning rate constants numerically (no edge changes).")
        return bp
    edges = bp.setdefault("edges", [])   # tolerate a blueprint without an edge list
    node_ids = [n["id"] for n in bp.get("nodes", [])]

    def has_edge(s, t, ty):
        return any(e.get("source") == s and e.get("target") == t
                   and e.get("type") == ty for e in edges)

    # --- decay: ensure some negative feedback exists ---
    has_inhibition = any(e.get("type") == "inhibition" for e in edges)
    if any(m.metric_type == "decay_ratio" for m, _ in failed_targets) and not has_inhibition:
        for m, _ in failed_targets:
            if m.metric_type == "decay_ratio" and m.species in node_ids:
                roots = [n for n in ("EGFR", "EGF") if n in node_ids and n != m.species]
                if not roots:
                    roots = [e["source"] for e in edges if e.get("source") != m.species][:1]
                if roots:
                    edges.append({"source": m.species, "target": roots[-1], "type": "inhibition",
                                  "parameters": {"k": 1.0, "K_d": 1.0, "n": 2.0}})
                    logs.append(f"Action: Added negative-feedback edge [{m.species} --| {roots[-1]}] to enable decay.")
                    break

    # --- oscillation: build a delayed negative-feedback loop through a node X drives ---
    for m, _ in failed_targets:
        if m.metric_type != "oscillation" or m.species not in node_ids:
            continue
        x = m.species

        def feeds_back(cand):
            return any(e.get("source") == cand and e.get("target") == x for e in edges)

        # Prefer closing the loop through a node X activates that does NOT already
        # feed back to X (so we get a clean, separate delayed negative loop rather
        # than a contradictory edge on an existing pair).
        downstream = [e["target"] for e in edges
                      if e.get("source") == x and e.get("type") == "activation" and e.get("target") != x]
        y = next((c for c in downstream if not feeds_back(c)), None)
        if y is None:
            y = next((c for c in downstream if not has_edge(c, x, "inhibition")), None)
        if y is None:
            # Otherwise pick another node and wire X -> Y so the loop has a delay.
            for cand in node_ids:
                if cand != x and not has_edge(cand, x, "inhibition"):
                    y = cand
                    if not has_edge(x, y, "activation"):
                        edges.append({"source": x, "target": y, "type": "activation",
                                      "parameters": {"k": 1.0, "K_d": 1.0, "n": 2.0}})
                        logs.append(f"Action: Added activation [{x} -> {y}] to seed an oscillator loop.")
                    break
        if y is not None and not has_edge(y, x, "inhibition"):
            edges.append({"source": y, "target": x, "type": "inhibition",
                          "parameters": {"k": 1.0, "K_d": 1.0, "n": 4.0}})
            logs.append(f"Action: Added delayed negative feedback [{y} --| {x}] to enable oscillation.")

    # --- bistability: ensure positive (self-)feedback with ultrasensitivity ---
    for m, _ in failed_targets:
        if m.metric_type != "bistability" or m.species not in node_ids:
            continue
        x = m.species
        if not has_edge(x, x, "activation"):
            edges.append({"source": x, "target": x, "type": "activation",
                          "parameters": {"k": 2.0, "K_d": 1.0, "n": 4.0}})
            logs.append(f"Action: Added self-activation [{x} -> {x}] to enable a bistable switch.")

    # --- fold-change: ensure an incoherent feedforward loop (input -> buffer -| output) ---
    for m, _ in failed_targets:
        if m.metric_type != "fold_change":
            continue
        inp = getattr(m, "input_species", None) or m.species
        out = getattr(m, "output_species", None) or m.species
        if inp not in node_ids or out not in node_ids or inp == out:
            continue
        if not has_edge(inp, out, "activation"):
            edges.append({"source": inp, "target": out, "type": "activation",
                          "parameters": {"k": 1.0, "K_d": 1.0, "n": 1.0}})
        # buffer node that the input activates and that inhibits the output
        buffer = next((n for n in node_ids if n not in (inp, out)), None)
        if buffer is None:
            buffer = f"{out}_BUF"
            bp["nodes"].append({"id": buffer, "name": buffer, "initial_value": 0.0})
            node_ids.append(buffer)
        if not has_edge(inp, buffer, "activation"):
            edges.append({"source": inp, "target": buffer, "type": "activation",
                          "parameters": {"k": 1.0, "K_d": 1.0, "n": 1.0}})
        if not has_edge(buffer, out, "inhibition"):
            edges.append({"source": buffer, "target": out, "type": "inhibition",
                          "parameters": {"k": 1.0, "K_d": 1.0, "n": 1.0}})
        logs.append(f"Action: Wired an incoherent feedforward loop [{inp} -> {buffer} -| {out}] for fold-change.")

    return bp


def _inject_basal_drives(blueprint: Dict[str, Any],
                         species: List[str]) -> Tuple[Dict[str, Any], List[str]]:
    """For a custom-kinetics blueprint, add a tunable DECAYING drive term
    (+ drv_<sp>*exp(-kdrv_<sp>*t)) to each named species' ODE.

    A cascade whose every production term is multiplicatively gated on an upstream
    species that is also 0 can never leave the origin by rate-constant tuning alone,
    so it needs an injected source. The source must be TRANSIENT, not constant: a
    constant drive can only produce a monotonic rise, whose maximum is always at the
    end of the window, so a peak_time target (e.g. "peak between 5 and 15 min") is
    structurally unreachable and the optimizer stalls reporting no improvement. A
    decaying drive gives a rise-then-fall pulse whose peak time is tunable, and it
    degenerates to a constant drive as kdrv -> 0, so it is strictly more general.

    Returns (new_blueprint, injected_param_names)."""
    if not blueprint.get("odes"):
        return blueprint, []
    bp = json.loads(json.dumps(blueprint))
    odes = bp.get("odes", {})
    params = bp.setdefault("parameters", {})
    injected: List[str] = []
    for sp in species:
        if sp not in odes:
            continue
        amp, rate, rem = "drv_" + sp, "kdrv_" + sp, "krem_" + sp
        if amp in params:
            continue
        # A removal knob is added alongside the pulse: without a loss term a species
        # can only rise, so "peak at 5-15 min" and "decay to <20% of peak" stay
        # unreachable no matter how the rates are tuned. Both knobs start NEUTRAL
        # (amplitude 0, removal 0), so the injected model is numerically identical to
        # the original until the optimizer chooses to use them, and unused knobs are
        # pruned from the result.
        odes[sp] = ("(" + str(odes[sp]) + ") + " + amp + "*exp(-" + rate + "*t)"
                    + " - " + rem + "*" + sp)
        params[amp] = 0.0
        params[rate] = 0.2
        params[rem] = 0.0
        injected.extend([amp, rate, rem])
    return bp, injected


def _prune_injected_drives(blueprint: Dict[str, Any], injected: List[str],
                           threshold: float = 1e-3) -> Dict[str, Any]:
    """Remove injected drive knobs the optimizer left at ~0, so the final model stays
    clean (and the parameter table isn't cluttered with dead knobs). Drives the
    optimizer actually used are kept, together with their decay-rate partner."""
    if not injected:
        return blueprint
    params = blueprint.get("parameters", {}) or {}
    odes = blueprint.get("odes", {}) or {}
    for amp in [p for p in injected if p.startswith("drv_")]:
        if abs(float(params.get(amp, 0.0))) >= threshold:
            continue                       # this drive is doing real work: keep it
        rate = "kdrv_" + amp[len("drv_"):]
        params.pop(amp, None)
        params.pop(rate, None)
        pattern = (r"\s*\+\s*" + re.escape(amp) + r"\s*\*\s*exp\(\s*-\s*"
                   + re.escape(rate) + r"\s*\*\s*t\s*\)")
        for sp, expr in list(odes.items()):
            cleaned = re.sub(pattern, "", str(expr))
            if cleaned != expr:
                odes[sp] = cleaned
    return blueprint


def optimize_parameters_to_targets(
    blueprint: Dict[str, Any],
    targets: List[Dict[str, Any]],
    time_budget: float = 45.0,
    max_seconds: Optional[float] = None,
) -> Tuple[Dict[str, Any], bool, float, List[str]]:
    """
    Fit the model's tunable parameters (edge k/K_d and per-node degradation/synthesis;
    Hill coefficients are held fixed) so the simulation satisfies the target
    constraints. Uses a fast local search (Powell) then a global search
    (differential evolution) with early-stop when all targets are met.
    Returns (refined_blueprint, all_met, best_loss, logs).
    """
    logs: List[str] = []
    try:
        model = ODEModel(blueprint)
    except Exception as e:
        return blueprint, False, float("inf"), [f"Could not compile model for optimization: {e}"]

    t_max = float(blueprint.get("simulation_config", {}).get("t_max", 50.0))
    # If a target names a constant (inactive) species, retarget to its active form so
    # the optimizer tunes toward a read-out that actually moves.
    targets, remap_notes = _remap_flat_targets(model, targets, t_max)
    logs.extend(remap_notes)

    # Un-freeze a stuck readout: if a single-trajectory target names a species that
    # sits frozen at ~0 (a cascade that cannot leave the origin because every
    # production term is multiplicatively gated on an upstream species that is also
    # 0), no rate-constant tuning can move it. Inject a tunable basal-drive knob on
    # each frozen species so the optimizer always has something to turn; unused
    # drives settle to ~0 and are pruned from the returned model.
    injected_drives: List[str] = []
    if blueprint.get("odes"):
        try:
            res0 = model.simulate(t_max, num_points=120)
            sp0 = res0.get("species", {})

            def _frozen_zero(sp: str) -> bool:
                a = np.asarray(sp0.get(sp, []), dtype=float)
                return a.size > 0 and bool(np.all(np.isfinite(a))) and float(np.max(np.abs(a))) < 1e-3

            def _cannot_peak(sp: str) -> bool:
                """True when the species cannot satisfy a peak target as written: it is
                pinned at zero, or its maximum sits at either END of the window rather
                than inside it. Peak-at-start means an inactive pool that only drains;
                peak-at-end means it never turns over, so "peak between 5 and 15 min"
                is out of reach until the model gains a way to rise and then fall."""
                if _frozen_zero(sp):
                    return True
                a = np.asarray(sp0.get(sp, []), dtype=float)
                if a.size < 3 or not bool(np.all(np.isfinite(a))):
                    return False
                i = int(np.argmax(a))
                return i <= 1 or i >= a.size - 2

            peak_species = {(t or {}).get("species") for t in targets
                            if (t or {}).get("type") in ("peak_time", "peak_value")}
            tgt_species = {(t or {}).get("species") for t in targets
                           if (t or {}).get("type") in SINGLE_TRAJECTORY_TYPES}
            needs_lift = ([s for s in tgt_species if s and _frozen_zero(s)]
                          or [s for s in peak_species if s and _cannot_peak(s)])
            if needs_lift:
                # Prefer knobs on the species the targets actually name: in a cascade
                # the feedback carries the lift upstream anyway, and every extra knob
                # widens the search space (15 knobs took ~85s, a handful takes ~20s).
                # Fall back to every frozen species if no target species is frozen.
                frozen = [s for s in getattr(model, "node_ids", []) if s in set(needs_lift)]
                if not frozen:
                    frozen = [s for s in getattr(model, "node_ids", []) if _frozen_zero(s)]
                aug_bp, candidate_drives = _inject_basal_drives(blueprint, frozen)
                if candidate_drives:
                    # Only adopt the augmented model if it actually compiles. Doing
                    # this without a guard once left injected_drives set while the
                    # model stayed un-augmented, so the optimizer silently tuned
                    # knobs that did not exist and the loop could never converge.
                    try:
                        model = ODEModel(aug_bp)
                        blueprint = aug_bp
                        injected_drives = candidate_drives
                        logs.append(f"Readout could not peak as written; added a tunable input pulse "
                                    f"and removal term for {', '.join(frozen)} so the cascade "
                                    f"can be lifted and can peak.")
                    except Exception as e:
                        injected_drives = []
                        logs.append(f"Could not add input-pulse knobs ({type(e).__name__}); "
                                    f"tuning the existing rate constants only.")
        except Exception as e:
            logs.append(f"Could not analyse the readout for reachability ({type(e).__name__}).")

    # Tune every parameter, including Hill coefficients — steep nonlinearity is
    # essential for reaching oscillatory/bistable regimes.
    tunable = list(model.param_names)
    if not tunable:
        return blueprint, False, float("inf"), ["No tunable parameters to optimize."]
    bounds = [_param_bounds(p) for p in tunable]
    x0 = [min(max(model.params_dict[p], b[0]), b[1]) for p, b in zip(tunable, bounds)]

    # Oscillation / bistability / fold-change are harder objectives (narrow windows
    # found by global search): use a finer grid and a bigger budget for them.
    single = [t for t in targets if (t or {}).get("type") in SINGLE_TRAJECTORY_TYPES]
    multi = [t for t in targets if (t or {}).get("type") in ("bistability", "fold_change")]
    hard = any((t or {}).get("type") == "oscillation" for t in targets) or bool(multi)
    # Use the SAME resolution the boolean evaluator uses (200 pts) for single-trajectory
    # targets. If the optimizer scored peaks on a coarser grid than the checker, it could
    # report loss 0 while the checker still failed peak_time (peak lands at a different t
    # on each grid), so the loop would stall at "so close but not met".
    num_points = 500 if hard else 200
    # Generation budget is FIXED (not clock-driven) so the result is reproducible;
    # it is sized so even a ~12-species model finishes a round in ~20s rather than
    # grinding for over a minute.
    de_popsize, de_maxiter = (30, 200) if hard else (12, 30)
    # Cap TOTAL objective evaluations by problem size. Differential evolution costs
    # popsize * n_params evaluations per generation, so a big model (30+ rate
    # constants, especially after drive knobs are added) silently costs an order of
    # magnitude more than a small one and blew past the wall-clock backstop. Deriving
    # the generation count from the parameter count keeps the cost bounded and stays
    # deterministic — unlike trimming by elapsed time, which would make the result
    # depend on machine load again.
    max_evals = 20000 if hard else 4000
    per_generation = max(1, de_popsize * len(tunable))
    de_maxiter = max(5, min(de_maxiter, max_evals // per_generation))
    if hard:
        time_budget = max(time_budget, 180.0)
    else:
        # Keep each round snappy for the everyday peak / steady-state / decay targets:
        # differential evolution early-stops the moment all targets are met, so this
        # cap only bites when a target is unsatisfiable and the search would grind.
        # A frozen cascade that needed basal-drive knobs is a harder search (the drive
        # combination that lifts the readout is a narrow region), so allow more time.
        # The deadline is only a runaway backstop (the real stop is "targets met" or
        # the fixed generation budget), so it sits above normal convergence time — a
        # tight clock made success depend on machine load. These were halved once the
        # solver switched to LSODA: the same number of evaluations now costs ~4.5x
        # less, so the old 75-90s allowances were simply idle time before the backstop.
        time_budget = max(time_budget, 50.0 if injected_drives else 40.0)
    if max_seconds is not None:
        # An explicit ceiling from the caller (used for the second, structural pass so
        # one refinement round cannot cost twice the budget) always wins.
        time_budget = min(time_budget, float(max_seconds))

    # Wall-clock BACKSTOP (not the primary control). It must never decide the answer:
    # feeding a huge constant into the search once time runs out made the result
    # depend on machine load — the same model met its targets on an idle run and
    # failed on a busy one. Instead the deadline raises, the search unwinds, and the
    # best-so-far is kept; the real stopping rule is "all targets met" plus a fixed
    # generation budget, which is deterministic for a fixed seed.
    class _OptTimeout(Exception):
        pass

    start = time.time()
    # Per-PHASE deadline. The local search gets only a small slice: Powell's
    # "maxiter" counts full line-search sweeps over every parameter, so on a large
    # model it silently burns thousands of evaluations and once consumed the entire
    # budget before differential evolution — the part that actually finds solutions —
    # ran at all. Giving each phase its own deadline guarantees DE gets the bulk.
    phase = {"deadline": start + time_budget * 0.25}

    # Score candidates: single-trajectory targets from one run; bistability and
    # fold-change run their own extra simulations.
    def loss_for(x) -> float:
        if time.time() > phase["deadline"]:
            raise _OptTimeout()
        cp = {p: float(v) for p, v in zip(tunable, x)}
        total = 0.0
        try:
            if single:
                res = model.simulate(t_max, num_points=num_points, custom_params=cp)
                total += _total_target_loss(res, single)
            for t in multi:
                if t.get("type") == "bistability":
                    total += bistability_violation(model, t.get("species"), t_max, cp,
                                                   t.get("min", 0.5))
                else:  # fold_change
                    total += fold_change_violation(model, t, t_max, cp)
        except _OptTimeout:
            raise                      # must unwind, never be scored as a candidate
        except Exception:
            return 1e6                 # an un-integrable candidate really is terrible
        return total

    try:
        best = {"x": list(x0), "loss": loss_for(x0)}
    except _OptTimeout:
        best = {"x": list(x0), "loss": float("inf")}

    def consider(x, l):
        if l < best["loss"]:
            best["loss"] = l
            best["x"] = list(x)

    # 1. SCREEN the parameters. Differential evolution costs popsize * n_params
    #    evaluations per generation, so tuning all ~34 rate constants of a large
    #    compiled model leaves only a handful of generations inside any sane budget —
    #    far too few to converge. Most of those constants (deactivation rates of
    #    upstream species, unrelated Hill constants) barely move the targeted
    #    read-out. One cheap up/down probe per parameter identifies the ones that do,
    #    and searching that small subspace converges instead of stalling.
    active_idx = list(range(len(tunable)))
    if len(tunable) > MAX_SEARCH_DIMS:
        try:
            impact = []
            for i in range(len(tunable)):
                lo, hi = bounds[i]
                base = x0[i]
                span = (hi - lo)
                probes = [min(max(base * 0.4, lo), hi), min(max(base * 2.5 if base else lo + 0.3 * span, lo), hi)]
                delta = 0.0
                for pv in probes:
                    if abs(pv - base) < 1e-12:
                        continue
                    xt = list(x0)
                    xt[i] = pv
                    delta = max(delta, abs(loss_for(xt) - best["loss"]))
                impact.append((delta, i))
            impact.sort(reverse=True)
            active_idx = sorted(i for _, i in impact[:MAX_SEARCH_DIMS])
        except _OptTimeout:
            active_idx = list(range(len(tunable)))   # screening ran out of time: tune all
        except Exception:
            active_idx = list(range(len(tunable)))
        # Injected pulse/removal knobs are the whole reason a frozen read-out can move,
        # so they are always searched even if a probe looked unremarkable.
        forced = {i for i, p in enumerate(tunable) if p in set(injected_drives)}
        active_idx = sorted(set(active_idx) | forced)
        if len(active_idx) < len(tunable):
            logs.append(f"Searching the {len(active_idx)} parameters that actually move the "
                        f"targets (of {len(tunable)}).")

    sub_bounds = [bounds[i] for i in active_idx]
    sub_x0 = [x0[i] for i in active_idx]

    def expand(xr):
        xf = list(best["x"])
        for slot, i in enumerate(active_idx):
            xf[i] = float(xr[slot])
        return xf

    # Re-derive the generation budget for the (smaller) search space.
    per_generation = max(1, de_popsize * len(active_idx))
    de_maxiter = max(8, min(200 if hard else 60, max_evals // per_generation))

    # 2. Fast local search from the current parameters. Powell only refines the
    #    starting point; the global search below is what actually finds solutions,
    #    so a timeout here must not consume the whole budget silently.
    try:
        def local_obj(xr):
            xf = expand(xr)
            l = loss_for(xf)
            consider(xf, l)
            return l

        scipy.optimize.minimize(
            local_obj, sub_x0, method="Powell", bounds=sub_bounds,
            options={"maxiter": 5, "xtol": 1e-2, "ftol": 1e-3},
        )
    except _OptTimeout:
        pass          # expected: the local phase is deliberately time-boxed
    except Exception as e:
        logs.append(f"Local optimization error: {e}")
    # Hand the remaining time to the global search.
    phase["deadline"] = start + time_budget

    # 3. Global search if targets not yet met. Stopping is by "all targets met" or a
    #    fixed generation budget (deterministic for the fixed seed) — NOT by clock —
    #    so the same model gives the same answer on a busy machine as on an idle one.
    if best["loss"] > 1e-3:
        try:
            def obj(xr):
                xf = expand(xr)
                l = loss_for(xf)
                consider(xf, l)
                return l

            def cb(xk, convergence=0.0):
                return best["loss"] <= 1e-3

            scipy.optimize.differential_evolution(
                obj, sub_bounds, maxiter=de_maxiter, popsize=de_popsize, tol=1e-4,
                mutation=(0.5, 1.0), recombination=0.7, polish=True,
                init="latinhypercube", callback=cb, seed=0,
            )
        except _OptTimeout:
            logs.append("Global search hit the time backstop; keeping the best point found.")
        except Exception as e:
            logs.append(f"Global optimization error: {e}")

    refined = _apply_params_to_blueprint(blueprint, tunable, best["x"])
    # Drop any basal-drive knobs the optimizer left unused, so the returned model
    # stays clean; keep the ones it actually needed to lift the readout.
    refined = _prune_injected_drives(refined, injected_drives)
    # Verify against the exact (boolean) evaluator on the tuned blueprint
    # (handles multi-condition bistability/fold-change too).
    try:
        met_count, _ = evaluate_targets_on_blueprint(refined, targets, t_max)
        all_met = met_count == len(targets)
    except Exception:
        all_met = best["loss"] <= 1e-3
    logs.append(f"Numerical optimizer tuned {len(tunable)} parameters (violation score: {best['loss']:.4f}).")
    return refined, all_met, best["loss"], logs


def refine_model(
    blueprint: Dict[str, Any],
    simulation_results: Dict[str, Any],
    targets: List[Dict[str, Any]],
    llm: Optional[Dict[str, Any]] = None
) -> Tuple[Dict[str, Any], List[str], bool]:
    """
    Compares simulation results with targets. If failed, it refines the blueprint parameters or structure.
    Returns: (refined_blueprint, log_messages, all_targets_met)
    """
    logs = []
    failed_targets = []

    # 1. Evaluate targets on the current blueprint (multi-condition aware:
    #    bistability/fold-change run their own extra simulations).
    _met_count, _results = evaluate_targets_on_blueprint(blueprint, targets)
    for tgt_def, res in zip(targets, _results):
        logs.append(res["detail"])
        if not res["met"]:
            failed_targets.append((TargetMetric(tgt_def), res["detail"]))

    if not failed_targets:
        logs.append("SUCCESS: All targets met!")
        return blueprint, logs, True
        
    logs.append(f"Model failed {len(failed_targets)} targets. Initiating model refinement...")

    # PDE models don't have the ODE parameter structure the optimizer tunes.
    if blueprint.get("type") == "PDE":
        logs.append("Numerical target optimization is available for ODE models.")
        return blueprint, logs, False

    # Step 1: numerical optimization on the CURRENT topology, no structure change.
    # For a well-posed model (like the EGFR cascade with its ERK->EGFR feedback) this
    # meets the targets on its own, and it avoids letting a structural or LLM revision
    # replace a good topology with a worse one.
    base_bp, base_met, base_loss, base_logs = optimize_parameters_to_targets(blueprint, targets)
    if base_met:
        logs.extend(base_logs)
        logs.append("SUCCESS: All targets met!")
        return base_bp, logs, True

    # Step 2: numerical tuning alone was not enough (e.g. an oscillation or bistability
    # target that needs a feedback loop the model lacks). Add the structural feedback the
    # target needs and re-fit, then keep whichever result is better so a revision can only
    # help. This path is numerical only, no LLM call, so each optimization round stays fast
    # and deterministic (an LLM topology rewrite here took 60-90s and stalled the loop).
    working_bp = _maybe_add_feedback(blueprint, failed_targets, logs)
    # If the structural step could not change anything (e.g. a peak-time target with no
    # feedback remedy, or a custom-kinetics model), a second optimization on the same
    # topology would just repeat the work, so keep the base result.
    same_edges = json.dumps(working_bp.get("edges"), sort_keys=True) == json.dumps(blueprint.get("edges"), sort_keys=True)
    if same_edges and working_bp.get("odes") == blueprint.get("odes"):
        logs.extend(base_logs)
        logs.append(f"No structural change available; best violation score {base_loss:.4f}.")
        return base_bp, logs, base_met
    # Bound the structural retry: without a ceiling one refinement round paid the full
    # optimisation budget twice (~3 minutes), which reads as a hang from the UI.
    alt_bp, alt_met, alt_loss, alt_logs = optimize_parameters_to_targets(
        working_bp, targets, max_seconds=25.0)

    if alt_loss < base_loss:
        logs.extend(alt_logs)
        logs.append("SUCCESS: All targets met!" if alt_met
                    else f"Best configuration leaves a violation score of {alt_loss:.4f}; continuing.")
        return alt_bp, logs, alt_met
    logs.extend(base_logs)
    logs.append("SUCCESS: All targets met!" if base_met
                else f"Structural revision did not help; kept the tuned model (violation {base_loss:.4f}).")
    return base_bp, logs, base_met


# ==========================================
# 3. SENSITIVITY-ENHANCED FEEDBACK
# ==========================================

def sensitivity_guided_refine(
    blueprint: Dict[str, Any],
    simulation_results: Dict[str, Any],
    targets: List[Dict[str, Any]],
    llm: Optional[Dict[str, Any]] = None
) -> Tuple[Dict[str, Any], List[str], bool, Dict[str, float]]:
    """
    Enhanced refinement with local sensitivity analysis.
    Identifies which parameters most influence the failed targets,
    enabling more targeted and efficient parameter adjustments.

    Returns: (refined_blueprint, logs, all_met, sensitivities)
    """
    from multiscale import local_sensitivity_analysis

    logs = []
    t = simulation_results.get("t", [])
    species_data = simulation_results.get("species", {})

    # 1. Evaluate targets
    failed_targets = []
    for tgt_def in targets:
        metric = TargetMetric(tgt_def)
        y = species_data.get(metric.species, [])
        success, msg = metric.evaluate(t, y)
        logs.append(msg)
        if not success:
            failed_targets.append((metric, msg))

    if not failed_targets:
        logs.append("SUCCESS: All targets met!")
        return blueprint, logs, True, {}

    # 2. Run sensitivity analysis for each failed target
    all_param_names = []
    for edge in blueprint.get("edges", []):
        src = edge.get("source", "")
        tgt = edge.get("target", "")
        etype = edge.get("type", "activation")
        prefix = "act" if etype == "activation" else "inh"
        all_param_names.extend([
            f"{prefix}_{src}_to_{tgt}_k",
            f"{prefix}_{src}_to_{tgt}_Kd",
        ])

    sensitivities = {}
    for metric, msg in failed_targets:
        logs.append(f"Running sensitivity analysis for {metric.species}...")
        try:
            sens = local_sensitivity_analysis(
                blueprint, metric.species, all_param_names,
                t_max=blueprint.get("simulation_config", {}).get("t_max", 50.0)
            )
            sensitivities.update(sens)
            # Log top-3 most sensitive parameters
            sorted_sens = sorted(sens.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
            for pname, sval in sorted_sens:
                logs.append(f"  Sensitivity: {pname} = {sval:.4f}")
        except Exception as e:
            logs.append(f"  Sensitivity analysis failed: {e}")

    # 3. Use sensitivity to guide refinement
    refined_bp, refine_logs, all_met = refine_model(
        blueprint, simulation_results, targets, llm
    )
    logs.extend(refine_logs)

    # 4. Record for DPO preference history
    refinement_history.append({
        "original": json.loads(json.dumps(blueprint)),
        "revised": json.loads(json.dumps(refined_bp)),
        "targets": targets,
        "sensitivities": sensitivities,
        "success": all_met,
        "timestamp": str(np.datetime64('now'))
    })

    return refined_bp, logs, all_met, sensitivities
