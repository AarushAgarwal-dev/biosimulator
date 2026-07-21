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

STEP 1 - VALIDATION. Act as a biological/physical consistency checker. If the description is physically impossible or self-contradictory (e.g. a membrane protein diffusing freely, mass created from nothing, a reaction that drives a concentration negative), do NOT build a model: return {"validation_errors": ["..."]} only.

STEP 2 - CHOOSE THE REPRESENTATION.
- MECHANISTIC (emit "odes" + "fluxes" + "parameters"): use this whenever the text describes explicit rate processes - entry/influx at a rate, pumping/transport between compartments or pools, release, leak, removal/extrusion, production/decay, or a rate that is "activated by"/"gated by" some quantity. This produces an exact mass-conserving rate-law model.
- QUALITATIVE (emit "edges"): use ONLY for a bare influence sketch ("A activates B, B inhibits C") with no rates, compartments, stimuli, or conservation.
- SPATIAL / PDE (emit "type":"PDE" with a "spatial" block): use when the text mentions diffusion, spatial patterns, or reaction-diffusion (Turing).
When in doubt, prefer MECHANISTIC. Words like enters, pumped, released, leaks, removed, extruded, at a (constant) rate, stimulus, store/compartment/pool => MECHANISTIC.

STEP 3 - HARD RULES (violating these makes the model wrong):
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

def sanitize_blueprint(bp: Dict[str, Any]) -> Dict[str, Any]:
    """
    Make a blueprint self-consistent so it can never crash the ODE compiler or the
    graph renderer: every edge endpoint must be a declared node. LLMs often write an
    edge to a compartment they forgot to declare (e.g. "extruded from the cell" ->
    an EXTRACELLULAR target). Auto-declare any such node (as an initial-0 sink)
    instead of failing.
    """
    if not isinstance(bp, dict):
        return bp
    nodes = bp.get("nodes")
    edges = bp.get("edges")
    if not isinstance(nodes, list) or not isinstance(edges, list):
        return bp
    node_ids = {n.get("id") for n in nodes if isinstance(n, dict) and n.get("id")}
    for e in edges:
        if not isinstance(e, dict):
            continue
        for key in ("source", "target"):
            nid = e.get(key)
            if nid and nid not in node_ids:
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

Return ONLY the corrected raw JSON (same schema: type, nodes, parameters, fluxes, odes, simulation_config). Keep the mechanism and mass conservation - just fix the defect. Enforce: constants/stimuli are PARAMETERS (never species); every term of each d/dt is a rate; a distinct rate constant per process (never reuse one symbol for two rates); no species may go negative (loss terms proportional to the species); a time-varying driver is a parameter or a smooth function of t, never a piecewise if(); the odes keys and the species set must be identical (no ODE for a non-species, no species without an ODE, no dead d/dt=0 species); and the system must be able to leave its initial state (if production is autocatalytic in a form that starts at zero, add a separate initiation/basal term so it is not frozen)."""


def _blueprint_status(bp: Dict[str, Any]) -> Tuple[str, str]:
    """Compile + test-simulate an ODE blueprint. Returns (status, message) where
    status is 'ok' (compiles, finite, non-negative), 'imperfect' (runs but a species
    dips negative), or 'broken' (won't compile, crashes, or blows up to NaN/Inf).
    This catches the failure modes LLM-written models actually have."""
    if not isinstance(bp, dict):
        return "broken", "Blueprint is not a JSON object."
    if bp.get("validation_errors"):
        return "ok", ""                      # model deliberately refused - a valid outcome
    if bp.get("type") == "PDE":
        return "ok", ""                      # PDE path is validated on its own endpoint
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
            last_error = ""
            for attempt in range(MAX_ATTEMPTS):
                if not (isinstance(data, dict) and ("nodes" in data or "validation_errors" in data)):
                    last_error = "the model was not valid JSON with a 'nodes' list"
                    break
                bp = sanitize_blueprint(data)
                status, err = _blueprint_status(bp)
                if status == "ok":
                    if attempt > 0:
                        bp["_llm_notice"] = f"Model self-repaired and validated after {attempt} fix round(s)."
                    return bp
                if status == "imperfect":
                    best_effort = bp          # runs but flawed; keep as a fallback candidate
                last_error = err
                if attempt < MAX_ATTEMPTS - 1:
                    repair = (_REPAIR_PROMPT
                              .replace("<<<DESCRIPTION>>>", text)
                              .replace("<<<PREVIOUS>>>", json.dumps(data)[:4000])
                              .replace("<<<ERROR>>>", err))
                    data = llm_provider.generate_json(client, repair, system=_SYSTEM_COMPILER)

            # No fully-valid model after all rounds: prefer a model that at least RUNS
            # (flagged), else degrade to the deterministic parser rather than hand back
            # something that crashes the compiler.
            if best_effort is not None:
                best_effort["_llm_notice"] = f"Model runs but did not fully validate: {last_error}"
                return best_effort
            fallback = rule_based_parse(text)
            fallback["_llm_notice"] = f"LLM model would not run ({last_error}); used the rule-based parser."
            return fallback
        except Exception as e:
            # Never hard-fail: degrade gracefully to the deterministic parser.
            print(f"LLM parsing failed ({e}); falling back to rule-based parser.")
            fallback = rule_based_parse(text)
            fallback["_llm_notice"] = f"LLM unavailable ({e}); used the rule-based parser."
            return fallback

    return sanitize_blueprint(rule_based_parse(text))


def rule_based_parse(text: str) -> Dict[str, Any]:
    """
    Fallback parser using regular expressions and keywords.
    """
    text_lower = text.lower()
    
    # Stopwords that should never be treated as species/node names
    STOPWORDS = {
        "IT", "ITSELF", "THEM", "THIS", "THAT", "WHICH", "THE", "AND", "OR",
        "TO", "IS", "ARE", "WAS", "WERE", "IN", "ON", "AT", "BY", "OF",
        "A", "AN", "ITS", "THEN", "ALSO", "BOTH", "EACH", "WITH", "FROM",
        "NOT", "BUT", "IF", "SO", "AS", "BE", "HAS", "HAD", "HAVE",
        "SELF", "DOES", "DO", "DID", "WILL", "CAN", "MAY", "SHOULD",
        "PROTEIN", "MOLECULE", "RECEPTOR", "ENZYME", "FACTOR", "COMPLEX",
        "SLOWLY", "QUICKLY", "RAPIDLY", "FAST", "SLOW",
    }
    
    def is_valid_species(name: str) -> bool:
        """Check if a name is a valid biological species identifier."""
        upper = name.upper()
        if upper in STOPWORDS:
            return False
        if len(upper) < 2:
            return False
        # Must contain at least one letter
        if not any(c.isalpha() for c in upper):
            return False
        return True
    
    # 1. Detect PDE vs ODE
    is_pde = any(w in text_lower for w in ["pde", "reaction-diffusion", "turing", "spatial", "pattern", "diffusion"])
    
    # Default structures
    if is_pde:
        # Default Turing model (Gierer-Meinhardt / Activator-Inhibitor)
        return {
            "type": "PDE",
            "nodes": [
                {"id": "U", "name": "Activator U", "initial_value": 1.0},
                {"id": "V", "name": "Inhibitor V", "initial_value": 1.0}
            ],
            "edges": [
                {"source": "U", "target": "U", "type": "activation", "parameters": {"k": 1.0, "K_d": 1.0, "n": 2.0}},
                {"source": "U", "target": "V", "type": "activation", "parameters": {"k": 1.0, "K_d": 1.0, "n": 2.0}},
                {"source": "V", "target": "U", "type": "inhibition", "parameters": {"k": 1.0, "K_d": 1.0, "n": 2.0}}
            ],
            "spatial": {
                "x_grid": 50,
                "y_grid": 50,
                "dx": 1.0,
                "dy": 1.0,
                "diffusion": {
                    "U": 0.05,
                    "V": 1.0
                },
                "reactions": {
                    "U": "U**2 / V - U + 0.02",
                    "V": "U**2 - V"
                }
            },
            "simulation_config": {
                "t_max": 200.0,
                "dt": 0.1
            }
        }
    
    # Otherwise assume ODE Signaling Pathway
    # Extract entities and relations using regex
    sentences = re.split(r'[.!?\n]', text)
    nodes = {}
    edges = []
    
    # Compound sentence pattern: "X binds to Y and activates Y/it"
    compound_activation_patterns = [
        r"(\w+)\s+(?:binds\s+to|binds)\s+(\w+)\s+and\s+(?:activates|stimulates|triggers|induces)\s+(?:it|itself|\w+)",
    ]
    
    activation_patterns = [
        r"(\w+)\s+(?:activates|stimulates|triggers|induces|phosphorylates)\s+(\w+)",
        r"(\w+)\s+(?:increases|promotes)\s+(\w+)",
        r"(\w+)\s+(?:binds\s+to|binds)\s+(\w+)",
    ]
    inhibition_patterns = [
        r"(\w+)\s+(?:inhibits|blocks|suppresses|dephosphorylates)\s+(\w+)",
        r"(\w+)\s+(?:decreases|represses)\s+(\w+)"
    ]
    initial_patterns = [
        r"(?:initial\s+)?(\w+)\s+(?:starts\s+at|is)\s+([0-9.]+)",
        r"([0-9.]+)\s+units?\s+of\s+(\w+)"
    ]
    
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
                if m[0].replace('.', '').isdigit():
                    val, name = float(m[0]), m[1].upper()
                else:
                    name, val = m[0].upper(), float(m[1])
                if is_valid_species(name):
                    nodes[name] = {"id": name, "name": f"{name} molecule", "initial_value": val}
        
        # First try compound patterns (e.g. "X binds to Y and activates it")
        compound_matched = False
        for pattern in compound_activation_patterns:
            matches = re.findall(pattern, sentence, re.IGNORECASE)
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
        for pattern in activation_patterns:
            matches = re.findall(pattern, sentence, re.IGNORECASE)
            for m in matches:
                src, tgt = m[0].upper(), m[1].upper()
                if not is_valid_species(src) or not is_valid_species(tgt):
                    continue
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
        for pattern in inhibition_patterns:
            matches = re.findall(pattern, sentence, re.IGNORECASE)
            for m in matches:
                src, tgt = m[0].upper(), m[1].upper()
                if not is_valid_species(src) or not is_valid_species(tgt):
                    continue
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

    # If no nodes parsed, return EGFR default cascade
    if not nodes:
        return get_default_egfr_blueprint()
        
    return {
        "type": "ODE",
        "nodes": list(nodes.values()),
        "edges": edges,
        "simulation_config": {
            "t_max": 50.0
        }
    }

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

        return True, "Unknown metric"

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
    if pname.startswith("syn_"):
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


def _target_violation(metric: "TargetMetric", t: np.ndarray, y: np.ndarray) -> float:
    """Continuous, non-negative distance from meeting a single target (0 = met)."""
    if y.size == 0:
        return 10.0
    if not np.all(np.isfinite(y)):
        return 1e3
    mt = metric.metric_type
    if mt == "peak_time":
        pt = float(t[int(np.argmax(y))])
        v = 0.0
        if metric.min_val is not None and pt < metric.min_val:
            v += (metric.min_val - pt) / max(abs(metric.min_val), 1.0)
        if metric.max_val is not None and pt > metric.max_val:
            v += (pt - metric.max_val) / max(abs(metric.max_val), 1.0)
        return v
    if mt == "peak_value":
        pv = float(np.max(y))
        v = 0.0
        if metric.min_val is not None and pv < metric.min_val:
            v += (metric.min_val - pv) / max(abs(metric.min_val), 1.0)
        if metric.max_val is not None and pv > metric.max_val:
            v += (pv - metric.max_val) / max(abs(metric.max_val), 1.0)
        return v
    if mt == "decay_ratio":
        pv = float(np.max(y))
        fv = float(y[-1])
        if pv <= 1e-9:
            return 1.0
        cap = metric.max_val if metric.max_val is not None else 1.0
        return max(0.0, (fv / pv) - cap) / max(cap, 0.05)
    if mt == "steady_state":
        fv = float(y[-1])
        val = metric.expected_val if metric.expected_val is not None else 0.0
        tol = metric.tolerance or 0.1
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

SINGLE_TRAJECTORY_TYPES = {
    "peak_time", "peak_value", "decay_ratio", "steady_state", "oscillation",
}


def _final_stable(y: np.ndarray) -> bool:
    y = np.asarray(y, dtype=float)
    if y.size < 10 or not np.all(np.isfinite(y)):
        return False
    tail = y[int(0.85 * len(y)):]
    return (tail.max() - tail.min()) <= 0.02 * max(abs(float(y[-1])), 1.0) + 0.01


def bistability_states(model: "ODEModel", species: str, t_max: float,
                       cp: Optional[Dict[str, float]] = None,
                       high_level: float = 10.0):
    """Settle from a LOW (0) and a HIGH start; return (low_final, high_final, both_stable)."""
    lo = model.simulate(t_max, num_points=150, custom_params=cp, custom_initial={species: 0.0})
    hi = model.simulate(t_max, num_points=150, custom_params=cp, custom_initial={species: high_level})
    ylo, yhi = lo["species"].get(species, []), hi["species"].get(species, [])
    if not ylo or not yhi:
        return 0.0, 0.0, False
    stable = _final_stable(np.array(ylo)) and _final_stable(np.array(yhi))
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


def evaluate_targets_on_blueprint(blueprint: Dict[str, Any],
                                  targets: List[Dict[str, Any]],
                                  t_max: Optional[float] = None,
                                  custom_params: Optional[Dict[str, float]] = None):
    """
    Evaluate ALL targets on a blueprint, running multi-condition simulations for
    bistability/fold-change. Returns (met_count, results) where results is a list
    of {species, type, met, detail}.
    """
    try:
        model = ODEModel(blueprint)
    except Exception as e:
        return 0, [{"met": False, "detail": f"Could not compile model: {e}"} for _ in targets]
    if t_max is None:
        t_max = float(blueprint.get("simulation_config", {}).get("t_max", 50.0))
    cp = custom_params or None

    single = [t for t in targets if (t or {}).get("type") in SINGLE_TRAJECTORY_TYPES]
    base_res = model.simulate(t_max, num_points=200, custom_params=cp) if single else None

    results, met_count = [], 0
    for tgt in targets:
        ty = (tgt or {}).get("type")
        sp = (tgt or {}).get("species")
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
            ok, detail = True, "Unknown metric."
        results.append({"species": sp, "type": ty, "met": bool(ok), "detail": detail})
        if ok:
            met_count += 1
    return met_count, results


def _apply_params_to_blueprint(blueprint: Dict[str, Any], tunable: List[str],
                               values: List[float]) -> Dict[str, Any]:
    """Write optimized parameter values back onto the blueprint so they persist."""
    bp = json.loads(json.dumps(blueprint))
    node_by_id = {n["id"]: n for n in bp.get("nodes", [])}

    def find_edge(src, tgt, etype):
        for e in bp.get("edges", []):
            if (e.get("source") == src and e.get("target") == tgt
                    and e.get("type", "activation") == etype):
                return e
        return None

    for p, val in zip(tunable, values):
        val = round(float(val), 5)
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
    edges = bp["edges"]
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


def optimize_parameters_to_targets(
    blueprint: Dict[str, Any],
    targets: List[Dict[str, Any]],
    time_budget: float = 45.0,
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
    num_points = 500 if hard else 100
    de_popsize, de_maxiter = (30, 200) if hard else (12, 40)
    if hard:
        time_budget = max(time_budget, 180.0)

    # Score candidates: single-trajectory targets from one run; bistability and
    # fold-change run their own extra simulations.
    def loss_for(x) -> float:
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
        except Exception:
            return 1e6
        return total

    start = time.time()
    best = {"x": list(x0), "loss": loss_for(x0)}

    def consider(x, l):
        if l < best["loss"]:
            best["loss"] = l
            best["x"] = list(x)

    # 1. Fast local search from the current parameters.
    try:
        r = scipy.optimize.minimize(
            loss_for, x0, method="Powell", bounds=bounds,
            options={"maxiter": 100, "xtol": 1e-3, "ftol": 1e-4},
        )
        consider(r.x, float(r.fun))
    except Exception as e:
        logs.append(f"Local optimization error: {e}")

    # 2. Global search if targets not yet met and there is time left.
    if best["loss"] > 1e-3 and (time.time() - start) < time_budget:
        try:
            def obj(x):
                l = loss_for(x)
                consider(x, l)
                return l

            def cb(xk, convergence=0.0):
                return best["loss"] <= 1e-3 or (time.time() - start) > time_budget

            scipy.optimize.differential_evolution(
                obj, bounds, maxiter=de_maxiter, popsize=de_popsize, tol=1e-4,
                mutation=(0.5, 1.0), recombination=0.7, polish=True,
                init="latinhypercube", callback=cb,
            )
        except Exception as e:
            logs.append(f"Global optimization error: {e}")

    refined = _apply_params_to_blueprint(blueprint, tunable, best["x"])
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

    working_bp = blueprint

    # 1. (Optional) let a configured LLM propose a structural/topology revision.
    #    Its parameters are then re-fit numerically, so a rough suggestion is fine.
    if llm_provider.wants_llm(llm):
        try:
            client = llm_provider.build_client(llm)
            prompt = f"""
            You are a biological model optimization agent. A model compiled from this blueprint failed to meet the targets. Propose a revised blueprint (you may adjust parameters, add/remove nodes, or add negative/positive feedback edges). Numerical fitting will tune the parameters afterwards, so focus on the topology and rough values.

            Current Blueprint:
            {json.dumps(blueprint, indent=2)}

            Failing Targets:
            {json.dumps([msg for _, msg in failed_targets], indent=2)}

            Return ONLY the updated blueprint JSON in the exact same schema. No markdown, no comments.
            """
            candidate = llm_provider.generate_json(
                client, prompt, system=_SYSTEM_COMPILER
            )
            if isinstance(candidate, dict) and "nodes" in candidate and "edges" in candidate:
                working_bp = sanitize_blueprint(candidate)
                logs.append("LLM proposed a revised topology; fitting its parameters numerically.")
            else:
                logs.append("LLM suggestion had an unexpected shape; keeping current topology.")
        except Exception as e:
            logs.append(f"LLM topology suggestion skipped ({e}).")

    # 2. Ensure a negative-feedback loop exists when a decay target needs one.
    working_bp = _maybe_add_feedback(working_bp, failed_targets, logs)

    # 3. Numerical optimizer: the reliable workhorse that actually meets targets.
    refined_bp, all_met, loss, opt_logs = optimize_parameters_to_targets(working_bp, targets)
    logs.extend(opt_logs)
    logs.append("SUCCESS: All targets met!" if all_met
                else f"Best configuration leaves a violation score of {loss:.4f}; continuing.")
    return refined_bp, logs, all_met


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
