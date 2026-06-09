import re
import json
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from google import genai

# DPO-style preference history: track (original, revised, outcome) tuples
refinement_history: List[Dict[str, Any]] = []

# Setup Gemini Client
def get_gemini_client(api_key: str):
    """Create a Gemini client with the given API key."""
    return genai.Client(api_key=api_key)


# ==========================================
# 1. BIOLOGICAL TEXT PARSER (STAGE 1)
# ==========================================

def parse_biological_text(text: str, api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Translates a natural language description into a structured biological blueprint.
    Uses Gemini if an API key is provided, otherwise falls back to a regex parser.
    """
    if api_key:
        try:
            client = get_gemini_client(api_key)
            
            prompt = f"""
            You are a biological model compiler agent. Your task is to translate the following natural language description of a biological system into a structured JSON blueprint for simulation.

            Biological Description:
            "{text}"

            The JSON blueprint must have the following schema:
            {{
                "type": "ODE" or "PDE",
                "nodes": [
                    {{
                        "id": "Short uppercase ID of node, e.g., EGFR",
                        "name": "Full name of entity",
                        "initial_value": float (initial concentration/abundance, default 0.0 or 1.0)
                    }}
                ],
                "edges": [
                    {{
                        "source": "ID of source node",
                        "target": "ID of target node",
                        "type": "activation" or "inhibition",
                        "parameters": {{
                            "k": float (activation/inhibition rate constant, default 0.5),
                            "K_d": float (half-saturation constant, default 1.0),
                            "n": float (Hill coefficient, default 2.0)
                        }}
                    }}
                ],
                "spatial": {{   // Only required if type is PDE
                    "x_grid": 50,
                    "y_grid": 50,
                    "dx": 1.0,
                    "dy": 1.0,
                    "diffusion": {{
                        "NODE_ID": float (diffusion coefficient, e.g., 0.1)
                    }},
                    "reactions": {{
                        "NODE_ID": "algebraic expression for reaction kinetics in terms of nodes, e.g., 'u**2/v - u + 0.01'"
                    }}
                }},
                "simulation_config": {{
                    "t_max": float (max simulation time, default 50.0),
                    "dt": float (time step, only for PDE, default 0.1)
                }}
            }}

            Return ONLY the raw JSON block, with no markdown styling, no backticks (e.g. ```json), and no extra comments.
            """
            response = client.models.generate_content(
                model="gemini-3.5-flash",
                contents=prompt
            )
            # Remove any possible code block markers from the LLM
            clean_text = response.text.replace("```json", "").replace("```", "").strip()
            return json.loads(clean_text)
        except Exception as e:
            print(f"Gemini API parsing failed ({e}), falling back to rule-based parser.")
            
    return rule_based_parse(text)


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

class TargetMetric:
    def __init__(self, target_def: Dict[str, Any]):
        self.species = target_def.get("species")
        self.metric_type = target_def.get("type")  # "peak_time", "peak_value", "decay_ratio", "steady_state"
        self.min_val = target_def.get("min")
        self.max_val = target_def.get("max")
        self.expected_val = target_def.get("value")
        self.tolerance = target_def.get("tolerance", 0.1)

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
            
        return True, "Unknown metric"

def refine_model(
    blueprint: Dict[str, Any],
    simulation_results: Dict[str, Any],
    targets: List[Dict[str, Any]],
    api_key: Optional[str] = None
) -> Tuple[Dict[str, Any], List[str], bool]:
    """
    Compares simulation results with targets. If failed, it refines the blueprint parameters or structure.
    Returns: (refined_blueprint, log_messages, all_targets_met)
    """
    t = simulation_results.get("t", [])
    species_data = simulation_results.get("species", {})
    
    logs = []
    failed_targets = []
    
    # 1. Evaluate targets
    for tgt_def in targets:
        metric = TargetMetric(tgt_def)
        y = species_data.get(metric.species, [])
        success, msg = metric.evaluate(t, y)
        logs.append(msg)
        if not success:
            failed_targets.append((metric, msg))
            
    if not failed_targets:
        logs.append("SUCCESS: All targets met!")
        return blueprint, logs, True
        
    logs.append(f"Model failed {len(failed_targets)} targets. Initiating model refinement...")
    
    # 2. Refinement: LLM or Rule-based
    if api_key:
        try:
            client = get_gemini_client(api_key)
            
            prompt = f"""
            You are a biological model optimization agent. A mathematical model (ODE or PDE) was compiled from a blueprint and simulated, but it failed to meet the experimental targets.

            Current Blueprint:
            {json.dumps(blueprint, indent=2)}

            Failing Targets & Error Messages:
            {json.dumps([msg for _, msg in failed_targets], indent=2)}

            Your job is to modify the blueprint (parameters or interactions) so that the simulation matches the targets. 
            For ODE models, you can:
            1. Adjust the edge parameters (k, K_d, n) or basal rates.
            2. Add/remove nodes or edges to introduce negative/positive feedback loops.
            For PDE models, you can:
            1. Adjust diffusion coefficients.
            2. Adjust reaction formulas.

            Please return a JSON response containing ONLY the updated blueprint in the exact same schema. Do not output code blocks (like ```json), comments, or text.
            """
            response = client.models.generate_content(
                model="gemini-3.5-flash",
                contents=prompt
            )
            clean_text = response.text.replace("```json", "").replace("```", "").strip()
            refined_bp = json.loads(clean_text)
            logs.append("AI agent successfully adjusted parameters/interactions based on target mismatch.")
            return refined_bp, logs, False
        except Exception as e:
            logs.append(f"AI refinement failed: {e}. Falling back to rule-based mathematical refinement.")
            
    # 3. Fallback Rule-Based Refinement (Parameter Tuning)
    refined_bp = json.loads(json.dumps(blueprint))  # Deep copy
    
    for metric, msg in failed_targets:
        sp_name = metric.species
        metric_type = metric.metric_type
        
        # Identify edges feeding into or out of the target species
        target_in_edges = [e for e in refined_bp.get("edges", []) if e.get("target") == sp_name]
        target_out_edges = [e for e in refined_bp.get("edges", []) if e.get("source") == sp_name]
        
        if metric_type == "peak_time":
            # If peak is too late, we need to accelerate the upstream cascade (increase activation k)
            if "too late" in msg:
                logs.append(f"Action: Accelerating cascade to target {sp_name} by increasing activation rate constants (k).")
                for edge in target_in_edges:
                    if edge["type"] == "activation":
                        edge["parameters"]["k"] = min(5.0, edge["parameters"]["k"] * 1.5)
            # If peak is too early, slow down cascade (decrease activation k)
            elif "too early" in msg:
                logs.append(f"Action: Decelerating cascade to target {sp_name} by decreasing activation rate constants (k).")
                for edge in target_in_edges:
                    if edge["type"] == "activation":
                        edge["parameters"]["k"] = max(0.01, edge["parameters"]["k"] * 0.6)
                        
        elif metric_type == "peak_value":
            # If peak value is too low, increase activation strengths or decrease degradation
            if "too low" in msg:
                logs.append(f"Action: Increasing activation k and/or decreasing degradation for {sp_name}.")
                for edge in target_in_edges:
                    if edge["type"] == "activation":
                        edge["parameters"]["k"] = min(5.0, edge["parameters"]["k"] * 1.6)
            # If peak value is too high, decrease activation strengths or increase degradation
            elif "too high" in msg:
                logs.append(f"Action: Decreasing activation k and/or increasing inhibition for {sp_name}.")
                for edge in target_in_edges:
                    if edge["type"] == "activation":
                        edge["parameters"]["k"] = max(0.01, edge["parameters"]["k"] * 0.5)
                        
        elif metric_type == "decay_ratio":
            # If decay is insufficient, we need to strengthen negative feedback (e.g. increase inhibition rate, or increase degradation)
            if "insufficient" in msg:
                logs.append(f"Action: Strengthening negative feedback loop targeting {sp_name}.")
                # Locate any feedback inhibition edges (like ERK inhibiting EGFR)
                inhibition_edges = [e for e in refined_bp.get("edges", []) if e["type"] == "inhibition"]
                if inhibition_edges:
                    for edge in inhibition_edges:
                        edge["parameters"]["k"] = min(5.0, edge["parameters"]["k"] * 1.8)
                        edge["parameters"]["K_d"] = max(0.05, edge["parameters"]["K_d"] * 0.5) # more sensitive
                else:
                    # If no negative feedback edge exists, add one from the target back to the root!
                    # For example, if target is ERK, EGFR is the root, we add ERK inhibits EGFR!
                    root_nodes = [n["id"] for n in refined_bp["nodes"] if n["id"] in ["EGFR", "EGF"]]
                    if root_nodes and sp_name not in root_nodes:
                        new_edge = {
                            "source": sp_name,
                            "target": root_nodes[-1],
                            "type": "inhibition",
                            "parameters": {"k": 1.0, "K_d": 1.0, "n": 2.0}
                        }
                        refined_bp["edges"].append(new_edge)
                        logs.append(f"Action: Added new feedback edge [{sp_name} --| {root_nodes[-1]}] to model topology.")
                        
        elif metric_type == "steady_state":
            # Fit basal parameters
            pass
            
    return refined_bp, logs, False


# ==========================================
# 3. SENSITIVITY-ENHANCED FEEDBACK
# ==========================================

def sensitivity_guided_refine(
    blueprint: Dict[str, Any],
    simulation_results: Dict[str, Any],
    targets: List[Dict[str, Any]],
    api_key: Optional[str] = None
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
        blueprint, simulation_results, targets, api_key
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
