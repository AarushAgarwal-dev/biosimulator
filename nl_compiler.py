"""Reliable plain-language -> biological model compiler.

The language model is restricted to a structured intermediate representation (IR).
All mathematics is assembled deterministically here, then checked against that IR.
"""
from __future__ import annotations

import copy
import json
import math
import re
import time
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import sympy as sp

import llm_provider
from simulation_engine import ODEModel, derive_edges_from_odes, solve_pde

COMPILER_VERSION = "nl_compiler/ir-v1"
KINDS = ("production", "degradation", "conversion", "binding", "custom")
MODEL_TYPES = ("ode", "reaction_diffusion")
TIME_UNITS = ("s", "min", "h", "day", "arbitrary")

_REGULATOR_SCHEMA = {
    "type": "object",
    "properties": {
        "species": {"type": ["string", "null"]},
        "stimulus": {"type": ["string", "null"]},
        "effect": {"type": "string", "enum": ["activate", "repress"]},
        "K": {"type": "string"},
        "n": {"type": "string"},
    },
    "required": ["effect", "K", "n"],
    "additionalProperties": False,
}
_STOICH_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {"species": {"type": "string"}, "stoich": {"type": "number"}},
        "required": ["species", "stoich"],
        "additionalProperties": False,
    },
}
IR_SCHEMA: Dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "BioSimulateAI biological model IR",
    "type": "object",
    "properties": {
        "model_type": {"type": "string", "enum": list(MODEL_TYPES)},
        "time_unit": {"type": "string", "enum": list(TIME_UNITS)},
        "t_end": {"type": ["number", "null"]},
        "species": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"}, "name": {"type": "string"},
                    "initial": {"type": "number"},
                    "initial_source": {"type": "string", "enum": ["text", "default"]},
                    "diffusion": {"type": ["number", "null"]},
                    "role": {"type": "string", "enum": ["state"]},
                },
                "required": ["id", "name", "initial", "initial_source", "diffusion", "role"],
                "additionalProperties": False,
            },
        },
        "parameters": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"}, "value": {"type": "number"},
                    "source": {"type": "string", "enum": ["text", "default"]},
                    "unit": {"type": "string"}, "meaning": {"type": "string"},
                },
                "required": ["name", "value", "source", "unit", "meaning"],
                "additionalProperties": False,
            },
        },
        "stimuli": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "profile": {"type": "string", "enum": ["constant", "step", "pulse"]},
                    "level": {"type": "number"}, "t_on": {"type": ["number", "null"]},
                    "t_off": {"type": ["number", "null"]}, "evidence": {"type": "string"},
                },
                "required": ["name", "profile", "level", "t_on", "t_off", "evidence"],
                "additionalProperties": False,
            },
        },
        "processes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"}, "kind": {"type": "string", "enum": list(KINDS)},
                    "target": {"type": "string"}, "species": {"type": "string"},
                    "k": {"type": "string"}, "half_life": {"type": "number"},
                    "regulators": {"type": "array", "items": _REGULATOR_SCHEMA},
                    "basal": {"type": "number"}, "logic": {"type": "string", "enum": ["or", "and"]},
                    "saturable": {"type": "boolean"}, "Km": {"type": "string"},
                    "reactants": _STOICH_SCHEMA, "products": _STOICH_SCHEMA,
                    "enzyme": {"type": "string"}, "kon": {"type": "string"},
                    "koff": {"type": "string"}, "complex": {"type": "string"},
                    "rate": {"type": "string"}, "evidence": {"type": "string"},
                    "assumed": {"type": "boolean"}, "reason": {"type": "string"},
                },
                "required": ["id", "kind", "evidence", "assumed"],
                "allOf": [
                    {"if": {"properties": {"kind": {"const": "production"}}},
                     "then": {"required": ["target", "k"]}},
                    {"if": {"properties": {"kind": {"const": "degradation"}}},
                     "then": {"required": ["species"], "anyOf": [{"required": ["k"]}, {"required": ["half_life"]}]}},
                    {"if": {"properties": {"kind": {"const": "conversion"}}},
                     "then": {"required": ["reactants", "products", "k"]}},
                    {"if": {"properties": {"kind": {"const": "binding"}}},
                     "then": {"required": ["reactants", "complex", "kon", "koff"]}},
                    {"if": {"properties": {"kind": {"const": "custom"}}},
                     "then": {"required": ["reactants", "products", "rate"]}},
                ],
                "additionalProperties": False,
            },
        },
        "assumptions": {"type": "array", "items": {"type": "string"}},
        "unmodeled": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["model_type", "time_unit", "t_end", "species", "parameters", "stimuli",
                 "processes", "assumptions", "unmodeled"],
    "additionalProperties": False,
}

_SYSTEM = """You are a biological-model information extractor. Return only the requested strict JSON IR.
Do not write ODEs. Model only what the description says. Preserve exact entity names and one process per
interaction. Every number copied from the description has source 'text'; all supplied values have source
'default'. Evidence must be a short verbatim quote from the description. Unsupported processes must be
marked assumed with a reason. Never invent upstream or downstream species."""
_PROMPT = """Extract a biological model IR (strict JSON) from DESCRIPTION.

PROCESS KINDS - use ONLY these five values for "kind":
- "production": ZERO-order synthesis of "target" at rate parameter "k". Regulation (X activates / induces /
  inhibits / represses / blocks Y) is a production of Y with a regulator {"species": X, "effect":
  "activate"|"repress", "K": <param>, "n": <param>}. NEVER use kinds such as "activation" or "inhibition".
- "degradation": first-order loss of "species" with rate parameter "k" (or "half_life" as a number).
  "X promotes the degradation of Y" = degradation of Y with regulator X, effect "activate".
- "conversion": "reactants" -> "products" (mass action, rate parameter "k"); optional "enzyme" + "Km".
  Use it for phosphorylation, transport between pools, infection/recovery, maturation.
- "binding": reactants A + B <-> "complex" with parameters "kon" and "koff"; the complex is a species.
- "custom": an explicit rate law "rate" (species ids, parameter names, t) with reactants/products.
RULES:
- Model ONLY what is described. Do not add upstream/downstream species. Keep the exact entity names.
- Every species needs at least one process. Every parameter a process references must be declared in
  "parameters" with value, source ("text" if the number is written in DESCRIPTION, else "default"), unit
  and meaning. Species "initial" values from DESCRIPTION have initial_source "text".
- "evidence" is a short VERBATIM quote of the sentence stating that process. A process that the text does
  not state (e.g. turnover needed so a produced species stays bounded) has "assumed": true and a "reason".
- An imposed NON-MOLECULAR condition (e.g. "DNA damage", "light", "heat", "a drug treatment", "a signal")
  goes in "stimuli", never also in "species". Named molecules - proteins, genes, mRNAs, ligands such as
  EGF, ions - are always species, even when the text never says how they are made.
- diffusion / spatial / Turing / pattern => "model_type": "reaction_diffusion" with species diffusion values.
- Put phrases you cannot represent in "unmodeled".

EXAMPLE
DESCRIPTION: "Kinase K activates X. X represses its own transcription and is degraded with a half-life of
20 min. L binds R to form C with kon 1 and koff 0.1. X starts at 0.5."
IR:
{"model_type":"ode","time_unit":"min","t_end":null,
 "species":[{"id":"K","name":"Kinase K","initial":1.0,"initial_source":"default","diffusion":null,"role":"state"},
  {"id":"X","name":"X","initial":0.5,"initial_source":"text","diffusion":null,"role":"state"},
  {"id":"L","name":"L","initial":1.0,"initial_source":"default","diffusion":null,"role":"state"},
  {"id":"R","name":"R","initial":1.0,"initial_source":"default","diffusion":null,"role":"state"},
  {"id":"C","name":"C","initial":0.0,"initial_source":"default","diffusion":null,"role":"state"}],
 "parameters":[{"name":"k_prod_X","value":1.0,"source":"default","unit":"1/min","meaning":"max production of X"},
  {"name":"K_K_X","value":1.0,"source":"default","unit":"conc","meaning":"K half-activation of X"},
  {"name":"n_K_X","value":2.0,"source":"default","unit":"1","meaning":"Hill coefficient"},
  {"name":"K_X_X","value":1.0,"source":"default","unit":"conc","meaning":"X self-repression threshold"},
  {"name":"n_X_X","value":2.0,"source":"default","unit":"1","meaning":"Hill coefficient"},
  {"name":"kon_L_R","value":1.0,"source":"text","unit":"1/(conc min)","meaning":"association"},
  {"name":"koff_C","value":0.1,"source":"text","unit":"1/min","meaning":"dissociation"},
  {"name":"k_turnover_K","value":0.1,"source":"default","unit":"1/min","meaning":"assumed turnover"}],
 "stimuli":[],
 "processes":[
  {"id":"p1","kind":"production","target":"X","k":"k_prod_X","evidence":"Kinase K activates X","assumed":false,
   "regulators":[{"species":"K","effect":"activate","K":"K_K_X","n":"n_K_X"}]},
  {"id":"p2","kind":"production","target":"X","k":"k_prod_X","evidence":"X represses its own transcription","assumed":false,
   "regulators":[{"species":"X","effect":"repress","K":"K_X_X","n":"n_X_X"}]},
  {"id":"p3","kind":"degradation","species":"X","half_life":20,"evidence":"is degraded with a half-life of 20 min","assumed":false},
  {"id":"p4","kind":"binding","reactants":[{"species":"L","stoich":1},{"species":"R","stoich":1}],"complex":"C",
   "kon":"kon_L_R","koff":"koff_C","evidence":"L binds R to form C with kon 1 and koff 0.1","assumed":false},
  {"id":"p5","kind":"degradation","species":"K","k":"k_turnover_K","evidence":"","assumed":true,
   "reason":"K is named only as a regulator; slow turnover keeps it defined"}],
 "assumptions":["Unspecified rate constants use default values."],"unmodeled":[]}

DESCRIPTION:
<<<TEXT>>>
"""

_ID_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]*")

# A non-assumed process must be supported by wording of ITS OWN KIND that genuinely occurs in the
# description. Token overlap alone let a model recycle "Y is converted to X" as the evidence for an
# invented degradation of Y, which silently broke the conservation the text implied.
_KIND_CUES: Dict[str, Tuple[str, ...]] = {
    "production": (r"produc", r"synthes", r"express", r"transcri", r"translat", r"secret", r"suppl",
                   r"influx", r"enter", r"\bmade\b", r"generat", r"creat", r"activat", r"induc",
                   r"stimulat", r"promot", r"upregulat", r"increas", r"enhanc", r"repress", r"inhibit",
                   r"block", r"suppress", r"downregulat", r"decreas", r"reduc", r"antagoni", r"itself",
                   r"\bown\b", r"grow", r"birth", r"\bborn", r"source", r"stabili", r"driv", r"trigger",
                   r"turns? on", r"switch", r"phosphorylat", r"recruit", r"feedback", r"regulat",
                   r"control", r"\bmakes?\b", r"releas", r"\bupstream", r"constant rate", r"basal"),
    "degradation": (r"degrad", r"decay", r"half[- ]?life", r"turnover", r"turn(?:s|ed)? over", r"remov",
                    r"\blost\b", r"\bloss", r"\blos(?:e|es|ing)\b", r"clear", r"internali", r"export",
                    r"efflux", r"excret", r"\bdie", r"death", r"dying", r"destr", r"cleav", r"proteoly",
                    r"consum", r"dilut", r"eliminat", r"stabili", r"ubiquitin", r"break(?:s|down| down)",
                    r"broken down", r"leak", r"wash", r"outflow", r"\btargets?\b", r"extru", r"secret",
                    r"eaten", r"kill", r"inactivat", r"deplet", r"catabol"),
    "conversion": (r"conver", r"phosphorylat", r"transport", r"pump", r"export", r"import", r"releas",
                   r"translat", r"becom", r"chang", r"transit", r"isomeri", r"cleav", r"process",
                   r"modif", r"->", r"\u2192", r"cataly", r"recover", r"infect", r"flow", r"\bmov",
                   r"enter", r"leak", r"uptake", r"taken up", r"glycoly", r"metaboli", r"oxidi", r"reduc",
                   r"\bturn", r"form", r"activat", r"matur", r"differentiat", r"bind", r"transfer",
                   r"shuttl", r"translocat", r"sequest", r"yield", r"split", r"dimeri", r"assembl",
                   r"produc", r"consum", r"degrad", r"inactivat", r"exchang", r"cycl"),
    "binding": (r"bind", r"bound", r"complex", r"associat", r"dimeri", r"form", r"assembl", r"sequest",
                r"attach", r"dock", r"captur", r"ligat", r"engag"),
}


# An interaction quoted from a negated clause ("A does not activate B") is evidence AGAINST it.
_NEGATED = re.compile(r"\b(?:does|do|did|is|are|was|were|can|could|will|would|should)\s+not\b|n't\b|\bnever\b|"
                      r"\bfails?\s+to\b|\bno\s+longer\b|\bneither\b|\bnor\b|\bcannot\b", re.I)


def _kind_supported(kind: str, evidence: str, text: str) -> bool:
    cues = _KIND_CUES.get(kind)
    if not cues:
        return True
    ev, tx = (evidence or "").lower(), (text or "").lower()
    return any(re.search(cue, ev) and re.search(cue, tx) for cue in cues)
_STOP = {"a", "an", "the", "and", "or", "to", "of", "in", "on", "at", "by", "with", "from",
         "is", "are", "was", "were", "be", "been", "it", "its", "itself", "then", "rate", "constant",
         "initial", "starts", "start", "minutes", "minute", "hours", "hour", "seconds", "second", "units",
         "protein", "gene", "species", "concentration", "expression", "transcription", "degradation"}

# Capitalised only because they open a sentence: instructions and connectives, never biological
# entities. Used by the coverage check alone (the evidence matcher keeps _STOP).
_NOT_ENTITIES = {"simulate", "simulation", "run", "model", "assume", "assuming", "use", "using", "consider",
                 "plot", "show", "initially", "then", "after", "before", "when", "while", "once", "each", "both",
                 "all", "there", "this", "these", "that", "those", "their", "they", "over", "during", "within",
                 "under", "for", "without", "upon", "because", "since", "also", "finally", "here", "note",
                 "given", "suppose", "let", "set", "treat", "include", "ignore", "keep", "make", "time",
                 "rates", "production", "its", "however", "meanwhile", "together", "otherwise", "if",
                 "not", "only", "every", "some", "no", "total", "starting", "begin", "begins", "end"}


def _empty_ir(model_type: str = "ode", time_unit: str = "arbitrary") -> Dict[str, Any]:
    return {"model_type": model_type, "time_unit": time_unit, "t_end": None, "species": [],
            "parameters": [], "stimuli": [], "processes": [], "assumptions": [], "unmodeled": []}


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _words(text: str) -> List[str]:
    return [w.lower() for w in _TOKEN_RE.findall(text or "") if w.lower() not in _STOP]


def _evidence_overlap(evidence: str, text: str) -> float:
    ev = set(_words(evidence)); src = Counter(_words(text))
    if not ev:
        return 0.0
    return sum(1 for token in ev if src[token]) / len(ev)


def _stoich(items: Any) -> List[Tuple[str, float]]:
    result = []
    for item in items or []:
        if isinstance(item, str):
            result.append((item, 1.0))
        elif isinstance(item, dict):
            result.append((str(item.get("species", "")), float(item.get("stoich", 1))))
    return result


def validate_ir(ir: Dict[str, Any], text: str) -> List[str]:
    """Return precise semantic/schema errors; an empty list means the IR is compilable."""
    errors: List[str] = []
    if not isinstance(ir, dict):
        return ["IR must be a JSON object."]
    required = ("model_type", "time_unit", "t_end", "species", "parameters", "stimuli",
                "processes", "assumptions", "unmodeled")
    for key in required:
        if key not in ir:
            errors.append(f"Missing required IR field '{key}'.")
    if errors:
        return errors
    if ir.get("model_type") not in MODEL_TYPES:
        errors.append("model_type must be 'ode' or 'reaction_diffusion'.")
    if ir.get("time_unit") not in TIME_UNITS:
        errors.append("time_unit is invalid.")
    if ir.get("t_end") is not None and (not _finite(ir["t_end"]) or float(ir["t_end"]) <= 0):
        errors.append("t_end must be null or a positive finite number.")

    species = ir.get("species")
    params = ir.get("parameters")
    stimuli = ir.get("stimuli")
    processes = ir.get("processes")
    if not isinstance(species, list) or not species:
        errors.append("At least one species is required.")
        species = []
    for key, value in (("parameters", params), ("stimuli", stimuli), ("processes", processes),
                       ("assumptions", ir.get("assumptions")), ("unmodeled", ir.get("unmodeled"))):
        if not isinstance(value, list):
            errors.append(f"{key} must be a list.")
    params = params if isinstance(params, list) else []
    stimuli = stimuli if isinstance(stimuli, list) else []
    processes = processes if isinstance(processes, list) else []

    sids: List[str] = []
    for i, item in enumerate(species):
        if not isinstance(item, dict):
            errors.append(f"species[{i}] must be an object."); continue
        sid = item.get("id")
        if not isinstance(sid, str) or not _ID_RE.match(sid):
            errors.append(f"species[{i}].id must be a safe identifier."); continue
        if sid in sids:
            errors.append(f"Duplicate species id '{sid}'.")
        sids.append(sid)
        if not _finite(item.get("initial")) or float(item.get("initial", -1)) < 0:
            errors.append(f"Species '{sid}' initial value must be finite and non-negative.")
        if item.get("initial_source") not in ("text", "default"):
            errors.append(f"Species '{sid}' initial_source must be text or default.")
        diff = item.get("diffusion")
        if diff is not None and (not _finite(diff) or float(diff) <= 0):
            errors.append(f"Species '{sid}' diffusion must be null or positive and finite.")
        if item.get("role") != "state":
            errors.append(f"Species '{sid}' role must be state.")
    sid_set = set(sids)

    pnames: List[str] = []
    for i, item in enumerate(params):
        if not isinstance(item, dict):
            errors.append(f"parameters[{i}] must be an object."); continue
        name = item.get("name")
        if not isinstance(name, str) or not _ID_RE.match(name):
            errors.append(f"parameters[{i}].name must be a safe identifier."); continue
        if name in pnames or name in sid_set:
            errors.append(f"Duplicate or colliding parameter name '{name}'.")
        pnames.append(name)
        if not _finite(item.get("value")) or float(item.get("value", 0)) <= 0:
            errors.append(f"Parameter '{name}' must have a positive finite value.")
        if item.get("source") not in ("text", "default"):
            errors.append(f"Parameter '{name}' source must be text or default.")
    pset = set(pnames)

    stimulus_names: List[str] = []
    for i, item in enumerate(stimuli):
        if not isinstance(item, dict):
            errors.append(f"stimuli[{i}] must be an object."); continue
        name = item.get("name")
        if not isinstance(name, str) or not _ID_RE.match(name):
            errors.append(f"stimuli[{i}].name must be a safe identifier."); continue
        if name in stimulus_names or name in sid_set or name in pset:
            errors.append(f"Duplicate or colliding stimulus name '{name}'.")
        stimulus_names.append(name)
        if item.get("profile") not in ("constant", "step", "pulse"):
            errors.append(f"Stimulus '{name}' profile is invalid.")
        if not _finite(item.get("level")) or float(item.get("level", -1)) < 0:
            errors.append(f"Stimulus '{name}' level must be finite and non-negative.")
        if _evidence_overlap(str(item.get("evidence", "")), text) < 0.6:
            errors.append(f"Stimulus '{name}' evidence is not supported by the description.")
    stimset = set(stimulus_names)

    seen_proc = set(); participating = set()
    symbols = {name: sp.Symbol(name) for name in sid_set | pset | {"t"}}
    for i, proc in enumerate(processes):
        if not isinstance(proc, dict):
            errors.append(f"processes[{i}] must be an object."); continue
        pid, kind = proc.get("id"), proc.get("kind")
        if not isinstance(pid, str) or not _ID_RE.match(pid):
            errors.append(f"processes[{i}].id must be a safe identifier.")
            pid = f"processes[{i}]"
        elif pid in seen_proc:
            errors.append(f"Duplicate process id '{pid}'.")
        seen_proc.add(pid)
        if kind not in KINDS:
            errors.append(f"Process '{pid}' has unknown kind '{kind}'."); continue
        assumed = proc.get("assumed")
        if not isinstance(assumed, bool):
            errors.append(f"Process '{pid}' assumed must be boolean.")
        if assumed and not str(proc.get("reason", "")).strip():
            errors.append(f"Assumed process '{pid}' requires a reason.")
        if not assumed and _evidence_overlap(str(proc.get("evidence", "")), text) < 0.6:
            errors.append(f"Process '{pid}' evidence is not supported by the description.")
        elif not assumed and _NEGATED.search(str(proc.get("evidence", ""))):
            errors.append(f"Process '{pid}' rests on a negated statement ('{str(proc.get('evidence'))[:80]}'). "
                          f"A description that says an interaction does NOT happen must not be modelled as that "
                          f"interaction - remove the process.")
        elif not assumed and not _kind_supported(kind, str(proc.get("evidence", "")), text):
            errors.append(f"Process '{pid}' is a {kind}, but its evidence quote contains no {kind} wording "
                          f"that occurs in the description. Quote the phrase that states this {kind}, or "
                          f"remove the process (or mark it assumed with a reason).")

        def species_ref(name: Any, field: str) -> bool:
            if name not in sid_set:
                errors.append(f"Process '{pid}' {field} references unknown species '{name}'.")
                return False
            participating.add(name); return True

        def param_ref(name: Any, field: str) -> bool:
            if name not in pset:
                errors.append(f"Process '{pid}' {field} references unknown parameter '{name}'.")
                return False
            return True

        if kind == "production":
            species_ref(proc.get("target"), "target")
            param_ref(proc.get("k"), "k")
            basal = proc.get("basal", 0)
            if not _finite(basal) or not 0 <= float(basal) <= 1:
                errors.append(f"Process '{pid}' basal must be between 0 and 1.")
        elif kind == "degradation":
            species_ref(proc.get("species"), "species")
            if proc.get("half_life") is None:
                param_ref(proc.get("k"), "k")
            elif not _finite(proc.get("half_life")) or float(proc["half_life"]) <= 0:
                errors.append(f"Process '{pid}' half_life must be positive and finite.")
            if proc.get("saturable"):
                param_ref(proc.get("Km"), "Km")
        elif kind in ("conversion", "custom"):
            reactants, products = _stoich(proc.get("reactants")), _stoich(proc.get("products"))
            if not reactants and not products:
                errors.append(f"Process '{pid}' needs reactants or products.")
            for field, items in (("reactants", reactants), ("products", products)):
                for name, coeff in items:
                    species_ref(name, field)
                    if not _finite(coeff) or coeff <= 0:
                        errors.append(f"Process '{pid}' {field} stoichiometry must be positive.")
            if kind == "conversion":
                param_ref(proc.get("k"), "k")
                enzyme = proc.get("enzyme")
                if enzyme is not None:
                    species_ref(enzyme, "enzyme"); param_ref(proc.get("Km"), "Km")
            else:
                rate = proc.get("rate")
                if not isinstance(rate, str) or not rate.strip():
                    errors.append(f"Custom process '{pid}' requires a rate expression.")
                else:
                    try:
                        expr = sp.sympify(rate, locals=symbols)
                        unknown = {str(s) for s in expr.free_symbols} - set(symbols)
                        if unknown: errors.append(f"Custom process '{pid}' rate has unknown symbols {sorted(unknown)}.")
                        if expr.has(sp.Derivative, sp.Integral): errors.append(f"Custom process '{pid}' rate is unsafe.")
                        # A process that consumes R must stop when R is exhausted, otherwise the
                        # compiled equations drive R negative. (Conversions and binding are built
                        # as mass action, which guarantees this; custom rates must be checked.)
                        for name, _coeff in reactants:
                            if name in symbols and not unknown:
                                at_zero = sp.simplify(expr.subs(symbols[name], 0))
                                if at_zero != 0:
                                    errors.append(f"Custom process '{pid}' consumes '{name}' at a rate that does not "
                                                  f"vanish when {name} = 0 ({at_zero}); {name} would go negative. "
                                                  f"Make the rate proportional to {name}.")
                    except Exception as exc:
                        errors.append(f"Custom process '{pid}' rate does not parse: {type(exc).__name__}.")
        elif kind == "binding":
            reactants = _stoich(proc.get("reactants"))
            if len(reactants) < 2:
                errors.append(f"Binding process '{pid}' needs at least two reactants.")
            for name, coeff in reactants:
                species_ref(name, "reactants")
                if coeff <= 0: errors.append(f"Process '{pid}' stoichiometry must be positive.")
            species_ref(proc.get("complex"), "complex")
            param_ref(proc.get("kon"), "kon"); param_ref(proc.get("koff"), "koff")

        for reg in proc.get("regulators") or []:
            if not isinstance(reg, dict):
                errors.append(f"Process '{pid}' has a malformed regulator."); continue
            refs = int(bool(reg.get("species"))) + int(bool(reg.get("stimulus")))
            if refs != 1:
                errors.append(f"Process '{pid}' regulator must name exactly one species or stimulus.")
            if reg.get("species"): species_ref(reg["species"], "regulator")
            if reg.get("stimulus") not in (None, "") and reg.get("stimulus") not in stimset:
                errors.append(f"Process '{pid}' regulator references unknown stimulus '{reg.get('stimulus')}'.")
            if reg.get("effect") not in ("activate", "repress"):
                errors.append(f"Process '{pid}' regulator effect is invalid.")
            param_ref(reg.get("K"), "regulator K"); param_ref(reg.get("n"), "regulator n")
    for sid in sorted(sid_set - participating):
        errors.append(f"Species '{sid}' does not participate in any process.")
    return errors


def _param_map(ir: Dict[str, Any]) -> Dict[str, float]:
    return {p["name"]: float(p["value"]) for p in ir.get("parameters", [])}


def _regulator_expr(reg: Dict[str, Any], stimuli: Dict[str, str]) -> str:
    source = reg.get("species") or stimuli.get(reg.get("stimulus"), reg.get("stimulus"))
    K, n = reg["K"], reg["n"]
    if reg["effect"] == "activate":
        return f"(({source})**{n}/({K}**{n}+({source})**{n}))"
    return f"({K}**{n}/({K}**{n}+({source})**{n}))"


def _stimulus_expressions(ir: Dict[str, Any], params: Dict[str, float]) -> Dict[str, str]:
    result = {}
    for stimulus in ir.get("stimuli", []):
        name = stimulus["name"]; level = float(stimulus["level"])
        if stimulus["profile"] == "constant":
            pname = f"stim_{name}"
            params[pname] = level
            result[name] = pname
        else:
            on = float(stimulus.get("t_on") or 0.0); slope = 10.0 / max(1.0, abs(on))
            rise = f"(1/(1+exp(-{slope:.12g}*(t-{on:.12g}))))"
            if stimulus["profile"] == "pulse":
                off = float(stimulus.get("t_off") or (on + 1.0))
                fall = f"(1/(1+exp(-{slope:.12g}*(t-{off:.12g}))))"
                shape = f"({rise}-{fall})"
            else:
                shape = rise
            result[name] = f"({level:.12g}*{shape})"
    return result


def _process_base_rate(proc: Dict[str, Any], stimuli: Dict[str, str]) -> str:
    kind = proc["kind"]
    if kind == "production":
        rate = proc["k"]
    elif kind == "degradation":
        x = proc["species"]
        if proc.get("half_life") is not None:
            rate = f"({math.log(2.0) / float(proc['half_life']):.15g}*{x})"
        elif proc.get("saturable"):
            rate = f"({proc['k']}*{x}/({proc['Km']}+{x}))"
        else:
            rate = f"({proc['k']}*{x})"
    elif kind == "conversion":
        reactants = _stoich(proc.get("reactants"))
        substrate = "*".join(name if coeff == 1 else f"{name}**{coeff:g}" for name, coeff in reactants) or "1"
        if proc.get("enzyme"):
            first = reactants[0][0]
            rate = f"({proc['k']}*{proc['enzyme']}*{first}/({proc['Km']}+{first}))"
        else:
            rate = f"({proc['k']}*{substrate})"
    elif kind == "custom":
        rate = f"({proc['rate']})"
    else:
        raise ValueError("binding has two rates")
    regs = proc.get("regulators") or []
    if regs:
        # Activators combine with the stated logic ("or" by default when several converge);
        # repressors always multiply, so any one of them can shut the process down. Applying OR
        # across a repressor factor (which is ~1 when the repressor is absent) would make the
        # process permanently on, so the two classes are never mixed in one logic gate.
        acts = [_regulator_expr(reg, stimuli) for reg in regs if reg.get("effect") == "activate"]
        reps = [_regulator_expr(reg, stimuli) for reg in regs if reg.get("effect") != "activate"]
        basal = float(proc.get("basal", 0.0) or 0.0)
        act_term = None
        if acts:
            if len(acts) > 1 and proc.get("logic") == "or":
                act_term = "(1-(" + ")*(".join(f"1-{factor}" for factor in acts) + "))"
            else:
                act_term = "*".join(acts)
        rep_term = "*".join(reps) if reps else None
        if act_term is not None:
            if basal:
                act_term = f"({basal:.12g}+{1-basal:.12g}*({act_term}))"
            combined = act_term if rep_term is None else f"({act_term})*({rep_term})"
        else:
            combined = f"({basal:.12g}+{1-basal:.12g}*({rep_term}))" if basal else rep_term
        rate = f"({rate})*({combined})"
    return rate


def _flux_name(proc: Dict[str, Any], suffix: str = "") -> str:
    kind = proc["kind"]
    if kind == "production": base = f"prod_{proc['target']}"
    elif kind == "degradation": base = f"deg_{proc['species']}"
    elif kind in ("conversion", "custom"):
        left = "_".join(x[0] for x in _stoich(proc.get("reactants"))) or "source"
        right = "_".join(x[0] for x in _stoich(proc.get("products"))) or "sink"
        base = f"conv_{left}_to_{right}"
    else:
        left = "_".join(x[0] for x in _stoich(proc.get("reactants")))
        base = f"{suffix}_{left}" if suffix else f"bind_{left}"
    return f"{base}_{proc['id']}"


def _process_row(proc: Dict[str, Any]) -> str:
    kind = proc["kind"]
    if kind == "production": desc = f"production of {proc['target']}"
    elif kind == "degradation": desc = f"degradation of {proc['species']}"
    elif kind in ("conversion", "custom"):
        desc = f"conversion of {' + '.join(x[0] for x in _stoich(proc.get('reactants')))} to {' + '.join(x[0] for x in _stoich(proc.get('products')))}"
    else:
        desc = f"binding of {' + '.join(x[0] for x in _stoich(proc.get('reactants')))} to form {proc['complex']}"
    regs = proc.get("regulators") or []
    if regs:
        desc += " " + " and ".join(f"{'activated' if r['effect']=='activate' else 'repressed'} by {r.get('species') or r.get('stimulus')}" for r in regs)
    quote = proc.get("evidence") or proc.get("reason", "assumed")
    return f"{proc['id']} · {desc} — '{quote}'"


def _fold_repressions(processes: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """"R inhibits Y" must inhibit Y's production, not add a second, separately-repressed source.

    Extraction yields one production process per sentence, so "EGF activates EGFR" and "ERK inhibits
    EGFR" arrive as two production terms for EGFR. Compiled side by side, ERK would only suppress the
    second term and leave the EGF-driven one untouched - the inhibition the text states would barely
    act. The established semantics (and the generic Hill compiler's) is multiplicative:
    production = (activation terms) x (inhibition factors). So every repressor-only production of Y is
    folded, as extra repressor factors, into each of Y's other production processes. When Y has no
    other production, the repressor-only process is itself Y's production and is kept unchanged.
    Returns the processes to compile and {folded process id: note}.
    """
    by_target: Dict[str, List[Dict[str, Any]]] = {}
    for proc in processes:
        if proc.get("kind") == "production" and proc.get("target"):
            by_target.setdefault(proc["target"], []).append(proc)
    replacement: Dict[str, Dict[str, Any]] = {}
    folded: Dict[str, str] = {}
    for target, plist in by_target.items():
        rep_only = [p for p in plist if p.get("regulators")
                    and all(r.get("effect") != "activate" for r in p["regulators"])]
        others = [p for p in plist if p not in rep_only]
        if not rep_only or not others:
            continue
        extra = [copy.deepcopy(r) for p in rep_only for r in p["regulators"]]
        for p in others:
            merged = copy.deepcopy(p)
            merged["regulators"] = list(merged.get("regulators") or []) + copy.deepcopy(extra)
            replacement[p["id"]] = merged
        for p in rep_only:
            folded[p["id"]] = (f"applied as a repressor factor on every other production term of {target} "
                               f"({', '.join(o['id'] for o in others)})")
    compiled = [replacement.get(p.get("id"), p) for p in processes if p.get("id") not in folded]
    return compiled, folded


def compile_ir(ir: Dict[str, Any]) -> Dict[str, Any]:
    """Deterministically compile a validated IR into the existing engine blueprint."""
    errors = validate_ir(ir, " ".join(str(p.get("evidence", "")) for p in ir.get("processes", [])) + " " +
                         " ".join(str(s.get("evidence", "")) for s in ir.get("stimuli", [])))
    # Ignore evidence-only errors in direct compilation; structural errors still block.
    errors = [e for e in errors if "evidence is not supported" not in e and "evidence quote contains no" not in e]
    if errors:
        raise ValueError("; ".join(errors))
    params = _param_map(ir)
    stimulus_exprs = _stimulus_expressions(ir, params)
    species = [s["id"] for s in ir["species"]]
    odes: Dict[str, List[str]] = {sid: [] for sid in species}
    fluxes: Dict[str, str] = {}
    edges: List[Dict[str, Any]] = []
    used_names = set()

    def unique(name: str) -> str:
        candidate = name; i = 2
        while candidate in used_names:
            candidate = f"{name}_{i}"; i += 1
        used_names.add(candidate); return candidate

    compiled_processes, folded = _fold_repressions(ir["processes"])
    for proc in compiled_processes:
        if proc["kind"] == "binding":
            bind = unique(_flux_name(proc)); unbind = unique(_flux_name(proc, "unbind"))
            mass = "*".join(name if coeff == 1 else f"{name}**{coeff:g}" for name, coeff in _stoich(proc["reactants"]))
            fluxes[bind] = f"{proc['kon']}*{mass}"; fluxes[unbind] = f"{proc['koff']}*{proc['complex']}"
            for sid, coeff in _stoich(proc["reactants"]):
                odes[sid].append(f"-{coeff:g}*{bind}"); odes[sid].append(f"+{coeff:g}*{unbind}")
            odes[proc["complex"]].append(f"+{bind}"); odes[proc["complex"]].append(f"-{unbind}")
            for sid, _ in _stoich(proc["reactants"]):
                edges.append({"source": sid, "target": proc["complex"], "type": "association", "process": proc["id"]})
            continue
        fname = unique(_flux_name(proc)); fluxes[fname] = _process_base_rate(proc, stimulus_exprs)
        kind = proc["kind"]
        if kind == "production":
            odes[proc["target"]].append(f"+{fname}")
        elif kind == "degradation":
            odes[proc["species"]].append(f"-{fname}")
        else:
            for sid, coeff in _stoich(proc.get("reactants")): odes[sid].append(f"-{coeff:g}*{fname}")
            for sid, coeff in _stoich(proc.get("products")): odes[sid].append(f"+{coeff:g}*{fname}")
            for source, _ in _stoich(proc.get("reactants")):
                for target, _ in _stoich(proc.get("products")):
                    edges.append({"source": source, "target": target, "type": "activation", "process": proc["id"]})
        target = proc.get("target") or proc.get("species")
        if kind in ("conversion", "custom") and _stoich(proc.get("products")):
            target = _stoich(proc.get("products"))[0][0]
        for reg in proc.get("regulators") or []:
            source = reg.get("species") or reg.get("stimulus")
            effect = reg["effect"]
            if kind == "degradation": effect = "repress" if effect == "activate" else "activate"
            edges.append({"source": source, "target": target,
                          "type": "activation" if effect == "activate" else "inhibition", "process": proc["id"]})

    ode_strings = {sid: " ".join(terms).lstrip("+") if terms else "0" for sid, terms in odes.items()}
    # Drop parameters no rate law uses (e.g. the k of a folded repressor-only process): a slider
    # that changes nothing is a trap. They stay listed in the IR and the provenance table.
    referenced = set()
    for expr in list(fluxes.values()) + list(stimulus_exprs.values()):
        referenced |= set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", expr))
    unused_params = sorted(name for name in params if name not in referenced)
    for name in unused_params:
        params.pop(name, None)
    positives = [float(p["value"]) for p in ir["parameters"] if float(p["value"]) > 0]
    slowest = 1.0 / min(positives) if positives else 10.0
    t_end = float(ir.get("t_end") or min(1000.0, max(10.0, 5.0 * slowest)))
    provenance = {p["name"]: {"value": float(p["value"]), "source": p["source"],
                                      "meaning": p.get("meaning", ""),
                                      "process": next((x["id"] for x in ir["processes"] if p["name"] in json.dumps(x)), None),
                                      **({"unused": True} if p["name"] in unused_params else {})}
                  for p in ir["parameters"]}
    table = []
    for p in ir["processes"]:
        row = _process_row(p)
        if p.get("id") in folded:
            row += f" [{folded[p['id']]}]"
        table.append(row)
    bp: Dict[str, Any] = {
        "type": "ODE", "nodes": [{"id": s["id"], "name": s["name"], "initial_value": float(s["initial"])} for s in ir["species"]],
        "parameters": params, "fluxes": fluxes, "odes": ode_strings, "edges": edges,
        "simulation_config": {"t_max": t_end}, "_ir": copy.deepcopy(ir), "_provenance": provenance,
        "_process_table": table, "_compiler": COMPILER_VERSION,
        "_llm_notice": f"Compiled {len(ir['processes'])} explicit processes; {len(ir.get('assumptions', []))} assumptions disclosed.",
    }
    if ir["model_type"] == "reaction_diffusion":
        numeric = {sp.Symbol(k): float(v) for k, v in params.items()}
        numeric[sp.Symbol("t")] = 0.0
        local = {sid: sp.Symbol(sid) for sid in species}
        local.update({k: sp.Symbol(k) for k in params}); local.update({name: sp.Symbol(name) for name in fluxes})
        parsed_flux = {name: sp.sympify(expr, locals=local) for name, expr in fluxes.items()}
        for _ in range(len(parsed_flux) + 1):
            parsed_flux = {name: expr.subs({sp.Symbol(other): other_expr for other, other_expr in parsed_flux.items() if other != name})
                           for name, expr in parsed_flux.items()}
        reactions = {}
        for sid, expr in ode_strings.items():
            parsed = sp.sympify(expr, locals=local).subs({sp.Symbol(k): v for k, v in parsed_flux.items()}).subs(numeric)
            reactions[sid] = str(sp.N(parsed, 14))
        diffusion = {s["id"]: float(s.get("diffusion") or 0.05) for s in ir["species"]}
        # A qualitative activator/inhibitor IR does not contain enough kinetic
        # constants to select a Turing regime. When (and only when) it explicitly
        # states self-activation, activator->inhibitor, and inhibitor-|activator,
        # compile the standard Gierer-Meinhardt realization and disclose it.
        if not any(p.get("kind") == "custom" for p in ir["processes"]):
            self_activators = set()
            activations, repressions = set(), set()
            for proc in ir["processes"]:
                target = proc.get("target")
                for reg in proc.get("regulators", []):
                    source = reg.get("species")
                    if not source or not target: continue
                    pair = (source, target)
                    if reg.get("effect") == "activate":
                        activations.add(pair)
                        if source == target: self_activators.add(source)
                    elif reg.get("effect") == "repress": repressions.add(pair)
            motif = next(((a, i) for a in self_activators for i in species
                          if i != a and (a, i) in activations and (i, a) in repressions
                          and diffusion.get(i, 0) > diffusion.get(a, 0)), None)
            if motif:
                activator, inhibitor = motif
                reactions[activator] = f"{activator}**2/{inhibitor}-{activator}+0.02"
                reactions[inhibitor] = f"{activator}**2-{inhibitor}"
                # A Turing instability grows from small noise around the HOMOGENEOUS STEADY STATE
                # (u* = 1.02, v* = u*^2). Started far from it (the 0.1 defaults), the system first
                # makes large relaxation swings and was still oscillating at t = 50. Unless the text
                # gave starting amounts, start there, and run to t = 200 as the Turing preset does.
                steady = {activator: 1.02, inhibitor: 1.02 ** 2}
                given = {s["id"] for s in ir["species"] if s.get("initial_source") == "text"}
                moved = []
                for node in bp["nodes"]:
                    if node["id"] in steady and node["id"] not in given:
                        node["initial_value"] = round(steady[node["id"]], 4)
                        moved.append(node["id"])
                if not ir.get("t_end"):
                    t_end = 200.0
                bp["_llm_notice"] += (f" The stated activator-inhibitor motif was realized with disclosed "
                                       f"Gierer-Meinhardt kinetics ({activator}**2/{inhibitor}-{activator}+0.02; "
                                       f"{activator}**2-{inhibitor}) so its Turing instability is simulatable"
                                       + (f"; {', '.join(moved)} start at the homogeneous steady state" if moved else "")
                                       + ".")
        dmax = max(diffusion.values()) if diffusion else 0.0
        dt_max = 0.5 / (dmax * 2.0) if dmax else 1.0
        dt = min(0.1, 0.8 * dt_max)
        bp.update({"type": "PDE", "spatial": {"x_grid": 50, "y_grid": 50, "dx": 1.0, "dy": 1.0,
                                                "diffusion": diffusion, "reactions": reactions},
                   "simulation_config": {"t_max": t_end, "dt": dt}})
        bp.pop("fluxes", None); bp.pop("odes", None); bp.pop("parameters", None)
    return bp


def _probe_subs(model: ODEModel) -> Dict[sp.Symbol, float]:
    result = {}
    for sid in model.node_ids:
        initial = float(model.node_map[sid].get("initial_value", 0.0) or 0.0)
        result[model.vars[sid]] = initial if initial > 1e-6 else 1.0
    for name, value in model.params_dict.items(): result[sp.Symbol(name)] = float(value)
    result[sp.Symbol("t")] = 1.0
    return result


def verify(bp: Dict[str, Any], ir: Dict[str, Any], text: str) -> Dict[str, Any]:
    """Perform coverage, symbolic sign, simulation, frozen-species, and PDE checks."""
    report: Dict[str, Any] = {"ok": True, "errors": [], "warnings": [], "coverage": {}, "sign_checks": [],
                              "simulation": {}, "frozen_species": [], "pde_stability": None}
    represented = {s["id"].lower() for s in ir.get("species", [])}
    represented |= {s["name"].lower() for s in ir.get("species", [])}
    represented |= {s["name"].lower() for s in ir.get("stimuli", [])}
    represented |= {p["name"].lower() for p in ir.get("parameters", [])}
    candidates = []
    for token in _TOKEN_RE.findall(text or ""):
        if (token.isupper() or any(c.isdigit() for c in token) or (token[:1].isupper() and len(token) > 2)) \
                and token.lower() not in _STOP and token.lower() not in _NOT_ENTITIES:
            candidates.append(token)
    def _covered(token: str) -> bool:
        low = token.lower()
        for y in represented:
            if low == y or low in re.split(r"[_\-\s]+", y):
                return True
            # Loose containment ("ERK" ~ "pERK", "DNA" ~ "DNA_damage") only for names long enough
            # that it is meaningful; a one-letter species "A" must not cover every word with an a.
            if len(y) >= 3 and len(low) >= 3 and (low in y or y in low):
                return True
        return False

    missing = sorted({x for x in candidates if not _covered(x)})
    report["coverage"] = {"entities": sorted(set(candidates)), "missing": missing}
    if missing: report["warnings"].append("Named entities not represented: " + ", ".join(missing))

    if bp.get("type") == "PDE":
        spatial = bp.get("spatial", {}); cfg = bp.get("simulation_config", {})
        dmax = max([float(v) for v in spatial.get("diffusion", {}).values()] or [0.0])
        number = float(cfg.get("dt", 0.1)) * dmax * (1 / float(spatial.get("dx", 1)) ** 2 + 1 / float(spatial.get("dy", 1)) ** 2)
        report["pde_stability"] = {"diffusion_number": number, "limit": 0.5, "stable": number <= 0.5 + 1e-12}
        if number > 0.5 + 1e-12: report["errors"].append("PDE explicit time step is unstable.")
        try:
            tiny = dict(spatial); tiny.update({"x_grid": 8, "y_grid": 8})
            initial = {s["id"]: {"type": "random_noise", "base_value": float(s["initial"]), "noise_amplitude": 0.01}
                       for s in ir["species"]}
            result = solve_pde(tiny, spatial["reactions"], initial,
                               min(float(cfg["t_max"]), 5 * float(cfg["dt"])), float(cfg["dt"]), save_every=1)
            arrays = {sid: np.asarray(values, dtype=float) for sid, values in result["species"].items()}
            if not all(np.all(np.isfinite(a)) for a in arrays.values()): raise ValueError("non-finite field")
            report["simulation"] = {"success": True, "end_time": result["t"][-1]}
            for sid, a in arrays.items():
                if float(np.ptp(a)) <= 1e-10: report["frozen_species"].append({"species": sid, "reason": "field did not change in smoke run"})
        except Exception as exc:
            report["errors"].append(f"PDE simulation failed: {type(exc).__name__}: {exc}")
    else:
        try:
            model = ODEModel(bp); probe = _probe_subs(model)
            for proc in ir.get("processes", []):
                kind = proc["kind"]
                targets = []
                if kind == "production": targets = [(proc["target"], 1)]
                elif kind == "degradation": targets = [(proc["species"], -1)]
                elif kind in ("conversion", "custom"):
                    targets = [(sid, 1) for sid, _ in _stoich(proc.get("products"))]
                elif kind == "binding": targets = [(proc["complex"], 1)]
                for reg in proc.get("regulators") or []:
                    source = reg.get("species")
                    if not source: continue
                    for target, orientation in targets:
                        expected = (1 if reg["effect"] == "activate" else -1) * orientation
                        deriv = sp.diff(model.deriv_exprs[target], model.vars[source])
                        value = float(deriv.subs(probe).evalf())
                        actual = 1 if value > 1e-10 else -1 if value < -1e-10 else 0
                        check = {"process": proc["id"], "source": source, "target": target,
                                 "expected": expected, "actual": actual, "derivative": value, "ok": actual == expected}
                        report["sign_checks"].append(check)
                        if not check["ok"]: report["errors"].append(f"Sign mismatch for {source} -> {target} in {proc['id']}.")
                if kind in ("conversion", "custom"):
                    for source, _ in _stoich(proc.get("reactants")):
                        for target, _ in _stoich(proc.get("products")):
                            deriv = sp.diff(model.deriv_exprs[target], model.vars[source])
                            value = float(deriv.subs(probe).evalf())
                            ok = value > 1e-10
                            report["sign_checks"].append({"process": proc["id"], "source": source, "target": target,
                                                          "expected": 1, "actual": 1 if value > 0 else -1 if value < 0 else 0,
                                                          "derivative": value, "ok": ok})
                            if not ok: report["errors"].append(f"Conversion sign mismatch for {source} -> {target} in {proc['id']}.")
            t_end = float(bp.get("simulation_config", {}).get("t_max", 50.0)); result = model.simulate(t_end, 240)
            all_values = np.concatenate([np.asarray(v, dtype=float) for v in result["species"].values()])
            scale = max(1.0, max(abs(float(s["initial"])) for s in ir["species"]),
                        max([abs(float(p["value"])) for p in ir.get("parameters", [])] or [1.0]))
            success = bool(np.all(np.isfinite(all_values)) and np.min(all_values) >= -1e-6 * scale and np.max(all_values) < 1e4 * scale)
            report["simulation"] = {"success": success, "end_time": result["t"][-1], "min": float(np.min(all_values)), "max": float(np.max(all_values)), "scale": scale}
            if not success: report["errors"].append("Simulation is non-finite, negative, or unbounded relative to model scale.")
            for sid, values in result["species"].items():
                arr = np.asarray(values); local = max(1.0, abs(float(arr[0])))
                if float(np.ptp(arr)) <= 1e-7 * local:
                    expr = str(model.deriv_exprs[sid])
                    reason = "zero derivative" if model.deriv_exprs[sid] == 0 else f"initial state is effectively stationary under {expr}"
                    report["frozen_species"].append({"species": sid, "reason": reason})
                    report["warnings"].append(f"Species '{sid}' is frozen: {reason}.")
        except Exception as exc:
            report["errors"].append(f"ODE verification failed: {type(exc).__name__}: {exc}")
    report["ok"] = not report["errors"]
    return report


def _normalise_ir(raw: Any) -> Dict[str, Any]:
    if not isinstance(raw, dict): return raw
    data = copy.deepcopy(raw)
    # Keep only the instance fields; unconstrained endpoints sometimes echo the JSON
    # Schema itself alongside the instance.
    ir = {key: data.get(key) for key in ("model_type", "time_unit", "t_end", "species", "parameters",
                                         "stimuli", "processes", "assumptions", "unmodeled")}
    ir["model_type"] = str(ir.get("model_type") or "ode").lower().replace("pde", "reaction_diffusion")
    ir["time_unit"] = str(ir.get("time_unit") or "arbitrary").lower()
    ir["t_end"] = ir.get("t_end")
    for key in ("species", "parameters", "stimuli", "processes", "assumptions", "unmodeled"):
        if not isinstance(ir.get(key), list): ir[key] = []

    canonical_species = []
    for item in ir["species"]:
        if not isinstance(item, dict): continue
        sid = item.get("id") or item.get("species") or item.get("name")
        if not sid: continue
        canonical_species.append({
            "id": str(sid), "name": str(item.get("name") or sid),
            "initial": item.get("initial", item.get("initial_value", 0.1)),
            "initial_source": item.get("initial_source", item.get("source", "default")),
            "diffusion": item.get("diffusion", item.get("diffusion_coefficient")),
            "role": "state",
        })
    ir["species"] = canonical_species

    canonical_params = []
    for item in ir["parameters"]:
        if not isinstance(item, dict) or not (item.get("name") or item.get("id")): continue
        name = item.get("name") or item.get("id")
        canonical_params.append({"name": str(name), "value": item.get("value", item.get("default", 1.0)),
                                 "source": item.get("source", "default"), "unit": str(item.get("unit", "arbitrary")),
                                 "meaning": str(item.get("meaning", item.get("description", name)))})
    ir["parameters"] = canonical_params

    canonical_stimuli = []
    for item in ir["stimuli"]:
        if not isinstance(item, dict) or not (item.get("name") or item.get("id")): continue
        canonical_stimuli.append({"name": str(item.get("name") or item.get("id")),
                                  "profile": item.get("profile", "constant"),
                                  "level": item.get("level", item.get("value", 1.0)),
                                  "t_on": item.get("t_on"), "t_off": item.get("t_off"),
                                  "evidence": str(item.get("evidence", ""))})
    ir["stimuli"] = canonical_stimuli

    def canon_stoich(value):
        if isinstance(value, str): value = [value]
        out = []
        for entry in value or []:
            if isinstance(entry, str): out.append({"species": entry, "stoich": 1})
            elif isinstance(entry, dict) and (entry.get("species") or entry.get("id") or entry.get("name")):
                out.append({"species": str(entry.get("species") or entry.get("id") or entry.get("name")),
                            "stoich": entry.get("stoich", entry.get("coefficient", 1))})
        return out

    canonical_processes = []
    for index, item in enumerate(ir["processes"], 1):
        if not isinstance(item, dict): continue
        kind = str(item.get("kind") or item.get("type") or "").lower()
        if kind in ("synthesis", "transcription"): kind = "production"
        if kind in ("decay", "removal"): kind = "degradation"
        proc = {"id": str(item.get("id") or f"p{index}"), "kind": kind,
                "evidence": str(item.get("evidence", "")), "assumed": bool(item.get("assumed", False))}
        if item.get("reason") or item.get("justification"): proc["reason"] = str(item.get("reason") or item.get("justification"))
        target = item.get("target") or item.get("product")
        state = item.get("species") or (target if kind == "degradation" else None)
        if target is not None and kind == "production": proc["target"] = str(target)
        if state is not None and kind == "degradation": proc["species"] = str(state)
        rate_ref = item.get("k") or item.get("rate_parameter") or item.get("rate_constant")
        if rate_ref is None and kind != "custom" and isinstance(item.get("rate"), str):
            rate_ref = item.get("rate")
        if rate_ref is not None: proc["k"] = str(rate_ref)
        for field in ("half_life", "basal", "logic", "saturable", "Km", "enzyme", "kon", "koff", "complex"):
            if item.get(field) is not None: proc[field] = item[field]
        if kind == "custom" and item.get("rate") is not None: proc["rate"] = item["rate"]
        if kind in ("conversion", "custom", "binding"):
            reactants = item.get("reactants", item.get("substrates", item.get("from")))
            products = item.get("products", item.get("to"))
            proc["reactants"] = canon_stoich(reactants)
            if kind != "binding": proc["products"] = canon_stoich(products)
            elif not proc.get("complex"):
                product_items = canon_stoich(products)
                if product_items: proc["complex"] = product_items[0]["species"]
        regs = []
        for reg in item.get("regulators") or []:
            if not isinstance(reg, dict): continue
            effect = str(reg.get("effect") or reg.get("type") or "").lower()
            effect = "repress" if effect in ("repressor", "inhibitor", "inhibit", "negative") else "activate" if effect in ("activator", "activate", "positive") else effect
            canonical = {"effect": effect,
                         "K": str(reg.get("K") or reg.get("K_parameter") or reg.get("Kd") or ""),
                         "n": str(reg.get("n") or reg.get("n_parameter") or reg.get("hill_parameter") or "")}
            if reg.get("species") is not None: canonical["species"] = reg.get("species")
            if reg.get("stimulus") is not None: canonical["stimulus"] = reg.get("stimulus")
            regs.append(canonical)
        if regs: proc["regulators"] = regs
        canonical_processes.append(proc)
    ir["processes"] = canonical_processes
    return ir


def _schema_text() -> str:
    return "\nReturn JSON matching this schema exactly:\n" + json.dumps(IR_SCHEMA, separators=(",", ":"))


def _auto_repair_ir(ir: Dict[str, Any], text: str) -> Dict[str, Any]:
    """Fix defects that are mechanical, not biological, and disclose every fix.

    Measured on Purdue gpt-oss:120b, most first-attempt rejections were of two kinds: an input
    named only as a regulator ("EGF activates EGFR") was referenced but never declared, and an
    imposed stimulus was given a display name with a space ("DNA damage"). Each cost a full
    15-25 s repair round although the intended model was unambiguous. Repairs made here:
      * unsafe identifiers are sanitised everywhere they are referenced;
      * a regulator naming a declared stimulus is re-pointed at the stimulus;
      * a regulator naming an entity that appears in the description but was not declared gets a
        species with a disclosed basal supply and turnover (the same as the rule-based path);
      * a referenced parameter that was not declared gets a default value, marked 'default'.
    Anything not in the description is left for the validator to reject.
    """
    if not isinstance(ir, dict):
        return ir
    notes: List[str] = []
    rename: Dict[str, str] = {}
    for item in ir.get("species", []) + ir.get("stimuli", []):
        key = "id" if "id" in item else "name"
        old = str(item.get(key, ""))
        if old and not _ID_RE.match(old):
            new = _safe_id(old)
            rename[old] = new
            item[key] = new
            notes.append(f"Identifier '{old}' was written as '{new}'.")
    for p in ir.get("parameters", []):
        old = str(p.get("name", ""))
        if old and not _ID_RE.match(old):
            rename[old] = _safe_id(old)
            p["name"] = rename[old]

    def fix(value: Any) -> Any:
        if isinstance(value, str) and value in rename:
            return rename[value]
        if isinstance(value, str) and not _ID_RE.match(value) and _safe_id(value) in rename.values():
            return _safe_id(value)
        return value

    for proc in ir.get("processes", []):
        for field in ("target", "species", "enzyme", "complex", "k", "Km", "kon", "koff"):
            if field in proc:
                proc[field] = fix(proc[field])
        for field in ("reactants", "products"):
            for entry in proc.get(field, []) or []:
                entry["species"] = fix(entry.get("species"))
        for reg in proc.get("regulators", []) or []:
            for field in ("species", "stimulus", "K", "n"):
                if field in reg:
                    reg[field] = fix(reg[field])

    species_ids = {s.get("id") for s in ir.get("species", [])}
    stimulus_names = {s.get("name") for s in ir.get("stimuli", [])}
    # A named MOLECULE is a species, not an imposed condition. gpt-oss sometimes filed a ligand
    # ("EGF activates EGFR") as a stimulus, which removed it from the graph and made the result
    # depend on which engine compiled it. Only non-molecular conditions stay stimuli.
    _non_molecular = re.compile(r"damage|light|stress|heat|cold|drug|treatment|signal|stimul|irradiat|radiat|"
                                r"hypoxi|shock|pulse|input|dose|temperature|exposure|injur|starv|serum|nutrient",
                                re.I)
    converted = []
    for stim in list(ir.get("stimuli", [])):
        name = str(stim.get("name", ""))
        if not name or _non_molecular.search(name.replace("_", " ")) or name in species_ids:
            continue
        ir["stimuli"].remove(stim)
        ir.setdefault("species", []).append({"id": name, "name": name, "initial": float(stim.get("level") or 1.0),
                                             "initial_source": "default", "diffusion": None, "role": "state"})
        species_ids.add(name)
        for proc in ir.get("processes", []):
            for reg in proc.get("regulators", []) or []:
                if reg.get("stimulus") == name:
                    reg.pop("stimulus", None)
                    reg["species"] = name
        converted.append(name)
    if converted:
        notes.append(f"{', '.join(converted)} {'is a molecule' if len(converted) == 1 else 'are molecules'}, "
                     f"so {'it is' if len(converted) == 1 else 'they are'} modelled as "
                     f"{'a species' if len(converted) == 1 else 'species'} rather than an imposed stimulus.")
    stimulus_names = {s.get("name") for s in ir.get("stimuli", [])}
    # A stimulus whose evidence quote is missing or paraphrased gets the sentence that names it.
    sentences = [s.strip() for s in re.split(r"(?<=[.;!?])\s+", text or "") if s.strip()]
    for stim in ir.get("stimuli", []):
        if _evidence_overlap(str(stim.get("evidence", "")), text) >= 0.6:
            continue
        spoken = str(stim.get("name", "")).replace("_", " ").lower()
        words = [w for w in spoken.split() if w not in _STOP]
        match = next((s for s in sentences if spoken and spoken in s.lower()), None) or \
            next((s for s in sentences if words and all(w in s.lower() for w in words)), None)
        if match:
            stim["evidence"] = match.rstrip(".;")
            notes.append(f"Evidence for stimulus '{stim.get('name')}' was taken from the sentence that names it.")
    stim_lower = {str(n).lower(): n for n in stimulus_names}
    param_names = {p.get("name") for p in ir.get("parameters", [])}
    lowered_text = (text or "").lower()
    added_species: List[str] = []
    for proc in ir.get("processes", []):
        for reg in proc.get("regulators", []) or []:
            sid = reg.get("species")
            if sid and sid not in species_ids:
                if str(sid).lower() in stim_lower:
                    reg.pop("species", None)
                    reg["stimulus"] = stim_lower[str(sid).lower()]
                    continue
                spoken = re.sub(r"_", " ", str(sid)).lower()
                if spoken in lowered_text or str(sid).lower() in lowered_text:
                    ir.setdefault("species", []).append({
                        "id": sid, "name": str(sid), "initial": 1.0, "initial_source": "default",
                        "diffusion": None, "role": "state"})
                    species_ids.add(sid)
                    added_species.append(sid)
    for sid in added_species:
        kp, kd = f"k_supply_{sid}", f"k_turnover_{sid}"
        for name, meaning in ((kp, f"assumed basal supply of {sid}"), (kd, f"assumed turnover of {sid}")):
            if name not in param_names:
                ir.setdefault("parameters", []).append({"name": name, "value": 0.1, "source": "default",
                                                        "unit": "arbitrary", "meaning": meaning})
                param_names.add(name)
        ir.setdefault("processes", []).append({
            "id": f"auto_supply_{sid}", "kind": "production", "target": sid, "k": kp, "evidence": "",
            "assumed": True, "reason": f"{sid} is named only as a regulator; a basal supply keeps it defined"})
        ir["processes"].append({
            "id": f"auto_turnover_{sid}", "kind": "degradation", "species": sid, "k": kd, "evidence": "",
            "assumed": True, "reason": "first-order turnover keeps the supplied regulator bounded"})
        notes.append(f"{sid} was named as a regulator but not declared; it was added as a species with a "
                     f"basal supply and turnover (steady level 1.0).")
    for proc in ir.get("processes", []):
        wanted = [proc.get(f) for f in ("k", "Km", "kon", "koff") if isinstance(proc.get(f), str)]
        wanted += [reg.get(f) for reg in proc.get("regulators", []) or [] for f in ("K", "n")
                   if isinstance(reg.get(f), str)]
        for name in wanted:
            if name and _ID_RE.match(name) and name not in param_names and name not in species_ids:
                default = 2.0 if re.match(r"^n(_|$)", name) else 1.0
                ir.setdefault("parameters", []).append({"name": name, "value": default, "source": "default",
                                                        "unit": "arbitrary", "meaning": f"parameter {name}"})
                param_names.add(name)
                notes.append(f"Parameter '{name}' was referenced but not declared; default {default:g} used.")
    if notes:
        ir.setdefault("assumptions", []).extend(notes)
    _drop_duplicate_processes(ir)
    _prune_unevidenced_species(ir)
    _protect_catalysts(ir)
    _ensure_regulator_sources(ir)
    return ir


def _drop_duplicate_processes(ir: Dict[str, Any]) -> None:
    """Do not count one described rate twice.

    Measured on gpt-oss:120b for "p53 activates Mdm2 transcription. Mdm2 promotes p53 degradation.
    DNA damage stabilizes p53.": the sentence "Mdm2 promotes p53 degradation" became BOTH a
    process "degradation of p53 activated by Mdm2" and a merged "degradation of p53 activated by
    Mdm2 and repressed by DNA damage", which doubles the Mdm2-driven loss. A production/degradation
    process is removed when another process of the same kind on the same species quotes the same
    sentence and already contains all of its regulators. A process carrying a number from the text
    that the other does not use is kept, so no stated value is lost.
    """
    processes = ir.get("processes", []) or []
    text_params = {p.get("name") for p in ir.get("parameters", []) if p.get("source") == "text"}

    def subject(p):
        return p.get("target") if p.get("kind") == "production" else p.get("species")

    def regs(p):
        return {(r.get("species") or r.get("stimulus"), r.get("effect"))
                for r in (p.get("regulators") or []) if isinstance(r, dict)}

    def refs(p):
        out = {p.get(f) for f in ("k", "Km", "kon", "koff") if isinstance(p.get(f), str)}
        out |= {r.get(f) for r in (p.get("regulators") or []) if isinstance(r, dict)
                for f in ("K", "n") if isinstance(r.get(f), str)}
        return out

    def norm(e):
        return re.sub(r"[^a-z0-9]+", " ", str(e or "").lower()).strip()

    removed: List[Dict[str, Any]] = []
    for i, p in enumerate(processes):
        rp = regs(p)
        if p.get("assumed") or p.get("kind") not in ("production", "degradation") or not rp:
            continue
        for j, q in enumerate(processes):
            if j == i or q in removed or q.get("kind") != p.get("kind") or subject(q) != subject(p):
                continue
            rq = regs(q)
            if not rp <= rq or (rp == rq and j > i):          # exact duplicates: keep the first
                continue
            if not norm(p.get("evidence")) or norm(p.get("evidence")) != norm(q.get("evidence")):
                continue
            if p.get("half_life") not in (None, "") or (refs(p) & text_params) - refs(q):
                continue
            removed.append(p)
            ir.setdefault("assumptions", []).append(
                f"Process '{p.get('id')}' repeated what '{q.get('id')}' already encodes from the same sentence "
                f"(\"{str(p.get('evidence'))[:80]}\"); the duplicate was removed so the rate is not counted twice.")
            break
    if removed:
        ir["processes"] = [p for p in processes if p not in removed]


def _prune_unevidenced_species(ir: Dict[str, Any]) -> None:
    """Model only what is described: drop a species that no EVIDENCED process involves.

    Measured on Bedrock Mistral Large: for "A does not activate B. A is degraded at rate 0.1." the
    model kept B and invented an assumed production and turnover for it - B then appears as a
    species the text explicitly says nothing happens to. A species that appears only in assumed
    processes (or in none) is removed together with those processes, and the removal is disclosed.
    """
    processes = ir.get("processes", []) or []

    def involved(p: Dict[str, Any]) -> set:
        out = {p.get("target"), p.get("species"), p.get("complex"), p.get("enzyme")}
        for field in ("reactants", "products"):
            out |= {e.get("species") for e in (p.get(field) or []) if isinstance(e, dict)}
        out |= {r.get("species") for r in (p.get("regulators") or []) if isinstance(r, dict)}
        return {x for x in out if x}

    evidenced = set()
    for p in processes:
        if not p.get("assumed"):
            evidenced |= involved(p)
    drop = [s.get("id") for s in ir.get("species", []) if s.get("id") not in evidenced]
    if not drop or len(drop) == len(ir.get("species", [])):
        return                       # nothing to prune, or nothing evidenced at all (validator reports it)
    dropset = set(drop)
    ir["species"] = [s for s in ir.get("species", []) if s.get("id") not in dropset]
    ir["processes"] = [p for p in processes if not (p.get("assumed") and involved(p) & dropset)]
    ir.setdefault("assumptions", []).append(
        f"Removed {', '.join(drop)}: the description states no process involving "
        f"{'it' if len(drop) == 1 else 'them'}, so modelling {'it' if len(drop) == 1 else 'them'} would invent biology.")


def _protect_catalysts(ir: Dict[str, Any]) -> None:
    """An enzyme described only as a catalyst is not consumed; an ASSUMED production or decay of it
    (occasionally invented by the model) would make the catalyst amount drift. Remove those."""
    processes = ir.get("processes", []) or []
    enzymes = {p.get("enzyme") for p in processes if p.get("kind") == "conversion" and p.get("enzyme")}
    other_roles = set()
    for p in processes:
        if p.get("assumed"):
            continue
        for field in ("target", "species", "complex"):
            if p.get(field):
                other_roles.add(p.get(field))
        for field in ("reactants", "products"):
            other_roles |= {e.get("species") for e in (p.get(field) or []) if isinstance(e, dict)}
    catalysts = {e for e in enzymes if e and e not in other_roles}
    removed = [p for p in processes if p.get("assumed") and p.get("kind") in ("production", "degradation")
               and (p.get("target") in catalysts or p.get("species") in catalysts)]
    if removed:
        ir["processes"] = [p for p in processes if p not in removed]
        names = sorted({p.get("target") or p.get("species") for p in removed})
        ir.setdefault("assumptions", []).append(
            f"{', '.join(names)} only catalyses a described conversion, so the invented turnover of it was removed; "
            f"the catalyst amount is conserved.")


def _ensure_regulator_sources(ir: Dict[str, Any]) -> None:
    """A regulator with a described loss but no described source would decay to zero and take
    the regulation with it (the AI's p53-Mdm2 model did exactly this). Give it a disclosed basal
    supply - the same rule the deterministic extractor applies."""
    processes = ir.get("processes", []) or []
    regulators = {r.get("species") for p in processes for r in (p.get("regulators") or []) if r.get("species")}
    sourced = {p.get("target") for p in processes if p.get("kind") == "production"}
    sourced |= {e.get("species") for p in processes if p.get("kind") in ("conversion", "custom")
                for e in (p.get("products") or [])}
    sourced |= {p.get("complex") for p in processes if p.get("kind") == "binding"}
    lost = {p.get("species") for p in processes if p.get("kind") == "degradation"}
    lost |= {e.get("species") for p in processes if p.get("kind") in ("conversion", "custom", "binding")
             for e in (p.get("reactants") or [])}
    params = {p.get("name") for p in ir.get("parameters", [])}
    # A species that only regulates (never produced, never lost) gets a disclosed basal supply AND
    # turnover (steady level 1.0), exactly as the deterministic extractor does, so both engines
    # give the same model for "EGF activates EGFR".
    only_regulators = sorted(s for s in regulators - sourced - lost if s)
    for sid in only_regulators:
        for name, meaning in ((f"k_supply_{sid}", f"assumed basal supply of {sid}"),
                              (f"k_turnover_{sid}", f"assumed turnover of {sid}")):
            if name not in params:
                ir.setdefault("parameters", []).append({"name": name, "value": 0.1, "source": "default",
                                                        "unit": "arbitrary", "meaning": meaning})
                params.add(name)
        processes.append({"id": f"auto_supply_{sid}", "kind": "production", "target": sid, "k": f"k_supply_{sid}",
                          "evidence": "", "assumed": True,
                          "reason": f"{sid} is named only as a regulator; a basal supply keeps it defined"})
        processes.append({"id": f"auto_turnover_{sid}", "kind": "degradation", "species": sid,
                          "k": f"k_turnover_{sid}", "evidence": "", "assumed": True,
                          "reason": "first-order turnover keeps the supplied regulator bounded"})
        ir.setdefault("assumptions", []).append(
            f"{sid} has basal supply and turnover (steady level 1.0) because its source was not described.")
        sourced.add(sid)
        lost.add(sid)
    for sid in sorted(s for s in (regulators & lost) - sourced if s):
        name = f"k_supply_{sid}"
        if name not in params:
            ir.setdefault("parameters", []).append({"name": name, "value": 0.1, "source": "default",
                                                    "unit": "arbitrary", "meaning": f"assumed basal supply of {sid}"})
            params.add(name)
        processes.append({"id": f"auto_supply_{sid}", "kind": "production", "target": sid, "k": name,
                          "evidence": "", "assumed": True,
                          "reason": f"{sid} regulates the model but only its loss was described"})
        ir.setdefault("assumptions", []).append(
            f"{sid} has a basal supply because only its loss was described; without it {sid} would decay to zero.")
    # Boundedness, the same rule the deterministic extractor applies: a species that is produced
    # but never lost (no degradation, never consumed) grows without limit - the AI's p53-Mdm2
    # model let Mdm2 climb linearly for the whole run. Add disclosed first-order turnover.
    produced = {p.get("target") for p in processes if p.get("kind") == "production"}
    consumed = {p.get("species") for p in processes if p.get("kind") == "degradation"}
    consumed |= {e.get("species") for p in processes if p.get("kind") in ("conversion", "custom", "binding")
                 for e in (p.get("reactants") or [])}
    for sid in sorted(s for s in produced - consumed if s):
        name = f"k_turnover_{sid}"
        if name not in params:
            ir.setdefault("parameters", []).append({"name": name, "value": 0.1, "source": "default",
                                                    "unit": "1/time", "meaning": f"assumed turnover of {sid}"})
            params.add(name)
        processes.append({"id": f"auto_turnover_{sid}", "kind": "degradation", "species": sid, "k": name,
                          "evidence": "", "assumed": True,
                          "reason": f"{sid} is produced but no loss was described; turnover keeps it bounded"})
        ir.setdefault("assumptions", []).append(f"{sid} has first-order turnover (rate 0.1) because no loss was described.")
    ir["processes"] = processes


LLM_TIME_BUDGET_S = 150.0
# A complete IR for a 6-10 species description is ~2-3k tokens. Strict JSON-schema decoding on
# Purdue's gpt-oss:120b occasionally degenerates into thousands of whitespace tokens until the
# output limit (measured: 8192 tokens, 50 s, 56% whitespace, then truncated). A tighter cap bounds
# that failure's cost, and the retry below switches to plain JSON mode, which did not degenerate.
IR_MAX_TOKENS = 4500


def extract_ir_llm(text: str, client: Any, time_budget_s: float = LLM_TIME_BUDGET_S) -> Dict[str, Any]:
    """Extract and repair IR (up to two rounds); fall back honestly to rules.

    A transport-level failure (engine down, rate-limited, timed out) ends the loop at once:
    repeating the same request cannot fix it, and each provider call already retries with
    backoff. A truncated (degenerate) response is retried once in plain JSON mode. Content
    failures (invalid IR) get up to two repair rounds within the time budget.
    """
    generate_structured = getattr(llm_provider, "generate_structured", None)
    prompt = _PROMPT.replace("<<<TEXT>>>", text)
    last_error = ""
    raw: Any = None
    started = time.monotonic()
    attempts_made = 0
    strict = generate_structured is not None
    for attempt in range(3):
        if attempt and time.monotonic() - started > time_budget_s:
            last_error += f" (repair stopped: {time_budget_s:.0f} s budget used)"
            break
        attempts_made = attempt + 1
        try:
            if strict:
                raw = generate_structured(client, prompt, IR_SCHEMA, system=_SYSTEM, temperature=0.0,
                                          max_tokens=IR_MAX_TOKENS, name="biological_model_ir")
            else:
                raw = llm_provider.generate_json(client, prompt + _schema_text(), system=_SYSTEM,
                                                 temperature=0.0, max_tokens=IR_MAX_TOKENS)
            ir = _auto_repair_ir(_normalise_ir(raw), text)
            errors = validate_ir(ir, text)
            if not errors:
                try:
                    candidate_bp = compile_ir(ir)
                    candidate_verification = verify(candidate_bp, ir, text)
                    errors = list(candidate_verification.get("errors", []))
                except Exception as exc:
                    errors = [f"Deterministic compilation failed: {type(exc).__name__}: {exc}"]
            if not errors:
                ir["_extraction_notice"] = f"LLM IR validated after {attempt} repair round(s)."
                ir["_llm_attempts"] = attempts_made
                ir["_llm_seconds"] = round(time.monotonic() - started, 1)
                return ir
            last_error = "; ".join(errors)
        except llm_provider.LLMError as exc:
            message = str(exc)
            if "truncat" in message.lower() and attempt < 2:
                # Degenerate strict decoding: retry the SAME request in plain JSON mode.
                last_error = "the model's structured output ran past its length limit"
                strict = False
                continue
            last_error = f"the AI engine was unavailable ({message})"
            break
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        if attempt < 2:
            prompt = ("Repair the previous IR. Return the complete corrected IR only. Do not change supported biology.\n"
                      f"DESCRIPTION:\n{text}\nVALIDATION ERRORS:\n{last_error}\nPREVIOUS IR:\n"
                      f"{json.dumps(raw, default=str)[:12000]}" + _schema_text())
    fallback = extract_ir_rules(text)
    fallback["_extraction_notice"] = (f"LLM IR rejected after {attempts_made} attempt(s) ({last_error[:600]}); "
                                      f"deterministic rules used.")
    fallback["_llm_attempts"] = attempts_made
    fallback["_llm_seconds"] = round(time.monotonic() - started, 1)
    fallback["_llm_fallback"] = True
    return fallback


def _safe_id(name: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_]", "_", name.strip())
    if not value or not value[0].isalpha() and value[0] != "_": value = "S_" + value
    return value


def _split_entities(chunk: str) -> List[str]:
    chunk = re.sub(r"\b(?:the|a|an)\b", " ", chunk, flags=re.I)
    return [_safe_id(x) for x in re.split(r"\s*(?:,|\band\b|\bor\b)\s*", chunk, flags=re.I)
            if x.strip() and re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", x.strip())]


_CARRY_WORDS = {"and", "also", "which", "that", "it", "then", "but"}
_CLAUSE_SUBJECT = re.compile(
    r"\b([A-Za-z][A-Za-z0-9_-]*)\s+(?:is\s+|are\s+)?(?:activates?|induces?|stimulates?|phosphorylates?|"
    r"upregulates?|promotes?|inhibits?|represses?|blocks?|suppresses?|downregulates?|diffuses?|binds?|"
    r"degrad(?:ed|es)|produced|synthesi[sz]ed|transcribed|converted|starts?)\b", re.I)


def _is_species_word(word: str) -> bool:
    return bool(word) and (word.lower() not in _STOP or (len(word) == 1 and word.isupper())) \
        and word.lower() not in _CARRY_WORDS


def _carry_subject(original: str, start: int, raw: str) -> Optional[str]:
    """Resolve a clause subject such as "and" in "X represses itself and is degraded ..." to the
    subject of the preceding clause in the same sentence; None when there is nothing to carry."""
    if raw.lower() not in _CARRY_WORDS:
        return raw if _is_species_word(raw) else None
    sentence_start = max(original.rfind(".", 0, start), original.rfind(";", 0, start)) + 1
    subjects = [s.group(1) for s in _CLAUSE_SUBJECT.finditer(original[sentence_start:start])
                if _is_species_word(s.group(1))]
    return subjects[-1] if subjects else None


def extract_ir_rules(text: str) -> Dict[str, Any]:
    """Deterministic common-pattern extractor producing the same strict IR."""
    original = (text or "").strip()
    lower = original.lower()
    time_unit = "min" if re.search(r"\bmin(?:ute)?s?\b", lower) else "h" if re.search(r"\bhours?\b", lower) else "s" if re.search(r"\bsec(?:ond)?s?\b", lower) else "day" if re.search(r"\bdays?\b", lower) else "arbitrary"
    is_pde = bool(re.search(r"\b(diffus\w*|spatial|turing|reaction[- ]diffusion|pattern)\b", lower))
    ir = _empty_ir("reaction_diffusion" if is_pde else "ode", time_unit)
    if not original:
        ir["unmodeled"].append("empty description"); return ir
    species: Dict[str, Dict[str, Any]] = {}
    params: Dict[str, Dict[str, Any]] = {}
    processes: List[Dict[str, Any]] = []
    explicit_initials = set(); pid = 0; pcount = Counter()

    def add_species(name: str, initial: float = 0.1, source: str = "default", diffusion: Optional[float] = None) -> str:
        sid = _safe_id(name)
        # Capital single-letter identifiers (A, B, X) are standard model species;
        # relation regexes already constrain slots, so do not apply prose stopwords here.
        if not sid: return sid
        if sid not in species:
            species[sid] = {"id": sid, "name": name, "initial": float(initial), "initial_source": source,
                            "diffusion": diffusion, "role": "state"}
        elif diffusion is not None: species[sid]["diffusion"] = float(diffusion)
        if source == "text": species[sid].update({"initial": float(initial), "initial_source": "text"}); explicit_initials.add(sid)
        return sid

    def add_param(prefix: str, value: float, source: str, meaning: str, unit: str = "arbitrary") -> str:
        base = _safe_id(prefix); pcount[base] += 1; name = base if pcount[base] == 1 else f"{base}_{pcount[base]}"
        params[name] = {"name": name, "value": float(value), "source": source, "unit": unit, "meaning": meaning}
        return name

    def evidence_for(match: re.Match) -> str:
        return match.group(0).strip(" .;,:")

    def add_process(kind: str, evidence: str, assumed: bool = False, reason: str = "", **fields: Any) -> Dict[str, Any]:
        nonlocal pid; pid += 1
        proc = {"id": f"p{pid}", "kind": kind, **fields, "evidence": evidence, "assumed": assumed}
        if assumed: proc["reason"] = reason
        processes.append(proc); return proc

    # Initial values: 'X starts at 2', 'initial X = 2', and 'X = 2 nM'.
    init_patterns = [r"\b([A-Za-z][A-Za-z0-9_-]*)\s+starts?\s+at\s+([-+]?\d*\.?\d+(?:e[-+]?\d+)?)",
                     r"\binitial(?:\s+value\s+of)?\s+([A-Za-z][A-Za-z0-9_-]*)\s*(?:=|is)\s*([-+]?\d*\.?\d+(?:e[-+]?\d+)?)",
                     r"\b([A-Z][A-Za-z0-9_-]*)\s*=\s*([-+]?\d*\.?\d+(?:e[-+]?\d+)?)\s*(?:nM|uM|mM|units?)?"]
    for pattern in init_patterns:
        for m in re.finditer(pattern, original, re.I if "initial" in pattern or "starts" in pattern else 0):
            add_species(m.group(1), float(m.group(2)), "text")
    # Elliptical continuations: "S starts at 999, I at 1, R at 0" / "... and R at 0".
    _num = r"[-+]?\d*\.?\d+(?:e[-+]?\d+)?"
    for m in re.finditer(r"\bstarts?\s+at\s+" + _num + r"((?:\s*(?:,|;|\band\b)\s*(?:and\s+)?[A-Za-z][A-Za-z0-9_]*\s+at\s+" + _num + r")+)",
                         original, re.I):
        for tail in re.finditer(r"([A-Za-z][A-Za-z0-9_]*)\s+at\s+(" + _num + r")", m.group(1)):
            if tail.group(1).lower() not in _STOP:
                add_species(tail.group(1), float(tail.group(2)), "text")
    tm = re.search(r"(?:simulate|run|over|for)\s+(?:to\s+)?([-+]?\d*\.?\d+)\s*(?:s|sec(?:ond)?s?|min(?:ute)?s?|h|hours?|days?)", original, re.I)
    if tm: ir["t_end"] = float(tm.group(1))

    # Explicit dX/dt equations are already an unambiguous safe custom process.
    # Capture until the next equation, a following parameter assignment, or the
    # next prose sentence; decimals inside expressions remain intact.
    equation_pat = (r"d\s*([A-Za-z_][A-Za-z0-9_]*)\s*/\s*d\s*t\s*=\s*(.*?)"
                    r"(?=,\s*d\s*[A-Za-z_]|,\s*(?:with\s+)?[A-Za-z_]\w*\s*=|\.\s+[A-Z]|$)")
    equations = list(re.finditer(equation_pat, original, re.I))
    if equations:
        assignments = {m.group(1): float(m.group(2)) for m in re.finditer(
            r"\b([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([-+]?\d*\.?\d+(?:e[-+]?\d+)?)", original, re.I)}
        lhs_names = {m.group(1) for m in equations}
        for m in equations:
            target = add_species(m.group(1)); rate = m.group(2).strip().replace("^", "**")
            symbols = set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", rate))
            for symbol in sorted(symbols - {"exp", "log", "sin", "cos", "t"}):
                if symbol in lhs_names or (len(symbol) <= 3 and symbol.upper() == symbol and symbol not in assignments):
                    add_species(symbol)
                elif symbol not in params:
                    # An assigned symbol used in an equation is a parameter even
                    # when the generic initial-value regex previously saw `K=100`.
                    species.pop(symbol, None); explicit_initials.discard(symbol)
                    value = assignments.get(symbol, 1.0)
                    add_param(symbol, value, "text" if symbol in assignments else "default", f"parameter in d{target}/dt")
            add_process("custom", evidence_for(m), products=[{"species": target, "stoich": 1}],
                        reactants=[], rate=rate)

    # Canonical predator-prey prose has an unambiguous mass-action interpretation.
    lv = re.search(r"\bPrey\s+([A-Za-z_]\w*)\s+grows?\s+at\s+rate\s+([-+]?\d*\.?\d+).*?"
                   r"Predator\s+([A-Za-z_]\w*)\s+consumes?\s+\1\s+at\s+rate\s+([-+]?\d*\.?\d+).*?"
                   r"yield\s+([-+]?\d*\.?\d+).*?\3\s+dies?\s+at\s+rate\s+([-+]?\d*\.?\d+)", original, re.I)
    if lv:
        prey, predator = add_species(lv.group(1)), add_species(lv.group(3))
        grow = add_param(f"r_{prey}", float(lv.group(2)), "text", "prey growth rate")
        attack = add_param(f"a_{prey}_{predator}", float(lv.group(4)), "text", "predation rate")
        death = add_param(f"d_{predator}", float(lv.group(6)), "text", "predator death rate")
        add_process("custom", f"Prey {lv.group(1)} grows at rate {lv.group(2)}", products=[{"species": prey, "stoich": 1}],
                    reactants=[], rate=f"{grow}*{prey}")
        add_process("custom", f"Predator {lv.group(3)} consumes {lv.group(1)} at rate {lv.group(4)} with mass-action {lv.group(1)} times {lv.group(3)}, and {lv.group(3)} grows from consumption with yield {lv.group(5)}",
                    reactants=[{"species": prey, "stoich": 1}], products=[{"species": predator, "stoich": float(lv.group(5))}],
                    rate=f"{attack}*{prey}*{predator}")
        add_process("degradation", f"{lv.group(3)} dies at rate {lv.group(6)}", species=predator, k=death)


    # Diffusion declarations ("U diffuses slowly", "V diffuses quickly", "A diffuses at 0.05").
    diff_pat = (r"\b([A-Za-z][A-Za-z0-9_-]*)\s+diffuses?"
                r"((?:\s+(?:much\s+|more\s+|very\s+)?(?:slowly|quickly|rapidly|fast(?:er)?|slow(?:er)?|readily|freely))?)"
                r"(?:\s+(?:at|with)(?:\s+(?:a\s+)?(?:rate|coefficient|constant|D))?\s*(?:of\s+)?=?\s*([-+]?\d*\.?\d+))?")
    for m in re.finditer(diff_pat, original, re.I):
        if m.group(1).lower() in _STOP and not (len(m.group(1)) == 1 and m.group(1).isupper()):
            continue
        adverb = (m.group(2) or "").lower()
        value = (float(m.group(3)) if m.group(3)
                 else 1.0 if re.search(r"quick|fast|rapid|readily|freely", adverb) else 0.05)
        add_species(m.group(1), diffusion=value)
    for m in re.finditer(r"\bD[_-]?([A-Za-z][A-Za-z0-9_-]*)\s*=\s*([-+]?\d*\.?\d+)", original):
        add_species(m.group(1), diffusion=float(m.group(2)))

    # Binding first, because generic 'binds' must not become activation.
    binding_spans = []
    bind_pat = (r"\b([A-Za-z][A-Za-z0-9_-]*)\s+binds?(?:\s+to)?\s+([A-Za-z][A-Za-z0-9_-]*)"
                r"(?:\s+reversibly)?\s+(?:to\s+form|forming)\s+([A-Za-z][A-Za-z0-9_-]*)"
                r"(?:\s+with\s+kon\s*(?:=|of)?\s*[-+]?\d*\.?\d+\s+(?:and\s+)?koff\s*(?:=|of)?\s*[-+]?\d*\.?\d+)?")
    for m in re.finditer(bind_pat, original, re.I):
        a, b, c = [add_species(x) for x in m.group(1, 2, 3)]
        kon_m = re.search(r"\bkon\s*(?:=|of)?\s*([-+]?\d*\.?\d+)", m.group(0), re.I)
        koff_m = re.search(r"\bkoff\s*(?:=|of)?\s*([-+]?\d*\.?\d+)", m.group(0), re.I)
        kon = add_param(f"kon_{a}_{b}", float(kon_m.group(1)) if kon_m else 1.0, "text" if kon_m else "default", "association rate")
        koff = add_param(f"koff_{c}", float(koff_m.group(1)) if koff_m else 0.1, "text" if koff_m else "default", "dissociation rate")
        add_process("binding", evidence_for(m), reactants=[{"species": a, "stoich": 1}, {"species": b, "stoich": 1}], complex=c, kon=kon, koff=koff)
        binding_spans.append(m.span())

    # Catalysed and ordinary conversions.
    # Active voice with an explicit rate law: "infection converts S to I at rate beta*S*I with beta 0.3".
    # The rate expression is kept verbatim as a custom process between the named pools, so the
    # conversion conserves S + I by construction.
    active_conv = re.compile(
        r"\b(?:[A-Za-z]+\s+)?converts?\s+([A-Za-z][A-Za-z0-9_]*)\s+(?:in)?to\s+([A-Za-z][A-Za-z0-9_]*)"
        r"\s+at\s+(?:a\s+)?rate\s+(?:of\s+)?([A-Za-z0-9_.*/+\-()^ ]+?)"
        r"(?:\s+with\s+([A-Za-z_][A-Za-z0-9_]*)\s*(?:=|of|is)?\s*(" + _num + r"))?"
        r"(?=\s*(?:[,;]|\.\s|\.$|$|\band\b))", re.I)
    assigned = {m.group(1): float(m.group(2)) for m in re.finditer(
        r"\b([A-Za-z_][A-Za-z0-9_]*)\s*(?:=|\bof\b|\bis\b)?\s*(" + _num + r")\b", original)
        if m.group(1).lower() not in _STOP}
    for m in active_conv.finditer(original):
        source, target = add_species(m.group(1)), add_species(m.group(2))
        rate = m.group(3).strip().replace("^", "**")
        if m.group(4):
            assigned[m.group(4)] = float(m.group(5))
        if re.fullmatch(_num, rate):
            k = add_param(f"k_{source}_to_{target}", float(rate), "text", "conversion rate")
            add_process("conversion", evidence_for(m), reactants=[{"species": source, "stoich": 1}],
                        products=[{"species": target, "stoich": 1}], k=k)
            continue
        symbols_in_rate = set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", rate)) - {"exp", "log", "t"}
        for symbol in sorted(symbols_in_rate):
            if symbol in (source, target) or symbol in species:
                continue
            if symbol not in params:
                value = assigned.get(symbol)
                add_param(symbol, value if value is not None else 1.0, "text" if value is not None else "default",
                          f"rate constant in the {source}->{target} conversion")
        add_process("custom", evidence_for(m), reactants=[{"species": source, "stoich": 1}],
                    products=[{"species": target, "stoich": 1}], rate=rate)
    for m in re.finditer(r"\b([A-Za-z][A-Za-z0-9_-]*)\s+cataly[sz]es?(?:\s+the\s+conversion\s+of)?\s+([A-Za-z][A-Za-z0-9_-]*)\s+(?:in)?to\s+([A-Za-z][A-Za-z0-9_-]*)", original, re.I):
        enz, source, target = [add_species(x) for x in m.group(1, 2, 3)]
        k = add_param(f"kcat_{enz}_{source}", 1.0, "default", "catalytic rate")
        km = add_param(f"Km_{source}", 1.0, "default", "Michaelis constant")
        add_process("conversion", evidence_for(m), reactants=[{"species": source, "stoich": 1}], products=[{"species": target, "stoich": 1}], enzyme=enz, Km=km, k=k)
    conv_pat = r"\b([A-Za-z][A-Za-z0-9_-]*)\s+(?:is\s+)?(?:converted|phosphorylated|transported|pumped|exported|released)\s+(?:in)?to\s+([A-Za-z][A-Za-z0-9_-]*)(?:[^.]*?\bat\s+(?:a\s+)?rate\s+(?:of\s+)?([-+]?\d*\.?\d+))?"
    for m in re.finditer(conv_pat, original, re.I):
        source, target = add_species(m.group(1)), add_species(m.group(2))
        k = add_param(f"k_{source}_to_{target}", float(m.group(3)) if m.group(3) else 1.0, "text" if m.group(3) else "default", "conversion rate")
        add_process("conversion", evidence_for(m), reactants=[{"species": source, "stoich": 1}], products=[{"species": target, "stoich": 1}], k=k)
    for m in re.finditer(r"\b([A-Za-z][A-Za-z0-9_-]*)\s*(?:->|→)\s*([A-Za-z][A-Za-z0-9_-]*)", original):
        source, target = add_species(m.group(1)), add_species(m.group(2))
        k = add_param(f"k_{source}_to_{target}", 1.0, "default", "conversion rate")
        add_process("conversion", evidence_for(m), reactants=[{"species": source, "stoich": 1}], products=[{"species": target, "stoich": 1}], k=k)

    # Production and degradation sentences.
    for m in re.finditer(r"\b([A-Za-z][A-Za-z0-9_-]*)\s+(?:is\s+)?(?:produced|synthesi[sz]ed|transcribed)(?:\s+at\s+(?:a\s+)?(?:constant\s+)?rate(?:\s+of)?\s*([-+]?\d*\.?\d+))?", original, re.I):
        subject = _carry_subject(original, m.start(), m.group(1))
        if not subject:
            continue
        target = add_species(subject); value = float(m.group(2)) if m.group(2) else 1.0
        k = add_param(f"k_prod_{target}", value, "text" if m.group(2) else "default", f"production of {target}")
        add_process("production", evidence_for(m), target=target, k=k)
    degradation_targets = set()
    deg_pat = r"\b([A-Za-z][A-Za-z0-9_-]*)\s+(?:is\s+)?degrad(?:ed|es)(?:\s+with\s+(?:a\s+)?half[- ]life\s+of\s+([-+]?\d*\.?\d+)(?:\s+\w+)?)?(?:\s+at\s+(?:a\s+)?rate(?:\s+of)?\s*([-+]?\d*\.?\d+))?"
    for m in re.finditer(deg_pat, original, re.I):
        subject = _carry_subject(original, m.start(), m.group(1))
        if not subject:
            continue
        target = add_species(subject); degradation_targets.add(target)
        if m.group(2): add_process("degradation", evidence_for(m), species=target, half_life=float(re.match(r"[-+]?\d*\.?\d+", m.group(2)).group()))
        else:
            value = float(m.group(3)) if m.group(3) else 0.1
            k = add_param(f"k_deg_{target}", value, "text" if m.group(3) else "default", f"degradation of {target}")
            add_process("degradation", evidence_for(m), species=target, k=k)

    # Promoted degradation has different semantics from expression inhibition.
    # Both word orders: "X promotes the degradation of Y" and "X promotes Y degradation".
    promoted_spans = []
    _loss_noun = r"(?:degradation|turnover|decay|destruction|breakdown|proteolysis|ubiquitination|ubiquitylation|clearance)"
    promoted_pats = [
        r"\b([A-Za-z][A-Za-z0-9_-]*)\s+(?:promotes?|stimulates?|accelerates?|induces?|causes?|mediates?|triggers?|"
        r"enhances?|increases?|drives?)\s+(?:the\s+)?" + _loss_noun + r"\s+of\s+([A-Za-z][A-Za-z0-9_-]*)",
        r"\b([A-Za-z][A-Za-z0-9_-]*)\s+(?:promotes?|stimulates?|accelerates?|induces?|causes?|mediates?|triggers?|"
        r"enhances?|increases?|drives?)\s+(?:the\s+)?([A-Za-z][A-Za-z0-9_-]*)(?:'s)?\s+" + _loss_noun + r"\b",
    ]
    for pattern in promoted_pats:
        for m in re.finditer(pattern, original, re.I):
            regulator = _carry_subject(original, m.start(), m.group(1))
            if not regulator or not _is_species_word(m.group(2)):
                continue
            if any(a <= m.start() < b for a, b in promoted_spans):
                continue
            regulator, target = add_species(regulator), add_species(m.group(2))
            promoted_spans.append(m.span()); degradation_targets.add(target)
            k = add_param(f"k_deg_{target}", 0.1, "default", f"degradation of {target}")
            K = add_param(f"K_{regulator}_deg_{target}", 1.0, "default", "degradation regulator half-saturation")
            n = add_param(f"n_{regulator}_deg_{target}", 1.0, "default", "degradation regulator Hill coefficient")
            add_process("degradation", evidence_for(m), species=target, k=k,
                        regulators=[{"species": regulator, "effect": "activate", "K": K, "n": n}])

    # Spatial prose often carries the subject only once: "A diffuses ... and
    # activates its own expression and inhibitor I". Resolve that conjunction
    # before the generic matcher can mistake the word "and" for the subject.
    spatial_handled = []
    spatial_reg = (r"\b(?:activator\s+|inhibitor\s+)?([A-Za-z][A-Za-z0-9_-]*)\s+diffuses?.{0,100}?\band\s+"
                   r"(activates?|stimulates?|inhibits?|represses?)\s+(?:its\s+own\s+(?:expression|transcription)"
                   r"(?:\s+and\s+(?:activator\s+|inhibitor\s+)?([A-Za-z][A-Za-z0-9_-]*))?"
                   r"|(?:activator\s+|inhibitor\s+)?([A-Za-z][A-Za-z0-9_-]*))")
    for m in re.finditer(spatial_reg, original, re.I):
        source = add_species(m.group(1)); verb = m.group(2)
        effect = "repress" if re.search(r"inhibit|repress", verb, re.I) else "activate"
        targets = [source] if "its own" in m.group(0).lower() else []
        target_other = m.group(3) or m.group(4)
        if target_other: targets.append(add_species(target_other))
        for target in targets:
            k = add_param(f"k_prod_{target}", 1.0, "default", f"regulated production of {target}")
            K = add_param(f"K_{source}_to_{target}", 1.0, "default", "regulatory half-saturation")
            n = add_param(f"n_{source}_to_{target}", 2.0, "default", "regulatory Hill coefficient")
            add_process("production", evidence_for(m), target=target, k=k,
                        regulators=[{"species": source, "effect": effect, "K": K, "n": n}],
                        basal=0.05 if target == source else 0.0)
        spatial_handled.append(m.span())

    # "EGF binds to EGFR and activates it": the pronoun is the bound partner, and with no named
    # complex the binding is the mechanism of the stated regulation, not a separate species.
    pronoun_spans = []
    pronoun_pat = (r"\b([A-Za-z][A-Za-z0-9_-]*)\s+(?:binds?(?:\s+to)?|associates?\s+with|interacts?\s+with|"
                   r"recogni[sz]es?|engages?)\s+([A-Za-z][A-Za-z0-9_-]*)\s+and\s+(activates?|stimulates?|induces?|"
                   r"phosphorylates?|inhibits?|represses?|blocks?|inactivates?|suppresses?)\s+(?:it|them)\b")
    for m in re.finditer(pronoun_pat, original, re.I):
        if m.group(1).lower() in _STOP or m.group(2).lower() in _STOP:
            continue
        source, target = add_species(m.group(1)), add_species(m.group(2))
        effect = "repress" if re.search(r"inhibit|repress|block|inactivat|suppress", m.group(3), re.I) else "activate"
        k = add_param(f"k_prod_{target}", 1.0, "default", f"regulated production of {target}")
        K = add_param(f"K_{source}_to_{target}", 1.0, "default", "regulatory half-saturation")
        n = add_param(f"n_{source}_to_{target}", 2.0, "default", "regulatory Hill coefficient")
        add_process("production", evidence_for(m), target=target, k=k,
                    regulators=[{"species": source, "effect": effect, "K": K, "n": n}])
        pronoun_spans.append(m.span())
        ir["assumptions"].append(f"'{m.group(0)}': binding with no named complex is modelled as {source} "
                                 f"{'activating' if effect == 'activate' else 'repressing'} {target}.")

    # Self regulation and generic regulatory interactions become regulated production.
    self_spans = list(spatial_handled) + pronoun_spans
    self_pat = (r"\b([A-Za-z][A-Za-z0-9_-]*)\s+(represses?|inhibits?|activates?|stimulates?|induces?|promotes?)\s+"
                r"(?:its\s+own\s+(?:transcription|expression|production|synthesis)"
                r"|itself(?:'s)?(?:\s+(?:transcription|expression|production))?)"
                r"(?:\s+and\s+(?:the\s+)?(?:inhibitor\s+|activator\s+)?([A-Za-z][A-Za-z0-9_-]*))?")
    _verbish = re.compile(r"^(?:activ|inhib|repress|stimul|induc|promot|block|suppress|degrad|diffus|produc|"
                          r"decay|bind|is$|are$|was$|its$|also$)", re.I)
    for m in re.finditer(self_pat, original, re.I):
        raw_source = m.group(1)
        if raw_source.lower() in _STOP and not (len(raw_source) == 1 and raw_source.isupper()):
            continue
        target = add_species(raw_source); self_spans.append(m.span())
        effect = "repress" if re.search(r"repress|inhibit", m.group(2), re.I) else "activate"
        k = add_param(f"k_prod_{target}", 1.0, "default", f"production of {target}")
        K = add_param(f"K_{target}_self", 1.0, "default", "self-regulation half-saturation")
        hm = re.search(r"Hill coefficient(?:\s+of)?\s*([-+]?\d*\.?\d+)", m.group(0), re.I)
        n = add_param(f"n_{target}_self", float(hm.group(1)) if hm else 2.0, "text" if hm else "default", "self-regulation Hill coefficient")
        other = m.group(3)
        accepted_other = bool(other and other.lower() not in _STOP and not _verbish.match(other))
        evidence = evidence_for(m) if (accepted_other or not other) else \
            re.sub(r"\s+and\s*$", "", original[m.start():m.start(3)], flags=re.I).strip(" .;,:")
        add_process("production", evidence, target=target, k=k, basal=0.05,
                    regulators=[{"species": target, "effect": effect, "K": K, "n": n}])
        if accepted_other:
            second = add_species(other)
            k2 = add_param(f"k_prod_{second}", 1.0, "default", f"regulated production of {second}")
            K2 = add_param(f"K_{target}_to_{second}", 1.0, "default", "regulatory half-saturation")
            n2 = add_param(f"n_{target}_to_{second}", 2.0, "default", "regulatory Hill coefficient")
            add_process("production", evidence_for(m), target=second, k=k2,
                        regulators=[{"species": target, "effect": effect, "K": K2, "n": n2}])

    interaction_pat = r"\b([A-Za-z][A-Za-z0-9_-]*)\s+(activates?|induces?|stimulates?|phosphorylates?|upregulates?|promotes?|inhibits?|represses?|blocks?|suppresses?|downregulates?)\s+(?:the\s+)?(?:transcription|expression|production|activation)?\s*(?:of\s+)?([A-Za-z][A-Za-z0-9_-]*)"
    subject_pat = re.compile(r"\b([A-Za-z][A-Za-z0-9_-]*)\s+(?:activates?|induces?|stimulates?|phosphorylates?|"
                             r"upregulates?|promotes?|inhibits?|represses?|blocks?|suppresses?|downregulates?|"
                             r"diffuses?|binds?)\b", re.I)
    for m in re.finditer(interaction_pat, original, re.I):
        if re.search(r"\b(?:does|do|did)\s+not\b", original[max(0, m.start()-12):m.end()], re.I): continue
        if any(a <= m.start() < b for a, b in promoted_spans + self_spans): continue
        raw_source, raw_target = m.group(1), m.group(3)
        if raw_source.lower() in _CARRY_WORDS:
            # "A activates B and inhibits C": the subject carries over within the sentence.
            carried = _carry_subject(original, m.start(), raw_source)
            if not carried:
                continue
            raw_source = carried
        invalid_source = raw_source.lower() in _STOP and not (len(raw_source) == 1 and raw_source.isupper())
        invalid_target = raw_target.lower() in _STOP and not (len(raw_target) == 1 and raw_target.isupper())
        if raw_source.lower() in {"damage", "signal", "stimulus", "drug"}:
            continue  # handled below as an imposed stimulus, not a state species
        if invalid_source or invalid_target: continue
        source, target = add_species(raw_source), add_species(raw_target)
        if not source or not target: continue
        effect = "repress" if re.search(r"inhibit|repress|block|suppress|downregulat", m.group(2), re.I) else "activate"
        k = add_param(f"k_prod_{target}", 1.0, "default", f"regulated production of {target}")
        K = add_param(f"K_{source}_to_{target}", 1.0, "default", "regulatory half-saturation")
        hill_m = re.search(r"Hill coefficient(?:\s+of)?\s*([-+]?\d*\.?\d+)", original[m.end():m.end()+80], re.I)
        n = add_param(f"n_{source}_to_{target}", float(hill_m.group(1)) if hill_m else 2.0,
                      "text" if hill_m else "default", "regulatory Hill coefficient")
        add_process("production", evidence_for(m), target=target, k=k,
                    regulators=[{"species": source, "effect": effect, "K": K, "n": n}])

    # An imposed cause ("DNA damage stabilizes p53", "a drug activates X") is a stimulus regulator,
    # never a dynamic species. "Stabilizes" acts on the target's LOSS: it represses its degradation.
    for m in re.finditer(r"\b(?:(DNA|UV|oxidative|osmotic|heat|cold|mechanical|genotoxic)\s+)?"
                         r"(damage|signal|stimulus|drug|stress|light|shock|irradiation|radiation|hypoxia)\s+"
                         r"(activates?|stabili[sz]es?|increases?|induces?|stimulates?)\s+([A-Za-z][A-Za-z0-9_-]*)",
                         original, re.I):
        phrase = f"{m.group(1)}_{m.group(2)}" if m.group(1) else m.group(2).lower()
        name = _safe_id(phrase); target = add_species(m.group(4))
        verb = m.group(3)
        ir["stimuli"].append({"name": name, "profile": "constant", "level": 1.0, "t_on": None, "t_off": None, "evidence": evidence_for(m)})
        K = add_param(f"K_{name}_to_{target}", 1.0, "default", "stimulus half-saturation")
        n = add_param(f"n_{name}_to_{target}", 1.0, "default", "stimulus Hill coefficient")
        if re.match(r"stabili", verb, re.I):
            k = add_param(f"k_deg_{target}", 0.1, "default", f"stimulus-sensitive degradation of {target}")
            add_process("degradation", evidence_for(m), species=target, k=k,
                        regulators=[{"stimulus": name, "effect": "repress", "K": K, "n": n}])
            degradation_targets.add(target)
        else:
            k = add_param(f"k_prod_{target}", 1.0, "default", f"stimulus production of {target}")
            add_process("production", evidence_for(m), target=target, k=k,
                        regulators=[{"stimulus": name, "effect": "activate", "K": K, "n": n}])

    # Boundedness: genuinely produced states receive disclosed assumed turnover.
    # Conversion products are bounded by their finite source pool and must NOT get
    # an invented loss, which would break the mass conservation guaranteed above.
    produced = {p.get("target") for p in processes if p["kind"] == "production"}
    for sid in sorted(produced - degradation_targets):
        k = add_param(f"k_deg_{sid}", 0.1, "default", f"assumed turnover of {sid}")
        add_process("degradation", "", True, "first-order turnover added for boundedness", species=sid, k=k)
        degradation_targets.add(sid); ir["assumptions"].append(f"{sid} has first-order turnover for boundedness.")
    # Ensure source-only species participate dynamically without inventing an entity.
    participants = {sid for p in processes for field in ("reactants", "products") for sid, _ in _stoich(p.get(field))}
    participants |= {p.get("target") for p in processes if p.get("target")} | {p.get("species") for p in processes if p.get("species")}
    regulators = {r.get("species") for p in processes for r in p.get("regulators", []) if r.get("species")}
    for sid in sorted(regulators - participants):
        kprod = add_param(f"k_prod_{sid}", 0.1, "default", f"basal supply of regulator {sid}")
        kdeg = add_param(f"k_deg_{sid}", 0.1, "default", f"turnover of regulator {sid}")
        add_process("production", "", True, "basal supply preserves a regulator named without its source", target=sid, k=kprod)
        add_process("degradation", "", True, "first-order turnover added for boundedness", species=sid, k=kdeg)
        ir["assumptions"].append(f"{sid} has basal supply and turnover because its source was not described.")
    # A regulator whose LOSS is described but whose source is not ("Mdm2 promotes p53 degradation")
    # would simply decay to zero and take everything it regulates with it. It gets a disclosed
    # basal supply; its described loss is kept as written.
    sourced = {p.get("target") for p in processes if p["kind"] == "production"}
    sourced |= {sid for p in processes if p["kind"] in ("conversion", "custom") for sid, _ in _stoich(p.get("products"))}
    sourced |= {p.get("complex") for p in processes if p["kind"] == "binding"}
    for sid in sorted((regulators & participants) - sourced):
        if sid in degradation_targets:
            kprod = add_param(f"k_prod_{sid}", 0.1, "default", f"basal supply of regulator {sid}")
            add_process("production", "", True, "basal supply for a regulator whose source was not described",
                        target=sid, k=kprod)
            ir["assumptions"].append(f"{sid} has a basal supply because only its loss was described.")

    # Break exact symmetry in mutual/cyclic repression to expose the stated dynamics.
    repress_edges = [(next((r.get("species") for r in p.get("regulators", []) if r["effect"] == "repress"), None), p.get("target"))
                     for p in processes if p["kind"] == "production"]
    repress_edges = [(a, b) for a, b in repress_edges if a and b]
    if len(repress_edges) >= 2:
        nodes = sorted({x for edge in repress_edges for x in edge})
        if all(sid not in explicit_initials for sid in nodes):
            for i, sid in enumerate(nodes): species[sid]["initial"] = round(0.1 * (1 + 0.07 * i), 6)
            ir["assumptions"].append("Slightly unequal default initials break exact repression-network symmetry.")

    ir["species"] = list(species.values()); ir["parameters"] = list(params.values()); ir["processes"] = processes
    if not processes:
        ir["unmodeled"].append(original)
    return ir


def _process_signature(proc: Dict[str, Any]) -> Tuple[Any, ...]:
    regs = tuple(sorted((r.get("species") or r.get("stimulus"), r.get("effect")) for r in proc.get("regulators", [])))
    return (proc.get("kind"), proc.get("target"), proc.get("species"), tuple(_stoich(proc.get("reactants"))),
            tuple(_stoich(proc.get("products"))), proc.get("complex"), regs)


def _merge_rule_crosscheck(ir: Dict[str, Any], rules: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
    # Explicit equations and canonical mass-action predator/prey prose are more
    # constrained than a free-form LLM interpretation. Use the deterministic IR
    # wholesale when those rules found custom kinetics, and disclose the takeover.
    custom_rules = [p for p in rules.get("processes", []) if p.get("kind") == "custom" and not p.get("assumed")]
    if custom_rules:
        return copy.deepcopy(rules), len(custom_rules)

    merged = copy.deepcopy(ir); existing = {_process_signature(p) for p in merged.get("processes", [])}; added = 0
    needed_species = {s["id"]: s for s in rules.get("species", [])}; needed_params = {p["name"]: p for p in rules.get("parameters", [])}
    exact_by_lower = {sid.lower(): sid for sid in needed_species}
    rename = {s.get("id"): exact_by_lower.get(str(s.get("id", "")).lower())
              for s in merged.get("species", []) if exact_by_lower.get(str(s.get("id", "")).lower())}
    rename = {old: new for old, new in rename.items() if old and new and old != new}
    if rename:
        for state in merged.get("species", []):
            if state.get("id") in rename: state["id"] = rename[state["id"]]
        for proc in merged.get("processes", []):
            for field in ("target", "species", "enzyme", "complex"):
                if proc.get(field) in rename: proc[field] = rename[proc[field]]
            for field in ("reactants", "products"):
                for entry in proc.get(field, []) or []:
                    if entry.get("species") in rename: entry["species"] = rename[entry["species"]]
            for reg in proc.get("regulators", []):
                if reg.get("species") in rename: reg["species"] = rename[reg["species"]]
            if proc.get("kind") == "custom" and isinstance(proc.get("rate"), str):
                for old, new in rename.items(): proc["rate"] = re.sub(r"\b" + re.escape(old) + r"\b", new, proc["rate"])
        existing = {_process_signature(p) for p in merged.get("processes", [])}
    current_species = {s["id"] for s in merged.get("species", [])}; current_params = {p["name"] for p in merged.get("parameters", [])}
    for proc in rules.get("processes", []):
        if proc.get("assumed") or _process_signature(proc) in existing: continue
        refs = set()
        refs |= {x for x in (proc.get("target"), proc.get("species"), proc.get("enzyme"), proc.get("complex")) if x}
        refs |= {sid for field in ("reactants", "products") for sid, _ in _stoich(proc.get(field))}
        refs |= {r.get("species") for r in proc.get("regulators", []) if r.get("species")}
        for sid in refs - current_species:
            if sid in needed_species: merged["species"].append(copy.deepcopy(needed_species[sid])); current_species.add(sid)
        p_refs = {value for key, value in proc.items() if key in ("k", "Km", "kon", "koff") and isinstance(value, str)}
        p_refs |= {r[key] for r in proc.get("regulators", []) for key in ("K", "n") if key in r}
        for name in p_refs - current_params:
            if name in needed_params: merged["parameters"].append(copy.deepcopy(needed_params[name])); current_params.add(name)
        candidate = copy.deepcopy(proc); candidate["id"] = f"rule_{candidate['id']}"
        merged["processes"].append(candidate); existing.add(_process_signature(proc)); added += 1

    # If rules identify an imposed stimulus, do not retain a case-variant dynamic
    # species invented for the same word (Damage vs damage).
    stimulus_by_lower = {s["name"].lower(): s for s in rules.get("stimuli", [])}
    if stimulus_by_lower:
        existing_stimuli = {s.get("name", "").lower() for s in merged.get("stimuli", [])}
        for low, stimulus in stimulus_by_lower.items():
            if low not in existing_stimuli: merged.setdefault("stimuli", []).append(copy.deepcopy(stimulus))
        remove_species = {s["id"] for s in merged.get("species", []) if s.get("id", "").lower() in stimulus_by_lower}
        for proc in merged.get("processes", []):
            for reg in proc.get("regulators", []):
                old_species = reg.get("species")
                if old_species in remove_species:
                    reg.pop("species", None); reg["stimulus"] = stimulus_by_lower[str(old_species).lower()]["name"]
        merged["processes"] = [p for p in merged.get("processes", []) if not (p.get("assumed") and (p.get("target") in remove_species or p.get("species") in remove_species))]
        merged["species"] = [s for s in merged.get("species", []) if s.get("id") not in remove_species]
        exact_stimuli = {s.get("name", "").lower(): s.get("name") for s in merged.get("stimuli", [])}
        for proc in merged.get("processes", []):
            for reg in proc.get("regulators", []):
                if reg.get("stimulus") and str(reg["stimulus"]).lower() in exact_stimuli:
                    reg["stimulus"] = exact_stimuli[str(reg["stimulus"]).lower()]

    # An assumed loss on a finite conversion product destroys the conservation
    # guaranteed by stoichiometric assembly and is not needed for boundedness.
    conversion_products = {sid for p in merged.get("processes", []) if p.get("kind") in ("conversion", "binding")
                           for sid, _ in _stoich(p.get("products"))}
    conversion_products |= {p.get("complex") for p in merged.get("processes", []) if p.get("kind") == "binding"}
    production_targets = {p.get("target") for p in merged.get("processes", []) if p.get("kind") == "production"}
    merged["processes"] = [p for p in merged.get("processes", [])
                           if not (p.get("kind") == "degradation" and p.get("assumed") and
                                   p.get("species") in conversion_products - production_targets)]

    # Exact symmetry hides cyclic/mutual repression. Break it deterministically,
    # even when prose says all states start "near" the same value.
    repress_nodes = {x for p in merged.get("processes", []) if p.get("kind") == "production"
                     for r in p.get("regulators", []) if r.get("effect") == "repress"
                     for x in (r.get("species"), p.get("target")) if x}
    symmetric = [s for s in merged.get("species", []) if s.get("id") in repress_nodes]
    if len(symmetric) >= 2 and len({round(float(s.get("initial", 0)), 12) for s in symmetric}) == 1:
        base = max(0.1, float(symmetric[0].get("initial", 0.1)))
        for index, state in enumerate(sorted(symmetric, key=lambda x: x["id"])):
            state["initial"] = round(base * (1 + 0.07 * index), 8)
        merged.setdefault("assumptions", []).append("Slightly unequal initials break exact repression-network symmetry.")
    return merged, added


def compile_text(text: str, llm_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Compile text without raising on bad input; errors are returned in the result."""
    try:
        if not isinstance(text, str) or not text.strip():
            return {"validation_errors": ["Description must be non-empty text."], "_llm_notice": "No model was built."}
        engine = "rules"; notice = ""; rules = None; llm_meta: Dict[str, Any] = {}
        client = None
        if llm_config and llm_provider.wants_llm(llm_config):
            try:
                client = llm_provider.build_client(llm_config)
            except Exception as exc:
                reason = str(exc) if isinstance(exc, llm_provider.LLMError) else type(exc).__name__
                notice = (f"The selected AI engine is unavailable ({reason}). This model was built by the "
                          f"deterministic compiler instead - review it before relying on it.")
                llm_meta = {"requested_engine": str(llm_config.get("engine")), "fallback": True}
        if client is not None:
            ir = extract_ir_llm(text, client)
            engine = str(getattr(client, "active_model", None) or llm_config.get("model") or llm_config.get("engine") or "llm")
            notice = ir.pop("_extraction_notice", "")
            llm_meta = {"requested_engine": str(llm_config.get("engine")),
                        "attempts": ir.pop("_llm_attempts", None), "seconds": ir.pop("_llm_seconds", None),
                        "fallback": bool(ir.pop("_llm_fallback", False))}
            if llm_meta["fallback"]:
                engine = "rules"
            rules = extract_ir_rules(text)
            if not llm_meta["fallback"]:
                ir, added = _merge_rule_crosscheck(ir, rules)
                if added: notice += f" Rule cross-check restored {added} unambiguous interaction(s) omitted by the LLM."
        else:
            ir = extract_ir_rules(text)
        errors = validate_ir(ir, text)
        if errors and rules is not None:
            rule_errors = validate_ir(rules, text)
            if not rule_errors:
                notice += " The merged LLM IR failed validation; the validated deterministic IR was used instead."
                ir, errors = rules, []
                engine = "rules"
        if errors:
            return {"validation_errors": errors, "_ir": ir, "_compiler": COMPILER_VERSION, "_engine": engine,
                    "_llm_meta": llm_meta,
                    "_llm_notice": (notice + " No model was built because the IR did not validate.").strip()}
        bp = compile_ir(ir); verification = verify(bp, ir, text); bp["_verification"] = verification
        if verification["errors"] and rules is not None and ir is not rules:
            rule_errors = validate_ir(rules, text)
            if not rule_errors:
                rule_bp = compile_ir(rules); rule_verification = verify(rule_bp, rules, text)
                if not rule_verification["errors"]:
                    ir, bp, verification = rules, rule_bp, rule_verification
                    bp["_verification"] = verification
                    engine = "rules"
                    notice += " The LLM blueprint failed semantic verification; the verified deterministic blueprint was used instead."
        bp["_engine"] = engine
        bp["_llm_meta"] = llm_meta
        bp["_llm_notice"] = (notice + " " + bp.get("_llm_notice", "")).strip()
        if verification["errors"]:
            return {"validation_errors": verification["errors"], "_ir": ir, "_verification": verification,
                    "_compiler": COMPILER_VERSION, "_engine": engine, "_llm_meta": llm_meta,
                    "_llm_notice": bp["_llm_notice"]}
        return bp
    except Exception as exc:
        return {"validation_errors": [f"Compiler failed safely: {type(exc).__name__}: {exc}"],
                "_compiler": COMPILER_VERSION, "_engine": "rules" if not llm_config else str(llm_config.get("model") or "llm"),
                "_llm_notice": "No model was built; the failure was returned instead of raised."}


__all__ = ["IR_SCHEMA", "compile_text", "extract_ir_rules", "extract_ir_llm", "validate_ir", "compile_ir", "verify"]
