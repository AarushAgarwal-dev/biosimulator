"""Exact SBML (Level 2/3) -> BioSimulateAI blueprint import, with no approximation of the kinetics.

The previous importer kept only the species and drew every reaction as a generic Hill
"activation" edge, so a curated BioModels entry was replaced by a different model (the
Kholodenko 2000 MAPK oscillator, BIOMD0000000010, became a monotone cascade). This module keeps
the model: every kinetic law is translated from MathML, reactions are assembled with their
stoichiometry, species concentrations are divided by their compartment volume, and local
parameters, function definitions, assignment rules, rate rules and initial assignments are
honoured. What cannot be represented (events, delays, algebraic rules) is REPORTED in
`_llm_notice` / `_sbml_report`, never silently dropped.

Output: a custom-kinetics blueprint (`nodes`, `parameters`, `fluxes`, `odes`) that
simulation_engine.ODEModel integrates directly.
"""
from __future__ import annotations

import keyword
import math
import re
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Tuple

import sympy as sp

MATHML = "http://www.w3.org/1998/Math/MathML"

# Names the generated expressions use as functions/constants, plus Python keywords. An SBML id
# that collides with one is renamed so SymPy cannot misread it.
_RESERVED = {
    "exp", "log", "sqrt", "Piecewise", "Abs", "sin", "cos", "tan", "asin", "acos", "atan",
    "sinh", "cosh", "tanh", "floor", "ceiling", "Max", "Min", "And", "Or", "Not", "Xor", "Eq", "Ne",
    "Lt", "Le", "Gt", "Ge", "factorial", "pi", "E", "true", "false", "oo", "zoo", "nan", "t",
    "sec", "csc", "cot", "root", "Symbol", "Function", "Integer", "Float", "Rational",
} | set(keyword.kwlist)


class SBMLImportError(ValueError):
    """The document is not SBML this importer can translate."""


def _local(tag: str) -> str:
    return tag.split("}", 1)[-1] if "}" in tag else tag


def _children(el: ET.Element, name: str) -> List[ET.Element]:
    return [c for c in list(el) if _local(c.tag) == name]


def _child(el: Optional[ET.Element], name: str) -> Optional[ET.Element]:
    if el is None:
        return None
    for c in list(el):
        if _local(c.tag) == name:
            return c
    return None


def _find_all(el: ET.Element, path: List[str]) -> List[ET.Element]:
    nodes = [el]
    for name in path:
        nxt: List[ET.Element] = []
        for n in nodes:
            nxt.extend(_children(n, name))
        nodes = nxt
    return nodes


def _safe_id(raw: str, taken: Dict[str, str]) -> str:
    if raw in taken:
        return taken[raw]
    name = re.sub(r"[^A-Za-z0-9_]", "_", raw or "x")
    if not re.match(r"[A-Za-z_]", name):
        name = "x_" + name
    if name in _RESERVED:
        name = name + "_sb"
    base, k = name, 2
    while name in taken.values():
        name = f"{base}_{k}"
        k += 1
    taken[raw] = name
    return name


# ------------------------------------------------------------------ MathML -> SymPy string
class _MathML:
    def __init__(self, names: Dict[str, str], functions: Dict[str, Tuple[List[str], ET.Element]]):
        self.names = names
        self.functions = functions
        self.unsupported: List[str] = []

    def convert(self, math_el: ET.Element, local_names: Optional[Dict[str, str]] = None) -> str:
        body = [c for c in list(math_el) if _local(c.tag) not in ("annotation", "semantics")]
        if _local(math_el.tag) == "math" and len(body) == 1:
            return self._node(body[0], local_names or {})
        return self._node(math_el, local_names or {})

    def _name(self, raw: str, local_names: Dict[str, str]) -> str:
        raw = raw.strip()
        if raw in local_names:
            return local_names[raw]
        if raw in self.names:
            return self.names[raw]
        # Unknown identifier: keep a safe version; the caller registers it as a parameter.
        return _safe_id(raw, self.names)

    def _node(self, el: ET.Element, ln: Dict[str, str]) -> str:
        tag = _local(el.tag)
        if tag == "ci":
            return self._name(el.text or "", ln)
        if tag == "cn":
            return self._number(el)
        if tag == "csymbol":
            url = (el.get("definitionURL") or "").lower()
            if url.endswith("/time"):
                return "t"
            if url.endswith("/avogadro"):
                return "6.02214076e23"
            self.unsupported.append(f"csymbol {url or el.text}")
            raise SBMLImportError(f"unsupported MathML csymbol {url or el.text!r}")
        if tag in ("true", "false"):
            return "True" if tag == "true" else "False"
        if tag == "pi":
            return "pi"
        if tag == "exponentiale":
            return "E"
        if tag in ("infinity",):
            return "oo"
        if tag == "notanumber":
            return "nan"
        if tag == "piecewise":
            pieces = []
            for piece in _children(el, "piece"):
                parts = [c for c in list(piece)]
                pieces.append(f"({self._node(parts[0], ln)}, {self._node(parts[1], ln)})")
            other = _child(el, "otherwise")
            if other is not None:
                pieces.append(f"({self._node(list(other)[0], ln)}, True)")
            return "Piecewise(" + ", ".join(pieces) + ")"
        if tag == "apply":
            return self._apply(el, ln)
        if tag == "semantics":
            first = list(el)[0]
            return self._node(first, ln)
        raise SBMLImportError(f"unsupported MathML element <{tag}>")

    @staticmethod
    def _number(el: ET.Element) -> str:
        kind = el.get("type", "real")
        parts = [p.strip() for p in (el.itertext())]
        text = "".join(el.itertext()).strip()
        if kind == "e-notation":
            mant = (el.text or "").strip()
            sep = _child(el, "sep")
            exp_part = (sep.tail or "").strip() if sep is not None else "0"
            return f"({float(mant) * 10 ** float(exp_part)!r})"
        if kind == "rational":
            num = (el.text or "").strip()
            sep = _child(el, "sep")
            den = (sep.tail or "1").strip() if sep is not None else "1"
            return f"({float(num) / float(den)!r})"
        del parts
        value = float(text)
        return f"({value!r})" if value < 0 else repr(value)

    def _apply(self, el: ET.Element, ln: Dict[str, str]) -> str:
        kids = list(el)
        op_el, args_el = kids[0], kids[1:]
        op = _local(op_el.tag)
        # qualifiers (degree / logbase) are not operands
        degree = next((a for a in args_el if _local(a.tag) == "degree"), None)
        logbase = next((a for a in args_el if _local(a.tag) == "logbase"), None)
        operands = [a for a in args_el if _local(a.tag) not in ("degree", "logbase", "bvar")]
        args = [self._node(a, ln) for a in operands]
        wrap = lambda s: f"({s})"  # noqa: E731
        if op == "plus":
            return wrap(" + ".join(wrap(a) for a in args)) if args else "0"
        if op == "times":
            return wrap(" * ".join(wrap(a) for a in args)) if args else "1"
        if op == "minus":
            return wrap(f"-{wrap(args[0])}") if len(args) == 1 else wrap(f"{wrap(args[0])} - {wrap(args[1])}")
        if op == "divide":
            return wrap(f"{wrap(args[0])} / {wrap(args[1])}")
        if op == "power":
            return wrap(f"{wrap(args[0])}**{wrap(args[1])}")
        if op == "root":
            n = self._node(list(degree)[0], ln) if degree is not None else "2"
            return wrap(f"{wrap(args[0])}**(1/{wrap(n)})")
        if op == "exp":
            return f"exp({args[0]})"
        if op == "ln":
            return f"log({args[0]})"
        if op == "log":
            base = self._node(list(logbase)[0], ln) if logbase is not None else "10"
            return f"(log({args[0]})/log({base}))"
        if op in ("abs", "floor", "ceiling", "factorial", "sin", "cos", "tan", "arcsin", "arccos",
                  "arctan", "sinh", "cosh", "tanh"):
            fn = {"abs": "Abs", "arcsin": "asin", "arccos": "acos", "arctan": "atan"}.get(op, op)
            return f"{fn}({args[0]})"
        rel = {"eq": "Eq", "neq": "Ne", "gt": "Gt", "lt": "Lt", "geq": "Ge", "leq": "Le"}
        if op in rel:
            if len(args) == 2:
                return f"{rel[op]}({args[0]}, {args[1]})"
            return "And(" + ", ".join(f"{rel[op]}({a}, {b})" for a, b in zip(args, args[1:])) + ")"
        if op in ("and", "or", "xor"):
            return f"{op.capitalize()}(" + ", ".join(args) + ")"
        if op == "not":
            return f"Not({args[0]})"
        if op in ("max", "min"):
            return f"{op.capitalize()}(" + ", ".join(args) + ")"
        if op == "ci":
            fname = (op_el.text or "").strip()
            if fname in self.functions:
                params, body = self.functions[fname]
                bound = {p: f"__arg{i}" for i, p in enumerate(params)}
                inner = self.convert(body, bound)
                expr = sp.sympify(inner, locals={v: sp.Symbol(v) for v in bound.values()})
                subs = {sp.Symbol(f"__arg{i}"): sp.sympify(a, locals=_symbols_for(a)) for i, a in enumerate(args)}
                return wrap(str(expr.xreplace(subs)))
            raise SBMLImportError(f"call to undefined function {fname!r}")
        if op == "csymbol" and (op_el.get("definitionURL") or "").lower().endswith("/delay"):
            self.unsupported.append("delay")
            raise SBMLImportError("delay() is not supported by the ODE engine")
        raise SBMLImportError(f"unsupported MathML operator <{op}>")


def _symbols_for(expr: str) -> Dict[str, sp.Symbol]:
    names = set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", expr))
    known = {"exp", "log", "sqrt", "Piecewise", "Abs", "sin", "cos", "tan", "asin", "acos", "atan",
             "sinh", "cosh", "tanh", "floor", "ceiling", "Max", "Min", "And", "Or", "Not", "Xor", "Eq",
             "Ne", "Lt", "Le", "Gt", "Ge", "factorial", "pi", "E", "True", "False", "oo", "nan"}
    return {n: sp.Symbol(n) for n in names - known}


# ------------------------------------------------------------------ SBML -> blueprint
def _float(value: Optional[str], default: Optional[float] = None) -> Optional[float]:
    if value is None or str(value).strip() == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _time_horizon(model: ET.Element, root: ET.Element) -> Tuple[float, str]:
    unit = (model.get("timeUnits") or "").lower()
    level = int(root.get("level", "2"))
    if not unit:
        for ud in _find_all(model, ["listOfUnitDefinitions", "unitDefinition"]):
            if (ud.get("id") or "").lower() == "time":
                kinds = [u.get("kind", "").lower() for u in _find_all(ud, ["listOfUnits", "unit"])]
                mult = [float(u.get("multiplier", "1")) for u in _find_all(ud, ["listOfUnits", "unit"])]
                if kinds == ["second"]:
                    unit = "minute" if mult and abs(mult[0] - 60) < 1e-9 else \
                        "hour" if mult and abs(mult[0] - 3600) < 1e-9 else "second"
    if not unit and level < 3:
        unit = "second (SBML Level 2 default)"
    if "second" in unit or unit == "s":
        return 3600.0, unit
    if "minute" in unit or unit == "min":
        return 300.0, "minute"
    if "hour" in unit or unit == "h":
        return 48.0, "hour"
    return 100.0, unit or "unspecified"


def sbml_to_blueprint(sbml: str) -> Dict[str, Any]:
    """Translate an SBML document into a custom-kinetics blueprint. Raises SBMLImportError."""
    try:
        root = ET.fromstring(sbml)
    except ET.ParseError as exc:
        raise SBMLImportError(f"not well-formed XML: {exc}") from None
    if _local(root.tag) != "sbml":
        raise SBMLImportError("the document is not SBML (root element is not <sbml>)")
    model = _child(root, "model")
    if model is None:
        raise SBMLImportError("the SBML document has no <model>")
    level = int(root.get("level", "2"))
    names: Dict[str, str] = {}
    report: Dict[str, Any] = {"level": level, "version": root.get("version"), "unsupported": [], "notes": []}

    comps: Dict[str, float] = {}
    for c in _find_all(model, ["listOfCompartments", "compartment"]):
        cid = c.get("id") or c.get("name")
        size = _float(c.get("size"), _float(c.get("volume"), 1.0))
        comps[_safe_id(cid, names)] = size if size and size > 0 else 1.0

    params: Dict[str, float] = {}
    for p in _find_all(model, ["listOfParameters", "parameter"]):
        params[_safe_id(p.get("id") or p.get("name"), names)] = _float(p.get("value"), 0.0)

    species: Dict[str, Dict[str, Any]] = {}
    for s in _find_all(model, ["listOfSpecies", "species"]):
        sid = _safe_id(s.get("id") or s.get("name"), names)
        comp = _safe_id(s.get("compartment"), names) if s.get("compartment") else None
        vol = comps.get(comp, 1.0) if comp else 1.0
        only_substance = (s.get("hasOnlySubstanceUnits") or "false").lower() == "true"
        conc = _float(s.get("initialConcentration"))
        amount = _float(s.get("initialAmount"))
        if conc is None and amount is not None:
            conc = amount if only_substance else amount / vol
        species[sid] = {
            "sbml_id": s.get("id"), "name": s.get("name") or s.get("id"), "compartment": comp, "volume": vol,
            "initial": conc if conc is not None else 0.0, "only_substance": only_substance,
            "boundary": (s.get("boundaryCondition") or "false").lower() == "true",
            "constant": (s.get("constant") or "false").lower() == "true",
        }

    functions: Dict[str, Tuple[List[str], ET.Element]] = {}
    for f in _find_all(model, ["listOfFunctionDefinitions", "functionDefinition"]):
        math_el = _child(f, "math")
        lam = _child(math_el, "lambda") if math_el is not None else None
        if lam is None:
            continue
        bvars = [(_child(b, "ci").text or "").strip() for b in _children(lam, "bvar") if _child(b, "ci") is not None]
        body = [c for c in list(lam) if _local(c.tag) != "bvar"][0]
        functions[(f.get("id") or "").strip()] = (bvars, body)
    mm = _MathML(names, functions)

    fluxes: Dict[str, str] = {}
    contributions: Dict[str, List[str]] = {sid: [] for sid in species}
    edges: List[Dict[str, Any]] = []
    for idx, r in enumerate(_find_all(model, ["listOfReactions", "reaction"]), 1):
        rid = _safe_id(r.get("id") or f"reaction_{idx}", names)
        law = _child(r, "kineticLaw")
        math_el = _child(law, "math") if law is not None else None
        if math_el is None:
            report["unsupported"].append(f"reaction {r.get('id')} has no kinetic law and was skipped")
            continue
        local_names: Dict[str, str] = {}
        for lp in (_find_all(law, ["listOfParameters", "parameter"]) + _find_all(law, ["listOfLocalParameters", "localParameter"])):
            raw = lp.get("id") or lp.get("name")
            scoped = _safe_id(f"{rid}__{raw}", names)
            local_names[raw] = scoped
            params[scoped] = _float(lp.get("value"), 0.0)
        try:
            rate = mm.convert(math_el, local_names)
        except SBMLImportError as exc:
            report["unsupported"].append(f"reaction {r.get('id')}: {exc}")
            continue
        fluxes[f"v_{rid}"] = rate
        reversible = (r.get("reversible") or "true").lower() == "true"

        def refs(list_name: str) -> List[Tuple[str, float]]:
            out = []
            for ref in _find_all(r, [list_name, "speciesReference"]):
                sid = _safe_id(ref.get("species"), names)
                stoich = _float(ref.get("stoichiometry"), 1.0)
                if _child(ref, "stoichiometryMath") is not None:
                    report["unsupported"].append(f"reaction {r.get('id')}: stoichiometryMath treated as 1")
                out.append((sid, stoich if stoich is not None else 1.0))
            return out

        reactants, products = refs("listOfReactants"), refs("listOfProducts")
        for sid, nu in reactants:
            if sid in contributions:
                contributions[sid].append(f"-{nu!r}*v_{rid}")
        for sid, nu in products:
            if sid in contributions:
                contributions[sid].append(f"+{nu!r}*v_{rid}")
        for a, _ in reactants:
            for b, _ in products:
                edges.append({"source": a, "target": b, "type": "activation", "reaction": r.get("id"),
                              "reversible": reversible})
        for mod in _find_all(r, ["listOfModifiers", "modifierSpeciesReference"]):
            m = _safe_id(mod.get("species"), names)
            for b, _ in products:
                edges.append({"source": m, "target": b, "type": "association", "reaction": r.get("id")})

    # Rules.
    assignment: Dict[str, str] = {}
    rate_rules: Dict[str, str] = {}
    for rule in list(_child(model, "listOfRules") or []):
        kind = _local(rule.tag)
        var = rule.get("variable") or rule.get("species") or rule.get("compartment") or rule.get("name")
        math_el = _child(rule, "math")
        if math_el is None:
            continue
        try:
            expr = mm.convert(math_el)
        except SBMLImportError as exc:
            report["unsupported"].append(f"rule for {var}: {exc}")
            continue
        if kind in ("assignmentRule", "speciesConcentrationRule", "parameterRule", "compartmentVolumeRule") \
                and (rule.get("type") in (None, "scalar")):
            assignment[_safe_id(var, names)] = expr
        elif kind == "rateRule" or rule.get("type") == "rate":
            rate_rules[_safe_id(var, names)] = expr
        else:
            report["unsupported"].append(f"{kind} ({var}) is not supported")

    n_events = len(_find_all(model, ["listOfEvents", "event"]))
    if n_events:
        report["unsupported"].append(f"{n_events} event(s) were not imported (the ODE engine has no discrete events)")
    if _find_all(model, ["listOfConstraints", "constraint"]):
        report["notes"].append("SBML constraints are informational and were not enforced")

    # Initial assignments evaluated from the values known at t = 0.
    values = {**params, **comps, **{k: v["initial"] for k, v in species.items()}}
    for ia in _find_all(model, ["listOfInitialAssignments", "initialAssignment"]):
        sym = _safe_id(ia.get("symbol"), names)
        math_el = _child(ia, "math")
        try:
            expr = sp.sympify(mm.convert(math_el), locals=_symbols_for(mm.convert(math_el)))
            val = float(expr.subs({sp.Symbol(k): v for k, v in values.items()}).subs(sp.Symbol("t"), 0))
        except Exception as exc:  # noqa: BLE001
            report["unsupported"].append(f"initialAssignment for {ia.get('symbol')}: {type(exc).__name__}")
            continue
        values[sym] = val
        if sym in species:
            species[sym]["initial"] = val
        elif sym in params:
            params[sym] = val
        elif sym in comps:
            comps[sym] = val

    # Assemble ODEs. A kinetic law gives amount/time; a concentration species divides by volume.
    odes: Dict[str, str] = {}
    nodes: List[Dict[str, Any]] = []
    for sid, info in species.items():
        if sid in assignment:
            continue                                   # defined algebraically, below
        if sid in rate_rules:
            odes[sid] = rate_rules[sid]
        elif info["boundary"] or info["constant"]:
            params[sid] = info["initial"]              # held fixed: a parameter, not a state
            continue
        else:
            terms = contributions.get(sid) or []
            if not terms:
                odes[sid] = "0"
            else:
                total = " ".join(terms).lstrip("+")
                odes[sid] = total if (info["only_substance"] or info["compartment"] is None) \
                    else f"({total})/{info['compartment']}"
        nodes.append({"id": sid, "name": info["name"], "initial_value": float(info["initial"]),
                      "sbml_id": info["sbml_id"], "compartment": info["compartment"]})
    for var, expr in rate_rules.items():
        if var in species:
            continue
        initial = values.get(var, 0.0)
        params.pop(var, None)
        odes[var] = expr
        nodes.append({"id": var, "name": var, "initial_value": float(initial), "sbml_rate_rule": True})
    for var, expr in assignment.items():
        params.pop(var, None)
        fluxes[var] = expr
    for cid, size in comps.items():
        if cid not in assignment and cid not in rate_rules:
            params[cid] = size

    # Any identifier used but never declared becomes a parameter; report it.
    declared = set(params) | set(odes) | set(fluxes) | {"t"}
    undeclared = set()
    for expr in list(odes.values()) + list(fluxes.values()):
        for name in re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", expr):
            if name not in declared and name not in {"exp", "log", "sqrt", "Piecewise", "Abs", "sin", "cos",
                                                      "tan", "asin", "acos", "atan", "sinh", "cosh", "tanh",
                                                      "floor", "ceiling", "Max", "Min", "And", "Or", "Not",
                                                      "Xor", "Eq", "Ne", "Lt", "Le", "Gt", "Ge", "factorial",
                                                      "pi", "E", "True", "False", "oo", "nan"}:
                undeclared.add(name)
    for name in sorted(undeclared):
        params[name] = 1.0
    if undeclared:
        report["unsupported"].append(f"undeclared identifiers set to 1.0: {', '.join(sorted(undeclared))}")

    t_max, time_unit = _time_horizon(model, root)
    renamed = {raw: new for raw, new in names.items() if raw != new and raw}
    notice = (f"Imported the SBML model exactly: {len(nodes)} species, {len(fluxes)} rate laws, "
              f"{len(params)} parameters. Model time unit: {time_unit}; SBML stores no simulation horizon, "
              f"so t_max = {t_max:g} was chosen - adjust it in the Simulator.")
    if renamed:
        notice += f" Renamed identifiers: {', '.join(f'{a}->{b}' for a, b in list(renamed.items())[:8])}."
    if report["unsupported"]:
        notice += " NOT imported: " + "; ".join(report["unsupported"]) + "."
    return {
        "type": "ODE",
        "name": model.get("name") or model.get("id") or "SBML model",
        "nodes": nodes,
        "parameters": {k: float(v) for k, v in params.items()},
        "fluxes": fluxes,
        "odes": odes,
        "edges": edges,
        "simulation_config": {"t_max": t_max},
        "_compiler": "sbml_import/exact",
        "_sbml_report": {**report, "time_unit": time_unit, "renamed": renamed},
        "_llm_notice": notice,
    }


__all__ = ["sbml_to_blueprint", "SBMLImportError"]
