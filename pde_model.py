"""
PDE / model configuration.

Stage 4: define the fields solved on the mesh. Each field is a scalar
reaction-diffusion-advection equation

    du/dt = div(D grad u) - v . grad u + R(u, x, t) + S(x, t)

Expression safety
-----------------
User expressions are parsed with ``sympy.parsing.sympy_parser.parse_expr`` and an
EXPLICIT ``global_dict``, then checked so every free symbol is either a field, a
coordinate, time, or a declared parameter. Nothing is ever passed to ``eval`` or
``exec``.

This is deliberately stricter than a bare ``sympify(text)``: ``sympify`` falls back
to Python evaluation of the string against sympy's namespace, so an expression such
as a dunder attribute walk can reach objects that have nothing to do with algebra.
Restricting ``global_dict`` to the symbols we intend, and rejecting any leftover free
symbol, closes both the code-reach and the silent-typo cases -- a mistyped parameter
becomes an error instead of an implicit new unknown.

Numerical assumptions and limitations
-------------------------------------
* Diffusion is a single constant per field. Anisotropic or spatially varying
  diffusion is not implemented.
* Advection is a constant velocity vector.
* Fields are solved independently unless a reaction expression references another
  field by name; there is no implicit coupling.
"""

from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations

from geometry import ValidationIssue, domain_dimension

#: Functions a modelling expression may use. Everything else is rejected, which is
#: what keeps parsing inside algebra rather than inside Python.
ALLOWED_FUNCTIONS: Dict[str, Any] = {
    "exp": sp.exp, "log": sp.log, "ln": sp.log, "sqrt": sp.sqrt, "Abs": sp.Abs,
    "abs": sp.Abs, "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
    "sinh": sp.sinh, "cosh": sp.cosh, "tanh": sp.tanh,
    "atan": sp.atan, "asin": sp.asin, "acos": sp.acos,
    "Min": sp.Min, "Max": sp.Max, "min": sp.Min, "max": sp.Max,
    "pi": sp.pi, "E": sp.E,
    "Heaviside": sp.Heaviside,
}

#: The ONLY names ``parse_expr`` may resolve from its global scope.
#:
#: An empty global_dict does not work: ``standard_transformations`` rewrites numeric
#: literals into ``Integer(2)`` / ``Float(2.5)`` calls and unknown names into
#: ``Symbol('k')``, all of which are resolved against this dict -- so even "0" fails
#: to parse without them. Supplying exactly these four keeps sympy's several hundred
#: other callables, and every Python builtin, out of reach, while letting a bare name
#: become a SYMBOL that the free-symbol allow-list check below can then reject BY NAME.
_PARSE_GLOBALS: Dict[str, Any] = {
    "Integer": sp.Integer,
    "Float": sp.Float,
    "Rational": sp.Rational,
    "Symbol": sp.Symbol,
}

COORDINATES: Tuple[str, ...] = ("x", "y", "t")

FIELD_DEFAULTS: Dict[str, Any] = {
    "name": "u",
    "units": "",
    "initial": "0.0",
    "diffusion": 1.0,
    "advection": {"vx": 0.0, "vy": 0.0},
    "reaction": "0",
    "source": "0",
    "t_start": 0.0,
    "t_end": 10.0,
    "output_interval": 0.5,
}


# =============================================================================
# Presets
# =============================================================================
def preset_names() -> List[str]:
    return ["diffusion", "diffusion_decay", "reaction_diffusion",
            "advection_diffusion", "spatial_source"]


def get_preset(name: str) -> Dict[str, Any]:
    """A ready-to-run model configuration. Raises for an unknown name."""
    key = str(name or "").strip().lower()
    if key == "diffusion":
        return {
            "label": "Pure diffusion",
            "description": "A substance spreads out; nothing is created or destroyed.",
            "parameters": {"D": 1.0},
            "fields": [dict(FIELD_DEFAULTS, name="u", diffusion="D",
                            initial="exp(-((x-0.5)**2)/0.01)", reaction="0", source="0")],
        }
    if key == "diffusion_decay":
        return {
            "label": "Diffusion with decay",
            "description": "A substance spreads and is removed at a rate proportional "
                           "to its own concentration.",
            "parameters": {"D": 1.0, "k": 0.5},
            "fields": [dict(FIELD_DEFAULTS, name="u", diffusion="D",
                            initial="1.0", reaction="-k*u", source="0")],
        }
    if key == "reaction_diffusion":
        return {
            "label": "Reaction-diffusion (logistic growth)",
            "description": "A population spreads and grows logistically to a carrying "
                           "capacity.",
            "parameters": {"D": 0.1, "r": 1.0, "K": 1.0},
            "fields": [dict(FIELD_DEFAULTS, name="u", diffusion="D",
                            initial="exp(-((x-0.2)**2)/0.005)",
                            reaction="r*u*(1 - u/K)", source="0")],
        }
    if key == "advection_diffusion":
        return {
            "label": "Advection-diffusion",
            "description": "A substance is carried along by flow while it also spreads.",
            "parameters": {"D": 0.01, "v": 1.0},
            "fields": [dict(FIELD_DEFAULTS, name="u", diffusion="D",
                            advection={"vx": "v", "vy": 0.0},
                            initial="exp(-((x-0.3)**2)/0.005)",
                            reaction="0", source="0")],
        }
    if key == "spatial_source":
        return {
            "label": "Spatial source and sink",
            "description": "Material is injected in one region and removed everywhere.",
            "parameters": {"D": 0.1, "S0": 1.0, "k": 0.2},
            "fields": [dict(FIELD_DEFAULTS, name="u", diffusion="D", initial="0.0",
                            reaction="-k*u",
                            source="S0*exp(-((x-0.25)**2)/0.002)")],
        }
    raise ValueError(f"Unknown preset {name!r}. Available: {', '.join(preset_names())}.")


# =============================================================================
# Expression parsing
# =============================================================================
class ExpressionError(ValueError):
    """An expression could not be parsed, or used a symbol that is not defined."""


def parse_safe(text: Any, allowed: Sequence[str]) -> sp.Expr:
    """Parse an expression, permitting only ``allowed`` symbols and safe functions.

    Raises :class:`ExpressionError` with an actionable message, naming the unknown
    symbol and what IS available, because "invalid expression" alone leaves a user
    hunting a typo.
    """
    source = "0" if text is None else str(text).strip()
    if not source:
        source = "0"
    if "__" in source:
        # Dunder access has no algebraic meaning and is the classic route out of a
        # parser into the object graph.
        raise ExpressionError("Expressions may not contain '__'.")

    symbol_table: Dict[str, Any] = {name: sp.Symbol(name) for name in allowed}
    local_dict = {**ALLOWED_FUNCTIONS, **symbol_table}
    try:
        expression = parse_expr(
            source,
            local_dict=local_dict,
            # An explicit global_dict is the control: it holds only the numeric and
            # symbol constructors the transformations need, so no builtin and no
            # unlisted sympy callable can resolve.
            global_dict=_PARSE_GLOBALS,
            transformations=standard_transformations,
            evaluate=True,
        )
    except ExpressionError:
        raise
    except Exception as exc:
        raise ExpressionError(f"Could not read the expression {source!r}: "
                              f"{type(exc).__name__}.") from None

    if not isinstance(expression, sp.Basic):
        raise ExpressionError(f"The expression {source!r} is not a mathematical expression.")

    unknown = sorted(str(s) for s in expression.free_symbols if str(s) not in set(allowed))
    if unknown:
        raise ExpressionError(
            f"The expression {source!r} uses undefined name(s): {', '.join(unknown)}. "
            f"Available: {', '.join(sorted(allowed)) or 'none'}. Declare them as "
            f"parameters, or check the spelling."
        )
    return expression


def lambdify_scalar(expression: sp.Expr, arguments: Sequence[str]) -> Callable:
    """Turn a parsed expression into a NumPy callable over the given argument names."""
    symbols = [sp.Symbol(name) for name in arguments]
    func = sp.lambdify(symbols, expression, "numpy")

    def wrapped(*values):
        result = func(*values)
        # A constant expression lambdifies to a scalar; broadcast it so callers can
        # always treat the result as an array.
        reference = next((v for v in values if isinstance(v, np.ndarray)), None)
        if reference is not None and not isinstance(result, np.ndarray):
            return np.full_like(reference, float(result), dtype=float)
        return result

    return wrapped


# =============================================================================
# Validation
# =============================================================================
def allowed_symbols(model: Dict[str, Any], domain: Optional[Dict[str, Any]] = None) -> Set[str]:
    names: Set[str] = set(COORDINATES)
    if domain is not None and domain_dimension(domain) == 1:
        names.discard("y")
    names |= {str(f.get("name")) for f in (model.get("fields") or [])
              if isinstance(f, dict) and f.get("name")}
    names |= {str(k) for k in (model.get("parameters") or {})}
    return names


def validate_model(model: Any, domain: Optional[Dict[str, Any]] = None) -> List[ValidationIssue]:
    """Validate field definitions, parameters and every expression."""
    issues: List[ValidationIssue] = []
    if not isinstance(model, dict):
        return [ValidationIssue("error", "model_not_object",
                                "The model must be an object.", "model")]

    parameters = model.get("parameters")
    if parameters is not None and not isinstance(parameters, dict):
        issues.append(ValidationIssue("error", "parameters_not_object",
                                      "'parameters' must be a name-to-number mapping.",
                                      "model.parameters"))
        parameters = {}
    for name, value in (parameters or {}).items():
        if not str(name).isidentifier():
            issues.append(ValidationIssue(
                "error", "parameter_name_invalid",
                f"Parameter name {name!r} is not a valid identifier.",
                f"model.parameters.{name}"))
        try:
            if not np.isfinite(float(value)):
                raise ValueError
        except (TypeError, ValueError):
            issues.append(ValidationIssue("error", "parameter_value_invalid",
                                          f"Parameter {name!r} must be a finite number.",
                                          f"model.parameters.{name}"))

    fields = model.get("fields")
    if not isinstance(fields, list) or not fields:
        issues.append(ValidationIssue("error", "no_fields",
                                      "Define at least one field to solve.", "model.fields"))
        return issues

    names: Set[str] = set()
    for index, field in enumerate(fields):
        path = f"model.fields[{index}]"
        if not isinstance(field, dict):
            issues.append(ValidationIssue("error", "field_invalid",
                                          "Each field must be an object.", path))
            continue
        name = str(field.get("name") or "").strip()
        if not name:
            issues.append(ValidationIssue("error", "field_name_missing",
                                          "Every field needs a name.", f"{path}.name"))
            continue
        if not name.isidentifier():
            issues.append(ValidationIssue(
                "error", "field_name_invalid",
                f"Field name {name!r} must be a valid identifier so it can appear in "
                f"expressions.", f"{path}.name"))
        if name in COORDINATES:
            issues.append(ValidationIssue(
                "error", "field_name_reserved",
                f"{name!r} is reserved for a coordinate; choose another field name.",
                f"{path}.name"))
        if name in names:
            issues.append(ValidationIssue("error", "field_name_duplicate",
                                          f"Duplicate field name {name!r}.", f"{path}.name"))
        names.add(name)
        if not str(field.get("units") or "").strip():
            issues.append(ValidationIssue(
                "warning", "field_units_missing",
                f"Field {name!r} has no units. Scientific values should carry units.",
                f"{path}.units"))

    allowed = allowed_symbols(model, domain)

    for index, field in enumerate(fields):
        if not isinstance(field, dict) or not field.get("name"):
            continue
        path = f"model.fields[{index}]"
        name = str(field["name"])

        for key, label in (("initial", "Initial value"), ("reaction", "Reaction term"),
                           ("source", "Source term"), ("diffusion", "Diffusion coefficient")):
            raw = field.get(key, FIELD_DEFAULTS.get(key))
            try:
                expression = parse_safe(raw, allowed)
            except ExpressionError as exc:
                issues.append(ValidationIssue("error", f"field_{key}_invalid",
                                              f"{label} for {name!r}: {exc}", f"{path}.{key}"))
                continue
            # Diffusion must not depend on the solution or on time: the solver treats
            # it as a constant, and pretending otherwise would silently mis-solve.
            if key == "diffusion":
                offending = sorted(str(s) for s in expression.free_symbols
                                   if str(s) in names | {"t"})
                if offending:
                    issues.append(ValidationIssue(
                        "error", "field_diffusion_not_constant",
                        f"The diffusion coefficient for {name!r} depends on "
                        f"{', '.join(offending)}. This solver supports a constant "
                        f"coefficient only.", f"{path}.diffusion"))
                else:
                    value = _numeric_value(expression, model.get("parameters") or {})
                    if value is not None and value < 0:
                        issues.append(ValidationIssue(
                            "error", "field_diffusion_negative",
                            f"The diffusion coefficient for {name!r} evaluates to "
                            f"{value:g}. A negative coefficient makes the problem "
                            f"ill-posed (it runs time backwards).", f"{path}.diffusion"))

        advection = field.get("advection") or {}
        if not isinstance(advection, dict):
            issues.append(ValidationIssue("error", "field_advection_invalid",
                                          "'advection' must be an object with vx and vy.",
                                          f"{path}.advection"))
        else:
            for axis in ("vx", "vy"):
                if axis not in advection:
                    continue
                try:
                    expression = parse_safe(advection.get(axis), allowed)
                except ExpressionError as exc:
                    issues.append(ValidationIssue("error", f"field_advection_{axis}_invalid",
                                                  f"Advection {axis} for {name!r}: {exc}",
                                                  f"{path}.advection.{axis}"))
                    continue
                offending = sorted(str(s) for s in expression.free_symbols
                                   if str(s) in names | set(COORDINATES))
                if offending:
                    issues.append(ValidationIssue(
                        "error", "field_advection_not_constant",
                        f"Advection {axis} for {name!r} depends on "
                        f"{', '.join(offending)}; a constant velocity is required.",
                        f"{path}.advection.{axis}"))

        t_start = _as_float(field.get("t_start", 0.0))
        t_end = _as_float(field.get("t_end", 0.0))
        interval = _as_float(field.get("output_interval", 0.0))
        if t_start is None or t_end is None:
            issues.append(ValidationIssue("error", "field_time_invalid",
                                          f"Start and end time for {name!r} must be numbers.",
                                          f"{path}.t_end"))
        elif t_end <= t_start:
            issues.append(ValidationIssue(
                "error", "field_time_range_invalid",
                f"End time ({t_end:g}) must be after the start time ({t_start:g}) "
                f"for {name!r}.", f"{path}.t_end"))
        if interval is None or interval <= 0:
            issues.append(ValidationIssue("error", "field_output_interval_invalid",
                                          f"Output interval for {name!r} must be positive.",
                                          f"{path}.output_interval"))
        elif t_start is not None and t_end is not None and interval > (t_end - t_start):
            issues.append(ValidationIssue(
                "warning", "field_output_interval_coarse",
                f"The output interval ({interval:g}) is longer than the whole run for "
                f"{name!r}, so only the final state will be saved.",
                f"{path}.output_interval"))
    return issues


def _as_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
        return out if np.isfinite(out) else None
    except (TypeError, ValueError):
        return None


def _numeric_value(expression: sp.Expr, parameters: Dict[str, Any]) -> Optional[float]:
    """Evaluate a parameter-only expression to a number, or return None."""
    try:
        substituted = expression.subs({sp.Symbol(k): float(v) for k, v in parameters.items()})
        if substituted.free_symbols:
            return None
        return float(substituted)
    except (TypeError, ValueError, AttributeError):
        return None


# =============================================================================
# Compilation
# =============================================================================
def compile_field(model: Dict[str, Any], field_name: str,
                  domain: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Turn one field's configuration into numbers and callables for a solver.

    Raises on invalid configuration -- a caller that catches this must not report the
    run as ready.
    """
    errors = [i for i in validate_model(model, domain) if i.severity == "error"]
    if errors:
        raise ValueError("The model is not valid: " + "; ".join(i.message for i in errors))

    fields = {str(f["name"]): f for f in model["fields"] if isinstance(f, dict) and f.get("name")}
    if field_name not in fields:
        raise KeyError(f"No field {field_name!r}. Defined: {', '.join(sorted(fields))}.")
    field = fields[field_name]
    parameters = {str(k): float(v) for k, v in (model.get("parameters") or {}).items()}
    allowed = allowed_symbols(model, domain)
    field_names = sorted(fields)

    def constant(raw: Any, default: float = 0.0) -> float:
        expression = parse_safe(raw if raw is not None else default, allowed)
        value = _numeric_value(expression, parameters)
        if value is None:
            raise ValueError(f"{raw!r} does not reduce to a number for field {field_name!r}.")
        return value

    diffusion = constant(field.get("diffusion", 1.0))
    advection = field.get("advection") or {}
    vx = constant(advection.get("vx", 0.0))
    vy = constant(advection.get("vy", 0.0))

    reaction_expression = parse_safe(field.get("reaction", "0"), allowed)
    source_expression = parse_safe(field.get("source", "0"), allowed)
    initial_expression = parse_safe(field.get("initial", "0"), allowed)

    substitutions = {sp.Symbol(k): v for k, v in parameters.items()}
    reaction_expression = reaction_expression.subs(substitutions)
    source_expression = source_expression.subs(substitutions)
    initial_expression = initial_expression.subs(substitutions)

    reaction_args = field_names + ["x", "y", "t"]
    reaction_func = lambdify_scalar(reaction_expression, reaction_args)
    source_func = lambdify_scalar(source_expression, ["x", "y", "t"])
    initial_func = lambdify_scalar(initial_expression, ["x", "y"])

    return {
        "name": field_name,
        "units": str(field.get("units") or ""),
        "diffusion": diffusion,
        "advection": {"vx": vx, "vy": vy},
        "t_start": float(field.get("t_start", 0.0)),
        "t_end": float(field.get("t_end", 10.0)),
        "output_interval": float(field.get("output_interval", 0.5)),
        "expressions": {
            "reaction": str(field.get("reaction", "0")),
            "source": str(field.get("source", "0")),
            "initial": str(field.get("initial", "0")),
        },
        "reaction": reaction_func,
        "source": source_func,
        "initial": initial_func,
        "field_order": field_names,
        "latex": equation_latex(field, parameters),
        "summary": plain_language_summary(field, parameters),
    }


# =============================================================================
# Summaries
# =============================================================================
def equation_latex(field: Dict[str, Any], parameters: Optional[Dict[str, Any]] = None) -> str:
    """LaTeX for the field's governing equation, for KaTeX rendering."""
    name = str(field.get("name", "u"))
    allowed = set(COORDINATES) | {name} | {str(k) for k in (parameters or {})}
    pieces: List[str] = []
    try:
        diffusion = sp.latex(parse_safe(field.get("diffusion", 1.0), allowed))
    except ExpressionError:
        diffusion = "D"
    pieces.append(rf"{diffusion} \nabla^2 {name}")

    advection = field.get("advection") or {}
    try:
        vx = parse_safe(advection.get("vx", 0.0), allowed)
        vy = parse_safe(advection.get("vy", 0.0), allowed)
        if vx != 0 or vy != 0:
            pieces.append(rf"- \left({sp.latex(vx)}\,\partial_x + "
                          rf"{sp.latex(vy)}\,\partial_y\right){name}")
    except ExpressionError:
        pass

    for key in ("reaction", "source"):
        try:
            expression = parse_safe(field.get(key, "0"), allowed)
        except ExpressionError:
            continue
        if expression != 0:
            pieces.append("+ " + sp.latex(expression))

    return rf"\frac{{\partial {name}}}{{\partial t}} = " + " ".join(pieces)


def plain_language_summary(field: Dict[str, Any],
                           parameters: Optional[Dict[str, Any]] = None) -> str:
    """A sentence a reader can check against their intent."""
    name = str(field.get("name", "u"))
    units = str(field.get("units") or "").strip()
    unit_text = f" ({units})" if units else ""
    parts = [f"{name}{unit_text} spreads by diffusion"]

    advection = field.get("advection") or {}
    try:
        allowed = set(COORDINATES) | {name} | {str(k) for k in (parameters or {})}
        vx = _numeric_value(parse_safe(advection.get("vx", 0.0), allowed), parameters or {})
        vy = _numeric_value(parse_safe(advection.get("vy", 0.0), allowed), parameters or {})
        if (vx or 0.0) or (vy or 0.0):
            parts.append(f"is carried by flow (vx={vx or 0:g}, vy={vy or 0:g})")
    except ExpressionError:
        pass

    reaction = str(field.get("reaction", "0")).strip()
    if reaction not in ("", "0", "0.0"):
        parts.append(f"changes locally according to {reaction}")
    source = str(field.get("source", "0")).strip()
    if source not in ("", "0", "0.0"):
        parts.append(f"is supplied by {source}")

    t_start = field.get("t_start", 0.0)
    t_end = field.get("t_end", 10.0)
    interval = field.get("output_interval", 0.5)
    tail = (f" Solved from t={float(t_start):g} to t={float(t_end):g}, "
            f"saving every {float(interval):g}.")
    if len(parts) == 1:
        return parts[0] + "." + tail
    return ", ".join(parts[:-1]) + f", and {parts[-1]}." + tail
