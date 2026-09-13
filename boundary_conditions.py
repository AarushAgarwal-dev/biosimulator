"""
Boundary and initial conditions.

Stage 5: conditions are attached to NAMED boundaries from the domain, never to raw
mesh node indices. That is what lets an assignment survive remeshing -- indices are
regenerated, the name "left" is not.

Supported condition types
-------------------------
=============  ====================================================================
 ``dirichlet``  fixed value:      u = g
 ``neumann``    prescribed flux:  -D du/dn = q      (q = 0 is the no-flux case)
 ``no_flux``    Neumann with q = 0, kept distinct so the UI can label it
 ``robin``      mixed:            -D du/dn = h (u - u_inf)
 ``periodic``   pairs two named boundaries: u and its flux match across them
=============  ====================================================================

A boundary carrying no condition is treated as no-flux by the solvers, which is the
conventional default for a closed domain; that default is reported in the summary so
it is a stated assumption rather than a silent one.
"""

from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from geometry import ValidationIssue

CONDITION_TYPES: Tuple[str, ...] = (
    "dirichlet", "neumann", "no_flux", "robin", "periodic",
)

#: Types that fix the value at the boundary. Two of these on one boundary for one
#: field is a contradiction, not a refinement.
VALUE_FIXING: Tuple[str, ...] = ("dirichlet",)
FLUX_FIXING: Tuple[str, ...] = ("neumann", "no_flux", "robin")


def make_dirichlet(field: str, boundary: str, value: float,
                   condition_id: Optional[str] = None, enabled: bool = True) -> Dict[str, Any]:
    return {"id": condition_id or f"bc_{field}_{boundary}_dirichlet",
            "type": "dirichlet", "field": field, "boundary": boundary,
            "value": float(value), "enabled": bool(enabled)}


def make_neumann(field: str, boundary: str, flux: float = 0.0,
                 condition_id: Optional[str] = None, enabled: bool = True) -> Dict[str, Any]:
    return {"id": condition_id or f"bc_{field}_{boundary}_neumann",
            "type": "neumann", "field": field, "boundary": boundary,
            "flux": float(flux), "enabled": bool(enabled)}


def make_no_flux(field: str, boundary: str, condition_id: Optional[str] = None,
                 enabled: bool = True) -> Dict[str, Any]:
    return {"id": condition_id or f"bc_{field}_{boundary}_noflux",
            "type": "no_flux", "field": field, "boundary": boundary,
            "flux": 0.0, "enabled": bool(enabled)}


def make_robin(field: str, boundary: str, transfer: float, ambient: float,
               condition_id: Optional[str] = None, enabled: bool = True) -> Dict[str, Any]:
    return {"id": condition_id or f"bc_{field}_{boundary}_robin",
            "type": "robin", "field": field, "boundary": boundary,
            "transfer_coefficient": float(transfer), "ambient_value": float(ambient),
            "enabled": bool(enabled)}


def make_periodic(field: str, boundary: str, partner: str,
                  condition_id: Optional[str] = None, enabled: bool = True) -> Dict[str, Any]:
    return {"id": condition_id or f"bc_{field}_{boundary}_periodic",
            "type": "periodic", "field": field, "boundary": boundary,
            "partner": partner, "enabled": bool(enabled)}


def enabled_conditions(conditions: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [c for c in (conditions or []) if isinstance(c, dict) and c.get("enabled", True)]


def conditions_for(conditions: Sequence[Dict[str, Any]], field: str,
                   boundary: str) -> List[Dict[str, Any]]:
    return [c for c in enabled_conditions(conditions)
            if str(c.get("field")) == str(field) and str(c.get("boundary")) == str(boundary)]


def validate_conditions(conditions: Any, domain: Dict[str, Any],
                        field_names: Sequence[str]) -> List[ValidationIssue]:
    """Check every condition, and detect contradictory assignments.

    Contradictions caught:
      * two value-fixing conditions on the same field and boundary,
      * a value-fixing and a flux-fixing condition on the same field and boundary
        (over-determined: the solver cannot honour both),
      * a periodic condition whose partner is missing, is itself, or is not
        reciprocated,
      * a periodic boundary that also carries any other condition.
    """
    issues: List[ValidationIssue] = []
    if conditions is None:
        return issues
    if not isinstance(conditions, list):
        return [ValidationIssue("error", "conditions_not_list",
                                "'conditions' must be a list.", "conditions")]

    known_boundaries = {str(b.get("id")) for b in (domain.get("boundaries") or [])
                        if isinstance(b, dict) and b.get("id")}
    known_fields = {str(f) for f in field_names}
    seen_ids: Dict[str, int] = {}
    # (field, boundary) -> list of (index, type)
    by_target: Dict[Tuple[str, str], List[Tuple[int, str]]] = {}
    periodic_pairs: Dict[Tuple[str, str], str] = {}

    for index, condition in enumerate(conditions):
        path = f"conditions[{index}]"
        if not isinstance(condition, dict):
            issues.append(ValidationIssue("error", "condition_invalid",
                                          "Each condition must be an object.", path))
            continue

        condition_id = str(condition.get("id") or "").strip()
        if not condition_id:
            issues.append(ValidationIssue("error", "condition_id_missing",
                                          "Every condition needs an id.", f"{path}.id"))
        elif condition_id in seen_ids:
            issues.append(ValidationIssue(
                "error", "condition_id_duplicate",
                f"Duplicate condition id {condition_id!r} (also at position "
                f"{seen_ids[condition_id] + 1}).", f"{path}.id"))
        else:
            seen_ids[condition_id] = index

        kind = str(condition.get("type") or "")
        if kind not in CONDITION_TYPES:
            issues.append(ValidationIssue(
                "error", "condition_type_unknown",
                f"Unknown condition type {kind!r}. Expected one of: "
                f"{', '.join(CONDITION_TYPES)}.", f"{path}.type"))
            continue

        field = str(condition.get("field") or "")
        if field not in known_fields:
            issues.append(ValidationIssue(
                "error", "condition_field_unknown",
                f"Condition references unknown field {field!r}. Defined fields: "
                f"{', '.join(sorted(known_fields)) or 'none'}.", f"{path}.field"))

        boundary = str(condition.get("boundary") or "")
        if boundary not in known_boundaries:
            issues.append(ValidationIssue(
                "error", "condition_boundary_unknown",
                f"Condition references unknown boundary {boundary!r}. Named boundaries: "
                f"{', '.join(sorted(known_boundaries)) or 'none'}.", f"{path}.boundary"))

        # Numeric parameters per type.
        if kind == "dirichlet":
            _require_number(condition, "value", path, issues, "Fixed value")
        elif kind == "neumann":
            _require_number(condition, "flux", path, issues, "Flux", allow_missing=True)
        elif kind == "robin":
            transfer = _require_number(condition, "transfer_coefficient", path, issues,
                                       "Transfer coefficient")
            _require_number(condition, "ambient_value", path, issues, "Ambient value")
            if transfer is not None and transfer < 0:
                issues.append(ValidationIssue(
                    "error", "condition_transfer_negative",
                    "A Robin transfer coefficient must not be negative; a negative value "
                    "would feed energy in through the boundary without limit.",
                    f"{path}.transfer_coefficient"))
        elif kind == "periodic":
            partner = str(condition.get("partner") or "")
            if not partner:
                issues.append(ValidationIssue("error", "condition_partner_missing",
                                              "A periodic condition needs a partner boundary.",
                                              f"{path}.partner"))
            elif partner == boundary:
                issues.append(ValidationIssue(
                    "error", "condition_partner_self",
                    "A boundary cannot be periodic with itself.", f"{path}.partner"))
            elif partner not in known_boundaries:
                issues.append(ValidationIssue(
                    "error", "condition_partner_unknown",
                    f"Periodic partner {partner!r} is not a named boundary.",
                    f"{path}.partner"))
            else:
                periodic_pairs[(field, boundary)] = partner

        if condition.get("enabled", True):
            by_target.setdefault((field, boundary), []).append((index, kind))

    # Contradictions on the same field+boundary.
    for (field, boundary), entries in by_target.items():
        kinds = [kind for _, kind in entries]
        value_count = sum(1 for k in kinds if k in VALUE_FIXING)
        flux_count = sum(1 for k in kinds if k in FLUX_FIXING)
        periodic_count = sum(1 for k in kinds if k == "periodic")

        if value_count > 1:
            issues.append(ValidationIssue(
                "error", "condition_conflict_duplicate_value",
                f"Field {field!r} has {value_count} fixed-value conditions on boundary "
                f"{boundary!r}. Only one can apply.", "conditions"))
        if value_count and flux_count:
            issues.append(ValidationIssue(
                "error", "condition_conflict_value_and_flux",
                f"Field {field!r} on boundary {boundary!r} has both a fixed value and a "
                f"flux condition. That over-determines the boundary; keep one.",
                "conditions"))
        if periodic_count and (value_count or flux_count):
            issues.append(ValidationIssue(
                "error", "condition_conflict_periodic",
                f"Boundary {boundary!r} is periodic for field {field!r} and also carries "
                f"another condition. A periodic boundary is defined by its partner alone.",
                "conditions"))
        if flux_count > 1:
            issues.append(ValidationIssue(
                "warning", "condition_duplicate_flux",
                f"Field {field!r} has {flux_count} flux conditions on boundary "
                f"{boundary!r}; the last one wins.", "conditions"))

    # Periodicity must be reciprocated, or the two sides disagree.
    for (field, boundary), partner in periodic_pairs.items():
        if periodic_pairs.get((field, partner)) != boundary:
            issues.append(ValidationIssue(
                "error", "condition_periodic_not_reciprocated",
                f"Boundary {boundary!r} is periodic with {partner!r} for field {field!r}, "
                f"but {partner!r} does not name {boundary!r} in return. Periodicity must be "
                f"declared on both sides.", "conditions"))
    return issues


def _require_number(condition: Dict[str, Any], key: str, path: str,
                    issues: List[ValidationIssue], label: str,
                    allow_missing: bool = False) -> Optional[float]:
    raw = condition.get(key)
    if raw is None and allow_missing:
        return 0.0
    try:
        value = float(raw)
    except (TypeError, ValueError):
        issues.append(ValidationIssue("error", f"condition_{key}_invalid",
                                      f"{label} must be a number.", f"{path}.{key}"))
        return None
    if not np.isfinite(value):
        issues.append(ValidationIssue("error", f"condition_{key}_invalid",
                                      f"{label} must be finite.", f"{path}.{key}"))
        return None
    return value


def describe_conditions(conditions: Sequence[Dict[str, Any]], domain: Dict[str, Any],
                        field_names: Sequence[str]) -> List[str]:
    """Plain-language lines for the stage summary, including the assumed default."""
    lines: List[str] = []
    boundaries = [str(b.get("id")) for b in (domain.get("boundaries") or [])
                  if isinstance(b, dict) and b.get("id")]
    for field in field_names:
        for boundary in boundaries:
            found = conditions_for(conditions, field, boundary)
            if not found:
                lines.append(f"{field} on {boundary}: no condition set, treated as no-flux.")
                continue
            for condition in found:
                kind = condition["type"]
                if kind == "dirichlet":
                    lines.append(f"{field} on {boundary}: held at {condition['value']:g}.")
                elif kind in ("neumann", "no_flux"):
                    flux = float(condition.get("flux", 0.0))
                    lines.append(f"{field} on {boundary}: "
                                 + ("no flux (sealed)." if flux == 0.0
                                    else f"flux of {flux:g} into the domain."))
                elif kind == "robin":
                    lines.append(
                        f"{field} on {boundary}: exchanges with an ambient value of "
                        f"{condition['ambient_value']:g} at rate "
                        f"{condition['transfer_coefficient']:g}.")
                elif kind == "periodic":
                    lines.append(f"{field} on {boundary}: wraps around to "
                                 f"{condition.get('partner')}.")
    return lines


def resolve_1d_conditions(conditions: Sequence[Dict[str, Any]], field: str,
                          left_id: str = "left", right_id: str = "right") -> Dict[str, Any]:
    """Collapse the condition list into what the 1D solver needs for one field.

    Returns ``{"left": spec, "right": spec, "periodic": bool}`` where a spec is
    ``{"type": ..., ...}``. An unassigned end becomes no-flux.
    """
    def pick(boundary_id: str) -> Dict[str, Any]:
        found = conditions_for(conditions, field, boundary_id)
        # A later condition overrides an earlier one of the same class; value-fixing
        # wins over flux only when validation allowed it through (it does not).
        for condition in reversed(found):
            if condition["type"] in ("dirichlet", "robin", "periodic"):
                return dict(condition)
        for condition in reversed(found):
            if condition["type"] in ("neumann", "no_flux"):
                return dict(condition)
        return {"type": "no_flux", "flux": 0.0}

    left, right = pick(left_id), pick(right_id)
    periodic = left.get("type") == "periodic" and right.get("type") == "periodic"
    return {"left": left, "right": right, "periodic": periodic}
