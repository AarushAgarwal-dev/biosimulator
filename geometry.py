"""
CAD-style domain definition and validation for biological simulation.

Stage 1 of the preparation workflow: the spatial domain every later stage is
expressed against (topology, mesh, PDE fields, boundary conditions, and all
three execution approaches).

Design notes
------------
* Domains are plain dicts, matching the existing blueprint convention used by
  ``simulation_engine`` and ``abm_blueprints`` -- so a domain round-trips through
  JSON with no custom encoder and is embeddable in the project file.
* Validation NEVER raises for malformed user input: it returns a list of issues,
  because the UI has to show a per-stage status rather than a stack trace. Only
  genuine programming errors raise.
* Units are required. A dimensionless length silently reinterpreted between
  stages is a scientific error, not a cosmetic one.

Supported domains: 1D interval, 2D rectangle, 2D circle, 2D polygon.
Geometry is stored in the domain's own units; no implicit conversion happens.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Length units accepted for a domain. Values are metres per unit, so a consumer
# that needs SI can convert explicitly rather than guessing.
LENGTH_UNITS: Dict[str, float] = {
    "m": 1.0,
    "cm": 1e-2,
    "mm": 1e-3,
    "um": 1e-6,
    "µm": 1e-6,
    "nm": 1e-9,
}

DOMAIN_KINDS: Tuple[str, ...] = ("interval", "rectangle", "circle", "polygon")

#: Kinds that are RECOGNISED as three-dimensional but NOT constructible: there is no
#: box/sphere/cylinder builder, no volumetric mesher and no tet/hex element kind, and
#: mesh nodes are [x, y] pairs. They are named here so ``domain_dimension`` can answer
#: 3 truthfully instead of silently reporting 2, which let a 3D request degrade into an
#: empty 2D region rather than an honest refusal. Consumers that cannot handle 3D
#: should refuse with a message naming the missing capability, not a generic error.
VOLUMETRIC_DOMAIN_KINDS: Tuple[str, ...] = ("box", "sphere", "cylinder")

#: Below this, two points are treated as coincident and a segment as zero-length.
GEOM_EPS = 1e-9


# =============================================================================
# Validation reporting
# =============================================================================
@dataclass
class ValidationIssue:
    """One problem found in user-supplied configuration.

    ``severity`` is 'error' (blocks compilation) or 'warning' (allowed to run).
    ``code`` is stable and machine-readable so the UI can map it to a control;
    ``message`` is the human sentence; ``path`` locates the offending field.
    """

    severity: str
    code: str
    message: str
    path: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "path": self.path,
        }


def errors_only(issues: Sequence[ValidationIssue]) -> List[ValidationIssue]:
    return [i for i in issues if i.severity == "error"]


def is_valid(issues: Sequence[ValidationIssue]) -> bool:
    return not errors_only(issues)


def issues_to_dicts(issues: Sequence[ValidationIssue]) -> List[Dict[str, Any]]:
    return [i.to_dict() for i in issues]


# =============================================================================
# Constructors -- convenience builders that produce a valid domain dict
# =============================================================================
def make_interval(length: float, units: str = "um", x_min: float = 0.0,
                  name: str = "Domain") -> Dict[str, Any]:
    return {
        "kind": "interval",
        "name": name,
        "units": units,
        "interval": {"x_min": float(x_min), "length": float(length)},
        "regions": [],
        "boundaries": [
            {"id": "left", "name": "Left end", "selector": {"type": "interval_end", "end": "min"}},
            {"id": "right", "name": "Right end", "selector": {"type": "interval_end", "end": "max"}},
        ],
    }


def make_rectangle(width: float, height: float, units: str = "um",
                   x_min: float = 0.0, y_min: float = 0.0,
                   name: str = "Domain") -> Dict[str, Any]:
    return {
        "kind": "rectangle",
        "name": name,
        "units": units,
        "rectangle": {
            "x_min": float(x_min), "y_min": float(y_min),
            "width": float(width), "height": float(height),
        },
        "regions": [],
        "boundaries": [
            {"id": side, "name": f"{side.capitalize()} edge",
             "selector": {"type": "rect_side", "side": side}}
            for side in ("left", "right", "bottom", "top")
        ],
    }


def make_circle(radius: float, units: str = "um", cx: float = 0.0, cy: float = 0.0,
                name: str = "Domain") -> Dict[str, Any]:
    return {
        "kind": "circle",
        "name": name,
        "units": units,
        "circle": {"cx": float(cx), "cy": float(cy), "radius": float(radius)},
        "regions": [],
        "boundaries": [
            {"id": "perimeter", "name": "Perimeter", "selector": {"type": "circle_perimeter"}},
        ],
    }


def make_polygon(points: Sequence[Sequence[float]], units: str = "um",
                 name: str = "Domain") -> Dict[str, Any]:
    return {
        "kind": "polygon",
        "name": name,
        "units": units,
        "polygon": {"points": [[float(p[0]), float(p[1])] for p in points]},
        "regions": [],
        "boundaries": [
            {"id": "perimeter", "name": "Perimeter", "selector": {"type": "polygon_perimeter"}},
        ],
    }


# =============================================================================
# Geometric predicates
# =============================================================================
def _finite(*values: Any) -> bool:
    try:
        return all(np.isfinite(float(v)) for v in values)
    except (TypeError, ValueError):
        return False


def _segments_properly_intersect(p1, p2, p3, p4) -> bool:
    """True when segment p1p2 crosses p3p4 at an interior point.

    Used for polygon self-intersection. Shared endpoints between ADJACENT edges
    are legitimate, so the caller skips those pairs; this routine reports a
    genuine crossing, including the collinear-overlap case.
    """
    def orient(a, b, c) -> float:
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    def on_segment(a, b, c) -> bool:
        return (
            min(a[0], b[0]) - GEOM_EPS <= c[0] <= max(a[0], b[0]) + GEOM_EPS
            and min(a[1], b[1]) - GEOM_EPS <= c[1] <= max(a[1], b[1]) + GEOM_EPS
        )

    d1, d2 = orient(p3, p4, p1), orient(p3, p4, p2)
    d3, d4 = orient(p1, p2, p3), orient(p1, p2, p4)

    if ((d1 > GEOM_EPS and d2 < -GEOM_EPS) or (d1 < -GEOM_EPS and d2 > GEOM_EPS)) and \
       ((d3 > GEOM_EPS and d4 < -GEOM_EPS) or (d3 < -GEOM_EPS and d4 > GEOM_EPS)):
        return True
    # Collinear overlap.
    for da, (a, b, c) in ((d1, (p3, p4, p1)), (d2, (p3, p4, p2)),
                          (d3, (p1, p2, p3)), (d4, (p1, p2, p4))):
        if abs(da) <= GEOM_EPS and on_segment(a, b, c):
            return True
    return False


def polygon_is_simple(points: Sequence[Sequence[float]]) -> bool:
    """True when no two non-adjacent edges of the closed polygon cross."""
    n = len(points)
    if n < 3:
        return False
    for i in range(n):
        a1, a2 = points[i], points[(i + 1) % n]
        for j in range(i + 1, n):
            # Skip adjacent edges (they legitimately share an endpoint).
            if j == i or (j + 1) % n == i or (i + 1) % n == j:
                continue
            b1, b2 = points[j], points[(j + 1) % n]
            if _segments_properly_intersect(a1, a2, b1, b2):
                return False
    return True


def polygon_signed_area(points: Sequence[Sequence[float]]) -> float:
    """Shoelace signed area; positive when the ring is counter-clockwise."""
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] < 3:
        return 0.0
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def polygon_area(points: Sequence[Sequence[float]]) -> float:
    return abs(polygon_signed_area(points))


def point_in_polygon(point: Sequence[float], points: Sequence[Sequence[float]]) -> bool:
    """Ray-casting test; boundary points count as inside."""
    x, y = float(point[0]), float(point[1])
    n = len(points)
    inside = False
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        # On-edge counts as inside.
        if abs((x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)) <= 1e-9 and \
           min(x1, x2) - GEOM_EPS <= x <= max(x1, x2) + GEOM_EPS and \
           min(y1, y2) - GEOM_EPS <= y <= max(y1, y2) + GEOM_EPS:
            return True
        if (y1 > y) != (y2 > y):
            x_cross = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < x_cross:
                inside = not inside
    return inside


def domain_dimension(domain: Dict[str, Any]) -> int:
    # Do NOT classify everything that is not an interval as two-dimensional. That made
    # a 3D domain report 2, and meshing, pde_model and the CompuCell3D adapter all
    # branch on this value -- so a 3D request degraded into an empty 2D region
    # (domain_measure returned 0.0, point_in_domain returned False) instead of being
    # refused. Naming the volumetric kinds here keeps the answer truthful even though
    # no constructor for them exists yet; callers refuse them explicitly.
    kind = domain.get("kind")
    if kind == "interval":
        return 1
    if kind in VOLUMETRIC_DOMAIN_KINDS:
        return 3
    return 2


def domain_bounds(domain: Dict[str, Any]) -> Tuple[float, float, float, float]:
    """Axis-aligned bounds ``(x_min, y_min, x_max, y_max)``.

    A 1D interval reports zero height so callers can share one bounds contract.
    """
    kind = domain.get("kind")
    if kind == "interval":
        spec = domain.get("interval", {})
        x0 = float(spec.get("x_min", 0.0))
        return x0, 0.0, x0 + float(spec.get("length", 0.0)), 0.0
    if kind == "rectangle":
        spec = domain.get("rectangle", {})
        x0, y0 = float(spec.get("x_min", 0.0)), float(spec.get("y_min", 0.0))
        return x0, y0, x0 + float(spec.get("width", 0.0)), y0 + float(spec.get("height", 0.0))
    if kind == "circle":
        spec = domain.get("circle", {})
        cx, cy, r = float(spec.get("cx", 0.0)), float(spec.get("cy", 0.0)), float(spec.get("radius", 0.0))
        return cx - r, cy - r, cx + r, cy + r
    if kind == "polygon":
        pts = np.asarray(domain.get("polygon", {}).get("points", []), dtype=float)
        if pts.size == 0:
            return 0.0, 0.0, 0.0, 0.0
        return (float(pts[:, 0].min()), float(pts[:, 1].min()),
                float(pts[:, 0].max()), float(pts[:, 1].max()))
    return 0.0, 0.0, 0.0, 0.0


def domain_measure(domain: Dict[str, Any]) -> float:
    """Length for a 1D interval, area for a 2D domain, in the domain's units."""
    kind = domain.get("kind")
    if kind == "interval":
        return abs(float(domain.get("interval", {}).get("length", 0.0)))
    if kind == "rectangle":
        spec = domain.get("rectangle", {})
        return abs(float(spec.get("width", 0.0)) * float(spec.get("height", 0.0)))
    if kind == "circle":
        return float(np.pi) * float(domain.get("circle", {}).get("radius", 0.0)) ** 2
    if kind == "polygon":
        return polygon_area(domain.get("polygon", {}).get("points", []))
    return 0.0


def point_in_domain(point: Sequence[float], domain: Dict[str, Any],
                    tol: float = 1e-9) -> bool:
    """True when a point lies inside the domain (boundary inclusive)."""
    kind = domain.get("kind")
    x = float(point[0])
    y = float(point[1]) if len(point) > 1 else 0.0
    if kind == "interval":
        spec = domain.get("interval", {})
        x0 = float(spec.get("x_min", 0.0))
        x1 = x0 + float(spec.get("length", 0.0))
        return (min(x0, x1) - tol) <= x <= (max(x0, x1) + tol) and abs(y) <= tol + GEOM_EPS
    if kind == "rectangle":
        x0, y0, x1, y1 = domain_bounds(domain)
        return (x0 - tol) <= x <= (x1 + tol) and (y0 - tol) <= y <= (y1 + tol)
    if kind == "circle":
        spec = domain.get("circle", {})
        cx, cy = float(spec.get("cx", 0.0)), float(spec.get("cy", 0.0))
        r = float(spec.get("radius", 0.0))
        return (x - cx) ** 2 + (y - cy) ** 2 <= (r + tol) ** 2
    if kind == "polygon":
        return point_in_polygon((x, y), domain.get("polygon", {}).get("points", []))
    return False


# =============================================================================
# Validation
# =============================================================================
def validate_domain(domain: Any) -> List[ValidationIssue]:
    """Check a domain for every failure mode the UI must surface.

    Covers: unknown kind, missing/unknown units, missing required dimensions,
    zero or negative dimensions, non-finite numbers, too few polygon points,
    duplicate consecutive points, self-intersection, and zero-area polygons.
    """
    issues: List[ValidationIssue] = []
    if not isinstance(domain, dict):
        return [ValidationIssue("error", "domain_not_object",
                                "The domain must be an object.", "domain")]

    kind = domain.get("kind")
    if kind not in DOMAIN_KINDS:
        return [ValidationIssue(
            "error", "domain_kind_unknown",
            f"Unknown domain kind {kind!r}. Expected one of: {', '.join(DOMAIN_KINDS)}.",
            "domain.kind")]

    units = domain.get("units")
    if not units:
        issues.append(ValidationIssue(
            "error", "units_missing",
            "The domain needs length units so every stage interprets distances the same way.",
            "domain.units"))
    elif units not in LENGTH_UNITS:
        issues.append(ValidationIssue(
            "error", "units_unknown",
            f"Unknown length unit {units!r}. Supported: {', '.join(LENGTH_UNITS)}.",
            "domain.units"))

    if kind == "interval":
        spec = domain.get("interval")
        if not isinstance(spec, dict):
            issues.append(ValidationIssue("error", "interval_missing",
                                          "A 1D domain needs an 'interval' definition.",
                                          "domain.interval"))
        else:
            length = spec.get("length")
            if length is None:
                issues.append(ValidationIssue("error", "length_missing",
                                              "Interval length is required.",
                                              "domain.interval.length"))
            elif not _finite(length, spec.get("x_min", 0.0)):
                issues.append(ValidationIssue("error", "geometry_not_finite",
                                              "Interval values must be finite numbers.",
                                              "domain.interval"))
            elif float(length) <= 0.0:
                issues.append(ValidationIssue(
                    "error", "length_not_positive",
                    f"Interval length must be greater than zero (got {float(length):g}).",
                    "domain.interval.length"))

    elif kind == "rectangle":
        spec = domain.get("rectangle")
        if not isinstance(spec, dict):
            issues.append(ValidationIssue("error", "rectangle_missing",
                                          "A rectangular domain needs a 'rectangle' definition.",
                                          "domain.rectangle"))
        else:
            for dim in ("width", "height"):
                value = spec.get(dim)
                if value is None:
                    issues.append(ValidationIssue("error", f"{dim}_missing",
                                                  f"Rectangle {dim} is required.",
                                                  f"domain.rectangle.{dim}"))
                elif not _finite(value):
                    issues.append(ValidationIssue("error", "geometry_not_finite",
                                                  f"Rectangle {dim} must be a finite number.",
                                                  f"domain.rectangle.{dim}"))
                elif float(value) <= 0.0:
                    issues.append(ValidationIssue(
                        "error", f"{dim}_not_positive",
                        f"Rectangle {dim} must be greater than zero (got {float(value):g}).",
                        f"domain.rectangle.{dim}"))

    elif kind == "circle":
        spec = domain.get("circle")
        if not isinstance(spec, dict):
            issues.append(ValidationIssue("error", "circle_missing",
                                          "A circular domain needs a 'circle' definition.",
                                          "domain.circle"))
        else:
            radius = spec.get("radius")
            if radius is None:
                issues.append(ValidationIssue("error", "radius_missing",
                                              "Circle radius is required.",
                                              "domain.circle.radius"))
            elif not _finite(radius, spec.get("cx", 0.0), spec.get("cy", 0.0)):
                issues.append(ValidationIssue("error", "geometry_not_finite",
                                              "Circle values must be finite numbers.",
                                              "domain.circle"))
            elif float(radius) <= 0.0:
                issues.append(ValidationIssue(
                    "error", "radius_not_positive",
                    f"Circle radius must be greater than zero (got {float(radius):g}).",
                    "domain.circle.radius"))

    elif kind == "polygon":
        spec = domain.get("polygon")
        points = (spec or {}).get("points") if isinstance(spec, dict) else None
        if not isinstance(points, list):
            issues.append(ValidationIssue("error", "polygon_missing",
                                          "A polygonal domain needs a list of points.",
                                          "domain.polygon.points"))
        else:
            clean: List[Tuple[float, float]] = []
            malformed = False
            for index, point in enumerate(points):
                if not (isinstance(point, (list, tuple)) and len(point) >= 2) or \
                        not _finite(point[0], point[1]):
                    malformed = True
                    issues.append(ValidationIssue(
                        "error", "point_malformed",
                        f"Point {index + 1} must be a pair of finite numbers [x, y].",
                        f"domain.polygon.points[{index}]"))
                    continue
                clean.append((float(point[0]), float(point[1])))

            if len(clean) < 3:
                if not malformed:
                    issues.append(ValidationIssue(
                        "error", "polygon_too_few_points",
                        f"A polygon needs at least 3 points (got {len(clean)}).",
                        "domain.polygon.points"))
            else:
                # Duplicate points: consecutive duplicates create zero-length
                # edges; repeats elsewhere create a pinched (non-simple) ring.
                for index in range(len(clean)):
                    a, b = clean[index], clean[(index + 1) % len(clean)]
                    if abs(a[0] - b[0]) <= GEOM_EPS and abs(a[1] - b[1]) <= GEOM_EPS:
                        issues.append(ValidationIssue(
                            "error", "duplicate_point",
                            f"Points {index + 1} and {(index + 1) % len(clean) + 1} are identical, "
                            f"which makes a zero-length edge.",
                            f"domain.polygon.points[{index}]"))
                seen: Dict[Tuple[float, float], int] = {}
                for index, point in enumerate(clean):
                    key = (round(point[0], 9), round(point[1], 9))
                    if key in seen:
                        issues.append(ValidationIssue(
                            "error", "duplicate_point",
                            f"Point {index + 1} repeats point {seen[key] + 1}.",
                            f"domain.polygon.points[{index}]"))
                    else:
                        seen[key] = index

                if not polygon_is_simple(clean):
                    issues.append(ValidationIssue(
                        "error", "polygon_self_intersecting",
                        "The polygon edges cross each other. A simulation domain must be a "
                        "simple (non-self-intersecting) ring.",
                        "domain.polygon.points"))
                elif polygon_area(clean) <= GEOM_EPS:
                    issues.append(ValidationIssue(
                        "error", "polygon_zero_area",
                        "The polygon encloses no area - its points are collinear.",
                        "domain.polygon.points"))

    issues.extend(_validate_named_collection(domain.get("regions"), "regions"))
    issues.extend(_validate_named_collection(domain.get("boundaries"), "boundaries"))
    return issues


def _validate_named_collection(items: Any, label: str) -> List[ValidationIssue]:
    """Named regions and boundaries must have unique, non-empty ids."""
    issues: List[ValidationIssue] = []
    if items is None:
        return issues
    if not isinstance(items, list):
        return [ValidationIssue("error", f"{label}_not_list",
                                f"'{label}' must be a list.", f"domain.{label}")]
    seen: Dict[str, int] = {}
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            issues.append(ValidationIssue("error", f"{label}_item_invalid",
                                          f"Each entry in '{label}' must be an object.",
                                          f"domain.{label}[{index}]"))
            continue
        item_id = str(item.get("id") or "").strip()
        if not item_id:
            issues.append(ValidationIssue(
                "error", f"{label}_id_missing",
                f"Every {label[:-1]} needs an id so later stages can refer to it by name.",
                f"domain.{label}[{index}].id"))
            continue
        if item_id in seen:
            issues.append(ValidationIssue(
                "error", f"{label}_id_duplicate",
                f"Duplicate {label[:-1]} id {item_id!r} (already used at position {seen[item_id] + 1}).",
                f"domain.{label}[{index}].id"))
        else:
            seen[item_id] = index
    return issues


def validate_points_inside_domain(points: Sequence[Sequence[float]],
                                  domain: Dict[str, Any],
                                  path: str = "topology.nodes") -> List[ValidationIssue]:
    """Report any point that falls outside the domain.

    Kept separate from :func:`validate_domain` because it is a cross-stage check:
    the topology stage calls it once the domain is already known good.
    """
    issues: List[ValidationIssue] = []
    for index, point in enumerate(points):
        try:
            inside = point_in_domain(point, domain)
        except (TypeError, ValueError, IndexError):
            inside = False
        if not inside:
            coords = ", ".join(f"{float(c):g}" for c in point[:2])
            issues.append(ValidationIssue(
                "error", "entity_outside_domain",
                f"Point {index + 1} ({coords}) lies outside the domain.",
                f"{path}[{index}]"))
    return issues


def describe_domain(domain: Dict[str, Any]) -> str:
    """One-line plain-language summary for the UI stage header."""
    kind = domain.get("kind")
    units = domain.get("units", "")
    if kind == "interval":
        return f"1D interval of length {float(domain['interval']['length']):g} {units}"
    if kind == "rectangle":
        spec = domain["rectangle"]
        return (f"2D rectangle {float(spec['width']):g} x {float(spec['height']):g} {units}"
                f" (area {domain_measure(domain):g} {units}\u00b2)")
    if kind == "circle":
        return (f"2D circle of radius {float(domain['circle']['radius']):g} {units}"
                f" (area {domain_measure(domain):g} {units}\u00b2)")
    if kind == "polygon":
        count = len(domain.get("polygon", {}).get("points", []))
        return (f"2D polygon with {count} vertices"
                f" (area {domain_measure(domain):g} {units}\u00b2)")
    return "Unknown domain"
