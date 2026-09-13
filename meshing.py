"""
Mesh generation and discretisation for the geometry preparation workflow.

Stage 3: turn a validated domain into a real discretisation the PDE stage and the
three execution approaches can run on.

  * 1D interval   -> ordered nodes and 2-node line elements
  * 2D rectangle  -> structured quadrilateral grid
  * 2D any shape  -> unstructured triangulation via ``scipy.spatial.Delaunay``

No new dependency: SciPy is already required by the project, so Delaunay
triangulation comes for free. If SciPy's spatial module is unavailable the
unstructured path reports that honestly instead of silently degrading.

Boundary tags
-------------
Named boundaries are re-derived GEOMETRICALLY on every mesh, from the domain's
own boundary selectors, rather than stored as node indices. That is what makes a
boundary condition survive remeshing: indices change when the mesh changes, but
"the left edge of the rectangle" does not.

Numerical assumptions and limitations
-------------------------------------
* Structured grids are uniform; graded spacing is not implemented.
* The unstructured mesher triangulates a point cloud sampled inside the domain
  and then discards triangles whose centroid falls outside it. That handles
  convex and mildly concave shapes well; a strongly re-entrant polygon can lose
  thin features, which ``mesh_stats`` surfaces as a low element count rather than
  hiding it.
* Element quality is reported (min angle, aspect ratio) but not optimised; there
  is no Laplacian smoothing or refinement pass.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from geometry import (
    GEOM_EPS,
    ValidationIssue,
    domain_bounds,
    domain_dimension,
    domain_measure,
    point_in_domain,
)

try:
    from scipy.spatial import Delaunay
    _HAS_DELAUNAY = True
except Exception:                      # pragma: no cover - SciPy without spatial
    _HAS_DELAUNAY = False

#: Refuse to build a mesh larger than this; the editor targets ~1000 nodes.
MAX_MESH_NODES = 200_000


# =============================================================================
# 1D
# =============================================================================
def mesh_1d(domain: Dict[str, Any], element_count: Optional[int] = None,
            target_spacing: Optional[float] = None) -> Dict[str, Any]:
    """Discretise a 1D interval into ordered nodes and line elements.

    Exactly one of ``element_count`` or ``target_spacing`` is used; the count wins
    when both are given, because it is the exact request.
    """
    if domain.get("kind") != "interval":
        raise ValueError("mesh_1d requires an interval domain.")
    spec = domain["interval"]
    x0 = float(spec.get("x_min", 0.0))
    length = float(spec["length"])
    if length <= 0.0:
        raise ValueError("Interval length must be positive.")

    if element_count is not None:
        count = int(element_count)
        if count < 1:
            raise ValueError("Element count must be at least 1.")
    elif target_spacing is not None:
        spacing = float(target_spacing)
        if spacing <= 0.0:
            raise ValueError("Target spacing must be positive.")
        count = max(1, int(round(length / spacing)))
    else:
        raise ValueError("Provide either element_count or target_spacing.")

    if count + 1 > MAX_MESH_NODES:
        raise ValueError(f"That would create {count + 1} nodes, above the {MAX_MESH_NODES} limit.")

    coords = np.linspace(x0, x0 + length, count + 1)
    nodes = [[float(x), 0.0] for x in coords]
    elements = [[i, i + 1] for i in range(count)]
    lengths = np.diff(coords)

    mesh: Dict[str, Any] = {
        "kind": "1d",
        "dimension": 1,
        "units": domain.get("units", ""),
        "nodes": nodes,
        "elements": elements,
        "element_kind": "line2",
        "element_sizes": [float(v) for v in lengths],
        "spacing": float(lengths[0]) if lengths.size else 0.0,
    }
    mesh["boundaries"] = tag_boundaries(mesh, domain)
    mesh["stats"] = mesh_stats(mesh, domain)
    return mesh


# =============================================================================
# 2D structured
# =============================================================================
def mesh_2d_structured(domain: Dict[str, Any], rows: Optional[int] = None,
                       cols: Optional[int] = None,
                       target_spacing: Optional[float] = None) -> Dict[str, Any]:
    """Structured quadrilateral grid over a rectangular domain."""
    if domain.get("kind") != "rectangle":
        raise ValueError("A structured grid requires a rectangular domain. "
                         "Use mesh_2d_unstructured for a circle or polygon.")
    spec = domain["rectangle"]
    x0, y0 = float(spec.get("x_min", 0.0)), float(spec.get("y_min", 0.0))
    width, height = float(spec["width"]), float(spec["height"])
    if width <= 0.0 or height <= 0.0:
        raise ValueError("Rectangle width and height must be positive.")

    if rows is not None and cols is not None:
        n_rows, n_cols = int(rows), int(cols)
    elif target_spacing is not None:
        spacing = float(target_spacing)
        if spacing <= 0.0:
            raise ValueError("Target spacing must be positive.")
        n_cols = max(1, int(round(width / spacing)))
        n_rows = max(1, int(round(height / spacing)))
    else:
        raise ValueError("Provide either rows and cols, or target_spacing.")
    if n_rows < 1 or n_cols < 1:
        raise ValueError("rows and cols must each be at least 1.")

    node_total = (n_rows + 1) * (n_cols + 1)
    if node_total > MAX_MESH_NODES:
        raise ValueError(f"That would create {node_total} nodes, above the "
                         f"{MAX_MESH_NODES} limit. Use a coarser spacing.")

    xs = np.linspace(x0, x0 + width, n_cols + 1)
    ys = np.linspace(y0, y0 + height, n_rows + 1)
    nodes: List[List[float]] = []
    for y in ys:
        for x in xs:
            nodes.append([float(x), float(y)])

    def index(row: int, col: int) -> int:
        return row * (n_cols + 1) + col

    elements: List[List[int]] = []
    for row in range(n_rows):
        for col in range(n_cols):
            # Counter-clockwise ring, matching the topology convention.
            elements.append([index(row, col), index(row, col + 1),
                             index(row + 1, col + 1), index(row + 1, col)])

    dx = width / n_cols
    dy = height / n_rows
    mesh: Dict[str, Any] = {
        "kind": "structured",
        "dimension": 2,
        "units": domain.get("units", ""),
        "nodes": nodes,
        "elements": elements,
        "element_kind": "quad4",
        "rows": n_rows,
        "cols": n_cols,
        "dx": float(dx),
        "dy": float(dy),
        "element_sizes": [float(dx * dy)] * len(elements),
    }
    mesh["boundaries"] = tag_boundaries(mesh, domain)
    mesh["stats"] = mesh_stats(mesh, domain)
    return mesh


# =============================================================================
# 2D unstructured
# =============================================================================
def mesh_2d_unstructured(domain: Dict[str, Any], target_spacing: float,
                         seed: int = 0) -> Dict[str, Any]:
    """Delaunay triangulation of points sampled inside a 2D domain.

    Boundary points are placed first so the mesh follows the outline, then an
    interior lattice fills it. Triangles whose centroid falls outside the domain
    are discarded, which is what keeps a circle circular and a concave polygon
    from being filled across its notch.
    """
    if domain_dimension(domain) != 2:
        raise ValueError("An unstructured mesh requires a 2D domain.")
    if not _HAS_DELAUNAY:
        raise RuntimeError(
            "Unstructured meshing needs scipy.spatial.Delaunay, which is not "
            "available in this environment. Use a structured grid instead."
        )
    spacing = float(target_spacing)
    if spacing <= 0.0:
        raise ValueError("Target spacing must be positive.")

    x0, y0, x1, y1 = domain_bounds(domain)
    if (x1 - x0) <= 0 or (y1 - y0) <= 0:
        raise ValueError("The domain has no extent to mesh.")

    estimate = int(((x1 - x0) / spacing + 2) * ((y1 - y0) / spacing + 2))
    if estimate > MAX_MESH_NODES:
        raise ValueError(f"Spacing {spacing:g} would create roughly {estimate} nodes, "
                         f"above the {MAX_MESH_NODES} limit.")

    points: List[Tuple[float, float]] = list(_boundary_sample(domain, spacing))

    # Interior lattice, offset by half a spacing so interior points do not land
    # exactly on the boundary samples and produce sliver triangles.
    rng = np.random.default_rng(seed)
    nx = max(1, int(np.ceil((x1 - x0) / spacing)))
    ny = max(1, int(np.ceil((y1 - y0) / spacing)))
    for i in range(nx + 1):
        for j in range(ny + 1):
            x = x0 + i * spacing
            y = y0 + j * spacing
            if not point_in_domain((x, y), domain):
                continue
            # Deterministic jitter (seeded) avoids degenerate co-circular sets
            # that make Delaunay output arbitrary diagonals.
            jx = float(rng.uniform(-0.02, 0.02) * spacing)
            jy = float(rng.uniform(-0.02, 0.02) * spacing)
            candidate = (x + jx, y + jy)
            if point_in_domain(candidate, domain):
                points.append(candidate)

    unique = _dedupe(points, spacing * 1e-3)
    if len(unique) < 3:
        raise ValueError("Not enough points to triangulate; use a finer spacing.")

    array = np.asarray(unique, dtype=float)
    triangulation = Delaunay(array)

    elements: List[List[int]] = []
    for simplex in triangulation.simplices:
        tri = array[simplex]
        centroid = tri.mean(axis=0)
        if not point_in_domain((float(centroid[0]), float(centroid[1])), domain):
            continue
        if _triangle_area(tri) <= GEOM_EPS:
            continue
        elements.append([int(i) for i in simplex])

    if not elements:
        raise ValueError("Triangulation produced no elements inside the domain.")

    used = sorted({i for element in elements for i in element})
    remap = {old: new for new, old in enumerate(used)}
    nodes = [[float(array[old][0]), float(array[old][1])] for old in used]
    elements = [[remap[i] for i in element] for element in elements]

    areas = [_triangle_area(np.asarray([nodes[i] for i in element], dtype=float))
             for element in elements]

    mesh: Dict[str, Any] = {
        "kind": "unstructured",
        "dimension": 2,
        "units": domain.get("units", ""),
        "nodes": nodes,
        "elements": elements,
        "element_kind": "tri3",
        "target_spacing": spacing,
        "element_sizes": [float(a) for a in areas],
    }
    mesh["boundaries"] = tag_boundaries(mesh, domain)
    mesh["stats"] = mesh_stats(mesh, domain)
    return mesh


def _triangle_area(tri: np.ndarray) -> float:
    return 0.5 * abs(float(
        (tri[1, 0] - tri[0, 0]) * (tri[2, 1] - tri[0, 1])
        - (tri[2, 0] - tri[0, 0]) * (tri[1, 1] - tri[0, 1])
    ))


def _dedupe(points: Sequence[Tuple[float, float]], tol: float) -> List[Tuple[float, float]]:
    """Drop points closer than ``tol``, keeping the first occurrence."""
    seen: Dict[Tuple[int, int], Tuple[float, float]] = {}
    scale = 1.0 / max(tol, 1e-12)
    for point in points:
        key = (int(round(point[0] * scale)), int(round(point[1] * scale)))
        if key not in seen:
            seen[key] = (float(point[0]), float(point[1]))
    return list(seen.values())


def _boundary_sample(domain: Dict[str, Any], spacing: float) -> List[Tuple[float, float]]:
    """Points along the domain outline, spaced at most ``spacing`` apart."""
    kind = domain.get("kind")
    out: List[Tuple[float, float]] = []
    if kind == "rectangle":
        x0, y0, x1, y1 = domain_bounds(domain)
        corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        for index in range(4):
            out.extend(_sample_segment(corners[index], corners[(index + 1) % 4], spacing))
    elif kind == "circle":
        spec = domain["circle"]
        cx, cy, r = float(spec["cx"]), float(spec["cy"]), float(spec["radius"])
        count = max(8, int(np.ceil(2.0 * np.pi * r / spacing)))
        for i in range(count):
            angle = 2.0 * np.pi * i / count
            out.append((cx + r * float(np.cos(angle)), cy + r * float(np.sin(angle))))
    elif kind == "polygon":
        pts = [(float(p[0]), float(p[1])) for p in domain["polygon"]["points"]]
        for index in range(len(pts)):
            out.extend(_sample_segment(pts[index], pts[(index + 1) % len(pts)], spacing))
    return out


def _sample_segment(a: Tuple[float, float], b: Tuple[float, float],
                    spacing: float) -> List[Tuple[float, float]]:
    length = float(np.hypot(b[0] - a[0], b[1] - a[1]))
    steps = max(1, int(np.ceil(length / spacing)))
    return [(a[0] + (b[0] - a[0]) * i / steps, a[1] + (b[1] - a[1]) * i / steps)
            for i in range(steps)]


# =============================================================================
# Boundary tagging -- geometric, so it survives remeshing
# =============================================================================
def tag_boundaries(mesh: Dict[str, Any], domain: Dict[str, Any],
                   tol: Optional[float] = None) -> Dict[str, Dict[str, List[Any]]]:
    """Map each named domain boundary to the mesh nodes that lie on it.

    Derived from geometry every time, never from stored indices, so a boundary
    condition attached to "left" still means the left edge after the mesh changes.
    """
    nodes = np.asarray(mesh.get("nodes") or [], dtype=float)
    result: Dict[str, Dict[str, List[Any]]] = {}
    if nodes.size == 0:
        return result

    x0, y0, x1, y1 = domain_bounds(domain)
    span = max(x1 - x0, y1 - y0, 1.0)
    eps = float(tol) if tol is not None else span * 1e-6

    for boundary in domain.get("boundaries") or []:
        if not isinstance(boundary, dict):
            continue
        boundary_id = str(boundary.get("id") or "")
        if not boundary_id:
            continue
        selector = boundary.get("selector") or {}
        kind = str(selector.get("type") or "")
        matched: List[int] = []

        if kind == "interval_end":
            target = x0 if str(selector.get("end")) == "min" else x1
            matched = [i for i in range(len(nodes)) if abs(nodes[i][0] - target) <= eps]
        elif kind == "rect_side":
            side = str(selector.get("side"))
            axis, target = {"left": (0, x0), "right": (0, x1),
                            "bottom": (1, y0), "top": (1, y1)}.get(side, (None, None))
            if axis is not None:
                matched = [i for i in range(len(nodes)) if abs(nodes[i][axis] - target) <= eps]
        elif kind == "circle_perimeter":
            spec = domain.get("circle", {})
            cx, cy = float(spec.get("cx", 0.0)), float(spec.get("cy", 0.0))
            r = float(spec.get("radius", 0.0))
            matched = [i for i in range(len(nodes))
                       if abs(float(np.hypot(nodes[i][0] - cx, nodes[i][1] - cy)) - r) <= max(eps, r * 1e-3)]
        elif kind == "polygon_perimeter":
            pts = [(float(p[0]), float(p[1])) for p in domain.get("polygon", {}).get("points", [])]
            matched = [i for i in range(len(nodes))
                       if _distance_to_ring((nodes[i][0], nodes[i][1]), pts) <= max(eps, span * 1e-4)]

        result[boundary_id] = {
            "name": str(boundary.get("name") or boundary_id),
            "nodes": matched,
            "selector": dict(selector),
        }
    return result


def _distance_to_ring(point: Tuple[float, float],
                      ring: Sequence[Tuple[float, float]]) -> float:
    if len(ring) < 2:
        return float("inf")
    best = float("inf")
    px, py = point
    for index in range(len(ring)):
        ax, ay = ring[index]
        bx, by = ring[(index + 1) % len(ring)]
        dx, dy = bx - ax, by - ay
        denom = dx * dx + dy * dy
        if denom <= 0:
            best = min(best, float(np.hypot(px - ax, py - ay)))
            continue
        t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / denom))
        best = min(best, float(np.hypot(px - (ax + t * dx), py - (ay + t * dy))))
    return best


# =============================================================================
# Statistics and validation
# =============================================================================
def mesh_stats(mesh: Dict[str, Any], domain: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Counts, element size statistics, quality, and a cost estimate."""
    nodes = np.asarray(mesh.get("nodes") or [], dtype=float)
    elements = mesh.get("elements") or []
    sizes = np.asarray(mesh.get("element_sizes") or [], dtype=float)

    stats: Dict[str, Any] = {
        "node_count": int(len(nodes)),
        "element_count": int(len(elements)),
        "edge_count": int(len(_unique_edges(elements))),
        "element_kind": mesh.get("element_kind", ""),
        "degenerate_elements": 0,
    }
    if sizes.size:
        stats["element_size"] = {
            "min": float(sizes.min()), "max": float(sizes.max()),
            "mean": float(sizes.mean()), "total": float(sizes.sum()),
        }
        stats["degenerate_elements"] = int(np.count_nonzero(sizes <= GEOM_EPS))

    if mesh.get("element_kind") == "tri3" and len(elements):
        # Only measure elements whose references are in range. This function must
        # survive a malformed mesh, because validate_mesh calls it while deciding
        # what to report about that mesh -- crashing here would turn a reportable
        # defect into a stack trace.
        node_total = len(nodes)
        angles = [
            _min_angle_degrees(np.asarray([nodes[i] for i in element], dtype=float))
            for element in elements
            if len(element) == 3
            and all(isinstance(i, (int, np.integer)) and 0 <= int(i) < node_total for i in element)
        ]
        if angles:
            arr = np.asarray(angles, dtype=float)
            stats["min_angle_deg"] = float(arr.min())
            stats["mean_min_angle_deg"] = float(arr.mean())
            # Below ~15 degrees an explicit scheme's stable step collapses.
            stats["sliver_elements"] = int(np.count_nonzero(arr < 15.0))
        stats["unmeasurable_elements"] = int(len(elements) - len(angles))

    if domain is not None and sizes.size and mesh.get("dimension") == 2:
        measure = domain_measure(domain)
        if measure > 0:
            stats["coverage_fraction"] = float(min(1.0, sizes.sum() / measure))

    # Cost proxy: unknowns x steps is what actually drives runtime, so report the
    # unknown count and let the PDE stage multiply by its own step count.
    stats["unknowns_per_field"] = int(len(nodes))
    stats["estimated_cost_class"] = (
        "small" if len(nodes) <= 2_000 else
        "moderate" if len(nodes) <= 20_000 else
        "large"
    )
    return stats


def _unique_edges(elements: Sequence[Sequence[int]]) -> set:
    edges = set()
    for element in elements:
        count = len(element)
        if count == 2:
            edges.add(tuple(sorted((int(element[0]), int(element[1])))))
            continue
        for index in range(count):
            a, b = int(element[index]), int(element[(index + 1) % count])
            edges.add(tuple(sorted((a, b))))
    return edges


def _min_angle_degrees(tri: np.ndarray) -> float:
    a = float(np.hypot(*(tri[1] - tri[0])))
    b = float(np.hypot(*(tri[2] - tri[1])))
    c = float(np.hypot(*(tri[0] - tri[2])))
    sides = sorted([a, b, c])
    if sides[0] <= GEOM_EPS:
        return 0.0
    # Law of cosines on the shortest side gives the smallest angle.
    cos_value = (sides[1] ** 2 + sides[2] ** 2 - sides[0] ** 2) / (2 * sides[1] * sides[2])
    return float(np.degrees(np.arccos(max(-1.0, min(1.0, cos_value)))))


def validate_mesh(mesh: Any, domain: Optional[Dict[str, Any]] = None) -> List[ValidationIssue]:
    """Report mesh defects: empty mesh, bad references, degenerate elements, slivers."""
    issues: List[ValidationIssue] = []
    if not isinstance(mesh, dict):
        return [ValidationIssue("error", "mesh_not_object", "The mesh must be an object.", "mesh")]

    nodes = mesh.get("nodes")
    elements = mesh.get("elements")
    if not isinstance(nodes, list) or not nodes:
        issues.append(ValidationIssue("error", "mesh_empty",
                                      "The mesh has no nodes. Generate it first.", "mesh.nodes"))
        return issues
    if not isinstance(elements, list) or not elements:
        issues.append(ValidationIssue("error", "mesh_no_elements",
                                      "The mesh has no elements.", "mesh.elements"))
        return issues

    coords = np.asarray(nodes, dtype=float)
    if not np.all(np.isfinite(coords)):
        issues.append(ValidationIssue("error", "mesh_nodes_not_finite",
                                      "Mesh node coordinates must all be finite.", "mesh.nodes"))

    node_total = len(nodes)
    for index, element in enumerate(elements):
        if not isinstance(element, (list, tuple)) or len(element) < 2:
            issues.append(ValidationIssue("error", "mesh_element_invalid",
                                          f"Element {index} is malformed.",
                                          f"mesh.elements[{index}]"))
            continue
        bad = [i for i in element if not isinstance(i, int) or i < 0 or i >= node_total]
        if bad:
            issues.append(ValidationIssue(
                "error", "mesh_reference_invalid",
                f"Element {index} references node index(es) outside the mesh: {bad}.",
                f"mesh.elements[{index}]"))
        elif len(set(element)) != len(element):
            issues.append(ValidationIssue(
                "error", "mesh_element_repeated_node",
                f"Element {index} uses the same node twice.", f"mesh.elements[{index}]"))

    stats = mesh.get("stats") or mesh_stats(mesh, domain)
    if stats.get("degenerate_elements"):
        issues.append(ValidationIssue(
            "error", "mesh_degenerate_elements",
            f"{stats['degenerate_elements']} element(s) have zero size.", "mesh.elements"))
    if stats.get("sliver_elements"):
        issues.append(ValidationIssue(
            "warning", "mesh_sliver_elements",
            f"{stats['sliver_elements']} element(s) have an angle below 15 degrees, which "
            f"forces a very small stable time step.", "mesh.elements"))

    if domain is not None:
        outside = [i for i in range(len(coords))
                   if not point_in_domain((float(coords[i][0]), float(coords[i][1])), domain)]
        if outside:
            issues.append(ValidationIssue(
                "error", "mesh_node_outside_domain",
                f"{len(outside)} mesh node(s) fall outside the domain.", "mesh.nodes"))
    return issues


def generate_mesh(domain: Dict[str, Any], settings: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Dispatch to the right mesher for the domain.

    ``settings`` keys: ``element_count``, ``target_spacing``, ``rows``, ``cols``,
    ``kind`` ('structured' | 'unstructured'), ``seed``.
    """
    settings = dict(settings or {})
    dimension = domain_dimension(domain)
    if dimension == 3:
        # Refuse EXPLICITLY. Previously a 3D domain fell through to the 2D path,
        # domain_bounds returned all zeros for the unrecognised kind, and the user got
        # "The domain has no extent to mesh." -- which sent them to check dimensions
        # that were perfectly valid (a box with width/height/depth of 40) instead of
        # telling them the product has no volumetric mesher.
        raise ValueError(
            f"A {domain.get('kind')!r} domain is three-dimensional, and there is no "
            f"volumetric mesher: this build generates 1D elements, 2D structured grids "
            f"and 2D Delaunay triangulations only (mesh nodes are [x, y] pairs and "
            f"there is no tetrahedral or hexahedral element kind). Use a 2D domain, or "
            f"run CompuCell3D on a 3D lattice, which does not require a mesh."
        )
    if dimension == 1:
        return mesh_1d(domain,
                       element_count=settings.get("element_count"),
                       target_spacing=settings.get("target_spacing"))

    requested = str(settings.get("kind") or "").lower()
    if domain.get("kind") == "rectangle" and requested != "unstructured":
        return mesh_2d_structured(domain,
                                  rows=settings.get("rows"),
                                  cols=settings.get("cols"),
                                  target_spacing=settings.get("target_spacing"))
    spacing = settings.get("target_spacing")
    if spacing is None:
        x0, y0, x1, y1 = domain_bounds(domain)
        spacing = max(x1 - x0, y1 - y0) / 20.0
    return mesh_2d_unstructured(domain, float(spacing), seed=int(settings.get("seed", 0)))
