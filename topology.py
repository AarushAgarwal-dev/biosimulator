"""
Node / edge / cell topology for the geometry preparation workflow.

Stage 2 of the workflow: the discrete entities every later stage refers to.
Nodes carry coordinates, edges connect two nodes, and a cell is an ordered ring
of nodes and the edges between them -- the vertex representation a Cellular Potts
or vertex model needs, and the table view the UI shows alongside the canvas.

Design notes
------------
* Plain dicts, like ``geometry`` and the existing blueprints, so a topology is
  JSON without a custom encoder and embeds directly in the project file.
* Ids are stable strings. They are never re-indexed by an edit, because boundary
  conditions and results reference entities by id across remeshing.
* Every mutation keeps the derived indices consistent, so validation and lookup
  stay O(1) per entity at the ~1000 node / 800 edge target.
* Validation returns issues; it does not raise. Only programming errors raise.
"""

import csv
import io
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from geometry import (
    GEOM_EPS,
    ValidationIssue,
    point_in_domain,
    polygon_area,
    polygon_is_simple,
    polygon_signed_area,
)

#: Caps that keep automatic loop discovery predictable on a large topology.
MAX_CYCLE_LENGTH = 12
MAX_CYCLES = 500


# =============================================================================
# Construction
# =============================================================================
def new_topology() -> Dict[str, Any]:
    return {"nodes": [], "edges": [], "cells": [], "next_ids": {"node": 1, "edge": 1, "cell": 1}}


def _next_id(topology: Dict[str, Any], kind: str, prefix: str) -> str:
    counters = topology.setdefault("next_ids", {"node": 1, "edge": 1, "cell": 1})
    used = {str(item.get("id")) for item in topology.get(f"{kind}s", [])}
    value = int(counters.get(kind, 1))
    while f"{prefix}{value}" in used:
        value += 1
    counters[kind] = value + 1
    return f"{prefix}{value}"


def node_index(topology: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {str(n.get("id")): n for n in topology.get("nodes", []) if isinstance(n, dict)}


def edge_index(topology: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {str(e.get("id")): e for e in topology.get("edges", []) if isinstance(e, dict)}


def cell_index(topology: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {str(c.get("id")): c for c in topology.get("cells", []) if isinstance(c, dict)}


def add_node(topology: Dict[str, Any], x: float, y: float = 0.0,
             z: Optional[float] = None, node_id: Optional[str] = None,
             region: str = "", metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Append a node and return it. ``node_id`` must be unique when supplied."""
    if node_id is not None:
        if str(node_id) in node_index(topology):
            raise ValueError(f"Node id {node_id!r} already exists.")
        new_id = str(node_id)
    else:
        new_id = _next_id(topology, "node", "n")
    node: Dict[str, Any] = {
        "id": new_id, "x": float(x), "y": float(y),
        "region": str(region or ""), "metadata": dict(metadata or {}),
    }
    if z is not None:
        node["z"] = float(z)
    topology.setdefault("nodes", []).append(node)
    return node


def move_node(topology: Dict[str, Any], node_id: str, x: float, y: float,
              z: Optional[float] = None) -> Dict[str, Any]:
    """Reposition a node. Cell geometry is recomputed, since area/centroid move with it."""
    node = node_index(topology).get(str(node_id))
    if node is None:
        raise KeyError(f"No node {node_id!r}.")
    node["x"], node["y"] = float(x), float(y)
    if z is not None:
        node["z"] = float(z)
    recompute_cell_geometry(topology)
    return node


def delete_node(topology: Dict[str, Any], node_id: str) -> Dict[str, int]:
    """Delete a node and everything that referenced it.

    Cascading is required rather than optional: an edge pointing at a removed node
    is an invalid reference, and a cell missing a vertex is degenerate. Returns the
    counts removed so the UI can tell the user what the edit actually did.
    """
    node_id = str(node_id)
    if node_id not in node_index(topology):
        raise KeyError(f"No node {node_id!r}.")

    topology["nodes"] = [n for n in topology.get("nodes", []) if str(n.get("id")) != node_id]

    dead_edges = {str(e.get("id")) for e in topology.get("edges", [])
                  if str(e.get("source")) == node_id or str(e.get("target")) == node_id}
    topology["edges"] = [e for e in topology.get("edges", [])
                         if str(e.get("id")) not in dead_edges]

    kept_cells = []
    removed_cells = 0
    for cell in topology.get("cells", []):
        node_ids = [str(v) for v in cell.get("nodes", [])]
        edge_ids = {str(v) for v in cell.get("edges", [])}
        if node_id in node_ids or (edge_ids & dead_edges):
            removed_cells += 1
            continue
        kept_cells.append(cell)
    topology["cells"] = kept_cells
    recompute_cell_geometry(topology)
    return {"nodes": 1, "edges": len(dead_edges), "cells": removed_cells}


def add_edge(topology: Dict[str, Any], source: str, target: str,
             edge_id: Optional[str] = None, directed: bool = False,
             boundary: str = "") -> Dict[str, Any]:
    """Connect two existing nodes.

    Refuses a self-loop and a duplicate connection: both are topology errors that
    would otherwise surface much later as a degenerate cell or a double-counted
    boundary.
    """
    nodes = node_index(topology)
    source, target = str(source), str(target)
    if source not in nodes:
        raise KeyError(f"No node {source!r}.")
    if target not in nodes:
        raise KeyError(f"No node {target!r}.")
    if source == target:
        raise ValueError("An edge cannot start and end at the same node.")
    for existing in topology.get("edges", []):
        pair = {str(existing.get("source")), str(existing.get("target"))}
        if pair == {source, target}:
            raise ValueError(f"Nodes {source} and {target} are already connected "
                             f"by edge {existing.get('id')}.")
    if edge_id is not None:
        if str(edge_id) in edge_index(topology):
            raise ValueError(f"Edge id {edge_id!r} already exists.")
        new_id = str(edge_id)
    else:
        new_id = _next_id(topology, "edge", "e")
    edge = {"id": new_id, "source": source, "target": target,
            "directed": bool(directed), "boundary": str(boundary or "")}
    topology.setdefault("edges", []).append(edge)
    return edge


def delete_edge(topology: Dict[str, Any], edge_id: str) -> Dict[str, int]:
    edge_id = str(edge_id)
    if edge_id not in edge_index(topology):
        raise KeyError(f"No edge {edge_id!r}.")
    topology["edges"] = [e for e in topology.get("edges", []) if str(e.get("id")) != edge_id]
    before = len(topology.get("cells", []))
    topology["cells"] = [c for c in topology.get("cells", [])
                         if edge_id not in {str(v) for v in c.get("edges", [])}]
    recompute_cell_geometry(topology)
    return {"edges": 1, "cells": before - len(topology.get("cells", []))}


def adjacency(topology: Dict[str, Any]) -> Dict[str, Set[str]]:
    """Undirected adjacency map. Built once per call, O(E)."""
    adj: Dict[str, Set[str]] = {str(n.get("id")): set() for n in topology.get("nodes", [])}
    for edge in topology.get("edges", []):
        s, t = str(edge.get("source")), str(edge.get("target"))
        if s in adj and t in adj:
            adj[s].add(t)
            adj[t].add(s)
    return adj


def edge_between(topology: Dict[str, Any], a: str, b: str) -> Optional[Dict[str, Any]]:
    pair = {str(a), str(b)}
    for edge in topology.get("edges", []):
        if {str(edge.get("source")), str(edge.get("target"))} == pair:
            return edge
    return None


# =============================================================================
# Loops and cells
# =============================================================================
def loop_edges(topology: Dict[str, Any], node_ids: Sequence[str]) -> List[str]:
    """Edge ids closing the ring ``node_ids``, or raise if the ring is broken.

    A loop is only convertible into a cell when every consecutive pair -- including
    last-to-first -- is an existing edge. Reporting exactly which link is missing
    is what lets the UI say "the loop is open between n3 and n4".
    """
    ids = [str(v) for v in node_ids]
    if len(ids) < 3:
        raise ValueError("A cell needs at least 3 nodes.")
    if len(set(ids)) != len(ids):
        raise ValueError("A cell loop cannot visit the same node twice.")
    found: List[str] = []
    for index in range(len(ids)):
        a, b = ids[index], ids[(index + 1) % len(ids)]
        edge = edge_between(topology, a, b)
        if edge is None:
            raise ValueError(f"The loop is open: no edge connects {a} and {b}.")
        found.append(str(edge.get("id")))
    return found


def make_cell(topology: Dict[str, Any], node_ids: Sequence[str], cell_type: str = "generic",
              region: str = "", cell_id: Optional[str] = None) -> Dict[str, Any]:
    """Convert a closed, non-self-intersecting loop into a cell."""
    ids = [str(v) for v in node_ids]
    edges = loop_edges(topology, ids)
    nodes = node_index(topology)
    points = [(float(nodes[i]["x"]), float(nodes[i].get("y", 0.0))) for i in ids]
    if not polygon_is_simple(points):
        raise ValueError("The loop crosses itself, so it does not enclose a single cell.")
    if polygon_area(points) <= GEOM_EPS:
        raise ValueError("The loop encloses no area - its nodes are collinear.")

    # Store the ring counter-clockwise so signed area and neighbour orientation
    # are comparable between cells.
    if polygon_signed_area(points) < 0:
        ids = [ids[0]] + list(reversed(ids[1:]))
        edges = loop_edges(topology, ids)
        points = [(float(nodes[i]["x"]), float(nodes[i].get("y", 0.0))) for i in ids]

    new_id = str(cell_id) if cell_id is not None else _next_id(topology, "cell", "c")
    if cell_id is not None and new_id in cell_index(topology):
        raise ValueError(f"Cell id {new_id!r} already exists.")
    cell = {
        "id": new_id, "nodes": ids, "edges": edges,
        "type": str(cell_type or "generic"), "region": str(region or ""),
        "centroid": _centroid(points), "area": float(polygon_area(points)),
        "neighbors": [],
    }
    topology.setdefault("cells", []).append(cell)
    compute_neighbors(topology)
    return cell


def _centroid(points: Sequence[Sequence[float]]) -> List[float]:
    """Area-weighted polygon centroid, falling back to the vertex mean.

    The vertex mean is only correct for a regular ring; the area-weighted formula
    is the actual centre of mass a cell measurement should report.
    """
    pts = np.asarray(points, dtype=float)
    signed = polygon_signed_area(pts)
    if abs(signed) <= GEOM_EPS:
        return [float(pts[:, 0].mean()), float(pts[:, 1].mean())]
    x, y = pts[:, 0], pts[:, 1]
    x_next, y_next = np.roll(x, -1), np.roll(y, -1)
    cross = x * y_next - x_next * y
    cx = float(np.sum((x + x_next) * cross) / (6.0 * signed))
    cy = float(np.sum((y + y_next) * cross) / (6.0 * signed))
    return [cx, cy]


def recompute_cell_geometry(topology: Dict[str, Any]) -> None:
    """Refresh centroid and area after nodes move or entities are removed."""
    nodes = node_index(topology)
    for cell in topology.get("cells", []):
        ids = [str(v) for v in cell.get("nodes", [])]
        if not all(i in nodes for i in ids) or len(ids) < 3:
            cell["area"] = 0.0
            cell["centroid"] = [0.0, 0.0]
            continue
        points = [(float(nodes[i]["x"]), float(nodes[i].get("y", 0.0))) for i in ids]
        cell["area"] = float(polygon_area(points))
        cell["centroid"] = _centroid(points)


def compute_neighbors(topology: Dict[str, Any]) -> None:
    """Two cells neighbour each other when they share at least one edge."""
    by_edge: Dict[str, List[str]] = {}
    for cell in topology.get("cells", []):
        for edge_id in {str(v) for v in cell.get("edges", [])}:
            by_edge.setdefault(edge_id, []).append(str(cell.get("id")))
    found: Dict[str, Set[str]] = {str(c.get("id")): set() for c in topology.get("cells", [])}
    for owners in by_edge.values():
        for a in owners:
            for b in owners:
                if a != b:
                    found[a].add(b)
    for cell in topology.get("cells", []):
        cell["neighbors"] = sorted(found.get(str(cell.get("id")), ()))


def find_cycles(topology: Dict[str, Any], max_length: int = MAX_CYCLE_LENGTH,
                max_cycles: int = MAX_CYCLES) -> List[List[str]]:
    """Find chordless cycles up to ``max_length``, deterministically and bounded.

    Enumerating every cycle in a general graph is exponential, so this is
    deliberately capped: it exists to offer the user candidate loops to turn into
    cells, not to be a complete cycle basis. Cycles are returned in a canonical
    rotation so the same ring is never offered twice.
    """
    adj = adjacency(topology)
    order = sorted(adj)
    rank = {node: index for index, node in enumerate(order)}
    seen: Set[Tuple[str, ...]] = set()
    cycles: List[List[str]] = []

    def canonical(path: List[str]) -> Tuple[str, ...]:
        best = min(range(len(path)), key=lambda i: rank[path[i]])
        rotated = path[best:] + path[:best]
        if len(rotated) > 2 and rank[rotated[1]] > rank[rotated[-1]]:
            rotated = [rotated[0]] + list(reversed(rotated[1:]))
        return tuple(rotated)

    def is_chordless(path: List[str]) -> bool:
        ring = set(path)
        for index, node in enumerate(path):
            allowed = {path[index - 1], path[(index + 1) % len(path)]}
            if (adj[node] & ring) - allowed:
                return False
        return True

    for start in order:
        if len(cycles) >= max_cycles:
            break
        stack: List[Tuple[str, List[str]]] = [(start, [start])]
        while stack:
            if len(cycles) >= max_cycles:
                break
            node, path = stack.pop()
            for neighbor in sorted(adj[node]):
                if neighbor == start and len(path) >= 3:
                    key = canonical(path)
                    if key not in seen and is_chordless(list(key)):
                        seen.add(key)
                        cycles.append(list(key))
                    continue
                # Only extend through higher-ranked nodes, so each ring is walked
                # from its lowest-ranked member exactly once.
                if rank[neighbor] > rank[start] and neighbor not in path and len(path) < max_length:
                    stack.append((neighbor, path + [neighbor]))
    return cycles


# =============================================================================
# Validation
# =============================================================================
def validate_topology(topology: Any, domain: Optional[Dict[str, Any]] = None) -> List[ValidationIssue]:
    """Report every topology defect the UI must surface before meshing.

    Detects: malformed containers, missing/duplicate ids, non-finite coordinates,
    self-referencing edges, duplicate edges, dangling references, zero-length
    edges, open cell loops, self-intersecting cells, degenerate (zero-area) cells,
    and nodes outside the domain when one is supplied.
    """
    issues: List[ValidationIssue] = []
    if not isinstance(topology, dict):
        return [ValidationIssue("error", "topology_not_object",
                                "The topology must be an object.", "topology")]

    # --- nodes ---------------------------------------------------------------
    nodes: Dict[str, Dict[str, Any]] = {}
    raw_nodes = topology.get("nodes")
    if not isinstance(raw_nodes, list):
        issues.append(ValidationIssue("error", "nodes_not_list",
                                      "'nodes' must be a list.", "topology.nodes"))
        raw_nodes = []
    for index, node in enumerate(raw_nodes):
        path = f"topology.nodes[{index}]"
        if not isinstance(node, dict):
            issues.append(ValidationIssue("error", "node_invalid",
                                          "Each node must be an object.", path))
            continue
        node_id = str(node.get("id") or "").strip()
        if not node_id:
            issues.append(ValidationIssue("error", "node_id_missing",
                                          "Every node needs a stable id.", f"{path}.id"))
            continue
        if node_id in nodes:
            issues.append(ValidationIssue("error", "node_id_duplicate",
                                          f"Duplicate node id {node_id!r}.", f"{path}.id"))
            continue
        try:
            x = float(node.get("x"))
            y = float(node.get("y", 0.0))
            finite = bool(np.isfinite(x) and np.isfinite(y))
            if "z" in node:
                finite = finite and bool(np.isfinite(float(node.get("z"))))
        except (TypeError, ValueError):
            finite = False
        if not finite:
            issues.append(ValidationIssue(
                "error", "node_coordinates_invalid",
                f"Node {node_id} needs finite numeric coordinates.", f"{path}.x"))
            continue
        nodes[node_id] = node

    # Coincident nodes make zero-length edges and degenerate cells later.
    positions: Dict[Tuple[float, float], str] = {}
    for node_id, node in nodes.items():
        key = (round(float(node["x"]), 9), round(float(node.get("y", 0.0)), 9))
        if key in positions:
            issues.append(ValidationIssue(
                "warning", "duplicate_node_position",
                f"Nodes {positions[key]} and {node_id} share the same position.",
                "topology.nodes"))
        else:
            positions[key] = node_id

    # --- edges ---------------------------------------------------------------
    edges: Dict[str, Dict[str, Any]] = {}
    pairs: Dict[Tuple[str, str], str] = {}
    raw_edges = topology.get("edges")
    if not isinstance(raw_edges, list):
        issues.append(ValidationIssue("error", "edges_not_list",
                                      "'edges' must be a list.", "topology.edges"))
        raw_edges = []
    for index, edge in enumerate(raw_edges):
        path = f"topology.edges[{index}]"
        if not isinstance(edge, dict):
            issues.append(ValidationIssue("error", "edge_invalid",
                                          "Each edge must be an object.", path))
            continue
        edge_id = str(edge.get("id") or "").strip()
        if not edge_id:
            issues.append(ValidationIssue("error", "edge_id_missing",
                                          "Every edge needs a stable id.", f"{path}.id"))
            continue
        if edge_id in edges:
            issues.append(ValidationIssue("error", "edge_id_duplicate",
                                          f"Duplicate edge id {edge_id!r}.", f"{path}.id"))
            continue
        source, target = str(edge.get("source")), str(edge.get("target"))
        missing = [n for n in (source, target) if n not in nodes]
        if missing:
            issues.append(ValidationIssue(
                "error", "edge_reference_invalid",
                f"Edge {edge_id} references unknown node(s): {', '.join(missing)}.",
                f"{path}.source"))
            continue
        if source == target:
            issues.append(ValidationIssue(
                "error", "edge_self_loop",
                f"Edge {edge_id} starts and ends at node {source}.", f"{path}.source"))
            continue
        key = tuple(sorted((source, target)))
        if key in pairs:
            issues.append(ValidationIssue(
                "error", "edge_duplicate",
                f"Edge {edge_id} duplicates edge {pairs[key]} between {source} and {target}.",
                f"{path}.source"))
            continue
        pairs[key] = edge_id
        a, b = nodes[source], nodes[target]
        length = float(np.hypot(float(b["x"]) - float(a["x"]),
                                float(b.get("y", 0.0)) - float(a.get("y", 0.0))))
        if length <= GEOM_EPS:
            issues.append(ValidationIssue(
                "error", "edge_zero_length",
                f"Edge {edge_id} has zero length: nodes {source} and {target} coincide.",
                f"{path}.source"))
            continue
        edges[edge_id] = edge

    # --- cells ---------------------------------------------------------------
    cell_ids: Set[str] = set()
    raw_cells = topology.get("cells")
    if not isinstance(raw_cells, list):
        issues.append(ValidationIssue("error", "cells_not_list",
                                      "'cells' must be a list.", "topology.cells"))
        raw_cells = []
    for index, cell in enumerate(raw_cells):
        path = f"topology.cells[{index}]"
        if not isinstance(cell, dict):
            issues.append(ValidationIssue("error", "cell_invalid",
                                          "Each cell must be an object.", path))
            continue
        cell_id = str(cell.get("id") or "").strip()
        if not cell_id:
            issues.append(ValidationIssue("error", "cell_id_missing",
                                          "Every cell needs a stable id.", f"{path}.id"))
            continue
        if cell_id in cell_ids:
            issues.append(ValidationIssue("error", "cell_id_duplicate",
                                          f"Duplicate cell id {cell_id!r}.", f"{path}.id"))
            continue
        cell_ids.add(cell_id)

        ring = [str(v) for v in (cell.get("nodes") or [])]
        if len(ring) < 3:
            issues.append(ValidationIssue(
                "error", "cell_too_few_nodes",
                f"Cell {cell_id} has {len(ring)} node(s); a cell needs at least 3.",
                f"{path}.nodes"))
            continue
        unknown = [n for n in ring if n not in nodes]
        if unknown:
            issues.append(ValidationIssue(
                "error", "cell_reference_invalid",
                f"Cell {cell_id} references unknown node(s): {', '.join(sorted(set(unknown)))}.",
                f"{path}.nodes"))
            continue
        if len(set(ring)) != len(ring):
            issues.append(ValidationIssue(
                "error", "cell_repeated_node",
                f"Cell {cell_id} visits the same node more than once.", f"{path}.nodes"))
            continue

        broken = [(ring[i], ring[(i + 1) % len(ring)]) for i in range(len(ring))
                  if tuple(sorted((ring[i], ring[(i + 1) % len(ring)]))) not in pairs]
        if broken:
            gaps = ", ".join(f"{a}-{b}" for a, b in broken)
            issues.append(ValidationIssue(
                "error", "cell_loop_broken",
                f"Cell {cell_id} is not a closed loop; no edge connects: {gaps}.",
                f"{path}.nodes"))
            continue

        points = [(float(nodes[n]["x"]), float(nodes[n].get("y", 0.0))) for n in ring]
        if not polygon_is_simple(points):
            issues.append(ValidationIssue(
                "error", "cell_self_intersecting",
                f"Cell {cell_id} crosses itself.", f"{path}.nodes"))
            continue
        if polygon_area(points) <= GEOM_EPS:
            issues.append(ValidationIssue(
                "error", "cell_degenerate",
                f"Cell {cell_id} encloses no area - its nodes are collinear.",
                f"{path}.nodes"))

    # --- domain containment --------------------------------------------------
    if domain:
        for node_id, node in nodes.items():
            point = (float(node["x"]), float(node.get("y", 0.0)))
            try:
                inside = point_in_domain(point, domain)
            except (TypeError, ValueError):
                inside = False
            if not inside:
                issues.append(ValidationIssue(
                    "error", "entity_outside_domain",
                    f"Node {node_id} at ({point[0]:g}, {point[1]:g}) lies outside the domain.",
                    "topology.nodes"))
    return issues


def topology_stats(topology: Dict[str, Any]) -> Dict[str, Any]:
    """Counts and size statistics for the stage header."""
    nodes = node_index(topology)
    lengths: List[float] = []
    for edge in topology.get("edges", []):
        a, b = nodes.get(str(edge.get("source"))), nodes.get(str(edge.get("target")))
        if a and b:
            lengths.append(float(np.hypot(float(b["x"]) - float(a["x"]),
                                          float(b.get("y", 0.0)) - float(a.get("y", 0.0)))))
    areas = [float(c.get("area", 0.0)) for c in topology.get("cells", [])]
    stats: Dict[str, Any] = {
        "node_count": len(nodes),
        "edge_count": len(topology.get("edges", [])),
        "cell_count": len(topology.get("cells", [])),
    }
    if lengths:
        arr = np.asarray(lengths, dtype=float)
        stats["edge_length"] = {
            "min": float(arr.min()), "max": float(arr.max()),
            "mean": float(arr.mean()), "total": float(arr.sum()),
        }
    if areas:
        arr = np.asarray(areas, dtype=float)
        stats["cell_area"] = {
            "min": float(arr.min()), "max": float(arr.max()),
            "mean": float(arr.mean()), "total": float(arr.sum()),
        }
    return stats


# =============================================================================
# Interchange: JSON and CSV
# =============================================================================
def to_json_dict(topology: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "nodes": [dict(n) for n in topology.get("nodes", [])],
        "edges": [dict(e) for e in topology.get("edges", [])],
        "cells": [dict(c) for c in topology.get("cells", [])],
        "next_ids": dict(topology.get("next_ids", {})),
    }


def from_json_dict(payload: Any) -> Dict[str, Any]:
    """Rebuild a topology from parsed JSON, normalising types and derived fields."""
    if not isinstance(payload, dict):
        raise ValueError("Topology JSON must be an object.")
    topology = new_topology()
    for node in payload.get("nodes") or []:
        if not isinstance(node, dict):
            continue
        entry = {
            "id": str(node.get("id")),
            "x": float(node.get("x", 0.0)),
            "y": float(node.get("y", 0.0)),
            "region": str(node.get("region") or ""),
            "metadata": dict(node.get("metadata") or {}),
        }
        if node.get("z") is not None:
            entry["z"] = float(node["z"])
        topology["nodes"].append(entry)
    for edge in payload.get("edges") or []:
        if not isinstance(edge, dict):
            continue
        topology["edges"].append({
            "id": str(edge.get("id")),
            "source": str(edge.get("source")),
            "target": str(edge.get("target")),
            "directed": bool(edge.get("directed", False)),
            "boundary": str(edge.get("boundary") or ""),
        })
    for cell in payload.get("cells") or []:
        if not isinstance(cell, dict):
            continue
        topology["cells"].append({
            "id": str(cell.get("id")),
            "nodes": [str(v) for v in (cell.get("nodes") or [])],
            "edges": [str(v) for v in (cell.get("edges") or [])],
            "type": str(cell.get("type") or "generic"),
            "region": str(cell.get("region") or ""),
            "centroid": list(cell.get("centroid") or [0.0, 0.0]),
            "area": float(cell.get("area", 0.0)),
            "neighbors": [str(v) for v in (cell.get("neighbors") or [])],
        })
    recompute_cell_geometry(topology)
    compute_neighbors(topology)
    return topology


def nodes_to_csv(topology: Dict[str, Any]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["id", "x", "y", "z", "region"])
    for node in topology.get("nodes", []):
        writer.writerow([node.get("id"), node.get("x"), node.get("y"),
                         node.get("z", ""), node.get("region", "")])
    return buffer.getvalue()


def edges_to_csv(topology: Dict[str, Any]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["id", "source", "target", "directed", "boundary"])
    for edge in topology.get("edges", []):
        writer.writerow([edge.get("id"), edge.get("source"), edge.get("target"),
                         int(bool(edge.get("directed"))), edge.get("boundary", "")])
    return buffer.getvalue()


def cells_to_csv(topology: Dict[str, Any]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["id", "type", "region", "area", "centroid_x", "centroid_y",
                     "nodes", "edges", "neighbors"])
    for cell in topology.get("cells", []):
        centroid = list(cell.get("centroid") or [0.0, 0.0])
        writer.writerow([
            cell.get("id"), cell.get("type", ""), cell.get("region", ""),
            cell.get("area", 0.0),
            centroid[0] if len(centroid) > 0 else "",
            centroid[1] if len(centroid) > 1 else "",
            " ".join(str(v) for v in cell.get("nodes", [])),
            " ".join(str(v) for v in cell.get("edges", [])),
            " ".join(str(v) for v in cell.get("neighbors", [])),
        ])
    return buffer.getvalue()


def nodes_from_csv(text: str) -> List[Dict[str, Any]]:
    """Parse a node CSV. Raises ValueError with the offending row on bad input."""
    rows = list(csv.DictReader(io.StringIO(text)))
    parsed: List[Dict[str, Any]] = []
    for number, row in enumerate(rows, start=2):
        node_id = (row.get("id") or "").strip()
        if not node_id:
            raise ValueError(f"Row {number}: missing node id.")
        try:
            entry: Dict[str, Any] = {
                "id": node_id,
                "x": float(row.get("x")),
                "y": float(row.get("y") or 0.0),
                "region": (row.get("region") or "").strip(),
                "metadata": {},
            }
        except (TypeError, ValueError):
            raise ValueError(f"Row {number}: x and y must be numbers.")
        z = (row.get("z") or "").strip()
        if z:
            try:
                entry["z"] = float(z)
            except ValueError:
                raise ValueError(f"Row {number}: z must be a number when present.")
        parsed.append(entry)
    return parsed


def edges_from_csv(text: str) -> List[Dict[str, Any]]:
    rows = list(csv.DictReader(io.StringIO(text)))
    parsed: List[Dict[str, Any]] = []
    for number, row in enumerate(rows, start=2):
        edge_id = (row.get("id") or "").strip()
        source = (row.get("source") or "").strip()
        target = (row.get("target") or "").strip()
        if not edge_id or not source or not target:
            raise ValueError(f"Row {number}: id, source and target are all required.")
        directed = str(row.get("directed") or "").strip().lower() in ("1", "true", "yes")
        parsed.append({"id": edge_id, "source": source, "target": target,
                       "directed": directed, "boundary": (row.get("boundary") or "").strip()})
    return parsed
