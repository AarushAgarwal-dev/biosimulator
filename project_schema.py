"""
Project format: save, import, export, migration.

Stage 12. One versioned document holds every stage of the workflow, so a project is
reproducible: geometry, topology, mesh settings, model fields, conditions, the
SELECTED approach, each approach's own configuration, visualisation settings and run
history.

Versioning
----------
``schema_version`` is an integer. :func:`migrate` upgrades an older document to the
current version and records what it did in ``migrations_applied``, so loading an old
project never silently reinterprets it. A document from a NEWER version is refused
rather than partially read -- guessing at fields we do not know about is how a project
gets quietly corrupted.

Each approach keeps its own block under ``approaches``. Switching the selected
approach therefore preserves the common geometry/mesh/model AND every approach's
individual settings, which is what makes comparing approaches practical.
"""

import copy
import csv
import io
import json
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import approach_base
import boundary_conditions as bc
import geometry as geo
import meshing
import pde_model
import topology as topo
from geometry import ValidationIssue, issues_to_dicts

#: Current schema version. Bump when a change needs a migration step.
SCHEMA_VERSION = 2

#: Versions this module can read (and migrate forward from).
SUPPORTED_VERSIONS: Tuple[int, ...] = (1, 2)


def new_project(name: str = "Untitled project",
                domain: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """A minimal valid project: a 1D interval with one diffusing field."""
    domain = domain or geo.make_interval(1.0, units="um")
    return {
        "schema_version": SCHEMA_VERSION,
        "name": name,
        "created_at": time.time(),
        "modified_at": time.time(),
        "domain": domain,
        "topology": topo.new_topology(),
        "mesh_settings": {"element_count": 40},
        "mesh": None,
        "model": {
            "parameters": {"D": 1.0},
            "fields": [dict(pde_model.FIELD_DEFAULTS, name="u", units="mM",
                            diffusion="D", initial="0.0", reaction="0", source="0",
                            t_start=0.0, t_end=1.0, output_interval=0.1)],
        },
        "conditions": [],
        "selected_approach": None,
        "approaches": {},
        "visualization": {"view": "mesh", "colormap": "viridis", "show_legend": True},
        "run_history": [],
    }


# =============================================================================
# Migration
# =============================================================================
class ProjectVersionError(ValueError):
    """The document's schema version cannot be read by this build."""


def migrate(document: Dict[str, Any]) -> Dict[str, Any]:
    """Upgrade a project document to :data:`SCHEMA_VERSION`.

    Returns a NEW document; the input is not modified. Every step is recorded in
    ``migrations_applied`` so a load is auditable.
    """
    if not isinstance(document, dict):
        raise ProjectVersionError("A project must be an object.")
    project = copy.deepcopy(document)
    raw_version = project.get("schema_version", 1)
    try:
        version = int(raw_version)
    except (TypeError, ValueError):
        raise ProjectVersionError(f"schema_version {raw_version!r} is not an integer.")

    if version > SCHEMA_VERSION:
        raise ProjectVersionError(
            f"This project was saved by a newer version (schema {version}); this build "
            f"reads up to {SCHEMA_VERSION}. Update the application rather than loading it "
            f"partially."
        )
    if version < min(SUPPORTED_VERSIONS):
        raise ProjectVersionError(
            f"Schema version {version} is too old to read. Supported: "
            f"{', '.join(str(v) for v in SUPPORTED_VERSIONS)}."
        )

    applied: List[str] = list(project.get("migrations_applied") or [])

    if version == 1:
        # v1 -> v2: approach configuration moved from a single flat 'approach_config'
        # block to a per-approach map, so switching approaches stops destroying the
        # other approaches' settings. Also renames 'bcs' to 'conditions'.
        legacy_config = project.pop("approach_config", None)
        selected = project.get("selected_approach")
        approaches = project.get("approaches")
        if not isinstance(approaches, dict):
            approaches = {}
        if legacy_config and selected:
            approaches.setdefault(str(selected), legacy_config)
            applied.append("v1->v2: moved approach_config into approaches[selected]")
        project["approaches"] = approaches

        if "bcs" in project and "conditions" not in project:
            project["conditions"] = project.pop("bcs")
            applied.append("v1->v2: renamed 'bcs' to 'conditions'")

        # v1 stored the third approach under the wrong spelling in some documents.
        if "mcp" in project["approaches"]:
            project["approaches"]["mpc"] = project["approaches"].pop("mcp")
            applied.append("v1->v2: corrected approach id 'mcp' to 'mpc'")
        if str(project.get("selected_approach") or "").lower() == "mcp":
            project["selected_approach"] = "mpc"
            applied.append("v1->v2: corrected selected approach 'mcp' to 'mpc'")

        version = 2

    project["schema_version"] = SCHEMA_VERSION
    if applied:
        project["migrations_applied"] = applied

    # Fill in anything a partial document is missing, so downstream code can rely on
    # the keys existing without defensive lookups everywhere.
    template = new_project()
    for key, value in template.items():
        project.setdefault(key, copy.deepcopy(value))
    return project


# =============================================================================
# Serialisation
# =============================================================================
def to_json(project: Dict[str, Any], indent: int = 2) -> str:
    payload = copy.deepcopy(project)
    payload["schema_version"] = SCHEMA_VERSION
    payload["modified_at"] = time.time()
    return json.dumps(payload, indent=indent, sort_keys=False, default=_json_default)


def _json_default(value: Any) -> Any:
    # numpy scalars and arrays reach here from mesh/statistics values.
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"{type(value).__name__} is not JSON serialisable")


def from_json(text: str) -> Dict[str, Any]:
    """Parse and migrate a project document."""
    try:
        document = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ProjectVersionError(f"The project file is not valid JSON: {exc.msg} "
                                  f"(line {exc.lineno}, column {exc.colno}).") from None
    return migrate(document)


# =============================================================================
# Aggregate validation
# =============================================================================
def validate_project(project: Dict[str, Any],
                     approach_id: Optional[str] = None) -> Dict[str, Any]:
    """Validate every stage and report per-stage status.

    Returns ``{"valid": bool, "stages": {name: {...}}, "issues": [...]}``. Per-stage
    status is what the UI needs to show a tick or a cross beside each step, rather
    than one opaque pass/fail for the whole project.
    """
    stages: Dict[str, Dict[str, Any]] = {}
    all_issues: List[ValidationIssue] = []

    def record(stage: str, issues: Sequence[ValidationIssue],
               skipped: bool = False, note: str = "") -> None:
        errors = [i for i in issues if i.severity == "error"]
        warnings = [i for i in issues if i.severity == "warning"]
        stages[stage] = {
            "status": "skipped" if skipped else ("invalid" if errors else "valid"),
            "errors": len(errors),
            "warnings": len(warnings),
            "issues": issues_to_dicts(issues),
            "note": note,
        }
        all_issues.extend(issues)

    domain = project.get("domain")
    domain_issues = geo.validate_domain(domain)
    record("domain", domain_issues)
    domain_ok = geo.is_valid(domain_issues)

    record("topology", topo.validate_topology(project.get("topology") or topo.new_topology(),
                                              domain if domain_ok else None))

    mesh = project.get("mesh")
    if not mesh:
        record("mesh", [], skipped=True, note="No mesh generated yet.")
    else:
        record("mesh", meshing.validate_mesh(mesh, domain if domain_ok else None))

    model = project.get("model") or {}
    model_issues = pde_model.validate_model(model, domain if domain_ok else None)
    record("model", model_issues)

    field_names = [str(f.get("name")) for f in (model.get("fields") or [])
                   if isinstance(f, dict) and f.get("name")]
    record("conditions", bc.validate_conditions(project.get("conditions"),
                                                domain if isinstance(domain, dict) else {},
                                                field_names))

    selected = str(approach_id or project.get("selected_approach") or "")
    if not selected:
        record("approach", [ValidationIssue(
            "error", "no_approach_selected",
            "Select an approach: ABM, CompuCell3D or MPC.", "selected_approach")])
    else:
        try:
            adapter = approach_base.get_approach(selected)
        except KeyError as exc:
            record("approach", [ValidationIssue(
                "error", "approach_unknown", str(exc), "selected_approach")])
        else:
            capabilities = adapter.get_capabilities()
            issues = list(adapter.validate(project))
            if not capabilities.available:
                issues.append(ValidationIssue(
                    "error", "approach_unavailable",
                    capabilities.unavailable_reason
                    or f"{capabilities.label} is not available on this machine.",
                    "selected_approach"))
            record("approach", issues,
                   note=f"{capabilities.label} ({'available' if capabilities.available else 'unavailable'})")

    errors = [i for i in all_issues if i.severity == "error"]
    return {
        "valid": not errors,
        "error_count": len(errors),
        "warning_count": len([i for i in all_issues if i.severity == "warning"]),
        "stages": stages,
        "issues": issues_to_dicts(all_issues),
    }


# =============================================================================
# Exports
# =============================================================================
def export_topology_csv(project: Dict[str, Any]) -> Dict[str, str]:
    topology = project.get("topology") or topo.new_topology()
    return {
        "nodes.csv": topo.nodes_to_csv(topology),
        "edges.csv": topo.edges_to_csv(topology),
        "cells.csv": topo.cells_to_csv(topology),
    }


def export_run_metadata(record: Any) -> Dict[str, Any]:
    """Metadata for one run: what ran, with which engine, and how it ended."""
    payload = record.to_dict() if hasattr(record, "to_dict") else dict(record)
    payload["configuration_snapshot"] = getattr(record, "snapshot", None)
    payload["exported_at"] = time.time()
    payload["schema_version"] = SCHEMA_VERSION
    return payload


def export_cells_csv(results: Dict[str, Any], frame: Optional[int] = None) -> str:
    """Per-cell rows. ``frame`` selects one time point; omitted exports every frame."""
    frames = results.get("cells") or []
    times = results.get("t") or list(range(len(frames)))
    if frame is not None:
        if not (0 <= int(frame) < len(frames)):
            raise IndexError(f"Frame {frame} is outside 0..{len(frames) - 1}.")
        selected = [(times[int(frame)], frames[int(frame)])]
    else:
        selected = list(zip(times, frames))

    # z is included: the CompuCell3D adapter parses cell.zCOM correctly and this export
    # used to drop it, which made a 3D result indistinguishable from a 2D one the
    # moment it left the app. A 2D run simply reports z = 0 for every cell, so the
    # extra column costs nothing and its absence silently destroyed information.
    columns = ["time", "id", "type", "state", "alive", "x", "y", "z", "volume",
               "surface", "age", "parent_id", "generation"]
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(columns)
    for time_value, rows in selected:
        for row in rows or []:
            writer.writerow([time_value] + [row.get(key, "") for key in columns[1:]])
    return buffer.getvalue()


def export_timeseries_csv(results: Dict[str, Any]) -> str:
    """Time series for a control run (MPC) or aggregate counts for an agent run."""
    times = results.get("t") or []
    series = results.get("series")
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")

    if isinstance(series, dict) and series:
        names = sorted(series)
        writer.writerow(["time"] + names)
        for index, time_value in enumerate(times):
            writer.writerow([time_value] + [
                series[name][index] if index < len(series[name]) else "" for name in names])
        return buffer.getvalue()

    counts = results.get("cell_counts") or []
    keys: List[str] = sorted({k for entry in counts if isinstance(entry, dict) for k in entry})
    writer.writerow(["time"] + keys)
    for index, time_value in enumerate(times):
        entry = counts[index] if index < len(counts) else {}
        writer.writerow([time_value] + [entry.get(key, "") for key in keys])
    return buffer.getvalue()


def export_cc3d_package(project: Dict[str, Any]) -> Dict[str, str]:
    """Files for a runnable CompuCell3D project. Works without CC3D installed."""
    adapter = approach_base.get_approach("cc3d")
    exported = adapter.export_configuration(project)
    files = dict(exported.get("files") or {})
    files["project.json"] = to_json(project)
    return files


def export_project_bundle(project: Dict[str, Any],
                          runs: Optional[Sequence[Any]] = None) -> Dict[str, str]:
    """Everything needed to reproduce the project: document, topology, run metadata."""
    bundle: Dict[str, str] = {"project.json": to_json(project)}
    bundle.update(export_topology_csv(project))
    for index, record in enumerate(runs or []):
        run_id = getattr(record, "run_id", f"run{index}")
        bundle[f"runs/{run_id}.json"] = json.dumps(
            export_run_metadata(record), indent=2, default=_json_default)
    return bundle


def record_run_in_history(project: Dict[str, Any], record: Any) -> Dict[str, Any]:
    """Append a compact run entry to the project's history."""
    history = project.setdefault("run_history", [])
    payload = record.to_dict() if hasattr(record, "to_dict") else dict(record)
    history.append({
        "run_id": payload.get("run_id"),
        "approach": payload.get("approach"),
        "state": payload.get("state"),
        "created_at": payload.get("created_at"),
        "finished_at": payload.get("finished_at"),
        "seed": payload.get("seed"),
        "engine": payload.get("engine"),
        "error": payload.get("error"),
    })
    project["modified_at"] = time.time()
    return project
