"""
ABM approach: the project's existing Cellular Potts (Glazier-Graner-Hogeweg) engine.

This adapter REUSES ``abm_engine`` entirely -- cell types, adhesion, volume and
surface constraints, chemotaxis, division, death and diffusible fields are all the
existing implementation. What the adapter adds is what the shared workflow needs
and the engine's own ``simulate()`` cannot provide:

  * a step loop built from ``run_mcs`` so a run can be paused and cancelled
    (``simulate()`` is a closed loop with no callbacks),
  * deterministic seeding, so a seeded run repeats exactly,
  * a per-cell table (id, type, position, volume, age, parent, alive) for the
    Cell # panel and per-cell export, which ``simulate()`` does not return.

Numerical assumptions and limitations
-------------------------------------
* The CPM lattice is independent of the PDE mesh: it is a regular pixel lattice
  sized from the domain, not the unstructured triangulation. Cell positions are
  reported in lattice coordinates and in domain units.
* Monte Carlo dynamics are stochastic. Determinism holds only for a fixed seed AND
  a fixed number of pixel attempts, because the engine draws from the module-level
  ``random``; running two ABM simulations concurrently in one process would
  interleave those draws.
"""

import random
import threading
from typing import Any, Dict, List, Optional

import numpy as np

#: Serialises ABM execution across the whole process.
#:
#: abm_engine draws from the module-level `random`, which is process-global, while the
#: run manager runs every simulation on its own thread in one process. Two overlapping
#: ABM runs therefore drew from the same generator and perturbed each other, so a
#: seeded run reproduced only when nothing else was running -- despite the adapter
#: advertising deterministic_with_seed=True. Holding this for the duration of a run is
#: what makes that advertised guarantee true.
_ABM_GLOBAL_RNG_LOCK = threading.Lock()

from approach_base import (
    ApproachAdapter,
    Capabilities,
    RunContext,
    register_approach,
)
from geometry import ValidationIssue, domain_bounds, domain_dimension

DEFAULTS: Dict[str, Any] = {
    "num_mcs": 200,
    "save_every": 10,
    "temperature": 10.0,
    "pixel_attempts_factor": 1.0,
    "seed": 0,
    "grid": {"width": 60, "height": 60},
    "cell_types": [],
    "placement": "random",
}


def _merged(config: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(DEFAULTS)
    merged["grid"] = dict(DEFAULTS["grid"])
    for key, value in (config or {}).items():
        if key == "grid" and isinstance(value, dict):
            merged["grid"] = {**merged["grid"], **value}
        else:
            merged[key] = value
    return merged


class ABMAdapter(ApproachAdapter):
    approach_id = "abm"
    label = "ABM (Cellular Potts)"

    # -- capabilities -------------------------------------------------------
    def get_capabilities(self) -> Capabilities:
        try:
            import abm_engine  # noqa: F401
            available, reason = True, ""
        except Exception as exc:                       # pragma: no cover
            available, reason = False, (
                f"The bundled ABM engine could not be imported ({type(exc).__name__}).")
        return Capabilities(
            approach_id=self.approach_id,
            label=self.label,
            available=available,
            unavailable_reason=reason,
            dimensions=(2,),
            supports_pause=True,
            supports_cancel=True,
            supports_fields=True,
            supports_seed=True,
            deterministic_with_seed=True,
            supports_export=True,
            engine_name="abm_engine (Glazier-Graner-Hogeweg Cellular Potts)",
            engine_version="in-tree",
            requirements=["numpy"],
            notes=("Cell-level Monte Carlo dynamics on a pixel lattice: adhesion, "
                   "volume and surface constraints, chemotaxis, division and death. "
                   "Does not require CompuCell3D."),
        )

    # -- validation ---------------------------------------------------------
    def validate(self, project: Dict[str, Any]) -> List[ValidationIssue]:
        config = _merged(self.approach_config(project))
        issues: List[ValidationIssue] = []
        path = "approaches.abm"

        domain = (project or {}).get("domain")
        if domain and domain_dimension(domain) != 2:
            issues.append(ValidationIssue(
                "error", "abm_requires_2d",
                "The ABM approach runs on a 2D lattice; the current domain is 1D.",
                "domain.kind"))

        grid = config.get("grid") or {}
        for key in ("width", "height"):
            try:
                value = int(grid.get(key, 0))
            except (TypeError, ValueError):
                issues.append(ValidationIssue("error", f"abm_grid_{key}_invalid",
                                              f"Lattice {key} must be a whole number.",
                                              f"{path}.grid.{key}"))
                continue
            if value < 8:
                issues.append(ValidationIssue(
                    "error", f"abm_grid_{key}_too_small",
                    f"Lattice {key} must be at least 8 pixels (got {value}).",
                    f"{path}.grid.{key}"))
            elif value > 512:
                issues.append(ValidationIssue(
                    "warning", f"abm_grid_{key}_large",
                    f"A lattice {key} of {value} makes each Monte Carlo step expensive.",
                    f"{path}.grid.{key}"))

        for key, minimum in (("num_mcs", 1), ("save_every", 1)):
            try:
                value = int(config.get(key))
            except (TypeError, ValueError):
                issues.append(ValidationIssue("error", f"abm_{key}_invalid",
                                              f"{key} must be a whole number.", f"{path}.{key}"))
                continue
            if value < minimum:
                issues.append(ValidationIssue(
                    "error", f"abm_{key}_too_small",
                    f"{key} must be at least {minimum} (got {value}).", f"{path}.{key}"))

        try:
            if float(config.get("temperature")) <= 0:
                issues.append(ValidationIssue(
                    "error", "abm_temperature_not_positive",
                    "Temperature must be greater than zero; at zero the lattice cannot "
                    "accept any energetically unfavourable move and nothing evolves.",
                    f"{path}.temperature"))
        except (TypeError, ValueError):
            issues.append(ValidationIssue("error", "abm_temperature_invalid",
                                          "Temperature must be a number.", f"{path}.temperature"))

        cell_types = config.get("cell_types")
        if not isinstance(cell_types, list) or not cell_types:
            issues.append(ValidationIssue(
                "error", "abm_no_cell_types",
                "Define at least one cell type before running an agent-based simulation.",
                f"{path}.cell_types"))
        else:
            seen_ids = set()
            for index, spec in enumerate(cell_types):
                item_path = f"{path}.cell_types[{index}]"
                if not isinstance(spec, dict):
                    issues.append(ValidationIssue("error", "abm_cell_type_invalid",
                                                  "Each cell type must be an object.", item_path))
                    continue
                try:
                    type_id = int(spec.get("type_id"))
                except (TypeError, ValueError):
                    issues.append(ValidationIssue("error", "abm_cell_type_id_invalid",
                                                  "Each cell type needs a whole-number type_id.",
                                                  f"{item_path}.type_id"))
                    continue
                if type_id <= 0:
                    issues.append(ValidationIssue(
                        "error", "abm_cell_type_id_reserved",
                        "type_id 0 is reserved for medium (empty space); use 1 or above.",
                        f"{item_path}.type_id"))
                elif type_id in seen_ids:
                    issues.append(ValidationIssue("error", "abm_cell_type_id_duplicate",
                                                  f"Duplicate cell type id {type_id}.",
                                                  f"{item_path}.type_id"))
                seen_ids.add(type_id)
                try:
                    if float(spec.get("target_volume", 25)) <= 0:
                        issues.append(ValidationIssue(
                            "error", "abm_target_volume_not_positive",
                            "Target volume must be greater than zero.",
                            f"{item_path}.target_volume"))
                except (TypeError, ValueError):
                    issues.append(ValidationIssue("error", "abm_target_volume_invalid",
                                                  "Target volume must be a number.",
                                                  f"{item_path}.target_volume"))
                count = spec.get("initial_count", spec.get("count"))
                if count is not None:
                    try:
                        if int(count) < 0:
                            issues.append(ValidationIssue(
                                "error", "abm_initial_count_negative",
                                "Initial cell count cannot be negative.",
                                f"{item_path}.initial_count"))
                    except (TypeError, ValueError):
                        issues.append(ValidationIssue("error", "abm_initial_count_invalid",
                                                      "Initial cell count must be a whole number.",
                                                      f"{item_path}.initial_count"))

            # A configuration that seeds nothing builds a valid but EMPTY lattice and
            # would report a "successful" run with no cells, so refuse it up front.
            if not config.get("initial_config"):
                total = 0
                for spec in cell_types:
                    if not isinstance(spec, dict):
                        continue
                    try:
                        total += max(0, int(spec.get("initial_count", spec.get("count", 0)) or 0))
                    except (TypeError, ValueError):
                        continue
                if total <= 0:
                    issues.append(ValidationIssue(
                        "error", "abm_no_initial_cells",
                        "No cells would be placed. Give at least one cell type a positive "
                        "initial count, or supply an explicit initial_config.",
                        f"{path}.cell_types"))
        return issues

    # -- compilation --------------------------------------------------------
    def compile(self, project: Dict[str, Any]) -> Dict[str, Any]:
        errors = [i for i in self.validate(project) if i.severity == "error"]
        if errors:
            raise ValueError("ABM configuration is not valid: " +
                             "; ".join(i.message for i in errors))
        config = _merged(self.approach_config(project))

        # The blueprint shape the existing engine already understands, so the CPM is
        # constructed by abm_engine.build_cpm_from_blueprint and not re-derived here.
        # Cell seeding must go through 'initial_config' -- that is the key the engine
        # reads. A per-type 'initial_count' is translated here rather than invented,
        # because a blueprint without initial_config builds a valid but EMPTY lattice.
        blueprint: Dict[str, Any] = {
            "type": "ABM",
            "name": config.get("name", "ABM run"),
            "description": config.get("description", ""),
            "grid": dict(config["grid"]),
            "temperature": float(config["temperature"]),
            "cell_types": [dict(spec) for spec in config["cell_types"]],
        }

        initial_config = config.get("initial_config")
        if not initial_config:
            initial_config = []
            default_region = str(config.get("placement", "random"))
            for spec in config["cell_types"]:
                count = spec.get("initial_count", spec.get("count", 0))
                try:
                    count = int(count)
                except (TypeError, ValueError):
                    count = 0
                if count > 0:
                    initial_config.append({
                        "type_id": int(spec["type_id"]),
                        "count": count,
                        "radius": int(spec.get("radius", 3)),
                        "region": str(spec.get("region", default_region)),
                    })
        blueprint["initial_config"] = initial_config

        if config.get("adhesion_matrix"):
            blueprint["adhesion_matrix"] = dict(config["adhesion_matrix"])
        if config.get("fields"):
            blueprint["fields"] = [dict(f) for f in config["fields"]]

        from abm_engine import build_cpm_from_blueprint
        # Build once at compile time: a blueprint the engine rejects must fail
        # compilation, never surface later as a failed run reported as ready.
        try:
            build_cpm_from_blueprint(blueprint)
        except Exception as exc:
            raise ValueError(f"The ABM engine rejected this configuration: "
                             f"{type(exc).__name__}: {exc}") from exc

        return {
            "approach": self.approach_id,
            "blueprint": blueprint,
            "config": config,
            "num_mcs": int(config["num_mcs"]),
            "save_every": int(config["save_every"]),
            "seed": config.get("seed"),
            "engine_version": "in-tree",
        }

    # -- execution ----------------------------------------------------------
    def run(self, compiled: Dict[str, Any], context: RunContext) -> Dict[str, Any]:
        config = compiled["config"]
        num_mcs = int(compiled["num_mcs"])
        save_every = max(1, int(compiled["save_every"]))
        seed = context.seed if context.seed is not None else compiled.get("seed")

        # The engine draws from the module-level `random`, which is PROCESS-GLOBAL, and
        # the run manager puts every run on its own thread in one process. Two ABM runs
        # overlapping therefore interleave their draws from the same generator, so a
        # seeded run reproduced only when nothing else happened to be running.
        # Measured: seed 4242 run twice concurrently gave lattice digests
        # f017292d6b341b7f vs 82b576d8ad10d69e -- while get_capabilities() advertised
        # deterministic_with_seed=True. A published seed that reproduces a figure only
        # when the machine is otherwise idle is not reproducible.
        #
        # This lock makes the advertised guarantee true: ABM runs execute one at a time,
        # so each one owns the global RNG for its whole duration. It serialises ABM
        # concurrency, which is the honest price of the engine's module-level RNG. The
        # better fix is to thread per-run random.Random(seed) / np.random.default_rng()
        # instances through abm_engine.py, which would restore concurrency -- that is a
        # change to the pre-existing engine and is deliberately not bundled here.
        with _ABM_GLOBAL_RNG_LOCK:
            return self._run_locked(compiled, config, num_mcs, save_every, seed, context)

    def _run_locked(self, compiled, config, num_mcs, save_every, seed, context):
        """The real run body. Called with the process-wide ABM RNG lock held."""
        from abm_engine import build_cpm_from_blueprint

        if seed is not None:
            random.seed(int(seed))
            np.random.seed(int(seed) % (2 ** 32))
            context.log(f"ABM seeded with {int(seed)} (run is reproducible).")
        else:
            context.log("No seed supplied; this ABM run is not reproducible.", "warning")

        cpm = build_cpm_from_blueprint(compiled["blueprint"])
        attempts = int(cpm.width * cpm.height * float(config.get("pixel_attempts_factor", 1.0)))

        lattice_frames: List[Any] = []
        cell_counts: List[Dict[str, int]] = []
        cell_tables: List[List[Dict[str, Any]]] = []
        field_frames: Dict[str, List[Any]] = {name: [] for name in getattr(cpm, "fields", {})}
        times: List[int] = []

        def record(step: int) -> None:
            lattice_frames.append(cpm._get_lattice_colored())
            cell_counts.append(cpm._get_cell_counts())
            cell_tables.append(self._cell_table(cpm, config))
            for name, field_obj in getattr(cpm, "fields", {}).items():
                grid = getattr(field_obj, "grid", None)
                field_frames.setdefault(name, []).append(
                    grid.copy().tolist() if grid is not None else [])
            times.append(int(step))

        record(0)
        for step in range(1, num_mcs + 1):
            context.checkpoint()
            cpm.run_mcs(attempts)
            if step % save_every == 0 or step == num_mcs:
                record(step)
                context.progress(step / num_mcs, f"MCS {step}/{num_mcs}")

        divisions = sum(1 for c in cpm.cells.values() if getattr(c, "parent_id", None) is not None)
        alive = sum(1 for c in cpm.cells.values() if c.alive)
        context.log(f"ABM finished: {alive} live cell(s) after {num_mcs} MCS, "
                    f"{divisions} division event(s).")

        return {
            "approach": self.approach_id,
            "kind": "agents",
            "type": "ABM",
            "width": cpm.width,
            "height": cpm.height,
            "t": times,
            "lattice_frames": lattice_frames,
            "cell_counts": cell_counts,
            "cells": cell_tables,
            "fields": field_frames,
            "cell_types": {
                ct.type_id: {"name": ct.name, "color": list(ct.color)}
                for ct in cpm.cell_types.values()
            },
            "total_mcs": num_mcs,
            "temperature": cpm.temperature,
            "seed": int(seed) if seed is not None else None,
            "summary": {
                "live_cells": alive,
                "division_events": divisions,
                "frames": len(times),
                "deterministic": seed is not None,
            },
        }

    @staticmethod
    def _cell_table(cpm: Any, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Per-cell rows for the Cell # panel.

        The engine's own ``simulate()`` returns only lattice frames and counts, so
        this is read directly off the live cell objects each time a frame is saved.
        """
        scale_x = float(config.get("units_per_pixel_x", 1.0))
        scale_y = float(config.get("units_per_pixel_y", 1.0))
        rows: List[Dict[str, Any]] = []
        for cell in cpm.cells.values():
            rows.append({
                "id": int(cell.cell_id),
                "type_id": int(cell.cell_type.type_id),
                "type": str(cell.cell_type.name),
                "state": "alive" if cell.alive else "dead",
                "alive": bool(cell.alive),
                "x": float(cell.center_x) * scale_x,
                "y": float(cell.center_y) * scale_y,
                "lattice_x": float(cell.center_x),
                "lattice_y": float(cell.center_y),
                "volume": int(cell.volume),
                "surface": int(cell.surface),
                "age": int(cell.age),
                "parent_id": (int(cell.parent_id) if getattr(cell, "parent_id", None) is not None
                              else None),
                "generation": int(getattr(cell, "generation", 0)),
                "measurements": {k: float(v) for k, v in (cell.internal_state or {}).items()},
            })
        rows.sort(key=lambda r: r["id"])
        return rows

    def export_configuration(self, project: Dict[str, Any]) -> Dict[str, Any]:
        config = _merged(self.approach_config(project))
        return {
            "approach": self.approach_id,
            "label": self.label,
            "engine": "abm_engine (in-tree Cellular Potts)",
            "configuration": config,
        }


def lattice_from_domain(domain: Dict[str, Any], pixels_per_unit: float = 1.0) -> Dict[str, int]:
    """Suggest a lattice size from the domain extent.

    Kept separate from validation so the UI can offer a sensible default without
    the adapter silently overriding whatever the user chose.
    """
    x0, y0, x1, y1 = domain_bounds(domain)
    width = max(8, int(round((x1 - x0) * float(pixels_per_unit))))
    height = max(8, int(round((y1 - y0) * float(pixels_per_unit))))
    return {"width": width, "height": height}


ABM_ADAPTER = register_approach(ABMAdapter())
