"""
CompuCell3D approach: drive the external CC3D simulator.

CompuCell3D is NOT reimplemented here. This adapter detects an installation, maps
the shared project model onto a CC3D project (CC3DML XML plus a Python steppable),
runs it under control, parses its output, and imports the result into the shared
viewer.

Honest unavailability
---------------------
When CC3D is not installed, this adapter:

  * reports ``available=False`` with the exact dependency that is missing,
  * still validates configuration and still exports a runnable CC3D project, so the
    work is not lost and can be run on a machine that has it,
  * refuses to run, raising :class:`ApproachUnavailable`,
  * NEVER silently falls back to the ABM engine or to the MPC approach, and never
    fabricates results.

The last point matters because ABM here is also a Cellular Potts model, so a
fallback would look plausible and be scientifically dishonest: contact energies,
lattice type and the CC3D version all differ. MPC is not even the same kind of
object -- it is a control method, not a discretisation -- so substituting it would be
worse still.

Backends, in detection order
----------------------------
1. ``cc3d`` importable in this interpreter (a pip/conda CompuCell3D install).
2. ``BIOSIM_CC3D_RUNSCRIPT`` pointing at a runScript executable.
3. A CC3D run script on PATH (``runScript.sh`` / ``runScript.bat``).
4. Remote execution on AWS Batch, when all four of ``BIOSIM_AWS_REGION``,
   ``BIOSIM_CC3D_JOB_QUEUE``, ``BIOSIM_CC3D_JOB_DEFINITION`` and
   ``BIOSIM_CC3D_BUCKET`` are set and boto3 imports.

Local backends are tried first because running in-process is faster and free; the
remote backend is what lets a host that cannot install the engine offer it at all.
Whichever one wins is named in :meth:`get_capabilities`, so the user knows where
their simulation is about to execute.

The remote backend additionally enforces a run-length cap
(:data:`REMOTE_RUN_LENGTH_CAP_MINUTES`) at validation time -- see ``cc3d_remote``.
"""

import json
import os
import shutil
import subprocess
import time
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Tuple

import cc3d_remote
from approach_base import (
    ApproachAdapter,
    ApproachUnavailable,
    Capabilities,
    RunCancelled,
    RunContext,
    register_approach,
)
from geometry import ValidationIssue, domain_dimension

RUNSCRIPT_ENV = "BIOSIM_CC3D_RUNSCRIPT"
RUNSCRIPT_NAMES = ("runScript.sh", "runScript.bat", "runScript")

#: Human wording for each backend, so availability can say WHICH one will run rather
#: than only that something will.
BACKEND_LABELS: Dict[str, str] = {
    "python-package": "the 'cc3d' Python package installed in this interpreter",
    "runscript-env": f"a local CompuCell3D runScript named by {RUNSCRIPT_ENV}",
    "runscript-path": "a local CompuCell3D runScript found on PATH",
    "aws-batch": "remote execution as an on-demand AWS Batch job",
}

#: Estimated-wall-clock ceiling for a REMOTE run, enforced in :meth:`validate`. Owned
#: by ``cc3d_remote`` (the reason for it is an AWS Batch spot behaviour) and bound here
#: so the adapter has one name for it.
REMOTE_RUN_LENGTH_CAP_MINUTES = cc3d_remote.REMOTE_RUN_LENGTH_CAP_MINUTES

#: Job sizing for the remote backend, matched to an m7i.xlarge job definition.
REMOTE_DEFAULT_VCPUS = cc3d_remote.DEFAULT_VCPUS
REMOTE_DEFAULT_MEMORY_MIB = cc3d_remote.DEFAULT_MEMORY_MIB

#: Wall-clock ceiling for one external CompuCell3D process. An external simulator that
#: hangs must not hold a worker thread forever; 0 disables the bound.
DEFAULT_TIMEOUT_SECS = 3600.0

DEFAULTS: Dict[str, Any] = {
    "lattice": {"x": 60, "y": 60, "z": 1},
    "steps": 500,
    "temperature": 10.0,
    "neighbor_order": 2,
    "cell_types": [],
    "contact_energies": [],
    "volume_constraint": {"target_volume": 25.0, "lambda_volume": 2.0},
    "fields": [],
    "seed": 0,
}


def _merged(config: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(DEFAULTS)
    merged["lattice"] = dict(DEFAULTS["lattice"])
    merged["volume_constraint"] = dict(DEFAULTS["volume_constraint"])
    incoming = config or {}
    explicit_target = (
        isinstance(incoming.get("volume_constraint"), dict)
        and incoming["volume_constraint"].get("target_volume") is not None
    )
    for key, value in incoming.items():
        if key in ("lattice", "volume_constraint") and isinstance(value, dict):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value

    # The DEFAULT target volume must match the DIMENSIONALITY of the lattice.
    #
    # DEFAULTS carries 25.0, which is the 2D area of the default 5-wide seed cell
    # (5**2). On a lattice with z > 1 the BlobInitializer seeds cells of 5**3 = 125
    # sites, so every cell was born 5x over target and the Volume constraint crushed
    # the population from MCS 0 -- a wrong simulation that still ran and reported
    # success. Derived HERE rather than in the XML builder because validate(), the
    # remote submit path and the UI all read this merged config, and they must agree.
    # An explicitly supplied target is always respected.
    if not explicit_target:
        # Defensive conversion: _merged runs inside validate(), and validate must
        # REPORT malformed input, never crash on it -- a bare int() here raised
        # ValueError straight out of validate for a lattice z of "deep". A value we
        # cannot read falls back to the 2D default and is left for validate to flag.
        def _as_int(value, fallback):
            try:
                return int(value)
            except (TypeError, ValueError):
                return fallback

        width = _as_int(merged.get("cell_width", 5), 5)
        z_extent = _as_int((merged.get("lattice") or {}).get("z", 1), 1)
        dim = 3 if z_extent > 1 else 2
        merged["volume_constraint"]["target_volume"] = float(width ** dim)
    return merged


def detect_cc3d() -> Dict[str, Any]:
    """Locate a way to run CompuCell3D. Never raises.

    Checks LOCAL installs first, then REMOTE execution on AWS. Remote counts as
    available because the engine genuinely runs -- just not on this host. That is
    what lets the Render deployment offer CompuCell3D at all: it cannot host the
    engine (Conda-only, ~2-3 GB, 512 MB instance) but it can dispatch to a Batch job.

    Returns a dict with ``available``, ``method``, ``version``, ``path`` and, when
    unavailable, ``reason`` naming precisely what is missing.
    """
    # 1. Python package.
    try:
        import cc3d  # type: ignore
        version = str(getattr(cc3d, "__version__", "") or "unknown")
        return {"available": True, "method": "python-package", "version": version,
                "path": getattr(cc3d, "__file__", ""), "reason": ""}
    except Exception:
        pass

    # 2. Explicit run script from the environment.
    configured = os.environ.get(RUNSCRIPT_ENV, "").strip()
    if configured:
        if os.path.isfile(configured) and os.access(configured, os.X_OK):
            return {"available": True, "method": "runscript-env", "version": "unknown",
                    "path": configured, "reason": ""}
        return {
            "available": False, "method": "", "version": "", "path": configured,
            "reason": (f"{RUNSCRIPT_ENV} is set to {configured!r}, but that is not an "
                       f"executable file."),
        }

    # 3. Run script on PATH.
    for name in RUNSCRIPT_NAMES:
        found = shutil.which(name)
        if found:
            return {"available": True, "method": "runscript-path", "version": "unknown",
                    "path": found, "reason": ""}

    # 4. Remote execution on AWS Batch. Checked last so a local engine always wins:
    #    running in-process is faster and free, and only the absence of one justifies
    #    paying for a job.
    remote = cc3d_remote.configuration_status()
    if remote.get("available"):
        return {
            "available": True, "method": "aws-batch", "version": "remote",
            "path": f"s3://{remote['bucket']}/{remote['prefix']}",
            "reason": "", "remote": remote,
        }

    return {
        "available": False, "method": "", "version": "", "path": "",
        "remote": remote,
        "reason": ("CompuCell3D is not available: none of the four ways to run it are "
                   "usable on this host. No local install was found (the 'cc3d' Python "
                   "package is not importable, " + RUNSCRIPT_ENV + " is not set to an "
                   "executable runScript, and no runScript is on PATH), and remote "
                   "execution is not configured on AWS Batch. " +
                   remote.get("reason", "") +
                   " Configuration and project export remain available without it."),
    }


def describe_backend(detected: Optional[Dict[str, Any]] = None) -> str:
    """Name the backend that WILL run the simulation, in words a user can act on.

    Availability alone is not enough information: 'CompuCell3D is available' reads the
    same whether the engine is about to run in this process or a billable AWS job is
    about to be submitted, and those are not the same decision.
    """
    detected = detected if detected is not None else detect_cc3d()
    method = str(detected.get("method") or "")
    if not detected.get("available") or not method:
        return "none (CompuCell3D cannot run on this host)"
    return BACKEND_LABELS.get(method, method)


class CompuCell3DAdapter(ApproachAdapter):
    approach_id = "cc3d"
    label = "CompuCell3D"

    # -- capabilities -------------------------------------------------------
    def get_capabilities(self) -> Capabilities:
        detected = detect_cc3d()
        remote = detected.get("method") == "aws-batch"
        backend = describe_backend(detected)

        # What this approach consumes from the earlier stages, and what it does not.
        # The workflow's premise is "prepare a domain, topology, mesh and model ONCE,
        # then choose one approach", which invites the reader to assume every approach
        # runs the model they prepared. This one does not, and the gap was previously
        # unstated: the CC3DML generator carries a field's name, diffusion constant and
        # decay only. Verified by search -- 'reaction' and 'advection' appear NOWHERE in
        # this module's code, and the only three matches for 'boundary' are comments, so
        # the stage-5 boundary conditions are not consumed either. A researcher who
        # wrote a logistic reaction R = r*u*(1 - u/K) in stage 4 gets pure diffusion
        # with linear decay, and nothing tells them.
        stage_use = (" USES: the cell types, lattice and field diffusion/decay "
                     "constants configured here. IGNORES: a stage-4 field's reaction "
                     "and advection expressions and the stage-5 boundary conditions -- "
                     "the generated CC3DML carries each field's diffusion constant and "
                     "decay rate only, so a reaction term you wrote in stage 4 is NOT "
                     "simulated here. Export the runnable project and add it as a "
                     "steppable if you need it.")

        notes = ("Backend: " + backend + ". Four backends are supported, tried in this "
                 "order: the 'cc3d' Python package, " + RUNSCRIPT_ENV + ", a runScript "
                 "on PATH, then remote execution on AWS Batch. Configuration and "
                 "runnable-project export are available even when CompuCell3D is not "
                 "installed. This approach never falls back to the in-tree ABM engine "
                 "or to MPC." + stage_use)
        if remote:
            notes = ("Running on AWS Batch on demand: the engine is not installed on "
                     "this host, so a job is dispatched and billed only while it runs. "
                     f"A run whose estimated wall clock exceeds "
                     f"{REMOTE_RUN_LENGTH_CAP_MINUTES:g} minutes is refused during "
                     "validation, because a spot interruption restarts the job from step "
                     "0 rather than resuming it. Pausing a remote run suspends result "
                     "polling only -- the AWS job keeps running and keeps billing, so "
                     "cancel is what stops the spend. " + notes)

        requirements = ["CompuCell3D (external)"]
        if remote:
            remote_config = detected.get("remote") or {}
            requirements = [
                f"AWS Batch job queue ({remote_config.get('job_queue') or 'unnamed'})",
                f"S3 bucket ({remote_config.get('bucket') or 'unnamed'})",
                "boto3",
            ]

        return Capabilities(
            approach_id=self.approach_id,
            label=self.label,
            available=bool(detected["available"]),
            unavailable_reason=detected.get("reason", ""),
            # 2D only, deliberately. CompuCell3D itself handles a z > 1 lattice, and
            # this adapter generates correct 3D CC3DML -- but the PRODUCT has no 3D
            # domain to attach it to: geometry.DOMAIN_KINDS is interval/rectangle/
            # circle/polygon, domain_dimension() cannot return 3, there is no
            # volumetric mesher and no tet/hex element kind, and mesh nodes are [x, y]
            # pairs. Advertising 3 put a dimension in the UI that no stage after the
            # lattice can service. Restore the 3 when a box domain and a volumetric
            # mesh exist, not before.
            dimensions=(2,),
            # A local subprocess cannot be paused; a Batch job's POLLING can be, which
            # is a real pause of the run record even though AWS keeps computing.
            supports_pause=bool(remote),
            supports_cancel=True,       # a subprocess and a Batch job can both be killed
            supports_fields=True,
            supports_seed=True,
            deterministic_with_seed=False,   # CC3D seeding is version dependent
            supports_export=True,            # export works even when unavailable
            engine_name="CompuCell3D" + (" (AWS Batch)" if remote else ""),
            engine_version=str(detected.get("version") or ""),
            requirements=requirements,
            notes=notes,
        )

    # -- validation (works with or without CC3D present) --------------------
    def validate(self, project: Dict[str, Any]) -> List[ValidationIssue]:
        config = _merged(self.approach_config(project))
        issues: List[ValidationIssue] = []
        path = "approaches.cc3d"

        domain = (project or {}).get("domain")
        if domain and domain_dimension(domain) != 2:
            issues.append(ValidationIssue(
                "warning", "cc3d_domain_1d",
                "CompuCell3D runs on a 2D or 3D lattice; a 1D domain will be mapped to a "
                "single-row lattice.", "domain.kind"))

        lattice = config.get("lattice") or {}

        # A z > 1 lattice on a 2D domain used to validate completely clean: a
        # 40x40x40 request against a flat rectangle produced ZERO issues, so a
        # researcher asking for 64,000 lattice sites over a two-dimensional domain got
        # no indication that the extra dimension has no geometry, no mesh and no
        # boundary tags behind it. The run costs 40x the compute of the 2D case.
        try:
            z_extent = int(lattice.get("z", 1))
        except (TypeError, ValueError):
            z_extent = 1
        if domain and z_extent > 1 and domain_dimension(domain) == 2:
            issues.append(ValidationIssue(
                "warning", "cc3d_lattice_3d_on_2d_domain",
                f"The lattice is {z_extent} sites deep in z, but the domain is "
                f"two-dimensional. CompuCell3D will run the 3D lattice, and it costs "
                f"about {z_extent}x the compute of a single layer, but no domain, mesh "
                f"or boundary condition exists in z to interpret it.",
                f"{path}.lattice.z"))

        for axis in ("x", "y", "z"):
            try:
                value = int(lattice.get(axis, 1))
            except (TypeError, ValueError):
                issues.append(ValidationIssue("error", f"cc3d_lattice_{axis}_invalid",
                                              f"Lattice {axis} must be a whole number.",
                                              f"{path}.lattice.{axis}"))
                continue
            if value < 1:
                issues.append(ValidationIssue(
                    "error", f"cc3d_lattice_{axis}_too_small",
                    f"Lattice {axis} must be at least 1 (got {value}).",
                    f"{path}.lattice.{axis}"))

        try:
            if int(config.get("steps")) < 1:
                issues.append(ValidationIssue("error", "cc3d_steps_too_small",
                                              "Step count must be at least 1.", f"{path}.steps"))
        except (TypeError, ValueError):
            issues.append(ValidationIssue("error", "cc3d_steps_invalid",
                                          "Step count must be a whole number.", f"{path}.steps"))

        try:
            if float(config.get("temperature")) <= 0:
                issues.append(ValidationIssue("error", "cc3d_temperature_not_positive",
                                              "Temperature must be greater than zero.",
                                              f"{path}.temperature"))
        except (TypeError, ValueError):
            issues.append(ValidationIssue("error", "cc3d_temperature_invalid",
                                          "Temperature must be a number.", f"{path}.temperature"))

        names = [str(t.get("name") or "").strip() for t in (config.get("cell_types") or [])
                 if isinstance(t, dict)]
        if not names:
            issues.append(ValidationIssue("error", "cc3d_no_cell_types",
                                          "Define at least one cell type.", f"{path}.cell_types"))
        if any(not n for n in names):
            issues.append(ValidationIssue("error", "cc3d_cell_type_name_missing",
                                          "Every CompuCell3D cell type needs a name.",
                                          f"{path}.cell_types"))
        if len(set(names)) != len(names):
            issues.append(ValidationIssue("error", "cc3d_cell_type_name_duplicate",
                                          "CompuCell3D cell type names must be unique.",
                                          f"{path}.cell_types"))
        if any(n.lower() == "medium" for n in names):
            issues.append(ValidationIssue(
                "error", "cc3d_medium_reserved",
                "'Medium' is CompuCell3D's reserved type for empty space; choose another name.",
                f"{path}.cell_types"))

        known = set(names) | {"Medium"}
        for index, entry in enumerate(config.get("contact_energies") or []):
            item_path = f"{path}.contact_energies[{index}]"
            if not isinstance(entry, dict):
                issues.append(ValidationIssue("error", "cc3d_contact_invalid",
                                              "Each contact energy must be an object.", item_path))
                continue
            for side in ("type1", "type2"):
                if str(entry.get(side) or "") not in known:
                    issues.append(ValidationIssue(
                        "error", "cc3d_contact_unknown_type",
                        f"Contact energy references unknown cell type "
                        f"{entry.get(side)!r}. Known: {', '.join(sorted(known))}.",
                        f"{item_path}.{side}"))
            try:
                float(entry.get("energy"))
            except (TypeError, ValueError):
                issues.append(ValidationIssue("error", "cc3d_contact_energy_invalid",
                                              "Contact energy must be a number.",
                                              f"{item_path}.energy"))

        issues.extend(self._remote_run_length_issues(config, path))
        return issues

    # -- remote run-length cap ---------------------------------------------
    def _remote_run_length_issues(self, config: Dict[str, Any],
                                  path: str = "approaches.cc3d",
                                  detected: Optional[Dict[str, Any]] = None
                                  ) -> List[ValidationIssue]:
        """Refuse a configuration too long to survive a spot interruption.

        Only applies when the REMOTE backend is the one that would run. A local
        subprocess is bounded by ``timeout_secs`` and resumes nothing either, but it is
        not restarted from scratch by someone else's capacity decision, so the same
        step count that is reckless on Batch is merely slow locally -- and refusing it
        there would block work the researcher's own machine can do.
        """
        detected = detected if detected is not None else detect_cc3d()
        if detected.get("method") != "aws-batch":
            return []

        check = cc3d_remote.run_length_check(
            config.get("steps"), config.get("lattice"),
            field_count=len(config.get("fields") or []))
        if check["within_cap"]:
            return []
        return [ValidationIssue("error", "cc3d_remote_run_too_long",
                                check["message"], f"{path}.steps")]

    # -- compilation: generate the CC3D project -----------------------------
    def compile(self, project: Dict[str, Any]) -> Dict[str, Any]:
        errors = [i for i in self.validate(project) if i.severity == "error"]
        if errors:
            raise ValueError("CompuCell3D configuration is not valid: " +
                             "; ".join(i.message for i in errors))
        config = _merged(self.approach_config(project))
        detected = detect_cc3d()
        return {
            "approach": self.approach_id,
            "config": config,
            "cc3dml": self.build_cc3dml(config),
            "steppable": self.build_steppable(config),
            "steps": int(config["steps"]),
            "available": bool(detected["available"]),
            "detection": detected,
            "engine_version": detected.get("version", ""),
        }

    @staticmethod
    def build_cc3dml(config: Dict[str, Any]) -> str:
        """Render the CC3DML XML for this configuration.

        Generated with ElementTree rather than string concatenation so a cell type
        name containing XML-significant characters cannot produce a corrupt project.
        """
        lattice = config["lattice"]
        # A project whose only cell type is Medium initialises NOTHING: it loads,
        # runs to completion and reports zero cells. That is the same silent-empty
        # failure the ABM path had, so it is refused here rather than exported. It
        # happens when a project reaches the exporter with no cc3d block at all --
        # e.g. imported JSON, or a project built server-side where `approaches` is {}.
        declared_types = [
            spec for spec in (config.get("cell_types") or [])
            if isinstance(spec, dict) and str(spec.get("name") or "").strip()
        ]
        if not declared_types:
            raise ValueError(
                "This CompuCell3D configuration defines no cell types, so the exported "
                "project would run and produce zero cells. Add at least one cell type "
                "(a name and a target volume) before exporting."
            )
        root = ET.Element("CompuCell3D", {"Revision": "0", "Version": "4"})

        potts = ET.SubElement(root, "Potts")
        # One <Dimensions x= y= z=/> element, which is the form the CompuCell3D
        # reference manual documents ("the line reading ... declares the dimensions
        # of the lattice to be 101 by 101 by 1 pixels" -- a single line carrying all
        # three values). The previous three separate <DimensionX>/<DimensionY>/
        # <DimensionZ> elements are at best an undocumented alias, so a project using
        # them risks loading with a default lattice instead of the configured one.
        ET.SubElement(potts, "Dimensions", {
            "x": str(int(lattice.get("x", 1))),
            "y": str(int(lattice.get("y", 1))),
            "z": str(int(lattice.get("z", 1))),
        })
        ET.SubElement(potts, "Steps").text = str(int(config["steps"]))
        ET.SubElement(potts, "Temperature").text = str(float(config["temperature"]))
        ET.SubElement(potts, "NeighborOrder").text = str(int(config.get("neighbor_order", 2)))
        if config.get("seed") is not None:
            ET.SubElement(potts, "RandomSeed").text = str(int(config["seed"]))

        cell_type_plugin = ET.SubElement(root, "Plugin", {"Name": "CellType"})
        ET.SubElement(cell_type_plugin, "CellType", {"TypeId": "0", "TypeName": "Medium"})
        for index, spec in enumerate(config.get("cell_types") or [], start=1):
            ET.SubElement(cell_type_plugin, "CellType", {
                "TypeId": str(int(spec.get("type_id", index))),
                "TypeName": str(spec.get("name")),
            })

        # VOLUME. Two defects fixed here.
        #
        # (1) The default target_volume of 25.0 is a 2D AREA: it is exactly
        # cell_width**2 for the default width of 5. On a lattice with z > 1 the
        # BlobInitializer seeds cells of width**3 = 125 sites, so every cell is born
        # 5x over target and the Volume constraint crushes the population from MCS 0 --
        # a completely wrong simulation that still runs and reports success. The
        # default is now derived from the lattice's actual dimensionality.
        #
        # (2) Per-cell-type target volumes were silently discarded: cell_types entries
        # carrying their own target_volume never reached the CC3DML, which emitted a
        # single global value. A researcher differentiating cell sizes by type got one
        # size for all of them, with no warning.
        volume = config.get("volume_constraint") or {}
        cell_width = int(config.get("cell_width", 5))
        lattice_dim = 3 if int(lattice.get("z", 1)) > 1 else 2
        derived_default = float(cell_width ** lattice_dim)
        global_target = float(volume.get("target_volume", derived_default))
        global_lambda = float(volume.get("lambda_volume", 2.0))

        volume_plugin = ET.SubElement(root, "Plugin", {"Name": "Volume"})
        per_type = [spec for spec in declared_types
                    if spec.get("target_volume") is not None]
        if per_type:
            # CC3DML's documented per-type form. Types without their own value fall
            # back to the global/derived one rather than being left unconstrained.
            for spec in declared_types:
                name = str(spec.get("name")).strip()
                target = spec.get("target_volume")
                lam = spec.get("lambda_volume")
                ET.SubElement(volume_plugin, "VolumeEnergyParameters", {
                    "CellType": name,
                    "TargetVolume": str(float(target if target is not None
                                              else global_target)),
                    "LambdaVolume": str(float(lam if lam is not None
                                              else global_lambda)),
                })
        else:
            ET.SubElement(volume_plugin, "TargetVolume").text = str(global_target)
            ET.SubElement(volume_plugin, "LambdaVolume").text = str(global_lambda)

        contacts = config.get("contact_energies") or []
        if contacts:
            contact_plugin = ET.SubElement(root, "Plugin", {"Name": "Contact"})
            for entry in contacts:
                element = ET.SubElement(contact_plugin, "Energy", {
                    "Type1": str(entry.get("type1")), "Type2": str(entry.get("type2")),
                })
                element.text = str(float(entry.get("energy", 0.0)))
            ET.SubElement(contact_plugin, "NeighborOrder").text = str(
                int(config.get("neighbor_order", 2)))

        for field_spec in config.get("fields") or []:
            if not isinstance(field_spec, dict) or not field_spec.get("name"):
                continue
            steppable = ET.SubElement(root, "Steppable", {"Type": "DiffusionSolverFE"})
            diffusion = ET.SubElement(steppable, "DiffusionField",
                                      {"Name": str(field_spec["name"])})
            data = ET.SubElement(diffusion, "DiffusionData")
            ET.SubElement(data, "FieldName").text = str(field_spec["name"])
            ET.SubElement(data, "GlobalDiffusionConstant").text = str(
                float(field_spec.get("diffusion", 0.1)))
            ET.SubElement(data, "GlobalDecayConstant").text = str(
                float(field_spec.get("decay", 0.0)))

        blob = ET.SubElement(root, "Steppable", {"Type": "BlobInitializer"})
        region = ET.SubElement(blob, "Region")
        ET.SubElement(region, "Center", {
            "x": str(int(lattice.get("x", 60)) // 2),
            "y": str(int(lattice.get("y", 60)) // 2),
            "z": str(max(0, int(lattice.get("z", 1)) // 2)),
        })
        # The default radius must fit inside the THINNEST axis, not just x. Using
        # max(4, x // 6) meant a 120x120x10 slab got radius 20 against a half-thickness
        # of 5, so the initial blob was clipped by the z boundary and the seeded
        # population was quietly smaller than configured -- with no warning, because
        # a clipped blob still initialises successfully.
        axes = [int(lattice.get("x", 60) or 60), int(lattice.get("y", 60) or 60)]
        z_extent = int(lattice.get("z", 1) or 1)
        if z_extent > 1:
            axes.append(z_extent)
        safe_radius = max(2, min(axes) // 2 - 1)
        default_radius = min(max(4, int(lattice.get("x", 60)) // 6), safe_radius)
        ET.SubElement(region, "Radius").text = str(
            int(config.get("blob_radius", default_radius)))
        ET.SubElement(region, "Gap").text = "0"
        ET.SubElement(region, "Width").text = str(int(config.get("cell_width", 5)))
        types_element = ET.SubElement(region, "Types")
        types_element.text = ",".join(str(spec.get("name"))
                                      for spec in (config.get("cell_types") or []))

        ET.indent(root, space="   ")
        return ET.tostring(root, encoding="unicode")

    @staticmethod
    def build_steppable(config: Dict[str, Any]) -> str:
        """A CC3D Python steppable that writes per-step cell measurements as CSV.

        The adapter parses that CSV back, which is how CC3D results reach the same
        viewer the other approaches use.
        """
        every = max(1, int(config.get("report_every", 10)))
        return f'''"""Generated by BioSimulateAI. Writes per-cell measurements for import."""

from cc3d.core.PySteppables import SteppableBasePy


class MeasurementSteppable(SteppableBasePy):
    def __init__(self, frequency={every}):
        SteppableBasePy.__init__(self, frequency=frequency)
        self.out = None

    def start(self):
        self.out = open("cells.csv", "w")
        self.out.write("mcs,cell_id,type,volume,surface,x,y,z\\n")

    def step(self, mcs):
        for cell in self.cell_list:
            self.out.write(
                f"{{mcs}},{{cell.id}},{{cell.type}},{{cell.volume}},"
                f"{{cell.surface}},{{cell.xCOM}},{{cell.yCOM}},{{cell.zCOM}}\\n"
            )
        self.out.flush()

    def finish(self):
        if self.out:
            self.out.close()
'''

    # -- execution ----------------------------------------------------------
    def run(self, compiled: Dict[str, Any], context: RunContext) -> Dict[str, Any]:
        detected = detect_cc3d()
        if not detected["available"]:
            # Refuse honestly: no fabricated results, and no substituting the ABM or MPC
            # approach for the one that was selected. The reason names every backend that
            # was checked, so the researcher knows what to install or configure.
            raise ApproachUnavailable(detected["reason"])

        if detected["method"] == "aws-batch":
            return self._run_remote(compiled, context, detected)

        import tempfile

        config = compiled["config"]
        workdir = tempfile.mkdtemp(prefix="cc3d_run_")
        simulation_dir = os.path.join(workdir, "Simulation")
        os.makedirs(simulation_dir, exist_ok=True)
        cc3dml_path = os.path.join(simulation_dir, "model.xml")
        steppable_path = os.path.join(simulation_dir, "steppables.py")
        with open(cc3dml_path, "w", encoding="utf-8") as handle:
            handle.write(compiled["cc3dml"])
        with open(steppable_path, "w", encoding="utf-8") as handle:
            handle.write(compiled["steppable"])

        context.log(f"CompuCell3D project written to {simulation_dir}")
        context.log(f"Using CompuCell3D via {describe_backend(detected)} "
                    f"(version {detected.get('version') or 'unknown'}).")

        if detected["method"] == "python-package":
            raise ApproachUnavailable(
                "A CompuCell3D Python package was found, but in-process execution is not "
                "implemented: CC3D drives its own event loop and must be launched through "
                "its runScript. Set " + RUNSCRIPT_ENV + " to the runScript path to run it, "
                "or use the exported project directly."
            )

        command = [detected["path"], "-i", cc3dml_path, "--noOutput"]
        context.log("Launching: " + " ".join(command))
        timeout_secs = float(config.get("timeout_secs", DEFAULT_TIMEOUT_SECS))
        logs: List[str] = []
        try:
            process = subprocess.Popen(
                command, cwd=workdir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )
        except OSError as exc:
            raise ApproachUnavailable(
                f"CompuCell3D could not be launched from {detected['path']!r}: {exc}") from exc

        # Output is drained on a separate thread, and the control loop polls.
        #
        # Reading stdout inline instead would tie cancellation to the arrival of a
        # line: a process that hangs silently could never be cancelled, and nothing
        # would bound the wait. Polling makes both cancellation and the timeout work
        # whether or not the simulator says anything.
        import threading

        def drain() -> None:
            try:
                assert process.stdout is not None
                for line in process.stdout:
                    text = line.rstrip()
                    logs.append(text)
                    context.log(text)
            except Exception:
                pass

        reader = threading.Thread(target=drain, name="cc3d-stdout", daemon=True)
        reader.start()

        started = time.monotonic()
        cancelled = False
        timed_out = False
        try:
            while True:
                try:
                    process.wait(timeout=0.25)
                    break
                except subprocess.TimeoutExpired:
                    pass
                if context.is_cancelled():
                    cancelled = True
                    break
                if timeout_secs > 0 and (time.monotonic() - started) > timeout_secs:
                    timed_out = True
                    break
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
            reader.join(timeout=5)
            # Close the stdout pipe explicitly. Popen(stdout=PIPE) hands back a
            # TextIOWrapper that nothing else owns: the drain thread only iterates
            # it. Leaving it to the garbage collector leaked a descriptor per run
            # and surfaced as "ResourceWarning: unclosed file" on every test pass.
            # Done after reader.join so the drain thread cannot read a closed pipe.
            if process.stdout is not None:
                try:
                    process.stdout.close()
                except Exception:
                    pass

        if cancelled:
            context.log("CompuCell3D process terminated on cancellation.", "warning")
            raise RunCancelled(f"Run {context.run_id} was cancelled.")
        if timed_out:
            raise RuntimeError(
                f"CompuCell3D did not finish within {timeout_secs:g} s and was stopped. "
                f"Raise 'timeout_secs' for a longer simulation, or reduce the step count. "
                f"Last output: " + (" | ".join(logs[-3:]) or "none"))

        code = process.returncode
        if code != 0:
            raise RuntimeError(
                f"CompuCell3D exited with status {code}. Last output: "
                + (" | ".join(logs[-5:]) or "none"))

        results = self.parse_output(os.path.join(workdir, "cells.csv"))
        results.update({
            "approach": self.approach_id,
            "kind": "agents",
            "engine": "CompuCell3D",
            "engine_version": detected.get("version", ""),
            "workdir": workdir,
            "logs": logs[-200:],
        })
        return results

    @staticmethod
    def parse_output(csv_path: str) -> Dict[str, Any]:
        """Parse the steppable's CSV into the shared per-frame cell-table shape."""
        import csv as csv_module

        if not os.path.isfile(csv_path):
            raise RuntimeError(
                f"CompuCell3D produced no measurement file at {csv_path}. The run "
                f"finished but wrote no cell data, so there is nothing to display.")
        frames: Dict[int, List[Dict[str, Any]]] = {}
        with open(csv_path, "r", encoding="utf-8") as handle:
            for row in csv_module.DictReader(handle):
                try:
                    mcs = int(row["mcs"])
                    frames.setdefault(mcs, []).append({
                        "id": int(row["cell_id"]),
                        "type": str(row["type"]),
                        "state": "alive",
                        "alive": True,
                        "volume": float(row["volume"]),
                        "surface": float(row["surface"]),
                        "x": float(row["x"]),
                        "y": float(row["y"]),
                        "z": float(row.get("z") or 0.0),
                        "age": None,
                        "parent_id": None,
                    })
                except (KeyError, TypeError, ValueError):
                    continue
        times = sorted(frames)
        return {
            "t": times,
            "cells": [frames[t] for t in times],
            "cell_counts": [{"total": len(frames[t])} for t in times],
            "summary": {"frames": len(times),
                        "final_cell_count": len(frames[times[-1]]) if times else 0},
        }

    # -- remote execution on AWS Batch --------------------------------------
    def _run_remote(self, compiled: Dict[str, Any], context: RunContext,
                    detected: Dict[str, Any]) -> Dict[str, Any]:
        """Run the simulation as an on-demand AWS Batch job.

        The engine runs on AWS; this process only submits, polls and imports. The
        results are parsed into the SAME shape as the local path, so the viewer and
        every export work identically regardless of where the simulation executed.

        Cancellation terminates the Batch job, so a cancelled run stops billing.
        Pausing suspends polling only -- see :meth:`cc3d_remote.BatchRunner.wait`.
        """
        config = compiled["config"]
        remote_config = detected.get("remote") or cc3d_remote.configuration_status()
        runner = cc3d_remote.BatchRunner(remote_config)

        # Re-checked here as well as in validate(): compile() may have happened before
        # the step count was raised, and a job that cannot finish must not be paid for.
        check = cc3d_remote.run_length_check(
            config.get("steps"), config.get("lattice"),
            field_count=len(config.get("fields") or []))
        if not check["within_cap"]:
            raise ValueError(check["message"])

        files = {
            "Simulation/model.xml": compiled["cc3dml"],
            "Simulation/steppables.py": compiled["steppable"],
            # The container builds its specs from THIS, not by re-parsing the CC3DML.
            # PyCoreSpecs.from_file rejected our valid CC3DML with "No Potts
            # specification", and round-tripping a config we already hold through XML
            # and back adds a parser we do not control to the critical path for no
            # benefit. The CC3DML is still exported -- it is what a researcher opens
            # in a local CompuCell3D install, which is its actual purpose.
            "cc3d_config.json": json.dumps(config, indent=2, sort_keys=True, default=str),
        }
        vcpus = int(config.get("remote_vcpus", REMOTE_DEFAULT_VCPUS))
        memory = int(config.get("remote_memory_mib", REMOTE_DEFAULT_MEMORY_MIB))

        context.log(f"Dispatching CompuCell3D to AWS Batch "
                    f"(queue {remote_config['job_queue']}, {vcpus} vCPU, {memory} MiB).")
        context.log(f"Estimated wall clock {check['estimated_minutes']:g} min, within the "
                    f"{check['cap_minutes']:g} min cap (at most {check['max_steps']} steps "
                    f"on this lattice).")
        submission = runner.submit(context.run_id, files, vcpus=vcpus,
                                   memory_mib=memory, steps=config.get("steps"))
        context.log(f"Job {submission['job_id']} submitted; inputs at "
                    f"s3://{submission['bucket']}/{submission['prefix']}")

        # Billed for what the job is expected to take, not a placeholder: the run-length
        # estimate is the same one the cap was applied to. 'expected_minutes' overrides it
        # when the researcher knows better from a previous run of the same model.
        minutes = float(config.get("expected_minutes") or check["estimated_minutes"])
        estimate = cc3d_remote.estimate_cost(vcpus, memory, minutes)
        context.log(f"Estimated compute cost for a {estimate['minutes']:g} minute run: "
                    f"about ${estimate['estimated_usd']:.4f} ({estimate['basis']}).")

        # AWS Batch reports coarse states rather than a percentage, so progress is
        # mapped from the state machine instead of invented.
        stage_progress = {"SUBMITTED": 0.05, "PENDING": 0.1, "RUNNABLE": 0.15,
                          "STARTING": 0.25, "RUNNING": 0.5}

        def on_progress(info: Dict[str, Any]) -> None:
            context.progress(stage_progress.get(info["status"], 0.5), info["status"])
            context.log(f"Batch state: {info['status']}"
                        + (f" — {info['reason']}" if info.get("reason") else ""))

        # Default the wait to the cap rather than the module's 2 hour ceiling: a job
        # validated as fitting inside the cap and still running well past it is stuck.
        timeout_secs = config.get("timeout_secs")
        if timeout_secs is None:
            timeout_secs = REMOTE_RUN_LENGTH_CAP_MINUTES * 60.0

        try:
            final = runner.wait(
                submission["job_id"],
                timeout_secs=timeout_secs,
                on_progress=on_progress,
                on_log=lambda line: context.log(line),
                is_cancelled=context.is_cancelled,
                wait_if_paused=context.wait_if_paused,
            )
        except TimeoutError as exc:
            raise RuntimeError(str(exc)) from None

        if final["status"] == "CANCELLED":
            raise RunCancelled(f"Run {context.run_id} was cancelled; the AWS job was "
                               f"terminated.")
        if final["status"] != cc3d_remote.BATCH_SUCCESS:
            status_detail = runner.fetch_text(context.run_id, "status.json") or ""
            raise RuntimeError(
                f"The CompuCell3D job failed on AWS (state {final['status']}, exit "
                f"{final.get('exit_code')}). {final.get('reason') or ''} "
                f"{status_detail[:400]}".strip())

        context.progress(0.85, "importing results")
        fetched = runner.fetch_results(context.run_id)

        # Reuse the local parser by writing the CSV where it expects to find it, so
        # there is exactly one implementation of the output contract.
        import tempfile
        with tempfile.TemporaryDirectory(prefix="cc3d_remote_") as directory:
            local_csv = os.path.join(directory, "cells.csv")
            with open(local_csv, "w", encoding="utf-8", newline="") as handle:
                handle.write(fetched["cells_csv"])
            results = self.parse_output(local_csv)

        results.update({
            "approach": self.approach_id,
            "kind": "agents",
            "engine": "CompuCell3D",
            "engine_version": str((fetched.get("status") or {}).get("version")
                                  or "remote"),
            "execution": {
                "location": "aws-batch",
                "backend": describe_backend(detected),
                "job_id": submission["job_id"],
                "region": remote_config["region"],
                "job_queue": remote_config["job_queue"],
                "s3_prefix": fetched["s3_prefix"],
                "seconds": (fetched.get("status") or {}).get("seconds"),
                "vcpus": vcpus,
                "memory_mib": memory,
                "run_length": check,
                "cost_estimate": estimate,
            },
        })
        context.progress(1.0, "completed")
        return results

    # -- export works even when CC3D is absent ------------------------------
    def export_configuration(self, project: Dict[str, Any]) -> Dict[str, Any]:
        config = _merged(self.approach_config(project))
        detected = detect_cc3d()
        return {
            "approach": self.approach_id,
            "label": self.label,
            "available": bool(detected["available"]),
            "unavailable_reason": detected.get("reason", ""),
            "configuration": config,
            "files": {
                "Simulation/model.xml": self.build_cc3dml(config),
                "Simulation/steppables.py": self.build_steppable(config),
                "README.txt": (
                    "CompuCell3D project exported by BioSimulateAI.\n\n"
                    "Run with:\n"
                    "    runScript.sh -i Simulation/model.xml\n\n"
                    "steppables.py writes cells.csv, which BioSimulateAI can import.\n"
                ),
            },
        }


CC3D_ADAPTER = register_approach(CompuCell3DAdapter())
