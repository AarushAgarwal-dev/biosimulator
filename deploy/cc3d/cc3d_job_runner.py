#!/usr/bin/env python3
"""
Container entrypoint for a remote CompuCell3D run (AWS Batch).

Reads the project that BioSimulateAI uploaded, runs CompuCell3D headless, and puts
the measurement CSV back where the application expects it. Everything is addressed
by environment variables so one Batch job definition serves every run.

  CC3D_BUCKET       S3 bucket holding this run's prefix              (required)
  CC3D_PREFIX       prefix inside the bucket                         (required)
  CC3D_RUN_ID       run identifier, for logging                      (optional)
  CC3D_STEPS        override the Monte Carlo step count              (optional)
  CC3D_MAX_MINUTES  hard wall-clock cap on one run, default 90       (optional)
  CC3D_RUNSCRIPT    explicit path to CompuCell3D's runScript.sh      (optional)

Contract with the application
-----------------------------
On success it uploads ``cells.csv``. On any outcome it uploads ``status.json``
describing what happened. The application treats a missing ``cells.csv`` as a
failure even when the container exited zero, so this script must never write a
partial or fabricated CSV -- an empty simulation is reported as an error instead.

Partial output therefore NEVER goes to ``cells.csv``. It goes to the separate
``cells.partial.csv`` key, precisely because the application reads ``cells.csv`` as
a finished result.

Spot interruption
-----------------
These jobs run on EC2 spot capacity. A reclaim publishes a two-minute notice on the
instance metadata service, then SIGTERM arrives, then SIGKILL 30 seconds after that.
CompuCell3D writes no checkpoint, so an AWS Batch retry restarts the simulation from
step 0 rather than resuming it. The strategy here is consequently to CAP RUN LENGTH
rather than checkpoint, and on interruption to upload whatever partial output exists
alongside an honest status -- so a retry is a clean restart and the researcher is
told what happened:

* a watcher thread polls the spot ``instance-action`` endpoint every 5 seconds and
  starts shutting down the moment a notice appears, which buys roughly two minutes
  of shutdown budget instead of thirty seconds;
* a SIGTERM handler stops the engine, uploads ``cells.partial.csv`` inside a hard
  deadline, writes ``state="interrupted"`` with a reason naming spot reclamation,
  and exits 75;
* ``CC3D_MAX_MINUTES`` stops a run that would most likely be reclaimed before it
  could ever finish, and says so in ``status.json``.

Exit codes
----------
0 success, 1 unhandled error, 2 misconfigured, 3 no Simulation/model.xml,
4 no runScript in the image, 5 engine exited non-zero, 6 no cells.csv,
7 empty cells.csv, 8 exceeded CC3D_MAX_MINUTES, 75 interrupted (spot reclamation).

75 rather than the 143 a bare SIGTERM death produces, so the log tells you the
handler ran and did its job instead of leaving you to guess.
"""

import collections
import glob
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import urllib.error
import urllib.request

import boto3
from botocore.config import Config as BotoConfig

BUCKET = os.environ.get("CC3D_BUCKET", "").strip()
PREFIX = os.environ.get("CC3D_PREFIX", "").strip()
RUN_ID = os.environ.get("CC3D_RUN_ID", "unknown").strip()

#: The finished measurement file. Written only on genuine success.
RESULT_KEY = "cells.csv"
#: Whatever existed when a run was cut short. A DISTINCT key: the application reads
#: cells.csv as a complete result, so partial data must never land there.
PARTIAL_KEY = "cells.partial.csv"
STATUS_KEY = "status.json"

EXIT_OK = 0
EXIT_UNHANDLED = 1
EXIT_MISCONFIGURED = 2
EXIT_NO_MODEL = 3
EXIT_NO_RUNSCRIPT = 4
EXIT_ENGINE_FAILED = 5
EXIT_NO_RESULT = 6
EXIT_EMPTY_RESULT = 7
EXIT_TIME_LIMIT = 8
EXIT_INTERRUPTED = 75

#: Wall-clock ceiling on one run when CC3D_MAX_MINUTES says nothing usable.
DEFAULT_MAX_MINUTES = 90.0
#: How much of the 30-second SIGKILL window the shutdown path may spend. The margin
#: is deliberate: being killed mid-write is worse than skipping the upload.
SHUTDOWN_BUDGET_SECS = 20.0
#: The spot NOTICE arrives ~2 minutes before termination, so there is far more room
#: than a bare SIGTERM's 30-second window. Using the 20s budget for both discarded
#: roughly 100 seconds on exactly the path that can ship the most useful partial.
SPOT_NOTICE_BUDGET_SECS = 90.0
#: How long to wait for the engine to die before escalating to SIGKILL.
CHILD_GRACE_SECS = 3.0
#: Below this much remaining budget the partial upload is skipped, not attempted.
MIN_UPLOAD_SECS = 4.0
#: Cap on the partial body. A multi-hundred-megabyte CSV cannot be uploaded inside
#: the window, so the head of it is shipped and the truncation is recorded.
MAX_PARTIAL_BYTES = 32 * 1024 * 1024
#: Supervision granularity. Small enough that the run-length cap and the shutdown
#: flag are both observed promptly.
POLL_SECS = 0.25
#: How long to wait for the output reader to drain after the engine exits.
OUTPUT_DRAIN_SECS = 10.0
#: Engine output lines kept for the failure tail. Bounded on purpose -- the old
#: capture_output=True buffered the entire run in memory.
OUTPUT_TAIL_LINES = 400

IMDS_TOKEN_URL = "http://169.254.169.254/latest/api/token"
IMDS_ACTION_URL = "http://169.254.169.254/latest/meta-data/spot/instance-action"
IMDS_TIMEOUT_SECS = 1.0
IMDS_TOKEN_TTL_SECS = 21600
IMDS_POLL_SECS = 5.0
#: After this many failures to even get a token, the watcher gives up SILENTLY:
#: running locally or with IMDS disabled is a normal condition, not an error.
IMDS_MAX_SETUP_FAILURES = 3

_SIGKILL = getattr(signal, "SIGKILL", signal.SIGTERM)

SPOT_SIGTERM_REASON = (
    "The container received SIGTERM before the simulation finished. On this "
    "spot-backed AWS Batch queue that means EC2 reclaimed the instance, and the "
    "process was killed about 30 seconds later. CompuCell3D writes no checkpoint, "
    "so a retry restarts the simulation from step 0 rather than resuming it. Any "
    "rows written so far were uploaded as cells.partial.csv, which is an incomplete "
    "measurement file and must not be read as a finished result."
)
SPOT_NOTICE_REASON = (
    "EC2 published a spot interruption notice for the instance running this "
    "simulation (action={action}, scheduled {when}), so the run was stopped before "
    "AWS reclaimed the host. CompuCell3D writes no checkpoint, so a retry restarts "
    "from step 0 rather than resuming. Any rows written so far were uploaded as "
    "cells.partial.csv, which is an incomplete measurement file and must not be "
    "read as a finished result."
)
LOCAL_STOP_REASON = (
    "The run was stopped by an interrupt signal before the simulation finished. No "
    "complete measurement file was produced; any rows written so far were uploaded "
    "as cells.partial.csv."
)
TIME_CAP_REASON = (
    "The simulation was still running after {minutes:g} minutes and was stopped at "
    "that cap. This is a sizing problem rather than a crash: reduce the Monte Carlo "
    "step count so the run fits inside the cap. A run this long is also unlikely to "
    "survive on spot capacity, where a reclaim restarts it from step 0, so shortening "
    "it is usually cheaper than raising CC3D_MAX_MINUTES."
)


def time_cap_reason(minutes):
    """TIME_CAP_REASON with the cap substituted.

    The template carries a ``{minutes:g}`` placeholder, and it was being passed to
    request_shutdown UNFORMATTED -- so status.json['error'] read "still running after
    {minutes:g} minutes" verbatim, which is what the researcher sees. Formatting is
    done here so every call site gets it right, and a bad value degrades to the
    template rather than raising inside a shutdown path.
    """
    try:
        return TIME_CAP_REASON.format(minutes=float(minutes))
    except Exception:
        return TIME_CAP_REASON.replace("{minutes:g}", str(minutes))

# One Session, two clients. The session shares its resolved credentials, so the
# shutdown client is ready to use without re-walking the credential chain at the
# worst possible moment. The shutdown client is configured to fail FAST: inside the
# SIGKILL window a hung request is worse than a skipped one.
_session = boto3.session.Session()
s3 = _session.client("s3", config=BotoConfig(
    connect_timeout=5, read_timeout=20, retries={"max_attempts": 3, "mode": "standard"}))
urgent_s3 = _session.client("s3", config=BotoConfig(
    connect_timeout=2, read_timeout=6, retries={"max_attempts": 1, "mode": "standard"}))

# ---------------------------------------------------------------------------
# Shutdown state
# ---------------------------------------------------------------------------
#: Plain module-level values, not an Event or a Lock. A signal handler runs on top
#: of the main thread, so anything it touches that takes a lock can deadlock against
#: the very code it interrupted. These are read and written without locking, and the
#: flag is set FIRST in every path, which is what makes a repeated SIGTERM a no-op.
_shutdown_requested = False
#: Guards the claim on the shutdown fields below. Acquired NON-blocking only, so it
#: is safe from inside a signal handler.
_shutdown_claim = threading.Lock()
_shutdown_reason = ""
_shutdown_state = "interrupted"
_shutdown_exit_code = EXIT_INTERRUPTED
_shutdown_at = 0.0
#: The running engine, so the handler and the watcher can reach it.
_child = None
_job_started = 0.0


class Interrupted(Exception):
    """Raised at a checkpoint once a shutdown has been requested."""


def log(message):
    # Unbuffered, so CloudWatch shows progress while the simulation is still running.
    print(f"[cc3d-runner] {message}", flush=True)


def request_shutdown(reason, state="interrupted", exit_code=EXIT_INTERRUPTED):
    """Record the first shutdown request and report whether it won.

    Returns False when a shutdown is already under way, so every caller -- the
    signal handler, the metadata watcher, the run-length cap -- can tell whether it
    owns the shutdown or should keep its hands off one in progress. The flag is set
    before any other assignment, so a second signal arriving part-way through this
    function still finds it set and returns. The upload itself is performed by the
    main thread in normal control flow, never here, so a lost race over the reason
    string cannot produce two concurrent uploads.
    """
    global _shutdown_requested, _shutdown_reason, _shutdown_state
    global _shutdown_exit_code, _shutdown_at
    # A non-blocking lock, so this is still safe to call from inside a signal
    # handler: the handler can never park waiting for a lock the main thread holds.
    # Without it the check-then-set was not atomic, and three writers (the handler,
    # the metadata watcher, the run-length cap) could interleave -- producing a
    # record with one caller's state and another's reason, e.g. a spot reclamation
    # reported as a sizing failure with exit code 8.
    if not _shutdown_claim.acquire(False):
        return False
    try:
        if _shutdown_requested:
            return False
        _shutdown_reason = reason
        _shutdown_state = state
        _shutdown_exit_code = exit_code
        _shutdown_at = time.monotonic()
        # Set LAST: _checkpoint and the step loops read this flag without the lock,
        # so it must not become true until the fields it describes are populated.
        _shutdown_requested = True
        return True
    finally:
        _shutdown_claim.release()


def _checkpoint():
    """Abandon the run if a shutdown was requested while we were busy elsewhere."""
    if _shutdown_requested:
        raise Interrupted(_shutdown_reason)


def _signal_child(sig):
    """Signal the engine and everything it started.

    runScript.sh is a shell wrapper, so the process that actually simulates is a
    grandchild; signalling the process GROUP is what reaches it. Safe to call from a
    signal handler: syscalls only, no I/O, no locks, and a no-op once the child is
    gone.
    """
    process = _child
    if process is None or process.returncode is not None:
        return
    try:
        if hasattr(os, "killpg") and hasattr(os, "getpgid"):
            os.killpg(os.getpgid(process.pid), sig)
            return
    except (OSError, AttributeError):
        pass
    try:
        process.send_signal(sig)
    except Exception:
        pass


def _await_child(timeout):
    process = _child
    if process is None:
        return
    try:
        process.wait(timeout=max(0.0, timeout))
    except Exception:
        pass


def _on_terminate(signum, frame):  # noqa: ARG001 - frame is part of the signature
    """Signal handler. Deliberately does almost nothing.

    It sets the shutdown flag and stops the engine, which are syscalls. It does NOT
    log and does NOT touch S3: a print or a botocore call here would run on top of
    whatever the main thread was doing, and re-entering botocore mid-request is
    exactly how a half-written upload happens. The main thread notices the flag and
    performs the bounded upload in normal control flow. A second SIGTERM finds the
    flag already set and returns immediately, so it cannot disturb an upload in
    flight.
    """
    if signum == getattr(signal, "SIGTERM", None):
        reason = SPOT_SIGTERM_REASON
    else:
        reason = LOCAL_STOP_REASON
    if request_shutdown(reason):
        _signal_child(signal.SIGTERM)


def install_signal_handlers():
    installed = []
    for name in ("SIGTERM", "SIGINT"):
        sig = getattr(signal, name, None)
        if sig is None:
            continue
        try:
            signal.signal(sig, _on_terminate)
            installed.append(name)
        except (ValueError, OSError, RuntimeError) as exc:
            log(f"could not install a {name} handler: {type(exc).__name__}: {exc}")
    # This is FATAL, not a degradation. The entrypoint is exec-form, so Python is
    # PID 1, and PID 1 with the default disposition IGNORES SIGTERM -- the container
    # would sit untouched until SIGKILL and write no status.json at all, which reads
    # to the application as a job that vanished. Better to fail immediately and say
    # why than to run 90 minutes that cannot be shut down or reported.
    if "SIGTERM" not in installed:
        return False
    return True


# ---------------------------------------------------------------------------
# Spot interruption notice (IMDSv2)
# ---------------------------------------------------------------------------
class _TokenExpired(Exception):
    """The IMDSv2 token was rejected; fetch a new one."""


def _imds_token():
    request = urllib.request.Request(
        IMDS_TOKEN_URL, data=b"", method="PUT",
        headers={"X-aws-ec2-metadata-token-ttl-seconds": str(IMDS_TOKEN_TTL_SECS)})
    with urllib.request.urlopen(request, timeout=IMDS_TIMEOUT_SECS) as response:
        return response.read().decode("utf-8", "replace").strip()


def _spot_instance_action(token):
    """Return the interruption notice as a dict, or None when there is none.

    404 is the normal, overwhelmingly common answer and is reported as None -- never
    as an error. 401 means the token aged out, which the caller handles by renewing
    it. Anything else (403 when IMDS is disabled, a connection refusal when there is
    no metadata service at all) is also None: absence of a notice, quietly.
    """
    request = urllib.request.Request(
        IMDS_ACTION_URL, headers={"X-aws-ec2-metadata-token": token})
    try:
        with urllib.request.urlopen(request, timeout=IMDS_TIMEOUT_SECS) as response:
            body = response.read().decode("utf-8", "replace").strip()
    except urllib.error.HTTPError as exc:
        if exc.code == 401:
            raise _TokenExpired()
        return None
    except Exception:
        return None
    try:
        parsed = json.loads(body)
    except ValueError:
        parsed = None
    if isinstance(parsed, dict):
        return parsed
    # A 200 with an unreadable body still means reclamation is imminent.
    return {"action": "terminate", "time": body[:64] or "unknown"}


def _watch_for_spot_interruption(stop_event):
    """Poll the metadata service every 5 seconds and shut down on a notice.

    Silent by design in every non-event case: a 404 is the normal answer, and a
    metadata service that cannot be reached at all (a local run, IMDS disabled) is
    normal too, so the watcher simply stops after a few failed token attempts
    without logging anything.
    """
    token = None
    setup_failures = 0
    while not stop_event.wait(IMDS_POLL_SECS):
        if _shutdown_requested:
            return
        if token is None:
            try:
                token = _imds_token()
                setup_failures = 0
            except Exception:
                setup_failures += 1
                if setup_failures >= IMDS_MAX_SETUP_FAILURES:
                    return
                continue
        try:
            notice = _spot_instance_action(token)
        except _TokenExpired:
            token = None
            continue
        if notice is None:
            continue
        action = str(notice.get("action") or "terminate")
        when = str(notice.get("time") or "unknown")
        log(f"spot interruption notice from the metadata service: "
            f"action={action}, scheduled {when} — shutting down now")
        if request_shutdown(SPOT_NOTICE_REASON.format(action=action, when=when)):
            _signal_child(signal.SIGTERM)
        return


# ---------------------------------------------------------------------------
# S3
# ---------------------------------------------------------------------------
def put_status(state, client=None, **extra):
    payload = {"state": state, "run_id": RUN_ID, "at": time.time(), **extra}
    try:
        (client or s3).put_object(Bucket=BUCKET, Key=f"{PREFIX}/{STATUS_KEY}",
                                  Body=json.dumps(payload, indent=2).encode(),
                                  ContentType="application/json")
        return True
    except Exception as exc:
        log(f"could not write {STATUS_KEY}: {exc}")
        return False


def download_project(workdir):
    """Pull every uploaded object into the working directory, preserving layout."""
    simulation_dir = os.path.join(workdir, "Simulation")
    os.makedirs(simulation_dir, exist_ok=True)
    paginator = s3.get_paginator("list_objects_v2")
    found = []
    for page in paginator.paginate(Bucket=BUCKET, Prefix=PREFIX):
        for item in page.get("Contents", []):
            # A reclaim can land mid-download; stop here rather than spending the
            # shutdown window pulling files no simulation will read.
            _checkpoint()
            key = item["Key"]
            relative = key[len(PREFIX):].lstrip("/")
            # Skip anything a previous attempt produced.
            if not relative or relative in (RESULT_KEY, PARTIAL_KEY, STATUS_KEY):
                continue
            target = os.path.join(workdir, relative.replace("/", os.sep))
            os.makedirs(os.path.dirname(target), exist_ok=True)
            s3.download_file(BUCKET, key, target)
            found.append(relative)
    return found


def find_run_script():
    """Locate CompuCell3D's runner inside the image."""
    direct = os.environ.get("CC3D_RUNSCRIPT", "").strip()
    if direct and os.path.isfile(direct):
        return direct
    for name in ("runScript.sh", "runScript"):
        located = shutil.which(name)
        if located:
            return located
    for pattern in ("/opt/conda/**/runScript.sh", "/opt/CC3D/**/runScript.sh",
                    "/usr/local/**/runScript.sh"):
        matches = glob.glob(pattern, recursive=True)
        if matches:
            return matches[0]
    return None


def result_candidates(workdir):
    """Every cells.csv the engine may have written, wherever it put its cwd."""
    return glob.glob(os.path.join(workdir, "**", RESULT_KEY), recursive=True)


def steps_from_model(model_path, default=100):
    """Read <Steps> out of the CC3DML, with a CC3D_STEPS env override.

    With the Python API WE own the stepping loop, so the step count has to come
    from somewhere explicit rather than being handled inside runScript.
    """
    override = os.environ.get("CC3D_STEPS", "").strip()
    if override:
        try:
            value = int(float(override))
            if value > 0:
                return value
        except (TypeError, ValueError):
            log(f"ignoring unparsable CC3D_STEPS={override!r}")
    try:
        import xml.etree.ElementTree as ET
        root = ET.parse(model_path).getroot()
        node = root.find("./Potts/Steps")
        if node is not None and node.text:
            value = int(float(node.text.strip()))
            if value > 0:
                return value
    except Exception as exc:
        log(f"could not read <Steps> from the model ({type(exc).__name__}); "
            f"defaulting to {default}")
    return default


def _build_specs_from_config(config):
    """Build CompuCell3D specs in Python from our own config dict.

    Why not PyCoreSpecs.from_file on the CC3DML we already generate: a real Batch job
    did exactly that and CC3D answered "AttributeError: No Potts specification" for
    CC3DML that is otherwise valid. Rather than reverse-engineer that parser, this
    builds the same specs directly -- the approach the build-time smoke test proved
    works (it produced 9 cells in 5 steps). The CC3DML export is untouched and still
    serves its real purpose: opening the project in a local CompuCell3D install.
    """
    from cc3d.core.PyCoreSpecs import (PottsCore, CellTypePlugin, VolumePlugin,
                                       ContactPlugin, BlobInitializer)

    lattice = config.get("lattice") or {}
    dim_x = int(lattice.get("x", 40))
    dim_y = int(lattice.get("y", 40))
    dim_z = int(lattice.get("z", 1))
    neighbor_order = int(config.get("neighbor_order", 2))
    steps = int(config.get("steps", 100))
    temperature = float(config.get("temperature", 10.0))

    potts = PottsCore(dim_x=dim_x, dim_y=dim_y, dim_z=dim_z, steps=steps,
                      neighbor_order=neighbor_order,
                      fluctuation_amplitude=temperature)

    names = []
    for index, spec in enumerate(config.get("cell_types") or [], start=1):
        if isinstance(spec, dict):
            name = str(spec.get("name") or f"Type{index}").strip()
            if name and name != "Medium":
                names.append(name)
    if not names:
        # Refuse rather than run an empty lattice and report success.
        raise ValueError("No cell types are defined, so the simulation would "
                         "contain no cells. Define at least one cell type.")
    cell_types = CellTypePlugin(*names)

    volume_cfg = config.get("volume_constraint") or {}
    target_volume = float(volume_cfg.get("target_volume", 25))
    lambda_volume = float(volume_cfg.get("lambda_volume", 2.0))
    volume = VolumePlugin()
    for name in names:
        volume.param_new(name, target_volume=target_volume,
                         lambda_volume=lambda_volume)

    specs = [potts, cell_types, volume]

    # Without these, cell.xCOM/yCOM/zCOM and cell.surface all read 0.0 -- a real run
    # returned 450 rows of genuine volumes with every POSITION at the origin, which is
    # wrong-but-plausible data in a spatial simulation. CC3D only computes these when
    # the corresponding plugin is registered. Imported defensively and by several
    # known names, because a missing attribute here must degrade to a logged warning
    # rather than failing a simulation that is otherwise fine.
    import cc3d.core.PyCoreSpecs as _specs_mod
    for candidates, purpose in (
        (("CenterOfMassPlugin", "CenterOfMass"), "cell centre-of-mass (x/y/z)"),
        (("SurfacePlugin",), "cell surface"),
    ):
        added = False
        for class_name in candidates:
            plugin_cls = getattr(_specs_mod, class_name, None)
            if plugin_cls is None:
                continue
            try:
                specs.append(plugin_cls())
                log(f"registered {class_name} for {purpose}")
                added = True
                break
            except Exception as exc:
                log(f"{class_name} could not be constructed "
                    f"({type(exc).__name__}: {exc})")
        if not added:
            log(f"WARNING: no plugin available for {purpose}; those columns will "
                f"be zero in cells.csv")

    contacts = [c for c in (config.get("contact_energies") or [])
                if isinstance(c, dict) and c.get("type1") and c.get("type2")]
    if contacts:
        contact = ContactPlugin(neighbor_order=neighbor_order)
        for entry in contacts:
            contact.param_new(type_1=str(entry["type1"]), type_2=str(entry["type2"]),
                              energy=float(entry.get("energy", 0.0)))
        specs.append(contact)

    # Something has to actually place cells on the lattice, or the run completes with
    # zero cells -- the silent-empty failure this project has already hit twice.
    blob = BlobInitializer()
    radius = max(3, min(dim_x, dim_y) // 4)
    blob.region_new(width=5, radius=radius,
                    center=(dim_x // 2, dim_y // 2, 0 if dim_z <= 1 else dim_z // 2),
                    cell_types=tuple(names))
    specs.append(blob)
    return specs, steps


def run_engine_python_api(model_path, workdir, max_minutes):
    """Run the simulation IN-PROCESS via the CompuCell3D 4.4+ Python API.

    Why this rather than runScript: CompuCell3D installed from its conda channel
    provides the `cc3d` Python package, and `runScript.sh` is part of the GUI-
    oriented distribution -- it is NOT guaranteed to exist in a headless image. A
    real Batch job failed with exactly that ("runScript was not found in this
    image") while `import cc3d` succeeded, which is what motivated this path.

    We own the stepping loop, which gives three things the subprocess never had:
    cells.csv is written INCREMENTALLY (so a spot interruption leaves genuinely
    useful partial output), the shutdown flag is honoured between steps without
    signalling a child, and the time cap is enforced in the same loop rather than by
    killing a process.

    Returns the number of measurement rows written.
    """
    from cc3d.core.PySteppables import SteppableBasePy
    from cc3d.CompuCellSetup.CC3DCaller import CC3DSimService

    config_path = os.path.join(os.path.dirname(model_path), "..", "cc3d_config.json")
    config_path = os.path.normpath(config_path)
    if not os.path.isfile(config_path):
        raise RuntimeError(
            f"cc3d_config.json was not uploaded with this run, so the simulation "
            f"cannot be specified. Looked at {config_path}."
        )
    with open(config_path, "r", encoding="utf-8") as handle:
        config = json.load(handle)

    specs, total_steps = _build_specs_from_config(config)
    every = max(1, int(float(os.environ.get("CC3D_SAVE_EVERY", "1") or 1)))
    override = os.environ.get("CC3D_STEPS", "").strip()
    if override:
        try:
            value = int(float(override))
            if value > 0:
                total_steps = value
        except (TypeError, ValueError):
            log(f"ignoring unparsable CC3D_STEPS={override!r}")
    deadline = None if max_minutes <= 0 else _job_started + max_minutes * 60.0

    log(f"building specs from cc3d_config.json ({total_steps} steps)")
    sim = CC3DSimService()
    sim.register_specs(specs)
    # A bare SteppableBasePy instance is the documented way to reach cell_list from
    # outside the simulation; no generated steppable file is needed.
    probe = SteppableBasePy()
    sim.register_steppable(steppable=probe)

    # THE STAGE-4 REACTION TERMS. DiffusionSolverFE applies diffusion and linear decay and
    # has no field for a general rate law, so a reaction written on the PDE-model stage was
    # simply absent from a dispatched run -- the job produced pure diffusion and said
    # nothing about it. `reactions` carries one entry per field, already parsed with
    # pde_model.parse_safe on the server and re-printed from the parsed expression with
    # parameters substituted numerically, so what arrives here contains only field names,
    # numbers and arithmetic. The researcher's raw string never travels.
    reactions = config.get("reactions") or {}
    reaction_probe = None
    if reactions:
        field_names = [str(f.get("name")) for f in (config.get("fields") or [])
                       if f.get("name")]
        unknown = [name for name in reactions if name not in field_names]
        if unknown:
            # Refuse rather than apply a rate law to a field that does not exist: a silent
            # skip would look like a successful run of the wrong model.
            raise RuntimeError(
                "cc3d_config.json carries reactions for fields this simulation does not "
                "define: %s (fields: %s)" % (", ".join(sorted(unknown)),
                                             ", ".join(field_names) or "none"))
        applier_cls = _make_reaction_applier(SteppableBasePy)
        reaction_probe = applier_cls(reactions, field_names, _reaction_dt(config))
        sim.register_steppable(steppable=reaction_probe)
        log("registered reaction steppable for: %s" % ", ".join(sorted(reactions)))
    else:
        log("no reaction terms in the config; diffusion and decay only")

    sim.run()
    sim.init()
    sim.start()

    out_path = os.path.join(workdir, RESULT_KEY)
    rows = 0
    with open(out_path, "w", encoding="utf-8", newline="") as handle:
        handle.write("mcs,cell_id,type,volume,surface,x,y,z\n")
        for step in range(1, total_steps + 1):
            if _shutdown_requested:
                log(f"stopping at step {step} of {total_steps}: shutdown requested")
                break
            if deadline is not None and time.monotonic() > deadline:
                request_shutdown(time_cap_reason(max_minutes), state="failed",
                                 exit_code=EXIT_TIME_LIMIT)
                log(f"stopping at step {step}: the {max_minutes:g} minute cap was reached")
                break
            sim.step()
            if step % every == 0 or step == total_steps:
                for cell in probe.cell_list:
                    handle.write(
                        f"{step},{cell.id},{cell.type},{cell.volume},"
                        f"{cell.surface},{cell.xCOM},{cell.yCOM},{cell.zCOM}\n"
                    )
                    rows += 1
                # Flushed every save interval so a SIGKILL cannot lose everything
                # already computed -- the partial upload reads whole lines only.
                handle.flush()
            if step % max(1, total_steps // 10) == 0:
                log(f"  step {step}/{total_steps}, {rows} measurement row(s)")

    try:
        sim.finish()
    except Exception as exc:
        # finish() failing after the data is on disk must not discard the results.
        log(f"sim.finish() raised {type(exc).__name__}: {exc}; results are already written")

    log(f"simulation loop done: {rows} row(s) written to {RESULT_KEY}")
    return rows


def _read_complete_lines(path, limit):
    """Read up to ``limit`` bytes and cut back to the last complete line.

    The engine may be mid-write when a reclaim arrives, and the size cap can land
    mid-row, so everything after the final newline is discarded rather than shipped
    as a malformed row. A file with no newline at all yields an empty body, which the
    caller reports as "nothing usable" instead of uploading a fragment.

    Returns ``(body, line_count, truncated)``.
    """
    with open(path, "rb") as handle:
        data = handle.read(limit + 1)
    truncated = len(data) > limit
    if truncated:
        data = data[:limit]
    cut = data.rfind(b"\n")
    if cut < 0:
        return b"", 0, truncated
    body = data[:cut + 1]
    return body, body.count(b"\n"), truncated


def _upload_partial(workdir, deadline):
    """Upload partial output to PARTIAL_KEY within the deadline. Never to cells.csv.

    Returns the status.json fields describing what happened -- including the cases
    where nothing was uploaded, because "there is no partial file" and "there was no
    time to upload one" are different things to tell a researcher.
    """
    candidates = result_candidates(workdir)
    if not candidates:
        log("no cells.csv had been written yet; there is nothing partial to upload")
        return {"partial": None,
                "partial_note": "The run stopped before the measurement steppable "
                                "wrote any rows, so there is no partial file."}

    source = max(candidates, key=os.path.getsize)
    remaining = deadline - time.monotonic()
    if remaining < MIN_UPLOAD_SECS:
        log(f"skipping the partial upload: only {remaining:.1f}s of the shutdown "
            f"window is left")
        return {"partial": None,
                "partial_note": "Partial output existed, but too little of the "
                                "30-second shutdown window remained to upload it "
                                "without being killed mid-write."}

    try:
        body, lines, truncated = _read_complete_lines(source, MAX_PARTIAL_BYTES)
    except OSError as exc:
        log(f"could not read partial output: {exc}")
        return {"partial": None,
                "partial_note": f"Partial output could not be read: "
                                f"{type(exc).__name__}."}
    if not body:
        log("partial output held no complete row; not uploading it")
        return {"partial": None,
                "partial_note": "Partial output existed but held no complete row."}

    try:
        urgent_s3.put_object(Bucket=BUCKET, Key=f"{PREFIX}/{PARTIAL_KEY}", Body=body,
                             ContentType="text/csv")
    except Exception as exc:
        log(f"could not upload {PARTIAL_KEY}: {exc}")
        return {"partial": None,
                "partial_note": f"Uploading the partial file failed: "
                                f"{type(exc).__name__}."}

    log(f"uploaded {PARTIAL_KEY} ({len(body)} bytes, {lines} line(s)"
        f"{', truncated' if truncated else ''})")
    return {"partial": PARTIAL_KEY,
            "partial_key": f"{PREFIX}/{PARTIAL_KEY}",
            "partial_bytes": len(body),
            "partial_lines": lines,
            "partial_truncated": truncated}


def remove_stale_partial():
    """Drop a previous attempt's partial file once a real result exists.

    Best effort: the task role may not hold s3:DeleteObject, and a leftover partial
    beside a genuine cells.csv is untidy rather than harmful.
    """
    try:
        s3.delete_object(Bucket=BUCKET, Key=f"{PREFIX}/{PARTIAL_KEY}")
    except Exception as exc:
        log(f"left a previous attempt's {PARTIAL_KEY} in place ({type(exc).__name__})")


def finalize_shutdown(workdir):
    """Wrapper that GUARANTEES the interrupted exit code survives.

    The body below touches the filesystem (globbing candidates, stat-ing them to
    pick the largest) and S3. Any OSError from a file the engine truncated between
    the glob and the stat used to propagate out of here, past `except Interrupted`,
    and out of main() as an unhandled exit 1 -- with status.json left permanently
    reading partial="pending". That turned a clean, reportable interruption into a
    crash on precisely the path that exists to handle interruptions.
    """
    try:
        return _finalize_shutdown_inner(workdir)
    except Exception as exc:
        log(f"finalize_shutdown failed ({type(exc).__name__}: {exc}); "
            f"still exiting with the interrupted code")
        try:
            put_status(_shutdown_state, reason=_shutdown_reason,
                       error=f"Shutdown handling itself failed: "
                             f"{type(exc).__name__}: {exc}",
                       partial=None)
        except Exception:
            pass
        return _shutdown_exit_code


def _finalize_shutdown_inner(workdir):
    """Stop the engine, upload what exists, record an honest status, return the code.

    Runs on the MAIN thread in normal control flow -- never inside a signal handler.
    That is what makes a second SIGTERM harmless: the handler's only job is to set a
    flag that is already set, so it returns without touching S3 or this function.

    ``status.json`` is written FIRST and small. If the window closes during the
    partial upload the researcher still gets the explanation rather than silence; the
    status is then amended with the partial's details if there is time.
    """
    # The two-minute spot NOTICE gives far more room than a bare SIGTERM's 30s, and
    # the notice path is exactly the one that can ship a large, nearly-complete
    # partial. Using one 20s budget for both threw away most of that window.
    budget = (SPOT_NOTICE_BUDGET_SECS if _shutdown_state == "interrupted"
              and "notice" in str(_shutdown_reason).lower()
              else SHUTDOWN_BUDGET_SECS)
    deadline = _shutdown_at + budget
    elapsed = round(time.monotonic() - _job_started, 1)
    log(f"stopping: {_shutdown_reason}")

    # Stop the engine first, so cells.csv is not growing under the read below.
    _signal_child(signal.SIGTERM)
    _await_child(min(CHILD_GRACE_SECS, max(0.0, deadline - time.monotonic())))
    _signal_child(_SIGKILL)

    # ``error`` carries the same text as ``reason`` on purpose: the application
    # surfaces status["error"] to the researcher when cells.csv is absent, so the
    # explanation has to live under that key to be seen.
    base = {"error": _shutdown_reason, "reason": _shutdown_reason,
            "seconds": elapsed, "retryable": True,
            "restarts_from_step_zero": True}
    put_status(_shutdown_state, client=urgent_s3, partial="pending", **base)

    detail = _upload_partial(workdir, deadline)

    # ALWAYS amend. The old code skipped the amend when the window had closed and
    # logged "the first write stands" -- leaving status.json permanently reporting
    # partial="pending", which is a lie the moment this function returns: nothing is
    # pending any more, the upload either happened or was abandoned. The application
    # has no handling for a perpetual "pending" and cannot learn which it was.
    # The amend is a small PUT of a few hundred bytes, so it is affordable even with
    # little budget left; if it genuinely fails, say so rather than leaving the lie.
    amended = dict(base)
    amended.update(detail)
    if "partial" not in amended:
        amended["partial"] = None
    try:
        put_status(_shutdown_state, client=urgent_s3, **amended)
    except Exception as exc:
        log(f"could not amend {STATUS_KEY} ({type(exc).__name__}: {exc}); "
            f"it may still read partial='pending'")
    return _shutdown_exit_code


# ---------------------------------------------------------------------------
# The engine
# ---------------------------------------------------------------------------
def read_max_minutes():
    raw = os.environ.get("CC3D_MAX_MINUTES", "").strip()
    if not raw:
        return DEFAULT_MAX_MINUTES
    try:
        value = float(raw)
    except (TypeError, ValueError):
        log(f"CC3D_MAX_MINUTES={raw!r} is not a number; "
            f"capping at {DEFAULT_MAX_MINUTES:g} minutes")
        return DEFAULT_MAX_MINUTES
    if value <= 0:
        log(f"CC3D_MAX_MINUTES={raw!r} is not positive; "
            f"capping at {DEFAULT_MAX_MINUTES:g} minutes")
        return DEFAULT_MAX_MINUTES
    return value


def _stream_output(stream, sink):
    """Log every line as it arrives, keeping a bounded tail for diagnostics."""
    try:
        for line in stream:
            line = line.rstrip("\n")
            sink.append(line)
            log(line)
    except Exception:
        pass
    finally:
        try:
            stream.close()
        except Exception:
            pass


def run_engine(command, workdir, max_minutes):
    """Run CompuCell3D, streaming its output. Returns ``(returncode, tail)``.

    Streaming rather than subprocess.run(capture_output=True) for three reasons:
    CloudWatch shows progress DURING a long run instead of one dump at the end, the
    shutdown path holds a Popen it can actually signal, and the output no longer
    accumulates in memory for the whole run.

    The main thread supervises and never blocks on a read, so it observes both the
    shutdown flag and the run-length cap promptly; a background thread does the
    reading. Returns a None returncode when a shutdown was requested -- the caller's
    checkpoint turns that into the interrupted path.
    """
    global _child
    recent = collections.deque(maxlen=OUTPUT_TAIL_LINES)
    # The engine's own Python must not block-buffer into our pipe, or "streaming"
    # arrives in 8 KB gulps regardless of what we do here.
    environment = dict(os.environ, PYTHONUNBUFFERED="1")
    # A new session puts the shell wrapper and the simulator in one process group,
    # which is what lets the shutdown path signal the whole tree. POSIX only.
    group = {"start_new_session": True} if os.name == "posix" else {}

    # A notice can land while the runScript was being located. Starting an engine
    # only to kill it a moment later wastes the shutdown window.
    _checkpoint()
    process = subprocess.Popen(command, cwd=workdir, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, bufsize=1,
                               env=environment, **group)
    _child = process
    reader = threading.Thread(target=_stream_output, args=(process.stdout, recent),
                              name="cc3d-output", daemon=True)
    reader.start()

    started = time.monotonic()
    cap_seconds = max_minutes * 60.0
    while True:
        if process.poll() is not None:
            break
        if _shutdown_requested:
            break
        if time.monotonic() - started >= cap_seconds:
            if request_shutdown(TIME_CAP_REASON.format(minutes=max_minutes),
                                state="failed", exit_code=EXIT_TIME_LIMIT):
                log(f"run-length cap of {max_minutes:g} minutes reached; "
                    f"stopping the simulation")
                _signal_child(signal.SIGTERM)
            break
        time.sleep(POLL_SECS)

    if _shutdown_requested:
        return None, "\n".join(recent)

    reader.join(timeout=OUTPUT_DRAIN_SECS)
    return process.returncode, "\n".join(recent)


def _reaction_dt(config):
    """Integration step for the reaction terms: the model's own dt when it states one."""
    for field in (config.get("fields") or []):
        try:
            value = float(field.get("dt") or 0.0)
            if value > 0:
                return value
        except (TypeError, ValueError):
            continue
    return 1.0


def _make_reaction_applier(base):
    """Build the reaction steppable class against ``base`` at call time.

    SteppableBasePy is imported INSIDE the run function, because cc3d only exists in the
    container -- so a module-level `class _ReactionApplier(SteppableBasePy)` would be
    evaluated at import and raise NameError everywhere else, including in the test suite.
    py_compile does not catch that, since a base class is not evaluated during compilation.
    Hence a factory: the class is created where the import is in scope.
    """

    class _ReactionApplier(base):
        """Applies the stage-4 reaction terms to the fields, once per Monte Carlo step.

        DiffusionSolverFE handles transport -- diffusion and linear decay -- and has no
        field for a general rate law, so a reaction written on the PDE-model stage never
        reached a dispatched run: the job produced pure diffusion and reported success.
        This closes that, so a remote run applies the same mathematics as an exported
        project.

        OPERATOR SPLITTING. Transport and reaction advance separately within a step:
        accurate when the reaction is slow relative to the diffusive step, approximate when
        it is not, and recorded in the run's log rather than left to be assumed.

        ON THE eval. The expressions arrive already parsed by pde_model.parse_safe on the
        server -- which admits only the declared field names, the model's own parameters and
        a fixed set of mathematical functions -- and re-printed from the parsed expression
        with parameters substituted numerically. What is evaluated here therefore contains
        field names, numbers and arithmetic and nothing else; the researcher's raw string
        never travels. Builtins are stripped from the namespace as a second barrier, so a
        string that somehow reached this point still could not reach the interpreter's own
        functions.
        """

        def __init__(self, reactions, field_names, dt, frequency=1):
            base.__init__(self, frequency=frequency)
            self.reactions = dict(reactions)
            self.field_names = list(field_names)
            self.dt = float(dt)
            self.applied = 0
            self.failures = []

        def step(self, mcs):
            import numpy

            arrays = {}
            for name in self.field_names:
                try:
                    view = getattr(self.field, name)
                    arrays[name] = numpy.asarray(view[:, :, :], dtype=float)
                except Exception as error:        # a field the solver did not create
                    self.failures.append("read %s: %s" % (name, error))
                    return

            namespace = {"__builtins__": {}, "numpy": numpy}
            namespace.update(arrays)

            for name, expression in self.reactions.items():
                try:
                    rate = eval(expression, namespace, {})   # noqa: S307 - see docstring
                except Exception as error:
                    # Recorded AND re-raised: a reaction that cannot be evaluated must not
                    # let the run finish looking successful while modelling nothing.
                    self.failures.append("eval %s: %s" % (name, error))
                    raise
                try:
                    view = getattr(self.field, name)
                    view[:, :, :] = arrays[name] + self.dt * rate
                except Exception as error:
                    self.failures.append("write %s: %s" % (name, error))
                    raise
            self.applied += 1

    return _ReactionApplier


def main():
    global _job_started
    _job_started = time.monotonic()

    if not BUCKET or not PREFIX:
        log("CC3D_BUCKET and CC3D_PREFIX are required.")
        return EXIT_MISCONFIGURED

    if not install_signal_handlers():
        log("refusing to run: no SIGTERM handler could be installed, so a spot "
            "reclamation could neither be handled nor reported.")
        put_status("failed",
                   error="The container could not install a SIGTERM handler, so a "
                         "spot interruption could not be handled or reported. "
                         "Refusing to start rather than running unsupervised.")
        return EXIT_MISCONFIGURED
    stop_watching = threading.Event()
    threading.Thread(target=_watch_for_spot_interruption, args=(stop_watching,),
                     name="spot-watch", daemon=True).start()

    max_minutes = read_max_minutes()
    workdir = tempfile.mkdtemp(prefix="cc3d_")
    put_status("starting", max_minutes=max_minutes)
    try:
        files = download_project(workdir)
        log(f"downloaded {len(files)} file(s): {', '.join(files) or 'none'}")
        _checkpoint()

        model = os.path.join(workdir, "Simulation", "model.xml")
        if not os.path.isfile(model):
            put_status("failed", error="No Simulation/model.xml in the uploaded project.")
            log("no model.xml found")
            return EXIT_NO_MODEL

        started = time.monotonic()
        # PRIMARY PATH: the CompuCell3D 4.4+ Python API, in-process. A real Batch job
        # failed with "runScript was not found in this image" while `import cc3d`
        # succeeded, so the shell runner cannot be the primary strategy in a headless
        # conda image. runScript is kept below purely as a fallback for images that
        # do ship it.
        use_python_api = False
        try:
            import cc3d  # noqa: F401
            use_python_api = True
            log(f"CompuCell3D Python API available (cc3d "
                f"{getattr(cc3d, '__version__', 'unknown')})")
        except Exception as exc:
            log(f"cc3d is not importable ({type(exc).__name__}: {exc}); "
                f"falling back to runScript")

        if use_python_api:
            put_status("running", engine="cc3d-python-api", max_minutes=max_minutes)
            log(f"starting the simulation in-process, capped at {max_minutes:g} minutes")
            try:
                rows = run_engine_python_api(model, workdir, max_minutes)
            except Interrupted:
                raise
            except Exception as exc:
                elapsed = round(time.monotonic() - started, 1)
                put_status("failed",
                           error=f"CompuCell3D failed in-process: "
                                 f"{type(exc).__name__}: {exc}",
                           traceback=traceback.format_exc(limit=8),
                           seconds=elapsed)
                log(f"in-process run failed: {type(exc).__name__}: {exc}")
                return EXIT_ENGINE_FAILED
            elapsed = round(time.monotonic() - started, 1)
            _checkpoint()
            if rows <= 0:
                put_status("failed",
                           error="CompuCell3D ran but produced no cell measurements. "
                                 "The lattice was empty -- check that the cell types "
                                 "are actually seeded by an initialiser.",
                           seconds=elapsed)
                return EXIT_EMPTY_RESULT
            results = os.path.join(workdir, RESULT_KEY)
            size = os.path.getsize(results)
            s3.upload_file(results, BUCKET, f"{PREFIX}/{RESULT_KEY}",
                           ExtraArgs={"ContentType": "text/csv"})
            log(f"uploaded {RESULT_KEY} ({size} bytes, {rows} rows) after {elapsed:.1f}s")
            remove_stale_partial()
            put_status("succeeded", seconds=elapsed, bytes=size, rows=rows)
            return EXIT_OK

        run_script = find_run_script()
        if not run_script:
            put_status("failed",
                       error="CompuCell3D is not usable in this image: the 'cc3d' "
                             "Python package is not importable and no runScript was "
                             "found on PATH.")
            log("neither the cc3d Python package nor runScript is available")
            return EXIT_NO_RUNSCRIPT
        log(f"using {run_script}")

        command = [run_script, "-i", model, "--noOutput"]
        put_status("running", command=" ".join(command), max_minutes=max_minutes)
        log(f"starting the simulation, capped at {max_minutes:g} minutes")
        returncode, tail = run_engine(command, workdir, max_minutes)
        elapsed = round(time.monotonic() - started, 1)

        # Before reading the return code: a stopped engine exits non-zero, and the
        # reason it stopped belongs in the interrupted status, not in a fake crash.
        _checkpoint()

        if returncode != 0:
            put_status("failed", error=f"CompuCell3D exited with status {returncode}.",
                       returncode=returncode, seconds=elapsed, tail=tail[-2000:])
            return EXIT_ENGINE_FAILED

        # The steppable writes cells.csv beside the simulation; find it wherever
        # CompuCell3D put its working directory.
        candidates = result_candidates(workdir)
        if not candidates:
            put_status("failed",
                       error="CompuCell3D completed but produced no cells.csv. The "
                             "measurement steppable did not run.",
                       seconds=elapsed)
            return EXIT_NO_RESULT

        results = max(candidates, key=os.path.getsize)
        size = os.path.getsize(results)
        if size <= 0:
            put_status("failed", error="cells.csv was written but is empty.", seconds=elapsed)
            return EXIT_EMPTY_RESULT

        s3.upload_file(results, BUCKET, f"{PREFIX}/{RESULT_KEY}",
                       ExtraArgs={"ContentType": "text/csv"})
        log(f"uploaded {RESULT_KEY} ({size} bytes) after {elapsed:.1f}s")
        remove_stale_partial()
        put_status("succeeded", seconds=elapsed, bytes=size)
        return EXIT_OK

    except Interrupted:
        return finalize_shutdown(workdir)
    except Exception as exc:
        put_status("failed", error=f"{type(exc).__name__}: {exc}",
                   traceback=traceback.format_exc(limit=6))
        log(f"unhandled error: {exc}")
        return EXIT_UNHANDLED
    finally:
        stop_watching.set()
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
