"""
Run management shared by all three approaches.

Owns the run lifecycle so adapters stay pure functions of their inputs:

    draft -> invalid                       (validation found errors)
    draft -> compiling -> failed           (compilation raised)
    draft -> compiling -> ready
    ready -> queued -> running -> completed | failed | cancelled
    running <-> paused

Guarantees the spec asks for, enforced here rather than trusted:

* Validation runs BEFORE compilation.
* A failed compilation can never produce ``ready`` -- the transition table has no
  edge from ``compiling`` to ``ready`` except on success, and ``failed`` is terminal.
* A failed simulation can never produce ``completed``.
* Every run stores an IMMUTABLE deep copy of the configuration it was started with,
  so editing the project afterwards cannot rewrite history.
* Engine and dependency versions are recorded per run.
* Runs execute on a worker thread, so a long simulation never blocks the caller.
"""

import copy
import threading
import time
import traceback
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple

from approach_base import (
    ApproachUnavailable,
    RunCancelled,
    RunContext,
    STARTABLE_STATES,
    TERMINAL_STATES,
    get_approach,
)
from geometry import ValidationIssue, issues_to_dicts

#: Legal transitions. Anything absent is refused, which is what makes the two
#: "cannot happen" guarantees structural instead of a matter of discipline.
TRANSITIONS: Dict[str, Tuple[str, ...]] = {
    "draft": ("invalid", "compiling", "cancelled"),
    "invalid": ("draft", "compiling", "cancelled"),
    "compiling": ("ready", "failed", "cancelled"),
    "ready": ("queued", "running", "draft", "cancelled"),
    "queued": ("running", "cancelled", "failed"),
    "running": ("paused", "completed", "failed", "cancelled"),
    "paused": ("running", "cancelled", "failed"),
    "completed": (),
    "failed": (),
    "cancelled": (),
}

MAX_LOG_LINES = 2000


class RunStateError(RuntimeError):
    """An illegal state transition or an operation the run's state does not allow."""


class RunRecord:
    """One run: its immutable snapshot, its state, and its outputs."""

    def __init__(self, run_id: str, approach_id: str, project: Dict[str, Any],
                 seed: Optional[int] = None):
        self.run_id = run_id
        self.approach_id = approach_id
        # Deep copy: the run must describe what was actually executed, even after the
        # user edits the project.
        self.snapshot: Dict[str, Any] = copy.deepcopy(project)
        self.seed = seed
        self.state = "draft"
        self.created_at = time.time()
        self.started_at: Optional[float] = None
        self.finished_at: Optional[float] = None
        self.progress = 0.0
        self.message = ""
        self.error: Optional[str] = None
        self.error_kind: Optional[str] = None
        self.issues: List[ValidationIssue] = []
        self.logs: List[Dict[str, Any]] = []
        self.results: Optional[Dict[str, Any]] = None
        self.compiled: Optional[Dict[str, Any]] = None
        self.engine: Dict[str, Any] = {}
        self.history: List[Dict[str, Any]] = [{"state": "draft", "at": self.created_at}]

        self._lock = threading.RLock()
        self._cancel = threading.Event()
        self._resume = threading.Event()
        self._resume.set()                   # not paused
        self._thread: Optional[threading.Thread] = None

    # -- state ---------------------------------------------------------------
    def transition(self, new_state: str, message: str = "") -> None:
        with self._lock:
            if new_state == self.state:
                return
            allowed = TRANSITIONS.get(self.state, ())
            if new_state not in allowed:
                raise RunStateError(
                    f"Run {self.run_id} cannot go from {self.state!r} to {new_state!r}. "
                    f"Allowed: {', '.join(allowed) or 'none (terminal)'}."
                )
            self.state = new_state
            if message:
                self.message = message
            self.history.append({"state": new_state, "at": time.time(), "message": message})
            if new_state == "running" and self.started_at is None:
                self.started_at = time.time()
            if new_state in TERMINAL_STATES:
                self.finished_at = time.time()

    def log(self, message: str, level: str = "info") -> None:
        with self._lock:
            self.logs.append({"at": time.time(), "level": level, "message": str(message)})
            if len(self.logs) > MAX_LOG_LINES:
                # Keep the head (setup) and the tail (what went wrong); the middle of a
                # long run is the least informative part to drop.
                self.logs = self.logs[:200] + self.logs[-(MAX_LOG_LINES - 200):]

    def to_dict(self, include_results: bool = False) -> Dict[str, Any]:
        with self._lock:
            payload: Dict[str, Any] = {
                "run_id": self.run_id,
                "approach": self.approach_id,
                "state": self.state,
                "progress": round(float(self.progress), 4),
                "message": self.message,
                "created_at": self.created_at,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "duration_secs": ((self.finished_at or time.time()) - self.started_at)
                if self.started_at else None,
                "seed": self.seed,
                "error": self.error,
                "error_kind": self.error_kind,
                "issues": issues_to_dicts(self.issues),
                "engine": dict(self.engine),
                "log_count": len(self.logs),
                # DEEP copy each history entry, not just the list. `list(self.history)`
                # copied the container but shared the dicts, so a consumer could do
                # status(id)["history"][0]["state"] = "forged" and rewrite the record
                # in place. The state timeline IS the provenance of a run, and the API
                # layer, the UI and the exporter all hold this payload.
                "history": [dict(entry) for entry in self.history],
                "has_results": self.results is not None,
            }
            if include_results:
                # A deep copy, for the same reason: results() used to hand out the LIVE
                # object, so any downstream unit conversion, smoothing or normalisation
                # silently rewrote the archived run and the second reader of a run saw
                # different numbers than the first, with no audit trail.
                payload["results"] = copy.deepcopy(self.results)
            return payload


class RunManager:
    """Creates, starts and controls runs. Thread-safe."""

    def __init__(self) -> None:
        self._runs: Dict[str, RunRecord] = {}
        self._lock = threading.RLock()

    # -- creation ------------------------------------------------------------
    def create_run(self, project: Dict[str, Any], approach_id: Optional[str] = None,
                   seed: Optional[int] = None) -> RunRecord:
        """Validate, then compile. Returns the record in ``ready``, ``invalid`` or
        ``failed`` -- never ``ready`` unless compilation actually succeeded."""
        selected = str(approach_id or project.get("selected_approach") or "").strip()
        if not selected:
            raise ValueError("No approach selected. Choose ABM, CompuCell3D or MPC.")
        adapter = get_approach(selected)

        run_id = f"run_{uuid.uuid4().hex[:12]}"
        record = RunRecord(run_id, selected, project, seed=seed)
        with self._lock:
            self._runs[run_id] = record

        capabilities = adapter.get_capabilities()
        record.engine = {
            "approach": selected,
            "engine_name": capabilities.engine_name,
            "engine_version": capabilities.engine_version,
            "available": capabilities.available,
        }

        # 1. Availability. An unavailable approach fails honestly here; it never
        #    silently becomes a different approach.
        if not capabilities.available:
            record.error_kind = "unavailable"
            record.error = capabilities.unavailable_reason or f"{selected} is not available."
            record.log(record.error, "error")
            record.transition("compiling", "checking availability")
            record.transition("failed", "approach unavailable")
            return record

        # 2. Validation BEFORE compilation.
        try:
            issues = adapter.validate(record.snapshot)
        except Exception as exc:
            record.error_kind = "validation_crash"
            record.error = f"Validation failed unexpectedly: {type(exc).__name__}: {exc}"
            record.log(record.error, "error")
            record.transition("compiling", "validating")
            record.transition("failed", "validation error")
            return record

        record.issues = list(issues)
        errors = [i for i in issues if i.severity == "error"]
        for issue in issues:
            record.log(f"[{issue.severity}] {issue.code}: {issue.message}", issue.severity)
        if errors:
            record.error_kind = "invalid"
            record.error = "; ".join(i.message for i in errors[:5])
            record.transition("invalid", f"{len(errors)} validation error(s)")
            return record

        # 3. Compilation.
        record.transition("compiling", "compiling")
        try:
            record.compiled = adapter.compile(record.snapshot)
        except ApproachUnavailable as exc:
            record.error_kind = "unavailable"
            record.error = str(exc)
            record.log(record.error, "error")
            record.transition("failed", "approach unavailable")
            return record
        except Exception as exc:
            record.error_kind = "compile_error"
            record.error = f"{type(exc).__name__}: {exc}"
            record.log(record.error, "error")
            record.log(traceback.format_exc(limit=5), "debug")
            # NOTE: 'compiling' -> 'ready' is not taken here. A failed compilation
            # cannot present as ready, which is the guarantee this branch exists for.
            record.transition("failed", "compilation failed")
            return record

        if isinstance(record.compiled, dict) and record.compiled.get("engine_version"):
            record.engine["engine_version"] = record.compiled["engine_version"]
        record.transition("ready", "compiled")
        record.log("Compilation succeeded; run is ready.")
        return record

    # -- execution -----------------------------------------------------------
    def start(self, run_id: str, on_complete: Optional[Callable[[RunRecord], None]] = None
              ) -> RunRecord:
        record = self.get(run_id)
        with record._lock:
            if record.state not in STARTABLE_STATES:
                raise RunStateError(
                    f"Run {run_id} is {record.state!r} and cannot be started. It must be "
                    f"one of: {', '.join(STARTABLE_STATES)}."
                )
            if record.compiled is None:
                raise RunStateError(f"Run {run_id} has no compiled model to execute.")
            # A second start() during the `queued` window used to be ACCEPTED and
            # dispatch a second worker onto the same record: `queued` is in
            # STARTABLE_STATES and transition() returns early when the new state equals
            # the current one, so nothing objected. Measured: the adapter ran twice for
            # one run id, `record._thread` was overwritten so wait() joined only the
            # last, and both workers wrote record.results -- the stored payload was
            # whichever finished last. A double-clicked Run button ran the simulation
            # twice. This flag is the authoritative "a worker exists" answer, checked
            # under the same lock that sets it.
            if getattr(record, "_dispatched", False):
                raise RunStateError(
                    f"Run {run_id} has already been dispatched and is {record.state!r}. "
                    f"Starting it again would execute the same run twice."
                )
            record._dispatched = True
            record.transition("queued", "queued")

        adapter = get_approach(record.approach_id)

        def worker() -> None:
            context = RunContext(
                run_id=record.run_id,
                seed=record.seed,
                on_progress=lambda fraction, message: self._progress(record, fraction, message),
                on_log=lambda message, level: record.log(message, level),
                is_cancelled=record._cancel.is_set,
                wait_if_paused=lambda: self._wait_if_paused(record),
            )
            try:
                record.transition("running", "running")
                results = adapter.run(record.compiled, context)
                if record._cancel.is_set():
                    record.transition("cancelled", "cancelled")
                elif results is None or (isinstance(results, dict) and not results):
                    # An adapter that returns nothing used to reach `completed` with
                    # progress=1.0, has_results=False and error=None -- a finished run
                    # with nothing to plot and no recorded reason, indistinguishable in
                    # list_runs from a real success. Silence is not a result.
                    record.error_kind = "empty_results"
                    record.error = (
                        "The approach completed without returning any results. Nothing "
                        "was computed, so this run has nothing to inspect or export."
                    )
                    record.log(record.error, "error")
                    record.transition("failed", "no results returned")
                else:
                    record.results = results
                    record.progress = 1.0
                    # The run FINISHED. If a pause landed after the adapter's last
                    # checkpoint, the record is 'paused' and 'paused'->'completed' is
                    # illegal, so the generic handler used to file a successful run as a
                    # simulation failure and DISCARD the computed results -- blaming the
                    # model for what the pause did. Clear a pause that can no longer be
                    # honoured: there is nothing left to pause.
                    with record._lock:
                        if record.state == "paused":
                            record._resume.set()
                            record.transition("running", "finishing")
                    record.transition("completed", "completed")
            except RunCancelled:
                record.transition("cancelled", "cancelled")
                record.log("Run cancelled.", "warning")
            except ApproachUnavailable as exc:
                record.error_kind = "unavailable"
                record.error = str(exc)
                record.log(record.error, "error")
                # NOTE: no 'completed' here either.
                record.transition("failed", "approach unavailable")
            except Exception as exc:
                record.error_kind = "run_error"
                record.error = f"{type(exc).__name__}: {exc}"
                record.log(record.error, "error")
                record.log(traceback.format_exc(limit=5), "debug")
                record.transition("failed", "run failed")
            finally:
                if on_complete:
                    try:
                        on_complete(record)
                    except Exception:
                        record.log("A completion callback raised; ignored.", "warning")

        thread = threading.Thread(target=worker, name=f"run-{run_id}", daemon=True)
        record._thread = thread
        thread.start()
        return record

    def _progress(self, record: RunRecord, fraction: float, message: str) -> None:
        with record._lock:
            record.progress = max(0.0, min(1.0, float(fraction)))
            if message:
                record.message = message

    def _wait_if_paused(self, record: RunRecord) -> None:
        # Bounded waits so a cancel issued while paused is still noticed promptly.
        while not record._resume.is_set():
            if record._cancel.is_set():
                return
            record._resume.wait(0.1)

    # -- control -------------------------------------------------------------
    def pause(self, run_id: str) -> RunRecord:
        record = self.get(run_id)
        adapter = get_approach(record.approach_id)
        if not adapter.get_capabilities().supports_pause:
            raise RunStateError(f"{record.approach_id} does not support pausing.")
        record.transition("paused", "paused")
        record._resume.clear()
        record.log("Run paused.")
        return record

    def resume(self, run_id: str) -> RunRecord:
        record = self.get(run_id)
        with record._lock:
            # resume() used to call transition("running") UNCONDITIONALLY, and
            # TRANSITIONS["ready"] contains "running" -- so resuming a run that had
            # never been paused (or never even started) moved it to `running` with no
            # worker thread behind it. wait() returned instantly because _thread was
            # None, the state stayed `running` for ever, and results() and delete()
            # both refused it: a phantom entry showing work that is not happening and
            # that cannot be cleared without cancelling it.
            if record.state != "paused":
                raise RunStateError(
                    f"Run {run_id} is {record.state!r}, not 'paused', so there is "
                    f"nothing to resume."
                )
        record.transition("running", "resumed")
        record._resume.set()
        record.log("Run resumed.")
        return record

    def cancel(self, run_id: str) -> RunRecord:
        record = self.get(run_id)
        with record._lock:
            if record.state in TERMINAL_STATES:
                raise RunStateError(f"Run {run_id} already finished as {record.state!r}.")
        record._cancel.set()
        record._resume.set()          # release a paused worker so it can observe the cancel
        record.log("Cancellation requested.", "warning")
        if record.state in ("draft", "invalid", "ready", "queued"):
            record.transition("cancelled", "cancelled before starting")
        return record

    def wait(self, run_id: str, timeout: Optional[float] = None) -> RunRecord:
        """Block until a run reaches a terminal state. For tests and CLI use."""
        record = self.get(run_id)
        thread = record._thread
        if thread is not None:
            thread.join(timeout)
        return record

    # -- queries -------------------------------------------------------------
    def get(self, run_id: str) -> RunRecord:
        with self._lock:
            record = self._runs.get(str(run_id))
        if record is None:
            raise KeyError(f"No run {run_id!r}.")
        return record

    def status(self, run_id: str) -> Dict[str, Any]:
        return self.get(run_id).to_dict()

    def results(self, run_id: str) -> Optional[Dict[str, Any]]:
        record = self.get(run_id)
        if record.state != "completed":
            raise RunStateError(
                f"Run {run_id} is {record.state!r}; results are only available for a "
                f"completed run."
            )
        # A DEEP COPY, not the live object. Handing out the stored dict meant any
        # in-place work downstream -- a unit conversion, a smoothing pass, a
        # normalisation -- rewrote the archived run, so the second reader of a run saw
        # different numbers than the first and nothing recorded that it had changed.
        # An archived result must be immutable to its readers.
        return copy.deepcopy(record.results)

    def logs(self, run_id: str, limit: int = 500) -> List[Dict[str, Any]]:
        record = self.get(run_id)
        with record._lock:
            return list(record.logs[-int(limit):])

    def list_runs(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._lock:
            records = sorted(self._runs.values(), key=lambda r: r.created_at, reverse=True)
        return [r.to_dict() for r in records[:int(limit)]]

    def delete(self, run_id: str) -> None:
        record = self.get(run_id)
        with record._lock:
            if record.state in ("running", "paused", "queued"):
                raise RunStateError(f"Run {run_id} is {record.state!r}; cancel it first.")
        with self._lock:
            self._runs.pop(run_id, None)

    def clear_finished(self) -> int:
        with self._lock:
            finished = [k for k, r in self._runs.items() if r.state in TERMINAL_STATES]
            for key in finished:
                self._runs.pop(key, None)
        return len(finished)


#: Process-wide manager used by the API layer.
RUNS = RunManager()
