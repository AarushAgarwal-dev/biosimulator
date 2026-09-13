"""Concurrency and adversarial state-machine tests for ``run_manager``.

``test_run_manager.py`` covers the single-run happy path and the documented
"cannot happen" transitions. This file covers what it does not: several runs in
flight at once, determinism when two runs overlap, control operations aimed at one
run while another is executing, immutability of the stored configuration under
hostile mutation, transitions that are illegal but not obviously so, and thread
hygiene across many runs.

Everything here is deliberately SMALL -- a handful of MPC control steps, a handful
of Monte Carlo steps, or a stub adapter -- so the whole file runs in seconds. The
stub adapter is injected by patching ``run_manager.get_approach``; it is never
registered in the global approach registry, so no other test module can see it.

Where the implementation is wrong, these tests assert the CORRECT behaviour and
fail. A failure here is a finding, not a flaky test.
"""

import hashlib
import json
import threading
import time
import unittest
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import numpy as np

import approach_base as base
import geometry as geo
import run_manager as rm

TERMINAL = set(base.TERMINAL_STATES)


# =============================================================================
# Comparison helpers -- results contain numpy arrays, so `==` is not usable
# =============================================================================
def _canon(value: Any) -> Any:
    """Normalise a results payload to plain JSON-able Python for comparison."""
    if isinstance(value, np.ndarray):
        return _canon(value.tolist())
    if isinstance(value, np.generic):
        return _canon(value.item())
    if isinstance(value, dict):
        return {str(k): _canon(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_canon(v) for v in value]
    if isinstance(value, float):
        return round(value, 12)
    return value


def _digest(value: Any) -> str:
    blob = json.dumps(_canon(value), sort_keys=True, default=repr).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def _first_difference(left: Any, right: Any, path: str = "results") -> Optional[str]:
    """A human-readable path to the first difference, or None when equal."""
    left, right = _canon(left), _canon(right)
    if isinstance(left, dict) and isinstance(right, dict):
        for key in sorted(set(left) | set(right)):
            if key not in left:
                return f"{path}.{key}: missing on the left"
            if key not in right:
                return f"{path}.{key}: missing on the right"
            found = _first_difference(left[key], right[key], f"{path}.{key}")
            if found:
                return found
        return None
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return f"{path}: length {len(left)} vs {len(right)}"
        for index, (a, b) in enumerate(zip(left, right)):
            found = _first_difference(a, b, f"{path}[{index}]")
            if found:
                return found
        return None
    if left != right:
        return f"{path}: {left!r} vs {right!r}"
    return None


def _wait_for(predicate, timeout: float = 10.0, interval: float = 0.002) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


def _join_all(manager: rm.RunManager, records, timeout: float = 60.0) -> None:
    deadline = time.time() + timeout
    for record in records:
        manager.wait(record.run_id, timeout=max(0.01, deadline - time.time()))
    for record in records:
        _wait_for(lambda r=record: r.state in TERMINAL, timeout=max(0.1, deadline - time.time()))


# =============================================================================
# Project fixtures -- small on purpose
# =============================================================================
def _mpc_project(**overrides) -> Dict[str, Any]:
    config = {"target": 0.9, "duration": 3.0, "control_interval": 0.5,
              "prediction_horizon": 6, "control_horizon": 2}
    config.update(overrides)
    return {"approaches": {"mpc": config}, "selected_approach": "mpc"}


def _abm_project(**overrides) -> Dict[str, Any]:
    config = {
        "grid": {"width": 24, "height": 24},
        "temperature": 10.0,
        "num_mcs": 12,
        "save_every": 3,
        "seed": 7,
        "cell_types": [{
            "type_id": 1, "name": "Cell", "target_volume": 20, "lambda_volume": 2.0,
            "max_volume_before_division": 400, "color": [200, 100, 100],
            "initial_count": 2,
        }],
    }
    config.update(overrides)
    return {"domain": geo.make_rectangle(24.0, 24.0), "approaches": {"abm": config},
            "selected_approach": "abm"}


def _stub_project(marker: str, mode: str = "loop", steps: int = 4) -> Dict[str, Any]:
    return {"selected_approach": "stub",
            "approaches": {"stub": {"marker": marker, "mode": mode, "steps": steps}}}


# =============================================================================
# Stub adapter: exact control over timing, failure and blocking
# =============================================================================
class _StubAdapter(base.ApproachAdapter):
    """A controllable adapter, injected by patching ``run_manager.get_approach``.

    Per-run behaviour comes from the project config, so ONE instance can serve a
    failing run and healthy runs in the same wave -- which is the point of the
    isolation tests.
    """

    approach_id = "stub"
    label = "Stub approach"

    def __init__(self, supports_pause: bool = True, step_sleep: float = 0.004):
        self._supports_pause = supports_pause
        self.step_sleep = step_sleep
        self.gate = threading.Event()          # released to end 'block' mode
        self.entered = threading.Event()       # set once any run reached run()
        self._lock = threading.Lock()
        self.calls: List[str] = []             # run_id per adapter invocation

    def invocation_count(self, run_id: str) -> int:
        with self._lock:
            return self.calls.count(run_id)

    def get_capabilities(self) -> base.Capabilities:
        return base.Capabilities(
            approach_id=self.approach_id, label=self.label, available=True,
            dimensions=(0,), supports_pause=self._supports_pause, supports_cancel=True,
            supports_seed=True, deterministic_with_seed=True,
            engine_name="stub-engine", engine_version="1.0")

    def validate(self, project: Dict[str, Any]):
        return []

    def compile(self, project: Dict[str, Any]) -> Dict[str, Any]:
        config = (project or {}).get("approaches", {}).get("stub", {}) or {}
        return {"approach": self.approach_id, "marker": config.get("marker"),
                "mode": str(config.get("mode", "loop")),
                "steps": int(config.get("steps", 4)), "engine_version": "1.0"}

    def run(self, compiled: Dict[str, Any], context: base.RunContext) -> Optional[Dict[str, Any]]:
        with self._lock:
            self.calls.append(context.run_id)
        self.entered.set()
        mode = compiled["mode"]
        if mode == "raise":
            raise RuntimeError(f"synthetic adapter failure in {compiled['marker']}")
        if mode == "none":
            return None
        if mode == "block":
            while not self.gate.is_set():
                context.checkpoint()
                time.sleep(0.004)
        else:
            steps = compiled["steps"]
            for index in range(steps):
                context.checkpoint()
                time.sleep(self.step_sleep)
                context.progress((index + 1) / steps, f"step {index + 1}")
        return {"approach": self.approach_id, "run_id": context.run_id,
                "seed": context.seed, "marker": compiled["marker"],
                "steps": compiled["steps"]}


class _RunCase(unittest.TestCase):
    """Base case that always releases paused or blocked runs on the way out.

    Without this, a single assertion failure can leave a worker parked in
    ``_wait_if_paused`` for the rest of the process, which then breaks the thread
    hygiene tests for reasons that have nothing to do with them.
    """

    def setUp(self) -> None:
        self.manager = rm.RunManager()
        self.addCleanup(self._release_all)

    def _release_all(self) -> None:
        gate = getattr(getattr(self, "stub", None), "gate", None)
        if gate is not None:
            gate.set()
        for entry in self.manager.list_runs(limit=200):
            record = self.manager.get(entry["run_id"])
            if record.state not in TERMINAL:
                record._cancel.set()
                record._resume.set()
                try:
                    self.manager.cancel(record.run_id)
                except rm.RunStateError:
                    pass
            thread = record._thread
            if thread is not None:
                thread.join(timeout=10)


class _StubCase(_RunCase):
    """Base class that patches the approach lookup to a fresh stub."""

    supports_pause = True

    def setUp(self) -> None:
        self.stub = _StubAdapter(supports_pause=self.supports_pause)
        self._patch = patch.object(rm, "get_approach", return_value=self.stub)
        self._patch.start()
        self.addCleanup(self._patch.stop)
        super().setUp()


# =============================================================================
# (1) Several runs at once: terminal, results kept, nothing crossed
# =============================================================================
class ConcurrentRunTests(_RunCase):
    def test_mixed_concurrent_runs_all_terminate_without_crossing_results(self):
        manager = self.manager
        expected: Dict[str, Dict[str, Any]] = {}
        records = []

        for target, seed in ((0.4, 101), (0.9, 202), (1.4, 303)):
            record = manager.create_run(_mpc_project(target=target), seed=seed)
            self.assertEqual(record.state, "ready", msg=record.error or "")
            expected[record.run_id] = {"approach": "mpc", "target": target, "seed": seed}
            records.append(record)
        for seed in (11, 22, 33):
            record = manager.create_run(_abm_project(num_mcs=6), seed=seed)
            self.assertEqual(record.state, "ready", msg=record.error or "")
            expected[record.run_id] = {"approach": "abm", "target": None, "seed": seed}
            records.append(record)

        # Start them as close to simultaneously as the interpreter allows.
        barrier = threading.Barrier(len(records) + 1)

        def launch(rec):
            barrier.wait()
            manager.start(rec.run_id)

        starters = [threading.Thread(target=launch, args=(r,), daemon=True) for r in records]
        for thread in starters:
            thread.start()
        barrier.wait()
        for thread in starters:
            thread.join(timeout=30)
        _join_all(manager, records, timeout=120)

        for record in records:
            info = expected[record.run_id]
            self.assertIn(record.state, TERMINAL,
                          msg=f"{record.run_id} ({info['approach']}) never reached a terminal "
                              f"state; it is {record.state!r}")
            self.assertEqual(record.state, "completed",
                             msg=f"{record.run_id} ({info['approach']}) ended {record.state!r}: "
                                 f"{record.error}")

        # No run may lose its results, and no run may serve another run's results.
        seen_ids = set()
        for record in records:
            info = expected[record.run_id]
            results = manager.results(record.run_id)
            self.assertIsNotNone(results, msg=f"{record.run_id} completed but lost its results")
            self.assertTrue(record.to_dict()["has_results"])
            self.assertEqual(results["approach"], info["approach"],
                             msg=f"{record.run_id} was an {info['approach']} run but returned "
                                 f"{results['approach']!r} results -- cross-contamination")
            self.assertEqual(manager.status(record.run_id)["run_id"], record.run_id)
            self.assertEqual(manager.status(record.run_id)["seed"], info["seed"])
            self.assertNotIn(id(results), seen_ids,
                             msg=f"{record.run_id} shares its results object with another run")
            seen_ids.add(id(results))
            if info["approach"] == "mpc":
                self.assertIn("series", results)
                self.assertAlmostEqual(
                    results["series"]["target"][-1], info["target"], places=9,
                    msg=f"{record.run_id} was configured for target {info['target']} but its "
                        f"results carry target {results['series']['target'][-1]} -- results "
                        f"belong to a different run")
            else:
                self.assertIn("lattice_frames", results)
                self.assertEqual(results["seed"], info["seed"],
                                 msg=f"{record.run_id} reports seed {results['seed']} but was "
                                     f"created with {info['seed']}")

    def test_concurrent_creation_loses_no_run(self):
        manager = self.manager
        created: List[str] = []
        errors: List[str] = []
        lock = threading.Lock()
        barrier = threading.Barrier(8)

        def create():
            try:
                barrier.wait()
                record = manager.create_run(_mpc_project(duration=1.0), seed=1)
                with lock:
                    created.append(record.run_id)
            except Exception as exc:                      # pragma: no cover - reported
                with lock:
                    errors.append(f"{type(exc).__name__}: {exc}")

        threads = [threading.Thread(target=create, daemon=True) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)

        self.assertEqual(errors, [])
        self.assertEqual(len(created), 8)
        self.assertEqual(len(set(created)), 8, msg="two concurrent runs got the same run_id")
        listed = {entry["run_id"] for entry in manager.list_runs(limit=100)}
        for run_id in created:
            self.assertIn(run_id, listed, msg=f"{run_id} vanished from the registry")
            manager.get(run_id)


# =============================================================================
# (2) Determinism under concurrency
# =============================================================================
class ConcurrentDeterminismTests(_RunCase):
    def test_two_concurrent_mpc_runs_with_the_same_seed_agree(self):
        manager = self.manager
        first = manager.create_run(_mpc_project(duration=4.0), seed=555)
        second = manager.create_run(_mpc_project(duration=4.0), seed=555)
        manager.start(first.run_id)
        manager.start(second.run_id)
        _join_all(manager, [first, second], timeout=60)
        self.assertEqual(first.state, "completed", msg=first.error or "")
        self.assertEqual(second.state, "completed", msg=second.error or "")
        difference = _first_difference(manager.results(first.run_id),
                                      manager.results(second.run_id))
        self.assertIsNone(difference,
                          msg=f"two concurrent MPC runs with seed 555 disagree at {difference}")

    def test_two_concurrent_abm_runs_with_the_same_seed_agree(self):
        """ABM advertises deterministic_with_seed=True, so this must hold."""
        capabilities = base.get_approach("abm").get_capabilities()
        self.assertTrue(capabilities.deterministic_with_seed)

        manager = self.manager
        first = manager.create_run(_abm_project(), seed=4242)
        second = manager.create_run(_abm_project(), seed=4242)
        manager.start(first.run_id)
        manager.start(second.run_id)
        _join_all(manager, [first, second], timeout=120)
        self.assertEqual(first.state, "completed", msg=first.error or "")
        self.assertEqual(second.state, "completed", msg=second.error or "")

        left, right = manager.results(first.run_id), manager.results(second.run_id)
        difference = _first_difference(
            {"t": left["t"], "cell_counts": left["cell_counts"],
             "lattice": _digest(left["lattice_frames"]), "summary": left["summary"]},
            {"t": right["t"], "cell_counts": right["cell_counts"],
             "lattice": _digest(right["lattice_frames"]), "summary": right["summary"]})
        self.assertIsNone(
            difference,
            msg=f"two CONCURRENT ABM runs with seed 4242 produced different results "
                f"({difference}). A seeded run must be reproducible even when another "
                f"run overlaps it.")

    def test_concurrent_abm_run_matches_the_same_seed_run_alone(self):
        """A neighbouring run with a different seed must not alter this run's output."""
        manager = self.manager
        alone = manager.create_run(_abm_project(), seed=4242)
        manager.start(alone.run_id)
        manager.wait(alone.run_id, timeout=120)
        self.assertEqual(alone.state, "completed", msg=alone.error or "")
        reference = manager.results(alone.run_id)

        same_seed = manager.create_run(_abm_project(), seed=4242)
        neighbour = manager.create_run(_abm_project(), seed=99999)
        manager.start(same_seed.run_id)
        manager.start(neighbour.run_id)
        _join_all(manager, [same_seed, neighbour], timeout=120)
        self.assertEqual(same_seed.state, "completed", msg=same_seed.error or "")
        self.assertEqual(neighbour.state, "completed", msg=neighbour.error or "")

        overlapped = manager.results(same_seed.run_id)
        difference = _first_difference(
            {"t": reference["t"], "cell_counts": reference["cell_counts"],
             "lattice": _digest(reference["lattice_frames"])},
            {"t": overlapped["t"], "cell_counts": overlapped["cell_counts"],
             "lattice": _digest(overlapped["lattice_frames"])})
        self.assertIsNone(
            difference,
            msg=f"ABM seed 4242 gave a different answer when a seed-99999 run overlapped it "
                f"({difference}). A neighbouring run leaked into this run's random stream, so "
                f"a published seed does not reproduce the figure.")


# =============================================================================
# (3) Cancelling one run must not disturb another
# =============================================================================
class CancellationIsolationTests(_RunCase):
    def test_cancelling_one_mpc_run_leaves_a_concurrent_run_bit_identical(self):
        manager = self.manager
        reference = manager.create_run(_mpc_project(duration=8.0), seed=7)
        manager.start(reference.run_id)
        manager.wait(reference.run_id, timeout=60)
        self.assertEqual(reference.state, "completed", msg=reference.error or "")
        expected = manager.results(reference.run_id)

        victim = manager.create_run(_mpc_project(duration=400.0), seed=7)
        survivor = manager.create_run(_mpc_project(duration=8.0), seed=7)
        manager.start(victim.run_id)
        manager.start(survivor.run_id)
        self.assertTrue(_wait_for(lambda: victim.state == "running"),
                        msg="the run to be cancelled never started")
        manager.cancel(victim.run_id)
        _join_all(manager, [victim, survivor], timeout=60)

        self.assertEqual(victim.state, "cancelled")
        self.assertIsNone(victim.results)
        with self.assertRaises(rm.RunStateError):
            manager.results(victim.run_id)

        self.assertEqual(survivor.state, "completed",
                         msg=f"cancelling another run left this one {survivor.state!r}: "
                             f"{survivor.error}")
        difference = _first_difference(expected, manager.results(survivor.run_id))
        self.assertIsNone(difference,
                          msg=f"a concurrent cancellation changed this run's results at "
                              f"{difference}")


class StubCancellationIsolationTests(_StubCase):
    def test_cancelling_the_middle_run_of_a_wave_leaves_the_others_intact(self):
        records = [self.manager.create_run(_stub_project(f"run-{i}", mode="loop", steps=8),
                                           seed=i) for i in range(3)]
        blocked = self.manager.create_run(_stub_project("victim", mode="block"), seed=99)
        for record in records + [blocked]:
            self.manager.start(record.run_id)
        self.assertTrue(_wait_for(lambda: blocked.state == "running"))
        self.manager.cancel(blocked.run_id)
        _join_all(self.manager, records + [blocked], timeout=30)

        self.assertEqual(blocked.state, "cancelled")
        for index, record in enumerate(records):
            self.assertEqual(record.state, "completed",
                             msg=f"run-{index} ended {record.state!r}: {record.error}")
            results = self.manager.results(record.run_id)
            self.assertEqual(results["marker"], f"run-{index}")
            self.assertEqual(results["run_id"], record.run_id)
            self.assertIsNone(record.error)

    def test_cancelling_a_run_does_not_release_another_runs_pause(self):
        paused = self.manager.create_run(_stub_project("paused", mode="block"), seed=1)
        other = self.manager.create_run(_stub_project("other", mode="block"), seed=2)
        self.manager.start(paused.run_id)
        self.manager.start(other.run_id)
        self.assertTrue(_wait_for(lambda: paused.state == "running" and other.state == "running"))
        self.manager.pause(paused.run_id)
        self.manager.cancel(other.run_id)
        self.manager.wait(other.run_id, timeout=30)
        self.assertEqual(other.state, "cancelled")
        self.assertEqual(paused.state, "paused",
                         msg=f"cancelling one run moved a paused run to {paused.state!r}")
        self.assertFalse(paused._resume.is_set(),
                         msg="cancelling one run released another run's pause gate")
        self.manager.resume(paused.run_id)
        self.stub.gate.set()
        self.manager.wait(paused.run_id, timeout=30)
        self.assertEqual(paused.state, "completed", msg=paused.error or "")


# =============================================================================
# (4) Pause then resume == an uninterrupted run
# =============================================================================
class PauseResumeEquivalenceTests(_RunCase):
    #: Long enough that a pause always lands with hundreds of checkpoints to go.
    # Wall-clock headroom for the mid-flight pause tests. Raised from 200.0: at 200 the
    # window between "started" and "finished" was short enough that the full suite's load
    # could close it before a pause landed, which showed up as a flaky failure in
    # test_pause_resume_gives_the_same_result_as_an_uninterrupted_run. Longer runs cost a
    # few seconds and buy a deterministic result.
    LONG = 500.0

    def _pause_mid_run(self, record) -> None:
        """Pause once the run is demonstrably in flight, and confirm the pause landed.

        This polled for `0.02 < progress < 0.45` and then asserted the state was "paused"
        immediately. Both halves are races, and they only bite under load: this module
        passes 3/3 in isolation but failed once inside the full 850-test suite, where a
        busy CPU makes the polling coarse enough to step straight over a narrow window,
        and makes a short run finish before the pause can be applied.

        A test that fails only when the suite is busy is worse than no test -- it teaches
        you to re-run rather than to read. So the window is wider (anywhere before the
        last fifteen percent still proves a mid-flight pause), and the pause is now WAITED
        for rather than asserted instantaneously, with the two failure modes reported
        separately: the gate not holding is a product defect, the run outrunning the pause
        is this harness being too slow, and they should never share one message.
        """
        self.assertTrue(
            _wait_for(lambda: record.state == "running" and 0.02 < record.progress < 0.85,
                      timeout=30),
            msg=f"never observed the run mid-flight (state={record.state!r}, "
                f"progress={record.progress})")
        self.manager.pause(record.run_id)
        settled = _wait_for(lambda: record.state in ("paused", "completed"), timeout=15)
        self.assertTrue(settled,
                        msg=f"pause() neither paused nor completed the run within 15s "
                            f"(state={record.state!r}, progress={record.progress})")
        self.assertNotEqual(
            record.state, "completed",
            msg="the run finished before the pause could be applied, so this harness "
                "is too slow to test the pause gate -- raise LONG rather than relaxing "
                "the assertion, because a pause that lands after completion proves "
                "nothing either way")

    def test_pause_resume_gives_the_same_result_as_an_uninterrupted_run(self):
        manager = self.manager
        straight = manager.create_run(_mpc_project(duration=self.LONG), seed=8080)
        manager.start(straight.run_id)
        manager.wait(straight.run_id, timeout=120)
        self.assertEqual(straight.state, "completed", msg=straight.error or "")
        expected = manager.results(straight.run_id)

        interrupted = manager.create_run(_mpc_project(duration=self.LONG), seed=8080)
        manager.start(interrupted.run_id)
        self._pause_mid_run(interrupted)
        self.assertEqual(interrupted.state, "paused")
        frozen = interrupted.progress
        time.sleep(0.2)
        self.assertEqual(interrupted.progress, frozen,
                         msg="a paused run kept making progress, so the pause gate is not "
                             "actually blocking the worker")
        manager.resume(interrupted.run_id)
        manager.wait(interrupted.run_id, timeout=120)

        self.assertEqual(interrupted.state, "completed", msg=interrupted.error or "")
        self.assertIn("paused", [entry["state"] for entry in interrupted.history])
        difference = _first_difference(expected, manager.results(interrupted.run_id))
        self.assertIsNone(difference,
                          msg=f"pause/resume changed the result of seed 8080 at {difference}")

    def test_pausing_one_run_does_not_stall_a_concurrent_run(self):
        manager = self.manager
        held = manager.create_run(_mpc_project(duration=self.LONG), seed=1)
        free = manager.create_run(_mpc_project(duration=6.0), seed=1)
        manager.start(held.run_id)
        self._pause_mid_run(held)
        manager.start(free.run_id)
        manager.wait(free.run_id, timeout=60)
        self.assertEqual(free.state, "completed",
                         msg=f"a paused run blocked an unrelated run, which ended "
                             f"{free.state!r}: {free.error}")
        manager.cancel(held.run_id)
        manager.wait(held.run_id, timeout=60)
        self.assertEqual(held.state, "cancelled")


class LatePauseTests(_StubCase):
    def test_pausing_a_run_that_then_finishes_does_not_turn_success_into_failure(self):
        """A pause issued after the adapter's last checkpoint.

        The worker is past its final ``checkpoint()``, so it never observes the
        pause: it returns its results and transitions to a terminal state from
        ``paused``.
        """
        self.stub.step_sleep = 0.4                    # one long step, one checkpoint
        record = self.manager.create_run(_stub_project("late-pause", steps=1), seed=1)
        self.manager.start(record.run_id)
        self.assertTrue(_wait_for(lambda: record.state == "running", timeout=30))
        time.sleep(0.05)                              # now inside the un-checkpointed tail
        self.manager.pause(record.run_id)
        self.assertEqual(record.state, "paused")
        self.assertTrue(_wait_for(lambda: record.state in TERMINAL, timeout=30))

        self.assertEqual(
            record.state, "completed",
            msg=f"the adapter finished successfully but the run reports {record.state!r} "
                f"({record.error}). Pausing a run that is already past its last checkpoint "
                f"leaves it in 'paused', from which the worker's transition to 'completed' "
                f"is illegal; the RunStateError is then caught by the worker's generic "
                f"handler and recorded as a run failure. A researcher who pauses to look at "
                f"a nearly-finished run loses it, and the stored reason blames the "
                f"simulation rather than the pause.")
        self.assertTrue(record.to_dict()["has_results"])
        self.assertIsNone(record.error)


# =============================================================================
# (5) The configuration snapshot really is immutable
# =============================================================================
class SnapshotImmutabilityTests(_RunCase):
    def test_deep_mutation_of_the_project_after_create_run_is_not_reflected(self):
        manager = self.manager
        project = _abm_project()
        record = manager.create_run(project)
        self.assertEqual(record.state, "ready", msg=record.error or "")
        before = _digest(record.snapshot)

        config = project["approaches"]["abm"]
        config["num_mcs"] = 9999
        config["grid"]["width"] = 1
        config["cell_types"][0]["target_volume"] = 12345
        config["cell_types"].append({"type_id": 2, "name": "Injected"})
        project["approaches"]["abm"] = {"num_mcs": 1}
        project["approaches"]["mpc"] = {"target": 42.0}
        project["selected_approach"] = "mpc"
        if isinstance(project.get("domain"), dict):
            project["domain"]["kind"] = "tampered"

        self.assertEqual(_digest(record.snapshot), before,
                         msg="editing the project after create_run rewrote the stored "
                             "snapshot, so the run record no longer describes what was run")
        stored = record.snapshot["approaches"]["abm"]
        self.assertEqual(stored["num_mcs"], 12)
        self.assertEqual(stored["grid"]["width"], 24)
        self.assertEqual(stored["cell_types"][0]["target_volume"], 20)
        self.assertEqual(len(stored["cell_types"]), 1)
        self.assertEqual(record.approach_id, "abm")

    def test_mutating_the_project_while_the_run_executes_changes_nothing(self):
        manager = self.manager
        project = _mpc_project(target=0.9, duration=40.0)
        record = manager.create_run(project, seed=5)
        before_snapshot = _digest(record.snapshot)
        before_compiled = _digest(record.compiled)
        manager.start(record.run_id)
        self.assertTrue(_wait_for(lambda: record.state == "running"))
        project["approaches"]["mpc"]["target"] = 77.0
        project["approaches"]["mpc"]["duration"] = 0.5
        manager.wait(record.run_id, timeout=120)

        self.assertEqual(record.state, "completed", msg=record.error or "")
        self.assertEqual(_digest(record.snapshot), before_snapshot,
                         msg="the snapshot changed while the run was executing")
        self.assertEqual(_digest(record.compiled), before_compiled,
                         msg="the compiled model changed while the run was executing")
        self.assertAlmostEqual(manager.results(record.run_id)["series"]["target"][-1], 0.9,
                               places=9)

    def test_status_payload_cannot_be_used_to_rewrite_run_history(self):
        manager = self.manager
        record = manager.create_run(_mpc_project(prediction_horizon=0))
        self.assertEqual(record.state, "invalid")
        recorded_states = [entry["state"] for entry in record.history]
        recorded_engine = dict(record.engine)
        recorded_messages = [issue.message for issue in record.issues]

        payload = manager.status(record.run_id)
        payload["history"].append({"state": "completed", "at": 0})
        payload["history"][0]["state"] = "forged"
        payload["engine"]["engine_name"] = "forged"
        for issue in payload["issues"]:
            issue["message"] = "forged"

        self.assertEqual(
            [entry["state"] for entry in record.history], recorded_states,
            msg="mutating an entry of status()['history'] rewrote the run's own history. "
                "to_dict() copies the history LIST but not the dicts inside it, so any "
                "caller holding a status payload -- the API layer, the UI, an exporter -- "
                "can silently rewrite the record of which states a run passed through.")
        self.assertEqual(dict(record.engine), recorded_engine,
                         msg="status()['engine'] aliases the record's engine dict")
        self.assertEqual([issue.message for issue in record.issues], recorded_messages,
                         msg="status()['issues'] aliases the record's validation issues")

    def test_results_handed_to_a_caller_cannot_rewrite_the_stored_results(self):
        manager = self.manager
        record = manager.create_run(_mpc_project(), seed=3)
        manager.start(record.run_id)
        manager.wait(record.run_id, timeout=60)
        self.assertEqual(record.state, "completed", msg=record.error or "")
        before = _digest(record.results)

        handed_out = manager.results(record.run_id)
        handed_out["series"]["output"][0] = 999.0
        handed_out["summary"]["mean_abs_error"] = -1.0
        handed_out["injected"] = True

        self.assertEqual(_digest(record.results), before,
                         msg="a caller that edits the dict returned by results() silently "
                             "rewrites the stored run results, so an exporter or unit "
                             "conversion can corrupt the archived record in place")


# =============================================================================
# (6) Illegal transitions are refused
# =============================================================================
class IllegalTransitionTests(_StubCase):
    def _capture_worker_crashes(self) -> List[str]:
        """Turn a worker thread dying outside the state machine into evidence."""
        crashes: List[str] = []
        previous = threading.excepthook

        def hook(args):
            if str(getattr(args, "thread", None) or "").find("run-") >= 0 or True:
                crashes.append(f"{args.exc_type.__name__}: {args.exc_value}")

        threading.excepthook = hook
        self.addCleanup(lambda: setattr(threading, "excepthook", previous))
        return crashes

    def test_starting_a_queued_run_again_is_refused_and_never_double_executes(self):
        record = self.manager.create_run(_stub_project("double", mode="block"), seed=1)
        crashes = self._capture_worker_crashes()
        original = rm.RunRecord.transition

        def slow(self_record, new_state, message=""):
            if new_state == "running":
                time.sleep(0.3)                 # hold the queued window open
            return original(self_record, new_state, message)

        threads = []
        with patch.object(rm.RunRecord, "transition", slow):
            self.manager.start(record.run_id)
            threads.append(record._thread)
            time.sleep(0.05)
            self.assertEqual(record.state, "queued",
                             msg="the queued window closed too early to test the second start")
            raised = None
            try:
                self.manager.start(record.run_id)
                threads.append(record._thread)
            except rm.RunStateError as exc:
                raised = exc
            time.sleep(0.6)                     # let both workers pass the delayed transition
            executions = self.stub.invocation_count(record.run_id)
        self.stub.gate.set()
        for thread in threads:
            if thread is not None:
                thread.join(timeout=30)
        _wait_for(lambda: record.state in TERMINAL, timeout=30)

        self.assertEqual(
            executions, 1,
            msg=f"the adapter ran {executions} times for ONE run id. A second start() while "
                f"the run was still 'queued' was accepted -- 'queued' is in STARTABLE_STATES "
                f"and re-entering 'queued' is a no-op transition -- so a second worker thread "
                f"executed the same compiled model and both wrote to record.results. The "
                f"stored output is whichever thread finished last, the recorded seed no "
                f"longer explains it, and for ABM the two workers also share one global RNG.")
        self.assertEqual(
            crashes, [],
            msg=f"a worker thread died outside the state machine: {crashes}. The worker's "
                f"own error handler calls transition('failed'), which itself raises once the "
                f"run is terminal, so the failure escapes the thread entirely: nothing is "
                f"logged on the record and the only trace is a traceback on stderr.")
        self.assertIsNotNone(
            raised,
            msg="start() was accepted twice for the same run; a run that is already "
                "dispatched must be refused, not queued again")

    def test_starting_a_running_run_is_refused(self):
        record = self.manager.create_run(_stub_project("busy", mode="block"), seed=1)
        self.manager.start(record.run_id)
        self.assertTrue(_wait_for(lambda: record.state == "running"))
        with self.assertRaises(rm.RunStateError):
            self.manager.start(record.run_id)
        self.stub.gate.set()
        self.manager.wait(record.run_id, timeout=30)
        self.assertEqual(record.state, "completed")
        self.assertEqual(self.stub.invocation_count(record.run_id), 1)

    def test_resuming_a_run_that_was_never_paused_is_refused(self):
        record = self.manager.create_run(_stub_project("never-paused"), seed=1)
        self.assertEqual(record.state, "ready")
        with self.assertRaises(rm.RunStateError):
            self.manager.resume(record.run_id)
        self.assertEqual(record.state, "ready")

    def test_resume_on_a_ready_run_must_not_create_a_phantom_running_run(self):
        """The harm the previous test guards against: 'running' with no worker."""
        record = self.manager.create_run(_stub_project("phantom"), seed=1)
        try:
            self.manager.resume(record.run_id)
        except rm.RunStateError:
            return                                   # correctly refused
        self.manager.wait(record.run_id, timeout=1)
        self.assertNotEqual(
            record.state, "running",
            msg="resume() moved a never-started run to 'running' with no worker thread. It "
                "reports itself as running for ever, results() refuses it, delete() refuses "
                "it, and the queue shows work that is not happening.")

    def test_resuming_a_completed_run_is_refused(self):
        record = self.manager.create_run(_stub_project("done", steps=2), seed=1)
        self.manager.start(record.run_id)
        self.manager.wait(record.run_id, timeout=30)
        self.assertEqual(record.state, "completed")
        with self.assertRaises(rm.RunStateError):
            self.manager.resume(record.run_id)
        self.assertEqual(record.state, "completed")

    def test_cancelling_a_completed_run_is_refused_and_keeps_its_results(self):
        record = self.manager.create_run(_stub_project("finished", steps=2), seed=1)
        self.manager.start(record.run_id)
        self.manager.wait(record.run_id, timeout=30)
        self.assertEqual(record.state, "completed")
        before = _digest(record.results)
        with self.assertRaises(rm.RunStateError):
            self.manager.cancel(record.run_id)
        self.assertEqual(record.state, "completed")
        self.assertEqual(_digest(record.results), before)

    def test_cancelling_twice_is_refused_the_second_time(self):
        record = self.manager.create_run(_stub_project("twice"), seed=1)
        self.manager.cancel(record.run_id)
        self.assertEqual(record.state, "cancelled")
        with self.assertRaises(rm.RunStateError):
            self.manager.cancel(record.run_id)

    def test_cancelling_a_failed_run_is_refused(self):
        record = self.manager.create_run(_stub_project("boom", mode="raise"), seed=1)
        self.manager.start(record.run_id)
        self.manager.wait(record.run_id, timeout=30)
        self.assertEqual(record.state, "failed")
        with self.assertRaises(rm.RunStateError):
            self.manager.cancel(record.run_id)

    def test_pausing_a_run_that_is_not_running_is_refused_without_arming_the_gate(self):
        record = self.manager.create_run(_stub_project("early", steps=2), seed=1)
        with self.assertRaises(rm.RunStateError):
            self.manager.pause(record.run_id)
        self.assertEqual(record.state, "ready")
        self.assertTrue(record._resume.is_set(),
                        msg="a refused pause left the pause gate closed, so the next start() "
                            "would hang for ever")
        self.manager.start(record.run_id)
        self.manager.wait(record.run_id, timeout=30)
        self.assertEqual(record.state, "completed")

    def test_pausing_a_completed_run_is_refused(self):
        record = self.manager.create_run(_stub_project("late", steps=2), seed=1)
        self.manager.start(record.run_id)
        self.manager.wait(record.run_id, timeout=30)
        self.assertEqual(record.state, "completed")
        with self.assertRaises(rm.RunStateError):
            self.manager.pause(record.run_id)
        self.assertEqual(record.state, "completed")

    def test_results_for_a_failed_run_are_refused(self):
        record = self.manager.create_run(_stub_project("failed-results", mode="raise"), seed=1)
        self.manager.start(record.run_id)
        self.manager.wait(record.run_id, timeout=30)
        self.assertEqual(record.state, "failed")
        with self.assertRaises(rm.RunStateError):
            self.manager.results(record.run_id)
        self.assertFalse(self.manager.status(record.run_id)["has_results"])


# =============================================================================
# (7) An adapter that raises
# =============================================================================
class AdapterFailureTests(_StubCase):
    def test_a_raising_adapter_reaches_failed_with_an_error_and_no_results(self):
        record = self.manager.create_run(_stub_project("kaboom", mode="raise"), seed=1)
        self.manager.start(record.run_id)
        self.manager.wait(record.run_id, timeout=30)

        self.assertEqual(record.state, "failed")
        self.assertNotEqual(record.state, "completed")
        self.assertNotIn("completed", [entry["state"] for entry in record.history])
        self.assertEqual(record.error_kind, "run_error")
        self.assertIn("synthetic adapter failure in kaboom", record.error or "")
        self.assertIsNone(record.results)
        self.assertFalse(record.to_dict()["has_results"])
        self.assertTrue(any(entry["level"] == "error" for entry in self.manager.logs(
            record.run_id)), msg="a failed run recorded no error-level log line")

    def test_one_failing_run_does_not_poison_concurrent_healthy_runs(self):
        failing = self.manager.create_run(_stub_project("bad", mode="raise"), seed=1)
        healthy = [self.manager.create_run(_stub_project(f"good-{i}", steps=8), seed=10 + i)
                   for i in range(2)]
        for record in [failing] + healthy:
            self.manager.start(record.run_id)
        _join_all(self.manager, [failing] + healthy, timeout=30)

        self.assertEqual(failing.state, "failed")
        for index, record in enumerate(healthy):
            self.assertEqual(record.state, "completed",
                             msg=f"good-{index} ended {record.state!r} because a concurrent "
                                 f"run failed: {record.error}")
            self.assertIsNone(record.error,
                              msg=f"good-{index} inherited an error from the failing run: "
                                  f"{record.error}")
            self.assertEqual(self.manager.results(record.run_id)["marker"], f"good-{index}")
            self.assertEqual(record.error_kind, None)

    def test_an_adapter_returning_nothing_is_not_reported_as_a_successful_run(self):
        record = self.manager.create_run(_stub_project("empty", mode="none"), seed=1)
        self.manager.start(record.run_id)
        self.manager.wait(record.run_id, timeout=30)
        self.assertIn(record.state, TERMINAL)
        if record.state == "completed":
            self.assertTrue(
                record.to_dict()["has_results"],
                msg="the run reports 'completed' with has_results=False. A researcher sees a "
                    "finished run with nothing to plot and no error explaining why; an "
                    "adapter that produced no output must fail, not complete.")


# =============================================================================
# (8) Thread and handle hygiene
# =============================================================================
class ThreadHygieneTests(_StubCase):
    @staticmethod
    def _run_threads() -> List[str]:
        return [t.name for t in threading.enumerate() if t.name.startswith("run-")]

    def _settle(self, ignore: Optional[set] = None, timeout: float = 5.0) -> List[str]:
        """Wait for this test's worker threads to end; return what is left."""
        ignore = ignore or set()
        _wait_for(lambda: not [n for n in self._run_threads() if n not in ignore],
                  timeout=timeout)
        return [n for n in self._run_threads() if n not in ignore]

    def test_many_sequential_runs_leave_no_worker_threads_behind(self):
        foreign = set(self._run_threads())          # anything another test left running
        baseline = threading.active_count()
        for index in range(12):
            record = self.manager.create_run(_stub_project(f"seq-{index}", steps=2), seed=index)
            self.manager.start(record.run_id)
            self.manager.wait(record.run_id, timeout=30)
            self.assertEqual(record.state, "completed", msg=record.error or "")
        leftover = self._settle(ignore=foreign)
        self.assertEqual(leftover, [],
                         msg=f"worker threads outlived their runs: {leftover}")
        self.assertLessEqual(
            threading.active_count(), baseline + 1,
            msg=f"thread count went from {baseline} to {threading.active_count()} over 12 "
                f"runs; a long session would accumulate one thread per run")

    def test_a_concurrent_wave_leaves_no_threads_or_file_handles_behind(self):
        foreign = set(self._run_threads())
        baseline_threads = threading.active_count()
        baseline_handles = self._handle_count()

        records = [self.manager.create_run(_stub_project(f"wave-{i}", steps=3), seed=i)
                   for i in range(8)]
        for record in records:
            self.manager.start(record.run_id)
        _join_all(self.manager, records, timeout=60)
        for record in records:
            self.assertEqual(record.state, "completed", msg=record.error or "")
        leftover = self._settle(ignore=foreign)

        self.assertEqual(leftover, [], msg=f"worker threads outlived their runs: {leftover}")
        self.assertLessEqual(threading.active_count(), baseline_threads + 1,
                             msg=f"threads grew from {baseline_threads} to "
                                 f"{threading.active_count()} after an 8-run wave")
        if baseline_handles is not None:
            grown = self._handle_count() - baseline_handles
            self.assertLessEqual(grown, 8,
                                 msg=f"open OS handles grew by {grown} across 8 runs")

    @staticmethod
    def _handle_count() -> Optional[int]:
        try:
            import psutil
        except Exception:                                # pragma: no cover
            return None
        process = psutil.Process()
        for attribute in ("num_handles", "num_fds"):
            counter = getattr(process, attribute, None)
            if counter is not None:
                try:
                    return int(counter())
                except Exception:                        # pragma: no cover
                    return None
        return None


if __name__ == "__main__":
    unittest.main(verbosity=2)
