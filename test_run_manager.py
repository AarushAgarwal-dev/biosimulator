"""Tests for run management and the run state machine."""

import time
import unittest
from unittest.mock import patch

import approach_base as base
import approach_cc3d as cc3d_mod
import geometry as geo
import run_manager as rm


def _abm_project(**overrides):
    config = {
        "grid": {"width": 24, "height": 24},
        "temperature": 10.0,
        "num_mcs": 4,
        "save_every": 2,
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


def _mpc_project(**overrides):
    config = {"target": 1.0, "duration": 6.0, "control_interval": 0.5,
              "prediction_horizon": 8, "control_horizon": 2}
    config.update(overrides)
    return {"approaches": {"mpc": config}, "selected_approach": "mpc"}


class TransitionTableTests(unittest.TestCase):
    def test_terminal_states_have_no_exits(self):
        for state in ("completed", "failed", "cancelled"):
            self.assertEqual(rm.TRANSITIONS[state], ())

    def test_compiling_cannot_reach_ready_except_deliberately(self):
        self.assertIn("ready", rm.TRANSITIONS["compiling"])
        self.assertIn("failed", rm.TRANSITIONS["compiling"])

    def test_illegal_transition_is_refused(self):
        record = rm.RunRecord("r1", "mpc", {})
        record.transition("compiling")
        record.transition("failed")
        with self.assertRaises(rm.RunStateError):
            record.transition("ready")

    def test_failed_is_terminal(self):
        record = rm.RunRecord("r2", "mpc", {})
        record.transition("compiling")
        record.transition("failed")
        for target in ("running", "completed", "ready", "queued"):
            with self.assertRaises(rm.RunStateError, msg=target):
                record.transition(target)


class SnapshotTests(unittest.TestCase):
    def test_configuration_snapshot_is_immutable_against_later_edits(self):
        project = _mpc_project()
        manager = rm.RunManager()
        record = manager.create_run(project)
        project["approaches"]["mpc"]["target"] = 999.0
        self.assertEqual(record.snapshot["approaches"]["mpc"]["target"], 1.0)

    def test_engine_version_is_recorded(self):
        manager = rm.RunManager()
        record = manager.create_run(_mpc_project())
        self.assertEqual(record.engine["approach"], "mpc")
        self.assertTrue(record.engine["engine_name"])


class ValidationBeforeCompilationTests(unittest.TestCase):
    def test_invalid_configuration_never_becomes_ready(self):
        manager = rm.RunManager()
        record = manager.create_run(_mpc_project(prediction_horizon=0))
        self.assertEqual(record.state, "invalid")
        self.assertEqual(record.error_kind, "invalid")
        self.assertTrue(record.issues)
        self.assertIsNone(record.compiled)

    def test_invalid_run_cannot_be_started(self):
        manager = rm.RunManager()
        record = manager.create_run(_mpc_project(control_horizon=99))
        with self.assertRaises(rm.RunStateError):
            manager.start(record.run_id)

    def test_failed_compilation_reports_failed_not_ready(self):
        manager = rm.RunManager()
        project = _mpc_project()
        with patch.object(base.get_approach("mpc"), "compile",
                          side_effect=RuntimeError("synthetic compile failure")):
            record = manager.create_run(project)
        self.assertEqual(record.state, "failed")
        self.assertNotEqual(record.state, "ready")
        self.assertEqual(record.error_kind, "compile_error")
        self.assertIn("synthetic compile failure", record.error)

    def test_missing_approach_selection_is_refused(self):
        manager = rm.RunManager()
        with self.assertRaises(ValueError):
            manager.create_run({"approaches": {}})

    def test_unknown_approach_is_refused(self):
        manager = rm.RunManager()
        with self.assertRaises(KeyError):
            manager.create_run({"approaches": {}}, approach_id="quantum")


class SuccessfulRunTests(unittest.TestCase):
    def test_mpc_run_completes_and_exposes_results(self):
        manager = rm.RunManager()
        record = manager.create_run(_mpc_project())
        self.assertEqual(record.state, "ready")
        manager.start(record.run_id)
        manager.wait(record.run_id, timeout=60)
        self.assertEqual(record.state, "completed")
        self.assertAlmostEqual(record.progress, 1.0)
        results = manager.results(record.run_id)
        self.assertIn("series", results)
        self.assertTrue(record.engine["engine_version"])

    def test_abm_run_completes(self):
        manager = rm.RunManager()
        record = manager.create_run(_abm_project())
        self.assertEqual(record.state, "ready")
        manager.start(record.run_id)
        manager.wait(record.run_id, timeout=120)
        self.assertEqual(record.state, "completed", msg=record.error or "")
        self.assertGreater(len(manager.results(record.run_id)["t"]), 1)

    def test_results_are_refused_before_completion(self):
        manager = rm.RunManager()
        record = manager.create_run(_mpc_project())
        with self.assertRaises(rm.RunStateError):
            manager.results(record.run_id)

    def test_history_records_every_state(self):
        manager = rm.RunManager()
        record = manager.create_run(_mpc_project())
        manager.start(record.run_id)
        manager.wait(record.run_id, timeout=60)
        states = [entry["state"] for entry in record.history]
        for expected in ("draft", "compiling", "ready", "queued", "running", "completed"):
            self.assertIn(expected, states)


class FailedRunTests(unittest.TestCase):
    def test_failed_run_never_reports_completed(self):
        manager = rm.RunManager()
        record = manager.create_run(_mpc_project())
        with patch.object(base.get_approach("mpc"), "run",
                          side_effect=RuntimeError("synthetic run failure")):
            manager.start(record.run_id)
            manager.wait(record.run_id, timeout=60)
        self.assertEqual(record.state, "failed")
        self.assertNotEqual(record.state, "completed")
        self.assertIsNone(record.results)
        self.assertEqual(record.error_kind, "run_error")

    def test_unavailable_approach_fails_with_the_reason(self):
        manager = rm.RunManager()
        project = {"domain": geo.make_rectangle(40.0, 40.0),
                   "selected_approach": "cc3d",
                   "approaches": {"cc3d": {
                       "lattice": {"x": 40, "y": 40, "z": 1}, "steps": 10,
                       "temperature": 10.0,
                       "cell_types": [{"type_id": 1, "name": "A"}]}}}
        with patch.object(cc3d_mod, "detect_cc3d",
                          return_value={"available": False, "method": "", "version": "",
                                        "path": "", "reason": "CompuCell3D is not installed."}):
            record = manager.create_run(project)
        self.assertEqual(record.state, "failed")
        self.assertEqual(record.error_kind, "unavailable")
        self.assertIn("not installed", record.error)
        # And it must not have quietly become a different approach.
        self.assertEqual(record.approach_id, "cc3d")


class ControlTests(unittest.TestCase):
    def test_cancel_before_start(self):
        manager = rm.RunManager()
        record = manager.create_run(_mpc_project())
        manager.cancel(record.run_id)
        self.assertEqual(record.state, "cancelled")
        with self.assertRaises(rm.RunStateError):
            manager.start(record.run_id)

    def test_cancel_during_run(self):
        manager = rm.RunManager()
        record = manager.create_run(_mpc_project(duration=400.0, control_interval=0.5))
        manager.start(record.run_id)
        deadline = time.time() + 10
        while record.state != "running" and time.time() < deadline:
            time.sleep(0.01)
        manager.cancel(record.run_id)
        manager.wait(record.run_id, timeout=60)
        self.assertEqual(record.state, "cancelled")
        self.assertIsNone(record.results)

    def test_pause_then_resume_then_complete(self):
        manager = rm.RunManager()
        record = manager.create_run(_mpc_project(duration=40.0, control_interval=0.5))
        manager.start(record.run_id)
        deadline = time.time() + 10
        while record.state != "running" and time.time() < deadline:
            time.sleep(0.01)
        manager.pause(record.run_id)
        self.assertEqual(record.state, "paused")
        time.sleep(0.15)
        manager.resume(record.run_id)
        manager.wait(record.run_id, timeout=120)
        self.assertEqual(record.state, "completed")

    def test_cancel_while_paused_is_honoured(self):
        manager = rm.RunManager()
        record = manager.create_run(_mpc_project(duration=400.0, control_interval=0.5))
        manager.start(record.run_id)
        deadline = time.time() + 10
        while record.state != "running" and time.time() < deadline:
            time.sleep(0.01)
        manager.pause(record.run_id)
        manager.cancel(record.run_id)
        manager.wait(record.run_id, timeout=60)
        self.assertEqual(record.state, "cancelled")

    def test_pause_is_refused_when_unsupported(self):
        manager = rm.RunManager()
        record = manager.create_run(_mpc_project())
        with patch.object(base.get_approach("mpc"), "get_capabilities") as capabilities:
            capabilities.return_value = base.Capabilities(
                approach_id="mpc", label="MPC", supports_pause=False)
            with self.assertRaises(rm.RunStateError):
                manager.pause(record.run_id)

    def test_cancel_after_completion_is_refused(self):
        manager = rm.RunManager()
        record = manager.create_run(_mpc_project())
        manager.start(record.run_id)
        manager.wait(record.run_id, timeout=60)
        with self.assertRaises(rm.RunStateError):
            manager.cancel(record.run_id)


class BookkeepingTests(unittest.TestCase):
    def test_logs_are_captured_and_capped(self):
        manager = rm.RunManager()
        record = manager.create_run(_mpc_project())
        manager.start(record.run_id)
        manager.wait(record.run_id, timeout=60)
        self.assertTrue(manager.logs(record.run_id))
        for _ in range(rm.MAX_LOG_LINES + 500):
            record.log("filler")
        self.assertLessEqual(len(record.logs), rm.MAX_LOG_LINES)

    def test_listing_and_deleting(self):
        manager = rm.RunManager()
        first = manager.create_run(_mpc_project())
        second = manager.create_run(_mpc_project())
        listed = [entry["run_id"] for entry in manager.list_runs()]
        self.assertIn(first.run_id, listed)
        self.assertIn(second.run_id, listed)
        manager.cancel(first.run_id)
        manager.delete(first.run_id)
        with self.assertRaises(KeyError):
            manager.get(first.run_id)

    def test_running_run_cannot_be_deleted(self):
        manager = rm.RunManager()
        record = manager.create_run(_mpc_project(duration=200.0))
        manager.start(record.run_id)
        deadline = time.time() + 10
        while record.state not in ("running", "queued") and time.time() < deadline:
            time.sleep(0.01)
        with self.assertRaises(rm.RunStateError):
            manager.delete(record.run_id)
        manager.cancel(record.run_id)
        manager.wait(record.run_id, timeout=60)

    def test_status_payload_is_serialisable_without_results(self):
        manager = rm.RunManager()
        record = manager.create_run(_mpc_project())
        payload = manager.status(record.run_id)
        self.assertNotIn("results", payload)
        self.assertIn("state", payload)
        self.assertIn("issues", payload)
        import json
        json.dumps(payload)          # raises if a non-serialisable value slipped in


if __name__ == "__main__":
    unittest.main()
