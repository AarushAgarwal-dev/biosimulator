"""Tests for the MPC approach.

The scientific requirements: input bounds are respected, tracking error decreases,
and optimisation failures are reported rather than hidden.
"""

import unittest

import numpy as np

import approach_base as base
import approach_mpc as mpc


def _project(**overrides):
    config = {
        "target": 1.0,
        "duration": 20.0,
        "control_interval": 0.5,
        "prediction_horizon": 10,
        "control_horizon": 3,
        "input_min": -2.0,
        "input_max": 2.0,
        "input_rate_limit": 0.5,
    }
    config.update(overrides)
    return {"approaches": {"mpc": config}}


def _run(project):
    adapter = mpc.MPC_ADAPTER
    compiled = adapter.compile(project)
    context = base.RunContext(run_id="test")
    return adapter.run(compiled, context)


def _codes(issues):
    return {i.code for i in issues}


class CapabilityTests(unittest.TestCase):
    def test_mpc_is_available_and_named_correctly(self):
        capabilities = mpc.MPC_ADAPTER.get_capabilities()
        self.assertTrue(capabilities.available)
        self.assertEqual(capabilities.approach_id, "mpc")
        self.assertIn("Model Predictive Control", capabilities.label)
        self.assertNotIn("MCP", capabilities.label)

    def test_registered_in_the_shared_registry(self):
        self.assertIn("mpc", base.registered_ids())
        self.assertIs(base.get_approach("mpc"), mpc.MPC_ADAPTER)


class ValidationTests(unittest.TestCase):
    def test_default_configuration_is_valid(self):
        self.assertEqual([i for i in mpc.MPC_ADAPTER.validate(_project())
                          if i.severity == "error"], [])

    def test_control_horizon_cannot_exceed_prediction_horizon(self):
        issues = mpc.MPC_ADAPTER.validate(_project(prediction_horizon=3, control_horizon=9))
        self.assertIn("mpc_control_horizon_too_long", _codes(issues))

    def test_inverted_input_bounds_rejected(self):
        issues = mpc.MPC_ADAPTER.validate(_project(input_min=2.0, input_max=-2.0))
        self.assertIn("mpc_input_bounds_inverted", _codes(issues))

    def test_non_positive_horizon_rejected(self):
        self.assertIn("mpc_prediction_horizon_too_small",
                      _codes(mpc.MPC_ADAPTER.validate(_project(prediction_horizon=0))))

    def test_negative_weight_rejected(self):
        self.assertIn("mpc_weight_tracking_too_small",
                      _codes(mpc.MPC_ADAPTER.validate(_project(weight_tracking=-1.0))))

    def test_unreachable_target_is_warned_not_silently_accepted(self):
        # Steady state is tau*gain*u, so with tau=5, gain=1, u<=2 the ceiling is 10.
        issues = mpc.MPC_ADAPTER.validate(_project(target=500.0))
        self.assertIn("mpc_target_unreachable", _codes(issues))
        self.assertTrue(all(i.severity == "warning" for i in issues
                            if i.code == "mpc_target_unreachable"))

    def test_unsupported_model_kind_rejected(self):
        issues = mpc.MPC_ADAPTER.validate(_project(model={"kind": "neural_net"}))
        self.assertIn("mpc_model_kind_unsupported", _codes(issues))

    def test_compile_refuses_an_invalid_configuration(self):
        with self.assertRaises(ValueError):
            mpc.MPC_ADAPTER.compile(_project(prediction_horizon=0))


class ControlBehaviourTests(unittest.TestCase):
    def test_input_bounds_are_never_violated(self):
        result = _run(_project(input_min=-0.4, input_max=0.4, target=5.0))
        controls = result["series"]["control"]
        self.assertTrue(controls)
        for u in controls:
            self.assertGreaterEqual(u, -0.4 - 1e-9)
            self.assertLessEqual(u, 0.4 + 1e-9)
        self.assertTrue(result["summary"]["input_bounds_respected"])

    def test_rate_limit_is_never_violated(self):
        limit = 0.15
        result = _run(_project(input_rate_limit=limit, target=8.0))
        controls = result["series"]["control"]
        previous = 0.0
        for u in controls:
            self.assertLessEqual(abs(u - previous), limit + 1e-9)
            previous = u
        self.assertTrue(result["summary"]["rate_limit_respected"])

    def test_tracking_error_decreases(self):
        result = _run(_project(target=1.0, duration=40.0))
        summary = result["summary"]
        self.assertTrue(summary["tracking_improved"],
                        f"error did not fall: {summary['initial_mean_abs_error']} -> "
                        f"{summary['final_mean_abs_error']}")
        self.assertLess(summary["final_mean_abs_error"], summary["initial_mean_abs_error"])

    def test_output_approaches_the_target(self):
        result = _run(_project(target=1.0, duration=60.0, input_rate_limit=1.0))
        self.assertLess(abs(result["summary"]["final_abs_error"]), 0.15)

    def test_series_are_aligned_and_complete(self):
        result = _run(_project(duration=10.0, control_interval=0.5))
        length = len(result["t"])
        self.assertEqual(length, 20)
        for name in ("output", "target", "control", "error", "objective"):
            self.assertEqual(len(result["series"][name]), length, name)

    def test_error_series_matches_output_minus_target(self):
        result = _run(_project())
        for output, target, error in zip(result["series"]["output"],
                                         result["series"]["target"],
                                         result["series"]["error"]):
            self.assertAlmostEqual(error, output - target, places=12)

    def test_step_change_in_target_is_tracked(self):
        result = _run(_project(
            target={"times": [0.0, 10.0, 10.001, 40.0], "values": [0.0, 0.0, 1.5, 1.5]},
            duration=40.0, input_rate_limit=1.0))
        outputs = result["series"]["output"]
        times = result["t"]
        early = [o for t, o in zip(times, outputs) if t < 9.0]
        late = [o for t, o in zip(times, outputs) if t > 35.0]
        self.assertLess(abs(np.mean(early)), 0.3)
        self.assertLess(abs(np.mean(late) - 1.5), 0.3)

    def test_run_is_deterministic(self):
        first = _run(_project())
        second = _run(_project())
        self.assertEqual(first["series"]["control"], second["series"]["control"])

    def test_model_mismatch_still_tracks(self):
        """The plant is twice as slow as the model the controller predicts with."""
        result = _run(_project(
            model={"tau": 5.0, "gain": 1.0},
            plant={"tau": 10.0, "gain": 0.7},
            duration=80.0, input_rate_limit=1.0, target=1.0))
        self.assertTrue(result["summary"]["tracking_improved"])
        self.assertLess(abs(result["summary"]["final_abs_error"]), 0.3)

    def test_output_constraint_is_respected(self):
        result = _run(_project(target=3.0, output_max=1.0, duration=40.0,
                               input_rate_limit=1.0))
        self.assertLessEqual(max(result["series"]["output"]), 1.0 + 0.05)

    def test_failures_are_counted_not_hidden(self):
        result = _run(_project())
        self.assertIn("optimisation_failures", result["summary"])
        self.assertIsInstance(result["summary"]["optimisation_failures"], int)


class ControlFlowTests(unittest.TestCase):
    def test_cancellation_raises_and_does_not_report_completion(self):
        adapter = mpc.MPC_ADAPTER
        compiled = adapter.compile(_project(duration=100.0))
        context = base.RunContext(run_id="cancel-me", is_cancelled=lambda: True)
        with self.assertRaises(base.RunCancelled):
            adapter.run(compiled, context)

    def test_progress_and_logs_are_emitted(self):
        adapter = mpc.MPC_ADAPTER
        compiled = adapter.compile(_project(duration=10.0))
        progress, logs = [], []
        context = base.RunContext(
            run_id="observed",
            on_progress=lambda f, m: progress.append(f),
            on_log=lambda m, level: logs.append((level, m)),
        )
        adapter.run(compiled, context)
        self.assertTrue(progress)
        self.assertLessEqual(max(progress), 1.0)
        self.assertTrue(logs)


class ExportTests(unittest.TestCase):
    def test_export_states_the_objective_and_constraints(self):
        exported = mpc.MPC_ADAPTER.export_configuration(_project())
        self.assertEqual(exported["approach"], "mpc")
        self.assertIn("w_track", exported["objective"])
        self.assertEqual(exported["constraints"]["input_bounds"], [-2.0, 2.0])


if __name__ == "__main__":
    unittest.main()
