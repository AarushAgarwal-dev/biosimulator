"""Target evaluation must grade the model the researcher actually asked about.

Regression cover for a measured defect: for the Turing PDE preset, POST /api/evaluate
reported "Steady state 0.47 met requirements" while the field POST /api/simulate
returned for the SAME blueprint spanned 0.028 to 2.89 over a 50x50 grid. The cause was
that evaluate_targets_on_blueprint compiled EVERY blueprint with ODEModel, which reads
only nodes/edges/odes and never blueprint["spatial"] - so a spatial model was graded on
an unrelated zero-dimensional influence graph's final value, under the researcher's
target name.

These tests pin three properties:
  * a PDE target is graded on the actual spatial solution, with the reduction named;
  * a target type with no defensible field-level meaning is REFUSED, not proxied;
  * a tolerance so wide the target cannot fail is reported as unfalsifiable.
"""

import unittest

import numpy as np
from fastapi.testclient import TestClient

import agent
from main import app
from simulation_engine import ODEModel

client = TestClient(app)


def _pure_diffusion_blueprint(base: float = 2.0, t_max: float = 20.0):
    """A PDE whose FIELD and whose nodes/edges ODE give very different answers.

    Field: pure diffusion under no-flux boundaries conserves mass, so the spatial mean
    stays at the initial mean (~base) for all time.
    ODE path (what the defect graded): no edges, so the compiler's default gives
    dU/dt = -0.1*U, which decays well away from base by t_max.
    """
    return {
        "type": "PDE",
        "nodes": [{"id": "U", "initial_value": base}],
        "edges": [],
        "spatial": {
            "x_grid": 12, "y_grid": 12, "dx": 1.0, "dy": 1.0,
            "diffusion": {"U": 0.2},
            "reactions": {"U": "0"},
        },
        "simulation_config": {"t_max": t_max, "dt": 0.1},
    }


def _turing_blueprint(n: int = 20, t_max: float = 40.0):
    """The shipped Turing (activator-inhibitor) preset, on a smaller grid for speed."""
    return {
        "type": "PDE",
        "nodes": [{"id": "U", "initial_value": 1.0}, {"id": "V", "initial_value": 1.0}],
        "edges": [
            {"source": "U", "target": "U", "type": "activation"},
            {"source": "U", "target": "V", "type": "activation"},
            {"source": "V", "target": "U", "type": "inhibition"},
        ],
        "spatial": {
            "x_grid": n, "y_grid": n, "dx": 1.0, "dy": 1.0,
            "diffusion": {"U": 0.05, "V": 1.0},
            "reactions": {"U": "U**2 / V - U + 0.02", "V": "U**2 - V"},
        },
        "simulation_config": {"t_max": t_max, "dt": 0.1},
    }


def _final_frame(pde_result, species):
    return np.asarray(pde_result["species"][species], dtype=float)[-1]


class PdeTargetsGradeTheField(unittest.TestCase):
    """A spatial model's target is graded on its spatial solution."""

    def test_steady_state_uses_the_field_spatial_mean_not_an_ode_scalar(self):
        bp = _pure_diffusion_blueprint(base=2.0, t_max=20.0)
        target = {"species": "U", "type": "steady_state", "value": 2.0, "tolerance": 0.05}

        # Ground truth, computed independently of the evaluator.
        field_mean = float(_final_frame(agent.simulate_pde_blueprint(bp, seed=0), "U").mean())
        # The number the defect reported: the nodes/edges ODE's final value.
        ode_final = float(ODEModel(bp).simulate(20.0, num_points=200)["species"]["U"][-1])

        met, results = agent.evaluate_targets_on_blueprint(bp, [target])

        self.assertEqual(met, 1)
        result = results[0]
        self.assertTrue(result["spatial"])
        self.assertFalse(result["refused"])
        self.assertIn("spatial mean", result["observable"])
        self.assertAlmostEqual(result["value"], field_mean, places=9)
        self.assertAlmostEqual(field_mean, 2.0, delta=0.02)   # no-flux diffusion conserves mass
        # The regression itself: the graded number is the field's, not the 0-D ODE's.
        self.assertLess(ode_final, 0.5)
        self.assertGreater(abs(result["value"] - ode_final), 1.0)
        self.assertIn("12x12", result["detail"])

    def test_turing_preset_is_graded_on_its_field_and_is_reproducible(self):
        bp = _turing_blueprint()
        target = {"species": "U", "type": "steady_state", "value": 1.0, "tolerance": 10.0}

        first = agent.simulate_pde_blueprint(bp, seed=0)
        second = agent.simulate_pde_blueprint(bp, seed=0)
        np.testing.assert_allclose(_final_frame(first, "U"), _final_frame(second, "U"))

        met, results = agent.evaluate_targets_on_blueprint(bp, [target])
        final = _final_frame(first, "U")
        self.assertEqual(met, 1)                       # tolerance 10 accepts anything...
        self.assertIsNotNone(results[0]["warning"])     # ...which is exactly the warning
        self.assertAlmostEqual(results[0]["value"], float(final.mean()), places=9)
        # The field is genuinely patterned, so the mean is not the whole story and the
        # report has to say so rather than presenting one number as the field's state.
        self.assertGreater(float(final.std()), 0.05)
        self.assertIn("spatial std", results[0]["detail"])

    def test_a_field_target_can_actually_fail(self):
        bp = _pure_diffusion_blueprint(base=2.0, t_max=20.0)
        met, results = agent.evaluate_targets_on_blueprint(
            bp, [{"species": "U", "type": "steady_state", "value": 0.3, "tolerance": 0.05}])
        self.assertEqual(met, 0)
        self.assertFalse(results[0]["met"])
        self.assertFalse(results[0]["refused"])
        self.assertIn("outside the target", results[0]["detail"])

    def test_a_target_on_a_species_with_no_field_is_refused(self):
        bp = _pure_diffusion_blueprint()
        met, results = agent.evaluate_targets_on_blueprint(
            bp, [{"species": "GHOST", "type": "steady_state", "value": 1.0, "tolerance": 0.1}])
        self.assertEqual(met, 0)
        self.assertTrue(results[0]["refused"])
        self.assertIn("GHOST", results[0]["detail"])


class SpatialReductionsAreExact(unittest.TestCase):
    """Each supported reduction measures exactly what it says it measures."""

    def setUp(self):
        # 3 frames on a 4x5 grid. The global maximum (9.0) is in an EARLIER frame, so a
        # peak_value reduction cannot pass by accidentally reading the final frame.
        # The final frame has an exact spatial mean of 3.0 and an exact std of 1.0.
        first = np.full((4, 5), 1.0)
        middle = np.full((4, 5), 9.0)
        last = np.empty((4, 5))
        last[:2] = 2.0
        last[2:] = 4.0
        self.field = np.stack([first, middle, last])

    def test_known_mean_field_grades_correctly(self):
        value, description = agent.reduce_spatial_field(self.field, "steady_state", "U")
        self.assertAlmostEqual(value, 3.0, places=12)
        self.assertIn("final frame", description)

        metric = agent.TargetMetric({"species": "U", "type": "steady_state",
                                     "value": 3.0, "tolerance": 0.1})
        ok, detail = agent._grade_scalar_against_target(metric, value)
        self.assertTrue(ok)
        self.assertIn("within the target", detail)

        wrong = agent.TargetMetric({"species": "U", "type": "steady_state",
                                    "value": 5.0, "tolerance": 0.1})
        ok, detail = agent._grade_scalar_against_target(wrong, value)
        self.assertFalse(ok)
        self.assertIn("outside the target", detail)

    def test_peak_value_is_the_maximum_over_space_and_time(self):
        value, _ = agent.reduce_spatial_field(self.field, "peak_value", "U")
        self.assertAlmostEqual(value, 9.0, places=12)

    def test_spatial_variance_is_the_final_frame_spread(self):
        value, _ = agent.reduce_spatial_field(self.field, "spatial_variance", "U")
        self.assertAlmostEqual(value, 1.0, places=12)

    def test_decay_ratio_is_bulk_final_over_bulk_peak(self):
        value, description = agent.reduce_spatial_field(self.field, "decay_ratio", "U")
        self.assertAlmostEqual(value, 3.0 / 9.0, places=12)
        self.assertIn("spatial mean", description)

    def test_a_diverged_field_is_refused_rather_than_graded(self):
        broken = self.field.copy()
        broken[-1, 0, 0] = np.nan
        with self.assertRaises(agent.SpatialTargetError) as ctx:
            agent.reduce_spatial_field(broken, "steady_state", "U")
        self.assertIn("non-finite", str(ctx.exception))


class MeaninglessSpatialTargetsAreRefused(unittest.TestCase):
    """A target type with no single field-level number is refused by name."""

    def test_peak_time_on_a_whole_field_is_refused_with_a_reason(self):
        bp = _pure_diffusion_blueprint()
        met, results = agent.evaluate_targets_on_blueprint(
            bp, [{"species": "U", "type": "peak_time", "min": 1.0, "max": 5.0}])

        self.assertEqual(met, 0)
        result = results[0]
        self.assertTrue(result["refused"])
        self.assertFalse(result["met"])
        self.assertIn("peak_time", result["detail"])
        self.assertIn("U", result["detail"])
        self.assertIn("no single peak time", result["detail"])
        # Nothing was measured, so no number is reported under the target's name.
        self.assertNotIn("value", result)
        self.assertNotIn("observable", result)

    def test_oscillation_on_a_whole_field_is_refused(self):
        with self.assertRaises(agent.SpatialTargetError) as ctx:
            agent.reduce_spatial_field(np.ones((3, 4, 4)), "oscillation", "U")
        message = str(ctx.exception)
        self.assertIn("oscillation", message)
        self.assertIn("travelling wave", message)

    def test_an_unknown_spatial_target_type_lists_what_is_supported(self):
        with self.assertRaises(agent.SpatialTargetError) as ctx:
            agent.reduce_spatial_field(np.ones((3, 4, 4)), "vibes", "U")
        message = str(ctx.exception)
        self.assertIn("vibes", message)
        self.assertIn("steady_state", message)

    def test_multi_condition_types_are_refused_for_fields(self):
        for metric_type in ("bistability", "fold_change"):
            with self.assertRaises(agent.SpatialTargetError):
                agent.reduce_spatial_field(np.ones((3, 4, 4)), metric_type, "U")


class UnfalsifiableTargetsAreReported(unittest.TestCase):
    """A target that cannot fail is surfaced, not silently rewritten."""

    SHIPPED = {"species": "U", "type": "steady_state", "value": 1.0, "tolerance": 10.0}

    def test_shipped_turing_tolerance_is_flagged(self):
        warning = agent.target_falsifiability_warning(self.SHIPPED)
        self.assertIsNotNone(warning)
        self.assertIn("UNFALSIFIABLE", warning)
        self.assertIn("[-9, 11]", warning)
        self.assertIn("steady_state", warning)

    def test_a_tight_tolerance_is_not_flagged(self):
        self.assertIsNone(agent.target_falsifiability_warning(
            {"species": "U", "type": "steady_state", "value": 1.0, "tolerance": 0.1}))

    def test_a_target_with_no_bounds_at_all_is_flagged(self):
        warning = agent.target_falsifiability_warning({"species": "U", "type": "peak_value"})
        self.assertIsNotNone(warning)
        self.assertIn("neither a min nor a max", warning)

    def test_a_decay_ratio_cap_of_one_is_flagged(self):
        warning = agent.target_falsifiability_warning(
            {"species": "U", "type": "decay_ratio", "max": 1.0})
        self.assertIsNotNone(warning)
        self.assertIn("UNFALSIFIABLE", warning)

    def test_the_warning_reaches_the_evaluation_payload(self):
        bp = _pure_diffusion_blueprint(base=2.0, t_max=5.0)
        target = dict(self.SHIPPED)
        response = client.post("/api/evaluate", json={"blueprint": bp, "targets": [target]})
        self.assertEqual(response.status_code, 200)
        payload = response.json()

        self.assertEqual(payload["graded_on"], "the spatial PDE field")
        self.assertEqual(len(payload["warnings"]), 1)
        self.assertIn("UNFALSIFIABLE", payload["warnings"][0])
        self.assertEqual(payload["refused_count"], 0)
        # The shipped tolerance is reported, never modified behind the researcher's back.
        self.assertEqual(target["tolerance"], 10.0)

    def test_refused_targets_are_counted_and_never_credited(self):
        bp = _pure_diffusion_blueprint(base=2.0, t_max=5.0)
        response = client.post("/api/evaluate", json={
            "blueprint": bp,
            "targets": [{"species": "U", "type": "peak_time", "min": 1.0, "max": 2.0}],
        })
        payload = response.json()
        self.assertEqual(payload["met_count"], 0)
        self.assertEqual(payload["refused_count"], 1)


class OdeEvaluationStillWorks(unittest.TestCase):
    """The non-spatial path is unchanged apart from the new honesty fields."""

    ODE_BP = {
        "type": "ODE",
        "nodes": [{"id": "A", "initial_value": 1.0}],
        "edges": [],
        "simulation_config": {"t_max": 10.0},
    }

    def test_ode_steady_state_refuses_a_trajectory_still_moving(self):
        """A(10) = 0.368 on a pure decay is NOT a steady state -- the real one is 0.

        This test previously asserted met == 1 here, which enshrined the defect a fresh
        review later measured: `steady_state` read y[-1] with no stationarity test, so the
        instantaneous value of a trajectory still travelling was reported as its steady
        state. Its sharpest demonstration was dX/dt = 0.002*(10 - X): X(100) = 1.8127 was
        reported as meeting a target of 1.8 +/- 0.1 while the true fixed point is 10, with
        X(1000) = 8.6466.

        A(t) = exp(-0.1 t) is the same shape. At t = 10 it is 0.3679 and still falling
        toward zero, so the honest answer is not "met" and not "failed" but that the run
        cannot answer the question -- extend t_max. The ODE path is still exercised; it is
        the verdict that changed.
        """
        met, results = agent.evaluate_targets_on_blueprint(
            self.ODE_BP, [{"species": "A", "type": "steady_state",
                           "value": 0.368, "tolerance": 0.02}])
        self.assertEqual(met, 0)
        self.assertTrue(results[0]["refused"],
                        f"an unsettled trajectory should be refused, not graded: "
                        f"{results[0]['detail']}")
        self.assertIn("settled", results[0]["detail"].lower())

    def test_ode_steady_state_is_graded_once_it_has_settled(self):
        """The gate must not refuse a genuine steady state -- only an unsettled one."""
        settled = {
            "type": "ODE",
            "nodes": [{"id": "A", "initial_value": 0.0}],
            "parameters": {"r": 0.5, "Amax": 2.0},
            "odes": {"A": "r*(Amax - A)"},
            "simulation_config": {"t_max": 60.0},
        }
        met, results = agent.evaluate_targets_on_blueprint(
            settled, [{"species": "A", "type": "steady_state",
                       "value": 2.0, "tolerance": 0.05}])
        self.assertEqual(met, 1, f"a settled trajectory was not graded: "
                                 f"{results[0]['detail']}")
        self.assertFalse(results[0]["refused"])

    def test_an_unknown_ode_target_type_is_refused_not_credited(self):
        met, results = agent.evaluate_targets_on_blueprint(
            self.ODE_BP, [{"species": "A", "type": "vibes"}])
        self.assertEqual(met, 0)
        self.assertTrue(results[0]["refused"])
        self.assertIn("vibes", results[0]["detail"])


class SimulateAndEvaluateSeeTheSameField(unittest.TestCase):
    """The displayed field and the graded field must be the same field."""

    def test_api_simulate_and_the_evaluator_agree(self):
        bp = _pure_diffusion_blueprint(base=2.0, t_max=5.0)
        response = client.post("/api/simulate", json={"blueprint": bp})
        self.assertEqual(response.status_code, 200)
        shown = np.asarray(response.json()["species"]["U"], dtype=float)[-1]

        met, results = agent.evaluate_targets_on_blueprint(
            bp, [{"species": "U", "type": "steady_state", "value": 2.0, "tolerance": 0.05}])
        self.assertEqual(met, 1)
        self.assertAlmostEqual(results[0]["value"], float(shown.mean()), places=9)

    def test_a_spatial_blueprint_with_no_reactions_is_a_400_not_a_500(self):
        bp = _pure_diffusion_blueprint()
        bp["spatial"]["reactions"] = {}
        response = client.post("/api/simulate", json={"blueprint": bp})
        self.assertEqual(response.status_code, 400)
        self.assertIn("no spatial reactions", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()
