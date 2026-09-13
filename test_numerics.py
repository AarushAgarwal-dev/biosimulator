"""Scientific validation of the simulation numerics.

These are not smoke tests: each one checks a property the mathematics guarantees,
so a regression in the integrator shows up as a violated conservation law or a
broken symmetry rather than as a plausible-looking wrong picture.

Covers the checks required of the solver stack:
  * a no-flux diffusion problem conserves mass
  * a symmetric problem stays symmetric
  * an unstable explicit time step is refused instead of silently diverging
  * the run ends at the requested time
  * divergence is reported rather than clamped away
  * ODE integration matches a closed-form solution
"""

import math
import unittest

import numpy as np

from simulation_engine import ODEModel, solve_pde


def _uniform_ic(name, value):
    return {name: {"type": "uniform", "base_value": value}}


def _square_grid(n=15, dx=1.0, diffusion=0.2, species="u"):
    return {
        "x_grid": n, "y_grid": n, "dx": dx, "dy": dx,
        "diffusion": {species: diffusion},
    }


class MassConservationTests(unittest.TestCase):
    """Zero-flux boundaries plus no reaction => the discrete integral is invariant.

    This could not be tested before: the loop clamped every field at zero on every
    step, which is a non-conservative operation, and no mass series was reported.
    """

    def test_no_flux_diffusion_conserves_mass(self):
        n = 15
        result = solve_pde(
            _square_grid(n=n, diffusion=0.25),
            {"u": "0"},
            {"u": {"type": "central_spot", "base_value": 0.0,
                   "spot_radius": 3, "spot_value": 5.0}},
            t_max=4.0, dt=0.2, save_every=1,
        )
        mass = result["mass"]["u"]
        self.assertGreater(len(mass), 5)
        initial = mass[0]
        self.assertGreater(initial, 0.0, "initial mass should be non-zero")
        for step, value in enumerate(mass):
            self.assertAlmostEqual(
                value / initial, 1.0, places=9,
                msg=f"mass drifted at frame {step}: {value} vs {initial}",
            )

    def test_reported_mass_matches_the_field(self):
        result = solve_pde(
            _square_grid(n=9, diffusion=0.1), {"u": "0"},
            _uniform_ic("u", 2.0), t_max=1.0, dt=0.25, save_every=1,
        )
        dx = 1.0
        for frame, reported in zip(result["species"]["u"], result["mass"]["u"]):
            expected = float(np.sum(np.asarray(frame, dtype=float)) * dx * dx)
            self.assertAlmostEqual(reported, expected, places=9)

    def test_diffusion_spreads_without_creating_material(self):
        """The peak must fall while the total is unchanged."""
        result = solve_pde(
            _square_grid(n=21, diffusion=0.25), {"u": "0"},
            {"u": {"type": "central_spot", "base_value": 0.0,
                   "spot_radius": 2, "spot_value": 10.0}},
            t_max=6.0, dt=0.2, save_every=5,
        )
        first = np.asarray(result["species"]["u"][0], dtype=float)
        last = np.asarray(result["species"]["u"][-1], dtype=float)
        self.assertLess(last.max(), first.max(), "peak should decay as it spreads")
        self.assertAlmostEqual(result["mass"]["u"][-1] / result["mass"]["u"][0], 1.0, places=9)


class SymmetryTests(unittest.TestCase):
    def test_symmetric_initial_condition_stays_symmetric(self):
        n = 21                      # odd, so the centre lands on a node
        result = solve_pde(
            _square_grid(n=n, diffusion=0.2), {"u": "0"},
            {"u": {"type": "central_spot", "base_value": 0.1,
                   "spot_radius": 4, "spot_value": 3.0}},
            t_max=5.0, dt=0.2, save_every=5,
        )
        final = np.asarray(result["species"]["u"][-1], dtype=float)
        np.testing.assert_allclose(final, final[::-1, :], rtol=0, atol=1e-12)
        np.testing.assert_allclose(final, final[:, ::-1], rtol=0, atol=1e-12)
        # A symmetric diffusion problem is also transpose-symmetric on a square grid.
        np.testing.assert_allclose(final, final.T, rtol=0, atol=1e-12)


class StabilityTests(unittest.TestCase):
    def test_unstable_time_step_is_refused(self):
        # dx=dy=1 => inv_h2 = 2, so the limit is dt <= 0.5/(2*D) = 0.25/D.
        with self.assertRaises(ValueError) as raised:
            solve_pde(
                _square_grid(n=9, diffusion=1.0), {"u": "0"},
                _uniform_ic("u", 1.0), t_max=1.0, dt=0.5, save_every=1,
            )
        message = str(raised.exception)
        self.assertIn("unstable", message.lower())
        self.assertIn("0.25", message, "the message should state the largest stable dt")

    def test_stable_time_step_is_accepted_and_reported(self):
        result = solve_pde(
            _square_grid(n=9, diffusion=1.0), {"u": "0"},
            _uniform_ic("u", 1.0), t_max=1.0, dt=0.2, save_every=1,
        )
        stability = result["stability"]
        self.assertAlmostEqual(stability["diffusion_number"], 0.4, places=12)
        self.assertAlmostEqual(stability["max_stable_dt"], 0.25, places=12)
        self.assertIsNone(stability["diverged_at"])

    def test_divergence_is_reported_not_hidden(self):
        """With the check bypassed, a diverging run must SAY it diverged.

        The previous implementation clamped at zero every step, so a diverging
        field came back finite, non-negative and indistinguishable from a result.
        """
        result = solve_pde(
            _square_grid(n=9, diffusion=5.0), {"u": "0"},
            {"u": {"type": "random_noise", "base_value": 1.0, "noise_amplitude": 0.5}},
            t_max=50.0, dt=2.0, save_every=1, strict_stability=False,
        )
        stability = result["stability"]
        diverged = stability["diverged_at"] is not None
        final = np.asarray(result["species"]["u"][-1], dtype=float)
        blew_up = (not np.all(np.isfinite(final))) or np.abs(final).max() > 1e6
        self.assertTrue(diverged or blew_up,
                        "an unstable run must be visibly wrong, not silently clamped")

    def test_negative_values_are_counted_and_not_clamped_by_default(self):
        # A pure sink drives the field negative; the honest scheme lets it happen
        # and records it, rather than flooring it at zero.
        result = solve_pde(
            _square_grid(n=7, diffusion=0.0), {"u": "-1.0"},
            _uniform_ic("u", 0.5), t_max=1.0, dt=0.25, save_every=1,
        )
        final = np.asarray(result["species"]["u"][-1], dtype=float)
        self.assertLess(final.min(), 0.0, "the sink should carry the field negative")
        self.assertGreater(result["stability"]["negative_value_steps"], 0)
        self.assertFalse(result["stability"]["clamped_negatives"])

    def test_clamping_is_available_but_declared(self):
        result = solve_pde(
            _square_grid(n=7, diffusion=0.0), {"u": "-1.0"},
            _uniform_ic("u", 0.5), t_max=1.0, dt=0.25, save_every=1,
            clamp_negative=True,
        )
        final = np.asarray(result["species"]["u"][-1], dtype=float)
        self.assertGreaterEqual(final.min(), 0.0)
        self.assertTrue(result["stability"]["clamped_negatives"])


class TimeGridTests(unittest.TestCase):
    def test_final_time_is_reached_exactly(self):
        # 1.0 is not a whole multiple of 0.3; int(t_max/dt) used to stop at 0.9
        # and report that as a completed run.
        result = solve_pde(
            _square_grid(n=7, diffusion=0.1), {"u": "0"},
            _uniform_ic("u", 1.0), t_max=1.0, dt=0.3, save_every=1,
        )
        self.assertAlmostEqual(result["t"][-1], 1.0, places=12)
        self.assertAlmostEqual(result["stability"]["end_time"], 1.0, places=12)

    def test_divisible_horizon_is_unchanged(self):
        result = solve_pde(
            _square_grid(n=7, diffusion=0.1), {"u": "0"},
            _uniform_ic("u", 1.0), t_max=5.0, dt=0.1, save_every=10,
        )
        self.assertAlmostEqual(result["t"][-1], 5.0, places=12)
        self.assertEqual(len(result["t"]), 6)      # initial + 5 saved frames

    def test_long_run_time_does_not_drift(self):
        """2000 steps of accumulated dt reported 199.99999999999292 for t_max=200.

        Time stamps are derived from the step index instead, so the error cannot
        grow with the step count and the last frame lands exactly on t_max.
        """
        result = solve_pde(
            _square_grid(n=5, diffusion=0.1), {"u": "0"},
            _uniform_ic("u", 1.0), t_max=200.0, dt=0.1, save_every=500,
        )
        self.assertEqual(result["t"][-1], 200.0)
        self.assertEqual(result["stability"]["end_time"], 200.0)
        # Every intermediate stamp should also be exact to well under one dt.
        for value in result["t"]:
            self.assertAlmostEqual(value, round(value, 6), places=9)


class ODEAccuracyTests(unittest.TestCase):
    def test_exponential_decay_matches_closed_form(self):
        """dX/dt = -k X has the exact solution X(t) = X0 exp(-k t).

        At SciPy's default rtol=1e-3 this drifts in the 4th digit, which is enough
        to move a fitted rate constant; the explicit tolerances hold it far tighter.
        """
        blueprint = {
            "type": "ODE",
            "nodes": [{"id": "X", "initial_value": 1.0}],
            "parameters": {"k": 0.7},
            "odes": {"X": "-k*X"},
            "simulation_config": {"t_max": 3.0},
        }
        model = ODEModel(blueprint)
        result = model.simulate(t_max=3.0, num_points=31)
        for t, value in zip(result["t"], result["species"]["X"]):
            self.assertAlmostEqual(value, math.exp(-0.7 * t), places=7,
                                   msg=f"drift at t={t}")

    def test_two_species_conversion_conserves_total(self):
        """A -> B with no source or sink keeps A + B constant."""
        blueprint = {
            "type": "ODE",
            "nodes": [{"id": "A", "initial_value": 2.0}, {"id": "B", "initial_value": 0.0}],
            "parameters": {"k": 1.3},
            "odes": {"A": "-k*A", "B": "k*A"},
            "simulation_config": {"t_max": 4.0},
        }
        result = ODEModel(blueprint).simulate(t_max=4.0, num_points=41)
        for a, b in zip(result["species"]["A"], result["species"]["B"]):
            self.assertAlmostEqual(a + b, 2.0, places=7)


if __name__ == "__main__":
    unittest.main()



class SteadyStateDetectionTests(unittest.TestCase):
    """A slowly decaying transient must not be reported as a steady state.

    The old test looked only at the SPREAD of the last 15% of the trajectory, with a
    tolerance that scaled with the value: at y=9.05 it allowed 0.02*9.05+0.01 = 0.191,
    which accepted an observed drift of 0.136 that was monotone decreasing rather than
    noise. That is how a model whose "high branch" decays from 9.05 at t=100 to 0.0026
    at t=10000 was reported as having a second stable state -- the app claimed
    bistability for a system with exactly one attractor.
    """

    def _stable(self, values):
        import agent
        import numpy as np
        return agent._final_stable(np.asarray(values, dtype=float))

    def test_a_genuine_plateau_is_accepted(self):
        # Settled at 5.0 with small non-directional jitter.
        import numpy as np
        rng = np.random.default_rng(7)
        y = 5.0 + rng.normal(0.0, 1e-4, 200)
        self.assertTrue(self._stable(y), "a noisy plateau is a steady state")

    def test_an_exponentially_approached_fixed_point_is_accepted(self):
        # Real fixed points are approached monotonically; a NEGLIGIBLE residual drift
        # must still count as settled or nothing would ever pass.
        import numpy as np
        t = np.linspace(0.0, 50.0, 300)
        y = 5.0 - 3.0 * np.exp(-t)
        self.assertTrue(self._stable(y),
                        "an exponentially settled trajectory is a steady state")

    def test_a_slow_monotone_decay_is_rejected(self):
        # The real failure: large value, small tail spread, but consistently falling.
        import numpy as np
        t = np.linspace(0.0, 100.0, 300)
        y = 9.05 * np.exp(-t / 900.0)          # still decaying at the end
        tail = y[int(0.85 * len(y)):]
        self.assertLess(float(tail.max() - tail.min()),
                        0.02 * abs(float(y[-1])) + 0.01,
                        "precondition: the OLD spread-only test would have passed this")
        self.assertFalse(self._stable(y),
                         "a monotone decaying tail is not a steady state")

    def test_a_diverging_trajectory_is_rejected(self):
        import numpy as np
        t = np.linspace(0.0, 10.0, 200)
        self.assertFalse(self._stable(np.exp(t)))

    def test_non_finite_values_are_rejected(self):
        import numpy as np
        y = np.concatenate([np.full(100, 2.0), [np.nan]])
        self.assertFalse(self._stable(y))
