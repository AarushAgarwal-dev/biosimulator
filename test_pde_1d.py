"""Scientific validation of the 1D solver and boundary conditions.

Key checks required of the stack:
  * fixed-value (Dirichlet) diffusion approaches the analytic LINEAR steady state
  * no-flux diffusion conserves mass
  * Robin exchange drives the field towards the ambient value
  * periodic ends transport material around the domain
  * advection moves a pulse at the right speed, without oscillation
  * an unstable step is refused, with both limits reported
"""

import unittest

import numpy as np

import boundary_conditions as bc
import geometry as geo
import meshing
from pde_solver_1d import solve_1d, steady_state_1d


def _mesh(n=41, length=1.0):
    return np.linspace(0.0, length, n)


def _codes(issues):
    return {i.code for i in issues}


class DirichletSteadyStateTests(unittest.TestCase):
    def test_fixed_ends_reach_the_analytic_linear_profile(self):
        x = _mesh(41, 1.0)
        dx = x[1] - x[0]
        D = 1.0
        dt = 0.4 * dx * dx / (2 * D)
        result = solve_1d(
            x, initial=np.zeros_like(x), diffusion=D, t_max=2.0, dt=dt,
            left=bc.make_dirichlet("u", "left", 0.0),
            right=bc.make_dirichlet("u", "right", 1.0),
            save_every=50,
        )
        final = np.asarray(result["u"][-1], dtype=float)
        expected = steady_state_1d(x, D, 0.0, 1.0)
        # The steady state is exact; the transient decays like exp(-pi^2 D t).
        np.testing.assert_allclose(final, expected, atol=2e-3)

    def test_steady_state_is_actually_linear(self):
        x = _mesh(31, 2.0)
        dx = x[1] - x[0]
        result = solve_1d(x, np.zeros_like(x), diffusion=0.5, t_max=8.0,
                          dt=0.4 * dx * dx / (2 * 0.5),
                          left=bc.make_dirichlet("u", "left", 2.0),
                          right=bc.make_dirichlet("u", "right", 5.0),
                          save_every=100)
        final = np.asarray(result["u"][-1], dtype=float)
        # A straight line has a constant first difference.
        differences = np.diff(final)
        self.assertLess(float(differences.std()), 1e-4)
        self.assertAlmostEqual(final[0], 2.0, places=9)
        self.assertAlmostEqual(final[-1], 5.0, places=9)

    def test_dirichlet_values_are_held_exactly(self):
        x = _mesh(21)
        dx = x[1] - x[0]
        result = solve_1d(x, np.full_like(x, 0.5), diffusion=1.0, t_max=0.5,
                          dt=0.4 * dx * dx / 2.0,
                          left=bc.make_dirichlet("u", "left", -1.0),
                          right=bc.make_dirichlet("u", "right", 3.0))
        for frame in result["u"][1:]:
            self.assertAlmostEqual(frame[0], -1.0, places=12)
            self.assertAlmostEqual(frame[-1], 3.0, places=12)


class NoFluxConservationTests(unittest.TestCase):
    def test_no_flux_diffusion_conserves_mass(self):
        x = _mesh(41)
        dx = x[1] - x[0]
        initial = np.exp(-((x - 0.5) ** 2) / 0.01)
        result = solve_1d(x, initial, diffusion=1.0, t_max=0.2,
                          dt=0.4 * dx * dx / 2.0, save_every=5)
        mass = result["mass"]
        for index, value in enumerate(mass):
            self.assertAlmostEqual(value / mass[0], 1.0, places=6,
                                   msg=f"mass drifted at frame {index}")

    def test_no_flux_relaxes_to_the_mean(self):
        x = _mesh(41)
        dx = x[1] - x[0]
        initial = np.where(x < 0.5, 1.0, 0.0)
        result = solve_1d(x, initial, diffusion=1.0, t_max=3.0,
                          dt=0.4 * dx * dx / 2.0, save_every=100)
        final = np.asarray(result["u"][-1], dtype=float)
        self.assertLess(float(final.std()), 1e-3)
        # The plateau is the CONSERVED mass divided by the length, not 0.5: this step
        # is not symmetric about the midpoint on a node-centred mesh (the node at
        # x=0.5 holds 0), so the trapezoidal mass is 0.4875. Asserting the conserved
        # value is both correct and a stronger statement than a guessed constant.
        length = x[-1] - x[0]
        expected = result["mass"][0] / length
        self.assertAlmostEqual(float(final.mean()), expected, places=4)
        self.assertAlmostEqual(result["mass"][-1] / result["mass"][0], 1.0, places=6)

    def test_inward_neumann_flux_adds_material(self):
        x = _mesh(21)
        dx = x[1] - x[0]
        result = solve_1d(x, np.zeros_like(x), diffusion=1.0, t_max=0.2,
                          dt=0.4 * dx * dx / 2.0,
                          left=bc.make_neumann("u", "left", flux=-1.0),
                          save_every=5)
        self.assertGreater(result["mass"][-1], result["mass"][0])
        self.assertGreater(result["u"][-1][0], 0.0)


class RobinTests(unittest.TestCase):
    def test_robin_pulls_the_field_towards_ambient(self):
        x = _mesh(21)
        dx = x[1] - x[0]
        result = solve_1d(x, np.full_like(x, 2.0), diffusion=1.0, t_max=5.0,
                          dt=0.4 * dx * dx / 2.0,
                          left=bc.make_robin("u", "left", transfer=5.0, ambient=0.0),
                          right=bc.make_robin("u", "right", transfer=5.0, ambient=0.0),
                          save_every=100)
        final = np.asarray(result["u"][-1], dtype=float)
        self.assertLess(float(final.max()), 2.0)
        self.assertGreaterEqual(float(final.min()), -1e-9)
        self.assertLess(result["mass"][-1], result["mass"][0])

    def test_zero_transfer_robin_behaves_like_no_flux(self):
        x = _mesh(21)
        dx = x[1] - x[0]
        common = dict(diffusion=1.0, t_max=0.2, dt=0.4 * dx * dx / 2.0, save_every=5)
        initial = np.exp(-((x - 0.5) ** 2) / 0.02)
        sealed = solve_1d(x, initial, **common)
        robin = solve_1d(x, initial,
                         left=bc.make_robin("u", "left", 0.0, 0.0),
                         right=bc.make_robin("u", "right", 0.0, 0.0), **common)
        np.testing.assert_allclose(sealed["u"][-1], robin["u"][-1], atol=1e-12)


class PeriodicTests(unittest.TestCase):
    def test_periodic_diffusion_conserves_mass(self):
        x = _mesh(41)
        dx = x[1] - x[0]
        initial = np.exp(-((x - 0.3) ** 2) / 0.01)
        result = solve_1d(x, initial, diffusion=1.0, t_max=0.2,
                          dt=0.4 * dx * dx / 2.0, periodic=True, save_every=5)
        for value in result["mass"]:
            self.assertAlmostEqual(value / result["mass"][0], 1.0, places=4)

    def test_periodic_advection_wraps_around(self):
        x = _mesh(41)
        dx = x[1] - x[0]
        initial = np.where(np.abs(x - 0.2) < 0.05, 1.0, 0.0)
        result = solve_1d(x, initial, diffusion=0.0, t_max=0.9, dt=0.5 * dx / 1.0,
                          advection=1.0, periodic=True, save_every=100)
        final = np.asarray(result["u"][-1], dtype=float)
        # After travelling ~0.9 at speed 1 from 0.2, the pulse has wrapped past x=1.
        self.assertGreater(final[:10].max(), 0.1)


class AdvectionTests(unittest.TestCase):
    def test_pulse_travels_in_the_flow_direction(self):
        x = _mesh(81, 2.0)
        dx = x[1] - x[0]
        initial = np.exp(-((x - 0.5) ** 2) / 0.005)
        result = solve_1d(x, initial, diffusion=0.0, t_max=0.5, dt=0.5 * dx / 1.0,
                          advection=1.0, save_every=200)
        final = np.asarray(result["u"][-1], dtype=float)
        centre_start = float(np.sum(x * initial) / np.sum(initial))
        centre_end = float(np.sum(x * final) / max(np.sum(final), 1e-12))
        self.assertGreater(centre_end, centre_start + 0.3)

    def test_upwind_does_not_produce_negative_overshoot(self):
        x = _mesh(81, 2.0)
        dx = x[1] - x[0]
        initial = np.where(np.abs(x - 0.4) < 0.1, 1.0, 0.0)
        result = solve_1d(x, initial, diffusion=0.0, t_max=0.4, dt=0.5 * dx,
                          advection=1.0, save_every=50)
        for frame in result["u"]:
            self.assertGreaterEqual(min(frame), -1e-12,
                                    "upwind differencing must stay monotone")

    def test_numerical_diffusion_is_reported(self):
        x = _mesh(41)
        dx = x[1] - x[0]
        result = solve_1d(x, np.zeros_like(x), diffusion=0.0, t_max=0.1,
                          dt=0.5 * dx, advection=2.0)
        self.assertAlmostEqual(result["stability"]["numerical_diffusion"], 2.0 * dx / 2.0,
                               places=12)


class StabilityTests(unittest.TestCase):
    def test_unstable_diffusive_step_is_refused_with_both_limits(self):
        x = _mesh(21)
        dx = x[1] - x[0]
        with self.assertRaises(ValueError) as raised:
            solve_1d(x, np.zeros_like(x), diffusion=1.0, t_max=1.0, dt=dx * dx)
        message = str(raised.exception)
        self.assertIn("unstable", message.lower())
        self.assertIn("diffusive limit", message)
        self.assertIn("advective limit", message)

    def test_unstable_advective_step_is_refused(self):
        x = _mesh(21)
        dx = x[1] - x[0]
        with self.assertRaises(ValueError):
            solve_1d(x, np.zeros_like(x), diffusion=0.0, t_max=1.0,
                     dt=2.0 * dx, advection=1.0)

    def test_courant_and_diffusion_numbers_are_reported(self):
        x = _mesh(21)
        dx = x[1] - x[0]
        result = solve_1d(x, np.zeros_like(x), diffusion=1.0, t_max=0.1,
                          dt=0.25 * dx * dx / 2.0, advection=0.5)
        stability = result["stability"]
        self.assertLess(stability["diffusion_number"], 0.5)
        self.assertLess(stability["courant_number"], 1.0)
        self.assertIsNone(stability["diverged_at"])

    def test_non_uniform_mesh_is_refused(self):
        x = np.array([0.0, 0.1, 0.35, 1.0])
        with self.assertRaises(ValueError) as raised:
            solve_1d(x, np.zeros_like(x), diffusion=1.0, t_max=0.1, dt=1e-4)
        self.assertIn("uniform", str(raised.exception))

    def test_end_time_is_exact(self):
        x = _mesh(21)
        dx = x[1] - x[0]
        result = solve_1d(x, np.zeros_like(x), diffusion=1.0, t_max=0.1,
                          dt=0.3 * dx * dx / 2.0)
        self.assertAlmostEqual(result["t"][-1], 0.1, places=12)


class ReactionAndSourceTests(unittest.TestCase):
    def test_decay_reaction_matches_closed_form(self):
        x = _mesh(21)
        dx = x[1] - x[0]
        k = 0.8
        result = solve_1d(x, np.ones_like(x), diffusion=0.0, t_max=1.0,
                          dt=0.001, reaction=lambda u, xs, t: -k * u, save_every=1000)
        final = np.asarray(result["u"][-1], dtype=float)
        # Uniform field, no diffusion: every node decays as exp(-k t). Forward Euler
        # at dt=1e-3 is accurate to about dt*k/2 relative.
        np.testing.assert_allclose(final, np.exp(-k * 1.0), rtol=1e-3)

    def test_localised_source_adds_material_where_specified(self):
        x = _mesh(41)
        dx = x[1] - x[0]
        result = solve_1d(
            x, np.zeros_like(x), diffusion=0.1, t_max=0.5, dt=0.4 * dx * dx / (2 * 0.1),
            source=lambda xs, t: np.where(np.abs(xs - 0.25) < 0.05, 1.0, 0.0),
            save_every=20,
        )
        final = np.asarray(result["u"][-1], dtype=float)
        near_source = final[np.abs(x - 0.25) < 0.05].mean()
        far_away = final[np.abs(x - 0.85) < 0.05].mean()
        self.assertGreater(near_source, far_away)
        self.assertGreater(result["mass"][-1], result["mass"][0])


class ConditionValidationTests(unittest.TestCase):
    def setUp(self):
        self.domain = geo.make_interval(1.0)
        self.fields = ["u"]

    def test_valid_conditions_pass(self):
        conditions = [bc.make_dirichlet("u", "left", 0.0),
                      bc.make_neumann("u", "right", 0.0)]
        self.assertEqual(bc.validate_conditions(conditions, self.domain, self.fields), [])

    def test_unknown_boundary_rejected(self):
        conditions = [bc.make_dirichlet("u", "nowhere", 1.0)]
        self.assertIn("condition_boundary_unknown",
                      _codes(bc.validate_conditions(conditions, self.domain, self.fields)))

    def test_unknown_field_rejected(self):
        conditions = [bc.make_dirichlet("ghost", "left", 1.0)]
        self.assertIn("condition_field_unknown",
                      _codes(bc.validate_conditions(conditions, self.domain, self.fields)))

    def test_two_dirichlet_on_one_boundary_is_a_conflict(self):
        conditions = [bc.make_dirichlet("u", "left", 0.0, condition_id="a"),
                      bc.make_dirichlet("u", "left", 1.0, condition_id="b")]
        self.assertIn("condition_conflict_duplicate_value",
                      _codes(bc.validate_conditions(conditions, self.domain, self.fields)))

    def test_value_plus_flux_on_one_boundary_is_a_conflict(self):
        conditions = [bc.make_dirichlet("u", "left", 0.0, condition_id="a"),
                      bc.make_neumann("u", "left", 1.0, condition_id="b")]
        self.assertIn("condition_conflict_value_and_flux",
                      _codes(bc.validate_conditions(conditions, self.domain, self.fields)))

    def test_disabled_condition_does_not_conflict(self):
        conditions = [bc.make_dirichlet("u", "left", 0.0, condition_id="a"),
                      bc.make_neumann("u", "left", 1.0, condition_id="b", enabled=False)]
        self.assertEqual(bc.validate_conditions(conditions, self.domain, self.fields), [])

    def test_periodic_must_be_reciprocated(self):
        conditions = [bc.make_periodic("u", "left", "right")]
        self.assertIn("condition_periodic_not_reciprocated",
                      _codes(bc.validate_conditions(conditions, self.domain, self.fields)))
        both = [bc.make_periodic("u", "left", "right"),
                bc.make_periodic("u", "right", "left")]
        self.assertEqual(bc.validate_conditions(both, self.domain, self.fields), [])

    def test_periodic_with_self_rejected(self):
        conditions = [bc.make_periodic("u", "left", "left")]
        self.assertIn("condition_partner_self",
                      _codes(bc.validate_conditions(conditions, self.domain, self.fields)))

    def test_negative_robin_transfer_rejected(self):
        conditions = [bc.make_robin("u", "left", -1.0, 0.0)]
        self.assertIn("condition_transfer_negative",
                      _codes(bc.validate_conditions(conditions, self.domain, self.fields)))

    def test_duplicate_ids_rejected(self):
        conditions = [bc.make_dirichlet("u", "left", 0.0, condition_id="same"),
                      bc.make_neumann("u", "right", 0.0, condition_id="same")]
        self.assertIn("condition_id_duplicate",
                      _codes(bc.validate_conditions(conditions, self.domain, self.fields)))

    def test_unassigned_boundary_is_described_as_no_flux(self):
        lines = bc.describe_conditions([], self.domain, self.fields)
        self.assertTrue(any("no-flux" in line for line in lines))

    def test_resolution_defaults_to_no_flux(self):
        resolved = bc.resolve_1d_conditions([], "u")
        self.assertEqual(resolved["left"]["type"], "no_flux")
        self.assertEqual(resolved["right"]["type"], "no_flux")
        self.assertFalse(resolved["periodic"])


class BoundaryPersistenceThroughMeshingTests(unittest.TestCase):
    def test_condition_still_applies_after_remeshing(self):
        """A condition attached to a NAME must survive a mesh change."""
        domain = geo.make_interval(1.0)
        conditions = [bc.make_dirichlet("u", "left", 0.0),
                      bc.make_dirichlet("u", "right", 1.0)]
        self.assertEqual(bc.validate_conditions(conditions, domain, ["u"]), [])

        finals = []
        for count in (20, 80):
            mesh = meshing.mesh_1d(domain, element_count=count)
            x = np.array([node[0] for node in mesh["nodes"]], dtype=float)
            # The tagged end nodes differ between meshes, which is the point.
            self.assertEqual(mesh["boundaries"]["right"]["nodes"], [count])
            dx = x[1] - x[0]
            resolved = bc.resolve_1d_conditions(conditions, "u")
            result = solve_1d(x, np.zeros_like(x), diffusion=1.0, t_max=2.0,
                              dt=0.4 * dx * dx / 2.0,
                              left=resolved["left"], right=resolved["right"],
                              save_every=1000)
            finals.append(np.asarray(result["u"][-1], dtype=float))
            # Both resolutions must converge on the same straight line.
            np.testing.assert_allclose(finals[-1], steady_state_1d(x, 1.0, 0.0, 1.0),
                                       atol=3e-3)
        self.assertAlmostEqual(finals[0][0], finals[1][0], places=9)
        self.assertAlmostEqual(finals[0][-1], finals[1][-1], places=9)


if __name__ == "__main__":
    unittest.main()
