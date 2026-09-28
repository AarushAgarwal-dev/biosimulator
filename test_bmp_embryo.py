"""Tests for the Umulis et al. 2010 BMP embryo implementation (bmp_embryo.py).

The numerical core is checked against exact operator identities, conservation, positivity and
grid convergence; the scientific behaviour is checked against the published curves and claims
(see bmp_embryo.validate). Failures documented as reproduction gaps are asserted to stay
REPORTED as failures rather than being silently hidden or "fixed" by tuning.
"""
import math
import unittest

import numpy as np
from scipy.sparse.linalg import expm_multiply

import bmp_embryo as bmp


class TestGeometryAndOperators(unittest.TestCase):
    def test_prolate_surface_area(self):
        _, area, _ = bmp._surface_laplacian(200.0, 90.0, 64, 40)
        e = math.sqrt(1.0 - (90.0 / 200.0) ** 2)
        exact = 2.0 * math.pi * 90.0 ** 2 * (1.0 + 200.0 / (90.0 * e) * math.asin(e))
        self.assertLess(abs(2.0 * area.sum() - exact) / exact, 0.005)

    def test_sphere_laplacian_cos_u(self):
        radius, nu, nv = 90.0, 64, 20
        lap, _, grid = bmp._surface_laplacian(radius, radius, nu, nv)
        numerical = (lap @ np.repeat(np.cos(grid["u"]), nv)).reshape(nu, nv)
        exact = (-2.0 / radius ** 2) * np.cos(grid["u"])
        rel = np.linalg.norm(numerical[2:-2, 0] - exact[2:-2]) / np.linalg.norm(exact[2:-2])
        self.assertLess(rel, 0.01)

    def test_surface_diffusion_conserves_and_relaxes(self):
        lap, area, _ = bmp._surface_laplacian(200.0, 90.0, 24, 16)
        initial = 0.2 + np.random.default_rng(12).random(lap.shape[0])
        final = expm_multiply(1000.0 * lap, initial)
        self.assertAlmostEqual(float(area @ initial), float(area @ final), delta=1e-8 * float(area @ initial))
        mean = float(area @ initial / area.sum())
        self.assertLess(float(area @ (final - mean) ** 2), float(area @ (initial - mean) ** 2))

    def test_cross_section_laplacian_is_second_order(self):
        # d2/dx2 cos(pi x / L) = -(pi/L)^2 cos(pi x / L), with zero-flux ends.
        errors = []
        for n in (41, 81):
            lap, _, s = bmp._cross_section_laplacian(n, 100.0)
            f = np.cos(math.pi * s / 100.0)
            errors.append(np.max(np.abs(lap @ f + (math.pi / 100.0) ** 2 * f)))
        self.assertGreater(errors[0] / errors[1], 3.5)      # ~4x per halving = 2nd order

    def test_reaction_jacobian_matches_finite_differences(self):
        p = bmp._base_params("sbp", "wt", None, 400.0, False)
        rng = np.random.default_rng(3)
        y = rng.random((len(bmp.SPECIES), 1)) * 20.0
        prod = tuple(np.array([v]) for v in (1.0, 400.0, 48.0))
        _, J = bmp._reaction(y, p, "sbp", prod, jacobian=True)
        h = 1e-6
        for j in range(len(bmp.SPECIES)):
            yp, ym = y.copy(), y.copy()
            yp[j] += h
            ym[j] -= h
            col = (bmp._reaction(yp, p, "sbp", prod) - bmp._reaction(ym, p, "sbp", prod))[:, 0] / (2 * h)
            np.testing.assert_allclose(J[:, j, 0], col, rtol=1e-5, atol=1e-7)


class TestModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.wt = bmp.simulate_cross_section(save_times=[30.0, 60.0])
        cls.report = bmp.validate(include_surface=False)
        cls.by_id = {row["id"]: row for row in cls.report["checks"]}

    def test_mass_balance_and_positivity(self):
        d = self.wt["diagnostics"]
        self.assertTrue(d["analytic_sparse_jacobian"])
        self.assertLess(d["ligand_mass_balance_relative_error"], 1e-3)
        self.assertGreater(d["min_value"], -1e-6 * d["max_abs_value"])

    def test_uniform_production_stays_uniform(self):
        r = bmp.simulate_cross_section(
            n_nodes=17, t_end=1.0, save_times=[1.0],
            overrides={"phi_S": 0.0, "phi_T": 0.0, "dorsal_edge": 10.0,
                       "sog_start": 0.0, "sog_end": 10.0, "smooth_width_um": 0.01})
        for name, history in r["fields"].items():
            final = np.asarray(history[-1])
            self.assertLess(float(np.ptp(final)), 1e-7 * max(1.0, float(np.max(np.abs(final)))), name)

    def test_grid_convergence(self):
        self.assertLess(bmp.grid_convergence(57, 113, t_end=60.0)["relative_l2_BR"], 0.03)

    def test_solver_reproduces_published_2006_curves(self):
        row = self.by_id["S1"]
        self.assertTrue(row["pass"], row["measured"])
        for err in list(row["measured"]["relative_error_BR"].values()) + list(row["measured"]["relative_error_BMP"].values()):
            self.assertLess(abs(err), 0.12)

    def test_printed_sog_value_is_shown_to_collapse(self):
        self.assertTrue(self.by_id["D1"]["pass"])
        self.assertGreaterEqual(self.by_id["D1"]["measured"]["fwhm_percent_circumference"], 90.0)

    def test_required_paper_claims(self):
        for key in ("V1", "V2", "V4", "V6", "V7"):
            self.assertTrue(self.by_id[key]["pass"], (key, self.by_id[key]["measured"]))
        self.assertTrue(self.report["required_pass"])

    def test_reproduction_gaps_stay_reported(self):
        # These depend on inputs the paper did not publish. They must remain visible failures
        # unless the model genuinely changes; flipping one needs a deliberate test update.
        for key in ("F2", "V3b", "M1"):
            self.assertIs(self.by_id[key]["pass"], False, key)

    def test_pass_flags_match_their_criteria(self):
        v1 = self.by_id["V1"]["measured"]
        flank = v1["flank60um_BR_nM"]
        self.assertEqual(self.by_id["V1"]["pass"], v1["width30_um"] > v1["width60_um"]
                         and v1["peak60_nM"] > v1["peak30_nM"] and flank["60.0"] < max(flank.values()))
        v3 = self.by_id["V3"]["measured"]
        self.assertEqual(self.by_id["V3"]["pass"], v3["difference20_cells"] > 0 and v3["difference40_cells"] < v3["difference20_cells"])
        v7 = self.by_id["V7"]["measured"]
        self.assertEqual(self.by_id["V7"]["pass"], v7["no_feedback_dm_60_nM"] < v7["wt_dm_60_nM"]
                         and v7["no_feedback_flank60um_60_nM"] > v7["wt_flank60um_60_nM"])

    def test_parameter_sets(self):
        printed = bmp._base_params("sbp", "wt", None, 400.0, False, "umulis2010_as_printed")
        default = bmp._base_params("sbp", "wt", None, 400.0, False)
        self.assertEqual(printed["phi_S"], 1360.0)
        self.assertEqual(default["phi_S"], 400.0)
        self.assertEqual(default["k3"], 11.4)            # Table S1 unchanged
        self.assertEqual(default["Rtot"], 394.3)         # Table S8 unchanged
        with self.assertRaises(ValueError):
            bmp.simulate_cross_section(mechanism="receptor", parameter_set="umulis2006", t_end=1)
        with self.assertRaises(ValueError):
            bmp.simulate_surface(nu=8, nv=8, t_end=1, parameter_set="umulis2006")

    def test_model_info_cites_every_parameter(self):
        info = bmp.model_info()
        self.assertEqual(info["mechanisms"], list(bmp.MECHANISMS))
        for name, entry in info["parameter_table"].items():
            self.assertIn("value", entry, name)
            self.assertIn("units", entry, name)
            self.assertTrue(entry.get("source"), name)

    def test_validation_errors(self):
        with self.assertRaisesRegex(ValueError, "valid names"):
            bmp.simulate_cross_section(mechanism="bogus", t_end=1)
        with self.assertRaisesRegex(ValueError, "valid names"):
            bmp.simulate_cross_section(perturbation="bogus", t_end=1)
        with self.assertRaisesRegex(ValueError, "valid names"):
            bmp.simulate_cross_section(parameter_set="bogus", t_end=1)
        with self.assertRaises(ValueError):
            bmp.simulate_cross_section(n_nodes=3, t_end=1)
        with self.assertRaises(ValueError):
            bmp.simulate_surface(nu=4, nv=4, t_end=1)
        with self.assertRaises(ValueError):
            bmp.simulate_cross_section(t_end=601)
        with self.assertRaises(ValueError):
            bmp.simulate_cross_section(overrides={"not_a_parameter": 1.0}, t_end=1)


class TestSurface(unittest.TestCase):
    def test_surface_is_mirror_symmetric_and_localises(self):
        r = bmp.simulate_surface(nu=16, nv=12, save_times=[60.0])
        br = np.asarray(r["fields"]["BR"][-1])
        np.testing.assert_allclose(br, br[::-1, :], rtol=1e-9, atol=1e-9)
        mid = br[8]
        self.assertEqual(int(np.argmax(mid)), 0)                 # maximum at the dorsal midline
        self.assertLess(mid[-1], 0.1 * mid[0])                   # ventral side is dark
        self.assertLess(r["diagnostics"]["ligand_mass_balance_relative_error"], 1e-3)


if __name__ == "__main__":
    unittest.main()
