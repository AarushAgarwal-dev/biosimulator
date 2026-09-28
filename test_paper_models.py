"""The two "published model" presets checked against the numbers in their papers.

What ships is what is tested: both blueprints are read out of static/app.js by
preset_loader, and every right-hand side below is the app's own compiled one
(ODEModel._f_lambdified), not a re-implementation.

* Goldbeter A, Dupont G, Berridge MJ (1990) PNAS 87:1461-1465 - minimal model for
  signal-induced Ca2+ oscillations. Checked: the Fig. 2 parameter values, Eqs. 1-2,
  the oscillation interval stated with Fig. 3 (beta = 29.1-77.5 %), the high steady
  state above it ("close to 0.7 uM"), the period and amplitude curves of Fig. 3
  (digitised), and the effect of a lower extrusion rate (Fig. 2 Lower).
* Zhabotinsky AM (2000) Biophys J 79:2211-2221 - CaMKII/phosphatase bistability, with
  the Ca2+-independent phosphatase of Fig. 9 (KM 0.4 uM, ek 20 uM, ep0 0.3 uM, I0 = 0).
  Checked: Table 1 constants, Eqs. 6/12/15/17, the four step experiments of Fig. 9B,
  and the single-valued characteristic of the cytosolic parameter set (Fig. 10A).
  ek is read as the holoenzyme concentration (sum of P_i): Fig. 10 reports ~4.6 uM of
  phosphorylated subunits with ek = 1.0 uM, impossible if ek counted subunits.
"""
import math
import unittest

import numpy as np
from scipy.optimize import brentq
from scipy.signal import find_peaks

import agent
import preset_loader
from simulation_engine import ODEModel

BLUEPRINTS = preset_loader.load_blueprints(text_compiler=agent.rule_based_parse)


def _rhs(model, params):
    values = [float(params.get(name, model.params_dict[name])) for name in model.param_names]
    return lambda y: np.asarray(model._f_lambdified(0.0, list(y), values), dtype=float)


# ------------------------------------------------------------------------------------
# Goldbeter, Dupont & Berridge (1990)
# ------------------------------------------------------------------------------------

# Fig. 3, digitised from the PNAS page image (filled diamonds; frame calibrated to
# 20-80 % and 0-2.5): (beta %, period s) and (beta %, maximum Z uM).
FIG3_PERIOD = [(29.65, 2.273), (30.28, 2.059), (31.58, 1.785), (34.22, 1.427), (38.23, 1.112),
               (42.26, 0.919), (46.37, 0.783), (50.40, 0.680), (54.53, 0.607), (59.93, 0.532),
               (64.03, 0.487), (69.51, 0.449), (72.21, 0.404), (74.92, 0.361)]
FIG3_AMPLITUDE = [(31.47, 1.336), (34.21, 1.331), (42.41, 1.282), (46.48, 1.253), (50.56, 1.224),
                  (54.69, 1.197), (60.10, 1.161), (64.23, 1.124), (69.68, 1.046), (72.38, 0.969),
                  (75.09, 0.850)]


class GoldbeterDupontBerridge1990Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bp = BLUEPRINTS["berridge"]
        cls.model = ODEModel(cls.bp)
        cls.ids = cls.model.node_ids

    def _steady(self, beta):
        p = dict(self.bp["parameters"], beta=beta)
        f = _rhs(self.model, p)
        z = (p["v0"] + p["v1"] * beta) / p["k"]          # the sum of Eqs. 1-2 fixes Z* exactly
        iz, iy = self.ids.index("Z"), self.ids.index("Y")

        def dy(y_store):
            y = np.zeros(len(self.ids)); y[iz], y[iy] = z, y_store
            return f(y)[iy]

        y = np.zeros(len(self.ids)); y[iz], y[iy] = z, brentq(dy, 1e-9, 1e4)
        return y, f

    def _max_real_eigenvalue(self, beta):
        y, f = self._steady(beta)
        n = len(y)
        J = np.zeros((n, n))
        for j in range(n):
            h = 1e-7 * max(1.0, abs(y[j]))
            d = np.zeros(n); d[j] = h
            J[:, j] = (f(y + d) - f(y - d)) / (2 * h)
        return float(np.max(np.linalg.eigvals(J).real))

    def _period_and_max(self, beta, **extra):
        res = self.model.simulate(60.0, num_points=12001, custom_params=dict(extra, beta=beta))
        t, z = np.asarray(res["t"]), np.asarray(res["species"]["Z"])
        keep = t > 30.0
        t, z = t[keep], z[keep]
        peaks, _ = find_peaks(z, prominence=0.05 * (z.max() - z.min() + 1e-12))
        period = float(np.mean(np.diff(t[peaks]))) if len(peaks) >= 3 else None
        return period, float(z.max())

    def test_parameters_are_the_fig2_values(self):
        expected = {"v0": 1.0, "k": 10.0, "kf": 1.0, "v1": 7.3, "VM2": 65.0, "VM3": 500.0,
                    "K2": 1.0, "KR": 2.0, "KA": 0.9}
        for name, value in expected.items():
            self.assertEqual(float(self.bp["parameters"][name]), value, name)

    def test_rate_laws_are_eqs_1_and_2(self):
        p = dict(self.bp["parameters"])
        f = _rhs(self.model, p)
        rng = np.random.default_rng(0)
        iz, iy = self.ids.index("Z"), self.ids.index("Y")
        for _ in range(20):
            Z, Y = rng.uniform(0.05, 3.0, size=2)
            v2 = p["VM2"] * Z ** 2 / (p["K2"] ** 2 + Z ** 2)                       # m = n = 2
            v3 = p["VM3"] * Y ** 2 / (p["KR"] ** 2 + Y ** 2) * Z ** 4 / (p["KA"] ** 4 + Z ** 4)   # p = 4
            y = np.zeros(len(self.ids)); y[iz], y[iy] = Z, Y
            out = f(y)
            self.assertAlmostEqual(out[iz], p["v0"] + p["v1"] * p["beta"] - v2 + v3 + p["kf"] * Y - p["k"] * Z, 9)
            self.assertAlmostEqual(out[iy], v2 - v3 - p["kf"] * Y, 9)

    def test_oscillation_interval_is_the_published_one(self):
        """Fig. 3 legend: oscillations occur when beta ranges from 29.1 % to 77.5 %."""
        low = brentq(self._max_real_eigenvalue, 0.25, 0.40)
        high = brentq(self._max_real_eigenvalue, 0.70, 0.90)
        self.assertAlmostEqual(low, 0.291, delta=0.005)
        self.assertAlmostEqual(high, 0.775, delta=0.005)
        self.assertLess(self._max_real_eigenvalue(0.25), 0.0)
        self.assertGreater(self._max_real_eigenvalue(0.5), 0.0)

    def test_high_steady_state_above_the_interval(self):
        """'... a larger value, close to 0.7 uM, for beta > 77.5 %.'"""
        y, _ = self._steady(0.78)
        self.assertAlmostEqual(y[self.ids.index("Z")], 0.7, delta=0.05)

    def test_period_and_amplitude_follow_fig3(self):
        for beta, published in FIG3_PERIOD:
            period, _ = self._period_and_max(beta / 100.0)
            self.assertIsNotNone(period, beta)
            self.assertLess(abs(period / published - 1.0), 0.05, (beta, period, published))
        for beta, published in FIG3_AMPLITUDE:
            _, peak = self._period_and_max(beta / 100.0)
            self.assertLess(abs(peak / published - 1.0), 0.03, (beta, peak, published))

    def test_lower_extrusion_rate_gives_larger_faster_spikes(self):
        """Fig. 2 Lower (k = 6 s^-1): larger Ca2+ spikes at a higher frequency."""
        p10, a10 = self._period_and_max(0.301)
        p6, a6 = self._period_and_max(0.301, k=6.0)
        self.assertLess(p6, p10)
        self.assertGreater(a6, a10)


# ------------------------------------------------------------------------------------
# Zhabotinsky (2000)
# ------------------------------------------------------------------------------------

class Zhabotinsky2000Test(unittest.TestCase):
    STATES = [f"P{i}" for i in range(11)]

    @classmethod
    def setUpClass(cls):
        cls.bp = BLUEPRINTS["zhabotinsky"]
        cls.model = ODEModel(cls.bp)
        cls.total = sum(float(n["initial_value"]) for n in cls.bp["nodes"] if n["id"] in cls.STATES)

    def _start(self, branch, total=None):
        total = self.total if total is None else total
        state = {s: 0.0 for s in self.STATES}
        state["P0" if branch == "low" else "P10"] = total
        return state

    def _settle(self, ca, start, t=4000.0, **params):
        init = dict(start, Ca=ca, A=sum(i * start[f"P{i}"] for i in range(11)))
        res = self.model.simulate(t, num_points=201, custom_params=dict(params, Cabase=ca, amp=0.0),
                                  custom_initial=init)
        final = {k: float(v[-1]) for k, v in res["species"].items()}
        return {s: final[s] for s in self.STATES}, sum(i * final[f"P{i}"] for i in range(11))

    def test_parameters_are_table1_and_fig9(self):
        p = self.bp["parameters"]
        self.assertEqual((p["k1"], p["k2"], p["KH1"]), (0.5, 2.0, 4.0))      # Table 1
        self.assertEqual((p["KM"], p["ep"]), (0.4, 0.3))                      # Fig. 9 legend
        self.assertAlmostEqual(self.total, 20.0, places=12)                   # ek = 20 uM

    def test_rate_laws_are_eqs_6_12_15_17(self):
        p = dict(self.bp["parameters"])
        f = _rhs(self.model, p)
        ids = self.model.node_ids
        w = [1.0, 1.8, 2.3, 2.7, 2.8, 2.7, 2.3, 1.8, 1.0]
        rng = np.random.default_rng(1)
        for _ in range(10):
            P = rng.uniform(0.0, 3.0, size=11)
            ca = rng.uniform(0.5, 3.0)
            y = np.zeros(len(ids))
            for i in range(11):
                y[ids.index(f"P{i}")] = P[i]
            y[ids.index("Ca")] = ca
            out = f(y)
            x4 = (ca / p["KH1"]) ** 4
            v1 = 10 * p["k1"] * x4 ** 2 * P[0] / (1 + x4) ** 2                     # Eq. 6
            v2 = p["k1"] * x4 / (1 + x4)                                          # Eq. 12
            v3 = p["k2"] * p["ep"] / (p["KM"] + sum(i * P[i] for i in range(1, 11)))   # Eq. 15
            expect = np.zeros(11)                                                  # Eq. 17
            expect[0] = -v1 + v3 * P[1]
            expect[1] = v1 - v3 * P[1] - v2 * P[1] + 2 * v3 * P[2]
            for i in range(2, 10):
                expect[i] = w[i - 2] * v2 * P[i - 1] - i * v3 * P[i] - w[i - 1] * v2 * P[i] + (i + 1) * v3 * P[i + 1]
            expect[10] = v2 * P[9] - 10 * v3 * P[10]
            for i in range(11):
                self.assertAlmostEqual(out[ids.index(f"P{i}")], expect[i], 9, f"P{i}")

    def test_fig9b_step_experiments(self):
        """Fig. 9B: 1.3->1.8 stays low; 1.3->2.2 switches up; 2.3->1.8 stays up; 2.3->1.5 switches down."""
        half = 0.5 * 10 * self.total
        for before, after, expect in ((1.3, 1.8, "low"), (1.3, 2.2, "high"), (2.3, 1.8, "high"), (2.3, 1.5, "low")):
            start, _ = self._settle(before, self._start("low" if before < 2.0 else "high"))
            _, subunits = self._settle(after, start)
            self.assertEqual("high" if subunits > half else "low", expect, (before, after, subunits))

    def test_baseline_calcium_is_inside_the_bistable_window(self):
        ca = float(self.bp["parameters"]["Cabase"])
        _, low = self._settle(ca, self._start("low"))
        _, high = self._settle(ca, self._start("high"))
        self.assertLess(low, 1.0)
        self.assertGreater(high, 100.0)

    def test_total_holoenzyme_is_conserved(self):
        res = self.model.simulate(float(self.bp["simulation_config"]["t_max"]), num_points=601)
        total = sum(np.asarray(res["species"][s]) for s in self.STATES)
        self.assertLess(float(np.max(np.abs(total - self.total))), 1e-6)

    def test_fig10_cytosolic_set_is_single_valued(self):
        """Fig. 10A: with KM 15 uM, ek 1 uM, ep0 0.05 uM the characteristic is single-valued."""
        for ca in (0.5, 1.0, 2.0, 3.0, 4.0, 5.0):
            _, low = self._settle(ca, self._start("low", 1.0), KM=15.0, ep=0.05)
            _, high = self._settle(ca, self._start("high", 1.0), KM=15.0, ep=0.05)
            self.assertLess(abs(high - low), 0.05, ca)


class ShippedPresetsThroughTheApiTest(unittest.TestCase):
    """Every preset the UI ships must survive the server's own validation.

    The published Ca2+ model names a parameter `beta`; the pre-validation parsed it as
    SymPy's beta function and /api/compile answered 422, so the preset button failed
    in the browser while every engine-level test passed.
    """

    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient

        import main
        cls.client = TestClient(main.app)

    def test_each_preset_compiles_and_simulates(self):
        for name, blueprint in BLUEPRINTS.items():
            if blueprint.get("type") == "PDE":
                continue
            with self.subTest(preset=name):
                compiled = self.client.post("/api/compile", json={"blueprint": blueprint})
                self.assertEqual(compiled.status_code, 200, compiled.text[:300])
                ran = self.client.post("/api/simulate", json={"blueprint": blueprint})
                self.assertEqual(ran.status_code, 200, ran.text[:300])

    def test_parameters_named_like_sympy_functions_are_symbols(self):
        blueprint = {"type": "ODE", "nodes": [{"id": "X", "initial_value": 1.0}], "edges": [],
                     "parameters": {"beta": 0.5, "gamma": 0.1, "zeta": 2.0},
                     "odes": {"X": "beta*zeta - gamma*X"}, "simulation_config": {"t_max": 5.0}}
        ran = self.client.post("/api/simulate", json={"blueprint": blueprint})
        self.assertEqual(ran.status_code, 200, ran.text[:300])


if __name__ == "__main__":
    unittest.main()
