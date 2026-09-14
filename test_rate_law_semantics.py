"""RATE-LAW SEMANTICS: an edge the researcher drew must change the trajectory.

A fresh review of the product as a computational biologist would use it found that
INHIBITION EDGES WERE COMPILED INTO NO-OPS. In ODEModel._compile_system, a species with
no incoming activation took `activation_expr = synthesis`, and `syn_<X>` defaults to
0.0, so `total_production = activation_expr * inhibition_expr` was 0 multiplied by the
Hill factor -- zero for every possible inhibitor concentration. The species collapsed to
`dX/dt = -deg*X`.

The measurement that made it undeniable, for "A activates B. B inhibits A.", A(0)=1.0:

    B(0)     |  0.0        0.25       1.0        10.0       1000.0
    A(40)    |  0.018315639 (identical to nine significant figures)

and equal to the analytic exp(-0.1*40) of a bare exponential decay to 2.3e-10. A
researcher typing the canonical negative-feedback loop watched the inhibited species
fall and concluded the inhibitor caused it. Deleting the inhibitor produced a
bit-identical plot.

This is the worst failure shape in the whole product: not an error, not a refusal, but a
plausible curve that answers a question nobody asked. These tests exist so it cannot
come back, and they assert the DIRECTION of the effect rather than merely that something
changed -- an inhibitor that raised its target would also "have an effect".
"""
import unittest

import numpy as np

from simulation_engine import ODEModel

HILL = {"k": 0.5, "K_d": 1.0, "n": 2.0}


def _feedback_blueprint(b_initial, a_initial=1.0):
    """A activates B, B inhibits A -- the canonical negative feedback loop."""
    return {
        "type": "ODE",
        "nodes": [
            {"id": "A", "name": "A", "initial_value": a_initial},
            {"id": "B", "name": "B", "initial_value": b_initial},
        ],
        "edges": [
            {"source": "A", "target": "B", "type": "activation", "parameters": dict(HILL)},
            {"source": "B", "target": "A", "type": "inhibition", "parameters": dict(HILL)},
        ],
        "simulation_config": {"t_max": 40.0},
    }


def _final(blueprint, species, t_max=40.0):
    result = ODEModel(blueprint).simulate(t_max=t_max, num_points=2000)
    return float(np.asarray(result["species"][species], dtype=float)[-1])


class InhibitionHasAnEffectTests(unittest.TestCase):
    def test_the_inhibitor_concentration_changes_the_outcome(self):
        finals = [_final(_feedback_blueprint(b0), "A")
                  for b0 in (0.0, 0.25, 1.0, 10.0, 1000.0)]
        spread = max(finals) - min(finals)
        self.assertGreater(
            spread, 1e-3,
            f"A(40) is {spread:.3e} across inhibitor levels spanning 0 to 1000, so the "
            f"inhibition edge is inert. Values: {finals}")

    def test_more_inhibitor_leaves_less_of_the_inhibited_species(self):
        """Direction matters: an effect in the wrong direction is not a fix."""
        low = _final(_feedback_blueprint(0.0), "A")
        high = _final(_feedback_blueprint(1000.0), "A")
        self.assertLess(
            high, low,
            f"a thousandfold more inhibitor left MORE of the inhibited species "
            f"(A(40): {low:.6f} at B(0)=0 vs {high:.6f} at B(0)=1000)")

    def test_the_inhibited_species_is_not_just_decaying(self):
        """The exact signature of the bug: A(40) equal to bare exponential decay."""
        bare_decay = float(np.exp(-0.1 * 40.0))  # deg defaults to 0.1
        held = _final(_feedback_blueprint(0.0), "A")
        self.assertGreater(
            abs(held - bare_decay), 1e-4,
            f"A(40) = {held:.9f} is the bare-decay value exp(-0.1*40) = "
            f"{bare_decay:.9f}, which is what the species does when the inhibition "
            f"edge has been multiplied into nothing")

    def test_a_lone_inhibitor_still_suppresses_its_target(self):
        """"STRESS inhibits GROWTH." -- one edge, no activation anywhere."""
        def blueprint(stress):
            return {
                "type": "ODE",
                "nodes": [
                    {"id": "STRESS", "name": "STRESS", "initial_value": stress},
                    {"id": "GROWTH", "name": "GROWTH", "initial_value": 1.0},
                ],
                "edges": [{"source": "STRESS", "target": "GROWTH",
                           "type": "inhibition", "parameters": dict(HILL)}],
                "simulation_config": {"t_max": 40.0},
            }
        unstressed = _final(blueprint(0.0), "GROWTH")
        stressed = _final(blueprint(5.0), "GROWTH")
        self.assertLess(
            stressed, unstressed,
            f"stress did not suppress growth (GROWTH(40): {unstressed:.6f} unstressed "
            f"vs {stressed:.6f} stressed)")

    def test_an_explicit_synthesis_is_never_overwritten(self):
        """The basal term is supplied only where the model would otherwise be inert."""
        blueprint = _feedback_blueprint(0.25)
        for node in blueprint["nodes"]:
            if node["id"] == "A":
                node["synthesis"] = 2.0
        model = ODEModel(blueprint)
        self.assertEqual(
            float(model.params_dict["syn_A"]), 2.0,
            "a synthesis rate stated on the node was replaced by the derived basal term")

    def test_an_unused_edge_parameter_is_disclosed(self):
        """The rate law cannot use an inhibitor's k, so it must not pretend to.

        The LLM schema emits {"k":...,"K_d":...,"n":...} on every edge and the UI shows
        a slider for each. `k` has no place in Kd^n/(Kd^n + I^n), and it used to be
        unpacked and thrown away -- so a researcher could tune an inhibitor's strength,
        see no change, and have no way to learn why.
        """
        model = ODEModel(_feedback_blueprint(0.25))
        self.assertIn(
            "inh_B_to_A_k", getattr(model, "inert_parameters", []),
            "the inhibitor's k is silently ignored rather than reported as inert")

    def test_activation_still_works(self):
        """Guard against fixing inhibition by breaking activation."""
        def blueprint(ligand):
            return {
                "type": "ODE",
                "nodes": [
                    {"id": "L", "name": "L", "initial_value": ligand},
                    {"id": "R", "name": "R", "initial_value": 0.0},
                ],
                "edges": [{"source": "L", "target": "R", "type": "activation",
                           "parameters": dict(HILL)}],
                "simulation_config": {"t_max": 40.0},
            }
        none = _final(blueprint(0.0), "R")
        lots = _final(blueprint(5.0), "R")
        self.assertGreater(lots, none,
                           f"more ligand did not produce more R ({none:.6f} -> {lots:.6f})")


if __name__ == "__main__":
    unittest.main()


class OscillatorWorksAtBiologicalCooperativityTests(unittest.TestCase):
    """The showcase oscillator must not require impossible cooperativity.

    The Goodwin loop with FIRST-ORDER removal has a limit cycle only for Hill n > 8
    (Griffith 1968), and the preset relied on exactly that: n = 16. Measured on this
    engine, nothing in the biological range oscillated at all --

        n         | 2     3     4     6     8     9     10    16
        amplitude | 0.000 0.000 0.000 0.000 0.015 0.161 0.686 1.977

    -- while measured Hill coefficients for cooperative transcriptional repression are
    1-5. So a biologist's correct conclusion was that the tool cannot build a
    physiological oscillator, and the optimizer could not find one either, because the
    _param_bounds cap for *_n is (1.0, 8.0), BELOW the old Hopf point.

    The fix was mechanistic, not a retune: saturable removal, -d*X/(Km + X), after Bliss,
    Painter & Marr (1982). Near saturation that is zero-order in X, and the resulting
    ultrasensitivity supplies what high cooperativity was standing in for.

    These tests exist because a future retune could silently walk the preset back into
    needing n > 8, and the shipped oscillation target would still pass.
    """

    @classmethod
    def setUpClass(cls):
        import agent
        import preset_loader
        cls.agent = agent
        cls.preset = preset_loader.load_presets()["oscillator"]
        cls.blueprint = preset_loader.load_blueprints(
            text_compiler=agent.rule_based_parse)["oscillator"]
        cls.readout = None
        for target in (cls.preset.get("targets") or []):
            if target.get("type") == "oscillation":
                cls.readout = target.get("species")

    def _measure(self, n=None, t_max=300.0, points=4000):
        import copy
        import numpy as np
        blueprint = copy.deepcopy(self.blueprint)
        if n is not None:
            blueprint.setdefault("parameters", {})["n"] = float(n)
        result = ODEModel(blueprint).simulate(t_max=t_max, num_points=points)
        y = np.asarray(result["species"][self.readout], dtype=float)
        return (float(self.agent.sustained_oscillation_amplitude(y)),
                int(self.agent.count_sustained_peaks(y)))

    def test_the_shipped_cooperativity_is_biologically_attainable(self):
        n = float((self.blueprint.get("parameters") or {}).get("n", 0))
        self.assertLessEqual(
            n, 5.0,
            f"the oscillator ships n = {n}, which implies {int(n)} cooperative binding "
            f"sites. Measured Hill coefficients for cooperative transcriptional "
            f"repression are 1-5, so this is not a physiological oscillator.")

    def test_it_oscillates_across_the_biological_range(self):
        for n in (1.0, 2.0, 3.0, 4.0):
            with self.subTest(n=n):
                amplitude, peaks = self._measure(n=n)
                self.assertGreater(
                    amplitude, 0.3,
                    f"no sustained oscillation at n = {n} (amplitude {amplitude:.4f}); "
                    f"the loop still needs cooperativity no molecule provides")
                self.assertGreaterEqual(
                    peaks, 4,
                    f"only {peaks} sustained peaks at n = {n}")

    def test_the_amplitude_survives_a_ten_times_horizon(self):
        """A limit cycle, not a slow transient that happens to look periodic."""
        short, _ = self._measure(t_max=300.0)
        long, _ = self._measure(t_max=3000.0, points=6000)
        self.assertGreater(long, 0.3, "the oscillation died out over a longer run")
        self.assertLess(
            abs(long - short) / max(short, 1e-9), 0.25,
            f"the amplitude moved from {short:.4f} to {long:.4f} over a 10x horizon, so "
            f"this is a transient rather than a limit cycle")

    def test_removal_is_saturable_not_first_order(self):
        """The mechanism is the fix; a linear-removal retune would regress silently."""
        odes = self.blueprint.get("odes") or {}
        self.assertTrue(odes, "the oscillator lost its explicit rate laws")
        joined = " ".join(str(v) for v in odes.values())
        self.assertIn(
            "Km", joined,
            f"removal is no longer saturable, so the loop is back to needing n > 8: "
            f"{joined}")



class GeneratedParameterDefaultsTests(unittest.TestCase):
    """The defaults a generated model gets must be defensible, and must be disclosed.

    Every edge used to be given n = 2.0 and K_d = 1.0 regardless of the description. Both
    are consequential:

      - A Hill function's 10-90% response spans an 81^(1/n)-fold change in the regulator:
        81x at n=1, 9x at n=2, 3x at n=4. Defaulting to 2 made every edge roughly nine
        times more switch-like than mass action, and it compounds multiplicatively down a
        cascade. Ligand-receptor binding, a monomeric transcription factor and a
        Michaelis-Menten step are all n = 1.
      - K_d = 1.0 alongside starting values near 1.0 placed every regulator exactly AT its
        half-saturation point, which for n > 1 is the point of maximum logarithmic gain --
        the default parameterisation was the maximum-sensitivity parameterisation. For a
        species whose scale is 250 it was simply wrong, because K_d only means anything
        relative to the concentration it is compared against.
    """

    def _parse(self, text):
        import agent
        return agent.rule_based_parse(text)

    def _first_edge(self, text):
        blueprint = self._parse(text)
        edges = blueprint.get("edges") or []
        self.assertTrue(edges, f"no edges parsed from {text!r}")
        return blueprint, edges[0]

    def test_mass_action_is_the_default(self):
        _bp, edge = self._first_edge("LIGAND activates RECEPTOR. LIGAND starts at 2.0.")
        self.assertEqual(float(edge["parameters"]["n"]), 1.0,
                         "a description implying no cooperativity still got a cooperative "
                         "Hill exponent")

    def test_cooperativity_is_raised_only_when_the_text_says_so(self):
        for text, expected in (
            ("A dimer of TF activates GENE. TF starts at 2.0.", 2.0),
            ("CAM cooperatively activates KIN. CAM starts at 2.0.", 2.0),
            ("X ultrasensitively activates Y. X starts at 2.0.", 2.0),
            ("A tetramer of HB represses GENE. HB starts at 2.0.", 4.0),
            ("A activates B. A starts at 2.0.", 1.0),
        ):
            with self.subTest(text=text):
                _bp, edge = self._first_edge(text)
                self.assertEqual(float(edge["parameters"]["n"]), expected,
                                 f"wrong cooperativity inferred from {text!r}")

    def test_inflected_forms_are_recognised(self):
        """"cooperatively" and "ultrasensitively" are how people actually write.

        An earlier version anchored the pattern with a trailing \\b, so every inflected
        form scored n = 1 -- which is most real writing.
        """
        for text in ("CAM cooperatively activates KIN. CAM starts at 1.0.",
                     "X ultrasensitively activates Y. X starts at 1.0."):
            with self.subTest(text=text):
                _bp, edge = self._first_edge(text)
                self.assertEqual(float(edge["parameters"]["n"]), 2.0,
                                 f"an inflected cooperativity word was missed in {text!r}")

    def test_half_saturation_follows_the_regulator_scale(self):
        """K_d = 1.0 is meaningless for a species that lives at 250."""
        blueprint = self._parse(
            "GLUCOSE activates INSULIN. INSULIN inhibits GLUCOSE. "
            "GLUCOSE starts at 250.0.")
        levels = {n["id"]: float(n["initial_value"]) for n in blueprint["nodes"]}
        self.assertEqual(levels["GLUCOSE"], 250.0)
        for edge in blueprint["edges"]:
            if edge["source"] == "GLUCOSE":
                self.assertAlmostEqual(
                    float(edge["parameters"]["K_d"]), 250.0, places=3,
                    msg="K_d was not scaled to the regulator's own level, so the "
                        "interaction sits far from its half-saturation point")

    def test_a_regulator_starting_at_zero_borrows_an_upstream_scale(self):
        """It has no scale of its own yet; inventing 1.0 would be arbitrary."""
        blueprint = self._parse(
            "LIGAND activates RECEPTOR. RECEPTOR activates KINASE. LIGAND starts at 80.0.")
        for edge in blueprint["edges"]:
            kd = float(edge["parameters"]["K_d"])
            self.assertGreater(kd, 1.0,
                               f"{edge['source']} -> {edge['target']} kept K_d = {kd} "
                               f"despite a model whose scale is 80")

    def test_the_parameterisation_is_disclosed(self):
        blueprint = self._parse("A activates B. A starts at 3.0.")
        notice = str(blueprint.get("_llm_notice") or "")
        self.assertIn("NOT fitted", notice,
                      "the model does not say its parameters were never fitted")
        self.assertIn("half-saturation", notice,
                      "the notice does not explain how K_d was chosen")
