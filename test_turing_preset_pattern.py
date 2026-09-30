"""The Turing preset must form a pattern from its own text.

"U activates itself and activates V" lost its second clause in the rule-based compiler: the word
after "and" is a verb, so the U -> V activation was dropped, V was left with only a basal supply,
and the field settled to a uniform state (U std 2e-10 at t = 200). The Turing target is a variance
threshold whose tolerance cannot fail, so no test noticed.
"""
import unittest

import numpy as np

import agent
import nl_compiler as nc
import preset_loader


def _regulated(ir, target):
    return [(r["species"], r["effect"]) for p in ir["processes"] if p["kind"] == "production"
            and p.get("target") == target for r in p.get("regulators", []) if r.get("species")]


class SelfRegulationConjunctionTests(unittest.TestCase):
    def test_a_second_verb_after_itself_keeps_its_own_target(self):
        ir = nc.extract_ir_rules("U activates itself and activates V. V inhibits U. U starts at 1. V starts at 1.")
        self.assertIn(("U", "activate"), _regulated(ir, "V"))
        self.assertIn(("U", "activate"), _regulated(ir, "U"))

    def test_the_second_verb_sets_the_second_effect(self):
        ir = nc.extract_ir_rules("A activates itself and inhibits B. A starts at 1. B starts at 1.")
        self.assertIn(("A", "repress"), _regulated(ir, "B"))

    def test_the_named_partner_form_still_works(self):
        ir = nc.extract_ir_rules("A activates itself and the inhibitor B. A starts at 1. B starts at 1.")
        self.assertIn(("A", "activate"), _regulated(ir, "B"))


class TuringPresetPatternTests(unittest.TestCase):
    def test_the_shipped_text_forms_a_spatial_pattern(self):
        text = preset_loader.load_presets()["turing"]["text"]
        bp = nc.compile_text(text, None)
        self.assertEqual(bp.get("type"), "PDE", bp.get("validation_errors"))
        result = agent.simulate_pde_blueprint(bp, seed=0)
        final_u = np.asarray(result["species"]["U"][-1])
        # Measured: std 0.92 around a mean of 0.59 (before the fix: 2e-10, i.e. uniform).
        self.assertGreater(float(final_u.std()), 0.1 * float(final_u.mean()))


if __name__ == "__main__":
    unittest.main()
