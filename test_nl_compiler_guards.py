"""Regression tests for the reliability guards added to nl_compiler after review.

* An LLM must not be able to recycle one sentence as evidence for a process of a different kind
  (e.g. quote "Y is converted to X" to justify an invented degradation of Y).
* A custom rate law that consumes a species must vanish when that species is exhausted.
* Active-voice conversion prose with an explicit rate law (SIR) compiles deterministically.
"""
import copy
import unittest

import numpy as np

import nl_compiler as nc
from simulation_engine import ODEModel

CALCIUM = ("CalciumStore is converted to CytosolicCalcium at rate 0.2. CytosolicCalcium is converted "
           "to CalciumStore at rate 0.1. CalciumStore starts at 10 and CytosolicCalcium starts at 0.")
SIR = ("SIR model: infection converts S to I at rate beta*S*I with beta 0.3, and recovery converts I to R "
       "at rate gamma*I with gamma 0.1. S starts at 999, I at 1, R at 0. Simulate 160 days.")


class KindEvidenceGuardTests(unittest.TestCase):
    def test_recycled_evidence_for_an_invented_degradation_is_rejected(self):
        ir = nc.extract_ir_rules(CALCIUM)
        self.assertEqual(nc.validate_ir(ir, CALCIUM), [])
        bad = copy.deepcopy(ir)
        bad["parameters"].append({"name": "k_leak", "value": 0.05, "source": "default", "unit": "1/s", "meaning": "leak"})
        bad["processes"].append({"id": "p_leak", "kind": "degradation", "species": "CytosolicCalcium", "k": "k_leak",
                                 "evidence": "CytosolicCalcium is converted to CalciumStore", "assumed": False})
        errors = nc.validate_ir(bad, CALCIUM)
        self.assertTrue(any("p_leak" in e and "degradation" in e for e in errors), errors)

    def test_genuine_degradation_wording_is_accepted(self):
        text = "A is produced at rate 1. A is degraded at rate 0.2."
        self.assertEqual(nc.validate_ir(nc.extract_ir_rules(text), text), [])

    def test_assumed_processes_are_exempt_but_must_say_why(self):
        ir = nc.extract_ir_rules("A activates B.")
        self.assertTrue(any(p["assumed"] for p in ir["processes"]))
        self.assertEqual(nc.validate_ir(ir, "A activates B."), [])


class CustomRatePositivityTests(unittest.TestCase):
    def test_consumption_rate_must_vanish_at_zero(self):
        ir = nc.extract_ir_rules(SIR)
        self.assertEqual(nc.validate_ir(ir, SIR), [])
        bad = copy.deepcopy(ir)
        for proc in bad["processes"]:
            if proc["kind"] == "custom" and proc["reactants"][0]["species"] == "S":
                proc["rate"] = "beta*I"            # does not vanish when S = 0
        errors = nc.validate_ir(bad, SIR)
        self.assertTrue(any("does not vanish" in e and "'S'" in e for e in errors), errors)


class SIRRuleTests(unittest.TestCase):
    def test_sir_compiles_conserves_and_matches_the_stated_numbers(self):
        bp = nc.compile_text(SIR)
        self.assertNotIn("validation_errors", bp)
        self.assertEqual(bp["parameters"], {"beta": 0.3, "gamma": 0.1})
        self.assertEqual({n["id"]: n["initial_value"] for n in bp["nodes"]}, {"S": 999.0, "I": 1.0, "R": 0.0})
        result = ODEModel(bp).simulate(20.0, 200)
        total = sum(np.asarray(result["species"][k]) for k in ("S", "I", "R"))
        self.assertLess(float(np.ptp(total)), 1e-6 * 1000.0)
        self.assertGreater(max(result["species"]["I"]), 1.0)      # the epidemic takes off (R0 = 3)


class RegulatorSourceGuardTests(unittest.TestCase):
    TEXT = "p53 activates Mdm2 transcription. Mdm2 promotes p53 degradation."

    def llm_style_ir(self):
        """The shape Purdue gpt-oss:120b returned: p53 has losses but no production."""
        return {
            "model_type": "ode", "time_unit": "arbitrary", "t_end": None,
            "species": [{"id": "p53", "name": "p53", "initial": 0.5, "initial_source": "default", "diffusion": None, "role": "state"},
                        {"id": "Mdm2", "name": "Mdm2", "initial": 1.0, "initial_source": "default", "diffusion": None, "role": "state"}],
            "parameters": [{"name": n, "value": v, "source": "default", "unit": "a", "meaning": n}
                           for n, v in (("k_prod_Mdm2", 1.0), ("K_a", 1.0), ("n_a", 2.0), ("k_deg_p53", 0.1),
                                        ("K_b", 1.0), ("n_b", 2.0), ("k_deg_Mdm2", 0.1))],
            "stimuli": [],
            "processes": [
                {"id": "p1", "kind": "production", "target": "Mdm2", "k": "k_prod_Mdm2", "evidence": "p53 activates Mdm2 transcription",
                 "assumed": False, "regulators": [{"species": "p53", "effect": "activate", "K": "K_a", "n": "n_a"}]},
                {"id": "p2", "kind": "degradation", "species": "p53", "k": "k_deg_p53", "evidence": "Mdm2 promotes p53 degradation",
                 "assumed": False, "regulators": [{"species": "Mdm2", "effect": "activate", "K": "K_b", "n": "n_b"}]},
                {"id": "p3", "kind": "degradation", "species": "Mdm2", "k": "k_deg_Mdm2", "evidence": "", "assumed": True,
                 "reason": "turnover"}],
            "assumptions": [], "unmodeled": [],
        }

    def test_source_less_regulator_gets_a_disclosed_supply(self):
        ir = nc._auto_repair_ir(self.llm_style_ir(), self.TEXT)
        supplies = [p for p in ir["processes"] if p["kind"] == "production" and p["target"] == "p53"]
        self.assertEqual(len(supplies), 1)
        self.assertTrue(supplies[0]["assumed"])
        self.assertTrue(any("p53 has a basal supply" in a for a in ir["assumptions"]))
        bp = nc.compile_ir(ir)
        result = ODEModel(bp).simulate(200.0, 400)
        self.assertGreater(result["species"]["p53"][-1], 0.01)       # no longer decays to zero

    def test_a_species_with_a_source_is_left_alone(self):
        ir = self.llm_style_ir()
        before = len(ir["processes"])
        ir = nc._auto_repair_ir(ir, self.TEXT)
        self.assertFalse(any(p["kind"] == "production" and p["target"] == "Mdm2" and p.get("assumed") for p in ir["processes"]))
        self.assertEqual(len(ir["processes"]), before + 1)            # only the p53 supply was added

    def test_produced_but_never_lost_species_gets_disclosed_turnover(self):
        ir = self.llm_style_ir()
        ir["processes"] = [p for p in ir["processes"] if p["id"] != "p3"]      # the AI omitted Mdm2 turnover
        ir = nc._auto_repair_ir(ir, self.TEXT)
        turnover = [p for p in ir["processes"] if p["kind"] == "degradation" and p["species"] == "Mdm2"]
        self.assertEqual(len(turnover), 1)
        self.assertTrue(turnover[0]["assumed"])
        result = ODEModel(nc.compile_ir(ir)).simulate(400.0, 400)
        mdm2 = result["species"]["Mdm2"]
        self.assertLess(abs(mdm2[-1] - mdm2[-40]), 1e-2 * max(1.0, mdm2[-1]))   # settles instead of climbing


if __name__ == "__main__":
    unittest.main()
