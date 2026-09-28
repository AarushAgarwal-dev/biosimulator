"""The Umulis et al. (2010) BMP reaction network, compiled from a plain-language description.

The description below uses only sentences a biologist would write (secretion, reversible binding,
Sog cleavage, degradation, feedback production). Compiled by the deterministic rule-based
compiler it must reproduce the embryo module's own kinetics (bmp_embryo._reaction, SBP mechanism,
Tables S1/S8) exactly, with the receptor total conserved. Getting there needed four compiler
fixes, each pinned separately below.
"""
import unittest

import numpy as np

import bmp_embryo
import nl_compiler as nc
from simulation_engine import ODEModel

UMULIS_2010_TEXT = """BMP is produced at rate 1.
Sog is produced at rate 400.
Tsg is produced at rate 48.
Sog binds Tsg reversibly to form SogTsg with kon 0.6 and koff 3.6.
SogTsg binds BMP reversibly to form SogTsgBMP with kon 11.4 and koff 0.36.
SogTsgBMP is converted to BMP and Tsg at rate 31.35.
Tsg is degraded at rate 0.05.
BMP binds SBP reversibly to form SBPBMP with kon 2 and koff 4.
BMP binds Tkv reversibly to form BMPTkv with kon 0.048 and koff 8.
SBPBMP binds Tkv reversibly to form SBPBMPTkv with kon 1 and koff 20.
BMPTkv binds SBP reversibly to form SBPBMPTkv with kon 0.25 and koff 20.
BMPTkv activates the production of SBP with maximum rate 24.45, half-saturation constant 61.83 and Hill coefficient 2.
SBP is degraded at rate 0.03.
SBPBMP is degraded at rate 0.03.
BMPTkv is converted to Tkv at rate 0.03.
SBPBMPTkv is converted to Tkv at rate 0.03.
Tkv starts at 394.3.
Simulate for 60 minutes."""

# bmp_embryo species -> the names used in the description
NAMES = {"B": "BMP", "S": "Sog", "T": "Tsg", "I": "SogTsg", "IB": "SogTsgBMP", "C": "SBP",
         "BC": "SBPBMP", "BCR": "SBPBMPTkv", "BR": "BMPTkv"}


def _processes(bp):
    return (bp.get("_ir") or {}).get("processes", [])


class Umulis2010FromTextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bp = nc.compile_text(UMULIS_2010_TEXT, None)

    def test_it_compiles_verified_with_nothing_assumed(self):
        self.assertFalse(self.bp.get("validation_errors"), self.bp.get("validation_errors"))
        self.assertTrue(self.bp["_verification"]["ok"], self.bp["_verification"]["errors"])
        self.assertEqual(self.bp["_verification"]["warnings"], [])
        self.assertEqual([p["id"] for p in _processes(self.bp) if p.get("assumed")], [])
        self.assertEqual(sorted(n["id"] for n in self.bp["nodes"]), sorted(list(NAMES.values()) + ["Tkv"]))

    def test_the_equations_are_the_papers(self):
        model = ODEModel(self.bp)
        names = list(model.node_ids)
        pvals = [float(self.bp["parameters"][k]) for k in model.param_names]
        p = bmp_embryo._base_params("sbp", "wt", None, 400.0, False, "umulis2010")
        production = (np.array([p["phi_B"]]), np.array([p["phi_S"]]), np.array([p["phi_T"]]))
        rng = np.random.default_rng(4)
        for _ in range(100):
            st = {k: rng.uniform(0, 40) for k in bmp_embryo.SPECIES}
            state = {NAMES[k]: st[k] for k in NAMES}
            state["Tkv"] = p["Rtot"] - st["BR"] - st["BCR"]           # R = Rtot - BR - BCR
            f = dict(zip(names, model._f_lambdified(0.0, np.array([state[n] for n in names]), pvals)))
            ref = bmp_embryo._reaction(np.array([[st[k]] for k in bmp_embryo.SPECIES]), p, "sbp", production)
            for i, k in enumerate(bmp_embryo.SPECIES):
                self.assertAlmostEqual(float(f[NAMES[k]]) / max(1.0, abs(float(ref[i][0]))),
                                       float(ref[i][0]) / max(1.0, abs(float(ref[i][0]))), places=9, msg=k)
            # The receptor is a species whose total the text conserves (internalisation returns it).
            self.assertAlmostEqual(float(f["Tkv"] + f["BMPTkv"] + f["SBPBMPTkv"]), 0.0, places=9)


class CompilerFixTests(unittest.TestCase):
    def test_a_conversion_can_release_two_products(self):
        ir = nc.extract_ir_rules("IB starts at 1. IB is converted to B and T at rate 2.")
        conv = [p for p in ir["processes"] if p["kind"] == "conversion"][0]
        self.assertEqual([s["species"] for s in conv["products"]], ["B", "T"])

    def test_a_following_clause_is_not_read_as_a_product(self):
        ir = nc.extract_ir_rules("X starts at 1. X is converted to Y and Z is degraded at rate 0.1.")
        conv = [p for p in ir["processes"] if p["kind"] == "conversion"][0]
        self.assertEqual([s["species"] for s in conv["products"]], ["Y"])
        self.assertTrue(any(p["kind"] == "degradation" and p["species"] == "Z" for p in ir["processes"]))

    def test_stated_constants_of_regulated_production_are_used(self):
        ir = nc.extract_ir_rules("A starts at 1. A activates the production of X with maximum rate 5, "
                                 "half-saturation constant 2 and Hill coefficient 3. X is degraded at rate 0.1.")
        prod = [p for p in ir["processes"] if p["kind"] == "production" and p["target"] == "X"][0]
        values = {q["name"]: (q["value"], q["source"]) for q in ir["parameters"]}
        reg = prod["regulators"][0]
        self.assertEqual(values[prod["k"]], (5.0, "text"))
        self.assertEqual(values[reg["K"]], (2.0, "text"))
        self.assertEqual(values[reg["n"]], (3.0, "text"))

    def test_no_invented_turnover_when_the_text_routes_a_species_to_a_loss(self):
        routed = nc.extract_ir_rules("Y starts at 1. X is produced at rate 1. X binds Y reversibly to form Z "
                                     "with kon 1 and koff 0.1. Z is degraded at rate 0.2.")
        self.assertFalse([p for p in routed["processes"] if p.get("assumed") and p.get("species") == "X"])
        dead_end = nc.extract_ir_rules("Y starts at 1. X is produced at rate 1. X binds Y reversibly to form Z "
                                       "with kon 1 and koff 0.1.")
        self.assertTrue([p for p in dead_end["processes"] if p.get("assumed") and p.get("species") == "X"],
                        "a produced species with no route to any loss still needs disclosed turnover")

    def test_a_regulation_is_sign_checked_on_its_own_rate_law(self):
        # BMPTkv activates SBP production AND binds SBP: the whole dSBP/dt falls with BMPTkv,
        # but the stated activation is true of the production term.
        checks = [c for c in nc.compile_text(UMULIS_2010_TEXT, None)["_verification"]["sign_checks"]
                  if c.get("source") == "BMPTkv" and c.get("target") == "SBP"]
        self.assertTrue(checks and all(c["ok"] for c in checks))
        self.assertTrue(all(c.get("scope") == "process rate law" for c in checks))

    def test_hill_is_not_reported_as_a_missing_entity(self):
        bp = nc.compile_text("A starts at 1. A activates X with Hill coefficient 3. X is degraded at rate 0.1.", None)
        self.assertFalse([w for w in bp["_verification"]["warnings"] if "Hill" in w])

    def test_long_descriptions_get_a_larger_output_budget(self):
        self.assertEqual(nc._ir_token_budget("A activates B. B is degraded at rate 1."), nc.IR_MAX_TOKENS)
        self.assertGreater(nc._ir_token_budget(UMULIS_2010_TEXT), 10000)
        self.assertLessEqual(nc._ir_token_budget(UMULIS_2010_TEXT * 5), nc.IR_MAX_TOKENS_CEILING)


if __name__ == "__main__":
    unittest.main()
