import copy
import math
import unittest
from unittest import mock

import numpy as np
import sympy as sp

import llm_provider
from simulation_engine import ODEModel, solve_pde
from nl_compiler import (IR_SCHEMA, compile_ir, compile_text, extract_ir_llm,
                         extract_ir_rules, validate_ir, verify)


def species(sid, initial=1.0, diffusion=None, source="default"):
    return {"id": sid, "name": sid, "initial": initial, "initial_source": source,
            "diffusion": diffusion, "role": "state"}


def parameter(name, value=1.0, source="default"):
    return {"name": name, "value": value, "source": source,
            "unit": "arbitrary", "meaning": name}


def base_ir(species_list, params, processes, model_type="ode"):
    return {"model_type": model_type, "time_unit": "min", "t_end": 10.0,
            "species": species_list, "parameters": params, "stimuli": [],
            "processes": processes, "assumptions": [], "unmodeled": []}


class SchemaAndValidationTests(unittest.TestCase):
    def test_schema_is_strict(self):
        self.assertFalse(IR_SCHEMA["additionalProperties"])
        self.assertEqual(set(IR_SCHEMA["required"]), {"model_type", "time_unit", "t_end", "species",
                                                       "parameters", "stimuli", "processes", "assumptions", "unmodeled"})

    def test_unknown_reference_and_unused_species(self):
        ir = base_ir([species("X"), species("Y")], [parameter("k")], [
            {"id": "p1", "kind": "production", "target": "Z", "k": "missing",
             "evidence": "X is produced", "assumed": False}])
        errors = validate_ir(ir, "X is produced")
        self.assertTrue(any("unknown species 'Z'" in e for e in errors))
        self.assertTrue(any("unknown parameter 'missing'" in e for e in errors))
        self.assertTrue(any("does not participate" in e for e in errors))

    def test_duplicate_ids_and_nonpositive_parameter(self):
        ir = base_ir([species("X"), species("X")], [parameter("k", 0)], [])
        errors = validate_ir(ir, "")
        self.assertTrue(any("Duplicate species" in e for e in errors))
        self.assertTrue(any("positive finite" in e for e in errors))

    def test_evidence_check(self):
        ir = base_ir([species("X")], [parameter("k")], [
            {"id": "p", "kind": "degradation", "species": "X", "k": "k",
             "evidence": "Y binds Z", "assumed": False}])
        self.assertTrue(any("evidence is not supported" in e for e in validate_ir(ir, "X is degraded")))

    def test_custom_unknown_symbol(self):
        ir = base_ir([species("X")], [parameter("k")], [
            {"id": "p", "kind": "custom", "reactants": [{"species": "X", "stoich": 1}],
             "products": [], "rate": "k*X*q", "evidence": "X is removed", "assumed": False}])
        self.assertTrue(any("unknown symbols" in e for e in validate_ir(ir, "X is removed")))


class CompilerMathTests(unittest.TestCase):
    def test_production_and_first_order_degradation(self):
        ir = base_ir([species("X", 0)], [parameter("kp", 2), parameter("kd", .5)], [
            {"id": "p1", "kind": "production", "target": "X", "k": "kp", "evidence": "X produced", "assumed": False},
            {"id": "p2", "kind": "degradation", "species": "X", "k": "kd", "evidence": "X degraded", "assumed": False}])
        model = ODEModel(compile_ir(ir))
        self.assertEqual(sp.simplify(model.deriv_exprs["X"] - (sp.Symbol("kp") - sp.Symbol("kd") * sp.Symbol("X"))), 0)
        self.assertAlmostEqual(model.simulate(30, 100)["species"]["X"][-1], 4.0, places=4)

    def test_half_life(self):
        ir = base_ir([species("X", 2)], [], [
            {"id": "p", "kind": "degradation", "species": "X", "half_life": 20,
             "evidence": "X half-life 20", "assumed": False}])
        model = ODEModel(compile_ir(ir)); value = float(model.deriv_exprs["X"].subs({sp.Symbol("X"): 1}))
        self.assertAlmostEqual(value, -math.log(2) / 20, places=12)

    def test_conversion_conserves_mass(self):
        ir = base_ir([species("A"), species("B", 0)], [parameter("k", .2)], [
            {"id": "p", "kind": "conversion", "reactants": [{"species": "A", "stoich": 1}],
             "products": [{"species": "B", "stoich": 1}], "k": "k", "evidence": "A converts to B", "assumed": False}])
        model = ODEModel(compile_ir(ir))
        self.assertEqual(sp.simplify(model.deriv_exprs["A"] + model.deriv_exprs["B"]), 0)
        sim = model.simulate(10, 50)
        self.assertLess(np.max(np.abs(np.asarray(sim["species"]["A"]) + np.asarray(sim["species"]["B"]) - 1)), 1e-8)

    def test_stoichiometric_conversion(self):
        ir = base_ir([species("A"), species("B", 0), species("C", 0)], [parameter("k", .2)], [
            {"id": "p", "kind": "conversion", "reactants": [{"species": "A", "stoich": 1}],
             "products": [{"species": "B", "stoich": 1}, {"species": "C", "stoich": 1}],
             "k": "k", "evidence": "A becomes B and C", "assumed": False}])
        m = ODEModel(compile_ir(ir))
        self.assertEqual(sp.simplify(m.deriv_exprs["A"] + m.deriv_exprs["B"]), 0)
        self.assertEqual(sp.simplify(m.deriv_exprs["A"] + m.deriv_exprs["C"]), 0)

    def test_binding_conservation(self):
        ir = base_ir([species("L"), species("R"), species("C", 0)], [parameter("kon"), parameter("koff", .1)], [
            {"id": "p", "kind": "binding", "reactants": [{"species": "L", "stoich": 1}, {"species": "R", "stoich": 1}],
             "complex": "C", "kon": "kon", "koff": "koff", "evidence": "L binds R to form C", "assumed": False}])
        m = ODEModel(compile_ir(ir))
        self.assertEqual(sp.simplify(m.deriv_exprs["L"] + m.deriv_exprs["C"]), 0)
        self.assertEqual(sp.simplify(m.deriv_exprs["R"] + m.deriv_exprs["C"]), 0)

    def test_michaelis_menten_enzyme_not_consumed(self):
        ir = base_ir([species("E"), species("S"), species("P", 0)], [parameter("kcat"), parameter("Km")], [
            {"id": "p", "kind": "conversion", "reactants": [{"species": "S", "stoich": 1}],
             "products": [{"species": "P", "stoich": 1}], "enzyme": "E", "Km": "Km", "k": "kcat",
             "evidence": "E catalyzes S to P", "assumed": False}])
        m = ODEModel(compile_ir(ir))
        self.assertEqual(m.deriv_exprs["E"], 0)
        self.assertEqual(sp.simplify(m.deriv_exprs["S"] + m.deriv_exprs["P"]), 0)
        self.assertIn(sp.Symbol("E"), m.deriv_exprs["P"].free_symbols)

    def test_hill_activation_and_repression(self):
        regs = [{"species": "A", "effect": "activate", "K": "K", "n": "n"},
                {"species": "R", "effect": "repress", "K": "K2", "n": "n2"}]
        ir = base_ir([species("A"), species("R"), species("X", 0)],
                     [parameter("k"), parameter("K"), parameter("n", 2), parameter("K2"), parameter("n2", 2),
                      parameter("ka", .1), parameter("kr", .1)], [
            {"id": "p", "kind": "production", "target": "X", "k": "k", "regulators": regs,
             "evidence": "A activates X and R represses X", "assumed": False},
            {"id": "pa", "kind": "degradation", "species": "A", "k": "ka", "evidence": "A turnover", "assumed": True, "reason": "bounded"},
            {"id": "pr", "kind": "degradation", "species": "R", "k": "kr", "evidence": "R turnover", "assumed": True, "reason": "bounded"}])
        m = ODEModel(compile_ir(ir)); subs = {s: 1 for s in m.deriv_exprs["X"].free_symbols}
        self.assertGreater(float(sp.diff(m.deriv_exprs["X"], sp.Symbol("A")).subs(subs)), 0)
        self.assertLess(float(sp.diff(m.deriv_exprs["X"], sp.Symbol("R")).subs(subs)), 0)

    def test_flux_names_are_readable(self):
        ir = base_ir([species("A"), species("B")], [parameter("k")], [
            {"id": "p1", "kind": "conversion", "reactants": [{"species": "A", "stoich": 1}],
             "products": [{"species": "B", "stoich": 1}], "k": "k", "evidence": "A to B", "assumed": False}])
        bp = compile_ir(ir)
        self.assertTrue(any(name.startswith("conv_A_to_B") for name in bp["fluxes"]))
        self.assertEqual(bp["_compiler"], "nl_compiler/ir-v1")
        self.assertIn("_provenance", bp); self.assertIn("_process_table", bp)


class VerificationTests(unittest.TestCase):
    def test_sign_check_passes(self):
        ir = base_ir([species("A"), species("X", .1)], [parameter("k"), parameter("K"), parameter("n", 2), parameter("kd", .1)], [
            {"id": "p", "kind": "production", "target": "X", "k": "k",
             "regulators": [{"species": "A", "effect": "activate", "K": "K", "n": "n"}],
             "evidence": "A activates X", "assumed": False},
            {"id": "d", "kind": "degradation", "species": "A", "k": "kd", "evidence": "", "assumed": True, "reason": "bounded"}])
        report = verify(compile_ir(ir), ir, "A activates X")
        self.assertTrue(report["sign_checks"][0]["ok"])

    def test_sign_check_catches_deliberately_wrong_rate(self):
        ir = base_ir([species("A"), species("X", .1)], [parameter("k"), parameter("K"), parameter("n", 2), parameter("kd", .1)], [
            {"id": "p", "kind": "production", "target": "X", "k": "k",
             "regulators": [{"species": "A", "effect": "activate", "K": "K", "n": "n"}],
             "evidence": "A activates X", "assumed": False},
            {"id": "d", "kind": "degradation", "species": "A", "k": "kd", "evidence": "", "assumed": True, "reason": "bounded"}])
        bp = compile_ir(ir)
        flux = next(name for name in bp["fluxes"] if name.startswith("prod_X"))
        bp["fluxes"][flux] = "k*K**n/(K**n+A**n)"
        report = verify(bp, ir, "A activates X")
        self.assertFalse(report["ok"]); self.assertTrue(any("Sign mismatch" in e for e in report["errors"]))

    def test_compile_text_bad_input_never_raises(self):
        result = compile_text("Weather is pleasant.")
        self.assertIn("validation_errors", result)


class RuleExtractionTests(unittest.TestCase):
    CASES = [
        ("A activates B.", "production", "activate"),
        ("A represses B.", "production", "repress"),
        ("X represses its own transcription.", "production", "repress"),
        ("X activates its own expression.", "production", "activate"),
        ("X is degraded with a half-life of 20 minutes.", "degradation", None),
        ("X is produced at a constant rate of 2.", "production", None),
        ("L binds R reversibly to form C with kon 1 and koff 0.1.", "binding", None),
        ("A is converted to B at rate 0.2.", "conversion", None),
        ("E catalyzes the conversion of S to P.", "conversion", None),
        ("Mdm2 promotes degradation of p53.", "degradation", "activate"),
        ("MEK phosphorylates ERK.", "production", "activate"),
        ("A inhibits B.", "production", "repress"),
        ("A upregulates B.", "production", "activate"),
        ("A downregulates B.", "production", "repress"),
        ("A stimulates B.", "production", "activate"),
    ]

    def test_common_sentence_patterns(self):
        for text, kind, effect in self.CASES:
            with self.subTest(text=text):
                ir = extract_ir_rules(text)
                matches = [p for p in ir["processes"] if p["kind"] == kind and not p.get("assumed")]
                self.assertTrue(matches, ir)
                if effect:
                    self.assertTrue(any(r.get("effect") == effect for p in matches for r in p.get("regulators", [])), ir)
                self.assertEqual(validate_ir(ir, text), [])

    def test_negation_does_not_create_edge(self):
        ir = extract_ir_rules("A does not activate B. A is degraded at rate 0.1.")
        self.assertFalse(any(p.get("target") == "B" for p in ir["processes"]))

    def test_initial_and_diffusion_numbers(self):
        ir = extract_ir_rules("A starts at 2. A diffuses at 0.05 and activates B. B diffuses at 1.")
        by_id = {s["id"]: s for s in ir["species"]}
        self.assertEqual(by_id["A"]["initial"], 2)
        self.assertEqual(by_id["A"]["diffusion"], .05)
        self.assertEqual(by_id["B"]["diffusion"], 1)
        self.assertEqual(ir["model_type"], "reaction_diffusion")

    def test_mutual_repression_breaks_symmetry(self):
        ir = extract_ir_rules("A represses B. B represses A.")
        vals = [s["initial"] for s in ir["species"]]
        self.assertEqual(len(set(vals)), 2)
        self.assertTrue(any("symmetry" in a for a in ir["assumptions"]))


class LLMPathTests(unittest.TestCase):
    def valid_ir(self):
        return base_ir([species("X")], [parameter("kd", .1)], [
            {"id": "p", "kind": "degradation", "species": "X", "k": "kd",
             "evidence": "X is degraded", "assumed": False}])

    def test_generate_structured_valid(self):
        fn = mock.Mock(return_value=self.valid_ir())
        with mock.patch.object(llm_provider, "generate_structured", fn, create=True):
            got = extract_ir_llm("X is degraded", object())
        self.assertEqual(got["species"][0]["id"], "X")
        self.assertEqual(fn.call_count, 1)

    def test_invalid_then_repaired(self):
        invalid = self.valid_ir(); invalid["processes"][0]["species"] = "Y"
        fn = mock.Mock(side_effect=[invalid, self.valid_ir()])
        with mock.patch.object(llm_provider, "generate_structured", fn, create=True):
            got = extract_ir_llm("X is degraded", object())
        self.assertEqual(fn.call_count, 2)
        self.assertIn("1 repair", got["_extraction_notice"])

    def test_garbage_falls_back_honestly(self):
        fn = mock.Mock(return_value="garbage")
        with mock.patch.object(llm_provider, "generate_structured", fn, create=True):
            got = extract_ir_llm("A activates B", object())
        self.assertIn("deterministic rules used", got["_extraction_notice"])
        self.assertTrue(any(p.get("target") == "B" for p in got["processes"]))

    def test_generate_json_feature_fallback(self):
        with mock.patch.object(llm_provider, "generate_json", return_value=self.valid_ir()) as fn:
            absent = getattr(llm_provider, "generate_structured", None)
            if absent is not None:
                with mock.patch.object(llm_provider, "generate_structured", None):
                    got = extract_ir_llm("X is degraded", object())
            else:
                got = extract_ir_llm("X is degraded", object())
        self.assertEqual(got["species"][0]["id"], "X"); self.assertEqual(fn.call_count, 1)


class PDETests(unittest.TestCase):
    def test_pde_numeric_reactions_and_stable_dt(self):
        ir = base_ir([species("A", 1, .05), species("I", 1, 1.0)],
                     [parameter("ka"), parameter("ki"), parameter("K"), parameter("n", 2),
                      parameter("dA"), parameter("dI")], [
            {"id": "pa", "kind": "production", "target": "A", "k": "ka",
             "regulators": [{"species": "I", "effect": "repress", "K": "K", "n": "n"}],
             "evidence": "I inhibits A", "assumed": False},
            {"id": "pi", "kind": "production", "target": "I", "k": "ki",
             "regulators": [{"species": "A", "effect": "activate", "K": "K", "n": "n"}],
             "evidence": "A activates I", "assumed": False},
            {"id": "da", "kind": "degradation", "species": "A", "k": "dA", "evidence": "", "assumed": True, "reason": "bounded"},
            {"id": "di", "kind": "degradation", "species": "I", "k": "dI", "evidence": "", "assumed": True, "reason": "bounded"}], "reaction_diffusion")
        bp = compile_ir(ir)
        self.assertEqual(bp["type"], "PDE")
        self.assertNotRegex(" ".join(bp["spatial"]["reactions"].values()), r"\b(?:ka|ki|K|n|dA|dI)\b")
        dt = bp["simulation_config"]["dt"]
        self.assertLessEqual(dt * 1.0 * 2, .5)
        result = solve_pde({**bp["spatial"], "x_grid": 8, "y_grid": 8}, bp["spatial"]["reactions"],
                           {"A": {"type": "uniform", "base_value": 1}, "I": {"type": "uniform", "base_value": 1}},
                           t_max=dt * 2, dt=dt, save_every=1)
        self.assertTrue(result["stability"]["diffusion_number"] <= .5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
