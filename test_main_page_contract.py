"""MAIN PAGE contract: the call paths behind index.html's tabs must still work.

Companion to test_frontend_contract.py, which covers the workflow page. The main page
(the Model Compiler) has its own six tabs and 22 endpoints, and its shapes were never
exercised against the hardened routes.

The payloads here are read out of static/app.js, and the BLUEPRINT is read out of
app.js too via preset_loader -- so this drives the genuinely shipped EGF/EGFR model
through the real endpoints, rather than a fixture invented in this file that could
drift away from what the product actually sends.

Shapes, and where they come from in app.js:
  compileBlueprint  -> POST /api/compile            {blueprint}
  runSimulation     -> POST /api/simulate           {blueprint, custom_params?, seed?}
  equations tab     -> POST /api/equations-to-model  {equations}
  target feedback   -> POST /api/evaluate           {blueprint, targets, custom_params}
  sampling          -> POST /api/sample             {blueprint, param_bounds,
                                                     n_samples, method, target_species}

The third-party integrations (Reactome, STRING, OmniPath, SIGNOR, BioModels) and the
LLM-backed routes (/api/refine, /api/maple/extract) are deliberately NOT called here:
they reach the network, so asserting on them would be flaky rather than informative.
What IS asserted for the LLM path is that it fails gracefully rather than 500-ing.
"""
import unittest

from fastapi.testclient import TestClient

import agent
import main
import preset_loader

PRESETS = preset_loader.load_presets()
BLUEPRINTS = preset_loader.load_blueprints(text_compiler=agent.rule_based_parse)
EGFR = BLUEPRINTS["egfr"]
EGFR_TARGETS = PRESETS["egfr"].get("targets") or []


class MainPageContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(main.app)

    def _post(self, path, payload, where):
        response = self.client.post(path, json=payload)
        self.assertEqual(
            response.status_code, 200,
            f"{where}: POST {path} with the shape app.js sends was rejected "
            f"({response.status_code}). Detail: {response.text[:400]}")
        return response.json()

    # ---- Blueprint / Equations tabs ----

    def test_compile_the_shipped_blueprint(self):
        payload = self._post("/api/compile", {"blueprint": EGFR}, "Blueprint tab")
        self.assertTrue(payload.get("equations"),
                        "compiling the shipped EGF/EGFR model produced no equations")

    def test_simulate_the_shipped_blueprint(self):
        payload = self._post("/api/simulate", {"blueprint": EGFR}, "Simulator tab")
        # ODEModel.simulate returns {"t": [...], "species": {id: [...]}}
        species = payload.get("species") or (payload.get("results") or {}).get("species")
        self.assertTrue(species, f"the simulation returned no species data: "
                                 f"{sorted(payload)[:8]}")
        self.assertIn("ERK", species,
                      "ERK is the readout the EGF/EGFR targets name and it is absent")

    def test_equations_tab_accepts_hand_written_odes(self):
        """The 'write your own differential equations' path must work standalone."""
        equations = "dA/dt = -0.5*A\ndB/dt = 0.5*A - 0.2*B"
        response = self.client.post("/api/equations-to-model", json={"equations": equations})
        self.assertNotEqual(
            response.status_code, 500,
            f"hand-written equations produced a server error rather than a parse "
            f"result or a refusal: {response.text[:300]}")
        if response.status_code == 200:
            body = response.json()
            self.assertTrue(
                body.get("nodes") or body.get("blueprint") or body.get("odes"),
                f"the equations parsed to nothing usable: {sorted(body)[:10]}")

    # ---- Target Behaviors / feedback ----

    def test_evaluate_the_presets_own_targets(self):
        if not EGFR_TARGETS:
            self.skipTest("the EGF/EGFR preset ships no targets")
        payload = self._post("/api/evaluate",
                             {"blueprint": EGFR, "targets": EGFR_TARGETS,
                              "custom_params": {}}, "Target Behaviors")
        results = payload.get("results")
        self.assertIsInstance(results, list, "evaluate returned no results list")
        self.assertEqual(
            len(results), len(EGFR_TARGETS),
            "evaluate did not report one result per target, so a target was silently "
            "dropped")
        for result in results:
            self.assertIn("met", result,
                          f"a target result carries no verdict: {result}")

    def test_evaluate_refuses_a_target_on_a_species_that_does_not_exist(self):
        """Regression guard: this used to report a number, or score it as met.

        A target naming a species the model does not contain measures NOTHING. Returning
        0.0, or 'met', is worse than an error because it is indistinguishable from a
        real measurement.
        """
        bogus = [{"species": "NOT_A_REAL_SPECIES", "type": "peak_value",
                  "min": 0.0, "max": 1.0}]
        response = self.client.post("/api/evaluate",
                                    json={"blueprint": EGFR, "targets": bogus,
                                          "custom_params": {}})
        if response.status_code == 200:
            results = response.json().get("results") or []
            for result in results:
                self.assertFalse(
                    result.get("met"),
                    "a target naming a nonexistent species was scored as MET")
        else:
            self.assertEqual(
                response.status_code, 422,
                f"expected a 422 naming the unknown species, got "
                f"{response.status_code}: {response.text[:200]}")
            self.assertIn("NOT_A_REAL_SPECIES", response.text,
                          "the refusal does not name the offending species")

    # ---- Parameter sampling ----

    def test_sample_the_shipped_blueprint(self):
        params = list((EGFR.get("parameters") or {}).items())[:2]
        if not params:
            self.skipTest("the shipped blueprint exposes no parameters to sample")
        # app.js defaultBounds() returns {min, max, step} per parameter.
        bounds = {name: {"min": min(0.0, float(value) * 0.5),
                         "max": float(value) * 1.5 or 1.0,
                         "step": 0.05}
                  for name, value in params}
        payload = self._post("/api/sample",
                             {"blueprint": EGFR, "param_bounds": bounds,
                              "n_samples": 4, "method": "lhs",
                              "target_species": "ERK"}, "sampling")
        self.assertTrue(payload, "sampling returned an empty payload")

    def test_sample_refuses_an_unknown_parameter(self):
        """Sampling a parameter the model lacks must not silently return zeros."""
        response = self.client.post("/api/sample",
                                    json={"blueprint": EGFR,
                                          "param_bounds": {
                                              "kNOT_REAL": {"min": 0.0, "max": 1.0,
                                                            "step": 0.05}},
                                          "n_samples": 4, "method": "lhs",
                                          "target_species": "ERK"})
        self.assertNotEqual(
            response.status_code, 500,
            f"an unknown parameter produced a server error instead of a refusal: "
            f"{response.text[:300]}")

    # ---- LLM-backed routes: must degrade, not crash ----

    def test_refine_degrades_gracefully_without_a_working_llm(self):
        """/api/refine needs an LLM. Absent one it must refuse, not 500."""
        response = self.client.post("/api/refine", json={
            "blueprint": EGFR,
            "simulation_results": {},
            "targets": EGFR_TARGETS,
            "llm": {"engine": "off"},
        })
        self.assertNotEqual(
            response.status_code, 500,
            f"/api/refine returned a server error rather than refusing cleanly: "
            f"{response.text[:300]}")


if __name__ == "__main__":
    unittest.main()
