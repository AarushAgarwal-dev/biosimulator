"""Access policy for the Purdue GenAI Studio key.

The key belongs to one Purdue account. On a deployment that requires the paid-access token,
every route that can spend it must require that token unless the operator opted into public use
(BIOSIM_PURDUE_PUBLIC=1), and every AI request is rate-limited per client and globally.
Nothing here reaches the network: the provider is patched out.
"""
import os
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

import main


class PurdueAccessPolicyTests(unittest.TestCase):
    TOKEN = "test-only-token-with-enough-entropy-123456"
    KEYS = (main.PAID_ACCESS_REQUIRED_ENV, main.PAID_ACCESS_TOKEN_ENV, main.PURDUE_PUBLIC_ENV,
            "BIOSIM_LLM_RATE_PER_MIN", "BIOSIM_LLM_GLOBAL_RATE_PER_MIN")

    def setUp(self):
        self.saved = {k: os.environ.get(k) for k in self.KEYS}
        os.environ[main.PAID_ACCESS_REQUIRED_ENV] = "1"
        os.environ[main.PAID_ACCESS_TOKEN_ENV] = self.TOKEN
        os.environ.pop(main.PURDUE_PUBLIC_ENV, None)
        main._LIMITER = main._SlidingWindowLimiter()
        self.client = TestClient(main.app)

    def tearDown(self):
        for k, v in self.saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        main._LIMITER = main._SlidingWindowLimiter()

    def test_purdue_compile_needs_the_token_when_not_public(self):
        with patch.object(main.nl_compiler, "compile_text") as compile_text:
            r = self.client.post("/api/blueprint", json={"text": "A activates B.", "llm": {"engine": "purdue"}})
        self.assertEqual(r.status_code, 401, r.text)
        compile_text.assert_not_called()

    def test_purdue_maple_needs_the_token_when_not_public(self):
        r = self.client.post("/api/maple/extract", json={"param_name": "k", "llm": {"engine": "purdue"}})
        self.assertEqual(r.status_code, 401, r.text)

    def test_purdue_photo_needs_the_token_when_not_public(self):
        r = self.client.post("/api/extract-equations", json={"image": "aGVsbG8=", "llm": {"engine": "purdue"}})
        self.assertEqual(r.status_code, 401, r.text)

    def test_rule_based_compile_stays_public(self):
        r = self.client.post("/api/blueprint", json={"text": "A activates B.", "llm": {"engine": "off"}})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json().get("_compiler"), "nl_compiler/ir-v1")

    def test_public_mode_is_rate_limited_per_client(self):
        os.environ[main.PURDUE_PUBLIC_ENV] = "1"
        os.environ["BIOSIM_LLM_RATE_PER_MIN"] = "2"
        fake = {"type": "ODE", "nodes": [{"id": "A", "initial_value": 1.0}], "odes": {"A": "-A"}, "parameters": {}}
        with patch.object(main.nl_compiler, "compile_text", return_value=fake):
            codes = [self.client.post("/api/blueprint", json={"text": "A decays.", "llm": {"engine": "purdue"}}).status_code
                     for _ in range(3)]
        self.assertEqual(codes, [200, 200, 429])

    def test_the_browser_can_never_supply_the_purdue_key(self):
        os.environ[main.PURDUE_PUBLIC_ENV] = "1"
        seen = {}

        def capture(text, config):
            seen.update(config or {})
            return {"validation_errors": ["stub"]}

        with patch.object(main.nl_compiler, "compile_text", side_effect=capture):
            self.client.post("/api/blueprint", json={"text": "A decays.",
                                                     "llm": {"engine": "purdue", "api_key": "sk-from-browser"}})
        self.assertNotIn("api_key", seen)

    def test_env_endpoint_reports_policy_without_secrets(self):
        r = self.client.get("/api/llm/env")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        for key in ("purdue_env_ready", "purdue_model", "purdue_public", "paid_access"):
            self.assertIn(key, body)
        key_value = os.environ.get("PURDUE_GENAI_API_KEY") or ""
        if key_value:
            self.assertNotIn(key_value, r.text)
        self.assertNotIn(self.TOKEN, r.text)


class BMPRouteTests(unittest.TestCase):
    def setUp(self):
        main._LIMITER = main._SlidingWindowLimiter()
        self.client = TestClient(main.app)

    def test_cross_section_route_solves_and_bounds_input(self):
        r = self.client.post("/api/bmp/cross-section", json={"perturbation": "wt", "save_times": [30, 60]})
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertGreater(body["readouts"]["final"]["dm_value"], 10.0)
        self.assertLess(body["diagnostics"]["ligand_mass_balance_relative_error"], 1e-3)
        self.assertEqual(self.client.post("/api/bmp/cross-section", json={"t_end": 5000}).status_code, 422)
        self.assertEqual(self.client.post("/api/bmp/cross-section", json={"mechanism": "bogus"}).status_code, 422)

    def test_surface_route_is_size_capped(self):
        r = self.client.post("/api/bmp/surface", json={"nu": 200, "nv": 200})
        self.assertEqual(r.status_code, 422)

    def test_info_and_validation_routes(self):
        info = self.client.get("/api/bmp/info").json()
        self.assertIn("equations_latex", info)
        self.assertTrue(all(entry.get("source") for entry in info["parameter_table"].values()))
        report = self.client.get("/api/bmp/validation").json()
        self.assertIn("checks", report)
        self.assertTrue(report["required_pass"])


if __name__ == "__main__":
    unittest.main()
