"""Access control for operations that can spend the deployment operator's money.

The Render URL is public. Ordinary ODE/PDE/ABM/MPC work remains public and bounded, but
AWS Batch and server-funded Bedrock calls require a deployment token. These tests assert
both halves: paid work cannot slip through, and the guard cannot accidentally turn the
whole research tool into a login wall.
"""
import os
import types
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

import main


class PaidAccessTests(unittest.TestCase):
    TOKEN = "test-only-token-with-enough-entropy-123456"
    ENV_KEYS = (main.PAID_ACCESS_REQUIRED_ENV, main.PAID_ACCESS_TOKEN_ENV)

    def setUp(self):
        self.original = {key: os.environ.get(key) for key in self.ENV_KEYS}
        os.environ[main.PAID_ACCESS_REQUIRED_ENV] = "1"
        os.environ[main.PAID_ACCESS_TOKEN_ENV] = self.TOKEN
        self.client = TestClient(main.app)

    def tearDown(self):
        for key, value in self.original.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def auth(self, token=None):
        return {"Authorization": "Bearer " + (token or self.TOKEN)}

    def test_rule_based_compilation_remains_public(self):
        response = self.client.post(
            "/api/blueprint",
            json={"text": "A activates B.", "llm": {"engine": "off"}},
        )
        self.assertEqual(response.status_code, 200, response.text)

    def test_free_model_compile_remains_public(self):
        response = self.client.post(
            "/api/compile",
            json={"blueprint": {
                "type": "ODE",
                "nodes": [{"id": "A", "initial_value": 1.0}],
                "parameters": {"d": 0.1},
                "odes": {"A": "-d*A"},
            }},
        )
        self.assertEqual(response.status_code, 200, response.text)

    def test_bedrock_blueprint_is_rejected_before_the_provider_is_called(self):
        with patch.object(main.agent, "parse_biological_text") as parse:
            response = self.client.post(
                "/api/blueprint",
                json={"text": "A activates B.", "llm": {"engine": "bedrock"}},
            )
        self.assertEqual(response.status_code, 401)
        parse.assert_not_called()
        self.assertIn("paid service", response.json()["detail"].lower())

    def test_wrong_token_is_rejected(self):
        response = self.client.post(
            "/api/blueprint",
            headers=self.auth("wrong-token"),
            json={"text": "A activates B.", "llm": {"engine": "bedrock"}},
        )
        self.assertEqual(response.status_code, 401)

    def test_correct_token_reaches_the_provider_path(self):
        expected = {"type": "ODE", "nodes": [{"id": "A"}], "edges": []}
        with patch.object(main.agent, "parse_biological_text", return_value=expected) as parse:
            response = self.client.post(
                "/api/blueprint",
                headers=self.auth(),
                json={"text": "A exists.", "llm": {"engine": "bedrock"}},
            )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json(), expected)
        parse.assert_called_once()

    def test_required_but_unconfigured_fails_closed(self):
        os.environ.pop(main.PAID_ACCESS_TOKEN_ENV, None)
        response = self.client.post(
            "/api/blueprint",
            json={"text": "A exists.", "llm": {"engine": "bedrock"}},
        )
        self.assertEqual(response.status_code, 503)
        self.assertIn(main.PAID_ACCESS_TOKEN_ENV, response.json()["detail"])

    def test_weak_server_token_fails_closed(self):
        os.environ[main.PAID_ACCESS_TOKEN_ENV] = "short"
        response = self.client.post(
            "/api/blueprint",
            headers=self.auth("short"),
            json={"text": "A exists.", "llm": {"engine": "bedrock"}},
        )
        self.assertEqual(response.status_code, 503)
        self.assertIn(str(main.PAID_ACCESS_MIN_CHARS), response.json()["detail"])

    def test_compucell3d_is_rejected_before_a_run_is_created(self):
        with patch.object(main.RUNS, "create_run") as create:
            response = self.client.post(
                "/api/runs",
                json={"project": {"selected_approach": "cc3d"}, "start": True},
            )
        self.assertEqual(response.status_code, 401)
        create.assert_not_called()

    def test_free_run_creation_does_not_require_a_token(self):
        record = types.SimpleNamespace(state="failed", run_id="run_free")
        status = {"run_id": "run_free", "approach": "abm", "state": "failed"}
        with patch.object(main.RUNS, "create_run", return_value=record) as create, \
             patch.object(main.RUNS, "status", return_value=status):
            response = self.client.post(
                "/api/runs",
                json={"project": {"selected_approach": "abm"}, "start": False},
            )
        self.assertEqual(response.status_code, 200, response.text)
        create.assert_called_once()

    def test_existing_compucell3d_run_control_is_protected(self):
        status = {"run_id": "run_paid", "approach": "cc3d", "state": "ready"}
        with patch.object(main.RUNS, "status", return_value=status), \
             patch.object(main.RUNS, "start") as start:
            response = self.client.post("/api/runs/run_paid/start", json={})
        self.assertEqual(response.status_code, 401)
        start.assert_not_called()

    def test_every_other_server_paid_entry_point_is_gated(self):
        requests = [
            ("/api/extract-equations", {"image": "aGVsbG8=", "llm": {"engine": "bedrock"}}),
            ("/api/refine", {"blueprint": {}, "simulation_results": {}, "targets": [],
                             "llm": {"engine": "bedrock"}}),
            ("/api/maple/extract", {"param_name": "k", "llm": {"engine": "bedrock"}}),
            ("/api/llm/download", {"model": "llama-3.2-3b"}),
        ]
        for path, body in requests:
            with self.subTest(path=path):
                response = self.client.post(path, json=body)
                self.assertEqual(response.status_code, 401, response.text)

    def test_health_reports_only_boolean_access_state(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["paid_access"], {"required": True, "configured": True})
        self.assertNotIn(self.TOKEN, response.text)

    def test_local_development_is_backward_compatible_when_guard_is_unset(self):
        os.environ.pop(main.PAID_ACCESS_REQUIRED_ENV, None)
        os.environ.pop(main.PAID_ACCESS_TOKEN_ENV, None)
        expected = {"type": "ODE", "nodes": [{"id": "A"}], "edges": []}
        with patch.object(main.agent, "parse_biological_text", return_value=expected):
            response = self.client.post(
                "/api/blueprint",
                json={"text": "A exists.", "llm": {"engine": "bedrock"}},
            )
        self.assertEqual(response.status_code, 200, response.text)


class PaidAccessFrontendContractTests(unittest.TestCase):
    def test_both_surfaces_send_a_bearer_token_from_session_storage(self):
        root = os.path.dirname(os.path.abspath(__file__))
        for name in ("app.js", "workflow.js"):
            with self.subTest(file=name):
                path = os.path.join(root, "static", name)
                with open(path, encoding="utf-8") as handle:
                    source = handle.read()
                self.assertIn("biosim_paid_access_token", source)
                self.assertIn("sessionStorage", source)
                self.assertIn("Authorization", source)
                self.assertIn("Bearer", source)

    def test_both_surfaces_have_a_password_field_for_the_token(self):
        root = os.path.dirname(os.path.abspath(__file__))
        for name in ("index.html", "workflow.html"):
            with self.subTest(file=name):
                path = os.path.join(root, "static", name)
                with open(path, encoding="utf-8") as handle:
                    source = handle.read()
                self.assertIn('id="paid-access-token"', source)
                self.assertIn('type="password"', source)

    def test_render_requires_the_guard_and_never_commits_the_token(self):
        import yaml
        root = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(root, "render.yaml"), encoding="utf-8") as handle:
            document = yaml.safe_load(handle)
        env = {entry["key"]: entry for entry in document["services"][0]["envVars"]}
        self.assertEqual(str(env[main.PAID_ACCESS_REQUIRED_ENV].get("value")), "1")
        token = env[main.PAID_ACCESS_TOKEN_ENV]
        self.assertTrue(token.get("sync") is False)
        self.assertNotIn("value", token)


if __name__ == "__main__":
    unittest.main()
