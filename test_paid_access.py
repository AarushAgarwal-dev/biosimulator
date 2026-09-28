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

    def test_numerical_refinement_remains_public(self):
        """The closed loop's refinement is numerical; a visitor sends engine 'off' and must be served.

        On the live deployment (token required) a visitor's loop sent the Bedrock engine and
        stopped with "requires the deployment access token" at its first refinement round.
        """
        bp = {"type": "ODE", "nodes": [{"id": "X", "initial_value": 0.0}],
              "parameters": {"k": 1.0, "d": 0.5}, "odes": {"X": "k - d*X"},
              "simulation_config": {"t_max": 20}}
        sim = self.client.post("/api/simulate", json={"blueprint": bp, "t_max": 20})
        self.assertEqual(sim.status_code, 200, sim.text)
        response = self.client.post("/api/refine", json={
            "blueprint": bp, "simulation_results": sim.json(),
            "targets": [{"species": "X", "type": "steady_state", "value": 5.0, "tolerance": 0.1}],
            "llm": {"engine": "off"}})
        self.assertEqual(response.status_code, 200, response.text)
        refused = self.client.post("/api/refine", json={
            "blueprint": bp, "simulation_results": sim.json(),
            "targets": [{"species": "X", "type": "steady_state", "value": 5.0, "tolerance": 0.1}],
            "llm": {"engine": "bedrock"}})
        self.assertEqual(refused.status_code, 401, refused.text)

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
        with patch.object(main.nl_compiler, "compile_text") as compile_text, \
             patch.object(main.agent, "parse_biological_text") as parse:
            response = self.client.post(
                "/api/blueprint",
                json={"text": "A activates B.", "llm": {"engine": "bedrock"}},
            )
        self.assertEqual(response.status_code, 401)
        compile_text.assert_not_called()
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
        with patch.object(main.nl_compiler, "compile_text", return_value=expected) as compile_text:
            response = self.client.post(
                "/api/blueprint",
                headers=self.auth(),
                json={"text": "A exists.", "llm": {"engine": "bedrock"}},
            )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json(), expected)
        compile_text.assert_called_once()

    def test_the_legacy_free_form_path_is_still_available_on_request(self):
        expected = {"type": "ODE", "nodes": [{"id": "A"}], "edges": []}
        with patch.object(main.agent, "parse_biological_text", return_value=expected) as parse:
            response = self.client.post(
                "/api/blueprint",
                headers=self.auth(),
                json={"text": "A exists.", "llm": {"engine": "bedrock"}, "compiler": "legacy"},
            )
        self.assertEqual(response.status_code, 200, response.text)
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
        with patch.object(main.nl_compiler, "compile_text", return_value=expected):
            response = self.client.post(
                "/api/blueprint",
                json={"text": "A exists.", "llm": {"engine": "bedrock"}},
            )
        self.assertEqual(response.status_code, 200, response.text)


class PaidAccessFrontendContractTests(unittest.TestCase):
    @staticmethod
    def _app_js():
        root = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(root, "static", "app.js"), encoding="utf-8") as handle:
            return handle.read()

    @staticmethod
    def _function_body(source, signature):
        start = source.index(signature)
        end = source.find("\nfunction ", start + len(signature))
        end2 = source.find("\nasync function ", start + len(signature))
        ends = [e for e in (end, end2) if e != -1]
        return source[start:min(ends) if ends else len(source)]

    def test_requests_do_not_send_a_token_gated_engine_without_the_token(self):
        """A visitor without the token must not provoke a guaranteed 401 on every compile."""
        source = self._app_js()
        body = self._function_body(source, "function requestLlmConfig()")
        self.assertIn("aiBlockedWithoutToken()", body)
        self.assertIn("{ engine: 'off' }", body)
        # The text compile and the closed loop's refinement both use it.
        parse_start = source.index('getElementById("btn-parse-text").addEventListener')
        parse_block = source[parse_start:parse_start + 4000]
        self.assertIn("const llm = requestLlmConfig();", parse_block)
        refine_start = source.index('apiJson("/api/refine"')
        self.assertIn("llm: requestLlmConfig()", source[refine_start:refine_start + 600])

    def test_photo_and_maple_explain_the_token_before_sending(self):
        source = self._app_js()
        photo = self._function_body(source, "async function handleEquationImageFile(file)")
        self.assertIn("aiBlockedWithoutToken()", photo)
        self.assertLess(photo.index("aiBlockedWithoutToken()"), photo.index('"/api/extract-equations"'))
        maple = self._function_body(source, "async function extractMapleParameters()")
        self.assertIn("aiBlockedWithoutToken()", maple)
        self.assertLess(maple.index("aiBlockedWithoutToken()"), maple.index('"/api/maple/extract"'))

    def test_loading_a_preset_clears_every_preset_highlight(self):
        body = self._function_body(self._app_js(), "async function loadPreset(name)")
        for button in ("load-berridge-btn", "load-zhabotinsky-btn", "load-lyashenko-btn"):
            self.assertIn(button, body)

    def test_a_new_model_clears_the_previous_models_target_results(self):
        body = self._function_body(self._app_js(), "function resetModelViews()")
        self.assertIn('getElementById("target-eval-list")', body)
        self.assertIn('innerHTML = ""', body)

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
