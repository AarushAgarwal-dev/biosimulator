"""The deployment surface: a health check that tells the truth, and a port that Render uses.

A Render service for this app already existed before any of this was written, configured
through the dashboard. That makes these tests worth MORE, not less: the settings the app
requires were recorded nowhere, so nothing could detect a drift between what the app needs
and what the service does.

/api/health is not decoration. A deployed instance differs from a laptop in exactly the ways
that fail silently -- no CompuCell3D backend, no LLM credential, run history that evaporates
on restart -- so the health check reports those instead of a bare {"ok": true}, and these
tests hold it to that.
"""
import os
import unittest

from fastapi.testclient import TestClient

import main


class HealthEndpointTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(main.app)

    def test_it_answers_200_so_render_does_not_restart_the_service(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json().get("status"), "ok")

    def test_it_reports_every_approach_and_why_an_absent_one_is_absent(self):
        body = self.client.get("/api/health").json()
        found = {a.get("approach_id") for a in body.get("approaches") or []}
        for expected in ("abm", "cc3d", "mpc"):
            self.assertIn(expected, found,
                          f"health does not report the {expected} approach; a deployment "
                          f"where an approach silently vanished would look healthy")
        for entry in body.get("approaches") or []:
            if not entry.get("available"):
                self.assertTrue(
                    str(entry.get("reason") or "").strip(),
                    f"{entry.get('approach_id')} is unavailable with no reason given")

    def test_it_says_run_history_does_not_survive_a_restart(self):
        """Runs are in memory. A researcher must not learn that from losing results."""
        body = self.client.get("/api/health").json()
        storage = str(body.get("run_storage") or "").lower()
        self.assertIn("memory", storage)
        self.assertIn("restart", storage)

    def test_it_never_returns_a_credential(self):
        """It may say a credential is PRESENT; it must never say what it is."""
        text = self.client.get("/api/health").text
        self.assertIsInstance(main.llm_provider.bedrock_env_ready(), bool)
        for secret_env in ("AWS_BEARER_TOKEN_BEDROCK", "AWS_SECRET_ACCESS_KEY",
                           "AWS_SESSION_TOKEN", "AWS_ACCESS_KEY_ID"):
            value = os.environ.get(secret_env)
            if value and len(value) > 8:
                self.assertNotIn(value, text,
                                 f"the health response leaks {secret_env}")
        # The KEY NAMES are fine to mention; the values are not. Guard the shape too: no
        # long opaque token-looking strings.
        self.assertNotIn("ABIA", text)
        self.assertNotIn("ASIA", text)

    def test_a_broken_llm_configuration_does_not_make_the_service_look_down(self):
        """A health check that fails for the wrong reason causes a redeploy loop."""
        original = main.llm_provider.bedrock_env_ready

        def explode():
            raise RuntimeError("credential store unreachable")

        main.llm_provider.bedrock_env_ready = explode
        try:
            response = self.client.get("/api/health")
            self.assertEqual(response.status_code, 200,
                             "a broken LLM config took the health check down with it")
            self.assertFalse(response.json().get("bedrock_configured"))
        finally:
            main.llm_provider.bedrock_env_ready = original


class PortBindingTests(unittest.TestCase):
    """Render assigns the port at runtime; a hardcoded one receives no traffic."""

    def test_the_entry_point_reads_PORT_and_defaults_host_to_loopback(self):
        with open("main.py", encoding="utf-8") as handle:
            text = handle.read()
        tail = text.split('if __name__ == "__main__":', 1)[-1]
        self.assertIn('os.environ.get("PORT")', tail,
                      "the entry point ignores $PORT, so a PaaS deploy gets no traffic")
        self.assertIn('os.environ.get("HOST", "127.0.0.1")', tail,
                      "HOST must default to loopback so running locally does not expose "
                      "the service on the network")
        self.assertNotIn('host="0.0.0.0"', tail,
                         "0.0.0.0 must not be the built-in default; it belongs in the "
                         "deployment's start command, as a recorded decision")


class RenderConfigTests(unittest.TestCase):
    """render.yaml must describe a service that would actually work, and hide no secrets."""

    PATH = "render.yaml"

    def setUp(self):
        try:
            import yaml                                   # noqa: F401
        except ImportError:                                # pragma: no cover
            self.skipTest("pyyaml is not installed")
        if not os.path.exists(self.PATH):
            self.skipTest("render.yaml is not present")

    def _service(self):
        import yaml
        with open(self.PATH, encoding="utf-8") as handle:
            document = yaml.safe_load(handle)
        services = document.get("services") or []
        self.assertTrue(services, "render.yaml declares no services")
        return services[0]

    def test_the_start_command_binds_the_assigned_port_on_all_interfaces(self):
        start = str(self._service().get("startCommand") or "")
        self.assertIn("0.0.0.0", start,
                      "the start command does not bind 0.0.0.0, so Render's router cannot "
                      "reach the process")
        self.assertIn("$PORT", start,
                      "the start command hardcodes a port instead of using $PORT")

    def test_it_runs_a_single_worker_because_runs_live_in_memory(self):
        start = str(self._service().get("startCommand") or "")
        self.assertIn("--workers 1", start,
                      "with more than one worker, run history would differ per process and "
                      "runs would appear to vanish at random")

    def test_the_health_check_path_is_a_route_that_exists(self):
        path = str(self._service().get("healthCheckPath") or "")
        self.assertTrue(path, "no healthCheckPath is configured")
        routes = {getattr(r, "path", None) for r in main.app.routes}
        self.assertIn(path, routes,
                      f"healthCheckPath {path!r} is not a route this app serves, so every "
                      f"deploy would be marked unhealthy")

    def test_no_secret_value_is_committed(self):
        """Every credential must be `sync: false`, never a literal."""
        import yaml
        with open(self.PATH, encoding="utf-8") as handle:
            document = yaml.safe_load(handle)
        secret_keys = {
            "BIOSIM_PAID_ACCESS_TOKEN",
            "AWS_BEARER_TOKEN_BEDROCK",
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_SESSION_TOKEN",
        }
        for env in (document["services"][0].get("envVars") or []):
            key = str(env.get("key") or "")
            if key in secret_keys:
                self.assertNotIn(
                    "value", env,
                    f"{key} has a literal value in render.yaml; credentials must use "
                    f"sync: false so Render prompts instead")

    def test_it_requires_a_token_for_money_spending_operations(self):
        """The public site may stay open, but Batch and Bedrock must fail closed."""
        with open(self.PATH, encoding="utf-8") as handle:
            text = handle.read().upper()
        self.assertIn("TOKEN-GATED", text)
        self.assertIn("BIOSIM_REQUIRE_PAID_ACCESS_TOKEN", text)
        self.assertIn("BATCH", text, "the AWS cost exposure is not spelled out")

    def test_it_warns_that_a_service_already_exists(self):
        """Connecting this as a new Blueprint would create a duplicate service."""
        with open(self.PATH, encoding="utf-8") as handle:
            text = handle.read().upper()
        self.assertIn("ALREADY EXISTS", text)


class VerificationScriptPortabilityTests(unittest.TestCase):
    def test_language_to_model_probe_has_no_machine_specific_path(self):
        root = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(root, "verify_language_to_model.py")
        with open(path, encoding="utf-8") as handle:
            source = handle.read()
        self.assertNotIn("D:\\SURF", source,
                         "the verification script only runs from one developer's machine")
        self.assertNotIn("sys.path.insert", source,
                         "a script in the repository root does not need to insert its own "
                         "absolute directory into sys.path")


if __name__ == "__main__":
    unittest.main()
