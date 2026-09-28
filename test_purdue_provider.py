"""Offline tests for the Purdue GenAI Studio provider (all HTTP is mocked)."""

import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import patch

import requests

import llm_provider


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        return self._payload


def chat_response(content='{"ok":true}', finish_reason="stop"):
    return FakeResponse(payload={
        "choices": [{
            "message": {"content": content},
            "finish_reason": finish_reason,
        }]
    })


SCHEMA = {
    "type": "object",
    "properties": {"ok": {"type": "boolean"}},
    "required": ["ok"],
    "additionalProperties": False,
}


class PurdueProviderTests(unittest.TestCase):
    def setUp(self):
        self.secret = "unit-test-secret-never-emit"
        self.env = patch.dict("os.environ", {
            "PURDUE_GENAI_API_KEY": self.secret,
            "PURDUE_GENAI_BASE_URL": "https://unit.invalid/api",
            "PURDUE_GENAI_MODEL": "gpt-oss:120b",
        }, clear=False)
        self.env.start()
        self.addCleanup(self.env.stop)
        with llm_provider.PurdueGenAILLM._rate_lock:
            llm_provider.PurdueGenAILLM._rate_tokens = 50.0
            llm_provider.PurdueGenAILLM._rate_updated = time.monotonic()
        with llm_provider.PurdueGenAILLM._structured_modes_lock:
            llm_provider.PurdueGenAILLM._structured_modes.clear()

    def client(self, model="gpt-oss:120b"):
        return llm_provider.PurdueGenAILLM(model)

    def test_headers_body_and_both_schema_fields(self):
        with patch("requests.post", return_value=chat_response()) as post:
            result = self.client().chat(
                [{"role": "user", "content": "hello"}],
                temperature=0.0,
                max_tokens=123,
                json_schema=SCHEMA,
                schema_name="answer",
            )
        self.assertEqual(result, '{"ok":true}')
        kwargs = post.call_args.kwargs
        self.assertEqual(kwargs["timeout"], (15, 180))
        self.assertEqual(kwargs["headers"]["Authorization"], f"Bearer {self.secret}")
        body = kwargs["json"]
        self.assertEqual(body["model"], "gpt-oss:120b")
        self.assertFalse(body["stream"])
        self.assertEqual(body["max_tokens"], 123)
        self.assertEqual(body["format"], SCHEMA)
        self.assertEqual(body["response_format"]["type"], "json_schema")
        self.assertEqual(body["response_format"]["json_schema"], {
            "name": "answer", "schema": SCHEMA,
        })

    def test_structured_400_retries_with_ollama_format_only(self):
        responses = [FakeResponse(400, {"error": "sensitive"}), chat_response()]
        with patch("requests.post", side_effect=responses) as post:
            self.client("qwen3:32b").chat(
                [{"role": "user", "content": "hello"}], json_schema=SCHEMA)
        self.assertEqual(post.call_count, 2)
        first = post.call_args_list[0].kwargs["json"]
        second = post.call_args_list[1].kwargs["json"]
        self.assertIn("response_format", first)
        self.assertIn("format", first)
        self.assertNotIn("response_format", second)
        self.assertEqual(second["format"], SCHEMA)

    def test_json_null_retries_then_succeeds(self):
        with patch("requests.post", side_effect=[FakeResponse(200, None), chat_response()]) as post, \
             patch("llm_provider.time.sleep"):
            self.assertEqual(self.client().chat([{"role": "user", "content": "x"}]),
                             '{"ok":true}')
        self.assertEqual(post.call_count, 2)

    def test_http_429_retries_then_succeeds(self):
        with patch("requests.post", side_effect=[FakeResponse(429), chat_response()]) as post, \
             patch("llm_provider.time.sleep"):
            self.client().chat([{"role": "user", "content": "x"}])
        self.assertEqual(post.call_count, 2)

    def test_http_5xx_exhaustion_is_safe(self):
        secret_body = "provider-body-must-not-appear"
        with patch("requests.post", return_value=FakeResponse(503, text=secret_body)) as post, \
             patch("llm_provider.time.sleep"), \
             self.assertRaises(llm_provider.LLMError) as raised:
            self.client().chat([{"role": "user", "content": "x"}])
        self.assertEqual(post.call_count, 4)
        self.assertIn("HTTP 503", str(raised.exception))
        self.assertNotIn(secret_body, str(raised.exception))
        self.assertNotIn(self.secret, str(raised.exception))

    def test_truncated_and_empty_content_raise(self):
        with patch("requests.post", return_value=chat_response("partial", "length")), \
             self.assertRaisesRegex(llm_provider.LLMError, "truncated"):
            self.client().chat([{"role": "user", "content": "x"}])
        with patch("requests.post", return_value=chat_response("   ")), \
             self.assertRaisesRegex(llm_provider.LLMError, "empty"):
            self.client().chat([{"role": "user", "content": "x"}])

    def test_key_never_appears_in_transport_exception(self):
        with patch("requests.post", side_effect=requests.ConnectionError(self.secret)), \
             patch("llm_provider.time.sleep"), \
             self.assertRaises(llm_provider.LLMError) as raised:
            self.client().chat([{"role": "user", "content": "x"}])
        self.assertIn("ConnectionError", str(raised.exception))
        self.assertNotIn(self.secret, str(raised.exception))

    def test_build_client_wants_llm_and_browser_key_is_ignored(self):
        client = llm_provider.build_client({
            "engine": "purdue",
            "model": "qwen3:32b",
            "api_key": "browser-supplied-must-be-ignored",
            "base_url": "https://browser.invalid",
        })
        self.assertTrue(llm_provider.wants_llm({"engine": "purdue"}))
        self.assertIsInstance(client, llm_provider.PurdueGenAILLM)
        self.assertEqual(client.model, "qwen3:32b")
        self.assertEqual(client.base_url, "https://unit.invalid/api")
        self.assertEqual(client._api_key, self.secret)
        self.assertNotEqual(client._api_key, "browser-supplied-must-be-ignored")

    def test_missing_or_placeholder_key_is_actionable(self):
        with patch.dict("os.environ", {
            "PURDUE_GENAI_API_KEY": "PASTE_YOUR_KEY",
            "GENAI_API_KEY": "",
        }, clear=False), self.assertRaises(llm_provider.LLMError) as raised:
            llm_provider.build_client({"engine": "purdue"})
        self.assertIn("PURDUE_GENAI_API_KEY", str(raised.exception))
        self.assertNotIn("PASTE_YOUR_KEY", str(raised.exception))

    def test_generate_structured_uses_native_schema_and_parses_wrappers(self):
        class Client:
            def __init__(self):
                self.kwargs = None

            def chat(self, messages, temperature=0.2, max_tokens=1,
                     json_mode=False, json_schema=None, schema_name="output"):
                self.kwargs = {
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "json_schema": json_schema,
                    "schema_name": schema_name,
                }
                return '<think>private reasoning</think>```json\n{"ok": true}\n```'

        client = Client()
        result = llm_provider.generate_structured(
            client, "prompt", SCHEMA, system="system", name="answer")
        self.assertEqual(result, {"ok": True})
        self.assertEqual(client.kwargs["json_schema"], SCHEMA)
        self.assertEqual(client.kwargs["schema_name"], "answer")
        self.assertEqual(client.kwargs["messages"][0]["role"], "system")

    def test_generate_structured_falls_back_for_legacy_client(self):
        class LegacyClient:
            def __init__(self):
                self.messages = None
                self.json_mode = None

            def chat(self, messages, temperature=0.2, max_tokens=1, json_mode=False):
                self.messages = messages
                self.json_mode = json_mode
                return '```json\n{"ok": true}\n```'

        client = LegacyClient()
        self.assertEqual(
            llm_provider.generate_structured(client, "prompt", SCHEMA),
            {"ok": True},
        )
        self.assertTrue(client.json_mode)
        self.assertIn('"required":["ok"]', client.messages[-1]["content"])

    def test_list_models_is_cached_for_ten_minutes(self):
        response = FakeResponse(payload={"data": [
            {"id": "gpt-oss:120b"}, {"id": "qwen3:32b"},
        ]})
        with patch("requests.get", return_value=response) as get:
            client = self.client()
            self.assertEqual(client.list_models(), ["gpt-oss:120b", "qwen3:32b"])
            self.assertEqual(client.list_models(), ["gpt-oss:120b", "qwen3:32b"])
            self.assertEqual(get.call_count, 1)
            client.list_models(force=True)
            self.assertEqual(get.call_count, 2)

    def test_transcribe_image_uses_data_url_and_available_vision_model(self):
        models = FakeResponse(payload={"data": [
            {"id": "gpt-oss:120b"}, {"id": "qwen3-vl:32b"},
        ]})
        with patch("requests.get", return_value=models), \
             patch("requests.post", return_value=chat_response("dX/dt = k1 - k2*X")) as post:
            client = self.client()
            text = client.transcribe_image(b"PNGDATA", "png", "Read equation")
        self.assertEqual(text, "dX/dt = k1 - k2*X")
        body = post.call_args.kwargs["json"]
        self.assertEqual(body["model"], "qwen3-vl:32b")
        blocks = body["messages"][0]["content"]
        self.assertEqual(blocks[0], {"type": "text", "text": "Read equation"})
        self.assertTrue(blocks[1]["image_url"]["url"].startswith(
            "data:image/png;base64,"))

    def test_engine_status_contains_no_key(self):
        status = llm_provider.engine_status()
        self.assertTrue(status["purdue"]["ready"])
        self.assertEqual(status["purdue"]["model"], "gpt-oss:120b")
        self.assertNotIn(self.secret, repr(status))

    def test_rate_limiter_and_semaphore_do_not_deadlock_twenty_threads(self):
        active = 0
        peak = 0
        lock = threading.Lock()

        def respond(*_args, **_kwargs):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            time.sleep(0.005)
            with lock:
                active -= 1
            return chat_response()

        client = self.client()
        with patch("requests.post", side_effect=respond):
            with ThreadPoolExecutor(max_workers=20) as pool:
                futures = [pool.submit(
                    client.chat, [{"role": "user", "content": str(i)}]
                ) for i in range(20)]
                results = [f.result(timeout=5) for f in futures]
        self.assertEqual(results, ['{"ok":true}'] * 20)
        self.assertLessEqual(peak, 6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
