"""MAPLE's demonstration target must never read as an extraction from the literature.

With no working language model MAPLE returns a demo target. The server marks it (target_id
"demo_<name>", a "Returning demo target" log line) and the page must say that no extraction
happened instead of showing the demo values like results.
"""
import os
import unittest

from fastapi.testclient import TestClient

import main


class MapleDemoTargetTests(unittest.TestCase):
    def test_the_server_marks_a_demo_target(self):
        response = TestClient(main.app).post("/api/maple/extract", json={
            "param_name": "k_prolif_cancer", "param_units": "1/day",
            "param_description": "Cancer cell proliferation rate", "mechanistic_context": "",
            "llm": {"engine": "off"}})
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertTrue(str(body["target"]["target_id"]).startswith("demo_"))
        self.assertTrue(any("Returning demo" in line for line in body["logs"]))

    def test_the_page_labels_a_demo_target(self):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "app.js")
        with open(path, encoding="utf-8") as handle:
            source = handle.read()
        start = source.index("async function extractMapleParameters()")
        body = source[start:source.index("\nasync function ", start + 10)]
        self.assertIn('startsWith("demo_")', body)
        self.assertIn("not a value from the literature", body)


if __name__ == "__main__":
    unittest.main()
