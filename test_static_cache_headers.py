"""Static assets must be served uncacheable, because a comment now says they are.

An earlier comment in index.html instructed maintainers to bump a ?v= token on every
app.js change and blamed hours of debugging on a stale one. That was wrong: main.py has
served no-store on every static response since before those changes, so the token could
not have decided which bundle ran. The replacement comment states the measured truth --
and a stated fact with nothing checking it is exactly how the previous claim survived
being false, so this pins it.

If someone removes NoCacheStaticFiles or adds a caching layer, these fail and the comment
gets corrected with them, instead of quietly becoming the next piece of folklore.
"""
import unittest

from fastapi.testclient import TestClient

import main


class StaticCacheHeaderTests(unittest.TestCase):
    ASSETS = ("/app.js", "/workflow.js", "/style.css", "/index.html")

    def setUp(self):
        self.client = TestClient(main.app)

    def test_every_static_asset_forbids_caching(self):
        for path in self.ASSETS:
            with self.subTest(path=path):
                response = self.client.get(path)
                self.assertEqual(response.status_code, 200, f"{path} is not served")
                cache_control = response.headers.get("cache-control", "")
                self.assertIn("no-store", cache_control,
                              f"{path} may be cached: Cache-Control={cache_control!r}")
                self.assertIn("no-cache", cache_control,
                              f"{path} lacks no-cache: Cache-Control={cache_control!r}")

    def test_the_pragma_fallback_is_present_for_old_intermediaries(self):
        response = self.client.get("/app.js")
        self.assertEqual(response.headers.get("pragma"), "no-cache")

    def test_the_comment_does_not_reinstate_the_false_instruction(self):
        """The index.html note must not tell maintainers the token decides what runs."""
        with open("static/index.html", encoding="utf-8") as handle:
            html = handle.read()
        self.assertIn("NOT LOAD-BEARING", html,
                      "index.html no longer records that the ?v= token is not decisive")
        self.assertNotIn("BUMP ?v= ON EVERY", html,
                         "the false cache instruction has been reinstated")

    def test_the_comment_records_the_real_hazard(self):
        """Truncated snapshots, not caching, caused the false defect reports."""
        with open("static/index.html", encoding="utf-8") as handle:
            html = handle.read()
        self.assertIn("TRUNCATE", html,
                      "index.html no longer warns that page snapshots truncate")


if __name__ == "__main__":
    unittest.main()
