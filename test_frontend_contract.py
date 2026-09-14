"""The frontend/backend CONTRACT: the payload shapes the JS actually sends must work.

Why this file exists. In one session two different agents changed two sides of the same
contract: one added input validation to every route in main.py, another owned the JS
that calls those routes. Nothing in the suite would have caught a break, because the
route tests post payloads written by the person hardening the route, and the browser
can only exercise whichever main.py the running server happened to load -- which was
the PRE-hardening one, so a browser walk would have passed while the next restart
broke the product.

Every payload below is the shape read out of static/workflow.js and static/app.js, not
a shape invented here. When a route's validation tightens, the matching test fails and
names the stage a researcher would have lost.

The shapes, and where they come from:
  applyDomain          -> POST /api/project/new        {domain}
  validateDomain       -> POST /api/geometry/validate   {domain}
  validateTopology     -> POST /api/topology/validate   {domain, topology}
  generateMesh         -> POST /api/mesh/generate       {domain, settings}
  validateConditions   -> POST /api/conditions/validate {conditions, domain, fields}
  solve1DNow           -> POST /api/pde/solve1d         {domain, model, conditions,
                                                         mesh_settings}
  app.js runAbm        -> POST /api/abm/simulate        {blueprint}   <- blueprint ONLY
"""
import unittest

from fastapi.testclient import TestClient

import main

RECTANGLE = {
    "kind": "rectangle",
    "rectangle": {"x_min": 0.0, "y_min": 0.0, "width": 40.0, "height": 25.0},
    "units": "um", "time_units": "s", "regions": [],
}
INTERVAL = {
    "kind": "interval",
    "interval": {"x_min": 0.0, "length": 100.0},
    "units": "um", "time_units": "s", "regions": [],
}


def _field_names(model):
    """Field names from a model block, whose `fields` is a list of objects."""
    fields = (model or {}).get("fields") or []
    if isinstance(fields, dict):
        return [name for name in fields if name]
    names = []
    for field in fields:
        name = field.get("name") or field.get("id") if isinstance(field, dict) else field
        if name:
            names.append(str(name))
    return names


class FrontendContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(main.app)

    def _post(self, path, payload, where):
        response = self.client.post(path, json=payload)
        self.assertEqual(
            response.status_code, 200,
            f"{where}: POST {path} with the shape the frontend sends was rejected "
            f"({response.status_code}). The researcher loses this stage. "
            f"Detail: {response.text[:400]}")
        return response.json()

    def _project(self, domain):
        payload = self._post("/api/project/new", {"domain": domain}, "stage 1")
        return payload.get("project") or payload

    # ---- the workflow stages, in the order a researcher walks them ----

    def test_stage1_domain(self):
        project = self._project(RECTANGLE)
        self.assertTrue(project.get("domain"), "the new project carries no domain")
        self._post("/api/geometry/validate", {"domain": project["domain"]}, "stage 1")

    def test_stage2_topology(self):
        project = self._project(RECTANGLE)
        self._post("/api/topology/validate",
                   {"domain": project["domain"],
                    "topology": project.get("topology")
                    or {"nodes": [], "edges": [], "cells": []}}, "stage 2")

    def test_stage3_mesh(self):
        project = self._project(RECTANGLE)
        payload = self._post("/api/mesh/generate",
                             {"domain": project["domain"],
                              "settings": {"target_spacing": 2.5}}, "stage 3")
        self.assertTrue(payload.get("mesh"), "mesh generation returned no mesh")

    def test_stage4_model_presets_load(self):
        listing = self.client.get("/api/model/presets")
        self.assertEqual(listing.status_code, 200)
        body = listing.json()
        names = body.get("presets") or body.get("names") or []
        self.assertTrue(names, "the PDE model stage has no presets to offer")
        first = names[0] if isinstance(names[0], str) else names[0].get("name")
        detail = self.client.get(f"/api/model/preset/{first}")
        self.assertEqual(detail.status_code, 200,
                         f"stage 4: preset {first!r} could not be loaded")

    def test_stage4_solve_1d(self):
        """The 1D solver needs an INTERVAL domain; a rectangle is correctly refused."""
        project = self._project(INTERVAL)
        payload = self._post("/api/pde/solve1d",
                             {"domain": project["domain"],
                              "model": project.get("model"),
                              "conditions": project.get("conditions") or [],
                              "mesh_settings": project.get("mesh_settings")
                              or {"target_spacing": 2.5}}, "stage 4")
        self.assertTrue(payload.get("u"), "the 1D solve returned no field data")

    def test_stage5_conditions(self):
        project = self._project(RECTANGLE)
        self._post("/api/conditions/validate",
                   {"conditions": project.get("conditions") or [],
                    "domain": project["domain"],
                    "fields": _field_names(project.get("model"))}, "stage 5")

    def test_stage6_approaches_are_offered(self):
        """All three approaches must be independently selectable, with no silent loss.

        `load_errors` is the important half: an approach that fails to import would
        otherwise just be absent from the stage-6 cards, and the researcher would pick
        one of the remaining two without ever being told the one they wanted broke.
        """
        response = self.client.get("/api/approaches")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        approaches = body.get("approaches") or []
        found = {str(a.get("approach_id")).lower(): a for a in approaches}
        for expected in ("abm", "cc3d", "mpc"):
            self.assertIn(
                expected, found,
                f"the {expected.upper()} approach is not offered; the three approaches "
                f"are meant to be independently selectable. Got: {sorted(found)}")
            # NOT `assertTrue(available)`. This used to assert every approach is available,
            # which is a claim about the MACHINE, not about the product -- it passed here
            # only because this developer's .env configures AWS Batch, and CI failed it on
            # a bare runner where CompuCell3D is legitimately absent. The product was
            # behaving correctly and saying so precisely; the test was wrong, and it was
            # also hiding the requirement that genuinely matters. What must hold everywhere
            # is that an unavailable approach is still OFFERED and EXPLAINS ITSELF, because
            # silently dropping it is how a researcher ends up picking a different engine
            # without noticing.
            entry = found[expected]
            if not entry.get("available"):
                reason = str(entry.get("unavailable_reason") or "").strip()
                self.assertTrue(
                    reason,
                    f"{expected.upper()} is unavailable and gives NO reason; the "
                    f"researcher is told nothing about why they cannot run it")
                self.assertGreater(
                    len(reason), 40,
                    f"{expected.upper()}'s unavailability reason is too terse to act on: "
                    f"{reason!r}")
                # A refusal must never point at a different engine as a substitute.
                others = {"abm": ("compucell", "cc3d", "mpc"),
                          "cc3d": ("in-tree abm", "falls back to abm", "use mpc"),
                          "mpc": ("compucell", "cc3d", "in-tree abm")}[expected]
                for forbidden in others:
                    self.assertNotIn(
                        forbidden, reason.lower(),
                        f"{expected.upper()}'s refusal offers {forbidden!r} as a "
                        f"substitute; approaches must never silently swap")
        self.assertFalse(
            body.get("load_errors"),
            f"an approach failed to load and would be silently missing from the "
            f"stage-6 cards: {body.get('load_errors')}")

    def test_stage7_project_validates(self):
        project = self._project(RECTANGLE)
        self._post("/api/project/validate", {"project": project}, "stage 7")

    # ---- the main page's ABM tab ----

    def test_main_page_abm_preset_then_simulate(self):
        """app.js sends {blueprint} ONLY -- no num_mcs, no save_every.

        /api/abm/simulate must keep defaulting the run length internally. The default
        was deliberately REMOVED from /api/multiscale/simulate (where num_mcs is a
        top-level request field) to close an amplification vector, and this asserts
        that removal did not also land on the route the main page actually calls.
        """
        preset = self.client.get("/api/abm/preset/cell_sorting")
        self.assertEqual(preset.status_code, 200, "the ABM preset could not be loaded")
        blueprint = preset.json()
        self.assertTrue(blueprint.get("cell_types"),
                        "the ABM preset carries no cell types")
        self._post("/api/abm/simulate", {"blueprint": blueprint}, "main page ABM tab")


if __name__ == "__main__":
    unittest.main()
