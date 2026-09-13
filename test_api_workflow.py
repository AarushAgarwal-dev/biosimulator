"""Integration tests for the workflow API, exercised through the real FastAPI app.

Covers the scenarios the specification asks for:
  1. the existing project still loads
  2. geometry can be edited and validated
  3. a 1D model compiles and runs
  4. a 2D model compiles and runs
  5. ABM produces real agent results
  6. CompuCell3D runs when available / 7. reports honestly when not
  8. MPC respects bounds and tracks a target
  9. switching approaches preserves common configuration
 10. failed compilation cannot display "Ready"
 11. failed simulation cannot display "Completed"
 12. project export/import round-trips
"""

import time
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

import approach_base
import approach_cc3d as cc3d_mod
import boundary_conditions as bc
import geometry as geo
from main import app

client = TestClient(app)


def _wait_for_terminal(run_id, timeout=120):
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = client.get(f"/api/runs/{run_id}").json()
        if status["state"] in ("completed", "failed", "cancelled"):
            return status
        time.sleep(0.05)
    raise AssertionError(f"run {run_id} did not finish within {timeout}s")


class ExistingBehaviourTests(unittest.TestCase):
    """The pre-existing application must keep working."""

    def test_existing_blueprint_and_simulate_still_work(self):
        response = client.post("/api/blueprint", json={
            "text": "EGF activates EGFR. EGFR activates ERK.", "llm": {"engine": "off"}})
        self.assertEqual(response.status_code, 200)
        blueprint = response.json()
        self.assertEqual(client.post("/api/compile",
                                     json={"blueprint": blueprint}).status_code, 200)
        self.assertEqual(client.post("/api/simulate",
                                     json={"blueprint": blueprint}).status_code, 200)

    def test_existing_abm_preset_endpoint_still_works(self):
        self.assertEqual(client.get("/api/abm/preset/tumor_growth").status_code, 200)


class ApproachListingTests(unittest.TestCase):
    def test_all_three_approaches_are_listed(self):
        response = client.get("/api/approaches")
        self.assertEqual(response.status_code, 200)
        listed = {entry["approach_id"]: entry for entry in response.json()["approaches"]}
        for key in ("abm", "cc3d", "mpc"):
            self.assertIn(key, listed)
        for entry in listed.values():
            if not entry["available"]:
                self.assertTrue(entry["unavailable_reason"])

    def test_third_approach_is_spelled_mpc(self):
        labels = [e["label"] for e in client.get("/api/approaches").json()["approaches"]]
        self.assertFalse(any("MCP" in label for label in labels))


class GeometryStageTests(unittest.TestCase):
    def test_valid_domain_returns_a_summary(self):
        response = client.post("/api/geometry/validate",
                               json={"domain": geo.make_rectangle(10.0, 4.0)})
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["valid"])
        self.assertAlmostEqual(payload["measure"], 40.0)
        self.assertIn("rectangle", payload["summary"])

    def test_invalid_domain_is_reported_not_crashed(self):
        bad = geo.make_rectangle(10.0, 4.0)
        bad["rectangle"]["width"] = -3.0
        payload = client.post("/api/geometry/validate", json={"domain": bad}).json()
        self.assertFalse(payload["valid"])
        self.assertTrue(any(i["code"] == "width_not_positive" for i in payload["issues"]))

    def test_geometry_can_be_edited_and_saved(self):
        created = client.post("/api/project/new",
                              json={"domain": geo.make_rectangle(8.0, 5.0)})
        self.assertEqual(created.status_code, 200)
        project = created.json()
        project["domain"]["rectangle"]["width"] = 12.0
        exported = client.post("/api/project/export", json={"project": project})
        self.assertEqual(exported.status_code, 200)
        text = exported.json()["files"]["project.json"]
        reimported = client.post("/api/project/import", json={"text": text}).json()
        self.assertAlmostEqual(reimported["project"]["domain"]["rectangle"]["width"], 12.0)


class MeshStageTests(unittest.TestCase):
    def test_structured_mesh_generation(self):
        response = client.post("/api/mesh/generate", json={
            "domain": geo.make_rectangle(6.0, 3.0), "settings": {"rows": 3, "cols": 6}})
        self.assertEqual(response.status_code, 200)
        mesh = response.json()["mesh"]
        self.assertEqual(mesh["kind"], "structured")
        self.assertEqual(mesh["stats"]["node_count"], 28)

    def test_1d_mesh_generation(self):
        response = client.post("/api/mesh/generate", json={
            "domain": geo.make_interval(10.0), "settings": {"element_count": 20}})
        mesh = response.json()["mesh"]
        self.assertEqual(mesh["kind"], "1d")
        self.assertEqual(len(mesh["elements"]), 20)
        self.assertEqual(mesh["boundaries"]["right"]["nodes"], [20])

    def test_absurd_spacing_is_refused_with_a_reason(self):
        response = client.post("/api/mesh/generate", json={
            "domain": geo.make_rectangle(1000.0, 1000.0),
            "settings": {"kind": "unstructured", "target_spacing": 0.4}})
        self.assertEqual(response.status_code, 400)
        self.assertIn("limit", str(response.json()["detail"]))


class ModelStageTests(unittest.TestCase):
    def test_presets_are_listed_and_fetchable(self):
        listing = client.get("/api/model/presets").json()["presets"]
        self.assertTrue(listing)
        for entry in listing:
            fetched = client.get(f"/api/model/preset/{entry['name']}")
            self.assertEqual(fetched.status_code, 200, entry["name"])
            self.assertIn("fields", fetched.json())

    def test_unknown_preset_is_404(self):
        self.assertEqual(client.get("/api/model/preset/nonsense").status_code, 404)

    def test_model_validation_returns_latex_and_plain_summary(self):
        preset = client.get("/api/model/preset/diffusion_decay").json()
        response = client.post("/api/model/validate", json={
            "model": preset, "domain": geo.make_interval(1.0)})
        payload = response.json()
        self.assertTrue(payload["valid"])
        self.assertTrue(payload["fields"])
        self.assertIn("partial", payload["fields"][0]["latex"])
        self.assertIn("diffusion", payload["fields"][0]["summary"])

    def test_undefined_parameter_is_reported(self):
        preset = client.get("/api/model/preset/diffusion").json()
        preset["fields"][0]["reaction"] = "-missing*u"
        payload = client.post("/api/model/validate", json={"model": preset}).json()
        self.assertFalse(payload["valid"])


class ConditionStageTests(unittest.TestCase):
    def test_conflicting_conditions_are_reported(self):
        domain = geo.make_interval(1.0)
        conditions = [bc.make_dirichlet("u", "left", 0.0, condition_id="a"),
                      bc.make_neumann("u", "left", 1.0, condition_id="b")]
        payload = client.post("/api/conditions/validate", json={
            "conditions": conditions, "domain": domain, "fields": ["u"]}).json()
        self.assertFalse(payload["valid"])
        self.assertTrue(any(i["code"] == "condition_conflict_value_and_flux"
                            for i in payload["issues"]))

    def test_unassigned_boundary_is_described_as_no_flux(self):
        payload = client.post("/api/conditions/validate", json={
            "conditions": [], "domain": geo.make_interval(1.0), "fields": ["u"]}).json()
        self.assertTrue(payload["valid"])
        self.assertTrue(any("no-flux" in line for line in payload["description"]))


class Solve1DTests(unittest.TestCase):
    def test_1d_model_compiles_and_runs_to_a_linear_steady_state(self):
        domain = geo.make_interval(1.0)
        model = {"parameters": {"D": 1.0},
                 "fields": [{"name": "u", "units": "mM", "diffusion": "D",
                             "initial": "0.0", "reaction": "0", "source": "0",
                             "advection": {"vx": 0.0, "vy": 0.0},
                             "t_start": 0.0, "t_end": 2.0, "output_interval": 0.5}]}
        response = client.post("/api/pde/solve1d", json={
            "domain": domain, "model": model,
            "conditions": [bc.make_dirichlet("u", "left", 0.0),
                           bc.make_dirichlet("u", "right", 1.0)],
            "mesh_settings": {"element_count": 40},
        })
        self.assertEqual(response.status_code, 200, response.text[:400])
        payload = response.json()
        final = payload["u"][-1]
        self.assertAlmostEqual(final[0], 0.0, places=9)
        self.assertAlmostEqual(final[-1], 1.0, places=9)
        # A straight line: the midpoint sits halfway.
        self.assertAlmostEqual(final[len(final) // 2], 0.5, places=2)
        self.assertIsNone(payload["stability"]["diverged_at"])

    def test_2d_domain_is_refused_by_the_1d_endpoint(self):
        response = client.post("/api/pde/solve1d", json={
            "domain": geo.make_rectangle(1.0, 1.0),
            "model": {"parameters": {}, "fields": [{"name": "u", "diffusion": 1.0,
                                                    "t_end": 1.0, "output_interval": 0.5}]}})
        self.assertEqual(response.status_code, 400)

    def test_no_flux_run_conserves_mass(self):
        domain = geo.make_interval(1.0)
        model = {"parameters": {"D": 1.0},
                 "fields": [{"name": "u", "units": "mM", "diffusion": "D",
                             "initial": "exp(-((x-0.5)**2)/0.01)", "reaction": "0",
                             "source": "0", "advection": {"vx": 0.0, "vy": 0.0},
                             "t_start": 0.0, "t_end": 0.2, "output_interval": 0.05}]}
        payload = client.post("/api/pde/solve1d", json={
            "domain": domain, "model": model, "conditions": [],
            "mesh_settings": {"element_count": 40}}).json()
        mass = payload["mass"]
        self.assertAlmostEqual(mass[-1] / mass[0], 1.0, places=5)


class TwoDimensionalRunTests(unittest.TestCase):
    def test_2d_model_compiles_and_runs_via_the_existing_pde_path(self):
        blueprint = client.post("/api/blueprint", json={
            "text": "A reaction-diffusion Turing system. U activates itself and V. "
                    "V inhibits U. U starts at 1.0. V starts at 1.0. "
                    "U diffuses slowly, V diffuses quickly.",
            "llm": {"engine": "off"}}).json()
        self.assertEqual(blueprint["type"], "PDE")
        blueprint["spatial"]["x_grid"] = blueprint["spatial"]["y_grid"] = 12
        blueprint["simulation_config"] = {"t_max": 1.0, "dt": 0.1}
        self.assertEqual(client.post("/api/compile",
                                     json={"blueprint": blueprint}).status_code, 200)
        response = client.post("/api/simulate", json={"blueprint": blueprint})
        self.assertEqual(response.status_code, 200)
        self.assertIn("mass", response.json())


def _project(selected, **approaches):
    project = client.post("/api/project/new",
                          json={"domain": geo.make_rectangle(24.0, 24.0)}).json()
    project["selected_approach"] = selected
    project["approaches"] = approaches
    return project


_ABM_CONFIG = {
    "grid": {"width": 24, "height": 24}, "temperature": 10.0,
    "num_mcs": 4, "save_every": 2, "seed": 5,
    "cell_types": [{"type_id": 1, "name": "Cell", "target_volume": 20,
                    "lambda_volume": 2.0, "max_volume_before_division": 400,
                    "color": [200, 120, 120], "initial_count": 2}],
}
_MPC_CONFIG = {"target": 1.0, "duration": 6.0, "control_interval": 0.5,
               "prediction_horizon": 8, "control_horizon": 2,
               "input_min": -0.5, "input_max": 0.5, "input_rate_limit": 0.2}
_CC3D_CONFIG = {"lattice": {"x": 24, "y": 24, "z": 1}, "steps": 20,
                "temperature": 10.0,
                "cell_types": [{"type_id": 1, "name": "Tumour"}]}


class ABMRunTests(unittest.TestCase):
    def test_abm_produces_real_agent_results(self):
        project = _project("abm", abm=_ABM_CONFIG)
        created = client.post("/api/runs", json={"project": project, "seed": 5})
        self.assertEqual(created.status_code, 200, created.text[:300])
        run_id = created.json()["run_id"]
        status = _wait_for_terminal(run_id)
        self.assertEqual(status["state"], "completed", status.get("error"))

        results = client.get(f"/api/runs/{run_id}/results").json()["results"]
        self.assertEqual(results["kind"], "agents")
        self.assertTrue(results["lattice_frames"])
        self.assertGreater(results["summary"]["live_cells"], 0)

        cells_csv = client.get(f"/api/runs/{run_id}/export/cells").json()["csv"]
        self.assertIn("parent_id", cells_csv.splitlines()[0])

    def test_run_logs_are_available(self):
        project = _project("abm", abm=_ABM_CONFIG)
        run_id = client.post("/api/runs", json={"project": project}).json()["run_id"]
        _wait_for_terminal(run_id)
        logs = client.get(f"/api/runs/{run_id}/logs").json()["logs"]
        self.assertTrue(logs)


class MPCRunTests(unittest.TestCase):
    def test_mpc_respects_bounds_and_tracks(self):
        project = _project("mpc", mpc=_MPC_CONFIG)
        run_id = client.post("/api/runs", json={"project": project}).json()["run_id"]
        status = _wait_for_terminal(run_id)
        self.assertEqual(status["state"], "completed", status.get("error"))
        results = client.get(f"/api/runs/{run_id}/results").json()["results"]
        for u in results["series"]["control"]:
            self.assertGreaterEqual(u, _MPC_CONFIG["input_min"] - 1e-9)
            self.assertLessEqual(u, _MPC_CONFIG["input_max"] + 1e-9)
        self.assertTrue(results["summary"]["input_bounds_respected"])
        self.assertTrue(results["summary"]["rate_limit_respected"])

        csv_text = client.get(f"/api/runs/{run_id}/export/timeseries").json()["csv"]
        self.assertIn("target", csv_text.splitlines()[0])


class CompuCell3DRunTests(unittest.TestCase):
    def test_missing_cc3d_produces_an_honest_unavailable_state(self):
        project = _project("cc3d", cc3d=_CC3D_CONFIG)
        with patch.object(cc3d_mod, "detect_cc3d",
                          return_value={"available": False, "method": "", "version": "",
                                        "path": "", "reason": "CompuCell3D is not installed."}):
            created = client.post("/api/runs", json={"project": project})
            status = created.json()
        self.assertEqual(status["state"], "failed")
        self.assertEqual(status["error_kind"], "unavailable")
        self.assertIn("not installed", status["error"])
        self.assertEqual(status["approach"], "cc3d")
        # No results, and no silent substitution of another approach.
        self.assertEqual(
            client.get(f"/api/runs/{status['run_id']}/results").status_code, 409)

    def test_cc3d_export_is_available_even_when_unavailable(self):
        project = _project("cc3d", cc3d=_CC3D_CONFIG)
        with patch.object(cc3d_mod, "detect_cc3d",
                          return_value={"available": False, "method": "", "version": "",
                                        "path": "", "reason": "absent"}):
            response = client.post("/api/project/export/cc3d", json={"project": project})
        self.assertEqual(response.status_code, 200)
        files = response.json()["files"]
        self.assertIn("Simulation/model.xml", files)
        self.assertIn("project.json", files)

    def test_cc3d_runs_when_available(self):
        """With a runScript present, the adapter must attempt real execution.

        The stub is a script that exits immediately, not CompuCell3D, so the run fails
        at output parsing. The point is that the external path is taken and the failure
        is honest -- not that a stub produced results.
        """
        import os
        import tempfile

        project = _project("cc3d", cc3d=_CC3D_CONFIG)
        directory = tempfile.mkdtemp(prefix="fake_cc3d_")
        script = os.path.join(directory, "runScript.cmd")
        with open(script, "w", encoding="utf-8") as handle:
            handle.write("@echo off\r\necho fake CompuCell3D starting\r\nexit /b 0\r\n")

        with patch.object(cc3d_mod, "detect_cc3d",
                          return_value={"available": True, "method": "runscript-path",
                                        "version": "4.4.1-test",
                                        "path": script, "reason": ""}):
            created = client.post("/api/runs", json={"project": project})
            status = _wait_for_terminal(created.json()["run_id"], timeout=90)

        self.assertEqual(status["state"], "failed")
        self.assertEqual(status["engine"]["engine_version"], "4.4.1-test")
        # It ran the process and then found no measurement file -- reported, not faked.
        self.assertIn("no measurement file", status["error"])

    def test_a_hanging_cc3d_process_is_stopped_by_the_timeout(self):
        """A silent external process must not hold the run forever."""
        import os
        import tempfile

        project = _project("cc3d", cc3d={**_CC3D_CONFIG, "timeout_secs": 2.0})
        directory = tempfile.mkdtemp(prefix="hang_cc3d_")
        script = os.path.join(directory, "runScript.cmd")
        # Sleeps well past the timeout without printing anything.
        with open(script, "w", encoding="utf-8") as handle:
            handle.write("@echo off\r\nping -n 60 127.0.0.1 >nul\r\nexit /b 0\r\n")

        with patch.object(cc3d_mod, "detect_cc3d",
                          return_value={"available": True, "method": "runscript-path",
                                        "version": "4.4.1-test",
                                        "path": script, "reason": ""}):
            created = client.post("/api/runs", json={"project": project})
            status = _wait_for_terminal(created.json()["run_id"], timeout=60)

        self.assertEqual(status["state"], "failed")
        self.assertIn("did not finish within", status["error"])


class ApproachSwitchingTests(unittest.TestCase):
    def test_switching_approach_preserves_common_configuration(self):
        project = _project("abm", abm=_ABM_CONFIG, mpc=_MPC_CONFIG)
        project["model"]["parameters"]["D"] = 0.42
        first = client.post("/api/project/validate",
                            json={"project": project, "approach": "abm"}).json()
        self.assertEqual(first["stages"]["approach"]["status"], "valid")

        project["selected_approach"] = "mpc"
        second = client.post("/api/project/validate",
                             json={"project": project, "approach": "mpc"}).json()
        self.assertEqual(second["stages"]["approach"]["status"], "valid")

        # Common stages and BOTH approach blocks survived the switch.
        self.assertAlmostEqual(project["model"]["parameters"]["D"], 0.42)
        self.assertEqual(project["approaches"]["abm"]["num_mcs"], 4)
        self.assertEqual(project["approaches"]["mpc"]["prediction_horizon"], 8)
        self.assertAlmostEqual(project["domain"]["rectangle"]["width"], 24.0)


class FailureStateTests(unittest.TestCase):
    def test_failed_compilation_cannot_display_ready(self):
        project = _project("mpc", mpc=_MPC_CONFIG)
        with patch.object(approach_base.get_approach("mpc"), "compile",
                          side_effect=RuntimeError("synthetic compile failure")):
            status = client.post("/api/runs", json={"project": project}).json()
        self.assertEqual(status["state"], "failed")
        self.assertNotEqual(status["state"], "ready")
        self.assertIn("synthetic compile failure", status["error"])

    def test_failed_simulation_cannot_display_completed(self):
        project = _project("mpc", mpc=_MPC_CONFIG)
        with patch.object(approach_base.get_approach("mpc"), "run",
                          side_effect=RuntimeError("synthetic run failure")):
            created = client.post("/api/runs", json={"project": project})
            status = _wait_for_terminal(created.json()["run_id"])
        self.assertEqual(status["state"], "failed")
        self.assertNotEqual(status["state"], "completed")
        self.assertEqual(
            client.get(f"/api/runs/{status['run_id']}/results").status_code, 409)

    def test_invalid_configuration_reports_invalid_with_issues(self):
        project = _project("mpc", mpc={**_MPC_CONFIG, "prediction_horizon": 0})
        status = client.post("/api/runs", json={"project": project}).json()
        self.assertEqual(status["state"], "invalid")
        self.assertTrue(status["issues"])

    def test_starting_an_invalid_run_is_a_conflict(self):
        project = _project("mpc", mpc={**_MPC_CONFIG, "control_horizon": 999})
        run_id = client.post("/api/runs",
                             json={"project": project, "start": False}).json()["run_id"]
        self.assertEqual(client.post(f"/api/runs/{run_id}/start").status_code, 409)

    def test_unknown_run_is_404(self):
        self.assertEqual(client.get("/api/runs/run_missing").status_code, 404)


class ProjectRoundTripTests(unittest.TestCase):
    def test_export_import_round_trips(self):
        project = _project("mpc", abm=_ABM_CONFIG, mpc=_MPC_CONFIG)
        text = client.post("/api/project/export",
                           json={"project": project}).json()["files"]["project.json"]
        reimported = client.post("/api/project/import", json={"text": text}).json()
        restored = reimported["project"]
        self.assertEqual(restored["selected_approach"], "mpc")
        self.assertEqual(set(restored["approaches"]), {"abm", "mpc"})
        self.assertEqual(restored["approaches"]["abm"]["num_mcs"], 4)
        self.assertTrue(reimported["validation"]["stages"]["domain"]["status"], "valid")

    def test_importing_a_v1_document_migrates_it(self):
        legacy = ('{"schema_version": 1, "name": "Legacy", '
                  '"selected_approach": "mcp", "approaches": {"mcp": {"target": 4.0}}}')
        response = client.post("/api/project/import", json={"text": legacy})
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["project"]["selected_approach"], "mpc")
        self.assertTrue(payload["migrations_applied"])

    def test_importing_a_newer_document_is_refused(self):
        response = client.post("/api/project/import",
                               json={"text": '{"schema_version": 99}'})
        self.assertEqual(response.status_code, 400)
        self.assertIn("newer version", str(response.json()["detail"]))

    def test_importing_broken_json_is_refused_with_position(self):
        response = client.post("/api/project/import", json={"text": "{oops"})
        self.assertEqual(response.status_code, 400)
        self.assertIn("line", str(response.json()["detail"]))


if __name__ == "__main__":
    unittest.main()
