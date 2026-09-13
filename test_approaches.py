"""Tests for the ABM and CompuCell3D adapters, and for approach independence."""

import os
import unittest
import xml.etree.ElementTree as ET
from unittest.mock import patch

import approach_abm as abm_mod
import approach_base as base
import approach_cc3d as cc3d_mod
import approach_mpc as mpc_mod
import geometry as geo


def _codes(issues):
    return {i.code for i in issues}


def _digest(value):
    """Stable hash of a nested structure, so a failure prints a hash not a lattice."""
    import hashlib
    import json
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _abm_project(**overrides):
    config = {
        "grid": {"width": 30, "height": 30},
        "temperature": 10.0,
        "num_mcs": 6,
        "save_every": 3,
        "seed": 12345,
        "cell_types": [{
            "type_id": 1, "name": "Tumour", "target_volume": 25, "lambda_volume": 2.0,
            "target_surface": 20, "lambda_surface": 0.5,
            "max_volume_before_division": 400, "growth_rate": 0.0,
            "death_probability": 0.0, "color": [255, 80, 80], "initial_count": 3,
        }],
    }
    config.update(overrides)
    return {"domain": geo.make_rectangle(30.0, 30.0), "approaches": {"abm": config}}


def _cc3d_project(**overrides):
    config = {
        "lattice": {"x": 40, "y": 40, "z": 1},
        "steps": 100,
        "temperature": 10.0,
        "cell_types": [{"type_id": 1, "name": "Tumour"}, {"type_id": 2, "name": "Stroma"}],
        "contact_energies": [
            {"type1": "Tumour", "type2": "Tumour", "energy": 8.0},
            {"type1": "Tumour", "type2": "Medium", "energy": 12.0},
        ],
    }
    config.update(overrides)
    return {"domain": geo.make_rectangle(40.0, 40.0), "approaches": {"cc3d": config}}


class RegistryTests(unittest.TestCase):
    def test_all_three_approaches_are_registered(self):
        self.assertEqual(set(base.registered_ids()), {"abm", "cc3d", "mpc"})

    def test_capabilities_are_listed_for_unavailable_approaches_too(self):
        listed = {c["approach_id"]: c for c in base.list_approaches()}
        self.assertEqual(set(listed), {"abm", "cc3d", "mpc"})
        for entry in listed.values():
            self.assertIn("available", entry)
            if not entry["available"]:
                self.assertTrue(entry["unavailable_reason"],
                                "an unavailable approach must say why")

    def test_third_approach_is_spelled_mpc(self):
        for entry in base.list_approaches():
            self.assertNotIn("MCP", entry["label"])
        self.assertIn("mpc", base.registered_ids())
        self.assertNotIn("mcp", [i for i in base.registered_ids() if i != "mpc"])

    def test_unknown_approach_raises_with_the_known_list(self):
        with self.assertRaises(KeyError) as raised:
            base.get_approach("nope")
        self.assertIn("abm", str(raised.exception))


class ApproachIndependenceTests(unittest.TestCase):
    """Selecting one approach must not require or involve the other two."""

    def test_abm_validates_without_any_cc3d_or_mpc_configuration(self):
        project = _abm_project()
        self.assertNotIn("cc3d", project["approaches"])
        self.assertNotIn("mpc", project["approaches"])
        self.assertEqual([i for i in abm_mod.ABM_ADAPTER.validate(project)
                          if i.severity == "error"], [])

    def test_mpc_validates_without_abm_configuration(self):
        project = {"approaches": {"mpc": {"target": 1.0}}}
        self.assertEqual([i for i in mpc_mod.MPC_ADAPTER.validate(project)
                          if i.severity == "error"], [])

    def test_abm_availability_does_not_depend_on_cc3d(self):
        with patch.object(cc3d_mod, "detect_cc3d",
                          return_value={"available": False, "method": "", "version": "",
                                        "path": "", "reason": "not installed"}):
            self.assertFalse(cc3d_mod.CC3D_ADAPTER.get_capabilities().available)
            self.assertTrue(abm_mod.ABM_ADAPTER.get_capabilities().available)

    def test_switching_approach_preserves_each_configuration(self):
        project = {
            "domain": geo.make_rectangle(30.0, 30.0),
            "approaches": {
                "abm": _abm_project()["approaches"]["abm"],
                "mpc": {"target": 2.5, "prediction_horizon": 8, "control_horizon": 2},
            },
            "selected_approach": "abm",
        }
        project["selected_approach"] = "mpc"
        self.assertEqual(project["approaches"]["abm"]["num_mcs"], 6)
        self.assertEqual(project["approaches"]["mpc"]["target"], 2.5)
        self.assertEqual(abm_mod.ABM_ADAPTER.approach_config(project)["num_mcs"], 6)
        self.assertEqual(mpc_mod.MPC_ADAPTER.approach_config(project)["target"], 2.5)


class ABMValidationTests(unittest.TestCase):
    def test_valid_configuration_passes(self):
        self.assertEqual([i for i in abm_mod.ABM_ADAPTER.validate(_abm_project())
                          if i.severity == "error"], [])

    def test_missing_cell_types_is_an_error(self):
        self.assertIn("abm_no_cell_types",
                      _codes(abm_mod.ABM_ADAPTER.validate(_abm_project(cell_types=[]))))

    def test_reserved_type_id_zero_rejected(self):
        project = _abm_project()
        project["approaches"]["abm"]["cell_types"][0]["type_id"] = 0
        self.assertIn("abm_cell_type_id_reserved",
                      _codes(abm_mod.ABM_ADAPTER.validate(project)))

    def test_zero_temperature_rejected(self):
        self.assertIn("abm_temperature_not_positive",
                      _codes(abm_mod.ABM_ADAPTER.validate(_abm_project(temperature=0.0))))

    def test_tiny_lattice_rejected(self):
        self.assertIn("abm_grid_width_too_small",
                      _codes(abm_mod.ABM_ADAPTER.validate(
                          _abm_project(grid={"width": 2, "height": 30}))))

    def test_1d_domain_rejected_for_abm(self):
        project = _abm_project()
        project["domain"] = geo.make_interval(30.0)
        self.assertIn("abm_requires_2d", _codes(abm_mod.ABM_ADAPTER.validate(project)))

    def test_compile_refuses_invalid_configuration(self):
        with self.assertRaises(ValueError):
            abm_mod.ABM_ADAPTER.compile(_abm_project(cell_types=[]))

    def test_configuration_that_seeds_no_cells_is_refused(self):
        """An empty lattice would otherwise report a successful run with no cells."""
        project = _abm_project()
        project["approaches"]["abm"]["cell_types"][0].pop("initial_count", None)
        issues = abm_mod.ABM_ADAPTER.validate(project)
        self.assertIn("abm_no_initial_cells", _codes(issues))
        with self.assertRaises(ValueError):
            abm_mod.ABM_ADAPTER.compile(project)

    def test_explicit_initial_config_is_accepted(self):
        project = _abm_project(initial_config=[{"type_id": 1, "count": 4, "radius": 3,
                                                "region": "center"}])
        project["approaches"]["abm"]["cell_types"][0].pop("initial_count", None)
        self.assertEqual([i for i in abm_mod.ABM_ADAPTER.validate(project)
                          if i.severity == "error"], [])


class ABMExecutionTests(unittest.TestCase):
    def test_run_produces_real_agent_results(self):
        adapter = abm_mod.ABM_ADAPTER
        compiled = adapter.compile(_abm_project())
        result = adapter.run(compiled, base.RunContext(run_id="abm-1", seed=12345))

        self.assertEqual(result["approach"], "abm")
        self.assertEqual(result["kind"], "agents")
        self.assertGreaterEqual(len(result["t"]), 2)
        self.assertEqual(len(result["lattice_frames"]), len(result["t"]))
        self.assertEqual(len(result["cells"]), len(result["t"]))
        self.assertGreater(result["summary"]["live_cells"], 0)

    def test_cell_table_has_the_required_columns(self):
        adapter = abm_mod.ABM_ADAPTER
        result = adapter.run(adapter.compile(_abm_project()),
                             base.RunContext(run_id="abm-2", seed=1))
        rows = result["cells"][-1]
        self.assertTrue(rows, "the final frame should contain cells")
        for key in ("id", "type", "state", "alive", "x", "y", "volume", "age",
                    "parent_id", "generation"):
            self.assertIn(key, rows[0], key)

    def test_seeded_run_is_repeatable(self):
        adapter = abm_mod.ABM_ADAPTER
        first = adapter.run(adapter.compile(_abm_project(seed=99)),
                            base.RunContext(run_id="a", seed=99))
        second = adapter.run(adapter.compile(_abm_project(seed=99)),
                             base.RunContext(run_id="b", seed=99))
        # Hashed: a mismatch on a 30x30x3 lattice across several frames would
        # otherwise print megabytes of diff.
        self.assertEqual(_digest(first["lattice_frames"]), _digest(second["lattice_frames"]))
        self.assertEqual(first["cell_counts"], second["cell_counts"])

    def test_different_seeds_diverge(self):
        adapter = abm_mod.ABM_ADAPTER
        first = adapter.run(adapter.compile(_abm_project(seed=1, num_mcs=8)),
                            base.RunContext(run_id="a", seed=1))
        second = adapter.run(adapter.compile(_abm_project(seed=2, num_mcs=8)),
                             base.RunContext(run_id="b", seed=2))
        self.assertNotEqual(_digest(first["lattice_frames"]),
                            _digest(second["lattice_frames"]))

    def test_cancellation_stops_the_run(self):
        adapter = abm_mod.ABM_ADAPTER
        compiled = adapter.compile(_abm_project(num_mcs=1000))
        with self.assertRaises(base.RunCancelled):
            adapter.run(compiled, base.RunContext(run_id="cancelled",
                                                  is_cancelled=lambda: True))

    def test_progress_is_reported(self):
        adapter = abm_mod.ABM_ADAPTER
        seen = []
        adapter.run(adapter.compile(_abm_project()),
                    base.RunContext(run_id="p", seed=3,
                                    on_progress=lambda f, m: seen.append(f)))
        self.assertTrue(seen)
        self.assertLessEqual(max(seen), 1.0)


class CC3DDetectionTests(unittest.TestCase):
    def test_detection_never_raises_and_explains_absence(self):
        detected = cc3d_mod.detect_cc3d()
        self.assertIn("available", detected)
        if not detected["available"]:
            self.assertIn("CompuCell3D", detected["reason"])
            self.assertIn(cc3d_mod.RUNSCRIPT_ENV, detected["reason"])

    def test_bad_env_var_is_reported_precisely(self):
        with patch.dict(os.environ, {cc3d_mod.RUNSCRIPT_ENV: r"C:\definitely\missing\runScript.sh"}):
            detected = cc3d_mod.detect_cc3d()
        if not detected["available"]:
            self.assertIn("not an executable file", detected["reason"])

    def test_unavailable_capability_states_the_missing_dependency(self):
        with patch.object(cc3d_mod, "detect_cc3d",
                          return_value={"available": False, "method": "", "version": "",
                                        "path": "", "reason": "CompuCell3D is not installed."}):
            capabilities = cc3d_mod.CC3D_ADAPTER.get_capabilities()
        self.assertFalse(capabilities.available)
        self.assertIn("not installed", capabilities.unavailable_reason)
        self.assertTrue(capabilities.supports_export,
                        "export must remain available without the engine")

    def test_run_refuses_honestly_and_does_not_fall_back(self):
        with patch.object(cc3d_mod, "detect_cc3d",
                          return_value={"available": False, "method": "", "version": "",
                                        "path": "", "reason": "CompuCell3D is not installed."}):
            compiled = cc3d_mod.CC3D_ADAPTER.compile(_cc3d_project())
            with self.assertRaises(base.ApproachUnavailable) as raised:
                cc3d_mod.CC3D_ADAPTER.run(compiled, base.RunContext(run_id="x"))
        message = str(raised.exception)
        self.assertIn("not installed", message)
        self.assertNotIn("abm", message.lower(),
                         "refusal must not point at a different approach as a substitute")

    def test_missing_output_file_is_an_error_not_empty_results(self):
        with self.assertRaises(RuntimeError) as raised:
            cc3d_mod.CompuCell3DAdapter.parse_output(r"C:\definitely\missing\cells.csv")
        self.assertIn("no measurement file", str(raised.exception))


class CC3DConfigurationTests(unittest.TestCase):
    def test_validation_works_without_cc3d_installed(self):
        with patch.object(cc3d_mod, "detect_cc3d",
                          return_value={"available": False, "method": "", "version": "",
                                        "path": "", "reason": "absent"}):
            issues = cc3d_mod.CC3D_ADAPTER.validate(_cc3d_project())
        self.assertEqual([i for i in issues if i.severity == "error"], [])

    def test_medium_is_a_reserved_type_name(self):
        project = _cc3d_project(cell_types=[{"type_id": 1, "name": "Medium"}])
        self.assertIn("cc3d_medium_reserved", _codes(cc3d_mod.CC3D_ADAPTER.validate(project)))

    def test_duplicate_type_names_rejected(self):
        project = _cc3d_project(cell_types=[{"type_id": 1, "name": "A"},
                                            {"type_id": 2, "name": "A"}])
        self.assertIn("cc3d_cell_type_name_duplicate",
                      _codes(cc3d_mod.CC3D_ADAPTER.validate(project)))

    def test_contact_energy_referencing_unknown_type_rejected(self):
        project = _cc3d_project(contact_energies=[
            {"type1": "Tumour", "type2": "Ghost", "energy": 4.0}])
        self.assertIn("cc3d_contact_unknown_type",
                      _codes(cc3d_mod.CC3D_ADAPTER.validate(project)))

    def test_generated_cc3dml_is_valid_xml_with_the_right_content(self):
        config = cc3d_mod._merged(_cc3d_project()["approaches"]["cc3d"])
        xml_text = cc3d_mod.CompuCell3DAdapter.build_cc3dml(config)
        root = ET.fromstring(xml_text)                  # raises if malformed
        self.assertEqual(root.tag, "CompuCell3D")
        potts = root.find("Potts")
        # One <Dimensions x= y= z=/> element, per the CompuCell3D reference manual.
        # Asserting all three axes, because a lattice that silently falls back to a
        # default size is a simulation of the wrong thing rather than a crash.
        dimensions = potts.find("Dimensions")
        self.assertIsNotNone(dimensions, "Potts must carry a <Dimensions> element")
        self.assertEqual(dimensions.get("x"), "40")
        self.assertEqual(dimensions.get("y"), "40")
        self.assertEqual(dimensions.get("z"), "1")
        self.assertIsNone(potts.find("DimensionX"),
                          "the undocumented <DimensionX> form must not be emitted")
        self.assertEqual(potts.find("Steps").text, "100")
        names = [e.get("TypeName") for e in root.iter("CellType")]
        self.assertIn("Medium", names)
        self.assertIn("Tumour", names)
        self.assertIn("Stroma", names)

    def test_xml_special_characters_cannot_corrupt_the_project(self):
        project = _cc3d_project(cell_types=[{"type_id": 1, "name": 'Odd<&>"Name'}],
                                contact_energies=[])
        config = cc3d_mod._merged(project["approaches"]["cc3d"])
        root = ET.fromstring(cc3d_mod.CompuCell3DAdapter.build_cc3dml(config))
        self.assertIn('Odd<&>"Name', [e.get("TypeName") for e in root.iter("CellType")])

    def test_export_is_available_even_when_unavailable(self):
        with patch.object(cc3d_mod, "detect_cc3d",
                          return_value={"available": False, "method": "", "version": "",
                                        "path": "", "reason": "absent"}):
            exported = cc3d_mod.CC3D_ADAPTER.export_configuration(_cc3d_project())
        self.assertFalse(exported["available"])
        self.assertIn("Simulation/model.xml", exported["files"])
        self.assertIn("Simulation/steppables.py", exported["files"])
        ET.fromstring(exported["files"]["Simulation/model.xml"])
        self.assertIn("runScript", exported["files"]["README.txt"])

    def test_steppable_is_syntactically_valid_python(self):
        config = cc3d_mod._merged(_cc3d_project()["approaches"]["cc3d"])
        source = cc3d_mod.CompuCell3DAdapter.build_steppable(config)
        compile(source, "steppables.py", "exec")        # raises SyntaxError if bad
        self.assertIn("cells.csv", source)

    def test_compile_records_availability_without_pretending(self):
        with patch.object(cc3d_mod, "detect_cc3d",
                          return_value={"available": False, "method": "", "version": "",
                                        "path": "", "reason": "absent"}):
            compiled = cc3d_mod.CC3D_ADAPTER.compile(_cc3d_project())
        self.assertFalse(compiled["available"])
        self.assertIn("cc3dml", compiled)


if __name__ == "__main__":
    unittest.main()
