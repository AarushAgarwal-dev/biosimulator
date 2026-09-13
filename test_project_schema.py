"""Tests for the project format: round-trip, migration, validation and export."""

import json
import unittest
import xml.etree.ElementTree as ET

import geometry as geo
import meshing
import project_schema as ps
import run_manager as rm
import topology as topo


def _project_with_everything():
    project = ps.new_project("Test project", geo.make_rectangle(10.0, 6.0))
    project["mesh_settings"] = {"rows": 6, "cols": 10}
    project["mesh"] = meshing.mesh_2d_structured(project["domain"], rows=6, cols=10)

    t = project["topology"]
    a = topo.add_node(t, 1.0, 1.0)["id"]
    b = topo.add_node(t, 2.0, 1.0)["id"]
    c = topo.add_node(t, 2.0, 2.0)["id"]
    d = topo.add_node(t, 1.0, 2.0)["id"]
    for pair in ((a, b), (b, c), (c, d), (d, a)):
        topo.add_edge(t, *pair)
    topo.make_cell(t, [a, b, c, d], cell_type="epithelial")

    project["selected_approach"] = "mpc"
    project["approaches"] = {
        "mpc": {"target": 1.0, "duration": 5.0, "control_interval": 0.5,
                "prediction_horizon": 8, "control_horizon": 2},
        "abm": {"grid": {"width": 24, "height": 24}, "temperature": 10.0,
                "num_mcs": 4, "save_every": 2,
                "cell_types": [{"type_id": 1, "name": "Cell", "target_volume": 20,
                                "initial_count": 2}]},
        # A cc3d block belongs here too: this fixture is named "with everything" and
        # the export test below calls export_cc3d_package on it. Without cell types
        # the exporter used to emit a project whose only type was Medium, which loads
        # and runs but produces ZERO cells -- so the test passed on XML that described
        # an empty simulation.
        "cc3d": {"lattice": {"x": 24, "y": 24, "z": 1}, "steps": 10,
                 "temperature": 10.0, "neighbor_order": 2,
                 "cell_types": [{"type_id": 1, "name": "Cell"}],
                 "contact_energies": [{"type1": "Cell", "type2": "Medium",
                                       "energy": 12.0}],
                 "volume_constraint": {"target_volume": 20, "lambda_volume": 2.0}},
    }
    return project


class NewProjectTests(unittest.TestCase):
    def test_new_project_is_at_the_current_version(self):
        project = ps.new_project()
        self.assertEqual(project["schema_version"], ps.SCHEMA_VERSION)

    def test_new_project_domain_and_model_validate(self):
        project = ps.new_project()
        self.assertTrue(geo.is_valid(geo.validate_domain(project["domain"])))
        report = ps.validate_project(project, approach_id=None)
        # No approach is selected yet, so that stage is the only invalid one.
        self.assertEqual(report["stages"]["domain"]["status"], "valid")
        self.assertEqual(report["stages"]["model"]["status"], "valid")
        self.assertEqual(report["stages"]["approach"]["status"], "invalid")


class RoundTripTests(unittest.TestCase):
    def test_project_round_trips_through_json(self):
        project = _project_with_everything()
        restored = ps.from_json(ps.to_json(project))
        self.assertEqual(restored["name"], project["name"])
        self.assertEqual(restored["domain"]["kind"], "rectangle")
        self.assertEqual(len(restored["topology"]["nodes"]), 4)
        self.assertEqual(len(restored["topology"]["cells"]), 1)
        self.assertEqual(restored["selected_approach"], "mpc")
        self.assertEqual(set(restored["approaches"]), {"mpc", "abm", "cc3d"})
        self.assertEqual(restored["mesh"]["stats"]["node_count"],
                         project["mesh"]["stats"]["node_count"])

    def test_round_trip_preserves_both_approach_configurations(self):
        project = _project_with_everything()
        restored = ps.from_json(ps.to_json(project))
        self.assertEqual(restored["approaches"]["mpc"]["target"], 1.0)
        self.assertEqual(restored["approaches"]["abm"]["num_mcs"], 4)

    def test_switching_approach_does_not_disturb_the_other(self):
        project = _project_with_everything()
        project["selected_approach"] = "abm"
        restored = ps.from_json(ps.to_json(project))
        self.assertEqual(restored["selected_approach"], "abm")
        self.assertEqual(restored["approaches"]["mpc"]["prediction_horizon"], 8)
        self.assertEqual(restored["domain"]["rectangle"]["width"], 10.0)

    def test_malformed_json_is_reported_with_position(self):
        with self.assertRaises(ps.ProjectVersionError) as raised:
            ps.from_json("{not json")
        self.assertIn("line", str(raised.exception))


class MigrationTests(unittest.TestCase):
    def test_v1_flat_approach_config_moves_into_the_map(self):
        legacy = {
            "schema_version": 1,
            "name": "Old",
            "domain": geo.make_interval(1.0),
            "selected_approach": "mpc",
            "approach_config": {"target": 3.0, "prediction_horizon": 5},
        }
        migrated = ps.migrate(legacy)
        self.assertEqual(migrated["schema_version"], ps.SCHEMA_VERSION)
        self.assertEqual(migrated["approaches"]["mpc"]["target"], 3.0)
        self.assertNotIn("approach_config", migrated)
        self.assertTrue(any("approach_config" in note
                            for note in migrated["migrations_applied"]))

    def test_v1_bcs_key_is_renamed(self):
        legacy = {"schema_version": 1, "domain": geo.make_interval(1.0),
                  "bcs": [{"id": "x", "type": "no_flux", "field": "u", "boundary": "left"}]}
        migrated = ps.migrate(legacy)
        self.assertNotIn("bcs", migrated)
        self.assertEqual(len(migrated["conditions"]), 1)

    def test_v1_misspelled_approach_is_corrected(self):
        legacy = {"schema_version": 1, "domain": geo.make_interval(1.0),
                  "selected_approach": "mcp", "approaches": {"mcp": {"target": 2.0}}}
        migrated = ps.migrate(legacy)
        self.assertEqual(migrated["selected_approach"], "mpc")
        self.assertIn("mpc", migrated["approaches"])
        self.assertNotIn("mcp", migrated["approaches"])

    def test_newer_schema_is_refused_rather_than_partially_read(self):
        with self.assertRaises(ps.ProjectVersionError) as raised:
            ps.migrate({"schema_version": ps.SCHEMA_VERSION + 5})
        self.assertIn("newer version", str(raised.exception))

    def test_non_integer_version_is_refused(self):
        with self.assertRaises(ps.ProjectVersionError):
            ps.migrate({"schema_version": "two"})

    def test_migration_does_not_modify_the_input(self):
        legacy = {"schema_version": 1, "domain": geo.make_interval(1.0),
                  "selected_approach": "mpc", "approach_config": {"target": 1.0}}
        ps.migrate(legacy)
        self.assertIn("approach_config", legacy)

    def test_partial_document_gains_missing_keys(self):
        migrated = ps.migrate({"schema_version": 2, "name": "Sparse"})
        for key in ("domain", "topology", "model", "conditions", "approaches",
                    "visualization", "run_history"):
            self.assertIn(key, migrated)


class AggregateValidationTests(unittest.TestCase):
    def test_valid_project_reports_valid(self):
        report = ps.validate_project(_project_with_everything())
        self.assertTrue(report["valid"], msg=json.dumps(report["issues"], indent=1))
        for stage in ("domain", "topology", "mesh", "model", "conditions", "approach"):
            self.assertIn(report["stages"][stage]["status"], ("valid", "skipped"))

    def test_bad_domain_is_isolated_to_its_stage(self):
        project = _project_with_everything()
        project["domain"]["rectangle"]["width"] = -1.0
        report = ps.validate_project(project)
        self.assertFalse(report["valid"])
        self.assertEqual(report["stages"]["domain"]["status"], "invalid")

    def test_missing_mesh_is_skipped_not_failed(self):
        project = _project_with_everything()
        project["mesh"] = None
        report = ps.validate_project(project)
        self.assertEqual(report["stages"]["mesh"]["status"], "skipped")

    def test_unknown_approach_reported_in_its_stage(self):
        project = _project_with_everything()
        project["selected_approach"] = "telepathy"
        report = ps.validate_project(project)
        self.assertEqual(report["stages"]["approach"]["status"], "invalid")

    def test_condition_on_unknown_field_is_caught(self):
        project = _project_with_everything()
        project["conditions"] = [{"id": "bad", "type": "dirichlet",
                                 "field": "ghost", "boundary": "left", "value": 1.0}]
        report = ps.validate_project(project)
        self.assertEqual(report["stages"]["conditions"]["status"], "invalid")


class ExportTests(unittest.TestCase):
    def test_topology_csv_export_has_all_three_files(self):
        files = ps.export_topology_csv(_project_with_everything())
        self.assertEqual(set(files), {"nodes.csv", "edges.csv", "cells.csv"})
        self.assertIn("centroid_x", files["cells.csv"])
        self.assertEqual(len(files["nodes.csv"].strip().splitlines()), 5)   # header + 4

    def test_cc3d_package_export_works_and_is_valid_xml(self):
        files = ps.export_cc3d_package(_project_with_everything())
        self.assertIn("Simulation/model.xml", files)
        self.assertIn("project.json", files)
        ET.fromstring(files["Simulation/model.xml"])

    def test_run_metadata_includes_the_snapshot(self):
        manager = rm.RunManager()
        project = _project_with_everything()
        record = manager.create_run(project)
        metadata = ps.export_run_metadata(record)
        self.assertEqual(metadata["approach"], "mpc")
        self.assertIsNotNone(metadata["configuration_snapshot"])
        self.assertEqual(metadata["schema_version"], ps.SCHEMA_VERSION)

    def test_timeseries_export_from_a_control_run(self):
        manager = rm.RunManager()
        record = manager.create_run(_project_with_everything())
        manager.start(record.run_id)
        manager.wait(record.run_id, timeout=60)
        self.assertEqual(record.state, "completed")
        csv_text = ps.export_timeseries_csv(record.results)
        header = csv_text.splitlines()[0]
        for column in ("time", "control", "error", "output", "target"):
            self.assertIn(column, header)
        self.assertGreater(len(csv_text.strip().splitlines()), 2)

    def test_cells_export_from_an_agent_run(self):
        manager = rm.RunManager()
        project = _project_with_everything()
        project["selected_approach"] = "abm"
        record = manager.create_run(project)
        manager.start(record.run_id)
        manager.wait(record.run_id, timeout=120)
        self.assertEqual(record.state, "completed", msg=record.error or "")
        csv_text = ps.export_cells_csv(record.results)
        header = csv_text.splitlines()[0]
        for column in ("time", "id", "type", "parent_id", "generation"):
            self.assertIn(column, header)

    def test_cells_export_can_select_one_frame(self):
        results = {"t": [0, 5], "cells": [[{"id": 1}], [{"id": 1}, {"id": 2}]]}
        one = ps.export_cells_csv(results, frame=1)
        self.assertEqual(len(one.strip().splitlines()), 3)      # header + 2 rows
        with self.assertRaises(IndexError):
            ps.export_cells_csv(results, frame=9)

    def test_bundle_contains_project_and_runs(self):
        manager = rm.RunManager()
        project = _project_with_everything()
        record = manager.create_run(project)
        bundle = ps.export_project_bundle(project, [record])
        self.assertIn("project.json", bundle)
        self.assertIn("nodes.csv", bundle)
        self.assertIn(f"runs/{record.run_id}.json", bundle)
        json.loads(bundle["project.json"])
        json.loads(bundle[f"runs/{record.run_id}.json"])

    def test_history_records_a_run(self):
        manager = rm.RunManager()
        project = _project_with_everything()
        record = manager.create_run(project)
        ps.record_run_in_history(project, record)
        self.assertEqual(len(project["run_history"]), 1)
        self.assertEqual(project["run_history"][0]["approach"], "mpc")


if __name__ == "__main__":
    unittest.main()
