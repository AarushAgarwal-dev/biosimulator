"""
Verification of the 3D path: geometry, the CompuCell3D lattice, meshing, and the
remote run-length cap.

Nothing in this project has ever exercised a z > 1 case, and the CompuCell3D
adapter advertises ``dimensions=(2, 3)``. These tests establish what 3D support
genuinely exists and pin it down.

Convention used here
--------------------
Tests that PASS document behaviour that is already correct. Tests that FAIL are
deliberate: they assert the CORRECT behaviour for a defect that is still present,
and their assertion messages carry the measured evidence needed to fix it. Nothing
in this file is a workaround for a defect -- a failing test here is a report, not a
bug in the test.

Run with:  python -m unittest test_three_d
"""

import io
import os
import tempfile
import unittest
import xml.etree.ElementTree as ET
from unittest.mock import patch

import approach_base as base
import approach_cc3d as cc3d_mod
import cc3d_remote
import geometry as geo
import meshing
import project_schema

# ---------------------------------------------------------------------------
# Detection stubs. detect_cc3d() reads the environment (and, for the remote
# backend, four BIOSIM_* variables), so every test that goes through validate()
# or get_capabilities() pins the backend instead of inheriting the host's.
# ---------------------------------------------------------------------------
ABSENT_DETECTION = {"available": False, "method": "", "version": "", "path": "",
                    "reason": "CompuCell3D is not installed."}

REMOTE_DETECTION = {
    "available": True, "method": "aws-batch", "version": "remote",
    "path": "s3://bucket/cc3d-runs", "reason": "",
    "remote": {"available": True, "configured": True, "reason": "",
               "region": "us-east-2", "job_queue": "q", "job_definition": "d",
               "bucket": "bucket", "prefix": "cc3d-runs"},
}


def _codes(issues):
    return [i.code for i in issues]


def _config(**overrides):
    """A minimally valid cc3d configuration; ``lattice`` may be overridden whole."""
    config = {
        "lattice": {"x": 40, "y": 40, "z": 1},
        "steps": 100,
        "temperature": 10.0,
        "cell_types": [{"type_id": 1, "name": "Tumour"}, {"type_id": 2, "name": "Stroma"}],
        "contact_energies": [{"type1": "Tumour", "type2": "Medium", "energy": 12.0}],
    }
    config.update(overrides)
    return config


def _project(domain=None, **overrides):
    return {
        "domain": domain if domain is not None else geo.make_rectangle(40.0, 40.0),
        "approaches": {"cc3d": _config(**overrides)},
    }


def _merged(**overrides):
    return cc3d_mod._merged(_config(**overrides))


def _xml(**overrides):
    return ET.fromstring(cc3d_mod.CompuCell3DAdapter.build_cc3dml(_merged(**overrides)))


def _box_domain():
    """What a user asking for a 3D domain would plausibly hand the geometry stage.

    There is no constructor for this shape, which is the point: it is written by
    hand because ``geometry`` offers no way to produce one.
    """
    return {
        "kind": "box",
        "name": "Domain",
        "units": "um",
        "box": {"x_min": 0.0, "y_min": 0.0, "z_min": 0.0,
                "width": 40.0, "height": 40.0, "depth": 40.0},
        "regions": [],
        "boundaries": [],
    }


def _buildable_dimensions():
    """Spatial dimensions the product can actually construct a domain for."""
    instances = [
        geo.make_interval(10.0),
        geo.make_rectangle(10.0, 10.0),
        geo.make_circle(5.0),
        geo.make_polygon([[0, 0], [10, 0], [10, 10]]),
    ]
    for instance in instances:
        assert geo.is_valid(geo.validate_domain(instance)), instance["kind"]
    return sorted({geo.domain_dimension(d) for d in instances})


# =============================================================================
# (1) Does a 3D domain type exist at all?
# =============================================================================
class ThreeDDomainSupportTests(unittest.TestCase):
    """The real supported set, documented."""

    def test_supported_domain_kinds_are_1d_and_2d_only(self):
        self.assertEqual(geo.DOMAIN_KINDS, ("interval", "rectangle", "circle", "polygon"))
        self.assertEqual(_buildable_dimensions(), [1, 2],
                         "geometry can construct 1D and 2D domains only")

    def test_geometry_exposes_no_3d_constructor(self):
        missing = [name for name in ("make_box", "make_cube", "make_sphere",
                                     "make_cylinder", "make_extrusion", "make_mesh_3d")
                   if hasattr(geo, name)]
        self.assertEqual(missing, [], "unexpected 3D constructor appeared")

    def test_validate_domain_refuses_a_3d_box_honestly(self):
        issues = geo.validate_domain(_box_domain())
        self.assertIn("domain_kind_unknown", _codes(issues))
        self.assertIn("interval, rectangle, circle, polygon", issues[0].message)

    def test_domain_dimension_can_never_report_3(self):
        for domain in (geo.make_interval(10.0), geo.make_rectangle(10.0, 10.0),
                       geo.make_circle(5.0), geo.make_polygon([[0, 0], [1, 0], [1, 1]])):
            self.assertIn(geo.domain_dimension(domain), (1, 2), domain["kind"])

    def test_domain_dimension_must_not_claim_a_3d_domain_is_2d(self):
        """domain_dimension is ``1 if interval else 2``, so every unrecognised kind
        -- including anything 3D -- is reported as two-dimensional."""
        reported = geo.domain_dimension(_box_domain())
        self.assertNotEqual(
            reported, 2,
            f"domain_dimension() reports {reported} for a 3D 'box' domain. An "
            f"unrecognised or 3D kind must not be silently classified as 2D: every "
            f"caller (meshing.generate_mesh, pde_model, approach_cc3d.validate) "
            f"branches on this value and will treat the box as a flat 2D region.")

    def test_measure_and_containment_are_silently_wrong_for_a_3d_domain(self):
        box = _box_domain()
        self.assertEqual(geo.domain_measure(box), 0.0,
                         "a 3D domain has no measure implementation")
        self.assertFalse(geo.point_in_domain((20.0, 20.0), box),
                         "containment falls through to False for a 3D domain")

    def test_cc3d_advertises_a_dimension_the_product_cannot_build(self):
        """The capability contract must not promise a dimension no domain can express."""
        with patch.object(cc3d_mod, "detect_cc3d", return_value=dict(ABSENT_DETECTION)):
            capabilities = cc3d_mod.CC3D_ADAPTER.get_capabilities()
        buildable = _buildable_dimensions()
        unsupportable = [d for d in capabilities.dimensions if d not in buildable]
        self.assertEqual(
            unsupportable, [],
            f"CompuCell3D advertises dimensions={list(capabilities.dimensions)} but "
            f"geometry can only build {buildable}. Dimension(s) {unsupportable} are "
            f"advertised in Capabilities.to_dict() (and therefore to the UI) with no "
            f"domain kind, no mesher and no validation behind them.")

    def test_the_other_two_approaches_do_not_over_advertise(self):
        """ABM claims (2,) and MPC claims (0,) -- both honest. Contrast for the report."""
        buildable = _buildable_dimensions()
        for approach_id in ("abm", "mpc"):
            capabilities = base.get_approach(approach_id).get_capabilities()
            over = [d for d in capabilities.dimensions if d not in buildable and d != 0]
            self.assertEqual(over, [], f"{approach_id} over-advertises {over}")


# =============================================================================
# (2) build_cc3dml with z > 1
# =============================================================================
class CC3DMLThreeDLatticeTests(unittest.TestCase):
    def test_potts_dimensions_carry_z(self):
        potts = _xml(lattice={"x": 40, "y": 40, "z": 40}).find("Potts")
        dimensions = potts.find("Dimensions")
        self.assertIsNotNone(dimensions, "Potts must carry a <Dimensions> element")
        self.assertEqual((dimensions.get("x"), dimensions.get("y"), dimensions.get("z")),
                         ("40", "40", "40"))

    def test_z_is_not_clamped_or_dropped_for_an_asymmetric_3d_lattice(self):
        dimensions = _xml(lattice={"x": 30, "y": 20, "z": 7}).find("Potts/Dimensions")
        self.assertEqual((dimensions.get("x"), dimensions.get("y"), dimensions.get("z")),
                         ("30", "20", "7"))

    def test_undocumented_dimension_elements_are_not_emitted_in_3d(self):
        potts = _xml(lattice={"x": 40, "y": 40, "z": 40}).find("Potts")
        for tag in ("DimensionX", "DimensionY", "DimensionZ"):
            self.assertIsNone(potts.find(tag), f"<{tag}> must not be emitted")

    def test_neighbor_order_is_a_legal_3d_value(self):
        """CC3D accepts orders 1-3 on a 3D lattice (6 / 18 / 26 neighbours)."""
        order = int(_xml(lattice={"x": 40, "y": 40, "z": 40}).find("Potts/NeighborOrder").text)
        self.assertIn(order, (1, 2, 3),
                      f"NeighborOrder {order} is not a valid 3D Potts neighbourhood")

    def test_neighbor_order_is_the_same_value_in_2d_and_3d(self):
        """Documented, not asserted as a defect: order 2 is legal in both, so the
        z-independent default is defensible -- but nothing adapts it either."""
        flat = _xml(lattice={"x": 40, "y": 40, "z": 1}).find("Potts/NeighborOrder").text
        cube = _xml(lattice={"x": 40, "y": 40, "z": 40}).find("Potts/NeighborOrder").text
        self.assertEqual(flat, cube)
        self.assertEqual(cube, "2")

    def test_contact_plugin_neighbor_order_matches_potts_in_3d(self):
        root = _xml(lattice={"x": 40, "y": 40, "z": 40})
        potts_order = root.find("Potts/NeighborOrder").text
        contact = [p for p in root.iter("Plugin") if p.get("Name") == "Contact"][0]
        self.assertEqual(contact.find("NeighborOrder").text, potts_order)

    def test_blob_initializer_centre_is_mid_lattice_in_z(self):
        centre = _xml(lattice={"x": 40, "y": 40, "z": 40}).find(
            ".//Steppable[@Type='BlobInitializer']/Region/Center")
        self.assertEqual((centre.get("x"), centre.get("y"), centre.get("z")),
                         ("20", "20", "20"))

    def test_blob_initializer_centre_z_is_zero_on_a_flat_lattice(self):
        """z=1 must seed on plane 0, not plane 1 -- CC3D indexes from 0."""
        centre = _xml(lattice={"x": 40, "y": 40, "z": 1}).find(
            ".//Steppable[@Type='BlobInitializer']/Region/Center")
        self.assertEqual(centre.get("z"), "0")

    def test_steppable_records_the_z_coordinate(self):
        source = cc3d_mod.CompuCell3DAdapter.build_steppable(_merged(
            lattice={"x": 40, "y": 40, "z": 40}))
        compile(source, "steppables.py", "exec")
        self.assertIn("mcs,cell_id,type,volume,surface,x,y,z", source)
        self.assertIn("cell.zCOM", source)

    def test_parse_output_preserves_z(self):
        csv_text = ("mcs,cell_id,type,volume,surface,x,y,z\n"
                    "0,1,Tumour,125,150,20.0,20.0,19.5\n"
                    "0,2,Tumour,124,149,21.0,20.0,21.5\n")
        with tempfile.TemporaryDirectory(prefix="three_d_") as directory:
            path = os.path.join(directory, "cells.csv")
            with open(path, "w", encoding="utf-8", newline="") as handle:
                handle.write(csv_text)
            results = cc3d_mod.CompuCell3DAdapter.parse_output(path)
        self.assertEqual([c["z"] for c in results["cells"][0]], [19.5, 21.5])


# =============================================================================
# (3) Volume constraint and cell-type mapping in 3D
# =============================================================================
class ThreeDVolumeAndCellTypeTests(unittest.TestCase):
    """The seeded cell size is ``Width ** dimension``. TargetVolume must agree with
    it, or every cell is born far off target and the volume constraint fights the
    initialiser from step 0."""

    @staticmethod
    def _volume_plugin(root):
        return [p for p in root.iter("Plugin") if p.get("Name") == "Volume"][0]

    def _seed_width(self, root):
        return int(root.find(".//Steppable[@Type='BlobInitializer']/Region/Width").text)

    def test_default_target_volume_is_a_2d_area(self):
        """Proof that 25.0 was chosen for 2D: it is exactly width**2."""
        root = _xml(lattice={"x": 40, "y": 40, "z": 1})
        target = float(self._volume_plugin(root).find("TargetVolume").text)
        self.assertEqual(target, float(self._seed_width(root) ** 2))

    def test_default_target_volume_must_be_a_volume_on_a_3d_lattice(self):
        root = _xml(lattice={"x": 40, "y": 40, "z": 40})
        width = self._seed_width(root)
        target = float(self._volume_plugin(root).find("TargetVolume").text)
        expected = float(width ** 3)
        self.assertAlmostEqual(
            target, expected, places=6,
            msg=(f"On a 3D lattice the BlobInitializer seeds cells of "
                 f"{width}x{width}x{width} = {width ** 3} sites, but the default "
                 f"TargetVolume is {target:g} -- the 2D area {width}x{width}. Every "
                 f"seeded cell starts {width ** 3 / target:.0f}x over target, so the "
                 f"volume constraint immediately crushes it. "
                 f"DEFAULTS['volume_constraint']['target_volume'] in approach_cc3d.py "
                 f"is dimension-blind."))

    def test_an_explicit_target_volume_is_honoured_in_3d(self):
        root = _xml(lattice={"x": 40, "y": 40, "z": 40},
                    volume_constraint={"target_volume": 125.0, "lambda_volume": 2.0})
        plugin = self._volume_plugin(root)
        self.assertEqual(float(plugin.find("TargetVolume").text), 125.0)
        self.assertEqual(float(plugin.find("LambdaVolume").text), 2.0)

    def test_blob_radius_default_must_fit_inside_the_thinnest_axis(self):
        """The auto radius is derived from x alone: ``max(4, x // 6)``."""
        lattice = {"x": 120, "y": 120, "z": 10}
        root = _xml(lattice=lattice)
        radius = int(root.find(".//Steppable[@Type='BlobInitializer']/Region/Radius").text)
        half_thinnest = min(lattice.values()) // 2
        self.assertLessEqual(
            radius, half_thinnest,
            f"Blob radius {radius} exceeds half the thinnest lattice axis "
            f"({half_thinnest}, from z={lattice['z']}) on lattice {lattice}. The "
            f"default is max(4, x // 6) = {max(4, lattice['x'] // 6)} and ignores y "
            f"and z entirely, so on a slab the initial blob is clipped by the z "
            f"boundary and the seeded population is smaller than configured.")

    def test_blob_radius_default_is_safe_on_a_cubic_lattice(self):
        """Scoped evidence: the radius defect needs an anisotropic lattice."""
        lattice = {"x": 40, "y": 40, "z": 40}
        radius = int(_xml(lattice=lattice).find(
            ".//Steppable[@Type='BlobInitializer']/Region/Radius").text)
        self.assertLessEqual(radius, min(lattice.values()) // 2)

    def test_per_type_target_volume_is_not_silently_dropped(self):
        """A per-type target volume matters more in 3D, where types differ by a cube
        rather than a square. It is neither emitted nor reported."""
        config = _config(
            lattice={"x": 40, "y": 40, "z": 40},
            cell_types=[{"type_id": 1, "name": "Tumour", "target_volume": 125.0},
                        {"type_id": 2, "name": "Stroma", "target_volume": 64.0}])
        root = ET.fromstring(cc3d_mod.CompuCell3DAdapter.build_cc3dml(
            cc3d_mod._merged(config)))
        with patch.object(cc3d_mod, "detect_cc3d", return_value=dict(ABSENT_DETECTION)):
            issues = cc3d_mod.CC3D_ADAPTER.validate({"approaches": {"cc3d": config}})
        emitted = ET.tostring(root, encoding="unicode")
        honoured = "125" in emitted and "64" in emitted
        reported = any("volume" in i.code for i in issues)
        self.assertTrue(
            honoured or reported,
            "Per-cell-type target_volume values (125.0, 64.0) appear nowhere in the "
            "generated CC3DML and raise no validation issue. build_cc3dml maps "
            "cell_types to TypeId/TypeName only and emits ONE global "
            "<Plugin Name='Volume'>, so every type silently shares the global target. "
            "Either emit VolumeEnergyParameters per type or refuse the input.")

    def test_cell_type_ids_and_medium_are_mapped_in_3d(self):
        root = _xml(lattice={"x": 40, "y": 40, "z": 40})
        mapped = {e.get("TypeName"): e.get("TypeId") for e in root.iter("CellType")}
        self.assertEqual(mapped, {"Medium": "0", "Tumour": "1", "Stroma": "2"})

    def test_blob_types_list_is_populated_in_3d(self):
        types = _xml(lattice={"x": 40, "y": 40, "z": 40}).find(
            ".//Steppable[@Type='BlobInitializer']/Region/Types").text
        self.assertEqual(types, "Tumour,Stroma")

    def test_a_config_with_no_cell_types_is_refused_in_3d_too(self):
        with self.assertRaises(ValueError) as raised:
            cc3d_mod.CompuCell3DAdapter.build_cc3dml(
                cc3d_mod._merged(_config(lattice={"x": 40, "y": 40, "z": 40},
                                         cell_types=[])))
        self.assertIn("zero cells", str(raised.exception))


# =============================================================================
# (3b) The domain and the lattice can silently disagree about dimension
# =============================================================================
class DomainLatticeConsistencyTests(unittest.TestCase):
    def test_a_1d_domain_is_warned_about(self):
        """The warning machinery exists and fires for 1D."""
        with patch.object(cc3d_mod, "detect_cc3d", return_value=dict(ABSENT_DETECTION)):
            issues = cc3d_mod.CC3D_ADAPTER.validate(
                _project(domain=geo.make_interval(40.0), lattice={"x": 40, "y": 1, "z": 1}))
        self.assertIn("cc3d_domain_1d", _codes(issues))

    def test_a_3d_lattice_under_a_2d_domain_must_be_flagged(self):
        project = _project(domain=geo.make_rectangle(40.0, 40.0),
                           lattice={"x": 40, "y": 40, "z": 40})
        with patch.object(cc3d_mod, "detect_cc3d", return_value=dict(ABSENT_DETECTION)):
            issues = cc3d_mod.CC3D_ADAPTER.validate(project)
        flagged = [i for i in issues
                   if "3d" in i.code.lower() or "dimension" in i.code.lower()
                   or "lattice_z" in i.code.lower()]
        self.assertTrue(
            flagged,
            f"A 40x40x40 CompuCell3D lattice under a 2D rectangle domain validates "
            f"with {len(issues)} issues and none of them mention the mismatch. The "
            f"domain, the mesh and the boundary conditions are all 2D while the "
            f"engine is asked to run a 64000-site 3D lattice: the extruded axis has "
            f"no geometry, no boundary tags and no PDE support behind it. Validation "
            f"already warns about a 1D domain (cc3d_domain_1d), so the same channel "
            f"should carry this.")

    def test_a_2d_lattice_under_a_2d_domain_is_clean(self):
        with patch.object(cc3d_mod, "detect_cc3d", return_value=dict(ABSENT_DETECTION)):
            issues = cc3d_mod.CC3D_ADAPTER.validate(_project())
        self.assertEqual([i for i in issues if i.severity == "error"], [])

    def test_lattice_z_below_one_is_still_rejected(self):
        with patch.object(cc3d_mod, "detect_cc3d", return_value=dict(ABSENT_DETECTION)):
            issues = cc3d_mod.CC3D_ADAPTER.validate(
                _project(lattice={"x": 40, "y": 40, "z": 0}))
        self.assertIn("cc3d_lattice_z_too_small", _codes(issues))

    def test_a_non_integer_z_is_rejected(self):
        with patch.object(cc3d_mod, "detect_cc3d", return_value=dict(ABSENT_DETECTION)):
            issues = cc3d_mod.CC3D_ADAPTER.validate(
                _project(lattice={"x": 40, "y": 40, "z": "deep"}))
        self.assertIn("cc3d_lattice_z_invalid", _codes(issues))


# =============================================================================
# (4) Does meshing refuse 3D cleanly, or fall back to a flat mesh?
# =============================================================================
class MeshingThreeDRefusalTests(unittest.TestCase):
    def test_generate_mesh_refuses_a_3d_domain_rather_than_meshing_it_flat(self):
        """An honest refusal is the pass condition; a degenerate 2D mesh is not."""
        with self.assertRaises((ValueError, RuntimeError, NotImplementedError)) as raised:
            mesh = meshing.generate_mesh(_box_domain(), {"target_spacing": 5.0})
            self.fail(f"generate_mesh returned a {mesh.get('dimension')}D "
                      f"{mesh.get('element_kind')} mesh with "
                      f"{len(mesh.get('nodes') or [])} nodes for a 3D domain instead "
                      f"of refusing -- a silent 2D fallback.")
        self.assertTrue(str(raised.exception))

    def test_unstructured_mesher_refuses_a_3d_domain(self):
        with self.assertRaises((ValueError, RuntimeError)):
            meshing.mesh_2d_unstructured(_box_domain(), 5.0)

    def test_structured_mesher_refuses_a_3d_domain(self):
        with self.assertRaises(ValueError) as raised:
            meshing.mesh_2d_structured(_box_domain(), rows=4, cols=4)
        self.assertIn("rectangular domain", str(raised.exception))

    def test_the_1d_mesher_refuses_a_3d_domain(self):
        with self.assertRaises(ValueError):
            meshing.mesh_1d(_box_domain(), element_count=4)

    def test_the_3d_refusal_must_name_the_real_reason(self):
        """The refusal is reached by accident: domain_dimension() says 2, the
        rectangle branch is skipped, and domain_bounds() returns all zeros for an
        unknown kind, so the unstructured mesher complains about extent."""
        with self.assertRaises((ValueError, RuntimeError)) as raised:
            meshing.generate_mesh(_box_domain(), {"target_spacing": 5.0})
        message = str(raised.exception).lower()
        self.assertTrue(
            any(token in message for token in ("3d", "three-dimensional", "unsupported",
                                               "not supported", "box", "kind")),
            f"generate_mesh refuses a 3D domain with {str(raised.exception)!r}, which "
            f"describes neither the domain kind nor the dimension. The box carries "
            f"width/height/depth of 40, so 'no extent' is misleading: a user reads it "
            f"as bad numbers rather than 'this product has no 3D mesher'.")

    def test_a_mesh_cannot_represent_the_3d_lattice_cc3d_would_run(self):
        """Even the supported path is 2D-only: nodes are [x, y] pairs."""
        mesh = meshing.generate_mesh(geo.make_rectangle(40.0, 40.0), {"rows": 4, "cols": 4})
        self.assertEqual(mesh["dimension"], 2)
        self.assertTrue(all(len(node) == 2 for node in mesh["nodes"]),
                        "mesh nodes carry no third coordinate")
        self.assertNotIn(mesh["element_kind"], ("tet4", "hex8"),
                         "no volumetric element type exists")

    def test_no_3d_element_kinds_or_volumetric_mesher_exist(self):
        present = [name for name in ("mesh_3d", "mesh_3d_structured",
                                     "mesh_3d_unstructured", "tetrahedralize")
                   if hasattr(meshing, name)]
        self.assertEqual(present, [], "unexpected 3D mesher appeared")


# =============================================================================
# (5) Does the run-length / cost estimator scale with z?
# =============================================================================
class RunLengthEstimatorThreeDTests(unittest.TestCase):
    def test_site_count_multiplies_all_three_axes(self):
        self.assertEqual(cc3d_remote.lattice_site_count({"x": 40, "y": 40, "z": 40}), 64000)
        self.assertEqual(cc3d_remote.lattice_site_count({"x": 40, "y": 40, "z": 1}), 1600)

    def test_a_missing_z_defaults_to_one_layer(self):
        self.assertEqual(cc3d_remote.lattice_site_count({"x": 40, "y": 40}), 1600)

    def test_runtime_estimate_scales_linearly_with_z(self):
        overhead = cc3d_remote.REMOTE_STARTUP_OVERHEAD_MINUTES
        flat = cc3d_remote.estimate_runtime_minutes(500, {"x": 40, "y": 40, "z": 1}) - overhead
        cube = cc3d_remote.estimate_runtime_minutes(500, {"x": 40, "y": 40, "z": 40}) - overhead
        self.assertAlmostEqual(
            cube, flat * 40.0, places=6,
            msg=f"40x40x40 estimated at {cube:.4f} min of compute vs {flat:.4f} min "
                f"for 40x40x1 -- a 3D lattice must cost z times a single layer.")

    def test_max_steps_within_cap_shrinks_by_z(self):
        flat = cc3d_remote.max_steps_within_cap({"x": 40, "y": 40, "z": 1})
        cube = cc3d_remote.max_steps_within_cap({"x": 40, "y": 40, "z": 40})
        self.assertGreater(flat, 0)
        self.assertAlmostEqual(cube, flat // 40, delta=max(1, flat // 4000),
                               msg=f"cap allows {flat} steps flat but {cube} at z=40")

    def test_the_cap_engages_for_a_3d_lattice_a_2d_one_would_pass(self):
        """Same step count, same x and y: only the extruded axis pushes it over."""
        steps, lattice_2d = 20_000, {"x": 100, "y": 100, "z": 1}
        lattice_3d = {"x": 100, "y": 100, "z": 100}
        flat = cc3d_remote.run_length_check(steps, lattice_2d)
        cube = cc3d_remote.run_length_check(steps, lattice_3d)
        self.assertTrue(flat["within_cap"],
                        f"2D control case must pass: {flat['estimated_minutes']} min")
        self.assertFalse(cube["within_cap"],
                         f"100x100x100 x {steps} steps estimated at only "
                         f"{cube['estimated_minutes']} min")
        self.assertEqual(cube["lattice_sites"], 1_000_000)
        self.assertLess(cube["max_steps"], flat["max_steps"],
                        "the permitted step count must fall as z grows")
        self.assertIn("90 minute cap", cube["message"])
        self.assertIn(str(cube["max_steps"]), cube["message"])

    def test_diffusion_fields_add_cost_on_a_3d_lattice(self):
        lattice = {"x": 40, "y": 40, "z": 40}
        without = cc3d_remote.estimate_runtime_minutes(500, lattice, field_count=0)
        with_two = cc3d_remote.estimate_runtime_minutes(500, lattice, field_count=2)
        self.assertGreater(with_two, without)

    def test_cost_estimate_grows_with_a_3d_runtime(self):
        lattice = {"x": 60, "y": 60, "z": 60}
        minutes = cc3d_remote.estimate_runtime_minutes(1000, lattice)
        cheap = cc3d_remote.estimate_cost(4, 15500, 1.0)["estimated_usd"]
        real = cc3d_remote.estimate_cost(4, 15500, minutes)["estimated_usd"]
        self.assertGreater(real, cheap)

    def test_remote_validation_refuses_an_oversized_3d_run(self):
        project = _project(lattice={"x": 200, "y": 200, "z": 200}, steps=1000)
        with patch.object(cc3d_mod, "detect_cc3d", return_value=dict(REMOTE_DETECTION)):
            issues = cc3d_mod.CC3D_ADAPTER.validate(project)
        self.assertIn("cc3d_remote_run_too_long", _codes(issues))

    def test_a_local_backend_does_not_apply_the_remote_cap_to_a_3d_run(self):
        local = {"available": True, "method": "runscript-path", "version": "unknown",
                 "path": "/usr/bin/runScript.sh", "reason": ""}
        project = _project(lattice={"x": 200, "y": 200, "z": 200}, steps=1000)
        with patch.object(cc3d_mod, "detect_cc3d", return_value=local):
            issues = cc3d_mod.CC3D_ADAPTER.validate(project)
        self.assertNotIn("cc3d_remote_run_too_long", _codes(issues))


# =============================================================================
# End to end, plus the export path
# =============================================================================
class ThreeDEndToEndTests(unittest.TestCase):
    def test_compile_of_a_3d_configuration_succeeds_and_carries_z(self):
        with patch.object(cc3d_mod, "detect_cc3d", return_value=dict(ABSENT_DETECTION)):
            compiled = cc3d_mod.CC3D_ADAPTER.compile(
                _project(lattice={"x": 40, "y": 40, "z": 40}))
        self.assertEqual(compiled["config"]["lattice"], {"x": 40, "y": 40, "z": 40})
        dimensions = ET.fromstring(compiled["cc3dml"]).find("Potts/Dimensions")
        self.assertEqual(dimensions.get("z"), "40")

    def test_exported_3d_package_is_well_formed(self):
        with patch.object(cc3d_mod, "detect_cc3d", return_value=dict(ABSENT_DETECTION)):
            exported = cc3d_mod.CC3D_ADAPTER.export_configuration(
                _project(lattice={"x": 40, "y": 40, "z": 40}))
        root = ET.fromstring(exported["files"]["Simulation/model.xml"])
        self.assertEqual(root.find("Potts/Dimensions").get("z"), "40")
        compile(exported["files"]["Simulation/steppables.py"], "s.py", "exec")

    def test_cells_csv_export_must_not_drop_the_z_coordinate(self):
        results = {
            "t": [0],
            "cells": [[{"id": 1, "type": "Tumour", "state": "alive", "alive": True,
                        "x": 20.0, "y": 20.0, "z": 19.5, "volume": 125.0,
                        "surface": 150.0, "age": None, "parent_id": None}]],
        }
        csv_text = project_schema.export_cells_csv(results)
        header = csv_text.splitlines()[0].split(",")
        self.assertIn(
            "z", header,
            f"export_cells_csv emits columns {header} -- there is no z column, so "
            f"every cell's third coordinate is discarded on export. A 3D "
            f"CompuCell3D run parses z correctly (parse_output keeps it) and then "
            f"loses it at the export boundary, making a 3D result "
            f"indistinguishable from a 2D one downstream. The column list is in "
            f"project_schema.export_cells_csv.")


if __name__ == "__main__":
    unittest.main()
