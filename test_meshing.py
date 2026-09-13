"""Unit tests for mesh generation and discretisation."""

import unittest

import numpy as np

import geometry as geo
import meshing


def _codes(issues):
    return {i.code for i in issues}


class Mesh1DTests(unittest.TestCase):
    def test_element_count_is_honoured(self):
        mesh = meshing.mesh_1d(geo.make_interval(10.0), element_count=5)
        self.assertEqual(len(mesh["nodes"]), 6)
        self.assertEqual(len(mesh["elements"]), 5)
        self.assertAlmostEqual(mesh["spacing"], 2.0)
        self.assertEqual(mesh["element_kind"], "line2")

    def test_target_spacing_is_honoured(self):
        mesh = meshing.mesh_1d(geo.make_interval(10.0), target_spacing=0.5)
        self.assertEqual(len(mesh["elements"]), 20)
        self.assertAlmostEqual(sum(mesh["element_sizes"]), 10.0, places=9)

    def test_nodes_are_ordered_and_span_the_interval(self):
        mesh = meshing.mesh_1d(geo.make_interval(4.0, x_min=1.0), element_count=4)
        xs = [n[0] for n in mesh["nodes"]]
        self.assertEqual(xs, sorted(xs))
        self.assertAlmostEqual(xs[0], 1.0)
        self.assertAlmostEqual(xs[-1], 5.0)

    def test_both_ends_are_tagged(self):
        mesh = meshing.mesh_1d(geo.make_interval(10.0), element_count=4)
        boundaries = mesh["boundaries"]
        self.assertEqual(boundaries["left"]["nodes"], [0])
        self.assertEqual(boundaries["right"]["nodes"], [4])

    def test_bad_settings_are_refused(self):
        with self.assertRaises(ValueError):
            meshing.mesh_1d(geo.make_interval(10.0), element_count=0)
        with self.assertRaises(ValueError):
            meshing.mesh_1d(geo.make_interval(10.0), target_spacing=-1.0)
        with self.assertRaises(ValueError):
            meshing.mesh_1d(geo.make_interval(10.0))          # neither given
        with self.assertRaises(ValueError):
            meshing.mesh_1d(geo.make_rectangle(2.0, 2.0), element_count=3)


class Mesh2DStructuredTests(unittest.TestCase):
    def test_grid_shape_and_counts(self):
        mesh = meshing.mesh_2d_structured(geo.make_rectangle(4.0, 2.0), rows=2, cols=4)
        self.assertEqual(len(mesh["nodes"]), 15)          # (2+1) * (4+1)
        self.assertEqual(len(mesh["elements"]), 8)
        self.assertEqual(mesh["element_kind"], "quad4")
        self.assertAlmostEqual(mesh["dx"], 1.0)
        self.assertAlmostEqual(mesh["dy"], 1.0)

    def test_element_areas_sum_to_the_domain_area(self):
        domain = geo.make_rectangle(6.0, 3.0)
        mesh = meshing.mesh_2d_structured(domain, rows=3, cols=6)
        self.assertAlmostEqual(sum(mesh["element_sizes"]), geo.domain_measure(domain), places=9)

    def test_all_four_sides_are_tagged(self):
        mesh = meshing.mesh_2d_structured(geo.make_rectangle(4.0, 2.0), rows=2, cols=4)
        boundaries = mesh["boundaries"]
        self.assertEqual(len(boundaries["left"]["nodes"]), 3)     # rows+1
        self.assertEqual(len(boundaries["right"]["nodes"]), 3)
        self.assertEqual(len(boundaries["bottom"]["nodes"]), 5)   # cols+1
        self.assertEqual(len(boundaries["top"]["nodes"]), 5)

    def test_target_spacing_path(self):
        mesh = meshing.mesh_2d_structured(geo.make_rectangle(10.0, 5.0), target_spacing=1.0)
        self.assertEqual(mesh["cols"], 10)
        self.assertEqual(mesh["rows"], 5)

    def test_structured_grid_refuses_a_circle(self):
        with self.assertRaises(ValueError) as raised:
            meshing.mesh_2d_structured(geo.make_circle(5.0), rows=3, cols=3)
        self.assertIn("unstructured", str(raised.exception).lower())

    def test_all_nodes_lie_in_the_domain(self):
        domain = geo.make_rectangle(4.0, 4.0)
        mesh = meshing.mesh_2d_structured(domain, rows=4, cols=4)
        self.assertTrue(geo.is_valid(meshing.validate_mesh(mesh, domain)))


@unittest.skipUnless(meshing._HAS_DELAUNAY, "scipy.spatial.Delaunay unavailable")
class Mesh2DUnstructuredTests(unittest.TestCase):
    def test_circle_is_triangulated_and_stays_inside(self):
        domain = geo.make_circle(5.0)
        mesh = meshing.mesh_2d_unstructured(domain, target_spacing=1.0)
        self.assertEqual(mesh["element_kind"], "tri3")
        self.assertGreater(len(mesh["elements"]), 20)
        for x, y in mesh["nodes"]:
            self.assertLessEqual(np.hypot(x, y), 5.0 + 1e-6)

    def test_triangulated_area_approximates_the_domain(self):
        domain = geo.make_circle(5.0)
        mesh = meshing.mesh_2d_unstructured(domain, target_spacing=0.5)
        covered = sum(mesh["element_sizes"])
        self.assertGreater(covered / geo.domain_measure(domain), 0.90)
        self.assertLessEqual(covered / geo.domain_measure(domain), 1.0 + 1e-9)

    def test_polygon_is_triangulated(self):
        domain = geo.make_polygon([[0, 0], [10, 0], [10, 6], [0, 6]])
        mesh = meshing.mesh_2d_unstructured(domain, target_spacing=1.0)
        self.assertGreater(len(mesh["elements"]), 10)
        self.assertTrue(geo.is_valid(meshing.validate_mesh(mesh, domain)))

    def test_perimeter_is_tagged(self):
        domain = geo.make_circle(4.0)
        mesh = meshing.mesh_2d_unstructured(domain, target_spacing=1.0)
        self.assertGreater(len(mesh["boundaries"]["perimeter"]["nodes"]), 8)

    def test_quality_statistics_are_reported(self):
        mesh = meshing.mesh_2d_unstructured(geo.make_circle(5.0), target_spacing=1.0)
        stats = mesh["stats"]
        self.assertIn("min_angle_deg", stats)
        self.assertGreater(stats["min_angle_deg"], 0.0)
        self.assertEqual(stats["degenerate_elements"], 0)

    def test_generation_is_deterministic_for_a_seed(self):
        domain = geo.make_circle(4.0)
        first = meshing.mesh_2d_unstructured(domain, target_spacing=1.0, seed=7)
        second = meshing.mesh_2d_unstructured(domain, target_spacing=1.0, seed=7)
        self.assertEqual(first["nodes"], second["nodes"])
        self.assertEqual(first["elements"], second["elements"])

    def test_too_fine_a_spacing_is_refused_before_allocating(self):
        with self.assertRaises(ValueError) as raised:
            meshing.mesh_2d_unstructured(geo.make_rectangle(1000.0, 1000.0), target_spacing=0.5)
        self.assertIn("limit", str(raised.exception))


class BoundaryPersistenceTests(unittest.TestCase):
    def test_named_boundaries_survive_remeshing(self):
        """The point of deriving tags geometrically: indices change, names do not.

        A boundary condition attached to 'left' must still mean the left edge after
        the mesh is regenerated at a different resolution.
        """
        domain = geo.make_rectangle(4.0, 2.0)
        coarse = meshing.mesh_2d_structured(domain, rows=2, cols=4)
        fine = meshing.mesh_2d_structured(domain, rows=8, cols=16)

        self.assertEqual(set(coarse["boundaries"]), set(fine["boundaries"]))
        self.assertNotEqual(coarse["boundaries"]["left"]["nodes"],
                            fine["boundaries"]["left"]["nodes"])

        # Every tagged node must actually sit on the named side, at both resolutions.
        for mesh in (coarse, fine):
            for index in mesh["boundaries"]["left"]["nodes"]:
                self.assertAlmostEqual(mesh["nodes"][index][0], 0.0, places=9)
            for index in mesh["boundaries"]["top"]["nodes"]:
                self.assertAlmostEqual(mesh["nodes"][index][1], 2.0, places=9)

    def test_1d_ends_survive_refinement(self):
        domain = geo.make_interval(10.0)
        coarse = meshing.mesh_1d(domain, element_count=4)
        fine = meshing.mesh_1d(domain, element_count=40)
        self.assertEqual(coarse["boundaries"]["right"]["nodes"], [4])
        self.assertEqual(fine["boundaries"]["right"]["nodes"], [40])
        self.assertAlmostEqual(fine["nodes"][40][0], 10.0)


class MeshValidationTests(unittest.TestCase):
    def test_empty_mesh_is_reported(self):
        self.assertIn("mesh_empty", _codes(meshing.validate_mesh({"nodes": [], "elements": []})))

    def test_bad_element_reference_is_reported(self):
        mesh = {"nodes": [[0, 0], [1, 0], [0, 1]], "elements": [[0, 1, 99]],
                "element_kind": "tri3", "element_sizes": [0.5], "dimension": 2}
        self.assertIn("mesh_reference_invalid", _codes(meshing.validate_mesh(mesh)))

    def test_repeated_node_in_element_is_reported(self):
        mesh = {"nodes": [[0, 0], [1, 0], [0, 1]], "elements": [[0, 1, 1]],
                "element_kind": "tri3", "element_sizes": [0.0], "dimension": 2}
        codes = _codes(meshing.validate_mesh(mesh))
        self.assertIn("mesh_element_repeated_node", codes)
        self.assertIn("mesh_degenerate_elements", codes)

    def test_valid_structured_mesh_has_no_issues(self):
        domain = geo.make_rectangle(3.0, 3.0)
        mesh = meshing.mesh_2d_structured(domain, rows=3, cols=3)
        self.assertEqual(meshing.validate_mesh(mesh, domain), [])


class DispatchTests(unittest.TestCase):
    def test_interval_dispatches_to_1d(self):
        mesh = meshing.generate_mesh(geo.make_interval(5.0), {"element_count": 5})
        self.assertEqual(mesh["kind"], "1d")

    def test_rectangle_defaults_to_structured(self):
        mesh = meshing.generate_mesh(geo.make_rectangle(4.0, 4.0), {"target_spacing": 1.0})
        self.assertEqual(mesh["kind"], "structured")

    @unittest.skipUnless(meshing._HAS_DELAUNAY, "scipy.spatial.Delaunay unavailable")
    def test_rectangle_can_be_forced_unstructured(self):
        mesh = meshing.generate_mesh(geo.make_rectangle(4.0, 4.0),
                                     {"target_spacing": 1.0, "kind": "unstructured"})
        self.assertEqual(mesh["kind"], "unstructured")

    @unittest.skipUnless(meshing._HAS_DELAUNAY, "scipy.spatial.Delaunay unavailable")
    def test_circle_dispatches_to_unstructured_with_a_default_spacing(self):
        mesh = meshing.generate_mesh(geo.make_circle(5.0))
        self.assertEqual(mesh["kind"], "unstructured")
        self.assertGreater(len(mesh["elements"]), 4)


class ScaleTests(unittest.TestCase):
    def test_target_editor_size_is_fast_enough(self):
        """~1000 nodes is the stated editor target."""
        domain = geo.make_rectangle(31.0, 31.0)
        mesh = meshing.mesh_2d_structured(domain, rows=31, cols=31)
        self.assertEqual(mesh["stats"]["node_count"], 1024)
        self.assertEqual(mesh["stats"]["estimated_cost_class"], "small")
        self.assertEqual(meshing.validate_mesh(mesh, domain), [])


if __name__ == "__main__":
    unittest.main()
