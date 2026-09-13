"""Unit tests for the geometry and topology preparation stages."""

import unittest

import geometry as geo
import topology as topo


def _codes(issues):
    return {i.code for i in issues}


class DomainValidationTests(unittest.TestCase):
    def test_valid_domains_pass(self):
        for domain in (
            geo.make_interval(100.0),
            geo.make_rectangle(40.0, 25.0),
            geo.make_circle(12.5),
            geo.make_polygon([[0, 0], [10, 0], [10, 8], [0, 8]]),
        ):
            self.assertTrue(geo.is_valid(geo.validate_domain(domain)),
                            msg=f"{domain['kind']} should be valid: "
                                f"{geo.issues_to_dicts(geo.validate_domain(domain))}")

    def test_zero_and_negative_dimensions_are_rejected(self):
        self.assertIn("length_not_positive", _codes(geo.validate_domain(geo.make_interval(0.0))))
        self.assertIn("length_not_positive", _codes(geo.validate_domain(geo.make_interval(-5.0))))
        self.assertIn("width_not_positive", _codes(geo.validate_domain(geo.make_rectangle(0.0, 5.0))))
        self.assertIn("height_not_positive", _codes(geo.validate_domain(geo.make_rectangle(5.0, -1.0))))
        self.assertIn("radius_not_positive", _codes(geo.validate_domain(geo.make_circle(0.0))))

    def test_missing_units_is_an_error(self):
        domain = geo.make_rectangle(10.0, 10.0)
        domain["units"] = ""
        self.assertIn("units_missing", _codes(geo.validate_domain(domain)))

    def test_unknown_units_is_an_error(self):
        domain = geo.make_rectangle(10.0, 10.0, units="furlongs")
        self.assertIn("units_unknown", _codes(geo.validate_domain(domain)))

    def test_unknown_kind_is_reported_once(self):
        issues = geo.validate_domain({"kind": "torus", "units": "um"})
        self.assertEqual(_codes(issues), {"domain_kind_unknown"})

    def test_polygon_needs_three_points(self):
        self.assertIn("polygon_too_few_points",
                      _codes(geo.validate_domain(geo.make_polygon([[0, 0], [1, 1]]))))

    def test_duplicate_polygon_points_are_rejected(self):
        issues = geo.validate_domain(geo.make_polygon([[0, 0], [5, 0], [5, 0], [5, 5]]))
        self.assertIn("duplicate_point", _codes(issues))

    def test_self_intersecting_polygon_is_rejected(self):
        # A bow-tie: edges cross in the middle.
        bowtie = geo.make_polygon([[0, 0], [10, 10], [10, 0], [0, 10]])
        self.assertIn("polygon_self_intersecting", _codes(geo.validate_domain(bowtie)))

    def test_collinear_polygon_has_no_area(self):
        line = geo.make_polygon([[0, 0], [5, 0], [10, 0]])
        self.assertIn("polygon_zero_area", _codes(geo.validate_domain(line)))

    def test_non_finite_geometry_is_rejected(self):
        domain = geo.make_rectangle(10.0, 10.0)
        domain["rectangle"]["width"] = float("inf")
        self.assertIn("geometry_not_finite", _codes(geo.validate_domain(domain)))

    def test_duplicate_boundary_ids_are_rejected(self):
        domain = geo.make_rectangle(10.0, 10.0)
        domain["boundaries"].append({"id": "left", "name": "Repeat"})
        self.assertIn("boundaries_id_duplicate", _codes(geo.validate_domain(domain)))

    def test_measures_and_containment(self):
        rect = geo.make_rectangle(4.0, 3.0)
        self.assertAlmostEqual(geo.domain_measure(rect), 12.0)
        self.assertTrue(geo.point_in_domain((2.0, 1.0), rect))
        self.assertFalse(geo.point_in_domain((5.0, 1.0), rect))

        circle = geo.make_circle(2.0, cx=1.0, cy=1.0)
        self.assertAlmostEqual(geo.domain_measure(circle), 3.141592653589793 * 4.0, places=9)
        self.assertTrue(geo.point_in_domain((1.0, 2.9), circle))
        self.assertFalse(geo.point_in_domain((1.0, 3.5), circle))

        interval = geo.make_interval(10.0)
        self.assertTrue(geo.point_in_domain((0.0,), interval))
        self.assertTrue(geo.point_in_domain((10.0,), interval))
        self.assertFalse(geo.point_in_domain((10.5,), interval))

    def test_points_outside_domain_are_reported_with_index(self):
        rect = geo.make_rectangle(10.0, 10.0)
        issues = geo.validate_points_inside_domain([(1, 1), (99, 1)], rect)
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].code, "entity_outside_domain")
        self.assertIn("[1]", issues[0].path)

    def test_polygon_area_is_orientation_independent(self):
        ccw = [[0, 0], [4, 0], [4, 3], [0, 3]]
        cw = list(reversed(ccw))
        self.assertAlmostEqual(geo.polygon_area(ccw), 12.0)
        self.assertAlmostEqual(geo.polygon_area(cw), 12.0)
        self.assertGreater(geo.polygon_signed_area(ccw), 0)
        self.assertLess(geo.polygon_signed_area(cw), 0)


def _unit_square():
    """Four nodes, four edges, one square cell of area 1."""
    t = topo.new_topology()
    a = topo.add_node(t, 0.0, 0.0)["id"]
    b = topo.add_node(t, 1.0, 0.0)["id"]
    c = topo.add_node(t, 1.0, 1.0)["id"]
    d = topo.add_node(t, 0.0, 1.0)["id"]
    for pair in ((a, b), (b, c), (c, d), (d, a)):
        topo.add_edge(t, *pair)
    return t, [a, b, c, d]


class TopologyConstructionTests(unittest.TestCase):
    def test_build_square_cell(self):
        t, ring = _unit_square()
        cell = topo.make_cell(t, ring, cell_type="epithelial")
        self.assertAlmostEqual(cell["area"], 1.0)
        self.assertAlmostEqual(cell["centroid"][0], 0.5)
        self.assertAlmostEqual(cell["centroid"][1], 0.5)
        self.assertEqual(len(cell["edges"]), 4)
        self.assertTrue(geo.is_valid(topo.validate_topology(t)))

    def test_ids_are_unique_and_stable(self):
        t, _ = _unit_square()
        first = topo.add_node(t, 5.0, 5.0)["id"]
        second = topo.add_node(t, 6.0, 6.0)["id"]
        self.assertNotEqual(first, second)
        with self.assertRaises(ValueError):
            topo.add_node(t, 7.0, 7.0, node_id=first)

    def test_self_loop_and_duplicate_edges_refused(self):
        t, ring = _unit_square()
        with self.assertRaises(ValueError):
            topo.add_edge(t, ring[0], ring[0])
        with self.assertRaises(ValueError):
            topo.add_edge(t, ring[0], ring[1])          # already connected

    def test_open_loop_cannot_become_a_cell(self):
        t = topo.new_topology()
        a = topo.add_node(t, 0.0, 0.0)["id"]
        b = topo.add_node(t, 1.0, 0.0)["id"]
        c = topo.add_node(t, 1.0, 1.0)["id"]
        topo.add_edge(t, a, b)
        topo.add_edge(t, b, c)                          # c-a missing
        with self.assertRaises(ValueError) as raised:
            topo.make_cell(t, [a, b, c])
        self.assertIn("open", str(raised.exception).lower())

    def test_self_intersecting_loop_refused(self):
        t = topo.new_topology()
        a = topo.add_node(t, 0.0, 0.0)["id"]
        b = topo.add_node(t, 10.0, 10.0)["id"]
        c = topo.add_node(t, 10.0, 0.0)["id"]
        d = topo.add_node(t, 0.0, 10.0)["id"]
        for pair in ((a, b), (b, c), (c, d), (d, a)):
            topo.add_edge(t, *pair)
        with self.assertRaises(ValueError) as raised:
            topo.make_cell(t, [a, b, c, d])
        self.assertIn("crosses itself", str(raised.exception))

    def test_collinear_loop_refused(self):
        t = topo.new_topology()
        a = topo.add_node(t, 0.0, 0.0)["id"]
        b = topo.add_node(t, 1.0, 0.0)["id"]
        c = topo.add_node(t, 2.0, 0.0)["id"]
        for pair in ((a, b), (b, c), (c, a)):
            topo.add_edge(t, *pair)
        with self.assertRaises(ValueError):
            topo.make_cell(t, [a, b, c])

    def test_cell_ring_is_stored_counter_clockwise(self):
        t, ring = _unit_square()
        cell = topo.make_cell(t, list(reversed(ring)))
        points = [(topo.node_index(t)[n]["x"], topo.node_index(t)[n]["y"])
                  for n in cell["nodes"]]
        self.assertGreater(geo.polygon_signed_area(points), 0)

    def test_moving_a_node_updates_cell_geometry(self):
        t, ring = _unit_square()
        cell = topo.make_cell(t, ring)
        self.assertAlmostEqual(cell["area"], 1.0)
        topo.move_node(t, ring[1], 2.0, 0.0)
        self.assertAlmostEqual(topo.cell_index(t)[cell["id"]]["area"], 1.5)

    def test_deleting_a_node_cascades(self):
        t, ring = _unit_square()
        topo.make_cell(t, ring)
        removed = topo.delete_node(t, ring[0])
        self.assertEqual(removed["nodes"], 1)
        self.assertEqual(removed["edges"], 2)
        self.assertEqual(removed["cells"], 1)
        self.assertTrue(geo.is_valid(topo.validate_topology(t)))

    def test_neighbors_share_an_edge(self):
        # Two unit squares side by side sharing the middle edge.
        t = topo.new_topology()
        ids = {}
        for name, (x, y) in {
            "a": (0, 0), "b": (1, 0), "c": (1, 1), "d": (0, 1),
            "e": (2, 0), "f": (2, 1),
        }.items():
            ids[name] = topo.add_node(t, x, y)["id"]
        for pair in (("a", "b"), ("b", "c"), ("c", "d"), ("d", "a"),
                     ("b", "e"), ("e", "f"), ("f", "c")):
            topo.add_edge(t, ids[pair[0]], ids[pair[1]])
        left = topo.make_cell(t, [ids["a"], ids["b"], ids["c"], ids["d"]])
        right = topo.make_cell(t, [ids["b"], ids["e"], ids["f"], ids["c"]])
        self.assertIn(right["id"], topo.cell_index(t)[left["id"]]["neighbors"])
        self.assertIn(left["id"], topo.cell_index(t)[right["id"]]["neighbors"])


class TopologyValidationTests(unittest.TestCase):
    def test_dangling_edge_reference(self):
        t, ring = _unit_square()
        t["edges"].append({"id": "bad", "source": ring[0], "target": "nope"})
        self.assertIn("edge_reference_invalid", _codes(topo.validate_topology(t)))

    def test_zero_length_edge(self):
        t = topo.new_topology()
        a = topo.add_node(t, 1.0, 1.0)["id"]
        b = topo.add_node(t, 1.0, 1.0)["id"]        # same position
        t["edges"].append({"id": "e1", "source": a, "target": b})
        codes = _codes(topo.validate_topology(t))
        self.assertIn("edge_zero_length", codes)
        self.assertIn("duplicate_node_position", codes)

    def test_broken_cell_loop_is_named(self):
        t, ring = _unit_square()
        topo.make_cell(t, ring)
        t["edges"] = [e for e in t["edges"]
                      if not {str(e["source"]), str(e["target"])} == {ring[3], ring[0]}]
        issues = topo.validate_topology(t)
        self.assertIn("cell_loop_broken", _codes(issues))

    def test_duplicate_ids_detected(self):
        t, ring = _unit_square()
        t["nodes"].append(dict(t["nodes"][0]))
        self.assertIn("node_id_duplicate", _codes(topo.validate_topology(t)))

    def test_non_finite_coordinates_detected(self):
        t, _ = _unit_square()
        t["nodes"][0]["x"] = float("nan")
        self.assertIn("node_coordinates_invalid", _codes(topo.validate_topology(t)))

    def test_nodes_outside_domain_detected(self):
        t, _ = _unit_square()
        tiny = geo.make_rectangle(0.5, 0.5)
        self.assertIn("entity_outside_domain", _codes(topo.validate_topology(t, tiny)))


class LoopDetectionTests(unittest.TestCase):
    def test_finds_the_single_square(self):
        t, ring = _unit_square()
        cycles = topo.find_cycles(t)
        self.assertEqual(len(cycles), 1)
        self.assertEqual(set(cycles[0]), set(ring))

    def test_finds_both_faces_of_two_squares(self):
        t = topo.new_topology()
        ids = {}
        for name, (x, y) in {
            "a": (0, 0), "b": (1, 0), "c": (1, 1), "d": (0, 1),
            "e": (2, 0), "f": (2, 1),
        }.items():
            ids[name] = topo.add_node(t, x, y)["id"]
        for pair in (("a", "b"), ("b", "c"), ("c", "d"), ("d", "a"),
                     ("b", "e"), ("e", "f"), ("f", "c")):
            topo.add_edge(t, ids[pair[0]], ids[pair[1]])
        cycles = topo.find_cycles(t)
        # The two unit squares are chordless; the outer 6-ring is not (b-c chords it).
        self.assertEqual(len(cycles), 2)
        for cycle in cycles:
            self.assertEqual(len(cycle), 4)

    def test_no_cycle_in_a_tree(self):
        t = topo.new_topology()
        a = topo.add_node(t, 0.0, 0.0)["id"]
        b = topo.add_node(t, 1.0, 0.0)["id"]
        c = topo.add_node(t, 2.0, 0.0)["id"]
        topo.add_edge(t, a, b)
        topo.add_edge(t, b, c)
        self.assertEqual(topo.find_cycles(t), [])


class InterchangeTests(unittest.TestCase):
    def test_json_round_trip_preserves_entities(self):
        t, ring = _unit_square()
        cell = topo.make_cell(t, ring, cell_type="tumour", region="core")
        restored = topo.from_json_dict(topo.to_json_dict(t))
        self.assertEqual(len(restored["nodes"]), 4)
        self.assertEqual(len(restored["edges"]), 4)
        self.assertEqual(len(restored["cells"]), 1)
        self.assertEqual(restored["cells"][0]["type"], "tumour")
        self.assertEqual(restored["cells"][0]["region"], "core")
        self.assertAlmostEqual(restored["cells"][0]["area"], cell["area"])
        self.assertTrue(geo.is_valid(topo.validate_topology(restored)))

    def test_csv_round_trip(self):
        t, ring = _unit_square()
        topo.make_cell(t, ring)
        nodes = topo.nodes_from_csv(topo.nodes_to_csv(t))
        edges = topo.edges_from_csv(topo.edges_to_csv(t))
        self.assertEqual(len(nodes), 4)
        self.assertEqual(len(edges), 4)
        rebuilt = topo.from_json_dict({"nodes": nodes, "edges": edges, "cells": []})
        self.assertTrue(geo.is_valid(topo.validate_topology(rebuilt)))

    def test_cells_csv_has_a_row_per_cell(self):
        t, ring = _unit_square()
        topo.make_cell(t, ring)
        lines = [line for line in topo.cells_to_csv(t).strip().splitlines() if line]
        self.assertEqual(len(lines), 2)                 # header + one cell
        self.assertIn("centroid_x", lines[0])

    def test_bad_csv_names_the_row(self):
        with self.assertRaises(ValueError) as raised:
            topo.nodes_from_csv("id,x,y\nn1,notanumber,0\n")
        self.assertIn("Row 2", str(raised.exception))


class ScaleTests(unittest.TestCase):
    def test_validation_handles_the_target_size(self):
        """The editor targets ~1000 nodes and ~800 edges; validation must stay linear."""
        t = topo.new_topology()
        ids = [topo.add_node(t, float(i % 40), float(i // 40))["id"] for i in range(1000)]
        for i in range(800):
            topo.add_edge(t, ids[i], ids[i + 1])
        issues = topo.validate_topology(t)
        self.assertEqual([i for i in issues if i.severity == "error"], [])
        stats = topo.topology_stats(t)
        self.assertEqual(stats["node_count"], 1000)
        self.assertEqual(stats["edge_count"], 800)


if __name__ == "__main__":
    unittest.main()
