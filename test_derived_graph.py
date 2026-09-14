"""THE DIAGRAM MUST BE THE EQUATIONS.

When a blueprint carries explicit ``odes``, ODEModel integrates those rate laws and never
reads the ``edges`` array -- so the arrows on screen were whatever the preset author drew,
and they disagreed with the mathematics. Measured before this was fixed:

    preset         ode lines   edges drawn
    zhabotinsky           13             2
    berridge               2             2

Thirteen coupled equations displayed as two arrows. Berridge drew Z -> Y and Y -> Z
activation for a model whose real structure is a Z-gated pump plus calcium-induced calcium
release, and omitted the pump entirely.

The dependency structure is already in the equations, so it is derived rather than
asserted: species Y regulates X exactly when Y appears in dX/dt, and the sign is the sign
of d(dX/dt)/dY. Two refinements earned their place by failing first:

  - The derivative is taken of the PRODUCTION terms only. Judging a self-edge by the net
    derivative reported "CAMKII inhibits itself" for the bistable preset -- the opposite of
    its published mechanism -- because at the starting value the -0.5*C decay outweighs the
    autocatalytic Hill term, and that same decay pushed STIM -> CAMKII below the
    significance cutoff so it vanished.
  - Weak influences are folded away per target. A literal Jacobian derived 122 edges for
    zhabotinsky, because its shared flux couples all 13 states: true, and less readable
    than the two arrows it replaced.
"""
import unittest

import agent
import preset_loader
import simulation_engine as se

BLUEPRINTS = preset_loader.load_blueprints(text_compiler=agent.rule_based_parse)


def _edges(name):
    return se.derive_edges_from_odes(BLUEPRINTS[name]) or []


def _pairs(name):
    return {(e["source"], e["target"], e["type"]) for e in _edges(name)}


class DerivedGraphTests(unittest.TestCase):
    def test_a_generic_hill_model_derives_nothing(self):
        """Turing has no explicit odes, so its drawn edges ARE the compiled topology."""
        self.assertIsNone(se.derive_edges_from_odes(BLUEPRINTS["turing"]),
                          "a model without explicit rate laws must keep its own edges")

    def test_every_explicit_equation_preset_derives_a_graph(self):
        for name, bp in BLUEPRINTS.items():
            if not bp.get("odes"):
                continue
            with self.subTest(preset=name):
                self.assertTrue(_edges(name),
                                f"{name} has {len(bp['odes'])} equations and derived no "
                                f"graph, so the diagram falls back to decoration")

    def test_bistable_shows_its_autoactivation(self):
        """The motif the preset exists to demonstrate must appear, and as ACTIVATION.

        An earlier attempt reported CAMKII -> CAMKII as inhibition, because the net
        derivative at the starting value is dominated by decay.
        """
        pairs = _pairs("bistable")
        self.assertIn(("CAMKII", "CAMKII", "activation"), pairs,
                      f"cooperative autoactivation is missing or mis-signed: {pairs}")
        self.assertIn(("STIM", "CAMKII", "activation"), pairs,
                      f"the stimulus edge was dropped: {pairs}")

    def test_berridge_surfaces_the_pump_the_drawn_graph_omitted(self):
        """Z's own saturable removal is a mechanism, so it must show as self-inhibition."""
        pairs = _pairs("berridge")
        self.assertIn(("Z", "Z", "inhibition"), pairs,
                      f"the Z-gated pump is not represented: {pairs}")

    def test_zhabotinsky_is_richer_than_two_arrows_but_still_readable(self):
        """13 coupled equations were shown as 2 arrows; a literal Jacobian gave 122."""
        edges = _edges("zhabotinsky")
        drawn = len(BLUEPRINTS["zhabotinsky"].get("edges") or [])
        self.assertGreater(len(edges), drawn,
                           "the derived graph is no richer than the hand-drawn one")
        self.assertLess(len(edges), 100,
                        f"{len(edges)} edges is a hairball, not a diagram -- the "
                        f"significance filter is not working")

    def test_no_edge_names_a_species_the_model_lacks(self):
        """A dangling edge crashes Cytoscape and takes the whole compile with it."""
        for name, bp in BLUEPRINTS.items():
            if not bp.get("odes"):
                continue
            ids = {n["id"] for n in bp.get("nodes") or []}
            for edge in _edges(name):
                with self.subTest(preset=name, edge=(edge["source"], edge["target"])):
                    self.assertIn(edge["source"], ids)
                    self.assertIn(edge["target"], ids)

    def test_every_derived_edge_is_marked_derived(self):
        """The UI distinguishes derived from hand-drawn, so the flag must survive."""
        for edge in _edges("egfr"):
            self.assertTrue(edge.get("derived"),
                            "a derived edge is not flagged, so the UI cannot label it")

    def test_plain_turnover_is_not_drawn_as_a_self_arrow(self):
        """Every species has a decay term; drawing all of them would bury the real ones."""
        blueprint = {
            "type": "ODE",
            "nodes": [{"id": "A", "initial_value": 1.0}, {"id": "B", "initial_value": 0.0}],
            "parameters": {"k": 0.5, "d": 0.1},
            "odes": {"A": "-d*A", "B": "k*A - d*B"},
        }
        pairs = {(e["source"], e["target"]) for e in
                 (se.derive_edges_from_odes(blueprint) or [])}
        self.assertNotIn(("A", "A"), pairs, "first-order decay was drawn as an arrow")
        self.assertNotIn(("B", "B"), pairs, "first-order decay was drawn as an arrow")
        self.assertIn(("A", "B"), pairs, "the real interaction A -> B is missing")


if __name__ == "__main__":
    unittest.main()
