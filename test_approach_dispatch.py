"""APPROACH DISPATCH: selecting one engine must never run a different one.

The three approaches are meant to be independently selectable with no silent fallback
between them, and that requirement is guarded at the ADAPTER layer already
(test_cc3d_remote.NoFallbackTests asserts CompuCell3D refuses rather than delegating,
and that its refusal never names ABM or MPC as a stand-in).

This file guards the layer ABOVE it, which was unguarded. run_manager resolves an
engine with `get_approach(record.approach_id)`, and nothing asserted that mapping. If
that lookup ever became a defaulting one -- `_REGISTRY.get(key, ABM_ADAPTER)`, which is
an easy and innocent-looking refactor -- then choosing CompuCell3D would run the
in-tree ABM engine, the adapter-level tests would still pass because the ABM adapter is
perfectly available, and the substitution would be invisible. ABM is also a Cellular
Potts model, so the numbers would look plausible while coming from different contact
energies, a different lattice and a different engine version.

The distinction that matters: an unknown approach must RAISE, never resolve.
"""
import unittest
from unittest.mock import patch

import approach_base
import run_manager


class ApproachRegistryTests(unittest.TestCase):
    def setUp(self):
        approach_base.ensure_approaches_loaded()

    def test_each_id_resolves_to_the_engine_it_names(self):
        for approach_id in ("abm", "cc3d", "mpc"):
            adapter = approach_base.get_approach(approach_id)
            capabilities = adapter.get_capabilities()
            self.assertEqual(
                getattr(capabilities, "approach_id", None), approach_id,
                f"get_approach({approach_id!r}) returned an adapter that identifies as "
                f"{getattr(capabilities, 'approach_id', None)!r} -- selecting one "
                f"engine would run another")

    def test_the_three_adapters_are_distinct(self):
        adapters = {name: approach_base.get_approach(name)
                    for name in ("abm", "cc3d", "mpc")}
        identities = {name: id(adapter) for name, adapter in adapters.items()}
        self.assertEqual(
            len(set(identities.values())), 3,
            f"two approach ids share one adapter object, so one silently substitutes "
            f"for another: {identities}")

    def test_an_unknown_approach_raises_and_never_defaults(self):
        for bogus in ("", "   ", "mcp", "compucell", "abm2", "unknown"):
            with self.subTest(approach=bogus):
                with self.assertRaises(KeyError, msg=(
                        f"get_approach({bogus!r}) resolved to an adapter instead of "
                        f"raising. A defaulting lookup here silently substitutes "
                        f"engines.")):
                    approach_base.get_approach(bogus)

    def test_mcp_is_not_an_alias_for_mpc(self):
        """The approach is MPC (Model Predictive Control). 'mcp' is a typo, not an id.

        Accepting it as an alias would let a typo pick a control method for someone who
        meant something else, so the refusal is deliberate.
        """
        with self.assertRaises(KeyError):
            approach_base.get_approach("mcp")

    def test_the_refusal_names_what_is_available(self):
        try:
            approach_base.get_approach("definitely-not-an-approach")
        except KeyError as error:
            message = str(error)
        else:
            self.fail("an unknown approach did not raise")
        for approach_id in ("abm", "cc3d", "mpc"):
            self.assertIn(
                approach_id, message,
                "the refusal must list the approaches that ARE available, so a caller "
                "can correct the id instead of guessing")


class RunManagerDispatchTests(unittest.TestCase):
    """The run manager must dispatch on the run's OWN approach id, not a default."""

    def test_dispatch_uses_the_runs_own_approach_id(self):
        seen = []
        real = approach_base.get_approach

        def recording_get_approach(approach_id):
            seen.append(approach_id)
            return real(approach_id)

        manager = run_manager.RunManager()
        project = {"selected_approach": "mpc"}
        with patch.object(run_manager, "get_approach", recording_get_approach):
            try:
                manager.create_run(project, "mpc")
            except Exception:
                # create_run may reject this minimal project; what matters is that when
                # it DID look an engine up, it looked up the id the caller asked for.
                pass
        for approach_id in seen:
            self.assertEqual(
                approach_id, "mpc",
                f"the run manager resolved {approach_id!r} for a run created as 'mpc'")

    def test_an_unknown_approach_does_not_silently_become_a_real_one(self):
        manager = run_manager.RunManager()
        project = {"selected_approach": "not-an-approach"}
        try:
            run_id = manager.create_run(project, "not-an-approach")
        except Exception:
            return  # refused outright, which is correct
        # If creation was permitted, starting it must not run some other engine.
        manager.start(run_id)
        status = manager.status(run_id)
        self.assertNotEqual(
            status.get("state"), "completed",
            "a run naming an approach that does not exist completed successfully, "
            "which means some other engine ran in its place")


class ApproachDisclosureTests(unittest.TestCase):
    """Each approach must say which prepared stages it uses and which it ignores.

    The workflow's premise -- "prepare a domain, topology, mesh and model ONCE, then
    choose one approach to compile and run" -- invites the reader to assume every
    approach runs the model they prepared. None of them fully does, and until now
    nothing said so:

      - MPC simulates its OWN first-order plant, dx/dt = -x/tau + gain*u. The domain,
        mesh, stage-4 model and boundary conditions are unused, and 'controlled_input' /
        'measured_output' are that plant's ports, not species in the network. It is a
        correct controller demonstration with no connection to the biology.
      - CompuCell3D carries each field's diffusion constant and decay only. Verified by
        search: 'reaction' and 'advection' appear nowhere in approach_cc3d.py's code, and
        the only matches for 'boundary' are comments. A logistic reaction written in
        stage 4 is simply not simulated.
      - ABM's Potts lattice is its own grid, independent of the stage-3 mesh.

    A researcher who prepares a model and then picks an approach deserves to know which
    of their work is actually being run. This does not fix the gap -- bridging the model
    into each engine is a design change -- but an unstated gap is the failure mode this
    codebase has been shedding, so the statement is the minimum.
    """

    def setUp(self):
        approach_base.ensure_approaches_loaded()

    def test_every_approach_states_what_it_ignores(self):
        for approach_id in ("abm", "cc3d", "mpc"):
            notes = str(getattr(approach_base.get_approach(approach_id)
                                .get_capabilities(), "notes", "") or "")
            self.assertTrue(
                notes.strip(),
                f"{approach_id} offers no notes at all, so the approach picker can say "
                f"nothing about what it consumes")
            self.assertIn(
                "IGNORES", notes,
                f"{approach_id}'s notes do not state which prepared stages it ignores. "
                f"A researcher picking it would assume their stage-4 model is run.")

    def test_mpc_explains_both_of_its_plants(self):
        """MPC can now control the prepared model, so the notes must distinguish the two.

        This test previously asserted that MPC ADMITS its plant is not the prepared model,
        which was the right assertion while that was the only option. It can now use the
        stage-4 reaction network as its plant, so the honest requirement changed: the notes
        must say which mode does what, because a researcher reading "MPC" needs to know
        whether the result concerns their biology or a first-order demonstration.
        """
        notes = str(approach_base.get_approach("mpc").get_capabilities().notes or "")
        lowered = notes.lower()
        self.assertIn("stage-4", lowered,
                      "MPC does not mention the stage-4 model at all")
        self.assertIn("reaction_network", notes,
                      "the notes do not name the plant kind that controls the prepared "
                      "model, so a researcher cannot discover it")
        self.assertIn("first_order", notes,
                      "the notes do not say what the default plant is, so a researcher "
                      "cannot tell a biological result from a controller demonstration")

    def test_cc3d_admits_reaction_terms_are_not_simulated(self):
        notes = str(approach_base.get_approach("cc3d").get_capabilities().notes or "")
        self.assertIn(
            "reaction", notes,
            "CompuCell3D drops a field's reaction expression without saying so")


if __name__ == "__main__":
    unittest.main()
