"""The REMOTE CompuCell3D run must apply the stage-4 reactions, and safely.

DiffusionSolverFE handles diffusion and linear decay and has no field for a general rate
law, so a reaction written on the PDE-model stage never reached a dispatched job: it
produced pure diffusion and reported success. The runner now registers a steppable that
applies the expressions the server put in cc3d_config.json.

CompuCell3D does not exist on a development host, so the applier is exercised against a
stub that mimics the parts it touches -- a `field` namespace of numpy views. That is the
same technique the spot-interruption suite uses, and it verifies the arithmetic, the
namespace restriction and the failure behaviour. What it cannot verify is CC3D's own field
semantics; only a container run does that.
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "deploy", "cc3d"))
import cc3d_job_runner as runner


class _StubBase:
    """Stands in for SteppableBasePy: records frequency, exposes a field namespace."""

    def __init__(self, frequency=1):
        self.frequency = frequency
        self.field = self


def _make(reactions, arrays, dt=1.0):
    cls = runner._make_reaction_applier(_StubBase)
    applier = cls(reactions, list(arrays), dt)
    for name, values in arrays.items():
        setattr(applier.field, name, np.array(values, dtype=float))
    return applier


class RemoteReactionApplierTests(unittest.TestCase):
    def test_a_reaction_changes_the_field(self):
        u = np.full((2, 2, 1), 0.5)
        applier = _make({"u": "0.4*u*(1 - 0.5*u)"}, {"u": u}, dt=0.1)
        before = np.array(applier.field.u, dtype=float).copy()
        applier.step(1)
        after = np.array(applier.field.u, dtype=float)
        self.assertFalse(np.allclose(before, after),
                         "the reaction did not change the field")
        # Logistic growth from 0.5 toward the carrying capacity: du = dt*0.4*0.5*(1-0.25)
        self.assertTrue(np.all(after > before), "logistic growth should increase u here")
        self.assertAlmostEqual(float(after[0, 0, 0] - before[0, 0, 0]),
                               0.1 * 0.4 * 0.5 * (1 - 0.5 * 0.5), places=9)

    def test_it_carries_no_reaction_quietly(self):
        applier = _make({}, {"u": np.ones((2, 2, 1))})
        applier.step(1)
        self.assertEqual(applier.failures, [])

    def test_a_cross_term_reads_both_fields(self):
        arrays = {"u": np.full((2, 2, 1), 1.0), "v": np.full((2, 2, 1), 2.0)}
        applier = _make({"u": "0.5*u - 0.25*v"}, arrays, dt=1.0)
        applier.step(1)
        got = float(np.array(applier.field.u, dtype=float)[0, 0, 0])
        self.assertAlmostEqual(got, 1.0 + (0.5 * 1.0 - 0.25 * 2.0), places=9)

    def test_builtins_are_unreachable_from_an_expression(self):
        """A second barrier: even a string reaching here cannot touch the interpreter."""
        applier = _make({"u": "__import__('os').system('echo pwned')"},
                        {"u": np.ones((2, 2, 1))})
        with self.assertRaises(Exception):
            applier.step(1)
        self.assertTrue(any("eval u" in f for f in applier.failures),
                        f"the failure was not recorded: {applier.failures}")

    def test_an_unevaluable_reaction_raises_rather_than_reporting_success(self):
        """A run that models nothing must not finish looking successful."""
        applier = _make({"u": "u + missing_symbol"}, {"u": np.ones((2, 2, 1))})
        with self.assertRaises(Exception):
            applier.step(1)

    def test_a_missing_field_is_recorded_and_the_step_is_skipped(self):
        cls = runner._make_reaction_applier(_StubBase)
        applier = cls({"u": "0.4*u"}, ["u"], 1.0)   # no array set on the stub field
        applier.step(1)
        self.assertTrue(applier.failures, "a missing field was not recorded")
        self.assertEqual(applier.applied, 0)

    def test_the_step_count_is_tracked(self):
        applier = _make({"u": "0.1*u"}, {"u": np.ones((2, 2, 1))})
        for _ in range(3):
            applier.step(1)
        self.assertEqual(applier.applied, 3)


class ReactionStepTests(unittest.TestCase):
    def test_dt_comes_from_the_model_when_stated(self):
        self.assertEqual(runner._reaction_dt({"fields": [{"name": "u", "dt": 0.05}]}), 0.05)

    def test_dt_falls_back_to_one_step_per_mcs(self):
        self.assertEqual(runner._reaction_dt({"fields": [{"name": "u"}]}), 1.0)
        self.assertEqual(runner._reaction_dt({}), 1.0)

    def test_a_nonsense_dt_does_not_crash_the_run(self):
        self.assertEqual(runner._reaction_dt({"fields": [{"name": "u", "dt": "soon"}]}), 1.0)


if __name__ == "__main__":
    unittest.main()
