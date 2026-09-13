"""MPC MUST BE ABLE TO CONTROL THE MODEL THE RESEARCHER PREPARED.

The workflow promises "prepare a domain, topology, mesh and model ONCE, then choose one
approach". MPC ignored all of it: it simulated its own dx/dt = -x/tau + gain*u with
tau = 5 and gain = 1, while `controlled_input` and `measured_output` were the strings "u"
and "y" -- ports of that toy system, not species in anyone's network. A fresh review of
the product as a computational biologist called this out: a correct, well-validated
controller demonstration with no connection to the biology, and nothing saying so.

With plant.kind = "reaction_network" the plant becomes the stage-4 model, integrated as a
well-mixed system, with the control dosed into a named species. That turns MPC into a
biological question: what infusion schedule holds this species at this level?
"""
import unittest

import approach_mpc as mpc
from approach_base import RunContext


def _model():
    """A substrate consumed to make a product, the product decaying -- stage-4 shaped."""
    return {
        "parameters": {"kcat": 0.8, "Km": 1.0, "dP": 0.25},
        "fields": [
            {"name": "S", "reaction": "-kcat*S/(Km + S)", "initial": "0.0"},
            {"name": "P", "reaction": "kcat*S/(Km + S) - dP*P", "initial": "0.0"},
        ],
    }


class ReactionNetworkPlantTests(unittest.TestCase):
    def test_the_control_drives_the_named_species(self):
        plant = mpc.ReactionNetworkPlant(_model(), "S", "P")
        for _ in range(200):
            plant.step(1.0, 0.05)
        dosed = plant.output()

        idle = mpc.ReactionNetworkPlant(_model(), "S", "P")
        for _ in range(200):
            idle.step(0.0, 0.05)

        self.assertGreater(dosed, 0.1,
                           "dosing the input species did not move the output")
        self.assertLessEqual(idle.output(), 1e-9,
                             f"the output moved with NO control input ({idle.output():.3e}), "
                             f"so u is not what drives this plant")

    def test_more_dose_gives_more_output(self):
        outputs = []
        for u in (0.2, 0.5, 1.0):
            plant = mpc.ReactionNetworkPlant(_model(), "S", "P")
            for _ in range(200):
                plant.step(u, 0.05)
            outputs.append(plant.output())
        self.assertEqual(outputs, sorted(outputs),
                         f"the response is not monotonic in the dose: {outputs}")

    def test_a_species_the_model_lacks_is_refused(self):
        """The old config named "u" and "y", which are not species anywhere."""
        for bad in ("u", "y", "ERK", ""):
            with self.subTest(name=bad):
                with self.assertRaises(ValueError):
                    mpc.ReactionNetworkPlant(_model(), bad, "P")
                with self.assertRaises(ValueError):
                    mpc.ReactionNetworkPlant(_model(), "S", bad)

    def test_the_refusal_names_the_available_species(self):
        try:
            mpc.ReactionNetworkPlant(_model(), "u", "P")
        except ValueError as error:
            self.assertIn("S", str(error))
            self.assertIn("P", str(error))
        else:
            self.fail("an unknown input species was accepted")

    def test_a_model_with_no_fields_is_refused(self):
        with self.assertRaises(ValueError):
            mpc.ReactionNetworkPlant({"fields": []}, "S", "P")

    def test_concentrations_never_go_negative(self):
        """A controller that overshoots must not drive the plant somewhere unphysical."""
        plant = mpc.ReactionNetworkPlant(_model(), "S", "P")
        for _ in range(300):
            plant.step(-5.0, 0.05)          # a deliberately illegal negative dose
        for name, value in zip(plant.names, plant.state):
            self.assertGreaterEqual(value, 0.0, f"{name} went negative: {value}")


class MpcControlsThePreparedModel(unittest.TestCase):
    def _project(self):
        return {
            "model": _model(),
            "selected_approach": "mpc",
            "approaches": {"mpc": {
                "plant": {"kind": "reaction_network"},
                "model": {"kind": "first_order", "tau": 4.0, "gain": 1.0},
                "controlled_input": "S",
                "measured_output": "P",
                "target": 0.5,
                "duration": 40.0,
                "control_interval": 0.5,
                "input_min": 0.0,
                "input_max": 3.0,
            }},
        }

    def test_a_run_against_the_prepared_model_tracks_its_target(self):
        adapter = mpc.MPCAdapter()
        project = self._project()
        errors = [i for i in adapter.validate(project) if i.severity == "error"]
        self.assertEqual(errors, [], f"the project did not validate: {errors}")

        compiled = adapter.compile(project)
        self.assertTrue(compiled.get("model", {}).get("fields"),
                        "the prepared model was not carried into the compiled plan, so "
                        "run() cannot build a plant from it")

        result = adapter.run(compiled, RunContext(run_id="mpc_network"))
        summary = result.get("summary") or {}
        self.assertIn("final_abs_error", summary)
        self.assertTrue(summary.get("input_bounds_respected", True),
                        "the controller violated its own input bounds")
        self.assertLess(float(summary["final_mean_abs_error"]),
                        float(summary["initial_mean_abs_error"]),
                        f"tracking did not improve on the researcher's own model: {summary}")


if __name__ == "__main__":
    unittest.main()
