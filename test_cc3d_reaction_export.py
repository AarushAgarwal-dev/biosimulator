"""CompuCell3D must carry the stage-4 REACTION terms, and must do it safely.

The generated CC3DML carries a field's name, GlobalDiffusionConstant and
GlobalDecayConstant and nothing else -- verified by search, 'reaction' and 'advection'
appeared nowhere in approach_cc3d.py's code. So a researcher who wrote a logistic reaction
r*u*(1 - u/K) on the PDE-model stage got pure diffusion with linear decay, and the run said
nothing about it. DiffusionSolverFE has no field for a general rate law, so the reaction is
emitted as a Python steppable instead.

The safety property matters as much as the feature: a reaction string is USER INPUT that
ends up inside generated Python. It is parsed with pde_model.parse_safe first -- which
permits only the declared field names, the model's parameters and a fixed set of
mathematical functions -- and only the re-printed sympy expression is emitted. The raw text
is never interpolated.
"""
import unittest

import approach_cc3d as cc3d

CELL_TYPES = [{"name": "CellA", "type_id": 1, "target_volume": 25,
               "lambda_volume": 2.0}]


def _project(reaction, parameters=None, extra_fields=()):
    fields = [{"name": "u", "dt": 0.05, "reaction": reaction}]
    fields.extend(extra_fields)
    return {
        "model": {"parameters": dict(parameters or {}), "fields": fields},
        "approaches": {"cc3d": {
            "cell_types": list(CELL_TYPES),
            "fields": [{"name": f["name"], "diffusion": 0.1, "decay": 0.0}
                       for f in fields],
        }},
    }


def _files(project):
    return cc3d.CC3D_ADAPTER.export_configuration(project)["files"]


class ReactionIsCarriedTests(unittest.TestCase):
    def test_a_reaction_is_exported_as_runnable_python(self):
        files = _files(_project("r*u*(1 - u/K)", {"r": 0.4, "K": 2.0}))
        code = files.get("Simulation/reaction_steppables.py")
        self.assertTrue(code, "the reaction term was dropped from the export")
        compile(code, "<reaction>", "exec")          # must be valid Python
        self.assertIn("self.field.u", code, "the field is never written")
        self.assertIn("numpy.asarray", code, "the field is never read as an array")

    def test_parameters_are_substituted_not_left_symbolic(self):
        """The generated file runs inside CC3D, where r and K do not exist."""
        code = _files(_project("r*u*(1 - u/K)", {"r": 0.4, "K": 2.0}))[
            "Simulation/reaction_steppables.py"]
        self.assertIn("0.4", code, "the rate constant was not substituted")
        self.assertNotIn("*r*", code, "a symbolic parameter survived into the file")

    def test_the_raw_expression_is_never_interpolated(self):
        """It must be re-printed from the parsed expression, not pasted."""
        raw = "r*u*(1 - u/K)"
        code = _files(_project(raw, {"r": 0.4, "K": 2.0}))[
            "Simulation/reaction_steppables.py"]
        self.assertNotIn(raw, code,
                         "the user's raw string was pasted into generated Python")

    def test_an_injected_expression_cannot_reach_the_file(self):
        """A reaction string is user input that ends up inside generated Python."""
        for evil in ("__import__('os').system('echo pwned')",
                     "eval('1+1')",
                     "open('/etc/passwd').read()"):
            with self.subTest(expression=evil):
                files = _files(_project(evil))
                code = files.get("Simulation/reaction_steppables.py", "")
                for marker in ("__import__", "system", "eval(", "open("):
                    self.assertNotIn(
                        marker, code,
                        f"{marker!r} from an injected reaction reached the generated file")

    def test_an_unparseable_reaction_is_named_not_guessed(self):
        files = _files(_project("this is not an expression"))
        code = files.get("Simulation/reaction_steppables.py", "")
        if code:
            self.assertIn("SKIPPED", code,
                          "an unparseable reaction was silently applied anyway")

    def test_no_file_when_no_field_has_a_reaction(self):
        files = _files(_project("0"))
        self.assertNotIn("Simulation/reaction_steppables.py", files,
                         "a pointless empty steppable was shipped")
        self.assertIn("pure diffusion", files["README.txt"],
                      "the README does not say the run is diffusion only")

    def test_every_field_is_readable_so_cross_terms_work(self):
        """A reaction in u that references v must not raise NameError at run time."""
        project = _project("r*u - a*v", {"r": 0.3, "a": 0.2},
                           extra_fields=({"name": "v", "reaction": "0"},))
        code = _files(project)["Simulation/reaction_steppables.py"]
        self.assertIn("self.field.v", code,
                      "v is never read, so a cross-term would raise NameError in CC3D")

    def test_the_readme_says_how_to_register_it(self):
        """CC3D does not auto-register it, so an unexplained file would do nothing."""
        readme = _files(_project("r*u", {"r": 0.4}))["README.txt"]
        self.assertIn("register_steppable", readme)
        self.assertIn("ReactionSteppable", readme)

    def test_the_operator_splitting_caveat_is_stated(self):
        """Transport and reaction advance separately; that is an approximation."""
        code = _files(_project("r*u", {"r": 0.4}))["Simulation/reaction_steppables.py"]
        self.assertIn("splitting", code.lower(),
                      "the file does not disclose that this is operator splitting")


class RemoteRunIsHonestTests(unittest.TestCase):
    """The notes must match what a remote run actually does -- in either direction.

    This test used to assert the OPPOSITE: that the notes say a remote run does not apply
    the reaction. That was true and worth guarding until the job runner registered the
    reaction steppable, and then the guard became the thing keeping a false statement in
    front of researchers. Understating what the product does is a smaller sin than
    overstating it, but it is still wrong, and a test that pins an understatement is not a
    safety net -- it is a lock on stale information.

    What replaced it is not a weaker claim, it is a checked one: verified on real AWS by
    run_9b2b37cb624f, which logged "registered reaction steppable for: u" and completed
    all twelve steps. The applier raises on any eval or write failure, so completing every
    step is what establishes the rate law reached the field inside real CompuCell3D.
    """

    def test_the_notes_say_the_reaction_is_applied_on_a_remote_run(self):
        notes = str(cc3d.CC3D_ADAPTER.get_capabilities().notes or "")
        self.assertIn("REMOTE", notes.upper(),
                      "the notes do not distinguish a remote run from a local one")
        self.assertIn("reaction_steppables.py", notes,
                      "the notes do not name the file that carries the reaction locally")
        stale = ("still pure diffusion", "registers only the measurement",
                 "run it locally if you need the reaction")
        for phrase in stale:
            self.assertNotIn(phrase, notes,
                             f"the notes still carry the obsolete caveat {phrase!r}")

    def test_the_notes_disclose_operator_splitting_rather_than_implying_exactness(self):
        """Applying transport and reaction in separate sub-steps is an approximation."""
        notes = str(cc3d.CC3D_ADAPTER.get_capabilities().notes or "")
        self.assertIn("splitting", notes.lower(),
                      "the notes do not disclose that this is operator splitting")

    def test_the_notes_still_name_what_no_backend_consumes(self):
        """Advection and the stage-5 boundary conditions are genuinely unconsumed."""
        notes = str(cc3d.CC3D_ADAPTER.get_capabilities().notes or "")
        self.assertIn("IGNORES", notes, "the notes no longer state what is ignored")
        ignored = notes.split("IGNORES", 1)[1].lower()
        self.assertIn("advection", ignored, "advection is no longer disclosed as ignored")
        self.assertIn("boundary", ignored,
                      "the stage-5 boundary conditions are no longer disclosed as ignored")


if __name__ == "__main__":
    unittest.main()
