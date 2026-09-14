"""The deployment manifests must be valid, because nothing else checks them.

Written after a real incident. The worker image could not be rebuilt AT ALL for hours and
nothing said so: CodeBuild rejected buildspec.yml with

    YAML_FILE_ERROR: Expected Commands[3] to be of string type: found subkeys instead
    at line 115, value of the key tag on line 114 might be empty

because an unquoted YAML list item containing a colon followed by a space parses as a
mapping, not a string. Every one of the project's 900 tests stayed green throughout, since
none of them reads a deployment manifest -- the break was only discoverable by spending a
build, and it was found by accident while doing something else.

These tests cost milliseconds and close that gap. They do not verify that the build
SUCCEEDS -- that needs Docker and a CodeBuild runner -- only that the manifests are
structurally valid, which is the failure that actually happened.
"""
import os
import unittest

DEPLOY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deploy", "cc3d")
BUILDSPEC = os.path.join(DEPLOY, "buildspec.yml")


class BuildspecTests(unittest.TestCase):
    def setUp(self):
        try:
            import yaml                                  # noqa: F401
        except ImportError:                               # pragma: no cover
            self.skipTest("pyyaml is not installed")
        self.assertTrue(os.path.exists(BUILDSPEC), f"{BUILDSPEC} is missing")

    def _load(self):
        import yaml
        with open(BUILDSPEC, encoding="utf-8") as handle:
            return yaml.safe_load(handle)

    def test_the_buildspec_parses_as_yaml(self):
        """The exact failure: CodeBuild could not parse the file at all."""
        document = self._load()
        self.assertIsInstance(document, dict, "buildspec.yml is not a YAML mapping")
        self.assertIn("phases", document, "buildspec.yml declares no phases")

    def test_every_command_is_a_string(self):
        """A command that parsed as a mapping is what CodeBuild rejected."""
        document = self._load()
        for phase, body in (document.get("phases") or {}).items():
            commands = (body or {}).get("commands") or []
            for index, command in enumerate(commands):
                with self.subTest(phase=phase, index=index):
                    self.assertIsInstance(
                        command, str,
                        f"phases.{phase}.commands[{index}] parsed as "
                        f"{type(command).__name__}, not a string: {command!r}. An unquoted "
                        f"YAML list item containing ': ' becomes a mapping -- quote the "
                        f"whole command.")

    def test_no_unquoted_command_contains_a_colon_space(self):
        """Catches the hazard by SHAPE, before it reaches a build.

        The parse test above catches this particular line, but a colon-space can also
        produce a value that still parses as a string while meaning something other than
        the author intended. Flagging the shape is cheaper than reasoning about which case
        applies.
        """
        offenders = []
        with open(BUILDSPEC, encoding="utf-8") as handle:
            for number, line in enumerate(handle, start=1):
                stripped = line.lstrip()
                if not stripped.startswith("- "):
                    continue
                payload = stripped[2:].strip()
                if payload[:1] in ("'", '"'):        # quoted as a whole: safe
                    continue
                if ": " in payload:
                    offenders.append(f"L{number}: {payload}")
        self.assertEqual(
            offenders, [],
            "unquoted buildspec command(s) contain ': ', which YAML reads as a mapping "
            "key. Single-quote the whole command:\n  " + "\n  ".join(offenders))

    def test_the_phases_the_image_build_depends_on_are_present(self):
        document = self._load()
        phases = document.get("phases") or {}
        for phase in ("pre_build", "build", "post_build"):
            self.assertIn(phase, phases, f"buildspec.yml has no {phase} phase")

    def test_the_runner_the_image_entrypoint_names_exists(self):
        """The Dockerfile's ENTRYPOINT must point at a file that is actually copied."""
        dockerfile = os.path.join(DEPLOY, "Dockerfile")
        self.assertTrue(os.path.exists(dockerfile), "deploy/cc3d/Dockerfile is missing")
        with open(dockerfile, encoding="utf-8") as handle:
            text = handle.read()
        self.assertIn("cc3d_job_runner.py", text,
                      "the Dockerfile does not copy the job runner")
        self.assertTrue(
            os.path.exists(os.path.join(DEPLOY, "cc3d_job_runner.py")),
            "the Dockerfile copies cc3d_job_runner.py but the file is not in deploy/cc3d")


if __name__ == "__main__":
    unittest.main()
