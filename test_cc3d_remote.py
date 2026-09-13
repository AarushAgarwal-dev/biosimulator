"""Tests for remote CompuCell3D execution on AWS Batch.

Every AWS interaction runs against fake clients, so the suite never touches AWS,
needs no credentials, and costs nothing.
"""

import json
import os
import time
import unittest
import xml.etree.ElementTree as ET
from unittest.mock import patch

import approach_base as base
import approach_cc3d as cc3d_mod
import cc3d_remote
import geometry as geo


REMOTE_ENV = {
    cc3d_remote.REGION_ENV: "us-east-2",
    cc3d_remote.QUEUE_ENV: "cc3d-queue",
    cc3d_remote.JOB_DEFINITION_ENV: "cc3d-jobdef",
    cc3d_remote.BUCKET_ENV: "biosim-cc3d",
}

CELLS_CSV = ("mcs,cell_id,type,volume,surface,x,y,z\n"
             "0,1,Tumour,25,20,10.0,12.0,0\n"
             "0,2,Tumour,24,19,14.0,11.0,0\n"
             "10,1,Tumour,26,21,10.5,12.4,0\n"
             "10,2,Tumour,25,20,14.4,11.2,0\n")


class FakeBody:
    def __init__(self, text):
        self._text = text

    def read(self):
        return self._text.encode("utf-8")


class FakeAwsError(Exception):
    """Shaped like a botocore ClientError: the code lives in .response.

    The fake used to raise a bare KeyError for a missing object, which meant the
    tests never exercised the distinction the real code has to make -- an object that
    is ABSENT (a normal answer) versus an object we were REFUSED or failed to read (a
    different fact entirely, which must not be reported as "no results").
    """

    def __init__(self, code, message=""):
        super().__init__(message or code)
        self.response = {"Error": {"Code": code, "Message": message or code}}


class FakeS3:
    def __init__(self, objects=None, error_on_get=None):
        self.objects = dict(objects or {})
        self.puts = []
        self.deleted = []
        #: Optional error code to raise from get_object regardless of contents,
        #: for testing that a read failure is not mistaken for absence.
        self.error_on_get = error_on_get

    def put_object(self, Bucket, Key, Body, ContentType=None):
        self.objects[Key] = Body.decode("utf-8") if isinstance(Body, bytes) else Body
        self.puts.append(Key)

    def get_object(self, Bucket, Key):
        if self.error_on_get:
            raise FakeAwsError(self.error_on_get, f"denied reading {Key}")
        if Key not in self.objects:
            raise FakeAwsError("NoSuchKey", f"missing {Key}")
        return {"Body": FakeBody(self.objects[Key])}

    def list_objects_v2(self, Bucket, Prefix):
        return {"Contents": [{"Key": k} for k in self.objects if k.startswith(Prefix)]}

    def delete_objects(self, Bucket, Delete):
        for item in Delete["Objects"]:
            self.objects.pop(item["Key"], None)
            self.deleted.append(item["Key"])


class FakeBatch:
    def __init__(self, states=None):
        # A sequence of states returned by successive describe_jobs calls.
        self.states = list(states or ["SUCCEEDED"])
        self.submitted = []
        self.terminated = []

    def submit_job(self, **kwargs):
        self.submitted.append(kwargs)
        return {"jobId": "job-123", "jobName": kwargs.get("jobName")}

    def describe_jobs(self, jobs):
        state = self.states[0] if len(self.states) == 1 else self.states.pop(0)
        return {"jobs": [{"status": state, "statusReason": "",
                          "container": {"logStreamName": "stream/abc", "exitCode": 0}}]}

    def terminate_job(self, jobId, reason):
        self.terminated.append((jobId, reason))


class FakeLogs:
    def __init__(self, lines=None):
        self.lines = list(lines or ["[cc3d-runner] starting"])
        self.served = False

    def get_log_events(self, **kwargs):
        if self.served:
            return {"events": [], "nextForwardToken": "t2"}
        self.served = True
        return {"events": [{"message": line} for line in self.lines],
                "nextForwardToken": "t2"}


def _runner(s3=None, batch=None, logs=None, **overrides):
    with patch.dict(os.environ, REMOTE_ENV, clear=False):
        config = cc3d_remote.configuration_status()
    config.update(overrides)
    return cc3d_remote.BatchRunner(config, s3_client=s3 or FakeS3(),
                                   batch_client=batch or FakeBatch(),
                                   logs_client=logs or FakeLogs())


class ConfigurationTests(unittest.TestCase):
    def test_unconfigured_names_every_missing_variable(self):
        with patch.dict(os.environ, {}, clear=True):
            status = cc3d_remote.configuration_status()
        self.assertFalse(status["available"])
        self.assertFalse(status["configured"])
        for name in (cc3d_remote.REGION_ENV, cc3d_remote.QUEUE_ENV,
                     cc3d_remote.JOB_DEFINITION_ENV, cc3d_remote.BUCKET_ENV):
            self.assertIn(name, status["reason"])

    def test_partial_configuration_is_not_available(self):
        with patch.dict(os.environ, {cc3d_remote.REGION_ENV: "us-east-2"}, clear=True):
            status = cc3d_remote.configuration_status()
        self.assertFalse(status["available"])
        self.assertIn(cc3d_remote.BUCKET_ENV, status["reason"])

    def test_full_configuration_is_available(self):
        with patch.dict(os.environ, REMOTE_ENV, clear=True):
            status = cc3d_remote.configuration_status()
        self.assertTrue(status["available"])
        self.assertEqual(status["region"], "us-east-2")
        self.assertEqual(status["prefix"], cc3d_remote.DEFAULT_PREFIX)

    def test_configuration_never_touches_the_network(self):
        # A construction failure here would mean an import-time AWS call.
        with patch.dict(os.environ, REMOTE_ENV, clear=True):
            for _ in range(3):
                self.assertTrue(cc3d_remote.configuration_status()["available"])

    def test_runner_refuses_when_unconfigured(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(cc3d_remote.RemoteConfigurationError):
                cc3d_remote.BatchRunner()


class SubmissionTests(unittest.TestCase):
    def test_project_files_are_uploaded_under_the_run_prefix(self):
        s3 = FakeS3()
        runner = _runner(s3=s3)
        result = runner.submit("run_abc", {"Simulation/model.xml": "<xml/>",
                                          "Simulation/steppables.py": "pass"})
        self.assertEqual(result["job_id"], "job-123")
        self.assertIn("cc3d-runs/run_abc/Simulation/model.xml", s3.objects)
        self.assertIn("cc3d-runs/run_abc/Simulation/steppables.py", s3.objects)

    def test_job_carries_the_bucket_and_prefix_in_its_environment(self):
        batch = FakeBatch()
        runner = _runner(batch=batch)
        runner.submit("run_abc", {"Simulation/model.xml": "<xml/>"}, vcpus=8,
                      memory_mib=16384)
        overrides = batch.submitted[0]["containerOverrides"]
        environment = {item["name"]: item["value"] for item in overrides["environment"]}
        self.assertEqual(environment["CC3D_BUCKET"], "biosim-cc3d")
        self.assertEqual(environment["CC3D_PREFIX"], "cc3d-runs/run_abc")
        resources = {item["type"]: item["value"] for item in overrides["resourceRequirements"]}
        self.assertEqual(resources["VCPU"], "8")
        self.assertEqual(resources["MEMORY"], "16384")

    def test_job_name_is_sanitised(self):
        batch = FakeBatch()
        _runner(batch=batch).submit("run_with_underscores", {"a.xml": "<x/>"})
        self.assertNotIn("_", batch.submitted[0]["jobName"])


class PollingTests(unittest.TestCase):
    def test_wait_returns_on_success_and_reports_states(self):
        batch = FakeBatch(["SUBMITTED", "RUNNING", "SUCCEEDED"])
        seen, lines = [], []
        runner = _runner(batch=batch)
        final = runner.wait("job-123", poll_secs=0,
                            on_progress=lambda info: seen.append(info["status"]),
                            on_log=lambda line: lines.append(line))
        self.assertEqual(final["status"], "SUCCEEDED")
        self.assertIn("SUBMITTED", seen)
        self.assertIn("RUNNING", seen)
        self.assertTrue(lines, "CloudWatch lines should be streamed")

    def test_failure_state_is_returned_not_masked(self):
        runner = _runner(batch=FakeBatch(["FAILED"]))
        self.assertEqual(runner.wait("job-123", poll_secs=0)["status"], "FAILED")

    def test_cancellation_terminates_the_job(self):
        batch = FakeBatch(["RUNNING"])
        runner = _runner(batch=batch)
        final = runner.wait("job-123", poll_secs=0, is_cancelled=lambda: True)
        self.assertEqual(final["status"], "CANCELLED")
        self.assertTrue(batch.terminated)

    def test_pause_is_honoured_between_polls(self):
        calls = []
        runner = _runner(batch=FakeBatch(["RUNNING", "SUCCEEDED"]))
        final = runner.wait("job-123", poll_secs=0,
                            wait_if_paused=lambda: calls.append("checked"))
        self.assertEqual(final["status"], "SUCCEEDED")
        self.assertTrue(calls, "the poll loop must give the run manager a pause point")

    def test_time_spent_paused_does_not_trip_the_timeout(self):
        """A paused run must not be terminated for a deadline it spent paused."""
        batch = FakeBatch(["RUNNING", "SUCCEEDED"])
        runner = _runner(batch=batch)

        def pause_once():
            # Longer than the deadline below, so an unadjusted deadline would kill it.
            time.sleep(0.05)

        final = runner.wait("job-123", timeout_secs=0.02, poll_secs=0,
                            wait_if_paused=pause_once)
        self.assertEqual(final["status"], "SUCCEEDED")
        self.assertEqual(batch.terminated, [], "a paused job must not be terminated")

    def test_cancelling_while_paused_still_terminates_the_job(self):
        batch = FakeBatch(["RUNNING"])
        runner = _runner(batch=batch)
        final = runner.wait("job-123", poll_secs=0, wait_if_paused=lambda: None,
                            is_cancelled=lambda: True)
        self.assertEqual(final["status"], "CANCELLED")
        self.assertTrue(batch.terminated)

    def test_remote_backend_reports_pause_and_cancel_support(self):
        with patch.object(cc3d_mod, "detect_cc3d", return_value=_remote_detected()):
            capabilities = cc3d_mod.CC3D_ADAPTER.get_capabilities()
        self.assertTrue(capabilities.supports_cancel)
        self.assertTrue(capabilities.supports_pause)
        # Honest about what a pause does and does not stop.
        self.assertIn("keeps billing", capabilities.notes)

    def test_timeout_terminates_and_raises(self):
        batch = FakeBatch(["RUNNING"])
        runner = _runner(batch=batch)
        with self.assertRaises(TimeoutError) as raised:
            runner.wait("job-123", timeout_secs=-1, poll_secs=0)
        self.assertIn("did not finish", str(raised.exception))
        self.assertTrue(batch.terminated)

    def test_missing_job_is_an_error(self):
        class Empty(FakeBatch):
            def describe_jobs(self, jobs):
                return {"jobs": []}
        with self.assertRaises(RuntimeError):
            _runner(batch=Empty()).describe("job-123")

    def test_log_failure_does_not_break_the_poll(self):
        class BrokenLogs:
            def get_log_events(self, **kwargs):
                raise RuntimeError("no such stream")
        runner = _runner(batch=FakeBatch(["SUCCEEDED"]), logs=BrokenLogs())
        self.assertEqual(runner.wait("job-123", poll_secs=0,
                                     on_log=lambda line: None)["status"], "SUCCEEDED")


class ResultTests(unittest.TestCase):
    def test_results_are_fetched(self):
        s3 = FakeS3({
            "cc3d-runs/run_abc/cells.csv": CELLS_CSV,
            "cc3d-runs/run_abc/status.json": json.dumps({"state": "succeeded", "seconds": 12}),
        })
        fetched = _runner(s3=s3).fetch_results("run_abc")
        self.assertIn("mcs,cell_id", fetched["cells_csv"])
        self.assertEqual(fetched["status"]["state"], "succeeded")
        self.assertTrue(fetched["s3_prefix"].startswith("s3://biosim-cc3d/"))

    def test_missing_results_is_an_error_not_an_empty_set(self):
        s3 = FakeS3({"cc3d-runs/run_abc/status.json":
                     json.dumps({"state": "failed", "error": "steppable did not run"})})
        with self.assertRaises(RuntimeError) as raised:
            _runner(s3=s3).fetch_results("run_abc")
        message = str(raised.exception)
        self.assertIn("no measurement file", message)
        self.assertIn("steppable did not run", message)

    def test_a_read_failure_is_not_reported_as_a_missing_result(self):
        """AccessDenied must not be described as "the job produced nothing".

        Collapsing every S3 error to "absent" sent a researcher to debug their model
        when the real problem was an IAM policy. The two facts are different and the
        message has to say which one happened.
        """
        s3 = FakeS3({"cc3d-runs/run_abc/cells.csv": CELLS_CSV}, error_on_get="AccessDenied")
        with self.assertRaises(RuntimeError) as raised:
            _runner(s3=s3).fetch_results("run_abc")
        message = str(raised.exception)
        self.assertIn("Could not read", message)
        self.assertIn("AccessDenied", message)
        self.assertNotIn("no measurement file", message)

    def test_an_interrupted_run_tells_the_researcher_a_partial_exists(self):
        """A spot reclamation writes cells.partial.csv. Saying only "nothing was
        produced" hides output that may already answer the question."""
        s3 = FakeS3({"cc3d-runs/run_abc/status.json": json.dumps({
            "state": "interrupted",
            "error": "the spot instance was reclaimed",
            "partial_key": "cells.partial.csv",
        })})
        with self.assertRaises(RuntimeError) as raised:
            _runner(s3=s3).fetch_results("run_abc")
        message = str(raised.exception)
        self.assertIn("cells.partial.csv", message)
        self.assertIn("PARTIAL", message)

    def test_cleanup_removes_the_run_objects(self):
        s3 = FakeS3({"cc3d-runs/run_abc/cells.csv": CELLS_CSV,
                     "cc3d-runs/run_abc/status.json": "{}",
                     "cc3d-runs/other/cells.csv": CELLS_CSV})
        removed = _runner(s3=s3).cleanup("run_abc")
        self.assertEqual(removed, 2)
        self.assertIn("cc3d-runs/other/cells.csv", s3.objects)

    def test_cost_estimate_is_labelled_an_estimate(self):
        estimate = cc3d_remote.estimate_cost(4, 8192, 30)
        self.assertTrue(estimate["is_estimate"])
        self.assertGreater(estimate["estimated_usd"], 0)
        self.assertIn("excludes", estimate["basis"])

    def test_cost_estimate_default_rate_is_the_documented_m7i_figure(self):
        estimate = cc3d_remote.estimate_cost(cc3d_remote.DEFAULT_VCPUS,
                                             cc3d_remote.DEFAULT_MEMORY_MIB, 60)
        self.assertEqual(estimate["usd_per_vcpu_hour"],
                         cc3d_remote.M7I_XLARGE_SPOT_USD_PER_VCPU_HOUR)
        self.assertIn("m7i.xlarge", estimate["basis"])
        # One hour of the whole 4 vCPU instance, so the rate times its vCPU count.
        self.assertAlmostEqual(
            estimate["estimated_usd"],
            round(4 * cc3d_remote.M7I_XLARGE_SPOT_USD_PER_VCPU_HOUR, 4), places=4)

    def test_cost_estimate_rate_is_overridable_by_the_caller(self):
        default = cc3d_remote.estimate_cost(4, 15500, 60)["estimated_usd"]
        override = cc3d_remote.estimate_cost(4, 15500, 60,
                                             spot_price_per_vcpu_hour=0.05)
        self.assertNotEqual(default, override["estimated_usd"])
        self.assertEqual(override["usd_per_vcpu_hour"], 0.05)


class JobSizingTests(unittest.TestCase):
    """The job definition targets m7i.xlarge: 4 vCPU, 16 GiB."""

    def test_submit_defaults_match_the_m7i_xlarge_job_definition(self):
        self.assertEqual(cc3d_remote.DEFAULT_VCPUS, 4)
        self.assertEqual(cc3d_remote.DEFAULT_MEMORY_MIB, 15500)
        batch = FakeBatch()
        _runner(batch=batch).submit("run_defaults", {"Simulation/model.xml": "<xml/>"})
        resources = {item["type"]: item["value"]
                     for item in batch.submitted[0]["containerOverrides"]["resourceRequirements"]}
        self.assertEqual(resources["VCPU"], "4")
        self.assertEqual(resources["MEMORY"], "15500")

    def test_memory_default_leaves_ecs_headroom_below_the_instance_total(self):
        # 16384 MiB would never be placeable: ECS holds memory back for its agent.
        self.assertLess(cc3d_remote.DEFAULT_MEMORY_MIB, 16384)
        self.assertGreater(cc3d_remote.DEFAULT_MEMORY_MIB, 15000)

    def test_adapter_passes_the_same_defaults_to_batch(self):
        self.assertEqual(cc3d_mod.REMOTE_DEFAULT_VCPUS, 4)
        self.assertEqual(cc3d_mod.REMOTE_DEFAULT_MEMORY_MIB, 15500)


class RunLengthCapTests(unittest.TestCase):
    """A remote run longer than the cap is refused before it can be submitted.

    AWS Batch spot capacity restarts an interrupted job from step 0 rather than
    resuming it, so an over-long run can be billed repeatedly and still never finish.
    """

    LATTICE = {"x": 200, "y": 200, "z": 50}

    def test_cap_is_ninety_minutes_and_a_module_constant(self):
        self.assertEqual(cc3d_remote.REMOTE_RUN_LENGTH_CAP_MINUTES, 90.0)
        self.assertEqual(cc3d_mod.REMOTE_RUN_LENGTH_CAP_MINUTES, 90.0)

    def test_estimate_grows_with_steps_and_lattice_size(self):
        small = cc3d_remote.estimate_runtime_minutes(1000, {"x": 50, "y": 50, "z": 1})
        more_steps = cc3d_remote.estimate_runtime_minutes(2000, {"x": 50, "y": 50, "z": 1})
        bigger = cc3d_remote.estimate_runtime_minutes(1000, {"x": 100, "y": 100, "z": 1})
        self.assertGreater(more_steps, small)
        self.assertGreater(bigger, small)
        # Startup overhead is charged even for a trivial run.
        self.assertGreaterEqual(cc3d_remote.estimate_runtime_minutes(1, {"x": 1, "y": 1, "z": 1}),
                                cc3d_remote.REMOTE_STARTUP_OVERHEAD_MINUTES)

    def test_diffusion_fields_make_the_estimate_more_pessimistic(self):
        without = cc3d_remote.estimate_runtime_minutes(1000, self.LATTICE, field_count=0)
        with_two = cc3d_remote.estimate_runtime_minutes(1000, self.LATTICE, field_count=2)
        self.assertGreater(with_two, without)

    def test_just_under_the_cap_is_accepted(self):
        steps = cc3d_remote.max_steps_within_cap(self.LATTICE)
        check = cc3d_remote.run_length_check(steps, self.LATTICE)
        self.assertTrue(check["within_cap"], check)
        self.assertLessEqual(check["estimated_minutes"], 90.0)
        self.assertEqual(check["message"], "")

    def test_just_over_the_cap_is_refused_with_a_message_naming_the_cap(self):
        steps = cc3d_remote.max_steps_within_cap(self.LATTICE) + 1000
        check = cc3d_remote.run_length_check(steps, self.LATTICE)
        self.assertFalse(check["within_cap"])
        self.assertIn("90 minute cap", check["message"])
        self.assertIn("Monte Carlo step count", check["message"])
        self.assertIn("spot", check["message"])
        self.assertIn(str(check["max_steps"]), check["message"],
                      "the refusal must name the step count that would fit")

    def test_unreadable_lattice_does_not_produce_a_spurious_cap_error(self):
        # The lattice validators already report this; a second message would confuse.
        check = cc3d_remote.run_length_check(10 ** 9, {"x": "wide", "y": 10, "z": 1})
        self.assertTrue(check["within_cap"])

    def test_validate_refuses_an_over_long_remote_configuration(self):
        steps = cc3d_remote.max_steps_within_cap(self.LATTICE) + 1000
        project = _cc3d_project(lattice=dict(self.LATTICE), steps=steps)
        with patch.object(cc3d_mod, "detect_cc3d", return_value=_remote_detected()):
            issues = cc3d_mod.CC3D_ADAPTER.validate(project)
        offending = [i for i in issues if i.code == "cc3d_remote_run_too_long"]
        self.assertEqual(len(offending), 1, [i.code for i in issues])
        self.assertEqual(offending[0].severity, "error")
        self.assertIn("90 minute cap", offending[0].message)
        self.assertIn("Monte Carlo step count", offending[0].message)

    def test_validate_accepts_a_configuration_just_under_the_cap(self):
        steps = cc3d_remote.max_steps_within_cap(self.LATTICE)
        project = _cc3d_project(lattice=dict(self.LATTICE), steps=steps)
        with patch.object(cc3d_mod, "detect_cc3d", return_value=_remote_detected()):
            issues = cc3d_mod.CC3D_ADAPTER.validate(project)
        self.assertEqual([i.code for i in issues if i.severity == "error"], [])

    def test_compile_refuses_an_over_long_remote_configuration(self):
        steps = cc3d_remote.max_steps_within_cap(self.LATTICE) + 1000
        project = _cc3d_project(lattice=dict(self.LATTICE), steps=steps)
        with patch.object(cc3d_mod, "detect_cc3d", return_value=_remote_detected()):
            with self.assertRaises(ValueError) as raised:
                cc3d_mod.CC3D_ADAPTER.compile(project)
        self.assertIn("90 minute cap", str(raised.exception))

    def test_the_cap_does_not_apply_to_a_local_backend(self):
        """A long run is merely slow locally; nothing restarts it from scratch."""
        steps = cc3d_remote.max_steps_within_cap(self.LATTICE) + 1000
        project = _cc3d_project(lattice=dict(self.LATTICE), steps=steps)
        local = {"available": True, "method": "runscript-path", "version": "4.4.1",
                 "path": "/opt/cc3d/runScript.sh", "reason": ""}
        with patch.object(cc3d_mod, "detect_cc3d", return_value=local):
            issues = cc3d_mod.CC3D_ADAPTER.validate(project)
        self.assertNotIn("cc3d_remote_run_too_long", [i.code for i in issues])

    def test_over_long_run_is_refused_before_a_job_is_submitted(self):
        steps = cc3d_remote.max_steps_within_cap(self.LATTICE) + 1000
        project = _cc3d_project(lattice=dict(self.LATTICE), steps=steps)
        batch, s3 = FakeBatch(), FakeS3()
        detected = _remote_detected()
        real_runner = cc3d_remote.BatchRunner           # captured before patching

        def runner_factory(config, **kwargs):
            return real_runner(config, s3_client=s3, batch_client=batch,
                               logs_client=FakeLogs())

        with patch.dict(os.environ, REMOTE_ENV, clear=True):
            # Compile under a local backend so the run path, not compile(), does the
            # refusing -- this is the "raised the step count after compiling" case.
            local = {"available": True, "method": "runscript-path", "version": "4.4.1",
                     "path": "/opt/cc3d/runScript.sh", "reason": ""}
            with patch.object(cc3d_mod, "detect_cc3d", return_value=local):
                compiled = cc3d_mod.CC3D_ADAPTER.compile(project)
            with patch.object(cc3d_remote, "BatchRunner", side_effect=runner_factory):
                with self.assertRaises(ValueError) as raised:
                    cc3d_mod.CC3D_ADAPTER._run_remote(
                        compiled, base.RunContext(run_id="run_too_long"), detected)
        self.assertIn("90 minute cap", str(raised.exception))
        self.assertEqual(batch.submitted, [], "no job may be submitted or billed")
        self.assertEqual(s3.puts, [], "no project may be uploaded")


class BackendReportingTests(unittest.TestCase):
    """Availability must say WHICH of the four backends will run the simulation."""

    def test_all_four_backends_have_a_label(self):
        self.assertEqual(set(cc3d_mod.BACKEND_LABELS),
                         {"python-package", "runscript-env", "runscript-path", "aws-batch"})

    def test_each_backend_is_named_in_its_capabilities(self):
        cases = {
            "python-package": {"available": True, "method": "python-package",
                               "version": "4.4.1", "path": "cc3d/__init__.py", "reason": ""},
            "runscript-env": {"available": True, "method": "runscript-env",
                              "version": "unknown", "path": "/opt/cc3d/runScript.sh",
                              "reason": ""},
            "runscript-path": {"available": True, "method": "runscript-path",
                               "version": "unknown", "path": "/usr/bin/runScript.sh",
                               "reason": ""},
            "aws-batch": _remote_detected(),
        }
        for method, detected in cases.items():
            with self.subTest(backend=method):
                with patch.object(cc3d_mod, "detect_cc3d", return_value=detected):
                    capabilities = cc3d_mod.CC3D_ADAPTER.get_capabilities()
                self.assertTrue(capabilities.available)
                self.assertIn(cc3d_mod.BACKEND_LABELS[method], capabilities.notes,
                              "capabilities must name the backend that will be used")
                self.assertIn("Backend:", capabilities.notes)

    def test_notes_enumerate_all_four_backends(self):
        with patch.object(cc3d_mod, "detect_cc3d", return_value=_remote_detected()):
            notes = cc3d_mod.CC3D_ADAPTER.get_capabilities().notes
        for fragment in ("'cc3d' Python package", cc3d_mod.RUNSCRIPT_ENV, "PATH",
                         "AWS Batch"):
            self.assertIn(fragment, notes)

    def test_remote_backend_is_available_on_the_four_env_vars_plus_boto3(self):
        with patch.dict(os.environ, REMOTE_ENV, clear=True):
            status = cc3d_remote.configuration_status()
        self.assertTrue(status["available"], status.get("reason"))
        for name in (cc3d_remote.REGION_ENV, cc3d_remote.QUEUE_ENV,
                     cc3d_remote.JOB_DEFINITION_ENV, cc3d_remote.BUCKET_ENV):
            partial = {k: v for k, v in REMOTE_ENV.items() if k != name}
            with self.subTest(missing=name):
                with patch.dict(os.environ, partial, clear=True):
                    self.assertFalse(cc3d_remote.configuration_status()["available"])

    def test_remote_availability_requires_boto3(self):
        import builtins
        real_import = builtins.__import__

        def no_boto3(name, *args, **kwargs):
            if name == "boto3":
                raise ImportError("no module named boto3")
            return real_import(name, *args, **kwargs)

        with patch.dict(os.environ, REMOTE_ENV, clear=True):
            with patch.object(builtins, "__import__", side_effect=no_boto3):
                status = cc3d_remote.configuration_status()
        self.assertFalse(status["available"])
        self.assertTrue(status["configured"], "the variables were set; only boto3 is missing")
        self.assertIn("boto3", status["reason"])

    def test_remote_capabilities_name_the_queue_and_bucket(self):
        with patch.object(cc3d_mod, "detect_cc3d", return_value=_remote_detected()):
            capabilities = cc3d_mod.CC3D_ADAPTER.get_capabilities()
        joined = " ".join(capabilities.requirements)
        self.assertIn("cc3d-queue", joined)
        self.assertIn("biosim-cc3d", joined)
        self.assertIn("boto3", joined)
        self.assertIn("AWS Batch", capabilities.engine_name)

    def test_no_backend_reports_every_one_that_was_checked(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch.object(cc3d_mod.shutil, "which", return_value=None):
                detected = cc3d_mod.detect_cc3d()
                capabilities = cc3d_mod.CC3D_ADAPTER.get_capabilities()
        self.assertFalse(detected["available"])
        self.assertEqual(cc3d_mod.describe_backend(detected),
                         "none (CompuCell3D cannot run on this host)")
        self.assertFalse(capabilities.available)
        for fragment in ("'cc3d' Python package", cc3d_mod.RUNSCRIPT_ENV, "PATH",
                         "AWS Batch"):
            self.assertIn(fragment, capabilities.unavailable_reason)


class NoFallbackTests(unittest.TestCase):
    """Selecting CompuCell3D and getting a different engine would be dishonest.

    ABM is also a Cellular Potts model, so a substituted result would look plausible
    while using different contact energies, lattice type and engine version. MPC is a
    control method and not a simulator at all. Neither may be offered as a stand-in.
    """

    UNAVAILABLE = {"available": False, "method": "", "version": "", "path": "",
                   "reason": "CompuCell3D is not installed."}

    def test_refusal_names_neither_abm_nor_mpc(self):
        with patch.object(cc3d_mod, "detect_cc3d", return_value=self.UNAVAILABLE):
            compiled = cc3d_mod.CC3D_ADAPTER.compile(_cc3d_project())
            with self.assertRaises(base.ApproachUnavailable) as raised:
                cc3d_mod.CC3D_ADAPTER.run(compiled, base.RunContext(run_id="no_engine"))
        message = str(raised.exception)
        self.assertIn("not installed", message)
        self.assertNotIn("abm", message.lower(),
                         "refusal must not point at a different approach as a substitute")
        self.assertNotIn("mpc", message.lower(),
                         "refusal must not point at a different approach as a substitute")

    def test_real_unavailable_reason_names_neither_abm_nor_mpc(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch.object(cc3d_mod.shutil, "which", return_value=None):
                reason = cc3d_mod.detect_cc3d()["reason"]
        self.assertNotIn("abm", reason.lower())
        self.assertNotIn("mpc", reason.lower())

    def test_configuration_and_export_survive_without_any_backend(self):
        project = _cc3d_project()
        with patch.object(cc3d_mod, "detect_cc3d", return_value=self.UNAVAILABLE):
            issues = cc3d_mod.CC3D_ADAPTER.validate(project)
            exported = cc3d_mod.CC3D_ADAPTER.export_configuration(project)
            capabilities = cc3d_mod.CC3D_ADAPTER.get_capabilities()
        self.assertEqual([i for i in issues if i.severity == "error"], [])
        self.assertFalse(exported["available"])
        self.assertTrue(capabilities.supports_export)
        # The export must be runnable elsewhere, not a stub.
        root = ET.fromstring(exported["files"]["Simulation/model.xml"])
        self.assertEqual(root.tag, "CompuCell3D")
        self.assertEqual(root.find("Potts").find("Steps").text, "50")
        self.assertIn("runScript", exported["files"]["README.txt"])

    def test_no_module_spells_mpc_as_mcp(self):
        here = os.path.dirname(os.path.abspath(__file__))
        for name in ("approach_cc3d.py", "cc3d_remote.py"):
            with self.subTest(module=name):
                with open(os.path.join(here, name), "r", encoding="utf-8") as handle:
                    self.assertNotIn("MCP", handle.read())


def _remote_detected():
    """A detection result for the remote backend, as detect_cc3d would return it."""
    with patch.dict(os.environ, REMOTE_ENV, clear=True):
        remote = cc3d_remote.configuration_status()
    return {"available": True, "method": "aws-batch", "version": "remote",
            "path": f"s3://{remote['bucket']}/{remote['prefix']}", "reason": "",
            "remote": remote}


def _cc3d_project(**overrides):
    config = {
        "lattice": {"x": 40, "y": 40, "z": 1}, "steps": 50, "temperature": 10.0,
        "cell_types": [{"type_id": 1, "name": "Tumour"}],
        "expected_minutes": 5,
    }
    config.update(overrides)
    return {
        "domain": geo.make_rectangle(40.0, 40.0),
        "selected_approach": "cc3d",
        "approaches": {"cc3d": config},
    }


class AdapterIntegrationTests(unittest.TestCase):
    def test_remote_configuration_makes_the_approach_available(self):
        with patch.dict(os.environ, REMOTE_ENV, clear=True):
            detected = cc3d_mod.detect_cc3d()
        # A local install would win; on a host without one, remote is the path.
        if detected["method"] == "aws-batch":
            self.assertTrue(detected["available"])
            capabilities = None
            with patch.dict(os.environ, REMOTE_ENV, clear=True):
                capabilities = cc3d_mod.CC3D_ADAPTER.get_capabilities()
            self.assertTrue(capabilities.available)
            self.assertIn("AWS Batch", capabilities.engine_name)
            self.assertIn("billed only while it runs", capabilities.notes)

    def test_local_install_takes_precedence_over_remote(self):
        """Running locally is faster and free, so it must win when both are possible."""
        with patch.dict(os.environ, {**REMOTE_ENV, cc3d_mod.RUNSCRIPT_ENV: ""}, clear=True):
            with patch.object(cc3d_mod.shutil, "which", return_value="/opt/cc3d/runScript.sh"):
                with patch.object(cc3d_mod.os.path, "isfile", return_value=True):
                    detected = cc3d_mod.detect_cc3d()
        self.assertEqual(detected["method"], "runscript-path")

    def test_unavailable_reason_covers_both_local_and_remote(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch.object(cc3d_mod.shutil, "which", return_value=None):
                detected = cc3d_mod.detect_cc3d()
        self.assertFalse(detected["available"])
        self.assertIn("No local install", detected["reason"])
        self.assertIn("remote execution is not configured", detected["reason"])

    def test_remote_run_end_to_end_with_fake_aws(self):
        project = _cc3d_project()
        s3 = FakeS3()
        batch = FakeBatch(["SUBMITTED", "RUNNING", "SUCCEEDED"])

        real_runner = cc3d_remote.BatchRunner

        def runner_factory(config, **kwargs):
            instance = real_runner(config, s3_client=s3, batch_client=batch,
                                   logs_client=FakeLogs())
            # The container would write these; the fake fills them in on submit.
            original_submit = instance.submit

            def submit(run_id, files, **rest):
                out = original_submit(run_id, files, **rest)
                s3.objects[instance.key(run_id, "cells.csv")] = CELLS_CSV
                s3.objects[instance.key(run_id, "status.json")] = json.dumps(
                    {"state": "succeeded", "seconds": 42, "version": "4.4.1"})
                return out

            instance.submit = submit
            return instance

        with patch.dict(os.environ, REMOTE_ENV, clear=True):
            with patch.object(cc3d_remote, "BatchRunner", side_effect=runner_factory):
                compiled = cc3d_mod.CC3D_ADAPTER.compile(project)
                logs = []
                context = base.RunContext(run_id="run_remote1",
                                          on_log=lambda m, level: logs.append(m))
                results = cc3d_mod.CC3D_ADAPTER._run_remote(
                    compiled, context, {"remote": cc3d_remote.configuration_status()})

        self.assertEqual(results["engine"], "CompuCell3D")
        self.assertEqual(results["execution"]["location"], "aws-batch")
        self.assertEqual(results["execution"]["job_id"], "job-123")
        self.assertEqual(results["engine_version"], "4.4.1")
        # Parsed into the same shape as a local run, so the viewer needs no branch.
        self.assertEqual(results["t"], [0, 10])
        self.assertEqual(len(results["cells"][0]), 2)
        self.assertEqual(results["summary"]["final_cell_count"], 2)
        self.assertTrue(any("Dispatching CompuCell3D to AWS Batch" in line for line in logs))
        self.assertTrue(any("Estimated compute cost" in line for line in logs))

    def test_remote_failure_is_reported_with_the_container_status(self):
        project = _cc3d_project()
        s3 = FakeS3()
        batch = FakeBatch(["FAILED"])

        real_runner = cc3d_remote.BatchRunner

        def runner_factory(config, **kwargs):
            instance = real_runner(config, s3_client=s3, batch_client=batch,
                                   logs_client=FakeLogs())
            original_submit = instance.submit

            def submit(run_id, files, **rest):
                out = original_submit(run_id, files, **rest)
                s3.objects[instance.key(run_id, "status.json")] = json.dumps(
                    {"state": "failed", "error": "runScript not found in image"})
                return out

            instance.submit = submit
            return instance

        with patch.dict(os.environ, REMOTE_ENV, clear=True):
            with patch.object(cc3d_remote, "BatchRunner", side_effect=runner_factory):
                compiled = cc3d_mod.CC3D_ADAPTER.compile(project)
                context = base.RunContext(run_id="run_remote2")
                with self.assertRaises(RuntimeError) as raised:
                    cc3d_mod.CC3D_ADAPTER._run_remote(
                        compiled, context, {"remote": cc3d_remote.configuration_status()})
        message = str(raised.exception)
        self.assertIn("failed on AWS", message)
        self.assertIn("runScript not found", message)


class ContainerRunnerTests(unittest.TestCase):
    def test_runner_script_is_valid_python(self):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "deploy", "cc3d", "cc3d_job_runner.py")
        with open(path, "r", encoding="utf-8") as handle:
            source = handle.read()
        compile(source, "cc3d_job_runner.py", "exec")
        # The contract the application depends on.
        self.assertIn("cells.csv", source)
        self.assertIn("status.json", source)

    def test_dockerfile_uses_the_official_channel_and_verifies_the_engine(self):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "deploy", "cc3d", "Dockerfile")
        with open(path, "r", encoding="utf-8") as handle:
            text = handle.read()
        self.assertIn("-c compucell3d", text)
        self.assertIn("import cc3d", text, "the build must fail if the engine is missing")
        self.assertIn("QT_QPA_PLATFORM=offscreen", text, "headless rendering is required")


if __name__ == "__main__":
    unittest.main()
