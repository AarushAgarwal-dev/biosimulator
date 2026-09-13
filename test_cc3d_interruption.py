"""Executes the spot-interruption logic in ``deploy/cc3d/cc3d_job_runner.py``.

That module is the AWS Batch container entrypoint. Roughly 90 lines of it -- the
SIGTERM handler, the IMDSv2 interruption-notice watcher, the bounded partial upload
and the status reporting -- had never been RUN, only read. CompuCell3D and a real
boto3/AWS are unavailable on a developer host, so the module is imported with fake
``boto3``/``botocore``/``cc3d`` packages injected into ``sys.modules`` and the
CC3D_* environment variables set, which leaves the pure logic fully exercisable.

The properties tested here are the ones that protect a researcher's data:

1. partial output can NEVER reach ``cells.csv`` by any path (the application reads
   that key as a COMPLETE result);
2. ``request_shutdown`` is atomic -- no record with one caller's state and another
   caller's reason;
3. ``status.json`` is written on every reachable exit path;
4. ``_read_complete_lines`` never ships a torn row and honours its byte cap;
5. the IMDSv2 poll treats 404 as normal, refreshes on 401, gives up silently when
   the endpoint is unreachable, and treats an unreadable 200 body as a notice;
6. ``finalize_shutdown`` returns the interrupted exit code even when the upload
   raises;
7. ``steps_from_model`` honours CC3D_STEPS and ignores an unparsable override;
8. ``_build_specs_from_config`` refuses a config with no cell types.

Nothing here touches AWS, the network, or CompuCell3D. Every test exits on its own.
"""

import glob
import importlib.util
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
import urllib.error
from unittest.mock import patch

TEST_BUCKET = "biosim-cc3d-unit"
TEST_PREFIX = "runs/run-unit"
TEST_RUN_ID = "run-unit"

RUNNER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "deploy", "cc3d", "cc3d_job_runner.py")


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------
class FakeS3Error(OSError):
    """Shaped like the OSError that once turned exit 75 into an unhandled exit 1."""


class _Paginator:
    def __init__(self, pages):
        self._pages = pages

    def paginate(self, **kwargs):
        return list(self._pages)


#: Global call sequence, so writes made through DIFFERENT clients can still be put
#: back in the order they actually happened -- finalize_shutdown deliberately uses
#: the urgent client for the first status and the normal one on its failure path,
#: and "which status.json write landed last" is the whole point of several tests.
_CALL_SEQ = [0]


def _next_seq():
    _CALL_SEQ[0] += 1
    return _CALL_SEQ[0]


class FakeS3:
    """Records every call. Any op named in ``fail`` is recorded, then raises."""

    def __init__(self, name="s3", fail=()):
        self.name = name
        self.fail = set(fail)
        self.calls = []
        self.objects = {}
        self.pages = []

    # -- helpers -----------------------------------------------------------
    def _record(self, op, **kwargs):
        kwargs["_seq"] = _next_seq()
        self.calls.append((op, kwargs))
        if op in self.fail:
            raise FakeS3Error(f"simulated {op} failure on the {self.name} client")

    def ops(self, *names):
        return [kw for op, kw in self.calls if op in names]

    def written_keys(self):
        return [kw.get("Key") for kw in self.ops("put_object", "upload_file")]

    def status_writes(self):
        """(sequence, payload) for every status.json write on this client."""
        out = []
        for kw in self.ops("put_object"):
            if str(kw.get("Key", "")).endswith(runner.STATUS_KEY):
                body = kw.get("Body")
                payload = json.loads(body.decode() if isinstance(body, bytes)
                                     else body)
                out.append((kw["_seq"], payload))
        return out

    def status_payloads(self):
        return [payload for _, payload in self.status_writes()]

    # -- the S3 surface the runner uses ------------------------------------
    def put_object(self, Bucket=None, Key=None, Body=None, ContentType=None):
        self._record("put_object", Bucket=Bucket, Key=Key, Body=Body,
                     ContentType=ContentType)
        self.objects[Key] = Body

    def upload_file(self, Filename, Bucket, Key, ExtraArgs=None):
        body = None
        try:
            with open(Filename, "rb") as handle:
                body = handle.read()
        except OSError:
            pass
        self._record("upload_file", Bucket=Bucket, Key=Key, Filename=Filename,
                     Body=body, ExtraArgs=ExtraArgs)
        self.objects[Key] = body

    def download_file(self, Bucket, Key, Filename):
        self._record("download_file", Bucket=Bucket, Key=Key, Filename=Filename)
        with open(Filename, "wb") as handle:
            handle.write(b"downloaded\n")

    def delete_object(self, Bucket=None, Key=None):
        self._record("delete_object", Bucket=Bucket, Key=Key)
        self.objects.pop(Key, None)

    def get_paginator(self, name):
        self._record("get_paginator", name=name)
        return _Paginator(self.pages)


class FakeResponse:
    """Minimal urlopen() context manager."""

    def __init__(self, body):
        self._body = body

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def http_error(code, url="http://169.254.169.254/x"):
    return urllib.error.HTTPError(url, code, f"status {code}", {}, None)


class ScriptedStopEvent:
    """A stop event whose ``wait`` returns False a fixed number of times.

    Deterministic, so the watcher's loop count is exact and the suite never sleeps.
    """

    def __init__(self, false_rounds):
        self.remaining = int(false_rounds)
        self.waits = []

    def wait(self, timeout=None):
        self.waits.append(timeout)
        if self.remaining <= 0:
            return True
        self.remaining -= 1
        return False

    def set(self):
        self.remaining = 0


class FakeCell:
    def __init__(self, cid, cell_type="Tumour"):
        self.id = cid
        self.type = cell_type
        self.volume = 25.0
        self.surface = 20.0
        self.xCOM = 10.0 + cid
        self.yCOM = 12.0 + cid
        self.zCOM = 0.0


# ---------------------------------------------------------------------------
# Stub third-party packages, injected before the runner is imported
# ---------------------------------------------------------------------------
class RecordingSpec:
    """Stands in for a CompuCell3D PyCoreSpecs object, recording what was asked."""

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.params = []
        self.regions = []

    def param_new(self, *args, **kwargs):
        self.params.append((args, kwargs))

    def region_new(self, *args, **kwargs):
        self.regions.append((args, kwargs))

    def __repr__(self):  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} {self.args} {self.kwargs}>"


class PottsCore(RecordingSpec):
    pass


class CellTypePlugin(RecordingSpec):
    pass


class VolumePlugin(RecordingSpec):
    pass


class ContactPlugin(RecordingSpec):
    pass


class BlobInitializer(RecordingSpec):
    pass


class CenterOfMassPlugin(RecordingSpec):
    pass


class SurfacePlugin(RecordingSpec):
    pass


class StubSteppable:
    """Stands in for SteppableBasePy; ``cell_list`` is set per test."""

    cell_list = []

    def __init__(self, *args, **kwargs):
        pass


class StubSimService:
    """Stands in for CC3DSimService, recording the lifecycle calls."""

    instances = []
    step_hook = None
    finish_error = None

    def __init__(self, *args, **kwargs):
        self.calls = []
        self.specs = None
        self.steppables = []
        self.steps = 0
        StubSimService.instances.append(self)

    @classmethod
    def reset(cls):
        cls.instances = []
        cls.step_hook = None
        cls.finish_error = None

    def register_specs(self, specs):
        self.specs = specs
        self.calls.append("register_specs")

    def register_steppable(self, steppable=None, **kwargs):
        self.steppables.append(steppable)
        self.calls.append("register_steppable")

    def run(self):
        self.calls.append("run")

    def init(self):
        self.calls.append("init")

    def start(self):
        self.calls.append("start")

    def step(self):
        self.steps += 1
        self.calls.append("step")
        if StubSimService.step_hook is not None:
            StubSimService.step_hook(self.steps)

    def finish(self):
        self.calls.append("finish")
        if StubSimService.finish_error is not None:
            raise StubSimService.finish_error


class _ImportTimeS3Client:
    """The client created at import time. Using it is a test bug, so it shouts."""

    def __getattr__(self, name):
        def _refuse(*args, **kwargs):
            raise AssertionError(
                f"the import-time stub S3 client was used ({name}); a test must "
                f"install a FakeS3 on the module first")
        return _refuse


def _module(name, **attrs):
    mod = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    return mod


#: The sys.modules entries this file replaces, captured BEFORE the stubs go in so
#: tearDownModule can put reality back.
#:
#: Why this matters: the stubs used to be installed at import time and never removed,
#: so every test module imported after this one saw a FAKE cc3d -- which made
#: approach_cc3d.detect_cc3d() report a local CompuCell3D install that does not exist,
#: and three unrelated tests in test_cc3d_remote failed purely because of import order.
#: The boto3 stub is worse: boto3 is a REAL dependency here, so the stub shadowed the
#: genuine module for anything that ran afterwards. A test that makes other tests lie
#: is more dangerous than the bug it was written to catch.
_STUBBED_MODULE_NAMES = (
    "boto3", "boto3.session", "botocore", "botocore.config",
    "cc3d", "cc3d.core", "cc3d.core.PyCoreSpecs", "cc3d.core.PySteppables",
    "cc3d.CompuCellSetup", "cc3d.CompuCellSetup.CC3DCaller",
)
_REAL_MODULES_BEFORE_STUBS = {
    name: sys.modules.get(name) for name in _STUBBED_MODULE_NAMES
}


def tearDownModule():
    """Restore sys.modules so this file cannot contaminate later test modules."""
    for name, original in _REAL_MODULES_BEFORE_STUBS.items():
        if original is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = original
    sys.modules.pop("cc3d_job_runner", None)


def _install_stub_packages():
    """Inject fake boto3/botocore/cc3d so the runner imports on a dev host."""
    class _Session:
        def client(self, *args, **kwargs):
            return _ImportTimeS3Client()

    boto3_session = _module("boto3.session", Session=_Session)
    boto3 = _module("boto3", session=boto3_session)

    class _BotoConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    botocore_config = _module("botocore.config", Config=_BotoConfig)
    botocore = _module("botocore", config=botocore_config)

    specs_mod = _module(
        "cc3d.core.PyCoreSpecs",
        PottsCore=PottsCore, CellTypePlugin=CellTypePlugin,
        VolumePlugin=VolumePlugin, ContactPlugin=ContactPlugin,
        BlobInitializer=BlobInitializer, CenterOfMassPlugin=CenterOfMassPlugin,
        SurfacePlugin=SurfacePlugin)
    steppables_mod = _module("cc3d.core.PySteppables", SteppableBasePy=StubSteppable)
    core_mod = _module("cc3d.core", PyCoreSpecs=specs_mod, PySteppables=steppables_mod)
    core_mod.__path__ = []
    caller_mod = _module("cc3d.CompuCellSetup.CC3DCaller", CC3DSimService=StubSimService)
    setup_mod = _module("cc3d.CompuCellSetup", CC3DCaller=caller_mod)
    setup_mod.__path__ = []
    cc3d_mod = _module("cc3d", core=core_mod, CompuCellSetup=setup_mod,
                       __version__="4.4.0-stub")
    cc3d_mod.__path__ = []

    sys.modules.update({
        "boto3": boto3, "boto3.session": boto3_session,
        "botocore": botocore, "botocore.config": botocore_config,
        "cc3d": cc3d_mod,
        "cc3d.core": core_mod,
        "cc3d.core.PyCoreSpecs": specs_mod,
        "cc3d.core.PySteppables": steppables_mod,
        "cc3d.CompuCellSetup": setup_mod,
        "cc3d.CompuCellSetup.CC3DCaller": caller_mod,
    })


def _load_runner():
    _install_stub_packages()
    os.environ["CC3D_BUCKET"] = TEST_BUCKET
    os.environ["CC3D_PREFIX"] = TEST_PREFIX
    os.environ["CC3D_RUN_ID"] = TEST_RUN_ID
    for name in ("CC3D_STEPS", "CC3D_MAX_MINUTES", "CC3D_SAVE_EVERY", "CC3D_RUNSCRIPT"):
        os.environ.pop(name, None)
    spec = importlib.util.spec_from_file_location("cc3d_job_runner", RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["cc3d_job_runner"] = module
    spec.loader.exec_module(module)
    return module


runner = _load_runner()

RESULT_FULL_KEY = f"{TEST_PREFIX}/{runner.RESULT_KEY}"
PARTIAL_FULL_KEY = f"{TEST_PREFIX}/{runner.PARTIAL_KEY}"
STATUS_FULL_KEY = f"{TEST_PREFIX}/{runner.STATUS_KEY}"

CSV_HEADER = "mcs,cell_id,type,volume,surface,x,y,z\n"
CSV_ROWS = ("1,1,Tumour,25,20,10.0,12.0,0\n"
            "1,2,Tumour,24,19,14.0,11.0,0\n"
            "2,1,Tumour,26,21,10.5,12.4,0\n")


# ---------------------------------------------------------------------------
# Base case
# ---------------------------------------------------------------------------
class RunnerTestCase(unittest.TestCase):
    """Resets the runner's module-level shutdown state and installs fake clients."""

    def setUp(self):
        self.s3 = FakeS3("s3")
        self.urgent = FakeS3("urgent_s3")
        runner.s3 = self.s3
        runner.urgent_s3 = self.urgent
        self.logs = []
        patcher = patch.object(runner, "log", self.logs.append)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.reset_shutdown()
        self.addCleanup(self.reset_shutdown)
        for name in ("CC3D_STEPS", "CC3D_MAX_MINUTES", "CC3D_SAVE_EVERY",
                     "CC3D_RUNSCRIPT"):
            self.addCleanup(os.environ.pop, name, None)
            os.environ.pop(name, None)

    def reset_shutdown(self):
        if runner._shutdown_claim.locked():
            try:
                runner._shutdown_claim.release()
            except RuntimeError:
                pass
        runner._shutdown_requested = False
        runner._shutdown_reason = ""
        runner._shutdown_state = "interrupted"
        runner._shutdown_exit_code = runner.EXIT_INTERRUPTED
        runner._shutdown_at = 0.0
        runner._child = None
        runner._job_started = time.monotonic()

    # -- shared helpers ----------------------------------------------------
    def workdir(self):
        path = tempfile.mkdtemp(prefix="cc3d_test_")
        self.addCleanup(_rmtree, path)
        return path

    def write_result(self, workdir, text, subdir=""):
        target = os.path.join(workdir, subdir) if subdir else workdir
        os.makedirs(target, exist_ok=True)
        path = os.path.join(target, runner.RESULT_KEY)
        with open(path, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
        return path

    def all_write_calls(self):
        calls = []
        for client in (self.s3, self.urgent):
            for op, kwargs in client.calls:
                if op in ("put_object", "upload_file"):
                    calls.append((client.name, op, kwargs))
        return calls

    def assertResultKeyNeverWritten(self, note=""):
        for name, op, kwargs in self.all_write_calls():
            self.assertNotEqual(
                kwargs.get("Key"), RESULT_FULL_KEY,
                f"{name}.{op} wrote partial data to the COMPLETE-result key "
                f"{RESULT_FULL_KEY}. {note}")

    def status_payloads(self):
        """Every status.json write, from BOTH clients, in the order they happened.

        finalize_shutdown writes its first status through the urgent client and its
        failure-path status through the normal one, so merging by client rather than
        by time would report the wrong "last" status.
        """
        writes = self.s3.status_writes() + self.urgent.status_writes()
        writes.sort(key=lambda item: item[0])
        return [payload for _, payload in writes]

    def assertStatusWritten(self, state=None):
        payloads = self.status_payloads()
        self.assertTrue(payloads, "no status.json was written on this exit path")
        if state is not None:
            self.assertEqual(payloads[-1]["state"], state)
        return payloads


def _rmtree(path):
    import shutil
    shutil.rmtree(path, ignore_errors=True)


# ---------------------------------------------------------------------------
# 1. THE CRITICAL PROPERTY: partial data can never reach cells.csv
# ---------------------------------------------------------------------------
class TestResultKeyIsSacred(RunnerTestCase):
    """``cells.csv`` means "complete". Partial data reaching it is silent corruption."""

    def test_partial_and_result_keys_are_distinct(self):
        self.assertNotEqual(runner.PARTIAL_KEY, runner.RESULT_KEY)
        self.assertFalse(runner.PARTIAL_KEY.endswith(f"/{runner.RESULT_KEY}"))
        self.assertNotEqual(PARTIAL_FULL_KEY, RESULT_FULL_KEY)

    def test_no_interrupted_scenario_ever_writes_the_result_key(self):
        """Ten shapes of interruption; none may target cells.csv."""
        big = CSV_HEADER + CSV_ROWS * 40
        scenarios = [
            ("complete rows", CSV_HEADER + CSV_ROWS, {}),
            ("torn final row", CSV_HEADER + CSV_ROWS + "3,1,Tumour,26,2", {}),
            ("no newline at all", "mcs,cell_id,type", {}),
            ("header only", CSV_HEADER, {}),
            ("over the byte cap", big, {"cap": 120}),
            ("nested output dir", CSV_HEADER + CSV_ROWS, {"subdir": "Simulation"}),
            ("expired shutdown window", CSV_HEADER + CSV_ROWS, {"age": 30.0}),
            ("s3 refuses everything", CSV_HEADER + CSV_ROWS, {"fail": True}),
            ("time-cap shutdown", CSV_HEADER + CSV_ROWS,
             {"state": "failed", "code": runner.EXIT_TIME_LIMIT}),
            ("nothing written yet", None, {}),
        ]
        for label, content, opts in scenarios:
            with self.subTest(scenario=label):
                self.setUp()  # fresh clients and shutdown state per scenario
                workdir = self.workdir()
                if content is not None:
                    self.write_result(workdir, content, opts.get("subdir", ""))
                if opts.get("fail"):
                    self.urgent.fail = {"put_object"}
                    self.s3.fail = {"put_object"}
                runner.request_shutdown(runner.SPOT_SIGTERM_REASON,
                                        state=opts.get("state", "interrupted"),
                                        exit_code=opts.get("code",
                                                           runner.EXIT_INTERRUPTED))
                runner._shutdown_at = time.monotonic() - opts.get("age", 0.0)
                cap = opts.get("cap", runner.MAX_PARTIAL_BYTES)
                with patch.object(runner, "MAX_PARTIAL_BYTES", cap):
                    code = runner.finalize_shutdown(workdir)
                self.assertEqual(code, opts.get("code", runner.EXIT_INTERRUPTED))
                self.assertResultKeyNeverWritten(f"scenario: {label}")

    def test_the_partial_body_lands_on_the_partial_key_verbatim(self):
        workdir = self.workdir()
        self.write_result(workdir, CSV_HEADER + CSV_ROWS)
        runner.request_shutdown(runner.SPOT_SIGTERM_REASON)
        runner._shutdown_at = time.monotonic()

        runner.finalize_shutdown(workdir)

        uploads = [kw for kw in self.urgent.ops("put_object")
                   if kw["Key"] == PARTIAL_FULL_KEY]
        self.assertEqual(len(uploads), 1, "the partial should be uploaded exactly once")
        self.assertEqual(uploads[0]["Body"], (CSV_HEADER + CSV_ROWS).encode())
        self.assertEqual(uploads[0]["ContentType"], "text/csv")
        self.assertResultKeyNeverWritten()

    def test_a_csv_body_is_never_sent_to_any_key_but_the_partial_key(self):
        workdir = self.workdir()
        self.write_result(workdir, CSV_HEADER + CSV_ROWS)
        runner.request_shutdown(runner.SPOT_SIGTERM_REASON)
        runner._shutdown_at = time.monotonic()

        runner.finalize_shutdown(workdir)

        for name, op, kwargs in self.all_write_calls():
            body = kwargs.get("Body") or b""
            if isinstance(body, bytes) and b",Tumour," in body:
                self.assertEqual(kwargs["Key"], PARTIAL_FULL_KEY,
                                 f"{name}.{op} sent measurement rows to "
                                 f"{kwargs['Key']}")

    def test_upload_partial_reports_absence_rather_than_uploading_nothing(self):
        workdir = self.workdir()
        detail = runner._upload_partial(workdir, time.monotonic() + 60)
        self.assertIsNone(detail["partial"])
        self.assertIn("no partial file", detail["partial_note"])
        self.assertEqual(self.urgent.written_keys(), [])

    def test_upload_partial_refuses_a_body_with_no_complete_row(self):
        workdir = self.workdir()
        self.write_result(workdir, "mcs,cell_id,type,volume")  # no newline
        detail = runner._upload_partial(workdir, time.monotonic() + 60)
        self.assertIsNone(detail["partial"])
        self.assertIn("no complete row", detail["partial_note"])
        self.assertEqual(self.urgent.written_keys(), [])

    def test_upload_partial_picks_the_largest_candidate_and_records_truncation(self):
        workdir = self.workdir()
        self.write_result(workdir, CSV_HEADER)
        self.write_result(workdir, CSV_HEADER + CSV_ROWS * 20, subdir="Simulation")
        with patch.object(runner, "MAX_PARTIAL_BYTES", 200):
            detail = runner._upload_partial(workdir, time.monotonic() + 60)
        self.assertEqual(detail["partial"], runner.PARTIAL_KEY)
        self.assertTrue(detail["partial_truncated"])
        self.assertLessEqual(detail["partial_bytes"], 200)
        body = self.urgent.objects[PARTIAL_FULL_KEY]
        self.assertTrue(body.endswith(b"\n"), "a truncated partial must end on a row")

    def test_main_interrupted_run_uploads_no_result_even_with_a_full_csv_on_disk(self):
        """The engine can leave a complete-LOOKING file; only success may ship it."""
        def fake_engine(model, workdir, max_minutes):
            self.write_result(workdir, CSV_HEADER + CSV_ROWS)
            runner.request_shutdown(runner.SPOT_SIGTERM_REASON)
            raise runner.Interrupted(runner.SPOT_SIGTERM_REASON)

        code = self.run_main(engine=fake_engine)
        self.assertEqual(code, runner.EXIT_INTERRUPTED)
        self.assertIn(PARTIAL_FULL_KEY, self.urgent.written_keys())
        self.assertResultKeyNeverWritten()

    def test_main_time_capped_run_uploads_no_result(self):
        def fake_engine(model, workdir, max_minutes):
            self.write_result(workdir, CSV_HEADER + CSV_ROWS)
            runner.request_shutdown(
                runner.TIME_CAP_REASON.format(minutes=max_minutes),
                state="failed", exit_code=runner.EXIT_TIME_LIMIT)
            return 3

        code = self.run_main(engine=fake_engine)
        self.assertEqual(code, runner.EXIT_TIME_LIMIT)
        self.assertResultKeyNeverWritten()
        self.assertIn(PARTIAL_FULL_KEY, self.urgent.written_keys())

    def test_main_success_is_the_only_path_that_writes_the_result_key(self):
        def fake_engine(model, workdir, max_minutes):
            self.write_result(workdir, CSV_HEADER + CSV_ROWS)
            return 3

        code = self.run_main(engine=fake_engine)
        self.assertEqual(code, runner.EXIT_OK)
        self.assertIn(RESULT_FULL_KEY, self.s3.written_keys())
        self.assertEqual(self.s3.objects[RESULT_FULL_KEY],
                         (CSV_HEADER + CSV_ROWS).encode())
        self.assertNotIn(PARTIAL_FULL_KEY, self.s3.written_keys())
        deletes = [kw["Key"] for kw in self.s3.ops("delete_object")]
        self.assertIn(PARTIAL_FULL_KEY, deletes,
                      "a stale partial must be removed once a real result exists")

    # -- shared main() driver ---------------------------------------------
    def run_main(self, engine, download=None, no_model=False):
        return _run_main(self, engine, download=download, no_model=no_model)


def _run_main(case, engine, download=None, no_model=False, patch_install=True):
    """Drive main() with the network, the watcher and the engine stubbed out."""
    def default_download(workdir):
        if not no_model:
            os.makedirs(os.path.join(workdir, "Simulation"), exist_ok=True)
            with open(os.path.join(workdir, "Simulation", "model.xml"), "w") as handle:
                handle.write("<CompuCell3D><Potts><Steps>5</Steps></Potts></CompuCell3D>")
        return ["Simulation/model.xml"]

    patches = [
        patch.object(runner, "_watch_for_spot_interruption", lambda stop: None),
        patch.object(runner, "download_project", download or default_download),
    ]
    if engine is not None:
        patches.append(patch.object(runner, "run_engine_python_api", engine))
    if patch_install:
        patches.append(patch.object(runner, "install_signal_handlers", lambda: True))
    for item in patches:
        item.start()
        case.addCleanup(item.stop)
    return runner.main()


# ---------------------------------------------------------------------------
# 2. request_shutdown atomicity
# ---------------------------------------------------------------------------
class TestRequestShutdownAtomicity(RunnerTestCase):
    """Three writers race here in production: the handler, the watcher, the cap."""

    def test_first_caller_wins_and_records_everything(self):
        self.assertTrue(runner.request_shutdown("first", state="interrupted",
                                                exit_code=runner.EXIT_INTERRUPTED))
        self.assertTrue(runner._shutdown_requested)
        self.assertEqual(runner._shutdown_reason, "first")
        self.assertEqual(runner._shutdown_state, "interrupted")
        self.assertEqual(runner._shutdown_exit_code, runner.EXIT_INTERRUPTED)
        self.assertGreater(runner._shutdown_at, 0.0)

    def test_second_caller_cannot_overwrite_any_field(self):
        runner.request_shutdown(runner.SPOT_SIGTERM_REASON)
        at = runner._shutdown_at
        self.assertFalse(runner.request_shutdown(
            runner.TIME_CAP_REASON, state="failed",
            exit_code=runner.EXIT_TIME_LIMIT))
        self.assertEqual(runner._shutdown_reason, runner.SPOT_SIGTERM_REASON)
        self.assertEqual(runner._shutdown_state, "interrupted")
        self.assertEqual(runner._shutdown_exit_code, runner.EXIT_INTERRUPTED)
        self.assertEqual(runner._shutdown_at, at)

    def test_a_lost_race_never_mixes_one_callers_state_with_anothers_reason(self):
        """The exact bug the non-blocking lock was added for."""
        triples = [
            (runner.SPOT_SIGTERM_REASON, "interrupted", runner.EXIT_INTERRUPTED),
            (runner.SPOT_NOTICE_REASON.format(action="terminate", when="soon"),
             "interrupted", runner.EXIT_INTERRUPTED),
            (runner.TIME_CAP_REASON.format(minutes=90), "failed",
             runner.EXIT_TIME_LIMIT),
            (runner.LOCAL_STOP_REASON, "interrupted", runner.EXIT_INTERRUPTED),
        ] * 2
        for round_index in range(40):
            self.reset_shutdown()
            barrier = threading.Barrier(len(triples))
            winners = []
            lock = threading.Lock()

            def contend(triple):
                barrier.wait()
                won = runner.request_shutdown(triple[0], state=triple[1],
                                              exit_code=triple[2])
                if won:
                    with lock:
                        winners.append(triple)

            threads = [threading.Thread(target=contend, args=(t,)) for t in triples]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=5)
                self.assertFalse(thread.is_alive(), "request_shutdown blocked a caller")

            self.assertEqual(len(winners), 1,
                             f"round {round_index}: exactly one caller must win, "
                             f"got {len(winners)}")
            reason, state, code = winners[0]
            record = (runner._shutdown_reason, runner._shutdown_state,
                      runner._shutdown_exit_code)
            self.assertEqual(record, (reason, state, code),
                             f"round {round_index}: the record does not match the "
                             f"winning caller -- fields were interleaved")

    def test_the_flag_is_never_visible_before_the_fields_it_describes(self):
        """_checkpoint reads the flag WITHOUT the lock, so ordering is load-bearing."""
        for _ in range(150):
            self.reset_shutdown()
            seen = []
            go = threading.Event()

            def observe():
                go.set()
                while True:
                    if runner._shutdown_requested:
                        seen.append((runner._shutdown_reason,
                                     runner._shutdown_state,
                                     runner._shutdown_exit_code,
                                     runner._shutdown_at))
                        return

            watcher = threading.Thread(target=observe)
            watcher.start()
            go.wait(1)
            runner.request_shutdown(runner.SPOT_SIGTERM_REASON)
            watcher.join(timeout=5)
            self.assertFalse(watcher.is_alive())
            self.assertEqual(len(seen), 1)
            reason, state, code, at = seen[0]
            self.assertEqual(reason, runner.SPOT_SIGTERM_REASON,
                             "the flag became true before the reason was written")
            self.assertEqual(state, "interrupted")
            self.assertEqual(code, runner.EXIT_INTERRUPTED)
            self.assertGreater(at, 0.0)

    def test_a_held_claim_makes_the_call_return_false_immediately(self):
        """A signal handler must never park on a lock the main thread holds."""
        runner._shutdown_claim.acquire()
        try:
            started = time.monotonic()
            self.assertFalse(runner.request_shutdown("from a signal handler"))
            self.assertLess(time.monotonic() - started, 0.5,
                            "request_shutdown blocked instead of failing fast")
        finally:
            runner._shutdown_claim.release()
        self.assertFalse(runner._shutdown_requested)
        self.assertEqual(runner._shutdown_reason, "")

    def test_checkpoint_raises_only_after_a_shutdown_is_recorded(self):
        runner._checkpoint()  # no shutdown: must not raise
        runner.request_shutdown(runner.SPOT_SIGTERM_REASON)
        with self.assertRaises(runner.Interrupted) as caught:
            runner._checkpoint()
        self.assertEqual(str(caught.exception), runner.SPOT_SIGTERM_REASON)


# ---------------------------------------------------------------------------
# 3. status.json on every reachable exit path
# ---------------------------------------------------------------------------
class TestStatusOnEveryExitPath(RunnerTestCase):

    def test_put_status_writes_a_parseable_payload_to_the_status_key(self):
        self.assertTrue(runner.put_status("running", engine="stub", rows=3))
        call = self.s3.ops("put_object")[-1]
        self.assertEqual(call["Key"], STATUS_FULL_KEY)
        self.assertEqual(call["ContentType"], "application/json")
        payload = json.loads(call["Body"].decode())
        self.assertEqual(payload["state"], "running")
        self.assertEqual(payload["run_id"], TEST_RUN_ID)
        self.assertEqual(payload["engine"], "stub")
        self.assertIn("at", payload)

    def test_put_status_never_propagates_an_s3_failure(self):
        self.s3.fail = {"put_object"}
        self.assertFalse(runner.put_status("failed", error="x"))
        self.assertTrue(any("could not write status.json" in line
                            for line in self.logs))

    def test_interrupted_exit_writes_status(self):
        def engine(model, workdir, max_minutes):
            runner.request_shutdown(runner.SPOT_SIGTERM_REASON)
            raise runner.Interrupted("stop")

        code = _run_main(self, engine)
        self.assertEqual(code, runner.EXIT_INTERRUPTED)
        payloads = self.assertStatusWritten("interrupted")
        self.assertIn("SIGTERM", payloads[-1]["error"])
        self.assertTrue(payloads[-1]["retryable"])
        self.assertTrue(payloads[-1]["restarts_from_step_zero"])

    def test_time_capped_exit_writes_status(self):
        def engine(model, workdir, max_minutes):
            runner.request_shutdown(
                runner.TIME_CAP_REASON.format(minutes=max_minutes),
                state="failed", exit_code=runner.EXIT_TIME_LIMIT)
            return 2

        code = _run_main(self, engine)
        self.assertEqual(code, runner.EXIT_TIME_LIMIT)
        self.assertStatusWritten("failed")

    def test_no_model_exit_writes_status(self):
        code = _run_main(self, engine=None, no_model=True)
        self.assertEqual(code, runner.EXIT_NO_MODEL)
        payloads = self.assertStatusWritten("failed")
        self.assertIn("model.xml", payloads[-1]["error"])

    def test_empty_result_exit_writes_status(self):
        code = _run_main(self, engine=lambda model, workdir, minutes: 0)
        self.assertEqual(code, runner.EXIT_EMPTY_RESULT)
        payloads = self.assertStatusWritten("failed")
        self.assertIn("no cell measurements", payloads[-1]["error"])

    def test_engine_failure_exit_writes_status_with_a_traceback(self):
        def engine(model, workdir, max_minutes):
            raise RuntimeError("cc3d blew up")

        code = _run_main(self, engine)
        self.assertEqual(code, runner.EXIT_ENGINE_FAILED)
        payloads = self.assertStatusWritten("failed")
        self.assertIn("cc3d blew up", payloads[-1]["error"])
        self.assertIn("traceback", payloads[-1])

    def test_unhandled_exception_exit_writes_status(self):
        def download(workdir):
            raise ValueError("something nobody anticipated")

        code = _run_main(self, engine=None, download=download)
        self.assertEqual(code, runner.EXIT_UNHANDLED)
        payloads = self.assertStatusWritten("failed")
        self.assertIn("something nobody anticipated", payloads[-1]["error"])
        self.assertIn("traceback", payloads[-1])

    def test_no_runscript_exit_writes_status(self):
        with patch.dict(sys.modules, {"cc3d": None}), \
                patch.object(runner, "find_run_script", lambda: None):
            code = _run_main(self, engine=None)
        self.assertEqual(code, runner.EXIT_NO_RUNSCRIPT)
        payloads = self.assertStatusWritten("failed")
        self.assertIn("runScript", payloads[-1]["error"])

    def test_refusing_to_start_without_a_sigterm_handler_writes_status(self):
        with patch.object(runner, "install_signal_handlers", lambda: False):
            code = _run_main(self, engine=None, patch_install=False)
        self.assertEqual(code, runner.EXIT_MISCONFIGURED)
        payloads = self.assertStatusWritten("failed")
        self.assertIn("SIGTERM", payloads[-1]["error"])

    def test_missing_bucket_exits_misconfigured_without_touching_s3(self):
        """No bucket means no place to write status; the exit code is all there is."""
        with patch.object(runner, "BUCKET", ""):
            code = runner.main()
        self.assertEqual(code, runner.EXIT_MISCONFIGURED)
        self.assertEqual(self.s3.calls, [])
        self.assertEqual(self.urgent.calls, [])

    def test_success_exit_writes_status(self):
        def engine(model, workdir, max_minutes):
            path = os.path.join(workdir, runner.RESULT_KEY)
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(CSV_HEADER + CSV_ROWS)
            return 3

        code = _run_main(self, engine)
        self.assertEqual(code, runner.EXIT_OK)
        payloads = self.assertStatusWritten("succeeded")
        self.assertEqual(payloads[-1]["rows"], 3)
        self.assertGreater(payloads[-1]["bytes"], 0)

    def test_a_status_is_written_before_the_partial_upload_is_attempted(self):
        """If the SIGKILL lands mid-upload the researcher still gets the reason."""
        workdir = self.workdir()
        self.write_result(workdir, CSV_HEADER + CSV_ROWS)
        runner.request_shutdown(runner.SPOT_SIGTERM_REASON)
        runner._shutdown_at = time.monotonic()

        runner.finalize_shutdown(workdir)

        keys = [kw["Key"] for kw in self.urgent.ops("put_object")]
        self.assertEqual(keys[0], STATUS_FULL_KEY)
        self.assertIn(PARTIAL_FULL_KEY, keys)
        self.assertLess(keys.index(STATUS_FULL_KEY), keys.index(PARTIAL_FULL_KEY))

    def test_the_final_status_never_still_claims_the_partial_is_pending(self):
        """A spent shutdown window must not freeze status at partial="pending".

        finalize_shutdown writes partial="pending" first and then amends it. The amend
        used to be SKIPPED once the budget was spent ("the first write stands"), so the
        last status a researcher saw claimed the upload was still in flight -- for ever,
        and the application has no handling for a perpetual "pending". It is also simply
        untrue the moment finalize_shutdown returns: nothing is pending any more, the
        upload either happened or was abandoned.

        FIXED: the amend is now unconditional. It is a few hundred bytes, so it is
        affordable even with almost no window left, and a genuine failure to write it is
        logged rather than leaving the misleading value in place.
        """
        workdir = self.workdir()
        self.write_result(workdir, CSV_HEADER + CSV_ROWS)
        runner.request_shutdown(runner.SPOT_SIGTERM_REASON)
        runner._shutdown_at = time.monotonic() - (runner.SHUTDOWN_BUDGET_SECS + 10)

        code = runner.finalize_shutdown(workdir)

        self.assertEqual(code, runner.EXIT_INTERRUPTED)
        payloads = self.status_payloads()
        self.assertGreaterEqual(
            len(payloads), 2,
            "the provisional status must be followed by an amend carrying the real "
            "explanation, even when the shutdown window is already spent")
        final = payloads[-1]
        self.assertNotEqual(
            final.get("partial"), "pending",
            "the last status.json still reports partial='pending' after "
            "finalize_shutdown returned, so the application can never learn that "
            "the partial upload was skipped")


# ---------------------------------------------------------------------------
# 4. _read_complete_lines
# ---------------------------------------------------------------------------
class TestReadCompleteLines(RunnerTestCase):

    def read(self, text, limit=1024):
        path = os.path.join(self.workdir(), "cells.csv")
        with open(path, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
        return runner._read_complete_lines(path, limit)

    def test_whole_lines_are_returned_intact(self):
        body, lines, truncated = self.read(CSV_HEADER + CSV_ROWS)
        self.assertEqual(body, (CSV_HEADER + CSV_ROWS).encode())
        self.assertEqual(lines, 4)
        self.assertFalse(truncated)

    def test_a_torn_final_line_is_discarded(self):
        body, lines, truncated = self.read(CSV_HEADER + CSV_ROWS + "3,2,Tumour,26,2")
        self.assertEqual(body, (CSV_HEADER + CSV_ROWS).encode())
        self.assertEqual(lines, 4)
        self.assertFalse(truncated)
        self.assertTrue(body.endswith(b"\n"))

    def test_a_file_with_no_newline_yields_nothing(self):
        body, lines, truncated = self.read("mcs,cell_id,type,volume")
        self.assertEqual(body, b"")
        self.assertEqual(lines, 0)
        self.assertFalse(truncated)

    def test_an_empty_file_yields_nothing(self):
        body, lines, truncated = self.read("")
        self.assertEqual(body, b"")
        self.assertEqual(lines, 0)
        self.assertFalse(truncated)

    def test_the_byte_cap_is_honoured_and_still_cuts_on_a_row_boundary(self):
        text = "".join(f"{i:08d}\n" for i in range(50))  # 9 bytes per line
        body, lines, truncated = self.read(text, limit=20)
        self.assertTrue(truncated)
        self.assertLessEqual(len(body), 20)
        self.assertEqual(body, b"00000000\n00000001\n")
        self.assertEqual(lines, 2)
        self.assertTrue(body.endswith(b"\n"))

    def test_a_cap_landing_exactly_on_a_newline_keeps_the_whole_row(self):
        text = "".join(f"{i:08d}\n" for i in range(50))
        body, lines, truncated = self.read(text, limit=18)
        self.assertEqual(body, b"00000000\n00000001\n")
        self.assertEqual(lines, 2)
        self.assertTrue(truncated)

    def test_a_file_exactly_at_the_cap_is_not_reported_as_truncated(self):
        text = "".join(f"{i:08d}\n" for i in range(4))  # 36 bytes
        body, lines, truncated = self.read(text, limit=36)
        self.assertFalse(truncated)
        self.assertEqual(lines, 4)
        self.assertEqual(body, text.encode())

    def test_every_returned_line_is_complete_across_many_caps(self):
        text = CSV_HEADER + CSV_ROWS * 12
        for limit in range(1, len(text) + 4):
            with self.subTest(limit=limit):
                body, lines, _ = self.read(text, limit=limit)
                if body:
                    self.assertTrue(body.endswith(b"\n"))
                    self.assertEqual(lines, body.count(b"\n"))
                    self.assertTrue(text.encode().startswith(body))
                    for line in body.splitlines():
                        self.assertEqual(line.count(b","), 7,
                                         "a row was shipped with missing columns")
                else:
                    self.assertEqual(lines, 0)

    def test_crlf_rows_survive_intact(self):
        body, lines, _ = self.read("a,1\r\nb,2\r\nc,3")
        self.assertEqual(body, b"a,1\r\nb,2\r\n")
        self.assertEqual(lines, 2)


# ---------------------------------------------------------------------------
# 5. The IMDSv2 interruption-notice poll
# ---------------------------------------------------------------------------
class TestImdsPolling(RunnerTestCase):

    def test_a_404_is_normal_and_is_never_logged(self):
        with patch("urllib.request.urlopen", side_effect=http_error(404)):
            self.assertIsNone(runner._spot_instance_action("tok"))
        self.assertEqual(self.logs, [], "a 404 from IMDS must not be logged")

    def test_a_401_asks_the_caller_for_a_new_token(self):
        with patch("urllib.request.urlopen", side_effect=http_error(401)):
            with self.assertRaises(runner._TokenExpired):
                runner._spot_instance_action("stale")
        self.assertEqual(self.logs, [])

    def test_a_403_is_absence_of_a_notice_not_an_error(self):
        with patch("urllib.request.urlopen", side_effect=http_error(403)):
            self.assertIsNone(runner._spot_instance_action("tok"))
        self.assertEqual(self.logs, [])

    def test_an_unreachable_endpoint_is_absence_of_a_notice(self):
        with patch("urllib.request.urlopen",
                   side_effect=urllib.error.URLError("connection refused")):
            self.assertIsNone(runner._spot_instance_action("tok"))
        self.assertEqual(self.logs, [])

    def test_a_valid_notice_is_returned_verbatim(self):
        payload = json.dumps({"action": "terminate",
                              "time": "2026-09-13T20:11:00Z"}).encode()
        with patch("urllib.request.urlopen", return_value=FakeResponse(payload)):
            notice = runner._spot_instance_action("tok")
        self.assertEqual(notice["action"], "terminate")
        self.assertEqual(notice["time"], "2026-09-13T20:11:00Z")

    def test_a_200_with_an_unparseable_body_is_still_treated_as_a_notice(self):
        """Reclamation is imminent; discarding the notice would lose the run."""
        with patch("urllib.request.urlopen",
                   return_value=FakeResponse(b"<html>gateway junk</html>")):
            notice = runner._spot_instance_action("tok")
        self.assertIsInstance(notice, dict)
        self.assertEqual(notice["action"], "terminate")
        self.assertIn("gateway junk", notice["time"])

    def test_a_200_with_json_that_is_not_an_object_is_still_a_notice(self):
        for body in (b"null", b"[]", b"42", b'"terminate"'):
            with self.subTest(body=body):
                with patch("urllib.request.urlopen", return_value=FakeResponse(body)):
                    notice = runner._spot_instance_action("tok")
                self.assertIsInstance(notice, dict)
                self.assertEqual(notice["action"], "terminate")

    def test_an_empty_200_body_is_a_notice_with_an_unknown_time(self):
        with patch("urllib.request.urlopen", return_value=FakeResponse(b"")):
            notice = runner._spot_instance_action("tok")
        self.assertEqual(notice, {"action": "terminate", "time": "unknown"})

    def test_an_oversized_body_is_clipped_before_it_reaches_status_json(self):
        with patch("urllib.request.urlopen", return_value=FakeResponse(b"x" * 500)):
            notice = runner._spot_instance_action("tok")
        self.assertEqual(len(notice["time"]), 64)

    def test_the_action_request_carries_the_imdsv2_token_header(self):
        captured = {}

        def fake_urlopen(request, timeout=None):
            captured["url"] = request.full_url
            captured["headers"] = dict(request.headers)
            captured["timeout"] = timeout
            return FakeResponse(b"{}")

        with patch("urllib.request.urlopen", fake_urlopen):
            runner._spot_instance_action("tok-123")
        self.assertEqual(captured["url"], runner.IMDS_ACTION_URL)
        self.assertEqual(captured["headers"].get("X-aws-ec2-metadata-token"), "tok-123")
        self.assertEqual(captured["timeout"], runner.IMDS_TIMEOUT_SECS)

    def test_the_token_request_is_a_put_with_a_ttl_header(self):
        captured = {}

        def fake_urlopen(request, timeout=None):
            captured["url"] = request.full_url
            captured["method"] = request.get_method()
            captured["headers"] = dict(request.headers)
            return FakeResponse(b"  a-token \n")

        with patch("urllib.request.urlopen", fake_urlopen):
            token = runner._imds_token()
        self.assertEqual(token, "a-token")
        self.assertEqual(captured["url"], runner.IMDS_TOKEN_URL)
        self.assertEqual(captured["method"], "PUT")
        self.assertEqual(
            captured["headers"].get("X-aws-ec2-metadata-token-ttl-seconds"),
            str(runner.IMDS_TOKEN_TTL_SECS))

    # -- the watcher loop --------------------------------------------------
    def test_the_watcher_gives_up_silently_when_imds_is_unreachable(self):
        attempts = []

        def token():
            attempts.append(1)
            raise ConnectionRefusedError("no metadata service here")

        stop = ScriptedStopEvent(20)
        with patch.object(runner, "_imds_token", token), \
                patch.object(runner, "IMDS_POLL_SECS", 0.0):
            runner._watch_for_spot_interruption(stop)

        self.assertEqual(len(attempts), runner.IMDS_MAX_SETUP_FAILURES)
        self.assertGreater(stop.remaining, 0, "the watcher should return early")
        self.assertEqual(self.logs, [],
                         "a local run without IMDS is normal and must be silent")
        self.assertFalse(runner._shutdown_requested)

    def test_the_watcher_stays_silent_while_imds_answers_404(self):
        stop = ScriptedStopEvent(5)
        calls = []
        with patch.object(runner, "_imds_token", lambda: "tok"), \
                patch.object(runner, "_spot_instance_action",
                             lambda token: calls.append(token) or None), \
                patch.object(runner, "IMDS_POLL_SECS", 0.0):
            runner._watch_for_spot_interruption(stop)
        self.assertEqual(len(calls), 5)
        self.assertEqual(self.logs, [])
        self.assertFalse(runner._shutdown_requested)

    def test_a_401_makes_the_watcher_fetch_a_fresh_token(self):
        tokens = ["tok-1", "tok-2"]
        handed = []
        seen = []

        def token():
            handed.append(tokens[len(handed)] if len(handed) < len(tokens) else "tok-x")
            return handed[-1]

        def action(tok):
            seen.append(tok)
            if len(seen) == 1:
                raise runner._TokenExpired()
            return {"action": "terminate", "time": "2026-09-13T20:15:00Z"}

        signalled = []
        stop = ScriptedStopEvent(10)
        with patch.object(runner, "_imds_token", token), \
                patch.object(runner, "_spot_instance_action", action), \
                patch.object(runner, "_signal_child", signalled.append), \
                patch.object(runner, "IMDS_POLL_SECS", 0.0):
            runner._watch_for_spot_interruption(stop)

        self.assertEqual(handed, ["tok-1", "tok-2"], "the token was not refreshed")
        self.assertEqual(seen, ["tok-1", "tok-2"])
        self.assertTrue(runner._shutdown_requested)
        self.assertEqual(signalled, [signal.SIGTERM])

    def test_a_notice_requests_shutdown_with_the_notice_reason(self):
        signalled = []
        stop = ScriptedStopEvent(4)
        notice = {"action": "stop", "time": "2026-09-13T20:20:00Z"}
        with patch.object(runner, "_imds_token", lambda: "tok"), \
                patch.object(runner, "_spot_instance_action", lambda t: notice), \
                patch.object(runner, "_signal_child", signalled.append), \
                patch.object(runner, "IMDS_POLL_SECS", 0.0):
            runner._watch_for_spot_interruption(stop)

        self.assertTrue(runner._shutdown_requested)
        self.assertEqual(runner._shutdown_state, "interrupted")
        self.assertEqual(runner._shutdown_exit_code, runner.EXIT_INTERRUPTED)
        self.assertIn("action=stop", runner._shutdown_reason)
        self.assertIn("2026-09-13T20:20:00Z", runner._shutdown_reason)
        self.assertNotIn("{action}", runner._shutdown_reason)
        self.assertNotIn("{when}", runner._shutdown_reason)
        self.assertEqual(signalled, [signal.SIGTERM])
        self.assertEqual(len(self.logs), 1)
        self.assertIn("spot interruption notice", self.logs[0])
        self.assertGreater(stop.remaining, 0, "the watcher must return after a notice")

    def test_a_notice_with_no_fields_still_produces_a_usable_reason(self):
        stop = ScriptedStopEvent(3)
        with patch.object(runner, "_imds_token", lambda: "tok"), \
                patch.object(runner, "_spot_instance_action", lambda t: {}), \
                patch.object(runner, "_signal_child", lambda sig: None), \
                patch.object(runner, "IMDS_POLL_SECS", 0.0):
            runner._watch_for_spot_interruption(stop)
        self.assertTrue(runner._shutdown_requested)
        self.assertIn("action=terminate", runner._shutdown_reason)
        self.assertIn("unknown", runner._shutdown_reason)

    def test_the_watcher_stands_down_when_a_shutdown_is_already_under_way(self):
        runner.request_shutdown(runner.SPOT_SIGTERM_REASON)
        calls = []
        stop = ScriptedStopEvent(5)
        with patch.object(runner, "_imds_token",
                          lambda: calls.append("token") or "tok"), \
                patch.object(runner, "IMDS_POLL_SECS", 0.0):
            runner._watch_for_spot_interruption(stop)
        self.assertEqual(calls, [], "the watcher must not poll during a shutdown")
        self.assertEqual(runner._shutdown_reason, runner.SPOT_SIGTERM_REASON)

    def test_the_notice_reason_earns_the_larger_shutdown_budget(self):
        """The ~2 minute notice window is what ships a large, nearly-complete partial.

        Both cases are 30 seconds past _shutdown_at: the notice path (90s budget)
        must still upload, the bare-SIGTERM path (20s budget) must skip.
        """
        for label, reason, expect_upload in (
            ("spot notice", runner.SPOT_NOTICE_REASON.format(action="terminate",
                                                             when="soon"), True),
            ("bare SIGTERM", runner.SPOT_SIGTERM_REASON, False),
        ):
            with self.subTest(path=label):
                self.setUp()
                workdir = self.workdir()
                self.write_result(workdir, CSV_HEADER + CSV_ROWS)
                runner.request_shutdown(reason)
                runner._shutdown_at = time.monotonic() - 30.0
                runner.finalize_shutdown(workdir)
                uploaded = PARTIAL_FULL_KEY in self.urgent.written_keys()
                self.assertEqual(uploaded, expect_upload)
                self.assertResultKeyNeverWritten()


# ---------------------------------------------------------------------------
# 6. finalize_shutdown must not turn an interruption into a crash
# ---------------------------------------------------------------------------
class TestFinalizeShutdownSurvivesFailures(RunnerTestCase):

    def prime(self, state="interrupted", code=None, age=0.0):
        workdir = self.workdir()
        self.write_result(workdir, CSV_HEADER + CSV_ROWS)
        runner.request_shutdown(runner.SPOT_SIGTERM_REASON, state=state,
                                exit_code=runner.EXIT_INTERRUPTED if code is None
                                else code)
        runner._shutdown_at = time.monotonic() - age
        return workdir

    def test_the_exit_code_survives_the_partial_upload_raising(self):
        workdir = self.prime()
        self.urgent.fail = {"put_object"}
        self.assertEqual(runner.finalize_shutdown(workdir), runner.EXIT_INTERRUPTED)
        self.assertTrue(any(runner.PARTIAL_KEY in kw["Key"]
                            for kw in self.urgent.ops("put_object")))

    def test_the_exit_code_survives_an_oserror_while_choosing_a_candidate(self):
        """The real bug: an OSError here used to escape as an unhandled exit 1."""
        workdir = self.prime()
        with patch("os.path.getsize", side_effect=OSError("file vanished")):
            code = runner.finalize_shutdown(workdir)
        self.assertEqual(code, runner.EXIT_INTERRUPTED)
        self.assertTrue(any("finalize_shutdown failed" in line for line in self.logs))

    def test_the_exit_code_survives_every_s3_call_failing(self):
        workdir = self.prime()
        self.urgent.fail = {"put_object", "upload_file"}
        self.s3.fail = {"put_object", "upload_file", "delete_object"}
        self.assertEqual(runner.finalize_shutdown(workdir), runner.EXIT_INTERRUPTED)

    def test_the_exit_code_survives_the_result_glob_raising(self):
        workdir = self.prime()
        with patch.object(runner, "result_candidates",
                          side_effect=OSError("stat storm")):
            code = runner.finalize_shutdown(workdir)
        self.assertEqual(code, runner.EXIT_INTERRUPTED)

    def test_a_time_cap_exit_code_also_survives_a_failure(self):
        workdir = self.prime(state="failed", code=runner.EXIT_TIME_LIMIT)
        with patch.object(runner, "result_candidates", side_effect=OSError("boom")):
            code = runner.finalize_shutdown(workdir)
        self.assertEqual(code, runner.EXIT_TIME_LIMIT)

    def test_the_failure_path_still_writes_a_status_the_researcher_can_read(self):
        workdir = self.prime()
        with patch.object(runner, "result_candidates", side_effect=OSError("boom")):
            runner.finalize_shutdown(workdir)
        payloads = self.status_payloads()
        self.assertTrue(payloads)
        final = payloads[-1]
        self.assertEqual(final["state"], "interrupted")
        self.assertIsNone(final["partial"])
        self.assertIn("Shutdown handling itself failed", final["error"])

    def test_the_amended_status_carries_the_partial_details(self):
        workdir = self.prime()
        runner.finalize_shutdown(workdir)
        final = self.status_payloads()[-1]
        self.assertEqual(final["partial"], runner.PARTIAL_KEY)
        self.assertEqual(final["partial_key"], PARTIAL_FULL_KEY)
        self.assertEqual(final["partial_lines"], 4)
        self.assertFalse(final["partial_truncated"])
        self.assertIn("must not be read as a finished result", final["error"])

    def test_a_missing_delete_permission_does_not_fail_a_successful_run(self):
        self.s3.fail = {"delete_object"}
        runner.remove_stale_partial()
        self.assertTrue(any("left a previous attempt" in line for line in self.logs))


# ---------------------------------------------------------------------------
# 7. steps_from_model
# ---------------------------------------------------------------------------
class TestStepsFromModel(RunnerTestCase):

    def model(self, xml):
        path = os.path.join(self.workdir(), "model.xml")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(xml)
        return path

    def test_the_model_steps_are_used_when_there_is_no_override(self):
        path = self.model("<CompuCell3D><Potts><Steps>777</Steps></Potts></CompuCell3D>")
        self.assertEqual(runner.steps_from_model(path), 777)

    def test_the_env_override_wins_over_the_model(self):
        path = self.model("<CompuCell3D><Potts><Steps>777</Steps></Potts></CompuCell3D>")
        os.environ["CC3D_STEPS"] = "42"
        self.assertEqual(runner.steps_from_model(path), 42)

    def test_a_float_override_is_truncated(self):
        path = self.model("<CompuCell3D><Potts><Steps>777</Steps></Potts></CompuCell3D>")
        os.environ["CC3D_STEPS"] = "12.9"
        self.assertEqual(runner.steps_from_model(path), 12)

    def test_an_unparsable_override_is_ignored_rather_than_crashing(self):
        path = self.model("<CompuCell3D><Potts><Steps>777</Steps></Potts></CompuCell3D>")
        os.environ["CC3D_STEPS"] = "not-a-number"
        self.assertEqual(runner.steps_from_model(path), 777)
        self.assertTrue(any("unparsable CC3D_STEPS" in line for line in self.logs))

    def test_a_non_positive_override_is_ignored(self):
        path = self.model("<CompuCell3D><Potts><Steps>777</Steps></Potts></CompuCell3D>")
        for value in ("0", "-5", "-0.4"):
            with self.subTest(value=value):
                os.environ["CC3D_STEPS"] = value
                self.assertEqual(runner.steps_from_model(path), 777)

    def test_a_missing_model_falls_back_to_the_default(self):
        missing = os.path.join(self.workdir(), "nope.xml")
        self.assertEqual(runner.steps_from_model(missing), 100)
        self.assertEqual(runner.steps_from_model(missing, default=7), 7)

    def test_malformed_xml_falls_back_to_the_default(self):
        path = self.model("<CompuCell3D><Potts><Steps>7")
        self.assertEqual(runner.steps_from_model(path), 100)
        self.assertTrue(any("could not read <Steps>" in line for line in self.logs))

    def test_a_model_without_a_steps_node_falls_back_to_the_default(self):
        path = self.model("<CompuCell3D><Potts/></CompuCell3D>")
        self.assertEqual(runner.steps_from_model(path), 100)

    def test_a_non_positive_model_value_falls_back_to_the_default(self):
        path = self.model("<CompuCell3D><Potts><Steps>0</Steps></Potts></CompuCell3D>")
        self.assertEqual(runner.steps_from_model(path), 100)

    def test_read_max_minutes_rejects_junk_and_non_positive_values(self):
        self.assertEqual(runner.read_max_minutes(), runner.DEFAULT_MAX_MINUTES)
        for value, expected in (("30", 30.0), ("0.5", 0.5)):
            os.environ["CC3D_MAX_MINUTES"] = value
            self.assertEqual(runner.read_max_minutes(), expected)
        for value in ("abc", "0", "-3"):
            with self.subTest(value=value):
                os.environ["CC3D_MAX_MINUTES"] = value
                self.assertEqual(runner.read_max_minutes(),
                                 runner.DEFAULT_MAX_MINUTES)


# ---------------------------------------------------------------------------
# 8. _build_specs_from_config
# ---------------------------------------------------------------------------
class TestBuildSpecsFromConfig(RunnerTestCase):

    def test_a_config_with_no_cell_types_is_refused(self):
        for config in ({}, {"cell_types": []}, {"cell_types": None},
                       {"cell_types": ["Tumour"]},          # not dicts
                       {"cell_types": [{"name": "Medium"}]},
                       {"cell_types": [{"name": "   "}]}):
            with self.subTest(config=config):
                with self.assertRaises(ValueError) as caught:
                    runner._build_specs_from_config(config)
                self.assertIn("cell type", str(caught.exception).lower())

    def test_a_single_cell_type_builds_a_seeded_simulation(self):
        specs, steps = runner._build_specs_from_config({
            "lattice": {"x": 60, "y": 50, "z": 1},
            "steps": 250,
            "temperature": 12.5,
            "cell_types": [{"name": "Tumour"}, {"name": "Medium"}],
        })
        self.assertEqual(steps, 250)
        kinds = {type(spec).__name__ for spec in specs}
        self.assertIn("PottsCore", kinds)
        self.assertIn("CellTypePlugin", kinds)
        self.assertIn("VolumePlugin", kinds)
        self.assertIn("BlobInitializer", kinds,
                      "without an initialiser the lattice is empty")

        potts = next(s for s in specs if isinstance(s, PottsCore))
        self.assertEqual(potts.kwargs["dim_x"], 60)
        self.assertEqual(potts.kwargs["dim_y"], 50)
        self.assertEqual(potts.kwargs["steps"], 250)
        self.assertEqual(potts.kwargs["fluctuation_amplitude"], 12.5)

        types_spec = next(s for s in specs if isinstance(s, CellTypePlugin))
        self.assertEqual(types_spec.args, ("Tumour",),
                         "Medium must not be declared as a cell type")

        blob = next(s for s in specs if isinstance(s, BlobInitializer))
        self.assertEqual(blob.regions[0][1]["cell_types"], ("Tumour",))

    def test_the_volume_constraint_is_applied_to_every_type(self):
        specs, _ = runner._build_specs_from_config({
            "cell_types": [{"name": "A"}, {"name": "B"}],
            "volume_constraint": {"target_volume": 33, "lambda_volume": 4.5},
        })
        volume = next(s for s in specs if isinstance(s, VolumePlugin))
        names = [call[0][0] for call in volume.params]
        self.assertEqual(names, ["A", "B"])
        self.assertEqual(volume.params[0][1]["target_volume"], 33.0)
        self.assertEqual(volume.params[0][1]["lambda_volume"], 4.5)

    def test_contact_energies_are_wired_only_when_both_types_are_named(self):
        specs, _ = runner._build_specs_from_config({
            "cell_types": [{"name": "A"}, {"name": "B"}],
            "contact_energies": [
                {"type1": "A", "type2": "B", "energy": 6},
                {"type1": "A"},                 # incomplete: must be dropped
                {"type2": "B"},                 # incomplete: must be dropped
                "not-a-dict",
            ],
        })
        contact = next(s for s in specs if isinstance(s, ContactPlugin))
        self.assertEqual(len(contact.params), 1)
        self.assertEqual(contact.params[0][1]["energy"], 6.0)

    def test_no_contact_plugin_is_added_when_there_are_no_valid_pairs(self):
        specs, _ = runner._build_specs_from_config({
            "cell_types": [{"name": "A"}],
            "contact_energies": [{"type1": "A"}],
        })
        self.assertFalse(any(isinstance(s, ContactPlugin) for s in specs))

    def test_measurement_plugins_are_registered_when_available(self):
        specs, _ = runner._build_specs_from_config({"cell_types": [{"name": "A"}]})
        kinds = {type(spec).__name__ for spec in specs}
        self.assertIn("CenterOfMassPlugin", kinds)
        self.assertIn("SurfacePlugin", kinds)

    def test_a_missing_measurement_plugin_warns_instead_of_failing(self):
        specs_mod = sys.modules["cc3d.core.PyCoreSpecs"]
        with patch.object(specs_mod, "CenterOfMassPlugin", None), \
                patch.object(specs_mod, "SurfacePlugin", None):
            specs, _ = runner._build_specs_from_config({"cell_types": [{"name": "A"}]})
        self.assertTrue(specs)
        warnings = [line for line in self.logs if "WARNING" in line]
        self.assertEqual(len(warnings), 2)
        self.assertTrue(any("centre-of-mass" in line for line in warnings))


# ---------------------------------------------------------------------------
# 9. The signal handler
# ---------------------------------------------------------------------------
class TestSignalHandler(RunnerTestCase):

    def test_sigterm_is_attributed_to_spot_reclamation(self):
        signalled = []
        with patch.object(runner, "_signal_child", signalled.append):
            runner._on_terminate(signal.SIGTERM, None)
        self.assertEqual(runner._shutdown_reason, runner.SPOT_SIGTERM_REASON)
        self.assertEqual(runner._shutdown_state, "interrupted")
        self.assertEqual(runner._shutdown_exit_code, runner.EXIT_INTERRUPTED)
        self.assertEqual(signalled, [signal.SIGTERM])

    def test_a_non_sigterm_interrupt_gets_the_local_reason(self):
        with patch.object(runner, "_signal_child", lambda sig: None):
            runner._on_terminate(signal.SIGINT, None)
        self.assertEqual(runner._shutdown_reason, runner.LOCAL_STOP_REASON)

    def test_the_handler_touches_neither_s3_nor_the_log(self):
        """A print or a botocore call here runs on top of an upload in flight."""
        with patch.object(runner, "_signal_child", lambda sig: None):
            runner._on_terminate(signal.SIGTERM, None)
        self.assertEqual(self.s3.calls, [])
        self.assertEqual(self.urgent.calls, [])
        self.assertEqual(self.logs, [])

    def test_a_second_sigterm_is_a_no_op(self):
        signalled = []
        with patch.object(runner, "_signal_child", signalled.append):
            runner._on_terminate(signal.SIGTERM, None)
            runner._on_terminate(signal.SIGINT, None)
            runner._on_terminate(signal.SIGTERM, None)
        self.assertEqual(runner._shutdown_reason, runner.SPOT_SIGTERM_REASON)
        self.assertEqual(len(signalled), 1,
                         "a repeated signal must not re-signal the engine")

    def test_signal_child_is_a_no_op_when_there_is_no_engine(self):
        runner._child = None
        runner._signal_child(signal.SIGTERM)  # must not raise
        runner._await_child(0.01)

    def test_install_signal_handlers_installs_a_sigterm_handler(self):
        previous = {name: signal.getsignal(getattr(signal, name))
                    for name in ("SIGTERM", "SIGINT")
                    if hasattr(signal, name)}
        try:
            self.assertTrue(runner.install_signal_handlers())
            self.assertIs(signal.getsignal(signal.SIGTERM), runner._on_terminate)
        finally:
            for name, handler in previous.items():
                signal.signal(getattr(signal, name), handler)

    def test_a_container_that_cannot_handle_sigterm_refuses_to_run(self):
        with patch.object(runner.signal, "signal", side_effect=OSError("denied")):
            self.assertFalse(runner.install_signal_handlers())
        self.assertTrue(any("could not install a SIGTERM handler" in line
                            for line in self.logs))


# ---------------------------------------------------------------------------
# 10. The in-process stepping loop
# ---------------------------------------------------------------------------
class TestInProcessEngineLoop(RunnerTestCase):

    def setUp(self):
        super().setUp()
        StubSimService.reset()
        StubSteppable.cell_list = [FakeCell(1), FakeCell(2)]
        self.addCleanup(StubSimService.reset)

    def project(self, steps=5, cell_types=(("Tumour"),)):
        workdir = self.workdir()
        os.makedirs(os.path.join(workdir, "Simulation"), exist_ok=True)
        model = os.path.join(workdir, "Simulation", "model.xml")
        with open(model, "w", encoding="utf-8") as handle:
            handle.write(f"<CompuCell3D><Potts><Steps>{steps}</Steps></Potts>"
                         f"</CompuCell3D>")
        config = {"lattice": {"x": 40, "y": 40, "z": 1}, "steps": steps,
                  "cell_types": [{"name": name} for name in cell_types]}
        with open(os.path.join(workdir, "cc3d_config.json"), "w",
                  encoding="utf-8") as handle:
            json.dump(config, handle)
        return model, workdir

    def test_the_loop_writes_a_header_and_one_row_per_cell_per_step(self):
        model, workdir = self.project(steps=4)
        rows = runner.run_engine_python_api(model, workdir, max_minutes=90)
        self.assertEqual(rows, 8)
        with open(os.path.join(workdir, runner.RESULT_KEY), encoding="utf-8") as handle:
            lines = handle.read().splitlines()
        self.assertEqual(lines[0], CSV_HEADER.strip())
        self.assertEqual(len(lines), 9)
        self.assertTrue(lines[1].startswith("1,1,Tumour,"))

    def test_a_missing_config_is_a_clear_error_not_an_empty_simulation(self):
        model, workdir = self.project()
        os.remove(os.path.join(workdir, "cc3d_config.json"))
        with self.assertRaises(RuntimeError) as caught:
            runner.run_engine_python_api(model, workdir, max_minutes=90)
        self.assertIn("cc3d_config.json", str(caught.exception))

    def test_the_loop_stops_at_the_next_step_after_a_shutdown_is_requested(self):
        model, workdir = self.project(steps=10)

        def hook(step):
            if step == 3:
                runner.request_shutdown(runner.SPOT_SIGTERM_REASON)

        StubSimService.step_hook = staticmethod(hook)
        rows = runner.run_engine_python_api(model, workdir, max_minutes=90)
        self.assertEqual(rows, 6, "steps 1-3 measured, then the loop stops")
        self.assertEqual(StubSimService.instances[0].steps, 3)
        with open(os.path.join(workdir, runner.RESULT_KEY), "rb") as handle:
            body = handle.read()
        self.assertTrue(body.endswith(b"\n"),
                        "the flushed partial must end on a whole row")

    def test_the_time_cap_requests_a_failed_shutdown_with_the_time_limit_code(self):
        model, workdir = self.project(steps=50)
        runner._job_started = time.monotonic() - 600.0  # the cap is long gone
        rows = runner.run_engine_python_api(model, workdir, max_minutes=1)
        self.assertEqual(rows, 0)
        self.assertTrue(runner._shutdown_requested)
        self.assertEqual(runner._shutdown_state, "failed")
        self.assertEqual(runner._shutdown_exit_code, runner.EXIT_TIME_LIMIT)

    def test_the_time_cap_reason_is_formatted_before_it_reaches_the_researcher(self):
        """status.json must not surface a raw format placeholder.

        The subprocess path calls TIME_CAP_REASON.format(minutes=...); this in-process
        path passes the template unformatted, so the researcher is shown the literal
        "{minutes:g}".
        """
        model, workdir = self.project(steps=50)
        runner._job_started = time.monotonic() - 600.0
        runner.run_engine_python_api(model, workdir, max_minutes=1)
        self.assertNotIn("{minutes", runner._shutdown_reason,
                         "the time-cap reason still contains an unformatted "
                         "placeholder, which lands verbatim in status.json['error']")

    def test_the_step_override_drives_the_loop(self):
        model, workdir = self.project(steps=3)
        os.environ["CC3D_STEPS"] = "2"
        rows = runner.run_engine_python_api(model, workdir, max_minutes=90)
        self.assertEqual(rows, 4)
        self.assertEqual(StubSimService.instances[0].steps, 2)

    def test_an_unparsable_step_override_is_ignored(self):
        model, workdir = self.project(steps=3)
        os.environ["CC3D_STEPS"] = "many"
        rows = runner.run_engine_python_api(model, workdir, max_minutes=90)
        self.assertEqual(rows, 6)
        self.assertTrue(any("unparsable CC3D_STEPS" in line for line in self.logs))

    def test_a_save_interval_thins_the_measurements_but_always_saves_the_last_step(self):
        model, workdir = self.project(steps=5)
        os.environ["CC3D_SAVE_EVERY"] = "2"
        rows = runner.run_engine_python_api(model, workdir, max_minutes=90)
        self.assertEqual(rows, 6, "steps 2, 4 and the final step 5")

    def test_a_failing_finish_does_not_discard_written_results(self):
        model, workdir = self.project(steps=2)
        StubSimService.finish_error = RuntimeError("teardown exploded")
        rows = runner.run_engine_python_api(model, workdir, max_minutes=90)
        self.assertEqual(rows, 4)
        self.assertTrue(any("results are already written" in line
                            for line in self.logs))

    def test_a_config_with_no_cell_types_stops_the_run_before_it_starts(self):
        model, workdir = self.project(steps=5, cell_types=())
        with self.assertRaises(ValueError):
            runner.run_engine_python_api(model, workdir, max_minutes=90)
        self.assertEqual(StubSimService.instances, [],
                         "no simulation may be constructed for an empty lattice")


# ---------------------------------------------------------------------------
# 11. Supporting paths: download, candidates, the subprocess cap
# ---------------------------------------------------------------------------
class TestSupportingPaths(RunnerTestCase):

    def test_download_skips_a_previous_attempts_outputs(self):
        workdir = self.workdir()
        self.s3.pages = [{"Contents": [
            {"Key": f"{TEST_PREFIX}/Simulation/model.xml"},
            {"Key": f"{TEST_PREFIX}/cc3d_config.json"},
            {"Key": f"{TEST_PREFIX}/{runner.RESULT_KEY}"},
            {"Key": f"{TEST_PREFIX}/{runner.PARTIAL_KEY}"},
            {"Key": f"{TEST_PREFIX}/{runner.STATUS_KEY}"},
            {"Key": f"{TEST_PREFIX}/"},
        ]}]
        found = runner.download_project(workdir)
        self.assertEqual(sorted(found), ["Simulation/model.xml", "cc3d_config.json"])
        downloaded = [kw["Key"] for kw in self.s3.ops("download_file")]
        self.assertNotIn(f"{TEST_PREFIX}/{runner.RESULT_KEY}", downloaded)
        self.assertNotIn(f"{TEST_PREFIX}/{runner.PARTIAL_KEY}", downloaded)

    def test_a_reclaim_during_the_download_abandons_it(self):
        workdir = self.workdir()
        self.s3.pages = [{"Contents": [{"Key": f"{TEST_PREFIX}/a.txt"},
                                       {"Key": f"{TEST_PREFIX}/b.txt"}]}]
        runner.request_shutdown(runner.SPOT_SIGTERM_REASON)
        with self.assertRaises(runner.Interrupted):
            runner.download_project(workdir)
        self.assertEqual(self.s3.ops("download_file"), [],
                         "no file should be pulled during a shutdown")

    def test_result_candidates_finds_the_csv_at_any_depth(self):
        workdir = self.workdir()
        self.write_result(workdir, CSV_HEADER)
        self.write_result(workdir, CSV_HEADER, subdir=os.path.join("a", "b"))
        found = runner.result_candidates(workdir)
        self.assertEqual(len(found), 2)

    def test_the_subprocess_path_formats_its_time_cap_reason(self):
        """Contrast with the in-process path, which passes the template raw."""
        command = [sys.executable, "-c", "import time; time.sleep(5)"]
        workdir = self.workdir()
        with patch.object(runner, "POLL_SECS", 0.01):
            returncode, _tail = runner.run_engine(command, workdir,
                                                  max_minutes=0.001)
        child = runner._child
        self.addCleanup(_reap, child)
        self.assertIsNone(returncode, "an interrupted run reports no return code")
        self.assertEqual(runner._shutdown_state, "failed")
        self.assertEqual(runner._shutdown_exit_code, runner.EXIT_TIME_LIMIT)
        self.assertNotIn("{minutes", runner._shutdown_reason)
        self.assertIn("0.001 minutes", runner._shutdown_reason)

    def test_a_reclaim_before_the_engine_starts_prevents_the_spawn(self):
        runner.request_shutdown(runner.SPOT_SIGTERM_REASON)
        with self.assertRaises(runner.Interrupted):
            runner.run_engine([sys.executable, "-c", "pass"], self.workdir(), 90)

    def test_the_exit_codes_are_distinct_and_interrupted_is_75(self):
        codes = {name: value for name, value in vars(runner).items()
                 if name.startswith("EXIT_")}
        self.assertEqual(codes["EXIT_INTERRUPTED"], 75)
        self.assertEqual(len(set(codes.values())), len(codes),
                         "two exit codes collide, so the log cannot say what happened")


def _reap(child):
    if child is None:
        return
    try:
        child.kill()
    except Exception:
        pass
    try:
        child.wait(timeout=5)
    except Exception:
        pass


if __name__ == "__main__":
    unittest.main(verbosity=2)
