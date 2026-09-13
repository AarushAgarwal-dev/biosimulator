"""
Remote CompuCell3D execution on AWS: S3 for payloads, Batch for compute.

Why this exists
---------------
CompuCell3D cannot run on the Render deployment: it is distributed through Conda
rather than PyPI, needs VTK and native rendering libraries (~2-3 GB installed), and
the web instance has 512 MB of RAM. Rather than move the whole application, the
engine runs as an ON-DEMAND AWS Batch job that only bills while a simulation is
executing. The web app stays where it is.

The flow
--------
1. The adapter writes the generated CC3DML project and steppable to S3 under a
   per-run prefix.
2. It submits an AWS Batch job whose container has CompuCell3D installed; the
   container downloads the prefix, runs the simulation headless, and uploads its
   measurement CSV plus a status file back to the same prefix.
3. The adapter polls the job, streams CloudWatch log lines when available, and
   downloads the CSV when the job succeeds.

Design constraints
------------------
* boto3 is already a project dependency (the Bedrock engine uses it), so this adds
  no new package.
* No AWS client is created at import. Nothing here touches the network until a run
  is actually submitted, so importing the module on a machine with no credentials
  is harmless and ``configuration_status`` stays a pure environment read.
* Every AWS interaction goes through a small number of methods, so the tests can
  substitute a fake client and never call AWS.
* Failures are reported, never swallowed: a job that fails, is cancelled by AWS, or
  produces no output raises with the reason rather than returning empty results.
* A run is capped at :data:`REMOTE_RUN_LENGTH_CAP_MINUTES` of ESTIMATED wall clock and
  refused before submission if it would exceed that, because a spot interruption
  restarts one of these jobs from step 0 instead of resuming it.
"""

import json
import os
import posixpath
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

#: Environment variables that configure remote execution. All four are required.
REGION_ENV = "BIOSIM_AWS_REGION"
QUEUE_ENV = "BIOSIM_CC3D_JOB_QUEUE"
JOB_DEFINITION_ENV = "BIOSIM_CC3D_JOB_DEFINITION"
BUCKET_ENV = "BIOSIM_CC3D_BUCKET"
#: Optional: prefix inside the bucket, and how long to wait before giving up.
PREFIX_ENV = "BIOSIM_CC3D_PREFIX"
TIMEOUT_ENV = "BIOSIM_CC3D_REMOTE_TIMEOUT"

DEFAULT_PREFIX = "cc3d-runs"
DEFAULT_TIMEOUT_SECS = 7200.0

#: Terminal AWS Batch job states.
BATCH_SUCCESS = "SUCCEEDED"
BATCH_FAILED = "FAILED"
BATCH_TERMINAL = (BATCH_SUCCESS, BATCH_FAILED)

# ---------------------------------------------------------------------------
# Job sizing
# ---------------------------------------------------------------------------
#: Matched to an m7i.xlarge job definition: 4 vCPU, 16 GiB.
DEFAULT_VCPUS = 4
#: 15500 MiB, not the instance's full 16384. AWS Batch places jobs through ECS, which
#: holds back memory for the container agent and the OS, so a job that requests every
#: MiB the instance advertises is never placeable and sits in RUNNABLE indefinitely.
#: The ~880 MiB left here is the headroom that makes the job schedulable.
DEFAULT_MEMORY_MIB = 15500

# ---------------------------------------------------------------------------
# Run-length cap
# ---------------------------------------------------------------------------
#: Ceiling on the ESTIMATED wall-clock length of one remote run, in minutes.
#:
#: These jobs run on spot capacity. A spot reclaim RESTARTS the job from step 0 --
#: CompuCell3D writes no checkpoint the container could resume from -- so an over-long
#: run is not merely slow: each attempt is billed and each is likely to be interrupted
#: before it finishes, which can mean paying repeatedly for a result that never
#: arrives. Refusing the configuration during validation is cheaper and more honest
#: than discovering that after an hour and a half.
REMOTE_RUN_LENGTH_CAP_MINUTES = 90.0

#: What a job costs before it has simulated anything: image pull, conda activation,
#: S3 download of the project, and the result upload at the end.
REMOTE_STARTUP_OVERHEAD_MINUTES = 2.5

#: Assumed throughput, in lattice-site update attempts per second.
#:
#: One Monte Carlo Step attempts one spin flip per lattice site, so a run costs about
#: ``steps * x * y * z`` attempts. CC3D's Potts loop is effectively serial -- the extra
#: vCPUs go to the PDE solvers -- and measured single-core throughput spans roughly
#: 1-5 million attempts/second depending on the plugin set. The LOW end is used
#: deliberately: an optimistic figure would let exactly the runs this cap exists to
#: stop through, so the estimate errs towards refusing a borderline configuration.
REMOTE_SITE_UPDATES_PER_SECOND = 1_200_000.0

#: Each diffusion field adds a full-lattice PDE sweep per step on top of the Potts
#: work; charged here as half a Potts sweep per field. Pessimistic, for the reason
#: above.
REMOTE_FIELD_COST_FACTOR = 0.5


def lattice_site_count(lattice: Any) -> int:
    """Sites in a lattice, from an ``{'x','y','z'}`` mapping or a plain count.

    Returns 0 for anything unreadable. A malformed lattice already produces its own
    validation errors, and the run-length estimate must not pile a second, more
    confusing message on top of them.
    """
    if isinstance(lattice, dict):
        total = 1
        for axis in ("x", "y", "z"):
            try:
                total *= max(1, int(lattice.get(axis, 1)))
            except (TypeError, ValueError):
                return 0
        return total
    try:
        return max(0, int(lattice))
    except (TypeError, ValueError):
        return 0


def _step_count(steps: Any) -> int:
    try:
        return max(0, int(steps))
    except (TypeError, ValueError):
        return 0


def estimate_runtime_minutes(steps: Any, lattice: Any, field_count: int = 0) -> float:
    """Conservative wall-clock estimate for one remote run, in minutes.

    ``startup + steps * sites * (1 + 0.5 * fields) / throughput``. See the constants
    above for every assumption baked into it; it is intentionally pessimistic.
    """
    sites = lattice_site_count(lattice)
    step_count = _step_count(steps)
    if sites <= 0 or step_count <= 0:
        return REMOTE_STARTUP_OVERHEAD_MINUTES
    attempts = float(step_count) * float(sites)
    attempts *= 1.0 + REMOTE_FIELD_COST_FACTOR * max(0, int(field_count or 0))
    return REMOTE_STARTUP_OVERHEAD_MINUTES + (attempts / REMOTE_SITE_UPDATES_PER_SECOND) / 60.0


def max_steps_within_cap(lattice: Any, field_count: int = 0) -> int:
    """Largest Monte Carlo step count that fits inside the cap on this lattice.

    The inverse of :func:`estimate_runtime_minutes`, so a refusal can tell the
    researcher the number to type rather than only that theirs is too big.
    """
    sites = lattice_site_count(lattice)
    if sites <= 0:
        return 0
    budget_minutes = max(0.0, REMOTE_RUN_LENGTH_CAP_MINUTES - REMOTE_STARTUP_OVERHEAD_MINUTES)
    attempts = budget_minutes * 60.0 * REMOTE_SITE_UPDATES_PER_SECOND
    attempts /= 1.0 + REMOTE_FIELD_COST_FACTOR * max(0, int(field_count or 0))
    return int(attempts // sites)


def run_length_check(steps: Any, lattice: Any, field_count: int = 0) -> Dict[str, Any]:
    """Decide whether a configuration may run remotely, and say why if not.

    The wording lives here rather than in the adapter so validation, the pre-submit
    re-check and the tests all quote one message.
    """
    estimated = estimate_runtime_minutes(steps, lattice, field_count)
    permitted = max_steps_within_cap(lattice, field_count)
    within_cap = estimated <= REMOTE_RUN_LENGTH_CAP_MINUTES
    message = ""
    if not within_cap:
        message = (
            f"This configuration is estimated to need about {estimated:.0f} minutes on "
            f"AWS Batch, which exceeds the {REMOTE_RUN_LENGTH_CAP_MINUTES:g} minute cap "
            f"for a remote CompuCell3D run. Reduce the Monte Carlo step count from "
            f"{_step_count(steps)} to at most {permitted}, or use a lattice smaller than "
            f"{lattice_site_count(lattice)} sites. The cap exists because AWS Batch runs "
            f"these jobs on spot capacity: an interruption restarts the job from step 0 "
            f"instead of resuming it, so a longer run can be billed several times over "
            f"and still never finish."
        )
    return {
        "within_cap": within_cap,
        "estimated_minutes": round(estimated, 2),
        "cap_minutes": REMOTE_RUN_LENGTH_CAP_MINUTES,
        "max_steps": permitted,
        "steps": _step_count(steps),
        "lattice_sites": lattice_site_count(lattice),
        "message": message,
    }


class RemoteConfigurationError(RuntimeError):
    """Remote execution is not configured, or is configured incompletely."""


def configuration_status() -> Dict[str, Any]:
    """Report whether remote execution is configured. Pure environment read.

    Never raises and never touches the network, so it is safe to call from a
    capability check on every request.
    """
    values = {
        "region": os.environ.get(REGION_ENV, "").strip(),
        "job_queue": os.environ.get(QUEUE_ENV, "").strip(),
        "job_definition": os.environ.get(JOB_DEFINITION_ENV, "").strip(),
        "bucket": os.environ.get(BUCKET_ENV, "").strip(),
        "prefix": os.environ.get(PREFIX_ENV, "").strip() or DEFAULT_PREFIX,
    }
    missing = [name for name, key in (
        (REGION_ENV, "region"), (QUEUE_ENV, "job_queue"),
        (JOB_DEFINITION_ENV, "job_definition"), (BUCKET_ENV, "bucket"),
    ) if not values[key]]

    if missing:
        return {
            "available": False,
            "configured": False,
            "reason": (
                "Remote CompuCell3D execution is not configured. Set "
                + ", ".join(missing)
                + ". These name an AWS Batch job queue and job definition whose "
                  "container has CompuCell3D installed, plus an S3 bucket for the "
                  "project files and results."
            ),
            **values,
        }

    try:
        import boto3  # noqa: F401
    except Exception as exc:
        return {
            "available": False,
            "configured": True,
            "reason": f"boto3 is required for remote execution but could not be "
                      f"imported ({type(exc).__name__}). Run: pip install boto3",
            **values,
        }

    try:
        timeout = float(os.environ.get(TIMEOUT_ENV, DEFAULT_TIMEOUT_SECS))
    except (TypeError, ValueError):
        timeout = DEFAULT_TIMEOUT_SECS

    return {"available": True, "configured": True, "reason": "",
            "timeout_secs": timeout, **values}


class BatchRunner:
    """Runs one CompuCell3D simulation as an AWS Batch job.

    Clients are injectable so tests exercise the whole submit/poll/fetch sequence
    against fakes. In production they are created lazily from the standard AWS
    credential chain, exactly like the Bedrock engine.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None,
                 s3_client: Any = None, batch_client: Any = None,
                 logs_client: Any = None):
        self.config = config or configuration_status()
        if not self.config.get("configured"):
            raise RemoteConfigurationError(self.config.get("reason", "not configured"))
        self._s3 = s3_client
        self._batch = batch_client
        self._logs = logs_client

    # -- clients ------------------------------------------------------------
    def _client(self, service: str, cached: Any) -> Any:
        if cached is not None:
            return cached
        import boto3
        return boto3.client(service, region_name=self.config["region"])

    @property
    def s3(self) -> Any:
        self._s3 = self._client("s3", self._s3)
        return self._s3

    @property
    def batch(self) -> Any:
        self._batch = self._client("batch", self._batch)
        return self._batch

    @property
    def logs(self) -> Any:
        self._logs = self._client("logs", self._logs)
        return self._logs

    # -- paths --------------------------------------------------------------
    def run_prefix(self, run_id: str) -> str:
        return posixpath.join(self.config["prefix"], str(run_id))

    def key(self, run_id: str, *parts: str) -> str:
        return posixpath.join(self.run_prefix(run_id), *parts)

    # -- submit -------------------------------------------------------------
    def upload_project(self, run_id: str, files: Dict[str, str]) -> List[str]:
        """Upload the generated CC3D project. Returns the keys written."""
        written: List[str] = []
        for name, contents in files.items():
            key = self.key(run_id, name)
            self.s3.put_object(
                Bucket=self.config["bucket"], Key=key,
                Body=contents.encode("utf-8"),
                ContentType="text/xml" if name.endswith(".xml") else "text/plain",
            )
            written.append(key)
        return written

    def submit(self, run_id: str, files: Dict[str, str],
               vcpus: int = DEFAULT_VCPUS, memory_mib: int = DEFAULT_MEMORY_MIB,
               steps: Optional[int] = None) -> Dict[str, Any]:
        """Upload the project and submit the Batch job. Returns job identifiers.

        The defaults size the job for an m7i.xlarge job definition; see
        :data:`DEFAULT_MEMORY_MIB` for why it is not the instance's full 16384 MiB.
        """
        keys = self.upload_project(run_id, files)
        bucket = self.config["bucket"]
        prefix = self.run_prefix(run_id)

        # The container reads everything it needs from the environment, so the job
        # definition stays generic and one definition serves every run.
        environment = [
            {"name": "CC3D_BUCKET", "value": bucket},
            {"name": "CC3D_PREFIX", "value": prefix},
            {"name": "CC3D_RUN_ID", "value": str(run_id)},
            {"name": "AWS_DEFAULT_REGION", "value": self.config["region"]},
            # The cap has to travel WITH the job. Previously the container fell back
            # to its own default and the Dockerfile's ENV, so three copies of "90"
            # agreed only by coincidence -- change one and the enforcement silently
            # diverges from what validation promised the researcher.
            {"name": "CC3D_MAX_MINUTES", "value": str(REMOTE_RUN_LENGTH_CAP_MINUTES)},
        ]
        if steps:
            environment.append({"name": "CC3D_STEPS", "value": str(int(steps))})

        response = self.batch.submit_job(
            jobName=f"cc3d-{str(run_id).replace('_', '-')[:100]}",
            jobQueue=self.config["job_queue"],
            jobDefinition=self.config["job_definition"],
            # AWS itself is the backstop. A container wedged before it can enforce its
            # own cap is otherwise bounded only by a client that happens to still be
            # polling, which is not a guarantee -- and an abandoned spot instance
            # bills for every minute of it.
            timeout={"attemptDurationSeconds":
                     int((float(REMOTE_RUN_LENGTH_CAP_MINUTES) + 10.0) * 60)},
            containerOverrides={
                "environment": environment,
                "resourceRequirements": [
                    {"type": "VCPU", "value": str(int(vcpus))},
                    {"type": "MEMORY", "value": str(int(memory_mib))},
                ],
            },
        )
        return {
            "job_id": response["jobId"],
            "job_name": response.get("jobName"),
            "bucket": bucket,
            "prefix": prefix,
            "uploaded": keys,
        }

    # -- poll ---------------------------------------------------------------
    def describe(self, job_id: str) -> Dict[str, Any]:
        response = self.batch.describe_jobs(jobs=[job_id])
        jobs = response.get("jobs") or []
        if not jobs:
            raise RuntimeError(f"AWS Batch does not know job {job_id!r}.")
        job = jobs[0]
        return {
            "status": job.get("status", "UNKNOWN"),
            "reason": job.get("statusReason", ""),
            "log_stream": ((job.get("container") or {}).get("logStreamName") or ""),
            "exit_code": (job.get("container") or {}).get("exitCode"),
            "started_at": job.get("startedAt"),
            "stopped_at": job.get("stoppedAt"),
        }

    def tail_log(self, log_stream: str, next_token: Optional[str] = None,
                 log_group: str = "/aws/batch/job") -> Tuple[List[str], Optional[str]]:
        """Fetch new CloudWatch lines. Log access is best effort: a missing stream
        must not fail a run that is otherwise progressing."""
        if not log_stream:
            return [], next_token
        try:
            kwargs: Dict[str, Any] = {
                "logGroupName": log_group,
                "logStreamName": log_stream,
                "startFromHead": True,
            }
            if next_token:
                kwargs["nextToken"] = next_token
            response = self.logs.get_log_events(**kwargs)
            lines = [event.get("message", "") for event in response.get("events", [])]
            return lines, response.get("nextForwardToken", next_token)
        except Exception:
            return [], next_token

    def wait(self, job_id: str, timeout_secs: Optional[float] = None,
             poll_secs: float = 5.0,
             on_progress: Optional[Callable[[Dict[str, Any]], None]] = None,
             on_log: Optional[Callable[[str], None]] = None,
             is_cancelled: Optional[Callable[[], bool]] = None,
             wait_if_paused: Optional[Callable[[], None]] = None) -> Dict[str, Any]:
        """Block until the job reaches a terminal state.

        Cancellation and the timeout are both checked between polls, so a job that
        produces no output is still cancellable and still bounded -- the same
        property the local subprocess path needed.

        Pause is honoured between polls too, and means exactly one thing: polling
        stops. The Batch job keeps running and keeps billing, because nothing inside a
        CompuCell3D container can be frozen mid-simulation -- cancel, not pause, is
        what stops the spend. Time spent paused is added back to the deadline, so a
        pause can never be the reason a job is terminated for timing out.
        """
        deadline = time.monotonic() + float(
            timeout_secs if timeout_secs is not None
            else self.config.get("timeout_secs", DEFAULT_TIMEOUT_SECS))
        token: Optional[str] = None
        last_status = ""

        while True:
            if wait_if_paused is not None:
                paused_at = time.monotonic()
                wait_if_paused()
                deadline += time.monotonic() - paused_at
            info = self.describe(job_id)
            if info["status"] != last_status:
                last_status = info["status"]
                if on_progress:
                    on_progress(info)
            if on_log and info["log_stream"]:
                lines, token = self.tail_log(info["log_stream"], token)
                for line in lines:
                    on_log(line)
            if info["status"] in BATCH_TERMINAL:
                return info
            if is_cancelled and is_cancelled():
                self.cancel(job_id, "cancelled by the user")
                info["status"] = "CANCELLED"
                return info
            if time.monotonic() > deadline:
                self.cancel(job_id, "exceeded the configured timeout")
                raise TimeoutError(
                    f"The CompuCell3D job did not finish within the configured limit "
                    f"and was terminated. Raise {TIMEOUT_ENV} for a longer simulation.")
            time.sleep(poll_secs)

    def cancel(self, job_id: str, reason: str = "cancelled") -> None:
        """Stop a job. Both calls are attempted because which one applies depends on
        whether AWS has already placed the job on a compute host."""
        for method in ("terminate_job", "cancel_job"):
            call = getattr(self.batch, method, None)
            if call is None:
                continue
            try:
                call(jobId=job_id, reason=reason[:1024])
                return
            except Exception:
                continue

    # -- results ------------------------------------------------------------
    def fetch_text(self, run_id: str, name: str) -> Optional[str]:
        try:
            response = self.s3.get_object(Bucket=self.config["bucket"],
                                          Key=self.key(run_id, name))
            body = response["Body"].read()
            return body.decode("utf-8") if isinstance(body, bytes) else str(body)
        except Exception as exc:
            # A genuinely absent object means "not produced" and is a normal answer.
            # ANYTHING ELSE -- AccessDenied, a throttle, a transport failure -- is a
            # different fact, and collapsing it to None made the caller tell the
            # researcher "the job wrote no measurement file" when the truth was that
            # we could not read it. Misdiagnosing a permissions problem as an empty
            # result sends someone to debug their model instead of their IAM policy.
            code = ""
            response_meta = getattr(exc, "response", None)
            if isinstance(response_meta, dict):
                code = str((response_meta.get("Error") or {}).get("Code") or "")
            if code in ("NoSuchKey", "404", "NotFound") or type(exc).__name__ == "NoSuchKey":
                return None
            raise RuntimeError(
                f"Could not read s3://{self.config['bucket']}/{self.key(run_id, name)}: "
                f"{code or type(exc).__name__}: {exc}"
            ) from exc

    def fetch_results(self, run_id: str) -> Dict[str, Any]:
        """Download what the container produced.

        A job can succeed at the container level and still have produced nothing
        usable, so the absence of the measurement file is an error rather than an
        empty result set.
        """
        status_text = self.fetch_text(run_id, "status.json")
        status: Dict[str, Any] = {}
        if status_text:
            try:
                status = json.loads(status_text)
            except ValueError:
                status = {"raw": status_text[:2000]}

        cells_csv = self.fetch_text(run_id, "cells.csv")
        if not cells_csv:
            # A spot interruption uploads whatever was computed to a DISTINCT partial
            # key. It was previously written and never mentioned again, so a
            # researcher whose 80-minute run was reclaimed at minute 78 was told only
            # that nothing was produced. Name it, so they can decide whether the
            # partial is already enough.
            partial_key = status.get("partial_key") or "cells.partial.csv"
            partial_note = ""
            if status.get("state") == "interrupted" or status.get("partial"):
                partial_note = (
                    f" A PARTIAL result from before the interruption is at "
                    f"s3://{self.config['bucket']}/{self.key(run_id, partial_key)} -- "
                    f"it is deliberately not treated as a complete result, but it may "
                    f"still be usable."
                )
            raise RuntimeError(
                f"The CompuCell3D job finished but wrote no measurement file to "
                f"s3://{self.config['bucket']}/{self.key(run_id, 'cells.csv')}. "
                f"Container status: {status.get('error') or status.get('state') or 'unknown'}."
                f"{partial_note}"
            )
        return {"cells_csv": cells_csv, "status": status,
                "s3_prefix": f"s3://{self.config['bucket']}/{self.run_prefix(run_id)}"}

    def cleanup(self, run_id: str) -> int:
        """Delete a run's S3 objects. Returns how many were removed."""
        bucket = self.config["bucket"]
        prefix = self.run_prefix(run_id)
        removed = 0
        try:
            listing = self.s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
            keys = [{"Key": item["Key"]} for item in listing.get("Contents", [])]
            if keys:
                self.s3.delete_objects(Bucket=bucket, Delete={"Objects": keys})
                removed = len(keys)
        except Exception:
            pass
        return removed


#: Default per-vCPU-hour rate for the cost estimate, in USD.
#:
#: Derived from an m7i.xlarge (4 vCPU / 16 GiB) at roughly $0.078/hour spot in
#: us-east-2 -- about 60% below its $0.2016/hour on-demand rate -- divided by its 4
#: vCPUs. That instance family is what the CompuCell3D job definition targets, so the
#: figure at least describes the hardware the job actually lands on. Spot prices move
#: continuously and differ by region and availability zone, so it is a display default
#: rather than a quote.
M7I_XLARGE_SPOT_USD_PER_VCPU_HOUR = 0.0195


def estimate_cost(vcpus: int, memory_mib: int, minutes: float,
                  spot_price_per_vcpu_hour: float = M7I_XLARGE_SPOT_USD_PER_VCPU_HOUR
                  ) -> Dict[str, Any]:
    """Rough compute cost for one run, for display beside a queued job.

    This is an ESTIMATE and nothing more. The default rate is the m7i.xlarge spot
    figure documented on :data:`M7I_XLARGE_SPOT_USD_PER_VCPU_HOUR`; the real charge
    depends on the live spot price, the availability zone, and which instance the
    Batch compute environment actually selects. A caller with a better number --
    a Cost Explorer figure, a negotiated rate, an on-demand fallback price -- should
    pass ``spot_price_per_vcpu_hour`` to override it.
    """
    hours = max(0.0, float(minutes)) / 60.0
    compute = vcpus * spot_price_per_vcpu_hour * hours
    return {
        "vcpus": int(vcpus),
        "memory_mib": int(memory_mib),
        "minutes": float(minutes),
        "usd_per_vcpu_hour": float(spot_price_per_vcpu_hour),
        "estimated_usd": round(compute, 4),
        "basis": ("m7i.xlarge spot vCPU-hour rate; an estimate only, and it excludes "
                  "S3 storage and data transfer"),
        "is_estimate": True,
    }
