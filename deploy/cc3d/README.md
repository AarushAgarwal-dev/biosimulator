# On-demand CompuCell3D compute on AWS

How BioSimulateAI runs CompuCell3D without hosting it: the web app stays on Render,
the engine runs as an AWS Batch job in **us-east-2** that exists only while a
simulation is executing.

Every dollar figure in this document is an **estimate** for us-east-2 at the time of
writing. Spot prices move and AWS list prices change; treat them as order of
magnitude, not as a quote.

---

## 1. Why this exists

CompuCell3D cannot be installed on the Render web instance:

| Constraint | Detail |
|---|---|
| Distribution channel | CC3D is published on its **own official conda channel** (`-c compucell3d`). It is **not on PyPI**, so `requirements.txt` cannot install it and Render's pip-only build cannot produce it. |
| Native dependencies | It pulls **VTK** and needs native GL/X libraries present even for offscreen use — `libgl1`, `libglu1-mesa`, `libxrender1`, `libxext6`, `libsm6`. |
| Installed size | **~2-3 GB** once the conda environment is solved (the built image lands around 2-4 GB). |
| Render web instance | **512 MB RAM.** The engine does not fit, and neither does its install step. |

The alternative to moving the entire application off Render is to move only the
engine. That is what this directory does: `Dockerfile` builds a worker image with
CC3D installed from its own channel, `cc3d_job_runner.py` is that image's
entrypoint, and `cc3d_remote.py` in the repo root is the Render-side client that
uploads a project, submits one Batch job, polls it, and imports the result.

**ABM and MPC are unaffected.** Both run normally on Render — pure Python, pip
installable, no VTK, no conda channel, no GPU, no native display. Only the
CompuCell3D approach dispatches to AWS. The three approaches are independently
selectable: nothing about this deployment couples them, and none of them falls back
to another. (MPC = Model Predictive Control.)

The adapter is honest about where the engine is. `detect_cc3d()` in
`D:\SURF\biosimulator\approach_cc3d.py` checks, in order: the `cc3d` Python package
in this interpreter, `BIOSIM_CC3D_RUNSCRIPT`, a `runScript` on `PATH`, and only
then AWS Batch. Remote is checked **last** so a local engine always wins — running
in-process is faster and free, and only the absence of one justifies paying for a
job. When neither exists, the adapter reports `available=False` with the exact
missing dependency, still validates, still exports a runnable CC3D project, and
**never** silently substitutes the in-tree ABM engine.

---

## 2. Architecture

```
  Browser
     │  POST /runs  (approach = cc3d)
     ▼
┌─────────────────────────────────────────────────────────────────┐
│ Render web service   (FastAPI, 512 MB, pip only)                │
│   approach_cc3d.py  → builds CC3DML XML + Python steppable      │
│   cc3d_remote.py    → BatchRunner: upload, submit, poll, fetch  │
│   holds only IAM keys in env vars; no engine, no VTK            │
└───────┬──────────────────────────────────────────┬──────────────┘
        │ ① S3 PutObject                           │ ④ DescribeJobs (5 s poll)
        │    cc3d-runs/<run_id>/                   │    + GetLogEvents (best effort)
        │      Simulation/model.xml                │ ⑤ GetObject cells.csv
        │      Simulation/steppables.py            │            status.json
        ▼                                          │
┌──────────────────────────┐                       │
│ S3  (one bucket)         │◀──────────────────────┼───────┐
│  one prefix per run      │                       │       │
└──────────────────────────┘                       │       │
        ▲ ② Batch SubmitJob                        │       │
        │    env: CC3D_BUCKET, CC3D_PREFIX,        │       │
        │         CC3D_RUN_ID, CC3D_STEPS          │       │
        │    override: 4 vCPU / 8192 MiB           │       │
        │                                          ▼       │
┌───────┴──────────────────────────────────────────────────┴─────┐
│ AWS Batch  (us-east-2)                                         │
│   job queue → compute environment: SPOT, m7i.xlarge            │
│   minvCpus = 0   desiredvCpus = 0   ← no idle instance         │
│                                                                │
│   ③ spot instance acquired → pulls image from ECR (2-3 GB)     │
│      container runs cc3d_job_runner.py:                        │
│        · downloads the whole run prefix into a temp workdir    │
│        · locates CC3D's runScript inside the image             │
│        · runScript -i Simulation/model.xml --noOutput          │
│          headless: QT_QPA_PLATFORM=offscreen,                  │
│          LIBGL_ALWAYS_SOFTWARE=1, MPLBACKEND=Agg — no X        │
│        · the CC3D 4.x Python steppable API (SteppableBasePy)   │
│          writes cells.csv every N Monte Carlo steps            │
│        · uploads cells.csv + status.json to the SAME prefix    │
│      instance scales back to zero when the queue drains        │
└────────────────────────────────────────────────────────────────┘
```

Details that matter when reading the code:

- **One job definition serves every run.** The container is addressed entirely
  through environment variables (`CC3D_BUCKET`, `CC3D_PREFIX`, `CC3D_RUN_ID`,
  optional `CC3D_STEPS`), passed as `containerOverrides.environment` at submit time,
  along with a `resourceRequirements` override (default 4 vCPU / 8192 MiB, from
  `remote_vcpus` / `remote_memory_mib` in the approach config).
- **The prefix is the contract.** Inputs and outputs share
  `s3://<bucket>/cc3d-runs/<run_id>/`. The runner skips `cells.csv` and
  `status.json` when downloading, so a retry does not ingest a previous attempt's
  output.
- **`status.json` is written on every outcome** — `starting`, `running`,
  `succeeded`, `failed` — so a failure has a reason even when CloudWatch is not
  reachable.
- **A missing `cells.csv` is a failure, not an empty result.** `fetch_results()`
  raises if the file is absent, even when the container exited zero. The runner
  never writes a partial or fabricated CSV on the success path.
- **Results are parsed by the same code as a local run.** `_run_remote()` writes the
  downloaded CSV to a temp file and calls the same `parse_output()`, so the viewer
  and every export behave identically regardless of where the simulation ran.
- **Progress is mapped from Batch states**, not invented: `SUBMITTED` 5%, `PENDING`
  10%, `RUNNABLE` 15%, `STARTING` 25%, `RUNNING` 50%, importing 85%.
- **Cancellation works.** `wait()` checks `context.is_cancelled()` between polls and
  calls `TerminateJob`/`CancelJob`. Pause does not — neither an external process nor
  a Batch job pauses, and `supports_pause=False` says so.

---

## 3. The cost model

### Why AWS Batch on EC2 spot, not Fargate

| | Batch on EC2 spot | Fargate |
|---|---|---|
| Image pull | Once per instance; a second job on the same warm instance reuses the local layer cache. | **Re-pulled per task, and pull time is billed.** At 2-3 GB that is paid on every single run. |
| Spot discount | Deep — commonly ~60-70% off on-demand for `m7i.xlarge` (estimate). | Shallower, and Fargate Spot has narrower capacity. |
| Instance shape | Choosable; `m7i.xlarge` (4 vCPU / 16 GiB) matches the 4 vCPU / 8192 MiB request. | vCPU/memory combinations are constrained. |
| Idle cost | Zero at `minvCpus=0`. | Zero, but the per-run pull tax is the dominant term for a 2-3 GB image. |

For a workload that is *rare, large-image, and short*, the repeated billed image
pull dominates, which is why Fargate is the wrong shape here.

### Why there is no idle compute charge

The compute environment is created with **`minvCpus = 0`** and
**`desiredvCpus = 0`**. Batch launches an instance when a job becomes `RUNNABLE`
and terminates it when the queue drains. Between runs the account owns **no running
instance**, so the compute line on the bill is genuinely zero — not small, zero.

### What CAN bill while nothing is running

| Item | Why it persists | Approximate monthly cost (estimate) |
|---|---|---|
| ECR image storage | The 2-3 GB worker image must stay pullable. | ~$0.10/GB-month → **~$0.20-0.40/mo** |
| Retained S3 objects | Each run keeps its project + `cells.csv` until deleted. `BatchRunner.cleanup(run_id)` deletes a run's objects; a lifecycle rule can expire the prefix. | ~$0.023/GB-month → **cents at this volume** |
| CloudWatch Logs | Job stdout under `/aws/batch/job`, **14-day retention policy**. | ~$0.50/GB ingest + ~$0.03/GB-month storage → **well under $1/mo** |

Realistic idle total: **under ~$1/month (estimate).** Everything else is per-run.

### What is deliberately NOT created

Each of these is a standing monthly charge that a conventional VPC build would have
added. None of them exists in this design.

| Not created | Monthly cost avoided (estimate) | Why it is unnecessary |
|---|---|---|
| **NAT Gateway** | **~$33/mo at zero traffic** (~$0.045/hr × 730 hr) plus ~$0.045/GB processed | The Batch instance sits in a **public subnet** with an **Internet Gateway** and a public IP. It needs egress to ECR, S3 and CloudWatch — an IGW provides that for free. |
| **Idle Elastic IP** | ~$0.005/hr → **~$3.60/mo** per unassociated address | Nothing needs a stable inbound address. The instance takes an ephemeral public IP and dies with the job. |
| **Interface VPC endpoints** | ~$0.01/hr **per endpoint per AZ** → **~$7-8/mo each**, multiplied by AZs and by services (ECR API, ECR DKR, Logs, Batch…) | The **free S3 Gateway endpoint** covers the S3 traffic — gateway endpoints have no hourly charge. Everything else reaches its public API through the IGW. |
| **Load balancer (ALB/NLB)** | ~$16-18/mo plus LCU charges | Nothing is served from AWS. Render terminates all HTTP. Batch jobs are dispatched, never addressed. |
| **RDS** | ~$13+/mo for the smallest always-on instance, plus storage | Run state lives in the app's existing store; job artifacts live in S3. |
| **ElastiCache** | ~$12+/mo for the smallest node | Polling `DescribeJobs` every 5 s needs no cache. |
| **EKS** | **~$73/mo control plane** ($0.10/hr) before any node | Batch *is* the scheduler. A Kubernetes control plane would be a second one. |
| **Any always-on instance** | An `m7i.xlarge` on-demand 24/7 is **~$147/mo** (~$0.2016/hr) | The engine is needed for minutes per run, not continuously. That is the whole premise. |

Networking summary: **public subnet + Internet Gateway instead of a NAT Gateway**,
and the **free S3 Gateway endpoint instead of a paid Interface endpoint**. Those two
choices remove roughly $40/month of fixed cost (estimate) from an otherwise
conventional design.

Per-run compute, for scale: `estimate_cost()` in
`D:\SURF\biosimulator\cc3d_remote.py` multiplies vCPUs × a spot vCPU-hour rate
(default $0.015, explicitly labelled an estimate) × hours. A 10-minute 4-vCPU run is
about **$0.01**. This is shown beside a queued job with `is_estimate: True` and
excludes S3 storage and data transfer.

---

## 4. How the submission path is authenticated

**This is a network-reachable way to spend money.** A request that reaches the
Render app can start a billable EC2 instance. Treat the submit path as a spend
control, not just a feature.

### IAM: what the Render app may do

The Render service holds credentials for one IAM principal scoped to exactly this:

| Action | Resource |
|---|---|
| `batch:SubmitJob` | **one** job queue ARN **and** **one** job definition ARN |
| `batch:DescribeJobs` | job ARNs in this account/region |
| `batch:TerminateJob` | job ARNs in this account/region |
| `s3:PutObject`, `s3:GetObject`, `s3:DeleteObject` | `arn:aws:s3:::<bucket>/cc3d-runs/*` |
| `s3:ListBucket` | `arn:aws:s3:::<bucket>` (prefix-conditioned) |

Nothing else. Concretely, the app **cannot**:

- create, modify or delete a compute environment — so it cannot create compute;
- call `UpdateComputeEnvironment` — so it **cannot raise `maxvCpus`**, and the
  ceiling you provision is the ceiling that holds;
- register a job definition, so it cannot point a job at a different image or
  attach a different execution role;
- read or write **any other bucket** — the resource ARN names one bucket, and the
  prefix condition narrows it to the run prefix;
- create IAM users, roles or keys, or touch EC2, RDS, or anything outside Batch/S3.

`SubmitJob` restricted to one queue + one definition means the worst case for a
leaked key is *jobs of the shape you already approved, on the queue whose
`maxvCpus` you already capped* — bounded, and visible in the Batch console.

### Credential handling

- Credentials live **only in Render's environment variables** (Render's
  environment-variable store / secret files). Nothing else on the instance holds
  them.
- **No AWS key is ever committed.** `.env` is gitignored; keep it that way and never
  paste a key into `Dockerfile`, a provisioning script, or this document. If a key
  is ever committed, rotate it — deleting the commit is not rotation.
- `cc3d_remote.py` creates **no AWS client at import**. `configuration_status()` is
  a pure environment read, so importing the module on a machine with no credentials
  is harmless, and the module never touches the network until a run is submitted.
- Rotate the access key on a schedule. Prefer a dedicated IAM user for Render with
  no console access.

### The app's own auth is the real gate

IAM bounds *what* can be spent; **the application's authentication decides *who*
can spend it.** An unauthenticated submit endpoint is a way for a stranger to spend
the project's money — every anonymous POST becomes a spot instance. Requirements:

1. The run-submission endpoint must require an authenticated session. No anonymous
   submits, no "unlisted URL" as a security control.
2. Rate-limit submissions per user, and cap concurrent CC3D runs per user.
3. Cap `maxvCpus` on the compute environment. That is the hard ceiling on
   simultaneous burn, and the app cannot raise it.

### Backstop: the budget alert

Set an **AWS Budgets alert at $20/month** with notifications at 50%, 80% and 100% of
forecast. It does not stop spending — it is the tripwire that tells you the model
was wrong (a runaway loop, a leaked key, a misconfigured retry) while the number is
still small. Given an expected idle cost under ~$1/month, a $20 alert fires long
before real damage.

---

## 5. Spot interruption

EC2 spot capacity can be reclaimed at any time. The sequence:

1. AWS issues a **2-minute interruption warning** (instance metadata / EventBridge).
2. The container receives **`SIGTERM`**.
3. **30 seconds later** the container receives **`SIGKILL`** and the instance is
   gone.

A Batch retry (`retryStrategy.attempts > 1`) starts the job **from scratch** — a new
container, an empty workdir, Monte Carlo step 0. Batch does not resume a partially
completed simulation, and CompuCell3D has no automatic restart-from-state here.

### Chosen strategy

1. **A 90-minute run-length cap.** A job whose projected wall clock exceeds 90
   minutes is **refused up front** with a clear message that names the limit and
   what to change (fewer Monte Carlo steps, a smaller lattice, or a lower
   measurement frequency). The refusal happens at submit time — before an instance
   is acquired and before anything is billed. Under 90 minutes, an interruption
   costs at most one repeat of a bounded amount of work.
2. **Partial-output upload on interruption.** The `SIGTERM` handler uploads whatever
   the steppable has written so far to **`cells.partial.csv`** in the run prefix,
   flushes a `status.json` marked interrupted, and exits inside the 30-second
   window. `cells.partial.csv` is deliberately a *different key* from `cells.csv`:
   the application treats a missing `cells.csv` as a hard failure, so partial data
   can never be mistaken for a completed run, but the researcher can still see how
   far it got and judge whether to resubmit.

### Why capping rather than checkpointing

- **Checkpointing CompuCell3D is not a small change.** A resumable checkpoint means
  serialising the full cell-field lattice, every cell's state, and the RNG stream,
  then restoring it in a way CC3D accepts as a valid start state. That is engine
  internals, version-sensitive, and easy to get subtly wrong.
- **A silently wrong resume is worse than a rerun.** If RNG state or contact-energy
  bookkeeping is restored incorrectly, the simulation still finishes and still
  produces a plausible CSV — a scientifically dishonest result, which is exactly the
  failure mode the adapter refuses elsewhere (it will not fall back to ABM for the
  same reason).
- **The economics do not justify it.** A capped run repeats at most 90 minutes of
  4-vCPU spot time — roughly **$0.05 (estimate)**. Checkpoint machinery would cost
  days of engineering and add a permanent correctness risk to save cents.
- **The cap is honest and immediate.** The user is told "too long, reduce it" before
  spending anything, rather than discovering after 4 hours that a resume was silently
  broken.

Configure the same bound in two places: Batch `timeout.attemptDurationSeconds = 5400`
(AWS stops the attempt) and the app-side `BIOSIM_CC3D_REMOTE_TIMEOUT` (the poller
gives up and terminates). Note the shipped default in `cc3d_remote.py` is
`DEFAULT_TIMEOUT_SECS = 7200` (2 hours) — set `BIOSIM_CC3D_REMOTE_TIMEOUT=5400` to
match the 90-minute policy.

---

## 6. Exact commands

Region is **us-east-2** throughout. Replace `<ACCOUNT_ID>` with the real 12-digit
account id. PowerShell, from the repo root `D:\SURF\biosimulator`.

### 6.1 Build and push the worker image to ECR

```powershell
$env:AWS_REGION = "us-east-2"
$ACCOUNT = "<ACCOUNT_ID>"
$REGISTRY = "$ACCOUNT.dkr.ecr.us-east-2.amazonaws.com"

# One-time: create the repository.
aws ecr create-repository --repository-name cc3d-worker --region us-east-2

# Build. Expect 10-25 minutes on a cold build: the conda solve for the
# compucell3d channel plus VTK is the slow part.
docker build -t cc3d-worker deploy/cc3d

# Authenticate Docker to ECR.
aws ecr get-login-password --region us-east-2 |
  docker login --username AWS --password-stdin $REGISTRY

# Tag and push (2-3 GB; the first push is slow).
docker tag cc3d-worker "$REGISTRY/cc3d-worker:latest"
docker push "$REGISTRY/cc3d-worker:latest"
```

The build fails loudly if the engine is missing: the Dockerfile runs
`python -c "import cc3d"` at build time, so a broken conda solve breaks the build
rather than the first simulation a researcher waits twenty minutes for.

### 6.2 Provision the AWS side

```powershell
cd deploy\cc3d
.\provision_aws.ps1 -Region us-east-2 -AccountId <ACCOUNT_ID> `
                    -ImageUri "<ACCOUNT_ID>.dkr.ecr.us-east-2.amazonaws.com/cc3d-worker:latest" `
                    -BucketName biosim-cc3d-<ACCOUNT_ID> `
                    -MaxVcpus 16 `
                    -BudgetUsd 20
```

`provision_aws.ps1` creates: the S3 bucket (private, versioning off, lifecycle rule
expiring `cc3d-runs/` after 30 days), a VPC with **one public subnet + Internet
Gateway**, the **free S3 Gateway endpoint**, a security group with no inbound rules,
the Batch instance role and the job role scoped to that one bucket, a **SPOT**
compute environment (`m7i.xlarge`, `minvCpus=0`, `desiredvCpus=0`,
`maxvCpus=16`), the job queue, the job definition (`attemptDurationSeconds=5400`,
`retryStrategy.attempts=2`), a CloudWatch Logs retention policy of **14 days** on
`/aws/batch/job`, the IAM user for Render with the scoped policy from §4, and the
**$20/month budget alert**. It prints the four environment variable values at the
end.

Verify before spending anything:

```powershell
aws batch describe-compute-environments --region us-east-2 `
  --query "computeEnvironments[].{name:computeEnvironmentName,state:state,status:status,min:computeResources.minvCpus,desired:computeResources.desiredvCpus,max:computeResources.maxvCpus,type:computeResources.type}"

aws batch describe-job-queues --region us-east-2 `
  --query "jobQueues[].{name:jobQueueName,state:state,status:status}"

# Expect min = 0 and desired = 0 — that is the no-idle-charge invariant.
```

### 6.3 Set the four environment variables on Render

In the Render dashboard → the web service → **Environment**:

```
BIOSIM_AWS_REGION         = us-east-2
BIOSIM_CC3D_JOB_QUEUE     = biosim-cc3d-queue
BIOSIM_CC3D_JOB_DEFINITION= biosim-cc3d-job
BIOSIM_CC3D_BUCKET        = biosim-cc3d-<ACCOUNT_ID>
```

Plus the credentials, as environment variables only, never in the repo:

```
AWS_ACCESS_KEY_ID         = <from provision_aws.ps1 output>
AWS_SECRET_ACCESS_KEY     = <from provision_aws.ps1 output>
```

Optional:

```
BIOSIM_CC3D_PREFIX          = cc3d-runs      # default
BIOSIM_CC3D_REMOTE_TIMEOUT  = 5400           # match the 90-minute cap
```

All four required variables must be set or `configuration_status()` reports
`available: False` and names exactly which are missing. Redeploy, then confirm the
CompuCell3D approach reports `engine_name: "CompuCell3D (AWS Batch)"` and
`method: "aws-batch"`.

### 6.4 Submit a test run

Cheapest possible check — a tiny lattice, few steps:

```powershell
aws batch submit-job --region us-east-2 `
  --job-name cc3d-smoke-test `
  --job-queue biosim-cc3d-queue `
  --job-definition biosim-cc3d-job `
  --container-overrides '{"environment":[{"name":"CC3D_BUCKET","value":"biosim-cc3d-<ACCOUNT_ID>"},{"name":"CC3D_PREFIX","value":"cc3d-runs/smoke-test"},{"name":"CC3D_RUN_ID","value":"smoke-test"},{"name":"CC3D_STEPS","value":"20"}],"resourceRequirements":[{"type":"VCPU","value":"4"},{"type":"MEMORY","value":"8192"}]}'
```

Upload a project to that prefix first, or expect the documented failure
(`No Simulation/model.xml in the uploaded project.`, exit 3) — which is itself a
valid test that the plumbing works:

```powershell
aws s3 cp .\Simulation\model.xml       s3://biosim-cc3d-<ACCOUNT_ID>/cc3d-runs/smoke-test/Simulation/model.xml
aws s3 cp .\Simulation\steppables.py   s3://biosim-cc3d-<ACCOUNT_ID>/cc3d-runs/smoke-test/Simulation/steppables.py
```

### 6.5 Watch it

```powershell
$JOB = "<jobId from submit-job>"

# State machine: SUBMITTED → PENDING → RUNNABLE → STARTING → RUNNING → SUCCEEDED
aws batch describe-jobs --region us-east-2 --jobs $JOB `
  --query "jobs[0].{status:status,reason:statusReason,exit:container.exitCode,stream:container.logStreamName}"

# Live stdout from the container.
aws logs tail /aws/batch/job --region us-east-2 --follow

# The instance Batch launched (should disappear after the queue drains).
aws ec2 describe-instances --region us-east-2 `
  --filters "Name=instance-state-name,Values=running" `
  --query "Reservations[].Instances[].{id:InstanceId,type:InstanceType,lifecycle:InstanceLifecycle}"

# Results.
aws s3 ls s3://biosim-cc3d-<ACCOUNT_ID>/cc3d-runs/smoke-test/
aws s3 cp s3://biosim-cc3d-<ACCOUNT_ID>/cc3d-runs/smoke-test/status.json -
```

Expect `cells.csv` and `status.json` with `"state": "succeeded"`. If the job
succeeded but `cells.csv` is absent, the measurement steppable did not run — the
runner reports that explicitly (exit 6) rather than writing an empty CSV.

### 6.6 Tear everything down

```powershell
cd deploy\cc3d
.\teardown_aws.ps1 -Region us-east-2 -AccountId <ACCOUNT_ID> -DeleteBucket
```

`teardown_aws.ps1` disables and deletes the job queue, disables and deletes the
compute environment (in that order — Batch refuses otherwise), deregisters the job
definition, deletes the ECR repository and its images, deletes the IAM user, policy
and access keys, deletes the VPC/subnet/IGW/gateway endpoint/security group, and
removes the budget. `-DeleteBucket` also empties and deletes the S3 bucket;
**omit it to keep past results.**

Confirm nothing is left billing:

```powershell
aws batch describe-compute-environments --region us-east-2 --query "computeEnvironments[].computeEnvironmentName"
aws ecr describe-repositories --region us-east-2 --query "repositories[].repositoryName"
aws ec2 describe-nat-gateways --region us-east-2 --query "NatGateways[?State!='deleted'].NatGatewayId"
aws ec2 describe-addresses --region us-east-2 --query "Addresses[].PublicIp"
aws ec2 describe-instances --region us-east-2 --filters "Name=instance-state-name,Values=running,pending" --query "Reservations[].Instances[].InstanceId"
```

All five should return empty. The NAT-gateway and Elastic-IP checks are there
because those are the two charges that most often survive a teardown unnoticed.

---

## 7. Limitations

Read this section before trusting any number above.

**This document describes a build that has not yet been executed on a live AWS
account.** No image has been pushed, no compute environment created, no job run. The
resource names, the ARNs, the timings and the costs are the *intended* design, not
observations. Expect the first real provisioning run to surface discrepancies.

**`provision_aws.ps1` and `teardown_aws.ps1` do not exist in this repository yet.**
Section 6 specifies what they must do; they are still to be written. The AWS-side
resources can be created by hand from the same specification in the meantime.

**The 90-minute cap and `cells.partial.csv` are policy, not yet code.** The current
`cc3d_job_runner.py` installs no `SIGTERM` handler and uploads no partial output, and
`cc3d_remote.py` still defaults to a 7200-second timeout. Section 5 describes the
chosen strategy; implementing it is outstanding work.

**Image size.** ~2-4 GB. It is large because CompuCell3D pulls VTK — which is
precisely why it does not belong in the web instance. Cold `docker build` is slow
(the conda solve dominates), and the first `docker push` moves the full image.

**Cold start.** A run does not begin immediately. Batch must find spot capacity for
an `m7i.xlarge`, boot it, and pull a 2-3 GB image before the container starts.
Budget **several minutes** before step 0 on a cold queue; the exact figure is
unmeasured because nothing has run yet. Back-to-back runs that land on a warm
instance skip the pull and start much faster. This is the price of `minvCpus=0`, and
it is a deliberate trade: minutes of latency per run in exchange for no idle spend.

**A Batch retry restarts from scratch.** Spot reclamation, an instance failure, or a
retry all mean the simulation begins again at Monte Carlo step 0. There is no
resume. With `retryStrategy.attempts=2` a run can therefore take up to twice the
expected wall clock and twice the compute cost.

**Spot capacity is not guaranteed.** `m7i.xlarge` spot in us-east-2 can be
unavailable, leaving a job `RUNNABLE` until capacity appears or the app-side timeout
terminates it. There is no automatic on-demand fallback.

**Pause is not supported.** `supports_pause=False`. A Batch job can be cancelled but
not suspended.

**Determinism.** `deterministic_with_seed=False`. CC3D's seeding behaviour is version
dependent, so two runs with the same seed are not guaranteed identical — and the
remote path is subject to whatever CC3D version the conda channel resolved at image
build time. Pin the image tag (not `:latest`) once a version is validated.

**No cross-region story.** Everything is us-east-2. Render egress to us-east-2 and
S3 transfer costs are not modelled here; at the volume of one CSV per run they are
negligible, but they are not zero.
