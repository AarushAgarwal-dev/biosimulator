# Least-privilege deployer identity for the CC3D AWS footprint

Everything in this directory currently runs as the **AWS account root principal**.
Verified, not assumed:

```
$ aws sts get-caller-identity
{ "UserId": "333308931113", "Account": "333308931113",
  "Arn": "arn:aws:iam::333308931113:root" }
```

Root cannot be scoped, cannot be bounded by a permissions boundary, and cannot be
denied anything — an SCP does not apply to it and neither does an IAM policy. One
mistake or one leaked credential is the whole account, billing and account closure
included. `aws_resources.json` even records `"provisionedBy": "...:root"`.

These files replace that with one IAM user whose permissions are the union of what
the scripts here actually call, and nothing else.

| File | What it is |
|---|---|
| `iam_policy_biosim_deployer.json` | Managed policy 1 of 2 — S3, ECR, Batch, CloudWatch Logs, CodeBuild, Budgets |
| `iam_policy_biosim_deployer_platform.json` | Managed policy 2 of 2 — IAM role management, EC2 networking, and the four `Deny` guardrails |
| `provision_iam_user.ps1` | Creates the user, publishes and attaches both policies, verifies by reading them back. Creates **no** access key. |

```powershell
.\provision_iam_user.ps1                 # plan only, changes nothing
.\provision_iam_user.ps1 -Confirm        # create / update, then verify
.\provision_iam_user.ps1 -VerifyOnly     # read back and diff only
```

## Why two policy documents and not one

An IAM managed policy is capped at **6144 characters, whitespace excluded**. The
single document that covers this footprint measured **9198** — 3054 over. The
options were to split it or to collapse statements into wildcards until it fit.
Collapsing would have meant replacing real ARNs with `*` for the sake of a
character count, which is the opposite of the point, so it is split at the natural
seam: *resources this project owns* versus *the IAM and network plumbing plus the
guardrails*. Both are attached to the same user, so the `Deny` statements in the
platform policy apply to everything the user can do.

Current sizes: 4087 / 6144 and 4693 / 6144. `provision_iam_user.ps1` re-checks
this before every publish and refuses to loosen a statement to make room.

## Where the permission list came from

Derived by reading `provision_aws.ps1`, `teardown_aws.ps1`,
`build_image_codebuild.ps1`, `buildspec.yml`, `cc3d_remote.py`,
`cc3d_job_runner.py` and `aws_resources.json`, then confirming every resource name
against the live account with read-only calls. 127 distinct CLI operations across
the three scripts; the runtime adds `batch:SubmitJob` / `DescribeJobs` /
`TerminateJob`, S3 object I/O, and `logs:GetLogEvents`.

Resource identities confirmed present in `us-east-2` / `333308931113`:

- Batch: compute environment `biosim-cc3d-ce`, queue `biosim-cc3d-queue`, job definition `biosim-cc3d-job:1`
- S3: `biosim-cc3d-333308931113-us-east-2` (run prefix `cc3d-runs/`, CodeBuild source `codebuild-source/cc3d-worker-src.zip`)
- ECR: `cc3d-worker`
- CodeBuild: `biosim-cc3d-image-build`
- Logs: `/aws/batch/job`, `/aws/codebuild/biosim-cc3d-image-build`
- IAM roles: `biosim-cc3d-ecsInstanceRole`, `biosim-cc3d-jobRole`, `biosim-cc3d-codebuild-role`; instance profile `biosim-cc3d-ecsInstanceProfile`
- EC2: VPC `vpc-089736efc61a057b1`, two subnets, IGW, route table, SG `biosim-cc3d-egress-only`, S3 gateway endpoint, launch template `biosim-cc3d-lt`

## Statement-by-statement

### `iam_policy_biosim_deployer.json`

| Sid | Scope | Notes |
|---|---|---|
| `ReadOwnIdentity` | `*` | `sts:GetCallerIdentity` takes no resource. Every script calls it as a preflight. |
| `RunBucketAdmin` | the one bucket ARN | `create-bucket`, `head-bucket`, encryption, versioning, lifecycle, tagging, public-access-block, `list-object-versions`. Bucket-level only — no object rights here. |
| `RunBucketObjects` | `bucket/*` | Object I/O for the run prefix, the CodeBuild source zip, and teardown's `delete-objects`. Versioned deletes included because the bucket has versioning suspended rather than never-enabled. |
| `WorkerImageRepoOnly` | the `cc3d-worker` repo ARN | Create/describe/delete, lifecycle policy, tags, image list/delete, plus the layer-upload verbs so the documented local `docker push` fallback still works. One repo, so a mistake cannot reach another. |
| `EcrRegistryAuthUnscopable` | `*` | **Unavoidable.** `ecr:GetAuthorizationToken` is a registry-level call; IAM documents it with no resource types, so an ARN here would deny every `docker login`. It is alone in its statement so nothing else rides on the wildcard. |
| `BatchStackMutations` | 4 ARNs | Create/update/delete the named CE, queue and job definition. `CreateJobQueue` needs the CE ARN as well as the queue ARN, which is why both appear. |
| `BatchSubmitOnlyThisQueueAndDefinition` | queue + job-definition ARNs | Lets the operator smoke-test a submission. Cannot submit to any other queue. |
| `BatchReadsAndTerminateUnscopable` | `*` | **Unavoidable.** `batch:DescribeJobs`, `DescribeComputeEnvironments`, `DescribeJobQueues`, `DescribeJobDefinitions`, `ListJobs`, `TerminateJob` and `CancelJob` are all documented as *Resource types: none*. A queue ARN here denies every call — the same reason the existing `biosim-cc3d-render-app` policy wildcards them. `batch:Describe*` is an action wildcard over a read-only family; enumerating the four names buys nothing but characters. |
| `StackLogGroupsOnly` | 4 log-group ARNs | `/aws/batch/job` and `/aws/codebuild/biosim-cc3d-image-build`, each with and without the `:*` suffix (the bare form authorizes group operations, the `:*` form the streams inside it). Covers `create-log-group`, `put-retention-policy`, `delete-log-group`, tag reads, and `aws logs tail`. |
| `EnumerateLogGroupsInThisAccountRegion` | `log-group:*` in this account and region | **Partial wildcard, named as one.** `logs:DescribeLogGroups` is how the scripts find out whether a group exists; it is a list call, so it cannot be scoped to the group being looked for. Narrowed to this account and region rather than `*`. |
| `ImageBuildProjectOnly` | the one project ARN | Create/update/delete the project, start and stop a build, read build status. Requires `iam:PassRole` on the CodeBuild service role — granted separately and conditioned. |
| `ListCodeBuildProjectsUnscopable` | `*` | **Unavoidable.** `codebuild:ListProjects` has no resource-level support. Read-only. |
| `CostGuardrailBudgetOnly` | the budget ARN | Budget ARNs are region-less (`arn:aws:budgets::333308931113:budget/...`) even though the API is called against `us-east-1`. Both the modern verbs and the legacy `ViewBudget`/`ModifyBudget` pair are present because AWS accepts either depending on CLI vintage. |

### `iam_policy_biosim_deployer_platform.json`

| Sid | Scope | Notes |
|---|---|---|
| `PassInstanceRoleToEc2Only` | `biosim-cc3d-ecsInstanceRole` + `iam:PassedToService = ec2.amazonaws.com` | Batch's compute environment takes the *instance profile*, but PassRole is evaluated against the role inside it. Three separate statements rather than one with three services, so each role can only be handed to the service that legitimately uses it. |
| `PassJobRoleToEcsTasksOnly` | `biosim-cc3d-jobRole` + `ecs-tasks.amazonaws.com` | The container job role. |
| `PassBuildRoleToCodeBuildOnly` | `biosim-cc3d-codebuild-role` + `codebuild.amazonaws.com` | Required by `codebuild:CreateProject`. |
| `StackIamObjectsOnly` | `role/biosim-cc3d-*`, `instance-profile/biosim-cc3d-*`, `policy/biosim-cc3d-*` | Role, instance-profile and managed-policy lifecycle for this stack's own objects, including `create-policy-version` for the `biosim-cc3d-render-app` policy that `provision_aws.ps1 -UpdateRenderPolicy` publishes. A name prefix, not `*` — it cannot touch `relay-*` or any other role in the account. |
| `AttachOnlyTheOneAwsManagedPolicy` | `role/biosim-cc3d-*` + `iam:PolicyARN` equals `AmazonEC2ContainerServiceforEC2Role` | `AttachRolePolicy` is a textbook escalation primitive — unconditioned, it could attach `AdministratorAccess` to a role this user can pass. The condition pins it to the single AWS managed policy `provision_aws.ps1` actually attaches. |
| `ReadServiceLinkedRoles` | `role/aws-service-role/*` | `provision_aws.ps1` probes for `AWSServiceRoleForBatch` with `iam:get-role`; its ARN lives under `role/aws-service-role/batch.amazonaws.com/`, so the prefix is required or provisioning fails at that step. Read-only. |
| `CreateServiceLinkedRolesForBatchStackOnly` | `role/aws-service-role/*` + `iam:AWSServiceName` in a five-item list | Service-linked role ARNs are chosen by AWS, so the resource cannot be a literal ARN. The condition is the real control: only Batch, ECS, Spot, Spot Fleet and Auto Scaling. |
| `Ec2ReadsUnscopable` | `*` | **Unavoidable.** No EC2 `Describe*` action supports resource-level permissions — this is an IAM-wide property of the EC2 API, not a shortcut. The scripts call 14 of them. Read-only, and the region `Deny` below still confines it. |
| `Ec2CreateBrandNewObjectsInRegion` | `*` + `aws:RequestedRegion = us-east-2` | **Unavoidable for the ARN.** At authorization time the VPC, subnet, IGW, route table, security group, endpoint and launch template do not exist, so there is no ARN to name. A `aws:RequestTag` condition was considered and rejected: `provision_aws.ps1` creates first and tags in a separate `ec2:CreateTags` call immediately after (`Add-Ec2Tags`), so a request-tag requirement would fail every create. `ec2:CreateTags` itself must be here for the same reason — the resource is untagged at the instant it is tagged. `ModifyVpcEndpoint` is here rather than tag-scoped because it runs against a just-created endpoint. |
| `Ec2MutateTaggedStackObjectsOnly` | `*` + `ec2:ResourceTag/Project = BioSimulateAI` + region | Every destructive EC2 action. The resource is `*` but the **condition is the scope**: the call only authorizes against objects already carrying this project's tag. Chosen over the 8 hardcoded resource ids in `aws_resources.json` deliberately — hardcoded ids stop matching after a teardown/re-provision cycle, and a policy that silently stops working is a policy someone widens to `*`. Verified safe: `Add-Ec2Tags` runs before any modify or delete. |
| `DenyTouchingTheDeployerItself` | `user/biosim-cc3d-deployer`, `policy/biosim-cc3d-deployer-*` | `StackIamObjectsOnly` would otherwise match the deployer's own two policies through the `biosim-cc3d-*` prefix. An identity that can rewrite its own policy is not bounded by it. Consequence: **run `provision_iam_user.ps1` from an admin session, not as the deployer.** |
| `DenyMintingCredentialsOrIdentities` | `*` | Closes the credential-minting escalation paths: new users, access keys, login profiles, group and user policy attachment, `UpdateAssumeRolePolicy` (rewriting a trust policy to make a role assumable), SAML/OIDC providers, `sts:GetFederationToken`, MFA deactivation, and Organizations/Account APIs. None of these appear anywhere in the repo, so denying them costs nothing. Explicit `Deny` rather than absence of `Allow`, so a future edit that adds a broad `Allow` cannot re-open them. |
| `DenyAlwaysBillingCapacity` | `*` | The project's measured idle cost is ~$0.54/month (an earlier ~$0.35 figure was corrected upward after three untagged 1.3 GB ECR images turned up), and `provision_aws.ps1` ends by scanning for NAT gateways, Elastic IPs, interface endpoints, load balancers and running instances. This makes that guard enforceable instead of advisory. Safe because Batch launches Spot instances through `AWSServiceRoleForBatch`, never through this user — nothing in the repo calls `ec2:RunInstances` directly. |
| `DenyOutsideStackRegions` | `*` except global services, when `aws:RequestedRegion` is neither `us-east-2` nor `us-east-1` | Confines the blast radius of every wildcard above to one region. `us-east-1` is included because AWS Budgets is called there (`$BudgetRegion = 'us-east-1'`). IAM, STS, S3, Cost Explorer, Organizations, Support and Health are excluded via `NotAction` because they are global endpoints whose `aws:RequestedRegion` does not behave regionally. |
| `StackInventoryForTeardown` | `*` + region | `tag:GetResources`, added after review. **This was missing from the first draft** and the omission mattered more than its size suggests: `teardown_aws.ps1` calls `aws resourcegroupstaggingapi get-resources` in three places (lines 29, 489, 1359) to build its inventory, so without this grant the scoped user could PROVISION the stack but never tear it down — and teardown is the operation that stops the spending. The gap was found by enumerating every AWS service invoked across `provision_aws.ps1`, `teardown_aws.ps1`, `build_image_codebuild.ps1`, `cc3d_remote.py` and `cc3d_job_runner.py` and checking each against the granted actions, rather than by re-reading the policy. |

## Every wildcard, in one list

Nothing below is scoped, and calling it scoped would be a lie:

| Wildcard | Why it cannot be narrowed |
|---|---|
| `ecr:GetAuthorizationToken` on `*` | Registry-level action, documented with no resource types. |
| `batch:Describe*`, `ListJobs`, `TerminateJob`, `CancelJob` on `*` | AWS documents these as *Resource types: none*. A queue or job ARN denies the call outright. |
| `ec2:Describe*` on `*` | No EC2 describe action supports resource-level permissions. |
| `tag:GetResources` on `*` | An account-and-region-wide query by its nature — it exists to discover resources, so it cannot be scoped to the resources it has not found yet. Bounded by the region condition, and read-only. |
| `codebuild:ListProjects` on `*` | No resource-level support. |
| EC2 create actions on `*` | The resource does not exist when the request is authorized, so no ARN exists to name. Bounded by `aws:RequestedRegion`. |
| `ec2:CreateTags` on `*` | Must tag a resource that is still untagged. Bounded by region. Cannot use `ec2:CreateAction` because this repo tags in a separate call after create. |
| EC2 mutate/delete on `*` | Resource is `*`, but the `ec2:ResourceTag/Project` condition is the actual scope. |
| `logs:DescribeLogGroups` on `log-group:*` | A list call cannot be scoped to the item being looked for. Narrowed to this account and region. |
| `Deny` statements on `*` | Intentional: a guardrail is supposed to be broad. |

## AWS APIs found in the repo that are **not** granted

Deliberate omissions, listed so nobody assumes they were missed:

- `elasticloadbalancing:DescribeLoadBalancers` — **granted** (statement `CostGuardLoadBalancerCheck`), after this was initially left out. `provision_aws.ps1:1411` calls it in the final cost-guard scan with `-AllowFailure`, so omitting it did not break provisioning — the guard just printed a warning instead of running. That is the wrong trade for this project: an idle load balancer is roughly $16/month against a $5 ceiling, so the one check that would catch it must not silently degrade into a warning. Granted as the SINGLE read action rather than `elasticloadbalancing:Describe*`, which was the reason it was skipped, and bounded by the region condition. Creating a load balancer stays denied outright by `DenyAlwaysBillingCapacity`, so the deployer can see one in order to warn about it and can never make one.
- `iam:CreateAccessKey` and friends — explicitly denied. See below.
- `bedrock-runtime` (`llm_provider.py`) — the application's LLM path, unrelated to deploying the CC3D footprint. It belongs on the app's own identity, not the deployer's.
- `ec2:RunInstances` — Batch does this through its service-linked role.

## Access keys: not created here

`provision_iam_user.ps1` never calls `aws iam create-access-key`, and the
verification harness asserts that mechanically rather than trusting the prose.
`create-access-key` returns the secret exactly once, in the response body; this
script's output goes to a terminal that is routinely captured — a PowerShell
transcript, a CI log, a scrollback, an agent session record. A secret written to
any of those is leaked from that moment, and rotating it afterwards does not
un-leak it.

The script prints the exact commands instead: `aws iam create-access-key
--user-name biosim-cc3d-deployer`, then `aws configure --profile
biosim-cc3d-deployer`, then `aws sts get-caller-identity --profile ...` to prove
the profile resolves to the user and not to root. Every script here already takes
`-AwsProfile`.

## Retiring the root credentials

The script also prints the console steps, because this cannot be done from a
session authenticated with the keys being deleted, and no IAM policy restrains
root. Observed state at the time of writing, from `iam get-account-summary`:

- `AccountAccessKeysPresent = 0` — **the root user has no long-lived access keys.**
- The active credentials report `TYPE : login` in `aws configure list`, i.e. a
  temporary root *session*, not a stored root key pair.

So the exposure here is root *privilege* on every provisioning call, not a stored
root key sitting in `~/.aws/credentials`. The fix is the same — stop provisioning
as root — but do not go hunting in the console for a key that is not there.
Confirm with `aws iam get-account-summary --query
'SummaryMap.AccountAccessKeysPresent'` (expect `0`) and enable root MFA if
`AccountMFAEnabled` is `0`. The script checks and reports both on every run.

## Residual risk, stated plainly

The deployer manages the IAM roles it also passes to Batch and CodeBuild. It can
therefore write an inline policy onto `biosim-cc3d-jobRole` and run a container as
that role — a privilege-escalation path that **every** identity which provisions
its own service roles has. `AttachRolePolicy` is conditioned to one managed policy
to close the easy version, but `PutRolePolicy` on `role/biosim-cc3d-*` remains
open because `provision_aws.ps1` writes the bucket-scoped inline policy with it.

Practical consequence: this is a **human-operated provisioning identity**, not a
credential for an unattended pipeline. If it ever needs to run unattended, attach
a permissions boundary to the roles it may create and add an
`iam:PermissionsBoundary` condition to `iam:CreateRole` — that condition is
omitted today because `provision_aws.ps1` does not pass `--permissions-boundary`,
and adding the condition without changing that script would break provisioning on
a fresh account.

## Verification performed

```
scriptblock::Create      : parsed ok
AST parse                : 0 errors, 2686 tokens
credential-creating calls: NONE (verified)
iam_policy_biosim_deployer.json           ok  stmts=13 (allow 13 / deny 0)  chars=4087/6144
iam_policy_biosim_deployer_platform.json  ok  stmts=14 (allow 10 / deny 4)  chars=4693/6144
```

Not verified: whether IAM accepts both documents, and whether the resulting user
can complete a real provision/teardown cycle. Nothing was applied — no user,
policy, key or AWS resource was created, modified or deleted. Run
`.\provision_iam_user.ps1` (plan mode) and then `-Confirm` from an admin session
to find out, and expect one or two `AccessDenied` errors on the first real
`provision_aws.ps1` run; each one names the exact missing action, which is the
cheapest way to close the last gaps.
