<#
===============================================================================
 provision_iam_user.ps1 -- create the scoped BioSimulateAI deployer identity

 WHY THIS EXISTS
   Every provisioning call in this project currently runs as the AWS ACCOUNT ROOT
   principal (verified: "aws sts get-caller-identity" returns
   arn:aws:iam::333308931113:root). Root cannot be scoped, cannot be bounded by a
   permissions boundary, and cannot be denied anything -- an SCP does not apply to
   it and neither does an IAM policy. So a single mistake or a single leaked
   credential is the whole account, including billing and account closure.

   This script creates ONE IAM user whose permissions are the union of what the
   scripts in this directory actually call -- derived by reading provision_aws.ps1,
   teardown_aws.ps1, build_image_codebuild.ps1, buildspec.yml, cc3d_remote.py and
   deploy/cc3d/cc3d_job_runner.py -- and nothing else.

 WHAT IT DELIBERATELY DOES NOT DO
   It never calls "aws iam create-access-key". The secret access key is returned
   exactly once, in the API response, and this script's output goes to a terminal
   that is usually captured -- a transcript, a CI log, an agent session record.
   Printing a live credential into a log is how credentials leak. The exact
   commands for the operator to run are printed at the end instead.

 USAGE
   .\provision_iam_user.ps1                  # plan only -- creates nothing
   .\provision_iam_user.ps1 -Confirm         # create / update, then verify
   .\provision_iam_user.ps1 -VerifyOnly      # read back and diff, change nothing

 IDEMPOTENT: safe to re-run. An existing user is reused, an existing policy whose
 document already matches is left alone, and a drifted policy gets a new default
 version (pruning the oldest non-default version first, because IAM caps a managed
 policy at five versions).

 RUN THIS AS AN ADMIN / ROOT SESSION. The deployer it creates is explicitly denied
 write access to its own two policies, so the deployer cannot re-run this script
 against itself. That is the point: an identity that can rewrite its own policy is
 not bounded by it.
===============================================================================
#>

[CmdletBinding()]
param(
    [string]$Region     = 'us-east-2',
    [string]$NamePrefix = 'biosim-cc3d',
    [string]$UserName   = 'biosim-cc3d-deployer',
    [string]$AwsProfile = '',
    [switch]$Confirm,
    [switch]$VerifyOnly
)

# The AWS CLI writes expected "NoSuchEntity" text to stderr; under
# ErrorActionPreference='Stop' that would abort the script mid-way. Exit codes are
# checked explicitly instead. Same convention as provision_aws.ps1.
$ErrorActionPreference = 'Continue'
$ProgressPreference    = 'SilentlyContinue'

$TagProject   = 'BioSimulateAI'
$TagComponent = 'CC3D-Compute'
$ScriptDir    = Split-Path -Parent $MyInvocation.MyCommand.Path

$StackPolicyName    = "$NamePrefix-deployer-policy"
$PlatformPolicyName = "$NamePrefix-deployer-platform"
$StackPolicyFile    = Join-Path $ScriptDir 'iam_policy_biosim_deployer.json'
$PlatformPolicyFile = Join-Path $ScriptDir 'iam_policy_biosim_deployer_platform.json'

# IAM hard limit on a managed policy document, whitespace excluded.
$IamPolicyCharLimit = 6144

$script:Actions = New-Object System.Collections.Generic.List[string]

# -----------------------------------------------------------------------------
# Output helpers
# -----------------------------------------------------------------------------
function Write-Section {
    param([string]$Text)
    Write-Host ''
    Write-Host ('=' * 79) -ForegroundColor DarkCyan
    Write-Host "  $Text" -ForegroundColor Cyan
    Write-Host ('=' * 79) -ForegroundColor DarkCyan
}
function Write-Step { param([string]$Text) Write-Host ''; Write-Host "-> $Text" -ForegroundColor White }
function Write-Made { param([string]$Text) Write-Host "     + created  $Text" -ForegroundColor Green;    $script:Actions.Add("created  $Text") }
function Write-Chg  { param([string]$Text) Write-Host "     ~ updated  $Text" -ForegroundColor Yellow;   $script:Actions.Add("updated  $Text") }
function Write-Have { param([string]$Text) Write-Host "     = exists   $Text" -ForegroundColor DarkGray; $script:Actions.Add("unchanged $Text") }
function Write-Info { param([string]$Text) Write-Host "       $Text" -ForegroundColor DarkGray }
function Write-Ok   { param([string]$Text) Write-Host "     OK  $Text" -ForegroundColor Green }
function Write-Bad  { param([string]$Text) Write-Host "     !!  $Text" -ForegroundColor Red }
function Write-Warn2 { param([string]$Text) Write-Host "     warning: $Text" -ForegroundColor Yellow }

function Stop-WithError {
    param([string]$Message, [string]$Detail = '')
    Write-Host ''
    Write-Host "FAILED: $Message" -ForegroundColor Red
    if ($Detail) { Write-Host $Detail -ForegroundColor DarkYellow }
    exit 1
}

# -----------------------------------------------------------------------------
# AWS CLI wrapper. IAM is a global service, so no --region is passed for iam calls
# unless the caller asks for it.
# -----------------------------------------------------------------------------
function Invoke-Aws {
    param(
        [Parameter(Mandatory = $true)][string[]]$AwsArgs,
        [string]$Context = '',
        [switch]$AllowFailure
    )
    $full = @() + $AwsArgs
    if ($AwsProfile) { $full += @('--profile', $AwsProfile) }

    $raw  = & aws.exe @full 2>&1
    $code = $LASTEXITCODE
    $text = ($raw | Out-String).Trim()

    if ($code -ne 0 -and -not $AllowFailure) {
        Stop-WithError -Message "aws $($AwsArgs -join ' ') failed (exit $code)." -Detail $text
    }
    return [pscustomobject]@{
        ExitCode = $code
        Text     = $text
        Command  = "aws $($AwsArgs -join ' ')"
        Context  = $Context
    }
}

function Get-AwsJson {
    param([string[]]$AwsArgs, [string]$Context = '', [switch]$AllowFailure)
    $r = Invoke-Aws -AwsArgs ($AwsArgs + @('--output', 'json')) -Context $Context -AllowFailure:$AllowFailure
    if ($r.ExitCode -ne 0) { return $null }
    try { return ($r.Text | ConvertFrom-Json) } catch { return $null }
}

# -----------------------------------------------------------------------------
# Policy document handling
# -----------------------------------------------------------------------------
function Read-PolicyFile {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) {
        Stop-WithError -Message "Policy document not found: $Path"
    }
    $raw = Get-Content -Raw -LiteralPath $Path
    try { $obj = $raw | ConvertFrom-Json }
    catch { Stop-WithError -Message "Policy document is not valid JSON: $Path" -Detail $_.Exception.Message }
    if ([string]$obj.Version -ne '2012-10-17') {
        Stop-WithError -Message "Policy document $Path has Version '$($obj.Version)'; expected 2012-10-17."
    }
    if (-not $obj.Statement -or @($obj.Statement).Count -eq 0) {
        Stop-WithError -Message "Policy document $Path has no statements."
    }
    $chars = ($raw -replace '\s', '').Length
    if ($chars -gt $IamPolicyCharLimit) {
        Stop-WithError -Message "Policy document $Path is $chars characters; IAM's managed-policy limit is $IamPolicyCharLimit (whitespace excluded)." `
                       -Detail 'Split it into a second managed policy attached to the same user rather than loosening the statements to make it fit.'
    }
    return [pscustomobject]@{ Raw = $raw; Object = $obj; Chars = $chars; Path = $Path }
}

# A canonical, order-insensitive fingerprint of a policy document. IAM does not
# promise to return statements, actions or resources in the order they were sent,
# so a raw string comparison reports drift that is not there.
function Get-PolicyFingerprint {
    param($PolicyObject)
    $lines = New-Object System.Collections.Generic.List[string]
    foreach ($s in @($PolicyObject.Statement)) {
        $sid       = [string]$s.Sid
        $effect    = [string]$s.Effect
        $actions   = (@($s.Action)    | Sort-Object) -join ','
        $notAction = (@($s.NotAction) | Sort-Object) -join ','
        $resources = (@($s.Resource)  | Sort-Object) -join ','
        $cond = ''
        if ($s.PSObject.Properties.Name -contains 'Condition' -and $null -ne $s.Condition) {
            $cond = ($s.Condition | ConvertTo-Json -Depth 12 -Compress)
        }
        $lines.Add("$sid|$effect|$actions|$notAction|$resources|$cond")
    }
    $sorted = @($lines) | Sort-Object
    return ($sorted -join "`n")
}

function Test-PolicyMatches {
    param($LocalObject, $RemoteObject)
    return ((Get-PolicyFingerprint -PolicyObject $LocalObject) -eq (Get-PolicyFingerprint -PolicyObject $RemoteObject))
}

# Ensure one managed policy exists with the document from disk as its DEFAULT
# version. Returns the policy ARN.
function Set-ManagedPolicy {
    param(
        [string]$PolicyName,
        [string]$PolicyArn,
        $Local,
        [switch]$Apply
    )

    $existing = Get-AwsJson -AwsArgs @('iam', 'get-policy', '--policy-arn', $PolicyArn) -Context 'iam get-policy' -AllowFailure

    if ($null -eq $existing) {
        if (-not $Apply) {
            Write-Info "WOULD CREATE managed policy $PolicyName ($($Local.Chars) chars, $(@($Local.Object.Statement).Count) statements)"
            return $PolicyArn
        }
        $createArgs = @(
            'iam', 'create-policy',
            '--policy-name', $PolicyName,
            '--policy-document', ("file://" + $Local.Path),
            '--description', 'BioSimulateAI CC3D deployer: least-privilege replacement for root access',
            '--tags', "Key=Project,Value=$TagProject", "Key=Component,Value=$TagComponent"
        )
        $created = Get-AwsJson -AwsArgs $createArgs -Context 'iam create-policy'
        if ($null -eq $created) { Stop-WithError -Message "iam create-policy returned nothing for $PolicyName." }
        Write-Made "managed policy $PolicyArn ($($Local.Chars) chars)"
        return $PolicyArn
    }

    # The policy exists. Compare its default version against the file on disk.
    $defaultVersionId = [string]$existing.Policy.DefaultVersionId
    $remote = Get-AwsJson -AwsArgs @('iam', 'get-policy-version', '--policy-arn', $PolicyArn, '--version-id', $defaultVersionId) `
                          -Context 'iam get-policy-version' -AllowFailure
    if ($null -eq $remote) {
        Write-Warn2 "could not read $PolicyArn version $defaultVersionId; leaving it alone"
        return $PolicyArn
    }

    if (Test-PolicyMatches -LocalObject $Local.Object -RemoteObject $remote.PolicyVersion.Document) {
        Write-Have "managed policy $PolicyArn (default version $defaultVersionId already matches the file on disk)"
        return $PolicyArn
    }

    Write-Info "$PolicyName has drifted from the file on disk"
    if (-not $Apply) {
        Write-Info "WOULD PUBLISH a new default version of $PolicyName"
        return $PolicyArn
    }

    # IAM allows five versions per managed policy. Prune the oldest non-default
    # one first, or create-policy-version fails with LimitExceeded.
    $versions = Get-AwsJson -AwsArgs @('iam', 'list-policy-versions', '--policy-arn', $PolicyArn) -Context 'iam list-policy-versions' -AllowFailure
    if ($null -ne $versions) {
        $nonDefault = @($versions.Versions | Where-Object { -not $_.IsDefaultVersion })
        if ($nonDefault.Count -ge 4) {
            $oldest = @($nonDefault | Sort-Object -Property CreateDate)[0]
            Invoke-Aws -AwsArgs @('iam', 'delete-policy-version', '--policy-arn', $PolicyArn, '--version-id', [string]$oldest.VersionId) `
                       -Context 'iam delete-policy-version' | Out-Null
            Write-Info "pruned old policy version $($oldest.VersionId) to stay under IAM's five-version cap"
        }
    }

    Invoke-Aws -AwsArgs @(
        'iam', 'create-policy-version',
        '--policy-arn', $PolicyArn,
        '--policy-document', ("file://" + $Local.Path),
        '--set-as-default'
    ) -Context 'iam create-policy-version' | Out-Null
    Write-Chg "managed policy $PolicyArn (new default version published)"
    return $PolicyArn
}

# =============================================================================
# Identity and preflight
# =============================================================================
Write-Section 'BioSimulateAI deployer IAM user'

Write-Step 'Caller identity'
$who = Get-AwsJson -AwsArgs @('sts', 'get-caller-identity') -Context 'sts get-caller-identity' -AllowFailure
if ($null -eq $who) {
    Stop-WithError -Message 'aws sts get-caller-identity failed -- there is no usable AWS session.' `
                   -Detail 'Configure credentials first (aws configure, or aws sso login), then re-run.'
}
$AccountId = [string]$who.Account
$CallerArn = [string]$who.Arn
Write-Info "account $AccountId"
Write-Info "caller  $CallerArn"

if ($CallerArn -match ':root$') {
    Write-Info 'this session IS the account root -- which is exactly what this script exists to stop using'
} else {
    Write-Warn2 "this session is not root. It needs iam:CreateUser, iam:CreatePolicy and iam:AttachUserPolicy to succeed."
}

$StackPolicyArn    = "arn:aws:iam::${AccountId}:policy/$StackPolicyName"
$PlatformPolicyArn = "arn:aws:iam::${AccountId}:policy/$PlatformPolicyName"
$UserArn           = "arn:aws:iam::${AccountId}:user/$UserName"

Write-Step 'Policy documents on disk'
$stackLocal    = Read-PolicyFile -Path $StackPolicyFile
$platformLocal = Read-PolicyFile -Path $PlatformPolicyFile
Write-Info ("{0}: {1} statements, {2}/{3} chars" -f (Split-Path $StackPolicyFile -Leaf), @($stackLocal.Object.Statement).Count, $stackLocal.Chars, $IamPolicyCharLimit)
Write-Info ("{0}: {1} statements, {2}/{3} chars" -f (Split-Path $PlatformPolicyFile -Leaf), @($platformLocal.Object.Statement).Count, $platformLocal.Chars, $IamPolicyCharLimit)

$apply = [bool]$Confirm
if ($VerifyOnly) { $apply = $false }

if (-not $apply) {
    Write-Section 'PLAN ONLY -- nothing will be created or changed'
    Write-Info "user           $UserArn"
    Write-Info "policy         $StackPolicyArn"
    Write-Info "policy         $PlatformPolicyArn"
    Write-Info 'no access key will be created in either mode'
    Write-Host ''
    Write-Host "  Re-run with -Confirm to apply:  .\provision_iam_user.ps1 -Confirm" -ForegroundColor Yellow
}

# =============================================================================
# 1. The IAM user
# =============================================================================
Write-Step "IAM user $UserName"
$userLookup = Get-AwsJson -AwsArgs @('iam', 'get-user', '--user-name', $UserName) -Context 'iam get-user' -AllowFailure
if ($null -ne $userLookup) {
    Write-Have "user $UserArn"
} elseif ($apply) {
    $mkUser = Get-AwsJson -AwsArgs @(
        'iam', 'create-user',
        '--user-name', $UserName,
        '--tags', "Key=Project,Value=$TagProject", "Key=Component,Value=$TagComponent"
    ) -Context 'iam create-user'
    if ($null -eq $mkUser) { Stop-WithError -Message 'iam create-user returned nothing.' }
    Write-Made "user $UserArn (no console password, no access key)"
} else {
    Write-Info "WOULD CREATE user $UserArn"
}

# =============================================================================
# 2. The two managed policies
# =============================================================================
Write-Step "Managed policy $StackPolicyName (S3 / ECR / Batch / Logs / CodeBuild / Budgets)"
Set-ManagedPolicy -PolicyName $StackPolicyName -PolicyArn $StackPolicyArn -Local $stackLocal -Apply:$apply | Out-Null

Write-Step "Managed policy $PlatformPolicyName (IAM roles / EC2 network / deny guardrails)"
Set-ManagedPolicy -PolicyName $PlatformPolicyName -PolicyArn $PlatformPolicyArn -Local $platformLocal -Apply:$apply | Out-Null

# =============================================================================
# 3. Attach them
# =============================================================================
Write-Step 'Attaching the policies to the user'
$attached = Get-AwsJson -AwsArgs @('iam', 'list-attached-user-policies', '--user-name', $UserName) -Context 'iam list-attached-user-policies' -AllowFailure
$attachedArns = @()
if ($null -ne $attached) { $attachedArns = @($attached.AttachedPolicies | ForEach-Object { [string]$_.PolicyArn }) }

foreach ($arn in @($StackPolicyArn, $PlatformPolicyArn)) {
    if ($attachedArns -contains $arn) {
        Write-Have "attachment $arn"
    } elseif ($apply) {
        Invoke-Aws -AwsArgs @('iam', 'attach-user-policy', '--user-name', $UserName, '--policy-arn', $arn) -Context 'iam attach-user-policy' | Out-Null
        Write-Made "attachment $arn"
    } else {
        Write-Info "WOULD ATTACH $arn"
    }
}

# =============================================================================
# 4. VERIFY by reading everything back
# =============================================================================
Write-Section 'VERIFICATION -- reading the result back from IAM'

$verifyFailures = @()

Write-Step 'User'
$vUser = Get-AwsJson -AwsArgs @('iam', 'get-user', '--user-name', $UserName) -Context 'verify iam get-user' -AllowFailure
if ($null -eq $vUser) {
    if ($apply) { $verifyFailures += "user $UserName does not exist" ; Write-Bad "user $UserName not found" }
    else { Write-Info "user $UserName does not exist yet (plan mode)" }
} else {
    Write-Ok "user $($vUser.User.Arn) created $($vUser.User.CreateDate)"
}

Write-Step 'Attached policies'
$vAttached = Get-AwsJson -AwsArgs @('iam', 'list-attached-user-policies', '--user-name', $UserName) -Context 'verify list-attached-user-policies' -AllowFailure
$vArns = @()
if ($null -ne $vAttached) { $vArns = @($vAttached.AttachedPolicies | ForEach-Object { [string]$_.PolicyArn }) }
foreach ($arn in @($StackPolicyArn, $PlatformPolicyArn)) {
    if ($vArns -contains $arn) { Write-Ok "attached $arn" }
    elseif ($apply) { $verifyFailures += "policy $arn is not attached"; Write-Bad "not attached $arn" }
    else { Write-Info "not attached yet $arn (plan mode)" }
}
$unexpected = @($vArns | Where-Object { $_ -ne $StackPolicyArn -and $_ -ne $PlatformPolicyArn })
foreach ($u in $unexpected) {
    Write-Warn2 "unexpected extra policy attached to this user: $u -- it widens the deployer beyond this repo's footprint"
}

Write-Step 'Inline user policies (there should be none)'
$vInline = Get-AwsJson -AwsArgs @('iam', 'list-user-policies', '--user-name', $UserName) -Context 'verify list-user-policies' -AllowFailure
if ($null -ne $vInline) {
    $inlineNames = @($vInline.PolicyNames)
    if ($inlineNames.Count -eq 0) { Write-Ok 'no inline user policies' }
    else { foreach ($n in $inlineNames) { Write-Warn2 "inline user policy present: $n -- not managed by this script" } }
}

Write-Step 'Policy documents match the files on disk'
foreach ($pair in @(
        @{ Name = $StackPolicyName;    Arn = $StackPolicyArn;    Local = $stackLocal },
        @{ Name = $PlatformPolicyName; Arn = $PlatformPolicyArn; Local = $platformLocal })) {

    $pv = Get-AwsJson -AwsArgs @('iam', 'get-policy', '--policy-arn', $pair.Arn) -Context 'verify iam get-policy' -AllowFailure
    if ($null -eq $pv) {
        if ($apply) { $verifyFailures += "policy $($pair.Name) does not exist"; Write-Bad "policy $($pair.Name) not found" }
        else { Write-Info "policy $($pair.Name) does not exist yet (plan mode)" }
        continue
    }
    $vid = [string]$pv.Policy.DefaultVersionId
    $doc = Get-AwsJson -AwsArgs @('iam', 'get-policy-version', '--policy-arn', $pair.Arn, '--version-id', $vid) -Context 'verify iam get-policy-version' -AllowFailure
    if ($null -eq $doc) {
        $verifyFailures += "could not read $($pair.Name) version $vid"
        Write-Bad "could not read $($pair.Name) version $vid"
        continue
    }
    $remoteDoc   = $doc.PolicyVersion.Document
    $remoteStmts = @($remoteDoc.Statement).Count
    if (Test-PolicyMatches -LocalObject $pair.Local.Object -RemoteObject $remoteDoc) {
        Write-Ok "$($pair.Name) default version $vid matches the file on disk ($remoteStmts statements)"
    } else {
        $verifyFailures += "$($pair.Name) in IAM differs from the file on disk"
        Write-Bad "$($pair.Name) default version $vid DIFFERS from the file on disk"
        $localSids  = @($pair.Local.Object.Statement | ForEach-Object { [string]$_.Sid }) | Sort-Object
        $remoteSids = @($remoteDoc.Statement          | ForEach-Object { [string]$_.Sid }) | Sort-Object
        $onlyLocal  = @($localSids  | Where-Object { $remoteSids -notcontains $_ })
        $onlyRemote = @($remoteSids | Where-Object { $localSids  -notcontains $_ })
        foreach ($s in $onlyLocal)  { Write-Info "  only in the file : $s" }
        foreach ($s in $onlyRemote) { Write-Info "  only in IAM      : $s" }
        if ($onlyLocal.Count -eq 0 -and $onlyRemote.Count -eq 0) {
            Write-Info '  same statement Sids, but an Action / Resource / Condition differs'
        }
    }
}

Write-Step 'Access keys on this user (this script creates none)'
$vKeys = Get-AwsJson -AwsArgs @('iam', 'list-access-keys', '--user-name', $UserName) -Context 'verify list-access-keys' -AllowFailure
if ($null -ne $vKeys) {
    $keys = @($vKeys.AccessKeyMetadata)
    if ($keys.Count -eq 0) {
        Write-Ok 'zero access keys -- as intended; the operator creates one out of band'
    } else {
        foreach ($k in $keys) {
            Write-Info "existing key $($k.AccessKeyId) status=$($k.Status) created=$($k.CreateDate)"
        }
        Write-Warn2 "$($keys.Count) access key(s) already exist on this user; this script did not create them. Rotate anything older than 90 days."
    }
}

Write-Step 'Root user credential state (read-only check)'
$summary = Get-AwsJson -AwsArgs @('iam', 'get-account-summary') -Context 'iam get-account-summary' -AllowFailure
if ($null -ne $summary) {
    $rootKeys = [int]$summary.SummaryMap.AccountAccessKeysPresent
    $rootMfa  = [int]$summary.SummaryMap.AccountMFAEnabled
    if ($rootKeys -gt 0) {
        Write-Bad 'the root user HAS long-lived access keys -- delete them (console steps below)'
    } else {
        Write-Ok 'the root user has no long-lived access keys (AccountAccessKeysPresent = 0)'
    }
    if ($rootMfa -gt 0) { Write-Ok 'root MFA is enabled' } else { Write-Bad 'root MFA is NOT enabled -- enable it (console steps below)' }
}

# =============================================================================
# 5. What happened
# =============================================================================
Write-Section 'WHAT THIS RUN DID'
if ($script:Actions.Count -eq 0) {
    Write-Host '  nothing' -ForegroundColor DarkGray
} else {
    foreach ($a in $script:Actions) { Write-Host "  $a" -ForegroundColor Gray }
}

if ($verifyFailures.Count -gt 0) {
    Write-Section 'VERIFICATION FAILED'
    foreach ($f in $verifyFailures) { Write-Host "  - $f" -ForegroundColor Red }
    Write-Host ''
    Write-Host '  The deployer identity is NOT ready. Fix the above and re-run.' -ForegroundColor Red
    exit 1
}

if (-not $apply) {
    Write-Section 'PLAN COMPLETE -- nothing was changed'
    Write-Host "  Re-run with -Confirm to apply." -ForegroundColor Yellow
    exit 0
}

# =============================================================================
# 6. Operator steps -- credentials are NOT created here, on purpose
# =============================================================================
Write-Section 'WHY NO ACCESS KEY WAS CREATED'
Write-Host @"
  This script did NOT run "aws iam create-access-key", and it never will.

  create-access-key returns the SECRET ACCESS KEY exactly once, in the API
  response body. Anything this script prints lands in a place that is routinely
  captured and kept: a PowerShell transcript, a CI job log, a terminal
  scrollback, an agent session record, a screen share. A secret written into any
  of those is a leaked secret from that moment on, and rotating it afterwards
  does not un-leak it. There is no way for a script whose output is a log to hand
  you a credential safely, so it does not try.

  You create the key yourself, in your own shell, and let the CLI write it
  straight into the credentials file without it ever passing through a log.
"@ -ForegroundColor Gray

Write-Section 'OPERATOR: CREATE THE KEY AND WIRE UP A PROFILE'
Write-Host @"
  1. Create the key pair. Run this yourself, in a shell that is NOT being
     transcribed or recorded:

       aws iam create-access-key --user-name $UserName

     Copy AccessKeyId and SecretAccessKey from the output. The secret is shown
     once and cannot be retrieved again -- if you lose it, delete the key and
     make a new one.

  2. Store it under a NAMED PROFILE so it is never the default identity:

       aws configure --profile $NamePrefix-deployer

     Answer the four prompts:
       AWS Access Key ID     : <AccessKeyId from step 1>
       AWS Secret Access Key : <SecretAccessKey from step 1>
       Default region name   : $Region
       Default output format : json

  3. Prove the profile is the scoped user and not root:

       aws sts get-caller-identity --profile $NamePrefix-deployer

     Expect: "Arn": "$UserArn"
     If it still says :root, the profile is not being picked up.

  4. Use it for every script in this directory:

       .\provision_aws.ps1        -AwsProfile $NamePrefix-deployer -NotificationEmail <you> -Confirm
       .\build_image_codebuild.ps1 -Confirm     # honours AWS_PROFILE
       .\teardown_aws.ps1         -AwsProfile $NamePrefix-deployer -Confirm

     Or set it for the whole shell session:

       `$env:AWS_PROFILE = '$NamePrefix-deployer'

  5. Rotate every 90 days -- create the new key, switch the profile over, verify,
     then delete the old one:

       aws iam create-access-key --user-name $UserName
       aws configure --profile $NamePrefix-deployer          # paste the new pair
       aws sts get-caller-identity --profile $NamePrefix-deployer
       aws iam delete-access-key --user-name $UserName --access-key-id <OLD_ID>

  Better than a key, if you can: give the deployer no key at all, create a role
  with these same two policies, and assume it with a short-lived STS session.
  A key that does not exist cannot leak.
"@ -ForegroundColor Gray

Write-Section 'OPERATOR: RETIRE THE ROOT CREDENTIALS (CONSOLE ONLY)'
Write-Host @"
  Root access keys cannot sensibly be managed from a session that is authenticated
  with those same keys -- you would be deleting the credential underneath the call
  that is deleting it. Nor can any IAM policy restrain root. So this is a console
  job, done by a human, after the deployer profile above is proven working.

  Do NOT start this until step 3 above printed the deployer's ARN.

  1. Sign in to the AWS console AS THE ROOT USER (email address + root password),
     not as an IAM user.

  2. Top-right account menu -> "Security credentials".

  3. Under "Access keys": for every key listed, click "Actions" -> "Deactivate",
     then confirm. Leave it deactivated for a few days and re-run the deploy
     scripts with the deployer profile. Once nothing has broken, come back and
     use "Actions" -> "Delete".

     Deactivate-then-delete rather than delete outright: if some forgotten
     automation is still signing with a root key, deactivating shows you
     immediately and is reversible in one click, whereas deletion is not.

  4. On the same page, under "Multi-factor authentication (MFA)", click
     "Assign MFA device" if none is listed. Root without MFA is a password away
     from total account control.

  5. Confirm the result from a shell:

       aws iam get-account-summary --query 'SummaryMap.AccountAccessKeysPresent'   # expect 0
       aws iam get-account-summary --query 'SummaryMap.AccountMFAEnabled'          # expect 1

  6. Then stop using root day to day. If you need an admin identity for tasks the
     deployer is deliberately denied -- creating IAM users, minting keys, editing
     the deployer's own policies -- create a separate admin IAM user with MFA for
     that, and keep root for the handful of things that genuinely require it
     (closing the account, changing the support plan, some billing settings).
"@ -ForegroundColor Gray

Write-Section 'WHAT THE DEPLOYER STILL CANNOT DO'
Write-Host @"
  By design, and this is not a gap to fix later:

  - It cannot create IAM users, access keys, login profiles, or federation
    tokens. It cannot widen anyone's permissions, including its own.
  - It cannot edit its own two policies ($StackPolicyName,
    $PlatformPolicyName). Re-run THIS script from an admin session to change them.
  - It cannot run an EC2 instance, allocate an Elastic IP, create a NAT gateway,
    or create a load balancer / RDS / EKS / SageMaker endpoint. AWS Batch still
    launches Spot instances for you, through its own service-linked role.
  - It cannot act outside us-east-2 (plus us-east-1, which global endpoints such
    as AWS Budgets use).

  Honest residual risk: the deployer manages the IAM roles it also passes to
  Batch and CodeBuild, so it can write an inline policy onto biosim-cc3d-jobRole
  and then run a job as that role. Any identity that provisions its own service
  roles has this property. That is why this is a human-operated provisioning
  identity, not a credential to hand to an unattended pipeline. If it ever needs
  to run unattended, add a permissions boundary to the roles it may create and
  re-scope iam:CreateRole with an iam:PermissionsBoundary condition.
"@ -ForegroundColor Gray

Write-Host ''
Write-Host "Done. Deployer identity: $UserArn" -ForegroundColor Green
Write-Host ''
