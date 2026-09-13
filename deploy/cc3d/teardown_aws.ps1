<#
=================================================================================
 teardown_aws.ps1 -- remove EVERY AWS resource created for on-demand CompuCell3D
                     compute, and nothing else.

 Windows PowerShell 5.1 compatible. Requires the AWS CLI on PATH.

 WHY THIS SCRIPT IS PARANOID
 ---------------------------
 The same AWS account runs Amazon Bedrock for the BioSimulateAI platform and
 holds pre-existing resources that must survive untouched. So deletion is gated
 on BOTH tags being present on the resource itself:

     Project=BioSimulateAI   AND   Component=CC3D-Compute

 Anything that does not prove both tags is WARNED ABOUT AND SKIPPED, never
 deleted. The only two exceptions are stated explicitly at the point of use:
   * VPC children (subnets, route tables, internet gateway, security group,
     gateway endpoint) may be authorised by living inside a VPC that itself
     proved both tags -- because provisioners frequently forget to tag children.
     The authorising rule is printed for every such delete.
   * The AWS Budget, because the Budgets API carries no tags at all. It is
     matched by exact name AND a $20 limit, or it is skipped.

 DISCOVERY -- BOTH PATHS ALWAYS RUN
 ----------------------------------
 (a) deploy/cc3d/aws_resources.json, if present (written by provision_aws.ps1).
 (b) An independent tag sweep:
       aws resourcegroupstaggingapi get-resources
         --tag-filters Key=Project,Values=BioSimulateAI
                       Key=Component,Values=CC3D-Compute
         --region us-east-2
     so an orphan left behind by a partial provision is still found even when
     the JSON file is missing, stale, or was never written.

 SAFETY MODEL
 ------------
   -WhatIf   (DEFAULT) list what would be deleted; change nothing.
   -Confirm  actually delete. Must be passed explicitly. Nothing is destructive
             without it.

 USAGE
 -----
   powershell -NoProfile -File deploy\cc3d\teardown_aws.ps1
   powershell -NoProfile -File deploy\cc3d\teardown_aws.ps1 -Confirm
   powershell -NoProfile -File deploy\cc3d\teardown_aws.ps1 -Confirm -AwsProfile biosim

 EXIT CODES
 ----------
   0  clean (dry run completed, or teardown verified with an empty tag sweep)
   1  finished but leftovers or skips remain -- read the summary
   2  could not start (no AWS CLI, bad credentials, contradictory switches)
=================================================================================
#>

param(
    # Explicitly required to delete anything. Absent => dry run.
    [switch]$Confirm,

    # Accepted for symmetry/readability; this is already the default.
    [switch]$WhatIf,

    [string]$Region = 'us-east-2',

    # Written by provision_aws.ps1. Optional -- the tag sweep stands alone.
    [string]$ResourceFile,

    # Named AWS CLI profile, if the account is not the default one.
    [string]$AwsProfile,

    # /aws/batch/job is an AWS DEFAULT, shared log group. Refused unless this is
    # passed AND the group proves both tags.
    [switch]$AllowSharedLogGroup,

    # Per-wait ceiling for "disable then wait" transitions (job queue, compute env).
    [int]$WaitTimeoutSeconds = 900
)

$ErrorActionPreference = 'Continue'
$env:AWS_PAGER = ''          # never block on a pager, CLI v1 and v2 alike

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

$script:RequiredTags = @{ 'Project' = 'BioSimulateAI'; 'Component' = 'CC3D-Compute' }

# Log groups AWS itself owns / shares across services. Never deleted casually.
$script:SharedLogGroups = @('/aws/batch/job', '/aws/ecs/containerinsights')

# Defaults matching provision_aws.ps1 and the application's env contract
# (BIOSIM_CC3D_JOB_QUEUE=cc3d-queue, BIOSIM_CC3D_JOB_DEFINITION=cc3d-jobdef,
# ECR repository cc3d-worker per deploy/cc3d/Dockerfile). The JSON file and the
# tag sweep both override these; they exist so a teardown still has something to
# probe when provisioning died before writing its manifest.
$script:Defaults = @{
    ComputeEnvironment = 'cc3d-compute-env'
    JobQueue           = 'cc3d-queue'
    JobDefinition      = 'cc3d-jobdef'
    LaunchTemplate     = 'cc3d-compute-lt'
    EcrRepository      = 'cc3d-worker'
    LogGroup           = '/biosimulateai/cc3d'
    BudgetName         = 'biosimulateai-cc3d-monthly'
    BudgetLimit        = 20
    Roles              = @('cc3d-batch-service-role', 'cc3d-ecs-instance-role', 'cc3d-spot-fleet-role')
    InstanceProfiles   = @('cc3d-ecs-instance-profile')
}

# Rolling tallies for the closing summary.
$script:Deleted = New-Object System.Collections.ArrayList
$script:AlreadyGone = New-Object System.Collections.ArrayList
$script:Skipped = New-Object System.Collections.ArrayList
$script:Failed = New-Object System.Collections.ArrayList
$script:Planned = New-Object System.Collections.ArrayList

$script:DryRun = $true      # decided in Initialize-Mode

# ---------------------------------------------------------------------------
# Console helpers
# ---------------------------------------------------------------------------

function Write-Head {
    param([string]$Text)
    Write-Host ''
    Write-Host ('=' * 78) -ForegroundColor DarkCyan
    Write-Host "  $Text" -ForegroundColor Cyan
    Write-Host ('=' * 78) -ForegroundColor DarkCyan
}

function Write-Step { param([string]$Text) Write-Host ''; Write-Host "-- $Text" -ForegroundColor White }
function Write-Info { param([string]$Text) Write-Host "   $Text" -ForegroundColor Gray }
function Write-Good { param([string]$Text) Write-Host "   [deleted] $Text" -ForegroundColor Green }
function Write-Gone { param([string]$Text) Write-Host "   [already gone] $Text" -ForegroundColor DarkGray }
function Write-Plan { param([string]$Text) Write-Host "   [would delete] $Text" -ForegroundColor Yellow }
function Write-Skip { param([string]$Text) Write-Host "   [SKIPPED] $Text" -ForegroundColor Magenta }
function Write-Bad  { param([string]$Text) Write-Host "   [FAILED] $Text" -ForegroundColor Red }

function Add-Deleted     { param([string]$T) [void]$script:Deleted.Add($T);     Write-Good $T }
function Add-AlreadyGone { param([string]$T) [void]$script:AlreadyGone.Add($T); Write-Gone $T }
function Add-Planned     { param([string]$T) [void]$script:Planned.Add($T);     Write-Plan $T }
function Add-Failed      { param([string]$T) [void]$script:Failed.Add($T);      Write-Bad  $T }

function Add-Skipped {
    param([string]$Target, [string]$Reason)
    $entry = "$Target -- $Reason"
    [void]$script:Skipped.Add($entry)
    Write-Skip $entry
}

# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

# Property access that tolerates ConvertFrom-Json objects missing a member.
function Get-Prop {
    param($Object, [string]$Name)
    if ($null -eq $Object) { return $null }
    $props = $Object.PSObject.Properties
    if ($null -eq $props) { return $null }
    $match = $props | Where-Object { $_.Name -eq $Name }
    if ($null -eq $match) { return $null }
    return $match.Value
}

function ConvertTo-JsonString {
    param([string]$Value)
    if ($null -eq $Value) { return '""' }
    return ($Value | ConvertTo-Json)
}

function New-TempFile {
    param([string]$Suffix = '.json')
    $root = $env:TEMP
    if ([string]::IsNullOrWhiteSpace($root)) { $root = [System.IO.Path]::GetTempPath() }
    return (Join-Path $root ("cc3d-teardown-" + [Guid]::NewGuid().ToString('N') + $Suffix))
}

# ---------------------------------------------------------------------------
# AWS CLI plumbing
# ---------------------------------------------------------------------------

function Initialize-AwsCli {
    $cmd = Get-Command aws -ErrorAction SilentlyContinue
    if ($null -eq $cmd) {
        Write-Host 'AWS CLI not found on PATH. Install it, then re-run.' -ForegroundColor Red
        exit 2
    }
    $script:AwsExe = $cmd.Source
    if ([string]::IsNullOrWhiteSpace($script:AwsExe)) { $script:AwsExe = 'aws' }
    Write-Info "aws cli: $script:AwsExe"
}

# Runs the CLI, never throws, returns { Ok, ExitCode, Text, Json }.
function Invoke-Aws {
    param(
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [switch]$NoRegion,
        [string]$RegionOverride
    )

    $argv = New-Object System.Collections.ArrayList
    foreach ($a in $Arguments) { [void]$argv.Add($a) }

    if (-not $NoRegion) {
        $useRegion = $Region
        if (-not [string]::IsNullOrWhiteSpace($RegionOverride)) { $useRegion = $RegionOverride }
        [void]$argv.Add('--region'); [void]$argv.Add($useRegion)
    }
    if (-not [string]::IsNullOrWhiteSpace($AwsProfile)) {
        [void]$argv.Add('--profile'); [void]$argv.Add($AwsProfile)
    }

    $lines = & $script:AwsExe @($argv.ToArray()) 2>&1 | ForEach-Object { [string]$_ }
    $code = $LASTEXITCODE
    $text = ''
    if ($null -ne $lines) { $text = ($lines -join "`n") }

    $json = $null
    if ($code -eq 0 -and -not [string]::IsNullOrWhiteSpace($text)) {
        try { $json = $text | ConvertFrom-Json } catch { $json = $null }
    }

    return [PSCustomObject]@{
        Ok       = ($code -eq 0)
        ExitCode = $code
        Text     = $text
        Json     = $json
        Command  = ($argv.ToArray() -join ' ')
    }
}

# Every "not found" shape the services below actually emit.
$script:NotFoundPatterns = @(
    'NotFound', 'NoSuchBucket', 'NoSuchTagSet', 'NoSuchEntity', 'NoSuchKey',
    'ResourceNotFoundException', 'RepositoryNotFoundException',
    'ImageNotFoundException', 'ResourceNotFound', 'NotFoundException',
    'does not exist', 'Does not exist', 'not found', 'Not Found', 'cannot be found',
    'InvalidVpcID', 'InvalidGroup', 'InvalidSubnetID', 'InvalidRouteTableID',
    'InvalidInternetGatewayID', 'InvalidVpcEndpointId', 'InvalidAssociationID',
    'InvalidLaunchTemplateId', 'InvalidLaunchTemplateName',
    'InvalidParameterValue.*does not exist', 'Unable to locate a budget',
    '404'
)

function Test-NotFound {
    param([string]$Text)
    if ([string]::IsNullOrWhiteSpace($Text)) { return $false }
    foreach ($p in $script:NotFoundPatterns) {
        if ($Text -match $p) { return $true }
    }
    return $false
}

function Test-Dependency {
    param([string]$Text)
    if ([string]::IsNullOrWhiteSpace($Text)) { return $false }
    return ($Text -match 'DependencyViolation' -or
            $Text -match 'has a dependent object' -or
            $Text -match 'resource is in use' -or
            $Text -match 'ResourceInUseException' -or
            $Text -match 'currently in use by' -or
            $Text -match 'InvalidGroup\.InUse')
}

# The single guarded delete. A missing resource is 'already gone', not a failure.
# Dependency violations are retried, because ENIs from a just-killed compute
# environment linger for a minute or two after the instances go away.
function Invoke-GuardedDelete {
    param(
        [Parameter(Mandatory = $true)][string]$Label,
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [int]$Retries = 1,
        [int]$RetryDelaySeconds = 15,
        [switch]$NoRegion,
        [string]$RegionOverride
    )

    if ($script:DryRun) { Add-Planned $Label; return $true }

    for ($attempt = 1; $attempt -le $Retries; $attempt++) {
        $r = Invoke-Aws -Arguments $Arguments -NoRegion:$NoRegion -RegionOverride $RegionOverride
        if ($r.Ok) { Add-Deleted $Label; return $true }
        if (Test-NotFound $r.Text) { Add-AlreadyGone $Label; return $true }
        if ((Test-Dependency $r.Text) -and $attempt -lt $Retries) {
            Write-Info "dependency still holding $Label; retry $attempt/$($Retries - 1) in ${RetryDelaySeconds}s"
            Start-Sleep -Seconds $RetryDelaySeconds
            continue
        }
        $firstLine = ($r.Text -split "`n" | Where-Object { $_.Trim() -ne '' } | Select-Object -First 1)
        Add-Failed "$Label -- $firstLine"
        return $false
    }
    return $false
}

# ---------------------------------------------------------------------------
# Tag reading -- one function per service family, all normalised to a hashtable
# ---------------------------------------------------------------------------

function Convert-TagListToMap {
    param($TagList, [string]$KeyName = 'Key', [string]$ValueName = 'Value')
    $map = @{}
    if ($null -eq $TagList) { return $map }
    foreach ($t in @($TagList)) {
        $k = Get-Prop $t $KeyName
        $v = Get-Prop $t $ValueName
        if (-not [string]::IsNullOrWhiteSpace($k)) { $map[[string]$k] = [string]$v }
    }
    return $map
}

function Convert-TagObjectToMap {
    param($TagObject)
    $map = @{}
    if ($null -eq $TagObject) { return $map }
    foreach ($p in $TagObject.PSObject.Properties) { $map[$p.Name] = [string]$p.Value }
    return $map
}

function Test-RequiredTags {
    param($TagMap)
    if ($null -eq $TagMap) { return $false }
    foreach ($k in $script:RequiredTags.Keys) {
        if (-not $TagMap.ContainsKey($k)) { return $false }
        if ([string]$TagMap[$k] -ne [string]$script:RequiredTags[$k]) { return $false }
    }
    return $true
}

# Works for every EC2 resource type: vpc, subnet, route-table, internet-gateway,
# security-group, launch-template, vpc-endpoint.
function Get-Ec2Tags {
    param([Parameter(Mandatory = $true)][string]$ResourceId)
    $r = Invoke-Aws -Arguments @('ec2', 'describe-tags',
        '--filters', "Name=resource-id,Values=$ResourceId", '--output', 'json')
    if (-not $r.Ok) { return $null }
    return (Convert-TagListToMap (Get-Prop $r.Json 'Tags'))
}

function Get-BatchTags {
    param([ValidateSet('compute-environment', 'job-queue', 'job-definition')][string]$Kind,
          [string]$Name)
    switch ($Kind) {
        'compute-environment' {
            $r = Invoke-Aws -Arguments @('batch', 'describe-compute-environments',
                '--compute-environments', $Name, '--output', 'json')
            if (-not $r.Ok) { return $null }
            $items = @(Get-Prop $r.Json 'computeEnvironments')
            if ($items.Count -eq 0) { return $null }
            return (Convert-TagObjectToMap (Get-Prop $items[0] 'tags'))
        }
        'job-queue' {
            $r = Invoke-Aws -Arguments @('batch', 'describe-job-queues',
                '--job-queues', $Name, '--output', 'json')
            if (-not $r.Ok) { return $null }
            $items = @(Get-Prop $r.Json 'jobQueues')
            if ($items.Count -eq 0) { return $null }
            return (Convert-TagObjectToMap (Get-Prop $items[0] 'tags'))
        }
        'job-definition' {
            $r = Invoke-Aws -Arguments @('batch', 'describe-job-definitions',
                '--job-definition-name', $Name, '--status', 'ACTIVE', '--output', 'json')
            if (-not $r.Ok) { return $null }
            $items = @(Get-Prop $r.Json 'jobDefinitions')
            if ($items.Count -eq 0) { return $null }
            return (Convert-TagObjectToMap (Get-Prop $items[0] 'tags'))
        }
    }
    return $null
}

function Get-BucketTags {
    param([string]$Bucket)
    $r = Invoke-Aws -Arguments @('s3api', 'get-bucket-tagging', '--bucket', $Bucket, '--output', 'json')
    if (-not $r.Ok) { return $null }
    return (Convert-TagListToMap (Get-Prop $r.Json 'TagSet'))
}

function Get-EcrTags {
    param([string]$RepositoryArn)
    $r = Invoke-Aws -Arguments @('ecr', 'list-tags-for-resource', '--resource-arn', $RepositoryArn, '--output', 'json')
    if (-not $r.Ok) { return $null }
    return (Convert-TagListToMap (Get-Prop $r.Json 'tags') 'Key' 'Value')
}

function Get-LogGroupTags {
    param([string]$LogGroupName)
    if (-not [string]::IsNullOrWhiteSpace($script:AccountId)) {
        $arn = "arn:aws:logs:${Region}:$($script:AccountId):log-group:$LogGroupName"
        $r = Invoke-Aws -Arguments @('logs', 'list-tags-for-resource', '--resource-arn', $arn, '--output', 'json')
        if ($r.Ok) { return (Convert-TagObjectToMap (Get-Prop $r.Json 'tags')) }
    }
    # CLI v1 / older API surface.
    $r2 = Invoke-Aws -Arguments @('logs', 'list-tags-log-group', '--log-group-name', $LogGroupName, '--output', 'json')
    if (-not $r2.Ok) { return $null }
    return (Convert-TagObjectToMap (Get-Prop $r2.Json 'tags'))
}

function Get-IamRoleTags {
    param([string]$RoleName)
    $r = Invoke-Aws -NoRegion -Arguments @('iam', 'list-role-tags', '--role-name', $RoleName, '--output', 'json')
    if (-not $r.Ok) { return $null }
    return (Convert-TagListToMap (Get-Prop $r.Json 'Tags'))
}

function Get-IamInstanceProfileTags {
    param([string]$ProfileName)
    $r = Invoke-Aws -NoRegion -Arguments @('iam', 'list-instance-profile-tags',
        '--instance-profile-name', $ProfileName, '--output', 'json')
    if (-not $r.Ok) { return $null }
    return (Convert-TagListToMap (Get-Prop $r.Json 'Tags'))
}

function Get-IamPolicyTags {
    param([string]$PolicyArn)
    $r = Invoke-Aws -NoRegion -Arguments @('iam', 'list-policy-tags', '--policy-arn', $PolicyArn, '--output', 'json')
    if (-not $r.Ok) { return $null }
    return (Convert-TagListToMap (Get-Prop $r.Json 'Tags'))
}

# The gate. Returns $true only when the resource proves both tags.
function Test-Eligible {
    param(
        [Parameter(Mandatory = $true)][string]$Label,
        $TagMap,
        [string]$ParentRule
    )
    if (Test-RequiredTags $TagMap) {
        Write-Info "$Label -- tags verified (Project=BioSimulateAI, Component=CC3D-Compute)"
        return $true
    }
    if (-not [string]::IsNullOrWhiteSpace($ParentRule)) {
        Write-Info "$Label -- own tags absent; authorised by: $ParentRule"
        return $true
    }
    if ($null -eq $TagMap) {
        Add-Skipped $Label 'could not read tags (ambiguous); refusing to touch it'
    } else {
        $seen = ($TagMap.Keys | Sort-Object) -join ','
        if ([string]::IsNullOrWhiteSpace($seen)) { $seen = '<none>' }
        Add-Skipped $Label "missing required tags (found: $seen); refusing to touch it"
    }
    return $false
}

# ---------------------------------------------------------------------------
# Discovery path (a): the manifest written by provision_aws.ps1
# ---------------------------------------------------------------------------

function Import-ResourceManifest {
    Write-Step 'Discovery (a): resource manifest'

    if ([string]::IsNullOrWhiteSpace($ResourceFile)) {
        $root = $PSScriptRoot
        if ([string]::IsNullOrWhiteSpace($root) -and $MyInvocation.MyCommand.Path) {
            $root = Split-Path -Parent $MyInvocation.MyCommand.Path
        }
        if ([string]::IsNullOrWhiteSpace($root)) { $root = (Get-Location).Path }
        $script:ResolvedResourceFile = Join-Path $root 'aws_resources.json'
    } else {
        $script:ResolvedResourceFile = $ResourceFile
    }

    if (-not (Test-Path -LiteralPath $script:ResolvedResourceFile)) {
        Write-Info "no manifest at $script:ResolvedResourceFile -- relying on the tag sweep alone"
        return $null
    }

    $raw = Get-Content -Raw -LiteralPath $script:ResolvedResourceFile
    if ([string]::IsNullOrWhiteSpace($raw)) {
        Write-Info "manifest $script:ResolvedResourceFile is empty -- relying on the tag sweep alone"
        return $null
    }
    try {
        $parsed = $raw | ConvertFrom-Json
    } catch {
        Write-Info "manifest $script:ResolvedResourceFile is not valid JSON ($($_.Exception.Message)) -- relying on the tag sweep alone"
        return $null
    }
    Write-Info "read $script:ResolvedResourceFile"
    return $parsed
}

# ---------------------------------------------------------------------------
# Discovery path (b): the independent tag sweep
# ---------------------------------------------------------------------------

function Get-TaggedResources {
    $r = Invoke-Aws -Arguments @('resourcegroupstaggingapi', 'get-resources',
        '--tag-filters', 'Key=Project,Values=BioSimulateAI', 'Key=Component,Values=CC3D-Compute',
        '--output', 'json')
    if (-not $r.Ok) {
        Write-Info "tag sweep failed: $($r.Text)"
        return $null
    }
    $list = @(Get-Prop $r.Json 'ResourceTagMappingList')
    $arns = New-Object System.Collections.ArrayList
    foreach ($item in $list) {
        $arn = Get-Prop $item 'ResourceARN'
        if (-not [string]::IsNullOrWhiteSpace($arn)) { [void]$arns.Add([string]$arn) }
    }
    # Leading comma: without it PowerShell unrolls the list and a single result
    # would arrive at the caller as a bare string with no .Add().
    return ,$arns
}

# arn:partition:service:region:account:resource-type/resource-id  (or :resource-id)
function Split-Arn {
    param([string]$Arn)
    $parts = $Arn -split ':', 6
    if ($parts.Count -lt 6) { return $null }
    $tail = $parts[5]
    $type = ''
    $id = $tail
    if ($tail -match '^([^/:]+)[/:](.+)$') { $type = $Matches[1]; $id = $Matches[2] }
    return [PSCustomObject]@{
        Arn = $Arn; Service = $parts[2]; Account = $parts[4]; Type = $type; Id = $id
    }
}

function Group-TaggedArns {
    param($Arns)
    $map = @{}
    foreach ($arn in @($Arns)) {
        $p = Split-Arn $arn
        if ($null -eq $p) { continue }
        $key = "$($p.Service)/$($p.Type)"
        if (-not $map.ContainsKey($key)) { $map[$key] = New-Object System.Collections.ArrayList }
        [void]$map[$key].Add($p)
    }
    return $map
}

function Get-DiscoveredIds {
    param($Map, [string]$Key)
    $out = New-Object System.Collections.ArrayList
    if ($null -ne $Map -and $Map.ContainsKey($Key)) {
        foreach ($p in $Map[$Key]) { [void]$out.Add($p.Id) }
    }
    return ,$out
}

# Union of manifest value(s) and tag-sweep ids, de-duplicated, order preserved.
function Merge-Candidates {
    param($FromManifest, $FromTags, $FromDefault)
    $seen = @{}
    $out = New-Object System.Collections.ArrayList
    foreach ($group in @($FromManifest, $FromTags, $FromDefault)) {
        foreach ($v in @($group)) {
            if ($null -eq $v) { continue }
            $s = ([string]$v).Trim()
            if ($s -eq '') { continue }
            if ($seen.ContainsKey($s)) { continue }
            $seen[$s] = $true
            [void]$out.Add($s)
        }
    }
    return ,$out
}

# ---------------------------------------------------------------------------
# Waiters
# ---------------------------------------------------------------------------

function Wait-JobQueueValid {
    param([string]$Name)
    if ($script:DryRun) { return $true }
    $deadline = (Get-Date).AddSeconds($WaitTimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        $r = Invoke-Aws -Arguments @('batch', 'describe-job-queues', '--job-queues', $Name, '--output', 'json')
        if (-not $r.Ok) { return $true }   # gone or unreadable; the delete below decides
        $items = @(Get-Prop $r.Json 'jobQueues')
        if ($items.Count -eq 0) { return $true }
        $state = [string](Get-Prop $items[0] 'state')
        $status = [string](Get-Prop $items[0] 'status')
        Write-Info "job queue $Name -- state=$state status=$status"
        if ($status -eq 'VALID' -and $state -eq 'DISABLED') { return $true }
        if ($status -eq 'INVALID') { Write-Info "job queue $Name is INVALID; attempting delete anyway"; return $true }
        Start-Sleep -Seconds 10
    }
    Write-Info "timed out after ${WaitTimeoutSeconds}s waiting for job queue $Name to settle"
    return $false
}

function Wait-ComputeEnvironmentDisabled {
    param([string]$Name)
    if ($script:DryRun) { return $true }
    $deadline = (Get-Date).AddSeconds($WaitTimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        $r = Invoke-Aws -Arguments @('batch', 'describe-compute-environments',
            '--compute-environments', $Name, '--output', 'json')
        if (-not $r.Ok) { return $true }
        $items = @(Get-Prop $r.Json 'computeEnvironments')
        if ($items.Count -eq 0) { return $true }
        $state = [string](Get-Prop $items[0] 'state')
        $status = [string](Get-Prop $items[0] 'status')
        Write-Info "compute env $Name -- state=$state status=$status"
        # AWS refuses the delete unless the environment is DISABLED and settled.
        if ($state -eq 'DISABLED' -and ($status -eq 'VALID' -or $status -eq 'INVALID')) { return $true }
        Start-Sleep -Seconds 15
    }
    Write-Info "timed out after ${WaitTimeoutSeconds}s waiting for compute env $Name to reach DISABLED"
    return $false
}

# ---------------------------------------------------------------------------
# 1. Job queue -- disable, wait VALID, delete
# ---------------------------------------------------------------------------

function Remove-JobQueues {
    param($Names)
    Write-Step '1/13  AWS Batch job queue (disable -> wait -> delete)'
    if (@($Names).Count -eq 0) { Write-Info 'nothing to consider'; return }

    foreach ($name in @($Names)) {
        $label = "batch job queue $name"
        $tags = Get-BatchTags -Kind 'job-queue' -Name $name
        if ($null -eq $tags) {
            $probe = Invoke-Aws -Arguments @('batch', 'describe-job-queues', '--job-queues', $name, '--output', 'json')
            $items = @(Get-Prop $probe.Json 'jobQueues')
            if ($probe.Ok -and $items.Count -eq 0) { Add-AlreadyGone $label; continue }
        }
        if (-not (Test-Eligible -Label $label -TagMap $tags)) { continue }

        if (-not $script:DryRun) {
            $d = Invoke-Aws -Arguments @('batch', 'update-job-queue', '--job-queue', $name, '--state', 'DISABLED')
            if ($d.Ok) { Write-Info "$name -> DISABLED requested" }
            elseif (Test-NotFound $d.Text) { Add-AlreadyGone $label; continue }
            else { Write-Info "could not disable ${name}: $($d.Text)" }
            [void](Wait-JobQueueValid -Name $name)
        }

        [void](Invoke-GuardedDelete -Label $label -Retries 4 -RetryDelaySeconds 20 `
            -Arguments @('batch', 'delete-job-queue', '--job-queue', $name))
    }
}

# ---------------------------------------------------------------------------
# 2. Job definition -- deregister every ACTIVE revision
# ---------------------------------------------------------------------------

function Remove-JobDefinitions {
    param($Names)
    Write-Step '2/13  AWS Batch job definition (deregister all ACTIVE revisions)'
    if (@($Names).Count -eq 0) { Write-Info 'nothing to consider'; return }

    foreach ($name in @($Names)) {
        # A manifest may carry either a bare name or a name:revision ARN tail.
        $bare = $name
        if ($bare -match '^(.+):\d+$') { $bare = $Matches[1] }

        $label = "batch job definition $bare"
        $tags = Get-BatchTags -Kind 'job-definition' -Name $bare
        $r = Invoke-Aws -Arguments @('batch', 'describe-job-definitions',
            '--job-definition-name', $bare, '--status', 'ACTIVE', '--output', 'json')
        $revs = @()
        if ($r.Ok) { $revs = @(Get-Prop $r.Json 'jobDefinitions') }
        if ($r.Ok -and $revs.Count -eq 0) { Add-AlreadyGone $label; continue }
        if (-not (Test-Eligible -Label $label -TagMap $tags)) { continue }

        foreach ($rev in $revs) {
            $arn = [string](Get-Prop $rev 'jobDefinitionArn')
            $n = [string](Get-Prop $rev 'jobDefinitionName')
            $v = [string](Get-Prop $rev 'revision')
            $target = $arn
            if ([string]::IsNullOrWhiteSpace($target)) { $target = "${n}:${v}" }
            [void](Invoke-GuardedDelete -Label "batch job definition ${n}:${v}" `
                -Arguments @('batch', 'deregister-job-definition', '--job-definition', $target))
        }
        if ($revs.Count -eq 0 -and $script:DryRun) { Add-Planned $label }
    }
}

# ---------------------------------------------------------------------------
# 3. Compute environment -- disable, wait DISABLED, delete
# ---------------------------------------------------------------------------

function Remove-ComputeEnvironments {
    param($Names)
    Write-Step '3/13  AWS Batch compute environment (disable -> wait DISABLED -> delete)'
    if (@($Names).Count -eq 0) { Write-Info 'nothing to consider'; return }

    foreach ($name in @($Names)) {
        $label = "batch compute environment $name"
        $tags = Get-BatchTags -Kind 'compute-environment' -Name $name
        if ($null -eq $tags) {
            $probe = Invoke-Aws -Arguments @('batch', 'describe-compute-environments',
                '--compute-environments', $name, '--output', 'json')
            $items = @(Get-Prop $probe.Json 'computeEnvironments')
            if ($probe.Ok -and $items.Count -eq 0) { Add-AlreadyGone $label; continue }
        }
        if (-not (Test-Eligible -Label $label -TagMap $tags)) { continue }

        if (-not $script:DryRun) {
            $d = Invoke-Aws -Arguments @('batch', 'update-compute-environment',
                '--compute-environment', $name, '--state', 'DISABLED')
            if ($d.Ok) { Write-Info "$name -> DISABLED requested" }
            elseif (Test-NotFound $d.Text) { Add-AlreadyGone $label; continue }
            else { Write-Info "could not disable ${name}: $($d.Text)" }
            if (-not (Wait-ComputeEnvironmentDisabled -Name $name)) {
                Add-Failed "$label -- never reached DISABLED; AWS will refuse the delete. Re-run the script."
                continue
            }
        }

        [void](Invoke-GuardedDelete -Label $label -Retries 5 -RetryDelaySeconds 20 `
            -Arguments @('batch', 'delete-compute-environment', '--compute-environment', $name))
    }
}

# ---------------------------------------------------------------------------
# 4. Launch template
# ---------------------------------------------------------------------------

function Remove-LaunchTemplates {
    param($NamesOrIds)
    Write-Step '4/13  EC2 launch template'
    if (@($NamesOrIds).Count -eq 0) { Write-Info 'nothing to consider'; return }

    foreach ($item in @($NamesOrIds)) {
        $isId = ($item -match '^lt-[0-9a-f]+$')
        $label = "launch template $item"

        $selector = @()
        if ($isId) { $selector = @('--launch-template-ids', $item) }
        else { $selector = @('--launch-template-names', $item) }

        $desc = Invoke-Aws -Arguments (@('ec2', 'describe-launch-templates') + $selector + @('--output', 'json'))
        if (-not $desc.Ok) {
            if (Test-NotFound $desc.Text) { Add-AlreadyGone $label; continue }
            Add-Skipped $label "could not describe it ($($desc.Text)); refusing to touch it"
            continue
        }
        $templates = @(Get-Prop $desc.Json 'LaunchTemplates')
        if ($templates.Count -eq 0) { Add-AlreadyGone $label; continue }

        foreach ($t in $templates) {
            $ltId = [string](Get-Prop $t 'LaunchTemplateId')
            $ltName = [string](Get-Prop $t 'LaunchTemplateName')
            $tags = Convert-TagListToMap (Get-Prop $t 'Tags')
            if ((-not (Test-RequiredTags $tags)) -and $ltId) {
                $fallback = Get-Ec2Tags -ResourceId $ltId
                if ($null -ne $fallback -and $fallback.Count -gt 0) { $tags = $fallback }
            }
            $lbl = "launch template $ltName ($ltId)"
            if (-not (Test-Eligible -Label $lbl -TagMap $tags)) { continue }
            [void](Invoke-GuardedDelete -Label $lbl `
                -Arguments @('ec2', 'delete-launch-template', '--launch-template-id', $ltId))
        }
    }
}

# ---------------------------------------------------------------------------
# 5. S3 bucket -- empty every version AND every delete marker, then delete
# ---------------------------------------------------------------------------

function Clear-BucketCompletely {
    param([string]$Bucket)

    $rounds = 0
    while ($true) {
        $rounds++
        if ($rounds -gt 500) {
            Add-Failed "s3://$Bucket -- still not empty after 500 delete rounds"
            return $false
        }

        $r = Invoke-Aws -Arguments @('s3api', 'list-object-versions', '--bucket', $Bucket,
            '--max-keys', '1000', '--output', 'json')
        if (-not $r.Ok) {
            if (Test-NotFound $r.Text) { return $true }
            Add-Failed "s3://$Bucket -- cannot list object versions: $($r.Text)"
            return $false
        }

        $entries = New-Object System.Collections.ArrayList
        foreach ($section in @('Versions', 'DeleteMarkers')) {
            foreach ($o in @(Get-Prop $r.Json $section)) {
                $k = [string](Get-Prop $o 'Key')
                $v = [string](Get-Prop $o 'VersionId')
                if ([string]::IsNullOrWhiteSpace($k)) { continue }
                [void]$entries.Add([PSCustomObject]@{ Key = $k; VersionId = $v })
            }
        }

        if ($entries.Count -eq 0) {
            Write-Info "s3://$Bucket is empty (no object versions, no delete markers)"
            return $true
        }

        # Hand-built JSON: ConvertTo-Json's single-element array behaviour is not
        # worth betting a delete payload on.
        $sb = New-Object System.Text.StringBuilder
        [void]$sb.Append('{"Objects":[')
        for ($i = 0; $i -lt $entries.Count; $i++) {
            if ($i -gt 0) { [void]$sb.Append(',') }
            [void]$sb.Append('{"Key":')
            [void]$sb.Append((ConvertTo-JsonString $entries[$i].Key))
            if (-not [string]::IsNullOrWhiteSpace($entries[$i].VersionId)) {
                [void]$sb.Append(',"VersionId":')
                [void]$sb.Append((ConvertTo-JsonString $entries[$i].VersionId))
            }
            [void]$sb.Append('}')
        }
        [void]$sb.Append('],"Quiet":true}')

        $tmp = New-TempFile
        try {
            Set-Content -LiteralPath $tmp -Value $sb.ToString() -Encoding ASCII
            $d = Invoke-Aws -Arguments @('s3api', 'delete-objects', '--bucket', $Bucket,
                '--delete', "file://$tmp", '--output', 'json')
            if (-not $d.Ok -and -not (Test-NotFound $d.Text)) {
                Add-Failed "s3://$Bucket -- delete-objects failed: $($d.Text)"
                return $false
            }
            Write-Info "s3://$Bucket -- removed $($entries.Count) version(s)/marker(s)"
        } finally {
            Remove-Item -LiteralPath $tmp -Force -ErrorAction SilentlyContinue
        }
    }
}

function Remove-Buckets {
    param($Names)
    Write-Step '5/13  S3 bucket (empty all versions + delete markers, then delete)'
    if (@($Names).Count -eq 0) { Write-Info 'nothing to consider'; return }

    foreach ($bucket in @($Names)) {
        $label = "s3 bucket $bucket"
        $head = Invoke-Aws -Arguments @('s3api', 'head-bucket', '--bucket', $bucket)
        if (-not $head.Ok -and (Test-NotFound $head.Text)) { Add-AlreadyGone $label; continue }

        $tags = Get-BucketTags -Bucket $bucket
        if (-not (Test-Eligible -Label $label -TagMap $tags)) { continue }

        if ($script:DryRun) {
            Add-Planned "$label (and every object version / delete marker inside it)"
            continue
        }
        if (-not (Clear-BucketCompletely -Bucket $bucket)) { continue }
        [void](Invoke-GuardedDelete -Label $label -Retries 3 -RetryDelaySeconds 10 `
            -Arguments @('s3api', 'delete-bucket', '--bucket', $bucket))
    }
}

# ---------------------------------------------------------------------------
# 6. ECR -- delete every image, then the repository
# ---------------------------------------------------------------------------

function Remove-EcrRepositories {
    param($Names)
    Write-Step '6/13  ECR images then repository'
    if (@($Names).Count -eq 0) { Write-Info 'nothing to consider'; return }

    foreach ($repo in @($Names)) {
        $label = "ecr repository $repo"
        $desc = Invoke-Aws -Arguments @('ecr', 'describe-repositories',
            '--repository-names', $repo, '--output', 'json')
        if (-not $desc.Ok) {
            if (Test-NotFound $desc.Text) { Add-AlreadyGone $label; continue }
            Add-Skipped $label "could not describe it ($($desc.Text)); refusing to touch it"
            continue
        }
        $repos = @(Get-Prop $desc.Json 'repositories')
        if ($repos.Count -eq 0) { Add-AlreadyGone $label; continue }
        $arn = [string](Get-Prop $repos[0] 'repositoryArn')

        $tags = Get-EcrTags -RepositoryArn $arn
        if (-not (Test-Eligible -Label $label -TagMap $tags)) { continue }

        if ($script:DryRun) {
            Add-Planned "$label (and every image in it)"
            continue
        }

        # Images first, in batches of 100 (the API ceiling).
        $guard = 0
        while ($true) {
            $guard++
            if ($guard -gt 200) { Add-Failed "$label -- images kept reappearing after 200 rounds"; break }
            $li = Invoke-Aws -Arguments @('ecr', 'list-images', '--repository-name', $repo,
                '--max-items', '100', '--output', 'json')
            if (-not $li.Ok) {
                if (Test-NotFound $li.Text) { break }
                Add-Failed "$label -- list-images failed: $($li.Text)"
                break
            }
            $ids = @(Get-Prop $li.Json 'imageIds')
            if ($ids.Count -eq 0) { Write-Info "$label -- no images left"; break }

            $sb = New-Object System.Text.StringBuilder
            [void]$sb.Append('[')
            for ($i = 0; $i -lt $ids.Count; $i++) {
                if ($i -gt 0) { [void]$sb.Append(',') }
                $digest = [string](Get-Prop $ids[$i] 'imageDigest')
                $tag = [string](Get-Prop $ids[$i] 'imageTag')
                [void]$sb.Append('{')
                if (-not [string]::IsNullOrWhiteSpace($digest)) {
                    [void]$sb.Append('"imageDigest":'); [void]$sb.Append((ConvertTo-JsonString $digest))
                } elseif (-not [string]::IsNullOrWhiteSpace($tag)) {
                    [void]$sb.Append('"imageTag":'); [void]$sb.Append((ConvertTo-JsonString $tag))
                }
                [void]$sb.Append('}')
            }
            [void]$sb.Append(']')

            $tmp = New-TempFile
            try {
                Set-Content -LiteralPath $tmp -Value $sb.ToString() -Encoding ASCII
                $bd = Invoke-Aws -Arguments @('ecr', 'batch-delete-image', '--repository-name', $repo,
                    '--image-ids', "file://$tmp", '--output', 'json')
                if (-not $bd.Ok -and -not (Test-NotFound $bd.Text)) {
                    Add-Failed "$label -- batch-delete-image failed: $($bd.Text)"
                    break
                }
                Write-Info "$label -- deleted $($ids.Count) image(s)"
            } finally {
                Remove-Item -LiteralPath $tmp -Force -ErrorAction SilentlyContinue
            }
        }

        $del = Invoke-Aws -Arguments @('ecr', 'delete-repository', '--repository-name', $repo)
        if ($del.Ok) { Add-Deleted $label }
        elseif (Test-NotFound $del.Text) { Add-AlreadyGone $label }
        else {
            # Untagged manifest lists can keep a repository "not empty"; --force
            # only ever removes images inside this already tag-verified repository.
            $forced = Invoke-Aws -Arguments @('ecr', 'delete-repository', '--repository-name', $repo, '--force')
            if ($forced.Ok) { Add-Deleted "$label (--force)" }
            elseif (Test-NotFound $forced.Text) { Add-AlreadyGone $label }
            else { Add-Failed "$label -- $($forced.Text)" }
        }
    }
}

# ---------------------------------------------------------------------------
# 7. S3 gateway VPC endpoint
# ---------------------------------------------------------------------------

function Remove-VpcEndpoints {
    param($Ids, $TaggedVpcIds)
    Write-Step '7/13  S3 gateway VPC endpoint'
    if (@($Ids).Count -eq 0) { Write-Info 'nothing to consider'; return }

    foreach ($id in @($Ids)) {
        $label = "vpc endpoint $id"
        $desc = Invoke-Aws -Arguments @('ec2', 'describe-vpc-endpoints', '--vpc-endpoint-ids', $id, '--output', 'json')
        if (-not $desc.Ok) {
            if (Test-NotFound $desc.Text) { Add-AlreadyGone $label; continue }
            Add-Skipped $label "could not describe it; refusing to touch it"
            continue
        }
        $eps = @(Get-Prop $desc.Json 'VpcEndpoints')
        if ($eps.Count -eq 0) { Add-AlreadyGone $label; continue }

        $tags = Convert-TagListToMap (Get-Prop $eps[0] 'Tags')
        $vpcId = [string](Get-Prop $eps[0] 'VpcId')
        $rule = ''
        if ((-not (Test-RequiredTags $tags)) -and (@($TaggedVpcIds) -contains $vpcId)) {
            $rule = "child of tag-verified VPC $vpcId"
        }
        if (-not (Test-Eligible -Label $label -TagMap $tags -ParentRule $rule)) { continue }

        [void](Invoke-GuardedDelete -Label $label -Retries 3 -RetryDelaySeconds 10 `
            -Arguments @('ec2', 'delete-vpc-endpoints', '--vpc-endpoint-ids', $id))
    }
}

# ---------------------------------------------------------------------------
# 8. Security group
# ---------------------------------------------------------------------------

function Remove-SecurityGroups {
    param($Ids, $TaggedVpcIds)
    Write-Step '8/13  Security group'
    if (@($Ids).Count -eq 0) { Write-Info 'nothing to consider'; return }

    foreach ($id in @($Ids)) {
        $label = "security group $id"
        $desc = Invoke-Aws -Arguments @('ec2', 'describe-security-groups', '--group-ids', $id, '--output', 'json')
        if (-not $desc.Ok) {
            if (Test-NotFound $desc.Text) { Add-AlreadyGone $label; continue }
            Add-Skipped $label 'could not describe it; refusing to touch it'
            continue
        }
        $groups = @(Get-Prop $desc.Json 'SecurityGroups')
        if ($groups.Count -eq 0) { Add-AlreadyGone $label; continue }

        $name = [string](Get-Prop $groups[0] 'GroupName')
        $vpcId = [string](Get-Prop $groups[0] 'VpcId')
        if ($name -eq 'default') {
            Add-Skipped "$label ($name)" 'default security group -- goes away with the VPC, never deleted directly'
            continue
        }
        $tags = Convert-TagListToMap (Get-Prop $groups[0] 'Tags')
        $rule = ''
        if ((-not (Test-RequiredTags $tags)) -and (@($TaggedVpcIds) -contains $vpcId)) {
            $rule = "non-default group inside tag-verified VPC $vpcId"
        }
        if (-not (Test-Eligible -Label "$label ($name)" -TagMap $tags -ParentRule $rule)) { continue }

        # ENIs from the compute environment can hold the group for a minute or two.
        [void](Invoke-GuardedDelete -Label "$label ($name)" -Retries 6 -RetryDelaySeconds 20 `
            -Arguments @('ec2', 'delete-security-group', '--group-id', $id))
    }
}

# ---------------------------------------------------------------------------
# 9. Internet gateway -- detach then delete
# ---------------------------------------------------------------------------

function Remove-InternetGateways {
    param($Ids, $TaggedVpcIds)
    Write-Step '9/13  Internet gateway (detach -> delete)'
    if (@($Ids).Count -eq 0) { Write-Info 'nothing to consider'; return }

    foreach ($id in @($Ids)) {
        $label = "internet gateway $id"
        $desc = Invoke-Aws -Arguments @('ec2', 'describe-internet-gateways',
            '--internet-gateway-ids', $id, '--output', 'json')
        if (-not $desc.Ok) {
            if (Test-NotFound $desc.Text) { Add-AlreadyGone $label; continue }
            Add-Skipped $label 'could not describe it; refusing to touch it'
            continue
        }
        $igws = @(Get-Prop $desc.Json 'InternetGateways')
        if ($igws.Count -eq 0) { Add-AlreadyGone $label; continue }

        $tags = Convert-TagListToMap (Get-Prop $igws[0] 'Tags')
        $attachments = @(Get-Prop $igws[0] 'Attachments')
        $attachedVpcs = New-Object System.Collections.ArrayList
        foreach ($a in $attachments) {
            $v = [string](Get-Prop $a 'VpcId')
            if (-not [string]::IsNullOrWhiteSpace($v)) { [void]$attachedVpcs.Add($v) }
        }

        $rule = ''
        if (-not (Test-RequiredTags $tags)) {
            foreach ($v in $attachedVpcs) {
                if (@($TaggedVpcIds) -contains $v) { $rule = "attached only to tag-verified VPC $v"; break }
            }
            # If it also hangs off a VPC we did not verify, do not touch it at all.
            if ($rule -ne '') {
                foreach ($v in $attachedVpcs) {
                    if (-not (@($TaggedVpcIds) -contains $v)) {
                        $rule = ''
                        Add-Skipped $label "also attached to unverified VPC $v; refusing to touch it"
                        break
                    }
                }
                if ($rule -eq '') { continue }
            }
        }
        if (-not (Test-Eligible -Label $label -TagMap $tags -ParentRule $rule)) { continue }

        foreach ($v in $attachedVpcs) {
            [void](Invoke-GuardedDelete -Label "detach $id from $v" -Retries 3 -RetryDelaySeconds 15 `
                -Arguments @('ec2', 'detach-internet-gateway', '--internet-gateway-id', $id, '--vpc-id', $v))
        }
        [void](Invoke-GuardedDelete -Label $label -Retries 3 -RetryDelaySeconds 15 `
            -Arguments @('ec2', 'delete-internet-gateway', '--internet-gateway-id', $id))
    }
}

# ---------------------------------------------------------------------------
# 10. Subnets -> route tables -> VPC
# ---------------------------------------------------------------------------

function Remove-Subnets {
    param($Ids, $TaggedVpcIds)
    Write-Step '10/13  Subnets'
    if (@($Ids).Count -eq 0) { Write-Info 'nothing to consider'; return }

    foreach ($id in @($Ids)) {
        $label = "subnet $id"
        $desc = Invoke-Aws -Arguments @('ec2', 'describe-subnets', '--subnet-ids', $id, '--output', 'json')
        if (-not $desc.Ok) {
            if (Test-NotFound $desc.Text) { Add-AlreadyGone $label; continue }
            Add-Skipped $label 'could not describe it; refusing to touch it'
            continue
        }
        $subnets = @(Get-Prop $desc.Json 'Subnets')
        if ($subnets.Count -eq 0) { Add-AlreadyGone $label; continue }

        $tags = Convert-TagListToMap (Get-Prop $subnets[0] 'Tags')
        $vpcId = [string](Get-Prop $subnets[0] 'VpcId')
        $rule = ''
        if ((-not (Test-RequiredTags $tags)) -and (@($TaggedVpcIds) -contains $vpcId)) {
            $rule = "subnet inside tag-verified VPC $vpcId"
        }
        if (-not (Test-Eligible -Label $label -TagMap $tags -ParentRule $rule)) { continue }

        [void](Invoke-GuardedDelete -Label $label -Retries 6 -RetryDelaySeconds 20 `
            -Arguments @('ec2', 'delete-subnet', '--subnet-id', $id))
    }
}

function Remove-RouteTables {
    param($Ids, $TaggedVpcIds)
    Write-Step '11/13  Route tables (disassociate -> delete; main table left to the VPC)'
    if (@($Ids).Count -eq 0) { Write-Info 'nothing to consider'; return }

    foreach ($id in @($Ids)) {
        $label = "route table $id"
        $desc = Invoke-Aws -Arguments @('ec2', 'describe-route-tables', '--route-table-ids', $id, '--output', 'json')
        if (-not $desc.Ok) {
            if (Test-NotFound $desc.Text) { Add-AlreadyGone $label; continue }
            Add-Skipped $label 'could not describe it; refusing to touch it'
            continue
        }
        $tables = @(Get-Prop $desc.Json 'RouteTables')
        if ($tables.Count -eq 0) { Add-AlreadyGone $label; continue }

        $tags = Convert-TagListToMap (Get-Prop $tables[0] 'Tags')
        $vpcId = [string](Get-Prop $tables[0] 'VpcId')
        $assocs = @(Get-Prop $tables[0] 'Associations')

        $isMain = $false
        foreach ($a in $assocs) { if ([bool](Get-Prop $a 'Main')) { $isMain = $true } }
        if ($isMain) {
            Add-Skipped $label 'main route table -- deleted with the VPC, cannot be deleted directly'
            continue
        }

        $rule = ''
        if ((-not (Test-RequiredTags $tags)) -and (@($TaggedVpcIds) -contains $vpcId)) {
            $rule = "non-main route table inside tag-verified VPC $vpcId"
        }
        if (-not (Test-Eligible -Label $label -TagMap $tags -ParentRule $rule)) { continue }

        foreach ($a in $assocs) {
            $aid = [string](Get-Prop $a 'RouteTableAssociationId')
            if ([string]::IsNullOrWhiteSpace($aid)) { continue }
            [void](Invoke-GuardedDelete -Label "route table association $aid" `
                -Arguments @('ec2', 'disassociate-route-table', '--association-id', $aid))
        }
        [void](Invoke-GuardedDelete -Label $label -Retries 4 -RetryDelaySeconds 15 `
            -Arguments @('ec2', 'delete-route-table', '--route-table-id', $id))
    }
}

function Remove-Vpcs {
    param($Ids)
    Write-Step '12/13  VPC'
    if (@($Ids).Count -eq 0) { Write-Info 'nothing to consider'; return }

    foreach ($id in @($Ids)) {
        $label = "vpc $id"
        $desc = Invoke-Aws -Arguments @('ec2', 'describe-vpcs', '--vpc-ids', $id, '--output', 'json')
        if (-not $desc.Ok) {
            if (Test-NotFound $desc.Text) { Add-AlreadyGone $label; continue }
            Add-Skipped $label 'could not describe it; refusing to touch it'
            continue
        }
        $vpcs = @(Get-Prop $desc.Json 'Vpcs')
        if ($vpcs.Count -eq 0) { Add-AlreadyGone $label; continue }

        if ([bool](Get-Prop $vpcs[0] 'IsDefault')) {
            Add-Skipped $label 'this is the account default VPC -- never deleted'
            continue
        }
        $tags = Convert-TagListToMap (Get-Prop $vpcs[0] 'Tags')
        if (-not (Test-Eligible -Label $label -TagMap $tags)) { continue }

        [void](Invoke-GuardedDelete -Label $label -Retries 6 -RetryDelaySeconds 20 `
            -Arguments @('ec2', 'delete-vpc', '--vpc-id', $id))
    }
}

# ---------------------------------------------------------------------------
# 11. IAM -- inline policies, attached policies, roles, instance profiles
# ---------------------------------------------------------------------------

function Remove-IamInstanceProfile {
    param([string]$ProfileName)
    $label = "iam instance profile $ProfileName"
    $desc = Invoke-Aws -NoRegion -Arguments @('iam', 'get-instance-profile',
        '--instance-profile-name', $ProfileName, '--output', 'json')
    if (-not $desc.Ok) {
        if (Test-NotFound $desc.Text) { Add-AlreadyGone $label; return }
        Add-Skipped $label 'could not read it; refusing to touch it'
        return
    }
    $tags = Get-IamInstanceProfileTags -ProfileName $ProfileName
    if (-not (Test-Eligible -Label $label -TagMap $tags)) { return }

    $attachedRoles = Get-Prop (Get-Prop $desc.Json 'InstanceProfile') 'Roles'
    foreach ($role in @($attachedRoles)) {
        $rn = [string](Get-Prop $role 'RoleName')
        if ([string]::IsNullOrWhiteSpace($rn)) { continue }
        [void](Invoke-GuardedDelete -NoRegion -Label "remove role $rn from instance profile $ProfileName" `
            -Arguments @('iam', 'remove-role-from-instance-profile',
                '--instance-profile-name', $ProfileName, '--role-name', $rn))
    }
    [void](Invoke-GuardedDelete -NoRegion -Label $label `
        -Arguments @('iam', 'delete-instance-profile', '--instance-profile-name', $ProfileName))
}

function Remove-IamRole {
    param([string]$RoleName)
    $label = "iam role $RoleName"
    $get = Invoke-Aws -NoRegion -Arguments @('iam', 'get-role', '--role-name', $RoleName, '--output', 'json')
    if (-not $get.Ok) {
        if (Test-NotFound $get.Text) { Add-AlreadyGone $label; return }
        Add-Skipped $label 'could not read it; refusing to touch it'
        return
    }
    $tags = Get-IamRoleTags -RoleName $RoleName
    if (-not (Test-Eligible -Label $label -TagMap $tags)) { return }

    # Inline policies this provisioner wrote.
    $inline = Invoke-Aws -NoRegion -Arguments @('iam', 'list-role-policies', '--role-name', $RoleName, '--output', 'json')
    if ($inline.Ok) {
        foreach ($p in @(Get-Prop $inline.Json 'PolicyNames')) {
            $pn = [string]$p
            if ([string]::IsNullOrWhiteSpace($pn)) { continue }
            [void](Invoke-GuardedDelete -NoRegion -Label "iam inline policy $RoleName/$pn" `
                -Arguments @('iam', 'delete-role-policy', '--role-name', $RoleName, '--policy-name', $pn))
        }
    }

    # Managed policies: always DETACH. Only delete the policy itself when it is
    # customer-managed AND carries both tags -- AWS-managed policies are shared.
    $attached = Invoke-Aws -NoRegion -Arguments @('iam', 'list-attached-role-policies',
        '--role-name', $RoleName, '--output', 'json')
    $customerManaged = New-Object System.Collections.ArrayList
    if ($attached.Ok) {
        foreach ($p in @(Get-Prop $attached.Json 'AttachedPolicies')) {
            $arn = [string](Get-Prop $p 'PolicyArn')
            $pn = [string](Get-Prop $p 'PolicyName')
            if ([string]::IsNullOrWhiteSpace($arn)) { continue }
            [void](Invoke-GuardedDelete -NoRegion -Label "detach $pn from $RoleName" `
                -Arguments @('iam', 'detach-role-policy', '--role-name', $RoleName, '--policy-arn', $arn))
            if ($arn -notmatch '^arn:aws[^:]*:iam::aws:policy/') {
                [void]$customerManaged.Add([PSCustomObject]@{ Arn = $arn; Name = $pn })
            }
        }
    }

    [void](Invoke-GuardedDelete -NoRegion -Label $label -Retries 3 -RetryDelaySeconds 10 `
        -Arguments @('iam', 'delete-role', '--role-name', $RoleName))

    foreach ($cm in $customerManaged) {
        $ptags = Get-IamPolicyTags -PolicyArn $cm.Arn
        if (-not (Test-Eligible -Label "iam managed policy $($cm.Name)" -TagMap $ptags)) { continue }
        [void](Invoke-GuardedDelete -NoRegion -Label "iam managed policy $($cm.Name)" `
            -Arguments @('iam', 'delete-policy', '--policy-arn', $cm.Arn))
    }
}

function Remove-IamResources {
    param($Roles, $InstanceProfiles)
    Write-Step '13a/13  IAM instance profiles, then inline policies + roles'
    if (@($InstanceProfiles).Count -eq 0 -and @($Roles).Count -eq 0) { Write-Info 'nothing to consider'; return }

    # Instance profile first: a role cannot be deleted while a profile holds it.
    foreach ($p in @($InstanceProfiles)) { Remove-IamInstanceProfile -ProfileName $p }
    foreach ($r in @($Roles)) { Remove-IamRole -RoleName $r }
}

# ---------------------------------------------------------------------------
# 12. CloudWatch log group
# ---------------------------------------------------------------------------

function Remove-LogGroups {
    param($Names)
    Write-Step '13b/13  CloudWatch Logs log group'
    if (@($Names).Count -eq 0) { Write-Info 'nothing to consider'; return }

    foreach ($name in @($Names)) {
        $label = "log group $name"
        $desc = Invoke-Aws -Arguments @('logs', 'describe-log-groups',
            '--log-group-name-prefix', $name, '--output', 'json')
        $exists = $false
        if ($desc.Ok) {
            foreach ($g in @(Get-Prop $desc.Json 'logGroups')) {
                if ([string](Get-Prop $g 'logGroupName') -eq $name) { $exists = $true }
            }
        }
        if (-not $exists) { Add-AlreadyGone $label; continue }

        $tags = Get-LogGroupTags -LogGroupName $name
        if (-not (Test-Eligible -Label $label -TagMap $tags)) { continue }

        if (@($script:SharedLogGroups) -contains $name -and -not $AllowSharedLogGroup) {
            Add-Skipped $label "AWS default/shared log group -- pass -AllowSharedLogGroup to include it"
            continue
        }

        [void](Invoke-GuardedDelete -Label $label `
            -Arguments @('logs', 'delete-log-group', '--log-group-name', $name))
    }
}

# ---------------------------------------------------------------------------
# 13. The $20 budget
# ---------------------------------------------------------------------------

function Remove-Budget {
    param([string]$BudgetName, $ExpectedLimit)
    Write-Step '13c/13  AWS Budget'
    if ([string]::IsNullOrWhiteSpace($BudgetName)) { Write-Info 'nothing to consider'; return }
    if ([string]::IsNullOrWhiteSpace($script:AccountId)) {
        Add-Skipped "budget $BudgetName" 'account id unknown; cannot address the Budgets API'
        return
    }

    $label = "budget $BudgetName"
    # Budgets is a global service reached through us-east-1.
    $desc = Invoke-Aws -RegionOverride 'us-east-1' -Arguments @('budgets', 'describe-budget',
        '--account-id', $script:AccountId, '--budget-name', $BudgetName, '--output', 'json')
    if (-not $desc.Ok) {
        if (Test-NotFound $desc.Text) { Add-AlreadyGone $label; return }
        Add-Skipped $label "could not read it ($($desc.Text)); refusing to touch it"
        return
    }

    # The Budgets API carries NO tags, so the tag gate cannot apply here. Name and
    # limit are the whole proof: an exact name match plus the expected dollar
    # amount. Anything else is left alone.
    $budget = Get-Prop $desc.Json 'Budget'
    $actualName = [string](Get-Prop $budget 'BudgetName')
    $limit = Get-Prop $budget 'BudgetLimit'
    $amount = [string](Get-Prop $limit 'Amount')
    $unit = [string](Get-Prop $limit 'Unit')

    if ($actualName -ne $BudgetName) {
        Add-Skipped $label "name mismatch (API returned '$actualName'); refusing to touch it"
        return
    }
    $amountOk = $false
    if (-not [string]::IsNullOrWhiteSpace($amount)) {
        $parsed = 0.0
        if ([double]::TryParse($amount, [ref]$parsed)) {
            if ([Math]::Abs($parsed - [double]$ExpectedLimit) -lt 0.01) { $amountOk = $true }
        }
    }
    if (-not $amountOk) {
        Add-Skipped $label "limit is $amount $unit, expected $ExpectedLimit USD -- not the budget this stack created; refusing to touch it"
        return
    }
    Write-Info "$label -- untagged service; authorised by exact name match + $ExpectedLimit USD limit"

    [void](Invoke-GuardedDelete -RegionOverride 'us-east-1' -Label $label `
        -Arguments @('budgets', 'delete-budget', '--account-id', $script:AccountId, '--budget-name', $BudgetName))
}

# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

function Test-Teardown {
    param($ComputeEnvNames, $BucketNames)
    Write-Head 'VERIFICATION'

    $clean = $true

    Write-Step 'Re-running the tag sweep (an empty list is the success criterion)'
    $r = Invoke-Aws -Arguments @('resourcegroupstaggingapi', 'get-resources',
        '--tag-filters', 'Key=Project,Values=BioSimulateAI', 'Key=Component,Values=CC3D-Compute',
        '--output', 'json')
    if (-not $r.Ok) {
        Write-Bad "tag sweep could not run: $($r.Text)"
        $clean = $false
    } else {
        Write-Host $r.Text -ForegroundColor DarkGray
        $left = @(Get-Prop $r.Json 'ResourceTagMappingList')
        if ($left.Count -eq 0) {
            Write-Host '   RESULT: [] -- no resource in us-east-2 still carries both tags.' -ForegroundColor Green
        } else {
            $clean = $false
            Write-Host "   RESULT: $($left.Count) tagged resource(s) still present:" -ForegroundColor Red
            foreach ($item in $left) { Write-Host "     $(Get-Prop $item 'ResourceARN')" -ForegroundColor Red }
        }
    }

    Write-Step 'Explicit re-check: compute environment'
    if (@($ComputeEnvNames).Count -eq 0) { Write-Info 'no compute environment name to re-check' }
    foreach ($name in @($ComputeEnvNames)) {
        $c = Invoke-Aws -Arguments @('batch', 'describe-compute-environments',
            '--compute-environments', $name, '--output', 'json')
        if (-not $c.Ok) {
            if (Test-NotFound $c.Text) { Write-Host "   $name -- gone" -ForegroundColor Green }
            else { Write-Bad "$name -- could not verify: $($c.Text)"; $clean = $false }
            continue
        }
        $items = @(Get-Prop $c.Json 'computeEnvironments')
        if ($items.Count -eq 0) { Write-Host "   $name -- gone (computeEnvironments: [])" -ForegroundColor Green }
        else {
            $clean = $false
            Write-Host "   $name -- STILL PRESENT (state=$(Get-Prop $items[0] 'state') status=$(Get-Prop $items[0] 'status'))" -ForegroundColor Red
        }
    }

    Write-Step 'Explicit re-check: S3 bucket'
    if (@($BucketNames).Count -eq 0) { Write-Info 'no bucket name to re-check' }
    foreach ($bucket in @($BucketNames)) {
        $h = Invoke-Aws -Arguments @('s3api', 'head-bucket', '--bucket', $bucket)
        if ($h.Ok) {
            $clean = $false
            Write-Host "   s3://$bucket -- STILL PRESENT" -ForegroundColor Red
        } elseif (Test-NotFound $h.Text) {
            Write-Host "   s3://$bucket -- gone (404)" -ForegroundColor Green
        } else {
            Write-Host "   s3://$bucket -- head-bucket says: $($h.Text)" -ForegroundColor Yellow
        }
    }

    return $clean
}

# ---------------------------------------------------------------------------
# Mode + identity
# ---------------------------------------------------------------------------

function Initialize-Mode {
    if ($Confirm -and $WhatIf) {
        Write-Host 'Pass either -WhatIf or -Confirm, not both. Refusing to guess.' -ForegroundColor Red
        exit 2
    }
    $script:DryRun = (-not $Confirm)

    Write-Head 'CompuCell3D on-demand compute -- AWS TEARDOWN'
    if ($script:DryRun) {
        Write-Host '  MODE: -WhatIf (default). Nothing will be deleted.' -ForegroundColor Yellow
        Write-Host '  Re-run with -Confirm to actually delete.' -ForegroundColor Yellow
    } else {
        Write-Host '  MODE: -Confirm. Resources WILL be deleted.' -ForegroundColor Red
    }
    Write-Host "  Region: $Region"
    Write-Host "  Gate:   Project=BioSimulateAI AND Component=CC3D-Compute (both, or skipped)"
    Write-Host '  Bedrock and every untagged/pre-existing resource in this account are out of scope.'
}

function Initialize-Identity {
    $r = Invoke-Aws -Arguments @('sts', 'get-caller-identity', '--output', 'json')
    if (-not $r.Ok) {
        Write-Host ''
        Write-Host 'Cannot reach AWS with the current credentials:' -ForegroundColor Red
        Write-Host $r.Text -ForegroundColor Red
        Write-Host ''
        Write-Host 'Refresh the session first (aws sso login, or aws configure), then re-run.' -ForegroundColor Yellow
        exit 2
    }
    $script:AccountId = [string](Get-Prop $r.Json 'Account')
    Write-Info "account: $script:AccountId  arn: $(Get-Prop $r.Json 'Arn')"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

Initialize-Mode
Initialize-AwsCli
Initialize-Identity

$manifest = Import-ResourceManifest

Write-Step 'Discovery (b): independent tag sweep'
$taggedArns = Get-TaggedResources
if ($null -eq $taggedArns) {
    Write-Info 'tag sweep unavailable -- continuing with the manifest and defaults only'
    $taggedArns = New-Object System.Collections.ArrayList
}
Write-Info "tag sweep found $($taggedArns.Count) resource(s) carrying both tags"
foreach ($a in $taggedArns) { Write-Info "  $a" }
$byType = Group-TaggedArns $taggedArns

# ---- Candidate sets: manifest UNION tag sweep UNION documented defaults -------

$computeEnvs = Merge-Candidates (Get-Prop $manifest 'ComputeEnvironment') `
    (Get-DiscoveredIds $byType 'batch/compute-environment') $script:Defaults.ComputeEnvironment
$jobQueues = Merge-Candidates (Get-Prop $manifest 'JobQueue') `
    (Get-DiscoveredIds $byType 'batch/job-queue') $script:Defaults.JobQueue
$jobDefs = Merge-Candidates (Get-Prop $manifest 'JobDefinition') `
    (Get-DiscoveredIds $byType 'batch/job-definition') $script:Defaults.JobDefinition
$launchTemplates = Merge-Candidates (Get-Prop $manifest 'LaunchTemplate') `
    (Get-DiscoveredIds $byType 'ec2/launch-template') $script:Defaults.LaunchTemplate
$buckets = Merge-Candidates (Get-Prop $manifest 'Bucket') `
    (Get-DiscoveredIds $byType 's3/') $null
# S3 ARNs are arn:aws:s3:::name -- no resource-type segment -- so read the names
# straight off the sweep as well, in case the grouping key differs by partition.
$bucketArnNames = New-Object System.Collections.ArrayList
foreach ($a in @($taggedArns)) {
    if ($a -match '^arn:aws[^:]*:s3:::([^/]+)$') { [void]$bucketArnNames.Add($Matches[1]) }
}
$buckets = Merge-Candidates $buckets $bucketArnNames $null
$ecrRepos = Merge-Candidates (Get-Prop $manifest 'EcrRepository') `
    (Get-DiscoveredIds $byType 'ecr/repository') $script:Defaults.EcrRepository
$vpcEndpoints = Merge-Candidates (Get-Prop $manifest 'VpcEndpoint') `
    (Get-DiscoveredIds $byType 'ec2/vpc-endpoint') $null
$securityGroups = Merge-Candidates (Get-Prop $manifest 'SecurityGroup') `
    (Get-DiscoveredIds $byType 'ec2/security-group') $null
$internetGateways = Merge-Candidates (Get-Prop $manifest 'InternetGateway') `
    (Get-DiscoveredIds $byType 'ec2/internet-gateway') $null
$subnets = Merge-Candidates (Get-Prop $manifest 'Subnets') `
    (Get-DiscoveredIds $byType 'ec2/subnet') $null
$routeTables = Merge-Candidates (Get-Prop $manifest 'RouteTables') `
    (Get-DiscoveredIds $byType 'ec2/route-table') $null
$vpcs = Merge-Candidates (Get-Prop $manifest 'Vpc') `
    (Get-DiscoveredIds $byType 'ec2/vpc') $null
$roles = Merge-Candidates (Get-Prop $manifest 'IamRoles') `
    (Get-DiscoveredIds $byType 'iam/role') $script:Defaults.Roles
$instanceProfiles = Merge-Candidates (Get-Prop $manifest 'IamInstanceProfiles') `
    (Get-DiscoveredIds $byType 'iam/instance-profile') $script:Defaults.InstanceProfiles
$logGroups = Merge-Candidates (Get-Prop $manifest 'LogGroup') `
    (Get-DiscoveredIds $byType 'logs/log-group') $script:Defaults.LogGroup

$budgetName = [string](Get-Prop $manifest 'BudgetName')
if ([string]::IsNullOrWhiteSpace($budgetName)) { $budgetName = $script:Defaults.BudgetName }
$budgetLimit = Get-Prop $manifest 'BudgetLimit'
if ($null -eq $budgetLimit -or [string]::IsNullOrWhiteSpace([string]$budgetLimit)) {
    $budgetLimit = $script:Defaults.BudgetLimit
}

# VPCs that prove both tags themselves -- the only parents allowed to authorise a
# child whose own tags are missing.
$script:TaggedVpcIds = New-Object System.Collections.ArrayList
foreach ($v in @($vpcs)) {
    $t = Get-Ec2Tags -ResourceId $v
    if (Test-RequiredTags $t) { [void]$script:TaggedVpcIds.Add($v) }
}
Write-Info "tag-verified VPC(s): $((@($script:TaggedVpcIds) -join ', '))"

# If a tagged VPC exists but its children were never recorded, enumerate them so
# an orphaned subnet or route table is not left behind.
foreach ($vpcId in @($script:TaggedVpcIds)) {
    $q = Invoke-Aws -Arguments @('ec2', 'describe-subnets', '--filters', "Name=vpc-id,Values=$vpcId", '--output', 'json')
    if ($q.Ok) { foreach ($s in @(Get-Prop $q.Json 'Subnets')) { $subnets = Merge-Candidates $subnets @([string](Get-Prop $s 'SubnetId')) $null } }

    $q = Invoke-Aws -Arguments @('ec2', 'describe-route-tables', '--filters', "Name=vpc-id,Values=$vpcId", '--output', 'json')
    if ($q.Ok) { foreach ($s in @(Get-Prop $q.Json 'RouteTables')) { $routeTables = Merge-Candidates $routeTables @([string](Get-Prop $s 'RouteTableId')) $null } }

    $q = Invoke-Aws -Arguments @('ec2', 'describe-security-groups', '--filters', "Name=vpc-id,Values=$vpcId", '--output', 'json')
    if ($q.Ok) { foreach ($s in @(Get-Prop $q.Json 'SecurityGroups')) { $securityGroups = Merge-Candidates $securityGroups @([string](Get-Prop $s 'GroupId')) $null } }

    $q = Invoke-Aws -Arguments @('ec2', 'describe-internet-gateways', '--filters', "Name=attachment.vpc-id,Values=$vpcId", '--output', 'json')
    if ($q.Ok) { foreach ($s in @(Get-Prop $q.Json 'InternetGateways')) { $internetGateways = Merge-Candidates $internetGateways @([string](Get-Prop $s 'InternetGatewayId')) $null } }

    $q = Invoke-Aws -Arguments @('ec2', 'describe-vpc-endpoints', '--filters', "Name=vpc-id,Values=$vpcId", '--output', 'json')
    if ($q.Ok) { foreach ($s in @(Get-Prop $q.Json 'VpcEndpoints')) { $vpcEndpoints = Merge-Candidates $vpcEndpoints @([string](Get-Prop $s 'VpcEndpointId')) $null } }
}

Write-Head 'DELETION'

Remove-JobQueues            $jobQueues
Remove-JobDefinitions       $jobDefs
Remove-ComputeEnvironments  $computeEnvs
Remove-LaunchTemplates      $launchTemplates
Remove-Buckets              $buckets
Remove-EcrRepositories      $ecrRepos
Remove-VpcEndpoints         $vpcEndpoints      $script:TaggedVpcIds
Remove-SecurityGroups       $securityGroups    $script:TaggedVpcIds
Remove-InternetGateways     $internetGateways  $script:TaggedVpcIds
Remove-Subnets              $subnets           $script:TaggedVpcIds
Remove-RouteTables          $routeTables       $script:TaggedVpcIds
Remove-Vpcs                 $vpcs
Remove-IamResources         $roles             $instanceProfiles
Remove-LogGroups            $logGroups
Remove-Budget               $budgetName        $budgetLimit

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

Write-Head 'SUMMARY'

if ($script:DryRun) {
    Write-Host "  Would delete: $($script:Planned.Count)" -ForegroundColor Yellow
    foreach ($p in $script:Planned) { Write-Host "    $p" -ForegroundColor Yellow }
} else {
    Write-Host "  Deleted:      $($script:Deleted.Count)" -ForegroundColor Green
    foreach ($p in $script:Deleted) { Write-Host "    $p" -ForegroundColor Green }
}
Write-Host "  Already gone: $($script:AlreadyGone.Count)" -ForegroundColor DarkGray
foreach ($p in $script:AlreadyGone) { Write-Host "    $p" -ForegroundColor DarkGray }
Write-Host "  Skipped:      $($script:Skipped.Count)" -ForegroundColor Magenta
foreach ($p in $script:Skipped) { Write-Host "    $p" -ForegroundColor Magenta }
Write-Host "  Failed:       $($script:Failed.Count)" -ForegroundColor Red
foreach ($p in $script:Failed) { Write-Host "    $p" -ForegroundColor Red }

if ($script:DryRun) {
    Write-Host ''
    Write-Host '  Dry run only -- nothing was deleted. Re-run with -Confirm to execute.' -ForegroundColor Yellow
    Write-Host '  (Verification below still runs read-only, so you can see the current state.)' -ForegroundColor Yellow
}

$verified = Test-Teardown -ComputeEnvNames $computeEnvs -BucketNames $buckets

Write-Host ''
if ($script:DryRun) {
    Write-Host 'Dry run complete. No changes were made.' -ForegroundColor Yellow
    exit 0
}
if ($verified -and $script:Failed.Count -eq 0) {
    Write-Host 'TEARDOWN VERIFIED: the tag sweep is empty and the explicit re-checks agree.' -ForegroundColor Green
    exit 0
}
Write-Host 'TEARDOWN INCOMPLETE: see Failed/Skipped above and the verification output.' -ForegroundColor Red
Write-Host 'Re-running the script is safe -- every delete is idempotent.' -ForegroundColor Yellow
exit 1
