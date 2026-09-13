#Requires -Version 5.1
<#
===============================================================================
 build_image_codebuild.ps1 -- build the CompuCell3D worker image INSIDE AWS
===============================================================================

 Why: AWS Batch only runs containers, and CompuCell3D is not a pip package (conda
 only, plus VTK and native GL libraries), so the engine has to be baked into a
 Linux image. That image is 3-4 GB. Building it locally then pushing means sending
 3-4 GB across a workstation's uplink -- the slowest, most failure-prone step in
 the whole setup, and it needs a running Docker daemon. CodeBuild does the build
 and the push inside the region instead, so no local Docker is required and the
 bytes never leave AWS.

 COST: a CodeBuild project is $0 at rest and billed per build-minute. The default
 BUILD_GENERAL1_MEDIUM is ~$0.01/min, so a 25-minute build is about $0.25. MEDIUM
 rather than SMALL deliberately: the conda solve for compucell3d + VTK is memory
 hungry and an OOM halfway through a 20-minute solve costs more than the upgrade.

 Creates: an IAM role, a CodeBuild project, and one S3 source object. No NAT
 Gateway, no Elastic IP, no always-on capacity -- nothing that bills while idle.

 USAGE
   .\build_image_codebuild.ps1                 # plan only, creates nothing
   .\build_image_codebuild.ps1 -Confirm        # create, build, wait, verify
===============================================================================
#>

[CmdletBinding()]
param(
    [string]$Region       = 'us-east-2',
    [string]$NamePrefix   = 'biosim-cc3d',
    [string]$RepoName     = 'cc3d-worker',
    [string]$ImageTag     = 'latest',
    [ValidateSet('BUILD_GENERAL1_SMALL', 'BUILD_GENERAL1_MEDIUM', 'BUILD_GENERAL1_LARGE')]
    [string]$ComputeType  = 'BUILD_GENERAL1_MEDIUM',
    # Minutes to wait for the build before giving up watching it (the build itself
    # keeps running in AWS; this only bounds the polling).
    [int]$WaitMinutes     = 45,
    [switch]$Confirm
)

# Native-command stderr must not be fatal: the AWS CLI writes expected 'not found'
# messages to stderr, and under ErrorActionPreference='Stop' that terminates the
# script mid-way. Exit codes are checked explicitly instead.
$ErrorActionPreference = 'Continue'
$ProgressPreference    = 'SilentlyContinue'

$TagProject   = 'BioSimulateAI'
$TagComponent = 'CC3D-Compute'
$ScriptDir    = Split-Path -Parent $MyInvocation.MyCommand.Path

function Say  { param($m) Write-Host $m }
function Made { param($m) Write-Host "     + created  $m" -ForegroundColor Green }
function Have { param($m) Write-Host "     = exists   $m" -ForegroundColor DarkGray }
function Die  { param($m, $d) Write-Host "`nFAILED: $m" -ForegroundColor Red
                if ($d) { Write-Host $d -ForegroundColor DarkYellow }
                exit 1 }

function Invoke-AwsCli {
    param([string[]]$A, [switch]$AllowFail)
    $out  = & aws.exe @A 2>&1
    $code = $LASTEXITCODE
    $text = ($out | Out-String).Trim()
    if ($code -ne 0 -and -not $AllowFail) {
        Die "aws $($A -join ' ') failed (exit $code)." $text
    }
    return [pscustomobject]@{ ExitCode = $code; Text = $text }
}

# -----------------------------------------------------------------------------
# Identity
# -----------------------------------------------------------------------------
$who = Invoke-AwsCli @('sts','get-caller-identity','--output','json')
try { $ident = $who.Text | ConvertFrom-Json } catch { Die 'Could not read AWS identity.' $who.Text }
$AccountId = $ident.Account

$RoleName    = "$NamePrefix-codebuild-role"
$PolicyName  = "$NamePrefix-codebuild-inline"
$ProjectName = "$NamePrefix-image-build"
$Bucket      = "$NamePrefix-$AccountId-$Region"
$SourceKey   = 'codebuild-source/cc3d-worker-src.zip'
$Registry    = "$AccountId.dkr.ecr.$Region.amazonaws.com"

Say ''
Say '=============================================================================='
Say 'BioSimulateAI :: build the CompuCell3D worker image in AWS CodeBuild'
Say '=============================================================================='
Say ''
Say "  Account      : $AccountId"
Say "  Region       : $Region"
Say "  ECR image    : $Registry/$RepoName`:$ImageTag"
Say "  Source       : s3://$Bucket/$SourceKey"
Say "  Project      : $ProjectName  ($ComputeType, privileged)"
Say ''
Say '  WILL CREATE                                        IDLE COST'
Say "  IAM role $RoleName            `$0.00  (IAM is free)"
Say "  CodeBuild project $ProjectName       `$0.00  (billed per build-minute only)"
Say "  S3 object $SourceKey    ~`$0.00  (a few KB in the existing bucket)"
Say ''
Say '  Build cost: ~$0.01/min on MEDIUM, so a 25-minute build is about $0.25.'
Say '  Nothing here bills while idle.'
Say ''

if (-not $Confirm) {
    Say 'PLAN ONLY -- nothing was created. Re-run with -Confirm to build.'
    exit 0
}

# -----------------------------------------------------------------------------
# 1. IAM role for CodeBuild
# -----------------------------------------------------------------------------
Say '[1] IAM role for CodeBuild'
$trust = @{
    Version   = '2012-10-17'
    Statement = @(@{
        Effect    = 'Allow'
        Principal = @{ Service = 'codebuild.amazonaws.com' }
        Action    = 'sts:AssumeRole'
    })
} | ConvertTo-Json -Depth 10 -Compress

$trustFile = Join-Path $env:TEMP "cb-trust-$([guid]::NewGuid().ToString('N')).json"
Set-Content -LiteralPath $trustFile -Value $trust -Encoding ascii

$existing = Invoke-AwsCli @('iam','get-role','--role-name',$RoleName,'--output','json') -AllowFail
if ($existing.ExitCode -eq 0) {
    Have "role $RoleName"
} else {
    Invoke-AwsCli @('iam','create-role','--role-name',$RoleName,
          '--assume-role-policy-document',"file://$trustFile",
          '--description','Builds the BioSimulateAI CompuCell3D worker image',
          '--tags',"Key=Project,Value=$TagProject","Key=Component,Value=$TagComponent",
          '--output','json') | Out-Null
    Made "role $RoleName"
}
Remove-Item -LiteralPath $trustFile -Force -ErrorAction SilentlyContinue

# Least privilege: push to ONE ECR repo, write its own logs, read ONE source object.
$policy = @{
    Version   = '2012-10-17'
    Statement = @(
        @{ Sid = 'EcrAuth'; Effect = 'Allow'; Action = @('ecr:GetAuthorizationToken'); Resource = @('*') },
        @{ Sid = 'EcrPushThisRepoOnly'; Effect = 'Allow'
           Action = @('ecr:BatchCheckLayerAvailability','ecr:InitiateLayerUpload',
                      'ecr:UploadLayerPart','ecr:CompleteLayerUpload','ecr:PutImage',
                      'ecr:BatchGetImage','ecr:GetDownloadUrlForLayer')
           Resource = @("arn:aws:ecr:${Region}:${AccountId}:repository/$RepoName") },
        @{ Sid = 'Logs'; Effect = 'Allow'
           Action = @('logs:CreateLogGroup','logs:CreateLogStream','logs:PutLogEvents')
           Resource = @("arn:aws:logs:${Region}:${AccountId}:log-group:/aws/codebuild/$ProjectName",
                        "arn:aws:logs:${Region}:${AccountId}:log-group:/aws/codebuild/${ProjectName}:*") },
        @{ Sid = 'ReadSourceObjectOnly'; Effect = 'Allow'
           Action = @('s3:GetObject','s3:GetObjectVersion')
           Resource = @("arn:aws:s3:::$Bucket/$SourceKey") }
    )
} | ConvertTo-Json -Depth 10 -Compress

$polFile = Join-Path $env:TEMP "cb-pol-$([guid]::NewGuid().ToString('N')).json"
Set-Content -LiteralPath $polFile -Value $policy -Encoding ascii
Invoke-AwsCli @('iam','put-role-policy','--role-name',$RoleName,'--policy-name',$PolicyName,
      '--policy-document',"file://$polFile") | Out-Null
Made "inline policy $PolicyName"
Remove-Item -LiteralPath $polFile -Force -ErrorAction SilentlyContinue

$RoleArn = "arn:aws:iam::${AccountId}:role/$RoleName"

# IAM is eventually consistent and CodeBuild validates sts:AssumeRole against the
# trust policy at CREATE time. This wait has to come BEFORE create-project: putting
# it after meant the very first run always failed with
# "CodeBuild is not authorized to perform: sts:AssumeRole on service role".
Say '     waiting 20s for the IAM role trust policy to propagate'
Start-Sleep -Seconds 20

# -----------------------------------------------------------------------------
# 2. Zip the build context and upload it
# -----------------------------------------------------------------------------
Say ''
Say '[2] Build context -> S3'
$staging = Join-Path $env:TEMP "cc3d-src-$([guid]::NewGuid().ToString('N'))"
New-Item -ItemType Directory -Path $staging -Force | Out-Null
foreach ($f in @('Dockerfile','cc3d_job_runner.py','buildspec.yml')) {
    $src = Join-Path $ScriptDir $f
    if (-not (Test-Path -LiteralPath $src)) { Die "Missing build input: $src" }
    Copy-Item -LiteralPath $src -Destination $staging -Force
}
$zipPath = Join-Path $env:TEMP "cc3d-worker-src-$([guid]::NewGuid().ToString('N')).zip"
Compress-Archive -Path (Join-Path $staging '*') -DestinationPath $zipPath -Force
$zipKb = [math]::Round((Get-Item $zipPath).Length / 1KB, 1)
Say "     packed Dockerfile + cc3d_job_runner.py + buildspec.yml ($zipKb KB)"

Invoke-AwsCli @('s3','cp',$zipPath,"s3://$Bucket/$SourceKey",'--region',$Region) | Out-Null
Made "s3://$Bucket/$SourceKey"
Remove-Item -LiteralPath $zipPath -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath $staging -Recurse -Force -ErrorAction SilentlyContinue

# -----------------------------------------------------------------------------
# 3. CodeBuild project
# -----------------------------------------------------------------------------
Say ''
Say '[3] CodeBuild project'
$envVars = @(
    @{ name = 'ECR_REGISTRY';   value = $Registry; type = 'PLAINTEXT' },
    @{ name = 'IMAGE_REPO_NAME'; value = $RepoName; type = 'PLAINTEXT' },
    @{ name = 'IMAGE_TAG';      value = $ImageTag; type = 'PLAINTEXT' }
)
$sourceSpec = @{ type = 'S3'; location = "$Bucket/$SourceKey" }
$artifacts  = @{ type = 'NO_ARTIFACTS' }
$envSpec    = @{
    type                = 'LINUX_CONTAINER'
    image               = 'aws/codebuild/standard:7.0'
    computeType         = $ComputeType
    privilegedMode      = $true     # docker build needs a daemon
    environmentVariables = $envVars
}

$srcFile = Join-Path $env:TEMP "cb-src-$([guid]::NewGuid().ToString('N')).json"
$envFile = Join-Path $env:TEMP "cb-env-$([guid]::NewGuid().ToString('N')).json"
$artFile = Join-Path $env:TEMP "cb-art-$([guid]::NewGuid().ToString('N')).json"
Set-Content -LiteralPath $srcFile -Value ($sourceSpec | ConvertTo-Json -Depth 10 -Compress) -Encoding ascii
Set-Content -LiteralPath $envFile -Value ($envSpec    | ConvertTo-Json -Depth 10 -Compress) -Encoding ascii
Set-Content -LiteralPath $artFile -Value ($artifacts  | ConvertTo-Json -Depth 10 -Compress) -Encoding ascii

$proj = Invoke-AwsCli @('codebuild','batch-get-projects','--names',$ProjectName,'--region',$Region,'--output','json') -AllowFail
$projExists = $false
if ($proj.ExitCode -eq 0) {
    try { $projExists = ((($proj.Text | ConvertFrom-Json).projects).Count -gt 0) } catch { $projExists = $false }
}

$verb = if ($projExists) { 'update-project' } else { 'create-project' }
Invoke-AwsCli @('codebuild',$verb,'--name',$ProjectName,
      '--source',"file://$srcFile",
      '--artifacts',"file://$artFile",
      '--environment',"file://$envFile",
      '--service-role',$RoleArn,
      '--timeout-in-minutes','60',
      '--region',$Region,'--output','json') | Out-Null
if ($projExists) { Have "project $ProjectName (updated)" } else { Made "project $ProjectName" }

# Tag it so teardown finds it by tag like everything else.
Invoke-AwsCli @('codebuild','update-project','--name',$ProjectName,
      '--tags',"key=Project,value=$TagProject","key=Component,value=$TagComponent",
      '--region',$Region,'--output','json') -AllowFail | Out-Null

foreach ($f in @($srcFile,$envFile,$artFile)) { Remove-Item -LiteralPath $f -Force -ErrorAction SilentlyContinue }

# -----------------------------------------------------------------------------
# 4. Start the build and watch it
# -----------------------------------------------------------------------------
Say ''
Say '[4] Starting the build'
$start = Invoke-AwsCli @('codebuild','start-build','--project-name',$ProjectName,'--region',$Region,'--output','json')
try { $buildId = ($start.Text | ConvertFrom-Json).build.id } catch { Die 'Could not read the build id.' $start.Text }
Say "     build id: $buildId"
Say "     watching for up to $WaitMinutes minutes (the conda solve is the slow part)"

$deadline = (Get-Date).AddMinutes($WaitMinutes)
$lastPhase = ''
$status = 'IN_PROGRESS'
while ((Get-Date) -lt $deadline) {
    Start-Sleep -Seconds 20
    $b = Invoke-AwsCli @('codebuild','batch-get-builds','--ids',$buildId,'--region',$Region,'--output','json') -AllowFail
    if ($b.ExitCode -ne 0) { continue }
    try { $build = ($b.Text | ConvertFrom-Json).builds[0] } catch { continue }
    $status = $build.buildStatus
    $phase  = $build.currentPhase
    if ($phase -ne $lastPhase) {
        $lastPhase = $phase
        Say ("     [{0:HH:mm:ss}] {1}  status={2}" -f (Get-Date), $phase, $status)
    }
    if ($status -ne 'IN_PROGRESS') { break }
}

Say ''
Say "  final status: $status"
if ($status -ne 'SUCCEEDED') {
    Say ''
    Say '  Build log:'
    Say "    aws logs tail /aws/codebuild/$ProjectName --region $Region --since 1h"
    Die "The image build did not succeed (status=$status)." `
        "Nothing was pushed to ECR, so the job definition still points at a missing image."
}

# -----------------------------------------------------------------------------
# 5. Verify the image really is in ECR
# -----------------------------------------------------------------------------
Say ''
Say '[5] Verifying the image in ECR'
$img = Invoke-AwsCli @('ecr','describe-images','--repository-name',$RepoName,
             '--image-ids',"imageTag=$ImageTag",'--region',$Region,'--output','json') -AllowFail
if ($img.ExitCode -ne 0) {
    Die 'CodeBuild reported success but no image with that tag is in ECR.' $img.Text
}
try {
    $d = ($img.Text | ConvertFrom-Json).imageDetails[0]
    $mb = [math]::Round($d.imageSizeInBytes / 1MB, 0)
    Say "     tag=$ImageTag  size=${mb} MB  pushed=$($d.imagePushedAt)"
    Say "     digest=$($d.imageDigest)"
} catch {
    Say '     (image present; could not parse the detail payload)'
}

Say ''
Say '=============================================================================='
Say 'IMAGE READY'
Say '=============================================================================='
Say "  $Registry/$RepoName`:$ImageTag"
Say ''
Say '  The Batch job definition already points at this image, so a simulation can'
Say '  be submitted now. Idle cost is unchanged apart from ECR image storage.'
Say ''
exit 0

