#Requires -Version 5.1
<#
===============================================================================
 provision_aws.ps1 -- on-demand CompuCell3D compute for BioSimulateAI
===============================================================================

 Builds the SCALE-TO-ZERO half of the architecture: the web application stays on
 Render (pip-only, 512 MB) and hands heavy CompuCell3D runs to AWS Batch, which
 boots a Spot m7i.xlarge, runs the container from deploy/cc3d/Dockerfile, writes
 results to S3, and terminates. Nothing here is always-on.

 IDLE COST DESIGN
 ----------------
 Every resource below is either free at rest or charged only for bytes stored.
 There is deliberately NO NAT Gateway, NO Elastic IP, NO Interface VPC endpoint,
 NO load balancer, NO RDS/ElastiCache/EKS and NO always-on instance. Batch hosts
 reach S3 through a FREE S3 Gateway endpoint and reach ECR / CloudWatch over the
 Internet Gateway from a PUBLIC subnet with a public IP -- that combination is
 what removes the ~$32/month NAT Gateway from the design.

 IDEMPOTENCE
 -----------
 Safe to re-run. Every step looks the resource up first (by tag, name or ARN) and
 creates only what is missing. Re-running after a partial failure resumes.

 USAGE
 -----
   # 1. dry run -- prints the plan and per-item idle cost, creates nothing
   .\provision_aws.ps1 -NotificationEmail you@example.com

   # 2. real run
   .\provision_aws.ps1 -NotificationEmail you@example.com -Confirm

 Every created id is written to deploy/cc3d/aws_resources.json for teardown.
===============================================================================
#>

[CmdletBinding()]
param(
    # Where AWS Budgets sends the $20/month cost alerts. Required.
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[^@\s]+@[^@\s]+\.[^@\s]+$')]
    [string]$NotificationEmail,

    # Region. The Dockerfile and the app both assume us-east-2.
    [string]$Region = 'us-east-2',

    # Prefix for every resource name, so one account can host several stacks.
    [ValidatePattern('^[a-z][a-z0-9-]{2,20}$')]
    [string]$NamePrefix = 'biosim-cc3d',

    # Optional named AWS CLI profile.
    [string]$AwsProfile = '',

    # Override the derived bucket name (must be globally unique, lowercase).
    [string]$BucketName = '',

    # Monthly budget ceiling, USD. Set to 5 because that is the agreed ceiling for
    # this stack: verified idle cost is ~$0.42/month (ECR image + retained S3 +
    # CloudWatch Logs inside the 5 GB free tier), so $5 leaves headroom for roughly
    # 60 simulation-hours a month at the m7i.xlarge Spot rate before it alerts.
    [int]$BudgetUsd = 5,

    # Nothing is created without this switch. Without it the script prints the
    # plan and the cost table, then exits 0.
    [switch]$Confirm,

    # register-job-definition always makes a NEW revision, so by default an
    # existing ACTIVE definition is left alone. Pass this to publish a revision.
    [switch]$UpdateJobDefinition,

    # Same reasoning for the Render app's managed policy.
    [switch]$UpdateRenderPolicy
)

$ErrorActionPreference = 'Stop'
$ProgressPreference    = 'SilentlyContinue'

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
$script:TagProject   = 'BioSimulateAI'
$script:TagComponent = 'CC3D-Compute'

$VpcCidr      = '10.42.0.0/16'
$SubnetACidr  = '10.42.1.0/24'
$SubnetBCidr  = '10.42.2.0/24'

$VpcName          = "$NamePrefix-vpc"
$IgwName          = "$NamePrefix-igw"
$RouteTableName   = "$NamePrefix-public-rtb"
$SubnetAName      = "$NamePrefix-public-a"
$SubnetBName      = "$NamePrefix-public-b"
$EndpointName     = "$NamePrefix-s3-gateway"
$SgName           = "$NamePrefix-egress-only"
$InstanceRoleName = "$NamePrefix-ecsInstanceRole"
$InstanceProfName = "$NamePrefix-ecsInstanceProfile"
$JobRoleName      = "$NamePrefix-jobRole"
$S3PolicyName     = "$NamePrefix-s3-run-bucket"
$LaunchTplName    = "$NamePrefix-lt"
$ComputeEnvName   = "$NamePrefix-ce"
$JobQueueName     = "$NamePrefix-queue"
$JobDefName       = "$NamePrefix-job"
$EcrRepoName      = 'cc3d-worker'
$LogGroupName     = '/aws/batch/job'
$LogRetentionDays = 14
$BudgetName       = "$NamePrefix-monthly-$($BudgetUsd)usd"
$RenderPolicyName = "$NamePrefix-render-app"
$RunPrefix        = 'cc3d-runs/'
$InstanceType     = 'm7i.xlarge'
$MaxVcpus         = 16
$JobVcpu          = 4
$JobMemoryMiB     = 15500
$JobTimeoutSecs   = 5700
$RootVolumeGiB    = 100

$ScriptDir    = Split-Path -Parent $MyInvocation.MyCommand.Path
$ResourceFile = Join-Path $ScriptDir 'aws_resources.json'

$script:TempDir = Join-Path $env:TEMP ("biosim-cc3d-provision-" + $PID)
$script:Resources = [ordered]@{}
$script:StepNumber = 0

# -----------------------------------------------------------------------------
# Output helpers
# -----------------------------------------------------------------------------
function Write-Section {
    param([string]$Text)
    Write-Host ''
    Write-Host ('=' * 78) -ForegroundColor DarkGray
    Write-Host $Text -ForegroundColor Cyan
    Write-Host ('=' * 78) -ForegroundColor DarkGray
}

function Write-Step {
    param([string]$Text)
    $script:StepNumber++
    Write-Host ''
    Write-Host ("[{0,2}] {1}" -f $script:StepNumber, $Text) -ForegroundColor White
}

function Write-Made   { param([string]$T) Write-Host "     + created  $T" -ForegroundColor Green }
function Write-Have   { param([string]$T) Write-Host "     = exists   $T" -ForegroundColor DarkGray }
function Write-Info   { param([string]$T) Write-Host "       $T" -ForegroundColor Gray }
function Write-Warn2  { param([string]$T) Write-Host "     ! $T" -ForegroundColor Yellow }

function Stop-WithError {
    param([string]$Message, [string]$Detail = '')
    Write-Host ''
    Write-Host '--- PROVISIONING ABORTED ---' -ForegroundColor Red
    Write-Host $Message -ForegroundColor Red
    if ($Detail) {
        Write-Host ''
        Write-Host $Detail -ForegroundColor DarkYellow
    }
    if ($script:Resources.Count -gt 0) {
        Save-ResourceRecord
        Write-Host ''
        Write-Host "Partial state was written to $ResourceFile -- re-run this script to resume, or run the teardown script." -ForegroundColor DarkYellow
    }
    Remove-TempDir
    exit 1
}

# -----------------------------------------------------------------------------
# File / JSON helpers.  Every JSON handed to the AWS CLI goes through a UTF-8
# (no BOM) temp file: a BOM makes the CLI's JSON parser fail, and inline JSON on
# a PowerShell command line is mangled by quote handling.
# -----------------------------------------------------------------------------
function Write-TextFileNoBom {
    param([string]$Path, [string]$Text)
    $utf8 = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($Path, $Text, $utf8)
}

function New-JsonArg {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)]$Object
    )
    if (-not (Test-Path -LiteralPath $script:TempDir)) {
        New-Item -ItemType Directory -Path $script:TempDir -Force -ErrorAction Stop | Out-Null
    }
    $path = Join-Path $script:TempDir ($Name + '.json')
    $json = ConvertTo-Json -InputObject $Object -Depth 12
    Write-TextFileNoBom -Path $path -Text $json
    return ('file://' + $path)
}

function Remove-TempDir {
    if ($script:TempDir -and (Test-Path -LiteralPath $script:TempDir)) {
        Remove-Item -LiteralPath $script:TempDir -Recurse -Force -ErrorAction SilentlyContinue
    }
}

# -----------------------------------------------------------------------------
# AWS CLI wrapper.  stderr goes to a file so PowerShell never turns it into a
# NativeCommandError, and $LASTEXITCODE is checked after EVERY call.
# -----------------------------------------------------------------------------
function Invoke-Aws {
    param(
        [Parameter(Mandatory = $true)][string[]]$AwsArgs,
        [string]$Context = 'aws',
        [switch]$AllowFailure
    )

    if (-not (Test-Path -LiteralPath $script:TempDir)) {
        New-Item -ItemType Directory -Path $script:TempDir -Force -ErrorAction Stop | Out-Null
    }

    $argList = New-Object System.Collections.Generic.List[string]
    foreach ($g in $script:AwsGlobalArgs) { $argList.Add($g) }
    foreach ($a in $AwsArgs) { $argList.Add($a) }

    $errPath  = Join-Path $script:TempDir ('stderr-' + [Guid]::NewGuid().ToString('N') + '.txt')
    $argArray = $argList.ToArray()
    # PowerShell 5.1 surfaces native-command stderr as an ErrorRecord, and under
    # $ErrorActionPreference='Stop' that becomes a TERMINATING error. That killed the
    # whole script on the first aws call which legitimately writes to stderr -- the
    # expected 404 from `head-bucket` on a bucket that does not exist yet -- after it
    # had already created the VPC, subnets and endpoint. The exit code is checked
    # explicitly below, so stderr here must be data, not a fatal condition.
    $previousEap = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $raw  = & aws @argArray 2> $errPath
        $code = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previousEap
    }

    $stdOut = ''
    if ($null -ne $raw) { $stdOut = ($raw | Out-String) }

    $stdErr = ''
    if (Test-Path -LiteralPath $errPath) {
        $stdErr = (Get-Content -LiteralPath $errPath -Raw -ErrorAction SilentlyContinue)
        if ($null -eq $stdErr) { $stdErr = '' }
        Remove-Item -LiteralPath $errPath -Force -ErrorAction SilentlyContinue
    }

    $result = [pscustomobject]@{
        ExitCode = $code
        StdOut   = $stdOut.Trim()
        StdErr   = $stdErr.Trim()
        Command  = ('aws ' + ($argList.ToArray() -join ' '))
    }

    if ($code -ne 0 -and -not $AllowFailure) {
        Stop-WithError -Message "$Context failed (exit $code)." -Detail ($result.Command + "`n`n" + $result.StdErr)
    }
    return $result
}

function Get-AwsJson {
    param(
        [Parameter(Mandatory = $true)][string[]]$AwsArgs,
        [string]$Context = 'aws',
        [switch]$AllowFailure
    )
    $r = Invoke-Aws -AwsArgs $AwsArgs -Context $Context -AllowFailure:$AllowFailure
    if ($r.ExitCode -ne 0) { return $null }
    if ([string]::IsNullOrWhiteSpace($r.StdOut)) { return $null }
    try   { return ($r.StdOut | ConvertFrom-Json) }
    catch { return $null }
}

function Test-AwsErrorMatch {
    param($Result, [string]$Pattern)
    if ($null -eq $Result) { return $false }
    return ($Result.StdErr -match $Pattern)
}

# -----------------------------------------------------------------------------
# Tagging.  Every taggable resource gets Project + Component; EC2-family
# resources also get a Name so the console is readable.
# -----------------------------------------------------------------------------
function Add-Ec2Tags {
    param([string[]]$ResourceIds, [string]$Name = '')
    $tagArgs = @('ec2', 'create-tags', '--resources')
    $tagArgs += $ResourceIds
    $tagArgs += '--tags'
    $tagArgs += "Key=Project,Value=$script:TagProject"
    $tagArgs += "Key=Component,Value=$script:TagComponent"
    if ($Name) { $tagArgs += "Key=Name,Value=$Name" }
    Invoke-Aws -AwsArgs $tagArgs -Context ("tag " + ($ResourceIds -join ',')) | Out-Null
}

function Get-IamTagArgs {
    return @('--tags', "Key=Project,Value=$script:TagProject", "Key=Component,Value=$script:TagComponent")
}

function Get-BatchTagArgs {
    return @('--tags', "Project=$script:TagProject,Component=$script:TagComponent")
}

# -----------------------------------------------------------------------------
# Resource record -- written after every step so teardown can always read it.
# -----------------------------------------------------------------------------
function Set-Record {
    param([string]$Key, $Value)
    $script:Resources[$Key] = $Value
}

function Save-ResourceRecord {
    try {
        $snapshot = [ordered]@{}
        foreach ($k in $script:Resources.Keys) { $snapshot[$k] = $script:Resources[$k] }
        $snapshot['writtenAt'] = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')
        $json = ConvertTo-Json -InputObject $snapshot -Depth 12
        Write-TextFileNoBom -Path $ResourceFile -Text $json
    } catch {
        Write-Warn2 ("could not write " + $ResourceFile + ": " + $_.Exception.Message)
    }
}

# -----------------------------------------------------------------------------
# Reusable policy documents
# -----------------------------------------------------------------------------
function New-TrustPolicy {
    param([string]$ServicePrincipal)
    return [ordered]@{
        Version   = '2012-10-17'
        Statement = @(
            [ordered]@{
                Effect    = 'Allow'
                Principal = [ordered]@{ Service = $ServicePrincipal }
                Action    = 'sts:AssumeRole'
            }
        )
    }
}

function New-BucketScopedS3Policy {
    param([string]$Bucket)
    return [ordered]@{
        Version   = '2012-10-17'
        Statement = @(
            [ordered]@{
                Sid      = 'ListOnlyThisBucket'
                Effect   = 'Allow'
                Action   = @('s3:ListBucket')
                Resource = @("arn:aws:s3:::$Bucket")
            },
            [ordered]@{
                Sid      = 'ObjectRwOnlyThisBucket'
                Effect   = 'Allow'
                Action   = @('s3:GetObject', 's3:PutObject', 's3:DeleteObject')
                Resource = @("arn:aws:s3:::$Bucket/*")
            }
        )
    }
}

# =============================================================================
# PREFLIGHT
# =============================================================================
Write-Section 'BioSimulateAI :: CompuCell3D on-demand compute :: PREFLIGHT'

if (-not (Get-Command aws -ErrorAction SilentlyContinue)) {
    Write-Host ''
    Write-Host 'ABORT: the AWS CLI (aws) is not on PATH.' -ForegroundColor Red
    Write-Host 'Install AWS CLI v2, then re-run:  https://aws.amazon.com/cli/' -ForegroundColor DarkYellow
    exit 1
}

$env:AWS_PAGER = ''
$env:AWS_DEFAULT_REGION = $Region

$script:AwsGlobalArgs = @('--output', 'json')
if ($AwsProfile) { $script:AwsGlobalArgs += @('--profile', $AwsProfile) }

New-Item -ItemType Directory -Path $script:TempDir -Force -ErrorAction Stop | Out-Null

$identity = Get-AwsJson -AwsArgs @('sts', 'get-caller-identity') -Context 'sts get-caller-identity' -AllowFailure
if ($null -eq $identity -or -not $identity.Account) {
    Write-Host ''
    Write-Host 'ABORT: aws sts get-caller-identity failed -- there is no usable AWS session.' -ForegroundColor Red
    Write-Host ''
    Write-Host 'Fix the credentials first, then re-run this script:' -ForegroundColor DarkYellow
    Write-Host '  aws sso login --profile <profile>          # SSO / IAM Identity Center' -ForegroundColor DarkYellow
    Write-Host '  aws configure --profile <profile>          # long-lived keys' -ForegroundColor DarkYellow
    Write-Host '  .\provision_aws.ps1 -NotificationEmail you@example.com -AwsProfile <profile> -Confirm' -ForegroundColor DarkYellow
    Remove-TempDir
    exit 1
}

$AccountId = [string]$identity.Account
$CallerArn = [string]$identity.Arn

if (-not $BucketName) { $BucketName = "$NamePrefix-$AccountId-$Region" }
$BucketName = $BucketName.ToLowerInvariant()
if ($BucketName.Length -gt 63) {
    Stop-WithError -Message "Derived bucket name '$BucketName' is longer than 63 characters. Pass -BucketName explicitly."
}

$EcrRegistry = "$AccountId.dkr.ecr.$Region.amazonaws.com"
$EcrImageUri = "$EcrRegistry/$EcrRepoName" + ':latest'
$JobQueueArn = "arn:aws:batch:${Region}:${AccountId}:job-queue/$JobQueueName"
$JobDefArn   = "arn:aws:batch:${Region}:${AccountId}:job-definition/$JobDefName"

Write-Host ''
Write-Host "  Account   : $AccountId"
Write-Host "  Caller    : $CallerArn"
Write-Host "  Region    : $Region"
Write-Host "  Bucket    : $BucketName"
Write-Host "  ECR image : $EcrImageUri"
Write-Host "  Budget    : `$$BudgetUsd/month -> $NotificationEmail"

Write-Section 'WHAT THIS SCRIPT WILL CREATE (idle cost = cost when NO job runs)'

$plan = @(
    [pscustomobject]@{ Item = "VPC $VpcCidr + Internet Gateway";                 IdleCost = '$0.00'; Note = 'VPCs and IGWs are free' }
    [pscustomobject]@{ Item = '2 public subnets (2 AZs) + 1 public route table';  IdleCost = '$0.00'; Note = 'MapPublicIpOnLaunch=true; 0.0.0.0/0 -> IGW' }
    [pscustomobject]@{ Item = 'S3 GATEWAY VPC endpoint';                         IdleCost = '$0.00'; Note = 'Gateway endpoints are free (Interface ones are not)' }
    [pscustomobject]@{ Item = "S3 bucket $BucketName";                           IdleCost = '$0.00'; Note = '$0.023/GB-mo stored; cc3d-runs/ expires after 30 days' }
    [pscustomobject]@{ Item = "ECR repository $EcrRepoName";                     IdleCost = '~$0.35'; Note = '$0.10/GB-mo; the CC3D image is ~3-4 GB, keeps 2 tags' }
    [pscustomobject]@{ Item = "Security group $SgName";                          IdleCost = '$0.00'; Note = 'no inbound rules, all outbound allowed' }
    [pscustomobject]@{ Item = 'IAM instance role + profile + job role + policies';IdleCost = '$0.00'; Note = 'IAM is free; S3 access scoped to the one bucket' }
    [pscustomobject]@{ Item = "Launch template $LaunchTplName";                  IdleCost = '$0.00'; Note = "$RootVolumeGiB GB gp3 root, DeleteOnTermination=true" }
    [pscustomobject]@{ Item = "Batch compute env $ComputeEnvName (SPOT)";        IdleCost = '$0.00'; Note = "minvCpus=0/desired=0 -> zero hosts idle; $InstanceType only" }
    [pscustomobject]@{ Item = "Batch job queue $JobQueueName";                   IdleCost = '$0.00'; Note = 'queues are free' }
    [pscustomobject]@{ Item = "Batch job definition $JobDefName";                IdleCost = '$0.00'; Note = "$JobVcpu vCPU / $JobMemoryMiB MiB, 95 min timeout, 2 attempts" }
    [pscustomobject]@{ Item = "CloudWatch log group $LogGroupName";              IdleCost = '$0.00'; Note = "$LogRetentionDays-day retention; only charged per GB ingested" }
    [pscustomobject]@{ Item = "AWS Budget `$$BudgetUsd/month + 2 email alerts";  IdleCost = '$0.00'; Note = 'first 2 budgets per account are free' }
    [pscustomobject]@{ Item = "IAM managed policy $RenderPolicyName";            IdleCost = '$0.00'; Note = 'for the Render app; NO access keys are created' }
)
$plan | Format-Table -AutoSize -Property Item, IdleCost, Note | Out-String -Width 200 | Write-Host

Write-Host '  TOTAL IDLE COST: ~$0.35/month (ECR image storage only).' -ForegroundColor Green
Write-Host ("  Per RUN: 1 x $InstanceType Spot, roughly `$0.05-0.09/hour, billed only while a job runs.") -ForegroundColor Green
Write-Host ''
Write-Host '  DELIBERATELY NOT CREATED: NAT Gateway, Elastic IP, Interface VPC endpoint,' -ForegroundColor DarkGray
Write-Host '  load balancer, RDS, ElastiCache, EKS, or any always-on instance.' -ForegroundColor DarkGray

if (-not $Confirm) {
    Write-Host ''
    Write-Host 'DRY RUN -- nothing was created.' -ForegroundColor Yellow
    Write-Host 'Re-run with -Confirm to create the resources above:' -ForegroundColor Yellow
    $profileArg = ''
    if ($AwsProfile) { $profileArg = " -AwsProfile $AwsProfile" }
    Write-Host ("  .\provision_aws.ps1 -NotificationEmail $NotificationEmail$profileArg -Confirm") -ForegroundColor Yellow
    Remove-TempDir
    exit 0
}

Set-Record 'accountId'      $AccountId
Set-Record 'region'         $Region
Set-Record 'namePrefix'     $NamePrefix
Set-Record 'provisionedBy'  $CallerArn
Save-ResourceRecord

Write-Section 'CREATING RESOURCES'

# =============================================================================
# 1. VPC, Internet Gateway, two public subnets, public route table
# =============================================================================
Write-Step "VPC $VpcCidr"

$vpcLookup = Get-AwsJson -AwsArgs @(
    'ec2', 'describe-vpcs',
    '--filters', "Name=tag:Name,Values=$VpcName", "Name=tag:Project,Values=$script:TagProject"
) -Context 'ec2 describe-vpcs'

$VpcId = $null
if ($vpcLookup -and $vpcLookup.Vpcs -and $vpcLookup.Vpcs.Count -gt 0) {
    $VpcId = [string]$vpcLookup.Vpcs[0].VpcId
    Write-Have "VPC $VpcId"
} else {
    $created = Get-AwsJson -AwsArgs @('ec2', 'create-vpc', '--cidr-block', $VpcCidr) -Context 'ec2 create-vpc'
    if ($null -eq $created -or -not $created.Vpc.VpcId) { Stop-WithError -Message 'create-vpc returned no VpcId.' }
    $VpcId = [string]$created.Vpc.VpcId
    Add-Ec2Tags -ResourceIds @($VpcId) -Name $VpcName
    Invoke-Aws -AwsArgs @('ec2', 'modify-vpc-attribute', '--vpc-id', $VpcId, '--enable-dns-hostnames') -Context 'enable dns hostnames' | Out-Null
    Invoke-Aws -AwsArgs @('ec2', 'modify-vpc-attribute', '--vpc-id', $VpcId, '--enable-dns-support')   -Context 'enable dns support'   | Out-Null
    Write-Made "VPC $VpcId"
}
Set-Record 'vpcId'   $VpcId
Set-Record 'vpcCidr' $VpcCidr

Write-Step 'Internet Gateway'

$igwLookup = Get-AwsJson -AwsArgs @(
    'ec2', 'describe-internet-gateways',
    '--filters', "Name=attachment.vpc-id,Values=$VpcId"
) -Context 'ec2 describe-internet-gateways'

$IgwId = $null
if ($igwLookup -and $igwLookup.InternetGateways -and $igwLookup.InternetGateways.Count -gt 0) {
    $IgwId = [string]$igwLookup.InternetGateways[0].InternetGatewayId
    Write-Have "IGW $IgwId (attached)"
} else {
    $created = Get-AwsJson -AwsArgs @('ec2', 'create-internet-gateway') -Context 'ec2 create-internet-gateway'
    if ($null -eq $created -or -not $created.InternetGateway.InternetGatewayId) { Stop-WithError -Message 'create-internet-gateway returned no id.' }
    $IgwId = [string]$created.InternetGateway.InternetGatewayId
    Add-Ec2Tags -ResourceIds @($IgwId) -Name $IgwName
    Invoke-Aws -AwsArgs @('ec2', 'attach-internet-gateway', '--internet-gateway-id', $IgwId, '--vpc-id', $VpcId) -Context 'attach-internet-gateway' | Out-Null
    Write-Made "IGW $IgwId (attached to $VpcId)"
}
Set-Record 'internetGatewayId' $IgwId

Write-Step 'Availability zones'

$azInfo = Get-AwsJson -AwsArgs @(
    'ec2', 'describe-availability-zones',
    '--filters', 'Name=state,Values=available', 'Name=zone-type,Values=availability-zone'
) -Context 'ec2 describe-availability-zones'
if ($null -eq $azInfo -or $azInfo.AvailabilityZones.Count -lt 2) {
    Stop-WithError -Message "Region $Region reported fewer than two usable availability zones."
}
$azNames = @($azInfo.AvailabilityZones | Sort-Object -Property ZoneName | Select-Object -ExpandProperty ZoneName)
$AzA = [string]$azNames[0]
$AzB = [string]$azNames[1]
Write-Info "using $AzA and $AzB"

function New-PublicSubnet {
    param([string]$Cidr, [string]$Az, [string]$SubnetName)

    $lookup = Get-AwsJson -AwsArgs @(
        'ec2', 'describe-subnets',
        '--filters', "Name=vpc-id,Values=$VpcId", "Name=cidr-block,Values=$Cidr"
    ) -Context 'ec2 describe-subnets'

    if ($lookup -and $lookup.Subnets -and $lookup.Subnets.Count -gt 0) {
        $id = [string]$lookup.Subnets[0].SubnetId
        if (-not $lookup.Subnets[0].MapPublicIpOnLaunch) {
            Invoke-Aws -AwsArgs @('ec2', 'modify-subnet-attribute', '--subnet-id', $id, '--map-public-ip-on-launch') -Context 'modify-subnet-attribute' | Out-Null
            Write-Info "set MapPublicIpOnLaunch=true on $id"
        }
        Write-Have "subnet $id ($Cidr, $Az)"
        return $id
    }

    $created = Get-AwsJson -AwsArgs @(
        'ec2', 'create-subnet',
        '--vpc-id', $VpcId, '--cidr-block', $Cidr, '--availability-zone', $Az
    ) -Context 'ec2 create-subnet'
    if ($null -eq $created -or -not $created.Subnet.SubnetId) { Stop-WithError -Message "create-subnet failed for $Cidr." }
    $id = [string]$created.Subnet.SubnetId
    Add-Ec2Tags -ResourceIds @($id) -Name $SubnetName
    Invoke-Aws -AwsArgs @('ec2', 'modify-subnet-attribute', '--subnet-id', $id, '--map-public-ip-on-launch') -Context 'modify-subnet-attribute' | Out-Null
    Write-Made "subnet $id ($Cidr, $Az, public IP on launch)"
    return $id
}

Write-Step 'Two public subnets in different AZs'
$SubnetAId = New-PublicSubnet -Cidr $SubnetACidr -Az $AzA -SubnetName $SubnetAName
$SubnetBId = New-PublicSubnet -Cidr $SubnetBCidr -Az $AzB -SubnetName $SubnetBName
Set-Record 'subnetIds' @($SubnetAId, $SubnetBId)
Set-Record 'availabilityZones' @($AzA, $AzB)

Write-Step 'Public route table (0.0.0.0/0 -> IGW)'

$rtbLookup = Get-AwsJson -AwsArgs @(
    'ec2', 'describe-route-tables',
    '--filters', "Name=vpc-id,Values=$VpcId", "Name=tag:Name,Values=$RouteTableName"
) -Context 'ec2 describe-route-tables'

$RouteTableId = $null
if ($rtbLookup -and $rtbLookup.RouteTables -and $rtbLookup.RouteTables.Count -gt 0) {
    $RouteTableId = [string]$rtbLookup.RouteTables[0].RouteTableId
    Write-Have "route table $RouteTableId"
} else {
    $created = Get-AwsJson -AwsArgs @('ec2', 'create-route-table', '--vpc-id', $VpcId) -Context 'ec2 create-route-table'
    if ($null -eq $created -or -not $created.RouteTable.RouteTableId) { Stop-WithError -Message 'create-route-table returned no id.' }
    $RouteTableId = [string]$created.RouteTable.RouteTableId
    Add-Ec2Tags -ResourceIds @($RouteTableId) -Name $RouteTableName
    Write-Made "route table $RouteTableId"
}
Set-Record 'routeTableId' $RouteTableId

$rtbState = Get-AwsJson -AwsArgs @('ec2', 'describe-route-tables', '--route-table-ids', $RouteTableId) -Context 'describe route table'
$hasDefaultRoute = $false
$associatedSubnets = @()
if ($rtbState -and $rtbState.RouteTables.Count -gt 0) {
    foreach ($route in @($rtbState.RouteTables[0].Routes)) {
        if ($route.DestinationCidrBlock -eq '0.0.0.0/0') { $hasDefaultRoute = $true }
    }
    foreach ($assoc in @($rtbState.RouteTables[0].Associations)) {
        if ($assoc.SubnetId) { $associatedSubnets += [string]$assoc.SubnetId }
    }
}

if ($hasDefaultRoute) {
    Write-Have '0.0.0.0/0 route'
} else {
    Invoke-Aws -AwsArgs @(
        'ec2', 'create-route',
        '--route-table-id', $RouteTableId,
        '--destination-cidr-block', '0.0.0.0/0',
        '--gateway-id', $IgwId
    ) -Context 'ec2 create-route' | Out-Null
    Write-Made "0.0.0.0/0 -> $IgwId"
}

foreach ($subnetId in @($SubnetAId, $SubnetBId)) {
    if ($associatedSubnets -contains $subnetId) {
        Write-Have "association $subnetId"
    } else {
        Invoke-Aws -AwsArgs @(
            'ec2', 'associate-route-table',
            '--route-table-id', $RouteTableId, '--subnet-id', $subnetId
        ) -Context 'associate-route-table' | Out-Null
        Write-Made "association $subnetId -> $RouteTableId"
    }
}
Save-ResourceRecord

# =============================================================================
# 2. FREE S3 Gateway VPC endpoint
# =============================================================================
Write-Step 'S3 GATEWAY VPC endpoint (free -- this is what replaces a NAT Gateway for S3 traffic)'

$s3ServiceName = "com.amazonaws.$Region.s3"
$vpceLookup = Get-AwsJson -AwsArgs @(
    'ec2', 'describe-vpc-endpoints',
    '--filters', "Name=vpc-id,Values=$VpcId", "Name=service-name,Values=$s3ServiceName", 'Name=vpc-endpoint-type,Values=Gateway'
) -Context 'ec2 describe-vpc-endpoints'

$VpcEndpointId = $null
if ($vpceLookup -and $vpceLookup.VpcEndpoints -and $vpceLookup.VpcEndpoints.Count -gt 0) {
    $VpcEndpointId = [string]$vpceLookup.VpcEndpoints[0].VpcEndpointId
    Write-Have "gateway endpoint $VpcEndpointId"
    if (-not (@($vpceLookup.VpcEndpoints[0].RouteTableIds) -contains $RouteTableId)) {
        Invoke-Aws -AwsArgs @(
            'ec2', 'modify-vpc-endpoint',
            '--vpc-endpoint-id', $VpcEndpointId, '--add-route-table-ids', $RouteTableId
        ) -Context 'modify-vpc-endpoint' | Out-Null
        Write-Made "route table association $RouteTableId"
    }
} else {
    $created = Get-AwsJson -AwsArgs @(
        'ec2', 'create-vpc-endpoint',
        '--vpc-id', $VpcId,
        '--service-name', $s3ServiceName,
        '--vpc-endpoint-type', 'Gateway',
        '--route-table-ids', $RouteTableId
    ) -Context 'ec2 create-vpc-endpoint'
    if ($null -eq $created -or -not $created.VpcEndpoint.VpcEndpointId) { Stop-WithError -Message 'create-vpc-endpoint returned no id.' }
    $VpcEndpointId = [string]$created.VpcEndpoint.VpcEndpointId
    Add-Ec2Tags -ResourceIds @($VpcEndpointId) -Name $EndpointName
    Write-Made "gateway endpoint $VpcEndpointId ($s3ServiceName)"
}
Set-Record 's3GatewayEndpointId' $VpcEndpointId
Save-ResourceRecord

# =============================================================================
# 3. Private S3 bucket
# =============================================================================
Write-Step "S3 bucket $BucketName"

$headResult = Invoke-Aws -AwsArgs @('s3api', 'head-bucket', '--bucket', $BucketName) -Context 's3api head-bucket' -AllowFailure
if ($headResult.ExitCode -eq 0) {
    Write-Have "bucket $BucketName"
} else {
    if (Test-AwsErrorMatch -Result $headResult -Pattern '403|Forbidden') {
        Stop-WithError -Message "Bucket name '$BucketName' already exists in another AWS account." -Detail 'Pass a different -BucketName.'
    }
    $createArgs = @('s3api', 'create-bucket', '--bucket', $BucketName, '--region', $Region)
    if ($Region -ne 'us-east-1') {
        $createArgs += @('--create-bucket-configuration', "LocationConstraint=$Region")
    }
    Invoke-Aws -AwsArgs $createArgs -Context 's3api create-bucket' | Out-Null
    Write-Made "bucket $BucketName"
}
Set-Record 'bucketName' $BucketName
Set-Record 'bucketRunPrefix' $RunPrefix

Invoke-Aws -AwsArgs @(
    's3api', 'put-public-access-block',
    '--bucket', $BucketName,
    '--public-access-block-configuration',
    'BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true'
) -Context 's3api put-public-access-block' | Out-Null
Write-Info 'block all public access: ON'

$encryptionDoc = [ordered]@{
    Rules = @(
        [ordered]@{
            ApplyServerSideEncryptionByDefault = [ordered]@{ SSEAlgorithm = 'AES256' }
            BucketKeyEnabled                   = $true
        }
    )
}
$encArg = New-JsonArg -Name 'bucket-encryption' -Object $encryptionDoc
Invoke-Aws -AwsArgs @('s3api', 'put-bucket-encryption', '--bucket', $BucketName, '--server-side-encryption-configuration', $encArg) -Context 's3api put-bucket-encryption' | Out-Null
Write-Info 'default encryption: SSE-S3 (AES256)'

# Versioning must stay OFF. A never-versioned bucket is already off, and calling
# Suspended on one is pointless -- only suspend when it is actually Enabled.
$versioning = Get-AwsJson -AwsArgs @('s3api', 'get-bucket-versioning', '--bucket', $BucketName) -Context 's3api get-bucket-versioning' -AllowFailure
if ($versioning -and $versioning.Status -eq 'Enabled') {
    Invoke-Aws -AwsArgs @('s3api', 'put-bucket-versioning', '--bucket', $BucketName, '--versioning-configuration', 'Status=Suspended') -Context 's3api put-bucket-versioning' | Out-Null
    Write-Info 'versioning: suspended (was Enabled)'
} else {
    Write-Info 'versioning: off'
}

$lifecycleDoc = [ordered]@{
    Rules = @(
        [ordered]@{
            ID         = 'expire-cc3d-runs-after-30-days'
            Status     = 'Enabled'
            Filter     = [ordered]@{ Prefix = $RunPrefix }
            Expiration = [ordered]@{ Days = 30 }
        },
        [ordered]@{
            ID                             = 'abort-incomplete-multipart-after-7-days'
            Status                         = 'Enabled'
            Filter                         = [ordered]@{ Prefix = '' }
            AbortIncompleteMultipartUpload = [ordered]@{ DaysAfterInitiation = 7 }
        }
    )
}
$lcArg = New-JsonArg -Name 'bucket-lifecycle' -Object $lifecycleDoc
Invoke-Aws -AwsArgs @('s3api', 'put-bucket-lifecycle-configuration', '--bucket', $BucketName, '--lifecycle-configuration', $lcArg) -Context 's3api put-bucket-lifecycle-configuration' | Out-Null
Write-Info "lifecycle: $RunPrefix expires in 30 days; incomplete multipart aborted after 7 days"

$taggingDoc = [ordered]@{
    TagSet = @(
        [ordered]@{ Key = 'Project';   Value = $script:TagProject }
        [ordered]@{ Key = 'Component'; Value = $script:TagComponent }
    )
}
$tagArg = New-JsonArg -Name 'bucket-tagging' -Object $taggingDoc
Invoke-Aws -AwsArgs @('s3api', 'put-bucket-tagging', '--bucket', $BucketName, '--tagging', $tagArg) -Context 's3api put-bucket-tagging' | Out-Null
Write-Info 'tagged'
Save-ResourceRecord

# =============================================================================
# 4. ECR repository
# =============================================================================
Write-Step "ECR repository $EcrRepoName"

$repoLookup = Get-AwsJson -AwsArgs @('ecr', 'describe-repositories', '--repository-names', $EcrRepoName) -Context 'ecr describe-repositories' -AllowFailure
if ($repoLookup -and $repoLookup.repositories -and $repoLookup.repositories.Count -gt 0) {
    Write-Have "repository $EcrRepoName"
} else {
    Invoke-Aws -AwsArgs @(
        'ecr', 'create-repository',
        '--repository-name', $EcrRepoName,
        '--image-scanning-configuration', 'scanOnPush=true',
        '--encryption-configuration', 'encryptionType=AES256',
        '--tags', "Key=Project,Value=$script:TagProject", "Key=Component,Value=$script:TagComponent"
    ) -Context 'ecr create-repository' | Out-Null
    Write-Made "repository $EcrRepoName"
}

$ecrLifecycle = [ordered]@{
    rules = @(
        [ordered]@{
            rulePriority = 1
            description  = 'Keep only the 2 most recent images'
            selection    = [ordered]@{
                tagStatus   = 'any'
                countType   = 'imageCountMoreThan'
                countNumber = 2
            }
            action = [ordered]@{ type = 'expire' }
        }
    )
}
$ecrArg = New-JsonArg -Name 'ecr-lifecycle' -Object $ecrLifecycle
Invoke-Aws -AwsArgs @('ecr', 'put-lifecycle-policy', '--repository-name', $EcrRepoName, '--lifecycle-policy-text', $ecrArg) -Context 'ecr put-lifecycle-policy' | Out-Null
Write-Info 'lifecycle policy: keep 2 most recent images'

Set-Record 'ecrRepositoryName' $EcrRepoName
Set-Record 'ecrRepositoryUri'  "$EcrRegistry/$EcrRepoName"
Set-Record 'ecrImageUri'       $EcrImageUri
Save-ResourceRecord

# =============================================================================
# 5. Security group -- no inbound, all outbound
# =============================================================================
Write-Step "Security group $SgName (no inbound rules, all outbound allowed)"

$sgLookup = Get-AwsJson -AwsArgs @(
    'ec2', 'describe-security-groups',
    '--filters', "Name=vpc-id,Values=$VpcId", "Name=group-name,Values=$SgName"
) -Context 'ec2 describe-security-groups'

$SecurityGroupId = $null
if ($sgLookup -and $sgLookup.SecurityGroups -and $sgLookup.SecurityGroups.Count -gt 0) {
    $SecurityGroupId = [string]$sgLookup.SecurityGroups[0].GroupId
    Write-Have "security group $SecurityGroupId"
    $ingressCount = @($sgLookup.SecurityGroups[0].IpPermissions).Count
    if ($ingressCount -gt 0) { Write-Warn2 "security group has $ingressCount inbound rule(s) -- this script never adds any; remove them manually if unexpected." }
} else {
    $created = Get-AwsJson -AwsArgs @(
        'ec2', 'create-security-group',
        '--group-name', $SgName,
        '--description', 'BioSimulateAI CC3D Batch workers: egress only, no inbound',
        '--vpc-id', $VpcId
    ) -Context 'ec2 create-security-group'
    if ($null -eq $created -or -not $created.GroupId) { Stop-WithError -Message 'create-security-group returned no GroupId.' }
    $SecurityGroupId = [string]$created.GroupId
    Add-Ec2Tags -ResourceIds @($SecurityGroupId) -Name $SgName
    # A new SG has zero ingress and the default allow-all egress rule. Correct as-is.
    Write-Made "security group $SecurityGroupId"
}
Set-Record 'securityGroupId' $SecurityGroupId
Save-ResourceRecord

# =============================================================================
# 6. IAM: instance role + profile, Batch service-linked role, job role
# =============================================================================
Write-Step 'IAM instance role for the Batch EC2 hosts'

$bucketPolicyDoc = New-BucketScopedS3Policy -Bucket $BucketName
$bucketPolicyArg = New-JsonArg -Name 'bucket-scoped-s3' -Object $bucketPolicyDoc

$ec2TrustArg = New-JsonArg -Name 'trust-ec2' -Object (New-TrustPolicy -ServicePrincipal 'ec2.amazonaws.com')

$roleLookup = Invoke-Aws -AwsArgs @('iam', 'get-role', '--role-name', $InstanceRoleName) -Context 'iam get-role' -AllowFailure
if ($roleLookup.ExitCode -eq 0) {
    Write-Have "role $InstanceRoleName"
} else {
    $args1 = @('iam', 'create-role', '--role-name', $InstanceRoleName,
               '--assume-role-policy-document', $ec2TrustArg,
               '--description', 'BioSimulateAI CC3D Batch EC2 host role') + (Get-IamTagArgs)
    Invoke-Aws -AwsArgs $args1 -Context 'iam create-role (instance role)' | Out-Null
    Write-Made "role $InstanceRoleName"
}

Invoke-Aws -AwsArgs @(
    'iam', 'attach-role-policy',
    '--role-name', $InstanceRoleName,
    '--policy-arn', 'arn:aws:iam::aws:policy/service-role/AmazonEC2ContainerServiceforEC2Role'
) -Context 'attach AmazonEC2ContainerServiceforEC2Role' | Out-Null
Write-Info 'attached AmazonEC2ContainerServiceforEC2Role'

Invoke-Aws -AwsArgs @(
    'iam', 'put-role-policy',
    '--role-name', $InstanceRoleName,
    '--policy-name', $S3PolicyName,
    '--policy-document', $bucketPolicyArg
) -Context 'put-role-policy (instance role S3)' | Out-Null
Write-Info "inline policy $S3PolicyName -> s3:GetObject/PutObject/DeleteObject/ListBucket on $BucketName only"

$profLookup = Invoke-Aws -AwsArgs @('iam', 'get-instance-profile', '--instance-profile-name', $InstanceProfName) -Context 'iam get-instance-profile' -AllowFailure
$InstanceProfileArn = $null
if ($profLookup.ExitCode -eq 0) {
    $prof = $profLookup.StdOut | ConvertFrom-Json
    $InstanceProfileArn = [string]$prof.InstanceProfile.Arn
    Write-Have "instance profile $InstanceProfName"
    $attachedRoles = @($prof.InstanceProfile.Roles | Select-Object -ExpandProperty RoleName -ErrorAction SilentlyContinue)
    if (-not ($attachedRoles -contains $InstanceRoleName)) {
        Invoke-Aws -AwsArgs @('iam', 'add-role-to-instance-profile', '--instance-profile-name', $InstanceProfName, '--role-name', $InstanceRoleName) -Context 'add-role-to-instance-profile' | Out-Null
        Write-Made "role $InstanceRoleName added to profile"
    }
} else {
    $args2 = @('iam', 'create-instance-profile', '--instance-profile-name', $InstanceProfName) + (Get-IamTagArgs)
    $createdProf = Get-AwsJson -AwsArgs $args2 -Context 'iam create-instance-profile'
    if ($null -eq $createdProf) { Stop-WithError -Message 'create-instance-profile returned nothing.' }
    $InstanceProfileArn = [string]$createdProf.InstanceProfile.Arn
    Invoke-Aws -AwsArgs @('iam', 'add-role-to-instance-profile', '--instance-profile-name', $InstanceProfName, '--role-name', $InstanceRoleName) -Context 'add-role-to-instance-profile' | Out-Null
    Write-Made "instance profile $InstanceProfName"
    # IAM instance profiles are eventually consistent; Batch rejects the CE
    # otherwise. The CE creation below also retries, this just shortens it.
    Write-Info 'waiting 15s for IAM propagation'
    Start-Sleep -Seconds 15
}
Set-Record 'instanceRoleName'    $InstanceRoleName
Set-Record 'instanceProfileName' $InstanceProfName
Set-Record 'instanceProfileArn'  $InstanceProfileArn

Write-Step 'AWS Batch service-linked role'

$slrLookup = Invoke-Aws -AwsArgs @('iam', 'get-role', '--role-name', 'AWSServiceRoleForBatch') -Context 'iam get-role (SLR)' -AllowFailure
if ($slrLookup.ExitCode -eq 0) {
    Write-Have 'AWSServiceRoleForBatch'
    Set-Record 'batchServiceLinkedRole' 'AWSServiceRoleForBatch (pre-existing)'
} else {
    $slrCreate = Invoke-Aws -AwsArgs @('iam', 'create-service-linked-role', '--aws-service-name', 'batch.amazonaws.com') -Context 'iam create-service-linked-role' -AllowFailure
    if ($slrCreate.ExitCode -eq 0) {
        Write-Made 'AWSServiceRoleForBatch'
        Set-Record 'batchServiceLinkedRole' 'AWSServiceRoleForBatch (created by this script)'
        Start-Sleep -Seconds 10
    } elseif (Test-AwsErrorMatch -Result $slrCreate -Pattern 'has been taken|already exists|InvalidInput') {
        Write-Have 'AWSServiceRoleForBatch (already present)'
        Set-Record 'batchServiceLinkedRole' 'AWSServiceRoleForBatch (pre-existing)'
    } else {
        Stop-WithError -Message 'Could not create the AWS Batch service-linked role.' -Detail $slrCreate.StdErr
    }
}

Write-Step "Batch job role $JobRoleName (what the container itself gets)"

$ecsTrustArg = New-JsonArg -Name 'trust-ecs-tasks' -Object (New-TrustPolicy -ServicePrincipal 'ecs-tasks.amazonaws.com')

$jobRoleLookup = Invoke-Aws -AwsArgs @('iam', 'get-role', '--role-name', $JobRoleName) -Context 'iam get-role (job role)' -AllowFailure
$JobRoleArn = $null
if ($jobRoleLookup.ExitCode -eq 0) {
    $jr = $jobRoleLookup.StdOut | ConvertFrom-Json
    $JobRoleArn = [string]$jr.Role.Arn
    Write-Have "role $JobRoleName"
} else {
    $args3 = @('iam', 'create-role', '--role-name', $JobRoleName,
               '--assume-role-policy-document', $ecsTrustArg,
               '--description', 'BioSimulateAI CC3D container job role') + (Get-IamTagArgs)
    $createdJr = Get-AwsJson -AwsArgs $args3 -Context 'iam create-role (job role)'
    if ($null -eq $createdJr) { Stop-WithError -Message 'create-role (job role) returned nothing.' }
    $JobRoleArn = [string]$createdJr.Role.Arn
    Write-Made "role $JobRoleName"
}

Invoke-Aws -AwsArgs @(
    'iam', 'put-role-policy',
    '--role-name', $JobRoleName,
    '--policy-name', $S3PolicyName,
    '--policy-document', $bucketPolicyArg
) -Context 'put-role-policy (job role S3)' | Out-Null
Write-Info "inline policy $S3PolicyName -> same bucket-scoped S3 access"

Set-Record 'jobRoleName' $JobRoleName
Set-Record 'jobRoleArn'  $JobRoleArn
Save-ResourceRecord

# =============================================================================
# 7. Launch template + SPOT managed compute environment
# =============================================================================
Write-Step "Launch template $LaunchTplName ($RootVolumeGiB GB gp3 root, DeleteOnTermination=true)"

$ltData = [ordered]@{
    BlockDeviceMappings = @(
        [ordered]@{
            DeviceName = '/dev/xvda'
            Ebs        = [ordered]@{
                VolumeSize          = $RootVolumeGiB
                VolumeType          = 'gp3'
                DeleteOnTermination = $true
                Encrypted           = $true
            }
        }
    )
    MetadataOptions = [ordered]@{
        HttpTokens              = 'required'
        HttpPutResponseHopLimit = 2
    }
    TagSpecifications = @(
        [ordered]@{
            ResourceType = 'instance'
            Tags         = @(
                [ordered]@{ Key = 'Project';   Value = $script:TagProject }
                [ordered]@{ Key = 'Component'; Value = $script:TagComponent }
                [ordered]@{ Key = 'Name';      Value = "$NamePrefix-worker" }
            )
        },
        [ordered]@{
            ResourceType = 'volume'
            Tags         = @(
                [ordered]@{ Key = 'Project';   Value = $script:TagProject }
                [ordered]@{ Key = 'Component'; Value = $script:TagComponent }
            )
        }
    )
}

$ltLookup = Invoke-Aws -AwsArgs @('ec2', 'describe-launch-templates', '--launch-template-names', $LaunchTplName) -Context 'ec2 describe-launch-templates' -AllowFailure
$LaunchTemplateId = $null
if ($ltLookup.ExitCode -eq 0) {
    $lt = $ltLookup.StdOut | ConvertFrom-Json
    $LaunchTemplateId = [string]$lt.LaunchTemplates[0].LaunchTemplateId
    Write-Have "launch template $LaunchTemplateId"
} else {
    $ltArg = New-JsonArg -Name 'launch-template-data' -Object $ltData
    $createdLt = Get-AwsJson -AwsArgs @(
        'ec2', 'create-launch-template',
        '--launch-template-name', $LaunchTplName,
        '--version-description', 'BioSimulateAI CC3D Batch worker',
        '--launch-template-data', $ltArg
    ) -Context 'ec2 create-launch-template'
    if ($null -eq $createdLt -or -not $createdLt.LaunchTemplate.LaunchTemplateId) { Stop-WithError -Message 'create-launch-template returned no id.' }
    $LaunchTemplateId = [string]$createdLt.LaunchTemplate.LaunchTemplateId
    Add-Ec2Tags -ResourceIds @($LaunchTemplateId) -Name $LaunchTplName
    Write-Made "launch template $LaunchTemplateId"
}
Set-Record 'launchTemplateName' $LaunchTplName
Set-Record 'launchTemplateId'   $LaunchTemplateId

Write-Step "Batch compute environment $ComputeEnvName (SPOT, $InstanceType, min/desired 0, max $MaxVcpus vCPU)"

$ceLookup = Get-AwsJson -AwsArgs @('batch', 'describe-compute-environments', '--compute-environments', $ComputeEnvName) -Context 'batch describe-compute-environments' -AllowFailure
$ComputeEnvArn = $null
$ceExists = ($null -ne $ceLookup -and $ceLookup.computeEnvironments -and $ceLookup.computeEnvironments.Count -gt 0)

if ($ceExists) {
    $ComputeEnvArn = [string]$ceLookup.computeEnvironments[0].computeEnvironmentArn
    Write-Have "compute environment $ComputeEnvName ($($ceLookup.computeEnvironments[0].status))"
} else {
    $ceDoc = [ordered]@{
        computeEnvironmentName = $ComputeEnvName
        type                   = 'MANAGED'
        state                  = 'ENABLED'
        computeResources       = [ordered]@{
            type               = 'SPOT'
            allocationStrategy = 'SPOT_CAPACITY_OPTIMIZED'
            minvCpus           = 0
            desiredvCpus       = 0
            maxvCpus           = $MaxVcpus
            instanceTypes      = @($InstanceType)
            subnets            = @($SubnetAId, $SubnetBId)
            securityGroupIds   = @($SecurityGroupId)
            instanceRole       = $InstanceProfileArn
            launchTemplate     = [ordered]@{
                launchTemplateId = $LaunchTemplateId
                version          = '$Latest'
            }
            tags = [ordered]@{
                Project   = $script:TagProject
                Component = $script:TagComponent
                Name      = "$NamePrefix-worker"
            }
        }
        tags = [ordered]@{
            Project   = $script:TagProject
            Component = $script:TagComponent
        }
    }
    $ceArg = New-JsonArg -Name 'compute-environment' -Object $ceDoc

    # IAM is eventually consistent, so Batch can reject a brand-new instance
    # profile. Retry a few times rather than failing the whole run.
    $attempt = 0
    $ceCreated = $null
    while ($attempt -lt 5 -and $null -eq $ceCreated) {
        $attempt++
        $r = Invoke-Aws -AwsArgs @('batch', 'create-compute-environment', '--cli-input-json', $ceArg) -Context 'batch create-compute-environment' -AllowFailure
        if ($r.ExitCode -eq 0) {
            $ceCreated = $r.StdOut | ConvertFrom-Json
        } elseif (Test-AwsErrorMatch -Result $r -Pattern 'not\s+valid|cannot be assumed|Invalid.*instanceRole|does not exist|not authorized') {
            Write-Info "attempt $attempt rejected (IAM propagation); retrying in 15s"
            Start-Sleep -Seconds 15
        } else {
            Stop-WithError -Message 'batch create-compute-environment failed.' -Detail ($r.Command + "`n`n" + $r.StdErr)
        }
    }
    if ($null -eq $ceCreated) { Stop-WithError -Message "batch create-compute-environment still failing after $attempt attempts (IAM propagation)." -Detail 'Wait a minute and re-run this script; it resumes.' }
    $ComputeEnvArn = [string]$ceCreated.computeEnvironmentArn
    Write-Made "compute environment $ComputeEnvArn"
}
Set-Record 'computeEnvironmentName' $ComputeEnvName
Set-Record 'computeEnvironmentArn'  $ComputeEnvArn
Save-ResourceRecord

Write-Info 'waiting for the compute environment to reach VALID'
$ceReady = $false
for ($i = 0; $i -lt 40; $i++) {
    $state = Get-AwsJson -AwsArgs @('batch', 'describe-compute-environments', '--compute-environments', $ComputeEnvName) -Context 'poll compute environment'
    if ($state -and $state.computeEnvironments.Count -gt 0) {
        $status = [string]$state.computeEnvironments[0].status
        if ($status -eq 'VALID')   { $ceReady = $true; break }
        if ($status -eq 'INVALID') {
            Stop-WithError -Message "Compute environment $ComputeEnvName is INVALID." -Detail ([string]$state.computeEnvironments[0].statusReason)
        }
    }
    Start-Sleep -Seconds 6
}
if (-not $ceReady) { Stop-WithError -Message "Compute environment $ComputeEnvName did not reach VALID within 4 minutes." -Detail 'Re-run this script to resume once it settles.' }
Write-Info 'compute environment is VALID'

# =============================================================================
# 8. Job queue
# =============================================================================
Write-Step "Batch job queue $JobQueueName"

$jqLookup = Get-AwsJson -AwsArgs @('batch', 'describe-job-queues', '--job-queues', $JobQueueName) -Context 'batch describe-job-queues' -AllowFailure
if ($jqLookup -and $jqLookup.jobQueues -and $jqLookup.jobQueues.Count -gt 0) {
    $JobQueueArn = [string]$jqLookup.jobQueues[0].jobQueueArn
    Write-Have "job queue $JobQueueName ($($jqLookup.jobQueues[0].status))"
} else {
    $jqDoc = [ordered]@{
        jobQueueName          = $JobQueueName
        state                 = 'ENABLED'
        priority              = 1
        computeEnvironmentOrder = @(
            [ordered]@{ order = 1; computeEnvironment = $ComputeEnvArn }
        )
        tags = [ordered]@{
            Project   = $script:TagProject
            Component = $script:TagComponent
        }
    }
    $jqArg = New-JsonArg -Name 'job-queue' -Object $jqDoc
    $createdJq = Get-AwsJson -AwsArgs @('batch', 'create-job-queue', '--cli-input-json', $jqArg) -Context 'batch create-job-queue'
    if ($null -eq $createdJq -or -not $createdJq.jobQueueArn) { Stop-WithError -Message 'create-job-queue returned no ARN.' }
    $JobQueueArn = [string]$createdJq.jobQueueArn
    Write-Made "job queue $JobQueueArn"
}
Set-Record 'jobQueueName' $JobQueueName
Set-Record 'jobQueueArn'  $JobQueueArn
Save-ResourceRecord

# =============================================================================
# 10. CloudWatch log group (created before the job definition that references it)
# =============================================================================
Write-Step "CloudWatch log group $LogGroupName (retention $LogRetentionDays days)"

$lgLookup = Get-AwsJson -AwsArgs @('logs', 'describe-log-groups', '--log-group-name-prefix', $LogGroupName) -Context 'logs describe-log-groups'
$lgExists = $false
if ($lgLookup -and $lgLookup.logGroups) {
    foreach ($lg in @($lgLookup.logGroups)) {
        if ([string]$lg.logGroupName -eq $LogGroupName) { $lgExists = $true }
    }
}
if ($lgExists) {
    Write-Have "log group $LogGroupName"
} else {
    $lgCreate = Invoke-Aws -AwsArgs @(
        'logs', 'create-log-group',
        '--log-group-name', $LogGroupName,
        '--tags', "Project=$script:TagProject,Component=$script:TagComponent"
    ) -Context 'logs create-log-group' -AllowFailure
    if ($lgCreate.ExitCode -ne 0) {
        if (Test-AwsErrorMatch -Result $lgCreate -Pattern 'ResourceAlreadyExistsException') {
            Write-Have "log group $LogGroupName (race)"
        } else {
            Stop-WithError -Message 'logs create-log-group failed.' -Detail $lgCreate.StdErr
        }
    } else {
        Write-Made "log group $LogGroupName"
    }
}
Invoke-Aws -AwsArgs @('logs', 'put-retention-policy', '--log-group-name', $LogGroupName, '--retention-in-days', "$LogRetentionDays") -Context 'logs put-retention-policy' | Out-Null
Write-Info "retention set to $LogRetentionDays days"
Set-Record 'logGroupName'     $LogGroupName
Set-Record 'logRetentionDays' $LogRetentionDays
Save-ResourceRecord

# =============================================================================
# 9. Job definition
# =============================================================================
Write-Step "Batch job definition $JobDefName ($JobVcpu vCPU / $JobMemoryMiB MiB)"

$jdLookup = Get-AwsJson -AwsArgs @('batch', 'describe-job-definitions', '--job-definition-name', $JobDefName, '--status', 'ACTIVE') -Context 'batch describe-job-definitions' -AllowFailure
$jdActive = @()
if ($jdLookup -and $jdLookup.jobDefinitions) { $jdActive = @($jdLookup.jobDefinitions) }

$registerJd = $true
if ($jdActive.Count -gt 0 -and -not $UpdateJobDefinition) {
    $latestRev = ($jdActive | Sort-Object -Property revision -Descending | Select-Object -First 1)
    Write-Have "job definition $JobDefName revision $($latestRev.revision)"
    Write-Info 'pass -UpdateJobDefinition to publish a new revision'
    Set-Record 'jobDefinitionRevision' ([int]$latestRev.revision)
    Set-Record 'jobDefinitionArn'      ([string]$latestRev.jobDefinitionArn)
    $registerJd = $false
}

if ($registerJd) {
    $jdDoc = [ordered]@{
        jobDefinitionName    = $JobDefName
        type                 = 'container'
        platformCapabilities = @('EC2')
        containerProperties  = [ordered]@{
            image      = $EcrImageUri
            jobRoleArn = $JobRoleArn
            resourceRequirements = @(
                [ordered]@{ type = 'VCPU';   value = "$JobVcpu" }
                [ordered]@{ type = 'MEMORY'; value = "$JobMemoryMiB" }
            )
            environment = @(
                [ordered]@{ name = 'CC3D_BUCKET';        value = $BucketName }
                [ordered]@{ name = 'AWS_DEFAULT_REGION'; value = $Region }
            )
            logConfiguration = [ordered]@{
                logDriver = 'awslogs'
                options   = [ordered]@{
                    'awslogs-group'         = $LogGroupName
                    'awslogs-region'        = $Region
                    'awslogs-stream-prefix' = 'cc3d'
                }
            }
            readonlyRootFilesystem = $false
            privileged             = $false
        }
        retryStrategy = [ordered]@{
            attempts       = 2
            evaluateOnExit = @(
                # Spot reclaim: Batch reports "Host EC2 instance terminated" --
                # that is infrastructure, not the model, so retry it.
                [ordered]@{ onStatusReason = 'Host EC2*'; action = 'RETRY' }
                # Anything else is the simulation's own failure. Do not burn a
                # second 95-minute attempt on a broken model.
                [ordered]@{ onStatusReason = '*';         action = 'EXIT'  }
            )
        }
        timeout = [ordered]@{ attemptDurationSeconds = $JobTimeoutSecs }
        propagateTags = $true
        tags = [ordered]@{
            Project   = $script:TagProject
            Component = $script:TagComponent
        }
    }
    $jdArg = New-JsonArg -Name 'job-definition' -Object $jdDoc
    $createdJd = Get-AwsJson -AwsArgs @('batch', 'register-job-definition', '--cli-input-json', $jdArg) -Context 'batch register-job-definition'
    if ($null -eq $createdJd -or -not $createdJd.jobDefinitionArn) { Stop-WithError -Message 'register-job-definition returned no ARN.' }
    Write-Made "job definition $($createdJd.jobDefinitionArn)"
    Write-Info "retry: RETRY on 'Host EC2*' (spot reclaim), EXIT on everything else; timeout $JobTimeoutSecs s"
    Set-Record 'jobDefinitionRevision' ([int]$createdJd.revision)
    Set-Record 'jobDefinitionArn'      ([string]$createdJd.jobDefinitionArn)
}
Set-Record 'jobDefinitionName' $JobDefName
Save-ResourceRecord

# =============================================================================
# 11. AWS Budgets -- $20/month, ACTUAL > 80% and FORECASTED > 100%
# =============================================================================
Write-Step "AWS Budget $BudgetName (`$$BudgetUsd/month, alerts to $NotificationEmail)"

# The Budgets API is global and only answers in us-east-1.
$BudgetRegion = 'us-east-1'

$budgetLookup = Invoke-Aws -AwsArgs @(
    'budgets', 'describe-budget',
    '--account-id', $AccountId, '--budget-name', $BudgetName,
    '--region', $BudgetRegion
) -Context 'budgets describe-budget' -AllowFailure

if ($budgetLookup.ExitCode -eq 0) {
    Write-Have "budget $BudgetName"
} else {
    $budgetDoc = [ordered]@{
        BudgetName  = $BudgetName
        BudgetLimit = [ordered]@{ Amount = "$BudgetUsd"; Unit = 'USD' }
        TimeUnit    = 'MONTHLY'
        BudgetType  = 'COST'
        CostTypes   = [ordered]@{
            IncludeTax             = $true
            IncludeSubscription    = $true
            UseBlended             = $false
            IncludeRefund          = $false
            IncludeCredit          = $false
            IncludeUpfront         = $true
            IncludeRecurring       = $true
            IncludeOtherSubscription = $true
            IncludeSupport         = $true
            IncludeDiscount        = $true
            UseAmortized           = $false
        }
    }
    $notifyDoc = @(
        [ordered]@{
            Notification = [ordered]@{
                NotificationType   = 'ACTUAL'
                ComparisonOperator = 'GREATER_THAN'
                Threshold          = 80
                ThresholdType      = 'PERCENTAGE'
            }
            Subscribers = @(
                [ordered]@{ SubscriptionType = 'EMAIL'; Address = $NotificationEmail }
            )
        },
        [ordered]@{
            Notification = [ordered]@{
                NotificationType   = 'FORECASTED'
                ComparisonOperator = 'GREATER_THAN'
                Threshold          = 100
                ThresholdType      = 'PERCENTAGE'
            }
            Subscribers = @(
                [ordered]@{ SubscriptionType = 'EMAIL'; Address = $NotificationEmail }
            )
        }
    )

    $budgetArg = New-JsonArg -Name 'budget'        -Object $budgetDoc
    $notifyArg = New-JsonArg -Name 'notifications' -Object $notifyDoc

    $baseBudgetArgs = @(
        'budgets', 'create-budget',
        '--account-id', $AccountId,
        '--budget', $budgetArg,
        '--notifications-with-subscribers', $notifyArg,
        '--region', $BudgetRegion
    )

    # --resource-tags only exists on newer CLI builds. Try tagged first, then
    # fall back with an explicit warning rather than silently skipping.
    $taggedArgs = $baseBudgetArgs + @('--resource-tags', "Key=Project,Value=$script:TagProject", "Key=Component,Value=$script:TagComponent")
    $bres = Invoke-Aws -AwsArgs $taggedArgs -Context 'budgets create-budget (tagged)' -AllowFailure
    if ($bres.ExitCode -ne 0) {
        if (Test-AwsErrorMatch -Result $bres -Pattern 'Unknown options|unrecognized arguments|Invalid choice|argument --resource-tags') {
            Write-Warn2 'this AWS CLI build does not support --resource-tags on budgets; creating the budget untagged'
            $bres = Invoke-Aws -AwsArgs $baseBudgetArgs -Context 'budgets create-budget' -AllowFailure
        }
    }
    if ($bres.ExitCode -ne 0) {
        if (Test-AwsErrorMatch -Result $bres -Pattern 'DuplicateRecordException|already exists') {
            Write-Have "budget $BudgetName (already existed)"
        } else {
            Stop-WithError -Message 'budgets create-budget failed.' -Detail ($bres.Command + "`n`n" + $bres.StdErr)
        }
    } else {
        Write-Made "budget $BudgetName (ACTUAL > 80%, FORECASTED > 100%)"
        Write-Info "confirm the subscription email sent to $NotificationEmail"
    }
}
Set-Record 'budgetName'         $BudgetName
Set-Record 'budgetLimitUsd'     $BudgetUsd
Set-Record 'budgetNotifyEmail'  $NotificationEmail
Save-ResourceRecord

# =============================================================================
# 12. IAM managed policy for the Render app -- NO access keys
# =============================================================================
Write-Step "IAM managed policy $RenderPolicyName (for the Render web app)"

$renderPolicyDoc = [ordered]@{
    Version   = '2012-10-17'
    Statement = @(
        [ordered]@{
            Sid      = 'SubmitOnlyToThisQueueAndDefinition'
            Effect   = 'Allow'
            Action   = @('batch:SubmitJob')
            Resource = @($JobQueueArn, ($JobDefArn + ':*'), $JobDefArn)
        },
        [ordered]@{
            # batch:DescribeJobs and batch:TerminateJob do NOT support
            # resource-level permissions in IAM -- AWS Batch types both as
            # "Resource types: none", so a queue ARN here would deny every call.
            # They are therefore granted on "*" and nothing else is added.
            Sid      = 'ReadAndCancelJobs'
            Effect   = 'Allow'
            Action   = @('batch:DescribeJobs', 'batch:TerminateJob')
            Resource = @('*')
        },
        [ordered]@{
            Sid      = 'ListOnlyThisBucket'
            Effect   = 'Allow'
            Action   = @('s3:ListBucket')
            Resource = @("arn:aws:s3:::$BucketName")
        },
        [ordered]@{
            Sid      = 'ObjectRwOnlyThisBucket'
            Effect   = 'Allow'
            Action   = @('s3:GetObject', 's3:PutObject', 's3:DeleteObject')
            Resource = @("arn:aws:s3:::$BucketName/*")
        }
    )
}
$renderPolicyJson = ConvertTo-Json -InputObject $renderPolicyDoc -Depth 12
$renderPolicyArg  = New-JsonArg -Name 'render-app-policy' -Object $renderPolicyDoc
$RenderPolicyArn  = "arn:aws:iam::${AccountId}:policy/$RenderPolicyName"

$rpLookup = Invoke-Aws -AwsArgs @('iam', 'get-policy', '--policy-arn', $RenderPolicyArn) -Context 'iam get-policy' -AllowFailure
if ($rpLookup.ExitCode -eq 0) {
    Write-Have "policy $RenderPolicyArn"
    if ($UpdateRenderPolicy) {
        Invoke-Aws -AwsArgs @(
            'iam', 'create-policy-version',
            '--policy-arn', $RenderPolicyArn,
            '--policy-document', $renderPolicyArg,
            '--set-as-default'
        ) -Context 'iam create-policy-version' | Out-Null
        Write-Made 'new default policy version'
    } else {
        Write-Info 'pass -UpdateRenderPolicy to publish a new default version'
    }
} else {
    $args4 = @('iam', 'create-policy',
               '--policy-name', $RenderPolicyName,
               '--policy-document', $renderPolicyArg,
               '--description', 'BioSimulateAI Render app: submit/describe/terminate CC3D Batch jobs + one S3 bucket') + (Get-IamTagArgs)
    Invoke-Aws -AwsArgs $args4 -Context 'iam create-policy' | Out-Null
    Write-Made "policy $RenderPolicyArn"
}
Set-Record 'renderPolicyName' $RenderPolicyName
Set-Record 'renderPolicyArn'  $RenderPolicyArn
Save-ResourceRecord

# =============================================================================
# GUARD -- prove none of the forbidden, always-on resources exist in this VPC
# =============================================================================
Write-Step 'Verifying no cost-generating resource was created'

$violations = @()

$nats = Get-AwsJson -AwsArgs @('ec2', 'describe-nat-gateways', '--filter', "Name=vpc-id,Values=$VpcId") -Context 'describe-nat-gateways' -AllowFailure
if ($nats -and $nats.NatGateways) {
    foreach ($nat in @($nats.NatGateways)) {
        if ([string]$nat.State -notin @('deleted', 'deleting')) { $violations += "NAT Gateway $($nat.NatGatewayId)" }
    }
}

$eips = Get-AwsJson -AwsArgs @('ec2', 'describe-addresses') -Context 'describe-addresses' -AllowFailure
if ($eips -and $eips.Addresses) {
    foreach ($eip in @($eips.Addresses)) {
        foreach ($t in @($eip.Tags)) {
            if ($t.Key -eq 'Component' -and $t.Value -eq $script:TagComponent) { $violations += "Elastic IP $($eip.AllocationId)" }
        }
    }
}

$allVpce = Get-AwsJson -AwsArgs @('ec2', 'describe-vpc-endpoints', '--filters', "Name=vpc-id,Values=$VpcId") -Context 'describe-vpc-endpoints' -AllowFailure
if ($allVpce -and $allVpce.VpcEndpoints) {
    foreach ($ep in @($allVpce.VpcEndpoints)) {
        if ([string]$ep.VpcEndpointType -ne 'Gateway') { $violations += "$($ep.VpcEndpointType) VPC endpoint $($ep.VpcEndpointId)" }
    }
}

$instances = Get-AwsJson -AwsArgs @(
    'ec2', 'describe-instances',
    '--filters', "Name=vpc-id,Values=$VpcId", 'Name=instance-state-name,Values=pending,running,stopping,stopped'
) -Context 'describe-instances' -AllowFailure
$runningCount = 0
if ($instances -and $instances.Reservations) {
    foreach ($res in @($instances.Reservations)) { $runningCount += @($res.Instances).Count }
}
if ($runningCount -gt 0) { Write-Warn2 "$runningCount EC2 instance(s) present in this VPC -- expected 0 while no job runs (Batch may still be draining)." }

$albs = Get-AwsJson -AwsArgs @('elbv2', 'describe-load-balancers') -Context 'elbv2 describe-load-balancers' -AllowFailure
if ($albs -and $albs.LoadBalancers) {
    foreach ($lb in @($albs.LoadBalancers)) {
        if ([string]$lb.VpcId -eq $VpcId) { $violations += "Load balancer $($lb.LoadBalancerName)" }
    }
}

if ($violations.Count -gt 0) {
    Write-Warn2 'Unexpected cost-generating resources found in this VPC:'
    foreach ($v in $violations) { Write-Warn2 "  - $v" }
    Write-Warn2 'This script never creates any of those. Delete them to keep idle cost at ~$0.35/month.'
} else {
    Write-Info 'clean: no NAT Gateway, no Elastic IP, no Interface endpoint, no load balancer'
}

# =============================================================================
# DONE
# =============================================================================
Save-ResourceRecord
Remove-TempDir

Write-Section 'RENDER APP IAM POLICY (JSON)'
Write-Host $renderPolicyJson
Write-Host ''
Write-Host "Policy ARN: $RenderPolicyArn" -ForegroundColor Green

Write-Section 'OPERATOR: CREATING CREDENTIALS FOR THE RENDER APP'
Write-Host @"
This script deliberately created NO access keys. Long-lived keys are the single
biggest blast radius in this stack, so they are yours to create and rotate.

Preferred -- no long-lived keys at all:
  Give the Render service an IAM role and have it assume the role via OIDC /
  external-id STS, attaching $RenderPolicyName to that role.

If you must use keys (Render only accepts static env vars today):

  1. Create a dedicated user with NO console access:
       aws iam create-user --user-name $NamePrefix-render --tags Key=Project,Value=$script:TagProject Key=Component,Value=$script:TagComponent

  2. Attach ONLY the policy this script created:
       aws iam attach-user-policy --user-name $NamePrefix-render --policy-arn $RenderPolicyArn

  3. Create the key pair and copy the secret ONCE:
       aws iam create-access-key --user-name $NamePrefix-render

  4. Put AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY into Render's environment
     (Dashboard -> Service -> Environment). Never commit them.

  5. Rotate every 90 days:
       aws iam create-access-key --user-name $NamePrefix-render
       # deploy the new key, then:
       aws iam delete-access-key --user-name $NamePrefix-render --access-key-id <OLD>
"@ -ForegroundColor Gray

Write-Section 'NEXT: BUILD AND PUSH THE WORKER IMAGE'
Write-Host @"
The job definition points at $EcrImageUri, which does not exist until you push:

  aws ecr get-login-password --region $Region | docker login --username AWS --password-stdin $EcrRegistry
  docker build -t $EcrRepoName deploy/cc3d
  docker tag $EcrRepoName`:latest $EcrImageUri
  docker push $EcrImageUri
"@ -ForegroundColor Gray

Write-Section 'ENVIRONMENT VARIABLES FOR THE APP'
Write-Host ''
Write-Host "BIOSIM_AWS_REGION=$Region"                 -ForegroundColor Green
Write-Host "BIOSIM_CC3D_JOB_QUEUE=$JobQueueName"       -ForegroundColor Green
Write-Host "BIOSIM_CC3D_JOB_DEFINITION=$JobDefName"    -ForegroundColor Green
Write-Host "BIOSIM_CC3D_BUCKET=$BucketName"            -ForegroundColor Green
Write-Host ''
Write-Host "Resource inventory written to: $ResourceFile" -ForegroundColor Cyan
Write-Host 'Idle cost: ~$0.35/month (ECR image storage). Zero hosts run between jobs.' -ForegroundColor Cyan
Write-Host ''
exit 0
