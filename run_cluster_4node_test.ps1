# 4-Node Heterogeneous Cluster Verification: Windows Host + 3 Cloud TPU Pods
param(
    [string]$SshKey = $(if ($env:HAVEN_SSH_KEY) { $env:HAVEN_SSH_KEY } else { "$HOME\.ssh\haven_ed25519" }),
    [string]$SshUser = $(if ($env:HAVEN_SSH_USER) { $env:HAVEN_SSH_USER } else { $env:USERNAME }),
    [int]$CoordPort = 11750,
    [int]$Steps = 10,
    [int]$TotalLayers = 32,
    [string]$Node1Ext = $(if ($env:HAVEN_NODE1_EXT) { $env:HAVEN_NODE1_EXT } else { "127.0.0.1" }),
    [string]$Node2Ext = $(if ($env:HAVEN_NODE2_EXT) { $env:HAVEN_NODE2_EXT } else { "127.0.0.1" }),
    [string]$Node3Ext = $(if ($env:HAVEN_NODE3_EXT) { $env:HAVEN_NODE3_EXT } else { "127.0.0.1" }),
    [string]$Node1Int = $(if ($env:HAVEN_NODE1_INT) { $env:HAVEN_NODE1_INT } else { "10.128.0.2" }),
    [string]$Node2Int = $(if ($env:HAVEN_NODE2_INT) { $env:HAVEN_NODE2_INT } else { "10.128.0.3" }),
    [string]$Node3Int = $(if ($env:HAVEN_NODE3_INT) { $env:HAVEN_NODE3_INT } else { "10.128.0.4" })
)
$sshKey = $SshKey
$coordPort = $CoordPort
$steps = $Steps
$totalLayers = $TotalLayers

$node1_ext = $Node1Ext
$node2_ext = $Node2Ext
$node3_ext = $Node3Ext

$node1_int = $Node1Int
$node2_int = $Node2Int
$node3_int = $Node3Int

$port1 = 11601
$port2 = 11602
$port3 = 11603

$coordOut = Join-Path $PSScriptRoot "coordinator_output.log"
$coordErr = Join-Path $PSScriptRoot "coordinator_error.log"
if (Test-Path $coordOut) { Remove-Item $coordOut -Force }
if (Test-Path $coordErr) { Remove-Item $coordErr -Force }

Write-Host "=================================================================="
Write-Host "👑 HAVEN 4-NODE HETEROGENEOUS CLUSTER FABRIC VERIFICATION"
Write-Host "=================================================================="
Write-Host "Master Host   : Windows Coordinator (127.0.0.1:$coordPort)"
Write-Host "Worker 1      : Cloud TPU Pod 1 ($node1_ext / $node1_int : $port1)"
Write-Host "Worker 2      : Cloud TPU Pod 2 ($node2_ext / $node2_int : $port2)"
Write-Host "Worker 3      : Cloud TPU Pod 3 ($node3_ext / $node3_int : $port3)"
Write-Host "Pipeline Ring : Windows -> TPU 1 -> TPU 2 -> TPU 3 -> Windows"
Write-Host "Target Model  : Ava (32 Sliced Transformer Layers)"
Write-Host "Steps         : $steps Autoregressive Pipeline Steps"
Write-Host "==================================================================`n"

# 1. Start Windows Host Coordinator
Write-Host "[1/4] Starting Host Coordinator on Windows (Port $coordPort)..."
$coordExe = Join-Path $PSScriptRoot "build\test-cross-system-node.exe"
$coordProc = Start-Process -FilePath $coordExe `
    -ArgumentList "coordinator $coordPort 3 $totalLayers $node1_ext $port1 $steps" `
    -RedirectStandardOutput $coordOut `
    -RedirectStandardError $coordErr `
    -PassThru

Start-Sleep -Seconds 1

# 2. Launch Worker 3 on Cloud TPU Pod 3
Write-Host "[2/4] Starting Worker 3 on Cloud TPU Pod 3 ($node3_ext)..."
$w3Cmd = "cd haven-cpp && ./build-linux/test-cross-system-node worker 3 $port3 127.0.0.1 $coordPort 127.0.0.1 $coordPort 0 1 $steps > /tmp/w3.log 2>&1"
$w3Job = Start-Job -ScriptBlock {
    param($key, $user, $ext, $cmd, $cPort)
    & ssh -i $key -o StrictHostKeyChecking=accept-new -R ${cPort}:127.0.0.1:${cPort} "${user}@${ext}" "$cmd"
} -ArgumentList $sshKey, $SshUser, $node3_ext, $w3Cmd, $coordPort

Start-Sleep -Seconds 1

# 3. Launch Worker 2 on Cloud TPU Pod 2
Write-Host "[3/4] Starting Worker 2 on Cloud TPU Pod 2 ($node2_ext)..."
$w2Cmd = "cd haven-cpp && ./build-linux/test-cross-system-node worker 2 $port2 127.0.0.1 $coordPort $node3_int $port3 3 0 $steps > /tmp/w2.log 2>&1"
$w2Job = Start-Job -ScriptBlock {
    param($key, $user, $ext, $cmd, $cPort)
    & ssh -i $key -o StrictHostKeyChecking=accept-new -R ${cPort}:127.0.0.1:${cPort} "${user}@${ext}" "$cmd"
} -ArgumentList $sshKey, $SshUser, $node2_ext, $w2Cmd, $coordPort

Start-Sleep -Seconds 1

# 4. Launch Worker 1 on Cloud TPU Pod 1
Write-Host "[4/4] Starting Worker 1 on Cloud TPU Pod 1 ($node1_ext)..."
$w1Cmd = "cd haven-cpp && ./build-linux/test-cross-system-node worker 1 $port1 127.0.0.1 $coordPort $node2_int $port2 2 0 $steps > /tmp/w1.log 2>&1"
$w1Job = Start-Job -ScriptBlock {
    param($key, $user, $ext, $cmd, $cPort)
    & ssh -i $key -o StrictHostKeyChecking=accept-new -R ${cPort}:127.0.0.1:${cPort} "${user}@${ext}" "$cmd"
} -ArgumentList $sshKey, $SshUser, $node1_ext, $w1Cmd, $coordPort

Write-Host "`n⏳ Executing distributed pipeline across all 4 nodes..."

# Monitor execution
$timeoutSec = 25
$elapsed = 0
while (!$coordProc.HasExited -and $elapsed -lt $timeoutSec) {
    Start-Sleep -Seconds 1
    $elapsed++
}

if (!$coordProc.HasExited) {
    Write-Host "⚠️ Coordinator timed out after $timeoutSec seconds. Terminating..."
    Stop-Process -Id $coordProc.Id -Force -ErrorAction SilentlyContinue
} else {
    Write-Host "✅ Coordinator finished with exit code $($coordProc.ExitCode)!"
}

# Fetch logs
Write-Host "`n=================================================================="
Write-Host "📋 HOST COORDINATOR LOGS (WINDOWS):"
Write-Host "=================================================================="
if (Test-Path $coordOut) { Get-Content $coordOut }
if (Test-Path $coordErr) { Get-Content $coordErr }

Write-Host "`n=================================================================="
Write-Host "📋 WORKER 1 LOGS (TPU POD 1):"
Write-Host "=================================================================="
ssh -i $sshKey -o StrictHostKeyChecking=accept-new ssfdre38@$node1_ext "cat /tmp/w1.log"

Write-Host "`n=================================================================="
Write-Host "📋 WORKER 2 LOGS (TPU POD 2):"
Write-Host "=================================================================="
ssh -i $sshKey -o StrictHostKeyChecking=accept-new ssfdre38@$node2_ext "cat /tmp/w2.log"

Write-Host "`n=================================================================="
Write-Host "📋 WORKER 3 LOGS (TPU POD 3):"
Write-Host "=================================================================="
ssh -i $sshKey -o StrictHostKeyChecking=accept-new ssfdre38@$node3_ext "cat /tmp/w3.log"

# Clean up jobs
Stop-Job $w1Job -ErrorAction SilentlyContinue
Stop-Job $w2Job -ErrorAction SilentlyContinue
Stop-Job $w3Job -ErrorAction SilentlyContinue
Remove-Job $w1Job -Force -ErrorAction SilentlyContinue
Remove-Job $w2Job -Force -ErrorAction SilentlyContinue
Remove-Job $w3Job -Force -ErrorAction SilentlyContinue
