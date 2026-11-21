# 4-Node Live Distributed Cluster Inference: Windows MicaNT Host + 3 Cloud TPU Pods
$sshKey = "C:\Users\admin\.ssh\haven_ed25519"
$coordPort = 11750

$node1_ext = "34.29.156.182"
$node2_ext = "34.56.222.251"
$node3_ext = "34.66.90.118"

$node1_int = "10.128.0.2"
$node2_int = "10.128.0.3"
$node3_int = "10.128.0.4"

$port1 = 11601
$port2 = 11602
$port3 = 11603

$coordOut = "C:\Users\admin\source\haven-cpp\coordinator_live_output.log"
$coordErr = "C:\Users\admin\source\haven-cpp\coordinator_live_error.log"
if (Test-Path $coordOut) { Remove-Item $coordOut -Force }
if (Test-Path $coordErr) { Remove-Item $coordErr -Force }

Write-Host "=================================================================="
Write-Host "👑 HAVEN 4-NODE SOVEREIGN CLUSTER: AVA LIVE GENERATION"
Write-Host "=================================================================="
Write-Host "Master Host   : Windows MicaNT Coordinator (Layers 0..7)"
Write-Host "Worker 1      : Cloud TPU Pod 1 ($node1_ext / $node1_int : Layers 8..15)"
Write-Host "Worker 2      : Cloud TPU Pod 2 ($node2_ext / $node2_int : Layers 16..23)"
Write-Host "Worker 3      : Cloud TPU Pod 3 ($node3_ext / $node3_int : Layers 24..31 + LM Head)"
Write-Host "Pipeline Ring : Windows -> TPU 1 -> TPU 2 -> TPU 3 -> Windows"
Write-Host "Orchestration : Google ML Pathways Asynchronous Dataflow DAG"
Write-Host "Model         : Ava (haven-chat-v5.0)"
Write-Host "==================================================================`n"

# 1. Start Windows Host Coordinator
Write-Host "[1/4] Starting Host Coordinator on Windows (Port $coordPort)..."
$coordProc = Start-Process -FilePath "C:\Users\admin\source\haven-cpp\build\test-tpu-live-generation.exe" `
    -ArgumentList "coordinator $coordPort 3 $node1_ext $port1" `
    -RedirectStandardOutput $coordOut `
    -RedirectStandardError $coordErr `
    -PassThru

Start-Sleep -Seconds 1

# 2. Launch Worker 3 on Cloud TPU Pod 3 (Layers 24..31 + LM Head)
Write-Host "[2/4] Starting Worker 3 on Cloud TPU Pod 3 (Layers 24..31)..."
$w3Cmd = "cd haven-cpp && ./build-linux/test-tpu-live-generation worker 3 $port3 127.0.0.1 $coordPort 127.0.0.1 $coordPort 0 1 24 32 > /tmp/w3_live.log 2>&1"
$w3Job = Start-Job -ScriptBlock {
    param($key, $ext, $cmd, $cPort)
    & ssh -i $key -o StrictHostKeyChecking=accept-new -R ${cPort}:127.0.0.1:${cPort} ssfdre38@$ext "$cmd"
} -ArgumentList $sshKey, $node3_ext, $w3Cmd, $coordPort

Start-Sleep -Seconds 1

# 3. Launch Worker 2 on Cloud TPU Pod 2 (Layers 16..23)
Write-Host "[3/4] Starting Worker 2 on Cloud TPU Pod 2 (Layers 16..23)..."
$w2Cmd = "cd haven-cpp && ./build-linux/test-tpu-live-generation worker 2 $port2 127.0.0.1 $coordPort $node3_int $port3 3 0 16 24 > /tmp/w2_live.log 2>&1"
$w2Job = Start-Job -ScriptBlock {
    param($key, $ext, $cmd, $cPort)
    & ssh -i $key -o StrictHostKeyChecking=accept-new -R ${cPort}:127.0.0.1:${cPort} ssfdre38@$ext "$cmd"
} -ArgumentList $sshKey, $node2_ext, $w2Cmd, $coordPort

Start-Sleep -Seconds 1

# 4. Launch Worker 1 on Cloud TPU Pod 1 (Layers 8..15)
Write-Host "[4/4] Starting Worker 1 on Cloud TPU Pod 1 (Layers 8..15)..."
$w1Cmd = "cd haven-cpp && ./build-linux/test-tpu-live-generation worker 1 $port1 127.0.0.1 $coordPort $node2_int $port2 2 0 8 16 > /tmp/w1_live.log 2>&1"
$w1Job = Start-Job -ScriptBlock {
    param($key, $ext, $cmd, $cPort)
    & ssh -i $key -o StrictHostKeyChecking=accept-new -R ${cPort}:127.0.0.1:${cPort} ssfdre38@$ext "$cmd"
} -ArgumentList $sshKey, $node1_ext, $w1Cmd, $coordPort

Write-Host "`n📡 Streaming live token activations through distributed Google ML Pathways DAG..."

# Monitor execution
$timeoutSec = 60
$elapsed = 0
while (!$coordProc.HasExited -and $elapsed -lt $timeoutSec) {
    Start-Sleep -Seconds 1
    $elapsed++
}

if (!$coordProc.HasExited) {
    Write-Host "⚠️ Coordinator timed out after $timeoutSec seconds. Terminating..."
    Stop-Process -Id $coordProc.Id -Force -ErrorAction SilentlyContinue
} else {
    Write-Host "✅ Distributed inference run completed with exit code $($coordProc.ExitCode)!"
}

# Fetch and print live output
Write-Host "`n=================================================================="
Write-Host "📋 LIVE STREAMED GENERATION & AUDIT REPORT (WINDOWS COORDINATOR):"
Write-Host "=================================================================="
if (Test-Path $coordOut) { Get-Content $coordOut }
if (Test-Path $coordErr) { Get-Content $coordErr }

Write-Host "`n=================================================================="
Write-Host "📋 WORKER 1 LOGS (TPU POD 1):"
Write-Host "=================================================================="
ssh -i $sshKey -o StrictHostKeyChecking=accept-new ssfdre38@$node1_ext "cat /tmp/w1_live.log"

Write-Host "`n=================================================================="
Write-Host "📋 WORKER 2 LOGS (TPU POD 2):"
Write-Host "=================================================================="
ssh -i $sshKey -o StrictHostKeyChecking=accept-new ssfdre38@$node2_ext "cat /tmp/w2_live.log"

Write-Host "`n=================================================================="
Write-Host "📋 WORKER 3 LOGS (TPU POD 3):"
Write-Host "=================================================================="
ssh -i $sshKey -o StrictHostKeyChecking=accept-new ssfdre38@$node3_ext "cat /tmp/w3_live.log"

# Clean up jobs
Stop-Job $w1Job -ErrorAction SilentlyContinue
Stop-Job $w2Job -ErrorAction SilentlyContinue
Stop-Job $w3Job -ErrorAction SilentlyContinue
Remove-Job $w1Job -Force -ErrorAction SilentlyContinue
Remove-Job $w2Job -Force -ErrorAction SilentlyContinue
Remove-Job $w3Job -Force -ErrorAction SilentlyContinue
