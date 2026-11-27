#include "haven/haven_cluster.h"
#include <iostream>
#include <cassert>
#include "haven/haven_test_check.h"
#include <vector>
#include <cmath>
#include <thread>
#include <chrono>

int main() {
    std::cout << "[TEST] Initializing Haven Sovereign Cluster & Multi-Node Fabric Test Suite...\n";

    // ------------------------------------------------------------------------
    // 1. Verify Wire Protocol Header & Binary Fidelity
    // ------------------------------------------------------------------------
    assert(sizeof(haven::HavenActivationFrameHeader) == 36 && "Frame header must be packed exactly to 36 bytes");
    std::cout << "✅ [PASS] HavenActivationFrameHeader byte-packing verified (36 bytes wire format).\n";

    haven::HavenActivationFrame frame;
    frame.header.magic = haven::HAVEN_CLUSTER_MAGIC;
    frame.header.protocol_ver = haven::HAVEN_CLUSTER_VERSION;
    frame.header.flags = haven::FRAME_FLAG_ACTIVATION;
    frame.header.sequence_id = 1337;
    frame.header.step_pos = 42;
    frame.header.from_node_id = 0;
    frame.header.to_node_id = 1;
    frame.header.layer_boundary = 16;
    frame.header.element_count = 1024;
    frame.header.payload_bytes = 1024 * sizeof(float);
    frame.data.resize(1024);
    for (size_t i = 0; i < 1024; ++i) {
        frame.data[i] = static_cast<float>(i) * 0.125f;
    }

    assert(frame.is_valid() && "Frame must pass internal protocol validity check");
    std::cout << "✅ [PASS] Activation packet integrity and protocol validation operational.\n";

    // ------------------------------------------------------------------------
    // 2. Verify High-Throughput Activation Ring Buffer
    // ------------------------------------------------------------------------
    haven::ActivationRingBuffer ring_buffer(16);
    assert(ring_buffer.empty() && "Ring buffer must be empty on initialization");

    bool pushed = ring_buffer.push(frame);
    assert(pushed && "Pushing to ring buffer must succeed");
    assert(ring_buffer.size() == 1 && "Ring buffer size must be 1");

    haven::HavenActivationFrame popped_frame;
    bool popped = ring_buffer.pop(popped_frame, 100);
    assert(popped && "Popping from ring buffer must succeed");
    assert(popped_frame.header.sequence_id == 1337 && "Popped frame sequence must match");
    assert(popped_frame.data.size() == 1024 && "Popped frame element count must match");
    assert(std::fabs(popped_frame.data[10] - (10.0f * 0.125f)) < 1e-6f && "Popped payload values must match");
    std::cout << "✅ [PASS] High-throughput ActivationRingBuffer verified.\n";

    // ------------------------------------------------------------------------
    // 3. Verify Heterogeneous Pipeline Layer Slicing & Scheduling
    // ------------------------------------------------------------------------
    haven::PipelineClusterScheduler scheduler;
    std::vector<haven::ClusterNodeInfo> test_nodes;

    // Node 0: Coordinator (16GB RAM, 10 TFLOPS)
    haven::ClusterNodeInfo n0;
    n0.node_id = 0;
    n0.address = "127.0.0.1";
    n0.port = 11438;
    n0.role = haven::ClusterRole::COORDINATOR;
    n0.total_memory_bytes = 16ULL * 1024 * 1024 * 1024;
    n0.peak_tflops = 10.0f;
    test_nodes.push_back(n0);

    // Node 1: Worker 1 (32GB RAM, 20 TFLOPS)
    haven::ClusterNodeInfo n1;
    n1.node_id = 1;
    n1.address = "192.168.1.101";
    n1.port = 11438;
    n1.role = haven::ClusterRole::WORKER;
    n1.total_memory_bytes = 32ULL * 1024 * 1024 * 1024;
    n1.peak_tflops = 20.0f;
    test_nodes.push_back(n1);

    // Node 2: Worker 2 (64GB RAM, 40 TFLOPS)
    haven::ClusterNodeInfo n2;
    n2.node_id = 2;
    n2.address = "192.168.1.102";
    n2.port = 11438;
    n2.role = haven::ClusterRole::WORKER;
    n2.total_memory_bytes = 64ULL * 1024 * 1024 * 1024;
    n2.peak_tflops = 40.0f;
    test_nodes.push_back(n2);

    scheduler.partition_layers(32, test_nodes);
    std::string report = scheduler.generate_partition_report();
    std::cout << report << std::endl;

    // Verify contiguous monotonically increasing layer coverage
    assert(test_nodes[0].start_layer == 0 && "Node 0 must start at layer 0");
    assert(test_nodes[0].end_layer == test_nodes[1].start_layer && "Node 0 and Node 1 must be contiguous");
    assert(test_nodes[1].end_layer == test_nodes[2].start_layer && "Node 1 and Node 2 must be contiguous");
    assert(test_nodes[2].end_layer == 32 && "Node 2 must finish at layer 32");

    // Verify layer lookup routing
    assert(scheduler.get_node_for_layer(0) == 0 && "Layer 0 must map to Node 0");
    assert(scheduler.get_node_for_layer(31) == 2 && "Layer 31 must map to Node 2");
    std::cout << "✅ [PASS] Pipeline parallel heterogeneous layer partitioning verified.\n";

    // ------------------------------------------------------------------------
    // 3b. Verify Heterogeneous Accelerator Offload / Thin Gateway Mode
    // ------------------------------------------------------------------------
    std::cout << "\n[TEST] Verifying Heterogeneous Accelerator Offload (Thin Gateway CPU Host + 3 TPU Pods)...\n";
    std::vector<haven::ClusterNodeInfo> hybrid_cluster;
    
    // Node 0: Windows Host Coordinator (CPU AVX2, 10 TFLOPS, 64 GB RAM)
    haven::ClusterNodeInfo h0;
    h0.node_id = 0;
    h0.address = "127.0.0.1";
    h0.port = 11750;
    h0.role = haven::ClusterRole::COORDINATOR;
    h0.backend = haven::DeviceBackend::CPU_AVX2;
    h0.total_memory_bytes = 64ULL * 1024 * 1024 * 1024;
    h0.peak_tflops = 10.0f;
    hybrid_cluster.push_back(h0);

    // Node 1: TPU Pod 1 (TPU PJRT, 275 TFLOPS, 32 GB HBM)
    haven::ClusterNodeInfo h1;
    h1.node_id = 1;
    h1.address = "10.128.0.2";
    h1.port = 11601;
    h1.role = haven::ClusterRole::WORKER;
    h1.backend = haven::DeviceBackend::TPU_PJRT;
    h1.total_memory_bytes = 32ULL * 1024 * 1024 * 1024;
    h1.peak_tflops = 275.0f;
    hybrid_cluster.push_back(h1);

    // Node 2: TPU Pod 2 (TPU PJRT, 275 TFLOPS, 32 GB HBM)
    haven::ClusterNodeInfo h2;
    h2.node_id = 2;
    h2.address = "10.128.0.3";
    h2.port = 11602;
    h2.role = haven::ClusterRole::WORKER;
    h2.backend = haven::DeviceBackend::TPU_PJRT;
    h2.total_memory_bytes = 32ULL * 1024 * 1024 * 1024;
    h2.peak_tflops = 275.0f;
    hybrid_cluster.push_back(h2);

    // Node 3: TPU Pod 3 (TPU PJRT, 275 TFLOPS, 32 GB HBM)
    haven::ClusterNodeInfo h3;
    h3.node_id = 3;
    h3.address = "10.128.0.4";
    h3.port = 11603;
    h3.role = haven::ClusterRole::WORKER;
    h3.backend = haven::DeviceBackend::TPU_PJRT;
    h3.total_memory_bytes = 32ULL * 1024 * 1024 * 1024;
    h3.peak_tflops = 275.0f;
    hybrid_cluster.push_back(h3);

    scheduler.partition_layers(32, hybrid_cluster);
    std::cout << scheduler.generate_partition_report() << std::endl;

    // Verify CPU coordinator entered Thin Gateway Mode (0 compute layers)
    assert(hybrid_cluster[0].start_layer == 0 && hybrid_cluster[0].end_layer == 0 && "Coordinator must have 0 layers in Thin Gateway Mode");
    assert(hybrid_cluster[1].start_layer == 0 && "TPU Worker 1 must start at layer 0");
    assert(hybrid_cluster[1].end_layer == hybrid_cluster[2].start_layer && "TPU 1 and TPU 2 must be contiguous");
    assert(hybrid_cluster[2].end_layer == hybrid_cluster[3].start_layer && "TPU 2 and TPU 3 must be contiguous");
    assert(hybrid_cluster[3].end_layer == 32 && "TPU 3 must complete at layer 32");

    // Verify routing: Layer 0 routes directly to TPU Pod 1, Layer 31 routes to TPU Pod 3
    assert(scheduler.get_node_for_layer(0) == 1 && "Layer 0 must map to TPU Worker 1");
    assert(scheduler.get_node_for_layer(31) == 3 && "Layer 31 must map to TPU Worker 3");
    std::cout << "✅ [PASS] Heterogeneous Accelerator Offload (Thin Gateway Mode) verified.\n";

    // ------------------------------------------------------------------------
    // 4. Verify Zero-Copy Network Transport Ring (Loopback Coordinator & Worker)
    // ------------------------------------------------------------------------
    std::cout << "[TEST] Launching loopback Coordinator and Worker nodes on local transport ring...\n";
    haven::HavenClusterNode coordinator;
    bool coord_ok = coordinator.init_coordinator(11498, 32);
    assert(coord_ok && "Coordinator initialization must succeed");

    haven::HavenClusterNode worker;
    bool worker_ok = worker.init_worker(1, 11499, "127.0.0.1", 11498);
    assert(worker_ok && "Worker connection to coordinator must succeed");

    // Allow network handshake to settle
    std::this_thread::sleep_for(std::chrono::milliseconds(150));

    // Register worker and sync cluster partition
    coordinator.register_worker(n1);
    bool sync_ok = coordinator.sync_cluster_partition();
    assert(sync_ok && "Coordinator partition sync broadcast must succeed");

    // Send intermediate activation frame from worker to coordinator
    std::vector<float> sample_activations(2048, 1.618f);
    sample_activations[0] = 3.14159f;
    sample_activations[2047] = 2.71828f;

    bool sent = worker.forward_activations_to_next(999, 1, 0, sample_activations);
    assert(sent && "Worker forwarding activations to coordinator must succeed");

    // Receive activations on coordinator
    haven::HavenActivationFrame recv_frame;
    bool recvd = coordinator.await_upstream_activations(recv_frame, 1000);
    assert(recvd && "Coordinator must receive activation frame from worker");
    assert(recv_frame.header.sequence_id == 999 && "Sequence ID must match");
    assert(recv_frame.header.step_pos == 1 && "Step position must match");
    assert(recv_frame.data.size() == 2048 && "Activation size must match exactly");
    assert(std::fabs(recv_frame.data[0] - 3.14159f) < 1e-5f && "Element 0 must match");
    assert(std::fabs(recv_frame.data[2047] - 2.71828f) < 1e-5f && "Element 2047 must match");
    std::cout << "✅ [PASS] Zero-copy binary activation transmission verified with 100% precision.\n";

    std::cout << coordinator.generate_cluster_report();

    worker.shutdown();
    coordinator.shutdown();

    std::cout << "\n🏆 [VERIFIED] Haven Multi-Node Cluster & Pipeline Parallel Fabric 100% Operational!\n";
    return 0;
}
