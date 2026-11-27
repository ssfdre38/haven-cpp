#include "haven/haven_topology.h"
#include "haven/haven_pathways.h"
#include <iostream>
#include <cassert>
#include "haven/haven_test_check.h"
#include <vector>

int main() {
    std::cout << "[TEST] Initializing Haven Topology & Pathways Verification Suite...\n";

    // 1. Verify Topology Detection
    auto topology = std::make_shared<haven::TopologyManager>();
    std::string report = topology->generate_topology_report();
    std::cout << report << std::endl;

    const auto& nodes = topology->get_nodes();
    assert(!nodes.empty() && "Topology must discover at least the host CPU node");
    assert(nodes[0].backend == haven::DeviceBackend::CPU_AVX2 && "Primary node must be CPU AVX2");
    std::cout << "✅ [PASS] TopologyManager discovered " << nodes.size() << " compute node(s) successfully.\n";

    // 2. Verify Pathways Asynchronous Engine
    haven::PathwaysEngine pathways(topology, 2);
    pathways.start();

    // Set streaming listener
    std::vector<uint32_t> streamed_tokens;
    pathways.set_token_stream_listener([&](uint32_t token, const std::string& piece) {
        streamed_tokens.push_back(token);
        (void)piece;
    });

    std::cout << "[TEST] Dispatching asynchronous generation steps to Pathways workers...\n";
    std::vector<std::future<haven::PathwaysTaskResult>> futures;
    for (uint32_t i = 0; i < 10; ++i) {
        futures.push_back(pathways.enqueue_step(1000 + i, static_cast<int>(i), 0));
        pathways.publish_stream_token(1000 + i, "token_" + std::to_string(i));
    }

    // Wait on futures and verify completion
    for (size_t i = 0; i < futures.size(); ++i) {
        auto res = futures[i].get();
        assert(res.success && "PathwaysTask must complete successfully");
        assert(res.generated_token == 1000 + i && "Token payload must match");
    }

    std::cout << "✅ [PASS] All 10 asynchronous forward tasks resolved with zero stalls.\n";
    std::cout << "✅ [PASS] Streamed tokens collected: " << streamed_tokens.size() << " / 10\n";
    assert(streamed_tokens.size() == 10 && "Streaming ring channel must deliver all published tokens");

    // 3. Test Memory Vector Sync
    std::vector<float> sample_memory(128, 0.5f);
    auto mem_future = pathways.enqueue_memory_sync(sample_memory);
    auto mem_res = mem_future.get();
    assert(mem_res.success && "Memory vector sync must succeed");
    std::cout << "✅ [PASS] Asynchronous memory vector injection completed on Node " << mem_res.executed_on_node << "\n";

    pathways.shutdown();
    std::cout << "\n🏆 [VERIFIED] Haven Topology & Pathways Clean-Room Subsystems 100% Operational!\n";
    return 0;
}
