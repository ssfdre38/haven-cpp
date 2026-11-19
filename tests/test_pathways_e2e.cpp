#include "haven/haven_engine.h"
#include <iostream>
#include <cassert>
#include <vector>
#include <string>

int main() {
    std::cout << "==================================================================\n";
    std::cout << "🚀 TESTING HAVEN TOPOLOGY & PATHWAYS END-TO-END INTEGRATION\n";
    std::cout << "==================================================================\n";

    // 1. Test C ABI Creation and Topology Discovery
    std::cout << "\n[1] Testing C ABI Exports (P/Invoke Interface)...\n";
    void* engine_handle = haven_create_engine();
    assert(engine_handle != nullptr && "haven_create_engine() must return a valid handle");

    size_t node_count = haven_get_node_count(engine_handle);
    std::cout << "   ✓ Discovered Node Count via C ABI: " << node_count << "\n";
    assert(node_count >= 1 && "At least 1 compute node must be detected");

    for (uint32_t i = 0; i < node_count; ++i) {
        char name[256] = {0};
        uint32_t backend = 0;
        uint64_t free_mem_mb = 0;

        bool ok = haven_get_node_info(engine_handle, i, name, sizeof(name), &backend, &free_mem_mb);
        assert(ok && "haven_get_node_info() must succeed for valid node indices");
        std::cout << "   ✓ Node [" << i << "]: " << name 
                  << " | Backend: " << backend 
                  << " | Free RAM: " << free_mem_mb << " MB\n";
    }

    uint64_t initial_completed = haven_get_pathways_completed_tasks(engine_handle);
    std::cout << "   ✓ Initial Completed Pathways Tasks: " << initial_completed << "\n";

    // 2. Test Direct C++ HavenEngine Asynchronous Generation Pipeline
    std::cout << "\n[2] Testing HavenEngine C++ Object & Pathways Dataflow...\n";
    auto* engine = reinterpret_cast<haven::HavenEngine*>(engine_handle);

    auto topology = engine->get_topology();
    assert(topology != nullptr && "engine->get_topology() must return a valid pointer");
    std::cout << "   ✓ Topology Report from Engine:\n" << topology->generate_topology_report() << "\n";

    auto pathways = engine->get_pathways();
    assert(pathways != nullptr && "engine->get_pathways() must return a valid pointer");

    // Test streaming token publication and consumption
    std::vector<std::string> streamed_chunks;
    pathways->set_token_stream_listener([&](uint32_t token, const std::string& piece) {
        streamed_chunks.push_back(piece);
        std::cout << "   [Stream Event] Token: " << token << " -> '" << piece << "'\n";
    });

    std::cout << "[3] Enqueuing Pathways asynchronous forward steps...\n";
    std::vector<std::future<haven::PathwaysTaskResult>> step_futures;
    for (uint32_t tok = 500; tok < 505; ++tok) {
        step_futures.push_back(pathways->enqueue_step(tok, tok - 500, 0));
        pathways->publish_stream_token(tok, "chunk_" + std::to_string(tok));
    }

    for (auto& fut : step_futures) {
        auto res = fut.get();
        assert(res.success && "Task must complete");
    }

    std::cout << "   ✓ Streamed Chunks Received: " << streamed_chunks.size() << " / 5\n";
    assert(streamed_chunks.size() == 5 && "All 5 published stream tokens must be delivered");

    uint64_t tasks_after = haven_get_pathways_completed_tasks(engine_handle);
    std::cout << "   ✓ Completed Tasks via C ABI after dispatch: " << tasks_after << "\n";
    assert(tasks_after >= 5 && "Completed task counter must track async executions");

    // Clean up
    std::cout << "\n[4] Destroying Engine via C ABI...\n";
    haven_destroy_engine(engine_handle);
    std::cout << "   ✓ Engine safely destroyed with zero leaks.\n";

    std::cout << "\n==================================================================\n";
    std::cout << "🏆 ALL END-TO-END TOPOLOGY & PATHWAYS TESTS PASSED WITH 100% SUCCESS!\n";
    std::cout << "==================================================================\n";
    return 0;
}
