#pragma once

#include "haven_types.h"
#include "haven_topology.h"
#include <cstdint>
#include <string>
#include <vector>
#include <functional>
#include <memory>
#include <future>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <atomic>

namespace haven {

// Categorical asynchronous execution task types
enum class PathwaysTaskType : uint32_t {
    FORWARD_STEP    = 0,
    PREFILL_CHUNK   = 1,
    MEMORY_INJECT   = 2,
    SENSORY_TICK    = 3,
    TOPOLOGY_SYNC   = 4
};

// Result payload of an asynchronous dataflow operation
struct PathwaysTaskResult {
    uint64_t task_id = 0;
    bool success = false;
    uint32_t generated_token = 0;
    std::string token_piece;
    float execution_time_ms = 0.0f;
    uint32_t executed_on_node = 0;
};

// Single task unit submitted to the asynchronous dataflow engine
struct PathwaysTask {
    uint64_t task_id = 0;
    PathwaysTaskType type = PathwaysTaskType::FORWARD_STEP;
    uint32_t target_node_id = 0;
    uint32_t token = 0;
    int position = 0;
    std::vector<float> input_vector;
    std::function<void(const PathwaysTaskResult&)> on_complete;
};

// Clean-Room Asynchronous Dataflow Scheduler & Execution Engine
class PathwaysEngine {
public:
    explicit PathwaysEngine(std::shared_ptr<TopologyManager> topology, size_t worker_threads = 2);
    ~PathwaysEngine();

    // Starts the asynchronous worker pool
    void start();

    // Stops workers and flushes execution queues
    void shutdown();

    // Submits an asynchronous forward generation step
    std::future<PathwaysTaskResult> enqueue_step(uint32_t token, int pos, uint32_t preferred_node = 0);

    // Submits a sensory or episodic memory vector injection task
    std::future<PathwaysTaskResult> enqueue_memory_sync(const std::vector<float>& memory_vector);

    // Streaming token callback: registers consumer listener for live generation chunks
    void set_token_stream_listener(std::function<void(uint32_t token, const std::string& piece)> listener);

    // Yields generated token piece into the non-blocking streaming channel
    void publish_stream_token(uint32_t token, const std::string& piece);

    // Diagnostics: total completed tasks and queue depth
    size_t get_queue_depth() const;
    uint64_t get_completed_task_count() const { return completed_tasks_.load(); }

    const TopologyManager& get_topology() const { return *topology_; }

private:
    void worker_loop(size_t worker_idx);

    std::shared_ptr<TopologyManager> topology_;
    size_t worker_count_;
    std::vector<std::thread> workers_;

    std::queue<std::pair<PathwaysTask, std::shared_ptr<std::promise<PathwaysTaskResult>>>> task_queue_;
    mutable std::mutex queue_mutex_;
    std::condition_variable queue_cv_;

    std::atomic<bool> is_running_{false};
    std::atomic<uint64_t> next_task_id_{1};
    std::atomic<uint64_t> completed_tasks_{0};

    std::function<void(uint32_t, const std::string&)> token_listener_;
    std::mutex listener_mutex_;
};

} // namespace haven
