#include "haven/haven_pathways.h"
#include <chrono>
#include <iostream>

namespace haven {

PathwaysEngine::PathwaysEngine(std::shared_ptr<TopologyManager> topology, size_t worker_threads)
    : topology_(std::move(topology)),
      worker_count_(worker_threads > 0 ? worker_threads : 2)
{
    if (!topology_) {
        topology_ = std::make_shared<TopologyManager>();
    }
}

PathwaysEngine::~PathwaysEngine() {
    shutdown();
}

void PathwaysEngine::start() {
    if (is_running_.exchange(true)) {
        return; // Already running
    }

    workers_.clear();
    workers_.reserve(worker_count_);
    for (size_t i = 0; i < worker_count_; ++i) {
        workers_.emplace_back(&PathwaysEngine::worker_loop, this, i);
    }
}

void PathwaysEngine::shutdown() {
    if (!is_running_.exchange(false)) {
        return;
    }

    queue_cv_.notify_all();
    for (auto& worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
    workers_.clear();
}

std::future<PathwaysTaskResult> PathwaysEngine::enqueue_step(uint32_t token, int pos, uint32_t preferred_node) {
    auto promise = std::make_shared<std::promise<PathwaysTaskResult>>();
    auto future = promise->get_future();

    PathwaysTask task;
    task.task_id = next_task_id_.fetch_add(1);
    task.type = PathwaysTaskType::FORWARD_STEP;
    task.target_node_id = preferred_node;
    task.token = token;
    task.position = pos;

    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        task_queue_.emplace(std::move(task), promise);
    }
    queue_cv_.notify_one();

    return future;
}

std::future<PathwaysTaskResult> PathwaysEngine::enqueue_memory_sync(const std::vector<float>& memory_vector) {
    auto promise = std::make_shared<std::promise<PathwaysTaskResult>>();
    auto future = promise->get_future();

    PathwaysTask task;
    task.task_id = next_task_id_.fetch_add(1);
    task.type = PathwaysTaskType::MEMORY_INJECT;
    task.target_node_id = 0; // Host CPU memory domain
    task.input_vector = memory_vector;

    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        task_queue_.emplace(std::move(task), promise);
    }
    queue_cv_.notify_one();

    return future;
}

void PathwaysEngine::set_token_stream_listener(std::function<void(uint32_t token, const std::string& piece)> listener) {
    std::lock_guard<std::mutex> lock(listener_mutex_);
    token_listener_ = std::move(listener);
}

void PathwaysEngine::publish_stream_token(uint32_t token, const std::string& piece) {
    std::lock_guard<std::mutex> lock(listener_mutex_);
    if (token_listener_) {
        token_listener_(token, piece);
    }
}

size_t PathwaysEngine::get_queue_depth() const {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    return task_queue_.size();
}

void PathwaysEngine::worker_loop(size_t worker_idx) {
    (void)worker_idx;
    while (is_running_) {
        std::pair<PathwaysTask, std::shared_ptr<std::promise<PathwaysTaskResult>>> item;
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            queue_cv_.wait(lock, [this]() {
                return !is_running_ || !task_queue_.empty();
            });

            if (!is_running_ && task_queue_.empty()) {
                break;
            }

            item = std::move(task_queue_.front());
            task_queue_.pop();
        }

        auto start_time = std::chrono::high_resolution_clock::now();
        const auto& task = item.first;
        auto& promise = item.second;

        PathwaysTaskResult result;
        result.task_id = task.task_id;
        result.executed_on_node = task.target_node_id;

        // Process Task based on categorical operation type
        if (task.type == PathwaysTaskType::FORWARD_STEP) {
            result.generated_token = task.token;
            result.success = true;
        } else if (task.type == PathwaysTaskType::MEMORY_INJECT) {
            result.success = !task.input_vector.empty();
        } else {
            result.success = true;
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        result.execution_time_ms = std::chrono::duration<float, std::milli>(end_time - start_time).count();

        completed_tasks_.fetch_add(1);
        if (promise) {
            promise->set_value(result);
        }
    }
}

} // namespace haven
