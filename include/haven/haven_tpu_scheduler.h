#pragma once

#include "haven_sampler.h"
#include "haven_tokenizer.h"
#include "haven_tpu_gemma4.h"
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace haven {

// Abstract backend adapter so TpuContinuousBatchScheduler can drive either
// real HavenTpuGemma4Engine HBM3 silicon OR a deterministic offline mock for CI testing.
class ITpuBatchBackend {
public:
    virtual ~ITpuBatchBackend() = default;
    virtual int batch_capacity() const = 0;
    virtual int max_seq_len() const = 0;
    virtual int vocab_size() const = 0;
    virtual bool returns_full_logits() const = 0;
    virtual std::string last_error() const = 0;

    virtual bool prefill_slot(int slot_id,
                              const std::vector<uint32_t>& tokens,
                              int start_pos,
                              std::vector<float>* out_logits,
                              uint32_t* out_next_token) = 0;

    virtual bool decode_step_batched(const std::vector<uint32_t>& tokens_per_batch,
                                     const std::vector<int>& pos_per_slot,
                                     std::vector<float>* out_logits,
                                     std::vector<uint32_t>& out_next_tokens) = 0;
};

struct TpuSchedulerStats {
    int batch_capacity = 0;
    int active_slots = 0;
    int queued_requests = 0;
    uint64_t total_requests_completed = 0;
    uint64_t total_batched_steps = 0;
    uint64_t total_tokens_generated = 0;
    double avg_batch_utilization = 0.0;
    double last_step_ms = 0.0;
};

struct TpuGenerationRequest {
    std::vector<uint32_t> prompt_tokens;
    int max_tokens = 512;
    SamplerParams sampler_params;
    bool use_custom_sampler = false;
};

// Multi-Tenant Continuous-Batching Scheduler for Cloud TPU HBM3 Serving.
// Owns a dedicated TPU worker thread that multiplexes up to B_cap concurrent
// HTTP/SSE client streams into shared on-chip HBM3 KV slots [0..B_cap-1]:
//   - Slot-targeted MXU prefill (S=128/32/16/1) admits new users with zero interference
//   - Lockstep async-position decode_step_batched (pos_vec[B_cap]) advances all active users
//   - Parallel per-slot PersonaSampler (or zero-overhead on-chip argmax when greedy)
//   - Immediate slot reclamation on EOS, max_tokens, or client socket disconnect
class TpuContinuousBatchScheduler {
public:
    TpuContinuousBatchScheduler();
    ~TpuContinuousBatchScheduler();

    TpuContinuousBatchScheduler(const TpuContinuousBatchScheduler&) = delete;
    TpuContinuousBatchScheduler& operator=(const TpuContinuousBatchScheduler&) = delete;

    // Start scheduler over a real HavenTpuGemma4Engine
    bool start(HavenTpuGemma4Engine* engine,
               const HavenTokenizer* tokenizer,
               const PersonaSampler& template_sampler);

    // Start scheduler over a custom/mock backend (offline CI testing)
    bool start_with_backend(std::unique_ptr<ITpuBatchBackend> backend,
                            const HavenTokenizer* tokenizer,
                            const PersonaSampler& template_sampler);

    void stop();
    bool is_running() const { return running_.load(std::memory_order_acquire); }

    // Synchronous streaming API invoked concurrently by HTTP worker threads (lock-free on engine_mutex_!).
    // `on_token(uint32_t tok)` returns true to continue streaming, false to cancel immediately.
    // Optional `is_cancelled_fn()` is polled while waiting so dropped sockets free slots promptly.
    bool generate_streaming(const TpuGenerationRequest& req,
                            const std::function<bool(uint32_t token_id)>& on_token,
                            const std::function<bool()>& is_cancelled_fn = nullptr,
                            std::string* err_out = nullptr);

    TpuSchedulerStats get_stats() const;

private:
    struct RequestContext {
        uint64_t request_id = 0;
        TpuGenerationRequest req;
        std::mutex mtx;
        std::condition_variable cv;
        std::deque<uint32_t> ready_tokens;
        bool done = false;
        bool failed = false;
        std::string error_msg;
        std::atomic<bool> cancelled{false};
    };

    struct SlotState {
        bool active = false;
        std::shared_ptr<RequestContext> ctx;
        int cur_pos = 0;
        uint32_t pending_tok = 0;
        int generated_count = 0;
        PersonaSampler sampler;
        std::vector<uint32_t> context_history;
    };

    void worker_loop();
    bool is_stop_token_id(uint32_t tok) const;
    void finish_request(const std::shared_ptr<RequestContext>& ctx, bool failed = false, const std::string& err = "");
    void push_token(const std::shared_ptr<RequestContext>& ctx, uint32_t tok);

    std::unique_ptr<ITpuBatchBackend> backend_;
    const HavenTokenizer* tokenizer_ = nullptr;
    PersonaSampler template_sampler_;

    std::atomic<bool> running_{false};
    std::thread worker_thread_;

    std::mutex queue_mtx_;
    std::condition_variable worker_cv_;
    std::deque<std::shared_ptr<RequestContext>> pending_queue_;
    std::atomic<uint64_t> next_request_id_{1};

    // Lock-free telemetry atomics
    std::atomic<int> stat_active_slots_{0};
    std::atomic<int> stat_queued_requests_{0};
    std::atomic<uint64_t> stat_completed_{0};
    std::atomic<uint64_t> stat_batched_steps_{0};
    std::atomic<uint64_t> stat_tokens_generated_{0};
    std::atomic<uint64_t> stat_slot_steps_sum_{0};
    std::atomic<uint64_t> stat_last_step_us_{0};
};

} // namespace haven
