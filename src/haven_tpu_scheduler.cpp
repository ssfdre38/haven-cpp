#include "haven/haven_tpu_scheduler.h"
#include <algorithm>
#include <chrono>
#include <iostream>

namespace haven {

namespace {

class RealTpuEngineAdapter final : public ITpuBatchBackend {
public:
    explicit RealTpuEngineAdapter(HavenTpuGemma4Engine* engine) : engine_(engine) {}

    int batch_capacity() const override {
        return engine_ ? engine_->get_config().batch_size : 0;
    }
    int max_seq_len() const override {
        return engine_ ? engine_->get_config().max_seq_len : 0;
    }
    int vocab_size() const override {
        return engine_ ? engine_->get_model_spec().vocab_size : 0;
    }
    bool returns_full_logits() const override {
        return engine_ ? engine_->get_config().return_full_logits : false;
    }
    std::string last_error() const override {
        return engine_ ? engine_->get_last_error() : "TPU engine is null";
    }
    bool prefill_slot(int slot_id,
                      const std::vector<uint32_t>& tokens,
                      int start_pos,
                      std::vector<float>* out_logits,
                      uint32_t* out_next_token) override {
        return engine_ && engine_->prefill_slot(slot_id, tokens, start_pos, out_logits, out_next_token);
    }
    bool decode_step_batched(const std::vector<uint32_t>& tokens_per_batch,
                             const std::vector<int>& pos_per_slot,
                             std::vector<float>* out_logits,
                             std::vector<uint32_t>& out_next_tokens) override {
        return engine_ && engine_->decode_step_batched(tokens_per_batch, pos_per_slot, out_logits, out_next_tokens);
    }

private:
    HavenTpuGemma4Engine* engine_ = nullptr;
};

} // namespace

TpuContinuousBatchScheduler::TpuContinuousBatchScheduler() = default;

TpuContinuousBatchScheduler::~TpuContinuousBatchScheduler() {
    stop();
}

bool TpuContinuousBatchScheduler::start(HavenTpuGemma4Engine* engine,
                                        const HavenTokenizer* tokenizer,
                                        const PersonaSampler& template_sampler) {
    if (!engine || !engine->is_initialized() || engine->get_config().batch_size < 1) {
        return false;
    }
    return start_with_backend(std::make_unique<RealTpuEngineAdapter>(engine), tokenizer, template_sampler);
}

bool TpuContinuousBatchScheduler::start_with_backend(std::unique_ptr<ITpuBatchBackend> backend,
                                                     const HavenTokenizer* tokenizer,
                                                     const PersonaSampler& template_sampler) {
    stop();
    std::lock_guard<std::mutex> life_lk(lifecycle_mtx_);
    if (!backend || backend->batch_capacity() < 1 || backend->max_seq_len() < 2) {
        return false;
    }
    cached_batch_cap_.store(backend->batch_capacity(), std::memory_order_release);
    cached_max_seq_len_.store(backend->max_seq_len(), std::memory_order_release);
    backend_ = std::move(backend);
    tokenizer_ = tokenizer;
    template_sampler_ = template_sampler;

    stat_active_slots_.store(0, std::memory_order_relaxed);
    stat_queued_requests_.store(0, std::memory_order_relaxed);
    stat_completed_.store(0, std::memory_order_relaxed);
    stat_batched_steps_.store(0, std::memory_order_relaxed);
    stat_tokens_generated_.store(0, std::memory_order_relaxed);
    stat_slot_steps_sum_.store(0, std::memory_order_relaxed);
    stat_last_step_us_.store(0, std::memory_order_relaxed);

    {
        std::lock_guard<std::mutex> q_lk(queue_mtx_);
        running_.store(true, std::memory_order_release);
    }
    worker_thread_ = std::thread(&TpuContinuousBatchScheduler::worker_loop, this);
    return true;
}

void TpuContinuousBatchScheduler::stop() {
    std::lock_guard<std::mutex> life_lk(lifecycle_mtx_);
    bool was_running = false;
    {
        std::lock_guard<std::mutex> lk(queue_mtx_);
        was_running = running_.exchange(false, std::memory_order_acq_rel);
        worker_cv_.notify_all();
    }
    if (was_running || worker_thread_.joinable()) {
        if (worker_thread_.joinable()) {
            worker_thread_.join();
        }
    }
    {
        std::lock_guard<std::mutex> lk(queue_mtx_);
        while (!pending_queue_.empty()) {
            finish_request(pending_queue_.front(), true, "TPU scheduler stopped.");
            pending_queue_.pop_front();
        }
        stat_queued_requests_.store(0, std::memory_order_relaxed);
    }
    cached_batch_cap_.store(0, std::memory_order_release);
    backend_.reset();
    tokenizer_ = nullptr;
}

bool TpuContinuousBatchScheduler::is_stop_token_id(uint32_t tok) const {
    if (tokenizer_) {
        return tokenizer_->is_stop_token(tok) || tok == tokenizer_->eos_token();
    }
    return (tok == 1 || tok == 106 || tok == 212);
}

void TpuContinuousBatchScheduler::finish_request(const std::shared_ptr<RequestContext>& ctx,
                                                 bool failed,
                                                 const std::string& err) {
    if (!ctx) return;
    {
        std::lock_guard<std::mutex> lk(ctx->mtx);
        if (failed) {
            ctx->failed = true;
            ctx->error_msg = err;
        }
        ctx->done = true;
    }
    ctx->cv.notify_all();
}

void TpuContinuousBatchScheduler::push_token(const std::shared_ptr<RequestContext>& ctx, uint32_t tok) {
    if (!ctx) return;
    {
        std::lock_guard<std::mutex> lk(ctx->mtx);
        ctx->ready_tokens.push_back(tok);
    }
    ctx->cv.notify_one();
}

bool TpuContinuousBatchScheduler::generate_streaming(const TpuGenerationRequest& req,
                                                     const std::function<bool(uint32_t token_id)>& on_token,
                                                     const std::function<bool()>& is_cancelled_fn,
                                                     std::string* err_out) {
    if (req.prompt_tokens.empty()) {
        if (err_out) *err_out = "Prompt token sequence is empty.";
        return false;
    }

    auto ctx = std::make_shared<RequestContext>();
    ctx->request_id = next_request_id_.fetch_add(1, std::memory_order_relaxed);
    ctx->req = req;

    {
        std::lock_guard<std::mutex> lk(queue_mtx_);
        if (!running_.load(std::memory_order_acquire)) {
            if (err_out) *err_out = "TPU continuous batch scheduler is not running.";
            return false;
        }
        const int max_ctx = std::max(2, cached_max_seq_len_.load(std::memory_order_acquire));
        const size_t max_prompt = static_cast<size_t>(std::max(1, max_ctx - 1));
        if (ctx->req.prompt_tokens.size() > max_prompt) {
            ctx->req.prompt_tokens.erase(
                ctx->req.prompt_tokens.begin(),
                ctx->req.prompt_tokens.begin() + static_cast<std::ptrdiff_t>(ctx->req.prompt_tokens.size() - max_prompt));
        }
        ctx->req.max_tokens = std::clamp(ctx->req.max_tokens, 1, max_ctx);
        pending_queue_.push_back(ctx);
        stat_queued_requests_.store(static_cast<int>(pending_queue_.size()), std::memory_order_relaxed);
    }
    worker_cv_.notify_one();

    while (true) {
        uint32_t tok = 0;
        bool has_tok = false;
        bool is_done = false;
        bool is_failed = false;
        std::string fail_msg;

        {
            std::unique_lock<std::mutex> lk(ctx->mtx);
            if (ctx->ready_tokens.empty() && !ctx->done) {
                if (is_cancelled_fn) {
                    ctx->cv.wait_for(lk, std::chrono::milliseconds(50), [&]() {
                        return !ctx->ready_tokens.empty() || ctx->done;
                    });
                } else {
                    ctx->cv.wait(lk, [&]() {
                        return !ctx->ready_tokens.empty() || ctx->done;
                    });
                }
            }
            if (!ctx->ready_tokens.empty()) {
                tok = ctx->ready_tokens.front();
                ctx->ready_tokens.pop_front();
                has_tok = true;
            } else if (ctx->done) {
                is_done = true;
                is_failed = ctx->failed;
                fail_msg = ctx->error_msg;
            }
        }

        if (has_tok) {
            if (on_token && !on_token(tok)) {
                ctx->cancelled.store(true, std::memory_order_release);
                worker_cv_.notify_one();
                return true;
            }
            continue;
        }

        if (is_done) {
            if (is_failed && err_out) *err_out = fail_msg;
            return !is_failed;
        }

        if (is_cancelled_fn && is_cancelled_fn()) {
            ctx->cancelled.store(true, std::memory_order_release);
            worker_cv_.notify_one();
            return true;
        }
    }
}

void TpuContinuousBatchScheduler::worker_loop() {
    const int B_cap = backend_->batch_capacity();
    const int max_ctx = backend_->max_seq_len();
    const int V = backend_->vocab_size();
    const bool has_full_logits = backend_->returns_full_logits();

    std::vector<SlotState> slots(static_cast<size_t>(B_cap));
    std::vector<float> prefill_logits;
    std::vector<float> batch_logits;
    std::vector<uint32_t> step_tokens(static_cast<size_t>(B_cap), 0);
    std::vector<int> step_pos(static_cast<size_t>(B_cap), 0);
    std::vector<uint32_t> argmax_tokens;
    std::vector<int> active_indices;
    active_indices.reserve(static_cast<size_t>(B_cap));

    auto sample_slot_token = [&](SlotState& slot, float* logits_row, uint32_t argmax_tok) -> uint32_t {
        if (!has_full_logits || !logits_row) return argmax_tok;
        const auto& sp = slot.sampler.get_params();
        if (sp.temperature <= 0.05f && sp.repetition_penalty <= 1.001f) {
            return argmax_tok;
        }
        return slot.sampler.sample(logits_row, static_cast<uint32_t>(V), slot.context_history);
    };

    while (running_.load(std::memory_order_acquire)) {
        // 1. Reclaim any cancelled active slots immediately
        int active_count = 0;
        for (int s = 0; s < B_cap; ++s) {
            if (slots[static_cast<size_t>(s)].active) {
                if (slots[static_cast<size_t>(s)].ctx->cancelled.load(std::memory_order_acquire)) {
                    finish_request(slots[static_cast<size_t>(s)].ctx, false);
                    slots[static_cast<size_t>(s)].active = false;
                    slots[static_cast<size_t>(s)].ctx.reset();
                    stat_completed_.fetch_add(1, std::memory_order_relaxed);
                } else {
                    active_count++;
                }
            }
        }

        // 2. If completely idle and queue empty, wait on condition variable
        {
            std::unique_lock<std::mutex> lk(queue_mtx_);
            if (active_count == 0 && pending_queue_.empty()) {
                stat_active_slots_.store(0, std::memory_order_relaxed);
                stat_queued_requests_.store(0, std::memory_order_relaxed);
                worker_cv_.wait(lk, [&]() {
                    return !running_.load(std::memory_order_acquire) || !pending_queue_.empty();
                });
                if (!running_.load(std::memory_order_acquire)) break;
            }
        }

        // 3. Admit waiting requests into free KV slots (bounded per tick for smooth streaming jitter)
        const int max_admit = (active_count == 0) ? std::min(B_cap, 8) : std::max(4, B_cap / 4);
        int admitted = 0;
        for (int s = 0; s < B_cap && admitted < max_admit; ++s) {
            if (slots[static_cast<size_t>(s)].active) continue;

            std::shared_ptr<RequestContext> req_ctx;
            {
                std::lock_guard<std::mutex> lk(queue_mtx_);
                while (!pending_queue_.empty()) {
                    auto cand = pending_queue_.front();
                    pending_queue_.pop_front();
                    if (cand->cancelled.load(std::memory_order_acquire)) {
                        finish_request(cand, false);
                        continue;
                    }
                    req_ctx = std::move(cand);
                    break;
                }
                stat_queued_requests_.store(static_cast<int>(pending_queue_.size()), std::memory_order_relaxed);
            }
            if (!req_ctx) break;

            // Run isolated slot-targeted MXU prefill directly into KV slot `s`
            uint32_t first_argmax = 0;
            bool pf_ok = backend_->prefill_slot(
                s, req_ctx->req.prompt_tokens, 0,
                has_full_logits ? &prefill_logits : nullptr,
                &first_argmax);
            if (!pf_ok) {
                finish_request(req_ctx, true, "TPU slot prefill failed: " + backend_->last_error());
                continue;
            }

            auto& slot = slots[static_cast<size_t>(s)];
            slot.ctx = req_ctx;
            slot.sampler = template_sampler_;
            if (req_ctx->req.use_custom_sampler) {
                slot.sampler.get_params() = req_ctx->req.sampler_params;
            }
            slot.sampler.seed(static_cast<uint32_t>((req_ctx->request_id * 2654435761ULL) ^ 0x9e3779b9ULL));

            slot.context_history.clear();
            const auto& ptoks = req_ctx->req.prompt_tokens;
            size_t h_start = (ptoks.size() > 64) ? (ptoks.size() - 64) : 0;
            slot.context_history.insert(slot.context_history.end(), ptoks.begin() + static_cast<std::ptrdiff_t>(h_start), ptoks.end());

            uint32_t first_tok = sample_slot_token(
                slot,
                (has_full_logits && prefill_logits.size() >= static_cast<size_t>(V)) ? prefill_logits.data() : nullptr,
                first_argmax);

            slot.context_history.push_back(first_tok);
            slot.cur_pos = static_cast<int>(ptoks.size());
            slot.pending_tok = first_tok;
            slot.generated_count = 1;
            stat_tokens_generated_.fetch_add(1, std::memory_order_relaxed);
            push_token(req_ctx, first_tok);

            if (is_stop_token_id(first_tok) ||
                slot.generated_count >= req_ctx->req.max_tokens ||
                slot.cur_pos >= max_ctx - 1 ||
                req_ctx->cancelled.load(std::memory_order_acquire)) {
                finish_request(req_ctx, false);
                slot.active = false;
                slot.ctx.reset();
                stat_completed_.fetch_add(1, std::memory_order_relaxed);
            } else {
                slot.active = true;
                active_count++;
            }
            admitted++;
        }

        // 4. Pack all active slots into ONE lockstep async-position TPU decode_step_batched!
        active_indices.clear();
        for (int s = 0; s < B_cap; ++s) {
            const auto& slot = slots[static_cast<size_t>(s)];
            if (slot.active) {
                active_indices.push_back(s);
                step_tokens[static_cast<size_t>(s)] = slot.pending_tok;
                step_pos[static_cast<size_t>(s)] = slot.cur_pos;
            } else {
                step_tokens[static_cast<size_t>(s)] = 0;
                step_pos[static_cast<size_t>(s)] = 0;
            }
        }
        stat_active_slots_.store(static_cast<int>(active_indices.size()), std::memory_order_relaxed);
        if (active_indices.empty()) continue;

        auto t0 = std::chrono::high_resolution_clock::now();
        bool dec_ok = backend_->decode_step_batched(
            step_tokens, step_pos,
            has_full_logits ? &batch_logits : nullptr,
            argmax_tokens);
        auto t1 = std::chrono::high_resolution_clock::now();
        uint64_t step_us = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());
        stat_last_step_us_.store(step_us, std::memory_order_relaxed);

        if (!dec_ok) {
            std::string err = "TPU batched decode failed: " + backend_->last_error();
            for (int s : active_indices) {
                finish_request(slots[static_cast<size_t>(s)].ctx, true, err);
                slots[static_cast<size_t>(s)].active = false;
                slots[static_cast<size_t>(s)].ctx.reset();
            }
            continue;
        }

        stat_batched_steps_.fetch_add(1, std::memory_order_relaxed);
        stat_slot_steps_sum_.fetch_add(static_cast<uint64_t>(active_indices.size()), std::memory_order_relaxed);

        // 5. Parallel per-slot sampling across all active streams (each slot owns disjoint logit row & RNG!)
        std::vector<uint32_t> sampled(active_indices.size(), 0);
        const int n_act = static_cast<int>(active_indices.size());
        #pragma omp parallel for schedule(static) if (n_act >= 4 && has_full_logits)
        for (int i = 0; i < n_act; ++i) {
            const int s = active_indices[static_cast<size_t>(i)];
            float* row = (has_full_logits && batch_logits.size() >= static_cast<size_t>(B_cap) * static_cast<size_t>(V))
                ? (batch_logits.data() + static_cast<size_t>(s) * static_cast<size_t>(V))
                : nullptr;
            uint32_t argm = (static_cast<size_t>(s) < argmax_tokens.size()) ? argmax_tokens[static_cast<size_t>(s)] : 0;
            sampled[static_cast<size_t>(i)] = sample_slot_token(slots[static_cast<size_t>(s)], row, argm);
        }

        // 6. Emit tokens to HTTP threads & evict finished/cancelled slots
        for (int i = 0; i < n_act; ++i) {
            const int s = active_indices[static_cast<size_t>(i)];
            auto& slot = slots[static_cast<size_t>(s)];
            const uint32_t tok = sampled[static_cast<size_t>(i)];

            slot.cur_pos++;
            slot.generated_count++;
            slot.pending_tok = tok;
            slot.context_history.push_back(tok);
            if (slot.context_history.size() > 256) {
                slot.context_history.erase(slot.context_history.begin(), slot.context_history.begin() + 128);
            }
            stat_tokens_generated_.fetch_add(1, std::memory_order_relaxed);
            push_token(slot.ctx, tok);

            if (is_stop_token_id(tok) ||
                slot.generated_count >= slot.ctx->req.max_tokens ||
                slot.cur_pos >= max_ctx - 1 ||
                slot.ctx->cancelled.load(std::memory_order_acquire)) {
                finish_request(slot.ctx, false);
                slot.active = false;
                slot.ctx.reset();
                stat_completed_.fetch_add(1, std::memory_order_relaxed);
            }
        }
    }

    // Drain any in-flight active slots cleanly on shutdown
    for (int s = 0; s < B_cap; ++s) {
        if (slots[static_cast<size_t>(s)].active && slots[static_cast<size_t>(s)].ctx) {
            finish_request(slots[static_cast<size_t>(s)].ctx, true, "TPU scheduler stopped.");
            slots[static_cast<size_t>(s)].active = false;
            slots[static_cast<size_t>(s)].ctx.reset();
        }
    }
    stat_active_slots_.store(0, std::memory_order_relaxed);
}

TpuSchedulerStats TpuContinuousBatchScheduler::get_stats() const {
    TpuSchedulerStats st;
    st.batch_capacity = cached_batch_cap_.load(std::memory_order_acquire);
    st.active_slots = stat_active_slots_.load(std::memory_order_relaxed);
    st.queued_requests = stat_queued_requests_.load(std::memory_order_relaxed);
    st.total_requests_completed = stat_completed_.load(std::memory_order_relaxed);
    st.total_batched_steps = stat_batched_steps_.load(std::memory_order_relaxed);
    st.total_tokens_generated = stat_tokens_generated_.load(std::memory_order_relaxed);
    uint64_t slot_sum = stat_slot_steps_sum_.load(std::memory_order_relaxed);
    if (st.total_batched_steps > 0 && st.batch_capacity > 0) {
        st.avg_batch_utilization = static_cast<double>(slot_sum) / static_cast<double>(st.total_batched_steps);
    }
    st.last_step_ms = static_cast<double>(stat_last_step_us_.load(std::memory_order_relaxed)) / 1000.0;
    return st;
}

} // namespace haven
