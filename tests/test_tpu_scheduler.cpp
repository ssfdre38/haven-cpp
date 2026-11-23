#include "haven/haven_tpu_scheduler.h"
#include <atomic>
#include <cassert>
#include <chrono>
#include <iostream>
#include <mutex>
#include <set>
#include <thread>
#include <vector>

using namespace haven;

namespace {

// Deterministic mock TPU backend verifying:
// 1. Slot isolation during prefill_slot()
// 2. Per-slot independent pos_vec[B] progression during decode_step_batched()
// 3. Full logits & on-chip argmax paths
class MockTpuBatchBackend final : public ITpuBatchBackend {
public:
    MockTpuBatchBackend(int B, int max_seq, int vocab, bool full_logits)
        : B_(B), max_seq_(max_seq), vocab_(vocab), full_logits_(full_logits),
          slot_last_pos_(static_cast<size_t>(B), -1),
          slot_tag_(static_cast<size_t>(B), 0) {}

    int batch_capacity() const override { return B_; }
    int max_seq_len() const override { return max_seq_; }
    int vocab_size() const override { return vocab_; }
    bool returns_full_logits() const override { return full_logits_; }
    std::string last_error() const override { return ""; }

    bool prefill_slot(int slot_id,
                      const std::vector<uint32_t>& tokens,
                      int start_pos,
                      std::vector<float>* out_logits,
                      uint32_t* out_next_token) override {
        if (slot_id < 0 || slot_id >= B_ || tokens.empty()) return false;
        prefill_calls_.fetch_add(1, std::memory_order_relaxed);
        slot_tag_[static_cast<size_t>(slot_id)] = tokens[0];
        slot_last_pos_[static_cast<size_t>(slot_id)] = start_pos + static_cast<int>(tokens.size()) - 1;

        // Encode (tag + pos) deterministically in output token >= 1000 so it never collides with EOS
        uint32_t next_tok = 1000u + (tokens[0] * 100u) + static_cast<uint32_t>(tokens.size());
        if (next_tok >= static_cast<uint32_t>(vocab_)) next_tok = 1000u + (next_tok % static_cast<uint32_t>(vocab_ - 1000));

        if (full_logits_ && out_logits) {
            out_logits->assign(static_cast<size_t>(vocab_), -10.0f);
            (*out_logits)[next_tok] = 25.0f;
        }
        if (out_next_token) *out_next_token = next_tok;
        return true;
    }

    bool decode_step_batched(const std::vector<uint32_t>& tokens_per_batch,
                             const std::vector<int>& pos_per_slot,
                             std::vector<float>* out_logits,
                             std::vector<uint32_t>& out_next_tokens) override {
        if (static_cast<int>(tokens_per_batch.size()) != B_ ||
            static_cast<int>(pos_per_slot.size()) != B_) {
            return false;
        }
        decode_calls_.fetch_add(1, std::memory_order_relaxed);

        // Verify per-slot async positions: check if distinct non-zero positions coexist in one step!
        std::set<int> distinct_pos;
        for (int b = 0; b < B_; ++b) {
            if (pos_per_slot[static_cast<size_t>(b)] > 0) {
                distinct_pos.insert(pos_per_slot[static_cast<size_t>(b)]);
            }
        }
        if (distinct_pos.size() >= 2) {
            saw_async_positions_.store(true, std::memory_order_relaxed);
        }

        out_next_tokens.assign(static_cast<size_t>(B_), 0);
        if (full_logits_ && out_logits) {
            out_logits->assign(static_cast<size_t>(B_) * static_cast<size_t>(vocab_), -10.0f);
        }
        for (int b = 0; b < B_; ++b) {
            uint32_t tag = slot_tag_[static_cast<size_t>(b)];
            int pos = pos_per_slot[static_cast<size_t>(b)];
            // Expected next token encodes the slot's tag AND exact pos!
            uint32_t nt = 1000u + (tag * 100u) + static_cast<uint32_t>(pos + 1);
            if (nt >= static_cast<uint32_t>(vocab_)) nt = 1000u + (nt % static_cast<uint32_t>(vocab_ - 1000));
            out_next_tokens[static_cast<size_t>(b)] = nt;
            if (full_logits_ && out_logits) {
                (*out_logits)[static_cast<size_t>(b) * static_cast<size_t>(vocab_) + nt] = 25.0f;
            }
        }
        // Tiny simulated silicon tick (200 us) so threads overlap realistically
        std::this_thread::sleep_for(std::chrono::microseconds(200));
        return true;
    }

    int prefill_calls() const { return prefill_calls_.load(); }
    int decode_calls() const { return decode_calls_.load(); }
    bool saw_async_positions() const { return saw_async_positions_.load(); }

private:
    int B_;
    int max_seq_;
    int vocab_;
    bool full_logits_;
    std::vector<int> slot_last_pos_;
    std::vector<uint32_t> slot_tag_;
    std::atomic<int> prefill_calls_{0};
    std::atomic<int> decode_calls_{0};
    std::atomic<bool> saw_async_positions_{false};
};

} // namespace

int main() {
    std::cout << "======================================================================\n";
    std::cout << "  HAVEN-CPP TPU CONTINUOUS BATCH SCHEDULER UNIT TEST SUITE\n";
    std::cout << "======================================================================\n";

    // Test 1: 32 concurrent clients over B_cap = 16 slots with DIFFERENT prompt lengths & greedy verification
    {
        auto mock_raw = std::make_unique<MockTpuBatchBackend>(16, 512, 4096, true);
        auto* mock = mock_raw.get();

        SamplerParams sp;
        sp.temperature = 0.0f;
        sp.repetition_penalty = 1.0f;
        PersonaSampler template_sampler(sp);

        TpuContinuousBatchScheduler sched;
        if (!sched.start_with_backend(std::move(mock_raw), nullptr, template_sampler)) {
            std::cerr << "FAIL: scheduler failed to start\n";
            return 1;
        }

        constexpr int kClients = 32;
        constexpr int kGenTokens = 20;
        std::atomic<int> ok_count{0};
        std::atomic<int> mismatch_count{0};
        std::vector<std::thread> threads;

        for (int c = 0; c < kClients; ++c) {
            threads.emplace_back([&, c]() {
                TpuGenerationRequest req;
                // Distinct prompt lengths (5..24) so active slots hold strictly different positions!
                const int plen = 5 + (c % 20);
                req.prompt_tokens.assign(static_cast<size_t>(plen), static_cast<uint32_t>(c + 1));
                req.max_tokens = kGenTokens;
                req.use_custom_sampler = true;
                req.sampler_params = sp;

                std::vector<uint32_t> received;
                std::string err;
                bool ok = sched.generate_streaming(req, [&](uint32_t tok) {
                    received.push_back(tok);
                    return true;
                }, nullptr, &err);

                if (!ok || static_cast<int>(received.size()) != kGenTokens) {
                    mismatch_count.fetch_add(1);
                    return;
                }
                // Verify EVERY SINGLE token matches this client's exact tag and exact per-slot position!
                for (int step = 0; step < kGenTokens; ++step) {
                    uint32_t expected = 1000u + static_cast<uint32_t>(c + 1) * 100u + static_cast<uint32_t>(plen + step);
                    if (expected >= 4096u) expected = 1000u + (expected % (4096u - 1000u));
                    if (received[static_cast<size_t>(step)] != expected) {
                        mismatch_count.fetch_add(1);
                        return;
                    }
                }
                ok_count.fetch_add(1);
            });
        }

        for (auto& th : threads) th.join();
        auto stats = sched.get_stats();
        const bool saw_async = mock->saw_async_positions();
        sched.stop();

        std::cout << "[Test 1: 32 Clients / B=16 Async Pos] ok=" << ok_count.load() << "/" << kClients
                  << " | mismatches=" << mismatch_count.load()
                  << " | saw_async_pos=" << (saw_async ? "YES" : "NO")
                  << " | batched_steps=" << stats.total_batched_steps
                  << " | avg_batch=" << stats.avg_batch_utilization
                  << " | total_toks=" << stats.total_tokens_generated << "\n";

        if (ok_count.load() != kClients || mismatch_count.load() != 0 || !saw_async ||
            stats.total_tokens_generated != static_cast<uint64_t>(kClients * kGenTokens)) {
            std::cerr << "FAIL: Test 1 assertions failed!\n";
            return 1;
        }
    }

    // Test 2: Early Client Cancellation & Immediate Slot Reclamation (On-Chip Argmax Mode)
    {
        auto mock_raw = std::make_unique<MockTpuBatchBackend>(8, 512, 4096, false);
        PersonaSampler template_sampler;
        TpuContinuousBatchScheduler sched;
        if (!sched.start_with_backend(std::move(mock_raw), nullptr, template_sampler)) {
            std::cerr << "FAIL: Test 2 scheduler start failed\n";
            return 1;
        }

        std::atomic<int> cancelled_ok{0};
        std::vector<std::thread> threads;
        for (int c = 0; c < 16; ++c) {
            threads.emplace_back([&, c]() {
                TpuGenerationRequest req;
                req.prompt_tokens = {static_cast<uint32_t>(c + 1), 2, 3, 4};
                req.max_tokens = 200;
                int count = 0;
                bool ok = sched.generate_streaming(req, [&](uint32_t) {
                    count++;
                    return (count < 5); // Cancel after 5 tokens!
                });
                if (ok && count == 5) cancelled_ok.fetch_add(1);
            });
        }
        for (auto& th : threads) th.join();
        auto stats = sched.get_stats();
        sched.stop();

        std::cout << "[Test 2: Client Cancellation / Argmax] ok=" << cancelled_ok.load() << "/16"
                  << " | active_after=" << stats.active_slots << "\n";
        if (cancelled_ok.load() != 16) {
            std::cerr << "FAIL: Test 2 cancellation failed!\n";
            return 1;
        }
    }

    std::cout << "ALL TPU SCHEDULER TESTS PASSED!\n";
    return 0;
}
