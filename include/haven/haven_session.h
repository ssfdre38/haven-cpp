#pragma once

#include "haven_engine.h"
#include "haven_intelligence.h"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace haven {

// ============================================================================
// Unified 4-Call Compute Session Contract (Zero-Bloat Backend Abstraction)
// ============================================================================
// Eliminates scattered `if (cuda_active) ... else if (tpu_active) ...` branches
// across CLI/server code. Every silicon backend implements the same minimal
// 4-call lifecycle: initialize -> prefill -> decode_step -> reset_kv/shutdown.
// ============================================================================

struct SessionConfig {
    int  device_id = 0;
    int  max_seq_len = 4096;
    int  max_prefill_chunk = 512;
    bool use_cuda_graph = true;
    bool exact_math = false;            // Enable exact FP32-activation math across CPU & CUDA (0.999999+ parity)
    bool fp32_kv_cache = false;         // Enable FP32 KV cache on CUDA (auto-enabled when exact_math == true)
    bool fp32_prefill = false;          // Enable FP32 cuBLAS prefill on CUDA (auto-enabled when exact_math == true)
    bool host_resident_pleias = false;  // Keep Gemma 4 Pleias token table on host RAM to save ~2.1 GiB VRAM
    bool verbose = false;
};

struct SessionMetrics {
    BackendKind backend = BackendKind::CPU_AVX2;
    std::string device_summary;
    size_t      weight_bytes = 0;
    size_t      kv_bytes = 0;
    double      init_ms = 0.0;
    double      last_prefill_ms = 0.0;
    double      last_decode_ms = 0.0;
};

class IHavenSession {
public:
    virtual ~IHavenSession() = default;

    virtual BackendKind backend_kind() const = 0;
    virtual std::string backend_name() const = 0;
    virtual bool is_initialized() const = 0;

    // 1. Bind & initialize compute backend against loaded host engine
    virtual bool initialize(HavenEngine& host, const SessionConfig& cfg = {}) = 0;

    // 2. Multi-token prompt prefill starting at `start_pos`.
    //    Populates KV cache and optionally returns final-token logits and/or greedy argmax token.
    virtual bool prefill(const uint32_t* tokens,
                         size_t count,
                         int start_pos,
                         float* out_last_logits = nullptr,
                         uint32_t* out_next_token = nullptr) = 0;

    // 3. Single-token decode step at position `pos`.
    virtual bool decode_step(uint32_t token,
                             int pos,
                             float* out_logits = nullptr,
                             uint32_t* out_next_token = nullptr) = 0;

    // 4. Lifecycle: KV reset & graceful shutdown
    virtual bool reset_kv_cache() = 0;
    virtual void shutdown() = 0;

    virtual SessionMetrics get_metrics() const = 0;
    virtual std::string get_last_error() const = 0;
};

// Factory bridging HavenModelIntelligence::ExecutionPlan -> active IHavenSession
class HavenSessionFactory {
public:
    static std::unique_ptr<IHavenSession> create_cpu_session();
    static std::unique_ptr<IHavenSession> create_cuda_session();

    // Instantiates and initializes the optimal session chosen by `plan`, with
    // automatic safe fallback to native AVX2 CPU if an accelerator fails init.
    static std::unique_ptr<IHavenSession> create_for_plan(HavenEngine& host,
                                                          const ExecutionPlan& plan,
                                                          const SessionConfig& cfg,
                                                          std::string& out_log);
};

} // namespace haven
