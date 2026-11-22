#pragma once

#include "haven_engine.h"
#include "haven_tpu_pjrt.h"
#include <cstdint>
#include <string>
#include <vector>

namespace haven {

struct TpuGemma4Config {
    int batch_size = 1;            // Concurrent sequences (1 = interactive, 16/32/64 = continuous batching)
    int max_seq_len = 2048;        // Static on-chip HBM3 KV cache capacity per sequence
    int prefill_chunk_size = 128;  // Batched prefill chunk length (0 = skip prefill graph compilation)
    int num_replicas = 1;          // 1 = single Trillium chip, 2 or 4 = Megatron Tensor-Parallel over Optical ICI
    bool use_int8_weights = false; // false = BF16 (~9.3 GB total), true = W8A16 INT8 (~4.6 GB total, ~2x HBM3 BW)
    bool return_full_logits = false; // true = return [B, vocab] F32 logits, false = on-chip argmax [B] i32 (4B PCIe)
    bool donate_kv_cache = true;   // Emit tf.aliasing_output on KV cache buffers for zero-copy in-place HBM3 updates
};

struct TpuGemma4Metrics {
    double weight_upload_ms = 0.0;
    double decode_compile_ms = 0.0;
    double prefill_compile_ms = 0.0;
    size_t total_hbm_weight_bytes = 0;
    size_t total_hbm_kv_bytes = 0;
    double last_h2d_ms = 0.0;
    double last_exec_ms = 0.0;
    double last_d2h_ms = 0.0;
};

// Full On-Chip 42-Layer Gemma 4 StableHLO Compiler & Persistent HBM3 Runtime.
// Compiles the entire transformer (Pre/Post RMSNorm, GQA Q/K/V/O, Q/K/V per-head norms,
// SWA-512 + 4-Sink & Global RoPE attention, shared-KV layers 24..41, GeGLU FFN,
// Pleias per-layer injection, layer output scales, 262k LM head, logit softcapping,
// and optional 4-chip optical ICI AllReduce tensor parallelism + on-chip argmax)
// into a single persistent PJRT executable with zero per-layer host round-trips.
class HavenTpuGemma4Engine {
public:
    HavenTpuGemma4Engine();
    ~HavenTpuGemma4Engine();

    HavenTpuGemma4Engine(const HavenTpuGemma4Engine&) = delete;
    HavenTpuGemma4Engine& operator=(const HavenTpuGemma4Engine&) = delete;

    // Emit textual StableHLO MLIR for the complete Gemma 4 transformer graph.
    // Exposed publicly for offline unit testing and graph inspection on any host.
    static std::string emit_stablehlo_mlir(const ModelConfig& model_cfg,
                                           const TpuGemma4Config& tpu_cfg,
                                           int seq_chunk_len);

    // Upload sharded/quantized weights to HBM3 across num_replicas chips and compile graphs.
    bool initialize(HavenTpuPjrtBackend* backend,
                    const HavenEngine& host_engine,
                    const TpuGemma4Config& cfg);
    void shutdown();

    bool is_initialized() const { return initialized_; }
    const TpuGemma4Config& get_config() const { return cfg_; }
    const TpuGemma4Metrics& get_metrics() const { return metrics_; }
    std::string get_last_error() const { return last_error_; }

    // Zero all on-chip HBM3 KV cache tensors across all replicas.
    bool reset_kv_cache();

    // Run batched prompt prefill (chunked by prefill_chunk_size + decode tail).
    // Populates on-chip HBM3 KV cache and outputs logits (if return_full_logits) and argmax token for batch 0.
    bool prefill_prompt(const std::vector<uint32_t>& tokens,
                        int start_pos,
                        std::vector<float>* out_logits,
                        uint32_t* out_next_token);

    // Single on-chip forward step across all 42 layers for `batch_size` concurrent sequences at `pos`.
    // Zero per-layer PCIe bounces: uploads only ~26 KB embeddings/batch and reads back logits or 4B*B argmax IDs.
    bool decode_step(const std::vector<uint32_t>& tokens_per_batch,
                     int pos,
                     std::vector<float>* out_logits,
                     std::vector<uint32_t>& out_next_tokens);

private:
    bool initialized_ = false;
    HavenTpuPjrtBackend* backend_ = nullptr;
    const HavenEngine* host_engine_ = nullptr;
    ModelConfig model_cfg_{};
    TpuGemma4Config cfg_{};
    TpuGemma4Metrics metrics_{};
    std::string last_error_;

    // Compiled executables (1-chip portable or R-chip SPMD replicated)
    PJRT_LoadedExecutable* decode_exec_ = nullptr;
    PJRT_LoadedExecutable* prefill_exec_ = nullptr;
    std::vector<int> replica_dev_indices_; // Exact XLA torus device assignment per replica (e.g. {0,1,3,2})

    // Per-replica persistent HBM3 weight buffers (constant across all steps)
    std::vector<std::vector<PJRT_Buffer*>> replica_weight_bufs_;

    // Per-replica mutable HBM3 KV cache buffers: [k_swa, v_swa, k_glob, v_glob]
    std::vector<std::vector<PJRT_Buffer*>> replica_kv_bufs_;

    // Pre-allocated host scratch buffers for zero-malloc per-step embedding lookup
    std::vector<float> emb_f32_scratch_;
    std::vector<uint16_t> emb_bf16_scratch_;
    std::vector<float> pleias_f32_scratch_;
    std::vector<uint16_t> pleias_bf16_scratch_;

    bool build_and_upload_weights(const HavenEngine& host_engine);
    bool allocate_zero_kv_cache();
    void free_kv_cache();
    bool execute_graph(PJRT_LoadedExecutable* exec,
                       int seq_chunk_len,
                       const std::vector<uint32_t>& flat_tokens,
                       int start_pos,
                       std::vector<float>* out_logits,
                       std::vector<uint32_t>& out_next_tokens);
};

} // namespace haven
