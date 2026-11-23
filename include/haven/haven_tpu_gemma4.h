#pragma once

#include "haven_engine.h"
#include "haven_tpu_pjrt.h"
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace haven {

// Architecture-agnostic dense + sparse MoE transformer specification compiled directly
// into a single persistent StableHLO HBM3 graph (supports Gemma 4 E4B & 26B-A4B MoE / Ava,
// Gemma 2/3, Llama 3/4, Qwen 2.5/3, Mistral, Phi-3/4).
struct TpuModelSpec {
    std::string architecture;
    int num_layers = 0;
    int embedding_dim = 0;
    int hidden_dim = 0;
    int vocab_size = 0;
    int num_heads = 0;
    int num_kv_heads = 0;
    int num_kv_heads_swa = 0;
    int num_kv_heads_glob = 0;
    int head_dim_swa = 0;
    int head_dim_glob = 0;
    int pleias_dim = 256;
    int sliding_window = 512;
    float rms_norm_eps = 1e-6f;
    float attn_logit_softcapping = 0.0f;
    float final_logit_softcapping = 0.0f;
    float emb_scale = 1.0f;
    FfnActivation ffn_activation = FfnActivation::GELU;

    bool is_gemma = false;
    bool has_pleias = false;
    bool has_q_norm = false;
    bool has_k_norm = false;
    bool has_v_norm = false;
    bool has_swa_v = true;
    bool has_glob_v = true;
    bool has_post_attn_norm = false;
    bool has_post_ffn_norm = false;
    bool has_post_norm = false;
    bool has_layer_scale = false;
    bool has_qkv_bias = false;
    bool has_tied_embeddings = true;

    // Sparse Mixture-of-Experts (Gemma 4 26B-A4B / Ava 26B Turbo MoE)
    bool is_moe = false;
    int num_experts = 0;
    int num_experts_used = 0;
    int expert_ffn_dim = 0;
    bool has_moe_router_scale = false;
    bool has_post_ffn_norm_1 = false;
    bool has_pre_ffn_norm_2 = false;
    bool has_post_ffn_norm_2 = false;

    int num_swa_layers = 0;
    int num_glob_layers = 0;
    int num_swa_kv_layers = 0;
    int num_glob_kv_layers = 0;
    int last_swa_kv_idx = 0;
    int last_glob_kv_idx = 0;

    std::vector<bool> layer_is_swa;
    std::vector<bool> layer_has_kv;
    std::vector<int>  layer_type_idx;
    std::vector<int>  layer_kv_idx;

    int num_kv_buffers() const {
        return 2 * num_swa_kv_layers + 2 * num_glob_kv_layers;
    }

    // Derive exact TPU execution graph specification from a loaded HavenEngine
    // (or synthetically from ModelConfig when host_engine == nullptr for CI/IR inspection).
    static TpuModelSpec from_model_config(const ModelConfig& mc, const HavenEngine* host_engine = nullptr);
};

struct TpuGemma4Config {
    int batch_size = 1;            // Concurrent KV cache slot capacity (1 = interactive, 16/32/64/128 = continuous batching)
    int max_seq_len = 2048;        // Static on-chip HBM3 KV cache capacity per sequence slot
    int prefill_chunk_size = 128;  // Max batched prefill bucket (0 = skip prefill graph compilation)
    std::vector<int> prefill_buckets; // Explicit bucket list (empty = auto {128, 32, 16} when prefill_chunk_size >= 128)
    int num_replicas = 1;          // 1 = single Trillium chip, 2/4/8 = Megatron Tensor-Parallel over Optical ICI
    bool use_int8_weights = false; // false = BF16, true = W8A16 INT8 (~2x HBM3 BW)
    bool return_full_logits = false; // true = return [B, vocab] F32 logits, false = on-chip argmax [B] i32 (4B/slot PCIe)
    bool use_onchip_embeddings = false; // true = upload only [B, S] i32 token IDs & gather embeddings on-chip in HBM3
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

// Model-General Full On-Chip Dense Transformer StableHLO Compiler & Persistent HBM3 Runtime.
// Compiles the complete transformer (Gemma 4/3/2, Llama 3/4, Qwen 2.5/3, Mistral, Phi-3/4, Ava 26B)
// with per-slot [B] independent positions, slot-targeted multi-bucket MXU prefill (S=128/32/16/1),
// W8A16 INT8 or BF16 weights, 1/2/4-chip Optical ICI tensor parallelism, on-chip sharded embedding
// gather, and on-chip argmax into persistent PJRT executables with zero per-layer host round-trips.
class HavenTpuGemma4Engine {
public:
    HavenTpuGemma4Engine();
    ~HavenTpuGemma4Engine();

    HavenTpuGemma4Engine(const HavenTpuGemma4Engine&) = delete;
    HavenTpuGemma4Engine& operator=(const HavenTpuGemma4Engine&) = delete;

    // Check whether a loaded GGUF model is supported by the on-chip TPU StableHLO compiler.
    static bool is_model_supported(const HavenEngine& host_engine, std::string& reason_out);

    // Emit textual StableHLO MLIR for any dense transformer graph (decode or slot-targeted prefill).
    static std::string emit_stablehlo_mlir(const TpuModelSpec& spec,
                                           const TpuGemma4Config& tpu_cfg,
                                           int seq_chunk_len,
                                           bool slot_targeted = false);

    // Convenience overload deriving TpuModelSpec from ModelConfig (offline CI & inspection).
    static std::string emit_stablehlo_mlir(const ModelConfig& model_cfg,
                                           const TpuGemma4Config& tpu_cfg,
                                           int seq_chunk_len,
                                           bool slot_targeted = false);

    // Upload sharded/quantized weights to HBM3 across num_replicas chips and compile decode + prefill graphs.
    bool initialize(HavenTpuPjrtBackend* backend,
                    const HavenEngine& host_engine,
                    const TpuGemma4Config& cfg);
    void shutdown();

    bool is_initialized() const { return initialized_; }
    const TpuGemma4Config& get_config() const { return cfg_; }
    const TpuModelSpec& get_model_spec() const { return spec_; }
    const TpuGemma4Metrics& get_metrics() const { return metrics_; }
    std::string get_last_error() const { return last_error_; }

    // Zero all on-chip HBM3 KV cache tensors across all replicas.
    bool reset_kv_cache();

    // Slot-targeted multi-bucket MXU prefill (chunks through e.g. 128 -> 32 -> 16 -> 1)
    // directly into KV slot `slot_id` (0 <= slot_id < batch_size) WITHOUT modifying any other active slot!
    bool prefill_slot(int slot_id,
                      const std::vector<uint32_t>& tokens,
                      int start_pos,
                      std::vector<float>* out_logits,
                      uint32_t* out_next_token);

    // Run prompt prefill (delegates to prefill_slot(0, ...) for B=1 or when prefill buckets are compiled;
    // broadcasts across all B slots via decode_step when B>1 and prefill_chunk_size == 0).
    bool prefill_prompt(const std::vector<uint32_t>& tokens,
                        int start_pos,
                        std::vector<float>* out_logits,
                        uint32_t* out_next_token);

    // Single on-chip forward step across all layers for `batch_size` concurrent sequences at
    // INDEPENDENT per-slot positions `pos_per_slot[0..batch_size-1]`.
    bool decode_step_batched(const std::vector<uint32_t>& tokens_per_batch,
                             const std::vector<int>& pos_per_slot,
                             std::vector<float>* out_logits,
                             std::vector<uint32_t>& out_next_tokens);

    // Single on-chip forward step with uniform position `pos` across all batch slots.
    bool decode_step(const std::vector<uint32_t>& tokens_per_batch,
                     int pos,
                     std::vector<float>* out_logits,
                     std::vector<uint32_t>& out_next_tokens);

private:
    bool initialized_ = false;
    HavenTpuPjrtBackend* backend_ = nullptr;
    const HavenEngine* host_engine_ = nullptr;
    ModelConfig model_cfg_{};
    TpuModelSpec spec_{};
    TpuGemma4Config cfg_{};
    TpuGemma4Metrics metrics_{};
    std::string last_error_;

    // Compiled executables (1-chip portable or R-chip SPMD replicated)
    PJRT_LoadedExecutable* decode_exec_ = nullptr;
    std::vector<std::pair<int, PJRT_LoadedExecutable*>> slot_prefill_execs_; // Sorted descending by S (e.g. 128, 32, 16, 1)
    std::vector<int> replica_dev_indices_; // Exact XLA torus device assignment per replica (e.g. {0,1,3,2})

    // Per-replica persistent HBM3 weight buffers (constant across all steps)
    std::vector<std::vector<PJRT_Buffer*>> replica_weight_bufs_;

    // Per-replica mutable donated 4D HBM3 KV cache buffers:
    // [ks_0, vs_0, ..., ks_{N_swa_kv-1}, vs_{N_swa_kv-1}, kg_0, vg_0, ..., kg_{N_glob_kv-1}, vg_{N_glob_kv-1}]
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
                       bool slot_targeted,
                       int slot_id,
                       const std::vector<uint32_t>& flat_tokens,
                       const std::vector<int>& pos_per_slot,
                       std::vector<float>* out_logits,
                       std::vector<uint32_t>& out_next_tokens);
};

} // namespace haven
