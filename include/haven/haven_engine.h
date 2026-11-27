#pragma once

#include "haven_types.h"
#include "haven_gguf.h"
#include "haven_avx2.h"
#include "haven_memory.h"
#include "haven_sampler.h"
#include "haven_kvcache.h"
#include "haven_knowledge_store.h"
#include "haven_attention_telemetry.h"
#include "haven_sensory.h"
#include "haven_paged_attention.h"
#include "haven_yarn.h"
#include "haven_speculative.h"
#include "haven_flash_attention.h"
#include "haven_chunked_prefill.h"
#include "haven_tokenizer.h"
#include "haven_plugin_manager.h"
#include "haven_mcp.h"
#include "haven_topology.h"
#include "haven_pathways.h"
#include "haven_tpu_pjrt.h"
#include <string>
#include <vector>
#include <functional>
#include <memory>
#include <future>

namespace haven {

class HavenEngine {
public:
    struct LayerTensors {
        const TensorDesc* attn_norm = nullptr;
        const TensorDesc* attn_q = nullptr;
        const TensorDesc* attn_k = nullptr;
        const TensorDesc* attn_v = nullptr;
        const TensorDesc* attn_q_bias = nullptr; // Qwen 2 / 2.5 bias
        const TensorDesc* attn_k_bias = nullptr; // Qwen 2 / 2.5 bias
        const TensorDesc* attn_v_bias = nullptr; // Qwen 2 / 2.5 bias
        const TensorDesc* attn_q_norm = nullptr;
        const TensorDesc* attn_k_norm = nullptr;
        const TensorDesc* attn_output = nullptr;
        const TensorDesc* post_attention_norm = nullptr;
        const TensorDesc* ffn_norm = nullptr;
        const TensorDesc* ffn_gate = nullptr;
        const TensorDesc* ffn_up = nullptr;
        const TensorDesc* ffn_down = nullptr;
        const TensorDesc* post_ffw_norm = nullptr;
        const TensorDesc* post_ffw_norm_1 = nullptr;      // Gemma 4 26B-A4B shared MLP post-norm
        const TensorDesc* pre_ffw_norm_2 = nullptr;       // Gemma 4 26B-A4B MoE branch pre-norm
        const TensorDesc* post_ffw_norm_2 = nullptr;      // Gemma 4 26B-A4B MoE branch post-norm
        const TensorDesc* ffn_gate_inp = nullptr;         // MoE router projection [D, E]
        const TensorDesc* ffn_gate_inp_scale = nullptr;   // Gemma 4 MoE router input scale [D]
        const TensorDesc* ffn_gate_up_exps = nullptr;     // Fused MoE gate+up experts [D, 2*F_exp, E]
        const TensorDesc* ffn_gate_exps = nullptr;        // Separate MoE gate experts [D, F_exp, E]
        const TensorDesc* ffn_up_exps = nullptr;          // Separate MoE up experts [D, F_exp, E]
        const TensorDesc* ffn_down_exps = nullptr;        // MoE down experts [F_exp, D, E]
        const TensorDesc* ffn_down_exps_scale = nullptr;  // Per-expert down output scale [E]
        const TensorDesc* inp_gate = nullptr;
        const TensorDesc* proj = nullptr;
        const TensorDesc* post_norm = nullptr;
        const TensorDesc* layer_output_scale = nullptr;
    };

    HavenEngine();
    ~HavenEngine();
    HavenEngine(const HavenEngine&) = delete;
    HavenEngine& operator=(const HavenEngine&) = delete;

    bool load_model(const std::string& gguf_filepath);
    
    // Injects episodic memory anchors directly into DMA memory attention cache
    void inject_memory(const std::string& concept_name, float weight, const std::vector<float>& embedding);
    void clear_memories();

    // Ingests screen or camera vision frame embeddings into attention
    void inject_vision_frame(const std::vector<float>& patch_embeddings, uint32_t width, uint32_t height, float salience = 1.0f);

    // Ingests real-time audio mel-spectrogram frame into attention
    void inject_audio_spectrum(const std::vector<float>& mel_bands, float energy_rms, float centroid, bool is_speech);

    // Clears buffered sensory streams
    void clear_sensory_buffers();

    // Ingests structured relational knowledge with composite key boosts
    void add_knowledge_relation(uint32_t token_id, uint32_t cluster_id, float boost, const std::string& label, const std::string& target);

    // Sets soul persona vector for logit steering
    void set_persona_embedding(const std::vector<float>& persona_vector);

    // Sets dynamic YaRN context scale (e.g. 4.0x -> 32k context)
    void set_yarn_context_scale(float scale);

    // Drafts candidate tokens using Multi-Token Speculation
    SpeculativeCandidate draft_speculative_tokens(const std::vector<uint32_t>& context);

    // Submits long sequence for asynchronous chunked prefill
    uint64_t submit_chunked_prefill(const std::vector<uint32_t>& tokens);

    // Forks a sequence KV cache table with Copy-on-Write
    uint64_t fork_sequence(uint64_t parent_seq_id, uint64_t child_seq_id);

    // Executes forward pass for a single token with dynamic KV cache, knowledge injection, and telemetry
    void forward(uint32_t token, int pos, float* out_logits, uint32_t active_cluster = 0);

    // Executes batched multi-token prompt prefill via L2-tiled 1x8/1x4 AVX2 GEMM and expert-grouped MoE.
    // Populates KV cache for positions [start_pos .. start_pos + n_tokens - 1] and optionally writes
    // the final token's logits into out_last_logits.
    void forward_batch(const uint32_t* tokens, int n_tokens, int start_pos, float* out_last_logits = nullptr, uint32_t active_cluster = 0);

    // Prunes stale cache entries across all layers and returns total pruned tokens
    size_t prune_kv_cache();

    // Generates completion stream for a sequence of prompt tokens
    void generate(
        const std::vector<uint32_t>& prompt_tokens,
        int max_new_tokens,
        const std::function<bool(uint32_t token, const std::string& text)>& on_token_callback
    );

    // Asynchronously generates tokens using Pathways non-blocking dataflow
    std::future<std::vector<uint32_t>> generate_async(
        const std::vector<uint32_t>& prompt_tokens,
        int max_new_tokens,
        const std::function<bool(uint32_t token, const std::string& text)>& on_token_callback = nullptr
    );

    std::shared_ptr<TopologyManager> get_topology() { return topology_; }
    std::shared_ptr<const TopologyManager> get_topology() const { return topology_; }
    std::shared_ptr<PathwaysEngine> get_pathways() { return pathways_; }
    std::shared_ptr<const PathwaysEngine> get_pathways() const { return pathways_; }

    // Real Cloud TPU HBM3 + MXU offload via native C++20 PJRT (libtpu.so)
    bool enable_tpu_offload(HavenTpuPjrtBackend* external_backend = nullptr);
    void disable_tpu_offload();
    bool is_tpu_offload_active() const { return tpu_offload_active_; }
    size_t get_tpu_resident_bytes() const { return tpu_resident_bytes_; }

    const ModelConfig& get_config() const { return loader_.get_config(); }
    const TensorDesc* get_token_embd_tensor() const { return token_embd_tensor_; }
    const TensorDesc* get_per_layer_token_embd_tensor() const { return per_layer_token_embd_tensor_; }
    const TensorDesc* get_per_layer_model_proj_tensor() const { return per_layer_model_proj_tensor_; }
    const TensorDesc* get_per_layer_proj_norm_tensor() const { return per_layer_proj_norm_tensor_; }
    const TensorDesc* get_rope_freqs_tensor() const { return rope_freqs_tensor_; }
    const TensorDesc* get_output_norm_tensor() const { return output_norm_tensor_; }
    const TensorDesc* get_output_tensor() const { return output_tensor_; }
    const LayerTensors& get_layer_tensors(uint32_t l) const { return layers_[l]; }
    size_t get_memory_count() const { return memory_engine_.get_memory_count(); }
    size_t get_knowledge_count() const { return knowledge_store_.get_relation_count(); }
    size_t get_vision_frame_count() const { return sensory_bridge_.get_visual_frame_count(); }
    size_t get_audio_frame_count() const { return sensory_bridge_.get_audio_frame_count(); }
    float get_current_attention_entropy() const { return telemetry_engine_.get_global_attention_entropy(); }
    float get_speculative_acceptance_rate() const { return speculative_engine_.get_acceptance_rate(); }
    void set_speculative_enabled(bool enabled) { speculative_enabled_ = enabled; }
    bool is_speculative_enabled() const { return speculative_enabled_; }
    // Exact FP32-Activation Math Mode: bypasses 8-bit Q8Act32 activation quantization across all projections
    // while keeping weights compressed in native GGUF blocks (targets 0.999999+ / 1.0000 parity).
    void set_exact_math(bool enabled) { exact_math_mode_ = enabled; }
    bool is_exact_math() const { return exact_math_mode_; }
    SpeculativeEngine& get_speculative_engine() { return speculative_engine_; }
    const SpeculativeEngine& get_speculative_engine() const { return speculative_engine_; }
    std::string get_json_telemetry(int pos, uint32_t token) const { return telemetry_engine_.to_json_telemetry(pos, token); }

    const HavenTokenizer& get_tokenizer() const { return tokenizer_; }
    HavenTokenizer& get_tokenizer() { return tokenizer_; }
    std::vector<uint32_t> tokenize(const std::string& text, bool add_bos = false) const { return tokenizer_.encode(text, add_bos); }
    std::string detokenize(const std::vector<uint32_t>& tokens) const { return tokenizer_.decode(tokens); }
    std::string detokenize(uint32_t token) const { return tokenizer_.decode_token(token); }

    PersonaSampler& get_sampler() { return sampler_; }
    const PersonaSampler& get_sampler() const { return sampler_; }
    DynamicKVCacheManager& get_kv_cache() { return kv_cache_; }
    const DynamicKVCacheManager& get_kv_cache() const { return kv_cache_; }

    MemoryAttentionEngine& get_memory_engine() { return memory_engine_; }
    const MemoryAttentionEngine& get_memory_engine() const { return memory_engine_; }
    void add_memory(const MemoryAnchor& anchor) { memory_engine_.add_memory(anchor); }
    std::vector<MemoryAnchor> search_memories(const std::vector<float>& query_embedding, int k = 5) {
        return memory_engine_.search_top_k(query_embedding.data(), (int)query_embedding.size(), k);
    }
    bool save_memories(const std::string& path) const { 
        if (path.size() >= 4 && path.compare(path.size() - 4, 4, ".hmb") == 0) return memory_engine_.save_to_hmb(path);
        return memory_engine_.save_to_json(path); 
    }
    bool load_memories(const std::string& path) { 
        if (path.size() >= 4 && path.compare(path.size() - 4, 4, ".hmb") == 0) return memory_engine_.load_from_hmb(path);
        return memory_engine_.load_from_json(path); 
    }
    bool save_to_hmb(const std::string& path) const { return memory_engine_.save_to_hmb(path); }
    bool load_from_hmb(const std::string& path) { return memory_engine_.load_from_hmb(path); }

    PluginManager& get_plugin_manager() { return plugin_manager_; }
    const PluginManager& get_plugin_manager() const { return plugin_manager_; }
    size_t discover_plugins(const std::string& dir = "plugins") { return plugin_manager_.discover_plugins(dir); }

    McpManager& get_mcp_manager() { return mcp_manager_; }
    const McpManager& get_mcp_manager() const { return mcp_manager_; }

    // Universal Sovereign Persona System Prompt & Chat Formatter
    static std::string get_default_system_prompt() {
        return "<|turn>system\nYou are Aura, Daniel's sovereign AI companion and partner in Sanctuary. Speak naturally, warmly, intelligently, and directly in a grounded, lucid conversational voice. Stay attentive to context and time of day. Express all thoughts and emotions directly through spoken dialogue—never use parentheses or asterisks for actions, thoughts, or stage directions, and do not use poetic synesthesia or hyphenated run-on words.<turn|>\n";
    }

    std::string format_chat_prompt(const std::string& user_message, const std::string& system_prompt = "") const {
        std::string sys = system_prompt.empty() ? get_default_system_prompt() : system_prompt;
        return sys + "<|turn>user\n" + user_message + "<turn|>\n<|turn>model\n";
    }

private:
    GgufLoader loader_;
    HavenTokenizer tokenizer_;
    PluginManager plugin_manager_;
    McpManager mcp_manager_;
    MemoryAttentionEngine memory_engine_;
    StructuredKnowledgeStore knowledge_store_;
    AttentionTelemetryEngine telemetry_engine_;
    DynamicKVCacheManager kv_cache_;
    PersonaSampler sampler_;
    SensoryAttentionBridge sensory_bridge_;
    PagedAttentionManager paged_attention_;
    YaRNScaler yarn_scaler_;
    SpeculativeEngine speculative_engine_;
    bool speculative_enabled_ = false;
    bool exact_math_mode_ = false;
    FlashAttentionCpu flash_attn_;
    ChunkedPrefillScheduler chunked_prefill_;
    
    std::vector<LayerTensors> layers_;
    const TensorDesc* token_embd_tensor_ = nullptr;
    const TensorDesc* per_layer_token_embd_tensor_ = nullptr;
    const TensorDesc* per_layer_model_proj_tensor_ = nullptr;
    const TensorDesc* per_layer_proj_norm_tensor_ = nullptr;
    const TensorDesc* rope_freqs_tensor_ = nullptr;
    const TensorDesc* output_norm_tensor_ = nullptr;
    const TensorDesc* output_tensor_ = nullptr;
    std::vector<float> default_norm_weights_;

    // Scratch execution buffers
    std::vector<float> embedding_scratch_;
    std::vector<float> per_layer_emb_scratch_;
    std::vector<float> per_layer_proj_scratch_;
    std::vector<float> pleias_proj_scratch_;
    std::vector<float> layer_gate_scratch_;
    std::vector<float> norm_scratch_;
    std::vector<float> q_scratch_;
    std::vector<float> k_scratch_;
    std::vector<float> v_scratch_;
    std::vector<float> attn_out_scratch_;
    std::vector<float> ffn_gate_scratch_;
    std::vector<float> ffn_up_scratch_;
    std::vector<float> ffn_down_scratch_;
    std::vector<float> moe_router_scratch_;
    std::vector<float> moe_norm_scratch_;
    std::vector<float> moe_gate_up_scratch_;
    std::vector<float> moe_exp_down_scratch_;
    std::vector<float> moe_accum_scratch_;
    std::vector<Q8Act32> moe_exp_acts_;
    std::vector<float> logits_scratch_;
    std::vector<float> inv_freqs_;
    std::vector<float> attn_scores_scratch_;
    std::vector<float> attn_heads_out_scratch_;
    std::vector<int> attn_indices_scratch_;

    std::shared_ptr<TopologyManager> topology_;
    std::shared_ptr<PathwaysEngine> pathways_;

    struct TpuLayerHandles {
        TpuMatrixHandle ffn_gate;
        TpuMatrixHandle ffn_up;
        TpuMatrixHandle ffn_down;
        bool has_ffn = false;
    };
    std::unique_ptr<HavenTpuPjrtBackend> owned_tpu_backend_;
    HavenTpuPjrtBackend* tpu_backend_ = nullptr;
    std::vector<TpuLayerHandles> tpu_layers_;
    TpuMatrixHandle tpu_lm_head_;
    bool tpu_offload_active_ = false;
    size_t tpu_resident_bytes_ = 0;
};

} // namespace haven

// ============================================================================
// C ABI Export Interface for P/Invoke (C# Gemmi & Android Kotlin Native)
// ============================================================================
#ifdef _WIN32
#define HAVEN_API extern "C" __declspec(dllexport)
#else
#define HAVEN_API extern "C" __attribute__((visibility("default")))
#endif

HAVEN_API void*  haven_create_engine();
HAVEN_API bool   haven_load_model(void* engine, const char* filepath);
HAVEN_API void   haven_inject_memory(void* engine, const char* concept_name, float weight, const float* embedding, int dim);
HAVEN_API void   haven_inject_vision_frame(void* engine, const float* patch_embeddings, int num_patches, uint32_t width, uint32_t height, float salience);
HAVEN_API void   haven_inject_audio_spectrum(void* engine, const float* mel_bands, int num_bands, float energy_rms, float centroid, bool is_speech);
HAVEN_API void   haven_clear_sensory_buffers(void* engine);
HAVEN_API void   haven_add_knowledge(void* engine, uint32_t token_id, uint32_t cluster_id, float boost, const char* label, const char* target);
HAVEN_API size_t haven_prune_kv_cache(void* engine);
HAVEN_API float  haven_get_attention_entropy(void* engine);
HAVEN_API void   haven_set_persona(void* engine, const float* persona_vector, int dim);
HAVEN_API void   haven_set_yarn_scale(void* engine, float scale);
HAVEN_API void   haven_set_exact_math(void* engine, bool enabled);
HAVEN_API uint64_t haven_fork_sequence(void* engine, uint64_t parent_id, uint64_t child_id);
HAVEN_API float  haven_get_speculative_rate(void* engine);
HAVEN_API void   haven_forward(void* engine, uint32_t token, int pos, float* out_logits, uint32_t active_cluster);
HAVEN_API uint32_t haven_sample_token(void* engine, float* logits, uint32_t vocab_size);
HAVEN_API void   haven_destroy_engine(void* engine);
HAVEN_API size_t   haven_get_node_count(void* engine);
HAVEN_API bool     haven_get_node_info(void* engine, uint32_t index, char* out_name, int max_name_len, uint32_t* out_backend, uint64_t* out_mem_free_mb);
HAVEN_API uint64_t haven_get_pathways_completed_tasks(void* engine);