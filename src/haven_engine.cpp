#include "haven/haven_engine.h"
#include <algorithm>
#include <iostream>
#include <cstring>
#include <cmath>
#include <filesystem>

namespace haven {

HavenEngine::HavenEngine()
    : telemetry_engine_(32, 32),
      kv_cache_(32, 8192, 8, 128, 0.035f, 0.985f)
{
    // Pre-allocate scratch buffers
    embedding_scratch_.resize(4096, 0.0f);
    norm_scratch_.resize(4096, 0.0f);
    q_scratch_.resize(4096, 0.0f);
    k_scratch_.resize(1024, 0.0f); // 8 kv heads * 128
    v_scratch_.resize(1024, 0.0f);
    attn_out_scratch_.resize(4096, 0.0f);
    ffn_gate_scratch_.resize(14336, 0.0f);
    ffn_up_scratch_.resize(14336, 0.0f);
    ffn_down_scratch_.resize(4096, 0.0f);
    logits_scratch_.resize(128256, 0.0f);

    // Initialize Clean-Room Topology and Pathways Engine
    topology_ = std::make_shared<TopologyManager>();
    topology_->detect_topology();
    pathways_ = std::make_shared<PathwaysEngine>(topology_);
    pathways_->start();

    // Link Model Context Protocol (MCP) Subsystem to Plugin Manager
    plugin_manager_.set_mcp_manager(&mcp_manager_);
}

HavenEngine::~HavenEngine() {
    disable_tpu_offload();
    if (pathways_) {
        pathways_->shutdown();
    }
}

bool HavenEngine::enable_tpu_offload(HavenTpuPjrtBackend* external_backend) {
    disable_tpu_offload();
    const auto& cfg = loader_.get_config();
    if (cfg.num_layers == 0 || cfg.embedding_dim == 0 || cfg.hidden_dim == 0 || cfg.vocab_size == 0) {
        return false;
    }

    if (external_backend) {
        if (!external_backend->is_initialized()) return false;
        tpu_backend_ = external_backend;
    } else {
        owned_tpu_backend_ = std::make_unique<HavenTpuPjrtBackend>();
        if (!owned_tpu_backend_->initialize()) {
            owned_tpu_backend_.reset();
            return false;
        }
        tpu_backend_ = owned_tpu_backend_.get();
    }

    const int n_devs = std::max(1, static_cast<int>(tpu_backend_->get_device_count()));
    tpu_layers_.resize(cfg.num_layers);
    size_t total_bytes = 0;

    // Upload all layer FFN weights (GeGLU) across available TPU HBM3 chips (round-robin shard by layer)
    if (cfg.ffn_activation == FfnActivation::GELU) {
        for (uint32_t l = 0; l < cfg.num_layers && l < layers_.size(); ++l) {
            const auto& lt = layers_[l];
            if (!lt.ffn_gate || !lt.ffn_up || !lt.ffn_down) continue;
            const int dev_idx = static_cast<int>(l % static_cast<uint32_t>(n_devs));
            bool ok = tpu_backend_->upload_gguf_weight_matrix(*lt.ffn_gate, static_cast<int>(cfg.hidden_dim), static_cast<int>(cfg.embedding_dim), tpu_layers_[l].ffn_gate, dev_idx)
                   && tpu_backend_->upload_gguf_weight_matrix(*lt.ffn_up,   static_cast<int>(cfg.hidden_dim), static_cast<int>(cfg.embedding_dim), tpu_layers_[l].ffn_up, dev_idx)
                   && tpu_backend_->upload_gguf_weight_matrix(*lt.ffn_down, static_cast<int>(cfg.embedding_dim), static_cast<int>(cfg.hidden_dim), tpu_layers_[l].ffn_down, dev_idx);
            if (ok) {
                tpu_layers_[l].has_ffn = true;
                total_bytes += tpu_layers_[l].ffn_gate.bytes + tpu_layers_[l].ffn_up.bytes + tpu_layers_[l].ffn_down.bytes;
            } else {
                tpu_backend_->free_matrix(tpu_layers_[l].ffn_gate);
                tpu_backend_->free_matrix(tpu_layers_[l].ffn_up);
                tpu_backend_->free_matrix(tpu_layers_[l].ffn_down);
                tpu_layers_[l].has_ffn = false;
            }
        }
    }

    // Upload LM head (e.g. 262,144 x 2,560) to TPU HBM3 Device 0
    if (output_tensor_) {
        if (tpu_backend_->upload_gguf_weight_matrix(*output_tensor_, static_cast<int>(cfg.vocab_size), static_cast<int>(cfg.embedding_dim), tpu_lm_head_, 0)) {
            total_bytes += tpu_lm_head_.bytes;
        }
    }

    tpu_resident_bytes_ = total_bytes;
    tpu_offload_active_ = (total_bytes > 0);
    return tpu_offload_active_;
}

void HavenEngine::disable_tpu_offload() {
    if (tpu_backend_) {
        for (auto& tl : tpu_layers_) {
            tpu_backend_->free_matrix(tl.ffn_gate);
            tpu_backend_->free_matrix(tl.ffn_up);
            tpu_backend_->free_matrix(tl.ffn_down);
            tl.has_ffn = false;
        }
        tpu_backend_->free_matrix(tpu_lm_head_);
    }
    tpu_layers_.clear();
    tpu_backend_ = nullptr;
    owned_tpu_backend_.reset();
    tpu_offload_active_ = false;
    tpu_resident_bytes_ = 0;
}

bool HavenEngine::load_model(const std::string& gguf_filepath) {
    if (!loader_.load_file(gguf_filepath)) {
        return false;
    }
    const auto& cfg = loader_.get_config();
    if (cfg.embedding_dim == 0 || cfg.hidden_dim == 0 || cfg.num_layers == 0 ||
        cfg.num_heads == 0 || cfg.num_kv_heads == 0 || cfg.head_dim == 0 ||
        cfg.vocab_size == 0 || cfg.max_context_length == 0) {
        std::cerr << "⛔ [HavenEngine] Invalid zero model dimension in GGUF config.\n";
        return false;
    }

    embedding_scratch_.resize(cfg.embedding_dim, 0.0f);
    norm_scratch_.resize(cfg.embedding_dim, 0.0f);
    q_scratch_.resize(std::max(cfg.embedding_dim, cfg.num_heads * cfg.head_dim), 0.0f);
    k_scratch_.resize(std::max(cfg.embedding_dim, cfg.num_kv_heads * cfg.head_dim), 0.0f);
    v_scratch_.resize(std::max(cfg.embedding_dim, cfg.num_kv_heads * cfg.head_dim), 0.0f);
    attn_out_scratch_.resize(cfg.embedding_dim, 0.0f);
    ffn_gate_scratch_.resize(cfg.hidden_dim, 0.0f);
    ffn_up_scratch_.resize(cfg.hidden_dim, 0.0f);
    ffn_down_scratch_.resize(cfg.embedding_dim, 0.0f);
    if (cfg.expert_count > 0) {
        const uint32_t exp_ff = (cfg.expert_ffn_dim > 0) ? cfg.expert_ffn_dim : cfg.hidden_dim;
        moe_router_scratch_.resize(cfg.expert_count, 0.0f);
        moe_norm_scratch_.resize(cfg.embedding_dim, 0.0f);
        moe_gate_up_scratch_.resize(static_cast<size_t>(exp_ff) * 2ULL, 0.0f);
        moe_exp_down_scratch_.resize(cfg.embedding_dim, 0.0f);
        moe_accum_scratch_.resize(cfg.embedding_dim, 0.0f);
    }
    logits_scratch_.resize(cfg.vocab_size, 0.0f);
    default_norm_weights_.assign(std::max({cfg.embedding_dim, cfg.hidden_dim, cfg.head_dim, 256u}), 1.0f);
    attn_scores_scratch_.resize(std::max(1024u, cfg.max_context_length), 0.0f);

    // Cache global tensors
    token_embd_tensor_ = loader_.get_tensor("token_embd.weight");
    per_layer_token_embd_tensor_ = loader_.get_tensor("per_layer_token_embd.weight");
    per_layer_model_proj_tensor_ = loader_.get_tensor("per_layer_model_proj.weight");
    per_layer_proj_norm_tensor_  = loader_.get_tensor("per_layer_proj_norm.weight");

    layer_gate_scratch_.resize(cfg.embedding_dim, 0.0f);

    output_norm_tensor_ = loader_.get_tensor("output_norm.weight");
    rope_freqs_tensor_  = loader_.get_tensor("rope_freqs.weight");
    output_tensor_ = loader_.get_tensor("output.weight");
    if (!output_tensor_) output_tensor_ = token_embd_tensor_;

    // Cache per-layer tensors & determine worst-case Q/KV projection dimensions across all layers
    layers_.resize(cfg.num_layers);
    uint32_t max_q_dim = cfg.num_heads * cfg.head_dim;
    uint32_t max_kv_dim = cfg.num_kv_heads * cfg.head_dim;
    for (uint32_t l = 0; l < cfg.num_layers; ++l) {
        std::string prefix = "blk." + std::to_string(l) + ".";
        layers_[l].attn_norm            = loader_.get_tensor(prefix + "attn_norm.weight");
        layers_[l].attn_q               = loader_.get_tensor(prefix + "attn_q.weight");
        layers_[l].attn_k               = loader_.get_tensor(prefix + "attn_k.weight");
        layers_[l].attn_v               = loader_.get_tensor(prefix + "attn_v.weight");
        layers_[l].attn_q_bias          = loader_.get_tensor(prefix + "attn_q.bias");
        layers_[l].attn_k_bias          = loader_.get_tensor(prefix + "attn_k.bias");
        layers_[l].attn_v_bias          = loader_.get_tensor(prefix + "attn_v.bias");
        layers_[l].attn_q_norm          = loader_.get_tensor(prefix + "attn_q_norm.weight");
        layers_[l].attn_k_norm          = loader_.get_tensor(prefix + "attn_k_norm.weight");
        layers_[l].attn_output          = loader_.get_tensor(prefix + "attn_output.weight");
        layers_[l].post_attention_norm  = loader_.get_tensor(prefix + "post_attention_norm.weight");
        layers_[l].ffn_norm             = loader_.get_tensor(prefix + "ffn_norm.weight");
        layers_[l].ffn_gate             = loader_.get_tensor(prefix + "ffn_gate.weight");
        layers_[l].ffn_up               = loader_.get_tensor(prefix + "ffn_up.weight");
        layers_[l].ffn_down             = loader_.get_tensor(prefix + "ffn_down.weight");
        layers_[l].post_ffw_norm        = loader_.get_tensor(prefix + "post_ffw_norm.weight");
        layers_[l].post_ffw_norm_1      = loader_.get_tensor(prefix + "post_ffw_norm_1.weight");
        layers_[l].pre_ffw_norm_2       = loader_.get_tensor(prefix + "pre_ffw_norm_2.weight");
        layers_[l].post_ffw_norm_2      = loader_.get_tensor(prefix + "post_ffw_norm_2.weight");
        layers_[l].ffn_gate_inp         = loader_.get_tensor(prefix + "ffn_gate_inp.weight");
        layers_[l].ffn_gate_inp_scale   = loader_.get_tensor(prefix + "ffn_gate_inp.scale");
        layers_[l].ffn_gate_up_exps     = loader_.get_tensor(prefix + "ffn_gate_up_exps.weight");
        layers_[l].ffn_gate_exps        = loader_.get_tensor(prefix + "ffn_gate_exps.weight");
        layers_[l].ffn_up_exps          = loader_.get_tensor(prefix + "ffn_up_exps.weight");
        layers_[l].ffn_down_exps        = loader_.get_tensor(prefix + "ffn_down_exps.weight");
        layers_[l].ffn_down_exps_scale  = loader_.get_tensor(prefix + "ffn_down_exps.scale");
        layers_[l].inp_gate             = loader_.get_tensor(prefix + "inp_gate.weight");
        layers_[l].proj                 = loader_.get_tensor(prefix + "proj.weight");
        layers_[l].post_norm            = loader_.get_tensor(prefix + "post_norm.weight");
        layers_[l].layer_output_scale   = loader_.get_tensor(prefix + "layer_output_scale.weight");

        if (layers_[l].attn_q && layers_[l].attn_q->shape.size() > 1) {
            max_q_dim = std::max(max_q_dim, static_cast<uint32_t>(layers_[l].attn_q->shape[1]));
        }
        if (layers_[l].attn_k && layers_[l].attn_k->shape.size() > 1) {
            max_kv_dim = std::max(max_kv_dim, static_cast<uint32_t>(layers_[l].attn_k->shape[1]));
        }
        if (layers_[l].attn_v && layers_[l].attn_v->shape.size() > 1) {
            max_kv_dim = std::max(max_kv_dim, static_cast<uint32_t>(layers_[l].attn_v->shape[1]));
        }
    }

    uint32_t max_head_dim = std::max(cfg.head_dim, (cfg.num_heads > 0 ? max_q_dim / cfg.num_heads : cfg.head_dim));
    uint32_t kv_cache_head_dim = std::max(max_head_dim, (cfg.num_kv_heads > 0 ? (max_kv_dim + cfg.num_kv_heads - 1) / cfg.num_kv_heads : cfg.head_dim));
    kv_cache_ = DynamicKVCacheManager(cfg.num_layers, cfg.max_context_length, cfg.num_kv_heads, kv_cache_head_dim, 0.0f, 1.0f);
    telemetry_engine_ = AttentionTelemetryEngine(cfg.num_layers, cfg.num_heads);
    tokenizer_.initialize(loader_.get_vocabulary());
    speculative_engine_.initialize_draft_heads(cfg.embedding_dim);

    // Dynamically resize all scratch buffers to exact worst-case model dimensions
    const size_t pleias_dim = static_cast<size_t>(cfg.num_layers) * 256ULL;
    embedding_scratch_.resize(cfg.embedding_dim, 0.0f);
    norm_scratch_.resize(cfg.embedding_dim, 0.0f);
    q_scratch_.resize(std::max<size_t>(cfg.embedding_dim, max_q_dim), 0.0f);
    k_scratch_.resize(std::max<size_t>(cfg.embedding_dim, max_kv_dim), 0.0f);
    v_scratch_.resize(std::max<size_t>(cfg.embedding_dim, max_kv_dim), 0.0f);
    attn_out_scratch_.resize(cfg.embedding_dim, 0.0f);
    ffn_gate_scratch_.resize(cfg.hidden_dim, 0.0f);
    ffn_up_scratch_.resize(cfg.hidden_dim, 0.0f);
    ffn_down_scratch_.resize(cfg.embedding_dim, 0.0f);
    if (cfg.expert_count > 0) {
        const uint32_t exp_ff = (cfg.expert_ffn_dim > 0) ? cfg.expert_ffn_dim : cfg.hidden_dim;
        moe_router_scratch_.resize(cfg.expert_count, 0.0f);
        moe_norm_scratch_.resize(cfg.embedding_dim, 0.0f);
        moe_gate_up_scratch_.resize(static_cast<size_t>(exp_ff) * 2ULL, 0.0f);
        moe_exp_down_scratch_.resize(cfg.embedding_dim, 0.0f);
        moe_accum_scratch_.resize(cfg.embedding_dim, 0.0f);
    }
    layer_gate_scratch_.resize(cfg.hidden_dim, 0.0f);
    per_layer_emb_scratch_.resize(pleias_dim, 0.0f);
    per_layer_proj_scratch_.resize(pleias_dim, 0.0f);
    pleias_proj_scratch_.resize(cfg.embedding_dim, 0.0f);
    attn_scores_scratch_.resize(static_cast<size_t>(cfg.num_heads) * cfg.max_context_length, 0.0f);
    attn_heads_out_scratch_.resize(std::max<size_t>(cfg.embedding_dim, max_q_dim), 0.0f);
    attn_indices_scratch_.resize(std::max<size_t>(static_cast<size_t>(cfg.max_context_length) + 4, 520), 0);
    logits_scratch_.resize(cfg.vocab_size, 0.0f);
    default_norm_weights_.assign(std::max({cfg.embedding_dim, cfg.hidden_dim, max_head_dim, 256u}), 1.0f);

    // Initialize Persistent Cognitive Memory Vault
    std::string mem_path = "wwwroot/memories.json";
    if (!std::filesystem::exists(mem_path)) {
        mem_path = "C:\\Users\\admin\\source\\haven-cpp\\wwwroot\\memories.json";
    }
    if (!memory_engine_.load_from_json(mem_path)) {
        memory_engine_.initialize_sovereign_anchors(cfg.head_dim);
        memory_engine_.save_to_json(mem_path);
    }

    // Discover and load sovereign plugins
    plugin_manager_.discover_plugins("plugins");
    if (plugin_manager_.get_plugin_count() == 0) {
        plugin_manager_.discover_plugins("C:\\Users\\admin\\source\\haven-cpp\\plugins");
    }

    // Initialize Universal Default Persona Sampler Configuration (Single Source of Truth across CLI & Server)
    sampler_.get_params().temperature = 0.48f;
    sampler_.get_params().top_p = 0.90f;
    sampler_.get_params().top_k = 40;
    sampler_.get_params().min_p = 0.05f;
    sampler_.get_params().repetition_penalty = 1.15f;
    sampler_.get_params().dry_multiplier = 0.5f;
    sampler_.get_params().dry_base = 1.75f;
    sampler_.get_params().dry_allowed_length = 2;
    sampler_.get_params().dry_penalty_last_n = 64;

    // Initialize cross-token lemma duplicate suppression map (e.g. ▁just vs just)
    sampler_.initialize_vocab_counterparts(loader_.get_vocabulary());

    return true;
}

void HavenEngine::inject_memory(const std::string& concept_name, float weight, const std::vector<float>& embedding) {
    memory_engine_.inject_memory(concept_name, weight, embedding);
}

void HavenEngine::clear_memories() {
    memory_engine_.clear_memories();
}

void HavenEngine::inject_vision_frame(const std::vector<float>& patch_embeddings, uint32_t width, uint32_t height, float salience) {
    VisualFrame frame;
    frame.width = width;
    frame.height = height;
    frame.patch_embeddings = patch_embeddings;
    frame.salience_score = salience;
    sensory_bridge_.ingest_vision_frame(frame);
}

void HavenEngine::inject_audio_spectrum(const std::vector<float>& mel_bands, float energy_rms, float centroid, bool is_speech) {
    AudioSpectrumFrame frame;
    frame.mel_bands = mel_bands;
    frame.energy_rms = energy_rms;
    frame.spectral_centroid = centroid;
    frame.is_speech_active = is_speech;
    sensory_bridge_.ingest_audio_spectrum(frame);
}

void HavenEngine::clear_sensory_buffers() {
    sensory_bridge_.clear_sensory_buffers();
}

void HavenEngine::add_knowledge_relation(
    uint32_t token_id,
    uint32_t cluster_id,
    float boost,
    const std::string& label,
    const std::string& target)
{
    knowledge_store_.add_relation(token_id, cluster_id, boost, label, target);
}

void HavenEngine::set_persona_embedding(const std::vector<float>& persona_vector) {
    sampler_.set_persona_embedding(persona_vector);
}

void HavenEngine::set_yarn_context_scale(float scale) {
    yarn_scaler_.set_scale(scale);
    yarn_scaler_.compute_frequencies(128, 10000.0f, inv_freqs_);
}

SpeculativeCandidate HavenEngine::draft_speculative_tokens(const std::vector<uint32_t>& context) {
    return speculative_engine_.draft_candidates(context, embedding_scratch_.data(), static_cast<uint32_t>(embedding_scratch_.size()));
}

uint64_t HavenEngine::submit_chunked_prefill(const std::vector<uint32_t>& tokens) {
    return chunked_prefill_.submit_prefill_task(tokens);
}

uint64_t HavenEngine::fork_sequence(uint64_t parent_seq_id, uint64_t child_seq_id) {
    return paged_attention_.fork_sequence(parent_seq_id, child_seq_id);
}

size_t HavenEngine::prune_kv_cache() {
    return kv_cache_.prune_all_layers();
}

void HavenEngine::forward(uint32_t token, int pos, float* out_logits, uint32_t active_cluster) {
    (void)active_cluster;
    const auto& cfg = loader_.get_config();
    if (cfg.embedding_dim == 0 || cfg.hidden_dim == 0 || cfg.num_heads == 0 || cfg.num_kv_heads == 0 || cfg.max_context_length == 0) {
        return;
    }
    if (pos < 0) pos = 0;
    if (pos >= static_cast<int>(cfg.max_context_length)) {
        pos = static_cast<int>(cfg.max_context_length) - 1;
    }

    auto safe_f32_weights = [&](const TensorDesc* t, size_t count) -> const float* {
        if (t && t->data && t->type == QuantType::F32 &&
            (t->size_bytes == 0 || t->size_bytes >= count * sizeof(float))) {
            return reinterpret_cast<const float*>(t->data);
        }
        return default_norm_weights_.data();
    };

    // 1. Token Embedding Lookup
    if (token_embd_tensor_ && token < cfg.vocab_size) {
        Avx2Math::dequantize_row(embedding_scratch_.data(), *token_embd_tensor_, static_cast<int>(token), static_cast<int>(cfg.embedding_dim));
        if (cfg.is_gemma) {
            float emb_scale = std::sqrt((float)cfg.embedding_dim);
            for (size_t i = 0; i < cfg.embedding_dim; ++i) {
                embedding_scratch_[i] *= emb_scale;
            }
        }
    } else {
        for (size_t i = 0; i < cfg.embedding_dim; ++i) {
            embedding_scratch_[i] = std::sin((float)(token + i) * 0.01f);
        }
    }

    // 1a-2. Dequantize & Project Per-Layer Direct Embeddings (Google Pleias Branch)
    if (per_layer_token_embd_tensor_ && token < cfg.vocab_size && !per_layer_emb_scratch_.empty()) {
        Avx2Math::dequantize_row(per_layer_emb_scratch_.data(), *per_layer_token_embd_tensor_, static_cast<int>(token), (int)per_layer_emb_scratch_.size());
        
        const float tok_scale = std::sqrt(256.0f); // sqrt(n_embd_per_layer) = 16.0f
        for (size_t i = 0; i < per_layer_emb_scratch_.size(); ++i) {
            per_layer_emb_scratch_[i] *= tok_scale;
        }

        if (per_layer_model_proj_tensor_ && !per_layer_proj_scratch_.empty()) {
            Avx2Math::gemv(per_layer_proj_scratch_.data(), *per_layer_model_proj_tensor_, embedding_scratch_.data(), (int)per_layer_proj_scratch_.size(), cfg.embedding_dim);
            
            float proj_scale = 1.0f / std::sqrt((float)cfg.embedding_dim);
            for (size_t i = 0; i < per_layer_proj_scratch_.size(); ++i) {
                per_layer_proj_scratch_[i] *= proj_scale;
            }
            
            const float* proj_norm_w = safe_f32_weights(per_layer_proj_norm_tensor_, 256);
            
            const float input_scale = 1.0f / std::sqrt(2.0f);
            for (uint32_t l = 0; l < cfg.num_layers; ++l) {
                float* p_slice = per_layer_proj_scratch_.data() + l * 256;
                float* e_slice = per_layer_emb_scratch_.data() + l * 256;
                Avx2Math::rms_norm(p_slice, p_slice, proj_norm_w, 256, cfg.rms_norm_eps);
                for (int i = 0; i < 256; ++i) {
                    e_slice[i] = (p_slice[i] + e_slice[i]) * input_scale;
                }
            }
        }
    }

    // 1b. Mix in Multimodal Sensory Token Vectors (Vision + Audio Mel-Spectrum)
    auto sensory_vec = sensory_bridge_.project_sensory_tokens(cfg.embedding_dim);
    for (size_t i = 0; i < cfg.embedding_dim; ++i) {
        embedding_scratch_[i] += sensory_vec[i];
    }

    // 2. Transformer Layer Loop
    for (uint32_t l = 0; l < cfg.num_layers; ++l) {
        const auto& lt = (l < layers_.size()) ? layers_[l] : LayerTensors{};

        const int q_dim = std::max(1, std::min<int>(
            static_cast<int>(q_scratch_.size()),
            (lt.attn_q && lt.attn_q->shape.size() > 1) ? (int)lt.attn_q->shape[1] : (int)(cfg.num_heads * cfg.head_dim)));
        const int actual_head_dim = std::max(1, (cfg.num_heads > 0) ? (q_dim / (int)cfg.num_heads) : (int)cfg.head_dim);
        const uint32_t layer_n_kv_heads = (l < cfg.head_count_kv_per_layer.size() && cfg.head_count_kv_per_layer[l] > 0)
            ? cfg.head_count_kv_per_layer[l]
            : ((lt.attn_k && lt.attn_k->shape.size() > 1 && actual_head_dim > 0)
                ? std::max(1u, static_cast<uint32_t>(lt.attn_k->shape[1]) / static_cast<uint32_t>(actual_head_dim))
                : cfg.num_kv_heads);
        const int kv_dim = std::max(1, std::min<int>(
            static_cast<int>(k_scratch_.size()),
            (lt.attn_k && lt.attn_k->shape.size() > 1) ? (int)lt.attn_k->shape[1] : (int)(layer_n_kv_heads * static_cast<uint32_t>(actual_head_dim))));
        const float attn_scale = cfg.is_gemma ? 1.0f : (1.0f / std::sqrt((float)actual_head_dim));
        const int gqa_ratio = std::max(1u, cfg.num_heads / std::max(1u, layer_n_kv_heads));

        // 2a. Pre-Attention RMSNorm
        const float* attn_norm_w = safe_f32_weights(lt.attn_norm, cfg.embedding_dim);
        Avx2Math::rms_norm(norm_scratch_.data(), embedding_scratch_.data(), attn_norm_w, cfg.embedding_dim, cfg.rms_norm_eps);

        // 2b. Q, K, V Projections (with Gemma 4 k_eq_v fallback when attn_v is omitted on global layers)
        if (lt.attn_q) {
            Avx2Math::gemv(q_scratch_.data(), *lt.attn_q, norm_scratch_.data(), q_dim, cfg.embedding_dim);
            if (lt.attn_q_bias) Avx2Math::add_bias(q_scratch_.data(), *lt.attn_q_bias, q_dim);
        } else {
            for (int i = 0; i < q_dim; ++i) q_scratch_[i] = norm_scratch_[i % cfg.embedding_dim];
        }

        if (lt.attn_k) {
            Avx2Math::gemv(k_scratch_.data(), *lt.attn_k, norm_scratch_.data(), kv_dim, cfg.embedding_dim);
            if (lt.attn_k_bias) Avx2Math::add_bias(k_scratch_.data(), *lt.attn_k_bias, kv_dim);
        } else {
            for (int i = 0; i < kv_dim; ++i) k_scratch_[i] = norm_scratch_[i % cfg.embedding_dim];
        }

        if (lt.attn_v) {
            Avx2Math::gemv(v_scratch_.data(), *lt.attn_v, norm_scratch_.data(), kv_dim, cfg.embedding_dim);
            if (lt.attn_v_bias) Avx2Math::add_bias(v_scratch_.data(), *lt.attn_v_bias, kv_dim);
        } else if (lt.attn_k) {
            // Gemma 4 k_eq_v: V shares raw K projection prior to K-norm and RoPE
            std::memcpy(v_scratch_.data(), k_scratch_.data(), static_cast<size_t>(kv_dim) * sizeof(float));
        } else {
            for (int i = 0; i < kv_dim; ++i) v_scratch_[i] = norm_scratch_[i % cfg.embedding_dim];
        }

        // 2b-2. Gemma 2 / 4 Per-Head Q-Norm, K-Norm, and V-Norm
        if (lt.attn_q_norm) {
            const float* q_norm_w = safe_f32_weights(lt.attn_q_norm, static_cast<size_t>(actual_head_dim));
            for (uint32_t h = 0; h < cfg.num_heads; ++h) {
                float* q_h = q_scratch_.data() + h * actual_head_dim;
                Avx2Math::rms_norm(q_h, q_h, q_norm_w, actual_head_dim, cfg.rms_norm_eps);
            }
        }
        if (lt.attn_k_norm) {
            const float* k_norm_w = safe_f32_weights(lt.attn_k_norm, static_cast<size_t>(actual_head_dim));
            for (uint32_t h = 0; h < layer_n_kv_heads; ++h) {
                float* k_h = k_scratch_.data() + h * actual_head_dim;
                Avx2Math::rms_norm(k_h, k_h, k_norm_w, actual_head_dim, cfg.rms_norm_eps);
            }
        }
        if (cfg.is_gemma && lt.attn_q_norm && lt.attn_k_norm) {
            for (uint32_t h = 0; h < layer_n_kv_heads; ++h) {
                float* v_h = v_scratch_.data() + h * actual_head_dim;
                Avx2Math::rms_norm(v_h, v_h, default_norm_weights_.data(), actual_head_dim, cfg.rms_norm_eps);
            }
        }

        // 2c. RoPE Rotary Embeddings
        bool has_swa = (cfg.sliding_window > 0) && (cfg.is_gemma || cfg.architecture == "mistral");
        bool is_swa = (l < cfg.sliding_window_pattern.size())
            ? cfg.sliding_window_pattern[l]
            : (has_swa && (cfg.architecture == "mistral" || (l % 6 != 5)));
        float layer_freq_base = is_swa ? cfg.rope_freq_base_swa : cfg.rope_freq_base;
        int rope_dim = actual_head_dim;
        const float* freq_factors_w = (!is_swa && rope_freqs_tensor_ && rope_freqs_tensor_->type == QuantType::F32 &&
            (rope_freqs_tensor_->size_bytes == 0 || rope_freqs_tensor_->size_bytes >= static_cast<size_t>(rope_dim / 2) * sizeof(float)))
            ? reinterpret_cast<const float*>(rope_freqs_tensor_->data) : nullptr;

        uint32_t kv_source_layer = l;
        bool has_kv = (lt.attn_k != nullptr);
        if (!has_kv) {
            if (cfg.num_layers > 20 && cfg.shared_kv_layers > 0 && cfg.shared_kv_layers < cfg.num_layers) {
                uint32_t n_kv_start = cfg.num_layers - cfg.shared_kv_layers;
                kv_source_layer = n_kv_start - (is_swa ? 2 : 1);
            } else {
                kv_source_layer = (l > 0) ? (l - 1) : 0;
            }
            if (kv_source_layer >= cfg.num_layers) {
                kv_source_layer = 0;
            }
        }

        if (has_kv) {
            Avx2Math::apply_rope(q_scratch_.data(), k_scratch_.data(), actual_head_dim, cfg.num_heads, layer_n_kv_heads, pos, layer_freq_base, cfg.rope_freq_scale, freq_factors_w, rope_dim);
            kv_cache_.write(l, pos, k_scratch_.data(), v_scratch_.data(), kv_dim);
        } else {
            Avx2Math::apply_rope(q_scratch_.data(), nullptr, actual_head_dim, cfg.num_heads, 0, pos, layer_freq_base, cfg.rope_freq_scale, freq_factors_w, rope_dim);
        }

        // 2e. Multi-Head Attention (GQA) with Attention Sinks & Zero-Allocation Scratch Buffers
        const auto& kv_lay = kv_cache_.get_layer(kv_source_layer);
        const size_t max_safe_positions = (kv_dim > 0) ? (kv_lay.keys.size() / static_cast<size_t>(kv_dim)) : 0;
        const int seq_len = std::min({pos + 1, (int)cfg.max_context_length, (int)max_safe_positions});
        const int swa_win = (cfg.sliding_window > 0) ? static_cast<int>(cfg.sliding_window) : 512;
        const int s_start = is_swa ? std::max(0, seq_len - swa_win) : 0;

        int active_attn_len = 0;
        const int max_indices = static_cast<int>(attn_indices_scratch_.size());
        if (is_swa && s_start > 4) {
            for (int s = 0; s < 4 && active_attn_len < max_indices; ++s) attn_indices_scratch_[active_attn_len++] = s;
            for (int s = s_start; s < seq_len && active_attn_len < max_indices; ++s) attn_indices_scratch_[active_attn_len++] = s;
        } else {
            for (int s = s_start; s < seq_len && active_attn_len < max_indices; ++s) attn_indices_scratch_[active_attn_len++] = s;
        }

        const float* layer_k = kv_lay.keys.data();
        const float* layer_v = kv_lay.values.data();

        const int num_heads_i = static_cast<int>(cfg.num_heads);
        #pragma omp parallel for schedule(static)
        for (int h = 0; h < num_heads_i; ++h) {
            const float* q_h = q_scratch_.data() + static_cast<size_t>(h) * static_cast<size_t>(actual_head_dim);
            const uint32_t kv_head_idx = std::min(static_cast<uint32_t>(h / gqa_ratio), std::max(1u, layer_n_kv_heads) - 1);
            const uint32_t kv_head_offset = kv_head_idx * actual_head_dim;
            float* out_h = attn_heads_out_scratch_.data() + static_cast<size_t>(h) * static_cast<size_t>(actual_head_dim);

            if (active_attn_len > 0 && kv_head_offset + static_cast<uint32_t>(actual_head_dim) <= static_cast<uint32_t>(kv_dim)) {
                FlashAttentionCpu::compute_tiled_attention(
                    q_h,
                    layer_k,
                    layer_v,
                    attn_indices_scratch_.data(),
                    (uint32_t)active_attn_len,
                    (uint32_t)actual_head_dim,
                    (uint32_t)kv_dim,
                    kv_head_offset,
                    attn_scale,
                    cfg.attn_logit_softcapping,
                    out_h
                );
            } else {
                std::memset(out_h, 0, static_cast<size_t>(actual_head_dim) * sizeof(float));
            }
        }

        // 2f. Attention Output Projection
        if (lt.attn_output) {
            Avx2Math::gemv(attn_out_scratch_.data(), *lt.attn_output, attn_heads_out_scratch_.data(), cfg.embedding_dim, q_dim);
        } else {
            for (size_t i = 0; i < cfg.embedding_dim; ++i) attn_out_scratch_[i] = attn_heads_out_scratch_[i % q_dim];
        }

        // 2f-2. Post-Attention Norm (Gemma 2 / 4)
        if (lt.post_attention_norm) {
            const float* post_attn_w = safe_f32_weights(lt.post_attention_norm, cfg.embedding_dim);
            Avx2Math::rms_norm(attn_out_scratch_.data(), attn_out_scratch_.data(), post_attn_w, cfg.embedding_dim, cfg.rms_norm_eps);
        }

        // Residual Addition 1
        for (size_t i = 0; i < cfg.embedding_dim; ++i) {
            embedding_scratch_[i] += attn_out_scratch_[i];
        }

        // 2g. Pre-FFN RMSNorm
        const float* ffn_norm_w = safe_f32_weights(lt.ffn_norm, cfg.embedding_dim);
        Avx2Math::rms_norm(norm_scratch_.data(), embedding_scratch_.data(), ffn_norm_w, cfg.embedding_dim, cfg.rms_norm_eps);

        // 2h + 2i. FFN Gate/Up/Activation/Down (Fused TPU MXU GeGLU when offload active, else AVX2 CPU)
        if (tpu_offload_active_ && tpu_backend_ && l < tpu_layers_.size() && tpu_layers_[l].has_ffn &&
            tpu_backend_->fused_geglu_ffn(norm_scratch_.data(), 1,
                                          tpu_layers_[l].ffn_gate, tpu_layers_[l].ffn_up, tpu_layers_[l].ffn_down,
                                          ffn_down_scratch_.data())) {
            // Computed directly on TPU HBM3 + MXU systolic array
        } else {
            if (lt.ffn_gate && lt.ffn_up) {
                Avx2Math::gemv(ffn_gate_scratch_.data(), *lt.ffn_gate, norm_scratch_.data(), cfg.hidden_dim, cfg.embedding_dim);
                Avx2Math::gemv(ffn_up_scratch_.data(), *lt.ffn_up, norm_scratch_.data(), cfg.hidden_dim, cfg.embedding_dim);
                if (cfg.ffn_activation == FfnActivation::GELU) {
                    Avx2Math::gelu(ffn_gate_scratch_.data(), cfg.hidden_dim);
                } else {
                    Avx2Math::silu(ffn_gate_scratch_.data(), cfg.hidden_dim);
                }
                for (size_t i = 0; i < cfg.hidden_dim; ++i) {
                    ffn_gate_scratch_[i] *= ffn_up_scratch_[i];
                }
            } else {
                for (size_t i = 0; i < cfg.hidden_dim; ++i) ffn_gate_scratch_[i] = norm_scratch_[i % cfg.embedding_dim] * 0.1f;
            }

            // 2i. FFN Down Projection
            if (lt.ffn_down) {
                Avx2Math::gemv(ffn_down_scratch_.data(), *lt.ffn_down, ffn_gate_scratch_.data(), cfg.embedding_dim, cfg.hidden_dim);
            } else {
                for (size_t i = 0; i < cfg.embedding_dim; ++i) ffn_down_scratch_[i] = ffn_gate_scratch_[i % cfg.hidden_dim] * 0.05f;
            }
        }

        // 2i-1b. Parallel Sparse Mixture-of-Experts Branch (Gemma 4 26B-A4B MoE)
        if (lt.ffn_gate_inp && lt.ffn_down_exps && (lt.ffn_gate_up_exps || (lt.ffn_gate_exps && lt.ffn_up_exps)) && cfg.expert_count > 0) {
            // Post-norm 1 on shared dense MLP branch before combining with MoE branch
            if (lt.post_ffw_norm_1) {
                const float* p1_w = safe_f32_weights(lt.post_ffw_norm_1, cfg.embedding_dim);
                Avx2Math::rms_norm(ffn_down_scratch_.data(), ffn_down_scratch_.data(), p1_w, cfg.embedding_dim, cfg.rms_norm_eps);
            }

            const int E = static_cast<int>(cfg.expert_count);
            const int K_used = std::clamp(static_cast<int>(cfg.expert_used_count > 0 ? cfg.expert_used_count : 8u), 1, E);
            const int F_exp = static_cast<int>(cfg.expert_ffn_dim > 0 ? cfg.expert_ffn_dim : cfg.hidden_dim);

            // Router input: parameter-free RMSNorm(attn_out) * (1 / sqrt(D)) * ffn_gate_inp.scale
            Avx2Math::rms_norm(moe_norm_scratch_.data(), embedding_scratch_.data(), default_norm_weights_.data(), cfg.embedding_dim, cfg.rms_norm_eps);
            const float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(cfg.embedding_dim));
            const float* rscale = lt.ffn_gate_inp_scale ? safe_f32_weights(lt.ffn_gate_inp_scale, cfg.embedding_dim) : nullptr;
            for (size_t i = 0; i < cfg.embedding_dim; ++i) {
                moe_norm_scratch_[i] *= inv_sqrt_d * (rscale ? rscale[i] : 1.0f);
            }
            Avx2Math::gemv(moe_router_scratch_.data(), *lt.ffn_gate_inp, moe_norm_scratch_.data(), E, cfg.embedding_dim);
            Avx2Math::softmax(moe_router_scratch_.data(), E);

            // Top-K expert selection & renormalization
            std::vector<int> exp_idx(static_cast<size_t>(E));
            for (int e = 0; e < E; ++e) exp_idx[static_cast<size_t>(e)] = e;
            std::partial_sort(exp_idx.begin(), exp_idx.begin() + K_used, exp_idx.end(),
                [&](int a, int b) { return moe_router_scratch_[static_cast<size_t>(a)] > moe_router_scratch_[static_cast<size_t>(b)]; });

            float w_sum = 0.0f;
            for (int k = 0; k < K_used; ++k) w_sum += moe_router_scratch_[static_cast<size_t>(exp_idx[static_cast<size_t>(k)])];
            if (w_sum < 6.103515625e-5f) w_sum = 6.103515625e-5f;

            // Pre-FFN norm 2 on attn_out for sparse experts
            const float* pre2_w = safe_f32_weights(lt.pre_ffw_norm_2, cfg.embedding_dim);
            Avx2Math::rms_norm(moe_norm_scratch_.data(), embedding_scratch_.data(), pre2_w, cfg.embedding_dim, cfg.rms_norm_eps);

            const float* down_s = lt.ffn_down_exps_scale ? safe_f32_weights(lt.ffn_down_exps_scale, static_cast<size_t>(E)) : nullptr;
            std::fill(moe_accum_scratch_.begin(), moe_accum_scratch_.end(), 0.0f);

            for (int k = 0; k < K_used; ++k) {
                const int e = exp_idx[static_cast<size_t>(k)];
                const float w_k = (moe_router_scratch_[static_cast<size_t>(e)] / w_sum) * (down_s ? down_s[e] : 1.0f);

                if (lt.ffn_gate_up_exps) {
                    Avx2Math::gemv_slice(moe_gate_up_scratch_.data(), *lt.ffn_gate_up_exps,
                        static_cast<size_t>(e) * static_cast<size_t>(2 * F_exp),
                        moe_norm_scratch_.data(), 2 * F_exp, cfg.embedding_dim);
                } else {
                    Avx2Math::gemv_slice(moe_gate_up_scratch_.data(), *lt.ffn_gate_exps,
                        static_cast<size_t>(e) * static_cast<size_t>(F_exp),
                        moe_norm_scratch_.data(), F_exp, cfg.embedding_dim);
                    Avx2Math::gemv_slice(moe_gate_up_scratch_.data() + F_exp, *lt.ffn_up_exps,
                        static_cast<size_t>(e) * static_cast<size_t>(F_exp),
                        moe_norm_scratch_.data(), F_exp, cfg.embedding_dim);
                }

                if (cfg.ffn_activation == FfnActivation::GELU) {
                    Avx2Math::gelu(moe_gate_up_scratch_.data(), F_exp);
                } else {
                    Avx2Math::silu(moe_gate_up_scratch_.data(), F_exp);
                }
                for (int j = 0; j < F_exp; ++j) {
                    moe_gate_up_scratch_[static_cast<size_t>(j)] *= moe_gate_up_scratch_[static_cast<size_t>(F_exp + j)];
                }

                Avx2Math::gemv_slice(moe_exp_down_scratch_.data(), *lt.ffn_down_exps,
                    static_cast<size_t>(e) * static_cast<size_t>(cfg.embedding_dim),
                    moe_gate_up_scratch_.data(), cfg.embedding_dim, F_exp);
                for (size_t i = 0; i < cfg.embedding_dim; ++i) {
                    moe_accum_scratch_[i] += w_k * moe_exp_down_scratch_[i];
                }
            }

            if (lt.post_ffw_norm_2) {
                const float* p2_w = safe_f32_weights(lt.post_ffw_norm_2, cfg.embedding_dim);
                Avx2Math::rms_norm(moe_accum_scratch_.data(), moe_accum_scratch_.data(), p2_w, cfg.embedding_dim, cfg.rms_norm_eps);
            }
            for (size_t i = 0; i < cfg.embedding_dim; ++i) {
                ffn_down_scratch_[i] += moe_accum_scratch_[i];
            }
        }

        // 2i-2. Post-FFN Norm (Gemma 2 / 4)
        if (lt.post_ffw_norm) {
            const float* post_ffn_w = safe_f32_weights(lt.post_ffw_norm, cfg.embedding_dim);
            Avx2Math::rms_norm(ffn_down_scratch_.data(), ffn_down_scratch_.data(), post_ffn_w, cfg.embedding_dim, cfg.rms_norm_eps);
        }

        // Residual Addition 2
        for (size_t i = 0; i < cfg.embedding_dim; ++i) {
            embedding_scratch_[i] += ffn_down_scratch_[i];
        }

        // 2-pleias. Per-Layer Token Injection (Google Pleias Branch)
        if (lt.inp_gate && lt.proj && !per_layer_emb_scratch_.empty() && (static_cast<size_t>(l) + 1) * 256 <= per_layer_emb_scratch_.size()) {
            float pleias_gate[256] = {0};
            Avx2Math::gemv(pleias_gate, *lt.inp_gate, embedding_scratch_.data(), 256, cfg.embedding_dim);
            Avx2Math::gelu(pleias_gate, 256);

            const float* layer_emb_slice = per_layer_emb_scratch_.data() + l * 256;
            for (int i = 0; i < 256; ++i) {
                pleias_gate[i] *= layer_emb_slice[i];
            }

            float* pleias_proj = pleias_proj_scratch_.data();
            Avx2Math::gemv(pleias_proj, *lt.proj, pleias_gate, cfg.embedding_dim, 256);

            if (lt.post_norm) {
                const float* post_w = safe_f32_weights(lt.post_norm, cfg.embedding_dim);
                Avx2Math::rms_norm(pleias_proj, pleias_proj, post_w, cfg.embedding_dim, cfg.rms_norm_eps);
            }

            for (size_t i = 0; i < cfg.embedding_dim; ++i) {
                embedding_scratch_[i] += pleias_proj[i];
            }
        } else if (lt.post_norm) {
            const float* post_norm_w = safe_f32_weights(lt.post_norm, cfg.embedding_dim);
            Avx2Math::rms_norm(embedding_scratch_.data(), embedding_scratch_.data(), post_norm_w, cfg.embedding_dim, cfg.rms_norm_eps);
        }

        // 2-scale. Layer Output Scale (Gemma 4 Pleias)
        if (lt.layer_output_scale && lt.layer_output_scale->data && lt.layer_output_scale->type == QuantType::F32 &&
            (lt.layer_output_scale->size_bytes == 0 || lt.layer_output_scale->size_bytes >= sizeof(float))) {
            const float scale = *reinterpret_cast<const float*>(lt.layer_output_scale->data);
            for (size_t i = 0; i < cfg.embedding_dim; ++i) {
                embedding_scratch_[i] *= scale;
            }
        }
    }

    // Advance KV cache sequence position
    kv_cache_.advance_position();
    telemetry_engine_.finalize_step();

    if (!out_logits) {
        return; // Fast prefill: Skip 262k logits projection on intermediate prompt tokens!
    }

    // 3. Final Output Norm & Logits Projection
    const float* out_norm_w = safe_f32_weights(output_norm_tensor_, cfg.embedding_dim);
    Avx2Math::rms_norm(norm_scratch_.data(), embedding_scratch_.data(), out_norm_w, cfg.embedding_dim, cfg.rms_norm_eps);

    if (tpu_offload_active_ && tpu_backend_ && tpu_lm_head_.buffer &&
        tpu_backend_->matmul_resident_nt(norm_scratch_.data(), 1, tpu_lm_head_, out_logits)) {
        // Computed 262k LM head directly on TPU HBM3 + MXU
    } else if (output_tensor_) {
        Avx2Math::gemv(out_logits, *output_tensor_, norm_scratch_.data(), cfg.vocab_size, cfg.embedding_dim);
    } else {
        for (size_t v = 0; v < cfg.vocab_size; ++v) {
            out_logits[v] = std::sin((float)v * 0.05f) * norm_scratch_[v % cfg.embedding_dim];
        }
    }

    // 4. Final Logits Softcapping (Gemma 2 / 4)
    if (cfg.final_logit_softcapping > 0.0f) {
        Avx2Math::softcap_logits(out_logits, cfg.vocab_size, cfg.final_logit_softcapping);
    }
}

void HavenEngine::generate(
    const std::vector<uint32_t>& prompt_tokens,
    int max_new_tokens,
    const std::function<bool(uint32_t token, const std::string& text)>& on_token_callback)
{
    if (prompt_tokens.empty()) return;
    const auto& cfg = loader_.get_config();
    std::vector<uint32_t> generated_tokens;

    // Reset KV cache for fresh generation sequence
    kv_cache_.reset();

    int pos = 0;
    // Fast Process Prompt Prefill: only compute full 262k logits on the final token!
    for (size_t i = 0; i < prompt_tokens.size(); ++i) {
        bool is_last_prompt = (i + 1 == prompt_tokens.size());
        forward(prompt_tokens[i], pos++, is_last_prompt ? logits_scratch_.data() : nullptr);
    }

    // Auto-Regressive Token Generation Loop with Dynamic Soft-Pruning & Speculative Decoding
    int gen = 0;
    while (gen < max_new_tokens) {
        uint32_t next_token = sampler_.sample(logits_scratch_.data(), cfg.vocab_size, generated_tokens);
        generated_tokens.push_back(next_token);
        gen++;

        // Check for end-of-turn / stop tokens across all architectures
        if (tokenizer_.is_stop_token(next_token)) {
            break; // Model emitted end-of-turn
        }

        std::string token_str = tokenizer_.decode_token(next_token);
        if (pathways_) {
            pathways_->publish_stream_token(next_token, token_str);
        }
        if (on_token_callback && !on_token_callback(next_token, token_str)) {
            break; // Callback signaled stop
        }

        forward(next_token, pos++, logits_scratch_.data());

        // Multi-Token Speculative Acceleration
        if (speculative_enabled_ && gen < max_new_tokens) {
            std::vector<uint32_t> current_context = prompt_tokens;
            current_context.insert(current_context.end(), generated_tokens.begin(), generated_tokens.end());

            SpeculativeCandidate candidate = speculative_engine_.draft_candidates(
                current_context, embedding_scratch_.data(), cfg.embedding_dim
            );

            std::vector<uint32_t> accepted;
            speculative_engine_.verify_candidates(candidate, logits_scratch_.data(), cfg.vocab_size, accepted);

            bool any_accepted = !accepted.empty();
            speculative_engine_.update_feedback(any_accepted);

            for (uint32_t tok : accepted) {
                if (gen >= max_new_tokens) break;
                if (tokenizer_.is_stop_token(tok)) {
                    generated_tokens.push_back(tok);
                    gen++;
                    return;
                }
                generated_tokens.push_back(tok);
                gen++;

                std::string s = tokenizer_.decode_token(tok);
                if (pathways_) pathways_->publish_stream_token(tok, s);
                if (on_token_callback && !on_token_callback(tok, s)) return;

                forward(tok, pos++, logits_scratch_.data());
            }
        }
    }
}

std::future<std::vector<uint32_t>> HavenEngine::generate_async(
    const std::vector<uint32_t>& prompt_tokens,
    int max_new_tokens,
    const std::function<bool(uint32_t token, const std::string& text)>& on_token_callback)
{
    return std::async(std::launch::async, [this, prompt_tokens, max_new_tokens, on_token_callback]() {
        std::vector<uint32_t> results;
        auto tracking_cb = [&](uint32_t tok, const std::string& text) -> bool {
            results.push_back(tok);
            if (on_token_callback) {
                return on_token_callback(tok, text);
            }
            return true;
        };
        this->generate(prompt_tokens, max_new_tokens, tracking_cb);
        return results;
    });
}

} // namespace haven

// ============================================================================
// C ABI Export Implementation for P/Invoke & Android JNI
// ============================================================================
HAVEN_API void* haven_create_engine() {
    return new haven::HavenEngine();
}

HAVEN_API bool haven_load_model(void* engine, const char* filepath) {
    if (!engine || !filepath) return false;
    return reinterpret_cast<haven::HavenEngine*>(engine)->load_model(filepath);
}

HAVEN_API void haven_inject_memory(void* engine, const char* concept_name, float weight, const float* embedding, int dim) {
    if (!engine || !concept_name || !embedding || dim <= 0 || dim > 1000000) return;
    std::vector<float> emb_vec(embedding, embedding + dim);
    reinterpret_cast<haven::HavenEngine*>(engine)->inject_memory(concept_name, weight, emb_vec);
}

HAVEN_API void haven_inject_vision_frame(void* engine, const float* patch_embeddings, int num_patches, uint32_t width, uint32_t height, float salience) {
    if (!engine || !patch_embeddings || num_patches <= 0 || num_patches > 10000000) return;
    std::vector<float> patches(patch_embeddings, patch_embeddings + num_patches);
    reinterpret_cast<haven::HavenEngine*>(engine)->inject_vision_frame(patches, width, height, salience);
}

HAVEN_API void haven_inject_audio_spectrum(void* engine, const float* mel_bands, int num_bands, float energy_rms, float centroid, bool is_speech) {
    if (!engine || !mel_bands || num_bands <= 0 || num_bands > 1000000) return;
    std::vector<float> bands(mel_bands, mel_bands + num_bands);
    reinterpret_cast<haven::HavenEngine*>(engine)->inject_audio_spectrum(bands, energy_rms, centroid, is_speech);
}

HAVEN_API void haven_clear_sensory_buffers(void* engine) {
    if (!engine) return;
    reinterpret_cast<haven::HavenEngine*>(engine)->clear_sensory_buffers();
}

HAVEN_API void haven_add_knowledge(void* engine, uint32_t token_id, uint32_t cluster_id, float boost, const char* label, const char* target) {
    if (!engine || !label || !target) return;
    reinterpret_cast<haven::HavenEngine*>(engine)->add_knowledge_relation(token_id, cluster_id, boost, label, target);
}

HAVEN_API size_t haven_prune_kv_cache(void* engine) {
    if (!engine) return 0;
    return reinterpret_cast<haven::HavenEngine*>(engine)->prune_kv_cache();
}

HAVEN_API float haven_get_attention_entropy(void* engine) {
    if (!engine) return 0.0f;
    return reinterpret_cast<haven::HavenEngine*>(engine)->get_current_attention_entropy();
}

HAVEN_API void haven_set_persona(void* engine, const float* persona_vector, int dim) {
    if (!engine || !persona_vector || dim <= 0 || dim > 1000000) return;
    std::vector<float> vec(persona_vector, persona_vector + dim);
    reinterpret_cast<haven::HavenEngine*>(engine)->set_persona_embedding(vec);
}

HAVEN_API void haven_set_yarn_scale(void* engine, float scale) {
    if (!engine) return;
    reinterpret_cast<haven::HavenEngine*>(engine)->set_yarn_context_scale(scale);
}

HAVEN_API uint64_t haven_fork_sequence(void* engine, uint64_t parent_id, uint64_t child_id) {
    if (!engine) return parent_id;
    return reinterpret_cast<haven::HavenEngine*>(engine)->fork_sequence(parent_id, child_id);
}

HAVEN_API float haven_get_speculative_rate(void* engine) {
    if (!engine) return 1.0f;
    return reinterpret_cast<haven::HavenEngine*>(engine)->get_speculative_acceptance_rate();
}

HAVEN_API void haven_forward(void* engine, uint32_t token, int pos, float* out_logits, uint32_t active_cluster) {
    if (!engine || !out_logits) return;
    reinterpret_cast<haven::HavenEngine*>(engine)->forward(token, pos, out_logits, active_cluster);
}

HAVEN_API uint32_t haven_sample_token(void* engine, float* logits, uint32_t vocab_size) {
    if (!engine || !logits || vocab_size == 0) return 0;
    static const std::vector<uint32_t> empty_history;
    return reinterpret_cast<haven::HavenEngine*>(engine)->get_sampler().sample(logits, vocab_size, empty_history);
}

HAVEN_API void haven_destroy_engine(void* engine) {
    if (engine) {
        delete reinterpret_cast<haven::HavenEngine*>(engine);
    }
}

HAVEN_API size_t haven_get_node_count(void* engine) {
    if (!engine) return 0;
    auto eng = reinterpret_cast<haven::HavenEngine*>(engine);
    auto top = eng->get_topology();
    return top ? top->get_nodes().size() : 0;
}

HAVEN_API bool haven_get_node_info(void* engine, uint32_t index, char* out_name, int max_name_len, uint32_t* out_backend, uint64_t* out_mem_free_mb) {
    if (!engine || !out_name || max_name_len <= 0) return false;
    auto eng = reinterpret_cast<haven::HavenEngine*>(engine);
    auto top = eng->get_topology();
    if (!top || index >= top->get_nodes().size()) return false;
    const auto& node = top->get_nodes()[index];
    std::strncpy(out_name, node.device_name.c_str(), max_name_len - 1);
    out_name[max_name_len - 1] = '\0';
    if (out_backend) *out_backend = static_cast<uint32_t>(node.backend);
    if (out_mem_free_mb) *out_mem_free_mb = node.free_memory_bytes / (1024 * 1024);
    return true;
}

HAVEN_API uint64_t haven_get_pathways_completed_tasks(void* engine) {
    if (!engine) return 0;
    auto eng = reinterpret_cast<haven::HavenEngine*>(engine);
    auto p = eng->get_pathways();
    return p ? p->get_completed_task_count() : 0;
}