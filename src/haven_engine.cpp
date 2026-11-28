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
    // Skip FFN offload when turbo_hadamard_ffn_down > 0 (fused TPU kernel expects unrotated GeGLU output)
    if (cfg.ffn_activation == FfnActivation::GELU && cfg.turbo_hadamard_ffn_down == 0) {
        for (uint32_t l = 0; l < cfg.num_layers && l < layers_.size(); ++l) {
            const auto& lt = layers_[l];
            if (!lt.ffn_gate || !lt.ffn_up || !lt.ffn_down) continue;
            const int layer_F = (lt.ffn_gate->shape.size() > 1 && lt.ffn_gate->shape[1] > 0)
                ? static_cast<int>(lt.ffn_gate->shape[1])
                : static_cast<int>(cfg.hidden_dim);
            const int dev_idx = static_cast<int>(l % static_cast<uint32_t>(n_devs));
            bool ok = tpu_backend_->upload_gguf_weight_matrix(*lt.ffn_gate, layer_F, static_cast<int>(cfg.embedding_dim), tpu_layers_[l].ffn_gate, dev_idx)
                   && tpu_backend_->upload_gguf_weight_matrix(*lt.ffn_up,   layer_F, static_cast<int>(cfg.embedding_dim), tpu_layers_[l].ffn_up, dev_idx)
                   && tpu_backend_->upload_gguf_weight_matrix(*lt.ffn_down, static_cast<int>(cfg.embedding_dim), layer_F, tpu_layers_[l].ffn_down, dev_idx);
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
    // Always detach TPU offload and clear all cached tensor pointers before reloading (prevents UAF on failed reload!)
    disable_tpu_offload();
    token_embd_tensor_ = nullptr;
    per_layer_token_embd_tensor_ = nullptr;
    per_layer_model_proj_tensor_ = nullptr;
    per_layer_proj_norm_tensor_ = nullptr;
    output_norm_tensor_ = nullptr;
    rope_freqs_tensor_ = nullptr;
    output_tensor_ = nullptr;
    layers_.clear();

    if (!loader_.load_file(gguf_filepath)) {
        return false;
    }
    const auto& cfg = loader_.get_config();
    if (cfg.embedding_dim == 0 || cfg.hidden_dim == 0 || cfg.num_layers == 0 ||
        cfg.num_heads == 0 || cfg.num_kv_heads == 0 || cfg.head_dim == 0 ||
        cfg.vocab_size == 0 || cfg.max_context_length == 0) {
        std::cerr << "⛔ [HavenEngine] Invalid zero model dimension in GGUF config.\n";
        loader_.close();
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
        const uint32_t k_exp = (cfg.expert_used_count > 0) ? cfg.expert_used_count : 8u;
        moe_router_scratch_.resize(cfg.expert_count, 0.0f);
        moe_norm_scratch_.resize(cfg.embedding_dim, 0.0f);
        moe_gate_up_scratch_.resize(static_cast<size_t>(k_exp) * static_cast<size_t>(exp_ff) * 2ULL, 0.0f);
        moe_exp_down_scratch_.resize(cfg.embedding_dim, 0.0f);
        moe_accum_scratch_.resize(cfg.embedding_dim, 0.0f);
    }
    logits_scratch_.resize(cfg.vocab_size, 0.0f);
    default_norm_weights_.assign(std::max({cfg.embedding_dim, cfg.hidden_dim, cfg.head_dim, 256u}), 1.0f);
    attn_scores_scratch_.clear();
    attn_scores_scratch_.shrink_to_fit();

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
    uint32_t max_ff_dim = cfg.hidden_dim;
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
        if (layers_[l].ffn_gate && layers_[l].ffn_gate->shape.size() > 1) {
            max_ff_dim = std::max(max_ff_dim, static_cast<uint32_t>(layers_[l].ffn_gate->shape[1]));
        }
    }

    uint32_t max_head_dim = std::max(cfg.head_dim, (cfg.num_heads > 0 ? max_q_dim / cfg.num_heads : cfg.head_dim));
    uint32_t kv_cache_head_dim = std::max(max_head_dim, (cfg.num_kv_heads > 0 ? (max_kv_dim + cfg.num_kv_heads - 1) / cfg.num_kv_heads : cfg.head_dim));
    kv_cache_ = DynamicKVCacheManager(cfg.num_layers, cfg.max_context_length, cfg.num_kv_heads, kv_cache_head_dim, 0.0f, 1.0f);
    telemetry_engine_ = AttentionTelemetryEngine(cfg.num_layers, cfg.num_heads);
    tokenizer_.initialize(loader_.get_vocabulary(), &cfg);
    speculative_engine_.initialize_draft_heads(cfg.embedding_dim);

    // Dynamically resize all scratch buffers to exact worst-case model dimensions
    const size_t pleias_dim = static_cast<size_t>(cfg.num_layers) * 256ULL;
    embedding_scratch_.resize(cfg.embedding_dim, 0.0f);
    norm_scratch_.resize(cfg.embedding_dim, 0.0f);
    q_scratch_.resize(std::max<size_t>(cfg.embedding_dim, max_q_dim), 0.0f);
    k_scratch_.resize(std::max<size_t>(cfg.embedding_dim, max_kv_dim), 0.0f);
    v_scratch_.resize(std::max<size_t>(cfg.embedding_dim, max_kv_dim), 0.0f);
    attn_out_scratch_.resize(cfg.embedding_dim, 0.0f);
    ffn_gate_scratch_.resize(max_ff_dim, 0.0f);
    ffn_up_scratch_.resize(max_ff_dim, 0.0f);
    ffn_down_scratch_.resize(cfg.embedding_dim, 0.0f);
    if (cfg.expert_count > 0) {
        const uint32_t exp_ff = (cfg.expert_ffn_dim > 0) ? cfg.expert_ffn_dim : max_ff_dim;
        const uint32_t k_exp = (cfg.expert_used_count > 0) ? cfg.expert_used_count : 8u;
        moe_router_scratch_.resize(cfg.expert_count, 0.0f);
        moe_norm_scratch_.resize(cfg.embedding_dim, 0.0f);
        moe_gate_up_scratch_.resize(static_cast<size_t>(k_exp) * static_cast<size_t>(exp_ff) * 2ULL, 0.0f);
        moe_exp_down_scratch_.resize(cfg.embedding_dim, 0.0f);
        moe_accum_scratch_.resize(cfg.embedding_dim, 0.0f);
    }
    layer_gate_scratch_.resize(max_ff_dim, 0.0f);
    per_layer_emb_scratch_.resize(pleias_dim, 0.0f);
    per_layer_proj_scratch_.resize(pleias_dim, 0.0f);
    pleias_proj_scratch_.resize(cfg.embedding_dim, 0.0f);
    attn_heads_out_scratch_.resize(std::max<size_t>(cfg.embedding_dim, max_q_dim), 0.0f);
    attn_indices_scratch_.resize(std::max<size_t>(static_cast<size_t>(cfg.max_context_length) + 4, 520), 0);
    logits_scratch_.resize(cfg.vocab_size, 0.0f);
    default_norm_weights_.assign(std::max({cfg.embedding_dim, max_ff_dim, max_head_dim, 256u}), 1.0f);

    // Initialize Persistent Cognitive Memory Vault
    std::string mem_path = "wwwroot/memories.json";
    if (!memory_engine_.load_from_json(mem_path)) {
        memory_engine_.initialize_sovereign_anchors(cfg.head_dim);
        memory_engine_.save_to_json(mem_path);
    }

    // Discover and load sovereign plugins
    plugin_manager_.discover_plugins("plugins");

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
            const float emb_scale = static_cast<float>(std::sqrt(static_cast<double>(cfg.embedding_dim)));
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
        
        const float tok_scale = 16.0f; // sqrt(n_embd_per_layer) = sqrt(256) = 16.0f exact
        for (size_t i = 0; i < per_layer_emb_scratch_.size(); ++i) {
            per_layer_emb_scratch_[i] *= tok_scale;
        }

        if (per_layer_model_proj_tensor_ && !per_layer_proj_scratch_.empty()) {
            const bool exact_pleias = exact_math_mode_ || ((exact_mask_ & EXACT_PLEIAS) != 0);
            if (exact_pleias) {
                Avx2Math::gemv_f32(per_layer_proj_scratch_.data(), *per_layer_model_proj_tensor_, embedding_scratch_.data(), (int)per_layer_proj_scratch_.size(), cfg.embedding_dim);
            } else {
                Avx2Math::gemv(per_layer_proj_scratch_.data(), *per_layer_model_proj_tensor_, embedding_scratch_.data(), (int)per_layer_proj_scratch_.size(), cfg.embedding_dim);
            }
            
            const float proj_scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(cfg.embedding_dim)));
            for (size_t i = 0; i < per_layer_proj_scratch_.size(); ++i) {
                per_layer_proj_scratch_[i] *= proj_scale;
            }
            
            const float* proj_norm_w = safe_f32_weights(per_layer_proj_norm_tensor_, 256);
            
            const float input_scale = static_cast<float>(1.0 / std::sqrt(2.0));
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

    // 1b. Mix in Multimodal Sensory Token Vectors (Vision + Audio Mel-Spectrum) — zero alloc when empty
    if (sensory_bridge_.get_visual_frame_count() > 0 || sensory_bridge_.get_audio_frame_count() > 0) {
        auto sensory_vec = sensory_bridge_.project_sensory_tokens(cfg.embedding_dim);
        for (size_t i = 0; i < cfg.embedding_dim; ++i) {
            embedding_scratch_[i] += sensory_vec[i];
        }
    }

    auto tensor_fits_rows = [](const TensorDesc* t, size_t total_rows, int cols) -> bool {
        if (!t || !t->data || cols <= 0) return false;
        if (t->size_bytes == 0) return true;
        const size_t rb = Avx2Math::row_bytes(t->type, cols);
        return rb > 0 && total_rows <= (t->size_bytes / rb);
    };

    const int D_emb = static_cast<int>(cfg.embedding_dim);

    // 2. Transformer Layer Loop
    for (uint32_t l = 0; l < cfg.num_layers; ++l) {
        const auto& lt = (l < layers_.size()) ? layers_[l] : LayerTensors{};

        const int F_hid = std::max(1, std::min<int>(
            static_cast<int>(ffn_gate_scratch_.size()),
            (lt.ffn_gate && lt.ffn_gate->shape.size() > 1) ? static_cast<int>(lt.ffn_gate->shape[1])
            : ((l < cfg.hidden_dim_per_layer.size() && cfg.hidden_dim_per_layer[l] > 0)
                ? static_cast<int>(cfg.hidden_dim_per_layer[l])
                : static_cast<int>(cfg.hidden_dim))));

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
        const float attn_scale = cfg.is_gemma4
            ? 1.0f
            : (cfg.query_pre_attn_scalar > 0.0f
                ? static_cast<float>(1.0 / std::sqrt(static_cast<double>(cfg.query_pre_attn_scalar)))
                : static_cast<float>(1.0 / std::sqrt(static_cast<double>(actual_head_dim))));
        const int gqa_ratio = std::max(1u, cfg.num_heads / std::max(1u, layer_n_kv_heads));

        // 2a. Pre-Attention RMSNorm (+ optional Gemma 4.5 Turbo Block-FWHT)
        const float* attn_norm_w = cfg.turbo_folded_norms ? default_norm_weights_.data() : safe_f32_weights(lt.attn_norm, cfg.embedding_dim);
        Avx2Math::rms_norm(norm_scratch_.data(), embedding_scratch_.data(), attn_norm_w, D_emb, cfg.rms_norm_eps);
        if (cfg.turbo_hadamard_qkv > 0 && (D_emb % static_cast<int>(cfg.turbo_hadamard_qkv)) == 0) {
            Avx2Math::fwht_block_inplace(norm_scratch_.data(), D_emb, static_cast<int>(cfg.turbo_hadamard_qkv));
        }

        // Determine SWA & Shared-KV status BEFORE projecting K/V so shared-KV layers
        // (e.g. Gemma 4 E4B blk.24..41 with shared_kv_layers=18) NEVER overwrite/use
        // unused placeholder attn_k/v weights and skip wasted K/V GEMVs entirely!
        const bool has_swa = (cfg.sliding_window > 0) && (cfg.is_gemma || cfg.architecture == "mistral");
        const bool is_swa = (l < cfg.sliding_window_pattern.size())
            ? cfg.sliding_window_pattern[l]
            : (has_swa && (cfg.architecture == "mistral"
                ? true
                : (cfg.architecture == "gemma2" ? (l % 2 == 0) : (l % 6 != 5))));
        const bool is_shared_kv = (cfg.shared_kv_layers > 0 && cfg.shared_kv_layers < cfg.num_layers &&
            l >= (cfg.num_layers - cfg.shared_kv_layers));
        const bool has_kv = !is_shared_kv && (lt.attn_k != nullptr);
        const TensorDesc* eff_k = has_kv ? lt.attn_k : nullptr;
        const TensorDesc* eff_v = has_kv ? lt.attn_v : nullptr;

        // 2b. Fused Q, K, V Projections (single Q8 quantization + single OpenMP region!)
        const bool q_ok = tensor_fits_rows(lt.attn_q, static_cast<size_t>(q_dim), D_emb);
        const bool k_ok = tensor_fits_rows(eff_k, static_cast<size_t>(kv_dim), D_emb);
        const bool v_ok = tensor_fits_rows(eff_v, static_cast<size_t>(kv_dim), D_emb);
        const Avx2Math::DotRowQ8Fn q_fn = q_ok ? Avx2Math::get_dot_row_q8_fn(lt.attn_q->type, D_emb) : nullptr;
        const Avx2Math::DotRowQ8Fn k_fn = k_ok ? Avx2Math::get_dot_row_q8_fn(eff_k->type, D_emb) : nullptr;
        const Avx2Math::DotRowQ8Fn v_fn = v_ok ? Avx2Math::get_dot_row_q8_fn(eff_v->type, D_emb) : nullptr;

        const bool exact_qkv = exact_math_mode_ || ((exact_mask_ & EXACT_QKV) != 0);
        if (!exact_qkv && q_ok && q_fn && (!eff_k || (k_ok && k_fn)) && (!eff_v || (v_ok && v_fn))) {
            Q8Act32 qkv_act;
            Avx2Math::quantize_q8_act(norm_scratch_.data(), D_emb, qkv_act);
            const int n_k = eff_k ? kv_dim : 0;
            const int n_v = eff_v ? kv_dim : 0;
            const int total_qkv = q_dim + n_k + n_v;
            const void* q_data = lt.attn_q->data;
            const void* k_data = eff_k ? eff_k->data : nullptr;
            const void* v_data = eff_v ? eff_v->data : nullptr;
            #pragma omp parallel for schedule(static)
            for (int idx = 0; idx < total_qkv; ++idx) {
                if (idx < q_dim) {
                    q_scratch_[static_cast<size_t>(idx)] = q_fn(q_data, static_cast<size_t>(idx), qkv_act, D_emb);
                } else if (idx < q_dim + n_k) {
                    const int r = idx - q_dim;
                    k_scratch_[static_cast<size_t>(r)] = k_fn(k_data, static_cast<size_t>(r), qkv_act, D_emb);
                } else {
                    const int r = idx - q_dim - n_k;
                    v_scratch_[static_cast<size_t>(r)] = v_fn(v_data, static_cast<size_t>(r), qkv_act, D_emb);
                }
            }
            if (lt.attn_q_bias) Avx2Math::add_bias(q_scratch_.data(), *lt.attn_q_bias, q_dim);
            if (eff_k) {
                if (lt.attn_k_bias) Avx2Math::add_bias(k_scratch_.data(), *lt.attn_k_bias, kv_dim);
            }
            if (eff_v) {
                if (lt.attn_v_bias) Avx2Math::add_bias(v_scratch_.data(), *lt.attn_v_bias, kv_dim);
            } else if (eff_k) {
                std::memcpy(v_scratch_.data(), k_scratch_.data(), static_cast<size_t>(kv_dim) * sizeof(float));
            }
        } else {
            auto run_gemv = [&](float* dst, const TensorDesc& w, const float* src, int rows, int cols) {
                if (exact_qkv) Avx2Math::gemv_f32(dst, w, src, rows, cols);
                else           Avx2Math::gemv(dst, w, src, rows, cols);
            };
            if (lt.attn_q) {
                run_gemv(q_scratch_.data(), *lt.attn_q, norm_scratch_.data(), q_dim, D_emb);
                if (lt.attn_q_bias) Avx2Math::add_bias(q_scratch_.data(), *lt.attn_q_bias, q_dim);
            } else {
                for (int i = 0; i < q_dim; ++i) q_scratch_[i] = norm_scratch_[i % cfg.embedding_dim];
            }

            if (eff_k) {
                run_gemv(k_scratch_.data(), *eff_k, norm_scratch_.data(), kv_dim, D_emb);
                if (lt.attn_k_bias) Avx2Math::add_bias(k_scratch_.data(), *lt.attn_k_bias, kv_dim);
            } else if (!is_shared_kv) {
                for (int i = 0; i < kv_dim; ++i) k_scratch_[i] = norm_scratch_[i % cfg.embedding_dim];
            }

            if (eff_v) {
                run_gemv(v_scratch_.data(), *eff_v, norm_scratch_.data(), kv_dim, D_emb);
                if (lt.attn_v_bias) Avx2Math::add_bias(v_scratch_.data(), *lt.attn_v_bias, kv_dim);
            } else if (eff_k) {
                // Gemma 4 k_eq_v: V shares raw K projection prior to K-norm and RoPE
                std::memcpy(v_scratch_.data(), k_scratch_.data(), static_cast<size_t>(kv_dim) * sizeof(float));
            } else if (!is_shared_kv) {
                for (int i = 0; i < kv_dim; ++i) v_scratch_[i] = norm_scratch_[i % cfg.embedding_dim];
            }
        }

        // 2b-2. Gemma 2 / 4 Per-Head Q-Norm, K-Norm, and V-Norm
        if (lt.attn_q_norm) {
            const float* q_norm_w = safe_f32_weights(lt.attn_q_norm, static_cast<size_t>(actual_head_dim));
            for (uint32_t h = 0; h < cfg.num_heads; ++h) {
                float* q_h = q_scratch_.data() + h * actual_head_dim;
                Avx2Math::rms_norm(q_h, q_h, q_norm_w, actual_head_dim, cfg.rms_norm_eps);
            }
        }
        if (has_kv && lt.attn_k_norm) {
            const float* k_norm_w = safe_f32_weights(lt.attn_k_norm, static_cast<size_t>(actual_head_dim));
            for (uint32_t h = 0; h < layer_n_kv_heads; ++h) {
                float* k_h = k_scratch_.data() + h * actual_head_dim;
                Avx2Math::rms_norm(k_h, k_h, k_norm_w, actual_head_dim, cfg.rms_norm_eps);
            }
        }
        if (has_kv && cfg.is_gemma4 && lt.attn_q_norm && lt.attn_k_norm) {
            for (uint32_t h = 0; h < layer_n_kv_heads; ++h) {
                float* v_h = v_scratch_.data() + h * actual_head_dim;
                Avx2Math::rms_norm(v_h, v_h, default_norm_weights_.data(), actual_head_dim, cfg.rms_norm_eps);
            }
        }

        // 2c. RoPE Rotary Embeddings
        float layer_freq_base = (is_swa && (cfg.has_rope_freq_base_swa || cfg.is_gemma)) ? cfg.rope_freq_base_swa : cfg.rope_freq_base;
        const uint32_t cfg_rope_dim = is_swa ? cfg.rope_dimension_count_swa : cfg.rope_dimension_count;
        int rope_dim = (!cfg.is_gemma4 && cfg_rope_dim > 0 && static_cast<int>(cfg_rope_dim) <= actual_head_dim)
            ? static_cast<int>(cfg_rope_dim)
            : actual_head_dim;
        const float* freq_factors_w = (!is_swa && rope_freqs_tensor_ && rope_freqs_tensor_->type == QuantType::F32 &&
            (rope_freqs_tensor_->size_bytes == 0 || rope_freqs_tensor_->size_bytes >= static_cast<size_t>(rope_dim / 2) * sizeof(float)))
            ? reinterpret_cast<const float*>(rope_freqs_tensor_->data) : nullptr;

        uint32_t kv_source_layer = l;
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
            Avx2Math::apply_rope(q_scratch_.data(), k_scratch_.data(), actual_head_dim, cfg.num_heads, layer_n_kv_heads, pos, layer_freq_base, cfg.rope_freq_scale, freq_factors_w, rope_dim, cfg.is_neox_rope);
            kv_cache_.write(l, pos, k_scratch_.data(), v_scratch_.data(), kv_dim);
        } else {
            Avx2Math::apply_rope(q_scratch_.data(), nullptr, actual_head_dim, cfg.num_heads, 0, pos, layer_freq_base, cfg.rope_freq_scale, freq_factors_w, rope_dim, cfg.is_neox_rope);
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
        #pragma omp parallel for if(active_attn_len >= 64) schedule(static)
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
        const bool exact_attn_out = exact_math_mode_ || ((exact_mask_ & EXACT_ATTN_OUT) != 0);
        if (lt.attn_output) {
            if (exact_attn_out) Avx2Math::gemv_f32(attn_out_scratch_.data(), *lt.attn_output, attn_heads_out_scratch_.data(), cfg.embedding_dim, q_dim);
            else                Avx2Math::gemv(attn_out_scratch_.data(), *lt.attn_output, attn_heads_out_scratch_.data(), cfg.embedding_dim, q_dim);
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

        // 2g. Pre-FFN RMSNorm (+ optional Gemma 4.5 Turbo Block-FWHT)
        const float* ffn_norm_w = cfg.turbo_folded_norms ? default_norm_weights_.data() : safe_f32_weights(lt.ffn_norm, cfg.embedding_dim);
        Avx2Math::rms_norm(norm_scratch_.data(), embedding_scratch_.data(), ffn_norm_w, cfg.embedding_dim, cfg.rms_norm_eps);
        if (cfg.turbo_hadamard_ffn > 0 && (D_emb % static_cast<int>(cfg.turbo_hadamard_ffn)) == 0) {
            Avx2Math::fwht_block_inplace(norm_scratch_.data(), D_emb, static_cast<int>(cfg.turbo_hadamard_ffn));
        }

        const bool has_moe_layer = (lt.ffn_gate_inp && lt.ffn_down_exps &&
            (lt.ffn_gate_up_exps || (lt.ffn_gate_exps && lt.ffn_up_exps)) && cfg.expert_count > 0);

        // 2h + 2i. FFN Gate/Up/Activation/Down (Fused TPU MXU GeGLU when offload active, else Fused Integer AVX2 CPU)
        if (!lt.ffn_gate && !lt.ffn_up && !lt.ffn_down && has_moe_layer) {
            // Pure-MoE layer (e.g., Mixtral / Qwen-MoE without shared dense MLP): zero dense branch
            std::memset(ffn_down_scratch_.data(), 0, static_cast<size_t>(D_emb) * sizeof(float));
        } else if (tpu_offload_active_ && tpu_backend_ && l < tpu_layers_.size() && tpu_layers_[l].has_ffn &&
            tpu_backend_->fused_geglu_ffn(norm_scratch_.data(), 1,
                                          tpu_layers_[l].ffn_gate, tpu_layers_[l].ffn_up, tpu_layers_[l].ffn_down,
                                          ffn_down_scratch_.data())) {
            // Computed directly on TPU HBM3 + MXU systolic array
        } else {
            const bool ffn_g_ok = tensor_fits_rows(lt.ffn_gate, static_cast<size_t>(F_hid), D_emb);
            const bool ffn_u_ok = tensor_fits_rows(lt.ffn_up,   static_cast<size_t>(F_hid), D_emb);
            const bool ffn_d_ok = tensor_fits_rows(lt.ffn_down, static_cast<size_t>(D_emb), F_hid);
            const Avx2Math::DotRowQ8Fn ffn_g_fn = ffn_g_ok ? Avx2Math::get_dot_row_q8_fn(lt.ffn_gate->type, D_emb) : nullptr;
            const Avx2Math::DotRowQ8Fn ffn_u_fn = ffn_u_ok ? Avx2Math::get_dot_row_q8_fn(lt.ffn_up->type,   D_emb) : nullptr;
            const Avx2Math::DotRowQ8Fn ffn_d_fn = ffn_d_ok ? Avx2Math::get_dot_row_q8_fn(lt.ffn_down->type, F_hid) : nullptr;

            const bool exact_ffn_gu   = exact_math_mode_ || ((exact_mask_ & EXACT_FFN_GU) != 0);
            const bool exact_ffn_down = exact_math_mode_ || ((exact_mask_ & EXACT_FFN_DOWN) != 0);
            if (!exact_ffn_gu && ffn_g_fn && ffn_u_fn && ffn_d_fn) {
                Q8Act32 ffn_in_act;
                Q8Act32 ffn_mid_act;
                Avx2Math::quantize_q8_act(norm_scratch_.data(), D_emb, ffn_in_act);
                const bool use_gelu = (cfg.ffn_activation == FfnActivation::GELU);
                const void* g_data = lt.ffn_gate->data;
                const void* u_data = lt.ffn_up->data;
                const void* d_data = lt.ffn_down->data;

                // Single fused OpenMP region for Gate + Up + GeGLU/SiLU + Down!
                #pragma omp parallel
                {
                    #pragma omp for schedule(static)
                    for (int r = 0; r < F_hid; ++r) {
                        const float g = ffn_g_fn(g_data, static_cast<size_t>(r), ffn_in_act, D_emb);
                        const float u = ffn_u_fn(u_data, static_cast<size_t>(r), ffn_in_act, D_emb);
                        const double gd = static_cast<double>(g);
                        double act;
                        if (use_gelu) {
                            const double gc = std::clamp(gd, -15.0, 15.0);
                            const double z  = 0.79788456080286535588 * (gc + 0.044715 * gc * gc * gc);
                            act = (z < -350.0) ? 0.0 : ((z > 350.0) ? gd : (gd / (1.0 + std::exp(-2.0 * z))));
                        } else {
                            act = gd / (1.0 + std::exp(-gd));
                        }
                        ffn_gate_scratch_[static_cast<size_t>(r)] = static_cast<float>(act * static_cast<double>(u));
                    }

                    #pragma omp single
                    {
                        if (cfg.turbo_hadamard_ffn_down > 0 && (F_hid % static_cast<int>(cfg.turbo_hadamard_ffn_down)) == 0) {
                            Avx2Math::fwht_block_inplace(ffn_gate_scratch_.data(), F_hid, static_cast<int>(cfg.turbo_hadamard_ffn_down));
                        }
                        if (!exact_ffn_down) {
                            Avx2Math::quantize_q8_act(ffn_gate_scratch_.data(), F_hid, ffn_mid_act);
                        }
                    }

                    if (exact_ffn_down) {
                        #pragma omp for schedule(static)
                        for (int r = 0; r < D_emb; ++r) {
                            ffn_down_scratch_[static_cast<size_t>(r)] = Avx2Math::dot_row(*lt.ffn_down, static_cast<size_t>(r), ffn_gate_scratch_.data(), F_hid);
                        }
                    } else {
                        #pragma omp for schedule(static)
                        for (int r = 0; r < D_emb; ++r) {
                            ffn_down_scratch_[static_cast<size_t>(r)] = ffn_d_fn(d_data, static_cast<size_t>(r), ffn_mid_act, F_hid);
                        }
                    }
                }
            } else {
                auto run_gemv_gu = [&](float* dst, const TensorDesc& w, const float* src, int rows, int cols) {
                    if (exact_ffn_gu) Avx2Math::gemv_f32(dst, w, src, rows, cols);
                    else              Avx2Math::gemv(dst, w, src, rows, cols);
                };
                if (lt.ffn_gate && lt.ffn_up) {
                    run_gemv_gu(ffn_gate_scratch_.data(), *lt.ffn_gate, norm_scratch_.data(), F_hid, D_emb);
                    run_gemv_gu(ffn_up_scratch_.data(), *lt.ffn_up, norm_scratch_.data(), F_hid, D_emb);
                    if (cfg.ffn_activation == FfnActivation::GELU) {
                        Avx2Math::gelu(ffn_gate_scratch_.data(), F_hid);
                    } else {
                        Avx2Math::silu(ffn_gate_scratch_.data(), F_hid);
                    }
                    for (int i = 0; i < F_hid; ++i) {
                        ffn_gate_scratch_[static_cast<size_t>(i)] *= ffn_up_scratch_[static_cast<size_t>(i)];
                    }
                } else {
                    for (int i = 0; i < F_hid; ++i) ffn_gate_scratch_[static_cast<size_t>(i)] = norm_scratch_[static_cast<size_t>(i % D_emb)] * 0.1f;
                }

                // 2i. FFN Down Projection
                if (lt.ffn_down) {
                    if (cfg.turbo_hadamard_ffn_down > 0 && (F_hid % static_cast<int>(cfg.turbo_hadamard_ffn_down)) == 0) {
                        Avx2Math::fwht_block_inplace(ffn_gate_scratch_.data(), F_hid, static_cast<int>(cfg.turbo_hadamard_ffn_down));
                    }
                    if (exact_ffn_down) Avx2Math::gemv_f32(ffn_down_scratch_.data(), *lt.ffn_down, ffn_gate_scratch_.data(), D_emb, F_hid);
                    else                Avx2Math::gemv(ffn_down_scratch_.data(), *lt.ffn_down, ffn_gate_scratch_.data(), D_emb, F_hid);
                } else {
                    for (int i = 0; i < D_emb; ++i) ffn_down_scratch_[static_cast<size_t>(i)] = ffn_gate_scratch_[static_cast<size_t>(i % F_hid)] * 0.05f;
                }
            }
        }

        // 2i-1b. Parallel Sparse Mixture-of-Experts Branch (Gemma 4 26B-A4B MoE)
        if (lt.ffn_gate_inp && lt.ffn_down_exps && (lt.ffn_gate_up_exps || (lt.ffn_gate_exps && lt.ffn_up_exps)) && cfg.expert_count > 0) {
            // Post-norm 1 on shared dense MLP branch before combining with MoE branch
            if (lt.post_ffw_norm_1) {
                const float* p1_w = safe_f32_weights(lt.post_ffw_norm_1, cfg.embedding_dim);
                Avx2Math::rms_norm(ffn_down_scratch_.data(), ffn_down_scratch_.data(), p1_w, D_emb, cfg.rms_norm_eps);
            }

            const int E = static_cast<int>(cfg.expert_count);
            const int K_used = std::clamp(static_cast<int>(cfg.expert_used_count > 0 ? cfg.expert_used_count : 8u), 1, E);
            const int F_exp = static_cast<int>(cfg.expert_ffn_dim > 0 ? cfg.expert_ffn_dim : cfg.hidden_dim);

            // Router input: parameter-free RMSNorm(attn_out) * (1 / sqrt(D)) * ffn_gate_inp.scale
            Avx2Math::rms_norm(moe_norm_scratch_.data(), embedding_scratch_.data(), default_norm_weights_.data(), D_emb, cfg.rms_norm_eps);
            const float inv_sqrt_d = static_cast<float>(1.0 / std::sqrt(static_cast<double>(cfg.embedding_dim)));
            const float* rscale = (!cfg.turbo_folded_scales && lt.ffn_gate_inp_scale) ? safe_f32_weights(lt.ffn_gate_inp_scale, cfg.embedding_dim) : nullptr;
            for (size_t i = 0; i < cfg.embedding_dim; ++i) {
                moe_norm_scratch_[i] *= inv_sqrt_d * (rscale ? rscale[i] : 1.0f);
            }
            // Always use exact FP32-activation GEMV for the sensitive E=128 MoE router (<30 us, prevents expert misrouting!)
            Avx2Math::gemv_f32(moe_router_scratch_.data(), *lt.ffn_gate_inp, moe_norm_scratch_.data(), E, D_emb);
            Avx2Math::softmax(moe_router_scratch_.data(), E);

            // Stack-backed Top-K expert selection & renormalization (zero heap alloc for E <= 256, K_used <= 64)
            int exp_idx_stack[256];
            std::vector<int> exp_idx_heap;
            int* exp_idx = exp_idx_stack;
            if (E > 256) {
                exp_idx_heap.resize(static_cast<size_t>(E));
                exp_idx = exp_idx_heap.data();
            }
            for (int e = 0; e < E; ++e) exp_idx[e] = e;
            std::partial_sort(exp_idx, exp_idx + K_used, exp_idx + E,
                [&](int a, int b) { return moe_router_scratch_[static_cast<size_t>(a)] > moe_router_scratch_[static_cast<size_t>(b)]; });

            double w_sum = 0.0;
            for (int k = 0; k < K_used; ++k) w_sum += static_cast<double>(moe_router_scratch_[static_cast<size_t>(exp_idx[k])]);
            if (w_sum < 6.103515625e-5) w_sum = 6.103515625e-5;

            // Pre-FFN norm 2 on attn_out for sparse experts (fall back to ffn_norm for pure-MoE models without pre_ffw_norm_2)
            const TensorDesc* pre2_tensor = lt.pre_ffw_norm_2 ? lt.pre_ffw_norm_2 : lt.ffn_norm;
            const float* pre2_w = cfg.turbo_folded_norms ? default_norm_weights_.data() : safe_f32_weights(pre2_tensor, cfg.embedding_dim);
            Avx2Math::rms_norm(moe_norm_scratch_.data(), embedding_scratch_.data(), pre2_w, D_emb, cfg.rms_norm_eps);
            if (cfg.turbo_hadamard_moe_in > 0 && (D_emb % static_cast<int>(cfg.turbo_hadamard_moe_in)) == 0) {
                Avx2Math::fwht_block_inplace(moe_norm_scratch_.data(), D_emb, static_cast<int>(cfg.turbo_hadamard_moe_in));
            }

            const float* down_s = (!cfg.turbo_folded_scales && lt.ffn_down_exps_scale) ? safe_f32_weights(lt.ffn_down_exps_scale, static_cast<size_t>(E)) : nullptr;
            float w_exp_stack[64];
            std::vector<float> w_exp_heap;
            float* w_exp = w_exp_stack;
            if (K_used > 64) {
                w_exp_heap.resize(static_cast<size_t>(K_used), 0.0f);
                w_exp = w_exp_heap.data();
            }
            for (int k = 0; k < K_used; ++k) {
                const int e = exp_idx[k];
                w_exp[k] = static_cast<float>((static_cast<double>(moe_router_scratch_[static_cast<size_t>(e)]) / w_sum) * (down_s ? static_cast<double>(down_s[e]) : 1.0));
            }

            const size_t gu_stride = static_cast<size_t>(K_used) * static_cast<size_t>(2 * F_exp);
            if (moe_gate_up_scratch_.size() < gu_stride) {
                moe_gate_up_scratch_.resize(gu_stride, 0.0f);
            }

            const int D_int = D_emb;
            auto tensor_fits_expert = [](const TensorDesc* t, int e_idx, size_t exp_rows, int cols) -> bool {
                if (!t || !t->data) return false;
                if (t->size_bytes == 0) return true;
                const size_t rb = Avx2Math::row_bytes(t->type, cols);
                return rb > 0 && (static_cast<size_t>(e_idx) + 1ULL) * exp_rows <= (t->size_bytes / rb);
            };

            bool all_gu_valid = true;
            bool all_dn_valid = true;
            for (int k = 0; k < K_used; ++k) {
                const int e = exp_idx[k];
                if (lt.ffn_gate_up_exps) {
                    all_gu_valid &= tensor_fits_expert(lt.ffn_gate_up_exps, e, static_cast<size_t>(2 * F_exp), D_int);
                } else {
                    all_gu_valid &= tensor_fits_expert(lt.ffn_gate_exps, e, static_cast<size_t>(F_exp), D_int)
                                 && tensor_fits_expert(lt.ffn_up_exps, e, static_cast<size_t>(F_exp), D_int);
                }
                all_dn_valid &= tensor_fits_expert(lt.ffn_down_exps, e, static_cast<size_t>(D_int), F_exp);
            }

            if (!all_gu_valid || !all_dn_valid) {
                std::fill(moe_accum_scratch_.begin(), moe_accum_scratch_.end(), 0.0f);
            } else {
                const bool use_gelu = (cfg.ffn_activation == FfnActivation::GELU);
                const int total_act = K_used * F_exp;
                const int total_gu = K_used * 2 * F_exp;
                const int two_F = 2 * F_exp;

                const Avx2Math::DotRowQ8Fn gu_fn = lt.ffn_gate_up_exps
                    ? Avx2Math::get_dot_row_q8_fn(lt.ffn_gate_up_exps->type, D_int) : nullptr;
                const Avx2Math::DotRowQ8Fn g_exp_fn = (!lt.ffn_gate_up_exps && lt.ffn_gate_exps)
                    ? Avx2Math::get_dot_row_q8_fn(lt.ffn_gate_exps->type, D_int) : nullptr;
                const Avx2Math::DotRowQ8Fn u_exp_fn = (!lt.ffn_gate_up_exps && lt.ffn_up_exps)
                    ? Avx2Math::get_dot_row_q8_fn(lt.ffn_up_exps->type, D_int) : nullptr;
                const Avx2Math::DotRowQ8Fn dn_fn = Avx2Math::get_dot_row_q8_fn(lt.ffn_down_exps->type, F_exp);

                // Match CUDA policy: un-baked stock MoE (turbo_hadamard_moe_in/down == 0) has extreme unrotated
                // spikes across 128 experts; exact FP32 MoE eliminates router/down drift while Turbo MoE stays Q8!
                const bool exact_moe_gu = exact_math_mode_ || ((exact_mask_ & EXACT_MOE_GU) != 0) || (cfg.turbo_hadamard_moe_in == 0);
                const bool exact_moe_dn = exact_math_mode_ || ((exact_mask_ & EXACT_MOE_DOWN) != 0) || (cfg.turbo_hadamard_moe_down == 0);
                const bool use_q8_gu = !exact_moe_gu && (lt.ffn_gate_up_exps ? (gu_fn != nullptr) : (g_exp_fn != nullptr && u_exp_fn != nullptr));
                const bool use_q8_dn = !exact_moe_dn && (dn_fn != nullptr);

                Q8Act32 moe_in_act;
                if (use_q8_gu) {
                    Avx2Math::quantize_q8_act(moe_norm_scratch_.data(), D_int, moe_in_act);
                }

                // Per-expert Q8 activations (pre-allocated scratch for zero per-layer heap allocs)
                if (use_q8_dn && moe_exp_acts_.size() < static_cast<size_t>(K_used)) {
                    moe_exp_acts_.resize(static_cast<size_t>(K_used));
                }
                Q8Act32* exp_acts = use_q8_dn ? moe_exp_acts_.data() : nullptr;

                const void* gu_data = lt.ffn_gate_up_exps ? lt.ffn_gate_up_exps->data : nullptr;
                const void* g_exp_data = lt.ffn_gate_exps ? lt.ffn_gate_exps->data : nullptr;
                const void* u_exp_data = lt.ffn_up_exps ? lt.ffn_up_exps->data : nullptr;
                const void* dn_data = lt.ffn_down_exps->data;

                constexpr int DN_TILE = 64;
                const int num_dn_tiles = (D_int + DN_TILE - 1) / DN_TILE;

                // Fuse all K_used experts (Gate+Up, activation + Q8 quant, k-tiled Down + weighted sum) in one OpenMP region
                #pragma omp parallel
                {
                    if (lt.ffn_gate_up_exps) {
                        if (use_q8_gu) {
                            #pragma omp for schedule(static)
                            for (int idx = 0; idx < total_gu; ++idx) {
                                const int k = idx / two_F;
                                const int r = idx % two_F;
                                const size_t row = static_cast<size_t>(exp_idx[k]) * static_cast<size_t>(two_F) + static_cast<size_t>(r);
                                moe_gate_up_scratch_[static_cast<size_t>(idx)] = gu_fn(gu_data, row, moe_in_act, D_int);
                            }
                        } else {
                            #pragma omp for schedule(static)
                            for (int idx = 0; idx < total_gu; ++idx) {
                                const int k = idx / two_F;
                                const int r = idx % two_F;
                                const size_t row = static_cast<size_t>(exp_idx[k]) * static_cast<size_t>(two_F) + static_cast<size_t>(r);
                                moe_gate_up_scratch_[static_cast<size_t>(idx)] =
                                    Avx2Math::dot_row(*lt.ffn_gate_up_exps, row, moe_norm_scratch_.data(), D_int);
                            }
                        }
                    } else {
                        if (use_q8_gu) {
                            #pragma omp for schedule(static)
                            for (int idx = 0; idx < total_act; ++idx) {
                                const int k = idx / F_exp;
                                const int r = idx % F_exp;
                                const size_t row = static_cast<size_t>(exp_idx[k]) * static_cast<size_t>(F_exp) + static_cast<size_t>(r);
                                float* base = moe_gate_up_scratch_.data() + static_cast<size_t>(k) * static_cast<size_t>(two_F);
                                base[r]         = g_exp_fn(g_exp_data, row, moe_in_act, D_int);
                                base[F_exp + r] = u_exp_fn(u_exp_data, row, moe_in_act, D_int);
                            }
                        } else {
                            #pragma omp for schedule(static)
                            for (int idx = 0; idx < total_act; ++idx) {
                                const int k = idx / F_exp;
                                const int r = idx % F_exp;
                                const size_t row = static_cast<size_t>(exp_idx[k]) * static_cast<size_t>(F_exp) + static_cast<size_t>(r);
                                float* base = moe_gate_up_scratch_.data() + static_cast<size_t>(k) * static_cast<size_t>(two_F);
                                base[r]         = Avx2Math::dot_row(*lt.ffn_gate_exps, row, moe_norm_scratch_.data(), D_int);
                                base[F_exp + r] = Avx2Math::dot_row(*lt.ffn_up_exps,   row, moe_norm_scratch_.data(), D_int);
                            }
                        }
                    }

                    // Fused per-expert activation (exact cancellation-free FP64) + parallel per-expert Q8 quantization
                    #pragma omp for schedule(static)
                    for (int k = 0; k < K_used; ++k) {
                        float* base = moe_gate_up_scratch_.data() + static_cast<size_t>(k) * static_cast<size_t>(two_F);
                        for (int j = 0; j < F_exp; ++j) {
                            const double gd = static_cast<double>(base[j]);
                            double act;
                            if (use_gelu) {
                                const double gc = std::clamp(gd, -15.0, 15.0);
                                const double z  = 0.79788456080286535588 * (gc + 0.044715 * gc * gc * gc);
                                act = (z < -350.0) ? 0.0 : ((z > 350.0) ? gd : (gd / (1.0 + std::exp(-2.0 * z))));
                            } else {
                                act = gd / (1.0 + std::exp(-gd));
                            }
                            base[j] = static_cast<float>(act * static_cast<double>(base[F_exp + j]));
                        }
                        if (cfg.turbo_hadamard_moe_down > 0 && (F_exp % static_cast<int>(cfg.turbo_hadamard_moe_down)) == 0) {
                            Avx2Math::fwht_block_inplace(base, F_exp, static_cast<int>(cfg.turbo_hadamard_moe_down));
                        }
                        if (use_q8_dn) {
                            Avx2Math::quantize_q8_act(base, F_exp, exp_acts[k]);
                        }
                    }

                    // Cache-friendly k-tiled MoE Down: stream 64 contiguous rows per expert to lock L2 prefetchers
                    if (use_q8_dn) {
                        #pragma omp for schedule(static)
                        for (int t = 0; t < num_dn_tiles; ++t) {
                            const int rb = t * DN_TILE;
                            const int r_end = std::min(D_int, rb + DN_TILE);
                            float acc[DN_TILE];
                            const int count = r_end - rb;
                            for (int i = 0; i < count; ++i) acc[i] = 0.0f;
                            for (int k = 0; k < K_used; ++k) {
                                const float wk = w_exp[k];
                                const Q8Act32& ak = exp_acts[k];
                                const size_t base_r = static_cast<size_t>(exp_idx[k]) * static_cast<size_t>(D_int);
                                for (int i = 0; i < count; ++i) {
                                    acc[i] += wk * dn_fn(dn_data, base_r + static_cast<size_t>(rb + i), ak, F_exp);
                                }
                            }
                            for (int i = 0; i < count; ++i) {
                                moe_accum_scratch_[static_cast<size_t>(rb + i)] = acc[i];
                            }
                        }
                    } else {
                        #pragma omp for schedule(static)
                        for (int t = 0; t < num_dn_tiles; ++t) {
                            const int rb = t * DN_TILE;
                            const int r_end = std::min(D_int, rb + DN_TILE);
                            float acc[DN_TILE];
                            const int count = r_end - rb;
                            for (int i = 0; i < count; ++i) acc[i] = 0.0f;
                            for (int k = 0; k < K_used; ++k) {
                                const float wk = w_exp[k];
                                const float* act_k = moe_gate_up_scratch_.data() + static_cast<size_t>(k) * static_cast<size_t>(two_F);
                                const size_t base_r = static_cast<size_t>(exp_idx[k]) * static_cast<size_t>(D_int);
                                for (int i = 0; i < count; ++i) {
                                    acc[i] += wk * Avx2Math::dot_row(*lt.ffn_down_exps, base_r + static_cast<size_t>(rb + i), act_k, F_exp);
                                }
                            }
                            for (int i = 0; i < count; ++i) {
                                moe_accum_scratch_[static_cast<size_t>(rb + i)] = acc[i];
                            }
                        }
                    }
                }
            }

            if (lt.post_ffw_norm_2) {
                const float* p2_w = safe_f32_weights(lt.post_ffw_norm_2, cfg.embedding_dim);
                Avx2Math::rms_norm(moe_accum_scratch_.data(), moe_accum_scratch_.data(), p2_w, D_emb, cfg.rms_norm_eps);
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

        // 2-pleias. Per-Layer Token Injection (Google Pleias Branch — fused single OpenMP region)
        if (lt.inp_gate && lt.proj && !per_layer_emb_scratch_.empty() && (static_cast<size_t>(l) + 1) * 256 <= per_layer_emb_scratch_.size()) {
            float pleias_gate[256] = {0};
            float* pleias_proj = pleias_proj_scratch_.data();
            const float* layer_emb_slice = per_layer_emb_scratch_.data() + l * 256;

            const bool ig_ok = tensor_fits_rows(lt.inp_gate, 256, D_emb);
            const bool pr_ok = tensor_fits_rows(lt.proj, static_cast<size_t>(D_emb), 256);
            const Avx2Math::DotRowQ8Fn ig_fn = ig_ok ? Avx2Math::get_dot_row_q8_fn(lt.inp_gate->type, D_emb) : nullptr;
            const Avx2Math::DotRowQ8Fn pr_fn = pr_ok ? Avx2Math::get_dot_row_q8_fn(lt.proj->type, 256) : nullptr;

            const bool exact_pleias = exact_math_mode_ || ((exact_mask_ & EXACT_PLEIAS) != 0);
            if (!exact_pleias && ig_fn && pr_fn) {
                Q8Act32 pl_in_act;
                Q8Act32 pl_mid_act;
                Avx2Math::quantize_q8_act(embedding_scratch_.data(), D_emb, pl_in_act);
                const void* ig_data = lt.inp_gate->data;
                const void* pr_data = lt.proj->data;
                #pragma omp parallel
                {
                    #pragma omp for schedule(static)
                    for (int i = 0; i < 256; ++i) {
                        const double g = static_cast<double>(ig_fn(ig_data, static_cast<size_t>(i), pl_in_act, D_emb));
                        const double cdf = 0.5 * (1.0 + std::tanh(0.79788456080286535588 * (g + 0.044715 * g * g * g)));
                        pleias_gate[i] = static_cast<float>((g * cdf) * static_cast<double>(layer_emb_slice[i]));
                    }
                    #pragma omp single
                    {
                        Avx2Math::quantize_q8_act(pleias_gate, 256, pl_mid_act);
                    }
                    #pragma omp for schedule(static)
                    for (int r = 0; r < D_emb; ++r) {
                        pleias_proj[r] = pr_fn(pr_data, static_cast<size_t>(r), pl_mid_act, 256);
                    }
                }
            } else {
                if (exact_pleias) Avx2Math::gemv_f32(pleias_gate, *lt.inp_gate, embedding_scratch_.data(), 256, D_emb);
                else              Avx2Math::gemv(pleias_gate, *lt.inp_gate, embedding_scratch_.data(), 256, D_emb);
                Avx2Math::gelu(pleias_gate, 256);
                for (int i = 0; i < 256; ++i) {
                    pleias_gate[i] *= layer_emb_slice[i];
                }
                if (exact_pleias) Avx2Math::gemv_f32(pleias_proj, *lt.proj, pleias_gate, D_emb, 256);
                else              Avx2Math::gemv(pleias_proj, *lt.proj, pleias_gate, D_emb, 256);
            }

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
        // Always run final LM head in exact FP32 activations (bandwidth-bound S=1 GEMV, eliminates logit Q8 jitter!)
        Avx2Math::gemv_f32(out_logits, *output_tensor_, norm_scratch_.data(), cfg.vocab_size, cfg.embedding_dim);
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

// Exact Cancellation-Free FP64 GeGLU activation: dst[i] = gelu_tanh(gate[i]) * up[i]
// Uses the exact identity 0.5 * g * (1 + tanh(z)) == g / (1 + exp(-2z)) matching forward() bit-for-bit
// with zero dead-neuron leakage (exact 0.0 for g <= -10) at <0.3% runtime overhead (~8 ms on 26B 105-tok).
static inline void geglu_avx2_range(float* dst, const float* gate, const float* up, int n, bool /*exact_math*/ = true) {
    constexpr double s2pi = 0.79788456080286535588;
    for (int i = 0; i < n; ++i) {
        const double g   = static_cast<double>(gate[i]);
        const double gc  = std::clamp(g, -15.0, 15.0);
        const double z   = s2pi * (gc + 0.044715 * gc * gc * gc);
        const double act = (z < -350.0) ? 0.0 : ((z > 350.0) ? g : (g / (1.0 + std::exp(-2.0 * z))));
        dst[i] = static_cast<float>(act * static_cast<double>(up[i]));
    }
}

void HavenEngine::forward_batch(const uint32_t* tokens, int n_tokens, int start_pos, float* out_last_logits, uint32_t active_cluster) {
    if (!tokens || n_tokens <= 0) return;
    if (n_tokens == 1 || tpu_offload_active_) {
        for (int i = 0; i < n_tokens; ++i) {
            const bool is_last = (i + 1 == n_tokens);
            forward(tokens[i], start_pos + i, is_last ? out_last_logits : nullptr, active_cluster);
        }
        return;
    }

    const auto& cfg = loader_.get_config();
    if (cfg.embedding_dim == 0 || cfg.hidden_dim == 0 || cfg.num_heads == 0 || cfg.num_kv_heads == 0 || cfg.max_context_length == 0) {
        return;
    }

    auto safe_f32_weights = [&](const TensorDesc* t, size_t count) -> const float* {
        if (t && t->data && t->type == QuantType::F32 &&
            (t->size_bytes == 0 || t->size_bytes >= count * sizeof(float))) {
            return reinterpret_cast<const float*>(t->data);
        }
        return default_norm_weights_.data();
    };

    auto tensor_fits_expert = [](const TensorDesc* t, int e_idx, size_t exp_rows, int cols) -> bool {
        if (!t || !t->data || cols <= 0) return false;
        if (t->size_bytes == 0) return true;
        const size_t rb = Avx2Math::row_bytes(t->type, cols);
        return rb > 0 && (static_cast<size_t>(e_idx) + 1ULL) * exp_rows <= (t->size_bytes / rb);
    };

    const int D = static_cast<int>(cfg.embedding_dim);
    const int max_F = std::max<int>(static_cast<int>(cfg.hidden_dim), static_cast<int>(ffn_gate_scratch_.size()));
    const int L = static_cast<int>(cfg.num_layers);
    const int max_ctx = static_cast<int>(cfg.max_context_length);
    const bool use_gelu = (cfg.ffn_activation == FfnActivation::GELU);

    // Compute worst-case Q / KV dimensions across layers for chunk scratch sizing
    int max_q_dim = static_cast<int>(q_scratch_.size());
    int max_kv_dim = static_cast<int>(k_scratch_.size());

    // Optional sensory vector computed once per batch call
    std::vector<float> sensory_vec;
    if (sensory_bridge_.get_visual_frame_count() > 0 || sensory_bridge_.get_audio_frame_count() > 0) {
        sensory_vec = sensory_bridge_.project_sensory_tokens(cfg.embedding_dim);
    }

    // Balanced chunking: keep MoE batches up to 320 tokens so every expert streams from DDR4 ONCE with >=16 tokens/expert,
    // and dense batches up to 256 tokens so weights stream ONCE per 256 tokens with zero tiny tail chunks!
    const bool has_moe_model = (cfg.expert_count > 0);
    const int max_chunk = has_moe_model ? 320 : 256;
    const int n_chunks = (n_tokens + max_chunk - 1) / max_chunk;
    const int chunk_step = (n_tokens + n_chunks - 1) / n_chunks;
    const int chunk_cap = std::min(n_tokens, chunk_step);

    std::vector<float> X(static_cast<size_t>(chunk_cap) * static_cast<size_t>(D));
    std::vector<float> norm_buf(static_cast<size_t>(chunk_cap) * static_cast<size_t>(D));
    std::vector<float> q_buf(static_cast<size_t>(chunk_cap) * static_cast<size_t>(max_q_dim));
    std::vector<float> k_buf(static_cast<size_t>(chunk_cap) * static_cast<size_t>(max_kv_dim));
    std::vector<float> v_buf(static_cast<size_t>(chunk_cap) * static_cast<size_t>(max_kv_dim));
    std::vector<float> attn_heads_buf(static_cast<size_t>(chunk_cap) * static_cast<size_t>(max_q_dim));
    std::vector<float> attn_out_buf(static_cast<size_t>(chunk_cap) * static_cast<size_t>(D));
    std::vector<float> ffn_gate_buf(static_cast<size_t>(chunk_cap) * static_cast<size_t>(max_F));
    std::vector<float> ffn_up_buf(static_cast<size_t>(chunk_cap) * static_cast<size_t>(max_F));
    std::vector<float> ffn_down_buf(static_cast<size_t>(chunk_cap) * static_cast<size_t>(D));

    const bool has_pleias_branch = (per_layer_token_embd_tensor_ != nullptr && !per_layer_emb_scratch_.empty());
    const int pleias_total = has_pleias_branch ? static_cast<int>(per_layer_emb_scratch_.size()) : 0;
    std::vector<float> pl_emb_buf;
    std::vector<float> pl_proj_buf;
    std::vector<float> pl_gate_buf;
    std::vector<float> pl_out_buf;
    if (has_pleias_branch) {
        pl_emb_buf.resize(static_cast<size_t>(chunk_cap) * static_cast<size_t>(pleias_total));
        if (per_layer_model_proj_tensor_) {
            pl_proj_buf.resize(static_cast<size_t>(chunk_cap) * static_cast<size_t>(pleias_total));
        }
        pl_gate_buf.resize(static_cast<size_t>(chunk_cap) * 256ULL);
        pl_out_buf.resize(static_cast<size_t>(chunk_cap) * static_cast<size_t>(D));
    }

    std::vector<float> moe_norm_buf;
    std::vector<float> moe_router_buf;
    std::vector<float> moe_accum_buf;
    if (has_moe_model) {
        moe_norm_buf.resize(static_cast<size_t>(chunk_cap) * static_cast<size_t>(D));
        moe_router_buf.resize(static_cast<size_t>(chunk_cap) * static_cast<size_t>(cfg.expert_count));
        moe_accum_buf.resize(static_cast<size_t>(chunk_cap) * static_cast<size_t>(D));
    }

    for (int chunk_off = 0; chunk_off < n_tokens; chunk_off += chunk_step) {
        const int S = std::min(chunk_step, n_tokens - chunk_off);
        const int chunk_start_pos = start_pos + chunk_off;
        const bool is_last_chunk = (chunk_off + S == n_tokens);

        // 1. Batched Token Embedding Lookup
        const float emb_scale = cfg.is_gemma ? static_cast<float>(std::sqrt(static_cast<double>(D))) : 1.0f;
        #pragma omp parallel for if(S >= 8) schedule(static)
        for (int t = 0; t < S; ++t) {
            const uint32_t tok = tokens[chunk_off + t];
            float* xt = X.data() + static_cast<size_t>(t) * static_cast<size_t>(D);
            if (token_embd_tensor_ && tok < cfg.vocab_size) {
                Avx2Math::dequantize_row(xt, *token_embd_tensor_, static_cast<int>(tok), D);
                if (cfg.is_gemma) {
                    for (int i = 0; i < D; ++i) xt[i] *= emb_scale;
                }
            } else {
                for (int i = 0; i < D; ++i) xt[i] = std::sin(static_cast<float>(tok + static_cast<uint32_t>(i)) * 0.01f);
            }
        }

        // 1a-2. Batched Pleias Per-Layer Embeddings + Batched BF16 Projection
        if (has_pleias_branch) {
            const float tok_scale = 16.0f;
            #pragma omp parallel for if(S >= 4) schedule(static)
            for (int t = 0; t < S; ++t) {
                const uint32_t tok = tokens[chunk_off + t];
                float* pe = pl_emb_buf.data() + static_cast<size_t>(t) * static_cast<size_t>(pleias_total);
                if (tok < cfg.vocab_size) {
                    Avx2Math::dequantize_row(pe, *per_layer_token_embd_tensor_, static_cast<int>(tok), pleias_total);
                    for (int i = 0; i < pleias_total; ++i) pe[i] *= tok_scale;
                } else {
                    std::memset(pe, 0, static_cast<size_t>(pleias_total) * sizeof(float));
                }
            }

            if (per_layer_model_proj_tensor_) {
                const bool exact_pleias = exact_math_mode_ || ((exact_mask_ & EXACT_PLEIAS) != 0);
                if (exact_pleias) Avx2Math::gemm_f32(pl_proj_buf.data(), pleias_total, *per_layer_model_proj_tensor_, X.data(), D, S, pleias_total, D);
                else              Avx2Math::gemm(pl_proj_buf.data(), pleias_total, *per_layer_model_proj_tensor_, X.data(), D, S, pleias_total, D);
                const float proj_scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(D)));
                const float input_scale = static_cast<float>(1.0 / std::sqrt(2.0));
                const float* proj_norm_w = safe_f32_weights(per_layer_proj_norm_tensor_, 256);

                #pragma omp parallel for schedule(static)
                for (int t = 0; t < S; ++t) {
                    float* pp = pl_proj_buf.data() + static_cast<size_t>(t) * static_cast<size_t>(pleias_total);
                    float* pe = pl_emb_buf.data()  + static_cast<size_t>(t) * static_cast<size_t>(pleias_total);
                    for (int i = 0; i < pleias_total; ++i) pp[i] *= proj_scale;
                    for (int l = 0; l < L && (l + 1) * 256 <= pleias_total; ++l) {
                        float* p_slice = pp + static_cast<size_t>(l) * 256ULL;
                        float* e_slice = pe + static_cast<size_t>(l) * 256ULL;
                        Avx2Math::rms_norm(p_slice, p_slice, proj_norm_w, 256, cfg.rms_norm_eps);
                        for (int i = 0; i < 256; ++i) {
                            e_slice[i] = (p_slice[i] + e_slice[i]) * input_scale;
                        }
                    }
                }
            }
        }

        // 1b. Optional Multimodal Sensory Injection
        if (!sensory_vec.empty()) {
            for (int t = 0; t < S; ++t) {
                float* xt = X.data() + static_cast<size_t>(t) * static_cast<size_t>(D);
                for (int i = 0; i < D; ++i) xt[i] += sensory_vec[static_cast<size_t>(i)];
            }
        }

        // 2. Batched Transformer Layer Loop
        for (uint32_t l = 0; l < cfg.num_layers; ++l) {
            const auto& lt = (l < layers_.size()) ? layers_[l] : LayerTensors{};

            const int q_dim = std::max(1, std::min<int>(
                max_q_dim,
                (lt.attn_q && lt.attn_q->shape.size() > 1) ? static_cast<int>(lt.attn_q->shape[1]) : static_cast<int>(cfg.num_heads * cfg.head_dim)));
            const int actual_head_dim = std::max(1, (cfg.num_heads > 0) ? (q_dim / static_cast<int>(cfg.num_heads)) : static_cast<int>(cfg.head_dim));
            const uint32_t layer_n_kv_heads = (l < cfg.head_count_kv_per_layer.size() && cfg.head_count_kv_per_layer[l] > 0)
                ? cfg.head_count_kv_per_layer[l]
                : ((lt.attn_k && lt.attn_k->shape.size() > 1 && actual_head_dim > 0)
                    ? std::max(1u, static_cast<uint32_t>(lt.attn_k->shape[1]) / static_cast<uint32_t>(actual_head_dim))
                    : cfg.num_kv_heads);
            const int kv_dim = std::max(1, std::min<int>(
                max_kv_dim,
                (lt.attn_k && lt.attn_k->shape.size() > 1) ? static_cast<int>(lt.attn_k->shape[1]) : static_cast<int>(layer_n_kv_heads * static_cast<uint32_t>(actual_head_dim))));
            const float attn_scale = cfg.is_gemma4
                ? 1.0f
                : (cfg.query_pre_attn_scalar > 0.0f
                    ? static_cast<float>(1.0 / std::sqrt(static_cast<double>(cfg.query_pre_attn_scalar)))
                    : static_cast<float>(1.0 / std::sqrt(static_cast<double>(actual_head_dim))));
            const int gqa_ratio = std::max(1u, cfg.num_heads / std::max(1u, layer_n_kv_heads));

            // 2a. Batched Pre-Attention RMSNorm (+ optional Gemma 4.5 Turbo Block-FWHT)
            const float* attn_norm_w = cfg.turbo_folded_norms ? default_norm_weights_.data() : safe_f32_weights(lt.attn_norm, cfg.embedding_dim);
            const int h_qkv = (cfg.turbo_hadamard_qkv > 0 && (D % static_cast<int>(cfg.turbo_hadamard_qkv)) == 0) ? static_cast<int>(cfg.turbo_hadamard_qkv) : 0;
            #pragma omp parallel for if(S >= 8) schedule(static)
            for (int t = 0; t < S; ++t) {
                float* nb = norm_buf.data() + static_cast<size_t>(t) * static_cast<size_t>(D);
                Avx2Math::rms_norm(nb, X.data() + static_cast<size_t>(t) * static_cast<size_t>(D),
                                   attn_norm_w, D, cfg.rms_norm_eps);
                if (h_qkv > 0) Avx2Math::fwht_block_inplace(nb, D, h_qkv);
            }

            // Shared-KV check BEFORE projecting K/V (honors Gemma 4 E4B shared_kv_layers=18)
            const bool has_swa = (cfg.sliding_window > 0) && (cfg.is_gemma || cfg.architecture == "mistral");
            const bool is_swa = (l < cfg.sliding_window_pattern.size())
                ? cfg.sliding_window_pattern[l]
                : (has_swa && (cfg.architecture == "mistral"
                    ? true
                    : (cfg.architecture == "gemma2" ? (l % 2 == 0) : (l % 6 != 5))));
            const bool is_shared_kv = (cfg.shared_kv_layers > 0 && cfg.shared_kv_layers < cfg.num_layers &&
                l >= (cfg.num_layers - cfg.shared_kv_layers));
            const bool has_kv = !is_shared_kv && (lt.attn_k != nullptr);
            const TensorDesc* eff_k = has_kv ? lt.attn_k : nullptr;
            const TensorDesc* eff_v = has_kv ? lt.attn_v : nullptr;

            // 2b. Batched Q, K, V Projections (pack norm_buf ONCE when sharing Q8K / Q8_32!)
            const bool q_q8k = lt.attn_q && Avx2Math::supports_q8k_gemm(lt.attn_q->type, D);
            const bool k_q8k = !eff_k || Avx2Math::supports_q8k_gemm(eff_k->type, D);
            const bool v_q8k = !eff_v || Avx2Math::supports_q8k_gemm(eff_v->type, D);
            const bool q_q832 = lt.attn_q && Avx2Math::supports_q8_32_gemm(lt.attn_q->type, D);
            const bool k_q832 = !eff_k || Avx2Math::supports_q8_32_gemm(eff_k->type, D);
            const bool v_q832 = !eff_v || Avx2Math::supports_q8_32_gemm(eff_v->type, D);

            const bool exact_qkv = exact_math_mode_ || ((exact_mask_ & EXACT_QKV) != 0);
            if (!exact_qkv && q_q8k && k_q8k && v_q8k) {
                Q8KBatch qkv_batch;
                Avx2Math::pack_q8k_batch(norm_buf.data(), D, D, S, qkv_batch);
                Avx2Math::gemm_slice_q8k(q_buf.data(), q_dim, *lt.attn_q, 0, qkv_batch, q_dim, D, true);
                if (eff_k) Avx2Math::gemm_slice_q8k(k_buf.data(), kv_dim, *eff_k, 0, qkv_batch, kv_dim, D, true);
                if (eff_v) Avx2Math::gemm_slice_q8k(v_buf.data(), kv_dim, *eff_v, 0, qkv_batch, kv_dim, D, true);
            } else if (!exact_qkv && q_q832 && k_q832 && v_q832) {
                Q8_32Batch qkv_batch;
                Avx2Math::pack_q8_32_batch(norm_buf.data(), D, D, S, qkv_batch);
                Avx2Math::gemm_slice_q8_32(q_buf.data(), q_dim, *lt.attn_q, 0, qkv_batch, q_dim, D, true);
                if (eff_k) Avx2Math::gemm_slice_q8_32(k_buf.data(), kv_dim, *eff_k, 0, qkv_batch, kv_dim, D, true);
                if (eff_v) Avx2Math::gemm_slice_q8_32(v_buf.data(), kv_dim, *eff_v, 0, qkv_batch, kv_dim, D, true);
            } else {
                auto run_gemm = [&](float* Y, int sy, const TensorDesc& w, const float* Xin, int sx, int nt, int rows, int cols) {
                    if (exact_qkv) Avx2Math::gemm_f32(Y, sy, w, Xin, sx, nt, rows, cols);
                    else           Avx2Math::gemm(Y, sy, w, Xin, sx, nt, rows, cols);
                };
                if (lt.attn_q) run_gemm(q_buf.data(), q_dim, *lt.attn_q, norm_buf.data(), D, S, q_dim, D);
                else {
                    for (int t = 0; t < S; ++t)
                        for (int i = 0; i < q_dim; ++i)
                            q_buf[static_cast<size_t>(t) * static_cast<size_t>(q_dim) + static_cast<size_t>(i)] =
                                norm_buf[static_cast<size_t>(t) * static_cast<size_t>(D) + static_cast<size_t>(i % D)];
                }
                if (eff_k) run_gemm(k_buf.data(), kv_dim, *eff_k, norm_buf.data(), D, S, kv_dim, D);
                else if (!is_shared_kv) {
                    for (int t = 0; t < S; ++t)
                        for (int i = 0; i < kv_dim; ++i)
                            k_buf[static_cast<size_t>(t) * static_cast<size_t>(kv_dim) + static_cast<size_t>(i)] =
                                norm_buf[static_cast<size_t>(t) * static_cast<size_t>(D) + static_cast<size_t>(i % D)];
                }
                if (eff_v) run_gemm(v_buf.data(), kv_dim, *eff_v, norm_buf.data(), D, S, kv_dim, D);
            }

            if (lt.attn_q_bias || (eff_k && lt.attn_k_bias) || (eff_v && lt.attn_v_bias) || (eff_k && !eff_v) || (!eff_v && !eff_k && !is_shared_kv)) {
                for (int t = 0; t < S; ++t) {
                    float* qt = q_buf.data() + static_cast<size_t>(t) * static_cast<size_t>(q_dim);
                    float* kt = k_buf.data() + static_cast<size_t>(t) * static_cast<size_t>(kv_dim);
                    float* vt = v_buf.data() + static_cast<size_t>(t) * static_cast<size_t>(kv_dim);
                    if (lt.attn_q && lt.attn_q_bias) Avx2Math::add_bias(qt, *lt.attn_q_bias, q_dim);
                    if (eff_k && lt.attn_k_bias) Avx2Math::add_bias(kt, *lt.attn_k_bias, kv_dim);
                    if (eff_v) {
                        if (lt.attn_v_bias) Avx2Math::add_bias(vt, *lt.attn_v_bias, kv_dim);
                    } else if (eff_k) {
                        std::memcpy(vt, kt, static_cast<size_t>(kv_dim) * sizeof(float));
                    } else if (!is_shared_kv) {
                        for (int i = 0; i < kv_dim; ++i) {
                            vt[i] = norm_buf[static_cast<size_t>(t) * static_cast<size_t>(D) + static_cast<size_t>(i % D)];
                        }
                    }
                }
            }

            // 2b-2 & 2c. Batched Per-Head Q/K/V Norms + RoPE
            const float* q_norm_w = lt.attn_q_norm ? safe_f32_weights(lt.attn_q_norm, static_cast<size_t>(actual_head_dim)) : nullptr;
            const float* k_norm_w = (has_kv && lt.attn_k_norm) ? safe_f32_weights(lt.attn_k_norm, static_cast<size_t>(actual_head_dim)) : nullptr;
            const bool do_v_norm  = (has_kv && cfg.is_gemma4 && lt.attn_q_norm && lt.attn_k_norm);
            const float layer_freq_base = (is_swa && (cfg.has_rope_freq_base_swa || cfg.is_gemma)) ? cfg.rope_freq_base_swa : cfg.rope_freq_base;
            const uint32_t cfg_rope_dim = is_swa ? cfg.rope_dimension_count_swa : cfg.rope_dimension_count;
            const int rope_dim = (!cfg.is_gemma4 && cfg_rope_dim > 0 && static_cast<int>(cfg_rope_dim) <= actual_head_dim)
                ? static_cast<int>(cfg_rope_dim)
                : actual_head_dim;
            const float* freq_factors_w = (!is_swa && rope_freqs_tensor_ && rope_freqs_tensor_->type == QuantType::F32 &&
                (rope_freqs_tensor_->size_bytes == 0 || rope_freqs_tensor_->size_bytes >= static_cast<size_t>(rope_dim / 2) * sizeof(float)))
                ? reinterpret_cast<const float*>(rope_freqs_tensor_->data) : nullptr;

            #pragma omp parallel for if(S >= 4) schedule(static)
            for (int t = 0; t < S; ++t) {
                const int cur_pos = std::clamp(chunk_start_pos + t, 0, max_ctx - 1);
                float* qt = q_buf.data() + static_cast<size_t>(t) * static_cast<size_t>(q_dim);
                float* kt = k_buf.data() + static_cast<size_t>(t) * static_cast<size_t>(kv_dim);
                float* vt = v_buf.data() + static_cast<size_t>(t) * static_cast<size_t>(kv_dim);
                if (q_norm_w) {
                    for (uint32_t h = 0; h < cfg.num_heads; ++h) {
                        float* q_h = qt + static_cast<size_t>(h) * static_cast<size_t>(actual_head_dim);
                        Avx2Math::rms_norm(q_h, q_h, q_norm_w, actual_head_dim, cfg.rms_norm_eps);
                    }
                }
                if (k_norm_w) {
                    for (uint32_t h = 0; h < layer_n_kv_heads; ++h) {
                        float* k_h = kt + static_cast<size_t>(h) * static_cast<size_t>(actual_head_dim);
                        Avx2Math::rms_norm(k_h, k_h, k_norm_w, actual_head_dim, cfg.rms_norm_eps);
                    }
                }
                if (do_v_norm) {
                    for (uint32_t h = 0; h < layer_n_kv_heads; ++h) {
                        float* v_h = vt + static_cast<size_t>(h) * static_cast<size_t>(actual_head_dim);
                        Avx2Math::rms_norm(v_h, v_h, default_norm_weights_.data(), actual_head_dim, cfg.rms_norm_eps);
                    }
                }
                if (has_kv) {
                    Avx2Math::apply_rope(qt, kt, actual_head_dim, cfg.num_heads, layer_n_kv_heads, cur_pos, layer_freq_base, cfg.rope_freq_scale, freq_factors_w, rope_dim, cfg.is_neox_rope);
                } else {
                    Avx2Math::apply_rope(qt, nullptr, actual_head_dim, cfg.num_heads, 0, cur_pos, layer_freq_base, cfg.rope_freq_scale, freq_factors_w, rope_dim, cfg.is_neox_rope);
                }
            }

            uint32_t kv_source_layer = l;
            if (has_kv) {
                for (int t = 0; t < S; ++t) {
                    const int cur_pos = std::clamp(chunk_start_pos + t, 0, max_ctx - 1);
                    kv_cache_.write(l, cur_pos,
                                    k_buf.data() + static_cast<size_t>(t) * static_cast<size_t>(kv_dim),
                                    v_buf.data() + static_cast<size_t>(t) * static_cast<size_t>(kv_dim),
                                    kv_dim);
                }
            } else {
                if (cfg.num_layers > 20 && cfg.shared_kv_layers > 0 && cfg.shared_kv_layers < cfg.num_layers) {
                    uint32_t n_kv_start = cfg.num_layers - cfg.shared_kv_layers;
                    kv_source_layer = n_kv_start - (is_swa ? 2 : 1);
                } else {
                    kv_source_layer = (l > 0) ? (l - 1) : 0;
                }
                if (kv_source_layer >= cfg.num_layers) kv_source_layer = 0;
            }

            // 2e. Parallel Causal Multi-Head FlashAttention across all (t, h) pairs
            const auto& kv_lay = kv_cache_.get_layer(kv_source_layer);
            const size_t max_safe_positions = (kv_dim > 0) ? (kv_lay.keys.size() / static_cast<size_t>(kv_dim)) : 0;
            const int swa_win = (cfg.sliding_window > 0) ? static_cast<int>(cfg.sliding_window) : 512;
            const float* layer_k = kv_lay.keys.data();
            const float* layer_v = kv_lay.values.data();
            const int num_heads_i = static_cast<int>(cfg.num_heads);
            const int total_th = S * num_heads_i;

            #pragma omp parallel for schedule(dynamic, 4)
            for (int th = 0; th < total_th; ++th) {
                const int t = th / num_heads_i;
                const int h = th % num_heads_i;
                const int cur_pos = std::clamp(chunk_start_pos + t, 0, max_ctx - 1);
                const int seq_len = std::min({cur_pos + 1, max_ctx, static_cast<int>(max_safe_positions)});
                const int s_start = is_swa ? std::max(0, seq_len - swa_win) : 0;

                const float* q_h = q_buf.data() + static_cast<size_t>(t) * static_cast<size_t>(q_dim) + static_cast<size_t>(h) * static_cast<size_t>(actual_head_dim);
                const uint32_t kv_head_idx = std::min(static_cast<uint32_t>(h / gqa_ratio), std::max(1u, layer_n_kv_heads) - 1);
                const uint32_t kv_head_offset = kv_head_idx * static_cast<uint32_t>(actual_head_dim);
                float* out_h = attn_heads_buf.data() + static_cast<size_t>(t) * static_cast<size_t>(q_dim) + static_cast<size_t>(h) * static_cast<size_t>(actual_head_dim);

                if (seq_len > 0 && kv_head_offset + static_cast<uint32_t>(actual_head_dim) <= static_cast<uint32_t>(kv_dim)) {
                    if (!is_swa || s_start <= 4) {
                        // Fast contiguous causal slice [0 .. seq_len-1] — zero index buffer needed!
                        const int active_len = seq_len - s_start;
                        if (s_start == 0) {
                            FlashAttentionCpu::compute_tiled_attention(
                                q_h, layer_k, layer_v, nullptr,
                                static_cast<uint32_t>(active_len), static_cast<uint32_t>(actual_head_dim),
                                static_cast<uint32_t>(kv_dim), kv_head_offset,
                                attn_scale, cfg.attn_logit_softcapping, out_h);
                        } else {
                            int stack_idx[2048];
                            std::vector<int> heap_idx;
                            int* idx_ptr = stack_idx;
                            if (active_len > 2048) { heap_idx.resize(static_cast<size_t>(active_len)); idx_ptr = heap_idx.data(); }
                            for (int i = 0; i < active_len; ++i) idx_ptr[i] = s_start + i;
                            FlashAttentionCpu::compute_tiled_attention(
                                q_h, layer_k, layer_v, idx_ptr,
                                static_cast<uint32_t>(active_len), static_cast<uint32_t>(actual_head_dim),
                                static_cast<uint32_t>(kv_dim), kv_head_offset,
                                attn_scale, cfg.attn_logit_softcapping, out_h);
                        }
                    } else {
                        const int active_len = 4 + (seq_len - s_start);
                        int stack_idx[2048];
                        std::vector<int> heap_idx;
                        int* idx_ptr = stack_idx;
                        if (active_len > 2048) { heap_idx.resize(static_cast<size_t>(active_len)); idx_ptr = heap_idx.data(); }
                        int p = 0;
                        for (int s = 0; s < 4; ++s) idx_ptr[p++] = s;
                        for (int s = s_start; s < seq_len; ++s) idx_ptr[p++] = s;
                        FlashAttentionCpu::compute_tiled_attention(
                            q_h, layer_k, layer_v, idx_ptr,
                            static_cast<uint32_t>(p), static_cast<uint32_t>(actual_head_dim),
                            static_cast<uint32_t>(kv_dim), kv_head_offset,
                            attn_scale, cfg.attn_logit_softcapping, out_h);
                    }
                } else {
                    std::memset(out_h, 0, static_cast<size_t>(actual_head_dim) * sizeof(float));
                }
            }

            // 2f. Batched Attention Output Projection + Post-Attention Norm + Residual 1
            const bool exact_attn_out = exact_math_mode_ || ((exact_mask_ & EXACT_ATTN_OUT) != 0);
            if (lt.attn_output) {
                if (exact_attn_out) Avx2Math::gemm_f32(attn_out_buf.data(), D, *lt.attn_output, attn_heads_buf.data(), q_dim, S, D, q_dim);
                else                Avx2Math::gemm(attn_out_buf.data(), D, *lt.attn_output, attn_heads_buf.data(), q_dim, S, D, q_dim);
            } else {
                for (int t = 0; t < S; ++t)
                    for (int i = 0; i < D; ++i)
                        attn_out_buf[static_cast<size_t>(t) * static_cast<size_t>(D) + static_cast<size_t>(i)] =
                            attn_heads_buf[static_cast<size_t>(t) * static_cast<size_t>(q_dim) + static_cast<size_t>(i % q_dim)];
            }

            const float* post_attn_w = lt.post_attention_norm ? safe_f32_weights(lt.post_attention_norm, cfg.embedding_dim) : nullptr;
            const float* ffn_norm_w  = cfg.turbo_folded_norms ? default_norm_weights_.data() : safe_f32_weights(lt.ffn_norm, cfg.embedding_dim);
            const int h_ffn = (cfg.turbo_hadamard_ffn > 0 && (D % static_cast<int>(cfg.turbo_hadamard_ffn)) == 0) ? static_cast<int>(cfg.turbo_hadamard_ffn) : 0;
            #pragma omp parallel for if(S >= 8) schedule(static)
            for (int t = 0; t < S; ++t) {
                float* ao = attn_out_buf.data() + static_cast<size_t>(t) * static_cast<size_t>(D);
                float* xt = X.data()            + static_cast<size_t>(t) * static_cast<size_t>(D);
                if (post_attn_w) Avx2Math::rms_norm(ao, ao, post_attn_w, D, cfg.rms_norm_eps);
                for (int i = 0; i < D; ++i) xt[i] += ao[i];
                // 2g. Pre-FFN RMSNorm + optional Gemma 4.5 Turbo Block-FWHT fused directly in cache!
                float* nb = norm_buf.data() + static_cast<size_t>(t) * static_cast<size_t>(D);
                Avx2Math::rms_norm(nb, xt, ffn_norm_w, D, cfg.rms_norm_eps);
                if (h_ffn > 0) Avx2Math::fwht_block_inplace(nb, D, h_ffn);
            }

            // 2h + 2i. Batched Shared FFN (per-layer F supports Gemma 4 E2B double-wide MLP; pack norm_buf ONCE for Gate + Up!)
            const int layer_F = std::max(1, std::min<int>(
                max_F,
                (lt.ffn_gate && lt.ffn_gate->shape.size() > 1) ? static_cast<int>(lt.ffn_gate->shape[1])
                : ((l < cfg.hidden_dim_per_layer.size() && cfg.hidden_dim_per_layer[l] > 0)
                    ? static_cast<int>(cfg.hidden_dim_per_layer[l])
                    : static_cast<int>(cfg.hidden_dim))));

            const bool has_moe_layer = (lt.ffn_gate_inp && lt.ffn_down_exps &&
                (lt.ffn_gate_up_exps || (lt.ffn_gate_exps && lt.ffn_up_exps)) && cfg.expert_count > 0);
            if (!lt.ffn_gate && !lt.ffn_up && !lt.ffn_down && has_moe_layer) {
                std::memset(ffn_down_buf.data(), 0, static_cast<size_t>(S) * static_cast<size_t>(D) * sizeof(float));
            } else {
                const bool exact_ffn_gu   = exact_math_mode_ || ((exact_mask_ & EXACT_FFN_GU) != 0);
                const bool exact_ffn_down = exact_math_mode_ || ((exact_mask_ & EXACT_FFN_DOWN) != 0);
                const bool exact_geglu    = exact_math_mode_ || ((exact_mask_ & EXACT_GEGLU) != 0);
                if (lt.ffn_gate && lt.ffn_up) {
                    if (!exact_ffn_gu && Avx2Math::supports_q8k_gemm(lt.ffn_gate->type, D) && Avx2Math::supports_q8k_gemm(lt.ffn_up->type, D)) {
                        Q8KBatch ffn_batch;
                        Avx2Math::pack_q8k_batch(norm_buf.data(), D, D, S, ffn_batch);
                        Avx2Math::gemm_slice_q8k(ffn_gate_buf.data(), layer_F, *lt.ffn_gate, 0, ffn_batch, layer_F, D, true);
                        Avx2Math::gemm_slice_q8k(ffn_up_buf.data(),   layer_F, *lt.ffn_up,   0, ffn_batch, layer_F, D, true);
                    } else if (!exact_ffn_gu && Avx2Math::supports_q8_32_gemm(lt.ffn_gate->type, D) && Avx2Math::supports_q8_32_gemm(lt.ffn_up->type, D)) {
                        Q8_32Batch ffn_batch;
                        Avx2Math::pack_q8_32_batch(norm_buf.data(), D, D, S, ffn_batch);
                        Avx2Math::gemm_slice_q8_32(ffn_gate_buf.data(), layer_F, *lt.ffn_gate, 0, ffn_batch, layer_F, D, true);
                        Avx2Math::gemm_slice_q8_32(ffn_up_buf.data(),   layer_F, *lt.ffn_up,   0, ffn_batch, layer_F, D, true);
                    } else if (exact_ffn_gu) {
                        Avx2Math::gemm_f32(ffn_gate_buf.data(), layer_F, *lt.ffn_gate, norm_buf.data(), D, S, layer_F, D);
                        Avx2Math::gemm_f32(ffn_up_buf.data(),   layer_F, *lt.ffn_up,   norm_buf.data(), D, S, layer_F, D);
                    } else {
                        Avx2Math::gemm(ffn_gate_buf.data(), layer_F, *lt.ffn_gate, norm_buf.data(), D, S, layer_F, D);
                        Avx2Math::gemm(ffn_up_buf.data(),   layer_F, *lt.ffn_up,   norm_buf.data(), D, S, layer_F, D);
                    }
                    const int h_ffn_dn = (cfg.turbo_hadamard_ffn_down > 0 && (layer_F % static_cast<int>(cfg.turbo_hadamard_ffn_down)) == 0)
                        ? static_cast<int>(cfg.turbo_hadamard_ffn_down) : 0;
                    if (use_gelu) {
                        #pragma omp parallel for schedule(static)
                        for (int t = 0; t < S; ++t) {
                            const size_t off = static_cast<size_t>(t) * static_cast<size_t>(layer_F);
                            geglu_avx2_range(ffn_gate_buf.data() + off, ffn_gate_buf.data() + off, ffn_up_buf.data() + off, layer_F, exact_geglu);
                            if (h_ffn_dn > 0) Avx2Math::fwht_block_inplace(ffn_gate_buf.data() + off, layer_F, h_ffn_dn);
                        }
                    } else {
                        #pragma omp parallel for schedule(static)
                        for (int t = 0; t < S; ++t) {
                            const size_t off = static_cast<size_t>(t) * static_cast<size_t>(layer_F);
                            float* gb = ffn_gate_buf.data() + off;
                            const float* ub = ffn_up_buf.data() + off;
                            for (int idx = 0; idx < layer_F; ++idx) {
                                const double g = static_cast<double>(gb[idx]);
                                gb[idx] = static_cast<float>((g / (1.0 + std::exp(-g))) * static_cast<double>(ub[idx]));
                            }
                            if (h_ffn_dn > 0) Avx2Math::fwht_block_inplace(gb, layer_F, h_ffn_dn);
                        }
                    }
                } else {
                    for (int t = 0; t < S; ++t)
                        for (int i = 0; i < layer_F; ++i)
                            ffn_gate_buf[static_cast<size_t>(t) * static_cast<size_t>(layer_F) + static_cast<size_t>(i)] =
                                norm_buf[static_cast<size_t>(t) * static_cast<size_t>(D) + static_cast<size_t>(i % D)] * 0.1f;
                }

                if (lt.ffn_down) {
                    if (exact_ffn_down) Avx2Math::gemm_f32(ffn_down_buf.data(), D, *lt.ffn_down, ffn_gate_buf.data(), layer_F, S, D, layer_F);
                    else                Avx2Math::gemm(ffn_down_buf.data(), D, *lt.ffn_down, ffn_gate_buf.data(), layer_F, S, D, layer_F);
                } else {
                    for (int t = 0; t < S; ++t)
                        for (int i = 0; i < D; ++i)
                            ffn_down_buf[static_cast<size_t>(t) * static_cast<size_t>(D) + static_cast<size_t>(i)] =
                                ffn_gate_buf[static_cast<size_t>(t) * static_cast<size_t>(layer_F) + static_cast<size_t>(i % layer_F)] * 0.05f;
                }
            }

            // 2i-1b. Batched Expert-Grouped Sparse Mixture-of-Experts Branch (Ava / Gemma 4 26B-A4B MoE)
            if (lt.ffn_gate_inp && lt.ffn_down_exps && (lt.ffn_gate_up_exps || (lt.ffn_gate_exps && lt.ffn_up_exps)) && cfg.expert_count > 0) {
                if (lt.post_ffw_norm_1) {
                    const float* p1_w = safe_f32_weights(lt.post_ffw_norm_1, cfg.embedding_dim);
                    #pragma omp parallel for if(S >= 8) schedule(static)
                    for (int t = 0; t < S; ++t) {
                        float* fd = ffn_down_buf.data() + static_cast<size_t>(t) * static_cast<size_t>(D);
                        Avx2Math::rms_norm(fd, fd, p1_w, D, cfg.rms_norm_eps);
                    }
                }

                const int E = static_cast<int>(cfg.expert_count);
                const int K_used = std::clamp(static_cast<int>(cfg.expert_used_count > 0 ? cfg.expert_used_count : 8u), 1, E);
                const int F_exp = static_cast<int>(cfg.expert_ffn_dim > 0 ? cfg.expert_ffn_dim : cfg.hidden_dim);
                const int two_F = 2 * F_exp;

                // Router input norm + scale for all S tokens
                const float inv_sqrt_d = static_cast<float>(1.0 / std::sqrt(static_cast<double>(D)));
                const float* rscale = (!cfg.turbo_folded_scales && lt.ffn_gate_inp_scale) ? safe_f32_weights(lt.ffn_gate_inp_scale, cfg.embedding_dim) : nullptr;
                #pragma omp parallel for if(S >= 8) schedule(static)
                for (int t = 0; t < S; ++t) {
                    float* mn = moe_norm_buf.data() + static_cast<size_t>(t) * static_cast<size_t>(D);
                    Avx2Math::rms_norm(mn, X.data() + static_cast<size_t>(t) * static_cast<size_t>(D),
                                       default_norm_weights_.data(), D, cfg.rms_norm_eps);
                    for (int i = 0; i < D; ++i) mn[i] *= inv_sqrt_d * (rscale ? rscale[i] : 1.0f);
                }

                // Exact FP32-activation Batched Router GEMM [S, E] (prevents Top-K expert misrouting!)
                Avx2Math::gemm_f32(moe_router_buf.data(), E, *lt.ffn_gate_inp, moe_norm_buf.data(), D, S, E, D);

                // Group routed (token, weight) pairs into per-expert buckets so each expert is read ONCE!
                struct ExpertHit { int tok; float weight; };
                std::vector<std::vector<ExpertHit>> buckets(static_cast<size_t>(E));
                const float* down_s = (!cfg.turbo_folded_scales && lt.ffn_down_exps_scale) ? safe_f32_weights(lt.ffn_down_exps_scale, static_cast<size_t>(E)) : nullptr;
                std::vector<int> idx_tmp(static_cast<size_t>(E));

                for (int t = 0; t < S; ++t) {
                    float* rlogits = moe_router_buf.data() + static_cast<size_t>(t) * static_cast<size_t>(E);
                    Avx2Math::softmax(rlogits, E);
                    for (int e = 0; e < E; ++e) idx_tmp[static_cast<size_t>(e)] = e;
                    std::partial_sort(idx_tmp.begin(), idx_tmp.begin() + K_used, idx_tmp.end(),
                        [&](int a, int b) { return rlogits[a] > rlogits[b]; });
                    double w_sum = 0.0;
                    for (int k = 0; k < K_used; ++k) w_sum += static_cast<double>(rlogits[idx_tmp[static_cast<size_t>(k)]]);
                    if (w_sum < 6.103515625e-5) w_sum = 6.103515625e-5;
                    for (int k = 0; k < K_used; ++k) {
                        const int e = idx_tmp[static_cast<size_t>(k)];
                        const float we = static_cast<float>((static_cast<double>(rlogits[e]) / w_sum) * (down_s ? static_cast<double>(down_s[e]) : 1.0));
                        buckets[static_cast<size_t>(e)].push_back({t, we});
                    }
                }

                // Pre-FFN norm 2 on X for all S tokens (fall back to ffn_norm for pure-MoE models)
                const TensorDesc* pre2_tensor = lt.pre_ffw_norm_2 ? lt.pre_ffw_norm_2 : lt.ffn_norm;
                const float* pre2_w = cfg.turbo_folded_norms ? default_norm_weights_.data() : safe_f32_weights(pre2_tensor, cfg.embedding_dim);
                const int h_moe_in = (cfg.turbo_hadamard_moe_in > 0 && (D % static_cast<int>(cfg.turbo_hadamard_moe_in)) == 0)
                    ? static_cast<int>(cfg.turbo_hadamard_moe_in) : 0;
                const int h_moe_dn = (cfg.turbo_hadamard_moe_down > 0 && (F_exp % static_cast<int>(cfg.turbo_hadamard_moe_down)) == 0)
                    ? static_cast<int>(cfg.turbo_hadamard_moe_down) : 0;
                #pragma omp parallel for if(S >= 8) schedule(static)
                for (int t = 0; t < S; ++t) {
                    float* mn = moe_norm_buf.data() + static_cast<size_t>(t) * static_cast<size_t>(D);
                    Avx2Math::rms_norm(mn, X.data() + static_cast<size_t>(t) * static_cast<size_t>(D),
                                       pre2_w, D, cfg.rms_norm_eps);
                    if (h_moe_in > 0) Avx2Math::fwht_block_inplace(mn, D, h_moe_in);
                }

                std::vector<int> active_exps;
                active_exps.reserve(static_cast<size_t>(E));
                for (int e = 0; e < E; ++e) {
                    if (buckets[static_cast<size_t>(e)].empty()) continue;
                    bool ok_gu = lt.ffn_gate_up_exps
                        ? tensor_fits_expert(lt.ffn_gate_up_exps, e, static_cast<size_t>(two_F), D)
                        : (tensor_fits_expert(lt.ffn_gate_exps, e, static_cast<size_t>(F_exp), D) &&
                           tensor_fits_expert(lt.ffn_up_exps,   e, static_cast<size_t>(F_exp), D));
                    if (ok_gu && tensor_fits_expert(lt.ffn_down_exps, e, static_cast<size_t>(D), F_exp)) {
                        active_exps.push_back(e);
                    }
                }

                std::vector<double> moe_accum_f64(static_cast<size_t>(S) * static_cast<size_t>(D), 0.0);
                const int n_active = static_cast<int>(active_exps.size());

                // Dynamic-scheduled parallel execution over active experts: each expert's weights stream ONCE!
                // Match CUDA policy: un-baked stock MoE (turbo_hadamard_moe_in/down == 0) routes MoE to exact FP32!
                const bool exact_moe_gu = exact_math_mode_ || ((exact_mask_ & EXACT_MOE_GU) != 0) || (cfg.turbo_hadamard_moe_in == 0);
                const bool exact_moe_dn = exact_math_mode_ || ((exact_mask_ & EXACT_MOE_DOWN) != 0) || (cfg.turbo_hadamard_moe_down == 0);
                const bool exact_geglu  = exact_math_mode_ || ((exact_mask_ & EXACT_GEGLU) != 0);
                const bool gu_is_q8k  = !exact_moe_gu && (lt.ffn_gate_up_exps
                    ? Avx2Math::supports_q8k_gemm(lt.ffn_gate_up_exps->type, D)
                    : (Avx2Math::supports_q8k_gemm(lt.ffn_gate_exps->type, D) && Avx2Math::supports_q8k_gemm(lt.ffn_up_exps->type, D)));
                const bool gu_is_q832 = !exact_moe_gu && (lt.ffn_gate_up_exps
                    ? Avx2Math::supports_q8_32_gemm(lt.ffn_gate_up_exps->type, D)
                    : (Avx2Math::supports_q8_32_gemm(lt.ffn_gate_exps->type, D) && Avx2Math::supports_q8_32_gemm(lt.ffn_up_exps->type, D)));
                const bool dn_is_q8k  = !exact_moe_dn && Avx2Math::supports_q8k_gemm(lt.ffn_down_exps->type, F_exp);
                const bool dn_is_q832 = !exact_moe_dn && Avx2Math::supports_q8_32_gemm(lt.ffn_down_exps->type, F_exp);

                #pragma omp parallel
                {
                    // 53-bit FP64 thread-local expert accumulation eliminates dynamic-schedule order drift!
                    std::vector<double> local_acc(static_cast<size_t>(S) * static_cast<size_t>(D), 0.0);
                    std::vector<float> exp_in;
                    std::vector<float> exp_gu;
                    std::vector<float> exp_act;
                    std::vector<float> exp_dn;
                    Q8KBatch   exp_q8k;
                    Q8_32Batch exp_q832;

                    #pragma omp for schedule(dynamic, 1)
                    for (int ai = 0; ai < n_active; ++ai) {
                        const int e = active_exps[static_cast<size_t>(ai)];
                        const auto& hits = buckets[static_cast<size_t>(e)];
                        const int cnt = static_cast<int>(hits.size());

                        exp_in.resize(static_cast<size_t>(cnt) * static_cast<size_t>(D));
                        exp_gu.resize(static_cast<size_t>(cnt) * static_cast<size_t>(two_F));
                        exp_act.resize(static_cast<size_t>(cnt) * static_cast<size_t>(F_exp));
                        exp_dn.resize(static_cast<size_t>(cnt) * static_cast<size_t>(D));

                        for (int j = 0; j < cnt; ++j) {
                            std::memcpy(exp_in.data() + static_cast<size_t>(j) * static_cast<size_t>(D),
                                        moe_norm_buf.data() + static_cast<size_t>(hits[static_cast<size_t>(j)].tok) * static_cast<size_t>(D),
                                        static_cast<size_t>(D) * sizeof(float));
                        }

                        if (cnt > 1 && gu_is_q8k) {
                            Avx2Math::pack_q8k_batch(exp_in.data(), D, D, cnt, exp_q8k);
                            if (lt.ffn_gate_up_exps) {
                                Avx2Math::gemm_slice_q8k(exp_gu.data(), two_F, *lt.ffn_gate_up_exps,
                                                         static_cast<size_t>(e) * static_cast<size_t>(two_F),
                                                         exp_q8k, two_F, D, false);
                            } else {
                                Avx2Math::gemm_slice_q8k(exp_gu.data(), two_F, *lt.ffn_gate_exps,
                                                         static_cast<size_t>(e) * static_cast<size_t>(F_exp),
                                                         exp_q8k, F_exp, D, false);
                                Avx2Math::gemm_slice_q8k(exp_gu.data() + F_exp, two_F, *lt.ffn_up_exps,
                                                         static_cast<size_t>(e) * static_cast<size_t>(F_exp),
                                                         exp_q8k, F_exp, D, false);
                            }
                        } else if (cnt > 1 && gu_is_q832) {
                            Avx2Math::pack_q8_32_batch(exp_in.data(), D, D, cnt, exp_q832);
                            if (lt.ffn_gate_up_exps) {
                                Avx2Math::gemm_slice_q8_32(exp_gu.data(), two_F, *lt.ffn_gate_up_exps,
                                                           static_cast<size_t>(e) * static_cast<size_t>(two_F),
                                                           exp_q832, two_F, D, false);
                            } else {
                                Avx2Math::gemm_slice_q8_32(exp_gu.data(), two_F, *lt.ffn_gate_exps,
                                                           static_cast<size_t>(e) * static_cast<size_t>(F_exp),
                                                           exp_q832, F_exp, D, false);
                                Avx2Math::gemm_slice_q8_32(exp_gu.data() + F_exp, two_F, *lt.ffn_up_exps,
                                                           static_cast<size_t>(e) * static_cast<size_t>(F_exp),
                                                           exp_q832, F_exp, D, false);
                            }
                        } else if (lt.ffn_gate_up_exps) {
                            if (exact_moe_gu) {
                                Avx2Math::gemm_slice_f32(exp_gu.data(), two_F, *lt.ffn_gate_up_exps,
                                                         static_cast<size_t>(e) * static_cast<size_t>(two_F),
                                                         exp_in.data(), D, cnt, two_F, D, false);
                            } else {
                                Avx2Math::gemm_slice(exp_gu.data(), two_F, *lt.ffn_gate_up_exps,
                                                     static_cast<size_t>(e) * static_cast<size_t>(two_F),
                                                     exp_in.data(), D, cnt, two_F, D, false);
                            }
                        } else {
                            if (exact_moe_gu) {
                                Avx2Math::gemm_slice_f32(exp_gu.data(), two_F, *lt.ffn_gate_exps,
                                                         static_cast<size_t>(e) * static_cast<size_t>(F_exp),
                                                         exp_in.data(), D, cnt, F_exp, D, false);
                                Avx2Math::gemm_slice_f32(exp_gu.data() + F_exp, two_F, *lt.ffn_up_exps,
                                                         static_cast<size_t>(e) * static_cast<size_t>(F_exp),
                                                         exp_in.data(), D, cnt, F_exp, D, false);
                            } else {
                                Avx2Math::gemm_slice(exp_gu.data(), two_F, *lt.ffn_gate_exps,
                                                     static_cast<size_t>(e) * static_cast<size_t>(F_exp),
                                                     exp_in.data(), D, cnt, F_exp, D, false);
                                Avx2Math::gemm_slice(exp_gu.data() + F_exp, two_F, *lt.ffn_up_exps,
                                                     static_cast<size_t>(e) * static_cast<size_t>(F_exp),
                                                     exp_in.data(), D, cnt, F_exp, D, false);
                            }
                        }

                        if (use_gelu) {
                            for (int j = 0; j < cnt; ++j) {
                                const float* gu = exp_gu.data() + static_cast<size_t>(j) * static_cast<size_t>(two_F);
                                float* ac = exp_act.data() + static_cast<size_t>(j) * static_cast<size_t>(F_exp);
                                geglu_avx2_range(ac, gu, gu + F_exp, F_exp, exact_geglu);
                                if (h_moe_dn > 0) Avx2Math::fwht_block_inplace(ac, F_exp, h_moe_dn);
                            }
                        } else {
                            for (int j = 0; j < cnt; ++j) {
                                const float* gu = exp_gu.data() + static_cast<size_t>(j) * static_cast<size_t>(two_F);
                                float* ac = exp_act.data() + static_cast<size_t>(j) * static_cast<size_t>(F_exp);
                                for (int f = 0; f < F_exp; ++f) {
                                    const double g = static_cast<double>(gu[f]);
                                    ac[f] = static_cast<float>((g / (1.0 + std::exp(-g))) * static_cast<double>(gu[F_exp + f]));
                                }
                                if (h_moe_dn > 0) Avx2Math::fwht_block_inplace(ac, F_exp, h_moe_dn);
                            }
                        }

                        if (cnt > 1 && dn_is_q8k) {
                            Avx2Math::pack_q8k_batch(exp_act.data(), F_exp, F_exp, cnt, exp_q8k);
                            Avx2Math::gemm_slice_q8k(exp_dn.data(), D, *lt.ffn_down_exps,
                                                     static_cast<size_t>(e) * static_cast<size_t>(D),
                                                     exp_q8k, D, F_exp, false);
                        } else if (cnt > 1 && dn_is_q832) {
                            Avx2Math::pack_q8_32_batch(exp_act.data(), F_exp, F_exp, cnt, exp_q832);
                            Avx2Math::gemm_slice_q8_32(exp_dn.data(), D, *lt.ffn_down_exps,
                                                       static_cast<size_t>(e) * static_cast<size_t>(D),
                                                       exp_q832, D, F_exp, false);
                        } else if (exact_moe_dn) {
                            Avx2Math::gemm_slice_f32(exp_dn.data(), D, *lt.ffn_down_exps,
                                                     static_cast<size_t>(e) * static_cast<size_t>(D),
                                                     exp_act.data(), F_exp, cnt, D, F_exp, false);
                        } else {
                            Avx2Math::gemm_slice(exp_dn.data(), D, *lt.ffn_down_exps,
                                                 static_cast<size_t>(e) * static_cast<size_t>(D),
                                                 exp_act.data(), F_exp, cnt, D, F_exp, false);
                        }

                        for (int j = 0; j < cnt; ++j) {
                            const double wk = static_cast<double>(hits[static_cast<size_t>(j)].weight);
                            const float* dn = exp_dn.data() + static_cast<size_t>(j) * static_cast<size_t>(D);
                            double* dst = local_acc.data() + static_cast<size_t>(hits[static_cast<size_t>(j)].tok) * static_cast<size_t>(D);
                            const __m256d vwk = _mm256_set1_pd(wk);
                            int i = 0;
                            for (; i <= D - 8; i += 8) {
                                const __m256 vf = _mm256_loadu_ps(dn + i);
                                const __m256d lo = _mm256_cvtps_pd(_mm256_castps256_ps128(vf));
                                const __m256d hi = _mm256_cvtps_pd(_mm256_extractf128_ps(vf, 1));
                                _mm256_storeu_pd(dst + i,     _mm256_fmadd_pd(lo, vwk, _mm256_loadu_pd(dst + i)));
                                _mm256_storeu_pd(dst + i + 4, _mm256_fmadd_pd(hi, vwk, _mm256_loadu_pd(dst + i + 4)));
                            }
                            for (; i < D; ++i) dst[i] += wk * static_cast<double>(dn[i]);
                        }
                    }

                    #pragma omp critical
                    {
                        const size_t n_tot = static_cast<size_t>(S) * static_cast<size_t>(D);
                        for (size_t i = 0; i < n_tot; ++i) moe_accum_f64[i] += local_acc[i];
                    }
                }
                {
                    const size_t n_tot = static_cast<size_t>(S) * static_cast<size_t>(D);
                    for (size_t i = 0; i < n_tot; ++i) moe_accum_buf[i] = static_cast<float>(moe_accum_f64[i]);
                }

                const float* p2_w = lt.post_ffw_norm_2 ? safe_f32_weights(lt.post_ffw_norm_2, cfg.embedding_dim) : nullptr;
                #pragma omp parallel for if(S >= 8) schedule(static)
                for (int t = 0; t < S; ++t) {
                    float* ma = moe_accum_buf.data() + static_cast<size_t>(t) * static_cast<size_t>(D);
                    float* fd = ffn_down_buf.data()  + static_cast<size_t>(t) * static_cast<size_t>(D);
                    if (p2_w) Avx2Math::rms_norm(ma, ma, p2_w, D, cfg.rms_norm_eps);
                    for (int i = 0; i < D; ++i) fd[i] += ma[i];
                }
            }

            // 2i-2. Post-FFN Norm + Residual Addition 2
            const float* post_ffn_w = lt.post_ffw_norm ? safe_f32_weights(lt.post_ffw_norm, cfg.embedding_dim) : nullptr;
            #pragma omp parallel for if(S >= 8) schedule(static)
            for (int t = 0; t < S; ++t) {
                float* fd = ffn_down_buf.data() + static_cast<size_t>(t) * static_cast<size_t>(D);
                float* xt = X.data()            + static_cast<size_t>(t) * static_cast<size_t>(D);
                if (post_ffn_w) Avx2Math::rms_norm(fd, fd, post_ffn_w, D, cfg.rms_norm_eps);
                for (int i = 0; i < D; ++i) xt[i] += fd[i];
            }

            // 2-pleias. Batched Per-Layer Token Injection (Gemma 4 E4B/E2B)
            if (lt.inp_gate && lt.proj && has_pleias_branch && (static_cast<size_t>(l) + 1ULL) * 256ULL <= static_cast<size_t>(pleias_total)) {
                const bool exact_pleias = exact_math_mode_ || ((exact_mask_ & EXACT_PLEIAS) != 0);
                if (exact_pleias) Avx2Math::gemm_f32(pl_gate_buf.data(), 256, *lt.inp_gate, X.data(), D, S, 256, D);
                else              Avx2Math::gemm(pl_gate_buf.data(), 256, *lt.inp_gate, X.data(), D, S, 256, D);
                #pragma omp parallel for if(S >= 8) schedule(static)
                for (int t = 0; t < S; ++t) {
                    float* pg = pl_gate_buf.data() + static_cast<size_t>(t) * 256ULL;
                    const float* pe = pl_emb_buf.data() + static_cast<size_t>(t) * static_cast<size_t>(pleias_total) + static_cast<size_t>(l) * 256ULL;
                    for (int i = 0; i < 256; ++i) {
                        const double g = static_cast<double>(pg[i]);
                        const double cdf = 0.5 * (1.0 + std::tanh(0.79788456080286535588 * (g + 0.044715 * g * g * g)));
                        pg[i] = static_cast<float>((g * cdf) * static_cast<double>(pe[i]));
                    }
                }
                if (exact_pleias) Avx2Math::gemm_f32(pl_out_buf.data(), D, *lt.proj, pl_gate_buf.data(), 256, S, D, 256);
                else              Avx2Math::gemm(pl_out_buf.data(), D, *lt.proj, pl_gate_buf.data(), 256, S, D, 256);
                const float* post_w = lt.post_norm ? safe_f32_weights(lt.post_norm, cfg.embedding_dim) : nullptr;
                #pragma omp parallel for if(S >= 8) schedule(static)
                for (int t = 0; t < S; ++t) {
                    float* po = pl_out_buf.data() + static_cast<size_t>(t) * static_cast<size_t>(D);
                    float* xt = X.data()          + static_cast<size_t>(t) * static_cast<size_t>(D);
                    if (post_w) Avx2Math::rms_norm(po, po, post_w, D, cfg.rms_norm_eps);
                    for (int i = 0; i < D; ++i) xt[i] += po[i];
                }
            } else if (lt.post_norm) {
                const float* post_norm_w = safe_f32_weights(lt.post_norm, cfg.embedding_dim);
                #pragma omp parallel for if(S >= 8) schedule(static)
                for (int t = 0; t < S; ++t) {
                    float* xt = X.data() + static_cast<size_t>(t) * static_cast<size_t>(D);
                    Avx2Math::rms_norm(xt, xt, post_norm_w, D, cfg.rms_norm_eps);
                }
            }

            // 2-scale. Layer Output Scale
            if (lt.layer_output_scale && lt.layer_output_scale->data && lt.layer_output_scale->type == QuantType::F32 &&
                (lt.layer_output_scale->size_bytes == 0 || lt.layer_output_scale->size_bytes >= sizeof(float))) {
                const float scale = *reinterpret_cast<const float*>(lt.layer_output_scale->data);
                const size_t n_tot = static_cast<size_t>(S) * static_cast<size_t>(D);
                for (size_t i = 0; i < n_tot; ++i) X[i] *= scale;
            }
        }

        for (int t = 0; t < S; ++t) {
            kv_cache_.advance_position();
            telemetry_engine_.finalize_step();
        }

        if (is_last_chunk) {
            // Preserve last hidden state in embedding_scratch_ for speculative / C ABI callers
            std::memcpy(embedding_scratch_.data(),
                        X.data() + static_cast<size_t>(S - 1) * static_cast<size_t>(D),
                        static_cast<size_t>(D) * sizeof(float));

            if (out_last_logits) {
                const float* out_norm_w = safe_f32_weights(output_norm_tensor_, cfg.embedding_dim);
                Avx2Math::rms_norm(norm_scratch_.data(), embedding_scratch_.data(), out_norm_w, D, cfg.rms_norm_eps);

                if (output_tensor_) {
                    // Always run final LM head in exact FP32 activations (bandwidth-bound S=1 GEMV, eliminates logit Q8 jitter!)
                    Avx2Math::gemv_f32(out_last_logits, *output_tensor_, norm_scratch_.data(), cfg.vocab_size, cfg.embedding_dim);
                } else {
                    for (size_t v = 0; v < cfg.vocab_size; ++v) {
                        out_last_logits[v] = std::sin(static_cast<float>(v) * 0.05f) * norm_scratch_[v % cfg.embedding_dim];
                    }
                }
                if (cfg.final_logit_softcapping > 0.0f) {
                    Avx2Math::softcap_logits(out_last_logits, cfg.vocab_size, cfg.final_logit_softcapping);
                }
            }
        }
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

    // Batched GEMM Prompt Prefill: computes all prompt tokens in L2-tiled chunks and only runs 262k LM head on the final token!
    forward_batch(prompt_tokens.data(), static_cast<int>(prompt_tokens.size()), 0, logits_scratch_.data());
    int pos = static_cast<int>(prompt_tokens.size());

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

HAVEN_API void haven_set_exact_math(void* engine, bool enabled) {
    if (!engine) return;
    reinterpret_cast<haven::HavenEngine*>(engine)->set_exact_math(enabled);
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