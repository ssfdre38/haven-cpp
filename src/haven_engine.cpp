#include "haven/haven_engine.h"
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
}

HavenEngine::~HavenEngine() = default;

bool HavenEngine::load_model(const std::string& gguf_filepath) {
    if (!loader_.load_file(gguf_filepath)) {
        return false;
    }
    const auto& cfg = loader_.get_config();
    embedding_scratch_.resize(cfg.embedding_dim, 0.0f);
    norm_scratch_.resize(cfg.embedding_dim, 0.0f);
    q_scratch_.resize(std::max(cfg.embedding_dim, cfg.num_heads * cfg.head_dim), 0.0f);
    k_scratch_.resize(std::max(cfg.embedding_dim, cfg.num_kv_heads * cfg.head_dim), 0.0f);
    v_scratch_.resize(std::max(cfg.embedding_dim, cfg.num_kv_heads * cfg.head_dim), 0.0f);
    attn_out_scratch_.resize(cfg.embedding_dim, 0.0f);
    ffn_gate_scratch_.resize(cfg.hidden_dim, 0.0f);
    ffn_up_scratch_.resize(cfg.hidden_dim, 0.0f);
    ffn_down_scratch_.resize(cfg.embedding_dim, 0.0f);
    logits_scratch_.resize(cfg.vocab_size, 0.0f);
    default_norm_weights_.assign(std::max(cfg.embedding_dim, cfg.hidden_dim), 1.0f);
    attn_scores_scratch_.resize(std::max(1024u, cfg.max_context_length), 0.0f);

    // Cache global tensors
    token_embd_tensor_ = loader_.get_tensor("token_embd.weight");
    per_layer_token_embd_tensor_ = loader_.get_tensor("per_layer_token_embd.weight");
    per_layer_model_proj_tensor_ = loader_.get_tensor("per_layer_model_proj.weight");
    per_layer_proj_norm_tensor_  = loader_.get_tensor("per_layer_proj_norm.weight");

    if (per_layer_token_embd_tensor_ && per_layer_token_embd_tensor_->shape.size() > 0) {
        per_layer_emb_scratch_.resize(per_layer_token_embd_tensor_->shape[0], 0.0f);
    }
    if (per_layer_model_proj_tensor_ && per_layer_model_proj_tensor_->shape.size() > 1) {
        per_layer_proj_scratch_.resize(per_layer_model_proj_tensor_->shape[1], 0.0f);
    }
    layer_gate_scratch_.resize(cfg.embedding_dim, 0.0f);

    output_norm_tensor_ = loader_.get_tensor("output_norm.weight");
    rope_freqs_tensor_  = loader_.get_tensor("rope_freqs.weight");
    output_tensor_ = loader_.get_tensor("output.weight");
    if (!output_tensor_) output_tensor_ = token_embd_tensor_;

    // Cache per-layer tensors
    layers_.resize(cfg.num_layers);
    for (uint32_t l = 0; l < cfg.num_layers; ++l) {
        std::string prefix = "blk." + std::to_string(l) + ".";
        layers_[l].attn_norm            = loader_.get_tensor(prefix + "attn_norm.weight");
        layers_[l].attn_q               = loader_.get_tensor(prefix + "attn_q.weight");
        layers_[l].attn_k               = loader_.get_tensor(prefix + "attn_k.weight");
        layers_[l].attn_v               = loader_.get_tensor(prefix + "attn_v.weight");
        layers_[l].attn_q_norm          = loader_.get_tensor(prefix + "attn_q_norm.weight");
        layers_[l].attn_k_norm          = loader_.get_tensor(prefix + "attn_k_norm.weight");
        layers_[l].attn_output          = loader_.get_tensor(prefix + "attn_output.weight");
        layers_[l].post_attention_norm  = loader_.get_tensor(prefix + "post_attention_norm.weight");
        layers_[l].ffn_norm             = loader_.get_tensor(prefix + "ffn_norm.weight");
        layers_[l].ffn_gate             = loader_.get_tensor(prefix + "ffn_gate.weight");
        layers_[l].ffn_up               = loader_.get_tensor(prefix + "ffn_up.weight");
        layers_[l].ffn_down             = loader_.get_tensor(prefix + "ffn_down.weight");
        layers_[l].post_ffw_norm        = loader_.get_tensor(prefix + "post_ffw_norm.weight");
        layers_[l].inp_gate             = loader_.get_tensor(prefix + "inp_gate.weight");
        layers_[l].proj                 = loader_.get_tensor(prefix + "proj.weight");
        layers_[l].post_norm            = loader_.get_tensor(prefix + "post_norm.weight");
        layers_[l].layer_output_scale   = loader_.get_tensor(prefix + "layer_output_scale.weight");
    }

    kv_cache_ = DynamicKVCacheManager(cfg.num_layers, cfg.max_context_length, cfg.num_kv_heads, cfg.head_dim, 0.0f, 1.0f);
    telemetry_engine_ = AttentionTelemetryEngine(cfg.num_layers, cfg.num_heads);
    tokenizer_.initialize(loader_.get_vocabulary());

    // Dynamically resize all scratch buffers to exact model dimensions
    const int pleias_dim = cfg.num_layers * 256;
    embedding_scratch_.resize(cfg.embedding_dim, 0.0f);
    norm_scratch_.resize(cfg.embedding_dim, 0.0f);
    q_scratch_.resize(cfg.num_heads * cfg.head_dim, 0.0f);
    k_scratch_.resize(cfg.num_kv_heads * cfg.head_dim, 0.0f);
    v_scratch_.resize(cfg.num_kv_heads * cfg.head_dim, 0.0f);
    attn_out_scratch_.resize(cfg.embedding_dim, 0.0f);
    ffn_gate_scratch_.resize(cfg.hidden_dim, 0.0f);
    ffn_up_scratch_.resize(cfg.hidden_dim, 0.0f);
    ffn_down_scratch_.resize(cfg.embedding_dim, 0.0f);
    layer_gate_scratch_.resize(cfg.hidden_dim, 0.0f);
    per_layer_emb_scratch_.resize(pleias_dim, 0.0f);
    per_layer_proj_scratch_.resize(pleias_dim, 0.0f);
    attn_scores_scratch_.resize(cfg.num_heads * cfg.max_context_length, 0.0f);
    attn_heads_out_scratch_.resize(cfg.num_heads * cfg.head_dim, 0.0f);
    attn_indices_scratch_.resize(cfg.max_context_length, 0);
    logits_scratch_.resize(cfg.vocab_size, 0.0f);

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
    sampler_.get_params().temperature = 0.65f;
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
    return speculative_engine_.draft_candidates(context, embedding_scratch_.data(), 4096);
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
    const auto& cfg = loader_.get_config();

    // 1. Token Embedding Lookup
    if (token_embd_tensor_ && token < cfg.vocab_size) {
        Avx2Math::dequantize_row(embedding_scratch_.data(), *token_embd_tensor_, token, cfg.embedding_dim);
        float emb_scale = std::sqrt((float)cfg.embedding_dim);
        for (size_t i = 0; i < cfg.embedding_dim; ++i) {
            embedding_scratch_[i] *= emb_scale;
        }
    } else {
        for (size_t i = 0; i < cfg.embedding_dim; ++i) {
            embedding_scratch_[i] = std::sin((float)(token + i) * 0.01f);
        }
    }

    // 1a-2. Dequantize & Project Per-Layer Direct Embeddings (Google Pleias Branch)
    if (per_layer_token_embd_tensor_ && !per_layer_emb_scratch_.empty()) {
        Avx2Math::dequantize_row(per_layer_emb_scratch_.data(), *per_layer_token_embd_tensor_, token, (int)per_layer_emb_scratch_.size());
        
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
            
            const float* proj_norm_w = per_layer_proj_norm_tensor_ ? reinterpret_cast<const float*>(per_layer_proj_norm_tensor_->data) : default_norm_weights_.data();
            
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
    const float* ones_w = default_norm_weights_.data();

    for (uint32_t l = 0; l < cfg.num_layers; ++l) {
        const auto& lt = (l < layers_.size()) ? layers_[l] : LayerTensors{};

        const int q_dim = (lt.attn_q && lt.attn_q->shape.size() > 1) ? (int)lt.attn_q->shape[1] : (cfg.num_heads * cfg.head_dim);
        const int kv_dim = (lt.attn_k && lt.attn_k->shape.size() > 1) ? (int)lt.attn_k->shape[1] : (cfg.num_kv_heads * cfg.head_dim);
        const int actual_head_dim = (cfg.num_heads > 0) ? (q_dim / cfg.num_heads) : cfg.head_dim;
        const float attn_scale = 1.0f; // Gemma 4 self.scaling = 1.0
        const int gqa_ratio = std::max(1u, cfg.num_heads / cfg.num_kv_heads);

        // 2a. Pre-Attention RMSNorm
        const float* attn_norm_w = lt.attn_norm ? reinterpret_cast<const float*>(lt.attn_norm->data) : ones_w;
        Avx2Math::rms_norm(norm_scratch_.data(), embedding_scratch_.data(), attn_norm_w, cfg.embedding_dim, cfg.rms_norm_eps);

        // 2b. Q, K, V Projections
        if (lt.attn_q) {
            Avx2Math::gemv(q_scratch_.data(), *lt.attn_q, norm_scratch_.data(), q_dim, cfg.embedding_dim);
        } else {
            for (int i = 0; i < q_dim; ++i) q_scratch_[i] = norm_scratch_[i % cfg.embedding_dim];
        }

        if (lt.attn_k) {
            Avx2Math::gemv(k_scratch_.data(), *lt.attn_k, norm_scratch_.data(), kv_dim, cfg.embedding_dim);
        } else {
            for (int i = 0; i < kv_dim; ++i) k_scratch_[i] = norm_scratch_[i % cfg.embedding_dim];
        }

        if (lt.attn_v) {
            Avx2Math::gemv(v_scratch_.data(), *lt.attn_v, norm_scratch_.data(), kv_dim, cfg.embedding_dim);
        } else {
            for (int i = 0; i < kv_dim; ++i) v_scratch_[i] = norm_scratch_[i % cfg.embedding_dim];
        }

        // 2b-2. Gemma 2 / 4 Per-Head Q-Norm, K-Norm, and V-Norm
        if (lt.attn_q_norm) {
            const float* q_norm_w = reinterpret_cast<const float*>(lt.attn_q_norm->data);
            for (uint32_t h = 0; h < cfg.num_heads; ++h) {
                float* q_h = q_scratch_.data() + h * actual_head_dim;
                Avx2Math::rms_norm(q_h, q_h, q_norm_w, actual_head_dim, cfg.rms_norm_eps);
            }
        }
        if (lt.attn_k_norm) {
            const float* k_norm_w = reinterpret_cast<const float*>(lt.attn_k_norm->data);
            for (uint32_t h = 0; h < cfg.num_kv_heads; ++h) {
                float* k_h = k_scratch_.data() + h * actual_head_dim;
                Avx2Math::rms_norm(k_h, k_h, k_norm_w, actual_head_dim, cfg.rms_norm_eps);
            }
        }
        for (uint32_t h = 0; h < cfg.num_kv_heads; ++h) {
            float* v_h = v_scratch_.data() + h * actual_head_dim;
            Avx2Math::rms_norm(v_h, v_h, default_norm_weights_.data(), actual_head_dim, cfg.rms_norm_eps);
        }

        // 2c. RoPE Rotary Embeddings (Gemma 4 ISWA: SWA 256 / Global 512 full head rotation)
        bool is_swa = (l % 6 != 5);
        float layer_freq_base = is_swa ? cfg.rope_freq_base_swa : cfg.rope_freq_base;
        int rope_dim = actual_head_dim;
        const float* freq_factors_w = (!is_swa && rope_freqs_tensor_) ? reinterpret_cast<const float*>(rope_freqs_tensor_->data) : nullptr;

        uint32_t kv_source_layer = l;
        bool has_kv = (lt.attn_k != nullptr);
        if (!has_kv) {
            uint32_t n_kv_start = (cfg.num_layers > 18) ? (cfg.num_layers - 18) : 24; // 24 for 42-layer model
            kv_source_layer = n_kv_start - (is_swa ? 2 : 1); // 22 for SWA, 23 for Global
        }

        if (has_kv) {
            Avx2Math::apply_rope(q_scratch_.data(), k_scratch_.data(), actual_head_dim, cfg.num_heads, cfg.num_kv_heads, pos, layer_freq_base, cfg.rope_freq_scale, freq_factors_w, rope_dim);
            kv_cache_.write(l, pos, k_scratch_.data(), v_scratch_.data(), kv_dim);
        } else {
            Avx2Math::apply_rope(q_scratch_.data(), nullptr, actual_head_dim, cfg.num_heads, 0, pos, layer_freq_base, cfg.rope_freq_scale, freq_factors_w, rope_dim);
        }

        // 2e. Multi-Head Attention (GQA) with Attention Sinks & Zero-Allocation Scratch Buffers
        const int seq_len = std::min(pos + 1, (int)cfg.max_context_length);
        const int s_start = is_swa ? std::max(0, seq_len - 512) : 0;

        int active_attn_len = 0;
        if (is_swa && s_start > 4) {
            for (int s = 0; s < 4; ++s) attn_indices_scratch_[active_attn_len++] = s; // Pinned Sink (BOS / System Anchor)
            for (int s = s_start; s < seq_len; ++s) attn_indices_scratch_[active_attn_len++] = s; // SWA Window
        } else {
            for (int s = s_start; s < seq_len; ++s) attn_indices_scratch_[active_attn_len++] = s;
        }

        const float* layer_k = kv_cache_.get_layer(kv_source_layer).keys.data();
        const float* layer_v = kv_cache_.get_layer(kv_source_layer).values.data();

        #pragma omp parallel for schedule(static)
        for (uint32_t h = 0; h < cfg.num_heads; ++h) {
            const float* q_h = q_scratch_.data() + h * actual_head_dim;
            const uint32_t kv_head_idx = h / gqa_ratio;
            float* scores = attn_scores_scratch_.data() + h * cfg.max_context_length;

            // Compute QK^T over active window with Softcapping
            for (int i = 0; i < active_attn_len; ++i) {
                int abs_s = attn_indices_scratch_[i];
                const float* k_s = layer_k + abs_s * kv_dim + kv_head_idx * actual_head_dim;
                float dot = 0.0f;
                for (int d = 0; d < actual_head_dim; ++d) {
                    dot += q_h[d] * k_s[d];
                }
                scores[i] = dot * attn_scale;
                scores[i] = 50.0f * std::tanh(scores[i] / 50.0f);
            }

            // Pure Raw Attention Softmax over active window
            Avx2Math::softmax(scores, active_attn_len);

            // Weighted Value Accumulation (Attn @ V) over active window
            float* out_h = attn_heads_out_scratch_.data() + h * actual_head_dim;
            for (int d = 0; d < actual_head_dim; ++d) {
                float acc = 0.0f;
                for (int i = 0; i < active_attn_len; ++i) {
                    int abs_s = attn_indices_scratch_[i];
                    const float* v_s = layer_v + abs_s * kv_dim + kv_head_idx * actual_head_dim;
                    acc += scores[i] * v_s[d];
                }
                out_h[d] = acc;
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
            const float* post_attn_w = reinterpret_cast<const float*>(lt.post_attention_norm->data);
            Avx2Math::rms_norm(attn_out_scratch_.data(), attn_out_scratch_.data(), post_attn_w, cfg.embedding_dim, cfg.rms_norm_eps);
        }

        // Residual Addition 1
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < cfg.embedding_dim; ++i) {
            embedding_scratch_[i] += attn_out_scratch_[i];
        }

        // 2g. Pre-FFN RMSNorm
        const float* ffn_norm_w = lt.ffn_norm ? reinterpret_cast<const float*>(lt.ffn_norm->data) : ones_w;
        Avx2Math::rms_norm(norm_scratch_.data(), embedding_scratch_.data(), ffn_norm_w, cfg.embedding_dim, cfg.rms_norm_eps);

        // 2h. FFN Gate & Up Projections (GeGLU)
        if (lt.ffn_gate && lt.ffn_up) {
            Avx2Math::gemv(ffn_gate_scratch_.data(), *lt.ffn_gate, norm_scratch_.data(), cfg.hidden_dim, cfg.embedding_dim);
            Avx2Math::gemv(ffn_up_scratch_.data(), *lt.ffn_up, norm_scratch_.data(), cfg.hidden_dim, cfg.embedding_dim);
            Avx2Math::gelu(ffn_gate_scratch_.data(), cfg.hidden_dim);
            #pragma omp parallel for schedule(static)
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

        // 2i-2. Post-FFN Norm (Gemma 2 / 4)
        if (lt.post_ffw_norm) {
            const float* post_ffn_w = reinterpret_cast<const float*>(lt.post_ffw_norm->data);
            Avx2Math::rms_norm(ffn_down_scratch_.data(), ffn_down_scratch_.data(), post_ffn_w, cfg.embedding_dim, cfg.rms_norm_eps);
        }

        // Residual Addition 2
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < cfg.embedding_dim; ++i) {
            embedding_scratch_[i] += ffn_down_scratch_[i];
        }

        // 2-pleias. Per-Layer Token Injection (Google Pleias Branch)
        if (lt.inp_gate && lt.proj && !per_layer_emb_scratch_.empty()) {
            float pleias_gate[256];
            Avx2Math::gemv(pleias_gate, *lt.inp_gate, embedding_scratch_.data(), 256, cfg.embedding_dim);
            Avx2Math::gelu(pleias_gate, 256);

            const float* layer_emb_slice = per_layer_emb_scratch_.data() + l * 256;
            for (int i = 0; i < 256; ++i) {
                pleias_gate[i] *= layer_emb_slice[i];
            }

            float pleias_proj[4096];
            Avx2Math::gemv(pleias_proj, *lt.proj, pleias_gate, cfg.embedding_dim, 256);

            if (lt.post_norm) {
                const float* post_w = reinterpret_cast<const float*>(lt.post_norm->data);
                Avx2Math::rms_norm(pleias_proj, pleias_proj, post_w, cfg.embedding_dim, cfg.rms_norm_eps);
            }

            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < cfg.embedding_dim; ++i) {
                embedding_scratch_[i] += pleias_proj[i];
            }
        } else if (lt.post_norm) {
            const float* post_norm_w = reinterpret_cast<const float*>(lt.post_norm->data);
            Avx2Math::rms_norm(embedding_scratch_.data(), embedding_scratch_.data(), post_norm_w, cfg.embedding_dim, cfg.rms_norm_eps);
        }

        // 2-scale. Layer Output Scale (Gemma 4 Pleias)
        if (lt.layer_output_scale && lt.layer_output_scale->data) {
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
    const float* out_norm_w = output_norm_tensor_ ? reinterpret_cast<const float*>(output_norm_tensor_->data) : ones_w;
    Avx2Math::rms_norm(norm_scratch_.data(), embedding_scratch_.data(), out_norm_w, cfg.embedding_dim, cfg.rms_norm_eps);

    if (output_tensor_) {
        Avx2Math::gemv(out_logits, *output_tensor_, norm_scratch_.data(), cfg.vocab_size, cfg.embedding_dim);
    } else {
        for (size_t v = 0; v < cfg.vocab_size; ++v) {
            out_logits[v] = std::sin((float)v * 0.05f) * norm_scratch_[v % cfg.embedding_dim];
        }
    }

    // 4. Gemma 4 Logits Softcapping
    Avx2Math::softcap_logits(out_logits, cfg.vocab_size, cfg.final_logit_softcapping > 0.0f ? cfg.final_logit_softcapping : 30.0f);
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

    // Auto-Regressive Token Generation Loop with Dynamic Soft-Pruning
    for (int gen = 0; gen < max_new_tokens; ++gen) {
        uint32_t next_token = sampler_.sample(logits_scratch_.data(), cfg.vocab_size, generated_tokens);
        generated_tokens.push_back(next_token);

        // Check for Gemma 4 end-of-turn / stop tokens: <eos> (1), <turn|> (106), </s> (212)
        if (next_token == tokenizer_.eos_token() || next_token == 1 || next_token == 106 || next_token == 212) {
            break; // Model emitted end-of-turn
        }

        std::string token_str = tokenizer_.decode_token(next_token);
        if (on_token_callback && !on_token_callback(next_token, token_str)) {
            break; // Callback signaled stop
        }

        forward(next_token, pos++, logits_scratch_.data());
    }
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
    if (!engine || !concept_name || !embedding) return;
    std::vector<float> emb_vec(embedding, embedding + dim);
    reinterpret_cast<haven::HavenEngine*>(engine)->inject_memory(concept_name, weight, emb_vec);
}

HAVEN_API void haven_inject_vision_frame(void* engine, const float* patch_embeddings, int num_patches, uint32_t width, uint32_t height, float salience) {
    if (!engine || !patch_embeddings) return;
    std::vector<float> patches(patch_embeddings, patch_embeddings + num_patches);
    reinterpret_cast<haven::HavenEngine*>(engine)->inject_vision_frame(patches, width, height, salience);
}

HAVEN_API void haven_inject_audio_spectrum(void* engine, const float* mel_bands, int num_bands, float energy_rms, float centroid, bool is_speech) {
    if (!engine || !mel_bands) return;
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
    if (!engine || !persona_vector) return;
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
    if (!engine || !logits) return 0;
    static haven::PersonaSampler sampler;
    static std::vector<uint32_t> empty_history;
    return sampler.sample(logits, vocab_size, empty_history);
}

HAVEN_API void haven_destroy_engine(void* engine) {
    if (engine) {
        delete reinterpret_cast<haven::HavenEngine*>(engine);
    }
}