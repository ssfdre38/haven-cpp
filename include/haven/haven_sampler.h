#pragma once

#include "haven_types.h"
#include <vector>
#include <string>
#include <unordered_map>
#include <random>

namespace haven {

struct SamplerParams {
    float temperature = 0.48f;
    float min_p = 0.05f;
    float top_p = 0.90f;
    int   top_k = 40;
    float repetition_penalty = 1.08f;
    float persona_fidelity_strength = 0.0f;
    float dry_multiplier = 0.8f;
    float dry_base = 1.75f;
    int   dry_allowed_length = 2;
    int   dry_penalty_last_n = 256;
};

class PersonaSampler {
public:
    PersonaSampler(const SamplerParams& params = SamplerParams());

    // Ingests soul persona embedding vector to steer candidate logits
    void set_persona_embedding(const std::vector<float>& persona_vector);
    
    // Penalizes robotic/corporate token sequences in real-time
    void add_anti_robotic_penalty(uint32_t token_id, float penalty_strength);

    // Samples next token from raw model logits with persona steering
    uint32_t sample(
        float* logits,
        uint32_t vocab_size,
        const std::vector<uint32_t>& recent_tokens,
        const float* token_embeddings = nullptr,
        int embedding_dim = 0
    );

    // Builds bidirectional lemma map to eliminate cross-token stutter (e.g. ▁just vs just)
    void initialize_vocab_counterparts(const std::vector<std::string>& vocab);

    SamplerParams& get_params() { return params_; }

private:
    SamplerParams params_;
    std::vector<float> persona_embedding_;
    std::unordered_map<uint32_t, float> token_penalties_;
    std::vector<std::vector<uint32_t>> lemma_family_;
    std::mt19937 rng_;
};

} // namespace haven