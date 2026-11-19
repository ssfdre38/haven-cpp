#pragma once

#include "haven_types.h"
#include <vector>
#include <cstdint>
#include <functional>

namespace haven {

struct SpeculativeCandidate {
    std::vector<uint32_t> tokens;
    std::vector<float> probabilities;
    size_t accepted_count = 0;
};

class SpeculativeEngine {
public:
    SpeculativeEngine(uint32_t num_draft_heads = 4, float acceptance_threshold = 0.70f);
    ~SpeculativeEngine();

    // Generates candidate draft tokens using multi-head speculation
    SpeculativeCandidate draft_candidates(
        const std::vector<uint32_t>& context_tokens,
        const float* last_hidden_state,
        uint32_t embedding_dim
    );

    // Verifies candidate draft tokens against main target model logits in a single batch step
    // Returns number of accepted tokens
    size_t verify_candidates(
        const SpeculativeCandidate& candidate,
        const std::vector<float>& target_logits,
        uint32_t vocab_size,
        std::vector<uint32_t>& accepted_tokens
    );

    // Fast pointer-based candidate verification avoiding std::vector allocations
    size_t verify_candidates(
        const SpeculativeCandidate& candidate,
        const float* target_logits,
        uint32_t vocab_size,
        std::vector<uint32_t>& accepted_tokens
    );

    // Updates speculative draft weights based on acceptance feedback
    void update_feedback(bool was_accepted);

    float get_acceptance_rate() const;
    size_t get_total_drafted_tokens() const { return total_drafted_; }
    size_t get_total_accepted_tokens() const { return total_accepted_; }

private:
    uint32_t num_draft_heads_;
    float acceptance_threshold_;
    size_t total_drafted_ = 0;
    size_t total_accepted_ = 0;

    std::vector<std::vector<float>> draft_head_weights_;
    void initialize_draft_heads(uint32_t embedding_dim);
};

} // namespace haven
