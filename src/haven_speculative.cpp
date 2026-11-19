#include "haven/haven_speculative.h"
#include <cmath>
#include <algorithm>

namespace haven {

SpeculativeEngine::SpeculativeEngine(uint32_t num_draft_heads, float acceptance_threshold)
    : num_draft_heads_(num_draft_heads),
      acceptance_threshold_(acceptance_threshold)
{
    initialize_draft_heads(4096);
}

SpeculativeEngine::~SpeculativeEngine() = default;

void SpeculativeEngine::initialize_draft_heads(uint32_t embedding_dim) {
    draft_head_weights_.resize(num_draft_heads_);
    for (uint32_t h = 0; h < num_draft_heads_; ++h) {
        draft_head_weights_[h].resize(embedding_dim, 0.0f);
        for (size_t i = 0; i < embedding_dim; ++i) {
            draft_head_weights_[h][i] = std::cos((float)(h * 100 + i) * 0.005f) * 0.02f;
        }
    }
}

SpeculativeCandidate SpeculativeEngine::draft_candidates(
    const std::vector<uint32_t>& context_tokens,
    const float* last_hidden_state,
    uint32_t embedding_dim)
{
    SpeculativeCandidate cand;
    cand.tokens.reserve(num_draft_heads_);
    cand.probabilities.reserve(num_draft_heads_);

    uint32_t last_token = context_tokens.empty() ? 101 : context_tokens.back();

    for (uint32_t h = 0; h < num_draft_heads_; ++h) {
        float projection = 0.0f;
        if (last_hidden_state && !draft_head_weights_[h].empty()) {
            for (size_t i = 0; i < embedding_dim; ++i) {
                projection += last_hidden_state[i] * draft_head_weights_[h][i];
            }
        }

        // Draft predicted token based on hidden state projection + token context
        uint32_t drafted_token = (last_token + (uint32_t)(std::abs(projection) * 100.0f) + (h + 1) * 3) % 32000;
        float confidence = std::clamp(0.70f + (float)std::abs(projection) * 0.25f - ((float)h * 0.05f), 0.50f, 0.99f);

        cand.tokens.push_back(drafted_token);
        cand.probabilities.push_back(confidence);
        last_token = drafted_token;
    }

    total_drafted_ += cand.tokens.size();
    return cand;
}

size_t SpeculativeEngine::verify_candidates(
    const SpeculativeCandidate& candidate,
    const std::vector<float>& target_logits,
    uint32_t vocab_size,
    std::vector<uint32_t>& accepted_tokens)
{
    return verify_candidates(
        candidate,
        target_logits.empty() ? nullptr : target_logits.data(),
        vocab_size,
        accepted_tokens
    );
}

size_t SpeculativeEngine::verify_candidates(
    const SpeculativeCandidate& candidate,
    const float* target_logits,
    uint32_t vocab_size,
    std::vector<uint32_t>& accepted_tokens)
{
    accepted_tokens.clear();
    size_t accepted = 0;

    for (size_t i = 0; i < candidate.tokens.size(); ++i) {
        uint32_t tok = candidate.tokens[i];
        if (tok < vocab_size) {
            float confidence = candidate.probabilities[i];
            if (target_logits && target_logits[tok] < -50.0f) {
                // Rejected by target model's logits distribution
                break;
            }
            if (confidence >= acceptance_threshold_) {
                accepted_tokens.push_back(tok);
                accepted++;
            } else {
                // Speculative verification stopped at first failed branch
                break;
            }
        } else {
            break;
        }
    }

    total_accepted_ += accepted;
    return accepted;
}

void SpeculativeEngine::update_feedback(bool was_accepted) {
    if (was_accepted) {
        // Lower threshold slightly to be more aggressive
        acceptance_threshold_ = std::max(0.55f, acceptance_threshold_ - 0.005f);
    } else {
        // Increase threshold slightly to be more conservative
        acceptance_threshold_ = std::min(0.85f, acceptance_threshold_ + 0.01f);
    }
}

float SpeculativeEngine::get_acceptance_rate() const {
    if (total_drafted_ == 0) return 1.0f;
    return (float)total_accepted_ / (float)total_drafted_;
}

} // namespace haven
