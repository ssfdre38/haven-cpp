#include "haven/haven_sampler.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <unordered_set>

namespace haven {

PersonaSampler::PersonaSampler(const SamplerParams& params)
    : params_(params), rng_(std::random_device{}())
{
}

void PersonaSampler::set_persona_embedding(const std::vector<float>& persona_vector) {
    persona_embedding_ = persona_vector;
}

void PersonaSampler::add_anti_robotic_penalty(uint32_t token_id, float penalty_strength) {
    token_penalties_[token_id] = penalty_strength;
}

void PersonaSampler::initialize_vocab_counterparts(const std::vector<std::string>& vocab) {
    lemma_family_.assign(vocab.size(), {});

    std::unordered_map<std::string, std::vector<uint32_t>> stem_to_tokens;
    stem_to_tokens.reserve(vocab.size());

    for (uint32_t i = 107; i < (uint32_t)vocab.size(); ++i) {
        std::string s = vocab[i];
        if (s.length() > 3 && 
            (unsigned char)s[0] == 0xe2 && 
            (unsigned char)s[1] == 0x96 && 
            (unsigned char)s[2] == 0x81) 
        {
            s = s.substr(3);
        }
        std::string lower;
        for (char c : s) lower += (char)std::tolower((unsigned char)c);
        if (!lower.empty() && lower.find_first_of("abcdefghijklmnopqrstuvwxyz") != std::string::npos) {
            stem_to_tokens[lower].push_back(i);
        }
    }

    for (const auto& [stem, tokens] : stem_to_tokens) {
        if (tokens.size() > 1) {
            for (uint32_t t : tokens) {
                lemma_family_[t] = tokens;
            }
        }
    }
}

uint32_t PersonaSampler::sample(
    float* logits,
    uint32_t vocab_size,
    const std::vector<uint32_t>& recent_tokens,
    const float* token_embeddings,
    int embedding_dim)
{
    if (!logits || vocab_size == 0) return 0;

    // 0. Greedy Argmax path for deterministic generation
    if (params_.temperature <= 0.05f) {
        uint32_t best_token = 0;
        float best_logit = logits[0];
        for (uint32_t i = 1; i < vocab_size; ++i) {
            if (logits[i] > best_logit) {
                best_logit = logits[i];
                best_token = i;
            }
        }
        return best_token;
    }

    // 1. Standard Token Frequency / Repetition Penalty (Sliding Window of 64 tokens)
    if (params_.repetition_penalty > 1.0f && !recent_tokens.empty()) {
        std::unordered_map<uint32_t, int> token_counts;
        size_t window_sz = std::min((size_t)64, (size_t)params_.dry_penalty_last_n);
        size_t start_idx = (recent_tokens.size() > window_sz) ? (recent_tokens.size() - window_sz) : 0;

        for (size_t i = start_idx; i < recent_tokens.size(); ++i) {
            uint32_t t = recent_tokens[i];
            if (t > 106) { // Skip control tokens (0..106: pad, eos, bos, start_of_turn, end_of_turn)
                token_counts[t]++;
            }
        }

        for (const auto& [token_id, count] : token_counts) {
            if (token_id < vocab_size) {
                float penalty = 1.0f + (params_.repetition_penalty - 1.0f) * std::min(count, 3);
                if (logits[token_id] > 0.0f) {
                    logits[token_id] /= penalty;
                } else {
                    logits[token_id] *= penalty;
                }
            }
        }
    }

    // 2. Intelligent N-Gram & Anti-Stutter Filtering (DRY)
    if (!recent_tokens.empty()) {
        const size_t n_tokens = recent_tokens.size();
        uint32_t last_token = recent_tokens.back();

        // 2a. Consecutive Exact Word & Full Lemma Stutter Suppression (e.g. Oh oh, Just just, Anything anything)
        if (last_token < vocab_size && last_token > 106) {
            logits[last_token] -= 4.5f;
            if (last_token < lemma_family_.size() && !lemma_family_[last_token].empty()) {
                for (uint32_t sibling : lemma_family_[last_token]) {
                    if (sibling < vocab_size) {
                        logits[sibling] -= 4.5f;
                    }
                }
            }
        }

        // 2b. A B A Loop Suppression
        if (n_tokens >= 2) {
            uint32_t prev2 = recent_tokens[n_tokens - 2];
            if (prev2 < vocab_size && prev2 > 106) {
                logits[prev2] -= 2.5f;
                if (prev2 < lemma_family_.size() && !lemma_family_[prev2].empty()) {
                    for (uint32_t sibling : lemma_family_[prev2]) {
                        if (sibling < vocab_size) {
                            logits[sibling] -= 2.5f;
                        }
                    }
                }
            }
        }

        // 2c. Turn-Start Parenthetical Action Suppression (Tokens 568 = ' (', 236769 = '(')
        if (n_tokens <= 2) {
            if (vocab_size > 568) logits[568] -= 6.0f; // Prevent starting turn with "(chuckles..."
            if (vocab_size > 236769) logits[236769] -= 6.0f;
        }

        // 2d. General & Consecutive Hyphen / Run-on Suppression (Token 236772 = '-')
        if (vocab_size > 236772) {
            logits[236772] -= 1.5f; // Mildly discourage coining hyphenated adjective compounds (anxious-storming)
            if (last_token == 236772) {
                logits[236772] -= 8.0f; // Strongly suppress hyphen-chains (e.g. I-don't-don't-can't)
            }
        }

        // 2e. Roleplay Asterisk Suppression (Token 808 = ' *', Token 236829 = '*')
        if (vocab_size > 808) {
            logits[808] -= 8.0f; // Strongly suppress spaced asterisk (' *you')
        }
        if (vocab_size > 236829) {
            logits[236829] -= 8.0f; // Strongly suppress raw asterisk ('*')
        }

        // 2e. 3-gram and 2-gram Phrase Repetition Suppression
        if (n_tokens >= 2) {
            uint32_t prev1 = recent_tokens[n_tokens - 1];
            uint32_t prev2 = recent_tokens[n_tokens - 2];

            for (size_t i = 0; i + 2 < n_tokens; ++i) {
                if (recent_tokens[i] == prev2 && recent_tokens[i + 1] == prev1) {
                    uint32_t repeated_follower = recent_tokens[i + 2];
                    if (repeated_follower < vocab_size && repeated_follower > 106) {
                        logits[repeated_follower] -= 3.5f; // Break 3-gram phrase loops cleanly
                    }
                }
            }
        }
    }

    // 3. Anti-Robotic Tone Penalties
    for (const auto& [token_id, penalty] : token_penalties_) {
        if (token_id < vocab_size) {
            logits[token_id] -= penalty;
        }
    }

    // 3. Persona Fidelity Logit Bias Layer
    if (!persona_embedding_.empty() && token_embeddings != nullptr && embedding_dim > 0) {
        const int safe_dim = std::min<int>(embedding_dim, static_cast<int>(persona_embedding_.size()));
        for (uint32_t t = 0; t < vocab_size; ++t) {
            const float* t_emb = token_embeddings + static_cast<size_t>(t) * static_cast<size_t>(embedding_dim);
            float dot = 0.0f, norm_t = 0.0f, norm_p = 0.0f;
            for (int d = 0; d < safe_dim; ++d) {
                dot += t_emb[d] * persona_embedding_[d];
                norm_t += t_emb[d] * t_emb[d];
                norm_p += persona_embedding_[d] * persona_embedding_[d];
            }
            float cosine_sim = dot / (std::sqrt(norm_t * norm_p) + 1e-8f);
            float persona_drift_penalty = (1.0f - cosine_sim) * params_.persona_fidelity_strength;
            logits[t] -= persona_drift_penalty;
        }
    }

    // 4. Fast Exact Top-Tail Softmax & Nucleus Filtering (O(V) linear scan + tiny sort, ~0.15 ms vs ~22 ms on 262k vocab!)
    float max_logit = -1e9f;
    for (uint32_t i = 0; i < vocab_size; ++i) {
        if (std::isnan(logits[i]) || std::isinf(logits[i])) {
            logits[i] = -1e4f;
        }
        if (logits[i] > max_logit) max_logit = logits[i];
    }

    const float eff_temp = std::max(0.01f, params_.temperature);
    const float inv_temp = 1.0f / eff_temp;
    // Any token with (logit - max_logit) / temp < -16.0 has relative probability < 1.1e-7
    const float logit_cutoff = max_logit - 16.0f * eff_temp;

    std::vector<std::pair<float, uint32_t>> probs;
    probs.reserve(512);
    for (uint32_t i = 0; i < vocab_size; ++i) {
        const float lv = logits[i];
        if (lv >= logit_cutoff) {
            probs.push_back({lv, i});
        }
    }
    if (probs.empty()) {
        probs.push_back({max_logit, 0});
    }

    float sum_exp = 0.0f;
    for (auto& item : probs) {
        item.first = std::exp((item.first - max_logit) * inv_temp);
        sum_exp += item.first;
    }

    if (sum_exp <= 1e-9f || std::isnan(sum_exp)) {
        sum_exp = 1.0f;
    }

    const float inv_sum = 1.0f / sum_exp;
    for (auto& item : probs) {
        item.first *= inv_sum;
    }

    // 5. Top-K, Top-P, and Min-P Joint Nucleus Filtering (partial_sort only top_k when bounded)
    int k_limit = (params_.top_k > 0) ? std::min<int>(params_.top_k, static_cast<int>(probs.size())) : static_cast<int>(probs.size());
    if (k_limit < static_cast<int>(probs.size())) {
        std::partial_sort(probs.begin(), probs.begin() + k_limit, probs.end(),
                          [](const auto& a, const auto& b) { return a.first > b.first; });
    } else {
        std::sort(probs.begin(), probs.end(),
                  [](const auto& a, const auto& b) { return a.first > b.first; });
    }

    const float top_prob = probs[0].first;
    const float min_p_threshold = top_prob * params_.min_p;

    float filtered_sum = 0.0f;
    std::vector<std::pair<float, uint32_t>> candidates;
    candidates.reserve(static_cast<size_t>(k_limit));

    for (int i = 0; i < k_limit; ++i) {
        const auto& item = probs[i];
        if (item.first < min_p_threshold && !candidates.empty()) break;
        candidates.push_back(item);
        filtered_sum += item.first;
        if (filtered_sum >= params_.top_p && !candidates.empty()) break;
    }

    if (candidates.empty()) return 0;

    // 6. Stochastic Categorical Sampling
    std::uniform_real_distribution<float> dist(0.0f, filtered_sum);
    float r = dist(rng_);
    float cum_sum = 0.0f;

    for (const auto& item : candidates) {
        cum_sum += item.first;
        if (r <= cum_sum) {
            return item.second;
        }
    }

    return candidates[0].second;
}

} // namespace haven