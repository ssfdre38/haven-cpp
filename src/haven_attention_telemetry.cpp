#include "haven/haven_attention_telemetry.h"
#include <numeric>
#include <sstream>
#include <iomanip>

namespace haven {

AttentionTelemetryEngine::AttentionTelemetryEngine(uint32_t num_layers, uint32_t num_heads)
    : num_layers_(num_layers), num_heads_(num_heads)
{
    telemetry_history_.resize(num_layers_);
    for (uint32_t l = 0; l < num_layers_; ++l) {
        telemetry_history_[l].layer_index = l;
        telemetry_history_[l].head_entropies.resize(num_heads_, 0.0f);
    }
}

float AttentionTelemetryEngine::compute_entropy(const float* probs, int size) {
    if (!probs || size <= 1) return 0.0f;
    float entropy = 0.0f;
    for (int i = 0; i < size; ++i) {
        float p = probs[i];
        if (p > 1e-7f) {
            entropy -= p * std::log2(p);
        }
    }
    return entropy;
}

void AttentionTelemetryEngine::record_head_attention(
    uint32_t layer,
    uint32_t head,
    const float* attention_weights,
    int seq_len)
{
    if (!attention_weights || seq_len <= 0 || layer >= num_layers_ || head >= num_heads_) return;

    float head_ent = compute_entropy(attention_weights, seq_len);
    telemetry_history_[layer].head_entropies[head] = head_ent;

    if (head == 0) {
        telemetry_history_[layer].peak_attention_weight = 0.0f;
        telemetry_history_[layer].max_attended_position = 0;
    }

    // Track peak attention anchor
    for (int i = 0; i < seq_len; ++i) {
        if (attention_weights[i] > telemetry_history_[layer].peak_attention_weight) {
            telemetry_history_[layer].peak_attention_weight = attention_weights[i];
            telemetry_history_[layer].max_attended_position = i;
        }
    }
}

void AttentionTelemetryEngine::finalize_step() {
    if (num_layers_ == 0 || num_heads_ == 0) {
        global_entropy_ = 0.0f;
        return;
    }
    float total_entropy = 0.0f;
    for (uint32_t l = 0; l < num_layers_; ++l) {
        auto& lay = telemetry_history_[l];
        float sum = 0.0f;
        for (float h : lay.head_entropies) sum += h;
        lay.mean_attention_entropy = sum / (float)num_heads_;
        total_entropy += lay.mean_attention_entropy;
    }
    global_entropy_ = total_entropy / (float)num_layers_;
}

std::string AttentionTelemetryEngine::to_json_telemetry(int current_pos, uint32_t active_token) const {
    std::ostringstream ss;
    ss << "{\"type\":\"attention_telemetry\",\"pos\":" << current_pos
       << ",\"active_token\":" << active_token
       << ",\"global_entropy\":" << std::fixed << std::setprecision(4) << global_entropy_
       << ",\"focus_state\":\"" << (global_entropy_ < 2.5f ? "LaserFocus" : "DiffuseContemplation") << "\""
       << ",\"layer_count\":" << num_layers_
       << "}";
    return ss.str();
}

} // namespace haven