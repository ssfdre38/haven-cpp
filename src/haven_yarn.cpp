#include "haven/haven_yarn.h"
#include <algorithm>

namespace haven {

YaRNScaler::YaRNScaler(const YaRNConfig& config)
    : config_(config)
{
    update_temperature();
}

YaRNScaler::~YaRNScaler() = default;

void YaRNScaler::update_temperature() {
    if (config_.scale <= 1.0f) {
        mscale_ = 1.0f;
    } else {
        // Temperature scale: 0.1 * ln(s) + 1.0
        mscale_ = 0.1f * std::log(config_.scale) + 1.0f;
    }
}

void YaRNScaler::compute_frequencies(uint32_t head_dim, float base_freq, std::vector<float>& inv_freqs) {
    if (head_dim < 2 || base_freq <= 0.0f || config_.alpha <= 0.0f || config_.beta <= 0.0f || config_.scale <= 0.0f) {
        inv_freqs.clear();
        return;
    }
    inv_freqs.resize(head_dim / 2);

    // YaRN paper: beta=32 (fast/high-freq cycles -> short wavelen), alpha=1 (slow/low-freq cycles -> long wavelen)
    float high_freq_wavelen = config_.original_context / config_.beta;
    float low_freq_wavelen = config_.original_context / config_.alpha;
    float denom = (config_.beta != config_.alpha) ? (config_.beta - config_.alpha) : 1.0f;

    for (size_t i = 0; i < head_dim / 2; ++i) {
        float freq = 1.0f / std::pow(base_freq, (float)(2 * i) / (float)head_dim);
        float wavelen = 2.0f * 3.141592653589793f / freq;

        if (wavelen < high_freq_wavelen) {
            // High frequency: No interpolation (extrapolate)
            inv_freqs[i] = freq;
        } else if (wavelen > low_freq_wavelen) {
            // Low frequency: Full linear scale interpolation
            inv_freqs[i] = freq / config_.scale;
        } else {
            // Mid frequency: Smooth ramp interpolation
            float smooth = (config_.original_context / wavelen - config_.alpha) / denom;
            inv_freqs[i] = (1.0f - smooth) * (freq / config_.scale) + smooth * freq;
        }
    }
}

void YaRNScaler::apply_rope(
    float* q,
    float* k,
    uint32_t num_heads,
    uint32_t num_kv_heads,
    uint32_t head_dim,
    int pos,
    const std::vector<float>& inv_freqs)
{
    if (head_dim < 2 || inv_freqs.size() < head_dim / 2) return;

    // Apply rotary embedding to Queries
    if (q) {
        for (uint32_t h = 0; h < num_heads; ++h) {
            float* q_head = q + (static_cast<size_t>(h) * head_dim);
            for (uint32_t i = 0; i < head_dim / 2; ++i) {
                float theta = (float)pos * inv_freqs[i];
                float cos_th = std::cos(theta) * mscale_;
                float sin_th = std::sin(theta) * mscale_;

                float q0 = q_head[i * 2];
                float q1 = q_head[i * 2 + 1];
                q_head[i * 2]     = q0 * cos_th - q1 * sin_th;
                q_head[i * 2 + 1] = q0 * sin_th + q1 * cos_th;
            }
        }
    }

    // Apply rotary embedding to Keys
    if (k) {
        for (uint32_t h = 0; h < num_kv_heads; ++h) {
            float* k_head = k + (static_cast<size_t>(h) * head_dim);
            for (uint32_t i = 0; i < head_dim / 2; ++i) {
                float theta = (float)pos * inv_freqs[i];
                float cos_th = std::cos(theta) * mscale_;
                float sin_th = std::sin(theta) * mscale_;

                float k0 = k_head[i * 2];
                float k1 = k_head[i * 2 + 1];
                k_head[i * 2]     = k0 * cos_th - k1 * sin_th;
                k_head[i * 2 + 1] = k0 * sin_th + k1 * cos_th;
            }
        }
    }
}

} // namespace haven
