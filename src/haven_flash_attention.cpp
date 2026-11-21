#include "haven/haven_flash_attention.h"
#include "haven/haven_avx2.h"
#include <cmath>
#include <cstring>
#include <algorithm>

namespace haven {

FlashAttentionCpu::FlashAttentionCpu(uint32_t head_dim)
    : head_dim_(head_dim)
{
    scores_scratch_.resize(FLASH_BLOCK_BC, 0.0f);
}

FlashAttentionCpu::~FlashAttentionCpu() = default;

void FlashAttentionCpu::compute_tiled_attention(
    const float* q,
    const float* k_cache,
    const float* v_cache,
    uint32_t num_tokens,
    uint32_t head_dim,
    float scale,
    float* out_attn)
{
    compute_tiled_attention(
        q, k_cache, v_cache,
        nullptr, num_tokens, head_dim,
        head_dim, 0, scale, 0.0f, out_attn
    );
}

void FlashAttentionCpu::compute_tiled_attention(
    const float* q,
    const float* layer_k,
    const float* layer_v,
    const int* indices,
    uint32_t num_tokens,
    uint32_t head_dim,
    uint32_t kv_dim,
    uint32_t kv_head_offset,
    float scale,
    float softcap,
    float* out_attn)
{
    if (!q || !layer_k || !layer_v || !out_attn || num_tokens == 0 || head_dim == 0) return;
    std::memset(out_attn, 0, head_dim * sizeof(float));

    float max_score = -1e9f;
    float sum_exp = 0.0f;
    float block_scores[FLASH_BLOCK_BC];

    // Tile across Key/Value cache blocks (fitting completely in L1/L2 cache)
    for (uint32_t block_start = 0; block_start < num_tokens; block_start += FLASH_BLOCK_BC) {
        uint32_t block_end = std::min(num_tokens, (uint32_t)(block_start + FLASH_BLOCK_BC));
        uint32_t current_block_size = block_end - block_start;

        float block_max = -1e9f;

        // Compute Q · K^T dot products for this block using AVX2 SIMD
        for (uint32_t i = 0; i < current_block_size; ++i) {
            uint32_t abs_s = indices ? (uint32_t)indices[block_start + i] : (block_start + i);
            const float* k_vec = layer_k + static_cast<size_t>(abs_s) * static_cast<size_t>(kv_dim) + kv_head_offset;

            float dot = Avx2Math::vec_dot_f32((int)head_dim, q, k_vec);
            float score = dot * scale;
            if (softcap > 0.0f) {
                score = softcap * std::tanh(score / softcap);
            }
            block_scores[i] = score;
            if (score > block_max) block_max = score;
        }

        // Online softmax update: Rescale previous accumulator
        float new_max = std::max(max_score, block_max);
        float rescale_prev = (max_score > -1e8f) ? std::exp(max_score - new_max) : 0.0f;
        sum_exp *= rescale_prev;

        __m256 vrescale = _mm256_set1_ps(rescale_prev);
        int d = 0;
        for (; d <= (int)head_dim - 8; d += 8) {
            __m256 vo = _mm256_loadu_ps(out_attn + d);
            _mm256_storeu_ps(out_attn + d, _mm256_mul_ps(vo, vrescale));
        }
        for (; d < (int)head_dim; ++d) {
            out_attn[d] *= rescale_prev;
        }

        // Accumulate new weighted values from this block with AVX2 FMA
        for (uint32_t i = 0; i < current_block_size; ++i) {
            uint32_t abs_s = indices ? (uint32_t)indices[block_start + i] : (block_start + i);
            const float* v_vec = layer_v + static_cast<size_t>(abs_s) * static_cast<size_t>(kv_dim) + kv_head_offset;

            float exp_val = std::exp(block_scores[i] - new_max);
            sum_exp += exp_val;
            __m256 vexp = _mm256_set1_ps(exp_val);

            d = 0;
            for (; d <= (int)head_dim - 8; d += 8) {
                __m256 vo = _mm256_loadu_ps(out_attn + d);
                __m256 vv = _mm256_loadu_ps(v_vec + d);
                _mm256_storeu_ps(out_attn + d, _mm256_fmadd_ps(vv, vexp, vo));
            }
            for (; d < (int)head_dim; ++d) {
                out_attn[d] += exp_val * v_vec[d];
            }
        }

        max_score = new_max;
    }

    // Final normalization
    if (sum_exp > 1e-6f) {
        float inv_sum = 1.0f / sum_exp;
        __m256 vinv = _mm256_set1_ps(inv_sum);
        int d = 0;
        for (; d <= (int)head_dim - 8; d += 8) {
            __m256 vo = _mm256_loadu_ps(out_attn + d);
            _mm256_storeu_ps(out_attn + d, _mm256_mul_ps(vo, vinv));
        }
        for (; d < (int)head_dim; ++d) {
            out_attn[d] *= inv_sum;
        }
    }
}

} // namespace haven
