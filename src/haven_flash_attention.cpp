#include "haven/haven_flash_attention.h"
#include "haven/haven_avx2.h"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <limits>

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

    double max_score = -std::numeric_limits<double>::infinity();
    double sum_exp = 0.0;
    double block_scores[FLASH_BLOCK_BC];

    double out_stack[512];
    std::vector<double> out_heap;
    double* out_acc = out_stack;
    if (head_dim > 512) {
        out_heap.assign(static_cast<size_t>(head_dim), 0.0);
        out_acc = out_heap.data();
    } else {
        std::memset(out_stack, 0, static_cast<size_t>(head_dim) * sizeof(double));
    }

    const double scale_d = static_cast<double>(scale);
    const double cap_d = static_cast<double>(softcap);

    // Tile across Key/Value cache blocks (fitting completely in L1/L2 cache)
    for (uint32_t block_start = 0; block_start < num_tokens; block_start += FLASH_BLOCK_BC) {
        uint32_t block_end = std::min(num_tokens, (uint32_t)(block_start + FLASH_BLOCK_BC));
        uint32_t current_block_size = block_end - block_start;

        double block_max = -std::numeric_limits<double>::infinity();

        // Compute Q · K^T dot products for this block using AVX2 SIMD
        for (uint32_t i = 0; i < current_block_size; ++i) {
            if (indices && indices[block_start + i] < 0) {
                block_scores[i] = -std::numeric_limits<double>::infinity();
                continue;
            }
            uint32_t abs_s = indices ? static_cast<uint32_t>(indices[block_start + i]) : (block_start + i);
            const float* k_vec = layer_k + static_cast<size_t>(abs_s) * static_cast<size_t>(kv_dim) + kv_head_offset;

            double dot = static_cast<double>(Avx2Math::vec_dot_f32((int)head_dim, q, k_vec));
            double score = dot * scale_d;
            if (softcap > 0.0f) {
                score = cap_d * std::tanh(score / cap_d);
            }
            if (std::isnan(score)) score = -std::numeric_limits<double>::infinity();
            block_scores[i] = score;
            if (score > block_max) block_max = score;
        }

        if (!std::isfinite(block_max)) continue;

        // Exact FP64 Online softmax update: Rescale previous accumulator
        const double new_max = std::max(max_score, block_max);
        const double rescale_prev = std::isfinite(max_score) ? std::exp(max_score - new_max) : 0.0;
        sum_exp *= rescale_prev;

        const __m256d vrescale = _mm256_set1_pd(rescale_prev);
        int d = 0;
        for (; d <= (int)head_dim - 4; d += 4) {
            __m256d vo = _mm256_loadu_pd(out_acc + d);
            _mm256_storeu_pd(out_acc + d, _mm256_mul_pd(vo, vrescale));
        }
        for (; d < (int)head_dim; ++d) {
            out_acc[d] *= rescale_prev;
        }

        // Accumulate new weighted values from this block in exact FP64 AVX2 FMA
        for (uint32_t i = 0; i < current_block_size; ++i) {
            if (!std::isfinite(block_scores[i]) || (indices && indices[block_start + i] < 0)) continue;
            uint32_t abs_s = indices ? static_cast<uint32_t>(indices[block_start + i]) : (block_start + i);
            const float* v_vec = layer_v + static_cast<size_t>(abs_s) * static_cast<size_t>(kv_dim) + kv_head_offset;

            const double exp_val = std::exp(block_scores[i] - new_max);
            sum_exp += exp_val;
            const __m256d vexp = _mm256_set1_pd(exp_val);

            d = 0;
            for (; d <= (int)head_dim - 4; d += 4) {
                __m256d vo = _mm256_loadu_pd(out_acc + d);
                __m256d vv = _mm256_cvtps_pd(_mm_loadu_ps(v_vec + d));
                _mm256_storeu_pd(out_acc + d, _mm256_fmadd_pd(vv, vexp, vo));
            }
            for (; d < (int)head_dim; ++d) {
                out_acc[d] += exp_val * static_cast<double>(v_vec[d]);
            }
        }

        max_score = new_max;
    }

    // Final exact FP64 normalization -> FP32 output
    if (sum_exp > 1e-30) {
        const double inv_sum = 1.0 / sum_exp;
        const __m256d vinv = _mm256_set1_pd(inv_sum);
        int d = 0;
        for (; d <= (int)head_dim - 4; d += 4) {
            __m256d vo = _mm256_mul_pd(_mm256_loadu_pd(out_acc + d), vinv);
            _mm_storeu_ps(out_attn + d, _mm256_cvtpd_ps(vo));
        }
        for (; d < (int)head_dim; ++d) {
            out_attn[d] = static_cast<float>(out_acc[d] * inv_sum);
        }
    }
}

} // namespace haven
