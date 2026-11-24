#include "haven/haven_avx2.h"
#include <algorithm>
#include <cstring>
#include <iostream>

namespace haven {

static inline float hsum256_ps(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    __m128 shuf = _mm_movehdup_ps(lo);
    __m128 sums = _mm_add_ps(lo, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}

// Fast AVX2 + FMA Q8_0 dot product
float Avx2Math::vec_dot_q8_0(int n, const block_q8_0* vx, const float* vy) {
    const int nb = n / 32;
    __m256 acc = _mm256_setzero_ps();

    for (int i = 0; i < nb; ++i) {
        const float d = fp16_to_fp32(vx[i].d);
        __m256 d_vec = _mm256_set1_ps(d);
        __m256 block_acc = _mm256_setzero_ps();
        const int8_t* qs = vx[i].qs;
        const float* y = vy + i * 32;

        for (int k = 0; k < 4; ++k) {
            __m128i q8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(qs + k * 8));
            __m256 vx_k = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q8));
            __m256 vy_k = _mm256_loadu_ps(y + k * 8);
            block_acc = _mm256_fmadd_ps(vx_k, vy_k, block_acc);
        }
        acc = _mm256_fmadd_ps(block_acc, d_vec, acc);
    }

    return hsum256_ps(acc);
}

// AVX2 RMS Normalization: y = (x / RMS(x)) * weight
void Avx2Math::rms_norm(float* out, const float* x, const float* weight, int size, float eps) {
    if (!out || !x || !weight || size <= 0) return;
    float sum_sq = 0.0f;
    __m256 acc = _mm256_setzero_ps();

    int i = 0;
    for (; i <= size - 8; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        acc = _mm256_fmadd_ps(vx, vx, acc);
    }
    float buffer[8];
    _mm256_storeu_ps(buffer, acc);
    for (int j = 0; j < 8; ++j) sum_sq += buffer[j];
    for (; i < size; ++i) {
        sum_sq += x[i] * x[i];
    }

    float scale = 1.0f / std::sqrt((sum_sq / (float)size) + eps);
    __m256 vscale = _mm256_set1_ps(scale);

    for (i = 0; i <= size - 8; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        __m256 vw = _mm256_loadu_ps(weight + i);
        __m256 res = _mm256_mul_ps(_mm256_mul_ps(vx, vscale), vw);
        _mm256_storeu_ps(out + i, res);
    }
    for (; i < size; ++i) {
        out[i] = x[i] * scale * weight[i];
    }
}

// Rotary Position Embeddings (RoPE) using Google Gemma HalfRope (SWA rope_dim=128 / Global rope_dim=256)
void Avx2Math::apply_rope(
    float* q, float* k,
    int head_dim, int num_heads, int num_kv_heads,
    int pos, float freq_base, float freq_scale, const float* freq_factors,
    int rope_dim)
{
    if (!q || head_dim <= 0 || num_heads < 0 || num_kv_heads < 0) return;
    if (rope_dim <= 0 || rope_dim > head_dim) rope_dim = head_dim / 2;
    const int num_pairs = rope_dim / 2;
    if (num_pairs <= 0 || freq_base <= 0.0f) return;

    float stack_cos[512];
    float stack_sin[512];
    std::vector<float> heap_cos;
    std::vector<float> heap_sin;
    float* cos_ptr = stack_cos;
    float* sin_ptr = stack_sin;
    if (num_pairs > 512) {
        heap_cos.resize(static_cast<size_t>(num_pairs));
        heap_sin.resize(static_cast<size_t>(num_pairs));
        cos_ptr = heap_cos.data();
        sin_ptr = heap_sin.data();
    }

    const float pos_scale = static_cast<float>(pos) * freq_scale;
    const float inv_rope_dim = -2.0f / static_cast<float>(rope_dim);
    for (int i = 0; i < num_pairs; ++i) {
        float theta = pos_scale * std::pow(freq_base, static_cast<float>(i) * inv_rope_dim);
        if (freq_factors && freq_factors[i] > 0.0f) {
            theta /= freq_factors[i];
        }
        cos_ptr[i] = std::cos(theta);
        sin_ptr[i] = std::sin(theta);
    }

    auto rotate_heads = [&](float* base, int count) {
        for (int h = 0; h < count; ++h) {
            float* head = base + static_cast<size_t>(h) * static_cast<size_t>(head_dim);
            for (int i = 0; i < num_pairs; ++i) {
                const float c = cos_ptr[i];
                const float s = sin_ptr[i];
                const float v0 = head[i];
                const float v1 = head[i + num_pairs];
                head[i]             = v0 * c - v1 * s;
                head[i + num_pairs] = v0 * s + v1 * c;
            }
        }
    };

    rotate_heads(q, num_heads);
    if (k && num_kv_heads > 0) {
        rotate_heads(k, num_kv_heads);
    }
}

// Fast In-Place Softmax
void Avx2Math::softmax(float* x, int size) {
    if (size <= 0) return;
    float max_val = *std::max_element(x, x + size);
    float sum = 0.0f;
    for (int i = 0; i < size; ++i) {
        x[i] = std::exp(x[i] - max_val);
        sum += x[i];
    }
    float inv_sum = 1.0f / (sum > 1e-9f ? sum : 1.0f);
    for (int i = 0; i < size; ++i) {
        x[i] *= inv_sum;
    }
}

// Fast AVX2 Q4_0 dot product (32 values per block)
float Avx2Math::vec_dot_q4_0(int n, const block_q4_0* vx, const float* vy) {
    const int nb = n / 32;
    float sumf = 0.0f;
    __m256 acc = _mm256_setzero_ps();

    for (int i = 0; i < nb; ++i) {
        const float d = fp16_to_fp32(vx[i].d);
        __m256 d_vec = _mm256_set1_ps(d);
        const uint8_t* q = vx[i].qs;
        const float* y = vy + i * 32;

        for (int k = 0; k < 2; ++k) {
            __m256 vy_k0 = _mm256_loadu_ps(y + k * 16 + 0);
            __m256 vy_k1 = _mm256_loadu_ps(y + k * 16 + 8);

            __m256 vx_k0 = _mm256_set_ps(
                (float)(int8_t)((q[k * 8 + 3] >> 4) - 8),
                (float)(int8_t)((q[k * 8 + 3] & 0x0F) - 8),
                (float)(int8_t)((q[k * 8 + 2] >> 4) - 8),
                (float)(int8_t)((q[k * 8 + 2] & 0x0F) - 8),
                (float)(int8_t)((q[k * 8 + 1] >> 4) - 8),
                (float)(int8_t)((q[k * 8 + 1] & 0x0F) - 8),
                (float)(int8_t)((q[k * 8 + 0] >> 4) - 8),
                (float)(int8_t)((q[k * 8 + 0] & 0x0F) - 8)
            );

            __m256 vx_k1 = _mm256_set_ps(
                (float)(int8_t)((q[k * 8 + 7] >> 4) - 8),
                (float)(int8_t)((q[k * 8 + 7] & 0x0F) - 8),
                (float)(int8_t)((q[k * 8 + 6] >> 4) - 8),
                (float)(int8_t)((q[k * 8 + 6] & 0x0F) - 8),
                (float)(int8_t)((q[k * 8 + 5] >> 4) - 8),
                (float)(int8_t)((q[k * 8 + 5] & 0x0F) - 8),
                (float)(int8_t)((q[k * 8 + 4] >> 4) - 8),
                (float)(int8_t)((q[k * 8 + 4] & 0x0F) - 8)
            );

            acc = _mm256_fmadd_ps(_mm256_mul_ps(vx_k0, d_vec), vy_k0, acc);
            acc = _mm256_fmadd_ps(_mm256_mul_ps(vx_k1, d_vec), vy_k1, acc);
        }
    }

    float buffer[8];
    _mm256_storeu_ps(buffer, acc);
    for (int j = 0; j < 8; ++j) sumf += buffer[j];
    return sumf;
}

static inline void get_scale_min_k4(int j, const uint8_t* q, uint8_t* d, uint8_t* m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

// Fast AVX2 + FMA Q4_K dot product (256 values per super-block)
float Avx2Math::vec_dot_q4_K(int n, const block_q4_K* vx, const float* vy) {
    const int nb = n / 256;
    const __m128i mask_lo = _mm_set1_epi8(0x0F);
    float sumf = 0.0f;

    for (int i = 0; i < nb; ++i) {
        const float d = fp16_to_fp32(vx[i].d);
        const float min = fp16_to_fp32(vx[i].dmin);
        const uint8_t* q = vx[i].qs;
        const float* y = vy + i * 256;
        uint8_t sc, m;

        for (int j = 0; j < 4; ++j) {
            get_scale_min_k4(2 * j, vx[i].scales, &sc, &m);
            const float d1 = d * (float)sc;
            const float m1 = min * (float)m;

            get_scale_min_k4(2 * j + 1, vx[i].scales, &sc, &m);
            const float d2 = d * (float)sc;
            const float m2 = min * (float)m;

            const float* y1 = y + j * 64;
            const float* y2 = y + j * 64 + 32;

            __m256 acc_q1 = _mm256_setzero_ps();
            __m256 acc_y1 = _mm256_setzero_ps();
            __m256 acc_q2 = _mm256_setzero_ps();
            __m256 acc_y2 = _mm256_setzero_ps();

            for (int k = 0; k < 4; ++k) {
                const int offset = k * 8;
                __m256 vy1 = _mm256_loadu_ps(y1 + offset);
                __m256 vy2 = _mm256_loadu_ps(y2 + offset);
                acc_y1 = _mm256_add_ps(acc_y1, vy1);
                acc_y2 = _mm256_add_ps(acc_y2, vy2);

                __m128i qb = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(q + offset));
                __m128i qlo = _mm_and_si128(qb, mask_lo);
                __m128i qhi = _mm_and_si128(_mm_srli_epi16(qb, 4), mask_lo);

                __m256 vq1 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(qlo));
                __m256 vq2 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(qhi));
                acc_q1 = _mm256_fmadd_ps(vq1, vy1, acc_q1);
                acc_q2 = _mm256_fmadd_ps(vq2, vy2, acc_q2);
            }

            const float sum_q1 = hsum256_ps(acc_q1);
            const float sum_y1 = hsum256_ps(acc_y1);
            const float sum_q2 = hsum256_ps(acc_q2);
            const float sum_y2 = hsum256_ps(acc_y2);

            sumf += (d1 * sum_q1 - m1 * sum_y1) + (d2 * sum_q2 - m2 * sum_y2);
            q += 32;
        }
    }
    return sumf;
}

// Fast AVX2 + FMA Q6_K dot product (256 values per super-block)
float Avx2Math::vec_dot_q6_K(int n, const block_q6_K* vx, const float* vy) {
    const int nb = n / 256;
    const __m128i mask_lo4 = _mm_set1_epi8(0x0F);
    const __m128i mask_hi2 = _mm_set1_epi8(0x30);
    const __m256i bias32 = _mm256_set1_epi32(32);
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();

    for (int i = 0; i < nb; ++i) {
        const float d = fp16_to_fp32(vx[i].d);
        const uint8_t* ql = vx[i].ql;
        const uint8_t* qh = vx[i].qh;
        const int8_t* sc = vx[i].scales;
        const float* y = vy + i * 256;

        for (int n_chunk = 0; n_chunk < 256; n_chunk += 128) {
            for (int k = 0; k < 4; ++k) {
                const int is = (k < 2) ? 0 : 1;
                const int l_base = k * 8;

                const __m256 vs1 = _mm256_set1_ps(d * (float)sc[is + 0]);
                const __m256 vs2 = _mm256_set1_ps(d * (float)sc[is + 2]);
                const __m256 vs3 = _mm256_set1_ps(d * (float)sc[is + 4]);
                const __m256 vs4 = _mm256_set1_ps(d * (float)sc[is + 6]);

                __m256 vy1 = _mm256_loadu_ps(y + n_chunk + l_base + 0);
                __m256 vy2 = _mm256_loadu_ps(y + n_chunk + l_base + 32);
                __m256 vy3 = _mm256_loadu_ps(y + n_chunk + l_base + 64);
                __m256 vy4 = _mm256_loadu_ps(y + n_chunk + l_base + 96);

                __m128i ql0 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(ql + l_base));
                __m128i ql1 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(ql + l_base + 32));
                __m128i qhb = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(qh + l_base));

                __m128i q1b = _mm_or_si128(_mm_and_si128(ql0, mask_lo4), _mm_and_si128(_mm_slli_epi16(qhb, 4), mask_hi2));
                __m128i q2b = _mm_or_si128(_mm_and_si128(ql1, mask_lo4), _mm_and_si128(_mm_slli_epi16(qhb, 2), mask_hi2));
                __m128i q3b = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(ql0, 4), mask_lo4), _mm_and_si128(qhb, mask_hi2));
                __m128i q4b = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(ql1, 4), mask_lo4), _mm_and_si128(_mm_srli_epi16(qhb, 2), mask_hi2));

                __m256 vq1 = _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(q1b), bias32));
                __m256 vq2 = _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(q2b), bias32));
                __m256 vq3 = _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(q3b), bias32));
                __m256 vq4 = _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(q4b), bias32));

                acc0 = _mm256_fmadd_ps(_mm256_mul_ps(vq1, vy1), vs1, acc0);
                acc1 = _mm256_fmadd_ps(_mm256_mul_ps(vq2, vy2), vs2, acc1);
                acc2 = _mm256_fmadd_ps(_mm256_mul_ps(vq3, vy3), vs3, acc2);
                acc3 = _mm256_fmadd_ps(_mm256_mul_ps(vq4, vy4), vs4, acc3);
            }
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
    return hsum256_ps(_mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3)));
}

// Fast AVX2 + FMA Q5_0 dot product (32 values per block)
float Avx2Math::vec_dot_q5_0(int n, const block_q5_0* vx, const float* vy) {
    const int nb = n / 32;
    float sumf = 0.0f;
    __m256 acc = _mm256_setzero_ps();

    for (int i = 0; i < nb; ++i) {
        const float d = fp16_to_fp32(vx[i].d);
        __m256 vd = _mm256_set1_ps(d);
        uint32_t qh;
        std::memcpy(&qh, vx[i].qh, sizeof(qh));
        const uint8_t* qs = vx[i].qs;
        const float* y = vy + i * 32;

        for (int k = 0; k < 2; ++k) {
            const int offset = k * 8;
            __m256 vy0 = _mm256_loadu_ps(y + offset);
            __m256 vy1 = _mm256_loadu_ps(y + 16 + offset);

            __m256 vw0 = _mm256_set_ps(
                (float)(int32_t)(((qs[offset + 7] & 0x0F) | (((qh >> (offset + 7)) << 4) & 0x10)) - 16),
                (float)(int32_t)(((qs[offset + 6] & 0x0F) | (((qh >> (offset + 6)) << 4) & 0x10)) - 16),
                (float)(int32_t)(((qs[offset + 5] & 0x0F) | (((qh >> (offset + 5)) << 4) & 0x10)) - 16),
                (float)(int32_t)(((qs[offset + 4] & 0x0F) | (((qh >> (offset + 4)) << 4) & 0x10)) - 16),
                (float)(int32_t)(((qs[offset + 3] & 0x0F) | (((qh >> (offset + 3)) << 4) & 0x10)) - 16),
                (float)(int32_t)(((qs[offset + 2] & 0x0F) | (((qh >> (offset + 2)) << 4) & 0x10)) - 16),
                (float)(int32_t)(((qs[offset + 1] & 0x0F) | (((qh >> (offset + 1)) << 4) & 0x10)) - 16),
                (float)(int32_t)(((qs[offset + 0] & 0x0F) | (((qh >> (offset + 0)) << 4) & 0x10)) - 16)
            );

            __m256 vw1 = _mm256_set_ps(
                (float)(int32_t)(((qs[offset + 7] >> 4) | (((qh >> (offset + 7 + 12))) & 0x10)) - 16),
                (float)(int32_t)(((qs[offset + 6] >> 4) | (((qh >> (offset + 6 + 12))) & 0x10)) - 16),
                (float)(int32_t)(((qs[offset + 5] >> 4) | (((qh >> (offset + 5 + 12))) & 0x10)) - 16),
                (float)(int32_t)(((qs[offset + 4] >> 4) | (((qh >> (offset + 4 + 12))) & 0x10)) - 16),
                (float)(int32_t)(((qs[offset + 3] >> 4) | (((qh >> (offset + 3 + 12))) & 0x10)) - 16),
                (float)(int32_t)(((qs[offset + 2] >> 4) | (((qh >> (offset + 2 + 12))) & 0x10)) - 16),
                (float)(int32_t)(((qs[offset + 1] >> 4) | (((qh >> (offset + 1 + 12))) & 0x10)) - 16),
                (float)(int32_t)(((qs[offset + 0] >> 4) | (((qh >> (offset + 0 + 12))) & 0x10)) - 16)
            );

            acc = _mm256_fmadd_ps(_mm256_mul_ps(vw0, vd), vy0, acc);
            acc = _mm256_fmadd_ps(_mm256_mul_ps(vw1, vd), vy1, acc);
        }
    }

    float buffer[8];
    _mm256_storeu_ps(buffer, acc);
    for (int j = 0; j < 8; ++j) sumf += buffer[j];
    return sumf;
}

// Fast AVX2 + FMA Q5_K dot product (256 values per super-block)
float Avx2Math::vec_dot_q5_K(int n, const block_q5_K* vx, const float* vy) {
    const int nb = n / 256;
    float sumf = 0.0f;

    for (int i = 0; i < nb; ++i) {
        const float d = fp16_to_fp32(vx[i].d);
        const float min = fp16_to_fp32(vx[i].dmin);
        const uint8_t* ql = vx[i].qs;
        const uint8_t* qh = vx[i].qh;
        const float* y = vy + i * 256;
        uint8_t sc, m;
        uint8_t u1 = 1, u2 = 2;

        for (int j = 0; j < 4; ++j) {
            get_scale_min_k4(2 * j, vx[i].scales, &sc, &m);
            const float d1 = d * (float)sc;
            const float m1 = min * (float)m;

            get_scale_min_k4(2 * j + 1, vx[i].scales, &sc, &m);
            const float d2 = d * (float)sc;
            const float m2 = min * (float)m;

            const float* y1 = y + j * 64;
            const float* y2 = y + j * 64 + 32;

            __m256 acc_q1 = _mm256_setzero_ps();
            __m256 acc_y1 = _mm256_setzero_ps();
            __m256 acc_q2 = _mm256_setzero_ps();
            __m256 acc_y2 = _mm256_setzero_ps();

            for (int k = 0; k < 4; ++k) {
                const int offset = k * 8;
                __m256 vy1 = _mm256_loadu_ps(y1 + offset);
                __m256 vy2 = _mm256_loadu_ps(y2 + offset);
                acc_y1 = _mm256_add_ps(acc_y1, vy1);
                acc_y2 = _mm256_add_ps(acc_y2, vy2);

                __m256 vq1 = _mm256_set_ps(
                    (float)((ql[offset + 7] & 0x0F) + ((qh[offset + 7] & u1) ? 16 : 0)),
                    (float)((ql[offset + 6] & 0x0F) + ((qh[offset + 6] & u1) ? 16 : 0)),
                    (float)((ql[offset + 5] & 0x0F) + ((qh[offset + 5] & u1) ? 16 : 0)),
                    (float)((ql[offset + 4] & 0x0F) + ((qh[offset + 4] & u1) ? 16 : 0)),
                    (float)((ql[offset + 3] & 0x0F) + ((qh[offset + 3] & u1) ? 16 : 0)),
                    (float)((ql[offset + 2] & 0x0F) + ((qh[offset + 2] & u1) ? 16 : 0)),
                    (float)((ql[offset + 1] & 0x0F) + ((qh[offset + 1] & u1) ? 16 : 0)),
                    (float)((ql[offset + 0] & 0x0F) + ((qh[offset + 0] & u1) ? 16 : 0))
                );
                acc_q1 = _mm256_fmadd_ps(vq1, vy1, acc_q1);

                __m256 vq2 = _mm256_set_ps(
                    (float)((ql[offset + 7] >> 4) + ((qh[offset + 7] & u2) ? 16 : 0)),
                    (float)((ql[offset + 6] >> 4) + ((qh[offset + 6] & u2) ? 16 : 0)),
                    (float)((ql[offset + 5] >> 4) + ((qh[offset + 5] & u2) ? 16 : 0)),
                    (float)((ql[offset + 4] >> 4) + ((qh[offset + 4] & u2) ? 16 : 0)),
                    (float)((ql[offset + 3] >> 4) + ((qh[offset + 3] & u2) ? 16 : 0)),
                    (float)((ql[offset + 2] >> 4) + ((qh[offset + 2] & u2) ? 16 : 0)),
                    (float)((ql[offset + 1] >> 4) + ((qh[offset + 1] & u2) ? 16 : 0)),
                    (float)((ql[offset + 0] >> 4) + ((qh[offset + 0] & u2) ? 16 : 0))
                );
                acc_q2 = _mm256_fmadd_ps(vq2, vy2, acc_q2);
            }

            float buf_q1[8], buf_y1[8], buf_q2[8], buf_y2[8];
            _mm256_storeu_ps(buf_q1, acc_q1);
            _mm256_storeu_ps(buf_y1, acc_y1);
            _mm256_storeu_ps(buf_q2, acc_q2);
            _mm256_storeu_ps(buf_y2, acc_y2);

            float sum_q1 = 0.0f, sum_y1 = 0.0f, sum_q2 = 0.0f, sum_y2 = 0.0f;
            for (int k = 0; k < 8; ++k) {
                sum_q1 += buf_q1[k];
                sum_y1 += buf_y1[k];
                sum_q2 += buf_q2[k];
                sum_y2 += buf_y2[k];
            }

            sumf += (d1 * sum_q1 - m1 * sum_y1) + (d2 * sum_q2 - m2 * sum_y2);
            ql += 32;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
    return sumf;
}

// Fast AVX2 + FMA Q3_K dot product (256 values per super-block)
float Avx2Math::vec_dot_q3_K(int n, const block_q3_K* vx, const float* vy) {
    const int nb = n / 256;
    float sumf = 0.0f;

    const uint32_t kmask1 = 0x03030303;
    const uint32_t kmask2 = 0x0f0f0f0f;
    uint32_t aux[4];
    const int8_t* scales = reinterpret_cast<const int8_t*>(aux);

    for (int i = 0; i < nb; ++i) {
        const float d_all = fp16_to_fp32(vx[i].d);
        const uint8_t* q = vx[i].qs;
        const uint8_t* hm = vx[i].hmask;
        uint8_t m = 1;

        std::memcpy(aux, vx[i].scales, 12);
        uint32_t tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);

        int is = 0;
        const float* y = vy + i * 256;

        for (int n_chunk = 0; n_chunk < 256; n_chunk += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                const float dl1 = d_all * (float)(scales[is++] - 32);
                const float dl2 = d_all * (float)(scales[is++] - 32);

                const float* y1 = y;
                const float* y2 = y + 16;
                y += 32;

                // Sub-block 1 (16 elements from q[0..15] and hm[0..15])
                __m256 vy1_0 = _mm256_loadu_ps(y1 + 0);
                __m256 vy1_1 = _mm256_loadu_ps(y1 + 8);

                __m256 vw1_0 = _mm256_set_ps(
                    (float)((int8_t)((q[7] >> shift) & 3) - ((hm[7] & m) ? 0 : 4)),
                    (float)((int8_t)((q[6] >> shift) & 3) - ((hm[6] & m) ? 0 : 4)),
                    (float)((int8_t)((q[5] >> shift) & 3) - ((hm[5] & m) ? 0 : 4)),
                    (float)((int8_t)((q[4] >> shift) & 3) - ((hm[4] & m) ? 0 : 4)),
                    (float)((int8_t)((q[3] >> shift) & 3) - ((hm[3] & m) ? 0 : 4)),
                    (float)((int8_t)((q[2] >> shift) & 3) - ((hm[2] & m) ? 0 : 4)),
                    (float)((int8_t)((q[1] >> shift) & 3) - ((hm[1] & m) ? 0 : 4)),
                    (float)((int8_t)((q[0] >> shift) & 3) - ((hm[0] & m) ? 0 : 4))
                );

                __m256 vw1_1 = _mm256_set_ps(
                    (float)((int8_t)((q[15] >> shift) & 3) - ((hm[15] & m) ? 0 : 4)),
                    (float)((int8_t)((q[14] >> shift) & 3) - ((hm[14] & m) ? 0 : 4)),
                    (float)((int8_t)((q[13] >> shift) & 3) - ((hm[13] & m) ? 0 : 4)),
                    (float)((int8_t)((q[12] >> shift) & 3) - ((hm[12] & m) ? 0 : 4)),
                    (float)((int8_t)((q[11] >> shift) & 3) - ((hm[11] & m) ? 0 : 4)),
                    (float)((int8_t)((q[10] >> shift) & 3) - ((hm[10] & m) ? 0 : 4)),
                    (float)((int8_t)((q[9] >> shift) & 3) - ((hm[9] & m) ? 0 : 4)),
                    (float)((int8_t)((q[8] >> shift) & 3) - ((hm[8] & m) ? 0 : 4))
                );

                __m256 acc1 = _mm256_fmadd_ps(vw1_0, vy1_0, _mm256_mul_ps(vw1_1, vy1_1));

                // Sub-block 2 (16 elements from q[16..31] and hm[16..31])
                __m256 vy2_0 = _mm256_loadu_ps(y2 + 0);
                __m256 vy2_1 = _mm256_loadu_ps(y2 + 8);

                __m256 vw2_0 = _mm256_set_ps(
                    (float)((int8_t)((q[23] >> shift) & 3) - ((hm[23] & m) ? 0 : 4)),
                    (float)((int8_t)((q[22] >> shift) & 3) - ((hm[22] & m) ? 0 : 4)),
                    (float)((int8_t)((q[21] >> shift) & 3) - ((hm[21] & m) ? 0 : 4)),
                    (float)((int8_t)((q[20] >> shift) & 3) - ((hm[20] & m) ? 0 : 4)),
                    (float)((int8_t)((q[19] >> shift) & 3) - ((hm[19] & m) ? 0 : 4)),
                    (float)((int8_t)((q[18] >> shift) & 3) - ((hm[18] & m) ? 0 : 4)),
                    (float)((int8_t)((q[17] >> shift) & 3) - ((hm[17] & m) ? 0 : 4)),
                    (float)((int8_t)((q[16] >> shift) & 3) - ((hm[16] & m) ? 0 : 4))
                );

                __m256 vw2_1 = _mm256_set_ps(
                    (float)((int8_t)((q[31] >> shift) & 3) - ((hm[31] & m) ? 0 : 4)),
                    (float)((int8_t)((q[30] >> shift) & 3) - ((hm[30] & m) ? 0 : 4)),
                    (float)((int8_t)((q[29] >> shift) & 3) - ((hm[29] & m) ? 0 : 4)),
                    (float)((int8_t)((q[28] >> shift) & 3) - ((hm[28] & m) ? 0 : 4)),
                    (float)((int8_t)((q[27] >> shift) & 3) - ((hm[27] & m) ? 0 : 4)),
                    (float)((int8_t)((q[26] >> shift) & 3) - ((hm[26] & m) ? 0 : 4)),
                    (float)((int8_t)((q[25] >> shift) & 3) - ((hm[25] & m) ? 0 : 4)),
                    (float)((int8_t)((q[24] >> shift) & 3) - ((hm[24] & m) ? 0 : 4))
                );

                __m256 acc2 = _mm256_fmadd_ps(vw2_0, vy2_0, _mm256_mul_ps(vw2_1, vy2_1));

                float b1[8], b2[8];
                _mm256_storeu_ps(b1, acc1);
                _mm256_storeu_ps(b2, acc2);
                float s1 = 0.0f, s2 = 0.0f;
                for (int b = 0; b < 8; ++b) {
                    s1 += b1[b];
                    s2 += b2[b];
                }

                sumf += dl1 * s1 + dl2 * s2;

                shift += 2;
                m <<= 1;
            }
            q += 32;
        }
    }
    return sumf;
}

static constexpr int8_t kvalues_iq4nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
};

static inline __m256 iq4_dot32_avx2(const uint8_t* qs, const float* y, __m128i lut, __m128i mask_lo) {
    const __m128i qb = _mm_loadu_si128(reinterpret_cast<const __m128i*>(qs));
    const __m128i qlo_i8 = _mm_shuffle_epi8(lut, _mm_and_si128(qb, mask_lo));
    const __m128i qhi_i8 = _mm_shuffle_epi8(lut, _mm_and_si128(_mm_srli_epi16(qb, 4), mask_lo));

    const __m256 w0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(qlo_i8));
    const __m256 w1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(qlo_i8, 8)));
    const __m256 w2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(qhi_i8));
    const __m256 w3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(qhi_i8, 8)));

    const __m256 acc0 = _mm256_fmadd_ps(w2, _mm256_loadu_ps(y + 16), _mm256_mul_ps(w0, _mm256_loadu_ps(y + 0)));
    const __m256 acc1 = _mm256_fmadd_ps(w3, _mm256_loadu_ps(y + 24), _mm256_mul_ps(w1, _mm256_loadu_ps(y + 8)));
    return _mm256_add_ps(acc0, acc1);
}

// Fast AVX2 + FMA Q5_1 dot product (32 values per block)
float Avx2Math::vec_dot_q5_1(int n, const block_q5_1* vx, const float* vy) {
    if (!vx || !vy || n < 32) return 0.0f;
    const int nb = n / 32;
    const __m128i mask_lo = _mm_set1_epi8(0x0F);
    const __m256i shifts = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    const __m256i mask10 = _mm256_set1_epi32(0x10);
    __m256 total_acc = _mm256_setzero_ps();

    auto accum_half16 = [&](__m128i q_bytes, uint32_t qh_16, const float* yp, __m256& acc_q, __m256& acc_y) {
        const __m256i xh0 = _mm256_and_si256(
            _mm256_slli_epi32(_mm256_srlv_epi32(_mm256_set1_epi32(static_cast<int>(qh_16)), shifts), 4), mask10);
        const __m256i x0  = _mm256_or_si256(_mm256_cvtepu8_epi32(q_bytes), xh0);
        const __m256 vy0  = _mm256_loadu_ps(yp);
        acc_q = _mm256_fmadd_ps(_mm256_cvtepi32_ps(x0), vy0, acc_q);
        acc_y = _mm256_add_ps(acc_y, vy0);

        const __m256i xh1 = _mm256_and_si256(
            _mm256_slli_epi32(_mm256_srlv_epi32(_mm256_set1_epi32(static_cast<int>(qh_16 >> 8)), shifts), 4), mask10);
        const __m256i x1  = _mm256_or_si256(_mm256_cvtepu8_epi32(_mm_srli_si128(q_bytes, 8)), xh1);
        const __m256 vy1  = _mm256_loadu_ps(yp + 8);
        acc_q = _mm256_fmadd_ps(_mm256_cvtepi32_ps(x1), vy1, acc_q);
        acc_y = _mm256_add_ps(acc_y, vy1);
    };

    for (int i = 0; i < nb; ++i) {
        const __m256 vd = _mm256_set1_ps(fp16_to_fp32(vx[i].d));
        const __m256 vm = _mm256_set1_ps(fp16_to_fp32(vx[i].m));
        uint32_t qh = 0;
        std::memcpy(&qh, vx[i].qh, sizeof(qh));
        const __m128i qb = _mm_loadu_si128(reinterpret_cast<const __m128i*>(vx[i].qs));
        const __m128i qlo = _mm_and_si128(qb, mask_lo);
        const __m128i qhi = _mm_and_si128(_mm_srli_epi16(qb, 4), mask_lo);
        const float* y = vy + i * 32;

        __m256 acc_q = _mm256_setzero_ps();
        __m256 acc_y = _mm256_setzero_ps();
        accum_half16(qlo, qh & 0xFFFFu, y, acc_q, acc_y);
        accum_half16(qhi, qh >> 16,     y + 16, acc_q, acc_y);

        total_acc = _mm256_fmadd_ps(vd, acc_q, total_acc);
        total_acc = _mm256_fmadd_ps(vm, acc_y, total_acc);
    }
    return hsum256_ps(total_acc);
}

// Fast AVX2 + FMA IQ4_NL dot product (32 values per block)
float Avx2Math::vec_dot_iq4_nl(int n, const block_iq4_nl* vx, const float* vy) {
    if (!vx || !vy || n < 32) return 0.0f;
    const int nb = n / 32;
    const __m128i lut = _mm_loadu_si128(reinterpret_cast<const __m128i*>(kvalues_iq4nl));
    const __m128i mask_lo = _mm_set1_epi8(0x0F);
    __m256 acc = _mm256_setzero_ps();

    for (int i = 0; i < nb; ++i) {
        const __m256 vd = _mm256_set1_ps(fp16_to_fp32(vx[i].d));
        const __m256 block_acc = iq4_dot32_avx2(vx[i].qs, vy + i * 32, lut, mask_lo);
        acc = _mm256_fmadd_ps(vd, block_acc, acc);
    }
    return hsum256_ps(acc);
}

// Fast AVX2 + FMA IQ4_XS dot product (256 values per super-block)
float Avx2Math::vec_dot_iq4_xs(int n, const block_iq4_xs* vx, const float* vy) {
    if (!vx || !vy || n < 256) return 0.0f;
    const int nb = n / 256;
    const __m128i lut = _mm_loadu_si128(reinterpret_cast<const __m128i*>(kvalues_iq4nl));
    const __m128i mask_lo = _mm_set1_epi8(0x0F);
    __m256 acc = _mm256_setzero_ps();

    for (int i = 0; i < nb; ++i) {
        const float d = fp16_to_fp32(vx[i].d);
        const uint16_t scales_h = vx[i].scales_h;
        const uint8_t* scales_l = vx[i].scales_l;
        const uint8_t* qs = vx[i].qs;
        const float* y = vy + i * 256;
        for (int ib = 0; ib < 8; ++ib) {
            const int ls = ((scales_l[ib / 2] >> (4 * (ib % 2))) & 0x0F)
                         | (((scales_h >> (2 * ib)) & 3) << 4);
            const __m256 vdl = _mm256_set1_ps(d * static_cast<float>(ls - 32));
            const __m256 sub_acc = iq4_dot32_avx2(qs, y, lut, mask_lo);
            acc = _mm256_fmadd_ps(vdl, sub_acc, acc);
            y += 32;
            qs += 16;
        }
    }
    return hsum256_ps(acc);
}

// Dot product of FP16 weights with FP32 vector (F16C SIMD, 4-accumulator unroll)
float Avx2Math::vec_dot_f16(int n, const uint16_t* vx, const float* vy) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();
    int i = 0;
    for (; i <= n - 32; i += 32) {
        acc0 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(vx + i + 0))),  _mm256_loadu_ps(vy + i + 0),  acc0);
        acc1 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(vx + i + 8))),  _mm256_loadu_ps(vy + i + 8),  acc1);
        acc2 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(vx + i + 16))), _mm256_loadu_ps(vy + i + 16), acc2);
        acc3 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(vx + i + 24))), _mm256_loadu_ps(vy + i + 24), acc3);
    }
    __m256 acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
    for (; i <= n - 8; i += 8) {
        __m256 vy_vec = _mm256_loadu_ps(vy + i);
        __m128i h = _mm_loadu_si128(reinterpret_cast<const __m128i*>(vx + i));
        __m256 vx_vec = _mm256_cvtph_ps(h);
        acc = _mm256_fmadd_ps(vx_vec, vy_vec, acc);
    }
    float sum = hsum256_ps(acc);
    for (; i < n; ++i) {
        sum += fp16_to_fp32(vx[i]) * vy[i];
    }
    return sum;
}

// Dot product of BF16 weights with FP32 vector (AVX2 SIMD, 4-accumulator unroll)
float Avx2Math::vec_dot_bf16(int n, const uint16_t* vx, const float* vy) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();
    int i = 0;
    for (; i <= n - 32; i += 32) {
        __m256 w0 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(vx + i + 0))),  16));
        __m256 w1 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(vx + i + 8))),  16));
        __m256 w2 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(vx + i + 16))), 16));
        __m256 w3 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(vx + i + 24))), 16));
        acc0 = _mm256_fmadd_ps(w0, _mm256_loadu_ps(vy + i + 0),  acc0);
        acc1 = _mm256_fmadd_ps(w1, _mm256_loadu_ps(vy + i + 8),  acc1);
        acc2 = _mm256_fmadd_ps(w2, _mm256_loadu_ps(vy + i + 16), acc2);
        acc3 = _mm256_fmadd_ps(w3, _mm256_loadu_ps(vy + i + 24), acc3);
    }
    __m256 acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
    for (; i <= n - 8; i += 8) {
        __m256 vy_vec = _mm256_loadu_ps(vy + i);
        __m128i bf = _mm_loadu_si128(reinterpret_cast<const __m128i*>(vx + i));
        __m256 vx_vec = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(bf), 16));
        acc = _mm256_fmadd_ps(vx_vec, vy_vec, acc);
    }
    float sum = hsum256_ps(acc);
    for (; i < n; ++i) {
        sum += bf16_to_fp32(vx[i]) * vy[i];
    }
    return sum;
}

// Dot product of FP32 weights with FP32 vector (4-accumulator AVX2 + FMA unroll)
float Avx2Math::vec_dot_f32(int n, const float* vx, const float* vy) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();
    int i = 0;
    for (; i <= n - 32; i += 32) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(vx + i + 0),  _mm256_loadu_ps(vy + i + 0),  acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(vx + i + 8),  _mm256_loadu_ps(vy + i + 8),  acc1);
        acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(vx + i + 16), _mm256_loadu_ps(vy + i + 16), acc2);
        acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(vx + i + 24), _mm256_loadu_ps(vy + i + 24), acc3);
    }
    __m256 acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
    for (; i <= n - 8; i += 8) {
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(vx + i), _mm256_loadu_ps(vy + i), acc);
    }
    float sum = hsum256_ps(acc);
    for (; i < n; ++i) {
        sum += vx[i] * vy[i];
    }
    return sum;
}

// Fast In-Place GELU for Gemma 4 (GeGLU)
void Avx2Math::gelu(float* x, int size) {
    const float sqrt_2_over_pi = 0.7978845608f;
    const float coef = 0.044715f;
    #pragma omp parallel for if(size >= 2048) schedule(static)
    for (int i = 0; i < size; ++i) {
        float xi = x[i];
        float cdf = 0.5f * (1.0f + std::tanh(sqrt_2_over_pi * (xi + coef * xi * xi * xi)));
        x[i] = xi * cdf;
    }
}

// Gemma 4 Logits Softcapping
void Avx2Math::softcap_logits(float* logits, int size, float cap) {
    if (cap <= 0.0f) return;
    float inv_cap = 1.0f / cap;
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < size; ++i) {
        logits[i] = cap * std::tanh(logits[i] * inv_cap);
    }
}

size_t Avx2Math::row_bytes(QuantType type, int cols) {
    if (cols <= 0) return 0;
    const size_t c = static_cast<size_t>(cols);
    switch (type) {
        case QuantType::Q8_0:   return (c / 32) * sizeof(block_q8_0);
        case QuantType::Q4_0:   return (c / 32) * sizeof(block_q4_0);
        case QuantType::Q5_0:   return (c / 32) * sizeof(block_q5_0);
        case QuantType::Q5_1:   return (c / 32) * sizeof(block_q5_1);
        case QuantType::Q4_K:   return (c / 256) * sizeof(block_q4_K);
        case QuantType::Q5_K:   return (c / 256) * sizeof(block_q5_K);
        case QuantType::Q6_K:   return (c / 256) * sizeof(block_q6_K);
        case QuantType::Q3_K:   return (c / 256) * sizeof(block_q3_K);
        case QuantType::IQ4_NL: return (c / 32) * sizeof(block_iq4_nl);
        case QuantType::IQ4_XS: return (c / 256) * sizeof(block_iq4_xs);
        case QuantType::BF16:
        case QuantType::F16:    return c * sizeof(uint16_t);
        case QuantType::F32:    return c * sizeof(float);
        default: return 0;
    }
}

// Dequantizes a single row of a tensor into float buffer (e.g. token embedding lookup)
void Avx2Math::dequantize_row(float* out, const TensorDesc& weight, int row, int cols) {
    if (!out || !weight.data || row < 0 || cols <= 0) return;
    if (weight.size_bytes > 0) {
        const size_t rb = row_bytes(weight.type, cols);
        if (rb == 0 || static_cast<size_t>(row) + 1 > weight.size_bytes / rb) {
            std::memset(out, 0, static_cast<size_t>(cols) * sizeof(float));
            return;
        }
    }

    if (weight.type == QuantType::Q8_0) {
        const auto* blocks = reinterpret_cast<const block_q8_0*>(weight.data);
        const int blocks_per_row = cols / 32;
        const block_q8_0* row_blocks = blocks + (size_t)row * blocks_per_row;
        for (int b = 0; b < blocks_per_row; ++b) {
            float d = fp16_to_fp32(row_blocks[b].d);
            for (int j = 0; j < 32; ++j) {
                out[b * 32 + j] = (float)row_blocks[b].qs[j] * d;
            }
        }
    }
    else if (weight.type == QuantType::Q4_0) {
        const auto* blocks = reinterpret_cast<const block_q4_0*>(weight.data);
        const int blocks_per_row = cols / 32;
        const block_q4_0* row_blocks = blocks + (size_t)row * blocks_per_row;
        for (int b = 0; b < blocks_per_row; ++b) {
            float d = fp16_to_fp32(row_blocks[b].d);
            for (int j = 0; j < 16; ++j) {
                uint8_t byte = row_blocks[b].qs[j];
                out[b * 32 + j * 2 + 0] = (float)(int8_t)((byte & 0x0F) - 8) * d;
                out[b * 32 + j * 2 + 1] = (float)(int8_t)((byte >> 4) - 8) * d;
            }
        }
    }
    else if (weight.type == QuantType::Q4_K) {
        const auto* blocks = reinterpret_cast<const block_q4_K*>(weight.data);
        const int blocks_per_row = cols / 256;
        const block_q4_K* row_blocks = blocks + (size_t)row * blocks_per_row;
        for (int b = 0; b < blocks_per_row; ++b) {
            const float d = fp16_to_fp32(row_blocks[b].d);
            const float min = fp16_to_fp32(row_blocks[b].dmin);
            const uint8_t* q = row_blocks[b].qs;
            float* out_b = out + b * 256;
            uint8_t sc, m;

            for (int j = 0; j < 4; ++j) {
                get_scale_min_k4(2 * j, row_blocks[b].scales, &sc, &m);
                const float d1 = d * (float)sc;
                const float m1 = min * (float)m;
                get_scale_min_k4(2 * j + 1, row_blocks[b].scales, &sc, &m);
                const float d2 = d * (float)sc;
                const float m2 = min * (float)m;

                for (int l = 0; l < 32; ++l) {
                    out_b[j * 64 + l] = d1 * (float)(q[l] & 0x0F) - m1;
                    out_b[j * 64 + l + 32] = d2 * (float)(q[l] >> 4) - m2;
                }
                q += 32;
            }
        }
    }
    else if (weight.type == QuantType::Q6_K) {
        const auto* blocks = reinterpret_cast<const block_q6_K*>(weight.data);
        const int blocks_per_row = cols / 256;
        const block_q6_K* row_blocks = blocks + (size_t)row * blocks_per_row;
        for (int b = 0; b < blocks_per_row; ++b) {
            const float d = fp16_to_fp32(row_blocks[b].d);
            const uint8_t* ql = row_blocks[b].ql;
            const uint8_t* qh = row_blocks[b].qh;
            const int8_t* sc = row_blocks[b].scales;
            float* out_b = out + b * 256;

            for (int n_chunk = 0; n_chunk < 256; n_chunk += 128) {
                for (int l = 0; l < 32; ++l) {
                    int is = l / 16;
                    const int8_t q1 = (int8_t)((ql[l + 0] & 0x0F) | (((qh[l] >> 0) & 3) << 4)) - 32;
                    const int8_t q2 = (int8_t)((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) - 32;
                    const int8_t q3 = (int8_t)((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                    const int8_t q4 = (int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;

                    out_b[n_chunk + l + 0] = d * (float)sc[is + 0] * (float)q1;
                    out_b[n_chunk + l + 32] = d * (float)sc[is + 2] * (float)q2;
                    out_b[n_chunk + l + 64] = d * (float)sc[is + 4] * (float)q3;
                    out_b[n_chunk + l + 96] = d * (float)sc[is + 6] * (float)q4;
                }
                ql += 64;
                qh += 32;
                sc += 8;
            }
        }
    }
    else if (weight.type == QuantType::Q5_0) {
        const auto* blocks = reinterpret_cast<const block_q5_0*>(weight.data);
        const int blocks_per_row = cols / 32;
        const block_q5_0* row_blocks = blocks + (size_t)row * blocks_per_row;
        for (int b = 0; b < blocks_per_row; ++b) {
            float d = fp16_to_fp32(row_blocks[b].d);
            uint32_t qh;
            std::memcpy(&qh, row_blocks[b].qh, sizeof(qh));
            for (int j = 0; j < 16; ++j) {
                const uint8_t xh_0 = ((qh >> (j + 0)) << 4) & 0x10;
                const uint8_t xh_1 = ((qh >> (j + 12))) & 0x10;
                const int32_t x0 = ((row_blocks[b].qs[j] & 0x0F) | xh_0) - 16;
                const int32_t x1 = ((row_blocks[b].qs[j] >> 4) | xh_1) - 16;
                out[b * 32 + j + 0] = (float)x0 * d;
                out[b * 32 + j + 16] = (float)x1 * d;
            }
        }
    }
    else if (weight.type == QuantType::Q5_K) {
        const auto* blocks = reinterpret_cast<const block_q5_K*>(weight.data);
        const int blocks_per_row = cols / 256;
        const block_q5_K* row_blocks = blocks + (size_t)row * blocks_per_row;
        for (int b = 0; b < blocks_per_row; ++b) {
            const float d = fp16_to_fp32(row_blocks[b].d);
            const float min = fp16_to_fp32(row_blocks[b].dmin);
            const uint8_t* ql = row_blocks[b].qs;
            const uint8_t* qh = row_blocks[b].qh;
            float* out_b = out + b * 256;
            int is = 0;
            uint8_t sc, m;
            uint8_t u1 = 1, u2 = 2;
            for (int j = 0; j < 256; j += 64) {
                get_scale_min_k4(is + 0, row_blocks[b].scales, &sc, &m);
                const float d1 = d * (float)sc;
                const float m1 = min * (float)m;
                get_scale_min_k4(is + 1, row_blocks[b].scales, &sc, &m);
                const float d2 = d * (float)sc;
                const float m2 = min * (float)m;
                for (int l = 0; l < 32; ++l) {
                    *out_b++ = d1 * (float)((ql[l] & 0xF) + ((qh[l] & u1) ? 16 : 0)) - m1;
                }
                for (int l = 0; l < 32; ++l) {
                    *out_b++ = d2 * (float)((ql[l] >> 4) + ((qh[l] & u2) ? 16 : 0)) - m2;
                }
                ql += 32;
                is += 2;
                u1 <<= 2;
                u2 <<= 2;
            }
        }
    }
    else if (weight.type == QuantType::Q3_K) {
        const auto* blocks = reinterpret_cast<const block_q3_K*>(weight.data);
        const int blocks_per_row = cols / 256;
        const block_q3_K* row_blocks = blocks + (size_t)row * blocks_per_row;
        const uint32_t kmask1 = 0x03030303;
        const uint32_t kmask2 = 0x0f0f0f0f;
        uint32_t aux[4];
        const int8_t* scales = reinterpret_cast<const int8_t*>(aux);

        for (int b = 0; b < blocks_per_row; ++b) {
            const float d_all = fp16_to_fp32(row_blocks[b].d);
            const uint8_t* q = row_blocks[b].qs;
            const uint8_t* hm = row_blocks[b].hmask;
            uint8_t m = 1;

            std::memcpy(aux, row_blocks[b].scales, 12);
            uint32_t tmp = aux[2];
            aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
            aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
            aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
            aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);

            int is = 0;
            float* out_b = out + b * 256;
            for (int n_chunk = 0; n_chunk < 256; n_chunk += 128) {
                int shift = 0;
                for (int j = 0; j < 4; ++j) {
                    float dl1 = d_all * (float)(scales[is++] - 32);
                    for (int l = 0; l < 16; ++l) {
                        *out_b++ = dl1 * (float)((int8_t)((q[l + 0] >> shift) & 3) - ((hm[l + 0] & m) ? 0 : 4));
                    }
                    float dl2 = d_all * (float)(scales[is++] - 32);
                    for (int l = 0; l < 16; ++l) {
                        *out_b++ = dl2 * (float)((int8_t)((q[l + 16] >> shift) & 3) - ((hm[l + 16] & m) ? 0 : 4));
                    }
                    shift += 2;
                    m <<= 1;
                }
                q += 32;
            }
        }
    }
    else if (weight.type == QuantType::Q5_1) {
        const auto* blocks = reinterpret_cast<const block_q5_1*>(weight.data);
        const int blocks_per_row = cols / 32;
        const block_q5_1* row_blocks = blocks + static_cast<size_t>(row) * static_cast<size_t>(blocks_per_row);
        for (int b = 0; b < blocks_per_row; ++b) {
            const float d = fp16_to_fp32(row_blocks[b].d);
            const float m = fp16_to_fp32(row_blocks[b].m);
            uint32_t qh = 0;
            std::memcpy(&qh, row_blocks[b].qh, sizeof(qh));
            for (int j = 0; j < 16; ++j) {
                const uint8_t xh_0 = static_cast<uint8_t>(((qh >> (j + 0)) << 4) & 0x10);
                const uint8_t xh_1 = static_cast<uint8_t>((qh >> (j + 12)) & 0x10);
                const int x0 = (row_blocks[b].qs[j] & 0x0F) | xh_0;
                const int x1 = (row_blocks[b].qs[j] >> 4) | xh_1;
                out[b * 32 + j + 0]  = static_cast<float>(x0) * d + m;
                out[b * 32 + j + 16] = static_cast<float>(x1) * d + m;
            }
        }
    }
    else if (weight.type == QuantType::IQ4_NL) {
        const auto* blocks = reinterpret_cast<const block_iq4_nl*>(weight.data);
        const int blocks_per_row = cols / 32;
        const block_iq4_nl* row_blocks = blocks + static_cast<size_t>(row) * static_cast<size_t>(blocks_per_row);
        for (int b = 0; b < blocks_per_row; ++b) {
            const float d = fp16_to_fp32(row_blocks[b].d);
            const uint8_t* qs = row_blocks[b].qs;
            float* out_b = out + b * 32;
            for (int j = 0; j < 16; ++j) {
                out_b[j + 0]  = d * static_cast<float>(kvalues_iq4nl[qs[j] & 0x0F]);
                out_b[j + 16] = d * static_cast<float>(kvalues_iq4nl[qs[j] >> 4]);
            }
        }
    }
    else if (weight.type == QuantType::IQ4_XS) {
        const auto* blocks = reinterpret_cast<const block_iq4_xs*>(weight.data);
        const int blocks_per_row = cols / 256;
        const block_iq4_xs* row_blocks = blocks + static_cast<size_t>(row) * static_cast<size_t>(blocks_per_row);
        for (int b = 0; b < blocks_per_row; ++b) {
            const float d = fp16_to_fp32(row_blocks[b].d);
            const uint16_t scales_h = row_blocks[b].scales_h;
            const uint8_t* scales_l = row_blocks[b].scales_l;
            const uint8_t* qs = row_blocks[b].qs;
            float* out_b = out + b * 256;
            for (int ib = 0; ib < 8; ++ib) {
                const int ls = ((scales_l[ib / 2] >> (4 * (ib % 2))) & 0x0F)
                             | (((scales_h >> (2 * ib)) & 3) << 4);
                const float dl = d * static_cast<float>(ls - 32);
                for (int j = 0; j < 16; ++j) {
                    out_b[j + 0]  = dl * static_cast<float>(kvalues_iq4nl[qs[j] & 0x0F]);
                    out_b[j + 16] = dl * static_cast<float>(kvalues_iq4nl[qs[j] >> 4]);
                }
                out_b += 32;
                qs += 16;
            }
        }
    }
    else if (weight.type == QuantType::BF16) {
        const auto* w = reinterpret_cast<const uint16_t*>(weight.data) + static_cast<size_t>(row) * static_cast<size_t>(cols);
        for (int c = 0; c < cols; ++c) {
            out[c] = bf16_to_fp32(w[c]);
        }
    }
    else if (weight.type == QuantType::F16) {
        const auto* w = reinterpret_cast<const uint16_t*>(weight.data) + static_cast<size_t>(row) * static_cast<size_t>(cols);
        for (int c = 0; c < cols; ++c) {
            out[c] = fp16_to_fp32(w[c]);
        }
    }
    else if (weight.type == QuantType::F32) {
        const auto* w = reinterpret_cast<const float*>(weight.data) + static_cast<size_t>(row) * static_cast<size_t>(cols);
        std::memcpy(out, w, static_cast<size_t>(cols) * sizeof(float));
    }
}

// Fast SiLU Activation: x * sigmoid(x)
void Avx2Math::silu(float* x, int size) {
    #pragma omp parallel for if(size >= 2048) schedule(static)
    for (int i = 0; i < size; ++i) {
        x[i] = x[i] / (1.0f + std::exp(-x[i]));
    }
}

// Matrix-Vector Multiplication: y = W * x
void Avx2Math::gemv(float* y, const TensorDesc& weight, const float* x, int rows, int cols) {
    gemv_slice(y, weight, 0, x, rows, cols);
}

template <typename ElemT, int BlockElems, float (*DotFn)(int, const ElemT*, const float*)>
static inline void run_gemv_rows(float* y, const void* data, size_t row_offset, const float* x, int rows, int cols) {
    const auto* base = reinterpret_cast<const ElemT*>(data);
    const size_t stride = static_cast<size_t>(cols / BlockElems);
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < rows; ++r) {
        y[r] = DotFn(cols, base + (row_offset + static_cast<size_t>(r)) * stride, x);
    }
}

float Avx2Math::dot_row(const TensorDesc& weight, size_t row, const float* x, int cols) {
    if (!weight.data || !x || cols <= 0) return 0.0f;
    switch (weight.type) {
        case QuantType::Q8_0:
            return vec_dot_q8_0(cols, reinterpret_cast<const block_q8_0*>(weight.data) + row * static_cast<size_t>(cols / 32), x);
        case QuantType::Q4_0:
            return vec_dot_q4_0(cols, reinterpret_cast<const block_q4_0*>(weight.data) + row * static_cast<size_t>(cols / 32), x);
        case QuantType::Q4_K:
            return vec_dot_q4_K(cols, reinterpret_cast<const block_q4_K*>(weight.data) + row * static_cast<size_t>(cols / 256), x);
        case QuantType::Q6_K:
            return vec_dot_q6_K(cols, reinterpret_cast<const block_q6_K*>(weight.data) + row * static_cast<size_t>(cols / 256), x);
        case QuantType::Q5_0:
            return vec_dot_q5_0(cols, reinterpret_cast<const block_q5_0*>(weight.data) + row * static_cast<size_t>(cols / 32), x);
        case QuantType::Q5_1:
            return vec_dot_q5_1(cols, reinterpret_cast<const block_q5_1*>(weight.data) + row * static_cast<size_t>(cols / 32), x);
        case QuantType::Q5_K:
            return vec_dot_q5_K(cols, reinterpret_cast<const block_q5_K*>(weight.data) + row * static_cast<size_t>(cols / 256), x);
        case QuantType::Q3_K:
            return vec_dot_q3_K(cols, reinterpret_cast<const block_q3_K*>(weight.data) + row * static_cast<size_t>(cols / 256), x);
        case QuantType::IQ4_NL:
            return vec_dot_iq4_nl(cols, reinterpret_cast<const block_iq4_nl*>(weight.data) + row * static_cast<size_t>(cols / 32), x);
        case QuantType::IQ4_XS:
            return vec_dot_iq4_xs(cols, reinterpret_cast<const block_iq4_xs*>(weight.data) + row * static_cast<size_t>(cols / 256), x);
        case QuantType::BF16:
            return vec_dot_bf16(cols, reinterpret_cast<const uint16_t*>(weight.data) + row * static_cast<size_t>(cols), x);
        case QuantType::F16:
            return vec_dot_f16(cols, reinterpret_cast<const uint16_t*>(weight.data) + row * static_cast<size_t>(cols), x);
        case QuantType::F32:
            return vec_dot_f32(cols, reinterpret_cast<const float*>(weight.data) + row * static_cast<size_t>(cols), x);
        default:
            return 0.0f;
    }
}

// ============================================================================
// High-Throughput Integer AVX2 (vpmaddubsw + vpmaddwd) Quantized Activation Kernels
// Pre-quantizes activation vector x into 32-element dynamic-scale INT8 blocks once
// per GEMV / fused layer pass, eliminating FP32 dequantization bottlenecks while
// preserving >0.9999 cosine fidelity even under extreme outlier activations.
// ============================================================================

static inline __m256i mm256_concat_si128(__m128i lo, __m128i hi) {
    return _mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1);
}

static inline int32_t hsum256_epi32(__m256i v) {
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i hi = _mm256_extracti128_si256(v, 1);
    __m128i s = _mm_add_epi32(lo, hi);
    s = _mm_add_epi32(s, _mm_srli_si128(s, 8));
    s = _mm_add_epi32(s, _mm_srli_si128(s, 4));
    return _mm_cvtsi128_si32(s);
}

void Avx2Math::quantize_q8_act(const float* x, int cols, Q8Act32& out) {
    if (!x || cols < 32) {
        out.nb = 0;
        return;
    }
    const int nb = cols / 32;
    out.nb = nb;
    if (nb > Q8Act32::MAX_STACK_BLOCKS) {
        out.qs_heap.resize(static_cast<size_t>(nb) * 32ULL);
        out.d_heap.resize(static_cast<size_t>(nb));
        out.sumf_heap.resize(static_cast<size_t>(nb));
        out.bsum_heap.resize(static_cast<size_t>(nb) * 2ULL);
        out.qs   = out.qs_heap.data();
        out.d    = out.d_heap.data();
        out.sumf = out.sumf_heap.data();
        out.bsum = out.bsum_heap.data();
    } else {
        out.qs   = out.qs_stack;
        out.d    = out.d_stack;
        out.sumf = out.sumf_stack;
        out.bsum = out.bsum_stack;
    }

    const __m256 sign_mask = _mm256_set1_ps(-0.0f);
    const __m256 v127  = _mm256_set1_ps(127.0f);
    const __m256 vn127 = _mm256_set1_ps(-127.0f);

    for (int b = 0; b < nb; ++b) {
        const float* xb = x + static_cast<size_t>(b) * 32ULL;
        const __m256 x0 = _mm256_loadu_ps(xb + 0);
        const __m256 x1 = _mm256_loadu_ps(xb + 8);
        const __m256 x2 = _mm256_loadu_ps(xb + 16);
        const __m256 x3 = _mm256_loadu_ps(xb + 24);

        const __m256 m01 = _mm256_max_ps(_mm256_andnot_ps(sign_mask, x0), _mm256_andnot_ps(sign_mask, x1));
        const __m256 m23 = _mm256_max_ps(_mm256_andnot_ps(sign_mask, x2), _mm256_andnot_ps(sign_mask, x3));
        const __m256 m   = _mm256_max_ps(m01, m23);
        __m128 m128 = _mm_max_ps(_mm256_castps256_ps128(m), _mm256_extractf128_ps(m, 1));
        m128 = _mm_max_ps(m128, _mm_movehdup_ps(m128));
        m128 = _mm_max_ps(m128, _mm_movehl_ps(m128, m128));
        const float max_abs = _mm_cvtss_f32(m128);

        out.sumf[b] = hsum256_ps(_mm256_add_ps(_mm256_add_ps(x0, x1), _mm256_add_ps(x2, x3)));

        if (max_abs < 1e-30f) {
            out.d[b] = 0.0f;
            out.bsum[2 * b + 0] = 0;
            out.bsum[2 * b + 1] = 0;
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(out.qs + static_cast<size_t>(b) * 32ULL), _mm256_setzero_si256());
            continue;
        }

        out.d[b] = max_abs / 127.0f;
        const __m256 vid = _mm256_set1_ps(127.0f / max_abs);

        const __m256i i0 = _mm256_cvtps_epi32(_mm256_min_ps(_mm256_max_ps(_mm256_mul_ps(x0, vid), vn127), v127));
        const __m256i i1 = _mm256_cvtps_epi32(_mm256_min_ps(_mm256_max_ps(_mm256_mul_ps(x1, vid), vn127), v127));
        const __m256i i2 = _mm256_cvtps_epi32(_mm256_min_ps(_mm256_max_ps(_mm256_mul_ps(x2, vid), vn127), v127));
        const __m256i i3 = _mm256_cvtps_epi32(_mm256_min_ps(_mm256_max_ps(_mm256_mul_ps(x3, vid), vn127), v127));

        const __m256i s01 = _mm256_permute4x64_epi64(_mm256_packs_epi32(i0, i1), 0xD8);
        const __m256i s23 = _mm256_permute4x64_epi64(_mm256_packs_epi32(i2, i3), 0xD8);
        const __m256i q8  = _mm256_permute4x64_epi64(_mm256_packs_epi16(s01, s23), 0xD8);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(out.qs + static_cast<size_t>(b) * 32ULL), q8);

        out.bsum[2 * b + 0] = static_cast<int16_t>(hsum256_epi32(_mm256_add_epi32(i0, i1)));
        out.bsum[2 * b + 1] = static_cast<int16_t>(hsum256_epi32(_mm256_add_epi32(i2, i3)));
    }
}

static float dot_row_q6_K_q8(const void* data, size_t row, const Q8Act32& act, int cols) {
    const int nb = cols / 256;
    const auto* vx = reinterpret_cast<const block_q6_K*>(data) + row * static_cast<size_t>(nb);
    const __m256i mask_lo4 = _mm256_set1_epi8(0x0F);
    const __m256i mask_hi2 = _mm256_set1_epi8(0x30);
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();
    float bias_total = 0.0f;

    for (int i = 0; i < nb; ++i) {
        const float d_w = Avx2Math::fp16_to_fp32(vx[i].d);
        const uint8_t* ql = vx[i].ql;
        const uint8_t* qh = vx[i].qh;
        const int8_t* sc = vx[i].scales;
        const int b0 = i * 8;
        float block_bias = 0.0f;

        for (int n_chunk = 0; n_chunk < 2; ++n_chunk) {
            const int yb = b0 + n_chunk * 4;
            const __m256i ql0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ql));
            const __m256i ql1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ql + 32));
            const __m256i qhb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qh));

            const __m256i q1 = _mm256_or_si256(_mm256_and_si256(ql0, mask_lo4), _mm256_and_si256(_mm256_slli_epi16(qhb, 4), mask_hi2));
            const __m256i q2 = _mm256_or_si256(_mm256_and_si256(ql1, mask_lo4), _mm256_and_si256(_mm256_slli_epi16(qhb, 2), mask_hi2));
            const __m256i q3 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(ql0, 4), mask_lo4), _mm256_and_si256(qhb, mask_hi2));
            const __m256i q4 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(ql1, 4), mask_lo4), _mm256_and_si256(_mm256_srli_epi16(qhb, 2), mask_hi2));

            const __m256i y1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(act.qs + static_cast<size_t>(yb + 0) * 32ULL));
            const __m256i y2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(act.qs + static_cast<size_t>(yb + 1) * 32ULL));
            const __m256i y3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(act.qs + static_cast<size_t>(yb + 2) * 32ULL));
            const __m256i y4 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(act.qs + static_cast<size_t>(yb + 3) * 32ULL));

            const __m256i s1 = mm256_concat_si128(_mm_set1_epi16(sc[0]), _mm_set1_epi16(sc[1]));
            const __m256i s2 = mm256_concat_si128(_mm_set1_epi16(sc[2]), _mm_set1_epi16(sc[3]));
            const __m256i s3 = mm256_concat_si128(_mm_set1_epi16(sc[4]), _mm_set1_epi16(sc[5]));
            const __m256i s4 = mm256_concat_si128(_mm_set1_epi16(sc[6]), _mm_set1_epi16(sc[7]));

            const __m256 f1 = _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(q1, y1), s1));
            const __m256 f2 = _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(q2, y2), s2));
            const __m256 f3 = _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(q3, y3), s3));
            const __m256 f4 = _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(q4, y4), s4));

            const float dy1 = act.d[yb + 0];
            const float dy2 = act.d[yb + 1];
            const float dy3 = act.d[yb + 2];
            const float dy4 = act.d[yb + 3];

            acc0 = _mm256_fmadd_ps(f1, _mm256_set1_ps(d_w * dy1), acc0);
            acc1 = _mm256_fmadd_ps(f2, _mm256_set1_ps(d_w * dy2), acc1);
            acc2 = _mm256_fmadd_ps(f3, _mm256_set1_ps(d_w * dy3), acc2);
            acc3 = _mm256_fmadd_ps(f4, _mm256_set1_ps(d_w * dy4), acc3);

            const int16_t* bs = act.bsum + static_cast<size_t>(yb) * 2ULL;
            block_bias += dy1 * static_cast<float>(static_cast<int32_t>(sc[0]) * bs[0] + static_cast<int32_t>(sc[1]) * bs[1])
                        + dy2 * static_cast<float>(static_cast<int32_t>(sc[2]) * bs[2] + static_cast<int32_t>(sc[3]) * bs[3])
                        + dy3 * static_cast<float>(static_cast<int32_t>(sc[4]) * bs[4] + static_cast<int32_t>(sc[5]) * bs[5])
                        + dy4 * static_cast<float>(static_cast<int32_t>(sc[6]) * bs[6] + static_cast<int32_t>(sc[7]) * bs[7]);

            ql += 64;
            qh += 32;
            sc += 8;
        }
        bias_total += d_w * block_bias;
    }
    return hsum256_ps(_mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3))) - 32.0f * bias_total;
}

static float dot_row_iq4_xs_q8(const void* data, size_t row, const Q8Act32& act, int cols) {
    const int nb = cols / 256;
    const auto* vx = reinterpret_cast<const block_iq4_xs*>(data) + row * static_cast<size_t>(nb);
    const __m256i lut256 = _mm256_broadcastsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(kvalues_iq4nl)));
    const __m128i mask_lo = _mm_set1_epi8(0x0F);
    const __m256i ones16  = _mm256_set1_epi16(1);
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();

    for (int i = 0; i < nb; ++i) {
        const float d_w = Avx2Math::fp16_to_fp32(vx[i].d);
        const uint16_t scales_h = vx[i].scales_h;
        const uint8_t* scales_l = vx[i].scales_l;
        const uint8_t* qs = vx[i].qs;
        const int b0 = i * 8;

        for (int ib = 0; ib < 8; ib += 2) {
            const __m128i qb0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(qs));
            const __m128i qb1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(qs + 16));

            const __m256i idx0 = mm256_concat_si128(_mm_and_si128(qb0, mask_lo), _mm_and_si128(_mm_srli_epi16(qb0, 4), mask_lo));
            const __m256i idx1 = mm256_concat_si128(_mm_and_si128(qb1, mask_lo), _mm_and_si128(_mm_srli_epi16(qb1, 4), mask_lo));

            const __m256i w0 = _mm256_shuffle_epi8(lut256, idx0);
            const __m256i w1 = _mm256_shuffle_epi8(lut256, idx1);

            const __m256i y0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(act.qs + static_cast<size_t>(b0 + ib) * 32ULL));
            const __m256i y1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(act.qs + static_cast<size_t>(b0 + ib + 1) * 32ULL));

            const __m256i p0 = _mm256_maddubs_epi16(_mm256_abs_epi8(w0), _mm256_sign_epi8(y0, w0));
            const __m256i p1 = _mm256_maddubs_epi16(_mm256_abs_epi8(w1), _mm256_sign_epi8(y1, w1));

            const int ls0 = (scales_l[ib / 2] & 0x0F) | (((scales_h >> (2 * ib)) & 3) << 4);
            const int ls1 = (scales_l[ib / 2] >> 4)   | (((scales_h >> (2 * (ib + 1))) & 3) << 4);

            acc0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_madd_epi16(p0, ones16)),
                                   _mm256_set1_ps(d_w * static_cast<float>(ls0 - 32) * act.d[b0 + ib]), acc0);
            acc1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_madd_epi16(p1, ones16)),
                                   _mm256_set1_ps(d_w * static_cast<float>(ls1 - 32) * act.d[b0 + ib + 1]), acc1);
            qs += 32;
        }
    }
    return hsum256_ps(_mm256_add_ps(acc0, acc1));
}

static float dot_row_iq4_nl_q8(const void* data, size_t row, const Q8Act32& act, int cols) {
    const int nb = cols / 32;
    const auto* vx = reinterpret_cast<const block_iq4_nl*>(data) + row * static_cast<size_t>(nb);
    const __m256i lut256 = _mm256_broadcastsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(kvalues_iq4nl)));
    const __m128i mask_lo = _mm_set1_epi8(0x0F);
    const __m256i ones16  = _mm256_set1_epi16(1);
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();

    int i = 0;
    for (; i + 1 < nb; i += 2) {
        const __m128i qb0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(vx[i].qs));
        const __m128i qb1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(vx[i + 1].qs));
        const __m256i idx0 = mm256_concat_si128(_mm_and_si128(qb0, mask_lo), _mm_and_si128(_mm_srli_epi16(qb0, 4), mask_lo));
        const __m256i idx1 = mm256_concat_si128(_mm_and_si128(qb1, mask_lo), _mm_and_si128(_mm_srli_epi16(qb1, 4), mask_lo));
        const __m256i w0 = _mm256_shuffle_epi8(lut256, idx0);
        const __m256i w1 = _mm256_shuffle_epi8(lut256, idx1);
        const __m256i y0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(act.qs + static_cast<size_t>(i) * 32ULL));
        const __m256i y1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(act.qs + static_cast<size_t>(i + 1) * 32ULL));
        const __m256i p0 = _mm256_maddubs_epi16(_mm256_abs_epi8(w0), _mm256_sign_epi8(y0, w0));
        const __m256i p1 = _mm256_maddubs_epi16(_mm256_abs_epi8(w1), _mm256_sign_epi8(y1, w1));
        acc0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_madd_epi16(p0, ones16)),
                               _mm256_set1_ps(Avx2Math::fp16_to_fp32(vx[i].d) * act.d[i]), acc0);
        acc1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_madd_epi16(p1, ones16)),
                               _mm256_set1_ps(Avx2Math::fp16_to_fp32(vx[i + 1].d) * act.d[i + 1]), acc1);
    }
    for (; i < nb; ++i) {
        const __m128i qb0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(vx[i].qs));
        const __m256i idx0 = mm256_concat_si128(_mm_and_si128(qb0, mask_lo), _mm_and_si128(_mm_srli_epi16(qb0, 4), mask_lo));
        const __m256i w0 = _mm256_shuffle_epi8(lut256, idx0);
        const __m256i y0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(act.qs + static_cast<size_t>(i) * 32ULL));
        const __m256i p0 = _mm256_maddubs_epi16(_mm256_abs_epi8(w0), _mm256_sign_epi8(y0, w0));
        acc0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_madd_epi16(p0, ones16)),
                               _mm256_set1_ps(Avx2Math::fp16_to_fp32(vx[i].d) * act.d[i]), acc0);
    }
    return hsum256_ps(_mm256_add_ps(acc0, acc1));
}

static float dot_row_q5_1_q8(const void* data, size_t row, const Q8Act32& act, int cols) {
    const int nb = cols / 32;
    const auto* vx = reinterpret_cast<const block_q5_1*>(data) + row * static_cast<size_t>(nb);
    const __m128i mask_lo  = _mm_set1_epi8(0x0F);
    const __m256i ones16   = _mm256_set1_epi16(1);
    const __m256i shuf_qh  = _mm256_setr_epi8(
        0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1,
        2,2,2,2,2,2,2,2, 3,3,3,3,3,3,3,3);
    const __m256i bit_mask = _mm256_set1_epi64x(static_cast<long long>(0x8040201008040201ULL));
    const __m256i val_16   = _mm256_set1_epi8(0x10);
    __m256 acc = _mm256_setzero_ps();
    float m_sum = 0.0f;

    for (int i = 0; i < nb; ++i) {
        uint32_t qh_u32 = 0;
        std::memcpy(&qh_u32, vx[i].qh, sizeof(qh_u32));
        const __m256i qh_b   = _mm256_shuffle_epi8(_mm256_set1_epi32(static_cast<int>(qh_u32)), shuf_qh);
        const __m256i qh_bit = _mm256_and_si256(_mm256_cmpeq_epi8(_mm256_and_si256(qh_b, bit_mask), bit_mask), val_16);

        const __m128i qb = _mm_loadu_si128(reinterpret_cast<const __m128i*>(vx[i].qs));
        const __m256i q4 = mm256_concat_si128(_mm_and_si128(qb, mask_lo), _mm_and_si128(_mm_srli_epi16(qb, 4), mask_lo));
        const __m256i q5 = _mm256_or_si256(q4, qh_bit);

        const __m256i y0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(act.qs + static_cast<size_t>(i) * 32ULL));
        const __m256i p  = _mm256_madd_epi16(_mm256_maddubs_epi16(q5, y0), ones16);
        acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p),
                              _mm256_set1_ps(Avx2Math::fp16_to_fp32(vx[i].d) * act.d[i]), acc);
        m_sum += Avx2Math::fp16_to_fp32(vx[i].m) * act.sumf[i];
    }
    return hsum256_ps(acc) + m_sum;
}

static float dot_row_q5_0_q8(const void* data, size_t row, const Q8Act32& act, int cols) {
    const int nb = cols / 32;
    const auto* vx = reinterpret_cast<const block_q5_0*>(data) + row * static_cast<size_t>(nb);
    const __m128i mask_lo  = _mm_set1_epi8(0x0F);
    const __m256i ones16   = _mm256_set1_epi16(1);
    const __m256i shuf_qh  = _mm256_setr_epi8(
        0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1,
        2,2,2,2,2,2,2,2, 3,3,3,3,3,3,3,3);
    const __m256i bit_mask = _mm256_set1_epi64x(static_cast<long long>(0x8040201008040201ULL));
    const __m256i val_16   = _mm256_set1_epi8(0x10);
    __m256 acc = _mm256_setzero_ps();
    float bias_sum = 0.0f;

    for (int i = 0; i < nb; ++i) {
        uint32_t qh_u32 = 0;
        std::memcpy(&qh_u32, vx[i].qh, sizeof(qh_u32));
        const __m256i qh_b   = _mm256_shuffle_epi8(_mm256_set1_epi32(static_cast<int>(qh_u32)), shuf_qh);
        const __m256i qh_bit = _mm256_and_si256(_mm256_cmpeq_epi8(_mm256_and_si256(qh_b, bit_mask), bit_mask), val_16);

        const __m128i qb = _mm_loadu_si128(reinterpret_cast<const __m128i*>(vx[i].qs));
        const __m256i q4 = mm256_concat_si128(_mm_and_si128(qb, mask_lo), _mm_and_si128(_mm_srli_epi16(qb, 4), mask_lo));
        const __m256i q5 = _mm256_or_si256(q4, qh_bit);

        const __m256i y0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(act.qs + static_cast<size_t>(i) * 32ULL));
        const __m256i p  = _mm256_madd_epi16(_mm256_maddubs_epi16(q5, y0), ones16);
        const float dw_dy = Avx2Math::fp16_to_fp32(vx[i].d) * act.d[i];
        acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p), _mm256_set1_ps(dw_dy), acc);
        bias_sum += dw_dy * static_cast<float>(static_cast<int32_t>(act.bsum[2 * i]) + static_cast<int32_t>(act.bsum[2 * i + 1]));
    }
    return hsum256_ps(acc) - 16.0f * bias_sum;
}

static float dot_row_q4_K_q8(const void* data, size_t row, const Q8Act32& act, int cols) {
    const int nb = cols / 256;
    const auto* vx = reinterpret_cast<const block_q4_K*>(data) + row * static_cast<size_t>(nb);
    const __m256i mask_lo4 = _mm256_set1_epi8(0x0F);
    const __m256i ones16   = _mm256_set1_epi16(1);
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    float min_total = 0.0f;

    for (int i = 0; i < nb; ++i) {
        const float d   = Avx2Math::fp16_to_fp32(vx[i].d);
        const float min = Avx2Math::fp16_to_fp32(vx[i].dmin);
        const uint8_t* q = vx[i].qs;
        const int b0 = i * 8;
        uint8_t sc, m;

        for (int j = 0; j < 4; ++j) {
            const int yb = b0 + 2 * j;
            get_scale_min_k4(2 * j,     vx[i].scales, &sc, &m);
            const float d1 = d * static_cast<float>(sc);
            const float m1 = min * static_cast<float>(m);

            get_scale_min_k4(2 * j + 1, vx[i].scales, &sc, &m);
            const float d2 = d * static_cast<float>(sc);
            const float m2 = min * static_cast<float>(m);

            const __m256i qb  = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q));
            const __m256i qlo = _mm256_and_si256(qb, mask_lo4);
            const __m256i qhi = _mm256_and_si256(_mm256_srli_epi16(qb, 4), mask_lo4);

            const __m256i y1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(act.qs + static_cast<size_t>(yb + 0) * 32ULL));
            const __m256i y2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(act.qs + static_cast<size_t>(yb + 1) * 32ULL));

            const __m256i p1 = _mm256_madd_epi16(_mm256_maddubs_epi16(qlo, y1), ones16);
            const __m256i p2 = _mm256_madd_epi16(_mm256_maddubs_epi16(qhi, y2), ones16);

            acc0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p1), _mm256_set1_ps(d1 * act.d[yb + 0]), acc0);
            acc1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p2), _mm256_set1_ps(d2 * act.d[yb + 1]), acc1);
            min_total += m1 * act.sumf[yb + 0] + m2 * act.sumf[yb + 1];
            q += 32;
        }
    }
    return hsum256_ps(_mm256_add_ps(acc0, acc1)) - min_total;
}

static float dot_row_q5_K_q8(const void* data, size_t row, const Q8Act32& act, int cols) {
    const int nb = cols / 256;
    const auto* vx = reinterpret_cast<const block_q5_K*>(data) + row * static_cast<size_t>(nb);
    const __m256i mask_lo4 = _mm256_set1_epi8(0x0F);
    const __m256i val_16   = _mm256_set1_epi8(0x10);
    const __m256i ones16   = _mm256_set1_epi16(1);
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    float min_total = 0.0f;

    for (int i = 0; i < nb; ++i) {
        const float d   = Avx2Math::fp16_to_fp32(vx[i].d);
        const float min = Avx2Math::fp16_to_fp32(vx[i].dmin);
        const uint8_t* ql = vx[i].qs;
        const __m256i qhb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(vx[i].qh));
        const int b0 = i * 8;
        uint8_t sc, m;
        uint8_t u1 = 1, u2 = 2;

        for (int j = 0; j < 4; ++j) {
            const int yb = b0 + 2 * j;
            get_scale_min_k4(2 * j,     vx[i].scales, &sc, &m);
            const float d1 = d * static_cast<float>(sc);
            const float m1 = min * static_cast<float>(m);

            get_scale_min_k4(2 * j + 1, vx[i].scales, &sc, &m);
            const float d2 = d * static_cast<float>(sc);
            const float m2 = min * static_cast<float>(m);

            const __m256i qb  = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ql));
            const __m256i vu1 = _mm256_set1_epi8(static_cast<char>(u1));
            const __m256i vu2 = _mm256_set1_epi8(static_cast<char>(u2));
            const __m256i h1  = _mm256_and_si256(_mm256_cmpeq_epi8(_mm256_and_si256(qhb, vu1), vu1), val_16);
            const __m256i h2  = _mm256_and_si256(_mm256_cmpeq_epi8(_mm256_and_si256(qhb, vu2), vu2), val_16);
            const __m256i q1  = _mm256_or_si256(_mm256_and_si256(qb, mask_lo4), h1);
            const __m256i q2  = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(qb, 4), mask_lo4), h2);

            const __m256i y1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(act.qs + static_cast<size_t>(yb + 0) * 32ULL));
            const __m256i y2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(act.qs + static_cast<size_t>(yb + 1) * 32ULL));

            const __m256i p1 = _mm256_madd_epi16(_mm256_maddubs_epi16(q1, y1), ones16);
            const __m256i p2 = _mm256_madd_epi16(_mm256_maddubs_epi16(q2, y2), ones16);

            acc0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p1), _mm256_set1_ps(d1 * act.d[yb + 0]), acc0);
            acc1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p2), _mm256_set1_ps(d2 * act.d[yb + 1]), acc1);
            min_total += m1 * act.sumf[yb + 0] + m2 * act.sumf[yb + 1];

            ql += 32;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
    return hsum256_ps(_mm256_add_ps(acc0, acc1)) - min_total;
}

static float dot_row_q8_0_q8(const void* data, size_t row, const Q8Act32& act, int cols) {
    const int nb = cols / 32;
    const auto* vx = reinterpret_cast<const block_q8_0*>(data) + row * static_cast<size_t>(nb);
    const __m256i ones16 = _mm256_set1_epi16(1);
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();

    int i = 0;
    for (; i + 1 < nb; i += 2) {
        const __m256i w0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(vx[i].qs));
        const __m256i w1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(vx[i + 1].qs));
        const __m256i y0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(act.qs + static_cast<size_t>(i) * 32ULL));
        const __m256i y1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(act.qs + static_cast<size_t>(i + 1) * 32ULL));
        const __m256i p0 = _mm256_maddubs_epi16(_mm256_abs_epi8(w0), _mm256_sign_epi8(y0, w0));
        const __m256i p1 = _mm256_maddubs_epi16(_mm256_abs_epi8(w1), _mm256_sign_epi8(y1, w1));
        acc0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_madd_epi16(p0, ones16)),
                               _mm256_set1_ps(Avx2Math::fp16_to_fp32(vx[i].d) * act.d[i]), acc0);
        acc1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_madd_epi16(p1, ones16)),
                               _mm256_set1_ps(Avx2Math::fp16_to_fp32(vx[i + 1].d) * act.d[i + 1]), acc1);
    }
    for (; i < nb; ++i) {
        const __m256i w0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(vx[i].qs));
        const __m256i y0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(act.qs + static_cast<size_t>(i) * 32ULL));
        const __m256i p0 = _mm256_maddubs_epi16(_mm256_abs_epi8(w0), _mm256_sign_epi8(y0, w0));
        acc0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_madd_epi16(p0, ones16)),
                               _mm256_set1_ps(Avx2Math::fp16_to_fp32(vx[i].d) * act.d[i]), acc0);
    }
    return hsum256_ps(_mm256_add_ps(acc0, acc1));
}

static float dot_row_q4_0_q8(const void* data, size_t row, const Q8Act32& act, int cols) {
    const int nb = cols / 32;
    const auto* vx = reinterpret_cast<const block_q4_0*>(data) + row * static_cast<size_t>(nb);
    const __m128i mask_lo = _mm_set1_epi8(0x0F);
    const __m256i ones16  = _mm256_set1_epi16(1);
    __m256 acc = _mm256_setzero_ps();
    float bias_sum = 0.0f;

    for (int i = 0; i < nb; ++i) {
        const __m128i qb  = _mm_loadu_si128(reinterpret_cast<const __m128i*>(vx[i].qs));
        const __m128i qlo = _mm_and_si128(qb, mask_lo);
        const __m128i qhi = _mm_and_si128(_mm_srli_epi16(qb, 4), mask_lo);
        // Interleave [lo0, hi0, lo1, hi1, ...] matching dequantize_row & vec_dot_q4_0
        const __m256i q4 = mm256_concat_si128(_mm_unpacklo_epi8(qlo, qhi), _mm_unpackhi_epi8(qlo, qhi));
        const __m256i y0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(act.qs + static_cast<size_t>(i) * 32ULL));
        const __m256i p  = _mm256_madd_epi16(_mm256_maddubs_epi16(q4, y0), ones16);
        const float dw_dy = Avx2Math::fp16_to_fp32(vx[i].d) * act.d[i];
        acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p), _mm256_set1_ps(dw_dy), acc);
        bias_sum += dw_dy * static_cast<float>(static_cast<int32_t>(act.bsum[2 * i]) + static_cast<int32_t>(act.bsum[2 * i + 1]));
    }
    return hsum256_ps(acc) - 8.0f * bias_sum;
}

Avx2Math::DotRowQ8Fn Avx2Math::get_dot_row_q8_fn(QuantType type, int cols) {
    if (cols < 32) return nullptr;
    switch (type) {
        case QuantType::Q6_K:   return (cols % 256 == 0) ? dot_row_q6_K_q8   : nullptr;
        case QuantType::IQ4_XS: return (cols % 256 == 0) ? dot_row_iq4_xs_q8 : nullptr;
        case QuantType::Q4_K:   return (cols % 256 == 0) ? dot_row_q4_K_q8   : nullptr;
        case QuantType::Q5_K:   return (cols % 256 == 0) ? dot_row_q5_K_q8   : nullptr;
        case QuantType::IQ4_NL: return (cols % 32  == 0) ? dot_row_iq4_nl_q8 : nullptr;
        case QuantType::Q5_1:   return (cols % 32  == 0) ? dot_row_q5_1_q8   : nullptr;
        case QuantType::Q5_0:   return (cols % 32  == 0) ? dot_row_q5_0_q8   : nullptr;
        case QuantType::Q8_0:   return (cols % 32  == 0) ? dot_row_q8_0_q8   : nullptr;
        case QuantType::Q4_0:   return (cols % 32  == 0) ? dot_row_q4_0_q8   : nullptr;
        default: return nullptr;
    }
}

float Avx2Math::dot_row_q8(const TensorDesc& weight, size_t row, const Q8Act32& act, int cols) {
    if (!weight.data || act.nb <= 0 || cols < 32) return 0.0f;
    DotRowQ8Fn fn = get_dot_row_q8_fn(weight.type, cols);
    return fn ? fn(weight.data, row, act, cols) : 0.0f;
}

// Matrix-Vector Multiplication on a contiguous row slice (e.g. expert e inside 3D [cols, rows, E] tensor)
void Avx2Math::gemv_slice(float* y, const TensorDesc& weight, size_t row_offset, const float* x, int rows, int cols) {
    if (!y || !weight.data || !x || rows <= 0 || cols <= 0) return;
    if (weight.size_bytes > 0) {
        const size_t rb = row_bytes(weight.type, cols);
        if (rb == 0 || row_offset + static_cast<size_t>(rows) > weight.size_bytes / rb) {
            std::memset(y, 0, static_cast<size_t>(rows) * sizeof(float));
            return;
        }
    }

    // Use integer AVX2 vpmaddubsw/vpmaddwd GEMV when amortizing Q8 activation quantization over >= 16 rows
    if (rows >= 16) {
        if (DotRowQ8Fn q8_fn = get_dot_row_q8_fn(weight.type, cols)) {
            Q8Act32 act;
            quantize_q8_act(x, cols, act);
            const void* wdata = weight.data;
            #pragma omp parallel for schedule(static)
            for (int r = 0; r < rows; ++r) {
                y[r] = q8_fn(wdata, row_offset + static_cast<size_t>(r), act, cols);
            }
            return;
        }
    }

    switch (weight.type) {
        case QuantType::Q8_0:   run_gemv_rows<block_q8_0,   32,  vec_dot_q8_0>(y, weight.data, row_offset, x, rows, cols); break;
        case QuantType::Q4_0:   run_gemv_rows<block_q4_0,   32,  vec_dot_q4_0>(y, weight.data, row_offset, x, rows, cols); break;
        case QuantType::Q4_K:   run_gemv_rows<block_q4_K,   256, vec_dot_q4_K>(y, weight.data, row_offset, x, rows, cols); break;
        case QuantType::Q6_K:   run_gemv_rows<block_q6_K,   256, vec_dot_q6_K>(y, weight.data, row_offset, x, rows, cols); break;
        case QuantType::Q5_0:   run_gemv_rows<block_q5_0,   32,  vec_dot_q5_0>(y, weight.data, row_offset, x, rows, cols); break;
        case QuantType::Q5_1:   run_gemv_rows<block_q5_1,   32,  vec_dot_q5_1>(y, weight.data, row_offset, x, rows, cols); break;
        case QuantType::Q5_K:   run_gemv_rows<block_q5_K,   256, vec_dot_q5_K>(y, weight.data, row_offset, x, rows, cols); break;
        case QuantType::Q3_K:   run_gemv_rows<block_q3_K,   256, vec_dot_q3_K>(y, weight.data, row_offset, x, rows, cols); break;
        case QuantType::IQ4_NL: run_gemv_rows<block_iq4_nl, 32,  vec_dot_iq4_nl>(y, weight.data, row_offset, x, rows, cols); break;
        case QuantType::IQ4_XS: run_gemv_rows<block_iq4_xs, 256, vec_dot_iq4_xs>(y, weight.data, row_offset, x, rows, cols); break;
        case QuantType::BF16:   run_gemv_rows<uint16_t,     1,   vec_dot_bf16>(y, weight.data, row_offset, x, rows, cols); break;
        case QuantType::F16:    run_gemv_rows<uint16_t,     1,   vec_dot_f16>(y, weight.data, row_offset, x, rows, cols); break;
        case QuantType::F32:    run_gemv_rows<float,        1,   vec_dot_f32>(y, weight.data, row_offset, x, rows, cols); break;
        default: break;
    }
}

// Fast AVX2 In-Place Bias Addition: out[i] += bias[i] (supports F32 and F16)
void Avx2Math::add_bias(float* out, const TensorDesc& bias_tensor, int size) {
    if (!out || !bias_tensor.data || size <= 0) return;
    if (!bias_tensor.shape.empty() && bias_tensor.shape[0] < static_cast<uint64_t>(size)) {
        size = static_cast<int>(bias_tensor.shape[0]);
    }

    if (bias_tensor.type == QuantType::F32) {
        const float* b = reinterpret_cast<const float*>(bias_tensor.data);
        int i = 0;
        for (; i <= size - 8; i += 8) {
            __m256 vo = _mm256_loadu_ps(out + i);
            __m256 vb = _mm256_loadu_ps(b + i);
            _mm256_storeu_ps(out + i, _mm256_add_ps(vo, vb));
        }
        for (; i < size; ++i) {
            out[i] += b[i];
        }
    } else if (bias_tensor.type == QuantType::F16) {
        const uint16_t* b = reinterpret_cast<const uint16_t*>(bias_tensor.data);
        int i = 0;
        for (; i <= size - 8; i += 8) {
            __m128i h = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b + i));
            __m256 vb = _mm256_cvtph_ps(h);
            __m256 vo = _mm256_loadu_ps(out + i);
            _mm256_storeu_ps(out + i, _mm256_add_ps(vo, vb));
        }
        for (; i < size; ++i) {
            out[i] += fp16_to_fp32(b[i]);
        }
    }
}

// Bare-metal AVX2 Vector Norm: sqrt(sum(x^2))
float Avx2Math::vector_norm(const float* x, int size) {
    if (!x || size <= 0) return 0.0f;
    float sum_sq = 0.0f;
    __m256 acc = _mm256_setzero_ps();
    int i = 0;
    for (; i <= size - 8; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        acc = _mm256_fmadd_ps(vx, vx, acc);
    }
    float buffer[8];
    _mm256_storeu_ps(buffer, acc);
    for (int j = 0; j < 8; ++j) sum_sq += buffer[j];
    for (; i < size; ++i) {
        sum_sq += x[i] * x[i];
    }
    return std::sqrt(sum_sq);
}

// Bare-metal AVX2 SIMD Cosine Similarity: (a . b) / (||a|| * ||b||)
float Avx2Math::cosine_similarity(const float* a, const float* b, int size) {
    if (!a || !b || size <= 0) return 0.0f;
    float dot = vec_dot_f32(size, a, b);
    float norm_a = vector_norm(a, size);
    float norm_b = vector_norm(b, size);
    float denom = norm_a * norm_b;
    if (denom <= 1e-8f) return 0.0f;
    return dot / denom;
}

} // namespace haven