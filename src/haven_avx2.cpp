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

// Fast AVX2 + FMA Q8_0 dot product (dual-accumulator unrolled)
float Avx2Math::vec_dot_q8_0(int n, const block_q8_0* vx, const float* vy) {
    const int nb = n / 32;
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();

    for (int i = 0; i < nb; ++i) {
        const __m256 d_vec = _mm256_set1_ps(fp16_to_fp32(vx[i].d));
        const int8_t* qs = vx[i].qs;
        const float* y = vy + static_cast<size_t>(i) * 32ULL;

        const __m128i q0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(qs));
        const __m128i q1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(qs + 16));

        const __m256 b0 = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q0)),
            _mm256_loadu_ps(y + 0),
            _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(q0, 8))), _mm256_loadu_ps(y + 8)));
        const __m256 b1 = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q1)),
            _mm256_loadu_ps(y + 16),
            _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(q1, 8))), _mm256_loadu_ps(y + 24)));

        acc0 = _mm256_fmadd_ps(b0, d_vec, acc0);
        acc1 = _mm256_fmadd_ps(b1, d_vec, acc1);
    }

    return hsum256_ps(_mm256_add_ps(acc0, acc1));
}

// AVX2 FP64-Accumulated RMS Normalization: y = (x / RMS(x)) * weight
// Accumulates sum-of-squares in exact 53-bit IEEE FP64 (__m256d) to eliminate variance cancellation
// on wide layers (D=5376) and extreme Gemma 4 outlier channels (±900x) at negligible latency cost.
void Avx2Math::rms_norm(float* out, const float* x, const float* weight, int size, float eps) {
    if (!out || !x || !weight || size <= 0) return;
    __m256d acc0 = _mm256_setzero_pd();
    __m256d acc1 = _mm256_setzero_pd();

    int i = 0;
    for (; i <= size - 8; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        __m256d lo = _mm256_cvtps_pd(_mm256_castps256_ps128(vx));
        __m256d hi = _mm256_cvtps_pd(_mm256_extractf128_ps(vx, 1));
        acc0 = _mm256_fmadd_pd(lo, lo, acc0);
        acc1 = _mm256_fmadd_pd(hi, hi, acc1);
    }
    __m256d acc = _mm256_add_pd(acc0, acc1);
    double buffer[4];
    _mm256_storeu_pd(buffer, acc);
    double sum_sq = (buffer[0] + buffer[1]) + (buffer[2] + buffer[3]);
    for (; i < size; ++i) {
        const double xi = static_cast<double>(x[i]);
        sum_sq += xi * xi;
    }

    const double inv_rms = 1.0 / std::sqrt((sum_sq / static_cast<double>(size)) + static_cast<double>(eps));
    const __m256d vscale = _mm256_set1_pd(inv_rms);

    for (i = 0; i <= size - 8; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        __m256 vw = _mm256_loadu_ps(weight + i);
        __m256d x_lo = _mm256_cvtps_pd(_mm256_castps256_ps128(vx));
        __m256d x_hi = _mm256_cvtps_pd(_mm256_extractf128_ps(vx, 1));
        __m256d w_lo = _mm256_cvtps_pd(_mm256_castps256_ps128(vw));
        __m256d w_hi = _mm256_cvtps_pd(_mm256_extractf128_ps(vw, 1));
        __m128 r_lo = _mm256_cvtpd_ps(_mm256_mul_pd(_mm256_mul_pd(x_lo, vscale), w_lo));
        __m128 r_hi = _mm256_cvtpd_ps(_mm256_mul_pd(_mm256_mul_pd(x_hi, vscale), w_hi));
        _mm256_storeu_ps(out + i, _mm256_insertf128_ps(_mm256_castps128_ps256(r_lo), r_hi, 1));
    }
    for (; i < size; ++i) {
        out[i] = static_cast<float>(static_cast<double>(x[i]) * inv_rms * static_cast<double>(weight[i]));
    }
}

// Exact FP64 Rotary Position Embeddings (RoPE) supporting both NeoX split-half (Gemma/Qwen/Phi) and GGUF interleaved (Llama/Mistral)
// Computes frequencies, angles, cos/sin, and 2D rotations in exact IEEE FP64 to eliminate long-context (32K-128K+) phase drift.
void Avx2Math::apply_rope(
    float* q, float* k,
    int head_dim, int num_heads, int num_kv_heads,
    int pos, float freq_base, float freq_scale, const float* freq_factors,
    int rope_dim, bool neox_style)
{
    if (!q || head_dim <= 0 || num_heads < 0 || num_kv_heads < 0) return;
    if (rope_dim <= 0 || rope_dim > head_dim) rope_dim = head_dim;
    const int num_pairs = rope_dim / 2;
    if (num_pairs <= 0 || freq_base <= 0.0f) return;

    double stack_cos[512];
    double stack_sin[512];
    std::vector<double> heap_cos;
    std::vector<double> heap_sin;
    double* cos_ptr = stack_cos;
    double* sin_ptr = stack_sin;
    if (num_pairs > 512) {
        heap_cos.resize(static_cast<size_t>(num_pairs));
        heap_sin.resize(static_cast<size_t>(num_pairs));
        cos_ptr = heap_cos.data();
        sin_ptr = heap_sin.data();
    }

    const double pos_scale = static_cast<double>(pos) * static_cast<double>(freq_scale);
    const double inv_rope_dim = -2.0 / static_cast<double>(rope_dim);
    const double base_d = static_cast<double>(freq_base);
    for (int i = 0; i < num_pairs; ++i) {
        double theta = pos_scale * std::pow(base_d, static_cast<double>(i) * inv_rope_dim);
        if (freq_factors && freq_factors[i] > 0.0f) {
            theta /= static_cast<double>(freq_factors[i]);
        }
        cos_ptr[i] = std::cos(theta);
        sin_ptr[i] = std::sin(theta);
    }

    auto rotate_heads = [&](float* base, int count) {
        for (int h = 0; h < count; ++h) {
            float* head = base + static_cast<size_t>(h) * static_cast<size_t>(head_dim);
            if (neox_style) {
                for (int i = 0; i < num_pairs; ++i) {
                    const double c = cos_ptr[i];
                    const double s = sin_ptr[i];
                    const double v0 = static_cast<double>(head[i]);
                    const double v1 = static_cast<double>(head[i + num_pairs]);
                    head[i]             = static_cast<float>(v0 * c - v1 * s);
                    head[i + num_pairs] = static_cast<float>(v0 * s + v1 * c);
                }
            } else {
                for (int i = 0; i < num_pairs; ++i) {
                    const double c = cos_ptr[i];
                    const double s = sin_ptr[i];
                    const double v0 = static_cast<double>(head[2 * i]);
                    const double v1 = static_cast<double>(head[2 * i + 1]);
                    head[2 * i]     = static_cast<float>(v0 * c - v1 * s);
                    head[2 * i + 1] = static_cast<float>(v0 * s + v1 * c);
                }
            }
        }
    };

    rotate_heads(q, num_heads);
    if (k && num_kv_heads > 0) {
        rotate_heads(k, num_kv_heads);
    }
}

// Exact FP64-Normalized In-Place Softmax
void Avx2Math::softmax(float* x, int size) {
    if (!x || size <= 0) return;
    double max_val = static_cast<double>(x[0]);
    for (int i = 1; i < size; ++i) {
        const double v = static_cast<double>(x[i]);
        if (v > max_val) max_val = v;
    }
    double sum = 0.0;
    for (int i = 0; i < size; ++i) {
        const double e = std::exp(static_cast<double>(x[i]) - max_val);
        x[i] = static_cast<float>(e);
        sum += e;
    }
    const double inv_sum = (sum > 1e-30) ? (1.0 / sum) : 1.0;
    for (int i = 0; i < size; ++i) {
        x[i] = static_cast<float>(static_cast<double>(x[i]) * inv_sum);
    }
}

// Fast AVX2 + FMA Q4_0 dot product (32 values per block, pure SIMD bit-unpack)
float Avx2Math::vec_dot_q4_0(int n, const block_q4_0* vx, const float* vy) {
    const int nb = n / 32;
    const __m128i mask_lo = _mm_set1_epi8(0x0F);
    const __m256i bias8   = _mm256_set1_epi32(8);
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();

    for (int i = 0; i < nb; ++i) {
        const __m256 d_vec = _mm256_set1_ps(fp16_to_fp32(vx[i].d));
        const __m128i qb   = _mm_loadu_si128(reinterpret_cast<const __m128i*>(vx[i].qs));
        const __m128i qlo  = _mm_and_si128(qb, mask_lo);
        const __m128i qhi  = _mm_and_si128(_mm_srli_epi16(qb, 4), mask_lo);
        const float* y     = vy + static_cast<size_t>(i) * 32ULL;

        // GGUF/ggml block_q4_0 split-half layout: low nibbles -> [0..15], high nibbles -> [16..31]
        const __m256 w0 = _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(qlo), bias8));
        const __m256 w1 = _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_srli_si128(qlo, 8)), bias8));
        const __m256 w2 = _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(qhi), bias8));
        const __m256 w3 = _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_srli_si128(qhi, 8)), bias8));

        const __m256 b0 = _mm256_fmadd_ps(w0, _mm256_loadu_ps(y + 0),  _mm256_mul_ps(w1, _mm256_loadu_ps(y + 8)));
        const __m256 b1 = _mm256_fmadd_ps(w2, _mm256_loadu_ps(y + 16), _mm256_mul_ps(w3, _mm256_loadu_ps(y + 24)));
        acc0 = _mm256_fmadd_ps(b0, d_vec, acc0);
        acc1 = _mm256_fmadd_ps(b1, d_vec, acc1);
    }

    return hsum256_ps(_mm256_add_ps(acc0, acc1));
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

// Fast AVX2 + FMA Q4_K dot product (256 values per super-block, vector-accumulated)
float Avx2Math::vec_dot_q4_K(int n, const block_q4_K* vx, const float* vy) {
    const int nb = n / 256;
    const __m128i mask_lo = _mm_set1_epi8(0x0F);
    __m256 total0 = _mm256_setzero_ps();
    __m256 total1 = _mm256_setzero_ps();

    for (int i = 0; i < nb; ++i) {
        const float d = fp16_to_fp32(vx[i].d);
        const float min = fp16_to_fp32(vx[i].dmin);
        const uint8_t* q = vx[i].qs;
        const float* y = vy + static_cast<size_t>(i) * 256ULL;
        uint8_t sc, m;

        for (int j = 0; j < 4; ++j) {
            get_scale_min_k4(2 * j, vx[i].scales, &sc, &m);
            const __m256 vd1 = _mm256_set1_ps(d * static_cast<float>(sc));
            const __m256 vm1 = _mm256_set1_ps(min * static_cast<float>(m));

            get_scale_min_k4(2 * j + 1, vx[i].scales, &sc, &m);
            const __m256 vd2 = _mm256_set1_ps(d * static_cast<float>(sc));
            const __m256 vm2 = _mm256_set1_ps(min * static_cast<float>(m));

            const float* y1 = y + j * 64;
            const float* y2 = y + j * 64 + 32;

            __m256 acc_q1 = _mm256_setzero_ps();
            __m256 acc_y1 = _mm256_setzero_ps();
            __m256 acc_q2 = _mm256_setzero_ps();
            __m256 acc_y2 = _mm256_setzero_ps();

            for (int k = 0; k < 4; ++k) {
                const int offset = k * 8;
                const __m256 vy1 = _mm256_loadu_ps(y1 + offset);
                const __m256 vy2 = _mm256_loadu_ps(y2 + offset);
                acc_y1 = _mm256_add_ps(acc_y1, vy1);
                acc_y2 = _mm256_add_ps(acc_y2, vy2);

                const __m128i qb  = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(q + offset));
                const __m128i qlo = _mm_and_si128(qb, mask_lo);
                const __m128i qhi = _mm_and_si128(_mm_srli_epi16(qb, 4), mask_lo);

                acc_q1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(qlo)), vy1, acc_q1);
                acc_q2 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(qhi)), vy2, acc_q2);
            }

            total0 = _mm256_fmadd_ps(acc_q1, vd1, _mm256_fnmadd_ps(acc_y1, vm1, total0));
            total1 = _mm256_fmadd_ps(acc_q2, vd2, _mm256_fnmadd_ps(acc_y2, vm2, total1));
            q += 32;
        }
    }
    return hsum256_ps(_mm256_add_ps(total0, total1));
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

// Fast AVX2 + FMA Q5_0 dot product (32 values per block, pure SIMD bit-unpack)
float Avx2Math::vec_dot_q5_0(int n, const block_q5_0* vx, const float* vy) {
    const int nb = n / 32;
    const __m128i mask_lo = _mm_set1_epi8(0x0F);
    const __m256i shifts  = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    const __m256i mask10  = _mm256_set1_epi32(0x10);
    const __m256i bias16  = _mm256_set1_epi32(16);
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();

    for (int i = 0; i < nb; ++i) {
        const __m256 vd = _mm256_set1_ps(fp16_to_fp32(vx[i].d));
        uint32_t qh = 0;
        std::memcpy(&qh, vx[i].qh, sizeof(qh));
        const __m128i qb  = _mm_loadu_si128(reinterpret_cast<const __m128i*>(vx[i].qs));
        const __m128i qlo = _mm_and_si128(qb, mask_lo);
        const __m128i qhi = _mm_and_si128(_mm_srli_epi16(qb, 4), mask_lo);
        const float* y    = vy + static_cast<size_t>(i) * 32ULL;

        auto mk_q5 = [&](__m128i q8, uint32_t qh8) -> __m256 {
            const __m256i xh = _mm256_and_si256(_mm256_slli_epi32(_mm256_srlv_epi32(_mm256_set1_epi32(static_cast<int>(qh8)), shifts), 4), mask10);
            return _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_or_si256(_mm256_cvtepu8_epi32(q8), xh), bias16));
        };

        const __m256 b0 = _mm256_fmadd_ps(mk_q5(qlo, qh), _mm256_loadu_ps(y + 0),
                                          _mm256_mul_ps(mk_q5(_mm_srli_si128(qlo, 8), qh >> 8), _mm256_loadu_ps(y + 8)));
        const __m256 b1 = _mm256_fmadd_ps(mk_q5(qhi, qh >> 16), _mm256_loadu_ps(y + 16),
                                          _mm256_mul_ps(mk_q5(_mm_srli_si128(qhi, 8), qh >> 24), _mm256_loadu_ps(y + 24)));
        acc0 = _mm256_fmadd_ps(b0, vd, acc0);
        acc1 = _mm256_fmadd_ps(b1, vd, acc1);
    }

    return hsum256_ps(_mm256_add_ps(acc0, acc1));
}

// Fast AVX2 + FMA Q5_K dot product (256 values per super-block, pure SIMD bit-unpack)
float Avx2Math::vec_dot_q5_K(int n, const block_q5_K* vx, const float* vy) {
    const int nb = n / 256;
    const __m128i mask_lo = _mm_set1_epi8(0x0F);
    const __m128i val16   = _mm_set1_epi8(0x10);
    __m256 total0 = _mm256_setzero_ps();
    __m256 total1 = _mm256_setzero_ps();

    for (int i = 0; i < nb; ++i) {
        const float d = fp16_to_fp32(vx[i].d);
        const float min = fp16_to_fp32(vx[i].dmin);
        const uint8_t* ql = vx[i].qs;
        const uint8_t* qh = vx[i].qh;
        const float* y = vy + static_cast<size_t>(i) * 256ULL;
        uint8_t sc, m;
        uint8_t u1 = 1, u2 = 2;

        for (int j = 0; j < 4; ++j) {
            get_scale_min_k4(2 * j, vx[i].scales, &sc, &m);
            const __m256 vd1 = _mm256_set1_ps(d * static_cast<float>(sc));
            const __m256 vm1 = _mm256_set1_ps(min * static_cast<float>(m));

            get_scale_min_k4(2 * j + 1, vx[i].scales, &sc, &m);
            const __m256 vd2 = _mm256_set1_ps(d * static_cast<float>(sc));
            const __m256 vm2 = _mm256_set1_ps(min * static_cast<float>(m));

            const __m128i vu1 = _mm_set1_epi8(static_cast<char>(u1));
            const __m128i vu2 = _mm_set1_epi8(static_cast<char>(u2));

            const float* y1 = y + j * 64;
            const float* y2 = y + j * 64 + 32;

            __m256 acc_q1 = _mm256_setzero_ps();
            __m256 acc_y1 = _mm256_setzero_ps();
            __m256 acc_q2 = _mm256_setzero_ps();
            __m256 acc_y2 = _mm256_setzero_ps();

            for (int k = 0; k < 4; ++k) {
                const int offset = k * 8;
                const __m256 vy1 = _mm256_loadu_ps(y1 + offset);
                const __m256 vy2 = _mm256_loadu_ps(y2 + offset);
                acc_y1 = _mm256_add_ps(acc_y1, vy1);
                acc_y2 = _mm256_add_ps(acc_y2, vy2);

                const __m128i qb  = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(ql + offset));
                const __m128i qhb = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(qh + offset));
                const __m128i h1  = _mm_and_si128(_mm_cmpeq_epi8(_mm_and_si128(qhb, vu1), vu1), val16);
                const __m128i h2  = _mm_and_si128(_mm_cmpeq_epi8(_mm_and_si128(qhb, vu2), vu2), val16);
                const __m128i q1  = _mm_or_si128(_mm_and_si128(qb, mask_lo), h1);
                const __m128i q2  = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(qb, 4), mask_lo), h2);

                acc_q1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(q1)), vy1, acc_q1);
                acc_q2 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(q2)), vy2, acc_q2);
            }

            total0 = _mm256_fmadd_ps(acc_q1, vd1, _mm256_fnmadd_ps(acc_y1, vm1, total0));
            total1 = _mm256_fmadd_ps(acc_q2, vd2, _mm256_fnmadd_ps(acc_y2, vm2, total1));
            ql += 32;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
    return hsum256_ps(_mm256_add_ps(total0, total1));
}

// Fast AVX2 + FMA Q3_K dot product (256 values per super-block, pure 32-byte SIMD bit-unpack)
float Avx2Math::vec_dot_q3_K(int n, const block_q3_K* vx, const float* vy) {
    const int nb = n / 256;
    const __m256i mask3 = _mm256_set1_epi8(0x03);
    const __m256i v4    = _mm256_set1_epi8(0x04);
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();

    const uint32_t kmask1 = 0x03030303u;
    const uint32_t kmask2 = 0x0f0f0f0fu;
    uint32_t aux[4];
    const int8_t* scales = reinterpret_cast<const int8_t*>(aux);

    for (int i = 0; i < nb; ++i) {
        const float d_all = fp16_to_fp32(vx[i].d);
        const uint8_t* q  = vx[i].qs;
        const __m256i hm32 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(vx[i].hmask));
        uint8_t m = 1;

        std::memcpy(aux, vx[i].scales, 12);
        const uint32_t tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);

        int is = 0;
        const float* y = vy + static_cast<size_t>(i) * 256ULL;

        for (int n_chunk = 0; n_chunk < 2; ++n_chunk) {
            __m256i q32 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q));
            for (int j = 0; j < 4; ++j) {
                const __m256 vdl1 = _mm256_set1_ps(d_all * static_cast<float>(scales[is++] - 32));
                const __m256 vdl2 = _mm256_set1_ps(d_all * static_cast<float>(scales[is++] - 32));
                const __m256i vm  = _mm256_set1_epi8(static_cast<char>(m));

                const __m256i lo2  = _mm256_and_si256(q32, mask3);
                const __m256i hsub = _mm256_andnot_si256(_mm256_cmpeq_epi8(_mm256_and_si256(hm32, vm), vm), v4);
                const __m256i w32  = _mm256_sub_epi8(lo2, hsub);

                const __m128i w_lo16 = _mm256_castsi256_si128(w32);
                const __m128i w_hi16 = _mm256_extracti128_si256(w32, 1);

                const __m256 s1 = _mm256_fmadd_ps(
                    _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(w_lo16)), _mm256_loadu_ps(y + 0),
                    _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(w_lo16, 8))), _mm256_loadu_ps(y + 8)));
                const __m256 s2 = _mm256_fmadd_ps(
                    _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(w_hi16)), _mm256_loadu_ps(y + 16),
                    _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(w_hi16, 8))), _mm256_loadu_ps(y + 24)));

                acc0 = _mm256_fmadd_ps(s1, vdl1, acc0);
                acc1 = _mm256_fmadd_ps(s2, vdl2, acc1);

                y += 32;
                q32 = _mm256_srli_epi16(q32, 2);
                m <<= 1;
            }
            q += 32;
        }
    }
    return hsum256_ps(_mm256_add_ps(acc0, acc1));
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

// Fast AVX2 + FMA IQ4_NL dot product (32 values per block, dual-accumulator unrolled for ILP)
float Avx2Math::vec_dot_iq4_nl(int n, const block_iq4_nl* vx, const float* vy) {
    if (!vx || !vy || n < 32) return 0.0f;
    const int nb = n / 32;
    const __m128i lut = _mm_loadu_si128(reinterpret_cast<const __m128i*>(kvalues_iq4nl));
    const __m128i mask_lo = _mm_set1_epi8(0x0F);
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();

    int i = 0;
    for (; i + 1 < nb; i += 2) {
        const __m256 vd0 = _mm256_set1_ps(fp16_to_fp32(vx[i].d));
        const __m256 vd1 = _mm256_set1_ps(fp16_to_fp32(vx[i + 1].d));
        acc0 = _mm256_fmadd_ps(vd0, iq4_dot32_avx2(vx[i].qs,     vy + i * 32,      lut, mask_lo), acc0);
        acc1 = _mm256_fmadd_ps(vd1, iq4_dot32_avx2(vx[i + 1].qs, vy + (i + 1) * 32, lut, mask_lo), acc1);
    }
    for (; i < nb; ++i) {
        const __m256 vd = _mm256_set1_ps(fp16_to_fp32(vx[i].d));
        acc0 = _mm256_fmadd_ps(vd, iq4_dot32_avx2(vx[i].qs, vy + i * 32, lut, mask_lo), acc0);
    }
    return hsum256_ps(_mm256_add_ps(acc0, acc1));
}

// Fast AVX2 + FMA IQ4_XS dot product (256 values per super-block, dual-accumulator unrolled for ILP)
float Avx2Math::vec_dot_iq4_xs(int n, const block_iq4_xs* vx, const float* vy) {
    if (!vx || !vy || n < 256) return 0.0f;
    const int nb = n / 256;
    const __m128i lut = _mm_loadu_si128(reinterpret_cast<const __m128i*>(kvalues_iq4nl));
    const __m128i mask_lo = _mm_set1_epi8(0x0F);
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();

    for (int i = 0; i < nb; ++i) {
        const float d = fp16_to_fp32(vx[i].d);
        const uint16_t scales_h = vx[i].scales_h;
        const uint8_t* scales_l = vx[i].scales_l;
        const uint8_t* qs = vx[i].qs;
        const float* y = vy + i * 256;
        for (int ib = 0; ib < 8; ib += 2) {
            const int ls0 = (scales_l[ib >> 1] & 0x0F) | (((scales_h >> (2 * ib)) & 3) << 4);
            const int ls1 = (scales_l[ib >> 1] >> 4)   | (((scales_h >> (2 * (ib + 1))) & 3) << 4);
            const __m256 vdl0 = _mm256_set1_ps(d * static_cast<float>(ls0 - 32));
            const __m256 vdl1 = _mm256_set1_ps(d * static_cast<float>(ls1 - 32));
            acc0 = _mm256_fmadd_ps(vdl0, iq4_dot32_avx2(qs,      y,      lut, mask_lo), acc0);
            acc1 = _mm256_fmadd_ps(vdl1, iq4_dot32_avx2(qs + 16, y + 32, lut, mask_lo), acc1);
            y += 64;
            qs += 32;
        }
    }
    return hsum256_ps(_mm256_add_ps(acc0, acc1));
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

// Exact FP64-evaluated In-Place GELU for Gemma 4 (GeGLU)
void Avx2Math::gelu(float* x, int size) {
    constexpr double sqrt_2_over_pi = 0.79788456080286535588;
    constexpr double coef = 0.044715;
    #pragma omp parallel for if(size >= 2048) schedule(static)
    for (int i = 0; i < size; ++i) {
        const double xi = static_cast<double>(x[i]);
        const double cdf = 0.5 * (1.0 + std::tanh(sqrt_2_over_pi * (xi + coef * xi * xi * xi)));
        x[i] = static_cast<float>(xi * cdf);
    }
}

// Exact FP64-evaluated Gemma 4 Logits Softcapping
void Avx2Math::softcap_logits(float* logits, int size, float cap) {
    if (cap <= 0.0f) return;
    const double dcap = static_cast<double>(cap);
    const double inv_cap = 1.0 / dcap;
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < size; ++i) {
        logits[i] = static_cast<float>(dcap * std::tanh(static_cast<double>(logits[i]) * inv_cap));
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
                out[b * 32 + j]      = static_cast<float>(static_cast<int>(byte & 0x0F) - 8) * d;
                out[b * 32 + 16 + j] = static_cast<float>(static_cast<int>(byte >> 4)   - 8) * d;
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

// Exact FP64-evaluated SiLU Activation: x * sigmoid(x)
void Avx2Math::silu(float* x, int size) {
    #pragma omp parallel for if(size >= 2048) schedule(static)
    for (int i = 0; i < size; ++i) {
        const double xi = static_cast<double>(x[i]);
        x[i] = static_cast<float>(xi / (1.0 + std::exp(-xi)));
    }
}

// Matrix-Vector Multiplication: y = W * x
void Avx2Math::gemv(float* y, const TensorDesc& weight, const float* x, int rows, int cols) {
    gemv_slice(y, weight, 0, x, rows, cols);
}

void Avx2Math::gemv_f32(float* y, const TensorDesc& weight, const float* x, int rows, int cols) {
    gemv_slice_f32(y, weight, 0, x, rows, cols);
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

// Exact scalar dequantizer for a single weight W[row, c] — used exclusively for sparse FP32 outlier compensation!
template <QuantType QT>
static inline float dequant_weight_elem(const void* data, size_t row, int cols, int c) {
    if constexpr (QT == QuantType::IQ4_XS) {
        const auto* blk = reinterpret_cast<const block_iq4_xs*>(data) + row * static_cast<size_t>(cols / 256) + static_cast<size_t>(c >> 8);
        const int ib = (c & 255) >> 5;
        const int j  = c & 31;
        const int ls = ((blk->scales_l[ib >> 1] >> (4 * (ib & 1))) & 0x0F) | (((blk->scales_h >> (2 * ib)) & 3) << 4);
        const uint8_t byte = blk->qs[(ib << 4) + (j & 15)];
        const int nib = (j < 16) ? (byte & 0x0F) : (byte >> 4);
        return Avx2Math::fp16_to_fp32(blk->d) * static_cast<float>(ls - 32) * static_cast<float>(kvalues_iq4nl[nib]);
    } else if constexpr (QT == QuantType::IQ4_NL) {
        const auto* blk = reinterpret_cast<const block_iq4_nl*>(data) + row * static_cast<size_t>(cols / 32) + static_cast<size_t>(c >> 5);
        const int j = c & 31;
        const uint8_t byte = blk->qs[j & 15];
        const int nib = (j < 16) ? (byte & 0x0F) : (byte >> 4);
        return Avx2Math::fp16_to_fp32(blk->d) * static_cast<float>(kvalues_iq4nl[nib]);
    } else if constexpr (QT == QuantType::Q4_K) {
        const auto* blk = reinterpret_cast<const block_q4_K*>(data) + row * static_cast<size_t>(cols / 256) + static_cast<size_t>(c >> 8);
        const int sub = (c & 255) >> 5;
        const int j = sub >> 1;
        const int l = c & 31;
        uint8_t sc, mn;
        get_scale_min_k4(sub, blk->scales, &sc, &mn);
        const uint8_t byte = blk->qs[(j << 5) + l];
        const int q = (sub & 1) ? (byte >> 4) : (byte & 0x0F);
        return Avx2Math::fp16_to_fp32(blk->d) * static_cast<float>(sc) * static_cast<float>(q)
             - Avx2Math::fp16_to_fp32(blk->dmin) * static_cast<float>(mn);
    } else if constexpr (QT == QuantType::Q5_K) {
        const auto* blk = reinterpret_cast<const block_q5_K*>(data) + row * static_cast<size_t>(cols / 256) + static_cast<size_t>(c >> 8);
        const int sub = (c & 255) >> 5;
        const int j = sub >> 1;
        const int l = c & 31;
        uint8_t sc, mn;
        get_scale_min_k4(sub, blk->scales, &sc, &mn);
        const uint8_t byte = blk->qs[(j << 5) + l];
        int q = (sub & 1) ? (byte >> 4) : (byte & 0x0F);
        if (blk->qh[l] & static_cast<uint8_t>(1u << sub)) q += 16;
        return Avx2Math::fp16_to_fp32(blk->d) * static_cast<float>(sc) * static_cast<float>(q)
             - Avx2Math::fp16_to_fp32(blk->dmin) * static_cast<float>(mn);
    } else if constexpr (QT == QuantType::Q6_K) {
        const auto* blk = reinterpret_cast<const block_q6_K*>(data) + row * static_cast<size_t>(cols / 256) + static_cast<size_t>(c >> 8);
        const int nc  = (c & 255) >> 7;
        const int rem = c & 127;
        const int grp = rem >> 5;
        const int l   = rem & 31;
        const int is  = l >> 4;
        const uint8_t* ql = blk->ql + (nc << 6);
        const uint8_t* qh = blk->qh + (nc << 5);
        const int8_t*  sc = blk->scales + (nc << 3);
        const uint8_t qb = ql[((grp & 1) << 5) + l];
        const uint32_t nib = (grp >= 2) ? (qb >> 4) : (qb & 0x0Fu);
        const uint32_t hi2 = (qh[l] >> (grp * 2)) & 3u;
        const int q = static_cast<int>(nib | (hi2 << 4)) - 32;
        return Avx2Math::fp16_to_fp32(blk->d) * static_cast<float>(sc[(grp << 1) + is]) * static_cast<float>(q);
    } else if constexpr (QT == QuantType::Q5_1) {
        const auto* blk = reinterpret_cast<const block_q5_1*>(data) + row * static_cast<size_t>(cols / 32) + static_cast<size_t>(c >> 5);
        const int j = c & 31;
        uint32_t qh = 0;
        std::memcpy(&qh, blk->qh, sizeof(qh));
        const uint8_t byte = blk->qs[j & 15];
        const uint32_t nib = (j < 16) ? (byte & 0x0Fu) : (byte >> 4);
        const uint32_t hi  = ((qh >> j) & 1u) << 4;
        return Avx2Math::fp16_to_fp32(blk->d) * static_cast<float>(nib | hi) + Avx2Math::fp16_to_fp32(blk->m);
    } else if constexpr (QT == QuantType::Q5_0) {
        const auto* blk = reinterpret_cast<const block_q5_0*>(data) + row * static_cast<size_t>(cols / 32) + static_cast<size_t>(c >> 5);
        const int j = c & 31;
        uint32_t qh = 0;
        std::memcpy(&qh, blk->qh, sizeof(qh));
        const uint8_t byte = blk->qs[j & 15];
        const uint32_t nib = (j < 16) ? (byte & 0x0Fu) : (byte >> 4);
        const uint32_t hi  = ((qh >> j) & 1u) << 4;
        return Avx2Math::fp16_to_fp32(blk->d) * static_cast<float>(static_cast<int>(nib | hi) - 16);
    } else if constexpr (QT == QuantType::Q8_0) {
        const auto* blk = reinterpret_cast<const block_q8_0*>(data) + row * static_cast<size_t>(cols / 32) + static_cast<size_t>(c >> 5);
        return Avx2Math::fp16_to_fp32(blk->d) * static_cast<float>(blk->qs[c & 31]);
    } else if constexpr (QT == QuantType::Q4_0) {
        const auto* blk = reinterpret_cast<const block_q4_0*>(data) + row * static_cast<size_t>(cols / 32) + static_cast<size_t>(c >> 5);
        const int j = c & 31;
        const uint8_t byte = blk->qs[j & 15];
        const int q = (j < 16) ? (byte & 0x0F) : (byte >> 4);
        return Avx2Math::fp16_to_fp32(blk->d) * static_cast<float>(q - 8);
    }
    return 0.0f;
}

template <QuantType QT>
static inline float dot_outliers(const void* data, size_t row, int cols, const Q8Outlier* outliers, int n_outliers) {
    float acc = 0.0f;
    for (int i = 0; i < n_outliers; ++i) {
        acc += dequant_weight_elem<QT>(data, row, cols, outliers[i].col) * outliers[i].val;
    }
    return acc;
}

void Avx2Math::quantize_q8_act(const float* x, int cols, Q8Act32& out, bool extract_outliers) {
    out.n_outliers = 0;
    out.use_f32 = false;
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
    constexpr int ADAPTIVE_F32_OUTLIER_LIMIT = 4;

    for (int b = 0; b < nb; ++b) {
        const float* xb = x + static_cast<size_t>(b) * 32ULL;
        __m256 x0 = _mm256_loadu_ps(xb + 0);
        __m256 x1 = _mm256_loadu_ps(xb + 8);
        __m256 x2 = _mm256_loadu_ps(xb + 16);
        __m256 x3 = _mm256_loadu_ps(xb + 24);

        const __m256 ax0 = _mm256_andnot_ps(sign_mask, x0);
        const __m256 ax1 = _mm256_andnot_ps(sign_mask, x1);
        const __m256 ax2 = _mm256_andnot_ps(sign_mask, x2);
        const __m256 ax3 = _mm256_andnot_ps(sign_mask, x3);

        const __m256 m = _mm256_max_ps(_mm256_max_ps(ax0, ax1), _mm256_max_ps(ax2, ax3));
        __m128 m128 = _mm_max_ps(_mm256_castps256_ps128(m), _mm256_extractf128_ps(m, 1));
        m128 = _mm_max_ps(m128, _mm_movehdup_ps(m128));
        m128 = _mm_max_ps(m128, _mm_movehl_ps(m128, m128));
        float max_abs = _mm_cvtss_f32(m128);

        // Sparse Outlier-Split Q8 + Adaptive Exact FP32 Promotion:
        // - 1..8 isolated spikes: peel into sparse FP32 residual (keeps fast integer Q8 SIMD!)
        // - >8 spikes across the vector (e.g. un-baked stock post-GeGLU / dense outlier channels):
        //   promote directly to exact FP32 dot product (faster than scalar outlier loops & 1.0000 exact!)
        if (extract_outliers && max_abs > 1e-4f && out.n_outliers < Q8Act32::MAX_OUTLIERS) {
            float sum_abs = hsum256_ps(_mm256_add_ps(_mm256_add_ps(ax0, ax1), _mm256_add_ps(ax2, ax3)));
            if (max_abs * 4.0f > sum_abs && (sum_abs - max_abs) > 1e-6f) {
                alignas(32) float tmp[32];
                std::memcpy(tmp, xb, 32 * sizeof(float));
                for (int peel = 0; peel < 4 && out.n_outliers < Q8Act32::MAX_OUTLIERS; ++peel) {
                    if (!(max_abs > 1e-4f && max_abs * 4.0f > sum_abs && (sum_abs - max_abs) > 1e-6f)) break;
                    int best_j = 0;
                    float best_a = -1.0f;
                    for (int j = 0; j < 32; ++j) {
                        const float a = std::fabs(tmp[j]);
                        if (a > best_a) { best_a = a; best_j = j; }
                    }
                    out.outliers[out.n_outliers++] = { b * 32 + best_j, tmp[best_j] };
                    if (out.n_outliers > ADAPTIVE_F32_OUTLIER_LIMIT) {
                        out.use_f32 = true;
                        out.n_outliers = 0;
                        out.x_f32_buf.assign(x, x + static_cast<size_t>(cols));
                        return;
                    }
                    sum_abs -= best_a;
                    tmp[best_j] = 0.0f;
                    float next_max = 0.0f;
                    for (int j = 0; j < 32; ++j) {
                        const float a = std::fabs(tmp[j]);
                        if (a > next_max) next_max = a;
                    }
                    max_abs = next_max;
                }
                x0 = _mm256_load_ps(tmp + 0);
                x1 = _mm256_load_ps(tmp + 8);
                x2 = _mm256_load_ps(tmp + 16);
                x3 = _mm256_load_ps(tmp + 24);
            }
        }

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
    if (act.use_f32) return Avx2Math::vec_dot_q6_K(cols, vx, act.x_f32_buf.data());
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
    float res = hsum256_ps(_mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3))) - 32.0f * bias_total;
    if (act.n_outliers > 0) res += dot_outliers<QuantType::Q6_K>(data, row, cols, act.outliers, act.n_outliers);
    return res;
}

static float dot_row_iq4_xs_q8(const void* data, size_t row, const Q8Act32& act, int cols) {
    const int nb = cols / 256;
    const auto* vx = reinterpret_cast<const block_iq4_xs*>(data) + row * static_cast<size_t>(nb);
    if (act.use_f32) return Avx2Math::vec_dot_iq4_xs(cols, vx, act.x_f32_buf.data());
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
    float res = hsum256_ps(_mm256_add_ps(acc0, acc1));
    if (act.n_outliers > 0) res += dot_outliers<QuantType::IQ4_XS>(data, row, cols, act.outliers, act.n_outliers);
    return res;
}

static float dot_row_iq4_nl_q8(const void* data, size_t row, const Q8Act32& act, int cols) {
    const int nb = cols / 32;
    const auto* vx = reinterpret_cast<const block_iq4_nl*>(data) + row * static_cast<size_t>(nb);
    if (act.use_f32) return Avx2Math::vec_dot_iq4_nl(cols, vx, act.x_f32_buf.data());
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
    float res = hsum256_ps(_mm256_add_ps(acc0, acc1));
    if (act.n_outliers > 0) res += dot_outliers<QuantType::IQ4_NL>(data, row, cols, act.outliers, act.n_outliers);
    return res;
}

static float dot_row_q5_1_q8(const void* data, size_t row, const Q8Act32& act, int cols) {
    const int nb = cols / 32;
    const auto* vx = reinterpret_cast<const block_q5_1*>(data) + row * static_cast<size_t>(nb);
    if (act.use_f32) return Avx2Math::vec_dot_q5_1(cols, vx, act.x_f32_buf.data());
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
    float res = hsum256_ps(acc) + m_sum;
    if (act.n_outliers > 0) res += dot_outliers<QuantType::Q5_1>(data, row, cols, act.outliers, act.n_outliers);
    return res;
}

static float dot_row_q5_0_q8(const void* data, size_t row, const Q8Act32& act, int cols) {
    const int nb = cols / 32;
    const auto* vx = reinterpret_cast<const block_q5_0*>(data) + row * static_cast<size_t>(nb);
    if (act.use_f32) return Avx2Math::vec_dot_q5_0(cols, vx, act.x_f32_buf.data());
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
    float res = hsum256_ps(acc) - 16.0f * bias_sum;
    if (act.n_outliers > 0) res += dot_outliers<QuantType::Q5_0>(data, row, cols, act.outliers, act.n_outliers);
    return res;
}

static float dot_row_q4_K_q8(const void* data, size_t row, const Q8Act32& act, int cols) {
    const int nb = cols / 256;
    const auto* vx = reinterpret_cast<const block_q4_K*>(data) + row * static_cast<size_t>(nb);
    if (act.use_f32) return Avx2Math::vec_dot_q4_K(cols, vx, act.x_f32_buf.data());
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
    float res = hsum256_ps(_mm256_add_ps(acc0, acc1)) - min_total;
    if (act.n_outliers > 0) res += dot_outliers<QuantType::Q4_K>(data, row, cols, act.outliers, act.n_outliers);
    return res;
}

static float dot_row_q5_K_q8(const void* data, size_t row, const Q8Act32& act, int cols) {
    const int nb = cols / 256;
    const auto* vx = reinterpret_cast<const block_q5_K*>(data) + row * static_cast<size_t>(nb);
    if (act.use_f32) return Avx2Math::vec_dot_q5_K(cols, vx, act.x_f32_buf.data());
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
    float res = hsum256_ps(_mm256_add_ps(acc0, acc1)) - min_total;
    if (act.n_outliers > 0) res += dot_outliers<QuantType::Q5_K>(data, row, cols, act.outliers, act.n_outliers);
    return res;
}

static float dot_row_q8_0_q8(const void* data, size_t row, const Q8Act32& act, int cols) {
    const int nb = cols / 32;
    const auto* vx = reinterpret_cast<const block_q8_0*>(data) + row * static_cast<size_t>(nb);
    if (act.use_f32) return Avx2Math::vec_dot_q8_0(cols, vx, act.x_f32_buf.data());
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
    float res = hsum256_ps(_mm256_add_ps(acc0, acc1));
    if (act.n_outliers > 0) res += dot_outliers<QuantType::Q8_0>(data, row, cols, act.outliers, act.n_outliers);
    return res;
}

static float dot_row_q4_0_q8(const void* data, size_t row, const Q8Act32& act, int cols) {
    const int nb = cols / 32;
    const auto* vx = reinterpret_cast<const block_q4_0*>(data) + row * static_cast<size_t>(nb);
    if (act.use_f32) return Avx2Math::vec_dot_q4_0(cols, vx, act.x_f32_buf.data());
    const __m128i mask_lo = _mm_set1_epi8(0x0F);
    const __m256i ones16  = _mm256_set1_epi16(1);
    __m256 acc = _mm256_setzero_ps();
    float bias_sum = 0.0f;

    for (int i = 0; i < nb; ++i) {
        const __m128i qb  = _mm_loadu_si128(reinterpret_cast<const __m128i*>(vx[i].qs));
        const __m128i qlo = _mm_and_si128(qb, mask_lo);
        const __m128i qhi = _mm_and_si128(_mm_srli_epi16(qb, 4), mask_lo);
        // GGUF split-half layout: [lo0..lo15, hi0..hi15] matching Q5_0/Q5_1/IQ4_NL
        const __m256i q4 = mm256_concat_si128(qlo, qhi);
        const __m256i y0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(act.qs + static_cast<size_t>(i) * 32ULL));
        const __m256i p  = _mm256_madd_epi16(_mm256_maddubs_epi16(q4, y0), ones16);
        const float dw_dy = Avx2Math::fp16_to_fp32(vx[i].d) * act.d[i];
        acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p), _mm256_set1_ps(dw_dy), acc);
        bias_sum += dw_dy * static_cast<float>(static_cast<int32_t>(act.bsum[2 * i]) + static_cast<int32_t>(act.bsum[2 * i + 1]));
    }
    float res = hsum256_ps(acc) - 8.0f * bias_sum;
    if (act.n_outliers > 0) res += dot_outliers<QuantType::Q4_0>(data, row, cols, act.outliers, act.n_outliers);
    return res;
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

// Matrix-Vector Multiplication on a contiguous row slice (exact FP32 activations — zero Q8 rounding!)
void Avx2Math::gemv_slice_f32(float* y, const TensorDesc& weight, size_t row_offset, const float* x, int rows, int cols) {
    if (!y || !weight.data || !x || rows <= 0 || cols <= 0) return;
    if (weight.size_bytes > 0) {
        const size_t rb = row_bytes(weight.type, cols);
        if (rb == 0 || row_offset + static_cast<size_t>(rows) > weight.size_bytes / rb) {
            std::memset(y, 0, static_cast<size_t>(rows) * sizeof(float));
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
            if (!act.use_f32) {
                const void* wdata = weight.data;
                #pragma omp parallel for schedule(static)
                for (int r = 0; r < rows; ++r) {
                    y[r] = q8_fn(wdata, row_offset + static_cast<size_t>(r), act, cols);
                }
                return;
            }
        }
    }

    gemv_slice_f32(y, weight, row_offset, x, rows, cols);
}

// ============================================================================
// High-Throughput Batched Multi-Token Register-Tiled AVX2 GEMM Kernels (1x8 & 1x4)
// Unpacks each quantized weight block ONCE per 8 (or 4) tokens and tiles rows
// in 64-row L2-resident blocks so weights stream from DDR4 RAM literally ONCE
// across an entire S-token prompt prefill chunk (~217-247 GMAC/s on 6-core Xeon).
// ============================================================================

// Pack n_tok (<= NTok) FP32 activation rows into L1-friendly contiguous [cols/8][NTok][8] layout.
// Eliminates 4KB L1D cache-set aliasing and frees 7 GPR pointer registers inside 1x4/1x8 FMA microkernels!
template <int NTok>
static inline void pack_f32_tile_ptr(const float* X, int stride_x, int cols, int n_tok, float* out) {
    const int n8 = cols >> 3;
    const __m256 vzero = _mm256_setzero_ps();
    for (int c8 = 0; c8 < n8; ++c8) {
        const int col = c8 << 3;
        float* base = out + static_cast<size_t>(c8) * static_cast<size_t>(NTok * 8);
        for (int t = 0; t < n_tok; ++t) {
            _mm256_storeu_ps(base + t * 8, _mm256_loadu_ps(X + static_cast<size_t>(t) * static_cast<size_t>(stride_x) + col));
        }
        for (int t = n_tok; t < NTok; ++t) {
            _mm256_storeu_ps(base + t * 8, vzero);
        }
    }
}

template <int NTok>
static inline void pack_f32_tile(const float* X, int stride_x, int cols, int n_tok, std::vector<float>& dst) {
    dst.resize(static_cast<size_t>(cols) * static_cast<size_t>(NTok));
    pack_f32_tile_ptr<NTok>(X, stride_x, cols, n_tok, dst.data());
}

void Avx2Math::pack_q8k_batch(const float* X, int stride_x, int cols, int n_tokens, Q8KBatch& out, bool extract_outliers) {
    out.nb256 = (cols >= 256) ? (cols / 256) : 0;
    out.n_tokens = (n_tokens > 0 && out.nb256 > 0) ? n_tokens : 0;
    out.tiles.clear();
    if (!X || out.n_tokens == 0) return;

    const int n_tiles = (n_tokens + 7) / 8;
    const int nb32 = out.nb256 * 8;
    out.tiles.resize(static_cast<size_t>(n_tiles));

    Q8Act32 tmp;
    for (int g = 0; g < n_tiles; ++g) {
        auto& tile = out.tiles[static_cast<size_t>(g)];
        const int t_base = g * 8;
        const int n_tok = std::min(8, n_tokens - t_base);
        tile.n_tok = n_tok;
        tile.use_f32 = false;
        tile.has_outliers = false;
        tile.x_f32.clear();
        tile.qs.assign(static_cast<size_t>(nb32) * 8ULL * 32ULL, 0);
        tile.d.assign(static_cast<size_t>(nb32) * 8ULL, 0.0f);
        tile.sumf.assign(static_cast<size_t>(nb32) * 8ULL, 0.0f);
        tile.bsum16.assign(static_cast<size_t>(nb32) * 8ULL * 2ULL, 0);

        bool promote_f32 = false;
        for (int t = 0; t < n_tok; ++t) {
            quantize_q8_act(X + static_cast<size_t>(t_base + t) * static_cast<size_t>(stride_x), cols, tmp, extract_outliers);
            if (extract_outliers && tmp.use_f32) {
                promote_f32 = true;
                break;
            }
            tile.n_outliers[t] = tmp.n_outliers;
            if (tmp.n_outliers > 0) {
                tile.has_outliers = true;
                std::memcpy(tile.outliers[t], tmp.outliers, static_cast<size_t>(tmp.n_outliers) * sizeof(Q8Outlier));
            }
            for (int sub = 0; sub < nb32; ++sub) {
                const size_t idx = static_cast<size_t>(sub) * 8ULL + static_cast<size_t>(t);
                std::memcpy(tile.qs.data() + idx * 32ULL, tmp.qs + static_cast<size_t>(sub) * 32ULL, 32);
                tile.d[idx]    = tmp.d[sub];
                tile.sumf[idx] = tmp.sumf[sub];
                tile.bsum16[idx * 2ULL + 0] = tmp.bsum[2 * sub + 0];
                tile.bsum16[idx * 2ULL + 1] = tmp.bsum[2 * sub + 1];
            }
        }
        if (promote_f32) {
            // Heavily spiked tile: skip Q8+scalar outliers & route straight to L1-packed 1x4/1x8 Exact FP32!
            tile.use_f32 = true;
            tile.has_outliers = false;
            tile.qs.clear();
            tile.d.clear();
            tile.sumf.clear();
            tile.bsum16.clear();
            for (int t = 0; t < 8; ++t) tile.n_outliers[t] = 0;
            const float* x_tile = X + static_cast<size_t>(t_base) * static_cast<size_t>(stride_x);
            if (n_tok <= 4) pack_f32_tile<4>(x_tile, stride_x, cols, n_tok, tile.x_f32);
            else            pack_f32_tile<8>(x_tile, stride_x, cols, n_tok, tile.x_f32);
        } else {
            for (int t = n_tok; t < 8; ++t) tile.n_outliers[t] = 0;
        }
    }
}

void Avx2Math::pack_q8_32_batch(const float* X, int stride_x, int cols, int n_tokens, Q8_32Batch& out, bool extract_outliers) {
    out.nb32 = (cols >= 32) ? (cols / 32) : 0;
    out.n_tokens = (n_tokens > 0 && out.nb32 > 0) ? n_tokens : 0;
    out.tiles.clear();
    if (!X || out.n_tokens == 0) return;

    const int n_tiles = (n_tokens + 7) / 8;
    out.tiles.resize(static_cast<size_t>(n_tiles));

    Q8Act32 tmp;
    for (int g = 0; g < n_tiles; ++g) {
        auto& tile = out.tiles[static_cast<size_t>(g)];
        const int t_base = g * 8;
        const int n_tok = std::min(8, n_tokens - t_base);
        tile.n_tok = n_tok;
        tile.use_f32 = false;
        tile.has_outliers = false;
        tile.x_f32.clear();
        tile.qs.assign(static_cast<size_t>(out.nb32) * 8ULL * 32ULL, 0);
        tile.d.assign(static_cast<size_t>(out.nb32) * 8ULL, 0.0f);
        tile.sumf.assign(static_cast<size_t>(out.nb32) * 8ULL, 0.0f);
        tile.bsum32.assign(static_cast<size_t>(out.nb32) * 8ULL, 0);

        bool promote_f32 = false;
        for (int t = 0; t < n_tok; ++t) {
            quantize_q8_act(X + static_cast<size_t>(t_base + t) * static_cast<size_t>(stride_x), cols, tmp, extract_outliers);
            if (extract_outliers && tmp.use_f32) {
                promote_f32 = true;
                break;
            }
            tile.n_outliers[t] = tmp.n_outliers;
            if (tmp.n_outliers > 0) {
                tile.has_outliers = true;
                std::memcpy(tile.outliers[t], tmp.outliers, static_cast<size_t>(tmp.n_outliers) * sizeof(Q8Outlier));
            }
            for (int b = 0; b < out.nb32; ++b) {
                std::memcpy(tile.qs.data() + (static_cast<size_t>(b) * 8ULL + static_cast<size_t>(t)) * 32ULL,
                            tmp.qs + static_cast<size_t>(b) * 32ULL, 32);
                tile.d[static_cast<size_t>(b) * 8ULL + static_cast<size_t>(t)]      = tmp.d[b];
                tile.sumf[static_cast<size_t>(b) * 8ULL + static_cast<size_t>(t)]   = tmp.sumf[b];
                tile.bsum32[static_cast<size_t>(b) * 8ULL + static_cast<size_t>(t)] =
                    static_cast<int16_t>(static_cast<int32_t>(tmp.bsum[2 * b]) + static_cast<int32_t>(tmp.bsum[2 * b + 1]));
            }
        }
        if (promote_f32) {
            tile.use_f32 = true;
            tile.has_outliers = false;
            tile.qs.clear();
            tile.d.clear();
            tile.sumf.clear();
            tile.bsum32.clear();
            for (int t = 0; t < 8; ++t) tile.n_outliers[t] = 0;
            const float* x_tile = X + static_cast<size_t>(t_base) * static_cast<size_t>(stride_x);
            if (n_tok <= 4) pack_f32_tile<4>(x_tile, stride_x, cols, n_tok, tile.x_f32);
            else            pack_f32_tile<8>(x_tile, stride_x, cols, n_tok, tile.x_f32);
        } else {
            for (int t = n_tok; t < 8; ++t) tile.n_outliers[t] = 0;
        }
    }
}

bool Avx2Math::supports_q8k_gemm(QuantType type, int cols) {
    if (cols < 256 || (cols % 256) != 0) return false;
    return type == QuantType::IQ4_XS || type == QuantType::Q4_K ||
           type == QuantType::Q5_K   || type == QuantType::Q6_K;
}

bool Avx2Math::supports_q8_32_gemm(QuantType type, int cols) {
    if (cols < 32 || (cols % 32) != 0) return false;
    return type == QuantType::IQ4_NL || type == QuantType::Q5_1 ||
           type == QuantType::Q5_0   || type == QuantType::Q8_0 ||
           type == QuantType::Q4_0;
}

template <int NTok>
static inline void dot_iq4_xs_q8k_xN(const void* data, size_t row, const Q8KBatch::Tile& tile, int nb256, float* out) {
    const auto* vx = reinterpret_cast<const block_iq4_xs*>(data) + row * static_cast<size_t>(nb256);
    const __m256i lut256  = _mm256_broadcastsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(kvalues_iq4nl)));
    const __m128i mask_lo = _mm_set1_epi8(0x0F);
    const __m256i ones16  = _mm256_set1_epi16(1);

    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps(), acc3 = _mm256_setzero_ps();
    __m256 acc4 = _mm256_setzero_ps(), acc5 = _mm256_setzero_ps();
    __m256 acc6 = _mm256_setzero_ps(), acc7 = _mm256_setzero_ps();

    for (int i = 0; i < nb256; ++i) {
        const float d_w = Avx2Math::fp16_to_fp32(vx[i].d);
        const uint16_t scales_h = vx[i].scales_h;
        const uint8_t* scales_l = vx[i].scales_l;
        const uint8_t* qs = vx[i].qs;

        for (int ib = 0; ib < 8; ++ib) {
            const __m128i qb  = _mm_loadu_si128(reinterpret_cast<const __m128i*>(qs));
            const __m256i idx = mm256_concat_si128(_mm_and_si128(qb, mask_lo), _mm_and_si128(_mm_srli_epi16(qb, 4), mask_lo));
            const __m256i w   = _mm256_shuffle_epi8(lut256, idx);
            const __m256i aw  = _mm256_abs_epi8(w);

            const int ls = (((scales_l[ib / 2] >> (4 * (ib % 2))) & 0x0F) | (((scales_h >> (2 * ib)) & 3) << 4)) - 32;
            const float dw_ls = d_w * static_cast<float>(ls);

            const size_t sub = static_cast<size_t>(i * 8 + ib);
            const int8_t* yp = tile.qs.data() + sub * 256ULL;
            const float*  dp = tile.d.data()  + sub * 8ULL;
            #define IQXS_MAC(T, ACC) do { \
                const __m256i y = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(yp + (T) * 32)); \
                const __m256i p = _mm256_madd_epi16(_mm256_maddubs_epi16(aw, _mm256_sign_epi8(y, w)), ones16); \
                ACC = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p), _mm256_set1_ps(dw_ls * dp[T]), ACC); \
            } while (0)
            IQXS_MAC(0, acc0); IQXS_MAC(1, acc1); IQXS_MAC(2, acc2); IQXS_MAC(3, acc3);
            if constexpr (NTok > 4) {
                IQXS_MAC(4, acc4); IQXS_MAC(5, acc5); IQXS_MAC(6, acc6); IQXS_MAC(7, acc7);
            }
            #undef IQXS_MAC
            qs += 16;
        }
    }
    out[0] = hsum256_ps(acc0); out[1] = hsum256_ps(acc1);
    out[2] = hsum256_ps(acc2); out[3] = hsum256_ps(acc3);
    if constexpr (NTok > 4) {
        out[4] = hsum256_ps(acc4); out[5] = hsum256_ps(acc5);
        out[6] = hsum256_ps(acc6); out[7] = hsum256_ps(acc7);
    }
}

template <int NTok>
static inline void dot_q4_K_q8k_xN(const void* data, size_t row, const Q8KBatch::Tile& tile, int nb256, float* out) {
    const auto* vx = reinterpret_cast<const block_q4_K*>(data) + row * static_cast<size_t>(nb256);
    const __m256i mask_lo4 = _mm256_set1_epi8(0x0F);
    const __m256i ones16   = _mm256_set1_epi16(1);

    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps(), acc3 = _mm256_setzero_ps();
    __m256 acc4 = _mm256_setzero_ps(), acc5 = _mm256_setzero_ps();
    __m256 acc6 = _mm256_setzero_ps(), acc7 = _mm256_setzero_ps();
    __m256 min_v = _mm256_setzero_ps();

    for (int i = 0; i < nb256; ++i) {
        const float d   = Avx2Math::fp16_to_fp32(vx[i].d);
        const float min = Avx2Math::fp16_to_fp32(vx[i].dmin);
        const uint8_t* q = vx[i].qs;
        uint8_t sc1, mn1, sc2, mn2;

        for (int j = 0; j < 4; ++j) {
            const size_t yb = static_cast<size_t>(i * 8 + 2 * j);
            get_scale_min_k4(2 * j,     vx[i].scales, &sc1, &mn1);
            get_scale_min_k4(2 * j + 1, vx[i].scales, &sc2, &mn2);
            const __m256i qb  = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q));
            const __m256i qlo = _mm256_and_si256(qb, mask_lo4);
            const __m256i qhi = _mm256_and_si256(_mm256_srli_epi16(qb, 4), mask_lo4);

            const float d1 = d * static_cast<float>(sc1);
            const int8_t* y1p = tile.qs.data() + yb * 256ULL;
            const float*  d1p = tile.d.data()  + yb * 8ULL;
            #define Q4K_MAC1(T, ACC) do { \
                const __m256i p = _mm256_madd_epi16(_mm256_maddubs_epi16(qlo, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(y1p + (T) * 32))), ones16); \
                ACC = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p), _mm256_set1_ps(d1 * d1p[T]), ACC); \
            } while (0)
            Q4K_MAC1(0, acc0); Q4K_MAC1(1, acc1); Q4K_MAC1(2, acc2); Q4K_MAC1(3, acc3);
            if constexpr (NTok > 4) {
                Q4K_MAC1(4, acc4); Q4K_MAC1(5, acc5); Q4K_MAC1(6, acc6); Q4K_MAC1(7, acc7);
            }
            #undef Q4K_MAC1

            const float d2 = d * static_cast<float>(sc2);
            const int8_t* y2p = tile.qs.data() + (yb + 1) * 256ULL;
            const float*  d2p = tile.d.data()  + (yb + 1) * 8ULL;
            #define Q4K_MAC2(T, ACC) do { \
                const __m256i p = _mm256_madd_epi16(_mm256_maddubs_epi16(qhi, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(y2p + (T) * 32))), ones16); \
                ACC = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p), _mm256_set1_ps(d2 * d2p[T]), ACC); \
            } while (0)
            Q4K_MAC2(0, acc0); Q4K_MAC2(1, acc1); Q4K_MAC2(2, acc2); Q4K_MAC2(3, acc3);
            if constexpr (NTok > 4) {
                Q4K_MAC2(4, acc4); Q4K_MAC2(5, acc5); Q4K_MAC2(6, acc6); Q4K_MAC2(7, acc7);
            }
            #undef Q4K_MAC2

            min_v = _mm256_fmadd_ps(_mm256_set1_ps(min * static_cast<float>(mn1)), _mm256_loadu_ps(tile.sumf.data() + yb * 8ULL), min_v);
            min_v = _mm256_fmadd_ps(_mm256_set1_ps(min * static_cast<float>(mn2)), _mm256_loadu_ps(tile.sumf.data() + (yb + 1) * 8ULL), min_v);
            q += 32;
        }
    }
    alignas(32) float mins[8];
    _mm256_store_ps(mins, min_v);
    out[0] = hsum256_ps(acc0) - mins[0]; out[1] = hsum256_ps(acc1) - mins[1];
    out[2] = hsum256_ps(acc2) - mins[2]; out[3] = hsum256_ps(acc3) - mins[3];
    if constexpr (NTok > 4) {
        out[4] = hsum256_ps(acc4) - mins[4]; out[5] = hsum256_ps(acc5) - mins[5];
        out[6] = hsum256_ps(acc6) - mins[6]; out[7] = hsum256_ps(acc7) - mins[7];
    }
}

template <int NTok>
static inline void dot_q5_K_q8k_xN(const void* data, size_t row, const Q8KBatch::Tile& tile, int nb256, float* out) {
    const auto* vx = reinterpret_cast<const block_q5_K*>(data) + row * static_cast<size_t>(nb256);
    const __m256i mask_lo4 = _mm256_set1_epi8(0x0F);
    const __m256i val_16   = _mm256_set1_epi8(0x10);
    const __m256i ones16   = _mm256_set1_epi16(1);

    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps(), acc3 = _mm256_setzero_ps();
    __m256 acc4 = _mm256_setzero_ps(), acc5 = _mm256_setzero_ps();
    __m256 acc6 = _mm256_setzero_ps(), acc7 = _mm256_setzero_ps();
    __m256 min_v = _mm256_setzero_ps();

    for (int i = 0; i < nb256; ++i) {
        const float d   = Avx2Math::fp16_to_fp32(vx[i].d);
        const float min = Avx2Math::fp16_to_fp32(vx[i].dmin);
        const uint8_t* ql = vx[i].qs;
        const __m256i qhb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(vx[i].qh));
        uint8_t sc1, mn1, sc2, mn2;
        uint8_t u1 = 1, u2 = 2;

        for (int j = 0; j < 4; ++j) {
            const size_t yb = static_cast<size_t>(i * 8 + 2 * j);
            get_scale_min_k4(2 * j,     vx[i].scales, &sc1, &mn1);
            get_scale_min_k4(2 * j + 1, vx[i].scales, &sc2, &mn2);
            const __m256i qb  = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ql));
            const __m256i vu1 = _mm256_set1_epi8(static_cast<char>(u1));
            const __m256i h1  = _mm256_and_si256(_mm256_cmpeq_epi8(_mm256_and_si256(qhb, vu1), vu1), val_16);
            const __m256i q1  = _mm256_or_si256(_mm256_and_si256(qb, mask_lo4), h1);
            const float d1 = d * static_cast<float>(sc1);
            const int8_t* y1p = tile.qs.data() + yb * 256ULL;
            const float*  d1p = tile.d.data()  + yb * 8ULL;

            #define Q5K_MAC1(T, ACC) do { \
                const __m256i p = _mm256_madd_epi16(_mm256_maddubs_epi16(q1, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(y1p + (T) * 32))), ones16); \
                ACC = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p), _mm256_set1_ps(d1 * d1p[T]), ACC); \
            } while (0)
            Q5K_MAC1(0, acc0); Q5K_MAC1(1, acc1); Q5K_MAC1(2, acc2); Q5K_MAC1(3, acc3);
            if constexpr (NTok > 4) {
                Q5K_MAC1(4, acc4); Q5K_MAC1(5, acc5); Q5K_MAC1(6, acc6); Q5K_MAC1(7, acc7);
            }
            #undef Q5K_MAC1

            const __m256i vu2 = _mm256_set1_epi8(static_cast<char>(u2));
            const __m256i h2  = _mm256_and_si256(_mm256_cmpeq_epi8(_mm256_and_si256(qhb, vu2), vu2), val_16);
            const __m256i q2  = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(qb, 4), mask_lo4), h2);
            const float d2 = d * static_cast<float>(sc2);
            const int8_t* y2p = tile.qs.data() + (yb + 1) * 256ULL;
            const float*  d2p = tile.d.data()  + (yb + 1) * 8ULL;

            #define Q5K_MAC2(T, ACC) do { \
                const __m256i p = _mm256_madd_epi16(_mm256_maddubs_epi16(q2, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(y2p + (T) * 32))), ones16); \
                ACC = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p), _mm256_set1_ps(d2 * d2p[T]), ACC); \
            } while (0)
            Q5K_MAC2(0, acc0); Q5K_MAC2(1, acc1); Q5K_MAC2(2, acc2); Q5K_MAC2(3, acc3);
            if constexpr (NTok > 4) {
                Q5K_MAC2(4, acc4); Q5K_MAC2(5, acc5); Q5K_MAC2(6, acc6); Q5K_MAC2(7, acc7);
            }
            #undef Q5K_MAC2

            min_v = _mm256_fmadd_ps(_mm256_set1_ps(min * static_cast<float>(mn1)), _mm256_loadu_ps(tile.sumf.data() + yb * 8ULL), min_v);
            min_v = _mm256_fmadd_ps(_mm256_set1_ps(min * static_cast<float>(mn2)), _mm256_loadu_ps(tile.sumf.data() + (yb + 1) * 8ULL), min_v);
            ql += 32;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
    alignas(32) float mins[8];
    _mm256_store_ps(mins, min_v);
    out[0] = hsum256_ps(acc0) - mins[0]; out[1] = hsum256_ps(acc1) - mins[1];
    out[2] = hsum256_ps(acc2) - mins[2]; out[3] = hsum256_ps(acc3) - mins[3];
    if constexpr (NTok > 4) {
        out[4] = hsum256_ps(acc4) - mins[4]; out[5] = hsum256_ps(acc5) - mins[5];
        out[6] = hsum256_ps(acc6) - mins[6]; out[7] = hsum256_ps(acc7) - mins[7];
    }
}

template <int NTok>
static inline void dot_q6_K_q8k_xN(const void* data, size_t row, const Q8KBatch::Tile& tile, int nb256, float* out) {
    const auto* vx = reinterpret_cast<const block_q6_K*>(data) + row * static_cast<size_t>(nb256);
    const __m256i mask_lo4 = _mm256_set1_epi8(0x0F);
    const __m256i mask_hi2 = _mm256_set1_epi8(0x30);

    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps(), acc3 = _mm256_setzero_ps();
    __m256 acc4 = _mm256_setzero_ps(), acc5 = _mm256_setzero_ps();
    __m256 acc6 = _mm256_setzero_ps(), acc7 = _mm256_setzero_ps();
    float bias_tot[8] = {0, 0, 0, 0, 0, 0, 0, 0};

    for (int i = 0; i < nb256; ++i) {
        const float d_w = Avx2Math::fp16_to_fp32(vx[i].d);
        const uint8_t* ql = vx[i].ql;
        const uint8_t* qh = vx[i].qh;
        const int8_t* sc = vx[i].scales;

        for (int n_chunk = 0; n_chunk < 2; ++n_chunk) {
            const size_t yb = static_cast<size_t>(i * 8 + n_chunk * 4);
            const __m256i ql0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ql));
            const __m256i ql1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ql + 32));
            const __m256i qhb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qh));

            #define Q6K_SUB(QV, SCV, SC0, SC1, SUB_IDX) do { \
                const int8_t*  yp  = tile.qs.data()     + (SUB_IDX) * 256ULL; \
                const float*   dp  = tile.d.data()      + (SUB_IDX) * 8ULL; \
                const int16_t* bsp = tile.bsum16.data() + (SUB_IDX) * 16ULL; \
                auto step_tok = [&](int T, __m256& ACC) { \
                    const float dw_dy = d_w * dp[T]; \
                    const __m256i p = _mm256_madd_epi16(_mm256_maddubs_epi16(QV, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(yp + T * 32))), SCV); \
                    ACC = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p), _mm256_set1_ps(dw_dy), ACC); \
                    bias_tot[T] += dw_dy * static_cast<float>(static_cast<int32_t>(SC0) * bsp[T * 2 + 0] + static_cast<int32_t>(SC1) * bsp[T * 2 + 1]); \
                }; \
                step_tok(0, acc0); step_tok(1, acc1); step_tok(2, acc2); step_tok(3, acc3); \
                if constexpr (NTok > 4) { \
                    step_tok(4, acc4); step_tok(5, acc5); step_tok(6, acc6); step_tok(7, acc7); \
                } \
            } while (0)

            {
                const __m256i q1   = _mm256_or_si256(_mm256_and_si256(ql0, mask_lo4), _mm256_and_si256(_mm256_slli_epi16(qhb, 4), mask_hi2));
                const __m256i sc_1 = mm256_concat_si128(_mm_set1_epi16(sc[0]), _mm_set1_epi16(sc[1]));
                Q6K_SUB(q1, sc_1, sc[0], sc[1], yb + 0);
            }
            {
                const __m256i q2   = _mm256_or_si256(_mm256_and_si256(ql1, mask_lo4), _mm256_and_si256(_mm256_slli_epi16(qhb, 2), mask_hi2));
                const __m256i sc_2 = mm256_concat_si128(_mm_set1_epi16(sc[2]), _mm_set1_epi16(sc[3]));
                Q6K_SUB(q2, sc_2, sc[2], sc[3], yb + 1);
            }
            {
                const __m256i q3   = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(ql0, 4), mask_lo4), _mm256_and_si256(qhb, mask_hi2));
                const __m256i sc_3 = mm256_concat_si128(_mm_set1_epi16(sc[4]), _mm_set1_epi16(sc[5]));
                Q6K_SUB(q3, sc_3, sc[4], sc[5], yb + 2);
            }
            {
                const __m256i q4   = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(ql1, 4), mask_lo4), _mm256_and_si256(_mm256_srli_epi16(qhb, 2), mask_hi2));
                const __m256i sc_4 = mm256_concat_si128(_mm_set1_epi16(sc[6]), _mm_set1_epi16(sc[7]));
                Q6K_SUB(q4, sc_4, sc[6], sc[7], yb + 3);
            }
            #undef Q6K_SUB

            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
    out[0] = hsum256_ps(acc0) - 32.0f * bias_tot[0]; out[1] = hsum256_ps(acc1) - 32.0f * bias_tot[1];
    out[2] = hsum256_ps(acc2) - 32.0f * bias_tot[2]; out[3] = hsum256_ps(acc3) - 32.0f * bias_tot[3];
    if constexpr (NTok > 4) {
        out[4] = hsum256_ps(acc4) - 32.0f * bias_tot[4]; out[5] = hsum256_ps(acc5) - 32.0f * bias_tot[5];
        out[6] = hsum256_ps(acc6) - 32.0f * bias_tot[6]; out[7] = hsum256_ps(acc7) - 32.0f * bias_tot[7];
    }
}

template <int NTok>
static inline void dot_iq4_nl_q8_32_xN(const void* data, size_t row, const Q8_32Batch::Tile& tile, int nb32, float* out) {
    const auto* vx = reinterpret_cast<const block_iq4_nl*>(data) + row * static_cast<size_t>(nb32);
    const __m256i lut256  = _mm256_broadcastsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(kvalues_iq4nl)));
    const __m128i mask_lo = _mm_set1_epi8(0x0F);
    const __m256i ones16  = _mm256_set1_epi16(1);

    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps(), acc3 = _mm256_setzero_ps();
    __m256 acc4 = _mm256_setzero_ps(), acc5 = _mm256_setzero_ps();
    __m256 acc6 = _mm256_setzero_ps(), acc7 = _mm256_setzero_ps();

    for (int i = 0; i < nb32; ++i) {
        const float d_w = Avx2Math::fp16_to_fp32(vx[i].d);
        const __m128i qb  = _mm_loadu_si128(reinterpret_cast<const __m128i*>(vx[i].qs));
        const __m256i idx = mm256_concat_si128(_mm_and_si128(qb, mask_lo), _mm_and_si128(_mm_srli_epi16(qb, 4), mask_lo));
        const __m256i w   = _mm256_shuffle_epi8(lut256, idx);
        const __m256i aw  = _mm256_abs_epi8(w);

        const int8_t* yp = tile.qs.data() + static_cast<size_t>(i) * 256ULL;
        const float*  dp = tile.d.data()  + static_cast<size_t>(i) * 8ULL;

        #define IQNL_MAC(T, ACC) do { \
            const __m256i y = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(yp + (T) * 32)); \
            const __m256i p = _mm256_madd_epi16(_mm256_maddubs_epi16(aw, _mm256_sign_epi8(y, w)), ones16); \
            ACC = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p), _mm256_set1_ps(d_w * dp[T]), ACC); \
        } while (0)
        IQNL_MAC(0, acc0); IQNL_MAC(1, acc1); IQNL_MAC(2, acc2); IQNL_MAC(3, acc3);
        if constexpr (NTok > 4) {
            IQNL_MAC(4, acc4); IQNL_MAC(5, acc5); IQNL_MAC(6, acc6); IQNL_MAC(7, acc7);
        }
        #undef IQNL_MAC
    }
    out[0] = hsum256_ps(acc0); out[1] = hsum256_ps(acc1);
    out[2] = hsum256_ps(acc2); out[3] = hsum256_ps(acc3);
    if constexpr (NTok > 4) {
        out[4] = hsum256_ps(acc4); out[5] = hsum256_ps(acc5);
        out[6] = hsum256_ps(acc6); out[7] = hsum256_ps(acc7);
    }
}

template <int NTok>
static inline void dot_q5_1_q8_32_xN(const void* data, size_t row, const Q8_32Batch::Tile& tile, int nb32, float* out) {
    const auto* vx = reinterpret_cast<const block_q5_1*>(data) + row * static_cast<size_t>(nb32);
    const __m128i mask_lo  = _mm_set1_epi8(0x0F);
    const __m256i ones16   = _mm256_set1_epi16(1);
    const __m256i shuf_qh  = _mm256_setr_epi8(
        0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1,
        2,2,2,2,2,2,2,2, 3,3,3,3,3,3,3,3);
    const __m256i bit_mask = _mm256_set1_epi64x(static_cast<long long>(0x8040201008040201ULL));
    const __m256i val_16   = _mm256_set1_epi8(0x10);

    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps(), acc3 = _mm256_setzero_ps();
    __m256 acc4 = _mm256_setzero_ps(), acc5 = _mm256_setzero_ps();
    __m256 acc6 = _mm256_setzero_ps(), acc7 = _mm256_setzero_ps();
    __m256 m_v  = _mm256_setzero_ps();

    for (int i = 0; i < nb32; ++i) {
        const float d_w = Avx2Math::fp16_to_fp32(vx[i].d);
        const float m_w = Avx2Math::fp16_to_fp32(vx[i].m);
        uint32_t qh_u32 = 0;
        std::memcpy(&qh_u32, vx[i].qh, sizeof(qh_u32));
        const __m256i qh_b   = _mm256_shuffle_epi8(_mm256_set1_epi32(static_cast<int>(qh_u32)), shuf_qh);
        const __m256i qh_bit = _mm256_and_si256(_mm256_cmpeq_epi8(_mm256_and_si256(qh_b, bit_mask), bit_mask), val_16);

        const __m128i qb = _mm_loadu_si128(reinterpret_cast<const __m128i*>(vx[i].qs));
        const __m256i q5 = _mm256_or_si256(
            mm256_concat_si128(_mm_and_si128(qb, mask_lo), _mm_and_si128(_mm_srli_epi16(qb, 4), mask_lo)), qh_bit);

        const int8_t* yp = tile.qs.data() + static_cast<size_t>(i) * 256ULL;
        const float*  dp = tile.d.data()  + static_cast<size_t>(i) * 8ULL;

        #define Q51_MAC(T, ACC) do { \
            const __m256i p = _mm256_madd_epi16(_mm256_maddubs_epi16(q5, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(yp + (T) * 32))), ones16); \
            ACC = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p), _mm256_set1_ps(d_w * dp[T]), ACC); \
        } while (0)
        Q51_MAC(0, acc0); Q51_MAC(1, acc1); Q51_MAC(2, acc2); Q51_MAC(3, acc3);
        if constexpr (NTok > 4) {
            Q51_MAC(4, acc4); Q51_MAC(5, acc5); Q51_MAC(6, acc6); Q51_MAC(7, acc7);
        }
        #undef Q51_MAC
        m_v = _mm256_fmadd_ps(_mm256_set1_ps(m_w), _mm256_loadu_ps(tile.sumf.data() + static_cast<size_t>(i) * 8ULL), m_v);
    }
    alignas(32) float ms[8];
    _mm256_store_ps(ms, m_v);
    out[0] = hsum256_ps(acc0) + ms[0]; out[1] = hsum256_ps(acc1) + ms[1];
    out[2] = hsum256_ps(acc2) + ms[2]; out[3] = hsum256_ps(acc3) + ms[3];
    if constexpr (NTok > 4) {
        out[4] = hsum256_ps(acc4) + ms[4]; out[5] = hsum256_ps(acc5) + ms[5];
        out[6] = hsum256_ps(acc6) + ms[6]; out[7] = hsum256_ps(acc7) + ms[7];
    }
}

template <int NTok>
static inline void dot_q5_0_q8_32_xN(const void* data, size_t row, const Q8_32Batch::Tile& tile, int nb32, float* out) {
    const auto* vx = reinterpret_cast<const block_q5_0*>(data) + row * static_cast<size_t>(nb32);
    const __m128i mask_lo  = _mm_set1_epi8(0x0F);
    const __m256i ones16   = _mm256_set1_epi16(1);
    const __m256i shuf_qh  = _mm256_setr_epi8(
        0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1,
        2,2,2,2,2,2,2,2, 3,3,3,3,3,3,3,3);
    const __m256i bit_mask = _mm256_set1_epi64x(static_cast<long long>(0x8040201008040201ULL));
    const __m256i val_16   = _mm256_set1_epi8(0x10);

    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps(), acc3 = _mm256_setzero_ps();
    __m256 acc4 = _mm256_setzero_ps(), acc5 = _mm256_setzero_ps();
    __m256 acc6 = _mm256_setzero_ps(), acc7 = _mm256_setzero_ps();
    float bsum[8] = {0,0,0,0,0,0,0,0};

    for (int i = 0; i < nb32; ++i) {
        const float d_w = Avx2Math::fp16_to_fp32(vx[i].d);
        uint32_t qh_u32 = 0;
        std::memcpy(&qh_u32, vx[i].qh, sizeof(qh_u32));
        const __m256i qh_b   = _mm256_shuffle_epi8(_mm256_set1_epi32(static_cast<int>(qh_u32)), shuf_qh);
        const __m256i qh_bit = _mm256_and_si256(_mm256_cmpeq_epi8(_mm256_and_si256(qh_b, bit_mask), bit_mask), val_16);
        const __m128i qb = _mm_loadu_si128(reinterpret_cast<const __m128i*>(vx[i].qs));
        const __m256i q5 = _mm256_or_si256(
            mm256_concat_si128(_mm_and_si128(qb, mask_lo), _mm_and_si128(_mm_srli_epi16(qb, 4), mask_lo)), qh_bit);

        const int8_t*  yp = tile.qs.data()     + static_cast<size_t>(i) * 256ULL;
        const float*   dp = tile.d.data()      + static_cast<size_t>(i) * 8ULL;
        const int16_t* bp = tile.bsum32.data() + static_cast<size_t>(i) * 8ULL;

        #define Q50_MAC(T, ACC) do { \
            const float dw_dy = d_w * dp[T]; \
            const __m256i p = _mm256_madd_epi16(_mm256_maddubs_epi16(q5, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(yp + (T) * 32))), ones16); \
            ACC = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p), _mm256_set1_ps(dw_dy), ACC); \
            bsum[T] += dw_dy * static_cast<float>(bp[T]); \
        } while (0)
        Q50_MAC(0, acc0); Q50_MAC(1, acc1); Q50_MAC(2, acc2); Q50_MAC(3, acc3);
        if constexpr (NTok > 4) {
            Q50_MAC(4, acc4); Q50_MAC(5, acc5); Q50_MAC(6, acc6); Q50_MAC(7, acc7);
        }
        #undef Q50_MAC
    }
    out[0] = hsum256_ps(acc0) - 16.0f * bsum[0]; out[1] = hsum256_ps(acc1) - 16.0f * bsum[1];
    out[2] = hsum256_ps(acc2) - 16.0f * bsum[2]; out[3] = hsum256_ps(acc3) - 16.0f * bsum[3];
    if constexpr (NTok > 4) {
        out[4] = hsum256_ps(acc4) - 16.0f * bsum[4]; out[5] = hsum256_ps(acc5) - 16.0f * bsum[5];
        out[6] = hsum256_ps(acc6) - 16.0f * bsum[6]; out[7] = hsum256_ps(acc7) - 16.0f * bsum[7];
    }
}

template <int NTok>
static inline void dot_q8_0_q8_32_xN(const void* data, size_t row, const Q8_32Batch::Tile& tile, int nb32, float* out) {
    const auto* vx = reinterpret_cast<const block_q8_0*>(data) + row * static_cast<size_t>(nb32);
    const __m256i ones16 = _mm256_set1_epi16(1);

    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps(), acc3 = _mm256_setzero_ps();
    __m256 acc4 = _mm256_setzero_ps(), acc5 = _mm256_setzero_ps();
    __m256 acc6 = _mm256_setzero_ps(), acc7 = _mm256_setzero_ps();

    for (int i = 0; i < nb32; ++i) {
        const float d_w = Avx2Math::fp16_to_fp32(vx[i].d);
        const __m256i w  = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(vx[i].qs));
        const __m256i aw = _mm256_abs_epi8(w);

        const int8_t* yp = tile.qs.data() + static_cast<size_t>(i) * 256ULL;
        const float*  dp = tile.d.data()  + static_cast<size_t>(i) * 8ULL;

        #define Q80_MAC(T, ACC) do { \
            const __m256i y = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(yp + (T) * 32)); \
            const __m256i p = _mm256_madd_epi16(_mm256_maddubs_epi16(aw, _mm256_sign_epi8(y, w)), ones16); \
            ACC = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p), _mm256_set1_ps(d_w * dp[T]), ACC); \
        } while (0)
        Q80_MAC(0, acc0); Q80_MAC(1, acc1); Q80_MAC(2, acc2); Q80_MAC(3, acc3);
        if constexpr (NTok > 4) {
            Q80_MAC(4, acc4); Q80_MAC(5, acc5); Q80_MAC(6, acc6); Q80_MAC(7, acc7);
        }
        #undef Q80_MAC
    }
    out[0] = hsum256_ps(acc0); out[1] = hsum256_ps(acc1);
    out[2] = hsum256_ps(acc2); out[3] = hsum256_ps(acc3);
    if constexpr (NTok > 4) {
        out[4] = hsum256_ps(acc4); out[5] = hsum256_ps(acc5);
        out[6] = hsum256_ps(acc6); out[7] = hsum256_ps(acc7);
    }
}

template <int NTok>
static inline void dot_q4_0_q8_32_xN(const void* data, size_t row, const Q8_32Batch::Tile& tile, int nb32, float* out) {
    const auto* vx = reinterpret_cast<const block_q4_0*>(data) + row * static_cast<size_t>(nb32);
    const __m128i mask_lo = _mm_set1_epi8(0x0F);
    const __m256i ones16  = _mm256_set1_epi16(1);

    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps(), acc3 = _mm256_setzero_ps();
    __m256 acc4 = _mm256_setzero_ps(), acc5 = _mm256_setzero_ps();
    __m256 acc6 = _mm256_setzero_ps(), acc7 = _mm256_setzero_ps();
    float bsum[8] = {0,0,0,0,0,0,0,0};

    for (int i = 0; i < nb32; ++i) {
        const float d_w = Avx2Math::fp16_to_fp32(vx[i].d);
        const __m128i qb  = _mm_loadu_si128(reinterpret_cast<const __m128i*>(vx[i].qs));
        const __m128i qlo = _mm_and_si128(qb, mask_lo);
        const __m128i qhi = _mm_and_si128(_mm_srli_epi16(qb, 4), mask_lo);
        const __m256i q4  = mm256_concat_si128(qlo, qhi);

        const int8_t*  yp = tile.qs.data()     + static_cast<size_t>(i) * 256ULL;
        const float*   dp = tile.d.data()      + static_cast<size_t>(i) * 8ULL;
        const int16_t* bp = tile.bsum32.data() + static_cast<size_t>(i) * 8ULL;

        #define Q40_MAC(T, ACC) do { \
            const float dw_dy = d_w * dp[T]; \
            const __m256i p = _mm256_madd_epi16(_mm256_maddubs_epi16(q4, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(yp + (T) * 32))), ones16); \
            ACC = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p), _mm256_set1_ps(dw_dy), ACC); \
            bsum[T] += dw_dy * static_cast<float>(bp[T]); \
        } while (0)
        Q40_MAC(0, acc0); Q40_MAC(1, acc1); Q40_MAC(2, acc2); Q40_MAC(3, acc3);
        if constexpr (NTok > 4) {
            Q40_MAC(4, acc4); Q40_MAC(5, acc5); Q40_MAC(6, acc6); Q40_MAC(7, acc7);
        }
        #undef Q40_MAC
    }
    out[0] = hsum256_ps(acc0) - 8.0f * bsum[0]; out[1] = hsum256_ps(acc1) - 8.0f * bsum[1];
    out[2] = hsum256_ps(acc2) - 8.0f * bsum[2]; out[3] = hsum256_ps(acc3) - 8.0f * bsum[3];
    if constexpr (NTok > 4) {
        out[4] = hsum256_ps(acc4) - 8.0f * bsum[4]; out[5] = hsum256_ps(acc5) - 8.0f * bsum[5];
        out[6] = hsum256_ps(acc6) - 8.0f * bsum[6]; out[7] = hsum256_ps(acc7) - 8.0f * bsum[7];
    }
}

static inline int l2_adaptive_row_tile(size_t rbytes) {
    // Keep weight tile <= 128 KB so weights + Q8 activation tile stay 100% resident in private 256 KB L2 cache
    if (rbytes == 0) return 64;
    int rt = static_cast<int>(131072ULL / rbytes);
    rt = std::clamp(rt, 8, 64);
    rt = (rt / 8) * 8;
    return (rt >= 8) ? rt : 8;
}

template <int NTok> static inline void dot_iq4_xs_f32_xN(const void*, size_t, const float*, int, float*);
template <int NTok> static inline void dot_iq4_nl_f32_xN(const void*, size_t, const float*, int, float*);
template <int NTok> static inline void dot_q4_K_f32_xN(const void*, size_t, const float*, int, float*);
template <int NTok> static inline void dot_q5_K_f32_xN(const void*, size_t, const float*, int, float*);
template <int NTok> static inline void dot_q6_K_f32_xN(const void*, size_t, const float*, int, float*);
template <int NTok> static inline void dot_q5_1_f32_xN(const void*, size_t, const float*, int, float*);
template <int NTok> static inline void dot_q5_0_f32_xN(const void*, size_t, const float*, int, float*);
template <int NTok> static inline void dot_q8_0_f32_xN(const void*, size_t, const float*, int, float*);
template <int NTok> static inline void dot_q4_0_f32_xN(const void*, size_t, const float*, int, float*);

template <QuantType QT, typename Dot4Fn, typename Dot8Fn, typename F32Dot4Fn, typename F32Dot8Fn>
static void run_gemm_q8k_tiled(float* Y, int stride_y, const void* wdata, size_t row_offset,
                               const Q8KBatch& batch, int rows, size_t rbytes, bool parallel,
                               Dot4Fn dot4, Dot8Fn dot8, F32Dot4Fn f32_dot4, F32Dot8Fn f32_dot8) {
    const int row_tile = l2_adaptive_row_tile(rbytes);
    const int num_rtiles = (rows + row_tile - 1) / row_tile;
    const int n_tiles = static_cast<int>(batch.tiles.size());
    const int nb256 = batch.nb256;
    const int cols = nb256 * 256;

    #pragma omp parallel for if(parallel && rows >= 32) schedule(static)
    for (int rt = 0; rt < num_rtiles; ++rt) {
        const int r0 = rt * row_tile;
        const int r1 = std::min(rows, r0 + row_tile);
        for (int g = 0; g < n_tiles; ++g) {
            const auto& tile = batch.tiles[static_cast<size_t>(g)];
            const int t_base = g * 8;
            const int n_tok = tile.n_tok;
            if (tile.use_f32) {
                const float* xpack = tile.x_f32.data();
                if (n_tok <= 4) {
                    for (int r = r0; r < r1; ++r) {
                        float tmp[4];
                        f32_dot4(wdata, row_offset + static_cast<size_t>(r), xpack, cols, tmp);
                        for (int t = 0; t < n_tok; ++t) {
                            Y[static_cast<size_t>(t_base + t) * static_cast<size_t>(stride_y) + static_cast<size_t>(r)] = tmp[t];
                        }
                    }
                } else {
                    for (int r = r0; r < r1; ++r) {
                        float tmp[8];
                        f32_dot8(wdata, row_offset + static_cast<size_t>(r), xpack, cols, tmp);
                        for (int t = 0; t < n_tok; ++t) {
                            Y[static_cast<size_t>(t_base + t) * static_cast<size_t>(stride_y) + static_cast<size_t>(r)] = tmp[t];
                        }
                    }
                }
            } else if (n_tok <= 4) {
                for (int r = r0; r < r1; ++r) {
                    const size_t wrow = row_offset + static_cast<size_t>(r);
                    float tmp[4];
                    dot4(wdata, wrow, tile, nb256, tmp);
                    if (tile.has_outliers) {
                        for (int t = 0; t < n_tok; ++t) {
                            if (tile.n_outliers[t] > 0) {
                                tmp[t] += dot_outliers<QT>(wdata, wrow, cols, tile.outliers[t], tile.n_outliers[t]);
                            }
                        }
                    }
                    for (int t = 0; t < n_tok; ++t) {
                        Y[static_cast<size_t>(t_base + t) * static_cast<size_t>(stride_y) + static_cast<size_t>(r)] = tmp[t];
                    }
                }
            } else {
                for (int r = r0; r < r1; ++r) {
                    const size_t wrow = row_offset + static_cast<size_t>(r);
                    float tmp[8];
                    dot8(wdata, wrow, tile, nb256, tmp);
                    if (tile.has_outliers) {
                        for (int t = 0; t < n_tok; ++t) {
                            if (tile.n_outliers[t] > 0) {
                                tmp[t] += dot_outliers<QT>(wdata, wrow, cols, tile.outliers[t], tile.n_outliers[t]);
                            }
                        }
                    }
                    for (int t = 0; t < n_tok; ++t) {
                        Y[static_cast<size_t>(t_base + t) * static_cast<size_t>(stride_y) + static_cast<size_t>(r)] = tmp[t];
                    }
                }
            }
        }
    }
}

template <QuantType QT, typename Dot4Fn, typename Dot8Fn, typename F32Dot4Fn, typename F32Dot8Fn>
static void run_gemm_q8_32_tiled(float* Y, int stride_y, const void* wdata, size_t row_offset,
                                 const Q8_32Batch& batch, int rows, size_t rbytes, bool parallel,
                                 Dot4Fn dot4, Dot8Fn dot8, F32Dot4Fn f32_dot4, F32Dot8Fn f32_dot8) {
    const int row_tile = l2_adaptive_row_tile(rbytes);
    const int num_rtiles = (rows + row_tile - 1) / row_tile;
    const int n_tiles = static_cast<int>(batch.tiles.size());
    const int nb32 = batch.nb32;
    const int cols = nb32 * 32;

    #pragma omp parallel for if(parallel && rows >= 32) schedule(static)
    for (int rt = 0; rt < num_rtiles; ++rt) {
        const int r0 = rt * row_tile;
        const int r1 = std::min(rows, r0 + row_tile);
        for (int g = 0; g < n_tiles; ++g) {
            const auto& tile = batch.tiles[static_cast<size_t>(g)];
            const int t_base = g * 8;
            const int n_tok = tile.n_tok;
            if (tile.use_f32) {
                const float* xpack = tile.x_f32.data();
                if (n_tok <= 4) {
                    for (int r = r0; r < r1; ++r) {
                        float tmp[4];
                        f32_dot4(wdata, row_offset + static_cast<size_t>(r), xpack, cols, tmp);
                        for (int t = 0; t < n_tok; ++t) {
                            Y[static_cast<size_t>(t_base + t) * static_cast<size_t>(stride_y) + static_cast<size_t>(r)] = tmp[t];
                        }
                    }
                } else {
                    for (int r = r0; r < r1; ++r) {
                        float tmp[8];
                        f32_dot8(wdata, row_offset + static_cast<size_t>(r), xpack, cols, tmp);
                        for (int t = 0; t < n_tok; ++t) {
                            Y[static_cast<size_t>(t_base + t) * static_cast<size_t>(stride_y) + static_cast<size_t>(r)] = tmp[t];
                        }
                    }
                }
            } else if (n_tok <= 4) {
                for (int r = r0; r < r1; ++r) {
                    const size_t wrow = row_offset + static_cast<size_t>(r);
                    float tmp[4];
                    dot4(wdata, wrow, tile, nb32, tmp);
                    if (tile.has_outliers) {
                        for (int t = 0; t < n_tok; ++t) {
                            if (tile.n_outliers[t] > 0) {
                                tmp[t] += dot_outliers<QT>(wdata, wrow, cols, tile.outliers[t], tile.n_outliers[t]);
                            }
                        }
                    }
                    for (int t = 0; t < n_tok; ++t) {
                        Y[static_cast<size_t>(t_base + t) * static_cast<size_t>(stride_y) + static_cast<size_t>(r)] = tmp[t];
                    }
                }
            } else {
                for (int r = r0; r < r1; ++r) {
                    const size_t wrow = row_offset + static_cast<size_t>(r);
                    float tmp[8];
                    dot8(wdata, wrow, tile, nb32, tmp);
                    if (tile.has_outliers) {
                        for (int t = 0; t < n_tok; ++t) {
                            if (tile.n_outliers[t] > 0) {
                                tmp[t] += dot_outliers<QT>(wdata, wrow, cols, tile.outliers[t], tile.n_outliers[t]);
                            }
                        }
                    }
                    for (int t = 0; t < n_tok; ++t) {
                        Y[static_cast<size_t>(t_base + t) * static_cast<size_t>(stride_y) + static_cast<size_t>(r)] = tmp[t];
                    }
                }
            }
        }
    }
}

// Branchless 1x4 / 1x8 batched float GEMM for BF16, F16, and F32 weights (e.g. per_layer_model_proj, MoE router)
template <QuantType QT, int NTok>
static inline void dot_float_xN(const void* wdata, size_t wrow, const float* const* xp, int cols, float* res) {
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps(), a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
    __m256 a4 = _mm256_setzero_ps(), a5 = _mm256_setzero_ps(), a6 = _mm256_setzero_ps(), a7 = _mm256_setzero_ps();
    int c = 0;
    for (; c <= cols - 8; c += 8) {
        __m256 vw;
        if constexpr (QT == QuantType::F32) {
            vw = _mm256_loadu_ps(reinterpret_cast<const float*>(wdata) + wrow * static_cast<size_t>(cols) + c);
        } else if constexpr (QT == QuantType::BF16) {
            __m128i bf = _mm_loadu_si128(reinterpret_cast<const __m128i*>(reinterpret_cast<const uint16_t*>(wdata) + wrow * static_cast<size_t>(cols) + c));
            vw = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(bf), 16));
        } else {
            __m128i h = _mm_loadu_si128(reinterpret_cast<const __m128i*>(reinterpret_cast<const uint16_t*>(wdata) + wrow * static_cast<size_t>(cols) + c));
            vw = _mm256_cvtph_ps(h);
        }
        a0 = _mm256_fmadd_ps(vw, _mm256_loadu_ps(xp[0] + c), a0);
        a1 = _mm256_fmadd_ps(vw, _mm256_loadu_ps(xp[1] + c), a1);
        a2 = _mm256_fmadd_ps(vw, _mm256_loadu_ps(xp[2] + c), a2);
        a3 = _mm256_fmadd_ps(vw, _mm256_loadu_ps(xp[3] + c), a3);
        if constexpr (NTok > 4) {
            a4 = _mm256_fmadd_ps(vw, _mm256_loadu_ps(xp[4] + c), a4);
            a5 = _mm256_fmadd_ps(vw, _mm256_loadu_ps(xp[5] + c), a5);
            a6 = _mm256_fmadd_ps(vw, _mm256_loadu_ps(xp[6] + c), a6);
            a7 = _mm256_fmadd_ps(vw, _mm256_loadu_ps(xp[7] + c), a7);
        }
    }
    res[0] = hsum256_ps(a0); res[1] = hsum256_ps(a1); res[2] = hsum256_ps(a2); res[3] = hsum256_ps(a3);
    if constexpr (NTok > 4) {
        res[4] = hsum256_ps(a4); res[5] = hsum256_ps(a5); res[6] = hsum256_ps(a6); res[7] = hsum256_ps(a7);
    }
    for (; c < cols; ++c) {
        float wv = 0.0f;
        if constexpr (QT == QuantType::F32) wv = reinterpret_cast<const float*>(wdata)[wrow * static_cast<size_t>(cols) + c];
        else if constexpr (QT == QuantType::BF16) wv = Avx2Math::bf16_to_fp32(reinterpret_cast<const uint16_t*>(wdata)[wrow * static_cast<size_t>(cols) + c]);
        else wv = Avx2Math::fp16_to_fp32(reinterpret_cast<const uint16_t*>(wdata)[wrow * static_cast<size_t>(cols) + c]);
        for (int t = 0; t < NTok; ++t) res[t] += wv * xp[t][c];
    }
}

template <QuantType QT>
static void run_gemm_float_tiled(float* Y, int stride_y, const void* wdata, size_t row_offset,
                                 const float* X, int stride_x, int n_tokens, int rows, int cols, bool parallel) {
    const size_t rbytes = static_cast<size_t>(cols) * (QT == QuantType::F32 ? 4ULL : 2ULL);
    const int row_tile = l2_adaptive_row_tile(rbytes);
    const int num_rtiles = (rows + row_tile - 1) / row_tile;

    #pragma omp parallel for if(parallel && rows >= 32) schedule(static)
    for (int rt = 0; rt < num_rtiles; ++rt) {
        const int r0 = rt * row_tile;
        const int r1 = std::min(rows, r0 + row_tile);
        for (int t0 = 0; t0 < n_tokens; t0 += 8) {
            const int n_tok = std::min(8, n_tokens - t0);
            const float* xp[8];
            for (int t = 0; t < n_tok; ++t) xp[t] = X + static_cast<size_t>(t0 + t) * static_cast<size_t>(stride_x);
            for (int t = n_tok; t < 8; ++t) xp[t] = xp[0];

            if (n_tok <= 4) {
                for (int r = r0; r < r1; ++r) {
                    float res[4];
                    dot_float_xN<QT, 4>(wdata, row_offset + static_cast<size_t>(r), xp, cols, res);
                    for (int t = 0; t < n_tok; ++t) {
                        Y[static_cast<size_t>(t0 + t) * static_cast<size_t>(stride_y) + static_cast<size_t>(r)] = res[t];
                    }
                }
            } else {
                for (int r = r0; r < r1; ++r) {
                    float res[8];
                    dot_float_xN<QT, 8>(wdata, row_offset + static_cast<size_t>(r), xp, cols, res);
                    for (int t = 0; t < n_tok; ++t) {
                        Y[static_cast<size_t>(t0 + t) * static_cast<size_t>(stride_y) + static_cast<size_t>(r)] = res[t];
                    }
                }
            }
        }
    }
}

void Avx2Math::gemm_slice_q8k(float* Y, int stride_y, const TensorDesc& weight, size_t row_offset,
                              const Q8KBatch& batch, int rows, int cols, bool parallel) {
    if (!Y || !weight.data || rows <= 0 || cols < 256 || batch.n_tokens <= 0) return;
    const size_t rb = row_bytes(weight.type, cols);
    if (weight.size_bytes > 0) {
        if (rb == 0 || row_offset + static_cast<size_t>(rows) > weight.size_bytes / rb) {
            for (int t = 0; t < batch.n_tokens; ++t) {
                std::memset(Y + static_cast<size_t>(t) * static_cast<size_t>(stride_y), 0, static_cast<size_t>(rows) * sizeof(float));
            }
            return;
        }
    }
    switch (weight.type) {
        case QuantType::IQ4_XS: run_gemm_q8k_tiled<QuantType::IQ4_XS>(Y, stride_y, weight.data, row_offset, batch, rows, rb, parallel, dot_iq4_xs_q8k_xN<4>, dot_iq4_xs_q8k_xN<8>, dot_iq4_xs_f32_xN<4>, dot_iq4_xs_f32_xN<8>); break;
        case QuantType::Q4_K:   run_gemm_q8k_tiled<QuantType::Q4_K>(Y, stride_y, weight.data, row_offset, batch, rows, rb, parallel, dot_q4_K_q8k_xN<4>,   dot_q4_K_q8k_xN<8>,   dot_q4_K_f32_xN<4>,   dot_q4_K_f32_xN<8>);   break;
        case QuantType::Q5_K:   run_gemm_q8k_tiled<QuantType::Q5_K>(Y, stride_y, weight.data, row_offset, batch, rows, rb, parallel, dot_q5_K_q8k_xN<4>,   dot_q5_K_q8k_xN<8>,   dot_q5_K_f32_xN<4>,   dot_q5_K_f32_xN<8>);   break;
        case QuantType::Q6_K:   run_gemm_q8k_tiled<QuantType::Q6_K>(Y, stride_y, weight.data, row_offset, batch, rows, rb, parallel, dot_q6_K_q8k_xN<4>,   dot_q6_K_q8k_xN<8>,   dot_q6_K_f32_xN<4>,   dot_q6_K_f32_xN<8>);   break;
        default: break;
    }
}

void Avx2Math::gemm_slice_q8_32(float* Y, int stride_y, const TensorDesc& weight, size_t row_offset,
                                const Q8_32Batch& batch, int rows, int cols, bool parallel) {
    if (!Y || !weight.data || rows <= 0 || cols < 32 || batch.n_tokens <= 0) return;
    const size_t rb = row_bytes(weight.type, cols);
    if (weight.size_bytes > 0) {
        if (rb == 0 || row_offset + static_cast<size_t>(rows) > weight.size_bytes / rb) {
            for (int t = 0; t < batch.n_tokens; ++t) {
                std::memset(Y + static_cast<size_t>(t) * static_cast<size_t>(stride_y), 0, static_cast<size_t>(rows) * sizeof(float));
            }
            return;
        }
    }
    switch (weight.type) {
        case QuantType::IQ4_NL: run_gemm_q8_32_tiled<QuantType::IQ4_NL>(Y, stride_y, weight.data, row_offset, batch, rows, rb, parallel, dot_iq4_nl_q8_32_xN<4>, dot_iq4_nl_q8_32_xN<8>, dot_iq4_nl_f32_xN<4>, dot_iq4_nl_f32_xN<8>); break;
        case QuantType::Q5_1:   run_gemm_q8_32_tiled<QuantType::Q5_1>(Y, stride_y, weight.data, row_offset, batch, rows, rb, parallel, dot_q5_1_q8_32_xN<4>,   dot_q5_1_q8_32_xN<8>,   dot_q5_1_f32_xN<4>,   dot_q5_1_f32_xN<8>);   break;
        case QuantType::Q5_0:   run_gemm_q8_32_tiled<QuantType::Q5_0>(Y, stride_y, weight.data, row_offset, batch, rows, rb, parallel, dot_q5_0_q8_32_xN<4>,   dot_q5_0_q8_32_xN<8>,   dot_q5_0_f32_xN<4>,   dot_q5_0_f32_xN<8>);   break;
        case QuantType::Q8_0:   run_gemm_q8_32_tiled<QuantType::Q8_0>(Y, stride_y, weight.data, row_offset, batch, rows, rb, parallel, dot_q8_0_q8_32_xN<4>,   dot_q8_0_q8_32_xN<8>,   dot_q8_0_f32_xN<4>,   dot_q8_0_f32_xN<8>);   break;
        case QuantType::Q4_0:   run_gemm_q8_32_tiled<QuantType::Q4_0>(Y, stride_y, weight.data, row_offset, batch, rows, rb, parallel, dot_q4_0_q8_32_xN<4>,   dot_q4_0_q8_32_xN<8>,   dot_q4_0_f32_xN<4>,   dot_q4_0_f32_xN<8>);   break;
        default: break;
    }
}

void Avx2Math::gemm_slice(float* Y, int stride_y, const TensorDesc& weight, size_t row_offset,
                          const float* X, int stride_x, int n_tokens, int rows, int cols, bool parallel) {
    if (!Y || !weight.data || !X || n_tokens <= 0 || rows <= 0 || cols <= 0) return;
    if (weight.size_bytes > 0) {
        const size_t rb = row_bytes(weight.type, cols);
        if (rb == 0 || row_offset + static_cast<size_t>(rows) > weight.size_bytes / rb) {
            for (int t = 0; t < n_tokens; ++t) {
                std::memset(Y + static_cast<size_t>(t) * static_cast<size_t>(stride_y), 0, static_cast<size_t>(rows) * sizeof(float));
            }
            return;
        }
    }

    if (n_tokens == 1) {
        if (parallel) {
            gemv_slice(Y, weight, row_offset, X, rows, cols);
        } else {
            if (DotRowQ8Fn q8_fn = get_dot_row_q8_fn(weight.type, cols)) {
                Q8Act32 act;
                quantize_q8_act(X, cols, act);
                for (int r = 0; r < rows; ++r) Y[r] = q8_fn(weight.data, row_offset + static_cast<size_t>(r), act, cols);
            } else {
                for (int r = 0; r < rows; ++r) Y[r] = dot_row(weight, row_offset + static_cast<size_t>(r), X, cols);
            }
        }
        return;
    }

    if (supports_q8k_gemm(weight.type, cols)) {
        Q8KBatch batch;
        pack_q8k_batch(X, stride_x, cols, n_tokens, batch);
        gemm_slice_q8k(Y, stride_y, weight, row_offset, batch, rows, cols, parallel);
        return;
    }
    if (supports_q8_32_gemm(weight.type, cols)) {
        Q8_32Batch batch;
        pack_q8_32_batch(X, stride_x, cols, n_tokens, batch);
        gemm_slice_q8_32(Y, stride_y, weight, row_offset, batch, rows, cols, parallel);
        return;
    }
    if (weight.type == QuantType::F32) {
        run_gemm_float_tiled<QuantType::F32>(Y, stride_y, weight.data, row_offset, X, stride_x, n_tokens, rows, cols, parallel);
        return;
    }
    if (weight.type == QuantType::BF16) {
        run_gemm_float_tiled<QuantType::BF16>(Y, stride_y, weight.data, row_offset, X, stride_x, n_tokens, rows, cols, parallel);
        return;
    }
    if (weight.type == QuantType::F16) {
        run_gemm_float_tiled<QuantType::F16>(Y, stride_y, weight.data, row_offset, X, stride_x, n_tokens, rows, cols, parallel);
        return;
    }
    if (weight.type == QuantType::Q3_K && (cols % 256) == 0 && cols >= 256) {
        gemm_slice_f32(Y, stride_y, weight, row_offset, X, stride_x, n_tokens, rows, cols, parallel);
        return;
    }

    // Fallback for remaining unaligned formats
    for (int t = 0; t < n_tokens; ++t) {
        const float* xt = X + static_cast<size_t>(t) * static_cast<size_t>(stride_x);
        float* yt = Y + static_cast<size_t>(t) * static_cast<size_t>(stride_y);
        if (parallel) {
            gemv_slice(yt, weight, row_offset, xt, rows, cols);
        } else {
            for (int r = 0; r < rows; ++r) yt[r] = dot_row(weight, row_offset + static_cast<size_t>(r), xt, cols);
        }
    }
}

void Avx2Math::gemm(float* Y, int stride_y, const TensorDesc& weight,
                    const float* X, int stride_x, int n_tokens, int rows, int cols) {
    gemm_slice(Y, stride_y, weight, 0, X, stride_x, n_tokens, rows, cols, true);
}

// ============================================================================
// High-Throughput 1x4 / 1x8 L1-Packed Exact FP32-Activation Quantized GEMM
// Unpacks each quantized weight block ONCE per 8 (or 4) tokens while reading
// L1-packed contiguous [cols/8][NTok][8] FP32 activations (zero 4KB aliasing,
// 1 base register instead of 8, zero Q8 quantization error across all 9 formats!).
// ============================================================================

#define HAVEN_F32_FMA8(VW, OFF) do { \
    const __m256 _w = (VW); \
    const float* _xp = xpack + static_cast<size_t>(OFF) * static_cast<size_t>(NTok); \
    a0 = _mm256_fmadd_ps(_w, _mm256_loadu_ps(_xp +  0), a0); \
    a1 = _mm256_fmadd_ps(_w, _mm256_loadu_ps(_xp +  8), a1); \
    a2 = _mm256_fmadd_ps(_w, _mm256_loadu_ps(_xp + 16), a2); \
    a3 = _mm256_fmadd_ps(_w, _mm256_loadu_ps(_xp + 24), a3); \
    if constexpr (NTok > 4) { \
        a4 = _mm256_fmadd_ps(_w, _mm256_loadu_ps(_xp + 32), a4); \
        a5 = _mm256_fmadd_ps(_w, _mm256_loadu_ps(_xp + 40), a5); \
        a6 = _mm256_fmadd_ps(_w, _mm256_loadu_ps(_xp + 48), a6); \
        a7 = _mm256_fmadd_ps(_w, _mm256_loadu_ps(_xp + 56), a7); \
    } \
} while (0)

template <int NTok>
static inline void dot_iq4_xs_f32_xN(const void* data, size_t row, const float* xpack, int cols, float* res) {
    const int nb = cols / 256;
    const auto* vx = reinterpret_cast<const block_iq4_xs*>(data) + row * static_cast<size_t>(nb);
    const __m128i lut = _mm_loadu_si128(reinterpret_cast<const __m128i*>(kvalues_iq4nl));
    const __m128i mask_lo = _mm_set1_epi8(0x0F);
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps(), a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
    __m256 a4 = _mm256_setzero_ps(), a5 = _mm256_setzero_ps(), a6 = _mm256_setzero_ps(), a7 = _mm256_setzero_ps();

    for (int b = 0; b < nb; ++b) {
        const float d = Avx2Math::fp16_to_fp32(vx[b].d);
        const uint16_t sh = vx[b].scales_h;
        const uint8_t* sl = vx[b].scales_l;
        const uint8_t* qs = vx[b].qs;
        for (int ib = 0; ib < 8; ++ib) {
            const int ls = ((sl[ib >> 1] >> (4 * (ib & 1))) & 0x0F) | (((sh >> (2 * ib)) & 3) << 4);
            const __m256 vdl = _mm256_set1_ps(d * static_cast<float>(ls - 32));
            const __m128i qb  = _mm_loadu_si128(reinterpret_cast<const __m128i*>(qs + (ib << 4)));
            const __m128i qlo = _mm_shuffle_epi8(lut, _mm_and_si128(qb, mask_lo));
            const __m128i qhi = _mm_shuffle_epi8(lut, _mm_and_si128(_mm_srli_epi16(qb, 4), mask_lo));
            const int off = (b << 8) + (ib << 5);
            HAVEN_F32_FMA8(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(qlo)), vdl), off + 0);
            HAVEN_F32_FMA8(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(qlo, 8))), vdl), off + 8);
            HAVEN_F32_FMA8(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(qhi)), vdl), off + 16);
            HAVEN_F32_FMA8(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(qhi, 8))), vdl), off + 24);
        }
    }
    res[0] = hsum256_ps(a0); res[1] = hsum256_ps(a1); res[2] = hsum256_ps(a2); res[3] = hsum256_ps(a3);
    if constexpr (NTok > 4) {
        res[4] = hsum256_ps(a4); res[5] = hsum256_ps(a5); res[6] = hsum256_ps(a6); res[7] = hsum256_ps(a7);
    }
}

template <int NTok>
static inline void dot_iq4_nl_f32_xN(const void* data, size_t row, const float* xpack, int cols, float* res) {
    const int nb = cols / 32;
    const auto* vx = reinterpret_cast<const block_iq4_nl*>(data) + row * static_cast<size_t>(nb);
    const __m128i lut = _mm_loadu_si128(reinterpret_cast<const __m128i*>(kvalues_iq4nl));
    const __m128i mask_lo = _mm_set1_epi8(0x0F);
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps(), a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
    __m256 a4 = _mm256_setzero_ps(), a5 = _mm256_setzero_ps(), a6 = _mm256_setzero_ps(), a7 = _mm256_setzero_ps();

    for (int b = 0; b < nb; ++b) {
        const __m256 vd = _mm256_set1_ps(Avx2Math::fp16_to_fp32(vx[b].d));
        const __m128i qb  = _mm_loadu_si128(reinterpret_cast<const __m128i*>(vx[b].qs));
        const __m128i qlo = _mm_shuffle_epi8(lut, _mm_and_si128(qb, mask_lo));
        const __m128i qhi = _mm_shuffle_epi8(lut, _mm_and_si128(_mm_srli_epi16(qb, 4), mask_lo));
        const int off = b << 5;
        HAVEN_F32_FMA8(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(qlo)), vd), off + 0);
        HAVEN_F32_FMA8(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(qlo, 8))), vd), off + 8);
        HAVEN_F32_FMA8(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(qhi)), vd), off + 16);
        HAVEN_F32_FMA8(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(qhi, 8))), vd), off + 24);
    }
    res[0] = hsum256_ps(a0); res[1] = hsum256_ps(a1); res[2] = hsum256_ps(a2); res[3] = hsum256_ps(a3);
    if constexpr (NTok > 4) {
        res[4] = hsum256_ps(a4); res[5] = hsum256_ps(a5); res[6] = hsum256_ps(a6); res[7] = hsum256_ps(a7);
    }
}

template <int NTok>
static inline void dot_q4_K_f32_xN(const void* data, size_t row, const float* xpack, int cols, float* res) {
    const int nb = cols / 256;
    const auto* vx = reinterpret_cast<const block_q4_K*>(data) + row * static_cast<size_t>(nb);
    const __m128i mask_lo = _mm_set1_epi8(0x0F);
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps(), a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
    __m256 a4 = _mm256_setzero_ps(), a5 = _mm256_setzero_ps(), a6 = _mm256_setzero_ps(), a7 = _mm256_setzero_ps();

    for (int b = 0; b < nb; ++b) {
        const float d = Avx2Math::fp16_to_fp32(vx[b].d);
        const float dmin = Avx2Math::fp16_to_fp32(vx[b].dmin);
        const uint8_t* q = vx[b].qs;
        uint8_t sc, mn;
        for (int j = 0; j < 4; ++j) {
            get_scale_min_k4(2 * j,     vx[b].scales, &sc, &mn);
            const __m256 vd1 = _mm256_set1_ps(d * static_cast<float>(sc));
            const __m256 vm1 = _mm256_set1_ps(dmin * static_cast<float>(mn));
            get_scale_min_k4(2 * j + 1, vx[b].scales, &sc, &mn);
            const __m256 vd2 = _mm256_set1_ps(d * static_cast<float>(sc));
            const __m256 vm2 = _mm256_set1_ps(dmin * static_cast<float>(mn));
            const int base = (b << 8) + (j << 6);
            for (int k = 0; k < 4; ++k) {
                const int off = k << 3;
                const __m128i qb  = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(q + off));
                const __m128i qlo = _mm_and_si128(qb, mask_lo);
                const __m128i qhi = _mm_and_si128(_mm_srli_epi16(qb, 4), mask_lo);
                HAVEN_F32_FMA8(_mm256_fmsub_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(qlo)), vd1, vm1), base + off);
                HAVEN_F32_FMA8(_mm256_fmsub_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(qhi)), vd2, vm2), base + 32 + off);
            }
            q += 32;
        }
    }
    res[0] = hsum256_ps(a0); res[1] = hsum256_ps(a1); res[2] = hsum256_ps(a2); res[3] = hsum256_ps(a3);
    if constexpr (NTok > 4) {
        res[4] = hsum256_ps(a4); res[5] = hsum256_ps(a5); res[6] = hsum256_ps(a6); res[7] = hsum256_ps(a7);
    }
}

template <int NTok>
static inline void dot_q5_K_f32_xN(const void* data, size_t row, const float* xpack, int cols, float* res) {
    const int nb = cols / 256;
    const auto* vx = reinterpret_cast<const block_q5_K*>(data) + row * static_cast<size_t>(nb);
    const __m128i mask_lo = _mm_set1_epi8(0x0F);
    const __m128i val16   = _mm_set1_epi8(0x10);
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps(), a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
    __m256 a4 = _mm256_setzero_ps(), a5 = _mm256_setzero_ps(), a6 = _mm256_setzero_ps(), a7 = _mm256_setzero_ps();

    for (int b = 0; b < nb; ++b) {
        const float d = Avx2Math::fp16_to_fp32(vx[b].d);
        const float dmin = Avx2Math::fp16_to_fp32(vx[b].dmin);
        const uint8_t* ql = vx[b].qs;
        const uint8_t* qh = vx[b].qh;
        uint8_t sc, mn;
        uint8_t u1 = 1, u2 = 2;
        for (int j = 0; j < 4; ++j) {
            get_scale_min_k4(2 * j,     vx[b].scales, &sc, &mn);
            const __m256 vd1 = _mm256_set1_ps(d * static_cast<float>(sc));
            const __m256 vm1 = _mm256_set1_ps(dmin * static_cast<float>(mn));
            get_scale_min_k4(2 * j + 1, vx[b].scales, &sc, &mn);
            const __m256 vd2 = _mm256_set1_ps(d * static_cast<float>(sc));
            const __m256 vm2 = _mm256_set1_ps(dmin * static_cast<float>(mn));
            const __m128i vu1 = _mm_set1_epi8(static_cast<char>(u1));
            const __m128i vu2 = _mm_set1_epi8(static_cast<char>(u2));
            const int base = (b << 8) + (j << 6);
            for (int k = 0; k < 4; ++k) {
                const int off = k << 3;
                const __m128i qb  = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(ql + off));
                const __m128i qhb = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(qh + off));
                const __m128i h1  = _mm_and_si128(_mm_cmpeq_epi8(_mm_and_si128(qhb, vu1), vu1), val16);
                const __m128i h2  = _mm_and_si128(_mm_cmpeq_epi8(_mm_and_si128(qhb, vu2), vu2), val16);
                const __m128i q1  = _mm_or_si128(_mm_and_si128(qb, mask_lo), h1);
                const __m128i q2  = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(qb, 4), mask_lo), h2);
                HAVEN_F32_FMA8(_mm256_fmsub_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(q1)), vd1, vm1), base + off);
                HAVEN_F32_FMA8(_mm256_fmsub_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(q2)), vd2, vm2), base + 32 + off);
            }
            ql += 32;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
    res[0] = hsum256_ps(a0); res[1] = hsum256_ps(a1); res[2] = hsum256_ps(a2); res[3] = hsum256_ps(a3);
    if constexpr (NTok > 4) {
        res[4] = hsum256_ps(a4); res[5] = hsum256_ps(a5); res[6] = hsum256_ps(a6); res[7] = hsum256_ps(a7);
    }
}

template <int NTok>
static inline void dot_q6_K_f32_xN(const void* data, size_t row, const float* xpack, int cols, float* res) {
    const int nb = cols / 256;
    const auto* vx = reinterpret_cast<const block_q6_K*>(data) + row * static_cast<size_t>(nb);
    const __m128i mask_lo4 = _mm_set1_epi8(0x0F);
    const __m128i mask_hi2 = _mm_set1_epi8(0x30);
    const __m256i bias32   = _mm256_set1_epi32(32);
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps(), a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
    __m256 a4 = _mm256_setzero_ps(), a5 = _mm256_setzero_ps(), a6 = _mm256_setzero_ps(), a7 = _mm256_setzero_ps();

    for (int b = 0; b < nb; ++b) {
        const float d = Avx2Math::fp16_to_fp32(vx[b].d);
        const uint8_t* ql = vx[b].ql;
        const uint8_t* qh = vx[b].qh;
        const int8_t*  sc = vx[b].scales;
        for (int n_chunk = 0; n_chunk < 256; n_chunk += 128) {
            for (int k = 0; k < 4; ++k) {
                const int is = (k < 2) ? 0 : 1;
                const int l_base = k << 3;
                const int base = (b << 8) + n_chunk + l_base;
                const __m128i ql0 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(ql + l_base));
                const __m128i ql1 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(ql + l_base + 32));
                const __m128i qhb = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(qh + l_base));
                const __m128i q1b = _mm_or_si128(_mm_and_si128(ql0, mask_lo4), _mm_and_si128(_mm_slli_epi16(qhb, 4), mask_hi2));
                const __m128i q2b = _mm_or_si128(_mm_and_si128(ql1, mask_lo4), _mm_and_si128(_mm_slli_epi16(qhb, 2), mask_hi2));
                const __m128i q3b = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(ql0, 4), mask_lo4), _mm_and_si128(qhb, mask_hi2));
                const __m128i q4b = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(ql1, 4), mask_lo4), _mm_and_si128(_mm_srli_epi16(qhb, 2), mask_hi2));
                HAVEN_F32_FMA8(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(q1b), bias32)), _mm256_set1_ps(d * static_cast<float>(sc[is + 0]))), base + 0);
                HAVEN_F32_FMA8(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(q2b), bias32)), _mm256_set1_ps(d * static_cast<float>(sc[is + 2]))), base + 32);
                HAVEN_F32_FMA8(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(q3b), bias32)), _mm256_set1_ps(d * static_cast<float>(sc[is + 4]))), base + 64);
                HAVEN_F32_FMA8(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(q4b), bias32)), _mm256_set1_ps(d * static_cast<float>(sc[is + 6]))), base + 96);
            }
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
    res[0] = hsum256_ps(a0); res[1] = hsum256_ps(a1); res[2] = hsum256_ps(a2); res[3] = hsum256_ps(a3);
    if constexpr (NTok > 4) {
        res[4] = hsum256_ps(a4); res[5] = hsum256_ps(a5); res[6] = hsum256_ps(a6); res[7] = hsum256_ps(a7);
    }
}

template <int NTok>
static inline void dot_q5_1_f32_xN(const void* data, size_t row, const float* xpack, int cols, float* res) {
    const int nb = cols / 32;
    const auto* vx = reinterpret_cast<const block_q5_1*>(data) + row * static_cast<size_t>(nb);
    const __m128i mask_lo = _mm_set1_epi8(0x0F);
    const __m256i shifts  = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    const __m256i mask10  = _mm256_set1_epi32(0x10);
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps(), a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
    __m256 a4 = _mm256_setzero_ps(), a5 = _mm256_setzero_ps(), a6 = _mm256_setzero_ps(), a7 = _mm256_setzero_ps();

    for (int b = 0; b < nb; ++b) {
        const __m256 vd = _mm256_set1_ps(Avx2Math::fp16_to_fp32(vx[b].d));
        const __m256 vm = _mm256_set1_ps(Avx2Math::fp16_to_fp32(vx[b].m));
        uint32_t qh = 0;
        std::memcpy(&qh, vx[b].qh, sizeof(qh));
        const __m128i qb  = _mm_loadu_si128(reinterpret_cast<const __m128i*>(vx[b].qs));
        const __m128i qlo = _mm_and_si128(qb, mask_lo);
        const __m128i qhi = _mm_and_si128(_mm_srli_epi16(qb, 4), mask_lo);
        const int off = b << 5;
        auto mk_w = [&](__m128i q8, uint32_t qh8) -> __m256 {
            const __m256i xh = _mm256_and_si256(_mm256_slli_epi32(_mm256_srlv_epi32(_mm256_set1_epi32(static_cast<int>(qh8)), shifts), 4), mask10);
            return _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_or_si256(_mm256_cvtepu8_epi32(q8), xh)), vd, vm);
        };
        HAVEN_F32_FMA8(mk_w(qlo, qh),                       off + 0);
        HAVEN_F32_FMA8(mk_w(_mm_srli_si128(qlo, 8), qh >> 8), off + 8);
        HAVEN_F32_FMA8(mk_w(qhi, qh >> 16),                 off + 16);
        HAVEN_F32_FMA8(mk_w(_mm_srli_si128(qhi, 8), qh >> 24), off + 24);
    }
    res[0] = hsum256_ps(a0); res[1] = hsum256_ps(a1); res[2] = hsum256_ps(a2); res[3] = hsum256_ps(a3);
    if constexpr (NTok > 4) {
        res[4] = hsum256_ps(a4); res[5] = hsum256_ps(a5); res[6] = hsum256_ps(a6); res[7] = hsum256_ps(a7);
    }
}

template <int NTok>
static inline void dot_q5_0_f32_xN(const void* data, size_t row, const float* xpack, int cols, float* res) {
    const int nb = cols / 32;
    const auto* vx = reinterpret_cast<const block_q5_0*>(data) + row * static_cast<size_t>(nb);
    const __m128i mask_lo = _mm_set1_epi8(0x0F);
    const __m256i shifts  = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    const __m256i mask10  = _mm256_set1_epi32(0x10);
    const __m256i bias16  = _mm256_set1_epi32(16);
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps(), a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
    __m256 a4 = _mm256_setzero_ps(), a5 = _mm256_setzero_ps(), a6 = _mm256_setzero_ps(), a7 = _mm256_setzero_ps();

    for (int b = 0; b < nb; ++b) {
        const __m256 vd = _mm256_set1_ps(Avx2Math::fp16_to_fp32(vx[b].d));
        uint32_t qh = 0;
        std::memcpy(&qh, vx[b].qh, sizeof(qh));
        const __m128i qb  = _mm_loadu_si128(reinterpret_cast<const __m128i*>(vx[b].qs));
        const __m128i qlo = _mm_and_si128(qb, mask_lo);
        const __m128i qhi = _mm_and_si128(_mm_srli_epi16(qb, 4), mask_lo);
        const int off = b << 5;
        auto mk_w = [&](__m128i q8, uint32_t qh8) -> __m256 {
            const __m256i xh = _mm256_and_si256(_mm256_slli_epi32(_mm256_srlv_epi32(_mm256_set1_epi32(static_cast<int>(qh8)), shifts), 4), mask10);
            return _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_or_si256(_mm256_cvtepu8_epi32(q8), xh), bias16)), vd);
        };
        HAVEN_F32_FMA8(mk_w(qlo, qh),                       off + 0);
        HAVEN_F32_FMA8(mk_w(_mm_srli_si128(qlo, 8), qh >> 8), off + 8);
        HAVEN_F32_FMA8(mk_w(qhi, qh >> 16),                 off + 16);
        HAVEN_F32_FMA8(mk_w(_mm_srli_si128(qhi, 8), qh >> 24), off + 24);
    }
    res[0] = hsum256_ps(a0); res[1] = hsum256_ps(a1); res[2] = hsum256_ps(a2); res[3] = hsum256_ps(a3);
    if constexpr (NTok > 4) {
        res[4] = hsum256_ps(a4); res[5] = hsum256_ps(a5); res[6] = hsum256_ps(a6); res[7] = hsum256_ps(a7);
    }
}

template <int NTok>
static inline void dot_q8_0_f32_xN(const void* data, size_t row, const float* xpack, int cols, float* res) {
    const int nb = cols / 32;
    const auto* vx = reinterpret_cast<const block_q8_0*>(data) + row * static_cast<size_t>(nb);
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps(), a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
    __m256 a4 = _mm256_setzero_ps(), a5 = _mm256_setzero_ps(), a6 = _mm256_setzero_ps(), a7 = _mm256_setzero_ps();

    for (int b = 0; b < nb; ++b) {
        const __m256 vd = _mm256_set1_ps(Avx2Math::fp16_to_fp32(vx[b].d));
        const __m128i q0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(vx[b].qs));
        const __m128i q1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(vx[b].qs + 16));
        const int off = b << 5;
        HAVEN_F32_FMA8(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q0)), vd), off + 0);
        HAVEN_F32_FMA8(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(q0, 8))), vd), off + 8);
        HAVEN_F32_FMA8(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q1)), vd), off + 16);
        HAVEN_F32_FMA8(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(q1, 8))), vd), off + 24);
    }
    res[0] = hsum256_ps(a0); res[1] = hsum256_ps(a1); res[2] = hsum256_ps(a2); res[3] = hsum256_ps(a3);
    if constexpr (NTok > 4) {
        res[4] = hsum256_ps(a4); res[5] = hsum256_ps(a5); res[6] = hsum256_ps(a6); res[7] = hsum256_ps(a7);
    }
}

template <int NTok>
static inline void dot_q4_0_f32_xN(const void* data, size_t row, const float* xpack, int cols, float* res) {
    const int nb = cols / 32;
    const auto* vx = reinterpret_cast<const block_q4_0*>(data) + row * static_cast<size_t>(nb);
    const __m128i mask_lo = _mm_set1_epi8(0x0F);
    const __m256i bias8   = _mm256_set1_epi32(8);
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps(), a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
    __m256 a4 = _mm256_setzero_ps(), a5 = _mm256_setzero_ps(), a6 = _mm256_setzero_ps(), a7 = _mm256_setzero_ps();

    for (int b = 0; b < nb; ++b) {
        const __m256 vd = _mm256_set1_ps(Avx2Math::fp16_to_fp32(vx[b].d));
        const __m128i qb  = _mm_loadu_si128(reinterpret_cast<const __m128i*>(vx[b].qs));
        const __m128i qlo = _mm_and_si128(qb, mask_lo);
        const __m128i qhi = _mm_and_si128(_mm_srli_epi16(qb, 4), mask_lo);
        const int off = b << 5;
        HAVEN_F32_FMA8(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(qlo), bias8)), vd), off + 0);
        HAVEN_F32_FMA8(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_srli_si128(qlo, 8)), bias8)), vd), off + 8);
        HAVEN_F32_FMA8(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(qhi), bias8)), vd), off + 16);
        HAVEN_F32_FMA8(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_srli_si128(qhi, 8)), bias8)), vd), off + 24);
    }
    res[0] = hsum256_ps(a0); res[1] = hsum256_ps(a1); res[2] = hsum256_ps(a2); res[3] = hsum256_ps(a3);
    if constexpr (NTok > 4) {
        res[4] = hsum256_ps(a4); res[5] = hsum256_ps(a5); res[6] = hsum256_ps(a6); res[7] = hsum256_ps(a7);
    }
}

template <int NTok>
static inline void dot_q3_K_f32_xN(const void* data, size_t row, const float* xpack, int cols, float* res) {
    const int nb = cols / 256;
    const auto* vx = reinterpret_cast<const block_q3_K*>(data) + row * static_cast<size_t>(nb);
    const __m256i mask3 = _mm256_set1_epi8(0x03);
    const __m256i v4    = _mm256_set1_epi8(0x04);
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps(), a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
    __m256 a4 = _mm256_setzero_ps(), a5 = _mm256_setzero_ps(), a6 = _mm256_setzero_ps(), a7 = _mm256_setzero_ps();

    const uint32_t kmask1 = 0x03030303u;
    const uint32_t kmask2 = 0x0f0f0f0fu;
    uint32_t aux[4];
    const int8_t* scales = reinterpret_cast<const int8_t*>(aux);

    for (int b = 0; b < nb; ++b) {
        const float d_all = Avx2Math::fp16_to_fp32(vx[b].d);
        const uint8_t* q  = vx[b].qs;
        const __m256i hm32 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(vx[b].hmask));
        uint8_t m = 1;

        std::memcpy(aux, vx[b].scales, 12);
        const uint32_t tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);

        int is = 0;
        int off = b << 8;
        for (int n_chunk = 0; n_chunk < 2; ++n_chunk) {
            __m256i q32 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q));
            for (int j = 0; j < 4; ++j) {
                const __m256 vdl1 = _mm256_set1_ps(d_all * static_cast<float>(scales[is++] - 32));
                const __m256 vdl2 = _mm256_set1_ps(d_all * static_cast<float>(scales[is++] - 32));
                const __m256i vm  = _mm256_set1_epi8(static_cast<char>(m));

                const __m256i lo2  = _mm256_and_si256(q32, mask3);
                const __m256i hsub = _mm256_andnot_si256(_mm256_cmpeq_epi8(_mm256_and_si256(hm32, vm), vm), v4);
                const __m256i w32  = _mm256_sub_epi8(lo2, hsub);

                const __m128i w_lo16 = _mm256_castsi256_si128(w32);
                const __m128i w_hi16 = _mm256_extracti128_si256(w32, 1);

                HAVEN_F32_FMA8(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(w_lo16)), vdl1), off + 0);
                HAVEN_F32_FMA8(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(w_lo16, 8))), vdl1), off + 8);
                HAVEN_F32_FMA8(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(w_hi16)), vdl2), off + 16);
                HAVEN_F32_FMA8(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(w_hi16, 8))), vdl2), off + 24);

                off += 32;
                q32 = _mm256_srli_epi16(q32, 2);
                m <<= 1;
            }
            q += 32;
        }
    }
    res[0] = hsum256_ps(a0); res[1] = hsum256_ps(a1); res[2] = hsum256_ps(a2); res[3] = hsum256_ps(a3);
    if constexpr (NTok > 4) {
        res[4] = hsum256_ps(a4); res[5] = hsum256_ps(a5); res[6] = hsum256_ps(a6); res[7] = hsum256_ps(a7);
    }
}

#undef HAVEN_F32_FMA8

template <typename Dot4Fn, typename Dot8Fn>
static void run_gemm_quant_f32_tiled(float* Y, int stride_y, const void* wdata, size_t row_offset,
                                     const float* X, int stride_x, int n_tokens, int rows, int cols,
                                     size_t rbytes, bool parallel, Dot4Fn dot4, Dot8Fn dot8) {
    const int row_tile = l2_adaptive_row_tile(rbytes);
    const int num_rtiles = (rows + row_tile - 1) / row_tile;
    const int n_tiles = (n_tokens + 7) / 8;

    // Pre-pack all 8-token tiles ONCE in a single flat buffer into L1-friendly [cols/8][NTok][8] layout!
    const size_t tile_stride = static_cast<size_t>(cols) * 8ULL;
    std::vector<float> packed_buf(static_cast<size_t>(n_tiles) * tile_stride);
    #pragma omp parallel for if(parallel && n_tiles >= 2) schedule(static)
    for (int g = 0; g < n_tiles; ++g) {
        const int t0 = g * 8;
        const int n_tok = std::min(8, n_tokens - t0);
        const float* x_tile = X + static_cast<size_t>(t0) * static_cast<size_t>(stride_x);
        float* dst = packed_buf.data() + static_cast<size_t>(g) * tile_stride;
        if (n_tok <= 4) pack_f32_tile_ptr<4>(x_tile, stride_x, cols, n_tok, dst);
        else            pack_f32_tile_ptr<8>(x_tile, stride_x, cols, n_tok, dst);
    }

    #pragma omp parallel for if(parallel && rows >= 32) schedule(static)
    for (int rt = 0; rt < num_rtiles; ++rt) {
        const int r0 = rt * row_tile;
        const int r1 = std::min(rows, r0 + row_tile);
        for (int g = 0; g < n_tiles; ++g) {
            const int t0 = g * 8;
            const int n_tok = std::min(8, n_tokens - t0);
            const float* xpack = packed_buf.data() + static_cast<size_t>(g) * tile_stride;

            if (n_tok <= 4) {
                for (int r = r0; r < r1; ++r) {
                    float res[4];
                    dot4(wdata, row_offset + static_cast<size_t>(r), xpack, cols, res);
                    for (int t = 0; t < n_tok; ++t) {
                        Y[static_cast<size_t>(t0 + t) * static_cast<size_t>(stride_y) + static_cast<size_t>(r)] = res[t];
                    }
                }
            } else {
                for (int r = r0; r < r1; ++r) {
                    float res[8];
                    dot8(wdata, row_offset + static_cast<size_t>(r), xpack, cols, res);
                    for (int t = 0; t < n_tok; ++t) {
                        Y[static_cast<size_t>(t0 + t) * static_cast<size_t>(stride_y) + static_cast<size_t>(r)] = res[t];
                    }
                }
            }
        }
    }
}

void Avx2Math::gemm_slice_f32(float* Y, int stride_y, const TensorDesc& weight, size_t row_offset,
                              const float* X, int stride_x, int n_tokens, int rows, int cols, bool parallel) {
    if (!Y || !weight.data || !X || n_tokens <= 0 || rows <= 0 || cols <= 0) return;
    const size_t rb = row_bytes(weight.type, cols);
    if (weight.size_bytes > 0) {
        if (rb == 0 || row_offset + static_cast<size_t>(rows) > weight.size_bytes / rb) {
            for (int t = 0; t < n_tokens; ++t) {
                std::memset(Y + static_cast<size_t>(t) * static_cast<size_t>(stride_y), 0, static_cast<size_t>(rows) * sizeof(float));
            }
            return;
        }
    }

    if (n_tokens == 1) {
        if (parallel) {
            gemv_slice_f32(Y, weight, row_offset, X, rows, cols);
        } else {
            for (int r = 0; r < rows; ++r) Y[r] = dot_row(weight, row_offset + static_cast<size_t>(r), X, cols);
        }
        return;
    }

    // Float weights already run exact FP32 activations in register-tiled 1x4/1x8 GEMM!
    if (weight.type == QuantType::F32) {
        run_gemm_float_tiled<QuantType::F32>(Y, stride_y, weight.data, row_offset, X, stride_x, n_tokens, rows, cols, parallel);
        return;
    }
    if (weight.type == QuantType::BF16) {
        run_gemm_float_tiled<QuantType::BF16>(Y, stride_y, weight.data, row_offset, X, stride_x, n_tokens, rows, cols, parallel);
        return;
    }
    if (weight.type == QuantType::F16) {
        run_gemm_float_tiled<QuantType::F16>(Y, stride_y, weight.data, row_offset, X, stride_x, n_tokens, rows, cols, parallel);
        return;
    }

    // L1-packed register-tiled 1x4 / 1x8 exact FP32-activation quantized GEMM (all 10 formats, zero Q8 error!)
    if ((cols % 256) == 0 && cols >= 256) {
        if (weight.type == QuantType::IQ4_XS) { run_gemm_quant_f32_tiled(Y, stride_y, weight.data, row_offset, X, stride_x, n_tokens, rows, cols, rb, parallel, dot_iq4_xs_f32_xN<4>, dot_iq4_xs_f32_xN<8>); return; }
        if (weight.type == QuantType::Q4_K)   { run_gemm_quant_f32_tiled(Y, stride_y, weight.data, row_offset, X, stride_x, n_tokens, rows, cols, rb, parallel, dot_q4_K_f32_xN<4>,   dot_q4_K_f32_xN<8>);   return; }
        if (weight.type == QuantType::Q5_K)   { run_gemm_quant_f32_tiled(Y, stride_y, weight.data, row_offset, X, stride_x, n_tokens, rows, cols, rb, parallel, dot_q5_K_f32_xN<4>,   dot_q5_K_f32_xN<8>);   return; }
        if (weight.type == QuantType::Q6_K)   { run_gemm_quant_f32_tiled(Y, stride_y, weight.data, row_offset, X, stride_x, n_tokens, rows, cols, rb, parallel, dot_q6_K_f32_xN<4>,   dot_q6_K_f32_xN<8>);   return; }
        if (weight.type == QuantType::Q3_K)   { run_gemm_quant_f32_tiled(Y, stride_y, weight.data, row_offset, X, stride_x, n_tokens, rows, cols, rb, parallel, dot_q3_K_f32_xN<4>,   dot_q3_K_f32_xN<8>);   return; }
    }
    if ((cols % 32) == 0 && cols >= 32) {
        if (weight.type == QuantType::IQ4_NL) { run_gemm_quant_f32_tiled(Y, stride_y, weight.data, row_offset, X, stride_x, n_tokens, rows, cols, rb, parallel, dot_iq4_nl_f32_xN<4>, dot_iq4_nl_f32_xN<8>); return; }
        if (weight.type == QuantType::Q5_1)   { run_gemm_quant_f32_tiled(Y, stride_y, weight.data, row_offset, X, stride_x, n_tokens, rows, cols, rb, parallel, dot_q5_1_f32_xN<4>,   dot_q5_1_f32_xN<8>);   return; }
        if (weight.type == QuantType::Q5_0)   { run_gemm_quant_f32_tiled(Y, stride_y, weight.data, row_offset, X, stride_x, n_tokens, rows, cols, rb, parallel, dot_q5_0_f32_xN<4>,   dot_q5_0_f32_xN<8>);   return; }
        if (weight.type == QuantType::Q8_0)   { run_gemm_quant_f32_tiled(Y, stride_y, weight.data, row_offset, X, stride_x, n_tokens, rows, cols, rb, parallel, dot_q8_0_f32_xN<4>,   dot_q8_0_f32_xN<8>);   return; }
        if (weight.type == QuantType::Q4_0)   { run_gemm_quant_f32_tiled(Y, stride_y, weight.data, row_offset, X, stride_x, n_tokens, rows, cols, rb, parallel, dot_q4_0_f32_xN<4>,   dot_q4_0_f32_xN<8>);   return; }
    }

    // Fallback for remaining rare formats (e.g. Q3_K, Q5_0, Q5_1)
    #pragma omp parallel for if(parallel && rows >= 16) schedule(static)
    for (int r = 0; r < rows; ++r) {
        const size_t wrow = row_offset + static_cast<size_t>(r);
        for (int t = 0; t < n_tokens; ++t) {
            const float* xt = X + static_cast<size_t>(t) * static_cast<size_t>(stride_x);
            Y[static_cast<size_t>(t) * static_cast<size_t>(stride_y) + static_cast<size_t>(r)] = dot_row(weight, wrow, xt, cols);
        }
    }
}

void Avx2Math::gemm_f32(float* Y, int stride_y, const TensorDesc& weight,
                        const float* X, int stride_x, int n_tokens, int rows, int cols) {
    gemm_slice_f32(Y, stride_y, weight, 0, X, stride_x, n_tokens, rows, cols, true);
}

// AVX2 In-Place Orthonormal Block Fast Walsh-Hadamard Transform (self-inverse: H_B^2 = I_B)
void Avx2Math::fwht_block_inplace(float* x, int n, int block_size) {
    if (!x || n <= 0 || block_size < 8 || (block_size & (block_size - 1)) != 0 || (n % block_size) != 0) {
        return;
    }
    const __m256 inv_sqrt_b = _mm256_set1_ps(static_cast<float>(1.0 / std::sqrt(static_cast<double>(block_size))));
    const int n_blocks = n / block_size;

    for (int blk = 0; blk < n_blocks; ++blk) {
        float* base = x + static_cast<size_t>(blk) * static_cast<size_t>(block_size);

        // Intra-register AVX2 butterfly stages: h = 1, 2, 4 (all 8 lanes in registers)
        for (int i = 0; i < block_size; i += 8) {
            __m256 v = _mm256_loadu_ps(base + i);
            // h = 1: pairs (0,1), (2,3), (4,5), (6,7)
            __m256 sw1 = _mm256_permute_ps(v, 0xB1);
            v = _mm256_blend_ps(_mm256_add_ps(v, sw1), _mm256_sub_ps(sw1, v), 0xAA);
            // h = 2: pairs (0,2), (1,3), (4,6), (5,7)
            __m256 sw2 = _mm256_permute_ps(v, 0x4E);
            v = _mm256_blend_ps(_mm256_add_ps(v, sw2), _mm256_sub_ps(sw2, v), 0xCC);
            // h = 4: pairs (0..3, 4..7)
            __m256 sw4 = _mm256_permute2f128_ps(v, v, 0x01);
            v = _mm256_blend_ps(_mm256_add_ps(v, sw4), _mm256_sub_ps(sw4, v), 0xF0);
            if (block_size == 8) {
                v = _mm256_mul_ps(v, inv_sqrt_b);
            }
            _mm256_storeu_ps(base + i, v);
        }

        // Inter-vector AVX2 butterfly stages: h = 8, 16, ..., block_size / 2 (fusing final 1/sqrt(B) scale!)
        for (int h = 8; h < block_size; h <<= 1) {
            const int step = h << 1;
            const bool is_last = (step == block_size);
            if (is_last) {
                for (int i = 0; i < block_size; i += step) {
                    float* p0 = base + i;
                    float* p1 = base + i + h;
                    for (int j = 0; j < h; j += 8) {
                        const __m256 a = _mm256_loadu_ps(p0 + j);
                        const __m256 b = _mm256_loadu_ps(p1 + j);
                        _mm256_storeu_ps(p0 + j, _mm256_mul_ps(_mm256_add_ps(a, b), inv_sqrt_b));
                        _mm256_storeu_ps(p1 + j, _mm256_mul_ps(_mm256_sub_ps(a, b), inv_sqrt_b));
                    }
                }
            } else {
                for (int i = 0; i < block_size; i += step) {
                    float* p0 = base + i;
                    float* p1 = base + i + h;
                    for (int j = 0; j < h; j += 8) {
                        const __m256 a = _mm256_loadu_ps(p0 + j);
                        const __m256 b = _mm256_loadu_ps(p1 + j);
                        _mm256_storeu_ps(p0 + j, _mm256_add_ps(a, b));
                        _mm256_storeu_ps(p1 + j, _mm256_sub_ps(a, b));
                    }
                }
            }
        }
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
    } else if (bias_tensor.type == QuantType::BF16) {
        const uint16_t* b = reinterpret_cast<const uint16_t*>(bias_tensor.data);
        for (int i = 0; i < size; ++i) {
            uint32_t u = static_cast<uint32_t>(b[i]) << 16;
            float f = 0.0f;
            std::memcpy(&f, &u, sizeof(f));
            out[i] += f;
        }
    }
}

// Bare-metal AVX2 FP64-accumulated Vector Norm: sqrt(sum(x^2))
float Avx2Math::vector_norm(const float* x, int size) {
    if (!x || size <= 0) return 0.0f;
    __m256d acc0 = _mm256_setzero_pd();
    __m256d acc1 = _mm256_setzero_pd();
    int i = 0;
    for (; i <= size - 8; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        __m256d lo = _mm256_cvtps_pd(_mm256_castps256_ps128(vx));
        __m256d hi = _mm256_cvtps_pd(_mm256_extractf128_ps(vx, 1));
        acc0 = _mm256_fmadd_pd(lo, lo, acc0);
        acc1 = _mm256_fmadd_pd(hi, hi, acc1);
    }
    double buf[4];
    _mm256_storeu_pd(buf, _mm256_add_pd(acc0, acc1));
    double sum_sq = (buf[0] + buf[1]) + (buf[2] + buf[3]);
    for (; i < size; ++i) {
        const double xi = static_cast<double>(x[i]);
        sum_sq += xi * xi;
    }
    return static_cast<float>(std::sqrt(sum_sq));
}

// Bare-metal AVX2 FP64-accumulated SIMD Cosine Similarity: (a . b) / (||a|| * ||b||)
float Avx2Math::cosine_similarity(const float* a, const float* b, int size) {
    if (!a || !b || size <= 0) return 0.0f;
    __m256d d0 = _mm256_setzero_pd(), d1 = _mm256_setzero_pd();
    __m256d na0 = _mm256_setzero_pd(), na1 = _mm256_setzero_pd();
    __m256d nb0 = _mm256_setzero_pd(), nb1 = _mm256_setzero_pd();
    int i = 0;
    for (; i <= size - 8; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256d a_lo = _mm256_cvtps_pd(_mm256_castps256_ps128(va));
        __m256d a_hi = _mm256_cvtps_pd(_mm256_extractf128_ps(va, 1));
        __m256d b_lo = _mm256_cvtps_pd(_mm256_castps256_ps128(vb));
        __m256d b_hi = _mm256_cvtps_pd(_mm256_extractf128_ps(vb, 1));
        d0  = _mm256_fmadd_pd(a_lo, b_lo, d0);
        d1  = _mm256_fmadd_pd(a_hi, b_hi, d1);
        na0 = _mm256_fmadd_pd(a_lo, a_lo, na0);
        na1 = _mm256_fmadd_pd(a_hi, a_hi, na1);
        nb0 = _mm256_fmadd_pd(b_lo, b_lo, nb0);
        nb1 = _mm256_fmadd_pd(b_hi, b_hi, nb1);
    }
    auto hsum4 = [](__m256d v) -> double {
        double tmp[4];
        _mm256_storeu_pd(tmp, v);
        return (tmp[0] + tmp[1]) + (tmp[2] + tmp[3]);
    };
    double dot = hsum4(_mm256_add_pd(d0, d1));
    double na  = hsum4(_mm256_add_pd(na0, na1));
    double nb  = hsum4(_mm256_add_pd(nb0, nb1));
    for (; i < size; ++i) {
        const double xa = static_cast<double>(a[i]);
        const double xb = static_cast<double>(b[i]);
        dot += xa * xb;
        na  += xa * xa;
        nb  += xb * xb;
    }
    const double denom = std::sqrt(na) * std::sqrt(nb);
    if (denom <= 1e-18) return 0.0f;
    return static_cast<float>(dot / denom);
}

} // namespace haven