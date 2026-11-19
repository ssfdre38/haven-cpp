#include "haven/haven_avx2.h"
#include <algorithm>
#include <cstring>
#include <iostream>

namespace haven {

// Fast AVX2 + FMA Q8_0 dot product
float Avx2Math::vec_dot_q8_0(int n, const block_q8_0* vx, const float* vy) {
    const int nb = n / 32;
    float sumf = 0.0f;
    __m256 acc = _mm256_setzero_ps();

    for (int i = 0; i < nb; ++i) {
        const float d = fp16_to_fp32(vx[i].d);
        __m256 d_vec = _mm256_set1_ps(d);

        for (int k = 0; k < 4; ++k) {
            __m256 vy_k = _mm256_loadu_ps(vy + i * 32 + k * 8);
            __m256 vx_k = _mm256_set_ps(
                (float)vx[i].qs[k * 8 + 7],
                (float)vx[i].qs[k * 8 + 6],
                (float)vx[i].qs[k * 8 + 5],
                (float)vx[i].qs[k * 8 + 4],
                (float)vx[i].qs[k * 8 + 3],
                (float)vx[i].qs[k * 8 + 2],
                (float)vx[i].qs[k * 8 + 1],
                (float)vx[i].qs[k * 8 + 0]
            );
            acc = _mm256_fmadd_ps(_mm256_mul_ps(vx_k, d_vec), vy_k, acc);
        }
    }

    float buffer[8];
    _mm256_storeu_ps(buffer, acc);
    for (int j = 0; j < 8; ++j) sumf += buffer[j];

    return sumf;
}

// AVX2 RMS Normalization: y = (x / RMS(x)) * weight
void Avx2Math::rms_norm(float* out, const float* x, const float* weight, int size, float eps) {
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
    if (rope_dim <= 0 || rope_dim > head_dim) rope_dim = head_dim / 2;
    const int num_pairs = rope_dim / 2;

    for (int h = 0; h < num_heads; ++h) {
        float* q_head = q + h * head_dim;
        for (int i = 0; i < num_pairs; ++i) {
            float theta = (float)pos * std::pow(freq_base, -2.0f * (float)i / (float)rope_dim) * freq_scale;
            if (freq_factors && freq_factors[i] > 0.0f) {
                theta /= freq_factors[i];
            }
            float cos_th = std::cos(theta);
            float sin_th = std::sin(theta);

            float q0 = q_head[i];
            float q1 = q_head[i + num_pairs];
            q_head[i]             = q0 * cos_th - q1 * sin_th;
            q_head[i + num_pairs] = q0 * sin_th + q1 * cos_th;
        }
    }

    if (k && num_kv_heads > 0) {
        for (int h = 0; h < num_kv_heads; ++h) {
            float* k_head = k + h * head_dim;
            for (int i = 0; i < num_pairs; ++i) {
                float theta = (float)pos * std::pow(freq_base, -2.0f * (float)i / (float)rope_dim) * freq_scale;
                if (freq_factors && freq_factors[i] > 0.0f) {
                    theta /= freq_factors[i];
                }
                float cos_th = std::cos(theta);
                float sin_th = std::sin(theta);

                float k0 = k_head[i];
                float k1 = k_head[i + num_pairs];
                k_head[i]             = k0 * cos_th - k1 * sin_th;
                k_head[i + num_pairs] = k0 * sin_th + k1 * cos_th;
            }
        }
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

                __m256 vq1 = _mm256_set_ps(
                    (float)(q[offset + 7] & 0x0F),
                    (float)(q[offset + 6] & 0x0F),
                    (float)(q[offset + 5] & 0x0F),
                    (float)(q[offset + 4] & 0x0F),
                    (float)(q[offset + 3] & 0x0F),
                    (float)(q[offset + 2] & 0x0F),
                    (float)(q[offset + 1] & 0x0F),
                    (float)(q[offset + 0] & 0x0F)
                );
                acc_q1 = _mm256_fmadd_ps(vq1, vy1, acc_q1);

                __m256 vq2 = _mm256_set_ps(
                    (float)(q[offset + 7] >> 4),
                    (float)(q[offset + 6] >> 4),
                    (float)(q[offset + 5] >> 4),
                    (float)(q[offset + 4] >> 4),
                    (float)(q[offset + 3] >> 4),
                    (float)(q[offset + 2] >> 4),
                    (float)(q[offset + 1] >> 4),
                    (float)(q[offset + 0] >> 4)
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
            q += 32;
        }
    }
    return sumf;
}

// Fast AVX2 + FMA Q6_K dot product (256 values per super-block)
float Avx2Math::vec_dot_q6_K(int n, const block_q6_K* vx, const float* vy) {
    const int nb = n / 256;
    float sumf = 0.0f;

    for (int i = 0; i < nb; ++i) {
        const float d = fp16_to_fp32(vx[i].d);
        const uint8_t* ql = vx[i].ql;
        const uint8_t* qh = vx[i].qh;
        const int8_t* sc = vx[i].scales;
        const float* y = vy + i * 256;

        float block_sum = 0.0f;
        for (int n_chunk = 0; n_chunk < 256; n_chunk += 128) {
            for (int k = 0; k < 4; ++k) {
                int is = (k < 2) ? 0 : 1;
                const int l_base = k * 8;

                const float scale1 = d * (float)sc[is + 0];
                const float scale2 = d * (float)sc[is + 2];
                const float scale3 = d * (float)sc[is + 4];
                const float scale4 = d * (float)sc[is + 6];

                __m256 vy1 = _mm256_loadu_ps(y + n_chunk + l_base + 0);
                __m256 vy2 = _mm256_loadu_ps(y + n_chunk + l_base + 32);
                __m256 vy3 = _mm256_loadu_ps(y + n_chunk + l_base + 64);
                __m256 vy4 = _mm256_loadu_ps(y + n_chunk + l_base + 96);

                __m256 vq1 = _mm256_set_ps(
                    (float)(int8_t)((ql[l_base + 7] & 0x0F) | (((qh[l_base + 7] >> 0) & 3) << 4)) - 32.0f,
                    (float)(int8_t)((ql[l_base + 6] & 0x0F) | (((qh[l_base + 6] >> 0) & 3) << 4)) - 32.0f,
                    (float)(int8_t)((ql[l_base + 5] & 0x0F) | (((qh[l_base + 5] >> 0) & 3) << 4)) - 32.0f,
                    (float)(int8_t)((ql[l_base + 4] & 0x0F) | (((qh[l_base + 4] >> 0) & 3) << 4)) - 32.0f,
                    (float)(int8_t)((ql[l_base + 3] & 0x0F) | (((qh[l_base + 3] >> 0) & 3) << 4)) - 32.0f,
                    (float)(int8_t)((ql[l_base + 2] & 0x0F) | (((qh[l_base + 2] >> 0) & 3) << 4)) - 32.0f,
                    (float)(int8_t)((ql[l_base + 1] & 0x0F) | (((qh[l_base + 1] >> 0) & 3) << 4)) - 32.0f,
                    (float)(int8_t)((ql[l_base + 0] & 0x0F) | (((qh[l_base + 0] >> 0) & 3) << 4)) - 32.0f
                );
                __m256 vq2 = _mm256_set_ps(
                    (float)(int8_t)((ql[l_base + 39] & 0x0F) | (((qh[l_base + 7] >> 2) & 3) << 4)) - 32.0f,
                    (float)(int8_t)((ql[l_base + 38] & 0x0F) | (((qh[l_base + 6] >> 2) & 3) << 4)) - 32.0f,
                    (float)(int8_t)((ql[l_base + 37] & 0x0F) | (((qh[l_base + 5] >> 2) & 3) << 4)) - 32.0f,
                    (float)(int8_t)((ql[l_base + 36] & 0x0F) | (((qh[l_base + 4] >> 2) & 3) << 4)) - 32.0f,
                    (float)(int8_t)((ql[l_base + 35] & 0x0F) | (((qh[l_base + 3] >> 2) & 3) << 4)) - 32.0f,
                    (float)(int8_t)((ql[l_base + 34] & 0x0F) | (((qh[l_base + 2] >> 2) & 3) << 4)) - 32.0f,
                    (float)(int8_t)((ql[l_base + 33] & 0x0F) | (((qh[l_base + 1] >> 2) & 3) << 4)) - 32.0f,
                    (float)(int8_t)((ql[l_base + 32] & 0x0F) | (((qh[l_base + 0] >> 2) & 3) << 4)) - 32.0f
                );
                __m256 vq3 = _mm256_set_ps(
                    (float)(int8_t)((ql[l_base + 7] >> 4) | (((qh[l_base + 7] >> 4) & 3) << 4)) - 32.0f,
                    (float)(int8_t)((ql[l_base + 6] >> 4) | (((qh[l_base + 6] >> 4) & 3) << 4)) - 32.0f,
                    (float)(int8_t)((ql[l_base + 5] >> 4) | (((qh[l_base + 5] >> 4) & 3) << 4)) - 32.0f,
                    (float)(int8_t)((ql[l_base + 4] >> 4) | (((qh[l_base + 4] >> 4) & 3) << 4)) - 32.0f,
                    (float)(int8_t)((ql[l_base + 3] >> 4) | (((qh[l_base + 3] >> 4) & 3) << 4)) - 32.0f,
                    (float)(int8_t)((ql[l_base + 2] >> 4) | (((qh[l_base + 2] >> 4) & 3) << 4)) - 32.0f,
                    (float)(int8_t)((ql[l_base + 1] >> 4) | (((qh[l_base + 1] >> 4) & 3) << 4)) - 32.0f,
                    (float)(int8_t)((ql[l_base + 0] >> 4) | (((qh[l_base + 0] >> 4) & 3) << 4)) - 32.0f
                );
                __m256 vq4 = _mm256_set_ps(
                    (float)(int8_t)((ql[l_base + 39] >> 4) | (((qh[l_base + 7] >> 6) & 3) << 4)) - 32.0f,
                    (float)(int8_t)((ql[l_base + 38] >> 4) | (((qh[l_base + 6] >> 6) & 3) << 4)) - 32.0f,
                    (float)(int8_t)((ql[l_base + 37] >> 4) | (((qh[l_base + 5] >> 6) & 3) << 4)) - 32.0f,
                    (float)(int8_t)((ql[l_base + 36] >> 4) | (((qh[l_base + 4] >> 6) & 3) << 4)) - 32.0f,
                    (float)(int8_t)((ql[l_base + 35] >> 4) | (((qh[l_base + 3] >> 6) & 3) << 4)) - 32.0f,
                    (float)(int8_t)((ql[l_base + 34] >> 4) | (((qh[l_base + 2] >> 6) & 3) << 4)) - 32.0f,
                    (float)(int8_t)((ql[l_base + 33] >> 4) | (((qh[l_base + 1] >> 6) & 3) << 4)) - 32.0f,
                    (float)(int8_t)((ql[l_base + 32] >> 4) | (((qh[l_base + 0] >> 6) & 3) << 4)) - 32.0f
                );

                __m256 p1 = _mm256_mul_ps(_mm256_mul_ps(vq1, vy1), _mm256_set1_ps(scale1));
                __m256 p2 = _mm256_mul_ps(_mm256_mul_ps(vq2, vy2), _mm256_set1_ps(scale2));
                __m256 p3 = _mm256_mul_ps(_mm256_mul_ps(vq3, vy3), _mm256_set1_ps(scale3));
                __m256 p4 = _mm256_mul_ps(_mm256_mul_ps(vq4, vy4), _mm256_set1_ps(scale4));

                __m256 p_sum = _mm256_add_ps(_mm256_add_ps(p1, p2), _mm256_add_ps(p3, p4));
                float buf[8];
                _mm256_storeu_ps(buf, p_sum);
                for (int b = 0; b < 8; ++b) block_sum += buf[b];
            }
            ql += 64;
            qh += 32;
            sc += 8;
        }
        sumf += block_sum;
    }
    return sumf;
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

// Dot product of FP16 weights with FP32 vector
float Avx2Math::vec_dot_f16(int n, const uint16_t* vx, const float* vy) {
    float sum = 0.0f;
    __m256 acc = _mm256_setzero_ps();
    int i = 0;
    for (; i <= n - 8; i += 8) {
        __m256 vy_vec = _mm256_loadu_ps(vy + i);
        __m256 vx_vec = _mm256_set_ps(
            fp16_to_fp32(vx[i + 7]),
            fp16_to_fp32(vx[i + 6]),
            fp16_to_fp32(vx[i + 5]),
            fp16_to_fp32(vx[i + 4]),
            fp16_to_fp32(vx[i + 3]),
            fp16_to_fp32(vx[i + 2]),
            fp16_to_fp32(vx[i + 1]),
            fp16_to_fp32(vx[i + 0])
        );
        acc = _mm256_fmadd_ps(vx_vec, vy_vec, acc);
    }
    float buffer[8];
    _mm256_storeu_ps(buffer, acc);
    for (int j = 0; j < 8; ++j) sum += buffer[j];
    for (; i < n; ++i) {
        sum += fp16_to_fp32(vx[i]) * vy[i];
    }
    return sum;
}

// Dot product of BF16 weights with FP32 vector
float Avx2Math::vec_dot_bf16(int n, const uint16_t* vx, const float* vy) {
    float sum = 0.0f;
    __m256 acc = _mm256_setzero_ps();
    int i = 0;
    for (; i <= n - 8; i += 8) {
        __m256 vy_vec = _mm256_loadu_ps(vy + i);
        __m256 vx_vec = _mm256_set_ps(
            bf16_to_fp32(vx[i + 7]),
            bf16_to_fp32(vx[i + 6]),
            bf16_to_fp32(vx[i + 5]),
            bf16_to_fp32(vx[i + 4]),
            bf16_to_fp32(vx[i + 3]),
            bf16_to_fp32(vx[i + 2]),
            bf16_to_fp32(vx[i + 1]),
            bf16_to_fp32(vx[i + 0])
        );
        acc = _mm256_fmadd_ps(vx_vec, vy_vec, acc);
    }
    float buffer[8];
    _mm256_storeu_ps(buffer, acc);
    for (int j = 0; j < 8; ++j) sum += buffer[j];
    for (; i < n; ++i) {
        sum += bf16_to_fp32(vx[i]) * vy[i];
    }
    return sum;
}

// Dot product of FP32 weights with FP32 vector
float Avx2Math::vec_dot_f32(int n, const float* vx, const float* vy) {
    float sum = 0.0f;
    __m256 acc = _mm256_setzero_ps();
    int i = 0;
    for (; i <= n - 8; i += 8) {
        __m256 vx_vec = _mm256_loadu_ps(vx + i);
        __m256 vy_vec = _mm256_loadu_ps(vy + i);
        acc = _mm256_fmadd_ps(vx_vec, vy_vec, acc);
    }
    float buffer[8];
    _mm256_storeu_ps(buffer, acc);
    for (int j = 0; j < 8; ++j) sum += buffer[j];
    for (; i < n; ++i) {
        sum += vx[i] * vy[i];
    }
    return sum;
}

// Fast In-Place GELU for Gemma 4 (GeGLU)
void Avx2Math::gelu(float* x, int size) {
    const float sqrt_2_over_pi = 0.7978845608f;
    const float coef = 0.044715f;
    #pragma omp parallel for schedule(static)
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

// Dequantizes a single row of a tensor into float buffer (e.g. token embedding lookup)
void Avx2Math::dequantize_row(float* out, const TensorDesc& weight, int row, int cols) {
    if (!weight.data || cols <= 0) return;

    if (weight.type == QuantType::Q8_0) {
        const auto* blocks = reinterpret_cast<const block_q8_0*>(weight.data);
        const int blocks_per_row = cols / 32;
        const block_q8_0* row_blocks = blocks + row * blocks_per_row;
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
        const block_q4_0* row_blocks = blocks + row * blocks_per_row;
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
        const block_q4_K* row_blocks = blocks + row * blocks_per_row;
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
        const block_q6_K* row_blocks = blocks + row * blocks_per_row;
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
        const block_q5_0* row_blocks = blocks + row * blocks_per_row;
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
        const block_q5_K* row_blocks = blocks + row * blocks_per_row;
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
        const block_q3_K* row_blocks = blocks + row * blocks_per_row;
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
    else if (weight.type == QuantType::BF16) {
        const auto* w = reinterpret_cast<const uint16_t*>(weight.data) + row * cols;
        for (int c = 0; c < cols; ++c) {
            out[c] = bf16_to_fp32(w[c]);
        }
    }
    else if (weight.type == QuantType::F16) {
        const auto* w = reinterpret_cast<const uint16_t*>(weight.data) + row * cols;
        for (int c = 0; c < cols; ++c) {
            out[c] = fp16_to_fp32(w[c]);
        }
    }
    else if (weight.type == QuantType::F32) {
        const auto* w = reinterpret_cast<const float*>(weight.data) + row * cols;
        std::memcpy(out, w, cols * sizeof(float));
    }
}

// Fast SiLU Activation: x * sigmoid(x)
void Avx2Math::silu(float* x, int size) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < size; ++i) {
        x[i] = x[i] / (1.0f + std::exp(-x[i]));
    }
}

// Matrix-Vector Multiplication: y = W * x
void Avx2Math::gemv(float* y, const TensorDesc& weight, const float* x, int rows, int cols) {
    if (!weight.data || rows <= 0 || cols <= 0) return;

    if (weight.type == QuantType::Q8_0) {
        const auto* blocks = reinterpret_cast<const block_q8_0*>(weight.data);
        const int blocks_per_row = cols / 32;
        #pragma omp parallel for schedule(static)
        for (int r = 0; r < rows; ++r) {
            y[r] = vec_dot_q8_0(cols, blocks + r * blocks_per_row, x);
        }
    } else if (weight.type == QuantType::Q4_0) {
        const auto* blocks = reinterpret_cast<const block_q4_0*>(weight.data);
        const int blocks_per_row = cols / 32;
        #pragma omp parallel for schedule(static)
        for (int r = 0; r < rows; ++r) {
            y[r] = vec_dot_q4_0(cols, blocks + r * blocks_per_row, x);
        }
    } else if (weight.type == QuantType::Q4_K) {
        const auto* blocks = reinterpret_cast<const block_q4_K*>(weight.data);
        const int blocks_per_row = cols / 256;
        #pragma omp parallel for schedule(static)
        for (int r = 0; r < rows; ++r) {
            y[r] = vec_dot_q4_K(cols, blocks + r * blocks_per_row, x);
        }
    } else if (weight.type == QuantType::Q6_K) {
        const auto* blocks = reinterpret_cast<const block_q6_K*>(weight.data);
        const int blocks_per_row = cols / 256;
        #pragma omp parallel for schedule(static)
        for (int r = 0; r < rows; ++r) {
            y[r] = vec_dot_q6_K(cols, blocks + r * blocks_per_row, x);
        }
    } else if (weight.type == QuantType::Q5_0) {
        const auto* blocks = reinterpret_cast<const block_q5_0*>(weight.data);
        const int blocks_per_row = cols / 32;
        #pragma omp parallel for schedule(static)
        for (int r = 0; r < rows; ++r) {
            y[r] = vec_dot_q5_0(cols, blocks + r * blocks_per_row, x);
        }
    } else if (weight.type == QuantType::Q5_K) {
        const auto* blocks = reinterpret_cast<const block_q5_K*>(weight.data);
        const int blocks_per_row = cols / 256;
        #pragma omp parallel for schedule(static)
        for (int r = 0; r < rows; ++r) {
            y[r] = vec_dot_q5_K(cols, blocks + r * blocks_per_row, x);
        }
    } else if (weight.type == QuantType::Q3_K) {
        const auto* blocks = reinterpret_cast<const block_q3_K*>(weight.data);
        const int blocks_per_row = cols / 256;
        #pragma omp parallel for schedule(static)
        for (int r = 0; r < rows; ++r) {
            y[r] = vec_dot_q3_K(cols, blocks + r * blocks_per_row, x);
        }
    } else if (weight.type == QuantType::BF16) {
        const auto* w = reinterpret_cast<const uint16_t*>(weight.data);
        #pragma omp parallel for schedule(static)
        for (int r = 0; r < rows; ++r) {
            y[r] = vec_dot_bf16(cols, w + r * cols, x);
        }
    } else if (weight.type == QuantType::F16) {
        const auto* w = reinterpret_cast<const uint16_t*>(weight.data);
        #pragma omp parallel for schedule(static)
        for (int r = 0; r < rows; ++r) {
            y[r] = vec_dot_f16(cols, w + r * cols, x);
        }
    } else if (weight.type == QuantType::F32) {
        const auto* w = reinterpret_cast<const float*>(weight.data);
        #pragma omp parallel for schedule(static)
        for (int r = 0; r < rows; ++r) {
            y[r] = vec_dot_f32(cols, w + r * cols, x);
        }
    }
}

// Fast AVX2 In-Place Bias Addition: out[i] += bias[i] (supports F32 and F16)
void Avx2Math::add_bias(float* out, const TensorDesc& bias_tensor, int size) {
    if (!out || !bias_tensor.data || size <= 0) return;

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