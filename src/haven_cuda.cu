#include "haven/haven_cuda.h"
#include "haven/haven_avx2.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <sstream>
#include <vector>

namespace haven {

namespace {

// ============================================================================
// Device Constants & Bit-Exact Format Helpers (100% parity with Avx2Math!)
// ============================================================================

__constant__ int8_t d_kvalues_iq4nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
};

__device__ __forceinline__ float d_fp16_to_f32(uint16_t h) {
    __half hh;
    memcpy(&hh, &h, sizeof(uint16_t));
    return __half2float(hh);
}

__device__ __forceinline__ float d_bf16_to_f32(uint16_t b) {
    uint32_t u = static_cast<uint32_t>(b) << 16;
    return __uint_as_float(u);
}

__device__ __forceinline__ void d_get_scale_min_k4(int j, const uint8_t* q, uint8_t& d, uint8_t& m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

__device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xFFFFFFFFu, v, offset);
    }
    return v;
}

__device__ __forceinline__ float warp_reduce_max(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v = fmaxf(v, __shfl_down_sync(0xFFFFFFFFu, v, offset));
    }
    return v;
}

__device__ __forceinline__ float d_gelu_tanh(float g) {
    const float cdf = 0.5f * (1.0f + tanhf(0.7978845608f * (g + 0.044715f * g * g * g)));
    return g * cdf;
}

__device__ __forceinline__ float d_silu(float g) {
    return g / (1.0f + expf(-g));
}

// ============================================================================
// Warp-Specialized Quantized Row Dot Product (32 lanes per row, 100% coalesced)
// Returns full warp-reduced dot product in lane 0!
// ============================================================================

__device__ __forceinline__ float warp_dot_iq4_xs(const uint8_t* __restrict__ row_ptr,
                                                 const float* __restrict__ x,
                                                 int K, int lane) {
    const int nb = K >> 8; // K / 256
    const int ib = lane >> 2;  // 0..7 sub-block within 256-elem super-block
    const int t  = lane & 3;   // 0..3 (4 bytes = 8 weights per lane -> 128B coalesced per warp!)
    const int sub_off = (ib << 5) + (t << 2);
    float acc = 0.0f;

    for (int b = 0; b < nb; ++b) {
        const block_iq4_xs* blk = reinterpret_cast<const block_iq4_xs*>(row_ptr + static_cast<size_t>(b) * sizeof(block_iq4_xs));
        const float d = d_fp16_to_f32(blk->d);
        const uint16_t sh = blk->scales_h;
        const uint8_t sl = blk->scales_l[ib >> 1];
        const int ls = ((sl >> (4 * (ib & 1))) & 0x0F) | (((sh >> (2 * ib)) & 3) << 4);
        const float dl = d * static_cast<float>(ls - 32);

        // Coalesced 4-byte load: blk->qs is at offset +8 inside 136B (8B-aligned) struct!
        const uint32_t q4 = *reinterpret_cast<const uint32_t*>(blk->qs + (lane << 2));
        const float* xb = x + (b << 8) + sub_off;
        const float4 xlo = *reinterpret_cast<const float4*>(xb);
        const float4 xhi = *reinterpret_cast<const float4*>(xb + 16);

        float s = 0.0f;
        s = fmaf(static_cast<float>(d_kvalues_iq4nl[(q4 >>  0) & 0x0Fu]), xlo.x, s);
        s = fmaf(static_cast<float>(d_kvalues_iq4nl[(q4 >>  4) & 0x0Fu]), xhi.x, s);
        s = fmaf(static_cast<float>(d_kvalues_iq4nl[(q4 >>  8) & 0x0Fu]), xlo.y, s);
        s = fmaf(static_cast<float>(d_kvalues_iq4nl[(q4 >> 12) & 0x0Fu]), xhi.y, s);
        s = fmaf(static_cast<float>(d_kvalues_iq4nl[(q4 >> 16) & 0x0Fu]), xlo.z, s);
        s = fmaf(static_cast<float>(d_kvalues_iq4nl[(q4 >> 20) & 0x0Fu]), xhi.z, s);
        s = fmaf(static_cast<float>(d_kvalues_iq4nl[(q4 >> 24) & 0x0Fu]), xlo.w, s);
        s = fmaf(static_cast<float>(d_kvalues_iq4nl[(q4 >> 28) & 0x0Fu]), xhi.w, s);
        acc = fmaf(dl, s, acc);
    }
    return warp_reduce_sum(acc);
}

__device__ __forceinline__ float warp_dot_iq4_nl(const uint8_t* __restrict__ row_ptr,
                                                 const float* __restrict__ x,
                                                 int K, int lane) {
    const int nb = K >> 5; // K / 32
    const int b_off = lane >> 4; // 0 or 1 (2 blocks processed concurrently by 32 lanes)
    const int j = lane & 15;     // 0..15 byte index within 16-byte qs
    float acc = 0.0f;
    for (int b = b_off; b < nb; b += 2) {
        const block_iq4_nl* blk = reinterpret_cast<const block_iq4_nl*>(row_ptr + static_cast<size_t>(b) * sizeof(block_iq4_nl));
        const float d = d_fp16_to_f32(blk->d);
        const uint8_t q = blk->qs[j];
        const float* xb = x + (b << 5);
        const float s = static_cast<float>(d_kvalues_iq4nl[q & 0x0Fu]) * xb[j]
                      + static_cast<float>(d_kvalues_iq4nl[q >> 4])    * xb[j + 16];
        acc = fmaf(d, s, acc);
    }
    return warp_reduce_sum(acc);
}

__device__ __forceinline__ float warp_dot_q6_k(const uint8_t* __restrict__ row_ptr,
                                               const float* __restrict__ x,
                                               int K, int lane) {
    const int nb = K >> 8; // K / 256
    const int is = lane >> 4; // 0 or 1
    float acc = 0.0f;
    for (int b = 0; b < nb; ++b) {
        const block_q6_K* blk = reinterpret_cast<const block_q6_K*>(row_ptr + static_cast<size_t>(b) * sizeof(block_q6_K));
        const float d = d_fp16_to_f32(blk->d);
        const float* xb = x + (b << 8);
        #pragma unroll
        for (int c = 0; c < 2; ++c) {
            const uint8_t* ql = blk->ql + (c << 6);
            const uint8_t* qh = blk->qh + (c << 5);
            const int8_t* sc = blk->scales + (c << 3);
            const float* xc = xb + (c << 7);

            const uint8_t ql0 = ql[lane];
            const uint8_t ql1 = ql[lane + 32];
            const uint8_t qhb = qh[lane];

            const float q1 = static_cast<float>(static_cast<int>((ql0 & 0x0Fu) | (((qhb >> 0) & 3u) << 4)) - 32);
            const float q2 = static_cast<float>(static_cast<int>((ql1 & 0x0Fu) | (((qhb >> 2) & 3u) << 4)) - 32);
            const float q3 = static_cast<float>(static_cast<int>((ql0 >> 4)    | (((qhb >> 4) & 3u) << 4)) - 32);
            const float q4 = static_cast<float>(static_cast<int>((ql1 >> 4)    | (((qhb >> 6) & 3u) << 4)) - 32);

            float s = 0.0f;
            s = fmaf(static_cast<float>(sc[is + 0]) * q1, xc[lane +  0], s);
            s = fmaf(static_cast<float>(sc[is + 2]) * q2, xc[lane + 32], s);
            s = fmaf(static_cast<float>(sc[is + 4]) * q3, xc[lane + 64], s);
            s = fmaf(static_cast<float>(sc[is + 6]) * q4, xc[lane + 96], s);
            acc = fmaf(d, s, acc);
        }
    }
    return warp_reduce_sum(acc);
}

__device__ __forceinline__ float warp_dot_q5_k(const uint8_t* __restrict__ row_ptr,
                                               const float* __restrict__ x,
                                               int K, int lane) {
    const int nb = K >> 8;
    float acc = 0.0f;
    for (int b = 0; b < nb; ++b) {
        const block_q5_K* blk = reinterpret_cast<const block_q5_K*>(row_ptr + static_cast<size_t>(b) * sizeof(block_q5_K));
        const float d = d_fp16_to_f32(blk->d);
        const float dmin = d_fp16_to_f32(blk->dmin);
        const uint8_t qhb = blk->qh[lane];
        const float* xb = x + (b << 8);
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            uint8_t sc1, m1, sc2, m2;
            d_get_scale_min_k4(2 * j,     blk->scales, sc1, m1);
            d_get_scale_min_k4(2 * j + 1, blk->scales, sc2, m2);
            const uint8_t ql = blk->qs[(j << 5) + lane];
            const uint8_t u1 = static_cast<uint8_t>(1u << (2 * j));
            const uint8_t u2 = static_cast<uint8_t>(2u << (2 * j));
            const float v1 = (d * static_cast<float>(sc1)) * static_cast<float>((ql & 0x0Fu) + ((qhb & u1) ? 16 : 0))
                           - (dmin * static_cast<float>(m1));
            const float v2 = (d * static_cast<float>(sc2)) * static_cast<float>((ql >> 4)    + ((qhb & u2) ? 16 : 0))
                           - (dmin * static_cast<float>(m2));
            acc = fmaf(v1, xb[(j << 6) + lane],      acc);
            acc = fmaf(v2, xb[(j << 6) + 32 + lane], acc);
        }
    }
    return warp_reduce_sum(acc);
}

__device__ __forceinline__ float warp_dot_q4_k(const uint8_t* __restrict__ row_ptr,
                                               const float* __restrict__ x,
                                               int K, int lane) {
    const int nb = K >> 8;
    float acc = 0.0f;
    for (int b = 0; b < nb; ++b) {
        const block_q4_K* blk = reinterpret_cast<const block_q4_K*>(row_ptr + static_cast<size_t>(b) * sizeof(block_q4_K));
        const float d = d_fp16_to_f32(blk->d);
        const float dmin = d_fp16_to_f32(blk->dmin);
        const float* xb = x + (b << 8);
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            uint8_t sc1, m1, sc2, m2;
            d_get_scale_min_k4(2 * j,     blk->scales, sc1, m1);
            d_get_scale_min_k4(2 * j + 1, blk->scales, sc2, m2);
            const uint8_t ql = blk->qs[(j << 5) + lane];
            const float v1 = (d * static_cast<float>(sc1)) * static_cast<float>(ql & 0x0Fu) - (dmin * static_cast<float>(m1));
            const float v2 = (d * static_cast<float>(sc2)) * static_cast<float>(ql >> 4)    - (dmin * static_cast<float>(m2));
            acc = fmaf(v1, xb[(j << 6) + lane],      acc);
            acc = fmaf(v2, xb[(j << 6) + 32 + lane], acc);
        }
    }
    return warp_reduce_sum(acc);
}

__device__ __forceinline__ float warp_dot_q5_1(const uint8_t* __restrict__ row_ptr,
                                               const float* __restrict__ x,
                                               int K, int lane) {
    const int nb = K >> 5;
    const int b_off = lane >> 4;
    const int j = lane & 15;
    float acc = 0.0f;
    for (int b = b_off; b < nb; b += 2) {
        const block_q5_1* blk = reinterpret_cast<const block_q5_1*>(row_ptr + static_cast<size_t>(b) * sizeof(block_q5_1));
        const float d = d_fp16_to_f32(blk->d);
        const float m = d_fp16_to_f32(blk->m);
        uint32_t qh;
        memcpy(&qh, blk->qh, sizeof(uint32_t));
        const uint8_t q = blk->qs[j];
        const int x0 = (q & 0x0Fu) | (((qh >> j) & 1u) << 4);
        const int x1 = (q >> 4)    | (((qh >> (j + 16)) & 1u) << 4);
        const float* xb = x + (b << 5);
        acc = fmaf(fmaf(d, static_cast<float>(x0), m), xb[j],      acc);
        acc = fmaf(fmaf(d, static_cast<float>(x1), m), xb[j + 16], acc);
    }
    return warp_reduce_sum(acc);
}

__device__ __forceinline__ float warp_dot_q5_0(const uint8_t* __restrict__ row_ptr,
                                               const float* __restrict__ x,
                                               int K, int lane) {
    const int nb = K >> 5;
    const int b_off = lane >> 4;
    const int j = lane & 15;
    float acc = 0.0f;
    for (int b = b_off; b < nb; b += 2) {
        const block_q5_0* blk = reinterpret_cast<const block_q5_0*>(row_ptr + static_cast<size_t>(b) * sizeof(block_q5_0));
        const float d = d_fp16_to_f32(blk->d);
        uint32_t qh;
        memcpy(&qh, blk->qh, sizeof(uint32_t));
        const uint8_t q = blk->qs[j];
        const int x0 = static_cast<int>((q & 0x0Fu) | (((qh >> j) & 1u) << 4)) - 16;
        const int x1 = static_cast<int>((q >> 4)    | (((qh >> (j + 16)) & 1u) << 4)) - 16;
        const float* xb = x + (b << 5);
        acc = fmaf(d * static_cast<float>(x0), xb[j],      acc);
        acc = fmaf(d * static_cast<float>(x1), xb[j + 16], acc);
    }
    return warp_reduce_sum(acc);
}

__device__ __forceinline__ float warp_dot_q8_0(const uint8_t* __restrict__ row_ptr,
                                               const float* __restrict__ x,
                                               int K, int lane) {
    const int nb = K >> 5;
    float acc = 0.0f;
    for (int b = 0; b < nb; ++b) {
        const block_q8_0* blk = reinterpret_cast<const block_q8_0*>(row_ptr + static_cast<size_t>(b) * sizeof(block_q8_0));
        const float d = d_fp16_to_f32(blk->d);
        acc = fmaf(d * static_cast<float>(blk->qs[lane]), x[(b << 5) + lane], acc);
    }
    return warp_reduce_sum(acc);
}

__device__ __forceinline__ float warp_dot_q4_0(const uint8_t* __restrict__ row_ptr,
                                               const float* __restrict__ x,
                                               int K, int lane) {
    const int nb = K >> 5;
    const int b_off = lane >> 4;
    const int j = lane & 15;
    float acc = 0.0f;
    for (int b = b_off; b < nb; b += 2) {
        const block_q4_0* blk = reinterpret_cast<const block_q4_0*>(row_ptr + static_cast<size_t>(b) * sizeof(block_q4_0));
        const float d = d_fp16_to_f32(blk->d);
        const uint8_t q = blk->qs[j];
        const float* xb = x + (b << 5);
        acc = fmaf(d * static_cast<float>(static_cast<int>(q & 0x0Fu) - 8), xb[j * 2 + 0], acc);
        acc = fmaf(d * static_cast<float>(static_cast<int>(q >> 4)    - 8), xb[j * 2 + 1], acc);
    }
    return warp_reduce_sum(acc);
}

__device__ __forceinline__ float warp_dot_q3_k(const uint8_t* __restrict__ row_ptr,
                                               const float* __restrict__ x,
                                               int K, int lane) {
    const int nb = K >> 8;
    float acc = 0.0f;
    for (int b = 0; b < nb; ++b) {
        const block_q3_K* blk = reinterpret_cast<const block_q3_K*>(row_ptr + static_cast<size_t>(b) * sizeof(block_q3_K));
        const float d_all = d_fp16_to_f32(blk->d);
        uint32_t aux[4];
        memcpy(aux, blk->scales, 12);
        const uint32_t kmask1 = 0x03030303u;
        const uint32_t kmask2 = 0x0f0f0f0fu;
        const uint32_t tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        const int8_t* sc = reinterpret_cast<const int8_t*>(aux);
        const float* xb = x + (b << 8);
        const int half = lane >> 4; // 0 for 0..15, 1 for 16..31
        const int is_base = half;
        #pragma unroll
        for (int c = 0; c < 2; ++c) {
            const uint8_t* q = blk->qs + (c << 5);
            const uint8_t qb = q[lane];
            const uint8_t hmb = blk->hmask[lane];
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const uint8_t m = static_cast<uint8_t>(1u << (c * 4 + j));
                const float dl = d_all * static_cast<float>(sc[c * 8 + j * 2 + is_base] - 32);
                const int w = static_cast<int>((qb >> (2 * j)) & 3u) - ((hmb & m) ? 0 : 4);
                acc = fmaf(dl * static_cast<float>(w), xb[(c << 7) + (j << 5) + lane], acc);
            }
        }
    }
    return warp_reduce_sum(acc);
}

__device__ __forceinline__ float warp_dot_bf16(const uint8_t* __restrict__ row_ptr,
                                               const float* __restrict__ x,
                                               int K, int lane) {
    const uint16_t* w = reinterpret_cast<const uint16_t*>(row_ptr);
    float acc = 0.0f;
    for (int i = lane; i < K; i += 32) {
        acc = fmaf(d_bf16_to_f32(w[i]), x[i], acc);
    }
    return warp_reduce_sum(acc);
}

__device__ __forceinline__ float warp_dot_f16(const uint8_t* __restrict__ row_ptr,
                                              const float* __restrict__ x,
                                              int K, int lane) {
    const uint16_t* w = reinterpret_cast<const uint16_t*>(row_ptr);
    float acc = 0.0f;
    for (int i = lane; i < K; i += 32) {
        acc = fmaf(d_fp16_to_f32(w[i]), x[i], acc);
    }
    return warp_reduce_sum(acc);
}

__device__ __forceinline__ float warp_dot_f32(const uint8_t* __restrict__ row_ptr,
                                              const float* __restrict__ x,
                                              int K, int lane) {
    const float* w = reinterpret_cast<const float*>(row_ptr);
    float acc = 0.0f;
    const int K4 = K >> 2;
    const float4* w4 = reinterpret_cast<const float4*>(w);
    const float4* x4 = reinterpret_cast<const float4*>(x);
    for (int i = lane; i < K4; i += 32) {
        float4 a = w4[i];
        float4 b = x4[i];
        acc = fmaf(a.x, b.x, acc);
        acc = fmaf(a.y, b.y, acc);
        acc = fmaf(a.z, b.z, acc);
        acc = fmaf(a.w, b.w, acc);
    }
    for (int i = (K4 << 2) + lane; i < K; i += 32) {
        acc = fmaf(w[i], x[i], acc);
    }
    return warp_reduce_sum(acc);
}

__device__ __forceinline__ float warp_dot_any(const uint8_t* __restrict__ row_ptr,
                                              QuantType qtype,
                                              const float* __restrict__ x,
                                              int K, int lane) {
    switch (qtype) {
        case QuantType::IQ4_XS: return warp_dot_iq4_xs(row_ptr, x, K, lane);
        case QuantType::IQ4_NL: return warp_dot_iq4_nl(row_ptr, x, K, lane);
        case QuantType::Q6_K:   return warp_dot_q6_k(row_ptr, x, K, lane);
        case QuantType::Q5_K:   return warp_dot_q5_k(row_ptr, x, K, lane);
        case QuantType::Q4_K:   return warp_dot_q4_k(row_ptr, x, K, lane);
        case QuantType::Q5_1:   return warp_dot_q5_1(row_ptr, x, K, lane);
        case QuantType::Q5_0:   return warp_dot_q5_0(row_ptr, x, K, lane);
        case QuantType::Q8_0:   return warp_dot_q8_0(row_ptr, x, K, lane);
        case QuantType::Q4_0:   return warp_dot_q4_0(row_ptr, x, K, lane);
        case QuantType::Q3_K:   return warp_dot_q3_k(row_ptr, x, K, lane);
        case QuantType::BF16:   return warp_dot_bf16(row_ptr, x, K, lane);
        case QuantType::F16:    return warp_dot_f16(row_ptr, x, K, lane);
        case QuantType::F32:    return warp_dot_f32(row_ptr, x, K, lane);
        default: return 0.0f;
    }
}

// ============================================================================
// Core GEMV & Fused GEMV Kernels (4 warps = 4 rows per 128-thread block)
// ============================================================================

__global__ void k_gemv_quant(const uint8_t* __restrict__ W, QuantType qtype, size_t row_bytes,
                             const float* __restrict__ x, const float* __restrict__ bias,
                             float* __restrict__ y, int N, int K, float scale_out) {
    const int warp_id = threadIdx.x >> 5;
    const int lane    = threadIdx.x & 31;
    const int row     = (blockIdx.x << 2) + warp_id;
    if (row >= N) return;
    const uint8_t* rp = W + static_cast<size_t>(row) * row_bytes;
    float sum = warp_dot_any(rp, qtype, x, K, lane);
    if (lane == 0) {
        if (bias) sum += bias[row];
        y[row] = sum * scale_out;
    }
}

// Fused Q+K+V GEMV: reads x[D] ONCE for Q, optional K, optional V (saves 2x activation traffic + 2 launches!)
__global__ void k_fused_qkv_gemv(const uint8_t* __restrict__ Wq, QuantType qt_q, size_t rb_q, const float* __restrict__ bq,
                                 const uint8_t* __restrict__ Wk, QuantType qt_k, size_t rb_k, const float* __restrict__ bk,
                                 const uint8_t* __restrict__ Wv, QuantType qt_v, size_t rb_v, const float* __restrict__ bv,
                                 const float* __restrict__ x,
                                 float* __restrict__ q_out, float* __restrict__ k_out, float* __restrict__ v_out,
                                 int q_dim, int k_dim, int v_dim, int D, bool k_eq_v) {
    const int warp_id = threadIdx.x >> 5;
    const int lane    = threadIdx.x & 31;
    const int idx     = (blockIdx.x << 2) + warp_id;
    const int total   = q_dim + k_dim + v_dim;
    if (idx >= total) return;

    if (idx < q_dim) {
        float s = warp_dot_any(Wq + static_cast<size_t>(idx) * rb_q, qt_q, x, D, lane);
        if (lane == 0) q_out[idx] = bq ? (s + bq[idx]) : s;
    } else if (idx < q_dim + k_dim) {
        const int r = idx - q_dim;
        float s = warp_dot_any(Wk + static_cast<size_t>(r) * rb_k, qt_k, x, D, lane);
        if (lane == 0) {
            float val = bk ? (s + bk[r]) : s;
            k_out[r] = val;
            if (k_eq_v && v_out) v_out[r] = val;
        }
    } else {
        const int r = idx - q_dim - k_dim;
        float s = warp_dot_any(Wv + static_cast<size_t>(r) * rb_v, qt_v, x, D, lane);
        if (lane == 0) v_out[r] = bv ? (s + bv[r]) : s;
    }
}

// Fused FFN Gate + Up + GeGLU/SiLU GEMV: reads x[D] ONCE and writes activated h[F] directly!
__global__ void k_fused_ffn_gu_act(const uint8_t* __restrict__ Wg, QuantType qt_g, size_t rb_g,
                                   const uint8_t* __restrict__ Wu, QuantType qt_u, size_t rb_u,
                                   const float* __restrict__ x, float* __restrict__ h_out,
                                   int F, int D, bool use_gelu) {
    const int warp_id = threadIdx.x >> 5;
    const int lane    = threadIdx.x & 31;
    const int row     = (blockIdx.x << 2) + warp_id;
    if (row >= F) return;
    const float g = warp_dot_any(Wg + static_cast<size_t>(row) * rb_g, qt_g, x, D, lane);
    const float u = warp_dot_any(Wu + static_cast<size_t>(row) * rb_u, qt_u, x, D, lane);
    if (lane == 0) {
        const float act = use_gelu ? d_gelu_tanh(g) : d_silu(g);
        h_out[row] = act * u;
    }
}

// ============================================================================
// Dequantize Raw GGUF Tensor Rows -> FP16 Matrix (for cuBLAS Tensor Core Prefill!)
// ============================================================================

__global__ void k_dequant_to_f16(const uint8_t* __restrict__ W, QuantType qtype, size_t row_bytes,
                                 __half* __restrict__ out_f16, int N, int K) {
    const int row = blockIdx.y;
    if (row >= N) return;
    const uint8_t* rp = W + static_cast<size_t>(row) * row_bytes;
    __half* dst = out_f16 + static_cast<size_t>(row) * static_cast<size_t>(K);
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (qtype == QuantType::IQ4_XS) {
        const int nb = K >> 8;
        const int b = tid >> 7;        // 128 threads per 256-elem block (each thread writes 2 elements: lo + hi)
        if (b >= nb) return;
        const int loc = tid & 127;
        const int ib = loc >> 4;       // 0..7
        const int j  = loc & 15;       // 0..15
        const block_iq4_xs* blk = reinterpret_cast<const block_iq4_xs*>(rp + static_cast<size_t>(b) * sizeof(block_iq4_xs));
        const float d = d_fp16_to_f32(blk->d);
        const int ls = ((blk->scales_l[ib >> 1] >> (4 * (ib & 1))) & 0x0F) | (((blk->scales_h >> (2 * ib)) & 3) << 4);
        const float dl = d * static_cast<float>(ls - 32);
        const uint8_t q = blk->qs[(ib << 4) + j];
        dst[(b << 8) + (ib << 5) + j]      = __float2half(dl * static_cast<float>(d_kvalues_iq4nl[q & 0x0Fu]));
        dst[(b << 8) + (ib << 5) + 16 + j] = __float2half(dl * static_cast<float>(d_kvalues_iq4nl[q >> 4]));
    } else if (qtype == QuantType::IQ4_NL) {
        const int nb = K >> 5;
        const int b = tid >> 4;        // 16 threads per 32-elem block
        if (b >= nb) return;
        const int j = tid & 15;
        const block_iq4_nl* blk = reinterpret_cast<const block_iq4_nl*>(rp + static_cast<size_t>(b) * sizeof(block_iq4_nl));
        const float d = d_fp16_to_f32(blk->d);
        const uint8_t q = blk->qs[j];
        dst[(b << 5) + j]      = __float2half(d * static_cast<float>(d_kvalues_iq4nl[q & 0x0Fu]));
        dst[(b << 5) + 16 + j] = __float2half(d * static_cast<float>(d_kvalues_iq4nl[q >> 4]));
    } else if (qtype == QuantType::Q6_K) {
        const int nb = K >> 8;
        const int b = tid >> 6;        // 64 threads per 256-elem block (each thread writes 4 elems)
        if (b >= nb) return;
        const int loc = tid & 63;
        const int c = loc >> 5;        // 0 or 1
        const int l = loc & 31;
        const int is = l >> 4;
        const block_q6_K* blk = reinterpret_cast<const block_q6_K*>(rp + static_cast<size_t>(b) * sizeof(block_q6_K));
        const float d = d_fp16_to_f32(blk->d);
        const uint8_t* ql = blk->ql + (c << 6);
        const uint8_t* qh = blk->qh + (c << 5);
        const int8_t* sc = blk->scales + (c << 3);
        const uint8_t ql0 = ql[l], ql1 = ql[l + 32], qhb = qh[l];
        const int q1 = static_cast<int>((ql0 & 0x0Fu) | (((qhb >> 0) & 3u) << 4)) - 32;
        const int q2 = static_cast<int>((ql1 & 0x0Fu) | (((qhb >> 2) & 3u) << 4)) - 32;
        const int q3 = static_cast<int>((ql0 >> 4)    | (((qhb >> 4) & 3u) << 4)) - 32;
        const int q4 = static_cast<int>((ql1 >> 4)    | (((qhb >> 6) & 3u) << 4)) - 32;
        const int base = (b << 8) + (c << 7) + l;
        dst[base +  0] = __float2half(d * static_cast<float>(sc[is + 0]) * static_cast<float>(q1));
        dst[base + 32] = __float2half(d * static_cast<float>(sc[is + 2]) * static_cast<float>(q2));
        dst[base + 64] = __float2half(d * static_cast<float>(sc[is + 4]) * static_cast<float>(q3));
        dst[base + 96] = __float2half(d * static_cast<float>(sc[is + 6]) * static_cast<float>(q4));
    } else if (qtype == QuantType::Q5_K) {
        const int nb = K >> 8;
        const int b = tid >> 7;        // 128 threads per 256-elem block
        if (b >= nb) return;
        const int loc = tid & 127;
        const int j = loc >> 5;        // 0..3
        const int l = loc & 31;
        const block_q5_K* blk = reinterpret_cast<const block_q5_K*>(rp + static_cast<size_t>(b) * sizeof(block_q5_K));
        const float d = d_fp16_to_f32(blk->d);
        const float dmin = d_fp16_to_f32(blk->dmin);
        uint8_t sc1, m1, sc2, m2;
        d_get_scale_min_k4(2 * j,     blk->scales, sc1, m1);
        d_get_scale_min_k4(2 * j + 1, blk->scales, sc2, m2);
        const uint8_t ql = blk->qs[(j << 5) + l];
        const uint8_t qhb = blk->qh[l];
        const uint8_t u1 = static_cast<uint8_t>(1u << (2 * j));
        const uint8_t u2 = static_cast<uint8_t>(2u << (2 * j));
        const int base = (b << 8) + (j << 6) + l;
        dst[base]      = __float2half((d * static_cast<float>(sc1)) * static_cast<float>((ql & 0x0Fu) + ((qhb & u1) ? 16 : 0)) - dmin * static_cast<float>(m1));
        dst[base + 32] = __float2half((d * static_cast<float>(sc2)) * static_cast<float>((ql >> 4)    + ((qhb & u2) ? 16 : 0)) - dmin * static_cast<float>(m2));
    } else if (qtype == QuantType::Q4_K) {
        const int nb = K >> 8;
        const int b = tid >> 7;
        if (b >= nb) return;
        const int loc = tid & 127;
        const int j = loc >> 5;
        const int l = loc & 31;
        const block_q4_K* blk = reinterpret_cast<const block_q4_K*>(rp + static_cast<size_t>(b) * sizeof(block_q4_K));
        const float d = d_fp16_to_f32(blk->d);
        const float dmin = d_fp16_to_f32(blk->dmin);
        uint8_t sc1, m1, sc2, m2;
        d_get_scale_min_k4(2 * j,     blk->scales, sc1, m1);
        d_get_scale_min_k4(2 * j + 1, blk->scales, sc2, m2);
        const uint8_t ql = blk->qs[(j << 5) + l];
        const int base = (b << 8) + (j << 6) + l;
        dst[base]      = __float2half((d * static_cast<float>(sc1)) * static_cast<float>(ql & 0x0Fu) - dmin * static_cast<float>(m1));
        dst[base + 32] = __float2half((d * static_cast<float>(sc2)) * static_cast<float>(ql >> 4)    - dmin * static_cast<float>(m2));
    } else if (qtype == QuantType::Q5_1) {
        const int nb = K >> 5;
        const int b = tid >> 4;
        if (b >= nb) return;
        const int j = tid & 15;
        const block_q5_1* blk = reinterpret_cast<const block_q5_1*>(rp + static_cast<size_t>(b) * sizeof(block_q5_1));
        const float d = d_fp16_to_f32(blk->d);
        const float m = d_fp16_to_f32(blk->m);
        uint32_t qh;
        memcpy(&qh, blk->qh, sizeof(uint32_t));
        const uint8_t q = blk->qs[j];
        const int x0 = (q & 0x0Fu) | (((qh >> j) & 1u) << 4);
        const int x1 = (q >> 4)    | (((qh >> (j + 16)) & 1u) << 4);
        dst[(b << 5) + j]      = __float2half(fmaf(d, static_cast<float>(x0), m));
        dst[(b << 5) + 16 + j] = __float2half(fmaf(d, static_cast<float>(x1), m));
    } else if (qtype == QuantType::Q5_0) {
        const int nb = K >> 5;
        const int b = tid >> 4;
        if (b >= nb) return;
        const int j = tid & 15;
        const block_q5_0* blk = reinterpret_cast<const block_q5_0*>(rp + static_cast<size_t>(b) * sizeof(block_q5_0));
        const float d = d_fp16_to_f32(blk->d);
        uint32_t qh;
        memcpy(&qh, blk->qh, sizeof(uint32_t));
        const uint8_t q = blk->qs[j];
        const int x0 = static_cast<int>((q & 0x0Fu) | (((qh >> j) & 1u) << 4)) - 16;
        const int x1 = static_cast<int>((q >> 4)    | (((qh >> (j + 16)) & 1u) << 4)) - 16;
        dst[(b << 5) + j]      = __float2half(d * static_cast<float>(x0));
        dst[(b << 5) + 16 + j] = __float2half(d * static_cast<float>(x1));
    } else if (qtype == QuantType::Q8_0) {
        if (tid >= K) return;
        const int b = tid >> 5;
        const int j = tid & 31;
        const block_q8_0* blk = reinterpret_cast<const block_q8_0*>(rp + static_cast<size_t>(b) * sizeof(block_q8_0));
        dst[tid] = __float2half(d_fp16_to_f32(blk->d) * static_cast<float>(blk->qs[j]));
    } else if (qtype == QuantType::Q4_0) {
        const int nb = K >> 5;
        const int b = tid >> 4;
        if (b >= nb) return;
        const int j = tid & 15;
        const block_q4_0* blk = reinterpret_cast<const block_q4_0*>(rp + static_cast<size_t>(b) * sizeof(block_q4_0));
        const float d = d_fp16_to_f32(blk->d);
        const uint8_t q = blk->qs[j];
        dst[(b << 5) + j * 2 + 0] = __float2half(d * static_cast<float>(static_cast<int>(q & 0x0Fu) - 8));
        dst[(b << 5) + j * 2 + 1] = __float2half(d * static_cast<float>(static_cast<int>(q >> 4)    - 8));
    } else if (qtype == QuantType::Q3_K) {
        const int nb = K >> 8;
        const int b = tid >> 5; // 32 threads per 256-elem block
        if (b >= nb) return;
        const int lane = tid & 31;
        const block_q3_K* blk = reinterpret_cast<const block_q3_K*>(rp + static_cast<size_t>(b) * sizeof(block_q3_K));
        const float d_all = d_fp16_to_f32(blk->d);
        uint32_t aux[4];
        memcpy(aux, blk->scales, 12);
        const uint32_t kmask1 = 0x03030303u, kmask2 = 0x0f0f0f0fu, tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        const int8_t* sc = reinterpret_cast<const int8_t*>(aux);
        const int is_base = lane >> 4;
        for (int c = 0; c < 2; ++c) {
            const uint8_t qb = blk->qs[(c << 5) + lane];
            const uint8_t hmb = blk->hmask[lane];
            for (int j = 0; j < 4; ++j) {
                const uint8_t m = static_cast<uint8_t>(1u << (c * 4 + j));
                const float dl = d_all * static_cast<float>(sc[c * 8 + j * 2 + is_base] - 32);
                const int w = static_cast<int>((qb >> (2 * j)) & 3u) - ((hmb & m) ? 0 : 4);
                dst[(b << 8) + (c << 7) + (j << 5) + lane] = __float2half(dl * static_cast<float>(w));
            }
        }
    } else if (qtype == QuantType::BF16) {
        if (tid >= K) return;
        const uint16_t* w = reinterpret_cast<const uint16_t*>(rp);
        dst[tid] = __float2half(d_bf16_to_f32(w[tid]));
    } else if (qtype == QuantType::F16) {
        if (tid >= K) return;
        const __half* w = reinterpret_cast<const __half*>(rp);
        dst[tid] = w[tid];
    } else if (qtype == QuantType::F32) {
        if (tid >= K) return;
        const float* w = reinterpret_cast<const float*>(rp);
        dst[tid] = __float2half(w[tid]);
    }
}

// Direct single-row dequantize to F32 (for embedding lookup & Pleias token table lookup on GPU!)
__global__ void k_dequant_rows_f32(const uint8_t* __restrict__ W, QuantType qtype, size_t row_bytes,
                                   const uint32_t* __restrict__ tok_ids, int S, int vocab_size,
                                   float* __restrict__ out_f32, int K, float scale) {
    const int s = blockIdx.y;
    if (s >= S) return;
    const uint32_t tok = tok_ids[s];
    const uint32_t safe_tok = (tok < static_cast<uint32_t>(vocab_size)) ? tok : 0u;
    const uint8_t* rp = W + static_cast<size_t>(safe_tok) * row_bytes;
    float* dst = out_f32 + static_cast<size_t>(s) * static_cast<size_t>(K);
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (qtype == QuantType::Q6_K) {
        const int nb = K >> 8;
        const int b = tid >> 6;
        if (b >= nb) return;
        const int loc = tid & 63;
        const int c = loc >> 5, l = loc & 31, is = l >> 4;
        const block_q6_K* blk = reinterpret_cast<const block_q6_K*>(rp + static_cast<size_t>(b) * sizeof(block_q6_K));
        const float d = d_fp16_to_f32(blk->d) * scale;
        const uint8_t* ql = blk->ql + (c << 6);
        const uint8_t* qh = blk->qh + (c << 5);
        const int8_t* sc = blk->scales + (c << 3);
        const uint8_t ql0 = ql[l], ql1 = ql[l + 32], qhb = qh[l];
        const int q1 = static_cast<int>((ql0 & 0x0Fu) | (((qhb >> 0) & 3u) << 4)) - 32;
        const int q2 = static_cast<int>((ql1 & 0x0Fu) | (((qhb >> 2) & 3u) << 4)) - 32;
        const int q3 = static_cast<int>((ql0 >> 4)    | (((qhb >> 4) & 3u) << 4)) - 32;
        const int q4 = static_cast<int>((ql1 >> 4)    | (((qhb >> 6) & 3u) << 4)) - 32;
        const int base = (b << 8) + (c << 7) + l;
        dst[base +  0] = d * static_cast<float>(sc[is + 0]) * static_cast<float>(q1);
        dst[base + 32] = d * static_cast<float>(sc[is + 2]) * static_cast<float>(q2);
        dst[base + 64] = d * static_cast<float>(sc[is + 4]) * static_cast<float>(q3);
        dst[base + 96] = d * static_cast<float>(sc[is + 6]) * static_cast<float>(q4);
    } else if (qtype == QuantType::IQ4_XS) {
        const int nb = K >> 8;
        const int b = tid >> 7;
        if (b >= nb) return;
        const int loc = tid & 127;
        const int ib = loc >> 4, j = loc & 15;
        const block_iq4_xs* blk = reinterpret_cast<const block_iq4_xs*>(rp + static_cast<size_t>(b) * sizeof(block_iq4_xs));
        const float d = d_fp16_to_f32(blk->d) * scale;
        const int ls = ((blk->scales_l[ib >> 1] >> (4 * (ib & 1))) & 0x0F) | (((blk->scales_h >> (2 * ib)) & 3) << 4);
        const float dl = d * static_cast<float>(ls - 32);
        const uint8_t q = blk->qs[(ib << 4) + j];
        dst[(b << 8) + (ib << 5) + j]      = dl * static_cast<float>(d_kvalues_iq4nl[q & 0x0Fu]);
        dst[(b << 8) + (ib << 5) + 16 + j] = dl * static_cast<float>(d_kvalues_iq4nl[q >> 4]);
    } else if (qtype == QuantType::Q8_0) {
        if (tid >= K) return;
        const block_q8_0* blk = reinterpret_cast<const block_q8_0*>(rp + static_cast<size_t>(tid >> 5) * sizeof(block_q8_0));
        dst[tid] = d_fp16_to_f32(blk->d) * static_cast<float>(blk->qs[tid & 31]) * scale;
    } else if (qtype == QuantType::Q4_K) {
        const int nb = K >> 8;
        const int b = tid >> 7;
        if (b >= nb) return;
        const int loc = tid & 127, j = loc >> 5, l = loc & 31;
        const block_q4_K* blk = reinterpret_cast<const block_q4_K*>(rp + static_cast<size_t>(b) * sizeof(block_q4_K));
        const float d = d_fp16_to_f32(blk->d), dmin = d_fp16_to_f32(blk->dmin);
        uint8_t sc1, m1, sc2, m2;
        d_get_scale_min_k4(2 * j, blk->scales, sc1, m1);
        d_get_scale_min_k4(2 * j + 1, blk->scales, sc2, m2);
        const uint8_t ql = blk->qs[(j << 5) + l];
        const int base = (b << 8) + (j << 6) + l;
        dst[base]      = ((d * static_cast<float>(sc1)) * static_cast<float>(ql & 0x0Fu) - dmin * static_cast<float>(m1)) * scale;
        dst[base + 32] = ((d * static_cast<float>(sc2)) * static_cast<float>(ql >> 4)    - dmin * static_cast<float>(m2)) * scale;
    } else if (qtype == QuantType::Q5_K) {
        const int nb = K >> 8;
        const int b = tid >> 7;
        if (b >= nb) return;
        const int loc = tid & 127, j = loc >> 5, l = loc & 31;
        const block_q5_K* blk = reinterpret_cast<const block_q5_K*>(rp + static_cast<size_t>(b) * sizeof(block_q5_K));
        const float d = d_fp16_to_f32(blk->d), dmin = d_fp16_to_f32(blk->dmin);
        uint8_t sc1, m1, sc2, m2;
        d_get_scale_min_k4(2 * j, blk->scales, sc1, m1);
        d_get_scale_min_k4(2 * j + 1, blk->scales, sc2, m2);
        const uint8_t ql = blk->qs[(j << 5) + l], qhb = blk->qh[l];
        const uint8_t u1 = static_cast<uint8_t>(1u << (2 * j)), u2 = static_cast<uint8_t>(2u << (2 * j));
        const int base = (b << 8) + (j << 6) + l;
        dst[base]      = ((d * static_cast<float>(sc1)) * static_cast<float>((ql & 0x0Fu) + ((qhb & u1) ? 16 : 0)) - dmin * static_cast<float>(m1)) * scale;
        dst[base + 32] = ((d * static_cast<float>(sc2)) * static_cast<float>((ql >> 4)    + ((qhb & u2) ? 16 : 0)) - dmin * static_cast<float>(m2)) * scale;
    } else if (qtype == QuantType::IQ4_NL) {
        const int nb = K >> 5;
        const int b = tid >> 4;
        if (b >= nb) return;
        const int j = tid & 15;
        const block_iq4_nl* blk = reinterpret_cast<const block_iq4_nl*>(rp + static_cast<size_t>(b) * sizeof(block_iq4_nl));
        const float d = d_fp16_to_f32(blk->d) * scale;
        const uint8_t q = blk->qs[j];
        dst[(b << 5) + j]      = d * static_cast<float>(d_kvalues_iq4nl[q & 0x0Fu]);
        dst[(b << 5) + 16 + j] = d * static_cast<float>(d_kvalues_iq4nl[q >> 4]);
    } else if (qtype == QuantType::Q5_1) {
        const int nb = K >> 5;
        const int b = tid >> 4;
        if (b >= nb) return;
        const int j = tid & 15;
        const block_q5_1* blk = reinterpret_cast<const block_q5_1*>(rp + static_cast<size_t>(b) * sizeof(block_q5_1));
        const float d = d_fp16_to_f32(blk->d), m = d_fp16_to_f32(blk->m);
        uint32_t qh; memcpy(&qh, blk->qh, sizeof(uint32_t));
        const uint8_t q = blk->qs[j];
        dst[(b << 5) + j]      = fmaf(d, static_cast<float>((q & 0x0Fu) | (((qh >> j) & 1u) << 4)), m) * scale;
        dst[(b << 5) + 16 + j] = fmaf(d, static_cast<float>((q >> 4)    | (((qh >> (j + 16)) & 1u) << 4)), m) * scale;
    } else if (qtype == QuantType::Q4_0) {
        const int nb = K >> 5;
        const int b = tid >> 4;
        if (b >= nb) return;
        const int j = tid & 15;
        const block_q4_0* blk = reinterpret_cast<const block_q4_0*>(rp + static_cast<size_t>(b) * sizeof(block_q4_0));
        const float d = d_fp16_to_f32(blk->d) * scale;
        const uint8_t q = blk->qs[j];
        dst[(b << 5) + j * 2 + 0] = d * static_cast<float>(static_cast<int>(q & 0x0Fu) - 8);
        dst[(b << 5) + j * 2 + 1] = d * static_cast<float>(static_cast<int>(q >> 4)    - 8);
    } else if (qtype == QuantType::BF16) {
        if (tid >= K) return;
        dst[tid] = d_bf16_to_f32(reinterpret_cast<const uint16_t*>(rp)[tid]) * scale;
    } else if (qtype == QuantType::F16) {
        if (tid >= K) return;
        dst[tid] = d_fp16_to_f32(reinterpret_cast<const uint16_t*>(rp)[tid]) * scale;
    } else if (qtype == QuantType::F32) {
        if (tid >= K) return;
        dst[tid] = reinterpret_cast<const float*>(rp)[tid] * scale;
    }
}

// ============================================================================
// Elementwise, Norm, FWHT, RoPE, Attention, MoE, Pleias, & Argmax Kernels
// ============================================================================

__global__ void k_f32_to_f16(const float* __restrict__ src, __half* __restrict__ dst, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2half(src[i]);
}

__global__ void k_add_bias_2d(float* __restrict__ y, const float* __restrict__ bias, int S, int N) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = S * N;
    if (i < total) y[i] += bias[i % N];
}

__global__ void k_add_residual(float* __restrict__ x, const float* __restrict__ delta, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += delta[i];
}

__global__ void k_scale_inplace(float* __restrict__ x, float s, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= s;
}

__global__ void k_act_mul(float* __restrict__ gate, const float* __restrict__ up, int n, bool use_gelu) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float g = gate[i];
        gate[i] = (use_gelu ? d_gelu_tanh(g) : d_silu(g)) * up[i];
    }
}

__global__ void k_softcap_inplace(float* __restrict__ x, int n, float cap) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        x[i] = cap * tanhf(x[i] / cap);
    }
}

// RMSNorm across rows: [S, D] with optional 1D weight [D] and optional extra scalar multiplier
__global__ void k_rms_norm_rows(float* __restrict__ out, const float* __restrict__ x,
                                const float* __restrict__ w, int D, float eps, float extra_scale = 1.0f) {
    const int s = blockIdx.x;
    const float* xs = x + static_cast<size_t>(s) * static_cast<size_t>(D);
    float* os = out + static_cast<size_t>(s) * static_cast<size_t>(D);

    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float v = xs[i];
        sum_sq = fmaf(v, v, sum_sq);
    }
    sum_sq = warp_reduce_sum(sum_sq);
    __shared__ float smem[32];
    const int wid = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    if (lane == 0) smem[wid] = sum_sq;
    __syncthreads();
    if (wid == 0) {
        const int n_warps = blockDim.x >> 5;
        float val = (lane < n_warps) ? smem[lane] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane == 0) smem[0] = rsqrtf(val / static_cast<float>(D) + eps) * extra_scale;
    }
    __syncthreads();
    const float sc = smem[0];
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        os[i] = xs[i] * sc * (w ? w[i] : 1.0f);
    }
}

// Shared-Memory Normalized Orthogonal Block-FWHT (Gemma 4.5 Turbo: block_size = 64, 128, 256, 512, or 1024)
// Grid = (total_elems / block_size), blockDim.x = block_size / 2
__global__ void k_fwht_block_inplace(float* __restrict__ data, int block_size) {
    extern __shared__ float smem[];
    const int half = block_size >> 1;
    const int tid = threadIdx.x;
    float* blk = data + static_cast<size_t>(blockIdx.x) * static_cast<size_t>(block_size);

    smem[tid]        = blk[tid];
    smem[tid + half] = blk[tid + half];
    __syncthreads();

    for (int h = 1; h < block_size; h <<= 1) {
        const int grp = tid / h;
        const int loc = tid - grp * h;
        const int i0  = (grp << 1) * h + loc;
        const int i1  = i0 + h;
        const float a = smem[i0];
        const float b = smem[i1];
        smem[i0] = a + b;
        smem[i1] = a - b;
        __syncthreads();
    }

    const float inv_s = rsqrtf(static_cast<float>(block_size));
    blk[tid]        = smem[tid] * inv_s;
    blk[tid + half] = smem[tid + half] * inv_s;
}

// Per-Head Half-Split RoPE on Q[S, H, HD] and optional K[S, Hkv, HD] + FP16 KV Cache Write!
__global__ void k_rope_and_kv_write(float* __restrict__ Q, float* __restrict__ K, const float* __restrict__ V,
                                    __half* __restrict__ k_cache, __half* __restrict__ v_cache,
                                    int S, int start_pos, int max_seq_len,
                                    int H, int Hkv, int HD,
                                    float freq_base, float freq_scale,
                                    const float* __restrict__ freq_factors, int freq_factors_len,
                                    bool has_kv) {
    const int s = blockIdx.y;
    const int h = blockIdx.x;
    const int i = threadIdx.x;
    const int num_pairs = HD >> 1;
    if (s >= S || i >= num_pairs) return;

    const int pos = start_pos + s;
    const float inv_rope_dim = -2.0f / static_cast<float>(HD);
    float theta = static_cast<float>(pos) * freq_scale * powf(freq_base, static_cast<float>(i) * inv_rope_dim);
    if (freq_factors && i < freq_factors_len && freq_factors[i] > 0.0f) {
        theta /= freq_factors[i];
    }
    float sv, cv;
    __sincosf(theta, &sv, &cv);

    if (h < H) {
        float* qh = Q + (static_cast<size_t>(s) * static_cast<size_t>(H) + static_cast<size_t>(h)) * static_cast<size_t>(HD);
        const float v0 = qh[i];
        const float v1 = qh[i + num_pairs];
        qh[i]             = v0 * cv - v1 * sv;
        qh[i + num_pairs] = v0 * sv + v1 * cv;
    }

    if (has_kv && h < Hkv && pos >= 0 && pos < max_seq_len) {
        const size_t kv_dim = static_cast<size_t>(Hkv) * static_cast<size_t>(HD);
        float* kh = K + (static_cast<size_t>(s) * static_cast<size_t>(Hkv) + static_cast<size_t>(h)) * static_cast<size_t>(HD);
        const float* vh = V + (static_cast<size_t>(s) * static_cast<size_t>(Hkv) + static_cast<size_t>(h)) * static_cast<size_t>(HD);
        const float k0 = kh[i];
        const float k1 = kh[i + num_pairs];
        const float kr0 = k0 * cv - k1 * sv;
        const float kr1 = k0 * sv + k1 * cv;
        __half* kc = k_cache + static_cast<size_t>(pos) * kv_dim + static_cast<size_t>(h) * static_cast<size_t>(HD);
        __half* vc = v_cache + static_cast<size_t>(pos) * kv_dim + static_cast<size_t>(h) * static_cast<size_t>(HD);
        kc[i]             = __float2half(kr0);
        kc[i + num_pairs] = __float2half(kr1);
        vc[i]             = __float2half(vh[i]);
        vc[i + num_pairs] = __float2half(vh[i + num_pairs]);
    }
}

// Exact Online-Softmax GQA Attention Kernel (works for both Decode S=1 and Causal Prefill S>1!)
// Grid: (H, S, 1), Block: 256 threads (8 warps).
// Each thread computes partial Q·K across HD, then parallelizes over KV positions t with exact online softmax!
__global__ void k_gqa_attention(const float* __restrict__ Q,
                                const __half* __restrict__ k_cache,
                                const __half* __restrict__ v_cache,
                                float* __restrict__ attn_out,
                                int S, int start_pos, int H, int Hkv, int HD,
                                int swa_win, bool is_swa, float attn_scale, float attn_softcap) {
    const int h = blockIdx.x;
    const int s = blockIdx.y;
    if (h >= H || s >= S) return;

    const int tid = threadIdx.x; // 0..255
    const int wid = tid >> 5;    // 0..7
    const int lane = tid & 31;
    const int gqa_ratio = (Hkv > 0) ? (H / Hkv) : 1;
    const int kvh = min(h / max(1, gqa_ratio), max(1, Hkv) - 1);
    const size_t kv_dim = static_cast<size_t>(Hkv) * static_cast<size_t>(HD);
    const size_t kv_off = static_cast<size_t>(kvh) * static_cast<size_t>(HD);

    const float* qh = Q + (static_cast<size_t>(s) * static_cast<size_t>(H) + static_cast<size_t>(h)) * static_cast<size_t>(HD);
    float* oh = attn_out + (static_cast<size_t>(s) * static_cast<size_t>(H) + static_cast<size_t>(h)) * static_cast<size_t>(HD);

    // Load Q head into shared memory (max HD=512 -> 2 KB)
    __shared__ float sq[512];
    for (int d = tid; d < HD; d += blockDim.x) {
        sq[d] = qh[d] * attn_scale;
    }
    __syncthreads();

    const int seq_len = start_pos + s + 1;
    const int s_start = is_swa ? max(0, seq_len - swa_win) : 0;
    const bool use_sinks = (is_swa && s_start > 4);
    const int active_len = use_sinks ? (4 + (seq_len - s_start)) : (seq_len - s_start);

    // Each of the 8 warps processes strided KV positions (t_idx = wid, wid+8, ...),
    // keeping a local online-softmax state (m_w, l_w) and local output accumulator in registers!
    // Max HD=512 across 32 lanes -> at most 16 floats/lane in registers!
    float acc[16];
    const int elems_per_lane = (HD + 31) >> 5;
    #pragma unroll
    for (int e = 0; e < 16; ++e) acc[e] = 0.0f;

    float m_w = -1.0e30f;
    float l_w = 0.0f;

    for (int ti = wid; ti < active_len; ti += 8) {
        const int pos_t = use_sinks ? ((ti < 4) ? ti : (s_start + (ti - 4))) : (s_start + ti);
        const __half* kt = k_cache + static_cast<size_t>(pos_t) * kv_dim + kv_off;
        const __half* vt = v_cache + static_cast<size_t>(pos_t) * kv_dim + kv_off;

        float dot = 0.0f;
        for (int e = 0; e < elems_per_lane; ++e) {
            const int d = (e << 5) + lane;
            if (d < HD) {
                dot = fmaf(sq[d], __half2float(kt[d]), dot);
            }
        }
        dot = warp_reduce_sum(dot);
        dot = __shfl_sync(0xFFFFFFFFu, dot, 0);
        if (attn_softcap > 0.0f) {
            dot = attn_softcap * tanhf(dot / attn_softcap);
        }

        const float m_new = fmaxf(m_w, dot);
        const float alpha = expf(m_w - m_new);
        const float beta  = expf(dot - m_new);
        l_w = l_w * alpha + beta;
        m_w = m_new;

        for (int e = 0; e < elems_per_lane; ++e) {
            const int d = (e << 5) + lane;
            if (d < HD) {
                acc[e] = fmaf(beta, __half2float(vt[d]), acc[e] * alpha);
            }
        }
    }

    // Merge the 8 warps' exact online-softmax states via shared memory
    __shared__ float sm_m[8];
    __shared__ float sm_l[8];
    __shared__ float sm_out[8 * 512]; // 16 KB shared memory -> high occupancy!
    if (lane == 0) {
        sm_m[wid] = m_w;
        sm_l[wid] = l_w;
    }
    for (int e = 0; e < elems_per_lane; ++e) {
        const int d = (e << 5) + lane;
        if (d < HD) sm_out[wid * 512 + d] = acc[e];
    }
    __syncthreads();

    float m_glob = sm_m[0];
    #pragma unroll
    for (int w = 1; w < 8; ++w) m_glob = fmaxf(m_glob, sm_m[w]);

    float l_glob = 0.0f;
    float w_scale[8];
    #pragma unroll
    for (int w = 0; w < 8; ++w) {
        w_scale[w] = (sm_l[w] > 0.0f) ? expf(sm_m[w] - m_glob) : 0.0f;
        l_glob += sm_l[w] * w_scale[w];
    }
    const float inv_l = (l_glob > 1e-12f) ? (1.0f / l_glob) : 0.0f;

    for (int d = tid; d < HD; d += blockDim.x) {
        float val = 0.0f;
        #pragma unroll
        for (int w = 0; w < 8; ++w) {
            val = fmaf(sm_out[w * 512 + d], w_scale[w], val);
        }
        oh[d] = val * inv_l;
    }
}

// ============================================================================
// Gemma 4.5 26B-A4B Sparse MoE Kernels (Zero Host-Device Sync!)
// ============================================================================

// Softmax over E router logits + Top-K (K <= 16) selection & renormalization + optional down_s fold
// Grid: (S, 1, 1), Block: 128 threads
__global__ void k_moe_router_topk(const float* __restrict__ logits, const float* __restrict__ down_s,
                                  int* __restrict__ exp_idx_out, float* __restrict__ exp_w_out,
                                  int E, int K_used) {
    const int s = blockIdx.x;
    const float* row = logits + static_cast<size_t>(s) * static_cast<size_t>(E);
    int* idx_o = exp_idx_out + static_cast<size_t>(s) * static_cast<size_t>(K_used);
    float* w_o = exp_w_out + static_cast<size_t>(s) * static_cast<size_t>(K_used);

    __shared__ float probs[256];
    const int tid = threadIdx.x;
    float local_max = -1.0e30f;
    for (int e = tid; e < E; e += blockDim.x) {
        local_max = fmaxf(local_max, row[e]);
    }
    local_max = warp_reduce_max(local_max);
    __shared__ float sm_red[8];
    const int wid = tid >> 5, lane = tid & 31;
    if (lane == 0) sm_red[wid] = local_max;
    __syncthreads();
    if (wid == 0) {
        float v = (lane < (blockDim.x >> 5)) ? sm_red[lane] : -1.0e30f;
        v = warp_reduce_max(v);
        if (lane == 0) sm_red[0] = v;
    }
    __syncthreads();
    const float max_v = sm_red[0];

    float local_sum = 0.0f;
    for (int e = tid; e < E; e += blockDim.x) {
        float ex = expf(row[e] - max_v);
        probs[e] = ex;
        local_sum += ex;
    }
    local_sum = warp_reduce_sum(local_sum);
    if (lane == 0) sm_red[wid] = local_sum;
    __syncthreads();
    if (wid == 0) {
        float v = (lane < (blockDim.x >> 5)) ? sm_red[lane] : 0.0f;
        v = warp_reduce_sum(v);
        if (lane == 0) sm_red[0] = (v > 1e-12f) ? (1.0f / v) : 1.0f;
    }
    __syncthreads();
    const float inv_sum = sm_red[0];
    for (int e = tid; e < E; e += blockDim.x) {
        probs[e] *= inv_sum;
    }
    __syncthreads();

    // Thread 0 performs exact deterministic Top-K selection over E <= 256 in shared mem (<1 us!)
    if (tid == 0) {
        float w_sum = 0.0f;
        for (int k = 0; k < K_used; ++k) {
            int best_e = 0;
            float best_p = -1.0f;
            for (int e = 0; e < E; ++e) {
                if (probs[e] > best_p) {
                    best_p = probs[e];
                    best_e = e;
                }
            }
            idx_o[k] = best_e;
            w_o[k]   = best_p;
            w_sum   += best_p;
            probs[best_e] = -2.0f; // Mark selected
        }
        if (w_sum < 6.103515625e-5f) w_sum = 6.103515625e-5f;
        for (int k = 0; k < K_used; ++k) {
            const int e = idx_o[k];
            w_o[k] = (w_o[k] / w_sum) * (down_s ? down_s[e] : 1.0f);
        }
    }
}

// Fused MoE Active-K Gate+Up+GeGLU/SiLU Kernel (reads ONLY the K_used active experts!)
// Grid: (ceil(K_used * F_exp / 4), S, 1), Block: 128 threads
__global__ void k_moe_gu_act(const uint8_t* __restrict__ Wgu, QuantType qt_gu, size_t rb_gu,
                             const uint8_t* __restrict__ Wg,  QuantType qt_g,  size_t rb_g,
                             const uint8_t* __restrict__ Wu,  QuantType qt_u,  size_t rb_u,
                             const float* __restrict__ x,
                             const int* __restrict__ exp_idx,
                             float* __restrict__ act_out,
                             int K_used, int F_exp, int D, bool fused_gu, bool use_gelu) {
    const int s       = blockIdx.y;
    const int warp_id = threadIdx.x >> 5;
    const int lane    = threadIdx.x & 31;
    const int idx     = (blockIdx.x << 2) + warp_id;
    if (idx >= K_used * F_exp) return;

    const int k = idx / F_exp;
    const int r = idx - k * F_exp;
    const int e = exp_idx[s * K_used + k];
    const float* xs = x + static_cast<size_t>(s) * static_cast<size_t>(D);

    float g = 0.0f, u = 0.0f;
    if (fused_gu) {
        const size_t base_row = static_cast<size_t>(e) * static_cast<size_t>(2 * F_exp);
        g = warp_dot_any(Wgu + (base_row + static_cast<size_t>(r)) * rb_gu, qt_gu, xs, D, lane);
        u = warp_dot_any(Wgu + (base_row + static_cast<size_t>(F_exp + r)) * rb_gu, qt_gu, xs, D, lane);
    } else {
        const size_t row = static_cast<size_t>(e) * static_cast<size_t>(F_exp) + static_cast<size_t>(r);
        g = warp_dot_any(Wg + row * rb_g, qt_g, xs, D, lane);
        u = warp_dot_any(Wu + row * rb_u, qt_u, xs, D, lane);
    }
    if (lane == 0) {
        const float act = use_gelu ? d_gelu_tanh(g) : d_silu(g);
        act_out[(static_cast<size_t>(s) * static_cast<size_t>(K_used) + static_cast<size_t>(k)) * static_cast<size_t>(F_exp) + static_cast<size_t>(r)] = act * u;
    }
}

// Fused MoE Active-K Weighted Down Projection Kernel: moe_out[s, d] = sum_{k=0..K-1} w[s,k] * dot(Wdn[e, d], act[s, k])
// Grid: (ceil(D / 4), S, 1), Block: 128 threads
__global__ void k_moe_down_accum(const uint8_t* __restrict__ Wdn, QuantType qt_dn, size_t rb_dn,
                                 const float* __restrict__ act_in,
                                 const int* __restrict__ exp_idx,
                                 const float* __restrict__ exp_w,
                                 float* __restrict__ out,
                                 int K_used, int F_exp, int D) {
    const int s       = blockIdx.y;
    const int warp_id = threadIdx.x >> 5;
    const int lane    = threadIdx.x & 31;
    const int d       = (blockIdx.x << 2) + warp_id;
    if (d >= D) return;

    float total = 0.0f;
    for (int k = 0; k < K_used; ++k) {
        const int e = exp_idx[s * K_used + k];
        const float w = exp_w[s * K_used + k];
        const float* ak = act_in + (static_cast<size_t>(s) * static_cast<size_t>(K_used) + static_cast<size_t>(k)) * static_cast<size_t>(F_exp);
        const size_t row = static_cast<size_t>(e) * static_cast<size_t>(D) + static_cast<size_t>(d);
        const float dot = warp_dot_any(Wdn + row * rb_dn, qt_dn, ak, F_exp, lane);
        if (lane == 0) total = fmaf(w, dot, total);
    }
    if (lane == 0) {
        out[static_cast<size_t>(s) * static_cast<size_t>(D) + static_cast<size_t>(d)] = total;
    }
}

// Pleias post-proj combine: e[s, l, p] = (p_proj[s, l, p] + e[s, l, p]) * inv_sqrt2
__global__ void k_pleias_combine(float* __restrict__ pl_emb, const float* __restrict__ pl_proj, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        pl_emb[i] = (pl_proj[i] + pl_emb[i]) * 0.7071067811865475f;
    }
}

// Pleias gate activation & multiply with layer embedding slice: gate[s, p] = gelu(gate[s, p]) * pl_emb[s, l, p]
__global__ void k_pleias_gate_mul(float* __restrict__ gate, const float* __restrict__ pl_emb,
                                  int S, int L, int l, int P_dim) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= S * P_dim) return;
    const int s = i / P_dim;
    const int p = i - s * P_dim;
    const float g = gate[i];
    gate[i] = d_gelu_tanh(g) * pl_emb[(static_cast<size_t>(s) * static_cast<size_t>(L) + static_cast<size_t>(l)) * static_cast<size_t>(P_dim) + static_cast<size_t>(p)];
}

// Two-Stage Fast Parallel Argmax over [V] F32 logits -> 4-byte uint32_t on GPU
__global__ void k_argmax_stage1(const float* __restrict__ logits, int V,
                                float* __restrict__ blk_vals, uint32_t* __restrict__ blk_idx) {
    float best_v = -1.0e30f;
    uint32_t best_i = 0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < V; i += gridDim.x * blockDim.x) {
        const float v = logits[i];
        if (v > best_v) { best_v = v; best_i = static_cast<uint32_t>(i); }
    }
    __shared__ float sv[256];
    __shared__ uint32_t si[256];
    sv[threadIdx.x] = best_v;
    si[threadIdx.x] = best_i;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride && sv[threadIdx.x + stride] > sv[threadIdx.x]) {
            sv[threadIdx.x] = sv[threadIdx.x + stride];
            si[threadIdx.x] = si[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        blk_vals[blockIdx.x] = sv[0];
        blk_idx[blockIdx.x]  = si[0];
    }
}

__global__ void k_argmax_stage2(const float* __restrict__ blk_vals, const uint32_t* __restrict__ blk_idx,
                                int n_blocks, uint32_t* __restrict__ out_idx) {
    __shared__ float sv[256];
    __shared__ uint32_t si[256];
    const int tid = threadIdx.x;
    sv[tid] = (tid < n_blocks) ? blk_vals[tid] : -1.0e30f;
    si[tid] = (tid < n_blocks) ? blk_idx[tid]  : 0u;
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride && sv[tid + stride] > sv[tid]) {
            sv[tid] = sv[tid + stride];
            si[tid] = si[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) out_idx[0] = si[0];
}

// ============================================================================
// GPU Tensor & Layer Descriptors
// ============================================================================

struct GpuTensor {
    uint8_t* ptr = nullptr;
    QuantType type = QuantType::F32;
    int rows = 0;
    int cols = 0;
    size_t row_bytes = 0;
    size_t size_bytes = 0;
    bool valid() const { return ptr != nullptr && rows > 0 && cols > 0; }
};

struct GpuLayer {
    int F_hid = 0;
    int q_dim = 0;
    int kv_dim = 0;
    int head_dim = 0;
    int n_kv_heads = 0;
    bool is_swa = false;
    bool has_kv = true;
    bool k_eq_v = false;
    int kv_source_layer = 0;
    float layer_output_scale = 1.0f;

    GpuTensor attn_q, attn_k, attn_v, attn_output;
    float* d_attn_q_bias = nullptr;
    float* d_attn_k_bias = nullptr;
    float* d_attn_v_bias = nullptr;
    float* d_attn_norm = nullptr;
    float* d_attn_q_norm = nullptr;
    float* d_attn_k_norm = nullptr;
    float* d_post_attn_norm = nullptr;

    GpuTensor ffn_gate, ffn_up, ffn_down;
    float* d_ffn_norm = nullptr;
    float* d_post_ffn_norm = nullptr;

    // Sparse MoE (Gemma 4.5 26B-A4B)
    bool has_moe = false;
    bool fused_moe_gu = false;
    GpuTensor moe_router;
    GpuTensor moe_gu, moe_g, moe_u, moe_dn;
    float* d_moe_rscale = nullptr;
    float* d_moe_down_scale = nullptr;
    float* d_post_ffn_norm_1 = nullptr;
    float* d_pre_ffn_norm_2 = nullptr;
    float* d_post_ffn_norm_2 = nullptr;

    // Pleias per-layer injection (Gemma 4.5 E2B/E4B)
    GpuTensor inp_gate, proj;
    float* d_post_norm = nullptr;

    // Per-layer FP16 KV Cache (allocated only if has_kv == true)
    __half* d_k_cache = nullptr;
    __half* d_v_cache = nullptr;
    size_t kv_bytes = 0;
};

} // namespace

struct HavenCudaEngine::Impl {
    cudaStream_t stream = nullptr;
    cublasHandle_t cublas = nullptr;
    ModelConfig mc{};

    std::vector<void*> all_allocs;
    std::vector<GpuLayer> layers;

    GpuTensor tok_emb;
    GpuTensor lm_head;
    GpuTensor pl_tok_emb;
    GpuTensor pl_model_proj;
    float* d_pl_proj_norm = nullptr;
    float* d_out_norm = nullptr;
    float* d_rope_freqs = nullptr;
    int rope_freqs_len = 0;

    bool has_pleias = false;
    bool tied_emb = true;
    int max_F = 0;
    int max_q_dim = 0;
    int max_kv_dim = 0;
    int max_S = 256;

    // Activation & GEMM scratch buffers
    uint32_t* d_tok_ids = nullptr;
    float* d_x = nullptr;           // [max_S, D]
    float* d_norm = nullptr;        // [max_S, D]
    float* d_q = nullptr;           // [max_S, max_q_dim]
    float* d_k = nullptr;           // [max_S, max_kv_dim]
    float* d_v = nullptr;           // [max_S, max_kv_dim]
    float* d_attn_h = nullptr;      // [max_S, max_q_dim]
    float* d_attn_o = nullptr;      // [max_S, D]
    float* d_ffn_g = nullptr;       // [max_S, max_F]
    float* d_ffn_u = nullptr;       // [max_S, max_F]
    float* d_ffn_d = nullptr;       // [max_S, D]

    // MoE scratch
    float* d_moe_norm = nullptr;    // [max_S, D]
    float* d_moe_router = nullptr;  // [max_S, E]
    int*   d_moe_idx = nullptr;     // [max_S, K_used]
    float* d_moe_w = nullptr;       // [max_S, K_used]
    float* d_moe_act = nullptr;     // [max_S, K_used * F_exp]
    float* d_moe_out = nullptr;     // [max_S, D]

    // Pleias scratch
    float* d_pl_emb = nullptr;      // [max_S, L * 256]
    float* d_pl_proj = nullptr;     // [max_S, L * 256]
    float* d_pl_gate = nullptr;     // [max_S, 256]
    float* d_pl_out = nullptr;      // [max_S, D]

    // Logits & Argmax scratch
    float* d_logits = nullptr;      // [V]
    float* d_am_vals = nullptr;     // [256]
    uint32_t* d_am_idx = nullptr;   // [256]
    uint32_t* d_am_out = nullptr;   // [1]

    // FP16 cuBLAS Tensor Core prefill scratch
    __half* d_x_f16 = nullptr;      // [max_S, max(D, max_F, max_q_dim)]
    __half* d_w_f16 = nullptr;      // [max_w_elems] (single-matrix reusable tile)
    size_t w_f16_cap_elems = 0;

    void* alloc_gpu(size_t bytes, size_t& counter) {
        if (bytes == 0) return nullptr;
        void* p = nullptr;
        if (cudaMalloc(&p, bytes) != cudaSuccess) return nullptr;
        all_allocs.push_back(p);
        counter += bytes;
        return p;
    }

    bool upload_raw_tensor(const TensorDesc* td, int rows, int cols, GpuTensor& out, size_t& weight_bytes, std::string& err) {
        if (!td || !td->data || rows <= 0 || cols <= 0) {
            out = {};
            return true;
        }
        const size_t rb = Avx2Math::row_bytes(td->type, cols);
        if (rb == 0) {
            err = "Unsupported GGUF QuantType=" + std::to_string(static_cast<int>(td->type)) + " for tensor " + td->name;
            return false;
        }
        const size_t need_bytes = static_cast<size_t>(rows) * rb;
        if (td->size_bytes > 0 && need_bytes > td->size_bytes) {
            err = "Tensor " + td->name + " byte size mismatch (need " + std::to_string(need_bytes) + ", have " + std::to_string(td->size_bytes) + ")";
            return false;
        }
        void* dp = alloc_gpu(need_bytes, weight_bytes);
        if (!dp) {
            err = "cudaMalloc failed for " + td->name + " (" + std::to_string(need_bytes / (1024 * 1024)) + " MB)";
            return false;
        }
        cudaError_t ce = cudaMemcpy(dp, td->data, need_bytes, cudaMemcpyHostToDevice);
        if (ce != cudaSuccess) {
            err = "cudaMemcpy H2D failed for " + td->name + ": " + cudaGetErrorString(ce);
            return false;
        }
        out.ptr = static_cast<uint8_t*>(dp);
        out.type = td->type;
        out.rows = rows;
        out.cols = cols;
        out.row_bytes = rb;
        out.size_bytes = need_bytes;
        return true;
    }

    float* upload_f32_1d(const TensorDesc* td, int count, float fallback, size_t& weight_bytes, std::string& err) {
        std::vector<float> host(static_cast<size_t>(count), fallback);
        if (td && td->data) {
            if (td->type == QuantType::F32 && (td->size_bytes == 0 || td->size_bytes >= static_cast<size_t>(count) * sizeof(float))) {
                memcpy(host.data(), td->data, static_cast<size_t>(count) * sizeof(float));
            } else {
                Avx2Math::dequantize_row(host.data(), *td, 0, count);
            }
        }
        void* dp = alloc_gpu(static_cast<size_t>(count) * sizeof(float), weight_bytes);
        if (!dp) {
            err = "cudaMalloc failed for 1D F32 vector";
            return nullptr;
        }
        if (cudaMemcpy(dp, host.data(), static_cast<size_t>(count) * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
            err = "cudaMemcpy H2D failed for 1D F32 vector";
            return nullptr;
        }
        return static_cast<float*>(dp);
    }

    // Dispatch linear projection: [S, K] x [N, K]^T -> [S, N] (F32)
    // Uses direct warp-specialized GEMV for S < 8 (zero scratch, 90% GDDR6 BW!)
    // Uses on-the-fly FP16 dequant + cuBLAS FP16 Tensor Core GEMM for S >= 8!
    void linear_forward(const GpuTensor& W, const float* d_bias, const float* d_in, float* d_out, int S, float scale_out = 1.0f) {
        const int N = W.rows;
        const int K = W.cols;
        if (S <= 1) {
            const int blocks = (N + 3) >> 2;
            k_gemv_quant<<<blocks, 128, 0, stream>>>(W.ptr, W.type, W.row_bytes, d_in, d_bias, d_out, N, K, scale_out);
            return;
        }
        if (S < 8 || static_cast<size_t>(N) * static_cast<size_t>(K) > w_f16_cap_elems) {
            const int blocks = (N + 3) >> 2;
            for (int s = 0; s < S; ++s) {
                k_gemv_quant<<<blocks, 128, 0, stream>>>(
                    W.ptr, W.type, W.row_bytes,
                    d_in + static_cast<size_t>(s) * static_cast<size_t>(K),
                    d_bias,
                    d_out + static_cast<size_t>(s) * static_cast<size_t>(N),
                    N, K, scale_out);
            }
            return;
        }

        // Convert X [S, K] F32 -> FP16
        const int x_n = S * K;
        k_f32_to_f16<<<(x_n + 255) / 256, 256, 0, stream>>>(d_in, d_x_f16, x_n);

        // Dequantize W [N, K] raw blocks -> FP16 scratch
        int threads_per_row = K;
        if (W.type == QuantType::IQ4_XS || W.type == QuantType::IQ4_NL ||
            W.type == QuantType::Q5_K || W.type == QuantType::Q4_K ||
            W.type == QuantType::Q5_1 || W.type == QuantType::Q5_0 || W.type == QuantType::Q4_0) {
            threads_per_row = K >> 1;
        } else if (W.type == QuantType::Q6_K) {
            threads_per_row = K >> 2;
        } else if (W.type == QuantType::Q3_K) {
            threads_per_row = K >> 3;
        }
        dim3 dq_grid((threads_per_row + 255) / 256, static_cast<unsigned int>(N), 1);
        k_dequant_to_f16<<<dq_grid, 256, 0, stream>>>(W.ptr, W.type, W.row_bytes, d_w_f16, N, K);

        // Row-major Y[S, N] = X[S, K] @ W[N, K]^T <=> Col-major Y^T[N, S] = (W^T)^T @ X^T
        const float alpha = scale_out;
        const float beta  = 0.0f;
        cublasGemmEx(
            cublas,
            CUBLAS_OP_T, CUBLAS_OP_N,
            N, S, K,
            &alpha,
            d_w_f16, CUDA_R_16F, K,
            d_x_f16, CUDA_R_16F, K,
            &beta,
            d_out,   CUDA_R_32F, N,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP);

        if (d_bias) {
            const int tot = S * N;
            k_add_bias_2d<<<(tot + 255) / 256, 256, 0, stream>>>(d_out, d_bias, S, N);
        }
    }

    void launch_fwht(float* d_buf, int total_elems, int block_size) {
        if (block_size <= 1 || block_size > 1024 || (block_size & (block_size - 1)) != 0 || (total_elems % block_size) != 0) {
            return;
        }
        const int num_blocks = total_elems / block_size;
        const int threads = block_size >> 1;
        k_fwht_block_inplace<<<num_blocks, threads, static_cast<size_t>(block_size) * sizeof(float), stream>>>(d_buf, block_size);
    }

    void dequant_embed_rows(const GpuTensor& tbl, const uint32_t* d_ids, int S, int vocab_size, float* d_out, float scale) {
        const int K = tbl.cols;
        int threads_per_row = K;
        if (tbl.type == QuantType::Q6_K) threads_per_row = K >> 2;
        else if (tbl.type == QuantType::IQ4_XS || tbl.type == QuantType::IQ4_NL ||
                 tbl.type == QuantType::Q4_K || tbl.type == QuantType::Q5_K ||
                 tbl.type == QuantType::Q5_1 || tbl.type == QuantType::Q5_0 || tbl.type == QuantType::Q4_0) {
            threads_per_row = K >> 1;
        }
        dim3 grid((threads_per_row + 255) / 256, static_cast<unsigned int>(S), 1);
        k_dequant_rows_f32<<<grid, 256, 0, stream>>>(tbl.ptr, tbl.type, tbl.row_bytes, d_ids, S, vocab_size, d_out, K, scale);
    }

    bool run_forward_chunk(const uint32_t* h_tokens, int S, int start_pos,
                           float* out_last_logits, uint32_t* out_next_token, std::string& err) {
        const int D = static_cast<int>(mc.embedding_dim);
        const int L = static_cast<int>(mc.num_layers);
        const int H = static_cast<int>(mc.num_heads);
        const int V = static_cast<int>(mc.vocab_size);
        const bool use_gelu = (mc.ffn_activation == FfnActivation::GELU);
        const int swa_win = (mc.sliding_window > 0) ? static_cast<int>(mc.sliding_window) : 512;

        // 1. Copy token IDs H2D (4 * S bytes!) & Gather Token Embeddings on GPU
        cudaMemcpyAsync(d_tok_ids, h_tokens, static_cast<size_t>(S) * sizeof(uint32_t), cudaMemcpyHostToDevice, stream);
        const float emb_scale = mc.is_gemma ? std::sqrt(static_cast<float>(D)) : 1.0f;
        dequant_embed_rows(tok_emb, d_tok_ids, S, V, d_x, emb_scale);

        // 1b. Optional Pleias Per-Layer Embeddings (Gemma 4.5 E2B / E4B)
        if (has_pleias) {
            dequant_embed_rows(pl_tok_emb, d_tok_ids, S, V, d_pl_emb, 16.0f);
            if (pl_model_proj.valid()) {
                linear_forward(pl_model_proj, nullptr, d_x, d_pl_proj, S, 1.0f / std::sqrt(static_cast<float>(D)));
                k_rms_norm_rows<<<S * L, 128, 0, stream>>>(d_pl_proj, d_pl_proj, d_pl_proj_norm, 256, mc.rms_norm_eps);
                const int tot_pl = S * L * 256;
                k_pleias_combine<<<(tot_pl + 255) / 256, 256, 0, stream>>>(d_pl_emb, d_pl_proj, tot_pl);
            }
        }

        // 2. Transformer Layer Loop (100% GPU-resident, zero host synchronization!)
        for (int l = 0; l < L; ++l) {
            const auto& gl = layers[static_cast<size_t>(l)];
            const int HD = gl.head_dim;
            const int Hkv = gl.n_kv_heads;
            const int q_dim = gl.q_dim;
            const int kv_dim = gl.kv_dim;
            const int F_hid = gl.F_hid;
            const float attn_scale = mc.is_gemma ? 1.0f : (1.0f / std::sqrt(static_cast<float>(HD)));

            // 2a. Pre-Attention RMSNorm + optional Gemma 4.5 Turbo Block-FWHT
            k_rms_norm_rows<<<S, 256, 0, stream>>>(d_norm, d_x, mc.turbo_folded_norms ? nullptr : gl.d_attn_norm, D, mc.rms_norm_eps);
            if (mc.turbo_hadamard_qkv > 0) {
                launch_fwht(d_norm, S * D, static_cast<int>(mc.turbo_hadamard_qkv));
            }

            // 2b. Q, K, V Projections
            if (S == 1) {
                const int k_n = gl.has_kv ? kv_dim : 0;
                const int v_n = (gl.has_kv && !gl.k_eq_v && gl.attn_v.valid()) ? kv_dim : 0;
                const int total_qkv = q_dim + k_n + v_n;
                k_fused_qkv_gemv<<<(total_qkv + 3) >> 2, 128, 0, stream>>>(
                    gl.attn_q.ptr, gl.attn_q.type, gl.attn_q.row_bytes, gl.d_attn_q_bias,
                    gl.has_kv ? gl.attn_k.ptr : nullptr, gl.attn_k.type, gl.attn_k.row_bytes, gl.d_attn_k_bias,
                    (v_n > 0) ? gl.attn_v.ptr : nullptr, gl.attn_v.type, gl.attn_v.row_bytes, gl.d_attn_v_bias,
                    d_norm, d_q, d_k, d_v, q_dim, k_n, v_n, D, gl.k_eq_v);
            } else {
                linear_forward(gl.attn_q, gl.d_attn_q_bias, d_norm, d_q, S);
                if (gl.has_kv) {
                    linear_forward(gl.attn_k, gl.d_attn_k_bias, d_norm, d_k, S);
                    if (!gl.k_eq_v && gl.attn_v.valid()) {
                        linear_forward(gl.attn_v, gl.d_attn_v_bias, d_norm, d_v, S);
                    } else {
                        cudaMemcpyAsync(d_v, d_k, static_cast<size_t>(S) * static_cast<size_t>(kv_dim) * sizeof(float), cudaMemcpyDeviceToDevice, stream);
                    }
                }
            }

            // 2c. Per-Head Q-Norm, K-Norm, Parameter-Free V-Norm
            if (gl.d_attn_q_norm) {
                k_rms_norm_rows<<<S * H, 128, 0, stream>>>(d_q, d_q, gl.d_attn_q_norm, HD, mc.rms_norm_eps);
            }
            if (gl.has_kv && gl.d_attn_k_norm) {
                k_rms_norm_rows<<<S * Hkv, 128, 0, stream>>>(d_k, d_k, gl.d_attn_k_norm, HD, mc.rms_norm_eps);
            }
            if (gl.has_kv && mc.is_gemma && gl.d_attn_q_norm && gl.d_attn_k_norm) {
                k_rms_norm_rows<<<S * Hkv, 128, 0, stream>>>(d_v, d_v, nullptr, HD, mc.rms_norm_eps);
            }

            // 2d. RoPE + FP16 KV Cache Write
            const float freq_base = gl.is_swa ? mc.rope_freq_base_swa : mc.rope_freq_base;
            const float* rf = (!gl.is_swa) ? d_rope_freqs : nullptr;
            const int rf_len = (!gl.is_swa) ? rope_freqs_len : 0;
            const int max_h = std::max(H, gl.has_kv ? Hkv : 0);
            const auto& kv_src = layers[static_cast<size_t>(gl.kv_source_layer)];
            dim3 rope_grid(static_cast<unsigned int>(max_h), static_cast<unsigned int>(S), 1);
            k_rope_and_kv_write<<<rope_grid, HD >> 1, 0, stream>>>(
                d_q, d_k, d_v,
                gl.has_kv ? gl.d_k_cache : nullptr,
                gl.has_kv ? gl.d_v_cache : nullptr,
                S, start_pos, static_cast<int>(max_seq_len_runtime),
                H, Hkv, HD, freq_base, mc.rope_freq_scale, rf, rf_len, gl.has_kv);

            // 2e. Online-Softmax GQA Attention (always match source layer KV head count for shared-KV layers!)
            dim3 attn_grid(static_cast<unsigned int>(H), static_cast<unsigned int>(S), 1);
            k_gqa_attention<<<attn_grid, 256, 0, stream>>>(
                d_q, kv_src.d_k_cache, kv_src.d_v_cache, d_attn_h,
                S, start_pos, H, kv_src.n_kv_heads, HD, swa_win, gl.is_swa, attn_scale, mc.attn_logit_softcapping);

            // 2f. Attention Output Projection + Optional Post-Attn Norm + Residual 1
            linear_forward(gl.attn_output, nullptr, d_attn_h, d_attn_o, S);
            if (gl.d_post_attn_norm) {
                k_rms_norm_rows<<<S, 256, 0, stream>>>(d_attn_o, d_attn_o, gl.d_post_attn_norm, D, mc.rms_norm_eps);
            }
            k_add_residual<<<(S * D + 255) / 256, 256, 0, stream>>>(d_x, d_attn_o, S * D);

            // 2g. Pre-FFN RMSNorm + Optional Gemma 4.5 Turbo Block-FWHT
            k_rms_norm_rows<<<S, 256, 0, stream>>>(d_norm, d_x, mc.turbo_folded_norms ? nullptr : gl.d_ffn_norm, D, mc.rms_norm_eps);
            if (mc.turbo_hadamard_ffn > 0) {
                launch_fwht(d_norm, S * D, static_cast<int>(mc.turbo_hadamard_ffn));
            }

            // 2h. Dense Shared FFN (Fused Gate+Up+Act for S=1, Tensor Core GEMM for S>=8) + Optional Turbo FWHT + Down
            if (S == 1) {
                k_fused_ffn_gu_act<<<(F_hid + 3) >> 2, 128, 0, stream>>>(
                    gl.ffn_gate.ptr, gl.ffn_gate.type, gl.ffn_gate.row_bytes,
                    gl.ffn_up.ptr,   gl.ffn_up.type,   gl.ffn_up.row_bytes,
                    d_norm, d_ffn_g, F_hid, D, use_gelu);
            } else {
                linear_forward(gl.ffn_gate, nullptr, d_norm, d_ffn_g, S);
                linear_forward(gl.ffn_up,   nullptr, d_norm, d_ffn_u, S);
                const int tot_f = S * F_hid;
                k_act_mul<<<(tot_f + 255) / 256, 256, 0, stream>>>(d_ffn_g, d_ffn_u, tot_f, use_gelu);
            }
            if (mc.turbo_hadamard_ffn_down > 0 && (F_hid % static_cast<int>(mc.turbo_hadamard_ffn_down)) == 0) {
                launch_fwht(d_ffn_g, S * F_hid, static_cast<int>(mc.turbo_hadamard_ffn_down));
            }
            linear_forward(gl.ffn_down, nullptr, d_ffn_g, d_ffn_d, S);

            // 2i. Optional Parallel Sparse MoE Branch (Gemma 4.5 26B-A4B MoE — 100% On-GPU!)
            if (gl.has_moe) {
                const int E = static_cast<int>(mc.expert_count);
                const int K_used = std::clamp(static_cast<int>(mc.expert_used_count > 0 ? mc.expert_used_count : 8u), 1, E);
                const int F_exp = static_cast<int>(mc.expert_ffn_dim > 0 ? mc.expert_ffn_dim : mc.hidden_dim);

                if (gl.d_post_ffn_norm_1) {
                    k_rms_norm_rows<<<S, 256, 0, stream>>>(d_ffn_d, d_ffn_d, gl.d_post_ffn_norm_1, D, mc.rms_norm_eps);
                }
                // Router input: parameter-free RMSNorm(x) * (1/sqrt(D)) * rscale
                k_rms_norm_rows<<<S, 256, 0, stream>>>(
                    d_moe_norm, d_x,
                    mc.turbo_folded_scales ? nullptr : gl.d_moe_rscale,
                    D, mc.rms_norm_eps, 1.0f / std::sqrt(static_cast<float>(D)));
                linear_forward(gl.moe_router, nullptr, d_moe_norm, d_moe_router, S);
                k_moe_router_topk<<<S, 128, 0, stream>>>(
                    d_moe_router, mc.turbo_folded_scales ? nullptr : gl.d_moe_down_scale,
                    d_moe_idx, d_moe_w, E, K_used);

                // Pre-FFN Norm 2 + Optional Turbo Block-256 FWHT on MoE input
                k_rms_norm_rows<<<S, 256, 0, stream>>>(
                    d_moe_norm, d_x, mc.turbo_folded_norms ? nullptr : gl.d_pre_ffn_norm_2, D, mc.rms_norm_eps);
                if (mc.turbo_hadamard_moe_in > 0) {
                    launch_fwht(d_moe_norm, S * D, static_cast<int>(mc.turbo_hadamard_moe_in));
                }

                // Active-K Experts Gate+Up+Act -> optional FWHT -> Active-K Weighted Down
                dim3 gu_grid((K_used * F_exp + 3) >> 2, static_cast<unsigned int>(S), 1);
                k_moe_gu_act<<<gu_grid, 128, 0, stream>>>(
                    gl.moe_gu.ptr, gl.moe_gu.type, gl.moe_gu.row_bytes,
                    gl.moe_g.ptr,  gl.moe_g.type,  gl.moe_g.row_bytes,
                    gl.moe_u.ptr,  gl.moe_u.type,  gl.moe_u.row_bytes,
                    d_moe_norm, d_moe_idx, d_moe_act,
                    K_used, F_exp, D, gl.fused_moe_gu, use_gelu);
                if (mc.turbo_hadamard_moe_down > 0 && (F_exp % static_cast<int>(mc.turbo_hadamard_moe_down)) == 0) {
                    launch_fwht(d_moe_act, S * K_used * F_exp, static_cast<int>(mc.turbo_hadamard_moe_down));
                }
                dim3 dn_grid((D + 3) >> 2, static_cast<unsigned int>(S), 1);
                k_moe_down_accum<<<dn_grid, 128, 0, stream>>>(
                    gl.moe_dn.ptr, gl.moe_dn.type, gl.moe_dn.row_bytes,
                    d_moe_act, d_moe_idx, d_moe_w, d_moe_out,
                    K_used, F_exp, D);

                if (gl.d_post_ffn_norm_2) {
                    k_rms_norm_rows<<<S, 256, 0, stream>>>(d_moe_out, d_moe_out, gl.d_post_ffn_norm_2, D, mc.rms_norm_eps);
                }
                k_add_residual<<<(S * D + 255) / 256, 256, 0, stream>>>(d_ffn_d, d_moe_out, S * D);
            }

            if (gl.d_post_ffn_norm) {
                k_rms_norm_rows<<<S, 256, 0, stream>>>(d_ffn_d, d_ffn_d, gl.d_post_ffn_norm, D, mc.rms_norm_eps);
            }
            k_add_residual<<<(S * D + 255) / 256, 256, 0, stream>>>(d_x, d_ffn_d, S * D);

            // 2j. Optional Pleias Per-Layer Token Injection (Gemma 4.5 E2B / E4B)
            if (has_pleias && gl.inp_gate.valid() && gl.proj.valid()) {
                linear_forward(gl.inp_gate, nullptr, d_x, d_pl_gate, S);
                const int tot_p = S * 256;
                k_pleias_gate_mul<<<(tot_p + 255) / 256, 256, 0, stream>>>(d_pl_gate, d_pl_emb, S, L, l, 256);
                linear_forward(gl.proj, nullptr, d_pl_gate, d_pl_out, S);
                if (gl.d_post_norm) {
                    k_rms_norm_rows<<<S, 256, 0, stream>>>(d_pl_out, d_pl_out, gl.d_post_norm, D, mc.rms_norm_eps);
                }
                k_add_residual<<<(S * D + 255) / 256, 256, 0, stream>>>(d_x, d_pl_out, S * D);
            } else if (gl.d_post_norm) {
                k_rms_norm_rows<<<S, 256, 0, stream>>>(d_x, d_x, gl.d_post_norm, D, mc.rms_norm_eps);
            }

            if (std::fabs(gl.layer_output_scale - 1.0f) > 1e-6f) {
                k_scale_inplace<<<(S * D + 255) / 256, 256, 0, stream>>>(d_x, gl.layer_output_scale, S * D);
            }
        }

        // 3. Final Output RMSNorm + 262k LM Head ONLY for the last token in chunk (if requested!)
        if (out_last_logits || out_next_token) {
            const float* d_x_last = d_x + static_cast<size_t>(S - 1) * static_cast<size_t>(D);
            k_rms_norm_rows<<<1, 256, 0, stream>>>(d_norm, d_x_last, d_out_norm, D, mc.rms_norm_eps);
            linear_forward(lm_head, nullptr, d_norm, d_logits, 1);
            if (mc.final_logit_softcapping > 0.0f) {
                k_softcap_inplace<<<(V + 255) / 256, 256, 0, stream>>>(d_logits, V, mc.final_logit_softcapping);
            }
            if (out_next_token) {
                k_argmax_stage1<<<256, 256, 0, stream>>>(d_logits, V, d_am_vals, d_am_idx);
                k_argmax_stage2<<<1, 256, 0, stream>>>(d_am_vals, d_am_idx, 256, d_am_out);
                cudaMemcpyAsync(out_next_token, d_am_out, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream);
            }
            if (out_last_logits) {
                cudaMemcpyAsync(out_last_logits, d_logits, static_cast<size_t>(V) * sizeof(float), cudaMemcpyDeviceToHost, stream);
            }
        }

        cudaError_t sync_err = cudaStreamSynchronize(stream);
        if (sync_err != cudaSuccess) {
            err = std::string("CUDA execution error: ") + cudaGetErrorString(sync_err);
            return false;
        }
        return true;
    }

    int max_seq_len_runtime = 2048;

    void free_all() {
        if (stream) { cudaStreamSynchronize(stream); }
        for (void* p : all_allocs) {
            if (p) cudaFree(p);
        }
        all_allocs.clear();
        layers.clear();
        if (cublas) { cublasDestroy(cublas); cublas = nullptr; }
        if (stream) { cudaStreamDestroy(stream); stream = nullptr; }
    }
};

// ============================================================================
// Public HavenCudaEngine Interface
// ============================================================================

HavenCudaEngine::HavenCudaEngine() = default;
HavenCudaEngine::~HavenCudaEngine() { shutdown(); }

bool HavenCudaEngine::is_cuda_compiled() { return true; }

int HavenCudaEngine::get_device_count() {
    int cnt = 0;
    if (cudaGetDeviceCount(&cnt) != cudaSuccess) return 0;
    return cnt;
}

std::string HavenCudaEngine::get_device_summary(int device_id) {
    int cnt = get_device_count();
    if (cnt <= 0 || device_id < 0 || device_id >= cnt) return "No CUDA GPU detected";
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, device_id) != cudaSuccess) return "CUDA device query failed";
    std::ostringstream ss;
    ss << prop.name << " (sm_" << prop.major << prop.minor << ", "
       << (prop.totalGlobalMem / (1024ULL * 1024ULL)) << " MiB VRAM, "
       << prop.multiProcessorCount << " SMs)";
    return ss.str();
}

void HavenCudaEngine::shutdown() {
    if (impl_) {
        impl_->free_all();
        impl_.reset();
    }
    initialized_ = false;
}

bool HavenCudaEngine::initialize(const HavenEngine& host_engine, const CudaEngineConfig& cfg) {
    shutdown();
    last_error_.clear();

    const int n_devs = get_device_count();
    if (n_devs <= 0 || cfg.device_id < 0 || cfg.device_id >= n_devs) {
        last_error_ = "No usable NVIDIA CUDA GPU found (device_count=" + std::to_string(n_devs) + ").";
        return false;
    }
    if (cudaSetDevice(cfg.device_id) != cudaSuccess) {
        last_error_ = "cudaSetDevice failed.";
        return false;
    }

    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, cfg.device_id);
    cfg_ = cfg;
    metrics_ = {};
    metrics_.device_name = prop.name;
    metrics_.sm_major = prop.major;
    metrics_.sm_minor = prop.minor;
    metrics_.total_vram_bytes = prop.totalGlobalMem;

    impl_ = std::make_unique<Impl>();
    impl_->mc = host_engine.get_config();
    impl_->max_seq_len_runtime = std::max(64, cfg.max_seq_len);
    impl_->max_S = std::clamp(cfg.max_prefill_chunk, 1, 512);

    if (cudaStreamCreate(&impl_->stream) != cudaSuccess ||
        cublasCreate(&impl_->cublas) != CUBLAS_STATUS_SUCCESS ||
        cublasSetStream(impl_->cublas, impl_->stream) != CUBLAS_STATUS_SUCCESS) {
        last_error_ = "Failed to create CUDA stream or cuBLAS handle.";
        shutdown();
        return false;
    }
    cublasSetMathMode(impl_->cublas, CUBLAS_TENSOR_OP_MATH);

    const auto& mc = impl_->mc;
    const int L = static_cast<int>(mc.num_layers);
    const int D = static_cast<int>(mc.embedding_dim);
    const int V = static_cast<int>(mc.vocab_size);
    if (L <= 0 || D <= 0 || V <= 0) {
        last_error_ = "Host model config is empty or invalid.";
        shutdown();
        return false;
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    size_t weight_bytes = 0, kv_bytes = 0, scratch_bytes = 0;

    // 1. Upload Embedding, LM Head, Output Norm, Optional RoPE Freqs & Pleias Tables
    const TensorDesc* te_td = host_engine.get_token_embd_tensor();
    const TensorDesc* lm_td = host_engine.get_output_tensor();
    if (!lm_td) lm_td = te_td;
    impl_->tied_emb = (lm_td == te_td);
    if (!impl_->upload_raw_tensor(te_td, V, D, impl_->tok_emb, weight_bytes, last_error_)) { shutdown(); return false; }
    if (impl_->tied_emb) {
        impl_->lm_head = impl_->tok_emb;
    } else {
        if (!impl_->upload_raw_tensor(lm_td, V, D, impl_->lm_head, weight_bytes, last_error_)) { shutdown(); return false; }
    }
    impl_->d_out_norm = impl_->upload_f32_1d(host_engine.get_output_norm_tensor(), D, 1.0f, weight_bytes, last_error_);
    if (!impl_->d_out_norm) { shutdown(); return false; }

    if (const TensorDesc* rf = host_engine.get_rope_freqs_tensor()) {
        int rf_cnt = (rf->shape.size() > 0 && rf->shape[0] > 0) ? static_cast<int>(rf->shape[0]) : static_cast<int>(mc.head_dim / 2);
        impl_->d_rope_freqs = impl_->upload_f32_1d(rf, rf_cnt, 1.0f, weight_bytes, last_error_);
        impl_->rope_freqs_len = rf_cnt;
    }

    if (const TensorDesc* pl_tok = host_engine.get_per_layer_token_embd_tensor()) {
        impl_->has_pleias = true;
        if (!impl_->upload_raw_tensor(pl_tok, V, L * 256, impl_->pl_tok_emb, weight_bytes, last_error_)) { shutdown(); return false; }
        if (const TensorDesc* pl_mp = host_engine.get_per_layer_model_proj_tensor()) {
            if (!impl_->upload_raw_tensor(pl_mp, L * 256, D, impl_->pl_model_proj, weight_bytes, last_error_)) { shutdown(); return false; }
        }
        impl_->d_pl_proj_norm = impl_->upload_f32_1d(host_engine.get_per_layer_proj_norm_tensor(), 256, 1.0f, weight_bytes, last_error_);
    }

    // 2. Upload All L Transformer Layers (Raw GGUF Blocks -> VRAM, Zero Dequant Blowup!)
    impl_->layers.resize(static_cast<size_t>(L));
    size_t max_w_elems = static_cast<size_t>(D) * static_cast<size_t>(D);
    if (impl_->has_pleias) max_w_elems = std::max(max_w_elems, static_cast<size_t>(L * 256) * static_cast<size_t>(D));

    for (int l = 0; l < L; ++l) {
        const auto& lt = host_engine.get_layer_tensors(static_cast<uint32_t>(l));
        auto& gl = impl_->layers[static_cast<size_t>(l)];

        gl.F_hid = (lt.ffn_gate && lt.ffn_gate->shape.size() > 1)
            ? static_cast<int>(lt.ffn_gate->shape[1])
            : ((static_cast<size_t>(l) < mc.hidden_dim_per_layer.size() && mc.hidden_dim_per_layer[static_cast<size_t>(l)] > 0)
                ? static_cast<int>(mc.hidden_dim_per_layer[static_cast<size_t>(l)])
                : static_cast<int>(mc.hidden_dim));
        gl.q_dim = (lt.attn_q && lt.attn_q->shape.size() > 1)
            ? static_cast<int>(lt.attn_q->shape[1])
            : static_cast<int>(mc.num_heads * mc.head_dim);
        gl.head_dim = std::max(1, gl.q_dim / static_cast<int>(mc.num_heads));
        gl.n_kv_heads = (static_cast<size_t>(l) < mc.head_count_kv_per_layer.size() && mc.head_count_kv_per_layer[static_cast<size_t>(l)] > 0)
            ? static_cast<int>(mc.head_count_kv_per_layer[static_cast<size_t>(l)])
            : ((lt.attn_k && lt.attn_k->shape.size() > 1)
                ? std::max(1, static_cast<int>(lt.attn_k->shape[1]) / gl.head_dim)
                : static_cast<int>(mc.num_kv_heads));
        gl.kv_dim = (lt.attn_k && lt.attn_k->shape.size() > 1)
            ? static_cast<int>(lt.attn_k->shape[1])
            : (gl.n_kv_heads * gl.head_dim);

        const bool has_swa_model = (mc.sliding_window > 0) && (mc.is_gemma || mc.architecture == "mistral");
        gl.is_swa = (static_cast<size_t>(l) < mc.sliding_window_pattern.size())
            ? mc.sliding_window_pattern[static_cast<size_t>(l)]
            : (has_swa_model && (mc.architecture == "mistral" || (l % 6 != 5)));
        const bool is_shared_kv = (mc.shared_kv_layers > 0 && mc.shared_kv_layers < mc.num_layers &&
            static_cast<uint32_t>(l) >= (mc.num_layers - mc.shared_kv_layers));
        gl.has_kv = !is_shared_kv && (lt.attn_k != nullptr);
        gl.k_eq_v = gl.has_kv && (lt.attn_v == nullptr);

        if (gl.has_kv) {
            gl.kv_source_layer = l;
        } else if (mc.num_layers > 20 && mc.shared_kv_layers > 0 && mc.shared_kv_layers < mc.num_layers) {
            int n_kv_start = static_cast<int>(mc.num_layers - mc.shared_kv_layers);
            gl.kv_source_layer = std::max(0, n_kv_start - (gl.is_swa ? 2 : 1));
        } else {
            gl.kv_source_layer = std::max(0, l - 1);
        }

        impl_->max_F = std::max(impl_->max_F, gl.F_hid);
        impl_->max_q_dim = std::max(impl_->max_q_dim, gl.q_dim);
        impl_->max_kv_dim = std::max(impl_->max_kv_dim, gl.kv_dim);
        max_w_elems = std::max(max_w_elems, static_cast<size_t>(gl.F_hid) * static_cast<size_t>(D));
        max_w_elems = std::max(max_w_elems, static_cast<size_t>(gl.q_dim) * static_cast<size_t>(D));

        if (!impl_->upload_raw_tensor(lt.attn_q, gl.q_dim, D, gl.attn_q, weight_bytes, last_error_)) { shutdown(); return false; }
        if (gl.has_kv) {
            if (!impl_->upload_raw_tensor(lt.attn_k, gl.kv_dim, D, gl.attn_k, weight_bytes, last_error_)) { shutdown(); return false; }
            if (!gl.k_eq_v) {
                if (!impl_->upload_raw_tensor(lt.attn_v, gl.kv_dim, D, gl.attn_v, weight_bytes, last_error_)) { shutdown(); return false; }
            }
        }
        if (!impl_->upload_raw_tensor(lt.attn_output, D, gl.q_dim, gl.attn_output, weight_bytes, last_error_)) { shutdown(); return false; }

        if (lt.attn_q_bias) gl.d_attn_q_bias = impl_->upload_f32_1d(lt.attn_q_bias, gl.q_dim, 0.0f, weight_bytes, last_error_);
        if (gl.has_kv && lt.attn_k_bias) gl.d_attn_k_bias = impl_->upload_f32_1d(lt.attn_k_bias, gl.kv_dim, 0.0f, weight_bytes, last_error_);
        if (gl.has_kv && !gl.k_eq_v && lt.attn_v_bias) gl.d_attn_v_bias = impl_->upload_f32_1d(lt.attn_v_bias, gl.kv_dim, 0.0f, weight_bytes, last_error_);

        if (!mc.turbo_folded_norms) gl.d_attn_norm = impl_->upload_f32_1d(lt.attn_norm, D, 1.0f, weight_bytes, last_error_);
        if (lt.attn_q_norm) gl.d_attn_q_norm = impl_->upload_f32_1d(lt.attn_q_norm, gl.head_dim, 1.0f, weight_bytes, last_error_);
        if (gl.has_kv && lt.attn_k_norm) gl.d_attn_k_norm = impl_->upload_f32_1d(lt.attn_k_norm, gl.head_dim, 1.0f, weight_bytes, last_error_);
        if (lt.post_attention_norm) gl.d_post_attn_norm = impl_->upload_f32_1d(lt.post_attention_norm, D, 1.0f, weight_bytes, last_error_);

        if (!impl_->upload_raw_tensor(lt.ffn_gate, gl.F_hid, D, gl.ffn_gate, weight_bytes, last_error_)) { shutdown(); return false; }
        if (!impl_->upload_raw_tensor(lt.ffn_up,   gl.F_hid, D, gl.ffn_up,   weight_bytes, last_error_)) { shutdown(); return false; }
        if (!impl_->upload_raw_tensor(lt.ffn_down, D, gl.F_hid, gl.ffn_down, weight_bytes, last_error_)) { shutdown(); return false; }
        if (!mc.turbo_folded_norms) gl.d_ffn_norm = impl_->upload_f32_1d(lt.ffn_norm, D, 1.0f, weight_bytes, last_error_);
        if (lt.post_ffw_norm) gl.d_post_ffn_norm = impl_->upload_f32_1d(lt.post_ffw_norm, D, 1.0f, weight_bytes, last_error_);

        // Sparse MoE (Gemma 4.5 26B-A4B)
        if (mc.expert_count > 0 && lt.ffn_gate_inp && lt.ffn_down_exps &&
            (lt.ffn_gate_up_exps || (lt.ffn_gate_exps && lt.ffn_up_exps))) {
            gl.has_moe = true;
            const int E = static_cast<int>(mc.expert_count);
            const int F_exp = static_cast<int>(mc.expert_ffn_dim > 0 ? mc.expert_ffn_dim : mc.hidden_dim);
            if (!impl_->upload_raw_tensor(lt.ffn_gate_inp, E, D, gl.moe_router, weight_bytes, last_error_)) { shutdown(); return false; }
            if (lt.ffn_gate_up_exps) {
                gl.fused_moe_gu = true;
                if (!impl_->upload_raw_tensor(lt.ffn_gate_up_exps, E * 2 * F_exp, D, gl.moe_gu, weight_bytes, last_error_)) { shutdown(); return false; }
            } else {
                gl.fused_moe_gu = false;
                if (!impl_->upload_raw_tensor(lt.ffn_gate_exps, E * F_exp, D, gl.moe_g, weight_bytes, last_error_)) { shutdown(); return false; }
                if (!impl_->upload_raw_tensor(lt.ffn_up_exps,   E * F_exp, D, gl.moe_u, weight_bytes, last_error_)) { shutdown(); return false; }
            }
            if (!impl_->upload_raw_tensor(lt.ffn_down_exps, E * D, F_exp, gl.moe_dn, weight_bytes, last_error_)) { shutdown(); return false; }
            if (!mc.turbo_folded_scales && lt.ffn_gate_inp_scale) {
                gl.d_moe_rscale = impl_->upload_f32_1d(lt.ffn_gate_inp_scale, D, 1.0f, weight_bytes, last_error_);
            }
            if (!mc.turbo_folded_scales && lt.ffn_down_exps_scale) {
                gl.d_moe_down_scale = impl_->upload_f32_1d(lt.ffn_down_exps_scale, E, 1.0f, weight_bytes, last_error_);
            }
            if (lt.post_ffw_norm_1) gl.d_post_ffn_norm_1 = impl_->upload_f32_1d(lt.post_ffw_norm_1, D, 1.0f, weight_bytes, last_error_);
            if (!mc.turbo_folded_norms && lt.pre_ffw_norm_2) gl.d_pre_ffn_norm_2 = impl_->upload_f32_1d(lt.pre_ffw_norm_2, D, 1.0f, weight_bytes, last_error_);
            if (lt.post_ffw_norm_2) gl.d_post_ffn_norm_2 = impl_->upload_f32_1d(lt.post_ffw_norm_2, D, 1.0f, weight_bytes, last_error_);
        }

        // Pleias
        if (impl_->has_pleias && lt.inp_gate && lt.proj) {
            if (!impl_->upload_raw_tensor(lt.inp_gate, 256, D, gl.inp_gate, weight_bytes, last_error_)) { shutdown(); return false; }
            if (!impl_->upload_raw_tensor(lt.proj,     D, 256, gl.proj,     weight_bytes, last_error_)) { shutdown(); return false; }
        }
        if (lt.post_norm) gl.d_post_norm = impl_->upload_f32_1d(lt.post_norm, D, 1.0f, weight_bytes, last_error_);
        if (lt.layer_output_scale && lt.layer_output_scale->data && lt.layer_output_scale->type == QuantType::F32) {
            gl.layer_output_scale = *reinterpret_cast<const float*>(lt.layer_output_scale->data);
        }

        // Allocate FP16 KV cache only for KV-owning layers
        if (gl.has_kv) {
            gl.kv_bytes = static_cast<size_t>(impl_->max_seq_len_runtime) * static_cast<size_t>(gl.kv_dim) * sizeof(__half);
            gl.d_k_cache = static_cast<__half*>(impl_->alloc_gpu(gl.kv_bytes, kv_bytes));
            gl.d_v_cache = static_cast<__half*>(impl_->alloc_gpu(gl.kv_bytes, kv_bytes));
            if (!gl.d_k_cache || !gl.d_v_cache) {
                last_error_ = "cudaMalloc failed for KV cache at layer " + std::to_string(l);
                shutdown();
                return false;
            }
            cudaMemset(gl.d_k_cache, 0, gl.kv_bytes);
            cudaMemset(gl.d_v_cache, 0, gl.kv_bytes);
        }
    }

    // 3. Allocate Activation & FP16 Prefill Scratch Buffers
    const size_t Smax = static_cast<size_t>(impl_->max_S);
    auto alloc_f32 = [&](size_t elems) -> float* {
        return static_cast<float*>(impl_->alloc_gpu(elems * sizeof(float), scratch_bytes));
    };
    impl_->d_tok_ids = static_cast<uint32_t*>(impl_->alloc_gpu(Smax * sizeof(uint32_t), scratch_bytes));
    impl_->d_x       = alloc_f32(Smax * static_cast<size_t>(D));
    impl_->d_norm    = alloc_f32(Smax * static_cast<size_t>(D));
    impl_->d_q       = alloc_f32(Smax * static_cast<size_t>(impl_->max_q_dim));
    impl_->d_k       = alloc_f32(Smax * static_cast<size_t>(impl_->max_kv_dim));
    impl_->d_v       = alloc_f32(Smax * static_cast<size_t>(impl_->max_kv_dim));
    impl_->d_attn_h  = alloc_f32(Smax * static_cast<size_t>(impl_->max_q_dim));
    impl_->d_attn_o  = alloc_f32(Smax * static_cast<size_t>(D));
    impl_->d_ffn_g   = alloc_f32(Smax * static_cast<size_t>(impl_->max_F));
    impl_->d_ffn_u   = alloc_f32(Smax * static_cast<size_t>(impl_->max_F));
    impl_->d_ffn_d   = alloc_f32(Smax * static_cast<size_t>(D));
    impl_->d_logits  = alloc_f32(static_cast<size_t>(V));
    impl_->d_am_vals = alloc_f32(256);
    impl_->d_am_idx  = static_cast<uint32_t*>(impl_->alloc_gpu(256 * sizeof(uint32_t), scratch_bytes));
    impl_->d_am_out  = static_cast<uint32_t*>(impl_->alloc_gpu(sizeof(uint32_t), scratch_bytes));

    if (mc.expert_count > 0) {
        const int E = static_cast<int>(mc.expert_count);
        const int K_used = std::clamp(static_cast<int>(mc.expert_used_count > 0 ? mc.expert_used_count : 8u), 1, E);
        const int F_exp = static_cast<int>(mc.expert_ffn_dim > 0 ? mc.expert_ffn_dim : mc.hidden_dim);
        impl_->d_moe_norm   = alloc_f32(Smax * static_cast<size_t>(D));
        impl_->d_moe_router = alloc_f32(Smax * static_cast<size_t>(E));
        impl_->d_moe_idx    = static_cast<int*>(impl_->alloc_gpu(Smax * static_cast<size_t>(K_used) * sizeof(int), scratch_bytes));
        impl_->d_moe_w      = alloc_f32(Smax * static_cast<size_t>(K_used));
        impl_->d_moe_act    = alloc_f32(Smax * static_cast<size_t>(K_used) * static_cast<size_t>(F_exp));
        impl_->d_moe_out    = alloc_f32(Smax * static_cast<size_t>(D));
    }
    if (impl_->has_pleias) {
        impl_->d_pl_emb  = alloc_f32(Smax * static_cast<size_t>(L) * 256ULL);
        impl_->d_pl_proj = alloc_f32(Smax * static_cast<size_t>(L) * 256ULL);
        impl_->d_pl_gate = alloc_f32(Smax * 256ULL);
        impl_->d_pl_out  = alloc_f32(Smax * static_cast<size_t>(D));
    }

    const size_t max_in_dim = static_cast<size_t>(std::max({D, impl_->max_F, impl_->max_q_dim}));
    impl_->d_x_f16 = static_cast<__half*>(impl_->alloc_gpu(Smax * max_in_dim * sizeof(__half), scratch_bytes));
    // Cap reusable FP16 dequant tile to 256M elements (512 MB) so even 31B Dense fits easily in 24 GB VRAM
    impl_->w_f16_cap_elems = std::min(max_w_elems, static_cast<size_t>(256ULL * 1024ULL * 1024ULL));
    impl_->d_w_f16 = static_cast<__half*>(impl_->alloc_gpu(impl_->w_f16_cap_elems * sizeof(__half), scratch_bytes));

    if (!impl_->d_x || !impl_->d_logits || !impl_->d_w_f16) {
        last_error_ = "cudaMalloc failed allocating CUDA scratch buffers.";
        shutdown();
        return false;
    }

    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();

    metrics_.weight_vram_bytes = weight_bytes;
    metrics_.kv_vram_bytes = kv_bytes;
    metrics_.scratch_vram_bytes = scratch_bytes;
    metrics_.upload_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    initialized_ = true;
    return true;
}

bool HavenCudaEngine::reset_kv_cache() {
    if (!initialized_ || !impl_) return false;
    for (auto& gl : impl_->layers) {
        if (gl.has_kv && gl.kv_bytes > 0) {
            cudaMemsetAsync(gl.d_k_cache, 0, gl.kv_bytes, impl_->stream);
            cudaMemsetAsync(gl.d_v_cache, 0, gl.kv_bytes, impl_->stream);
        }
    }
    return cudaStreamSynchronize(impl_->stream) == cudaSuccess;
}

bool HavenCudaEngine::forward_decode(uint32_t token, int pos, float* out_logits, uint32_t* out_next_token) {
    if (!initialized_ || !impl_) {
        last_error_ = "HavenCudaEngine is not initialized.";
        return false;
    }
    auto t0 = std::chrono::high_resolution_clock::now();
    bool ok = impl_->run_forward_chunk(&token, 1, pos, out_logits, out_next_token, last_error_);
    auto t1 = std::chrono::high_resolution_clock::now();
    metrics_.last_decode_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return ok;
}

bool HavenCudaEngine::forward_prefill(const std::vector<uint32_t>& tokens,
                                      int start_pos,
                                      float* out_last_logits,
                                      uint32_t* out_next_token) {
    if (!initialized_ || !impl_ || tokens.empty()) {
        last_error_ = "HavenCudaEngine not initialized or empty prompt.";
        return false;
    }
    auto t0 = std::chrono::high_resolution_clock::now();
    const int total = static_cast<int>(tokens.size());
    int offset = 0;
    while (offset < total) {
        const int chunk = std::min(impl_->max_S, total - offset);
        const bool is_last = (offset + chunk == total);
        if (!impl_->run_forward_chunk(tokens.data() + offset, chunk, start_pos + offset,
                                      is_last ? out_last_logits : nullptr,
                                      is_last ? out_next_token : nullptr,
                                      last_error_)) {
            return false;
        }
        offset += chunk;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    metrics_.last_prefill_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return true;
}

} // namespace haven
