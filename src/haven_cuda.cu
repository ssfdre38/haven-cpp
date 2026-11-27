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
// Device Constants, Q8_1 Activation Format & Bit-Exact Helpers
// ============================================================================

__constant__ int8_t d_kvalues_iq4nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
};

// 16-byte aligned 32-element quantized activation block for __dp4a integer GEMV & grouped MoE
struct alignas(16) block_q8_1_gpu {
    int8_t qs[32];  // 32 int8 quantized activations (8 x int32 for __dp4a)
    float  d;       // scale = max_abs / 127.0f
    float  s_lo;    // exact float sum of x[0..15]
    float  s_hi;    // exact float sum of x[16..31]
    float  _pad;    // 48 bytes total (16-byte aligned)
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

__device__ __forceinline__ int32_t pack4_i8(int a, int b, int c, int d) {
    return (a & 0xFF) | ((b & 0xFF) << 8) | ((c & 0xFF) << 16) | ((d & 0xFF) << 24);
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
// Fast Q8_1 Activation Quantizer (1 warp per 32 floats -> 8 warps/CTA)
// ============================================================================
__global__ void k_quantize_q8_1(const float* __restrict__ x, block_q8_1_gpu* __restrict__ out, int total_blocks) {
    const int wid = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int b = (blockIdx.x << 3) + wid;
    if (b >= total_blocks) return;

    const float v = x[(b << 5) + lane];
    float amax = fabsf(v);
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, offset));
    }

    // Half-warp sums (lanes 0..15 -> s_lo, lanes 16..31 -> s_hi)
    float hsum = v;
    #pragma unroll
    for (int offset = 8; offset > 0; offset >>= 1) {
        hsum += __shfl_down_sync(0xFFFFFFFFu, hsum, offset);
    }

    const float d = amax * (1.0f / 127.0f);
    const float inv_d = (amax > 1e-30f) ? (127.0f / amax) : 0.0f;
    int q = __float2int_rn(v * inv_d);
    q = max(-127, min(127, q));

    block_q8_1_gpu* dst = out + b;
    dst->qs[lane] = static_cast<int8_t>(q);
    if (lane == 0) {
        dst->d = d;
        dst->s_lo = hsum;
    } else if (lane == 16) {
        dst->s_hi = hsum;
    }
}

// ============================================================================
// V2 Warp-Specialized __dp4a + __shfl_sync Quantized Dot Products
// ZERO __constant__ LUT serialization! 4x lower activation traffic via Q8_1!
// ============================================================================

__device__ __forceinline__ float warp_dot_iq4_xs_q8(const uint8_t* __restrict__ row_ptr,
                                                    const block_q8_1_gpu* __restrict__ x_q8,
                                                    int K, int lane) {
    // Hold 16-entry IQ4NL LUT directly in warp registers (1-cycle SHFL.IDX, zero bank/const stalls!)
    const int lut_i = static_cast<int>(d_kvalues_iq4nl[lane & 15]);
    const int nb = K >> 8;     // K / 256
    const int ib = lane >> 2;  // 0..7 sub-block within 256-elem super-block
    const int t  = lane & 3;   // 0..3 (4 bytes = 8 nibbles -> 128B coalesced per warp!)
    float acc = 0.0f;

    for (int b = 0; b < nb; ++b) {
        const block_iq4_xs* blk = reinterpret_cast<const block_iq4_xs*>(row_ptr + static_cast<size_t>(b) * sizeof(block_iq4_xs));
        const float d = d_fp16_to_f32(blk->d);
        const uint16_t sh = blk->scales_h;
        const uint8_t sl = blk->scales_l[ib >> 1];
        const int ls = ((sl >> (4 * (ib & 1))) & 0x0F) | (((sh >> (2 * ib)) & 3) << 4);
        const float dl = d * static_cast<float>(ls - 32);

        const uint32_t q4 = *reinterpret_cast<const uint32_t*>(blk->qs + (lane << 2));
        const int32_t w_lo = pack4_i8(
            __shfl_sync(0xFFFFFFFFu, lut_i, (q4 >>  0) & 0x0Fu),
            __shfl_sync(0xFFFFFFFFu, lut_i, (q4 >>  8) & 0x0Fu),
            __shfl_sync(0xFFFFFFFFu, lut_i, (q4 >> 16) & 0x0Fu),
            __shfl_sync(0xFFFFFFFFu, lut_i, (q4 >> 24) & 0x0Fu));
        const int32_t w_hi = pack4_i8(
            __shfl_sync(0xFFFFFFFFu, lut_i, (q4 >>  4) & 0x0Fu),
            __shfl_sync(0xFFFFFFFFu, lut_i, (q4 >> 12) & 0x0Fu),
            __shfl_sync(0xFFFFFFFFu, lut_i, (q4 >> 20) & 0x0Fu),
            __shfl_sync(0xFFFFFFFFu, lut_i, (q4 >> 28) & 0x0Fu));

        const block_q8_1_gpu& xb = x_q8[(b << 3) + ib];
        const int32_t* xq = reinterpret_cast<const int32_t*>(xb.qs);
        int sumi = __dp4a(w_lo, xq[t], 0);
        sumi     = __dp4a(w_hi, xq[t + 4], sumi);
        acc = fmaf(dl * xb.d, static_cast<float>(sumi), acc);
    }
    return warp_reduce_sum(acc);
}

__device__ __forceinline__ float warp_dot_iq4_nl_q8(const uint8_t* __restrict__ row_ptr,
                                                    const block_q8_1_gpu* __restrict__ x_q8,
                                                    int K, int lane) {
    const int lut_i = static_cast<int>(d_kvalues_iq4nl[lane & 15]);
    const int nb = K >> 5;       // K / 32
    const int b_off = lane >> 2; // 0..7 (8 blocks concurrently per warp!)
    const int t = lane & 3;      // 0..3 (4 bytes = 8 nibbles)
    float acc = 0.0f;

    // Loop with uniform warp trips so __shfl_sync always sees full active mask
    const int n_iters = (nb + 7) >> 3;
    for (int it = 0; it < n_iters; ++it) {
        const int b = (it << 3) + b_off;
        const bool active = (b < nb);
        uint32_t q4 = 0u;
        float d = 0.0f;
        if (active) {
            const block_iq4_nl* blk = reinterpret_cast<const block_iq4_nl*>(row_ptr + static_cast<size_t>(b) * sizeof(block_iq4_nl));
            d = d_fp16_to_f32(blk->d);
            memcpy(&q4, blk->qs + (t << 2), sizeof(uint32_t));
        }
        const int32_t w_lo = pack4_i8(
            __shfl_sync(0xFFFFFFFFu, lut_i, (q4 >>  0) & 0x0Fu),
            __shfl_sync(0xFFFFFFFFu, lut_i, (q4 >>  8) & 0x0Fu),
            __shfl_sync(0xFFFFFFFFu, lut_i, (q4 >> 16) & 0x0Fu),
            __shfl_sync(0xFFFFFFFFu, lut_i, (q4 >> 24) & 0x0Fu));
        const int32_t w_hi = pack4_i8(
            __shfl_sync(0xFFFFFFFFu, lut_i, (q4 >>  4) & 0x0Fu),
            __shfl_sync(0xFFFFFFFFu, lut_i, (q4 >> 12) & 0x0Fu),
            __shfl_sync(0xFFFFFFFFu, lut_i, (q4 >> 20) & 0x0Fu),
            __shfl_sync(0xFFFFFFFFu, lut_i, (q4 >> 28) & 0x0Fu));
        if (active) {
            const block_q8_1_gpu& xb = x_q8[b];
            const int32_t* xq = reinterpret_cast<const int32_t*>(xb.qs);
            int sumi = __dp4a(w_lo, xq[t], 0);
            sumi     = __dp4a(w_hi, xq[t + 4], sumi);
            acc = fmaf(d * xb.d, static_cast<float>(sumi), acc);
        }
    }
    return warp_reduce_sum(acc);
}

__device__ __forceinline__ float warp_dot_q6_k_q8(const uint8_t* __restrict__ row_ptr,
                                                  const block_q8_1_gpu* __restrict__ x_q8,
                                                  int K, int lane) {
    const int nb = K >> 8;
    const int grp = lane >> 3;        // 0..3 (q1, q2, q3, q4)
    const int l4  = (lane & 7) << 2;  // 0, 4, 8, ..., 28
    const int half16 = l4 >> 4;       // 0 or 1
    const int ql_off = ((grp & 1) << 5) + l4;
    const int qh_shift = grp << 1;
    const bool high_nib = (grp >= 2);
    float acc = 0.0f;

    for (int b = 0; b < nb; ++b) {
        const block_q6_K* blk = reinterpret_cast<const block_q6_K*>(row_ptr + static_cast<size_t>(b) * sizeof(block_q6_K));
        const float d = d_fp16_to_f32(blk->d);
        #pragma unroll
        for (int c = 0; c < 2; ++c) {
            const uint8_t* ql = blk->ql + (c << 6);
            const uint8_t* qh = blk->qh + (c << 5);
            const int8_t*  sc = blk->scales + (c << 3);

            uint32_t ql4, qh4;
            memcpy(&ql4, ql + ql_off, sizeof(uint32_t));
            memcpy(&qh4, qh + l4, sizeof(uint32_t));

            auto dec = [&](int shift_b) -> int {
                const uint32_t qb = (ql4 >> shift_b) & 0xFFu;
                const uint32_t hb = (qh4 >> shift_b) & 0xFFu;
                const uint32_t nib = high_nib ? (qb >> 4) : (qb & 0x0Fu);
                const uint32_t hi2 = (hb >> qh_shift) & 3u;
                return static_cast<int>(nib | (hi2 << 4)) - 32;
            };
            const int32_t w4 = pack4_i8(dec(0), dec(8), dec(16), dec(24));
            const block_q8_1_gpu& xb = x_q8[(b << 3) + (c << 2) + grp];
            const int32_t x4 = reinterpret_cast<const int32_t*>(xb.qs)[l4 >> 2];
            const int sumi = __dp4a(w4, x4, 0);
            acc = fmaf(d * static_cast<float>(sc[(grp << 1) + half16]) * xb.d, static_cast<float>(sumi), acc);
        }
    }
    return warp_reduce_sum(acc);
}

__device__ __forceinline__ float warp_dot_q5_k_q8(const uint8_t* __restrict__ row_ptr,
                                                  const block_q8_1_gpu* __restrict__ x_q8,
                                                  int K, int lane) {
    const int nb = K >> 8;
    const int j  = lane >> 3;        // 0..3 (all 4 64-elem groups in parallel!)
    const int l4 = (lane & 7) << 2;  // 0, 4, ..., 28
    const uint8_t u1 = static_cast<uint8_t>(1u << (2 * j));
    const uint8_t u2 = static_cast<uint8_t>(2u << (2 * j));
    float acc = 0.0f;

    for (int b = 0; b < nb; ++b) {
        const block_q5_K* blk = reinterpret_cast<const block_q5_K*>(row_ptr + static_cast<size_t>(b) * sizeof(block_q5_K));
        const float d = d_fp16_to_f32(blk->d);
        const float dmin = d_fp16_to_f32(blk->dmin);
        uint8_t sc1, m1, sc2, m2;
        d_get_scale_min_k4(2 * j,     blk->scales, sc1, m1);
        d_get_scale_min_k4(2 * j + 1, blk->scales, sc2, m2);

        const uint32_t ql4 = *reinterpret_cast<const uint32_t*>(blk->qs + (j << 5) + l4);
        const uint32_t qh4 = *reinterpret_cast<const uint32_t*>(blk->qh + l4);

        auto dlo = [&](int sb) -> int {
            return static_cast<int>(((ql4 >> sb) & 0x0Fu) + ((((qh4 >> sb) & 0xFFu) & u1) ? 16u : 0u));
        };
        auto dhi = [&](int sb) -> int {
            return static_cast<int>((((ql4 >> (sb + 4)) & 0x0Fu)) + ((((qh4 >> sb) & 0xFFu) & u2) ? 16u : 0u));
        };
        const int32_t w_lo = pack4_i8(dlo(0), dlo(8), dlo(16), dlo(24));
        const int32_t w_hi = pack4_i8(dhi(0), dhi(8), dhi(16), dhi(24));

        const block_q8_1_gpu& x0 = x_q8[(b << 3) + (j << 1) + 0];
        const block_q8_1_gpu& x1 = x_q8[(b << 3) + (j << 1) + 1];
        const int s0 = __dp4a(w_lo, reinterpret_cast<const int32_t*>(x0.qs)[l4 >> 2], 0);
        const int s1 = __dp4a(w_hi, reinterpret_cast<const int32_t*>(x1.qs)[l4 >> 2], 0);
        acc = fmaf(d * static_cast<float>(sc1) * x0.d, static_cast<float>(s0), acc);
        acc = fmaf(d * static_cast<float>(sc2) * x1.d, static_cast<float>(s1), acc);
        if ((lane & 7) == 0) {
            acc = fmaf(-dmin * static_cast<float>(m1), x0.s_lo + x0.s_hi, acc);
            acc = fmaf(-dmin * static_cast<float>(m2), x1.s_lo + x1.s_hi, acc);
        }
    }
    return warp_reduce_sum(acc);
}

__device__ __forceinline__ float warp_dot_q4_k_q8(const uint8_t* __restrict__ row_ptr,
                                                  const block_q8_1_gpu* __restrict__ x_q8,
                                                  int K, int lane) {
    const int nb = K >> 8;
    const int j  = lane >> 3;
    const int l4 = (lane & 7) << 2;
    float acc = 0.0f;

    for (int b = 0; b < nb; ++b) {
        const block_q4_K* blk = reinterpret_cast<const block_q4_K*>(row_ptr + static_cast<size_t>(b) * sizeof(block_q4_K));
        const float d = d_fp16_to_f32(blk->d);
        const float dmin = d_fp16_to_f32(blk->dmin);
        uint8_t sc1, m1, sc2, m2;
        d_get_scale_min_k4(2 * j,     blk->scales, sc1, m1);
        d_get_scale_min_k4(2 * j + 1, blk->scales, sc2, m2);

        const uint32_t ql4 = *reinterpret_cast<const uint32_t*>(blk->qs + (j << 5) + l4);
        const int32_t w_lo = static_cast<int32_t>(ql4 & 0x0F0F0F0Fu);
        const int32_t w_hi = static_cast<int32_t>((ql4 >> 4) & 0x0F0F0F0Fu);

        const block_q8_1_gpu& x0 = x_q8[(b << 3) + (j << 1) + 0];
        const block_q8_1_gpu& x1 = x_q8[(b << 3) + (j << 1) + 1];
        const int s0 = __dp4a(w_lo, reinterpret_cast<const int32_t*>(x0.qs)[l4 >> 2], 0);
        const int s1 = __dp4a(w_hi, reinterpret_cast<const int32_t*>(x1.qs)[l4 >> 2], 0);
        acc = fmaf(d * static_cast<float>(sc1) * x0.d, static_cast<float>(s0), acc);
        acc = fmaf(d * static_cast<float>(sc2) * x1.d, static_cast<float>(s1), acc);
        if ((lane & 7) == 0) {
            acc = fmaf(-dmin * static_cast<float>(m1), x0.s_lo + x0.s_hi, acc);
            acc = fmaf(-dmin * static_cast<float>(m2), x1.s_lo + x1.s_hi, acc);
        }
    }
    return warp_reduce_sum(acc);
}

__device__ __forceinline__ float warp_dot_q5_1_q8(const uint8_t* __restrict__ row_ptr,
                                                  const block_q8_1_gpu* __restrict__ x_q8,
                                                  int K, int lane) {
    const int nb = K >> 5;
    const int b_off = lane >> 2; // 8 blocks concurrently per warp
    const int t = lane & 3;
    float acc = 0.0f;

    for (int b = b_off; b < nb; b += 8) {
        const block_q5_1* blk = reinterpret_cast<const block_q5_1*>(row_ptr + static_cast<size_t>(b) * sizeof(block_q5_1));
        const float d = d_fp16_to_f32(blk->d);
        const float m = d_fp16_to_f32(blk->m);
        uint32_t qh, q4;
        memcpy(&qh, blk->qh, sizeof(uint32_t));
        memcpy(&q4, blk->qs + (t << 2), sizeof(uint32_t));
        const int j0 = t << 2;
        auto xlo = [&](int r) -> int {
            return static_cast<int>(((q4 >> (r * 8)) & 0x0Fu) | (((qh >> (j0 + r)) & 1u) << 4));
        };
        auto xhi = [&](int r) -> int {
            return static_cast<int>(((q4 >> (r * 8 + 4)) & 0x0Fu) | (((qh >> (j0 + r + 16)) & 1u) << 4));
        };
        const int32_t w_lo = pack4_i8(xlo(0), xlo(1), xlo(2), xlo(3));
        const int32_t w_hi = pack4_i8(xhi(0), xhi(1), xhi(2), xhi(3));
        const block_q8_1_gpu& xb = x_q8[b];
        const int32_t* xq = reinterpret_cast<const int32_t*>(xb.qs);
        int sumi = __dp4a(w_lo, xq[t], 0);
        sumi     = __dp4a(w_hi, xq[t + 4], sumi);
        acc = fmaf(d * xb.d, static_cast<float>(sumi), acc);
        if (t == 0) {
            acc = fmaf(m, xb.s_lo + xb.s_hi, acc);
        }
    }
    return warp_reduce_sum(acc);
}

__device__ __forceinline__ float warp_dot_q8_0_q8(const uint8_t* __restrict__ row_ptr,
                                                  const block_q8_1_gpu* __restrict__ x_q8,
                                                  int K, int lane) {
    const int nb = K >> 5;
    const int b_off = lane >> 2;
    const int t = lane & 3;
    float acc = 0.0f;
    for (int b = b_off; b < nb; b += 8) {
        const block_q8_0* blk = reinterpret_cast<const block_q8_0*>(row_ptr + static_cast<size_t>(b) * sizeof(block_q8_0));
        const float d = d_fp16_to_f32(blk->d);
        int32_t w0, w1;
        memcpy(&w0, blk->qs + (t << 2), sizeof(int32_t));
        memcpy(&w1, blk->qs + 16 + (t << 2), sizeof(int32_t));
        const block_q8_1_gpu& xb = x_q8[b];
        const int32_t* xq = reinterpret_cast<const int32_t*>(xb.qs);
        int sumi = __dp4a(w0, xq[t], 0);
        sumi     = __dp4a(w1, xq[t + 4], sumi);
        acc = fmaf(d * xb.d, static_cast<float>(sumi), acc);
    }
    return warp_reduce_sum(acc);
}

// Exact F32 fallback dot products for rare / unquantized formats (Q5_0, Q4_0, Q3_K, BF16, F16, F32)
__device__ __forceinline__ float warp_dot_q5_0(const uint8_t* __restrict__ row_ptr, const float* __restrict__ x, int K, int lane) {
    const int nb = K >> 5;
    const int b_off = lane >> 4;
    const int j = lane & 15;
    float acc = 0.0f;
    for (int b = b_off; b < nb; b += 2) {
        const block_q5_0* blk = reinterpret_cast<const block_q5_0*>(row_ptr + static_cast<size_t>(b) * sizeof(block_q5_0));
        const float d = d_fp16_to_f32(blk->d);
        uint32_t qh; memcpy(&qh, blk->qh, sizeof(uint32_t));
        const uint8_t q = blk->qs[j];
        const int x0 = static_cast<int>((q & 0x0Fu) | (((qh >> j) & 1u) << 4)) - 16;
        const int x1 = static_cast<int>((q >> 4)    | (((qh >> (j + 16)) & 1u) << 4)) - 16;
        const float* xb = x + (b << 5);
        acc = fmaf(d * static_cast<float>(x0), xb[j],      acc);
        acc = fmaf(d * static_cast<float>(x1), xb[j + 16], acc);
    }
    return warp_reduce_sum(acc);
}

__device__ __forceinline__ float warp_dot_q4_0(const uint8_t* __restrict__ row_ptr, const float* __restrict__ x, int K, int lane) {
    const int nb = K >> 5;
    const int b_off = lane >> 4;
    const int j = lane & 15;
    float acc = 0.0f;
    for (int b = b_off; b < nb; b += 2) {
        const block_q4_0* blk = reinterpret_cast<const block_q4_0*>(row_ptr + static_cast<size_t>(b) * sizeof(block_q4_0));
        const float d = d_fp16_to_f32(blk->d);
        const uint8_t q = blk->qs[j];
        const float* xb = x + (b << 5);
        // GGUF split-half Q4_0 layout: low nibbles -> [0..15], high nibbles -> [16..31]
        acc = fmaf(d * static_cast<float>(static_cast<int>(q & 0x0Fu) - 8), xb[j], acc);
        acc = fmaf(d * static_cast<float>(static_cast<int>(q >> 4)    - 8), xb[j + 16], acc);
    }
    return warp_reduce_sum(acc);
}

__device__ __forceinline__ float warp_dot_q3_k(const uint8_t* __restrict__ row_ptr, const float* __restrict__ x, int K, int lane) {
    const int nb = K >> 8;
    float acc = 0.0f;
    for (int b = 0; b < nb; ++b) {
        const block_q3_K* blk = reinterpret_cast<const block_q3_K*>(row_ptr + static_cast<size_t>(b) * sizeof(block_q3_K));
        const float d_all = d_fp16_to_f32(blk->d);
        uint32_t aux[4];
        memcpy(aux, blk->scales, 12);
        const uint32_t kmask1 = 0x03030303u, kmask2 = 0x0f0f0f0fu, tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        const int8_t* sc = reinterpret_cast<const int8_t*>(aux);
        const float* xb = x + (b << 8);
        const int is_base = lane >> 4;
        #pragma unroll
        for (int c = 0; c < 2; ++c) {
            const uint8_t qb = blk->qs[(c << 5) + lane];
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

__device__ __forceinline__ float warp_dot_bf16(const uint8_t* __restrict__ row_ptr, const float* __restrict__ x, int K, int lane) {
    const uint16_t* w = reinterpret_cast<const uint16_t*>(row_ptr);
    float acc = 0.0f;
    const int K2 = K >> 1;
    const uint32_t* w2 = reinterpret_cast<const uint32_t*>(w);
    const float2* x2 = reinterpret_cast<const float2*>(x);
    for (int i = lane; i < K2; i += 32) {
        const uint32_t pair = w2[i];
        const float2 xv = x2[i];
        acc = fmaf(d_bf16_to_f32(static_cast<uint16_t>(pair & 0xFFFFu)), xv.x, acc);
        acc = fmaf(d_bf16_to_f32(static_cast<uint16_t>(pair >> 16)),      xv.y, acc);
    }
    return warp_reduce_sum(acc);
}

__device__ __forceinline__ float warp_dot_f16(const uint8_t* __restrict__ row_ptr, const float* __restrict__ x, int K, int lane) {
    const uint16_t* w = reinterpret_cast<const uint16_t*>(row_ptr);
    float acc = 0.0f;
    for (int i = lane; i < K; i += 32) {
        acc = fmaf(d_fp16_to_f32(w[i]), x[i], acc);
    }
    return warp_reduce_sum(acc);
}

__device__ __forceinline__ float warp_dot_f32(const uint8_t* __restrict__ row_ptr, const float* __restrict__ x, int K, int lane) {
    const float* w = reinterpret_cast<const float*>(row_ptr);
    float acc = 0.0f;
    const int K4 = K >> 2;
    const float4* w4 = reinterpret_cast<const float4*>(w);
    const float4* x4 = reinterpret_cast<const float4*>(x);
    for (int i = lane; i < K4; i += 32) {
        float4 a = w4[i], b = x4[i];
        acc = fmaf(a.x, b.x, acc);
        acc = fmaf(a.y, b.y, acc);
        acc = fmaf(a.z, b.z, acc);
        acc = fmaf(a.w, b.w, acc);
    }
    for (int i = (K4 << 2) + lane; i < K; i += 32) acc = fmaf(w[i], x[i], acc);
    return warp_reduce_sum(acc);
}

__device__ __forceinline__ float warp_dot_any_q8(const uint8_t* __restrict__ row_ptr,
                                                 QuantType qtype,
                                                 const block_q8_1_gpu* __restrict__ x_q8,
                                                 const float* __restrict__ x_f32,
                                                 int K, int lane) {
    switch (qtype) {
        case QuantType::IQ4_XS: return warp_dot_iq4_xs_q8(row_ptr, x_q8, K, lane);
        case QuantType::IQ4_NL: return warp_dot_iq4_nl_q8(row_ptr, x_q8, K, lane);
        case QuantType::Q6_K:   return warp_dot_q6_k_q8(row_ptr, x_q8, K, lane);
        case QuantType::Q5_K:   return warp_dot_q5_k_q8(row_ptr, x_q8, K, lane);
        case QuantType::Q4_K:   return warp_dot_q4_k_q8(row_ptr, x_q8, K, lane);
        case QuantType::Q5_1:   return warp_dot_q5_1_q8(row_ptr, x_q8, K, lane);
        case QuantType::Q8_0:   return warp_dot_q8_0_q8(row_ptr, x_q8, K, lane);
        case QuantType::Q5_0:   return warp_dot_q5_0(row_ptr, x_f32, K, lane);
        case QuantType::Q4_0:   return warp_dot_q4_0(row_ptr, x_f32, K, lane);
        case QuantType::Q3_K:   return warp_dot_q3_k(row_ptr, x_f32, K, lane);
        case QuantType::BF16:   return warp_dot_bf16(row_ptr, x_f32, K, lane);
        case QuantType::F16:    return warp_dot_f16(row_ptr, x_f32, K, lane);
        case QuantType::F32:    return warp_dot_f32(row_ptr, x_f32, K, lane);
        default: return 0.0f;
    }
}

// ============================================================================
// Core GEMV & Fused GEMV Kernels (4 warps = 4 rows per 128-thread block)
// ============================================================================

__global__ void k_gemv_quant_q8(const uint8_t* __restrict__ W, QuantType qtype, size_t row_bytes,
                                const block_q8_1_gpu* __restrict__ x_q8, const float* __restrict__ x_f32,
                                const float* __restrict__ bias,
                                float* __restrict__ y, int N, int K, float scale_out) {
    const int warp_id = threadIdx.x >> 5;
    const int lane    = threadIdx.x & 31;
    const int row     = (blockIdx.x << 2) + warp_id;
    if (row >= N) return;
    const uint8_t* rp = W + static_cast<size_t>(row) * row_bytes;
    float sum = warp_dot_any_q8(rp, qtype, x_q8, x_f32, K, lane);
    if (lane == 0) {
        if (bias) sum += bias[row];
        y[row] = sum * scale_out;
    }
}

__global__ void k_fused_qkv_gemv_q8(const uint8_t* __restrict__ Wq, QuantType qt_q, size_t rb_q, const float* __restrict__ bq,
                                    const uint8_t* __restrict__ Wk, QuantType qt_k, size_t rb_k, const float* __restrict__ bk,
                                    const uint8_t* __restrict__ Wv, QuantType qt_v, size_t rb_v, const float* __restrict__ bv,
                                    const block_q8_1_gpu* __restrict__ x_q8, const float* __restrict__ x_f32,
                                    float* __restrict__ q_out, float* __restrict__ k_out, float* __restrict__ v_out,
                                    int q_dim, int k_dim, int v_dim, int D, bool k_eq_v) {
    const int warp_id = threadIdx.x >> 5;
    const int lane    = threadIdx.x & 31;
    const int idx     = (blockIdx.x << 2) + warp_id;
    const int total   = q_dim + k_dim + v_dim;
    if (idx >= total) return;

    if (idx < q_dim) {
        float s = warp_dot_any_q8(Wq + static_cast<size_t>(idx) * rb_q, qt_q, x_q8, x_f32, D, lane);
        if (lane == 0) q_out[idx] = bq ? (s + bq[idx]) : s;
    } else if (idx < q_dim + k_dim) {
        const int r = idx - q_dim;
        float s = warp_dot_any_q8(Wk + static_cast<size_t>(r) * rb_k, qt_k, x_q8, x_f32, D, lane);
        if (lane == 0) {
            float val = bk ? (s + bk[r]) : s;
            k_out[r] = val;
            if (k_eq_v && v_out) v_out[r] = val;
        }
    } else {
        const int r = idx - q_dim - k_dim;
        float s = warp_dot_any_q8(Wv + static_cast<size_t>(r) * rb_v, qt_v, x_q8, x_f32, D, lane);
        if (lane == 0) v_out[r] = bv ? (s + bv[r]) : s;
    }
}

__global__ void k_fused_ffn_gu_act_q8(const uint8_t* __restrict__ Wg, QuantType qt_g, size_t rb_g,
                                      const uint8_t* __restrict__ Wu, QuantType qt_u, size_t rb_u,
                                      const block_q8_1_gpu* __restrict__ x_q8, const float* __restrict__ x_f32,
                                      float* __restrict__ h_out, int F, int D, bool use_gelu) {
    const int warp_id = threadIdx.x >> 5;
    const int lane    = threadIdx.x & 31;
    const int row     = (blockIdx.x << 2) + warp_id;
    if (row >= F) return;
    const float g = warp_dot_any_q8(Wg + static_cast<size_t>(row) * rb_g, qt_g, x_q8, x_f32, D, lane);
    const float u = warp_dot_any_q8(Wu + static_cast<size_t>(row) * rb_u, qt_u, x_q8, x_f32, D, lane);
    if (lane == 0) {
        const float act = use_gelu ? d_gelu_tanh(g) : d_silu(g);
        h_out[row] = act * u;
    }
}

// ============================================================================
// Dequantize Raw GGUF Tensor Rows -> FP16 Matrix (Zero-Stall __shfl_sync LUT!)
// ============================================================================

__global__ void k_dequant_to_f16(const uint8_t* __restrict__ W, QuantType qtype, size_t row_bytes,
                                 __half* __restrict__ out_f16, int N, int K) {
    const int row = blockIdx.y;
    if (row >= N) return;
    const uint8_t* rp = W + static_cast<size_t>(row) * row_bytes;
    __half* dst = out_f16 + static_cast<size_t>(row) * static_cast<size_t>(K);
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (qtype == QuantType::IQ4_XS) {
        const int lut_i = static_cast<int>(d_kvalues_iq4nl[threadIdx.x & 15]);
        const int nb = K >> 8;
        const int b = tid >> 7;
        const bool active = (b < nb);
        const int loc = tid & 127;
        const int ib = loc >> 4;
        const int j  = loc & 15;
        float dl = 0.0f;
        uint8_t q = 0u;
        if (active) {
            const block_iq4_xs* blk = reinterpret_cast<const block_iq4_xs*>(rp + static_cast<size_t>(b) * sizeof(block_iq4_xs));
            const float d = d_fp16_to_f32(blk->d);
            const int ls = ((blk->scales_l[ib >> 1] >> (4 * (ib & 1))) & 0x0F) | (((blk->scales_h >> (2 * ib)) & 3) << 4);
            dl = d * static_cast<float>(ls - 32);
            q = blk->qs[(ib << 4) + j];
        }
        const int w0 = __shfl_sync(0xFFFFFFFFu, lut_i, q & 0x0Fu);
        const int w1 = __shfl_sync(0xFFFFFFFFu, lut_i, q >> 4);
        if (active) {
            dst[(b << 8) + (ib << 5) + j]      = __float2half(dl * static_cast<float>(w0));
            dst[(b << 8) + (ib << 5) + 16 + j] = __float2half(dl * static_cast<float>(w1));
        }
    } else if (qtype == QuantType::IQ4_NL) {
        const int lut_i = static_cast<int>(d_kvalues_iq4nl[threadIdx.x & 15]);
        const int nb = K >> 5;
        const int b = tid >> 4;
        const bool active = (b < nb);
        const int j = tid & 15;
        float d = 0.0f;
        uint8_t q = 0u;
        if (active) {
            const block_iq4_nl* blk = reinterpret_cast<const block_iq4_nl*>(rp + static_cast<size_t>(b) * sizeof(block_iq4_nl));
            d = d_fp16_to_f32(blk->d);
            q = blk->qs[j];
        }
        const int w0 = __shfl_sync(0xFFFFFFFFu, lut_i, q & 0x0Fu);
        const int w1 = __shfl_sync(0xFFFFFFFFu, lut_i, q >> 4);
        if (active) {
            dst[(b << 5) + j]      = __float2half(d * static_cast<float>(w0));
            dst[(b << 5) + 16 + j] = __float2half(d * static_cast<float>(w1));
        }
    } else if (qtype == QuantType::Q6_K) {
        const int nb = K >> 8;
        const int b = tid >> 6;
        if (b >= nb) return;
        const int loc = tid & 63;
        const int c = loc >> 5, l = loc & 31, is = l >> 4;
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
        const int b = tid >> 7;
        if (b >= nb) return;
        const int loc = tid & 127, j = loc >> 5, l = loc & 31;
        const block_q5_K* blk = reinterpret_cast<const block_q5_K*>(rp + static_cast<size_t>(b) * sizeof(block_q5_K));
        const float d = d_fp16_to_f32(blk->d), dmin = d_fp16_to_f32(blk->dmin);
        uint8_t sc1, m1, sc2, m2;
        d_get_scale_min_k4(2 * j,     blk->scales, sc1, m1);
        d_get_scale_min_k4(2 * j + 1, blk->scales, sc2, m2);
        const uint8_t ql = blk->qs[(j << 5) + l], qhb = blk->qh[l];
        const uint8_t u1 = static_cast<uint8_t>(1u << (2 * j)), u2 = static_cast<uint8_t>(2u << (2 * j));
        const int base = (b << 8) + (j << 6) + l;
        dst[base]      = __float2half((d * static_cast<float>(sc1)) * static_cast<float>((ql & 0x0Fu) + ((qhb & u1) ? 16 : 0)) - dmin * static_cast<float>(m1));
        dst[base + 32] = __float2half((d * static_cast<float>(sc2)) * static_cast<float>((ql >> 4)    + ((qhb & u2) ? 16 : 0)) - dmin * static_cast<float>(m2));
    } else if (qtype == QuantType::Q4_K) {
        const int nb = K >> 8;
        const int b = tid >> 7;
        if (b >= nb) return;
        const int loc = tid & 127, j = loc >> 5, l = loc & 31;
        const block_q4_K* blk = reinterpret_cast<const block_q4_K*>(rp + static_cast<size_t>(b) * sizeof(block_q4_K));
        const float d = d_fp16_to_f32(blk->d), dmin = d_fp16_to_f32(blk->dmin);
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
        const float d = d_fp16_to_f32(blk->d), m = d_fp16_to_f32(blk->m);
        uint32_t qh; memcpy(&qh, blk->qh, sizeof(uint32_t));
        const uint8_t q = blk->qs[j];
        dst[(b << 5) + j]      = __float2half(fmaf(d, static_cast<float>((q & 0x0Fu) | (((qh >> j) & 1u) << 4)), m));
        dst[(b << 5) + 16 + j] = __float2half(fmaf(d, static_cast<float>((q >> 4)    | (((qh >> (j + 16)) & 1u) << 4)), m));
    } else if (qtype == QuantType::Q5_0) {
        const int nb = K >> 5;
        const int b = tid >> 4;
        if (b >= nb) return;
        const int j = tid & 15;
        const block_q5_0* blk = reinterpret_cast<const block_q5_0*>(rp + static_cast<size_t>(b) * sizeof(block_q5_0));
        const float d = d_fp16_to_f32(blk->d);
        uint32_t qh; memcpy(&qh, blk->qh, sizeof(uint32_t));
        const uint8_t q = blk->qs[j];
        dst[(b << 5) + j]      = __float2half(d * static_cast<float>(static_cast<int>((q & 0x0Fu) | (((qh >> j) & 1u) << 4)) - 16));
        dst[(b << 5) + 16 + j] = __float2half(d * static_cast<float>(static_cast<int>((q >> 4)    | (((qh >> (j + 16)) & 1u) << 4)) - 16));
    } else if (qtype == QuantType::Q8_0) {
        if (tid >= K) return;
        const block_q8_0* blk = reinterpret_cast<const block_q8_0*>(rp + static_cast<size_t>(tid >> 5) * sizeof(block_q8_0));
        dst[tid] = __float2half(d_fp16_to_f32(blk->d) * static_cast<float>(blk->qs[tid & 31]));
    } else if (qtype == QuantType::Q4_0) {
        const int nb = K >> 5;
        const int b = tid >> 4;
        if (b >= nb) return;
        const int j = tid & 15;
        const block_q4_0* blk = reinterpret_cast<const block_q4_0*>(rp + static_cast<size_t>(b) * sizeof(block_q4_0));
        const float d = d_fp16_to_f32(blk->d);
        const uint8_t q = blk->qs[j];
        dst[(b << 5) + j]      = __float2half(d * static_cast<float>(static_cast<int>(q & 0x0Fu) - 8));
        dst[(b << 5) + 16 + j] = __float2half(d * static_cast<float>(static_cast<int>(q >> 4)    - 8));
    } else if (qtype == QuantType::Q3_K) {
        const int nb = K >> 8;
        const int b = tid >> 5;
        if (b >= nb) return;
        const int lane = tid & 31;
        const block_q3_K* blk = reinterpret_cast<const block_q3_K*>(rp + static_cast<size_t>(b) * sizeof(block_q3_K));
        const float d_all = d_fp16_to_f32(blk->d);
        uint32_t aux[4]; memcpy(aux, blk->scales, 12);
        const uint32_t kmask1 = 0x03030303u, kmask2 = 0x0f0f0f0fu, tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        const int8_t* sc = reinterpret_cast<const int8_t*>(aux);
        const int is_base = lane >> 4;
        for (int c = 0; c < 2; ++c) {
            const uint8_t qb = blk->qs[(c << 5) + lane], hmb = blk->hmask[lane];
            for (int j = 0; j < 4; ++j) {
                const uint8_t m = static_cast<uint8_t>(1u << (c * 4 + j));
                const float dl = d_all * static_cast<float>(sc[c * 8 + j * 2 + is_base] - 32);
                const int w = static_cast<int>((qb >> (2 * j)) & 3u) - ((hmb & m) ? 0 : 4);
                dst[(b << 8) + (c << 7) + (j << 5) + lane] = __float2half(dl * static_cast<float>(w));
            }
        }
    } else if (qtype == QuantType::BF16) {
        if (tid >= K) return;
        dst[tid] = __float2half(d_bf16_to_f32(reinterpret_cast<const uint16_t*>(rp)[tid]));
    } else if (qtype == QuantType::F16) {
        if (tid >= K) return;
        dst[tid] = reinterpret_cast<const __half*>(rp)[tid];
    } else if (qtype == QuantType::F32) {
        if (tid >= K) return;
        dst[tid] = __float2half(reinterpret_cast<const float*>(rp)[tid]);
    }
}

// Direct single-row dequantize to F32 (for embedding lookup & Pleias token table lookup on GPU)
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
        const int loc = tid & 63, c = loc >> 5, l = loc & 31, is = l >> 4;
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
        const int lut_i = static_cast<int>(d_kvalues_iq4nl[threadIdx.x & 15]);
        const int nb = K >> 8;
        const int b = tid >> 7;
        const bool active = (b < nb);
        const int loc = tid & 127, ib = loc >> 4, j = loc & 15;
        float dl = 0.0f; uint8_t q = 0u;
        if (active) {
            const block_iq4_xs* blk = reinterpret_cast<const block_iq4_xs*>(rp + static_cast<size_t>(b) * sizeof(block_iq4_xs));
            const float d = d_fp16_to_f32(blk->d) * scale;
            const int ls = ((blk->scales_l[ib >> 1] >> (4 * (ib & 1))) & 0x0F) | (((blk->scales_h >> (2 * ib)) & 3) << 4);
            dl = d * static_cast<float>(ls - 32);
            q = blk->qs[(ib << 4) + j];
        }
        const int w0 = __shfl_sync(0xFFFFFFFFu, lut_i, q & 0x0Fu);
        const int w1 = __shfl_sync(0xFFFFFFFFu, lut_i, q >> 4);
        if (active) {
            dst[(b << 8) + (ib << 5) + j]      = dl * static_cast<float>(w0);
            dst[(b << 8) + (ib << 5) + 16 + j] = dl * static_cast<float>(w1);
        }
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
        const int lut_i = static_cast<int>(d_kvalues_iq4nl[threadIdx.x & 15]);
        const int nb = K >> 5;
        const int b = tid >> 4;
        const bool active = (b < nb);
        const int j = tid & 15;
        float d = 0.0f; uint8_t q = 0u;
        if (active) {
            const block_iq4_nl* blk = reinterpret_cast<const block_iq4_nl*>(rp + static_cast<size_t>(b) * sizeof(block_iq4_nl));
            d = d_fp16_to_f32(blk->d) * scale;
            q = blk->qs[j];
        }
        const int w0 = __shfl_sync(0xFFFFFFFFu, lut_i, q & 0x0Fu);
        const int w1 = __shfl_sync(0xFFFFFFFFu, lut_i, q >> 4);
        if (active) {
            dst[(b << 5) + j]      = d * static_cast<float>(w0);
            dst[(b << 5) + 16 + j] = d * static_cast<float>(w1);
        }
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
    } else if (qtype == QuantType::Q5_0) {
        const int nb = K >> 5;
        const int b = tid >> 4;
        if (b >= nb) return;
        const int j = tid & 15;
        const block_q5_0* blk = reinterpret_cast<const block_q5_0*>(rp + static_cast<size_t>(b) * sizeof(block_q5_0));
        const float d = d_fp16_to_f32(blk->d) * scale;
        uint32_t qh; memcpy(&qh, blk->qh, sizeof(uint32_t));
        const uint8_t q = blk->qs[j];
        dst[(b << 5) + j]      = d * static_cast<float>(static_cast<int>((q & 0x0Fu) | (((qh >> j) & 1u) << 4)) - 16);
        dst[(b << 5) + 16 + j] = d * static_cast<float>(static_cast<int>((q >> 4)    | (((qh >> (j + 16)) & 1u) << 4)) - 16);
    } else if (qtype == QuantType::Q4_0) {
        const int nb = K >> 5;
        const int b = tid >> 4;
        if (b >= nb) return;
        const int j = tid & 15;
        const block_q4_0* blk = reinterpret_cast<const block_q4_0*>(rp + static_cast<size_t>(b) * sizeof(block_q4_0));
        const float d = d_fp16_to_f32(blk->d) * scale;
        const uint8_t q = blk->qs[j];
        dst[(b << 5) + j]      = d * static_cast<float>(static_cast<int>(q & 0x0Fu) - 8);
        dst[(b << 5) + 16 + j] = d * static_cast<float>(static_cast<int>(q >> 4)    - 8);
    } else if (qtype == QuantType::Q3_K) {
        const int nb = K >> 8;
        const int b = tid >> 5;
        if (b >= nb) return;
        const int lane = tid & 31;
        const block_q3_K* blk = reinterpret_cast<const block_q3_K*>(rp + static_cast<size_t>(b) * sizeof(block_q3_K));
        const float d_all = d_fp16_to_f32(blk->d) * scale;
        uint32_t aux[4]; memcpy(aux, blk->scales, 12);
        const uint32_t tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & 0x0F0F0F0Fu) | (((tmp >> 4) & 0x03030303u) << 4);
        aux[3] = ((aux[1] >> 4) & 0x0F0F0F0Fu) | (((tmp >> 6) & 0x03030303u) << 4);
        aux[0] = (aux[0] & 0x0F0F0F0Fu) | (((tmp >> 0) & 0x03030303u) << 4);
        aux[1] = (aux[1] & 0x0F0F0F0Fu) | (((tmp >> 2) & 0x03030303u) << 4);
        const int8_t* sc = reinterpret_cast<const int8_t*>(aux);
        for (int n = 0; n < 2; ++n) {
            const uint8_t* q = blk->qs + (n << 5);
            const uint8_t m0 = static_cast<uint8_t>(1u << (n << 2));
            const uint8_t m1 = static_cast<uint8_t>(m0 << 1);
            const uint8_t m2 = static_cast<uint8_t>(m0 << 2);
            const uint8_t m3 = static_cast<uint8_t>(m0 << 3);
            const int base = (b << 8) + (n << 7) + lane;
            const int is0 = (n << 3) + (lane >> 4);
            const uint8_t qv = q[lane], hm = blk->hmask[lane];
            dst[base +  0] = d_all * static_cast<float>(sc[is0 + 0] - 32) * static_cast<float>(static_cast<int>((qv >> 0) & 3u) - ((hm & m0) ? 0 : 4));
            dst[base + 32] = d_all * static_cast<float>(sc[is0 + 2] - 32) * static_cast<float>(static_cast<int>((qv >> 2) & 3u) - ((hm & m1) ? 0 : 4));
            dst[base + 64] = d_all * static_cast<float>(sc[is0 + 4] - 32) * static_cast<float>(static_cast<int>((qv >> 4) & 3u) - ((hm & m2) ? 0 : 4));
            dst[base + 96] = d_all * static_cast<float>(sc[is0 + 6] - 32) * static_cast<float>(static_cast<int>((qv >> 6) & 3u) - ((hm & m3) ? 0 : 4));
        }
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
    if (i < n) x[i] = cap * tanhf(x[i] / cap);
}

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

// Per-Head Half-Split RoPE + FP16 KV Cache Write (supports dynamic GPU d_pos for CUDA Graphs!)
__global__ void k_rope_and_kv_write(float* __restrict__ Q, float* __restrict__ K, const float* __restrict__ V,
                                    __half* __restrict__ k_cache, __half* __restrict__ v_cache,
                                    int S, const int* __restrict__ d_pos, int start_pos_val, int max_seq_len,
                                    int H, int Hkv, int HD,
                                    float freq_base, float freq_scale,
                                    const float* __restrict__ freq_factors, int freq_factors_len,
                                    bool has_kv) {
    const int s = blockIdx.y;
    const int h = blockIdx.x;
    const int i = threadIdx.x;
    const int num_pairs = HD >> 1;
    if (s >= S || i >= num_pairs) return;

    const int start_pos = d_pos ? (*d_pos) : start_pos_val;
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

// Exact Online-Softmax GQA Attention Kernel (CUDA Graph compatible via d_pos!)
__global__ void k_gqa_attention(const float* __restrict__ Q,
                                const __half* __restrict__ k_cache,
                                const __half* __restrict__ v_cache,
                                float* __restrict__ attn_out,
                                int S, const int* __restrict__ d_pos, int start_pos_val, int max_seq_len,
                                int H, int Hkv, int HD,
                                int swa_win, bool is_swa, float attn_scale, float attn_softcap) {
    const int h = blockIdx.x;
    const int s = blockIdx.y;
    if (h >= H || s >= S || !k_cache || !v_cache) return;

    const int start_pos = d_pos ? (*d_pos) : start_pos_val;
    const int tid = threadIdx.x;
    const int wid = tid >> 5;
    const int lane = tid & 31;
    const int gqa_ratio = (Hkv > 0) ? (H / Hkv) : 1;
    const int kvh = min(h / max(1, gqa_ratio), max(1, Hkv) - 1);
    const size_t kv_dim = static_cast<size_t>(Hkv) * static_cast<size_t>(HD);
    const size_t kv_off = static_cast<size_t>(kvh) * static_cast<size_t>(HD);

    const float* qh = Q + (static_cast<size_t>(s) * static_cast<size_t>(H) + static_cast<size_t>(h)) * static_cast<size_t>(HD);
    float* oh = attn_out + (static_cast<size_t>(s) * static_cast<size_t>(H) + static_cast<size_t>(h)) * static_cast<size_t>(HD);

    __shared__ float sq[512];
    for (int d = tid; d < HD; d += blockDim.x) {
        sq[d] = qh[d] * attn_scale;
    }
    __syncthreads();

    const int raw_seq_len = start_pos + s + 1;
    const int seq_len = min(raw_seq_len, max_seq_len);
    if (seq_len <= 0) {
        for (int d = tid; d < HD; d += blockDim.x) oh[d] = 0.0f;
        return;
    }
    const int s_start = is_swa ? max(0, seq_len - swa_win) : 0;
    const bool use_sinks = (is_swa && s_start > 4);
    const int active_len = use_sinks ? (4 + (seq_len - s_start)) : (seq_len - s_start);

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
            if (d < HD) dot = fmaf(sq[d], __half2float(kt[d]), dot);
        }
        dot = warp_reduce_sum(dot);
        dot = __shfl_sync(0xFFFFFFFFu, dot, 0);
        if (attn_softcap > 0.0f) dot = attn_softcap * tanhf(dot / attn_softcap);

        const float m_new = fmaxf(m_w, dot);
        const float alpha = expf(m_w - m_new);
        const float beta  = expf(dot - m_new);
        l_w = l_w * alpha + beta;
        m_w = m_new;

        for (int e = 0; e < elems_per_lane; ++e) {
            const int d = (e << 5) + lane;
            if (d < HD) acc[e] = fmaf(beta, __half2float(vt[d]), acc[e] * alpha);
        }
    }

    __shared__ float sm_m[8];
    __shared__ float sm_l[8];
    __shared__ float sm_out[8 * 512];
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
        for (int w = 0; w < 8; ++w) val = fmaf(sm_out[w * 512 + d], w_scale[w], val);
        oh[d] = val * inv_l;
    }
}

// ============================================================================
// Gemma 4.5 26B-A4B Sparse MoE Kernels (Zero Host-Sync! B=1 + Grouped Prefill!)
// ============================================================================

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
    for (int e = tid; e < E; e += blockDim.x) local_max = fmaxf(local_max, row[e]);
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
    __syncthreads(); // Prevent WAR race before warp 0 overwrites sm_red[0] with local_sum!

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
    for (int e = tid; e < E; e += blockDim.x) probs[e] *= inv_sum;
    __syncthreads();

    if (tid == 0) {
        float w_sum = 0.0f;
        for (int k = 0; k < K_used; ++k) {
            int best_e = 0;
            float best_p = -1.0f;
            for (int e = 0; e < E; ++e) {
                if (probs[e] > best_p) { best_p = probs[e]; best_e = e; }
            }
            idx_o[k] = best_e;
            w_o[k]   = best_p;
            w_sum   += best_p;
            probs[best_e] = -2.0f;
        }
        if (w_sum < 6.103515625e-5f) w_sum = 6.103515625e-5f;
        for (int k = 0; k < K_used; ++k) {
            const int e = idx_o[k];
            w_o[k] = (w_o[k] / w_sum) * (down_s ? down_s[e] : 1.0f);
        }
    }
}

// Single/small-batch MoE Gate+Up+Act using Q8_1 __dp4a
__global__ void k_moe_gu_act_q8(const uint8_t* __restrict__ Wgu, QuantType qt_gu, size_t rb_gu,
                                const uint8_t* __restrict__ Wg,  QuantType qt_g,  size_t rb_g,
                                const uint8_t* __restrict__ Wu,  QuantType qt_u,  size_t rb_u,
                                const block_q8_1_gpu* __restrict__ x_q8, const float* __restrict__ x_f32,
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
    const block_q8_1_gpu* xq_s = x_q8 + static_cast<size_t>(s) * static_cast<size_t>(D >> 5);
    const float* xf_s = x_f32 + static_cast<size_t>(s) * static_cast<size_t>(D);

    float g = 0.0f, u = 0.0f;
    if (fused_gu) {
        const size_t base_row = static_cast<size_t>(e) * static_cast<size_t>(2 * F_exp);
        g = warp_dot_any_q8(Wgu + (base_row + static_cast<size_t>(r)) * rb_gu, qt_gu, xq_s, xf_s, D, lane);
        u = warp_dot_any_q8(Wgu + (base_row + static_cast<size_t>(F_exp + r)) * rb_gu, qt_gu, xq_s, xf_s, D, lane);
    } else {
        const size_t row = static_cast<size_t>(e) * static_cast<size_t>(F_exp) + static_cast<size_t>(r);
        g = warp_dot_any_q8(Wg + row * rb_g, qt_g, xq_s, xf_s, D, lane);
        u = warp_dot_any_q8(Wu + row * rb_u, qt_u, xq_s, xf_s, D, lane);
    }
    if (lane == 0) {
        const float act = use_gelu ? d_gelu_tanh(g) : d_silu(g);
        act_out[(static_cast<size_t>(s) * static_cast<size_t>(K_used) + static_cast<size_t>(k)) * static_cast<size_t>(F_exp) + static_cast<size_t>(r)] = act * u;
    }
}

// Single/small-batch MoE Weighted Down using Q8_1 __dp4a
__global__ void k_moe_down_accum_q8(const uint8_t* __restrict__ Wdn, QuantType qt_dn, size_t rb_dn,
                                    const block_q8_1_gpu* __restrict__ act_q8, const float* __restrict__ act_f32,
                                    const int* __restrict__ exp_idx, const float* __restrict__ exp_w,
                                    float* __restrict__ out, int K_used, int F_exp, int D) {
    const int s       = blockIdx.y;
    const int warp_id = threadIdx.x >> 5;
    const int lane    = threadIdx.x & 31;
    const int d       = (blockIdx.x << 2) + warp_id;
    if (d >= D) return;

    const int nb_f = F_exp >> 5;
    float total = 0.0f;
    for (int k = 0; k < K_used; ++k) {
        const int e = exp_idx[s * K_used + k];
        const float w = exp_w[s * K_used + k];
        const size_t slot_idx = static_cast<size_t>(s) * static_cast<size_t>(K_used) + static_cast<size_t>(k);
        const block_q8_1_gpu* aq = act_q8 + slot_idx * static_cast<size_t>(nb_f);
        const float* af = act_f32 + slot_idx * static_cast<size_t>(F_exp);
        const size_t row = static_cast<size_t>(e) * static_cast<size_t>(D) + static_cast<size_t>(d);
        const float dot = warp_dot_any_q8(Wdn + row * rb_dn, qt_dn, aq, af, F_exp, lane);
        if (lane == 0) total = fmaf(w, dot, total);
    }
    if (lane == 0) out[static_cast<size_t>(s) * static_cast<size_t>(D) + static_cast<size_t>(d)] = total;
}

// On-GPU CSR Expert Bucket Sort for Prefill (S >= 8, E <= 256, zero host sync, <4 us!)
__global__ void k_moe_bucket_sort(const int* __restrict__ exp_idx, const float* __restrict__ exp_w,
                                  int S, int K_used, int E,
                                  int* __restrict__ exp_offsets,
                                  int* __restrict__ sorted_tok,
                                  int* __restrict__ sorted_slot,
                                  float* __restrict__ sorted_w) {
    __shared__ int sm_counts[256];
    __shared__ int sm_cursor[256];
    const int tid = threadIdx.x;
    if (tid < E) sm_counts[tid] = 0;
    __syncthreads();

    const int total = S * K_used;
    for (int i = tid; i < total; i += blockDim.x) {
        const int e = exp_idx[i];
        if (e >= 0 && e < E) atomicAdd(&sm_counts[e], 1);
    }
    __syncthreads();

    if (tid == 0) {
        int acc = 0;
        for (int e = 0; e < E; ++e) {
            exp_offsets[e] = acc;
            sm_cursor[e] = acc;
            acc += sm_counts[e];
        }
        exp_offsets[E] = acc;
    }
    __syncthreads();

    for (int i = tid; i < total; i += blockDim.x) {
        const int e = exp_idx[i];
        if (e >= 0 && e < E) {
            const int dst = atomicAdd(&sm_cursor[e], 1);
            const int tok = i / K_used;
            const int slot = i - tok * K_used;
            sorted_tok[dst] = tok;
            sorted_slot[dst] = slot;
            sorted_w[dst] = exp_w[i];
        }
    }
}

// Grouped-Expert 8-Token Register-Tiled MoE Gate+Up+Act (reads expert weights ONCE per 8 tokens!)
// Grid: (ceil(F_exp / 4), E, 1), Block: 128 threads
__global__ void k_moe_gu_act_grouped(const uint8_t* __restrict__ Wgu, QuantType qt_gu, size_t rb_gu,
                                     const uint8_t* __restrict__ Wg,  QuantType qt_g,  size_t rb_g,
                                     const uint8_t* __restrict__ Wu,  QuantType qt_u,  size_t rb_u,
                                     const block_q8_1_gpu* __restrict__ x_q8, const float* __restrict__ x_f32,
                                     const int* __restrict__ exp_offsets,
                                     const int* __restrict__ sorted_tok,
                                     const int* __restrict__ sorted_slot,
                                     float* __restrict__ act_out,
                                     int K_used, int F_exp, int D, bool fused_gu, bool use_gelu) {
    const int e = blockIdx.y;
    const int start = exp_offsets[e];
    const int end   = exp_offsets[e + 1];
    if (start >= end) return;

    const int warp_id = threadIdx.x >> 5;
    const int lane    = threadIdx.x & 31;
    const int r       = (blockIdx.x << 2) + warp_id;
    if (r >= F_exp) return;

    const uint8_t* row_g = fused_gu
        ? (Wgu + (static_cast<size_t>(e) * static_cast<size_t>(2 * F_exp) + static_cast<size_t>(r)) * rb_gu)
        : (Wg  + (static_cast<size_t>(e) * static_cast<size_t>(F_exp)     + static_cast<size_t>(r)) * rb_g);
    const uint8_t* row_u = fused_gu
        ? (Wgu + (static_cast<size_t>(e) * static_cast<size_t>(2 * F_exp) + static_cast<size_t>(F_exp + r)) * rb_gu)
        : (Wu  + (static_cast<size_t>(e) * static_cast<size_t>(F_exp)     + static_cast<size_t>(r)) * rb_u);
    const QuantType qt_eff_g = fused_gu ? qt_gu : qt_g;
    const QuantType qt_eff_u = fused_gu ? qt_gu : qt_u;
    const int nb32_d = D >> 5;

    if (qt_eff_g == QuantType::IQ4_XS && qt_eff_u == QuantType::IQ4_XS) {
        const int lut_i = static_cast<int>(d_kvalues_iq4nl[lane & 15]);
        const int nb = D >> 8;
        const int ib = lane >> 2;
        const int t  = lane & 3;

        for (int m = start; m < end; m += 8) {
            const int T = min(8, end - m);
            int toks[8];
            float acc_g[8], acc_u[8];
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                toks[i] = (i < T) ? sorted_tok[m + i] : 0;
                acc_g[i] = 0.0f;
                acc_u[i] = 0.0f;
            }

            for (int b = 0; b < nb; ++b) {
                // Gate block: decode ONCE, reuse across up to 8 tokens!
                const block_iq4_xs* bg = reinterpret_cast<const block_iq4_xs*>(row_g + static_cast<size_t>(b) * sizeof(block_iq4_xs));
                const float dg = d_fp16_to_f32(bg->d) * static_cast<float>((((bg->scales_l[ib >> 1] >> (4 * (ib & 1))) & 0x0F) | (((bg->scales_h >> (2 * ib)) & 3) << 4)) - 32);
                const uint32_t qg = *reinterpret_cast<const uint32_t*>(bg->qs + (lane << 2));
                const int32_t wg_lo = pack4_i8(
                    __shfl_sync(0xFFFFFFFFu, lut_i, (qg >>  0) & 0x0Fu),
                    __shfl_sync(0xFFFFFFFFu, lut_i, (qg >>  8) & 0x0Fu),
                    __shfl_sync(0xFFFFFFFFu, lut_i, (qg >> 16) & 0x0Fu),
                    __shfl_sync(0xFFFFFFFFu, lut_i, (qg >> 24) & 0x0Fu));
                const int32_t wg_hi = pack4_i8(
                    __shfl_sync(0xFFFFFFFFu, lut_i, (qg >>  4) & 0x0Fu),
                    __shfl_sync(0xFFFFFFFFu, lut_i, (qg >> 12) & 0x0Fu),
                    __shfl_sync(0xFFFFFFFFu, lut_i, (qg >> 20) & 0x0Fu),
                    __shfl_sync(0xFFFFFFFFu, lut_i, (qg >> 28) & 0x0Fu));

                // Up block: decode ONCE, reuse across up to 8 tokens!
                const block_iq4_xs* bu = reinterpret_cast<const block_iq4_xs*>(row_u + static_cast<size_t>(b) * sizeof(block_iq4_xs));
                const float du = d_fp16_to_f32(bu->d) * static_cast<float>((((bu->scales_l[ib >> 1] >> (4 * (ib & 1))) & 0x0F) | (((bu->scales_h >> (2 * ib)) & 3) << 4)) - 32);
                const uint32_t qu = *reinterpret_cast<const uint32_t*>(bu->qs + (lane << 2));
                const int32_t wu_lo = pack4_i8(
                    __shfl_sync(0xFFFFFFFFu, lut_i, (qu >>  0) & 0x0Fu),
                    __shfl_sync(0xFFFFFFFFu, lut_i, (qu >>  8) & 0x0Fu),
                    __shfl_sync(0xFFFFFFFFu, lut_i, (qu >> 16) & 0x0Fu),
                    __shfl_sync(0xFFFFFFFFu, lut_i, (qu >> 24) & 0x0Fu));
                const int32_t wu_hi = pack4_i8(
                    __shfl_sync(0xFFFFFFFFu, lut_i, (qu >>  4) & 0x0Fu),
                    __shfl_sync(0xFFFFFFFFu, lut_i, (qu >> 12) & 0x0Fu),
                    __shfl_sync(0xFFFFFFFFu, lut_i, (qu >> 20) & 0x0Fu),
                    __shfl_sync(0xFFFFFFFFu, lut_i, (qu >> 28) & 0x0Fu));

                const int x_sub = (b << 3) + ib;
                for (int i = 0; i < T; ++i) {
                    const block_q8_1_gpu& xb = x_q8[static_cast<size_t>(toks[i]) * static_cast<size_t>(nb32_d) + static_cast<size_t>(x_sub)];
                    const int32_t* xq = reinterpret_cast<const int32_t*>(xb.qs);
                    const int32_t xlo = xq[t], xhi = xq[t + 4];
                    const int sg = __dp4a(wg_hi, xhi, __dp4a(wg_lo, xlo, 0));
                    const int su = __dp4a(wu_hi, xhi, __dp4a(wu_lo, xlo, 0));
                    acc_g[i] = fmaf(dg * xb.d, static_cast<float>(sg), acc_g[i]);
                    acc_u[i] = fmaf(du * xb.d, static_cast<float>(su), acc_u[i]);
                }
            }

            for (int i = 0; i < T; ++i) {
                const float g = warp_reduce_sum(acc_g[i]);
                const float u = warp_reduce_sum(acc_u[i]);
                if (lane == 0) {
                    const int tok = toks[i];
                    const int slot = sorted_slot[m + i];
                    const float act = use_gelu ? d_gelu_tanh(g) : d_silu(g);
                    act_out[(static_cast<size_t>(tok) * static_cast<size_t>(K_used) + static_cast<size_t>(slot)) * static_cast<size_t>(F_exp) + static_cast<size_t>(r)] = act * u;
                }
            }
        }
    } else {
        for (int m = start; m < end; ++m) {
            const int tok = sorted_tok[m];
            const int slot = sorted_slot[m];
            const block_q8_1_gpu* xq_s = x_q8 + static_cast<size_t>(tok) * static_cast<size_t>(nb32_d);
            const float* xf_s = x_f32 + static_cast<size_t>(tok) * static_cast<size_t>(D);
            const float g = warp_dot_any_q8(row_g, qt_eff_g, xq_s, xf_s, D, lane);
            const float u = warp_dot_any_q8(row_u, qt_eff_u, xq_s, xf_s, D, lane);
            if (lane == 0) {
                const float act = use_gelu ? d_gelu_tanh(g) : d_silu(g);
                act_out[(static_cast<size_t>(tok) * static_cast<size_t>(K_used) + static_cast<size_t>(slot)) * static_cast<size_t>(F_exp) + static_cast<size_t>(r)] = act * u;
            }
        }
    }
}

// Grouped-Expert 8-Token Register-Tiled MoE Weighted Down (reads expert weights ONCE per 8 tokens!)
// Grid: (ceil(D / 4), E, 1), Block: 128 threads
__global__ void k_moe_down_grouped(const uint8_t* __restrict__ Wdn, QuantType qt_dn, size_t rb_dn,
                                   const block_q8_1_gpu* __restrict__ act_q8, const float* __restrict__ act_f32,
                                   const int* __restrict__ exp_offsets,
                                   const int* __restrict__ sorted_tok,
                                   const int* __restrict__ sorted_slot,
                                   const float* __restrict__ sorted_w,
                                   float* __restrict__ out,
                                   int K_used, int F_exp, int D) {
    const int e = blockIdx.y;
    const int start = exp_offsets[e];
    const int end   = exp_offsets[e + 1];
    if (start >= end) return;

    const int warp_id = threadIdx.x >> 5;
    const int lane    = threadIdx.x & 31;
    const int d       = (blockIdx.x << 2) + warp_id;
    if (d >= D) return;

    const uint8_t* row_dn = Wdn + (static_cast<size_t>(e) * static_cast<size_t>(D) + static_cast<size_t>(d)) * rb_dn;
    const int nb32_f = F_exp >> 5;

    if (qt_dn == QuantType::IQ4_NL) {
        const int lut_i = static_cast<int>(d_kvalues_iq4nl[lane & 15]);
        const int nb = F_exp >> 5;
        const int b_off = lane >> 2;
        const int t = lane & 3;
        const int n_iters = (nb + 7) >> 3;

        for (int m = start; m < end; m += 8) {
            const int T = min(8, end - m);
            size_t slot_bases[8];
            float acc[8];
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                if (i < T) {
                    slot_bases[i] = (static_cast<size_t>(sorted_tok[m + i]) * static_cast<size_t>(K_used) + static_cast<size_t>(sorted_slot[m + i])) * static_cast<size_t>(nb32_f);
                } else {
                    slot_bases[i] = 0;
                }
                acc[i] = 0.0f;
            }

            for (int it = 0; it < n_iters; ++it) {
                const int b = (it << 3) + b_off;
                const bool active = (b < nb);
                uint32_t q4 = 0u;
                float dw = 0.0f;
                if (active) {
                    const block_iq4_nl* blk = reinterpret_cast<const block_iq4_nl*>(row_dn + static_cast<size_t>(b) * sizeof(block_iq4_nl));
                    dw = d_fp16_to_f32(blk->d);
                    memcpy(&q4, blk->qs + (t << 2), sizeof(uint32_t));
                }
                const int32_t w_lo = pack4_i8(
                    __shfl_sync(0xFFFFFFFFu, lut_i, (q4 >>  0) & 0x0Fu),
                    __shfl_sync(0xFFFFFFFFu, lut_i, (q4 >>  8) & 0x0Fu),
                    __shfl_sync(0xFFFFFFFFu, lut_i, (q4 >> 16) & 0x0Fu),
                    __shfl_sync(0xFFFFFFFFu, lut_i, (q4 >> 24) & 0x0Fu));
                const int32_t w_hi = pack4_i8(
                    __shfl_sync(0xFFFFFFFFu, lut_i, (q4 >>  4) & 0x0Fu),
                    __shfl_sync(0xFFFFFFFFu, lut_i, (q4 >> 12) & 0x0Fu),
                    __shfl_sync(0xFFFFFFFFu, lut_i, (q4 >> 20) & 0x0Fu),
                    __shfl_sync(0xFFFFFFFFu, lut_i, (q4 >> 28) & 0x0Fu));
                if (active) {
                    for (int i = 0; i < T; ++i) {
                        const block_q8_1_gpu& xb = act_q8[slot_bases[i] + static_cast<size_t>(b)];
                        const int32_t* xq = reinterpret_cast<const int32_t*>(xb.qs);
                        const int sumi = __dp4a(w_hi, xq[t + 4], __dp4a(w_lo, xq[t], 0));
                        acc[i] = fmaf(dw * xb.d, static_cast<float>(sumi), acc[i]);
                    }
                }
            }

            for (int i = 0; i < T; ++i) {
                const float dot = warp_reduce_sum(acc[i]);
                if (lane == 0) {
                    const int tok = sorted_tok[m + i];
                    atomicAdd(&out[static_cast<size_t>(tok) * static_cast<size_t>(D) + static_cast<size_t>(d)], sorted_w[m + i] * dot);
                }
            }
        }
    } else if (qt_dn == QuantType::IQ4_XS) {
        const int lut_i = static_cast<int>(d_kvalues_iq4nl[lane & 15]);
        const int nb = F_exp >> 8;
        const int ib = lane >> 2;
        const int t  = lane & 3;

        for (int m = start; m < end; m += 8) {
            const int T = min(8, end - m);
            size_t slot_bases[8];
            float acc[8];
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                slot_bases[i] = (i < T)
                    ? (static_cast<size_t>(sorted_tok[m + i]) * static_cast<size_t>(K_used) + static_cast<size_t>(sorted_slot[m + i])) * static_cast<size_t>(nb32_f)
                    : 0;
                acc[i] = 0.0f;
            }
            for (int b = 0; b < nb; ++b) {
                const block_iq4_xs* blk = reinterpret_cast<const block_iq4_xs*>(row_dn + static_cast<size_t>(b) * sizeof(block_iq4_xs));
                const float dl = d_fp16_to_f32(blk->d) * static_cast<float>((((blk->scales_l[ib >> 1] >> (4 * (ib & 1))) & 0x0F) | (((blk->scales_h >> (2 * ib)) & 3) << 4)) - 32);
                const uint32_t q4 = *reinterpret_cast<const uint32_t*>(blk->qs + (lane << 2));
                const int32_t w_lo = pack4_i8(
                    __shfl_sync(0xFFFFFFFFu, lut_i, (q4 >>  0) & 0x0Fu),
                    __shfl_sync(0xFFFFFFFFu, lut_i, (q4 >>  8) & 0x0Fu),
                    __shfl_sync(0xFFFFFFFFu, lut_i, (q4 >> 16) & 0x0Fu),
                    __shfl_sync(0xFFFFFFFFu, lut_i, (q4 >> 24) & 0x0Fu));
                const int32_t w_hi = pack4_i8(
                    __shfl_sync(0xFFFFFFFFu, lut_i, (q4 >>  4) & 0x0Fu),
                    __shfl_sync(0xFFFFFFFFu, lut_i, (q4 >> 12) & 0x0Fu),
                    __shfl_sync(0xFFFFFFFFu, lut_i, (q4 >> 20) & 0x0Fu),
                    __shfl_sync(0xFFFFFFFFu, lut_i, (q4 >> 28) & 0x0Fu));
                const int x_sub = (b << 3) + ib;
                for (int i = 0; i < T; ++i) {
                    const block_q8_1_gpu& xb = act_q8[slot_bases[i] + static_cast<size_t>(x_sub)];
                    const int32_t* xq = reinterpret_cast<const int32_t*>(xb.qs);
                    const int sumi = __dp4a(w_hi, xq[t + 4], __dp4a(w_lo, xq[t], 0));
                    acc[i] = fmaf(dl * xb.d, static_cast<float>(sumi), acc[i]);
                }
            }
            for (int i = 0; i < T; ++i) {
                const float dot = warp_reduce_sum(acc[i]);
                if (lane == 0) {
                    const int tok = sorted_tok[m + i];
                    atomicAdd(&out[static_cast<size_t>(tok) * static_cast<size_t>(D) + static_cast<size_t>(d)], sorted_w[m + i] * dot);
                }
            }
        }
    } else {
        for (int m = start; m < end; ++m) {
            const int tok = sorted_tok[m];
            const int slot = sorted_slot[m];
            const size_t slot_idx = static_cast<size_t>(tok) * static_cast<size_t>(K_used) + static_cast<size_t>(slot);
            const float dot = warp_dot_any_q8(row_dn, qt_dn, act_q8 + slot_idx * static_cast<size_t>(nb32_f), act_f32 + slot_idx * static_cast<size_t>(F_exp), F_exp, lane);
            if (lane == 0) {
                atomicAdd(&out[static_cast<size_t>(tok) * static_cast<size_t>(D) + static_cast<size_t>(d)], sorted_w[m] * dot);
            }
        }
    }
}

__global__ void k_pleias_combine(float* __restrict__ pl_emb, const float* __restrict__ pl_proj, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) pl_emb[i] = (pl_proj[i] + pl_emb[i]) * 0.7071067811865475f;
}

__global__ void k_pleias_gate_mul(float* __restrict__ gate, const float* __restrict__ pl_emb,
                                  int S, int L, int l, int P_dim) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= S * P_dim) return;
    const int s = i / P_dim;
    const int p = i - s * P_dim;
    const float g = gate[i];
    gate[i] = d_gelu_tanh(g) * pl_emb[(static_cast<size_t>(s) * static_cast<size_t>(L) + static_cast<size_t>(l)) * static_cast<size_t>(P_dim) + static_cast<size_t>(p)];
}

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

    bool has_moe = false;
    bool fused_moe_gu = false;
    GpuTensor moe_router;
    GpuTensor moe_gu, moe_g, moe_u, moe_dn;
    float* d_moe_rscale = nullptr;
    float* d_moe_down_scale = nullptr;
    float* d_post_ffn_norm_1 = nullptr;
    float* d_pre_ffn_norm_2 = nullptr;
    float* d_post_ffn_norm_2 = nullptr;

    GpuTensor inp_gate, proj;
    float* d_post_norm = nullptr;

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
    bool use_cuda_graph = true;
    int max_F = 0;
    int max_q_dim = 0;
    int max_kv_dim = 0;
    int max_S = 512;
    int max_seq_len_runtime = 2048;

    // Pinned host + device args for single-shot B=1 CUDA Graph decode
    uint32_t* h_step_args = nullptr;   // [2]: {token, pos}
    uint32_t* h_am_out = nullptr;      // [1]
    uint32_t* d_step_args = nullptr;   // [2]: {token, pos} (d_step_args[0]=tok, d_step_args[1]=pos)
    cudaGraph_t decode_graph = nullptr;
    cudaGraphExec_t decode_graph_exec = nullptr;
    bool decode_graph_captured = false;

    // Activation & Q8_1 scratch buffers
    uint32_t* d_tok_ids = nullptr;
    float* d_x = nullptr;
    float* d_norm = nullptr;
    float* d_q = nullptr;
    float* d_k = nullptr;
    float* d_v = nullptr;
    float* d_attn_h = nullptr;
    float* d_attn_o = nullptr;
    float* d_ffn_g = nullptr;
    float* d_ffn_u = nullptr;
    float* d_ffn_d = nullptr;
    block_q8_1_gpu* d_x_q8 = nullptr;       // [max_S * (max_in_dim / 32)]

    // MoE scratch (including GPU CSR bucket sort buffers for grouped prefill!)
    float* d_moe_norm = nullptr;
    float* d_moe_router = nullptr;
    int*   d_moe_idx = nullptr;
    float* d_moe_w = nullptr;
    float* d_moe_act = nullptr;
    block_q8_1_gpu* d_moe_act_q8 = nullptr; // [max_S * K_used * (F_exp / 32)]
    float* d_moe_out = nullptr;
    int*   d_moe_exp_offsets = nullptr;     // [257]
    int*   d_moe_sorted_tok = nullptr;      // [max_S * K_used]
    int*   d_moe_sorted_slot = nullptr;     // [max_S * K_used]
    float* d_moe_sorted_w = nullptr;        // [max_S * K_used]

    // Pleias scratch
    float* d_pl_emb = nullptr;
    float* d_pl_proj = nullptr;
    float* d_pl_gate = nullptr;
    float* d_pl_out = nullptr;

    // Logits & Argmax scratch
    float* d_logits = nullptr;
    float* d_am_vals = nullptr;
    uint32_t* d_am_idx = nullptr;
    uint32_t* d_am_out = nullptr;

    // FP16 cuBLAS Tensor Core prefill scratch
    __half* d_x_f16 = nullptr;
    __half* d_w_f16 = nullptr;
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
            err = "Tensor " + td->name + " byte size mismatch";
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

    void quantize_q8(const float* d_in, block_q8_1_gpu* d_out, int n_floats) {
        const int nb = n_floats >> 5;
        if (nb <= 0) return;
        k_quantize_q8_1<<<(nb + 7) >> 3, 256, 0, stream>>>(d_in, d_out, nb);
    }

    void linear_forward(const GpuTensor& W, const float* d_bias, const float* d_in, float* d_out, int S,
                        float scale_out = 1.0f, bool reuse_q8 = false) {
        const int N = W.rows;
        const int K = W.cols;
        if (S <= 1) {
            if (!reuse_q8 && (K & 31) == 0 && W.type != QuantType::F32 && W.type != QuantType::BF16 && W.type != QuantType::F16) {
                quantize_q8(d_in, d_x_q8, K);
            }
            k_gemv_quant_q8<<<(N + 3) >> 2, 128, 0, stream>>>(
                W.ptr, W.type, W.row_bytes, d_x_q8, d_in, d_bias, d_out, N, K, scale_out);
            return;
        }
        if (S < 8 || static_cast<size_t>(N) * static_cast<size_t>(K) > w_f16_cap_elems) {
            if (!reuse_q8 && (K & 31) == 0) quantize_q8(d_in, d_x_q8, S * K);
            const int nb32 = K >> 5;
            const int blocks = (N + 3) >> 2;
            for (int s = 0; s < S; ++s) {
                k_gemv_quant_q8<<<blocks, 128, 0, stream>>>(
                    W.ptr, W.type, W.row_bytes,
                    d_x_q8 + static_cast<size_t>(s) * static_cast<size_t>(nb32),
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

        // Zero-stall __shfl_sync dequantize W [N, K] raw blocks -> FP16 scratch
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
        else if (tbl.type == QuantType::Q3_K) threads_per_row = K >> 3;
        else if (tbl.type == QuantType::IQ4_XS || tbl.type == QuantType::IQ4_NL ||
                 tbl.type == QuantType::Q4_K || tbl.type == QuantType::Q5_K ||
                 tbl.type == QuantType::Q5_1 || tbl.type == QuantType::Q5_0 || tbl.type == QuantType::Q4_0) {
            threads_per_row = K >> 1;
        }
        dim3 grid((threads_per_row + 255) / 256, static_cast<unsigned int>(S), 1);
        k_dequant_rows_f32<<<grid, 256, 0, stream>>>(tbl.ptr, tbl.type, tbl.row_bytes, d_ids, S, vocab_size, d_out, K, scale);
    }

    // Core GPU Transformer Graph Body (100% stream-enqueueable & CUDA-Graph capturable for S==1!)
    void enqueue_transformer_body(const uint32_t* d_ids_ptr, int S,
                                  const int* d_pos_ptr, int start_pos_val,
                                  bool compute_lm_head, bool compute_argmax) {
        const int D = static_cast<int>(mc.embedding_dim);
        const int L = static_cast<int>(mc.num_layers);
        const int H = static_cast<int>(mc.num_heads);
        const int V = static_cast<int>(mc.vocab_size);
        const bool use_gelu = (mc.ffn_activation == FfnActivation::GELU);
        const int swa_win = (mc.sliding_window > 0) ? static_cast<int>(mc.sliding_window) : 512;

        // 1. Gather Token Embeddings on GPU
        const float emb_scale = mc.is_gemma ? std::sqrt(static_cast<float>(D)) : 1.0f;
        dequant_embed_rows(tok_emb, d_ids_ptr, S, V, d_x, emb_scale);

        // 1b. Optional Pleias Per-Layer Embeddings (Gemma 4.5 E2B / E4B)
        if (has_pleias) {
            dequant_embed_rows(pl_tok_emb, d_ids_ptr, S, V, d_pl_emb, 16.0f);
            if (pl_model_proj.valid()) {
                linear_forward(pl_model_proj, nullptr, d_x, d_pl_proj, S, 1.0f / std::sqrt(static_cast<float>(D)));
                k_rms_norm_rows<<<S * L, 128, 0, stream>>>(d_pl_proj, d_pl_proj, d_pl_proj_norm, 256, mc.rms_norm_eps);
                const int tot_pl = S * L * 256;
                k_pleias_combine<<<(tot_pl + 255) / 256, 256, 0, stream>>>(d_pl_emb, d_pl_proj, tot_pl);
            }
        }

        // 2. Transformer Layer Loop
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
                quantize_q8(d_norm, d_x_q8, D);
                const int k_n = gl.has_kv ? kv_dim : 0;
                const int v_n = (gl.has_kv && !gl.k_eq_v && gl.attn_v.valid()) ? kv_dim : 0;
                const int total_qkv = q_dim + k_n + v_n;
                k_fused_qkv_gemv_q8<<<(total_qkv + 3) >> 2, 128, 0, stream>>>(
                    gl.attn_q.ptr, gl.attn_q.type, gl.attn_q.row_bytes, gl.d_attn_q_bias,
                    gl.has_kv ? gl.attn_k.ptr : nullptr, gl.attn_k.type, gl.attn_k.row_bytes, gl.d_attn_k_bias,
                    (v_n > 0) ? gl.attn_v.ptr : nullptr, gl.attn_v.type, gl.attn_v.row_bytes, gl.d_attn_v_bias,
                    d_x_q8, d_norm, d_q, d_k, d_v, q_dim, k_n, v_n, D, gl.k_eq_v);
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
            if (gl.d_attn_q_norm) k_rms_norm_rows<<<S * H, 128, 0, stream>>>(d_q, d_q, gl.d_attn_q_norm, HD, mc.rms_norm_eps);
            if (gl.has_kv && gl.d_attn_k_norm) k_rms_norm_rows<<<S * Hkv, 128, 0, stream>>>(d_k, d_k, gl.d_attn_k_norm, HD, mc.rms_norm_eps);
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
                S, d_pos_ptr, start_pos_val, static_cast<int>(max_seq_len_runtime),
                H, Hkv, HD, freq_base, mc.rope_freq_scale, rf, rf_len, gl.has_kv);

            // 2e. Online-Softmax GQA Attention
            dim3 attn_grid(static_cast<unsigned int>(H), static_cast<unsigned int>(S), 1);
            k_gqa_attention<<<attn_grid, 256, 0, stream>>>(
                d_q, kv_src.d_k_cache, kv_src.d_v_cache, d_attn_h,
                S, d_pos_ptr, start_pos_val, static_cast<int>(max_seq_len_runtime),
                H, kv_src.n_kv_heads, HD, swa_win, gl.is_swa, attn_scale, mc.attn_logit_softcapping);

            // 2f. Attention Output Projection + Optional Post-Attn Norm + Residual 1
            linear_forward(gl.attn_output, nullptr, d_attn_h, d_attn_o, S);
            if (gl.d_post_attn_norm) k_rms_norm_rows<<<S, 256, 0, stream>>>(d_attn_o, d_attn_o, gl.d_post_attn_norm, D, mc.rms_norm_eps);
            k_add_residual<<<(S * D + 255) / 256, 256, 0, stream>>>(d_x, d_attn_o, S * D);

            // 2g. Pre-FFN RMSNorm + Optional Gemma 4.5 Turbo Block-FWHT
            k_rms_norm_rows<<<S, 256, 0, stream>>>(d_norm, d_x, mc.turbo_folded_norms ? nullptr : gl.d_ffn_norm, D, mc.rms_norm_eps);
            if (mc.turbo_hadamard_ffn > 0) {
                launch_fwht(d_norm, S * D, static_cast<int>(mc.turbo_hadamard_ffn));
            }

            // 2h. Dense Shared FFN
            if (S == 1) {
                quantize_q8(d_norm, d_x_q8, D);
                k_fused_ffn_gu_act_q8<<<(F_hid + 3) >> 2, 128, 0, stream>>>(
                    gl.ffn_gate.ptr, gl.ffn_gate.type, gl.ffn_gate.row_bytes,
                    gl.ffn_up.ptr,   gl.ffn_up.type,   gl.ffn_up.row_bytes,
                    d_x_q8, d_norm, d_ffn_g, F_hid, D, use_gelu);
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

            // 2i. Optional Parallel Sparse MoE Branch (100% GPU-resident, zero host sync!)
            if (gl.has_moe) {
                const int E = static_cast<int>(mc.expert_count);
                const int K_used = std::clamp(static_cast<int>(mc.expert_used_count > 0 ? mc.expert_used_count : 8u), 1, E);
                const int F_exp = static_cast<int>(mc.expert_ffn_dim > 0 ? mc.expert_ffn_dim : mc.hidden_dim);

                if (gl.d_post_ffn_norm_1) {
                    k_rms_norm_rows<<<S, 256, 0, stream>>>(d_ffn_d, d_ffn_d, gl.d_post_ffn_norm_1, D, mc.rms_norm_eps);
                }
                k_rms_norm_rows<<<S, 256, 0, stream>>>(
                    d_moe_norm, d_x, mc.turbo_folded_scales ? nullptr : gl.d_moe_rscale,
                    D, mc.rms_norm_eps, 1.0f / std::sqrt(static_cast<float>(D)));
                linear_forward(gl.moe_router, nullptr, d_moe_norm, d_moe_router, S);
                k_moe_router_topk<<<S, 128, 0, stream>>>(
                    d_moe_router, mc.turbo_folded_scales ? nullptr : gl.d_moe_down_scale,
                    d_moe_idx, d_moe_w, E, K_used);

                k_rms_norm_rows<<<S, 256, 0, stream>>>(
                    d_moe_norm, d_x, mc.turbo_folded_norms ? nullptr : gl.d_pre_ffn_norm_2, D, mc.rms_norm_eps);
                if (mc.turbo_hadamard_moe_in > 0) {
                    launch_fwht(d_moe_norm, S * D, static_cast<int>(mc.turbo_hadamard_moe_in));
                }
                quantize_q8(d_moe_norm, d_x_q8, S * D);

                if (S >= 8 && E <= 256) {
                    // Grouped 8-token register-tiled MoE prefill (zero host sync, 8x weight & LUT reuse!)
                    k_moe_bucket_sort<<<1, 256, 0, stream>>>(
                        d_moe_idx, d_moe_w, S, K_used, E,
                        d_moe_exp_offsets, d_moe_sorted_tok, d_moe_sorted_slot, d_moe_sorted_w);
                    dim3 gu_grid((F_exp + 3) >> 2, static_cast<unsigned int>(E), 1);
                    k_moe_gu_act_grouped<<<gu_grid, 128, 0, stream>>>(
                        gl.moe_gu.ptr, gl.moe_gu.type, gl.moe_gu.row_bytes,
                        gl.moe_g.ptr,  gl.moe_g.type,  gl.moe_g.row_bytes,
                        gl.moe_u.ptr,  gl.moe_u.type,  gl.moe_u.row_bytes,
                        d_x_q8, d_moe_norm,
                        d_moe_exp_offsets, d_moe_sorted_tok, d_moe_sorted_slot,
                        d_moe_act, K_used, F_exp, D, gl.fused_moe_gu, use_gelu);
                    if (mc.turbo_hadamard_moe_down > 0 && (F_exp % static_cast<int>(mc.turbo_hadamard_moe_down)) == 0) {
                        launch_fwht(d_moe_act, S * K_used * F_exp, static_cast<int>(mc.turbo_hadamard_moe_down));
                    }
                    quantize_q8(d_moe_act, d_moe_act_q8, S * K_used * F_exp);
                    cudaMemsetAsync(d_moe_out, 0, static_cast<size_t>(S) * static_cast<size_t>(D) * sizeof(float), stream);
                    dim3 dn_grid((D + 3) >> 2, static_cast<unsigned int>(E), 1);
                    k_moe_down_grouped<<<dn_grid, 128, 0, stream>>>(
                        gl.moe_dn.ptr, gl.moe_dn.type, gl.moe_dn.row_bytes,
                        d_moe_act_q8, d_moe_act,
                        d_moe_exp_offsets, d_moe_sorted_tok, d_moe_sorted_slot, d_moe_sorted_w,
                        d_moe_out, K_used, F_exp, D);
                } else {
                    dim3 gu_grid((K_used * F_exp + 3) >> 2, static_cast<unsigned int>(S), 1);
                    k_moe_gu_act_q8<<<gu_grid, 128, 0, stream>>>(
                        gl.moe_gu.ptr, gl.moe_gu.type, gl.moe_gu.row_bytes,
                        gl.moe_g.ptr,  gl.moe_g.type,  gl.moe_g.row_bytes,
                        gl.moe_u.ptr,  gl.moe_u.type,  gl.moe_u.row_bytes,
                        d_x_q8, d_moe_norm, d_moe_idx, d_moe_act,
                        K_used, F_exp, D, gl.fused_moe_gu, use_gelu);
                    if (mc.turbo_hadamard_moe_down > 0 && (F_exp % static_cast<int>(mc.turbo_hadamard_moe_down)) == 0) {
                        launch_fwht(d_moe_act, S * K_used * F_exp, static_cast<int>(mc.turbo_hadamard_moe_down));
                    }
                    quantize_q8(d_moe_act, d_moe_act_q8, S * K_used * F_exp);
                    dim3 dn_grid((D + 3) >> 2, static_cast<unsigned int>(S), 1);
                    k_moe_down_accum_q8<<<dn_grid, 128, 0, stream>>>(
                        gl.moe_dn.ptr, gl.moe_dn.type, gl.moe_dn.row_bytes,
                        d_moe_act_q8, d_moe_act, d_moe_idx, d_moe_w, d_moe_out,
                        K_used, F_exp, D);
                }

                if (gl.d_post_ffn_norm_2) {
                    k_rms_norm_rows<<<S, 256, 0, stream>>>(d_moe_out, d_moe_out, gl.d_post_ffn_norm_2, D, mc.rms_norm_eps);
                }
                k_add_residual<<<(S * D + 255) / 256, 256, 0, stream>>>(d_ffn_d, d_moe_out, S * D);
            }

            if (gl.d_post_ffn_norm) k_rms_norm_rows<<<S, 256, 0, stream>>>(d_ffn_d, d_ffn_d, gl.d_post_ffn_norm, D, mc.rms_norm_eps);
            k_add_residual<<<(S * D + 255) / 256, 256, 0, stream>>>(d_x, d_ffn_d, S * D);

            // 2j. Optional Pleias Per-Layer Token Injection
            if (has_pleias && gl.inp_gate.valid() && gl.proj.valid()) {
                linear_forward(gl.inp_gate, nullptr, d_x, d_pl_gate, S);
                const int tot_p = S * 256;
                k_pleias_gate_mul<<<(tot_p + 255) / 256, 256, 0, stream>>>(d_pl_gate, d_pl_emb, S, L, l, 256);
                linear_forward(gl.proj, nullptr, d_pl_gate, d_pl_out, S);
                if (gl.d_post_norm) k_rms_norm_rows<<<S, 256, 0, stream>>>(d_pl_out, d_pl_out, gl.d_post_norm, D, mc.rms_norm_eps);
                k_add_residual<<<(S * D + 255) / 256, 256, 0, stream>>>(d_x, d_pl_out, S * D);
            } else if (gl.d_post_norm) {
                k_rms_norm_rows<<<S, 256, 0, stream>>>(d_x, d_x, gl.d_post_norm, D, mc.rms_norm_eps);
            }

            if (std::fabs(gl.layer_output_scale - 1.0f) > 1e-6f) {
                k_scale_inplace<<<(S * D + 255) / 256, 256, 0, stream>>>(d_x, gl.layer_output_scale, S * D);
            }
        }

        // 3. Final Output RMSNorm + 262k LM Head + Softcap + Optional On-GPU Argmax
        if (compute_lm_head) {
            const float* d_x_last = d_x + static_cast<size_t>(S - 1) * static_cast<size_t>(D);
            k_rms_norm_rows<<<1, 256, 0, stream>>>(d_norm, d_x_last, d_out_norm, D, mc.rms_norm_eps);
            linear_forward(lm_head, nullptr, d_norm, d_logits, 1);
            if (mc.final_logit_softcapping > 0.0f) {
                k_softcap_inplace<<<(V + 255) / 256, 256, 0, stream>>>(d_logits, V, mc.final_logit_softcapping);
            }
            if (compute_argmax) {
                k_argmax_stage1<<<256, 256, 0, stream>>>(d_logits, V, d_am_vals, d_am_idx);
                k_argmax_stage2<<<1, 256, 0, stream>>>(d_am_vals, d_am_idx, 256, d_am_out);
            }
        }
    }

    bool run_decode_step(uint32_t token, int pos, float* out_logits, uint32_t* out_next_token, std::string& err) {
        const int V = static_cast<int>(mc.vocab_size);
        h_step_args[0] = token;
        h_step_args[1] = static_cast<uint32_t>(pos);
        cudaMemcpyAsync(d_step_args, h_step_args, 2 * sizeof(uint32_t), cudaMemcpyHostToDevice, stream);

        if (use_cuda_graph) {
            if (!decode_graph_captured) {
                cudaError_t ce = cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
                if (ce == cudaSuccess) {
                    enqueue_transformer_body(d_step_args, 1, reinterpret_cast<const int*>(d_step_args + 1), 0, true, true);
                    ce = cudaStreamEndCapture(stream, &decode_graph);
                    if (ce == cudaSuccess && decode_graph != nullptr) {
                        ce = cudaGraphInstantiate(&decode_graph_exec, decode_graph, nullptr, nullptr, 0);
                        if (ce == cudaSuccess && decode_graph_exec != nullptr) {
                            decode_graph_captured = true;
                        } else {
                            if (decode_graph) { cudaGraphDestroy(decode_graph); decode_graph = nullptr; }
                            use_cuda_graph = false;
                        }
                    } else {
                        use_cuda_graph = false;
                    }
                } else {
                    use_cuda_graph = false;
                }
            }
            if (decode_graph_captured) {
                cudaGraphLaunch(decode_graph_exec, stream);
            } else {
                enqueue_transformer_body(d_step_args, 1, reinterpret_cast<const int*>(d_step_args + 1), 0,
                                         (out_logits || out_next_token), (out_next_token != nullptr));
            }
        } else {
            enqueue_transformer_body(d_step_args, 1, reinterpret_cast<const int*>(d_step_args + 1), 0,
                                     (out_logits || out_next_token), (out_next_token != nullptr));
        }

        if (out_next_token) {
            cudaMemcpyAsync(h_am_out, d_am_out, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream);
        }
        if (out_logits) {
            cudaMemcpyAsync(out_logits, d_logits, static_cast<size_t>(V) * sizeof(float), cudaMemcpyDeviceToHost, stream);
        }
        cudaError_t sync_err = cudaStreamSynchronize(stream);
        if (sync_err != cudaSuccess) {
            err = std::string("CUDA decode error: ") + cudaGetErrorString(sync_err);
            return false;
        }
        if (out_next_token) *out_next_token = *h_am_out;
        return true;
    }

    bool run_prefill_chunk(const uint32_t* h_tokens, int S, int start_pos,
                           float* out_last_logits, uint32_t* out_next_token, std::string& err) {
        const int V = static_cast<int>(mc.vocab_size);
        cudaMemcpyAsync(d_tok_ids, h_tokens, static_cast<size_t>(S) * sizeof(uint32_t), cudaMemcpyHostToDevice, stream);
        enqueue_transformer_body(d_tok_ids, S, nullptr, start_pos, (out_last_logits || out_next_token), (out_next_token != nullptr));

        if (out_next_token) {
            cudaMemcpyAsync(h_am_out, d_am_out, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream);
        }
        if (out_last_logits) {
            cudaMemcpyAsync(out_last_logits, d_logits, static_cast<size_t>(V) * sizeof(float), cudaMemcpyDeviceToHost, stream);
        }
        cudaError_t sync_err = cudaStreamSynchronize(stream);
        if (sync_err != cudaSuccess) {
            err = std::string("CUDA prefill error: ") + cudaGetErrorString(sync_err);
            return false;
        }
        if (out_next_token) *out_next_token = *h_am_out;
        return true;
    }

    void free_all() {
        if (stream) { cudaStreamSynchronize(stream); }
        if (decode_graph_exec) { cudaGraphExecDestroy(decode_graph_exec); decode_graph_exec = nullptr; }
        if (decode_graph) { cudaGraphDestroy(decode_graph); decode_graph = nullptr; }
        decode_graph_captured = false;
        if (h_step_args) { cudaFreeHost(h_step_args); h_step_args = nullptr; }
        if (h_am_out) { cudaFreeHost(h_am_out); h_am_out = nullptr; }
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
    impl_->use_cuda_graph = cfg.use_cuda_graph;

    if (cudaStreamCreate(&impl_->stream) != cudaSuccess ||
        cublasCreate(&impl_->cublas) != CUBLAS_STATUS_SUCCESS ||
        cublasSetStream(impl_->cublas, impl_->stream) != CUBLAS_STATUS_SUCCESS ||
        cudaHostAlloc(&impl_->h_step_args, 2 * sizeof(uint32_t), cudaHostAllocDefault) != cudaSuccess ||
        cudaHostAlloc(&impl_->h_am_out, sizeof(uint32_t), cudaHostAllocDefault) != cudaSuccess) {
        last_error_ = "Failed to create CUDA stream, cuBLAS handle, or pinned host staging.";
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

    // 2. Upload All L Transformer Layers (Raw GGUF Blocks -> VRAM)
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

        if (gl.head_dim > 512) {
            last_error_ = "CUDA kernel limit exceeded: head_dim > 512 at layer " + std::to_string(l);
            shutdown();
            return false;
        }
        if (mc.expert_count > 256) {
            last_error_ = "CUDA kernel limit exceeded: expert_count > 256 (got " + std::to_string(mc.expert_count) + ")";
            shutdown();
            return false;
        }
        if (gl.has_kv) {
            gl.kv_source_layer = l;
        } else {
            // Find the latest preceding KV-owning layer with matching SWA/Global type (safe for any num_layers / pattern!)
            int found = -1;
            for (int p = l - 1; p >= 0; --p) {
                if (impl_->layers[static_cast<size_t>(p)].has_kv &&
                    impl_->layers[static_cast<size_t>(p)].is_swa == gl.is_swa) {
                    found = p;
                    break;
                }
            }
            if (found < 0) {
                for (int p = l - 1; p >= 0; --p) {
                    if (impl_->layers[static_cast<size_t>(p)].has_kv) { found = p; break; }
                }
            }
            if (found < 0) {
                last_error_ = "No valid KV source layer found for shared-KV layer " + std::to_string(l);
                shutdown();
                return false;
            }
            gl.kv_source_layer = found;
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

        if (impl_->has_pleias && lt.inp_gate && lt.proj) {
            if (!impl_->upload_raw_tensor(lt.inp_gate, 256, D, gl.inp_gate, weight_bytes, last_error_)) { shutdown(); return false; }
            if (!impl_->upload_raw_tensor(lt.proj,     D, 256, gl.proj,     weight_bytes, last_error_)) { shutdown(); return false; }
        }
        if (lt.post_norm) gl.d_post_norm = impl_->upload_f32_1d(lt.post_norm, D, 1.0f, weight_bytes, last_error_);
        if (lt.layer_output_scale && lt.layer_output_scale->data && lt.layer_output_scale->type == QuantType::F32) {
            gl.layer_output_scale = *reinterpret_cast<const float*>(lt.layer_output_scale->data);
        }

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

    // 3. Allocate Activation, Q8_1, Grouped MoE & FP16 Prefill Scratch Buffers
    const size_t Smax = static_cast<size_t>(impl_->max_S);
    auto alloc_f32 = [&](size_t elems) -> float* {
        return static_cast<float*>(impl_->alloc_gpu(elems * sizeof(float), scratch_bytes));
    };
    impl_->d_step_args = static_cast<uint32_t*>(impl_->alloc_gpu(2 * sizeof(uint32_t), scratch_bytes));
    impl_->d_tok_ids   = static_cast<uint32_t*>(impl_->alloc_gpu(Smax * sizeof(uint32_t), scratch_bytes));
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

    const size_t max_in_dim = static_cast<size_t>(std::max({D, impl_->max_F, impl_->max_q_dim}));
    const size_t q8_blocks = Smax * ((max_in_dim + 31ULL) >> 5);
    impl_->d_x_q8 = static_cast<block_q8_1_gpu*>(impl_->alloc_gpu(q8_blocks * sizeof(block_q8_1_gpu), scratch_bytes));

    if (mc.expert_count > 0) {
        const int E = static_cast<int>(mc.expert_count);
        const int K_used = std::clamp(static_cast<int>(mc.expert_used_count > 0 ? mc.expert_used_count : 8u), 1, E);
        const int F_exp = static_cast<int>(mc.expert_ffn_dim > 0 ? mc.expert_ffn_dim : mc.hidden_dim);
        impl_->d_moe_norm   = alloc_f32(Smax * static_cast<size_t>(D));
        impl_->d_moe_router = alloc_f32(Smax * static_cast<size_t>(E));
        impl_->d_moe_idx    = static_cast<int*>(impl_->alloc_gpu(Smax * static_cast<size_t>(K_used) * sizeof(int), scratch_bytes));
        impl_->d_moe_w      = alloc_f32(Smax * static_cast<size_t>(K_used));
        impl_->d_moe_act    = alloc_f32(Smax * static_cast<size_t>(K_used) * static_cast<size_t>(F_exp));
        impl_->d_moe_act_q8 = static_cast<block_q8_1_gpu*>(impl_->alloc_gpu(
            Smax * static_cast<size_t>(K_used) * static_cast<size_t>(F_exp >> 5) * sizeof(block_q8_1_gpu), scratch_bytes));
        impl_->d_moe_out    = alloc_f32(Smax * static_cast<size_t>(D));
        impl_->d_moe_exp_offsets = static_cast<int*>(impl_->alloc_gpu(257 * sizeof(int), scratch_bytes));
        impl_->d_moe_sorted_tok  = static_cast<int*>(impl_->alloc_gpu(Smax * static_cast<size_t>(K_used) * sizeof(int), scratch_bytes));
        impl_->d_moe_sorted_slot = static_cast<int*>(impl_->alloc_gpu(Smax * static_cast<size_t>(K_used) * sizeof(int), scratch_bytes));
        impl_->d_moe_sorted_w    = alloc_f32(Smax * static_cast<size_t>(K_used));
    }
    if (impl_->has_pleias) {
        impl_->d_pl_emb  = alloc_f32(Smax * static_cast<size_t>(L) * 256ULL);
        impl_->d_pl_proj = alloc_f32(Smax * static_cast<size_t>(L) * 256ULL);
        impl_->d_pl_gate = alloc_f32(Smax * 256ULL);
        impl_->d_pl_out  = alloc_f32(Smax * static_cast<size_t>(D));
    }

    impl_->d_x_f16 = static_cast<__half*>(impl_->alloc_gpu(Smax * max_in_dim * sizeof(__half), scratch_bytes));
    impl_->w_f16_cap_elems = std::min(max_w_elems, static_cast<size_t>(256ULL * 1024ULL * 1024ULL));
    impl_->d_w_f16 = static_cast<__half*>(impl_->alloc_gpu(impl_->w_f16_cap_elems * sizeof(__half), scratch_bytes));

    const bool core_scratch_ok =
        impl_->d_step_args && impl_->d_tok_ids && impl_->d_x && impl_->d_norm &&
        impl_->d_q && impl_->d_k && impl_->d_v && impl_->d_attn_h && impl_->d_attn_o &&
        impl_->d_ffn_g && impl_->d_ffn_u && impl_->d_ffn_d && impl_->d_logits &&
        impl_->d_am_vals && impl_->d_am_idx && impl_->d_am_out &&
        impl_->d_x_q8 && impl_->d_x_f16 && impl_->d_w_f16;
    const bool moe_scratch_ok = (mc.expert_count == 0) ||
        (impl_->d_moe_norm && impl_->d_moe_router && impl_->d_moe_idx && impl_->d_moe_w &&
         impl_->d_moe_act && impl_->d_moe_act_q8 && impl_->d_moe_out &&
         impl_->d_moe_exp_offsets && impl_->d_moe_sorted_tok && impl_->d_moe_sorted_slot && impl_->d_moe_sorted_w);
    const bool pleias_scratch_ok = (!impl_->has_pleias) ||
        (impl_->d_pl_emb && impl_->d_pl_proj && impl_->d_pl_gate && impl_->d_pl_out);
    if (!core_scratch_ok || !moe_scratch_ok || !pleias_scratch_ok) {
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
    if (cudaSetDevice(cfg_.device_id) != cudaSuccess) return false;
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
    if (pos < 0 || pos >= impl_->max_seq_len_runtime) {
        last_error_ = "CUDA decode position " + std::to_string(pos) + " out of KV cache bounds [0, " +
                      std::to_string(impl_->max_seq_len_runtime) + ").";
        return false;
    }
    if (cudaSetDevice(cfg_.device_id) != cudaSuccess) {
        last_error_ = "cudaSetDevice failed in forward_decode.";
        return false;
    }
    auto t0 = std::chrono::high_resolution_clock::now();
    bool ok = impl_->run_decode_step(token, pos, out_logits, out_next_token, last_error_);
    if (ok) {
        cudaError_t le = cudaGetLastError();
        if (le != cudaSuccess) {
            last_error_ = std::string("CUDA launch error: ") + cudaGetErrorString(le);
            ok = false;
        }
    }
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
    const int total = static_cast<int>(tokens.size());
    if (start_pos < 0 || start_pos + total > impl_->max_seq_len_runtime) {
        last_error_ = "CUDA prefill range [" + std::to_string(start_pos) + ", " +
                      std::to_string(start_pos + total) + ") exceeds KV cache capacity " +
                      std::to_string(impl_->max_seq_len_runtime) + ".";
        return false;
    }
    if (cudaSetDevice(cfg_.device_id) != cudaSuccess) {
        last_error_ = "cudaSetDevice failed in forward_prefill.";
        return false;
    }
    auto t0 = std::chrono::high_resolution_clock::now();
    int offset = 0;
    while (offset < total) {
        const int chunk = std::min(impl_->max_S, total - offset);
        const bool is_last = (offset + chunk == total);
        if (!impl_->run_prefill_chunk(tokens.data() + offset, chunk, start_pos + offset,
                                      is_last ? out_last_logits : nullptr,
                                      is_last ? out_next_token : nullptr,
                                      last_error_)) {
            return false;
        }
        offset += chunk;
    }
    cudaError_t le = cudaGetLastError();
    if (le != cudaSuccess) {
        last_error_ = std::string("CUDA prefill launch error: ") + cudaGetErrorString(le);
        return false;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    metrics_.last_prefill_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return true;
}

} // namespace haven
