#pragma once

#include "haven_types.h"
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace haven {

// ============================================================================
// Single-Source GGUF Quantization & Numeric Conversion Specification
// ============================================================================
// Shared across CPU AVX2, Native CUDA, Cloud TPU PJRT, GGUF Loader,
// Model Intelligence Probe, and haven-bake to guarantee zero constant drift
// and exact cross-backend parity.
// ============================================================================

inline constexpr int8_t kQuantIq4NlLut[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10,
       1,   13,  25,  38,  53,  69,  89, 113
};

struct QuantBlockSpec {
    uint32_t block_elems;
    uint32_t block_bytes;
};

inline constexpr QuantBlockSpec quant_block_spec(QuantType type) {
    switch (type) {
        case QuantType::F32:    return {1u,   4u};
        case QuantType::F16:
        case QuantType::BF16:   return {1u,   2u};
        case QuantType::Q4_0:   return {32u,  static_cast<uint32_t>(sizeof(block_q4_0))};
        case QuantType::Q4_1:   return {32u,  20u};
        case QuantType::Q5_0:   return {32u,  static_cast<uint32_t>(sizeof(block_q5_0))};
        case QuantType::Q5_1:   return {32u,  static_cast<uint32_t>(sizeof(block_q5_1))};
        case QuantType::Q8_0:   return {32u,  static_cast<uint32_t>(sizeof(block_q8_0))};
        case QuantType::Q8_1:   return {32u,  36u};
        case QuantType::Q2_K:   return {256u, 84u};
        case QuantType::Q3_K:   return {256u, static_cast<uint32_t>(sizeof(block_q3_K))};
        case QuantType::Q4_K:   return {256u, static_cast<uint32_t>(sizeof(block_q4_K))};
        case QuantType::Q5_K:   return {256u, static_cast<uint32_t>(sizeof(block_q5_K))};
        case QuantType::Q6_K:   return {256u, static_cast<uint32_t>(sizeof(block_q6_K))};
        case QuantType::Q8_K:   return {256u, 292u};
        case QuantType::IQ4_NL: return {32u,  static_cast<uint32_t>(sizeof(block_iq4_nl))};
        case QuantType::IQ4_XS: return {256u, static_cast<uint32_t>(sizeof(block_iq4_xs))};
        case QuantType::I8:     return {1u,   1u};
        case QuantType::I16:    return {1u,   2u};
        case QuantType::I32:    return {1u,   4u};
        case QuantType::I64:
        case QuantType::F64:    return {1u,   8u};
        default:                return {1u,   1u};
    }
}

// Overflow-safe total byte size calculation for any GGUF tensor shape & QuantType
inline uint64_t quant_tensor_byte_size(QuantType type, const std::vector<uint64_t>& shape) {
    if (shape.empty()) return 0;
    uint64_t n_elems = 1;
    for (uint64_t dim : shape) {
        if (dim == 0 || dim > std::numeric_limits<uint64_t>::max() / n_elems) return 0;
        n_elems *= dim;
    }
    const QuantBlockSpec spec = quant_block_spec(type);
    const uint64_t be = static_cast<uint64_t>(spec.block_elems);
    const uint64_t bb = static_cast<uint64_t>(spec.block_bytes);
    if (be == 0 || n_elems > std::numeric_limits<uint64_t>::max() - (be - 1ULL)) return 0;
    const uint64_t blocks = (n_elems + be - 1ULL) / be;
    if (blocks > std::numeric_limits<uint64_t>::max() / bb) return 0;
    return blocks * bb;
}

inline float quant_bf16_to_f32(uint16_t b) {
    uint32_t u = static_cast<uint32_t>(b) << 16;
    float f = 0.0f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

inline uint16_t quant_f32_to_bf16_rne(float f) {
    uint32_t u = 0;
    std::memcpy(&u, &f, sizeof(u));
    if ((u & 0x7FFFFFFFu) > 0x7F800000u) {
        return static_cast<uint16_t>((u >> 16) | 0x0040u); // preserve quiet NaN
    }
    u += 0x00007FFFu + ((u >> 16) & 1u);
    return static_cast<uint16_t>(u >> 16);
}

inline float quant_fp16_to_f32_soft(uint16_t h) {
    uint32_t sign = (static_cast<uint32_t>(h & 0x8000u)) << 16;
    uint32_t exp  = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x03FFu;
    uint32_t f_bits = 0;
    if (exp == 0) {
        if (mant == 0) {
            f_bits = sign;
        } else {
            exp = 1;
            while ((mant & 0x0400u) == 0) {
                mant <<= 1;
                --exp;
            }
            mant &= 0x03FFu;
            f_bits = sign | ((exp + 112u) << 23) | (mant << 13);
        }
    } else if (exp == 31u) {
        f_bits = sign | 0x7F800000u | (mant << 13);
    } else {
        f_bits = sign | ((exp + 112u) << 23) | (mant << 13);
    }
    float out = 0.0f;
    std::memcpy(&out, &f_bits, sizeof(out));
    return out;
}

inline void quant_get_scale_min_k4(int j, const uint8_t* q, uint8_t& d, uint8_t& m) {
    if (j < 4) {
        d = q[j] & 63u;
        m = q[j + 4] & 63u;
    } else {
        d = static_cast<uint8_t>((q[j + 4] & 0x0Fu) | ((q[j - 4] >> 6) << 4));
        m = static_cast<uint8_t>((q[j + 4] >> 4)    | ((q[j]     >> 6) << 4));
    }
}

} // namespace haven
