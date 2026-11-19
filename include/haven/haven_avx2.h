#pragma once

#include "haven_types.h"
#include <immintrin.h>
#include <vector>
#include <cmath>
#include <cstring>

namespace haven {

class Avx2Math {
public:
    // Convert FP16 uint16_t to float using hardware native F16C SIMD instruction
    static inline float fp16_to_fp32(uint16_t h) {
        return _cvtsh_ss(h);
    }

    // Convert BF16 uint16_t to float
    static inline float bf16_to_fp32(uint16_t b) {
        uint32_t val = (uint32_t)b << 16;
        float res = 0.0f;
        std::memcpy(&res, &val, sizeof(float));
        return res;
    }

    // Dot product of quantized Q8_0 weights with FP32 vector using AVX2 and FMA
    static float vec_dot_q8_0(int n, const block_q8_0* vx, const float* vy);

    // Dot product of quantized Q4_0 weights with FP32 vector using AVX2 and FMA
    static float vec_dot_q4_0(int n, const block_q4_0* vx, const float* vy);

    // Dot product of quantized Q4_K weights with FP32 vector using AVX2 and FMA
    static float vec_dot_q4_K(int n, const block_q4_K* vx, const float* vy);

    // Dot product of quantized Q6_K weights with FP32 vector using AVX2 and FMA
    static float vec_dot_q6_K(int n, const block_q6_K* vx, const float* vy);

    // Dot product of FP16 weights with FP32 vector
    static float vec_dot_f16(int n, const uint16_t* vx, const float* vy);

    // Dot product of BF16 weights with FP32 vector
    static float vec_dot_bf16(int n, const uint16_t* vx, const float* vy);

    // Dot product of FP32 weights with FP32 vector
    static float vec_dot_f32(int n, const float* vx, const float* vy);

    // In-place RMS Normalization: y = x / sqrt(mean(x^2) + eps) * weight
    static void rms_norm(float* out, const float* x, const float* weight, int size, float eps);

    // RoPE Rotary Position Embedding Kernel with partial dimension support
    static void apply_rope(
        float* q, float* k,
        int head_dim, int num_heads, int num_kv_heads,
        int pos, float freq_base = 10000.0f, float freq_scale = 1.0f,
        const float* freq_factors = nullptr,
        int rope_dim = 0
    );

    // Fast In-Place Softmax over an array of floats
    static void softmax(float* x, int size);

    // Fast In-Place SiLU Activation: x * sigmoid(x)
    static void silu(float* x, int size);

    // Fast In-Place GELU Activation for Gemma 4 (GeGLU)
    static void gelu(float* x, int size);

    // Gemma 4 Logits Softcapping: logits = cap * tanh(logits / cap)
    static void softcap_logits(float* logits, int size, float cap = 30.0f);

    // Dequantizes a single row of a tensor into float buffer (e.g. token embedding lookup)
    static void dequantize_row(float* out, const TensorDesc& weight, int row, int cols);

    // Matrix-Vector Multiplication: y = W * x
    static void gemv(float* y, const TensorDesc& weight, const float* x, int rows, int cols);

    // Bare-metal AVX2 Vector Norm: sqrt(sum(x^2))
    static float vector_norm(const float* x, int size);

    // Bare-metal AVX2 SIMD Cosine Similarity: (a . b) / (||a|| * ||b||)
    static float cosine_similarity(const float* a, const float* b, int size);
};

} // namespace haven