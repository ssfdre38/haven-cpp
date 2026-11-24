#pragma once

#include "haven_types.h"
#include <immintrin.h>
#include <vector>
#include <cmath>
#include <cstring>

namespace haven {

// Cache-aligned 32-element INT8 activation buffer for high-throughput AVX2 vpmaddubsw/vpmaddwd GEMV
// Stack-backed up to 16,384 elements (512 blocks) for zero heap allocations on the hot inference path.
struct Q8Act32 {
    static constexpr int MAX_STACK_BLOCKS = 512;
    alignas(32) int8_t  qs_stack[MAX_STACK_BLOCKS * 32];
    alignas(32) float   d_stack[MAX_STACK_BLOCKS];
    alignas(32) float   sumf_stack[MAX_STACK_BLOCKS];
    alignas(32) int16_t bsum_stack[MAX_STACK_BLOCKS * 2];

    std::vector<int8_t>  qs_heap;
    std::vector<float>   d_heap;
    std::vector<float>   sumf_heap;
    std::vector<int16_t> bsum_heap;

    int8_t*  qs   = qs_stack;
    float*   d    = d_stack;
    float*   sumf = sumf_stack;
    int16_t* bsum = bsum_stack;
    int nb = 0;
};

class Avx2Math {
public:
    // Convert FP16 uint16_t to float using hardware native F16C SIMD instruction (MSVC/GCC/Clang portable)
    static inline float fp16_to_fp32(uint16_t h) {
        return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(static_cast<int>(h))));
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

    // Dot product of quantized Q5_0 weights with FP32 vector using AVX2 and FMA
    static float vec_dot_q5_0(int n, const block_q5_0* vx, const float* vy);

    // Dot product of quantized Q5_1 weights with FP32 vector
    static float vec_dot_q5_1(int n, const block_q5_1* vx, const float* vy);

    // Dot product of quantized Q5_K weights with FP32 vector using AVX2 and FMA
    static float vec_dot_q5_K(int n, const block_q5_K* vx, const float* vy);

    // Dot product of quantized Q3_K weights with FP32 vector using AVX2 and FMA
    static float vec_dot_q3_K(int n, const block_q3_K* vx, const float* vy);

    // Dot product of quantized IQ4_NL weights with FP32 vector
    static float vec_dot_iq4_nl(int n, const block_iq4_nl* vx, const float* vy);

    // Dot product of quantized IQ4_XS weights with FP32 vector
    static float vec_dot_iq4_xs(int n, const block_iq4_xs* vx, const float* vy);

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

    // Fast AVX2 In-Place Bias Addition: out[i] += bias[i] (supports F32 and F16)
    static void add_bias(float* out, const TensorDesc& bias_tensor, int size);

    // Dequantizes a single row of a tensor into float buffer (e.g. token embedding lookup)
    static void dequantize_row(float* out, const TensorDesc& weight, int row, int cols);

    // Matrix-Vector Multiplication: y = W * x
    static void gemv(float* y, const TensorDesc& weight, const float* x, int rows, int cols);

    // Matrix-Vector Multiplication on a contiguous row slice (e.g. expert e inside 3D [cols, rows, E] tensor)
    static void gemv_slice(float* y, const TensorDesc& weight, size_t row_offset, const float* x, int rows, int cols);

    // Single-row quantized/float dot product for fused parallel callers (no internal OpenMP fork)
    static float dot_row(const TensorDesc& weight, size_t row, const float* x, int cols);

    // Pre-quantize FP32 activation vector into 32-element dynamic-scale Q8Act32 blocks for integer AVX2 GEMV
    static void quantize_q8_act(const float* x, int cols, Q8Act32& out);

    // Function pointer type for branch-free per-row integer AVX2 dot product against pre-quantized Q8Act32
    using DotRowQ8Fn = float (*)(const void* data, size_t row, const Q8Act32& act, int cols);

    // Returns integer AVX2 Q8Act32 row-dot kernel if supported & `cols` is block-aligned, else nullptr
    static DotRowQ8Fn get_dot_row_q8_fn(QuantType type, int cols);

    // Single-row integer AVX2 dot product against pre-quantized Q8Act32 (falls back to 0 if unsupported)
    static float dot_row_q8(const TensorDesc& weight, size_t row, const Q8Act32& act, int cols);

    // Exact byte stride for one row of `cols` elements (0 if unsupported)
    static size_t row_bytes(QuantType type, int cols);

    // Bare-metal AVX2 Vector Norm: sqrt(sum(x^2))
    static float vector_norm(const float* x, int size);

    // Bare-metal AVX2 SIMD Cosine Similarity: (a . b) / (||a|| * ||b||)
    static float cosine_similarity(const float* a, const float* b, int size);
};

} // namespace haven