#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>
#include <memory>

namespace haven {

// Supported GGUF Tensor Quantization Types
enum class QuantType : uint32_t {
    F32  = 0,
    F16  = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    IQ4_NL = 20,
    IQ4_XS = 23,
    I8   = 24,
    I16  = 25,
    I32  = 26,
    I64  = 27,
    F64  = 28,
    BF16 = 30,
    COUNT
};

// 32-value quantized block for Q8_0
#pragma pack(push, 1)
struct block_q8_0 {
    uint16_t d;       // FP16 scale factor
    int8_t   qs[32];  // 32 int8 quantized weights
};
#pragma pack(pop)

// 32-value quantized block for Q4_0
#pragma pack(push, 1)
struct block_q4_0 {
    uint16_t d;       // FP16 scale factor
    uint8_t  qs[16];  // 32 4-bit quantized weights (2 per byte)
};
#pragma pack(pop)

// 256-value quantized block for Q4_K
#pragma pack(push, 1)
struct block_q4_K {
    uint16_t d;           // Super-block scale (FP16)
    uint16_t dmin;        // Super-block min (FP16)
    uint8_t  scales[12];  // 6-bit sub-block scales
    uint8_t  qs[128];     // 4-bit quantized weights (2 per byte)
};
#pragma pack(pop)

// 256-value quantized block for Q6_K
#pragma pack(push, 1)
struct block_q6_K {
    uint8_t  ql[128];      // lower 4 bits of 256 weights
    uint8_t  qh[64];       // upper 2 bits of 256 weights (4 per byte)
    int8_t   scales[16];   // 16 8-bit sub-block scales
    uint16_t d;            // Super-block scale (FP16)
};
#pragma pack(pop)

// 32-value quantized block for Q5_0
#pragma pack(push, 1)
struct block_q5_0 {
    uint16_t d;       // FP16 scale factor
    uint8_t  qh[4];   // 32 high bits (1 bit per weight)
    uint8_t  qs[16];  // 32 low 4-bit nibbles (2 per byte)
};
#pragma pack(pop)

// 256-value quantized block for Q5_K
#pragma pack(push, 1)
struct block_q5_K {
    uint16_t d;          // Super-block scale (FP16)
    uint16_t dmin;       // Super-block min (FP16)
    uint8_t  scales[12]; // 6-bit sub-block scales/mins
    uint8_t  qh[32];     // 256 high bits (1 bit per weight)
    uint8_t  qs[128];    // 4-bit quantized weights (2 per byte)
};
#pragma pack(pop)

// 32-value quantized block for Q5_1
#pragma pack(push, 1)
struct block_q5_1 {
    uint16_t d;       // FP16 delta
    uint16_t m;       // FP16 min
    uint8_t  qh[4];   // 32 high bits (1 bit per weight)
    uint8_t  qs[16];  // 32 low 4-bit nibbles (2 per byte)
};
#pragma pack(pop)
static_assert(sizeof(block_q5_1) == 24, "block_q5_1 must be 24 bytes");

// 32-value non-linear 4-bit quantized block for IQ4_NL
#pragma pack(push, 1)
struct block_iq4_nl {
    uint16_t d;       // FP16 scale factor
    uint8_t  qs[16];  // 32 4-bit LUT indices (2 per byte)
};
#pragma pack(pop)
static_assert(sizeof(block_iq4_nl) == 18, "block_iq4_nl must be 18 bytes");

// 256-value non-linear 4-bit super-block for IQ4_XS
#pragma pack(push, 1)
struct block_iq4_xs {
    uint16_t d;           // Super-block scale (FP16)
    uint16_t scales_h;    // Upper 2 bits of 8 6-bit sub-block scales
    uint8_t  scales_l[4]; // Lower 4 bits of 8 6-bit sub-block scales
    uint8_t  qs[128];     // 256 4-bit LUT indices (2 per byte)
};
#pragma pack(pop)
static_assert(sizeof(block_iq4_xs) == 136, "block_iq4_xs must be 136 bytes");

// 256-value quantized block for Q3_K
#pragma pack(push, 1)
struct block_q3_K {
    uint8_t  hmask[32];   // 256 high bits (1 bit per weight)
    uint8_t  qs[64];      // 256 2-bit quantized weights (4 per byte)
    uint8_t  scales[12];  // 16 6-bit scales
    uint16_t d;           // Super-block scale (FP16)
};
#pragma pack(pop)

struct TensorDesc {
    std::string name;
    QuantType type = QuantType::F32;
    std::vector<uint64_t> shape; // [ne0, ne1, ne2, ne3]
    uint64_t offset = 0;         // Byte offset in GGUF file
    size_t size_bytes = 0;       // Total byte size
    const void* data = nullptr;  // Direct mmap pointer
};

enum class FfnActivation : uint32_t {
    SILU = 0,
    GELU = 1
};

struct ModelConfig {
    std::string architecture = "llama";
    uint32_t vocab_size = 32000;
    uint32_t embedding_dim = 4096;
    uint32_t hidden_dim = 11008;
    uint32_t num_layers = 32;
    uint32_t num_heads = 32;
    uint32_t num_kv_heads = 0;
    uint32_t head_dim = 0;
    uint32_t head_dim_swa = 0;
    uint32_t max_context_length = 131072;
    float rope_freq_base = 1000000.0f;
    float rope_freq_base_swa = 10000.0f;
    float rope_freq_scale = 1.0f;
    uint32_t rope_dimension_count = 512;
    uint32_t rope_dimension_count_swa = 256;
    float rms_norm_eps = 1e-5f;
    uint32_t sliding_window = 0;
    uint32_t shared_kv_layers = 0;
    bool has_shared_kv_layers_kv = false;
    uint32_t expert_count = 0;
    uint32_t expert_used_count = 0;
    uint32_t expert_ffn_dim = 0;
    std::vector<uint32_t> hidden_dim_per_layer;
    std::vector<uint32_t> head_count_kv_per_layer;
    std::vector<bool> sliding_window_pattern;
    float final_logit_softcapping = 0.0f;
    float attn_logit_softcapping = 0.0f;
    bool is_gemma = false;
    FfnActivation ffn_activation = FfnActivation::SILU;

    // Gemma 4.5 Turbo Baked Architecture Metadata (0 / false for standard unmodified GGUFs)
    uint32_t turbo_version = 0;
    uint32_t turbo_hadamard_qkv = 0;
    uint32_t turbo_hadamard_ffn = 0;
    uint32_t turbo_hadamard_ffn_down = 0;
    uint32_t turbo_hadamard_moe_in = 0;
    uint32_t turbo_hadamard_moe_down = 0;
    bool     turbo_folded_norms = false;
    bool     turbo_folded_scales = false;

    // GGUF Tokenizer Metadata (overrides heuristic vocab scans when present)
    bool     has_bos_token_id = false;
    uint32_t bos_token_id = 2;
    bool     has_eos_token_id = false;
    uint32_t eos_token_id = 1;
    bool     has_pad_token_id = false;
    uint32_t pad_token_id = 0;
    bool     has_unk_token_id = false;
    uint32_t unk_token_id = 3;
    bool     has_add_space_prefix = false;
    bool     add_space_prefix = true;
};

} // namespace haven