#pragma once

#include <cstddef>
#include <cstdint>

#ifdef _WIN32
#define HAVEN_MOD_EXPORT extern "C" __declspec(dllexport)
#else
#define HAVEN_MOD_EXPORT extern "C" __attribute__((visibility("default")))
#endif

// ============================================================================
// Haven Microkernel Module C-ABI (Zero-Bloat Dynamic Subsystem Contract)
// ============================================================================
// Allows haven-core to stay lean by dynamically loading ONLY the exact
// architecture, silicon backend, and optional capability modules (.dll / .so)
// dictated by a model's GGUF header DNA + host hardware profile.
// ============================================================================

static constexpr uint32_t HAVEN_MODULE_ABI_VERSION = 1u;

enum HavenModuleKind : uint32_t {
    HAVEN_MOD_ARCH       = 1u, // Model architecture graph & runtime contract (gemma4, llama, qwen, phi, ...)
    HAVEN_MOD_BACKEND    = 2u, // Compute silicon backend (avx2, cuda, tpu, mlx)
    HAVEN_MOD_CAPABILITY = 3u  // Optional subsystem (moe, turbo_fwht, vision_siglip, audio, hmb_kv)
};

enum HavenFeatureFlags : uint64_t {
    HAVEN_FEAT_NONE           = 0ULL,
    HAVEN_FEAT_SWA            = 1ULL << 0,
    HAVEN_FEAT_SHARED_KV      = 1ULL << 1,
    HAVEN_FEAT_K_EQ_V         = 1ULL << 2,
    HAVEN_FEAT_PLEIAS         = 1ULL << 3,
    HAVEN_FEAT_MOE            = 1ULL << 4,
    HAVEN_FEAT_QKV_BIAS       = 1ULL << 5,
    HAVEN_FEAT_QK_NORM        = 1ULL << 6,
    HAVEN_FEAT_TURBO_FWHT     = 1ULL << 7,
    HAVEN_FEAT_VISION_SIGLIP  = 1ULL << 8,
    HAVEN_FEAT_AUDIO_ENCODER  = 1ULL << 9,
    HAVEN_FEAT_SOFTCAP        = 1ULL << 10,
    HAVEN_FEAT_PROPORTIONAL_ROPE = 1ULL << 11,
    HAVEN_FEAT_ZERO_COMPILE   = 1ULL << 12,
    HAVEN_FEAT_CUDA_GRAPHS    = 1ULL << 13,
    HAVEN_FEAT_HMB_VMM        = 1ULL << 14
};

struct HavenModuleDescriptor {
    uint32_t abi_version;         // Must equal HAVEN_MODULE_ABI_VERSION
    uint32_t kind;                // HavenModuleKind
    const char* id;               // Canonical unique ID, e.g. "haven_arch_gemma4"
    const char* name;             // Human-readable name
    const char* version;          // Semver string
    const char* matches_csv;      // Comma-separated arch/backend/cap tags this module serves
    uint64_t feature_flags;       // Bitmask of HavenFeatureFlags supported
    int32_t  priority;            // Higher wins on tie
};

struct HavenArchProbeInput {
    const char* architecture;     // Raw GGUF general.architecture string
    uint32_t num_layers;
    uint32_t embedding_dim;
    uint32_t hidden_dim;
    uint32_t num_heads;
    uint32_t num_kv_heads;
    uint32_t head_dim;
    uint32_t head_dim_swa;
    uint32_t sliding_window;
    uint32_t shared_kv_layers;
    uint32_t expert_count;
    uint32_t expert_used_count;
    uint32_t turbo_version;
    uint64_t detected_features;   // Bitmask of HavenFeatureFlags detected in GGUF header
};

struct HavenArchRuntimeSpec {
    uint32_t supported;           // 1 if module handles this model, 0 otherwise
    uint32_t ffn_activation;      // 0 = SiLU, 1 = GELU-tanh
    float    attention_scale;     // 0.0f = 1/sqrt(head_dim), >0 = explicit (1.0f for Gemma 4)
    float    final_logit_softcap; // 0.0f = disabled, 30.0f for Gemma 2/3/4
    float    attn_logit_softcap;  // 0.0f = disabled, 50.0f for Gemma 2
    uint32_t use_embed_sqrt_scale;// 1 for Gemma family (sqrt(D)), 0 otherwise
    uint32_t use_proportional_rope;// 1 for Gemma 4 global layers (partial_rotary_factor)
    uint32_t use_v_unit_rmsnorm;  // 1 for Gemma 4 V parameter-free RMSNorm
    uint32_t requires_moe_cap;    // 1 if haven_cap_moe should be loaded
    uint32_t requires_turbo_cap;  // 1 if haven_cap_turbo should be loaded
    uint32_t requires_vision_cap; // 1 if haven_cap_vision should be loaded
    char     chat_template_hint[32];
    char     summary[192];
};

struct HavenBackendProbeResult {
    uint32_t available;           // 1 if silicon/runtime is present and usable
    uint32_t device_count;
    uint64_t total_mem_bytes;
    uint64_t free_mem_bytes;
    uint64_t supported_features;  // HavenFeatureFlags bitmask
    int32_t  preference_score;    // Baseline ranking (CUDA/TPU > MLX > AVX2)
    char     device_summary[160];
};

// Required C-ABI entry points exported by every Haven Microkernel module
using HavenQueryDescriptorFn = const HavenModuleDescriptor* (*)();

// Optional role-specific entry points (present per HavenModuleKind)
using HavenArchResolveSpecFn = int (*)(const HavenArchProbeInput* in_probe, HavenArchRuntimeSpec* out_spec);
using HavenBackendProbeFn    = int (*)(HavenBackendProbeResult* out_result);
