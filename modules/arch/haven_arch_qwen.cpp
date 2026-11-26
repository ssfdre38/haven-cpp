#include "haven/haven_module_abi.h"
#include <cstdio>
#include <cstring>

namespace {

static const HavenModuleDescriptor kDesc = {
    HAVEN_MODULE_ABI_VERSION,
    HAVEN_MOD_ARCH,
    "haven_arch_qwen",
    "Haven Qwen 2 / 2.5 / 3 Architecture Module",
    "1.0.0",
    "qwen2,qwen2moe,qwen3,qwen3moe",
    HAVEN_FEAT_QKV_BIAS | HAVEN_FEAT_QK_NORM | HAVEN_FEAT_MOE,
    95
};

static bool str_contains(const char* haystack, const char* needle) {
    return haystack && needle && std::strstr(haystack, needle) != nullptr;
}

} // namespace

HAVEN_MOD_EXPORT const HavenModuleDescriptor* haven_module_query_descriptor() {
    return &kDesc;
}

HAVEN_MOD_EXPORT int haven_arch_resolve_spec(const HavenArchProbeInput* in, HavenArchRuntimeSpec* out) {
    if (!in || !out || !in->architecture) return 0;
    std::memset(out, 0, sizeof(*out));
    if (!str_contains(in->architecture, "qwen")) return 0;

    const bool has_qkv_bias = (in->detected_features & HAVEN_FEAT_QKV_BIAS) != 0;
    const bool has_qk_norm = (in->detected_features & HAVEN_FEAT_QK_NORM) != 0;

    out->supported = 1u;
    out->ffn_activation = 0u; // SiLU SwiGLU
    out->attention_scale = 0.0f; // 1 / sqrt(head_dim)
    out->final_logit_softcap = 0.0f;
    out->attn_logit_softcap = 0.0f;
    out->use_embed_sqrt_scale = 0u;
    out->use_proportional_rope = 0u;
    out->use_v_unit_rmsnorm = 0u;
    out->requires_moe_cap = (in->expert_count > 0 || (in->detected_features & HAVEN_FEAT_MOE)) ? 1u : 0u;
    out->requires_turbo_cap = (in->turbo_version > 0) ? 1u : 0u;
    out->requires_vision_cap = (in->detected_features & HAVEN_FEAT_VISION_SIGLIP) ? 1u : 0u;

    std::snprintf(out->chat_template_hint, sizeof(out->chat_template_hint), "chatml");
    std::snprintf(out->summary, sizeof(out->summary),
                  "Qwen Native | SwiGLU(SiLU) | QKV-Bias=%s | QK-Norm=%s | MoE=%uE",
                  has_qkv_bias ? "ON" : "OFF",
                  has_qk_norm ? "ON" : "OFF",
                  in->expert_count);
    return 1;
}
