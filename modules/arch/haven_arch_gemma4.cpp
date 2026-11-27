#include "haven/haven_module_abi.h"
#include <cctype>
#include <cstdio>
#include <cstring>

namespace {

static const HavenModuleDescriptor kDesc = {
    HAVEN_MODULE_ABI_VERSION,
    HAVEN_MOD_ARCH,
    "haven_arch_gemma4",
    "Haven Gemma 4 / 4.5 Turbo Architecture Module",
    "1.0.0",
    "gemma4,gemma3,gemma2,gemma",
    HAVEN_FEAT_SWA | HAVEN_FEAT_SHARED_KV | HAVEN_FEAT_K_EQ_V | HAVEN_FEAT_PLEIAS |
        HAVEN_FEAT_MOE | HAVEN_FEAT_QK_NORM | HAVEN_FEAT_TURBO_FWHT |
        HAVEN_FEAT_VISION_SIGLIP | HAVEN_FEAT_AUDIO_ENCODER |
        HAVEN_FEAT_SOFTCAP | HAVEN_FEAT_PROPORTIONAL_ROPE,
    100
};

static bool str_contains(const char* haystack, const char* needle) {
    if (!haystack || !needle) return false;
    const size_t nlen = std::strlen(needle);
    if (nlen == 0) return true;
    for (const char* p = haystack; *p != '\0'; ++p) {
        size_t i = 0;
        while (i < nlen && p[i] != '\0' &&
               std::tolower(static_cast<unsigned char>(p[i])) == std::tolower(static_cast<unsigned char>(needle[i]))) {
            ++i;
        }
        if (i == nlen) return true;
    }
    return false;
}

} // namespace

HAVEN_MOD_EXPORT const HavenModuleDescriptor* haven_module_query_descriptor() {
    return &kDesc;
}

HAVEN_MOD_EXPORT int haven_arch_resolve_spec(const HavenArchProbeInput* in, HavenArchRuntimeSpec* out) {
    if (!in || !out || !in->architecture) return 0;
    std::memset(out, 0, sizeof(*out));

    const bool is_g4 = str_contains(in->architecture, "gemma4");
    const bool is_g2 = str_contains(in->architecture, "gemma2");
    const bool is_gemma = str_contains(in->architecture, "gemma");
    if (!is_gemma) return 0;

    out->supported = 1u;
    out->ffn_activation = 1u; // GELU-tanh
    out->attention_scale = is_g4 ? 1.0f : 0.0f; // Gemma 4 uses scale=1.0 w/ QK-norm + unit V-norm
    out->final_logit_softcap = 30.0f;
    out->attn_logit_softcap = is_g2 ? 50.0f : 0.0f;
    out->use_embed_sqrt_scale = 1u;
    out->use_proportional_rope = is_g4 ? 1u : 0u;
    out->use_v_unit_rmsnorm = is_g4 ? 1u : 0u;
    out->requires_moe_cap = (in->expert_count > 0 || (in->detected_features & HAVEN_FEAT_MOE)) ? 1u : 0u;
    out->requires_turbo_cap = (in->turbo_version > 0 || (in->detected_features & HAVEN_FEAT_TURBO_FWHT)) ? 1u : 0u;
    out->requires_vision_cap = (in->detected_features & HAVEN_FEAT_VISION_SIGLIP) ? 1u : 0u;

    std::snprintf(out->chat_template_hint, sizeof(out->chat_template_hint), "%s",
                  is_g4 ? "gemma4_turn" : "gemma_legacy");
    std::snprintf(out->summary, sizeof(out->summary),
                  "%s | GeGLU | %s | Softcap=%.0f | SWA=%u | SharedKV=%u | MoE=%uE | Turbo=%u",
                  is_g4 ? "Gemma 4/4.5 Native" : "Gemma 2/3 Legacy",
                  is_g4 ? "AttnScale=1.0+PropRoPE+UnitVNorm" : "StdGemmaAttn",
                  out->final_logit_softcap,
                  in->sliding_window,
                  in->shared_kv_layers,
                  in->expert_count,
                  in->turbo_version);
    return 1;
}
