#include "haven/haven_module_abi.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>

namespace {

static const HavenModuleDescriptor kDesc = {
    HAVEN_MODULE_ABI_VERSION,
    HAVEN_MOD_BACKEND,
    "haven_backend_tpu",
    "Haven Native Cloud TPU v6e PJRT (libtpu.so) Backend Module",
    "1.0.0",
    "tpu,pjrt,trillium",
    HAVEN_FEAT_TURBO_FWHT | HAVEN_FEAT_MOE | HAVEN_FEAT_SWA | HAVEN_FEAT_K_EQ_V | HAVEN_FEAT_PLEIAS,
    110
};

} // namespace

HAVEN_MOD_EXPORT const HavenModuleDescriptor* haven_module_query_descriptor() {
    return &kDesc;
}

HAVEN_MOD_EXPORT int haven_backend_probe_runtime(HavenBackendProbeResult* out) {
    if (!out) return 0;
    std::memset(out, 0, sizeof(*out));
    out->supported_features = kDesc.feature_flags;
    out->preference_score = 110;

    const char* env_p = std::getenv("HAVEN_LIBTPU_PATH");
    const char* cands[4] = {
        env_p ? env_p : "",
        "/usr/local/lib/libtpu.so",
        "/usr/lib/libtpu.so",
        "/opt/tpu/libtpu.so"
    };
    for (const char* p : cands) {
        std::error_code ec;
        if (p && p[0] != '\0' && std::filesystem::exists(p, ec)) {
            out->available = 1u;
            out->device_count = 1u;
            out->total_mem_bytes = 32ULL * 1024ULL * 1024ULL * 1024ULL;
            out->free_mem_bytes = out->total_mem_bytes;
            std::snprintf(out->device_summary, sizeof(out->device_summary), "Cloud TPU PJRT (%s)", p);
            return 1;
        }
    }
    std::snprintf(out->device_summary, sizeof(out->device_summary), "No Cloud TPU PJRT library detected");
    return 1;
}
