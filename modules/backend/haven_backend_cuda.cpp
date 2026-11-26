#include "haven/haven_module_abi.h"
#include <cstdio>
#include <cstring>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace {

static const HavenModuleDescriptor kDesc = {
    HAVEN_MODULE_ABI_VERSION,
    HAVEN_MOD_BACKEND,
    "haven_backend_cuda",
    "Haven Native Zero-Compile CUDA V2 Backend Module",
    "2.0.0",
    "cuda,gpu,nvidia",
    HAVEN_FEAT_ZERO_COMPILE | HAVEN_FEAT_CUDA_GRAPHS | HAVEN_FEAT_TURBO_FWHT |
        HAVEN_FEAT_MOE | HAVEN_FEAT_SWA | HAVEN_FEAT_K_EQ_V | HAVEN_FEAT_PLEIAS,
    100
};

using CuInitFn = int (*)(unsigned int);
using CuDeviceGetCountFn = int (*)(int*);
using CuDeviceGetFn = int (*)(int*, int);
using CuDeviceGetNameFn = int (*)(char*, int, int);
using CuDeviceTotalMemFn = int (*)(size_t*, int);

} // namespace

HAVEN_MOD_EXPORT const HavenModuleDescriptor* haven_module_query_descriptor() {
    return &kDesc;
}

HAVEN_MOD_EXPORT int haven_backend_probe_runtime(HavenBackendProbeResult* out) {
    if (!out) return 0;
    std::memset(out, 0, sizeof(*out));
    out->supported_features = kDesc.feature_flags;
    out->preference_score = 100;

    // Lightweight dynamic driver probe (works even when compiled without CUDA Toolkit!)
#ifdef _WIN32
    HMODULE lib = LoadLibraryA("nvcuda.dll");
    if (!lib) {
        std::snprintf(out->device_summary, sizeof(out->device_summary), "No NVIDIA CUDA driver (nvcuda.dll absent)");
        return 1;
    }
    auto p_init  = reinterpret_cast<CuInitFn>(GetProcAddress(lib, "cuInit"));
    auto p_cnt   = reinterpret_cast<CuDeviceGetCountFn>(GetProcAddress(lib, "cuDeviceGetCount"));
    auto p_get   = reinterpret_cast<CuDeviceGetFn>(GetProcAddress(lib, "cuDeviceGet"));
    auto p_name  = reinterpret_cast<CuDeviceGetNameFn>(GetProcAddress(lib, "cuDeviceGetName"));
    auto p_mem   = reinterpret_cast<CuDeviceTotalMemFn>(GetProcAddress(lib, "cuDeviceTotalMem_v2"));
#else
    void* lib = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_LOCAL);
    if (!lib) {
        std::snprintf(out->device_summary, sizeof(out->device_summary), "No NVIDIA CUDA driver (libcuda.so.1 absent)");
        return 1;
    }
    auto p_init  = reinterpret_cast<CuInitFn>(dlsym(lib, "cuInit"));
    auto p_cnt   = reinterpret_cast<CuDeviceGetCountFn>(dlsym(lib, "cuDeviceGetCount"));
    auto p_get   = reinterpret_cast<CuDeviceGetFn>(dlsym(lib, "cuDeviceGet"));
    auto p_name  = reinterpret_cast<CuDeviceGetNameFn>(dlsym(lib, "cuDeviceGetName"));
    auto p_mem   = reinterpret_cast<CuDeviceTotalMemFn>(dlsym(lib, "cuDeviceTotalMem_v2"));
#endif

    int cnt = 0;
    if (p_init && p_cnt && p_init(0) == 0 && p_cnt(&cnt) == 0 && cnt > 0) {
        out->available = 1u;
        out->device_count = static_cast<uint32_t>(cnt);
        int dev = 0;
        char name[128] = "NVIDIA CUDA GPU";
        size_t mem = 0;
        if (p_get && p_get(&dev, 0) == 0) {
            if (p_name) p_name(name, static_cast<int>(sizeof(name)), dev);
            if (p_mem) p_mem(&mem, dev);
        }
        out->total_mem_bytes = static_cast<uint64_t>(mem);
        out->free_mem_bytes = static_cast<uint64_t>(mem);
        std::snprintf(out->device_summary, sizeof(out->device_summary),
                      "%s (%llu MiB VRAM, %d device%s)",
                      name,
                      static_cast<unsigned long long>(mem / (1024ULL * 1024ULL)),
                      cnt,
                      cnt > 1 ? "s" : "");
    } else {
        std::snprintf(out->device_summary, sizeof(out->device_summary), "CUDA driver present but 0 GPUs detected");
    }

#ifdef _WIN32
    FreeLibrary(lib);
#else
    dlclose(lib);
#endif
    return 1;
}
