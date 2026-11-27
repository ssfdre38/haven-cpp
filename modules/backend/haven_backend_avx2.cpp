#include "haven/haven_module_abi.h"
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <thread>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <intrin.h>
#else
#include <fstream>
#include <sstream>
#include <string>
#if defined(__x86_64__) || defined(__i386__)
#include <cpuid.h>
#endif
#endif

namespace {

static const HavenModuleDescriptor kDesc = {
    HAVEN_MODULE_ABI_VERSION,
    HAVEN_MOD_BACKEND,
    "haven_backend_avx2",
    "Haven Native CPU AVX2 + FMA + OpenMP Backend Module",
    "1.0.0",
    "cpu,avx2",
    HAVEN_FEAT_ZERO_COMPILE | HAVEN_FEAT_TURBO_FWHT | HAVEN_FEAT_MOE | HAVEN_FEAT_HMB_VMM,
    50
};

} // namespace

HAVEN_MOD_EXPORT const HavenModuleDescriptor* haven_module_query_descriptor() {
    return &kDesc;
}

HAVEN_MOD_EXPORT int haven_backend_probe_runtime(HavenBackendProbeResult* out) {
    if (!out) return 0;
    std::memset(out, 0, sizeof(*out));
    out->available = 1u;
    out->device_count = std::thread::hardware_concurrency();
    if (out->device_count == 0) out->device_count = 1u;
    out->supported_features = HAVEN_FEAT_ZERO_COMPILE | HAVEN_FEAT_TURBO_FWHT | HAVEN_FEAT_MOE;
    out->preference_score = 50;

    bool has_avx2 = false;
    bool has_hmb_vmm = false;

#ifdef _WIN32
    int info[4] = {0, 0, 0, 0};
    __cpuidex(info, 7, 0);
    has_avx2 = (info[1] & (1 << 5)) != 0;
    MEMORYSTATUSEX ms{};
    ms.dwLength = sizeof(ms);
    if (GlobalMemoryStatusEx(&ms)) {
        out->total_mem_bytes = static_cast<uint64_t>(ms.ullTotalPhys);
        out->free_mem_bytes = static_cast<uint64_t>(ms.ullAvailPhys);
    }
#else
#if defined(__x86_64__) || defined(__i386__)
    unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
    if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
        has_avx2 = (ebx & (1u << 5)) != 0;
    }
#endif
    std::ifstream mi("/proc/meminfo");
    std::string line;
    while (std::getline(mi, line)) {
        if (line.rfind("MemTotal:", 0) == 0) {
            uint64_t kb = 0;
            std::istringstream iss(line.substr(9));
            if (iss >> kb) out->total_mem_bytes = kb * 1024ULL;
        } else if (line.rfind("MemAvailable:", 0) == 0) {
            uint64_t kb = 0;
            std::istringstream iss(line.substr(13));
            if (iss >> kb) out->free_mem_bytes = kb * 1024ULL;
        }
    }
    std::error_code ec1, ec2;
    has_hmb_vmm = std::filesystem::exists("/proc/hmb_pagefile", ec1) || std::filesystem::exists("/proc/hmb_swap", ec2);
    if (has_hmb_vmm) out->supported_features |= HAVEN_FEAT_HMB_VMM;
#endif

    std::snprintf(out->device_summary, sizeof(out->device_summary),
                  "CPU %u threads (%s)%s",
                  out->device_count,
                  has_avx2 ? "AVX2+FMA" : "Scalar",
                  has_hmb_vmm ? " + HMB-VMM" : "");
    return 1;
}
