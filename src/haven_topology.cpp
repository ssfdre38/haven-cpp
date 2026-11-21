#include "haven/haven_topology.h"
#include <sstream>
#include <iostream>
#include <thread>
#include <cstdlib>
#include <fstream>
#include <filesystem>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <unistd.h>
#include <sys/sysinfo.h>
#endif

namespace haven {

TopologyManager::TopologyManager() {
    detect_topology();
}

TopologyManager::~TopologyManager() = default;

void TopologyManager::detect_topology() {
    nodes_.clear();
    has_tpu_ = false;
    has_gpu_ = false;
    tpu_chip_count_ = 0;

    // 1. Detect Host CPU Node & Unified Memory
    ComputeNode cpu_node;
    cpu_node.node_id = 0;
    cpu_node.backend = DeviceBackend::CPU_AVX2;
    cpu_node.device_name = "Host x86_64 AVX2 Core Complex";
    cpu_node.numa_domain = 0;
    cpu_node.primary_bus = InterconnectBus::SYSTEM_RAM_BUS;
    cpu_node.is_available = true;

    uint32_t logical_cores = std::thread::hardware_concurrency();
    if (logical_cores == 0) logical_cores = 8;
    cpu_node.peak_tflops = static_cast<float>(logical_cores) * 0.12f; // Estimated AVX2 FMA throughput

#ifdef _WIN32
    MEMORYSTATUSEX mem_status;
    mem_status.dwLength = sizeof(mem_status);
    if (GlobalMemoryStatusEx(&mem_status)) {
        cpu_node.total_memory_bytes = mem_status.ullTotalPhys;
        cpu_node.free_memory_bytes  = mem_status.ullAvailPhys;
    } else {
        cpu_node.total_memory_bytes = 64ULL * 1024 * 1024 * 1024; // Fallback 64GB
        cpu_node.free_memory_bytes  = 32ULL * 1024 * 1024 * 1024;
    }
#else
    struct sysinfo si;
    if (sysinfo(&si) == 0) {
        cpu_node.total_memory_bytes = static_cast<size_t>(si.totalram) * si.mem_unit;
        cpu_node.free_memory_bytes  = static_cast<size_t>(si.freeram) * si.mem_unit;
    } else {
        cpu_node.total_memory_bytes = 64ULL * 1024 * 1024 * 1024;
        cpu_node.free_memory_bytes  = 32ULL * 1024 * 1024 * 1024;
    }
#endif

    cpu_node.memory_bandwidth_gbps = 68.0f; // Standard dual-channel DDR5 bus floor
    nodes_.push_back(cpu_node);

    // 2. Probe for Native TPU Accelerator Interface (Sysfs, VFIO, PJRT, Cloud Environment)
    size_t detected_tpu_chips = 0;
#ifndef _WIN32
    try {
        if (std::filesystem::exists("/sys/bus/pci/devices")) {
            for (const auto& entry : std::filesystem::directory_iterator("/sys/bus/pci/devices")) {
                auto dev_path = entry.path() / "device";
                auto ven_path = entry.path() / "vendor";
                if (std::filesystem::exists(dev_path) && std::filesystem::exists(ven_path)) {
                    std::ifstream dev_file(dev_path);
                    std::ifstream ven_file(ven_path);
                    std::string dev_id, ven_id;
                    if (dev_file >> dev_id && ven_file >> ven_id) {
                        // Google PCI Vendor: 0x1ae0, TPU v6e Trillium Device: 0x006f
                        if (ven_id.find("1ae0") != std::string::npos && dev_id.find("006f") != std::string::npos) {
                            detected_tpu_chips++;
                        }
                    }
                }
            }
        }
    } catch (...) {}
#endif

    const char* tpu_env = std::getenv("TPU_NAME");
    const char* pjrt_env = std::getenv("PJRT_DEVICE");

    if (detected_tpu_chips > 0 || tpu_env || (pjrt_env && std::string(pjrt_env).find("TPU") != std::string::npos)) {
        has_tpu_ = true;
        tpu_chip_count_ = detected_tpu_chips > 0 ? detected_tpu_chips : 1;

        ComputeNode tpu_node;
        tpu_node.node_id = static_cast<uint32_t>(nodes_.size());
        tpu_node.backend = DeviceBackend::TPU_PJRT;

        std::string name_suffix;
        if (tpu_env) {
            name_suffix = std::string(tpu_env);
        } else {
            char hostname[256] = {0};
#ifdef _WIN32
            DWORD sz = sizeof(hostname);
            GetComputerNameA(hostname, &sz);
#else
            gethostname(hostname, sizeof(hostname));
#endif
            if (hostname[0] != '\0') {
                name_suffix = hostname;
            }
            if (name_suffix.empty()) name_suffix = "haven-tpu-v6e";
        }

        if (tpu_chip_count_ > 1) {
            tpu_node.device_name = "Google TPU v6e Trillium Torus Fabric (" + std::to_string(tpu_chip_count_) + "x Cores, " + name_suffix + ")";
        } else {
            tpu_node.device_name = "Google TPU Silicon (" + name_suffix + ")";
        }

        tpu_node.numa_domain = 1;
        tpu_node.total_memory_bytes = tpu_chip_count_ * 32ULL * 1024 * 1024 * 1024; // 32GB HBM per TPU chip
        tpu_node.free_memory_bytes  = tpu_chip_count_ * 30ULL * 1024 * 1024 * 1024;
        tpu_node.peak_tflops = static_cast<float>(tpu_chip_count_) * 275.0f; // 275 TFLOPS per chip -> 2,200 TFLOPS for 8 chips
        tpu_node.memory_bandwidth_gbps = static_cast<float>(tpu_chip_count_) * 1200.0f; // 1,200 GB/s per chip -> 9,600 GB/s
        tpu_node.primary_bus = InterconnectBus::OPTICAL_ICI;
        tpu_node.is_available = true;

        nodes_.push_back(tpu_node);
    }

    // 3. Probe for Discrete GPU Topology (CUDA/Vulkan/DirectML Device Node)
    const char* gpu_env = std::getenv("CUDA_VISIBLE_DEVICES");
    if (gpu_env && std::string(gpu_env) != "-1") {
        ComputeNode gpu_node;
        gpu_node.node_id = static_cast<uint32_t>(nodes_.size());
        gpu_node.backend = DeviceBackend::GPU_NUMA;
        gpu_node.device_name = "Discrete Accelerator GPU (Direct Topology)";
        gpu_node.numa_domain = 2;
        gpu_node.total_memory_bytes = 24ULL * 1024 * 1024 * 1024; // 24GB VRAM
        gpu_node.free_memory_bytes  = 22ULL * 1024 * 1024 * 1024;
        gpu_node.peak_tflops = 165.0f;
        gpu_node.memory_bandwidth_gbps = 1008.0f;
        gpu_node.primary_bus = InterconnectBus::PCIE_GEN4_GEN5;
        gpu_node.is_available = true;

        nodes_.push_back(gpu_node);
        has_gpu_ = true;
    }
}

uint32_t TopologyManager::select_optimal_node(size_t flops_required, size_t memory_footprint, bool requires_accel) const {
    if (requires_accel) {
        for (const auto& node : nodes_) {
            if ((node.backend == DeviceBackend::TPU_PJRT || node.backend == DeviceBackend::GPU_NUMA) && node.is_available) {
                if (node.free_memory_bytes >= memory_footprint) {
                    return node.node_id;
                }
            }
        }
    }

    // Default to highest throughput node with adequate memory
    uint32_t best_node = 0;
    float best_score = -1.0f;

    for (const auto& node : nodes_) {
        if (!node.is_available || node.free_memory_bytes < memory_footprint) continue;
        float score = (flops_required > 1000000) ? node.peak_tflops : node.memory_bandwidth_gbps;
        if (score > best_score) {
            best_score = score;
            best_node = node.node_id;
        }
    }

    return best_node;
}

std::string TopologyManager::generate_topology_report() const {
    std::ostringstream ss;
    ss << "╔══════════════════════════════════════════════════════════════════════╗\n";
    ss << "║              HAVEN TOPOLOGY FABRIC & ACCELERATOR MAP                ║\n";
    ss << "╠══════════════════════════════════════════════════════════════════════╣\n";

    for (const auto& node : nodes_) {
        ss << "║ Node " << node.node_id << ": " << node.device_name << "\n";
        ss << "║   Backend Type : ";
        switch (node.backend) {
            case DeviceBackend::CPU_AVX2:      ss << "CPU_AVX2 (Host Direct)\n"; break;
            case DeviceBackend::GPU_NUMA:      ss << "GPU_NUMA (Discrete PCIe/NVLink)\n"; break;
            case DeviceBackend::TPU_PJRT:      ss << "TPU_PJRT (Optical ICI Fabric)\n"; break;
            case DeviceBackend::FABRIC_REMOTE: ss << "FABRIC_REMOTE (Cluster Mesh)\n"; break;
        }
        ss << "║   Memory Total : " << (node.total_memory_bytes / (1024 * 1024 * 1024)) << " GB (" 
           << (node.free_memory_bytes / (1024 * 1024 * 1024)) << " GB Free)\n";
        ss << "║   Peak Compute : " << node.peak_tflops << " TFLOPS | Bandwidth: " << node.memory_bandwidth_gbps << " GB/s\n";
        ss << "╟──────────────────────────────────────────────────────────────────────╢\n";
    }
    ss << "╚══════════════════════════════════════════════════════════════════════╝\n";
    return ss.str();
}

} // namespace haven
