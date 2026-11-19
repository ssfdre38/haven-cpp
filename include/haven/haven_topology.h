#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>
#include <memory>
#include <chrono>

namespace haven {

// Supported execution backend targets across the hardware topology
enum class DeviceBackend : uint32_t {
    CPU_AVX2        = 0,
    GPU_NUMA        = 1,
    TPU_PJRT        = 2,
    FABRIC_REMOTE   = 3
};

// Physical hardware bus and interconnect architecture
enum class InterconnectBus : uint32_t {
    SYSTEM_RAM_BUS  = 0,
    PCIE_GEN4_GEN5   = 1,
    NVLINK_MESH     = 2,
    OPTICAL_ICI     = 3, // TPU 3D Torus Inter-Chip Interconnect
    RDMA_ETHERNET   = 4
};

// Compute node representation in the topology fabric
struct ComputeNode {
    uint32_t node_id = 0;
    DeviceBackend backend = DeviceBackend::CPU_AVX2;
    std::string device_name;
    size_t total_memory_bytes = 0;
    size_t free_memory_bytes = 0;
    uint32_t numa_domain = 0;
    float peak_tflops = 0.0f;
    float memory_bandwidth_gbps = 0.0f;
    InterconnectBus primary_bus = InterconnectBus::SYSTEM_RAM_BUS;
    bool is_available = true;
};

// Hardware Topology Fabric & Device Dispatcher
class TopologyManager {
public:
    TopologyManager();
    ~TopologyManager();

    // Probes host CPU, memory topology, and available accelerator interfaces
    void detect_topology();

    // Returns discovered compute nodes
    const std::vector<ComputeNode>& get_nodes() const { return nodes_; }

    // Selects the optimal node for a specific operation (e.g. GEMV vs embedding vs attention)
    uint32_t select_optimal_node(size_t flops_required, size_t memory_footprint, bool requires_accel = false) const;

    // Checks if native TPU PJRT runtime or driver interface is reachable
    bool has_tpu_accelerator() const { return has_tpu_; }
    size_t get_tpu_chip_count() const { return tpu_chip_count_; }

    // Checks if discrete GPU topology is mapped
    bool has_gpu_accelerator() const { return has_gpu_; }

    // Returns detected NUMA domains on host
    uint32_t get_host_numa_nodes() const { return host_numa_nodes_; }

    // Human-readable topology diagnostic report
    std::string generate_topology_report() const;

private:
    std::vector<ComputeNode> nodes_;
    bool has_tpu_ = false;
    bool has_gpu_ = false;
    size_t tpu_chip_count_ = 0;
    uint32_t host_numa_nodes_ = 1;
};

} // namespace haven
