#pragma once

#include "haven_types.h"
#include "haven_topology.h"
#include "haven_security.h"
#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>
#include <memory>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <atomic>
#include <unordered_map>
#include <functional>

namespace haven {

// Categorical cluster node roles
enum class ClusterRole : uint32_t {
    STANDALONE  = 0,
    COORDINATOR = 1,
    WORKER      = 2
};

// Protocol magic and flags
constexpr uint32_t HAVEN_CLUSTER_MAGIC = 0x4841564E; // 'HAVN'
constexpr uint16_t HAVEN_CLUSTER_VERSION = 1;

enum ClusterFrameFlags : uint16_t {
    FRAME_FLAG_NONE         = 0x0000,
    FRAME_FLAG_ACTIVATION   = 0x0001,
    FRAME_FLAG_KV_SYNC      = 0x0002,
    FRAME_FLAG_EOS          = 0x0004,
    FRAME_FLAG_LOGITS       = 0x0008,
    FRAME_FLAG_HEARTBEAT    = 0x0010,
    FRAME_FLAG_HANDSHAKE    = 0x0020,
    FRAME_FLAG_AUTH_REJECT  = 0x0040,
    FRAME_FLAG_ENCRYPTED    = 0x0080
};

// Fixed 36-byte wire header for zero-copy binary streaming
#pragma pack(push, 1)
struct HavenActivationFrameHeader {
    uint32_t magic;          // 'HAVN' (0x4841564E)
    uint16_t protocol_ver;   // 1
    uint16_t flags;          // ClusterFrameFlags bitmask
    uint64_t sequence_id;    // Active request / conversation session
    uint32_t step_pos;       // Autoregressive token position index
    uint16_t from_node_id;   // Sender node identifier
    uint16_t to_node_id;     // Destination node identifier
    uint16_t layer_boundary; // Model split boundary layer index
    uint16_t reserved;       // Padding / alignment
    uint32_t element_count;  // Count of float elements (hidden dimension)
    uint32_t payload_bytes;  // element_count * sizeof(float)
};
#pragma pack(pop)

// In-memory activation packet payload
struct HavenActivationFrame {
    HavenActivationFrameHeader header{};
    std::vector<float> data;

    bool is_valid() const {
        return header.magic == HAVEN_CLUSTER_MAGIC &&
               header.protocol_ver == HAVEN_CLUSTER_VERSION &&
               header.payload_bytes == header.element_count * sizeof(float) &&
               data.size() == header.element_count;
    }
};

// Node metadata across the cluster fabric
struct ClusterNodeInfo {
    uint32_t node_id = 0;
    std::string address = "127.0.0.1";
    uint16_t port = 11438;
    ClusterRole role = ClusterRole::WORKER;
    DeviceBackend backend = DeviceBackend::CPU_AVX2;
    InterconnectBus bus = InterconnectBus::SYSTEM_RAM_BUS;
    std::string device_name;
    uint32_t start_layer = 0;
    uint32_t end_layer = 0; // [start_layer, end_layer)
    uint64_t total_memory_bytes = 0;
    uint64_t free_memory_bytes = 0;
    float peak_tflops = 0.0f;
    float interconnect_latency_ms = 0.0f;
    uint32_t torus_x = 0;
    uint32_t torus_y = 0;
    uint32_t torus_z = 0;
    bool is_alive = true;
};

// Lock-Free / Condition-Synchronized High-Throughput Activation Ring Buffer
class ActivationRingBuffer {
public:
    explicit ActivationRingBuffer(size_t capacity = 128);
    ~ActivationRingBuffer();

    bool push(HavenActivationFrame frame);
    bool pop(HavenActivationFrame& out_frame, int timeout_ms = 100);
    size_t size() const;
    bool empty() const;
    void close();
    void reset();

private:
    std::queue<HavenActivationFrame> queue_;
    size_t capacity_;
    mutable std::mutex mutex_;
    std::condition_variable not_empty_cv_;
    std::condition_variable not_full_cv_;
    bool closed_ = false;
};

// Raw Socket Zero-Copy Transport Ring (Cross-Platform Win32 Winsock & POSIX)
class ClusterTransportRing {
public:
    ClusterTransportRing();
    ~ClusterTransportRing();

    // Starts listening socket for incoming activation packets
    bool start_server(uint16_t port);

    // Connects to a remote peer node
    bool connect_to_peer(uint32_t node_id, const std::string& address, uint16_t port);

    // Closes all peer sockets and stops listener
    void stop();

    // Sends an activation frame to a specific connected peer node
    bool send_activation(uint32_t target_node_id, const HavenActivationFrame& frame);

    // Broadcasts an activation frame or handshake to all connected peers
    bool broadcast_activation(const HavenActivationFrame& frame);

    // Receives the next incoming activation packet from the inbound ring buffer
    bool receive_activation(HavenActivationFrame& out_frame, int timeout_ms = 100);

    // Diagnostic peer connection state
    bool is_peer_connected(uint32_t node_id) const;
    std::vector<uint32_t> get_connected_peer_ids() const;
    uint16_t get_listen_port() const { return listen_port_; }
    void set_cluster_token(const std::string& token) { cluster_token_ = token; }
    const std::string& get_cluster_token() const { return cluster_token_; }

private:
    void server_listen_loop();
    void client_recv_loop(uint64_t socket_handle, uint32_t peer_node_id);

    uint16_t listen_port_ = 0;
    uint64_t server_socket_ = ~0ULL;
    std::atomic<bool> is_running_{false};
    std::string cluster_token_;

    std::thread server_thread_;
    mutable std::mutex threads_mutex_;
    std::vector<std::thread> client_threads_;

    mutable std::mutex peers_mutex_;
    std::unordered_map<uint32_t, uint64_t> peer_sockets_; // node_id -> socket descriptor

    std::shared_ptr<ActivationRingBuffer> inbound_ring_;
};

// Pipeline Parallel Cluster Slicing & DAG Coordinator
class PipelineClusterScheduler {
public:
    PipelineClusterScheduler();
    ~PipelineClusterScheduler();

    // Partitions total model layers across available cluster nodes
    void partition_layers(uint32_t total_layers, std::vector<ClusterNodeInfo>& nodes);

    // Directly sets pre-partitioned cluster topology
    void set_nodes(const std::vector<ClusterNodeInfo>& nodes, uint32_t total_layers);
    const std::vector<ClusterNodeInfo>& get_nodes() const { return nodes_; }

    // Returns the node ID responsible for executing a specific layer index
    uint32_t get_node_for_layer(uint32_t layer_idx) const;

    // Checks if a given layer belongs to the specified local node
    bool is_local_layer(uint32_t layer_idx, uint32_t node_id) const;

    // Returns layer span for a given node
    std::pair<uint32_t, uint32_t> get_node_layer_range(uint32_t node_id) const;

    // Diagnostic summary of cluster layer layout
    std::string generate_partition_report() const;

private:
    uint32_t total_layers_ = 0;
    std::vector<ClusterNodeInfo> nodes_;
};

// Master Sovereign Multi-Node Cluster Controller
class HavenClusterNode {
public:
    HavenClusterNode();
    ~HavenClusterNode();

    // Initializes node as Cluster Coordinator
    bool init_coordinator(uint16_t listen_port, uint32_t total_model_layers);

    // Initializes node as Worker Node connected to Coordinator
    bool init_worker(uint32_t local_node_id, uint16_t listen_port, 
                     const std::string& coordinator_addr, uint16_t coordinator_port);

    // Security & Authentication
    void set_cluster_token(const std::string& token);
    const std::string& get_cluster_token() const { return cluster_token_; }
    std::string generate_invite_uri(const std::string& host_override = "") const;
    bool init_worker_with_token(uint32_t local_node_id, uint16_t listen_port, 
                                const std::string& invite_or_host, uint16_t coordinator_port = 0,
                                const std::string& token = "");

    // Shuts down cluster networking
    void shutdown();

    // Registers a discovered worker node into the coordinator registry
    bool register_worker(const ClusterNodeInfo& worker);

    // Slices model and synchronizes partition table across all connected nodes
    bool sync_cluster_partition();

    // Connects directly to a peer node in the cluster mesh
    bool connect_peer_node(uint32_t node_id, const std::string& address, uint16_t port);

    // Sends forward layer activations directly to a specific target node
    bool send_activations_to_node(uint32_t target_node_id, uint64_t sequence_id, uint32_t step_pos,
                                  uint32_t layer_boundary, const std::vector<float>& activations);

    // Sends forward layer activations to the next downstream pipeline stage
    bool forward_activations_to_next(uint64_t sequence_id, uint32_t step_pos, 
                                     uint32_t current_layer, const std::vector<float>& activations);

    // Awaits intermediate layer activations from upstream node
    bool await_upstream_activations(HavenActivationFrame& out_frame, int timeout_ms = 500);

    // Status queries
    ClusterRole get_role() const { return role_; }
    uint32_t get_local_node_id() const { return local_node_id_; }
    uint16_t get_listen_port() const { return listen_port_; }
    bool is_active() const { return is_active_; }
    const std::vector<ClusterNodeInfo>& get_nodes() const { return nodes_; }
    const PipelineClusterScheduler& get_scheduler() const { return scheduler_; }
    void set_scheduler(const PipelineClusterScheduler& scheduler);
    bool is_peer_connected(uint32_t node_id) const;

    // Generates cluster health and pipeline topology report
    std::string generate_cluster_report() const;

private:
    ClusterRole role_ = ClusterRole::STANDALONE;
    uint32_t local_node_id_ = 0;
    uint16_t listen_port_ = 11438;
    std::atomic<bool> is_active_{false};

    std::vector<ClusterNodeInfo> nodes_;
    PipelineClusterScheduler scheduler_;
    std::unique_ptr<ClusterTransportRing> transport_;
    std::string cluster_token_;
};

} // namespace haven
