#include "haven/haven_cluster.h"
#include <iostream>
#include <sstream>
#include <cstring>
#include <algorithm>
#include <chrono>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <winsock2.h>
#include <ws2tcpip.h>
typedef SOCKET socket_t;
#define HAVEN_INVALID_SOCKET INVALID_SOCKET
#define HAVEN_SOCKET_ERROR SOCKET_ERROR
#else
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fcntl.h>
#include <poll.h>
typedef int socket_t;
#define HAVEN_INVALID_SOCKET (-1)
#define HAVEN_SOCKET_ERROR (-1)
#define closesocket close
#endif

namespace haven {

// Initialize platform socket subsystem
static void init_network_subsystem() {
#ifdef _WIN32
    static bool initialized = false;
    if (!initialized) {
        WSADATA wsa;
        if (WSAStartup(MAKEWORD(2, 2), &wsa) == 0) {
            initialized = true;
        }
    }
#endif
}

// Set TCP socket performance parameters
static void configure_cluster_socket(socket_t sock) {
    int nodelay = 1;
    setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, (const char*)&nodelay, sizeof(nodelay));

    int buf_size = 4 * 1024 * 1024; // 4MB socket window for activation throughput
    setsockopt(sock, SOL_SOCKET, SO_SNDBUF, (const char*)&buf_size, sizeof(buf_size));
    setsockopt(sock, SOL_SOCKET, SO_RCVBUF, (const char*)&buf_size, sizeof(buf_size));

    int reuse = 1;
    setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, (const char*)&reuse, sizeof(reuse));
}

// Helper: send exact number of bytes
static bool send_all(socket_t sock, const void* data, size_t total_bytes) {
    const char* ptr = static_cast<const char*>(data);
    size_t remaining = total_bytes;
    while (remaining > 0) {
        int sent = ::send(sock, ptr, static_cast<int>(remaining), 0);
        if (sent <= 0) {
            return false;
        }
        remaining -= sent;
        ptr += sent;
    }
    return true;
}

// Helper: receive exact number of bytes with polling timeout
static bool recv_all(socket_t sock, void* data, size_t total_bytes, const std::atomic<bool>& is_running) {
    char* ptr = static_cast<char*>(data);
    size_t remaining = total_bytes;
    while (remaining > 0 && is_running.load()) {
        fd_set read_fds;
        FD_ZERO(&read_fds);
        FD_SET(sock, &read_fds);

        timeval tv;
        tv.tv_sec = 0;
        tv.tv_usec = 100000; // 100ms timeout
        int sel = ::select(static_cast<int>(sock + 1), &read_fds, nullptr, nullptr, &tv);
        if (sel <= 0) {
            continue; // check is_running
        }

        int recvd = ::recv(sock, ptr, static_cast<int>(remaining), 0);
        if (recvd <= 0) {
            return false;
        }
        remaining -= recvd;
        ptr += recvd;
    }
    return remaining == 0;
}

// ============================================================================
// ActivationRingBuffer Implementation
// ============================================================================

ActivationRingBuffer::ActivationRingBuffer(size_t capacity)
    : capacity_(capacity > 0 ? capacity : 128) {}

ActivationRingBuffer::~ActivationRingBuffer() {
    close();
}

bool ActivationRingBuffer::push(HavenActivationFrame frame) {
    std::unique_lock<std::mutex> lock(mutex_);
    not_full_cv_.wait(lock, [this]() {
        return closed_ || queue_.size() < capacity_;
    });

    if (closed_) {
        return false;
    }

    queue_.push(std::move(frame));
    not_empty_cv_.notify_one();
    return true;
}

bool ActivationRingBuffer::pop(HavenActivationFrame& out_frame, int timeout_ms) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (timeout_ms <= 0) {
        not_empty_cv_.wait(lock, [this]() {
            return closed_ || !queue_.empty();
        });
    } else {
        bool arrived = not_empty_cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms), [this]() {
            return closed_ || !queue_.empty();
        });
        if (!arrived && queue_.empty()) {
            return false;
        }
    }

    if (queue_.empty()) {
        return false;
    }

    out_frame = std::move(queue_.front());
    queue_.pop();
    not_full_cv_.notify_one();
    return true;
}

size_t ActivationRingBuffer::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.size();
}

bool ActivationRingBuffer::empty() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.empty();
}

void ActivationRingBuffer::close() {
    std::lock_guard<std::mutex> lock(mutex_);
    closed_ = true;
    not_empty_cv_.notify_all();
    not_full_cv_.notify_all();
}

void ActivationRingBuffer::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    std::queue<HavenActivationFrame> empty;
    std::swap(queue_, empty);
    closed_ = false;
}

// ============================================================================
// ClusterTransportRing Implementation
// ============================================================================

ClusterTransportRing::ClusterTransportRing()
    : inbound_ring_(std::make_shared<ActivationRingBuffer>(256))
{
    init_network_subsystem();
}

ClusterTransportRing::~ClusterTransportRing() {
    stop();
}

bool ClusterTransportRing::start_server(uint16_t port) {
    if (is_running_.exchange(true)) {
        return true; // Already active
    }

    listen_port_ = port;

    socket_t listen_fd = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (listen_fd == HAVEN_INVALID_SOCKET) {
        is_running_ = false;
        return false;
    }

    configure_cluster_socket(listen_fd);

    sockaddr_in server_addr{};
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = htonl(INADDR_ANY);
    server_addr.sin_port = htons(port);

    if (::bind(listen_fd, (struct sockaddr*)&server_addr, sizeof(server_addr)) == HAVEN_SOCKET_ERROR) {
        closesocket(listen_fd);
        is_running_ = false;
        return false;
    }

    if (::listen(listen_fd, 16) == HAVEN_SOCKET_ERROR) {
        closesocket(listen_fd);
        is_running_ = false;
        return false;
    }

    server_socket_ = static_cast<uint64_t>(listen_fd);
    server_thread_ = std::thread(&ClusterTransportRing::server_listen_loop, this);

    std::cout << "🌐 [ClusterTransportRing] Sovereign cluster transport online on port " << port << " (Zero-Copy Active)\n";
    return true;
}

bool ClusterTransportRing::connect_to_peer(uint32_t node_id, const std::string& address, uint16_t port) {
    socket_t client_fd = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (client_fd == HAVEN_INVALID_SOCKET) {
        return false;
    }

    configure_cluster_socket(client_fd);

    sockaddr_in peer_addr{};
    peer_addr.sin_family = AF_INET;
    peer_addr.sin_port = htons(port);
    if (inet_pton(AF_INET, address.c_str(), &peer_addr.sin_addr) <= 0) {
        closesocket(client_fd);
        return false;
    }

    if (::connect(client_fd, (struct sockaddr*)&peer_addr, sizeof(peer_addr)) == HAVEN_SOCKET_ERROR) {
        closesocket(client_fd);
        return false;
    }

    uint64_t sock_val = static_cast<uint64_t>(client_fd);
    {
        std::lock_guard<std::mutex> lock(peers_mutex_);
        peer_sockets_[node_id] = sock_val;
    }

    // Launch background receiver thread for this peer connection
    {
        std::lock_guard<std::mutex> lock(threads_mutex_);
        client_threads_.emplace_back(&ClusterTransportRing::client_recv_loop, this, sock_val, node_id);
    }

    std::cout << "🔗 [ClusterTransportRing] Connected to peer Node " << node_id 
              << " (" << address << ":" << port << ")\n";
    return true;
}

void ClusterTransportRing::stop() {
    if (!is_running_.exchange(false)) {
        return;
    }

    inbound_ring_->close();

    // 1. Shutdown all peer sockets to immediately unblock any blocking recv calls
    {
        std::lock_guard<std::mutex> lock(peers_mutex_);
        for (auto& [id, s_val] : peer_sockets_) {
            socket_t s = static_cast<socket_t>(s_val);
#if defined(_WIN32)
            ::shutdown(s, SD_BOTH);
#else
            ::shutdown(s, SHUT_RDWR);
#endif
        }
    }

    // 2. Close server listening socket
    if (server_socket_ != ~0ULL) {
        socket_t s = static_cast<socket_t>(server_socket_);
        server_socket_ = ~0ULL;
        closesocket(s);
    }

    if (server_thread_.joinable()) {
        server_thread_.join();
    }

    // 3. Join all client receiver threads cleanly
    std::vector<std::thread> to_join;
    {
        std::lock_guard<std::mutex> lock(threads_mutex_);
        to_join = std::move(client_threads_);
    }

    for (auto& t : to_join) {
        if (t.joinable()) {
            t.join();
        }
    }

    // 4. Ensure any remaining peer sockets are closed
    {
        std::lock_guard<std::mutex> lock(peers_mutex_);
        for (auto& [id, s_val] : peer_sockets_) {
            closesocket(static_cast<socket_t>(s_val));
        }
        peer_sockets_.clear();
    }
}

bool ClusterTransportRing::send_activation(uint32_t target_node_id, const HavenActivationFrame& frame) {
    socket_t sock = HAVEN_INVALID_SOCKET;
    {
        std::lock_guard<std::mutex> lock(peers_mutex_);
        auto it = peer_sockets_.find(target_node_id);
        if (it != peer_sockets_.end()) {
            sock = static_cast<socket_t>(it->second);
        }
    }

    if (sock == HAVEN_INVALID_SOCKET) {
        return false;
    }

    // Encrypt payload if token is active and frame is an activation tensor
    if (!cluster_token_.empty() && (frame.header.flags & FRAME_FLAG_ACTIVATION) && !frame.data.empty()) {
        HavenActivationFrame enc_frame = frame;
        enc_frame.header.flags |= FRAME_FLAG_ENCRYPTED;
        HavenSecurity::mask_activation_payload(enc_frame.data.data(), enc_frame.data.size(), 
                                              cluster_token_, enc_frame.header.sequence_id, enc_frame.header.step_pos);
        if (!send_all(sock, &enc_frame.header, sizeof(enc_frame.header))) {
            return false;
        }
        if (enc_frame.header.payload_bytes > 0 && !enc_frame.data.empty()) {
            if (!send_all(sock, enc_frame.data.data(), enc_frame.header.payload_bytes)) {
                return false;
            }
        }
        return true;
    }

    // 1. Send fixed 36-byte header
    if (!send_all(sock, &frame.header, sizeof(frame.header))) {
        return false;
    }

    // 2. Send contiguous raw float data payload with zero copy
    if (frame.header.payload_bytes > 0 && !frame.data.empty()) {
        if (!send_all(sock, frame.data.data(), frame.header.payload_bytes)) {
            return false;
        }
    }

    return true;
}

bool ClusterTransportRing::broadcast_activation(const HavenActivationFrame& frame) {
    std::vector<uint32_t> targets = get_connected_peer_ids();
    bool all_ok = true;
    for (uint32_t target_id : targets) {
        if (!send_activation(target_id, frame)) {
            all_ok = false;
        }
    }
    return all_ok;
}

bool ClusterTransportRing::receive_activation(HavenActivationFrame& out_frame, int timeout_ms) {
    return inbound_ring_->pop(out_frame, timeout_ms);
}

bool ClusterTransportRing::is_peer_connected(uint32_t node_id) const {
    std::lock_guard<std::mutex> lock(peers_mutex_);
    return peer_sockets_.find(node_id) != peer_sockets_.end();
}

std::vector<uint32_t> ClusterTransportRing::get_connected_peer_ids() const {
    std::lock_guard<std::mutex> lock(peers_mutex_);
    std::vector<uint32_t> ids;
    ids.reserve(peer_sockets_.size());
    for (const auto& [id, _] : peer_sockets_) {
        ids.push_back(id);
    }
    return ids;
}

void ClusterTransportRing::server_listen_loop() {
    socket_t server_fd = static_cast<socket_t>(server_socket_);
    uint32_t incoming_node_counter = 100;

    while (is_running_) {
        fd_set read_fds;
        FD_ZERO(&read_fds);
        FD_SET(server_fd, &read_fds);

        timeval tv;
        tv.tv_sec = 0;
        tv.tv_usec = 100000; // 100ms poll timeout

        int sel = ::select(static_cast<int>(server_fd + 1), &read_fds, nullptr, nullptr, &tv);
        if (sel <= 0) {
            continue; // Timeout or interrupted; re-check is_running_
        }

        sockaddr_in client_addr{};
        socklen_t client_len = sizeof(client_addr);
        socket_t client_fd = ::accept(server_fd, (struct sockaddr*)&client_addr, &client_len);

        if (client_fd == HAVEN_INVALID_SOCKET) {
            if (!is_running_) break;
            continue;
        }

        configure_cluster_socket(client_fd);
        uint32_t assigned_id = incoming_node_counter++;
        uint64_t sock_val = static_cast<uint64_t>(client_fd);

        {
            std::lock_guard<std::mutex> lock(peers_mutex_);
            peer_sockets_[assigned_id] = sock_val;
        }

        {
            std::lock_guard<std::mutex> lock(threads_mutex_);
            client_threads_.emplace_back(&ClusterTransportRing::client_recv_loop, this, sock_val, assigned_id);
        }
    }
}

void ClusterTransportRing::client_recv_loop(uint64_t socket_handle, uint32_t peer_node_id) {
    socket_t sock = static_cast<socket_t>(socket_handle);

    while (is_running_) {
        HavenActivationFrame frame;
        // 1. Read fixed 36-byte header
        if (!recv_all(sock, &frame.header, sizeof(frame.header), is_running_)) {
            break; // Connection terminated
        }

        // Validate protocol header
        if (frame.header.magic != HAVEN_CLUSTER_MAGIC || frame.header.protocol_ver != HAVEN_CLUSTER_VERSION) {
            std::cerr << "⚠️ [ClusterTransportRing] Corrupted packet header received from peer Node " << peer_node_id << "\n";
            break;
        }

        // 2. Read contiguous raw float data payload
        if (frame.header.payload_bytes > 0 && frame.header.element_count > 0) {
            constexpr uint32_t MAX_FRAME_BYTES = 256 * 1024 * 1024;
            if (frame.header.payload_bytes > MAX_FRAME_BYTES) {
                std::cerr << "⚠️ [ClusterTransportRing] Frame payload exceeds 256MB limit on socket " << socket_handle << "\n";
                break;
            }
            frame.data.resize(frame.header.element_count);
            if (!recv_all(sock, frame.data.data(), frame.header.payload_bytes, is_running_)) {
                break;
            }
        }

        // Handle Handshake: authenticate peer if cluster token is configured
        if (frame.header.flags & FRAME_FLAG_HANDSHAKE) {
            uint32_t real_node_id = frame.header.from_node_id;

            if (!cluster_token_.empty()) {
                std::string received_token;
                if (!frame.data.empty() && frame.header.payload_bytes > 0) {
                    const char* bytes = reinterpret_cast<const char*>(frame.data.data());
                    size_t len = std::min<size_t>(frame.header.payload_bytes, 64);
                    received_token = std::string(bytes, len);
                }

                if (!HavenSecurity::constant_time_verify(received_token, cluster_token_)) {
                    std::cerr << "⛔ [ClusterSecurity] Unauthorized peer rejected on socket " 
                              << socket_handle << " (Invalid Cluster Token). Terminating connection.\n";
                    HavenActivationFrame reject_frame;
                    reject_frame.header.magic = HAVEN_CLUSTER_MAGIC;
                    reject_frame.header.protocol_ver = HAVEN_CLUSTER_VERSION;
                    reject_frame.header.flags = FRAME_FLAG_AUTH_REJECT;
                    send_all(sock, &reject_frame.header, sizeof(reject_frame.header));
                    closesocket(sock);
                    sock = -1;
                    break;
                }
                std::cout << "🔒 [ClusterSecurity] Peer Node " << real_node_id 
                          << " successfully authenticated with 256-bit cluster token.\n";
            }

            if (real_node_id != peer_node_id) {
                std::lock_guard<std::mutex> lock(peers_mutex_);
                peer_sockets_.erase(peer_node_id);
                peer_sockets_[real_node_id] = socket_handle;
                peer_node_id = real_node_id;
            }
            continue;
        }

        // Handle Wire Decryption if payload is encrypted
        if (frame.header.flags & FRAME_FLAG_ENCRYPTED) {
            if (!cluster_token_.empty() && !frame.data.empty()) {
                HavenSecurity::mask_activation_payload(frame.data.data(), frame.data.size(), 
                                                      cluster_token_, frame.header.sequence_id, frame.header.step_pos);
            }
        }

        // Push directly into inbound ring buffer
        inbound_ring_->push(std::move(frame));
    }

    // Peer disconnected: clean up under mutex
    if (is_running_) {
        std::lock_guard<std::mutex> lock(peers_mutex_);
        auto it = peer_sockets_.find(peer_node_id);
        if (it != peer_sockets_.end() && it->second == socket_handle) {
            peer_sockets_.erase(it);
        }
        if (sock != INVALID_SOCKET) {
            closesocket(sock);
        }
    }
}

// ============================================================================
// PipelineClusterScheduler Implementation
// ============================================================================

PipelineClusterScheduler::PipelineClusterScheduler() = default;
PipelineClusterScheduler::~PipelineClusterScheduler() = default;

void PipelineClusterScheduler::partition_layers(uint32_t total_layers, std::vector<ClusterNodeInfo>& nodes) {
    total_layers_ = total_layers;
    nodes_ = nodes;

    if (nodes_.empty() || total_layers_ == 0) {
        return;
    }

    size_t num_nodes = nodes_.size();
    if (num_nodes == 1) {
        nodes_[0].start_layer = 0;
        nodes_[0].end_layer = total_layers_;
        nodes = nodes_;
        return;
    }

    // 1. Detect if dedicated high-throughput accelerator workers exist
    bool has_accelerator_workers = false;
    float max_worker_tflops = 0.0f;
    for (size_t i = 0; i < num_nodes; ++i) {
        if (nodes_[i].role == ClusterRole::WORKER) {
            if (nodes_[i].backend == DeviceBackend::TPU_PJRT || 
                nodes_[i].backend == DeviceBackend::GPU_NUMA || 
                nodes_[i].peak_tflops >= 100.0f) {
                has_accelerator_workers = true;
            }
            if (nodes_[i].peak_tflops > max_worker_tflops) {
                max_worker_tflops = nodes_[i].peak_tflops;
            }
        }
    }

    // 2. Compute dynamic compute weights
    // If high-throughput accelerator workers are attached, a CPU host coordinator
    // enters Thin Gateway Mode (0 compute layers) to prevent pipeline stall bubbles.
    std::vector<double> weights(num_nodes, 0.0);
    double total_weight = 0.0;

    for (size_t i = 0; i < num_nodes; ++i) {
        // Explicit pure coordinator or Thin Gateway when accelerator workers exist
        if (nodes_[i].role == ClusterRole::COORDINATOR) {
            if (has_accelerator_workers && (nodes_[i].backend == DeviceBackend::CPU_AVX2 || nodes_[i].peak_tflops < 0.2f * max_worker_tflops)) {
                weights[i] = 0.0;
                continue;
            }
            if (nodes_[i].total_memory_bytes == 0 && nodes_[i].peak_tflops == 0.0f) {
                weights[i] = 0.0;
                continue;
            }
        }

        // Hardware backend multiplier
        double backend_mult = 1.0;
        switch (nodes_[i].backend) {
            case DeviceBackend::TPU_PJRT:      backend_mult = 1.00; break;
            case DeviceBackend::GPU_NUMA:      backend_mult = 0.95; break;
            case DeviceBackend::FABRIC_REMOTE: backend_mult = 0.85; break;
            case DeviceBackend::CPU_AVX2:      backend_mult = 0.25; break;
            default:                           backend_mult = 0.50; break;
        }

        double flops = (nodes_[i].peak_tflops > 0.0f) ? (double)nodes_[i].peak_tflops : 10.0;
        double mem_gb = (nodes_[i].total_memory_bytes > 0) ? 
                        (double)nodes_[i].total_memory_bytes / (1024.0 * 1024.0 * 1024.0) : 16.0;

        // Compute throughput dominates layer allocation (85% FLOPS capacity, 15% Memory capacity)
        double compute_capacity = (flops * 0.85 + (mem_gb * 0.15)) * backend_mult;

        // Latency penalty for high-latency WAN links: penalize intermediate hops across public internet
        if (nodes_[i].interconnect_latency_ms > 2.0f) {
            compute_capacity /= (1.0 + 0.05 * (double)nodes_[i].interconnect_latency_ms);
        }

        weights[i] = std::max(0.01, compute_capacity);
        total_weight += weights[i];
    }

    if (total_weight <= 0.0) {
        total_weight = 1.0;
        weights[0] = 1.0;
    }

    // 3. Assign layer spans strictly proportional to compute capacity
    uint32_t current_layer = 0;
    size_t last_active_idx = 0;
    for (size_t i = 0; i < num_nodes; ++i) {
        if (weights[i] > 0.0) last_active_idx = i;
    }

    for (size_t i = 0; i < num_nodes; ++i) {
        if (weights[i] <= 0.0) {
            nodes_[i].start_layer = current_layer;
            nodes_[i].end_layer = current_layer;
            continue;
        }

        uint32_t node_layers = 0;
        if (i == last_active_idx || current_layer >= total_layers_) {
            node_layers = (total_layers_ > current_layer) ? (total_layers_ - current_layer) : 0;
        } else {
            double fraction = weights[i] / total_weight;
            node_layers = static_cast<uint32_t>(std::round(fraction * total_layers_));
            if (node_layers == 0 && current_layer < total_layers_) {
                node_layers = 1;
            }
            if (current_layer + node_layers > total_layers_) {
                node_layers = total_layers_ - current_layer;
            }
        }

        nodes_[i].start_layer = current_layer;
        nodes_[i].end_layer = current_layer + node_layers;
        current_layer = nodes_[i].end_layer;
    }

    // Ensure last active node closes out the remainder
    if (current_layer < total_layers_ && weights[last_active_idx] > 0.0) {
        nodes_[last_active_idx].end_layer = total_layers_;
    }

    nodes = nodes_; // update caller
}

void PipelineClusterScheduler::set_nodes(const std::vector<ClusterNodeInfo>& nodes, uint32_t total_layers) {
    nodes_ = nodes;
    total_layers_ = total_layers;
}

uint32_t PipelineClusterScheduler::get_node_for_layer(uint32_t layer_idx) const {
    for (const auto& node : nodes_) {
        if (layer_idx >= node.start_layer && layer_idx < node.end_layer) {
            return node.node_id;
        }
    }
    return 0; // Default to coordinator / node 0
}

bool PipelineClusterScheduler::is_local_layer(uint32_t layer_idx, uint32_t node_id) const {
    for (const auto& node : nodes_) {
        if (node.node_id == node_id) {
            return (layer_idx >= node.start_layer && layer_idx < node.end_layer);
        }
    }
    return false;
}

std::pair<uint32_t, uint32_t> PipelineClusterScheduler::get_node_layer_range(uint32_t node_id) const {
    for (const auto& node : nodes_) {
        if (node.node_id == node_id) {
            return {node.start_layer, node.end_layer};
        }
    }
    return {0, 0};
}

std::string PipelineClusterScheduler::generate_partition_report() const {
    std::ostringstream ss;
    ss << "╔══════════════════════════════════════════════════════════════════════╗\n";
    ss << "║        HAVEN PIPELINE PARALLEL CLUSTER PARTITION TOPOLOGY           ║\n";
    ss << "╠══════════════════════════════════════════════════════════════════════╣\n";
    ss << "║ Total Model Layers: " << total_layers_ << "\n";
    ss << "╟──────────────────────────────────────────────────────────────────────╢\n";
    for (const auto& node : nodes_) {
        std::string backend_str = "CPU_AVX2 (Host Direct)";
        if (node.backend == DeviceBackend::TPU_PJRT) {
            backend_str = "TPU_PJRT (Optical ICI Torus)";
        } else if (node.backend == DeviceBackend::GPU_NUMA) {
            backend_str = "GPU_NUMA (Discrete PCIe/NVLink)";
        } else if (node.backend == DeviceBackend::FABRIC_REMOTE) {
            backend_str = "FABRIC_REMOTE (Zero-Copy Cluster Mesh)";
        }

        uint32_t span = (node.end_layer >= node.start_layer) ? (node.end_layer - node.start_layer) : 0;
        ss << "║ Node " << node.node_id << " (" << (node.role == ClusterRole::COORDINATOR ? "COORDINATOR" : "WORKER") << ") "
           << "@ " << node.address << ":" << node.port << "\n"
           << "║   Hardware : " << (node.device_name.empty() ? backend_str : node.device_name) << "\n"
           << "║   Backend  : " << backend_str;
        if (node.backend == DeviceBackend::TPU_PJRT) {
            ss << " | Torus Mesh: (" << node.torus_x << ", " << node.torus_y << ", " << node.torus_z << ")";
        }
        ss << "\n";
        if (span == 0 && node.role == ClusterRole::COORDINATOR) {
            ss << "║   Layers   : [Thin Gateway / Sensory Coordinator] (0 compute layers)\n";
        } else {
            ss << "║   Layers   : [" << node.start_layer << " .. " << (node.end_layer > 0 ? node.end_layer - 1 : 0) << "] (" << span << " layers)\n";
        }
        ss << "║   Memory   : " << (node.total_memory_bytes / (1024*1024*1024)) << " GB Total\n"
           << "║   Compute  : " << node.peak_tflops << " TFLOPS\n"
           << "╟──────────────────────────────────────────────────────────────────────╢\n";
    }
    ss << "╚══════════════════════════════════════════════════════════════════════╝\n";
    return ss.str();
}

// ============================================================================
// HavenClusterNode Implementation
// ============================================================================

HavenClusterNode::HavenClusterNode()
    : transport_(std::make_unique<ClusterTransportRing>()) {}

HavenClusterNode::~HavenClusterNode() {
    shutdown();
}

bool HavenClusterNode::init_coordinator(uint16_t listen_port, uint32_t total_model_layers) {
    role_ = ClusterRole::COORDINATOR;
    local_node_id_ = 0;
    listen_port_ = listen_port;

    if (!transport_->start_server(listen_port)) {
        return false;
    }

    nodes_.clear();
    ClusterNodeInfo coordinator_info;
    coordinator_info.node_id = 0;
    coordinator_info.address = "127.0.0.1";
    coordinator_info.port = listen_port;
    coordinator_info.role = ClusterRole::COORDINATOR;
    coordinator_info.start_layer = 0;
    coordinator_info.end_layer = total_model_layers;
    coordinator_info.is_alive = true;

    nodes_.push_back(coordinator_info);
    scheduler_.partition_layers(total_model_layers, nodes_);

    is_active_ = true;
    std::cout << "👑 [HavenClusterNode] Cluster Coordinator initialized on port " << listen_port 
              << " (" << total_model_layers << " model layers)\n";
    return true;
}

bool HavenClusterNode::init_worker(uint32_t local_node_id, uint16_t listen_port,
                                   const std::string& coordinator_addr, uint16_t coordinator_port) {
    role_ = ClusterRole::WORKER;
    local_node_id_ = local_node_id;
    listen_port_ = listen_port;

    if (!transport_->start_server(listen_port)) {
        return false;
    }

    if (!transport_->connect_to_peer(0, coordinator_addr, coordinator_port)) {
        std::cerr << "⚠️ [HavenClusterNode] Worker failed to connect to coordinator at " 
                  << coordinator_addr << ":" << coordinator_port << "\n";
        transport_->stop();
        return false;
    }

    // Send handshake frame to coordinator
    HavenActivationFrame handshake;
    handshake.header.magic = HAVEN_CLUSTER_MAGIC;
    handshake.header.protocol_ver = HAVEN_CLUSTER_VERSION;
    handshake.header.flags = FRAME_FLAG_HANDSHAKE;
    handshake.header.from_node_id = static_cast<uint16_t>(local_node_id);
    handshake.header.to_node_id = 0;

    if (!cluster_token_.empty()) {
        size_t token_bytes = cluster_token_.size();
        handshake.header.payload_bytes = static_cast<uint32_t>(token_bytes);
        handshake.header.element_count = static_cast<uint32_t>((token_bytes + sizeof(float) - 1) / sizeof(float));
        handshake.data.resize(handshake.header.element_count, 0.0f);
        std::memcpy(handshake.data.data(), cluster_token_.data(), token_bytes);
    } else {
        handshake.header.element_count = 0;
        handshake.header.payload_bytes = 0;
    }
    transport_->send_activation(0, handshake);

    is_active_ = true;
    std::cout << "⚡ [HavenClusterNode] Worker Node " << local_node_id << " joined coordinator @ "
              << coordinator_addr << ":" << coordinator_port << "\n";
    return true;
}

void HavenClusterNode::set_cluster_token(const std::string& token) {
    cluster_token_ = token;
    if (transport_) {
        transport_->set_cluster_token(token);
    }
}

std::string HavenClusterNode::generate_invite_uri(const std::string& host_override) const {
    std::string host = host_override.empty() ? "127.0.0.1" : host_override;
    std::string token = cluster_token_.empty() ? HavenSecurity::generate_cluster_token() : cluster_token_;
    return HavenSecurity::create_invite_uri(host, listen_port_, token);
}

bool HavenClusterNode::init_worker_with_token(uint32_t local_node_id, uint16_t listen_port, 
                                             const std::string& invite_or_host, uint16_t coordinator_port,
                                             const std::string& token) {
    std::string target_host = invite_or_host;
    uint16_t target_port = coordinator_port;
    std::string target_token = token;
    std::string target_name;

    if (invite_or_host.rfind("haven://", 0) == 0) {
        if (!HavenSecurity::parse_invite_uri(invite_or_host, target_host, target_port, target_token, target_name)) {
            std::cerr << "❌ Failed to parse cluster invite URI: " << invite_or_host << "\n";
            return false;
        }
    }

    if (!target_token.empty()) {
        set_cluster_token(target_token);
    }

    return init_worker(local_node_id, listen_port, target_host, target_port);
}

void HavenClusterNode::shutdown() {
    if (!is_active_.exchange(false)) {
        return;
    }

    transport_->stop();
    nodes_.clear();
}

bool HavenClusterNode::register_worker(const ClusterNodeInfo& worker) {
    for (auto& n : nodes_) {
        if (n.node_id == worker.node_id) {
            n = worker;
            return true;
        }
    }
    nodes_.push_back(worker);
    return true;
}

bool HavenClusterNode::sync_cluster_partition() {
    if (role_ != ClusterRole::COORDINATOR) {
        return false;
    }

    uint32_t total = 0;
    for (const auto& n : nodes_) {
        if (n.end_layer > total) total = n.end_layer;
    }

    scheduler_.partition_layers(total > 0 ? total : 32, nodes_);

    // Broadcast partition sync table to all connected workers
    HavenActivationFrame sync_frame;
    sync_frame.header.magic = HAVEN_CLUSTER_MAGIC;
    sync_frame.header.protocol_ver = HAVEN_CLUSTER_VERSION;
    sync_frame.header.flags = FRAME_FLAG_KV_SYNC;
    sync_frame.header.from_node_id = 0;
    sync_frame.header.to_node_id = 0; // Broadcast
    sync_frame.header.element_count = 0;
    sync_frame.header.payload_bytes = 0;

    return transport_->broadcast_activation(sync_frame);
}

bool HavenClusterNode::connect_peer_node(uint32_t node_id, const std::string& address, uint16_t port) {
    if (!transport_->connect_to_peer(node_id, address, port)) {
        return false;
    }

    // Send handshake frame so peer knows who connected
    HavenActivationFrame handshake;
    handshake.header.magic = HAVEN_CLUSTER_MAGIC;
    handshake.header.protocol_ver = HAVEN_CLUSTER_VERSION;
    handshake.header.flags = FRAME_FLAG_HANDSHAKE;
    handshake.header.from_node_id = static_cast<uint16_t>(local_node_id_);
    handshake.header.to_node_id = static_cast<uint16_t>(node_id);

    if (!cluster_token_.empty()) {
        size_t token_bytes = cluster_token_.size();
        handshake.header.payload_bytes = static_cast<uint32_t>(token_bytes);
        handshake.header.element_count = static_cast<uint32_t>((token_bytes + sizeof(float) - 1) / sizeof(float));
        handshake.data.resize(handshake.header.element_count, 0.0f);
        std::memcpy(handshake.data.data(), cluster_token_.data(), token_bytes);
    } else {
        handshake.header.element_count = 0;
        handshake.header.payload_bytes = 0;
    }
    return transport_->send_activation(node_id, handshake);
}

bool HavenClusterNode::send_activations_to_node(uint32_t target_node_id, uint64_t sequence_id, uint32_t step_pos,
                                               uint32_t layer_boundary, const std::vector<float>& activations) {
    HavenActivationFrame frame;
    frame.header.magic = HAVEN_CLUSTER_MAGIC;
    frame.header.protocol_ver = HAVEN_CLUSTER_VERSION;
    frame.header.flags = FRAME_FLAG_ACTIVATION;
    frame.header.sequence_id = sequence_id;
    frame.header.step_pos = step_pos;
    frame.header.from_node_id = static_cast<uint16_t>(local_node_id_);
    frame.header.to_node_id = static_cast<uint16_t>(target_node_id);
    frame.header.layer_boundary = static_cast<uint16_t>(layer_boundary);
    frame.header.element_count = static_cast<uint32_t>(activations.size());
    frame.header.payload_bytes = static_cast<uint32_t>(activations.size() * sizeof(float));
    frame.data = activations;

    return transport_->send_activation(target_node_id, frame);
}

bool HavenClusterNode::forward_activations_to_next(uint64_t sequence_id, uint32_t step_pos,
                                                  uint32_t current_layer, const std::vector<float>& activations) {
    uint32_t target_layer = (local_node_id_ == 0 && current_layer == 0) ? 0 : (current_layer + 1);
    uint32_t next_node = scheduler_.get_node_for_layer(target_layer);
    if (next_node == local_node_id_) {
        next_node = 0; // Return to coordinator
    }

    return send_activations_to_node(next_node, sequence_id, step_pos, target_layer, activations);
}

void HavenClusterNode::set_scheduler(const PipelineClusterScheduler& scheduler) {
    scheduler_ = scheduler;
}

bool HavenClusterNode::is_peer_connected(uint32_t node_id) const {
    return transport_ && transport_->is_peer_connected(node_id);
}

bool HavenClusterNode::await_upstream_activations(HavenActivationFrame& out_frame, int timeout_ms) {
    auto start = std::chrono::steady_clock::now();
    while (true) {
        int remaining_ms = timeout_ms;
        if (timeout_ms > 0) {
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
            if (elapsed >= timeout_ms) return false;
            remaining_ms = static_cast<int>(timeout_ms - elapsed);
        }
        if (!transport_->receive_activation(out_frame, remaining_ms)) {
            return false;
        }
        if (out_frame.header.flags & FRAME_FLAG_ACTIVATION) {
            return true;
        }
    }
}

std::string HavenClusterNode::generate_cluster_report() const {
    std::ostringstream ss;
    ss << "╔══════════════════════════════════════════════════════════════════════╗\n";
    ss << "║               HAVEN SOVEREIGN MULTI-NODE CLUSTER FABRIC              ║\n";
    ss << "╠══════════════════════════════════════════════════════════════════════╣\n";
    ss << "║ Local Node ID : " << local_node_id_ << "\n";
    ss << "║ Role          : " << (role_ == ClusterRole::COORDINATOR ? "CLUSTER COORDINATOR" : 
                                (role_ == ClusterRole::WORKER ? "PIPELINE WORKER NODE" : "STANDALONE")) << "\n";
    ss << "║ Listen Port   : " << listen_port_ << "\n";
    ss << "║ Status        : " << (is_active_ ? "ONLINE (Transport Active)" : "OFFLINE") << "\n";
    ss << "║ Connected Peers: " << transport_->get_connected_peer_ids().size() << " peer node(s)\n";
    ss << "╚══════════════════════════════════════════════════════════════════════╝\n";
    ss << scheduler_.generate_partition_report();
    return ss.str();
}

} // namespace haven
