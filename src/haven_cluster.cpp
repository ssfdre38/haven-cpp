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
        int chunk = static_cast<int>(std::min<size_t>(remaining, 16ULL * 1024 * 1024));
        int sent = ::send(sock, ptr, chunk, 0);
        if (sent <= 0) {
            return false;
        }
        remaining -= static_cast<size_t>(sent);
        ptr += sent;
    }
    return true;
}

// Helper: receive exact number of bytes with polling timeout and optional max deadline (0 = infinite while running)
static bool recv_all(socket_t sock, void* data, size_t total_bytes, const std::atomic<bool>& is_running, int max_wait_ms = 0) {
    char* ptr = static_cast<char*>(data);
    size_t remaining = total_bytes;
    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(max_wait_ms);
    while (remaining > 0 && is_running.load()) {
        if (max_wait_ms > 0 && std::chrono::steady_clock::now() >= deadline) {
            return false;
        }
#ifdef _WIN32
        fd_set read_fds;
        FD_ZERO(&read_fds);
        FD_SET(sock, &read_fds);

        timeval tv;
        tv.tv_sec = 0;
        tv.tv_usec = 100000; // 100ms timeout
        int sel = ::select(0, &read_fds, nullptr, nullptr, &tv);
        if (sel < 0) return false;
        if (sel == 0) continue; // check is_running / deadline
#else
        struct pollfd pfd{};
        pfd.fd = sock;
        pfd.events = POLLIN;
        int sel = ::poll(&pfd, 1, 100);
        if (sel < 0) return false;
        if (sel == 0) continue;
        if (pfd.revents & (POLLERR | POLLNVAL)) return false;
#endif
        int chunk = static_cast<int>(std::min<size_t>(remaining, 16ULL * 1024 * 1024));
        int recvd = ::recv(sock, ptr, chunk, 0);
        if (recvd <= 0) {
            return false;
        }
        remaining -= static_cast<size_t>(recvd);
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

static void build_aead_nonce(bool sender_is_initiator, uint64_t wire_counter, uint8_t nonce[12]) {
    // Direction-separated 4-byte domain tag + 8-byte strictly monotonic LE counter
    if (sender_is_initiator) {
        nonce[0] = 'C'; nonce[1] = 'T'; nonce[2] = 'W'; nonce[3] = '1';
    } else {
        nonce[0] = 'R'; nonce[1] = 'T'; nonce[2] = 'C'; nonce[3] = '1';
    }
    for (int i = 0; i < 8; ++i) {
        nonce[4 + i] = static_cast<uint8_t>((wire_counter >> (i * 8)) & 0xFFU);
    }
}

static uint64_t load_le64_bytes(const uint8_t p[8]) {
    uint64_t v = 0;
    for (int i = 0; i < 8; ++i) {
        v |= (static_cast<uint64_t>(p[i]) << (i * 8));
    }
    return v;
}

static void store_le64_bytes(uint8_t p[8], uint64_t v) {
    for (int i = 0; i < 8; ++i) {
        p[i] = static_cast<uint8_t>((v >> (i * 8)) & 0xFFU);
    }
}

bool ClusterTransportRing::connect_to_peer(uint32_t node_id, const std::string& address, uint16_t port, uint32_t local_node_id) {
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

    std::string active_token = get_cluster_token();
    PeerSession session{};
    session.socket_handle = static_cast<uint64_t>(client_fd);
    session.is_initiator = true;
    session.is_authenticated = true;
    std::vector<uint8_t> init_key_vec;

    // SHCP v2 Zero-Secret-On-Wire Mutual Challenge-Response Handshake
    if (!active_token.empty()) {
        std::atomic<bool> hs_alive{true};

        // 1. Receive server challenge nonce (36-byte header + 32-byte server_nonce)
        HavenActivationFrameHeader chal_hdr{};
        if (!recv_all(client_fd, &chal_hdr, sizeof(chal_hdr), hs_alive, 5000) ||
            chal_hdr.magic != HAVEN_CLUSTER_MAGIC ||
            chal_hdr.protocol_ver != HAVEN_CLUSTER_VERSION ||
            !(chal_hdr.flags & FRAME_FLAG_CHALLENGE) ||
            chal_hdr.payload_bytes != 32)
        {
            std::cerr << "⛔ [ClusterSecurity] Peer failed to issue valid SHCP v2 challenge.\n";
            closesocket(client_fd);
            return false;
        }
        uint8_t server_nonce[32] = {0};
        if (!recv_all(client_fd, server_nonce, sizeof(server_nonce), hs_alive, 5000)) {
            closesocket(client_fd);
            return false;
        }

        // 2. Generate 256-bit client nonce & compute HMAC-SHA256 client proof (token never leaves host)
        uint8_t client_nonce[32] = {0};
        uint8_t client_proof[32] = {0};
        HavenSecurity::generate_random_bytes(client_nonce, sizeof(client_nonce));
        HavenSecurity::derive_shcp_client_proof(active_token, server_nonce, client_nonce,
                                                static_cast<uint16_t>(local_node_id), client_proof);

        HavenActivationFrameHeader hs_hdr{};
        hs_hdr.magic = HAVEN_CLUSTER_MAGIC;
        hs_hdr.protocol_ver = HAVEN_CLUSTER_VERSION;
        hs_hdr.flags = FRAME_FLAG_HANDSHAKE;
        hs_hdr.from_node_id = static_cast<uint16_t>(local_node_id);
        hs_hdr.to_node_id = static_cast<uint16_t>(node_id);
        hs_hdr.element_count = 16; // 64 bytes / 4
        hs_hdr.payload_bytes = 64;

        uint8_t hs_payload[64];
        std::memcpy(hs_payload, client_nonce, 32);
        std::memcpy(hs_payload + 32, client_proof, 32);

        if (!send_all(client_fd, &hs_hdr, sizeof(hs_hdr)) ||
            !send_all(client_fd, hs_payload, sizeof(hs_payload)))
        {
            HavenSecurity::secure_zero(client_proof, sizeof(client_proof));
            closesocket(client_fd);
            return false;
        }

        // 3. Receive and verify server mutual authentication proof (FRAME_FLAG_HANDSHAKE_ACK)
        HavenActivationFrameHeader ack_hdr{};
        if (!recv_all(client_fd, &ack_hdr, sizeof(ack_hdr), hs_alive, 5000) ||
            ack_hdr.magic != HAVEN_CLUSTER_MAGIC ||
            ack_hdr.protocol_ver != HAVEN_CLUSTER_VERSION ||
            (ack_hdr.flags & FRAME_FLAG_AUTH_REJECT) ||
            !(ack_hdr.flags & FRAME_FLAG_HANDSHAKE_ACK) ||
            ack_hdr.payload_bytes != 32)
        {
            std::cerr << "⛔ [ClusterSecurity] Peer rejected SHCP v2 authentication handshake.\n";
            HavenSecurity::secure_zero(client_proof, sizeof(client_proof));
            closesocket(client_fd);
            return false;
        }
        uint8_t server_proof[32] = {0};
        if (!recv_all(client_fd, server_proof, sizeof(server_proof), hs_alive, 5000)) {
            HavenSecurity::secure_zero(client_proof, sizeof(client_proof));
            closesocket(client_fd);
            return false;
        }

        uint8_t expected_server_proof[32] = {0};
        HavenSecurity::derive_shcp_server_proof(active_token, server_nonce, client_nonce,
                                                static_cast<uint16_t>(local_node_id), expected_server_proof);
        bool server_ok = HavenSecurity::constant_time_memeq(server_proof, expected_server_proof, 32);
        HavenSecurity::secure_zero(expected_server_proof, sizeof(expected_server_proof));
        HavenSecurity::secure_zero(client_proof, sizeof(client_proof));
        if (!server_ok) {
            std::cerr << "⛔ [ClusterSecurity] Server mutual authentication proof FAILED! Possible MitM.\n";
            closesocket(client_fd);
            return false;
        }

        // 4. Derive unique per-session 256-bit ChaCha20-Poly1305 AEAD key
        HavenSecurity::derive_shcp_session_key(active_token, server_nonce, client_nonce, session.session_key);
        session.has_session_key = true;
        init_key_vec.assign(session.session_key, session.session_key + 32);
        HavenSecurity::secure_zero(server_nonce, sizeof(server_nonce));
        HavenSecurity::secure_zero(client_nonce, sizeof(client_nonce));
    }

    uint64_t sock_val = static_cast<uint64_t>(client_fd);
    {
        std::lock_guard<std::mutex> lock(peers_mutex_);
        peer_sessions_[node_id] = session;
    }

    // Launch background receiver thread for this peer connection (outbound)
    {
        std::lock_guard<std::mutex> lock(threads_mutex_);
        client_threads_.emplace_back(&ClusterTransportRing::client_recv_loop, this,
                                     sock_val, node_id, false, session.has_session_key, true, init_key_vec);
    }

    std::cout << "🔗 [ClusterTransportRing] Connected to peer Node " << node_id 
              << " (" << address << ":" << port << ")"
              << (session.has_session_key ? " [SHCP v2 AEAD Mutual Auth]" : "") << "\n";
    return true;
}

void ClusterTransportRing::stop() {
    is_running_.store(false);

    if (inbound_ring_) {
        inbound_ring_->close();
    }

    // 1. Shutdown all peer sockets to immediately unblock any blocking recv calls
    {
        std::lock_guard<std::mutex> lock(peers_mutex_);
        for (auto& [id, sess] : peer_sessions_) {
            socket_t s = static_cast<socket_t>(sess.socket_handle);
            if (s != HAVEN_INVALID_SOCKET) {
#if defined(_WIN32)
                ::shutdown(s, SD_BOTH);
#else
                ::shutdown(s, SHUT_RDWR);
#endif
            }
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

    // 3. Join all client receiver threads cleanly (even if connect_to_peer was used without start_server)
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

    // 4. Close remaining sockets and securely zero all session keys
    {
        std::lock_guard<std::mutex> lock(peers_mutex_);
        for (auto& [id, sess] : peer_sessions_) {
            socket_t s = static_cast<socket_t>(sess.socket_handle);
            if (s != HAVEN_INVALID_SOCKET) {
                closesocket(s);
            }
            HavenSecurity::secure_zero(sess.session_key, sizeof(sess.session_key));
        }
        peer_sessions_.clear();
    }
}

bool ClusterTransportRing::send_activation(uint32_t target_node_id, const HavenActivationFrame& frame) {
    // Lock ordering: ALWAYS send_mutex_ -> peers_mutex_ to preserve strictly monotonic AEAD TX counters
    std::lock_guard<std::mutex> send_lock(send_mutex_);

    socket_t sock = HAVEN_INVALID_SOCKET;
    bool has_key = false;
    bool is_initiator = false;
    uint8_t session_key[32] = {0};
    uint64_t wire_ctr = 0;

    {
        std::lock_guard<std::mutex> lock(peers_mutex_);
        auto it = peer_sessions_.find(target_node_id);
        if (it == peer_sessions_.end() || !it->second.is_authenticated) {
            return false;
        }
        sock = static_cast<socket_t>(it->second.socket_handle);
        has_key = it->second.has_session_key;
        is_initiator = it->second.is_initiator;
        if (has_key) {
            std::memcpy(session_key, it->second.session_key, 32);
            wire_ctr = ++it->second.tx_counter;
        }
    }

    if (sock == HAVEN_INVALID_SOCKET) {
        HavenSecurity::secure_zero(session_key, sizeof(session_key));
        return false;
    }

    const bool is_handshake_frame = (frame.header.flags & FRAME_FLAG_HANDSHAKE) != 0;

    if (has_key && !is_handshake_frame) {
        HavenActivationFrameHeader wire_hdr = frame.header;
        wire_hdr.flags |= FRAME_FLAG_ENCRYPTED;
        const size_t pt_bytes = frame.data.size() * sizeof(float);
        wire_hdr.element_count = static_cast<uint32_t>(frame.data.size());
        // Wire payload = ciphertext (pt_bytes) + wire_counter (8B) + Poly1305 tag (16B)
        wire_hdr.payload_bytes = static_cast<uint32_t>(pt_bytes + 24);

        uint8_t nonce[12];
        build_aead_nonce(is_initiator, wire_ctr, nonce);

        std::vector<uint8_t> wire_payload(pt_bytes + 24, 0);
        uint8_t tag[16] = {0};
        HavenSecurity::aead_encrypt(
            session_key, nonce,
            reinterpret_cast<const uint8_t*>(&wire_hdr), sizeof(wire_hdr),
            pt_bytes > 0 ? reinterpret_cast<const uint8_t*>(frame.data.data()) : nullptr, pt_bytes,
            pt_bytes > 0 ? wire_payload.data() : nullptr, tag
        );
        store_le64_bytes(wire_payload.data() + pt_bytes, wire_ctr);
        std::memcpy(wire_payload.data() + pt_bytes + 8, tag, 16);
        HavenSecurity::secure_zero(session_key, sizeof(session_key));

        if (!send_all(sock, &wire_hdr, sizeof(wire_hdr))) {
            return false;
        }
        return send_all(sock, wire_payload.data(), wire_payload.size());
    }

    HavenSecurity::secure_zero(session_key, sizeof(session_key));

    // Unauthenticated local benchmark mode or control frame
    if (!send_all(sock, &frame.header, sizeof(frame.header))) {
        return false;
    }
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
    auto it = peer_sessions_.find(node_id);
    return it != peer_sessions_.end() && it->second.is_authenticated;
}

std::vector<uint32_t> ClusterTransportRing::get_connected_peer_ids() const {
    std::lock_guard<std::mutex> lock(peers_mutex_);
    std::vector<uint32_t> ids;
    ids.reserve(peer_sessions_.size());
    for (const auto& [id, sess] : peer_sessions_) {
        if (sess.is_authenticated) {
            ids.push_back(id);
        }
    }
    return ids;
}

void ClusterTransportRing::server_listen_loop() {
    socket_t server_fd = static_cast<socket_t>(server_socket_);
    uint32_t incoming_node_counter = 100;
    constexpr size_t MAX_CLUSTER_PEERS = 64;

    while (is_running_) {
#ifdef _WIN32
        fd_set read_fds;
        FD_ZERO(&read_fds);
        FD_SET(server_fd, &read_fds);

        timeval tv;
        tv.tv_sec = 0;
        tv.tv_usec = 100000; // 100ms poll timeout

        int sel = ::select(0, &read_fds, nullptr, nullptr, &tv);
        if (sel <= 0) {
            continue;
        }
#else
        struct pollfd pfd{};
        pfd.fd = server_fd;
        pfd.events = POLLIN;
        int sel = ::poll(&pfd, 1, 100);
        if (sel <= 0) {
            continue;
        }
#endif

        sockaddr_in client_addr{};
        socklen_t client_len = sizeof(client_addr);
        socket_t client_fd = ::accept(server_fd, (struct sockaddr*)&client_addr, &client_len);

        if (client_fd == HAVEN_INVALID_SOCKET) {
            if (!is_running_) break;
            continue;
        }

        if (active_peer_threads_.load(std::memory_order_relaxed) >= MAX_CLUSTER_PEERS) {
            std::cerr << "⛔ [ClusterTransportRing] Inbound peer limit (" << MAX_CLUSTER_PEERS << ") reached. Dropping connection.\n";
            closesocket(client_fd);
            continue;
        }

        configure_cluster_socket(client_fd);
        uint32_t assigned_id = incoming_node_counter++;
        uint64_t sock_val = static_cast<uint64_t>(client_fd);
        const bool token_required = !get_cluster_token().empty();

        {
            std::lock_guard<std::mutex> lock(peers_mutex_);
            PeerSession pending{};
            pending.socket_handle = sock_val;
            pending.is_authenticated = !token_required;
            pending.is_initiator = false;
            pending.has_session_key = false;
            peer_sessions_[assigned_id] = pending;
        }

        active_peer_threads_.fetch_add(1, std::memory_order_relaxed);
        {
            std::lock_guard<std::mutex> lock(threads_mutex_);
            client_threads_.emplace_back(&ClusterTransportRing::client_recv_loop, this,
                                         sock_val, assigned_id, true, false, false, std::vector<uint8_t>{});
        }
    }
}

void ClusterTransportRing::client_recv_loop(uint64_t socket_handle, uint32_t peer_node_id, bool require_auth,
                                            bool pre_auth_aead, bool local_is_initiator,
                                            const std::vector<uint8_t>& initial_session_key) {
    struct PeerThreadGuard {
        std::atomic<size_t>& counter;
        bool tracked;
        ~PeerThreadGuard() {
            if (tracked) counter.fetch_sub(1, std::memory_order_relaxed);
        }
    } thread_guard{active_peer_threads_, require_auth};

    socket_t sock = static_cast<socket_t>(socket_handle);
    std::string init_token = get_cluster_token();
    bool is_authenticated = !require_auth || init_token.empty();
    bool has_aead_session = pre_auth_aead && initial_session_key.size() == 32;
    uint8_t session_key[32] = {0};
    if (has_aead_session) {
        std::memcpy(session_key, initial_session_key.data(), 32);
    }
    uint64_t rx_last_counter = 0;

    uint8_t server_nonce[32] = {0};
    bool challenge_issued = false;

    auto detach_and_close_sock = [&]() {
        // Erase from peer_sessions_ BEFORE closing the socket so stop() can never double-close it
        {
            std::lock_guard<std::mutex> lock(peers_mutex_);
            auto it = peer_sessions_.find(peer_node_id);
            if (it != peer_sessions_.end() && it->second.socket_handle == socket_handle) {
                HavenSecurity::secure_zero(it->second.session_key, sizeof(it->second.session_key));
                peer_sessions_.erase(it);
            }
        }
        if (sock != HAVEN_INVALID_SOCKET) {
            closesocket(sock);
            sock = HAVEN_INVALID_SOCKET;
        }
    };

    // Immediately issue 256-bit random challenge nonce to inbound peers when cluster token is active
    if (require_auth && !init_token.empty()) {
        HavenSecurity::generate_random_bytes(server_nonce, sizeof(server_nonce));
        HavenActivationFrameHeader chal_hdr{};
        chal_hdr.magic = HAVEN_CLUSTER_MAGIC;
        chal_hdr.protocol_ver = HAVEN_CLUSTER_VERSION;
        chal_hdr.flags = FRAME_FLAG_CHALLENGE;
        chal_hdr.element_count = 8; // 32 bytes / 4
        chal_hdr.payload_bytes = 32;
        if (!send_all(sock, &chal_hdr, sizeof(chal_hdr)) ||
            !send_all(sock, server_nonce, sizeof(server_nonce)))
        {
            detach_and_close_sock();
        } else {
            challenge_issued = true;
        }
    }

    auto send_reject_and_close = [&]() {
        if (sock != HAVEN_INVALID_SOCKET) {
            HavenActivationFrameHeader reject_hdr{};
            reject_hdr.magic = HAVEN_CLUSTER_MAGIC;
            reject_hdr.protocol_ver = HAVEN_CLUSTER_VERSION;
            reject_hdr.flags = FRAME_FLAG_AUTH_REJECT;
            send_all(sock, &reject_hdr, sizeof(reject_hdr));
#ifdef _WIN32
            ::shutdown(sock, SD_SEND);
#else
            ::shutdown(sock, SHUT_WR);
#endif
            detach_and_close_sock();
        }
    };

    while (is_running_ && sock != HAVEN_INVALID_SOCKET) {
        HavenActivationFrame frame;
        int hdr_deadline = is_authenticated ? 0 : 5000;
        if (!recv_all(sock, &frame.header, sizeof(frame.header), is_running_, hdr_deadline)) {
            break;
        }

        if (frame.header.magic != HAVEN_CLUSTER_MAGIC || frame.header.protocol_ver != HAVEN_CLUSTER_VERSION) {
            std::cerr << "⚠️ [ClusterTransportRing] Corrupted packet header received from peer Node " << peer_node_id << "\n";
            break;
        }

        const bool is_handshake = (frame.header.flags & FRAME_FLAG_HANDSHAKE) != 0;
        const bool is_encrypted = (frame.header.flags & FRAME_FLAG_ENCRYPTED) != 0;

        // Enforce authentication gate BEFORE accepting any non-handshake frame
        if (!is_authenticated && !is_handshake) {
            std::cerr << "⛔ [ClusterSecurity] Unauthenticated frame (flags=0x" << std::hex << frame.header.flags
                      << std::dec << ") rejected on socket " << socket_handle << ". Handshake required.\n";
            send_reject_and_close();
            break;
        }

        constexpr uint32_t MAX_FRAME_BYTES = 256U * 1024U * 1024U;
        constexpr uint32_t MAX_HANDSHAKE_BYTES = 256U;

        // Handle Handshake (SHCP v2 Challenge-Response when token active; legacy node-id announce when token empty)
        if (is_handshake) {
            if (has_aead_session || (challenge_issued && is_authenticated)) {
                std::cerr << "⛔ [ClusterSecurity] Re-handshake rejected on authenticated session (Node "
                          << peer_node_id << "). Severing connection.\n";
                send_reject_and_close();
                break;
            }
            uint32_t real_node_id = frame.header.from_node_id;
            std::string active_token = get_cluster_token();

            if (challenge_issued || !active_token.empty()) {
                if (!challenge_issued || frame.header.payload_bytes != 64 || frame.header.element_count != 16) {
                    std::cerr << "⛔ [ClusterSecurity] Invalid SHCP v2 handshake size on socket " << socket_handle << "\n";
                    send_reject_and_close();
                    break;
                }
                uint8_t hs_payload[64] = {0};
                if (!recv_all(sock, hs_payload, sizeof(hs_payload), is_running_, 5000)) {
                    break;
                }
                const uint8_t* client_nonce = hs_payload;
                const uint8_t* client_proof = hs_payload + 32;

                uint8_t expected_client_proof[32] = {0};
                HavenSecurity::derive_shcp_client_proof(active_token, server_nonce, client_nonce,
                                                        static_cast<uint16_t>(real_node_id), expected_client_proof);
                bool verified = HavenSecurity::constant_time_memeq(client_proof, expected_client_proof, 32);
                HavenSecurity::secure_zero(expected_client_proof, sizeof(expected_client_proof));

                if (!verified) {
                    std::cerr << "⛔ [ClusterSecurity] Unauthorized peer rejected on socket "
                              << socket_handle << " (Invalid SHCP v2 HMAC proof). Terminating connection.\n";
                    send_reject_and_close();
                    break;
                }

                // Send mutual server proof (FRAME_FLAG_HANDSHAKE_ACK)
                uint8_t server_proof[32] = {0};
                HavenSecurity::derive_shcp_server_proof(active_token, server_nonce, client_nonce,
                                                        static_cast<uint16_t>(real_node_id), server_proof);
                HavenActivationFrameHeader ack_hdr{};
                ack_hdr.magic = HAVEN_CLUSTER_MAGIC;
                ack_hdr.protocol_ver = HAVEN_CLUSTER_VERSION;
                ack_hdr.flags = FRAME_FLAG_HANDSHAKE_ACK;
                ack_hdr.element_count = 8;
                ack_hdr.payload_bytes = 32;
                bool ack_sent = send_all(sock, &ack_hdr, sizeof(ack_hdr)) &&
                                send_all(sock, server_proof, sizeof(server_proof));
                HavenSecurity::secure_zero(server_proof, sizeof(server_proof));
                if (!ack_sent) {
                    break;
                }

                // Derive per-session 256-bit ChaCha20-Poly1305 AEAD key
                HavenSecurity::derive_shcp_session_key(active_token, server_nonce, client_nonce, session_key);
                has_aead_session = true;
                is_authenticated = true;
                HavenSecurity::secure_zero(server_nonce, sizeof(server_nonce));

                {
                    std::lock_guard<std::mutex> lock(peers_mutex_);
                    peer_sessions_.erase(peer_node_id);
                    PeerSession authed{};
                    authed.socket_handle = socket_handle;
                    authed.is_authenticated = true;
                    authed.is_initiator = false;
                    authed.has_session_key = true;
                    std::memcpy(authed.session_key, session_key, 32);
                    authed.tx_counter = 0;
                    peer_sessions_[real_node_id] = authed;
                    peer_node_id = real_node_id;
                }

                std::cout << "🔒 [ClusterSecurity] Peer Node " << real_node_id
                          << " mutually authenticated (SHCP v2 HMAC-SHA256 + ChaCha20-Poly1305 AEAD).\n";
                continue;
            } else {
                // Unauthenticated local benchmark mode handshake
                uint32_t expected_elems = (frame.header.payload_bytes + sizeof(float) - 1) / sizeof(float);
                if (frame.header.payload_bytes > MAX_HANDSHAKE_BYTES || frame.header.element_count != expected_elems) {
                    break;
                }
                if (frame.header.payload_bytes > 0) {
                    std::vector<uint8_t> discard(frame.header.payload_bytes);
                    if (!recv_all(sock, discard.data(), discard.size(), is_running_, 5000)) break;
                }
                is_authenticated = true;
                if (real_node_id != peer_node_id) {
                    std::lock_guard<std::mutex> lock(peers_mutex_);
                    auto it = peer_sessions_.find(peer_node_id);
                    PeerSession sess = (it != peer_sessions_.end()) ? it->second : PeerSession{};
                    sess.socket_handle = socket_handle;
                    sess.is_authenticated = true;
                    peer_sessions_.erase(peer_node_id);
                    peer_sessions_[real_node_id] = sess;
                    peer_node_id = real_node_id;
                }
                continue;
            }
        }

        // Enforce zero plaintext downgrade on ALL non-handshake frames once AEAD session is established
        if (has_aead_session && !is_encrypted) {
            std::cerr << "⛔ [ClusterSecurity] Plaintext downgrade frame rejected on AEAD session (Node "
                      << peer_node_id << "). Severing connection.\n";
            send_reject_and_close();
            break;
        }

        if (has_aead_session && is_encrypted) {
            // Wire format: [ciphertext (element_count * 4) || wire_counter (8B) || Poly1305 tag (16B)]
            if (frame.header.element_count > (MAX_FRAME_BYTES - 24U) / sizeof(float) ||
                frame.header.payload_bytes < 24U ||
                frame.header.payload_bytes > MAX_FRAME_BYTES ||
                frame.header.payload_bytes != frame.header.element_count * sizeof(float) + 24U)
            {
                std::cerr << "⛔ [ClusterSecurity] Invalid AEAD frame geometry from Node " << peer_node_id << "\n";
                send_reject_and_close();
                break;
            }

            std::vector<uint8_t> wire_buf(frame.header.payload_bytes);
            if (!recv_all(sock, wire_buf.data(), wire_buf.size(), is_running_, 0)) {
                break;
            }

            const size_t pt_bytes = static_cast<size_t>(frame.header.element_count) * sizeof(float);
            uint64_t wire_ctr = load_le64_bytes(wire_buf.data() + pt_bytes);
            const uint8_t* tag = wire_buf.data() + pt_bytes + 8;

            // Strict monotonic anti-replay gate
            if (wire_ctr == 0 || wire_ctr <= rx_last_counter) {
                std::cerr << "⛔ [ClusterSecurity] Replay/out-of-order frame detected from Node "
                          << peer_node_id << " (ctr=" << wire_ctr << ", last=" << rx_last_counter
                          << "). Severing connection.\n";
                send_reject_and_close();
                break;
            }

            // Direction of sender is opposite of local side
            uint8_t nonce[12];
            build_aead_nonce(!local_is_initiator, wire_ctr, nonce);

            frame.data.resize(frame.header.element_count, 0.0f);
            bool aead_ok = HavenSecurity::aead_decrypt(
                session_key, nonce,
                reinterpret_cast<const uint8_t*>(&frame.header), sizeof(frame.header),
                pt_bytes > 0 ? wire_buf.data() : nullptr, pt_bytes,
                tag,
                pt_bytes > 0 ? reinterpret_cast<uint8_t*>(frame.data.data()) : nullptr
            );

            if (!aead_ok) {
                std::cerr << "⛔ [ClusterSecurity] Poly1305 AEAD authentication FAILED on frame from Node "
                          << peer_node_id << " (Tampering detected). Severing connection.\n";
                send_reject_and_close();
                break;
            }

            rx_last_counter = wire_ctr;
            frame.header.payload_bytes = static_cast<uint32_t>(pt_bytes);
            inbound_ring_->push(std::move(frame));
            continue;
        }

        // Unauthenticated local benchmark mode payload read
        if (frame.header.payload_bytes > 0 || frame.header.element_count > 0) {
            if (frame.header.element_count > MAX_FRAME_BYTES / sizeof(float) ||
                frame.header.payload_bytes > MAX_FRAME_BYTES ||
                frame.header.payload_bytes != frame.header.element_count * sizeof(float))
            {
                std::cerr << "⛔ [ClusterTransportRing] Frame header invariant violation (payload_bytes="
                          << frame.header.payload_bytes << ", element_count=" << frame.header.element_count << ")\n";
                break;
            }
            frame.data.resize(frame.header.element_count, 0.0f);
            if (frame.header.payload_bytes > 0) {
                if (!recv_all(sock, frame.data.data(), frame.header.payload_bytes, is_running_, 0)) {
                    break;
                }
            }
        }

        inbound_ring_->push(std::move(frame));
    }

    HavenSecurity::secure_zero(session_key, sizeof(session_key));
    HavenSecurity::secure_zero(server_nonce, sizeof(server_nonce));

    // Peer disconnected or stopping: atomically detach from peer_sessions_ before closing socket
    detach_and_close_sock();
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

    {
        std::lock_guard<std::mutex> lock(state_mutex_);
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
    }

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

    if (!transport_->connect_to_peer(0, coordinator_addr, coordinator_port, local_node_id)) {
        std::cerr << "⚠️ [HavenClusterNode] Worker failed to connect to coordinator at " 
                  << coordinator_addr << ":" << coordinator_port << "\n";
        transport_->stop();
        return false;
    }

    // In unauthenticated benchmark mode, send node-ID announce frame;
    // when cluster token is active, connect_to_peer already completed SHCP v2 mutual HMAC handshake.
    if (get_cluster_token().empty()) {
        HavenActivationFrame handshake;
        handshake.header.magic = HAVEN_CLUSTER_MAGIC;
        handshake.header.protocol_ver = HAVEN_CLUSTER_VERSION;
        handshake.header.flags = FRAME_FLAG_HANDSHAKE;
        handshake.header.from_node_id = static_cast<uint16_t>(local_node_id);
        handshake.header.to_node_id = 0;
        handshake.header.element_count = 0;
        handshake.header.payload_bytes = 0;
        transport_->send_activation(0, handshake);
    }

    is_active_ = true;
    std::cout << "⚡ [HavenClusterNode] Worker Node " << local_node_id << " joined coordinator @ "
              << coordinator_addr << ":" << coordinator_port << "\n";
    return true;
}

void HavenClusterNode::set_cluster_token(const std::string& token) {
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        cluster_token_ = token;
    }
    if (transport_) {
        transport_->set_cluster_token(token);
    }
}

std::string HavenClusterNode::generate_invite_uri(const std::string& host_override) {
    std::string host = host_override.empty() ? "127.0.0.1" : host_override;
    std::string current_token;
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        current_token = cluster_token_;
    }
    if (current_token.empty()) {
        current_token = HavenSecurity::generate_cluster_token();
        set_cluster_token(current_token);
    }
    return HavenSecurity::create_invite_uri(host, listen_port_, current_token);
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
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        nodes_.clear();
    }
}

bool HavenClusterNode::register_worker(const ClusterNodeInfo& worker) {
    std::lock_guard<std::mutex> lock(state_mutex_);
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

    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        uint32_t total = 0;
        for (const auto& n : nodes_) {
            if (n.end_layer > total) total = n.end_layer;
        }

        scheduler_.partition_layers(total > 0 ? total : 32, nodes_);
    }

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
    if (!transport_->connect_to_peer(node_id, address, port, local_node_id_)) {
        return false;
    }

    // In unauthenticated benchmark mode, announce node_id;
    // when token is active, connect_to_peer already completed SHCP v2 mutual HMAC handshake.
    if (get_cluster_token().empty()) {
        HavenActivationFrame handshake;
        handshake.header.magic = HAVEN_CLUSTER_MAGIC;
        handshake.header.protocol_ver = HAVEN_CLUSTER_VERSION;
        handshake.header.flags = FRAME_FLAG_HANDSHAKE;
        handshake.header.from_node_id = static_cast<uint16_t>(local_node_id_);
        handshake.header.to_node_id = static_cast<uint16_t>(node_id);
        handshake.header.element_count = 0;
        handshake.header.payload_bytes = 0;
        return transport_->send_activation(node_id, handshake);
    }
    return true;
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
