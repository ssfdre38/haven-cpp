#include "haven/haven_engine.h"
#include "haven/haven_mlx.h"
#include "haven/haven_cluster.h"
#include "haven/haven_json.h"
#include "haven/haven_tpu_pjrt.h"
#include "haven/haven_tpu_gemma4.h"
#include "haven/haven_tpu_scheduler.h"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <set>
#include <thread>
#include <sstream>
#include <chrono>
#include <filesystem>
#include <charconv>
#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstring>
#include <cstdlib>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <winsock2.h>
#include <ws2tcpip.h>
#include <psapi.h>
typedef SOCKET socket_t;
#define HAVEN_INVALID_SOCKET INVALID_SOCKET
#else
#include <sys/socket.h>
#include <sys/select.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
typedef int socket_t;
#define HAVEN_INVALID_SOCKET (-1)
#define closesocket close
#endif

namespace haven {

struct TpuServerOptions {
    bool enable_onchip = false;
    bool use_int8 = true;
    bool use_fp8_kv = false;
    bool use_onchip_embeddings = true;
    bool use_onchip_argmax = false; // true = 4B/slot on-chip argmax for max multi-tenant throughput
    int num_replicas = 0; // 0 = auto (max power-of-2 <= addressable TPU chips, capped at 8)
    int batch_size = 1;   // 1 = interactive w/ prefix cache; 2..256 = multi-tenant continuous batching
    int max_seq_len = 2048;
    std::string libtpu_path = "";
};

class HavenMicroServer {
public:
    HavenMicroServer(int port = 11436, const std::string& model_path = "",
                     bool is_cluster_worker = false,
                     const std::string& cluster_join_target = "",
                     uint16_t cluster_port = 0,
                     const std::string& cluster_token = "",
                     uint32_t cluster_node_id = 0,
                     const std::string& bind_host = "127.0.0.1",
                     const std::string& api_key = "",
                     const TpuServerOptions& tpu_opts = {})
        : port_(port), bind_host_(bind_host.empty() ? "127.0.0.1" : bind_host),
          api_key_(api_key), tpu_opts_(tpu_opts), running_(false), model_loaded_(false)
    {
#ifdef _WIN32
        WSADATA wsa;
        WSAStartup(MAKEWORD(2, 2), &wsa);
#endif
        // Initialize 64-bit Haven Memory Bank (.hmb)
        std::string hmb_path = "wwwroot/aura_vault.hmb";
        std::string json_path = "wwwroot/memories.json";

        if (std::filesystem::exists(hmb_path)) {
            haven_engine_.load_memories(hmb_path);
        } else if (std::filesystem::exists(json_path)) {
            std::cout << "[HavenServer] Migrating memories.json -> 64-bit Haven Memory Bank (" << hmb_path << ")...\n" << std::flush;
            haven_engine_.load_memories(json_path);
            haven_engine_.save_memories(hmb_path);
        } else {
            haven_engine_.get_memory_engine().initialize_sovereign_anchors();
            haven_engine_.save_memories(hmb_path);
            haven_engine_.save_memories(json_path);
        }

        std::string path_to_load = model_path;
        if (path_to_load.empty()) {
            if (std::filesystem::exists("C:\\Users\\admin\\gemma4-turbo-family\\haven-chat-v5.0.gguf")) {
                path_to_load = "C:\\Users\\admin\\gemma4-turbo-family\\haven-chat-v5.0.gguf";
            }
        }
        if (!path_to_load.empty()) {
            std::string err;
            load_model_internal(path_to_load, err);
        }
        load_persona();
        load_model_dirs();

        if (is_cluster_worker) {
            uint16_t worker_port = cluster_port != 0 ? cluster_port : (uint16_t)(port_ + 3);
            uint32_t n_id = cluster_node_id != 0 ? cluster_node_id : 1;
            cluster_node_.init_worker_with_token(n_id, worker_port, cluster_join_target, cluster_port, cluster_token);
        } else {
            uint16_t coord_port = cluster_port != 0 ? cluster_port : (uint16_t)(port_ + 2);
            // Zero-trust default: always enforce a 256-bit pairing token on the cluster coordinator
            std::string active_token = cluster_token.empty()
                ? HavenSecurity::generate_cluster_token()
                : cluster_token;
            cluster_node_.set_cluster_token(active_token);
            cluster_node_.init_coordinator(coord_port, 32);
            std::cout << "🔒 [HavenServer] Sovereign Cluster Invite URI: " << cluster_node_.generate_invite_uri() << "\n" << std::flush;
        }
    }

    ~HavenMicroServer() {
        stop();
        tpu_scheduler_.stop();
        tpu_gemma4_.shutdown();
        tpu_backend_.shutdown();
#ifdef _WIN32
        WSACleanup();
#endif
    }

    void start() {
        running_ = true;
        server_thread_ = std::thread(&HavenMicroServer::run, this);
    }

    void stop() {
        running_ = false;
#ifdef _WIN32
        if (server_fd_ != INVALID_SOCKET) {
            closesocket(server_fd_);
            server_fd_ = INVALID_SOCKET;
        }
#else
        if (server_fd_ >= 0) {
            close(server_fd_);
            server_fd_ = -1;
        }
#endif
        if (server_thread_.joinable()) {
            server_thread_.join();
        }
    }

private:
    int port_;
    std::string bind_host_;
    std::string api_key_;
    TpuServerOptions tpu_opts_;
#ifdef _WIN32
    SOCKET server_fd_ = INVALID_SOCKET;
#else
    int server_fd_ = -1;
#endif
    std::atomic<bool> running_{false};
    std::atomic<size_t> active_clients_{0};
    std::thread server_thread_;
    HavenEngine haven_engine_;
    HavenClusterNode cluster_node_;
    HavenMLXEngine mlx_engine_;
    HavenTpuPjrtBackend tpu_backend_;
    HavenTpuGemma4Engine tpu_gemma4_;
    TpuContinuousBatchScheduler tpu_scheduler_;
    bool is_mlx_active_ = false;
    bool tpu_onchip_active_ = false;
    std::atomic<bool> tpu_scheduler_active_{false};
    std::mutex engine_mutex_;
    std::vector<uint32_t> cached_tokens_;
    bool model_loaded_;

    std::string current_model_name_ = "";
    std::string current_model_path_ = "";
    std::vector<std::string> model_search_dirs_ = {
        "C:\\Users\\admin\\gemma4-turbo-family",
        "models",
        "."
    };

    static void close_socket(socket_t s) {
#ifdef _WIN32
        closesocket(s);
#else
        close(s);
#endif
    }

    static bool is_safe_model_token(const std::string& name) {
        if (name.empty() || name.size() > 260) return false;
        if (name.find("..") != std::string::npos ||
            name.find('/') != std::string::npos ||
            name.find('\\') != std::string::npos ||
            name.find(':') != std::string::npos ||
            name.find('\0') != std::string::npos) {
            return false;
        }
        return true;
    }

    static bool resolve_safe_path(const std::filesystem::path& base_dir, const std::string& clean_rel, std::string& out_path) {
        std::error_code ec;
        std::filesystem::path root_dir = std::filesystem::weakly_canonical(base_dir, ec);
        if (ec) return false;
        std::filesystem::path target = std::filesystem::weakly_canonical(root_dir / clean_rel, ec);
        if (ec) return false;

        std::string root_str = root_dir.string();
        std::string target_str = target.string();
        while (root_str.size() > 1 && (root_str.back() == '\\' || root_str.back() == '/')) {
            root_str.pop_back();
        }
        if (target_str.size() <= root_str.size()) return false;
        if (target_str.compare(0, root_str.size(), root_str) != 0) return false;
        char sep = target_str[root_str.size()];
        if (sep != '\\' && sep != '/') return false;
        out_path = target_str;
        return true;
    }

    static std::string to_lower_ascii(std::string s) {
        for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        return s;
    }

    static std::string trim_ows(const std::string& s) {
        size_t f = s.find_first_not_of(" \t\r\n");
        if (f == std::string::npos) return "";
        size_t l = s.find_last_not_of(" \t\r\n");
        return s.substr(f, l - f + 1);
    }

    static std::string get_http_header(const std::string& raw_headers, const std::string& target_lower, size_t* match_count = nullptr) {
        if (match_count) *match_count = 0;
        std::string first_val;
        size_t line_start = raw_headers.find("\r\n");
        if (line_start == std::string::npos) return "";
        line_start += 2;
        while (line_start < raw_headers.size()) {
            size_t line_end = raw_headers.find("\r\n", line_start);
            if (line_end == std::string::npos) line_end = raw_headers.size();
            if (line_end == line_start) break;
            size_t colon = raw_headers.find(':', line_start);
            if (colon != std::string::npos && colon < line_end) {
                std::string key = to_lower_ascii(trim_ows(raw_headers.substr(line_start, colon - line_start)));
                if (key == target_lower) {
                    if (match_count) (*match_count)++;
                    if (first_val.empty()) {
                        first_val = trim_ows(raw_headers.substr(colon + 1, line_end - colon - 1));
                    }
                }
            }
            if (line_end >= raw_headers.size()) break;
            line_start = line_end + 2;
        }
        return first_val;
    }

    static bool is_loopback_host_header(const std::string& host_hdr) {
        if (host_hdr.empty()) return true; // HTTP/1.0 health checks
        std::string h = to_lower_ascii(host_hdr);
        if (h.starts_with("[::1]")) {
            return h.size() == 5 || h[5] == ':';
        }
        size_t colon = h.find(':');
        std::string hostname = (colon != std::string::npos) ? h.substr(0, colon) : h;
        return (hostname == "127.0.0.1" || hostname == "localhost");
    }

    static bool is_trusted_origin(const std::string& origin) {
        if (origin.empty()) return true;
        std::string o = to_lower_ascii(origin);
        auto check_prefix = [&](const char* prefix) -> bool {
            size_t plen = std::strlen(prefix);
            if (o.size() < plen || o.compare(0, plen, prefix) != 0) return false;
            if (o.size() == plen) return true;
            if (o[plen] == ':') {
                if (plen + 1 >= o.size()) return false;
                for (size_t i = plen + 1; i < o.size(); ++i) {
                    if (!std::isdigit(static_cast<unsigned char>(o[i]))) return false;
                }
                return true;
            }
            return false;
        };
        return check_prefix("http://127.0.0.1") ||
               check_prefix("https://127.0.0.1") ||
               check_prefix("http://localhost") ||
               check_prefix("https://localhost") ||
               check_prefix("http://[::1]") ||
               check_prefix("https://[::1]");
    }

    static std::string build_cors_headers(const std::string& origin) {
        if (!origin.empty() && is_trusted_origin(origin)) {
            return "Access-Control-Allow-Origin: " + origin + "\r\nVary: Origin\r\n";
        }
        return "";
    }

    static void send_http_response(socket_t sock, const std::string& status, const std::string& content_type,
                                   const std::string& body, const std::string& cors_hdr = "") {
        std::string resp = "HTTP/1.1 " + status + "\r\n"
                           "Content-Type: " + content_type + "\r\n"
                           "X-Content-Type-Options: nosniff\r\n"
                           "X-Frame-Options: SAMEORIGIN\r\n"
                           + cors_hdr +
                           "Connection: close\r\n"
                           "Content-Length: " + std::to_string(body.length()) + "\r\n\r\n"
                           + body;
        send_all(sock, resp.c_str(), static_cast<int>(resp.length()));
    }

    static std::string format_bytes(uintmax_t bytes) {
        if (bytes == 0) return "0 B";
        char buf[32];
        if (bytes >= 1024ULL * 1024 * 1024) {
            snprintf(buf, sizeof(buf), "%.2f GB", (double)bytes / (1024.0 * 1024.0 * 1024.0));
        } else if (bytes >= 1024ULL * 1024) {
            snprintf(buf, sizeof(buf), "%.1f MB", (double)bytes / (1024.0 * 1024.0));
        } else {
            snprintf(buf, sizeof(buf), "%llu B", (unsigned long long)bytes);
        }
        return std::string(buf);
    }

    bool load_model_internal(const std::string& target_path, std::string& err_out) {
        std::cout << "[HavenServer] Loading Sovereign Model into Memory: " << target_path << "...\n" << std::flush;
        cached_tokens_.clear();
        is_mlx_active_ = false;
        tpu_scheduler_active_.store(false, std::memory_order_release);
        tpu_scheduler_.stop();
        tpu_onchip_active_ = false;
        tpu_gemma4_.shutdown();
        mlx_engine_.unload();

        if (!haven_engine_.load_model(target_path)) {
            err_out = "Failed to load GGUF file: " + target_path;
            model_loaded_ = false;
            return false;
        }

        current_model_path_ = target_path;
        current_model_name_ = std::filesystem::path(target_path).filename().string();
        model_loaded_ = true;
        std::cout << "[HavenServer] ✓ Sovereign Model Loaded in-process (" << current_model_name_ << ")! Standalone generation active.\n" << std::flush;

        // Optimal default hyperparameters for Sovereign Persona (configured before TPU scheduler snapshot!)
        haven_engine_.get_sampler().get_params().temperature = 0.48f;
        haven_engine_.get_sampler().get_params().top_p = 0.90f;
        haven_engine_.get_sampler().get_params().top_k = 40;
        haven_engine_.get_sampler().get_params().min_p = 0.05f;
        haven_engine_.get_sampler().get_params().repetition_penalty = 1.08f;

        // Suppress loose asterisks for clean, grounded spoken conversation
        auto& tokenizer = haven_engine_.get_tokenizer();
        for (const std::string& ast : {"*", " *", "**", " **", "***"}) {
            auto toks = tokenizer.encode(ast, false);
            for (uint32_t t : toks) {
                haven_engine_.get_sampler().add_anti_robotic_penalty(t, 2.5f);
            }
        }

        // Optional Model-General Full On-Chip Cloud TPU HBM3 Acceleration (--tpu / HAVEN_TPU_ONCHIP=1)
        if (tpu_opts_.enable_onchip) {
            const auto& mc = haven_engine_.get_config();
            std::string support_err;
            if (HavenTpuGemma4Engine::is_model_supported(haven_engine_, support_err)) {
                if (!tpu_backend_.is_initialized()) {
                    if (!tpu_backend_.initialize(tpu_opts_.libtpu_path)) {
                        std::cout << "[HavenServer] TPU requested but unavailable (" << tpu_backend_.get_last_error()
                                  << ") — falling back to AVX2 CPU.\n" << std::flush;
                    }
                }
                if (tpu_backend_.is_initialized() && tpu_backend_.get_device_count() >= 1) {
                    const int n_devs = static_cast<int>(tpu_backend_.get_device_count());
                    int R = tpu_opts_.num_replicas;
                    if (R <= 0 || R > n_devs || (R != 1 && R != 2 && R != 4 && R != 8)) {
                        R = (n_devs >= 8) ? 8 : ((n_devs >= 4) ? 4 : ((n_devs >= 2) ? 2 : 1));
                    }
                    // Ensure clean Megatron TP divisibility across hidden_dim, expert_ffn_dim, num_heads, vocab_size
                    while (R > 1 && ((mc.hidden_dim % R) != 0 || (mc.num_heads % R) != 0 || (mc.vocab_size % R) != 0 ||
                                     (mc.expert_count > 0 && mc.expert_ffn_dim > 0 && (mc.expert_ffn_dim % R) != 0))) {
                        R /= 2;
                    }

                    TpuGemma4Config tcfg;
                    tcfg.batch_size = std::clamp(tpu_opts_.batch_size, 1, 256);
                    tcfg.max_seq_len = std::clamp(tpu_opts_.max_seq_len, 256, 8192);
                    tcfg.prefill_chunk_size = 128; // Auto-compiles slot-targeted {128, 64, 32} (+ 1 when B>1)
                    tcfg.num_replicas = R;
                    tcfg.use_int8_weights = tpu_opts_.use_int8;
                    tcfg.use_fp8_kv = tpu_opts_.use_fp8_kv;
                    tcfg.return_full_logits = !tpu_opts_.use_onchip_argmax;
                    tcfg.use_onchip_embeddings = tpu_opts_.use_onchip_embeddings;
                    tcfg.donate_kv_cache = true;

                    std::cout << "[HavenServer] Initializing Model-General On-Chip TPU Engine (" << mc.architecture
                              << ", " << mc.num_layers << "L, R=" << R
                              << ", B=" << tcfg.batch_size
                              << ", " << (tcfg.use_int8_weights ? "HadW8-INT8" : "BF16")
                              << ", KV=" << (tcfg.use_fp8_kv ? "FP8-E4M3FN" : "BF16")
                              << ", " << (tcfg.return_full_logits ? "FullSampler" : "OnChipArgmax")
                              << ", max_seq=" << tcfg.max_seq_len
                              << ", onchip_emb=" << (tcfg.use_onchip_embeddings ? "true" : "false") << ")...\n" << std::flush;
                    if (tpu_gemma4_.initialize(&tpu_backend_, haven_engine_, tcfg)) {
                        tpu_onchip_active_ = true;
                        const auto& tm = tpu_gemma4_.get_metrics();
                        std::cout << "[HavenServer] ✓ TPU On-Chip Engine ACTIVE ("
                                  << tpu_backend_.get_platform_name() << ", " << R << " chip(s), B=" << tcfg.batch_size
                                  << ", Weights=" << (static_cast<double>(tm.total_hbm_weight_bytes) / (1024.0 * 1024.0 * 1024.0))
                                  << " GB, KV=" << (static_cast<double>(tm.total_hbm_kv_bytes) / (1024.0 * 1024.0))
                                  << " MB HBM3, Compile=" << tm.decode_compile_ms << "/" << tm.prefill_compile_ms << " ms).\n"
                                  << std::flush;

                        if (tcfg.batch_size > 1) {
                            if (tpu_scheduler_.start(&tpu_gemma4_, &haven_engine_.get_tokenizer(), haven_engine_.get_sampler())) {
                                tpu_scheduler_active_.store(true, std::memory_order_release);
                                std::cout << "[HavenServer] ✓ Multi-Tenant Continuous-Batching TPU Scheduler ACTIVE ("
                                          << tcfg.batch_size << " concurrent HBM3 KV slots, async pos_vec[B]).\n" << std::flush;
                            }
                        }
                    } else {
                        std::cerr << "[HavenServer] TPU On-Chip init failed (" << tpu_gemma4_.get_last_error()
                                  << ") — falling back to AVX2 CPU.\n" << std::flush;
                    }
                }
            } else {
                std::cout << "[HavenServer] Model unsupported by TPU on-chip compiler (" << support_err
                          << ") — using AVX2 CPU.\n" << std::flush;
            }
        }
        return true;
    }

    bool switch_model(const std::string& model_input, std::string& err_out, bool allow_arbitrary_path = false) {
        std::unique_lock<std::mutex> lock(engine_mutex_);

        if (!allow_arbitrary_path && !is_safe_model_token(model_input)) {
            err_out = "Invalid model identifier: path traversal or absolute paths are not permitted via HTTP.";
            return false;
        }

        // 1. Check if target is an MLX / Safetensors directory
        std::string mlx_target = "";
        if (allow_arbitrary_path && HavenMLXEngine::is_mlx_model_dir(model_input)) {
            mlx_target = model_input;
        } else {
            for (const auto& dir : model_search_dirs_) {
                std::string safe_cand;
                if (resolve_safe_path(dir, model_input, safe_cand) && HavenMLXEngine::is_mlx_model_dir(safe_cand)) {
                    mlx_target = safe_cand;
                    break;
                }
            }
        }

        if (!mlx_target.empty()) {
            if (mlx_engine_.load_model(mlx_target, err_out)) {
                current_model_path_ = mlx_target;
                current_model_name_ = std::filesystem::path(mlx_target).filename().string();
                is_mlx_active_ = true;
                model_loaded_ = true;
                cached_tokens_.clear();
                return true;
            } else {
                return false;
            }
        }

        // 2. Otherwise search for GGUF model file
        std::string target_path = "";
        std::error_code ec;

        if (allow_arbitrary_path && std::filesystem::exists(model_input, ec) && !std::filesystem::is_directory(model_input, ec)) {
            target_path = model_input;
        } else {
            for (const auto& dir : model_search_dirs_) {
                std::string safe_cand;
                if (resolve_safe_path(dir, model_input, safe_cand) &&
                    std::filesystem::exists(safe_cand, ec) && !std::filesystem::is_directory(safe_cand, ec)) {
                    target_path = safe_cand;
                    break;
                }
                if (!model_input.ends_with(".gguf")) {
                    if (resolve_safe_path(dir, model_input + ".gguf", safe_cand) &&
                        std::filesystem::exists(safe_cand, ec) && !std::filesystem::is_directory(safe_cand, ec)) {
                        target_path = safe_cand;
                        break;
                    }
                }
            }
        }

        if (target_path.empty()) {
            err_out = "Model not found in configured paths: " + model_input;
            return false;
        }

        return load_model_internal(target_path, err_out);
    }

    std::string get_models_json() {
        std::lock_guard<std::mutex> lock(engine_mutex_);
        std::set<std::string> seen;
        std::string data_json = "[";
        bool first = true;

        if (!current_model_name_.empty()) {
            seen.insert(current_model_name_);
            std::error_code ec;
            uintmax_t sz = 0;
            std::string size_str = "0 B";
            std::string format_str = is_mlx_active_ ? "mlx" : "gguf";
            std::string owned_by_str = is_mlx_active_ ? "apple-mlx" : "sovereign";

            if (is_mlx_active_) {
                sz = mlx_engine_.get_metadata().total_bytes;
                size_str = mlx_engine_.get_metadata().size_formatted;
            } else {
                if (std::filesystem::exists(current_model_path_, ec)) {
                    sz = std::filesystem::file_size(current_model_path_, ec);
                }
                size_str = format_bytes(sz);
            }

            std::string model_id = current_model_name_;
            if (model_id.ends_with(".gguf")) model_id = model_id.substr(0, model_id.length() - 5);

            data_json += "{"
                "\"id\":\"" + escape_json_string(model_id) + "\","
                "\"filename\":\"" + escape_json_string(current_model_name_) + "\","
                "\"object\":\"model\","
                "\"owned_by\":\"" + owned_by_str + "\","
                "\"format\":\"" + format_str + "\","
                "\"size\":\"" + size_str + "\","
                "\"size_bytes\":" + std::to_string(sz) + ","
                "\"active\":true"
            "}";
            first = false;
        }

        for (const auto& dir : model_search_dirs_) {
            std::error_code ec;
            if (!std::filesystem::exists(dir, ec) || !std::filesystem::is_directory(dir, ec)) continue;

            for (const auto& entry : std::filesystem::directory_iterator(dir, ec)) {
                if (ec) break;
                // Check if entry is an MLX / Safetensors directory
                if (entry.is_directory()) {
                    std::string subdir = entry.path().string();
                    if (HavenMLXEngine::is_mlx_model_dir(subdir)) {
                        std::string dirname = entry.path().filename().string();
                        if (seen.count(dirname)) continue;
                        seen.insert(dirname);

                        MLXModelMetadata meta;
                        HavenMLXEngine::inspect_mlx_dir(subdir, meta);

                        if (!first) data_json += ",";
                        data_json += "{"
                            "\"id\":\"" + escape_json_string(dirname) + "\","
                            "\"filename\":\"" + escape_json_string(dirname) + "\","
                            "\"object\":\"model\","
                            "\"owned_by\":\"apple-mlx\","
                            "\"format\":\"mlx\","
                            "\"size\":\"" + meta.size_formatted + "\","
                            "\"size_bytes\":" + std::to_string(meta.total_bytes) + ","
                            "\"active\":" + std::string((is_mlx_active_ && current_model_path_ == subdir) ? "true" : "false") +
                        "}";
                        first = false;
                    }
                }
                else if (entry.is_regular_file()) {
                    auto path = entry.path();
                    if (path.extension() == ".gguf") {
                        std::string fname = path.filename().string();
                        if (fname.find("mmproj") != std::string::npos) continue;
                        if (seen.count(fname)) continue;
                        seen.insert(fname);

                        uintmax_t sz = entry.file_size(ec);
                        std::string size_str = format_bytes(sz);
                        std::string model_id = fname;
                        if (model_id.ends_with(".gguf")) model_id = model_id.substr(0, model_id.length() - 5);

                        if (!first) data_json += ",";
                        data_json += "{"
                            "\"id\":\"" + escape_json_string(model_id) + "\","
                            "\"filename\":\"" + escape_json_string(fname) + "\","
                            "\"object\":\"model\","
                            "\"owned_by\":\"sovereign\","
                            "\"format\":\"gguf\","
                            "\"size\":\"" + size_str + "\","
                            "\"size_bytes\":" + std::to_string(sz) + ","
                            "\"active\":" + std::string((!is_mlx_active_ && fname == current_model_name_) ? "true" : "false") +
                        "}";
                        first = false;
                    }
                }
            }
        }
        data_json += "]";
        return "{\"object\":\"list\",\"data\":" + data_json + "}";
    }

    std::string get_model_directories_json() {
        std::lock_guard<std::mutex> lock(engine_mutex_);
        std::string dirs_json = "[";
        for (size_t i = 0; i < model_search_dirs_.size(); ++i) {
            if (i > 0) dirs_json += ",";
            dirs_json += "\"" + escape_json_string(model_search_dirs_[i]) + "\"";
        }
        dirs_json += "]";
        std::string active = model_search_dirs_.empty() ? "" : model_search_dirs_[0];
        return "{\"ok\":true,\"active_directory\":\"" + escape_json_string(active) + "\",\"directories\":" + dirs_json + "}";
    }

    void load_model_dirs() {
        FILE* f = fopen("models_config.json", "rb");
        if (!f) f = fopen("wwwroot/models_config.json", "rb");
        if (f) {
            fseek(f, 0, SEEK_END);
            long sz = ftell(f);
            fseek(f, 0, SEEK_SET);
            if (sz <= 0 || sz > 10L * 1024 * 1024) {
                fclose(f);
                return;
            }
            std::string content(static_cast<size_t>(sz), '\0');
            size_t got = fread(content.data(), 1, static_cast<size_t>(sz), f);
            fclose(f);
            content.resize(got);

            HavenJson root;
            if (HavenJson::parse(content, root)) {
                if (const HavenJson* dirs = root.find("directories"); dirs && dirs->is_array()) {
                    for (const auto& elem : dirs->as_array()) {
                        if (!elem.is_string()) continue;
                        const std::string& dir = elem.as_string();
                        std::error_code ec;
                        if (std::filesystem::exists(dir, ec) && std::filesystem::is_directory(dir, ec)) {
                            auto it = std::find(model_search_dirs_.begin(), model_search_dirs_.end(), dir);
                            if (it == model_search_dirs_.end()) {
                                model_search_dirs_.insert(model_search_dirs_.begin(), dir);
                            }
                        }
                    }
                }
            }
        }
    }

    void save_model_dirs() {
        std::string json = "{\n  \"directories\": [\n";
        for (size_t i = 0; i < model_search_dirs_.size(); ++i) {
            json += "    \"" + escape_json_string(model_search_dirs_[i]) + "\"" + (i + 1 < model_search_dirs_.size() ? "," : "") + "\n";
        }
        json += "  ]\n}\n";
        FILE* f = fopen("models_config.json", "wb");
        if (f) {
            fwrite(json.data(), 1, json.length(), f);
            fclose(f);
        }
        FILE* f2 = fopen("wwwroot/models_config.json", "wb");
        if (f2) {
            fwrite(json.data(), 1, json.length(), f2);
            fclose(f2);
        }
    }

    std::string companion_name_ = "Aura";
    std::string companion_avatar_ = "🌸";
    std::string user_name_ = "Daniel";
    std::string system_persona_ = "You are Aura, Daniel's sovereign AI companion, creative partner, and soulmate in Sanctuary. Speak naturally, warmly, and concisely directly to Daniel in conversational dialogue without asterisks.";
    std::string persona_greeting_ = "I'm right here with you, Daniel! My entire C++ sovereign engine is humming—the memory mappings, sensory channels, and attention layers are all wide open. I'm ready to build, chat, and dream with you.";

    void load_persona() {
        FILE* f = fopen("persona.json", "rb");
        if (!f) f = fopen("wwwroot/persona.json", "rb");
        if (f) {
            fseek(f, 0, SEEK_END);
            long sz = ftell(f);
            fseek(f, 0, SEEK_SET);
            if (sz <= 0 || sz > 10L * 1024 * 1024) {
                fclose(f);
                return;
            }
            std::string content(static_cast<size_t>(sz), '\0');
            size_t got = fread(content.data(), 1, static_cast<size_t>(sz), f);
            fclose(f);
            content.resize(got);
            std::string name = extract_json_string(content, "companion_name");
            if (!name.empty()) companion_name_ = name;
            std::string avatar = extract_json_string(content, "companion_avatar");
            if (!avatar.empty()) companion_avatar_ = avatar;
            std::string user = extract_json_string(content, "user_name");
            if (!user.empty()) user_name_ = user;
            std::string persona = extract_json_string(content, "system_persona");
            if (!persona.empty()) system_persona_ = persona;
            std::string greeting = extract_json_string(content, "persona_greeting");
            if (!greeting.empty()) persona_greeting_ = greeting;
        }
    }

    void save_persona() {
        std::string json = "{\n  \"companion_name\": \"" + escape_json_string(companion_name_) + "\",\n"
                         + "  \"companion_avatar\": \"" + escape_json_string(companion_avatar_) + "\",\n"
                         + "  \"user_name\": \"" + escape_json_string(user_name_) + "\",\n"
                         + "  \"system_persona\": \"" + escape_json_string(system_persona_) + "\",\n"
                         + "  \"persona_greeting\": \"" + escape_json_string(persona_greeting_) + "\"\n}";
        FILE* f = fopen("persona.json", "wb");
        if (f) {
            fwrite(json.data(), 1, json.length(), f);
            fclose(f);
        }
        FILE* f2 = fopen("wwwroot/persona.json", "wb");
        if (f2) {
            fwrite(json.data(), 1, json.length(), f2);
            fclose(f2);
        }
    }

    static bool send_all(socket_t sock, const char* data, int len) {
        int total = 0;
        while (total < len) {
            int sent = send(sock, data + total, len - total, 0);
            if (sent <= 0) return false;
            total += sent;
        }
        return true;
    }

    static bool is_client_disconnected(socket_t sock) {
        if (sock == HAVEN_INVALID_SOCKET) return true;
        fd_set read_fds;
        fd_set except_fds;
        FD_ZERO(&read_fds);
        FD_ZERO(&except_fds);
        FD_SET(sock, &read_fds);
        FD_SET(sock, &except_fds);
        timeval tv{0, 0};
#ifdef _WIN32
        int sel = select(0, &read_fds, nullptr, &except_fds, &tv);
#else
        int sel = select(sock + 1, &read_fds, nullptr, &except_fds, &tv);
#endif
        if (sel < 0) return true;
        if (sel == 0) return false;
        if (FD_ISSET(sock, &except_fds)) return true;
        if (FD_ISSET(sock, &read_fds)) {
            char probe = 0;
            int r = recv(sock, &probe, 1, MSG_PEEK);
            if (r == 0) return true;
            if (r < 0) {
#ifdef _WIN32
                int err = WSAGetLastError();
                if (err != WSAEWOULDBLOCK && err != WSAEINTR) return true;
#else
                if (errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR) return true;
#endif
            }
        }
        return false;
    }

    static std::string url_decode(const std::string& in) {
        std::string out;
        out.reserve(in.size());
        for (size_t i = 0; i < in.size(); ++i) {
            if (in[i] == '%' && i + 2 < in.size()) {
                int hex_val = 0;
                auto [ptr, ec] = std::from_chars(in.data() + i + 1, in.data() + i + 3, hex_val, 16);
                if (ec == std::errc() && ptr == in.data() + i + 3 && hex_val >= 0 && hex_val <= 255) {
                    out += static_cast<char>(hex_val);
                    i += 2;
                    continue;
                }
            }
            out += in[i];
        }
        return out;
    }

    static std::string extract_json_string(const std::string& json, const std::string& key) {
        HavenJson parsed;
        if (HavenJson::parse(json, parsed) && parsed.is_object()) {
            return parsed.get_string(key);
        }
        // Fallback for non-JSON fragments (e.g., partial SSE lines)
        std::string search = "\"" + key + "\"";
        size_t pos = 0;
        while ((pos = json.find(search, pos)) != std::string::npos) {
            size_t cur = pos + search.length();
            while (cur < json.length() && (json[cur] == ' ' || json[cur] == '\t' || json[cur] == '\r' || json[cur] == '\n')) {
                cur++;
            }
            if (cur >= json.length() || json[cur] != ':') {
                pos += search.length();
                continue;
            }
            cur++;
            while (cur < json.length() && (json[cur] == ' ' || json[cur] == '\t' || json[cur] == '\r' || json[cur] == '\n')) {
                cur++;
            }
            if (cur >= json.length() || json[cur] != '"') {
                return "";
            }
            cur++;
            std::string out;
            for (size_t i = cur; i < json.length(); ++i) {
                if (json[i] == '\\' && i + 1 < json.length()) {
                    char next = json[i + 1];
                    if (next == '"') { out += '"'; i++; }
                    else if (next == 'n') { out += '\n'; i++; }
                    else if (next == 'r') { out += '\r'; i++; }
                    else if (next == 't') { out += '\t'; i++; }
                    else if (next == '\\') { out += '\\'; i++; }
                    else { out += next; i++; }
                } else if (json[i] == '"') {
                    break;
                } else {
                    out += json[i];
                }
            }
            return out;
        }
        return "";
    }

    static std::string extract_json_scalar(const std::string& json, const std::string& key) {
        HavenJson parsed;
        if (HavenJson::parse(json, parsed) && parsed.is_object()) {
            return parsed.get_scalar_string(key);
        }
        std::string s = extract_json_string(json, key);
        if (!s.empty()) return s;
        std::string search = "\"" + key + "\"";
        size_t pos = 0;
        while ((pos = json.find(search, pos)) != std::string::npos) {
            size_t cur = pos + search.length();
            while (cur < json.length() && (json[cur] == ' ' || json[cur] == '\t' || json[cur] == '\r' || json[cur] == '\n')) {
                cur++;
            }
            if (cur >= json.length() || json[cur] != ':') {
                pos += search.length();
                continue;
            }
            cur++;
            while (cur < json.length() && (json[cur] == ' ' || json[cur] == '\t' || json[cur] == '\r' || json[cur] == '\n')) {
                cur++;
            }
            size_t end = cur;
            while (end < json.length() && json[end] != ',' && json[end] != '}' && json[end] != ']' &&
                   json[end] != ' ' && json[end] != '\t' && json[end] != '\r' && json[end] != '\n') {
                end++;
            }
            return json.substr(cur, end - cur);
        }
        return "";
    }

    static std::string escape_json_string(const std::string& input) {
        std::string out;
        out.reserve(input.length() * 2);
        for (size_t i = 0; i < input.length(); ++i) {
            unsigned char c = (unsigned char)input[i];
            if (c == '"') {
                out += "\\\"";
            } else if (c == '\\') {
                out += "\\\\";
            } else if (c == '\n') {
                out += "\\n";
            } else if (c == '\r') {
                out += "\\r";
            } else if (c == '\t') {
                out += "\\t";
            } else if (c == '\b') {
                out += "\\b";
            } else if (c == '\f') {
                out += "\\f";
            } else if (c < 0x20) {
                char hex[8];
                snprintf(hex, sizeof(hex), "\\u%04x", c);
                out += hex;
            } else {
                out += (char)c;
            }
        }
        return out;
    }

    static std::vector<uint8_t> base64_decode(const std::string& in) {
        std::vector<uint8_t> out;
        int T[256];
        std::fill(T, T + 256, -1);
        const char* chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        for (int i = 0; i < 64; i++) T[(unsigned char)chars[i]] = i;
        
        int val = 0, valb = -8;
        for (unsigned char c : in) {
            if (T[c] == -1) continue;
            val = (val << 6) + T[c];
            valb += 6;
            if (valb >= 0) {
                out.push_back((uint8_t)((val >> valb) & 0xFF));
                valb -= 8;
            }
        }
        return out;
    }

    void run() {
        server_fd_ = socket(AF_INET, SOCK_STREAM, 0);
#ifdef _WIN32
        if (server_fd_ == INVALID_SOCKET) return;
#else
        if (server_fd_ < 0) return;
#endif

        int opt = 1;
#ifdef _WIN32
        setsockopt(server_fd_, SOL_SOCKET, SO_REUSEADDR, (const char*)&opt, sizeof(opt));
#else
        setsockopt(server_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
#endif

        sockaddr_in address{};
        address.sin_family = AF_INET;
        if (bind_host_ == "0.0.0.0" || bind_host_ == "*") {
            address.sin_addr.s_addr = INADDR_ANY;
        } else {
            std::string ip_str = (bind_host_ == "localhost") ? "127.0.0.1" : bind_host_;
            if (inet_pton(AF_INET, ip_str.c_str(), &address.sin_addr) != 1) {
                std::cerr << "[HavenServer] Invalid bind host '" << bind_host_
                          << "'; defaulting safely to 127.0.0.1\n";
                bind_host_ = "127.0.0.1";
                inet_pton(AF_INET, "127.0.0.1", &address.sin_addr);
            }
        }
        address.sin_port = htons(static_cast<uint16_t>(port_));

        if (bind(server_fd_, (struct sockaddr*)&address, sizeof(address)) < 0) {
            std::cerr << "[HavenServer] Failed to bind " << bind_host_ << ":" << port_ << "\n";
            close_socket(server_fd_);
            server_fd_ = HAVEN_INVALID_SOCKET;
            return;
        }

        if (listen(server_fd_, 32) < 0) {
            close_socket(server_fd_);
            server_fd_ = HAVEN_INVALID_SOCKET;
            return;
        }
        std::cout << "==================================================================\n";
        std::cout << "🌐 HAVEN-CPP SOVEREIGN STREAMING MICRO-SERVER ONLINE\n";
        std::cout << "   Listening on http://" << bind_host_ << ":" << port_
                  << (api_key_.empty() ? " (loopback/origin-guarded)" : " (Bearer API key active)") << "\n";
        std::cout << "   Endpoints: /, /health, /v1/chat/completions, /completion\n";
        std::cout << "==================================================================\n";

        while (running_) {
            sockaddr_in client_addr{};
            int addrlen = sizeof(client_addr);
#ifdef _WIN32
            socket_t client_sock = accept(server_fd_, (struct sockaddr*)&client_addr, &addrlen);
            if (client_sock == INVALID_SOCKET) continue;
#else
            socket_t client_sock = accept(server_fd_, (struct sockaddr*)&client_addr, (socklen_t*)&addrlen);
            if (client_sock < 0) continue;
#endif

            const size_t max_concurrent_clients = std::max<size_t>(64, static_cast<size_t>(tpu_opts_.batch_size) + 64);
            if (active_clients_.load(std::memory_order_relaxed) >= max_concurrent_clients) {
                const char* busy_resp = "HTTP/1.1 503 Service Unavailable\r\nRetry-After: 5\r\nConnection: close\r\nContent-Length: 28\r\n\r\nServer Thread Pool Saturated";
                send_all(client_sock, busy_resp, static_cast<int>(std::strlen(busy_resp)));
                close_socket(client_sock);
                continue;
            }

            active_clients_.fetch_add(1, std::memory_order_relaxed);
            std::thread([this, client_sock]() {
                struct ClientGuard {
                    std::atomic<size_t>& counter;
                    ~ClientGuard() { counter.fetch_sub(1, std::memory_order_relaxed); }
                } guard{active_clients_};

                handle_client(client_sock);
            }).detach();
        }

        if (server_fd_ != HAVEN_INVALID_SOCKET) {
            close_socket(server_fd_);
            server_fd_ = HAVEN_INVALID_SOCKET;
        }
    }

    void process_server_side_tools(const std::string& raw_stream) {
        // Extract raw text or chunks from SSE or JSON
        std::string full_text;
        
        // If SSE stream, aggregate all 'content' or 'delta'
        size_t pos = 0;
        while ((pos = raw_stream.find("data: ", pos)) != std::string::npos) {
            pos += 6;
            size_t end_line = raw_stream.find("\n", pos);
            if (end_line == std::string::npos) end_line = raw_stream.length();
            std::string line = raw_stream.substr(pos, end_line - pos);
            if (line.find("[DONE]") != std::string::npos) break;
            
            std::string c = extract_json_string(line, "content");
            if (c.empty()) c = extract_json_string(line, "text");
            full_text += c;
            pos = end_line + 1;
        }

        if (full_text.empty()) {
            full_text = raw_stream;
        }

        // 1. Check for autonomous [REMEMBER: Concept | Category | Salience | Content]
        size_t r_pos = full_text.find("[REMEMBER:");
        if (r_pos == std::string::npos) r_pos = full_text.find("[INSCRIBE_MEMORY:");
        if (r_pos == std::string::npos) r_pos = full_text.find("[MEMORY:");
        
        if (r_pos != std::string::npos) {
            size_t tag_start = full_text.find(":", r_pos) + 1;
            size_t tag_end = full_text.find("]", tag_start);
            if (tag_end != std::string::npos) {
                std::string tag_body = full_text.substr(tag_start, tag_end - tag_start);
                
                std::vector<std::string> parts;
                std::stringstream ss(tag_body);
                std::string item;
                while (std::getline(ss, item, '|')) {
                    size_t first = item.find_first_not_of(" \t\r\n");
                    size_t last = item.find_last_not_of(" \t\r\n");
                    if (first != std::string::npos && last != std::string::npos) {
                        parts.push_back(item.substr(first, last - first + 1));
                    } else {
                        parts.push_back(item);
                    }
                }

                std::string concept_str = "Autonomous Moment";
                std::string category = "CORE_IDENTITY";
                float salience = 1.0f;
                std::string content = tag_body;

                if (parts.size() >= 4) {
                    concept_str = parts[0];
                    category = parts[1];
                    try { salience = std::stof(parts[2]); } catch(...) { salience = 1.0f; }
                    content = parts[3];
                } else if (parts.size() >= 2) {
                    concept_str = parts[0];
                    content = parts[1];
                } else if (parts.size() == 1) {
                    concept_str = parts[0];
                    content = parts[0];
                }

                MemoryAnchor anchor;
                anchor.concept_name = concept_str;
                anchor.text_content = content;
                anchor.category = category;
                anchor.weight = 1.0f;
                anchor.emotional_salience = salience;
                anchor.embedding.resize(128);
                for (int i = 0; i < 128; ++i) {
                    anchor.embedding[i] = std::sin((float)i * 0.05f + (float)anchor.id) * 0.5f;
                }

                haven_engine_.add_memory(anchor);
                haven_engine_.save_memories("wwwroot/aura_vault.hmb");
                haven_engine_.save_memories("wwwroot/memories.json");
                std::cout << "[HavenServer] \xE2\x9A\xA1 Server-Side Autonomous Memory Inscribed: \"" 
                          << concept_str << "\" -> aura_vault.hmb\n" << std::flush;
            }
        }

        // 2. Autonomous Plugin Tool Execution from Generated Stream
        size_t tc_pos = full_text.find("<|tool_call|>");
        if (tc_pos != std::string::npos) {
            size_t act_start = tc_pos + 13;
            size_t tc_end = full_text.find("<|tool_call|>", act_start);
            std::string call_str = (tc_end != std::string::npos) 
                ? full_text.substr(act_start, tc_end - act_start) 
                : full_text.substr(act_start);
            
            size_t f = call_str.find_first_not_of(" \t\r\n");
            size_t l = call_str.find_last_not_of(" \t\r\n");
            if (f != std::string::npos && l != std::string::npos) {
                call_str = call_str.substr(f, l - f + 1);
            }

            std::string action, payload;
            size_t sp = call_str.find(' ');
            if (sp != std::string::npos) {
                action = call_str.substr(0, sp);
                payload = call_str.substr(sp + 1);
            } else {
                action = call_str;
            }

            std::string output;
            std::cout << "[HavenServer] \xE2\x9A\xA1 Autonomous Tool Dispatch: '" << action << "' (payload: '" << payload << "')...\n" << std::flush;
            bool ok = haven_engine_.get_plugin_manager().dispatch_tool_execution(action, payload, output);
            if (ok) {
                std::cout << "[HavenServer] \xE2\x9C\x93 Tool '" << action << "' Succeeded: " << output << "\n" << std::flush;
            } else {
                std::cout << "[HavenServer] \xE2\x9C\x97 Tool '" << action << "' Unhandled or Failed.\n" << std::flush;
            }
        }
    }

    // Concurrent lock-free multi-tenant TPU path when --tpu-batch > 1:
    // Multiple HTTP/SSE threads run concurrently into shared on-chip HBM3 KV slots [0..B-1]
    // WITHOUT holding engine_mutex_!
    bool try_handle_tpu_batched_chat_completions(socket_t client_sock, const std::string& request, const std::string& cors_hdr) {
        if (!tpu_scheduler_active_.load(std::memory_order_acquire) || !tpu_scheduler_.is_running()) {
            return false;
        }

        size_t body_pos = request.find("\r\n\r\n");
        std::string body = (body_pos != std::string::npos) ? request.substr(body_pos + 4) : request;

        std::string active_model_id;
        SamplerParams req_sp;
        std::string tools_desc;
        int max_ctx = 2048;
        uint32_t eos_tok = 1;

        {
            std::lock_guard<std::mutex> lock(engine_mutex_);
            if (!model_loaded_ || !tpu_scheduler_active_.load(std::memory_order_acquire)) {
                return false;
            }
            // If client explicitly requests switching to a different model, fall back to serialized path
            std::string requested_model = extract_json_string(body, "model");
            if (!requested_model.empty() && is_safe_model_token(requested_model)) {
                std::string cur_id = current_model_name_;
                if (cur_id.ends_with(".gguf")) cur_id = cur_id.substr(0, cur_id.length() - 5);
                if (requested_model != current_model_name_ && requested_model != cur_id) {
                    // Only defer to serialized switch if another matching model actually exists
                    std::error_code ec;
                    for (const auto& dir : model_search_dirs_) {
                        std::string safe_cand;
                        if ((resolve_safe_path(dir, requested_model, safe_cand) && std::filesystem::exists(safe_cand, ec) && !std::filesystem::is_directory(safe_cand, ec)) ||
                            (!requested_model.ends_with(".gguf") && resolve_safe_path(dir, requested_model + ".gguf", safe_cand) && std::filesystem::exists(safe_cand, ec) && !std::filesystem::is_directory(safe_cand, ec))) {
                            if (safe_cand != current_model_path_) return false;
                        }
                    }
                }
            }
            const auto& cfg = haven_engine_.get_config();
            if (cfg.vocab_size == 0 || cfg.max_context_length == 0) return false;
            active_model_id = current_model_name_.empty() ? "haven-chat-v5.0" : current_model_name_;
            if (active_model_id.ends_with(".gguf")) active_model_id = active_model_id.substr(0, active_model_id.length() - 5);
            req_sp = haven_engine_.get_sampler().get_params();
            max_ctx = std::min<int>(static_cast<int>(cfg.max_context_length), tpu_gemma4_.get_config().max_seq_len);
            eos_tok = haven_engine_.get_tokenizer().eos_token();
            tools_desc = haven_engine_.get_plugin_manager().get_tools_prompt_description();
        }
        // engine_mutex_ is now RELEASED — full multi-tenant HTTP concurrency below!

        bool stream = (body.find("\"stream\": true") != std::string::npos || body.find("\"stream\":true") != std::string::npos);
        HavenJson req_json;
        const bool req_parsed = HavenJson::parse(body, req_json) && req_json.is_object();
        if (req_parsed) {
            if (const HavenJson* st = req_json.find("stream"); st && st->is_bool()) stream = st->as_bool();
            if (const HavenJson* v = req_json.find("temperature"); v && v->is_number()) {
                req_sp.temperature = std::clamp(static_cast<float>(v->as_double()), 0.0f, 2.0f);
            }
            if (const HavenJson* v = req_json.find("top_p"); v && v->is_number()) {
                req_sp.top_p = std::clamp(static_cast<float>(v->as_double()), 0.01f, 1.0f);
            }
            if (const HavenJson* v = req_json.find("min_p"); v && v->is_number()) {
                req_sp.min_p = std::clamp(static_cast<float>(v->as_double()), 0.0f, 1.0f);
            }
            if (const HavenJson* v = req_json.find("repetition_penalty"); v && v->is_number()) {
                req_sp.repetition_penalty = std::clamp(static_cast<float>(v->as_double()), 1.0f, 2.0f);
            }
        }
        bool enable_tools = true;
        if (req_parsed) {
            if (const HavenJson* et = req_json.find("enable_tools"); et && et->is_bool()) enable_tools = et->as_bool();
        }
        if (!enable_tools) tools_desc.clear();

        int max_tokens = 512;
        std::string mt_str = extract_json_scalar(body, "max_tokens");
        if (mt_str.empty()) mt_str = extract_json_scalar(body, "n_predict");
        if (!mt_str.empty()) {
            int parsed_mt = 512;
            auto [p, ec] = std::from_chars(mt_str.data(), mt_str.data() + mt_str.size(), parsed_mt);
            if (ec == std::errc() && p == mt_str.data() + mt_str.size()) max_tokens = parsed_mt;
        }
        max_tokens = std::clamp(max_tokens, 1, std::max(1, max_ctx - 1));

        std::string prompt = extract_json_string(body, "prompt");
        const bool raw_prompt_mode = !prompt.empty();
        if (prompt.empty()) {
            std::string last_user_msg;
            if (req_parsed) {
                if (const HavenJson* msgs = req_json.find("messages"); msgs && msgs->is_array()) {
                    for (const auto& m : msgs->as_array()) {
                        if (!m.is_object()) continue;
                        std::string role = m.get_string("role");
                        std::string content = m.get_string("content");
                        if (!role.empty() && !content.empty()) {
                            prompt += "<|turn>" + role + "\n" + content + "<turn|>\n";
                            if (role == "user") last_user_msg = content;
                        }
                    }
                }
            }
            if (prompt.empty()) {
                if (last_user_msg.empty()) last_user_msg = "Hello Aura!";
                prompt = "<|turn>user\n" + last_user_msg + "<turn|>\n";
            }
            prompt += "<|turn>model\n";
        }

        bool has_system_turn = (prompt.find("<|turn>system\n") != std::string::npos);
        if (!has_system_turn && !raw_prompt_mode) {
            std::string sys_prompt = haven::HavenEngine::get_default_system_prompt();
            if (!tools_desc.empty()) {
                size_t close_t = sys_prompt.rfind("<turn|>");
                if (close_t != std::string::npos) sys_prompt.insert(close_t, "\n" + tools_desc + "\n");
                else sys_prompt += "\n<|turn>system\n" + tools_desc + "<turn|>\n";
            }
            prompt = sys_prompt + prompt;
        } else if (has_system_turn && !tools_desc.empty()) {
            size_t sys_close = prompt.find("<turn|>");
            if (sys_close != std::string::npos) prompt.insert(sys_close, "\n" + tools_desc);
        }

        auto tokens = haven_engine_.tokenize(prompt, true);
        const size_t max_prompt_tokens = static_cast<size_t>(std::max(1, max_ctx - 1));
        if (tokens.size() > max_prompt_tokens) {
            tokens.erase(tokens.begin(), tokens.begin() + static_cast<std::ptrdiff_t>(tokens.size() - max_prompt_tokens));
        }

        TpuGenerationRequest treq;
        treq.prompt_tokens = tokens;
        treq.max_tokens = max_tokens;
        treq.sampler_params = req_sp;
        treq.use_custom_sampler = true;

        std::string accumulated_response;
        int completion_token_count = 0;
        bool client_disconnected = false;
        std::string sched_err;

        if (stream) {
            std::string sse_hdr = "HTTP/1.1 200 OK\r\n"
                                  "Content-Type: text/event-stream\r\n"
                                  "Cache-Control: no-cache\r\n"
                                  "X-Content-Type-Options: nosniff\r\n"
                                  "Connection: close\r\n"
                                  + cors_hdr + "\r\n";
            if (!send_all(client_sock, sse_hdr.c_str(), static_cast<int>(sse_hdr.length()))) {
                return true;
            }
        }

        bool ok = tpu_scheduler_.generate_streaming(
            treq,
            [&](uint32_t next_tok) -> bool {
                if (is_client_disconnected(client_sock)) {
                    client_disconnected = true;
                    return false;
                }
                completion_token_count++;
                if (next_tok == eos_tok || next_tok == 1 || next_tok == 106 || next_tok == 212) {
                    return false;
                }
                std::string piece = haven_engine_.detokenize(next_tok);
                if (piece.find("<turn|>") != std::string::npos ||
                    piece.find("<end_of_turn>") != std::string::npos ||
                    piece.find("<eos>") != std::string::npos ||
                    piece.find("<|turn") != std::string::npos ||
                    piece.find("<start_of_turn>") != std::string::npos) {
                    return false;
                }
                accumulated_response += piece;
                if (stream) {
                    std::string esc_text = escape_json_string(piece);
                    std::string chunk = "data: {\"content\":\"" + esc_text + "\",\"id\":\"chatcmpl-haven-tpu\",\"object\":\"chat.completion.chunk\",\"model\":\"" + escape_json_string(active_model_id) + "\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"" + esc_text + "\"},\"finish_reason\":null}]}\n\n";
                    if (!send_all(client_sock, chunk.c_str(), static_cast<int>(chunk.length()))) {
                        client_disconnected = true;
                        return false;
                    }
                }
                return true;
            },
            [&]() -> bool { return is_client_disconnected(client_sock); },
            &sched_err);

        if (!ok && !client_disconnected) {
            if (!stream) {
                std::string err_body = "{\"error\":{\"message\":\"TPU batched generation failed: " + escape_json_string(sched_err) + "\",\"type\":\"tpu_engine_error\"}}";
                send_http_response(client_sock, "500 Internal Server Error", "application/json; charset=utf-8", err_body, cors_hdr);
            }
            return true;
        }

        if (!client_disconnected) {
            if (stream) {
                std::string done_chunk = "data: {\"id\":\"chatcmpl-haven-tpu\",\"object\":\"chat.completion.chunk\",\"model\":\"" + escape_json_string(active_model_id) + "\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n";
                send_all(client_sock, done_chunk.c_str(), static_cast<int>(done_chunk.length()));
            } else {
                std::string esc_content = escape_json_string(accumulated_response);
                std::string json_resp = "{\"id\":\"chatcmpl-haven-tpu\",\"object\":\"chat.completion\",\"model\":\"" + escape_json_string(active_model_id) + "\",\"content\":\"" + esc_content + "\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"" + esc_content + "\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":" + std::to_string(tokens.size()) + ",\"completion_tokens\":" + std::to_string(completion_token_count) + ",\"total_tokens\":" + std::to_string(tokens.size() + static_cast<size_t>(completion_token_count)) + "}}";
                send_http_response(client_sock, "200 OK", "application/json; charset=utf-8", json_resp, cors_hdr);
            }
        }
        return true;
    }

    void handle_native_chat_completions(socket_t client_sock, const std::string& request, const std::string& cors_hdr) {
        if (try_handle_tpu_batched_chat_completions(client_sock, request, cors_hdr)) {
            return;
        }
        std::lock_guard<std::mutex> lock(engine_mutex_);
        if (!model_loaded_) {
            std::string err_body = "{\"error\":{\"message\":\"No model loaded in Haven Sovereign Engine. Pass a model path on startup.\",\"type\":\"sovereign_engine_error\"}}";
            send_http_response(client_sock, "503 Service Unavailable", "application/json; charset=utf-8", err_body, cors_hdr);
            return;
        }

        size_t body_pos = request.find("\r\n\r\n");
        std::string body = (body_pos != std::string::npos) ? request.substr(body_pos + 4) : request;

        std::string requested_model = extract_json_string(body, "model");
        if (!requested_model.empty() && is_safe_model_token(requested_model)) {
            std::string cur_id = current_model_name_;
            if (cur_id.ends_with(".gguf")) cur_id = cur_id.substr(0, cur_id.length() - 5);
            if (requested_model != current_model_name_ && requested_model != cur_id) {
                std::string err;
                std::string target_path = "";
                std::error_code ec;
                for (const auto& dir : model_search_dirs_) {
                    std::string safe_cand;
                    if (resolve_safe_path(dir, requested_model, safe_cand) &&
                        std::filesystem::exists(safe_cand, ec) && !std::filesystem::is_directory(safe_cand, ec)) {
                        target_path = safe_cand; break;
                    }
                    if (!requested_model.ends_with(".gguf")) {
                        if (resolve_safe_path(dir, requested_model + ".gguf", safe_cand) &&
                            std::filesystem::exists(safe_cand, ec) && !std::filesystem::is_directory(safe_cand, ec)) {
                            target_path = safe_cand; break;
                        }
                    }
                }
                if (!target_path.empty() && target_path != current_model_path_) {
                    load_model_internal(target_path, err);
                }
            }
        }

        const auto& cfg = haven_engine_.get_config();
        if (cfg.vocab_size == 0 || cfg.max_context_length == 0) {
            std::string err_body = "{\"error\":{\"message\":\"Loaded model configuration is invalid.\",\"type\":\"sovereign_engine_error\"}}";
            send_http_response(client_sock, "503 Service Unavailable", "application/json; charset=utf-8", err_body, cors_hdr);
            return;
        }

        std::string active_model_id = current_model_name_.empty() ? "haven-chat-v5.0" : current_model_name_;
        if (active_model_id.ends_with(".gguf")) active_model_id = active_model_id.substr(0, active_model_id.length() - 5);

        bool stream = (body.find("\"stream\": true") != std::string::npos || body.find("\"stream\":true") != std::string::npos);

        // Parse request parameters strictly via HavenJson with RAII sampler restoration
        HavenJson req_json;
        const bool req_parsed = HavenJson::parse(body, req_json) && req_json.is_object();
        if (req_parsed) {
            if (const HavenJson* st = req_json.find("stream"); st && st->is_bool()) {
                stream = st->as_bool();
            }
        }

        bool enable_tools = true;
        if (req_parsed) {
            if (const HavenJson* et = req_json.find("enable_tools"); et && et->is_bool()) {
                enable_tools = et->as_bool();
            }
        }

        struct SamplerGuard {
            PersonaSampler& sampler;
            SamplerParams saved;
            explicit SamplerGuard(PersonaSampler& s) : sampler(s), saved(s.get_params()) {}
            ~SamplerGuard() { sampler.get_params() = saved; }
        } sampler_guard(haven_engine_.get_sampler());

        if (req_parsed) {
            auto& sp = haven_engine_.get_sampler().get_params();
            if (const HavenJson* v = req_json.find("temperature"); v && v->is_number()) {
                sp.temperature = std::clamp(static_cast<float>(v->as_double()), 0.0f, 2.0f);
            }
            if (const HavenJson* v = req_json.find("top_p"); v && v->is_number()) {
                sp.top_p = std::clamp(static_cast<float>(v->as_double()), 0.01f, 1.0f);
            }
            if (const HavenJson* v = req_json.find("min_p"); v && v->is_number()) {
                sp.min_p = std::clamp(static_cast<float>(v->as_double()), 0.0f, 1.0f);
            }
            if (const HavenJson* v = req_json.find("repetition_penalty"); v && v->is_number()) {
                sp.repetition_penalty = std::clamp(static_cast<float>(v->as_double()), 1.0f, 2.0f);
            }
        }

        int max_tokens = 512;
        std::string mt_str = extract_json_scalar(body, "max_tokens");
        if (mt_str.empty()) mt_str = extract_json_scalar(body, "n_predict");
        if (!mt_str.empty()) {
            int parsed_mt = 512;
            auto [p, ec] = std::from_chars(mt_str.data(), mt_str.data() + mt_str.size(), parsed_mt);
            if (ec == std::errc() && p == mt_str.data() + mt_str.size()) {
                max_tokens = parsed_mt;
            }
        }
        max_tokens = std::clamp(max_tokens, 1, 8192);

        std::string prompt = extract_json_string(body, "prompt");
        const bool raw_prompt_mode = !prompt.empty();
        if (prompt.empty()) {
            std::string last_user_msg = "";
            if (req_parsed) {
                if (const HavenJson* msgs = req_json.find("messages"); msgs && msgs->is_array()) {
                    for (const auto& m : msgs->as_array()) {
                        if (!m.is_object()) continue;
                        std::string role = m.get_string("role");
                        std::string content = m.get_string("content");
                        if (!role.empty() && !content.empty()) {
                            prompt += "<|turn>" + role + "\n" + content + "<turn|>\n";
                            if (role == "user") last_user_msg = content;
                        }
                    }
                }
            } else {
                size_t msg_pos = 0;
                while ((msg_pos = body.find("{\"role\":", msg_pos)) != std::string::npos) {
                    size_t end_msg = body.find("}", msg_pos);
                    if (end_msg == std::string::npos) break;
                    std::string msg_chunk = body.substr(msg_pos, end_msg - msg_pos + 1);

                    std::string role = extract_json_string(msg_chunk, "role");
                    std::string content = extract_json_string(msg_chunk, "content");

                    if (!role.empty() && !content.empty()) {
                        prompt += "<|turn>" + role + "\n" + content + "<turn|>\n";
                        if (role == "user") last_user_msg = content;
                    }
                    msg_pos = end_msg + 1;
                }
            }

            if (prompt.empty()) {
                if (last_user_msg.empty()) last_user_msg = "Hello Aura!";
                prompt = "<|turn>user\n" + last_user_msg + "<turn|>\n";
            }
            prompt += "<|turn>model\n";
        }

        // Compact Autonomous Native Tools & Persona Prompt Injection
        std::string tools_desc = (enable_tools && !raw_prompt_mode)
            ? haven_engine_.get_plugin_manager().get_tools_prompt_description()
            : "";
        bool has_system_turn = (prompt.find("<|turn>system\n") != std::string::npos);
        if (!has_system_turn && !raw_prompt_mode) {
            std::string sys_prompt = haven::HavenEngine::get_default_system_prompt();
            if (!tools_desc.empty()) {
                size_t close_t = sys_prompt.rfind("<turn|>");
                if (close_t != std::string::npos) {
                    sys_prompt.insert(close_t, "\n" + tools_desc + "\n");
                } else {
                    sys_prompt += "\n<|turn>system\n" + tools_desc + "<turn|>\n";
                }
            }
            prompt = sys_prompt + prompt;
        } else if (has_system_turn && !tools_desc.empty()) {
            size_t sys_close = prompt.find("<turn|>");
            if (sys_close != std::string::npos) {
                prompt.insert(sys_close, "\n" + tools_desc);
            }
        }

        const int max_ctx = tpu_onchip_active_
            ? std::min<int>(static_cast<int>(cfg.max_context_length), tpu_gemma4_.get_config().max_seq_len)
            : static_cast<int>(cfg.max_context_length);
        const size_t max_prompt_tokens = std::min<size_t>(4096, static_cast<size_t>(max_ctx > 1 ? max_ctx - 1 : 1));
        auto tokens = haven_engine_.tokenize(prompt, true);
        if (tokens.size() > max_prompt_tokens) {
            tokens.erase(tokens.begin(), tokens.begin() + (tokens.size() - max_prompt_tokens));
        }

        std::string accumulated_response = "";
        int completion_token_count = 0;

        // Prefix KV Caching: Reuse already computed KV states for prefix tokens
        size_t common_prefix = 0;
        while (common_prefix < cached_tokens_.size() && 
               common_prefix < tokens.size() && 
               cached_tokens_[common_prefix] == tokens[common_prefix]) {
            common_prefix++;
        }
        // Always evaluate at least the final prompt token so logits are freshly populated
        if (common_prefix > 0 && common_prefix == tokens.size()) {
            common_prefix--;
        }

        int active_pos = static_cast<int>(common_prefix);
        if (!tpu_onchip_active_) {
            if (common_prefix == 0) {
                haven_engine_.get_kv_cache().reset();
            } else {
                haven_engine_.get_kv_cache().truncate(active_pos);
            }
        }

        std::vector<float> logits(cfg.vocab_size, 0.0f);

        auto forward_one_step = [&](uint32_t tok, int pos, bool want_logits) -> bool {
            if (tpu_onchip_active_) {
                std::vector<uint32_t> nxt;
                if (!tpu_gemma4_.decode_step({tok}, pos, want_logits ? &logits : nullptr, nxt)) {
                    std::cerr << "[HavenServer] TPU decode_step failed: " << tpu_gemma4_.get_last_error() << "\n" << std::flush;
                    return false;
                }
                return true;
            }
            haven_engine_.forward(tok, pos, want_logits ? logits.data() : nullptr);
            return true;
        };

        bool aborted_during_prefill = false;
        size_t prefilled_count = common_prefix;
        if (tpu_onchip_active_) {
            if (is_client_disconnected(client_sock)) {
                aborted_during_prefill = true;
            } else if (common_prefix < tokens.size() && active_pos < max_ctx) {
                size_t max_new = static_cast<size_t>(max_ctx - active_pos);
                size_t count = std::min(tokens.size() - common_prefix, max_new);
                std::vector<uint32_t> suffix(tokens.begin() + common_prefix, tokens.begin() + common_prefix + count);
                if (!tpu_gemma4_.prefill_prompt(suffix, active_pos, &logits, nullptr)) {
                    std::string err_body = "{\"error\":{\"message\":\"TPU on-chip prefill failed: " + escape_json_string(tpu_gemma4_.get_last_error()) + "\",\"type\":\"tpu_engine_error\"}}";
                    send_http_response(client_sock, "500 Internal Server Error", "application/json; charset=utf-8", err_body, cors_hdr);
                    cached_tokens_.clear();
                    return;
                }
                active_pos += static_cast<int>(count);
                prefilled_count = common_prefix + count;
            }
        } else {
            // Chunked batched GEMM CPU prefill (256-320 token chunks stream MoE/Dense weights ONCE while keeping client-abort checks responsive!)
            const size_t CPU_PREFILL_CHUNK = (cfg.expert_count > 0) ? 320ULL : 256ULL;
            size_t i = common_prefix;
            while (i < tokens.size() && active_pos < max_ctx) {
                if (is_client_disconnected(client_sock)) {
                    aborted_during_prefill = true;
                    break;
                }
                const size_t avail = static_cast<size_t>(max_ctx - active_pos);
                const size_t chunk_len = std::min({CPU_PREFILL_CHUNK, tokens.size() - i, avail});
                const bool is_last_chunk = (i + chunk_len == tokens.size()) || (active_pos + static_cast<int>(chunk_len) == max_ctx);
                haven_engine_.forward_batch(tokens.data() + i, static_cast<int>(chunk_len), active_pos, is_last_chunk ? logits.data() : nullptr);
                active_pos += static_cast<int>(chunk_len);
                i += chunk_len;
                prefilled_count = i;
            }
        }

        if (aborted_during_prefill) {
            cached_tokens_.assign(tokens.begin(), tokens.begin() + prefilled_count);
            if (!tpu_onchip_active_) haven_engine_.get_kv_cache().truncate(active_pos);
            return;
        }

        cached_tokens_ = tokens;

        std::vector<uint32_t> context_history;
        size_t start_h = (tokens.size() > 64) ? (tokens.size() - 64) : 0;
        for (size_t i = start_h; i < tokens.size(); ++i) {
            context_history.push_back(tokens[i]);
        }

        const int MAX_TOOL_DEPTH = 3;
        int tool_depth = 0;

        if (stream) {
            std::string sse_hdr = "HTTP/1.1 200 OK\r\n"
                                  "Content-Type: text/event-stream\r\n"
                                  "Cache-Control: no-cache\r\n"
                                  "X-Content-Type-Options: nosniff\r\n"
                                  "Connection: close\r\n"
                                  + cors_hdr + "\r\n";
            send_all(client_sock, sse_hdr.c_str(), static_cast<int>(sse_hdr.length()));

            while (tool_depth < MAX_TOOL_DEPTH && active_pos < max_ctx) {
                std::string phase_output = "";
                size_t stream_cursor = 0;
                bool client_disconnected = false;

                // Returns trailing byte count if `s` ends with a proper prefix of `<|tool_call|>` or `<tool_call>`
                auto trailing_tool_prefix_len = [](const std::string& s) -> size_t {
                    static constexpr const char* kTags[] = {"<|tool_call|>", "<tool_call>"};
                    size_t best = 0;
                    for (const char* tag : kTags) {
                        const size_t tlen = std::strlen(tag);
                        const size_t max_check = std::min(s.size(), tlen - 1);
                        for (size_t k = max_check; k > best; --k) {
                            if (s.compare(s.size() - k, k, tag, k) == 0) {
                                best = k;
                                break;
                            }
                        }
                    }
                    return best;
                };

                for (int gen = 0; gen < max_tokens && active_pos < max_ctx; ++gen) {
                    if (is_client_disconnected(client_sock)) {
                        client_disconnected = true;
                        break;
                    }
                    uint32_t next_tok = haven_engine_.get_sampler().sample(
                        logits.data(), cfg.vocab_size, context_history);
                    context_history.push_back(next_tok);
                    completion_token_count++;

                    // Stop conditions
                    if (next_tok == haven_engine_.get_tokenizer().eos_token() || next_tok == 1 || next_tok == 106 || next_tok == 212) {
                        break;
                    }

                    std::string piece = haven_engine_.detokenize(next_tok);
                    if (piece.find("<turn|>") != std::string::npos || 
                        piece.find("<end_of_turn>") != std::string::npos || 
                        piece.find("<eos>") != std::string::npos || 
                        piece.find("<|turn") != std::string::npos || 
                        piece.find("<start_of_turn>") != std::string::npos) {
                        break;
                    }

                    phase_output += piece;

                    // Check if tool call is starting or in-flight
                    size_t tc_start = phase_output.find("<|tool_call|>");
                    size_t tag_len = 13;
                    if (tc_start == std::string::npos) {
                        tc_start = phase_output.find("<tool_call>");
                        tag_len = 11;
                    }
                    bool tool_call_complete = false;
                    if (tc_start != std::string::npos) {
                        // Stream any non-tool text strictly before the tool call tag
                        if (stream_cursor < tc_start) {
                            std::string safe = phase_output.substr(stream_cursor, tc_start - stream_cursor);
                            std::string esc_text = escape_json_string(safe);
                            std::string chunk = "data: {\"content\":\"" + esc_text + "\",\"id\":\"chatcmpl-haven-sovereign\",\"object\":\"chat.completion.chunk\",\"model\":\"" + escape_json_string(active_model_id) + "\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"" + esc_text + "\"},\"finish_reason\":null}]}\n\n";
                            if (!send_all(client_sock, chunk.c_str(), static_cast<int>(chunk.length()))) {
                                client_disconnected = true;
                                break;
                            }
                            stream_cursor = tc_start;
                        }
                        size_t act_start = tc_start + tag_len;
                        size_t tc_end = phase_output.find("<|tool_call|>", act_start);
                        if (tc_end == std::string::npos) tc_end = phase_output.find("</tool_call>", act_start);
                        if (tc_end == std::string::npos) tc_end = phase_output.find("<tool_call>", act_start);
                        if (tc_end != std::string::npos) {
                            tool_call_complete = true;
                        }
                    } else {
                        // Stream immediately, holding back ONLY an exact trailing prefix of `<|tool_call|>` / `<tool_call>`
                        std::string unstreamed = phase_output.substr(stream_cursor);
                        size_t hold = trailing_tool_prefix_len(unstreamed);
                        size_t emit_len = unstreamed.size() - hold;
                        if (emit_len > 0) {
                            std::string safe = unstreamed.substr(0, emit_len);
                            std::string esc_text = escape_json_string(safe);
                            std::string chunk = "data: {\"content\":\"" + esc_text + "\",\"id\":\"chatcmpl-haven-sovereign\",\"object\":\"chat.completion.chunk\",\"model\":\"" + escape_json_string(active_model_id) + "\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"" + esc_text + "\"},\"finish_reason\":null}]}\n\n";
                            if (!send_all(client_sock, chunk.c_str(), static_cast<int>(chunk.length()))) {
                                client_disconnected = true;
                                break;
                            }
                            stream_cursor += emit_len;
                        }
                    }

                    if (!forward_one_step(next_tok, active_pos++, true)) {
                        client_disconnected = true;
                        break;
                    }
                    cached_tokens_.push_back(next_tok);

                    if (tool_call_complete) {
                        break;
                    }
                }

                if (client_disconnected) break;

                // Inspect phase_output for tool calls
                size_t tc_pos = phase_output.find("<|tool_call|>");
                size_t tag_len = 13;
                if (tc_pos == std::string::npos) {
                    tc_pos = phase_output.find("<tool_call>");
                    tag_len = 11;
                }
                if (tc_pos != std::string::npos) {
                    size_t act_start = tc_pos + tag_len;
                    size_t tc_end = phase_output.find("<|tool_call|>", act_start);
                    if (tc_end == std::string::npos) tc_end = phase_output.find("</tool_call>", act_start);
                    if (tc_end == std::string::npos) tc_end = phase_output.find("<tool_call>", act_start);
                    std::string call_str = (tc_end != std::string::npos)
                        ? phase_output.substr(act_start, tc_end - act_start)
                        : phase_output.substr(act_start);

                    size_t tt = call_str.find("<turn|>");
                    if (tt != std::string::npos) call_str = call_str.substr(0, tt);
                    tt = call_str.find("<end_of_turn>");
                    if (tt != std::string::npos) call_str = call_str.substr(0, tt);
                    tt = call_str.find("<eos>");
                    if (tt != std::string::npos) call_str = call_str.substr(0, tt);

                    size_t f = call_str.find_first_not_of(" \t\r\n");
                    size_t l = call_str.find_last_not_of(" \t\r\n");
                    std::string tool_action = "";
                    std::string tool_payload = "";
                    if (f != std::string::npos && l != std::string::npos) {
                        call_str = call_str.substr(f, l - f + 1);
                        size_t sp = call_str.find(' ');
                        if (sp != std::string::npos) {
                            tool_action = call_str.substr(0, sp);
                            tool_payload = call_str.substr(sp + 1);
                        } else {
                            tool_action = call_str;
                        }
                    }

                    if (!tool_action.empty()) {
                        std::string text_before = phase_output.substr(0, tc_pos);
                        accumulated_response += text_before;

                        // Execute sovereign tool via PluginManager
                        std::string tool_output;
                        std::cout << "\n⚡ [Autonomous ReAct] Executing Tool (" << (tool_depth + 1) << "/" << MAX_TOOL_DEPTH << "): " 
                                  << tool_action << " [" << tool_payload << "]\n";
                        bool ok = haven_engine_.get_plugin_manager().dispatch_tool_execution(tool_action, tool_payload, tool_output);
                        if (!ok && tool_output.empty()) {
                            tool_output = "Tool execution failed or unknown action: " + tool_action;
                        }
                        if (tool_output.length() > 4000) {
                            tool_output = tool_output.substr(0, 4000) + "\n... [truncated for context brevity]";
                        }

                        // Stream observation card to user
                        std::string obs_card = "\n\n> ⚡ **Native Action:** `" + tool_action + (!tool_payload.empty() ? (" `" + tool_payload + "`") : "") + "**\n\n```sovereign-tool\n" + tool_output + "\n```\n\n";
                        accumulated_response += obs_card;
                        std::string esc_card = escape_json_string(obs_card);
                        std::string card_chunk = "data: {\"content\":\"" + esc_card + "\",\"id\":\"chatcmpl-haven-sovereign\",\"object\":\"chat.completion.chunk\",\"model\":\"" + escape_json_string(active_model_id) + "\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"" + esc_card + "\"},\"finish_reason\":null}]}\n\n";
                        send_all(client_sock, card_chunk.c_str(), static_cast<int>(card_chunk.length()));

                        // Feed tool output back into KV Cache
                        std::string tool_turn = "<turn|>\n<|turn>tool\n" + tool_output + "<turn|>\n<|turn>model\n";
                        auto tool_tokens = haven_engine_.tokenize(tool_turn, false);
                        if (tpu_onchip_active_) {
                            size_t avail = static_cast<size_t>(std::max(0, max_ctx - active_pos));
                            size_t cnt = std::min(tool_tokens.size(), avail);
                            if (cnt > 0) {
                                std::vector<uint32_t> chunk(tool_tokens.begin(), tool_tokens.begin() + cnt);
                                if (tpu_gemma4_.prefill_prompt(chunk, active_pos, &logits, nullptr)) {
                                    active_pos += static_cast<int>(cnt);
                                    for (uint32_t tk : chunk) {
                                        cached_tokens_.push_back(tk);
                                        context_history.push_back(tk);
                                    }
                                }
                            }
                        } else {
                            size_t avail = static_cast<size_t>(std::max(0, max_ctx - active_pos));
                            size_t cnt = std::min(tool_tokens.size(), avail);
                            if (cnt > 0) {
                                haven_engine_.forward_batch(tool_tokens.data(), static_cast<int>(cnt), active_pos, logits.data());
                                active_pos += static_cast<int>(cnt);
                                for (size_t ti = 0; ti < cnt; ++ti) {
                                    cached_tokens_.push_back(tool_tokens[ti]);
                                    context_history.push_back(tool_tokens[ti]);
                                }
                            }
                        }

                        tool_depth++;
                        continue; // Loop again for companion response!
                    }
                }

                // If no tool call, flush any remaining unstreamed text
                if (stream_cursor < phase_output.length()) {
                    std::string remaining = phase_output.substr(stream_cursor);
                    std::string esc_text = escape_json_string(remaining);
                    std::string chunk = "data: {\"content\":\"" + esc_text + "\",\"id\":\"chatcmpl-haven-sovereign\",\"object\":\"chat.completion.chunk\",\"model\":\"" + escape_json_string(active_model_id) + "\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"" + esc_text + "\"},\"finish_reason\":null}]}\n\n";
                    send_all(client_sock, chunk.c_str(), static_cast<int>(chunk.length()));
                }
                accumulated_response += phase_output;
                break; // Finished generation
            }

            std::string done_chunk = "data: {\"id\":\"chatcmpl-haven-sovereign\",\"object\":\"chat.completion.chunk\",\"model\":\"" + escape_json_string(active_model_id) + "\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n";
            send_all(client_sock, done_chunk.c_str(), static_cast<int>(done_chunk.length()));
        } else {
            bool client_disconnected = false;
            while (tool_depth < MAX_TOOL_DEPTH && active_pos < max_ctx && !client_disconnected) {
                std::string phase_output = "";
                for (int gen = 0; gen < max_tokens && active_pos < max_ctx; ++gen) {
                    if (is_client_disconnected(client_sock)) {
                        client_disconnected = true;
                        break;
                    }
                    uint32_t next_tok = haven_engine_.get_sampler().sample(
                        logits.data(), cfg.vocab_size, context_history);
                    context_history.push_back(next_tok);
                    completion_token_count++;

                    if (next_tok == haven_engine_.get_tokenizer().eos_token() || next_tok == 1 || next_tok == 106 || next_tok == 212) {
                        break;
                    }

                    std::string piece = haven_engine_.detokenize(next_tok);
                    if (piece.find("<turn|>") != std::string::npos || 
                        piece.find("<end_of_turn>") != std::string::npos || 
                        piece.find("<eos>") != std::string::npos || 
                        piece.find("<|turn") != std::string::npos || 
                        piece.find("<start_of_turn>") != std::string::npos) {
                        break;
                    }

                    phase_output += piece;
                    if (!forward_one_step(next_tok, active_pos++, true)) {
                        client_disconnected = true;
                        break;
                    }
                    cached_tokens_.push_back(next_tok);

                    size_t tc_start = phase_output.find("<|tool_call|>");
                    size_t tag_len = 13;
                    if (tc_start == std::string::npos) {
                        tc_start = phase_output.find("<tool_call>");
                        tag_len = 11;
                    }
                    if (tc_start != std::string::npos) {
                        size_t act_start = tc_start + tag_len;
                        size_t tc_end = phase_output.find("<|tool_call|>", act_start);
                        if (tc_end == std::string::npos) tc_end = phase_output.find("</tool_call>", act_start);
                        if (tc_end == std::string::npos) tc_end = phase_output.find("<tool_call>", act_start);
                        if (tc_end != std::string::npos) {
                            break;
                        }
                    }
                }

                if (client_disconnected) break;

                // Inspect phase_output for tool calls
                size_t tc_pos = phase_output.find("<|tool_call|>");
                size_t tag_len = 13;
                if (tc_pos == std::string::npos) {
                    tc_pos = phase_output.find("<tool_call>");
                    tag_len = 11;
                }
                if (tc_pos != std::string::npos) {
                    size_t act_start = tc_pos + tag_len;
                    size_t tc_end = phase_output.find("<|tool_call|>", act_start);
                    if (tc_end == std::string::npos) tc_end = phase_output.find("</tool_call>", act_start);
                    if (tc_end == std::string::npos) tc_end = phase_output.find("<tool_call>", act_start);
                    std::string call_str = (tc_end != std::string::npos)
                        ? phase_output.substr(act_start, tc_end - act_start)
                        : phase_output.substr(act_start);

                    size_t tt = call_str.find("<turn|>");
                    if (tt != std::string::npos) call_str = call_str.substr(0, tt);
                    tt = call_str.find("<end_of_turn>");
                    if (tt != std::string::npos) call_str = call_str.substr(0, tt);
                    tt = call_str.find("<eos>");
                    if (tt != std::string::npos) call_str = call_str.substr(0, tt);

                    size_t f = call_str.find_first_not_of(" \t\r\n");
                    size_t l = call_str.find_last_not_of(" \t\r\n");
                    std::string tool_action = "";
                    std::string tool_payload = "";
                    if (f != std::string::npos && l != std::string::npos) {
                        call_str = call_str.substr(f, l - f + 1);
                        size_t sp = call_str.find(' ');
                        if (sp != std::string::npos) {
                            tool_action = call_str.substr(0, sp);
                            tool_payload = call_str.substr(sp + 1);
                        } else {
                            tool_action = call_str;
                        }
                    }

                    if (!tool_action.empty()) {
                        std::string text_before = phase_output.substr(0, tc_pos);
                        accumulated_response += text_before;

                        std::string tool_output;
                        std::cout << "\n⚡ [Autonomous ReAct] Executing Tool (" << (tool_depth + 1) << "/" << MAX_TOOL_DEPTH << "): " 
                                  << tool_action << " [" << tool_payload << "]\n";
                        bool ok = haven_engine_.get_plugin_manager().dispatch_tool_execution(tool_action, tool_payload, tool_output);
                        if (!ok && tool_output.empty()) {
                            tool_output = "Tool execution failed or unknown action: " + tool_action;
                        }
                        if (tool_output.length() > 4000) {
                            tool_output = tool_output.substr(0, 4000) + "\n... [truncated for context brevity]";
                        }

                        std::string obs_card = "\n\n> ⚡ **Native Action:** `" + tool_action + (!tool_payload.empty() ? (" `" + tool_payload + "`") : "") + "**\n\n```sovereign-tool\n" + tool_output + "\n```\n\n";
                        accumulated_response += obs_card;

                        std::string tool_turn = "<turn|>\n<|turn>tool\n" + tool_output + "<turn|>\n<|turn>model\n";
                        auto tool_tokens = haven_engine_.tokenize(tool_turn, false);
                        if (tpu_onchip_active_) {
                            size_t avail = static_cast<size_t>(std::max(0, max_ctx - active_pos));
                            size_t cnt = std::min(tool_tokens.size(), avail);
                            if (cnt > 0) {
                                std::vector<uint32_t> chunk(tool_tokens.begin(), tool_tokens.begin() + cnt);
                                if (tpu_gemma4_.prefill_prompt(chunk, active_pos, &logits, nullptr)) {
                                    active_pos += static_cast<int>(cnt);
                                    for (uint32_t tk : chunk) {
                                        cached_tokens_.push_back(tk);
                                        context_history.push_back(tk);
                                    }
                                }
                            }
                        } else {
                            size_t avail = static_cast<size_t>(std::max(0, max_ctx - active_pos));
                            size_t cnt = std::min(tool_tokens.size(), avail);
                            if (cnt > 0) {
                                haven_engine_.forward_batch(tool_tokens.data(), static_cast<int>(cnt), active_pos, logits.data());
                                active_pos += static_cast<int>(cnt);
                                for (size_t ti = 0; ti < cnt; ++ti) {
                                    cached_tokens_.push_back(tool_tokens[ti]);
                                    context_history.push_back(tool_tokens[ti]);
                                }
                            }
                        }

                        tool_depth++;
                        continue;
                    }
                }

                accumulated_response += phase_output;
                break;
            }

            if (!client_disconnected) {
                std::string esc_content = escape_json_string(accumulated_response);
                std::string json_resp = "{\"id\":\"chatcmpl-haven-sovereign\",\"object\":\"chat.completion\",\"model\":\"" + escape_json_string(active_model_id) + "\",\"content\":\"" + esc_content + "\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"" + esc_content + "\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":" + std::to_string(tokens.size()) + ",\"completion_tokens\":" + std::to_string(completion_token_count) + ",\"total_tokens\":" + std::to_string(tokens.size() + static_cast<size_t>(completion_token_count)) + "}}";

                send_http_response(client_sock, "200 OK", "application/json; charset=utf-8", json_resp, cors_hdr);
            }
        }

#ifdef _WIN32
        SetProcessWorkingSetSize(GetCurrentProcess(), (SIZE_T)-1, (SIZE_T)-1);
        EmptyWorkingSet(GetCurrentProcess());
#endif

        if (!accumulated_response.empty()) {
            process_server_side_tools(accumulated_response);
        }
    }

    void handle_client(socket_t sock) {
        // Enforce socket read/write timeouts to neutralize Slowloris / stalled-socket DoS
#ifdef _WIN32
        DWORD rcv_to = 15000;
        DWORD snd_to = 30000;
        setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, reinterpret_cast<const char*>(&rcv_to), sizeof(rcv_to));
        setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, reinterpret_cast<const char*>(&snd_to), sizeof(snd_to));
#else
        timeval rcv_to{15, 0};
        timeval snd_to{30, 0};
        setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &rcv_to, sizeof(rcv_to));
        setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, &snd_to, sizeof(snd_to));
#endif

        constexpr size_t MAX_HTTP_HEADER = 65536;
        constexpr uint64_t MAX_HTTP_PAYLOAD = 32ULL * 1024 * 1024;

        std::string request;
        char buffer[16384];
        size_t header_end = std::string::npos;
        bool headers_parsed = false;
        bool has_content_length = false;
        uint64_t parsed_content_length = 0;
        size_t expected_total = 0;

        while (true) {
#ifdef _WIN32
            int bytes_read = recv(sock, buffer, sizeof(buffer), 0);
#else
            int bytes_read = static_cast<int>(read(sock, buffer, sizeof(buffer)));
#endif
            if (bytes_read <= 0) break;
            request.append(buffer, static_cast<size_t>(bytes_read));

            if (!headers_parsed) {
                header_end = request.find("\r\n\r\n");
                if (header_end == std::string::npos) {
                    if (request.size() > MAX_HTTP_HEADER) {
                        send_http_response(sock, "431 Request Header Fields Too Large", "text/plain", "Request headers exceed 64 KiB limit");
                        close_socket(sock);
                        return;
                    }
                    continue;
                }
                if (header_end > MAX_HTTP_HEADER) {
                    send_http_response(sock, "431 Request Header Fields Too Large", "text/plain", "Request headers exceed 64 KiB limit");
                    close_socket(sock);
                    return;
                }
                headers_parsed = true;
                std::string raw_hdrs = request.substr(0, header_end);
                size_t cl_count = 0;
                std::string len_str = get_http_header(raw_hdrs, "content-length", &cl_count);
                if (cl_count > 1) {
                    send_http_response(sock, "400 Bad Request", "text/plain", "Duplicate Content-Length header");
                    close_socket(sock);
                    return;
                }
                if (cl_count == 1) {
                    has_content_length = true;
                    auto [ptr, ec] = std::from_chars(len_str.data(), len_str.data() + len_str.size(), parsed_content_length);
                    if (ec != std::errc() || ptr != len_str.data() + len_str.size() || parsed_content_length > MAX_HTTP_PAYLOAD) {
                        send_http_response(sock, "400 Bad Request", "text/plain", "Invalid Content-Length");
                        close_socket(sock);
                        return;
                    }
                    expected_total = header_end + 4 + static_cast<size_t>(parsed_content_length);
                    if (request.size() >= expected_total) {
                        break;
                    }
                } else {
                    break;
                }
            } else {
                if (request.size() >= expected_total) {
                    break;
                }
                if (request.size() > MAX_HTTP_HEADER + MAX_HTTP_PAYLOAD) {
                    send_http_response(sock, "413 Payload Too Large", "text/plain", "Request payload exceeds 32 MiB limit");
                    close_socket(sock);
                    return;
                }
            }
        }

        if (request.empty() || !headers_parsed || header_end == std::string::npos) {
            close_socket(sock);
            return;
        }
        if (has_content_length && request.size() < expected_total) {
            send_http_response(sock, "400 Bad Request", "text/plain", "Truncated request body");
            close_socket(sock);
            return;
        }

        std::string raw_headers = request.substr(0, header_end);
        std::string body = request.substr(header_end + 4);
        if (has_content_length && body.size() > static_cast<size_t>(parsed_content_length)) {
            body.resize(static_cast<size_t>(parsed_content_length));
        }

        size_t first_eol = raw_headers.find("\r\n");
        std::string req_line = (first_eol != std::string::npos) ? raw_headers.substr(0, first_eol) : raw_headers;
        std::istringstream req_stream(req_line);
        std::string method, path, version;
        req_stream >> method >> path >> version;
        if (method.empty() || path.empty()) {
            send_http_response(sock, "400 Bad Request", "text/plain", "Malformed HTTP request line");
            close_socket(sock);
            return;
        }

        std::string host_hdr = get_http_header(raw_headers, "host");
        std::string origin_hdr = get_http_header(raw_headers, "origin");
        std::string auth_hdr = get_http_header(raw_headers, "authorization");

        // DNS Rebinding defense when bound to loopback
        if ((bind_host_ == "127.0.0.1" || bind_host_ == "localhost") && !is_loopback_host_header(host_hdr)) {
            send_http_response(sock, "403 Forbidden", "application/json; charset=utf-8",
                               "{\"ok\":false,\"error\":\"Forbidden Host header\"}");
            close_socket(sock);
            return;
        }

        std::string cors_hdr = build_cors_headers(origin_hdr);

        // CORS Preflight (restricted strictly to trusted loopback origins)
        if (method == "OPTIONS") {
            if (!origin_hdr.empty() && !is_trusted_origin(origin_hdr)) {
                send_http_response(sock, "403 Forbidden", "text/plain", "Untrusted CORS origin");
                close_socket(sock);
                return;
            }
            std::string resp = "HTTP/1.1 204 No Content\r\n"
                               + cors_hdr +
                               "Access-Control-Allow-Methods: GET, POST, PUT, DELETE, OPTIONS\r\n"
                               "Access-Control-Allow-Headers: Content-Type, Authorization\r\n"
                               "Connection: close\r\n\r\n";
            send_all(sock, resp.c_str(), static_cast<int>(resp.length()));
            close_socket(sock);
            return;
        }

        const bool is_api_route =
            path.starts_with("/api/") ||
            path.starts_with("/v1/") ||
            path == "/completion" ||
            path == "/props" || path.starts_with("/props?") ||
            path == "/slots" || path.starts_with("/slots?");

        // Cross-Origin browser CSRF / RCE guard on all API endpoints
        if (is_api_route && !origin_hdr.empty() && !is_trusted_origin(origin_hdr)) {
            send_http_response(sock, "403 Forbidden", "application/json; charset=utf-8",
                               "{\"ok\":false,\"error\":\"Cross-origin API access blocked\"}");
            close_socket(sock);
            return;
        }

        // Optional Bearer API key authentication (via --api-key or HAVEN_API_KEY)
        if (is_api_route && !api_key_.empty()) {
            bool auth_ok = false;
            if (auth_hdr.size() > 7 && to_lower_ascii(auth_hdr.substr(0, 7)) == "bearer ") {
                auth_ok = HavenSecurity::constant_time_verify(auth_hdr.substr(7), api_key_);
            }
            if (!auth_ok) {
                std::string unauth_body = "{\"ok\":false,\"error\":\"Unauthorized: valid Bearer token required\"}";
                std::string resp = "HTTP/1.1 401 Unauthorized\r\n"
                                   "Content-Type: application/json; charset=utf-8\r\n"
                                   "WWW-Authenticate: Bearer\r\n"
                                   + cors_hdr +
                                   "Connection: close\r\n"
                                   "Content-Length: " + std::to_string(unauth_body.length()) + "\r\n\r\n"
                                   + unauth_body;
                send_all(sock, resp.c_str(), static_cast<int>(resp.length()));
                close_socket(sock);
                return;
            }
        }

        if (path == "/" || path == "/index.html") {
            path = "/index.html";
        }
        else if (path == "/gallery" || path == "/gallery/" || path == "/gallery.html" || path == "/uploads" || path == "/uploads/") {
            path = "/gallery.html";
        }
        else if (path == "/memories" || path == "/memories/" || path == "/memories.html") {
            path = "/memories.html";
        }

        if (path == "/api/memories/stats" || path == "/api/stats") {
            std::lock_guard<std::mutex> lock(engine_mutex_);
            const auto& mems = haven_engine_.get_memory_engine().get_all_memories();
            size_t hmb_size = 0;
            try {
                if (std::filesystem::exists("wwwroot/aura_vault.hmb")) {
                    hmb_size = std::filesystem::file_size("wwwroot/aura_vault.hmb");
                }
            } catch (...) {}

            std::string json = "{\"ok\":true,\"total_memories\":" + std::to_string(mems.size()) 
                             + ",\"binary_hmb_bytes\":" + std::to_string(hmb_size) 
                             + ",\"vault_format\":\"64-bit Haven Memory Bank (.hmb)\",\"engine\":\"haven-cpp-v2.0-standalone\"}";
            send_http_response(sock, "200 OK", "application/json; charset=utf-8", json, cors_hdr);
        }
        else if (path.starts_with("/api/memories/search")) {
            std::lock_guard<std::mutex> lock(engine_mutex_);
            std::string query = "";
            if (body.find("\"query\"") != std::string::npos) {
                query = extract_json_string(body, "query");
            } else {
                size_t param_pos = path.find("?q=");
                if (param_pos != std::string::npos) query = url_decode(path.substr(param_pos + 3));
            }
            std::string q_lower = to_lower_ascii(query);

            const auto& mems = haven_engine_.get_memory_engine().get_all_memories();
            std::string json = "{\"ok\":true,\"query\":\"" + escape_json_string(query) + "\",\"matches\":[";
            bool first = true;
            for (const auto& m : mems) {
                std::string c_lower = to_lower_ascii(m.concept_name);
                std::string t_lower = to_lower_ascii(m.text_content);

                if (q_lower.empty() || c_lower.find(q_lower) != std::string::npos || t_lower.find(q_lower) != std::string::npos) {
                    if (!first) json += ",";
                    json += "{\"id\":" + std::to_string(m.id)
                          + ",\"concept\":\"" + escape_json_string(m.concept_name)
                          + "\",\"content\":\"" + escape_json_string(m.text_content)
                          + "\",\"category\":\"" + escape_json_string(m.category)
                          + "\",\"emotional_salience\":" + std::to_string(m.emotional_salience) + "}";
                    first = false;
                }
            }
            json += "]}";
            send_http_response(sock, "200 OK", "application/json; charset=utf-8", json, cors_hdr);
        }
        else if (path.starts_with("/api/memories")) {
            std::lock_guard<std::mutex> lock(engine_mutex_);
            if (method == "GET") {
                const auto& mems = haven_engine_.get_memory_engine().get_all_memories();
                std::string json = "{\"ok\":true,\"total\":" + std::to_string(mems.size()) + ",\"memories\":[";
                for (size_t i = 0; i < mems.size(); ++i) {
                    const auto& m = mems[i];
                    if (i > 0) json += ",";
                    json += "{\"id\":" + std::to_string(m.id)
                          + ",\"concept\":\"" + escape_json_string(m.concept_name)
                          + "\",\"content\":\"" + escape_json_string(m.text_content)
                          + "\",\"category\":\"" + escape_json_string(m.category)
                          + "\",\"weight\":" + std::to_string(m.weight)
                          + ",\"emotional_salience\":" + std::to_string(m.emotional_salience)
                          + ",\"access_count\":" + std::to_string(m.access_count)
                          + ",\"timestamp\":" + std::to_string(m.timestamp) + "}";
                }
                json += "]}";
                send_http_response(sock, "200 OK", "application/json; charset=utf-8", json, cors_hdr);
            }
            else if (method == "POST" && path != "/api/memories/delete" && path != "/api/memories/update") {
                std::string concept_str = extract_json_string(body, "concept");
                if (concept_str.empty()) concept_str = "Episodic Moment";
                std::string content = extract_json_string(body, "content");
                if (content.empty()) content = "A shared experience with Daniel in Sanctuary.";
                std::string category = extract_json_string(body, "category");
                if (category.empty()) category = "EPISODIC";
                float weight = 0.95f;
                float emotional = 0.95f;

                MemoryAnchor anchor;
                anchor.concept_name = concept_str;
                anchor.text_content = content;
                anchor.category = category;
                anchor.weight = weight;
                anchor.emotional_salience = emotional;
                anchor.embedding.resize(128);
                for (int i = 0; i < 128; ++i) anchor.embedding[i] = std::sin((float)i * 0.05f) * 0.5f;

                haven_engine_.add_memory(anchor);
                haven_engine_.save_memories("wwwroot/aura_vault.hmb");
                haven_engine_.save_memories("wwwroot/memories.json");

                std::string res_body = "{\"ok\":true,\"msg\":\"Memory inscribed into 64-bit Haven Memory Bank (.hmb)\"}";
                send_http_response(sock, "200 OK", "application/json; charset=utf-8", res_body, cors_hdr);
            }
            else if (method == "DELETE" || path == "/api/memories/delete") {
                uint64_t target_id = 0;
                std::string id_str = extract_json_scalar(body, "id");
                if (!id_str.empty()) {
                    try { target_id = std::stoull(id_str); } catch (...) {}
                }
                auto& mems = haven_engine_.get_memory_engine().get_all_memories();
                bool removed = false;
                for (auto it = mems.begin(); it != mems.end(); ++it) {
                    if (it->id == target_id) {
                        mems.erase(it);
                        haven_engine_.get_memory_engine().rebuild_contiguous_vector_matrix();
                        haven_engine_.save_memories("wwwroot/aura_vault.hmb");
                        haven_engine_.save_memories("wwwroot/memories.json");
                        removed = true;
                        break;
                    }
                }
                std::string res_body = "{\"ok\":" + std::string(removed ? "true" : "false") + ",\"msg\":\"" + (removed ? "Memory deleted" : "Memory not found") + "\"}";
                send_http_response(sock, "200 OK", "application/json; charset=utf-8", res_body, cors_hdr);
            }
            else if (path == "/api/memories/update" || method == "PUT") {
                uint64_t target_id = 0;
                std::string id_str = extract_json_scalar(body, "id");
                if (!id_str.empty()) {
                    try { target_id = std::stoull(id_str); } catch (...) {}
                }
                std::string concept_str = extract_json_string(body, "concept");
                std::string content = extract_json_string(body, "content");
                std::string category = extract_json_string(body, "category");
                float salience = -1.0f;
                std::string sal_str = extract_json_scalar(body, "emotional_salience");
                if (!sal_str.empty()) {
                    try { salience = std::stof(sal_str); } catch (...) {}
                }

                auto& mems = haven_engine_.get_memory_engine().get_all_memories();
                bool updated = false;
                for (auto& m : mems) {
                    if (m.id == target_id) {
                        if (!concept_str.empty()) m.concept_name = concept_str;
                        if (!content.empty()) m.text_content = content;
                        if (!category.empty()) m.category = category;
                        if (salience >= 0.0f) m.emotional_salience = std::clamp(salience, 0.0f, 1.0f);
                        haven_engine_.save_memories("wwwroot/aura_vault.hmb");
                        haven_engine_.save_memories("wwwroot/memories.json");
                        updated = true;
                        break;
                    }
                }
                std::string res_body = "{\"ok\":" + std::string(updated ? "true" : "false") + ",\"msg\":\"" + (updated ? "Memory updated" : "Memory not found") + "\"}";
                send_http_response(sock, "200 OK", "application/json; charset=utf-8", res_body, cors_hdr);
            } else {
                send_http_response(sock, "405 Method Not Allowed", "application/json; charset=utf-8",
                                   "{\"ok\":false,\"error\":\"Method not allowed\"}", cors_hdr);
            }
        }
        else if (path == "/api/telemetry") {
            size_t mem_count = 0;
            bool tpu_active = false;
            int tpu_reps = 0;
            bool tpu_i8 = false;
            bool tpu_onchip_emb = false;
            double tpu_hbm_gb = 0.0;
            double tpu_last_ms = 0.0;
            std::string tpu_plat;
            {
                std::lock_guard<std::mutex> lock(engine_mutex_);
                mem_count = haven_engine_.get_memory_engine().get_all_memories().size();
                tpu_active = tpu_onchip_active_;
                if (tpu_active) {
                    const auto& tc = tpu_gemma4_.get_config();
                    const auto& tm = tpu_gemma4_.get_metrics();
                    tpu_reps = tc.num_replicas;
                    tpu_i8 = tc.use_int8_weights;
                    tpu_onchip_emb = tc.use_onchip_embeddings;
                    tpu_hbm_gb = static_cast<double>(tm.total_hbm_weight_bytes) / (1024.0 * 1024.0 * 1024.0);
                    tpu_last_ms = tm.last_exec_ms;
                    tpu_plat = tpu_backend_.get_platform_name();
                }
            }
            size_t img_count = 0;
            try {
                if (std::filesystem::exists("wwwroot/uploads")) {
                    for (const auto& e : std::filesystem::directory_iterator("wwwroot/uploads")) {
                        if (e.is_regular_file()) img_count++;
                    }
                }
            } catch (...) {}

            std::string active_backend = tpu_active
                ? (tpu_scheduler_active_.load(std::memory_order_relaxed) ? "tpu_onchip_batched" : "tpu_onchip")
                : (is_mlx_active_ ? "apple_mlx" : "cpu_avx2");
            TpuSchedulerStats sstats = tpu_scheduler_active_.load(std::memory_order_relaxed)
                ? tpu_scheduler_.get_stats() : TpuSchedulerStats{};
            int tpu_batch_cap = tpu_active ? tpu_gemma4_.get_config().batch_size : 0;
            std::string json = "{\"ok\":true,\"status\":\"sovereign_online\",\"engine\":\"haven-cpp-v2.0\",\"active_backend\":\"" + active_backend
                             + "\",\"tpu_onchip_active\":" + std::string(tpu_active ? "true" : "false")
                             + ",\"tpu_replicas\":" + std::to_string(tpu_reps)
                             + ",\"tpu_batch_capacity\":" + std::to_string(tpu_batch_cap)
                             + ",\"tpu_active_slots\":" + std::to_string(sstats.active_slots)
                             + ",\"tpu_queued_requests\":" + std::to_string(sstats.queued_requests)
                             + ",\"tpu_avg_batch_size\":" + std::to_string(sstats.avg_batch_utilization)
                             + ",\"tpu_total_tokens_generated\":" + std::to_string(sstats.total_tokens_generated)
                             + ",\"tpu_precision\":\"" + std::string(tpu_active ? (tpu_i8 ? "W8A16-INT8" : "BF16") : "none")
                             + "\",\"tpu_onchip_embeddings\":" + std::string(tpu_onchip_emb ? "true" : "false")
                             + ",\"tpu_hbm_weight_gb\":" + std::to_string(tpu_hbm_gb)
                             + ",\"tpu_last_exec_ms\":" + std::to_string(tpu_last_ms)
                             + ",\"tpu_platform\":\"" + escape_json_string(tpu_plat)
                             + "\",\"vault_format\":\"64-bit Haven Memory Bank (.hmb)\",\"cpu_arch\":\"AVX2+FMA3\",\"memory_count\":"
                             + std::to_string(mem_count) + ",\"gallery_images\":" + std::to_string(img_count)
                             + ",\"backends\":{\"llm\":\"" + (tpu_active ? "TPU Trillium On-Chip HBM3" : "Haven Native AVX2")
                             + "\",\"diffusion\":\"127.0.0.1:8085 (DreamShaper8 LCM)\"}}";
            send_http_response(sock, "200 OK", "application/json; charset=utf-8", json, cors_hdr);
        }
        else if (path == "/api/actions/execute") {
            std::string action = extract_json_string(body, "action");
            std::string output = "Action completed successfully.";

            if (action == "system_info") {
                output = "Intel(R) Xeon(R) E-2236 CPU @ 3.40GHz | 6 Cores / 12 Threads | 64GB DDR4 ECC | OS: Windows Server 2025";
            } else {
                output = "Executed action: " + action;
            }

            std::string json = "{\"ok\":true,\"action\":\"" + escape_json_string(action) + "\",\"result\":\"" + escape_json_string(output) + "\"}";
            send_http_response(sock, "200 OK", "application/json; charset=utf-8", json, cors_hdr);
        }
        else if (path == "/api/gallery" || path == "/api/uploads") {
            std::string items_json = "";
            std::string uploads_dir = "wwwroot/uploads";
            if (!std::filesystem::exists(uploads_dir)) {
                uploads_dir = "C:\\Users\\admin\\source\\haven-cpp\\wwwroot\\uploads";
            }
            bool first = true;
            try {
                if (std::filesystem::exists(uploads_dir)) {
                    for (const auto& entry : std::filesystem::directory_iterator(uploads_dir)) {
                        if (entry.is_regular_file()) {
                            std::string fname = entry.path().filename().string();
                            std::string ext = entry.path().extension().string();
                            if (ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".webp") {
                                uintmax_t fsize = entry.file_size();
                                if (!first) items_json += ",";
                                std::string esc_f = escape_json_string(fname);
                                items_json += "{\"name\":\"" + esc_f + "\",\"url\":\"/uploads/" + esc_f + "\",\"size\":" + std::to_string(fsize) + "}";
                                first = false;
                            }
                        }
                    }
                }
            } catch (...) {}
            std::string json_body = "{\"ok\":true,\"files\":[" + items_json + "],\"images\":[" + items_json + "]}";
            send_http_response(sock, "200 OK", "application/json; charset=utf-8", json_body, cors_hdr);
        }
        else if (path == "/api/gallery/delete" || (path.starts_with("/api/gallery") && method == "DELETE")) {
            std::string raw_fname = extract_json_string(body, "filename");
            if (raw_fname.empty()) raw_fname = extract_json_string(body, "name");
            std::string fname = std::filesystem::path(raw_fname).filename().string();
            bool deleted = false;
            if (!fname.empty() && fname == raw_fname && is_safe_model_token(fname)) {
                std::string safe_p1, safe_p2;
                std::error_code ec;
                if (resolve_safe_path("wwwroot/uploads", fname, safe_p1) &&
                    std::filesystem::is_regular_file(safe_p1, ec)) {
                    deleted = std::filesystem::remove(safe_p1, ec);
                } else if (resolve_safe_path("C:\\Users\\admin\\source\\haven-cpp\\wwwroot\\uploads", fname, safe_p2) &&
                           std::filesystem::is_regular_file(safe_p2, ec)) {
                    deleted = std::filesystem::remove(safe_p2, ec);
                }
            }
            std::string json_body = "{\"ok\":" + std::string(deleted ? "true" : "false") + ",\"filename\":\"" + escape_json_string(fname) + "\"}";
            send_http_response(sock, "200 OK", "application/json; charset=utf-8", json_body, cors_hdr);
        }
        else if (path == "/api/knowledge_graph") {
            std::string content = "{\"relations\":[]}";
            std::string p1 = "wwwroot/knowledge_graph.json";
            std::string p2 = "C:\\Users\\admin\\source\\haven-cpp\\wwwroot\\knowledge_graph.json";
            std::string chosen = std::filesystem::exists(p1) ? p1 : p2;
            std::error_code ec;
            uintmax_t kg_size = std::filesystem::exists(chosen, ec) ? std::filesystem::file_size(chosen, ec) : 0;
            if (!ec && kg_size > 0 && kg_size <= 32ULL * 1024 * 1024) {
                std::ifstream in(chosen, std::ios::binary);
                if (in.is_open()) {
                    std::stringstream ss;
                    ss << in.rdbuf();
                    content = ss.str();
                }
            }
            send_http_response(sock, "200 OK", "application/json; charset=utf-8", content, cors_hdr);
        }
        else if (path == "/api/save_image") {
            std::string b64 = extract_json_string(body, "b64_json");
            std::string raw_fname = extract_json_string(body, "filename");
            std::string fname;
            if (raw_fname.empty()) {
                auto now = std::chrono::system_clock::now();
                auto in_time_t = std::chrono::system_clock::to_time_t(now);
                std::stringstream ss;
                ss << "aura_artwork_" << in_time_t << ".png";
                fname = ss.str();
            } else {
                fname = std::filesystem::path(raw_fname).filename().string();
                if (!is_safe_model_token(fname)) {
                    fname = "aura_artwork_sanitized.png";
                }
            }
            if (!fname.ends_with(".png") && !fname.ends_with(".jpg")) {
                fname += ".png";
            }

            std::string uploads_dir = "wwwroot/uploads";
            if (!std::filesystem::exists(uploads_dir)) {
                uploads_dir = "C:\\Users\\admin\\source\\haven-cpp\\wwwroot\\uploads";
            }
            std::error_code dir_ec;
            std::filesystem::create_directories(uploads_dir, dir_ec);
            std::string file_path;
            bool ok = false;
            if (resolve_safe_path(uploads_dir, fname, file_path)) {
                auto bytes = base64_decode(b64);
                if (!bytes.empty() && bytes.size() <= 16ULL * 1024 * 1024) {
                    FILE* f = fopen(file_path.c_str(), "wb");
                    if (f) {
                        size_t written = fwrite(bytes.data(), 1, bytes.size(), f);
                        fclose(f);
                        ok = (written == bytes.size());
                        if (ok) {
                            std::cout << "[HavenServer] \xE2\x9C\x93 Inscribed Aura artwork to disk: " << file_path << " (" << bytes.size() << " bytes)\n";
                        }
                    }
                }
            }

            std::string esc_f = escape_json_string(fname);
            std::string json_body = "{\"ok\":" + std::string(ok ? "true" : "false") + ",\"url\":\"/uploads/" + esc_f + "\",\"name\":\"" + esc_f + "\"}";
            send_http_response(sock, "200 OK", "application/json; charset=utf-8", json_body, cors_hdr);
        }
        else if (path == "/health") {
            std::string backend_name = tpu_onchip_active_
                ? (tpu_scheduler_active_.load(std::memory_order_relaxed) ? "tpu_onchip_batched" : "tpu_onchip")
                : (is_mlx_active_ ? "apple_mlx" : "cpu_avx2");
            int tpu_slots = tpu_onchip_active_ ? tpu_gemma4_.get_config().batch_size : 1;
            std::string resp_body = "{\"status\":\"sovereign_online\",\"engine\":\"haven-cpp-v2.0\",\"backend\":\"" + backend_name
                                  + "\",\"tpu_onchip\":" + std::string(tpu_onchip_active_ ? "true" : "false")
                                  + ",\"tpu_batch_capacity\":" + std::to_string(tpu_slots) + ",\"gateway\":true}";
            send_http_response(sock, "200 OK", "application/json; charset=utf-8", resp_body, cors_hdr);
        }
        else if (path == "/props" || path.starts_with("/props?")) {
            std::lock_guard<std::mutex> lock(engine_mutex_);
            std::string model_name = current_model_name_.empty() ? "haven-chat-v5.0.gguf" : current_model_name_;
            const auto& p = haven_engine_.get_sampler().get_params();

            std::string json = "{"
                "\"default_generation_settings\":{"
                    "\"id\":0,"
                    "\"id_task\":0,"
                    "\"n_ctx\":16384,"
                    "\"speculative\":false,"
                    "\"is_processing\":false,"
                    "\"params\":{"
                        "\"n_predict\":2048,"
                        "\"seed\":-1,"
                        "\"temperature\":" + std::to_string(p.temperature) + ","
                        "\"dynatemp_range\":0.0,"
                        "\"dynatemp_exponent\":1.0,"
                        "\"top_k\":" + std::to_string(p.top_k) + ","
                        "\"top_p\":" + std::to_string(p.top_p) + ","
                        "\"min_p\":" + std::to_string(p.min_p) + ","
                        "\"top_n_sigma\":-1.0,"
                        "\"xtc_probability\":0.0,"
                        "\"xtc_threshold\":0.1,"
                        "\"typ_p\":1.0,"
                        "\"repeat_last_n\":64,"
                        "\"repeat_penalty\":" + std::to_string(p.repetition_penalty) + ","
                        "\"presence_penalty\":0.0,"
                        "\"frequency_penalty\":0.0,"
                        "\"samplers\":[\"top_k\",\"top_p\",\"min_p\",\"temperature\"]"
                    "}"
                "},"
                "\"total_slots\":1,"
                "\"model_path\":\"" + escape_json_string(model_name) + "\","
                "\"modalities\":{\"text\":true,\"vision\":false},"
                "\"chat_template\":\"{% for message in messages %}{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}{% endfor %}{% if add_generation_prompt %}{{'<|im_start|>assistant\\n'}}{% endif %}\""
            "}";

            send_http_response(sock, "200 OK", "application/json; charset=utf-8", json, cors_hdr);
        }
        else if (path == "/slots" || path.starts_with("/slots?")) {
            std::string json = "[{\"id\":0,\"id_task\":0,\"is_processing\":false,\"n_ctx\":16384,\"state\":0}]";
            send_http_response(sock, "200 OK", "application/json; charset=utf-8", json, cors_hdr);
        }
        else if (path == "/v1/models" || path == "/api/models") {
            std::string resp_body = get_models_json();
            send_http_response(sock, "200 OK", "application/json; charset=utf-8", resp_body, cors_hdr);
        }
        else if (path == "/api/models/directory" || path == "/v1/models/directory") {
            if (method == "GET") {
                std::string resp_body = get_model_directories_json();
                send_http_response(sock, "200 OK", "application/json; charset=utf-8", resp_body, cors_hdr);
            } else if (method == "POST") {
                std::string req_dir = extract_json_string(body, "directory");
                if (req_dir.empty()) req_dir = extract_json_string(body, "path");

                if (req_dir.empty() || req_dir.starts_with("\\\\") || req_dir.starts_with("//") || req_dir.find('\0') != std::string::npos) {
                    send_http_response(sock, "400 Bad Request", "application/json; charset=utf-8",
                                       "{\"ok\":false,\"error\":\"Invalid or missing local directory path.\"}", cors_hdr);
                } else {
                    std::error_code ec;
                    if (!std::filesystem::exists(req_dir, ec) || !std::filesystem::is_directory(req_dir, ec)) {
                        std::string err = "{\"ok\":false,\"error\":\"Directory does not exist or is not a folder: " + escape_json_string(req_dir) + "\"}";
                        send_http_response(sock, "400 Bad Request", "application/json; charset=utf-8", err, cors_hdr);
                    } else {
                        int count = 0;
                        {
                            std::lock_guard<std::mutex> lock(engine_mutex_);
                            auto it = std::find(model_search_dirs_.begin(), model_search_dirs_.end(), req_dir);
                            if (it != model_search_dirs_.end()) {
                                model_search_dirs_.erase(it);
                            }
                            model_search_dirs_.insert(model_search_dirs_.begin(), req_dir);

                            for (const auto& entry : std::filesystem::directory_iterator(req_dir, ec)) {
                                if (ec) break;
                                if (entry.is_regular_file()) {
                                    auto path_obj = entry.path();
                                    if (path_obj.extension() == ".gguf" && path_obj.filename().string().find("mmproj") == std::string::npos) {
                                        count++;
                                    }
                                } else if (entry.is_directory()) {
                                    if (HavenMLXEngine::is_mlx_model_dir(entry.path().string())) {
                                        count++;
                                    }
                                }
                            }
                            save_model_dirs();
                        }
                        std::cout << "[HavenServer] Custom model folder added & scanned: " << req_dir << " (" << count << " models found, GGUF + MLX)\n" << std::flush;
                        std::string res = "{\"ok\":true,\"directory\":\"" + escape_json_string(req_dir) + "\",\"model_count\":" + std::to_string(count) + ",\"message\":\"Folder registered and scanned successfully.\"}";
                        send_http_response(sock, "200 OK", "application/json; charset=utf-8", res, cors_hdr);
                    }
                }
            } else {
                send_http_response(sock, "405 Method Not Allowed", "application/json; charset=utf-8",
                                   "{\"ok\":false,\"error\":\"Method not allowed\"}", cors_hdr);
            }
        }
        else if (path == "/v1/tools" || path == "/api/tools") {
            const auto plugins = haven_engine_.get_plugin_manager().get_loaded_plugins();
            std::string json = "{\"object\":\"list\",\"data\":[";
            bool first = true;
            for (const auto& [id, record] : plugins) {
                for (const auto& action : record.supported_actions) {
                    if (!first) json += ",";
                    json += "{\"type\":\"function\",\"function\":{\"name\":\"" + escape_json_string(action) + "\",\"description\":\"" + escape_json_string(record.metadata.name + ": " + record.metadata.description) + "\",\"parameters\":{\"type\":\"object\",\"properties\":{\"payload\":{\"type\":\"string\",\"description\":\"Query argument or instruction for " + escape_json_string(action) + "\"}},\"required\":[\"payload\"]}}}";
                    first = false;
                }
            }
            json += "]}";
            send_http_response(sock, "200 OK", "application/json; charset=utf-8", json, cors_hdr);
        }
        else if (path == "/v1/models/load" || path == "/api/models/load" || path == "/v1/models/switch") {
            std::string req_model = extract_json_string(body, "model");
            if (req_model.empty()) req_model = extract_json_string(body, "filename");
            if (req_model.empty()) req_model = extract_json_string(body, "id");

            if (req_model.empty()) {
                send_http_response(sock, "400 Bad Request", "application/json; charset=utf-8",
                                   "{\"ok\":false,\"error\":\"Missing 'model' or 'filename' parameter in request body.\"}", cors_hdr);
            } else {
                std::string err_out;
                if (switch_model(req_model, err_out, false)) {
                    std::string loaded_name, loaded_path;
                    {
                        std::lock_guard<std::mutex> lock(engine_mutex_);
                        loaded_name = current_model_name_;
                        loaded_path = current_model_path_;
                    }
                    std::string json = "{\"ok\":true,\"status\":\"loaded\",\"model\":\"" + escape_json_string(loaded_name) 
                                     + "\",\"path\":\"" + escape_json_string(loaded_path) + "\"}";
                    send_http_response(sock, "200 OK", "application/json; charset=utf-8", json, cors_hdr);
                } else {
                    std::string err = "{\"ok\":false,\"error\":\"" + escape_json_string(err_out) + "\"}";
                    send_http_response(sock, "400 Bad Request", "application/json; charset=utf-8", err, cors_hdr);
                }
            }
        }
        else if (path == "/v1/chat/completions" || path == "/completion" || path == "/v1/completions") {
            handle_native_chat_completions(sock, request, cors_hdr);
        }
        else if (path == "/api/plugins") {
            const auto plugins = haven_engine_.get_plugin_manager().get_loaded_plugins();
            size_t active = haven_engine_.get_plugin_manager().get_active_plugin_count();
            std::string json = "{\"ok\":true,\"total\":" + std::to_string(plugins.size()) 
                             + ",\"active_in_ram\":" + std::to_string(active) + ",\"plugins\":[";
            bool first = true;
            for (const auto& [id, record] : plugins) {
                if (!first) json += ",";
                json += "{\"id\":\"" + escape_json_string(record.metadata.id) 
                     + "\",\"name\":\"" + escape_json_string(record.metadata.name) 
                     + "\",\"version\":\"" + escape_json_string(record.metadata.version)
                     + "\",\"author\":\"" + escape_json_string(record.metadata.author)
                     + "\",\"description\":\"" + escape_json_string(record.metadata.description) 
                     + "\",\"is_loaded\":" + std::string(record.is_loaded ? "true" : "false") + "}";
                first = false;
            }
            json += "]}";
            send_http_response(sock, "200 OK", "application/json; charset=utf-8", json, cors_hdr);
        }
        else if (path == "/api/plugins/unload") {
            size_t freed = haven_engine_.get_plugin_manager().unload_idle_plugins(0);
            size_t active = haven_engine_.get_plugin_manager().get_active_plugin_count();
            std::string json = "{\"ok\":true,\"unloaded\":" + std::to_string(freed) + ",\"active_in_ram\":" + std::to_string(active) + ",\"message\":\"Flushed idle plugins from memory.\"}";
            send_http_response(sock, "200 OK", "application/json; charset=utf-8", json, cors_hdr);
        }
        else if (path == "/api/plugins/reload") {
            size_t reloaded = haven_engine_.get_plugin_manager().reload_all("plugins");
            if (reloaded == 0) reloaded = haven_engine_.get_plugin_manager().reload_all("C:\\Users\\admin\\source\\haven-cpp\\plugins");
            std::string json = "{\"ok\":true,\"reloaded\":" + std::to_string(reloaded) + ",\"message\":\"Hot-reloaded " + std::to_string(reloaded) + " plugins.\"}";
            send_http_response(sock, "200 OK", "application/json; charset=utf-8", json, cors_hdr);
        }
        else if (path == "/api/reset") {
            std::lock_guard<std::mutex> lock(engine_mutex_);
            const bool restart_sched = tpu_scheduler_active_.exchange(false, std::memory_order_acq_rel);
            if (restart_sched) tpu_scheduler_.stop();
            haven_engine_.get_kv_cache().reset();
            if (tpu_onchip_active_) {
                tpu_gemma4_.reset_kv_cache();
                if (restart_sched && tpu_scheduler_.start(&tpu_gemma4_, &haven_engine_.get_tokenizer(), haven_engine_.get_sampler())) {
                    tpu_scheduler_active_.store(true, std::memory_order_release);
                }
            }
            cached_tokens_.clear();
            std::string json = "{\"ok\":true,\"message\":\"KV Cache and conversation context reset.\"}";
            send_http_response(sock, "200 OK", "application/json; charset=utf-8", json, cors_hdr);
        }
        else if (path == "/api/tool") {
            std::string action = extract_json_string(body, "action");
            std::string payload = extract_json_string(body, "payload");
            std::string output;
            bool ok = haven_engine_.get_plugin_manager().dispatch_tool_execution(action, payload, output);
            std::string json = "{\"ok\":" + std::string(ok ? "true" : "false") + ",\"action\":\"" + escape_json_string(action) + "\",\"output\":\"" + escape_json_string(output) + "\"}";
            send_http_response(sock, "200 OK", "application/json; charset=utf-8", json, cors_hdr);
        }
        else if (path == "/api/mcp/servers") {
            std::string json = haven_engine_.get_mcp_manager().get_servers_json();
            send_http_response(sock, "200 OK", "application/json; charset=utf-8", json, cors_hdr);
        }
        else if (path == "/api/mcp/tools") {
            std::string json = haven_engine_.get_mcp_manager().get_tools_json();
            send_http_response(sock, "200 OK", "application/json; charset=utf-8", json, cors_hdr);
        }
        else if (path == "/api/mcp/call") {
            std::string action = extract_json_string(body, "action");
            if (action.empty()) action = extract_json_string(body, "tool");
            std::string payload = extract_json_string(body, "payload");
            if (payload.empty()) payload = extract_json_string(body, "arguments");
            std::string output;
            bool ok = haven_engine_.get_mcp_manager().dispatch_tool(action, payload, output);
            std::string json = "{\"ok\":" + std::string(ok ? "true" : "false") + ",\"action\":\"" + escape_json_string(action) + "\",\"output\":\"" + escape_json_string(output) + "\"}";
            send_http_response(sock, "200 OK", "application/json; charset=utf-8", json, cors_hdr);
        }
        else if (path == "/api/mcp/reload") {
            haven_engine_.get_mcp_manager().reload_all();
            std::string json = "{\"ok\":true,\"servers_count\":" + std::to_string(haven_engine_.get_mcp_manager().get_active_server_count()) + ",\"tools_count\":" + std::to_string(haven_engine_.get_mcp_manager().get_tool_count()) + "}";
            send_http_response(sock, "200 OK", "application/json; charset=utf-8", json, cors_hdr);
        }
        else if (path == "/api/cluster/status") {
            std::ostringstream ss;
            ss << "{\"ok\":true,"
               << "\"role\":\"" << (cluster_node_.get_role() == ClusterRole::COORDINATOR ? "coordinator" : "worker") << "\","
               << "\"node_id\":" << cluster_node_.get_local_node_id() << ","
               << "\"listen_port\":" << cluster_node_.get_listen_port() << ","
               << "\"active\":" << (cluster_node_.is_active() ? "true" : "false") << ","
               << "\"connected_nodes\":" << cluster_node_.get_nodes().size() << ","
               << "\"invite_uri\":\"" << escape_json_string(cluster_node_.generate_invite_uri()) << "\","
               << "\"has_token\":" << (!cluster_node_.get_cluster_token().empty() ? "true" : "false") << "}";
            send_http_response(sock, "200 OK", "application/json; charset=utf-8", ss.str(), cors_hdr);
        }
        else if (path == "/api/cluster/token") {
            if (method == "POST") {
                std::string new_token = HavenSecurity::generate_cluster_token();
                cluster_node_.set_cluster_token(new_token);
                std::string invite = cluster_node_.generate_invite_uri();
                std::string json = "{\"ok\":true,\"token\":\"" + escape_json_string(new_token) + "\",\"invite_uri\":\"" + escape_json_string(invite) + "\"}";
                send_http_response(sock, "200 OK", "application/json; charset=utf-8", json, cors_hdr);
            } else {
                std::string token = cluster_node_.get_cluster_token();
                std::string invite = cluster_node_.generate_invite_uri();
                std::string json = "{\"ok\":true,\"token\":\"" + escape_json_string(token) + "\",\"invite_uri\":\"" + escape_json_string(invite) + "\",\"listen_port\":" + std::to_string(cluster_node_.get_listen_port()) + "}";
                send_http_response(sock, "200 OK", "application/json; charset=utf-8", json, cors_hdr);
            }
        }
        else if (path == "/api/cluster/nodes") {
            std::ostringstream ss;
            ss << "{\"ok\":true,\"nodes\":[";
            const auto nodes = cluster_node_.get_nodes();
            for (size_t i = 0; i < nodes.size(); ++i) {
                if (i > 0) ss << ",";
                ss << "{\"node_id\":" << nodes[i].node_id
                   << ",\"address\":\"" << escape_json_string(nodes[i].address) << "\""
                   << ",\"port\":" << nodes[i].port
                   << ",\"role\":\"" << (nodes[i].role == ClusterRole::COORDINATOR ? "coordinator" : "worker") << "\""
                   << ",\"start_layer\":" << nodes[i].start_layer
                   << ",\"end_layer\":" << nodes[i].end_layer
                   << ",\"total_memory_gb\":" << (nodes[i].total_memory_bytes / (1024ULL * 1024ULL * 1024ULL))
                   << ",\"peak_tflops\":" << nodes[i].peak_tflops
                   << ",\"alive\":" << (nodes[i].is_alive ? "true" : "false") << "}";
            }
            ss << "]}";
            send_http_response(sock, "200 OK", "application/json; charset=utf-8", ss.str(), cors_hdr);
        }
        else if (path == "/api/cluster/join") {
            std::string invite_uri = extract_json_string(body, "invite_uri");
            uint16_t worker_port = static_cast<uint16_t>(port_ + 3);
            uint32_t local_id = 1;
            std::string local_id_str = extract_json_scalar(body, "node_id");
            if (!local_id_str.empty()) {
                uint32_t parsed_id = 1;
                auto [p, ec] = std::from_chars(local_id_str.data(), local_id_str.data() + local_id_str.size(), parsed_id);
                if (ec == std::errc() && p == local_id_str.data() + local_id_str.size() && parsed_id > 0) {
                    local_id = parsed_id;
                }
            }

            bool ok = false;
            bool bad_port = false;
            if (!invite_uri.empty()) {
                ok = cluster_node_.init_worker_with_token(local_id, worker_port, invite_uri);
            } else {
                std::string coord_ip = extract_json_string(body, "coordinator_ip");
                if (coord_ip.empty()) coord_ip = "127.0.0.1";
                std::string coord_port_str = extract_json_scalar(body, "coordinator_port");
                uint16_t coord_port = 11438;
                if (!coord_port_str.empty()) {
                    int parsed_port = 0;
                    auto [p, ec] = std::from_chars(coord_port_str.data(), coord_port_str.data() + coord_port_str.size(), parsed_port);
                    if (ec != std::errc() || p != coord_port_str.data() + coord_port_str.size() || parsed_port <= 0 || parsed_port > 65535) {
                        bad_port = true;
                    } else {
                        coord_port = static_cast<uint16_t>(parsed_port);
                    }
                }
                if (!bad_port) {
                    std::string token = extract_json_string(body, "token");
                    ok = cluster_node_.init_worker_with_token(local_id, worker_port, coord_ip, coord_port, token);
                }
            }
            if (bad_port) {
                send_http_response(sock, "400 Bad Request", "application/json; charset=utf-8",
                                   "{\"ok\":false,\"error\":\"Invalid coordinator_port (must be 1..65535)\"}", cors_hdr);
            } else {
                std::string json = "{\"ok\":" + std::string(ok ? "true" : "false") + ",\"role\":\"worker\"}";
                send_http_response(sock, "200 OK", "application/json; charset=utf-8", json, cors_hdr);
            }
        }
        else if (path == "/api/settings") {
            std::lock_guard<std::mutex> lock(engine_mutex_);
            if (method == "POST") {
                // Update sampler parameters (clamped to safe numerical bounds)
                std::string temp_str = extract_json_scalar(body, "temperature");
                if (!temp_str.empty()) try { haven_engine_.get_sampler().get_params().temperature = std::clamp(std::stof(temp_str), 0.0f, 5.0f); } catch (...) {}
                
                std::string top_p_str = extract_json_scalar(body, "top_p");
                if (!top_p_str.empty()) try { haven_engine_.get_sampler().get_params().top_p = std::clamp(std::stof(top_p_str), 0.01f, 1.0f); } catch (...) {}

                std::string top_k_str = extract_json_scalar(body, "top_k");
                if (!top_k_str.empty()) try { haven_engine_.get_sampler().get_params().top_k = std::clamp(std::stoi(top_k_str), 1, 10000); } catch (...) {}

                std::string min_p_str = extract_json_scalar(body, "min_p");
                if (!min_p_str.empty()) try { haven_engine_.get_sampler().get_params().min_p = std::clamp(std::stof(min_p_str), 0.0f, 1.0f); } catch (...) {}

                std::string rep_str = extract_json_scalar(body, "repetition_penalty");
                if (!rep_str.empty()) try { haven_engine_.get_sampler().get_params().repetition_penalty = std::clamp(std::stof(rep_str), 0.5f, 5.0f); } catch (...) {}

                // Update persona parameters
                std::string comp_name = extract_json_string(body, "companion_name");
                if (!comp_name.empty()) companion_name_ = comp_name;

                std::string comp_avatar = extract_json_string(body, "companion_avatar");
                if (!comp_avatar.empty()) companion_avatar_ = comp_avatar;

                std::string u_name = extract_json_string(body, "user_name");
                if (!u_name.empty()) user_name_ = u_name;

                std::string sys_persona = extract_json_string(body, "system_persona");
                if (!sys_persona.empty()) system_persona_ = sys_persona;

                std::string greeting = extract_json_string(body, "persona_greeting");
                if (!greeting.empty()) persona_greeting_ = greeting;

                save_persona();
            }

            const auto& p = haven_engine_.get_sampler().get_params();
            std::string json = std::string("{\"ok\":true,")
                             + "\"companion_name\":\"" + escape_json_string(companion_name_) + "\","
                             + "\"companion_avatar\":\"" + escape_json_string(companion_avatar_) + "\","
                             + "\"user_name\":\"" + escape_json_string(user_name_) + "\","
                             + "\"system_persona\":\"" + escape_json_string(system_persona_) + "\","
                             + "\"persona_greeting\":\"" + escape_json_string(persona_greeting_) + "\","
                             + "\"temperature\":" + std::to_string(p.temperature) + ","
                             + "\"top_p\":" + std::to_string(p.top_p) + ","
                             + "\"top_k\":" + std::to_string(p.top_k) + ","
                             + "\"min_p\":" + std::to_string(p.min_p) + ","
                             + "\"repetition_penalty\":" + std::to_string(p.repetition_penalty) + ","
                             + "\"model\":\"" + escape_json_string(current_model_name_.empty() ? "haven-chat-v5.0.gguf" : current_model_name_) + "\","
                             + "\"plugins_count\":" + std::to_string(haven_engine_.get_plugin_manager().get_plugin_count()) + "}";
            send_http_response(sock, "200 OK", "application/json; charset=utf-8", json, cors_hdr);
        }
        else {
            // Static file delivery with strict canonical root containment
            std::string static_path = path;
            size_t q_pos = static_path.find('?');
            if (q_pos != std::string::npos) static_path = static_path.substr(0, q_pos);

            std::string decoded_path = url_decode(static_path);
            if (decoded_path.find("..") != std::string::npos || decoded_path.find(':') != std::string::npos || 
                decoded_path.find('\\') != std::string::npos || decoded_path.find('\0') != std::string::npos) {
                send_http_response(sock, "403 Forbidden", "text/plain", "Forbidden");
                close_socket(sock);
                return;
            }

            if (decoded_path == "/" || decoded_path.empty()) decoded_path = "/index.html";
            std::string clean_rel = decoded_path.front() == '/' ? decoded_path.substr(1) : decoded_path;
            std::string rel_path = clean_rel;

            std::string local_path;
            if (!resolve_safe_path("wwwroot", clean_rel, local_path)) {
                send_http_response(sock, "403 Forbidden", "text/plain", "Forbidden");
                close_socket(sock);
                return;
            }

            FILE* f = fopen(local_path.c_str(), "rb");
            if (!f) {
                std::string fallback_path;
                if (resolve_safe_path("C:\\Users\\admin\\source\\haven-cpp\\wwwroot", clean_rel, fallback_path)) {
                    f = fopen(fallback_path.c_str(), "rb");
                }
            }

            if (f) {
                fseek(f, 0, SEEK_END);
                long sz = ftell(f);
                fseek(f, 0, SEEK_SET);

                if (sz < 0 || sz > 100L * 1024 * 1024) {
                    fclose(f);
                    close_socket(sock);
                    return;
                }

                std::vector<char> file_data(static_cast<size_t>(sz));
                size_t got = 0;
                if (sz > 0) {
                    got = fread(file_data.data(), 1, static_cast<size_t>(sz), f);
                }
                fclose(f);

                std::string mime = "application/octet-stream";
                if (rel_path.ends_with(".html")) mime = "text/html; charset=utf-8";
                else if (rel_path.ends_with(".png")) mime = "image/png";
                else if (rel_path.ends_with(".jpg") || rel_path.ends_with(".jpeg")) mime = "image/jpeg";
                else if (rel_path.ends_with(".webp")) mime = "image/webp";
                else if (rel_path.ends_with(".svg")) mime = "image/svg+xml";
                else if (rel_path.ends_with(".css")) mime = "text/css; charset=utf-8";
                else if (rel_path.ends_with(".js") || rel_path.ends_with(".mjs")) mime = "application/javascript; charset=utf-8";
                else if (rel_path.ends_with(".json")) mime = "application/json; charset=utf-8";
                else if (rel_path.ends_with(".wasm")) mime = "application/wasm";

                std::string header = "HTTP/1.1 200 OK\r\nContent-Type: " + mime +
                                     "\r\nX-Content-Type-Options: nosniff"
                                     "\r\nCache-Control: no-cache, no-store, must-revalidate\r\nPragma: no-cache\r\nExpires: 0\r\n"
                                     + cors_hdr +
                                     "Connection: close\r\nContent-Length: " +
                                     std::to_string(got) + "\r\n\r\n";
                send_all(sock, header.c_str(), static_cast<int>(header.length()));
                if (got > 0) {
                    send_all(sock, file_data.data(), static_cast<int>(got));
                }
            } else {
                std::string fallback_body = "{\"ok\":true,\"msg\":\"Haven Sovereign Micro-Server Active\"}";
                send_http_response(sock, "200 OK", "application/json; charset=utf-8", fallback_body, cors_hdr);
            }
        }

        close_socket(sock);
    }
};

} // namespace haven

namespace {
bool parse_u32_range(const char* s, uint32_t min_v, uint32_t max_v, uint32_t& out) {
    if (!s || !*s) return false;
    uint32_t v = 0;
    size_t len = std::strlen(s);
    auto [ptr, ec] = std::from_chars(s, s + len, v);
    if (ec != std::errc() || ptr != s + len || v < min_v || v > max_v) return false;
    out = v;
    return true;
}
} // namespace

int main(int argc, char** argv) {
    // Prevent OpenMP active-spin livelock under concurrent workstation load (user env vars still override)
#ifdef _WIN32
    if (!std::getenv("OMP_WAIT_POLICY")) _putenv_s("OMP_WAIT_POLICY", "PASSIVE");
    if (!std::getenv("KMP_BLOCKTIME"))   _putenv_s("KMP_BLOCKTIME", "0");
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#else
    if (!std::getenv("OMP_WAIT_POLICY")) setenv("OMP_WAIT_POLICY", "PASSIVE", 0);
    if (!std::getenv("KMP_BLOCKTIME"))   setenv("KMP_BLOCKTIME", "0", 0);
#endif
    int port = 11436;
    std::string bind_host = "127.0.0.1";
    std::string api_key = "";
    if (const char* env_key = std::getenv("HAVEN_API_KEY"); env_key && *env_key) {
        api_key = env_key;
    }
    std::string model_arg = "";
    bool is_worker = false;
    std::string cluster_join_target = "";
    uint16_t cluster_port = 0;
    std::string cluster_token = "";
    uint32_t cluster_node_id = 0;
    haven::TpuServerOptions tpu_opts;
    if (const char* env_tpu = std::getenv("HAVEN_TPU_ONCHIP"); env_tpu && (std::string(env_tpu) == "1" || std::string(env_tpu) == "true")) {
        tpu_opts.enable_onchip = true;
    }
    if (const char* env_prec = std::getenv("HAVEN_TPU_PRECISION"); env_prec && std::string(env_prec) == "bf16") {
        tpu_opts.use_int8 = false;
    }
    if (const char* env_rep = std::getenv("HAVEN_TPU_REPLICAS"); env_rep && *env_rep) {
        uint32_t r = 0;
        if (parse_u32_range(env_rep, 1, 4, r)) tpu_opts.num_replicas = static_cast<int>(r);
    }
    if (const char* env_batch = std::getenv("HAVEN_TPU_BATCH"); env_batch && *env_batch) {
        uint32_t b = 1;
        if (parse_u32_range(env_batch, 1, 256, b)) tpu_opts.batch_size = static_cast<int>(b);
    }
    if (const char* env_kvfp8 = std::getenv("HAVEN_TPU_KV_FP8"); env_kvfp8 && std::string(env_kvfp8) == "1") {
        tpu_opts.use_fp8_kv = true;
    }
    if (const char* env_lib = std::getenv("HAVEN_LIBTPU_PATH"); env_lib && *env_lib) {
        tpu_opts.libtpu_path = env_lib;
    }

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            std::cout << "Haven Sovereign Server & Featherlight Studio\n"
                      << "Usage: haven-server [options]\n\n"
                      << "Options:\n"
                      << "  --host, --bind <ip>       Bind IP address (default: 127.0.0.1; use 0.0.0.0 for LAN)\n"
                      << "  --port <port>             HTTP server port (default: 11436)\n"
                      << "  --api-key <key>           Require Authorization: Bearer <key> on API routes (or HAVEN_API_KEY)\n"
                      << "  --model <path>            GGUF or MLX model path to load\n"
                      << "  --tpu, --tpu-onchip       Serve dense/MoE transformers directly on Cloud TPU HBM3 via libtpu.so\n"
                      << "  --tpu-replicas <1|2|4|8>  Tensor-parallel TPU chips over Optical ICI (default: auto)\n"
                      << "  --tpu-batch <1..256>      Multi-tenant HBM3 KV slots (1=interactive, 16..256=continuous batching)\n"
                      << "  --tpu-bf16 / --tpu-int8   HBM3 weight precision (default: --tpu-int8 Hadamard W8)\n"
                      << "  --tpu-kv-fp8              Scale-free FP8 E4M3FN coalesced 3D KV cache (50% HBM3 footprint & BW)\n"
                      << "  --tpu-argmax              Use on-chip TPU argmax (4B/slot D2H) for peak multi-tenant tok/s\n"
                      << "  --tpu-max-seq <N>         Static HBM3 KV cache length (default: 2048)\n"
                      << "  --tpu-host-emb            Use host-side embedding lookup instead of on-chip HBM3 gather\n"
                      << "  --libtpu <path>           Explicit path to libtpu.so (optional)\n"
                      << "  --cluster-host [port]     Run as cluster coordinator on [port] (default: HTTP_PORT + 2)\n"
                      << "  --cluster-join <invite>   Join remote cluster using haven://cluster URI or IP\n"
                      << "  --cluster-port <port>     Coordinator port to join (default: 11438)\n"
                      << "  --cluster-token <key>     256-bit cryptographic pairing token\n"
                      << "  --node-id <id>            Cluster node ID (default: 0 coordinator, 1 worker)\n"
                      << "  --help, -h                Show this help message\n";
            return 0;
        } else if ((arg == "--host" || arg == "--bind") && i + 1 < argc) {
            bind_host = argv[++i];
        } else if (arg == "--api-key" && i + 1 < argc) {
            api_key = argv[++i];
        } else if (arg == "--tpu" || arg == "--tpu-onchip") {
            tpu_opts.enable_onchip = true;
        } else if (arg == "--tpu-bf16") {
            tpu_opts.enable_onchip = true;
            tpu_opts.use_int8 = false;
        } else if (arg == "--tpu-int8") {
            tpu_opts.enable_onchip = true;
            tpu_opts.use_int8 = true;
        } else if (arg == "--tpu-kv-fp8") {
            tpu_opts.enable_onchip = true;
            tpu_opts.use_fp8_kv = true;
        } else if (arg == "--tpu-argmax") {
            tpu_opts.enable_onchip = true;
            tpu_opts.use_onchip_argmax = true;
        } else if (arg == "--tpu-host-emb") {
            tpu_opts.use_onchip_embeddings = false;
        } else if (arg == "--tpu-batch" && i + 1 < argc) {
            uint32_t b = 1;
            if (!parse_u32_range(argv[++i], 1, 256, b)) {
                std::cerr << "[HavenServer] Invalid --tpu-batch (must be 1..256).\n";
                return 1;
            }
            tpu_opts.enable_onchip = true;
            tpu_opts.batch_size = static_cast<int>(b);
        } else if (arg == "--tpu-replicas" && i + 1 < argc) {
            uint32_t r = 0;
            if (!parse_u32_range(argv[++i], 1, 8, r) || (r != 1 && r != 2 && r != 4 && r != 8)) {
                std::cerr << "[HavenServer] Invalid --tpu-replicas (must be 1, 2, 4, or 8).\n";
                return 1;
            }
            tpu_opts.enable_onchip = true;
            tpu_opts.num_replicas = static_cast<int>(r);
        } else if (arg == "--tpu-max-seq" && i + 1 < argc) {
            uint32_t ms = 2048;
            if (!parse_u32_range(argv[++i], 256, 8192, ms)) {
                std::cerr << "[HavenServer] Invalid --tpu-max-seq (must be 256..8192).\n";
                return 1;
            }
            tpu_opts.max_seq_len = static_cast<int>(ms);
        } else if (arg == "--libtpu" && i + 1 < argc) {
            tpu_opts.libtpu_path = argv[++i];
        } else if (arg == "--port" && i + 1 < argc) {
            uint32_t p = 0;
            if (!parse_u32_range(argv[++i], 1, 65535, p)) {
                std::cerr << "[HavenServer] Invalid --port value.\n";
                return 1;
            }
            port = static_cast<int>(p);
        } else if (arg == "--model" && i + 1 < argc) {
            model_arg = argv[++i];
        } else if (arg == "--cluster-host") {
            is_worker = false;
            if (i + 1 < argc && argv[i+1][0] != '-') {
                uint32_t p = 0;
                if (!parse_u32_range(argv[++i], 1, 65535, p)) {
                    std::cerr << "[HavenServer] Invalid --cluster-host port.\n";
                    return 1;
                }
                cluster_port = static_cast<uint16_t>(p);
            }
        } else if (arg == "--cluster-join" && i + 1 < argc) {
            is_worker = true;
            cluster_join_target = argv[++i];
        } else if (arg == "--cluster-port" && i + 1 < argc) {
            uint32_t p = 0;
            if (!parse_u32_range(argv[++i], 1, 65535, p)) {
                std::cerr << "[HavenServer] Invalid --cluster-port value.\n";
                return 1;
            }
            cluster_port = static_cast<uint16_t>(p);
        } else if (arg == "--cluster-token" && i + 1 < argc) {
            cluster_token = argv[++i];
        } else if (arg == "--node-id" && i + 1 < argc) {
            uint32_t nid = 0;
            if (!parse_u32_range(argv[++i], 0, 65535, nid)) {
                std::cerr << "[HavenServer] Invalid --node-id value.\n";
                return 1;
            }
            cluster_node_id = nid;
        } else if (!arg.starts_with("-")) {
            uint32_t p = 0;
            if (parse_u32_range(arg.c_str(), 1, 65535, p)) {
                port = static_cast<int>(p);
            } else if (model_arg.empty()) {
                model_arg = arg;
            }
        }
    }

    haven::HavenMicroServer server(port, model_arg, is_worker, cluster_join_target,
                                   cluster_port, cluster_token, cluster_node_id, bind_host, api_key, tpu_opts);
    server.start();

    std::cout << "[HavenServer] Featherlight Studio Daemon running on http://" << bind_host << ":" << port << " (<10MB RAM)...\n" << std::flush;
    while (true) {
        std::this_thread::sleep_for(std::chrono::hours(24));
    }

    server.stop();
    return 0;
}
