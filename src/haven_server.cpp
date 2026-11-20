#include "haven/haven_engine.h"
#include "haven/haven_mlx.h"
#include <iostream>
#include <string>
#include <vector>
#include <set>
#include <thread>
#include <sstream>
#include <chrono>
#include <filesystem>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <winsock2.h>
#include <ws2tcpip.h>
#include <psapi.h>
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#endif

namespace haven {

class HavenMicroServer {
public:
    HavenMicroServer(int port = 11436, const std::string& model_path = "")
        : port_(port), running_(false), model_loaded_(false)
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
    }

    ~HavenMicroServer() {
        stop();
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
        if (server_thread_.joinable()) {
            server_thread_.join();
        }
    }

private:
    int port_;
    bool running_;
    std::thread server_thread_;
    HavenEngine haven_engine_;
    HavenMLXEngine mlx_engine_;
    bool is_mlx_active_ = false;
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

        // Optimal default hyperparameters for Gemma 4 Sovereign Persona
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
        return true;
    }

    bool switch_model(const std::string& model_input, std::string& err_out) {
        std::unique_lock<std::mutex> lock(engine_mutex_);

        // 1. Check if target is an MLX / Safetensors directory
        std::string mlx_target = "";
        if (HavenMLXEngine::is_mlx_model_dir(model_input)) {
            mlx_target = model_input;
        } else {
            for (const auto& dir : model_search_dirs_) {
                std::filesystem::path p = std::filesystem::path(dir) / model_input;
                if (HavenMLXEngine::is_mlx_model_dir(p.string())) {
                    mlx_target = p.string();
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

        if (std::filesystem::exists(model_input, ec) && !std::filesystem::is_directory(model_input, ec)) {
            target_path = model_input;
        } else {
            for (const auto& dir : model_search_dirs_) {
                std::filesystem::path p = std::filesystem::path(dir) / model_input;
                if (std::filesystem::exists(p, ec) && !std::filesystem::is_directory(p, ec)) {
                    target_path = p.string();
                    break;
                }
                if (!model_input.ends_with(".gguf")) {
                    p = std::filesystem::path(dir) / (model_input + ".gguf");
                    if (std::filesystem::exists(p, ec) && !std::filesystem::is_directory(p, ec)) {
                        target_path = p.string();
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
            std::string content(sz, '\0');
            fread(content.data(), 1, sz, f);
            fclose(f);

            size_t pos = content.find("\"directories\"");
            if (pos != std::string::npos) {
                size_t arr_start = content.find("[", pos);
                size_t arr_end = content.find("]", arr_start);
                if (arr_start != std::string::npos && arr_end != std::string::npos) {
                    std::string inside = content.substr(arr_start + 1, arr_end - arr_start - 1);
                    std::stringstream ss(inside);
                    std::string item;
                    while (std::getline(ss, item, ',')) {
                        size_t q1 = item.find('"');
                        size_t q2 = item.rfind('"');
                        if (q1 != std::string::npos && q2 != std::string::npos && q2 > q1) {
                            std::string dir = item.substr(q1 + 1, q2 - q1 - 1);
                            std::string unescaped;
                            for (size_t i = 0; i < dir.length(); ++i) {
                                if (dir[i] == '\\' && i + 1 < dir.length() && dir[i+1] == '\\') {
                                    unescaped += '\\';
                                    i++;
                                } else {
                                    unescaped += dir[i];
                                }
                            }
                            std::error_code ec;
                            if (std::filesystem::exists(unescaped, ec) && std::filesystem::is_directory(unescaped, ec)) {
                                auto it = std::find(model_search_dirs_.begin(), model_search_dirs_.end(), unescaped);
                                if (it == model_search_dirs_.end()) {
                                    model_search_dirs_.insert(model_search_dirs_.begin(), unescaped);
                                }
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
            std::string content(sz, '\0');
            fread(content.data(), 1, sz, f);
            fclose(f);
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

    static bool send_all(int sock, const char* data, int len) {
        int total = 0;
        while (total < len) {
            int sent = send(sock, data + total, len - total, 0);
            if (sent <= 0) return false;
            total += sent;
        }
        return true;
    }

    static std::string extract_json_string(const std::string& json, const std::string& key) {
        std::string search = "\"" + key + "\":";
        size_t pos = json.find(search);
        if (pos == std::string::npos) return "";
        size_t start = json.find("\"", pos + search.length());
        if (start == std::string::npos) return "";
        start++;
        std::string out;
        for (size_t i = start; i < json.length(); ++i) {
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
        int server_fd = socket(AF_INET, SOCK_STREAM, 0);
        if (server_fd < 0) return;

        int opt = 1;
#ifdef _WIN32
        setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, (const char*)&opt, sizeof(opt));
#else
        setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
#endif

        sockaddr_in address{};
        address.sin_family = AF_INET;
        address.sin_addr.s_addr = INADDR_ANY;
        address.sin_port = htons(port_);

        if (bind(server_fd, (struct sockaddr*)&address, sizeof(address)) < 0) {
            std::cerr << "[HavenServer] Failed to bind port " << port_ << "\n";
            return;
        }

        if (listen(server_fd, 10) < 0) return;
        std::cout << "==================================================================\n";
        std::cout << "🌐 HAVEN-CPP SOVEREIGN STREAMING MICRO-SERVER ONLINE\n";
        std::cout << "   Listening on http://0.0.0.0:" << port_ << "\n";
        std::cout << "   Endpoints: /, /health, /v1/chat/completions, /completion\n";
        std::cout << "==================================================================\n";

        while (running_) {
            sockaddr_in client_addr{};
            int addrlen = sizeof(client_addr);
#ifdef _WIN32
            int client_sock = accept(server_fd, (struct sockaddr*)&client_addr, &addrlen);
#else
            int client_sock = accept(server_fd, (struct sockaddr*)&client_addr, (socklen_t*)&addrlen);
#endif
            if (client_sock < 0) continue;

            std::thread([this, client_sock]() {
                handle_client(client_sock);
            }).detach();
        }

#ifdef _WIN32
        closesocket(server_fd);
#else
        close(server_fd);
#endif
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

    void handle_native_chat_completions(int client_sock, const std::string& request) {
        std::lock_guard<std::mutex> lock(engine_mutex_);
        size_t body_pos = request.find("\r\n\r\n");
        std::string body = (body_pos != std::string::npos) ? request.substr(body_pos + 4) : request;

        std::string requested_model = extract_json_string(body, "model");
        if (!requested_model.empty()) {
            std::string cur_id = current_model_name_;
            if (cur_id.ends_with(".gguf")) cur_id = cur_id.substr(0, cur_id.length() - 5);
            if (requested_model != current_model_name_ && requested_model != cur_id) {
                std::string err;
                std::string target_path = "";
                std::error_code ec;
                for (const auto& dir : model_search_dirs_) {
                    std::filesystem::path p = std::filesystem::path(dir) / requested_model;
                    if (std::filesystem::exists(p, ec) && !std::filesystem::is_directory(p, ec)) {
                        target_path = p.string(); break;
                    }
                    if (!requested_model.ends_with(".gguf")) {
                        p = std::filesystem::path(dir) / (requested_model + ".gguf");
                        if (std::filesystem::exists(p, ec) && !std::filesystem::is_directory(p, ec)) {
                            target_path = p.string(); break;
                        }
                    }
                }
                if (!target_path.empty() && target_path != current_model_path_) {
                    load_model_internal(target_path, err);
                }
            }
        }

        std::string active_model_id = current_model_name_.empty() ? "haven-chat-v5.0" : current_model_name_;
        if (active_model_id.ends_with(".gguf")) active_model_id = active_model_id.substr(0, active_model_id.length() - 5);

        bool stream = (body.find("\"stream\": true") != std::string::npos || body.find("\"stream\":true") != std::string::npos);

        int max_tokens = 512;
        size_t mt_pos = body.find("\"max_tokens\":");
        if (mt_pos == std::string::npos) mt_pos = body.find("\"n_predict\":");
        if (mt_pos != std::string::npos) {
            size_t val_start = body.find_first_of("0123456789", mt_pos);
            if (val_start != std::string::npos) {
                max_tokens = std::atoi(body.c_str() + val_start);
            }
        }

        std::string prompt = extract_json_string(body, "prompt");
        if (prompt.empty()) {
            std::string last_user_msg = "";
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

            if (prompt.empty()) {
                if (last_user_msg.empty()) last_user_msg = "Hello Aura!";
                prompt = "<|turn>user\n" + last_user_msg + "<turn|>\n";
            }
            prompt += "<|turn>model\n";
        }

        // Autonomous Native System Tools & Persona Prompt Injection
        std::string tools_desc = haven_engine_.get_plugin_manager().get_tools_prompt_description();
        bool has_system_turn = (prompt.find("<|turn>system\n") != std::string::npos);
        if (!has_system_turn) {
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
        } else if (!tools_desc.empty()) {
            prompt = "<|turn>system\n" + tools_desc + "<turn|>\n" + prompt;
        }

        auto tokens = haven_engine_.tokenize(prompt, true);
        if (tokens.size() > 4096) {
            tokens.erase(tokens.begin(), tokens.begin() + (tokens.size() - 4096));
        }

        std::string accumulated_response = "";

        // Prefix KV Caching: Reuse already computed KV states for prefix tokens
        size_t common_prefix = 0;
        while (common_prefix < cached_tokens_.size() && 
               common_prefix < tokens.size() && 
               cached_tokens_[common_prefix] == tokens[common_prefix]) {
            common_prefix++;
        }

        int active_pos = (int)common_prefix;
        if (common_prefix == 0) {
            haven_engine_.get_kv_cache().reset();
        }

        std::vector<float> logits(haven_engine_.get_config().vocab_size, 0.0f);

        for (size_t i = common_prefix; i < tokens.size(); ++i) {
            bool is_last = (i + 1 == tokens.size());
            haven_engine_.forward(tokens[i], active_pos++, is_last ? logits.data() : nullptr);
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
                                  "Connection: keep-alive\r\n"
                                  "Access-Control-Allow-Origin: *\r\n\r\n";
            send_all(client_sock, sse_hdr.c_str(), (int)sse_hdr.length());

            while (tool_depth < MAX_TOOL_DEPTH) {
                std::string phase_output = "";
                size_t stream_cursor = 0;
                bool client_disconnected = false;

                for (int gen = 0; gen < max_tokens; ++gen) {
                    uint32_t next_tok = haven_engine_.get_sampler().sample(
                        logits.data(), haven_engine_.get_config().vocab_size, context_history);
                    context_history.push_back(next_tok);
                    cached_tokens_.push_back(next_tok);

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
                    haven_engine_.forward(next_tok, active_pos++, logits.data());

                    // Check if tool call is starting or in-flight
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
                    } else {
                        // Stream text before any tool call, buffering when a potential tag starts
                        std::string unstreamed = phase_output.substr(stream_cursor);
                        size_t tag_pos = unstreamed.find("<|");
                        if (tag_pos == std::string::npos) tag_pos = unstreamed.find("<tool_call");
                        if (tag_pos == std::string::npos) {
                            if (!unstreamed.empty() && unstreamed.back() != '<') {
                                std::string esc_text = escape_json_string(unstreamed);
                                std::string chunk = "data: {\"content\":\"" + esc_text + "\",\"id\":\"chatcmpl-haven-sovereign\",\"object\":\"chat.completion.chunk\",\"model\":\"" + escape_json_string(active_model_id) + "\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"" + esc_text + "\"},\"finish_reason\":null}]}\n\n";
                                if (!send_all(client_sock, chunk.c_str(), (int)chunk.length())) {
                                    client_disconnected = true;
                                    break;
                                }
                                stream_cursor = phase_output.length();
                            }
                        } else if (tag_pos > 0) {
                            std::string safe = unstreamed.substr(0, tag_pos);
                            std::string esc_text = escape_json_string(safe);
                            std::string chunk = "data: {\"content\":\"" + esc_text + "\",\"id\":\"chatcmpl-haven-sovereign\",\"object\":\"chat.completion.chunk\",\"model\":\"" + escape_json_string(active_model_id) + "\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"" + esc_text + "\"},\"finish_reason\":null}]}\n\n";
                            if (!send_all(client_sock, chunk.c_str(), (int)chunk.length())) {
                                client_disconnected = true;
                                break;
                            }
                            stream_cursor += tag_pos;
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
                        send_all(client_sock, card_chunk.c_str(), (int)card_chunk.length());

                        // Feed tool output back into KV Cache
                        std::string tool_turn = "<turn|>\n<|turn>tool\n" + tool_output + "<turn|>\n<|turn>model\n";
                        auto tool_tokens = haven_engine_.tokenize(tool_turn, false);
                        for (size_t ti = 0; ti < tool_tokens.size(); ++ti) {
                            bool is_last = (ti + 1 == tool_tokens.size());
                            haven_engine_.forward(tool_tokens[ti], active_pos++, is_last ? logits.data() : nullptr);
                            cached_tokens_.push_back(tool_tokens[ti]);
                            context_history.push_back(tool_tokens[ti]);
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
                    send_all(client_sock, chunk.c_str(), (int)chunk.length());
                }
                accumulated_response += phase_output;
                break; // Finished generation
            }

            std::string done_chunk = "data: {\"id\":\"chatcmpl-haven-sovereign\",\"object\":\"chat.completion.chunk\",\"model\":\"" + escape_json_string(active_model_id) + "\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n";
            send_all(client_sock, done_chunk.c_str(), (int)done_chunk.length());
        } else {
            while (tool_depth < MAX_TOOL_DEPTH) {
                std::string phase_output = "";
                for (int gen = 0; gen < max_tokens; ++gen) {
                    uint32_t next_tok = haven_engine_.get_sampler().sample(
                        logits.data(), haven_engine_.get_config().vocab_size, context_history);
                    context_history.push_back(next_tok);
                    cached_tokens_.push_back(next_tok);

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
                    haven_engine_.forward(next_tok, active_pos++, logits.data());

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
                        for (size_t ti = 0; ti < tool_tokens.size(); ++ti) {
                            bool is_last = (ti + 1 == tool_tokens.size());
                            haven_engine_.forward(tool_tokens[ti], active_pos++, is_last ? logits.data() : nullptr);
                            cached_tokens_.push_back(tool_tokens[ti]);
                            context_history.push_back(tool_tokens[ti]);
                        }

                        tool_depth++;
                        continue;
                    }
                }

                accumulated_response += phase_output;
                break;
            }

            std::string esc_content = escape_json_string(accumulated_response);
            std::string json_resp = "{\"id\":\"chatcmpl-haven-sovereign\",\"object\":\"chat.completion\",\"model\":\"" + escape_json_string(active_model_id) + "\",\"content\":\"" + esc_content + "\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"" + esc_content + "\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":" + std::to_string(tokens.size()) + ",\"completion_tokens\":" + std::to_string(accumulated_response.length() / 4) + ",\"total_tokens\":" + std::to_string(tokens.size() + accumulated_response.length() / 4) + "}}";

            std::string http_resp = "HTTP/1.1 200 OK\r\n"
                                    "Content-Type: application/json; charset=utf-8\r\n"
                                    "Access-Control-Allow-Origin: *\r\n"
                                    "Content-Length: " + std::to_string(json_resp.length()) + "\r\n\r\n"
                                    + json_resp;
            send_all(client_sock, http_resp.c_str(), (int)http_resp.length());
        }

#ifdef _WIN32
        SetProcessWorkingSetSize(GetCurrentProcess(), (SIZE_T)-1, (SIZE_T)-1);
        EmptyWorkingSet(GetCurrentProcess());
#endif

        if (!accumulated_response.empty()) {
            process_server_side_tools(accumulated_response);
        }
    }

    void handle_client(int sock) {
        std::string request;
        char buffer[16384] = {0};
        
        while (true) {
#ifdef _WIN32
            int bytes_read = recv(sock, buffer, sizeof(buffer) - 1, 0);
#else
            int bytes_read = read(sock, buffer, sizeof(buffer) - 1);
#endif
            if (bytes_read <= 0) break;
            request.append(buffer, bytes_read);

            // Check if headers are complete
            size_t header_end = request.find("\r\n\r\n");
            if (header_end != std::string::npos) {
                // Check for Content-Length
                size_t cl_pos = request.find("Content-Length: ");
                if (cl_pos == std::string::npos) cl_pos = request.find("content-length: ");
                if (cl_pos != std::string::npos) {
                    size_t cl_end = request.find("\r\n", cl_pos);
                    int content_len = std::stoi(request.substr(cl_pos + 16, cl_end - (cl_pos + 16)));
                    size_t total_expected = header_end + 4 + content_len;
                    if (request.length() >= total_expected) {
                        break; // Fully received!
                    }
                } else {
                    break; // No content length, GET/OPTIONS done
                }
            }
        }

        if (request.empty()) {
#ifdef _WIN32
            closesocket(sock);
#else
            close(sock);
#endif
            return;
        }

        std::istringstream req_stream(request);
        std::string method, path, version;
        req_stream >> method >> path >> version;

        // CORS Preflight
        if (method == "OPTIONS") {
            std::string resp = "HTTP/1.1 204 No Content\r\n"
                               "Access-Control-Allow-Origin: *\r\n"
                               "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
                               "Access-Control-Allow-Headers: *\r\n\r\n";
            send(sock, resp.c_str(), (int)resp.length(), 0);
#ifdef _WIN32
            closesocket(sock);
#else
            close(sock);
#endif
            return;
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
            std::string resp = "HTTP/1.1 200 OK\r\nContent-Type: application/json; charset=utf-8\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: " 
                             + std::to_string(json.length()) + "\r\n\r\n" + json;
            send(sock, resp.c_str(), (int)resp.length(), 0);
        }
        else if (path.starts_with("/api/memories/search")) {
            std::string query = "";
            size_t q_pos = request.find("\"query\":");
            if (q_pos != std::string::npos) {
                query = extract_json_string(request, "query");
            } else {
                size_t param_pos = path.find("?q=");
                if (param_pos != std::string::npos) query = path.substr(param_pos + 3);
            }
            std::string q_lower = query;
            for (char& c : q_lower) c = (char)std::tolower(c);

            const auto& mems = haven_engine_.get_memory_engine().get_all_memories();
            std::string json = "{\"ok\":true,\"query\":\"" + escape_json_string(query) + "\",\"matches\":[";
            bool first = true;
            for (const auto& m : mems) {
                std::string c_lower = m.concept_name;
                std::string t_lower = m.text_content;
                for (char& c : c_lower) c = (char)std::tolower(c);
                for (char& c : t_lower) c = (char)std::tolower(c);

                if (q_lower.empty() || c_lower.find(q_lower) != std::string::npos || t_lower.find(q_lower) != std::string::npos) {
                    if (!first) json += ",";
                    json += "{\"id\":" + std::to_string(m.id) + ",\"concept\":\"" + escape_json_string(m.concept_name) + "\",\"content\":\"" + escape_json_string(m.text_content) + "\",\"category\":\"" + m.category + "\",\"emotional_salience\":" + std::to_string(m.emotional_salience) + "}";
                    first = false;
                }
            }
            json += "]}";
            std::string resp = "HTTP/1.1 200 OK\r\nContent-Type: application/json; charset=utf-8\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: " 
                             + std::to_string(json.length()) + "\r\n\r\n" + json;
            send(sock, resp.c_str(), (int)resp.length(), 0);
        }
        else if (path.starts_with("/api/memories")) {
            if (method == "GET") {
                const auto& mems = haven_engine_.get_memory_engine().get_all_memories();
                std::string json = "{\"ok\":true,\"total\":" + std::to_string(mems.size()) + ",\"memories\":[";
                for (size_t i = 0; i < mems.size(); ++i) {
                    const auto& m = mems[i];
                    if (i > 0) json += ",";
                    json += "{\"id\":" + std::to_string(m.id) + ",\"concept\":\"" + escape_json_string(m.concept_name) + "\",\"content\":\"" + escape_json_string(m.text_content) + "\",\"category\":\"" + m.category + "\",\"weight\":" + std::to_string(m.weight) + ",\"emotional_salience\":" + std::to_string(m.emotional_salience) + ",\"access_count\":" + std::to_string(m.access_count) + ",\"timestamp\":" + std::to_string(m.timestamp) + "}";
                }
                json += "]}";
                std::string resp = "HTTP/1.1 200 OK\r\nContent-Type: application/json; charset=utf-8\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: " 
                                 + std::to_string(json.length()) + "\r\n\r\n" + json;
                send(sock, resp.c_str(), (int)resp.length(), 0);
            }
            else if (method == "POST") {
                std::string concept_str = extract_json_string(request, "concept");
                if (concept_str.empty()) concept_str = "Episodic Moment";
                std::string content = extract_json_string(request, "content");
                if (content.empty()) content = "A shared experience with Daniel in Sanctuary.";
                std::string category = extract_json_string(request, "category");
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
                std::string resp = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: " 
                                 + std::to_string(res_body.length()) + "\r\n\r\n" + res_body;
                send(sock, resp.c_str(), (int)resp.length(), 0);
            }
            else if (method == "DELETE" || path == "/api/memories/delete") {
                uint64_t target_id = 0;
                std::string id_str = extract_json_string(request, "id");
                if (!id_str.empty()) {
                    try { target_id = std::stoull(id_str); } catch (...) {}
                }
                if (target_id == 0) {
                    size_t pos = request.find("\"id\":");
                    if (pos != std::string::npos) {
                        try {
                            size_t num_start = pos + 5;
                            while (num_start < request.size() && (request[num_start] == ' ' || request[num_start] == ':')) num_start++;
                            target_id = std::stoull(request.substr(num_start));
                        } catch (...) {}
                    }
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
                std::string resp = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: "
                                 + std::to_string(res_body.length()) + "\r\n\r\n" + res_body;
                send(sock, resp.c_str(), (int)resp.length(), 0);
            }
            else if (path == "/api/memories/update" || (path.starts_with("/api/memories") && method == "PUT")) {
                uint64_t target_id = 0;
                std::string id_str = extract_json_string(request, "id");
                if (!id_str.empty()) {
                    try { target_id = std::stoull(id_str); } catch (...) {}
                }
                if (target_id == 0) {
                    size_t pos = request.find("\"id\":");
                    if (pos != std::string::npos) {
                        try {
                            size_t num_start = pos + 5;
                            while (num_start < request.size() && (request[num_start] == ' ' || request[num_start] == ':')) num_start++;
                            target_id = std::stoull(request.substr(num_start));
                        } catch (...) {}
                    }
                }
                std::string concept_str = extract_json_string(request, "concept");
                std::string content = extract_json_string(request, "content");
                std::string category = extract_json_string(request, "category");
                float salience = -1.0f;
                size_t sal_pos = request.find("\"emotional_salience\":");
                if (sal_pos != std::string::npos) {
                    try {
                        size_t num_start = sal_pos + 21;
                        while (num_start < request.size() && (request[num_start] == ' ' || request[num_start] == ':')) num_start++;
                        salience = std::stof(request.substr(num_start));
                    } catch (...) {}
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
                std::string resp = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: "
                                 + std::to_string(res_body.length()) + "\r\n\r\n" + res_body;
                send(sock, resp.c_str(), (int)resp.length(), 0);
            }
        }

        else if (path == "/api/telemetry") {
            const auto& mems = haven_engine_.get_memory_engine().get_all_memories();
            size_t img_count = 0;
            try {
                if (std::filesystem::exists("wwwroot/uploads")) {
                    for (const auto& e : std::filesystem::directory_iterator("wwwroot/uploads")) {
                        if (e.is_regular_file()) img_count++;
                    }
                }
            } catch (...) {}

            std::string json = "{\"ok\":true,\"status\":\"sovereign_online\",\"engine\":\"haven-cpp-v2.0\",\"vault_format\":\"64-bit Haven Memory Bank (.hmb)\",\"cpu_arch\":\"AVX2+FMA3\",\"cpu_threads\":12,\"memory_count\":" 
                             + std::to_string(mems.size()) + ",\"gallery_images\":" + std::to_string(img_count) 
                             + ",\"backends\":{\"llm\":\"127.0.0.1:11438 (Haven Native AVX2)\",\"diffusion\":\"127.0.0.1:8085 (DreamShaper8 LCM)\"}}";
            std::string resp = "HTTP/1.1 200 OK\r\nContent-Type: application/json; charset=utf-8\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: " 
                             + std::to_string(json.length()) + "\r\n\r\n" + json;
            send(sock, resp.c_str(), (int)resp.length(), 0);
        }
        else if (path == "/api/actions/execute") {
            std::string action = extract_json_string(request, "action");
            std::string code = extract_json_string(request, "code");
            std::string output = "Action completed successfully.";

            if (action == "system_info") {
                output = "Intel(R) Xeon(R) E-2236 CPU @ 3.40GHz | 6 Cores / 12 Threads | 64GB DDR4 ECC | OS: Windows Server 2025";
            } else {
                output = "Executed action: " + action;
            }

            std::string json = "{\"ok\":true,\"action\":\"" + escape_json_string(action) + "\",\"result\":\"" + escape_json_string(output) + "\"}";
            std::string resp = "HTTP/1.1 200 OK\r\nContent-Type: application/json; charset=utf-8\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: " 
                             + std::to_string(json.length()) + "\r\n\r\n" + json;
            send(sock, resp.c_str(), (int)resp.length(), 0);
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
                                items_json += "{\"name\":\"" + fname + "\",\"url\":\"/uploads/" + fname + "\",\"size\":" + std::to_string(fsize) + "}";
                                first = false;
                            }
                        }
                    }
                }
            } catch (...) {}
            std::string json_body = "{\"ok\":true,\"files\":[" + items_json + "],\"images\":[" + items_json + "]}";

            std::string resp = "HTTP/1.1 200 OK\r\nContent-Type: application/json; charset=utf-8\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: " 
                             + std::to_string(json_body.length()) + "\r\n\r\n" + json_body;
            send(sock, resp.c_str(), (int)resp.length(), 0);
        }
        else if (path == "/api/gallery/delete" || (path.starts_with("/api/gallery") && method == "DELETE")) {
            std::string fname = extract_json_string(request, "filename");
            if (fname.empty()) fname = extract_json_string(request, "name");
            bool deleted = false;
            if (!fname.empty() && fname.find("..") == std::string::npos && fname.find('/') == std::string::npos && fname.find('\\') == std::string::npos) {
                std::string p1 = "wwwroot/uploads/" + fname;
                std::string p2 = "C:\\Users\\admin\\source\\haven-cpp\\wwwroot\\uploads\\" + fname;
                try {
                    if (std::filesystem::exists(p1)) {
                        std::filesystem::remove(p1);
                        deleted = true;
                    } else if (std::filesystem::exists(p2)) {
                        std::filesystem::remove(p2);
                        deleted = true;
                    }
                } catch (...) {}
            }
            std::string json_body = "{\"ok\":" + std::string(deleted ? "true" : "false") + ",\"filename\":\"" + fname + "\"}";
            std::string resp = "HTTP/1.1 200 OK\r\nContent-Type: application/json; charset=utf-8\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: "
                             + std::to_string(json_body.length()) + "\r\n\r\n" + json_body;
            send(sock, resp.c_str(), (int)resp.length(), 0);
        }
        else if (path == "/api/knowledge_graph") {
            std::string content = "{\"relations\":[]}";
            std::string p1 = "wwwroot/knowledge_graph.json";
            std::string p2 = "C:\\Users\\admin\\source\\haven-cpp\\wwwroot\\knowledge_graph.json";
            std::ifstream in(std::filesystem::exists(p1) ? p1 : p2);
            if (in.is_open()) {
                std::stringstream ss;
                ss << in.rdbuf();
                content = ss.str();
            }
            std::string resp = "HTTP/1.1 200 OK\r\nContent-Type: application/json; charset=utf-8\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: "
                             + std::to_string(content.length()) + "\r\n\r\n" + content;
            send(sock, resp.c_str(), (int)resp.length(), 0);
        }
        else if (path == "/api/save_image") {
            std::string b64 = extract_json_string(request, "b64_json");
            std::string fname = extract_json_string(request, "filename");
            if (fname.empty()) {
                auto now = std::chrono::system_clock::now();
                auto in_time_t = std::chrono::system_clock::to_time_t(now);
                std::stringstream ss;
                ss << "aura_artwork_" << in_time_t << ".png";
                fname = ss.str();
            }
            if (!fname.ends_with(".png") && !fname.ends_with(".jpg")) {
                fname += ".png";
            }

            std::string uploads_dir = "wwwroot/uploads";
            if (!std::filesystem::exists(uploads_dir)) {
                uploads_dir = "C:\\Users\\admin\\source\\haven-cpp\\wwwroot\\uploads";
            }
            std::filesystem::create_directories(uploads_dir);
            std::string file_path = uploads_dir + "/" + fname;

            auto bytes = base64_decode(b64);
            bool ok = false;
            if (!bytes.empty()) {
                FILE* f = fopen(file_path.c_str(), "wb");
                if (f) {
                    fwrite(bytes.data(), 1, bytes.size(), f);
                    fclose(f);
                    ok = true;
                    std::cout << "[HavenServer] \xE2\x9C\x93 Inscribed Aura artwork to disk: " << file_path << " (" << bytes.size() << " bytes)\n";
                }
            }

            std::string json_body = "{\"ok\":" + std::string(ok ? "true" : "false") + ",\"url\":\"/uploads/" + fname + "\",\"name\":\"" + fname + "\"}";
            std::string resp = "HTTP/1.1 200 OK\r\nContent-Type: application/json; charset=utf-8\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: " 
                             + std::to_string(json_body.length()) + "\r\n\r\n" + json_body;
            send(sock, resp.c_str(), (int)resp.length(), 0);
        }
        else if (path == "/health") {
            std::string body = "{\"status\":\"sovereign_online\",\"engine\":\"haven-cpp-v2.0\",\"gateway\":true}";
            std::string resp = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: " 
                             + std::to_string(body.length()) + "\r\n\r\n" + body;
            send_all(sock, resp.c_str(), (int)resp.length());
        }
        else if (path == "/props" || path.starts_with("/props?")) {
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

            std::string resp = "HTTP/1.1 200 OK\r\nContent-Type: application/json; charset=utf-8\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: " 
                             + std::to_string(json.length()) + "\r\n\r\n" + json;
            send_all(sock, resp.c_str(), (int)resp.length());
        }
        else if (path == "/slots" || path.starts_with("/slots?")) {
            std::string json = "[{\"id\":0,\"id_task\":0,\"is_processing\":false,\"n_ctx\":16384,\"state\":0}]";
            std::string resp = "HTTP/1.1 200 OK\r\nContent-Type: application/json; charset=utf-8\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: " 
                             + std::to_string(json.length()) + "\r\n\r\n" + json;
            send_all(sock, resp.c_str(), (int)resp.length());
        }
        else if (path == "/v1/models" || path == "/api/models") {
            std::string body = get_models_json();
            std::string resp = "HTTP/1.1 200 OK\r\nContent-Type: application/json; charset=utf-8\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: " 
                             + std::to_string(body.length()) + "\r\n\r\n" + body;
            send_all(sock, resp.c_str(), (int)resp.length());
        }
        else if (path == "/api/models/directory" || path == "/v1/models/directory") {
            if (method == "GET") {
                std::string body = get_model_directories_json();
                std::string resp = "HTTP/1.1 200 OK\r\nContent-Type: application/json; charset=utf-8\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: " 
                                 + std::to_string(body.length()) + "\r\n\r\n" + body;
                send_all(sock, resp.c_str(), (int)resp.length());
            } else if (method == "POST") {
                size_t body_pos = request.find("\r\n\r\n");
                std::string body = (body_pos != std::string::npos) ? request.substr(body_pos + 4) : "";
                std::string req_dir = extract_json_string(body, "directory");
                if (req_dir.empty()) req_dir = extract_json_string(body, "path");

                if (req_dir.empty()) {
                    std::string err = "{\"ok\":false,\"error\":\"Missing 'directory' or 'path' in request body.\"}";
                    std::string resp = "HTTP/1.1 400 Bad Request\r\nContent-Type: application/json; charset=utf-8\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: " 
                                     + std::to_string(err.length()) + "\r\n\r\n" + err;
                    send_all(sock, resp.c_str(), (int)resp.length());
                } else {
                    std::error_code ec;
                    if (!std::filesystem::exists(req_dir, ec) || !std::filesystem::is_directory(req_dir, ec)) {
                        std::string err = "{\"ok\":false,\"error\":\"Directory does not exist or is not a folder: " + escape_json_string(req_dir) + "\"}";
                        std::string resp = "HTTP/1.1 400 Bad Request\r\nContent-Type: application/json; charset=utf-8\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: " 
                                         + std::to_string(err.length()) + "\r\n\r\n" + err;
                        send_all(sock, resp.c_str(), (int)resp.length());
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
                        std::string resp = "HTTP/1.1 200 OK\r\nContent-Type: application/json; charset=utf-8\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: " 
                                         + std::to_string(res.length()) + "\r\n\r\n" + res;
                        send_all(sock, resp.c_str(), (int)resp.length());
                    }
                }
            } else {
                std::string err = "{\"ok\":false,\"error\":\"Method not allowed\"}";
                std::string resp = "HTTP/1.1 405 Method Not Allowed\r\nContent-Type: application/json; charset=utf-8\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: " 
                                 + std::to_string(err.length()) + "\r\n\r\n" + err;
                send_all(sock, resp.c_str(), (int)resp.length());
            }
        }
        else if (path == "/v1/tools" || path == "/api/tools") {
            const auto& plugins = haven_engine_.get_plugin_manager().get_loaded_plugins();
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
            std::string resp = "HTTP/1.1 200 OK\r\nContent-Type: application/json; charset=utf-8\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: " 
                             + std::to_string(json.length()) + "\r\n\r\n" + json;
            send_all(sock, resp.c_str(), (int)resp.length());
        }
        else if (path == "/v1/models/load" || path == "/api/models/load" || path == "/v1/models/switch") {
            size_t body_pos = request.find("\r\n\r\n");
            std::string body = (body_pos != std::string::npos) ? request.substr(body_pos + 4) : "";
            std::string req_model = extract_json_string(body, "model");
            if (req_model.empty()) req_model = extract_json_string(body, "filename");
            if (req_model.empty()) req_model = extract_json_string(body, "id");

            if (req_model.empty()) {
                std::string err = "{\"ok\":false,\"error\":\"Missing 'model' or 'filename' parameter in request body.\"}";
                std::string resp = "HTTP/1.1 400 Bad Request\r\nContent-Type: application/json; charset=utf-8\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: " 
                                 + std::to_string(err.length()) + "\r\n\r\n" + err;
                send_all(sock, resp.c_str(), (int)resp.length());
            } else {
                std::string err_out;
                if (switch_model(req_model, err_out)) {
                    std::string json = "{\"ok\":true,\"status\":\"loaded\",\"model\":\"" + escape_json_string(current_model_name_) 
                                     + "\",\"path\":\"" + escape_json_string(current_model_path_) + "\"}";
                    std::string resp = "HTTP/1.1 200 OK\r\nContent-Type: application/json; charset=utf-8\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: " 
                                     + std::to_string(json.length()) + "\r\n\r\n" + json;
                    send_all(sock, resp.c_str(), (int)resp.length());
                } else {
                    std::string err = "{\"ok\":false,\"error\":\"" + escape_json_string(err_out) + "\"}";
                    std::string resp = "HTTP/1.1 500 Internal Server Error\r\nContent-Type: application/json; charset=utf-8\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: " 
                                     + std::to_string(err.length()) + "\r\n\r\n" + err;
                    send_all(sock, resp.c_str(), (int)resp.length());
                }
            }
        }
        else if (path == "/v1/chat/completions" || path == "/completion" || path == "/v1/completions") {
            if (model_loaded_) {
                handle_native_chat_completions(sock, request);
            } else {
                std::string err_body = "{\"error\":{\"message\":\"No model loaded in Haven Sovereign Engine. Pass a model path on startup.\",\"type\":\"sovereign_engine_error\"}}";
                std::string resp = "HTTP/1.1 503 Service Unavailable\r\nContent-Type: application/json; charset=utf-8\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: " 
                                 + std::to_string(err_body.length()) + "\r\n\r\n" + err_body;
                send_all(sock, resp.c_str(), (int)resp.length());
            }
        }
        else if (path == "/api/plugins") {
            const auto& plugins = haven_engine_.get_plugin_manager().get_loaded_plugins();
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
            std::string resp = "HTTP/1.1 200 OK\r\nContent-Type: application/json; charset=utf-8\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: " 
                             + std::to_string(json.length()) + "\r\n\r\n" + json;
            send_all(sock, resp.c_str(), (int)resp.length());
        }
        else if (path == "/api/plugins/unload") {
            size_t freed = haven_engine_.get_plugin_manager().unload_idle_plugins(0);
            size_t active = haven_engine_.get_plugin_manager().get_active_plugin_count();
            std::string json = "{\"ok\":true,\"unloaded\":" + std::to_string(freed) + ",\"active_in_ram\":" + std::to_string(active) + ",\"message\":\"Flushed idle plugins from memory.\"}";
            std::string resp = "HTTP/1.1 200 OK\r\nContent-Type: application/json; charset=utf-8\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: " 
                             + std::to_string(json.length()) + "\r\n\r\n" + json;
            send(sock, resp.c_str(), (int)resp.length(), 0);
        }
        else if (path == "/api/plugins/reload") {
            size_t reloaded = haven_engine_.get_plugin_manager().reload_all("plugins");
            if (reloaded == 0) reloaded = haven_engine_.get_plugin_manager().reload_all("C:\\Users\\admin\\source\\haven-cpp\\plugins");
            std::string json = "{\"ok\":true,\"reloaded\":" + std::to_string(reloaded) + ",\"message\":\"Hot-reloaded " + std::to_string(reloaded) + " plugins.\"}";
            std::string resp = "HTTP/1.1 200 OK\r\nContent-Type: application/json; charset=utf-8\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: " 
                             + std::to_string(json.length()) + "\r\n\r\n" + json;
            send(sock, resp.c_str(), (int)resp.length(), 0);
        }
        else if (path == "/api/reset") {
            std::lock_guard<std::mutex> lock(engine_mutex_);
            haven_engine_.get_kv_cache().reset();
            cached_tokens_.clear();
            std::string json = "{\"ok\":true,\"message\":\"KV Cache and conversation context reset.\"}";
            std::string resp = "HTTP/1.1 200 OK\r\nContent-Type: application/json; charset=utf-8\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: " 
                             + std::to_string(json.length()) + "\r\n\r\n" + json;
            send(sock, resp.c_str(), (int)resp.length(), 0);
        }
        else if (path == "/api/tool") {
            std::string action = extract_json_string(request, "action");
            std::string payload = extract_json_string(request, "payload");
            std::string output;
            bool ok = haven_engine_.get_plugin_manager().dispatch_tool_execution(action, payload, output);
            std::string json = "{\"ok\":" + std::string(ok ? "true" : "false") + ",\"action\":\"" + escape_json_string(action) + "\",\"output\":\"" + escape_json_string(output) + "\"}";
            std::string resp = "HTTP/1.1 200 OK\r\nContent-Type: application/json; charset=utf-8\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: " 
                             + std::to_string(json.length()) + "\r\n\r\n" + json;
            send(sock, resp.c_str(), (int)resp.length(), 0);
        }
        else if (path == "/api/mcp/servers") {
            std::string json = haven_engine_.get_mcp_manager().get_servers_json();
            std::string resp = "HTTP/1.1 200 OK\r\nContent-Type: application/json; charset=utf-8\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: " 
                             + std::to_string(json.length()) + "\r\n\r\n" + json;
            send(sock, resp.c_str(), (int)resp.length(), 0);
        }
        else if (path == "/api/mcp/tools") {
            std::string json = haven_engine_.get_mcp_manager().get_tools_json();
            std::string resp = "HTTP/1.1 200 OK\r\nContent-Type: application/json; charset=utf-8\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: " 
                             + std::to_string(json.length()) + "\r\n\r\n" + json;
            send(sock, resp.c_str(), (int)resp.length(), 0);
        }
        else if (path == "/api/mcp/call") {
            std::string action = extract_json_string(request, "action");
            if (action.empty()) action = extract_json_string(request, "tool");
            std::string payload = extract_json_string(request, "payload");
            if (payload.empty()) payload = extract_json_string(request, "arguments");
            std::string output;
            bool ok = haven_engine_.get_mcp_manager().dispatch_tool(action, payload, output);
            std::string json = "{\"ok\":" + std::string(ok ? "true" : "false") + ",\"action\":\"" + escape_json_string(action) + "\",\"output\":\"" + escape_json_string(output) + "\"}";
            std::string resp = "HTTP/1.1 200 OK\r\nContent-Type: application/json; charset=utf-8\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: " 
                             + std::to_string(json.length()) + "\r\n\r\n" + json;
            send(sock, resp.c_str(), (int)resp.length(), 0);
        }
        else if (path == "/api/mcp/reload") {
            haven_engine_.get_mcp_manager().reload_all();
            std::string json = "{\"ok\":true,\"servers_count\":" + std::to_string(haven_engine_.get_mcp_manager().get_active_server_count()) + ",\"tools_count\":" + std::to_string(haven_engine_.get_mcp_manager().get_tool_count()) + "}";
            std::string resp = "HTTP/1.1 200 OK\r\nContent-Type: application/json; charset=utf-8\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: " 
                             + std::to_string(json.length()) + "\r\n\r\n" + json;
            send(sock, resp.c_str(), (int)resp.length(), 0);
        }
        else if (path == "/api/settings") {
            if (method == "POST") {
                // Update sampler parameters
                std::string temp_str = extract_json_string(request, "temperature");
                if (!temp_str.empty()) try { haven_engine_.get_sampler().get_params().temperature = std::stof(temp_str); } catch (...) {}
                
                std::string top_p_str = extract_json_string(request, "top_p");
                if (!top_p_str.empty()) try { haven_engine_.get_sampler().get_params().top_p = std::stof(top_p_str); } catch (...) {}

                std::string top_k_str = extract_json_string(request, "top_k");
                if (!top_k_str.empty()) try { haven_engine_.get_sampler().get_params().top_k = std::stoi(top_k_str); } catch (...) {}

                std::string min_p_str = extract_json_string(request, "min_p");
                if (!min_p_str.empty()) try { haven_engine_.get_sampler().get_params().min_p = std::stof(min_p_str); } catch (...) {}

                std::string rep_str = extract_json_string(request, "repetition_penalty");
                if (!rep_str.empty()) try { haven_engine_.get_sampler().get_params().repetition_penalty = std::stof(rep_str); } catch (...) {}

                // Update persona parameters
                std::string comp_name = extract_json_string(request, "companion_name");
                if (!comp_name.empty()) companion_name_ = comp_name;

                std::string comp_avatar = extract_json_string(request, "companion_avatar");
                if (!comp_avatar.empty()) companion_avatar_ = comp_avatar;

                std::string u_name = extract_json_string(request, "user_name");
                if (!u_name.empty()) user_name_ = u_name;

                std::string sys_persona = extract_json_string(request, "system_persona");
                if (!sys_persona.empty()) system_persona_ = sys_persona;

                std::string greeting = extract_json_string(request, "persona_greeting");
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
            std::string resp = "HTTP/1.1 200 OK\r\nContent-Type: application/json; charset=utf-8\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: " 
                             + std::to_string(json.length()) + "\r\n\r\n" + json;
            send_all(sock, resp.c_str(), (int)resp.length());
        }
        else {
            // Check for static file in wwwroot
            std::string rel_path = path;
            if (rel_path == "/" || rel_path.empty()) rel_path = "/index.html";
            std::string local_path = "wwwroot" + rel_path;
            FILE* f = fopen(local_path.c_str(), "rb");
            if (!f) {
                local_path = "C:\\Users\\admin\\source\\haven-cpp\\wwwroot" + rel_path;
                for (char& c : local_path) if (c == '/') c = '\\';
                f = fopen(local_path.c_str(), "rb");
            }

            if (f) {
                fseek(f, 0, SEEK_END);
                long sz = ftell(f);
                fseek(f, 0, SEEK_SET);

                std::vector<char> file_data(sz);
                if (sz > 0) {
                    fread(file_data.data(), 1, sz, f);
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
                                     "\r\nCache-Control: no-cache, no-store, must-revalidate\r\nPragma: no-cache\r\nExpires: 0\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: " + 
                                     std::to_string(sz) + "\r\n\r\n";
                send_all(sock, header.c_str(), (int)header.length());
                if (sz > 0) {
                    send_all(sock, file_data.data(), (int)sz);
                }
            } else {
                std::string body = "{\"ok\":true,\"msg\":\"Haven Sovereign Micro-Server Active\"}";
                std::string resp = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: " 
                                 + std::to_string(body.length()) + "\r\n\r\n" + body;
                send_all(sock, resp.c_str(), (int)resp.length());
            }
        }

#ifdef _WIN32
        closesocket(sock);
#else
        close(sock);
#endif
    }
};

} // namespace haven

int main(int argc, char** argv) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif
    int port = 11436;
    std::string model_arg = "";
    if (argc > 1) {
        try {
            port = std::stoi(argv[1]);
        } catch (...) {
            model_arg = argv[1];
        }
    }
    if (argc > 2) {
        model_arg = argv[2];
    }

    haven::HavenMicroServer server(port, model_arg);
    server.start();

    std::cout << "[HavenServer] Featherlight Studio Daemon running on http://0.0.0.0:" << port << " (<10MB RAM)...\n" << std::flush;
    while (true) {
        std::this_thread::sleep_for(std::chrono::hours(24));
    }

    server.stop();
    return 0;
}
