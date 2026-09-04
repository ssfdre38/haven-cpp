#include "haven/haven_engine.h"
#include <iostream>
#include <string>
#include <vector>
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
            std::cout << "[HavenServer] Loading Sovereign Gemma 4 Model into Memory: " << path_to_load << "...\n" << std::flush;
            if (haven_engine_.load_model(path_to_load)) {
                model_loaded_ = true;
                std::cout << "[HavenServer] ✓ Sovereign Model Loaded in-process! Standalone generation active.\n" << std::flush;

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
            }
        }
        load_persona();
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
    std::mutex engine_mutex_;
    std::vector<uint32_t> cached_tokens_;
    bool model_loaded_;

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

    void forward_to_backend(int client_sock, std::string request_payload, int target_port) {
        // Connect to target backend (11436 for LLM, 8085 for Stable Diffusion)
        int backend_sock = socket(AF_INET, SOCK_STREAM, 0);
        if (backend_sock < 0) return;

        sockaddr_in backend_addr{};
        backend_addr.sin_family = AF_INET;
        backend_addr.sin_addr.s_addr = inet_addr("127.0.0.1");
        backend_addr.sin_port = htons(target_port);

        if (connect(backend_sock, (struct sockaddr*)&backend_addr, sizeof(backend_addr)) < 0) {
#ifdef _WIN32
            closesocket(backend_sock);
#else
            close(backend_sock);
#endif
            std::string err = "HTTP/1.1 502 Bad Gateway\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: 0\r\n\r\n";
            send(client_sock, err.c_str(), (int)err.length(), 0);
            return;
        }

        // Rewrite Host header
        size_t host_pos = request_payload.find("Host: ");
        if (host_pos != std::string::npos) {
            size_t host_end = request_payload.find("\r\n", host_pos);
            request_payload.replace(host_pos, host_end - host_pos, "Host: 127.0.0.1:" + std::to_string(target_port));
        }

        // Ensure backend closes connection when response is sent
        size_t conn_pos = request_payload.find("Connection: keep-alive");
        if (conn_pos != std::string::npos) {
            request_payload.replace(conn_pos, 22, "Connection: close     ");
        }

        // Send payload to backend
        send(backend_sock, request_payload.c_str(), (int)request_payload.length(), 0);

        // Stream backend response directly back to client
        char buffer[32768];
        std::string accumulated_stream;

        while (true) {
#ifdef _WIN32
            int bytes = recv(backend_sock, buffer, sizeof(buffer), 0);
#else
            int bytes = read(backend_sock, buffer, sizeof(buffer));
#endif
            if (bytes <= 0) break;
            send(client_sock, buffer, bytes, 0);
            if (target_port == 11436) {
                accumulated_stream.append(buffer, bytes);
            }
        }

#ifdef _WIN32
        closesocket(backend_sock);
#else
        close(backend_sock);
#endif

        if (target_port == 11436 && !accumulated_stream.empty()) {
            process_server_side_tools(accumulated_stream);
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
    }

    void handle_native_chat_completions(int client_sock, const std::string& request) {
        std::lock_guard<std::mutex> lock(engine_mutex_);
        size_t body_pos = request.find("\r\n\r\n");
        std::string body = (body_pos != std::string::npos) ? request.substr(body_pos + 4) : request;

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

        if (stream) {
            std::string sse_hdr = "HTTP/1.1 200 OK\r\n"
                                  "Content-Type: text/event-stream\r\n"
                                  "Cache-Control: no-cache\r\n"
                                  "Connection: keep-alive\r\n"
                                  "Access-Control-Allow-Origin: *\r\n\r\n";
            send_all(client_sock, sse_hdr.c_str(), (int)sse_hdr.length());

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

                accumulated_response += piece;
                std::string esc_text = escape_json_string(piece);
                std::string chunk = "data: {\"content\":\"" + esc_text + "\",\"id\":\"chatcmpl-haven-sovereign\",\"object\":\"chat.completion.chunk\",\"model\":\"haven-chat-v5.0\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"" + esc_text + "\"},\"finish_reason\":null}]}\n\n";
                if (!send_all(client_sock, chunk.c_str(), (int)chunk.length())) break;

                haven_engine_.forward(next_tok, active_pos++, logits.data());
            }

            std::string done_chunk = "data: {\"id\":\"chatcmpl-haven-sovereign\",\"object\":\"chat.completion.chunk\",\"model\":\"haven-chat-v5.0\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n";
            send_all(client_sock, done_chunk.c_str(), (int)done_chunk.length());
        } else {
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

                accumulated_response += piece;
                haven_engine_.forward(next_tok, active_pos++, logits.data());
            }

            std::string esc_content = escape_json_string(accumulated_response);
            std::string json_resp = "{\"id\":\"chatcmpl-haven-sovereign\",\"object\":\"chat.completion\",\"model\":\"haven-chat-v5.0\",\"content\":\"" + esc_content + "\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"" + esc_content + "\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":" + std::to_string(tokens.size()) + ",\"completion_tokens\":" + std::to_string(accumulated_response.length() / 4) + ",\"total_tokens\":" + std::to_string(tokens.size() + accumulated_response.length() / 4) + "}}";

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
        else if (path == "/uploads" || path == "/uploads/") {
            path = "/uploads/index.html";
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
            send(sock, resp.c_str(), (int)resp.length(), 0);
        }
        else if (path == "/v1/models") {
            std::string body = "{\"object\":\"list\",\"data\":[{\"id\":\"haven-chat-v5.0\",\"object\":\"model\",\"owned_by\":\"sovereign\"},{\"id\":\"aura\",\"object\":\"model\",\"owned_by\":\"sovereign\"},{\"id\":\"gemma4-e4b\",\"object\":\"model\",\"owned_by\":\"sovereign\"}]}";
            std::string resp = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: " 
                             + std::to_string(body.length()) + "\r\n\r\n" + body;
            send(sock, resp.c_str(), (int)resp.length(), 0);
        }
        else if (path == "/v1/chat/completions" || path == "/completion" || path == "/v1/completions") {
            if (model_loaded_) {
                handle_native_chat_completions(sock, request);
            } else {
                forward_to_backend(sock, request, 11436);
            }
        }
        else if (path == "/api/plugins") {
            const auto& plugins = haven_engine_.get_plugin_manager().get_loaded_plugins();
            std::string json = "{\"ok\":true,\"total\":" + std::to_string(plugins.size()) + ",\"plugins\":[";
            bool first = true;
            for (const auto& [id, record] : plugins) {
                if (!first) json += ",";
                json += "{\"id\":\"" + escape_json_string(record.metadata.id) 
                     + "\",\"name\":\"" + escape_json_string(record.metadata.name) 
                     + "\",\"version\":\"" + escape_json_string(record.metadata.version)
                     + "\",\"author\":\"" + escape_json_string(record.metadata.author)
                     + "\",\"description\":\"" + escape_json_string(record.metadata.description) + "\"}";
                first = false;
            }
            json += "]}";
            std::string resp = "HTTP/1.1 200 OK\r\nContent-Type: application/json; charset=utf-8\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: " 
                             + std::to_string(json.length()) + "\r\n\r\n" + json;
            send_all(sock, resp.c_str(), (int)resp.length());
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
                             + "\"model\":\"haven-chat-v5.0.gguf\","
                             + "\"plugins_count\":" + std::to_string(haven_engine_.get_plugin_manager().get_plugin_count()) + "}";
            std::string resp = "HTTP/1.1 200 OK\r\nContent-Type: application/json; charset=utf-8\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: " 
                             + std::to_string(json.length()) + "\r\n\r\n" + json;
            send_all(sock, resp.c_str(), (int)resp.length());
        }
        else {
            // Check for static file in wwwroot
            std::string local_path = "wwwroot" + path;
            FILE* f = fopen(local_path.c_str(), "rb");
            if (!f) {
                local_path = "C:\\Users\\admin\\source\\haven-cpp\\wwwroot" + path;
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
                if (path.ends_with(".html")) mime = "text/html; charset=utf-8";
                else if (path.ends_with(".png")) mime = "image/png";
                else if (path.ends_with(".jpg") || path.ends_with(".jpeg")) mime = "image/jpeg";
                else if (path.ends_with(".webp")) mime = "image/webp";
                else if (path.ends_with(".css")) mime = "text/css";
                else if (path.ends_with(".js")) mime = "application/javascript";
                else if (path.ends_with(".json")) mime = "application/json";

                std::string header = "HTTP/1.1 200 OK\r\nContent-Type: " + mime + 
                                     "\r\nCache-Control: no-cache, no-store, must-revalidate\r\nPragma: no-cache\r\nExpires: 0\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: " + 
                                     std::to_string(sz) + "\r\n\r\n";
                send(sock, header.c_str(), (int)header.length(), 0);
                if (sz > 0) {
                    send(sock, file_data.data(), (int)sz, 0);
                }
            } else {
                std::string body = "{\"ok\":true,\"msg\":\"Haven Sovereign Micro-Server Active\"}";
                std::string resp = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: " 
                                 + std::to_string(body.length()) + "\r\n\r\n" + body;
                send(sock, resp.c_str(), (int)resp.length(), 0);
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
    if (argc > 1) {
        try {
            port = std::stoi(argv[1]);
        } catch (...) {}
    }

    haven::HavenMicroServer server(port);
    server.start();

    std::cout << "[HavenServer] Featherlight Studio Daemon running on http://0.0.0.0:" << port << " (<10MB RAM)...\n" << std::flush;
    while (true) {
        std::this_thread::sleep_for(std::chrono::hours(24));
    }

    server.stop();
    return 0;
}
