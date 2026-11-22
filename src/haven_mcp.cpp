#include "haven/haven_mcp.h"
#include "haven/haven_json.h"
#include <iostream>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <chrono>
#include <thread>
#include <filesystem>
#include <unordered_set>

namespace haven {

// Internal lightweight JSON string utilities
static std::string mcp_escape_json(const std::string& s) {
    std::string out;
    out.reserve(s.length() + 16);
    for (char c : s) {
        if (c == '"') out += "\\\"";
        else if (c == '\\') out += "\\\\";
        else if (c == '\b') out += "\\b";
        else if (c == '\f') out += "\\f";
        else if (c == '\n') out += "\\n";
        else if (c == '\r') out += "\\r";
        else if (c == '\t') out += "\\t";
        else out += c;
    }
    return out;
}

static std::string mcp_unescape_json(const std::string& s) {
    std::string out;
    out.reserve(s.length());
    for (size_t i = 0; i < s.length(); ++i) {
        if (s[i] == '\\' && i + 1 < s.length()) {
            char next = s[++i];
            if (next == '"') out += '"';
            else if (next == '\\') out += '\\';
            else if (next == 'n') out += '\n';
            else if (next == 'r') out += '\r';
            else if (next == 't') out += '\t';
            else out += next;
        } else {
            out += s[i];
        }
    }
    return out;
}

static bool is_unescaped_quote(const std::string& s, size_t idx) {
    if (idx >= s.length() || s[idx] != '"') return false;
    size_t backslashes = 0;
    size_t j = idx;
    while (j > 0 && s[j - 1] == '\\') {
        backslashes++;
        j--;
    }
    return (backslashes % 2) == 0;
}

static bool is_balanced_json_object(const std::string& s) {
    if (s.size() < 2 || s.front() != '{' || s.back() != '}') return false;
    if (s.find('\r') != std::string::npos || s.find('\n') != std::string::npos) return false;
    HavenJson parsed;
    return HavenJson::parse(s, parsed) && parsed.is_object();
}

static std::string mcp_extract_json_field(const std::string& json, const std::string& key) {
    std::string pattern = "\"" + key + "\":";
    size_t pos = json.find(pattern);
    if (pos == std::string::npos) return "";

    size_t start = json.find_first_not_of(" \t\r\n", pos + pattern.length());
    if (start == std::string::npos) return "";

    if (json[start] == '"') {
        size_t end = start + 1;
        while (end < json.length()) {
            if (is_unescaped_quote(json, end)) break;
            end++;
        }
        return mcp_unescape_json(json.substr(start + 1, end - start - 1));
    } else if (json[start] == '{') {
        int depth = 0;
        size_t end = start;
        for (; end < json.length(); ++end) {
            if (json[end] == '{') depth++;
            else if (json[end] == '}') {
                depth--;
                if (depth == 0) { end++; break; }
            }
        }
        return json.substr(start, end - start);
    } else if (json[start] == '[') {
        int depth = 0;
        size_t end = start;
        for (; end < json.length(); ++end) {
            if (json[end] == '[') depth++;
            else if (json[end] == ']') {
                depth--;
                if (depth == 0) { end++; break; }
            }
        }
        return json.substr(start, end - start);
    } else {
        size_t end = json.find_first_of(",}\r\n \t", start);
        if (end == std::string::npos) end = json.length();
        return json.substr(start, end - start);
    }
}

// -----------------------------------------------------------------------------
// McpProcessBridge Implementation
// -----------------------------------------------------------------------------

McpProcessBridge::McpProcessBridge(const McpServerConfig& config)
    : config_(config) {}

McpProcessBridge::~McpProcessBridge() {
    stop();
}

bool McpProcessBridge::start() {
    std::lock_guard<std::recursive_mutex> lock(io_mutex_);
    if (is_running_) return true;

#ifdef _WIN32
    SECURITY_ATTRIBUTES saAttr{};
    saAttr.nLength = sizeof(SECURITY_ATTRIBUTES);
    saAttr.bInheritHandle = TRUE;
    saAttr.lpSecurityDescriptor = NULL;

    HANDLE h_child_stdout_wr = NULL;
    HANDLE h_child_stdin_rd = NULL;

    // Create stdout pipe
    if (!CreatePipe(&h_child_stdout_rd_, &h_child_stdout_wr, &saAttr, 0)) {
        std::cerr << "❌ [MCP:" << config_.name << "] Failed to create stdout pipe.\n";
        return false;
    }
    SetHandleInformation(h_child_stdout_rd_, HANDLE_FLAG_INHERIT, 0);

    // Create stdin pipe
    if (!CreatePipe(&h_child_stdin_rd, &h_child_stdin_wr_, &saAttr, 0)) {
        CloseHandle(h_child_stdout_rd_);
        CloseHandle(h_child_stdout_wr);
        std::cerr << "❌ [MCP:" << config_.name << "] Failed to create stdin pipe.\n";
        return false;
    }
    SetHandleInformation(h_child_stdin_wr_, HANDLE_FLAG_INHERIT, 0);

    // Build command line
    std::string cmdline = config_.command;
    if (cmdline.find(' ') != std::string::npos && cmdline.front() != '"') {
        cmdline = "\"" + cmdline + "\"";
    }
    for (const auto& arg : config_.args) {
        cmdline += " ";
        if (arg.find(' ') != std::string::npos && arg.front() != '"') {
            cmdline += "\"" + arg + "\"";
        } else {
            cmdline += arg;
        }
    }

    STARTUPINFOA siStartInfo{};
    siStartInfo.cb = sizeof(STARTUPINFOA);
    siStartInfo.hStdError = GetStdHandle(STD_ERROR_HANDLE);
    siStartInfo.hStdOutput = h_child_stdout_wr;
    siStartInfo.hStdInput = h_child_stdin_rd;
    siStartInfo.dwFlags |= STARTF_USESTDHANDLES;

    const char* workdir = config_.cwd.empty() ? nullptr : config_.cwd.c_str();

    // Create child process using writable command-line buffer per Win32 CreateProcessA contract
    std::vector<char> cmd_buf(cmdline.begin(), cmdline.end());
    cmd_buf.push_back('\0');
    BOOL success = CreateProcessA(
        NULL,
        cmd_buf.data(),
        NULL,
        NULL,
        TRUE,
        CREATE_NO_WINDOW,
        NULL,
        workdir,
        &siStartInfo,
        &proc_info_
    );

    // Close child-side pipe handles in parent
    CloseHandle(h_child_stdout_wr);
    CloseHandle(h_child_stdin_rd);

    if (!success) {
        CloseHandle(h_child_stdout_rd_);
        CloseHandle(h_child_stdin_wr_);
        h_child_stdout_rd_ = NULL;
        h_child_stdin_wr_ = NULL;
        std::cerr << "❌ [MCP:" << config_.name << "] Failed to start process: " << cmdline << "\n";
        return false;
    }

    is_running_ = true;
    std::cout << "🔌 [MCP:" << config_.name << "] Subprocess started (PID: " << proc_info_.dwProcessId << ")\n";
    return true;
#else
    // POSIX pipe implementation
    int in_pipe[2];
    int out_pipe[2];
    if (pipe(in_pipe) < 0 || pipe(out_pipe) < 0) return false;

    child_pid_ = fork();
    if (child_pid_ == 0) {
        dup2(in_pipe[0], STDIN_FILENO);
        dup2(out_pipe[1], STDOUT_FILENO);
        close(in_pipe[1]);
        close(out_pipe[0]);

        std::vector<char*> argv;
        argv.push_back(const_cast<char*>(config_.command.c_str()));
        for (const auto& a : config_.args) argv.push_back(const_cast<char*>(a.c_str()));
        argv.push_back(nullptr);

        if (!config_.cwd.empty()) chdir(config_.cwd.c_str());
        execvp(config_.command.c_str(), argv.data());
        _exit(127);
    }

    close(in_pipe[0]);
    close(out_pipe[1]);
    stdin_fd_ = in_pipe[1];
    stdout_fd_ = out_pipe[0];
    is_running_ = true;
    return true;
#endif
}

void McpProcessBridge::stop() {
    std::lock_guard<std::recursive_mutex> lock(io_mutex_);
    if (!is_running_) return;

#ifdef _WIN32
    if (h_child_stdin_wr_) {
        CloseHandle(h_child_stdin_wr_);
        h_child_stdin_wr_ = NULL;
    }
    if (h_child_stdout_rd_) {
        CloseHandle(h_child_stdout_rd_);
        h_child_stdout_rd_ = NULL;
    }
    if (proc_info_.hProcess) {
        TerminateProcess(proc_info_.hProcess, 0);
        WaitForSingleObject(proc_info_.hProcess, 500);
        CloseHandle(proc_info_.hProcess);
        CloseHandle(proc_info_.hThread);
        proc_info_ = {};
    }
#else
    if (stdin_fd_ >= 0) close(stdin_fd_);
    if (stdout_fd_ >= 0) close(stdout_fd_);
    if (child_pid_ > 0) {
        kill(child_pid_, SIGTERM);
        waitpid(child_pid_, nullptr, WNOHANG);
    }
    stdin_fd_ = -1;
    stdout_fd_ = -1;
    child_pid_ = -1;
#endif

    is_running_ = false;
    is_initialized_ = false;
    read_buffer_.clear();
}

bool McpProcessBridge::write_json_line(const std::string& line) {
    std::string data = line + "\n";
#ifdef _WIN32
    if (!h_child_stdin_wr_) return false;
    DWORD written = 0;
    return WriteFile(h_child_stdin_wr_, data.c_str(), (DWORD)data.length(), &written, NULL) && (written == data.length());
#else
    if (stdin_fd_ < 0) return false;
    ssize_t w = write(stdin_fd_, data.c_str(), data.length());
    return (w == (ssize_t)data.length());
#endif
}

bool McpProcessBridge::read_json_line(std::string& out_line, int timeout_ms) {
    auto start_time = std::chrono::steady_clock::now();

    while (true) {
        // Check if full line already in buffer
        size_t nl = read_buffer_.find('\n');
        if (nl != std::string::npos) {
            out_line = read_buffer_.substr(0, nl);
            read_buffer_.erase(0, nl + 1);
            if (!out_line.empty() && out_line.back() == '\r') out_line.pop_back();
            if (!out_line.empty()) return true;
        }

        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start_time).count();
        if (elapsed > timeout_ms) {
            return false;
        }

#ifdef _WIN32
        if (!h_child_stdout_rd_) return false;

        DWORD bytes_avail = 0;
        if (PeekNamedPipe(h_child_stdout_rd_, NULL, 0, NULL, &bytes_avail, NULL) && bytes_avail > 0) {
            char chunk[4096];
            DWORD bytes_read = 0;
            if (ReadFile(h_child_stdout_rd_, chunk, sizeof(chunk) - 1, &bytes_read, NULL) && bytes_read > 0) {
                chunk[bytes_read] = '\0';
                constexpr size_t MAX_MCP_READ_BUFFER = 8ULL * 1024 * 1024;
                if (read_buffer_.size() + bytes_read > MAX_MCP_READ_BUFFER) {
                    read_buffer_.clear();
                    return false;
                }
                read_buffer_.append(chunk, bytes_read);
                continue;
            }
        }

        DWORD exit_code = 0;
        if (GetExitCodeProcess(proc_info_.hProcess, &exit_code) && exit_code != STILL_ACTIVE) {
            is_running_ = false;
            return false;
        }
#else
        fd_set fds;
        FD_ZERO(&fds);
        FD_SET(stdout_fd_, &fds);
        struct timeval tv{};
        tv.tv_sec = 0;
        tv.tv_usec = 15000;
        int res = select(stdout_fd_ + 1, &fds, NULL, NULL, &tv);
        if (res > 0) {
            char chunk[4096];
            ssize_t r = read(stdout_fd_, chunk, sizeof(chunk) - 1);
            if (r > 0) {
                chunk[r] = '\0';
                constexpr size_t MAX_MCP_READ_BUFFER = 8ULL * 1024 * 1024;
                if (read_buffer_.size() + static_cast<size_t>(r) > MAX_MCP_READ_BUFFER) {
                    read_buffer_.clear();
                    return false;
                }
                read_buffer_.append(chunk, static_cast<size_t>(r));
                continue;
            } else if (r <= 0) {
                is_running_ = false;
                return false;
            }
        }
#endif

        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

bool McpProcessBridge::perform_handshake(int timeout_ms) {
    std::lock_guard<std::recursive_mutex> lock(io_mutex_);
    if (!start()) return false;

    // 1. Send MCP initialize request
    std::string init_req = "{\"jsonrpc\":\"2.0\",\"id\":" + std::to_string(next_req_id_++) +
                           ",\"method\":\"initialize\",\"params\":{\"protocolVersion\":\"2024-11-05\","
                           "\"capabilities\":{},\"clientInfo\":{\"name\":\"haven-cpp\",\"version\":\"1.0.0\"}}}";

    if (!write_json_line(init_req)) return false;

    // 2. Read initialize response
    std::string resp;
    bool got_response = false;
    for (int i = 0; i < 5; ++i) {
        if (read_json_line(resp, timeout_ms)) {
            if (resp.find("\"result\"") != std::string::npos || resp.find("capabilities") != std::string::npos) {
                got_response = true;
                break;
            }
        }
    }

    if (!got_response) {
        std::cerr << "⚠️ [MCP:" << config_.name << "] Handshake timeout waiting for initialize response.\n";
        return false;
    }

    // 3. Send notifications/initialized
    std::string init_notif = "{\"jsonrpc\":\"2.0\",\"method\":\"notifications/initialized\"}";
    write_json_line(init_notif);

    is_initialized_ = true;
    std::cout << "✓ [MCP:" << config_.name << "] Handshake complete. Connected via stdio JSON-RPC 2.0!\n";
    return true;
}

bool McpProcessBridge::discover_tools(std::vector<McpToolInfo>& out_tools, int timeout_ms) {
    std::lock_guard<std::recursive_mutex> lock(io_mutex_);
    if (!is_initialized_) {
        if (!perform_handshake(timeout_ms)) return false;
    }

    int list_id = next_req_id_++;
    std::string list_req = "{\"jsonrpc\":\"2.0\",\"id\":" + std::to_string(list_id) + ",\"method\":\"tools/list\"}";
    if (!write_json_line(list_req)) return false;

    std::string resp;
    bool found = false;
    for (int i = 0; i < 5; ++i) {
        if (read_json_line(resp, timeout_ms)) {
            if (resp.find("\"tools\"") != std::string::npos) {
                found = true;
                break;
            }
        }
    }

    if (!found) {
        std::cerr << "⚠️ [MCP:" << config_.name << "] tools/list request returned no tools.\n";
        return false;
    }

    // Parse tools array
    size_t tools_pos = resp.find("\"tools\":");
    if (tools_pos == std::string::npos) return false;

    size_t arr_start = resp.find('[', tools_pos);
    if (arr_start == std::string::npos) return false;

    size_t cur = arr_start + 1;
    while (cur < resp.length()) {
        size_t obj_start = resp.find('{', cur);
        if (obj_start == std::string::npos) break;

        // Find matching closing brace
        int depth = 0;
        size_t obj_end = obj_start;
        for (; obj_end < resp.length(); ++obj_end) {
            if (resp[obj_end] == '{') depth++;
            else if (resp[obj_end] == '}') {
                depth--;
                if (depth == 0) { obj_end++; break; }
            }
        }

        std::string tool_blob = resp.substr(obj_start, obj_end - obj_start);
        std::string name = mcp_extract_json_field(tool_blob, "name");
        std::string desc = mcp_extract_json_field(tool_blob, "description");
        std::string schema = mcp_extract_json_field(tool_blob, "inputSchema");

        if (!name.empty()) {
            McpToolInfo info;
            info.server_name = config_.name;
            info.tool_name = name;
            info.full_action_name = "mcp:" + config_.name + ":" + name;
            info.description = desc;
            info.input_schema_json = schema;
            out_tools.push_back(info);
        }

        cur = obj_end;
        if (cur >= resp.length() || resp[cur] == ']') break;
    }

    std::cout << "📦 [MCP:" << config_.name << "] Registered " << out_tools.size() << " tools from server.\n";
    return true;
}

bool McpProcessBridge::call_tool(const std::string& tool_name, const std::string& arguments_json, std::string& out_text, int timeout_ms) {
    std::lock_guard<std::recursive_mutex> lock(io_mutex_);
    if (!is_initialized_) {
        if (!perform_handshake()) return false;
    }

    int call_id = next_req_id_++;
    std::string args_payload = arguments_json;
    // Trim whitespace for JSON object validation
    size_t af = args_payload.find_first_not_of(" \t\r\n");
    size_t al = args_payload.find_last_not_of(" \t\r\n");
    std::string trimmed = (af != std::string::npos && al != std::string::npos) ? args_payload.substr(af, al - af + 1) : "";
    if (!trimmed.empty() && is_balanced_json_object(trimmed)) {
        args_payload = trimmed;
    } else if (!args_payload.empty()) {
        args_payload = "{\"input\":\"" + mcp_escape_json(args_payload) + "\"}";
    } else {
        args_payload = "{}";
    }

    std::string call_req = "{\"jsonrpc\":\"2.0\",\"id\":" + std::to_string(call_id) +
                           ",\"method\":\"tools/call\",\"params\":{\"name\":\"" + mcp_escape_json(tool_name) +
                           "\",\"arguments\":" + args_payload + "}}";

    if (!write_json_line(call_req)) {
        out_text = "❌ [MCP Bridge] Failed to send tool call over pipe.";
        return false;
    }

    std::string target_id_str = "\"id\":" + std::to_string(call_id);
    std::string resp;
    bool got_res = false;

    while (read_json_line(resp, timeout_ms)) {
        if (resp.find(target_id_str) != std::string::npos) {
            got_res = true;
            break;
        }
    }

    if (!got_res) {
        out_text = "⚠️ [MCP Bridge] Tool call timed out waiting for server response.";
        return false;
    }

    // Check for error
    if (resp.find("\"error\":") != std::string::npos) {
        std::string err_obj = mcp_extract_json_field(resp, "error");
        std::string msg = mcp_extract_json_field(err_obj, "message");
        out_text = "⚠️ [MCP Error]: " + (msg.empty() ? err_obj : msg);
        return true;
    }

    // Extract text from content array
    size_t content_pos = resp.find("\"content\":");
    if (content_pos != std::string::npos) {
        std::string full_extracted = "";
        size_t text_pos = content_pos;
        while ((text_pos = resp.find("\"text\":", text_pos)) != std::string::npos) {
            size_t val_start = resp.find('"', text_pos + 7);
            if (val_start == std::string::npos) break;
            size_t val_end = val_start + 1;
            while (val_end < resp.length()) {
                if (is_unescaped_quote(resp, val_end)) break;
                val_end++;
            }
            if (!full_extracted.empty()) full_extracted += "\n";
            full_extracted += mcp_unescape_json(resp.substr(val_start + 1, val_end - val_start - 1));
            text_pos = val_end + 1;
        }
        if (!full_extracted.empty()) {
            out_text = full_extracted;
            return true;
        }
    }

    // Fallback: return raw result
    std::string res_raw = mcp_extract_json_field(resp, "result");
    out_text = res_raw.empty() ? resp : res_raw;
    return true;
}

// -----------------------------------------------------------------------------
// McpManager Implementation
// -----------------------------------------------------------------------------

McpManager::McpManager() {
    load_config();
}

McpManager::~McpManager() {
    stop_all();
}

bool McpManager::load_config(const std::string& config_path) {
    std::lock_guard<std::mutex> lock(mutex_);
    config_filepath_ = config_path;

    std::string resolved_path = config_path;
    if (!std::filesystem::exists(resolved_path)) {
        if (std::filesystem::exists("wwwroot/" + config_path)) {
            resolved_path = "wwwroot/" + config_path;
        } else {
            return false;
        }
    }

    std::ifstream f(resolved_path);
    if (!f.is_open()) return false;

    std::string content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    f.close();

    size_t srv_pos = content.find("\"mcpServers\":");
    if (srv_pos == std::string::npos) return false;

    size_t map_start = content.find('{', srv_pos + 13);
    if (map_start == std::string::npos) return false;

    // Simple parser for mcpServers dictionary
    size_t cur = map_start + 1;
    while (cur < content.length()) {
        size_t k_start = content.find('"', cur);
        if (k_start == std::string::npos) break;
        size_t k_end = content.find('"', k_start + 1);
        if (k_end == std::string::npos) break;
        std::string server_name = content.substr(k_start + 1, k_end - k_start - 1);

        size_t colon = content.find(':', k_end);
        if (colon == std::string::npos) break;

        size_t obj_start = content.find('{', colon);
        if (obj_start == std::string::npos) break;

        int depth = 0;
        size_t obj_end = obj_start;
        for (; obj_end < content.length(); ++obj_end) {
            if (content[obj_end] == '{') depth++;
            else if (content[obj_end] == '}') {
                depth--;
                if (depth == 0) { obj_end++; break; }
            }
        }

        std::string server_blob = content.substr(obj_start, obj_end - obj_start);
        std::string command = mcp_extract_json_field(server_blob, "command");

        McpServerConfig cfg;
        cfg.name = server_name;
        cfg.command = command;

        // Parse args array
        size_t args_pos = server_blob.find("\"args\":");
        if (args_pos != std::string::npos) {
            size_t arr_s = server_blob.find('[', args_pos);
            size_t arr_e = server_blob.find(']', arr_s);
            if (arr_s != std::string::npos && arr_e != std::string::npos) {
                std::string args_inner = server_blob.substr(arr_s + 1, arr_e - arr_s - 1);
                std::stringstream ss(args_inner);
                std::string item;
                while (std::getline(ss, item, ',')) {
                    size_t f_q = item.find('"');
                    size_t l_q = item.rfind('"');
                    if (f_q != std::string::npos && l_q != std::string::npos && l_q > f_q) {
                        cfg.args.push_back(item.substr(f_q + 1, l_q - f_q - 1));
                    }
                }
            }
        }

        if (!cfg.name.empty() && !cfg.command.empty()) {
            auto bridge = std::make_shared<McpProcessBridge>(cfg);
            servers_[cfg.name] = bridge;

            // Handshake & discover tools immediately
            std::vector<McpToolInfo> server_tools;
            if (bridge->discover_tools(server_tools)) {
                for (const auto& t : server_tools) {
                    tools_[t.full_action_name] = t;
                    // Also register simple aliases if not conflicting
                    if (tools_.find(t.tool_name) == tools_.end()) {
                        tools_[t.tool_name] = t;
                    }
                }
            }
        }

        cur = obj_end;
        if (cur >= content.length() || content[cur] == '}') break;
    }

    std::cout << "✨ [MCP Subsystem] Loaded " << servers_.size() << " MCP servers with " 
              << tools_.size() << " actions available.\n";
    return true;
}

bool McpManager::save_config(const std::string& config_path) {
    std::lock_guard<std::mutex> lock(mutex_);
    std::string path = config_path.empty() ? config_filepath_ : config_path;

    std::ostringstream ss;
    ss << "{\n  \"mcpServers\": {\n";
    bool first = true;
    for (const auto& [name, bridge] : servers_) {
        if (!first) ss << ",\n";
        first = false;
        const auto& cfg = bridge->get_config();
        ss << "    \"" << mcp_escape_json(cfg.name) << "\": {\n";
        ss << "      \"command\": \"" << mcp_escape_json(cfg.command) << "\",\n";
        ss << "      \"args\": [";
        for (size_t a = 0; a < cfg.args.size(); ++a) {
            if (a > 0) ss << ", ";
            ss << "\"" << mcp_escape_json(cfg.args[a]) << "\"";
        }
        ss << "]\n";
        ss << "    }";
    }
    ss << "\n  }\n}\n";

    std::ofstream f(path);
    if (!f.is_open()) return false;
    f << ss.str();
    return true;
}

void McpManager::reload_all(const std::string& config_path) {
    stop_all();
    load_config(config_path);
}

void McpManager::stop_all() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& [name, bridge] : servers_) {
        if (bridge) bridge->stop();
    }
    servers_.clear();
    tools_.clear();
}

bool McpManager::add_server(const McpServerConfig& config, bool autostart) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto bridge = std::make_shared<McpProcessBridge>(config);
    servers_[config.name] = bridge;

    if (autostart) {
        std::vector<McpToolInfo> server_tools;
        if (bridge->discover_tools(server_tools)) {
            for (const auto& t : server_tools) {
                tools_[t.full_action_name] = t;
                if (tools_.find(t.tool_name) == tools_.end()) {
                    tools_[t.tool_name] = t;
                }
            }
        }
    }
    return true;
}

bool McpManager::remove_server(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = servers_.find(name);
    if (it != servers_.end()) {
        it->second->stop();
        servers_.erase(it);

        // Remove associated tools
        for (auto t_it = tools_.begin(); t_it != tools_.end();) {
            if (t_it->second.server_name == name) {
                t_it = tools_.erase(t_it);
            } else {
                ++t_it;
            }
        }
        return true;
    }
    return false;
}

bool McpManager::restart_server(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = servers_.find(name);
    if (it != servers_.end()) {
        it->second->stop();
        std::vector<McpToolInfo> server_tools;
        return it->second->discover_tools(server_tools);
    }
    return false;
}

bool McpManager::can_handle_tool(const std::string& action) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return tools_.find(action) != tools_.end();
}

bool McpManager::dispatch_tool(const std::string& action, const std::string& payload, std::string& out_result) {
    std::shared_ptr<McpProcessBridge> target_bridge;
    std::string tool_name;

    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = tools_.find(action);
        if (it == tools_.end()) return false;

        auto srv_it = servers_.find(it->second.server_name);
        if (srv_it == servers_.end()) return false;

        target_bridge = srv_it->second;
        tool_name = it->second.tool_name;
    }

    return target_bridge->call_tool(tool_name, payload, out_result);
}

size_t McpManager::get_active_server_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t count = 0;
    for (const auto& [name, bridge] : servers_) {
        if (bridge && bridge->is_running() && bridge->is_initialized()) count++;
    }
    return count;
}

std::string McpManager::get_tools_prompt_description() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (tools_.empty()) return "";

    std::ostringstream ss;
    ss << "MCP tools: ";
    std::unordered_set<std::string> seen_tools;
    bool first = true;
    for (const auto& [action_name, info] : tools_) {
        if (seen_tools.count(info.full_action_name)) continue;
        seen_tools.insert(info.full_action_name);
        if (!first) ss << ", ";
        first = false;
        ss << info.full_action_name;
    }
    ss << ".";
    return ss.str();
}

std::string McpManager::get_servers_json() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::ostringstream ss;
    ss << "[";
    bool first = true;
    for (const auto& [name, bridge] : servers_) {
        if (!first) ss << ",";
        first = false;
        const auto& cfg = bridge->get_config();
        bool active = bridge->is_running() && bridge->is_initialized();
        
        int tool_cnt = 0;
        for (const auto& [t_name, info] : tools_) {
            if (info.server_name == name && t_name == info.full_action_name) tool_cnt++;
        }

        ss << "{"
           << "\"name\":\"" << mcp_escape_json(cfg.name) << "\","
           << "\"command\":\"" << mcp_escape_json(cfg.command) << "\","
           << "\"status\":\"" << (active ? "CONNECTED" : "OFFLINE") << "\","
           << "\"tools_count\":" << tool_cnt
           << "}";
    }
    ss << "]";
    return ss.str();
}

std::string McpManager::get_tools_json() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::ostringstream ss;
    ss << "[";
    bool first = true;
    std::unordered_set<std::string> seen;
    for (const auto& [action_name, info] : tools_) {
        if (seen.count(info.full_action_name)) continue;
        seen.insert(info.full_action_name);

        if (!first) ss << ",";
        first = false;

        ss << "{"
           << "\"server\":\"" << mcp_escape_json(info.server_name) << "\","
           << "\"name\":\"" << mcp_escape_json(info.tool_name) << "\","
           << "\"action\":\"" << mcp_escape_json(info.full_action_name) << "\","
           << "\"description\":\"" << mcp_escape_json(info.description) << "\","
           << "\"inputSchema\":" << (info.input_schema_json.empty() ? "{}" : info.input_schema_json)
           << "}";
    }
    ss << "]";
    return ss.str();
}

} // namespace haven
