#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <mutex>
#include <chrono>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#endif

namespace haven {

struct McpServerConfig {
    std::string name;
    std::string command;
    std::vector<std::string> args;
    std::unordered_map<std::string, std::string> env;
    std::string cwd;
    bool enabled = true;
};

struct McpToolInfo {
    std::string server_name;
    std::string tool_name;
    std::string full_action_name; // e.g. "mcp:ash:query_memory" or "mcp_query_memory"
    std::string description;
    std::string input_schema_json;
};

// Low-level subprocess session communicating over stdio JSON-RPC 2.0
class McpProcessBridge {
public:
    McpProcessBridge(const McpServerConfig& config);
    ~McpProcessBridge();

    bool start();
    void stop();
    bool is_running() const { return is_running_; }
    bool is_initialized() const { return is_initialized_; }
    const McpServerConfig& get_config() const { return config_; }

    // Handshake and discovery
    bool perform_handshake(int timeout_ms = 8000);
    bool discover_tools(std::vector<McpToolInfo>& out_tools, int timeout_ms = 8000);

    // Tool execution
    bool call_tool(const std::string& tool_name, const std::string& arguments_json, std::string& out_text, int timeout_ms = 30000);

private:
    bool write_json_line(const std::string& line);
    bool read_json_line(std::string& out_line, int timeout_ms);

    McpServerConfig config_;
    bool is_running_ = false;
    bool is_initialized_ = false;
    int next_req_id_ = 1;
    std::string read_buffer_;
    std::mutex io_mutex_;

#ifdef _WIN32
    HANDLE h_child_stdin_wr_ = NULL;
    HANDLE h_child_stdout_rd_ = NULL;
    PROCESS_INFORMATION proc_info_{};
#else
    int stdin_fd_ = -1;
    int stdout_fd_ = -1;
    pid_t child_pid_ = -1;
#endif
};

// High-level manager orchestrating all MCP servers and their discovered tools
class McpManager {
public:
    McpManager();
    ~McpManager();

    // Configuration and lifecycle
    bool load_config(const std::string& config_path = "mcp_servers.json");
    bool save_config(const std::string& config_path = "mcp_servers.json");
    void reload_all(const std::string& config_path = "mcp_servers.json");
    void stop_all();

    // Server management
    bool add_server(const McpServerConfig& config, bool autostart = true);
    bool remove_server(const std::string& name);
    bool restart_server(const std::string& name);

    // Tool dispatch and queries
    bool can_handle_tool(const std::string& action) const;
    bool dispatch_tool(const std::string& action, const std::string& payload, std::string& out_result);

    const std::unordered_map<std::string, McpToolInfo>& get_tools() const { return tools_; }
    size_t get_active_server_count() const;
    size_t get_total_server_count() const { return servers_.size(); }
    size_t get_tool_count() const { return tools_.size(); }

    // Prompt & Web UI export
    std::string get_tools_prompt_description() const;
    std::string get_servers_json() const;
    std::string get_tools_json() const;

private:
    std::string config_filepath_ = "mcp_servers.json";
    std::unordered_map<std::string, std::shared_ptr<McpProcessBridge>> servers_;
    std::unordered_map<std::string, McpToolInfo> tools_;
    mutable std::mutex mutex_;
};

} // namespace haven
