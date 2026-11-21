#pragma once

#include "haven_plugin.h"
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <filesystem>
#include <chrono>
#include <mutex>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

namespace haven {

class McpManager;

struct LoadedPluginRecord {
    std::string filepath;
    PluginMetadata metadata;
    std::shared_ptr<IHavenPlugin> instance = nullptr;
#ifdef _WIN32
    HMODULE dll_handle = nullptr;
#else
    void* so_handle = nullptr;
#endif
    DestroyPluginFunc destroy_func = nullptr;
    bool is_loaded = false;
    std::vector<std::string> supported_actions;
    std::chrono::steady_clock::time_point last_used;
};

class PluginManager {
public:
    PluginManager();
    ~PluginManager();

    // Auto-discovers plugins in the given directory. If lazy_load is true,
    // registers plugin metadata & tool capabilities without keeping DLLs resident in RAM.
    size_t discover_plugins(const std::string& plugins_directory = "plugins", bool lazy_load = true);

    // Manually loads a single dynamic library plugin (.dll / .so)
    bool load_plugin_file(const std::string& filepath, PluginExecutionContext* ctx = nullptr);

    // Registers an in-process native plugin directly (always resident)
    bool register_native_plugin(std::shared_ptr<IHavenPlugin> plugin, PluginExecutionContext* ctx = nullptr);

    // Ensures a specific plugin is loaded into memory (hot-loads on demand if currently cold)
    bool ensure_plugin_loaded(LoadedPluginRecord& record, PluginExecutionContext* ctx = nullptr);
    bool ensure_plugin_loaded_by_id(const std::string& plugin_id, PluginExecutionContext* ctx = nullptr);

    // Unloads a specific plugin instance to free memory while retaining discovery metadata
    bool unload_plugin_instance(LoadedPluginRecord& record);
    bool unload_plugin(const std::string& plugin_id);

    // Unloads any dynamic plugins that have been idle for longer than max_idle_seconds
    size_t unload_idle_plugins(int max_idle_seconds = 180);

    // Unloads all active plugins safely
    void unload_all();

    // Plugin inspection (returns thread-safe snapshot copy)
    std::unordered_map<std::string, LoadedPluginRecord> get_loaded_plugins() const {
        std::lock_guard<std::recursive_mutex> lock(plugins_mutex_);
        return plugins_;
    }
    size_t get_plugin_count() const {
        std::lock_guard<std::recursive_mutex> lock(plugins_mutex_);
        return plugins_.size();
    }
    size_t get_active_plugin_count() const;
    bool has_plugin(const std::string& plugin_id) const {
        std::lock_guard<std::recursive_mutex> lock(plugins_mutex_);
        return plugins_.find(plugin_id) != plugins_.end();
    }

    // Hot-reloads all plugins from the directory without restarting engine
    size_t reload_all(const std::string& plugins_directory = "plugins", PluginExecutionContext* ctx = nullptr);

    // Returns formatted prompt describing all available tools and how Aura can call them
    std::string get_tools_prompt_description() const;

    // Dispatchers for engine lifecycle events
    void dispatch_prompt_prefill(std::string& prompt, std::vector<uint32_t>& tokens);
    void dispatch_token_generated(uint32_t token_id, const std::string& piece);
    bool dispatch_tool_execution(const std::string& action, const std::string& payload, std::string& output);
    void dispatch_sensory_tick(std::vector<float>& vision_emb, std::vector<float>& audio_emb);

    void set_execution_context(PluginExecutionContext* ctx) { execution_ctx_ = ctx; }
    void set_mcp_manager(McpManager* mcp) { mcp_manager_ = mcp; }
    McpManager* get_mcp_manager() const { return mcp_manager_; }

private:
    bool probe_plugin_metadata(const std::string& filepath, PluginMetadata& out_meta, std::vector<std::string>& out_actions);

    mutable std::recursive_mutex plugins_mutex_;
    std::unordered_map<std::string, LoadedPluginRecord> plugins_;
    std::string plugins_dir_ = "plugins";
    PluginExecutionContext* execution_ctx_ = nullptr;
    McpManager* mcp_manager_ = nullptr;
    bool lazy_load_enabled_ = true;
};

} // namespace haven
