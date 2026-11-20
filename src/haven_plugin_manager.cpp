#include "haven/haven_plugin_manager.h"
#include "haven/haven_mcp.h"
#include <iostream>
#include <vector>
#include <chrono>

#ifndef _WIN32
#include <dlfcn.h>
#endif

namespace haven {

static const std::vector<std::string> KNOWN_TOOL_ACTIONS = {
    // Circadian
    "daily_briefing", "hydrate_check", "circadian_status", "care_reminder",
    // Clipboard
    "read_clipboard", "get_clipboard", "copy_to_clipboard", "set_clipboard", "clipboard_status",
    // Discord
    "discord_send", "discord_status", "discord_set_webhook", "discord_set_channel", "discord_format_speaker",
    // Gaming
    "game_status", "game_log", "cheer_on", "list_games",
    // Screen Vision
    "get_active_window", "screen_info", "list_windows",
    // Stable Diffusion
    "generate_artwork", "paint_scene", "sd_paint", "sd_status",
    // Smart Home IoT
    "home_status", "home_set_light", "home_toggle_switch", "home_ambient_scene",
    // Browser
    "browse_url", "read_page", "open_browser", "open_in_browser", "browser_status",
    // Graph
    "graph_add_relation", "graph_query_entity", "graph_list_all", "graph_stats",
    // System Sensory
    "system_telemetry", "get_time", "get_battery",
    // Terminal
    "run_command", "exec_shell", "cmd_run", "shell_status",
    // Vault Sync
    "vault_search", "vault_read", "vault_list", "diary_write", "diary_read",
    // Voice Synth
    "speak_text", "synthesize_voice", "say_voice", "voice_status",
    // Web Research
    "web_search", "wiki_summary", "get_weather",
    // Whisper
    "whisper_status", "listen_transcribe", "whisper_transcribe",
    // Coder
    "read_file", "read_code", "list_directory", "list_files", "search_code", "file_info",
    // YouTube Music
    "yt_music_play", "yt_music_pause", "yt_music_toggle", "yt_music_next", "yt_music_prev", "yt_music_search", "yt_music_open"
};

PluginManager::PluginManager() = default;

PluginManager::~PluginManager() {
    unload_all();
}

bool PluginManager::probe_plugin_metadata(const std::string& filepath, PluginMetadata& out_meta, std::vector<std::string>& out_actions) {
#ifdef _WIN32
    HMODULE hModule = LoadLibraryA(filepath.c_str());
    if (!hModule) {
        return false;
    }

    auto create_func = reinterpret_cast<CreatePluginFunc>(GetProcAddress(hModule, "create_haven_plugin"));
    auto destroy_func = reinterpret_cast<DestroyPluginFunc>(GetProcAddress(hModule, "destroy_haven_plugin"));

    if (!create_func) {
        FreeLibrary(hModule);
        return false;
    }

    IHavenPlugin* raw_instance = create_func();
    if (!raw_instance) {
        FreeLibrary(hModule);
        return false;
    }

    out_meta = raw_instance->get_metadata();
    out_actions.clear();
    for (const auto& act : KNOWN_TOOL_ACTIONS) {
        if (raw_instance->can_handle_tool(act)) {
            out_actions.push_back(act);
        }
    }

    if (destroy_func) destroy_func(raw_instance);
    else delete raw_instance;

    FreeLibrary(hModule);
    return true;
#else
    void* handle = dlopen(filepath.c_str(), RTLD_NOW);
    if (!handle) return false;

    auto create_func = (CreatePluginFunc)dlsym(handle, "create_haven_plugin");
    auto destroy_func = (DestroyPluginFunc)dlsym(handle, "destroy_haven_plugin");
    if (!create_func) {
        dlclose(handle);
        return false;
    }

    IHavenPlugin* raw_instance = create_func();
    if (!raw_instance) {
        dlclose(handle);
        return false;
    }

    out_meta = raw_instance->get_metadata();
    out_actions.clear();
    for (const auto& act : KNOWN_TOOL_ACTIONS) {
        if (raw_instance->can_handle_tool(act)) {
            out_actions.push_back(act);
        }
    }

    if (destroy_func) destroy_func(raw_instance);
    else delete raw_instance;

    dlclose(handle);
    return true;
#endif
}

size_t PluginManager::discover_plugins(const std::string& plugins_directory, bool lazy_load) {
    plugins_dir_ = plugins_directory;
    lazy_load_enabled_ = lazy_load;
    if (!std::filesystem::exists(plugins_directory)) {
        try {
            std::filesystem::create_directories(plugins_directory);
        } catch (...) {}
        return 0;
    }

    size_t count = 0;
    for (const auto& entry : std::filesystem::recursive_directory_iterator(plugins_directory)) {
        if (entry.is_regular_file()) {
            std::string ext = entry.path().extension().string();
#ifdef _WIN32
            if (ext == ".dll")
#else
            if (ext == ".so")
#endif
            {
                std::string path_str = entry.path().string();
                if (lazy_load) {
                    PluginMetadata meta;
                    std::vector<std::string> actions;
                    if (probe_plugin_metadata(path_str, meta, actions)) {
                        LoadedPluginRecord record;
                        record.filepath = path_str;
                        record.metadata = meta;
                        record.supported_actions = std::move(actions);
                        record.is_loaded = false;
                        record.instance = nullptr;
                        plugins_[meta.id] = record;
                        std::cout << "📦 [Plugin Discovered (Cold/On-Demand)] " << meta.name << " v" << meta.version 
                                  << " (" << record.supported_actions.size() << " tools registered)\n";
                        count++;
                    }
                } else {
                    if (load_plugin_file(path_str, execution_ctx_)) {
                        count++;
                    }
                }
            }
        }
    }
    return count;
}

bool PluginManager::load_plugin_file(const std::string& filepath, PluginExecutionContext* ctx) {
#ifdef _WIN32
    HMODULE hModule = LoadLibraryA(filepath.c_str());
    if (!hModule) {
        std::cerr << "⚠️ Failed to load plugin library: " << filepath << " (Error " << GetLastError() << ")\n";
        return false;
    }

    auto create_func = reinterpret_cast<CreatePluginFunc>(GetProcAddress(hModule, "create_haven_plugin"));
    auto destroy_func = reinterpret_cast<DestroyPluginFunc>(GetProcAddress(hModule, "destroy_haven_plugin"));

    if (!create_func) {
        std::cerr << "⚠️ Plugin library missing 'create_haven_plugin' entrypoint: " << filepath << "\n";
        FreeLibrary(hModule);
        return false;
    }

    IHavenPlugin* raw_instance = create_func();
    if (!raw_instance) {
        std::cerr << "⚠️ Plugin factory returned null instance: " << filepath << "\n";
        FreeLibrary(hModule);
        return false;
    }

    PluginMetadata meta = raw_instance->get_metadata();
    if (!raw_instance->on_load(ctx ? ctx : execution_ctx_)) {
        std::cerr << "⚠️ Plugin failed on_load initialization: " << meta.name << "\n";
        if (destroy_func) destroy_func(raw_instance);
        else delete raw_instance;
        FreeLibrary(hModule);
        return false;
    }

    LoadedPluginRecord record;
    record.filepath = filepath;
    record.metadata = meta;
    for (const auto& act : KNOWN_TOOL_ACTIONS) {
        if (raw_instance->can_handle_tool(act)) {
            record.supported_actions.push_back(act);
        }
    }
    record.instance = std::shared_ptr<IHavenPlugin>(raw_instance, [destroy_func](IHavenPlugin* p) {
        if (p) {
            p->on_unload();
            if (destroy_func) destroy_func(p);
            else delete p;
        }
    });
    record.dll_handle = hModule;
    record.destroy_func = destroy_func;
    record.is_loaded = true;
    record.last_used = std::chrono::steady_clock::now();

    plugins_[meta.id] = record;
    std::cout << "✨ [Plugin Loaded] " << meta.name << " v" << meta.version << " by " << meta.author << " (" << meta.description << ")\n";
    return true;
#else
    void* handle = dlopen(filepath.c_str(), RTLD_NOW);
    if (!handle) {
        std::cerr << "⚠️ Failed to dlopen plugin library: " << filepath << " (" << dlerror() << ")\n";
        return false;
    }

    auto create_func = (CreatePluginFunc)dlsym(handle, "create_haven_plugin");
    auto destroy_func = (DestroyPluginFunc)dlsym(handle, "destroy_haven_plugin");

    if (!create_func) {
        std::cerr << "⚠️ Plugin missing 'create_haven_plugin': " << filepath << "\n";
        dlclose(handle);
        return false;
    }

    IHavenPlugin* raw_instance = create_func();
    if (!raw_instance) {
        dlclose(handle);
        return false;
    }

    PluginMetadata meta = raw_instance->get_metadata();
    if (!raw_instance->on_load(ctx ? ctx : execution_ctx_)) {
        if (destroy_func) destroy_func(raw_instance);
        else delete raw_instance;
        dlclose(handle);
        return false;
    }

    LoadedPluginRecord record;
    record.filepath = filepath;
    record.metadata = meta;
    for (const auto& act : KNOWN_TOOL_ACTIONS) {
        if (raw_instance->can_handle_tool(act)) {
            record.supported_actions.push_back(act);
        }
    }
    record.instance = std::shared_ptr<IHavenPlugin>(raw_instance, [destroy_func](IHavenPlugin* p) {
        if (p) {
            p->on_unload();
            if (destroy_func) destroy_func(p);
            else delete p;
        }
    });
    record.so_handle = handle;
    record.destroy_func = destroy_func;
    record.is_loaded = true;
    record.last_used = std::chrono::steady_clock::now();

    plugins_[meta.id] = record;
    std::cout << "✨ [Plugin Loaded] " << meta.name << " v" << meta.version << " (" << meta.description << ")\n";
    return true;
#endif
}

bool PluginManager::register_native_plugin(std::shared_ptr<IHavenPlugin> plugin, PluginExecutionContext* ctx) {
    if (!plugin) return false;
    PluginMetadata meta = plugin->get_metadata();
    if (!plugin->on_load(ctx ? ctx : execution_ctx_)) return false;

    LoadedPluginRecord record;
    record.filepath = "[in-process]";
    record.metadata = meta;
    for (const auto& act : KNOWN_TOOL_ACTIONS) {
        if (plugin->can_handle_tool(act)) {
            record.supported_actions.push_back(act);
        }
    }
    record.instance = plugin;
    record.is_loaded = true;
    record.last_used = std::chrono::steady_clock::now();
    plugins_[meta.id] = record;

    std::cout << "✨ [In-Process Plugin Registered] " << meta.name << " (" << meta.description << ")\n";
    return true;
}

bool PluginManager::ensure_plugin_loaded(LoadedPluginRecord& record, PluginExecutionContext* ctx) {
    if (record.is_loaded && record.instance) {
        record.last_used = std::chrono::steady_clock::now();
        return true;
    }

    if (record.filepath == "[in-process]" || record.filepath.empty()) {
        return record.instance != nullptr;
    }

    PluginExecutionContext* actual_ctx = ctx ? ctx : execution_ctx_;

#ifdef _WIN32
    HMODULE hModule = LoadLibraryA(record.filepath.c_str());
    if (!hModule) {
        std::cerr << "⚠️ [Hot-Load Error] Failed to LoadLibraryA: " << record.filepath << " (Error " << GetLastError() << ")\n";
        return false;
    }
    auto create_func = reinterpret_cast<CreatePluginFunc>(GetProcAddress(hModule, "create_haven_plugin"));
    auto destroy_func = reinterpret_cast<DestroyPluginFunc>(GetProcAddress(hModule, "destroy_haven_plugin"));
    if (!create_func) {
        FreeLibrary(hModule);
        return false;
    }
    IHavenPlugin* raw_instance = create_func();
    if (!raw_instance) {
        FreeLibrary(hModule);
        return false;
    }
    if (!raw_instance->on_load(actual_ctx)) {
        std::cerr << "⚠️ [Hot-Load Error] Plugin on_load() failed: " << record.metadata.name << "\n";
        if (destroy_func) destroy_func(raw_instance);
        else delete raw_instance;
        FreeLibrary(hModule);
        return false;
    }

    record.dll_handle = hModule;
    record.destroy_func = destroy_func;
    record.instance = std::shared_ptr<IHavenPlugin>(raw_instance, [destroy_func](IHavenPlugin* p) {
        if (p) {
            p->on_unload();
            if (destroy_func) destroy_func(p);
            else delete p;
        }
    });
    record.is_loaded = true;
    record.last_used = std::chrono::steady_clock::now();
    std::cout << "⚡ [Hot-Loaded Plugin on Demand] " << record.metadata.name << " v" << record.metadata.version << " (Now Active in RAM)\n";
    return true;
#else
    void* handle = dlopen(record.filepath.c_str(), RTLD_NOW);
    if (!handle) {
        std::cerr << "⚠️ [Hot-Load Error] Failed dlopen: " << record.filepath << " (" << dlerror() << ")\n";
        return false;
    }
    auto create_func = (CreatePluginFunc)dlsym(handle, "create_haven_plugin");
    auto destroy_func = (DestroyPluginFunc)dlsym(handle, "destroy_haven_plugin");
    if (!create_func) {
        dlclose(handle);
        return false;
    }
    IHavenPlugin* raw_instance = create_func();
    if (!raw_instance) {
        dlclose(handle);
        return false;
    }
    if (!raw_instance->on_load(actual_ctx)) {
        if (destroy_func) destroy_func(raw_instance);
        else delete raw_instance;
        dlclose(handle);
        return false;
    }
    record.so_handle = handle;
    record.destroy_func = destroy_func;
    record.instance = std::shared_ptr<IHavenPlugin>(raw_instance, [destroy_func](IHavenPlugin* p) {
        if (p) {
            p->on_unload();
            if (destroy_func) destroy_func(p);
            else delete p;
        }
    });
    record.is_loaded = true;
    record.last_used = std::chrono::steady_clock::now();
    std::cout << "⚡ [Hot-Loaded Plugin on Demand] " << record.metadata.name << "\n";
    return true;
#endif
}

bool PluginManager::ensure_plugin_loaded_by_id(const std::string& plugin_id, PluginExecutionContext* ctx) {
    auto it = plugins_.find(plugin_id);
    if (it == plugins_.end()) return false;
    return ensure_plugin_loaded(it->second, ctx);
}

bool PluginManager::unload_plugin_instance(LoadedPluginRecord& record) {
    if (!record.is_loaded) return true;
    if (record.filepath == "[in-process]") return false;

    std::string name = record.metadata.name;
    record.instance.reset(); // custom deleter triggers on_unload() and destroy_func()

#ifdef _WIN32
    if (record.dll_handle) {
        FreeLibrary(record.dll_handle);
        record.dll_handle = nullptr;
    }
#else
    if (record.so_handle) {
        dlclose(record.so_handle);
        record.so_handle = nullptr;
    }
#endif

    record.is_loaded = false;
    std::cout << "❄️ [Plugin Unloaded to Cold Storage] " << name << " (RAM Released)\n";
    return true;
}

bool PluginManager::unload_plugin(const std::string& plugin_id) {
    auto it = plugins_.find(plugin_id);
    if (it == plugins_.end()) return false;
    unload_plugin_instance(it->second);
    plugins_.erase(it);
    return true;
}

size_t PluginManager::unload_idle_plugins(int max_idle_seconds) {
    size_t count = 0;
    auto now = std::chrono::steady_clock::now();
    for (auto& [id, record] : plugins_) {
        if (record.is_loaded && record.filepath != "[in-process]") {
            auto idle_sec = std::chrono::duration_cast<std::chrono::seconds>(now - record.last_used).count();
            if (idle_sec >= max_idle_seconds) {
                if (unload_plugin_instance(record)) {
                    count++;
                }
            }
        }
    }
    return count;
}

void PluginManager::unload_all() {
    for (auto& [id, record] : plugins_) {
        if (record.is_loaded) {
            unload_plugin_instance(record);
        }
    }
    plugins_.clear();
}

size_t PluginManager::get_active_plugin_count() const {
    size_t count = 0;
    for (const auto& [id, record] : plugins_) {
        if (record.is_loaded) count++;
    }
    return count;
}

void PluginManager::dispatch_prompt_prefill(std::string& prompt, std::vector<uint32_t>& tokens) {
    for (auto& [id, record] : plugins_) {
        if (record.is_loaded && record.instance) {
            record.instance->on_prompt_prefill(prompt, tokens);
        }
    }
}

void PluginManager::dispatch_token_generated(uint32_t token_id, const std::string& piece) {
    for (auto& [id, record] : plugins_) {
        if (record.is_loaded && record.instance) {
            record.instance->on_token_generated(token_id, piece);
        }
    }
}

bool PluginManager::dispatch_tool_execution(const std::string& action, const std::string& payload, std::string& output) {
    // 1. Periodically prune idle plugins (older than 180s)
    unload_idle_plugins(180);

    // 2. Check currently active/loaded plugins first
    for (auto& [id, record] : plugins_) {
        if (record.is_loaded && record.instance && record.instance->can_handle_tool(action)) {
            record.last_used = std::chrono::steady_clock::now();
            return record.instance->execute_tool(action, payload, output);
        }
    }

    // 3. Find cold plugin that matches this action from discovery probe
    for (auto& [id, record] : plugins_) {
        if (!record.is_loaded) {
            bool matches = false;
            for (const auto& act : record.supported_actions) {
                if (act == action) {
                    matches = true;
                    break;
                }
            }

            if (matches) {
                if (ensure_plugin_loaded(record)) {
                    if (record.instance->can_handle_tool(action)) {
                        record.last_used = std::chrono::steady_clock::now();
                        return record.instance->execute_tool(action, payload, output);
                    }
                }
            }
        }
    }

    // 4. Fallback: on-demand probe cold plugins for custom or dynamic actions
    for (auto& [id, record] : plugins_) {
        if (!record.is_loaded) {
            if (ensure_plugin_loaded(record)) {
                if (record.instance->can_handle_tool(action)) {
                    record.last_used = std::chrono::steady_clock::now();
                    return record.instance->execute_tool(action, payload, output);
                } else {
                    unload_plugin_instance(record);
                }
            }
        }
    }

    // 5. Check connected Model Context Protocol (MCP) servers
    if (mcp_manager_ && mcp_manager_->can_handle_tool(action)) {
        return mcp_manager_->dispatch_tool(action, payload, output);
    }

    return false;
}

void PluginManager::dispatch_sensory_tick(std::vector<float>& vision_emb, std::vector<float>& audio_emb) {
    for (auto& [id, record] : plugins_) {
        if (record.is_loaded && record.instance) {
            record.instance->on_sensory_tick(vision_emb, audio_emb);
        }
    }
}

size_t PluginManager::reload_all(const std::string& plugins_directory, PluginExecutionContext* ctx) {
    plugins_dir_ = plugins_directory;
    unload_all();
    return discover_plugins(plugins_dir_, lazy_load_enabled_);
}

std::string PluginManager::get_tools_prompt_description() const {
    if (plugins_.empty()) return "";
    std::string s = "\n[Autonomous Sovereign Capabilities & Native Tools]:\n"
                    "You have direct native system execution and workspace inspection capabilities via C++20 bare-metal plugins.\n"
                    "Active Sovereign Plugins:\n";
    for (const auto& [id, record] : plugins_) {
        std::string status = record.is_loaded ? "ACTIVE" : "ON-DEMAND";
        s += " • " + record.metadata.name + " [" + status + "]: " + record.metadata.description + "\n";
    }
    s += "\nSupported Native Action Tools:\n"
         " • run_command <cmd>          : Run local shell, PowerShell, compiler, or diagnostic commands on Windows.\n"
         " • read_file <path>           : Read source code or text files from the local filesystem.\n"
         " • write_file <path>\n<content>: Create or overwrite a local project file.\n"
         " • edit_file {\"path\": \"...\", \"target\": \"...\", \"replacement\": \"...\"} : Replace exact text block in a file.\n"
         " • list_directory <path>      : List folder contents and file sizes.\n"
         " • search_code <query>        : Recursive pattern/code search across files.\n"
         " • process_list               : List active Windows processes and thread counts.\n"
         " • volume_info                : Inspect storage drives, file systems (NTFS/ReFS/etc.), and free space.\n"
         " • hardware_vitals            : Bare-metal CPU threads, RAM metrics, and uptime.\n"
         " • system_telemetry           : Real-time local clock, memory pressure, and AC/battery state.\n"
         " • clipboard_get              : Inspect current clipboard text.\n"
         "\nTo invoke a tool to inspect or act on the system, output immediately:\n"
         "<|tool_call|> tool_name payload <|tool_call|>\n"
         "Example:\n"
         "<|tool_call|> volume_info <|tool_call|>\n"
         "Example:\n"
         "<|tool_call|> run_command git status <|tool_call|>\n";

    if (mcp_manager_) {
        s += mcp_manager_->get_tools_prompt_description();
    }

    return s;
}

} // namespace haven