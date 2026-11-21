#include "haven/haven_plugin.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <filesystem>
#include <chrono>
#include <ctime>

namespace haven {

class VaultSyncPlugin : public IHavenPlugin {
public:
    PluginMetadata get_metadata() const override {
        PluginMetadata meta;
        meta.id = "haven.plugin.vault_sync";
        meta.name = "Obsidian Vault Sync & Companion Diary";
        meta.version = "1.0.0";
        meta.author = "Haven Sovereign Core";
        meta.description = "Searches markdown knowledge vaults, syncs project documentation, and maintains Aura's personal companion journal";
        meta.capabilities = { PluginCapability::CognitiveMemory, PluginCapability::ActionTool };
        return meta;
    }

    bool on_load(PluginExecutionContext* ctx) override {
        (void)ctx;
        vault_dir_ = "vault";
        if (!std::filesystem::exists(vault_dir_)) {
            try { std::filesystem::create_directories(vault_dir_); } catch (...) {}
        }
        std::cout << "📚 [VaultSyncPlugin] Markdown vault & companion diary initialized.\n";
        return true;
    }

    void on_unload() override {
        std::cout << "📚 [VaultSyncPlugin] Unloaded.\n";
    }

    bool can_handle_tool(const std::string& action) const override {
        return (action == "vault_search" ||
                action == "vault_read" ||
                action == "vault_list" ||
                action == "diary_write" ||
                action == "diary_read");
    }

    bool execute_tool(const std::string& action, const std::string& payload, std::string& output) override {
        if (action == "vault_list") {
            std::ostringstream ss;
            ss << "📚 Notes in Vault (" << vault_dir_ << "):\n";
            int count = 0;
            if (std::filesystem::exists(vault_dir_)) {
                for (const auto& entry : std::filesystem::directory_iterator(vault_dir_)) {
                    if (entry.is_regular_file() && entry.path().extension() == ".md") {
                        ss << "   • " << entry.path().filename().string() << " (" 
                           << entry.file_size() << " bytes)\n";
                        count++;
                    }
                }
            }
            if (count == 0) ss << "   (Vault directory is empty. Add .md notes to " << vault_dir_ << ")\n";
            output = ss.str();
            return true;
        }
        else if (action == "vault_search" && !payload.empty()) {
            std::ostringstream ss;
            ss << "🔍 Vault Search Results for '" << payload << "':\n";
            int matches = 0;
            if (std::filesystem::exists(vault_dir_)) {
                for (const auto& entry : std::filesystem::directory_iterator(vault_dir_)) {
                    if (entry.is_regular_file() && entry.path().extension() == ".md") {
                        std::ifstream file(entry.path());
                        std::string line;
                        int line_no = 1;
                        while (std::getline(file, line)) {
                            if (line.find(payload) != std::string::npos) {
                                ss << "   [" << entry.path().filename().string() << ":" << line_no << "] " 
                                   << line.substr(0, 100) << "\n";
                                matches++;
                                if (matches >= 8) break;
                            }
                            line_no++;
                        }
                    }
                }
            }
            if (matches == 0) ss << "   No notes matching '" << payload << "' found in vault.\n";
            output = ss.str();
            return true;
        }
        else if (action == "vault_read" && !payload.empty()) {
            if (payload.find("..") != std::string::npos || payload.find('/') != std::string::npos ||
                payload.find('\\') != std::string::npos || payload.find(':') != std::string::npos ||
                payload.find('\0') != std::string::npos) {
                output = "⚠️ Blocked: Invalid note path outside vault.";
                return false;
            }
            std::error_code ec;
            std::filesystem::path root = std::filesystem::weakly_canonical(vault_dir_, ec);
            std::filesystem::path rel = std::filesystem::path(payload).filename();
            if (!rel.has_extension()) rel += ".md";
            std::filesystem::path p = std::filesystem::weakly_canonical(root / rel, ec);
            std::string root_s = root.string();
            std::string p_s = p.string();
            if (ec || p_s.size() <= root_s.size() || p_s.compare(0, root_s.size(), root_s) != 0 ||
                (p_s[root_s.size()] != '\\' && p_s[root_s.size()] != '/')) {
                output = "⚠️ Blocked: Path escapes vault directory.";
                return false;
            }
            if (!std::filesystem::exists(p) || !std::filesystem::is_regular_file(p)) {
                output = "⚠️ Note not found: " + p.filename().string();
                return false;
            }
            std::ifstream file(p);
            std::stringstream buffer;
            buffer << file.rdbuf();
            output = "📖 [" + p.filename().string() + "]:\n" + buffer.str().substr(0, 2048);
            return true;
        }
        else if (action == "diary_write" && !payload.empty()) {
            auto now = std::chrono::system_clock::now();
            std::time_t now_c = std::chrono::system_clock::to_time_t(now);
            std::tm local_tm;
#ifdef _WIN32
            localtime_s(&local_tm, &now_c);
#else
            localtime_r(&now_c, &local_tm);
#endif
            char time_buf[64];
            std::strftime(time_buf, sizeof(time_buf), "%Y-%m-%d %I:%M %p", &local_tm);

            std::ofstream diary_file(vault_dir_ + "/Aura_Chronicle.md", std::ios::app);
            diary_file << "\n### Entry — " << time_buf << "\n" << payload << "\n---\n";
            output = "✍️ Wrote new entry to Aura's Chronicle (" + std::string(time_buf) + ")";
            return true;
        }
        else if (action == "diary_read") {
            std::string path = vault_dir_ + "/Aura_Chronicle.md";
            if (!std::filesystem::exists(path)) {
                output = "✍️ Aura's Chronicle has no entries yet. Use 'diary_write' to record a reflection!";
                return true;
            }
            std::ifstream file(path);
            std::stringstream buffer;
            buffer << file.rdbuf();
            output = "📖 Aura's Chronicle Journal:\n" + buffer.str().substr(0, 3000);
            return true;
        }

        output = "⚠️ Unrecognized vault action: " + action;
        return false;
    }

private:
    std::string vault_dir_;
};

} // namespace haven

// Native Dynamic Library Export Entrypoints
extern "C" {
#ifdef _WIN32
    HAVEN_PLUGIN_EXPORT haven::IHavenPlugin* create_haven_plugin() {
        return new haven::VaultSyncPlugin();
    }
    HAVEN_PLUGIN_EXPORT void destroy_haven_plugin(haven::IHavenPlugin* plugin) {
        delete plugin;
    }
#else
    haven::IHavenPlugin* create_haven_plugin() {
        return new haven::VaultSyncPlugin();
    }
    void destroy_haven_plugin(haven::IHavenPlugin* plugin) {
        delete plugin;
    }
#endif
}
