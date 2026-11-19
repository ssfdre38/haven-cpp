#include "haven/haven_plugin.h"
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <filesystem>
#include <fstream>
#include <algorithm>

namespace haven {

class WorkspaceCoderPlugin : public IHavenPlugin {
public:
    PluginMetadata get_metadata() const override {
        PluginMetadata meta;
        meta.id = "haven.plugin.workspace_coder";
        meta.name = "Sovereign Workspace & Code Assistant";
        meta.version = "1.0.0";
        meta.author = "Haven Sovereign Core";
        meta.description = "Inspects local project files, reads source code, lists directories, and searches codebases with zero cloud leakage";
        meta.capabilities = { PluginCapability::ActionTool, PluginCapability::CognitiveMemory };
        return meta;
    }

    bool on_load(PluginExecutionContext* ctx) override {
        (void)ctx;
        std::cout << "📁 [WorkspaceCoderPlugin] Local workspace and code analysis engine active.\n";
        return true;
    }

    void on_unload() override {
        std::cout << "📁 [WorkspaceCoderPlugin] Unloaded.\n";
    }

    bool can_handle_tool(const std::string& action) const override {
        return (action == "read_file" ||
                action == "read_code" ||
                action == "list_directory" ||
                action == "list_files" ||
                action == "search_code" ||
                action == "file_info");
    }

    bool execute_tool(const std::string& action, const std::string& payload, std::string& output) override {
        if (action == "read_file" || action == "read_code") {
            if (payload.empty()) {
                output = "📁 [WorkspaceCoder] Please specify a file path to read.";
                return true;
            }

            std::string path_str = payload;
            // Trim quotes if passed
            if (path_str.front() == '"' || path_str.front() == '\'') path_str = path_str.substr(1);
            if (path_str.back() == '"' || path_str.back() == '\'') path_str.pop_back();

            if (!std::filesystem::exists(path_str)) {
                output = "📁 [WorkspaceCoder] File not found: " + path_str;
                return true;
            }

            try {
                auto sz = std::filesystem::file_size(path_str);
                if (sz > 500000) { // 500KB cap
                    output = "📁 [WorkspaceCoder] File is too large (" + std::to_string(sz / 1024) + " KB) to read entirely in one turn.";
                    return true;
                }

                std::ifstream ifs(path_str);
                if (!ifs.is_open()) {
                    output = "📁 [WorkspaceCoder] Failed to open file: " + path_str;
                    return true;
                }

                std::string content((std::istreambuf_iterator<char>(ifs)), (std::istreambuf_iterator<char>()));
                if (content.length() > 5000) {
                    content = content.substr(0, 5000) + "\n\n... [Truncated for attention efficiency]";
                }

                output = "📁 File Content [" + path_str + "]:\n```\n" + content + "\n```";
                return true;
            } catch (const std::exception& e) {
                output = "📁 [WorkspaceCoder] Error reading file: " + std::string(e.what());
                return true;
            }
        }

        if (action == "list_directory" || action == "list_files") {
            std::string target_dir = payload.empty() ? "." : payload;
            if (!std::filesystem::exists(target_dir)) {
                output = "📁 [WorkspaceCoder] Directory not found: " + target_dir;
                return true;
            }

            std::stringstream ss;
            ss << "📁 Directory Listing [" << target_dir << "]:\n";
            int count = 0;
            try {
                for (const auto& entry : std::filesystem::directory_iterator(target_dir)) {
                    if (count++ > 50) {
                        ss << "   ... and more items\n";
                        break;
                    }
                    std::string name = entry.path().filename().string();
                    if (entry.is_directory()) {
                        ss << "   📂 " << name << "/\n";
                    } else {
                        auto sz = entry.file_size();
                        ss << "   📄 " << name << " (" << sz << " bytes)\n";
                    }
                }
                output = ss.str();
                return true;
            } catch (const std::exception& e) {
                output = "📁 [WorkspaceCoder] Error listing directory: " + std::string(e.what());
                return true;
            }
        }

        if (action == "search_code") {
            std::string query = payload;
            if (query.empty()) {
                output = "📁 [WorkspaceCoder] Please specify a query to search.";
                return true;
            }

            std::string query_lower = query;
            for (char& c : query_lower) c = (char)tolower(c);

            std::stringstream ss;
            ss << "🔍 Code Search Results for '" << query << "':\n";
            int matches = 0;

            try {
                for (const auto& entry : std::filesystem::recursive_directory_iterator(".")) {
                    if (entry.is_regular_file()) {
                        std::string ext = entry.path().extension().string();
                        if (ext == ".cpp" || ext == ".h" || ext == ".hpp" || ext == ".py" || ext == ".json" || ext == ".md" || ext == ".bat" || ext == ".ps1") {
                            std::ifstream ifs(entry.path().string());
                            std::string line;
                            int line_num = 1;
                            while (std::getline(ifs, line)) {
                                std::string l_lower = line;
                                for (char& c : l_lower) c = (char)tolower(c);
                                if (l_lower.find(query_lower) != std::string::npos) {
                                    ss << "   📍 " << entry.path().string() << ":" << line_num << " -> " << line.substr(0, 100) << "\n";
                                    matches++;
                                    if (matches >= 15) break;
                                }
                                line_num++;
                            }
                        }
                    }
                    if (matches >= 15) break;
                }
                if (matches == 0) ss << "   No matches found.\n";
                output = ss.str();
                return true;
            } catch (const std::exception& e) {
                output = "📁 [WorkspaceCoder] Search error: " + std::string(e.what());
                return true;
            }
        }

        if (action == "file_info") {
            if (!std::filesystem::exists(payload)) {
                output = "📁 [WorkspaceCoder] Target not found: " + payload;
                return true;
            }
            auto sz = std::filesystem::is_regular_file(payload) ? std::filesystem::file_size(payload) : 0;
            output = "📁 Target Info: " + payload + " | Size: " + std::to_string(sz) + " bytes | Type: " + 
                     (std::filesystem::is_directory(payload) ? "Directory" : "File");
            return true;
        }

        return false;
    }
};

} // namespace haven

HAVEN_PLUGIN_EXPORT haven::IHavenPlugin* create_haven_plugin() {
    return new haven::WorkspaceCoderPlugin();
}

HAVEN_PLUGIN_EXPORT void destroy_haven_plugin(haven::IHavenPlugin* plugin) {
    delete plugin;
}
