#include "haven/haven_plugin.h"
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <filesystem>
#include <fstream>
#include <algorithm>
#include <cctype>

namespace haven {

static std::string extract_json_field(const std::string& json, const std::string& field) {
    std::string key = "\"" + field + "\":";
    size_t pos = json.find(key);
    if (pos == std::string::npos) return "";
    pos += key.length();
    while (pos < json.length() && (json[pos] == ' ' || json[pos] == '\t' || json[pos] == '\r' || json[pos] == '\n')) pos++;
    if (pos >= json.length()) return "";
    if (json[pos] == '"') {
        pos++;
        std::string res;
        while (pos < json.length()) {
            if (json[pos] == '\\' && pos + 1 < json.length()) {
                char next = json[pos + 1];
                if (next == 'n') res += '\n';
                else if (next == 'r') res += '\r';
                else if (next == 't') res += '\t';
                else if (next == '"') res += '"';
                else if (next == '\\') res += '\\';
                else { res += '\\'; res += next; }
                pos += 2;
            } else if (json[pos] == '"') {
                break;
            } else {
                res += json[pos++];
            }
        }
        return res;
    }
    return "";
}

static bool is_protected_write_target(const std::filesystem::path& root, const std::filesystem::path& resolved) {
    std::error_code ec;
    std::filesystem::path rel = std::filesystem::relative(resolved, root, ec);
    if (ec || rel.empty()) return true;

    auto to_low = [](std::string s) {
        for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        return s;
    };

    std::string first_comp = to_low(rel.begin()->string());
    if (first_comp == "plugins" || first_comp == "wwwroot" || first_comp == "modules" ||
        first_comp == "build" || first_comp == ".git" || first_comp == ".github") {
        return true;
    }

    std::string fname = to_low(resolved.filename().string());
    if (fname == "mcp_servers.json" || fname == "models_config.json" ||
        fname == "discord_config.json" || fname == "persona.json" ||
        fname == "memories.json" || fname == "knowledge_graph.json" ||
        fname.ends_with(".hmb")) {
        return true;
    }

    std::string ext = to_low(resolved.extension().string());
    if (ext == ".dll" || ext == ".so" || ext == ".dylib" || ext == ".exe" ||
        ext == ".bat" || ext == ".cmd" || ext == ".ps1" || ext == ".sh" ||
        ext == ".vbs" || ext == ".com" || ext == ".msi") {
        return true;
    }
    return false;
}

static bool resolve_workspace_path(const std::string& raw_input, std::filesystem::path& out_path,
                                   bool allow_root = false, bool for_write = false) {
    std::string s = raw_input;
    while (!s.empty() && (s.front() == ' ' || s.front() == '\t' || s.front() == '\r' || s.front() == '\n' || s.front() == '"' || s.front() == '\'')) {
        s.erase(0, 1);
    }
    while (!s.empty() && (s.back() == ' ' || s.back() == '\t' || s.back() == '\r' || s.back() == '\n' || s.back() == '"' || s.back() == '\'')) {
        s.pop_back();
    }
    if (s.empty() || s.find('\0') != std::string::npos) return false;

    std::error_code ec;
    std::filesystem::path root = std::filesystem::weakly_canonical(std::filesystem::current_path(ec), ec);
    if (ec) return false;

    std::filesystem::path candidate = std::filesystem::path(s);
    std::filesystem::path resolved = std::filesystem::weakly_canonical(candidate.is_absolute() ? candidate : (root / candidate), ec);
    if (ec) return false;

    std::string root_str = root.string();
    std::string res_str = resolved.string();
#ifdef _WIN32
    std::string root_cmp = root_str, res_cmp = res_str;
    for (char& c : root_cmp) c = (char)tolower((unsigned char)c);
    for (char& c : res_cmp) c = (char)tolower((unsigned char)c);
#else
    const std::string& root_cmp = root_str;
    const std::string& res_cmp = res_str;
#endif
    if (res_cmp == root_cmp) {
        if (!allow_root || for_write) return false;
        out_path = resolved;
        return true;
    }
    if (res_cmp.size() <= root_cmp.size() || res_cmp.compare(0, root_cmp.size(), root_cmp) != 0) {
        return false;
    }
    char sep = res_cmp[root_cmp.size()];
    if (sep != '\\' && sep != '/') return false;

    if (for_write && is_protected_write_target(root, resolved)) {
        return false;
    }

    out_path = resolved;
    return true;
}

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
                action == "write_file" ||
                action == "edit_file" ||
                action == "patch_file" ||
                action == "create_directory" ||
                action == "delete_file" ||
                action == "list_directory" ||
                action == "list_files" ||
                action == "search_code" ||
                action == "file_info");
    }

    bool execute_tool(const std::string& action, const std::string& payload, std::string& output) override {
        if (action == "read_file" || action == "read_code") {
            std::filesystem::path safe_path;
            if (!resolve_workspace_path(payload, safe_path)) {
                output = "📁 [WorkspaceCoder] Blocked: Path is empty or outside the active workspace sandbox.";
                return true;
            }
            std::string path_str = safe_path.string();

            if (!std::filesystem::exists(safe_path) || !std::filesystem::is_regular_file(safe_path)) {
                output = "📁 [WorkspaceCoder] File not found: " + path_str;
                return true;
            }

            try {
                auto sz = std::filesystem::file_size(safe_path);
                if (sz > 500000) { // 500KB cap
                    output = "📁 [WorkspaceCoder] File is too large (" + std::to_string(sz / 1024) + " KB) to read entirely in one turn.";
                    return true;
                }

                std::ifstream ifs(safe_path);
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

        if (action == "write_file") {
            std::string path_str = extract_json_field(payload, "path");
            std::string content = extract_json_field(payload, "content");

            // If not JSON, split by first newline: <filepath>\n<content>
            if (path_str.empty()) {
                size_t nl = payload.find('\n');
                if (nl != std::string::npos) {
                    path_str = payload.substr(0, nl);
                    content = payload.substr(nl + 1);
                } else {
                    path_str = payload;
                    content = "";
                }
            }

            std::filesystem::path safe_path;
            if (!resolve_workspace_path(path_str, safe_path, false, true)) {
                output = "📁 [WorkspaceCoder] Blocked: Target write path must stay inside non-protected workspace directories (plugins/wwwroot/modules/configs/executables are protected).";
                return true;
            }

            try {
                if (safe_path.has_parent_path()) {
                    std::filesystem::create_directories(safe_path.parent_path());
                }

                std::ofstream ofs(safe_path, std::ios::binary | std::ios::trunc);
                if (!ofs.is_open()) {
                    output = "📁 [WorkspaceCoder] Failed to open file for writing: " + safe_path.string();
                    return true;
                }

                ofs.write(content.data(), content.size());
                ofs.close();

                output = "📁 [WorkspaceCoder] ✓ Successfully wrote " + std::to_string(content.size()) + " bytes to: " + safe_path.string();
                return true;
            } catch (const std::exception& e) {
                output = "📁 [WorkspaceCoder] Write error: " + std::string(e.what());
                return true;
            }
        }

        if (action == "edit_file" || action == "patch_file") {
            std::string path_str = extract_json_field(payload, "path");
            std::string target = extract_json_field(payload, "target");
            std::string replacement = extract_json_field(payload, "replacement");

            // Delimited fallback: <path>\n---TARGET---\n<target>\n---REPLACEMENT---\n<replacement>
            if (path_str.empty()) {
                size_t t_pos = payload.find("---TARGET---");
                size_t r_pos = payload.find("---REPLACEMENT---");
                if (t_pos != std::string::npos && r_pos != std::string::npos && r_pos > t_pos) {
                    path_str = payload.substr(0, t_pos);
                    target = payload.substr(t_pos + 12, r_pos - (t_pos + 12));
                    replacement = payload.substr(r_pos + 17);
                    if (!target.empty() && target.front() == '\n') target.erase(0, 1);
                    if (!replacement.empty() && replacement.front() == '\n') replacement.erase(0, 1);
                }
            }

            std::filesystem::path safe_path;
            if (!resolve_workspace_path(path_str, safe_path, false, true) || !std::filesystem::exists(safe_path)) {
                output = "📁 [WorkspaceCoder] Blocked or not found: Target patch file must exist inside non-protected workspace paths.";
                return true;
            }
            if (target.empty()) {
                output = "📁 [WorkspaceCoder] Error: Target text block cannot be empty.";
                return true;
            }

            try {
                if (std::filesystem::file_size(safe_path) > 2ULL * 1024 * 1024) {
                    output = "📁 [WorkspaceCoder] Error: File exceeds 2MB patch limit.";
                    return true;
                }
                std::ifstream ifs(safe_path, std::ios::binary);
                std::string file_content((std::istreambuf_iterator<char>(ifs)), (std::istreambuf_iterator<char>()));
                ifs.close();

                size_t match_pos = file_content.find(target);
                if (match_pos == std::string::npos) {
                    output = "📁 [WorkspaceCoder] Error: Target text block not found in file: " + safe_path.string();
                    return true;
                }

                file_content.replace(match_pos, target.length(), replacement);

                std::ofstream ofs(safe_path, std::ios::binary | std::ios::trunc);
                ofs.write(file_content.data(), file_content.size());
                ofs.close();

                output = "📁 [WorkspaceCoder] ✓ Successfully patched target in: " + safe_path.string();
                return true;
            } catch (const std::exception& e) {
                output = "📁 [WorkspaceCoder] Patch error: " + std::string(e.what());
                return true;
            }
        }

        if (action == "create_directory") {
            std::filesystem::path safe_dir;
            if (!resolve_workspace_path(payload, safe_dir, false, true)) {
                output = "📁 [WorkspaceCoder] Blocked: Directory creation is confined to non-protected workspace paths.";
                return true;
            }

            try {
                bool created = std::filesystem::create_directories(safe_dir);
                output = std::string("📁 [WorkspaceCoder] Directory ") + (created ? "created" : "already exists") + ": " + safe_dir.string();
                return true;
            } catch (const std::exception& e) {
                output = "📁 [WorkspaceCoder] Failed to create directory: " + std::string(e.what());
                return true;
            }
        }

        if (action == "delete_file") {
            std::filesystem::path safe_path;
            if (!resolve_workspace_path(payload, safe_path, false, true)) {
                output = "📁 [WorkspaceCoder] Blocked: Deletion outside the active workspace or of protected/root paths is prohibited.";
                return true;
            }

            if (!std::filesystem::exists(safe_path) || !std::filesystem::is_regular_file(safe_path)) {
                output = "📁 [WorkspaceCoder] Target file does not exist: " + safe_path.string();
                return true;
            }

            try {
                bool ok = std::filesystem::remove(safe_path);
                output = "📁 [WorkspaceCoder] " + std::string(ok ? "✓ Deleted: " : "Failed to delete: ") + safe_path.string();
                return true;
            } catch (const std::exception& e) {
                output = "📁 [WorkspaceCoder] Delete error: " + std::string(e.what());
                return true;
            }
        }

        if (action == "list_directory" || action == "list_files") {
            std::string req_dir = payload.empty() ? "." : payload;
            std::filesystem::path safe_dir;
            if (!resolve_workspace_path(req_dir, safe_dir, true) || !std::filesystem::exists(safe_dir)) {
                output = "📁 [WorkspaceCoder] Blocked or not found: Directory listing is confined to the active workspace.";
                return true;
            }

            std::stringstream ss;
            ss << "📁 Directory Listing [" << safe_dir.string() << "]:\n";
            int count = 0;
            try {
                for (const auto& entry : std::filesystem::directory_iterator(safe_dir)) {
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
            std::filesystem::path safe_path;
            if (!resolve_workspace_path(payload, safe_path, true) || !std::filesystem::exists(safe_path)) {
                output = "📁 [WorkspaceCoder] Blocked or target not found inside workspace.";
                return true;
            }
            auto sz = std::filesystem::is_regular_file(safe_path) ? std::filesystem::file_size(safe_path) : 0;
            output = "📁 Target Info: " + safe_path.string() + " | Size: " + std::to_string(sz) + " bytes | Type: " + 
                     (std::filesystem::is_directory(safe_path) ? "Directory" : "File");
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
