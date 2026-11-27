#include "haven/haven_plugin.h"
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <fstream>
#include <filesystem>
#include <cstdio>
#include <ctime>

namespace haven {

struct RelationTriple {
    std::string subject;
    std::string predicate;
    std::string object;
    float confidence = 1.0f;
    int64_t timestamp = 0;
};

class SQLiteGraphPlugin : public IHavenPlugin {
public:
    PluginMetadata get_metadata() const override {
        PluginMetadata meta;
        meta.id = "haven.plugin.knowledge_graph";
        meta.name = "Persistent Knowledge Graph & Relational Memory";
        meta.version = "1.0.0";
        meta.author = "Haven Sovereign Core";
        meta.description = "Maintains structured (subject -> predicate -> object) relational memory triples with persistence and exact entity lookup";
        meta.capabilities = { PluginCapability::CognitiveMemory, PluginCapability::ActionTool };
        return meta;
    }

    bool on_load(PluginExecutionContext* ctx) override {
        (void)ctx;
        load_graph_from_disk();
        std::cout << "🧠 [SQLiteGraphPlugin] Relational knowledge graph active with " << triples_.size() << " structured relations.\n";
        return true;
    }

    void on_unload() override {
        save_graph_to_disk();
        std::cout << "🧠 [SQLiteGraphPlugin] Unloaded.\n";
    }

    bool can_handle_tool(const std::string& action) const override {
        return (action == "graph_add_relation" ||
                action == "graph_query_entity" ||
                action == "graph_list_all" ||
                action == "graph_stats");
    }

    bool execute_tool(const std::string& action, const std::string& payload, std::string& output) override {
        if (action == "graph_stats") {
            output = "🧠 Knowledge Graph Stats:\n   Total Relational Edges: " + std::to_string(triples_.size()) + 
                     "\n   Storage: wwwroot/knowledge_graph.json\n   Structure: (Subject -> Predicate -> Object)";
            return true;
        }

        if (action == "graph_add_relation") {
            // Payload format: "Subject | Predicate | Object"
            std::stringstream ss(payload);
            std::string s, p, o;
            std::getline(ss, s, '|');
            std::getline(ss, p, '|');
            std::getline(ss, o, '|');

            auto trim = [](std::string& str) {
                size_t f = str.find_first_not_of(" \t\r\n");
                size_t l = str.find_last_not_of(" \t\r\n");
                if (f != std::string::npos && l != std::string::npos) str = str.substr(f, l - f + 1);
            };
            trim(s); trim(p); trim(o);

            if (s.empty() || p.empty() || o.empty()) {
                output = "🧠 [KnowledgeGraph] Format error. Please use: `Subject | Predicate | Object` (e.g. `Daniel | creator_of | Sanctuary`)";
                return true;
            }

            RelationTriple rt;
            rt.subject = s;
            rt.predicate = p;
            rt.object = o;
            rt.timestamp = std::time(nullptr);
            triples_.push_back(rt);
            save_graph_to_disk();

            output = "🧠 Inscribed Relational Edge: (" + s + " -> [" + p + "] -> " + o + ")";
            return true;
        }

        if (action == "graph_query_entity") {
            std::string target = payload;
            for (char& c : target) c = (char)tolower(c);

            std::stringstream ss;
            ss << "🧠 Relational Graph Matches for [" << payload << "]:\n";
            int count = 0;

            for (const auto& t : triples_) {
                std::string s_low = t.subject, p_low = t.predicate, o_low = t.object;
                for (char& c : s_low) c = (char)tolower(c);
                for (char& c : p_low) c = (char)tolower(c);
                for (char& c : o_low) c = (char)tolower(c);

                if (s_low.find(target) != std::string::npos || o_low.find(target) != std::string::npos || p_low.find(target) != std::string::npos) {
                    ss << "   🔗 (" << t.subject << ") ──[" << t.predicate << "]──> (" << t.object << ")\n";
                    count++;
                }
            }

            if (count == 0) ss << "   No direct relations found.\n";
            output = ss.str();
            return true;
        }

        if (action == "graph_list_all") {
            std::stringstream ss;
            ss << "🧠 Knowledge Graph (All " << triples_.size() << " Relations):\n";
            int count = 0;
            for (const auto& t : triples_) {
                ss << "   • (" << t.subject << ") ──[" << t.predicate << "]──> (" << t.object << ")\n";
                if (++count > 25) {
                    ss << "   ... and " << (triples_.size() - 25) << " more relations.\n";
                    break;
                }
            }
            output = ss.str();
            return true;
        }

        return false;
    }

private:
    std::vector<RelationTriple> triples_;

    static std::string extract_json_str(const std::string& obj, const std::string& key) {
        std::string needle = "\"" + key + "\"";
        size_t pos = obj.find(needle);
        if (pos == std::string::npos) return "";
        pos += needle.size();
        while (pos < obj.size() && (obj[pos] == ' ' || obj[pos] == '\t' || obj[pos] == '\r' || obj[pos] == '\n')) ++pos;
        if (pos >= obj.size() || obj[pos] != ':') return "";
        ++pos;
        while (pos < obj.size() && (obj[pos] == ' ' || obj[pos] == '\t' || obj[pos] == '\r' || obj[pos] == '\n')) ++pos;
        if (pos >= obj.size() || obj[pos] != '"') return "";
        ++pos;
        std::string out;
        while (pos < obj.size()) {
            char c = obj[pos++];
            if (c == '\\' && pos < obj.size()) {
                char n = obj[pos++];
                if (n == '"') out += '"';
                else if (n == '\\') out += '\\';
                else if (n == 'n') out += '\n';
                else if (n == 'r') out += '\r';
                else if (n == 't') out += '\t';
                else out += n;
            } else if (c == '"') {
                break;
            } else {
                out += c;
            }
        }
        return out;
    }

    void load_graph_from_disk() {
        triples_.clear();
        std::string path = "wwwroot/knowledge_graph.json";
        std::error_code ec;
        if (std::filesystem::exists(path, ec)) {
            std::ifstream ifs(path, std::ios::binary);
            if (ifs.is_open()) {
                std::string content((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
                size_t rel_pos = content.find("\"relations\"");
                size_t cur = (rel_pos != std::string::npos) ? content.find('[', rel_pos) : 0;
                while (cur != std::string::npos && cur < content.size()) {
                    size_t obj_start = content.find('{', cur);
                    if (obj_start == std::string::npos) break;
                    bool in_str = false;
                    size_t obj_end = obj_start;
                    for (; obj_end < content.size(); ++obj_end) {
                        if (content[obj_end] == '"' && (obj_end == 0 || content[obj_end - 1] != '\\')) {
                            in_str = !in_str;
                        } else if (!in_str && content[obj_end] == '}') {
                            ++obj_end;
                            break;
                        }
                    }
                    if (obj_end <= obj_start) break;
                    std::string obj = content.substr(obj_start, obj_end - obj_start);
                    std::string s = extract_json_str(obj, "subject");
                    std::string p = extract_json_str(obj, "predicate");
                    std::string o = extract_json_str(obj, "object");
                    if (!s.empty() && !p.empty() && !o.empty()) {
                        triples_.push_back({s, p, o, 1.0f, 0});
                    }
                    cur = obj_end;
                }
            }
        }

        if (triples_.empty()) {
            // Seed default anchors only when no persisted relations exist on disk
            triples_.push_back({"haven-cpp", "runs_on", "Bare-Metal C++20 AVX2 SIMD", 1.0f, 0});
        }
    }

    void save_graph_to_disk() {
        auto esc = [](const std::string& s) {
            std::string out;
            for (unsigned char c : s) {
                if (c == '"') out += "\\\"";
                else if (c == '\\') out += "\\\\";
                else if (c == '\n') out += "\\n";
                else if (c == '\r') out += "\\r";
                else if (c == '\t') out += "\\t";
                else if (c < 0x20) {
                    char buf[8];
                    snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else out += static_cast<char>(c);
            }
            return out;
        };
        std::string path = "wwwroot/knowledge_graph.json";
        std::string tmp_path = path + ".tmp";
        std::error_code ec;
        std::filesystem::create_directories("wwwroot", ec);
        {
            std::ofstream ofs(tmp_path, std::ios::binary | std::ios::trunc);
            if (!ofs.is_open()) return;
            ofs << "{\n  \"relations\": [\n";
            for (size_t i = 0; i < triples_.size(); ++i) {
                const auto& t = triples_[i];
                ofs << "    {\"subject\": \"" << esc(t.subject) << "\", \"predicate\": \"" << esc(t.predicate) << "\", \"object\": \"" << esc(t.object) << "\"}";
                if (i + 1 < triples_.size()) ofs << ",";
                ofs << "\n";
            }
            ofs << "  ]\n}\n";
            ofs.flush();
            if (!ofs.good()) return;
        }
        std::filesystem::rename(tmp_path, path, ec);
        if (ec) {
            std::filesystem::copy_file(tmp_path, path, std::filesystem::copy_options::overwrite_existing, ec);
            std::filesystem::remove(tmp_path, ec);
        }
    }
};

} // namespace haven

HAVEN_PLUGIN_EXPORT haven::IHavenPlugin* create_haven_plugin() {
    return new haven::SQLiteGraphPlugin();
}

HAVEN_PLUGIN_EXPORT void destroy_haven_plugin(haven::IHavenPlugin* plugin) {
    delete plugin;
}
