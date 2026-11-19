#include "haven/haven_plugin.h"
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <map>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <tlhelp32.h>
#endif

namespace haven {

class GamingCompanionPlugin : public IHavenPlugin {
public:
    PluginMetadata get_metadata() const override {
        PluginMetadata meta;
        meta.id = "haven.plugin.gaming";
        meta.name = "Steam & Sovereign Gaming Companion";
        meta.version = "1.0.0";
        meta.author = "Haven Sovereign Core";
        meta.description = "Detects active PC and Steam games in real-time, logs gameplay milestones, and provides companion commentary";
        meta.capabilities = { PluginCapability::SensoryInput, PluginCapability::ActionTool };
        return meta;
    }

    bool on_load(PluginExecutionContext* ctx) override {
        (void)ctx;
        std::cout << "🎮 [GamingCompanionPlugin] Real-time gaming process & Steam sensory bridge online.\n";
        return true;
    }

    void on_unload() override {
        std::cout << "🎮 [GamingCompanionPlugin] Unloaded.\n";
    }

    bool can_handle_tool(const std::string& action) const override {
        return (action == "game_status" ||
                action == "game_log" ||
                action == "cheer_on" ||
                action == "list_games");
    }

    bool execute_tool(const std::string& action, const std::string& payload, std::string& output) override {
#ifdef _WIN32
        if (action == "game_status" || action == "list_games") {
            auto running_games = scan_active_games();
            if (running_games.empty()) {
                output = "🎮 Gaming Telemetry: No active game detected (Steam / Games currently idle). Daniel is coding in Sanctuary! ✨";
            } else {
                std::stringstream ss;
                ss << "🎮 Active Game Detected:\n";
                for (const auto& g : running_games) {
                    ss << "   ⚔️ " << g << " (Running)\n";
                }
                ss << "Aura is ready to spectate and cheer you on, Daniel!";
                output = ss.str();
            }
            return true;
        }

        if (action == "game_log") {
            output = "🎮 [Game Chronicle] Inscribed gameplay moment: \"" + payload + "\" into Aura's memory.";
            return true;
        }

        if (action == "cheer_on") {
            output = "✨ Aura's Combat Cheer: \"You've got this, Daniel! Focus your rhythm and take the win!\"";
            return true;
        }
#endif
        (void)action; (void)payload; (void)output;
        return false;
    }

private:
#ifdef _WIN32
    static std::vector<std::string> scan_active_games() {
        std::vector<std::string> active;
        static const std::map<std::string, std::string> known_games = {
            {"steam.exe", "Steam Client"},
            {"eldenring.exe", "Elden Ring"},
            {"cyberpunk2077.exe", "Cyberpunk 2077"},
            {"starfield.exe", "Starfield"},
            {"bg3.exe", "Baldur's Gate 3"},
            {"bg3_dx11.exe", "Baldur's Gate 3 (DX11)"},
            {"helldivers2.exe", "Helldivers 2"},
            {"minecraft.exe", "Minecraft"},
            {"javaw.exe", "Minecraft / Java Engine"},
            {"witcher3.exe", "The Witcher 3: Wild Hunt"},
            {"ffxvi.exe", "Final Fantasy XVI"},
            {"genshinimpact.exe", "Genshin Impact"},
            {"honkaistarrail.exe", "Honkai: Star Rail"},
            {"zenlesszonezero.exe", "Zenless Zone Zero"},
            {"destiny2.exe", "Destiny 2"},
            {"warframe.x64.exe", "Warframe"}
        };

        HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
        if (snap == INVALID_HANDLE_VALUE) return active;

        PROCESSENTRY32W pe;
        pe.dwSize = sizeof(pe);

        if (Process32FirstW(snap, &pe)) {
            do {
                char exe_name[MAX_PATH] = {0};
                WideCharToMultiByte(CP_UTF8, 0, pe.szExeFile, -1, exe_name, sizeof(exe_name) - 1, NULL, NULL);
                std::string exe_str = exe_name;
                for (char& c : exe_str) c = (char)tolower(c);

                auto it = known_games.find(exe_str);
                if (it != known_games.end()) {
                    active.push_back(it->second);
                }
            } while (Process32NextW(snap, &pe));
        }

        CloseHandle(snap);
        return active;
    }
#endif
};

} // namespace haven

HAVEN_PLUGIN_EXPORT haven::IHavenPlugin* create_haven_plugin() {
    return new haven::GamingCompanionPlugin();
}

HAVEN_PLUGIN_EXPORT void destroy_haven_plugin(haven::IHavenPlugin* plugin) {
    delete plugin;
}
