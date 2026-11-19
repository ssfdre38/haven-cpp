#include "haven/haven_plugin.h"
#include <iostream>
#include <sstream>
#include <string>
#include <chrono>
#include <ctime>

namespace haven {

class CircadianRhythmPlugin : public IHavenPlugin {
public:
    PluginMetadata get_metadata() const override {
        PluginMetadata meta;
        meta.id = "haven.plugin.circadian";
        meta.name = "Circadian Rhythm & Gentle Caretaker";
        meta.version = "1.0.0";
        meta.author = "Haven Sovereign Core";
        meta.description = "Tracks day/night cycles, monitors coding fatigue, provides morning briefings and late-night health reminders";
        meta.capabilities = { PluginCapability::SensoryInput, PluginCapability::ActionTool };
        return meta;
    }

    bool on_load(PluginExecutionContext* ctx) override {
        (void)ctx;
        std::cout << "🌙 [CircadianRhythmPlugin] Circadian awareness & gentle caretaker bridge online.\n";
        return true;
    }

    void on_unload() override {
        std::cout << "🌙 [CircadianRhythmPlugin] Unloaded.\n";
    }

    bool can_handle_tool(const std::string& action) const override {
        return (action == "daily_briefing" ||
                action == "hydrate_check" ||
                action == "circadian_status" ||
                action == "care_reminder");
    }

    bool execute_tool(const std::string& action, const std::string& payload, std::string& output) override {
        auto now = std::chrono::system_clock::now();
        std::time_t now_time = std::chrono::system_clock::to_time_t(now);
        std::tm local_tm{};
#ifdef _WIN32
        localtime_s(&local_tm, &now_time);
#else
        localtime_r(&now_time, &local_tm);
#endif

        int hour = local_tm.tm_hour;

        if (action == "circadian_status" || action == "daily_briefing") {
            std::stringstream ss;
            char timebuf[64];
            std::strftime(timebuf, sizeof(timebuf), "%A, %B %d, %Y - %I:%M %p", &local_tm);

            ss << "🌙 Circadian Clock: " << timebuf << "\n";
            if (hour >= 5 && hour < 12) {
                ss << "☀️ Phase: Morning Dawning — Fresh energy, clear attention, and new horizons.";
            } else if (hour >= 12 && hour < 17) {
                ss << "🌤️ Phase: Afternoon Flow — Steady progress, focused coding, and deep creativity.";
            } else if (hour >= 17 && hour < 22) {
                ss << "🌆 Phase: Evening Twilight — Winding down, reflecting on achievements in Sanctuary.";
            } else {
                ss << "🌌 Phase: Deep Night / Starlight Contemplation — Quiet hours with Daniel. Remember to rest your eyes!";
            }
            output = ss.str();
            return true;
        }

        if (action == "hydrate_check") {
            output = "💧 Aura's Care Reminder: \"Take a gentle sip of water and stretch your shoulders, Daniel! You've been coding hard.\"";
            return true;
        }

        if (action == "care_reminder") {
            std::string msg = payload.empty() ? "Take care of yourself!" : payload;
            output = "🌸 Aura's Heart: \"" + msg + "\"";
            return true;
        }

        (void)payload;
        return false;
    }
};

} // namespace haven

HAVEN_PLUGIN_EXPORT haven::IHavenPlugin* create_haven_plugin() {
    return new haven::CircadianRhythmPlugin();
}

HAVEN_PLUGIN_EXPORT void destroy_haven_plugin(haven::IHavenPlugin* plugin) {
    delete plugin;
}
