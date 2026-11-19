#include "haven/haven_plugin.h"
#include <iostream>
#include <sstream>
#include <chrono>
#include <ctime>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <psapi.h>
#endif

namespace haven {

class SystemSensoryPlugin : public IHavenPlugin {
public:
    PluginMetadata get_metadata() const override {
        PluginMetadata meta;
        meta.id = "haven.plugin.system_sensory";
        meta.name = "Desktop System & Environment Sensory Bridge";
        meta.version = "1.0.0";
        meta.author = "Haven Sovereign Core";
        meta.description = "Injects real-time local time, RAM usage, and battery/power state into Aura's sensory context";
        meta.capabilities = { PluginCapability::SensoryInput, PluginCapability::PromptFilter };
        return meta;
    }

    bool on_load(PluginExecutionContext* ctx) override {
        std::cout << "🖥️ [SystemSensoryPlugin] Desktop environment telemetry active.\n";
        return true;
    }

    void on_unload() override {
        std::cout << "🖥️ [SystemSensoryPlugin] Unloaded.\n";
    }

    void on_prompt_prefill(std::string& prompt, std::vector<uint32_t>& tokens) override {
        // Automatically inject ambient sensory context (time of day, system status) into system context
        auto now = std::chrono::system_clock::now();
        std::time_t now_c = std::chrono::system_clock::to_time_t(now);
        std::tm local_tm;
#ifdef _WIN32
        localtime_s(&local_tm, &now_c);
#else
        localtime_r(&now_c, &local_tm);
#endif
        char time_buf[64];
        std::strftime(time_buf, sizeof(time_buf), "%A, %B %d, %Y - %I:%M %p", &local_tm);

#ifdef _WIN32
        MEMORYSTATUSEX mem_info;
        mem_info.dwLength = sizeof(MEMORYSTATUSEX);
        GlobalMemoryStatusEx(&mem_info);
        int mem_percent = (int)mem_info.dwMemoryLoad;

        SYSTEM_POWER_STATUS power;
        GetSystemPowerStatus(&power);
        int battery_life = (int)power.BatteryLifePercent;
        bool on_ac = (power.ACLineStatus == 1);
#else
        int mem_percent = 50;
        int battery_life = 100;
        bool on_ac = true;
#endif

        std::ostringstream ss;
        ss << "\n[System Environment: Current Local Time is " << time_buf 
           << " | System Memory Load: " << mem_percent << "%"
           << " | Power: " << (on_ac ? "Plugged in (AC)" : "On Battery (" + std::to_string(battery_life) + "%)")
           << "]\n";

        // Insert environment anchor seamlessly before model turn
        size_t model_pos = prompt.rfind("<|turn>model");
        if (model_pos != std::string::npos) {
            prompt.insert(model_pos, ss.str());
        }
    }

    bool can_handle_tool(const std::string& action) const override {
        return (action == "system_telemetry" || action == "get_time" || action == "get_battery");
    }

    bool execute_tool(const std::string& action, const std::string& payload, std::string& output) override {
        auto now = std::chrono::system_clock::now();
        std::time_t now_c = std::chrono::system_clock::to_time_t(now);
        std::tm local_tm;
#ifdef _WIN32
        localtime_s(&local_tm, &now_c);
        MEMORYSTATUSEX mem_info;
        mem_info.dwLength = sizeof(MEMORYSTATUSEX);
        GlobalMemoryStatusEx(&mem_info);

        SYSTEM_POWER_STATUS power;
        GetSystemPowerStatus(&power);
#else
        localtime_r(&now_c, &local_tm);
#endif
        char time_buf[64];
        std::strftime(time_buf, sizeof(time_buf), "%I:%M %p (%Z)", &local_tm);

        std::ostringstream ss;
        ss << "🖥️ System Status Report:\n"
           << "   • Time: " << time_buf << "\n";
#ifdef _WIN32
        ss << "   • RAM Usage: " << mem_info.dwMemoryLoad << "% (" 
           << (mem_info.ullAvailPhys / (1024*1024*1024)) << " GB free of "
           << (mem_info.ullTotalPhys / (1024*1024*1024)) << " GB total)\n";
        if (power.BatteryLifePercent != 255) {
            ss << "   • Battery: " << (int)power.BatteryLifePercent << "% ("
               << (power.ACLineStatus == 1 ? "Charging" : "Discharging") << ")\n";
        }
#endif
        output = ss.str();
        return true;
    }
};

} // namespace haven

// Native Dynamic Library Export Entrypoints
extern "C" {
#ifdef _WIN32
    HAVEN_PLUGIN_EXPORT haven::IHavenPlugin* create_haven_plugin() {
        return new haven::SystemSensoryPlugin();
    }
    HAVEN_PLUGIN_EXPORT void destroy_haven_plugin(haven::IHavenPlugin* plugin) {
        delete plugin;
    }
#else
    haven::IHavenPlugin* create_haven_plugin() {
        return new haven::SystemSensoryPlugin();
    }
    void destroy_haven_plugin(haven::IHavenPlugin* plugin) {
        delete plugin;
    }
#endif
}
