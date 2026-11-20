#include "haven/haven_plugin.h"
#include <iostream>
#include <sstream>
#include <chrono>
#include <ctime>
#include <thread>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <psapi.h>
#include <tlhelp32.h>
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
        return (action == "system_telemetry" || action == "get_time" || action == "get_battery" ||
                action == "process_list" || action == "list_processes" ||
                action == "volume_info" || action == "disk_usage" ||
                action == "hardware_vitals");
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

        if (action == "process_list" || action == "list_processes") {
            HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
            if (snap == INVALID_HANDLE_VALUE) {
                output = "🖥️ [SystemSensory] Failed to create process snapshot.";
                return true;
            }

            PROCESSENTRY32 pe;
            pe.dwSize = sizeof(PROCESSENTRY32);
            std::ostringstream ss;
            ss << "🖥️ Active Windows Processes (Top Active Snapshot):\n";
            ss << "   PID     | Threads | Process Name\n";
            ss << "   --------+---------+----------------------------\n";
            int count = 0;
            if (Process32First(snap, &pe)) {
                do {
                    if (count++ < 35) {
                        char line[128];
                        snprintf(line, sizeof(line), "   %-7lu | %-7lu | %s\n", pe.th32ProcessID, pe.cntThreads, pe.szExeFile);
                        ss << line;
                    }
                } while (Process32Next(snap, &pe));
            }
            CloseHandle(snap);
            ss << "   [Total processes enumerated: " << count << "]\n";
            output = ss.str();
            return true;
        }

        if (action == "volume_info" || action == "disk_usage") {
            char drives[512];
            DWORD len = GetLogicalDriveStringsA(sizeof(drives) - 1, drives);
            if (len == 0) {
                output = "🖥️ [SystemSensory] Failed to enumerate logical drives.";
                return true;
            }

            std::ostringstream ss;
            ss << "💾 Sovereign Disk & Volume Telemetry:\n";
            ss << "   Drive | File System | Total Space | Free Space | Label\n";
            ss << "   ------+-------------+-------------+------------+------------------\n";

            char* d = drives;
            while (*d) {
                char vol_name[MAX_PATH + 1] = {0};
                char fs_name[MAX_PATH + 1] = {0};
                DWORD serial = 0, max_comp = 0, flags = 0;
                ULARGE_INTEGER free_bytes, total_bytes, total_free;

                BOOL v_ok = GetVolumeInformationA(d, vol_name, sizeof(vol_name), &serial, &max_comp, &flags, fs_name, sizeof(fs_name));
                BOOL s_ok = GetDiskFreeSpaceExA(d, &free_bytes, &total_bytes, &total_free);

                if (v_ok && s_ok) {
                    double total_gb = (double)total_bytes.QuadPart / (1024.0 * 1024.0 * 1024.0);
                    double free_gb = (double)free_bytes.QuadPart / (1024.0 * 1024.0 * 1024.0);
                    char line[256];
                    snprintf(line, sizeof(line), "   %-5s | %-11s | %8.1f GB | %7.1f GB | %s\n",
                             d, fs_name, total_gb, free_gb, vol_name[0] ? vol_name : "[Local Disk]");
                    ss << line;
                }
                d += (int)strlen(d) + 1;
            }
            output = ss.str();
            return true;
        }

        if (action == "hardware_vitals") {
            ULONGLONG uptime_sec = GetTickCount64() / 1000;
            ULONGLONG uptime_hrs = uptime_sec / 3600;
            ULONGLONG uptime_min = (uptime_sec % 3600) / 60;
            unsigned int threads = std::thread::hardware_concurrency();

            std::ostringstream ss;
            ss << "⚡ Sovereign Hardware Vitals:\n"
               << "   • CPU Architecture: Bare-Metal x86_64 / AVX2 SIMD\n"
               << "   • Hardware Concurrency: " << threads << " logical threads\n"
               << "   • System Uptime: " << uptime_hrs << " hours, " << uptime_min << " minutes\n"
               << "   • Memory Load: " << mem_info.dwMemoryLoad << "%\n"
               << "   • Physical RAM: " << (mem_info.ullTotalPhys / (1024*1024*1024)) << " GB Total ("
               << (mem_info.ullAvailPhys / (1024*1024*1024)) << " GB Available)\n"
               << "   • Pagefile: " << (mem_info.ullTotalPageFile / (1024*1024*1024)) << " GB Total ("
               << (mem_info.ullAvailPageFile / (1024*1024*1024)) << " GB Free)\n";
            output = ss.str();
            return true;
        }
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
