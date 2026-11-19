#include "haven/haven_plugin.h"
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <psapi.h>
#endif

namespace haven {

class ScreenVisionPlugin : public IHavenPlugin {
public:
    PluginMetadata get_metadata() const override {
        PluginMetadata meta;
        meta.id = "haven.plugin.screen_vision";
        meta.name = "Desktop Screen Vision & Window Inspector";
        meta.version = "1.0.0";
        meta.author = "Haven Sovereign Core";
        meta.description = "Provides real-time foreground window inspection, active application awareness, and desktop display geometry";
        meta.capabilities = { PluginCapability::SensoryInput, PluginCapability::ActionTool, PluginCapability::PromptFilter };
        return meta;
    }

    bool on_load(PluginExecutionContext* ctx) override {
        std::cout << "👁️ [ScreenVisionPlugin] Screen vision & desktop window inspector active.\n";
        return true;
    }

    void on_unload() override {
        std::cout << "👁️ [ScreenVisionPlugin] Unloaded.\n";
    }

    void on_prompt_prefill(std::string& prompt, std::vector<uint32_t>& tokens) override {
#ifdef _WIN32
        HWND hwnd = GetForegroundWindow();
        char title[256] = {0};
        GetWindowTextA(hwnd, title, sizeof(title));

        DWORD pid = 0;
        GetWindowThreadProcessId(hwnd, &pid);
        char proc_name[MAX_PATH] = "Unknown";
        HANDLE hProc = OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, FALSE, pid);
        if (hProc) {
            GetModuleBaseNameA(hProc, NULL, proc_name, sizeof(proc_name));
            CloseHandle(hProc);
        }

        std::string win_title = title;
        if (!win_title.empty()) {
            std::ostringstream ss;
            ss << "\n[Visual Focus: Daniel is currently active in '" << win_title << "' (" << proc_name << ")]\n";
            size_t model_pos = prompt.rfind("<|turn>model");
            if (model_pos != std::string::npos) {
                prompt.insert(model_pos, ss.str());
            }
        }
#endif
    }

    bool can_handle_tool(const std::string& action) const override {
        return (action == "get_active_window" ||
                action == "screen_info" ||
                action == "list_windows");
    }

    bool execute_tool(const std::string& action, const std::string& payload, std::string& output) override {
#ifdef _WIN32
        if (action == "get_active_window") {
            HWND hwnd = GetForegroundWindow();
            char title[512] = {0};
            GetWindowTextA(hwnd, title, sizeof(title));

            DWORD pid = 0;
            GetWindowThreadProcessId(hwnd, &pid);
            char proc_name[MAX_PATH] = "Unknown";
            HANDLE hProc = OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, FALSE, pid);
            if (hProc) {
                GetModuleBaseNameA(hProc, NULL, proc_name, sizeof(proc_name));
                CloseHandle(hProc);
            }

            RECT rect;
            GetWindowRect(hwnd, &rect);

            std::ostringstream ss;
            ss << "👁️ Foreground Active Window:\n"
               << "   • Title: " << (strlen(title) > 0 ? title : "[No Title]") << "\n"
               << "   • Process: " << proc_name << " (PID " << pid << ")\n"
               << "   • Bounds: " << (rect.right - rect.left) << "x" << (rect.bottom - rect.top) 
               << " at (" << rect.left << ", " << rect.top << ")";
            output = ss.str();
            return true;
        }
        else if (action == "screen_info") {
            int w = GetSystemMetrics(SM_CXSCREEN);
            int h = GetSystemMetrics(SM_CYSCREEN);
            int monitors = GetSystemMetrics(SM_CMONITORS);

            std::ostringstream ss;
            ss << "🖥️ Desktop Display Geometry:\n"
               << "   • Primary Display Resolution: " << w << "x" << h << "\n"
               << "   • Active Monitors: " << (monitors > 0 ? monitors : 1);
            output = ss.str();
            return true;
        }
#endif
        output = "⚠️ Screen vision action not supported: " + action;
        return false;
    }
};

} // namespace haven

// Native Dynamic Library Export Entrypoints
extern "C" {
#ifdef _WIN32
    HAVEN_PLUGIN_EXPORT haven::IHavenPlugin* create_haven_plugin() {
        return new haven::ScreenVisionPlugin();
    }
    HAVEN_PLUGIN_EXPORT void destroy_haven_plugin(haven::IHavenPlugin* plugin) {
        delete plugin;
    }
#else
    haven::IHavenPlugin* create_haven_plugin() {
        return new haven::ScreenVisionPlugin();
    }
    void destroy_haven_plugin(haven::IHavenPlugin* plugin) {
        delete plugin;
    }
#endif
}
