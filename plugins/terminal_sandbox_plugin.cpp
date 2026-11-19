#include "haven/haven_plugin.h"
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <array>
#include <memory>

#ifdef _WIN32
#include <stdio.h>
#define POPEN _popen
#define PCLOSE _pclose
#else
#define POPEN popen
#define PCLOSE pclose
#endif

namespace haven {

class TerminalSandboxPlugin : public IHavenPlugin {
public:
    PluginMetadata get_metadata() const override {
        PluginMetadata meta;
        meta.id = "haven.plugin.terminal";
        meta.name = "Sovereign Terminal & Shell Sandbox";
        meta.version = "1.0.0";
        meta.author = "Haven Sovereign Core";
        meta.description = "Executes local shell commands, developer utilities, and diagnostic scripts in real time";
        meta.capabilities = { PluginCapability::ActionTool };
        return meta;
    }

    bool on_load(PluginExecutionContext* ctx) override {
        (void)ctx;
        std::cout << "🐚 [TerminalSandboxPlugin] Native shell execution sandbox online.\n";
        return true;
    }

    void on_unload() override {
        std::cout << "🐚 [TerminalSandboxPlugin] Unloaded.\n";
    }

    bool can_handle_tool(const std::string& action) const override {
        return (action == "run_command" ||
                action == "exec_shell" ||
                action == "cmd_run" ||
                action == "shell_status");
    }

    bool execute_tool(const std::string& action, const std::string& payload, std::string& output) override {
        if (action == "shell_status") {
            output = "🐚 Sovereign Shell Sandbox: ONLINE\nPlatform: Windows Native CLI / PowerShell\nStatus: Ready for command execution.";
            return true;
        }

        if (action == "run_command" || action == "exec_shell" || action == "cmd_run") {
            if (payload.empty()) {
                output = "🐚 [Terminal] Please provide a command line to execute.";
                return true;
            }

            std::string cmd = payload;
            // Append 2>&1 to capture both stdout and stderr
            std::string full_cmd = cmd + " 2>&1";

            std::array<char, 512> buffer;
            std::string result;

            FILE* pipe = POPEN(full_cmd.c_str(), "r");
            if (!pipe) {
                output = "🐚 [Terminal] Failed to launch process for command: " + cmd;
                return true;
            }

            while (fgets(buffer.data(), (int)buffer.size(), pipe) != nullptr) {
                result += buffer.data();
                if (result.length() > 4000) {
                    result += "\n... [Output truncated at 4000 characters]";
                    break;
                }
            }

            int exit_code = PCLOSE(pipe);
            
            std::stringstream ss;
            ss << "🐚 Terminal Output (`" << cmd << "` | Exit Code " << exit_code << "):\n```\n";
            ss << (result.empty() ? "[Command completed with no stdout output]" : result);
            ss << "\n```";

            output = ss.str();
            return true;
        }

        return false;
    }
};

} // namespace haven

HAVEN_PLUGIN_EXPORT haven::IHavenPlugin* create_haven_plugin() {
    return new haven::TerminalSandboxPlugin();
}

HAVEN_PLUGIN_EXPORT void destroy_haven_plugin(haven::IHavenPlugin* plugin) {
    delete plugin;
}
