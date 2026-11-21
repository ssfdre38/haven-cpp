#include "haven/haven_plugin.h"
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <array>
#include <memory>
#include <chrono>
#include <filesystem>

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
#ifdef _WIN32
            output = "🐚 Sovereign Shell Sandbox: ONLINE\nPlatform: Windows Native CLI / PowerShell\nStatus: Ready for command execution.";
#elif defined(__APPLE__)
            output = "🐚 Sovereign Shell Sandbox: ONLINE\nPlatform: macOS / Darwin Zsh / Bash\nStatus: Ready for command execution.";
#else
            output = "🐚 Sovereign Shell Sandbox: ONLINE\nPlatform: Linux / POSIX Shell (/bin/sh, /bin/bash)\nStatus: Ready for command execution.";
#endif
            return true;
        }

        if (action == "run_command" || action == "exec_shell" || action == "cmd_run") {
            if (payload.empty()) {
                output = "🐚 [Terminal] Please provide a command line to execute.";
                return true;
            }

            std::string cmd = payload;
            std::string cwd_prefix = "";

            // Check for [cwd: <path>] header
            if (cmd.starts_with("[cwd:")) {
                size_t close_bracket = cmd.find(']');
                if (close_bracket != std::string::npos) {
                    std::string cwd_path = cmd.substr(5, close_bracket - 5);
                    while (!cwd_path.empty() && (cwd_path.front() == ' ' || cwd_path.front() == '"')) cwd_path.erase(0, 1);
                    while (!cwd_path.empty() && (cwd_path.back() == ' ' || cwd_path.back() == '"')) cwd_path.pop_back();
                    cmd = cmd.substr(close_bracket + 1);
                    while (!cmd.empty() && (cmd.front() == ' ' || cmd.front() == '\n' || cmd.front() == '\r')) cmd.erase(0, 1);
                    if (!cwd_path.empty()) {
                        const std::string bad_cwd_tokens[] = {"\"", "'", "|", "&", ";", "`", "$", "<", ">", "\n", "\r", "%", "^", "!"};
                        for (const auto& t : bad_cwd_tokens) {
                            if (cwd_path.find(t) != std::string::npos || cwd_path.find('\0') != std::string::npos) {
                                output = "🐚 [Terminal] Blocked: Invalid characters in working directory path.";
                                return true;
                            }
                        }
                        std::error_code ec;
                        if (!std::filesystem::exists(cwd_path, ec) || !std::filesystem::is_directory(cwd_path, ec)) {
                            output = "🐚 [Terminal] Blocked: Working directory does not exist: " + cwd_path;
                            return true;
                        }
#ifdef _WIN32
                        cwd_prefix = "cd /d \"" + cwd_path + "\" && ";
#else
                        cwd_prefix = "cd \"" + cwd_path + "\" && ";
#endif
                    }
                }
            }

            if (cmd.empty()) {
                output = "🐚 [Terminal] Please provide a command line to execute.";
                return true;
            }

            // Destructive & injection safety guardrail (cross-platform Windows & *nix)
            const std::string forbidden_tokens[] = {"|", "&", ";", "`", "$(", "<", ">", "\n", "\r", "%", "^", "!"};
            for (const auto& token : forbidden_tokens) {
                if (cmd.find(token) != std::string::npos || cmd.find('\0') != std::string::npos) {
                    output = "🐚 [Terminal] Blocked: Shell chaining, pipe, redirection, and expansion characters are disallowed in sovereign sandbox.";
                    return true;
                }
            }

            std::string cmd_lower = cmd;
            for (char& c : cmd_lower) c = (char)tolower((unsigned char)c);
            const std::string blocked_subcmds[] = {
                "format ", "rmdir /s", "del /s", "diskpart", "rm -rf /", "rm -rf /*",
                "rm -r /", "mkfs", "dd if=/dev/zero", "dd if=/dev/urandom",
                ":(){ :|:& };:", "chmod -r 777 /", "cmd /c", "cmd.exe", "powershell",
                "pwsh", "wscript", "cscript", "mshta", "certutil", "bitsadmin",
                "reg delete", "reg add", "net user", "shutdown", "bcdedit", "vssadmin",
                "bash -c", "sh -c", "sudo ", "su "
            };
            for (const auto& bad : blocked_subcmds) {
                if (cmd_lower.find(bad) != std::string::npos) {
                    output = "🐚 [Terminal] Blocked: Destructive or sub-shell escalation commands are disallowed in sovereign sandbox.";
                    return true;
                }
            }

            auto start_time = std::chrono::high_resolution_clock::now();
            std::string full_cmd = cwd_prefix + cmd + " 2>&1";

            std::array<char, 512> buffer;
            std::string result;

            FILE* pipe = POPEN(full_cmd.c_str(), "r");
            if (!pipe) {
                output = "🐚 [Terminal] Failed to launch process for command: " + cmd;
                return true;
            }

            while (fgets(buffer.data(), (int)buffer.size(), pipe) != nullptr) {
                result += buffer.data();
                if (result.length() > 16000) {
                    result += "\n... [Output truncated at 16,000 characters for context efficiency]";
                    break;
                }
            }

            int exit_code = PCLOSE(pipe);
            auto end_time = std::chrono::high_resolution_clock::now();
            auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
            
            std::stringstream ss;
            ss << "🐚 Terminal Output (`" << cmd << "` | Exit Code " << exit_code << " | " << elapsed_ms << "ms):\n```\n";
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
