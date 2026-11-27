#include "haven/haven_plugin.h"
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <array>
#include <memory>
#include <chrono>
#include <filesystem>
#include <cctype>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <poll.h>
#include <sys/wait.h>
#include <sys/resource.h>
#include <signal.h>
#endif

namespace haven {

class TerminalSandboxPlugin : public IHavenPlugin {
public:
    PluginMetadata get_metadata() const override {
        PluginMetadata meta;
        meta.id = "haven.plugin.terminal";
        meta.name = "Sovereign Terminal & Shell Sandbox";
        meta.version = "2.0.0";
        meta.author = "Haven Sovereign Core";
        meta.description = "Executes local shell commands inside an OS-enforced Job Object sandbox with hard memory, process-tree, UI, and timeout containment";
        meta.capabilities = { PluginCapability::ActionTool };
        return meta;
    }

    bool on_load(PluginExecutionContext* ctx) override {
        (void)ctx;
        std::cout << "🐚 [TerminalSandboxPlugin] OS-enforced Job Object shell sandbox online.\n";
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
            output = "🐚 Sovereign Shell Sandbox v2.0: ONLINE\nPlatform: Windows Native Job Object Sandbox (Kill-on-Close, 4-Proc Cap, 512MB RAM Cap, UI Isolation, 15s Timeout)";
#elif defined(__APPLE__)
            output = "🐚 Sovereign Shell Sandbox v2.0: ONLINE\nPlatform: macOS / Darwin Restricted Subprocess";
#else
            output = "🐚 Sovereign Shell Sandbox v2.0: ONLINE\nPlatform: Linux / POSIX Restricted Subprocess";
#endif
            return true;
        }

        if (action == "run_command" || action == "exec_shell" || action == "cmd_run") {
            if (payload.empty()) {
                output = "🐚 [Terminal] Please provide a command line to execute.";
                return true;
            }

            std::string cmd = payload;
            std::string cwd_path = "";

            // Parse optional [cwd: <path>] header — passed directly via OS lpCurrentDirectory (never interpolated into shell)
            if (cmd.starts_with("[cwd:")) {
                size_t close_bracket = cmd.find(']');
                if (close_bracket != std::string::npos) {
                    cwd_path = cmd.substr(5, close_bracket - 5);
                    while (!cwd_path.empty() && (cwd_path.front() == ' ' || cwd_path.front() == '"')) cwd_path.erase(0, 1);
                    while (!cwd_path.empty() && (cwd_path.back() == ' ' || cwd_path.back() == '"')) cwd_path.pop_back();
                    cmd = cmd.substr(close_bracket + 1);
                    while (!cmd.empty() && (cmd.front() == ' ' || cmd.front() == '\n' || cmd.front() == '\r')) cmd.erase(0, 1);
                    if (!cwd_path.empty()) {
                        const std::string bad_cwd_tokens[] = {"\"", "'", "|", "&", ";", "`", "$", "<", ">", "\n", "\r", "%", "^", "!"};
                        for (const auto& t : bad_cwd_tokens) {
                            if (cwd_path.find(t) != std::string::npos || cwd_path.find('\0') != std::string::npos) {
                                output = "🐚 [Terminal] Security Block: Invalid characters in working directory path.";
                                return true;
                            }
                        }
                        std::error_code ec;
                        if (!std::filesystem::exists(cwd_path, ec) || !std::filesystem::is_directory(cwd_path, ec)) {
                            output = "🐚 [Terminal] Blocked: Working directory does not exist: " + cwd_path;
                            return true;
                        }
                    }
                }
            }

            if (cmd.empty()) {
                output = "🐚 [Terminal] Please provide a command line to execute.";
                return true;
            }
            if (cmd.size() > 2048) {
                output = "🐚 [Terminal] Security Block: Command exceeds 2048-character limit.";
                return true;
            }

            // Destructive & injection safety guardrail (defense-in-depth above OS sandbox)
            const std::string forbidden_tokens[] = {"|", "&", ";", "`", "$(", "<", ">", "\n", "\r", "%", "^", "!"};
            for (const auto& token : forbidden_tokens) {
                if (cmd.find(token) != std::string::npos || cmd.find('\0') != std::string::npos) {
                    output = "🐚 [Terminal] Security Block: Shell chaining, pipe, redirection, and expansion characters are disallowed in sovereign sandbox.";
                    return true;
                }
            }

            // Normalize out quotes so quote-splitting obfuscation (e.g. c"m"d / p'o'wershell) cannot bypass checks
            std::string cmd_norm;
            cmd_norm.reserve(cmd.size());
            for (char c : cmd) {
                if (c == '"' || c == '\'') continue;
                cmd_norm += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            }
            const std::string blocked_subcmds[] = {
                "format ", "rmdir ", "del ", "erase ", "diskpart", "rm ", "unlink ",
                "mkfs", "dd ", ":(){ :|:& };:", "chmod ", "chown ",
                "cmd /c", "cmd.exe", "powershell", "pwsh", "wscript", "cscript",
                "mshta", "certutil", "bitsadmin", "rundll32", "regsvr32", "msiexec",
                "python", "node ", "node.exe", "perl ", "ruby ", "curl ", "wget ",
                "nc ", "ncat ", "ssh ", "scp ", "ftp ", "telnet ",
                "reg delete", "reg add", "net user", "net localgroup", "shutdown",
                "bcdedit", "vssadmin", "bash ", "sh -c", "zsh ", "sudo ", "su ",
                "runas ", "schtasks", "icacls", "takeown", "wmic ", "sc "
            };
            for (const auto& bad : blocked_subcmds) {
                if (cmd_norm.find(bad) != std::string::npos) {
                    output = "🐚 [Terminal] Security Block: Destructive, network, interpreter, or sub-shell escalation commands are disallowed in sovereign sandbox.";
                    return true;
                }
            }

            auto start_time = std::chrono::high_resolution_clock::now();
            std::string result;
            int exit_code = -1;
            bool timed_out = false;

#ifdef _WIN32
            if (!execute_win32_job_sandbox(cmd, cwd_path, result, exit_code, timed_out)) {
                output = "🐚 [Terminal] Failed to launch OS-sandboxed process for command: " + cmd;
                return true;
            }
#else
            if (!execute_posix_sandbox(cmd, cwd_path, result, exit_code, timed_out)) {
                output = "🐚 [Terminal] Failed to launch OS-sandboxed process for command: " + cmd;
                return true;
            }
#endif

            auto end_time = std::chrono::high_resolution_clock::now();
            auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

            std::stringstream ss;
            ss << "🐚 Terminal Output (`" << cmd << "` | Exit Code " << exit_code << " | " << elapsed_ms << "ms"
               << (timed_out ? " | KILLED: 15s TIMEOUT" : "") << "):\n```\n";
            ss << (result.empty() ? "[Command completed with no stdout output]" : result);
            ss << "\n```";

            output = ss.str();
            return true;
        }

        return false;
    }

private:
#ifdef _WIN32
    static bool execute_win32_job_sandbox(const std::string& cmd, const std::string& cwd_path,
                                          std::string& result, int& exit_code, bool& timed_out) {
        constexpr DWORD TIMEOUT_MS = 15000;
        constexpr size_t MAX_OUTPUT_BYTES = 16000;

        // 1. Create hardened Windows Job Object with strict resource, process-tree, and UI containment
        HANDLE hJob = CreateJobObjectA(NULL, NULL);
        if (!hJob) return false;

        JOBOBJECT_EXTENDED_LIMIT_INFORMATION ext_limits{};
        ext_limits.BasicLimitInformation.LimitFlags =
            JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE |
            JOB_OBJECT_LIMIT_ACTIVE_PROCESS |
            JOB_OBJECT_LIMIT_PROCESS_MEMORY;
        ext_limits.BasicLimitInformation.ActiveProcessLimit = 4;
        ext_limits.ProcessMemoryLimit = 512ULL * 1024 * 1024; // 512 MiB cap
        SetInformationJobObject(hJob, JobObjectExtendedLimitInformation, &ext_limits, sizeof(ext_limits));

        JOBOBJECT_BASIC_UI_RESTRICTIONS ui_limits{};
        ui_limits.UIRestrictionsClass =
            JOB_OBJECT_UILIMIT_DESKTOP |
            JOB_OBJECT_UILIMIT_DISPLAYSETTINGS |
            JOB_OBJECT_UILIMIT_EXITWINDOWS |
            JOB_OBJECT_UILIMIT_GLOBALATOMS |
            JOB_OBJECT_UILIMIT_HANDLES |
            JOB_OBJECT_UILIMIT_READCLIPBOARD |
            JOB_OBJECT_UILIMIT_SYSTEMPARAMETERS |
            JOB_OBJECT_UILIMIT_WRITECLIPBOARD;
        SetInformationJobObject(hJob, JobObjectBasicUIRestrictions, &ui_limits, sizeof(ui_limits));

        // 2. Create stdout/stderr anonymous pipe (non-inheritable read end)
        SECURITY_ATTRIBUTES sa{};
        sa.nLength = sizeof(sa);
        sa.bInheritHandle = TRUE;
        HANDLE hReadPipe = NULL, hWritePipe = NULL;
        if (!CreatePipe(&hReadPipe, &hWritePipe, &sa, 0)) {
            CloseHandle(hJob);
            return false;
        }
        SetHandleInformation(hReadPipe, HANDLE_FLAG_INHERIT, 0);

        // Provide NUL stdin so interactive prompts never hang
        HANDLE hNullIn = CreateFileA("NUL", GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE,
                                     &sa, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);

        STARTUPINFOA si{};
        si.cb = sizeof(si);
        si.dwFlags = STARTF_USESTDHANDLES;
        si.hStdInput = hNullIn ? hNullIn : INVALID_HANDLE_VALUE;
        si.hStdOutput = hWritePipe;
        si.hStdError = hWritePipe;

        PROCESS_INFORMATION pi{};
        // /D disables AutoRun registry command execution; /Q suppresses echo
        std::string full_cmdline = "C:\\Windows\\System32\\cmd.exe /D /Q /C " + cmd;
        std::vector<char> cmd_buf(full_cmdline.begin(), full_cmdline.end());
        cmd_buf.push_back('\0');

        const char* workdir = cwd_path.empty() ? nullptr : cwd_path.c_str();

        // 3. Launch suspended so process is bound to Job Object BEFORE any user instruction executes
        BOOL created = CreateProcessA(
            "C:\\Windows\\System32\\cmd.exe",
            cmd_buf.data(),
            NULL,
            NULL,
            TRUE,
            CREATE_NO_WINDOW | CREATE_SUSPENDED,
            NULL,
            workdir,
            &si,
            &pi
        );

        CloseHandle(hWritePipe);
        if (hNullIn && hNullIn != INVALID_HANDLE_VALUE) CloseHandle(hNullIn);

        if (!created) {
            CloseHandle(hReadPipe);
            CloseHandle(hJob);
            return false;
        }

        if (!AssignProcessToJobObject(hJob, pi.hProcess)) {
            TerminateProcess(pi.hProcess, 1);
            CloseHandle(pi.hThread);
            CloseHandle(pi.hProcess);
            CloseHandle(hReadPipe);
            CloseHandle(hJob);
            return false;
        }

        ResumeThread(pi.hThread);

        // 4. Non-blocking pipe drain with strict wall-clock deadline
        auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(TIMEOUT_MS);
        char buf[1024];
        bool truncated = false;

        while (true) {
            DWORD avail = 0;
            while (PeekNamedPipe(hReadPipe, NULL, 0, NULL, &avail, NULL) && avail > 0) {
                DWORD to_read = (avail < sizeof(buf)) ? avail : static_cast<DWORD>(sizeof(buf));
                DWORD got = 0;
                if (!ReadFile(hReadPipe, buf, to_read, &got, NULL) || got == 0) break;
                if (result.size() < MAX_OUTPUT_BYTES) {
                    size_t take = std::min<size_t>(got, MAX_OUTPUT_BYTES - result.size());
                    result.append(buf, take);
                    if (result.size() >= MAX_OUTPUT_BYTES && !truncated) {
                        result += "\n... [Output truncated at 16,000 characters]";
                        truncated = true;
                    }
                }
            }

            DWORD wait_rc = WaitForSingleObject(pi.hProcess, 25);
            if (wait_rc == WAIT_OBJECT_0) {
                // Final drain after exit
                while (PeekNamedPipe(hReadPipe, NULL, 0, NULL, &avail, NULL) && avail > 0) {
                    DWORD to_read = (avail < sizeof(buf)) ? avail : static_cast<DWORD>(sizeof(buf));
                    DWORD got = 0;
                    if (!ReadFile(hReadPipe, buf, to_read, &got, NULL) || got == 0) break;
                    if (result.size() < MAX_OUTPUT_BYTES) {
                        size_t take = std::min<size_t>(got, MAX_OUTPUT_BYTES - result.size());
                        result.append(buf, take);
                    }
                }
                break;
            }

            if (std::chrono::steady_clock::now() >= deadline) {
                timed_out = true;
                TerminateJobObject(hJob, 124);
                WaitForSingleObject(pi.hProcess, 500);
                break;
            }
        }

        DWORD dwExit = 0;
        if (timed_out) {
            exit_code = 124;
        } else if (GetExitCodeProcess(pi.hProcess, &dwExit)) {
            exit_code = static_cast<int>(dwExit);
        } else {
            exit_code = -1;
        }

        // Closing hJob immediately reaps any lingering descendant processes (KILL_ON_JOB_CLOSE)
        CloseHandle(hReadPipe);
        CloseHandle(pi.hThread);
        CloseHandle(pi.hProcess);
        CloseHandle(hJob);
        return true;
    }
#else
    static bool execute_posix_sandbox(const std::string& cmd, const std::string& cwd_path,
                                      std::string& result, int& exit_code, bool& timed_out) {
        constexpr int TIMEOUT_MS = 15000;
        constexpr size_t MAX_OUTPUT_BYTES = 16000;

        int pipefd[2];
        if (pipe(pipefd) < 0) return false;

        pid_t pid = fork();
        if (pid < 0) {
            close(pipefd[0]);
            close(pipefd[1]);
            return false;
        }
        if (pid == 0) {
            setpgid(0, 0);
            int devnull = open("/dev/null", O_RDONLY);
            if (devnull >= 0) {
                dup2(devnull, STDIN_FILENO);
                close(devnull);
            }
            dup2(pipefd[1], STDOUT_FILENO);
            dup2(pipefd[1], STDERR_FILENO);
            close(pipefd[0]);
            close(pipefd[1]);

            struct rlimit rl_cpu{15, 15};
            setrlimit(RLIMIT_CPU, &rl_cpu);
#ifdef RLIMIT_AS
            struct rlimit rl_as{512ULL * 1024 * 1024, 512ULL * 1024 * 1024};
            setrlimit(RLIMIT_AS, &rl_as);
#endif
#ifdef RLIMIT_NPROC
            struct rlimit rl_proc{32, 32};
            setrlimit(RLIMIT_NPROC, &rl_proc);
#endif
            if (!cwd_path.empty() && chdir(cwd_path.c_str()) != 0) {
                _exit(126);
            }
            const char* argv[] = {"/bin/sh", "-c", cmd.c_str(), nullptr};
            execv("/bin/sh", const_cast<char* const*>(argv));
            _exit(127);
        }

        close(pipefd[1]);
        fcntl(pipefd[0], F_SETFL, O_NONBLOCK);

        auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(TIMEOUT_MS);
        char buf[512];
        bool child_exited = false;
        int status = 0;

        while (true) {
            struct pollfd pfd{pipefd[0], POLLIN, 0};
            auto now = std::chrono::steady_clock::now();
            int rem_ms = static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now).count());
            if (rem_ms <= 0) {
                timed_out = true;
                kill(-pid, SIGKILL);
                waitpid(pid, &status, 0);
                break;
            }
            int pr = poll(&pfd, 1, std::min(rem_ms, 50));
            if (pr > 0) {
                ssize_t n = read(pipefd[0], buf, sizeof(buf));
                if (n > 0) {
                    if (result.size() < MAX_OUTPUT_BYTES) {
                        size_t take = std::min<size_t>(static_cast<size_t>(n), MAX_OUTPUT_BYTES - result.size());
                        result.append(buf, take);
                    }
                } else if (n == 0) {
                    break;
                }
            }
            pid_t w = waitpid(pid, &status, WNOHANG);
            if (w == pid) {
                child_exited = true;
                // Final non-blocking drain
                while (true) {
                    ssize_t n = read(pipefd[0], buf, sizeof(buf));
                    if (n <= 0) break;
                    if (result.size() < MAX_OUTPUT_BYTES) {
                        size_t take = std::min<size_t>(static_cast<size_t>(n), MAX_OUTPUT_BYTES - result.size());
                        result.append(buf, take);
                    }
                }
                break;
            }
        }
        close(pipefd[0]);
        if (!child_exited && !timed_out) {
            waitpid(pid, &status, 0);
        }
        if (timed_out) {
            exit_code = 124;
        } else if (WIFEXITED(status)) {
            exit_code = WEXITSTATUS(status);
        } else if (WIFSIGNALED(status)) {
            exit_code = 128 + WTERMSIG(status);
        } else {
            exit_code = -1;
        }
        return true;
    }
#endif
};

} // namespace haven

HAVEN_PLUGIN_EXPORT haven::IHavenPlugin* create_haven_plugin() {
    return new haven::TerminalSandboxPlugin();
}

HAVEN_PLUGIN_EXPORT void destroy_haven_plugin(haven::IHavenPlugin* plugin) {
    delete plugin;
}
