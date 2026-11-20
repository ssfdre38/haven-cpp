#include "haven/haven_plugin.h"
#include <iostream>
#include <sstream>
#include <chrono>
#include <ctime>
#include <thread>
#include <vector>
#include <algorithm>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <psapi.h>
#include <tlhelp32.h>
#else
#include <unistd.h>
#include <sys/types.h>
#include <dirent.h>
#include <fstream>
#include <sys/statvfs.h>
#if defined(__linux__) || defined(__ANDROID__)
#include <sys/sysinfo.h>
#include <mntent.h>
#include <sys/utsname.h>
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#include <mach/mach.h>
#include <mach/vm_statistics.h>
#include <mach/mach_types.h>
#include <mach/mach_init.h>
#include <mach/mach_host.h>
#include <libproc.h>
#include <sys/mount.h>
#include <stdlib.h>
#endif
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
        int mem_percent = 0;
        int battery_life = 100;
        bool on_ac = true;

#if defined(__linux__) || defined(__ANDROID__)
        struct sysinfo si;
        if (sysinfo(&si) == 0 && si.totalram > 0) {
            unsigned long long total_b = (unsigned long long)si.totalram * si.mem_unit;
            unsigned long long free_b = (unsigned long long)si.freeram * si.mem_unit;
            if (total_b > 0) {
                mem_percent = (int)(100.0 * (1.0 - (double)free_b / (double)total_b));
            }
        }
        // Inspect Linux power supply: /sys/class/power_supply/BAT0 or BAT1
        std::ifstream bat_cap("/sys/class/power_supply/BAT0/capacity");
        if (!bat_cap.is_open()) {
            bat_cap.open("/sys/class/power_supply/BAT1/capacity");
        }
        if (bat_cap.is_open()) {
            bat_cap >> battery_life;
            std::ifstream bat_status("/sys/class/power_supply/BAT0/status");
            if (!bat_status.is_open()) bat_status.open("/sys/class/power_supply/BAT1/status");
            if (bat_status.is_open()) {
                std::string st;
                bat_status >> st;
                on_ac = (st == "Charging" || st == "Full");
            }
        } else {
            // Check AC online
            std::ifstream ac_online("/sys/class/power_supply/AC/online");
            if (!ac_online.is_open()) ac_online.open("/sys/class/power_supply/ACAD/online");
            if (ac_online.is_open()) {
                int online = 1;
                ac_online >> online;
                on_ac = (online == 1);
            }
        }
#elif defined(__APPLE__)
        int64_t total_ram = 0;
        size_t len = sizeof(total_ram);
        if (sysctlbyname("hw.memsize", &total_ram, &len, nullptr, 0) == 0 && total_ram > 0) {
            mach_port_t host_port = mach_host_self();
            mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
            vm_statistics64_data_t vm_stat;
            vm_size_t page_size = 4096;
            host_page_size(host_port, &page_size);

            if (host_statistics64(host_port, HOST_VM_INFO64, (host_info64_t)&vm_stat, &count) == KERN_SUCCESS) {
                int64_t free_mem = ((int64_t)vm_stat.free_count + (int64_t)vm_stat.inactive_count) * (int64_t)page_size;
                if (total_ram > free_mem) {
                    mem_percent = (int)(100.0 * (1.0 - (double)free_mem / (double)total_ram));
                }
            }
        }
#endif
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

        if (action == "process_list" || action == "list_processes") {
#if defined(__linux__) || defined(__ANDROID__)
            DIR* proc_dir = opendir("/proc");
            if (!proc_dir) {
                output = "🖥️ [SystemSensory] Failed to open /proc directory.";
                return true;
            }

            struct ProcessInfo {
                pid_t pid;
                std::string name;
                int threads;
            };
            std::vector<ProcessInfo> procs;

            struct dirent* entry;
            while ((entry = readdir(proc_dir)) != nullptr) {
                if (entry->d_name[0] >= '0' && entry->d_name[0] <= '9') {
                    pid_t pid = (pid_t)std::atoi(entry->d_name);
                    if (pid <= 0) continue;

                    std::string comm_path = std::string("/proc/") + entry->d_name + "/comm";
                    std::ifstream comm_file(comm_path);
                    std::string comm_name = "[unknown]";
                    if (comm_file.is_open()) {
                        std::getline(comm_file, comm_name);
                    }

                    int threads = 1;
                    std::string status_path = std::string("/proc/") + entry->d_name + "/status";
                    std::ifstream status_file(status_path);
                    if (status_file.is_open()) {
                        std::string line;
                        while (std::getline(status_file, line)) {
                            if (line.starts_with("Threads:")) {
                                size_t pos = line.find_first_of("0123456789");
                                if (pos != std::string::npos) {
                                    threads = std::atoi(line.c_str() + pos);
                                }
                                break;
                            }
                        }
                    }

                    procs.push_back({pid, comm_name, threads});
                }
            }
            closedir(proc_dir);

            std::sort(procs.begin(), procs.end(), [](const ProcessInfo& a, const ProcessInfo& b) {
                return a.pid < b.pid;
            });

            std::ostringstream ss;
            ss << "🖥️ Active Linux / POSIX Processes (Top Active Snapshot):\n";
            ss << "   PID     | Threads | Process Name\n";
            ss << "   --------+---------+----------------------------\n";
            int count = 0;
            for (const auto& p : procs) {
                if (count++ < 35) {
                    char line[128];
                    snprintf(line, sizeof(line), "   %-7d | %-7d | %s\n", p.pid, p.threads, p.name.c_str());
                    ss << line;
                }
            }
            ss << "   [Total processes enumerated: " << procs.size() << "]\n";
            output = ss.str();
            return true;
#elif defined(__APPLE__)
            int bytes_needed = proc_listpids(PROC_ALL_PIDS, 0, nullptr, 0);
            if (bytes_needed <= 0) {
                output = "🖥️ [SystemSensory] Failed to enumerate macOS processes.";
                return true;
            }

            int count_pids = bytes_needed / (int)sizeof(pid_t);
            std::vector<pid_t> pids(count_pids);
            int bytes_read = proc_listpids(PROC_ALL_PIDS, 0, pids.data(), bytes_needed);
            int actual_pids = bytes_read / (int)sizeof(pid_t);

            struct ProcessInfo {
                pid_t pid;
                std::string name;
                int threads;
            };
            std::vector<ProcessInfo> procs;

            for (int i = 0; i < actual_pids; ++i) {
                pid_t pid = pids[i];
                if (pid <= 0) continue;

                char name_buf[256] = {0};
                int nlen = proc_name(pid, name_buf, sizeof(name_buf));
                std::string pname = (nlen > 0) ? name_buf : "[system]";

                struct proc_taskinfo ti;
                int ti_res = proc_pidinfo(pid, PROC_PIDTASKINFO, 0, &ti, sizeof(ti));
                int threads = (ti_res > 0) ? ti.pti_threadnum : 1;

                procs.push_back({pid, pname, threads});
            }

            std::sort(procs.begin(), procs.end(), [](const ProcessInfo& a, const ProcessInfo& b) {
                return a.pid < b.pid;
            });

            std::ostringstream ss;
            ss << "🖥️ Active macOS / Darwin Processes (Top Active Snapshot):\n";
            ss << "   PID     | Threads | Process Name\n";
            ss << "   --------+---------+----------------------------\n";
            int count = 0;
            for (const auto& p : procs) {
                if (count++ < 35) {
                    char line[128];
                    snprintf(line, sizeof(line), "   %-7d | %-7d | %s\n", p.pid, p.threads, p.name.c_str());
                    ss << line;
                }
            }
            ss << "   [Total processes enumerated: " << procs.size() << "]\n";
            output = ss.str();
            return true;
#else
            output = "🖥️ [SystemSensory] Process listing not supported on this POSIX variant.";
            return true;
#endif
        }

        if (action == "volume_info" || action == "disk_usage") {
            std::ostringstream ss;
            ss << "💾 Sovereign Disk & Volume Telemetry (POSIX / Linux):\n";
            ss << "   Mount Point          | File System | Total Space | Free Space | Device\n";
            ss << "   ---------------------+-------------+-------------+------------+------------------\n";

            bool found_any = false;
#if defined(__linux__) || defined(__ANDROID__)
            FILE* mounts = setmntent("/proc/mounts", "r");
            if (!mounts) mounts = setmntent("/etc/mtab", "r");
            if (mounts) {
                struct mntent* ent;
                while ((ent = getmntent(mounts)) != nullptr) {
                    std::string fs_type = ent->mnt_type ? ent->mnt_type : "";
                    std::string fs_name = ent->mnt_fsname ? ent->mnt_fsname : "";
                    std::string dir = ent->mnt_dir ? ent->mnt_dir : "";

                    if (fs_type == "proc" || fs_type == "sysfs" || fs_type == "devpts" || 
                        fs_type == "cgroup" || fs_type == "cgroup2" || fs_type == "pstore" || 
                        fs_type == "bpf" || fs_type == "tracefs" || fs_type == "autofs" ||
                        fs_type == "securityfs" || fs_type == "configfs" || fs_type == "debugfs" ||
                        fs_type == "fusectl" || fs_type == "hugetlbfs" || fs_type == "mqueue" ||
                        fs_type == "devtmpfs") {
                        continue;
                    }

                    struct statvfs sv;
                    if (statvfs(ent->mnt_dir, &sv) == 0 && sv.f_blocks > 0) {
                        double total_gb = (double)(sv.f_blocks * sv.f_frsize) / (1024.0 * 1024.0 * 1024.0);
                        double free_gb = (double)(sv.f_bavail * sv.f_frsize) / (1024.0 * 1024.0 * 1024.0);
                        char line[256];
                        snprintf(line, sizeof(line), "   %-20s | %-11s | %8.1f GB | %7.1f GB | %s\n",
                                 dir.c_str(), fs_type.c_str(), total_gb, free_gb, fs_name.c_str());
                        ss << line;
                        found_any = true;
                    }
                }
                endmntent(mounts);
            }
#elif defined(__APPLE__)
            struct statfs* mntbuf = nullptr;
            int n_mounts = getmntinfo(&mntbuf, MNT_WAIT);
            if (n_mounts > 0 && mntbuf) {
                for (int i = 0; i < n_mounts; ++i) {
                    std::string fstype = mntbuf[i].f_fstypename;
                    std::string dir = mntbuf[i].f_mntonname;
                    std::string dev = mntbuf[i].f_mntfromname;

                    if (fstype == "devfs" || fstype == "autofs") continue;

                    double total_gb = (double)(mntbuf[i].f_blocks * mntbuf[i].f_bsize) / (1024.0 * 1024.0 * 1024.0);
                    double free_gb = (double)(mntbuf[i].f_bavail * mntbuf[i].f_bsize) / (1024.0 * 1024.0 * 1024.0);
                    char line[256];
                    snprintf(line, sizeof(line), "   %-20s | %-11s | %8.1f GB | %7.1f GB | %s\n",
                             dir.c_str(), fstype.c_str(), total_gb, free_gb, dev.c_str());
                    ss << line;
                    found_any = true;
                }
            }
#endif
            if (!found_any) {
                struct statvfs sv;
                if (statvfs("/", &sv) == 0) {
                    double total_gb = (double)(sv.f_blocks * sv.f_frsize) / (1024.0 * 1024.0 * 1024.0);
                    double free_gb = (double)(sv.f_bavail * sv.f_frsize) / (1024.0 * 1024.0 * 1024.0);
                    char line[256];
                    snprintf(line, sizeof(line), "   %-20s | %-11s | %8.1f GB | %7.1f GB | %s\n",
                             "/", "posix", total_gb, free_gb, "/dev/root");
                    ss << line;
                }
            }
            output = ss.str();
            return true;
        }

        if (action == "hardware_vitals") {
            unsigned int threads = std::thread::hardware_concurrency();
            std::ostringstream ss;
#if defined(__APPLE__)
            ss << "⚡ Sovereign Hardware Vitals (macOS / Apple Silicon):\n";
            char brand[256] = {0};
            size_t b_len = sizeof(brand);
            if (sysctlbyname("machdep.cpu.brand_string", brand, &b_len, nullptr, 0) == 0 && brand[0] != '\0') {
                ss << "   • Processor: " << brand << "\n";
            } else {
                ss << "   • Architecture: Apple Silicon ARM64 / MLX Unified Memory\n";
            }

            struct timeval boottime;
            size_t bt_len = sizeof(boottime);
            if (sysctlbyname("kern.boottime", &boottime, &bt_len, nullptr, 0) == 0) {
                time_t bsec = boottime.tv_sec;
                time_t csec = std::time(nullptr);
                long uptime_sec = (long)(csec - bsec);
                long uptime_hrs = uptime_sec / 3600;
                long uptime_min = (uptime_sec % 3600) / 60;
                ss << "   • System Uptime: " << uptime_hrs << " hours, " << uptime_min << " minutes\n";
            }

            double loads[3] = {0.0, 0.0, 0.0};
            if (getloadavg(loads, 3) != -1) {
                char load_buf[128];
                snprintf(load_buf, sizeof(load_buf), "   • Load Average: %.2f, %.2f, %.2f (1m, 5m, 15m)\n", loads[0], loads[1], loads[2]);
                ss << load_buf;
            }

            int64_t total_ram = 0;
            size_t mlen = sizeof(total_ram);
            sysctlbyname("hw.memsize", &total_ram, &mlen, nullptr, 0);

            mach_port_t host_port = mach_host_self();
            mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
            vm_statistics64_data_t vm_stat;
            vm_size_t page_size = 4096;
            host_page_size(host_port, &page_size);

            int mem_load = 0;
            int64_t free_mem = 0;
            if (host_statistics64(host_port, HOST_VM_INFO64, (host_info64_t)&vm_stat, &count) == KERN_SUCCESS && total_ram > 0) {
                free_mem = ((int64_t)vm_stat.free_count + (int64_t)vm_stat.inactive_count) * (int64_t)page_size;
                if (total_ram > free_mem) {
                    mem_load = (int)(100.0 * (1.0 - (double)free_mem / (double)total_ram));
                }
            }

            ss << "   • Hardware Concurrency: " << threads << " logical threads\n"
               << "   • Unified Memory Load: " << mem_load << "%\n"
               << "   • Unified RAM: " << (total_ram / (1024*1024*1024)) << " GB Total ("
               << (free_mem / (1024*1024*1024)) << " GB Available)\n";
            output = ss.str();
            return true;
#elif defined(__linux__) || defined(__ANDROID__)
            ss << "⚡ Sovereign Hardware Vitals (Linux / POSIX):\n";
            struct utsname un;
            if (uname(&un) == 0) {
                ss << "   • OS Kernel: " << un.sysname << " " << un.release << " (" << un.machine << ")\n";
            } else {
                ss << "   • OS Platform: POSIX / Linux Bare-Metal\n";
            }

            struct sysinfo si;
            if (sysinfo(&si) == 0) {
                long uptime_sec = si.uptime;
                long uptime_hrs = uptime_sec / 3600;
                long uptime_min = (uptime_sec % 3600) / 60;

                double load1 = si.loads[0] / 65536.0;
                double load5 = si.loads[1] / 65536.0;
                double load15 = si.loads[2] / 65536.0;

                unsigned long long total_ram_mb = ((unsigned long long)si.totalram * si.mem_unit) / (1024 * 1024);
                unsigned long long free_ram_mb = ((unsigned long long)si.freeram * si.mem_unit) / (1024 * 1024);
                unsigned long long total_swap_mb = ((unsigned long long)si.totalswap * si.mem_unit) / (1024 * 1024);
                unsigned long long free_swap_mb = ((unsigned long long)si.freeswap * si.mem_unit) / (1024 * 1024);
                int mem_load = total_ram_mb > 0 ? (int)(100.0 * (1.0 - (double)free_ram_mb / (double)total_ram_mb)) : 0;

                char vitals_buf[256];
                snprintf(vitals_buf, sizeof(vitals_buf), "   • Load Average: %.2f, %.2f, %.2f (1m, 5m, 15m)\n", load1, load5, load15);

                ss << "   • Hardware Concurrency: " << threads << " logical threads\n"
                   << "   • System Uptime: " << uptime_hrs << " hours, " << uptime_min << " minutes\n"
                   << vitals_buf
                   << "   • Memory Load: " << mem_load << "%\n"
                   << "   • Physical RAM: " << (total_ram_mb / 1024) << " GB Total (" << (free_ram_mb / 1024) << " GB Free)\n"
                   << "   • Swap Space: " << (total_swap_mb / 1024) << " GB Total (" << (free_swap_mb / 1024) << " GB Free)\n"
                   << "   • Running Processes: " << si.procs << " active\n";
            } else {
                ss << "   • Hardware Concurrency: " << threads << " logical threads\n";
            }
#else
            ss << "   • Hardware Concurrency: " << threads << " logical threads\n"
               << "   • OS Architecture: POSIX Sovereign Environment\n";
#endif
            output = ss.str();
            return true;
        }
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
#else
#if defined(__linux__) || defined(__ANDROID__)
        struct sysinfo si;
        if (sysinfo(&si) == 0 && si.totalram > 0) {
            unsigned long long total_ram_mb = ((unsigned long long)si.totalram * si.mem_unit) / (1024 * 1024);
            unsigned long long free_ram_mb = ((unsigned long long)si.freeram * si.mem_unit) / (1024 * 1024);
            int mem_load = total_ram_mb > 0 ? (int)(100.0 * (1.0 - (double)free_ram_mb / (double)total_ram_mb)) : 0;
            ss << "   • RAM Usage: " << mem_load << "% (" 
               << (free_ram_mb / 1024) << " GB free of "
               << (total_ram_mb / 1024) << " GB total)\n";
        }
#elif defined(__APPLE__)
        int64_t total_ram = 0;
        size_t mlen = sizeof(total_ram);
        sysctlbyname("hw.memsize", &total_ram, &mlen, nullptr, 0);
        mach_port_t host_port = mach_host_self();
        mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
        vm_statistics64_data_t vm_stat;
        vm_size_t page_size = 4096;
        host_page_size(host_port, &page_size);
        if (host_statistics64(host_port, HOST_VM_INFO64, (host_info64_t)&vm_stat, &count) == KERN_SUCCESS && total_ram > 0) {
            int64_t free_mem = ((int64_t)vm_stat.free_count + (int64_t)vm_stat.inactive_count) * (int64_t)page_size;
            int mem_load = (int)(100.0 * (1.0 - (double)free_mem / (double)total_ram));
            ss << "   • Unified RAM Usage: " << mem_load << "% (" 
               << (free_mem / (1024*1024*1024)) << " GB available of "
               << (total_ram / (1024*1024*1024)) << " GB total)\n";
        }
#endif
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
