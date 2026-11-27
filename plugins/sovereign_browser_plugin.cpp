#include "haven/haven_plugin.h"
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <cstring>
#include <cctype>
#include <algorithm>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#include <winhttp.h>
#include <shellapi.h>
#endif

namespace haven {

class SovereignBrowserPlugin : public IHavenPlugin {
public:
    PluginMetadata get_metadata() const override {
        PluginMetadata meta;
        meta.id = "haven.plugin.browser";
        meta.name = "Sovereign Web Browser & Page Navigator";
        meta.version = "1.0.0";
        meta.author = "Haven Sovereign Core";
        meta.description = "Real-time web browsing, HTML-to-markdown reader, HTTP/HTTPS DOM extractor, and native desktop browser launcher";
        meta.capabilities = { PluginCapability::ActionTool, PluginCapability::CognitiveMemory };
        return meta;
    }

    bool on_load(PluginExecutionContext* ctx) override {
        (void)ctx;
        std::cout << "🌐 [SovereignBrowserPlugin] Real browser engine active with WinHTTP & DOM reader.\n";
        return true;
    }

    void on_unload() override {
        std::cout << "🌐 [SovereignBrowserPlugin] Unloaded.\n";
    }

    bool can_handle_tool(const std::string& action) const override {
        return (action == "browse_url" ||
                action == "read_page" ||
                action == "open_browser" ||
                action == "open_in_browser" ||
                action == "browser_status");
    }

    bool execute_tool(const std::string& action, const std::string& payload, std::string& output) override {
#ifdef _WIN32
        if (action == "browser_status") {
            output = "🌐 Sovereign Browser Engine v1.0.0: ONLINE\nProtocol: WinHTTP/2.0 Native SSL\nUser-Agent: HavenSovereignBrowser/2.0";
            return true;
        }

        if (action == "open_browser" || action == "open_in_browser") {
            std::string url = payload;
            if (!url.starts_with("http://") && !url.starts_with("https://")) {
                url = "https://" + url;
            }
            bool safe_url = (url.starts_with("http://") || url.starts_with("https://")) && url.size() <= 2048;
            for (unsigned char c : url) {
                if (c <= 0x20 || c >= 0x7F || c == '"' || c == '\'' || c == '<' || c == '>' || c == '|' || c == '^' || c == '`') {
                    safe_url = false;
                    break;
                }
            }
            if (!safe_url) {
                output = "🌐 [Browser] Blocked: Malformed or unsafe URL rejected.";
                return true;
            }
            HINSTANCE res = ShellExecuteA(NULL, "open", url.c_str(), NULL, NULL, SW_SHOWNORMAL);
            if ((INT_PTR)res > 32) {
                output = "🌐 Launched native desktop browser navigating to: " + url;
            } else {
                output = "🌐 Failed to launch desktop browser for: " + url;
            }
            return true;
        }

        if (action == "browse_url" || action == "read_page") {
            if (payload.empty()) {
                output = "🌐 [Browser] Please provide a URL to browse.";
                return true;
            }

            std::string raw_html = fetch_url_html(payload);
            if (raw_html.empty()) {
                output = "🌐 [Browser] Failed to fetch content from: " + payload + " (Connection timeout or SSL error)";
                return true;
            }

            std::string readable_text = html_to_clean_markdown(raw_html);
            if (readable_text.length() > 4000) {
                readable_text = readable_text.substr(0, 4000) + "\n\n... [Content truncated for context efficiency]";
            }

            output = "🌐 Web Page Content [" + payload + "]:\n\n" + readable_text;
            return true;
        }
#endif
        (void)action; (void)payload; (void)output;
        return false;
    }

private:
#ifdef _WIN32
    static std::string html_to_clean_markdown(const std::string& html) {
        // Linear O(N) HTML-to-text scrubber (avoids std::regex backtracking/stack-overflow DoS)
        std::string text;
        text.reserve(html.size());
        size_t i = 0;
        const size_t n = html.size();

        auto starts_ci = [&](size_t pos, const char* tag) -> bool {
            size_t len = std::strlen(tag);
            if (pos + len > n) return false;
            for (size_t k = 0; k < len; ++k) {
                if (std::tolower(static_cast<unsigned char>(html[pos + k])) != tag[k]) return false;
            }
            return true;
        };

        auto skip_block = [&](const char* close_tag) {
            size_t len = std::strlen(close_tag);
            while (i < n) {
                if (html[i] == '<' && starts_ci(i, close_tag)) {
                    i += len;
                    while (i < n && html[i] != '>') ++i;
                    if (i < n) ++i;
                    return;
                }
                ++i;
            }
        };

        while (i < n) {
            if (html[i] == '<') {
                if (starts_ci(i, "<script")) { skip_block("</script"); text += ' '; continue; }
                if (starts_ci(i, "<style"))  { skip_block("</style");  text += ' '; continue; }
                if (starts_ci(i, "<head"))   { skip_block("</head");   text += ' '; continue; }

                bool newline_tag = starts_ci(i, "<br") || starts_ci(i, "</p") || starts_ci(i, "</div") ||
                                   starts_ci(i, "</li") || starts_ci(i, "</tr") ||
                                   (i + 3 < n && html[i + 1] == '/' && std::tolower((unsigned char)html[i + 2]) == 'h' &&
                                    html[i + 3] >= '1' && html[i + 3] <= '6');
                while (i < n && html[i] != '>') ++i;
                if (i < n) ++i;
                text += newline_tag ? '\n' : ' ';
            } else {
                text += html[i++];
            }
        }

        // Decode common HTML entities
        auto replace_all = [](std::string& str, const std::string& from, const std::string& to) {
            size_t start_pos = 0;
            while ((start_pos = str.find(from, start_pos)) != std::string::npos) {
                str.replace(start_pos, from.length(), to);
                start_pos += to.length();
            }
        };
        replace_all(text, "&nbsp;", " ");
        replace_all(text, "&amp;", "&");
        replace_all(text, "&lt;", "<");
        replace_all(text, "&gt;", ">");
        replace_all(text, "&quot;", "\"");
        replace_all(text, "&#39;", "'");
        replace_all(text, "&apos;", "'");
        replace_all(text, "&mdash;", "—");
        replace_all(text, "&ndash;", "–");

        // Clean up excessive blank lines & whitespace
        std::string cleaned;
        cleaned.reserve(text.length());
        bool last_was_space = false;
        int consecutive_newlines = 0;

        for (char c : text) {
            if (c == '\r') continue;
            if (c == '\n') {
                if (consecutive_newlines < 2) {
                    cleaned += '\n';
                    consecutive_newlines++;
                }
                last_was_space = false;
            } else if (c == ' ' || c == '\t') {
                if (!last_was_space && consecutive_newlines == 0) {
                    cleaned += ' ';
                    last_was_space = true;
                }
            } else {
                cleaned += c;
                last_was_space = false;
                consecutive_newlines = 0;
            }
        }

        // Trim leading and trailing whitespace
        size_t first = cleaned.find_first_not_of(" \t\n");
        size_t last = cleaned.find_last_not_of(" \t\n");
        if (first == std::string::npos) return "[Empty page or non-text content]";
        return cleaned.substr(first, last - first + 1);
    }

    static std::string fetch_url_html(const std::string& url_str) {
        std::string url = url_str;
        bool is_https = true;
        if (url.starts_with("https://")) {
            url = url.substr(8);
            is_https = true;
        } else if (url.starts_with("http://")) {
            url = url.substr(7);
            is_https = false;
        }

        std::string host;
        std::string path = "/";
        size_t slash_pos = url.find('/');
        if (slash_pos != std::string::npos) {
            host = url.substr(0, slash_pos);
            path = url.substr(slash_pos);
        } else {
            host = url;
        }

        int port = is_https ? 443 : 80;
        size_t colon_pos = host.find(':');
        if (colon_pos != std::string::npos) {
            try {
                port = std::stoi(host.substr(colon_pos + 1));
            } catch (...) { return ""; }
            host = host.substr(0, colon_pos);
        }
        if (port <= 0 || port > 65535 || host.empty()) return "";

        // Strict DNS-resolved SSRF defense: block loopback, link-local, metadata, CGNAT/VPN, and RFC1918 IPs
        std::string hlow = host;
        for (char& c : hlow) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        if (hlow == "localhost" || hlow == "::1" || hlow == "[::1]" || hlow == "0.0.0.0" ||
            hlow.ends_with(".local") || hlow.ends_with(".internal") ||
            hlow.find_first_of("\r\n \t@") != std::string::npos) {
            return "";
        }
        if (path.find_first_of("\r\n") != std::string::npos) return "";

        auto is_blocked_ipv4 = [](uint32_t ip_host_order) -> bool {
            uint8_t a = static_cast<uint8_t>((ip_host_order >> 24) & 0xFF);
            uint8_t b = static_cast<uint8_t>((ip_host_order >> 16) & 0xFF);
            uint8_t c = static_cast<uint8_t>((ip_host_order >> 8) & 0xFF);
            if (a == 0) return true;                                  // 0.0.0.0/8
            if (a == 10) return true;                                 // 10.0.0.0/8 (RFC1918)
            if (a == 100 && b >= 64 && b <= 127) return true;         // 100.64.0.0/10 (CGNAT / Tailscale / NetBird)
            if (a == 127) return true;                                // 127.0.0.0/8 (Loopback)
            if (a == 169 && b == 254) return true;                    // 169.254.0.0/16 (Link-local / Cloud Metadata)
            if (a == 172 && b >= 16 && b <= 31) return true;          // 172.16.0.0/12 (RFC1918)
            if (a == 192 && b == 0 && c == 0) return true;            // 192.0.0.0/24
            if (a == 192 && b == 168) return true;                    // 192.168.0.0/16 (RFC1918)
            if (a == 198 && (b == 18 || b == 19)) return true;        // 198.18.0.0/15 (Benchmark)
            if (a >= 224) return true;                                // Multicast & Reserved (224.0.0.0/3)
            return false;
        };

        auto is_blocked_ipv6 = [&](const uint8_t ip6[16]) -> bool {
            bool all_zero = true;
            for (int i = 0; i < 16; ++i) if (ip6[i] != 0) { all_zero = false; break; }
            if (all_zero) return true; // ::/128
            bool is_loopback = true;
            for (int i = 0; i < 15; ++i) if (ip6[i] != 0) { is_loopback = false; break; }
            if (is_loopback && ip6[15] == 1) return true; // ::1/128
            if ((ip6[0] & 0xFE) == 0xFC) return true;     // fc00::/7 (Unique Local)
            if (ip6[0] == 0xFE && (ip6[1] & 0xC0) == 0x80) return true; // fe80::/10 (Link-local)
            if (ip6[0] == 0xFF) return true;              // ff00::/8 (Multicast)
            // IPv4-mapped ::ffff:a.b.c.d and IPv4-compatible ::a.b.c.d
            bool first_10_zero = true;
            for (int i = 0; i < 10; ++i) if (ip6[i] != 0) { first_10_zero = false; break; }
            if (first_10_zero && ((ip6[10] == 0xFF && ip6[11] == 0xFF) || (ip6[10] == 0x00 && ip6[11] == 0x00))) {
                uint32_t v4 = (static_cast<uint32_t>(ip6[12]) << 24) |
                              (static_cast<uint32_t>(ip6[13]) << 16) |
                              (static_cast<uint32_t>(ip6[14]) << 8) |
                              static_cast<uint32_t>(ip6[15]);
                return is_blocked_ipv4(v4);
            }
            // NAT64 well-known prefix 64:ff9b::/96 (check embedded IPv4)
            if (ip6[0] == 0x00 && ip6[1] == 0x64 && ip6[2] == 0xFF && ip6[3] == 0x9B) {
                uint32_t v4 = (static_cast<uint32_t>(ip6[12]) << 24) |
                              (static_cast<uint32_t>(ip6[13]) << 16) |
                              (static_cast<uint32_t>(ip6[14]) << 8) |
                              static_cast<uint32_t>(ip6[15]);
                if (is_blocked_ipv4(v4)) return true;
            }
            // 6to4 (2002::/16) and Teredo (2001:0000::/32) transition ranges & 100::/64 discard
            if (ip6[0] == 0x20 && ip6[1] == 0x02) return true;
            if (ip6[0] == 0x20 && ip6[1] == 0x01 && ip6[2] == 0x00 && ip6[3] == 0x00) return true;
            if (ip6[0] == 0x01 && ip6[1] == 0x00 && ip6[2] == 0x00 && ip6[3] == 0x00) return true;
            return false;
        };

        // Resolve DNS and inspect EVERY returned address before opening WinHTTP connection
        WSADATA wsa_tmp{};
        bool wsa_started = (WSAStartup(MAKEWORD(2, 2), &wsa_tmp) == 0);
        struct addrinfo hints{};
        hints.ai_family = AF_UNSPEC;
        hints.ai_socktype = SOCK_STREAM;
        struct addrinfo* res_list = nullptr;
        std::string resolve_host = host;
        if (resolve_host.size() >= 2 && resolve_host.front() == '[' && resolve_host.back() == ']') {
            resolve_host = resolve_host.substr(1, resolve_host.size() - 2);
        }
        if (getaddrinfo(resolve_host.c_str(), nullptr, &hints, &res_list) != 0 || !res_list) {
            if (wsa_started) WSACleanup();
            return "";
        }
        bool has_safe_addr = false;
        bool has_blocked_addr = false;
        std::string pinned_ip;
        for (struct addrinfo* ai = res_list; ai != nullptr; ai = ai->ai_next) {
            if (ai->ai_family == AF_INET) {
                auto* sin = reinterpret_cast<sockaddr_in*>(ai->ai_addr);
                uint32_t ip_h = ntohl(sin->sin_addr.s_addr);
                if (is_blocked_ipv4(ip_h)) { has_blocked_addr = true; break; }
                if (pinned_ip.empty()) {
                    char ipbuf[INET_ADDRSTRLEN] = {0};
                    if (inet_ntop(AF_INET, &sin->sin_addr, ipbuf, sizeof(ipbuf))) {
                        pinned_ip = ipbuf;
                    }
                }
                has_safe_addr = true;
            } else if (ai->ai_family == AF_INET6) {
                auto* sin6 = reinterpret_cast<sockaddr_in6*>(ai->ai_addr);
                if (is_blocked_ipv6(reinterpret_cast<const uint8_t*>(&sin6->sin6_addr))) {
                    has_blocked_addr = true; break;
                }
                if (pinned_ip.empty()) {
                    char ipbuf[INET6_ADDRSTRLEN] = {0};
                    if (inet_ntop(AF_INET6, &sin6->sin6_addr, ipbuf, sizeof(ipbuf))) {
                        pinned_ip = std::string("[") + ipbuf + "]";
                    }
                }
                has_safe_addr = true;
            }
        }
        freeaddrinfo(res_list);
        if (wsa_started) WSACleanup();
        if (has_blocked_addr || !has_safe_addr || pinned_ip.empty()) return "";

        // For plaintext HTTP, pin directly to the validated resolved IP with explicit Host header
        // (prevents DNS rebinding TOCTOU). For HTTPS, use hostname for strict SNI/cert verification
        // and verify the actual connected peer IP via WINHTTP_OPTION_CONNECTION_INFO before reading!
        std::string connect_target = is_https ? host : pinned_ip;
        std::wstring whost(connect_target.begin(), connect_target.end());
        std::wstring wpath(path.begin(), path.end());

        HINTERNET hSession = WinHttpOpen(L"Mozilla/5.0 (Windows NT 10.0; Win64; x64) HavenSovereignBrowser/2.0",
                                         WINHTTP_ACCESS_TYPE_NO_PROXY, WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, 0);
        if (!hSession) return "";

        WinHttpSetTimeouts(hSession, 5000, 5000, 10000, 10000);

        HINTERNET hConnect = WinHttpConnect(hSession, whost.c_str(), (INTERNET_PORT)port, 0);
        if (!hConnect) {
            WinHttpCloseHandle(hSession);
            return "";
        }

        DWORD flags = is_https ? WINHTTP_FLAG_SECURE : 0;
        HINTERNET hRequest = WinHttpOpenRequest(hConnect, L"GET", wpath.c_str(), NULL, NULL, NULL, flags);
        if (!hRequest) {
            WinHttpCloseHandle(hConnect);
            WinHttpCloseHandle(hSession);
            return "";
        }

        // Disable automatic redirects to prevent 302 -> http://127.0.0.1 SSRF redirect bypasses
        DWORD redirect_policy = WINHTTP_OPTION_REDIRECT_POLICY_NEVER;
        WinHttpSetOption(hRequest, WINHTTP_OPTION_REDIRECT_POLICY, &redirect_policy, sizeof(redirect_policy));

        std::wstring extra_hdrs;
        if (!is_https) {
            std::string host_val = (port == 80) ? host : (host + ":" + std::to_string(port));
            extra_hdrs = L"Host: " + std::wstring(host_val.begin(), host_val.end()) + L"\r\n";
        }

        // Strict TLS certificate validation enforced by default (CWE-295 compliance)

        std::string response;
        if (WinHttpSendRequest(hRequest,
                               extra_hdrs.empty() ? WINHTTP_NO_ADDITIONAL_HEADERS : extra_hdrs.c_str(),
                               static_cast<DWORD>(extra_hdrs.size()),
                               WINHTTP_NO_REQUEST_DATA, 0, 0, 0) &&
            WinHttpReceiveResponse(hRequest, NULL)) {
#ifdef WINHTTP_OPTION_CONNECTION_INFO
            WINHTTP_CONNECTION_INFO conn_info{};
            DWORD ci_len = sizeof(conn_info);
            if (WinHttpQueryOption(hRequest, WINHTTP_OPTION_CONNECTION_INFO, &conn_info, &ci_len)) {
                bool peer_blocked = false;
                if (conn_info.RemoteAddress.ss_family == AF_INET) {
                    const auto* s4 = reinterpret_cast<const sockaddr_in*>(&conn_info.RemoteAddress);
                    peer_blocked = is_blocked_ipv4(ntohl(s4->sin_addr.s_addr));
                } else if (conn_info.RemoteAddress.ss_family == AF_INET6) {
                    const auto* s6 = reinterpret_cast<const sockaddr_in6*>(&conn_info.RemoteAddress);
                    peer_blocked = is_blocked_ipv6(reinterpret_cast<const uint8_t*>(&s6->sin6_addr));
                }
                if (peer_blocked) {
                    WinHttpCloseHandle(hRequest);
                    WinHttpCloseHandle(hConnect);
                    WinHttpCloseHandle(hSession);
                    return "";
                }
            }
#endif
            DWORD dwSize = 0;
            DWORD dwDownloaded = 0;
            do {
                dwSize = 0;
                if (!WinHttpQueryDataAvailable(hRequest, &dwSize)) break;
                if (dwSize == 0) break;
                std::vector<char> buffer(dwSize + 1, 0);
                if (WinHttpReadData(hRequest, buffer.data(), dwSize, &dwDownloaded)) {
                    response.append(buffer.data(), dwDownloaded);
                }
            } while (dwSize > 0 && response.length() < 500000); // 500KB cap
        }

        WinHttpCloseHandle(hRequest);
        WinHttpCloseHandle(hConnect);
        WinHttpCloseHandle(hSession);
        return response;
    }
#endif
};

} // namespace haven

HAVEN_PLUGIN_EXPORT haven::IHavenPlugin* create_haven_plugin() {
    return new haven::SovereignBrowserPlugin();
}

HAVEN_PLUGIN_EXPORT void destroy_haven_plugin(haven::IHavenPlugin* plugin) {
    delete plugin;
}
