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

        // SSRF defense: block loopback, link-local, metadata, and private RFC1918 hosts
        std::string hlow = host;
        for (char& c : hlow) c = (char)tolower((unsigned char)c);
        if (hlow == "localhost" || hlow == "::1" || hlow == "[::1]" || hlow == "0.0.0.0" ||
            hlow.starts_with("127.") || hlow.starts_with("10.") || hlow.starts_with("192.168.") ||
            hlow.starts_with("169.254.") || hlow.starts_with("0.") ||
            hlow.ends_with(".local") || hlow.ends_with(".internal") ||
            hlow.find_first_of("\r\n \t@") != std::string::npos) {
            return "";
        }
        if (hlow.starts_with("172.")) {
            size_t dot2 = hlow.find('.', 4);
            if (dot2 != std::string::npos) {
                try {
                    int octet2 = std::stoi(hlow.substr(4, dot2 - 4));
                    if (octet2 >= 16 && octet2 <= 31) return "";
                } catch (...) {}
            }
        }
        if (path.find_first_of("\r\n") != std::string::npos) return "";

        std::wstring whost(host.begin(), host.end());
        std::wstring wpath(path.begin(), path.end());

        HINTERNET hSession = WinHttpOpen(L"Mozilla/5.0 (Windows NT 10.0; Win64; x64) HavenSovereignBrowser/2.0",
                                         WINHTTP_ACCESS_TYPE_DEFAULT_PROXY, WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, 0);
        if (!hSession) return "";

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

        // Strict TLS certificate validation enforced by default (CWE-295 compliance)

        std::string response;
        if (WinHttpSendRequest(hRequest, WINHTTP_NO_ADDITIONAL_HEADERS, 0, WINHTTP_NO_REQUEST_DATA, 0, 0, 0) &&
            WinHttpReceiveResponse(hRequest, NULL)) {
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
