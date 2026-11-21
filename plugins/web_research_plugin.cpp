#include "haven/haven_plugin.h"
#include <iostream>
#include <sstream>
#include <string>
#include <algorithm>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <winhttp.h>
#endif

namespace haven {

class WebResearchPlugin : public IHavenPlugin {
public:
    PluginMetadata get_metadata() const override {
        PluginMetadata meta;
        meta.id = "haven.plugin.web_research";
        meta.name = "Sovereign Web Research & Wikipedia Intelligence";
        meta.version = "1.0.0";
        meta.author = "Haven Sovereign Core";
        meta.description = "Real-time web research, Wikipedia knowledge retrieval, and live weather telemetry";
        meta.capabilities = { PluginCapability::ActionTool, PluginCapability::CognitiveMemory };
        return meta;
    }

    bool on_load(PluginExecutionContext* ctx) override {
        (void)ctx;
        std::cout << "🌐 [WebResearchPlugin] Sovereign web intelligence & Wikipedia pipeline active.\n";
        return true;
    }

    void on_unload() override {
        std::cout << "🌐 [WebResearchPlugin] Unloaded.\n";
    }

    bool can_handle_tool(const std::string& action) const override {
        return (action == "web_search" ||
                action == "wiki_summary" ||
                action == "get_weather");
    }

    static std::string url_encode(const std::string& s, char space_char) {
        std::string out;
        out.reserve(s.size() * 3);
        static const char hex[] = "0123456789ABCDEF";
        for (unsigned char c : s) {
            if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '-' || c == '_' || c == '.' || c == '~') {
                out += static_cast<char>(c);
            } else if (c == ' ') {
                out += space_char;
            } else {
                out += '%';
                out += hex[(c >> 4) & 0xF];
                out += hex[c & 0xF];
            }
        }
        return out;
    }

    bool execute_tool(const std::string& action, const std::string& payload, std::string& output) override {
#ifdef _WIN32
        if (action == "wiki_summary" && !payload.empty()) {
            std::string topic = url_encode(payload, '_');
            std::string path = "/api/rest_v1/page/summary/" + topic;
            std::string json_resp = http_get(L"en.wikipedia.org", std::wstring(path.begin(), path.end()));
            
            // Extract extract field from JSON
            size_t extract_pos = json_resp.find("\"extract\":");
            if (extract_pos != std::string::npos) {
                size_t start = json_resp.find('"', extract_pos + 10);
                size_t end = json_resp.find('"', start + 1);
                if (start != std::string::npos && end != std::string::npos) {
                    output = "🌐 Wikipedia Fact Summary [" + payload + "]:\n" + json_resp.substr(start + 1, end - start - 1);
                    return true;
                }
            }
            output = "🌐 Wikipedia Summary for '" + payload + "':\n" + (json_resp.empty() ? "[No summary found]" : json_resp.substr(0, 500));
            return true;
        }
        else if (action == "get_weather") {
            std::string city = payload.empty() ? "" : url_encode(payload, '+');
            std::string path = "/" + city + "?format=3";
            std::string resp = http_get(L"wttr.in", std::wstring(path.begin(), path.end()));
            output = "🌤️ Live Weather: " + (resp.empty() ? "Unable to retrieve weather" : resp);
            return true;
        }
        else if (action == "web_search" && !payload.empty()) {
            std::string query = url_encode(payload, '+');
            std::string path = "/html/?q=" + query;
            std::string resp = http_get(L"html.duckduckgo.com", std::wstring(path.begin(), path.end()));
            output = "🔍 DuckDuckGo Research Query for '" + payload + "': [Response length " + std::to_string(resp.size()) + " bytes]";
            return true;
        }
#endif
        output = "⚠️ Web research action failed or unsupported: " + action;
        return false;
    }

private:
#ifdef _WIN32
    std::string http_get(const std::wstring& host, const std::wstring& path) {
        std::string result;
        HINTERNET hSession = WinHttpOpen(L"HavenCompanion/1.0", WINHTTP_ACCESS_TYPE_DEFAULT_PROXY, WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, 0);
        if (!hSession) return result;

        HINTERNET hConnect = WinHttpConnect(hSession, host.c_str(), INTERNET_DEFAULT_HTTPS_PORT, 0);
        if (!hConnect) {
            WinHttpCloseHandle(hSession);
            return result;
        }

        HINTERNET hRequest = WinHttpOpenRequest(hConnect, L"GET", path.c_str(), NULL, WINHTTP_NO_REFERER, WINHTTP_DEFAULT_ACCEPT_TYPES, WINHTTP_FLAG_SECURE);
        if (!hRequest) {
            WinHttpCloseHandle(hConnect);
            WinHttpCloseHandle(hSession);
            return result;
        }

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
                    result.append(buffer.data(), dwDownloaded);
                }
            } while (dwSize > 0 && result.size() < 512 * 1024);
        }

        WinHttpCloseHandle(hRequest);
        WinHttpCloseHandle(hConnect);
        WinHttpCloseHandle(hSession);
        return result;
    }
#endif
};

} // namespace haven

// Native Dynamic Library Export Entrypoints
extern "C" {
#ifdef _WIN32
    HAVEN_PLUGIN_EXPORT haven::IHavenPlugin* create_haven_plugin() {
        return new haven::WebResearchPlugin();
    }
    HAVEN_PLUGIN_EXPORT void destroy_haven_plugin(haven::IHavenPlugin* plugin) {
        delete plugin;
    }
#else
    haven::IHavenPlugin* create_haven_plugin() {
        return new haven::WebResearchPlugin();
    }
    void destroy_haven_plugin(haven::IHavenPlugin* plugin) {
        delete plugin;
    }
#endif
}
