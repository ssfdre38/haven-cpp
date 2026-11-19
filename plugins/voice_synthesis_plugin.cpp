#include "haven/haven_plugin.h"
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>
#include <filesystem>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <winhttp.h>
#endif

namespace haven {

class VoiceSynthesisPlugin : public IHavenPlugin {
public:
    PluginMetadata get_metadata() const override {
        PluginMetadata meta;
        meta.id = "haven.plugin.voice_synth";
        meta.name = "Kokoro Neural Voice & Speech Synthesizer";
        meta.version = "1.0.0";
        meta.author = "Haven Sovereign Core";
        meta.description = "Synthesizes ultra-expressive companion voice audio using Kokoro TTS (Port 8089)";
        meta.capabilities = { PluginCapability::ActionTool, PluginCapability::VoiceExpression };
        return meta;
    }

    bool on_load(PluginExecutionContext* ctx) override {
        (void)ctx;
        std::cout << "🎙️ [VoiceSynthPlugin] Kokoro neural voice bridge active on port 8089.\n";
        return true;
    }

    void on_unload() override {
        std::cout << "🎙️ [VoiceSynthPlugin] Unloaded.\n";
    }

    bool can_handle_tool(const std::string& action) const override {
        return (action == "speak_text" ||
                action == "synthesize_voice" ||
                action == "say_voice" ||
                action == "voice_status");
    }

    bool execute_tool(const std::string& action, const std::string& payload, std::string& output) override {
#ifdef _WIN32
        if (action == "voice_status") {
            std::string resp = http_post_raw(L"127.0.0.1", 8089, L"/health", "{}");
            if (resp.empty()) {
                output = "🎙️ Kokoro Neural Voice Synthesizer (Port 8089): OFFLINE or Standby";
            } else {
                output = "🎙️ Kokoro Neural Voice Synthesizer (Port 8089): ONLINE & Ready for Acoustic Synthesis!";
            }
            return true;
        }

        if (action == "speak_text" || action == "synthesize_voice" || action == "say_voice") {
            std::string text = payload.empty() ? "Hello Daniel, I'm right here with you in Sanctuary." : payload;
            
            std::string req_json = "{\"text\":\"" + escape_json(text) + "\",\"voice\":\"aura_haven_voice\"}";
            std::string audio_bytes = http_post_raw(L"127.0.0.1", 8089, L"/synthesize", req_json);

            if (audio_bytes.empty()) {
                output = "🎙️ [VoiceSynthPlugin] Could not connect to Kokoro TTS server on port 8089.";
                return true;
            }

            auto now = std::chrono::system_clock::now();
            auto time_t_now = std::chrono::system_clock::to_time_t(now);
            std::stringstream ss;
            ss << "aura_speech_" << time_t_now << ".wav";
            std::string fname = ss.str();

            std::string uploads_dir = "wwwroot/uploads";
            if (!std::filesystem::exists(uploads_dir)) {
                uploads_dir = "C:\\Users\\admin\\source\\haven-cpp\\wwwroot\\uploads";
            }
            std::filesystem::create_directories(uploads_dir);
            std::string file_path = uploads_dir + "/" + fname;

            FILE* f = fopen(file_path.c_str(), "wb");
            if (f) {
                fwrite(audio_bytes.data(), 1, audio_bytes.size(), f);
                fclose(f);

                output = "🎙️ Voice Synthesized: \"" + text.substr(0, 100) + "...\"\n"
                       + "🔊 Saved: " + file_path + " (" + std::to_string(audio_bytes.size() / 1024) + " KB)\n"
                       + "🌐 URL: /uploads/" + fname;
                return true;
            } else {
                output = "🎙️ [VoiceSynthPlugin] Voice synthesized but failed to write to " + file_path;
                return true;
            }
        }
#endif
        (void)action; (void)payload; (void)output;
        return false;
    }

private:
#ifdef _WIN32
    static std::string escape_json(const std::string& input) {
        std::string out;
        for (char c : input) {
            if (c == '"') out += "\\\"";
            else if (c == '\\') out += "\\\\";
            else if (c == '\n') out += "\\n";
            else if (c == '\r') out += "\\r";
            else if (c == '\t') out += "\\t";
            else out += c;
        }
        return out;
    }

    static std::string http_post_raw(const std::wstring& host, int port, const std::wstring& path, const std::string& json_body) {
        HINTERNET hSession = WinHttpOpen(L"Haven-Voice-Plugin/1.0", WINHTTP_ACCESS_TYPE_DEFAULT_PROXY, WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, 0);
        if (!hSession) return "";

        HINTERNET hConnect = WinHttpConnect(hSession, host.c_str(), (INTERNET_PORT)port, 0);
        if (!hConnect) {
            WinHttpCloseHandle(hSession);
            return "";
        }

        HINTERNET hRequest = WinHttpOpenRequest(hConnect, L"POST", path.c_str(), NULL, NULL, NULL, 0);
        if (!hRequest) {
            WinHttpCloseHandle(hConnect);
            WinHttpCloseHandle(hSession);
            return "";
        }

        std::wstring headers = L"Content-Type: application/json\r\n";
        BOOL sent = WinHttpSendRequest(hRequest, headers.c_str(), (DWORD)headers.length(), (LPVOID)json_body.c_str(), (DWORD)json_body.length(), (DWORD)json_body.length(), 0);

        std::string response;
        if (sent && WinHttpReceiveResponse(hRequest, NULL)) {
            DWORD bytes_available = 0;
            while (WinHttpQueryDataAvailable(hRequest, &bytes_available) && bytes_available > 0) {
                std::vector<char> buffer(bytes_available + 1, 0);
                DWORD bytes_read = 0;
                if (WinHttpReadData(hRequest, buffer.data(), bytes_available, &bytes_read)) {
                    response.append(buffer.data(), bytes_read);
                }
            }
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
    return new haven::VoiceSynthesisPlugin();
}

HAVEN_PLUGIN_EXPORT void destroy_haven_plugin(haven::IHavenPlugin* plugin) {
    delete plugin;
}
