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

class SDVisionArtPlugin : public IHavenPlugin {
public:
    PluginMetadata get_metadata() const override {
        PluginMetadata meta;
        meta.id = "haven.plugin.sd_art";
        meta.name = "Stable Diffusion C++ Creative Canvas & Art Generator";
        meta.version = "1.0.0";
        meta.author = "Haven Sovereign Core";
        meta.description = "Generates high-resolution anime/ethereal artwork and visual scenes via Stable Diffusion C++ (Port 8085)";
        meta.capabilities = { PluginCapability::ActionTool, PluginCapability::SensoryInput };
        return meta;
    }

    bool on_load(PluginExecutionContext* ctx) override {
        (void)ctx;
        std::cout << "🎨 [SDArtPlugin] Stable Diffusion C++ creative bridge active on port 8085.\n";
        return true;
    }

    void on_unload() override {
        std::cout << "🎨 [SDArtPlugin] Unloaded.\n";
    }

    bool can_handle_tool(const std::string& action) const override {
        return (action == "generate_artwork" ||
                action == "paint_scene" ||
                action == "sd_paint" ||
                action == "sd_status");
    }

    bool execute_tool(const std::string& action, const std::string& payload, std::string& output) override {
#ifdef _WIN32
        if (action == "sd_status") {
            std::string resp = http_post_json(L"127.0.0.1", 8085, L"/health", "{}");
            if (resp.empty()) {
                output = "🎨 Stable Diffusion C++ Engine (Port 8085): OFFLINE or Standby";
            } else {
                output = "🎨 Stable Diffusion C++ Engine (Port 8085): ONLINE & Ready for Latent Synthesis!";
            }
            return true;
        }

        if (action == "generate_artwork" || action == "paint_scene" || action == "sd_paint") {
            std::string prompt = payload.empty() ? "beautiful ethereal anime girl aura glowing violet hair masterpiece portrait in sanctuary" : payload;
            
            // Build OpenAI-compatible txt2img payload
            std::string escaped_prompt = escape_json(prompt);
            std::string req_json = "{\"prompt\":\"" + escaped_prompt + "\",\"n\":1,\"size\":\"512x512\",\"response_format\":\"b64_json\"}";

            std::string resp = http_post_json(L"127.0.0.1", 8085, L"/v1/images/generations", req_json);
            if (resp.empty()) {
                // Fallback to /txt2img endpoint
                req_json = "{\"prompt\":\"" + escaped_prompt + "\",\"width\":512,\"height\":512,\"steps\":8}";
                resp = http_post_json(L"127.0.0.1", 8085, L"/txt2img", req_json);
            }

            if (resp.empty()) {
                // Try auto-spawning sd-server.exe in the background if available
                STARTUPINFOA si = { sizeof(si) };
                PROCESS_INFORMATION pi;
                std::string sd_cmd = "C:\\Users\\admin\\stable-diffusion-cpp\\sd-server.exe -m C:\\Users\\admin\\stable-diffusion-cpp\\models\\DreamShaper8_LCM_q4_0.gguf --taesd C:\\Users\\admin\\stable-diffusion-cpp\\models\\taesd.safetensors --sampling-method lcm --steps 6 --cfg-scale 1.8 --listen-port 8085 --threads 8";
                if (CreateProcessA(NULL, (char*)sd_cmd.c_str(), NULL, NULL, FALSE, CREATE_NO_WINDOW, NULL, "C:\\Users\\admin\\stable-diffusion-cpp", &si, &pi)) {
                    CloseHandle(pi.hProcess);
                    CloseHandle(pi.hThread);
                    Sleep(3000);
                    resp = http_post_json(L"127.0.0.1", 8085, L"/v1/images/generations", req_json);
                }
            }

            if (resp.empty()) {
                output = "🎨 [SDArtPlugin] Could not connect to Stable Diffusion C++ backend on port 8085. Please ensure sd-server is running.";
                return true;
            }

            // Extract base64 image data
            std::string b64 = extract_json_string(resp, "b64_json");
            if (b64.empty()) {
                b64 = extract_json_string(resp, "image");
            }

            if (b64.empty()) {
                output = "🎨 [SDArtPlugin] Stable Diffusion responded but no image buffer was returned:\n" + resp.substr(0, 300);
                return true;
            }

            // Decode and write to wwwroot/uploads
            auto bytes = base64_decode(b64);
            if (bytes.empty()) {
                output = "🎨 [SDArtPlugin] Failed to decode base64 image buffer.";
                return true;
            }

            auto now = std::chrono::system_clock::now();
            auto time_t_now = std::chrono::system_clock::to_time_t(now);
            std::stringstream ss;
            ss << "aura_art_" << time_t_now << ".png";
            std::string fname = ss.str();

            std::string uploads_dir = "wwwroot/uploads";
            if (!std::filesystem::exists(uploads_dir)) {
                uploads_dir = "C:\\Users\\admin\\source\\haven-cpp\\wwwroot\\uploads";
            }
            std::filesystem::create_directories(uploads_dir);
            std::string file_path = uploads_dir + "/" + fname;

            FILE* f = fopen(file_path.c_str(), "wb");
            if (f) {
                fwrite(bytes.data(), 1, bytes.size(), f);
                fclose(f);

                output = "🎨 Artwork Painted: \"" + prompt + "\"\n"
                       + "🖼️ Saved: " + file_path + " (" + std::to_string(bytes.size() / 1024) + " KB)\n"
                       + "🌐 URL: /uploads/" + fname + "\n"
                       + "![Artwork](/uploads/" + fname + ")";
                return true;
            } else {
                output = "🎨 [SDArtPlugin] Image generated successfully, but failed to write to " + file_path;
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

    static std::string extract_json_string(const std::string& json, const std::string& key) {
        std::string search = "\"" + key + "\":";
        size_t pos = json.find(search);
        if (pos == std::string::npos) return "";
        size_t start = json.find("\"", pos + search.length());
        if (start == std::string::npos) return "";
        start++;
        size_t end = json.find("\"", start);
        if (end == std::string::npos) return "";
        return json.substr(start, end - start);
    }

    static std::vector<uint8_t> base64_decode(const std::string& in) {
        std::vector<uint8_t> out;
        int T[256];
        std::fill(T, T + 256, -1);
        const char* chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        for (int i = 0; i < 64; i++) T[(unsigned char)chars[i]] = i;
        
        int val = 0, valb = -8;
        for (unsigned char c : in) {
            if (T[c] == -1) continue;
            val = (val << 6) + T[c];
            valb += 6;
            if (valb >= 0) {
                out.push_back((uint8_t)((val >> valb) & 0xFF));
                valb -= 8;
            }
        }
        return out;
    }

    static std::string http_post_json(const std::wstring& host, int port, const std::wstring& path, const std::string& json_body) {
        HINTERNET hSession = WinHttpOpen(L"Haven-SD-Plugin/1.0", WINHTTP_ACCESS_TYPE_DEFAULT_PROXY, WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, 0);
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
    return new haven::SDVisionArtPlugin();
}

HAVEN_PLUGIN_EXPORT void destroy_haven_plugin(haven::IHavenPlugin* plugin) {
    delete plugin;
}
