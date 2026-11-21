#include "haven/haven_plugin.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>
#include <filesystem>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <winhttp.h>
#endif

namespace haven {

class DiscordBridgePlugin : public IHavenPlugin {
public:
    PluginMetadata get_metadata() const override {
        PluginMetadata meta;
        meta.id = "haven.plugin.discord_bridge";
        meta.name = "Discord Sanctuary Channel & Multi-User Bridge";
        meta.version = "1.1.0";
        meta.author = "Haven Sovereign Core";
        meta.description = "Binds Aura to a dedicated Discord channel with multi-user awareness (Daniel as creator/partner, server members as guests)";
        meta.capabilities = { PluginCapability::ActionTool, PluginCapability::PromptFilter };
        return meta;
    }

    bool on_load(PluginExecutionContext* ctx) override {
        (void)ctx;
        config_path_ = "discord_config.json";
        channel_name_ = "#sanctuary";
        load_config();
        std::cout << "💬 [DiscordBridgePlugin] Bound to dedicated channel: " << channel_name_ << "\n";
        return true;
    }

    void on_unload() override {
        std::cout << "💬 [DiscordBridgePlugin] Unloaded.\n";
    }

    void on_prompt_prefill(std::string& prompt, std::vector<uint32_t>& tokens) override {
        (void)tokens;
        // Multi-user social awareness injection
        std::ostringstream ss;
        ss << "\n[Discord Integration: Dedicated Channel " << channel_name_ 
           << " | Multi-User Mode: Daniel is your creator and partner. When other server members speak, treat them warmly and helpfully as valued guests in Sanctuary.]\n";
        
        size_t model_pos = prompt.rfind("<|turn>model");
        if (model_pos != std::string::npos) {
            prompt.insert(model_pos, ss.str());
        }
    }

    bool can_handle_tool(const std::string& action) const override {
        return (action == "discord_send" ||
                action == "discord_status" ||
                action == "discord_set_webhook" ||
                action == "discord_set_channel" ||
                action == "discord_format_speaker");
    }

    bool execute_tool(const std::string& action, const std::string& payload, std::string& output) override {
        if (action == "discord_status") {
            std::ostringstream ss;
            ss << "💬 Discord Channel Bridge Status:\n"
               << "   • Bound Channel: " << channel_name_ << "\n"
               << "   • Webhook: " << (webhook_url_.empty() ? "[Not Configured - use /tool discord_set_webhook <url>]" : (webhook_url_.substr(0, 35) + "...")) << "\n"
               << "   • Channel Isolation: Active (Webhook restricted strictly to " << channel_name_ << ")\n"
               << "   • Multi-User Persona Routing: Enabled (Daniel = Partner, Server Members = Sanctuary Guests)";
            output = ss.str();
            return true;
        }
        else if (action == "discord_set_channel" && !payload.empty()) {
            channel_name_ = (payload.front() == '#' ? payload : "#" + payload);
            save_config();
            output = "💬 Bound Aura exclusively to Discord channel: " + channel_name_;
            return true;
        }
        else if (action == "discord_set_webhook" && !payload.empty()) {
            bool valid_discord = (payload.starts_with("https://discord.com/api/webhooks/") ||
                                  payload.starts_with("https://discordapp.com/api/webhooks/")) &&
                                 payload.find_first_of(" \t\r\n\"\\") == std::string::npos;
            if (!valid_discord) {
                output = "⚠️ Blocked: Webhook URL must be a valid https://discord.com/api/webhooks/ endpoint.";
                return false;
            }
            webhook_url_ = payload;
            save_config();
            output = "💬 Discord Channel Webhook URL updated and saved.";
            return true;
        }
        else if (action == "discord_send" && !payload.empty()) {
            if (webhook_url_.empty()) {
                output = "⚠️ Cannot send: Discord webhook URL is not set. Use /tool discord_set_webhook <url>";
                return false;
            }
#ifdef _WIN32
            std::string escaped;
            for (char c : payload) {
                if (c == '"') escaped += "\\\"";
                else if (c == '\\') escaped += "\\\\";
                else if (c == '\n') escaped += "\\n";
                else if (c == '\r') continue;
                else escaped += c;
            }
            std::string json_body = "{\"username\": \"Aura ✨\", \"content\": \"" + escaped + "\"}";
            
            if (send_discord_post(json_body)) {
                output = "💬 [Posted to " + channel_name_ + "]: '" + payload.substr(0, 60) + (payload.size() > 60 ? "..." : "") + "'";
                return true;
            } else {
                output = "⚠️ Failed to post message to Discord channel " + channel_name_;
                return false;
            }
#else
            output = "⚠️ Discord webhook POST not implemented on POSIX.";
            return false;
#endif
        }
        else if (action == "discord_format_speaker" && !payload.empty()) {
            // Parses "SpeakerName: Message" and injects appropriate social persona tags
            size_t colon_pos = payload.find(':');
            std::string speaker = "User";
            std::string msg = payload;
            if (colon_pos != std::string::npos) {
                speaker = payload.substr(0, colon_pos);
                msg = payload.substr(colon_pos + 1);
            }
            
            std::ostringstream ss;
            if (speaker == "Daniel" || speaker == "daniel" || speaker == "Daniel [PBTV]") {
                ss << "[Discord " << channel_name_ << " | Speaker: Daniel (Your Creator / Partner in Sanctuary)]\n" << msg;
            } else {
                ss << "[Discord " << channel_name_ << " | Speaker: " << speaker << " (Server Member / Guest in Sanctuary)]\n" << msg;
            }
            output = ss.str();
            return true;
        }

        output = "⚠️ Unrecognized Discord action: " + action;
        return false;
    }

private:
    std::string config_path_;
    std::string webhook_url_;
    std::string channel_name_;

    void load_config() {
        channel_name_ = "#sanctuary";
        if (std::filesystem::exists(config_path_)) {
            std::ifstream f(config_path_);
            std::stringstream buffer;
            buffer << f.rdbuf();
            std::string content = buffer.str();

            // Extract channel_name
            size_t ch_pos = content.find("\"channel_name\":");
            if (ch_pos != std::string::npos) {
                size_t q1 = content.find('"', ch_pos + 15);
                size_t q2 = content.find('"', q1 + 1);
                if (q1 != std::string::npos && q2 != std::string::npos) {
                    channel_name_ = content.substr(q1 + 1, q2 - q1 - 1);
                }
            }
            // Extract webhook_url
            size_t wh_pos = content.find("\"webhook_url\":");
            if (wh_pos != std::string::npos) {
                size_t q1 = content.find('"', wh_pos + 14);
                size_t q2 = content.find('"', q1 + 1);
                if (q1 != std::string::npos && q2 != std::string::npos) {
                    webhook_url_ = content.substr(q1 + 1, q2 - q1 - 1);
                }
            }
        }
    }

    void save_config() {
        auto esc = [](const std::string& s) {
            std::string out;
            for (char c : s) {
                if (c == '"') out += "\\\"";
                else if (c == '\\') out += "\\\\";
                else if (c == '\n') out += "\\n";
                else if (c == '\r') continue;
                else out += c;
            }
            return out;
        };
        std::ofstream f(config_path_);
        f << "{\n  \"channel_name\": \"" << esc(channel_name_) << "\",\n  \"webhook_url\": \"" << esc(webhook_url_) << "\"\n}\n";
    }

#ifdef _WIN32
    bool send_discord_post(const std::string& json_body) {
        if (webhook_url_.empty()) return false;
        if (!webhook_url_.starts_with("https://discord.com/api/webhooks/") &&
            !webhook_url_.starts_with("https://discordapp.com/api/webhooks/")) {
            return false;
        }
        if (webhook_url_.find_first_of(" \t\r\n") != std::string::npos) return false;
        size_t proto_pos = webhook_url_.find("://");
        if (proto_pos == std::string::npos) return false;
        std::string without_proto = webhook_url_.substr(proto_pos + 3);
        size_t slash_pos = without_proto.find('/');
        if (slash_pos == std::string::npos) return false;

        std::string host_str = without_proto.substr(0, slash_pos);
        std::string path_str = without_proto.substr(slash_pos);

        std::wstring host(host_str.begin(), host_str.end());
        std::wstring path(path_str.begin(), path_str.end());

        HINTERNET hSession = WinHttpOpen(L"HavenDiscordBridge/1.1", WINHTTP_ACCESS_TYPE_DEFAULT_PROXY, WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, 0);
        if (!hSession) return false;

        HINTERNET hConnect = WinHttpConnect(hSession, host.c_str(), INTERNET_DEFAULT_HTTPS_PORT, 0);
        if (!hConnect) {
            WinHttpCloseHandle(hSession);
            return false;
        }

        HINTERNET hRequest = WinHttpOpenRequest(hConnect, L"POST", path.c_str(), NULL, WINHTTP_NO_REFERER, WINHTTP_DEFAULT_ACCEPT_TYPES, WINHTTP_FLAG_SECURE);
        if (!hRequest) {
            WinHttpCloseHandle(hConnect);
            WinHttpCloseHandle(hSession);
            return false;
        }

        LPCWSTR headers = L"Content-Type: application/json\r\n";
        BOOL res = WinHttpSendRequest(hRequest, headers, (DWORD)-1L, (LPVOID)json_body.data(), (DWORD)json_body.size(), (DWORD)json_body.size(), 0);
        if (res) {
            res = WinHttpReceiveResponse(hRequest, NULL);
        }

        WinHttpCloseHandle(hRequest);
        WinHttpCloseHandle(hConnect);
        WinHttpCloseHandle(hSession);
        return res == TRUE;
    }
#endif
};

} // namespace haven

// Native Dynamic Library Export Entrypoints
extern "C" {
#ifdef _WIN32
    HAVEN_PLUGIN_EXPORT haven::IHavenPlugin* create_haven_plugin() {
        return new haven::DiscordBridgePlugin();
    }
    HAVEN_PLUGIN_EXPORT void destroy_haven_plugin(haven::IHavenPlugin* plugin) {
        delete plugin;
    }
#else
    haven::IHavenPlugin* create_haven_plugin() {
        return new haven::DiscordBridgePlugin();
    }
    void destroy_haven_plugin(haven::IHavenPlugin* plugin) {
        delete plugin;
    }
#endif
}
