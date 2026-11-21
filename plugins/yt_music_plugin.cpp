#include "haven/haven_plugin.h"
#include <iostream>
#include <string>
#include <algorithm>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <shellapi.h>
#endif

namespace haven {

class YouTubeMusicPlugin : public IHavenPlugin {
public:
    PluginMetadata get_metadata() const override {
        PluginMetadata meta;
        meta.id = "haven.plugin.yt_music";
        meta.name = "YouTube Music Sovereign Controller";
        meta.version = "1.0.0";
        meta.author = "Haven Sovereign Core";
        meta.description = "Controls YouTube Music playback, search, track navigation, and volume on Windows";
        meta.capabilities = { PluginCapability::ActionTool };
        return meta;
    }

    bool on_load(PluginExecutionContext* ctx) override {
        (void)ctx;
        std::cout << "🎵 [YouTubeMusicPlugin] Initialized YouTube Music media controller hook.\n";
        return true;
    }

    void on_unload() override {
        std::cout << "🎵 [YouTubeMusicPlugin] Unloaded.\n";
    }

    bool can_handle_tool(const std::string& action) const override {
        return (action == "yt_music_play" ||
                action == "yt_music_pause" ||
                action == "yt_music_toggle" ||
                action == "yt_music_next" ||
                action == "yt_music_prev" ||
                action == "yt_music_search" ||
                action == "yt_music_open");
    }

    bool execute_tool(const std::string& action, const std::string& payload, std::string& output) override {
#ifdef _WIN32
        if (action == "yt_music_play" || action == "yt_music_pause" || action == "yt_music_toggle") {
            keybd_event(VK_MEDIA_PLAY_PAUSE, 0, 0, 0);
            keybd_event(VK_MEDIA_PLAY_PAUSE, 0, KEYEVENTF_KEYUP, 0);
            output = "🎵 Sent Play/Pause toggle to YouTube Music.";
            return true;
        }
        else if (action == "yt_music_next") {
            keybd_event(VK_MEDIA_NEXT_TRACK, 0, 0, 0);
            keybd_event(VK_MEDIA_NEXT_TRACK, 0, KEYEVENTF_KEYUP, 0);
            output = "🎵 Skipped to next YouTube Music track.";
            return true;
        }
        else if (action == "yt_music_prev") {
            keybd_event(VK_MEDIA_PREV_TRACK, 0, 0, 0);
            keybd_event(VK_MEDIA_PREV_TRACK, 0, KEYEVENTF_KEYUP, 0);
            output = "🎵 Returned to previous YouTube Music track.";
            return true;
        }
        else if (action == "yt_music_search" && !payload.empty()) {
            std::string encoded_query;
            encoded_query.reserve(payload.size() * 3);
            static const char hex[] = "0123456789ABCDEF";
            for (unsigned char c : payload) {
                if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '-' || c == '_' || c == '.' || c == '~') {
                    encoded_query += static_cast<char>(c);
                } else if (c == ' ') {
                    encoded_query += '+';
                } else {
                    encoded_query += '%';
                    encoded_query += hex[(c >> 4) & 0xF];
                    encoded_query += hex[c & 0xF];
                }
            }
            std::string url = "https://music.youtube.com/search?q=" + encoded_query;
            ShellExecuteA(NULL, "open", url.c_str(), NULL, NULL, SW_SHOWNORMAL);
            output = "🎵 Opened YouTube Music search for: '" + payload + "'";
            return true;
        }
        else if (action == "yt_music_open") {
            ShellExecuteA(NULL, "open", "https://music.youtube.com", NULL, NULL, SW_SHOWNORMAL);
            output = "🎵 Opened YouTube Music in default browser.";
            return true;
        }
#endif
        output = "⚠️ YouTube Music action not supported on this platform: " + action;
        return false;
    }
};

} // namespace haven

// Native Dynamic Library Export Entrypoints
extern "C" {
#ifdef _WIN32
    HAVEN_PLUGIN_EXPORT haven::IHavenPlugin* create_haven_plugin() {
        return new haven::YouTubeMusicPlugin();
    }
    HAVEN_PLUGIN_EXPORT void destroy_haven_plugin(haven::IHavenPlugin* plugin) {
        delete plugin;
    }
#else
    haven::IHavenPlugin* create_haven_plugin() {
        return new haven::YouTubeMusicPlugin();
    }
    void destroy_haven_plugin(haven::IHavenPlugin* plugin) {
        delete plugin;
    }
#endif
}
