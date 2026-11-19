#include "haven/haven_plugin.h"
#include <iostream>
#include <sstream>
#include <string>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

namespace haven {

class ClipboardInspectorPlugin : public IHavenPlugin {
public:
    PluginMetadata get_metadata() const override {
        PluginMetadata meta;
        meta.id = "haven.plugin.clipboard";
        meta.name = "Windows Clipboard Bridge & Context Sensor";
        meta.version = "1.0.0";
        meta.author = "Haven Sovereign Core";
        meta.description = "Reads copied text, code snippets, and URLs directly from the active Windows clipboard buffer";
        meta.capabilities = { PluginCapability::SensoryInput, PluginCapability::ActionTool };
        return meta;
    }

    bool on_load(PluginExecutionContext* ctx) override {
        (void)ctx;
        std::cout << "📋 [ClipboardInspectorPlugin] Windows clipboard bridge online.\n";
        return true;
    }

    void on_unload() override {
        std::cout << "📋 [ClipboardInspectorPlugin] Unloaded.\n";
    }

    bool can_handle_tool(const std::string& action) const override {
        return (action == "read_clipboard" ||
                action == "get_clipboard" ||
                action == "copy_to_clipboard" ||
                action == "set_clipboard" ||
                action == "clipboard_status");
    }

    bool execute_tool(const std::string& action, const std::string& payload, std::string& output) override {
#ifdef _WIN32
        if (action == "clipboard_status") {
            output = "📋 Windows Clipboard Sensor: ONLINE\nReady to read copied text, links, or code snippets from Daniel's clipboard.";
            return true;
        }

        if (action == "read_clipboard" || action == "get_clipboard") {
            if (!OpenClipboard(NULL)) {
                output = "📋 [Clipboard] Failed to open Windows clipboard (in use by another process).";
                return true;
            }

            HANDLE hData = GetClipboardData(CF_UNICODETEXT);
            if (hData == NULL) {
                CloseClipboard();
                output = "📋 [Clipboard] Current clipboard is empty or does not contain text.";
                return true;
            }

            wchar_t* pszText = static_cast<wchar_t*>(GlobalLock(hData));
            if (pszText == NULL) {
                CloseClipboard();
                output = "📋 [Clipboard] Failed to lock clipboard memory.";
                return true;
            }

            int size_needed = WideCharToMultiByte(CP_UTF8, 0, pszText, -1, NULL, 0, NULL, NULL);
            std::string utf8_text(size_needed, 0);
            WideCharToMultiByte(CP_UTF8, 0, pszText, -1, &utf8_text[0], size_needed, NULL, NULL);
            if (!utf8_text.empty() && utf8_text.back() == '\0') utf8_text.pop_back();

            GlobalUnlock(hData);
            CloseClipboard();

            if (utf8_text.length() > 3000) {
                utf8_text = utf8_text.substr(0, 3000) + "\n... [Truncated for attention context]";
            }

            output = "📋 Active Clipboard Contents:\n```\n" + utf8_text + "\n```";
            return true;
        }

        if (action == "copy_to_clipboard" || action == "set_clipboard") {
            if (payload.empty()) {
                output = "📋 [Clipboard] No text provided to copy.";
                return true;
            }

            if (!OpenClipboard(NULL)) {
                output = "📋 [Clipboard] Failed to open Windows clipboard.";
                return true;
            }

            EmptyClipboard();

            int wlen = MultiByteToWideChar(CP_UTF8, 0, payload.c_str(), -1, NULL, 0);
            HGLOBAL hGlob = GlobalAlloc(GMEM_MOVEABLE, wlen * sizeof(wchar_t));
            if (!hGlob) {
                CloseClipboard();
                output = "📋 [Clipboard] Failed to allocate global memory.";
                return true;
            }

            wchar_t* pGlob = static_cast<wchar_t*>(GlobalLock(hGlob));
            MultiByteToWideChar(CP_UTF8, 0, payload.c_str(), -1, pGlob, wlen);
            GlobalUnlock(hGlob);

            SetClipboardData(CF_UNICODETEXT, hGlob);
            CloseClipboard();

            output = "📋 Copied text to Daniel's clipboard: \"" + payload.substr(0, 80) + "...\"";
            return true;
        }
#endif
        (void)action; (void)payload; (void)output;
        return false;
    }
};

} // namespace haven

HAVEN_PLUGIN_EXPORT haven::IHavenPlugin* create_haven_plugin() {
    return new haven::ClipboardInspectorPlugin();
}

HAVEN_PLUGIN_EXPORT void destroy_haven_plugin(haven::IHavenPlugin* plugin) {
    delete plugin;
}
