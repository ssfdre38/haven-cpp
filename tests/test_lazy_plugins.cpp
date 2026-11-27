#include "haven/haven_plugin_manager.h"
#include <iostream>
#include <cassert>
#include "haven/haven_test_check.h"

using namespace haven;

int main() {
    std::cout << "🧪 Testing Haven-CPP On-Demand Lazy Plugin Architecture...\n\n";

    PluginManager pm;
    size_t count = pm.discover_plugins("plugins", true);
    if (count == 0) count = pm.discover_plugins("C:\\Users\\admin\\source\\haven-cpp\\plugins", true);
    std::cout << "\n✓ Discovered " << count << " plugins.\n";
    std::cout << "✓ Active in RAM at boot: " << pm.get_active_plugin_count() << " (Expected: 0)\n";
    assert(pm.get_active_plugin_count() == 0);

    // Verify metadata was captured even though DLL is cold
    std::string prompt_desc = pm.get_tools_prompt_description();
    std::cout << "✓ Prompt tools description length: " << prompt_desc.length() << " chars\n";
    assert(prompt_desc.find("ON-DEMAND") != std::string::npos);

    // Test on-demand hot-loading of a tool (e.g. system_telemetry)
    std::cout << "\n⚡ Invoking 'system_telemetry' tool action...\n";
    std::string output;
    bool ok = pm.dispatch_tool_execution("system_telemetry", "", output);
    std::cout << "✓ Tool execution ok: " << (ok ? "true" : "false") << "\n";
    std::cout << "✓ Tool output: " << output << "\n";
    std::cout << "✓ Active in RAM after invocation: " << pm.get_active_plugin_count() << " (Expected: 1)\n";
    assert(pm.get_active_plugin_count() == 1);

    // Test idle plugin unloading (force unload with 0s threshold)
    std::cout << "\n❄️ Testing idle plugin reaper...\n";
    size_t unloaded = pm.unload_idle_plugins(0);
    std::cout << "✓ Unloaded plugins: " << unloaded << "\n";
    std::cout << "✓ Active in RAM after reaper: " << pm.get_active_plugin_count() << " (Expected: 0)\n";
    assert(pm.get_active_plugin_count() == 0);

    std::cout << "\n🎉 ALL LAZY HOT-LOADING TESTS PASSED WITH 100% SUCCESS!\n";
    return 0;
}
