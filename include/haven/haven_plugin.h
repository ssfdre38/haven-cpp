#pragma once

#include "haven_types.h"
#include <string>
#include <vector>
#include <memory>
#include <functional>

namespace haven {

// Capabilities that a plugin can register
enum class PluginCapability {
    SensoryInput,    // Screen vision, microphone, system telemetry
    CognitiveMemory, // Obsidian, Notion, journal, relational graph
    ActionTool,      // YouTube Music, shell, web search, home automation
    VoiceExpression, // TTS synthesis, avatar blendshapes, haptics
    PromptFilter     // Persona shaping, safety filters, contextual anchors
};

struct PluginMetadata {
    std::string id;
    std::string name;
    std::string version;
    std::string author;
    std::string description;
    std::vector<PluginCapability> capabilities;
};

// Context passed to plugins during engine events
struct PluginExecutionContext {
    int active_seq_pos = 0;
    float current_temperature = 0.70f;
    std::string current_model_name;
    void* raw_engine_ptr = nullptr;
};

// Pure abstract interface for all Haven sovereign plugins
class IHavenPlugin {
public:
    virtual ~IHavenPlugin() = default;

    // Lifecycle
    virtual PluginMetadata get_metadata() const = 0;
    virtual bool on_load(PluginExecutionContext* ctx) = 0;
    virtual void on_unload() = 0;

    // Prompt & Token Hooks
    virtual void on_prompt_prefill(std::string& /*prompt*/, std::vector<uint32_t>& /*tokens*/) {}
    virtual void on_token_generated(uint32_t /*token_id*/, const std::string& /*piece*/) {}

    // Tool & Action Dispatcher (e.g. YouTube Music, System Control, Web Search)
    virtual bool can_handle_tool(const std::string& /*action*/) const { return false; }
    virtual bool execute_tool(const std::string& /*action*/, const std::string& /*payload*/, std::string& /*output*/) { return false; }

    // Sensory Ingestion Hook
    virtual void on_sensory_tick(std::vector<float>& /*vision_embedding*/, std::vector<float>& /*audio_embedding*/) {}
};

// Function signatures exported by native plugin dynamic libraries (.dll / .so)
#ifdef _WIN32
#define HAVEN_PLUGIN_EXPORT extern "C" __declspec(dllexport)
#else
#define HAVEN_PLUGIN_EXPORT extern "C" __attribute__((visibility("default")))
#endif

using CreatePluginFunc = IHavenPlugin* (*)();
using DestroyPluginFunc = void (*)(IHavenPlugin*);

} // namespace haven
