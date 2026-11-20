cmake_minimum_required(VERSION 3.20)
project(haven_cpp CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable AVX2, FMA, F16C, OpenMP and Aggressive Optimization
if (MSVC)
    add_compile_options(/O2 /arch:AVX2 /W4 /openmp)
else()
    add_compile_options(-O3 -march=native -mavx2 -mfma -mf16c -fopenmp -Wall -Wextra)
    add_link_options(-fopenmp)
endif()

include_directories(include "C:/Users/admin/source/ollama-main/mlx/include")

link_directories(
    "C:/Users/admin/source/ollama-main/build/mlx/mlx/_deps/mlx-c-build"
    "C:/Users/admin/source/ollama-main/build/mlx/mlx/_deps/mlx-build"
    "C:/Users/admin/source/ollama-main/build/mlx/mlx"
)

set(HAVEN_CORE_SRCS
    src/haven_avx2.cpp
    src/haven_memory.cpp
    src/haven_sampler.cpp
    src/haven_gguf.cpp
    src/haven_kvcache.cpp
    src/haven_knowledge_store.cpp
    src/haven_attention_telemetry.cpp
    src/haven_sensory.cpp
    src/haven_paged_attention.cpp
    src/haven_yarn.cpp
    src/haven_speculative.cpp
    src/haven_flash_attention.cpp
    src/haven_chunked_prefill.cpp
    src/haven_tokenizer.cpp
    src/haven_engine.cpp
    src/haven_plugin_manager.cpp
    src/haven_topology.cpp
    src/haven_pathways.cpp
    src/haven_mlx.cpp
    src/haven_mcp.cpp
)

# Shared Library (haven_engine.dll / libhaven.so) for C# Gemmi & Android JNI
add_library(haven_engine SHARED
    ${HAVEN_CORE_SRCS}
)

# Standalone CLI Executable & Benchmark Runner
add_executable(haven-cli
    src/main.cpp
    ${HAVEN_CORE_SRCS}
)

# Standalone Sovereign Streaming Micro-Server
add_executable(haven-server
    src/haven_server.cpp
    ${HAVEN_CORE_SRCS}
)

# Test Standalone Forward Logits Executable
add_executable(test-standalone
    tests/test_forward_logits.cpp
    ${HAVEN_CORE_SRCS}
)

# Test Quantization Precision Executable
add_executable(test-quant
    tests/test_quant_precision.cpp
    ${HAVEN_CORE_SRCS}
)

# Test Lazy Plugin Manager Executable
add_executable(test-lazy-plugins
    tests/test_lazy_plugins.cpp
    ${HAVEN_CORE_SRCS}
)

# Test Haven Topology Fabric & Clean-Room Pathways Engine
add_executable(test-topology-pathways
    tests/test_topology_pathways.cpp
    ${HAVEN_CORE_SRCS}
)

# Test Haven Topology Fabric & Pathways End-to-End & C ABI Exports
add_executable(test-pathways-e2e
    tests/test_pathways_e2e.cpp
    ${HAVEN_CORE_SRCS}
)

# Test Multi-Model Architecture Support (Qwen 2.5, LLaMA 3, Phi-3, Gemma)
add_executable(test-multi-model-arch
    tests/test_multi_model_arch.cpp
    ${HAVEN_CORE_SRCS}
)

# Native Sovereign Plugins
add_library(plugin_yt_music SHARED
    plugins/yt_music_plugin.cpp
)
set_target_properties(plugin_yt_music PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/plugins"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/plugins"
)

add_library(plugin_system_sensory SHARED
    plugins/system_sensory_plugin.cpp
)
set_target_properties(plugin_system_sensory PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/plugins"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/plugins"
)

add_library(plugin_screen_vision SHARED
    plugins/screen_vision_plugin.cpp
)
set_target_properties(plugin_screen_vision PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/plugins"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/plugins"
)

add_library(plugin_vault_sync SHARED
    plugins/vault_sync_plugin.cpp
)
set_target_properties(plugin_vault_sync PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/plugins"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/plugins"
)

add_library(plugin_web_research SHARED
    plugins/web_research_plugin.cpp
)
set_target_properties(plugin_web_research PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/plugins"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/plugins"
)

add_library(plugin_discord_bridge SHARED
    plugins/discord_bridge_plugin.cpp
)
set_target_properties(plugin_discord_bridge PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/plugins"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/plugins"
)

add_library(plugin_sd_art SHARED
    plugins/sd_vision_art_plugin.cpp
)
set_target_properties(plugin_sd_art PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/plugins"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/plugins"
)

add_library(plugin_voice_synth SHARED
    plugins/voice_synthesis_plugin.cpp
)
set_target_properties(plugin_voice_synth PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/plugins"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/plugins"
)

add_library(plugin_browser SHARED
    plugins/sovereign_browser_plugin.cpp
)
set_target_properties(plugin_browser PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/plugins"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/plugins"
)

add_library(plugin_gaming SHARED
    plugins/gaming_companion_plugin.cpp
)
set_target_properties(plugin_gaming PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/plugins"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/plugins"
)

add_library(plugin_circadian SHARED
    plugins/circadian_rhythm_plugin.cpp
)
set_target_properties(plugin_circadian PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/plugins"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/plugins"
)

add_library(plugin_workspace_coder SHARED
    plugins/workspace_coder_plugin.cpp
)
set_target_properties(plugin_workspace_coder PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/plugins"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/plugins"
)

add_library(plugin_terminal SHARED
    plugins/terminal_sandbox_plugin.cpp
)
set_target_properties(plugin_terminal PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/plugins"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/plugins"
)

add_library(plugin_clipboard SHARED
    plugins/clipboard_inspector_plugin.cpp
)
set_target_properties(plugin_clipboard PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/plugins"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/plugins"
)

add_library(plugin_sqlite_graph SHARED
    plugins/sqlite_graph_plugin.cpp
)
set_target_properties(plugin_sqlite_graph PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/plugins"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/plugins"
)

add_library(plugin_whisper_ear SHARED
    plugins/whisper_audio_ear_plugin.cpp
)
set_target_properties(plugin_whisper_ear PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/plugins"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/plugins"
)

add_library(plugin_smart_home SHARED
    plugins/smart_home_iot_plugin.cpp
)
set_target_properties(plugin_smart_home PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/plugins"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/plugins"
)

if (WIN32)
    target_link_libraries(haven_engine mlxc mlx openblas)
    target_link_libraries(haven-cli mlxc mlx openblas)
    target_link_libraries(haven-server ws2_32 psapi mlxc mlx openblas)
    target_link_libraries(test-standalone ws2_32 mlxc mlx openblas)
    target_link_libraries(test-quant ws2_32 mlxc mlx openblas)
    target_link_libraries(test-lazy-plugins ws2_32 psapi mlxc mlx openblas)
    target_link_libraries(test-topology-pathways mlxc mlx openblas)
    target_link_libraries(test-pathways-e2e mlxc mlx openblas)
    target_link_libraries(test-multi-model-arch mlxc mlx openblas)
    target_link_libraries(plugin_system_sensory psapi)
    target_link_libraries(plugin_screen_vision psapi)
    target_link_libraries(plugin_web_research winhttp)
    target_link_libraries(plugin_discord_bridge winhttp)
    target_link_libraries(plugin_sd_art winhttp)
    target_link_libraries(plugin_voice_synth winhttp)
    target_link_libraries(plugin_browser winhttp shell32)
    target_link_libraries(plugin_gaming psapi)
    target_link_libraries(plugin_whisper_ear winhttp)
    target_link_libraries(plugin_smart_home winhttp)
endif()