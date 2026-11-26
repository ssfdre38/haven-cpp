cmake_minimum_required(VERSION 3.20)
project(haven_cpp CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(HAVEN_ENABLE_CUDA "Build native NVIDIA CUDA GPU backend (IQ4_XS/Q-block GEMV + cuBLAS Tensor Core prefill)" OFF)
if (HAVEN_ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "80;86;89;90")
    endif()
    add_compile_definitions(HAVEN_HAS_CUDA)
endif()

# Enable AVX2, FMA, F16C, OpenMP and Aggressive Optimization
if (MSVC)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/O2> $<$<COMPILE_LANGUAGE:CXX>:/arch:AVX2> $<$<COMPILE_LANGUAGE:CXX>:/W4> $<$<COMPILE_LANGUAGE:CXX>:/openmp> $<$<COMPILE_LANGUAGE:CXX>:/utf-8>)
    add_compile_definitions(NOMINMAX _CRT_SECURE_NO_WARNINGS)
else()
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-O3> $<$<COMPILE_LANGUAGE:CXX>:-march=native> $<$<COMPILE_LANGUAGE:CXX>:-mavx2> $<$<COMPILE_LANGUAGE:CXX>:-mfma> $<$<COMPILE_LANGUAGE:CXX>:-mf16c> $<$<COMPILE_LANGUAGE:CXX>:-fopenmp> $<$<COMPILE_LANGUAGE:CXX>:-Wall> $<$<COMPILE_LANGUAGE:CXX>:-Wextra>)
    if (HAVEN_ENABLE_CUDA)
        add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-O3> $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)
    endif()
    add_link_options(-fopenmp)
    if (WIN32)
        add_compile_definitions(NOMINMAX)
    endif()
endif()

if (WIN32)
    find_path(MLX_INCLUDE_DIR "mlx/c/mlx.h"
        PATHS "C:/Users/admin/source/ollama-main/mlx/include"
    )
    if (MLX_INCLUDE_DIR)
        include_directories(include ${MLX_INCLUDE_DIR})
        add_definitions(-DHAVEN_HAS_MLX)
        link_directories(
            "C:/Users/admin/source/ollama-main/build/mlx/mlx/_deps/mlx-c-build"
            "C:/Users/admin/source/ollama-main/build/mlx/mlx/_deps/mlx-build"
            "C:/Users/admin/source/ollama-main/build/mlx/mlx"
        )
    else()
        include_directories(include)
    endif()
elseif (APPLE)
    find_path(MLX_INCLUDE_DIR "mlx/c/mlx.h"
        PATHS "/usr/local/include"
    )
    if (MLX_INCLUDE_DIR)
        include_directories(include ${MLX_INCLUDE_DIR})
        add_definitions(-DHAVEN_HAS_MLX)
    else()
        include_directories(include)
    endif()
else()
    include_directories(include)
endif()

set(HAVEN_CORE_SRCS
    src/haven_avx2.cpp
    src/haven_memory.cpp
    src/haven_sampler.cpp
    src/haven_gguf.cpp
    src/haven_intelligence.cpp
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
    src/haven_tpu_pjrt.cpp
    src/haven_tpu_gemma4.cpp
    src/haven_tpu_scheduler.cpp
    src/haven_mlx.cpp
    src/haven_mcp.cpp
    src/haven_cluster.cpp
)

if (HAVEN_ENABLE_CUDA)
    list(APPEND HAVEN_CORE_SRCS src/haven_cuda.cu)
else()
    list(APPEND HAVEN_CORE_SRCS src/haven_cuda_stub.cpp)
endif()

# Core Static Engine Library (builds all core sources once for fast compilation)
add_library(haven_core STATIC
    ${HAVEN_CORE_SRCS}
)

if (WIN32)
    if (MLX_INCLUDE_DIR)
        target_link_libraries(haven_core PUBLIC ws2_32 psapi mlxc mlx openblas)
    else()
        target_link_libraries(haven_core PUBLIC ws2_32 psapi)
    endif()
else()
    target_link_libraries(haven_core PUBLIC pthread dl)
endif()
if (HAVEN_ENABLE_CUDA)
    target_link_libraries(haven_core PUBLIC CUDA::cudart CUDA::cublas)
endif()

# Shared Library (haven_engine.dll / libhaven.so) for C# Gemmi & Android JNI
add_library(haven_engine SHARED
    ${HAVEN_CORE_SRCS}
)
if (WIN32)
    if (MLX_INCLUDE_DIR)
        target_link_libraries(haven_engine PUBLIC ws2_32 psapi mlxc mlx openblas)
    else()
        target_link_libraries(haven_engine PUBLIC ws2_32 psapi)
    endif()
else()
    target_link_libraries(haven_engine PUBLIC pthread dl)
endif()
if (HAVEN_ENABLE_CUDA)
    target_link_libraries(haven_engine PUBLIC CUDA::cudart CUDA::cublas)
endif()

# Standalone CLI Executable & Benchmark Runner
add_executable(haven-cli src/main.cpp)
target_link_libraries(haven-cli PRIVATE haven_core)

# Standalone Sovereign Streaming Micro-Server
add_executable(haven-server src/haven_server.cpp)
target_link_libraries(haven-server PRIVATE haven_core)

# Native Gemma 4.5 GGUF Architecture Optimizer (Scale/Norm Folding + QuaRot R2 + Block-FWHT)
add_executable(haven-bake src/haven_bake.cpp)
target_link_libraries(haven-bake PRIVATE haven_core)

# Gemma 4.5 Turbo Bake & AVX2 FWHT Verification Test Suite
add_executable(test-turbo-bake tests/test_turbo_bake.cpp)
target_link_libraries(test-turbo-bake PRIVATE haven_core)

# Test Standalone Forward Logits Executable
add_executable(test-standalone tests/test_forward_logits.cpp)
target_link_libraries(test-standalone PRIVATE haven_core)

# Test Quantization Precision Executable
add_executable(test-quant tests/test_quant_precision.cpp)
target_link_libraries(test-quant PRIVATE haven_core)

# Test Lazy Plugin Manager Executable
add_executable(test-lazy-plugins tests/test_lazy_plugins.cpp)
target_link_libraries(test-lazy-plugins PRIVATE haven_core)

# Test Haven Topology Fabric & Clean-Room Pathways Engine
add_executable(test-topology-pathways tests/test_topology_pathways.cpp)
target_link_libraries(test-topology-pathways PRIVATE haven_core)

# Test Haven Topology Fabric & Pathways End-to-End & C ABI Exports
add_executable(test-pathways-e2e tests/test_pathways_e2e.cpp)
target_link_libraries(test-pathways-e2e PRIVATE haven_core)

# Test Multi-Model Architecture Support (Qwen 2.5, LLaMA 3, Phi-3, Gemma)
add_executable(test-multi-model-arch tests/test_multi_model_arch.cpp)
target_link_libraries(test-multi-model-arch PRIVATE haven_core)

# Test Haven Sovereign Multi-Node Cluster & Pipeline Parallel Fabric
add_executable(test-cluster-fabric tests/test_cluster_fabric.cpp)
target_link_libraries(test-cluster-fabric PRIVATE haven_core)

# Test Haven Cloud TPU v6e Trillium 8-Core Torus Mesh Cluster
add_executable(test-tpu-cluster-mesh tests/test_tpu_cluster_mesh.cpp)
target_link_libraries(test-tpu-cluster-mesh PRIVATE haven_core)

# Cross-System Multi-Node Cluster Executable (Windows / Linux / Cloud)
add_executable(test-cross-system-node tests/test_cross_system_node.cpp)
target_link_libraries(test-cross-system-node PRIVATE haven_core)

# Distributed Cluster + Pathways DAG Executable (Unified Multi-Node Engine)
add_executable(test-pathways-distributed-cluster tests/test_pathways_distributed_cluster.cpp)
target_link_libraries(test-pathways-distributed-cluster PRIVATE haven_core)

# Live Multi-Node Cluster Execution Test (Real-time Token Generation across Nodes)
add_executable(test-live-cluster-generation tests/test_live_cluster_generation.cpp)
target_link_libraries(test-live-cluster-generation PRIVATE haven_core)

# TPU Live Multi-Node Generation Test (Ava 4-Node Sovereign Cluster)
add_executable(test-tpu-live-generation tests/test_tpu_live_generation.cpp)
target_link_libraries(test-tpu-live-generation PRIVATE haven_core)

# Native C++20 PJRT (libtpu.so) Cloud TPU Silicon Benchmark
add_executable(test-tpu-pjrt-bench tests/test_tpu_pjrt_bench.cpp)
target_link_libraries(test-tpu-pjrt-bench PRIVATE haven_core)

# Full On-Chip Transformer StableHLO Persistent HBM3 TPU Engine Benchmark
add_executable(test-tpu-gemma4-onchip tests/test_tpu_gemma4_onchip.cpp)
target_link_libraries(test-tpu-gemma4-onchip PRIVATE haven_core)

# Native Zero-Compile-Time CUDA Engine Verification & Benchmark
add_executable(test-cuda-gemma4 tests/test_cuda_gemma4.cpp)
target_link_libraries(test-cuda-gemma4 PRIVATE haven_core)

# Model Intelligence Header DNA Router & Dynamic Microkernel Module Suite
add_executable(test-model-intelligence tests/test_model_intelligence.cpp)
target_link_libraries(test-model-intelligence PRIVATE haven_core)

# Dynamic Zero-Bloat Microkernel Modules (Arch / Backend / Capability)
function(haven_add_microkernel_module target_name subdir source_file)
    add_library(${target_name} SHARED ${source_file})
    set_target_properties(${target_name} PROPERTIES
        PREFIX ""
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/modules/${subdir}"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/modules/${subdir}"
    )
    if (NOT WIN32)
        target_link_libraries(${target_name} PRIVATE dl)
    endif()
endfunction()

haven_add_microkernel_module(haven_arch_gemma4  arch    modules/arch/haven_arch_gemma4.cpp)
haven_add_microkernel_module(haven_arch_llama   arch    modules/arch/haven_arch_llama.cpp)
haven_add_microkernel_module(haven_arch_qwen    arch    modules/arch/haven_arch_qwen.cpp)
haven_add_microkernel_module(haven_arch_phi     arch    modules/arch/haven_arch_phi.cpp)
haven_add_microkernel_module(haven_backend_avx2 backend modules/backend/haven_backend_avx2.cpp)
haven_add_microkernel_module(haven_backend_cuda backend modules/backend/haven_backend_cuda.cpp)
haven_add_microkernel_module(haven_backend_tpu  backend modules/backend/haven_backend_tpu.cpp)
haven_add_microkernel_module(haven_cap_moe      cap     modules/cap/haven_cap_moe.cpp)
haven_add_microkernel_module(haven_cap_turbo    cap     modules/cap/haven_cap_turbo.cpp)
haven_add_microkernel_module(haven_cap_vision   cap     modules/cap/haven_cap_vision.cpp)

add_custom_target(haven-modules DEPENDS
    haven_arch_gemma4 haven_arch_llama haven_arch_qwen haven_arch_phi
    haven_backend_avx2 haven_backend_cuda haven_backend_tpu
    haven_cap_moe haven_cap_turbo haven_cap_vision
)

# Multi-Tenant TPU Continuous-Batching Scheduler Unit Test
add_executable(test-tpu-scheduler tests/test_tpu_scheduler.cpp)
target_link_libraries(test-tpu-scheduler PRIVATE haven_core)

# Secure Cluster Protocol Verification Test (SHCP Mutual Authentication & Crypto Handshake)
add_executable(test-secure-cluster tests/test_secure_cluster.cpp)
target_link_libraries(test-secure-cluster PRIVATE haven_core)

# Adversarial Security, Crypto KAT, & Binary/JSON Fuzz Test Suite
add_executable(test-security-fuzz tests/test_security_fuzz.cpp)
target_link_libraries(test-security-fuzz PRIVATE haven_core)

# Autonomic Postural Sway Engine Targets
add_executable(test-haven-sway tests/test_haven_sway.cpp)
target_link_libraries(test-haven-sway PRIVATE haven_core)

add_executable(quickstart-sway examples/quickstart_sway.cpp)
target_link_libraries(quickstart-sway PRIVATE haven_core)

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
    target_link_libraries(plugin_system_sensory psapi)
    target_link_libraries(plugin_screen_vision psapi)
    target_link_libraries(plugin_web_research winhttp)
    target_link_libraries(plugin_discord_bridge winhttp)
    target_link_libraries(plugin_sd_art winhttp)
    target_link_libraries(plugin_voice_synth winhttp)
    target_link_libraries(plugin_browser winhttp shell32 ws2_32)
    target_link_libraries(plugin_gaming psapi)
    target_link_libraries(plugin_whisper_ear winhttp)
    target_link_libraries(plugin_smart_home winhttp)
endif()