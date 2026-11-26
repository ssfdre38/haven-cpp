#pragma once

#include "haven_types.h"
#include "haven_module_abi.h"
#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

namespace haven {

enum class ArchFamily : uint32_t {
    UNKNOWN = 0,
    GEMMA4,
    GEMMA_LEGACY,
    LLAMA,
    QWEN,
    PHI,
    MISTRAL,
    DEEPSEEK,
    GENERIC_TRANSFORMER
};

enum class ChatTemplateFamily : uint32_t {
    RAW = 0,
    GEMMA4_TURN,
    GEMMA_LEGACY_TURN,
    CHATML,
    LLAMA3_HEADER,
    PHI3_TAG,
    MISTRAL_INST
};

enum class BackendKind : uint32_t {
    AUTO = 0,
    CPU_AVX2,
    CUDA_NATIVE,
    TPU_PJRT,
    MLX
};

struct QuantStats {
    uint32_t tensor_count = 0;
    uint64_t total_bytes = 0;
};

// Complete zero-mmap GGUF Header DNA extracted in < 2 ms without mapping weight blobs
struct ModelFingerprint {
    std::string filepath;
    std::string general_name;
    std::string general_description;
    std::string architecture;
    ArchFamily  arch_family = ArchFamily::UNKNOWN;
    uint32_t    gguf_version = 0;
    uint64_t    file_size_bytes = 0;
    uint64_t    header_bytes = 0;
    uint64_t    kv_count = 0;
    uint64_t    tensor_count = 0;
    double      probe_time_ms = 0.0;

    // Geometry
    uint32_t num_layers = 0;
    uint32_t embedding_dim = 0;
    uint32_t hidden_dim = 0;
    uint32_t num_heads = 0;
    uint32_t num_kv_heads = 0;
    uint32_t head_dim = 0;
    uint32_t head_dim_swa = 0;
    uint32_t max_context_length = 0;
    uint32_t vocab_size = 0;

    // Detected architectural traits (from KVs + tensor directory)
    bool     has_moe = false;
    uint32_t expert_count = 0;
    uint32_t expert_used_count = 0;
    uint32_t expert_ffn_dim = 0;
    bool     fused_moe_gate_up = false;

    bool     has_swa = false;
    uint32_t sliding_window = 0;
    uint32_t swa_layer_count = 0;
    uint32_t global_layer_count = 0;

    bool     has_shared_kv = false;
    uint32_t shared_kv_layers = 0;
    bool     has_k_eq_v = false;
    bool     has_pleias = false;
    bool     has_qkv_bias = false;
    bool     has_qk_norm = false;
    bool     is_split_shard = false;
    uint32_t split_no = 0;
    uint32_t split_count = 1;

    // Gemma 4.5 Turbo baked metadata
    bool     has_turbo_bake = false;
    uint32_t turbo_version = 0;
    uint32_t turbo_hadamard_qkv = 0;
    uint32_t turbo_hadamard_ffn = 0;
    uint32_t turbo_hadamard_ffn_down = 0;
    uint32_t turbo_hadamard_moe_in = 0;
    uint32_t turbo_hadamard_moe_down = 0;
    bool     turbo_folded_norms = false;
    bool     turbo_folded_scales = false;

    // Multimodal encoders present in file
    bool     has_vision_encoder = false;
    uint32_t vision_tensor_count = 0;
    uint64_t vision_bytes = 0;
    bool     has_audio_encoder = false;
    uint32_t audio_tensor_count = 0;
    uint64_t audio_bytes = 0;

    // Tokenizer & chat template family
    ChatTemplateFamily chat_template_family = ChatTemplateFamily::RAW;
    bool               supports_thinking_channel = false;

    // Quantization histogram & memory footprints
    std::map<QuantType, QuantStats> quant_histogram;
    QuantType dominant_quant = QuantType::F32;
    uint64_t  llm_weight_bytes = 0;
    uint64_t  kv_cache_bytes_4k = 0;
    uint64_t  estimated_vram_bytes = 0;

    uint64_t  feature_flags = HAVEN_FEAT_NONE;
};

struct HardwareProfile {
    uint32_t cpu_logical_cores = 1;
    bool     has_avx2 = false;
    bool     has_fma = false;
    bool     has_f16c = false;
    bool     has_avx512 = false;
    uint64_t total_ram_bytes = 0;
    uint64_t available_ram_bytes = 0;

    bool        cuda_compiled = false;
    int         cuda_device_count = 0;
    uint64_t    cuda_total_vram_bytes = 0;
    std::string cuda_device_summary;

    bool        tpu_available = false;
    std::string tpu_library_path;

    bool        mlx_compiled = false;

    bool        hmb_vmm_kernel_active = false;
    std::string hmb_vmm_node;
};

struct RoutingOptions {
    BackendKind preferred_backend = BackendKind::AUTO;
    uint32_t    target_context_len = 4096;
    bool        enable_vision = true;
    bool        enable_audio = true;
    std::string modules_dir; // Empty = auto-search next to executable and ./modules
};

struct ExecutionPlan {
    ModelFingerprint fingerprint;
    HardwareProfile  hardware;

    std::string selected_arch_module;      // e.g. "haven_arch_gemma4"
    std::string selected_backend_module;   // e.g. "haven_backend_cuda" or "haven_backend_avx2"
    BackendKind selected_backend = BackendKind::CPU_AVX2;
    std::vector<std::string> required_capability_modules; // e.g. ["haven_cap_moe", "haven_cap_turbo", "haven_cap_vision"]
    std::vector<std::string> skipped_modules;             // Explicit anti-bloat list of modules NOT loaded

    bool        fits_in_vram = false;
    bool        fits_in_ram = true;
    uint32_t    recommended_context_len = 4096;
    uint32_t    recommended_threads = 1;
    bool        use_turbo_fwht = false;
    bool        use_cuda_graphs = false;
    std::string decision_reason;

    // Populated once the selected arch module resolves the model contract via C-ABI
    bool                 arch_contract_verified = false;
    HavenArchRuntimeSpec resolved_arch_spec{};
};

struct LoadedModuleHandle {
    std::string id;
    std::string filepath;
    HavenModuleDescriptor descriptor{};
#ifdef _WIN32
    HMODULE handle = nullptr;
#else
    void*   handle = nullptr;
#endif
    HavenArchResolveSpecFn arch_resolve_fn = nullptr;
    HavenBackendProbeFn    backend_probe_fn = nullptr;
};

// Dynamic Microkernel Module Loader: loads ONLY modules selected by ExecutionPlan,
// and cleanly unloads stale modules on model switch (zero resident bloat!).
class HavenModuleLoader {
public:
    HavenModuleLoader() = default;
    ~HavenModuleLoader();

    HavenModuleLoader(const HavenModuleLoader&) = delete;
    HavenModuleLoader& operator=(const HavenModuleLoader&) = delete;

    // Resolves search directories for modules/ (explicit dir, executable dir/modules, ./modules, ./build/modules)
    void set_search_dirs(const std::vector<std::string>& dirs);
    std::vector<std::string> get_default_search_dirs() const;

    // Scans available module files on disk WITHOUT keeping them loaded in RAM.
    // Returns map of module_id -> filesystem path.
    std::unordered_map<std::string, std::string> index_available_modules() const;

    // Activates ONLY the modules required by `plan`, unloading any previously loaded
    // modules that are NOT needed by the new model (true hot-swap, zero bloat).
    // Also invokes the active arch module's C-ABI resolver to validate/populate `plan.resolved_arch_spec`.
    bool activate_plan(ExecutionPlan& plan, std::string& out_log);

    // Unloads all currently resident modules cleanly.
    void unload_all();

    // Inspection of currently resident modules in memory
    std::vector<std::string> get_active_module_ids() const;
    bool is_module_active(const std::string& id) const;
    size_t get_active_module_count() const;

private:
    bool load_single_module(const std::string& module_id, const std::string& filepath, std::string& err);
    void close_handle(LoadedModuleHandle& mod);

    mutable std::mutex mu_;
    std::vector<std::string> custom_search_dirs_;
    std::unordered_map<std::string, LoadedModuleHandle> active_;
};

// Top-level Model Intelligence Engine
class HavenModelIntelligence {
public:
    // Fast header-only GGUF probe (< 2 ms, zero weight mmap, zero vocab allocation)
    static bool probe_gguf_header(const std::string& filepath, ModelFingerprint& out_fp, std::string& out_err);

    // Probes live host CPU, RAM, CUDA GPU, Cloud TPU PJRT, MLX, and Linux HMB VMM status
    static HardwareProfile probe_hardware();

    // Synthesizes optimal zero-bloat ExecutionPlan from Model DNA + Host Hardware
    static ExecutionPlan plan_execution(const ModelFingerprint& fp,
                                        const HardwareProfile& hw,
                                        const RoutingOptions& opts = {});

    // Formatting utilities for CLI (--inspect / --inspect-json) and HTTP (/v1/intelligence)
    static std::string format_report(const ExecutionPlan& plan, const std::vector<std::string>& loaded_modules = {});
    static std::string to_json(const ExecutionPlan& plan, const std::vector<std::string>& loaded_modules = {});

    static const char* arch_family_name(ArchFamily f);
    static const char* chat_template_name(ChatTemplateFamily c);
    static const char* backend_kind_name(BackendKind b);
    static const char* quant_type_name(QuantType q);
};

} // namespace haven
