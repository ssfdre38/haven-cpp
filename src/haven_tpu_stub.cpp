#include "haven/haven_tpu_pjrt.h"
#include "haven/haven_tpu_gemma4.h"
#include "haven/haven_tpu_scheduler.h"
#include "haven/haven_quant_spec.h"

// ============================================================================
// Zero-Bloat Cloud TPU Stub (Compiled ONLY when -DHAVEN_ENABLE_TPU=OFF)
// ============================================================================
// Drops 5,729 lines of Cloud TPU PJRT / StableHLO MLIR compiler code for
// ultra-lean desktop, laptop, Android, and edge builds with zero API breaks.
// ============================================================================

namespace haven {

// ---- HavenTpuPjrtBackend Stub ----
HavenTpuPjrtBackend::HavenTpuPjrtBackend() = default;
HavenTpuPjrtBackend::~HavenTpuPjrtBackend() = default;

bool HavenTpuPjrtBackend::initialize(const std::string& plugin_path_override) {
    (void)plugin_path_override;
    last_error_ = "Cloud TPU PJRT disabled in this lean build (-DHAVEN_ENABLE_TPU=OFF).";
    return false;
}
void HavenTpuPjrtBackend::shutdown() { initialized_ = false; }
void HavenTpuPjrtBackend::set_cache_dir(const std::string& dir) { disk_cache_dir_ = dir; }
std::string HavenTpuPjrtBackend::compute_stablehlo_cache_key(const std::string&, const std::string&, const std::string&, int, int, bool, const std::string&) { return ""; }
std::string HavenTpuPjrtBackend::resolve_default_cache_dir() { return ""; }
uint16_t HavenTpuPjrtBackend::fp32_to_bf16(float f) { return quant_f32_to_bf16_rne(f); }
float HavenTpuPjrtBackend::bf16_to_fp32(uint16_t b) { return quant_bf16_to_f32(b); }
void HavenTpuPjrtBackend::convert_f32_to_bf16(const float* src, uint16_t* dst, size_t n) {
    for (size_t i = 0; i < n; ++i) dst[i] = quant_f32_to_bf16_rne(src[i]);
}
void HavenTpuPjrtBackend::convert_bf16_to_f32(const uint16_t* src, float* dst, size_t n) {
    for (size_t i = 0; i < n; ++i) dst[i] = quant_bf16_to_f32(src[i]);
}
PJRT_Buffer* HavenTpuPjrtBackend::upload_bf16_2d(const uint16_t*, int64_t, int64_t, int) { return nullptr; }
PJRT_Buffer* HavenTpuPjrtBackend::upload_f32_2d(const float*, int64_t, int64_t, int) { return nullptr; }
PJRT_Buffer* HavenTpuPjrtBackend::upload_bf16_nd(const uint16_t*, const std::vector<int64_t>&, int) { return nullptr; }
PJRT_Buffer* HavenTpuPjrtBackend::upload_f32_nd(const float*, const std::vector<int64_t>&, int) { return nullptr; }
PJRT_Buffer* HavenTpuPjrtBackend::upload_s8_nd(const int8_t*, const std::vector<int64_t>&, int) { return nullptr; }
PJRT_Buffer* HavenTpuPjrtBackend::upload_f8e4m3fn_nd(const uint8_t*, const std::vector<int64_t>&, int) { return nullptr; }
PJRT_Buffer* HavenTpuPjrtBackend::upload_i32_nd(const int32_t*, const std::vector<int64_t>&, int) { return nullptr; }
bool HavenTpuPjrtBackend::download_f32(PJRT_Buffer*, float*, size_t) { return false; }
bool HavenTpuPjrtBackend::download_bf16(PJRT_Buffer*, uint16_t*, size_t) { return false; }
bool HavenTpuPjrtBackend::download_i32(PJRT_Buffer*, int32_t*, size_t) { return false; }
void HavenTpuPjrtBackend::destroy_buffer(PJRT_Buffer*) {}
PJRT_LoadedExecutable* HavenTpuPjrtBackend::compile_stablehlo(const std::string&, int) { return nullptr; }
PJRT_LoadedExecutable* HavenTpuPjrtBackend::compile_stablehlo_replicated(const std::string&, int) { return nullptr; }
std::vector<int> HavenTpuPjrtBackend::get_executable_device_indices(PJRT_LoadedExecutable*) { return {}; }
void HavenTpuPjrtBackend::destroy_executable(PJRT_LoadedExecutable*) {}
bool HavenTpuPjrtBackend::execute_1d(PJRT_LoadedExecutable*, int, const std::vector<PJRT_Buffer*>&, size_t, std::vector<PJRT_Buffer*>&) { return false; }
bool HavenTpuPjrtBackend::execute_replicated(PJRT_LoadedExecutable*, int, const std::vector<std::vector<PJRT_Buffer*>>&, size_t, std::vector<std::vector<PJRT_Buffer*>>&) { return false; }
bool HavenTpuPjrtBackend::upload_gguf_weight_matrix(const TensorDesc&, int, int, TpuMatrixHandle&, int) { return false; }
void HavenTpuPjrtBackend::free_matrix(TpuMatrixHandle& h) { h = {}; }
bool HavenTpuPjrtBackend::matmul_resident_nt(const float*, int, const TpuMatrixHandle&, float*) { return false; }
bool HavenTpuPjrtBackend::fused_geglu_ffn(const float*, int, const TpuMatrixHandle&, const TpuMatrixHandle&, const TpuMatrixHandle&, float*) { return false; }
TpuGemmBenchResult HavenTpuPjrtBackend::benchmark_bf16_gemm(int, int, int, int, int) { return {}; }
void HavenTpuPjrtBackend::refresh_memory_stats() {}

// ---- HavenTpuGemma4Engine Stub ----
TpuModelSpec TpuModelSpec::from_model_config(const ModelConfig& mc, const HavenEngine*) {
    TpuModelSpec s;
    s.architecture = mc.architecture;
    return s;
}
HavenTpuGemma4Engine::HavenTpuGemma4Engine() = default;
HavenTpuGemma4Engine::~HavenTpuGemma4Engine() = default;
bool HavenTpuGemma4Engine::is_model_supported(const HavenEngine&, std::string& reason_out) {
    reason_out = "Cloud TPU PJRT disabled in this lean build (-DHAVEN_ENABLE_TPU=OFF).";
    return false;
}
int HavenTpuGemma4Engine::compute_w4_group_size(int) { return 128; }
void HavenTpuGemma4Engine::quantize_f32_to_s4_packed_group(const float*, int, int, int, int8_t*, float*) {}
void HavenTpuGemma4Engine::dequantize_s4_packed_group_to_f32(const int8_t*, const float*, int, int, int, float*) {}
std::string HavenTpuGemma4Engine::emit_stablehlo_mlir(const TpuModelSpec&, const TpuGemma4Config&, int, bool) { return ""; }
std::string HavenTpuGemma4Engine::emit_stablehlo_mlir(const ModelConfig&, const TpuGemma4Config&, int, bool) { return ""; }
bool HavenTpuGemma4Engine::initialize(HavenTpuPjrtBackend*, const HavenEngine&, const TpuGemma4Config&) {
    last_error_ = "Cloud TPU PJRT disabled in this lean build (-DHAVEN_ENABLE_TPU=OFF).";
    return false;
}
void HavenTpuGemma4Engine::shutdown() { initialized_ = false; }
bool HavenTpuGemma4Engine::reset_kv_cache() { return false; }
bool HavenTpuGemma4Engine::prefill_slot(int, const std::vector<uint32_t>&, int, std::vector<float>*, uint32_t*) { return false; }
bool HavenTpuGemma4Engine::prefill_prompt(const std::vector<uint32_t>&, int, std::vector<float>*, uint32_t*) { return false; }
bool HavenTpuGemma4Engine::decode_step_batched(const std::vector<uint32_t>&, const std::vector<int>&, std::vector<float>*, std::vector<uint32_t>&) { return false; }
bool HavenTpuGemma4Engine::decode_step(const std::vector<uint32_t>&, int, std::vector<float>*, std::vector<uint32_t>&) { return false; }

// ---- TpuContinuousBatchScheduler Stub ----
TpuContinuousBatchScheduler::TpuContinuousBatchScheduler() = default;
TpuContinuousBatchScheduler::~TpuContinuousBatchScheduler() = default;
bool TpuContinuousBatchScheduler::start(HavenTpuGemma4Engine*, const HavenTokenizer*, const PersonaSampler&) { return false; }
bool TpuContinuousBatchScheduler::start_with_backend(std::unique_ptr<ITpuBatchBackend>, const HavenTokenizer*, const PersonaSampler&) { return false; }
void TpuContinuousBatchScheduler::stop() { running_.store(false, std::memory_order_release); }
bool TpuContinuousBatchScheduler::generate_streaming(const TpuGenerationRequest&, const std::function<bool(uint32_t)>&, const std::function<bool()>&, std::string* err_out) {
    if (err_out) *err_out = "Cloud TPU scheduler disabled in this lean build (-DHAVEN_ENABLE_TPU=OFF).";
    return false;
}
TpuSchedulerStats TpuContinuousBatchScheduler::get_stats() const { return {}; }

} // namespace haven
