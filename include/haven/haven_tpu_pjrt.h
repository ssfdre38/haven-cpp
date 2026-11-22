#pragma once

#include "haven_types.h"
#include <cstdint>
#include <cstddef>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

struct PJRT_Api;
struct PJRT_Client;
struct PJRT_Device;
struct PJRT_Buffer;
struct PJRT_LoadedExecutable;
struct PJRT_Event;

namespace haven {

struct TpuDeviceInfo {
    int id = -1;
    int local_hardware_id = -1;
    std::string kind;
    std::string debug_string;
    int64_t hbm_bytes_in_use = 0;
    int64_t hbm_bytes_limit = 0;
};

struct TpuGemmBenchResult {
    bool ok = false;
    std::string error;
    int M = 0;
    int N = 0;
    int K = 0;
    int num_devices = 0;
    int iterations = 0;
    double compile_ms = 0.0;
    double avg_step_ms = 0.0;
    double total_tflops = 0.0;
    double per_device_tflops = 0.0;
    float checksum = 0.0f;
};

struct TpuMatrixHandle {
    PJRT_Buffer* buffer = nullptr;
    int64_t rows = 0;
    int64_t cols = 0;
    int device_index = 0;
    size_t bytes = 0;
};

// Native C++20 OpenXLA PJRT C API backend dynamically loading Google's libtpu.so.
// Zero Python runtime dependency; compiles textual StableHLO MLIR directly onto
// Cloud TPU MXU systolic arrays and manages persistent HBM3 weight buffers.
class HavenTpuPjrtBackend {
public:
    HavenTpuPjrtBackend();
    ~HavenTpuPjrtBackend();

    HavenTpuPjrtBackend(const HavenTpuPjrtBackend&) = delete;
    HavenTpuPjrtBackend& operator=(const HavenTpuPjrtBackend&) = delete;

    // Dynamically loads libtpu.so via dlopen and creates a PJRT TPU client.
    // Returns false cleanly on non-TPU / Windows hosts.
    bool initialize(const std::string& libtpu_path_override = "");
    void shutdown();

    bool is_initialized() const { return initialized_; }
    std::string get_last_error() const { return last_error_; }
    std::string get_platform_name() const { return platform_name_; }
    std::string get_platform_version() const { return platform_version_; }
    int get_pjrt_major() const { return pjrt_major_; }
    int get_pjrt_minor() const { return pjrt_minor_; }
    size_t get_device_count() const { return devices_.size(); }
    const std::vector<TpuDeviceInfo>& get_device_infos() const { return device_infos_; }

    // IEEE-754 FP32 <-> BF16 round-to-nearest-even converters (strict-aliasing safe)
    static uint16_t fp32_to_bf16(float f);
    static float bf16_to_fp32(uint16_t b);
    static void convert_f32_to_bf16(const float* src, uint16_t* dst, size_t n);
    static void convert_bf16_to_f32(const uint16_t* src, float* dst, size_t n);

    // HBM3 Tensor Transfer & Lifecycle (2D convenience + general N-D)
    PJRT_Buffer* upload_bf16_2d(const uint16_t* host_data, int64_t rows, int64_t cols, int device_index = 0);
    PJRT_Buffer* upload_f32_2d(const float* host_data, int64_t rows, int64_t cols, int device_index = 0);
    PJRT_Buffer* upload_bf16_nd(const uint16_t* host_data, const std::vector<int64_t>& dims, int device_index = 0);
    PJRT_Buffer* upload_f32_nd(const float* host_data, const std::vector<int64_t>& dims, int device_index = 0);
    PJRT_Buffer* upload_s8_nd(const int8_t* host_data, const std::vector<int64_t>& dims, int device_index = 0);
    PJRT_Buffer* upload_i32_nd(const int32_t* host_data, const std::vector<int64_t>& dims, int device_index = 0);
    bool download_f32(PJRT_Buffer* buf, float* host_out, size_t num_elems);
    bool download_bf16(PJRT_Buffer* buf, uint16_t* host_out, size_t num_elems);
    bool download_i32(PJRT_Buffer* buf, int32_t* host_out, size_t num_elems);
    void destroy_buffer(PJRT_Buffer* buf);

    // Compile textual StableHLO MLIR module (portable single-device, ordinal-bound, or multi-replica SPMD)
    PJRT_LoadedExecutable* compile_stablehlo(const std::string& mlir_code, int device_ordinal = -1);
    PJRT_LoadedExecutable* compile_stablehlo_replicated(const std::string& mlir_code, int num_replicas);
    std::vector<int> get_executable_device_indices(PJRT_LoadedExecutable* exec);
    void destroy_executable(PJRT_LoadedExecutable* exec);

    // Execute a compiled StableHLO executable on a single addressable TPU device
    bool execute_1d(PJRT_LoadedExecutable* exec,
                    int device_index,
                    const std::vector<PJRT_Buffer*>& inputs,
                    size_t num_outputs,
                    std::vector<PJRT_Buffer*>& outputs);

    // Execute a multi-replica SPMD StableHLO executable across R TensorCores simultaneously
    bool execute_replicated(PJRT_LoadedExecutable* exec,
                            int num_replicas,
                            const std::vector<std::vector<PJRT_Buffer*>>& inputs_per_replica,
                            size_t num_outputs,
                            std::vector<std::vector<PJRT_Buffer*>>& outputs_per_replica);

    // Upload a quantized GGUF weight matrix [rows, cols] into persistent TPU HBM3 as BF16
    bool upload_gguf_weight_matrix(const TensorDesc& weight, int rows, int cols,
                                   TpuMatrixHandle& out_handle, int device_index = 0);
    void free_matrix(TpuMatrixHandle& handle);

    // Execute persistent-weight Y = X * W^T on TPU MXU:
    //   X: [batch_m, K] (f32 host -> bf16 device), W: [N, K] resident in TPU HBM3 (bf16),
    //   Y: [batch_m, N] (f32 accumulated on MXU -> f32 host)
    bool matmul_resident_nt(const float* x_host, int batch_m,
                            const TpuMatrixHandle& w_resident,
                            float* y_host);

    // Fused Gemma 4 GeGLU FFN block on TPU HBM3 (1 dispatch instead of 3):
    //   gate = X * W_gate^T, up = X * W_up^T, h = gelu(gate) * up, out = h * W_down^T
    bool fused_geglu_ffn(const float* x_host, int batch_m,
                         const TpuMatrixHandle& w_gate,
                         const TpuMatrixHandle& w_up,
                         const TpuMatrixHandle& w_down,
                         float* y_host);

    // Direct MXU BF16 GEMM saturation benchmark (1 device or parallel across all N devices)
    TpuGemmBenchResult benchmark_bf16_gemm(int M, int N, int K, int iterations = 15, int num_devices = 1);

    // Refresh live HBM memory usage stats for all addressable devices
    void refresh_memory_stats();

private:
    bool initialized_ = false;
#ifndef _WIN32
    void* dl_handle_ = nullptr;
    const PJRT_Api* api_ = nullptr;
    PJRT_Client* client_ = nullptr;
#endif
    std::vector<PJRT_Device*> devices_;
    std::vector<TpuDeviceInfo> device_infos_;
    std::string platform_name_;
    std::string platform_version_;
    std::string last_error_;
    int pjrt_major_ = 0;
    int pjrt_minor_ = 0;

    std::mutex cache_mutex_;
    std::unordered_map<std::string, PJRT_LoadedExecutable*> exec_cache_;

    std::string consume_error(void* pjrt_error_ptr);
    bool await_and_destroy_event(PJRT_Event* ev);
    PJRT_LoadedExecutable* get_or_compile_matmul_nt(int batch_m, int N, int K, int device_index);
    PJRT_LoadedExecutable* get_or_compile_geglu_ffn(int batch_m, int hidden_dim, int emb_dim, int device_index);
};

} // namespace haven
