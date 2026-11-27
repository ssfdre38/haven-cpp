#pragma once

#include "haven_engine.h"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace haven {

struct CudaEngineConfig {
    int device_id = 0;
    int max_seq_len = 2048;              // Static KV cache capacity per KV-owning layer
    int max_prefill_chunk = 512;         // Max tokens per cuBLAS prefill / grouped MoE chunk
    bool use_cuda_graph = true;          // Single-shot CUDA Graph capture for zero-launch-overhead B=1 decode
    bool exact_fp32_activations = false; // Zero-quantization FP32-act GEMV (weights stay 4-bit in VRAM, zero Q8_1 error!)
    bool fp32_kv_cache = false;          // Store K/V cache in full FP32 instead of FP16 (zero KV mantissa truncation!)
    bool fp32_prefill = false;           // Dequantize W -> F32 + strict FP32 cuBLAS SGEMM prefill (zero FP16 truncation!)
    bool host_resident_pleias = false;   // Keep Pleias per_layer_token_embd on host RAM (~2.1 GiB VRAM saved on E4B!)
    bool verbose = false;
};

struct CudaEngineMetrics {
    std::string device_name;
    int sm_major = 0;
    int sm_minor = 0;
    size_t total_vram_bytes = 0;
    size_t weight_vram_bytes = 0;
    size_t kv_vram_bytes = 0;
    size_t scratch_vram_bytes = 0;
    double upload_ms = 0.0;
    double last_prefill_ms = 0.0;
    double last_decode_ms = 0.0;
};

// Native Zero-Compile-Time CUDA Backend for haven.cpp (Gemma 4/4.5 Turbo & Stock, Llama 3/4, Qwen 2.5/3, Mistral, Gemma 2).
// Uploads raw GGUF quantized blocks directly to GPU VRAM (IQ4_XS, IQ4_NL, Q6_K, Q5_K, Q5_1, Q5_0, Q4_K, Q4_0, Q8_0, Q3_K,
// BF16, F16, F32), executes warp-specialized bandwidth-bound GEMV kernels for B=1 decode, cuBLAS FP16 Tensor Core GEMM
// for multi-token prefill, shared-memory Block-FWHT for Gemma 4.5 Turbo, Hybrid SWA+Global GQA attention with FP16 KV
// cache, Pleias per-layer injection, and zero-host-sync 128-Expert Top-8 Sparse MoE!
class HavenCudaEngine {
public:
    HavenCudaEngine();
    ~HavenCudaEngine();

    HavenCudaEngine(const HavenCudaEngine&) = delete;
    HavenCudaEngine& operator=(const HavenCudaEngine&) = delete;

    // Runtime & compile-time capability probes (safe on non-CUDA builds)
    static bool is_cuda_compiled();
    static int get_device_count();
    static std::string get_device_summary(int device_id = 0);

    bool initialize(const HavenEngine& host_engine, const CudaEngineConfig& cfg = {});
    void shutdown();

    bool is_initialized() const { return initialized_; }
    const CudaEngineConfig& get_config() const { return cfg_; }
    const CudaEngineMetrics& get_metrics() const { return metrics_; }
    std::string get_last_error() const { return last_error_; }

    bool reset_kv_cache();

    // Single-token forward step at sequence position `pos`.
    // - If out_logits != nullptr: copies [vocab_size] F32 logits to host and optionally writes argmax into out_next_token.
    // - If out_logits == nullptr: runs on-GPU argmax and transfers only 4 bytes (uint32_t) D2H!
    bool forward_decode(uint32_t token, int pos, float* out_logits, uint32_t* out_next_token);

    // Multi-token prompt prefill starting at `start_pos` (uses cuBLAS FP16 Tensor Cores for S >= 8).
    bool forward_prefill(const std::vector<uint32_t>& tokens,
                         int start_pos,
                         float* out_last_logits,
                         uint32_t* out_next_token);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    bool initialized_ = false;
    CudaEngineConfig cfg_{};
    CudaEngineMetrics metrics_{};
    std::string last_error_;
};

} // namespace haven
