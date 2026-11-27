#include "haven/haven_session.h"
#include "haven/haven_cuda.h"
#include <chrono>
#include <iomanip>
#include <limits>
#include <sstream>
#include <vector>

namespace haven {

namespace {

static uint32_t host_argmax(const float* logits, size_t vocab_size) {
    if (!logits || vocab_size == 0) return 0u;
    uint32_t best_idx = 0u;
    float best_val = -std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < vocab_size; ++i) {
        if (logits[i] > best_val) {
            best_val = logits[i];
            best_idx = static_cast<uint32_t>(i);
        }
    }
    return best_idx;
}

class Avx2CpuSession final : public IHavenSession {
public:
    BackendKind backend_kind() const override { return BackendKind::CPU_AVX2; }
    std::string backend_name() const override { return "Native AVX2+FMA CPU"; }
    bool is_initialized() const override { return initialized_; }

    bool initialize(HavenEngine& host, const SessionConfig& cfg) override {
        host_ = &host;
        host_->set_exact_math(cfg.exact_math);
        initialized_ = true;
        metrics_ = {};
        metrics_.backend = BackendKind::CPU_AVX2;
        metrics_.device_summary = cfg.exact_math
            ? "Bare-Metal AVX2/FMA C++20 CPU (Exact FP32-Act + FP64 Norms/RoPE/Attn)"
            : "Bare-Metal AVX2/FMA C++20 CPU";
        last_error_.clear();
        return true;
    }

    bool prefill(const uint32_t* tokens,
                 size_t count,
                 int start_pos,
                 float* out_last_logits,
                 uint32_t* out_next_token) override {
        if (!initialized_ || !host_) {
            last_error_ = "Avx2CpuSession not initialized";
            return false;
        }
        if (count == 0) return true;
        if (!tokens) {
            last_error_ = "Avx2CpuSession::prefill received null tokens pointer with count > 0";
            return false;
        }

        auto t0 = std::chrono::high_resolution_clock::now();
        const size_t vocab = host_->get_config().vocab_size;
        float* target_logits = out_last_logits;
        if (!target_logits && out_next_token && vocab > 0) {
            scratch_logits_.resize(vocab);
            target_logits = scratch_logits_.data();
        }
        host_->forward_batch(tokens, static_cast<int>(count), start_pos, target_logits);
        if (out_next_token && target_logits && vocab > 0) {
            *out_next_token = host_argmax(target_logits, vocab);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        metrics_.last_prefill_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        return true;
    }

    bool decode_step(uint32_t token,
                     int pos,
                     float* out_logits,
                     uint32_t* out_next_token) override {
        if (!initialized_ || !host_) {
            last_error_ = "Avx2CpuSession not initialized";
            return false;
        }
        auto t0 = std::chrono::high_resolution_clock::now();
        const size_t vocab = host_->get_config().vocab_size;
        float* target_logits = out_logits;
        if (!target_logits && out_next_token && vocab > 0) {
            scratch_logits_.resize(vocab);
            target_logits = scratch_logits_.data();
        }
        host_->forward(token, pos, target_logits);
        if (out_next_token && target_logits && vocab > 0) {
            *out_next_token = host_argmax(target_logits, vocab);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        metrics_.last_decode_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        return true;
    }

    bool reset_kv_cache() override {
        if (!initialized_ || !host_) return false;
        host_->get_kv_cache().reset();
        return true;
    }

    void shutdown() override {
        initialized_ = false;
        host_ = nullptr;
        scratch_logits_.clear();
    }

    SessionMetrics get_metrics() const override { return metrics_; }
    std::string get_last_error() const override { return last_error_; }

private:
    HavenEngine* host_ = nullptr;
    bool initialized_ = false;
    SessionMetrics metrics_{};
    std::string last_error_;
    std::vector<float> scratch_logits_;
};

class CudaNativeSession final : public IHavenSession {
public:
    BackendKind backend_kind() const override { return BackendKind::CUDA_NATIVE; }
    std::string backend_name() const override { return "Native Zero-Compile CUDA V2"; }
    bool is_initialized() const override { return cuda_.is_initialized(); }

    bool initialize(HavenEngine& host, const SessionConfig& cfg) override {
        CudaEngineConfig ccfg;
        ccfg.device_id = cfg.device_id;
        ccfg.max_seq_len = cfg.max_seq_len;
        ccfg.max_prefill_chunk = cfg.max_prefill_chunk;
        ccfg.use_cuda_graph = cfg.use_cuda_graph;
        ccfg.exact_fp32_activations = cfg.exact_math;
        ccfg.fp32_kv_cache = cfg.fp32_kv_cache || cfg.exact_math;
        ccfg.fp32_prefill = cfg.fp32_prefill || cfg.exact_math;
        ccfg.host_resident_pleias = cfg.host_resident_pleias;
        ccfg.verbose = cfg.verbose;

        if (!cuda_.initialize(host, ccfg)) {
            last_error_ = cuda_.get_last_error();
            return false;
        }
        const auto& cm = cuda_.get_metrics();
        metrics_ = {};
        metrics_.backend = BackendKind::CUDA_NATIVE;
        std::ostringstream ss;
        ss << cm.device_name << " (sm_" << cm.sm_major << cm.sm_minor << ", "
           << std::fixed << std::setprecision(2)
           << (static_cast<double>(cm.weight_vram_bytes) / (1024.0 * 1024.0 * 1024.0))
           << " GiB VRAM, " << std::setprecision(0) << cm.upload_ms << " ms upload)";
        metrics_.device_summary = ss.str();
        metrics_.weight_bytes = cm.weight_vram_bytes;
        metrics_.kv_bytes = cm.kv_vram_bytes;
        metrics_.init_ms = cm.upload_ms;
        last_error_.clear();
        return true;
    }

    bool prefill(const uint32_t* tokens,
                 size_t count,
                 int start_pos,
                 float* out_last_logits,
                 uint32_t* out_next_token) override {
        if (!cuda_.is_initialized()) {
            last_error_ = "CudaNativeSession not initialized";
            return false;
        }
        if (count == 0) return true;
        if (!tokens) {
            last_error_ = "CudaNativeSession::prefill received null tokens pointer with count > 0";
            return false;
        }
        std::vector<uint32_t> vec(tokens, tokens + count);
        bool ok = cuda_.forward_prefill(vec, start_pos, out_last_logits, out_next_token);
        if (!ok) last_error_ = cuda_.get_last_error();
        metrics_.last_prefill_ms = cuda_.get_metrics().last_prefill_ms;
        return ok;
    }

    bool decode_step(uint32_t token,
                     int pos,
                     float* out_logits,
                     uint32_t* out_next_token) override {
        if (!cuda_.is_initialized()) {
            last_error_ = "CudaNativeSession not initialized";
            return false;
        }
        bool ok = cuda_.forward_decode(token, pos, out_logits, out_next_token);
        if (!ok) last_error_ = cuda_.get_last_error();
        metrics_.last_decode_ms = cuda_.get_metrics().last_decode_ms;
        return ok;
    }

    bool reset_kv_cache() override {
        bool ok = cuda_.reset_kv_cache();
        if (!ok) last_error_ = cuda_.get_last_error();
        return ok;
    }

    void shutdown() override {
        cuda_.shutdown();
        metrics_ = {};
        last_error_.clear();
    }

    SessionMetrics get_metrics() const override { return metrics_; }
    std::string get_last_error() const override { return last_error_; }

private:
    HavenCudaEngine cuda_;
    SessionMetrics metrics_{};
    std::string last_error_;
};

} // namespace

std::unique_ptr<IHavenSession> HavenSessionFactory::create_cpu_session() {
    return std::make_unique<Avx2CpuSession>();
}

std::unique_ptr<IHavenSession> HavenSessionFactory::create_cuda_session() {
    return std::make_unique<CudaNativeSession>();
}

std::unique_ptr<IHavenSession> HavenSessionFactory::create_for_plan(HavenEngine& host,
                                                                    const ExecutionPlan& plan,
                                                                    const SessionConfig& cfg,
                                                                    std::string& out_log) {
    std::ostringstream log;
    if (plan.selected_backend == BackendKind::CUDA_NATIVE) {
        if (HavenCudaEngine::is_cuda_compiled() && HavenCudaEngine::get_device_count() > 0) {
            auto s = create_cuda_session();
            if (s->initialize(host, cfg)) {
                log << "[HavenSession] Activated " << s->backend_name() << " (" << s->get_metrics().device_summary << ")";
                out_log = log.str();
                return s;
            }
            log << "[HavenSession] CUDA init failed (" << s->get_last_error() << ") -> falling back to AVX2 CPU. ";
        } else {
            log << "[HavenSession] CUDA requested but unavailable -> falling back to AVX2 CPU. ";
        }
    } else if (plan.selected_backend == BackendKind::TPU_PJRT) {
        log << "[HavenSession] TPU_PJRT session dispatch uses dedicated HavenTpuGemma4Engine -> falling back to AVX2 CPU session. ";
    } else if (plan.selected_backend == BackendKind::MLX) {
        log << "[HavenSession] Apple MLX session execution not yet wired -> falling back to AVX2 CPU session. ";
    }

    auto cpu = create_cpu_session();
    cpu->initialize(host, cfg);
    log << "[HavenSession] Activated " << cpu->backend_name();
    out_log = log.str();
    return cpu;
}

} // namespace haven
