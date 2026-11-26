#include "haven/haven_cuda.h"

namespace haven {

struct HavenCudaEngine::Impl {};

HavenCudaEngine::HavenCudaEngine() = default;
HavenCudaEngine::~HavenCudaEngine() = default;

bool HavenCudaEngine::is_cuda_compiled() { return false; }
int HavenCudaEngine::get_device_count() { return 0; }
std::string HavenCudaEngine::get_device_summary(int) { return "CUDA not compiled into this binary"; }

bool HavenCudaEngine::initialize(const HavenEngine&, const CudaEngineConfig&) {
    last_error_ = "Native CUDA backend not enabled in this build (rebuild with -DHAVEN_ENABLE_CUDA=ON).";
    return false;
}

void HavenCudaEngine::shutdown() {
    initialized_ = false;
}

bool HavenCudaEngine::reset_kv_cache() {
    last_error_ = "Native CUDA backend not initialized.";
    return false;
}

bool HavenCudaEngine::forward_decode(uint32_t, int, float*, uint32_t*) {
    last_error_ = "Native CUDA backend not initialized.";
    return false;
}

bool HavenCudaEngine::forward_prefill(const std::vector<uint32_t>&, int, float*, uint32_t*) {
    last_error_ = "Native CUDA backend not initialized.";
    return false;
}

} // namespace haven
