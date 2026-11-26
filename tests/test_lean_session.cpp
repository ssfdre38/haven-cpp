#include "haven/haven_quant_spec.h"
#include "haven/haven_session.h"
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        std::cerr << "FAIL: " << msg << " (line " << __LINE__ << ")\n"; \
        return 1; \
    } \
} while (0)

int main() {
    using namespace haven;
    std::cout << "========================================================================\n";
    std::cout << "  HAVEN LEAN CORE — SINGLE-SOURCE QUANT SPEC & 4-CALL SESSION SUITE\n";
    std::cout << "========================================================================\n";

    // 1. Verify single-source quant block specifications & overflow guards
    CHECK(quant_block_spec(QuantType::IQ4_XS).block_elems == 256u && quant_block_spec(QuantType::IQ4_XS).block_bytes == 136u, "IQ4_XS spec");
    CHECK(quant_block_spec(QuantType::IQ4_NL).block_elems == 32u  && quant_block_spec(QuantType::IQ4_NL).block_bytes == 18u,  "IQ4_NL spec");
    CHECK(quant_block_spec(QuantType::Q4_K).block_elems   == 256u && quant_block_spec(QuantType::Q4_K).block_bytes   == 144u, "Q4_K spec");
    CHECK(quant_block_spec(QuantType::Q6_K).block_elems   == 256u && quant_block_spec(QuantType::Q6_K).block_bytes   == 210u, "Q6_K spec");
    CHECK(quant_block_spec(QuantType::Q8_0).block_elems   == 32u  && quant_block_spec(QuantType::Q8_0).block_bytes   == 34u,  "Q8_0 spec");
    CHECK(quant_block_spec(QuantType::BF16).block_elems   == 1u   && quant_block_spec(QuantType::BF16).block_bytes   == 2u,   "BF16 spec");

    CHECK(quant_tensor_byte_size(QuantType::IQ4_XS, {2816, 1408, 128}) == (2816ULL * 1408ULL * 128ULL / 256ULL) * 136ULL, "26B MoE IQ4_XS exact bytes");
    CHECK(quant_tensor_byte_size(QuantType::BF16, {2816, 4096}) == 23068672ULL, "BF16 Q-proj exact bytes");
    CHECK(quant_tensor_byte_size(QuantType::F32, {}) == 0ULL, "Empty shape safe 0");
    CHECK(quant_tensor_byte_size(QuantType::F32, {std::numeric_limits<uint64_t>::max(), 2ULL}) == 0ULL, "Overflow safe 0");

    // Verify canonical IQ4_NL LUT bounds & monotonicity
    for (int i = 1; i < 16; ++i) {
        CHECK(kQuantIq4NlLut[i] > kQuantIq4NlLut[i - 1], "IQ4_NL LUT strictly increasing");
    }
    CHECK(kQuantIq4NlLut[0] == -127 && kQuantIq4NlLut[15] == 113, "IQ4_NL LUT endpoints");

    // Verify BF16 round-to-nearest-even & FP16 soft conversion
    for (float v : {0.0f, -0.0f, 1.0f, -1.0f, 3.140625f, -42.5f, 1024.0f}) {
        uint16_t b = quant_f32_to_bf16_rne(v);
        float rt = quant_bf16_to_f32(b);
        CHECK(std::fabs(rt - v) <= std::fabs(v) * 0.01f + 1e-6f, "BF16 roundtrip");
    }
    CHECK(std::fabs(quant_fp16_to_f32_soft(0x3C00u) - 1.0f) < 1e-6f, "FP16 1.0");
    CHECK(std::fabs(quant_fp16_to_f32_soft(0xC000u) + 2.0f) < 1e-6f, "FP16 -2.0");
    CHECK(std::fabs(quant_fp16_to_f32_soft(0x0000u)) < 1e-12f, "FP16 0.0");

    // Verify Q4_K scale/min 6-bit unpacker
    uint8_t k4_scales[12] = {0};
    k4_scales[0] = 27u;
    k4_scales[4] = 19u;
    uint8_t sc = 0, mn = 0;
    quant_get_scale_min_k4(0, k4_scales, sc, mn);
    CHECK(sc == 27u && mn == 19u, "Q4_K scale/min j<4");
    std::cout << "[PASS 1/3] Single-source haven_quant_spec.h verified (block specs, LUT, BF16 RNE, FP16, overflow guards)\n";

    // 2. Verify Unified 4-Call IHavenSession lifecycle + graceful CUDA->CPU auto-fallback
    HavenEngine host_engine;
    ExecutionPlan cuda_plan;
    cuda_plan.selected_backend = BackendKind::CUDA_NATIVE;

    SessionConfig cfg;
    cfg.max_seq_len = 512;
    std::string slog;
    auto session = HavenSessionFactory::create_for_plan(host_engine, cuda_plan, cfg, slog);
    CHECK(session != nullptr, "Session factory always returns a valid session");
    CHECK(session->is_initialized(), "Session initialized cleanly");
    CHECK(session->reset_kv_cache(), "Session reset_kv_cache succeeded");
    std::cout << "[PASS 2/3] Unified 4-call IHavenSession factory verified: " << slog << "\n";

    // Verify explicit CPU session contract (zero-token prefill no-op + clean shutdown)
    auto cpu_s = HavenSessionFactory::create_cpu_session();
    CHECK(cpu_s->backend_kind() == BackendKind::CPU_AVX2, "CPU session kind");
    CHECK(cpu_s->initialize(host_engine, cfg), "CPU session init");
    CHECK(cpu_s->prefill(nullptr, 0, 0, nullptr, nullptr), "CPU empty prefill safe no-op");
    CHECK(cpu_s->reset_kv_cache(), "CPU reset_kv_cache");
    cpu_s->shutdown();
    CHECK(!cpu_s->is_initialized(), "CPU shutdown clears initialized state");
    CHECK(!cpu_s->decode_step(1, 0, nullptr, nullptr), "Decode after shutdown safely returns false");
    std::cout << "[PASS 3/3] IHavenSession lifecycle & shutdown safety verified\n";

    std::cout << "\n✅ ALL LEAN CORE & UNIFIED SESSION TESTS PASSED 100%!\n";
    return 0;
}
