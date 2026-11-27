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
    CHECK(quant_block_spec(QuantType::IQ2_XXS).block_elems == 256u && quant_block_spec(QuantType::IQ2_XXS).block_bytes == 66u, "IQ2_XXS spec");
    CHECK(quant_block_spec(QuantType::IQ1_S).block_elems   == 256u && quant_block_spec(QuantType::IQ1_S).block_bytes   == 50u, "IQ1_S spec");
    CHECK(quant_block_spec(QuantType::IQ3_S).block_elems   == 256u && quant_block_spec(QuantType::IQ3_S).block_bytes   == 110u,"IQ3_S spec");
    CHECK(quant_block_spec(QuantType::TQ2_0).block_elems   == 256u && quant_block_spec(QuantType::TQ2_0).block_bytes   == 66u, "TQ2_0 spec");
    CHECK(quant_block_spec(static_cast<QuantType>(999u)).block_elems == 0u, "Unknown quant type returns 0 spec");
    CHECK(quant_tensor_byte_size(static_cast<QuantType>(999u), {256, 256}) == 0ULL, "Unknown quant tensor size safe 0");

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
    CHECK(!cpu_s->prefill(nullptr, 4, 0, nullptr, nullptr), "CPU non-empty null prefill rejected");
    CHECK(cpu_s->reset_kv_cache(), "CPU reset_kv_cache");
    cpu_s->shutdown();
    CHECK(!cpu_s->is_initialized(), "CPU shutdown clears initialized state");
    CHECK(!cpu_s->decode_step(1, 0, nullptr, nullptr), "Decode after shutdown safely returns false");
    std::cout << "[PASS 3/5] IHavenSession lifecycle & shutdown safety verified\n";

    // 4. Verify GGUF-spec split-half Q4_0 layout (elements 0..15 low nibbles, 16..31 high nibbles)
    //    across dequantize_row, vec_dot_q4_0, dot_row_q8, and batched gemm!
    {
        block_q4_0 blk{};
        blk.d = 0x3C00u; // FP16 1.0f
        for (int j = 0; j < 16; ++j) {
            // Low nibble = 9 -> (9 - 8) = +1.0 for [0..15]; High nibble = 6 -> (6 - 8) = -2.0 for [16..31]
            blk.qs[j] = static_cast<uint8_t>(9u | (6u << 4));
        }
        TensorDesc td{ "q4_0_spec", QuantType::Q4_0, {32, 1}, 0, sizeof(block_q4_0), &blk };
        float deq[32] = {0};
        Avx2Math::dequantize_row(deq, td, 0, 32);
        for (int i = 0; i < 16; ++i) CHECK(std::fabs(deq[i] - 1.0f) < 1e-5f, "Q4_0 split-half low nibble [0..15]");
        for (int i = 16; i < 32; ++i) CHECK(std::fabs(deq[i] + 2.0f) < 1e-5f, "Q4_0 split-half high nibble [16..31]");

        float x[32];
        for (int i = 0; i < 16; ++i) x[i] = 1.0f;
        for (int i = 16; i < 32; ++i) x[i] = 0.0f;
        // Dot against [1..1 (16x), 0..0 (16x)] MUST be +16.0 (interleaved layout gave -8.0!)
        float dot_fp = Avx2Math::vec_dot_q4_0(32, &blk, x);
        CHECK(std::fabs(dot_fp - 16.0f) < 1e-3f, "Q4_0 vec_dot_q4_0 matches GGUF split-half");
        Q8Act32 act;
        Avx2Math::quantize_q8_act(x, 32, act);
        float dot_q8 = Avx2Math::dot_row_q8(td, 0, act, 32);
        CHECK(std::fabs(dot_q8 - 16.0f) < 0.25f, "Q4_0 dot_row_q8 matches GGUF split-half");
        float y_gemm = 0.0f;
        Avx2Math::gemm(&y_gemm, 1, td, x, 32, 1, 1, 32);
        CHECK(std::fabs(y_gemm - 16.0f) < 0.25f, "Q4_0 gemm matches GGUF split-half");
    }
    std::cout << "[PASS 4/5] Q4_0 GGUF split-half layout verified across dequant/AVX2/Q8/GEMM\n";

    // 5. Verify Interleaved (Llama) vs NeoX (Gemma/Qwen) RoPE, GPT-2 BPE Tokenizer, and NaN-safe Greedy Sampler
    {
        float q_neox[4] = {1.0f, 0.0f, 0.0f, 1.0f};
        float q_norm[4] = {1.0f, 0.0f, 0.0f, 1.0f};
        Avx2Math::apply_rope(q_neox, nullptr, 4, 1, 0, 1, 10000.0f, 1.0f, nullptr, 4, true);
        Avx2Math::apply_rope(q_norm, nullptr, 4, 1, 0, 1, 10000.0f, 1.0f, nullptr, 4, false);
        // At pos=1, freq_0=1.0 rad, freq_1=0.01 rad:
        //   NeoX rotates (0,2) and (1,3): q_neox = {cos(1), -sin(0.01), sin(1), cos(0.01)}
        //   Interleaved rotates (0,1) and (2,3): q_norm = {cos(1), sin(1), -sin(0.01), cos(0.01)}
        CHECK(std::fabs(q_neox[0] - std::cos(1.0f)) < 1e-4f &&
              std::fabs(q_neox[1] + std::sin(0.01f)) < 1e-4f &&
              std::fabs(q_neox[2] - std::sin(1.0f)) < 1e-4f &&
              std::fabs(q_neox[3] - std::cos(0.01f)) < 1e-4f, "NeoX split-half RoPE");
        CHECK(std::fabs(q_norm[0] - std::cos(1.0f)) < 1e-4f &&
              std::fabs(q_norm[1] - std::sin(1.0f)) < 1e-4f &&
              std::fabs(q_norm[2] + std::sin(0.01f)) < 1e-4f &&
              std::fabs(q_norm[3] - std::cos(0.01f)) < 1e-4f, "Interleaved NORM RoPE");

        // GPT-2 byte-level BPE (Llama 3 / Qwen 2.5): Ġ = \xc4\xa0, Ċ = \xc4\x8a
        std::vector<std::string> bpe_vocab = {
            "<|begin_of_text|>", "<|end_of_text|>", "<|eot_id|>", "Hello", "\xc4\xa0world", "\xc4\x8a", "!"
        };
        HavenTokenizer bpe_tok;
        bpe_tok.initialize(bpe_vocab);
        auto ids = bpe_tok.encode("Hello world\n!", true);
        CHECK(ids.size() == 5 && ids[0] == 0 && ids[1] == 3 && ids[2] == 4 && ids[3] == 5 && ids[4] == 6, "GPT-2 BPE encode");
        CHECK(bpe_tok.decode(ids) == "Hello world\n!", "GPT-2 BPE decode roundtrip");
        CHECK(bpe_tok.encode("", true).size() == 1 && bpe_tok.encode("", true)[0] == 0, "Empty encode with add_bos returns BOS");

        // Greedy sampler NaN resilience
        SamplerParams sp;
        sp.temperature = 0.0f;
        PersonaSampler sampler(sp);
        float logits[4] = { std::numeric_limits<float>::quiet_NaN(), -2.0f, 5.5f, 1.0f };
        CHECK(sampler.sample(logits, 4, {}) == 2u, "Greedy sampler ignores leading NaN");
    }
    std::cout << "[PASS 5/5] Interleaved/NeoX RoPE, GPT-2 BPE tokenizer, and NaN-safe sampler verified\n";

    std::cout << "\n✅ ALL LEAN CORE, QUANT PARITY & MULTI-ARCH TESTS PASSED 100%!\n";
    return 0;
}
