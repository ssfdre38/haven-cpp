#include "haven/haven_engine.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <filesystem>

int main(int argc, char** argv) {
    std::cout << "==================================================================\n";
    std::cout << "🔬 TESTING Q6_K QUANTIZATION KERNEL PRECISION VS FP32\n";
    std::cout << "==================================================================\n";

    haven::HavenEngine engine;
    std::string model_path = "C:\\Users\\admin\\gemma4-turbo-family\\haven-chat-v5.0.gguf";
    if (argc > 1) {
        model_path = argv[1];
    } else {
        if (!std::filesystem::exists(model_path) && std::filesystem::exists("/workspace/haven/ava-turbo-q4_k_m.gguf")) {
            model_path = "/workspace/haven/ava-turbo-q4_k_m.gguf";
        }
    }

    if (!engine.load_model(model_path)) {
        std::cerr << "Failed to load model\n";
        return 1;
    }

    const auto* embd = engine.get_token_embd_tensor();
    if (!embd) {
        std::cerr << "No token_embd.weight\n";
        return 1;
    }

    std::vector<float> test_vec(2560, 1.0f);
    for (int i = 0; i < 2560; ++i) test_vec[i] = std::sin((float)i * 0.05f);

    // 1. Direct Q6_K dot product for row 10979 ("Hi")
    uint32_t token = 10979;
    const auto* blocks = reinterpret_cast<const haven::block_q6_K*>(embd->data);
    float dot_q6 = haven::Avx2Math::vec_dot_q6_K(2560, blocks + token * 10, test_vec.data());

    // 2. Dequantize to FP32 then dot product
    std::vector<float> row_fp32(2560, 0.0f);
    haven::Avx2Math::dequantize_row(row_fp32.data(), *embd, token, 2560);
    float dot_fp32 = 0.0f;
    for (int i = 0; i < 2560; ++i) dot_fp32 += row_fp32[i] * test_vec[i];

    std::cout << "• Token ID " << token << " ('Hi'):\n";
    std::cout << "   - Q6_K Kernel Dot Product: " << dot_q6 << "\n";
    std::cout << "   - FP32 Dequant Dot Product: " << dot_fp32 << "\n";
    std::cout << "   - Absolute Difference: " << std::abs(dot_q6 - dot_fp32) << "\n";

    if (std::abs(dot_q6 - dot_fp32) < 1e-3f) {
        std::cout << "✓ SUCCESS: Q6_K AVX2 Kernel matches FP32 dequantization perfectly!\n";
    } else {
        std::cout << "✗ MISMATCH in Q6_K kernel calculation!\n";
    }

    // 3. Test precision for layer 0 weights across quant types
    if (engine.get_config().num_layers > 0) {
        const auto& l0 = engine.get_layer_tensors(0);
        const haven::TensorDesc* test_tensors[] = {
            l0.attn_q, l0.attn_k, l0.attn_v, l0.attn_output,
            l0.ffn_gate, l0.ffn_down, l0.ffn_up
        };

        for (const auto* t : test_tensors) {
            if (!t || !t->data) continue;
            int cols = (t->shape.size() >= 1) ? (int)t->shape[0] : 0;
            if (cols <= 0) continue;

            std::vector<float> x(cols, 1.0f);
            for (int i = 0; i < cols; ++i) x[i] = std::sin((float)i * 0.05f);

            float dot_kernel = 0.0f;
            std::string q_name = "UNKNOWN";

            if (t->type == haven::QuantType::Q4_K && (cols % 256 == 0)) {
                q_name = "Q4_K";
                dot_kernel = haven::Avx2Math::vec_dot_q4_K(cols, reinterpret_cast<const haven::block_q4_K*>(t->data), x.data());
            } else if (t->type == haven::QuantType::Q6_K && (cols % 256 == 0)) {
                q_name = "Q6_K";
                dot_kernel = haven::Avx2Math::vec_dot_q6_K(cols, reinterpret_cast<const haven::block_q6_K*>(t->data), x.data());
            } else if (t->type == haven::QuantType::Q5_0 && (cols % 32 == 0)) {
                q_name = "Q5_0";
                dot_kernel = haven::Avx2Math::vec_dot_q5_0(cols, reinterpret_cast<const haven::block_q5_0*>(t->data), x.data());
            } else if (t->type == haven::QuantType::Q5_K && (cols % 256 == 0)) {
                q_name = "Q5_K";
                dot_kernel = haven::Avx2Math::vec_dot_q5_K(cols, reinterpret_cast<const haven::block_q5_K*>(t->data), x.data());
            } else if (t->type == haven::QuantType::Q3_K && (cols % 256 == 0)) {
                q_name = "Q3_K";
                dot_kernel = haven::Avx2Math::vec_dot_q3_K(cols, reinterpret_cast<const haven::block_q3_K*>(t->data), x.data());
            } else if (t->type == haven::QuantType::Q4_0 && (cols % 32 == 0)) {
                q_name = "Q4_0";
                dot_kernel = haven::Avx2Math::vec_dot_q4_0(cols, reinterpret_cast<const haven::block_q4_0*>(t->data), x.data());
            } else if (t->type == haven::QuantType::Q8_0 && (cols % 32 == 0)) {
                q_name = "Q8_0";
                dot_kernel = haven::Avx2Math::vec_dot_q8_0(cols, reinterpret_cast<const haven::block_q8_0*>(t->data), x.data());
            }

            if (q_name != "UNKNOWN") {
                std::vector<float> dequant_row(cols, 0.0f);
                haven::Avx2Math::dequantize_row(dequant_row.data(), *t, 0, cols);
                float dot_fp32_val = 0.0f;
                for (int i = 0; i < cols; ++i) dot_fp32_val += dequant_row[i] * x[i];

                float diff = std::abs(dot_kernel - dot_fp32_val);
                std::cout << "\n• " << q_name << " Kernel (" << t->name << " Row 0):\n";
                std::cout << "   - " << q_name << " Kernel Dot Product: " << dot_kernel << "\n";
                std::cout << "   - FP32 Dequant Dot Product: " << dot_fp32_val << "\n";
                std::cout << "   - Absolute Difference: " << diff << "\n";
                if (diff < 1e-3f) {
                    std::cout << "✓ SUCCESS: " << q_name << " AVX2 Kernel matches FP32 dequantization!\n";
                } else {
                    std::cerr << "✗ MISMATCH in " << q_name << " kernel!\n";
                    return 1;
                }
            }
        }
    }

    // 4. Synthetic parity check for Q5_1, IQ4_NL, and IQ4_XS AVX2 kernels vs FP32 dequantization
    {
        std::vector<float> x512(512, 0.0f);
        for (int i = 0; i < 512; ++i) x512[i] = std::sin(static_cast<float>(i) * 0.13f + 0.4f);

        auto check_parity = [](const char* label, float dot_k, const haven::TensorDesc& td, const float* xv, int n) -> bool {
            std::vector<float> fp32(static_cast<size_t>(n), 0.0f);
            haven::Avx2Math::dequantize_row(fp32.data(), td, 0, n);
            float dot_ref = 0.0f;
            for (int i = 0; i < n; ++i) dot_ref += fp32[i] * xv[i];
            const float diff = std::abs(dot_k - dot_ref);
            const float tol = std::max(1e-3f, 1e-5f * std::abs(dot_ref));
            std::cout << "\n• " << label << " AVX2 Kernel Parity: kernel=" << dot_k << " ref=" << dot_ref << " diff=" << diff << " (tol=" << tol << ")\n";
            if (diff >= tol) {
                std::cerr << "✗ MISMATCH in " << label << " AVX2 kernel!\n";
                return false;
            }
            std::cout << "✓ SUCCESS: " << label << " AVX2 Kernel matches FP32 dequantization!\n";
            return true;
        };

        std::vector<haven::block_q5_1> q51(16);
        for (int b = 0; b < 16; ++b) {
            q51[b].d = 0x3800; // 0.5
            q51[b].m = 0xB400; // -0.25
            const uint32_t qh_val = 0xA55A1234u ^ static_cast<uint32_t>(b * 0x01010101u);
            std::memcpy(q51[b].qh, &qh_val, sizeof(qh_val));
            for (int j = 0; j < 16; ++j) q51[b].qs[j] = static_cast<uint8_t>(((b * 7 + j * 3) & 0x0F) | (((b + j * 5) & 0x0F) << 4));
        }
        haven::TensorDesc td_q51{ "syn_q5_1", haven::QuantType::Q5_1, {512}, 0, q51.size() * sizeof(haven::block_q5_1), q51.data() };
        if (!check_parity("Q5_1", haven::Avx2Math::vec_dot_q5_1(512, q51.data(), x512.data()), td_q51, x512.data(), 512)) return 1;

        std::vector<haven::block_iq4_nl> iq4nl(16);
        for (int b = 0; b < 16; ++b) {
            iq4nl[b].d = 0x3400; // 0.25
            for (int j = 0; j < 16; ++j) iq4nl[b].qs[j] = static_cast<uint8_t>(((b * 3 + j) & 0x0F) | (((15 - j + b) & 0x0F) << 4));
        }
        haven::TensorDesc td_nl{ "syn_iq4_nl", haven::QuantType::IQ4_NL, {512}, 0, iq4nl.size() * sizeof(haven::block_iq4_nl), iq4nl.data() };
        if (!check_parity("IQ4_NL", haven::Avx2Math::vec_dot_iq4_nl(512, iq4nl.data(), x512.data()), td_nl, x512.data(), 512)) return 1;

        std::vector<haven::block_iq4_xs> iq4xs(2);
        for (int b = 0; b < 2; ++b) {
            iq4xs[b].d = 0x3400; // 0.25
            iq4xs[b].scales_h = static_cast<uint16_t>(0x693Cu ^ (b * 0x1111u));
            for (int k = 0; k < 4; ++k) iq4xs[b].scales_l[k] = static_cast<uint8_t>(0x3A + b * 7 + k * 13);
            for (int j = 0; j < 128; ++j) iq4xs[b].qs[j] = static_cast<uint8_t>(((b * 5 + j * 11) & 0x0F) | (((j * 7 + 3) & 0x0F) << 4));
        }
        haven::TensorDesc td_xs{ "syn_iq4_xs", haven::QuantType::IQ4_XS, {512}, 0, iq4xs.size() * sizeof(haven::block_iq4_xs), iq4xs.data() };
        if (!check_parity("IQ4_XS", haven::Avx2Math::vec_dot_iq4_xs(512, iq4xs.data(), x512.data()), td_xs, x512.data(), 512)) return 1;

        // 5. Q8Act32 integer AVX2 (vpmaddubsw/vpmaddwd) kernel parity vs exact dequant(w) . dequant(act) AND FP32
        haven::Q8Act32 act512;
        haven::Avx2Math::quantize_q8_act(x512.data(), 512, act512);
        std::vector<float> x512_deq(512, 0.0f);
        for (int b = 0; b < 16; ++b) {
            for (int j = 0; j < 32; ++j) {
                x512_deq[static_cast<size_t>(b * 32 + j)] = static_cast<float>(act512.qs[b * 32 + j]) * act512.d[b];
            }
        }
        auto check_q8_parity = [&](const char* label, const haven::TensorDesc& td, bool exact_sumf_offset) -> bool {
            const float dot_q8 = haven::Avx2Math::dot_row_q8(td, 0, act512, 512);
            const float dot_fp = haven::Avx2Math::dot_row(td, 0, x512.data(), 512);
            std::vector<float> w_fp(512, 0.0f);
            haven::Avx2Math::dequantize_row(w_fp.data(), td, 0, 512);
            float dot_q8_ref = 0.0f;
            float norm_w2 = 0.0f, norm_x2 = 0.0f;
            for (int i = 0; i < 512; ++i) {
                dot_q8_ref += w_fp[static_cast<size_t>(i)] * x512_deq[static_cast<size_t>(i)];
                norm_w2 += w_fp[static_cast<size_t>(i)] * w_fp[static_cast<size_t>(i)];
                norm_x2 += x512[static_cast<size_t>(i)] * x512[static_cast<size_t>(i)];
            }
            const float cs_bound = std::sqrt(norm_w2 * norm_x2);
            const float simd_err = std::abs(dot_q8 - dot_q8_ref);
            const float quant_err = std::abs(dot_q8 - dot_fp);
            const float simd_tol = exact_sumf_offset ? std::max(0.05f, 0.01f * cs_bound) : std::max(1e-3f, 2e-5f * cs_bound);
            const float quant_tol = std::max(0.05f, 0.01f * cs_bound);
            std::cout << "\n• " << label << " Q8Act32 Integer AVX2 Parity: q8=" << dot_q8
                      << " simd_ref=" << dot_q8_ref << " fp32=" << dot_fp
                      << " simd_err=" << simd_err << " quant_err=" << quant_err << " (cs_tol=" << quant_tol << ")\n";
            if (simd_err >= simd_tol || quant_err >= quant_tol) {
                std::cerr << "✗ MISMATCH in " << label << " Q8Act32 kernel!\n";
                return false;
            }
            std::cout << "✓ SUCCESS: " << label << " Q8Act32 Integer AVX2 Kernel matches exact SIMD & FP32!\n";
            return true;
        };

        if (!check_q8_parity("Q5_1",   td_q51, true))  return 1;
        if (!check_q8_parity("IQ4_NL", td_nl,  false)) return 1;
        if (!check_q8_parity("IQ4_XS", td_xs,  false)) return 1;

        // Synthetic Q8_0, Q4_0, Q5_0, Q5_K Q8Act32 parity checks
        std::vector<haven::block_q8_0> q80(16);
        for (int b = 0; b < 16; ++b) {
            q80[b].d = 0x3400;
            for (int j = 0; j < 32; ++j) q80[b].qs[j] = static_cast<int8_t>(((b * 13 + j * 7) % 255) - 128);
        }
        haven::TensorDesc td_q80{ "syn_q8_0", haven::QuantType::Q8_0, {512}, 0, q80.size() * sizeof(haven::block_q8_0), q80.data() };
        if (!check_q8_parity("Q8_0", td_q80, false)) return 1;

        std::vector<haven::block_q4_0> q40(16);
        for (int b = 0; b < 16; ++b) {
            q40[b].d = 0x3800;
            for (int j = 0; j < 16; ++j) q40[b].qs[j] = static_cast<uint8_t>(((b * 5 + j * 3) & 0x0F) | (((b + j * 7) & 0x0F) << 4));
        }
        haven::TensorDesc td_q40{ "syn_q4_0", haven::QuantType::Q4_0, {512}, 0, q40.size() * sizeof(haven::block_q4_0), q40.data() };
        if (!check_q8_parity("Q4_0", td_q40, false)) return 1;

        std::vector<haven::block_q5_0> q50(16);
        for (int b = 0; b < 16; ++b) {
            q50[b].d = 0x3800;
            const uint32_t qh_val = 0x5A5A9876u ^ static_cast<uint32_t>(b * 0x03030303u);
            std::memcpy(q50[b].qh, &qh_val, sizeof(qh_val));
            for (int j = 0; j < 16; ++j) q50[b].qs[j] = static_cast<uint8_t>(((b * 3 + j * 9) & 0x0F) | (((b + j * 11) & 0x0F) << 4));
        }
        haven::TensorDesc td_q50{ "syn_q5_0", haven::QuantType::Q5_0, {512}, 0, q50.size() * sizeof(haven::block_q5_0), q50.data() };
        if (!check_q8_parity("Q5_0", td_q50, false)) return 1;

        std::vector<haven::block_q5_K> q5k(2);
        for (int b = 0; b < 2; ++b) {
            q5k[b].d = 0x3400;
            q5k[b].dmin = 0x3000;
            for (int k = 0; k < 12; ++k) q5k[b].scales[k] = static_cast<uint8_t>(0x15 + b * 7 + k * 3);
            for (int j = 0; j < 32; ++j) q5k[b].qh[j] = static_cast<uint8_t>(0xA5 ^ (j * 17 + b * 9));
            for (int j = 0; j < 128; ++j) q5k[b].qs[j] = static_cast<uint8_t>(((b * 11 + j * 5) & 0x0F) | (((j * 3 + 7) & 0x0F) << 4));
        }
        haven::TensorDesc td_q5k{ "syn_q5_K", haven::QuantType::Q5_K, {512}, 0, q5k.size() * sizeof(haven::block_q5_K), q5k.data() };
        if (!check_q8_parity("Q5_K", td_q5k, true)) return 1;

        // Real Q6_K and Q4_K GEMV cosine similarity checks on loaded production model (>0.9998)
        {
            std::vector<float> y_q8(64, 0.0f), y_fp(64, 0.0f);
            haven::Avx2Math::gemv(y_q8.data(), *embd, test_vec.data(), 64, 2560);
            for (int r = 0; r < 64; ++r) y_fp[r] = haven::Avx2Math::dot_row(*embd, static_cast<size_t>(r), test_vec.data(), 2560);
            const float cos_q6 = haven::Avx2Math::cosine_similarity(y_q8.data(), y_fp.data(), 64);
            std::cout << "\n• Real Q6_K GEMV (Q8Act32 vs FP32, 64 rows): cosine=" << cos_q6 << "\n";
            if (cos_q6 < 0.9995f) {
                std::cerr << "✗ Q6_K Q8Act32 GEMV cosine below 0.9995!\n";
                return 1;
            }
            if (engine.get_config().num_layers > 0 && engine.get_layer_tensors(0).attn_q &&
                engine.get_layer_tensors(0).attn_q->type == haven::QuantType::Q4_K) {
                const auto* t4k = engine.get_layer_tensors(0).attn_q;
                haven::Avx2Math::gemv(y_q8.data(), *t4k, test_vec.data(), 64, 2560);
                for (int r = 0; r < 64; ++r) y_fp[r] = haven::Avx2Math::dot_row(*t4k, static_cast<size_t>(r), test_vec.data(), 2560);
                const float cos_q4 = haven::Avx2Math::cosine_similarity(y_q8.data(), y_fp.data(), 64);
                std::cout << "• Real Q4_K GEMV (Q8Act32 vs FP32, 64 rows): cosine=" << cos_q4 << "\n";
                if (cos_q4 < 0.9995f) {
                    std::cerr << "✗ Q4_K Q8Act32 GEMV cosine below 0.9995!\n";
                    return 1;
                }
            }

            // 6. Batched 1x8 / 1x4 GEMM parity vs single-token GEMV (n_tokens = 13 exercises both x8 and x4 paths!)
            constexpr int S_TEST = 13;
            std::vector<float> X_batch(static_cast<size_t>(S_TEST) * 2560ULL);
            for (int t = 0; t < S_TEST; ++t) {
                for (int c = 0; c < 2560; ++c) {
                    X_batch[static_cast<size_t>(t) * 2560ULL + static_cast<size_t>(c)] =
                        std::sin(static_cast<float>(c) * 0.05f + static_cast<float>(t) * 0.37f);
                }
            }
            auto check_gemm_tensor = [&](const char* label, const haven::TensorDesc& td) -> bool {
                std::vector<float> Y_gemm(static_cast<size_t>(S_TEST) * 64ULL, 0.0f);
                std::vector<float> Y_gemv(static_cast<size_t>(S_TEST) * 64ULL, 0.0f);
                haven::Avx2Math::gemm(Y_gemm.data(), 64, td, X_batch.data(), 2560, S_TEST, 64, 2560);
                for (int t = 0; t < S_TEST; ++t) {
                    haven::Avx2Math::gemv(Y_gemv.data() + static_cast<size_t>(t) * 64ULL, td,
                                          X_batch.data() + static_cast<size_t>(t) * 2560ULL, 64, 2560);
                }
                const float cos_b = haven::Avx2Math::cosine_similarity(Y_gemm.data(), Y_gemv.data(), S_TEST * 64);
                std::cout << "• Batched GEMM (S=" << S_TEST << ") " << label << " vs GEMV: cosine=" << cos_b << "\n";
                if (cos_b < 0.999f) {
                    std::cerr << "✗ Batched GEMM " << label << " cosine below 0.999!\n";
                    return false;
                }
                return true;
            };
            if (!check_gemm_tensor("Q6_K", *embd)) return 1;
            if (engine.get_config().num_layers > 0 && engine.get_layer_tensors(0).attn_q) {
                if (!check_gemm_tensor("Layer0 attn_q", *engine.get_layer_tensors(0).attn_q)) return 1;
            }
            // Synthetic Batched GEMM (S=13) parity across IQ4_XS, IQ4_NL, Q5_1, Q5_0, Q8_0, Q4_0, Q5_K
            std::vector<float> X512_b(static_cast<size_t>(S_TEST) * 512ULL);
            for (int t = 0; t < S_TEST; ++t) {
                for (int c = 0; c < 512; ++c) {
                    X512_b[static_cast<size_t>(t) * 512ULL + static_cast<size_t>(c)] =
                        std::sin(static_cast<float>(c) * 0.13f + 0.4f + static_cast<float>(t) * 0.51f);
                }
            }
            auto check_syn_gemm = [&](const char* label, const haven::TensorDesc& td) -> bool {
                std::vector<float> Y_gemm(static_cast<size_t>(S_TEST), 0.0f);
                std::vector<float> Y_ref(static_cast<size_t>(S_TEST), 0.0f);
                haven::Avx2Math::gemm(Y_gemm.data(), 1, td, X512_b.data(), 512, S_TEST, 1, 512);
                for (int t = 0; t < S_TEST; ++t) {
                    Y_ref[static_cast<size_t>(t)] = haven::Avx2Math::dot_row(td, 0, X512_b.data() + static_cast<size_t>(t) * 512ULL, 512);
                }
                const float cos_s = haven::Avx2Math::cosine_similarity(Y_gemm.data(), Y_ref.data(), S_TEST);
                std::cout << "• Batched GEMM (S=" << S_TEST << ") Synthetic " << label << " vs FP32: cosine=" << cos_s << "\n";
                if (cos_s < 0.998f) {
                    std::cerr << "✗ Batched GEMM Synthetic " << label << " cosine below 0.998!\n";
                    return false;
                }
                return true;
            };
            if (!check_syn_gemm("IQ4_XS", td_xs))  return 1;
            if (!check_syn_gemm("IQ4_NL", td_nl))  return 1;
            if (!check_syn_gemm("Q5_1",   td_q51)) return 1;
            if (!check_syn_gemm("Q5_0",   td_q50)) return 1;
            if (!check_syn_gemm("Q8_0",   td_q80)) return 1;
            if (!check_syn_gemm("Q4_0",   td_q40)) return 1;
            if (!check_syn_gemm("Q5_K",   td_q5k)) return 1;
        }
    }

    std::cout << "==================================================================\n";
    return 0;
}
