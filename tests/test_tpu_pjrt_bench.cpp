#include "haven/haven_engine.h"
#include "haven/haven_tpu_pjrt.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

using namespace haven;

static double compute_cosine(const float* a, const float* b, size_t n) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < n; ++i) {
        dot += static_cast<double>(a[i]) * static_cast<double>(b[i]);
        na  += static_cast<double>(a[i]) * static_cast<double>(a[i]);
        nb  += static_cast<double>(b[i]) * static_cast<double>(b[i]);
    }
    double denom = std::sqrt(na) * std::sqrt(nb);
    return (denom > 1e-12) ? (dot / denom) : 0.0;
}

static double compute_rel_l2(const float* ref, const float* test, size_t n) {
    double err2 = 0.0, ref2 = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double d = static_cast<double>(ref[i]) - static_cast<double>(test[i]);
        err2 += d * d;
        ref2 += static_cast<double>(ref[i]) * static_cast<double>(ref[i]);
    }
    return (ref2 > 1e-12) ? std::sqrt(err2 / ref2) : 0.0;
}

int main(int argc, char** argv) {
    std::cout << std::unitbuf;
    std::string model_path = "";
    std::string libtpu_override = "";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--libtpu" && i + 1 < argc) {
            libtpu_override = argv[++i];
        } else if (!arg.starts_with("-")) {
            model_path = arg;
        }
    }

    std::cout << "======================================================================\n";
    std::cout << "  HAVEN-CPP NATIVE C++20 PJRT (libtpu.so) CLOUD TPU SILICON BENCHMARK\n";
    std::cout << "======================================================================\n";
    std::cout << "Host Logical CPUs : " << std::thread::hardware_concurrency() << "\n";

    // 1. Initialize Native C++20 PJRT Backend (Zero Python)
    HavenTpuPjrtBackend tpu;
    if (!tpu.initialize(libtpu_override)) {
        std::cerr << "[TPU PJRT] Initialization failed: " << tpu.get_last_error() << "\n";
        return 1;
    }

    std::cout << "PJRT ABI Version  : " << tpu.get_pjrt_major() << "." << tpu.get_pjrt_minor() << "\n";
    std::cout << "Platform Name     : " << tpu.get_platform_name() << "\n";
    std::cout << "Platform Version  : " << tpu.get_platform_version() << "\n";
    std::cout << "Addressable Chips : " << tpu.get_device_count() << "\n";
    for (size_t i = 0; i < tpu.get_device_infos().size(); ++i) {
        const auto& d = tpu.get_device_infos()[i];
        std::cout << "  - Device[" << i << "] id=" << d.id
                  << " hw_id=" << d.local_hardware_id
                  << " kind=\"" << d.kind << "\""
                  << " HBM_limit=" << std::fixed << std::setprecision(2)
                  << (static_cast<double>(d.hbm_bytes_limit) / (1024.0 * 1024.0 * 1024.0)) << " GB\n";
    }

    // 2. Real MXU BF16 Systolic Array Saturation (1-Chip & All-Chips Parallel)
    std::cout << "\n----------------------------------------------------------------------\n";
    std::cout << "[STAGE 1] PURE C++20 STABLEHLO -> TPU MXU BF16 GEMM SATURATION\n";
    std::cout << "----------------------------------------------------------------------\n";

    struct Shape { int m, n, k, iters; const char* label; };
    const Shape shapes[] = {
        {4096,  4096,  4096,  20, "4096x4096   (Attention/Proj)"},
        {8192,  8192,  8192,  15, "8192x8192   (FFN / Large Proj)"},
        {16384, 16384, 16384, 10, "16384x16384 (Peak MXU Saturation)"}
    };

    const int n_devs = static_cast<int>(tpu.get_device_count());
    for (const auto& s : shapes) {
        auto r1 = tpu.benchmark_bf16_gemm(s.m, s.n, s.k, s.iters, 1);
        const float expected = static_cast<float>(s.k) * (1.0f / 64.0f) * (1.0f / 64.0f);
        if (r1.ok) {
            std::cout << "  [1-Chip ] " << s.label
                      << " | compile=" << std::fixed << std::setprecision(1) << r1.compile_ms << " ms"
                      << " | step=" << std::setprecision(2) << r1.avg_step_ms << " ms"
                      << " | " << std::setprecision(2) << r1.total_tflops << " TFLOPS"
                      << " | verify=" << r1.checksum << " (expect " << expected << ")\n";
        } else {
            std::cerr << "  [1-Chip ] " << s.label << " FAILED: " << r1.error << "\n";
        }

        if (n_devs > 1) {
            auto rN = tpu.benchmark_bf16_gemm(s.m, s.n, s.k, s.iters, n_devs);
            if (rN.ok) {
                std::cout << "  [" << n_devs << "-Chips] " << s.label
                          << " | compile=" << std::fixed << std::setprecision(1) << rN.compile_ms << " ms"
                          << " | step=" << std::setprecision(2) << rN.avg_step_ms << " ms"
                          << " | " << std::setprecision(2) << rN.total_tflops << " TFLOPS ("
                          << rN.per_device_tflops << "/chip)"
                          << " | verify=" << rN.checksum << "\n";
            } else {
                std::cerr << "  [" << n_devs << "-Chips] " << s.label << " FAILED: " << rN.error << "\n";
            }
        }
    }

    // 3. Real Gemma 4 GGUF Weight-Resident Head-to-Head: Host CPU AVX2 vs TPU MXU
    if (!model_path.empty() && std::filesystem::exists(model_path)) {
        std::cout << "\n----------------------------------------------------------------------\n";
        std::cout << "[STAGE 2] REAL GEMMA 4 GGUF HEAD-TO-HEAD: HOST CPU vs TPU HBM3 + MXU\n";
        std::cout << "----------------------------------------------------------------------\n";
        std::cout << "Model: " << model_path << "\n";

        HavenEngine engine;
        auto t_load0 = std::chrono::high_resolution_clock::now();
        if (!engine.load_model(model_path)) {
            std::cerr << "Failed to load GGUF model: " << model_path << "\n";
            return 1;
        }
        auto t_load1 = std::chrono::high_resolution_clock::now();
        const auto& cfg = engine.get_config();
        std::cout << "Loaded GGUF in " << std::fixed << std::setprecision(1)
                  << std::chrono::duration<double, std::milli>(t_load1 - t_load0).count() << " ms"
                  << " (layers=" << cfg.num_layers << ", dim=" << cfg.embedding_dim
                  << ", ffn=" << cfg.hidden_dim << ", vocab=" << cfg.vocab_size << ")\n";

        const auto& l0 = engine.get_layer_tensors(0);
        const TensorDesc* lm_head = engine.get_token_embd_tensor();

        // 3A. Fused Gemma 4 GeGLU FFN Block (Layer 0) — Decode (M=1) & Batched Prefill (M=128, M=512)
        if (l0.ffn_gate && l0.ffn_up && l0.ffn_down) {
            TpuMatrixHandle wg, wu, wd;
            auto u0 = std::chrono::high_resolution_clock::now();
            bool u_ok = tpu.upload_gguf_weight_matrix(*l0.ffn_gate, static_cast<int>(cfg.hidden_dim), static_cast<int>(cfg.embedding_dim), wg, 0)
                     && tpu.upload_gguf_weight_matrix(*l0.ffn_up,   static_cast<int>(cfg.hidden_dim), static_cast<int>(cfg.embedding_dim), wu, 0)
                     && tpu.upload_gguf_weight_matrix(*l0.ffn_down, static_cast<int>(cfg.embedding_dim), static_cast<int>(cfg.hidden_dim), wd, 0);
            auto u1 = std::chrono::high_resolution_clock::now();
            if (u_ok) {
                std::cout << "\n[2A] Layer 0 Fused GeGLU FFN (" << cfg.embedding_dim << " -> " << cfg.hidden_dim << " -> " << cfg.embedding_dim
                          << ") uploaded to TPU HBM3 (" << std::setprecision(1)
                          << static_cast<double>(wg.bytes + wu.bytes + wd.bytes) / (1024.0 * 1024.0) << " MB BF16 in "
                          << std::chrono::duration<double, std::milli>(u1 - u0).count() << " ms)\n";

                for (int batch_m : {1, 128, 512}) {
                    std::vector<float> x_in(static_cast<size_t>(batch_m) * cfg.embedding_dim);
                    for (size_t i = 0; i < x_in.size(); ++i) {
                        x_in[i] = std::sin(static_cast<float>(i) * 0.013f) * 0.25f;
                    }

                    // Host CPU reference (sequential per token, as HavenEngine::forward does today)
                    std::vector<float> y_cpu(static_cast<size_t>(batch_m) * cfg.embedding_dim, 0.0f);
                    std::vector<float> g_tmp(cfg.hidden_dim), u_tmp(cfg.hidden_dim);
                    const int cpu_iters = (batch_m == 1) ? 10 : 1;
                    auto c0 = std::chrono::high_resolution_clock::now();
                    for (int it = 0; it < cpu_iters; ++it) {
                        for (int m = 0; m < batch_m; ++m) {
                            const float* xm = x_in.data() + static_cast<size_t>(m) * cfg.embedding_dim;
                            float* ym = y_cpu.data() + static_cast<size_t>(m) * cfg.embedding_dim;
                            Avx2Math::gemv(g_tmp.data(), *l0.ffn_gate, xm, static_cast<int>(cfg.hidden_dim), static_cast<int>(cfg.embedding_dim));
                            Avx2Math::gemv(u_tmp.data(), *l0.ffn_up,   xm, static_cast<int>(cfg.hidden_dim), static_cast<int>(cfg.embedding_dim));
                            Avx2Math::gelu(g_tmp.data(), static_cast<int>(cfg.hidden_dim));
                            for (size_t k = 0; k < cfg.hidden_dim; ++k) g_tmp[k] *= u_tmp[k];
                            Avx2Math::gemv(ym, *l0.ffn_down, g_tmp.data(), static_cast<int>(cfg.embedding_dim), static_cast<int>(cfg.hidden_dim));
                        }
                    }
                    auto c1 = std::chrono::high_resolution_clock::now();
                    double cpu_ms = std::chrono::duration<double, std::milli>(c1 - c0).count() / cpu_iters;

                    // TPU MXU fused GeGLU (warmup + timed)
                    std::vector<float> y_tpu(static_cast<size_t>(batch_m) * cfg.embedding_dim, 0.0f);
                    if (!tpu.fused_geglu_ffn(x_in.data(), batch_m, wg, wu, wd, y_tpu.data())) {
                        std::cerr << "  Fused GeGLU M=" << batch_m << " FAILED: " << tpu.get_last_error() << "\n";
                        continue;
                    }
                    const int tpu_iters = 20;
                    auto p0 = std::chrono::high_resolution_clock::now();
                    for (int it = 0; it < tpu_iters; ++it) {
                        tpu.fused_geglu_ffn(x_in.data(), batch_m, wg, wu, wd, y_tpu.data());
                    }
                    auto p1 = std::chrono::high_resolution_clock::now();
                    double tpu_ms = std::chrono::duration<double, std::milli>(p1 - p0).count() / tpu_iters;

                    double cos_sim = compute_cosine(y_cpu.data(), y_tpu.data(), y_cpu.size());
                    double rel_l2  = compute_rel_l2(y_cpu.data(), y_tpu.data(), y_cpu.size());
                    std::cout << "  M=" << std::setw(3) << batch_m
                              << " | Host CPU: " << std::setw(8) << std::setprecision(2) << cpu_ms << " ms"
                              << " | TPU MXU (incl H2D+D2H): " << std::setw(6) << tpu_ms << " ms"
                              << " | Speedup: " << std::setprecision(1) << (cpu_ms / std::max(0.001, tpu_ms)) << "x"
                              << " | Cosine=" << std::setprecision(6) << cos_sim
                              << " | RelL2=" << std::setprecision(4) << rel_l2 << "\n";
                }
            }
            tpu.free_matrix(wg);
            tpu.free_matrix(wu);
            tpu.free_matrix(wd);
        }

        // 3B. Giant Gemma 4 LM Head Projection (262,144 x 2,560) — Host CPU vs TPU HBM3 + MXU
        if (lm_head) {
            TpuMatrixHandle w_lm;
            auto u0 = std::chrono::high_resolution_clock::now();
            bool u_ok = tpu.upload_gguf_weight_matrix(*lm_head, static_cast<int>(cfg.vocab_size), static_cast<int>(cfg.embedding_dim), w_lm, 0);
            auto u1 = std::chrono::high_resolution_clock::now();
            if (u_ok) {
                std::cout << "\n[2B] Gemma 4 LM Head (" << cfg.vocab_size << " x " << cfg.embedding_dim
                          << ") uploaded to TPU HBM3 (" << std::setprecision(1)
                          << static_cast<double>(w_lm.bytes) / (1024.0 * 1024.0) << " MB BF16 in "
                          << std::chrono::duration<double, std::milli>(u1 - u0).count() << " ms)\n";

                std::vector<float> x_in(cfg.embedding_dim);
                for (size_t i = 0; i < x_in.size(); ++i) x_in[i] = std::cos(static_cast<float>(i) * 0.017f) * 0.5f;

                std::vector<float> y_cpu(cfg.vocab_size, 0.0f);
                auto c0 = std::chrono::high_resolution_clock::now();
                for (int it = 0; it < 5; ++it) {
                    Avx2Math::gemv(y_cpu.data(), *lm_head, x_in.data(), static_cast<int>(cfg.vocab_size), static_cast<int>(cfg.embedding_dim));
                }
                auto c1 = std::chrono::high_resolution_clock::now();
                double cpu_ms = std::chrono::duration<double, std::milli>(c1 - c0).count() / 5.0;

                std::vector<float> y_tpu(cfg.vocab_size, 0.0f);
                tpu.matmul_resident_nt(x_in.data(), 1, w_lm, y_tpu.data()); // warmup
                auto p0 = std::chrono::high_resolution_clock::now();
                for (int it = 0; it < 20; ++it) {
                    tpu.matmul_resident_nt(x_in.data(), 1, w_lm, y_tpu.data());
                }
                auto p1 = std::chrono::high_resolution_clock::now();
                double tpu_ms = std::chrono::duration<double, std::milli>(p1 - p0).count() / 20.0;

                double cos_sim = compute_cosine(y_cpu.data(), y_tpu.data(), cfg.vocab_size);
                uint32_t cpu_top = static_cast<uint32_t>(std::max_element(y_cpu.begin(), y_cpu.end()) - y_cpu.begin());
                uint32_t tpu_top = static_cast<uint32_t>(std::max_element(y_tpu.begin(), y_tpu.end()) - y_tpu.begin());
                std::cout << "  M=  1 | Host CPU: " << std::setw(8) << std::setprecision(2) << cpu_ms << " ms"
                          << " | TPU MXU (incl H2D+D2H): " << std::setw(6) << tpu_ms << " ms"
                          << " | Speedup: " << std::setprecision(1) << (cpu_ms / std::max(0.001, tpu_ms)) << "x"
                          << " | Cosine=" << std::setprecision(6) << cos_sim
                          << " | Top1 Match=" << (cpu_top == tpu_top ? "YES" : "NO")
                          << " (" << cpu_top << " vs " << tpu_top << ")\n";
            }
            tpu.free_matrix(w_lm);
        }

        // 3C. Full 42-Layer Live Gemma 4 Generation on Host CPU (All Cores)
        std::cout << "\n----------------------------------------------------------------------\n";
        std::cout << "[STAGE 3] FULL 42-LAYER GEMMA 4 LIVE GENERATION (HOST CPU vs TPU-OFFLOAD)\n";
        std::cout << "----------------------------------------------------------------------\n";
        auto prompt_toks = engine.tokenize(
            engine.format_chat_prompt("Hey Aura! We just booted you on a Cloud TPU v6e-8 Trillium pod. How does it feel?"),
            true);
        std::cout << "Prompt tokens: " << prompt_toks.size() << "\n";

        auto g0 = std::chrono::high_resolution_clock::now();
        double ttft_ms = -1.0;
        int n_gen = 0;
        std::string reply;
        engine.generate(prompt_toks, 48, [&](uint32_t, const std::string& piece) {
            if (ttft_ms < 0.0) {
                ttft_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - g0).count();
            }
            reply += piece;
            n_gen++;
            return true;
        });
        auto g1 = std::chrono::high_resolution_clock::now();
        double total_ms = std::chrono::duration<double, std::milli>(g1 - g0).count();
        double dec_s = std::max(0.001, (total_ms - std::max(0.0, ttft_ms)) / 1000.0);
        std::cout << "[Host CPU (" << std::thread::hardware_concurrency() << " vCPUs)]"
                  << " TTFT=" << std::setprecision(1) << ttft_ms << " ms"
                  << " | Total=" << std::setprecision(2) << (total_ms / 1000.0) << " s"
                  << " | Generated=" << n_gen << " tokens (" << std::setprecision(2) << (n_gen / dec_s) << " tok/s decode)\n";
        std::cout << "Aura Reply (CPU): " << reply << "\n";

        // 3D. Enable Live TPU HBM3 + MXU Offload inside HavenEngine and run exact same prompt!
        auto up0 = std::chrono::high_resolution_clock::now();
        if (engine.enable_tpu_offload(&tpu)) {
            auto up1 = std::chrono::high_resolution_clock::now();
            std::cout << "\n[TPU Offload Active] Uploaded " << std::setprecision(2)
                      << (static_cast<double>(engine.get_tpu_resident_bytes()) / (1024.0 * 1024.0 * 1024.0))
                      << " GB BF16 weights (42-layer GeGLU FFN + 262k LM Head) to TPU HBM3 in "
                      << std::setprecision(1) << std::chrono::duration<double, std::milli>(up1 - up0).count() << " ms\n";

            // Warmup 1 token to compile per-device StableHLO FFN + LM Head graphs
            std::vector<float> warm_logits(cfg.vocab_size, 0.0f);
            engine.get_kv_cache().reset();
            engine.forward(prompt_toks[0], 0, warm_logits.data());

            auto tg0 = std::chrono::high_resolution_clock::now();
            double tpu_ttft_ms = -1.0;
            int tpu_n_gen = 0;
            std::string tpu_reply;
            engine.generate(prompt_toks, 48, [&](uint32_t, const std::string& piece) {
                if (tpu_ttft_ms < 0.0) {
                    tpu_ttft_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - tg0).count();
                }
                tpu_reply += piece;
                tpu_n_gen++;
                return true;
            });
            auto tg1 = std::chrono::high_resolution_clock::now();
            double tpu_total_ms = std::chrono::duration<double, std::milli>(tg1 - tg0).count();
            double tpu_dec_s = std::max(0.001, (tpu_total_ms - std::max(0.0, tpu_ttft_ms)) / 1000.0);
            std::cout << "[TPU HBM3+MXU Offload]"
                      << " TTFT=" << std::setprecision(1) << tpu_ttft_ms << " ms"
                      << " | Total=" << std::setprecision(2) << (tpu_total_ms / 1000.0) << " s"
                      << " | Generated=" << tpu_n_gen << " tokens (" << std::setprecision(2) << (tpu_n_gen / tpu_dec_s) << " tok/s decode)"
                      << " | Speedup vs Host CPU: " << std::setprecision(2) << (total_ms / std::max(1.0, tpu_total_ms)) << "x\n";
            std::cout << "Aura Reply (TPU): " << tpu_reply << "\n";
            engine.disable_tpu_offload();
        } else {
            std::cerr << "TPU offload enablement failed: " << tpu.get_last_error() << "\n";
        }
    }

    tpu.refresh_memory_stats();
    std::cout << "\n======================================================================\n";
    std::cout << "  HAVEN-CPP NATIVE PJRT CLOUD TPU BENCHMARK COMPLETE\n";
    std::cout << "======================================================================\n";
    return 0;
}
