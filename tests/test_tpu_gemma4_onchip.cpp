#include "haven/haven_engine.h"
#include "haven/haven_tpu_pjrt.h"
#include "haven/haven_tpu_gemma4.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace haven;

static double cosine_sim(const float* a, const float* b, size_t n) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < n; ++i) {
        dot += static_cast<double>(a[i]) * static_cast<double>(b[i]);
        na  += static_cast<double>(a[i]) * static_cast<double>(a[i]);
        nb  += static_cast<double>(b[i]) * static_cast<double>(b[i]);
    }
    double denom = std::sqrt(na) * std::sqrt(nb);
    return (denom > 1e-12) ? (dot / denom) : 0.0;
}

int main(int argc, char** argv) {
    std::cout << std::unitbuf;
    std::string model_path = "";
    std::string libtpu_path = "";
    bool emit_only = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--libtpu" && i + 1 < argc) libtpu_path = argv[++i];
        else if (a == "--emit-only") emit_only = true;
        else if (!a.starts_with("-")) model_path = a;
    }

    if (model_path.empty()) {
        if (std::filesystem::exists("/tmp/haven-chat-v5.0.gguf")) model_path = "/tmp/haven-chat-v5.0.gguf";
        else model_path = "C:\\Users\\admin\\gemma4-turbo-family\\haven-chat-v5.0.gguf";
    }

    std::cout << "======================================================================\n";
    std::cout << "  HAVEN-CPP FULL ON-CHIP 42-LAYER GEMMA 4 TPU HBM3 SILICON BENCHMARK\n";
    std::cout << "======================================================================\n";

    HavenEngine host_engine;
    if (!host_engine.load_model(model_path)) {
        std::cerr << "Failed to load model: " << model_path << "\n";
        return 1;
    }
    const auto& mc = host_engine.get_config();
    std::cout << "Loaded Model: " << model_path << " (" << mc.num_layers << " layers, dim=" << mc.embedding_dim
              << ", ffn=" << mc.hidden_dim << ", vocab=" << mc.vocab_size << ")\n";

    // Offline StableHLO IR verification mode (runs locally on Windows or Linux with zero TPU!)
    if (emit_only) {
        for (int R : {1, 4}) {
            for (bool i8 : {false, true}) {
                TpuGemma4Config cfg;
                cfg.batch_size = 1;
                cfg.max_seq_len = 2048;
                cfg.prefill_chunk_size = 128;
                cfg.num_replicas = R;
                cfg.use_int8_weights = i8;
                cfg.return_full_logits = false;
                std::string dec_ir = HavenTpuGemma4Engine::emit_stablehlo_mlir(mc, cfg, 1);
                std::string pre_ir = HavenTpuGemma4Engine::emit_stablehlo_mlir(mc, cfg, 128);
                std::cout << "  [IR OK] R=" << R << " " << (i8 ? "INT8" : "BF16")
                          << " | decode_mlir=" << (dec_ir.size() / 1024) << " KB"
                          << " | prefill_128_mlir=" << (pre_ir.size() / 1024) << " KB\n";
            }
        }
        return 0;
    }

    HavenTpuPjrtBackend tpu;
    if (!tpu.initialize(libtpu_path)) {
        std::cerr << "[TPU PJRT] Init failed: " << tpu.get_last_error() << "\n";
        return 1;
    }
    const int n_devs = static_cast<int>(tpu.get_device_count());
    std::cout << "TPU Platform: " << tpu.get_platform_name() << " (" << n_devs << " addressable chips, PJRT "
              << tpu.get_pjrt_major() << "." << tpu.get_pjrt_minor() << ")\n";

    // Prepare exact reference prompt & compute CPU Ground-Truth Logits for parity check!
    const std::string prompt_str = host_engine.format_chat_prompt(
        "Hey Aura! All 42 of your transformer layers are now running fully on-chip inside Google TPU Trillium HBM3. How does pure silicon speed feel?");
    auto prompt_toks = host_engine.tokenize(prompt_str, true);
    std::cout << "Prompt tokens: " << prompt_toks.size() << "\n";

    std::vector<float> cpu_ref_logits(mc.vocab_size, 0.0f);
    host_engine.get_kv_cache().reset();
    auto cpu0 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < prompt_toks.size(); ++i) {
        bool is_last = (i + 1 == prompt_toks.size());
        host_engine.forward(prompt_toks[i], static_cast<int>(i), is_last ? cpu_ref_logits.data() : nullptr);
    }
    auto cpu1 = std::chrono::high_resolution_clock::now();
    uint32_t cpu_top1 = static_cast<uint32_t>(std::max_element(cpu_ref_logits.begin(), cpu_ref_logits.end()) - cpu_ref_logits.begin());
    std::cout << "[CPU Ground Truth] Prefill (" << prompt_toks.size() << " toks): "
              << std::fixed << std::setprecision(1)
              << std::chrono::duration<double, std::milli>(cpu1 - cpu0).count()
              << " ms | Top1 Token=" << cpu_top1 << " (\"" << host_engine.detokenize(cpu_top1) << "\")\n";

    // Helper to benchmark any (R, B, use_int8, return_logits) configuration end-to-end!
    auto run_config_bench = [&](const std::string& label, int R, int B, bool use_int8, bool full_logits, int gen_steps, int prefill_chunk) {
        if (R > n_devs) return;
        std::cout << "\n----------------------------------------------------------------------\n";
        std::cout << "[BENCH] " << label << " (R=" << R << ", B=" << B
                  << ", " << (use_int8 ? "W8A16-INT8" : "BF16")
                  << ", " << (full_logits ? "FullLogits" : "OnChipArgmax") << ")\n";
        std::cout << "----------------------------------------------------------------------\n";

        TpuGemma4Config cfg;
        cfg.batch_size = B;
        cfg.max_seq_len = 2048;
        cfg.prefill_chunk_size = prefill_chunk;
        cfg.num_replicas = R;
        cfg.use_int8_weights = use_int8;
        cfg.return_full_logits = full_logits;
        cfg.donate_kv_cache = true;

        HavenTpuGemma4Engine gemma_tpu;
        if (!gemma_tpu.initialize(&tpu, host_engine, cfg)) {
            std::cerr << "  INIT FAILED: " << gemma_tpu.get_last_error() << "\n";
            return;
        }
        const auto& m = gemma_tpu.get_metrics();
        std::cout << "  HBM3 Footprint : Weights=" << std::setprecision(2)
                  << (static_cast<double>(m.total_hbm_weight_bytes) / (1024.0 * 1024.0 * 1024.0)) << " GB"
                  << " | KV=" << (static_cast<double>(m.total_hbm_kv_bytes) / (1024.0 * 1024.0)) << " MB"
                  << " | Upload=" << std::setprecision(1) << m.weight_upload_ms << " ms"
                  << " | Compile(dec/pre)=" << m.decode_compile_ms << "/" << m.prefill_compile_ms << " ms\n";

        // Warmup 1 decode step (abort cleanly on error so device queue never desyncs)
        std::vector<uint32_t> warm_out;
        if (!gemma_tpu.decode_step(std::vector<uint32_t>(static_cast<size_t>(B), prompt_toks[0]), 0, nullptr, warm_out)) {
            std::cerr << "  WARMUP FAILED: " << gemma_tpu.get_last_error() << "\n";
            return;
        }
        gemma_tpu.reset_kv_cache();

        // Optional dedicated S=prefill_chunk MXU batched prefill measurement (e.g. 128 tokens in 1 single TPU launch!)
        double chunk_prefill_ms = 0.0;
        if (prefill_chunk > 1 && B == 1) {
            std::vector<uint32_t> long_prompt;
            long_prompt.reserve(static_cast<size_t>(prefill_chunk));
            while (long_prompt.size() < static_cast<size_t>(prefill_chunk)) {
                for (uint32_t tk : prompt_toks) {
                    if (long_prompt.size() >= static_cast<size_t>(prefill_chunk)) break;
                    long_prompt.push_back(tk);
                }
            }
            // Warmup once
            uint32_t dummy_tok = 0;
            gemma_tpu.reset_kv_cache();
            gemma_tpu.prefill_prompt(long_prompt, 0, nullptr, &dummy_tok);
            // Measure 5 runs
            constexpr int kPfIters = 5;
            double acc_ms = 0.0;
            for (int it = 0; it < kPfIters; ++it) {
                gemma_tpu.reset_kv_cache();
                auto p0 = std::chrono::high_resolution_clock::now();
                gemma_tpu.prefill_prompt(long_prompt, 0, nullptr, &dummy_tok);
                auto p1 = std::chrono::high_resolution_clock::now();
                acc_ms += std::chrono::duration<double, std::milli>(p1 - p0).count();
            }
            chunk_prefill_ms = acc_ms / kPfIters;
            gemma_tpu.reset_kv_cache();
        }

        // Timed Chat Prompt Prefill (exact CPU parity & Aura live generation)
        std::vector<float> tpu_logits;
        uint32_t first_tok = 0;
        auto pf0 = std::chrono::high_resolution_clock::now();
        bool pf_ok = gemma_tpu.prefill_prompt(prompt_toks, 0, full_logits ? &tpu_logits : nullptr, &first_tok);
        auto pf1 = std::chrono::high_resolution_clock::now();
        if (!pf_ok) {
            std::cerr << "  PREFILL FAILED: " << gemma_tpu.get_last_error() << "\n";
            return;
        }
        double prefill_ms = std::chrono::duration<double, std::milli>(pf1 - pf0).count();

        if (full_logits && tpu_logits.size() >= mc.vocab_size) {
            double cos_v = cosine_sim(cpu_ref_logits.data(), tpu_logits.data(), mc.vocab_size);
            std::cout << "  Parity vs CPU  : Cosine=" << std::setprecision(6) << cos_v
                      << " | Top1=" << first_tok << " (CPU=" << cpu_top1
                      << ", Match=" << (first_tok == cpu_top1 ? "YES" : "NO") << ")\n";
        } else {
            std::cout << "  First Token    : ID=" << first_tok << " (CPU=" << cpu_top1
                      << ", Match=" << (first_tok == cpu_top1 ? "YES" : "NO") << ")\n";
        }
        if (chunk_prefill_ms > 0.0) {
            std::cout << "  MXU Prefill S=" << prefill_chunk << " : " << std::setprecision(2) << chunk_prefill_ms
                      << " ms -> " << std::setprecision(0)
                      << (static_cast<double>(prefill_chunk) / (chunk_prefill_ms / 1000.0)) << " tok/s batched prefill\n";
        }

        // Timed Autoregressive On-Chip Decode Loop (gen_steps steps across B concurrent streams!)
        std::vector<uint32_t> cur_toks(static_cast<size_t>(B), first_tok);
        std::vector<uint32_t> next_toks;
        std::string generated_text = host_engine.detokenize(first_tok);
        int pos = static_cast<int>(prompt_toks.size());

        auto d0 = std::chrono::high_resolution_clock::now();
        int actual_steps = 0;
        double sum_h2d = 0.0, sum_exec = 0.0, sum_d2h = 0.0;
        for (int s = 0; s < gen_steps; ++s) {
            if (!gemma_tpu.decode_step(cur_toks, pos++, nullptr, next_toks)) {
                std::cerr << "  DECODE FAILED at step " << s << ": " << gemma_tpu.get_last_error() << "\n";
                break;
            }
            actual_steps++;
            sum_h2d  += gemma_tpu.get_metrics().last_h2d_ms;
            sum_exec += gemma_tpu.get_metrics().last_exec_ms;
            sum_d2h  += gemma_tpu.get_metrics().last_d2h_ms;
            cur_toks = next_toks;
            if (B == 1) {
                uint32_t nt = next_toks[0];
                if (host_engine.get_tokenizer().is_stop_token(nt) || nt == 1 || nt == 106 || nt == 212) {
                    // Continue running synthetic token for steady-state timing if stop hit early,
                    // but stop appending text to Aura's reply!
                } else if (actual_steps <= 64) {
                    std::string pc = host_engine.detokenize(nt);
                    if (pc.find("<turn|>") == std::string::npos && pc.find("<eos>") == std::string::npos) {
                        generated_text += pc;
                    }
                }
            }
        }
        auto d1 = std::chrono::high_resolution_clock::now();
        double dec_ms = std::chrono::duration<double, std::milli>(d1 - d0).count();
        double ms_per_step = dec_ms / std::max(1, actual_steps);
        double stream_tok_s = 1000.0 / std::max(0.001, ms_per_step);
        double agg_tok_s = stream_tok_s * static_cast<double>(B);

        std::cout << "  Prefill TTFT   : " << std::setprecision(2) << prefill_ms << " ms ("
                  << prompt_toks.size() << " tokens -> " << std::setprecision(0)
                  << (static_cast<double>(prompt_toks.size()) / (prefill_ms / 1000.0)) << " tok/s prefill)\n";
        std::cout << "  On-Chip Decode : " << std::setprecision(2) << ms_per_step << " ms/step"
                  << " (H2D=" << (sum_h2d / std::max(1, actual_steps))
                  << " / TPU=" << (sum_exec / std::max(1, actual_steps))
                  << " / D2H=" << (sum_d2h / std::max(1, actual_steps)) << " ms)"
                  << " | Per-Stream: " << std::setprecision(1) << stream_tok_s << " tok/s"
                  << " | Aggregate (B=" << B << "): " << agg_tok_s << " tok/s\n";
        if (B == 1) {
            std::cout << "  Aura Output    : " << generated_text << "\n";
        }
        gemma_tpu.shutdown();
    };

    // 1. Exact 262k Logit Parity Verification (1-Chip BF16 & 4-Chip SPMD BF16)
    run_config_bench("1A. Parity Check (1-Chip BF16 Full Logits)", 1, 1, false, true, 64, 128);
    if (n_devs >= 4) {
        run_config_bench("1B. Parity Check (4-Chip Optical ICI Tensor-Parallel BF16)", 4, 1, false, true, 64, 128);
    }

    // 2. Peak Interactive Single-Stream Decode (B=1, On-Chip Argmax — 4 bytes over PCIe!)
    run_config_bench("2A. Single-Stream Interactive (1-Chip BF16)", 1, 1, false, false, 128, 128);
    run_config_bench("2B. Single-Stream Interactive (1-Chip W8A16 INT8)", 1, 1, true, false, 128, 128);
    if (n_devs >= 4) {
        run_config_bench("2C. Single-Stream Interactive (4-Chip Tensor-Parallel BF16)", 4, 1, false, false, 128, 128);
        run_config_bench("2D. Single-Stream Interactive (4-Chip Tensor-Parallel W8A16 INT8)", 4, 1, true, false, 128, 128);
    }

    // 3. Cloud-Scale Continuous Batching (B=16, B=32, B=64, B=128 concurrent users on 4-Chip TP!)
    if (n_devs >= 4) {
        run_config_bench("3A. Cloud Continuous Batching (4-Chip TP, B=16, W8A16 INT8)", 4, 16, true, false, 64, 0);
        run_config_bench("3B. Cloud Continuous Batching (4-Chip TP, B=32, W8A16 INT8)", 4, 32, true, false, 64, 0);
        run_config_bench("3C. Cloud Continuous Batching (4-Chip TP, B=64, W8A16 INT8)", 4, 64, true, false, 64, 0);
        run_config_bench("3D. Cloud Continuous Batching (4-Chip TP, B=64, BF16)",       4, 64, false, false, 64, 0);
        run_config_bench("3E. Cloud Continuous Batching (4-Chip TP, B=128, W8A16 INT8)", 4, 128, true, false, 64, 0);
    } else {
        run_config_bench("3A. Single-Chip Continuous Batching (1-Chip, B=32, W8A16 INT8)", 1, 32, true, false, 64, 0);
    }

    std::cout << "\n======================================================================\n";
    std::cout << "  FULL ON-CHIP 42-LAYER GEMMA 4 TPU BENCHMARK COMPLETE\n";
    std::cout << "======================================================================\n";
    return 0;
}
