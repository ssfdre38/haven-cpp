#include "haven/haven_cuda.h"
#include "haven/haven_engine.h"
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
    std::string model_path;
    int max_seq_len = 2048;
    int gen_steps = 128;
    int pp_len = 512;
    int repeats = 3;
    int device_id = 0;
    bool skip_cpu_parity = false;
    bool exact_mode = false;
    bool fp32_act = false;
    bool fp32_kv = false;
    bool fp32_prefill = false;
    bool host_pleias = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--device" && i + 1 < argc) device_id = std::stoi(argv[++i]);
        else if (a == "--max-seq" && i + 1 < argc) max_seq_len = std::stoi(argv[++i]);
        else if (a == "--steps" && i + 1 < argc) gen_steps = std::stoi(argv[++i]);
        else if (a == "--pp" && i + 1 < argc) pp_len = std::stoi(argv[++i]);
        else if (a == "--repeats" && i + 1 < argc) repeats = std::max(1, std::stoi(argv[++i]));
        else if (a == "--skip-parity") skip_cpu_parity = true;
        else if (a == "--exact") { exact_mode = true; fp32_act = true; fp32_kv = true; fp32_prefill = true; }
        else if (a == "--fp32-act") fp32_act = true;
        else if (a == "--fp32-kv") fp32_kv = true;
        else if (a == "--fp32-prefill") fp32_prefill = true;
        else if (a == "--host-pleias") host_pleias = true;
        else if (a == "--probe-only") {
            std::cout << "CUDA Compiled: " << (HavenCudaEngine::is_cuda_compiled() ? "YES" : "NO")
                      << " | Devices: " << HavenCudaEngine::get_device_count()
                      << " | Summary: " << HavenCudaEngine::get_device_summary(device_id) << "\n";
            return 0;
        } else if (!a.starts_with("-")) {
            model_path = a;
        }
    }

    std::cout << "======================================================================\n";
    std::cout << "  HAVEN-CPP NATIVE CUDA KERNEL SILICON VERIFICATION & BENCHMARK\n";
    std::cout << "======================================================================\n";
    std::cout << "CUDA Backend: " << (HavenCudaEngine::is_cuda_compiled() ? "COMPILED" : "STUB (rebuild with -DHAVEN_ENABLE_CUDA=ON)")
              << " | Devices: " << HavenCudaEngine::get_device_count()
              << " | " << HavenCudaEngine::get_device_summary(device_id) << "\n";

    if (!HavenCudaEngine::is_cuda_compiled() || HavenCudaEngine::get_device_count() <= 0) {
        std::cout << "[INFO] No active CUDA GPU backend in this build/host — exiting cleanly.\n";
        return 0;
    }

    if (model_path.empty()) {
        std::cerr << "Usage: test-cuda-gemma4 <model.gguf> [--steps 128] [--pp 512] [--repeats 3] [--skip-parity] [--exact] [--fp32-act] [--fp32-kv] [--fp32-prefill] [--host-pleias]\n";
        return 1;
    }

    HavenEngine host_engine;
    if (!host_engine.load_model(model_path)) {
        std::cerr << "Failed to load GGUF: " << model_path << "\n";
        return 1;
    }
    if (exact_mode || fp32_act) {
        host_engine.set_exact_math(true);
    }
    const auto& mc = host_engine.get_config();
    std::cout << "Loaded Model: " << model_path << " (" << mc.architecture
              << ", " << mc.num_layers << "L, D=" << mc.embedding_dim
              << ", F=" << mc.hidden_dim << ", V=" << mc.vocab_size
              << ", MoE=" << (mc.expert_count > 0 ? (std::to_string(mc.expert_count) + "E/top" + std::to_string(mc.expert_used_count)) : "dense")
              << ", TurboV=" << mc.turbo_version
              << ", Precision=" << (exact_mode ? "EXACT(F32-Act+F32-KV+F32-Prefill)" : (fp32_act ? "F32-Act" : "FAST-Q8"))
              << (host_pleias ? ", HostPleias" : "") << ")\n";

    CudaEngineConfig ccfg;
    ccfg.device_id = device_id;
    ccfg.max_seq_len = max_seq_len;
    ccfg.max_prefill_chunk = 256;
    ccfg.exact_fp32_activations = fp32_act;
    ccfg.fp32_kv_cache = fp32_kv;
    ccfg.fp32_prefill = fp32_prefill;
    ccfg.host_resident_pleias = host_pleias;

    HavenCudaEngine cuda_eng;
    if (!cuda_eng.initialize(host_engine, ccfg)) {
        std::cerr << "CUDA initialize failed: " << cuda_eng.get_last_error() << "\n";
        return 1;
    }
    const auto& m = cuda_eng.get_metrics();
    std::cout << "VRAM Footprint : Weights=" << std::fixed << std::setprecision(2)
              << (static_cast<double>(m.weight_vram_bytes) / (1024.0 * 1024.0 * 1024.0)) << " GiB"
              << " | KV=" << (static_cast<double>(m.kv_vram_bytes) / (1024.0 * 1024.0)) << " MiB"
              << " | Scratch=" << (static_cast<double>(m.scratch_vram_bytes) / (1024.0 * 1024.0)) << " MiB"
              << " | Raw Upload=" << std::setprecision(1) << m.upload_ms << " ms (ZERO compile time!)\n";

    // Build test prompt
    const std::string prompt_str = host_engine.format_chat_prompt(
        "In one crisp sentence, how does running Gemma 4.5 natively inside pure C++/CUDA haven.cpp feel?");
    auto prompt_toks = host_engine.tokenize(prompt_str, true);
    std::cout << "Prompt tokens: " << prompt_toks.size() << "\n";

    // 1. Exact 262k-Logit Parity vs CPU Ground Truth (Prefill + Step-2 Decode)
    if (!skip_cpu_parity) {
        std::vector<float> cpu_lg1(mc.vocab_size, 0.0f), cpu_lg2(mc.vocab_size, 0.0f);
        host_engine.get_kv_cache().reset();
        auto c0 = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < prompt_toks.size(); ++i) {
            host_engine.forward(prompt_toks[i], static_cast<int>(i), (i + 1 == prompt_toks.size()) ? cpu_lg1.data() : nullptr);
        }
        uint32_t cpu_t1 = static_cast<uint32_t>(std::max_element(cpu_lg1.begin(), cpu_lg1.end()) - cpu_lg1.begin());
        host_engine.forward(cpu_t1, static_cast<int>(prompt_toks.size()), cpu_lg2.data());
        uint32_t cpu_t2 = static_cast<uint32_t>(std::max_element(cpu_lg2.begin(), cpu_lg2.end()) - cpu_lg2.begin());
        auto c1 = std::chrono::high_resolution_clock::now();

        std::vector<float> gpu_lg1(mc.vocab_size, 0.0f), gpu_lg2(mc.vocab_size, 0.0f);
        uint32_t gpu_t1 = 0, gpu_t2 = 0;
        cuda_eng.reset_kv_cache();
        if (!cuda_eng.forward_prefill(prompt_toks, 0, gpu_lg1.data(), &gpu_t1) ||
            !cuda_eng.forward_decode(cpu_t1, static_cast<int>(prompt_toks.size()), gpu_lg2.data(), &gpu_t2)) {
            std::cerr << "CUDA parity forward failed: " << cuda_eng.get_last_error() << "\n";
            return 1;
        }
        double cos1 = cosine_sim(cpu_lg1.data(), gpu_lg1.data(), mc.vocab_size);
        double cos2 = cosine_sim(cpu_lg2.data(), gpu_lg2.data(), mc.vocab_size);
        double max_d1 = 0.0, max_d2 = 0.0;
        for (size_t i = 0; i < mc.vocab_size; ++i) {
            max_d1 = std::max(max_d1, std::fabs(static_cast<double>(cpu_lg1[i]) - static_cast<double>(gpu_lg1[i])));
            max_d2 = std::max(max_d2, std::fabs(static_cast<double>(cpu_lg2[i]) - static_cast<double>(gpu_lg2[i])));
        }
        std::cout << "[PARITY vs F32 CPU (" << std::setprecision(0)
                  << std::chrono::duration<double, std::milli>(c1 - c0).count() << " ms CPU)]\n"
                  << "  Prefill (S=" << prompt_toks.size() << "): Cosine=" << std::setprecision(7) << cos1
                  << " | max_abs=" << std::scientific << std::setprecision(2) << max_d1 << std::fixed
                  << " | Top1=" << gpu_t1 << " (CPU=" << cpu_t1 << ", Match=" << (gpu_t1 == cpu_t1 ? "YES" : "NO") << ")\n"
                  << "  Decode  (S=1) : Cosine=" << std::setprecision(7) << cos2
                  << " | max_abs=" << std::scientific << std::setprecision(2) << max_d2 << std::fixed
                  << " | Top1=" << gpu_t2 << " (CPU=" << cpu_t2 << ", Match=" << (gpu_t2 == cpu_t2 ? "YES" : "NO") << ")\n";
        if (cos1 < 0.99 || cos2 < 0.99) {
            std::cerr << "[FAIL] Native CUDA parity below 0.99! Check kernel math!\n";
            return 1;
        }
    }

    // 2. Live Coherent Generation Check (64 tokens)
    {
        cuda_eng.reset_kv_cache();
        uint32_t tok = 0;
        cuda_eng.forward_prefill(prompt_toks, 0, nullptr, &tok);
        std::string out = host_engine.detokenize(tok);
        int pos = static_cast<int>(prompt_toks.size());
        for (int s = 0; s < 63; ++s) {
            if (!cuda_eng.forward_decode(tok, pos++, nullptr, &tok)) break;
            if (host_engine.get_tokenizer().is_stop_token(tok) || tok == 1 || tok == 106 || tok == 212) break;
            out += host_engine.detokenize(tok);
        }
        std::cout << "[LIVE CUDA OUTPUT] " << out << "\n";
    }

    // 3. Timed pp128 & pp512 Tensor Core Prefill Benchmark
    for (int pl : {128, pp_len}) {
        if (pl <= 0 || pl > max_seq_len - 16) continue;
        std::vector<uint32_t> synthetic(static_cast<size_t>(pl));
        for (int i = 0; i < pl; ++i) synthetic[static_cast<size_t>(i)] = prompt_toks[static_cast<size_t>(i) % prompt_toks.size()];

        // Warmup
        uint32_t dummy = 0;
        cuda_eng.reset_kv_cache();
        cuda_eng.forward_prefill(synthetic, 0, nullptr, &dummy);

        double best_ms = 1e18, sum_ms = 0.0;
        for (int r = 0; r < repeats; ++r) {
            cuda_eng.reset_kv_cache();
            auto p0 = std::chrono::high_resolution_clock::now();
            cuda_eng.forward_prefill(synthetic, 0, nullptr, &dummy);
            auto p1 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(p1 - p0).count();
            best_ms = std::min(best_ms, ms);
            sum_ms += ms;
        }
        double avg_ms = sum_ms / static_cast<double>(repeats);
        std::cout << "[BENCH pp" << pl << "] avg=" << std::setprecision(2) << avg_ms << " ms ("
                  << std::setprecision(1) << (static_cast<double>(pl) / (avg_ms / 1000.0)) << " tok/s) | peak="
                  << (static_cast<double>(pl) / (best_ms / 1000.0)) << " tok/s\n";
        if (pl == pp_len) break;
    }

    // 4. Timed tg128 (B=1) Warp-GEMV Decode Benchmark (3 repeats)
    {
        double best_ms = 1e18, sum_ms = 0.0;
        for (int r = 0; r < repeats; ++r) {
            cuda_eng.reset_kv_cache();
            uint32_t tok = prompt_toks[0];
            cuda_eng.forward_decode(tok, 0, nullptr, &tok); // warmup step
            auto d0 = std::chrono::high_resolution_clock::now();
            for (int s = 0; s < gen_steps; ++s) {
                if (!cuda_eng.forward_decode(tok, s + 1, nullptr, &tok)) {
                    std::cerr << "Decode failed at step " << s << ": " << cuda_eng.get_last_error() << "\n";
                    return 1;
                }
            }
            auto d1 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(d1 - d0).count();
            best_ms = std::min(best_ms, ms);
            sum_ms += ms;
        }
        double avg_ms = sum_ms / static_cast<double>(repeats);
        double avg_tok_s = static_cast<double>(gen_steps) / (avg_ms / 1000.0);
        double peak_tok_s = static_cast<double>(gen_steps) / (best_ms / 1000.0);
        double eff_gb_s = (static_cast<double>(m.weight_vram_bytes) / (1024.0 * 1024.0 * 1024.0)) * avg_tok_s;
        std::cout << "[BENCH tg" << gen_steps << "] avg=" << std::setprecision(2) << (avg_ms / gen_steps) << " ms/tok -> "
                  << std::setprecision(2) << avg_tok_s << " tok/s (peak " << peak_tok_s << " tok/s"
                  << (mc.expert_count == 0 ? (", eff BW=" + std::to_string(static_cast<int>(eff_gb_s)) + " GiB/s") : "") << ")\n";
    }

    cuda_eng.shutdown();
    std::cout << "======================================================================\n";
    std::cout << "  NATIVE CUDA VERIFICATION & BENCHMARK COMPLETE\n";
    std::cout << "======================================================================\n";
    return 0;
}
