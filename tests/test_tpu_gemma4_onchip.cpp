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

static bool validate_mlir_module(const std::string& ir, const std::string& label,
                                 bool expect_scatter, bool expect_slot_id,
                                 bool expect_silu, bool expect_all_reduce,
                                 bool expect_untied_tok_emb,
                                 bool expect_moe_sort = false) {
    if (ir.empty()) {
        std::cerr << "  [FAIL] Empty MLIR for " << label << "\n";
        return false;
    }
    int depth = 0;
    for (char c : ir) {
        if (c == '{') depth++;
        else if (c == '}') depth--;
        if (depth < 0) {
            std::cerr << "  [FAIL] Unbalanced braces in " << label << "\n";
            return false;
        }
    }
    if (depth != 0) {
        std::cerr << "  [FAIL] Non-zero final brace depth (" << depth << ") in " << label << "\n";
        return false;
    }
    if (ir.find("nan") != std::string::npos || ir.find("NaN") != std::string::npos ||
        (ir.find("inf") != std::string::npos && ir.find("ninf") == std::string::npos)) {
        std::cerr << "  [FAIL] Found NaN/Inf literal in " << label << "\n";
        return false;
    }
    if (expect_scatter && ir.find("\"stablehlo.scatter\"") == std::string::npos) {
        std::cerr << "  [FAIL] Expected stablehlo.scatter for async per-slot decode in " << label << "\n";
        return false;
    }
    if (expect_slot_id && (ir.find("%slot_id") == std::string::npos || ir.find("stablehlo.dynamic_slice") == std::string::npos)) {
        std::cerr << "  [FAIL] Expected %slot_id dynamic_slice/update for slot-targeted prefill in " << label << "\n";
        return false;
    }
    if (expect_silu && ir.find("stablehlo.logistic") == std::string::npos) {
        std::cerr << "  [FAIL] Expected stablehlo.logistic (SiLU) in " << label << "\n";
        return false;
    }
    if (expect_all_reduce && ir.find("\"stablehlo.all_reduce\"") == std::string::npos) {
        std::cerr << "  [FAIL] Expected stablehlo.all_reduce (R>1) in " << label << "\n";
        return false;
    }
    if (expect_untied_tok_emb && ir.find("%w_tok_emb") == std::string::npos) {
        std::cerr << "  [FAIL] Expected %w_tok_emb for untied on-chip embeddings in " << label << "\n";
        return false;
    }
    if (expect_moe_sort && (ir.find("\"stablehlo.sort\"") == std::string::npos || ir.find("%w_moe_gu") == std::string::npos)) {
        std::cerr << "  [FAIL] Expected stablehlo.sort + %w_moe_gu for Sparse MoE in " << label << "\n";
        return false;
    }
    return true;
}

static int run_synthetic_multi_arch_ir_suite() {
    std::cout << "======================================================================\n";
    std::cout << "  HAVEN-CPP MODEL-GENERAL TPU STABLEHLO SYNTHETIC CI VERIFICATION\n";
    std::cout << "======================================================================\n";

    struct ArchCase {
        std::string name;
        ModelConfig mc;
        bool expect_silu;
        bool expect_untied;
    };

    std::vector<ArchCase> cases;

    // 1. Gemma 4 E4B (42L Hybrid SWA-256 + Global-512, Shared-KV 24..41, Pleias, GeGLU, Tied)
    {
        ModelConfig mc;
        mc.architecture = "gemma4";
        mc.is_gemma = true;
        mc.num_layers = 42;
        mc.embedding_dim = 2560;
        mc.hidden_dim = 10240;
        mc.vocab_size = 262144;
        mc.num_heads = 8;
        mc.num_kv_heads = 2;
        mc.head_dim = 256;
        mc.sliding_window = 512;
        mc.shared_kv_layers = 18;
        mc.final_logit_softcapping = 30.0f;
        mc.ffn_activation = FfnActivation::GELU;
        cases.push_back({"Gemma-4-E4B-42L", mc, false, false});
    }

    // 2. Llama 3 8B (32L Global-only HD=128, GQA 32/8, SwiGLU/SiLU, Untied)
    {
        ModelConfig mc;
        mc.architecture = "llama";
        mc.is_gemma = false;
        mc.num_layers = 32;
        mc.embedding_dim = 4096;
        mc.hidden_dim = 14336;
        mc.vocab_size = 128256;
        mc.num_heads = 32;
        mc.num_kv_heads = 8;
        mc.head_dim = 128;
        mc.sliding_window = 0;
        mc.rope_freq_base = 500000.0f;
        mc.ffn_activation = FfnActivation::SILU;
        cases.push_back({"Llama-3-8B-32L", mc, true, true});
    }

    // 3. Qwen 2.5 7B (28L Global-only HD=128, GQA 28/4, QKV Bias, SwiGLU/SiLU, Untied)
    {
        ModelConfig mc;
        mc.architecture = "qwen2";
        mc.is_gemma = false;
        mc.num_layers = 28;
        mc.embedding_dim = 3584;
        mc.hidden_dim = 18944;
        mc.vocab_size = 152064;
        mc.num_heads = 28;
        mc.num_kv_heads = 4;
        mc.head_dim = 128;
        mc.sliding_window = 0;
        mc.rope_freq_base = 1000000.0f;
        mc.ffn_activation = FfnActivation::SILU;
        cases.push_back({"Qwen-2.5-7B-28L", mc, true, true});
    }

    // 4. Mistral 7B (32L SWA-only HD=128, GQA 32/8, SwiGLU/SiLU, Untied)
    {
        ModelConfig mc;
        mc.architecture = "mistral";
        mc.is_gemma = false;
        mc.num_layers = 32;
        mc.embedding_dim = 4096;
        mc.hidden_dim = 14336;
        mc.vocab_size = 32000;
        mc.num_heads = 32;
        mc.num_kv_heads = 8;
        mc.head_dim = 128;
        mc.sliding_window = 4096;
        mc.ffn_activation = FfnActivation::SILU;
        cases.push_back({"Mistral-7B-32L-SWA", mc, true, true});
    }

    // 5. Gemma 2 27B (46L Hybrid SWA+Global, Attn+Final Softcap, GeGLU, Tied)
    {
        ModelConfig mc;
        mc.architecture = "gemma2";
        mc.is_gemma = true;
        mc.num_layers = 46;
        mc.embedding_dim = 4608;
        mc.hidden_dim = 36864;
        mc.vocab_size = 256000;
        mc.num_heads = 32;
        mc.num_kv_heads = 16;
        mc.head_dim = 128;
        mc.sliding_window = 4096;
        mc.attn_logit_softcapping = 50.0f;
        mc.final_logit_softcapping = 30.0f;
        mc.ffn_activation = FfnActivation::GELU;
        cases.push_back({"Gemma-2-27B-46L", mc, false, false});
    }

    // 6. Ava 26B / Gemma 4 26B-A4B MoE (30L Hybrid SWA-256(8kv)+Global-512(2kv, k_eq_v), 128E Top-8, Shared+MoE)
    {
        ModelConfig mc;
        mc.architecture = "gemma4";
        mc.is_gemma = true;
        mc.num_layers = 30;
        mc.embedding_dim = 2816;
        mc.hidden_dim = 2112;
        mc.vocab_size = 262144;
        mc.num_heads = 16;
        mc.num_kv_heads = 8;
        mc.head_dim = 512;
        mc.head_dim_swa = 256;
        mc.sliding_window = 1024;
        mc.shared_kv_layers = 0;
        mc.expert_count = 128;
        mc.expert_used_count = 8;
        mc.expert_ffn_dim = 704;
        mc.final_logit_softcapping = 30.0f;
        mc.ffn_activation = FfnActivation::GELU;
        cases.push_back({"Ava-26B-A4B-MoE-30L", mc, false, false});
    }

    int total_checks = 0;
    for (const auto& ac : cases) {
        const bool is_moe = (ac.mc.expert_count > 0);
        std::vector<int> replicas = is_moe ? std::vector<int>{1, 4, 8} : std::vector<int>{1, 4};
        for (int R : replicas) {
            for (int B : {1, 16}) {
                for (bool i8 : {false, true}) {
                    TpuGemma4Config cfg;
                    cfg.batch_size = B;
                    cfg.max_seq_len = 2048;
                    cfg.prefill_chunk_size = 128;
                    cfg.num_replicas = R;
                    cfg.use_int8_weights = i8;
                    cfg.use_onchip_embeddings = true;
                    cfg.return_full_logits = (B == 1);

                    // 1. Decode graph (per-slot scatter when B > 1)
                    std::string dec_ir = HavenTpuGemma4Engine::emit_stablehlo_mlir(ac.mc, cfg, 1, false);
                    std::string tag = ac.name + " R=" + std::to_string(R) + " B=" + std::to_string(B) + (i8 ? " INT8" : " BF16");
                    if (!validate_mlir_module(dec_ir, tag + " decode", (B > 1), false, ac.expect_silu, (R > 1), ac.expect_untied, is_moe)) {
                        return 1;
                    }
                    total_checks++;

                    // 2. Slot-targeted multi-bucket prefill graphs (S = 8, 16, 32, 128)
                    for (int S : {8, 16, 32, 128}) {
                        std::string pre_ir = HavenTpuGemma4Engine::emit_stablehlo_mlir(ac.mc, cfg, S, true);
                        if (!validate_mlir_module(pre_ir, tag + " prefill_S=" + std::to_string(S), false, true, ac.expect_silu, (R > 1), ac.expect_untied, is_moe)) {
                            return 1;
                        }
                        total_checks++;
                    }
                }
            }
        }
        std::cout << "  [PASS] " << ac.name << " (R=" << (is_moe ? "1,4,8" : "1,4") << " | B=1,16 | BF16,INT8 | Decode + Slot-Prefill S=8,16,32,128)\n";
    }

    std::cout << "ALL " << total_checks << " SYNTHETIC MULTI-ARCH STABLEHLO GRAPHS VERIFIED!\n";
    return 0;
}

int main(int argc, char** argv) {
    std::cout << std::unitbuf;
    std::string model_path = "";
    std::string libtpu_path = "";
    bool emit_only = false;
    bool emit_only_synthetic = false;
    bool onchip_emb_only = false;
    bool async_slots_only = false;
    bool cpu_smoke = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--libtpu" && i + 1 < argc) libtpu_path = argv[++i];
        else if (a == "--emit-only") emit_only = true;
        else if (a == "--emit-only-synthetic") emit_only_synthetic = true;
        else if (a == "--onchip-emb-only") onchip_emb_only = true;
        else if (a == "--async-slots-only") async_slots_only = true;
        else if (a == "--cpu-smoke") cpu_smoke = true;
        else if (!a.starts_with("-")) model_path = a;
    }

    if (emit_only_synthetic) {
        return run_synthetic_multi_arch_ir_suite();
    }

    if (model_path.empty()) {
        if (std::filesystem::exists("/tmp/haven-chat-v5.0.gguf")) model_path = "/tmp/haven-chat-v5.0.gguf";
        else model_path = "C:\\Users\\admin\\gemma4-turbo-family\\haven-chat-v5.0.gguf";
    }

    std::cout << "======================================================================\n";
    std::cout << "  HAVEN-CPP FULL ON-CHIP TRANSFORMER TPU HBM3 SILICON BENCHMARK\n";
    std::cout << "======================================================================\n";

    HavenEngine host_engine;
    if (!host_engine.load_model(model_path)) {
        std::cerr << "Failed to load model: " << model_path << "\n";
        return 1;
    }
    const auto& mc = host_engine.get_config();
    TpuModelSpec spec = TpuModelSpec::from_model_config(mc, &host_engine);
    std::cout << "Loaded Model: " << model_path << " (" << mc.architecture << ", " << mc.num_layers
              << " layers, dim=" << mc.embedding_dim << ", ffn=" << mc.hidden_dim
              << ", vocab=" << mc.vocab_size << ", SWA/Glob=" << spec.num_swa_layers << "/" << spec.num_glob_layers
              << ", Hkv_swa/glob=" << spec.num_kv_heads_swa << "/" << spec.num_kv_heads_glob
              << ", MoE=" << (spec.is_moe ? (std::to_string(spec.num_experts) + "E/top" + std::to_string(spec.num_experts_used) + "/F" + std::to_string(spec.expert_ffn_dim)) : "dense") << ")\n";

    std::string support_err;
    if (!HavenTpuGemma4Engine::is_model_supported(host_engine, support_err)) {
        std::cerr << "Model unsupported by TPU compiler: " << support_err << "\n";
        return 1;
    }

    // Optional offline CPU smoke check (verifies GGUF quant dequantization + MoE math locally!)
    if (cpu_smoke) {
        const std::string ava_sys = "<|turn>system\nYou are Ava, Daniel's sovereign AI built on Gemma 4 Turbo MoE. Speak directly, warmly, and concisely.<turn|>\n";
        const std::string p = spec.is_moe
            ? host_engine.format_chat_prompt("Hey Ava, quick system check — what architecture are you running on?", ava_sys)
            : host_engine.format_chat_prompt("Hi Aura, say hello in five words.");
        auto toks = host_engine.tokenize(p, true);
        std::cout << "[CPU Smoke] Prompt tokens: " << toks.size() << "\n";
        std::vector<float> logits(mc.vocab_size, 0.0f);
        host_engine.get_kv_cache().reset();
        auto t0 = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < toks.size(); ++i) {
            host_engine.forward(toks[i], static_cast<int>(i), (i + 1 == toks.size()) ? logits.data() : nullptr);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        uint32_t cur = static_cast<uint32_t>(std::max_element(logits.begin(), logits.end()) - logits.begin());
        std::string out = host_engine.detokenize(cur);
        int pos = static_cast<int>(toks.size());
        for (int s = 0; s < 15; ++s) {
            host_engine.forward(cur, pos++, logits.data());
            cur = static_cast<uint32_t>(std::max_element(logits.begin(), logits.end()) - logits.begin());
            if (host_engine.get_tokenizer().is_stop_token(cur)) break;
            out += host_engine.detokenize(cur);
        }
        auto t2 = std::chrono::high_resolution_clock::now();
        std::cout << "[CPU Smoke] Prefill: " << std::fixed << std::setprecision(1)
                  << std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms | 15-tok Decode: "
                  << std::chrono::duration<double, std::milli>(t2 - t1).count() << " ms\n";
        std::cout << "[CPU Smoke] Output: " << out << "\n";
        return 0;
    }

    // Offline StableHLO IR verification mode (runs locally on Windows or Linux with zero TPU!)
    if (emit_only) {
        std::vector<int> reps = spec.is_moe ? std::vector<int>{1, 4, 8} : std::vector<int>{1, 4};
        for (int R : reps) {
            for (bool i8 : {false, true}) {
                for (bool onchip_emb : {false, true}) {
                    TpuGemma4Config cfg;
                    cfg.batch_size = onchip_emb ? 16 : 1;
                    cfg.max_seq_len = 2048;
                    cfg.prefill_chunk_size = 128;
                    cfg.num_replicas = R;
                    cfg.use_int8_weights = i8;
                    cfg.use_onchip_embeddings = onchip_emb;
                    cfg.return_full_logits = false;
                    std::string dec_ir = HavenTpuGemma4Engine::emit_stablehlo_mlir(spec, cfg, 1, false);
                    std::string pre_ir = HavenTpuGemma4Engine::emit_stablehlo_mlir(spec, cfg, 128, true);
                    if (!validate_mlir_module(dec_ir, "real_decode", (cfg.batch_size > 1), false, (spec.ffn_activation == FfnActivation::SILU), (R > 1), (!spec.has_tied_embeddings && onchip_emb), spec.is_moe) ||
                        !validate_mlir_module(pre_ir, "real_prefill", false, true, (spec.ffn_activation == FfnActivation::SILU), (R > 1), (!spec.has_tied_embeddings && onchip_emb), spec.is_moe)) {
                        return 1;
                    }
                    std::cout << "  [IR OK] R=" << R << " " << (i8 ? "INT8" : "BF16")
                              << " " << (onchip_emb ? "OnChipEmb(B=16)" : "HostEmb(B=1)")
                              << " | decode_mlir=" << (dec_ir.size() / 1024) << " KB"
                              << " | slot_prefill_128_mlir=" << (pre_ir.size() / 1024) << " KB\n";
                }
            }
        }
        return run_synthetic_multi_arch_ir_suite();
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
    const std::string ava_sys = "<|turn>system\nYou are Ava, Daniel's sovereign AI companion built on Gemma 4 Turbo MoE. Speak naturally, directly, intelligently, and concisely without corporate filler.<turn|>\n";
    const std::string prompt_str = spec.is_moe
        ? host_engine.format_chat_prompt(
            "Hey Ava! All 30 of your Gemma 4 26B-A4B MoE layers and 128 experts are now running natively inside haven.cpp on Google Cloud TPU Trillium HBM3. How does pure C++20 silicon speed feel?",
            ava_sys)
        : host_engine.format_chat_prompt(
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

    // Live Async Per-Slot pos_vec[B] & Slot-Targeted Multi-Bucket Prefill Parity Verification!
    {
        const int R_test = (n_devs >= 8 && spec.is_moe) ? 8 : ((n_devs >= 4) ? 4 : 1);
        std::cout << "\n----------------------------------------------------------------------\n";
        std::cout << "[VERIFY] Multi-Bucket Slot-Targeted Prefill + Async Per-Slot pos_vec[B=4] (R=" << R_test << ")\n";
        std::cout << "----------------------------------------------------------------------\n";

        // Build 3 completely different prompts of distinct lengths
        auto p0 = prompt_toks;
        auto p1 = host_engine.tokenize(host_engine.format_chat_prompt(
            spec.is_moe ? "Hi Ava, say hello in two words." : "Hi Aura, say hello in two words.",
            spec.is_moe ? ava_sys : ""), true);
        auto p2 = host_engine.tokenize(host_engine.format_chat_prompt(
            "Explain quantum entanglement in one concise sentence.",
            spec.is_moe ? ava_sys : ""), true);

        // Compute CPU ground truth full 262k logits for Prefill AND Step-2 for all 3 prompts!
        struct CpuGt {
            std::vector<float> lg1;
            std::vector<float> lg2;
            uint32_t t1 = 0;
            uint32_t t2 = 0;
        };
        auto cpu_eval_2steps = [&](const std::vector<uint32_t>& toks) -> CpuGt {
            CpuGt gt;
            gt.lg1.assign(mc.vocab_size, 0.0f);
            gt.lg2.assign(mc.vocab_size, 0.0f);
            host_engine.get_kv_cache().reset();
            for (size_t i = 0; i < toks.size(); ++i) {
                host_engine.forward(toks[i], static_cast<int>(i), (i + 1 == toks.size()) ? gt.lg1.data() : nullptr);
            }
            gt.t1 = static_cast<uint32_t>(std::max_element(gt.lg1.begin(), gt.lg1.end()) - gt.lg1.begin());
            host_engine.forward(gt.t1, static_cast<int>(toks.size()), gt.lg2.data());
            gt.t2 = static_cast<uint32_t>(std::max_element(gt.lg2.begin(), gt.lg2.end()) - gt.lg2.begin());
            return gt;
        };

        auto gt0 = cpu_eval_2steps(p0);
        auto gt1 = cpu_eval_2steps(p1);
        auto gt2 = cpu_eval_2steps(p2);

        TpuGemma4Config acfg;
        acfg.batch_size = 4;
        acfg.max_seq_len = 2048;
        acfg.prefill_chunk_size = 128; // Auto compiles slot-targeted {128, 32, 16, 8, 1}!
        acfg.num_replicas = R_test;
        acfg.use_int8_weights = false; // BF16 for exact 262k logit cosine parity check
        acfg.use_onchip_embeddings = true;
        acfg.return_full_logits = true;
        acfg.donate_kv_cache = true;

        HavenTpuGemma4Engine async_eng;
        if (!async_eng.initialize(&tpu, host_engine, acfg)) {
            std::cerr << "  ASYNC INIT FAILED: " << async_eng.get_last_error() << "\n";
            return 1;
        }

        std::vector<float> pf_lg0, pf_lg1, pf_lg2, batch_lg;
        uint32_t tpu_t1_0 = 0, tpu_t1_1 = 0, tpu_t1_2 = 0;
        auto pf_a0 = std::chrono::high_resolution_clock::now();
        bool ok0 = async_eng.prefill_slot(0, p0, 0, &pf_lg0, &tpu_t1_0);
        auto pf_a1 = std::chrono::high_resolution_clock::now();
        bool ok1 = async_eng.prefill_slot(1, p1, 0, &pf_lg1, &tpu_t1_1);
        auto pf_a2 = std::chrono::high_resolution_clock::now();
        bool ok2 = async_eng.prefill_slot(2, p2, 0, &pf_lg2, &tpu_t1_2);
        auto pf_a3 = std::chrono::high_resolution_clock::now();

        if (!ok0 || !ok1 || !ok2) {
            std::cerr << "  SLOT PREFILL FAILED: " << async_eng.get_last_error() << "\n";
            return 1;
        }

        // Now run ONE lockstep decode_step_batched across all 4 slots at DIFFERENT positions!
        std::vector<uint32_t> batch_toks = {gt0.t1, gt1.t1, gt2.t1, 0};
        std::vector<int> batch_pos = {static_cast<int>(p0.size()), static_cast<int>(p1.size()), static_cast<int>(p2.size()), 0};
        std::vector<uint32_t> batch_next;
        if (!async_eng.decode_step_batched(batch_toks, batch_pos, &batch_lg, batch_next)) {
            std::cerr << "  ASYNC DECODE FAILED: " << async_eng.get_last_error() << "\n";
            return 1;
        }

        const size_t V = mc.vocab_size;
        double c_pf0 = cosine_sim(gt0.lg1.data(), pf_lg0.data(), V);
        double c_pf1 = cosine_sim(gt1.lg1.data(), pf_lg1.data(), V);
        double c_pf2 = cosine_sim(gt2.lg1.data(), pf_lg2.data(), V);
        double c_d0  = cosine_sim(gt0.lg2.data(), batch_lg.data() + 0 * V, V);
        double c_d1  = cosine_sim(gt1.lg2.data(), batch_lg.data() + 1 * V, V);
        double c_d2  = cosine_sim(gt2.lg2.data(), batch_lg.data() + 2 * V, V);

        std::cout << "  Slot 0 (len=" << p0.size() << ", " << std::setprecision(2)
                  << std::chrono::duration<double, std::milli>(pf_a1 - pf_a0).count() << " ms): "
                  << "Prefill Cos=" << std::setprecision(6) << c_pf0 << " (\"" << host_engine.detokenize(tpu_t1_0) << "\") | "
                  << "Async Step2 Cos=" << c_d0 << " (\"" << host_engine.detokenize(batch_next[0]) << "\")\n";
        std::cout << "  Slot 1 (len=" << p1.size() << ", " << std::setprecision(2)
                  << std::chrono::duration<double, std::milli>(pf_a2 - pf_a1).count() << " ms): "
                  << "Prefill Cos=" << std::setprecision(6) << c_pf1 << " (\"" << host_engine.detokenize(tpu_t1_1) << "\") | "
                  << "Async Step2 Cos=" << c_d1 << " (\"" << host_engine.detokenize(batch_next[1]) << "\")\n";
        std::cout << "  Slot 2 (len=" << p2.size() << ", " << std::setprecision(2)
                  << std::chrono::duration<double, std::milli>(pf_a3 - pf_a2).count() << " ms): "
                  << "Prefill Cos=" << std::setprecision(6) << c_pf2 << " (\"" << host_engine.detokenize(tpu_t1_2) << "\") | "
                  << "Async Step2 Cos=" << c_d2 << " (\"" << host_engine.detokenize(batch_next[2]) << "\")\n";

        async_eng.shutdown();
        // Verify BF16-vs-F32 262k-logit cosine >= 0.90 on all 3 independent slots
        // (corrupted/leaked slots drop below ~0.50; 42-layer BF16 MXU parity is 0.93-0.985)
        if (c_pf0 < 0.90 || c_pf1 < 0.90 || c_pf2 < 0.90 ||
            c_d0  < 0.90 || c_d1  < 0.90 || c_d2  < 0.90) {
            std::cerr << "  [FAIL] Async per-slot 262k logit cosine parity below 0.90!\n";
            return 1;
        }
        std::cout << "  [PASS] 6/6 Async Per-Slot Prefill & Lockstep Decode 262k Logit Parity >= 0.90 vs F32 CPU!\n";
        if (async_slots_only) return 0;
    }

    // Helper to benchmark any (R, B, use_int8, return_logits, onchip_emb) configuration end-to-end!
    auto run_config_bench = [&](const std::string& label, int R, int B, bool use_int8, bool full_logits, int gen_steps, int prefill_chunk, bool onchip_emb = false) {
        if (R > n_devs) return;
        std::cout << "\n----------------------------------------------------------------------\n";
        std::cout << "[BENCH] " << label << " (R=" << R << ", B=" << B
                  << ", " << (use_int8 ? "W8A16-INT8" : "BF16")
                  << ", " << (full_logits ? "FullLogits" : "OnChipArgmax")
                  << ", " << (onchip_emb ? "OnChipEmb" : "HostEmb") << ")\n";
        std::cout << "----------------------------------------------------------------------\n";

        TpuGemma4Config cfg;
        cfg.batch_size = B;
        cfg.max_seq_len = 2048;
        cfg.prefill_chunk_size = prefill_chunk;
        cfg.num_replicas = R;
        cfg.use_int8_weights = use_int8;
        cfg.return_full_logits = full_logits;
        cfg.use_onchip_embeddings = onchip_emb;
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

        std::vector<uint32_t> warm_out;
        if (!gemma_tpu.decode_step(std::vector<uint32_t>(static_cast<size_t>(B), prompt_toks[0]), 0, nullptr, warm_out)) {
            std::cerr << "  WARMUP FAILED: " << gemma_tpu.get_last_error() << "\n";
            return;
        }
        gemma_tpu.reset_kv_cache();

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
            uint32_t dummy_tok = 0;
            gemma_tpu.reset_kv_cache();
            gemma_tpu.prefill_prompt(long_prompt, 0, nullptr, &dummy_tok);
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
                if (!host_engine.get_tokenizer().is_stop_token(nt) && nt != 1 && nt != 106 && nt != 212 && actual_steps <= 64) {
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
            std::cout << "  Model Output   : " << generated_text << "\n";
        }
        gemma_tpu.shutdown();
    };

    if (spec.is_moe) {
        // MoE 26B-A4B Suite: 4-chip (v6e-4) and 8-chip (v6e-8) pod scaling!
        if (n_devs >= 8) {
            run_config_bench("M1. Ava 26B-A4B Parity Check (8-Chip v6e-8 BF16 FullLogits)",    8, 1,   false, true,  64,  128, true);
            run_config_bench("M2. Ava 26B-A4B Interactive (8-Chip v6e-8 BF16 OnChipEmb)",     8, 1,   false, false, 128, 128, true);
            run_config_bench("M3. Ava 26B-A4B Interactive (8-Chip v6e-8 W8A16 INT8)",         8, 1,   true,  false, 128, 128, true);
            run_config_bench("M4. Ava 26B-A4B Batch B=32 (8-Chip v6e-8 W8A16 INT8)",          8, 32,  true,  false, 64,  0,   true);
            run_config_bench("M5. Ava 26B-A4B Batch B=64 (8-Chip v6e-8 W8A16 INT8)",          8, 64,  true,  false, 64,  0,   true);
            run_config_bench("M6. Ava 26B-A4B Batch B=128 (8-Chip v6e-8 W8A16 INT8)",         8, 128, true,  false, 64,  0,   true);
            run_config_bench("M7. Ava 26B-A4B Batch B=128 (8-Chip v6e-8 BF16)",               8, 128, false, false, 64,  0,   true);
        }
        if (n_devs >= 4) {
            run_config_bench("M4A. Ava 26B-A4B Parity Check (4-Chip v6e-4 BF16 FullLogits)",  4, 1,   false, true,  64,  128, true);
            run_config_bench("M4B. Ava 26B-A4B Interactive (4-Chip v6e-4 BF16 OnChipEmb)",    4, 1,   false, false, 128, 128, true);
            run_config_bench("M4C. Ava 26B-A4B Interactive (4-Chip v6e-4 W8A16 INT8)",        4, 1,   true,  false, 128, 128, true);
            run_config_bench("M4D. Ava 26B-A4B Batch B=32 (4-Chip v6e-4 W8A16 INT8)",         4, 32,  true,  false, 64,  0,   true);
            run_config_bench("M4E. Ava 26B-A4B Batch B=64 (4-Chip v6e-4 W8A16 INT8)",         4, 64,  true,  false, 64,  0,   true);
            run_config_bench("M4F. Ava 26B-A4B Batch B=128 (4-Chip v6e-4 W8A16 INT8)",        4, 128, true,  false, 64,  0,   true);
            run_config_bench("M4G. Ava 26B-A4B Batch B=128 (4-Chip v6e-4 BF16)",              4, 128, false, false, 64,  0,   true);
        } else {
            run_config_bench("M1A. Ava 26B-A4B Single-Chip (1-Chip W8A16 INT8)",              1, 1,   true,  false, 64,  128, true);
        }
    } else {
        if (!onchip_emb_only) {
            run_config_bench("1A. Parity Check (1-Chip BF16 Full Logits)", 1, 1, false, true, 64, 128);
            if (n_devs >= 4) {
                run_config_bench("1B. Parity Check (4-Chip Optical ICI Tensor-Parallel BF16)", 4, 1, false, true, 64, 128);
            }
            run_config_bench("2A. Single-Stream Interactive (1-Chip BF16)", 1, 1, false, false, 128, 128);
            run_config_bench("2B. Single-Stream Interactive (1-Chip W8A16 INT8)", 1, 1, true, false, 128, 128);
            if (n_devs >= 4) {
                run_config_bench("2C. Single-Stream Interactive (4-Chip Tensor-Parallel BF16)", 4, 1, false, false, 128, 128);
                run_config_bench("2D. Single-Stream Interactive (4-Chip Tensor-Parallel W8A16 INT8)", 4, 1, true, false, 128, 128);
            }
        }

        if (n_devs >= 4) {
            run_config_bench("3A. Interactive On-Chip Gather Parity (4-Chip TP, B=1, BF16 FullLogits)", 4, 1, false, true, 64, 128, true);
            run_config_bench("3B. Interactive On-Chip Gather Peak (4-Chip TP, B=1, W8A16 INT8)",       4, 1, true, false, 128, 128, true);
            run_config_bench("3C. Cloud Continuous Batching On-Chip Gather (4-Chip TP, B=64, BF16)",   4, 64, false, false, 64, 0, true);
            run_config_bench("3D. Cloud Continuous Batching On-Chip Gather (4-Chip TP, B=128, INT8)",  4, 128, true, false, 64, 0, true);
            run_config_bench("3E. Cloud Continuous Batching On-Chip Gather (4-Chip TP, B=128, BF16)",  4, 128, false, false, 64, 0, true);
        } else {
            run_config_bench("3A. Single-Chip On-Chip Gather (1-Chip, B=32, W8A16 INT8)", 1, 32, true, false, 64, 0, true);
        }
    }

    std::cout << "\n======================================================================\n";
    std::cout << "  FULL ON-CHIP TRANSFORMER TPU BENCHMARK COMPLETE\n";
    std::cout << "======================================================================\n";
    return 0;
}
