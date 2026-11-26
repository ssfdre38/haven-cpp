#include "haven/haven_intelligence.h"
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

// Minimal synthetic GGUF v3 header writer (zero external model files required!)
struct SyntheticGgufBuilder {
    struct Kv {
        std::string key;
        uint32_t type;
        std::vector<uint8_t> val;
    };
    struct Tensor {
        std::string name;
        std::vector<uint64_t> dims;
        uint32_t type;
        uint64_t offset;
    };

    std::vector<Kv> kvs;
    std::vector<Tensor> tensors;

    static void append_u32(std::vector<uint8_t>& b, uint32_t v) {
        size_t o = b.size();
        b.resize(o + 4);
        std::memcpy(b.data() + o, &v, 4);
    }
    static void append_u64(std::vector<uint8_t>& b, uint64_t v) {
        size_t o = b.size();
        b.resize(o + 8);
        std::memcpy(b.data() + o, &v, 8);
    }
    static void append_str(std::vector<uint8_t>& b, const std::string& s) {
        append_u64(b, static_cast<uint64_t>(s.size()));
        size_t o = b.size();
        b.resize(o + s.size());
        if (!s.empty()) std::memcpy(b.data() + o, s.data(), s.size());
    }

    void add_str(const std::string& k, const std::string& v) {
        std::vector<uint8_t> b;
        append_str(b, v);
        kvs.push_back({k, 8u, std::move(b)});
    }
    void add_u32(const std::string& k, uint32_t v) {
        std::vector<uint8_t> b;
        append_u32(b, v);
        kvs.push_back({k, 4u, std::move(b)});
    }
    void add_bool(const std::string& k, bool v) {
        kvs.push_back({k, 7u, {static_cast<uint8_t>(v ? 1 : 0)}});
    }
    void add_bool_array(const std::string& k, const std::vector<bool>& arr) {
        std::vector<uint8_t> b;
        append_u32(b, 7u);
        append_u64(b, static_cast<uint64_t>(arr.size()));
        for (bool x : arr) b.push_back(static_cast<uint8_t>(x ? 1 : 0));
        kvs.push_back({k, 9u, std::move(b)});
    }
    void add_dummy_vocab(uint32_t n_tokens) {
        std::vector<uint8_t> b;
        append_u32(b, 8u);
        append_u64(b, static_cast<uint64_t>(n_tokens));
        for (uint32_t i = 0; i < n_tokens; ++i) {
            append_str(b, "tok");
        }
        kvs.push_back({"tokenizer.ggml.tokens", 9u, std::move(b)});
    }
    void add_tensor(const std::string& name, std::vector<uint64_t> dims, haven::QuantType qt) {
        tensors.push_back({name, std::move(dims), static_cast<uint32_t>(qt), 0ULL});
    }

    bool write(const std::string& path) const {
        std::vector<uint8_t> out;
        append_u32(out, 0x46554747u); // "GGUF"
        append_u32(out, 3u);          // version 3
        append_u64(out, static_cast<uint64_t>(tensors.size()));
        append_u64(out, static_cast<uint64_t>(kvs.size()));

        for (const auto& kv : kvs) {
            append_str(out, kv.key);
            append_u32(out, kv.type);
            out.insert(out.end(), kv.val.begin(), kv.val.end());
        }
        for (const auto& t : tensors) {
            append_str(out, t.name);
            append_u32(out, static_cast<uint32_t>(t.dims.size()));
            for (uint64_t d : t.dims) append_u64(out, d);
            append_u32(out, t.type);
            append_u64(out, t.offset);
        }
        while (out.size() % 32 != 0) out.push_back(0);
        // Tiny 64-byte sentinel payload (header probe never maps/reads tensor payloads!)
        out.resize(out.size() + 64, 0);

        std::ofstream f(path, std::ios::binary);
        if (!f) return false;
        f.write(reinterpret_cast<const char*>(out.data()), static_cast<std::streamsize>(out.size()));
        return static_cast<bool>(f);
    }
};

#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        std::cerr << "FAIL: " << msg << " (line " << __LINE__ << ")\n"; \
        return 1; \
    } \
} while (0)

} // namespace

int main() {
    using namespace haven;
    std::cout << "========================================================================\n";
    std::cout << "  HAVEN MODEL INTELLIGENCE & MICROKERNEL MODULE LOADER TEST SUITE\n";
    std::cout << "========================================================================\n";

    const auto tmp_dir = std::filesystem::temp_directory_path() / "haven_intel_test";
    std::filesystem::create_directories(tmp_dir);

    const std::string p_g4_e2b  = (tmp_dir / "gemma4_e2b_turbo.gguf").string();
    const std::string p_g4_26b  = (tmp_dir / "gemma4_26b_moe_mm.gguf").string();
    const std::string p_qwen25  = (tmp_dir / "qwen25_7b.gguf").string();
    const std::string p_llama3  = (tmp_dir / "llama3_8b.gguf").string();
    const std::string p_phi3    = (tmp_dir / "phi3_mini.gguf").string();
    const std::string p_corrupt = (tmp_dir / "corrupt.gguf").string();

    // 1. Build synthetic Gemma 4.5 E2B Turbo (Pleias + SWA + SharedKV + Turbo FWHT, NO MoE, NO Vision)
    {
        SyntheticGgufBuilder b;
        b.add_str("general.architecture", "gemma4");
        b.add_str("general.name", "Gemma-4.5-E2B-Haven-Turbo");
        b.add_u32("gemma4.block_count", 35);
        b.add_u32("gemma4.embedding_length", 1536);
        b.add_u32("gemma4.feed_forward_length", 6144);
        b.add_u32("gemma4.attention.head_count", 8);
        b.add_u32("gemma4.attention.head_count_kv", 1);
        b.add_u32("gemma4.attention.key_length", 512);
        b.add_u32("gemma4.attention.key_length_swa", 256);
        b.add_u32("gemma4.attention.sliding_window", 512);
        b.add_u32("gemma4.attention.shared_kv_layers", 20);
        b.add_u32("gemma4.context_length", 131072);
        b.add_u32("gemma4.turbo.version", 1);
        b.add_u32("gemma4.turbo.hadamard_qkv", 256);
        b.add_u32("gemma4.turbo.hadamard_ffn", 256);
        b.add_bool("gemma4.turbo.folded_norms", true);
        b.add_str("tokenizer.chat_template", "<|turn>user\n{{content}}<turn|>\n<|turn>model\n<|channel>thought\n<channel|>");
        b.add_dummy_vocab(1024);
        b.add_tensor("token_embd.weight", {1536, 262144}, QuantType::Q6_K);
        b.add_tensor("per_layer_token_embd.weight", {8960, 262144}, QuantType::IQ4_XS);
        b.add_tensor("blk.0.attn_q.weight", {1536, 2048}, QuantType::IQ4_XS);
        b.add_tensor("blk.0.attn_k.weight", {1536, 256}, QuantType::IQ4_XS);
        b.add_tensor("blk.0.attn_v.weight", {1536, 256}, QuantType::IQ4_XS);
        b.add_tensor("blk.0.attn_q_norm.weight", {256}, QuantType::F32);
        CHECK(b.write(p_g4_e2b), "write gemma4_e2b_turbo.gguf");
    }

    // 2. Build synthetic Gemma 4.5 26B-A4B MoE + SigLIP Multimodal
    {
        SyntheticGgufBuilder b;
        b.add_str("general.architecture", "gemma4");
        b.add_str("general.name", "Ava-4.5-26B-A4B-Turbo-MM");
        b.add_u32("gemma4.block_count", 30);
        b.add_u32("gemma4.embedding_length", 2816);
        b.add_u32("gemma4.feed_forward_length", 2112);
        b.add_u32("gemma4.attention.head_count", 16);
        b.add_u32("gemma4.attention.head_count_kv", 8);
        b.add_u32("gemma4.attention.key_length", 512);
        b.add_u32("gemma4.attention.key_length_swa", 256);
        b.add_u32("gemma4.attention.sliding_window", 1024);
        b.add_u32("gemma4.expert_count", 128);
        b.add_u32("gemma4.expert_used_count", 8);
        b.add_u32("gemma4.expert_feed_forward_length", 704);
        b.add_u32("gemma4.context_length", 262144);
        b.add_u32("gemma4.turbo.version", 1);
        b.add_u32("gemma4.turbo.hadamard_qkv", 256);
        b.add_u32("gemma4.turbo.hadamard_moe_in", 256);
        b.add_bool("gemma4.turbo.folded_scales", true);
        b.add_u32("gemma4.vision.image_size", 896);
        b.add_str("tokenizer.chat_template", "<|turn>user\n{{content}}<turn|>\n<|turn>model\n<|channel>thought\n<channel|>");
        b.add_dummy_vocab(1024);
        b.add_tensor("blk.0.attn_k.weight", {2816, 2048}, QuantType::IQ4_XS);
        b.add_tensor("blk.0.attn_v.weight", {2816, 2048}, QuantType::IQ4_XS);
        b.add_tensor("blk.5.attn_k.weight", {2816, 1024}, QuantType::IQ4_XS); // k_eq_v global layer (no v!)
        b.add_tensor("blk.0.ffn_gate_up_exps.weight", {2816, 1408, 128}, QuantType::IQ4_XS);
        b.add_tensor("blk.0.ffn_down_exps.weight", {704, 2816, 128}, QuantType::IQ4_XS);
        b.add_tensor("v.patch_embd.weight", {1152, 768}, QuantType::F16);
        b.add_tensor("mm.input_projection.weight", {1152, 2816}, QuantType::F16);
        CHECK(b.write(p_g4_26b), "write gemma4_26b_moe_mm.gguf");
    }

    // 3. Build synthetic Qwen 2.5 7B (QKV bias, ChatML, Q4_K, NO MoE, NO Turbo, NO Vision)
    {
        SyntheticGgufBuilder b;
        b.add_str("general.architecture", "qwen2");
        b.add_str("general.name", "Qwen2.5-7B-Instruct");
        b.add_u32("qwen2.block_count", 28);
        b.add_u32("qwen2.embedding_length", 3584);
        b.add_u32("qwen2.feed_forward_length", 18944);
        b.add_u32("qwen2.attention.head_count", 28);
        b.add_u32("qwen2.attention.head_count_kv", 4);
        b.add_u32("qwen2.context_length", 32768);
        b.add_str("tokenizer.chat_template", "<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n");
        b.add_dummy_vocab(512);
        b.add_tensor("blk.0.attn_q.weight", {3584, 3584}, QuantType::Q4_K);
        b.add_tensor("blk.0.attn_q.bias", {3584}, QuantType::F32);
        b.add_tensor("blk.0.attn_k.weight", {3584, 512}, QuantType::Q4_K);
        b.add_tensor("blk.0.attn_v.weight", {3584, 512}, QuantType::Q4_K);
        CHECK(b.write(p_qwen25), "write qwen25_7b.gguf");
    }

    // 4. Build synthetic Llama 3.1 8B
    {
        SyntheticGgufBuilder b;
        b.add_str("general.architecture", "llama");
        b.add_str("general.name", "Llama-3.1-8B-Instruct");
        b.add_u32("llama.block_count", 32);
        b.add_u32("llama.embedding_length", 4096);
        b.add_u32("llama.feed_forward_length", 14336);
        b.add_u32("llama.attention.head_count", 32);
        b.add_u32("llama.attention.head_count_kv", 8);
        b.add_u32("llama.context_length", 131072);
        b.add_str("tokenizer.chat_template", "<|start_header_id|>user<|end_header_id|>\n{{content}}<|eot_id|>");
        b.add_dummy_vocab(512);
        b.add_tensor("blk.0.attn_q.weight", {4096, 4096}, QuantType::Q4_K);
        b.add_tensor("blk.0.attn_k.weight", {4096, 1024}, QuantType::Q4_K);
        b.add_tensor("blk.0.attn_v.weight", {4096, 1024}, QuantType::Q4_K);
        CHECK(b.write(p_llama3), "write llama3_8b.gguf");
    }

    // 5. Build synthetic Phi-3.5 Mini
    {
        SyntheticGgufBuilder b;
        b.add_str("general.architecture", "phi3");
        b.add_str("general.name", "Phi-3.5-mini-instruct");
        b.add_u32("phi3.block_count", 32);
        b.add_u32("phi3.embedding_length", 3072);
        b.add_u32("phi3.feed_forward_length", 8192);
        b.add_u32("phi3.attention.head_count", 32);
        b.add_u32("phi3.attention.head_count_kv", 32);
        b.add_u32("phi3.context_length", 131072);
        b.add_str("tokenizer.chat_template", "<|user|>\n{{content}}<|end|>\n<|assistant|>\n");
        b.add_dummy_vocab(512);
        b.add_tensor("blk.0.attn_q.weight", {3072, 3072}, QuantType::Q4_0);
        CHECK(b.write(p_phi3), "write phi3_mini.gguf");
    }

    // 6. Write corrupt/malicious GGUF header (verifies safe bounds rejection)
    {
        std::ofstream f(p_corrupt, std::ios::binary);
        const char bad[16] = "NOT_A_GGUF_FILE";
        f.write(bad, sizeof(bad));
    }

    // ---- Test A: Zero-mmap Header DNA Probe accuracy ----
    std::string err;
    ModelFingerprint fp_e2b, fp_26b, fp_qwen, fp_llama, fp_phi, fp_bad;

    CHECK(HavenModelIntelligence::probe_gguf_header(p_g4_e2b, fp_e2b, err), "probe E2B: " + err);
    CHECK(fp_e2b.arch_family == ArchFamily::GEMMA4, "E2B arch_family == GEMMA4");
    CHECK(fp_e2b.has_pleias && fp_e2b.has_swa && fp_e2b.has_shared_kv && fp_e2b.has_turbo_bake, "E2B traits");
    CHECK(!fp_e2b.has_moe && !fp_e2b.has_vision_encoder, "E2B has NO MoE and NO Vision");
    CHECK(fp_e2b.chat_template_family == ChatTemplateFamily::GEMMA4_TURN && fp_e2b.supports_thinking_channel, "E2B chat template");
    std::cout << "[PASS 1/6] Gemma 4.5 E2B Turbo DNA probed in " << fp_e2b.probe_time_ms << " ms\n";

    CHECK(HavenModelIntelligence::probe_gguf_header(p_g4_26b, fp_26b, err), "probe 26B: " + err);
    CHECK(fp_26b.arch_family == ArchFamily::GEMMA4, "26B arch_family == GEMMA4");
    CHECK(fp_26b.has_moe && fp_26b.expert_count == 128 && fp_26b.fused_moe_gate_up, "26B 128-expert fused MoE");
    CHECK(fp_26b.has_k_eq_v && fp_26b.has_turbo_bake && fp_26b.has_vision_encoder && fp_26b.vision_tensor_count == 2, "26B k_eq_v + Turbo + Vision");
    CHECK(!fp_26b.has_pleias, "26B has NO Pleias");
    std::cout << "[PASS 2/6] Gemma 4.5 26B-A4B MoE + SigLIP DNA probed in " << fp_26b.probe_time_ms << " ms\n";

    CHECK(HavenModelIntelligence::probe_gguf_header(p_qwen25, fp_qwen, err), "probe Qwen: " + err);
    CHECK(fp_qwen.arch_family == ArchFamily::QWEN && fp_qwen.has_qkv_bias, "Qwen QKV bias detected");
    CHECK(!fp_qwen.has_moe && !fp_qwen.has_turbo_bake && !fp_qwen.has_vision_encoder, "Qwen dense only");
    CHECK(fp_qwen.chat_template_family == ChatTemplateFamily::CHATML, "Qwen ChatML template");
    CHECK(fp_qwen.dominant_quant == QuantType::Q4_K, "Qwen dominant quant Q4_K");

    CHECK(HavenModelIntelligence::probe_gguf_header(p_llama3, fp_llama, err), "probe Llama: " + err);
    CHECK(fp_llama.arch_family == ArchFamily::LLAMA && fp_llama.chat_template_family == ChatTemplateFamily::LLAMA3_HEADER, "Llama 3 DNA");

    CHECK(HavenModelIntelligence::probe_gguf_header(p_phi3, fp_phi, err), "probe Phi3: " + err);
    CHECK(fp_phi.arch_family == ArchFamily::PHI && fp_phi.chat_template_family == ChatTemplateFamily::PHI3_TAG, "Phi-3 DNA");
    CHECK(!HavenModelIntelligence::probe_gguf_header(p_corrupt, fp_bad, err), "Corrupt GGUF safely rejected");
    std::cout << "[PASS 3/6] Qwen 2.5, Llama 3.1, Phi-3.5 DNA + Corrupt GGUF bounds rejection verified\n";

    // ---- Test B: Simulated L4 24GB VRAM Auto-Routing vs VRAM Overflow Fallback ----
    HardwareProfile fake_l4;
    fake_l4.cpu_logical_cores = 8;
    fake_l4.has_avx2 = true;
    fake_l4.total_ram_bytes = 32ULL << 30;
    fake_l4.available_ram_bytes = 28ULL << 30;
    fake_l4.cuda_compiled = true;
    fake_l4.cuda_device_count = 1;
    fake_l4.cuda_total_vram_bytes = 22ULL << 30;
    fake_l4.cuda_device_summary = "NVIDIA L4 (sm_89, 22528 MiB VRAM, 58 SMs)";

    auto plan_fit = HavenModelIntelligence::plan_execution(fp_26b, fake_l4);
    CHECK(plan_fit.selected_backend == BackendKind::CUDA_NATIVE, "26B fits on L4 -> auto CUDA");
    CHECK(plan_fit.selected_arch_module == "haven_arch_gemma4", "26B selects haven_arch_gemma4");
    CHECK(plan_fit.required_capability_modules.size() == 3, "26B requires moe + turbo + vision");

    ModelFingerprint fp_huge = fp_llama;
    fp_huge.llm_weight_bytes = 42ULL << 30; // 42 GiB model exceeds 22 GiB L4 VRAM
    auto plan_overflow = HavenModelIntelligence::plan_execution(fp_huge, fake_l4);
    CHECK(plan_overflow.selected_backend == BackendKind::CPU_AVX2, "42 GiB model overflows 22 GiB L4 -> auto CPU_AVX2");
    CHECK(plan_overflow.selected_arch_module == "haven_arch_llama", "Llama selects haven_arch_llama");
    std::cout << "[PASS 4/6] VRAM-aware CUDA vs AVX2 auto-routing verified\n";

    // ---- Test C: Live Dynamic .dll/.so Microkernel Hot-Load & Model-Switch Hot-Unload ----
    HavenModuleLoader loader;
    auto avail = loader.index_available_modules();
    std::cout << "  Discovered " << avail.size() << " dynamic microkernel modules on disk.\n";
    CHECK(avail.size() >= 10, "All 10 microkernel modules must be built and discoverable in modules/");

    HardwareProfile real_hw = HavenModelIntelligence::probe_hardware();
    std::string log;

    // Load Model 1: Gemma 4.5 26B MoE + Turbo + Vision
    auto plan_26b = HavenModelIntelligence::plan_execution(fp_26b, real_hw);
    CHECK(loader.activate_plan(plan_26b, log), "activate 26B plan:\n" + log);
    CHECK(plan_26b.arch_contract_verified, "Gemma 4 C-ABI arch contract verified");
    CHECK(plan_26b.resolved_arch_spec.ffn_activation == 1u, "Gemma 4 GeGLU activation");
    CHECK(std::fabs(plan_26b.resolved_arch_spec.attention_scale - 1.0f) < 1e-6f, "Gemma 4 attn_scale == 1.0");
    CHECK(std::fabs(plan_26b.resolved_arch_spec.final_logit_softcap - 30.0f) < 1e-6f, "Gemma 4 softcap == 30.0");
    CHECK(plan_26b.resolved_arch_spec.use_proportional_rope == 1u && plan_26b.resolved_arch_spec.use_v_unit_rmsnorm == 1u, "Gemma 4 propRoPE + unit V-norm");
    CHECK(loader.is_module_active("haven_arch_gemma4"), "gemma4 loaded");
    CHECK(loader.is_module_active("haven_cap_moe"), "cap_moe loaded");
    CHECK(loader.is_module_active("haven_cap_turbo"), "cap_turbo loaded");
    CHECK(loader.is_module_active("haven_cap_vision"), "cap_vision loaded");
    CHECK(!loader.is_module_active("haven_arch_qwen") && !loader.is_module_active("haven_arch_llama") && !loader.is_module_active("haven_arch_phi"),
          "Unused architectures MUST NOT be loaded!");
    CHECK(loader.get_active_module_count() == 5, "Exactly 5 modules resident for 26B MoE+Turbo+Vision");
    std::cout << "[PASS 5/6] Dynamic hot-load verified for Gemma 4.5 26B MoE+Turbo+Vision (5 active / 5 skipped)\n";

    // Hot-Swap to Model 2: Qwen 2.5 7B (MUST unload gemma4, moe, turbo, vision & load ONLY qwen + backend!)
    auto plan_qwen = HavenModelIntelligence::plan_execution(fp_qwen, real_hw);
    CHECK(loader.activate_plan(plan_qwen, log), "activate Qwen plan:\n" + log);
    CHECK(plan_qwen.arch_contract_verified, "Qwen C-ABI arch contract verified");
    CHECK(plan_qwen.resolved_arch_spec.ffn_activation == 0u, "Qwen SiLU activation");
    CHECK(std::fabs(plan_qwen.resolved_arch_spec.final_logit_softcap) < 1e-6f, "Qwen zero softcap");
    CHECK(std::string(plan_qwen.resolved_arch_spec.chat_template_hint) == "chatml", "Qwen ChatML hint");
    CHECK(loader.is_module_active("haven_arch_qwen"), "qwen loaded");
    CHECK(!loader.is_module_active("haven_arch_gemma4"), "gemma4 UNLOADED on swap!");
    CHECK(!loader.is_module_active("haven_cap_moe"), "cap_moe UNLOADED on swap!");
    CHECK(!loader.is_module_active("haven_cap_turbo"), "cap_turbo UNLOADED on swap!");
    CHECK(!loader.is_module_active("haven_cap_vision"), "cap_vision UNLOADED on swap!");
    CHECK(loader.get_active_module_count() == 2, "Exactly 2 modules resident for dense Qwen 2.5!");

    // Hot-Swap to Model 3 (Llama) and Model 4 (Phi), then clean full unload
    auto plan_llama = HavenModelIntelligence::plan_execution(fp_llama, real_hw);
    CHECK(loader.activate_plan(plan_llama, log), "activate Llama plan");
    CHECK(plan_llama.arch_contract_verified && loader.is_module_active("haven_arch_llama") && !loader.is_module_active("haven_arch_qwen"), "Llama swap verified");

    auto plan_phi = HavenModelIntelligence::plan_execution(fp_phi, real_hw);
    CHECK(loader.activate_plan(plan_phi, log), "activate Phi plan");
    CHECK(plan_phi.arch_contract_verified && loader.is_module_active("haven_arch_phi") && !loader.is_module_active("haven_arch_llama"), "Phi swap verified");

    // Verify JSON serialization is non-empty and well-formed
    std::string js = HavenModelIntelligence::to_json(plan_phi, loader.get_active_module_ids());
    CHECK(js.find("\"selected_arch_module\":\"haven_arch_phi\"") != std::string::npos, "JSON export contains arch module");

    loader.unload_all();
    CHECK(loader.get_active_module_count() == 0, "Zero modules left resident after unload_all()");
    std::cout << "[PASS 6/6] Hot-swap auto-unload verified (Gemma4 26B -> Qwen2.5 -> Llama3 -> Phi3 -> 0 resident)\n";

    std::error_code ec;
    std::filesystem::remove_all(tmp_dir, ec);
    std::cout << "\n✅ ALL MODEL INTELLIGENCE & MICROKERNEL MODULE TESTS PASSED 100%!\n";
    return 0;
}
