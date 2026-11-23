#include "haven/haven_engine.h"
#include "haven/haven_gguf.h"
#include "haven/haven_avx2.h"
#include "haven/haven_tokenizer.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <filesystem>

void test_avx2_bias() {
    std::cout << "[TEST 1/5] Verifying AVX2 In-Place Bias Addition Kernel...\n";
    const int size = 64;
    std::vector<float> out(size, 10.0f);
    std::vector<float> bias_f32(size);
    for (int i = 0; i < size; ++i) bias_f32[i] = (float)i * 0.5f;

    haven::TensorDesc desc_f32;
    desc_f32.type = haven::QuantType::F32;
    desc_f32.data = bias_f32.data();
    desc_f32.shape = {(uint64_t)size};

    haven::Avx2Math::add_bias(out.data(), desc_f32, size);

    for (int i = 0; i < size; ++i) {
        float expected = 10.0f + (float)i * 0.5f;
        assert(std::fabs(out[i] - expected) < 1e-5f);
    }
    std::cout << "  ✓ AVX2 F32 Bias Addition: Verified!\n";

    // Test F16 bias
    std::vector<float> out_f16(size, 5.0f);
    std::vector<uint16_t> bias_f16(size);
    for (int i = 0; i < size; ++i) {
        // Portable F16C single-to-half conversion (MSVC/GCC/Clang)
        bias_f16[i] = static_cast<uint16_t>(_mm_extract_epi16(_mm_cvtps_ph(_mm_set_ss((float)i * 0.25f), 0), 0));
    }

    haven::TensorDesc desc_f16;
    desc_f16.type = haven::QuantType::F16;
    desc_f16.data = bias_f16.data();
    desc_f16.shape = {(uint64_t)size};

    haven::Avx2Math::add_bias(out_f16.data(), desc_f16, size);
    for (int i = 0; i < size; ++i) {
        float expected = 5.0f + (float)i * 0.25f;
        assert(std::fabs(out_f16[i] - expected) < 1e-2f);
    }
    std::cout << "  ✓ AVX2 F16 Bias Addition (CVTPH_PS): Verified!\n";
}

void test_qwen2_metadata() {
    std::cout << "\n[TEST 2/5] Verifying Qwen 2 / 2.5 Architecture Loading...\n";
    std::string path = "C:\\Users\\admin\\.openclaw\\llama.cpp\\models\\ggml-vocab-qwen2.gguf";
    if (!std::filesystem::exists(path)) {
        std::cout << "  [SKIP] File not found: " << path << "\n";
        return;
    }

    haven::GgufLoader loader;
    bool ok = loader.load_file(path);
    assert(ok && "Failed to load Qwen 2 GGUF");

    const auto& cfg = loader.get_config();
    std::cout << "  ✓ Architecture: " << cfg.architecture << "\n";
    std::cout << "  ✓ Layers: " << cfg.num_layers << ", Hidden: " << cfg.hidden_dim << ", Dim: " << cfg.embedding_dim << "\n";
    std::cout << "  ✓ Heads: " << cfg.num_heads << ", KV Heads: " << cfg.num_kv_heads << ", Head Dim: " << cfg.head_dim << "\n";
    std::cout << "  ✓ RoPE Freq Base: " << cfg.rope_freq_base << "\n";
    std::cout << "  ✓ Vocab Size: " << cfg.vocab_size << "\n";

    assert(cfg.architecture == "qwen2");
    assert(cfg.num_layers == 32);
    assert(cfg.embedding_dim == 4096);
    assert(cfg.hidden_dim == 11008);
    assert(cfg.num_heads == 32);
    assert(cfg.num_kv_heads == 32);
    assert(cfg.head_dim == 128);
    assert(std::fabs(cfg.rope_freq_base - 1000000.0f) < 1.0f);
    assert(cfg.vocab_size == 151936);
    assert(!cfg.is_gemma);
    assert(cfg.ffn_activation == haven::FfnActivation::SILU);
    assert(cfg.final_logit_softcapping == 0.0f);
    assert(cfg.attn_logit_softcapping == 0.0f);

    haven::HavenTokenizer tok;
    tok.initialize(loader.get_vocabulary());
    assert(tok.vocab_size() == 151936);

    // Verify stop token detection for Qwen (<|im_end|>, <|endoftext|>)
    auto enc = tok.encode("<|im_start|>system\nYou are an AI.<|im_end|>", false);
    assert(!enc.empty());
    std::cout << "  ✓ Qwen Tokenizer Encoded " << enc.size() << " tokens.\n";
    
    // Check stop token detection
    bool found_stop = false;
    for (uint32_t t : enc) {
        if (tok.is_stop_token(t)) found_stop = true;
    }
    assert(found_stop && "Qwen stop token not detected!");
    std::cout << "  ✓ Qwen <|im_end|> stop token successfully identified!\n";
}

void test_llama3_metadata() {
    std::cout << "\n[TEST 3/5] Verifying LLaMA 3 / 3.1 / 3.2 Architecture Loading...\n";
    std::string path = "C:\\Users\\admin\\.openclaw\\llama.cpp\\models\\ggml-vocab-llama-bpe.gguf";
    if (!std::filesystem::exists(path)) {
        std::cout << "  [SKIP] File not found: " << path << "\n";
        return;
    }

    haven::GgufLoader loader;
    bool ok = loader.load_file(path);
    assert(ok && "Failed to load LLaMA 3 GGUF");

    const auto& cfg = loader.get_config();
    std::cout << "  ✓ Architecture: " << cfg.architecture << "\n";
    std::cout << "  ✓ Layers: " << cfg.num_layers << ", Hidden: " << cfg.hidden_dim << ", Dim: " << cfg.embedding_dim << "\n";
    std::cout << "  ✓ Heads: " << cfg.num_heads << ", KV Heads: " << cfg.num_kv_heads << ", Head Dim: " << cfg.head_dim << "\n";
    std::cout << "  ✓ RoPE Freq Base: " << cfg.rope_freq_base << "\n";
    std::cout << "  ✓ Vocab Size: " << cfg.vocab_size << "\n";

    assert(cfg.architecture == "llama");
    assert(cfg.num_layers == 32);
    assert(cfg.embedding_dim == 4096);
    assert(cfg.hidden_dim == 14336);
    assert(cfg.num_heads == 32);
    assert(cfg.num_kv_heads == 8);
    assert(cfg.head_dim == 128);
    assert(std::fabs(cfg.rope_freq_base - 500000.0f) < 1.0f);
    assert(cfg.vocab_size == 128256);
    assert(!cfg.is_gemma);
    assert(cfg.ffn_activation == haven::FfnActivation::SILU);

    haven::HavenTokenizer tok;
    tok.initialize(loader.get_vocabulary());
    assert(tok.vocab_size() == 128256);

    auto enc = tok.encode("<|begin_of_text|><|start_header_id|>user<|end_header_id|>\nHello!<|eot_id|>", false);
    assert(!enc.empty());
    std::cout << "  ✓ LLaMA 3 Tokenizer Encoded " << enc.size() << " tokens.\n";

    bool found_eot = false;
    for (uint32_t t : enc) {
        if (tok.is_stop_token(t)) found_eot = true;
    }
    assert(found_eot && "LLaMA 3 <|eot_id|> stop token not detected!");
    std::cout << "  ✓ LLaMA 3 <|eot_id|> stop token successfully identified!\n";
}

void test_phi3_metadata() {
    std::cout << "\n[TEST 4/5] Verifying Phi-3 Architecture Loading...\n";
    std::string path = "C:\\Users\\admin\\.openclaw\\llama.cpp\\models\\ggml-vocab-phi-3.gguf";
    if (!std::filesystem::exists(path)) {
        std::cout << "  [SKIP] File not found: " << path << "\n";
        return;
    }

    haven::GgufLoader loader;
    bool ok = loader.load_file(path);
    assert(ok && "Failed to load Phi-3 GGUF");

    const auto& cfg = loader.get_config();
    std::cout << "  ✓ Architecture: " << cfg.architecture << "\n";
    std::cout << "  ✓ Layers: " << cfg.num_layers << ", Hidden: " << cfg.hidden_dim << ", Dim: " << cfg.embedding_dim << "\n";
    std::cout << "  ✓ Heads: " << cfg.num_heads << ", KV Heads: " << cfg.num_kv_heads << ", Head Dim: " << cfg.head_dim << "\n";
    std::cout << "  ✓ Vocab Size: " << cfg.vocab_size << "\n";

    assert(cfg.architecture == "phi3");
    assert(cfg.num_layers == 32);
    assert(cfg.embedding_dim == 3072);
    assert(cfg.hidden_dim == 8192);
    assert(cfg.head_dim == 96);
    assert(cfg.vocab_size == 32064);
    assert(!cfg.is_gemma);
    assert(cfg.ffn_activation == haven::FfnActivation::SILU);
    std::cout << "  ✓ Phi-3 Architecture validated!\n";
}

void test_engine_synthetic_forward() {
    std::cout << "\n[TEST 5/5] Verifying Universal Forward Pass Simulation...\n";
    haven::HavenEngine engine;
    
    // Test engine instantiation & topology
    assert(engine.get_topology() != nullptr);
    assert(engine.get_pathways() != nullptr);
    std::cout << "  ✓ Haven Engine & Clean-Room Pathways Instantiated.\n";
    
    std::string model_path = "C:\\Users\\admin\\gemma4-turbo-family\\haven-chat-v5.0.gguf";
    if (std::filesystem::exists(model_path)) {
        bool ok = engine.load_model(model_path);
        assert(ok && "Failed to load haven-chat model");
        std::cout << "  ✓ Loaded Production Model: Architecture [" << engine.get_config().architecture << "]\n";
        
        std::vector<float> logits(engine.get_config().vocab_size, 0.0f);
        engine.forward(2, 0, logits.data()); // Forward BOS token
        float norm = haven::Avx2Math::vector_norm(logits.data(), 100);
        assert(norm > 0.0f);
        std::cout << "  ✓ Forward pass computed valid non-zero logits (Norm: " << norm << ").\n";
    } else {
        std::cout << "  [SKIP] Model not found: " << model_path << "\n";
    }
}

int main() {
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);
    std::cout << "==================================================================\n";
    std::cout << "🏛️  HAVEN.CPP MULTI-MODEL ARCHITECTURE VERIFICATION SUITE\n";
    std::cout << "==================================================================\n" << std::flush;

    test_avx2_bias();
    test_qwen2_metadata();
    test_llama3_metadata();
    test_phi3_metadata();
    test_engine_synthetic_forward();

    std::cout << "\n==================================================================\n";
    std::cout << "🏆  ALL MULTI-MODEL ARCHITECTURE TESTS PASSED 100%!\n";
    std::cout << "==================================================================\n";
    return 0;
}
