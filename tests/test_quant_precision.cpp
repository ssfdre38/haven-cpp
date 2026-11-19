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

    // 3. Test Q4_K precision for blk.0.attn_q.weight
    const auto* q_tensor = engine.get_config().num_layers > 0 ? engine.get_layer_tensors(0).attn_q : nullptr;
    if (q_tensor && q_tensor->type == haven::QuantType::Q4_K) {
        const auto* q4_blocks = reinterpret_cast<const haven::block_q4_K*>(q_tensor->data);
        float dot_q4 = haven::Avx2Math::vec_dot_q4_K(2560, q4_blocks, test_vec.data());
        std::vector<float> q4_fp32(2560, 0.0f);
        haven::Avx2Math::dequantize_row(q4_fp32.data(), *q_tensor, 0, 2560);
        float dot_q4_fp32 = 0.0f;
        for (int i = 0; i < 2560; ++i) dot_q4_fp32 += q4_fp32[i] * test_vec[i];

        std::cout << "\n• Q4_K Kernel (blk.0.attn_q.weight Row 0):\n";
        std::cout << "   - Q4_K Kernel Dot Product: " << dot_q4 << "\n";
        std::cout << "   - FP32 Dequant Dot Product: " << dot_q4_fp32 << "\n";
        std::cout << "   - Absolute Difference: " << std::abs(dot_q4 - dot_q4_fp32) << "\n";
        if (std::abs(dot_q4 - dot_q4_fp32) < 1e-3f) {
            std::cout << "✓ SUCCESS: Q4_K AVX2 Kernel matches FP32 dequantization perfectly!\n";
        }
    }

    std::cout << "==================================================================\n";
    return 0;
}
