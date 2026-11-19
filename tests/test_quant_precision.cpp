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
                }
            }
        }
    }

    std::cout << "==================================================================\n";
    return 0;
}
