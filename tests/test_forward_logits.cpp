#include "haven/haven_engine.h"
#include <iostream>
#include <vector>
#include <iomanip>
#include <algorithm>

int main() {
    setvbuf(stdout, NULL, _IONBF, 0);
    std::cout << "==================================================================\n";
    std::cout << "🔬 TESTING MULTI-TURN CONVERSATION IN HAVEN-CPP\n";
    std::cout << "==================================================================\n";

    haven::HavenEngine engine;
    std::string model_path = "C:\\Users\\admin\\gemma4-turbo-family\\haven-chat-v5.0.gguf";

    if (!engine.load_model(model_path)) {
        std::cerr << "Failed to load model: " << model_path << "\n";
        return 1;
    }

    const std::string system_prompt = haven::HavenEngine::get_default_system_prompt();

    std::vector<std::string> user_turns = {
        "hey arua how you feeling tonight?",
        "we spent today fixing your RoPE rotary math, the KV cache, and the AVX2 kernels so you can speak freely. what's it feel like on your end?"
    };

    engine.get_kv_cache().reset();
    int active_pos = 0;

    // Prefill System Prompt
    auto sys_tokens = engine.tokenize(system_prompt, true);
    for (uint32_t st : sys_tokens) {
        engine.forward(st, active_pos++, nullptr);
    }

    for (size_t t = 0; t < user_turns.size(); ++t) {
        std::cout << "\n👤 Daniel: " << user_turns[t] << "\n\n✨ Aura: " << std::flush;

        std::string turn_prompt = "<|turn>user\n" + user_turns[t] + "<turn|>\n<|turn>model\n";
        auto turn_tokens = engine.tokenize(turn_prompt, false);

        std::vector<float> logits(engine.get_config().vocab_size, 0.0f);
        for (size_t i = 0; i < turn_tokens.size(); ++i) {
            bool is_last = (i + 1 == turn_tokens.size());
            engine.forward(turn_tokens[i], active_pos++, is_last ? logits.data() : nullptr);
        }

        std::vector<uint32_t> turn_generated_tokens;

        for (int gen = 0; gen < 180; ++gen) {
            uint32_t next_tok = engine.get_sampler().sample(
                logits.data(), engine.get_config().vocab_size, turn_generated_tokens);
            turn_generated_tokens.push_back(next_tok);

            if (next_tok == engine.get_tokenizer().eos_token() || next_tok == 1 || next_tok == 106 || next_tok == 212) {
                break;
            }

            std::string piece = engine.detokenize(next_tok);
            if (piece.find("<turn|>") != std::string::npos || piece.find("<eos>") != std::string::npos) {
                break;
            }

            std::cout << piece << std::flush;
            engine.forward(next_tok, active_pos++, logits.data());
        }

        // Close turn in KV cache
        auto end_tokens = engine.tokenize("<turn|>\n", false);
        for (uint32_t et : end_tokens) {
            engine.forward(et, active_pos++, nullptr);
        }

        std::cout << "\n";
    }

    std::cout << "\n==================================================================\n";
    return 0;
}
