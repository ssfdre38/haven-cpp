#pragma once

#include "haven_types.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <cstdint>

namespace haven {

class HavenTokenizer {
public:
    HavenTokenizer();
    ~HavenTokenizer();

    // Initializes tokenizer vocabulary and fast reverse-lookup map (honors exact GGUF metadata when cfg provided)
    void initialize(const std::vector<std::string>& vocab, const ModelConfig* cfg = nullptr);

    // Encodes UTF-8 text into token IDs
    std::vector<uint32_t> encode(const std::string& text, bool add_bos = false) const;

    // Decodes a single token ID into a UTF-8 string chunk
    std::string decode_token(uint32_t token_id) const;

    // Decodes a sequence of token IDs into a full UTF-8 string
    std::string decode(const std::vector<uint32_t>& tokens) const;

    // Formats multi-turn conversation template
    std::string format_turn(const std::string& role, const std::string& content) const;

    bool is_stop_token(uint32_t token_id) const {
        return stop_tokens_.find(token_id) != stop_tokens_.end() || token_id == eos_token_;
    }

    uint32_t bos_token() const { return bos_token_; }
    uint32_t eos_token() const { return eos_token_; }
    uint32_t pad_token() const { return pad_token_; }

    size_t vocab_size() const { return vocab_.size(); }

private:
    std::vector<std::string> vocab_;
    std::unordered_map<std::string, uint32_t> token_to_id_;
    std::unordered_set<uint32_t> stop_tokens_;
    uint32_t bos_token_ = 2;
    uint32_t eos_token_ = 1;
    uint32_t pad_token_ = 0;
    uint32_t unk_token_ = 3;
    bool     add_space_prefix_ = true;
    bool     is_gpt2_bpe_ = false;
};

} // namespace haven
