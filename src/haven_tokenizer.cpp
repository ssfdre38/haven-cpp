#include "haven/haven_tokenizer.h"
#include <sstream>
#include <iostream>
#include <iomanip>

namespace haven {

HavenTokenizer::HavenTokenizer() = default;
HavenTokenizer::~HavenTokenizer() = default;

void HavenTokenizer::initialize(const std::vector<std::string>& vocab, const ModelConfig* cfg) {
    vocab_ = vocab;
    token_to_id_.clear();
    token_to_id_.reserve(vocab.size());
    stop_tokens_.clear();

    bos_token_ = 2;
    eos_token_ = 1;
    pad_token_ = 0;
    unk_token_ = 3;
    add_space_prefix_ = true;
    is_gpt2_bpe_ = false;

    bool has_sp_space = false;
    bool has_gpt2_space = false;

    // Priority flags prevent literal vocab tokens like Gemma 4's "<s>" (203) / "</s>" (212)
    // from overwriting canonical "<bos>" (2) / "<eos>" (1)!
    bool found_primary_bos = false;
    bool found_any_bos     = false;
    bool found_primary_eos = false;
    bool found_any_eos     = false;

    for (uint32_t i = 0; i < (uint32_t)vocab.size(); ++i) {
        if (token_to_id_.find(vocab[i]) == token_to_id_.end()) {
            token_to_id_[vocab[i]] = i;
        }
        if (!has_sp_space && vocab[i].find("\xe2\x96\x81") != std::string::npos) {
            has_sp_space = true;
        }
        if (!has_gpt2_space && vocab[i].find("\xc4\xa0") != std::string::npos) {
            has_gpt2_space = true;
        }
        if (vocab[i] == "<bos>" || vocab[i] == "<|begin_of_text|>") {
            if (!found_primary_bos) { bos_token_ = i; found_primary_bos = true; found_any_bos = true; }
        } else if (vocab[i] == "<s>") {
            if (!found_any_bos) { bos_token_ = i; found_any_bos = true; }
        } else if (vocab[i] == "<eos>" || vocab[i] == "<|end_of_text|>" || vocab[i] == "<|endoftext|>") {
            if (!found_primary_eos) { eos_token_ = i; found_primary_eos = true; found_any_eos = true; }
        } else if (vocab[i] == "</s>") {
            if (!found_any_eos) { eos_token_ = i; found_any_eos = true; }
        } else if (vocab[i] == "<pad>") {
            pad_token_ = i;
        } else if (vocab[i] == "<unk>") {
            unk_token_ = i;
        }

        // Register universal stop tokens (Gemma, Llama 3, Qwen 2/2.5/3, Phi-3/4, Mistral)
        if (vocab[i] == "<eos>" || vocab[i] == "<turn|>" || vocab[i] == "<|turn|>" ||
            vocab[i] == "<end_of_turn>" || vocab[i] == "</end_of_turn>" ||
            vocab[i] == "<|eot_id|>" || vocab[i] == "<|end_of_text|>" ||
            vocab[i] == "<|im_end|>" || vocab[i] == "<|endoftext|>" ||
            vocab[i] == "<|end|>" ||
            (vocab[i] == "</s>" && (!cfg || !cfg->is_gemma))) {
            stop_tokens_.insert(i);
        }
    }

    // Auto-detect GPT-2 byte-level BPE (Llama 3, Qwen 2/2.5/3): uses Ġ (\xc4\xa0) and Ċ (\xc4\x8a)
    is_gpt2_bpe_ = has_gpt2_space && !has_sp_space;
    if (is_gpt2_bpe_) {
        add_space_prefix_ = false;
    }

    // Honor authoritative GGUF header tokenizer metadata whenever present
    if (cfg) {
        if (cfg->has_bos_token_id && cfg->bos_token_id < vocab.size()) bos_token_ = cfg->bos_token_id;
        if (cfg->has_eos_token_id && cfg->eos_token_id < vocab.size()) eos_token_ = cfg->eos_token_id;
        if (cfg->has_pad_token_id && cfg->pad_token_id < vocab.size()) pad_token_ = cfg->pad_token_id;
        if (cfg->has_unk_token_id && cfg->unk_token_id < vocab.size()) unk_token_ = cfg->unk_token_id;
        if (cfg->has_add_space_prefix) {
            add_space_prefix_ = cfg->add_space_prefix;
        } else if (cfg->is_gemma) {
            add_space_prefix_ = false; // Gemma 2/3/4 BPE/SentencePiece does not prepend dummy space
        }
    }

    stop_tokens_.insert(eos_token_);
}

std::vector<uint32_t> HavenTokenizer::encode(const std::string& text, bool add_bos) const {
    std::vector<uint32_t> tokens;
    if (vocab_.empty()) return tokens;

    if (add_bos) {
        tokens.push_back(bos_token_);
    }
    if (text.empty()) return tokens;

    // List of priority control tokens across architectures (Gemma, LLaMA 3, Qwen 2.5, Phi-3, Mistral)
    static const std::vector<std::string> control_tokens = {
        "<|turn>", "<turn|>", "<bos>", "<eos>", "<pad>",
        "<|think|>", "<|channel>", "<channel|>",
        "<|tool_call>", "<tool_call|>", "<|tool_response>", "<tool_response|>",
        "<start_of_turn>", "<end_of_turn>", "</end_of_turn>",
        "<|begin_of_text|>", "<|end_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>",
        "<|im_start|>", "<|im_end|>", "<|endoftext|>",
        "<|system|>", "<|user|>", "<|assistant|>", "<|end|>",
        "<s>", "</s>"
    };

    size_t pos = 0;
    const size_t len = text.length();

    while (pos < len) {
        // Check for control tokens at current position first
        bool matched_control = false;
        for (const auto& ct : control_tokens) {
            if (pos + ct.length() <= len && text.compare(pos, ct.length(), ct) == 0) {
                auto it = token_to_id_.find(ct);
                if (it != token_to_id_.end()) {
                    tokens.push_back(it->second);
                    pos += ct.length();
                    matched_control = true;
                    break;
                }
            }
        }
        if (matched_control) continue;

        // Otherwise, find the next valid control token that exists in vocab
        size_t next_ctrl = len;
        for (const auto& ct : control_tokens) {
            if (token_to_id_.find(ct) != token_to_id_.end()) {
                size_t c_pos = text.find(ct, pos);
                if (c_pos != std::string::npos && c_pos < next_ctrl) {
                    next_ctrl = c_pos;
                }
            }
        }

        // If next_ctrl == pos (unrecognized control token), consume at least 1 character to avoid infinite loop
        if (next_ctrl == pos) {
            next_ctrl = pos + 1;
        }

        std::string segment = text.substr(pos, next_ctrl - pos);
        pos = next_ctrl;

        // Normalize spaces/newlines according to vocabulary style (GPT-2 BPE Ġ/Ċ vs SentencePiece ▁)
        std::string normalized;
        normalized.reserve(segment.length() * 2);
        if (is_gpt2_bpe_) {
            for (size_t i = 0; i < segment.length(); ++i) {
                if (segment[i] == ' ') normalized += "\xc4\xa0";       // Ġ (U+0120)
                else if (segment[i] == '\n') normalized += "\xc4\x8a"; // Ċ (U+010A)
                else if (segment[i] == '\t') normalized += "\xc4\x89"; // ĉ (U+0109)
                else normalized += segment[i];
            }
        } else {
            for (size_t i = 0; i < segment.length(); ++i) {
                if (segment[i] == ' ') {
                    normalized += "\xe2\x96\x81";
                } else {
                    if (add_space_prefix_ && pos == next_ctrl && i == 0 && tokens.size() <= (add_bos ? 1u : 0u) && segment[i] != '\n') {
                        normalized += "\xe2\x96\x81";
                    }
                    normalized += segment[i];
                }
            }
        }

        // Maximal prefix match for segment
        size_t seg_pos = 0;
        const size_t seg_len = normalized.length();

        while (seg_pos < seg_len) {
            size_t max_match_len = 0;
            uint32_t best_token_id = unk_token_;

            size_t try_len = std::min((size_t)64, seg_len - seg_pos);
            for (size_t l = try_len; l >= 1; --l) {
                std::string sub = normalized.substr(seg_pos, l);
                auto it = token_to_id_.find(sub);
                if (it != token_to_id_.end()) {
                    max_match_len = l;
                    best_token_id = it->second;
                    break;
                }
            }

            if (max_match_len > 0) {
                tokens.push_back(best_token_id);
                seg_pos += max_match_len;
            } else {
                uint8_t byte_val = (uint8_t)normalized[seg_pos];
                char hex_buf[16];
                snprintf(hex_buf, sizeof(hex_buf), "<0x%02X>", byte_val);
                auto it = token_to_id_.find(hex_buf);
                if (it != token_to_id_.end()) {
                    tokens.push_back(it->second);
                } else {
                    tokens.push_back(unk_token_);
                }
                seg_pos++;
            }
        }
    }

    return tokens;
}

std::string HavenTokenizer::decode_token(uint32_t token_id) const {
    if (token_id >= vocab_.size()) return "";

    const std::string& raw = vocab_[token_id];

    // Filter control tokens
    if (raw == "<bos>" || raw == "<eos>" || raw == "<|turn>" || raw == "<turn|>" ||
        raw == "<start_of_turn>" || raw == "<end_of_turn>" || raw == "</end_of_turn>" || 
        raw == "<pad>" || raw == "</s>" || raw == "<s>" ||
        raw == "<|begin_of_text|>" || raw == "<|end_of_text|>" ||
        raw == "<|start_header_id|>" || raw == "<|end_header_id|>" || raw == "<|eot_id|>" ||
        raw == "<|im_start|>" || raw == "<|im_end|>" || raw == "<|endoftext|>" ||
        raw == "<|system|>" || raw == "<|user|>" || raw == "<|assistant|>" || raw == "<|end|>") {
        return "";
    }

    // Parse byte fallback tokens <0xXX>
    if (raw.length() == 6 && raw.rfind("<0x", 0) == 0 && raw[5] == '>') {
        unsigned int byte_val = 0;
        std::stringstream ss;
        ss << std::hex << raw.substr(3, 2);
        if ((ss >> byte_val) && ss.eof() && byte_val <= 0xFFu) {
            return std::string(1, static_cast<char>(static_cast<uint8_t>(byte_val)));
        }
    }

    // Replace SentencePiece (▁) and GPT-2 BPE (Ġ/Ċ/ĉ) markers with standard ASCII whitespace
    std::string out;
    out.reserve(raw.length());
    size_t i = 0;
    while (i < raw.length()) {
        if (i + 2 < raw.length() && 
            (unsigned char)raw[i] == 0xe2 && 
            (unsigned char)raw[i+1] == 0x96 && 
            (unsigned char)raw[i+2] == 0x81) 
        {
            out += ' ';
            i += 3;
        } else if (is_gpt2_bpe_ && i + 1 < raw.length() && (unsigned char)raw[i] == 0xc4) {
            if ((unsigned char)raw[i+1] == 0xa0) { out += ' '; i += 2; continue; }
            if ((unsigned char)raw[i+1] == 0x8a) { out += '\n'; i += 2; continue; }
            if ((unsigned char)raw[i+1] == 0x89) { out += '\t'; i += 2; continue; }
            out += raw[i++];
        } else {
            out += raw[i];
            i++;
        }
    }

    return out;
}

std::string HavenTokenizer::decode(const std::vector<uint32_t>& tokens) const {
    std::string result;
    for (uint32_t tok : tokens) {
        result += decode_token(tok);
    }
    return result;
}

std::string HavenTokenizer::format_turn(const std::string& role, const std::string& content) const {
    return "<|turn>" + role + "\n" + content + "<turn|>\n";
}

} // namespace haven
