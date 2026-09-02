#include "haven/haven_tokenizer.h"
#include <sstream>
#include <iostream>
#include <iomanip>

namespace haven {

HavenTokenizer::HavenTokenizer() = default;
HavenTokenizer::~HavenTokenizer() = default;

void HavenTokenizer::initialize(const std::vector<std::string>& vocab) {
    vocab_ = vocab;
    token_to_id_.clear();
    token_to_id_.reserve(vocab.size());

    bos_token_ = 2;
    eos_token_ = 1;
    pad_token_ = 0;
    unk_token_ = 3;

    for (uint32_t i = 0; i < (uint32_t)vocab.size(); ++i) {
        if (token_to_id_.find(vocab[i]) == token_to_id_.end()) {
            token_to_id_[vocab[i]] = i;
        }
        if (vocab[i] == "<bos>") bos_token_ = i;
        else if (vocab[i] == "<eos>") eos_token_ = i;
        else if (vocab[i] == "<pad>") pad_token_ = i;
        else if (vocab[i] == "<unk>") unk_token_ = i;
    }
}

std::vector<uint32_t> HavenTokenizer::encode(const std::string& text, bool add_bos) const {
    std::vector<uint32_t> tokens;
    if (vocab_.empty() || text.empty()) return tokens;

    if (add_bos) {
        tokens.push_back(bos_token_);
    }

    // List of priority control tokens in Gemma 4
    static const std::vector<std::string> control_tokens = {
        "<start_of_turn>", "<end_of_turn>", "<bos>", "<eos>", "<pad>",
        "<|thought>", "</thought>", "<|tool_call|>", "<|tool_response|>"
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

        // Replace regular spaces with SentencePiece space indicator (\xe2\x96\x81 =  )
        std::string normalized;
        normalized.reserve(segment.length() * 2);
        for (size_t i = 0; i < segment.length(); ++i) {
            if (segment[i] == ' ') {
                normalized += "\xe2\x96\x81";
            } else {
                normalized += segment[i];
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
    if (raw == "<bos>" || raw == "<eos>" || raw == "<start_of_turn>" || raw == "<end_of_turn>" || raw == "<pad>") {
        return "";
    }

    // Parse byte fallback tokens <0xXX>
    if (raw.length() == 6 && raw.rfind("<0x", 0) == 0 && raw[5] == '>') {
        try {
            unsigned int byte_val = 0;
            std::stringstream ss;
            ss << std::hex << raw.substr(3, 2);
            ss >> byte_val;
            return std::string(1, (char)(uint8_t)byte_val);
        } catch (...) {
            return "";
        }
    }

    // Replace SentencePiece space character (\xe2\x96\x81) with standard ASCII space
    std::string out;
    size_t i = 0;
    while (i < raw.length()) {
        if (i + 2 < raw.length() && 
            (unsigned char)raw[i] == 0xe2 && 
            (unsigned char)raw[i+1] == 0x96 && 
            (unsigned char)raw[i+2] == 0x81) 
        {
            out += " ";
            i += 3;
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
    return "<start_of_turn>" + role + "\n" + content + "<end_of_turn>\n";
}

} // namespace haven
