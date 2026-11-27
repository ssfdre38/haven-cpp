#include "haven/haven_intelligence.h"
#include "haven/haven_cuda.h"
#include "haven/haven_quant_spec.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <set>
#include <sstream>
#include <thread>

#ifdef _WIN32
#include <intrin.h>
#else
#include <dlfcn.h>
#include <unistd.h>
#if defined(__x86_64__) || defined(__i386__)
#include <cpuid.h>
#endif
#endif

namespace haven {

namespace {

constexpr uint32_t kGgufMagic = 0x46554747u; // "GGUF"

class BufferedFileReader {
public:
    static constexpr size_t kBufCap = 65536;

    explicit BufferedFileReader(const std::string& path)
        : f_(path, std::ios::binary | std::ios::ate), file_size_(0) {
        if (f_.is_open()) {
            std::streampos sz = f_.tellg();
            if (sz > 0) file_size_ = static_cast<uint64_t>(sz);
            f_.seekg(0, std::ios::beg);
            pos_ = 0;
        }
    }

    bool is_open() const { return f_.is_open() && file_size_ > 0; }
    uint64_t file_size() const { return file_size_; }
    uint64_t tell() const { return pos_; }
    uint64_t remaining() const { return (pos_ <= file_size_) ? (file_size_ - pos_) : 0; }

    bool read_exact(void* dst, uint64_t n) {
        if (n > remaining()) return false;
        if (n == 0) return true;
        uint8_t* out = static_cast<uint8_t*>(dst);
        while (n > 0) {
            size_t avail = buf_end_ - buf_pos_;
            if (avail == 0) {
                if (n >= kBufCap) {
                    f_.read(reinterpret_cast<char*>(out), static_cast<std::streamsize>(n));
                    if (!f_) return false;
                    pos_ += n;
                    return true;
                }
                if (!refill()) return false;
                avail = buf_end_ - buf_pos_;
            }
            size_t take = (n < avail) ? static_cast<size_t>(n) : avail;
            std::memcpy(out, buf_ + buf_pos_, take);
            buf_pos_ += take;
            out += take;
            pos_ += take;
            n -= take;
        }
        return true;
    }

    bool skip(uint64_t n) {
        if (n > remaining()) return false;
        if (n == 0) return true;
        size_t avail = buf_end_ - buf_pos_;
        if (n <= avail) {
            buf_pos_ += static_cast<size_t>(n);
            pos_ += n;
            return true;
        }
        pos_ += n;
        buf_pos_ = 0;
        buf_end_ = 0;
        f_.seekg(static_cast<std::streamoff>(pos_), std::ios::beg);
        return static_cast<bool>(f_);
    }

    template <typename T>
    bool read_pod(T& out) {
        return read_exact(&out, sizeof(T));
    }

    bool read_string(std::string& out, uint64_t max_len = 16ULL * 1024ULL * 1024ULL) {
        uint64_t len = 0;
        if (!read_pod(len)) return false;
        if (len > max_len || len > remaining()) return false;
        out.resize(static_cast<size_t>(len));
        if (len > 0 && !read_exact(out.data(), len)) return false;
        return true;
    }

    bool skip_string(uint64_t max_len = 64ULL * 1024ULL * 1024ULL) {
        uint64_t len = 0;
        if (!read_pod(len)) return false;
        if (len > max_len || len > remaining()) return false;
        return skip(len);
    }

    bool skip_gguf_value(uint32_t val_type, int depth = 0) {
        if (depth > 16) return false;
        switch (val_type) {
            case 0: case 1: case 7: return skip(1);
            case 2: case 3: return skip(2);
            case 4: case 5: case 6: return skip(4);
            case 10: case 11: case 12: return skip(8);
            case 8: return skip_string();
            case 9: {
                uint32_t arr_type = 0;
                uint64_t arr_len = 0;
                if (!read_pod(arr_type) || !read_pod(arr_len)) return false;
                uint64_t elem_sz = 0;
                if (arr_type == 0 || arr_type == 1 || arr_type == 7) elem_sz = 1;
                else if (arr_type == 2 || arr_type == 3) elem_sz = 2;
                else if (arr_type == 4 || arr_type == 5 || arr_type == 6) elem_sz = 4;
                else if (arr_type == 10 || arr_type == 11 || arr_type == 12) elem_sz = 8;
                if (elem_sz > 0) {
                    if (arr_len > remaining() / elem_sz) return false;
                    return skip(arr_len * elem_sz);
                }
                if (arr_type == 8) {
                    if (arr_len > 4000000ULL) return false;
                    for (uint64_t i = 0; i < arr_len; ++i) {
                        if (!skip_string()) return false;
                    }
                    return true;
                }
                if (arr_len > 4000000ULL) return false;
                for (uint64_t i = 0; i < arr_len; ++i) {
                    if (!skip_gguf_value(arr_type, depth + 1)) return false;
                }
                return true;
            }
            default:
                return false;
        }
    }

    bool read_uint_val(uint32_t val_type, uint32_t& out) {
        if (val_type == 4 || val_type == 5) {
            uint32_t v = 0;
            if (!read_pod(v)) return false;
            out = v;
            return true;
        }
        if (val_type == 10 || val_type == 11) {
            uint64_t v = 0;
            if (!read_pod(v)) return false;
            out = static_cast<uint32_t>(v);
            return true;
        }
        return false;
    }

private:
    bool refill() {
        uint64_t rem = remaining();
        if (rem == 0) return false;
        size_t want = (rem < kBufCap) ? static_cast<size_t>(rem) : kBufCap;
        f_.read(reinterpret_cast<char*>(buf_), static_cast<std::streamsize>(want));
        std::streamsize got = f_.gcount();
        if (got <= 0) return false;
        buf_pos_ = 0;
        buf_end_ = static_cast<size_t>(got);
        return true;
    }

    std::ifstream f_;
    uint64_t file_size_ = 0;
    uint64_t pos_ = 0;
    uint8_t  buf_[kBufCap]{};
    size_t   buf_pos_ = 0;
    size_t   buf_end_ = 0;
};

static bool ends_with(const std::string& s, const std::string& suf) {
    return s.size() >= suf.size() && s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
}

static bool starts_with(const std::string& s, const std::string& pre) {
    return s.size() >= pre.size() && s.compare(0, pre.size(), pre) == 0;
}

static std::string to_lower_copy(std::string s) {
    for (char& c : s) {
        if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
    }
    return s;
}

static ArchFamily classify_arch(const std::string& arch_raw) {
    std::string a = to_lower_copy(arch_raw);
    if (a.find("gemma4") != std::string::npos) return ArchFamily::GEMMA4;
    if (a.find("gemma") != std::string::npos)  return ArchFamily::GEMMA_LEGACY;
    if (a.find("qwen") != std::string::npos)   return ArchFamily::QWEN;
    if (a.find("mistral") != std::string::npos || a.find("mixtral") != std::string::npos) return ArchFamily::MISTRAL;
    if (a.find("deepseek") != std::string::npos) return ArchFamily::DEEPSEEK;
    if (a.find("llama") != std::string::npos)  return ArchFamily::LLAMA;
    if (a == "phi" || starts_with(a, "phi2") || starts_with(a, "phi3") || starts_with(a, "phi4") || starts_with(a, "phi-")) {
        return ArchFamily::PHI;
    }
    return a.empty() ? ArchFamily::UNKNOWN : ArchFamily::GENERIC_TRANSFORMER;
}

static ChatTemplateFamily classify_chat_template(const std::string& tmpl, ArchFamily fam, bool& has_thinking) {
    has_thinking = (tmpl.find("<|channel>thought") != std::string::npos ||
                    tmpl.find("<think>") != std::string::npos);
    if (tmpl.find("<|turn>") != std::string::npos) return ChatTemplateFamily::GEMMA4_TURN;
    if (tmpl.find("<start_of_turn>") != std::string::npos) return ChatTemplateFamily::GEMMA_LEGACY_TURN;
    if (tmpl.find("<|im_start|>") != std::string::npos) return ChatTemplateFamily::CHATML;
    if (tmpl.find("<|start_header_id|>") != std::string::npos) return ChatTemplateFamily::LLAMA3_HEADER;
    if (tmpl.find("<|user|>") != std::string::npos && tmpl.find("<|end|>") != std::string::npos) return ChatTemplateFamily::PHI3_TAG;
    if (tmpl.find("[INST]") != std::string::npos) return ChatTemplateFamily::MISTRAL_INST;

    // Fallback default by architecture family when template KV is omitted
    switch (fam) {
        case ArchFamily::GEMMA4:       return ChatTemplateFamily::GEMMA4_TURN;
        case ArchFamily::GEMMA_LEGACY: return ChatTemplateFamily::GEMMA_LEGACY_TURN;
        case ArchFamily::QWEN:         return ChatTemplateFamily::CHATML;
        case ArchFamily::LLAMA:        return ChatTemplateFamily::LLAMA3_HEADER;
        case ArchFamily::PHI:          return ChatTemplateFamily::PHI3_TAG;
        case ArchFamily::MISTRAL:      return ChatTemplateFamily::MISTRAL_INST;
        default:                       return ChatTemplateFamily::RAW;
    }
}

static std::string json_escape(const std::string& in) {
    std::string out;
    out.reserve(in.size() + 8);
    for (unsigned char c : in) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out.push_back(static_cast<char>(c));
                }
        }
    }
    return out;
}

static std::string fmt_gib(uint64_t bytes) {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(2) << (static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0)) << " GiB";
    return ss.str();
}

static std::string get_executable_dir() {
#ifdef _WIN32
    char buf[MAX_PATH];
    DWORD len = GetModuleFileNameA(nullptr, buf, MAX_PATH);
    if (len > 0 && len < MAX_PATH) {
        return std::filesystem::path(std::string(buf, len)).parent_path().string();
    }
    return ".";
#else
    char buf[4096];
    ssize_t len = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (len > 0) {
        buf[len] = '\0';
        return std::filesystem::path(buf).parent_path().string();
    }
    return ".";
#endif
}

} // namespace

// ============================================================================
// HavenModelIntelligence — Zero-Mmap Header DNA Probe
// ============================================================================

bool HavenModelIntelligence::probe_gguf_header(const std::string& filepath,
                                               ModelFingerprint& out_fp,
                                               std::string& out_err) {
    auto t0 = std::chrono::high_resolution_clock::now();
    out_fp = ModelFingerprint{};
    out_fp.filepath = filepath;
    out_err.clear();

    BufferedFileReader rd(filepath);
    if (!rd.is_open()) {
        out_err = "Failed to open GGUF file: " + filepath;
        return false;
    }
    out_fp.file_size_bytes = rd.file_size();

    uint32_t magic = 0;
    if (!rd.read_pod(magic) || magic != kGgufMagic) {
        out_err = "Invalid GGUF magic header (expected 0x46554747 'GGUF')";
        return false;
    }

    uint32_t version = 0;
    uint64_t tensor_count = 0;
    uint64_t kv_count = 0;
    if (!rd.read_pod(version) || !rd.read_pod(tensor_count) || !rd.read_pod(kv_count)) {
        out_err = "Truncated GGUF file header";
        return false;
    }
    if (version < 2 || version > 4 || kv_count > 1000000ULL || tensor_count > 1000000ULL) {
        out_err = "GGUF header version/counts out of safe bounds";
        return false;
    }
    out_fp.gguf_version = version;
    out_fp.kv_count = kv_count;
    out_fp.tensor_count = tensor_count;

    uint32_t alignment = 32;
    std::string chat_template;
    std::vector<bool> swa_pattern;
    std::vector<uint32_t> head_count_kv_per_layer;
    bool has_shared_kv_key = false;

    for (uint64_t i = 0; i < kv_count; ++i) {
        std::string key;
        uint32_t vt = 0;
        if (!rd.read_string(key, 65536) || !rd.read_pod(vt)) {
            out_err = "Truncated GGUF KV entry #" + std::to_string(i);
            return false;
        }

        // Track multimodal sub-config keys without letting them clobber LLM geometry
        if (key.find(".vision.") != std::string::npos) {
            out_fp.has_vision_encoder = true;
            if (!rd.skip_gguf_value(vt)) { out_err = "Failed skipping vision KV: " + key; return false; }
            continue;
        }
        if (key.find(".audio.") != std::string::npos) {
            out_fp.has_audio_encoder = true;
            if (!rd.skip_gguf_value(vt)) { out_err = "Failed skipping audio KV: " + key; return false; }
            continue;
        }

        if (key == "general.architecture" && vt == 8) {
            if (!rd.read_string(out_fp.architecture, 256)) { out_err = "Bad general.architecture"; return false; }
        } else if (key == "general.name" && vt == 8) {
            if (!rd.read_string(out_fp.general_name, 1024)) { out_err = "Bad general.name"; return false; }
        } else if (key == "general.description" && vt == 8) {
            if (!rd.read_string(out_fp.general_description, 4096)) { out_err = "Bad general.description"; return false; }
        } else if (key == "general.alignment" && (vt == 4 || vt == 10)) {
            uint32_t al = 32;
            if (!rd.read_uint_val(vt, al)) return false;
            if (al > 0 && (al & (al - 1)) == 0 && al <= 4096) alignment = al;
        } else if (key == "split.no" && (vt == 4 || vt == 5 || vt == 2 || vt == 10)) {
            if (vt == 2) {
                uint16_t v = 0;
                if (!rd.read_pod(v)) return false;
                out_fp.split_no = v;
            } else {
                if (!rd.read_uint_val(vt, out_fp.split_no)) return false;
            }
            out_fp.is_split_shard = true;
        } else if (key == "split.count" && (vt == 4 || vt == 5 || vt == 2 || vt == 10)) {
            if (vt == 2) {
                uint16_t v = 0;
                if (!rd.read_pod(v)) return false;
                out_fp.split_count = v;
            } else {
                if (!rd.read_uint_val(vt, out_fp.split_count)) return false;
            }
            if (out_fp.split_count > 1) out_fp.is_split_shard = true;
        } else if (key == "tokenizer.chat_template" && vt == 8) {
            if (!rd.read_string(chat_template, 131072)) { out_err = "Bad tokenizer.chat_template"; return false; }
        } else if (key == "tokenizer.ggml.tokens" && vt == 9) {
            // Read vocab_size directly from array header WITHOUT allocating 262,144 std::strings!
            uint32_t arr_type = 0;
            uint64_t arr_len = 0;
            if (!rd.read_pod(arr_type) || !rd.read_pod(arr_len) || arr_type != 8 || arr_len > 2000000ULL) {
                out_err = "Invalid tokenizer.ggml.tokens header";
                return false;
            }
            out_fp.vocab_size = static_cast<uint32_t>(arr_len);
            for (uint64_t t = 0; t < arr_len; ++t) {
                if (!rd.skip_string(65536)) { out_err = "Truncated tokenizer.ggml.tokens"; return false; }
            }
        } else if (ends_with(key, ".context_length") && (vt == 4 || vt == 10)) {
            if (!rd.read_uint_val(vt, out_fp.max_context_length)) return false;
        } else if (ends_with(key, ".embedding_length") && (vt == 4 || vt == 10)) {
            if (!rd.read_uint_val(vt, out_fp.embedding_dim)) return false;
        } else if (ends_with(key, ".feed_forward_length") && (vt == 4 || vt == 10)) {
            if (!rd.read_uint_val(vt, out_fp.hidden_dim)) return false;
        } else if (ends_with(key, ".feed_forward_length") && vt == 9) {
            uint32_t arr_type = 0;
            uint64_t arr_len = 0;
            if (!rd.read_pod(arr_type) || !rd.read_pod(arr_len) || arr_len > 4096) return false;
            uint32_t max_ff = 0;
            for (uint64_t j = 0; j < arr_len; ++j) {
                uint32_t v = 0;
                if (!rd.read_uint_val(arr_type, v)) return false;
                if (v > max_ff) max_ff = v;
            }
            if (max_ff > 0) out_fp.hidden_dim = max_ff;
        } else if (ends_with(key, ".block_count") && (vt == 4 || vt == 10)) {
            if (!rd.read_uint_val(vt, out_fp.num_layers)) return false;
            if (out_fp.num_layers > 4096u) {
                out_err = "GGUF block_count exceeds safety limit (4096)";
                return false;
            }
        } else if (ends_with(key, ".attention.head_count") && (vt == 4 || vt == 10)) {
            if (!rd.read_uint_val(vt, out_fp.num_heads)) return false;
        } else if (ends_with(key, ".attention.head_count_kv") && (vt == 4 || vt == 10)) {
            if (!rd.read_uint_val(vt, out_fp.num_kv_heads)) return false;
        } else if (ends_with(key, ".attention.head_count_kv") && vt == 9) {
            uint32_t arr_type = 0;
            uint64_t arr_len = 0;
            if (!rd.read_pod(arr_type) || !rd.read_pod(arr_len) || arr_len > 4096) return false;
            head_count_kv_per_layer.clear();
            head_count_kv_per_layer.reserve(static_cast<size_t>(arr_len));
            uint32_t max_kvh = 0;
            for (uint64_t j = 0; j < arr_len; ++j) {
                uint32_t v = 0;
                if (!rd.read_uint_val(arr_type, v)) return false;
                head_count_kv_per_layer.push_back(v);
                if (v > max_kvh) max_kvh = v;
            }
            if (max_kvh > 0) out_fp.num_kv_heads = max_kvh;
        } else if (ends_with(key, ".attention.sliding_window_pattern") && vt == 9) {
            uint32_t arr_type = 0;
            uint64_t arr_len = 0;
            if (!rd.read_pod(arr_type) || !rd.read_pod(arr_len) || arr_type != 7 || arr_len > 4096) return false;
            swa_pattern.resize(static_cast<size_t>(arr_len));
            for (uint64_t j = 0; j < arr_len; ++j) {
                uint8_t b = 0;
                if (!rd.read_pod(b)) return false;
                swa_pattern[static_cast<size_t>(j)] = (b != 0);
            }
        } else if (ends_with(key, ".attention.key_length_swa") && (vt == 4 || vt == 10)) {
            if (!rd.read_uint_val(vt, out_fp.head_dim_swa)) return false;
        } else if (ends_with(key, ".attention.key_length") && (vt == 4 || vt == 10)) {
            if (!rd.read_uint_val(vt, out_fp.head_dim)) return false;
        } else if (ends_with(key, ".attention.sliding_window") && (vt == 4 || vt == 10)) {
            if (!rd.read_uint_val(vt, out_fp.sliding_window)) return false;
        } else if (ends_with(key, ".attention.shared_kv_layers") && (vt == 4 || vt == 10)) {
            if (!rd.read_uint_val(vt, out_fp.shared_kv_layers)) return false;
            has_shared_kv_key = true;
        } else if (ends_with(key, ".expert_count") && (vt == 4 || vt == 10)) {
            if (!rd.read_uint_val(vt, out_fp.expert_count)) return false;
        } else if (ends_with(key, ".expert_used_count") && (vt == 4 || vt == 10)) {
            if (!rd.read_uint_val(vt, out_fp.expert_used_count)) return false;
        } else if (ends_with(key, ".expert_feed_forward_length") && (vt == 4 || vt == 10)) {
            if (!rd.read_uint_val(vt, out_fp.expert_ffn_dim)) return false;
        } else if (ends_with(key, ".turbo.version") && (vt == 4 || vt == 10)) {
            if (!rd.read_uint_val(vt, out_fp.turbo_version)) return false;
        } else if (ends_with(key, ".turbo.hadamard_qkv") && (vt == 4 || vt == 10)) {
            if (!rd.read_uint_val(vt, out_fp.turbo_hadamard_qkv)) return false;
        } else if (ends_with(key, ".turbo.hadamard_ffn") && (vt == 4 || vt == 10)) {
            if (!rd.read_uint_val(vt, out_fp.turbo_hadamard_ffn)) return false;
        } else if (ends_with(key, ".turbo.hadamard_ffn_down") && (vt == 4 || vt == 10)) {
            if (!rd.read_uint_val(vt, out_fp.turbo_hadamard_ffn_down)) return false;
        } else if (ends_with(key, ".turbo.hadamard_moe_in") && (vt == 4 || vt == 10)) {
            if (!rd.read_uint_val(vt, out_fp.turbo_hadamard_moe_in)) return false;
        } else if (ends_with(key, ".turbo.hadamard_moe_down") && (vt == 4 || vt == 10)) {
            if (!rd.read_uint_val(vt, out_fp.turbo_hadamard_moe_down)) return false;
        } else if (ends_with(key, ".turbo.folded_norms") && vt == 7) {
            uint8_t b = 0;
            if (!rd.read_pod(b)) return false;
            out_fp.turbo_folded_norms = (b != 0);
        } else if (ends_with(key, ".turbo.folded_scales") && vt == 7) {
            uint8_t b = 0;
            if (!rd.read_pod(b)) return false;
            out_fp.turbo_folded_scales = (b != 0);
        } else {
            if (!rd.skip_gguf_value(vt)) {
                out_err = "Failed skipping GGUF KV value for key: " + key;
                return false;
            }
        }
    }

    // Parse tensor directory headers (ZERO tensor payload reads!)
    std::set<std::string> layers_with_k;
    std::set<std::string> layers_with_v;

    for (uint64_t i = 0; i < tensor_count; ++i) {
        std::string tname;
        uint32_t nd = 0;
        if (!rd.read_string(tname, 1024) || !rd.read_pod(nd) || nd == 0 || nd > 8) {
            out_err = "Corrupted tensor header #" + std::to_string(i);
            return false;
        }
        std::vector<uint64_t> shape(nd);
        for (uint32_t d = 0; d < nd; ++d) {
            if (!rd.read_pod(shape[d])) { out_err = "Truncated tensor shape: " + tname; return false; }
        }
        uint32_t raw_type = 0;
        uint64_t raw_offset = 0;
        if (!rd.read_pod(raw_type) || !rd.read_pod(raw_offset)) {
            out_err = "Truncated tensor type/offset: " + tname;
            return false;
        }
        (void)raw_offset;
        QuantType qt = static_cast<QuantType>(raw_type);
        uint64_t nbytes = quant_tensor_byte_size(qt, shape);

        // Classify multimodal vs LLM tensors
        const bool is_vision = starts_with(tname, "v.") || starts_with(tname, "mm.");
        const bool is_audio  = starts_with(tname, "a.");
        if (is_vision) {
            out_fp.has_vision_encoder = true;
            out_fp.vision_tensor_count++;
            out_fp.vision_bytes += nbytes;
        } else if (is_audio) {
            out_fp.has_audio_encoder = true;
            out_fp.audio_tensor_count++;
            out_fp.audio_bytes += nbytes;
        } else {
            out_fp.llm_weight_bytes += nbytes;
            auto& qs = out_fp.quant_histogram[qt];
            qs.tensor_count++;
            qs.total_bytes += nbytes;
        }

        // Architectural DNA signals from tensor names
        if (tname == "per_layer_token_embd.weight") out_fp.has_pleias = true;
        if (ends_with(tname, ".ffn_gate_up_exps.weight")) {
            out_fp.has_moe = true;
            out_fp.fused_moe_gate_up = true;
            if (out_fp.expert_count == 0 && shape.size() >= 3) out_fp.expert_count = static_cast<uint32_t>(shape[2]);
        } else if (ends_with(tname, ".ffn_down_exps.weight") || ends_with(tname, ".ffn_gate_exps.weight")) {
            out_fp.has_moe = true;
            if (out_fp.expert_count == 0 && shape.size() >= 3) out_fp.expert_count = static_cast<uint32_t>(shape[2]);
        }
        if (ends_with(tname, ".attn_q.bias") || ends_with(tname, ".attn_k.bias") || ends_with(tname, ".attn_v.bias")) {
            out_fp.has_qkv_bias = true;
        }
        if (ends_with(tname, ".attn_q_norm.weight") || ends_with(tname, ".attn_k_norm.weight")) {
            out_fp.has_qk_norm = true;
        }
        if (starts_with(tname, "blk.")) {
            size_t dot2 = tname.find('.', 4);
            if (dot2 != std::string::npos) {
                std::string blk_id = tname.substr(4, dot2 - 4);
                std::string suf = tname.substr(dot2 + 1);
                if (suf == "attn_k.weight") layers_with_k.insert(blk_id);
                else if (suf == "attn_v.weight") layers_with_v.insert(blk_id);
            }
        }
    }

    uint64_t align_mask = static_cast<uint64_t>(alignment - 1);
    out_fp.header_bytes = (rd.tell() + align_mask) & ~align_mask;

    // Finalize derived DNA properties
    out_fp.arch_family = classify_arch(out_fp.architecture);
    if (out_fp.head_dim == 0 && out_fp.num_heads > 0) {
        out_fp.head_dim = out_fp.embedding_dim / out_fp.num_heads;
    }
    if (out_fp.head_dim == 0) out_fp.head_dim = 128;
    if (out_fp.num_kv_heads == 0) out_fp.num_kv_heads = out_fp.num_heads > 0 ? out_fp.num_heads : 8;
    if (!has_shared_kv_key && out_fp.arch_family == ArchFamily::GEMMA4 && out_fp.num_layers == 42) {
        out_fp.shared_kv_layers = 18;
    }
    out_fp.has_shared_kv = (out_fp.shared_kv_layers > 0);
    if (out_fp.expert_count > 0) out_fp.has_moe = true;

    for (const auto& lid : layers_with_k) {
        if (layers_with_v.find(lid) == layers_with_v.end()) {
            out_fp.has_k_eq_v = true;
            break;
        }
    }

    out_fp.has_swa = (out_fp.sliding_window > 0 || !swa_pattern.empty());
    if (!swa_pattern.empty()) {
        for (bool is_s : swa_pattern) {
            if (is_s) out_fp.swa_layer_count++;
            else out_fp.global_layer_count++;
        }
    } else if (out_fp.has_swa && out_fp.num_layers > 0) {
        for (uint32_t l = 0; l < out_fp.num_layers; ++l) {
            if (out_fp.arch_family == ArchFamily::MISTRAL || (l % 6 != 5)) out_fp.swa_layer_count++;
            else out_fp.global_layer_count++;
        }
    } else {
        out_fp.global_layer_count = out_fp.num_layers;
    }

    // Native runtime Turbo FWHT / folded-norm mode (Stock --mode stock folds scales to 1.0 in-file and needs no runtime Turbo module)
    out_fp.has_turbo_bake = (out_fp.turbo_version > 0 ||
                             out_fp.turbo_hadamard_qkv > 0 ||
                             out_fp.turbo_hadamard_ffn > 0 ||
                             out_fp.turbo_hadamard_ffn_down > 0 ||
                             out_fp.turbo_hadamard_moe_in > 0 ||
                             out_fp.turbo_hadamard_moe_down > 0 ||
                             out_fp.turbo_folded_norms);

    out_fp.chat_template_family = classify_chat_template(
        chat_template, out_fp.arch_family, out_fp.supports_thinking_channel);

    // Dominant quant type by weight bytes
    uint64_t best_qb = 0;
    for (const auto& [qt, qs] : out_fp.quant_histogram) {
        if (qs.total_bytes > best_qb) {
            best_qb = qs.total_bytes;
            out_fp.dominant_quant = qt;
        }
    }

    // Exact FP16 KV cache budget at 4K context (respecting shared_kv_layers & per-layer head dims!)
    constexpr uint32_t kMaxLayersCap = 4096u;
    if (out_fp.num_layers > kMaxLayersCap) {
        out_err = "GGUF num_layers exceeds safety limit";
        return false;
    }
    uint32_t kv_layers = (out_fp.num_layers > out_fp.shared_kv_layers)
        ? (out_fp.num_layers - out_fp.shared_kv_layers)
        : out_fp.num_layers;
    uint64_t kv_bytes_4k = 0;
    auto sat_add_u64 = [](uint64_t a, uint64_t b) -> uint64_t {
        return (a > std::numeric_limits<uint64_t>::max() - b) ? std::numeric_limits<uint64_t>::max() : (a + b);
    };
    for (uint32_t l = 0; l < kv_layers; ++l) {
        bool is_swa = (l < swa_pattern.size()) ? swa_pattern[l] : (out_fp.has_swa && (l % 6 != 5));
        uint32_t hd = (is_swa && out_fp.head_dim_swa > 0) ? out_fp.head_dim_swa : out_fp.head_dim;
        uint32_t hkv = (l < head_count_kv_per_layer.size() && head_count_kv_per_layer[l] > 0)
            ? head_count_kv_per_layer[l]
            : out_fp.num_kv_heads;
        // K + V in FP16 (2 bytes each) * 4096 tokens (overflow-safe)
        uint64_t hkv_hd = static_cast<uint64_t>(hkv) * static_cast<uint64_t>(hd);
        uint64_t layer_kv = (hkv_hd > std::numeric_limits<uint64_t>::max() / 16384ULL)
            ? std::numeric_limits<uint64_t>::max()
            : (hkv_hd * 16384ULL);
        kv_bytes_4k = sat_add_u64(kv_bytes_4k, layer_kv);
    }
    out_fp.kv_cache_bytes_4k = kv_bytes_4k;
    // Scratch + activations headroom (~384 MiB for dense, ~640 MiB for MoE)
    uint64_t scratch_est = out_fp.has_moe ? (640ULL * 1024ULL * 1024ULL) : (384ULL * 1024ULL * 1024ULL);
    out_fp.estimated_vram_bytes = sat_add_u64(sat_add_u64(out_fp.llm_weight_bytes, kv_bytes_4k), scratch_est);

    // Compose feature flags bitmask
    uint64_t f = HAVEN_FEAT_NONE;
    if (out_fp.has_swa)            f |= HAVEN_FEAT_SWA;
    if (out_fp.has_shared_kv)      f |= HAVEN_FEAT_SHARED_KV;
    if (out_fp.has_k_eq_v)         f |= HAVEN_FEAT_K_EQ_V;
    if (out_fp.has_pleias)         f |= HAVEN_FEAT_PLEIAS;
    if (out_fp.has_moe)            f |= HAVEN_FEAT_MOE;
    if (out_fp.has_qkv_bias)       f |= HAVEN_FEAT_QKV_BIAS;
    if (out_fp.has_qk_norm)        f |= HAVEN_FEAT_QK_NORM;
    if (out_fp.has_turbo_bake)     f |= HAVEN_FEAT_TURBO_FWHT;
    if (out_fp.has_vision_encoder) f |= HAVEN_FEAT_VISION_SIGLIP;
    if (out_fp.has_audio_encoder)  f |= HAVEN_FEAT_AUDIO_ENCODER;
    if (out_fp.arch_family == ArchFamily::GEMMA4 || out_fp.arch_family == ArchFamily::GEMMA_LEGACY) {
        f |= HAVEN_FEAT_SOFTCAP;
    }
    if (out_fp.arch_family == ArchFamily::GEMMA4) {
        f |= HAVEN_FEAT_PROPORTIONAL_ROPE;
    }
    out_fp.feature_flags = f;

    auto t1 = std::chrono::high_resolution_clock::now();
    out_fp.probe_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return true;
}

// ============================================================================
// Hardware & OS Silicon Probe
// ============================================================================

HardwareProfile HavenModelIntelligence::probe_hardware() {
    HardwareProfile hw;
    unsigned int hc = std::thread::hardware_concurrency();
    hw.cpu_logical_cores = hc > 0 ? hc : 1;

#ifdef _WIN32
    int info[4] = {0, 0, 0, 0};
    __cpuid(info, 1);
    hw.has_fma  = (info[2] & (1 << 12)) != 0;
    hw.has_f16c = (info[2] & (1 << 29)) != 0;
    __cpuidex(info, 7, 0);
    hw.has_avx2   = (info[1] & (1 << 5)) != 0;
    hw.has_avx512 = (info[1] & (1 << 16)) != 0;

    MEMORYSTATUSEX ms{};
    ms.dwLength = sizeof(ms);
    if (GlobalMemoryStatusEx(&ms)) {
        hw.total_ram_bytes = static_cast<uint64_t>(ms.ullTotalPhys);
        hw.available_ram_bytes = static_cast<uint64_t>(ms.ullAvailPhys);
    }
#else
#if defined(__x86_64__) || defined(__i386__)
    unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
    if (__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
        hw.has_fma  = (ecx & (1u << 12)) != 0;
        hw.has_f16c = (ecx & (1u << 29)) != 0;
    }
    if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
        hw.has_avx2   = (ebx & (1u << 5)) != 0;
        hw.has_avx512 = (ebx & (1u << 16)) != 0;
    }
#endif
    std::ifstream meminfo("/proc/meminfo");
    std::string line;
    while (std::getline(meminfo, line)) {
        if (starts_with(line, "MemTotal:")) {
            uint64_t kb = 0;
            std::istringstream iss(line.substr(9));
            if (iss >> kb) hw.total_ram_bytes = kb * 1024ULL;
        } else if (starts_with(line, "MemAvailable:")) {
            uint64_t kb = 0;
            std::istringstream iss(line.substr(13));
            if (iss >> kb) hw.available_ram_bytes = kb * 1024ULL;
        }
    }
    std::error_code hmb_ec;
    if (std::filesystem::exists("/proc/hmb_pagefile", hmb_ec)) {
        hw.hmb_vmm_kernel_active = true;
        hw.hmb_vmm_node = "/proc/hmb_pagefile";
    } else if (std::filesystem::exists("/proc/hmb_swap", hmb_ec)) {
        hw.hmb_vmm_kernel_active = true;
        hw.hmb_vmm_node = "/proc/hmb_swap";
    } else if (std::filesystem::exists("/dev/hmb0", hmb_ec)) {
        hw.hmb_vmm_kernel_active = true;
        hw.hmb_vmm_node = "/dev/hmb0";
    }
#endif

    // Probe native zero-compile CUDA backend
    hw.cuda_compiled = HavenCudaEngine::is_cuda_compiled();
    if (hw.cuda_compiled) {
        hw.cuda_device_count = HavenCudaEngine::get_device_count();
        if (hw.cuda_device_count > 0) {
            hw.cuda_device_summary = HavenCudaEngine::get_device_summary(0);
            // Parse "<N> MiB VRAM" from summary so we don't need direct CUDA headers here
            size_t mib_pos = hw.cuda_device_summary.find(" MiB VRAM");
            if (mib_pos != std::string::npos) {
                size_t start = mib_pos;
                while (start > 0 && hw.cuda_device_summary[start - 1] >= '0' && hw.cuda_device_summary[start - 1] <= '9') {
                    --start;
                }
                if (start < mib_pos) {
                    uint64_t mib = std::strtoull(hw.cuda_device_summary.substr(start, mib_pos - start).c_str(), nullptr, 10);
                    hw.cuda_total_vram_bytes = mib * 1024ULL * 1024ULL;
                }
            }
        }
    }

    // Probe Cloud TPU PJRT library
    const char* env_tpu = std::getenv("HAVEN_LIBTPU_PATH");
    const std::vector<std::string> tpu_candidates = {
        env_tpu ? std::string(env_tpu) : std::string(),
        "/usr/local/lib/libtpu.so",
        "/usr/lib/libtpu.so",
        "/opt/tpu/libtpu.so"
    };
    for (const auto& c : tpu_candidates) {
        std::error_code ec;
        if (!c.empty() && std::filesystem::exists(c, ec)) {
            hw.tpu_available = true;
            hw.tpu_library_path = c;
            break;
        }
    }

#ifdef HAVEN_HAS_MLX
    hw.mlx_compiled = true;
#endif

    return hw;
}

// ============================================================================
// Intelligent Zero-Bloat Execution Planner
// ============================================================================

ExecutionPlan HavenModelIntelligence::plan_execution(const ModelFingerprint& fp,
                                                     const HardwareProfile& hw,
                                                     const RoutingOptions& opts) {
    ExecutionPlan plan;
    plan.fingerprint = fp;
    plan.hardware = hw;

    // 1. Select exact architecture microkernel module (ONLY ONE loaded!)
    static const std::vector<std::string> kAllArchModules = {
        "haven_arch_gemma4",
        "haven_arch_llama",
        "haven_arch_qwen",
        "haven_arch_phi"
    };
    switch (fp.arch_family) {
        case ArchFamily::GEMMA4:
        case ArchFamily::GEMMA_LEGACY:
            plan.selected_arch_module = "haven_arch_gemma4";
            break;
        case ArchFamily::QWEN:
            plan.selected_arch_module = "haven_arch_qwen";
            break;
        case ArchFamily::PHI:
            plan.selected_arch_module = "haven_arch_phi";
            break;
        case ArchFamily::LLAMA:
        case ArchFamily::MISTRAL:
        case ArchFamily::DEEPSEEK:
        case ArchFamily::GENERIC_TRANSFORMER:
        default:
            plan.selected_arch_module = "haven_arch_llama";
            break;
    }

    // 2. Compute VRAM / RAM fit at target context length
    uint32_t ctx = opts.target_context_len > 0 ? opts.target_context_len : 4096u;
    if (fp.max_context_length > 0 && ctx > fp.max_context_length) ctx = fp.max_context_length;
    plan.recommended_context_len = ctx;

    uint64_t kv_scaled = (fp.kv_cache_bytes_4k * static_cast<uint64_t>(ctx)) / 4096ULL;
    uint64_t scratch_est = fp.has_moe ? (640ULL * 1024ULL * 1024ULL) : (384ULL * 1024ULL * 1024ULL);
    uint64_t needed_bytes = fp.llm_weight_bytes + kv_scaled + scratch_est;

    plan.fits_in_vram = (hw.cuda_device_count > 0 && hw.cuda_total_vram_bytes > 0 &&
                         needed_bytes <= static_cast<uint64_t>(static_cast<double>(hw.cuda_total_vram_bytes) * 0.94));
    plan.fits_in_ram  = (hw.total_ram_bytes == 0 || needed_bytes <= hw.total_ram_bytes);

    // 3. Select optimal silicon backend (respecting explicit override)
    if (opts.preferred_backend == BackendKind::CUDA_NATIVE && hw.cuda_compiled && hw.cuda_device_count > 0) {
        plan.selected_backend = BackendKind::CUDA_NATIVE;
        plan.selected_backend_module = "haven_backend_cuda";
        plan.use_cuda_graphs = true;
        plan.decision_reason = "Explicit --cuda requested on " + hw.cuda_device_summary;
    } else if (opts.preferred_backend == BackendKind::TPU_PJRT && hw.tpu_available) {
        plan.selected_backend = BackendKind::TPU_PJRT;
        plan.selected_backend_module = "haven_backend_tpu";
        plan.decision_reason = "Explicit --tpu requested (" + hw.tpu_library_path + ")";
    } else if (opts.preferred_backend == BackendKind::CPU_AVX2) {
        plan.selected_backend = BackendKind::CPU_AVX2;
        plan.selected_backend_module = "haven_backend_avx2";
        plan.decision_reason = "Explicit --cpu requested -> native AVX2+FMA CPU backend";
    } else {
        // Automatic intelligent routing
        if (hw.tpu_available && fp.arch_family == ArchFamily::GEMMA4) {
            plan.selected_backend = BackendKind::TPU_PJRT;
            plan.selected_backend_module = "haven_backend_tpu";
            plan.decision_reason = "Cloud TPU PJRT detected (" + hw.tpu_library_path + ") for Gemma 4 on-chip HBM execution";
        } else if (hw.cuda_compiled && hw.cuda_device_count > 0 && plan.fits_in_vram) {
            plan.selected_backend = BackendKind::CUDA_NATIVE;
            plan.selected_backend_module = "haven_backend_cuda";
            plan.use_cuda_graphs = true;
            plan.decision_reason = "Zero-compile CUDA V2 on " + hw.cuda_device_summary +
                " fits " + fmt_gib(needed_bytes) + " cleanly in VRAM";
        } else if (hw.cuda_compiled && hw.cuda_device_count > 0 && !plan.fits_in_vram) {
            plan.selected_backend = BackendKind::CPU_AVX2;
            plan.selected_backend_module = "haven_backend_avx2";
            plan.decision_reason = "Model (" + fmt_gib(needed_bytes) + ") exceeds GPU VRAM (" +
                fmt_gib(hw.cuda_total_vram_bytes) + ") -> auto-routed to native AVX2 CPU backend";
        } else {
            plan.selected_backend = BackendKind::CPU_AVX2;
            plan.selected_backend_module = "haven_backend_avx2";
            plan.decision_reason = hw.hmb_vmm_kernel_active
                ? ("Native AVX2+FMA CPU backend accelerated by Linux HMB VMM (" + hw.hmb_vmm_node + ")")
                : "No CUDA/TPU accelerator detected -> native AVX2+FMA CPU backend";
        }
    }

    plan.use_turbo_fwht = fp.has_turbo_bake;
    plan.recommended_threads = std::max(1u, hw.cpu_logical_cores > 2 ? (hw.cpu_logical_cores - 1u) : hw.cpu_logical_cores);

    // 4. Select ONLY required optional capability modules
    if (fp.has_moe)                                      plan.required_capability_modules.push_back("haven_cap_moe");
    if (fp.has_turbo_bake)                               plan.required_capability_modules.push_back("haven_cap_turbo");
    if (fp.has_vision_encoder && opts.enable_vision)     plan.required_capability_modules.push_back("haven_cap_vision");

    // 5. Build explicit anti-bloat skipped module manifest
    for (const auto& a : kAllArchModules) {
        if (a != plan.selected_arch_module) plan.skipped_modules.push_back(a);
    }
    static const std::vector<std::string> kAllBackends = {
        "haven_backend_avx2", "haven_backend_cuda", "haven_backend_tpu"
    };
    for (const auto& b : kAllBackends) {
        if (b != plan.selected_backend_module) plan.skipped_modules.push_back(b);
    }
    static const std::vector<std::string> kAllCaps = {
        "haven_cap_moe", "haven_cap_turbo", "haven_cap_vision"
    };
    for (const auto& c : kAllCaps) {
        if (std::find(plan.required_capability_modules.begin(),
                      plan.required_capability_modules.end(), c) == plan.required_capability_modules.end()) {
            plan.skipped_modules.push_back(c);
        }
    }

    return plan;
}

// ============================================================================
// HavenModuleLoader — True On-Demand Dynamic Microkernel Loader
// ============================================================================

HavenModuleLoader::~HavenModuleLoader() {
    unload_all();
}

void HavenModuleLoader::set_search_dirs(const std::vector<std::string>& dirs) {
    std::lock_guard<std::mutex> lk(mu_);
    custom_search_dirs_ = dirs;
}

std::vector<std::string> HavenModuleLoader::get_default_search_dirs() const {
    std::vector<std::string> dirs;
    std::string exe_dir = get_executable_dir();
    dirs.push_back((std::filesystem::path(exe_dir) / "modules").string());
    dirs.push_back("modules");
    dirs.push_back("build/modules");
    return dirs;
}

std::unordered_map<std::string, std::string> HavenModuleLoader::index_available_modules() const {
    std::unordered_map<std::string, std::string> index;
    std::vector<std::string> dirs;
    {
        std::lock_guard<std::mutex> lk(mu_);
        dirs = custom_search_dirs_.empty() ? get_default_search_dirs() : custom_search_dirs_;
    }

#ifdef _WIN32
    const std::string want_ext = ".dll";
#elif defined(__APPLE__)
    const std::string want_ext = ".dylib";
#else
    const std::string want_ext = ".so";
#endif

    for (const auto& d : dirs) {
        std::error_code ec;
        if (!std::filesystem::exists(d, ec) || !std::filesystem::is_directory(d, ec)) continue;
        for (const auto& entry : std::filesystem::recursive_directory_iterator(d, ec)) {
            if (!entry.is_regular_file(ec)) continue;
            if (entry.path().extension().string() != want_ext) continue;
            std::string stem = entry.path().stem().string();
            // Strip Linux/MinGW "lib" prefix so libhaven_arch_gemma4.so/.dll maps to "haven_arch_gemma4"
            if (starts_with(stem, "libhaven_")) {
                stem = stem.substr(3);
            }
            if (starts_with(stem, "haven_") && index.find(stem) == index.end()) {
                index[stem] = entry.path().string();
            }
        }
    }
    return index;
}

void HavenModuleLoader::close_handle(LoadedModuleHandle& mod) {
    if (!mod.handle) return;
#ifdef _WIN32
    FreeLibrary(mod.handle);
#else
    dlclose(mod.handle);
#endif
    mod.handle = nullptr;
    mod.arch_resolve_fn = nullptr;
    mod.backend_probe_fn = nullptr;
}

bool HavenModuleLoader::load_single_module(const std::string& module_id,
                                           const std::string& filepath,
                                           std::string& err) {
    LoadedModuleHandle mod;
    mod.id = module_id;
    mod.filepath = filepath;

#ifdef _WIN32
    std::error_code ec;
    auto canon = std::filesystem::weakly_canonical(filepath, ec);
    std::string target = ec ? filepath : canon.string();
    mod.handle = LoadLibraryExA(target.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (!mod.handle) {
        err = "LoadLibrary failed for " + target;
        return false;
    }
    auto query_fn = reinterpret_cast<HavenQueryDescriptorFn>(GetProcAddress(mod.handle, "haven_module_query_descriptor"));
    mod.arch_resolve_fn = reinterpret_cast<HavenArchResolveSpecFn>(GetProcAddress(mod.handle, "haven_arch_resolve_spec"));
    mod.backend_probe_fn = reinterpret_cast<HavenBackendProbeFn>(GetProcAddress(mod.handle, "haven_backend_probe_runtime"));
#else
    std::error_code ec;
    auto canon = std::filesystem::weakly_canonical(filepath, ec);
    std::string target = ec ? filepath : canon.string();
    mod.handle = dlopen(target.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!mod.handle) {
        const char* dlerr = dlerror();
        err = std::string("dlopen failed for ") + target + (dlerr ? (": " + std::string(dlerr)) : "");
        return false;
    }
    auto query_fn = reinterpret_cast<HavenQueryDescriptorFn>(dlsym(mod.handle, "haven_module_query_descriptor"));
    mod.arch_resolve_fn = reinterpret_cast<HavenArchResolveSpecFn>(dlsym(mod.handle, "haven_arch_resolve_spec"));
    mod.backend_probe_fn = reinterpret_cast<HavenBackendProbeFn>(dlsym(mod.handle, "haven_backend_probe_runtime"));
#endif

    if (!query_fn) {
        err = "Missing haven_module_query_descriptor export in " + filepath;
        close_handle(mod);
        return false;
    }
    const HavenModuleDescriptor* desc = query_fn();
    if (!desc || desc->abi_version != HAVEN_MODULE_ABI_VERSION || !desc->id || !desc->name || !desc->version) {
        err = "ABI version mismatch or null descriptor field in " + filepath;
        close_handle(mod);
        return false;
    }
    mod.descriptor = *desc;
    active_[module_id] = mod;
    return true;
}

bool HavenModuleLoader::activate_plan(ExecutionPlan& plan, std::string& out_log) {
    std::ostringstream log;
    auto available = index_available_modules();

    // Build exact target set of modules needed for THIS model
    std::vector<std::string> desired;
    desired.push_back(plan.selected_arch_module);
    desired.push_back(plan.selected_backend_module);
    for (const auto& cap : plan.required_capability_modules) {
        desired.push_back(cap);
    }

    std::lock_guard<std::mutex> lk(mu_);

    // 1. Unload any currently resident module NOT in `desired` (zero stale bloat on model switch!)
    std::vector<std::string> to_unload;
    for (const auto& [id, _] : active_) {
        if (std::find(desired.begin(), desired.end(), id) == desired.end()) {
            to_unload.push_back(id);
        }
    }
    for (const auto& id : to_unload) {
        close_handle(active_[id]);
        active_.erase(id);
        log << "  [-] Unloaded unused module: " << id << "\n";
    }

    // 2. Load ONLY the modules in `desired` that are not already resident
    bool all_found = true;
    for (const auto& id : desired) {
        if (active_.find(id) != active_.end()) {
            log << "  [=] Reused resident module: " << id << "\n";
            continue;
        }
        auto it = available.find(id);
        if (it == available.end()) {
            all_found = false;
            log << "  [~] Module '" << id << "' not found on disk -> using built-in core fallback\n";
            continue;
        }
        std::string err;
        if (load_single_module(id, it->second, err)) {
            log << "  [+] Hot-loaded module: " << id << " (" << active_[id].descriptor.name
                << " v" << active_[id].descriptor.version << ")\n";
        } else {
            all_found = false;
            log << "  [!] Failed loading " << id << ": " << err << "\n";
        }
    }

    // 3. Query active architecture module via C-ABI to resolve/verify exact runtime math contract
    plan.arch_contract_verified = false;
    auto ait = active_.find(plan.selected_arch_module);
    if (ait != active_.end() && ait->second.arch_resolve_fn) {
        HavenArchProbeInput in{};
        in.architecture      = plan.fingerprint.architecture.c_str();
        in.num_layers        = plan.fingerprint.num_layers;
        in.embedding_dim     = plan.fingerprint.embedding_dim;
        in.hidden_dim        = plan.fingerprint.hidden_dim;
        in.num_heads         = plan.fingerprint.num_heads;
        in.num_kv_heads      = plan.fingerprint.num_kv_heads;
        in.head_dim          = plan.fingerprint.head_dim;
        in.head_dim_swa      = plan.fingerprint.head_dim_swa;
        in.sliding_window    = plan.fingerprint.sliding_window;
        in.shared_kv_layers  = plan.fingerprint.shared_kv_layers;
        in.expert_count      = plan.fingerprint.expert_count;
        in.expert_used_count = plan.fingerprint.expert_used_count;
        in.turbo_version     = plan.fingerprint.turbo_version;
        in.detected_features = plan.fingerprint.feature_flags;

        HavenArchRuntimeSpec spec{};
        if (ait->second.arch_resolve_fn(&in, &spec) != 0 && spec.supported != 0) {
            plan.resolved_arch_spec = spec;
            plan.arch_contract_verified = true;
            log << "  [✓] Arch contract verified: " << spec.summary << "\n";
        }
    }

    out_log = log.str();
    return all_found;
}

void HavenModuleLoader::unload_all() {
    std::lock_guard<std::mutex> lk(mu_);
    for (auto& [_, mod] : active_) {
        close_handle(mod);
    }
    active_.clear();
}

std::vector<std::string> HavenModuleLoader::get_active_module_ids() const {
    std::lock_guard<std::mutex> lk(mu_);
    std::vector<std::string> ids;
    ids.reserve(active_.size());
    for (const auto& [id, _] : active_) ids.push_back(id);
    std::sort(ids.begin(), ids.end());
    return ids;
}

bool HavenModuleLoader::is_module_active(const std::string& id) const {
    std::lock_guard<std::mutex> lk(mu_);
    return active_.find(id) != active_.end();
}

size_t HavenModuleLoader::get_active_module_count() const {
    std::lock_guard<std::mutex> lk(mu_);
    return active_.size();
}

// ============================================================================
// String Helpers, Pretty Report & JSON Serialization
// ============================================================================

const char* HavenModelIntelligence::arch_family_name(ArchFamily f) {
    switch (f) {
        case ArchFamily::GEMMA4:              return "Gemma 4 / 4.5";
        case ArchFamily::GEMMA_LEGACY:        return "Gemma 2/3";
        case ArchFamily::LLAMA:               return "Llama";
        case ArchFamily::QWEN:                return "Qwen";
        case ArchFamily::PHI:                 return "Phi";
        case ArchFamily::MISTRAL:             return "Mistral";
        case ArchFamily::DEEPSEEK:            return "DeepSeek";
        case ArchFamily::GENERIC_TRANSFORMER: return "Generic Transformer";
        default:                              return "Unknown";
    }
}

const char* HavenModelIntelligence::chat_template_name(ChatTemplateFamily c) {
    switch (c) {
        case ChatTemplateFamily::GEMMA4_TURN:       return "gemma4 (<|turn> / <turn|>)";
        case ChatTemplateFamily::GEMMA_LEGACY_TURN: return "gemma-legacy (<start_of_turn>)";
        case ChatTemplateFamily::CHATML:            return "chatml (<|im_start|>)";
        case ChatTemplateFamily::LLAMA3_HEADER:     return "llama3 (<|start_header_id|>)";
        case ChatTemplateFamily::PHI3_TAG:          return "phi3 (<|user|>)";
        case ChatTemplateFamily::MISTRAL_INST:      return "mistral ([INST])";
        default:                                    return "raw";
    }
}

const char* HavenModelIntelligence::backend_kind_name(BackendKind b) {
    switch (b) {
        case BackendKind::CPU_AVX2:    return "CPU_AVX2";
        case BackendKind::CUDA_NATIVE: return "CUDA_NATIVE";
        case BackendKind::TPU_PJRT:    return "TPU_PJRT";
        case BackendKind::MLX:         return "MLX";
        default:                       return "AUTO";
    }
}

const char* HavenModelIntelligence::quant_type_name(QuantType q) {
    switch (q) {
        case QuantType::F32:    return "F32";
        case QuantType::F16:    return "F16";
        case QuantType::BF16:   return "BF16";
        case QuantType::Q4_0:   return "Q4_0";
        case QuantType::Q4_1:   return "Q4_1";
        case QuantType::Q5_0:   return "Q5_0";
        case QuantType::Q5_1:   return "Q5_1";
        case QuantType::Q8_0:   return "Q8_0";
        case QuantType::Q8_1:   return "Q8_1";
        case QuantType::Q2_K:   return "Q2_K";
        case QuantType::Q3_K:   return "Q3_K";
        case QuantType::Q4_K:   return "Q4_K";
        case QuantType::Q5_K:   return "Q5_K";
        case QuantType::Q6_K:   return "Q6_K";
        case QuantType::Q8_K:   return "Q8_K";
        case QuantType::IQ4_NL: return "IQ4_NL";
        case QuantType::IQ4_XS: return "IQ4_XS";
        default:                return "OTHER";
    }
}

std::string HavenModelIntelligence::format_report(const ExecutionPlan& plan,
                                                  const std::vector<std::string>& loaded_modules) {
    const auto& fp = plan.fingerprint;
    const auto& hw = plan.hardware;
    std::ostringstream ss;

    ss << "========================================================================\n";
    ss << "  HAVEN MODEL INTELLIGENCE — ZERO-BLOAT MICROKERNEL ROUTING REPORT\n";
    ss << "========================================================================\n";
    ss << "  Model File      : " << fp.filepath << " (" << fmt_gib(fp.file_size_bytes) << ", GGUF v" << fp.gguf_version << ")\n";
    if (!fp.general_name.empty()) {
        ss << "  Model Name      : " << fp.general_name << "\n";
    }
    ss << "  Header Probe    : " << std::fixed << std::setprecision(2) << fp.probe_time_ms
       << " ms (zero weight mmap | " << fp.kv_count << " KVs | " << fp.tensor_count << " tensors)\n";
    ss << "  Architecture    : " << fp.architecture << " [" << arch_family_name(fp.arch_family) << "]\n";
    ss << "  Geometry        : " << fp.num_layers << "L | D=" << fp.embedding_dim
       << " | F=" << fp.hidden_dim << " | H=" << fp.num_heads << " (Hkv=" << fp.num_kv_heads
       << ", HD=" << fp.head_dim << (fp.head_dim_swa ? ("/SWA=" + std::to_string(fp.head_dim_swa)) : "")
       << ") | Vocab=" << fp.vocab_size << "\n";

    ss << "  Detected Traits :";
    bool any_trait = false;
    auto add_trait = [&](const std::string& t) {
        ss << (any_trait ? " | " : " ") << t;
        any_trait = true;
    };
    if (fp.has_moe) add_trait("MoE(" + std::to_string(fp.expert_count) + "E/Top-" + std::to_string(fp.expert_used_count ? fp.expert_used_count : 8u) + ")");
    if (fp.has_swa) add_trait("SWA(" + std::to_string(fp.swa_layer_count) + "L/" + std::to_string(fp.global_layer_count) + "G, win=" + std::to_string(fp.sliding_window) + ")");
    if (fp.has_shared_kv) add_trait("SharedKV(" + std::to_string(fp.shared_kv_layers) + "L)");
    if (fp.has_k_eq_v) add_trait("k_eq_v");
    if (fp.has_pleias) add_trait("Pleias");
    if (fp.has_qkv_bias) add_trait("QKV-Bias");
    if (fp.has_qk_norm) add_trait("QK-Norm");
    if (fp.has_turbo_bake) add_trait("Turbo-v" + std::to_string(fp.turbo_version) + "(FWHT=" + std::to_string(fp.turbo_hadamard_qkv) + ")");
    if (fp.has_vision_encoder) add_trait("Vision(" + std::to_string(fp.vision_tensor_count) + "t, " + fmt_gib(fp.vision_bytes) + ")");
    if (fp.has_audio_encoder) add_trait("Audio(" + std::to_string(fp.audio_tensor_count) + "t)");
    if (!any_trait) ss << " Standard Dense";
    ss << "\n";

    ss << "  Dominant Quant  : " << quant_type_name(fp.dominant_quant)
       << " (LLM weights=" << fmt_gib(fp.llm_weight_bytes)
       << ", KV@4K=" << fmt_gib(fp.kv_cache_bytes_4k)
       << ", Est VRAM=" << fmt_gib(fp.estimated_vram_bytes) << ")\n";
    ss << "  Chat Template   : " << chat_template_name(fp.chat_template_family)
       << (fp.supports_thinking_channel ? " [Thinking Channel]" : "") << "\n";
    ss << "------------------------------------------------------------------------\n";
    ss << "  Host Silicon    : " << hw.cpu_logical_cores << " threads"
       << (hw.has_avx2 ? " [AVX2/FMA]" : "")
       << " | RAM=" << fmt_gib(hw.available_ram_bytes) << "/" << fmt_gib(hw.total_ram_bytes)
       << " | CUDA=" << (hw.cuda_device_count > 0 ? hw.cuda_device_summary : "none")
       << (hw.tpu_available ? (" | TPU=" + hw.tpu_library_path) : "")
       << (hw.hmb_vmm_kernel_active ? (" | HMB-VMM=" + hw.hmb_vmm_node) : "") << "\n";
    ss << "  Routing Choice  : " << backend_kind_name(plan.selected_backend)
       << " (" << plan.decision_reason << ")\n";
    ss << "  Load Modules    : " << plan.selected_arch_module << ", " << plan.selected_backend_module;
    for (const auto& c : plan.required_capability_modules) ss << ", " << c;
    ss << "\n";
    ss << "  Skipped (0 RAM) : ";
    for (size_t i = 0; i < plan.skipped_modules.size(); ++i) {
        if (i > 0) ss << ", ";
        ss << plan.skipped_modules[i];
    }
    ss << "\n";
    if (plan.arch_contract_verified) {
        ss << "  Arch Contract   : " << plan.resolved_arch_spec.summary << "\n";
    }
    if (!loaded_modules.empty()) {
        ss << "  Resident DLL/SO : " << loaded_modules.size() << " active (";
        for (size_t i = 0; i < loaded_modules.size(); ++i) {
            if (i > 0) ss << ", ";
            ss << loaded_modules[i];
        }
        ss << ")\n";
    }
    ss << "========================================================================\n";
    return ss.str();
}

std::string HavenModelIntelligence::to_json(const ExecutionPlan& plan,
                                            const std::vector<std::string>& loaded_modules) {
    const auto& fp = plan.fingerprint;
    const auto& hw = plan.hardware;
    std::ostringstream ss;
    ss << "{";
    ss << "\"model\":{";
    ss << "\"filepath\":\"" << json_escape(fp.filepath) << "\",";
    ss << "\"name\":\"" << json_escape(fp.general_name) << "\",";
    ss << "\"architecture\":\"" << json_escape(fp.architecture) << "\",";
    ss << "\"arch_family\":\"" << arch_family_name(fp.arch_family) << "\",";
    ss << "\"probe_time_ms\":" << std::fixed << std::setprecision(3) << fp.probe_time_ms << ",";
    ss << "\"file_size_bytes\":" << fp.file_size_bytes << ",";
    ss << "\"llm_weight_bytes\":" << fp.llm_weight_bytes << ",";
    ss << "\"kv_cache_bytes_4k\":" << fp.kv_cache_bytes_4k << ",";
    ss << "\"estimated_vram_bytes\":" << fp.estimated_vram_bytes << ",";
    ss << "\"num_layers\":" << fp.num_layers << ",";
    ss << "\"embedding_dim\":" << fp.embedding_dim << ",";
    ss << "\"hidden_dim\":" << fp.hidden_dim << ",";
    ss << "\"num_heads\":" << fp.num_heads << ",";
    ss << "\"num_kv_heads\":" << fp.num_kv_heads << ",";
    ss << "\"head_dim\":" << fp.head_dim << ",";
    ss << "\"vocab_size\":" << fp.vocab_size << ",";
    ss << "\"dominant_quant\":\"" << quant_type_name(fp.dominant_quant) << "\",";
    ss << "\"has_moe\":" << (fp.has_moe ? "true" : "false") << ",";
    ss << "\"expert_count\":" << fp.expert_count << ",";
    ss << "\"has_swa\":" << (fp.has_swa ? "true" : "false") << ",";
    ss << "\"has_k_eq_v\":" << (fp.has_k_eq_v ? "true" : "false") << ",";
    ss << "\"has_pleias\":" << (fp.has_pleias ? "true" : "false") << ",";
    ss << "\"has_qkv_bias\":" << (fp.has_qkv_bias ? "true" : "false") << ",";
    ss << "\"has_qk_norm\":" << (fp.has_qk_norm ? "true" : "false") << ",";
    ss << "\"has_turbo_bake\":" << (fp.has_turbo_bake ? "true" : "false") << ",";
    ss << "\"has_vision_encoder\":" << (fp.has_vision_encoder ? "true" : "false") << ",";
    ss << "\"has_audio_encoder\":" << (fp.has_audio_encoder ? "true" : "false") << ",";
    ss << "\"chat_template\":\"" << json_escape(chat_template_name(fp.chat_template_family)) << "\",";
    ss << "\"supports_thinking\":" << (fp.supports_thinking_channel ? "true" : "false");
    ss << "},";
    ss << "\"hardware\":{";
    ss << "\"cpu_cores\":" << hw.cpu_logical_cores << ",";
    ss << "\"avx2\":" << (hw.has_avx2 ? "true" : "false") << ",";
    ss << "\"total_ram_bytes\":" << hw.total_ram_bytes << ",";
    ss << "\"available_ram_bytes\":" << hw.available_ram_bytes << ",";
    ss << "\"cuda_device_count\":" << hw.cuda_device_count << ",";
    ss << "\"cuda_device\":\"" << json_escape(hw.cuda_device_summary) << "\",";
    ss << "\"tpu_available\":" << (hw.tpu_available ? "true" : "false") << ",";
    ss << "\"hmb_vmm_active\":" << (hw.hmb_vmm_kernel_active ? "true" : "false");
    ss << "},";
    ss << "\"plan\":{";
    ss << "\"selected_arch_module\":\"" << json_escape(plan.selected_arch_module) << "\",";
    ss << "\"selected_backend_module\":\"" << json_escape(plan.selected_backend_module) << "\",";
    ss << "\"selected_backend\":\"" << backend_kind_name(plan.selected_backend) << "\",";
    ss << "\"decision_reason\":\"" << json_escape(plan.decision_reason) << "\",";
    ss << "\"arch_contract_verified\":" << (plan.arch_contract_verified ? "true" : "false") << ",";
    ss << "\"arch_summary\":\"" << json_escape(plan.arch_contract_verified ? std::string(plan.resolved_arch_spec.summary) : "") << "\",";
    ss << "\"required_capabilities\":[";
    for (size_t i = 0; i < plan.required_capability_modules.size(); ++i) {
        if (i > 0) ss << ",";
        ss << "\"" << json_escape(plan.required_capability_modules[i]) << "\"";
    }
    ss << "],\"skipped_modules\":[";
    for (size_t i = 0; i < plan.skipped_modules.size(); ++i) {
        if (i > 0) ss << ",";
        ss << "\"" << json_escape(plan.skipped_modules[i]) << "\"";
    }
    ss << "],\"resident_modules\":[";
    for (size_t i = 0; i < loaded_modules.size(); ++i) {
        if (i > 0) ss << ",";
        ss << "\"" << json_escape(loaded_modules[i]) << "\"";
    }
    ss << "]}}";
    return ss.str();
}

} // namespace haven
