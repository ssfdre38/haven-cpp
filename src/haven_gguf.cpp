#include "haven/haven_gguf.h"
#include <iostream>
#include <cstring>
#include <limits>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

namespace haven {

GgufLoader::GgufLoader() = default;

GgufLoader::~GgufLoader() {
    close();
}

bool GgufLoader::load_file(const std::string& filepath) {
    close();

#ifdef _WIN32
    HANDLE hFile = CreateFileA(filepath.c_str(), GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_RANDOM_ACCESS, NULL);
    if (hFile == INVALID_HANDLE_VALUE) {
        std::cerr << "Failed to open GGUF file: " << filepath << "\n";
        return false;
    }

    LARGE_INTEGER size;
    if (!GetFileSizeEx(hFile, &size) || size.QuadPart <= 0) {
        CloseHandle(hFile);
        std::cerr << "Invalid or empty GGUF file: " << filepath << "\n";
        return false;
    }
    file_size_ = static_cast<size_t>(size.QuadPart);

    HANDLE hMap = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
    CloseHandle(hFile);

    if (!hMap) {
        file_size_ = 0;
        std::cerr << "Failed to create file mapping for: " << filepath << "\n";
        return false;
    }

    mmap_handle_ = hMap;
    mmap_data_ = reinterpret_cast<const uint8_t*>(MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0));
    if (!mmap_data_) {
        close();
        std::cerr << "Failed to map view of file: " << filepath << "\n";
        return false;
    }
#else
    int fd = open(filepath.c_str(), O_RDONLY);
    if (fd < 0) return false;
    struct stat sb;
    if (fstat(fd, &sb) != 0 || sb.st_size <= 0) {
        ::close(fd);
        return false;
    }
    file_size_ = static_cast<size_t>(sb.st_size);
    void* mapped = mmap(NULL, file_size_, PROT_READ, MAP_SHARED, fd, 0);
    ::close(fd);
    if (mapped == MAP_FAILED || mapped == nullptr) {
        file_size_ = 0;
        return false;
    }
    mmap_data_ = static_cast<const uint8_t*>(mapped);
#endif

    if (!parse_header()) {
        std::cerr << "Invalid GGUF header.\n";
        close();
        return false;
    }

    return true;
}

void GgufLoader::close() {
#ifdef _WIN32
    if (mmap_data_) {
        UnmapViewOfFile(mmap_data_);
        mmap_data_ = nullptr;
    }
    if (mmap_handle_) {
        CloseHandle(reinterpret_cast<HANDLE>(mmap_handle_));
        mmap_handle_ = nullptr;
    }
#else
    if (mmap_data_) {
        munmap((void*)mmap_data_, file_size_);
        mmap_data_ = nullptr;
    }
#endif
    tensors_.clear();
    vocab_.clear();
    file_size_ = 0;
}

static bool has_bytes(const uint8_t* ptr, const uint8_t* end, uint64_t needed) {
    if (ptr > end) return false;
    return needed <= static_cast<uint64_t>(end - ptr);
}

static std::string read_gguf_string(const uint8_t*& ptr, const uint8_t* end, bool& ok) {
    if (!has_bytes(ptr, end, 8)) {
        ptr = end;
        ok = false;
        return "";
    }
    uint64_t len = 0;
    std::memcpy(&len, ptr, sizeof(len));
    ptr += 8;
    constexpr uint64_t MAX_GGUF_STR_LEN = 64ULL * 1024 * 1024;
    if (len > MAX_GGUF_STR_LEN || !has_bytes(ptr, end, len)) {
        ptr = end;
        ok = false;
        return "";
    }
    std::string s(reinterpret_cast<const char*>(ptr), static_cast<size_t>(len));
    ptr += len;
    return s;
}

static bool skip_gguf_value(uint32_t val_type, const uint8_t*& ptr, const uint8_t* end, int depth = 0) {
    if (depth > 16 || ptr > end) {
        ptr = end;
        return false;
    }
    auto advance = [&](uint64_t n) -> bool {
        if (!has_bytes(ptr, end, n)) {
            ptr = end;
            return false;
        }
        ptr += n;
        return true;
    };

    if (val_type == 0 || val_type == 1 || val_type == 7) return advance(1);
    if (val_type == 2 || val_type == 3) return advance(2);
    if (val_type == 4 || val_type == 5 || val_type == 6) return advance(4);
    if (val_type == 10 || val_type == 11 || val_type == 12) return advance(8);
    if (val_type == 8) { // string
        if (!has_bytes(ptr, end, 8)) { ptr = end; return false; }
        uint64_t slen = 0;
        std::memcpy(&slen, ptr, sizeof(slen));
        ptr += 8;
        return advance(slen);
    }
    if (val_type == 9) { // array
        if (!has_bytes(ptr, end, 12)) { ptr = end; return false; }
        uint32_t arr_type = 0;
        uint64_t arr_len = 0;
        std::memcpy(&arr_type, ptr, sizeof(arr_type));
        std::memcpy(&arr_len, ptr + 4, sizeof(arr_len));
        ptr += 12;

        uint64_t elem_size = 0;
        if (arr_type == 0 || arr_type == 1 || arr_type == 7) elem_size = 1;
        else if (arr_type == 2 || arr_type == 3) elem_size = 2;
        else if (arr_type == 4 || arr_type == 5 || arr_type == 6) elem_size = 4;
        else if (arr_type == 10 || arr_type == 11 || arr_type == 12) elem_size = 8;

        if (elem_size > 0) {
            uint64_t avail = static_cast<uint64_t>(end - ptr);
            if (arr_len > avail / elem_size) {
                ptr = end;
                return false;
            }
            ptr += arr_len * elem_size;
            return true;
        }
        if (arr_type == 8) {
            for (uint64_t i = 0; i < arr_len; ++i) {
                if (!has_bytes(ptr, end, 8)) { ptr = end; return false; }
                uint64_t slen = 0;
                std::memcpy(&slen, ptr, sizeof(slen));
                ptr += 8;
                if (!advance(slen)) return false;
            }
            return true;
        }
        for (uint64_t i = 0; i < arr_len; ++i) {
            if (!skip_gguf_value(arr_type, ptr, end, depth + 1)) return false;
        }
        return true;
    }

    // Unknown type: abort parse safely
    ptr = end;
    return false;
}

bool GgufLoader::parse_header() {
    if (!mmap_data_ || file_size_ < 24) return false;

    // Check GGUF Magic: "GGUF" in ASCII (0x46554747)
    uint32_t magic = 0;
    std::memcpy(&magic, mmap_data_, sizeof(magic));
    if (magic != 0x46554747) {
        std::cerr << "Magic header mismatch: " << std::hex << magic << std::dec << "\n";
        return false;
    }

    uint32_t version = 0;
    uint64_t tensor_count = 0;
    uint64_t kv_count = 0;
    std::memcpy(&version, mmap_data_ + 4, sizeof(version));
    std::memcpy(&tensor_count, mmap_data_ + 8, sizeof(tensor_count));
    std::memcpy(&kv_count, mmap_data_ + 16, sizeof(kv_count));
    (void)version;

    constexpr uint64_t MAX_GGUF_KV = 1000000ULL;
    constexpr uint64_t MAX_GGUF_TENSORS = 1000000ULL;
    if (kv_count > MAX_GGUF_KV || tensor_count > MAX_GGUF_TENSORS) {
        std::cerr << "⛔ GGUF metadata or tensor count exceeds safe bounds\n";
        return false;
    }

    const uint8_t* ptr = mmap_data_ + 24;
    const uint8_t* end = mmap_data_ + file_size_;

    uint32_t alignment = 32;

    auto ends_with = [](const std::string& str, const std::string& suffix) -> bool {
        if (str.length() < suffix.length()) return false;
        return str.compare(str.length() - suffix.length(), suffix.length(), suffix) == 0;
    };

    auto read_uint_val = [&](uint32_t val_type, uint32_t& out) -> bool {
        if ((val_type == 4 || val_type == 5) && has_bytes(ptr, end, 4)) {
            uint32_t v = 0;
            std::memcpy(&v, ptr, sizeof(v));
            ptr += 4;
            out = v;
            return true;
        }
        if ((val_type == 10 || val_type == 11) && has_bytes(ptr, end, 8)) {
            uint64_t v = 0;
            std::memcpy(&v, ptr, sizeof(v));
            ptr += 8;
            out = static_cast<uint32_t>(v);
            return true;
        }
        return false;
    };

    auto read_float_val = [&](uint32_t val_type, float& out) -> bool {
        if (val_type == 6 && has_bytes(ptr, end, 4)) {
            float v = 0.0f;
            std::memcpy(&v, ptr, sizeof(v));
            ptr += 4;
            out = v;
            return true;
        }
        if (val_type == 12 && has_bytes(ptr, end, 8)) {
            double v = 0.0;
            std::memcpy(&v, ptr, sizeof(v));
            ptr += 8;
            out = static_cast<float>(v);
            return true;
        }
        if (val_type == 4 && has_bytes(ptr, end, 4)) {
            uint32_t v = 0;
            std::memcpy(&v, ptr, sizeof(v));
            ptr += 4;
            out = static_cast<float>(v);
            return true;
        }
        return skip_gguf_value(val_type, ptr, end);
    };

    for (uint64_t i = 0; i < kv_count; ++i) {
        bool str_ok = true;
        std::string key = read_gguf_string(ptr, end, str_ok);
        if (!str_ok || !has_bytes(ptr, end, 4)) return false;

        uint32_t val_type = 0;
        std::memcpy(&val_type, ptr, sizeof(val_type));
        ptr += 4;

        // Ignore multimodal tower sub-configs (.vision.*, .audio.*) so they never overwrite LLM geometry
        if (key.find(".vision.") != std::string::npos || key.find(".audio.") != std::string::npos) {
            if (!skip_gguf_value(val_type, ptr, end)) return false;
        }
        else if (key == "general.architecture" && val_type == 8) {
            config_.architecture = read_gguf_string(ptr, end, str_ok);
            if (!str_ok) return false;
        }
        else if (key == "general.alignment" && val_type == 4) {
            uint32_t al = 32;
            if (!read_uint_val(val_type, al)) return false;
            if (al == 0 || (al & (al - 1)) != 0 || al > 4096) al = 32;
            alignment = al;
        }
        else if (ends_with(key, ".context_length") && (val_type == 4 || val_type == 10)) {
            if (!read_uint_val(val_type, config_.max_context_length)) return false;
        }
        else if (ends_with(key, ".embedding_length") && (val_type == 4 || val_type == 10)) {
            if (!read_uint_val(val_type, config_.embedding_dim)) return false;
        }
        else if (ends_with(key, ".feed_forward_length") && (val_type == 4 || val_type == 10)) {
            if (!read_uint_val(val_type, config_.hidden_dim)) return false;
        }
        else if (ends_with(key, ".feed_forward_length") && val_type == 9) {
            if (!has_bytes(ptr, end, 12)) return false;
            uint32_t arr_type = 0;
            uint64_t arr_len = 0;
            std::memcpy(&arr_type, ptr, sizeof(arr_type));
            std::memcpy(&arr_len, ptr + 4, sizeof(arr_len));
            ptr += 12;
            if ((arr_type != 4 && arr_type != 5 && arr_type != 10 && arr_type != 11) || arr_len > 4096) return false;
            config_.hidden_dim_per_layer.clear();
            config_.hidden_dim_per_layer.reserve(static_cast<size_t>(arr_len));
            uint32_t max_ff = 0;
            for (uint64_t j = 0; j < arr_len; ++j) {
                uint32_t v = 0;
                if (!read_uint_val(arr_type, v)) return false;
                config_.hidden_dim_per_layer.push_back(v);
                if (v > max_ff) max_ff = v;
            }
            if (max_ff > 0) config_.hidden_dim = max_ff;
        }
        else if (ends_with(key, ".block_count") && (val_type == 4 || val_type == 10)) {
            if (!read_uint_val(val_type, config_.num_layers)) return false;
        }
        else if (ends_with(key, ".attention.head_count") && (val_type == 4 || val_type == 10)) {
            if (!read_uint_val(val_type, config_.num_heads)) return false;
        }
        else if (ends_with(key, ".attention.head_count_kv") && (val_type == 4 || val_type == 10)) {
            if (!read_uint_val(val_type, config_.num_kv_heads)) return false;
        }
        else if (ends_with(key, ".attention.head_count_kv") && val_type == 9) {
            if (!has_bytes(ptr, end, 12)) return false;
            uint32_t arr_type = 0;
            uint64_t arr_len = 0;
            std::memcpy(&arr_type, ptr, sizeof(arr_type));
            std::memcpy(&arr_len, ptr + 4, sizeof(arr_len));
            ptr += 12;
            if ((arr_type != 4 && arr_type != 5 && arr_type != 10 && arr_type != 11) || arr_len > 4096) return false;
            config_.head_count_kv_per_layer.clear();
            config_.head_count_kv_per_layer.reserve(static_cast<size_t>(arr_len));
            uint32_t max_kv_h = 0;
            for (uint64_t j = 0; j < arr_len; ++j) {
                uint32_t v = 0;
                if (!read_uint_val(arr_type, v)) return false;
                config_.head_count_kv_per_layer.push_back(v);
                if (v > max_kv_h) max_kv_h = v;
            }
            if (max_kv_h > 0) config_.num_kv_heads = max_kv_h;
        }
        else if (ends_with(key, ".attention.sliding_window_pattern") && val_type == 9) {
            if (!has_bytes(ptr, end, 12)) return false;
            uint32_t arr_type = 0;
            uint64_t arr_len = 0;
            std::memcpy(&arr_type, ptr, sizeof(arr_type));
            std::memcpy(&arr_len, ptr + 4, sizeof(arr_len));
            ptr += 12;
            if (arr_type != 7 || arr_len > 4096 || !has_bytes(ptr, end, arr_len)) return false;
            config_.sliding_window_pattern.resize(static_cast<size_t>(arr_len));
            for (uint64_t j = 0; j < arr_len; ++j) {
                config_.sliding_window_pattern[static_cast<size_t>(j)] = (ptr[j] != 0);
            }
            ptr += arr_len;
        }
        else if (ends_with(key, ".attention.key_length_swa") && (val_type == 4 || val_type == 10)) {
            if (!read_uint_val(val_type, config_.head_dim_swa)) return false;
        }
        else if (ends_with(key, ".attention.key_length") && (val_type == 4 || val_type == 10)) {
            if (!read_uint_val(val_type, config_.head_dim)) return false;
        }
        else if (ends_with(key, ".attention.sliding_window") && (val_type == 4 || val_type == 10)) {
            if (!read_uint_val(val_type, config_.sliding_window)) return false;
        }
        else if (ends_with(key, ".attention.shared_kv_layers") && (val_type == 4 || val_type == 10)) {
            if (!read_uint_val(val_type, config_.shared_kv_layers)) return false;
            config_.has_shared_kv_layers_kv = true;
        }
        else if (ends_with(key, ".expert_count") && (val_type == 4 || val_type == 10)) {
            if (!read_uint_val(val_type, config_.expert_count)) return false;
        }
        else if (ends_with(key, ".expert_used_count") && (val_type == 4 || val_type == 10)) {
            if (!read_uint_val(val_type, config_.expert_used_count)) return false;
        }
        else if (ends_with(key, ".expert_feed_forward_length") && (val_type == 4 || val_type == 10)) {
            if (!read_uint_val(val_type, config_.expert_ffn_dim)) return false;
        }
        else if (ends_with(key, ".attention.layer_norm_rms_epsilon")) {
            if (!read_float_val(val_type, config_.rms_norm_eps)) return false;
        }
        else if (ends_with(key, ".rope.freq_base_swa")) {
            if (!read_float_val(val_type, config_.rope_freq_base_swa)) return false;
        }
        else if (ends_with(key, ".rope.freq_base")) {
            if (!read_float_val(val_type, config_.rope_freq_base)) return false;
        }
        else if (ends_with(key, ".rope.dimension_count_swa") && (val_type == 4 || val_type == 10)) {
            if (!read_uint_val(val_type, config_.rope_dimension_count_swa)) return false;
        }
        else if (ends_with(key, ".rope.dimension_count") && (val_type == 4 || val_type == 10)) {
            if (!read_uint_val(val_type, config_.rope_dimension_count)) return false;
        }
        else if (ends_with(key, ".final_logit_softcapping")) {
            if (!read_float_val(val_type, config_.final_logit_softcapping)) return false;
        }
        else if (ends_with(key, ".attn_logit_softcapping")) {
            if (!read_float_val(val_type, config_.attn_logit_softcapping)) return false;
        }
        else if (ends_with(key, ".turbo.version") && (val_type == 4 || val_type == 10)) {
            if (!read_uint_val(val_type, config_.turbo_version)) return false;
        }
        else if (ends_with(key, ".turbo.hadamard_qkv") && (val_type == 4 || val_type == 10)) {
            uint32_t b = 0;
            if (!read_uint_val(val_type, b)) return false;
            if (b >= 8 && b <= 1024 && (b & (b - 1)) == 0) config_.turbo_hadamard_qkv = b;
        }
        else if (ends_with(key, ".turbo.hadamard_ffn") && (val_type == 4 || val_type == 10)) {
            uint32_t b = 0;
            if (!read_uint_val(val_type, b)) return false;
            if (b >= 8 && b <= 1024 && (b & (b - 1)) == 0) config_.turbo_hadamard_ffn = b;
        }
        else if (ends_with(key, ".turbo.hadamard_ffn_down") && (val_type == 4 || val_type == 10)) {
            uint32_t b = 0;
            if (!read_uint_val(val_type, b)) return false;
            if (b >= 8 && b <= 1024 && (b & (b - 1)) == 0) config_.turbo_hadamard_ffn_down = b;
        }
        else if (ends_with(key, ".turbo.hadamard_moe_in") && (val_type == 4 || val_type == 10)) {
            uint32_t b = 0;
            if (!read_uint_val(val_type, b)) return false;
            if (b >= 8 && b <= 1024 && (b & (b - 1)) == 0) config_.turbo_hadamard_moe_in = b;
        }
        else if (ends_with(key, ".turbo.hadamard_moe_down") && (val_type == 4 || val_type == 10)) {
            uint32_t b = 0;
            if (!read_uint_val(val_type, b)) return false;
            if (b >= 8 && b <= 1024 && (b & (b - 1)) == 0) config_.turbo_hadamard_moe_down = b;
        }
        else if (ends_with(key, ".turbo.folded_norms") && val_type == 7) {
            if (!has_bytes(ptr, end, 1)) return false;
            config_.turbo_folded_norms = (*ptr++ != 0);
        }
        else if (ends_with(key, ".turbo.folded_scales") && val_type == 7) {
            if (!has_bytes(ptr, end, 1)) return false;
            config_.turbo_folded_scales = (*ptr++ != 0);
        }
        else if (key == "tokenizer.ggml.bos_token_id" && (val_type == 4 || val_type == 5 || val_type == 10)) {
            if (!read_uint_val(val_type, config_.bos_token_id)) return false;
            config_.has_bos_token_id = true;
        }
        else if (key == "tokenizer.ggml.eos_token_id" && (val_type == 4 || val_type == 5 || val_type == 10)) {
            if (!read_uint_val(val_type, config_.eos_token_id)) return false;
            config_.has_eos_token_id = true;
        }
        else if (key == "tokenizer.ggml.padding_token_id" && (val_type == 4 || val_type == 5 || val_type == 10)) {
            if (!read_uint_val(val_type, config_.pad_token_id)) return false;
            config_.has_pad_token_id = true;
        }
        else if (key == "tokenizer.ggml.unknown_token_id" && (val_type == 4 || val_type == 5 || val_type == 10)) {
            if (!read_uint_val(val_type, config_.unk_token_id)) return false;
            config_.has_unk_token_id = true;
        }
        else if (key == "tokenizer.ggml.add_space_prefix" && val_type == 7) {
            if (!has_bytes(ptr, end, 1)) return false;
            config_.add_space_prefix = (*ptr++ != 0);
            config_.has_add_space_prefix = true;
        }
        else if (key == "tokenizer.ggml.tokens" && val_type == 9) {
            if (!has_bytes(ptr, end, 12)) return false;
            uint32_t arr_type = 0;
            uint64_t arr_len = 0;
            std::memcpy(&arr_type, ptr, sizeof(arr_type));
            std::memcpy(&arr_len, ptr + 4, sizeof(arr_len));
            ptr += 12;
            constexpr uint64_t MAX_VOCAB_SIZE = 2000000ULL;
            if (arr_type != 8 || arr_len > MAX_VOCAB_SIZE) {
                std::cerr << "⛔ Invalid GGUF vocabulary array type or size\n";
                return false;
            }
            config_.vocab_size = static_cast<uint32_t>(arr_len);
            vocab_.clear();
            vocab_.reserve(static_cast<size_t>(arr_len));
            for (uint64_t t = 0; t < arr_len; ++t) {
                std::string tok = read_gguf_string(ptr, end, str_ok);
                if (!str_ok) return false;
                vocab_.push_back(std::move(tok));
            }
        }
        else {
            if (!skip_gguf_value(val_type, ptr, end)) return false;
        }
    }

    // Determine derived architecture hyperparameters
    config_.is_gemma = (config_.architecture.find("gemma") != std::string::npos);
    if (config_.is_gemma) {
        config_.ffn_activation = FfnActivation::GELU;
        if (config_.final_logit_softcapping == 0.0f) config_.final_logit_softcapping = 30.0f;
        if (config_.attn_logit_softcapping == 0.0f && config_.architecture == "gemma2") {
            config_.attn_logit_softcapping = 50.0f;
        }
        if (!config_.has_shared_kv_layers_kv && config_.num_layers == 42) {
            config_.shared_kv_layers = 18; // Gemma 4 E4B default when key omitted
        }
    } else {
        config_.ffn_activation = FfnActivation::SILU;
        config_.final_logit_softcapping = 0.0f;
        config_.attn_logit_softcapping = 0.0f;
    }

    if (config_.head_dim == 0 && config_.num_heads > 0) {
        config_.head_dim = config_.embedding_dim / config_.num_heads;
    }
    if (config_.head_dim == 0) {
        config_.head_dim = 128;
    }
    if (config_.num_kv_heads == 0) {
        config_.num_kv_heads = config_.num_heads > 0 ? config_.num_heads : 8;
    }

    // Parse Tensor Info Headers
    std::vector<std::pair<std::string, TensorDesc>> parsed_tensors;
    parsed_tensors.reserve(static_cast<size_t>(tensor_count));

    for (uint64_t i = 0; i < tensor_count; ++i) {
        bool str_ok = true;
        std::string name = read_gguf_string(ptr, end, str_ok);
        if (!str_ok || !has_bytes(ptr, end, 4)) return false;

        uint32_t n_dims = 0;
        std::memcpy(&n_dims, ptr, sizeof(n_dims));
        ptr += 4;

        if (n_dims == 0 || n_dims > 8) {
            std::cerr << "⛔ Invalid tensor dimension count (" << n_dims << ") for: " << name << "\n";
            return false;
        }

        uint64_t dims_and_meta_bytes = static_cast<uint64_t>(n_dims) * 8ULL + 12ULL;
        if (!has_bytes(ptr, end, dims_and_meta_bytes)) {
            std::cerr << "⛔ Truncated tensor descriptor for: " << name << "\n";
            return false;
        }

        TensorDesc desc;
        desc.name = name;
        desc.shape.resize(n_dims);
        for (uint32_t d = 0; d < n_dims; ++d) {
            uint64_t dim_val = 0;
            std::memcpy(&dim_val, ptr, sizeof(dim_val));
            desc.shape[d] = dim_val;
            ptr += 8;
        }

        uint32_t raw_type = 0;
        uint64_t raw_offset = 0;
        std::memcpy(&raw_type, ptr, sizeof(raw_type));
        std::memcpy(&raw_offset, ptr + 4, sizeof(raw_offset));
        desc.type = static_cast<QuantType>(raw_type);
        desc.offset = raw_offset;
        ptr += 12;

        parsed_tensors.push_back({std::move(name), std::move(desc)});
    }

    // Calculate Data Offset (aligned to alignment boundary)
    if (alignment == 0) alignment = 32;
    uint64_t current_pos = static_cast<uint64_t>(ptr - mmap_data_);
    uint64_t align_mask = static_cast<uint64_t>(alignment - 1);
    if (current_pos > std::numeric_limits<uint64_t>::max() - align_mask) return false;
    uint64_t data_start = (current_pos + align_mask) & ~align_mask;

    if (!parsed_tensors.empty() && data_start > file_size_) {
        std::cerr << "⛔ Data start offset exceeds mapped file size\n";
        return false;
    }

    auto get_tensor_data_size = [](QuantType type, const std::vector<uint64_t>& shape) -> uint64_t {
        if (shape.empty()) return 0;
        uint64_t n_elems = 1;
        for (uint64_t dim : shape) {
            if (dim == 0) return 0;
            if (dim > std::numeric_limits<uint64_t>::max() / n_elems) return 0;
            n_elems *= dim;
        }

        auto safe_block_bytes = [](uint64_t elems, uint64_t block_elems, uint64_t block_bytes) -> uint64_t {
            if (elems > std::numeric_limits<uint64_t>::max() - (block_elems - 1)) return 0;
            uint64_t blocks = (elems + block_elems - 1) / block_elems;
            if (blocks > std::numeric_limits<uint64_t>::max() / block_bytes) return 0;
            return blocks * block_bytes;
        };

        switch (type) {
            case QuantType::F32:  return safe_block_bytes(n_elems, 1, 4);
            case QuantType::F16:
            case QuantType::BF16: return safe_block_bytes(n_elems, 1, 2);
            case QuantType::Q4_0: return safe_block_bytes(n_elems, 32, sizeof(block_q4_0));
            case QuantType::Q4_1: return safe_block_bytes(n_elems, 32, 20);
            case QuantType::Q5_0: return safe_block_bytes(n_elems, 32, sizeof(block_q5_0));
            case QuantType::Q5_1: return safe_block_bytes(n_elems, 32, sizeof(block_q5_1));
            case QuantType::Q8_0: return safe_block_bytes(n_elems, 32, sizeof(block_q8_0));
            case QuantType::Q8_1: return safe_block_bytes(n_elems, 32, 36);
            case QuantType::Q2_K: return safe_block_bytes(n_elems, 256, 84);
            case QuantType::Q3_K: return safe_block_bytes(n_elems, 256, sizeof(block_q3_K));
            case QuantType::Q4_K: return safe_block_bytes(n_elems, 256, sizeof(block_q4_K));
            case QuantType::Q5_K: return safe_block_bytes(n_elems, 256, sizeof(block_q5_K));
            case QuantType::Q6_K: return safe_block_bytes(n_elems, 256, sizeof(block_q6_K));
            case QuantType::Q8_K: return safe_block_bytes(n_elems, 256, 292);
            case QuantType::IQ4_NL: return safe_block_bytes(n_elems, 32, sizeof(block_iq4_nl));
            case QuantType::IQ4_XS: return safe_block_bytes(n_elems, 256, sizeof(block_iq4_xs));
            case QuantType::I8:   return safe_block_bytes(n_elems, 1, 1);
            case QuantType::I16:  return safe_block_bytes(n_elems, 1, 2);
            case QuantType::I32:  return safe_block_bytes(n_elems, 1, 4);
            case QuantType::I64:
            case QuantType::F64:  return safe_block_bytes(n_elems, 1, 8);
            default: return safe_block_bytes(n_elems, 1, 1);
        }
    };

    for (auto& item : parsed_tensors) {
        TensorDesc& desc = item.second;
        uint64_t expected_bytes = get_tensor_data_size(desc.type, desc.shape);
        if (expected_bytes == 0 ||
            desc.offset > file_size_ ||
            data_start > file_size_ - desc.offset ||
            expected_bytes > file_size_ - (data_start + desc.offset))
        {
            std::cerr << "⛔ Corrupted tensor offset/size for: " << desc.name << "\n";
            return false;
        }
        desc.size_bytes = static_cast<size_t>(expected_bytes);
        desc.data = mmap_data_ + data_start + desc.offset;
        tensors_[item.first] = desc;
    }

    std::cout << "   ✓ GGUF Loaded: Architecture [" << config_.architecture << "] ("
              << config_.num_layers << " layers, " << config_.embedding_dim << " hidden, "
              << config_.vocab_size << " vocab, " << tensors_.size() << " tensors mapped)\n";

    return true;
}

const TensorDesc* GgufLoader::get_tensor(const std::string& name) const {
    auto it = tensors_.find(name);
    if (it != tensors_.end()) {
        return &it->second;
    }
    return nullptr;
}

} // namespace haven
