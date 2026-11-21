#include "haven/haven_gguf.h"
#include <iostream>
#include <cstring>

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
    GetFileSizeEx(hFile, &size);
    file_size_ = (size_t)size.QuadPart;

    HANDLE hMap = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
    CloseHandle(hFile);

    if (!hMap) {
        std::cerr << "Failed to create file mapping for: " << filepath << "\n";
        return false;
    }

    mmap_handle_ = hMap;
    mmap_data_ = reinterpret_cast<const uint8_t*>(MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0));
    if (!mmap_data_) {
        std::cerr << "Failed to map view of file: " << filepath << "\n";
        return false;
    }
#else
    int fd = open(filepath.c_str(), O_RDONLY);
    if (fd < 0) return false;
    struct stat sb;
    fstat(fd, &sb);
    file_size_ = sb.st_size;
    mmap_data_ = (const uint8_t*)mmap(NULL, file_size_, PROT_READ, MAP_SHARED, fd, 0);
    ::close(fd);
#endif

    if (!parse_header()) {
        std::cerr << "Invalid GGUF header.\n";
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

static std::string read_gguf_string(const uint8_t*& ptr, const uint8_t* end) {
    if (ptr + 8 > end) return "";
    uint64_t len = *reinterpret_cast<const uint64_t*>(ptr);
    ptr += 8;
    if (ptr + len > end) return "";
    std::string s(reinterpret_cast<const char*>(ptr), len);
    ptr += len;
    return s;
}

static void skip_gguf_value(uint32_t val_type, const uint8_t*& ptr, const uint8_t* end) {
    if (val_type == 0 || val_type == 1 || val_type == 7) ptr += 1;
    else if (val_type == 2 || val_type == 3) ptr += 2;
    else if (val_type == 4 || val_type == 5 || val_type == 6) ptr += 4;
    else if (val_type == 10 || val_type == 11 || val_type == 12) ptr += 8;
    else if (val_type == 8) { // string
        if (ptr + 8 <= end) {
            uint64_t slen = *reinterpret_cast<const uint64_t*>(ptr);
            ptr += 8 + slen;
        }
    }
    else if (val_type == 9) { // array
        if (ptr + 12 <= end) {
            uint32_t arr_type = *reinterpret_cast<const uint32_t*>(ptr);
            uint64_t arr_len = *reinterpret_cast<const uint64_t*>(ptr + 4);
            ptr += 12;
            if (arr_type == 0 || arr_type == 1 || arr_type == 7) {
                ptr += arr_len * 1;
            } else if (arr_type == 2 || arr_type == 3) {
                ptr += arr_len * 2;
            } else if (arr_type == 4 || arr_type == 5 || arr_type == 6) {
                ptr += arr_len * 4;
            } else if (arr_type == 10 || arr_type == 11 || arr_type == 12) {
                ptr += arr_len * 8;
            } else if (arr_type == 8) { // array of strings (e.g. tokenizer merges)
                for (uint64_t i = 0; i < arr_len && ptr + 8 <= end; ++i) {
                    uint64_t slen = *reinterpret_cast<const uint64_t*>(ptr);
                    ptr += 8 + slen;
                }
            } else {
                for (uint64_t i = 0; i < arr_len && ptr < end; ++i) {
                    skip_gguf_value(arr_type, ptr, end);
                }
            }
        }
    }
}

bool GgufLoader::parse_header() {
    if (file_size_ < 24) return false;

    // Check GGUF Magic: "GGUF" in ASCII (0x46554747)
    uint32_t magic = *reinterpret_cast<const uint32_t*>(mmap_data_);
    if (magic != 0x46554747) {
        std::cerr << "Magic header mismatch: " << std::hex << magic << "\n";
        return false;
    }

    [[maybe_unused]] uint32_t version = *reinterpret_cast<const uint32_t*>(mmap_data_ + 4);
    uint64_t tensor_count = *reinterpret_cast<const uint64_t*>(mmap_data_ + 8);
    uint64_t kv_count = *reinterpret_cast<const uint64_t*>(mmap_data_ + 16);

    const uint8_t* ptr = mmap_data_ + 24;
    const uint8_t* end = mmap_data_ + file_size_;

    uint32_t alignment = 32;

    // Parse Metadata Key-Value pairs
    auto ends_with = [](const std::string& str, const std::string& suffix) -> bool {
        if (str.length() < suffix.length()) return false;
        return str.compare(str.length() - suffix.length(), suffix.length(), suffix) == 0;
    };

    auto read_float_val = [](uint32_t val_type, const uint8_t*& p, const uint8_t* e) -> float {
        if (val_type == 6 && p + 4 <= e) {
            float v = *reinterpret_cast<const float*>(p);
            p += 4;
            return v;
        } else if (val_type == 12 && p + 8 <= e) {
            double v = *reinterpret_cast<const double*>(p);
            p += 8;
            return static_cast<float>(v);
        } else if (val_type == 4 && p + 4 <= e) {
            uint32_t v = *reinterpret_cast<const uint32_t*>(p);
            p += 4;
            return static_cast<float>(v);
        }
        return 0.0f;
    };

    for (uint64_t i = 0; i < kv_count && ptr < end; ++i) {
        std::string key = read_gguf_string(ptr, end);
        if (ptr + 4 > end) break;
        uint32_t val_type = *reinterpret_cast<const uint32_t*>(ptr);
        ptr += 4;

        if (key == "general.architecture" && val_type == 8) {
            config_.architecture = read_gguf_string(ptr, end);
        }
        else if (key == "general.alignment" && val_type == 4) {
            alignment = *reinterpret_cast<const uint32_t*>(ptr);
            if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
                alignment = 32;
            }
            ptr += 4;
        }
        else if (ends_with(key, ".context_length") && (val_type == 4 || val_type == 10)) {
            config_.max_context_length = (val_type == 4) ? *reinterpret_cast<const uint32_t*>(ptr) : (uint32_t)(*reinterpret_cast<const uint64_t*>(ptr));
            ptr += (val_type == 4 ? 4 : 8);
        }
        else if (ends_with(key, ".embedding_length") && (val_type == 4 || val_type == 10)) {
            config_.embedding_dim = (val_type == 4) ? *reinterpret_cast<const uint32_t*>(ptr) : (uint32_t)(*reinterpret_cast<const uint64_t*>(ptr));
            ptr += (val_type == 4 ? 4 : 8);
        }
        else if (ends_with(key, ".feed_forward_length") && (val_type == 4 || val_type == 10)) {
            config_.hidden_dim = (val_type == 4) ? *reinterpret_cast<const uint32_t*>(ptr) : (uint32_t)(*reinterpret_cast<const uint64_t*>(ptr));
            ptr += (val_type == 4 ? 4 : 8);
        }
        else if (ends_with(key, ".block_count") && (val_type == 4 || val_type == 10)) {
            config_.num_layers = (val_type == 4) ? *reinterpret_cast<const uint32_t*>(ptr) : (uint32_t)(*reinterpret_cast<const uint64_t*>(ptr));
            ptr += (val_type == 4 ? 4 : 8);
        }
        else if (ends_with(key, ".attention.head_count") && (val_type == 4 || val_type == 10)) {
            config_.num_heads = (val_type == 4) ? *reinterpret_cast<const uint32_t*>(ptr) : (uint32_t)(*reinterpret_cast<const uint64_t*>(ptr));
            ptr += (val_type == 4 ? 4 : 8);
        }
        else if (ends_with(key, ".attention.head_count_kv") && (val_type == 4 || val_type == 10)) {
            config_.num_kv_heads = (val_type == 4) ? *reinterpret_cast<const uint32_t*>(ptr) : (uint32_t)(*reinterpret_cast<const uint64_t*>(ptr));
            ptr += (val_type == 4 ? 4 : 8);
        }
        else if (ends_with(key, ".attention.key_length") && (val_type == 4 || val_type == 10)) {
            config_.head_dim = (val_type == 4) ? *reinterpret_cast<const uint32_t*>(ptr) : (uint32_t)(*reinterpret_cast<const uint64_t*>(ptr));
            ptr += (val_type == 4 ? 4 : 8);
        }
        else if (ends_with(key, ".attention.sliding_window") && (val_type == 4 || val_type == 10)) {
            config_.sliding_window = (val_type == 4) ? *reinterpret_cast<const uint32_t*>(ptr) : (uint32_t)(*reinterpret_cast<const uint64_t*>(ptr));
            ptr += (val_type == 4 ? 4 : 8);
        }
        else if (ends_with(key, ".attention.layer_norm_rms_epsilon")) {
            config_.rms_norm_eps = read_float_val(val_type, ptr, end);
        }
        else if (ends_with(key, ".rope.freq_base_swa")) {
            config_.rope_freq_base_swa = read_float_val(val_type, ptr, end);
        }
        else if (ends_with(key, ".rope.freq_base")) {
            config_.rope_freq_base = read_float_val(val_type, ptr, end);
        }
        else if (ends_with(key, ".rope.dimension_count_swa") && (val_type == 4 || val_type == 10)) {
            config_.rope_dimension_count_swa = (val_type == 4) ? *reinterpret_cast<const uint32_t*>(ptr) : (uint32_t)(*reinterpret_cast<const uint64_t*>(ptr));
            ptr += (val_type == 4 ? 4 : 8);
        }
        else if (ends_with(key, ".rope.dimension_count") && (val_type == 4 || val_type == 10)) {
            config_.rope_dimension_count = (val_type == 4) ? *reinterpret_cast<const uint32_t*>(ptr) : (uint32_t)(*reinterpret_cast<const uint64_t*>(ptr));
            ptr += (val_type == 4 ? 4 : 8);
        }
        else if (ends_with(key, ".final_logit_softcapping")) {
            config_.final_logit_softcapping = read_float_val(val_type, ptr, end);
        }
        else if (ends_with(key, ".attn_logit_softcapping")) {
            config_.attn_logit_softcapping = read_float_val(val_type, ptr, end);
        }
        else if (key == "tokenizer.ggml.tokens" && val_type == 9) {
            [[maybe_unused]] uint32_t arr_type = *reinterpret_cast<const uint32_t*>(ptr);
            uint64_t arr_len = *reinterpret_cast<const uint64_t*>(ptr + 4);
            ptr += 12;
            config_.vocab_size = (uint32_t)arr_len;
            vocab_.reserve(arr_len);
            for (uint64_t t = 0; t < arr_len && ptr < end; ++t) {
                vocab_.push_back(read_gguf_string(ptr, end));
            }
        }
        else {
            skip_gguf_value(val_type, ptr, end);
        }
    }

    // Determine derived architecture hyperparameters
    config_.is_gemma = (config_.architecture.find("gemma") != std::string::npos);
    if (config_.is_gemma) {
        config_.ffn_activation = FfnActivation::GELU;
        if (config_.final_logit_softcapping == 0.0f) config_.final_logit_softcapping = 30.0f;
        if (config_.attn_logit_softcapping == 0.0f) config_.attn_logit_softcapping = 50.0f;
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
    parsed_tensors.reserve(tensor_count);

    for (uint64_t i = 0; i < tensor_count && ptr < end; ++i) {
        std::string name = read_gguf_string(ptr, end);
        if (ptr + 4 > end) break;
        uint32_t n_dims = *reinterpret_cast<const uint32_t*>(ptr);
        ptr += 4;

        TensorDesc desc;
        desc.name = name;
        desc.shape.resize(n_dims);
        for (uint32_t d = 0; d < n_dims && ptr + 8 <= end; ++d) {
            desc.shape[d] = *reinterpret_cast<const uint64_t*>(ptr);
            ptr += 8;
        }

        if (ptr + 12 > end) break;
        desc.type = static_cast<QuantType>(*reinterpret_cast<const uint32_t*>(ptr));
        desc.offset = *reinterpret_cast<const uint64_t*>(ptr + 4);
        ptr += 12;

        parsed_tensors.push_back({name, desc});
    }

    // Calculate Data Offset (aligned to alignment boundary)
    if (alignment == 0) alignment = 32;
    uint64_t current_pos = static_cast<uint64_t>(ptr - mmap_data_);
    uint64_t data_start = (current_pos + alignment - 1) & ~(static_cast<uint64_t>(alignment - 1));

    if (data_start > file_size_) {
        std::cerr << "⛔ Data start offset exceeds mapped file size\n";
        return false;
    }

    auto get_tensor_data_size = [](QuantType type, const std::vector<uint64_t>& shape) -> size_t {
        if (shape.empty()) return 0;
        uint64_t n_elems = 1;
        for (uint64_t dim : shape) {
            if (dim == 0) return 0;
            n_elems *= dim;
        }
        switch (type) {
            case QuantType::F32:  return static_cast<size_t>(n_elems * 4);
            case QuantType::F16:
            case QuantType::BF16: return static_cast<size_t>(n_elems * 2);
            case QuantType::Q8_0: return static_cast<size_t>((n_elems + 31) / 32 * sizeof(block_q8_0));
            case QuantType::Q4_0: return static_cast<size_t>((n_elems + 31) / 32 * sizeof(block_q4_0));
            case QuantType::Q5_0: return static_cast<size_t>((n_elems + 31) / 32 * sizeof(block_q5_0));
            case QuantType::Q4_K: return static_cast<size_t>((n_elems + 255) / 256 * sizeof(block_q4_K));
            case QuantType::Q5_K: return static_cast<size_t>((n_elems + 255) / 256 * sizeof(block_q5_K));
            case QuantType::Q6_K: return static_cast<size_t>((n_elems + 255) / 256 * sizeof(block_q6_K));
            case QuantType::I8:   return static_cast<size_t>(n_elems);
            case QuantType::I16:  return static_cast<size_t>(n_elems * 2);
            case QuantType::I32:  return static_cast<size_t>(n_elems * 4);
            case QuantType::I64:
            case QuantType::F64:  return static_cast<size_t>(n_elems * 8);
            default: return static_cast<size_t>(n_elems);
        }
    };

    for (auto& item : parsed_tensors) {
        TensorDesc& desc = item.second;
        desc.size_bytes = get_tensor_data_size(desc.type, desc.shape);
        if (desc.offset > file_size_ ||
            data_start > file_size_ - desc.offset ||
            desc.size_bytes > file_size_ - (data_start + desc.offset)) 
        {
            std::cerr << "⛔ Corrupted tensor offset/size for: " << desc.name << "\n";
            return false;
        }
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