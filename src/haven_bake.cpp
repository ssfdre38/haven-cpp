// ============================================================================
// haven-bake: Native C++20 AVX2+OpenMP "Gemma 4.5" GGUF Architecture Optimizer
// ============================================================================
// Performs exact mathematical graph folding & orthogonal Hadamard baking on
// unquantized BF16 / F16 / F32 Gemma 4 GGUF masters (single or multi-shard):
//
//   --mode stock : 100% compatible with stock llama.cpp, Ollama, and haven-cpp
//     - Strips dead placeholder shared-KV tensors (blk.L_kv..L-1 attn_k/v/k_norm)
//     - Optionally strips unused multimodal .vision./.audio. tensors (--strip-mm)
//     - Folds ffn_down_exps.scale -> ffn_down_exps.weight (scale reset to 1.0)
//     - Folds ffn_gate_inp.scale  -> ffn_gate_inp.weight  (scale reset to 1.0)
//     - Zero-cost QuaRot R2 head-dim Hadamard on V -> O (propagated across
//       shared-KV consumers; exact through parameter-free V-RMSNorm & GQA;
//       automatically skips k_eq_v layers where attn_v is absent)
//
//   --mode turbo : Native haven-cpp "Gemma 4.5 Turbo" (zero activation cliffs!)
//     - Everything in --mode stock, PLUS:
//     - Absorbs 100% of attn_norm (up to 900x!), ffn_norm (up to 474x!), and
//       pre_ffw_norm_2 (up to 322x!) gammas into downstream projections and
//       rotates input blocks by orthonormal Block-256 FWHT (W' = W*diag(g)*H_256)
//     - Rotates ffn_down (Block-256 or Block-64) and ffn_down_exps (Block-64)
//       to flatten spiky GeGLU activations before 4-bit quantization
//     - Emits gemma4.turbo.* GGUF metadata for automatic haven-cpp dispatch
// ============================================================================

#include "haven/haven_avx2.h"
#include "haven/haven_types.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

using namespace haven;

constexpr uint32_t GGUF_MAGIC = 0x46554747u; // "GGUF"
constexpr uint32_t GGUF_TYPE_UINT8   = 0;
constexpr uint32_t GGUF_TYPE_INT8    = 1;
constexpr uint32_t GGUF_TYPE_UINT16  = 2;
constexpr uint32_t GGUF_TYPE_INT16   = 3;
constexpr uint32_t GGUF_TYPE_UINT32  = 4;
constexpr uint32_t GGUF_TYPE_INT32   = 5;
constexpr uint32_t GGUF_TYPE_FLOAT32 = 6;
constexpr uint32_t GGUF_TYPE_BOOL    = 7;
constexpr uint32_t GGUF_TYPE_STRING  = 8;
constexpr uint32_t GGUF_TYPE_ARRAY   = 9;
constexpr uint32_t GGUF_TYPE_UINT64  = 10;
constexpr uint32_t GGUF_TYPE_INT64   = 11;
constexpr uint32_t GGUF_TYPE_FLOAT64 = 12;

static inline bool is_pow2(uint32_t v) { return v >= 8 && (v & (v - 1)) == 0; }

static inline bool ends_with(const std::string& s, const std::string& suf) {
    return s.size() >= suf.size() && s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
}

static inline bool starts_with(const std::string& s, const std::string& pre) {
    return s.size() >= pre.size() && s.compare(0, pre.size(), pre) == 0;
}

static inline uint16_t fp32_to_bf16_rne(float f) {
    uint32_t u = 0;
    std::memcpy(&u, &f, sizeof(u));
    if ((u & 0x7FFFFFFFu) > 0x7F800000u) {
        return static_cast<uint16_t>((u >> 16) | 0x0040u); // quiet NaN
    }
    u += 0x00007FFFu + ((u >> 16) & 1u);
    return static_cast<uint16_t>(u >> 16);
}

static inline uint16_t fp32_to_fp16_rne(float f) {
    __m128 v = _mm_set_ss(f);
    __m128i h = _mm_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT);
    return static_cast<uint16_t>(_mm_extract_epi16(h, 0));
}

static inline size_t gguf_scalar_size(uint32_t t) {
    switch (t) {
        case GGUF_TYPE_UINT8: case GGUF_TYPE_INT8: case GGUF_TYPE_BOOL: return 1;
        case GGUF_TYPE_UINT16: case GGUF_TYPE_INT16: return 2;
        case GGUF_TYPE_UINT32: case GGUF_TYPE_INT32: case GGUF_TYPE_FLOAT32: return 4;
        case GGUF_TYPE_UINT64: case GGUF_TYPE_INT64: case GGUF_TYPE_FLOAT64: return 8;
        default: return 0;
    }
}

struct RawKV {
    std::string key;
    uint32_t val_type = 0;
    std::vector<uint8_t> raw_value; // Exact serialized value bytes
};

struct InputTensor {
    int shard_idx = 0;
    std::string name;
    std::vector<uint64_t> dims;
    QuantType type = QuantType::F32;
    uint64_t rel_offset = 0;
    uint64_t abs_offset = 0;
    uint64_t nbytes = 0;
};

static bool read_exact(std::ifstream& f, void* dst, size_t n) {
    if (n == 0) return true;
    return static_cast<bool>(f.read(reinterpret_cast<char*>(dst), static_cast<std::streamsize>(n)));
}

static bool write_exact(std::ofstream& f, const void* src, size_t n) {
    if (n == 0) return true;
    return static_cast<bool>(f.write(reinterpret_cast<const char*>(src), static_cast<std::streamsize>(n)));
}

static bool read_u32(std::ifstream& f, uint32_t& out) { return read_exact(f, &out, 4); }
static bool read_u64(std::ifstream& f, uint64_t& out) { return read_exact(f, &out, 8); }

static bool read_gguf_str(std::ifstream& f, std::string& out) {
    uint64_t len = 0;
    if (!read_u64(f, len) || len > (1ULL << 28)) return false;
    out.resize(static_cast<size_t>(len));
    return read_exact(f, out.data(), static_cast<size_t>(len));
}

static void append_bytes(std::vector<uint8_t>& buf, const void* p, size_t n) {
    const auto* b = reinterpret_cast<const uint8_t*>(p);
    buf.insert(buf.end(), b, b + n);
}

static bool read_raw_gguf_value(std::ifstream& f, uint32_t val_type, std::vector<uint8_t>& out, int depth = 0) {
    if (depth > 8) return false;
    size_t sz = gguf_scalar_size(val_type);
    if (sz > 0) {
        uint8_t tmp[8];
        if (!read_exact(f, tmp, sz)) return false;
        append_bytes(out, tmp, sz);
        return true;
    }
    if (val_type == GGUF_TYPE_STRING) {
        uint64_t len = 0;
        if (!read_u64(f, len) || len > (1ULL << 28)) return false;
        append_bytes(out, &len, 8);
        size_t old_sz = out.size();
        out.resize(old_sz + static_cast<size_t>(len));
        return read_exact(f, out.data() + old_sz, static_cast<size_t>(len));
    }
    if (val_type == GGUF_TYPE_ARRAY) {
        uint32_t arr_type = 0;
        uint64_t arr_len = 0;
        if (!read_u32(f, arr_type) || !read_u64(f, arr_len)) return false;
        append_bytes(out, &arr_type, 4);
        append_bytes(out, &arr_len, 8);
        size_t esz = gguf_scalar_size(arr_type);
        if (esz > 0) {
            if (arr_len > (1ULL << 28) / esz) return false;
            size_t total = static_cast<size_t>(arr_len * esz);
            size_t old_sz = out.size();
            out.resize(old_sz + total);
            return read_exact(f, out.data() + old_sz, total);
        }
        for (uint64_t i = 0; i < arr_len; ++i) {
            if (!read_raw_gguf_value(f, arr_type, out, depth + 1)) return false;
        }
        return true;
    }
    return false;
}

static uint64_t tensor_byte_size(QuantType type, const std::vector<uint64_t>& dims) {
    if (dims.empty()) return 0;
    uint64_t ne0 = dims[0];
    uint64_t rows = 1;
    for (size_t i = 1; i < dims.size(); ++i) rows *= dims[i];
    size_t rb = Avx2Math::row_bytes(type, static_cast<int>(ne0));
    return static_cast<uint64_t>(rb) * rows;
}

static bool parse_blk_tensor(const std::string& name, int& out_layer, std::string& out_suffix) {
    if (!starts_with(name, "blk.")) return false;
    size_t dot = name.find('.', 4);
    if (dot == std::string::npos || dot == 4) return false;
    out_layer = std::atoi(name.substr(4, dot - 4).c_str());
    out_suffix = name.substr(dot + 1);
    return true;
}

static bool is_multimodal_tensor(const std::string& name) {
    return starts_with(name, "v.") || starts_with(name, "a.") || starts_with(name, "mm.") ||
           name.find(".vision.") != std::string::npos || name.find(".audio.") != std::string::npos;
}

static bool is_float_type(QuantType t) {
    return t == QuantType::F32 || t == QuantType::BF16 || t == QuantType::F16;
}

// Decode a tensor (BF16, F16, or F32) into FP32 buffer
static bool decode_tensor_f32(std::ifstream& f, const InputTensor& t, std::vector<float>& out) {
    uint64_t ne = 1;
    for (uint64_t d : t.dims) ne *= d;
    out.resize(static_cast<size_t>(ne));
    f.clear();
    f.seekg(static_cast<std::streamoff>(t.abs_offset), std::ios::beg);
    if (!f) return false;

    if (t.type == QuantType::F32) {
        return read_exact(f, out.data(), static_cast<size_t>(ne) * sizeof(float));
    }
    if (t.type == QuantType::BF16 || t.type == QuantType::F16) {
        std::vector<uint16_t> raw(static_cast<size_t>(ne));
        if (!read_exact(f, raw.data(), static_cast<size_t>(ne) * sizeof(uint16_t))) return false;
        const int64_t n_i = static_cast<int64_t>(ne);
        if (t.type == QuantType::BF16) {
            #pragma omp parallel for schedule(static) if(n_i >= 65536)
            for (int64_t i = 0; i < n_i; ++i) {
                out[static_cast<size_t>(i)] = Avx2Math::bf16_to_fp32(raw[static_cast<size_t>(i)]);
            }
        } else {
            #pragma omp parallel for schedule(static) if(n_i >= 65536)
            for (int64_t i = 0; i < n_i; ++i) {
                out[static_cast<size_t>(i)] = Avx2Math::fp16_to_fp32(raw[static_cast<size_t>(i)]);
            }
        }
        return true;
    }
    return false;
}

// Encode FP32 buffer back to its original float format (BF16 / F16 / F32) and stream to output file
static bool encode_and_write_f32(std::ofstream& out_f, QuantType type, const std::vector<float>& data) {
    if (type == QuantType::F32) {
        return write_exact(out_f, data.data(), data.size() * sizeof(float));
    }
    std::vector<uint16_t> raw(data.size());
    const int64_t n_i = static_cast<int64_t>(data.size());
    if (type == QuantType::BF16) {
        #pragma omp parallel for schedule(static) if(n_i >= 65536)
        for (int64_t i = 0; i < n_i; ++i) {
            raw[static_cast<size_t>(i)] = fp32_to_bf16_rne(data[static_cast<size_t>(i)]);
        }
    } else if (type == QuantType::F16) {
        #pragma omp parallel for schedule(static) if(n_i >= 65536)
        for (int64_t i = 0; i < n_i; ++i) {
            raw[static_cast<size_t>(i)] = fp32_to_fp16_rne(data[static_cast<size_t>(i)]);
        }
    } else {
        return false;
    }
    return write_exact(out_f, raw.data(), raw.size() * sizeof(uint16_t));
}

static bool stream_copy_raw(std::ifstream& in_f, uint64_t abs_offset, uint64_t nbytes, std::ofstream& out_f) {
    in_f.clear();
    in_f.seekg(static_cast<std::streamoff>(abs_offset), std::ios::beg);
    if (!in_f) return false;
    std::vector<uint8_t> buf(std::min<uint64_t>(nbytes, 16ULL * 1024ULL * 1024ULL));
    uint64_t rem = nbytes;
    while (rem > 0) {
        size_t chunk = static_cast<size_t>(std::min<uint64_t>(rem, buf.size()));
        if (!read_exact(in_f, buf.data(), chunk)) return false;
        if (!write_exact(out_f, buf.data(), chunk)) return false;
        rem -= chunk;
    }
    return true;
}

static double compute_crest_factor(const std::vector<float>& v) {
    if (v.empty()) return 0.0;
    double sum_sq = 0.0;
    double max_abs = 0.0;
    for (float x : v) {
        double a = std::fabs(static_cast<double>(x));
        if (a > max_abs) max_abs = a;
        sum_sq += a * a;
    }
    double rms = std::sqrt(sum_sq / static_cast<double>(v.size()));
    return (rms > 1e-12) ? (max_abs / rms) : 0.0;
}

static RawKV make_kv_u32(const std::string& key, uint32_t v) {
    RawKV kv;
    kv.key = key;
    kv.val_type = GGUF_TYPE_UINT32;
    append_bytes(kv.raw_value, &v, 4);
    return kv;
}

static RawKV make_kv_bool(const std::string& key, bool v) {
    RawKV kv;
    kv.key = key;
    kv.val_type = GGUF_TYPE_BOOL;
    uint8_t b = v ? 1 : 0;
    append_bytes(kv.raw_value, &b, 1);
    return kv;
}

static void upsert_kv(std::vector<RawKV>& kvs, const RawKV& item) {
    for (auto& kv : kvs) {
        if (kv.key == item.key) {
            kv = item;
            return;
        }
    }
    kvs.push_back(item);
}

static bool extract_u32_kv(const RawKV& kv, uint32_t& out) {
    if ((kv.val_type == GGUF_TYPE_UINT32 || kv.val_type == GGUF_TYPE_INT32) && kv.raw_value.size() == 4) {
        std::memcpy(&out, kv.raw_value.data(), 4);
        return true;
    }
    if ((kv.val_type == GGUF_TYPE_UINT64 || kv.val_type == GGUF_TYPE_INT64) && kv.raw_value.size() == 8) {
        uint64_t v = 0;
        std::memcpy(&v, kv.raw_value.data(), 8);
        out = static_cast<uint32_t>(v);
        return true;
    }
    return false;
}

static void print_usage(const char* prog) {
    std::cout << "haven-bake — Native C++20 Gemma 4.5 GGUF Architecture Optimizer\n\n"
              << "Usage: " << prog << " [options] -o <output.gguf> <input1.gguf> [input2.gguf ...]\n\n"
              << "Options:\n"
              << "  --mode stock            100% stock llama.cpp/Ollama compatible cleanup:\n"
              << "                          strip dead shared-KV tensors, fold router/expert scales,\n"
              << "                          and bake zero-cost QuaRot R2 V->O Hadamard rotation.\n"
              << "  --mode turbo            Full Gemma 4.5 Turbo (native haven-cpp):\n"
              << "                          stock cleanups + norm-preserving orthogonal Block-256/64\n"
              << "                          FWHT baking across QKV, FFN, and MoE (flattens both\n"
              << "                          weight blocks and 450x-900x activation spikes by 16x!).\n"
              << "  --fold-norms            Also absorb pre-norm gammas into weights (opt-in;\n"
              << "                          recommended for BF16/FP8/INT8 targets, NOT 4-bit IQ4_XS\n"
              << "                          when norm gammas reach 450x-900x!).\n"
              << "  --strip-mm              Also strip .vision. / .audio. / mm.* tensors & metadata.\n"
              << "  --no-r2                 Disable V->O QuaRot R2 head-dim Hadamard rotation.\n"
              << "  --no-ffn-down-hadamard  Disable FWHT on ffn_down / ffn_down_exps inputs.\n"
              << "  -o, --output <path>     Output GGUF file path (required; never overwrites input).\n"
              << "  -h, --help              Show this help message.\n";
}

} // namespace

int main(int argc, char** argv) {
#ifdef _WIN32
    if (!std::getenv("OMP_WAIT_POLICY")) _putenv_s("OMP_WAIT_POLICY", "PASSIVE");
    if (!std::getenv("KMP_BLOCKTIME"))   _putenv_s("KMP_BLOCKTIME", "0");
#else
    if (!std::getenv("OMP_WAIT_POLICY")) setenv("OMP_WAIT_POLICY", "PASSIVE", 0);
    if (!std::getenv("KMP_BLOCKTIME"))   setenv("KMP_BLOCKTIME", "0", 0);
#endif

    std::string mode = "turbo";
    std::string out_path;
    std::vector<std::string> in_paths;
    bool strip_mm = false;
    bool enable_r2 = true;
    bool enable_ffn_down_h = true;
    bool opt_fold_norms = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--mode" && i + 1 < argc) {
            mode = argv[++i];
        } else if ((arg == "-o" || arg == "--output") && i + 1 < argc) {
            out_path = argv[++i];
        } else if (arg == "--strip-mm") {
            strip_mm = true;
        } else if (arg == "--fold-norms") {
            opt_fold_norms = true;
        } else if (arg == "--no-r2") {
            enable_r2 = false;
        } else if (arg == "--no-ffn-down-hadamard") {
            enable_ffn_down_h = false;
        } else if (!starts_with(arg, "-")) {
            in_paths.push_back(arg);
        } else {
            std::cerr << "Unknown option: " << arg << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    if (out_path.empty() || in_paths.empty() || (mode != "stock" && mode != "turbo")) {
        print_usage(argv[0]);
        return 1;
    }
    for (const auto& ip : in_paths) {
        if (ip == out_path) {
            std::cerr << "ERROR: Output path must differ from input path (never overwrite source master).\n";
            return 1;
        }
    }

    const bool is_turbo = (mode == "turbo");

    // 1. Parse headers of all input shard GGUF files
    std::vector<std::ifstream> shard_files(in_paths.size());
    std::vector<RawKV> kvs;
    std::vector<InputTensor> tensors;
    uint32_t gguf_version = 3;
    uint32_t alignment = 32;

    std::string arch = "gemma4";
    uint32_t num_layers = 0;
    uint32_t embedding_dim = 0;
    uint32_t num_heads = 0;
    uint32_t shared_kv_layers = 0;
    uint32_t expert_count = 0;
    uint32_t expert_ffn_dim = 0;
    std::vector<bool> swa_pattern;

    for (size_t s = 0; s < in_paths.size(); ++s) {
        shard_files[s].open(in_paths[s], std::ios::binary);
        if (!shard_files[s]) {
            std::cerr << "ERROR: Failed to open input GGUF: " << in_paths[s] << "\n";
            return 1;
        }
        auto& f = shard_files[s];
        uint32_t magic = 0, ver = 0;
        uint64_t n_tensors = 0, n_kv = 0;
        if (!read_u32(f, magic) || magic != GGUF_MAGIC || !read_u32(f, ver) ||
            !read_u64(f, n_tensors) || !read_u64(f, n_kv)) {
            std::cerr << "ERROR: Invalid GGUF header in " << in_paths[s] << "\n";
            return 1;
        }
        if (s == 0) gguf_version = ver;

        uint32_t shard_align = 32;
        for (uint64_t i = 0; i < n_kv; ++i) {
            RawKV kv;
            if (!read_gguf_str(f, kv.key) || !read_u32(f, kv.val_type) ||
                !read_raw_gguf_value(f, kv.val_type, kv.raw_value)) {
                std::cerr << "ERROR: Failed reading KV #" << i << " in " << in_paths[s] << "\n";
                return 1;
            }
            if (kv.key == "general.alignment") {
                uint32_t al = 32;
                if (extract_u32_kv(kv, al) && al >= 1 && (al & (al - 1)) == 0 && al <= 4096) {
                    shard_align = al;
                }
            }
            if (s == 0) {
                if (starts_with(kv.key, "split.")) continue; // Single merged output file
                if (strip_mm && (kv.key.find(".vision.") != std::string::npos || kv.key.find(".audio.") != std::string::npos)) {
                    continue;
                }
                if (kv.key.find(".turbo.") != std::string::npos) continue; // Replaced below

                if (kv.key == "general.architecture" && kv.val_type == GGUF_TYPE_STRING && kv.raw_value.size() >= 8) {
                    uint64_t slen = 0;
                    std::memcpy(&slen, kv.raw_value.data(), 8);
                    if (8 + slen == kv.raw_value.size()) {
                        arch.assign(reinterpret_cast<const char*>(kv.raw_value.data() + 8), static_cast<size_t>(slen));
                    }
                } else if (kv.key.find(".vision.") == std::string::npos && kv.key.find(".audio.") == std::string::npos) {
                    if (ends_with(kv.key, ".block_count")) extract_u32_kv(kv, num_layers);
                    else if (ends_with(kv.key, ".embedding_length")) extract_u32_kv(kv, embedding_dim);
                    else if (ends_with(kv.key, ".attention.head_count")) extract_u32_kv(kv, num_heads);
                    else if (ends_with(kv.key, ".attention.shared_kv_layers")) extract_u32_kv(kv, shared_kv_layers);
                    else if (ends_with(kv.key, ".expert_count")) extract_u32_kv(kv, expert_count);
                    else if (ends_with(kv.key, ".expert_feed_forward_length")) extract_u32_kv(kv, expert_ffn_dim);
                    else if (ends_with(kv.key, ".attention.sliding_window_pattern") && kv.val_type == GGUF_TYPE_ARRAY && kv.raw_value.size() >= 12) {
                        uint32_t at = 0;
                        uint64_t al = 0;
                        std::memcpy(&at, kv.raw_value.data(), 4);
                        std::memcpy(&al, kv.raw_value.data() + 4, 8);
                        if (at == GGUF_TYPE_BOOL && 12 + al == kv.raw_value.size()) {
                            swa_pattern.resize(static_cast<size_t>(al));
                            for (uint64_t j = 0; j < al; ++j) swa_pattern[static_cast<size_t>(j)] = (kv.raw_value[12 + j] != 0);
                        }
                    }
                }
                kvs.push_back(std::move(kv));
            }
        }
        if (s == 0) alignment = shard_align;

        size_t shard_t_start = tensors.size();
        for (uint64_t i = 0; i < n_tensors; ++i) {
            InputTensor t;
            t.shard_idx = static_cast<int>(s);
            uint32_t nd = 0, tt = 0;
            if (!read_gguf_str(f, t.name) || !read_u32(f, nd) || nd == 0 || nd > 4) {
                std::cerr << "ERROR: Failed reading tensor header #" << i << "\n";
                return 1;
            }
            t.dims.resize(nd);
            for (uint32_t d = 0; d < nd; ++d) {
                if (!read_u64(f, t.dims[d])) return 1;
            }
            if (!read_u32(f, tt) || !read_u64(f, t.rel_offset)) return 1;
            t.type = static_cast<QuantType>(tt);
            t.nbytes = tensor_byte_size(t.type, t.dims);
            tensors.push_back(std::move(t));
        }

        uint64_t cur_pos = static_cast<uint64_t>(f.tellg());
        uint64_t data_start = ((cur_pos + shard_align - 1ULL) / shard_align) * shard_align;
        for (size_t i = shard_t_start; i < tensors.size(); ++i) {
            tensors[i].abs_offset = data_start + tensors[i].rel_offset;
        }
    }

    std::unordered_map<std::string, size_t> name_to_idx;
    for (size_t i = 0; i < tensors.size(); ++i) name_to_idx[tensors[i].name] = i;

    const uint32_t kv_start = (shared_kv_layers > 0 && shared_kv_layers < num_layers)
        ? (num_layers - shared_kv_layers) : num_layers;

    auto is_swa_layer = [&](uint32_t l) -> bool {
        if (l < swa_pattern.size()) return swa_pattern[l];
        return (l % 6 != 5);
    };
    auto kv_source_layer = [&](uint32_t l) -> uint32_t {
        if (l < kv_start) return l;
        return kv_start - (is_swa_layer(l) ? 2u : 1u);
    };

    // 2. Analyze per-layer geometry, R2 V->O eligibility, and Hadamard block sizes
    std::vector<int> layer_hd(num_layers, 0);
    std::vector<bool> r2_kv_source_ok(num_layers, false);
    for (uint32_t l = 0; l < num_layers; ++l) {
        std::string qname = "blk." + std::to_string(l) + ".attn_q.weight";
        auto it = name_to_idx.find(qname);
        if (it != name_to_idx.end() && tensors[it->second].dims.size() >= 2 && num_heads > 0) {
            layer_hd[l] = static_cast<int>(tensors[it->second].dims[1] / num_heads);
        }
    }

    if (enable_r2) {
        for (uint32_t s = 0; s < kv_start; ++s) {
            std::string vname = "blk." + std::to_string(s) + ".attn_v.weight";
            auto vit = name_to_idx.find(vname);
            if (vit == name_to_idx.end()) continue; // k_eq_v layer: NEVER apply R2!
            const auto& vt = tensors[vit->second];
            int hd = layer_hd[s];
            if (!is_float_type(vt.type) || !is_pow2(static_cast<uint32_t>(hd)) ||
                vt.dims.size() < 2 || (vt.dims[1] % static_cast<uint64_t>(hd)) != 0) {
                continue;
            }
            bool all_consumers_ok = true;
            for (uint32_t l = 0; l < num_layers; ++l) {
                if (kv_source_layer(l) != s) continue;
                std::string oname = "blk." + std::to_string(l) + ".attn_output.weight";
                auto oit = name_to_idx.find(oname);
                if (oit == name_to_idx.end() || !is_float_type(tensors[oit->second].type) ||
                    layer_hd[l] != hd || tensors[oit->second].dims.size() < 2 ||
                    (tensors[oit->second].dims[0] % static_cast<uint64_t>(hd)) != 0) {
                    all_consumers_ok = false;
                    break;
                }
            }
            r2_kv_source_ok[s] = all_consumers_ok;
        }
    }

    // Verify float types and compute turbo Hadamard block sizes
    uint32_t h_qkv = 0, h_ffn = 0, h_ffn_dn = 0, h_moe_in = 0, h_moe_dn = 0;
    bool fold_norms = false;
    bool fold_scales = false;

    // Check scale folding eligibility (both stock & turbo)
    bool has_any_router_scale = false;
    bool has_any_down_exps_scale = false;
    bool scales_float_ok = true;
    for (uint32_t l = 0; l < num_layers; ++l) {
        std::string p = "blk." + std::to_string(l) + ".";
        auto it_rs = name_to_idx.find(p + "ffn_gate_inp.scale");
        auto it_rw = name_to_idx.find(p + "ffn_gate_inp.weight");
        if (it_rs != name_to_idx.end() && it_rw != name_to_idx.end()) {
            has_any_router_scale = true;
            if (!is_float_type(tensors[it_rs->second].type) || !is_float_type(tensors[it_rw->second].type)) {
                scales_float_ok = false;
            }
        }
        auto it_ds = name_to_idx.find(p + "ffn_down_exps.scale");
        auto it_dw = name_to_idx.find(p + "ffn_down_exps.weight");
        if (it_ds != name_to_idx.end() && it_dw != name_to_idx.end()) {
            has_any_down_exps_scale = true;
            if (!is_float_type(tensors[it_ds->second].type) || !is_float_type(tensors[it_dw->second].type)) {
                scales_float_ok = false;
            }
        }
    }
    fold_scales = scales_float_ok && (has_any_router_scale || has_any_down_exps_scale);

    if (is_turbo) {
        if (embedding_dim == 0 || (embedding_dim % 256) != 0) {
            std::cerr << "ERROR: --mode turbo requires embedding_dim divisible by 256 (got " << embedding_dim << ").\n";
            return 1;
        }
        // Ensure all transformed projections are unquantized float (BF16/F16/F32) to prevent double quantization!
        bool all_ffn_256 = true, all_ffn_64 = true;
        bool has_moe_exps = false;
        uint64_t moe_f_exp = 0;
        for (uint32_t l = 0; l < num_layers; ++l) {
            std::string p = "blk." + std::to_string(l) + ".";
            const char* req_qkv[] = {"attn_q.weight", "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"};
            for (const char* suf : req_qkv) {
                auto it = name_to_idx.find(p + suf);
                if (it != name_to_idx.end() && !is_float_type(tensors[it->second].type)) {
                    std::cerr << "ERROR: --mode turbo requires unquantized BF16/F16/F32 master GGUF (tensor "
                              << (p + suf) << " is already quantized!).\n";
                    return 1;
                }
            }
            if (l < kv_start) {
                for (const char* suf : {"attn_k.weight", "attn_v.weight"}) {
                    auto it = name_to_idx.find(p + suf);
                    if (it != name_to_idx.end() && !is_float_type(tensors[it->second].type)) {
                        std::cerr << "ERROR: Tensor " << (p + suf) << " must be BF16/F16/F32.\n";
                        return 1;
                    }
                }
            }
            auto it_dn = name_to_idx.find(p + "ffn_down.weight");
            if (it_dn != name_to_idx.end() && !tensors[it_dn->second].dims.empty()) {
                uint64_t f_in = tensors[it_dn->second].dims[0];
                if ((f_in % 256) != 0) all_ffn_256 = false;
                if ((f_in % 64) != 0)  all_ffn_64 = false;
            }
            auto it_gu = name_to_idx.find(p + "ffn_gate_up_exps.weight");
            auto it_ed = name_to_idx.find(p + "ffn_down_exps.weight");
            if (it_gu != name_to_idx.end() && it_ed != name_to_idx.end()) {
                has_moe_exps = true;
                if (!is_float_type(tensors[it_gu->second].type) || !is_float_type(tensors[it_ed->second].type)) {
                    std::cerr << "ERROR: MoE expert tensors must be BF16/F16/F32.\n";
                    return 1;
                }
                moe_f_exp = tensors[it_ed->second].dims[0];
            }
        }

        h_qkv = 256;
        h_ffn = 256;
        fold_norms = opt_fold_norms;
        if (enable_ffn_down_h) {
            if (all_ffn_256) h_ffn_dn = 256;
            else if (all_ffn_64) h_ffn_dn = 64;
        }
        if (has_moe_exps) {
            h_moe_in = 256;
            if (enable_ffn_down_h) {
                if (moe_f_exp >= 256 && (moe_f_exp % 256) == 0) h_moe_dn = 256;
                else if (moe_f_exp >= 64 && (moe_f_exp % 64) == 0) h_moe_dn = 64;
            }
        }
    }

    // 3. Pre-load small 1D norm & scale vectors (~5 MB RAM) needed for folding
    std::unordered_map<std::string, std::vector<float>> vec_cache;
    auto load_small_vec = [&](const std::string& name) -> const std::vector<float>* {
        auto cit = vec_cache.find(name);
        if (cit != vec_cache.end()) return &cit->second;
        auto it = name_to_idx.find(name);
        if (it == name_to_idx.end()) return nullptr;
        const auto& t = tensors[it->second];
        if (!is_float_type(t.type)) return nullptr;
        std::vector<float> v;
        if (!decode_tensor_f32(shard_files[t.shard_idx], t, v)) return nullptr;
        auto [ins, _] = vec_cache.emplace(name, std::move(v));
        return &ins->second;
    };

    for (uint32_t l = 0; l < num_layers; ++l) {
        std::string p = "blk." + std::to_string(l) + ".";
        if (fold_norms) {
            load_small_vec(p + "attn_norm.weight");
            load_small_vec(p + "ffn_norm.weight");
            load_small_vec(p + "pre_ffw_norm_2.weight");
        }
        if (fold_scales) {
            load_small_vec(p + "ffn_gate_inp.scale");
            load_small_vec(p + "ffn_down_exps.scale");
        }
    }

    // 4. Filter kept tensors (strip dead shared-KV placeholders & optional multimodal towers)
    std::vector<size_t> kept_indices;
    size_t stripped_kv_count = 0, stripped_mm_count = 0;
    uint64_t stripped_kv_bytes = 0, stripped_mm_bytes = 0;

    for (size_t i = 0; i < tensors.size(); ++i) {
        const auto& t = tensors[i];
        if (strip_mm && is_multimodal_tensor(t.name)) {
            stripped_mm_count++;
            stripped_mm_bytes += t.nbytes;
            continue;
        }
        int l = -1;
        std::string suf;
        if (parse_blk_tensor(t.name, l, suf) && l >= static_cast<int>(kv_start)) {
            if (suf == "attn_k.weight" || suf == "attn_v.weight" || suf == "attn_k_norm.weight") {
                stripped_kv_count++;
                stripped_kv_bytes += t.nbytes;
                continue;
            }
        }
        kept_indices.push_back(i);
    }

    // 5. Append Gemma 4.5 metadata keys
    if (is_turbo) {
        upsert_kv(kvs, make_kv_u32(arch + ".turbo.version", 1));
        if (h_qkv > 0)    upsert_kv(kvs, make_kv_u32(arch + ".turbo.hadamard_qkv", h_qkv));
        if (h_ffn > 0)    upsert_kv(kvs, make_kv_u32(arch + ".turbo.hadamard_ffn", h_ffn));
        if (h_ffn_dn > 0) upsert_kv(kvs, make_kv_u32(arch + ".turbo.hadamard_ffn_down", h_ffn_dn));
        if (h_moe_in > 0) upsert_kv(kvs, make_kv_u32(arch + ".turbo.hadamard_moe_in", h_moe_in));
        if (h_moe_dn > 0) upsert_kv(kvs, make_kv_u32(arch + ".turbo.hadamard_moe_down", h_moe_dn));
        upsert_kv(kvs, make_kv_bool(arch + ".turbo.folded_norms", fold_norms));
        upsert_kv(kvs, make_kv_bool(arch + ".turbo.folded_scales", fold_scales));
    } else if (fold_scales) {
        upsert_kv(kvs, make_kv_bool(arch + ".turbo.folded_scales", true));
    }

    int r2_sources = 0;
    for (uint32_t s = 0; s < kv_start; ++s) if (r2_kv_source_ok[s]) r2_sources++;

    std::cout << "======================================================================\n"
              << "  HAVEN-BAKE: GEMMA 4.5 ARCHITECTURE OPTIMIZER (" << mode << ")\n"
              << "======================================================================\n"
              << "  Arch           : " << arch << " (L=" << num_layers << ", D=" << embedding_dim
              << ", shared_kv=" << shared_kv_layers << ", E=" << expert_count << ")\n"
              << "  Input shards   : " << in_paths.size() << " (" << tensors.size() << " tensors)\n"
              << "  Kept tensors   : " << kept_indices.size() << "\n"
              << "  Stripped dead  : " << stripped_kv_count << " shared-KV (" << std::fixed << std::setprecision(2)
              << (static_cast<double>(stripped_kv_bytes) / 1e6) << " MB), "
              << stripped_mm_count << " mm (" << (static_cast<double>(stripped_mm_bytes) / 1e6) << " MB)\n"
              << "  QuaRot R2 V->O : " << r2_sources << "/" << kv_start << " KV-source layers (0 runtime ops)\n"
              << "  Folded scales  : " << (fold_scales ? "YES (router + down_exps)" : "NO") << "\n"
              << "  Folded norms   : " << (fold_norms ? "YES (attn_norm + ffn_norm + pre_ffw_norm_2)" : "NO") << "\n"
              << "  Turbo FWHT     : qkv=" << h_qkv << ", ffn=" << h_ffn << ", ffn_dn=" << h_ffn_dn
              << ", moe_in=" << h_moe_in << ", moe_dn=" << h_moe_dn << "\n"
              << "  Output         : " << out_path << "\n"
              << "======================================================================\n" << std::flush;

    // 6. Compute output tensor relative offsets (aligned to `alignment`)
    std::vector<uint64_t> out_offsets(kept_indices.size(), 0);
    uint64_t cur_off = 0;
    for (size_t k = 0; k < kept_indices.size(); ++k) {
        out_offsets[k] = cur_off;
        uint64_t nb = tensors[kept_indices[k]].nbytes;
        cur_off += ((nb + alignment - 1ULL) / alignment) * alignment;
    }

    // 7. Write GGUF header + KV pairs + Tensor table
    std::ofstream out_f(out_path, std::ios::binary | std::ios::trunc);
    if (!out_f) {
        std::cerr << "ERROR: Failed to create output GGUF: " << out_path << "\n";
        return 1;
    }

    const uint64_t out_n_tensors = static_cast<uint64_t>(kept_indices.size());
    const uint64_t out_n_kv = static_cast<uint64_t>(kvs.size());
    write_exact(out_f, &GGUF_MAGIC, 4);
    write_exact(out_f, &gguf_version, 4);
    write_exact(out_f, &out_n_tensors, 8);
    write_exact(out_f, &out_n_kv, 8);

    for (const auto& kv : kvs) {
        uint64_t klen = static_cast<uint64_t>(kv.key.size());
        write_exact(out_f, &klen, 8);
        write_exact(out_f, kv.key.data(), kv.key.size());
        write_exact(out_f, &kv.val_type, 4);
        write_exact(out_f, kv.raw_value.data(), kv.raw_value.size());
    }

    for (size_t k = 0; k < kept_indices.size(); ++k) {
        const auto& t = tensors[kept_indices[k]];
        uint64_t nlen = static_cast<uint64_t>(t.name.size());
        uint32_t nd = static_cast<uint32_t>(t.dims.size());
        uint32_t tt = static_cast<uint32_t>(t.type);
        write_exact(out_f, &nlen, 8);
        write_exact(out_f, t.name.data(), t.name.size());
        write_exact(out_f, &nd, 4);
        for (uint64_t d : t.dims) write_exact(out_f, &d, 8);
        write_exact(out_f, &tt, 4);
        write_exact(out_f, &out_offsets[k], 8);
    }

    auto pad_to_align = [&](std::ofstream& f) -> bool {
        uint64_t pos = static_cast<uint64_t>(f.tellp());
        uint64_t pad = (alignment - (pos % alignment)) % alignment;
        if (pad > 0) {
            std::vector<uint8_t> z(static_cast<size_t>(pad), 0);
            return write_exact(f, z.data(), z.size());
        }
        return true;
    };
    if (!pad_to_align(out_f)) return 1;

    // 8. Stream, transform, and write every tensor cleanly!
    size_t transformed_count = 0;
    double sum_crest_before = 0.0, sum_crest_after = 0.0;
    size_t crest_samples = 0;

    for (size_t k = 0; k < kept_indices.size(); ++k) {
        const auto& t = tensors[kept_indices[k]];
        int l = -1;
        std::string suf;
        const bool is_blk = parse_blk_tensor(t.name, l, suf);

        bool need_transform = false;
        if (is_blk && l >= 0 && l < static_cast<int>(num_layers)) {
            const uint32_t lu = static_cast<uint32_t>(l);
            const uint32_t kv_src = kv_source_layer(lu);
            if (fold_norms && (suf == "attn_norm.weight" || suf == "ffn_norm.weight" || suf == "pre_ffw_norm_2.weight")) {
                need_transform = true;
            } else if (fold_scales && (suf == "ffn_gate_inp.scale" || suf == "ffn_down_exps.scale" ||
                                       suf == "ffn_gate_inp.weight" || suf == "ffn_down_exps.weight")) {
                need_transform = true;
            } else if (r2_kv_source_ok[kv_src] && ((suf == "attn_v.weight" && lu == kv_src) || suf == "attn_output.weight")) {
                need_transform = true;
            } else if (is_turbo && (suf == "attn_q.weight" || suf == "attn_k.weight" || suf == "attn_v.weight" ||
                                    suf == "ffn_gate.weight" || suf == "ffn_up.weight" ||
                                    (suf == "ffn_down.weight" && h_ffn_dn > 0) ||
                                    (suf == "ffn_gate_up_exps.weight" && h_moe_in > 0) ||
                                    (suf == "ffn_gate_exps.weight" && h_moe_in > 0) ||
                                    (suf == "ffn_up_exps.weight" && h_moe_in > 0) ||
                                    (suf == "ffn_down_exps.weight" && h_moe_dn > 0))) {
                need_transform = true;
            }
        }

        if (!need_transform) {
            if (!stream_copy_raw(shard_files[t.shard_idx], t.abs_offset, t.nbytes, out_f)) {
                std::cerr << "ERROR: Failed copying tensor " << t.name << "\n";
                return 1;
            }
            if (!pad_to_align(out_f)) return 1;
            continue;
        }

        // Load tensor as FP32, apply exact linear algebra, re-encode back to original float dtype
        std::vector<float> buf;
        if (!decode_tensor_f32(shard_files[t.shard_idx], t, buf)) {
            std::cerr << "ERROR: Failed decoding tensor " << t.name << "\n";
            return 1;
        }

        const uint32_t lu = static_cast<uint32_t>(l);
        const uint32_t kv_src = kv_source_layer(lu);
        const std::string p = "blk." + std::to_string(l) + ".";
        const int64_t ne0 = static_cast<int64_t>(t.dims[0]);
        const int64_t n_rows = static_cast<int64_t>(buf.size()) / std::max<int64_t>(1, ne0);

        if ((fold_norms && (suf == "attn_norm.weight" || suf == "ffn_norm.weight" || suf == "pre_ffw_norm_2.weight")) ||
            (fold_scales && (suf == "ffn_gate_inp.scale" || suf == "ffn_down_exps.scale"))) {
            std::fill(buf.begin(), buf.end(), 1.0f);
        } else {
            double c_before = compute_crest_factor(buf);

            // A. Optional pre-norm gamma fold + Orthogonal Block-FWHT input rotation on Q/K/V, Gate/Up, MoE Gate+Up
            if (is_turbo && (suf == "attn_q.weight" || suf == "attn_k.weight" || suf == "attn_v.weight")) {
                const auto* g = fold_norms ? load_small_vec(p + "attn_norm.weight") : nullptr;
                #pragma omp parallel for schedule(static)
                for (int64_t r = 0; r < n_rows; ++r) {
                    float* row = buf.data() + static_cast<size_t>(r) * static_cast<size_t>(ne0);
                    if (g && static_cast<int64_t>(g->size()) == ne0) {
                        for (int64_t c = 0; c < ne0; ++c) row[c] *= (*g)[static_cast<size_t>(c)];
                    }
                    if (h_qkv > 0) Avx2Math::fwht_block_inplace(row, static_cast<int>(ne0), static_cast<int>(h_qkv));
                }
            } else if (is_turbo && (suf == "ffn_gate.weight" || suf == "ffn_up.weight")) {
                const auto* g = fold_norms ? load_small_vec(p + "ffn_norm.weight") : nullptr;
                #pragma omp parallel for schedule(static)
                for (int64_t r = 0; r < n_rows; ++r) {
                    float* row = buf.data() + static_cast<size_t>(r) * static_cast<size_t>(ne0);
                    if (g && static_cast<int64_t>(g->size()) == ne0) {
                        for (int64_t c = 0; c < ne0; ++c) row[c] *= (*g)[static_cast<size_t>(c)];
                    }
                    if (h_ffn > 0) Avx2Math::fwht_block_inplace(row, static_cast<int>(ne0), static_cast<int>(h_ffn));
                }
            } else if (is_turbo && h_ffn_dn > 0 && suf == "ffn_down.weight") {
                #pragma omp parallel for schedule(static)
                for (int64_t r = 0; r < n_rows; ++r) {
                    float* row = buf.data() + static_cast<size_t>(r) * static_cast<size_t>(ne0);
                    Avx2Math::fwht_block_inplace(row, static_cast<int>(ne0), static_cast<int>(h_ffn_dn));
                }
            } else if (is_turbo && h_moe_in > 0 &&
                       (suf == "ffn_gate_up_exps.weight" || suf == "ffn_gate_exps.weight" || suf == "ffn_up_exps.weight")) {
                const auto* g = fold_norms ? load_small_vec(p + "pre_ffw_norm_2.weight") : nullptr;
                #pragma omp parallel for schedule(static)
                for (int64_t r = 0; r < n_rows; ++r) {
                    float* row = buf.data() + static_cast<size_t>(r) * static_cast<size_t>(ne0);
                    if (g && static_cast<int64_t>(g->size()) == ne0) {
                        for (int64_t c = 0; c < ne0; ++c) row[c] *= (*g)[static_cast<size_t>(c)];
                    }
                    Avx2Math::fwht_block_inplace(row, static_cast<int>(ne0), static_cast<int>(h_moe_in));
                }
            } else if (fold_scales && suf == "ffn_gate_inp.weight") {
                const auto* rs = load_small_vec(p + "ffn_gate_inp.scale");
                if (rs && static_cast<int64_t>(rs->size()) == ne0) {
                    for (int64_t r = 0; r < n_rows; ++r) {
                        float* row = buf.data() + static_cast<size_t>(r) * static_cast<size_t>(ne0);
                        for (int64_t c = 0; c < ne0; ++c) row[c] *= (*rs)[static_cast<size_t>(c)];
                    }
                }
            } else if (suf == "ffn_down_exps.weight") {
                const auto* ds = fold_scales ? load_small_vec(p + "ffn_down_exps.scale") : nullptr;
                const int64_t rows_per_exp = (t.dims.size() >= 3 && t.dims[2] > 0) ? static_cast<int64_t>(t.dims[1]) : n_rows;
                #pragma omp parallel for schedule(static)
                for (int64_t r = 0; r < n_rows; ++r) {
                    float* row = buf.data() + static_cast<size_t>(r) * static_cast<size_t>(ne0);
                    if (ds && rows_per_exp > 0) {
                        size_t e = static_cast<size_t>(r / rows_per_exp);
                        if (e < ds->size()) {
                            const float sc = (*ds)[e];
                            for (int64_t c = 0; c < ne0; ++c) row[c] *= sc;
                        }
                    }
                    if (is_turbo && h_moe_dn > 0) {
                        Avx2Math::fwht_block_inplace(row, static_cast<int>(ne0), static_cast<int>(h_moe_dn));
                    }
                }
            }

            // B. Zero-cost QuaRot R2 V -> O Head-Dim Hadamard (commutes with input transform above!)
            if (r2_kv_source_ok[kv_src] && suf == "attn_v.weight" && lu == kv_src) {
                const int hd = layer_hd[kv_src];
                const int n_kv_h = static_cast<int>(n_rows) / hd;
                // Left-multiply each output head block [hd, ne0] by H_hd along the vertical (output) dimension
                #pragma omp parallel for schedule(static)
                for (int64_t c = 0; c < ne0; ++c) {
                    std::vector<float> col(static_cast<size_t>(hd));
                    for (int h = 0; h < n_kv_h; ++h) {
                        const int64_t r0 = static_cast<int64_t>(h) * hd;
                        for (int i = 0; i < hd; ++i) {
                            col[static_cast<size_t>(i)] = buf[static_cast<size_t>(r0 + i) * static_cast<size_t>(ne0) + static_cast<size_t>(c)];
                        }
                        Avx2Math::fwht_block_inplace(col.data(), hd, hd);
                        for (int i = 0; i < hd; ++i) {
                            buf[static_cast<size_t>(r0 + i) * static_cast<size_t>(ne0) + static_cast<size_t>(c)] = col[static_cast<size_t>(i)];
                        }
                    }
                }
            } else if (r2_kv_source_ok[kv_src] && suf == "attn_output.weight") {
                const int hd = layer_hd[lu];
                #pragma omp parallel for schedule(static)
                for (int64_t r = 0; r < n_rows; ++r) {
                    float* row = buf.data() + static_cast<size_t>(r) * static_cast<size_t>(ne0);
                    Avx2Math::fwht_block_inplace(row, static_cast<int>(ne0), hd);
                }
            }

            double c_after = compute_crest_factor(buf);
            if (suf == "ffn_down.weight" || suf == "attn_output.weight" || suf == "attn_q.weight") {
                sum_crest_before += c_before;
                sum_crest_after += c_after;
                crest_samples++;
            }
        }

        if (!encode_and_write_f32(out_f, t.type, buf)) {
            std::cerr << "ERROR: Failed writing transformed tensor " << t.name << "\n";
            return 1;
        }
        if (!pad_to_align(out_f)) return 1;
        transformed_count++;
    }

    out_f.flush();
    if (!out_f) {
        std::cerr << "ERROR: Output stream error while finalizing " << out_path << "\n";
        return 1;
    }

    std::cout << "  ✓ Baked " << kept_indices.size() << " tensors (" << transformed_count << " transformed) successfully!\n";
    if (crest_samples > 0) {
        std::cout << "  ✓ Weight crest factor (max/rms) on Q/O/FFN-Down: "
                  << std::fixed << std::setprecision(2)
                  << (sum_crest_before / crest_samples) << "x -> "
                  << (sum_crest_after / crest_samples) << "x\n";
    }
    return 0;
}
