#include "haven/haven_mlx.h"

#if defined(HAVEN_HAS_MLX)
#include "mlx/c/mlx.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <algorithm>

namespace haven {

struct HavenMLXEngine::Impl {
    mlx_device dev = { nullptr };
    mlx_stream stream = { nullptr };
    bool has_dev_stream = false;
    std::vector<mlx_map_string_to_array> tensor_maps;
    std::unordered_map<std::string, mlx_array> tensor_registry;
    std::vector<std::string> tensor_names;

    ~Impl() {
        cleanup();
    }

    void cleanup() {
        tensor_registry.clear();
        tensor_names.clear();

        for (auto& map : tensor_maps) {
            if (map.ctx) {
                mlx_map_string_to_array_free(map);
            }
        }
        tensor_maps.clear();

        if (has_dev_stream) {
            if (stream.ctx) mlx_stream_free(stream);
            if (dev.ctx) mlx_device_free(dev);
            stream = { nullptr };
            dev = { nullptr };
            has_dev_stream = false;
        }
    }
};

HavenMLXEngine::HavenMLXEngine() : impl_(std::make_unique<Impl>()) {}

HavenMLXEngine::~HavenMLXEngine() {
    unload();
}

static std::string format_bytes(uintmax_t bytes) {
    if (bytes == 0) return "0 B";
    char buf[32];
    if (bytes >= 1024ULL * 1024 * 1024) {
        snprintf(buf, sizeof(buf), "%.2f GB", (double)bytes / (1024.0 * 1024.0 * 1024.0));
    } else if (bytes >= 1024ULL * 1024) {
        snprintf(buf, sizeof(buf), "%.1f MB", (double)bytes / (1024.0 * 1024.0));
    } else {
        snprintf(buf, sizeof(buf), "%llu B", (unsigned long long)bytes);
    }
    return std::string(buf);
}

static std::string extract_json_val(const std::string& json, const std::string& key) {
    std::string search = "\"" + key + "\":";
    size_t pos = json.find(search);
    if (pos == std::string::npos) return "";
    size_t start = json.find_first_not_of(" \t\r\n", pos + search.length());
    if (start == std::string::npos) return "";

    if (json[start] == '"') {
        size_t end = json.find('"', start + 1);
        if (end == std::string::npos) return "";
        return json.substr(start + 1, end - start - 1);
    } else {
        size_t end = json.find_first_of(",}\r\n \t", start);
        if (end == std::string::npos) end = json.length();
        return json.substr(start, end - start);
    }
}

bool HavenMLXEngine::is_available() {
    try {
        mlx_array a = mlx_array_new_float(1.0f);
        mlx_array_free(a);
        return true;
    } catch (...) {
        return false;
    }
}

bool HavenMLXEngine::is_mlx_model_dir(const std::string& dir_path) {
    std::error_code ec;
    if (!std::filesystem::exists(dir_path, ec) || !std::filesystem::is_directory(dir_path, ec)) {
        return false;
    }

    std::filesystem::path cfg = std::filesystem::path(dir_path) / "config.json";
    if (!std::filesystem::exists(cfg, ec)) return false;

    // Must contain at least one .safetensors file
    for (const auto& entry : std::filesystem::directory_iterator(dir_path, ec)) {
        if (ec) break;
        if (entry.is_regular_file() && entry.path().extension() == ".safetensors") {
            return true;
        }
    }
    return false;
}

bool HavenMLXEngine::inspect_mlx_dir(const std::string& dir_path, MLXModelMetadata& meta) {
    std::error_code ec;
    if (!is_mlx_model_dir(dir_path)) return false;

    meta.model_name = std::filesystem::path(dir_path).filename().string();
    meta.total_bytes = 0;
    meta.total_tensors = 0;

    std::filesystem::path cfg_path = std::filesystem::path(dir_path) / "config.json";
    std::ifstream f(cfg_path, std::ios::binary);
    if (f) {
        std::string json((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        meta.model_type = extract_json_val(json, "model_type");
        if (meta.model_type.empty()) meta.model_type = "transformer";

        std::string hs = extract_json_val(json, "hidden_size");
        if (!hs.empty()) meta.hidden_size = std::stoi(hs);

        std::string inter = extract_json_val(json, "intermediate_size");
        if (!inter.empty()) meta.intermediate_size = std::stoi(inter);

        std::string nl = extract_json_val(json, "num_hidden_layers");
        if (!nl.empty()) meta.num_hidden_layers = std::stoi(nl);

        std::string ah = extract_json_val(json, "num_attention_heads");
        if (!ah.empty()) meta.num_attention_heads = std::stoi(ah);

        std::string kv = extract_json_val(json, "num_key_value_heads");
        if (!kv.empty()) meta.num_key_value_heads = std::stoi(kv);

        std::string vs = extract_json_val(json, "vocab_size");
        if (!vs.empty()) meta.vocab_size = std::stoi(vs);

        std::string eps = extract_json_val(json, "rms_norm_eps");
        if (!eps.empty()) meta.rms_norm_eps = std::stof(eps);

        std::string theta = extract_json_val(json, "rope_theta");
        if (!theta.empty()) meta.rope_theta = std::stof(theta);

        if (json.find("\"quantization\"") != std::string::npos || json.find("\"bits\":") != std::string::npos) {
            meta.is_quantized = true;
            std::string bits = extract_json_val(json, "bits");
            if (!bits.empty()) meta.quant_bits = std::stoi(bits);
            std::string grp = extract_json_val(json, "group_size");
            if (!grp.empty()) meta.quant_group_size = std::stoi(grp);
        }
    }

    // Accumulate size of all safetensors
    for (const auto& entry : std::filesystem::directory_iterator(dir_path, ec)) {
        if (entry.is_regular_file() && entry.path().extension() == ".safetensors") {
            meta.total_bytes += entry.file_size(ec);
        }
    }
    meta.size_formatted = format_bytes(meta.total_bytes);
    return true;
}

bool HavenMLXEngine::load_model(const std::string& dir_path, std::string& err_out) {
    unload();

    if (!inspect_mlx_dir(dir_path, metadata_)) {
        err_out = "Directory is not a valid MLX / Safetensors model: " + dir_path;
        return false;
    }

    model_dir_ = dir_path;
    impl_->dev = mlx_device_new_type(MLX_CPU, 0);
    impl_->stream = mlx_stream_new_device(impl_->dev);
    impl_->has_dev_stream = true;

    std::vector<std::string> safetensors_files;
    std::error_code ec;
    for (const auto& entry : std::filesystem::directory_iterator(dir_path, ec)) {
        if (entry.is_regular_file() && entry.path().extension() == ".safetensors") {
            safetensors_files.push_back(entry.path().string());
        }
    }
    std::sort(safetensors_files.begin(), safetensors_files.end());

    std::cout << "[HavenMLX] Ingesting MLX / Safetensors weights from " << dir_path << " (" 
              << safetensors_files.size() << " files, " << metadata_.size_formatted << ")\n" << std::flush;

    for (const auto& file : safetensors_files) {
        mlx_map_string_to_array tensors = mlx_map_string_to_array_new();
        mlx_map_string_to_string meta_map = mlx_map_string_to_string_new();

        int res = mlx_load_safetensors(&tensors, &meta_map, file.c_str(), impl_->stream);
        mlx_map_string_to_string_free(meta_map);

        if (res != 0) {
            err_out = "Failed to load safetensors file via MLX: " + file;
            mlx_map_string_to_array_free(tensors);
            unload();
            return false;
        }

        impl_->tensor_maps.push_back(tensors);

        mlx_map_string_to_array_iterator it = mlx_map_string_to_array_iterator_new(tensors);
        const char* key = nullptr;
        mlx_array val = mlx_array_new();

        while (mlx_map_string_to_array_iterator_next(&key, &val, it) == 0) {
            if (key) {
                std::string tname(key);
                impl_->tensor_registry[tname] = val;
                impl_->tensor_names.push_back(tname);
            }
        }
        mlx_map_string_to_array_iterator_free(it);
    }

    metadata_.total_tensors = impl_->tensor_names.size();
    loaded_ = true;

    std::cout << "[HavenMLX] ✓ MLX Model Loaded into RAM! (" << metadata_.total_tensors << " tensors registered via mlxc)\n" << std::flush;
    return true;
}

void HavenMLXEngine::unload() {
    if (impl_) {
        impl_->cleanup();
    }
    loaded_ = false;
    model_dir_.clear();
    metadata_ = MLXModelMetadata();
}

bool HavenMLXEngine::has_tensor(const std::string& name) const {
    if (!loaded_ || !impl_) return false;
    return impl_->tensor_registry.find(name) != impl_->tensor_registry.end();
}

std::vector<std::string> HavenMLXEngine::get_tensor_names() const {
    if (!loaded_ || !impl_) return {};
    return impl_->tensor_names;
}

} // namespace haven

#else // !defined(HAVEN_HAS_MLX)

namespace haven {

struct HavenMLXEngine::Impl {};

HavenMLXEngine::HavenMLXEngine() = default;
HavenMLXEngine::~HavenMLXEngine() = default;

bool HavenMLXEngine::is_available() {
    return false;
}

bool HavenMLXEngine::is_mlx_model_dir(const std::string&) {
    return false;
}

bool HavenMLXEngine::inspect_mlx_dir(const std::string&, MLXModelMetadata&) {
    return false;
}

bool HavenMLXEngine::load_model(const std::string&, std::string& err_out) {
    err_out = "MLX runtime is not available on this platform.";
    return false;
}

void HavenMLXEngine::unload() {
    loaded_ = false;
}

bool HavenMLXEngine::has_tensor(const std::string&) const {
    return false;
}

std::vector<std::string> HavenMLXEngine::get_tensor_names() const {
    return {};
}

} // namespace haven

#endif // HAVEN_HAS_MLX
