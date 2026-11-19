#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include "haven/haven_types.h"

namespace haven {

struct MLXTensorInfo {
    std::string name;
    std::vector<int> shape;
    std::string dtype;
    size_t nbytes = 0;
};

struct MLXModelMetadata {
    std::string model_name;
    std::string model_type;
    int hidden_size = 0;
    int intermediate_size = 0;
    int num_hidden_layers = 0;
    int num_attention_heads = 0;
    int num_key_value_heads = 0;
    int vocab_size = 0;
    float rms_norm_eps = 1e-6f;
    float rope_theta = 10000.0f;
    bool is_quantized = false;
    int quant_bits = 16;
    int quant_group_size = 64;
    size_t total_tensors = 0;
    uintmax_t total_bytes = 0;
    std::string size_formatted;
};

class HavenMLXEngine {
public:
    HavenMLXEngine();
    ~HavenMLXEngine();

    // Check if MLX runtime dynamic library is operational
    static bool is_available();

    // Directory inspection: checks for config.json + *.safetensors
    static bool is_mlx_model_dir(const std::string& dir_path);
    static bool inspect_mlx_dir(const std::string& dir_path, MLXModelMetadata& meta);

    // Load an MLX / Safetensors model directory into active memory
    bool load_model(const std::string& dir_path, std::string& err_out);
    void unload();

    bool is_loaded() const { return loaded_; }
    const std::string& get_model_dir() const { return model_dir_; }
    const MLXModelMetadata& get_metadata() const { return metadata_; }

    bool has_tensor(const std::string& name) const;
    std::vector<std::string> get_tensor_names() const;

private:
    bool loaded_ = false;
    std::string model_dir_;
    MLXModelMetadata metadata_;

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace haven
