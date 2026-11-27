#pragma once

#include "haven_types.h"
#include <string>
#include <unordered_map>
#include <vector>
#include <fstream>

namespace haven {

class GgufLoader {
public:
    GgufLoader();
    ~GgufLoader();
    GgufLoader(const GgufLoader&) = delete;
    GgufLoader& operator=(const GgufLoader&) = delete;

    bool load_file(const std::string& filepath);
    void close();

    const ModelConfig& get_config() const { return config_; }
    const TensorDesc* get_tensor(const std::string& name) const;
    const std::vector<std::string>& get_vocabulary() const { return vocab_; }

    const std::unordered_map<std::string, TensorDesc>& get_tensors() const { return tensors_; }
    size_t get_tensor_count() const { return tensors_.size(); }
    size_t get_file_size() const { return file_size_; }

private:
    bool parse_header();
    bool parse_metadata();
    bool parse_tensors();

    void* mmap_handle_ = nullptr;
    const uint8_t* mmap_data_ = nullptr;
    size_t file_size_ = 0;

    ModelConfig config_;
    std::vector<std::string> vocab_;
    std::unordered_map<std::string, TensorDesc> tensors_;
};

} // namespace haven