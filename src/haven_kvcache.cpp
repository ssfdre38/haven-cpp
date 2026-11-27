#include "haven/haven_kvcache.h"
#include <algorithm>
#include <cstring>
#include <iostream>

namespace haven {

DynamicKVCacheManager::DynamicKVCacheManager(
    uint32_t num_layers,
    uint32_t max_context_length,
    uint32_t num_kv_heads,
    uint32_t head_dim,
    float pruning_threshold,
    float salience_decay)
    : num_layers_(num_layers),
      max_context_length_(max_context_length),
      num_kv_heads_(num_kv_heads),
      head_dim_(head_dim),
      pruning_threshold_(pruning_threshold),
      salience_decay_(salience_decay),
      current_length_(0),
      capacity_(std::min(128u, max_context_length))
{
    reinit(num_layers_, max_context_length_, num_kv_heads_, head_dim_);
}

void DynamicKVCacheManager::reinit(
    uint32_t num_layers,
    uint32_t max_context_length,
    uint32_t num_kv_heads,
    uint32_t head_dim)
{
    num_layers_ = num_layers;
    max_context_length_ = max_context_length;
    num_kv_heads_ = num_kv_heads;
    head_dim_ = head_dim;
    current_length_ = 0;
    capacity_ = std::min(128u, max_context_length_);

    layers_.clear();
    layers_.resize(num_layers_);
    const size_t initial_stride = (size_t)capacity_ * (num_kv_heads_ * head_dim_);

    for (uint32_t l = 0; l < num_layers_; ++l) {
        layers_[l].keys.resize(initial_stride, 0.0f);
        layers_[l].values.resize(initial_stride, 0.0f);
        layers_[l].salience_scores.resize(capacity_, 1.0f);
        layers_[l].active_mask.resize(capacity_, true);
        layers_[l].current_len = 0;
        layers_[l].kv_dim = num_kv_heads_ * head_dim_;
    }
}

void DynamicKVCacheManager::write(uint32_t layer, int pos, const float* k, const float* v, uint32_t kv_dim) {
    if (!k || !v || kv_dim == 0 || layer >= num_layers_ || pos < 0 || static_cast<uint32_t>(pos) >= max_context_length_) return;

    const size_t elem_stride = std::max(static_cast<size_t>(num_kv_heads_) * head_dim_, static_cast<size_t>(kv_dim));

    // Dynamically grow capacity if needed
    if ((uint32_t)pos >= capacity_ || (static_cast<size_t>(pos) + 1) * kv_dim > layers_[layer].keys.size()) {
        uint32_t new_cap = std::min(max_context_length_, std::max(capacity_ * 2, (uint32_t)pos + 512));
        if (new_cap < (uint32_t)pos + 1) new_cap = (uint32_t)pos + 1;
        capacity_ = std::max(capacity_, new_cap);
        const size_t new_stride = static_cast<size_t>(capacity_) * elem_stride;
        for (uint32_t l = 0; l < num_layers_; ++l) {
            if (layers_[l].keys.size() < new_stride) {
                layers_[l].keys.resize(new_stride, 0.0f);
                layers_[l].values.resize(new_stride, 0.0f);
            }
            if (layers_[l].salience_scores.size() < capacity_) {
                layers_[l].salience_scores.resize(capacity_, 1.0f);
                layers_[l].active_mask.resize(capacity_, true);
            }
        }
    }

    auto& lay = layers_[layer];
    lay.kv_dim = kv_dim;
    const size_t offset = static_cast<size_t>(pos) * static_cast<size_t>(kv_dim);
    const size_t num_bytes = static_cast<size_t>(kv_dim) * sizeof(float);

    if (offset + kv_dim <= lay.keys.size()) {
        std::memcpy(lay.keys.data() + offset, k, num_bytes);
        std::memcpy(lay.values.data() + offset, v, num_bytes);
    }

    if (static_cast<size_t>(pos) < lay.salience_scores.size()) {
        lay.salience_scores[pos] = 1.0f; // New token starts with full salience
        lay.active_mask[pos] = true;
    }
    lay.current_len = std::max(lay.current_len, pos + 1);
    current_length_ = std::max(current_length_, pos + 1);
}

void DynamicKVCacheManager::append(uint32_t layer, const float* k, const float* v, uint32_t kv_dim) {
    if (layer >= num_layers_) return;
    write(layer, layers_[layer].current_len, k, v, kv_dim);
}

void DynamicKVCacheManager::update_salience(uint32_t layer, const float* attention_weights, int seq_len) {
    if (!attention_weights || seq_len <= 0 || layer >= num_layers_) return;

    auto& lay = layers_[layer];
    const size_t max_len = std::min({static_cast<size_t>(seq_len), static_cast<size_t>(current_length_ + 1), lay.salience_scores.size()});

    // Apply temporal exponential decay and accumulate new attention weight
    for (size_t i = 0; i < max_len; ++i) {
        lay.salience_scores[i] = lay.salience_scores[i] * salience_decay_ + attention_weights[i];
    }
}

size_t DynamicKVCacheManager::prune_stale_entries(uint32_t layer) {
    if (layer >= num_layers_) return 0;

    auto& lay = layers_[layer];
    size_t pruned_count = 0;
    const size_t vec_stride = (lay.kv_dim > 0)
        ? static_cast<size_t>(lay.kv_dim)
        : (static_cast<size_t>(num_kv_heads_) * head_dim_);
    const size_t safe_len = std::min({static_cast<size_t>(std::max(0, lay.current_len)), lay.active_mask.size(), lay.salience_scores.size()});

    // Pinned Anchors: Never prune the first 4 tokens (system / soul core anchors)
    const size_t pin_count = std::min<size_t>(4, safe_len);

    for (size_t i = pin_count; i < safe_len; ++i) {
        if (lay.active_mask[i] && lay.salience_scores[i] < pruning_threshold_) {
            lay.active_mask[i] = false;
            // Zero out memory to save cache bandwidth during attention GEMV
            if (vec_stride > 0 && (i + 1) * vec_stride <= lay.keys.size()) {
                std::memset(lay.keys.data() + i * vec_stride, 0, vec_stride * sizeof(float));
                std::memset(lay.values.data() + i * vec_stride, 0, vec_stride * sizeof(float));
            }
            pruned_count++;
        }
    }

    return pruned_count;
}

size_t DynamicKVCacheManager::prune_all_layers() {
    size_t total_pruned = 0;
    for (uint32_t l = 0; l < num_layers_; ++l) {
        total_pruned += prune_stale_entries(l);
    }
    return total_pruned;
}

void DynamicKVCacheManager::truncate(int new_len) {
    if (new_len <= 0) {
        reset();
        return;
    }
    if (new_len < current_length_) {
        current_length_ = new_len;
    }
    for (auto& lay : layers_) {
        if (lay.current_len > new_len) {
            lay.current_len = new_len;
        }
    }
}

void DynamicKVCacheManager::reset() {
    current_length_ = 0;
    for (auto& lay : layers_) {
        std::fill(lay.salience_scores.begin(), lay.salience_scores.end(), 1.0f);
        std::fill(lay.active_mask.begin(), lay.active_mask.end(), true);
        lay.current_len = 0;
    }
}

} // namespace haven