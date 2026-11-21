#pragma once

#include "haven_types.h"
#include <vector>
#include <immintrin.h>
#include <cmath>

namespace haven {

// Represents a stateful, pruned key-value cache layer
struct GatedKVCacheLayer {
    std::vector<float> keys;             // [max_seq, num_kv_heads, head_dim]
    std::vector<float> values;           // [max_seq, num_kv_heads, head_dim]
    std::vector<float> salience_scores;  // [max_seq] Salience tracker per token position
    std::vector<bool> active_mask;       // [max_seq] Boolean active mask (true = active, false = pruned)
    int current_len = 0;
};

class DynamicKVCacheManager {
public:
    DynamicKVCacheManager(
        uint32_t num_layers = 32,
        uint32_t max_context_length = 131072,
        uint32_t num_kv_heads = 8,
        uint32_t head_dim = 128,
        float pruning_threshold = 0.035f,
        float salience_decay = 0.985f
    );

    // Writes K and V vectors for the given layer directly at target position 'pos'
    void write(uint32_t layer, int pos, const float* k, const float* v, uint32_t kv_dim = 1024);

    // Appends new K and V vectors for the given layer at the current position
    void append(uint32_t layer, const float* k, const float* v, uint32_t kv_dim = 1024);

    // Updates token salience scores based on attention weights: salience = decay * salience + attn_weight
    void update_salience(uint32_t layer, const float* attention_weights, int seq_len);

    // Soft-Pruning: Zeroes out and marks inactive any keys/values below the salience threshold
    size_t prune_stale_entries(uint32_t layer);

    // Prunes across all 32 transformer layers and returns total pruned token count
    size_t prune_all_layers();

    const GatedKVCacheLayer& get_layer(uint32_t layer) const { return layers_[layer]; }
    GatedKVCacheLayer& get_layer(uint32_t layer) { return layers_[layer]; }

    int get_current_length() const { return current_length_; }
    void advance_position() { }
    void reset();
    void reinit(uint32_t num_layers, uint32_t max_context_length, uint32_t num_kv_heads, uint32_t head_dim);

    float get_pruning_threshold() const { return pruning_threshold_; }
    void set_pruning_threshold(float t) { pruning_threshold_ = t; }

private:
    uint32_t num_layers_;
    uint32_t max_context_length_;
    uint32_t num_kv_heads_;
    uint32_t head_dim_;
    float pruning_threshold_;
    float salience_decay_;
    int current_length_ = 0;
    uint32_t capacity_ = 128;

    std::vector<GatedKVCacheLayer> layers_;
};

} // namespace haven