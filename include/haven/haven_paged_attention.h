#pragma once

#include "haven_types.h"
#include <vector>
#include <memory>
#include <cstdint>
#include <unordered_map>
#include <mutex>

namespace haven {

// Virtual Page block of 16 tokens per block
constexpr size_t PAGE_BLOCK_SIZE = 16;

struct KVPagedBlock {
    int32_t block_id = -1;
    size_t num_tokens = 0;
    // Paged KV storage: [num_layers, 2 (K/V), PAGE_BLOCK_SIZE, num_kv_heads, head_dim]
    std::vector<float> data; 
    uint32_t ref_count = 1;
};

class PagedAttentionManager {
public:
    PagedAttentionManager(
        uint32_t num_layers = 32,
        uint32_t num_kv_heads = 8,
        uint32_t head_dim = 128,
        size_t total_blocks = 2048
    );
    ~PagedAttentionManager();

    // Allocates a new physical block or returns free block
    int32_t allocate_block();

    // Appends token KV to a sequence's virtual block table
    void append_token_kv(
        uint64_t seq_id,
        uint32_t layer_idx,
        uint32_t token_pos,
        const float* k_vec,
        const float* v_vec
    );

    // Retrieves physical block by virtual block index
    KVPagedBlock* get_block(int32_t block_id);

    // Forks a sequence virtual block table with Copy-on-Write (instant timeline branch)
    uint64_t fork_sequence(uint64_t parent_seq_id, uint64_t child_seq_id);

    // Frees a sequence and reclaims unreferenced blocks
    void free_sequence(uint64_t seq_id);

    // Clears all allocations
    void reset();

    size_t get_num_free_blocks() const {
        std::lock_guard<std::recursive_mutex> lock(paged_mutex_);
        return free_block_ids_.size();
    }
    size_t get_num_allocated_blocks() const {
        std::lock_guard<std::recursive_mutex> lock(paged_mutex_);
        return total_blocks_ - free_block_ids_.size();
    }
    const std::vector<int32_t>& get_block_table(uint64_t seq_id) const;

private:
    mutable std::recursive_mutex paged_mutex_;
    uint32_t num_layers_;
    uint32_t num_kv_heads_;
    uint32_t head_dim_;
    size_t total_blocks_;
    size_t block_data_size_;

    std::vector<KVPagedBlock> block_pool_;
    std::vector<int32_t> free_block_ids_;
    std::unordered_map<uint64_t, std::vector<int32_t>> sequence_block_tables_;
    static const std::vector<int32_t> empty_table_;
};

} // namespace haven
