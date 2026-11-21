#include "haven/haven_paged_attention.h"
#include <cstring>
#include <algorithm>

namespace haven {

const std::vector<int32_t> PagedAttentionManager::empty_table_ = {};

PagedAttentionManager::PagedAttentionManager(
    uint32_t num_layers,
    uint32_t num_kv_heads,
    uint32_t head_dim,
    size_t total_blocks)
    : num_layers_(num_layers),
      num_kv_heads_(num_kv_heads),
      head_dim_(head_dim),
      total_blocks_(total_blocks)
{
    // Block size: num_layers * 2 (K/V) * PAGE_BLOCK_SIZE * num_kv_heads * head_dim
    block_data_size_ = num_layers_ * 2 * PAGE_BLOCK_SIZE * num_kv_heads_ * head_dim_;
    block_pool_.resize(total_blocks_);
    free_block_ids_.reserve(total_blocks_);

    for (size_t i = 0; i < total_blocks_; ++i) {
        block_pool_[i].block_id = (int32_t)i;
        block_pool_[i].num_tokens = 0;
        block_pool_[i].ref_count = 0;
        // Data is lazily allocated on first allocate_block() to avoid 8.0 GiB upfront memory bomb
        free_block_ids_.push_back((int32_t)i);
    }
}

PagedAttentionManager::~PagedAttentionManager() = default;

int32_t PagedAttentionManager::allocate_block() {
    if (free_block_ids_.empty()) {
        return -1; // Out of memory blocks
    }
    int32_t block_id = free_block_ids_.back();
    free_block_ids_.pop_back();

    block_pool_[block_id].ref_count = 1;
    block_pool_[block_id].num_tokens = 0;
    if (block_pool_[block_id].data.size() != block_data_size_) {
        block_pool_[block_id].data.assign(block_data_size_, 0.0f);
    }
    return block_id;
}

KVPagedBlock* PagedAttentionManager::get_block(int32_t block_id) {
    if (block_id >= 0 && (size_t)block_id < block_pool_.size()) {
        return &block_pool_[block_id];
    }
    return nullptr;
}

void PagedAttentionManager::append_token_kv(
    uint64_t seq_id,
    uint32_t layer_idx,
    uint32_t token_pos,
    const float* k_vec,
    const float* v_vec)
{
    auto& table = sequence_block_tables_[seq_id];
    size_t block_idx = token_pos / PAGE_BLOCK_SIZE;
    size_t offset_in_block = token_pos % PAGE_BLOCK_SIZE;

    while (table.size() <= block_idx) {
        int32_t new_block = allocate_block();
        if (new_block < 0) return; // Pool full
        table.push_back(new_block);
    }

    int32_t physical_block_id = table[block_idx];
    auto* block = get_block(physical_block_id);
    if (!block) return;

    // Copy-on-Write (CoW) isolation: branch if block is shared by another sequence
    if (block->ref_count > 1) {
        int32_t new_block_id = allocate_block();
        if (new_block_id >= 0) {
            auto* new_block = get_block(new_block_id);
            if (new_block) {
                new_block->num_tokens = block->num_tokens;
                new_block->data = block->data;
                new_block->ref_count = 1;
                block->ref_count--;
                table[block_idx] = new_block_id;
                block = new_block;
            }
        }
    }

    if (offset_in_block >= block->num_tokens) {
        block->num_tokens = offset_in_block + 1;
    }

    // Offset calculation
    size_t head_bytes = num_kv_heads_ * head_dim_;
    size_t layer_stride = 2 * PAGE_BLOCK_SIZE * head_bytes;
    size_t k_offset = (layer_idx * layer_stride) + (0 * PAGE_BLOCK_SIZE * head_bytes) + (offset_in_block * head_bytes);
    size_t v_offset = (layer_idx * layer_stride) + (1 * PAGE_BLOCK_SIZE * head_bytes) + (offset_in_block * head_bytes);

    if (k_vec && k_offset + head_bytes <= block->data.size()) {
        std::memcpy(&block->data[k_offset], k_vec, head_bytes * sizeof(float));
    }
    if (v_vec && v_offset + head_bytes <= block->data.size()) {
        std::memcpy(&block->data[v_offset], v_vec, head_bytes * sizeof(float));
    }
}

uint64_t PagedAttentionManager::fork_sequence(uint64_t parent_seq_id, uint64_t child_seq_id) {
    auto parent_it = sequence_block_tables_.find(parent_seq_id);
    if (parent_it == sequence_block_tables_.end()) return parent_seq_id;

    auto& child_table = sequence_block_tables_[child_seq_id];
    child_table = parent_it->second;

    for (int32_t block_id : child_table) {
        if (block_id >= 0 && (size_t)block_id < block_pool_.size()) {
            block_pool_[block_id].ref_count++;
        }
    }
    return child_seq_id;
}

void PagedAttentionManager::free_sequence(uint64_t seq_id) {
    auto it = sequence_block_tables_.find(seq_id);
    if (it == sequence_block_tables_.end()) return;

    for (int32_t block_id : it->second) {
        if (block_id >= 0 && (size_t)block_id < block_pool_.size()) {
            if (--block_pool_[block_id].ref_count == 0) {
                block_pool_[block_id].num_tokens = 0;
                free_block_ids_.push_back(block_id);
            }
        }
    }
    sequence_block_tables_.erase(it);
}

void PagedAttentionManager::reset() {
    free_block_ids_.clear();
    sequence_block_tables_.clear();
    for (size_t i = 0; i < total_blocks_; ++i) {
        block_pool_[i].ref_count = 0;
        block_pool_[i].num_tokens = 0;
        free_block_ids_.push_back((int32_t)i);
    }
}

const std::vector<int32_t>& PagedAttentionManager::get_block_table(uint64_t seq_id) const {
    auto it = sequence_block_tables_.find(seq_id);
    if (it != sequence_block_tables_.end()) {
        return it->second;
    }
    return empty_table_;
}

} // namespace haven
