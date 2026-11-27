#include "haven/haven_chunked_prefill.h"
#include <algorithm>

namespace haven {

ChunkedPrefillScheduler::ChunkedPrefillScheduler(size_t chunk_size)
    : chunk_size_(chunk_size > 0 ? chunk_size : 1)
{
}

ChunkedPrefillScheduler::~ChunkedPrefillScheduler() = default;

uint64_t ChunkedPrefillScheduler::submit_prefill_task(const std::vector<uint32_t>& prompt_tokens) {
    PrefillTask task;
    task.task_id = next_task_id_++;
    task.tokens = prompt_tokens;
    task.processed_tokens = 0;
    task.is_complete = prompt_tokens.empty();

    pending_tasks_.push_back(task);
    return task.task_id;
}

bool ChunkedPrefillScheduler::step_prefill(
    const std::function<void(uint32_t token, size_t pos)>& on_token_step)
{
    if (pending_tasks_.empty()) return false;

    const size_t start_pos = pending_tasks_.front().processed_tokens;
    const size_t total_len = pending_tasks_.front().tokens.size();
    const size_t remaining = (total_len > start_pos) ? (total_len - start_pos) : 0;
    const size_t step = (chunk_size_ > 0) ? chunk_size_ : 1;
    const size_t slice = std::min(step, remaining);

    // Copy slice before invoking callback so reentrant submit_prefill_task() cannot invalidate references
    std::vector<uint32_t> batch_tokens;
    batch_tokens.reserve(slice);
    for (size_t i = 0; i < slice; ++i) {
        batch_tokens.push_back(pending_tasks_.front().tokens[start_pos + i]);
    }

    pending_tasks_.front().processed_tokens = start_pos + slice;
    if (pending_tasks_.front().processed_tokens >= total_len) {
        pending_tasks_.front().is_complete = true;
        pending_tasks_.erase(pending_tasks_.begin());
    }

    if (on_token_step) {
        for (size_t i = 0; i < batch_tokens.size(); ++i) {
            on_token_step(batch_tokens[i], start_pos + i);
        }
    }

    return true;
}

} // namespace haven
