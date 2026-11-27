#include "haven/haven_knowledge_store.h"
#include <iostream>

namespace haven {

StructuredKnowledgeStore::StructuredKnowledgeStore() = default;

void StructuredKnowledgeStore::add_relation(
    uint32_t source_token_id,
    uint32_t cluster_id,
    float boost_magnitude,
    const std::string& concept_label,
    const std::string& target_concept)
{
    uint64_t key = make_composite_key(source_token_id, cluster_id);
    lookup_table_[key] = KnowledgeEntry{
        source_token_id,
        cluster_id,
        boost_magnitude,
        concept_label,
        target_concept
    };
}

void StructuredKnowledgeStore::apply_knowledge_boost(
    float* attention_scores,
    uint32_t current_token_id,
    uint32_t active_cluster_id,
    int seq_len)
{
    if (!attention_scores || lookup_table_.empty() || seq_len <= 0) return;

    uint64_t key = make_composite_key(current_token_id, active_cluster_id);
    auto it = lookup_table_.find(key);
    if (it != lookup_table_.end()) {
        const auto& entry = it->second;
        // Inject factual relational boost across the active sequence attention positions
        for (int i = 0; i < seq_len; ++i) {
            attention_scores[i] += entry.boost_magnitude * (1.0f / (float)(seq_len - i + 1));
        }
    }
}

} // namespace haven