// Experimental canonical-weight cache for CPU MUL_MAT_ID. No model pruning.
#pragma once
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <vector>

namespace moe_cache {
struct stats {
    size_t requests = 0, hits = 0, misses = 0, evictions = 0;
    size_t copied_bytes = 0, allocated_bytes = 0;
    size_t oversized = 0, budget_fallback = 0, identity_fallback = 0;
};
struct entry {
    const void * source;
    size_t expert_bytes;
    int experts, slots;
    std::vector<uint8_t> bytes;
    std::vector<int> owner, location;
    std::vector<uint64_t> age;
    std::vector<int> changed_slots;
    uint64_t clock = 0;
    bool stores_weights;
    entry(const void * data, size_t size, int count, int capacity, bool keep_weight_bytes) :
        source(data), expert_bytes(size), experts(count), slots(capacity),
        owner(capacity, -1), location(count, -1), age(capacity, 0), stores_weights(keep_weight_bytes) {
        if (stores_weights) bytes.resize(size * capacity);
    }
};
class cache {
    int capacity;
    size_t limit;
    bool stores_weights;
    std::map<const void *, std::unique_ptr<entry>> tensors;
public:
    stats counters;
    cache(int slots, size_t byte_limit, bool keep_weight_bytes = true) :
        capacity(slots), limit(byte_limit), stores_weights(keep_weight_bytes) {}
    int slots() const { return capacity; }
    // source and key are immutable for this backend/context lifetime.
    // On success all selected experts are pinned until this synchronous op ends.
    entry * prepare(const void * key, const void * source, size_t expert_bytes,
                    int experts, const int32_t * ids, size_t count,
                    std::vector<int32_t> & remapped) {
        ++counters.requests;
        if (!source || !expert_bytes || experts <= 0 || count == 0) return nullptr;
        std::vector<uint8_t> needed(experts, 0);
        size_t unique = 0;
        for (size_t i = 0; i < count; ++i) {
            if (ids[i] < 0 || ids[i] >= experts) return nullptr;
            if (!needed[ids[i]]) { needed[ids[i]] = 1; ++unique; }
        }
        const int slots = std::min(capacity, experts);
        if (unique > size_t(slots)) { ++counters.oversized; return nullptr; }
        auto found = tensors.find(key);
        if (found == tensors.end()) {
            if (expert_bytes > std::numeric_limits<size_t>::max() / size_t(slots) ||
                expert_bytes * slots > limit - counters.allocated_bytes) {
                ++counters.budget_fallback; return nullptr;
            }
            std::unique_ptr<entry> value(new entry(source, expert_bytes, experts, slots, stores_weights));
            found = tensors.emplace(key, std::move(value)).first;
            counters.allocated_bytes += expert_bytes * slots;
        }
        auto & value = *found->second;
        value.changed_slots.clear();
        if (value.source != source || value.expert_bytes != expert_bytes || value.experts != experts) {
            ++counters.identity_fallback; return nullptr;
        }
        // Choose victims only from experts NOT used by this operation. A hit
        // appearing later in the ID list cannot be evicted by an earlier miss.
        for (int expert = 0; expert < experts; ++expert) {
            if (!needed[expert]) continue;
            int slot = value.location[expert];
            if (slot >= 0) { ++counters.hits; }
            else {
                ++counters.misses;
                slot = -1;
                for (int candidate = 0; candidate < slots; ++candidate) {
                    int old = value.owner[candidate];
                    if (old < 0) { slot = candidate; break; }
                    if (!needed[old] && (slot < 0 || value.age[candidate] < value.age[slot])) slot = candidate;
                }
                if (slot < 0) throw std::logic_error("no unpinned slot");
                if (value.stores_weights) {
                    std::memcpy(value.bytes.data() + slot * expert_bytes,
                                static_cast<const uint8_t *>(source) + expert * expert_bytes, expert_bytes);
                    counters.copied_bytes += expert_bytes;
                }
                int old = value.owner[slot];
                if (old >= 0) { value.location[old] = -1; ++counters.evictions; }
                value.owner[slot] = expert; value.location[expert] = slot;
                value.changed_slots.push_back(slot);
            }
            value.age[slot] = ++value.clock;
        }
        // Touch in request order so eviction is recency-based across calls.
        remapped.resize(count);
        for (size_t i = 0; i < count; ++i) {
            remapped[i] = value.location[ids[i]];
            value.age[remapped[i]] = ++value.clock;
        }
        return &value;
    }
};
} // namespace moe_cache
