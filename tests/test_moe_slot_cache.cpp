#include "../scripts/moe-slot-cache.h"
#include <cassert>
#include <cstdio>

int main() {
    const int source[] = {10, 20, 30, 40, 50};
    moe_cache::cache cache(2, 8);
    std::vector<int32_t> mapped;
    auto check = [&](std::vector<int32_t> ids) {
        auto * entry = cache.prepare(source, source, sizeof(int), 5, ids.data(), ids.size(), mapped);
        assert(entry && entry->bytes.size() == 8);
        for (size_t i = 0; i < ids.size(); ++i) {
            int copied; memcpy(&copied, entry->bytes.data() + mapped[i] * sizeof(int), sizeof(int));
            assert(copied == source[ids[i]]);
        }
    };
    auto * first = cache.prepare(source, source, sizeof(int), 5,
                                 std::vector<int32_t>{1, 4}.data(), 2, mapped);
    assert(first && first->changed_slots.size() == 2);
    check({1, 1, 4}); // duplicates count as one expert request
    auto * before = cache.prepare(source, source, sizeof(int), 5,
                                  std::vector<int32_t>{0, 4}.data(), 2, mapped);
    assert(before && before->changed_slots.size() == 1); // miss first, hit later stays pinned
    auto * after = cache.prepare(source, source, sizeof(int), 5,
                                 std::vector<int32_t>{3, 0}.data(), 2, mapped);
    assert(after && after->changed_slots.size() == 1); // actual eviction
    const int32_t too_large[] = {0, 1, 2};
    assert(!cache.prepare(source, source, 4, 5, too_large, 3, mapped));
    assert(cache.counters.oversized == 1 && cache.counters.allocated_bytes == 8);
    const int second[] = {1, 2, 3, 4, 5};
    const int32_t one[] = {0};
    assert(!cache.prepare(second, second, 4, 5, one, 1, mapped));
    assert(cache.counters.budget_fallback == 1 && cache.counters.allocated_bytes == 8);
    assert(!cache.prepare(source, second, 4, 5, one, 1, mapped));
    assert(cache.counters.identity_fallback == 1);
    const int32_t invalid[] = {5};
    assert(!cache.prepare(source, source, 4, 5, invalid, 1, mapped));
    assert(cache.counters.evictions == 2 && cache.counters.misses == 4 && cache.counters.hits == 4);
    printf("slot cache: eviction, pinning, duplicates, oversize, budget, identity, bounds passed\n");

    moe_cache::cache metadata_only(2, 8, false);
    auto * metadata = metadata_only.prepare(source, source, sizeof(int), 5, one, 1, mapped);
    assert(metadata && metadata->bytes.empty() && metadata->changed_slots.size() == 1);
    assert(metadata_only.counters.allocated_bytes == 8 && metadata_only.counters.copied_bytes == 0);
    assert(mapped.size() == 1 && mapped[0] >= 0 && mapped[0] < 2);
    printf("slot cache metadata-only mode: no duplicate host weight allocation passed\n");
}
