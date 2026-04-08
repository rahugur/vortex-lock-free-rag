#include "scheduler/prefix_cache.h"

#include "utils/metrics.h"

namespace vortex {

std::pair<uint64_t, size_t> PrefixCache::find_prefix(const std::string& text) const {
    std::lock_guard<std::mutex> lock(mutex_);

    const Node* current = &root_;
    uint64_t best_cache_id = 0;
    size_t best_length = 0;

    for (size_t i = 0; i < text.size(); ++i) {
        auto it = current->children.find(text[i]);
        if (it == current->children.end()) {
            break;
        }
        current = it->second.get();
        if (current->cache_id != 0) {
            best_cache_id = current->cache_id;
            best_length = i + 1;
            current->hits.fetch_add(1, std::memory_order_relaxed);
        }
    }

    if (best_cache_id != 0) {
        hits_.fetch_add(1, std::memory_order_relaxed);
        Metrics::instance().prefix_cache_hits.fetch_add(1, std::memory_order_relaxed);
    } else {
        misses_.fetch_add(1, std::memory_order_relaxed);
        Metrics::instance().prefix_cache_misses.fetch_add(1, std::memory_order_relaxed);
    }

    return {best_cache_id, best_length};
}

void PrefixCache::insert(const std::string& text, uint64_t cache_id) {
    std::lock_guard<std::mutex> lock(mutex_);

    Node* current = &root_;
    for (char c : text) {
        auto it = current->children.find(c);
        if (it == current->children.end()) {
            current->children[c] = std::make_unique<Node>();
            current = current->children[c].get();
        } else {
            current = it->second.get();
        }
    }
    current->cache_id = cache_id;
    size_t current_size = size_.fetch_add(1, std::memory_order_relaxed);

    // TODO: Implement proper LRU/LFU eviction when size exceeds max_entries_.
    // For now, log a warning. The cache remains correct but unbounded.
    if (current_size + 1 > max_entries_) {
        // Already over limit — eviction not yet implemented.
        // This is safe (cache still works) but memory usage is unbounded.
    }
}

size_t PrefixCache::size() const {
    return size_.load(std::memory_order_relaxed);
}

}  // namespace vortex
