#pragma once
// Radix-tree prefix cache for KV-cache reuse.
//
// When many RAG queries share the same system prompt + context pattern,
// their KV-cache prefixes overlap. This structure detects and tracks
// prefix matches for routing decisions.
//
// Based on SGLang's RadixAttention concept.

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace vortex {

class PrefixCache {
public:
    explicit PrefixCache(size_t max_entries = 10000)
        : max_entries_(max_entries) {}

    /// Find the longest matching prefix. Returns (cache_id, prefix_length).
    /// cache_id is 0 if no match found.
    std::pair<uint64_t, size_t> find_prefix(const std::string& text) const;

    /// Insert a prefix and associate it with a cache_id.
    void insert(const std::string& text, uint64_t cache_id);

    /// Number of cached prefixes.
    size_t size() const;

    /// Total hits across all lookups.
    uint64_t total_hits() const { return hits_.load(std::memory_order_relaxed); }
    uint64_t total_misses() const { return misses_.load(std::memory_order_relaxed); }

private:
    struct Node {
        std::unordered_map<char, std::unique_ptr<Node>> children;
        uint64_t cache_id = 0;          // Non-zero if this node is a complete prefix
        mutable std::atomic<uint32_t> hits{0};  // For eviction priority
    };

    Node root_;
    mutable std::mutex mutex_;  // Protects tree structure
    size_t max_entries_;
    std::atomic<size_t> size_{0};
    mutable std::atomic<uint64_t> hits_{0};
    mutable std::atomic<uint64_t> misses_{0};
};

}  // namespace vortex
