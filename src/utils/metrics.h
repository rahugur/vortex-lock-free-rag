#pragma once
// Global atomic counters for Vortex runtime metrics.

#include <atomic>
#include <cstdint>

namespace vortex {

struct Metrics {
    // Index
    std::atomic<uint64_t> vectors_indexed{0};
    std::atomic<uint64_t> searches_completed{0};

    // Pipeline
    std::atomic<uint64_t> queries_received{0};
    std::atomic<uint64_t> queries_completed{0};
    std::atomic<uint64_t> queries_failed{0};

    // Latency accumulators (microseconds, for computing averages)
    std::atomic<uint64_t> total_search_us{0};
    std::atomic<uint64_t> total_embed_us{0};
    std::atomic<uint64_t> total_pipeline_overhead_us{0};

    // Scheduler
    std::atomic<uint64_t> batches_formed{0};
    std::atomic<uint64_t> requests_batched{0};
    std::atomic<uint64_t> prefix_cache_hits{0};
    std::atomic<uint64_t> prefix_cache_misses{0};

    // Tokens
    std::atomic<uint64_t> total_prompt_tokens{0};
    std::atomic<uint64_t> total_completion_tokens{0};
    std::atomic<uint64_t> total_embed_tokens{0};

    static Metrics& instance() {
        static Metrics m;
        return m;
    }
};

}  // namespace vortex
