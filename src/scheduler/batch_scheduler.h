#pragma once
// Continuous Batching Request Scheduler.
//
// Groups incoming LLM inference requests into batches to maximize throughput.
// Uses Forge's MPSC queue for lock-free request ingestion and semaphore
// for concurrency control.
//
// Scheduling algorithm:
//   1. Drain incoming requests from MPSC queue
//   2. Group by prefix (for KV-cache reuse)
//   3. Form batches up to max_batch_size
//   4. Submit batches, respecting rate limits
//   5. Deliver results via Future/Promise

#include "scheduler/prefix_cache.h"

#include <core/future.h>
#include <core/mpsc_queue.h>
#include <core/rate_limiter.h>
#include <core/semaphore.h>
#include <core/thread_pool.h>

#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace vortex {

/// A single inference request.
struct InferenceRequest {
    std::string prompt;
    std::string system_message;
    std::shared_ptr<forge::Promise<std::string>> promise;
    std::chrono::steady_clock::time_point submitted_at;
    uint64_t prefix_hash = 0;
};

/// LLM backend function: takes prompt, returns completion.
using LLMBackendFn = std::function<std::string(const std::string& prompt)>;

struct BatchSchedulerConfig {
    uint32_t max_batch_size = 32;
    uint32_t max_wait_us = 5000;           // 5ms max wait before forming batch
    uint32_t max_concurrent_calls = 8;
    double   rate_limit = 60.0;            // requests/sec
    double   rate_burst = 10.0;
    bool     enable_prefix_cache = true;
};

class BatchScheduler {
public:
    BatchScheduler(forge::ThreadPool& pool, LLMBackendFn backend,
                   const BatchSchedulerConfig& config = {});
    ~BatchScheduler();

    // Non-copyable
    BatchScheduler(const BatchScheduler&) = delete;
    BatchScheduler& operator=(const BatchScheduler&) = delete;

    /// Submit an inference request. Returns a Future for the result.
    forge::Future<std::string> submit(const std::string& prompt,
                                       const std::string& system_message = "");

    /// Number of pending requests.
    uint32_t pending() const { return pending_.load(std::memory_order_relaxed); }

    /// Total batches formed.
    uint64_t batches_formed() const { return batches_.load(std::memory_order_relaxed); }

private:
    void scheduler_loop();
    void process_batch(std::vector<InferenceRequest> batch);
    uint64_t compute_prefix_hash(const std::string& system_message);

    forge::ThreadPool& pool_;
    LLMBackendFn backend_;
    BatchSchedulerConfig config_;

    forge::MPSCQueue<InferenceRequest> incoming_;
    forge::Semaphore semaphore_;
    forge::RateLimiter rate_limiter_;
    PrefixCache prefix_cache_;

    std::atomic<uint32_t> pending_{0};
    std::atomic<uint64_t> batches_{0};
    std::atomic<bool> stop_{false};
    std::thread scheduler_thread_;
};

}  // namespace vortex
