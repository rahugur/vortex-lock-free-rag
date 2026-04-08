#include "scheduler/batch_scheduler.h"
#include "utils/metrics.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <chrono>

namespace vortex {

BatchScheduler::BatchScheduler(forge::ThreadPool& pool, LLMBackendFn backend,
                               const BatchSchedulerConfig& config)
    : pool_(pool),
      backend_(std::move(backend)),
      config_(config),
      semaphore_(config.max_concurrent_calls),
      rate_limiter_(config.rate_limit, config.rate_burst) {

    scheduler_thread_ = std::thread(&BatchScheduler::scheduler_loop, this);
    spdlog::debug("BatchScheduler started: max_batch={}, max_wait={}us, "
                  "max_concurrent={}, rate={}/s",
                  config_.max_batch_size, config_.max_wait_us,
                  config_.max_concurrent_calls, config_.rate_limit);
}

BatchScheduler::~BatchScheduler() {
    stop_.store(true, std::memory_order_release);
    if (scheduler_thread_.joinable()) {
        scheduler_thread_.join();
    }
}

forge::Future<std::string> BatchScheduler::submit(
    const std::string& prompt, const std::string& system_message) {

    auto promise = std::make_shared<forge::Promise<std::string>>();
    auto future = promise->get_future();

    InferenceRequest req;
    req.prompt = prompt;
    req.system_message = system_message;
    req.promise = std::move(promise);
    req.submitted_at = std::chrono::steady_clock::now();
    req.prefix_hash = compute_prefix_hash(system_message);

    incoming_.push(std::move(req));
    pending_.fetch_add(1, std::memory_order_relaxed);

    return future;
}

uint64_t BatchScheduler::compute_prefix_hash(const std::string& system_message) {
    // FNV-1a hash of system message
    uint64_t hash = 14695981039346656037ULL;
    for (char c : system_message) {
        hash ^= static_cast<uint64_t>(c);
        hash *= 1099511628211ULL;
    }
    return hash;
}

void BatchScheduler::scheduler_loop() {
    while (!stop_.load(std::memory_order_acquire)) {
        // Drain incoming queue
        std::vector<InferenceRequest> pending;
        while (auto req = incoming_.pop()) {
            pending.push_back(std::move(*req));
        }

        if (pending.empty()) {
            std::this_thread::sleep_for(
                std::chrono::microseconds(config_.max_wait_us / 10));
            continue;
        }

        // Group by prefix hash
        std::unordered_map<uint64_t, std::vector<InferenceRequest>> groups;
        for (auto& req : pending) {
            groups[req.prefix_hash].push_back(std::move(req));
        }

        // Form batches from each group
        for (auto& [hash, requests] : groups) {
            for (size_t i = 0; i < requests.size(); i += config_.max_batch_size) {
                size_t end = std::min(i + config_.max_batch_size, requests.size());
                std::vector<InferenceRequest> batch(
                    std::make_move_iterator(requests.begin() + i),
                    std::make_move_iterator(requests.begin() + end));

                batches_.fetch_add(1, std::memory_order_relaxed);
                Metrics::instance().batches_formed.fetch_add(1, std::memory_order_relaxed);
                Metrics::instance().requests_batched.fetch_add(
                    batch.size(), std::memory_order_relaxed);

                // Submit batch processing to thread pool
                auto batch_ptr = std::make_shared<std::vector<InferenceRequest>>(
                    std::move(batch));
                pool_.submit([this, batch_ptr]() {
                    process_batch(std::move(*batch_ptr));
                });
            }
        }
    }

    // Drain remaining on shutdown
    while (auto req = incoming_.pop()) {
        try {
            req->promise->set_exception(
                std::make_exception_ptr(
                    std::runtime_error("Scheduler shutting down")));
        } catch (...) {}
    }
}

void BatchScheduler::process_batch(std::vector<InferenceRequest> batch) {
    for (auto& req : batch) {
        // Acquire semaphore slot (blocks if at concurrency limit)
        semaphore_.acquire();

        // Rate limit
        rate_limiter_.acquire();

        try {
            std::string result = backend_(req.prompt);
            req.promise->set_value(std::move(result));
        } catch (...) {
            req.promise->set_exception(std::current_exception());
        }

        semaphore_.release();
        pending_.fetch_sub(1, std::memory_order_relaxed);
    }
}

}  // namespace vortex
