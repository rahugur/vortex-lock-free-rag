#include <catch2/catch_test_macros.hpp>

#include "scheduler/batch_scheduler.h"
#include "scheduler/prefix_cache.h"

#include <core/thread_pool.h>

#include <thread>
#include <chrono>

using namespace vortex;

TEST_CASE("Prefix cache - basic insert and find", "[scheduler]") {
    PrefixCache cache;

    cache.insert("You are a helpful assistant", 1);
    cache.insert("You are a code reviewer", 2);

    auto [id1, len1] = cache.find_prefix("You are a helpful assistant. Answer:");
    REQUIRE(id1 == 1);
    REQUIRE(len1 == 27);  // Full "You are a helpful assistant"

    auto [id2, len2] = cache.find_prefix("You are a code reviewer. Check:");
    REQUIRE(id2 == 2);
    REQUIRE(len2 == 23);  // Full "You are a code reviewer"
}

TEST_CASE("Prefix cache - no match", "[scheduler]") {
    PrefixCache cache;
    cache.insert("System prompt A", 1);

    auto [id, len] = cache.find_prefix("Completely different text");
    REQUIRE(id == 0);
    REQUIRE(len == 0);
}

TEST_CASE("Prefix cache - longest match", "[scheduler]") {
    PrefixCache cache;
    cache.insert("You are", 1);
    cache.insert("You are a helpful", 2);
    cache.insert("You are a helpful assistant", 3);

    auto [id, len] = cache.find_prefix("You are a helpful assistant. Do X.");
    REQUIRE(id == 3);  // Longest match
}

TEST_CASE("Prefix cache - tracks hits", "[scheduler]") {
    PrefixCache cache;
    cache.insert("prefix", 1);

    cache.find_prefix("prefix and more");
    cache.find_prefix("prefix and other");
    cache.find_prefix("no match");

    REQUIRE(cache.total_hits() == 2);
    REQUIRE(cache.total_misses() == 1);
}

TEST_CASE("Batch scheduler - single request", "[scheduler]") {
    forge::ThreadPool pool(2);

    auto backend = [](const std::string& prompt) -> std::string {
        return "Response to: " + prompt;
    };

    BatchSchedulerConfig cfg;
    cfg.max_concurrent_calls = 4;
    cfg.rate_limit = 100.0;
    cfg.rate_burst = 10.0;

    BatchScheduler scheduler(pool, backend, cfg);

    auto future = scheduler.submit("Hello world");
    auto result = future.get();

    REQUIRE(result == "Response to: Hello world");
}

TEST_CASE("Batch scheduler - concurrent requests", "[scheduler]") {
    forge::ThreadPool pool(4);

    std::atomic<int> call_count{0};
    auto backend = [&](const std::string& prompt) -> std::string {
        call_count.fetch_add(1);
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        return "OK";
    };

    BatchSchedulerConfig cfg;
    cfg.max_concurrent_calls = 8;
    cfg.rate_limit = 1000.0;
    cfg.rate_burst = 100.0;

    BatchScheduler scheduler(pool, backend, cfg);

    // Submit 20 requests
    std::vector<forge::Future<std::string>> futures;
    for (int i = 0; i < 20; ++i) {
        futures.push_back(scheduler.submit("request " + std::to_string(i)));
    }

    // Wait for all
    for (auto& f : futures) {
        auto result = f.get();
        REQUIRE(result == "OK");
    }

    REQUIRE(call_count.load() == 20);
}

TEST_CASE("Batch scheduler - handles errors", "[scheduler]") {
    forge::ThreadPool pool(2);

    auto backend = [](const std::string& prompt) -> std::string {
        if (prompt == "fail") {
            throw std::runtime_error("Backend error");
        }
        return "OK";
    };

    BatchScheduler scheduler(pool, backend);

    auto f1 = scheduler.submit("succeed");
    auto f2 = scheduler.submit("fail");

    REQUIRE(f1.get() == "OK");
    REQUIRE_THROWS_AS(f2.get(), std::runtime_error);
}
