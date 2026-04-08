#include <catch2/catch_test_macros.hpp>

#include "index/hnsw_index.h"

#include <atomic>
#include <random>
#include <thread>
#include <vector>

using namespace vortex;

TEST_CASE("Concurrent insert", "[stress]") {
    HNSWConfig cfg;
    cfg.dim = 32;
    cfg.M = 16;
    cfg.ef_construct = 100;
    cfg.max_elements = 10000;
    HNSWIndex index(cfg);

    constexpr uint32_t N = 2000;
    constexpr uint32_t THREADS = 4;

    // Generate vectors
    std::vector<std::vector<float>> vecs(N, std::vector<float>(32));
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (auto& v : vecs) {
        for (auto& x : v) x = dist(rng);
    }

    // Insert from multiple threads
    std::vector<std::thread> threads;
    for (uint32_t t = 0; t < THREADS; ++t) {
        threads.emplace_back([&, t]() {
            uint32_t start = t * (N / THREADS);
            uint32_t end = (t + 1) * (N / THREADS);
            for (uint32_t i = start; i < end; ++i) {
                index.insert(i, vecs[i].data());
            }
        });
    }

    for (auto& th : threads) th.join();

    REQUIRE(index.size() == N);

    // Verify searchability
    auto results = index.search(vecs[0].data(), 5);
    REQUIRE(!results.empty());
}

TEST_CASE("Concurrent search", "[stress]") {
    HNSWConfig cfg;
    cfg.dim = 32;
    cfg.M = 16;
    HNSWIndex index(cfg);

    // Build index single-threaded
    std::vector<std::vector<float>> vecs(1000, std::vector<float>(32));
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (auto& v : vecs) {
        for (auto& x : v) x = dist(rng);
    }
    for (uint32_t i = 0; i < 1000; ++i) {
        index.insert(i, vecs[i].data());
    }

    // Search from multiple threads concurrently
    constexpr uint32_t THREADS = 8;
    constexpr uint32_t QUERIES_PER_THREAD = 100;
    std::atomic<uint32_t> total_results{0};

    std::vector<std::thread> threads;
    for (uint32_t t = 0; t < THREADS; ++t) {
        threads.emplace_back([&, t]() {
            std::mt19937 local_rng(t * 100);
            std::normal_distribution<float> d(0.0f, 1.0f);
            std::vector<float> query(32);
            for (uint32_t q = 0; q < QUERIES_PER_THREAD; ++q) {
                for (auto& x : query) x = d(local_rng);
                auto results = index.search(query.data(), 10);
                total_results.fetch_add(results.size());
            }
        });
    }

    for (auto& th : threads) th.join();

    // Should have gotten results for every query
    REQUIRE(total_results.load() > 0);
    REQUIRE(total_results.load() == THREADS * QUERIES_PER_THREAD * 10);
}

TEST_CASE("Concurrent insert + search", "[stress]") {
    HNSWConfig cfg;
    cfg.dim = 16;
    cfg.M = 12;
    cfg.max_elements = 5000;
    HNSWIndex index(cfg);

    // Pre-populate with 500 vectors
    std::vector<std::vector<float>> vecs(3000, std::vector<float>(16));
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (auto& v : vecs) {
        for (auto& x : v) x = dist(rng);
    }
    for (uint32_t i = 0; i < 500; ++i) {
        index.insert(i, vecs[i].data());
    }

    // Run inserters and searchers concurrently
    std::atomic<bool> done{false};
    std::atomic<uint32_t> insert_count{500};
    std::atomic<uint32_t> search_count{0};

    // 2 inserter threads
    std::vector<std::thread> threads;
    for (int t = 0; t < 2; ++t) {
        threads.emplace_back([&, t]() {
            uint32_t start = 500 + t * 1000;
            uint32_t end = start + 1000;
            for (uint32_t i = start; i < end && i < 3000; ++i) {
                index.insert(i, vecs[i].data());
                insert_count.fetch_add(1);
            }
        });
    }

    // 4 searcher threads
    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([&, t]() {
            std::mt19937 local_rng(t * 77);
            std::normal_distribution<float> d(0.0f, 1.0f);
            std::vector<float> query(16);
            while (!done.load(std::memory_order_relaxed)) {
                for (auto& x : query) x = d(local_rng);
                auto results = index.search(query.data(), 5);
                search_count.fetch_add(1);
            }
        });
    }

    // Wait for inserters
    threads[0].join();
    threads[1].join();

    done.store(true, std::memory_order_release);

    // Wait for searchers
    for (size_t i = 2; i < threads.size(); ++i) {
        threads[i].join();
    }

    REQUIRE(index.size() >= 2000);
    REQUIRE(search_count.load() > 0);
}
