#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "index/hnsw_index.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <unordered_set>
#include <vector>

using namespace vortex;

// Helper: generate random unit vectors
static std::vector<std::vector<float>> random_vectors(
    uint32_t n, uint32_t dim, uint32_t seed = 42) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<std::vector<float>> vecs(n, std::vector<float>(dim));
    for (auto& v : vecs) {
        float norm = 0.0f;
        for (auto& x : v) {
            x = dist(rng);
            norm += x * x;
        }
        norm = std::sqrt(norm);
        for (auto& x : v) x /= norm;
    }
    return vecs;
}

// Brute-force kNN for recall comparison
static std::vector<uint32_t> brute_force_knn(
    const std::vector<std::vector<float>>& data,
    const float* query, uint32_t k, uint32_t dim) {

    std::vector<std::pair<float, uint32_t>> dists;
    dists.reserve(data.size());
    for (uint32_t i = 0; i < data.size(); ++i) {
        float d = l2_scalar(query, data[i].data(), dim);
        dists.push_back({d, i});
    }
    std::partial_sort(dists.begin(), dists.begin() + k, dists.end());
    std::vector<uint32_t> result;
    for (uint32_t i = 0; i < k; ++i) result.push_back(dists[i].second);
    return result;
}

TEST_CASE("HNSW empty index returns empty", "[hnsw]") {
    HNSWConfig cfg;
    cfg.dim = 4;
    HNSWIndex idx(cfg);

    auto results = idx.search(std::vector<float>(4, 1.0f).data(), 5);
    REQUIRE(results.empty());
}

TEST_CASE("HNSW single element", "[hnsw]") {
    HNSWConfig cfg;
    cfg.dim = 4;
    HNSWIndex idx(cfg);

    std::vector<float> v = {1.0f, 0.0f, 0.0f, 0.0f};
    idx.insert(100, v.data());

    REQUIRE(idx.size() == 1);

    auto results = idx.search(v.data(), 1);
    REQUIRE(results.size() == 1);
    REQUIRE(results[0].id == 100);
    REQUIRE(results[0].distance < 1e-6f);
}

TEST_CASE("HNSW exact match in small set", "[hnsw]") {
    HNSWConfig cfg;
    cfg.dim = 8;
    cfg.M = 8;
    cfg.ef_construct = 50;
    HNSWIndex idx(cfg);

    auto vecs = random_vectors(100, 8);
    for (uint32_t i = 0; i < 100; ++i) {
        idx.insert(i, vecs[i].data());
    }

    REQUIRE(idx.size() == 100);

    // Search for each vector — should find itself as nearest
    for (uint32_t i = 0; i < 100; ++i) {
        auto results = idx.search(vecs[i].data(), 1, 100);
        REQUIRE(!results.empty());
        REQUIRE(results[0].id == i);
    }
}

TEST_CASE("HNSW recall on random data", "[hnsw]") {
    HNSWConfig cfg;
    cfg.dim = 32;
    cfg.M = 16;
    cfg.ef_construct = 200;
    cfg.ef_search = 64;
    HNSWIndex idx(cfg);

    uint32_t n = 5000;
    uint32_t k = 10;
    uint32_t n_queries = 50;

    auto data = random_vectors(n, 32);
    for (uint32_t i = 0; i < n; ++i) {
        idx.insert(i, data[i].data());
    }

    auto queries = random_vectors(n_queries, 32, 999);

    // Measure recall@10
    double total_recall = 0.0;
    for (uint32_t q = 0; q < n_queries; ++q) {
        auto hnsw_results = idx.search(queries[q].data(), k);
        auto bf_results = brute_force_knn(data, queries[q].data(), k, 32);

        std::unordered_set<uint32_t> bf_set(bf_results.begin(), bf_results.end());
        uint32_t hits = 0;
        for (auto& r : hnsw_results) {
            if (bf_set.count(r.id)) ++hits;
        }
        total_recall += static_cast<double>(hits) / k;
    }

    double avg_recall = total_recall / n_queries;
    REQUIRE(avg_recall > 0.90);  // Target: >90% recall
    INFO("Average recall@10: " << avg_recall);
}

TEST_CASE("HNSW results sorted by distance", "[hnsw]") {
    HNSWConfig cfg;
    cfg.dim = 16;
    HNSWIndex idx(cfg);

    auto vecs = random_vectors(500, 16);
    for (uint32_t i = 0; i < 500; ++i) {
        idx.insert(i, vecs[i].data());
    }

    auto query = random_vectors(1, 16, 123)[0];
    auto results = idx.search(query.data(), 20);

    for (size_t i = 1; i < results.size(); ++i) {
        REQUIRE(results[i].distance >= results[i - 1].distance);
    }
}

TEST_CASE("HNSW save and load", "[hnsw]") {
    HNSWConfig cfg;
    cfg.dim = 8;
    cfg.M = 8;
    HNSWIndex idx(cfg);

    auto vecs = random_vectors(200, 8);
    for (uint32_t i = 0; i < 200; ++i) {
        idx.insert(i, vecs[i].data());
    }

    // Save
    std::string path = "/tmp/test_hnsw_index.vrtx";
    idx.save(path);

    // Load
    auto loaded = HNSWIndex::load(path);
    REQUIRE(loaded->size() == 200);

    // Search should produce same results
    auto query = random_vectors(1, 8, 77)[0];
    auto results_orig = idx.search(query.data(), 5);
    auto results_loaded = loaded->search(query.data(), 5);

    REQUIRE(results_orig.size() == results_loaded.size());
    for (size_t i = 0; i < results_orig.size(); ++i) {
        REQUIRE(results_orig[i].id == results_loaded[i].id);
    }

    // Cleanup
    std::remove(path.c_str());
}

TEST_CASE("HNSW handles duplicate IDs", "[hnsw]") {
    HNSWConfig cfg;
    cfg.dim = 4;
    HNSWIndex idx(cfg);

    // Insert two vectors with same external ID but different data
    std::vector<float> v1 = {1, 0, 0, 0};
    std::vector<float> v2 = {0, 1, 0, 0};
    idx.insert(42, v1.data());
    idx.insert(42, v2.data());

    REQUIRE(idx.size() == 2);  // Both stored (external IDs can repeat)
}

TEST_CASE("HNSW large ef_search improves recall", "[hnsw]") {
    HNSWConfig cfg;
    cfg.dim = 32;
    cfg.M = 8;  // Deliberately low M to make recall harder
    cfg.ef_construct = 100;
    HNSWIndex idx(cfg);

    auto data = random_vectors(2000, 32);
    for (uint32_t i = 0; i < 2000; ++i) {
        idx.insert(i, data[i].data());
    }

    auto query = random_vectors(1, 32, 555)[0];
    auto bf = brute_force_knn(data, query.data(), 10, 32);
    std::unordered_set<uint32_t> bf_set(bf.begin(), bf.end());

    // Low ef
    auto results_low = idx.search(query.data(), 10, 16);
    uint32_t hits_low = 0;
    for (auto& r : results_low) if (bf_set.count(r.id)) ++hits_low;

    // High ef
    auto results_high = idx.search(query.data(), 10, 256);
    uint32_t hits_high = 0;
    for (auto& r : results_high) if (bf_set.count(r.id)) ++hits_high;

    REQUIRE(hits_high >= hits_low);  // Higher ef should not decrease recall
}
