// HNSW micro-benchmark: insert and search throughput at various scales.

#include "index/hnsw_index.h"
#include "index/distance.h"

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <unordered_set>
#include <vector>

using namespace vortex;
using Clock = std::chrono::steady_clock;

// Generate random unit vectors
static std::vector<std::vector<float>> gen_vectors(
    uint32_t n, uint32_t dim, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<std::vector<float>> vecs(n, std::vector<float>(dim));
    for (auto& v : vecs) {
        float norm = 0.0f;
        for (auto& x : v) { x = dist(rng); norm += x * x; }
        norm = std::sqrt(norm);
        for (auto& x : v) x /= norm;
    }
    return vecs;
}

// Brute-force kNN
static std::vector<uint32_t> brute_knn(
    const std::vector<std::vector<float>>& data,
    const float* query, uint32_t k, uint32_t dim) {
    std::vector<std::pair<float, uint32_t>> dists;
    dists.reserve(data.size());
    for (uint32_t i = 0; i < data.size(); ++i) {
        dists.push_back({l2_scalar(query, data[i].data(), dim), i});
    }
    std::partial_sort(dists.begin(), dists.begin() + k, dists.end());
    std::vector<uint32_t> result;
    for (uint32_t i = 0; i < k; ++i) result.push_back(dists[i].second);
    return result;
}

int main(int argc, char* argv[]) {
    uint32_t n = 100000;
    uint32_t dim = 128;
    uint32_t n_queries = 1000;
    uint32_t k = 10;

    // Parse args
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--size" && i + 1 < argc) n = std::stoul(argv[++i]);
        if (arg == "--dim" && i + 1 < argc) dim = std::stoul(argv[++i]);
        if (arg == "--queries" && i + 1 < argc) n_queries = std::stoul(argv[++i]);
    }

    std::cout << "=== HNSW Benchmark ===" << std::endl;
    std::cout << "Index size: " << n << " vectors, " << dim << "-dim" << std::endl;
    std::cout << std::endl;

    // Generate data
    std::cout << "Generating vectors..." << std::flush;
    auto data = gen_vectors(n, dim, 42);
    auto queries = gen_vectors(n_queries, dim, 999);
    std::cout << " done." << std::endl;

    // Build index
    HNSWConfig cfg;
    cfg.dim = dim;
    cfg.M = 32;
    cfg.M_max0 = 64;
    cfg.ef_construct = 200;
    cfg.ef_search = 200;
    cfg.max_elements = n;
    HNSWIndex index(cfg);

    std::cout << "\n--- Build ---" << std::endl;
    auto build_start = Clock::now();
    for (uint32_t i = 0; i < n; ++i) {
        index.insert(i, data[i].data());
        if ((i + 1) % (n / 10) == 0) {
            std::cout << "  " << (i + 1) << "/" << n << std::endl;
        }
    }
    auto build_end = Clock::now();
    double build_ms = std::chrono::duration<double, std::milli>(build_end - build_start).count();
    std::cout << "  Build time: " << std::fixed << std::setprecision(1)
              << build_ms << " ms" << std::endl;
    std::cout << "  Per-insert: " << std::setprecision(1)
              << (build_ms * 1000.0 / n) << " us" << std::endl;

    // Search throughput and recall at multiple ef values
    uint32_t recall_queries = std::min(n_queries, 100u);

    // Pre-compute brute-force results for recall measurement
    std::vector<std::unordered_set<uint32_t>> bf_sets(recall_queries);
    for (uint32_t q = 0; q < recall_queries; ++q) {
        auto bf_results = brute_knn(data, queries[q].data(), k, dim);
        bf_sets[q] = std::unordered_set<uint32_t>(bf_results.begin(), bf_results.end());
    }

    std::cout << "\n--- Search Throughput & Recall@" << k << " ---" << std::endl;
    for (uint32_t ef : {64u, 128u, 200u, 400u}) {
        // Throughput
        auto search_start = Clock::now();
        for (uint32_t q = 0; q < n_queries; ++q) {
            index.search(queries[q].data(), k, ef);
        }
        auto search_end = Clock::now();
        double search_ms = std::chrono::duration<double, std::milli>(
            search_end - search_start).count();
        double per_query_us = (search_ms * 1000.0) / n_queries;
        double qps = n_queries / (search_ms / 1000.0);

        // Recall
        double total_recall = 0.0;
        for (uint32_t q = 0; q < recall_queries; ++q) {
            auto hnsw_results = index.search(queries[q].data(), k, ef);
            uint32_t hits = 0;
            for (auto& r : hnsw_results) {
                if (bf_sets[q].count(r.id)) ++hits;
            }
            total_recall += static_cast<double>(hits) / k;
        }
        double avg_recall = total_recall / recall_queries;

        std::cout << "  ef=" << std::setw(4) << ef
                  << "  recall=" << std::setprecision(4) << avg_recall
                  << "  latency=" << std::setprecision(0) << per_query_us << "us"
                  << "  QPS=" << std::setprecision(0) << qps << std::endl;
    }

    // Memory estimate
    size_t vec_bytes = static_cast<size_t>(n) * dim * sizeof(float);
    size_t meta_bytes = n * (sizeof(uint32_t) + sizeof(uint8_t) + 64);  // NodeMeta with padding
    size_t graph_bytes = n * cfg.M_max0 * sizeof(uint32_t);  // layer 0 approximation
    size_t total_bytes = vec_bytes + meta_bytes + graph_bytes;
    std::cout << "\n--- Memory ---" << std::endl;
    std::cout << "  Vectors: " << (vec_bytes / 1024 / 1024) << " MB" << std::endl;
    std::cout << "  Graph (est): " << (graph_bytes / 1024 / 1024) << " MB" << std::endl;
    std::cout << "  Total (est): " << (total_bytes / 1024 / 1024) << " MB" << std::endl;
    std::cout << "  Per vector: " << (total_bytes / n) << " bytes" << std::endl;

    // Distance benchmark
    std::cout << "\n--- Distance Function ---" << std::endl;
    auto dist_fn = resolve(DistanceType::L2);
    auto t0 = Clock::now();
    float dummy = 0.0f;
    for (uint32_t i = 0; i < 100000; ++i) {
        dummy += dist_fn(data[0].data(), data[i % n].data(), dim);
    }
    auto t1 = Clock::now();
    double dist_ns = std::chrono::duration<double, std::nano>(t1 - t0).count() / 100000;
    std::cout << "  " << dim << "-dim L2: " << std::setprecision(0) << dist_ns << " ns/call"
              << " (dummy=" << dummy << ")" << std::endl;

    return 0;
}
