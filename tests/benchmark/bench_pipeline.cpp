// End-to-end RAG pipeline benchmark.
// Measures pipeline overhead (everything except LLM generation).

#include "index/hnsw_index.h"
#include "pipeline/document_processor.h"
#include "pipeline/embedder.h"
#include "pipeline/rag_pipeline.h"

#include <core/thread_pool.h>

#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace vortex;
using Clock = std::chrono::steady_clock;

int main(int argc, char* argv[]) {
    uint32_t index_size = 10000;
    uint32_t n_queries = 500;
    uint32_t concurrent = 1;
    uint32_t dim = 128;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--index-size" && i + 1 < argc) index_size = std::stoul(argv[++i]);
        if (arg == "--queries" && i + 1 < argc) n_queries = std::stoul(argv[++i]);
        if (arg == "--concurrent" && i + 1 < argc) concurrent = std::stoul(argv[++i]);
        if (arg == "--dim" && i + 1 < argc) dim = std::stoul(argv[++i]);
    }

    std::cout << "=== RAG Pipeline Benchmark ===" << std::endl;
    std::cout << "Index: " << index_size << " vectors, " << dim << "-dim" << std::endl;
    std::cout << "Queries: " << n_queries << ", concurrent: " << concurrent << std::endl;
    std::cout << std::endl;

    // Setup
    forge::ThreadPool pool(0);
    HNSWConfig cfg;
    cfg.dim = dim;
    cfg.max_elements = index_size;
    HNSWIndex index(cfg);
    MockEmbedder embedder(dim);
    DocumentProcessor doc_proc(index, embedder, pool);

    // Build index with mock documents
    std::cout << "Building index..." << std::flush;
    auto build_start = Clock::now();
    for (uint32_t i = 0; i < index_size; ++i) {
        std::string text = "Document " + std::to_string(i) + " with some content about topic "
                          + std::to_string(i % 100) + ". This is filler text.";
        doc_proc.index_document(i, text);
    }
    auto build_end = Clock::now();
    double build_ms = std::chrono::duration<double, std::milli>(build_end - build_start).count();
    std::cout << " done (" << std::fixed << std::setprecision(0) << build_ms << "ms)" << std::endl;

    // Pipeline config
    RAGPipelineConfig rag_cfg;
    rag_cfg.top_k = 10;
    rag_cfg.top_n = 3;
    RAGPipeline pipeline(index, embedder, doc_proc, pool, rag_cfg);

    // --- Retrieval-only benchmark (E2E mode) ---
    std::cout << "\n--- Retrieval-Only (E2E mode) ---" << std::endl;

    if (concurrent == 1) {
        // Single-threaded
        uint64_t total_search_us = 0;
        uint64_t total_embed_us = 0;
        uint64_t total_overhead_us = 0;
        uint64_t total_us = 0;

        auto start = Clock::now();
        for (uint32_t q = 0; q < n_queries; ++q) {
            std::string query = "Query about topic " + std::to_string(q % 100);
            auto result = pipeline.retrieve(query);
            total_search_us += result.search_us;
            total_embed_us += result.embed_us;
            total_overhead_us += result.pipeline_overhead_us;
            total_us += result.total_us;
        }
        auto end = Clock::now();
        double wall_ms = std::chrono::duration<double, std::milli>(end - start).count();
        double qps = n_queries / (wall_ms / 1000.0);

        std::cout << "  Wall time: " << std::setprecision(1) << wall_ms << " ms" << std::endl;
        std::cout << "  QPS: " << std::setprecision(0) << qps << std::endl;
        std::cout << "  Avg embed: " << (total_embed_us / n_queries) << " us" << std::endl;
        std::cout << "  Avg search: " << (total_search_us / n_queries) << " us" << std::endl;
        std::cout << "  Avg overhead: " << (total_overhead_us / n_queries) << " us" << std::endl;
        std::cout << "  Avg total: " << (total_us / n_queries) << " us" << std::endl;
    } else {
        // Multi-threaded
        std::atomic<uint64_t> completed{0};
        std::atomic<uint64_t> total_us_atomic{0};

        auto start = Clock::now();
        std::vector<std::thread> threads;
        for (uint32_t t = 0; t < concurrent; ++t) {
            threads.emplace_back([&, t]() {
                uint32_t per_thread = n_queries / concurrent;
                for (uint32_t q = 0; q < per_thread; ++q) {
                    std::string query = "Query " + std::to_string(t * per_thread + q);
                    auto result = pipeline.retrieve(query);
                    total_us_atomic.fetch_add(result.total_us);
                    completed.fetch_add(1);
                }
            });
        }
        for (auto& th : threads) th.join();
        auto end = Clock::now();

        double wall_ms = std::chrono::duration<double, std::milli>(end - start).count();
        double qps = completed.load() / (wall_ms / 1000.0);
        double avg_us = total_us_atomic.load() / completed.load();

        std::cout << "  Wall time: " << std::setprecision(1) << wall_ms << " ms" << std::endl;
        std::cout << "  Completed: " << completed.load() << " queries" << std::endl;
        std::cout << "  QPS: " << std::setprecision(0) << qps << std::endl;
        std::cout << "  Avg latency: " << std::setprecision(0) << avg_us << " us" << std::endl;
    }

    // --- Full RAG benchmark (with mock LLM) ---
    std::cout << "\n--- Full RAG (with mock LLM, 50ms latency) ---" << std::endl;
    {
        auto mock_llm = [](const std::string& prompt) -> std::string {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            return "Mock answer based on the context.";
        };

        auto start = Clock::now();
        for (uint32_t q = 0; q < std::min(n_queries, 50u); ++q) {
            std::string query = "Full query " + std::to_string(q);
            auto result = pipeline.query(query, mock_llm);
        }
        auto end = Clock::now();
        double wall_ms = std::chrono::duration<double, std::milli>(end - start).count();
        uint32_t actual = std::min(n_queries, 50u);
        double qps = actual / (wall_ms / 1000.0);
        double per_query = wall_ms / actual;

        std::cout << "  " << actual << " queries in " << std::setprecision(0)
                  << wall_ms << " ms" << std::endl;
        std::cout << "  QPS: " << std::setprecision(1) << qps << std::endl;
        std::cout << "  Per query: " << std::setprecision(1) << per_query << " ms "
                  << "(~50ms is LLM, rest is pipeline)" << std::endl;
    }

    return 0;
}
