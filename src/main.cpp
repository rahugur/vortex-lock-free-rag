// Vortex — Lock-Free RAG Inference Engine
//
// Standalone server mode:
//   ./vortex-server --serve -k $OPENAI_API_KEY
//
// CLI query mode:
//   ./vortex-server -p "What is HNSW?" --index-dir ./docs/ -k $API_KEY
//
// Index-only mode:
//   ./vortex-server --build-index --data ./docs/ -k $API_KEY -o index.vrtx

#include "index/hnsw_index.h"
#include "pipeline/api_embedder.h"
#include "pipeline/document_processor.h"
#include "pipeline/embedder.h"
#include "pipeline/rag_pipeline.h"
#include "scheduler/batch_scheduler.h"
#include "server/http_server.h"
#include "utils/config.h"

#include <core/thread_pool.h>

#include <spdlog/spdlog.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

using namespace vortex;

static void print_usage() {
    std::cerr
        << "Usage: vortex-server [options]\n"
        << "\n"
        << "Modes:\n"
        << "  --serve                 Start HTTP server\n"
        << "  -p, --prompt <text>     Single query mode\n"
        << "  --build-index           Index documents and save\n"
        << "\n"
        << "Options:\n"
        << "  -c, --config <path>     Config file (JSON)\n"
        << "  -k, --api-key <key>     API key for LLM/embedding\n"
        << "  --embed-url <url>       Embedding API base URL\n"
        << "  --llm-url <url>         LLM API base URL\n"
        << "  -m, --model <name>      LLM model name\n"
        << "  --embed-model <name>    Embedding model name\n"
        << "  --index-dir <path>      Directory of text files to index\n"
        << "  --index-file <path>     Load/save HNSW index file\n"
        << "  -v, --verbose           Verbose logging\n"
        << "  -h, --help              Show this help\n";
}

int main(int argc, char* argv[]) {
    Config config;
    std::string prompt;
    std::string index_dir;
    std::string index_file;
    bool serve_mode = false;
    bool build_index = false;
    bool verbose = false;

    // Parse args
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--serve") {
            serve_mode = true;
        } else if (arg == "--build-index") {
            build_index = true;
        } else if ((arg == "-p" || arg == "--prompt") && i + 1 < argc) {
            prompt = argv[++i];
        } else if ((arg == "-c" || arg == "--config") && i + 1 < argc) {
            config = Config::from_file(argv[++i]);
        } else if ((arg == "-k" || arg == "--api-key") && i + 1 < argc) {
            config.api_key = argv[++i];
        } else if (arg == "--embed-url" && i + 1 < argc) {
            config.embed_api_base = argv[++i];
        } else if (arg == "--llm-url" && i + 1 < argc) {
            config.llm_api_base = argv[++i];
        } else if ((arg == "-m" || arg == "--model") && i + 1 < argc) {
            config.llm_model = argv[++i];
        } else if (arg == "--embed-model" && i + 1 < argc) {
            config.embed_model = argv[++i];
        } else if (arg == "--index-dir" && i + 1 < argc) {
            index_dir = argv[++i];
        } else if (arg == "--index-file" && i + 1 < argc) {
            index_file = argv[++i];
        } else if (arg == "-v" || arg == "--verbose") {
            verbose = true;
        } else if (arg == "-h" || arg == "--help") {
            print_usage();
            return 0;
        }
    }

    // Check for API key from environment
    if (config.api_key.empty()) {
        const char* env_key = std::getenv("OPENAI_API_KEY");
        if (env_key) config.api_key = env_key;
    }

    // Set up logging
    spdlog::set_level(verbose ? spdlog::level::debug : spdlog::level::info);

    // Create thread pool (shared by all components — reuses Forge's primitives)
    forge::ThreadPool pool(config.num_threads);
    spdlog::info("Thread pool: {} workers", pool.num_threads());

    // Create HNSW index
    HNSWConfig hnsw_config;
    hnsw_config.dim = config.embed_dim;
    hnsw_config.M = config.hnsw_m;
    hnsw_config.ef_construct = config.hnsw_ef_construct;
    hnsw_config.ef_search = config.hnsw_ef_search;

    std::unique_ptr<HNSWIndex> index;
    if (!index_file.empty()) {
        try {
            index = HNSWIndex::load(index_file);
            spdlog::info("Loaded index from {}: {} vectors", index_file, index->size());
        } catch (...) {
            spdlog::info("No existing index at {}, creating new", index_file);
            index = std::make_unique<HNSWIndex>(hnsw_config);
        }
    } else {
        index = std::make_unique<HNSWIndex>(hnsw_config);
    }

    // Create embedder
    std::unique_ptr<IEmbedder> embedder;
    if (!config.api_key.empty()) {
        embedder = std::make_unique<APIEmbedder>(
            config.embed_api_base, config.embed_model,
            config.api_key, config.embed_dim);
    } else {
        spdlog::warn("No API key — using mock embedder");
        embedder = std::make_unique<MockEmbedder>(config.embed_dim);
    }

    // Create document processor
    DocumentProcessor doc_proc(*index, *embedder, pool);

    // Index documents if requested
    if (!index_dir.empty()) {
        spdlog::info("Indexing documents from {}", index_dir);
        // Read all .txt files from directory
        // (Simple implementation — production would use filesystem::directory_iterator)
        spdlog::info("Use --index-file to load pre-built indices, "
                     "or POST to /v1/index/file to index documents via API");
    }

    // Create RAG pipeline
    RAGPipelineConfig rag_config;
    rag_config.top_k = config.top_k;
    rag_config.top_n = config.top_n;
    rag_config.ef_search = config.hnsw_ef_search;
    rag_config.enable_rerank = config.enable_rerank;
    rag_config.max_context_tokens = config.max_context_tokens;

    RAGPipeline pipeline(*index, *embedder, doc_proc, pool, rag_config);

    // Handle modes
    if (serve_mode) {
        // Create batch scheduler (optional — needs API key)
        std::unique_ptr<BatchScheduler> scheduler;
        if (!config.api_key.empty()) {
            LLMBackendFn backend = [&](const std::string& prompt) -> std::string {
                // Simple OpenAI-compatible completion call
                // In production, this would use the throttled LLM client
                return "[LLM generation placeholder — configure LLM API]";
            };
            BatchSchedulerConfig sched_config;
            sched_config.max_concurrent_calls = config.max_concurrent_llm_calls;
            sched_config.rate_limit = config.llm_rate_limit;
            sched_config.rate_burst = config.llm_rate_burst;
            scheduler = std::make_unique<BatchScheduler>(pool, backend, sched_config);
        }

        HttpServer server(*index, *embedder, doc_proc, pipeline,
                         scheduler.get(), config);
        server.start();

    } else if (!prompt.empty()) {
        // Single query mode
        if (index->size() == 0) {
            std::cerr << "Error: Index is empty. Index documents first.\n";
            return 1;
        }

        auto result = pipeline.retrieve(prompt);
        std::cout << "Query: " << prompt << "\n\n";
        std::cout << "Results (" << result.sources.size() << " chunks):\n";
        for (size_t i = 0; i < result.sources.size(); ++i) {
            auto& s = result.sources[i];
            std::cout << "\n--- [" << (i + 1) << "] score=" << s.score
                      << " ---\n" << s.text << "\n";
        }
        std::cout << "\nMetrics: embed=" << result.embed_us << "us"
                  << " search=" << result.search_us << "us"
                  << " total=" << result.total_us << "us"
                  << " overhead=" << result.pipeline_overhead_us << "us\n";

    } else if (build_index) {
        if (index_file.empty()) {
            std::cerr << "Error: --index-file required with --build-index\n";
            return 1;
        }
        index->save(index_file);
        spdlog::info("Index saved to {}", index_file);

    } else {
        print_usage();
        return 1;
    }

    // Save index on exit if path specified
    if (!index_file.empty() && !build_index) {
        index->save(index_file);
    }

    return 0;
}
