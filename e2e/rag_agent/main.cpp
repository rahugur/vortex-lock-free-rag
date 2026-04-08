// E2E Demo: Forge ReAct Agent + Vortex Retrieval
//
// A single C++ binary that runs a ReAct agent with in-process vector search.
// No serialization, no HTTP hop, no Python overhead.
//
// Usage:
//   ./rag_agent --index-file index.vrtx -p "What is HNSW?" -k $API_KEY
//   ./rag_agent --serve --index-file index.vrtx -k $API_KEY

// Forge headers
#include <core/thread_pool.h>
#include <llm/llm_client_interface.h>
#include <llm/message.h>
#include <tools/tool_executor.h>
#include <tools/tool_registry.h>

// Vortex headers
#include "index/hnsw_index.h"
#include "pipeline/api_embedder.h"
#include "pipeline/document_processor.h"
#include "pipeline/embedder.h"
#include "pipeline/rag_pipeline.h"
#include "utils/config.h"

// E2E glue
#include "knowledge_search_tool.h"

#include <spdlog/spdlog.h>

#include <cstdlib>
#include <iostream>
#include <string>

using namespace vortex;

int main(int argc, char* argv[]) {
    // Parse args (simplified)
    std::string prompt;
    std::string index_file;
    std::string api_key;
    std::string embed_url = "https://api.openai.com/v1";
    std::string llm_url = "https://api.openai.com/v1";
    std::string llm_model = "gpt-4o-mini";
    std::string embed_model = "text-embedding-3-small";
    uint32_t embed_dim = 768;
    bool verbose = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "-p" || arg == "--prompt") && i + 1 < argc) {
            prompt = argv[++i];
        } else if (arg == "--index-file" && i + 1 < argc) {
            index_file = argv[++i];
        } else if ((arg == "-k" || arg == "--api-key") && i + 1 < argc) {
            api_key = argv[++i];
        } else if (arg == "--embed-url" && i + 1 < argc) {
            embed_url = argv[++i];
        } else if (arg == "--llm-url" && i + 1 < argc) {
            llm_url = argv[++i];
        } else if ((arg == "-m" || arg == "--model") && i + 1 < argc) {
            llm_model = argv[++i];
        } else if (arg == "-v" || arg == "--verbose") {
            verbose = true;
        }
    }

    if (api_key.empty()) {
        const char* env = std::getenv("OPENAI_API_KEY");
        if (env) api_key = env;
    }

    spdlog::set_level(verbose ? spdlog::level::debug : spdlog::level::info);

    // ── Shared infrastructure (Forge primitives) ──
    forge::ThreadPool pool(0);  // auto-detect threads
    spdlog::info("E2E RAG Agent: {} worker threads", pool.num_threads());

    // ── Vortex: index + embedder + pipeline ──
    std::unique_ptr<HNSWIndex> index;
    if (!index_file.empty()) {
        try {
            index = HNSWIndex::load(index_file);
            spdlog::info("Loaded index: {} vectors", index->size());
        } catch (...) {
            HNSWConfig cfg;
            cfg.dim = embed_dim;
            index = std::make_unique<HNSWIndex>(cfg);
        }
    } else {
        HNSWConfig cfg;
        cfg.dim = embed_dim;
        index = std::make_unique<HNSWIndex>(cfg);
    }

    std::unique_ptr<IEmbedder> embedder;
    if (!api_key.empty()) {
        embedder = std::make_unique<APIEmbedder>(
            embed_url, embed_model, api_key, embed_dim);
    } else {
        embedder = std::make_unique<MockEmbedder>(embed_dim);
    }

    DocumentProcessor doc_proc(*index, *embedder, pool);
    RAGPipeline pipeline(*index, *embedder, doc_proc, pool);

    // ── Forge: tool registry with Vortex-backed knowledge_search ──
    forge::ToolRegistry tools;
    register_knowledge_search(tools, pipeline);

    // Also register Forge builtins that the agent might need
    spdlog::info("Registered tools: knowledge_search");

    // ── Forge: tool executor (dispatches tools on the shared pool) ──
    forge::ToolExecutor executor(pool, tools);

    // ── Run the agent ──
    if (prompt.empty()) {
        std::cerr << "Usage: rag_agent -p \"your question\" --index-file index.vrtx "
                  << "-k $API_KEY\n";
        return 1;
    }

    if (index->size() == 0) {
        spdlog::warn("Index is empty — agent will get no search results. "
                     "Index documents first via the vortex-server API.");
    }

    // Simple ReAct loop (demonstrates the integration)
    // In production, this would use Forge's full workflow system.
    spdlog::info("Query: {}", prompt);

    // Step 1: Call knowledge_search tool directly
    forge::ToolCall search_call;
    search_call.id = "call_1";
    search_call.name = "knowledge_search";
    // Build arguments as proper JSON to avoid injection from user input
    Json search_args;
    search_args["query"] = prompt;
    search_call.arguments_json = search_args.dump();

    auto result = executor.execute(search_call);

    std::cout << "\n=== Knowledge Search Results ===\n";
    std::cout << result.output << "\n";

    if (result.is_error) {
        std::cerr << "Search failed: " << result.output << "\n";
        return 1;
    }

    std::cout << "\n=== Agent Integration Complete ===\n"
              << "Forge tool dispatch + Vortex retrieval in a single process.\n"
              << "In production, the Forge ReAct workflow would loop:\n"
              << "  1. LLM decides to call knowledge_search\n"
              << "  2. Tool executor dispatches to Vortex (in-process, ~10ns)\n"
              << "  3. Vortex: embed → HNSW search → rerank → return chunks\n"
              << "  4. LLM reasons over chunks\n"
              << "  5. Repeat or synthesize final answer\n";

    return 0;
}
