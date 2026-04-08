#pragma once
// Bridge between Forge's ToolRegistry and Vortex's RAG pipeline.
//
// Registers a "knowledge_search" tool that Forge agents can call.
// The tool does retrieval only (embed → search → rerank → return chunks).
// The agent handles generation via its ReAct loop — no double LLM call.
//
// This is the E2E glue: a Forge ReAct agent + Vortex retrieval, in-process,
// single binary, zero serialization overhead.

// Forge headers
#include <tools/tool_registry.h>
#include <utils/json.h>

// Vortex headers
#include "pipeline/rag_pipeline.h"

#include <sstream>

namespace vortex {

/// Register a knowledge_search tool in a Forge ToolRegistry.
///
/// The tool searches the Vortex index and returns relevant chunks.
/// The agent (not Vortex) handles LLM generation.
///
/// @param registry   Forge tool registry to register into.
/// @param pipeline   Vortex RAG pipeline (must outlive the registry).
inline void register_knowledge_search(
    forge::ToolRegistry& registry,
    RAGPipeline& pipeline)
{
    forge::ToolSpec spec;
    spec.name = "knowledge_search";
    spec.description =
        "Search the knowledge base for information relevant to a query. "
        "Returns the most relevant text chunks with relevance scores. "
        "Use this whenever you need factual information to answer a question. "
        "You can call this multiple times with different queries for multi-hop retrieval.";
    spec.parameters = forge::Json::parse(R"({
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query - be specific and descriptive"
            },
            "top_k": {
                "type": "integer",
                "description": "Number of results to return (default: 5)",
                "default": 5
            }
        },
        "required": ["query"]
    })");
    spec.timeout = std::chrono::milliseconds(5000);
    spec.allow_parallel = true;   // Multiple searches can run concurrently
    spec.idempotent = true;       // Same query = same results

    registry.register_tool(std::move(spec),
        [&pipeline](const std::string& args_json) -> std::string {
            auto j = forge::Json::parse(args_json);
            std::string query = j.value("query", "");

            if (query.empty()) {
                throw std::runtime_error("Missing 'query' parameter");
            }

            // Retrieval only — no LLM generation
            auto result = pipeline.retrieve(query);

            // Truncate to requested top_k (default 5)
            uint32_t top_k = j.value("top_k", 5u);
            if (result.sources.size() > top_k) {
                result.sources.resize(top_k);
            }

            // Format results for the agent
            std::ostringstream out;
            out << "Found " << result.sources.size() << " relevant passages:\n\n";
            for (size_t i = 0; i < result.sources.size(); ++i) {
                out << "--- Result " << (i + 1)
                    << " (score: " << result.sources[i].score << ") ---\n"
                    << result.sources[i].text << "\n\n";
            }
            out << "[Search took " << result.search_us << "us, "
                << "embed " << result.embed_us << "us, "
                << "total " << result.total_us << "us]";

            return out.str();
        });
}

}  // namespace vortex
