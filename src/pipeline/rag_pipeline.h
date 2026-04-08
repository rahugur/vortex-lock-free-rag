#pragma once
// RAG Pipeline Controller.
//
// 5-stage pipeline: embed → search → rerank → augment → generate
// Each stage is dispatched to the ThreadPool via Forge's lock-free Future.
//
// In standalone mode, the pipeline calls the LLM for generation.
// In E2E mode (Forge agent), the generate step is skipped — the agent
// handles generation via its ReAct loop.

#include "index/hnsw_index.h"
#include "pipeline/document_processor.h"
#include "pipeline/embedder.h"
#include "utils/json.h"
#include "utils/metrics.h"

#include <core/future.h>
#include <core/semaphore.h>
#include <core/thread_pool.h>

#include <chrono>
#include <functional>
#include <string>
#include <vector>

namespace vortex {

/// Result of a RAG query.
struct RAGResult {
    std::string answer;                      // Final generated answer
    std::vector<SearchResult> sources;       // Retrieved chunks with scores
    std::string augmented_prompt;            // The prompt sent to LLM

    // Timing breakdown (microseconds)
    uint64_t embed_us = 0;
    uint64_t search_us = 0;
    uint64_t rerank_us = 0;
    uint64_t augment_us = 0;
    uint64_t generate_us = 0;
    uint64_t total_us = 0;
    uint64_t pipeline_overhead_us = 0;       // total - embed - search - generate

    int prompt_tokens = 0;
    int completion_tokens = 0;
};

/// LLM completion function type (for standalone mode).
/// Takes a prompt, returns the completion text.
using LLMCompleteFn = std::function<std::string(const std::string& prompt)>;

/// RAG pipeline configuration.
struct RAGPipelineConfig {
    uint32_t top_k = 10;           // Candidates from vector search
    uint32_t top_n = 3;            // Final chunks after reranking
    uint32_t ef_search = 64;       // HNSW beam width
    bool enable_rerank = false;
    uint32_t max_context_tokens = 2048;

    std::string prompt_template =
        "Answer the question based on the following context.\n\n"
        "Context:\n{context}\n\n"
        "Question: {query}\n\n"
        "Answer:";

    std::string system_message =
        "You are a helpful assistant that answers questions based on the "
        "provided context. Be concise and accurate.";
};

class RAGPipeline {
public:
    RAGPipeline(HNSWIndex& index, IEmbedder& embedder,
                DocumentProcessor& doc_proc, forge::ThreadPool& /*pool*/,
                const RAGPipelineConfig& config = {})
        : index_(index), embedder_(embedder), doc_proc_(doc_proc),
          config_(config) {}

    /// Full RAG query (standalone mode). Calls LLM for generation.
    RAGResult query(const std::string& query_text, LLMCompleteFn llm_fn);

    /// Retrieval-only query (E2E mode). Returns chunks without LLM generation.
    /// This is what the knowledge_search tool calls.
    RAGResult retrieve(const std::string& query_text);

    /// Set pipeline config.
    void set_config(const RAGPipelineConfig& config) { config_ = config; }
    const RAGPipelineConfig& config() const { return config_; }

private:
    // Pipeline stages
    std::vector<float> stage_embed(const std::string& text);
    std::vector<SearchResult> stage_search(const std::vector<float>& query_vec);
    std::vector<SearchResult> stage_rerank(const std::string& query,
                                           std::vector<SearchResult> candidates);
    std::string stage_augment(const std::string& query,
                              const std::vector<SearchResult>& chunks);

    HNSWIndex& index_;
    IEmbedder& embedder_;
    DocumentProcessor& doc_proc_;
    RAGPipelineConfig config_;
};

}  // namespace vortex
