#include "pipeline/rag_pipeline.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <chrono>
#include <sstream>

namespace vortex {

using Clock = std::chrono::steady_clock;

// ============================================================================
// Pipeline stages
// ============================================================================

std::vector<float> RAGPipeline::stage_embed(const std::string& text) {
    return embedder_.embed(text);
}

std::vector<SearchResult> RAGPipeline::stage_search(
    const std::vector<float>& query_vec) {
    return index_.search(query_vec.data(), config_.top_k, config_.ef_search);
}

std::vector<SearchResult> RAGPipeline::stage_rerank(
    const std::string& query, std::vector<SearchResult> candidates) {

    if (!config_.enable_rerank || candidates.size() <= config_.top_n) {
        // No reranking — just truncate to top_n
        if (candidates.size() > config_.top_n) {
            candidates.resize(config_.top_n);
        }
        return candidates;
    }

    // Simple reranking heuristic: score by keyword overlap
    // (In production, this would call a cross-encoder model)
    for (auto& r : candidates) {
        float keyword_score = 0.0f;
        // Count query words that appear in the chunk
        std::istringstream iss(query);
        std::string word;
        while (iss >> word) {
            // Case-insensitive search
            std::string lower_word = word;
            std::transform(lower_word.begin(), lower_word.end(),
                         lower_word.begin(), ::tolower);
            std::string lower_text = r.text;
            std::transform(lower_text.begin(), lower_text.end(),
                         lower_text.begin(), ::tolower);
            if (lower_text.find(lower_word) != std::string::npos) {
                keyword_score += 1.0f;
            }
        }
        // Blend vector score with keyword score
        r.score = r.score * 0.7f + keyword_score * 0.3f;
    }

    // Sort by blended score descending
    std::sort(candidates.begin(), candidates.end(),
              [](const SearchResult& a, const SearchResult& b) {
                  return a.score > b.score;
              });

    if (candidates.size() > config_.top_n) {
        candidates.resize(config_.top_n);
    }
    return candidates;
}

std::string RAGPipeline::stage_augment(
    const std::string& query, const std::vector<SearchResult>& chunks) {

    // Build context from chunks
    std::ostringstream ctx;
    for (size_t i = 0; i < chunks.size(); ++i) {
        ctx << "[" << (i + 1) << "] " << chunks[i].text << "\n\n";
    }

    // Apply template
    std::string prompt = config_.prompt_template;

    // Replace {context}
    auto pos = prompt.find("{context}");
    if (pos != std::string::npos) {
        prompt.replace(pos, 9, ctx.str());
    }

    // Replace {query}
    pos = prompt.find("{query}");
    if (pos != std::string::npos) {
        prompt.replace(pos, 7, query);
    }

    return prompt;
}

// ============================================================================
// Full RAG query (standalone mode)
// ============================================================================

RAGResult RAGPipeline::query(const std::string& query_text, LLMCompleteFn llm_fn) {
    auto total_start = Clock::now();
    RAGResult result;

    Metrics::instance().queries_received.fetch_add(1, std::memory_order_relaxed);

    // Stage 1: Embed
    auto t0 = Clock::now();
    auto query_vec = stage_embed(query_text);
    auto t1 = Clock::now();
    result.embed_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

    // Stage 2: Search
    auto t2 = Clock::now();
    auto candidates = stage_search(query_vec);
    auto t3 = Clock::now();
    result.search_us = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();

    // Populate chunk texts from document processor
    for (auto& r : candidates) {
        r.text = doc_proc_.get_chunk_text(r.id);
    }

    // Stage 3: Rerank
    auto t4 = Clock::now();
    auto chunks = stage_rerank(query_text, std::move(candidates));
    auto t5 = Clock::now();
    result.rerank_us = std::chrono::duration_cast<std::chrono::microseconds>(t5 - t4).count();

    result.sources = chunks;

    // Stage 4: Augment
    auto t6 = Clock::now();
    result.augmented_prompt = stage_augment(query_text, chunks);
    auto t7 = Clock::now();
    result.augment_us = std::chrono::duration_cast<std::chrono::microseconds>(t7 - t6).count();

    // Stage 5: Generate
    auto t8 = Clock::now();
    result.answer = llm_fn(result.augmented_prompt);
    auto t9 = Clock::now();
    result.generate_us = std::chrono::duration_cast<std::chrono::microseconds>(t9 - t8).count();

    auto total_end = Clock::now();
    result.total_us = std::chrono::duration_cast<std::chrono::microseconds>(
        total_end - total_start).count();
    result.pipeline_overhead_us = result.total_us - result.embed_us
                                  - result.search_us - result.generate_us;

    Metrics::instance().queries_completed.fetch_add(1, std::memory_order_relaxed);
    Metrics::instance().total_search_us.fetch_add(result.search_us, std::memory_order_relaxed);
    Metrics::instance().total_embed_us.fetch_add(result.embed_us, std::memory_order_relaxed);
    Metrics::instance().total_pipeline_overhead_us.fetch_add(
        result.pipeline_overhead_us, std::memory_order_relaxed);

    spdlog::debug("RAG query: embed={}us search={}us rerank={}us augment={}us "
                  "generate={}us total={}us overhead={}us",
                  result.embed_us, result.search_us, result.rerank_us,
                  result.augment_us, result.generate_us, result.total_us,
                  result.pipeline_overhead_us);

    return result;
}

// ============================================================================
// Retrieval-only (E2E mode — agent handles generation)
// ============================================================================

RAGResult RAGPipeline::retrieve(const std::string& query_text) {
    auto total_start = Clock::now();
    RAGResult result;

    Metrics::instance().queries_received.fetch_add(1, std::memory_order_relaxed);

    // Stage 1: Embed
    auto t0 = Clock::now();
    auto query_vec = stage_embed(query_text);
    auto t1 = Clock::now();
    result.embed_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

    // Stage 2: Search
    auto t2 = Clock::now();
    auto candidates = stage_search(query_vec);
    auto t3 = Clock::now();
    result.search_us = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();

    // Populate chunk texts
    for (auto& r : candidates) {
        r.text = doc_proc_.get_chunk_text(r.id);
    }

    // Stage 3: Rerank
    auto t4 = Clock::now();
    auto chunks = stage_rerank(query_text, std::move(candidates));
    auto t5 = Clock::now();
    result.rerank_us = std::chrono::duration_cast<std::chrono::microseconds>(t5 - t4).count();

    result.sources = chunks;

    auto total_end = Clock::now();
    result.total_us = std::chrono::duration_cast<std::chrono::microseconds>(
        total_end - total_start).count();
    result.pipeline_overhead_us = result.total_us - result.embed_us - result.search_us;

    Metrics::instance().queries_completed.fetch_add(1, std::memory_order_relaxed);
    Metrics::instance().searches_completed.fetch_add(1, std::memory_order_relaxed);
    Metrics::instance().total_search_us.fetch_add(result.search_us, std::memory_order_relaxed);

    return result;
}

}  // namespace vortex
