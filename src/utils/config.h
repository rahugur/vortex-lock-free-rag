#pragma once
// Vortex runtime configuration.

#include "utils/json.h"

#include <string>

namespace vortex {

struct Config {
    // Thread pool
    unsigned num_threads = 0;  // 0 = auto-detect

    // HNSW index defaults
    uint32_t hnsw_m = 32;             // edges per node
    uint32_t hnsw_ef_construct = 200; // beam width during insert
    uint32_t hnsw_ef_search = 200;    // beam width during search

    // RAG pipeline
    uint32_t top_k = 10;              // candidates from vector search
    uint32_t top_n = 3;               // final chunks after reranking
    bool enable_rerank = false;
    uint32_t max_context_tokens = 2048;
    uint32_t chunk_size = 512;
    uint32_t chunk_overlap = 64;

    // Batch scheduler
    uint32_t max_batch_size = 32;
    uint32_t max_wait_us = 5000;        // 5ms max wait
    uint32_t max_concurrent_llm_calls = 8;
    double   llm_rate_limit = 60.0;
    double   llm_rate_burst = 10.0;
    uint32_t max_concurrent_embed_calls = 16;

    // Server
    std::string host = "127.0.0.1";
    int port = 8081;

    // LLM / Embedding
    std::string llm_api_base = "https://api.openai.com/v1";
    std::string llm_model = "gpt-4o-mini";
    std::string embed_api_base = "https://api.openai.com/v1";
    std::string embed_model = "text-embedding-3-small";
    uint32_t embed_dim = 768;
    std::string api_key;

    // Logging
    std::string log_level = "info";

    static Config from_json(const Json& j) {
        Config c;
        if (j.contains("num_threads")) c.num_threads = j["num_threads"];
        if (j.contains("hnsw_m")) c.hnsw_m = j["hnsw_m"];
        if (j.contains("hnsw_ef_construct")) c.hnsw_ef_construct = j["hnsw_ef_construct"];
        if (j.contains("hnsw_ef_search")) c.hnsw_ef_search = j["hnsw_ef_search"];
        if (j.contains("top_k")) c.top_k = j["top_k"];
        if (j.contains("top_n")) c.top_n = j["top_n"];
        if (j.contains("enable_rerank")) c.enable_rerank = j["enable_rerank"];
        if (j.contains("max_context_tokens")) c.max_context_tokens = j["max_context_tokens"];
        if (j.contains("chunk_size")) c.chunk_size = j["chunk_size"];
        if (j.contains("chunk_overlap")) c.chunk_overlap = j["chunk_overlap"];
        if (j.contains("max_batch_size")) c.max_batch_size = j["max_batch_size"];
        if (j.contains("max_wait_us")) c.max_wait_us = j["max_wait_us"];
        if (j.contains("max_concurrent_llm_calls")) c.max_concurrent_llm_calls = j["max_concurrent_llm_calls"];
        if (j.contains("llm_rate_limit")) c.llm_rate_limit = j["llm_rate_limit"];
        if (j.contains("llm_rate_burst")) c.llm_rate_burst = j["llm_rate_burst"];
        if (j.contains("max_concurrent_embed_calls")) c.max_concurrent_embed_calls = j["max_concurrent_embed_calls"];
        if (j.contains("host")) c.host = j["host"];
        if (j.contains("port")) c.port = j["port"];
        if (j.contains("llm_api_base")) c.llm_api_base = j["llm_api_base"];
        if (j.contains("llm_model")) c.llm_model = j["llm_model"];
        if (j.contains("embed_api_base")) c.embed_api_base = j["embed_api_base"];
        if (j.contains("embed_model")) c.embed_model = j["embed_model"];
        if (j.contains("embed_dim")) c.embed_dim = j["embed_dim"];
        if (j.contains("api_key")) c.api_key = j["api_key"];
        if (j.contains("log_level")) c.log_level = j["log_level"];
        return c;
    }

    static Config from_file(const std::string& path) {
        return from_json(json_util::parse_file(path));
    }
};

}  // namespace vortex
