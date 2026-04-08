#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "index/hnsw_index.h"
#include "pipeline/document_processor.h"
#include "pipeline/embedder.h"
#include "pipeline/rag_pipeline.h"

#include <core/thread_pool.h>

using namespace vortex;

TEST_CASE("Document processor - basic chunking", "[pipeline]") {
    HNSWConfig cfg;
    cfg.dim = 64;
    HNSWIndex index(cfg);
    MockEmbedder embedder(64);
    forge::ThreadPool pool(2);
    DocumentProcessor proc(index, embedder, pool);

    std::string text =
        "This is paragraph one. It has some content.\n\n"
        "This is paragraph two. It also has content.\n\n"
        "This is paragraph three. More content here.";

    auto result = proc.index_document(0, text);
    REQUIRE(result.documents_indexed == 1);
    REQUIRE(result.chunks_created > 0);
    REQUIRE(result.vectors_inserted > 0);
    REQUIRE(index.size() > 0);
}

TEST_CASE("RAG pipeline - retrieve returns results", "[pipeline]") {
    HNSWConfig cfg;
    cfg.dim = 64;
    HNSWIndex index(cfg);
    MockEmbedder embedder(64);
    forge::ThreadPool pool(2);
    DocumentProcessor doc_proc(index, embedder, pool);

    // Index some documents
    doc_proc.index_document(0, "HNSW is a graph-based approximate nearest neighbor algorithm.");
    doc_proc.index_document(1, "Lock-free queues use atomic operations for thread safety.");
    doc_proc.index_document(2, "Vector databases store high-dimensional embeddings for search.");

    REQUIRE(index.size() >= 3);

    // Query
    RAGPipelineConfig rag_cfg;
    rag_cfg.top_k = 5;
    rag_cfg.top_n = 2;
    RAGPipeline pipeline(index, embedder, doc_proc, pool, rag_cfg);

    auto result = pipeline.retrieve("How does HNSW work?");
    REQUIRE(!result.sources.empty());
    REQUIRE(result.total_us >= 0);  // timing may be sub-microsecond in tests
}

TEST_CASE("RAG pipeline - full query with mock LLM", "[pipeline]") {
    HNSWConfig cfg;
    cfg.dim = 64;
    HNSWIndex index(cfg);
    MockEmbedder embedder(64);
    forge::ThreadPool pool(2);
    DocumentProcessor doc_proc(index, embedder, pool);

    doc_proc.index_document(0, "The capital of France is Paris.");

    RAGPipeline pipeline(index, embedder, doc_proc, pool);

    auto result = pipeline.query("What is the capital of France?",
        [](const std::string& prompt) -> std::string {
            return "Based on the context, the capital of France is Paris.";
        });

    REQUIRE(!result.answer.empty());
    REQUIRE(!result.sources.empty());
    REQUIRE(result.total_us >= 0);  // timing may be sub-microsecond in tests
}

TEST_CASE("Document processor - batch indexing", "[pipeline]") {
    HNSWConfig cfg;
    cfg.dim = 64;
    HNSWIndex index(cfg);
    MockEmbedder embedder(64);
    forge::ThreadPool pool(2);
    DocumentProcessor proc(index, embedder, pool);

    std::vector<Document> docs = {
        {"doc1", "First document about machine learning."},
        {"doc2", "Second document about databases."},
        {"doc3", "Third document about networking."},
    };

    auto result = proc.index_documents(docs);
    REQUIRE(result.documents_indexed == 3);
    REQUIRE(result.vectors_inserted >= 3);
}

TEST_CASE("RAG pipeline - empty index returns empty results", "[pipeline]") {
    HNSWConfig cfg;
    cfg.dim = 64;
    HNSWIndex index(cfg);
    MockEmbedder embedder(64);
    forge::ThreadPool pool(2);
    DocumentProcessor doc_proc(index, embedder, pool);

    RAGPipeline pipeline(index, embedder, doc_proc, pool);
    auto result = pipeline.retrieve("anything");
    REQUIRE(result.sources.empty());
}
