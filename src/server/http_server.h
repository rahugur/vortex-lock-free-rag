#pragma once
// Vortex HTTP Server.
//
// REST API for RAG queries, index management, and metrics.
// SSE streaming for real-time token delivery.

#include "index/hnsw_index.h"
#include "pipeline/document_processor.h"
#include "pipeline/embedder.h"
#include "pipeline/rag_pipeline.h"
#include "scheduler/batch_scheduler.h"
#include "utils/config.h"
#include "utils/metrics.h"

#include <httplib.h>

#include <memory>
#include <string>

namespace vortex {

class HttpServer {
public:
    HttpServer(HNSWIndex& index, IEmbedder& embedder,
               DocumentProcessor& doc_proc, RAGPipeline& pipeline,
               BatchScheduler* scheduler,  // nullable — standalone mode
               const Config& config);

    /// Start the server (blocks).
    void start();

    /// Stop the server.
    void stop();

private:
    // Route handlers
    void handle_query(const httplib::Request& req, httplib::Response& res);
    void handle_search(const httplib::Request& req, httplib::Response& res);
    void handle_index(const httplib::Request& req, httplib::Response& res);
    void handle_index_file(const httplib::Request& req, httplib::Response& res);
    void handle_metrics(const httplib::Request& req, httplib::Response& res);
    void handle_health(const httplib::Request& req, httplib::Response& res);

    HNSWIndex& index_;
    DocumentProcessor& doc_proc_;
    RAGPipeline& pipeline_;
    BatchScheduler* scheduler_;
    Config config_;

    httplib::Server server_;
};

}  // namespace vortex
