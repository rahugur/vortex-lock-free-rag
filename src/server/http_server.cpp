#include "server/http_server.h"
#include "utils/json.h"

#include <spdlog/spdlog.h>

#include <chrono>

namespace vortex {

HttpServer::HttpServer(HNSWIndex& index, IEmbedder& /*embedder*/,
                       DocumentProcessor& doc_proc, RAGPipeline& pipeline,
                       BatchScheduler* scheduler, const Config& config)
    : index_(index), doc_proc_(doc_proc),
      pipeline_(pipeline), scheduler_(scheduler), config_(config) {

    // Limit request body to 64MB to prevent OOM from oversized payloads
    server_.set_payload_max_length(64 * 1024 * 1024);

    // Routes
    server_.Post("/v1/query",
        [this](const httplib::Request& req, httplib::Response& res) {
            handle_query(req, res);
        });

    server_.Post("/v1/search",
        [this](const httplib::Request& req, httplib::Response& res) {
            handle_search(req, res);
        });

    server_.Post("/v1/index",
        [this](const httplib::Request& req, httplib::Response& res) {
            handle_index(req, res);
        });

    server_.Post("/v1/index/file",
        [this](const httplib::Request& req, httplib::Response& res) {
            handle_index_file(req, res);
        });

    server_.Get("/v1/metrics",
        [this](const httplib::Request& req, httplib::Response& res) {
            handle_metrics(req, res);
        });

    server_.Get("/health",
        [this](const httplib::Request& req, httplib::Response& res) {
            handle_health(req, res);
        });
}

void HttpServer::start() {
    spdlog::info("Vortex server starting on {}:{}", config_.host, config_.port);
    server_.listen(config_.host, config_.port);
}

void HttpServer::stop() {
    server_.stop();
}

// ============================================================================
// Route handlers
// ============================================================================

void HttpServer::handle_query(const httplib::Request& req, httplib::Response& res) {
    try {
        auto body = Json::parse(req.body);
        std::string query = body.value("query", "");
        if (query.empty()) {
            res.status = 400;
            res.set_content(R"({"error":"Missing 'query' field"})", "application/json");
            return;
        }

        // Use batch scheduler if available, otherwise direct pipeline call
        LLMCompleteFn llm_fn;
        if (scheduler_) {
            llm_fn = [this](const std::string& prompt) -> std::string {
                auto future = scheduler_->submit(prompt);
                return future.get();
            };
        } else {
            // No LLM configured — return retrieval results only
            auto result = pipeline_.retrieve(query);
            Json response;
            response["answer"] = "";
            response["sources"] = Json::array();
            for (auto& s : result.sources) {
                response["sources"].push_back({
                    {"id", s.id},
                    {"score", s.score},
                    {"text", s.text}
                });
            }
            response["metrics"] = {
                {"embed_us", result.embed_us},
                {"search_us", result.search_us},
                {"rerank_us", result.rerank_us},
                {"total_us", result.total_us},
                {"pipeline_overhead_us", result.pipeline_overhead_us}
            };
            res.set_content(response.dump(), "application/json");
            return;
        }

        auto result = pipeline_.query(query, llm_fn);

        Json response;
        response["answer"] = result.answer;
        response["sources"] = Json::array();
        for (auto& s : result.sources) {
            response["sources"].push_back({
                {"id", s.id},
                {"score", s.score},
                {"text", s.text}
            });
        }
        response["metrics"] = {
            {"embed_us", result.embed_us},
            {"search_us", result.search_us},
            {"rerank_us", result.rerank_us},
            {"augment_us", result.augment_us},
            {"generate_us", result.generate_us},
            {"total_us", result.total_us},
            {"pipeline_overhead_us", result.pipeline_overhead_us}
        };

        res.set_content(response.dump(), "application/json");

    } catch (const Json::parse_error&) {
        res.status = 400;
        res.set_content(R"({"error":"Invalid JSON"})", "application/json");
    } catch (const std::exception& e) {
        spdlog::error("Query error: {}", e.what());
        res.status = 500;
        res.set_content(R"({"error":"Internal server error"})", "application/json");
    }
}

void HttpServer::handle_search(const httplib::Request& req, httplib::Response& res) {
    try {
        auto body = Json::parse(req.body);
        std::string query = body.value("query", "");

        if (query.empty()) {
            res.status = 400;
            res.set_content(R"({"error":"Missing 'query' field"})", "application/json");
            return;
        }

        auto result = pipeline_.retrieve(query);

        Json response;
        response["results"] = Json::array();
        for (auto& s : result.sources) {
            response["results"].push_back({
                {"id", s.id},
                {"score", s.score},
                {"distance", s.distance},
                {"text", s.text}
            });
        }
        response["metrics"] = {
            {"embed_us", result.embed_us},
            {"search_us", result.search_us},
            {"total_us", result.total_us}
        };

        res.set_content(response.dump(), "application/json");

    } catch (const Json::parse_error&) {
        res.status = 400;
        res.set_content(R"({"error":"Invalid JSON"})", "application/json");
    } catch (const std::exception& e) {
        spdlog::error("Search error: {}", e.what());
        res.status = 500;
        res.set_content(R"({"error":"Internal server error"})", "application/json");
    }
}

void HttpServer::handle_index(const httplib::Request& req, httplib::Response& res) {
    try {
        auto body = Json::parse(req.body);
        auto& vectors = body["vectors"];

        uint32_t count = 0;
        for (auto& v : vectors) {
            uint32_t id = v["id"];
            auto& data = v["data"];
            if (data.size() != index_.dim()) {
                res.status = 400;
                Json err;
                err["error"] = "Vector dimension mismatch: expected "
                    + std::to_string(index_.dim()) + ", got "
                    + std::to_string(data.size());
                res.set_content(err.dump(), "application/json");
                return;
            }
            std::vector<float> vec;
            vec.reserve(data.size());
            for (auto& f : data) {
                vec.push_back(f.get<float>());
            }
            index_.insert(id, vec.data());
            ++count;
        }

        Json response;
        response["indexed"] = count;
        res.status = 201;
        res.set_content(response.dump(), "application/json");

    } catch (const Json::parse_error&) {
        res.status = 400;
        res.set_content(R"({"error":"Invalid JSON"})", "application/json");
    } catch (const std::exception& e) {
        spdlog::error("Index error: {}", e.what());
        res.status = 500;
        res.set_content(R"({"error":"Internal server error"})", "application/json");
    }
}

void HttpServer::handle_index_file(const httplib::Request& req, httplib::Response& res) {
    try {
        auto body = Json::parse(req.body);
        auto& docs = body["documents"];

        ChunkConfig chunk_config;
        if (body.contains("chunk_size")) chunk_config.chunk_size = body["chunk_size"];
        if (body.contains("chunk_overlap")) chunk_config.chunk_overlap = body["chunk_overlap"];

        std::vector<Document> documents;
        for (auto& d : docs) {
            documents.push_back({d.value("id", ""), d.value("text", "")});
        }

        auto result = doc_proc_.index_documents(documents, chunk_config);

        Json response;
        response["indexed"] = result.documents_indexed;
        response["chunks"] = result.chunks_created;
        response["vectors"] = result.vectors_inserted;
        response["time_ms"] = result.time_ms;
        res.status = 201;
        res.set_content(response.dump(), "application/json");

    } catch (const Json::parse_error&) {
        res.status = 400;
        res.set_content(R"({"error":"Invalid JSON"})", "application/json");
    } catch (const std::exception& e) {
        spdlog::error("Index file error: {}", e.what());
        res.status = 500;
        res.set_content(R"({"error":"Internal server error"})", "application/json");
    }
}

void HttpServer::handle_metrics(const httplib::Request& /*req*/, httplib::Response& res) {
    auto& m = Metrics::instance();
    Json response;
    response["index"] = {
        {"vectors_indexed", m.vectors_indexed.load()},
        {"searches_completed", m.searches_completed.load()}
    };
    response["pipeline"] = {
        {"queries_received", m.queries_received.load()},
        {"queries_completed", m.queries_completed.load()},
        {"queries_failed", m.queries_failed.load()},
        {"avg_search_us", m.searches_completed > 0
            ? m.total_search_us.load() / m.searches_completed.load() : 0},
        {"avg_embed_us", m.queries_completed > 0
            ? m.total_embed_us.load() / m.queries_completed.load() : 0},
        {"avg_pipeline_overhead_us", m.queries_completed > 0
            ? m.total_pipeline_overhead_us.load() / m.queries_completed.load() : 0}
    };
    response["scheduler"] = {
        {"batches_formed", m.batches_formed.load()},
        {"requests_batched", m.requests_batched.load()},
        {"prefix_cache_hits", m.prefix_cache_hits.load()},
        {"prefix_cache_misses", m.prefix_cache_misses.load()}
    };
    response["tokens"] = {
        {"total_prompt", m.total_prompt_tokens.load()},
        {"total_completion", m.total_completion_tokens.load()},
        {"total_embed", m.total_embed_tokens.load()}
    };
    res.set_content(response.dump(2), "application/json");
}

void HttpServer::handle_health(const httplib::Request& /*req*/, httplib::Response& res) {
    Json response;
    response["status"] = "ok";
    response["index_size"] = index_.size();
    response["chunks"] = doc_proc_.chunk_count();
    res.set_content(response.dump(), "application/json");
}

}  // namespace vortex
