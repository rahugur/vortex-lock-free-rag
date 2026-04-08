#include "pipeline/api_embedder.h"

#include <spdlog/spdlog.h>

namespace vortex {

void APIEmbedder::parse_url(const std::string& url) {
    // Parse "https://api.openai.com/v1" into scheme, host, port, path
    std::regex url_regex(R"(^(https?)://([^/:]+)(?::(\d+))?(/.*)?)");
    std::smatch match;
    if (!std::regex_match(url, match, url_regex)) {
        throw std::runtime_error("Invalid API base URL: " + url);
    }
    scheme_ = match[1];
    host_ = match[2];
    port_ = match[3].matched ? std::stoi(match[3]) : (scheme_ == "https" ? 443 : 80);
    path_prefix_ = match[4].matched ? std::string(match[4]) : "";
    // Remove trailing slash
    if (!path_prefix_.empty() && path_prefix_.back() == '/') {
        path_prefix_.pop_back();
    }
}

std::vector<std::vector<float>> APIEmbedder::embed_batch(
    const std::vector<std::string>& texts) {

    // Build request body
    Json body;
    body["model"] = model_;
    if (texts.size() == 1) {
        body["input"] = texts[0];
    } else {
        body["input"] = texts;
    }

    std::string body_str = body.dump();
    std::string path = path_prefix_ + "/embeddings";

    // Make HTTP request
    std::unique_ptr<httplib::Client> client;
    if (scheme_ == "https") {
        client = std::make_unique<httplib::Client>(
            scheme_ + "://" + host_ + ":" + std::to_string(port_));
    } else {
        client = std::make_unique<httplib::Client>(host_, port_);
    }
    client->set_connection_timeout(10);
    client->set_read_timeout(30);

    httplib::Headers headers = {
        {"Content-Type", "application/json"},
        {"Authorization", "Bearer " + api_key_}
    };

    auto res = client->Post(path, headers, body_str, "application/json");

    if (!res) {
        throw std::runtime_error("Embedding API request failed: connection error");
    }

    if (res->status != 200) {
        // Log truncated body to avoid leaking sensitive data
        std::string truncated = res->body.substr(0, std::min<size_t>(res->body.size(), 200));
        spdlog::error("Embedding API error {}: {}", res->status, truncated);
        throw std::runtime_error("Embedding API error: " + std::to_string(res->status));
    }

    // Parse response
    auto json_resp = Json::parse(res->body);
    auto& data = json_resp["data"];

    std::vector<std::vector<float>> results;
    results.reserve(data.size());
    for (auto& item : data) {
        auto& embedding = item["embedding"];
        std::vector<float> vec;
        vec.reserve(embedding.size());
        for (auto& v : embedding) {
            vec.push_back(v.get<float>());
        }
        results.push_back(std::move(vec));
    }

    return results;
}

}  // namespace vortex
