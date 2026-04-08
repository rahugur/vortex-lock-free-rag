#pragma once
// OpenAI-compatible embedding API client.

#include "pipeline/embedder.h"
#include "utils/json.h"

#include <httplib.h>
#include <spdlog/spdlog.h>

#include <stdexcept>
#include <string>
#include <regex>

namespace vortex {

class APIEmbedder : public IEmbedder {
public:
    APIEmbedder(const std::string& api_base, const std::string& model,
                const std::string& api_key, uint32_t dim = 768)
        : model_(model), api_key_(api_key), dim_(dim) {
        parse_url(api_base);
    }

    std::vector<float> embed(const std::string& text) override {
        auto batch = embed_batch({text});
        if (batch.empty()) {
            throw std::runtime_error("Embedding API returned no results");
        }
        return batch[0];
    }

    std::vector<std::vector<float>> embed_batch(
        const std::vector<std::string>& texts) override;

    uint32_t dim() const override { return dim_; }

private:
    void parse_url(const std::string& url);

    std::string scheme_;
    std::string host_;
    int port_;
    std::string path_prefix_;
    std::string model_;
    std::string api_key_;
    uint32_t dim_;
};

}  // namespace vortex
