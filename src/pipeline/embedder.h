#pragma once
// Embedding interface and implementations.
//
// IEmbedder: abstract interface for converting text → vector.
// APIEmbedder: calls an OpenAI-compatible embedding API.
// MockEmbedder: returns deterministic vectors for testing.

#include <cstdint>
#include <string>
#include <vector>

namespace vortex {

/// Abstract embedding interface.
class IEmbedder {
public:
    virtual ~IEmbedder() = default;

    /// Embed a single text string into a vector.
    virtual std::vector<float> embed(const std::string& text) = 0;

    /// Batch embed multiple texts. Default: sequential calls.
    virtual std::vector<std::vector<float>> embed_batch(
        const std::vector<std::string>& texts) {
        std::vector<std::vector<float>> results;
        results.reserve(texts.size());
        for (auto& t : texts) {
            results.push_back(embed(t));
        }
        return results;
    }

    /// Vector dimensionality.
    virtual uint32_t dim() const = 0;
};

/// Mock embedder for testing — returns deterministic hash-based vectors.
class MockEmbedder : public IEmbedder {
public:
    explicit MockEmbedder(uint32_t dim = 768) : dim_(dim) {}

    std::vector<float> embed(const std::string& text) override {
        std::vector<float> vec(dim_);
        // Simple hash-based deterministic embedding
        uint64_t hash = 5381;
        for (char c : text) {
            hash = ((hash << 5) + hash) + static_cast<uint64_t>(c);
        }
        for (uint32_t i = 0; i < dim_; ++i) {
            hash = hash * 6364136223846793005ULL + 1442695040888963407ULL;
            vec[i] = static_cast<float>(static_cast<int64_t>(hash) % 1000) / 1000.0f;
        }
        // Normalize
        float norm = 0.0f;
        for (float v : vec) norm += v * v;
        norm = std::sqrt(norm);
        if (norm > 0.0f) {
            for (float& v : vec) v /= norm;
        }
        return vec;
    }

    uint32_t dim() const override { return dim_; }

private:
    uint32_t dim_;
};

}  // namespace vortex
