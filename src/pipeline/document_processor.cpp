#include "pipeline/document_processor.h"
#include "utils/metrics.h"

#include <spdlog/spdlog.h>

#include <chrono>

namespace vortex {

std::vector<Chunk> DocumentProcessor::chunk_text(
    uint32_t doc_id, const std::string& text, const ChunkConfig& config) {

    std::vector<Chunk> chunks;
    if (text.empty()) return chunks;

    // Approximate: 1 token ≈ 4 chars
    uint32_t char_size = config.chunk_size * 4;
    uint32_t char_overlap = config.chunk_overlap * 4;

    size_t pos = 0;
    uint32_t idx = 0;

    while (pos < text.size()) {
        size_t end = std::min(pos + char_size, text.size());

        // Try to find a separator near the end for clean breaks
        if (end < text.size()) {
            size_t sep_pos = text.rfind(config.separator, end);
            if (sep_pos != std::string::npos && sep_pos > pos + char_size / 2) {
                end = sep_pos + config.separator.size();
            } else {
                // Fall back to newline
                sep_pos = text.rfind('\n', end);
                if (sep_pos != std::string::npos && sep_pos > pos + char_size / 2) {
                    end = sep_pos + 1;
                }
                // Otherwise just cut at char_size
            }
        }

        chunks.push_back({text.substr(pos, end - pos), doc_id, idx++});

        // Advance with overlap
        if (end >= text.size()) break;
        pos = (end > char_overlap) ? end - char_overlap : end;
    }

    return chunks;
}

IndexResult DocumentProcessor::index_document(
    uint32_t doc_id, const std::string& text, const ChunkConfig& config) {

    auto start = std::chrono::steady_clock::now();

    // Chunk
    auto chunks = chunk_text(doc_id, text, config);
    if (chunks.empty()) return {1, 0, 0, 0.0};

    // Extract texts for batch embedding
    std::vector<std::string> texts;
    texts.reserve(chunks.size());
    for (auto& c : chunks) {
        texts.push_back(c.text);
    }

    // Embed
    auto embeddings = embedder_.embed_batch(texts);

    // Store chunk texts and insert vectors
    uint32_t inserted = 0;
    for (size_t i = 0; i < chunks.size(); ++i) {
        uint32_t id = next_id_.fetch_add(1, std::memory_order_relaxed);

        // Store chunk text
        {
            std::lock_guard<std::mutex> lock(chunk_mutex_);
            if (id >= chunk_texts_.size()) {
                chunk_texts_.resize(id + 1024);
            }
            chunk_texts_[id] = std::move(chunks[i].text);
        }

        // Insert into HNSW index
        index_.insert(id, embeddings[i].data());
        chunk_count_.fetch_add(1, std::memory_order_relaxed);
        ++inserted;
    }

    Metrics::instance().vectors_indexed.fetch_add(inserted, std::memory_order_relaxed);

    auto elapsed = std::chrono::steady_clock::now() - start;
    double ms = std::chrono::duration<double, std::milli>(elapsed).count();

    spdlog::debug("Indexed document {}: {} chunks in {:.1f}ms", doc_id, inserted, ms);

    return {1, static_cast<uint32_t>(chunks.size()), inserted, ms};
}

IndexResult DocumentProcessor::index_documents(
    const std::vector<Document>& docs, const ChunkConfig& config) {

    auto start = std::chrono::steady_clock::now();

    // Chunk all documents
    std::vector<Chunk> all_chunks;
    for (size_t i = 0; i < docs.size(); ++i) {
        auto chunks = chunk_text(static_cast<uint32_t>(i), docs[i].text, config);
        all_chunks.insert(all_chunks.end(),
                         std::make_move_iterator(chunks.begin()),
                         std::make_move_iterator(chunks.end()));
    }

    if (all_chunks.empty()) {
        return {static_cast<uint32_t>(docs.size()), 0, 0, 0.0};
    }

    // Batch embed all chunks
    std::vector<std::string> texts;
    texts.reserve(all_chunks.size());
    for (auto& c : all_chunks) {
        texts.push_back(c.text);
    }
    auto embeddings = embedder_.embed_batch(texts);

    // Insert all vectors
    uint32_t inserted = 0;
    for (size_t i = 0; i < all_chunks.size(); ++i) {
        uint32_t id = next_id_.fetch_add(1, std::memory_order_relaxed);

        {
            std::lock_guard<std::mutex> lock(chunk_mutex_);
            if (id >= chunk_texts_.size()) {
                chunk_texts_.resize(id + 1024);
            }
            chunk_texts_[id] = std::move(all_chunks[i].text);
        }

        index_.insert(id, embeddings[i].data());
        chunk_count_.fetch_add(1, std::memory_order_relaxed);
        ++inserted;
    }

    Metrics::instance().vectors_indexed.fetch_add(inserted, std::memory_order_relaxed);

    auto elapsed = std::chrono::steady_clock::now() - start;
    double ms = std::chrono::duration<double, std::milli>(elapsed).count();

    spdlog::info("Indexed {} documents: {} chunks in {:.1f}ms",
                 docs.size(), inserted, ms);

    return {static_cast<uint32_t>(docs.size()),
            static_cast<uint32_t>(all_chunks.size()), inserted, ms};
}

std::string DocumentProcessor::get_chunk_text(uint32_t internal_id) const {
    std::lock_guard<std::mutex> lock(chunk_mutex_);
    if (internal_id < chunk_texts_.size()) {
        return chunk_texts_[internal_id];
    }
    return "";
}

}  // namespace vortex
