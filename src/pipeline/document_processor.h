#pragma once
// Document chunking and ingestion pipeline.

#include "index/hnsw_index.h"
#include "pipeline/embedder.h"

#include <core/thread_pool.h>
#include <core/future.h>

#include <atomic>
#include <cstdint>
#include <string>
#include <vector>

namespace vortex {

struct ChunkConfig {
    uint32_t chunk_size = 512;       // Max tokens per chunk (approx chars/4)
    uint32_t chunk_overlap = 64;     // Overlapping chars between chunks
    std::string separator = "\n\n";  // Preferred split boundary
};

struct Chunk {
    std::string text;
    uint32_t doc_id;
    uint32_t chunk_idx;
};

struct IndexResult {
    uint32_t documents_indexed;
    uint32_t chunks_created;
    uint32_t vectors_inserted;
    double   time_ms;
};

struct Document {
    std::string id;    // External document ID
    std::string text;  // Full document text
};

class DocumentProcessor {
public:
    DocumentProcessor(HNSWIndex& index, IEmbedder& embedder,
                      forge::ThreadPool& /*pool*/)
        : index_(index), embedder_(embedder) {}

    /// Chunk and index a single document.
    IndexResult index_document(uint32_t doc_id, const std::string& text,
                               const ChunkConfig& config = {});

    /// Chunk and index multiple documents (parallel embedding).
    IndexResult index_documents(const std::vector<Document>& docs,
                                const ChunkConfig& config = {});

    /// Get the text chunk associated with an internal vector ID.
    std::string get_chunk_text(uint32_t internal_id) const;

    /// Total chunks stored.
    uint32_t chunk_count() const {
        return chunk_count_.load(std::memory_order_acquire);
    }

private:
    /// Split text into chunks at separator boundaries.
    std::vector<Chunk> chunk_text(uint32_t doc_id, const std::string& text,
                                  const ChunkConfig& config);

    HNSWIndex& index_;
    IEmbedder& embedder_;

    // Chunk text storage (indexed by internal vector ID)
    std::vector<std::string> chunk_texts_;
    mutable std::mutex chunk_mutex_;
    std::atomic<uint32_t> chunk_count_{0};
    std::atomic<uint32_t> next_id_{0};
};

}  // namespace vortex
