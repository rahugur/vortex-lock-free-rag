#pragma once
// Lock-Free HNSW (Hierarchical Navigable Small World) Vector Index.
//
// Based on: Malkov & Yashunin, "Efficient and robust approximate nearest
// neighbor using Hierarchical Navigable Small World graphs" (2018).
//
// Concurrency model:
//   - Searches are fully lock-free (read atomics only, no mutex).
//   - Inserts use fine-grained per-node spinlocks (only lock neighbor nodes
//     being updated, not the whole graph).
//   - Concurrent search + insert is safe — a search may see a partially-
//     connected new node (slightly lower recall), but never corrupt data.
//
// KNOWN LIMITATION: std::vector storage can reallocate during resize, causing
// use-after-free if a concurrent search holds a pointer into the old buffer.
// Mitigation: call reserve(N) with your expected capacity before concurrent
// inserts. A proper fix would use paged/chunked storage or epoch-based
// reclamation — tracked as a future enhancement.
//
// Memory layout:
//   - Vectors stored in a flat, aligned buffer for SIMD distance computation.
//   - Per-layer adjacency stored as flat arrays of atomic<uint32_t>.
//   - Cache-line padding on hot atomics to prevent false sharing.

#include "index/distance.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <queue>
#include <random>
#include <string>
#include <vector>

namespace vortex {

/// Result of a nearest-neighbor search.
struct SearchResult {
    uint32_t id;        // Node ID
    float    distance;  // Distance to query
    std::string text;   // Associated text chunk (populated by pipeline)

    float score = 0.0f; // Similarity score (1 / (1 + distance) for L2)

    bool operator<(const SearchResult& o) const { return distance < o.distance; }
    bool operator>(const SearchResult& o) const { return distance > o.distance; }
};

/// HNSW index configuration.
struct HNSWConfig {
    uint32_t dim = 768;            // Vector dimensionality
    uint32_t M = 32;               // Max edges per node per layer
    uint32_t M_max0 = 64;          // Max edges at layer 0 (typically 2*M)
    uint32_t ef_construct = 200;   // Beam width during insert
    uint32_t ef_search = 200;      // Default beam width during search
    uint32_t max_elements = 0;     // Pre-allocate capacity (0 = dynamic)
    DistanceType distance_type = DistanceType::L2;
    uint32_t random_seed = 42;
};

class HNSWIndex {
public:
    explicit HNSWIndex(const HNSWConfig& config);
    ~HNSWIndex();

    // Non-copyable
    HNSWIndex(const HNSWIndex&) = delete;
    HNSWIndex& operator=(const HNSWIndex&) = delete;

    /// Insert a vector into the index. Thread-safe.
    /// @param id   External ID for this vector.
    /// @param data Pointer to dim floats.
    void insert(uint32_t id, const float* data);

    /// Search for the top-k nearest neighbors. Lock-free, fully concurrent.
    /// @param query   Pointer to dim floats.
    /// @param k       Number of results.
    /// @param ef      Search beam width (0 = use default from config).
    /// @return        Results sorted by distance (ascending).
    std::vector<SearchResult> search(const float* query, uint32_t k,
                                     uint32_t ef = 0) const;

    /// Number of vectors currently in the index.
    uint32_t size() const { return node_count_.load(std::memory_order_acquire); }

    /// Vector dimensionality.
    uint32_t dim() const { return config_.dim; }

    /// Get the raw vector data for a node.
    const float* get_vector(uint32_t internal_id) const;

    /// Save index to a file (binary format).
    void save(const std::string& path) const;

    /// Load index from a file.
    static std::unique_ptr<HNSWIndex> load(const std::string& path);

    /// Reserve capacity for n vectors (avoids reallocations).
    void reserve(uint32_t n);

private:
    // --- Node metadata ---
    struct NodeMeta {
        uint32_t external_id = 0;   // User-facing ID
        uint8_t  max_layer = 0;     // Highest layer this node appears on
        alignas(64) std::atomic<uint8_t> lock{0};  // Per-node spinlock for insert

        NodeMeta() = default;
        NodeMeta(const NodeMeta& o)
            : external_id(o.external_id), max_layer(o.max_layer) {
            lock.store(0, std::memory_order_relaxed);
        }
        NodeMeta& operator=(const NodeMeta& o) {
            external_id = o.external_id;
            max_layer = o.max_layer;
            lock.store(0, std::memory_order_relaxed);
            return *this;
        }
    };

    // --- Layer data ---
    // Neighbors stored as flat array: neighbors_[node * M_max + i]
    // Uses unique_ptr<atomic[]> since std::atomic is non-copyable/non-movable
    // and std::vector requires that for resize().
    struct LayerData {
        std::unique_ptr<std::atomic<uint32_t>[]> neighbors;
        std::unique_ptr<std::atomic<uint8_t>[]>  neighbor_counts;
        uint32_t M_max = 0;
        uint32_t capacity_ = 0;

        LayerData() = default;
        explicit LayerData(uint32_t m_max, uint32_t capacity);

        // Move-only (for std::vector<LayerData>)
        LayerData(LayerData&& o) noexcept
            : neighbors(std::move(o.neighbors)),
              neighbor_counts(std::move(o.neighbor_counts)),
              M_max(o.M_max), capacity_(o.capacity_) {}
        LayerData& operator=(LayerData&& o) noexcept {
            neighbors = std::move(o.neighbors);
            neighbor_counts = std::move(o.neighbor_counts);
            M_max = o.M_max;
            capacity_ = o.capacity_;
            return *this;
        }

        void ensure_capacity(uint32_t node_count);

        uint32_t get_neighbor(uint32_t node, uint32_t idx) const;
        uint8_t  get_neighbor_count(uint32_t node) const;
        void     set_neighbor(uint32_t node, uint32_t idx, uint32_t neighbor);
        void     set_neighbor_count(uint32_t node, uint8_t count);
    };

    // --- Internal helpers ---

    /// Randomly assign a layer to a new node (geometric distribution).
    uint8_t random_level();

    /// Greedy search from entry_point down to target_layer.
    /// Returns the closest node at target_layer.
    uint32_t greedy_search(const float* query, uint32_t entry,
                           uint8_t top_layer, uint8_t target_layer) const;

    /// Beam search at a single layer. Returns up to ef nearest candidates.
    /// This is the hot inner loop — fully lock-free.
    using MaxHeap = std::priority_queue<SearchResult>;
    using MinHeap = std::priority_queue<SearchResult, std::vector<SearchResult>,
                                        std::greater<SearchResult>>;

    MaxHeap search_layer(const float* query, uint32_t entry_node,
                         uint32_t ef, uint8_t layer) const;

    /// Select neighbors using the simple heuristic (closest M).
    std::vector<uint32_t> select_neighbors_simple(
        const float* query, const MaxHeap& candidates, uint32_t M) const;

    /// Select neighbors using the heuristic from the paper
    /// (considers diversity — avoids redundant connections).
    std::vector<uint32_t> select_neighbors_heuristic(
        const float* query, MaxHeap candidates, uint32_t M) const;

    /// Add bidirectional edge: node ↔ neighbor at the given layer.
    /// Uses per-node spinlock on the neighbor's adjacency list.
    void connect(uint32_t node, uint32_t neighbor, uint8_t layer);

    /// Lock a node's adjacency list (spinlock).
    void lock_node(uint32_t node);
    void unlock_node(uint32_t node);

    /// Compute distance between query and a stored vector.
    float dist(const float* query, uint32_t internal_id) const;

    /// Get pointer to vector storage for internal_id.
    float* vector_data(uint32_t internal_id);
    const float* vector_data(uint32_t internal_id) const;

    // --- Members ---
    HNSWConfig config_;
    DistanceFn dist_fn_;

    // Vector storage: flat buffer, aligned for SIMD.
    // Layout: [vec0_d0, vec0_d1, ..., vec0_d(dim-1), vec1_d0, ...]
    std::vector<float> vectors_;  // size = capacity * dim

    // Per-node metadata
    std::vector<NodeMeta> nodes_;

    // Per-layer graph structure. layers_[0] is the bottom (densest) layer.
    std::vector<LayerData> layers_;

    // Entry point and current max level
    alignas(64) std::atomic<uint32_t> entry_point_{UINT32_MAX};
    alignas(64) std::atomic<uint8_t>  max_level_{0};
    alignas(64) std::atomic<uint32_t> node_count_{0};

    // Capacity management
    uint32_t capacity_ = 0;
    std::mutex resize_mutex_;  // Only held during rare capacity growth

    // Random number generator (per-thread for insert)
    mutable std::mt19937 rng_;
    mutable std::mutex rng_mutex_;

    // Reverse multiplier for level generation: 1/ln(M)
    double level_mult_;

    static constexpr uint32_t INVALID = UINT32_MAX;
    static constexpr uint8_t MAX_LAYERS = 16;
};

}  // namespace vortex
