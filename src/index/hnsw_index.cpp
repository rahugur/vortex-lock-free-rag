// HNSW Index Implementation.
//
// Key design decisions:
//   1. Vectors stored in flat aligned buffer — cache-friendly SIMD access.
//   2. Neighbor lists use atomics — searches never lock.
//   3. Per-node spinlocks for insert — only locks the nodes being connected.
//   4. Geometric random level assignment — upper layers are exponentially sparser.

#include "index/hnsw_index.h"
#include "index/distance.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <unordered_set>

namespace vortex {

// ============================================================================
// LayerData
// ============================================================================

HNSWIndex::LayerData::LayerData(uint32_t m_max, uint32_t capacity)
    : M_max(m_max), capacity_(capacity) {
    size_t n_neighbors = static_cast<size_t>(capacity) * m_max;
    neighbors = std::make_unique<std::atomic<uint32_t>[]>(n_neighbors);
    neighbor_counts = std::make_unique<std::atomic<uint8_t>[]>(capacity);
    for (size_t i = 0; i < n_neighbors; ++i) {
        neighbors[i].store(UINT32_MAX, std::memory_order_relaxed);
    }
    for (size_t i = 0; i < capacity; ++i) {
        neighbor_counts[i].store(0, std::memory_order_relaxed);
    }
}

void HNSWIndex::LayerData::ensure_capacity(uint32_t node_count) {
    if (node_count <= capacity_) return;

    size_t old_cap = capacity_;
    size_t new_cap = std::max<size_t>(node_count, old_cap * 2);

    // Allocate new arrays
    auto new_neighbors = std::make_unique<std::atomic<uint32_t>[]>(new_cap * M_max);
    auto new_counts = std::make_unique<std::atomic<uint8_t>[]>(new_cap);

    // Copy old data
    for (size_t i = 0; i < old_cap * M_max; ++i) {
        new_neighbors[i].store(neighbors[i].load(std::memory_order_relaxed),
                               std::memory_order_relaxed);
    }
    for (size_t i = 0; i < old_cap; ++i) {
        new_counts[i].store(neighbor_counts[i].load(std::memory_order_relaxed),
                            std::memory_order_relaxed);
    }

    // Initialize new entries
    for (size_t i = old_cap * M_max; i < new_cap * M_max; ++i) {
        new_neighbors[i].store(UINT32_MAX, std::memory_order_relaxed);
    }
    for (size_t i = old_cap; i < new_cap; ++i) {
        new_counts[i].store(0, std::memory_order_relaxed);
    }

    neighbors = std::move(new_neighbors);
    neighbor_counts = std::move(new_counts);
    capacity_ = static_cast<uint32_t>(new_cap);
}

uint32_t HNSWIndex::LayerData::get_neighbor(uint32_t node, uint32_t idx) const {
    return neighbors[static_cast<size_t>(node) * M_max + idx]
        .load(std::memory_order_acquire);
}

uint8_t HNSWIndex::LayerData::get_neighbor_count(uint32_t node) const {
    return neighbor_counts[node].load(std::memory_order_acquire);
}

void HNSWIndex::LayerData::set_neighbor(uint32_t node, uint32_t idx, uint32_t neighbor) {
    neighbors[static_cast<size_t>(node) * M_max + idx]
        .store(neighbor, std::memory_order_release);
}

void HNSWIndex::LayerData::set_neighbor_count(uint32_t node, uint8_t count) {
    neighbor_counts[node].store(count, std::memory_order_release);
}

// ============================================================================
// HNSWIndex construction
// ============================================================================

HNSWIndex::HNSWIndex(const HNSWConfig& config)
    : config_(config),
      dist_fn_(resolve(config.distance_type)),
      rng_(config.random_seed),
      level_mult_(1.0 / std::log(static_cast<double>(config.M))) {

    if (config_.M_max0 == 0) {
        config_.M_max0 = config_.M * 2;
    }

    // Neighbor counts are stored as uint8_t, so M_max must fit
    if (config_.M > 255) {
        throw std::invalid_argument("M must be <= 255 (neighbor count stored as uint8_t)");
    }
    if (config_.M_max0 > 255) {
        throw std::invalid_argument("M_max0 must be <= 255 (neighbor count stored as uint8_t)");
    }
    if (config_.dim == 0) {
        throw std::invalid_argument("Vector dimension must be > 0");
    }

    // Pre-allocate layers (up to MAX_LAYERS)
    layers_.reserve(MAX_LAYERS);

    uint32_t init_cap = config.max_elements > 0 ? config.max_elements : 1024;
    reserve(init_cap);

    spdlog::debug("HNSWIndex created: dim={}, M={}, ef_construct={}, capacity={}",
                  config_.dim, config_.M, config_.ef_construct, capacity_);
}

HNSWIndex::~HNSWIndex() = default;

void HNSWIndex::reserve(uint32_t n) {
    std::lock_guard<std::mutex> lock(resize_mutex_);
    if (n <= capacity_) return;

    capacity_ = n;
    vectors_.resize(static_cast<size_t>(capacity_) * config_.dim, 0.0f);
    nodes_.resize(capacity_);

    // Ensure all existing layers have capacity
    for (auto& layer : layers_) {
        layer.ensure_capacity(capacity_);
    }
}

// ============================================================================
// Insert
// ============================================================================

uint8_t HNSWIndex::random_level() {
    std::lock_guard<std::mutex> lock(rng_mutex_);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    double r = dist(rng_);
    return static_cast<uint8_t>(
        std::min(static_cast<int>(-std::log(r) * level_mult_),
                 static_cast<int>(MAX_LAYERS - 1)));
}

void HNSWIndex::insert(uint32_t external_id, const float* data) {
    uint32_t internal_id = node_count_.fetch_add(1, std::memory_order_acq_rel);

    // Ensure capacity
    if (internal_id >= capacity_) {
        reserve(std::max(capacity_ * 2, internal_id + 1));
    }

    // Copy vector data
    std::memcpy(vector_data(internal_id), data,
                config_.dim * sizeof(float));

    // Assign random level
    uint8_t node_level = random_level();

    // Store metadata
    nodes_[internal_id].external_id = external_id;
    nodes_[internal_id].max_layer = node_level;
    nodes_[internal_id].lock.store(0, std::memory_order_relaxed);

    // Ensure layers exist up to node_level
    {
        std::lock_guard<std::mutex> lock(resize_mutex_);
        while (layers_.size() <= node_level) {
            uint32_t m_max = (layers_.size() == 0) ? config_.M_max0 : config_.M;
            layers_.emplace_back(m_max, capacity_);
        }
    }

    // First node — just set as entry point
    uint32_t ep = entry_point_.load(std::memory_order_acquire);
    if (ep == INVALID) {
        // CAS to become entry point (handles race with other first inserts)
        if (entry_point_.compare_exchange_strong(ep, internal_id,
                std::memory_order_acq_rel)) {
            max_level_.store(node_level, std::memory_order_release);
            return;
        }
        // Lost the race — another thread set entry point first. Fall through.
        ep = entry_point_.load(std::memory_order_acquire);
    }

    uint8_t cur_max_level = max_level_.load(std::memory_order_acquire);

    // Phase 1: Greedy descent from top layer down to node_level + 1
    uint32_t cur_node = ep;
    for (int level = static_cast<int>(cur_max_level);
         level > static_cast<int>(node_level); --level) {
        cur_node = greedy_search(data, cur_node, level, level);
    }

    // Phase 2: At each layer from min(node_level, cur_max_level) down to 0,
    //          do beam search and connect to nearest neighbors.
    uint8_t insert_top = std::min(node_level, cur_max_level);

    for (int level = static_cast<int>(insert_top); level >= 0; --level) {
        // Beam search to find candidates
        auto candidates = search_layer(data, cur_node, config_.ef_construct,
                                       static_cast<uint8_t>(level));

        // Select neighbors
        uint32_t M_max = (level == 0) ? config_.M_max0 : config_.M;
        auto neighbors = select_neighbors_heuristic(data, std::move(candidates), M_max);

        // Set this node's neighbors at this layer
        uint8_t count = static_cast<uint8_t>(
            std::min(static_cast<size_t>(M_max), neighbors.size()));
        layers_[level].set_neighbor_count(internal_id, count);
        for (uint8_t i = 0; i < count; ++i) {
            layers_[level].set_neighbor(internal_id, i, neighbors[i]);
        }

        // Add bidirectional connections
        for (uint8_t i = 0; i < count; ++i) {
            connect(internal_id, neighbors[i], static_cast<uint8_t>(level));
        }

        // Use the closest candidate as entry point for next lower layer
        if (!neighbors.empty()) {
            cur_node = neighbors[0];
        }
    }

    // If this node's level exceeds the current max, update entry point
    if (node_level > cur_max_level) {
        // CAS loop: update max_level and entry_point atomically-ish.
        // Slight race is OK — worst case, a concurrent search starts from
        // a slightly suboptimal entry point for one query.
        uint8_t expected_level = cur_max_level;
        if (max_level_.compare_exchange_strong(expected_level, node_level,
                std::memory_order_acq_rel)) {
            entry_point_.store(internal_id, std::memory_order_release);
        }
    }
}

// ============================================================================
// Search
// ============================================================================

std::vector<SearchResult> HNSWIndex::search(const float* query, uint32_t k,
                                             uint32_t ef) const {
    uint32_t count = node_count_.load(std::memory_order_acquire);
    if (count == 0) return {};

    if (ef == 0) ef = config_.ef_search;
    ef = std::max(ef, k);  // ef must be >= k

    uint32_t ep = entry_point_.load(std::memory_order_acquire);
    if (ep == INVALID) return {};

    uint8_t top_level = max_level_.load(std::memory_order_acquire);

    // Phase 1: Greedy descent from top to layer 1
    uint32_t cur_node = ep;
    for (int level = static_cast<int>(top_level); level > 0; --level) {
        cur_node = greedy_search(query, cur_node, level, level);
    }

    // Phase 2: Beam search at layer 0
    auto candidates = search_layer(query, cur_node, ef, 0);

    // Drain max-heap into vector, then sort to get k nearest
    std::vector<SearchResult> results;
    results.reserve(candidates.size());
    while (!candidates.empty()) {
        results.push_back(candidates.top());
        candidates.pop();
    }

    // Sort by distance ascending and keep only top-k
    std::sort(results.begin(), results.end());
    if (results.size() > k) {
        results.resize(k);
    }

    // Populate external IDs and similarity scores
    for (auto& r : results) {
        uint32_t internal = r.id;
        r.id = nodes_[internal].external_id;
        r.score = 1.0f / (1.0f + r.distance);  // L2 → similarity
    }

    return results;
}

// ============================================================================
// Internal: greedy_search
// ============================================================================

uint32_t HNSWIndex::greedy_search(const float* query, uint32_t entry,
                                   uint8_t top_layer, uint8_t target_layer) const {
    uint32_t cur = entry;
    float cur_dist = dist(query, cur);

    for (int level = static_cast<int>(top_layer);
         level >= static_cast<int>(target_layer); --level) {
        bool changed = true;
        while (changed) {
            changed = false;
            uint8_t count = layers_[level].get_neighbor_count(cur);
            for (uint8_t i = 0; i < count; ++i) {
                uint32_t neighbor = layers_[level].get_neighbor(cur, i);
                if (neighbor == INVALID) continue;
                // Bounds check: neighbor might be from a concurrent insert
                if (neighbor >= node_count_.load(std::memory_order_acquire)) continue;

                float d = dist(query, neighbor);
                if (d < cur_dist) {
                    cur = neighbor;
                    cur_dist = d;
                    changed = true;
                }
            }
        }
    }
    return cur;
}

// ============================================================================
// Internal: search_layer (beam search)
// ============================================================================

HNSWIndex::MaxHeap HNSWIndex::search_layer(const float* query, uint32_t entry_node,
                                             uint32_t ef, uint8_t layer) const {
    float entry_dist = dist(query, entry_node);

    // visited set — use a flat bitset for cache efficiency
    uint32_t count = node_count_.load(std::memory_order_acquire);
    std::vector<bool> visited(count, false);
    visited[entry_node] = true;

    // candidates: min-heap of nodes to explore (closest first)
    MinHeap candidates;
    candidates.push({entry_node, entry_dist, {}, 0.0f});

    // results: max-heap of best results so far (farthest on top for easy pruning)
    MaxHeap results;
    results.push({entry_node, entry_dist, {}, 0.0f});

    while (!candidates.empty()) {
        auto nearest = candidates.top();
        candidates.pop();

        // If the nearest candidate is farther than the worst result, stop
        if (nearest.distance > results.top().distance && results.size() >= ef) {
            break;
        }

        // Explore neighbors
        uint8_t n_count = layers_[layer].get_neighbor_count(nearest.id);
        for (uint8_t i = 0; i < n_count; ++i) {
            uint32_t neighbor = layers_[layer].get_neighbor(nearest.id, i);
            if (neighbor == INVALID) continue;
            if (neighbor >= count) continue;  // concurrent insert safety
            if (visited[neighbor]) continue;
            visited[neighbor] = true;

            float d = dist(query, neighbor);

            if (d < results.top().distance || results.size() < ef) {
                candidates.push({neighbor, d, {}, 0.0f});
                results.push({neighbor, d, {}, 0.0f});

                if (results.size() > ef) {
                    results.pop();  // remove farthest
                }
            }
        }
    }

    return results;
}

// ============================================================================
// Neighbor selection
// ============================================================================

std::vector<uint32_t> HNSWIndex::select_neighbors_simple(
    const float* query, const MaxHeap& candidates, uint32_t M) const {

    // Copy to a min-heap to get closest first
    MinHeap sorted;
    auto copy = candidates;
    while (!copy.empty()) {
        sorted.push(copy.top());
        copy.pop();
    }

    std::vector<uint32_t> result;
    result.reserve(M);
    while (!sorted.empty() && result.size() < M) {
        result.push_back(sorted.top().id);
        sorted.pop();
    }
    return result;
}

std::vector<uint32_t> HNSWIndex::select_neighbors_heuristic(
    const float* query, MaxHeap candidates, uint32_t M) const {

    // Convert max-heap to min-heap (closest first)
    MinHeap working;
    while (!candidates.empty()) {
        working.push(candidates.top());
        candidates.pop();
    }

    std::vector<uint32_t> result;
    result.reserve(M);

    // Rejected candidates kept for backfill (closest first)
    MinHeap discarded;

    while (!working.empty() && result.size() < M) {
        auto nearest = working.top();
        working.pop();

        // Heuristic: add this neighbor only if it's closer to the query
        // than to any already-selected neighbor. This promotes diversity.
        bool good = true;
        for (uint32_t selected : result) {
            float dist_to_selected = dist_fn_(
                vector_data(nearest.id), vector_data(selected), config_.dim);
            if (dist_to_selected < nearest.distance) {
                good = false;
                break;
            }
        }

        if (good) {
            result.push_back(nearest.id);
        } else {
            discarded.push(nearest);
        }
    }

    // Backfill: if heuristic was too aggressive, pad with closest discarded
    // candidates to ensure each node has enough connections for good recall.
    while (result.size() < M && !discarded.empty()) {
        result.push_back(discarded.top().id);
        discarded.pop();
    }

    return result;
}

// ============================================================================
// Bidirectional connection
// ============================================================================

void HNSWIndex::connect(uint32_t node, uint32_t neighbor, uint8_t layer) {
    uint32_t M_max = layers_[layer].M_max;

    // Lock the neighbor node to update its adjacency list
    lock_node(neighbor);

    uint8_t cur_count = layers_[layer].get_neighbor_count(neighbor);

    if (cur_count < static_cast<uint8_t>(M_max)) {
        // Simple case: just append
        layers_[layer].set_neighbor(neighbor, cur_count, node);
        layers_[layer].set_neighbor_count(neighbor, cur_count + 1);
    } else {
        // Neighbor list is full — keep M_max closest neighbors.
        // Compute distance from neighbor to the new node and find the worst
        // existing neighbor. If new is closer, replace worst.
        const float* neighbor_vec = vector_data(neighbor);
        float new_dist = dist_fn_(neighbor_vec, vector_data(node), config_.dim);

        // Find the worst (farthest) existing neighbor
        uint8_t worst_idx = 0;
        float worst_dist = 0.0f;
        for (uint8_t i = 0; i < cur_count; ++i) {
            uint32_t existing = layers_[layer].get_neighbor(neighbor, i);
            if (existing == INVALID) continue;
            float d = dist_fn_(neighbor_vec, vector_data(existing), config_.dim);
            if (d > worst_dist) {
                worst_dist = d;
                worst_idx = i;
            }
        }

        if (new_dist < worst_dist) {
            layers_[layer].set_neighbor(neighbor, worst_idx, node);
        }
    }

    unlock_node(neighbor);
}

// ============================================================================
// Spinlock
// ============================================================================

void HNSWIndex::lock_node(uint32_t node) {
    static constexpr unsigned MAX_SPIN_ITERS = 100000;  // ~100ms worst case
    unsigned iter = 0;
    while (true) {
        uint8_t expected = 0;
        if (nodes_[node].lock.compare_exchange_weak(
                expected, 1, std::memory_order_acquire, std::memory_order_relaxed)) {
            return;
        }
        if (iter < 4) {
            // Spin
        } else if (iter < 32) {
            std::this_thread::yield();
        } else if (iter < MAX_SPIN_ITERS) {
            std::this_thread::sleep_for(std::chrono::microseconds(1));
        } else {
            throw std::runtime_error(
                "Spinlock timeout on node " + std::to_string(node) +
                " — possible deadlock");
        }
        ++iter;
    }
}

void HNSWIndex::unlock_node(uint32_t node) {
    nodes_[node].lock.store(0, std::memory_order_release);
}

// ============================================================================
// Distance / vector access
// ============================================================================

float HNSWIndex::dist(const float* query, uint32_t internal_id) const {
    return dist_fn_(query, vector_data(internal_id), config_.dim);
}

float* HNSWIndex::vector_data(uint32_t internal_id) {
    return vectors_.data() + static_cast<size_t>(internal_id) * config_.dim;
}

const float* HNSWIndex::vector_data(uint32_t internal_id) const {
    return vectors_.data() + static_cast<size_t>(internal_id) * config_.dim;
}

const float* HNSWIndex::get_vector(uint32_t internal_id) const {
    return vector_data(internal_id);
}

// ============================================================================
// Persistence
// ============================================================================

// Binary format:
// [magic: 4 bytes "VRTX"]
// [version: uint32]
// [dim: uint32]
// [M: uint32]
// [M_max0: uint32]
// [ef_construct: uint32]
// [distance_type: uint32]
// [node_count: uint32]
// [max_level: uint8]
// [entry_point: uint32]
// [vectors: node_count * dim * float]
// [nodes: node_count * (external_id: uint32, max_layer: uint8)]
// [num_layers: uint32]
// For each layer:
//   [M_max: uint32]
//   [neighbor_counts: node_count * uint8]
//   [neighbors: node_count * M_max * uint32]

void HNSWIndex::save(const std::string& path) const {
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs.is_open()) {
        throw std::runtime_error("Cannot open file for writing: " + path);
    }

    uint32_t count = node_count_.load(std::memory_order_acquire);
    uint8_t ml = max_level_.load(std::memory_order_acquire);
    uint32_t ep = entry_point_.load(std::memory_order_acquire);
    uint32_t dt = static_cast<uint32_t>(config_.distance_type);

    // Header
    ofs.write("VRTX", 4);
    uint32_t version = 1;
    ofs.write(reinterpret_cast<const char*>(&version), 4);
    ofs.write(reinterpret_cast<const char*>(&config_.dim), 4);
    ofs.write(reinterpret_cast<const char*>(&config_.M), 4);
    ofs.write(reinterpret_cast<const char*>(&config_.M_max0), 4);
    ofs.write(reinterpret_cast<const char*>(&config_.ef_construct), 4);
    ofs.write(reinterpret_cast<const char*>(&dt), 4);
    ofs.write(reinterpret_cast<const char*>(&count), 4);
    ofs.write(reinterpret_cast<const char*>(&ml), 1);
    ofs.write(reinterpret_cast<const char*>(&ep), 4);

    // Vectors
    ofs.write(reinterpret_cast<const char*>(vectors_.data()),
              static_cast<std::streamsize>(count) * config_.dim * sizeof(float));

    // Node metadata
    for (uint32_t i = 0; i < count; ++i) {
        ofs.write(reinterpret_cast<const char*>(&nodes_[i].external_id), 4);
        ofs.write(reinterpret_cast<const char*>(&nodes_[i].max_layer), 1);
    }

    // Layers
    uint32_t num_layers = static_cast<uint32_t>(layers_.size());
    ofs.write(reinterpret_cast<const char*>(&num_layers), 4);

    for (uint32_t l = 0; l < num_layers; ++l) {
        auto& layer = layers_[l];
        ofs.write(reinterpret_cast<const char*>(&layer.M_max), 4);

        // Neighbor counts
        for (uint32_t i = 0; i < count; ++i) {
            uint8_t nc = layer.get_neighbor_count(i);
            ofs.write(reinterpret_cast<const char*>(&nc), 1);
        }

        // Neighbors
        for (uint32_t i = 0; i < count; ++i) {
            uint8_t nc = layer.get_neighbor_count(i);
            for (uint8_t j = 0; j < layer.M_max; ++j) {
                uint32_t n = (j < nc) ? layer.get_neighbor(i, j) : INVALID;
                ofs.write(reinterpret_cast<const char*>(&n), 4);
            }
        }
    }

    spdlog::info("Saved HNSW index: {} vectors, {} layers to {}",
                 count, num_layers, path);
}

std::unique_ptr<HNSWIndex> HNSWIndex::load(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs.is_open()) {
        throw std::runtime_error("Cannot open file for reading: " + path);
    }

    // Header
    char magic[4];
    ifs.read(magic, 4);
    if (std::string(magic, 4) != "VRTX") {
        throw std::runtime_error("Invalid HNSW index file: bad magic");
    }

    uint32_t version;
    ifs.read(reinterpret_cast<char*>(&version), 4);
    if (version != 1) {
        throw std::runtime_error("Unsupported HNSW index version: " + std::to_string(version));
    }

    HNSWConfig config;
    uint32_t count, dt, ep;
    uint8_t ml;

    ifs.read(reinterpret_cast<char*>(&config.dim), 4);
    ifs.read(reinterpret_cast<char*>(&config.M), 4);
    ifs.read(reinterpret_cast<char*>(&config.M_max0), 4);
    ifs.read(reinterpret_cast<char*>(&config.ef_construct), 4);
    ifs.read(reinterpret_cast<char*>(&dt), 4);
    ifs.read(reinterpret_cast<char*>(&count), 4);
    ifs.read(reinterpret_cast<char*>(&ml), 1);
    ifs.read(reinterpret_cast<char*>(&ep), 4);

    config.distance_type = static_cast<DistanceType>(dt);
    config.max_elements = count;

    // Validate deserialized values before allocating
    if (config.dim == 0 || config.dim > 65536) {
        throw std::runtime_error("Invalid dimension in index file: " + std::to_string(config.dim));
    }
    if (config.M > 255 || config.M_max0 > 255) {
        throw std::runtime_error("Invalid M/M_max0 in index file");
    }
    if (count > 100'000'000) {  // 100M vectors sanity cap
        throw std::runtime_error("Suspiciously large vector count: " + std::to_string(count));
    }
    if (ml >= MAX_LAYERS) {
        throw std::runtime_error("Invalid max_level in index file: " + std::to_string(ml));
    }
    if (ep != INVALID && ep >= count) {
        throw std::runtime_error("Invalid entry_point in index file: " + std::to_string(ep));
    }
    if (!ifs.good()) {
        throw std::runtime_error("Truncated index file header");
    }

    auto index = std::make_unique<HNSWIndex>(config);
    index->node_count_.store(count, std::memory_order_relaxed);
    index->max_level_.store(ml, std::memory_order_relaxed);
    index->entry_point_.store(ep, std::memory_order_relaxed);

    // Vectors
    index->reserve(count);
    ifs.read(reinterpret_cast<char*>(index->vectors_.data()),
             static_cast<std::streamsize>(count) * config.dim * sizeof(float));
    if (!ifs.good()) {
        throw std::runtime_error("Truncated index file: vector data incomplete");
    }

    // Node metadata
    for (uint32_t i = 0; i < count; ++i) {
        ifs.read(reinterpret_cast<char*>(&index->nodes_[i].external_id), 4);
        ifs.read(reinterpret_cast<char*>(&index->nodes_[i].max_layer), 1);
        index->nodes_[i].lock.store(0, std::memory_order_relaxed);
    }

    // Layers
    uint32_t num_layers;
    ifs.read(reinterpret_cast<char*>(&num_layers), 4);
    if (num_layers > MAX_LAYERS) {
        throw std::runtime_error("Invalid layer count in index file: " + std::to_string(num_layers));
    }

    index->layers_.clear();
    for (uint32_t l = 0; l < num_layers; ++l) {
        uint32_t m_max;
        ifs.read(reinterpret_cast<char*>(&m_max), 4);

        index->layers_.emplace_back(m_max, count);
        auto& layer = index->layers_.back();

        // Neighbor counts
        for (uint32_t i = 0; i < count; ++i) {
            uint8_t nc;
            ifs.read(reinterpret_cast<char*>(&nc), 1);
            layer.set_neighbor_count(i, nc);
        }

        // Neighbors
        for (uint32_t i = 0; i < count; ++i) {
            for (uint32_t j = 0; j < m_max; ++j) {
                uint32_t n;
                ifs.read(reinterpret_cast<char*>(&n), 4);
                layer.set_neighbor(i, j, n);
            }
        }
    }

    spdlog::info("Loaded HNSW index: {} vectors, {} layers from {}",
                 count, num_layers, path);
    return index;
}

}  // namespace vortex
