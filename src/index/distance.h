#pragma once
// SIMD-accelerated distance functions for vector similarity search.
//
// Supports:
//   - ARM NEON (Apple Silicon, Arm servers)
//   - SSE4.1 (x86-64 baseline)
//   - AVX2 (modern x86-64)
//   - Scalar fallback
//
// Distance function is resolved once at startup via resolve().

#include <cstdint>
#include <functional>

namespace vortex {

enum class DistanceType {
    L2,           // Euclidean (squared)
    InnerProduct  // Negative inner product (lower = more similar)
};

/// Distance function signature: (a, b, dim) → distance
using DistanceFn = float(*)(const float* a, const float* b, uint32_t dim);

/// Scalar fallback implementations
float l2_scalar(const float* a, const float* b, uint32_t dim);
float ip_scalar(const float* a, const float* b, uint32_t dim);

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
float l2_neon(const float* a, const float* b, uint32_t dim);
float ip_neon(const float* a, const float* b, uint32_t dim);
#endif

#if defined(__SSE4_1__) || defined(__AVX2__)
float l2_sse(const float* a, const float* b, uint32_t dim);
float ip_sse(const float* a, const float* b, uint32_t dim);
#endif

#if defined(__AVX2__)
float l2_avx2(const float* a, const float* b, uint32_t dim);
float ip_avx2(const float* a, const float* b, uint32_t dim);
#endif

/// Resolve the best distance function for this CPU at runtime.
DistanceFn resolve(DistanceType type);

/// Convenience: compute distance between two vectors using resolved function.
/// Thread-safe — the resolved function pointer is immutable after first call.
float distance(const float* a, const float* b, uint32_t dim, DistanceType type);

}  // namespace vortex
