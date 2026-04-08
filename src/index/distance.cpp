// SIMD distance function implementations.

#include "index/distance.h"

#include <cmath>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif

#if defined(__SSE4_1__)
#include <smmintrin.h>
#endif

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace vortex {

// ============================================================================
// Scalar fallback
// ============================================================================

float l2_scalar(const float* a, const float* b, uint32_t dim) {
    float sum = 0.0f;
    for (uint32_t i = 0; i < dim; ++i) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}

float ip_scalar(const float* a, const float* b, uint32_t dim) {
    float sum = 0.0f;
    for (uint32_t i = 0; i < dim; ++i) {
        sum += a[i] * b[i];
    }
    // Return negative so that "lower = more similar" convention holds
    return -sum;
}

// ============================================================================
// ARM NEON (Apple Silicon, Graviton, etc.)
// ============================================================================

#if defined(__ARM_NEON) || defined(__ARM_NEON__)

float l2_neon(const float* a, const float* b, uint32_t dim) {
    float32x4_t sum_vec = vdupq_n_f32(0.0f);
    uint32_t i = 0;

    // Process 16 floats per iteration (4 NEON registers)
    for (; i + 15 < dim; i += 16) {
        float32x4_t a0 = vld1q_f32(a + i);
        float32x4_t b0 = vld1q_f32(b + i);
        float32x4_t d0 = vsubq_f32(a0, b0);
        sum_vec = vmlaq_f32(sum_vec, d0, d0);

        float32x4_t a1 = vld1q_f32(a + i + 4);
        float32x4_t b1 = vld1q_f32(b + i + 4);
        float32x4_t d1 = vsubq_f32(a1, b1);
        sum_vec = vmlaq_f32(sum_vec, d1, d1);

        float32x4_t a2 = vld1q_f32(a + i + 8);
        float32x4_t b2 = vld1q_f32(b + i + 8);
        float32x4_t d2 = vsubq_f32(a2, b2);
        sum_vec = vmlaq_f32(sum_vec, d2, d2);

        float32x4_t a3 = vld1q_f32(a + i + 12);
        float32x4_t b3 = vld1q_f32(b + i + 12);
        float32x4_t d3 = vsubq_f32(a3, b3);
        sum_vec = vmlaq_f32(sum_vec, d3, d3);
    }

    // Process 4 floats per iteration
    for (; i + 3 < dim; i += 4) {
        float32x4_t av = vld1q_f32(a + i);
        float32x4_t bv = vld1q_f32(b + i);
        float32x4_t dv = vsubq_f32(av, bv);
        sum_vec = vmlaq_f32(sum_vec, dv, dv);
    }

    // Horizontal sum
    float sum = vaddvq_f32(sum_vec);

    // Scalar tail
    for (; i < dim; ++i) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}

float ip_neon(const float* a, const float* b, uint32_t dim) {
    float32x4_t sum_vec = vdupq_n_f32(0.0f);
    uint32_t i = 0;

    for (; i + 15 < dim; i += 16) {
        sum_vec = vmlaq_f32(sum_vec, vld1q_f32(a + i),      vld1q_f32(b + i));
        sum_vec = vmlaq_f32(sum_vec, vld1q_f32(a + i + 4),  vld1q_f32(b + i + 4));
        sum_vec = vmlaq_f32(sum_vec, vld1q_f32(a + i + 8),  vld1q_f32(b + i + 8));
        sum_vec = vmlaq_f32(sum_vec, vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
    }

    for (; i + 3 < dim; i += 4) {
        sum_vec = vmlaq_f32(sum_vec, vld1q_f32(a + i), vld1q_f32(b + i));
    }

    float sum = vaddvq_f32(sum_vec);

    for (; i < dim; ++i) {
        sum += a[i] * b[i];
    }
    return -sum;
}

#endif  // ARM NEON

// ============================================================================
// SSE4.1
// ============================================================================

#if defined(__SSE4_1__) || defined(__AVX2__)

float l2_sse(const float* a, const float* b, uint32_t dim) {
    __m128 sum_vec = _mm_setzero_ps();
    uint32_t i = 0;

    for (; i + 3 < dim; i += 4) {
        __m128 av = _mm_loadu_ps(a + i);
        __m128 bv = _mm_loadu_ps(b + i);
        __m128 dv = _mm_sub_ps(av, bv);
        sum_vec = _mm_add_ps(sum_vec, _mm_mul_ps(dv, dv));
    }

    // Horizontal sum
    sum_vec = _mm_hadd_ps(sum_vec, sum_vec);
    sum_vec = _mm_hadd_ps(sum_vec, sum_vec);
    float sum = _mm_cvtss_f32(sum_vec);

    for (; i < dim; ++i) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}

float ip_sse(const float* a, const float* b, uint32_t dim) {
    __m128 sum_vec = _mm_setzero_ps();
    uint32_t i = 0;

    for (; i + 3 < dim; i += 4) {
        __m128 av = _mm_loadu_ps(a + i);
        __m128 bv = _mm_loadu_ps(b + i);
        sum_vec = _mm_add_ps(sum_vec, _mm_mul_ps(av, bv));
    }

    sum_vec = _mm_hadd_ps(sum_vec, sum_vec);
    sum_vec = _mm_hadd_ps(sum_vec, sum_vec);
    float sum = _mm_cvtss_f32(sum_vec);

    for (; i < dim; ++i) {
        sum += a[i] * b[i];
    }
    return -sum;
}

#endif  // SSE4.1

// ============================================================================
// AVX2
// ============================================================================

#if defined(__AVX2__)

float l2_avx2(const float* a, const float* b, uint32_t dim) {
    __m256 sum_vec = _mm256_setzero_ps();
    uint32_t i = 0;

    for (; i + 7 < dim; i += 8) {
        __m256 av = _mm256_loadu_ps(a + i);
        __m256 bv = _mm256_loadu_ps(b + i);
        __m256 dv = _mm256_sub_ps(av, bv);
        sum_vec = _mm256_fmadd_ps(dv, dv, sum_vec);
    }

    // Horizontal sum: 256 → 128 → scalar
    __m128 hi = _mm256_extractf128_ps(sum_vec, 1);
    __m128 lo = _mm256_castps256_ps128(sum_vec);
    __m128 sum128 = _mm_add_ps(lo, hi);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    float sum = _mm_cvtss_f32(sum128);

    // Scalar tail
    for (; i < dim; ++i) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}

float ip_avx2(const float* a, const float* b, uint32_t dim) {
    __m256 sum_vec = _mm256_setzero_ps();
    uint32_t i = 0;

    for (; i + 7 < dim; i += 8) {
        __m256 av = _mm256_loadu_ps(a + i);
        __m256 bv = _mm256_loadu_ps(b + i);
        sum_vec = _mm256_fmadd_ps(av, bv, sum_vec);
    }

    __m128 hi = _mm256_extractf128_ps(sum_vec, 1);
    __m128 lo = _mm256_castps256_ps128(sum_vec);
    __m128 sum128 = _mm_add_ps(lo, hi);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    float sum = _mm_cvtss_f32(sum128);

    for (; i < dim; ++i) {
        sum += a[i] * b[i];
    }
    return -sum;
}

#endif  // AVX2

// ============================================================================
// Resolution
// ============================================================================

DistanceFn resolve(DistanceType type) {
    if (type == DistanceType::L2) {
#if defined(__AVX2__)
        return l2_avx2;
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
        return l2_neon;
#elif defined(__SSE4_1__)
        return l2_sse;
#else
        return l2_scalar;
#endif
    } else {
#if defined(__AVX2__)
        return ip_avx2;
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
        return ip_neon;
#elif defined(__SSE4_1__)
        return ip_sse;
#else
        return ip_scalar;
#endif
    }
}

float distance(const float* a, const float* b, uint32_t dim, DistanceType type) {
    static DistanceFn l2_fn = resolve(DistanceType::L2);
    static DistanceFn ip_fn = resolve(DistanceType::InnerProduct);
    return (type == DistanceType::L2) ? l2_fn(a, b, dim) : ip_fn(a, b, dim);
}

}  // namespace vortex
