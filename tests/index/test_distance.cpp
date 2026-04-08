#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "index/distance.h"

#include <cmath>
#include <vector>

using namespace vortex;
using Catch::Matchers::WithinAbs;

TEST_CASE("L2 distance - identical vectors", "[distance]") {
    std::vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f};
    float d = l2_scalar(a.data(), a.data(), 4);
    REQUIRE_THAT(d, WithinAbs(0.0, 1e-6));
}

TEST_CASE("L2 distance - known values", "[distance]") {
    std::vector<float> a = {1.0f, 0.0f, 0.0f};
    std::vector<float> b = {0.0f, 1.0f, 0.0f};
    float d = l2_scalar(a.data(), b.data(), 3);
    REQUIRE_THAT(d, WithinAbs(2.0, 1e-6));  // (1-0)^2 + (0-1)^2 + 0 = 2
}

TEST_CASE("Inner product distance - orthogonal", "[distance]") {
    std::vector<float> a = {1.0f, 0.0f, 0.0f};
    std::vector<float> b = {0.0f, 1.0f, 0.0f};
    float d = ip_scalar(a.data(), b.data(), 3);
    REQUIRE_THAT(d, WithinAbs(0.0, 1e-6));  // -0 = 0
}

TEST_CASE("Inner product distance - parallel", "[distance]") {
    std::vector<float> a = {1.0f, 2.0f, 3.0f};
    float d = ip_scalar(a.data(), a.data(), 3);
    // -(1+4+9) = -14
    REQUIRE_THAT(d, WithinAbs(-14.0, 1e-6));
}

TEST_CASE("SIMD distance matches scalar - L2", "[distance]") {
    // Generate a 768-dim vector pair
    std::vector<float> a(768), b(768);
    for (int i = 0; i < 768; ++i) {
        a[i] = static_cast<float>(i % 37) / 37.0f;
        b[i] = static_cast<float>((i * 7 + 13) % 41) / 41.0f;
    }

    float scalar = l2_scalar(a.data(), b.data(), 768);
    auto resolved = resolve(DistanceType::L2);
    float simd = resolved(a.data(), b.data(), 768);

    REQUIRE_THAT(simd, WithinAbs(scalar, 1e-2));
}

TEST_CASE("SIMD distance matches scalar - IP", "[distance]") {
    std::vector<float> a(768), b(768);
    for (int i = 0; i < 768; ++i) {
        a[i] = static_cast<float>(i % 37) / 37.0f;
        b[i] = static_cast<float>((i * 7 + 13) % 41) / 41.0f;
    }

    float scalar = ip_scalar(a.data(), b.data(), 768);
    auto resolved = resolve(DistanceType::InnerProduct);
    float simd = resolved(a.data(), b.data(), 768);

    REQUIRE_THAT(simd, WithinAbs(scalar, 1e-2));
}

TEST_CASE("Distance with odd dimensions", "[distance]") {
    // Non-multiple-of-4 dimensions should work with scalar tail
    std::vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    std::vector<float> b = {5.0f, 4.0f, 3.0f, 2.0f, 1.0f};

    float scalar = l2_scalar(a.data(), b.data(), 5);
    auto resolved = resolve(DistanceType::L2);
    float simd = resolved(a.data(), b.data(), 5);

    // (4+4+0+4+16) = 40 (actually: 16+4+0+4+16 = 40)
    REQUIRE_THAT(scalar, WithinAbs(40.0, 1e-6));
    REQUIRE_THAT(simd, WithinAbs(scalar, 1e-4));
}
