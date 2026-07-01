// Line-segment intersection function variant migrated from the legacy app corpus.

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>

namespace {

constexpr uint32_t kCount = 64;
constexpr uint32_t kLineElems = kCount * 4u;

void initialize_lines(std::array<float, kLineElems> &line_a,
                      std::array<float, kLineElems> &line_b) {
    line_a[0] = 0.0f;
    line_a[1] = 0.0f;
    line_a[2] = 2.0f;
    line_a[3] = 2.0f;
    line_b[0] = 0.0f;
    line_b[1] = 2.0f;
    line_b[2] = 2.0f;
    line_b[3] = 0.0f;

    line_a[4] = 0.0f;
    line_a[5] = 0.0f;
    line_a[6] = 1.0f;
    line_a[7] = 0.0f;
    line_b[4] = 0.0f;
    line_b[5] = 1.0f;
    line_b[6] = 1.0f;
    line_b[7] = 1.0f;

    line_a[8] = 0.0f;
    line_a[9] = 0.0f;
    line_a[10] = 1.0f;
    line_a[11] = 0.0f;
    line_b[8] = 0.0f;
    line_b[9] = 1.0f;
    line_b[10] = 1.0f;
    line_b[11] = 1.0f;

    for (uint32_t i = 3; i < kCount; ++i) {
        const float offset = static_cast<float>(i) * 0.1f;
        line_a[i * 4u + 0u] = offset;
        line_a[i * 4u + 1u] = 0.0f;
        line_a[i * 4u + 2u] = 2.0f + offset;
        line_a[i * 4u + 3u] = 2.0f;

        line_b[i * 4u + 0u] = offset;
        line_b[i * 4u + 1u] = 2.0f;
        line_b[i * 4u + 2u] = 2.0f + offset;
        line_b[i * 4u + 3u] = 0.0f;
    }
}

void line_intersect_ref(const float *line_a, const float *line_b,
                        uint32_t *out, uint32_t count) {
    for (uint32_t i = 0; i < count; ++i) {
        const float ax1 = line_a[i * 4u + 0u];
        const float ay1 = line_a[i * 4u + 1u];
        const float ax2 = line_a[i * 4u + 2u];
        const float ay2 = line_a[i * 4u + 3u];
        const float bx1 = line_b[i * 4u + 0u];
        const float by1 = line_b[i * 4u + 1u];
        const float bx2 = line_b[i * 4u + 2u];
        const float by2 = line_b[i * 4u + 3u];

        const float dax = ax2 - ax1;
        const float day = ay2 - ay1;
        const float dbx = bx2 - bx1;
        const float dby = by2 - by1;
        const float denom = dax * dby - day * dbx;
        if (std::fabs(denom) < 1e-8f) {
            out[i] = 0;
            continue;
        }

        const float dx = bx1 - ax1;
        const float dy = by1 - ay1;
        const float t = (dx * dby - dy * dbx) / denom;
        const float u = (dx * day - dy * dax) / denom;
        out[i] = (t >= 0.0f && t <= 1.0f && u >= 0.0f && u <= 1.0f) ? 1u : 0u;
    }
}

uint64_t checksum(const std::array<uint32_t, kCount> &values) {
    uint64_t sum = 0;
    for (uint32_t i = 0; i < kCount; ++i) {
        sum += static_cast<uint64_t>(i + 1u) * values[i];
    }
    return sum;
}

} // namespace

extern "C" __attribute__((noinline))
void line_intersect_kernel(const float *line_a, const float *line_b,
                           uint32_t *out, uint32_t count) {
    for (uint32_t i = 0; i < count; ++i) {
        const float ax1 = line_a[i * 4u + 0u];
        const float ay1 = line_a[i * 4u + 1u];
        const float ax2 = line_a[i * 4u + 2u];
        const float ay2 = line_a[i * 4u + 3u];
        const float bx1 = line_b[i * 4u + 0u];
        const float by1 = line_b[i * 4u + 1u];
        const float bx2 = line_b[i * 4u + 2u];
        const float by2 = line_b[i * 4u + 3u];

        const float dax = ax2 - ax1;
        const float day = ay2 - ay1;
        const float dbx = bx2 - bx1;
        const float dby = by2 - by1;
        const float denom = dax * dby - day * dbx;
        if (std::fabs(denom) < 1e-8f) {
            out[i] = 0;
            continue;
        }

        const float dx = bx1 - ax1;
        const float dy = by1 - ay1;
        const float t = (dx * dby - dy * dbx) / denom;
        const float u = (dx * day - dy * dax) / denom;
        out[i] = (t >= 0.0f && t <= 1.0f && u >= 0.0f && u <= 1.0f) ? 1u : 0u;
    }
}

int main() {
    std::array<float, kLineElems> line_a = {};
    std::array<float, kLineElems> line_b = {};
    std::array<uint32_t, kCount> reference = {};
    std::array<uint32_t, kCount> candidate = {};

    initialize_lines(line_a, line_b);
    line_intersect_ref(line_a.data(), line_b.data(), reference.data(), kCount);
    line_intersect_kernel(line_a.data(), line_b.data(), candidate.data(), kCount);

    for (uint32_t i = 0; i < kCount; ++i) {
        if (reference[i] != candidate[i]) {
            std::puts("FAILED");
            return 1;
        }
    }

    std::printf("line_intersect checksum: %llu\n",
                static_cast<unsigned long long>(checksum(candidate)));
    std::puts("PASSED");
    return 0;
}
