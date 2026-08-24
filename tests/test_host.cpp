// Host-only unit tests for the CPU oracle, compare, and stats (no GPU required).

#include "../src/common/cpu_reference.hpp"
#include "../src/common/stats.hpp"
#include "../src/common/gb10.hpp"

#include <cstdio>
#include <cassert>
#include <cmath>
#include <limits>

static int g_failures = 0;
static int g_checks   = 0;

#define CHECK(cond, msg)                                                       \
    do {                                                                       \
        ++g_checks;                                                            \
        if (!(cond)) {                                                         \
            std::printf("  FAIL  %s\n        at %s:%d\n", (msg), __FILE__, __LINE__); \
            ++g_failures;                                                      \
        }                                                                      \
    } while (0)

#define CHECK_NEAR(a, b, tol, msg)                                             \
    do {                                                                       \
        ++g_checks;                                                            \
        if (std::fabs((a) - (b)) > (tol)) {                                    \
            std::printf("  FAIL  %s\n        got %.9g want %.9g (tol %.3g) at %s:%d\n", \
                        (msg), (double)(a), (double)(b), (double)(tol), __FILE__, __LINE__); \
            ++g_failures;                                                      \
        }                                                                      \
    } while (0)

static void test_sgemm_identity() {
    // A is 2x3, B is 3x2; expected values hand-computed.
    std::vector<float> A = {1, 2, 3, 4, 5, 6};
    std::vector<float> B = {1, 0, 0, 1, 1, 1};
    std::vector<float> C(4, 0.0f);
    ladder::sgemm_cpu(2, 2, 3, 1.0f, A.data(), B.data(), 0.0f, C.data());
    CHECK_NEAR(C[0],  4.0f, 1e-6, "sgemm_cpu C[0,0]");
    CHECK_NEAR(C[1],  5.0f, 1e-6, "sgemm_cpu C[0,1]");
    CHECK_NEAR(C[2], 10.0f, 1e-6, "sgemm_cpu C[1,0]");
    CHECK_NEAR(C[3], 11.0f, 1e-6, "sgemm_cpu C[1,1]");
}

static void test_sgemm_alpha_beta() {
    std::vector<float> A = {1, 1};        // 1x2
    std::vector<float> B = {1, 1};        // 2x1
    std::vector<float> C = {10.0f};       // 1x1, pre-filled
    // C = 2*(1*1 + 1*1) + 3*10 = 4 + 30 = 34
    ladder::sgemm_cpu(1, 1, 2, 2.0f, A.data(), B.data(), 3.0f, C.data());
    CHECK_NEAR(C[0], 34.0f, 1e-6, "sgemm_cpu honors alpha and beta");
}

// A float accumulator drifts measurably at this K; a double one does not.
static void test_oracle_uses_double_accumulation() {
    const int K = 100000;
    std::vector<float> A(K, 1.0f);
    std::vector<float> B(K, 0.1f);        // 0.1 is not representable in binary
    std::vector<float> C(1, 0.0f);
    ladder::sgemm_cpu(1, 1, K, 1.0f, A.data(), B.data(), 0.0f, C.data());
    // True answer is 10000.0; a naive float accumulator lands near 9998.5.
    CHECK_NEAR(C[0], 10000.0f, 1.0f, "oracle accumulates in double (K=100k)");
}

static void test_compare_catches_errors() {
    std::vector<float> want = {1.0f, 2.0f, 3.0f};
    std::vector<float> good = {1.0f, 2.0f, 3.0f};
    std::vector<float> bad  = {1.0f, 2.0f, 3.5f};

    auto r_good = ladder::compare(good, want, 1e-5);
    CHECK(r_good.passed, "compare passes identical vectors");
    CHECK(r_good.num_mismatches == 0, "compare reports zero mismatches when equal");

    auto r_bad = ladder::compare(bad, want, 1e-5);
    CHECK(!r_bad.passed, "compare fails a real mismatch");
    CHECK(r_bad.num_mismatches == 1, "compare counts exactly one mismatch");
    CHECK(r_bad.first_bad_index == 2, "compare reports the index of the first bad element");
}

// Scale floor: fp32 noise on tiny elements must pass, problem-scale errors must fail.
static void test_compare_scale_floor() {
    std::vector<float> want = {22.6f, 1e-4f};
    std::vector<float> noisy = {22.6f, 1e-4f + 1e-5f};   // fp32-noise-sized absolute error

    auto r_strict = ladder::compare(noisy, want, 2e-4);            // scale = 1.0
    CHECK(r_strict.passed, "tiny absolute error on tiny element passes at scale 1");

    auto r_scaled = ladder::compare(noisy, want, 2e-4, 22.6);      // scale = sqrt(512)
    CHECK(r_scaled.passed, "fp32 noise passes with a GEMM-sized scale floor");

    std::vector<float> wrong = {22.6f, 0.5f};                      // O(1) error, tiny target
    auto r_wrong = ladder::compare(wrong, want, 2e-4, 22.6);
    CHECK(!r_wrong.passed, "problem-scale error on a tiny element still fails");
}

// NaN/Inf must never pass, regardless of tolerance.
static void test_compare_rejects_nan() {
    std::vector<float> want = {1.0f, 2.0f};
    std::vector<float> nan_got = {1.0f, std::numeric_limits<float>::quiet_NaN()};
    auto r = ladder::compare(nan_got, want, 1e5);   // absurdly loose tolerance
    CHECK(!r.passed, "compare rejects NaN even with a huge tolerance");

    std::vector<float> inf_got = {1.0f, std::numeric_limits<float>::infinity()};
    auto r2 = ladder::compare(inf_got, want, 1e5);
    CHECK(!r2.passed, "compare rejects Inf even with a huge tolerance");
}

static void test_compare_size_mismatch() {
    std::vector<float> a = {1.0f};
    std::vector<float> b = {1.0f, 2.0f};
    CHECK(!ladder::compare(a, b, 1e-5).passed, "compare fails on size mismatch");
}

static void test_percentiles() {
    // Linear interpolation over indices 0..99: p50 = 50.5, p90 = 90.1.
    std::vector<double> v;
    for (int i = 1; i <= 100; ++i) v.push_back(static_cast<double>(i));

    CHECK_NEAR(ladder::percentile(v, 0.50), 50.5, 1e-9, "p50 of 1..100");
    CHECK_NEAR(ladder::percentile(v, 0.90), 90.1, 1e-9, "p90 of 1..100");
    CHECK_NEAR(ladder::percentile(v, 0.00),  1.0, 1e-9, "p0 is the minimum");
    CHECK_NEAR(ladder::percentile(v, 1.00),100.0, 1e-9, "p100 is the maximum");

    std::vector<double> shuffled = {100, 1, 50, 2, 99};
    std::vector<double> sorted   = {1, 2, 50, 99, 100};
    CHECK_NEAR(ladder::percentile(shuffled, 0.5),
               ladder::percentile(sorted, 0.5), 1e-9, "percentile sorts internally");

    CHECK_NEAR(ladder::percentile({}, 0.5), 0.0, 1e-9, "percentile of empty is 0");
    CHECK_NEAR(ladder::percentile({7.0}, 0.9), 7.0, 1e-9, "percentile of one sample");
}

static void test_summarize() {
    std::vector<double> v = {10.0, 12.0, 11.0, 100.0};   // one deliberate outlier
    auto s = ladder::summarize(v);

    CHECK(s.n == 4, "summarize counts samples");
    CHECK_NEAR(s.min_ms,  10.0, 1e-9, "min");
    CHECK_NEAR(s.max_ms, 100.0, 1e-9, "max");
    CHECK_NEAR(s.mean_ms, 33.25, 1e-9, "mean");

    CHECK_NEAR(s.p50_ms, 11.5, 1e-9, "median resists the outlier");
    CHECK(s.mean_ms > s.p50_ms * 2.0, "mean is distorted by the tail, median is not");

    CHECK(s.stddev_ms > 0.0, "stddev is positive for varying samples");
    CHECK(s.cv() > 0.0, "cv is positive");
    CHECK_NEAR(s.spread(), 0.15, 1e-9, "spread = (p50-min)/min");

    auto one = ladder::summarize({5.0});
    CHECK_NEAR(one.stddev_ms, 0.0, 1e-9, "stddev of a single sample is 0, not NaN");
    CHECK_NEAR(one.cv(), 0.0, 1e-9, "cv of a single sample is 0, not NaN");

    auto none = ladder::summarize({});
    CHECK(none.n == 0, "summarize of empty is empty");
}

static void test_flop_and_byte_accounting() {
    CHECK_NEAR(ladder::gemm_flops(2, 3, 4), 48.0, 1e-9, "gemm_flops = 2MNK");

    // (M*K + K*N + M*N) * 4 bytes = (8 + 12 + 6) * 4 = 104
    CHECK_NEAR(ladder::gemm_min_bytes(2, 3, 4), 104.0, 1e-9, "gemm_min_bytes fp32");
    CHECK_NEAR(ladder::gemm_min_bytes(2, 3, 4, 2), 52.0, 1e-9, "gemm_min_bytes bf16");

    const double ai_4096 = ladder::arithmetic_intensity(4096, 4096, 4096);
    CHECK(ai_4096 > gb10::kRidgePointFp32,
          "4096^3 GEMM is above the fp32 ridge point (compute-bound is achievable)");

    const double ai_skinny = ladder::arithmetic_intensity(4096, 4096, 1);
    CHECK(ai_skinny < gb10::kRidgePointFp32,
          "K=1 GEMM is below the ridge point (memory-bound, no point tiling)");

    CHECK_NEAR(ladder::gflops(2e9, 1000.0), 2.0, 1e-9, "gflops conversion");
    CHECK_NEAR(ladder::gbytes_per_s(1e9, 1000.0), 1.0, 1e-9, "gbytes/s conversion");
    CHECK_NEAR(ladder::gflops(1e9, 0.0), 0.0, 1e-9, "gflops guards divide-by-zero");
}

static void test_roofline_constants_are_sane() {
    CHECK(gb10::kDramBandwidthMeasuredGBs < gb10::kDramBandwidthPeakGBs,
          "measured bandwidth is below spec-sheet peak (we use the honest number)");
    CHECK(gb10::kRidgePointFp32 > 100.0 && gb10::kRidgePointFp32 < 200.0,
          "fp32 ridge point lands near ~129 FLOP/byte");
    CHECK(gb10::kRidgePointBf16 > gb10::kRidgePointFp32 * 5.0,
          "bf16 tensor-core ridge point is far higher -- tiling matters much more");
    CHECK(gb10::kMaxSmemPerBlockBytes == 99 * 1024,
          "shared memory per block is capped at 99KB on GB10, not Hopper's 228KB");
}

static void test_make_matrix_is_deterministic() {
    auto a = ladder::make_matrix(4, 4, 1234);
    auto b = ladder::make_matrix(4, 4, 1234);
    auto c = ladder::make_matrix(4, 4, 5678);
    CHECK(a == b, "same seed produces identical matrices (reproducible tests)");
    CHECK(a != c, "different seed produces different matrices");
    CHECK(a.size() == 16u, "make_matrix returns rows*cols elements");

    bool has_neg = false, has_pos = false;
    for (float v : a) { if (v < 0) has_neg = true; if (v > 0) has_pos = true; }
    CHECK(has_neg && has_pos, "test data has both signs (catches sign bugs)");
}

int main() {
    std::printf("\n=== kernel-ladder host tests (no GPU required) ===\n\n");

    std::printf("[oracle]\n");
    test_sgemm_identity();
    test_sgemm_alpha_beta();
    test_oracle_uses_double_accumulation();

    std::printf("[compare]\n");
    test_compare_catches_errors();
    test_compare_scale_floor();
    test_compare_rejects_nan();
    test_compare_size_mismatch();

    std::printf("[stats]\n");
    test_percentiles();
    test_summarize();

    std::printf("[roofline]\n");
    test_flop_and_byte_accounting();
    test_roofline_constants_are_sane();

    std::printf("[testdata]\n");
    test_make_matrix_is_deterministic();

    std::printf("\n%d checks, %d failures\n", g_checks, g_failures);
    if (g_failures == 0) {
        std::printf("PASS\n\n");
        return 0;
    }
    std::printf("FAIL\n\n");
    return 1;
}
