// Host-only unit tests for the attention oracle and its FLOP/byte accounting.

#include "../src/attention/attention_reference.hpp"

#include <cstdio>
#include <cmath>
#include <vector>
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
        if (!(std::fabs((double)(a) - (double)(b)) <= (double)(tol))) {        \
            std::printf("  FAIL  %s\n        got %.9g want %.9g (tol %.3g) at %s:%d\n", \
                        (msg), (double)(a), (double)(b), (double)(tol), __FILE__, __LINE__); \
            ++g_failures;                                                      \
        }                                                                      \
    } while (0)

// Q = 0 makes every score 0, so each output row is the column mean of V.
static void test_uniform_attention_is_the_mean_of_v() {
    const int B = 1, H = 1, S = 2, D = 2;
    std::vector<float> Q = {0, 0, 0, 0};
    std::vector<float> K = {5, -7, 11, 13};
    std::vector<float> V = {1, 2, 3, 4};
    std::vector<float> O(4, -999.0f);

    ladder::attention_cpu(B, H, S, D, ladder::attention_scale(D), /*causal=*/false,
                          Q.data(), K.data(), V.data(), O.data());

    CHECK_NEAR(O[0], 2.0f, 1e-6, "uniform attention O[0][0] = mean of V col 0");
    CHECK_NEAR(O[1], 3.0f, 1e-6, "uniform attention O[0][1] = mean of V col 1");
    CHECK_NEAR(O[2], 2.0f, 1e-6, "uniform attention O[1][0] = mean of V col 0");
    CHECK_NEAR(O[3], 3.0f, 1e-6, "uniform attention O[1][1] = mean of V col 1");
}

// Q = K = I: expected values recomputed here from the closed-form softmax weights.
static void test_two_by_two_against_closed_form() {
    const int B = 1, H = 1, S = 2, D = 2;
    std::vector<float> Q = {1, 0, 0, 1};
    std::vector<float> K = {1, 0, 0, 1};
    std::vector<float> V = {1, 2, 3, 4};
    std::vector<float> O(4, -999.0f);

    const float scale = ladder::attention_scale(D);
    ladder::attention_cpu(B, H, S, D, scale, false,
                          Q.data(), K.data(), V.data(), O.data());

    const double s  = static_cast<double>(scale);
    const double e  = std::exp(s);
    const double p0 = e / (e + 1.0);
    const double p1 = 1.0 / (e + 1.0);

    CHECK_NEAR(O[0], p0 * 1 + p1 * 3, 1e-6, "2x2 softmax O[0][0]");
    CHECK_NEAR(O[1], p0 * 2 + p1 * 4, 1e-6, "2x2 softmax O[0][1]");
    CHECK_NEAR(O[2], p1 * 1 + p0 * 3, 1e-6, "2x2 softmax O[1][0]");
    CHECK_NEAR(O[3], p1 * 2 + p0 * 4, 1e-6, "2x2 softmax O[1][1]");

    // Direction of the asymmetry: catches a transposed score matrix.
    CHECK(O[0] < 2.0f, "query 0 attends mostly to key 0 (output pulled toward V[0])");
    CHECK(O[2] > 2.0f, "query 1 attends mostly to key 1 (output pulled toward V[1])");
}

// Softmax rows sum to 1, so with every V row identical, O must equal V exactly.
static void test_rows_sum_to_one_via_constant_v() {
    const int B = 2, H = 3, S = 9, D = 4;
    auto Q = ladder::make_qkv(B, H, S, D, 11);
    auto K = ladder::make_qkv(B, H, S, D, 22);
    std::vector<float> V(static_cast<std::size_t>(B) * H * S * D);
    for (std::size_t i = 0; i < V.size(); ++i) V[i] = 1.0f + static_cast<float>(i % D);

    std::vector<float> O(V.size(), -999.0f);
    for (int causal = 0; causal < 2; ++causal) {
        ladder::attention_cpu(B, H, S, D, ladder::attention_scale(D), causal != 0,
                              Q.data(), K.data(), V.data(), O.data());
        double worst = 0.0;
        for (std::size_t i = 0; i < O.size(); ++i)
            worst = std::max(worst, static_cast<double>(std::fabs(O[i] - V[i])));
        CHECK_NEAR(worst, 0.0, 1e-6,
                   causal ? "constant V: causal rows still sum to 1"
                          : "constant V: rows sum to 1");
    }
}

// Causal: query 0 sees only key 0, so its output is exactly V row 0.
static void test_causal_first_row_is_v0() {
    const int B = 1, H = 1, S = 2, D = 2;
    std::vector<float> Q = {0, 0, 0, 0};
    std::vector<float> K = {1, 2, 3, 4};
    std::vector<float> V = {1, 2, 3, 4};
    std::vector<float> O(4, -999.0f);

    ladder::attention_cpu(B, H, S, D, ladder::attention_scale(D), /*causal=*/true,
                          Q.data(), K.data(), V.data(), O.data());

    CHECK_NEAR(O[0], 1.0f, 1e-6, "causal: query 0 sees only key 0 -> O[0] = V[0]");
    CHECK_NEAR(O[1], 2.0f, 1e-6, "causal: query 0 sees only key 0 -> O[0] = V[0]");
    CHECK_NEAR(O[2], 2.0f, 1e-6, "causal: query 1 sees keys 0,1 -> mean of V");
    CHECK_NEAR(O[3], 3.0f, 1e-6, "causal: query 1 sees keys 0,1 -> mean of V");
    CHECK(!std::isnan(O[0]) && !std::isnan(O[2]),
          "causal produces no NaN (every row has at least one visible key)");
}

// Causal must change row 0 but leave the last row identical (mask not inverted).
static void test_causal_differs_from_full() {
    const int B = 1, H = 1, S = 8, D = 4;
    auto Q = ladder::make_qkv(B, H, S, D, 101);
    auto K = ladder::make_qkv(B, H, S, D, 202);
    auto V = ladder::make_qkv(B, H, S, D, 303);
    std::vector<float> Of(Q.size()), Oc(Q.size());

    const float sc = ladder::attention_scale(D);
    ladder::attention_cpu(B, H, S, D, sc, false, Q.data(), K.data(), V.data(), Of.data());
    ladder::attention_cpu(B, H, S, D, sc, true,  Q.data(), K.data(), V.data(), Oc.data());

    bool row0_differs = false;
    for (int d = 0; d < D; ++d)
        if (std::fabs(Of[d] - Oc[d]) > 1e-5f) row0_differs = true;
    CHECK(row0_differs, "causal mask actually changes row 0");

    for (int d = 0; d < D; ++d)
        CHECK_NEAR(Oc[d], V[d], 1e-6, "causal row 0 equals V row 0");

    const std::size_t last = static_cast<std::size_t>(S - 1) * D;
    for (int d = 0; d < D; ++d)
        CHECK_NEAR(Oc[last + d], Of[last + d], 1e-6,
                   "causal last row equals unmasked last row (mask not inverted)");
}

// Causal probs: exact zeros above the diagonal, and each row sums to 1.
static void test_probs_are_masked_and_normalized() {
    const int B = 1, H = 2, S = 6, D = 3;
    auto Q = ladder::make_qkv(B, H, S, D, 7);
    auto K = ladder::make_qkv(B, H, S, D, 8);

    auto P = ladder::attention_probs_cpu(B, H, S, D, ladder::attention_scale(D),
                                         /*causal=*/true, Q.data(), K.data());
    bool upper_is_zero = true;
    double worst_rowsum_err = 0.0;
    for (int h = 0; h < H; ++h)
        for (int i = 0; i < S; ++i) {
            double sum = 0.0;
            for (int j = 0; j < S; ++j) {
                const float p = P[ladder::score_index(0, h, i, j, H, S)];
                if (j > i && p != 0.0f) upper_is_zero = false;
                sum += p;
            }
            worst_rowsum_err = std::max(worst_rowsum_err, std::fabs(sum - 1.0));
        }
    CHECK(upper_is_zero, "causal probs are exactly 0 above the diagonal");
    CHECK_NEAR(worst_rowsum_err, 0.0, 1e-6, "every causal prob row sums to 1");
}

// Raw scores ~1.4e8 overflow exp() even in double; the row-max subtraction must hold.
static void test_large_scores_do_not_overflow() {
    const int B = 1, H = 1, S = 2, D = 2;
    const float BIG = 1e4f;
    std::vector<float> Q = {BIG, BIG, BIG, BIG};
    std::vector<float> K = {BIG, BIG, 0.0f, 0.0f};
    std::vector<float> V = {1, 2, 3, 4};
    std::vector<float> O(4, -999.0f);

    ladder::attention_cpu(B, H, S, D, ladder::attention_scale(D), false,
                          Q.data(), K.data(), V.data(), O.data());

    for (int i = 0; i < 4; ++i) {
        CHECK(!std::isnan(O[i]), "no NaN with 1e4-magnitude inputs (stable softmax)");
        CHECK(!std::isinf(O[i]), "no Inf with 1e4-magnitude inputs (stable softmax)");
    }
    CHECK_NEAR(O[0], 1.0f, 1e-6, "huge scores: output collapses onto V[0]");
    CHECK_NEAR(O[1], 2.0f, 1e-6, "huge scores: output collapses onto V[0]");

    // Prove the unstable formula really fails on the same input.
    const double raw = static_cast<double>(ladder::attention_scale(D)) *
                       (static_cast<double>(BIG) * BIG * 2.0);
    CHECK(std::isinf(std::exp(raw)),
          "exp(raw score) really is inf here -- the max subtraction is load-bearing");
    CHECK(std::isnan(std::exp(raw) / (std::exp(raw) + 1.0)),
          "the unstable formula really does produce NaN on this input");
}

// exp(-big) underflows to 0; the max subtraction keeps the denominator nonzero.
static void test_large_negative_scores_do_not_underflow_to_zero_over_zero() {
    const int B = 1, H = 1, S = 3, D = 2;
    const float BIG = 1e4f;
    std::vector<float> Q = {BIG, BIG,  BIG, BIG,  BIG, BIG};
    std::vector<float> K = {0.0f, 0.0f, -BIG, -BIG, -BIG, -BIG};
    std::vector<float> V = {1, 1,  100, 100,  100, 100};
    std::vector<float> O(static_cast<std::size_t>(S) * D, -999.0f);

    ladder::attention_cpu(B, H, S, D, ladder::attention_scale(D), false,
                          Q.data(), K.data(), V.data(), O.data());
    for (std::size_t i = 0; i < O.size(); ++i) {
        CHECK(!std::isnan(O[i]), "no NaN when most weights underflow to zero");
        CHECK_NEAR(O[i], 1.0f, 1e-6, "all mass on key 0 -> output is V[0]");
    }
}

static void test_flop_counting() {
    CHECK_NEAR(ladder::attention_flops(2, 3, 5, 7, false), 4200.0, 1e-9,
               "attention_flops = 4*B*H*S*S*D");

    CHECK_NEAR(ladder::attention_flops(2, 3, 5, 7, true), 2520.0, 1e-9,
               "causal attention_flops uses S(S+1)/2, not S*S");

    const double ratio = ladder::attention_flops(1, 1, 4096, 64, true) /
                         ladder::attention_flops(1, 1, 4096, 64, false);
    CHECK(ratio > 0.5 && ratio < 0.5002, "causal is ~half the work at S=4096");
}

static void test_byte_counting_and_intensity() {
    CHECK_NEAR(ladder::attention_ideal_bytes(1, 1, 8, 4), 512.0, 1e-9,
               "ideal bytes = 4*B*H*S*D*dtype");

    CHECK_NEAR(ladder::attention_naive_bytes(1, 1, 8, 4), 1536.0, 1e-9,
               "naive bytes include four passes over the S x S matrix");

    CHECK(ladder::attention_naive_bytes(1, 1, 1024, 64) >
          ladder::attention_ideal_bytes(1, 1, 1024, 64) * 10.0,
          "at S=1024 the naive version moves >10x the ideal traffic");

    const double ai_d64  = ladder::attention_intensity_ideal(1, 1, 1024, 64,  false, 4);
    const double ai_d128 = ladder::attention_intensity_ideal(1, 1, 1024, 128, false, 4);
    CHECK_NEAR(ai_d64, 1024.0 / 4.0, 1e-9, "fused AI = S/dtype_bytes");
    CHECK_NEAR(ai_d64, ai_d128, 1e-9, "fused AI is independent of head_dim");

    const double nai_1k  = ladder::attention_intensity_naive(1, 1, 1024,  64, false, 4);
    const double nai_16k = ladder::attention_intensity_naive(1, 1, 16384, 64, false, 4);
    CHECK(nai_1k  < 64.0 / 4.0, "naive AI stays below D/dtype_bytes");
    CHECK(nai_16k < 64.0 / 4.0, "naive AI stays below D/dtype_bytes even at S=16384");
    CHECK(nai_16k > nai_1k * 0.9 && nai_16k < nai_1k * 1.2,
          "naive AI is flat in S -- longer sequences do not help it");

    CHECK_NEAR(ladder::score_matrix_bytes(1, 1, 4096), 67108864.0, 1e-9,
               "one 4096x4096 fp32 score matrix is 67.1 MB");
    CHECK_NEAR(ladder::score_matrix_bytes(4, 8, 4096), 4.0 * 8.0 * 67108864.0, 1e-3,
               "B=4,H=32 heads of 4096x4096 scores is 2.1 GB");
}

static void test_scale_and_layout_helpers() {
    CHECK_NEAR(ladder::attention_scale(64), 0.125f, 1e-7, "scale(64) = 1/8");
    CHECK_NEAR(ladder::attention_scale(128), 1.0f / std::sqrt(128.0f), 1e-7,
               "scale(128) = 1/sqrt(128)");

    CHECK(ladder::qkv_index(1, 2, 3, 1, 4, 5, 3) == 100u, "qkv_index arithmetic");
    CHECK(ladder::score_index(1, 2, 3, 4, 4, 5) == 169u, "score_index arithmetic");
}

// Distinct B, H, S, D catch index-order mixups that B=H=S=D would hide.
static void test_ragged_shape_runs_and_is_finite() {
    const int B = 2, H = 3, S = 5, D = 4;
    auto Q = ladder::make_qkv(B, H, S, D, 1);
    auto K = ladder::make_qkv(B, H, S, D, 2);
    auto V = ladder::make_qkv(B, H, S, D, 3);
    std::vector<float> O(Q.size(), std::numeric_limits<float>::quiet_NaN());

    ladder::attention_cpu(B, H, S, D, ladder::attention_scale(D), true,
                          Q.data(), K.data(), V.data(), O.data());
    bool all_finite = true;
    for (float v : O) if (std::isnan(v) || std::isinf(v)) all_finite = false;
    CHECK(all_finite, "B!=H!=S!=D shape produces finite output everywhere");

    // Perturb V, not K: a constant K shift is absorbed by softmax and proves nothing.
    auto V2 = V;
    const std::size_t h1_base = ladder::qkv_index(0, 1, 0, 0, H, S, D);
    for (int i = 0; i < S * D; ++i) V2[h1_base + i] += 5.0f;   // perturb head 1 only
    std::vector<float> O2(Q.size());
    ladder::attention_cpu(B, H, S, D, ladder::attention_scale(D), true,
                          Q.data(), K.data(), V2.data(), O2.data());
    bool head0_unchanged = true, head1_changed = false;
    for (int i = 0; i < S * D; ++i) {
        if (std::fabs(O[i] - O2[i]) > 1e-6f) head0_unchanged = false;
        if (std::fabs(O[h1_base + i] - O2[h1_base + i]) > 1e-6f) head1_changed = true;
    }
    CHECK(head0_unchanged, "perturbing head 1 leaves head 0 untouched");
    CHECK(head1_changed,   "perturbing head 1 does change head 1");
}

int main() {
    std::printf("\n=== attention oracle host tests (no GPU required) ===\n\n");

    std::printf("[arithmetic]\n");
    test_uniform_attention_is_the_mean_of_v();
    test_two_by_two_against_closed_form();
    test_rows_sum_to_one_via_constant_v();

    std::printf("[causal]\n");
    test_causal_first_row_is_v0();
    test_causal_differs_from_full();
    test_probs_are_masked_and_normalized();

    std::printf("[stability]\n");
    test_large_scores_do_not_overflow();
    test_large_negative_scores_do_not_underflow_to_zero_over_zero();

    std::printf("[accounting]\n");
    test_flop_counting();
    test_byte_counting_and_intensity();
    test_scale_and_layout_helpers();

    std::printf("[shapes]\n");
    test_ragged_shape_runs_and_is_finite();

    std::printf("\n%d checks, %d failures\n", g_checks, g_failures);
    if (g_failures == 0) { std::printf("PASS\n\n"); return 0; }
    std::printf("FAIL\n\n");
    return 1;
}
