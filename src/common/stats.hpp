// Timing statistics (percentiles over repeated samples). Pure C++, no CUDA.

#pragma once

#include <vector>
#include <algorithm>
#include <cmath>
#include <string>

namespace ladder {

struct TimingStats {
    double min_ms    = 0.0;
    double p50_ms    = 0.0;
    double p90_ms    = 0.0;
    double p99_ms    = 0.0;
    double max_ms    = 0.0;
    double mean_ms   = 0.0;
    double stddev_ms = 0.0;
    int    n         = 0;

    // Coefficient of variation; > ~0.05 indicates a noisy measurement.
    double cv() const { return mean_ms > 0.0 ? stddev_ms / mean_ms : 0.0; }

    // Median's relative distance above the best run; measurement health check.
    double spread() const { return min_ms > 0.0 ? (p50_ms - min_ms) / min_ms : 0.0; }
};

// Linear-interpolated percentile (sorts a copy).
inline double percentile(std::vector<double> sorted_or_not, double q) {
    if (sorted_or_not.empty()) return 0.0;
    std::sort(sorted_or_not.begin(), sorted_or_not.end());
    if (sorted_or_not.size() == 1) return sorted_or_not[0];
    const double pos   = q * (static_cast<double>(sorted_or_not.size()) - 1.0);
    const std::size_t lo = static_cast<std::size_t>(std::floor(pos));
    const std::size_t hi = static_cast<std::size_t>(std::ceil(pos));
    const double frac  = pos - static_cast<double>(lo);
    return sorted_or_not[lo] * (1.0 - frac) + sorted_or_not[hi] * frac;
}

inline TimingStats summarize(std::vector<double> samples_ms) {
    TimingStats s;
    if (samples_ms.empty()) return s;
    s.n = static_cast<int>(samples_ms.size());

    std::sort(samples_ms.begin(), samples_ms.end());
    s.min_ms = samples_ms.front();
    s.max_ms = samples_ms.back();
    s.p50_ms = percentile(samples_ms, 0.50);
    s.p90_ms = percentile(samples_ms, 0.90);
    s.p99_ms = percentile(samples_ms, 0.99);

    double sum = 0.0;
    for (double v : samples_ms) sum += v;
    s.mean_ms = sum / static_cast<double>(s.n);

    // Sample stddev (n-1 correction); 0 when n == 1 since variance is undefined.
    if (s.n > 1) {
        double sq = 0.0;
        for (double v : samples_ms) {
            const double d = v - s.mean_ms;
            sq += d * d;
        }
        s.stddev_ms = std::sqrt(sq / static_cast<double>(s.n - 1));
    }
    return s;
}

// Time is passed explicitly: the caller chooses which statistic to convert.
inline double gflops(double total_flops, double time_ms) {
    return time_ms > 0.0 ? (total_flops / (time_ms * 1e-3)) / 1e9 : 0.0;
}

inline double gbytes_per_s(double total_bytes, double time_ms) {
    return time_ms > 0.0 ? (total_bytes / (time_ms * 1e-3)) / 1e9 : 0.0;
}

}  // namespace ladder
