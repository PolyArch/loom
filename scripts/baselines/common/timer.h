// High-resolution timer wrapper for CPU baseline measurements.
// Uses std::chrono steady_clock for portable, monotonic timing.

#ifndef BASELINES_COMMON_TIMER_H
#define BASELINES_COMMON_TIMER_H

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <string>
#include <vector>

namespace baselines {

// Measures wall-clock elapsed time in milliseconds.
class Timer {
public:
  void start() { start_ = std::chrono::steady_clock::now(); }

  void stop() {
    end_ = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end_ - start_).count();
    samples_.push_back(ms);
  }

  void reset() { samples_.clear(); }

  // Discard the first n samples (warm-up).
  void discard_warmup(size_t n) {
    if (n < samples_.size()) {
      samples_.erase(samples_.begin(), samples_.begin() + n);
    } else {
      samples_.clear();
    }
  }

  double mean_ms() const {
    if (samples_.empty())
      return 0.0;
    double sum = std::accumulate(samples_.begin(), samples_.end(), 0.0);
    return sum / static_cast<double>(samples_.size());
  }

  double stddev_ms() const {
    if (samples_.size() < 2)
      return 0.0;
    double m = mean_ms();
    double sq_sum = 0.0;
    for (double s : samples_) {
      sq_sum += (s - m) * (s - m);
    }
    return std::sqrt(sq_sum / static_cast<double>(samples_.size() - 1));
  }

  double min_ms() const {
    if (samples_.empty())
      return 0.0;
    return *std::min_element(samples_.begin(), samples_.end());
  }

  double max_ms() const {
    if (samples_.empty())
      return 0.0;
    return *std::max_element(samples_.begin(), samples_.end());
  }

  size_t count() const { return samples_.size(); }

  const std::vector<double> &samples() const { return samples_; }

  // Pretty-print summary.
  void print_summary(const std::string &label) const {
    std::printf("[%s] mean=%.3f ms  stddev=%.3f ms  min=%.3f ms  max=%.3f ms  "
                "(n=%zu)\n",
                label.c_str(), mean_ms(), stddev_ms(), min_ms(), max_ms(),
                samples_.size());
  }

private:
  std::chrono::steady_clock::time_point start_;
  std::chrono::steady_clock::time_point end_;
  std::vector<double> samples_;
};

// Benchmark helper: run `fn` for warmup+measure iterations, return Timer.
template <typename Fn>
Timer benchmark(Fn &&fn, size_t warmup_iters = 3, size_t measure_iters = 10) {
  Timer timer;

  // Warm-up + measurement combined.
  for (size_t i = 0; i < warmup_iters + measure_iters; ++i) {
    timer.start();
    fn();
    timer.stop();
  }

  timer.discard_warmup(warmup_iters);
  return timer;
}

} // namespace baselines

#endif // BASELINES_COMMON_TIMER_H
