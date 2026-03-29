#ifndef LOOM_SYSTEMCOMPILER_CONVERGENCETRACKER_H
#define LOOM_SYSTEMCOMPILER_CONVERGENCETRACKER_H

#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace loom {

/// Record of a single bilevel loop iteration.
struct IterationRecord {
  unsigned iterationIndex = 0;
  unsigned numNewCuts = 0;
  double objective = std::numeric_limits<double>::max();
  bool allMapped = false;
};

/// Tracks convergence state for the hierarchical bilevel compilation loop.
///
/// Maintains a history of iteration outcomes, the best-known objective value,
/// and provides stall/convergence detection queries.
class ConvergenceTracker {
public:
  /// Construct with the maximum iteration budget and the stall window size.
  /// Stall is detected when the last `stallWindow` iterations show no
  /// objective improvement and no new cut types.
  ConvergenceTracker(unsigned maxIterations, unsigned stallWindow);

  /// Record an iteration's outcome.
  void recordIteration(unsigned iter, unsigned numNewCuts, double objective,
                       bool allMapped);

  /// Record a successful iteration and update the best solution if improved.
  /// The `resultTag` is an opaque index the caller can use to identify which
  /// iteration produced the best result.
  void recordSuccess(unsigned iter, double objective, unsigned resultTag);

  /// Return true if the last `stallWindow` iterations produced no objective
  /// improvement AND no new cuts.
  bool isStalled() const;

  /// Return true if the most recent iteration had zero new cuts and all
  /// kernels mapped successfully.
  bool isConverged() const;

  /// Return true if at least one successful solution was recorded.
  bool hasSolution() const;

  /// Return the best objective value seen, or max double if none.
  double getBestObjective() const { return bestObjective_; }

  /// Return the result tag of the best solution.
  unsigned getBestResultTag() const { return bestResultTag_; }

  /// Return the iteration history.
  const std::vector<IterationRecord> &history() const { return history_; }

  unsigned getMaxIterations() const { return maxIterations_; }

private:
  unsigned maxIterations_;
  unsigned stallWindow_;
  std::vector<IterationRecord> history_;
  double bestObjective_ = std::numeric_limits<double>::max();
  unsigned bestResultTag_ = 0;
  bool hasSolution_ = false;
};

} // namespace loom

#endif // LOOM_SYSTEMCOMPILER_CONVERGENCETRACKER_H
