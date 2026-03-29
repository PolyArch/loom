#include "loom/SystemCompiler/ConvergenceTracker.h"

#include <algorithm>

namespace loom {

ConvergenceTracker::ConvergenceTracker(unsigned maxIterations,
                                       unsigned stallWindow)
    : maxIterations_(maxIterations), stallWindow_(stallWindow) {}

void ConvergenceTracker::recordIteration(unsigned iter, unsigned numNewCuts,
                                         double objective, bool allMapped) {
  IterationRecord rec;
  rec.iterationIndex = iter;
  rec.numNewCuts = numNewCuts;
  rec.objective = objective;
  rec.allMapped = allMapped;
  history_.push_back(rec);
}

void ConvergenceTracker::recordSuccess(unsigned iter, double objective,
                                       unsigned resultTag) {
  if (objective < bestObjective_) {
    bestObjective_ = objective;
    bestResultTag_ = resultTag;
  }
  hasSolution_ = true;
}

bool ConvergenceTracker::isStalled() const {
  if (history_.size() < stallWindow_)
    return false;

  // Check the last stallWindow_ iterations.
  size_t startIdx = history_.size() - stallWindow_;

  // Find the best objective before the stall window.
  double priorBest = std::numeric_limits<double>::max();
  for (size_t i = 0; i < startIdx; ++i) {
    if (history_[i].objective < priorBest)
      priorBest = history_[i].objective;
  }

  // Also consider the first entry in the window as the baseline
  // (relevant when stallWindow covers the entire history).
  if (startIdx == 0 && !history_.empty()) {
    priorBest = history_[0].objective;
    startIdx = 1;
    // Need at least stallWindow entries total.
    if (history_.size() < stallWindow_)
      return false;
    startIdx = history_.size() - stallWindow_;
  }

  // Check if any iteration in the window improved the objective or added cuts.
  for (size_t i = history_.size() - stallWindow_; i < history_.size(); ++i) {
    if (history_[i].objective < priorBest)
      return false; // Objective improved.
    if (history_[i].numNewCuts > 0)
      return false; // New cuts were generated.
  }

  return true;
}

bool ConvergenceTracker::isConverged() const {
  if (history_.empty())
    return false;
  const auto &last = history_.back();
  return last.allMapped && last.numNewCuts == 0;
}

bool ConvergenceTracker::hasSolution() const { return hasSolution_; }

} // namespace loom
