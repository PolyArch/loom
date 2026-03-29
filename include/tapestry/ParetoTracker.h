//===-- ParetoTracker.h - Pareto frontier tracking ----------------*- C++ -*-===//
//
// Maintains a non-dominated archive of (throughput, area) design points.
// Provides dominance checking, front extraction, 2-D hypervolume indicator
// computation, and CSV/JSON export for downstream visualization.
//
//===----------------------------------------------------------------------===//

#ifndef TAPESTRY_PARETO_TRACKER_H
#define TAPESTRY_PARETO_TRACKER_H

#include "tapestry/co_optimizer.h"

#include "llvm/Support/raw_ostream.h"

#include <vector>

namespace tapestry {

//===----------------------------------------------------------------------===//
// HypervolumeResult
//===----------------------------------------------------------------------===//

/// Result of a 2-D hypervolume indicator computation.
struct HypervolumeResult {
  /// Scalar hypervolume value (area of dominated region).
  double value = 0.0;

  /// Reference point used for the computation.
  ParetoPoint refPoint;

  /// Number of frontier points used.
  unsigned numPoints = 0;
};

//===----------------------------------------------------------------------===//
// ParetoTracker
//===----------------------------------------------------------------------===//

/// Maintains a non-dominated archive and provides Pareto analysis utilities.
class ParetoTracker {
public:
  ParetoTracker() = default;

  /// Returns true if point A dominates point B.
  /// A dominates B iff A.throughput >= B.throughput AND A.area <= B.area,
  /// with strict inequality in at least one dimension.
  static bool dominates(const ParetoPoint &a, const ParetoPoint &b);

  /// Check if a candidate point is dominated by any point in the archive.
  bool isDominated(const ParetoPoint &candidate) const;

  /// Insert a candidate point into the archive if it is non-dominated.
  /// Removes any existing points dominated by the candidate.
  void addPoint(const ParetoPoint &candidate);

  /// Return the current non-dominated front, sorted by throughput ascending
  /// (ties broken by area ascending).
  std::vector<ParetoPoint> frontier() const;

  /// Number of points on the frontier.
  size_t size() const { return archive_.size(); }

  /// Compute 2-D hypervolume indicator relative to a reference point.
  /// The reference point should be dominated by all frontier points (i.e.,
  /// refPoint.throughput < min frontier throughput, refPoint.area > max
  /// frontier area).
  HypervolumeResult hypervolume(const ParetoPoint &refPoint) const;

  /// Write the frontier as CSV with columns: throughput,area,round.
  void exportCSV(llvm::raw_ostream &os) const;

  /// Write the frontier as a JSON array of objects.
  void exportJSON(llvm::raw_ostream &os) const;

  /// Reset the archive.
  void clear();

  /// Compare hypervolumes of two trackers and return the ratio
  /// (this / baseline). Returns 0.0 if baseline hypervolume is zero.
  double compareToBaseline(const ParetoTracker &baseline,
                           const ParetoPoint &refPoint) const;

private:
  std::vector<ParetoPoint> archive_;
};

} // namespace tapestry

#endif // TAPESTRY_PARETO_TRACKER_H
