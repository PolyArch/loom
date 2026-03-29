//===-- ParetoTracker.cpp - Pareto frontier tracking ---------------*- C++ -*-===//
//
// Implements the ParetoTracker class: dominance checking, non-dominated
// archive maintenance, 2-D hypervolume indicator, and CSV/JSON export.
//
//===----------------------------------------------------------------------===//

#include "tapestry/ParetoTracker.h"

#include "llvm/Support/Format.h"

#include <algorithm>
#include <cmath>

namespace tapestry {

//===----------------------------------------------------------------------===//
// Dominance predicate
//===----------------------------------------------------------------------===//

bool ParetoTracker::dominates(const ParetoPoint &a, const ParetoPoint &b) {
  return (a.throughput >= b.throughput && a.area <= b.area) &&
         (a.throughput > b.throughput || a.area < b.area);
}

//===----------------------------------------------------------------------===//
// isDominated
//===----------------------------------------------------------------------===//

bool ParetoTracker::isDominated(const ParetoPoint &candidate) const {
  for (const auto &existing : archive_) {
    if (dominates(existing, candidate))
      return true;
  }
  return false;
}

//===----------------------------------------------------------------------===//
// addPoint
//===----------------------------------------------------------------------===//

void ParetoTracker::addPoint(const ParetoPoint &candidate) {
  // Check if any existing point dominates the candidate.
  if (isDominated(candidate))
    return;

  // Remove existing points dominated by the candidate.
  archive_.erase(
      std::remove_if(archive_.begin(), archive_.end(),
                     [&candidate](const ParetoPoint &p) {
                       return dominates(candidate, p);
                     }),
      archive_.end());

  archive_.push_back(candidate);
}

//===----------------------------------------------------------------------===//
// frontier
//===----------------------------------------------------------------------===//

std::vector<ParetoPoint> ParetoTracker::frontier() const {
  std::vector<ParetoPoint> sorted = archive_;
  std::sort(sorted.begin(), sorted.end(),
            [](const ParetoPoint &a, const ParetoPoint &b) {
              if (a.throughput != b.throughput)
                return a.throughput < b.throughput;
              return a.area < b.area;
            });
  return sorted;
}

//===----------------------------------------------------------------------===//
// hypervolume (2-D sweep-line)
//===----------------------------------------------------------------------===//

HypervolumeResult
ParetoTracker::hypervolume(const ParetoPoint &refPoint) const {
  HypervolumeResult result;
  result.refPoint = refPoint;

  std::vector<ParetoPoint> front = frontier();
  result.numPoints = static_cast<unsigned>(front.size());

  if (front.empty()) {
    result.value = 0.0;
    return result;
  }

  // Sweep left-to-right by throughput ascending.
  // Each point contributes a rectangle:
  //   width  = throughput[i] - prevThroughput  (where prevThroughput starts
  //            at refPoint.throughput)
  //   height = refPoint.area - point.area
  double hv = 0.0;
  double prevThroughput = refPoint.throughput;

  for (const auto &pt : front) {
    double width = pt.throughput - prevThroughput;
    double height = refPoint.area - pt.area;
    if (width > 0.0 && height > 0.0)
      hv += width * height;
    prevThroughput = pt.throughput;
  }

  result.value = hv;
  return result;
}

//===----------------------------------------------------------------------===//
// exportCSV
//===----------------------------------------------------------------------===//

void ParetoTracker::exportCSV(llvm::raw_ostream &os) const {
  std::vector<ParetoPoint> front = frontier();

  os << "throughput,area,round\n";
  for (const auto &pt : front) {
    os << llvm::format("%g", pt.throughput) << ","
       << llvm::format("%g", pt.area) << "," << pt.round << "\n";
  }
}

//===----------------------------------------------------------------------===//
// exportJSON
//===----------------------------------------------------------------------===//

void ParetoTracker::exportJSON(llvm::raw_ostream &os) const {
  std::vector<ParetoPoint> front = frontier();

  os << "[\n";
  for (size_t idx = 0; idx < front.size(); ++idx) {
    const auto &pt = front[idx];
    os << "  {\"throughput\": " << llvm::format("%g", pt.throughput)
       << ", \"area\": " << llvm::format("%g", pt.area)
       << ", \"round\": " << pt.round << "}";
    if (idx + 1 < front.size())
      os << ",";
    os << "\n";
  }
  os << "]\n";
}

//===----------------------------------------------------------------------===//
// clear
//===----------------------------------------------------------------------===//

void ParetoTracker::clear() { archive_.clear(); }

//===----------------------------------------------------------------------===//
// compareToBaseline
//===----------------------------------------------------------------------===//

double ParetoTracker::compareToBaseline(const ParetoTracker &baseline,
                                        const ParetoPoint &refPoint) const {
  double baseHV = baseline.hypervolume(refPoint).value;
  if (baseHV <= 0.0)
    return 0.0;
  return hypervolume(refPoint).value / baseHV;
}

} // namespace tapestry
