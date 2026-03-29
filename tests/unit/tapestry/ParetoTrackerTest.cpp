/// ParetoTracker unit tests.
///
/// Tests:
///  T1  - Dominance: basic case (both dimensions better)
///  T2  - Dominance: equal in one dimension
///  T3  - Dominance: equal in both dimensions (non-reflexive)
///  T4  - Dominance: incomparable (trade-off)
///  T5  - addPoint: dominated candidate rejected
///  T6  - addPoint: new point dominates existing
///  T7  - addPoint: incomparable point expands frontier
///  T8  - addPoint: batch insertion builds correct front
///  T9  - Hypervolume: known 3-point set
///  T10 - Hypervolume: single-point frontier
///  T11 - Hypervolume: empty frontier
///  T12 - Frontier extraction returns sorted order
///  T13 - CSV export format
///  T14 - JSON export format
///  T15 - isDominated query

#include "tapestry/ParetoTracker.h"

#include "llvm/Support/raw_ostream.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <string>

using namespace tapestry;

static constexpr double kEps = 1e-9;

static bool approxEq(double a, double b) {
  return std::fabs(a - b) < kEps;
}

static ParetoPoint mkPt(double throughput, double area, unsigned round = 0) {
  ParetoPoint p;
  p.throughput = throughput;
  p.area = area;
  p.round = round;
  return p;
}

// -------------------------------------------------------------------------
// T1: Dominance -- basic (A better in both dimensions)
// -------------------------------------------------------------------------
static bool testDominanceBasic() {
  auto a = mkPt(10, 5);
  auto b = mkPt(8, 7);
  if (!ParetoTracker::dominates(a, b)) {
    std::cerr << "FAIL: testDominanceBasic - A should dominate B\n";
    return false;
  }
  if (ParetoTracker::dominates(b, a)) {
    std::cerr << "FAIL: testDominanceBasic - B should not dominate A\n";
    return false;
  }
  std::cout << "PASS: T1 testDominanceBasic\n";
  return true;
}

// -------------------------------------------------------------------------
// T2: Dominance -- equal in one dimension
// -------------------------------------------------------------------------
static bool testDominanceEqualOne() {
  auto a = mkPt(10, 5);
  auto b = mkPt(10, 7);
  if (!ParetoTracker::dominates(a, b)) {
    std::cerr << "FAIL: testDominanceEqualOne - A should dominate B "
              << "(equal throughput, strictly better area)\n";
    return false;
  }
  std::cout << "PASS: T2 testDominanceEqualOne\n";
  return true;
}

// -------------------------------------------------------------------------
// T3: Dominance -- equal in both dimensions (non-reflexive)
// -------------------------------------------------------------------------
static bool testDominanceEqualBoth() {
  auto a = mkPt(10, 5);
  auto b = mkPt(10, 5);
  if (ParetoTracker::dominates(a, b)) {
    std::cerr << "FAIL: testDominanceEqualBoth - should not dominate "
              << "when equal in both\n";
    return false;
  }
  std::cout << "PASS: T3 testDominanceEqualBoth\n";
  return true;
}

// -------------------------------------------------------------------------
// T4: Dominance -- incomparable (trade-off)
// -------------------------------------------------------------------------
static bool testDominanceIncomparable() {
  auto a = mkPt(10, 7);
  auto b = mkPt(8, 5);
  if (ParetoTracker::dominates(a, b)) {
    std::cerr << "FAIL: testDominanceIncomparable - A should not dominate B\n";
    return false;
  }
  if (ParetoTracker::dominates(b, a)) {
    std::cerr << "FAIL: testDominanceIncomparable - B should not dominate A\n";
    return false;
  }
  std::cout << "PASS: T4 testDominanceIncomparable\n";
  return true;
}

// -------------------------------------------------------------------------
// T5: addPoint -- dominated candidate is rejected
// -------------------------------------------------------------------------
static bool testAddPointDominated() {
  ParetoTracker tracker;
  tracker.addPoint(mkPt(10, 5, 1));
  tracker.addPoint(mkPt(8, 7, 2));

  if (tracker.size() != 1) {
    std::cerr << "FAIL: testAddPointDominated - size=" << tracker.size()
              << " (expected 1)\n";
    return false;
  }
  auto front = tracker.frontier();
  if (!approxEq(front[0].throughput, 10) || !approxEq(front[0].area, 5)) {
    std::cerr << "FAIL: testAddPointDominated - wrong surviving point\n";
    return false;
  }
  std::cout << "PASS: T5 testAddPointDominated\n";
  return true;
}

// -------------------------------------------------------------------------
// T6: addPoint -- new point dominates existing
// -------------------------------------------------------------------------
static bool testAddPointDominatesExisting() {
  ParetoTracker tracker;
  tracker.addPoint(mkPt(8, 7, 1));
  tracker.addPoint(mkPt(10, 5, 2));

  if (tracker.size() != 1) {
    std::cerr << "FAIL: testAddPointDominatesExisting - size="
              << tracker.size() << " (expected 1)\n";
    return false;
  }
  auto front = tracker.frontier();
  if (!approxEq(front[0].throughput, 10) || !approxEq(front[0].area, 5)) {
    std::cerr << "FAIL: testAddPointDominatesExisting - wrong point\n";
    return false;
  }
  if (front[0].round != 2) {
    std::cerr << "FAIL: testAddPointDominatesExisting - wrong round\n";
    return false;
  }
  std::cout << "PASS: T6 testAddPointDominatesExisting\n";
  return true;
}

// -------------------------------------------------------------------------
// T7: addPoint -- incomparable point expands frontier
// -------------------------------------------------------------------------
static bool testAddPointIncomparable() {
  ParetoTracker tracker;
  tracker.addPoint(mkPt(10, 7, 1));
  tracker.addPoint(mkPt(8, 5, 2));

  if (tracker.size() != 2) {
    std::cerr << "FAIL: testAddPointIncomparable - size=" << tracker.size()
              << " (expected 2)\n";
    return false;
  }
  auto front = tracker.frontier();
  // Sorted by throughput ascending: {8,5,2}, {10,7,1}.
  if (!approxEq(front[0].throughput, 8) || !approxEq(front[0].area, 5)) {
    std::cerr << "FAIL: testAddPointIncomparable - front[0] wrong\n";
    return false;
  }
  if (!approxEq(front[1].throughput, 10) || !approxEq(front[1].area, 7)) {
    std::cerr << "FAIL: testAddPointIncomparable - front[1] wrong\n";
    return false;
  }
  std::cout << "PASS: T7 testAddPointIncomparable\n";
  return true;
}

// -------------------------------------------------------------------------
// T8: addPoint -- batch insertion builds correct front
// -------------------------------------------------------------------------
static bool testAddPointBatch() {
  ParetoTracker tracker;
  tracker.addPoint(mkPt(5, 10, 1));
  tracker.addPoint(mkPt(10, 5, 2));
  tracker.addPoint(mkPt(7, 8, 3));
  tracker.addPoint(mkPt(8, 6, 4));
  tracker.addPoint(mkPt(6, 11, 5));
  tracker.addPoint(mkPt(9, 4, 6));

  // Non-dominated survivors: {9,4,6} and {10,5,2}.
  auto front = tracker.frontier();
  if (front.size() != 2) {
    std::cerr << "FAIL: testAddPointBatch - size=" << front.size()
              << " (expected 2)\n";
    for (const auto &p : front)
      std::cerr << "  {" << p.throughput << ", " << p.area << ", "
                << p.round << "}\n";
    return false;
  }
  // Sorted by throughput ascending: {9,4,6}, {10,5,2}.
  if (!approxEq(front[0].throughput, 9) || !approxEq(front[0].area, 4)) {
    std::cerr << "FAIL: testAddPointBatch - front[0] wrong\n";
    return false;
  }
  if (!approxEq(front[1].throughput, 10) || !approxEq(front[1].area, 5)) {
    std::cerr << "FAIL: testAddPointBatch - front[1] wrong\n";
    return false;
  }
  std::cout << "PASS: T8 testAddPointBatch\n";
  return true;
}

// -------------------------------------------------------------------------
// T9: Hypervolume on known set
// -------------------------------------------------------------------------
static bool testHypervolumeKnown() {
  ParetoTracker tracker;
  // Use a proper non-dominated front where area increases with throughput
  // (natural trade-off for max-throughput / min-area).
  tracker.addPoint(mkPt(1, 1));
  tracker.addPoint(mkPt(2, 2));
  tracker.addPoint(mkPt(3, 3));

  auto ref = mkPt(0, 4);
  auto hv = tracker.hypervolume(ref);

  // Sweep left-to-right by throughput ascending:
  //   {1,1}: (1-0)*(4-1) = 3
  //   {2,2}: (2-1)*(4-2) = 2
  //   {3,3}: (3-2)*(4-3) = 1
  // Total = 6
  if (!approxEq(hv.value, 6.0)) {
    std::cerr << "FAIL: testHypervolumeKnown - hv=" << hv.value
              << " (expected 6)\n";
    return false;
  }
  if (hv.numPoints != 3) {
    std::cerr << "FAIL: testHypervolumeKnown - numPoints=" << hv.numPoints
              << " (expected 3)\n";
    return false;
  }
  std::cout << "PASS: T9 testHypervolumeKnown\n";
  return true;
}

// -------------------------------------------------------------------------
// T10: Hypervolume on single-point frontier
// -------------------------------------------------------------------------
static bool testHypervolumeSingle() {
  ParetoTracker tracker;
  tracker.addPoint(mkPt(5, 3));

  auto ref = mkPt(0, 10);
  auto hv = tracker.hypervolume(ref);

  // Expected: 5 * (10 - 3) = 35
  if (!approxEq(hv.value, 35.0)) {
    std::cerr << "FAIL: testHypervolumeSingle - hv=" << hv.value
              << " (expected 35)\n";
    return false;
  }
  std::cout << "PASS: T10 testHypervolumeSingle\n";
  return true;
}

// -------------------------------------------------------------------------
// T11: Hypervolume on empty frontier
// -------------------------------------------------------------------------
static bool testHypervolumeEmpty() {
  ParetoTracker tracker;

  auto ref = mkPt(0, 10);
  auto hv = tracker.hypervolume(ref);

  if (!approxEq(hv.value, 0.0)) {
    std::cerr << "FAIL: testHypervolumeEmpty - hv=" << hv.value
              << " (expected 0)\n";
    return false;
  }
  if (hv.numPoints != 0) {
    std::cerr << "FAIL: testHypervolumeEmpty - numPoints=" << hv.numPoints
              << " (expected 0)\n";
    return false;
  }
  std::cout << "PASS: T11 testHypervolumeEmpty\n";
  return true;
}

// -------------------------------------------------------------------------
// T12: Frontier extraction returns sorted order
// -------------------------------------------------------------------------
static bool testFrontierSorted() {
  ParetoTracker tracker;
  // Use points that form a true trade-off (none dominates another):
  //   {3, 2}: low throughput, very low area
  //   {7, 6}: medium throughput, medium area
  //   {10, 9}: high throughput, high area
  tracker.addPoint(mkPt(10, 9));
  tracker.addPoint(mkPt(3, 2));
  tracker.addPoint(mkPt(7, 6));

  auto front = tracker.frontier();
  if (front.size() != 3) {
    std::cerr << "FAIL: testFrontierSorted - size=" << front.size()
              << " (expected 3)\n";
    return false;
  }
  // Sorted by throughput ascending: {3,2}, {7,6}, {10,9}.
  if (!approxEq(front[0].throughput, 3) ||
      !approxEq(front[1].throughput, 7) ||
      !approxEq(front[2].throughput, 10)) {
    std::cerr << "FAIL: testFrontierSorted - wrong sort order\n";
    return false;
  }
  std::cout << "PASS: T12 testFrontierSorted\n";
  return true;
}

// -------------------------------------------------------------------------
// T13: CSV export format
// -------------------------------------------------------------------------
static bool testExportCSV() {
  ParetoTracker tracker;
  tracker.addPoint(mkPt(10, 5, 1));
  tracker.addPoint(mkPt(8, 3, 2));

  std::string output;
  llvm::raw_string_ostream os(output);
  tracker.exportCSV(os);
  os.flush();

  std::string expected = "throughput,area,round\n"
                         "8,3,2\n"
                         "10,5,1\n";
  if (output != expected) {
    std::cerr << "FAIL: testExportCSV\n"
              << "  expected: [" << expected << "]\n"
              << "  got:      [" << output << "]\n";
    return false;
  }
  std::cout << "PASS: T13 testExportCSV\n";
  return true;
}

// -------------------------------------------------------------------------
// T14: JSON export format
// -------------------------------------------------------------------------
static bool testExportJSON() {
  ParetoTracker tracker;
  tracker.addPoint(mkPt(10, 5, 1));

  std::string output;
  llvm::raw_string_ostream os(output);
  tracker.exportJSON(os);
  os.flush();

  // Verify structural correctness: starts with '[', contains key fields.
  if (output.find('[') == std::string::npos) {
    std::cerr << "FAIL: testExportJSON - missing '['\n";
    return false;
  }
  if (output.find(']') == std::string::npos) {
    std::cerr << "FAIL: testExportJSON - missing ']'\n";
    return false;
  }
  if (output.find("\"throughput\"") == std::string::npos) {
    std::cerr << "FAIL: testExportJSON - missing 'throughput' key\n";
    return false;
  }
  if (output.find("\"area\"") == std::string::npos) {
    std::cerr << "FAIL: testExportJSON - missing 'area' key\n";
    return false;
  }
  if (output.find("\"round\"") == std::string::npos) {
    std::cerr << "FAIL: testExportJSON - missing 'round' key\n";
    return false;
  }
  // Verify values appear.
  if (output.find("10") == std::string::npos) {
    std::cerr << "FAIL: testExportJSON - missing throughput value 10\n";
    return false;
  }
  if (output.find("5") == std::string::npos) {
    std::cerr << "FAIL: testExportJSON - missing area value 5\n";
    return false;
  }
  std::cout << "PASS: T14 testExportJSON\n";
  return true;
}

// -------------------------------------------------------------------------
// T15: isDominated query
// -------------------------------------------------------------------------
static bool testIsDominated() {
  ParetoTracker tracker;
  tracker.addPoint(mkPt(10, 5, 1));
  tracker.addPoint(mkPt(6, 3, 2));

  // {7,6,3} is dominated by {10,5,1}
  if (!tracker.isDominated(mkPt(7, 6, 3))) {
    std::cerr << "FAIL: testIsDominated - {7,6} should be dominated\n";
    return false;
  }
  // {11,2,4} is NOT dominated (better in both dimensions than everything)
  if (tracker.isDominated(mkPt(11, 2, 4))) {
    std::cerr << "FAIL: testIsDominated - {11,2} should not be dominated\n";
    return false;
  }
  // {4,2,5} is NOT dominated (incomparable: worse throughput, better area
  // than {6,3})
  if (tracker.isDominated(mkPt(4, 2, 5))) {
    std::cerr << "FAIL: testIsDominated - {4,2} should not be dominated\n";
    return false;
  }
  std::cout << "PASS: T15 testIsDominated\n";
  return true;
}

// -------------------------------------------------------------------------
// main
// -------------------------------------------------------------------------
int main() {
  int failures = 0;

  if (!testDominanceBasic())       ++failures;
  if (!testDominanceEqualOne())    ++failures;
  if (!testDominanceEqualBoth())   ++failures;
  if (!testDominanceIncomparable()) ++failures;
  if (!testAddPointDominated())    ++failures;
  if (!testAddPointDominatesExisting()) ++failures;
  if (!testAddPointIncomparable()) ++failures;
  if (!testAddPointBatch())        ++failures;
  if (!testHypervolumeKnown())     ++failures;
  if (!testHypervolumeSingle())    ++failures;
  if (!testHypervolumeEmpty())     ++failures;
  if (!testFrontierSorted())       ++failures;
  if (!testExportCSV())            ++failures;
  if (!testExportJSON())           ++failures;
  if (!testIsDominated())          ++failures;

  std::cout << "\n=== ParetoTracker: "
            << (15 - failures) << "/15 tests passed";
  if (failures > 0)
    std::cout << " (" << failures << " FAILED)";
  std::cout << " ===\n";

  return failures > 0 ? 1 : 0;
}
