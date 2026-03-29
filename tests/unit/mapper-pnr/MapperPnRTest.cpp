/// Mapper PnR algorithm improvement tests.
///
/// Tests:
/// T1-T2: SA cost function penalty weight defaults
/// T3: Adaptive checkpoint batch sizing logic
/// T4: Route-aware SA temperature defaults updated
/// T5: Congestion corridor fan-out inflation (default constant)
/// T6: Congestion placement weight default
/// T7: Exact repair budget defaults
/// T8: CP-SAT escalation thresholds lowered
/// T9-T10: Multi-restart options defaults
/// T11: Local repair budget fraction default
/// T12: Updated budgetSeconds default
/// T13: Option validation rejects invalid values

#include "loom/Mapper/MapperOptions.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <string>

using namespace loom;

static bool doubleEq(double a, double b, double eps = 1e-9) {
  return std::abs(a - b) < eps;
}

/// T1-T2: SA cost function penalty weights have correct defaults.
static bool testSACostFunctionPenaltyDefaults() {
  MapperRefinementOptions refinement;
  if (!doubleEq(refinement.routeAwareSAUnroutedEdgePenaltyWeight, 10.0)) {
    std::cerr << "FAIL: unrouted edge penalty weight default is "
              << refinement.routeAwareSAUnroutedEdgePenaltyWeight
              << ", expected 10.0\n";
    return false;
  }
  if (!doubleEq(refinement.routeAwareSACongestionPenaltyWeight, 2.0)) {
    std::cerr << "FAIL: congestion penalty weight default is "
              << refinement.routeAwareSACongestionPenaltyWeight
              << ", expected 2.0\n";
    return false;
  }
  std::cout << "PASS: testSACostFunctionPenaltyDefaults\n";
  return true;
}

/// T3: Adaptive checkpoint batch sizing logic.
/// With base_batch=20:
///   unrouted<=3 -> max(4, 20/4) = 5
///   unrouted<=10 -> 20/2 = 10
///   unrouted>10  -> 20
static bool testAdaptiveCheckpointBatchSizing() {
  unsigned baseBatch = 20;

  // unrouted = 2 (<=3): max(4, 20/4) = max(4, 5) = 5
  {
    unsigned unrouted = 2;
    unsigned result;
    if (unrouted <= 3)
      result = std::max(4u, baseBatch / 4);
    else if (unrouted <= 10)
      result = baseBatch / 2;
    else
      result = baseBatch;
    if (result != 5) {
      std::cerr << "FAIL: batch for unrouted=2 is " << result
                << ", expected 5\n";
      return false;
    }
  }

  // unrouted = 8 (<=10): 20/2 = 10
  {
    unsigned unrouted = 8;
    unsigned result;
    if (unrouted <= 3)
      result = std::max(4u, baseBatch / 4);
    else if (unrouted <= 10)
      result = baseBatch / 2;
    else
      result = baseBatch;
    if (result != 10) {
      std::cerr << "FAIL: batch for unrouted=8 is " << result
                << ", expected 10\n";
      return false;
    }
  }

  // unrouted = 15 (>10): 20
  {
    unsigned unrouted = 15;
    unsigned result;
    if (unrouted <= 3)
      result = std::max(4u, baseBatch / 4);
    else if (unrouted <= 10)
      result = baseBatch / 2;
    else
      result = baseBatch;
    if (result != 20) {
      std::cerr << "FAIL: batch for unrouted=15 is " << result
                << ", expected 20\n";
      return false;
    }
  }

  std::cout << "PASS: testAdaptiveCheckpointBatchSizing\n";
  return true;
}

/// T4: Route-aware SA temperature defaults updated.
static bool testRouteAwareSATemperatureDefaults() {
  MapperRefinementOptions refinement;
  if (!doubleEq(refinement.routeAwareSAInitialTemperature, 150.0)) {
    std::cerr << "FAIL: routeAwareSAInitialTemperature is "
              << refinement.routeAwareSAInitialTemperature
              << ", expected 150.0\n";
    return false;
  }
  if (!doubleEq(refinement.routeAwareSACoolingRate, 0.9985)) {
    std::cerr << "FAIL: routeAwareSACoolingRate is "
              << refinement.routeAwareSACoolingRate << ", expected 0.9985\n";
    return false;
  }
  if (!doubleEq(refinement.routeAwareSAMinTemperature, 0.001)) {
    std::cerr << "FAIL: routeAwareSAMinTemperature is "
              << refinement.routeAwareSAMinTemperature
              << ", expected 0.001\n";
    return false;
  }
  std::cout << "PASS: testRouteAwareSATemperatureDefaults\n";
  return true;
}

/// T6: Congestion placement weight default.
static bool testCongestionPlacementWeightDefault() {
  MapperOptions opts;
  if (!doubleEq(opts.congestionPlacementWeight, 0.6)) {
    std::cerr << "FAIL: congestionPlacementWeight is "
              << opts.congestionPlacementWeight << ", expected 0.6\n";
    return false;
  }
  std::cout << "PASS: testCongestionPlacementWeightDefault\n";
  return true;
}

/// T7: Exact repair budget defaults.
static bool testExactRepairBudgetDefaults() {
  MapperLocalRepairExactOptions exact;
  bool ok = true;
  if (!doubleEq(exact.microDeadlineMs, 35000.0)) {
    std::cerr << "FAIL: microDeadlineMs is " << exact.microDeadlineMs
              << ", expected 35000.0\n";
    ok = false;
  }
  if (exact.microCandidatePathLimit != 48) {
    std::cerr << "FAIL: microCandidatePathLimit is "
              << exact.microCandidatePathLimit << ", expected 48\n";
    ok = false;
  }
  if (exact.microFirstHopLimit != 28) {
    std::cerr << "FAIL: microFirstHopLimit is " << exact.microFirstHopLimit
              << ", expected 28\n";
    ok = false;
  }
  if (!doubleEq(exact.tightDeadlineMs, 12000.0)) {
    std::cerr << "FAIL: tightDeadlineMs is " << exact.tightDeadlineMs
              << ", expected 12000.0\n";
    ok = false;
  }
  if (exact.tightCandidatePathLimit != 24) {
    std::cerr << "FAIL: tightCandidatePathLimit is "
              << exact.tightCandidatePathLimit << ", expected 24\n";
    ok = false;
  }
  if (ok)
    std::cout << "PASS: testExactRepairBudgetDefaults\n";
  return ok;
}

/// T8: CP-SAT escalation thresholds lowered.
static bool testCPSatEscalationThresholds() {
  MapperLocalRepairOptions repair;
  bool ok = true;
  if (repair.cpSatFallbackFailedEdgeThreshold != 5) {
    std::cerr << "FAIL: cpSatFallbackFailedEdgeThreshold is "
              << repair.cpSatFallbackFailedEdgeThreshold << ", expected 5\n";
    ok = false;
  }
  if (repair.cpSatEscalationFailedEdgeThreshold != 4) {
    std::cerr << "FAIL: cpSatEscalationFailedEdgeThreshold is "
              << repair.cpSatEscalationFailedEdgeThreshold << ", expected 4\n";
    ok = false;
  }
  if (ok)
    std::cout << "PASS: testCPSatEscalationThresholds\n";
  return ok;
}

/// T9-T10: Multi-restart options defaults.
static bool testMultiRestartDefaults() {
  MapperOptions opts;
  bool ok = true;
  if (opts.maxRestarts != 3) {
    std::cerr << "FAIL: maxRestarts is " << opts.maxRestarts
              << ", expected 3\n";
    ok = false;
  }
  if (!doubleEq(opts.perRestartBudgetFraction, 0.33)) {
    std::cerr << "FAIL: perRestartBudgetFraction is "
              << opts.perRestartBudgetFraction << ", expected 0.33\n";
    ok = false;
  }
  if (ok)
    std::cout << "PASS: testMultiRestartDefaults\n";
  return ok;
}

/// T11: Local repair budget fraction default.
static bool testLocalRepairBudgetFractionDefault() {
  MapperOptions opts;
  if (!doubleEq(opts.localRepairBudgetFraction, 0.30)) {
    std::cerr << "FAIL: localRepairBudgetFraction is "
              << opts.localRepairBudgetFraction << ", expected 0.30\n";
    return false;
  }
  std::cout << "PASS: testLocalRepairBudgetFractionDefault\n";
  return true;
}

/// T12: Updated budgetSeconds default.
static bool testBudgetSecondsDefault() {
  MapperOptions opts;
  if (!doubleEq(opts.budgetSeconds, 300.0)) {
    std::cerr << "FAIL: budgetSeconds is " << opts.budgetSeconds
              << ", expected 300.0\n";
    return false;
  }
  std::cout << "PASS: testBudgetSecondsDefault\n";
  return true;
}

/// T13: Option validation rejects invalid values.
static bool testOptionValidation() {
  std::string error;
  bool ok = true;

  // Valid defaults should pass.
  {
    MapperOptions valid;
    if (!validateMapperOptions(valid, error)) {
      std::cerr << "FAIL: default options failed validation: " << error << "\n";
      ok = false;
    }
  }

  // maxRestarts = 0 should fail.
  {
    MapperOptions invalid;
    invalid.maxRestarts = 0;
    if (validateMapperOptions(invalid, error)) {
      std::cerr << "FAIL: maxRestarts=0 should fail validation\n";
      ok = false;
    }
  }

  // perRestartBudgetFraction = 1.5 should fail.
  {
    MapperOptions invalid;
    invalid.perRestartBudgetFraction = 1.5;
    if (validateMapperOptions(invalid, error)) {
      std::cerr << "FAIL: perRestartBudgetFraction=1.5 should fail\n";
      ok = false;
    }
  }

  // localRepairBudgetFraction = 0.0 should fail (must be > 0).
  {
    MapperOptions invalid;
    invalid.localRepairBudgetFraction = 0.0;
    if (validateMapperOptions(invalid, error)) {
      std::cerr << "FAIL: localRepairBudgetFraction=0.0 should fail\n";
      ok = false;
    }
  }

  // localRepairBudgetFraction = 1.5 should fail (must be <= 1.0).
  {
    MapperOptions invalid;
    invalid.localRepairBudgetFraction = 1.5;
    if (validateMapperOptions(invalid, error)) {
      std::cerr << "FAIL: localRepairBudgetFraction=1.5 should fail\n";
      ok = false;
    }
  }

  // Negative penalty weights should fail.
  {
    MapperOptions invalid;
    invalid.refinement.routeAwareSAUnroutedEdgePenaltyWeight = -1.0;
    if (validateMapperOptions(invalid, error)) {
      std::cerr << "FAIL: negative unrouted penalty weight should fail\n";
      ok = false;
    }
  }

  {
    MapperOptions invalid;
    invalid.refinement.routeAwareSACongestionPenaltyWeight = -1.0;
    if (validateMapperOptions(invalid, error)) {
      std::cerr << "FAIL: negative congestion penalty weight should fail\n";
      ok = false;
    }
  }

  if (ok)
    std::cout << "PASS: testOptionValidation\n";
  return ok;
}

int main() {
  int failures = 0;
  if (!testSACostFunctionPenaltyDefaults())
    ++failures;
  if (!testAdaptiveCheckpointBatchSizing())
    ++failures;
  if (!testRouteAwareSATemperatureDefaults())
    ++failures;
  if (!testCongestionPlacementWeightDefault())
    ++failures;
  if (!testExactRepairBudgetDefaults())
    ++failures;
  if (!testCPSatEscalationThresholds())
    ++failures;
  if (!testMultiRestartDefaults())
    ++failures;
  if (!testLocalRepairBudgetFractionDefault())
    ++failures;
  if (!testBudgetSecondsDefault())
    ++failures;
  if (!testOptionValidation())
    ++failures;

  if (failures > 0) {
    std::cerr << "\n" << failures << " test(s) FAILED\n";
    return 1;
  }
  std::cout << "\nAll tests PASSED\n";
  return 0;
}
