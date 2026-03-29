/// Contract types unit tests: MappingResult, AssignmentPlan, and extended
/// InfeasibilityCut with suggestedConstraints.
///
/// Tests JSON round-trip serialization, default initialization, and
/// backward compatibility.

#include "loom/SystemCompiler/AssignmentPlan.h"
#include "loom/SystemCompiler/InfeasibilityCut.h"
#include "loom/SystemCompiler/MappingResult.h"
#include "llvm/Support/JSON.h"

#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <string>
#include <variant>
#include <vector>

using namespace loom;

static bool approxEqual(double a, double b, double tol = 0.001) {
  return std::fabs(a - b) < tol;
}

/// Test 1: MappingResult default initialization.
static bool testMappingResultDefaults() {
  MappingResult mr;
  if (mr.success != false) {
    std::cerr << "FAIL: testMappingResultDefaults - success\n";
    return false;
  }
  if (mr.resourceUsage.peUtilization != 0.0 ||
      mr.resourceUsage.fuUtilization != 0.0 ||
      mr.resourceUsage.spmBytesUsed != 0) {
    std::cerr << "FAIL: testMappingResultDefaults - resourceUsage\n";
    return false;
  }
  if (!mr.perKernelResults.empty()) {
    std::cerr << "FAIL: testMappingResultDefaults - perKernelResults\n";
    return false;
  }
  if (mr.configBlob.has_value()) {
    std::cerr << "FAIL: testMappingResultDefaults - configBlob\n";
    return false;
  }
  std::cerr << "PASS: testMappingResultDefaults\n";
  return true;
}

/// Test 2: MappingResult JSON round-trip.
static bool testMappingResultJsonRoundTrip() {
  MappingResult original;
  original.success = true;
  original.resourceUsage.peUtilization = 0.85;
  original.resourceUsage.fuUtilization = 0.72;
  original.resourceUsage.spmBytesUsed = 4096;
  original.cycleEstimate.achievedII = 4;
  original.cycleEstimate.totalExecutionCycles = 40000;
  original.cycleEstimate.tripCount = 1000;
  original.routingCongestion.maxSwitchUtilization = 0.45;
  original.routingCongestion.unroutedEdgeCount = 0;

  KernelMetrics km;
  km.kernelName = "matmul";
  km.achievedII = 4;
  km.peUtilization = 0.85;
  km.fuUtilization = 0.72;
  km.switchUtilization = 0.45;
  km.spmBytesUsed = 2048;
  km.achievedStreamRate = 1.5;
  original.perKernelResults.push_back(km);

  llvm::json::Value json = original.toJSON();
  MappingResult restored = MappingResult::fromJSON(json);

  if (restored.success != true) {
    std::cerr << "FAIL: testMappingResultJsonRoundTrip - success\n";
    return false;
  }
  if (!approxEqual(restored.resourceUsage.peUtilization, 0.85)) {
    std::cerr << "FAIL: testMappingResultJsonRoundTrip - peUtilization\n";
    return false;
  }
  if (!approxEqual(restored.resourceUsage.fuUtilization, 0.72)) {
    std::cerr << "FAIL: testMappingResultJsonRoundTrip - fuUtilization\n";
    return false;
  }
  if (restored.resourceUsage.spmBytesUsed != 4096) {
    std::cerr << "FAIL: testMappingResultJsonRoundTrip - spmBytesUsed\n";
    return false;
  }
  if (restored.cycleEstimate.achievedII != 4) {
    std::cerr << "FAIL: testMappingResultJsonRoundTrip - achievedII\n";
    return false;
  }
  if (restored.cycleEstimate.totalExecutionCycles != 40000) {
    std::cerr << "FAIL: testMappingResultJsonRoundTrip - totalExecCycles\n";
    return false;
  }
  if (!approxEqual(restored.routingCongestion.maxSwitchUtilization, 0.45)) {
    std::cerr << "FAIL: testMappingResultJsonRoundTrip - maxSwitchUtil\n";
    return false;
  }
  if (restored.routingCongestion.unroutedEdgeCount != 0) {
    std::cerr << "FAIL: testMappingResultJsonRoundTrip - unroutedEdgeCount\n";
    return false;
  }
  if (restored.perKernelResults.size() != 1) {
    std::cerr << "FAIL: testMappingResultJsonRoundTrip - perKernelResults\n";
    return false;
  }
  if (restored.perKernelResults[0].kernelName != "matmul") {
    std::cerr << "FAIL: testMappingResultJsonRoundTrip - kernelName\n";
    return false;
  }
  std::cerr << "PASS: testMappingResultJsonRoundTrip\n";
  return true;
}

/// Test 3: AssignmentPlan default initialization.
static bool testAssignmentPlanDefaults() {
  AssignmentPlan plan;
  if (!plan.kernelToCore.empty()) {
    std::cerr << "FAIL: testAssignmentPlanDefaults - kernelToCore\n";
    return false;
  }
  if (!plan.schedulingOrder.empty()) {
    std::cerr << "FAIL: testAssignmentPlanDefaults - schedulingOrder\n";
    return false;
  }
  if (!plan.nocPaths.empty()) {
    std::cerr << "FAIL: testAssignmentPlanDefaults - nocPaths\n";
    return false;
  }
  if (plan.objectiveValue.latency != 0.0 ||
      plan.objectiveValue.nocCost != 0.0 ||
      plan.objectiveValue.localityBonus != 0.0) {
    std::cerr << "FAIL: testAssignmentPlanDefaults - objective\n";
    return false;
  }
  std::cerr << "PASS: testAssignmentPlanDefaults\n";
  return true;
}

/// Test 4: AssignmentPlan JSON round-trip.
static bool testAssignmentPlanJsonRoundTrip() {
  AssignmentPlan original;
  original.kernelToCore["matmul"] = 0;
  original.kernelToCore["relu"] = 1;

  CoreAssignment ca0;
  ca0.coreInstanceIdx = 0;
  ca0.coreTypeName = "PE_A";
  ca0.assignedKernels = {"matmul"};
  ca0.estimatedUtilization = 0.75;
  original.coreAssignments.push_back(ca0);

  CoreAssignment ca1;
  ca1.coreInstanceIdx = 1;
  ca1.coreTypeName = "PE_B";
  ca1.assignedKernels = {"relu"};
  ca1.estimatedUtilization = 0.50;
  original.coreAssignments.push_back(ca1);

  original.schedulingOrder = {"matmul", "relu"};

  NoCRoute route;
  route.contractEdgeName = "matmul->relu";
  route.producerCore = "PE_A_0";
  route.consumerCore = "PE_B_0";
  route.hops = {{0, 0}, {0, 1}};
  route.numHops = 1;
  route.bandwidthFlitsPerCycle = 4;
  route.transferLatencyCycles = 3;
  original.nocPaths.push_back(route);

  original.objectiveValue.latency = 100.0;
  original.objectiveValue.nocCost = 20.0;
  original.objectiveValue.localityBonus = 5.0;

  llvm::json::Value json = original.toJSON();
  AssignmentPlan restored = AssignmentPlan::fromJSON(json);

  if (restored.kernelToCore.size() != 2) {
    std::cerr << "FAIL: testAssignmentPlanJsonRoundTrip - k2c size\n";
    return false;
  }
  if (restored.kernelToCore.at("matmul") != 0 ||
      restored.kernelToCore.at("relu") != 1) {
    std::cerr << "FAIL: testAssignmentPlanJsonRoundTrip - k2c values\n";
    return false;
  }
  if (restored.coreAssignments.size() != 2) {
    std::cerr << "FAIL: testAssignmentPlanJsonRoundTrip - coreAssignments\n";
    return false;
  }
  if (restored.coreAssignments[0].coreTypeName != "PE_A") {
    std::cerr << "FAIL: testAssignmentPlanJsonRoundTrip - coreTypeName\n";
    return false;
  }
  if (restored.schedulingOrder.size() != 2 ||
      restored.schedulingOrder[0] != "matmul" ||
      restored.schedulingOrder[1] != "relu") {
    std::cerr << "FAIL: testAssignmentPlanJsonRoundTrip - schedulingOrder\n";
    return false;
  }
  if (restored.nocPaths.size() != 1) {
    std::cerr << "FAIL: testAssignmentPlanJsonRoundTrip - nocPaths size\n";
    return false;
  }
  if (restored.nocPaths[0].contractEdgeName != "matmul->relu") {
    std::cerr << "FAIL: testAssignmentPlanJsonRoundTrip - nocPaths edge\n";
    return false;
  }
  if (restored.nocPaths[0].hops.size() != 2) {
    std::cerr << "FAIL: testAssignmentPlanJsonRoundTrip - nocPaths hops\n";
    return false;
  }
  if (!approxEqual(restored.objectiveValue.latency, 100.0) ||
      !approxEqual(restored.objectiveValue.nocCost, 20.0) ||
      !approxEqual(restored.objectiveValue.localityBonus, 5.0)) {
    std::cerr << "FAIL: testAssignmentPlanJsonRoundTrip - objective\n";
    return false;
  }
  std::cerr << "PASS: testAssignmentPlanJsonRoundTrip\n";
  return true;
}

/// Test 5: InfeasibilityCut with ExclusionConstraint.
static bool testInfeasibilityCutWithExclusionConstraint() {
  InfeasibilityCut cut;
  cut.kernelName = "conv";
  cut.coreType = "PE_B";
  cut.reason = CutReason::TYPE_MISMATCH;
  cut.evidence = FUShortage{"arith.muli", 4, 0};

  ExclusionConstraint ec;
  ec.kernel = "conv";
  ec.coreType = "PE_B";
  cut.suggestedConstraints.push_back(ec);

  llvm::json::Value json = infeasibilityCutToJSON(cut);
  InfeasibilityCut restored = infeasibilityCutFromJSON(json);

  if (restored.suggestedConstraints.size() != 1) {
    std::cerr << "FAIL: testInfeasibilityCutWithExclusion - size\n";
    return false;
  }
  if (!std::holds_alternative<ExclusionConstraint>(
          restored.suggestedConstraints[0])) {
    std::cerr << "FAIL: testInfeasibilityCutWithExclusion - variant type\n";
    return false;
  }
  auto &restoredEC =
      std::get<ExclusionConstraint>(restored.suggestedConstraints[0]);
  if (restoredEC.kernel != "conv" || restoredEC.coreType != "PE_B") {
    std::cerr << "FAIL: testInfeasibilityCutWithExclusion - fields\n";
    return false;
  }
  std::cerr << "PASS: testInfeasibilityCutWithExclusionConstraint\n";
  return true;
}

/// Test 6: InfeasibilityCut with CapacityConstraint.
static bool testInfeasibilityCutWithCapacityConstraint() {
  InfeasibilityCut cut;
  cut.kernelName = "matmul";
  cut.coreType = "PE_A";
  cut.reason = CutReason::INSUFFICIENT_FU;
  cut.evidence = FUShortage{"arith.muli", 8, 2};

  CapacityConstraint cc;
  cc.coreType = "PE_A";
  cc.resourceName = "FU:mul";
  cc.minRequired = 8;
  cut.suggestedConstraints.push_back(cc);

  llvm::json::Value json = infeasibilityCutToJSON(cut);
  InfeasibilityCut restored = infeasibilityCutFromJSON(json);

  if (restored.suggestedConstraints.size() != 1) {
    std::cerr << "FAIL: testInfeasibilityCutWithCapacity - size\n";
    return false;
  }
  if (!std::holds_alternative<CapacityConstraint>(
          restored.suggestedConstraints[0])) {
    std::cerr << "FAIL: testInfeasibilityCutWithCapacity - variant type\n";
    return false;
  }
  auto &restoredCC =
      std::get<CapacityConstraint>(restored.suggestedConstraints[0]);
  if (restoredCC.coreType != "PE_A" || restoredCC.resourceName != "FU:mul" ||
      restoredCC.minRequired != 8) {
    std::cerr << "FAIL: testInfeasibilityCutWithCapacity - fields\n";
    return false;
  }
  std::cerr << "PASS: testInfeasibilityCutWithCapacityConstraint\n";
  return true;
}

/// Test 7: InfeasibilityCut with multiple heterogeneous constraints.
static bool testInfeasibilityCutMultipleConstraints() {
  InfeasibilityCut cut;
  cut.kernelName = "conv";
  cut.coreType = "PE_B";
  cut.reason = CutReason::INSUFFICIENT_FU;
  cut.evidence = FUShortage{"arith.muli", 4, 0};

  ExclusionConstraint ec;
  ec.kernel = "conv";
  ec.coreType = "PE_B";
  cut.suggestedConstraints.push_back(ec);

  CapacityConstraint cc;
  cc.coreType = "PE_B";
  cc.resourceName = "FU:mul";
  cc.minRequired = 4;
  cut.suggestedConstraints.push_back(cc);

  llvm::json::Value json = infeasibilityCutToJSON(cut);
  InfeasibilityCut restored = infeasibilityCutFromJSON(json);

  if (restored.suggestedConstraints.size() != 2) {
    std::cerr << "FAIL: testInfeasibilityCutMultiple - size\n";
    return false;
  }
  if (!std::holds_alternative<ExclusionConstraint>(
          restored.suggestedConstraints[0])) {
    std::cerr << "FAIL: testInfeasibilityCutMultiple - first type\n";
    return false;
  }
  if (!std::holds_alternative<CapacityConstraint>(
          restored.suggestedConstraints[1])) {
    std::cerr << "FAIL: testInfeasibilityCutMultiple - second type\n";
    return false;
  }
  std::cerr << "PASS: testInfeasibilityCutMultipleConstraints\n";
  return true;
}

/// Test 8: InfeasibilityCut backward compatibility (empty suggestedConstraints).
static bool testInfeasibilityCutBackwardCompat() {
  InfeasibilityCut cut;
  cut.kernelName = "relu";
  cut.coreType = "PE_A";
  cut.reason = CutReason::SPM_OVERFLOW;
  cut.evidence = SPMInfo{8192, 4096};
  // No suggestedConstraints -- empty.

  llvm::json::Value json = infeasibilityCutToJSON(cut);
  InfeasibilityCut restored = infeasibilityCutFromJSON(json);

  if (!restored.suggestedConstraints.empty()) {
    std::cerr << "FAIL: testInfeasibilityCutBackwardCompat - not empty\n";
    return false;
  }
  if (restored.kernelName != "relu" || restored.coreType != "PE_A") {
    std::cerr << "FAIL: testInfeasibilityCutBackwardCompat - fields\n";
    return false;
  }
  if (restored.reason != CutReason::SPM_OVERFLOW) {
    std::cerr << "FAIL: testInfeasibilityCutBackwardCompat - reason\n";
    return false;
  }
  std::cerr << "PASS: testInfeasibilityCutBackwardCompat\n";
  return true;
}

/// Test 9: MappingResult with configBlob.
static bool testMappingResultWithConfigBlob() {
  MappingResult original;
  original.success = true;

  std::vector<uint8_t> blob(16);
  for (unsigned i = 0; i < 16; ++i)
    blob[i] = static_cast<uint8_t>(i * 17 + 3);
  original.configBlob = blob;

  llvm::json::Value json = original.toJSON();
  MappingResult restored = MappingResult::fromJSON(json);

  if (!restored.configBlob.has_value()) {
    std::cerr << "FAIL: testMappingResultWithConfigBlob - no value\n";
    return false;
  }
  if (restored.configBlob->size() != 16) {
    std::cerr << "FAIL: testMappingResultWithConfigBlob - size "
              << restored.configBlob->size() << "\n";
    return false;
  }
  for (unsigned i = 0; i < 16; ++i) {
    if ((*restored.configBlob)[i] != blob[i]) {
      std::cerr << "FAIL: testMappingResultWithConfigBlob - byte " << i
                << " expected " << (int)blob[i]
                << " got " << (int)(*restored.configBlob)[i] << "\n";
      return false;
    }
  }
  std::cerr << "PASS: testMappingResultWithConfigBlob\n";
  return true;
}

/// Test 10: AssignmentPlan with empty schedule.
static bool testAssignmentPlanEmptySchedule() {
  AssignmentPlan original;
  original.kernelToCore["matmul"] = 0;

  CoreAssignment ca;
  ca.coreInstanceIdx = 0;
  ca.coreTypeName = "PE_A";
  ca.assignedKernels = {"matmul"};
  original.coreAssignments.push_back(ca);
  // schedulingOrder and nocPaths intentionally left empty.

  llvm::json::Value json = original.toJSON();
  AssignmentPlan restored = AssignmentPlan::fromJSON(json);

  if (restored.schedulingOrder.size() != 0) {
    std::cerr << "FAIL: testAssignmentPlanEmptySchedule - schedulingOrder\n";
    return false;
  }
  if (restored.nocPaths.size() != 0) {
    std::cerr << "FAIL: testAssignmentPlanEmptySchedule - nocPaths\n";
    return false;
  }
  if (restored.kernelToCore.size() != 1) {
    std::cerr << "FAIL: testAssignmentPlanEmptySchedule - kernelToCore\n";
    return false;
  }
  std::cerr << "PASS: testAssignmentPlanEmptySchedule\n";
  return true;
}

int main() {
  int failures = 0;

  if (!testMappingResultDefaults()) ++failures;
  if (!testMappingResultJsonRoundTrip()) ++failures;
  if (!testAssignmentPlanDefaults()) ++failures;
  if (!testAssignmentPlanJsonRoundTrip()) ++failures;
  if (!testInfeasibilityCutWithExclusionConstraint()) ++failures;
  if (!testInfeasibilityCutWithCapacityConstraint()) ++failures;
  if (!testInfeasibilityCutMultipleConstraints()) ++failures;
  if (!testInfeasibilityCutBackwardCompat()) ++failures;
  if (!testMappingResultWithConfigBlob()) ++failures;
  if (!testAssignmentPlanEmptySchedule()) ++failures;

  std::cerr << "\n" << (10 - failures) << "/10 tests passed";
  if (failures > 0)
    std::cerr << " (" << failures << " FAILED)";
  std::cerr << "\n";

  return failures > 0 ? 1 : 0;
}
