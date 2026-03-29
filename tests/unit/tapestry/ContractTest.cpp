/// Contract data structure tests: construction, JSON serialization,
/// and legality checking.
///
/// Tests:
/// 1. ContractSpec default construction
/// 2. ContractSpec JSON round-trip
/// 3. InfeasibilityCut JSON round-trip for each evidence type
/// 4. CoreCostSummary JSON round-trip
/// 5. Enum string conversions

#include "loom/SystemCompiler/Contract.h"
#include "loom/SystemCompiler/CostSummary.h"
#include "loom/SystemCompiler/InfeasibilityCut.h"
#include "llvm/Support/JSON.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <string>

using namespace loom;

/// Test 1: ContractSpec default construction with expected defaults.
static bool testContractDefaults() {
  ContractSpec spec;
  if (spec.ordering != Ordering::FIFO) {
    std::cerr << "FAIL: testContractDefaults - ordering\n";
    return false;
  }
  if (spec.backpressure != Backpressure::BLOCK) {
    std::cerr << "FAIL: testContractDefaults - backpressure\n";
    return false;
  }
  if (spec.visibility != Visibility::LOCAL_SPM) {
    std::cerr << "FAIL: testContractDefaults - visibility\n";
    return false;
  }
  if (spec.doubleBuffering != false) {
    std::cerr << "FAIL: testContractDefaults - doubleBuffering\n";
    return false;
  }
  if (spec.mayFuse != true || spec.mayReplicate != true ||
      spec.mayPipeline != true || spec.mayReorder != false ||
      spec.mayRetile != true) {
    std::cerr << "FAIL: testContractDefaults - permissions\n";
    return false;
  }
  std::cerr << "PASS: testContractDefaults\n";
  return true;
}

/// Test 2: ContractSpec JSON round-trip.
static bool testContractJSONRoundTrip() {
  ContractSpec spec;
  spec.producerKernel = "matmul";
  spec.consumerKernel = "relu";
  spec.dataTypeName = "f32";
  spec.ordering = Ordering::UNORDERED;
  spec.productionRate = 64;
  spec.consumptionRate = 32;
  spec.steadyStateRatio = {2, 1};
  spec.tileShape = {16, 16};
  spec.minBufferElements = 256;
  spec.maxBufferElements = 1024;
  spec.backpressure = Backpressure::DROP;
  spec.doubleBuffering = true;
  spec.visibility = Visibility::SHARED_L2;
  spec.producerWriteback = Writeback::LAZY;
  spec.consumerPrefetch = Prefetch::NEXT_TILE;
  spec.mayFuse = false;
  spec.mayReplicate = false;
  spec.mayPipeline = false;
  spec.mayReorder = true;
  spec.mayRetile = false;
  spec.achievedProductionRate = 60;
  spec.achievedConsumptionRate = 30;
  spec.achievedBufferSize = 512;

  auto json = contractSpecToJSON(spec);
  ContractSpec spec2 = contractSpecFromJSON(json);

  if (spec2.producerKernel != "matmul" || spec2.consumerKernel != "relu") {
    std::cerr << "FAIL: testContractJSONRoundTrip - kernel names\n";
    return false;
  }
  if (spec2.dataTypeName != "f32") {
    std::cerr << "FAIL: testContractJSONRoundTrip - dataTypeName\n";
    return false;
  }
  if (spec2.ordering != Ordering::UNORDERED) {
    std::cerr << "FAIL: testContractJSONRoundTrip - ordering\n";
    return false;
  }
  if (spec2.productionRate != 64 || spec2.consumptionRate != 32) {
    std::cerr << "FAIL: testContractJSONRoundTrip - rates\n";
    return false;
  }
  if (!spec2.steadyStateRatio || spec2.steadyStateRatio->first != 2 ||
      spec2.steadyStateRatio->second != 1) {
    std::cerr << "FAIL: testContractJSONRoundTrip - steadyStateRatio\n";
    return false;
  }
  if (spec2.tileShape.size() != 2 || spec2.tileShape[0] != 16 ||
      spec2.tileShape[1] != 16) {
    std::cerr << "FAIL: testContractJSONRoundTrip - tileShape\n";
    return false;
  }
  if (spec2.minBufferElements != 256 || spec2.maxBufferElements != 1024) {
    std::cerr << "FAIL: testContractJSONRoundTrip - buffer elements\n";
    return false;
  }
  if (spec2.backpressure != Backpressure::DROP) {
    std::cerr << "FAIL: testContractJSONRoundTrip - backpressure\n";
    return false;
  }
  if (spec2.doubleBuffering != true) {
    std::cerr << "FAIL: testContractJSONRoundTrip - doubleBuffering\n";
    return false;
  }
  if (spec2.visibility != Visibility::SHARED_L2) {
    std::cerr << "FAIL: testContractJSONRoundTrip - visibility\n";
    return false;
  }
  if (spec2.producerWriteback != Writeback::LAZY) {
    std::cerr << "FAIL: testContractJSONRoundTrip - writeback\n";
    return false;
  }
  if (spec2.consumerPrefetch != Prefetch::NEXT_TILE) {
    std::cerr << "FAIL: testContractJSONRoundTrip - prefetch\n";
    return false;
  }
  if (spec2.mayFuse != false || spec2.mayReplicate != false ||
      spec2.mayPipeline != false || spec2.mayReorder != true ||
      spec2.mayRetile != false) {
    std::cerr << "FAIL: testContractJSONRoundTrip - permissions\n";
    return false;
  }
  if (spec2.achievedProductionRate != 60 ||
      spec2.achievedConsumptionRate != 30 ||
      spec2.achievedBufferSize != 512) {
    std::cerr << "FAIL: testContractJSONRoundTrip - achieved values\n";
    return false;
  }

  std::cerr << "PASS: testContractJSONRoundTrip\n";
  return true;
}

/// Test 3: InfeasibilityCut JSON round-trip for FUShortage evidence.
static bool testInfeasibilityCutFUShortage() {
  InfeasibilityCut cut;
  cut.kernelName = "matmul";
  cut.coreType = "PE_A";
  cut.reason = CutReason::INSUFFICIENT_FU;
  cut.evidence = FUShortage{"mul", 8, 4};

  auto json = infeasibilityCutToJSON(cut);
  auto cut2 = infeasibilityCutFromJSON(json);

  if (cut2.kernelName != "matmul" || cut2.coreType != "PE_A") {
    std::cerr << "FAIL: testInfeasibilityCutFUShortage - names\n";
    return false;
  }
  if (cut2.reason != CutReason::INSUFFICIENT_FU) {
    std::cerr << "FAIL: testInfeasibilityCutFUShortage - reason\n";
    return false;
  }
  auto *shortage = std::get_if<FUShortage>(&cut2.evidence);
  if (!shortage || shortage->fuType != "mul" ||
      shortage->needed != 8 || shortage->available != 4) {
    std::cerr << "FAIL: testInfeasibilityCutFUShortage - evidence\n";
    return false;
  }

  std::cerr << "PASS: testInfeasibilityCutFUShortage\n";
  return true;
}

/// Test 3b: InfeasibilityCut JSON round-trip for IIInfo evidence.
static bool testInfeasibilityCutIIInfo() {
  InfeasibilityCut cut;
  cut.kernelName = "conv";
  cut.coreType = "PE_B";
  cut.reason = CutReason::II_UNACHIEVABLE;
  cut.evidence = IIInfo{5, 3};

  auto json = infeasibilityCutToJSON(cut);
  auto cut2 = infeasibilityCutFromJSON(json);

  if (cut2.reason != CutReason::II_UNACHIEVABLE) {
    std::cerr << "FAIL: testInfeasibilityCutIIInfo - reason\n";
    return false;
  }
  auto *info = std::get_if<IIInfo>(&cut2.evidence);
  if (!info || info->minII != 5 || info->targetII != 3) {
    std::cerr << "FAIL: testInfeasibilityCutIIInfo - evidence\n";
    return false;
  }

  std::cerr << "PASS: testInfeasibilityCutIIInfo\n";
  return true;
}

/// Test 4: CoreCostSummary JSON round-trip.
static bool testCoreCostSummaryRoundTrip() {
  CoreCostSummary summary;
  summary.coreInstanceName = "core_0";
  summary.coreType = "PE_A";
  summary.success = true;

  KernelMetrics km;
  km.kernelName = "matmul";
  km.achievedII = 4;
  km.peUtilization = 0.85;
  km.fuUtilization = 0.72;
  km.switchUtilization = 0.45;
  km.spmBytesUsed = 2048;
  km.achievedStreamRate = 32.5;
  summary.kernelMetrics.push_back(km);

  summary.totalPEUtilization = 0.85;
  summary.totalSPMUtilization = 0.5;
  summary.routingPressure = 0.45;

  auto json = coreCostSummaryToJSON(summary);
  auto summary2 = coreCostSummaryFromJSON(json);

  if (summary2.coreInstanceName != "core_0" ||
      summary2.coreType != "PE_A" || !summary2.success) {
    std::cerr << "FAIL: testCoreCostSummaryRoundTrip - basic fields\n";
    return false;
  }
  if (summary2.kernelMetrics.size() != 1) {
    std::cerr << "FAIL: testCoreCostSummaryRoundTrip - metrics count\n";
    return false;
  }
  auto &km2 = summary2.kernelMetrics[0];
  if (km2.kernelName != "matmul" || km2.achievedII != 4) {
    std::cerr << "FAIL: testCoreCostSummaryRoundTrip - kernel metrics\n";
    return false;
  }
  if (std::abs(km2.peUtilization - 0.85) > 0.001 ||
      std::abs(km2.fuUtilization - 0.72) > 0.001) {
    std::cerr << "FAIL: testCoreCostSummaryRoundTrip - utilizations\n";
    return false;
  }
  if (std::abs(summary2.routingPressure - 0.45) > 0.001) {
    std::cerr << "FAIL: testCoreCostSummaryRoundTrip - routing pressure\n";
    return false;
  }

  std::cerr << "PASS: testCoreCostSummaryRoundTrip\n";
  return true;
}

/// Test 5: Enum string conversion round-trips.
static bool testEnumConversions() {
  // Ordering
  if (orderingFromString(orderingToString(Ordering::FIFO)) != Ordering::FIFO ||
      orderingFromString(orderingToString(Ordering::UNORDERED)) !=
          Ordering::UNORDERED) {
    std::cerr << "FAIL: testEnumConversions - Ordering\n";
    return false;
  }

  // Backpressure
  if (backpressureFromString(backpressureToString(Backpressure::BLOCK)) !=
          Backpressure::BLOCK ||
      backpressureFromString(backpressureToString(Backpressure::DROP)) !=
          Backpressure::DROP ||
      backpressureFromString(backpressureToString(Backpressure::OVERWRITE)) !=
          Backpressure::OVERWRITE) {
    std::cerr << "FAIL: testEnumConversions - Backpressure\n";
    return false;
  }

  // Visibility (now aliases Placement; EXTERNAL_DRAM -> EXTERNAL)
  if (visibilityFromString(visibilityToString(Visibility::LOCAL_SPM)) !=
          Visibility::LOCAL_SPM ||
      visibilityFromString(visibilityToString(Visibility::SHARED_L2)) !=
          Visibility::SHARED_L2 ||
      visibilityFromString(visibilityToString(Placement::EXTERNAL)) !=
          Placement::EXTERNAL) {
    std::cerr << "FAIL: testEnumConversions - Visibility\n";
    return false;
  }
  // Legacy EXTERNAL_DRAM string must still parse to EXTERNAL
  if (visibilityFromString("EXTERNAL_DRAM") != Placement::EXTERNAL) {
    std::cerr << "FAIL: testEnumConversions - EXTERNAL_DRAM compat\n";
    return false;
  }

  // Writeback
  if (writebackFromString(writebackToString(Writeback::EAGER)) !=
          Writeback::EAGER ||
      writebackFromString(writebackToString(Writeback::LAZY)) !=
          Writeback::LAZY) {
    std::cerr << "FAIL: testEnumConversions - Writeback\n";
    return false;
  }

  // Prefetch
  if (prefetchFromString(prefetchToString(Prefetch::NONE)) != Prefetch::NONE ||
      prefetchFromString(prefetchToString(Prefetch::NEXT_TILE)) !=
          Prefetch::NEXT_TILE ||
      prefetchFromString(prefetchToString(Prefetch::DOUBLE_BUFFER)) !=
          Prefetch::DOUBLE_BUFFER) {
    std::cerr << "FAIL: testEnumConversions - Prefetch\n";
    return false;
  }

  // CutReason
  if (cutReasonFromString(cutReasonToString(CutReason::INSUFFICIENT_FU)) !=
          CutReason::INSUFFICIENT_FU ||
      cutReasonFromString(cutReasonToString(CutReason::ROUTING_CONGESTION)) !=
          CutReason::ROUTING_CONGESTION ||
      cutReasonFromString(cutReasonToString(CutReason::SPM_OVERFLOW)) !=
          CutReason::SPM_OVERFLOW ||
      cutReasonFromString(cutReasonToString(CutReason::II_UNACHIEVABLE)) !=
          CutReason::II_UNACHIEVABLE ||
      cutReasonFromString(cutReasonToString(CutReason::TYPE_MISMATCH)) !=
          CutReason::TYPE_MISMATCH) {
    std::cerr << "FAIL: testEnumConversions - CutReason\n";
    return false;
  }

  std::cerr << "PASS: testEnumConversions\n";
  return true;
}

int main() {
  int passed = 0;
  int total = 0;

  auto run = [&](bool (*test)()) {
    total++;
    if (test())
      passed++;
  };

  run(testContractDefaults);
  run(testContractJSONRoundTrip);
  run(testInfeasibilityCutFUShortage);
  run(testInfeasibilityCutIIInfo);
  run(testCoreCostSummaryRoundTrip);
  run(testEnumConversions);

  std::cerr << "\nResults: " << passed << "/" << total << " tests passed\n";
  return (passed == total) ? 0 : 1;
}
