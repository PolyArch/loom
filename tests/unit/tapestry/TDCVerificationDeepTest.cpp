/// Deepened TDCVerification tests: contract inference, static verification
/// (placement, shape, ordering), dynamic verification (throughput, latency,
/// ordering violations), and aggregate report generation.
///
/// Tests:
///  1. Inference fills missing Ordering with FIFO
///  2. Inference preserves user-specified Ordering
///  3. Inference fills missing Placement with AUTO
///  4. Inference leaves missing Throughput as absent
///  5. Inference leaves missing Shape as absent
///  6. Path contract validates edge references
///  7. Fully specified spec triggers no inference
///  8. Static: placement violation detected
///  9. Static: placement satisfied (LOCAL_SPM matches SPM_CONSUMER)
/// 10. Static: shape violation detected
/// 11. Static: inferred dimension skipped
/// 12. Dynamic: throughput violation
/// 13. Dynamic: throughput satisfied
/// 14. Dynamic: path latency violation
/// 15. Dynamic: path latency satisfied
/// 16. Dynamic: FIFO ordering violation
/// 17. Aggregate: all satisfied -> clean report
/// 18. Aggregate: one failure -> report fails
/// 19. Static-only when no dynamic metrics

#include "loom/SystemCompiler/TDCVerification.h"

#include <cassert>
#include <iostream>
#include <string>
#include <vector>

using namespace loom;

//===----------------------------------------------------------------------===//
// Inference tests
//===----------------------------------------------------------------------===//

/// T1: Missing ordering defaults to FIFO
static bool testInferMissingOrdering() {
  TDCEdgeSpec spec;
  spec.producerKernel = "a";
  spec.consumerKernel = "b";
  spec.dataTypeName = "f32";
  spec.throughput = "64";
  // ordering is nullopt

  auto result = inferEdgeContracts({spec});

  if (result.edgeSpecs.size() != 1 || result.origins.size() != 1) {
    std::cerr << "FAIL: testInferMissingOrdering - wrong sizes\n";
    return false;
  }

  if (!result.edgeSpecs[0].ordering.has_value() ||
      *result.edgeSpecs[0].ordering != Ordering::FIFO) {
    std::cerr << "FAIL: testInferMissingOrdering - expected FIFO\n";
    return false;
  }

  if (result.origins[0].ordering != DimensionOrigin::INFERRED) {
    std::cerr << "FAIL: testInferMissingOrdering - expected INFERRED\n";
    return false;
  }

  std::cerr << "PASS: testInferMissingOrdering\n";
  return true;
}

/// T2: User-specified ordering preserved
static bool testInferPreservesOrdering() {
  TDCEdgeSpec spec;
  spec.producerKernel = "a";
  spec.consumerKernel = "b";
  spec.dataTypeName = "f32";
  spec.ordering = Ordering::UNORDERED;

  auto result = inferEdgeContracts({spec});

  if (*result.edgeSpecs[0].ordering != Ordering::UNORDERED) {
    std::cerr << "FAIL: testInferPreservesOrdering - ordering changed\n";
    return false;
  }
  if (result.origins[0].ordering != DimensionOrigin::USER_SPECIFIED) {
    std::cerr << "FAIL: testInferPreservesOrdering - expected USER_SPECIFIED\n";
    return false;
  }

  std::cerr << "PASS: testInferPreservesOrdering\n";
  return true;
}

/// T3: Missing placement defaults to AUTO
static bool testInferMissingPlacement() {
  TDCEdgeSpec spec;
  spec.producerKernel = "a";
  spec.consumerKernel = "b";
  spec.dataTypeName = "f32";

  auto result = inferEdgeContracts({spec});

  if (!result.edgeSpecs[0].placement.has_value() ||
      *result.edgeSpecs[0].placement != Placement::AUTO) {
    std::cerr << "FAIL: testInferMissingPlacement - expected AUTO\n";
    return false;
  }
  if (result.origins[0].placement != DimensionOrigin::INFERRED) {
    std::cerr << "FAIL: testInferMissingPlacement - expected INFERRED\n";
    return false;
  }

  std::cerr << "PASS: testInferMissingPlacement\n";
  return true;
}

/// T4: Missing throughput remains absent
static bool testInferMissingThroughput() {
  TDCEdgeSpec spec;
  spec.producerKernel = "a";
  spec.consumerKernel = "b";
  spec.dataTypeName = "f32";

  auto result = inferEdgeContracts({spec});

  if (result.edgeSpecs[0].throughput.has_value()) {
    std::cerr << "FAIL: testInferMissingThroughput - should be nullopt\n";
    return false;
  }
  if (result.origins[0].throughput != DimensionOrigin::ABSENT) {
    std::cerr << "FAIL: testInferMissingThroughput - expected ABSENT\n";
    return false;
  }

  std::cerr << "PASS: testInferMissingThroughput\n";
  return true;
}

/// T5: Missing shape remains absent
static bool testInferMissingShape() {
  TDCEdgeSpec spec;
  spec.producerKernel = "a";
  spec.consumerKernel = "b";
  spec.dataTypeName = "f32";

  auto result = inferEdgeContracts({spec});

  if (result.edgeSpecs[0].shape.has_value()) {
    std::cerr << "FAIL: testInferMissingShape - should be nullopt\n";
    return false;
  }
  if (result.origins[0].shape != DimensionOrigin::ABSENT) {
    std::cerr << "FAIL: testInferMissingShape - expected ABSENT\n";
    return false;
  }

  std::cerr << "PASS: testInferMissingShape\n";
  return true;
}

/// T6: Path contract with invalid edge reference produces error
static bool testPathInvalidEdgeRef() {
  TDCEdgeSpec e1;
  e1.producerKernel = "a";
  e1.consumerKernel = "b";
  e1.dataTypeName = "f32";

  TDCPathSpec path;
  path.startProducer = "nonexistent";
  path.startConsumer = "b";
  path.endProducer = "a";
  path.endConsumer = "b";
  path.latency = "100";

  auto errors = validatePathReferences({path}, {e1});

  if (errors.empty()) {
    std::cerr << "FAIL: testPathInvalidEdgeRef - expected errors\n";
    return false;
  }

  bool foundNonexistent = false;
  for (const auto &err : errors) {
    if (err.find("nonexistent") != std::string::npos)
      foundNonexistent = true;
  }
  if (!foundNonexistent) {
    std::cerr << "FAIL: testPathInvalidEdgeRef - error should mention "
                 "'nonexistent'\n";
    return false;
  }

  std::cerr << "PASS: testPathInvalidEdgeRef\n";
  return true;
}

/// T7: Fully specified spec -> no inference changes
static bool testInferFullySpecified() {
  TDCEdgeSpec spec;
  spec.producerKernel = "a";
  spec.consumerKernel = "b";
  spec.dataTypeName = "f32";
  spec.ordering = Ordering::UNORDERED;
  spec.throughput = "100";
  spec.placement = Placement::SHARED_L2;
  spec.shape = "[64, 64]";

  auto result = inferEdgeContracts({spec});

  if (*result.edgeSpecs[0].ordering != Ordering::UNORDERED ||
      *result.edgeSpecs[0].throughput != "100" ||
      *result.edgeSpecs[0].placement != Placement::SHARED_L2 ||
      *result.edgeSpecs[0].shape != "[64, 64]") {
    std::cerr << "FAIL: testInferFullySpecified - values changed\n";
    return false;
  }

  auto &orig = result.origins[0];
  if (orig.ordering != DimensionOrigin::USER_SPECIFIED ||
      orig.throughput != DimensionOrigin::USER_SPECIFIED ||
      orig.placement != DimensionOrigin::USER_SPECIFIED ||
      orig.shape != DimensionOrigin::USER_SPECIFIED) {
    std::cerr << "FAIL: testInferFullySpecified - expected all USER_SPECIFIED\n";
    return false;
  }

  std::cerr << "PASS: testInferFullySpecified\n";
  return true;
}

//===----------------------------------------------------------------------===//
// Static verification tests
//===----------------------------------------------------------------------===//

/// T8: Static verification catches placement violation
static bool testStaticPlacementViolated() {
  TDCEdgeSpec spec;
  spec.producerKernel = "a";
  spec.consumerKernel = "b";
  spec.dataTypeName = "f32";
  spec.placement = Placement::LOCAL_SPM;

  TDCEdgeSpecOrigin origin;
  origin.placement = DimensionOrigin::USER_SPECIFIED;

  // Buffer placed in SHARED_L2 (violates LOCAL_SPM)
  BufferAllocationPlan plan;
  BufferAllocation alloc;
  alloc.contractEdgeName = "a->b";
  alloc.location = BufferAllocation::SHARED_L2;
  plan.allocations.push_back(alloc);

  auto report =
      verifyStatic({spec}, {origin}, plan, {}, {});

  if (report.edgeResults.size() != 1) {
    std::cerr << "FAIL: testStaticPlacementViolated - wrong result count\n";
    return false;
  }
  if (report.edgeResults[0].placementSatisfied) {
    std::cerr << "FAIL: testStaticPlacementViolated - expected violated\n";
    return false;
  }
  if (report.allSatisfied) {
    std::cerr << "FAIL: testStaticPlacementViolated - allSatisfied should be "
                 "false\n";
    return false;
  }
  if (report.edgeResults[0].diagnostics.empty()) {
    std::cerr << "FAIL: testStaticPlacementViolated - expected diagnostic\n";
    return false;
  }

  std::cerr << "PASS: testStaticPlacementViolated\n";
  return true;
}

/// T9: Static verification passes LOCAL_SPM matching SPM_CONSUMER
static bool testStaticPlacementSatisfied() {
  TDCEdgeSpec spec;
  spec.producerKernel = "a";
  spec.consumerKernel = "b";
  spec.dataTypeName = "f32";
  spec.placement = Placement::LOCAL_SPM;

  TDCEdgeSpecOrigin origin;
  origin.placement = DimensionOrigin::USER_SPECIFIED;

  BufferAllocationPlan plan;
  BufferAllocation alloc;
  alloc.contractEdgeName = "a->b";
  alloc.location = BufferAllocation::SPM_CONSUMER;
  plan.allocations.push_back(alloc);

  auto report =
      verifyStatic({spec}, {origin}, plan, {}, {});

  if (!report.edgeResults[0].placementSatisfied) {
    std::cerr << "FAIL: testStaticPlacementSatisfied - expected satisfied\n";
    return false;
  }

  std::cerr << "PASS: testStaticPlacementSatisfied\n";
  return true;
}

/// T10: Static verification catches shape violation
static bool testStaticShapeViolated() {
  TDCEdgeSpec spec;
  spec.producerKernel = "a";
  spec.consumerKernel = "b";
  spec.dataTypeName = "f32";
  spec.shape = "[128, 256]";

  TDCEdgeSpecOrigin origin;
  origin.shape = DimensionOrigin::USER_SPECIFIED;

  EdgeTileDimensions td;
  td.producerKernel = "a";
  td.consumerKernel = "b";
  td.dimensions = {128, 128}; // Mismatch: 128 vs 256 in second dim

  auto report =
      verifyStatic({spec}, {origin}, {}, {td}, {});

  if (report.edgeResults[0].shapeSatisfied) {
    std::cerr << "FAIL: testStaticShapeViolated - expected violated\n";
    return false;
  }
  if (report.edgeResults[0].diagnostics.empty()) {
    std::cerr << "FAIL: testStaticShapeViolated - expected diagnostic\n";
    return false;
  }

  std::cerr << "PASS: testStaticShapeViolated\n";
  return true;
}

/// T11: Inferred dimension is skipped during verification
static bool testStaticInferredSkipped() {
  TDCEdgeSpec spec;
  spec.producerKernel = "a";
  spec.consumerKernel = "b";
  spec.dataTypeName = "f32";
  spec.ordering = Ordering::FIFO;

  TDCEdgeSpecOrigin origin;
  origin.ordering = DimensionOrigin::INFERRED; // Not user-specified

  // Create a schedule that violates FIFO
  EdgeSchedulingSlot slot;
  slot.producerKernel = "a";
  slot.consumerKernel = "b";
  slot.producerCompletionTimes = {100, 200};
  slot.consumerStartTimes = {50, 150}; // Consumer starts before producer

  auto report =
      verifyStatic({spec}, {origin}, {}, {}, {slot});

  // Should be satisfied because the dimension was inferred, not user-specified
  if (!report.edgeResults[0].orderingSatisfied) {
    std::cerr << "FAIL: testStaticInferredSkipped - inferred dim should be "
                 "skipped\n";
    return false;
  }

  std::cerr << "PASS: testStaticInferredSkipped\n";
  return true;
}

//===----------------------------------------------------------------------===//
// Dynamic verification tests
//===----------------------------------------------------------------------===//

/// T12: Dynamic throughput violation
static bool testDynamicThroughputViolated() {
  TDCEdgeSpec spec;
  spec.producerKernel = "a";
  spec.consumerKernel = "b";
  spec.dataTypeName = "f32";
  spec.throughput = "100";

  TDCEdgeSpecOrigin origin;
  origin.throughput = DimensionOrigin::USER_SPECIFIED;

  DynamicEdgeMetrics metrics;
  metrics.producerKernel = "a";
  metrics.consumerKernel = "b";
  metrics.sustainedThroughput = 80.0;

  auto report =
      verifyDynamic({spec}, {origin}, {}, {metrics}, {}, {});

  if (report.edgeResults[0].throughputSatisfied) {
    std::cerr << "FAIL: testDynamicThroughputViolated - expected violated\n";
    return false;
  }
  if (!report.edgeResults[0].achievedThroughput.has_value() ||
      *report.edgeResults[0].achievedThroughput != 80.0) {
    std::cerr << "FAIL: testDynamicThroughputViolated - wrong achieved value\n";
    return false;
  }

  std::cerr << "PASS: testDynamicThroughputViolated\n";
  return true;
}

/// T13: Dynamic throughput satisfied
static bool testDynamicThroughputSatisfied() {
  TDCEdgeSpec spec;
  spec.producerKernel = "a";
  spec.consumerKernel = "b";
  spec.dataTypeName = "f32";
  spec.throughput = "100";

  TDCEdgeSpecOrigin origin;
  origin.throughput = DimensionOrigin::USER_SPECIFIED;

  DynamicEdgeMetrics metrics;
  metrics.producerKernel = "a";
  metrics.consumerKernel = "b";
  metrics.sustainedThroughput = 120.0;

  auto report =
      verifyDynamic({spec}, {origin}, {}, {metrics}, {}, {});

  if (!report.edgeResults[0].throughputSatisfied) {
    std::cerr << "FAIL: testDynamicThroughputSatisfied - expected satisfied\n";
    return false;
  }
  if (*report.edgeResults[0].achievedThroughput != 120.0) {
    std::cerr << "FAIL: testDynamicThroughputSatisfied - wrong achieved\n";
    return false;
  }

  std::cerr << "PASS: testDynamicThroughputSatisfied\n";
  return true;
}

/// T14: Path latency violation
static bool testDynamicPathLatencyViolated() {
  TDCPathSpec path;
  path.startProducer = "a";
  path.startConsumer = "b";
  path.endProducer = "b";
  path.endConsumer = "c";
  path.latency = "256";

  DynamicPathMetrics metrics;
  metrics.startProducer = "a";
  metrics.startConsumer = "b";
  metrics.endProducer = "b";
  metrics.endConsumer = "c";
  metrics.observedLatency = 310;

  auto report =
      verifyDynamic({}, {}, {path}, {}, {metrics}, {});

  if (report.pathResults.size() != 1) {
    std::cerr << "FAIL: testDynamicPathLatencyViolated - wrong count\n";
    return false;
  }
  if (report.pathResults[0].latencySatisfied) {
    std::cerr << "FAIL: testDynamicPathLatencyViolated - expected violated\n";
    return false;
  }
  if (*report.pathResults[0].achievedLatency != 310) {
    std::cerr << "FAIL: testDynamicPathLatencyViolated - wrong achieved\n";
    return false;
  }

  std::cerr << "PASS: testDynamicPathLatencyViolated\n";
  return true;
}

/// T15: Path latency satisfied
static bool testDynamicPathLatencySatisfied() {
  TDCPathSpec path;
  path.startProducer = "a";
  path.startConsumer = "b";
  path.endProducer = "b";
  path.endConsumer = "c";
  path.latency = "256";

  DynamicPathMetrics metrics;
  metrics.startProducer = "a";
  metrics.startConsumer = "b";
  metrics.endProducer = "b";
  metrics.endConsumer = "c";
  metrics.observedLatency = 200;

  auto report =
      verifyDynamic({}, {}, {path}, {}, {metrics}, {});

  if (!report.pathResults[0].latencySatisfied) {
    std::cerr << "FAIL: testDynamicPathLatencySatisfied - expected satisfied\n";
    return false;
  }

  std::cerr << "PASS: testDynamicPathLatencySatisfied\n";
  return true;
}

/// T16: Dynamic FIFO ordering violation
static bool testDynamicOrderingViolated() {
  TDCEdgeSpec spec;
  spec.producerKernel = "a";
  spec.consumerKernel = "b";
  spec.dataTypeName = "f32";
  spec.ordering = Ordering::FIFO;

  TDCEdgeSpecOrigin origin;
  origin.ordering = DimensionOrigin::USER_SPECIFIED;

  DynamicEdgeMetrics metrics;
  metrics.producerKernel = "a";
  metrics.consumerKernel = "b";
  metrics.orderingViolationCount = 3;

  auto report =
      verifyDynamic({spec}, {origin}, {}, {metrics}, {}, {});

  if (report.edgeResults[0].orderingSatisfied) {
    std::cerr << "FAIL: testDynamicOrderingViolated - expected violated\n";
    return false;
  }

  std::cerr << "PASS: testDynamicOrderingViolated\n";
  return true;
}

//===----------------------------------------------------------------------===//
// Aggregate report tests
//===----------------------------------------------------------------------===//

/// T17: All satisfied -> clean report
static bool testAggregateAllSatisfied() {
  // Two edges, both with satisfied constraints
  TDCEdgeSpec e1;
  e1.producerKernel = "a";
  e1.consumerKernel = "b";
  e1.dataTypeName = "f32";
  e1.placement = Placement::LOCAL_SPM;

  TDCEdgeSpec e2;
  e2.producerKernel = "b";
  e2.consumerKernel = "c";
  e2.dataTypeName = "f32";
  e2.placement = Placement::SHARED_L2;

  TDCEdgeSpecOrigin o1;
  o1.placement = DimensionOrigin::USER_SPECIFIED;
  TDCEdgeSpecOrigin o2;
  o2.placement = DimensionOrigin::USER_SPECIFIED;

  BufferAllocationPlan plan;
  BufferAllocation a1;
  a1.contractEdgeName = "a->b";
  a1.location = BufferAllocation::SPM_PRODUCER;
  plan.allocations.push_back(a1);
  BufferAllocation a2;
  a2.contractEdgeName = "b->c";
  a2.location = BufferAllocation::SHARED_L2;
  plan.allocations.push_back(a2);

  TDCPathSpec path;
  path.startProducer = "a";
  path.startConsumer = "b";
  path.endProducer = "b";
  path.endConsumer = "c";
  path.latency = "500";

  DynamicPathMetrics pm;
  pm.startProducer = "a";
  pm.startConsumer = "b";
  pm.endProducer = "b";
  pm.endConsumer = "c";
  pm.observedLatency = 300;

  auto report = verifyContracts(
      {e1, e2}, {o1, o2}, {path}, plan, {}, {},
      std::optional<std::vector<DynamicEdgeMetrics>>(
          std::vector<DynamicEdgeMetrics>{}),
      std::optional<std::vector<DynamicPathMetrics>>({pm}),
      {});

  if (!report.allSatisfied) {
    std::cerr << "FAIL: testAggregateAllSatisfied - expected all satisfied\n";
    for (const auto &er : report.edgeResults)
      for (const auto &d : er.diagnostics)
        std::cerr << "  diag: " << d << "\n";
    for (const auto &pr : report.pathResults)
      for (const auto &d : pr.diagnostics)
        std::cerr << "  diag: " << d << "\n";
    return false;
  }

  std::cerr << "PASS: testAggregateAllSatisfied\n";
  return true;
}

/// T18: One failure makes report fail
static bool testAggregateOneFailure() {
  TDCEdgeSpec e1;
  e1.producerKernel = "a";
  e1.consumerKernel = "b";
  e1.dataTypeName = "f32";
  e1.placement = Placement::LOCAL_SPM;

  TDCEdgeSpec e2;
  e2.producerKernel = "b";
  e2.consumerKernel = "c";
  e2.dataTypeName = "f32";
  e2.placement = Placement::LOCAL_SPM;

  TDCEdgeSpecOrigin o1;
  o1.placement = DimensionOrigin::USER_SPECIFIED;
  TDCEdgeSpecOrigin o2;
  o2.placement = DimensionOrigin::USER_SPECIFIED;

  BufferAllocationPlan plan;
  BufferAllocation a1;
  a1.contractEdgeName = "a->b";
  a1.location = BufferAllocation::SPM_CONSUMER;
  plan.allocations.push_back(a1);
  BufferAllocation a2;
  a2.contractEdgeName = "b->c";
  a2.location = BufferAllocation::SHARED_L2; // Violates LOCAL_SPM
  plan.allocations.push_back(a2);

  auto report = verifyContracts(
      {e1, e2}, {o1, o2}, {}, plan, {}, {},
      std::nullopt, std::nullopt, {});

  if (report.allSatisfied) {
    std::cerr << "FAIL: testAggregateOneFailure - expected not satisfied\n";
    return false;
  }
  // First edge should pass, second should fail
  if (!report.edgeResults[0].placementSatisfied) {
    std::cerr << "FAIL: testAggregateOneFailure - first edge should pass\n";
    return false;
  }
  if (report.edgeResults[1].placementSatisfied) {
    std::cerr << "FAIL: testAggregateOneFailure - second edge should fail\n";
    return false;
  }

  std::cerr << "PASS: testAggregateOneFailure\n";
  return true;
}

/// T19: Static-only when no dynamic metrics
static bool testStaticOnlyNoDynamic() {
  TDCEdgeSpec spec;
  spec.producerKernel = "a";
  spec.consumerKernel = "b";
  spec.dataTypeName = "f32";
  spec.throughput = "100";
  spec.placement = Placement::SHARED_L2;

  TDCEdgeSpecOrigin origin;
  origin.throughput = DimensionOrigin::USER_SPECIFIED;
  origin.placement = DimensionOrigin::USER_SPECIFIED;

  BufferAllocationPlan plan;
  BufferAllocation alloc;
  alloc.contractEdgeName = "a->b";
  alloc.location = BufferAllocation::SHARED_L2;
  plan.allocations.push_back(alloc);

  // No dynamic metrics provided
  auto report = verifyContracts(
      {spec}, {origin}, {}, plan, {}, {},
      std::nullopt, std::nullopt, {});

  // Placement should be satisfied (static check)
  if (!report.edgeResults[0].placementSatisfied) {
    std::cerr << "FAIL: testStaticOnlyNoDynamic - placement should pass\n";
    return false;
  }
  // Throughput should be satisfied (not checked since no dynamic metrics)
  if (!report.edgeResults[0].throughputSatisfied) {
    std::cerr << "FAIL: testStaticOnlyNoDynamic - throughput should pass "
                 "(not checked)\n";
    return false;
  }

  std::cerr << "PASS: testStaticOnlyNoDynamic\n";
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

  // Inference tests
  run(testInferMissingOrdering);
  run(testInferPreservesOrdering);
  run(testInferMissingPlacement);
  run(testInferMissingThroughput);
  run(testInferMissingShape);
  run(testPathInvalidEdgeRef);
  run(testInferFullySpecified);

  // Static verification tests
  run(testStaticPlacementViolated);
  run(testStaticPlacementSatisfied);
  run(testStaticShapeViolated);
  run(testStaticInferredSkipped);

  // Dynamic verification tests
  run(testDynamicThroughputViolated);
  run(testDynamicThroughputSatisfied);
  run(testDynamicPathLatencyViolated);
  run(testDynamicPathLatencySatisfied);
  run(testDynamicOrderingViolated);

  // Aggregate report tests
  run(testAggregateAllSatisfied);
  run(testAggregateOneFailure);
  run(testStaticOnlyNoDynamic);

  std::cerr << "\nResults: " << passed << "/" << total << " tests passed\n";
  return (passed == total) ? 0 : 1;
}
