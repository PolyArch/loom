/// TDC contract inference and post-compilation verification tests.
///
/// Inference tests:
///   1. Infer_MissingOrdering_DefaultsFIFO
///   2. Infer_UserOrdering_Preserved
///   3. Infer_MissingPlacement_DefaultsAUTO
///   4. Infer_MissingThroughput_RemainsAbsent
///   5. Infer_MissingShape_RemainsAbsent
///   6. Infer_PathContract_InvalidEdgeRef_Error
///   7. Infer_FullySpecified_NoChanges
///
/// Static verification tests:
///   8.  StaticVerify_Placement_Violated
///   9.  StaticVerify_Placement_Satisfied
///  10.  StaticVerify_Shape_Violated
///  11.  StaticVerify_InferredDimension_Skipped
///
/// Dynamic verification tests:
///  12.  DynamicVerify_Throughput_Violated
///  13.  DynamicVerify_Throughput_Satisfied
///  14.  DynamicVerify_PathLatency_Violated
///  15.  DynamicVerify_PathLatency_Satisfied
///  16.  DynamicVerify_Ordering_FIFO_Violated
///
/// Aggregate report tests:
///  17.  VerifyContracts_AllSatisfied_CleanReport
///  18.  VerifyContracts_OneFailure_ReportFails
///  19.  VerifyContracts_NoDynamicMetrics_StaticOnly

#include "loom/SystemCompiler/TDCVerification.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

using namespace loom;

//===----------------------------------------------------------------------===//
// Helpers: build minimal static inputs for tests
//===----------------------------------------------------------------------===//

static StaticVerificationInputs makeEmptyStaticInputs() {
  StaticVerificationInputs si;
  si.assignment.feasible = true;
  si.bufferPlan.feasible = true;
  return si;
}

/// Make a buffer allocation entry for a given edge.
static BufferAllocation makeAlloc(const std::string &producer,
                                  const std::string &consumer,
                                  BufferAllocation::Location loc) {
  BufferAllocation alloc;
  alloc.contractEdgeName = producer + "_" + consumer;
  alloc.location = loc;
  alloc.sizeBytes = 1024;
  alloc.elementCount = 256;
  return alloc;
}

/// Make an EdgeTileDimensions entry.
static EdgeTileDimensions makeTileDims(const std::string &producer,
                                       const std::string &consumer,
                                       std::vector<int64_t> dims) {
  EdgeTileDimensions td;
  td.producerKernel = producer;
  td.consumerKernel = consumer;
  td.tileDims = std::move(dims);
  return td;
}

//===----------------------------------------------------------------------===//
// Inference tests
//===----------------------------------------------------------------------===//

/// T1: Missing ordering is filled with FIFO default.
static bool testInferMissingOrderingDefaultsFIFO() {
  TDCEdgeSpec spec;
  spec.producerKernel = "A";
  spec.consumerKernel = "B";
  spec.dataTypeName = "f32";
  spec.throughput = "64";
  // ordering intentionally left as nullopt

  TDCContractInferrer inferrer;
  auto result = inferrer.infer({spec}, {});

  if (result.hasErrors()) {
    std::cerr << "FAIL: testInferMissingOrderingDefaultsFIFO - errors\n";
    return false;
  }
  if (result.edgeSpecs.size() != 1 || result.edgeOrigins.size() != 1) {
    std::cerr << "FAIL: testInferMissingOrderingDefaultsFIFO - count\n";
    return false;
  }
  if (!result.edgeSpecs[0].ordering.has_value() ||
      *result.edgeSpecs[0].ordering != Ordering::FIFO) {
    std::cerr << "FAIL: testInferMissingOrderingDefaultsFIFO - value\n";
    return false;
  }
  if (result.edgeOrigins[0].ordering != DimensionOrigin::INFERRED) {
    std::cerr << "FAIL: testInferMissingOrderingDefaultsFIFO - origin\n";
    return false;
  }

  std::cerr << "PASS: testInferMissingOrderingDefaultsFIFO\n";
  return true;
}

/// T2: User-specified ordering is preserved.
static bool testInferUserOrderingPreserved() {
  TDCEdgeSpec spec;
  spec.producerKernel = "A";
  spec.consumerKernel = "B";
  spec.dataTypeName = "f32";
  spec.ordering = Ordering::UNORDERED;

  TDCContractInferrer inferrer;
  auto result = inferrer.infer({spec}, {});

  if (*result.edgeSpecs[0].ordering != Ordering::UNORDERED) {
    std::cerr << "FAIL: testInferUserOrderingPreserved - value\n";
    return false;
  }
  if (result.edgeOrigins[0].ordering != DimensionOrigin::USER_SPECIFIED) {
    std::cerr << "FAIL: testInferUserOrderingPreserved - origin\n";
    return false;
  }

  std::cerr << "PASS: testInferUserOrderingPreserved\n";
  return true;
}

/// T3: Missing placement defaults to AUTO.
static bool testInferMissingPlacementDefaultsAUTO() {
  TDCEdgeSpec spec;
  spec.producerKernel = "A";
  spec.consumerKernel = "B";
  spec.dataTypeName = "f32";
  // placement intentionally left as nullopt

  TDCContractInferrer inferrer;
  auto result = inferrer.infer({spec}, {});

  if (!result.edgeSpecs[0].placement.has_value() ||
      *result.edgeSpecs[0].placement != Placement::AUTO) {
    std::cerr << "FAIL: testInferMissingPlacementDefaultsAUTO - value\n";
    return false;
  }
  if (result.edgeOrigins[0].placement != DimensionOrigin::INFERRED) {
    std::cerr << "FAIL: testInferMissingPlacementDefaultsAUTO - origin\n";
    return false;
  }

  std::cerr << "PASS: testInferMissingPlacementDefaultsAUTO\n";
  return true;
}

/// T4: Missing throughput remains absent (no default).
static bool testInferMissingThroughputRemainsAbsent() {
  TDCEdgeSpec spec;
  spec.producerKernel = "A";
  spec.consumerKernel = "B";
  spec.dataTypeName = "f32";
  // throughput intentionally left as nullopt

  TDCContractInferrer inferrer;
  auto result = inferrer.infer({spec}, {});

  if (result.edgeSpecs[0].throughput.has_value()) {
    std::cerr << "FAIL: testInferMissingThroughputRemainsAbsent - value\n";
    return false;
  }
  if (result.edgeOrigins[0].throughput != DimensionOrigin::ABSENT) {
    std::cerr << "FAIL: testInferMissingThroughputRemainsAbsent - origin\n";
    return false;
  }

  std::cerr << "PASS: testInferMissingThroughputRemainsAbsent\n";
  return true;
}

/// T5: Missing shape remains absent (no default).
static bool testInferMissingShapeRemainsAbsent() {
  TDCEdgeSpec spec;
  spec.producerKernel = "A";
  spec.consumerKernel = "B";
  spec.dataTypeName = "f32";
  // shape intentionally left as nullopt

  TDCContractInferrer inferrer;
  auto result = inferrer.infer({spec}, {});

  if (result.edgeSpecs[0].shape.has_value()) {
    std::cerr << "FAIL: testInferMissingShapeRemainsAbsent - value\n";
    return false;
  }
  if (result.edgeOrigins[0].shape != DimensionOrigin::ABSENT) {
    std::cerr << "FAIL: testInferMissingShapeRemainsAbsent - origin\n";
    return false;
  }

  std::cerr << "PASS: testInferMissingShapeRemainsAbsent\n";
  return true;
}

/// T6: Path contract referencing non-existent edge produces error.
static bool testInferPathContractInvalidEdgeRefError() {
  TDCEdgeSpec edge;
  edge.producerKernel = "A";
  edge.consumerKernel = "B";
  edge.dataTypeName = "f32";

  TDCPathSpec path;
  path.startProducer = "nonexistent";
  path.startConsumer = "B";
  path.endProducer = "A";
  path.endConsumer = "B";
  path.latency = "256";

  TDCContractInferrer inferrer;
  auto result = inferrer.infer({edge}, {path});

  if (!result.hasErrors()) {
    std::cerr << "FAIL: testInferPathContractInvalidEdgeRefError - "
                 "expected error\n";
    return false;
  }
  // Should mention "nonexistent"
  bool foundMention = false;
  for (const auto &err : result.errors) {
    if (err.find("nonexistent") != std::string::npos) {
      foundMention = true;
      break;
    }
  }
  if (!foundMention) {
    std::cerr << "FAIL: testInferPathContractInvalidEdgeRefError - "
                 "error message should mention nonexistent edge\n";
    return false;
  }

  std::cerr << "PASS: testInferPathContractInvalidEdgeRefError\n";
  return true;
}

/// T7: Fully specified edge spec triggers no inference changes.
static bool testInferFullySpecifiedNoChanges() {
  TDCEdgeSpec spec;
  spec.producerKernel = "A";
  spec.consumerKernel = "B";
  spec.dataTypeName = "f32";
  spec.ordering = Ordering::UNORDERED;
  spec.throughput = "128";
  spec.placement = Placement::SHARED_L2;
  spec.shape = "[64, 64]";

  TDCContractInferrer inferrer;
  auto result = inferrer.infer({spec}, {});

  if (result.hasErrors()) {
    std::cerr << "FAIL: testInferFullySpecifiedNoChanges - errors\n";
    return false;
  }

  const auto &inferred = result.edgeSpecs[0];
  const auto &origin = result.edgeOrigins[0];

  if (*inferred.ordering != Ordering::UNORDERED) {
    std::cerr << "FAIL: testInferFullySpecifiedNoChanges - ordering\n";
    return false;
  }
  if (*inferred.throughput != "128") {
    std::cerr << "FAIL: testInferFullySpecifiedNoChanges - throughput\n";
    return false;
  }
  if (*inferred.placement != Placement::SHARED_L2) {
    std::cerr << "FAIL: testInferFullySpecifiedNoChanges - placement\n";
    return false;
  }
  if (*inferred.shape != "[64, 64]") {
    std::cerr << "FAIL: testInferFullySpecifiedNoChanges - shape\n";
    return false;
  }

  if (origin.ordering != DimensionOrigin::USER_SPECIFIED ||
      origin.throughput != DimensionOrigin::USER_SPECIFIED ||
      origin.placement != DimensionOrigin::USER_SPECIFIED ||
      origin.shape != DimensionOrigin::USER_SPECIFIED) {
    std::cerr << "FAIL: testInferFullySpecifiedNoChanges - origins\n";
    return false;
  }

  std::cerr << "PASS: testInferFullySpecifiedNoChanges\n";
  return true;
}

//===----------------------------------------------------------------------===//
// Static verification tests
//===----------------------------------------------------------------------===//

/// T8: Placement violation detected.
static bool testStaticVerifyPlacementViolated() {
  TDCEdgeSpec spec;
  spec.producerKernel = "A";
  spec.consumerKernel = "B";
  spec.dataTypeName = "f32";
  spec.placement = Placement::LOCAL_SPM;

  TDCEdgeSpecOrigin origin;
  origin.placement = DimensionOrigin::USER_SPECIFIED;

  auto si = makeEmptyStaticInputs();
  si.bufferPlan.allocations.push_back(
      makeAlloc("A", "B", BufferAllocation::SHARED_L2));

  auto report =
      verifyContracts({spec}, {origin}, {}, si, nullptr, nullptr);

  if (report.edgeResults.size() != 1) {
    std::cerr << "FAIL: testStaticVerifyPlacementViolated - count\n";
    return false;
  }
  if (report.edgeResults[0].placementSatisfied) {
    std::cerr << "FAIL: testStaticVerifyPlacementViolated - should fail\n";
    return false;
  }
  if (report.edgeResults[0].diagnostic.empty()) {
    std::cerr << "FAIL: testStaticVerifyPlacementViolated - no diagnostic\n";
    return false;
  }
  if (report.allSatisfied) {
    std::cerr << "FAIL: testStaticVerifyPlacementViolated - allSatisfied\n";
    return false;
  }

  std::cerr << "PASS: testStaticVerifyPlacementViolated\n";
  return true;
}

/// T9: Placement satisfied (LOCAL_SPM matches SPM_CONSUMER).
static bool testStaticVerifyPlacementSatisfied() {
  TDCEdgeSpec spec;
  spec.producerKernel = "A";
  spec.consumerKernel = "B";
  spec.dataTypeName = "f32";
  spec.placement = Placement::LOCAL_SPM;

  TDCEdgeSpecOrigin origin;
  origin.placement = DimensionOrigin::USER_SPECIFIED;

  auto si = makeEmptyStaticInputs();
  si.bufferPlan.allocations.push_back(
      makeAlloc("A", "B", BufferAllocation::SPM_CONSUMER));

  auto report =
      verifyContracts({spec}, {origin}, {}, si, nullptr, nullptr);

  if (!report.edgeResults[0].placementSatisfied) {
    std::cerr << "FAIL: testStaticVerifyPlacementSatisfied - should pass\n";
    return false;
  }

  std::cerr << "PASS: testStaticVerifyPlacementSatisfied\n";
  return true;
}

/// T10: Shape violation detected (dimension mismatch).
static bool testStaticVerifyShapeViolated() {
  TDCEdgeSpec spec;
  spec.producerKernel = "A";
  spec.consumerKernel = "B";
  spec.dataTypeName = "f32";
  spec.shape = "[128, 256]";

  TDCEdgeSpecOrigin origin;
  origin.shape = DimensionOrigin::USER_SPECIFIED;

  auto si = makeEmptyStaticInputs();
  si.tileDimensions.push_back(makeTileDims("A", "B", {128, 128}));

  auto report =
      verifyContracts({spec}, {origin}, {}, si, nullptr, nullptr);

  if (report.edgeResults[0].shapeSatisfied) {
    std::cerr << "FAIL: testStaticVerifyShapeViolated - should fail\n";
    return false;
  }
  if (report.edgeResults[0].diagnostic.empty()) {
    std::cerr << "FAIL: testStaticVerifyShapeViolated - no diagnostic\n";
    return false;
  }

  std::cerr << "PASS: testStaticVerifyShapeViolated\n";
  return true;
}

/// T11: Inferred dimension is skipped in verification.
static bool testStaticVerifyInferredDimensionSkipped() {
  TDCEdgeSpec spec;
  spec.producerKernel = "A";
  spec.consumerKernel = "B";
  spec.dataTypeName = "f32";
  spec.ordering = Ordering::FIFO;

  TDCEdgeSpecOrigin origin;
  // Mark ordering as INFERRED (not user-specified).
  origin.ordering = DimensionOrigin::INFERRED;

  auto si = makeEmptyStaticInputs();
  // Even with a potentially violating schedule, the inferred dimension
  // should be skipped.

  auto report =
      verifyContracts({spec}, {origin}, {}, si, nullptr, nullptr);

  if (!report.edgeResults[0].orderingSatisfied) {
    std::cerr << "FAIL: testStaticVerifyInferredDimensionSkipped - "
                 "should be skipped\n";
    return false;
  }

  std::cerr << "PASS: testStaticVerifyInferredDimensionSkipped\n";
  return true;
}

//===----------------------------------------------------------------------===//
// Dynamic verification tests
//===----------------------------------------------------------------------===//

/// T12: Throughput violation detected.
static bool testDynamicVerifyThroughputViolated() {
  TDCEdgeSpec spec;
  spec.producerKernel = "A";
  spec.consumerKernel = "B";
  spec.dataTypeName = "f32";
  spec.throughput = "100";

  TDCEdgeSpecOrigin origin;
  origin.throughput = DimensionOrigin::USER_SPECIFIED;

  auto si = makeEmptyStaticInputs();

  DynamicEdgeMetrics dm;
  dm.producerKernel = "A";
  dm.consumerKernel = "B";
  dm.sustainedThroughput = 80.0;
  dm.orderingViolationCount = 0;

  std::vector<DynamicEdgeMetrics> dynamicEdge = {dm};

  auto report =
      verifyContracts({spec}, {origin}, {}, si, &dynamicEdge, nullptr);

  if (report.edgeResults[0].throughputSatisfied) {
    std::cerr << "FAIL: testDynamicVerifyThroughputViolated - should fail\n";
    return false;
  }
  if (!report.edgeResults[0].achievedThroughput.has_value() ||
      std::abs(*report.edgeResults[0].achievedThroughput - 80.0) > 0.001) {
    std::cerr << "FAIL: testDynamicVerifyThroughputViolated - achieved\n";
    return false;
  }
  if (report.edgeResults[0].diagnostic.empty()) {
    std::cerr << "FAIL: testDynamicVerifyThroughputViolated - diagnostic\n";
    return false;
  }

  std::cerr << "PASS: testDynamicVerifyThroughputViolated\n";
  return true;
}

/// T13: Throughput satisfied (achieved >= specified).
static bool testDynamicVerifyThroughputSatisfied() {
  TDCEdgeSpec spec;
  spec.producerKernel = "A";
  spec.consumerKernel = "B";
  spec.dataTypeName = "f32";
  spec.throughput = "100";

  TDCEdgeSpecOrigin origin;
  origin.throughput = DimensionOrigin::USER_SPECIFIED;

  auto si = makeEmptyStaticInputs();

  DynamicEdgeMetrics dm;
  dm.producerKernel = "A";
  dm.consumerKernel = "B";
  dm.sustainedThroughput = 120.0;
  dm.orderingViolationCount = 0;

  std::vector<DynamicEdgeMetrics> dynamicEdge = {dm};

  auto report =
      verifyContracts({spec}, {origin}, {}, si, &dynamicEdge, nullptr);

  if (!report.edgeResults[0].throughputSatisfied) {
    std::cerr << "FAIL: testDynamicVerifyThroughputSatisfied - should pass\n";
    return false;
  }
  if (!report.edgeResults[0].achievedThroughput.has_value() ||
      std::abs(*report.edgeResults[0].achievedThroughput - 120.0) > 0.001) {
    std::cerr << "FAIL: testDynamicVerifyThroughputSatisfied - achieved\n";
    return false;
  }

  std::cerr << "PASS: testDynamicVerifyThroughputSatisfied\n";
  return true;
}

/// T14: Path latency violation detected.
static bool testDynamicVerifyPathLatencyViolated() {
  TDCEdgeSpec edgeAB;
  edgeAB.producerKernel = "A";
  edgeAB.consumerKernel = "B";
  edgeAB.dataTypeName = "f32";

  TDCEdgeSpec edgeBC;
  edgeBC.producerKernel = "B";
  edgeBC.consumerKernel = "C";
  edgeBC.dataTypeName = "f32";

  TDCPathSpec path;
  path.startProducer = "A";
  path.startConsumer = "B";
  path.endProducer = "B";
  path.endConsumer = "C";
  path.latency = "256";

  TDCEdgeSpecOrigin originDefault;
  auto si = makeEmptyStaticInputs();

  DynamicPathMetrics dpm;
  dpm.startProducer = "A";
  dpm.startConsumer = "B";
  dpm.endProducer = "B";
  dpm.endConsumer = "C";
  dpm.observedLatency = 310;

  std::vector<DynamicPathMetrics> dynamicPath = {dpm};

  auto report = verifyContracts({edgeAB, edgeBC},
                                {originDefault, originDefault}, {path}, si,
                                nullptr, &dynamicPath);

  if (report.pathResults.size() != 1) {
    std::cerr << "FAIL: testDynamicVerifyPathLatencyViolated - count\n";
    return false;
  }
  if (report.pathResults[0].latencySatisfied) {
    std::cerr << "FAIL: testDynamicVerifyPathLatencyViolated - should fail\n";
    return false;
  }
  if (!report.pathResults[0].achievedLatency.has_value() ||
      *report.pathResults[0].achievedLatency != 310) {
    std::cerr << "FAIL: testDynamicVerifyPathLatencyViolated - achieved\n";
    return false;
  }
  if (report.pathResults[0].diagnostic.empty()) {
    std::cerr << "FAIL: testDynamicVerifyPathLatencyViolated - diagnostic\n";
    return false;
  }

  std::cerr << "PASS: testDynamicVerifyPathLatencyViolated\n";
  return true;
}

/// T15: Path latency satisfied.
static bool testDynamicVerifyPathLatencySatisfied() {
  TDCEdgeSpec edgeAB;
  edgeAB.producerKernel = "A";
  edgeAB.consumerKernel = "B";
  edgeAB.dataTypeName = "f32";

  TDCEdgeSpec edgeBC;
  edgeBC.producerKernel = "B";
  edgeBC.consumerKernel = "C";
  edgeBC.dataTypeName = "f32";

  TDCPathSpec path;
  path.startProducer = "A";
  path.startConsumer = "B";
  path.endProducer = "B";
  path.endConsumer = "C";
  path.latency = "256";

  TDCEdgeSpecOrigin originDefault;
  auto si = makeEmptyStaticInputs();

  DynamicPathMetrics dpm;
  dpm.startProducer = "A";
  dpm.startConsumer = "B";
  dpm.endProducer = "B";
  dpm.endConsumer = "C";
  dpm.observedLatency = 200;

  std::vector<DynamicPathMetrics> dynamicPath = {dpm};

  auto report = verifyContracts({edgeAB, edgeBC},
                                {originDefault, originDefault}, {path}, si,
                                nullptr, &dynamicPath);

  if (!report.pathResults[0].latencySatisfied) {
    std::cerr << "FAIL: testDynamicVerifyPathLatencySatisfied - should pass\n";
    return false;
  }
  if (!report.pathResults[0].achievedLatency.has_value() ||
      *report.pathResults[0].achievedLatency != 200) {
    std::cerr << "FAIL: testDynamicVerifyPathLatencySatisfied - achieved\n";
    return false;
  }

  std::cerr << "PASS: testDynamicVerifyPathLatencySatisfied\n";
  return true;
}

/// T16: Dynamic ordering FIFO violation detected.
static bool testDynamicVerifyOrderingFIFOViolated() {
  TDCEdgeSpec spec;
  spec.producerKernel = "A";
  spec.consumerKernel = "B";
  spec.dataTypeName = "f32";
  spec.ordering = Ordering::FIFO;

  TDCEdgeSpecOrigin origin;
  origin.ordering = DimensionOrigin::USER_SPECIFIED;

  auto si = makeEmptyStaticInputs();

  DynamicEdgeMetrics dm;
  dm.producerKernel = "A";
  dm.consumerKernel = "B";
  dm.sustainedThroughput = 100.0;
  dm.orderingViolationCount = 3;

  std::vector<DynamicEdgeMetrics> dynamicEdge = {dm};

  auto report =
      verifyContracts({spec}, {origin}, {}, si, &dynamicEdge, nullptr);

  if (report.edgeResults[0].orderingSatisfied) {
    std::cerr << "FAIL: testDynamicVerifyOrderingFIFOViolated - should fail\n";
    return false;
  }
  if (report.edgeResults[0].diagnostic.find("3") == std::string::npos) {
    std::cerr << "FAIL: testDynamicVerifyOrderingFIFOViolated - diagnostic "
                 "should mention violation count\n";
    return false;
  }

  std::cerr << "PASS: testDynamicVerifyOrderingFIFOViolated\n";
  return true;
}

//===----------------------------------------------------------------------===//
// Aggregate report tests
//===----------------------------------------------------------------------===//

/// T17: All satisfied produces clean report.
static bool testVerifyContractsAllSatisfiedCleanReport() {
  // Two edges, one path, all satisfied.
  TDCEdgeSpec edgeAB;
  edgeAB.producerKernel = "A";
  edgeAB.consumerKernel = "B";
  edgeAB.dataTypeName = "f32";
  edgeAB.placement = Placement::LOCAL_SPM;

  TDCEdgeSpec edgeBC;
  edgeBC.producerKernel = "B";
  edgeBC.consumerKernel = "C";
  edgeBC.dataTypeName = "f32";
  edgeBC.placement = Placement::SHARED_L2;

  TDCEdgeSpecOrigin originAB;
  originAB.placement = DimensionOrigin::USER_SPECIFIED;

  TDCEdgeSpecOrigin originBC;
  originBC.placement = DimensionOrigin::USER_SPECIFIED;

  TDCPathSpec path;
  path.startProducer = "A";
  path.startConsumer = "B";
  path.endProducer = "B";
  path.endConsumer = "C";
  path.latency = "500";

  auto si = makeEmptyStaticInputs();
  si.bufferPlan.allocations.push_back(
      makeAlloc("A", "B", BufferAllocation::SPM_CONSUMER));
  si.bufferPlan.allocations.push_back(
      makeAlloc("B", "C", BufferAllocation::SHARED_L2));

  DynamicEdgeMetrics dmAB;
  dmAB.producerKernel = "A";
  dmAB.consumerKernel = "B";
  dmAB.sustainedThroughput = 200.0;
  dmAB.orderingViolationCount = 0;

  DynamicEdgeMetrics dmBC;
  dmBC.producerKernel = "B";
  dmBC.consumerKernel = "C";
  dmBC.sustainedThroughput = 150.0;
  dmBC.orderingViolationCount = 0;

  DynamicPathMetrics dpm;
  dpm.startProducer = "A";
  dpm.startConsumer = "B";
  dpm.endProducer = "B";
  dpm.endConsumer = "C";
  dpm.observedLatency = 300;

  std::vector<DynamicEdgeMetrics> dynamicEdge = {dmAB, dmBC};
  std::vector<DynamicPathMetrics> dynamicPath = {dpm};

  auto report =
      verifyContracts({edgeAB, edgeBC}, {originAB, originBC}, {path}, si,
                      &dynamicEdge, &dynamicPath);

  if (!report.allSatisfied) {
    std::cerr << "FAIL: testVerifyContractsAllSatisfiedCleanReport - "
                 "allSatisfied should be true\n";
    for (const auto &er : report.edgeResults) {
      if (!er.diagnostic.empty())
        std::cerr << "  diagnostic: " << er.diagnostic << "\n";
    }
    return false;
  }
  if (report.edgeResults.size() != 2 || report.pathResults.size() != 1) {
    std::cerr << "FAIL: testVerifyContractsAllSatisfiedCleanReport - "
                 "result counts\n";
    return false;
  }

  std::cerr << "PASS: testVerifyContractsAllSatisfiedCleanReport\n";
  return true;
}

/// T18: One placement failure makes report fail.
static bool testVerifyContractsOneFailureReportFails() {
  TDCEdgeSpec edgeAB;
  edgeAB.producerKernel = "A";
  edgeAB.consumerKernel = "B";
  edgeAB.dataTypeName = "f32";
  edgeAB.placement = Placement::LOCAL_SPM;

  TDCEdgeSpec edgeBC;
  edgeBC.producerKernel = "B";
  edgeBC.consumerKernel = "C";
  edgeBC.dataTypeName = "f32";
  edgeBC.placement = Placement::LOCAL_SPM;

  TDCEdgeSpecOrigin originAB;
  originAB.placement = DimensionOrigin::USER_SPECIFIED;

  TDCEdgeSpecOrigin originBC;
  originBC.placement = DimensionOrigin::USER_SPECIFIED;

  auto si = makeEmptyStaticInputs();
  // First edge: correctly placed in SPM.
  si.bufferPlan.allocations.push_back(
      makeAlloc("A", "B", BufferAllocation::SPM_PRODUCER));
  // Second edge: incorrectly placed in SHARED_L2.
  si.bufferPlan.allocations.push_back(
      makeAlloc("B", "C", BufferAllocation::SHARED_L2));

  auto report = verifyContracts({edgeAB, edgeBC}, {originAB, originBC}, {},
                                si, nullptr, nullptr);

  if (report.allSatisfied) {
    std::cerr << "FAIL: testVerifyContractsOneFailureReportFails - "
                 "should not be allSatisfied\n";
    return false;
  }
  // First edge should pass.
  if (!report.edgeResults[0].placementSatisfied) {
    std::cerr << "FAIL: testVerifyContractsOneFailureReportFails - "
                 "first edge should pass\n";
    return false;
  }
  // Second edge should fail.
  if (report.edgeResults[1].placementSatisfied) {
    std::cerr << "FAIL: testVerifyContractsOneFailureReportFails - "
                 "second edge should fail\n";
    return false;
  }

  std::cerr << "PASS: testVerifyContractsOneFailureReportFails\n";
  return true;
}

/// T19: Static-only verification when no dynamic metrics provided.
static bool testVerifyContractsNoDynamicMetricsStaticOnly() {
  TDCEdgeSpec edgeAB;
  edgeAB.producerKernel = "A";
  edgeAB.consumerKernel = "B";
  edgeAB.dataTypeName = "f32";
  edgeAB.throughput = "100";
  edgeAB.placement = Placement::LOCAL_SPM;

  TDCEdgeSpec edgeBC;
  edgeBC.producerKernel = "B";
  edgeBC.consumerKernel = "C";
  edgeBC.dataTypeName = "f32";
  edgeBC.throughput = "50";

  TDCEdgeSpecOrigin originAB;
  originAB.throughput = DimensionOrigin::USER_SPECIFIED;
  originAB.placement = DimensionOrigin::USER_SPECIFIED;

  TDCEdgeSpecOrigin originBC;
  originBC.throughput = DimensionOrigin::USER_SPECIFIED;

  auto si = makeEmptyStaticInputs();
  si.bufferPlan.allocations.push_back(
      makeAlloc("A", "B", BufferAllocation::SPM_CONSUMER));

  // No dynamic metrics provided (nullptr).
  auto report = verifyContracts({edgeAB, edgeBC}, {originAB, originBC}, {},
                                si, nullptr, nullptr);

  // Throughput is not checked without dynamic metrics, so it should be true.
  if (!report.edgeResults[0].throughputSatisfied) {
    std::cerr << "FAIL: testVerifyContractsNoDynamicMetricsStaticOnly - "
                 "throughput should be true without dynamic metrics\n";
    return false;
  }
  if (!report.edgeResults[1].throughputSatisfied) {
    std::cerr << "FAIL: testVerifyContractsNoDynamicMetricsStaticOnly - "
                 "second edge throughput should be true\n";
    return false;
  }
  // Static placement should still be checked.
  if (!report.edgeResults[0].placementSatisfied) {
    std::cerr << "FAIL: testVerifyContractsNoDynamicMetricsStaticOnly - "
                 "placement should pass\n";
    return false;
  }

  std::cerr << "PASS: testVerifyContractsNoDynamicMetricsStaticOnly\n";
  return true;
}

//===----------------------------------------------------------------------===//
// Main
//===----------------------------------------------------------------------===//

int main() {
  int passed = 0;
  int total = 0;

  auto run = [&](bool (*test)()) {
    total++;
    if (test())
      passed++;
  };

  // Inference tests
  run(testInferMissingOrderingDefaultsFIFO);
  run(testInferUserOrderingPreserved);
  run(testInferMissingPlacementDefaultsAUTO);
  run(testInferMissingThroughputRemainsAbsent);
  run(testInferMissingShapeRemainsAbsent);
  run(testInferPathContractInvalidEdgeRefError);
  run(testInferFullySpecifiedNoChanges);

  // Static verification tests
  run(testStaticVerifyPlacementViolated);
  run(testStaticVerifyPlacementSatisfied);
  run(testStaticVerifyShapeViolated);
  run(testStaticVerifyInferredDimensionSkipped);

  // Dynamic verification tests
  run(testDynamicVerifyThroughputViolated);
  run(testDynamicVerifyThroughputSatisfied);
  run(testDynamicVerifyPathLatencyViolated);
  run(testDynamicVerifyPathLatencySatisfied);
  run(testDynamicVerifyOrderingFIFOViolated);

  // Aggregate report tests
  run(testVerifyContractsAllSatisfiedCleanReport);
  run(testVerifyContractsOneFailureReportFails);
  run(testVerifyContractsNoDynamicMetricsStaticOnly);

  std::cerr << "\nResults: " << passed << "/" << total << " tests passed\n";
  return (passed == total) ? 0 : 1;
}
