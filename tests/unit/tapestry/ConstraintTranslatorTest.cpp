/// ContractConstraintTranslator tests: edge translation, path translation,
/// batch translation, legacy conversion, and unconstrained edges.
///
/// Tests:
///  1. Fully constrained edge produces 4 constraints
///  2. Partially constrained edge produces only set dimensions
///  3. Unconstrained edge produces zero constraints
///  4. Path spec produces latency constraint
///  5. Path spec with empty latency produces no constraints
///  6. Batch translation merges edge and path constraints
///  7. Legacy ContractSpec to TDCEdgeSpec conversion

#include "loom/SystemCompiler/ContractConstraintTranslator.h"

#include <cassert>
#include <iostream>
#include <string>

using namespace loom;

/// T1: Fully constrained edge produces 4 constraints (one per dimension).
static bool testFullEdgeTranslation() {
  TDCEdgeSpec spec;
  spec.producerKernel = "matmul";
  spec.consumerKernel = "softmax";
  spec.dataTypeName = "f32";
  spec.ordering = Ordering::FIFO;
  spec.throughput = "batch * hidden / 1000";
  spec.placement = Placement::SHARED_L2;
  spec.shape = "[128, hidden_dim]";

  auto constraints = translateEdgeConstraints(spec);

  if (constraints.size() != 4) {
    std::cerr << "FAIL: testFullEdgeTranslation - expected 4, got "
              << constraints.size() << "\n";
    return false;
  }

  // Verify dimensions are present.
  bool hasOrdering = false, hasThroughput = false;
  bool hasPlacement = false, hasShape = false;
  for (const auto &c : constraints) {
    if (c.dimension == "ordering")
      hasOrdering = true;
    else if (c.dimension == "throughput")
      hasThroughput = true;
    else if (c.dimension == "placement")
      hasPlacement = true;
    else if (c.dimension == "shape")
      hasShape = true;
  }

  if (!hasOrdering || !hasThroughput || !hasPlacement || !hasShape) {
    std::cerr << "FAIL: testFullEdgeTranslation - missing dimensions\n";
    return false;
  }

  std::cerr << "PASS: testFullEdgeTranslation\n";
  return true;
}

/// T2: Partially constrained edge produces only set dimensions.
static bool testPartialEdgeTranslation() {
  TDCEdgeSpec spec;
  spec.producerKernel = "conv";
  spec.consumerKernel = "pool";
  spec.dataTypeName = "f16";
  spec.ordering = Ordering::UNORDERED;
  // throughput, placement, shape unset.

  auto constraints = translateEdgeConstraints(spec);

  if (constraints.size() != 1) {
    std::cerr << "FAIL: testPartialEdgeTranslation - expected 1, got "
              << constraints.size() << "\n";
    return false;
  }

  if (constraints[0].dimension != "ordering") {
    std::cerr << "FAIL: testPartialEdgeTranslation - wrong dimension: "
              << constraints[0].dimension << "\n";
    return false;
  }

  if (constraints[0].enumValue != "UNORDERED") {
    std::cerr << "FAIL: testPartialEdgeTranslation - wrong enum value: "
              << constraints[0].enumValue << "\n";
    return false;
  }

  std::cerr << "PASS: testPartialEdgeTranslation\n";
  return true;
}

/// T3: Unconstrained edge produces zero constraints.
static bool testUnconstrainedEdge() {
  TDCEdgeSpec spec;
  spec.producerKernel = "fft";
  spec.consumerKernel = "ifft";
  spec.dataTypeName = "c64";

  auto constraints = translateEdgeConstraints(spec);

  if (!constraints.empty()) {
    std::cerr << "FAIL: testUnconstrainedEdge - expected 0, got "
              << constraints.size() << "\n";
    return false;
  }

  std::cerr << "PASS: testUnconstrainedEdge\n";
  return true;
}

/// T4: Path spec produces latency constraint.
static bool testPathTranslation() {
  TDCPathSpec spec;
  spec.startProducer = "matmul";
  spec.startConsumer = "softmax";
  spec.endProducer = "softmax";
  spec.endConsumer = "relu";
  spec.latency = "4 * tile_m";

  auto constraints = translatePathConstraints(spec);

  if (constraints.size() != 1) {
    std::cerr << "FAIL: testPathTranslation - expected 1, got "
              << constraints.size() << "\n";
    return false;
  }

  if (constraints[0].dimension != "latency") {
    std::cerr << "FAIL: testPathTranslation - wrong dimension\n";
    return false;
  }

  if (constraints[0].expression != "4 * tile_m") {
    std::cerr << "FAIL: testPathTranslation - wrong expression: "
              << constraints[0].expression << "\n";
    return false;
  }

  std::cerr << "PASS: testPathTranslation\n";
  return true;
}

/// T5: Path spec with empty latency produces no constraints.
static bool testPathEmptyLatency() {
  TDCPathSpec spec;
  spec.startProducer = "a";
  spec.startConsumer = "b";
  spec.endProducer = "c";
  spec.endConsumer = "d";
  spec.latency = "";

  auto constraints = translatePathConstraints(spec);

  if (!constraints.empty()) {
    std::cerr << "FAIL: testPathEmptyLatency - expected 0, got "
              << constraints.size() << "\n";
    return false;
  }

  std::cerr << "PASS: testPathEmptyLatency\n";
  return true;
}

/// T6: Batch translation merges edge and path constraints.
static bool testBatchTranslation() {
  TDCEdgeSpec e1;
  e1.producerKernel = "a";
  e1.consumerKernel = "b";
  e1.dataTypeName = "f32";
  e1.ordering = Ordering::FIFO;

  TDCEdgeSpec e2;
  e2.producerKernel = "b";
  e2.consumerKernel = "c";
  e2.dataTypeName = "f32";
  e2.placement = Placement::EXTERNAL;
  e2.throughput = "1024";

  TDCPathSpec p1;
  p1.startProducer = "a";
  p1.startConsumer = "b";
  p1.endProducer = "b";
  p1.endConsumer = "c";
  p1.latency = "100";

  auto all = translateAllConstraints({e1, e2}, {p1});

  // e1: 1 (ordering), e2: 2 (placement + throughput), p1: 1 (latency) = 4
  if (all.size() != 4) {
    std::cerr << "FAIL: testBatchTranslation - expected 4, got " << all.size()
              << "\n";
    return false;
  }

  std::cerr << "PASS: testBatchTranslation\n";
  return true;
}

/// T7: Legacy ContractSpec to TDCEdgeSpec conversion.
static bool testLegacyConversion() {
  ContractSpec legacy;
  legacy.producerKernel = "matmul";
  legacy.consumerKernel = "relu";
  legacy.dataTypeName = "f32";
  legacy.ordering = Ordering::UNORDERED;
  legacy.visibility = Placement::SHARED_L2;

  TDCEdgeSpec spec = contractSpecToEdgeSpec(legacy);

  if (spec.producerKernel != "matmul" || spec.consumerKernel != "relu") {
    std::cerr << "FAIL: testLegacyConversion - kernel names\n";
    return false;
  }

  if (spec.dataTypeName != "f32") {
    std::cerr << "FAIL: testLegacyConversion - dataTypeName\n";
    return false;
  }

  if (!spec.ordering.has_value() || *spec.ordering != Ordering::UNORDERED) {
    std::cerr << "FAIL: testLegacyConversion - ordering\n";
    return false;
  }

  if (!spec.placement.has_value() || *spec.placement != Placement::SHARED_L2) {
    std::cerr << "FAIL: testLegacyConversion - placement\n";
    return false;
  }

  // throughput and shape should not be set from legacy conversion.
  if (spec.throughput.has_value()) {
    std::cerr << "FAIL: testLegacyConversion - throughput should be unset\n";
    return false;
  }
  if (spec.shape.has_value()) {
    std::cerr << "FAIL: testLegacyConversion - shape should be unset\n";
    return false;
  }

  std::cerr << "PASS: testLegacyConversion\n";
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

  run(testFullEdgeTranslation);
  run(testPartialEdgeTranslation);
  run(testUnconstrainedEdge);
  run(testPathTranslation);
  run(testPathEmptyLatency);
  run(testBatchTranslation);
  run(testLegacyConversion);

  std::cerr << "\nResults: " << passed << "/" << total << " tests passed\n";
  return (passed == total) ? 0 : 1;
}
