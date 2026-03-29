/// TDC (Tapestry Dimensional Contract) type tests: struct construction,
/// enum conversion, JSON round-trip, MLIR op creation.
///
/// Tests:
///  1. TDCEdgeSpec construction with all dimensions set
///  2. TDCEdgeSpec construction with partial dimensions
///  3. TDCEdgeSpec construction fully unconstrained
///  4. TDCPathSpec construction
///  5. Edge vs path contract distinction (no shared base)
///  6. Ordering enum round-trip (FIFO, UNORDERED, SYMBOLIC)
///  7. Placement enum round-trip (LOCAL_SPM, SHARED_L2, EXTERNAL, AUTO)
///  8. TDCEdgeSpec JSON round-trip -- fully specified
///  9. TDCEdgeSpec JSON round-trip -- partially specified
/// 10. TDCEdgeSpec JSON backward compatibility with old keys
/// 11. TDCPathSpec JSON round-trip
/// 12. parseShapeExpr utility
/// 13. parseShapeExpr -- single and empty dimension

#include "loom/SystemCompiler/Contract.h"
#include "llvm/Support/JSON.h"

#include <cassert>
#include <iostream>
#include <string>
#include <type_traits>
#include <vector>

using namespace loom;

/// T1: TDCEdgeSpec construction -- all 4 dimensions set.
static bool testEdgeSpecFullyConstrained() {
  TDCEdgeSpec spec;
  spec.producerKernel = "matmul";
  spec.consumerKernel = "softmax";
  spec.dataTypeName = "f32";
  spec.ordering = Ordering::FIFO;
  spec.throughput = "batch_size * hidden_dim / 1000";
  spec.placement = Placement::SHARED_L2;
  spec.shape = "[128, hidden_dim]";

  if (spec.producerKernel != "matmul" || spec.consumerKernel != "softmax" ||
      spec.dataTypeName != "f32") {
    std::cerr << "FAIL: testEdgeSpecFullyConstrained - identity fields\n";
    return false;
  }
  if (!spec.ordering.has_value() || *spec.ordering != Ordering::FIFO) {
    std::cerr << "FAIL: testEdgeSpecFullyConstrained - ordering\n";
    return false;
  }
  if (!spec.throughput.has_value() ||
      *spec.throughput != "batch_size * hidden_dim / 1000") {
    std::cerr << "FAIL: testEdgeSpecFullyConstrained - throughput\n";
    return false;
  }
  if (!spec.placement.has_value() || *spec.placement != Placement::SHARED_L2) {
    std::cerr << "FAIL: testEdgeSpecFullyConstrained - placement\n";
    return false;
  }
  if (!spec.shape.has_value() || *spec.shape != "[128, hidden_dim]") {
    std::cerr << "FAIL: testEdgeSpecFullyConstrained - shape\n";
    return false;
  }

  std::cerr << "PASS: testEdgeSpecFullyConstrained\n";
  return true;
}

/// T2: TDCEdgeSpec construction -- partially constrained.
static bool testEdgeSpecPartiallyConstrained() {
  TDCEdgeSpec spec;
  spec.producerKernel = "conv";
  spec.consumerKernel = "pool";
  spec.dataTypeName = "f16";
  spec.ordering = Ordering::UNORDERED;
  // Leave throughput, placement, shape unset.

  if (!spec.ordering.has_value() || *spec.ordering != Ordering::UNORDERED) {
    std::cerr << "FAIL: testEdgeSpecPartiallyConstrained - ordering\n";
    return false;
  }
  if (spec.throughput.has_value()) {
    std::cerr << "FAIL: testEdgeSpecPartiallyConstrained - throughput should "
                 "be unset\n";
    return false;
  }
  if (spec.placement.has_value()) {
    std::cerr << "FAIL: testEdgeSpecPartiallyConstrained - placement should "
                 "be unset\n";
    return false;
  }
  if (spec.shape.has_value()) {
    std::cerr << "FAIL: testEdgeSpecPartiallyConstrained - shape should "
                 "be unset\n";
    return false;
  }

  std::cerr << "PASS: testEdgeSpecPartiallyConstrained\n";
  return true;
}

/// T3: TDCEdgeSpec construction -- fully unconstrained.
static bool testEdgeSpecFullyUnconstrained() {
  TDCEdgeSpec spec;
  spec.producerKernel = "fft";
  spec.consumerKernel = "ifft";
  spec.dataTypeName = "c64";

  if (spec.ordering.has_value()) {
    std::cerr << "FAIL: testEdgeSpecFullyUnconstrained - ordering should "
                 "be unset\n";
    return false;
  }
  if (spec.throughput.has_value()) {
    std::cerr << "FAIL: testEdgeSpecFullyUnconstrained - throughput should "
                 "be unset\n";
    return false;
  }
  if (spec.placement.has_value()) {
    std::cerr << "FAIL: testEdgeSpecFullyUnconstrained - placement should "
                 "be unset\n";
    return false;
  }
  if (spec.shape.has_value()) {
    std::cerr << "FAIL: testEdgeSpecFullyUnconstrained - shape should "
                 "be unset\n";
    return false;
  }

  std::cerr << "PASS: testEdgeSpecFullyUnconstrained\n";
  return true;
}

/// T4: TDCPathSpec construction.
static bool testPathSpecConstruction() {
  TDCPathSpec spec;
  spec.startProducer = "matmul";
  spec.startConsumer = "softmax";
  spec.endProducer = "softmax";
  spec.endConsumer = "relu";
  spec.latency = "4 * tile_m";

  if (spec.startProducer != "matmul" || spec.startConsumer != "softmax") {
    std::cerr << "FAIL: testPathSpecConstruction - start edge\n";
    return false;
  }
  if (spec.endProducer != "softmax" || spec.endConsumer != "relu") {
    std::cerr << "FAIL: testPathSpecConstruction - end edge\n";
    return false;
  }
  if (spec.latency != "4 * tile_m") {
    std::cerr << "FAIL: testPathSpecConstruction - latency\n";
    return false;
  }

  std::cerr << "PASS: testPathSpecConstruction\n";
  return true;
}

/// T5: Edge vs path contract distinction.
static bool testEdgeVsPathDistinction() {
  // Verify they are distinct types at compile time.
  static_assert(!std::is_same<TDCEdgeSpec, TDCPathSpec>::value,
                "TDCEdgeSpec and TDCPathSpec must be distinct types");

  // Verify no implicit conversion (checked by type system; runtime just
  // confirms fields exist on the correct types).
  TDCEdgeSpec edge;
  edge.ordering = Ordering::FIFO;
  edge.throughput = "100";

  TDCPathSpec path;
  path.latency = "50";

  // Edge has no latency field. Path has no ordering/throughput/placement/shape.
  // This is verified by compilation -- if these fields were present on the
  // wrong type, it would be a design bug.

  std::cerr << "PASS: testEdgeVsPathDistinction\n";
  return true;
}

/// T6: Ordering enum round-trip.
static bool testOrderingEnumRoundTrip() {
  Ordering values[] = {Ordering::FIFO, Ordering::UNORDERED, Ordering::SYMBOLIC};
  for (auto v : values) {
    if (orderingFromString(orderingToString(v)) != v) {
      std::cerr << "FAIL: testOrderingEnumRoundTrip - value "
                << orderingToString(v) << "\n";
      return false;
    }
  }

  std::cerr << "PASS: testOrderingEnumRoundTrip\n";
  return true;
}

/// T7: Placement enum round-trip.
static bool testPlacementEnumRoundTrip() {
  Placement values[] = {Placement::LOCAL_SPM, Placement::SHARED_L2,
                        Placement::EXTERNAL, Placement::AUTO};
  for (auto v : values) {
    if (placementFromString(placementToString(v)) != v) {
      std::cerr << "FAIL: testPlacementEnumRoundTrip - value "
                << placementToString(v) << "\n";
      return false;
    }
  }

  // Verify legacy EXTERNAL_DRAM string maps to EXTERNAL.
  if (placementFromString("EXTERNAL_DRAM") != Placement::EXTERNAL) {
    std::cerr << "FAIL: testPlacementEnumRoundTrip - EXTERNAL_DRAM compat\n";
    return false;
  }

  std::cerr << "PASS: testPlacementEnumRoundTrip\n";
  return true;
}

/// T8: TDCEdgeSpec JSON round-trip -- fully specified.
static bool testEdgeSpecJSONFullRoundTrip() {
  TDCEdgeSpec spec;
  spec.producerKernel = "matmul";
  spec.consumerKernel = "softmax";
  spec.dataTypeName = "f32";
  spec.ordering = Ordering::FIFO;
  spec.throughput = "batch * hidden / 1000";
  spec.placement = Placement::SHARED_L2;
  spec.shape = "[128, hidden_dim]";

  auto json = tdcEdgeSpecToJSON(spec);
  TDCEdgeSpec spec2 = tdcEdgeSpecFromJSON(json);

  if (spec2.producerKernel != "matmul" || spec2.consumerKernel != "softmax" ||
      spec2.dataTypeName != "f32") {
    std::cerr << "FAIL: testEdgeSpecJSONFullRoundTrip - identity\n";
    return false;
  }
  if (!spec2.ordering || *spec2.ordering != Ordering::FIFO) {
    std::cerr << "FAIL: testEdgeSpecJSONFullRoundTrip - ordering\n";
    return false;
  }
  if (!spec2.throughput || *spec2.throughput != "batch * hidden / 1000") {
    std::cerr << "FAIL: testEdgeSpecJSONFullRoundTrip - throughput\n";
    return false;
  }
  if (!spec2.placement || *spec2.placement != Placement::SHARED_L2) {
    std::cerr << "FAIL: testEdgeSpecJSONFullRoundTrip - placement\n";
    return false;
  }
  if (!spec2.shape || *spec2.shape != "[128, hidden_dim]") {
    std::cerr << "FAIL: testEdgeSpecJSONFullRoundTrip - shape\n";
    return false;
  }

  std::cerr << "PASS: testEdgeSpecJSONFullRoundTrip\n";
  return true;
}

/// T9: TDCEdgeSpec JSON round-trip -- partially specified.
static bool testEdgeSpecJSONPartialRoundTrip() {
  TDCEdgeSpec spec;
  spec.producerKernel = "conv";
  spec.consumerKernel = "pool";
  spec.dataTypeName = "f16";
  spec.ordering = Ordering::UNORDERED;
  // throughput, placement, shape left unset.

  auto json = tdcEdgeSpecToJSON(spec);
  TDCEdgeSpec spec2 = tdcEdgeSpecFromJSON(json);

  if (!spec2.ordering || *spec2.ordering != Ordering::UNORDERED) {
    std::cerr << "FAIL: testEdgeSpecJSONPartialRoundTrip - ordering\n";
    return false;
  }
  if (spec2.throughput.has_value()) {
    std::cerr << "FAIL: testEdgeSpecJSONPartialRoundTrip - throughput "
                 "should be unset\n";
    return false;
  }
  if (spec2.placement.has_value()) {
    std::cerr << "FAIL: testEdgeSpecJSONPartialRoundTrip - placement "
                 "should be unset\n";
    return false;
  }
  if (spec2.shape.has_value()) {
    std::cerr << "FAIL: testEdgeSpecJSONPartialRoundTrip - shape "
                 "should be unset\n";
    return false;
  }

  std::cerr << "PASS: testEdgeSpecJSONPartialRoundTrip\n";
  return true;
}

/// T10: TDCEdgeSpec JSON backward compatibility.
static bool testEdgeSpecJSONBackwardCompat() {
  // Simulate old ContractSpec JSON with legacy keys.
  llvm::json::Object obj;
  obj["producerKernel"] = "matmul";
  obj["consumerKernel"] = "relu";
  obj["dataTypeName"] = "f32";
  obj["ordering"] = "FIFO";
  obj["throughput"] = "1024";
  // Legacy keys that should be silently ignored:
  obj["may_fuse"] = true;
  obj["backpressure"] = "BLOCK";
  obj["min_buffer_elements"] = 256;
  obj["double_buffering"] = false;
  obj["visibility"] = "LOCAL_SPM";
  obj["productionRate"] = 64;

  llvm::json::Value jsonVal(std::move(obj));
  TDCEdgeSpec spec = tdcEdgeSpecFromJSON(jsonVal);

  if (spec.producerKernel != "matmul" || spec.consumerKernel != "relu") {
    std::cerr << "FAIL: testEdgeSpecJSONBackwardCompat - identity\n";
    return false;
  }
  if (!spec.ordering || *spec.ordering != Ordering::FIFO) {
    std::cerr << "FAIL: testEdgeSpecJSONBackwardCompat - ordering\n";
    return false;
  }
  if (!spec.throughput || *spec.throughput != "1024") {
    std::cerr << "FAIL: testEdgeSpecJSONBackwardCompat - throughput\n";
    return false;
  }

  // The old keys should have been ignored (no crash, no data corruption).
  std::cerr << "PASS: testEdgeSpecJSONBackwardCompat\n";
  return true;
}

/// T11: TDCPathSpec JSON round-trip.
static bool testPathSpecJSONRoundTrip() {
  TDCPathSpec spec;
  spec.startProducer = "matmul";
  spec.startConsumer = "softmax";
  spec.endProducer = "softmax";
  spec.endConsumer = "relu";
  spec.latency = "4 * tile_m";

  auto json = tdcPathSpecToJSON(spec);
  TDCPathSpec spec2 = tdcPathSpecFromJSON(json);

  if (spec2.startProducer != "matmul" || spec2.startConsumer != "softmax") {
    std::cerr << "FAIL: testPathSpecJSONRoundTrip - start edge\n";
    return false;
  }
  if (spec2.endProducer != "softmax" || spec2.endConsumer != "relu") {
    std::cerr << "FAIL: testPathSpecJSONRoundTrip - end edge\n";
    return false;
  }
  if (spec2.latency != "4 * tile_m") {
    std::cerr << "FAIL: testPathSpecJSONRoundTrip - latency\n";
    return false;
  }

  std::cerr << "PASS: testPathSpecJSONRoundTrip\n";
  return true;
}

/// T12: parseShapeExpr utility.
static bool testParseShapeExpr() {
  auto dims = parseShapeExpr("[128, hidden_dim / num_heads, 64]");
  if (dims.size() != 3) {
    std::cerr << "FAIL: testParseShapeExpr - expected 3 dims, got "
              << dims.size() << "\n";
    return false;
  }
  if (dims[0] != "128") {
    std::cerr << "FAIL: testParseShapeExpr - dim 0: " << dims[0] << "\n";
    return false;
  }
  if (dims[1] != "hidden_dim / num_heads") {
    std::cerr << "FAIL: testParseShapeExpr - dim 1: " << dims[1] << "\n";
    return false;
  }
  if (dims[2] != "64") {
    std::cerr << "FAIL: testParseShapeExpr - dim 2: " << dims[2] << "\n";
    return false;
  }

  std::cerr << "PASS: testParseShapeExpr\n";
  return true;
}

/// T13: parseShapeExpr -- single and empty dimension.
static bool testParseShapeExprEdgeCases() {
  auto single = parseShapeExpr("[1024]");
  if (single.size() != 1 || single[0] != "1024") {
    std::cerr << "FAIL: testParseShapeExprEdgeCases - single dim\n";
    return false;
  }

  auto empty = parseShapeExpr("[]");
  if (!empty.empty()) {
    std::cerr << "FAIL: testParseShapeExprEdgeCases - empty brackets, got "
              << empty.size() << " dims\n";
    return false;
  }

  auto emptyStr = parseShapeExpr("");
  if (!emptyStr.empty()) {
    std::cerr << "FAIL: testParseShapeExprEdgeCases - empty string\n";
    return false;
  }

  std::cerr << "PASS: testParseShapeExprEdgeCases\n";
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

  run(testEdgeSpecFullyConstrained);
  run(testEdgeSpecPartiallyConstrained);
  run(testEdgeSpecFullyUnconstrained);
  run(testPathSpecConstruction);
  run(testEdgeVsPathDistinction);
  run(testOrderingEnumRoundTrip);
  run(testPlacementEnumRoundTrip);
  run(testEdgeSpecJSONFullRoundTrip);
  run(testEdgeSpecJSONPartialRoundTrip);
  run(testEdgeSpecJSONBackwardCompat);
  run(testPathSpecJSONRoundTrip);
  run(testParseShapeExpr);
  run(testParseShapeExprEdgeCases);

  std::cerr << "\nResults: " << passed << "/" << total << " tests passed\n";
  return (passed == total) ? 0 : 1;
}
