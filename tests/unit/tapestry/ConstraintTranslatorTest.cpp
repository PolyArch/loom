/// ContractConstraintTranslator unit tests.
///
/// Tests TDC constraint translation, symbolic expression evaluation,
/// shape parsing, and pruning mask computation.

#include "loom/SystemCompiler/ContractConstraintTranslator.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

using namespace loom;

//===----------------------------------------------------------------------===//
// Symbolic Expression Evaluator Tests
//===----------------------------------------------------------------------===//

static bool testEvalArithmetic() {
  ParameterMap params = {{"a", 10}, {"b", 20}, {"c", 4}};
  auto result = evaluateSymbolicExpr("(a + b) * c / 2", params);
  if (!result.ok() || result.value != 60) {
    std::cerr << "FAIL: testEvalArithmetic - expected 60, got " << result.value
              << " error='" << result.error << "'\n";
    return false;
  }
  std::cerr << "PASS: testEvalArithmetic\n";
  return true;
}

static bool testEvalConcrete() {
  ParameterMap params;
  auto result = evaluateSymbolicExpr("64", params);
  if (!result.ok() || result.value != 64) {
    std::cerr << "FAIL: testEvalConcrete - expected 64, got " << result.value
              << "\n";
    return false;
  }
  std::cerr << "PASS: testEvalConcrete\n";
  return true;
}

static bool testEvalSymbolicProduct() {
  ParameterMap params = {{"batch_size", 32}, {"hidden_dim", 512}};
  auto result =
      evaluateSymbolicExpr("batch_size * hidden_dim / 1000", params);
  // 32 * 512 = 16384, 16384 / 1000 = 16 (integer truncation)
  if (!result.ok() || result.value != 16) {
    std::cerr << "FAIL: testEvalSymbolicProduct - expected 16, got "
              << result.value << "\n";
    return false;
  }
  std::cerr << "PASS: testEvalSymbolicProduct\n";
  return true;
}

static bool testEvalUnknownVariable() {
  ParameterMap params = {{"a", 10}};
  auto result = evaluateSymbolicExpr("a + unknown_var", params);
  if (result.ok()) {
    std::cerr << "FAIL: testEvalUnknownVariable - expected error\n";
    return false;
  }
  if (result.error.find("unknown_var") == std::string::npos) {
    std::cerr << "FAIL: testEvalUnknownVariable - error should mention "
                 "'unknown_var', got: "
              << result.error << "\n";
    return false;
  }
  std::cerr << "PASS: testEvalUnknownVariable\n";
  return true;
}

static bool testEvalEmptyExpr() {
  ParameterMap params;
  auto result = evaluateSymbolicExpr("", params);
  if (result.ok()) {
    std::cerr << "FAIL: testEvalEmptyExpr - expected error\n";
    return false;
  }
  std::cerr << "PASS: testEvalEmptyExpr\n";
  return true;
}

static bool testEvalUnaryMinus() {
  ParameterMap params = {{"x", 5}};
  auto result = evaluateSymbolicExpr("-x + 10", params);
  if (!result.ok() || result.value != 5) {
    std::cerr << "FAIL: testEvalUnaryMinus - expected 5, got " << result.value
              << "\n";
    return false;
  }
  std::cerr << "PASS: testEvalUnaryMinus\n";
  return true;
}

static bool testEvalDivisionByZero() {
  ParameterMap params;
  auto result = evaluateSymbolicExpr("10 / 0", params);
  if (result.ok()) {
    std::cerr << "FAIL: testEvalDivisionByZero - expected error\n";
    return false;
  }
  std::cerr << "PASS: testEvalDivisionByZero\n";
  return true;
}

//===----------------------------------------------------------------------===//
// Shape Parsing Tests
//===----------------------------------------------------------------------===//

static bool testParseShapeConcrete() {
  auto dims = parseShapeDimensions("[128, 256]");
  if (dims.size() != 2 || dims[0] != "128" || dims[1] != "256") {
    std::cerr << "FAIL: testParseShapeConcrete - wrong dimensions\n";
    return false;
  }
  std::cerr << "PASS: testParseShapeConcrete\n";
  return true;
}

static bool testParseShapeSymbolic() {
  auto dims = parseShapeDimensions("[batch_size, hidden_dim]");
  if (dims.size() != 2 || dims[0] != "batch_size" || dims[1] != "hidden_dim") {
    std::cerr << "FAIL: testParseShapeSymbolic - wrong dimensions\n";
    return false;
  }
  std::cerr << "PASS: testParseShapeSymbolic\n";
  return true;
}

static bool testParseShapeEmpty() {
  auto dims = parseShapeDimensions("");
  if (!dims.empty()) {
    std::cerr << "FAIL: testParseShapeEmpty - expected empty\n";
    return false;
  }
  std::cerr << "PASS: testParseShapeEmpty\n";
  return true;
}

//===----------------------------------------------------------------------===//
// Constraint Translation Tests
//===----------------------------------------------------------------------===//

static bool testTranslateOrdering_FIFO() {
  TDCEdgeSpec spec;
  spec.producerKernel = "matmul";
  spec.consumerKernel = "softmax";
  spec.ordering = TDCOrdering::FIFO;

  ContractConstraintTranslator translator;
  ParameterMap params;
  auto cs = translator.translate({spec}, {}, params);

  if (cs.schedulingConstraints.size() != 1) {
    std::cerr << "FAIL: testTranslateOrdering_FIFO - expected 1 scheduling, "
                 "got "
              << cs.schedulingConstraints.size() << "\n";
    return false;
  }
  if (cs.schedulingConstraints[0].producer != "matmul" ||
      cs.schedulingConstraints[0].consumer != "softmax") {
    std::cerr << "FAIL: testTranslateOrdering_FIFO - wrong kernel names\n";
    return false;
  }
  if (!cs.rateConstraints.empty() || !cs.memoryConstraints.empty() ||
      !cs.tilingConstraints.empty() || !cs.pathLatencyConstraints.empty()) {
    std::cerr << "FAIL: testTranslateOrdering_FIFO - unexpected constraints\n";
    return false;
  }
  std::cerr << "PASS: testTranslateOrdering_FIFO\n";
  return true;
}

static bool testTranslateOrdering_UNORDERED() {
  TDCEdgeSpec spec;
  spec.producerKernel = "matmul";
  spec.consumerKernel = "softmax";
  spec.ordering = TDCOrdering::UNORDERED;

  ContractConstraintTranslator translator;
  ParameterMap params;
  auto cs = translator.translate({spec}, {}, params);

  if (!cs.empty()) {
    std::cerr << "FAIL: testTranslateOrdering_UNORDERED - expected empty "
                 "constraints, got "
              << cs.totalConstraintCount() << "\n";
    return false;
  }
  std::cerr << "PASS: testTranslateOrdering_UNORDERED\n";
  return true;
}

static bool testTranslateThroughput_Symbolic() {
  TDCEdgeSpec spec;
  spec.producerKernel = "matmul";
  spec.consumerKernel = "softmax";
  spec.throughput = "batch_size * hidden_dim / 1000";

  ParameterMap params = {{"batch_size", 32}, {"hidden_dim", 512}};
  ContractConstraintTranslator translator;
  auto cs = translator.translate({spec}, {}, params);

  if (cs.rateConstraints.size() != 1) {
    std::cerr << "FAIL: testTranslateThroughput_Symbolic - expected 1 rate, "
                 "got "
              << cs.rateConstraints.size() << "\n";
    return false;
  }
  if (cs.rateConstraints[0].minRate != 16) {
    std::cerr << "FAIL: testTranslateThroughput_Symbolic - expected rate=16, "
                 "got "
              << cs.rateConstraints[0].minRate << "\n";
    return false;
  }
  if (cs.rateConstraints[0].edgeProducer != "matmul" ||
      cs.rateConstraints[0].edgeConsumer != "softmax") {
    std::cerr << "FAIL: testTranslateThroughput_Symbolic - wrong edge names\n";
    return false;
  }
  std::cerr << "PASS: testTranslateThroughput_Symbolic\n";
  return true;
}

static bool testTranslateThroughput_Concrete() {
  TDCEdgeSpec spec;
  spec.producerKernel = "matmul";
  spec.consumerKernel = "softmax";
  spec.throughput = "64";

  ContractConstraintTranslator translator;
  ParameterMap params;
  auto cs = translator.translate({spec}, {}, params);

  if (cs.rateConstraints.size() != 1 || cs.rateConstraints[0].minRate != 64) {
    std::cerr << "FAIL: testTranslateThroughput_Concrete\n";
    return false;
  }
  std::cerr << "PASS: testTranslateThroughput_Concrete\n";
  return true;
}

static bool testTranslatePlacement_LocalSPM() {
  TDCEdgeSpec spec;
  spec.producerKernel = "matmul";
  spec.consumerKernel = "softmax";
  spec.placement = TDCPlacement::LOCAL_SPM;

  ContractConstraintTranslator translator;
  ParameterMap params;
  auto cs = translator.translate({spec}, {}, params);

  if (cs.memoryConstraints.size() != 1) {
    std::cerr << "FAIL: testTranslatePlacement_LocalSPM - expected 1 memory, "
                 "got "
              << cs.memoryConstraints.size() << "\n";
    return false;
  }
  if (cs.memoryConstraints[0].level != MemoryLevel::LOCAL_SPM) {
    std::cerr
        << "FAIL: testTranslatePlacement_LocalSPM - wrong memory level\n";
    return false;
  }
  std::cerr << "PASS: testTranslatePlacement_LocalSPM\n";
  return true;
}

static bool testTranslatePlacement_AUTO() {
  TDCEdgeSpec spec;
  spec.producerKernel = "matmul";
  spec.consumerKernel = "softmax";
  spec.placement = TDCPlacement::AUTO;

  ContractConstraintTranslator translator;
  ParameterMap params;
  auto cs = translator.translate({spec}, {}, params);

  if (!cs.memoryConstraints.empty()) {
    std::cerr << "FAIL: testTranslatePlacement_AUTO - expected no memory "
                 "constraints\n";
    return false;
  }
  std::cerr << "PASS: testTranslatePlacement_AUTO\n";
  return true;
}

static bool testTranslateShape_Concrete() {
  TDCEdgeSpec spec;
  spec.producerKernel = "matmul";
  spec.consumerKernel = "softmax";
  spec.shape = "[128, 256]";

  ContractConstraintTranslator translator;
  ParameterMap params;
  auto cs = translator.translate({spec}, {}, params);

  if (cs.tilingConstraints.size() != 1) {
    std::cerr << "FAIL: testTranslateShape_Concrete - expected 1 tiling, got "
              << cs.tilingConstraints.size() << "\n";
    return false;
  }
  auto &tc = cs.tilingConstraints[0];
  if (tc.dimensions.size() != 2 || tc.dimensions[0] != 128 ||
      tc.dimensions[1] != 256) {
    std::cerr << "FAIL: testTranslateShape_Concrete - wrong dimensions\n";
    return false;
  }
  std::cerr << "PASS: testTranslateShape_Concrete\n";
  return true;
}

static bool testTranslateShape_Symbolic() {
  TDCEdgeSpec spec;
  spec.producerKernel = "matmul";
  spec.consumerKernel = "softmax";
  spec.shape = "[batch_size, hidden_dim]";

  ParameterMap params = {{"batch_size", 32}, {"hidden_dim", 512}};
  ContractConstraintTranslator translator;
  auto cs = translator.translate({spec}, {}, params);

  if (cs.tilingConstraints.size() != 1) {
    std::cerr << "FAIL: testTranslateShape_Symbolic - expected 1 tiling\n";
    return false;
  }
  auto &tc = cs.tilingConstraints[0];
  if (tc.dimensions.size() != 2 || tc.dimensions[0] != 32 ||
      tc.dimensions[1] != 512) {
    std::cerr << "FAIL: testTranslateShape_Symbolic - wrong dimensions, got ["
              << (tc.dimensions.empty() ? -1 : tc.dimensions[0]) << ", "
              << (tc.dimensions.size() < 2 ? -1 : tc.dimensions[1]) << "]\n";
    return false;
  }
  std::cerr << "PASS: testTranslateShape_Symbolic\n";
  return true;
}

static bool testTranslatePathLatency() {
  TDCPathSpec pathSpec;
  pathSpec.startProducer = "matmul";
  pathSpec.startConsumer = "softmax";
  pathSpec.endProducer = "softmax";
  pathSpec.endConsumer = "relu";
  pathSpec.latency = "4 * tile_m";

  ParameterMap params = {{"tile_m", 64}};
  ContractConstraintTranslator translator;
  auto cs = translator.translate({}, {pathSpec}, params);

  if (cs.pathLatencyConstraints.size() != 1) {
    std::cerr << "FAIL: testTranslatePathLatency - expected 1 path, got "
              << cs.pathLatencyConstraints.size() << "\n";
    return false;
  }
  auto &plc = cs.pathLatencyConstraints[0];
  if (plc.maxCycles != 256) {
    std::cerr << "FAIL: testTranslatePathLatency - expected 256 cycles, got "
              << plc.maxCycles << "\n";
    return false;
  }
  if (plc.startProducer != "matmul" || plc.startConsumer != "softmax" ||
      plc.endProducer != "softmax" || plc.endConsumer != "relu") {
    std::cerr << "FAIL: testTranslatePathLatency - wrong edge identifiers\n";
    return false;
  }
  std::cerr << "PASS: testTranslatePathLatency\n";
  return true;
}

static bool testTranslateMultipleDimensions() {
  TDCEdgeSpec spec;
  spec.producerKernel = "matmul";
  spec.consumerKernel = "softmax";
  spec.ordering = TDCOrdering::FIFO;
  spec.throughput = "64";
  spec.placement = TDCPlacement::SHARED_L2;
  spec.shape = "[128, 128]";

  ContractConstraintTranslator translator;
  ParameterMap params;
  auto cs = translator.translate({spec}, {}, params);

  if (cs.schedulingConstraints.size() != 1) {
    std::cerr << "FAIL: testTranslateMultipleDimensions - scheduling\n";
    return false;
  }
  if (cs.rateConstraints.size() != 1) {
    std::cerr << "FAIL: testTranslateMultipleDimensions - rate\n";
    return false;
  }
  if (cs.memoryConstraints.size() != 1) {
    std::cerr << "FAIL: testTranslateMultipleDimensions - memory\n";
    return false;
  }
  if (cs.tilingConstraints.size() != 1) {
    std::cerr << "FAIL: testTranslateMultipleDimensions - tiling\n";
    return false;
  }
  if (!cs.pathLatencyConstraints.empty()) {
    std::cerr << "FAIL: testTranslateMultipleDimensions - path should be "
                 "empty\n";
    return false;
  }
  std::cerr << "PASS: testTranslateMultipleDimensions\n";
  return true;
}

static bool testTranslateEmptySpec() {
  TDCEdgeSpec spec;
  spec.producerKernel = "matmul";
  spec.consumerKernel = "softmax";
  // All optional dimensions left as nullopt.

  ContractConstraintTranslator translator;
  ParameterMap params;
  auto cs = translator.translate({spec}, {}, params);

  if (!cs.empty()) {
    std::cerr << "FAIL: testTranslateEmptySpec - expected empty, got "
              << cs.totalConstraintCount() << " constraints\n";
    return false;
  }
  std::cerr << "PASS: testTranslateEmptySpec\n";
  return true;
}

//===----------------------------------------------------------------------===//
// Pruning Mask Tests
//===----------------------------------------------------------------------===//

static bool testPruningMask_PartialSpec() {
  TDCEdgeSpec spec;
  spec.producerKernel = "matmul";
  spec.consumerKernel = "softmax";
  spec.ordering = TDCOrdering::FIFO;
  spec.shape = "[64, 64]";
  // throughput and placement are nullopt.

  ContractConstraintTranslator translator;
  auto masks = translator.computePruningMasks({spec});

  if (masks.size() != 1) {
    std::cerr << "FAIL: testPruningMask_PartialSpec - expected 1 mask, got "
              << masks.size() << "\n";
    return false;
  }

  auto &pm = masks[0];
  if (!pm.isOrderingLocked()) {
    std::cerr << "FAIL: testPruningMask_PartialSpec - ordering should be "
                 "locked\n";
    return false;
  }
  if (pm.isThroughputLocked()) {
    std::cerr << "FAIL: testPruningMask_PartialSpec - throughput should NOT "
                 "be locked\n";
    return false;
  }
  if (pm.isPlacementLocked()) {
    std::cerr << "FAIL: testPruningMask_PartialSpec - placement should NOT "
                 "be locked\n";
    return false;
  }
  if (!pm.isShapeLocked()) {
    std::cerr << "FAIL: testPruningMask_PartialSpec - shape should be locked\n";
    return false;
  }

  uint8_t expectedMask =
      PruningMask::ORDERING_LOCKED | PruningMask::SHAPE_LOCKED;
  if (pm.mask != expectedMask) {
    std::cerr << "FAIL: testPruningMask_PartialSpec - expected mask=0x"
              << std::hex << (int)expectedMask << ", got 0x" << (int)pm.mask
              << std::dec << "\n";
    return false;
  }

  std::cerr << "PASS: testPruningMask_PartialSpec\n";
  return true;
}

static bool testPruningMask_AllLocked() {
  TDCEdgeSpec spec;
  spec.producerKernel = "a";
  spec.consumerKernel = "b";
  spec.ordering = TDCOrdering::FIFO;
  spec.throughput = "100";
  spec.placement = TDCPlacement::SHARED_L2;
  spec.shape = "[32]";

  ContractConstraintTranslator translator;
  auto masks = translator.computePruningMasks({spec});

  if (masks.size() != 1 || masks[0].mask != 0x0F) {
    std::cerr << "FAIL: testPruningMask_AllLocked - expected mask=0x0F, got 0x"
              << std::hex << (int)masks[0].mask << std::dec << "\n";
    return false;
  }
  std::cerr << "PASS: testPruningMask_AllLocked\n";
  return true;
}

static bool testPruningMask_NothingLocked() {
  TDCEdgeSpec spec;
  spec.producerKernel = "a";
  spec.consumerKernel = "b";
  // All dimensions nullopt.

  ContractConstraintTranslator translator;
  auto masks = translator.computePruningMasks({spec});

  if (masks.size() != 1 || masks[0].mask != 0) {
    std::cerr << "FAIL: testPruningMask_NothingLocked\n";
    return false;
  }
  std::cerr << "PASS: testPruningMask_NothingLocked\n";
  return true;
}

static bool testPruningMask_UNORDERED_NotLocked() {
  TDCEdgeSpec spec;
  spec.producerKernel = "a";
  spec.consumerKernel = "b";
  spec.ordering = TDCOrdering::UNORDERED;

  ContractConstraintTranslator translator;
  auto masks = translator.computePruningMasks({spec});

  // UNORDERED should NOT lock the ordering bit.
  if (masks.size() != 1 || masks[0].isOrderingLocked()) {
    std::cerr << "FAIL: testPruningMask_UNORDERED_NotLocked\n";
    return false;
  }
  std::cerr << "PASS: testPruningMask_UNORDERED_NotLocked\n";
  return true;
}

static bool testPruningMask_AUTO_NotLocked() {
  TDCEdgeSpec spec;
  spec.producerKernel = "a";
  spec.consumerKernel = "b";
  spec.placement = TDCPlacement::AUTO;

  ContractConstraintTranslator translator;
  auto masks = translator.computePruningMasks({spec});

  // AUTO placement should NOT lock the placement bit.
  if (masks.size() != 1 || masks[0].isPlacementLocked()) {
    std::cerr << "FAIL: testPruningMask_AUTO_NotLocked\n";
    return false;
  }
  std::cerr << "PASS: testPruningMask_AUTO_NotLocked\n";
  return true;
}

//===----------------------------------------------------------------------===//
// Diagnostic Tests
//===----------------------------------------------------------------------===//

static bool testDiagnosticOnBadExpression() {
  TDCEdgeSpec spec;
  spec.producerKernel = "a";
  spec.consumerKernel = "b";
  spec.throughput = "x + unknown_var";

  ParameterMap params = {{"x", 10}};
  ContractConstraintTranslator translator;
  auto cs = translator.translate({spec}, {}, params);

  // The rate constraint should NOT have been emitted.
  if (!cs.rateConstraints.empty()) {
    std::cerr << "FAIL: testDiagnosticOnBadExpression - rate should be empty\n";
    return false;
  }

  // There should be an error diagnostic.
  auto &diags = translator.getDiagnostics();
  if (diags.empty()) {
    std::cerr << "FAIL: testDiagnosticOnBadExpression - expected diagnostic\n";
    return false;
  }
  if (diags[0].severity != TranslatorDiagnostic::ERROR) {
    std::cerr
        << "FAIL: testDiagnosticOnBadExpression - expected ERROR severity\n";
    return false;
  }

  std::cerr << "PASS: testDiagnosticOnBadExpression\n";
  return true;
}

//===----------------------------------------------------------------------===//
// main
//===----------------------------------------------------------------------===//

int main() {
  int passed = 0;
  int total = 0;

  auto run = [&](bool (*test)()) {
    total++;
    if (test())
      passed++;
  };

  // Expression evaluator tests.
  run(testEvalArithmetic);
  run(testEvalConcrete);
  run(testEvalSymbolicProduct);
  run(testEvalUnknownVariable);
  run(testEvalEmptyExpr);
  run(testEvalUnaryMinus);
  run(testEvalDivisionByZero);

  // Shape parsing tests.
  run(testParseShapeConcrete);
  run(testParseShapeSymbolic);
  run(testParseShapeEmpty);

  // Constraint translation tests.
  run(testTranslateOrdering_FIFO);
  run(testTranslateOrdering_UNORDERED);
  run(testTranslateThroughput_Symbolic);
  run(testTranslateThroughput_Concrete);
  run(testTranslatePlacement_LocalSPM);
  run(testTranslatePlacement_AUTO);
  run(testTranslateShape_Concrete);
  run(testTranslateShape_Symbolic);
  run(testTranslatePathLatency);
  run(testTranslateMultipleDimensions);
  run(testTranslateEmptySpec);

  // Pruning mask tests.
  run(testPruningMask_PartialSpec);
  run(testPruningMask_AllLocked);
  run(testPruningMask_NothingLocked);
  run(testPruningMask_UNORDERED_NotLocked);
  run(testPruningMask_AUTO_NotLocked);

  // Diagnostic tests.
  run(testDiagnosticOnBadExpression);

  std::cerr << "\nResults: " << passed << "/" << total << " tests passed\n";
  return (passed == total) ? 0 : 1;
}
