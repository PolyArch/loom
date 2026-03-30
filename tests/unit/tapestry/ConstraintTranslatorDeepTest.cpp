/// Deepened ContractConstraintTranslator tests: typed constraint generation,
/// symbolic expression evaluator, search space pruning masks.
///
/// Tests:
///  1. FIFO ordering emits SchedulingConstraint
///  2. UNORDERED ordering emits nothing
///  3. Symbolic throughput emits RateConstraint
///  4. Concrete throughput emits RateConstraint
///  5. LOCAL_SPM placement emits MemoryConstraint
///  6. AUTO placement emits no MemoryConstraint
///  7. Concrete shape emits TilingConstraint
///  8. Symbolic shape emits TilingConstraint
///  9. Path latency emits PathLatencyConstraint
/// 10. Multiple dimensions on one edge generate independent constraints
/// 11. Pruning mask reflects specified dimensions
/// 12. Empty edge spec generates zero constraints
/// 13. Symbolic evaluator handles arithmetic
/// 14. Symbolic evaluator rejects unknown variable
/// 15. Symbolic evaluator handles unary minus
/// 16. Symbolic evaluator rejects division by zero

#include "loom/SystemCompiler/ContractConstraintTranslator.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <string>

using namespace loom;

/// T1: FIFO ordering -> SchedulingConstraint
static bool testOrderingFIFO() {
  TDCEdgeSpec spec;
  spec.producerKernel = "matmul";
  spec.consumerKernel = "softmax";
  spec.dataTypeName = "f32";
  spec.ordering = Ordering::FIFO;

  ContractConstraintTranslator translator;
  auto cs = translator.translate({spec}, {});

  if (cs.scheduling.size() != 1) {
    std::cerr << "FAIL: testOrderingFIFO - expected 1 scheduling, got "
              << cs.scheduling.size() << "\n";
    return false;
  }
  if (cs.scheduling[0].producer != "matmul" ||
      cs.scheduling[0].consumer != "softmax") {
    std::cerr << "FAIL: testOrderingFIFO - wrong producer/consumer\n";
    return false;
  }
  if (!cs.rate.empty() || !cs.memory.empty() ||
      !cs.tiling.empty() || !cs.pathLatency.empty()) {
    std::cerr << "FAIL: testOrderingFIFO - unexpected extra constraints\n";
    return false;
  }

  std::cerr << "PASS: testOrderingFIFO\n";
  return true;
}

/// T2: UNORDERED ordering -> no constraints
static bool testOrderingUnordered() {
  TDCEdgeSpec spec;
  spec.producerKernel = "a";
  spec.consumerKernel = "b";
  spec.dataTypeName = "f32";
  spec.ordering = Ordering::UNORDERED;

  ContractConstraintTranslator translator;
  auto cs = translator.translate({spec}, {});

  if (!cs.empty()) {
    std::cerr << "FAIL: testOrderingUnordered - expected empty ConstraintSet\n";
    return false;
  }

  std::cerr << "PASS: testOrderingUnordered\n";
  return true;
}

/// T3: Symbolic throughput -> RateConstraint
static bool testThroughputSymbolic() {
  TDCEdgeSpec spec;
  spec.producerKernel = "matmul";
  spec.consumerKernel = "softmax";
  spec.dataTypeName = "f32";
  spec.throughput = "batch_size * hidden_dim / 1000";

  std::map<std::string, int64_t> params = {
      {"batch_size", 32}, {"hidden_dim", 512}};
  ContractConstraintTranslator translator(params);
  auto cs = translator.translate({spec}, {});

  if (cs.rate.size() != 1) {
    std::cerr << "FAIL: testThroughputSymbolic - expected 1 rate, got "
              << cs.rate.size() << "\n";
    return false;
  }
  // 32 * 512 / 1000 = 16384 / 1000 = 16 (integer division)
  if (cs.rate[0].minRate != 16) {
    std::cerr << "FAIL: testThroughputSymbolic - expected minRate=16, got "
              << cs.rate[0].minRate << "\n";
    return false;
  }
  if (cs.rate[0].edgeProducer != "matmul" ||
      cs.rate[0].edgeConsumer != "softmax") {
    std::cerr << "FAIL: testThroughputSymbolic - wrong edge identity\n";
    return false;
  }

  std::cerr << "PASS: testThroughputSymbolic\n";
  return true;
}

/// T4: Concrete throughput -> RateConstraint
static bool testThroughputConcrete() {
  TDCEdgeSpec spec;
  spec.producerKernel = "a";
  spec.consumerKernel = "b";
  spec.dataTypeName = "f32";
  spec.throughput = "64";

  ContractConstraintTranslator translator;
  auto cs = translator.translate({spec}, {});

  if (cs.rate.size() != 1) {
    std::cerr << "FAIL: testThroughputConcrete - expected 1 rate\n";
    return false;
  }
  if (cs.rate[0].minRate != 64) {
    std::cerr << "FAIL: testThroughputConcrete - expected 64, got "
              << cs.rate[0].minRate << "\n";
    return false;
  }

  std::cerr << "PASS: testThroughputConcrete\n";
  return true;
}

/// T5: LOCAL_SPM placement -> MemoryConstraint
static bool testPlacementLocalSPM() {
  TDCEdgeSpec spec;
  spec.producerKernel = "a";
  spec.consumerKernel = "b";
  spec.dataTypeName = "f32";
  spec.placement = Placement::LOCAL_SPM;

  ContractConstraintTranslator translator;
  auto cs = translator.translate({spec}, {});

  if (cs.memory.size() != 1) {
    std::cerr << "FAIL: testPlacementLocalSPM - expected 1 memory\n";
    return false;
  }
  if (cs.memory[0].level != MemoryLevel::LOCAL_SPM) {
    std::cerr << "FAIL: testPlacementLocalSPM - wrong level\n";
    return false;
  }

  std::cerr << "PASS: testPlacementLocalSPM\n";
  return true;
}

/// T6: AUTO placement -> no MemoryConstraint
static bool testPlacementAuto() {
  TDCEdgeSpec spec;
  spec.producerKernel = "a";
  spec.consumerKernel = "b";
  spec.dataTypeName = "f32";
  spec.placement = Placement::AUTO;

  ContractConstraintTranslator translator;
  auto cs = translator.translate({spec}, {});

  if (!cs.memory.empty()) {
    std::cerr << "FAIL: testPlacementAuto - expected no MemoryConstraint\n";
    return false;
  }

  std::cerr << "PASS: testPlacementAuto\n";
  return true;
}

/// T7: Concrete shape -> TilingConstraint
static bool testShapeConcrete() {
  TDCEdgeSpec spec;
  spec.producerKernel = "a";
  spec.consumerKernel = "b";
  spec.dataTypeName = "f32";
  spec.shape = "[128, 256]";

  ContractConstraintTranslator translator;
  auto cs = translator.translate({spec}, {});

  if (cs.tiling.size() != 1) {
    std::cerr << "FAIL: testShapeConcrete - expected 1 tiling\n";
    return false;
  }
  if (cs.tiling[0].dimensions.size() != 2 ||
      cs.tiling[0].dimensions[0] != 128 ||
      cs.tiling[0].dimensions[1] != 256) {
    std::cerr << "FAIL: testShapeConcrete - wrong dimensions\n";
    return false;
  }

  std::cerr << "PASS: testShapeConcrete\n";
  return true;
}

/// T8: Symbolic shape -> TilingConstraint
static bool testShapeSymbolic() {
  TDCEdgeSpec spec;
  spec.producerKernel = "a";
  spec.consumerKernel = "b";
  spec.dataTypeName = "f32";
  spec.shape = "[batch_size, hidden_dim]";

  std::map<std::string, int64_t> params = {
      {"batch_size", 32}, {"hidden_dim", 512}};
  ContractConstraintTranslator translator(params);
  auto cs = translator.translate({spec}, {});

  if (cs.tiling.size() != 1) {
    std::cerr << "FAIL: testShapeSymbolic - expected 1 tiling\n";
    return false;
  }
  if (cs.tiling[0].dimensions.size() != 2 ||
      cs.tiling[0].dimensions[0] != 32 ||
      cs.tiling[0].dimensions[1] != 512) {
    std::cerr << "FAIL: testShapeSymbolic - wrong dims\n";
    return false;
  }

  std::cerr << "PASS: testShapeSymbolic\n";
  return true;
}

/// T9: Path latency -> PathLatencyConstraint
static bool testPathLatency() {
  TDCPathSpec path;
  path.startProducer = "matmul";
  path.startConsumer = "softmax";
  path.endProducer = "softmax";
  path.endConsumer = "relu";
  path.latency = "4 * tile_m";

  std::map<std::string, int64_t> params = {{"tile_m", 64}};
  ContractConstraintTranslator translator(params);
  auto cs = translator.translate({}, {path});

  if (cs.pathLatency.size() != 1) {
    std::cerr << "FAIL: testPathLatency - expected 1 pathLatency\n";
    return false;
  }
  if (cs.pathLatency[0].maxCycles != 256) {
    std::cerr << "FAIL: testPathLatency - expected 256, got "
              << cs.pathLatency[0].maxCycles << "\n";
    return false;
  }
  if (cs.pathLatency[0].startProducer != "matmul" ||
      cs.pathLatency[0].endConsumer != "relu") {
    std::cerr << "FAIL: testPathLatency - wrong endpoints\n";
    return false;
  }

  std::cerr << "PASS: testPathLatency\n";
  return true;
}

/// T10: Multiple dimensions on one edge -> independent constraints
static bool testMultipleDimensions() {
  TDCEdgeSpec spec;
  spec.producerKernel = "a";
  spec.consumerKernel = "b";
  spec.dataTypeName = "f32";
  spec.ordering = Ordering::FIFO;
  spec.throughput = "64";
  spec.placement = Placement::SHARED_L2;
  spec.shape = "[128, 128]";

  ContractConstraintTranslator translator;
  auto cs = translator.translate({spec}, {});

  if (cs.scheduling.size() != 1 || cs.rate.size() != 1 ||
      cs.memory.size() != 1 || cs.tiling.size() != 1) {
    std::cerr << "FAIL: testMultipleDimensions - expected 1 of each type\n";
    std::cerr << "  scheduling=" << cs.scheduling.size()
              << " rate=" << cs.rate.size()
              << " memory=" << cs.memory.size()
              << " tiling=" << cs.tiling.size() << "\n";
    return false;
  }
  if (!cs.pathLatency.empty()) {
    std::cerr << "FAIL: testMultipleDimensions - unexpected pathLatency\n";
    return false;
  }

  std::cerr << "PASS: testMultipleDimensions\n";
  return true;
}

/// T11: Pruning mask reflects specified dimensions
static bool testPruningMask() {
  TDCEdgeSpec spec;
  spec.producerKernel = "a";
  spec.consumerKernel = "b";
  spec.dataTypeName = "f32";
  spec.ordering = Ordering::FIFO;
  spec.shape = "[64, 64]";
  // throughput and placement unset

  ContractConstraintTranslator translator;
  uint8_t mask = translator.computePruningMask(spec);

  // Bit 0 (ordering) and bit 3 (shape) should be set
  bool bit0 = (mask >> PRUNING_ORDERING_LOCKED) & 1;
  bool bit1 = (mask >> PRUNING_THROUGHPUT_FLOOR) & 1;
  bool bit2 = (mask >> PRUNING_PLACEMENT_LOCKED) & 1;
  bool bit3 = (mask >> PRUNING_SHAPE_LOCKED) & 1;

  if (!bit0 || bit1 || bit2 || !bit3) {
    std::cerr << "FAIL: testPruningMask - expected bits 0,3 set, got mask=0x"
              << std::hex << (int)mask << std::dec << "\n";
    return false;
  }

  std::cerr << "PASS: testPruningMask\n";
  return true;
}

/// T12: Empty edge spec -> zero constraints
static bool testEmptySpec() {
  TDCEdgeSpec spec;
  spec.producerKernel = "a";
  spec.consumerKernel = "b";
  spec.dataTypeName = "f32";
  // All optional dimensions are nullopt

  ContractConstraintTranslator translator;
  auto cs = translator.translate({spec}, {});

  if (!cs.empty()) {
    std::cerr << "FAIL: testEmptySpec - expected empty ConstraintSet\n";
    return false;
  }

  std::cerr << "PASS: testEmptySpec\n";
  return true;
}

/// T13: Symbolic expression evaluator with arithmetic
static bool testEvalArithmetic() {
  std::map<std::string, int64_t> params = {{"a", 10}, {"b", 20}, {"c", 4}};
  auto result = evaluateSymbolicExpr("(a + b) * c / 2", params);

  if (!result.ok()) {
    std::cerr << "FAIL: testEvalArithmetic - error: " << result.error << "\n";
    return false;
  }
  // (10 + 20) * 4 / 2 = 60
  if (*result.value != 60) {
    std::cerr << "FAIL: testEvalArithmetic - expected 60, got "
              << *result.value << "\n";
    return false;
  }

  std::cerr << "PASS: testEvalArithmetic\n";
  return true;
}

/// T14: Symbolic expression evaluator rejects unknown variable
static bool testEvalUnknownVar() {
  std::map<std::string, int64_t> params = {{"a", 10}};
  auto result = evaluateSymbolicExpr("a + unknown_var", params);

  if (result.ok()) {
    std::cerr << "FAIL: testEvalUnknownVar - expected error\n";
    return false;
  }
  if (result.error.find("unknown_var") == std::string::npos) {
    std::cerr << "FAIL: testEvalUnknownVar - error should mention "
                 "'unknown_var': " << result.error << "\n";
    return false;
  }

  std::cerr << "PASS: testEvalUnknownVar\n";
  return true;
}

/// T15: Unary minus support
static bool testEvalUnaryMinus() {
  std::map<std::string, int64_t> params = {{"x", 5}};
  auto result = evaluateSymbolicExpr("-x + 10", params);

  if (!result.ok()) {
    std::cerr << "FAIL: testEvalUnaryMinus - error: " << result.error << "\n";
    return false;
  }
  if (*result.value != 5) {
    std::cerr << "FAIL: testEvalUnaryMinus - expected 5, got "
              << *result.value << "\n";
    return false;
  }

  std::cerr << "PASS: testEvalUnaryMinus\n";
  return true;
}

/// T16: Division by zero detection
static bool testEvalDivZero() {
  std::map<std::string, int64_t> params = {};
  auto result = evaluateSymbolicExpr("10 / 0", params);

  if (result.ok()) {
    std::cerr << "FAIL: testEvalDivZero - expected error\n";
    return false;
  }
  if (result.error.find("division by zero") == std::string::npos) {
    std::cerr << "FAIL: testEvalDivZero - wrong error: " << result.error
              << "\n";
    return false;
  }

  std::cerr << "PASS: testEvalDivZero\n";
  return true;
}

/// T17: Pruning masks batch computation
static bool testPruningMasksBatch() {
  TDCEdgeSpec e1;
  e1.producerKernel = "a";
  e1.consumerKernel = "b";
  e1.dataTypeName = "f32";
  e1.ordering = Ordering::FIFO;

  TDCEdgeSpec e2;
  e2.producerKernel = "b";
  e2.consumerKernel = "c";
  e2.dataTypeName = "f32";
  e2.placement = Placement::SHARED_L2;
  e2.throughput = "100";

  ContractConstraintTranslator translator;
  auto masks = translator.computePruningMasks({e1, e2});

  EdgeKey k1{"a", "b"};
  EdgeKey k2{"b", "c"};

  if (masks.count(k1) == 0 || masks.count(k2) == 0) {
    std::cerr << "FAIL: testPruningMasksBatch - missing edge keys\n";
    return false;
  }

  // e1: only ordering set -> bit 0
  if (masks[k1] != (1u << PRUNING_ORDERING_LOCKED)) {
    std::cerr << "FAIL: testPruningMasksBatch - e1 mask wrong: 0x"
              << std::hex << (int)masks[k1] << std::dec << "\n";
    return false;
  }

  // e2: placement + throughput -> bits 1, 2
  uint8_t expected = (1u << PRUNING_THROUGHPUT_FLOOR) |
                     (1u << PRUNING_PLACEMENT_LOCKED);
  if (masks[k2] != expected) {
    std::cerr << "FAIL: testPruningMasksBatch - e2 mask wrong: 0x"
              << std::hex << (int)masks[k2] << std::dec << "\n";
    return false;
  }

  std::cerr << "PASS: testPruningMasksBatch\n";
  return true;
}

/// T18: Diagnostic is emitted when throughput expression has unknown variable
static bool testDiagnosticOnBadThroughput() {
  TDCEdgeSpec spec;
  spec.producerKernel = "a";
  spec.consumerKernel = "b";
  spec.dataTypeName = "f32";
  spec.throughput = "unknown_var * 10";

  ContractConstraintTranslator translator; // no params
  auto cs = translator.translate({spec}, {});

  if (!cs.rate.empty()) {
    std::cerr << "FAIL: testDiagnosticOnBadThroughput - should emit no rate\n";
    return false;
  }
  if (cs.diagnostics.empty()) {
    std::cerr << "FAIL: testDiagnosticOnBadThroughput - expected diagnostic\n";
    return false;
  }

  std::cerr << "PASS: testDiagnosticOnBadThroughput\n";
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

  run(testOrderingFIFO);
  run(testOrderingUnordered);
  run(testThroughputSymbolic);
  run(testThroughputConcrete);
  run(testPlacementLocalSPM);
  run(testPlacementAuto);
  run(testShapeConcrete);
  run(testShapeSymbolic);
  run(testPathLatency);
  run(testMultipleDimensions);
  run(testPruningMask);
  run(testEmptySpec);
  run(testEvalArithmetic);
  run(testEvalUnknownVar);
  run(testEvalUnaryMinus);
  run(testEvalDivZero);
  run(testPruningMasksBatch);
  run(testDiagnosticOnBadThroughput);

  std::cerr << "\nResults: " << passed << "/" << total << " tests passed\n";
  return (passed == total) ? 0 : 1;
}
