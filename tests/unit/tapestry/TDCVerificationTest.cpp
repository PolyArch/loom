/// TDCVerification tests: edge/path validation, batch verification,
/// and diagnostic output.
///
/// Tests:
///  1. Valid edge spec passes verification
///  2. Edge spec with empty producerKernel fails
///  3. Edge spec with empty dataTypeName fails
///  4. Edge spec with valid shape passes
///  5. Valid path spec passes verification
///  6. Path spec with empty latency fails
///  7. Path spec with empty endpoint fails
///  8. Batch verification merges diagnostics
///  9. Batch verification with all valid specs passes

#include "loom/SystemCompiler/TDCVerification.h"

#include <cassert>
#include <iostream>
#include <string>

using namespace loom;

/// T1: Valid edge spec passes.
static bool testValidEdge() {
  TDCEdgeSpec spec;
  spec.producerKernel = "matmul";
  spec.consumerKernel = "softmax";
  spec.dataTypeName = "f32";
  spec.ordering = Ordering::FIFO;

  auto result = verifyEdgeSpec(spec);

  if (!result.valid) {
    std::cerr << "FAIL: testValidEdge - expected valid\n";
    for (const auto &d : result.diagnostics)
      std::cerr << "  diagnostic: " << d.message << "\n";
    return false;
  }

  std::cerr << "PASS: testValidEdge\n";
  return true;
}

/// T2: Edge spec with empty producerKernel fails.
static bool testEdgeEmptyProducer() {
  TDCEdgeSpec spec;
  spec.producerKernel = "";
  spec.consumerKernel = "softmax";
  spec.dataTypeName = "f32";

  auto result = verifyEdgeSpec(spec);

  if (result.valid) {
    std::cerr << "FAIL: testEdgeEmptyProducer - expected invalid\n";
    return false;
  }

  bool foundError = false;
  for (const auto &d : result.diagnostics) {
    if (d.severity == TDCDiagnostic::Severity::Error &&
        d.message.find("producerKernel") != std::string::npos) {
      foundError = true;
    }
  }
  if (!foundError) {
    std::cerr << "FAIL: testEdgeEmptyProducer - missing producerKernel error\n";
    return false;
  }

  std::cerr << "PASS: testEdgeEmptyProducer\n";
  return true;
}

/// T3: Edge spec with empty dataTypeName fails.
static bool testEdgeEmptyDataType() {
  TDCEdgeSpec spec;
  spec.producerKernel = "matmul";
  spec.consumerKernel = "softmax";
  spec.dataTypeName = "";

  auto result = verifyEdgeSpec(spec);

  if (result.valid) {
    std::cerr << "FAIL: testEdgeEmptyDataType - expected invalid\n";
    return false;
  }

  bool foundError = false;
  for (const auto &d : result.diagnostics) {
    if (d.severity == TDCDiagnostic::Severity::Error &&
        d.message.find("dataTypeName") != std::string::npos) {
      foundError = true;
    }
  }
  if (!foundError) {
    std::cerr << "FAIL: testEdgeEmptyDataType - missing dataTypeName error\n";
    return false;
  }

  std::cerr << "PASS: testEdgeEmptyDataType\n";
  return true;
}

/// T4: Edge spec with valid shape passes (and shape parses correctly).
static bool testEdgeWithShape() {
  TDCEdgeSpec spec;
  spec.producerKernel = "matmul";
  spec.consumerKernel = "relu";
  spec.dataTypeName = "f32";
  spec.shape = "[128, 64]";

  auto result = verifyEdgeSpec(spec);

  if (!result.valid) {
    std::cerr << "FAIL: testEdgeWithShape - expected valid\n";
    for (const auto &d : result.diagnostics)
      std::cerr << "  diagnostic: " << d.message << "\n";
    return false;
  }

  std::cerr << "PASS: testEdgeWithShape\n";
  return true;
}

/// T5: Valid path spec passes.
static bool testValidPath() {
  TDCPathSpec spec;
  spec.startProducer = "matmul";
  spec.startConsumer = "softmax";
  spec.endProducer = "softmax";
  spec.endConsumer = "relu";
  spec.latency = "4 * tile_m";

  auto result = verifyPathSpec(spec);

  if (!result.valid) {
    std::cerr << "FAIL: testValidPath - expected valid\n";
    for (const auto &d : result.diagnostics)
      std::cerr << "  diagnostic: " << d.message << "\n";
    return false;
  }

  std::cerr << "PASS: testValidPath\n";
  return true;
}

/// T6: Path spec with empty latency fails.
static bool testPathEmptyLatency() {
  TDCPathSpec spec;
  spec.startProducer = "matmul";
  spec.startConsumer = "softmax";
  spec.endProducer = "softmax";
  spec.endConsumer = "relu";
  spec.latency = "";

  auto result = verifyPathSpec(spec);

  if (result.valid) {
    std::cerr << "FAIL: testPathEmptyLatency - expected invalid\n";
    return false;
  }

  bool foundError = false;
  for (const auto &d : result.diagnostics) {
    if (d.severity == TDCDiagnostic::Severity::Error &&
        d.message.find("latency") != std::string::npos) {
      foundError = true;
    }
  }
  if (!foundError) {
    std::cerr << "FAIL: testPathEmptyLatency - missing latency error\n";
    return false;
  }

  std::cerr << "PASS: testPathEmptyLatency\n";
  return true;
}

/// T7: Path spec with empty endpoint fails.
static bool testPathEmptyEndpoint() {
  TDCPathSpec spec;
  spec.startProducer = "matmul";
  spec.startConsumer = "";
  spec.endProducer = "softmax";
  spec.endConsumer = "relu";
  spec.latency = "100";

  auto result = verifyPathSpec(spec);

  if (result.valid) {
    std::cerr << "FAIL: testPathEmptyEndpoint - expected invalid\n";
    return false;
  }

  std::cerr << "PASS: testPathEmptyEndpoint\n";
  return true;
}

/// T8: Batch verification merges diagnostics.
static bool testBatchVerificationErrors() {
  // One invalid edge, one invalid path.
  TDCEdgeSpec badEdge;
  badEdge.producerKernel = "";
  badEdge.consumerKernel = "b";
  badEdge.dataTypeName = "f32";

  TDCPathSpec badPath;
  badPath.startProducer = "a";
  badPath.startConsumer = "b";
  badPath.endProducer = "c";
  badPath.endConsumer = "d";
  badPath.latency = "";

  auto result = verifyContracts({badEdge}, {badPath});

  if (result.valid) {
    std::cerr << "FAIL: testBatchVerificationErrors - expected invalid\n";
    return false;
  }

  // Should have at least 2 errors (one from edge, one from path).
  int errorCount = 0;
  for (const auto &d : result.diagnostics) {
    if (d.severity == TDCDiagnostic::Severity::Error)
      errorCount++;
  }
  if (errorCount < 2) {
    std::cerr << "FAIL: testBatchVerificationErrors - expected >= 2 errors, "
                 "got "
              << errorCount << "\n";
    return false;
  }

  std::cerr << "PASS: testBatchVerificationErrors\n";
  return true;
}

/// T9: Batch verification with all valid specs passes.
static bool testBatchVerificationAllValid() {
  TDCEdgeSpec e1;
  e1.producerKernel = "a";
  e1.consumerKernel = "b";
  e1.dataTypeName = "f32";

  TDCPathSpec p1;
  p1.startProducer = "a";
  p1.startConsumer = "b";
  p1.endProducer = "b";
  p1.endConsumer = "c";
  p1.latency = "100";

  auto result = verifyContracts({e1}, {p1});

  if (!result.valid) {
    std::cerr << "FAIL: testBatchVerificationAllValid - expected valid\n";
    for (const auto &d : result.diagnostics)
      std::cerr << "  diagnostic: " << d.message << "\n";
    return false;
  }

  std::cerr << "PASS: testBatchVerificationAllValid\n";
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

  run(testValidEdge);
  run(testEdgeEmptyProducer);
  run(testEdgeEmptyDataType);
  run(testEdgeWithShape);
  run(testValidPath);
  run(testPathEmptyLatency);
  run(testPathEmptyEndpoint);
  run(testBatchVerificationErrors);
  run(testBatchVerificationAllValid);

  std::cerr << "\nResults: " << passed << "/" << total << " tests passed\n";
  return (passed == total) ? 0 : 1;
}
