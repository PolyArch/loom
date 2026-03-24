/// TaskGraph API unit tests.
///
/// Tests:
/// 1. Basic graph construction (kernels + edges)
/// 2. Contract field propagation via chainable setters
/// 3. Default contract values (all nullopt)
/// 4. Edge lookup by kernel names
/// 5. Kernel execution target
/// 6. DerivedContractMetrics computation
/// 7. TDG MLIR emission

#include "tapestry/task_graph.h"
#include "tapestry/compile.h"
#include "tapestry/derived_metrics.h"
#include "tapestry/tdg_emitter.h"

#include "loom/Dialect/TDG/TDGDialect.h"
#include "loom/Dialect/TDG/TDGOps.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

using namespace tapestry;

// Dummy kernel functions (never executed, only referenced by pointer).
static void fft(float *, float *, int) {}
static void filter(float *, float *, int) {}
static void post_process(float *, float *, int) {}

// -------------------------------------------------------------------------
// Test 1: Basic graph construction
// -------------------------------------------------------------------------
static bool testBasicConstruction() {
  TaskGraph tdg("test_pipeline");

  auto k1 = tdg.kernel("fft", fft);
  auto k2 = tdg.kernel("filter", filter);
  auto k3 = tdg.kernel("post", post_process);

  tdg.connect(k1, k2);
  tdg.connect(k2, k3);

  if (tdg.numKernels() != 3) {
    std::cerr << "FAIL: testBasicConstruction - numKernels="
              << tdg.numKernels() << " (expected 3)\n";
    return false;
  }
  if (tdg.numEdges() != 2) {
    std::cerr << "FAIL: testBasicConstruction - numEdges="
              << tdg.numEdges() << " (expected 2)\n";
    return false;
  }

  // Verify kernel names via forEachKernel.
  std::vector<std::string> names;
  tdg.forEachKernel([&](const KernelInfo &ki) { names.push_back(ki.name); });
  if (names.size() != 3 || names[0] != "fft" || names[1] != "filter" ||
      names[2] != "post") {
    std::cerr << "FAIL: testBasicConstruction - kernel names\n";
    return false;
  }

  std::cout << "PASS: testBasicConstruction\n";
  return true;
}

// -------------------------------------------------------------------------
// Test 2: Contract field propagation
// -------------------------------------------------------------------------
static bool testContractPropagation() {
  TaskGraph tdg("contract_test");

  auto k1 = tdg.kernel("fft", fft);
  auto k2 = tdg.kernel("filter", filter);

  tdg.connect(k1, k2)
      .ordering(Ordering::FIFO)
      .data_type<float>()
      .tile_shape({64})
      .visibility(Visibility::LOCAL_SPM)
      .rate(1024)
      .double_buffering(true)
      .backpressure(Backpressure::BLOCK)
      .may_fuse(false)
      .may_replicate(true);

  // Inspect via forEachEdge.
  bool found = false;
  tdg.forEachEdge([&](const std::string &prod, const std::string &cons,
                       const Contract &c) {
    if (prod == "fft" && cons == "filter") {
      found = true;
      if (!c.ordering || *c.ordering != Ordering::FIFO) {
        std::cerr << "FAIL: testContractPropagation - ordering\n";
        found = false;
      }
      if (!c.dataTypeName || *c.dataTypeName != "f32") {
        std::cerr << "FAIL: testContractPropagation - dataTypeName\n";
        found = false;
      }
      if (!c.tileShape || c.tileShape->size() != 1 ||
          (*c.tileShape)[0] != 64) {
        std::cerr << "FAIL: testContractPropagation - tileShape\n";
        found = false;
      }
      if (!c.visibility || *c.visibility != Visibility::LOCAL_SPM) {
        std::cerr << "FAIL: testContractPropagation - visibility\n";
        found = false;
      }
      if (!c.rate || *c.rate != 1024) {
        std::cerr << "FAIL: testContractPropagation - rate\n";
        found = false;
      }
      if (!c.doubleBuffering || *c.doubleBuffering != true) {
        std::cerr << "FAIL: testContractPropagation - doubleBuffering\n";
        found = false;
      }
      if (c.mayFuse != false) {
        std::cerr << "FAIL: testContractPropagation - mayFuse\n";
        found = false;
      }
      if (c.mayReplicate != true) {
        std::cerr << "FAIL: testContractPropagation - mayReplicate\n";
        found = false;
      }
    }
  });
  if (!found) {
    std::cerr << "FAIL: testContractPropagation - edge not found\n";
    return false;
  }

  std::cout << "PASS: testContractPropagation\n";
  return true;
}

// -------------------------------------------------------------------------
// Test 3: Default contract values (all nullopt)
// -------------------------------------------------------------------------
static bool testDefaults() {
  TaskGraph tdg("defaults_test");

  auto k1 = tdg.kernel("a", fft);
  auto k2 = tdg.kernel("b", filter);

  tdg.connect(k1, k2); // no setters

  bool ok = true;
  tdg.forEachEdge([&](const std::string &, const std::string &,
                       const Contract &c) {
    if (c.ordering.has_value()) {
      std::cerr << "FAIL: testDefaults - ordering should be nullopt\n";
      ok = false;
    }
    if (c.dataTypeName.has_value()) {
      std::cerr << "FAIL: testDefaults - dataTypeName should be nullopt\n";
      ok = false;
    }
    if (c.rate.has_value()) {
      std::cerr << "FAIL: testDefaults - rate should be nullopt\n";
      ok = false;
    }
    if (c.tileShape.has_value()) {
      std::cerr << "FAIL: testDefaults - tileShape should be nullopt\n";
      ok = false;
    }
    if (c.visibility.has_value()) {
      std::cerr << "FAIL: testDefaults - visibility should be nullopt\n";
      ok = false;
    }
    // Permission defaults.
    if (!c.mayFuse || !c.mayReplicate || !c.mayPipeline || !c.mayRetile) {
      std::cerr << "FAIL: testDefaults - permission defaults\n";
      ok = false;
    }
    if (c.mayReorder) {
      std::cerr << "FAIL: testDefaults - mayReorder should be false\n";
      ok = false;
    }
  });

  if (!ok)
    return false;
  std::cout << "PASS: testDefaults\n";
  return true;
}

// -------------------------------------------------------------------------
// Test 4: Edge lookup by kernel names
// -------------------------------------------------------------------------
static bool testEdgeLookup() {
  TaskGraph tdg("lookup_test");

  auto k1 = tdg.kernel("fft", fft);
  auto k2 = tdg.kernel("filter", filter);

  tdg.connect(k1, k2).tile_shape({128, 64});

  // Look up by name.
  auto e = tdg.edge("fft", "filter");
  const auto &c = e.contract();
  if (!c.tileShape || c.tileShape->size() != 2 ||
      (*c.tileShape)[0] != 128 || (*c.tileShape)[1] != 64) {
    std::cerr << "FAIL: testEdgeLookup - tileShape mismatch\n";
    return false;
  }
  if (e.producerName() != "fft" || e.consumerName() != "filter") {
    std::cerr << "FAIL: testEdgeLookup - endpoint names\n";
    return false;
  }

  std::cout << "PASS: testEdgeLookup\n";
  return true;
}

// -------------------------------------------------------------------------
// Test 5: Kernel execution target
// -------------------------------------------------------------------------
static bool testExecutionTarget() {
  TaskGraph tdg("target_test");

  auto k1 = tdg.kernel("fft", fft);
  k1.target(ExecutionTarget::CGRA);

  auto k2 = tdg.kernel("normalize", filter);
  k2.target(ExecutionTarget::HOST);

  if (k1.executionTarget() != ExecutionTarget::CGRA) {
    std::cerr << "FAIL: testExecutionTarget - k1 target\n";
    return false;
  }
  if (k2.executionTarget() != ExecutionTarget::HOST) {
    std::cerr << "FAIL: testExecutionTarget - k2 target\n";
    return false;
  }

  // Default should be AUTO_DETECT.
  auto k3 = tdg.kernel("post", post_process);
  if (k3.executionTarget() != ExecutionTarget::AUTO_DETECT) {
    std::cerr << "FAIL: testExecutionTarget - k3 default\n";
    return false;
  }

  std::cout << "PASS: testExecutionTarget\n";
  return true;
}

// -------------------------------------------------------------------------
// Test 6: DerivedContractMetrics computation
// -------------------------------------------------------------------------
static bool testDerivedMetrics() {
  Contract c;
  c.rate = 1024;
  c.tileShape = std::vector<int64_t>{64};

  auto m = computeDerivedMetrics(c, 4); // f32 = 4 bytes

  // bandwidth = rate * element_size = 1024 * 4 = 4096
  if (std::abs(m.bandwidth - 4096.0) > 1e-6) {
    std::cerr << "FAIL: testDerivedMetrics - bandwidth=" << m.bandwidth
              << " (expected 4096)\n";
    return false;
  }

  // dataVolume = rate * tile_elements * element_size = 1024 * 64 * 4 = 262144
  if (m.dataVolume != 1024 * 64 * 4) {
    std::cerr << "FAIL: testDerivedMetrics - dataVolume=" << m.dataVolume
              << "\n";
    return false;
  }

  // Before assignment, crossesCores should be false.
  if (m.crossesCores) {
    std::cerr << "FAIL: testDerivedMetrics - crossesCores\n";
    return false;
  }

  // After assignment with crossesCores=true.
  updatePostAssignment(m, /*crossesCores=*/true, Visibility::LOCAL_SPM);
  if (!m.crossesCores) {
    std::cerr << "FAIL: testDerivedMetrics - crossesCores post-assign\n";
    return false;
  }
  if (std::abs(m.requiredNoCBandwidth - 4096.0) > 1e-6) {
    std::cerr << "FAIL: testDerivedMetrics - requiredNoCBandwidth\n";
    return false;
  }
  if (m.requiredSPMBytes != 1024 * 64 * 4) {
    std::cerr << "FAIL: testDerivedMetrics - requiredSPMBytes\n";
    return false;
  }

  // With SHARED_L2 visibility, requiredSPMBytes should be 0.
  updatePostAssignment(m, /*crossesCores=*/true, Visibility::SHARED_L2);
  if (m.requiredSPMBytes != 0) {
    std::cerr << "FAIL: testDerivedMetrics - requiredSPMBytes SHARED_L2\n";
    return false;
  }

  std::cout << "PASS: testDerivedMetrics\n";
  return true;
}

// -------------------------------------------------------------------------
// Test 7: TDG MLIR emission
// -------------------------------------------------------------------------
static bool testMLIREmission() {
  TaskGraph tdg("emit_test");

  auto k1 = tdg.kernel("fft", fft);
  k1.target(ExecutionTarget::CGRA);
  auto k2 = tdg.kernel("filter", filter);

  tdg.connect(k1, k2)
      .ordering(Ordering::FIFO)
      .data_type<float>()
      .tile_shape({64});

  mlir::MLIRContext ctx;
  auto module = emitTDG(tdg, ctx);
  if (!module) {
    std::cerr << "FAIL: testMLIREmission - null module\n";
    return false;
  }

  // Walk the module and count graph/kernel/contract ops.
  unsigned graphCount = 0, kernelCount = 0, contractCount = 0;
  module->walk([&](mlir::Operation *op) {
    if (llvm::isa<loom::tdg::GraphOp>(op))
      ++graphCount;
    else if (llvm::isa<loom::tdg::KernelOp>(op))
      ++kernelCount;
    else if (llvm::isa<loom::tdg::ContractOp>(op))
      ++contractCount;
  });

  if (graphCount != 1) {
    std::cerr << "FAIL: testMLIREmission - graphCount=" << graphCount << "\n";
    return false;
  }
  if (kernelCount != 2) {
    std::cerr << "FAIL: testMLIREmission - kernelCount=" << kernelCount
              << "\n";
    return false;
  }
  if (contractCount != 1) {
    std::cerr << "FAIL: testMLIREmission - contractCount=" << contractCount
              << "\n";
    return false;
  }

  std::cout << "PASS: testMLIREmission\n";
  return true;
}

// -------------------------------------------------------------------------
// Test 8: Kernel provenance
// -------------------------------------------------------------------------
static bool testKernelProvenance() {
  TaskGraph tdg("provenance_test");

  auto k = tdg.kernel("fft", fft);
  const auto &prov = k.provenance();

  if (prov.functionName != "fft") {
    std::cerr << "FAIL: testKernelProvenance - functionName\n";
    return false;
  }
  if (prov.funcPtr != reinterpret_cast<void *>(fft)) {
    std::cerr << "FAIL: testKernelProvenance - funcPtr\n";
    return false;
  }

  std::cout << "PASS: testKernelProvenance\n";
  return true;
}

// -------------------------------------------------------------------------
// Test 9: Name-only kernel (auto_analyze path)
// -------------------------------------------------------------------------
static bool testNameOnlyKernel() {
  TaskGraph tdg("nameonly_test");

  auto k1 = tdg.kernel("analyze_fft");
  auto k2 = tdg.kernel("analyze_filter");
  tdg.connect(k1, k2);

  if (tdg.numKernels() != 2 || tdg.numEdges() != 1) {
    std::cerr << "FAIL: testNameOnlyKernel - counts\n";
    return false;
  }

  if (k1.provenance().funcPtr != nullptr) {
    std::cerr << "FAIL: testNameOnlyKernel - funcPtr should be null\n";
    return false;
  }

  std::cout << "PASS: testNameOnlyKernel\n";
  return true;
}

// -------------------------------------------------------------------------
// Test 10: DOT dump
// -------------------------------------------------------------------------
static bool testDotDump() {
  TaskGraph tdg("dot_test");

  auto k1 = tdg.kernel("a", fft);
  auto k2 = tdg.kernel("b", filter);
  tdg.connect(k1, k2).ordering(Ordering::FIFO).data_type<float>();

  std::string dot = tdg.dumpDot();
  // Verify it contains the graph name and kernel labels.
  if (dot.find("dot_test") == std::string::npos) {
    std::cerr << "FAIL: testDotDump - graph name not found\n";
    return false;
  }
  if (dot.find("\"a\"") == std::string::npos) {
    std::cerr << "FAIL: testDotDump - kernel 'a' label not found\n";
    return false;
  }
  if (dot.find("->") == std::string::npos) {
    std::cerr << "FAIL: testDotDump - no edge arrow\n";
    return false;
  }

  std::cout << "PASS: testDotDump\n";
  return true;
}

// -------------------------------------------------------------------------
// Test 11: dataTypeBytes helper
// -------------------------------------------------------------------------
static bool testDataTypeBytes() {
  if (dataTypeBytes("f32") != 4) {
    std::cerr << "FAIL: testDataTypeBytes - f32\n";
    return false;
  }
  if (dataTypeBytes("f64") != 8) {
    std::cerr << "FAIL: testDataTypeBytes - f64\n";
    return false;
  }
  if (dataTypeBytes("i8") != 1) {
    std::cerr << "FAIL: testDataTypeBytes - i8\n";
    return false;
  }
  if (dataTypeBytes("unknown") != 0) {
    std::cerr << "FAIL: testDataTypeBytes - unknown\n";
    return false;
  }

  std::cout << "PASS: testDataTypeBytes\n";
  return true;
}

// =========================================================================
// Main
// =========================================================================
int main() {
  int passed = 0, failed = 0;

  auto run = [&](bool (*test)(), const char *name) {
    if (test()) {
      ++passed;
    } else {
      std::cerr << "  FAILED: " << name << "\n";
      ++failed;
    }
  };

  run(testBasicConstruction, "testBasicConstruction");
  run(testContractPropagation, "testContractPropagation");
  run(testDefaults, "testDefaults");
  run(testEdgeLookup, "testEdgeLookup");
  run(testExecutionTarget, "testExecutionTarget");
  run(testDerivedMetrics, "testDerivedMetrics");
  run(testMLIREmission, "testMLIREmission");
  run(testKernelProvenance, "testKernelProvenance");
  run(testNameOnlyKernel, "testNameOnlyKernel");
  run(testDotDump, "testDotDump");
  run(testDataTypeBytes, "testDataTypeBytes");

  std::cout << "\n" << passed << " passed, " << failed << " failed\n";
  return failed > 0 ? 1 : 0;
}
