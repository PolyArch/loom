/// Auto-analyze bridge unit tests.
///
/// Tests:
/// T1: buildTaskGraphFromAnalysis basic conversion (2 kernels, 1 edge)
/// T2: buildTaskGraphFromAnalysis with zero edges (3 kernels, 0 edges)
/// T3: Kernel target mapping (CGRA, HOST, AUTO)
/// T4: sizeOfType correctness for all type strings
/// T5: Default + unroll variant registration
/// T6: TDGToSSGBuilder node count from TDG MLIR
/// T7: TDGToSSGBuilder edge data volume propagation
/// T8: TDGToSSGBuilder handles missing DFG module gracefully
/// T9: Provenance sourcePath is propagated to kernels

#include "tapestry/auto_analyze.h"
#include "tapestry/task_graph.h"
#include "tapestry/tdg_emitter.h"

#include "loom/Dialect/TDG/TDGDialect.h"
#include "loom/Dialect/TDG/TDGOps.h"
#include "loom/SystemCompiler/TDGToSSGBuilder.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

#include <cassert>
#include <cstdint>
#include <iostream>
#include <map>
#include <string>
#include <vector>

using namespace tapestry;

//===----------------------------------------------------------------------===//
// Test helpers
//===----------------------------------------------------------------------===//

/// Build a simple AutoAnalyzeResult with the given number of kernels and edges.
static AutoAnalyzeResult makeResult(
    const std::vector<std::pair<std::string, KernelTarget>> &kernels,
    const std::vector<std::tuple<unsigned, unsigned, std::string, bool,
                                 std::optional<uint64_t>>> &edges) {

  AutoAnalyzeResult result;
  result.success = true;
  result.sourcePath = "test.c";
  result.entryFunc = "test_entry";

  unsigned order = 0;
  for (const auto &[name, target] : kernels) {
    CallSiteBinding binding;
    binding.kernelName = name;
    binding.target = target;
    binding.callOrder = order++;
    result.callBindings.push_back(std::move(binding));
  }

  for (const auto &[prodIdx, consIdx, dataType, isSeq, elemCount] : edges) {
    InferredEdge edge;
    edge.producerIndex = prodIdx;
    edge.consumerIndex = consIdx;
    edge.dependency.exists = true;
    edge.dependency.dataType = dataType;
    edge.dependency.isSequential = isSeq;
    edge.dependency.elementCount = elemCount;
    edge.ordering =
        isSeq ? loom::Ordering::FIFO : loom::Ordering::UNORDERED;
    result.edges.push_back(std::move(edge));
  }

  return result;
}

//===----------------------------------------------------------------------===//
// T1: buildTaskGraphFromAnalysis basic conversion
//===----------------------------------------------------------------------===//
static bool testBasicConversion() {
  auto result = makeResult(
      {{"alpha", KernelTarget::CGRA}, {"beta", KernelTarget::CGRA}},
      {{0, 1, "f32", true, std::optional<uint64_t>(256)}});

  TaskGraph tg = buildTaskGraphFromAnalysis(result);

  if (tg.numKernels() != 2) {
    std::cerr << "FAIL T1: numKernels=" << tg.numKernels()
              << " (expected 2)\n";
    return false;
  }
  if (tg.numEdges() != 1) {
    std::cerr << "FAIL T1: numEdges=" << tg.numEdges() << " (expected 1)\n";
    return false;
  }

  // Check edge contract.
  auto eh = tg.edge("alpha", "beta");
  const auto &contract = eh.contract();
  if (!contract.ordering || *contract.ordering != Ordering::FIFO) {
    std::cerr << "FAIL T1: ordering should be FIFO\n";
    return false;
  }
  if (!contract.dataTypeName || *contract.dataTypeName != "f32") {
    std::cerr << "FAIL T1: dataTypeName should be f32\n";
    return false;
  }
  // Data volume = 256 * 4 = 1024.
  if (!contract.dataVolume || *contract.dataVolume != 1024) {
    std::cerr << "FAIL T1: dataVolume should be 1024, got "
              << (contract.dataVolume ? std::to_string(*contract.dataVolume)
                                      : "nullopt")
              << "\n";
    return false;
  }

  std::cout << "PASS T1: basic conversion\n";
  return true;
}

//===----------------------------------------------------------------------===//
// T2: buildTaskGraphFromAnalysis with zero edges
//===----------------------------------------------------------------------===//
static bool testZeroEdges() {
  auto result = makeResult(
      {{"k1", KernelTarget::CGRA},
       {"k2", KernelTarget::HOST},
       {"k3", KernelTarget::AUTO}},
      {});

  TaskGraph tg = buildTaskGraphFromAnalysis(result);

  if (tg.numKernels() != 3) {
    std::cerr << "FAIL T2: numKernels=" << tg.numKernels()
              << " (expected 3)\n";
    return false;
  }
  if (tg.numEdges() != 0) {
    std::cerr << "FAIL T2: numEdges=" << tg.numEdges() << " (expected 0)\n";
    return false;
  }

  std::cout << "PASS T2: zero edges\n";
  return true;
}

//===----------------------------------------------------------------------===//
// T3: Kernel target mapping
//===----------------------------------------------------------------------===//
static bool testTargetMapping() {
  auto result = makeResult(
      {{"k_cgra", KernelTarget::CGRA},
       {"k_host", KernelTarget::HOST},
       {"k_auto", KernelTarget::AUTO}},
      {});

  TaskGraph tg = buildTaskGraphFromAnalysis(result);

  // Check each kernel's execution target by iterating.
  std::map<std::string, ExecutionTarget> targets;
  tg.forEachKernel([&](const KernelInfo &ki) { targets[ki.name] = ki.target; });

  if (targets["k_cgra"] != ExecutionTarget::CGRA) {
    std::cerr << "FAIL T3: k_cgra target mismatch\n";
    return false;
  }
  if (targets["k_host"] != ExecutionTarget::HOST) {
    std::cerr << "FAIL T3: k_host target mismatch\n";
    return false;
  }
  if (targets["k_auto"] != ExecutionTarget::AUTO_DETECT) {
    std::cerr << "FAIL T3: k_auto target mismatch\n";
    return false;
  }

  std::cout << "PASS T3: target mapping\n";
  return true;
}

//===----------------------------------------------------------------------===//
// T4: sizeOfType correctness
//===----------------------------------------------------------------------===//
static bool testSizeOfType() {
  struct TestCase {
    std::string typeName;
    unsigned expected;
  };

  std::vector<TestCase> cases = {
      {"f32", 4},  {"f64", 8},  {"i8", 1},     {"i16", 2},
      {"i32", 4},  {"i64", 8},  {"u8", 1},     {"u16", 2},
      {"u32", 4},  {"u64", 8},  {"f16", 2},    {"unknown", 4},
  };

  for (const auto &tc : cases) {
    unsigned actual = sizeOfType(tc.typeName);
    if (actual != tc.expected) {
      std::cerr << "FAIL T4: sizeOfType(\"" << tc.typeName << "\") = " << actual
                << " (expected " << tc.expected << ")\n";
      return false;
    }
  }

  std::cout << "PASS T4: sizeOfType correctness\n";
  return true;
}

//===----------------------------------------------------------------------===//
// T5: Default + unroll variant registration
//===----------------------------------------------------------------------===//
static bool testDefaultVariant() {
  auto result = makeResult({{"single_kernel", KernelTarget::CGRA}}, {});

  TaskGraph tg = buildTaskGraphFromAnalysis(result);

  if (tg.numKernels() < 1) {
    std::cerr << "FAIL T5: expected at least 1 kernel\n";
    return false;
  }

  // After the fix, each kernel should have 2 variants:
  //   (1) the default (unroll=1) created by addKernelImpl
  //   (2) the unroll-2 variant registered by buildTaskGraphFromAnalysis
  const auto &variants = tg.variants(0u);
  if (variants.size() != 2) {
    std::cerr << "FAIL T5: expected 2 variants, got "
              << variants.size() << "\n";
    return false;
  }

  // Default variant.
  if (variants[0].variantName != "single_kernel_default") {
    std::cerr << "FAIL T5: default variant name='"
              << variants[0].variantName << "'\n";
    return false;
  }
  if (variants[0].options.unrollFactor != 1) {
    std::cerr << "FAIL T5: default variant unrollFactor="
              << variants[0].options.unrollFactor << "\n";
    return false;
  }

  // Unroll-2 variant.
  if (variants[1].variantName != "single_kernel_u2") {
    std::cerr << "FAIL T5: unroll variant name='"
              << variants[1].variantName << "'\n";
    return false;
  }
  if (variants[1].options.unrollFactor != 2) {
    std::cerr << "FAIL T5: unroll variant unrollFactor="
              << variants[1].options.unrollFactor << "\n";
    return false;
  }

  std::cout << "PASS T5: default + unroll variant registration\n";
  return true;
}

//===----------------------------------------------------------------------===//
// T6: TDGToSSGBuilder node count
//===----------------------------------------------------------------------===//
static bool testSSGNodeCount() {
  // Build a 3-kernel, 2-edge linear pipeline through the full path.
  auto result = makeResult(
      {{"k1", KernelTarget::CGRA},
       {"k2", KernelTarget::CGRA},
       {"k3", KernelTarget::CGRA}},
      {{0, 1, "f32", true, std::nullopt},
       {1, 2, "f32", true, std::nullopt}});

  TaskGraph tg = buildTaskGraphFromAnalysis(result);

  mlir::MLIRContext ctx;
  ctx.getOrLoadDialect<loom::tdg::TDGDialect>();
  auto tdgModule = emitTDG(tg, ctx);
  assert(tdgModule && "emitTDG should succeed");

  std::map<std::string, mlir::ModuleOp> dfgModules; // empty
  loom::TDGToSSGBuilder builder;
  loom::SSG ssg = builder.build(*tdgModule, dfgModules, ctx);

  if (ssg.numNodes() != 3) {
    std::cerr << "FAIL T6: SSG nodes=" << ssg.numNodes()
              << " (expected 3)\n";
    return false;
  }
  if (ssg.numEdges() != 2) {
    std::cerr << "FAIL T6: SSG edges=" << ssg.numEdges()
              << " (expected 2)\n";
    return false;
  }

  std::cout << "PASS T6: SSG node count\n";
  return true;
}

//===----------------------------------------------------------------------===//
// T7: TDGToSSGBuilder edge data
//===----------------------------------------------------------------------===//
static bool testSSGEdgeData() {
  auto result = makeResult(
      {{"p", KernelTarget::CGRA}, {"c", KernelTarget::CGRA}},
      {{0, 1, "f64", true, std::nullopt}});

  TaskGraph tg = buildTaskGraphFromAnalysis(result);

  mlir::MLIRContext ctx;
  ctx.getOrLoadDialect<loom::tdg::TDGDialect>();
  auto tdgModule = emitTDG(tg, ctx);
  assert(tdgModule && "emitTDG should succeed");

  std::map<std::string, mlir::ModuleOp> dfgModules;
  loom::TDGToSSGBuilder builder;
  loom::SSG ssg = builder.build(*tdgModule, dfgModules, ctx);

  if (ssg.numEdges() != 1) {
    std::cerr << "FAIL T7: SSG edges=" << ssg.numEdges()
              << " (expected 1)\n";
    return false;
  }

  const auto &dep = ssg.edge(0);
  if (dep.producerKernel != "p" || dep.consumerKernel != "c") {
    std::cerr << "FAIL T7: edge endpoints wrong\n";
    return false;
  }

  std::cout << "PASS T7: SSG edge data\n";
  return true;
}

//===----------------------------------------------------------------------===//
// T8: TDGToSSGBuilder handles missing DFG module
//===----------------------------------------------------------------------===//
static bool testSSGMissingDFG() {
  auto result = makeResult(
      {{"has_dfg", KernelTarget::CGRA}, {"no_dfg", KernelTarget::CGRA}},
      {{0, 1, "f32", true, std::nullopt}});

  TaskGraph tg = buildTaskGraphFromAnalysis(result);

  mlir::MLIRContext ctx;
  ctx.getOrLoadDialect<loom::tdg::TDGDialect>();
  auto tdgModule = emitTDG(tg, ctx);
  assert(tdgModule && "emitTDG should succeed");

  // Both kernels have no DFG module.
  std::map<std::string, mlir::ModuleOp> dfgModules;
  loom::TDGToSSGBuilder builder;
  loom::SSG ssg = builder.build(*tdgModule, dfgModules, ctx);

  // Should not crash, and should still have 2 nodes.
  if (ssg.numNodes() != 2) {
    std::cerr << "FAIL T8: SSG nodes=" << ssg.numNodes()
              << " (expected 2)\n";
    return false;
  }

  // Both nodes should have hasDFG=false.
  for (unsigned i = 0; i < ssg.numNodes(); ++i) {
    const auto &node = ssg.node(i);
    if (node.hasDFG) {
      std::cerr << "FAIL T8: node " << node.name
                << " should have hasDFG=false\n";
      return false;
    }
    if (!node.variantSet.empty()) {
      std::cerr << "FAIL T8: node " << node.name
                << " should have empty variantSet\n";
      return false;
    }
  }

  std::cout << "PASS T8: SSG missing DFG handled gracefully\n";
  return true;
}

//===----------------------------------------------------------------------===//
// T9: Provenance sourcePath propagated to kernels
//===----------------------------------------------------------------------===//
static bool testProvenance() {
  auto result = makeResult(
      {{"kern_a", KernelTarget::CGRA}, {"kern_b", KernelTarget::HOST}}, {});

  TaskGraph tg = buildTaskGraphFromAnalysis(result);

  // Each kernel's provenance should carry the sourcePath from the
  // AutoAnalyzeResult ("test.c" is set by makeResult).
  bool ok = true;
  tg.forEachKernel([&](const KernelInfo &ki) {
    if (ki.provenance.sourcePath != "test.c") {
      std::cerr << "FAIL T9: kernel '" << ki.name
                << "' sourcePath='" << ki.provenance.sourcePath
                << "' (expected 'test.c')\n";
      ok = false;
    }
  });

  if (!ok)
    return false;

  std::cout << "PASS T9: provenance sourcePath propagated\n";
  return true;
}

//===----------------------------------------------------------------------===//
// Main
//===----------------------------------------------------------------------===//

int main() {
  unsigned passed = 0;
  unsigned failed = 0;

  auto run = [&](bool (*test)(), const char *name) {
    if (test())
      ++passed;
    else {
      std::cerr << "  FAILED: " << name << "\n";
      ++failed;
    }
  };

  run(testBasicConversion, "T1: basic conversion");
  run(testZeroEdges, "T2: zero edges");
  run(testTargetMapping, "T3: target mapping");
  run(testSizeOfType, "T4: sizeOfType");
  run(testDefaultVariant, "T5: default + unroll variant");
  run(testSSGNodeCount, "T6: SSG node count");
  run(testSSGEdgeData, "T7: SSG edge data");
  run(testSSGMissingDFG, "T8: SSG missing DFG");
  run(testProvenance, "T9: provenance sourcePath");

  std::cout << "\n" << passed << " passed, " << failed << " failed out of "
            << (passed + failed) << " tests\n";

  return failed > 0 ? 1 : 0;
}
