/// SSG auto-analyze pipeline tests.
///
/// Verifies the end-to-end path:
///   AutoAnalyzeResult -> buildTaskGraphFromAnalysis -> emitTDG -> TDGToSSGBuilder
/// produces a valid SSG with correct node/edge structure.
///
/// Tests:
/// T1: Linear 3-kernel pipeline produces 3 SSG nodes and 2 SSG edges
/// T2: ComputeProfileEstimate fields propagate through CallSiteBinding
/// T3: Diamond DAG (4 kernels, 4 edges) produces correct SSG topology
/// T4: Single kernel with no edges produces 1-node SSG
/// T5: Edge endpoint names in SSG match kernel names from AutoAnalyzeResult

#include "tapestry/auto_analyze.h"
#include "tapestry/task_graph.h"
#include "tapestry/tdg_emitter.h"

#include "loom/Dialect/TDG/TDGDialect.h"
#include "loom/SystemCompiler/TDGToSSGBuilder.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

#include <cassert>
#include <iostream>
#include <map>
#include <optional>
#include <string>
#include <vector>

using namespace tapestry;

//===----------------------------------------------------------------------===//
// Test helpers
//===----------------------------------------------------------------------===//

/// Build a synthetic AutoAnalyzeResult for testing (no real compilation).
static AutoAnalyzeResult makeSyntheticResult(
    const std::vector<std::pair<std::string, KernelTarget>> &kernels,
    const std::vector<std::tuple<unsigned, unsigned, std::string, bool,
                                 std::optional<uint64_t>>> &edges) {

  AutoAnalyzeResult result;
  result.success = true;
  result.sourcePath = "synthetic.c";
  result.entryFunc = "pipeline_main";

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

/// Run the full pipeline: AutoAnalyzeResult -> TaskGraph -> TDG MLIR -> SSG.
static loom::SSG runPipeline(const AutoAnalyzeResult &result) {
  TaskGraph tg = buildTaskGraphFromAnalysis(result);

  mlir::MLIRContext ctx;
  ctx.getOrLoadDialect<loom::tdg::TDGDialect>();
  auto tdgModule = emitTDG(tg, ctx);
  assert(tdgModule && "emitTDG must succeed");

  std::map<std::string, mlir::ModuleOp> dfgModules; // empty
  loom::TDGToSSGBuilder builder;
  return builder.build(*tdgModule, dfgModules, ctx);
}

//===----------------------------------------------------------------------===//
// T1: Linear 3-kernel pipeline
//===----------------------------------------------------------------------===//
static bool testLinearPipeline() {
  auto result = makeSyntheticResult(
      {{"read_input", KernelTarget::CGRA},
       {"compute", KernelTarget::CGRA},
       {"write_output", KernelTarget::CGRA}},
      {{0, 1, "f32", true, std::nullopt},
       {1, 2, "f32", true, std::nullopt}});

  loom::SSG ssg = runPipeline(result);

  if (ssg.numNodes() != 3) {
    std::cerr << "FAIL T1: SSG nodes=" << ssg.numNodes()
              << " (expected 3)\n";
    return false;
  }
  if (ssg.numEdges() != 2) {
    std::cerr << "FAIL T1: SSG edges=" << ssg.numEdges()
              << " (expected 2)\n";
    return false;
  }

  std::cout << "PASS T1: linear 3-kernel pipeline\n";
  return true;
}

//===----------------------------------------------------------------------===//
// T2: ComputeProfileEstimate field propagation
//===----------------------------------------------------------------------===//
static bool testComputeEstimateFields() {
  ComputeProfileEstimate est;
  est.opCount = 42;
  est.memoryAccessCount = 10;
  est.hasLoops = true;
  est.hasExternalCalls = false;

  // Verify struct fields are independently accessible.
  if (est.opCount != 42) {
    std::cerr << "FAIL T2: opCount=" << est.opCount << " (expected 42)\n";
    return false;
  }
  if (est.memoryAccessCount != 10) {
    std::cerr << "FAIL T2: memoryAccessCount=" << est.memoryAccessCount
              << " (expected 10)\n";
    return false;
  }
  if (!est.hasLoops) {
    std::cerr << "FAIL T2: hasLoops should be true\n";
    return false;
  }
  if (est.hasExternalCalls) {
    std::cerr << "FAIL T2: hasExternalCalls should be false\n";
    return false;
  }

  // Verify it can be attached to a CallSiteBinding.
  CallSiteBinding binding;
  binding.kernelName = "test_kernel";
  binding.computeEstimate = est;
  if (binding.computeEstimate.opCount != 42) {
    std::cerr << "FAIL T2: binding.computeEstimate.opCount mismatch\n";
    return false;
  }

  // Verify default initialization (all zeros/false).
  ComputeProfileEstimate defaultEst;
  if (defaultEst.opCount != 0 || defaultEst.memoryAccessCount != 0 ||
      defaultEst.hasLoops || defaultEst.hasExternalCalls) {
    std::cerr << "FAIL T2: default ComputeProfileEstimate not zero-init\n";
    return false;
  }

  std::cout << "PASS T2: ComputeProfileEstimate field propagation\n";
  return true;
}

//===----------------------------------------------------------------------===//
// T3: Diamond DAG (4 kernels, 4 edges)
//===----------------------------------------------------------------------===//
static bool testDiamondDAG() {
  //   A
  //  / \
  // B   C
  //  \ /
  //   D
  auto result = makeSyntheticResult(
      {{"A", KernelTarget::CGRA},
       {"B", KernelTarget::CGRA},
       {"C", KernelTarget::CGRA},
       {"D", KernelTarget::CGRA}},
      {{0, 1, "f32", true, std::nullopt},
       {0, 2, "f32", true, std::nullopt},
       {1, 3, "f32", true, std::nullopt},
       {2, 3, "f32", true, std::nullopt}});

  loom::SSG ssg = runPipeline(result);

  if (ssg.numNodes() != 4) {
    std::cerr << "FAIL T3: SSG nodes=" << ssg.numNodes()
              << " (expected 4)\n";
    return false;
  }
  if (ssg.numEdges() != 4) {
    std::cerr << "FAIL T3: SSG edges=" << ssg.numEdges()
              << " (expected 4)\n";
    return false;
  }

  std::cout << "PASS T3: diamond DAG topology\n";
  return true;
}

//===----------------------------------------------------------------------===//
// T4: Single kernel, no edges
//===----------------------------------------------------------------------===//
static bool testSingleKernel() {
  auto result = makeSyntheticResult(
      {{"standalone", KernelTarget::CGRA}}, {});

  loom::SSG ssg = runPipeline(result);

  if (ssg.numNodes() != 1) {
    std::cerr << "FAIL T4: SSG nodes=" << ssg.numNodes()
              << " (expected 1)\n";
    return false;
  }
  if (ssg.numEdges() != 0) {
    std::cerr << "FAIL T4: SSG edges=" << ssg.numEdges()
              << " (expected 0)\n";
    return false;
  }

  std::cout << "PASS T4: single kernel, no edges\n";
  return true;
}

//===----------------------------------------------------------------------===//
// T5: Edge endpoint names match
//===----------------------------------------------------------------------===//
static bool testEdgeEndpointNames() {
  auto result = makeSyntheticResult(
      {{"producer_k", KernelTarget::CGRA},
       {"consumer_k", KernelTarget::CGRA}},
      {{0, 1, "i32", true, std::optional<uint64_t>(128)}});

  loom::SSG ssg = runPipeline(result);

  if (ssg.numEdges() != 1) {
    std::cerr << "FAIL T5: SSG edges=" << ssg.numEdges()
              << " (expected 1)\n";
    return false;
  }

  const auto &dep = ssg.edge(0);
  if (dep.producerKernel != "producer_k") {
    std::cerr << "FAIL T5: producerKernel='" << dep.producerKernel
              << "' (expected 'producer_k')\n";
    return false;
  }
  if (dep.consumerKernel != "consumer_k") {
    std::cerr << "FAIL T5: consumerKernel='" << dep.consumerKernel
              << "' (expected 'consumer_k')\n";
    return false;
  }

  std::cout << "PASS T5: edge endpoint names match\n";
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

  run(testLinearPipeline, "T1: linear pipeline");
  run(testComputeEstimateFields, "T2: compute estimate fields");
  run(testDiamondDAG, "T3: diamond DAG");
  run(testSingleKernel, "T4: single kernel");
  run(testEdgeEndpointNames, "T5: edge endpoint names");

  std::cout << "\n" << passed << " passed, " << failed << " failed out of "
            << (passed + failed) << " tests\n";

  return failed > 0 ? 1 : 0;
}
