/// SSG parity tests.
///
/// Verifies that the SSG produced from the auto-analyze bridge path has
/// structural parity with the input AutoAnalyzeResult: kernel names, kernel
/// types, edge producer/consumer endpoints, and node counts all match.
///
/// Tests:
/// T1: SSG kernel names match AutoAnalyzeResult callBindings names
/// T2: SSG kernel types reflect CGRA/HOST target mapping
/// T3: Mixed CGRA+HOST pipeline preserves all kernel types
/// T4: Edge ordering preserved -- edges appear in expected order
/// T5: Multi-edge graph edge count parity

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
#include <set>
#include <string>
#include <vector>

using namespace tapestry;

//===----------------------------------------------------------------------===//
// Test helpers
//===----------------------------------------------------------------------===//

static AutoAnalyzeResult makeResult(
    const std::vector<std::pair<std::string, KernelTarget>> &kernels,
    const std::vector<std::tuple<unsigned, unsigned, std::string, bool,
                                 std::optional<uint64_t>>> &edges) {

  AutoAnalyzeResult result;
  result.success = true;
  result.sourcePath = "parity_test.c";
  result.entryFunc = "parity_main";

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

static loom::SSG runPipeline(const AutoAnalyzeResult &result) {
  TaskGraph tg = buildTaskGraphFromAnalysis(result);

  mlir::MLIRContext ctx;
  ctx.getOrLoadDialect<loom::tdg::TDGDialect>();
  auto tdgModule = emitTDG(tg, ctx);
  assert(tdgModule && "emitTDG must succeed");

  std::map<std::string, mlir::ModuleOp> dfgModules;
  loom::TDGToSSGBuilder builder;
  return builder.build(*tdgModule, dfgModules, ctx);
}

//===----------------------------------------------------------------------===//
// T1: SSG kernel names match
//===----------------------------------------------------------------------===//
static bool testKernelNamesParity() {
  auto result = makeResult(
      {{"alpha", KernelTarget::CGRA},
       {"beta", KernelTarget::HOST},
       {"gamma", KernelTarget::CGRA}},
      {{0, 1, "f32", true, std::nullopt},
       {1, 2, "f32", true, std::nullopt}});

  loom::SSG ssg = runPipeline(result);

  // Collect SSG node names.
  std::set<std::string> ssgNames;
  for (unsigned i = 0; i < ssg.numNodes(); ++i)
    ssgNames.insert(ssg.node(i).name);

  // Collect expected names from AutoAnalyzeResult.
  std::set<std::string> expectedNames;
  for (const auto &cb : result.callBindings)
    expectedNames.insert(cb.kernelName);

  if (ssgNames != expectedNames) {
    std::cerr << "FAIL T1: SSG kernel names do not match expected names\n";
    std::cerr << "  SSG names:";
    for (const auto &n : ssgNames)
      std::cerr << " " << n;
    std::cerr << "\n  Expected:";
    for (const auto &n : expectedNames)
      std::cerr << " " << n;
    std::cerr << "\n";
    return false;
  }

  std::cout << "PASS T1: SSG kernel names match\n";
  return true;
}

//===----------------------------------------------------------------------===//
// T2: SSG kernel types reflect CGRA/HOST
//===----------------------------------------------------------------------===//
static bool testKernelTypeParity() {
  auto result = makeResult(
      {{"cgra_kern", KernelTarget::CGRA},
       {"host_kern", KernelTarget::HOST}},
      {{0, 1, "f32", true, std::nullopt}});

  loom::SSG ssg = runPipeline(result);

  // Build name->type map from SSG.
  std::map<std::string, std::string> typeMap;
  for (unsigned i = 0; i < ssg.numNodes(); ++i)
    typeMap[ssg.node(i).name] = ssg.node(i).kernelType;

  if (typeMap["cgra_kern"] != "cgra") {
    std::cerr << "FAIL T2: cgra_kern type='" << typeMap["cgra_kern"]
              << "' (expected 'cgra')\n";
    return false;
  }
  if (typeMap["host_kern"] != "host") {
    std::cerr << "FAIL T2: host_kern type='" << typeMap["host_kern"]
              << "' (expected 'host')\n";
    return false;
  }

  std::cout << "PASS T2: SSG kernel types match\n";
  return true;
}

//===----------------------------------------------------------------------===//
// T3: Mixed CGRA+HOST pipeline
//===----------------------------------------------------------------------===//
static bool testMixedPipeline() {
  auto result = makeResult(
      {{"setup", KernelTarget::HOST},
       {"accel_a", KernelTarget::CGRA},
       {"accel_b", KernelTarget::CGRA},
       {"teardown", KernelTarget::HOST}},
      {{0, 1, "f32", true, std::nullopt},
       {1, 2, "f32", true, std::nullopt},
       {2, 3, "f32", true, std::nullopt}});

  loom::SSG ssg = runPipeline(result);

  if (ssg.numNodes() != 4) {
    std::cerr << "FAIL T3: SSG nodes=" << ssg.numNodes()
              << " (expected 4)\n";
    return false;
  }

  // Verify we have exactly 2 host and 2 cgra nodes.
  unsigned hostCount = 0, cgraCount = 0;
  for (unsigned i = 0; i < ssg.numNodes(); ++i) {
    const auto &kt = ssg.node(i).kernelType;
    if (kt == "host")
      ++hostCount;
    else if (kt == "cgra")
      ++cgraCount;
  }

  if (hostCount != 2) {
    std::cerr << "FAIL T3: host count=" << hostCount << " (expected 2)\n";
    return false;
  }
  if (cgraCount != 2) {
    std::cerr << "FAIL T3: cgra count=" << cgraCount << " (expected 2)\n";
    return false;
  }

  std::cout << "PASS T3: mixed CGRA+HOST pipeline\n";
  return true;
}

//===----------------------------------------------------------------------===//
// T4: Edge ordering preserved
//===----------------------------------------------------------------------===//
static bool testEdgeOrdering() {
  auto result = makeResult(
      {{"src", KernelTarget::CGRA},
       {"mid", KernelTarget::CGRA},
       {"dst", KernelTarget::CGRA}},
      {{0, 1, "f32", true, std::nullopt},
       {1, 2, "i32", true, std::nullopt}});

  loom::SSG ssg = runPipeline(result);

  if (ssg.numEdges() != 2) {
    std::cerr << "FAIL T4: SSG edges=" << ssg.numEdges()
              << " (expected 2)\n";
    return false;
  }

  // First edge: src -> mid
  const auto &e0 = ssg.edge(0);
  if (e0.producerKernel != "src" || e0.consumerKernel != "mid") {
    std::cerr << "FAIL T4: edge 0 endpoints wrong ("
              << e0.producerKernel << " -> " << e0.consumerKernel << ")\n";
    return false;
  }

  // Second edge: mid -> dst
  const auto &e1 = ssg.edge(1);
  if (e1.producerKernel != "mid" || e1.consumerKernel != "dst") {
    std::cerr << "FAIL T4: edge 1 endpoints wrong ("
              << e1.producerKernel << " -> " << e1.consumerKernel << ")\n";
    return false;
  }

  std::cout << "PASS T4: edge ordering preserved\n";
  return true;
}

//===----------------------------------------------------------------------===//
// T5: Multi-edge graph edge count parity
//===----------------------------------------------------------------------===//
static bool testEdgeCountParity() {
  // Fan-out topology: one source feeds 4 consumers.
  auto result = makeResult(
      {{"source", KernelTarget::CGRA},
       {"sink_a", KernelTarget::CGRA},
       {"sink_b", KernelTarget::CGRA},
       {"sink_c", KernelTarget::CGRA},
       {"sink_d", KernelTarget::CGRA}},
      {{0, 1, "f32", true, std::nullopt},
       {0, 2, "f32", true, std::nullopt},
       {0, 3, "f32", true, std::nullopt},
       {0, 4, "f32", true, std::nullopt}});

  loom::SSG ssg = runPipeline(result);

  if (ssg.numNodes() != 5) {
    std::cerr << "FAIL T5: SSG nodes=" << ssg.numNodes()
              << " (expected 5)\n";
    return false;
  }
  if (ssg.numEdges() != 4) {
    std::cerr << "FAIL T5: SSG edges=" << ssg.numEdges()
              << " (expected 4)\n";
    return false;
  }

  // All edges should have "source" as producer.
  for (unsigned i = 0; i < ssg.numEdges(); ++i) {
    if (ssg.edge(i).producerKernel != "source") {
      std::cerr << "FAIL T5: edge " << i << " producer='"
                << ssg.edge(i).producerKernel
                << "' (expected 'source')\n";
      return false;
    }
  }

  std::cout << "PASS T5: multi-edge graph edge count parity\n";
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

  run(testKernelNamesParity, "T1: kernel names parity");
  run(testKernelTypeParity, "T2: kernel type parity");
  run(testMixedPipeline, "T3: mixed pipeline");
  run(testEdgeOrdering, "T4: edge ordering");
  run(testEdgeCountParity, "T5: edge count parity");

  std::cout << "\n" << passed << " passed, " << failed << " failed out of "
            << (passed + failed) << " tests\n";

  return failed > 0 ? 1 : 0;
}
