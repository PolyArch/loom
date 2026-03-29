/// SystemGraph template unit tests: construction, traversal, JSON round-trip,
/// DOT export, and mutation for both SSG and SHG graph types.

#include "loom/Graph/SystemGraphTypes.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <set>
#include <string>
#include <vector>

using namespace loom;

// ---------------------------------------------------------------------------
// Test 1: Empty graph
// ---------------------------------------------------------------------------
static bool testEmptyGraph() {
  SSG g;
  if (g.numNodes() != 0 || g.numEdges() != 0) {
    std::cerr << "FAIL: testEmptyGraph\n";
    return false;
  }
  std::cerr << "PASS: testEmptyGraph\n";
  return true;
}

// ---------------------------------------------------------------------------
// Test 2: Add node
// ---------------------------------------------------------------------------
static bool testAddNode() {
  SSG g;
  KernelNode kn;
  kn.kernelId = "matmul";
  auto id = g.addNode(kn);
  if (g.numNodes() != 1 || g.node(id).kernelId != "matmul") {
    std::cerr << "FAIL: testAddNode\n";
    return false;
  }
  std::cerr << "PASS: testAddNode\n";
  return true;
}

// ---------------------------------------------------------------------------
// Test 3: Add edge
// ---------------------------------------------------------------------------
static bool testAddEdge() {
  SSG g;
  KernelNode a;
  a.kernelId = "A";
  KernelNode b;
  b.kernelId = "B";
  g.addNode(a);
  g.addNode(b);

  DataDependency dep;
  dep.dataVolume = 1024;
  auto eid = g.addEdge(0, 1, dep);

  if (g.numEdges() != 1 || g.edge(eid).dataVolume != 1024) {
    std::cerr << "FAIL: testAddEdge\n";
    return false;
  }
  std::cerr << "PASS: testAddEdge\n";
  return true;
}

// ---------------------------------------------------------------------------
// Test 4: Successors and predecessors on a linear chain A->B->C
// ---------------------------------------------------------------------------
static bool testSuccessorsAndPredecessors() {
  SSG g;
  KernelNode na, nb, nc;
  na.kernelId = "A";
  nb.kernelId = "B";
  nc.kernelId = "C";
  auto idA = g.addNode(na);
  auto idB = g.addNode(nb);
  auto idC = g.addNode(nc);

  DataDependency d1, d2;
  d1.dataVolume = 100;
  d2.dataVolume = 200;
  g.addEdge(idA, idB, d1);
  g.addEdge(idB, idC, d2);

  auto succA = g.successors(idA);
  auto succB = g.successors(idB);
  auto predC = g.predecessors(idC);
  auto predA = g.predecessors(idA);

  if (succA.size() != 1 || succA[0] != idB) {
    std::cerr << "FAIL: testSuccessorsAndPredecessors - succA\n";
    return false;
  }
  if (succB.size() != 1 || succB[0] != idC) {
    std::cerr << "FAIL: testSuccessorsAndPredecessors - succB\n";
    return false;
  }
  if (predC.size() != 1 || predC[0] != idB) {
    std::cerr << "FAIL: testSuccessorsAndPredecessors - predC\n";
    return false;
  }
  if (!predA.empty()) {
    std::cerr << "FAIL: testSuccessorsAndPredecessors - predA\n";
    return false;
  }
  std::cerr << "PASS: testSuccessorsAndPredecessors\n";
  return true;
}

// ---------------------------------------------------------------------------
// Test 5: Diamond traversal A->B, A->C, B->D, C->D
// ---------------------------------------------------------------------------
static bool testDiamondTraversal() {
  SSG g;
  KernelNode na, nb, nc, nd;
  na.kernelId = "A";
  nb.kernelId = "B";
  nc.kernelId = "C";
  nd.kernelId = "D";
  auto idA = g.addNode(na);
  auto idB = g.addNode(nb);
  auto idC = g.addNode(nc);
  auto idD = g.addNode(nd);

  DataDependency d;
  d.dataVolume = 1;
  g.addEdge(idA, idB, d);
  g.addEdge(idA, idC, d);
  g.addEdge(idB, idD, d);
  g.addEdge(idC, idD, d);

  auto succA = g.successors(idA);
  std::set<unsigned> succASet(succA.begin(), succA.end());
  if (succASet.size() != 2 || succASet.count(idB) != 1 ||
      succASet.count(idC) != 1) {
    std::cerr << "FAIL: testDiamondTraversal - succA\n";
    return false;
  }

  auto predD = g.predecessors(idD);
  std::set<unsigned> predDSet(predD.begin(), predD.end());
  if (predDSet.size() != 2 || predDSet.count(idB) != 1 ||
      predDSet.count(idC) != 1) {
    std::cerr << "FAIL: testDiamondTraversal - predD\n";
    return false;
  }

  if (g.outEdges(idA).size() != 2) {
    std::cerr << "FAIL: testDiamondTraversal - outEdges(A)\n";
    return false;
  }
  if (g.inEdges(idD).size() != 2) {
    std::cerr << "FAIL: testDiamondTraversal - inEdges(D)\n";
    return false;
  }

  std::cerr << "PASS: testDiamondTraversal\n";
  return true;
}

// ---------------------------------------------------------------------------
// Test 6: SSG JSON round-trip
// ---------------------------------------------------------------------------
static bool testSSGJsonRoundTrip() {
  SSG g;
  KernelNode n1;
  n1.kernelId = "matmul";
  n1.variantSet = {"tiled", "unrolled"};
  KernelNode n2;
  n2.kernelId = "relu";

  g.addNode(n1);
  g.addNode(n2);

  DataDependency dep;
  dep.producerKernel = "matmul";
  dep.consumerKernel = "relu";
  dep.dataVolume = 4096;
  dep.edgeContractRef = "c0";
  g.addEdge(0, 1, dep);

  auto json = g.toJSON();
  auto g2 = SSG::fromJSON(json);

  if (g2.numNodes() != 2 || g2.numEdges() != 1) {
    std::cerr << "FAIL: testSSGJsonRoundTrip - counts\n";
    return false;
  }
  if (g2.node(0).kernelId != "matmul" || g2.node(1).kernelId != "relu") {
    std::cerr << "FAIL: testSSGJsonRoundTrip - kernelIds\n";
    return false;
  }
  if (g2.node(0).variantSet.size() != 2 ||
      g2.node(0).variantSet.count("tiled") != 1 ||
      g2.node(0).variantSet.count("unrolled") != 1) {
    std::cerr << "FAIL: testSSGJsonRoundTrip - variantSet\n";
    return false;
  }
  if (g2.edge(0).dataVolume != 4096 ||
      !g2.edge(0).edgeContractRef ||
      *g2.edge(0).edgeContractRef != "c0") {
    std::cerr << "FAIL: testSSGJsonRoundTrip - edge data\n";
    return false;
  }

  std::cerr << "PASS: testSSGJsonRoundTrip\n";
  return true;
}

// ---------------------------------------------------------------------------
// Test 7: SHG JSON round-trip
// ---------------------------------------------------------------------------
static bool testSHGJsonRoundTrip() {
  SHG g;
  CoreNode c1;
  c1.coreType = "PE_A";
  c1.peCount = 16;
  c1.fuCount = 8;
  c1.spmBytes = 4096;
  CoreNode c2;
  c2.coreType = "PE_B";
  c2.peCount = 8;
  c2.fuCount = 4;
  c2.spmBytes = 2048;
  g.addNode(c1);
  g.addNode(c2);

  NoCLink link;
  link.srcCore = "PE_A";
  link.dstCore = "PE_B";
  link.bandwidth = 4;
  link.latency = 2;
  g.addEdge(0, 1, link);

  auto json = g.toJSON();
  auto g2 = SHG::fromJSON(json);

  if (g2.numNodes() != 2 || g2.numEdges() != 1) {
    std::cerr << "FAIL: testSHGJsonRoundTrip - counts\n";
    return false;
  }
  if (g2.node(0).coreType != "PE_A" || g2.node(0).peCount != 16) {
    std::cerr << "FAIL: testSHGJsonRoundTrip - node0\n";
    return false;
  }
  if (g2.node(1).coreType != "PE_B" || g2.node(1).peCount != 8) {
    std::cerr << "FAIL: testSHGJsonRoundTrip - node1\n";
    return false;
  }
  if (g2.edge(0).bandwidth != 4 || g2.edge(0).latency != 2) {
    std::cerr << "FAIL: testSHGJsonRoundTrip - edge\n";
    return false;
  }

  std::cerr << "PASS: testSHGJsonRoundTrip\n";
  return true;
}

// ---------------------------------------------------------------------------
// Test 8: JSON round-trip on empty graph
// ---------------------------------------------------------------------------
static bool testJsonRoundTripEmpty() {
  SSG g;
  auto json = g.toJSON();
  auto g2 = SSG::fromJSON(json);

  if (g2.numNodes() != 0 || g2.numEdges() != 0) {
    std::cerr << "FAIL: testJsonRoundTripEmpty\n";
    return false;
  }
  std::cerr << "PASS: testJsonRoundTripEmpty\n";
  return true;
}

// ---------------------------------------------------------------------------
// Test 9: DOT export basic structure (SSG)
// ---------------------------------------------------------------------------
static bool testDotExportBasicStructure() {
  SSG g;
  KernelNode n1;
  n1.kernelId = "matmul";
  KernelNode n2;
  n2.kernelId = "relu";
  g.addNode(n1);
  g.addNode(n2);

  DataDependency dep;
  dep.dataVolume = 512;
  g.addEdge(0, 1, dep);

  std::string dot;
  llvm::raw_string_ostream os(dot);
  g.exportDot(os);
  os.flush();

  if (dot.find("digraph") == std::string::npos) {
    std::cerr << "FAIL: testDotExportBasicStructure - digraph\n";
    return false;
  }
  if (dot.find("matmul") == std::string::npos) {
    std::cerr << "FAIL: testDotExportBasicStructure - matmul\n";
    return false;
  }
  if (dot.find("relu") == std::string::npos) {
    std::cerr << "FAIL: testDotExportBasicStructure - relu\n";
    return false;
  }
  if (dot.find("\"n0\" -> \"n1\"") == std::string::npos) {
    std::cerr << "FAIL: testDotExportBasicStructure - edge\n";
    return false;
  }

  std::cerr << "PASS: testDotExportBasicStructure\n";
  return true;
}

// ---------------------------------------------------------------------------
// Test 10: DOT export for SHG
// ---------------------------------------------------------------------------
static bool testDotExportSHG() {
  SHG g;
  CoreNode c1;
  c1.coreType = "PE_A";
  c1.peCount = 16;
  CoreNode c2;
  c2.coreType = "PE_B";
  c2.peCount = 8;
  CoreNode c3;
  c3.coreType = "MEM";
  c3.peCount = 0;
  g.addNode(c1);
  g.addNode(c2);
  g.addNode(c3);

  NoCLink l1;
  l1.bandwidth = 4;
  l1.latency = 1;
  NoCLink l2;
  l2.bandwidth = 8;
  l2.latency = 2;
  g.addEdge(0, 1, l1);
  g.addEdge(1, 2, l2);

  std::string dot;
  llvm::raw_string_ostream os(dot);
  g.exportDot(os);
  os.flush();

  if (dot.find("digraph") == std::string::npos) {
    std::cerr << "FAIL: testDotExportSHG - digraph\n";
    return false;
  }
  // Should have 3 node declarations
  size_t nodeCount = 0;
  std::string::size_type pos = 0;
  while ((pos = dot.find("[label=", pos)) != std::string::npos) {
    nodeCount++;
    pos++;
  }
  // 3 node labels + 2 edge labels = 5 label occurrences
  // Actually nodes have [label=...], edges also have [label=...]
  // Just check for core type names
  if (dot.find("PE_A") == std::string::npos ||
      dot.find("PE_B") == std::string::npos ||
      dot.find("MEM") == std::string::npos) {
    std::cerr << "FAIL: testDotExportSHG - core types\n";
    return false;
  }
  // Check 2 edge declarations
  size_t edgeCount = 0;
  pos = 0;
  while ((pos = dot.find("->", pos)) != std::string::npos) {
    edgeCount++;
    pos++;
  }
  if (edgeCount != 2) {
    std::cerr << "FAIL: testDotExportSHG - edge count " << edgeCount << "\n";
    return false;
  }

  std::cerr << "PASS: testDotExportSHG\n";
  return true;
}

// ---------------------------------------------------------------------------
// Test 11: Node mutation via non-const access
// ---------------------------------------------------------------------------
static bool testNodeMutation() {
  SSG g;
  KernelNode kn;
  kn.kernelId = "conv";
  auto id = g.addNode(kn);
  g.node(id).variantSet.insert("v2");

  if (g.node(id).variantSet.count("v2") != 1) {
    std::cerr << "FAIL: testNodeMutation\n";
    return false;
  }
  std::cerr << "PASS: testNodeMutation\n";
  return true;
}

// ---------------------------------------------------------------------------
// Test 12: Edge mutation via non-const access
// ---------------------------------------------------------------------------
static bool testEdgeMutation() {
  SSG g;
  KernelNode a, b;
  a.kernelId = "A";
  b.kernelId = "B";
  g.addNode(a);
  g.addNode(b);
  DataDependency dep;
  dep.dataVolume = 100;
  auto eid = g.addEdge(0, 1, dep);
  g.edge(eid).dataVolume = 9999;

  if (g.edge(eid).dataVolume != 9999) {
    std::cerr << "FAIL: testEdgeMutation\n";
    return false;
  }
  std::cerr << "PASS: testEdgeMutation\n";
  return true;
}

// ---------------------------------------------------------------------------
// Test 13: KernelNode standalone toJSON
// ---------------------------------------------------------------------------
static bool testKernelNodeToJson() {
  KernelNode kn;
  kn.kernelId = "fft";
  kn.variantSet = {"radix2", "radix4"};
  KernelNode::ComputeProfile cp;
  cp.estimatedMinII = 4;
  cp.estimatedSPMBytes = 8192;
  cp.estimatedComputeCycles = 100.5;
  kn.computeProfile = cp;

  auto json = kn.toJSON();
  auto *obj = json.getAsObject();
  if (!obj) {
    std::cerr << "FAIL: testKernelNodeToJson - not object\n";
    return false;
  }
  if (!obj->getString("kernelId") ||
      obj->getString("kernelId")->str() != "fft") {
    std::cerr << "FAIL: testKernelNodeToJson - kernelId\n";
    return false;
  }
  if (!obj->getArray("variantSet") ||
      obj->getArray("variantSet")->size() != 2) {
    std::cerr << "FAIL: testKernelNodeToJson - variantSet\n";
    return false;
  }
  if (!obj->getObject("computeProfile")) {
    std::cerr << "FAIL: testKernelNodeToJson - computeProfile\n";
    return false;
  }

  std::cerr << "PASS: testKernelNodeToJson\n";
  return true;
}

// ---------------------------------------------------------------------------
// Test 14: DataDependency standalone toJSON
// ---------------------------------------------------------------------------
static bool testDataDependencyToJson() {
  DataDependency dep;
  dep.producerKernel = "matmul";
  dep.consumerKernel = "relu";
  dep.dataVolume = 2048;
  dep.edgeContractRef = "c1";

  auto json = dep.toJSON();
  auto *obj = json.getAsObject();
  if (!obj) {
    std::cerr << "FAIL: testDataDependencyToJson - not object\n";
    return false;
  }
  if (!obj->getString("producerKernel") ||
      obj->getString("producerKernel")->str() != "matmul") {
    std::cerr << "FAIL: testDataDependencyToJson - producerKernel\n";
    return false;
  }
  if (!obj->getString("consumerKernel") ||
      obj->getString("consumerKernel")->str() != "relu") {
    std::cerr << "FAIL: testDataDependencyToJson - consumerKernel\n";
    return false;
  }
  if (!obj->getInteger("dataVolume") ||
      *obj->getInteger("dataVolume") != 2048) {
    std::cerr << "FAIL: testDataDependencyToJson - dataVolume\n";
    return false;
  }
  if (!obj->getString("edgeContractRef") ||
      obj->getString("edgeContractRef")->str() != "c1") {
    std::cerr << "FAIL: testDataDependencyToJson - edgeContractRef\n";
    return false;
  }

  std::cerr << "PASS: testDataDependencyToJson\n";
  return true;
}

// ---------------------------------------------------------------------------
// Test 15: CoreNode standalone toJSON
// ---------------------------------------------------------------------------
static bool testCoreNodeToJson() {
  CoreNode cn;
  cn.coreType = "PE_X";
  cn.adgRef = "adg_module_pe_x";
  cn.peCount = 32;
  cn.fuCount = 16;
  cn.spmBytes = 65536;

  auto json = cn.toJSON();
  auto *obj = json.getAsObject();
  if (!obj) {
    std::cerr << "FAIL: testCoreNodeToJson - not object\n";
    return false;
  }
  if (!obj->getString("coreType") ||
      obj->getString("coreType")->str() != "PE_X") {
    std::cerr << "FAIL: testCoreNodeToJson - coreType\n";
    return false;
  }
  if (!obj->getString("adgRef") ||
      obj->getString("adgRef")->str() != "adg_module_pe_x") {
    std::cerr << "FAIL: testCoreNodeToJson - adgRef\n";
    return false;
  }
  if (!obj->getInteger("peCount") || *obj->getInteger("peCount") != 32) {
    std::cerr << "FAIL: testCoreNodeToJson - peCount\n";
    return false;
  }
  if (!obj->getInteger("fuCount") || *obj->getInteger("fuCount") != 16) {
    std::cerr << "FAIL: testCoreNodeToJson - fuCount\n";
    return false;
  }
  if (!obj->getInteger("spmBytes") || *obj->getInteger("spmBytes") != 65536) {
    std::cerr << "FAIL: testCoreNodeToJson - spmBytes\n";
    return false;
  }

  std::cerr << "PASS: testCoreNodeToJson\n";
  return true;
}

// ---------------------------------------------------------------------------
// Test 16: NoCLink standalone toJSON
// ---------------------------------------------------------------------------
static bool testNoCLinkToJson() {
  NoCLink link;
  link.srcCore = "core0";
  link.dstCore = "core1";
  link.bandwidth = 16;
  link.latency = 3;

  auto json = link.toJSON();
  auto *obj = json.getAsObject();
  if (!obj) {
    std::cerr << "FAIL: testNoCLinkToJson - not object\n";
    return false;
  }
  if (!obj->getString("srcCore") ||
      obj->getString("srcCore")->str() != "core0") {
    std::cerr << "FAIL: testNoCLinkToJson - srcCore\n";
    return false;
  }
  if (!obj->getString("dstCore") ||
      obj->getString("dstCore")->str() != "core1") {
    std::cerr << "FAIL: testNoCLinkToJson - dstCore\n";
    return false;
  }
  if (!obj->getInteger("bandwidth") ||
      *obj->getInteger("bandwidth") != 16) {
    std::cerr << "FAIL: testNoCLinkToJson - bandwidth\n";
    return false;
  }
  if (!obj->getInteger("latency") || *obj->getInteger("latency") != 3) {
    std::cerr << "FAIL: testNoCLinkToJson - latency\n";
    return false;
  }

  std::cerr << "PASS: testNoCLinkToJson\n";
  return true;
}

// ---------------------------------------------------------------------------
// Test 17: Large graph (100 nodes, 200 edges) with JSON round-trip
// ---------------------------------------------------------------------------
static bool testLargeGraph() {
  SSG g;
  for (unsigned i = 0; i < 100; ++i) {
    KernelNode kn;
    kn.kernelId = "k" + std::to_string(i);
    g.addNode(kn);
  }

  // Deterministic pseudo-random edges using a simple LCG
  unsigned seed = 42;
  for (unsigned i = 0; i < 200; ++i) {
    seed = seed * 1103515245 + 12345;
    unsigned src = (seed >> 16) % 100;
    seed = seed * 1103515245 + 12345;
    unsigned dst = (seed >> 16) % 100;
    DataDependency dep;
    dep.dataVolume = static_cast<uint64_t>(i * 10);
    g.addEdge(src, dst, dep);
  }

  if (g.numNodes() != 100 || g.numEdges() != 200) {
    std::cerr << "FAIL: testLargeGraph - counts\n";
    return false;
  }

  auto json = g.toJSON();
  auto g2 = SSG::fromJSON(json);

  if (g2.numNodes() != 100 || g2.numEdges() != 200) {
    std::cerr << "FAIL: testLargeGraph - round-trip counts\n";
    return false;
  }

  // Spot-check a few nodes and edges
  for (unsigned i = 0; i < 100; ++i) {
    if (g2.node(i).kernelId != "k" + std::to_string(i)) {
      std::cerr << "FAIL: testLargeGraph - node " << i << " kernelId\n";
      return false;
    }
  }
  for (unsigned i = 0; i < 200; ++i) {
    if (g2.edge(i).dataVolume != static_cast<uint64_t>(i * 10)) {
      std::cerr << "FAIL: testLargeGraph - edge " << i << " dataVolume\n";
      return false;
    }
  }

  std::cerr << "PASS: testLargeGraph\n";
  return true;
}

// ---------------------------------------------------------------------------
// Test 18: Self-loop edge
// ---------------------------------------------------------------------------
static bool testSelfLoop() {
  SSG g;
  KernelNode kn;
  kn.kernelId = "loop";
  g.addNode(kn);

  DataDependency dep;
  dep.dataVolume = 256;
  g.addEdge(0, 0, dep);

  auto succs = g.successors(0);
  if (succs.size() != 1 || succs[0] != 0) {
    std::cerr << "FAIL: testSelfLoop - successors\n";
    return false;
  }
  auto preds = g.predecessors(0);
  if (preds.size() != 1 || preds[0] != 0) {
    std::cerr << "FAIL: testSelfLoop - predecessors\n";
    return false;
  }

  // JSON round-trip
  auto json = g.toJSON();
  auto g2 = SSG::fromJSON(json);
  if (g2.numNodes() != 1 || g2.numEdges() != 1) {
    std::cerr << "FAIL: testSelfLoop - round-trip counts\n";
    return false;
  }
  if (g2.edgeSrc(0) != 0 || g2.edgeDst(0) != 0) {
    std::cerr << "FAIL: testSelfLoop - round-trip endpoints\n";
    return false;
  }

  std::cerr << "PASS: testSelfLoop\n";
  return true;
}

// ---------------------------------------------------------------------------
// Test 19: DOT label escaping for special characters
// ---------------------------------------------------------------------------
static bool testDotLabelEscaping() {
  SSG g;
  KernelNode kn;
  kn.kernelId = "node\"with<special>&chars";
  g.addNode(kn);

  std::string dot;
  llvm::raw_string_ostream os(dot);
  g.exportDot(os);
  os.flush();

  // The raw " should be escaped to \"
  // The < should be escaped to &lt;
  // The & before 'chars' should be escaped to &amp;
  if (dot.find("\\\"") == std::string::npos) {
    std::cerr << "FAIL: testDotLabelEscaping - quote not escaped\n";
    return false;
  }
  if (dot.find("&lt;") == std::string::npos) {
    std::cerr << "FAIL: testDotLabelEscaping - angle bracket not escaped\n";
    return false;
  }
  if (dot.find("&amp;") == std::string::npos) {
    std::cerr << "FAIL: testDotLabelEscaping - ampersand not escaped\n";
    return false;
  }
  // The original unescaped characters should NOT appear unescaped
  // (Skip checking raw < and & since they appear as parts of &lt; and &amp;)

  std::cerr << "PASS: testDotLabelEscaping\n";
  return true;
}

// ---------------------------------------------------------------------------
// Main test runner
// ---------------------------------------------------------------------------
int main() {
  int passed = 0;
  int total = 0;

  auto run = [&](bool (*test)()) {
    total++;
    if (test())
      passed++;
  };

  run(testEmptyGraph);
  run(testAddNode);
  run(testAddEdge);
  run(testSuccessorsAndPredecessors);
  run(testDiamondTraversal);
  run(testSSGJsonRoundTrip);
  run(testSHGJsonRoundTrip);
  run(testJsonRoundTripEmpty);
  run(testDotExportBasicStructure);
  run(testDotExportSHG);
  run(testNodeMutation);
  run(testEdgeMutation);
  run(testKernelNodeToJson);
  run(testDataDependencyToJson);
  run(testCoreNodeToJson);
  run(testNoCLinkToJson);
  run(testLargeGraph);
  run(testSelfLoop);
  run(testDotLabelEscaping);

  std::cerr << "\nResults: " << passed << "/" << total << " tests passed\n";
  return (passed == total) ? 0 : 1;
}
