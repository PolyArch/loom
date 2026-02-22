//===-- dot_mapped_basic.cpp - Mapped DOT export basic test -------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Verify mapped DOT export: overlay mode with mapped/unmapped coloring,
// routing paths, and side-by-side export.
//
//===----------------------------------------------------------------------===//

#include "TestUtil.h"
#include "loom/Mapper/Graph.h"
#include "loom/Mapper/MappingState.h"
#include "loom/Viz/DOTExporter.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"

using namespace loom;

static void setStrAttr(Node *node, mlir::MLIRContext *ctx,
                       llvm::StringRef key, llvm::StringRef value) {
  auto id = mlir::StringAttr::get(ctx, key);
  auto val = mlir::StringAttr::get(ctx, value);
  node->attributes.push_back(mlir::NamedAttribute(id, val));
}

int main() {
  mlir::MLIRContext ctx;

  // Build a minimal DFG: input -> addi -> output.
  Graph dfg(&ctx);
  {
    auto n = std::make_unique<Node>();
    n->kind = Node::ModuleInputNode;
    dfg.addNode(std::move(n));
    setStrAttr(dfg.getNode(0), &ctx, "op_name", "ModuleInputNode");

    auto n2 = std::make_unique<Node>();
    n2->kind = Node::OperationNode;
    dfg.addNode(std::move(n2));
    setStrAttr(dfg.getNode(1), &ctx, "op_name", "arith.addi");

    auto n3 = std::make_unique<Node>();
    n3->kind = Node::ModuleOutputNode;
    dfg.addNode(std::move(n3));
    setStrAttr(dfg.getNode(2), &ctx, "op_name", "ModuleOutputNode");

    // Ports and edges.
    auto mkPort = [&](IdIndex parent, Port::Direction dir) -> IdIndex {
      auto p = std::make_unique<Port>();
      p->parentNode = parent;
      p->direction = dir;
      IdIndex pid = dfg.addPort(std::move(p));
      if (dir == Port::Input)
        dfg.getNode(parent)->inputPorts.push_back(pid);
      else
        dfg.getNode(parent)->outputPorts.push_back(pid);
      return pid;
    };
    IdIndex p0 = mkPort(0, Port::Output);
    IdIndex p1 = mkPort(1, Port::Input);
    IdIndex p2 = mkPort(1, Port::Output);
    IdIndex p3 = mkPort(2, Port::Input);

    auto mkEdge = [&](IdIndex src, IdIndex dst) {
      auto e = std::make_unique<Edge>();
      e->srcPort = src;
      e->dstPort = dst;
      IdIndex eid = dfg.addEdge(std::move(e));
      dfg.getPort(src)->connectedEdges.push_back(eid);
      dfg.getPort(dst)->connectedEdges.push_back(eid);
    };
    mkEdge(p0, p1);
    mkEdge(p2, p3);
  }

  // Build a minimal ADG: in -> pe0 -> sw -> pe1(unmapped) -> out.
  Graph adg(&ctx);
  {
    auto n = std::make_unique<Node>();
    n->kind = Node::ModuleInputNode;
    adg.addNode(std::move(n));
    setStrAttr(adg.getNode(0), &ctx, "op_name", "ModuleInputNode");

    auto n2 = std::make_unique<Node>();
    n2->kind = Node::OperationNode;
    adg.addNode(std::move(n2));
    setStrAttr(adg.getNode(1), &ctx, "op_name", "fabric.pe");
    setStrAttr(adg.getNode(1), &ctx, "sym_name", "pe_0");

    auto n3 = std::make_unique<Node>();
    n3->kind = Node::OperationNode;
    adg.addNode(std::move(n3));
    setStrAttr(adg.getNode(2), &ctx, "op_name", "fabric.switch");
    setStrAttr(adg.getNode(2), &ctx, "sym_name", "sw_0");

    auto n4 = std::make_unique<Node>();
    n4->kind = Node::OperationNode;
    adg.addNode(std::move(n4));
    setStrAttr(adg.getNode(3), &ctx, "op_name", "fabric.pe");
    setStrAttr(adg.getNode(3), &ctx, "sym_name", "pe_1");

    auto n5 = std::make_unique<Node>();
    n5->kind = Node::ModuleOutputNode;
    adg.addNode(std::move(n5));
    setStrAttr(adg.getNode(4), &ctx, "op_name", "ModuleOutputNode");

    auto mkPort = [&](IdIndex parent, Port::Direction dir) -> IdIndex {
      auto p = std::make_unique<Port>();
      p->parentNode = parent;
      p->direction = dir;
      IdIndex pid = adg.addPort(std::move(p));
      if (dir == Port::Input)
        adg.getNode(parent)->inputPorts.push_back(pid);
      else
        adg.getNode(parent)->outputPorts.push_back(pid);
      return pid;
    };

    IdIndex ap0 = mkPort(0, Port::Output);
    IdIndex ap1 = mkPort(1, Port::Input);
    IdIndex ap2 = mkPort(1, Port::Output);
    IdIndex ap3 = mkPort(2, Port::Input);
    IdIndex ap4 = mkPort(2, Port::Output);
    IdIndex ap5 = mkPort(3, Port::Input);
    IdIndex ap6 = mkPort(3, Port::Output);
    IdIndex ap7 = mkPort(4, Port::Input);

    auto mkEdge = [&](IdIndex src, IdIndex dst) {
      auto e = std::make_unique<Edge>();
      e->srcPort = src;
      e->dstPort = dst;
      IdIndex eid = adg.addEdge(std::move(e));
      adg.getPort(src)->connectedEdges.push_back(eid);
      adg.getPort(dst)->connectedEdges.push_back(eid);
    };
    mkEdge(ap0, ap1);  // edge 0: in -> pe0
    mkEdge(ap2, ap3);  // edge 1: pe0 -> sw
    mkEdge(ap4, ap5);  // edge 2: sw -> pe1
    mkEdge(ap6, ap7);  // edge 3: pe1 -> out
  }

  // Build mapping state.
  MappingState state;
  state.init(dfg, adg);

  // Map SW node 0 (input) -> HW node 0 (input).
  state.swNodeToHwNode[0] = 0;
  state.hwNodeToSwNodes[0].push_back(0);

  // Map SW node 1 (arith.addi) -> HW node 1 (pe_0).
  state.swNodeToHwNode[1] = 1;
  state.hwNodeToSwNodes[1].push_back(1);

  // Map SW node 2 (output) -> HW node 4 (output).
  state.swNodeToHwNode[2] = 4;
  state.hwNodeToSwNodes[4].push_back(2);

  // Map SW edge 0: port-sequence [outPort, inPort] for in -> pe0.
  state.swEdgeToHwPaths[0] = {0, 1};
  // Map SW edge 1: port-sequence pairs for pe0 -> sw -> pe1 -> out.
  state.swEdgeToHwPaths[1] = {2, 3, 4, 5, 6, 7};

  // Test overlay DOT.
  {
    std::string dot = viz::exportMappedOverlayDot(dfg, adg, state);

    TEST_CONTAINS(dot, "digraph MappedOverlay");
    TEST_CONTAINS(dot, "rankdir=LR");

    // Mapped pe_0 should have lightblue (arith dialect color).
    TEST_CONTAINS(dot, "lightblue");

    // Unmapped pe_1 (HW node 3) should have white fill and dashed style.
    TEST_CONTAINS(dot, "white");
    TEST_CONTAINS(dot, "dashed");

    // Mapped label should include SW op name.
    TEST_CONTAINS(dot, "arith.addi");

    // Route overlay edges should exist with route colors.
    TEST_CONTAINS(dot, "penwidth=3.0");
    TEST_CONTAINS(dot, "#e6194b");
  }

  // Test side-by-side DFG DOT.
  {
    std::string dot = viz::exportMappedDFGDot(dfg, state);

    TEST_CONTAINS(dot, "digraph MappedDFG");
    TEST_CONTAINS(dot, "rankdir=TB");
    TEST_CONTAINS(dot, "sw_0");
    TEST_CONTAINS(dot, "sw_1");
    TEST_CONTAINS(dot, "sw_2");
  }

  // Test side-by-side ADG DOT.
  {
    std::string dot = viz::exportMappedADGDot(dfg, adg, state);

    TEST_CONTAINS(dot, "digraph MappedADG");
    TEST_CONTAINS(dot, "hw_0");
    TEST_CONTAINS(dot, "hw_1");
    // Mapped node labels include SW op.
    TEST_CONTAINS(dot, "arith.addi");
  }

  return 0;
}
