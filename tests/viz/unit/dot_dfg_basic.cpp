//===-- dot_dfg_basic.cpp - DFG DOT export basic test -------------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Verify basic DFG DOT export: correct graph structure, node styles, edge
// styles per spec-viz-dfg.md.
//
//===----------------------------------------------------------------------===//

#include "TestUtil.h"
#include "loom/Mapper/Graph.h"
#include "loom/Viz/DOTExporter.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"

using namespace loom;

// Helper to set string attribute on a node.
static void setStrAttr(Node *node, mlir::MLIRContext *ctx,
                       llvm::StringRef key, llvm::StringRef value) {
  auto id = mlir::StringAttr::get(ctx, key);
  auto val = mlir::StringAttr::get(ctx, value);
  node->attributes.push_back(mlir::NamedAttribute(id, val));
}

int main() {
  mlir::MLIRContext ctx;

  Graph dfg(&ctx);

  // Create 3 nodes: ModuleInput -> arith.addi -> ModuleOutput.
  auto inNode = std::make_unique<Node>();
  inNode->kind = Node::ModuleInputNode;
  IdIndex inId = dfg.addNode(std::move(inNode));
  setStrAttr(dfg.getNode(inId), &ctx, "op_name", "ModuleInputNode");
  setStrAttr(dfg.getNode(inId), &ctx, "sym_name", "arg0");

  auto opNode = std::make_unique<Node>();
  opNode->kind = Node::OperationNode;
  IdIndex opId = dfg.addNode(std::move(opNode));
  setStrAttr(dfg.getNode(opId), &ctx, "op_name", "arith.addi");
  setStrAttr(dfg.getNode(opId), &ctx, "type_summary", "i32");
  setStrAttr(dfg.getNode(opId), &ctx, "src_loc", "test.cpp:10");

  auto outNode = std::make_unique<Node>();
  outNode->kind = Node::ModuleOutputNode;
  IdIndex outId = dfg.addNode(std::move(outNode));
  setStrAttr(dfg.getNode(outId), &ctx, "op_name", "ModuleOutputNode");
  setStrAttr(dfg.getNode(outId), &ctx, "sym_name", "ret0");

  // Add ports.
  auto inOutPort = std::make_unique<Port>();
  inOutPort->parentNode = inId;
  inOutPort->direction = Port::Output;
  IdIndex p0 = dfg.addPort(std::move(inOutPort));
  dfg.getNode(inId)->outputPorts.push_back(p0);

  auto opInPort = std::make_unique<Port>();
  opInPort->parentNode = opId;
  opInPort->direction = Port::Input;
  IdIndex p1 = dfg.addPort(std::move(opInPort));
  dfg.getNode(opId)->inputPorts.push_back(p1);

  auto opOutPort = std::make_unique<Port>();
  opOutPort->parentNode = opId;
  opOutPort->direction = Port::Output;
  IdIndex p2 = dfg.addPort(std::move(opOutPort));
  dfg.getNode(opId)->outputPorts.push_back(p2);

  auto outInPort = std::make_unique<Port>();
  outInPort->parentNode = outId;
  outInPort->direction = Port::Input;
  IdIndex p3 = dfg.addPort(std::move(outInPort));
  dfg.getNode(outId)->inputPorts.push_back(p3);

  // Add edges: input -> arith.addi -> output.
  {
    auto edge = std::make_unique<Edge>();
    edge->srcPort = p0;
    edge->dstPort = p1;
    IdIndex eid = dfg.addEdge(std::move(edge));
    dfg.getPort(p0)->connectedEdges.push_back(eid);
    dfg.getPort(p1)->connectedEdges.push_back(eid);
  }
  {
    auto edge = std::make_unique<Edge>();
    edge->srcPort = p2;
    edge->dstPort = p3;
    IdIndex eid = dfg.addEdge(std::move(edge));
    dfg.getPort(p2)->connectedEdges.push_back(eid);
    dfg.getPort(p3)->connectedEdges.push_back(eid);
  }

  // Export DFG DOT.
  std::string dot = viz::exportDFGDot(dfg);

  // Verify basic structure.
  TEST_CONTAINS(dot, "digraph DFG");
  TEST_CONTAINS(dot, "rankdir=TB");

  // Verify node IDs.
  TEST_CONTAINS(dot, "sw_0");
  TEST_CONTAINS(dot, "sw_1");
  TEST_CONTAINS(dot, "sw_2");

  // Verify node styles per spec-viz-dfg.md.
  // arith.addi should use box shape and lightblue fill.
  TEST_CONTAINS(dot, "arith.addi");
  TEST_CONTAINS(dot, "lightblue");
  TEST_CONTAINS(dot, "shape=box");

  // ModuleInputNode should use invhouse shape and lightpink fill.
  TEST_CONTAINS(dot, "shape=invhouse");
  TEST_CONTAINS(dot, "lightpink");

  // ModuleOutputNode should use house shape and lightcoral fill.
  TEST_CONTAINS(dot, "shape=house");
  TEST_CONTAINS(dot, "lightcoral");

  // Verify type summary and source location in label.
  TEST_CONTAINS(dot, "i32");
  TEST_CONTAINS(dot, "test.cpp:10");

  // Verify edges exist.
  TEST_CONTAINS(dot, "sw_0 -> sw_1");
  TEST_CONTAINS(dot, "sw_1 -> sw_2");

  // Verify edge styles: solid, black, penwidth=2.0 for data edges.
  TEST_CONTAINS(dot, "style=solid");
  TEST_CONTAINS(dot, "penwidth=2.0");

  return 0;
}
