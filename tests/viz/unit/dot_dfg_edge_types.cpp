//===-- dot_dfg_edge_types.cpp - DFG edge style verification ------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Verify DFG DOT export uses correct styles for data vs control edges
// as specified in spec-viz-dfg.md.
//
//===----------------------------------------------------------------------===//

#include "TestUtil.h"
#include "loom/Mapper/Graph.h"
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

static void setEdgeStrAttr(Edge *edge, mlir::MLIRContext *ctx,
                           llvm::StringRef key, llvm::StringRef value) {
  auto id = mlir::StringAttr::get(ctx, key);
  auto val = mlir::StringAttr::get(ctx, value);
  edge->attributes.push_back(mlir::NamedAttribute(id, val));
}

int main() {
  mlir::MLIRContext ctx;

  Graph dfg(&ctx);

  // Two nodes.
  auto n1 = std::make_unique<Node>();
  n1->kind = Node::OperationNode;
  IdIndex nid1 = dfg.addNode(std::move(n1));
  setStrAttr(dfg.getNode(nid1), &ctx, "op_name", "arith.addi");

  auto n2 = std::make_unique<Node>();
  n2->kind = Node::OperationNode;
  IdIndex nid2 = dfg.addNode(std::move(n2));
  setStrAttr(dfg.getNode(nid2), &ctx, "op_name", "arith.muli");

  auto n3 = std::make_unique<Node>();
  n3->kind = Node::OperationNode;
  IdIndex nid3 = dfg.addNode(std::move(n3));
  setStrAttr(dfg.getNode(nid3), &ctx, "op_name", "handshake.cond_br");

  // Ports: n1 out -> n2 in (data edge), n1 out -> n3 in (control edge).
  auto p1out = std::make_unique<Port>();
  p1out->parentNode = nid1;
  p1out->direction = Port::Output;
  IdIndex pid1o = dfg.addPort(std::move(p1out));
  dfg.getNode(nid1)->outputPorts.push_back(pid1o);

  auto p2in = std::make_unique<Port>();
  p2in->parentNode = nid2;
  p2in->direction = Port::Input;
  IdIndex pid2i = dfg.addPort(std::move(p2in));
  dfg.getNode(nid2)->inputPorts.push_back(pid2i);

  auto p1out2 = std::make_unique<Port>();
  p1out2->parentNode = nid1;
  p1out2->direction = Port::Output;
  IdIndex pid1o2 = dfg.addPort(std::move(p1out2));
  dfg.getNode(nid1)->outputPorts.push_back(pid1o2);

  auto p3in = std::make_unique<Port>();
  p3in->parentNode = nid3;
  p3in->direction = Port::Input;
  IdIndex pid3i = dfg.addPort(std::move(p3in));
  dfg.getNode(nid3)->inputPorts.push_back(pid3i);

  // Data edge.
  {
    auto edge = std::make_unique<Edge>();
    edge->srcPort = pid1o;
    edge->dstPort = pid2i;
    IdIndex eid = dfg.addEdge(std::move(edge));
    dfg.getPort(pid1o)->connectedEdges.push_back(eid);
    dfg.getPort(pid2i)->connectedEdges.push_back(eid);
  }

  // Control edge.
  {
    auto edge = std::make_unique<Edge>();
    edge->srcPort = pid1o2;
    edge->dstPort = pid3i;
    IdIndex eid = dfg.addEdge(std::move(edge));
    dfg.getPort(pid1o2)->connectedEdges.push_back(eid);
    dfg.getPort(pid3i)->connectedEdges.push_back(eid);
    setEdgeStrAttr(dfg.getEdge(eid), &ctx, "edge_type", "control");
  }

  std::string dot = viz::exportDFGDot(dfg);

  // Data edge: solid, black, penwidth=2.0.
  TEST_CONTAINS(dot, "style=solid");
  TEST_CONTAINS(dot, "color=\"black\"");
  TEST_CONTAINS(dot, "penwidth=2.0");

  // Control edge: dashed, gray, penwidth=1.0.
  TEST_CONTAINS(dot, "style=dashed");
  TEST_CONTAINS(dot, "color=\"gray\"");
  TEST_CONTAINS(dot, "penwidth=1.0");

  return 0;
}
