//===-- dot_dfg_node_styles.cpp - DFG node style verification -----*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Verify DFG DOT export uses correct shapes and colors for each operation
// type as specified in spec-viz-dfg.md.
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

static IdIndex addOpNode(Graph &g, mlir::MLIRContext *ctx,
                         const char *opName) {
  auto node = std::make_unique<Node>();
  node->kind = Node::OperationNode;
  IdIndex nid = g.addNode(std::move(node));
  setStrAttr(g.getNode(nid), ctx, "op_name", opName);
  return nid;
}

int main() {
  mlir::MLIRContext ctx;

  Graph dfg(&ctx);

  // Add various operation types.
  addOpNode(dfg, &ctx, "arith.addi");           // box, lightblue
  addOpNode(dfg, &ctx, "handshake.constant");    // ellipse, gold
  addOpNode(dfg, &ctx, "handshake.cond_br");     // diamond, lightyellow
  addOpNode(dfg, &ctx, "handshake.mux");         // invtriangle, lightyellow
  addOpNode(dfg, &ctx, "handshake.join");        // triangle, lightyellow
  addOpNode(dfg, &ctx, "handshake.load");        // box, skyblue
  addOpNode(dfg, &ctx, "handshake.store");       // box, lightsalmon
  addOpNode(dfg, &ctx, "handshake.memory");      // cylinder, skyblue
  addOpNode(dfg, &ctx, "handshake.extmemory");   // hexagon, gold
  addOpNode(dfg, &ctx, "dataflow.carry");        // octagon, lightgreen
  addOpNode(dfg, &ctx, "dataflow.gate");         // octagon, palegreen
  addOpNode(dfg, &ctx, "dataflow.stream");       // doubleoctagon, lightgreen
  addOpNode(dfg, &ctx, "math.sqrt");             // box, plum

  std::string dot = viz::exportDFGDot(dfg);

  // Verify shapes present.
  TEST_CONTAINS(dot, "shape=box");
  TEST_CONTAINS(dot, "shape=ellipse");
  TEST_CONTAINS(dot, "shape=diamond");
  TEST_CONTAINS(dot, "shape=invtriangle");
  TEST_CONTAINS(dot, "shape=triangle");
  TEST_CONTAINS(dot, "shape=cylinder");
  TEST_CONTAINS(dot, "shape=hexagon");
  TEST_CONTAINS(dot, "shape=octagon");
  TEST_CONTAINS(dot, "shape=doubleoctagon");

  // Verify colors present.
  TEST_CONTAINS(dot, "lightblue");
  TEST_CONTAINS(dot, "gold");
  TEST_CONTAINS(dot, "lightyellow");
  TEST_CONTAINS(dot, "skyblue");
  TEST_CONTAINS(dot, "lightsalmon");
  TEST_CONTAINS(dot, "lightgreen");
  TEST_CONTAINS(dot, "palegreen");
  TEST_CONTAINS(dot, "plum");

  return 0;
}
