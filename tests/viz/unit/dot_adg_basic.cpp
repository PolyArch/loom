//===-- dot_adg_basic.cpp - ADG DOT export basic test -------------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Verify basic ADG DOT export: correct graph direction, node shapes/colors
// per spec-viz-adg.md for both Structure and Detailed modes.
//
//===----------------------------------------------------------------------===//

#include "TestUtil.h"
#include "loom/Mapper/Graph.h"
#include "loom/Viz/DOTExporter.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"

using namespace loom;

static void setStrAttr(Node *node, mlir::MLIRContext *ctx,
                       llvm::StringRef key, llvm::StringRef value) {
  auto id = mlir::StringAttr::get(ctx, key);
  auto val = mlir::StringAttr::get(ctx, value);
  node->attributes.push_back(mlir::NamedAttribute(id, val));
}

static void setIntAttr(Node *node, mlir::MLIRContext *ctx,
                       llvm::StringRef key, int64_t value) {
  auto id = mlir::StringAttr::get(ctx, key);
  auto val = mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64), value);
  node->attributes.push_back(mlir::NamedAttribute(id, val));
}

int main() {
  mlir::MLIRContext ctx;

  Graph adg(&ctx);

  // Create a simple ADG: Input -> PE -> Switch -> Output.
  auto inNode = std::make_unique<Node>();
  inNode->kind = Node::ModuleInputNode;
  IdIndex inId = adg.addNode(std::move(inNode));
  setStrAttr(adg.getNode(inId), &ctx, "op_name", "ModuleInputNode");

  auto peNode = std::make_unique<Node>();
  peNode->kind = Node::OperationNode;
  IdIndex peId = adg.addNode(std::move(peNode));
  setStrAttr(adg.getNode(peId), &ctx, "op_name", "fabric.pe");
  setStrAttr(adg.getNode(peId), &ctx, "sym_name", "pe_0");

  auto swNode = std::make_unique<Node>();
  swNode->kind = Node::OperationNode;
  IdIndex swId = adg.addNode(std::move(swNode));
  setStrAttr(adg.getNode(swId), &ctx, "op_name", "fabric.switch");
  setStrAttr(adg.getNode(swId), &ctx, "sym_name", "sw_0");

  auto tpeNode = std::make_unique<Node>();
  tpeNode->kind = Node::OperationNode;
  IdIndex tpeId = adg.addNode(std::move(tpeNode));
  setStrAttr(adg.getNode(tpeId), &ctx, "op_name", "fabric.temporal_pe");
  setStrAttr(adg.getNode(tpeId), &ctx, "sym_name", "tpe_0");
  setIntAttr(adg.getNode(tpeId), &ctx, "num_instruction", 16);
  setIntAttr(adg.getNode(tpeId), &ctx, "num_register", 4);

  auto outNode = std::make_unique<Node>();
  outNode->kind = Node::ModuleOutputNode;
  IdIndex outId = adg.addNode(std::move(outNode));
  setStrAttr(adg.getNode(outId), &ctx, "op_name", "ModuleOutputNode");

  // Add ports and edges: in -> pe -> sw -> tpe -> out.
  auto mkPort = [&](IdIndex parent, Port::Direction dir) -> IdIndex {
    auto port = std::make_unique<Port>();
    port->parentNode = parent;
    port->direction = dir;
    IdIndex pid = adg.addPort(std::move(port));
    if (dir == Port::Input)
      adg.getNode(parent)->inputPorts.push_back(pid);
    else
      adg.getNode(parent)->outputPorts.push_back(pid);
    return pid;
  };

  auto mkEdge = [&](IdIndex srcPort, IdIndex dstPort) {
    auto edge = std::make_unique<Edge>();
    edge->srcPort = srcPort;
    edge->dstPort = dstPort;
    IdIndex eid = adg.addEdge(std::move(edge));
    adg.getPort(srcPort)->connectedEdges.push_back(eid);
    adg.getPort(dstPort)->connectedEdges.push_back(eid);
  };

  IdIndex inOut = mkPort(inId, Port::Output);
  IdIndex peIn = mkPort(peId, Port::Input);
  IdIndex peOut = mkPort(peId, Port::Output);
  IdIndex swIn = mkPort(swId, Port::Input);
  IdIndex swOut = mkPort(swId, Port::Output);
  IdIndex tpeIn = mkPort(tpeId, Port::Input);
  IdIndex tpeOut = mkPort(tpeId, Port::Output);
  IdIndex outIn = mkPort(outId, Port::Input);

  mkEdge(inOut, peIn);
  mkEdge(peOut, swIn);
  mkEdge(swOut, tpeIn);
  mkEdge(tpeOut, outIn);

  // Test Structure mode.
  {
    viz::DOTOptions opts;
    opts.mode = viz::DOTMode::Structure;
    std::string dot = viz::exportADGDot(adg, opts);

    TEST_CONTAINS(dot, "digraph ADG");
    TEST_CONTAINS(dot, "rankdir=LR");

    // PE node: Msquare shape, darkgreen fill.
    TEST_CONTAINS(dot, "Msquare");
    TEST_CONTAINS(dot, "darkgreen");

    // Switch node: diamond shape, lightgray fill.
    TEST_CONTAINS(dot, "diamond");
    TEST_CONTAINS(dot, "lightgray");

    // Temporal PE: Msquare, purple4, larger size.
    TEST_CONTAINS(dot, "purple4");
    TEST_CONTAINS(dot, "width=1.5");

    // Sentinel nodes.
    TEST_CONTAINS(dot, "invhouse");
    TEST_CONTAINS(dot, "lightpink");
    TEST_CONTAINS(dot, "house");
    TEST_CONTAINS(dot, "lightcoral");

    // Edges exist.
    TEST_CONTAINS(dot, "hw_0 -> hw_1");
    TEST_CONTAINS(dot, "hw_1 -> hw_2");
    TEST_CONTAINS(dot, "hw_2 -> hw_3");
    TEST_CONTAINS(dot, "hw_3 -> hw_4");
  }

  // Test Detailed mode.
  {
    viz::DOTOptions opts;
    opts.mode = viz::DOTMode::Detailed;
    std::string dot = viz::exportADGDot(adg, opts);

    TEST_CONTAINS(dot, "rankdir=TB");

    // Detailed mode should include key params.
    TEST_CONTAINS(dot, "insn: 16");
    TEST_CONTAINS(dot, "reg: 4");
  }

  return 0;
}
