//===-- viz_dot_export.cpp - DOT export structure tests ----------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Verify that DOTExporter produces well-formed DOT output with expected
// structure, node styles, and edge properties for DFG, ADG, and mapped views.
//
//===----------------------------------------------------------------------===//

#include "TestUtil.h"
#include "loom/Mapper/Graph.h"
#include "loom/Mapper/Mapper.h"
#include "loom/Mapper/MappingState.h"
#include "loom/Viz/DOTExporter.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"

using namespace loom;

namespace {

void setStringAttr(Node *node, mlir::MLIRContext &ctx, llvm::StringRef name,
                   llvm::StringRef value) {
  node->attributes.push_back(mlir::NamedAttribute(
      mlir::StringAttr::get(&ctx, name), mlir::StringAttr::get(&ctx, value)));
}

IdIndex addOpNode(Graph &g, mlir::MLIRContext &ctx, llvm::StringRef opName,
                  llvm::StringRef resClass, unsigned numIn = 1,
                  unsigned numOut = 1) {
  auto node = std::make_unique<Node>();
  node->kind = Node::OperationNode;
  setStringAttr(node.get(), ctx, "op_name", opName);
  setStringAttr(node.get(), ctx, "resource_class", resClass);
  IdIndex nodeId = g.addNode(std::move(node));
  for (unsigned i = 0; i < numIn; ++i) {
    auto port = std::make_unique<Port>();
    port->parentNode = nodeId;
    port->direction = Port::Input;
    IdIndex pid = g.addPort(std::move(port));
    g.getNode(nodeId)->inputPorts.push_back(pid);
  }
  for (unsigned i = 0; i < numOut; ++i) {
    auto port = std::make_unique<Port>();
    port->parentNode = nodeId;
    port->direction = Port::Output;
    IdIndex pid = g.addPort(std::move(port));
    g.getNode(nodeId)->outputPorts.push_back(pid);
  }
  return nodeId;
}

IdIndex addSentinelNode(Graph &g, Node::Kind kind) {
  auto node = std::make_unique<Node>();
  node->kind = kind;
  IdIndex nodeId = g.addNode(std::move(node));
  if (kind == Node::ModuleInputNode) {
    auto port = std::make_unique<Port>();
    port->parentNode = nodeId;
    port->direction = Port::Output;
    IdIndex pid = g.addPort(std::move(port));
    g.getNode(nodeId)->outputPorts.push_back(pid);
  } else {
    auto port = std::make_unique<Port>();
    port->parentNode = nodeId;
    port->direction = Port::Input;
    IdIndex pid = g.addPort(std::move(port));
    g.getNode(nodeId)->inputPorts.push_back(pid);
  }
  return nodeId;
}

IdIndex addEdgeBetween(Graph &g, IdIndex srcNode, unsigned srcPortIdx,
                       IdIndex dstNode, unsigned dstPortIdx) {
  IdIndex srcPort = g.getNode(srcNode)->outputPorts[srcPortIdx];
  IdIndex dstPort = g.getNode(dstNode)->inputPorts[dstPortIdx];
  auto edge = std::make_unique<Edge>();
  edge->srcPort = srcPort;
  edge->dstPort = dstPort;
  IdIndex eid = g.addEdge(std::move(edge));
  g.getPort(srcPort)->connectedEdges.push_back(eid);
  g.getPort(dstPort)->connectedEdges.push_back(eid);
  return eid;
}

void addADGEdge(Graph &adg, IdIndex srcPort, IdIndex dstPort) {
  auto e = std::make_unique<Edge>();
  e->srcPort = srcPort;
  e->dstPort = dstPort;
  IdIndex eid = adg.addEdge(std::move(e));
  adg.getPort(srcPort)->connectedEdges.push_back(eid);
  adg.getPort(dstPort)->connectedEdges.push_back(eid);
}

} // namespace

int main() {
  mlir::MLIRContext ctx;
  ctx.loadAllAvailableDialects();

  // Test 1: DFG DOT export produces valid digraph with expected node count.
  {
    Graph dfg(&ctx);
    IdIndex in0 = addSentinelNode(dfg, Node::ModuleInputNode);
    IdIndex op0 = addOpNode(dfg, ctx, "arith.addi", "functional", 1, 1);
    IdIndex out0 = addSentinelNode(dfg, Node::ModuleOutputNode);
    addEdgeBetween(dfg, in0, 0, op0, 0);
    addEdgeBetween(dfg, op0, 0, out0, 0);

    std::string dot = viz::exportDFGDot(dfg);
    TEST_ASSERT(!dot.empty());
    TEST_ASSERT(dot.find("digraph") != std::string::npos);
    // Should contain node references for the 3 nodes.
    TEST_ASSERT(dot.find("->") != std::string::npos);
  }

  // Test 2: ADG DOT export in Structure mode.
  {
    Graph adg(&ctx);
    IdIndex pe0 = addOpNode(adg, ctx, "arith.addi", "functional", 1, 1);
    IdIndex pe1 = addOpNode(adg, ctx, "arith.addi", "functional", 1, 1);
    IdIndex sw0 = addOpNode(adg, ctx, "fabric.switch", "routing", 2, 2);
    addADGEdge(adg, adg.getNode(pe0)->outputPorts[0],
               adg.getNode(sw0)->inputPorts[0]);
    addADGEdge(adg, adg.getNode(pe1)->outputPorts[0],
               adg.getNode(sw0)->inputPorts[1]);
    addADGEdge(adg, adg.getNode(sw0)->outputPorts[0],
               adg.getNode(pe0)->inputPorts[0]);
    addADGEdge(adg, adg.getNode(sw0)->outputPorts[1],
               adg.getNode(pe1)->inputPorts[0]);

    viz::DOTOptions opts;
    opts.mode = viz::DOTMode::Structure;
    std::string dot = viz::exportADGDot(adg, opts);
    TEST_ASSERT(!dot.empty());
    TEST_ASSERT(dot.find("digraph") != std::string::npos);
    TEST_ASSERT(dot.find("->") != std::string::npos);
  }

  // Test 3: ADG DOT export in Detailed mode.
  {
    Graph adg(&ctx);
    addOpNode(adg, ctx, "arith.addi", "functional", 2, 1);

    viz::DOTOptions opts;
    opts.mode = viz::DOTMode::Detailed;
    std::string dot = viz::exportADGDot(adg, opts);
    TEST_ASSERT(!dot.empty());
    TEST_ASSERT(dot.find("digraph") != std::string::npos);
  }

  // Test 4: Mapped overlay DOT with a successful 2-node mapping.
  {
    Graph dfg(&ctx);
    Graph adg(&ctx);

    IdIndex sw0 = addOpNode(dfg, ctx, "arith.addi", "functional", 1, 1);
    IdIndex sw1 = addOpNode(dfg, ctx, "arith.addi", "functional", 1, 1);
    addEdgeBetween(dfg, sw0, 0, sw1, 0);

    IdIndex hw0 = addOpNode(adg, ctx, "arith.addi", "functional", 1, 1);
    IdIndex hw1 = addOpNode(adg, ctx, "arith.addi", "functional", 1, 1);
    addADGEdge(adg, adg.getNode(hw0)->outputPorts[0],
               adg.getNode(hw1)->inputPorts[0]);

    Mapper mapper;
    Mapper::Options mopts;
    mopts.budgetSeconds = 5.0;
    mopts.seed = 42;
    Mapper::Result result = mapper.run(dfg, adg, mopts);
    TEST_ASSERT(result.success);

    std::string overlayDot =
        viz::exportMappedOverlayDot(dfg, adg, result.state);
    TEST_ASSERT(!overlayDot.empty());
    TEST_ASSERT(overlayDot.find("digraph") != std::string::npos);
  }

  // Test 5: HTML generation produces self-contained output with viz.js.
  {
    Graph dfg(&ctx);
    addOpNode(dfg, ctx, "arith.addi", "functional", 1, 1);

    std::string dot = viz::exportDFGDot(dfg);
    std::string html = viz::generateHTML(dot, "Test DFG");
    TEST_ASSERT(!html.empty());
    TEST_ASSERT(html.find("<html") != std::string::npos ||
                html.find("<!DOCTYPE") != std::string::npos ||
                html.find("<HTML") != std::string::npos);
    // Should contain the DOT string or viz.js reference.
    TEST_ASSERT(html.find("viz") != std::string::npos ||
                html.find("Viz") != std::string::npos ||
                html.find("graphviz") != std::string::npos ||
                html.find("digraph") != std::string::npos);
  }

  // Test 6: Mapped HTML generation with side-by-side panels.
  {
    Graph dfg(&ctx);
    Graph adg(&ctx);

    IdIndex sw0 = addOpNode(dfg, ctx, "arith.addi", "functional", 1, 1);
    IdIndex sw1 = addOpNode(dfg, ctx, "arith.addi", "functional", 1, 1);
    addEdgeBetween(dfg, sw0, 0, sw1, 0);

    IdIndex hw0 = addOpNode(adg, ctx, "arith.addi", "functional", 1, 1);
    IdIndex hw1 = addOpNode(adg, ctx, "arith.addi", "functional", 1, 1);
    addADGEdge(adg, adg.getNode(hw0)->outputPorts[0],
               adg.getNode(hw1)->inputPorts[0]);

    Mapper mapper;
    Mapper::Options mopts;
    mopts.budgetSeconds = 5.0;
    mopts.seed = 42;
    Mapper::Result result = mapper.run(dfg, adg, mopts);
    TEST_ASSERT(result.success);

    std::string overlayDot =
        viz::exportMappedOverlayDot(dfg, adg, result.state);
    std::string dfgDot = viz::exportMappedDFGDot(dfg, result.state);
    std::string adgDot = viz::exportMappedADGDot(dfg, adg, result.state);

    std::string html = viz::generateMappedHTML(overlayDot, dfgDot, adgDot,
                                                "{}", "{}", "Test Mapped");
    TEST_ASSERT(!html.empty());
    TEST_ASSERT(html.find("<html") != std::string::npos ||
                html.find("<!DOCTYPE") != std::string::npos ||
                html.find("<HTML") != std::string::npos);
  }

  return 0;
}
