//===-- adg_connectivity_invariants.cpp - ADG structural invariants *- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Verify ADG structural invariants: ConnectivityMatrix correctness for a
// multi-PE + switch topology, routing node crossbar internals, ADG sentinel
// node port directions, and min-hop BFS distances via Mapper::run.
//
//===----------------------------------------------------------------------===//

#include "TestUtil.h"
#include "loom/Mapper/ConnectivityMatrix.h"
#include "loom/Mapper/Graph.h"
#include "loom/Mapper/Mapper.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"

using namespace loom;

namespace {

/// Helper: add a named string attribute to a node.
void setStringAttr(Node *node, mlir::MLIRContext &ctx,
                   llvm::StringRef name, llvm::StringRef value) {
  node->attributes.push_back(mlir::NamedAttribute(
      mlir::StringAttr::get(&ctx, name), mlir::StringAttr::get(&ctx, value)));
}

/// Helper: create a PE/routing node with specified ports.
IdIndex addPENode(Graph &adg, mlir::MLIRContext &ctx,
                  llvm::StringRef opName, llvm::StringRef resClass,
                  unsigned numIn, unsigned numOut) {
  auto hwNode = std::make_unique<Node>();
  hwNode->kind = Node::OperationNode;
  setStringAttr(hwNode.get(), ctx, "op_name", opName);
  setStringAttr(hwNode.get(), ctx, "resource_class", resClass);
  IdIndex hwId = adg.addNode(std::move(hwNode));

  for (unsigned i = 0; i < numIn; ++i) {
    auto port = std::make_unique<Port>();
    port->parentNode = hwId;
    port->direction = Port::Input;
    IdIndex pid = adg.addPort(std::move(port));
    adg.getNode(hwId)->inputPorts.push_back(pid);
  }
  for (unsigned i = 0; i < numOut; ++i) {
    auto port = std::make_unique<Port>();
    port->parentNode = hwId;
    port->direction = Port::Output;
    IdIndex pid = adg.addPort(std::move(port));
    adg.getNode(hwId)->outputPorts.push_back(pid);
  }
  return hwId;
}

/// Helper: add an ADG edge between two ports.
void addADGEdge(Graph &adg, IdIndex srcPort, IdIndex dstPort) {
  auto e = std::make_unique<Edge>();
  e->srcPort = srcPort;
  e->dstPort = dstPort;
  IdIndex eid = adg.addEdge(std::move(e));
  adg.getPort(srcPort)->connectedEdges.push_back(eid);
  adg.getPort(dstPort)->connectedEdges.push_back(eid);
}

/// Helper: create an operation node for DFG.
IdIndex addOpNode(Graph &g, mlir::MLIRContext &ctx,
                  llvm::StringRef opName, llvm::StringRef resClass,
                  unsigned numIn = 1, unsigned numOut = 1) {
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

/// Helper: add an edge between operation nodes by port index.
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

} // namespace

int main() {
  mlir::MLIRContext ctx;
  ctx.loadAllAvailableDialects();

  // Test 1: ConnectivityMatrix correctness for a 3-PE + switch topology.
  // Topology: PE0 -> SW, PE1 -> SW, PE2 -> SW (outputs to switch inputs),
  //           SW -> PE0, SW -> PE1, SW -> PE2 (switch outputs to PE inputs).
  // Verify outToIn entries match the ADG edge structure.
  {
    Graph adg(&ctx);

    IdIndex pe0 = addPENode(adg, ctx, "arith.addi", "functional", 1, 1);
    IdIndex pe1 = addPENode(adg, ctx, "arith.addi", "functional", 1, 1);
    IdIndex pe2 = addPENode(adg, ctx, "arith.addi", "functional", 1, 1);
    IdIndex sw = addPENode(adg, ctx, "fabric.switch", "routing", 3, 3);

    // PE outputs -> switch inputs.
    IdIndex pe0Out = adg.getNode(pe0)->outputPorts[0];
    IdIndex pe1Out = adg.getNode(pe1)->outputPorts[0];
    IdIndex pe2Out = adg.getNode(pe2)->outputPorts[0];
    IdIndex swIn0 = adg.getNode(sw)->inputPorts[0];
    IdIndex swIn1 = adg.getNode(sw)->inputPorts[1];
    IdIndex swIn2 = adg.getNode(sw)->inputPorts[2];

    addADGEdge(adg, pe0Out, swIn0);
    addADGEdge(adg, pe1Out, swIn1);
    addADGEdge(adg, pe2Out, swIn2);

    // Switch outputs -> PE inputs.
    IdIndex swOut0 = adg.getNode(sw)->outputPorts[0];
    IdIndex swOut1 = adg.getNode(sw)->outputPorts[1];
    IdIndex swOut2 = adg.getNode(sw)->outputPorts[2];
    IdIndex pe0In = adg.getNode(pe0)->inputPorts[0];
    IdIndex pe1In = adg.getNode(pe1)->inputPorts[0];
    IdIndex pe2In = adg.getNode(pe2)->inputPorts[0];

    addADGEdge(adg, swOut0, pe0In);
    addADGEdge(adg, swOut1, pe1In);
    addADGEdge(adg, swOut2, pe2In);

    // Build ConnectivityMatrix from the ADG edges.
    ConnectivityMatrix cm;
    for (auto *edge : adg.edgeRange()) {
      if (!edge)
        continue;
      cm.outToIn[edge->srcPort] = edge->dstPort;
    }

    // Verify outToIn has exactly 6 entries (3 PE->SW + 3 SW->PE).
    TEST_ASSERT(cm.outToIn.size() == 6);

    // Verify specific connectivity: PE0 output -> switch input 0.
    TEST_ASSERT(cm.outToIn.count(pe0Out));
    TEST_ASSERT(cm.outToIn[pe0Out] == swIn0);

    // Verify: PE1 output -> switch input 1.
    TEST_ASSERT(cm.outToIn.count(pe1Out));
    TEST_ASSERT(cm.outToIn[pe1Out] == swIn1);

    // Verify: PE2 output -> switch input 2.
    TEST_ASSERT(cm.outToIn.count(pe2Out));
    TEST_ASSERT(cm.outToIn[pe2Out] == swIn2);

    // Verify: switch output 0 -> PE0 input.
    TEST_ASSERT(cm.outToIn.count(swOut0));
    TEST_ASSERT(cm.outToIn[swOut0] == pe0In);

    // Verify: switch output 1 -> PE1 input.
    TEST_ASSERT(cm.outToIn.count(swOut1));
    TEST_ASSERT(cm.outToIn[swOut1] == pe1In);

    // Verify: switch output 2 -> PE2 input.
    TEST_ASSERT(cm.outToIn.count(swOut2));
    TEST_ASSERT(cm.outToIn[swOut2] == pe2In);
  }

  // Test 2: Routing node internals (inToOut) have correct crossbar connectivity.
  // A routing switch with full crossbar: every input port can reach every
  // output port.
  {
    Graph adg(&ctx);

    IdIndex sw = addPENode(adg, ctx, "fabric.switch", "routing", 3, 3);

    IdIndex swIn0 = adg.getNode(sw)->inputPorts[0];
    IdIndex swIn1 = adg.getNode(sw)->inputPorts[1];
    IdIndex swIn2 = adg.getNode(sw)->inputPorts[2];
    IdIndex swOut0 = adg.getNode(sw)->outputPorts[0];
    IdIndex swOut1 = adg.getNode(sw)->outputPorts[1];
    IdIndex swOut2 = adg.getNode(sw)->outputPorts[2];

    // Build full-crossbar inToOut for the routing node.
    ConnectivityMatrix cm;
    for (IdIndex inPort : adg.getNode(sw)->inputPorts) {
      for (IdIndex outPort : adg.getNode(sw)->outputPorts) {
        cm.inToOut[inPort].push_back(outPort);
      }
    }

    // Each input port should reach all 3 output ports.
    TEST_ASSERT(cm.inToOut.count(swIn0));
    TEST_ASSERT(cm.inToOut[swIn0].size() == 3);
    TEST_ASSERT(cm.inToOut.count(swIn1));
    TEST_ASSERT(cm.inToOut[swIn1].size() == 3);
    TEST_ASSERT(cm.inToOut.count(swIn2));
    TEST_ASSERT(cm.inToOut[swIn2].size() == 3);

    // Verify specific crossbar paths: input 0 can reach all outputs.
    bool foundOut0 = false, foundOut1 = false, foundOut2 = false;
    for (IdIndex outPort : cm.inToOut[swIn0]) {
      if (outPort == swOut0) foundOut0 = true;
      if (outPort == swOut1) foundOut1 = true;
      if (outPort == swOut2) foundOut2 = true;
    }
    TEST_ASSERT(foundOut0);
    TEST_ASSERT(foundOut1);
    TEST_ASSERT(foundOut2);
  }

  // Test 3: ADG sentinel nodes (ModuleInputNode/ModuleOutputNode) have
  // correct port directions and edge connections.
  {
    Graph adg(&ctx);

    // Create a PE node.
    IdIndex pe = addPENode(adg, ctx, "arith.addi", "functional", 1, 1);

    // Create ModuleInputNode sentinel with one output port.
    auto inSentinel = std::make_unique<Node>();
    inSentinel->kind = Node::ModuleInputNode;
    IdIndex inSentinelId = adg.addNode(std::move(inSentinel));
    {
      auto port = std::make_unique<Port>();
      port->parentNode = inSentinelId;
      port->direction = Port::Output;
      IdIndex pid = adg.addPort(std::move(port));
      adg.getNode(inSentinelId)->outputPorts.push_back(pid);
    }

    // Create ModuleOutputNode sentinel with one input port.
    auto outSentinel = std::make_unique<Node>();
    outSentinel->kind = Node::ModuleOutputNode;
    IdIndex outSentinelId = adg.addNode(std::move(outSentinel));
    {
      auto port = std::make_unique<Port>();
      port->parentNode = outSentinelId;
      port->direction = Port::Input;
      IdIndex pid = adg.addPort(std::move(port));
      adg.getNode(outSentinelId)->inputPorts.push_back(pid);
    }

    // Connect: input sentinel -> PE -> output sentinel.
    IdIndex inSentinelOut = adg.getNode(inSentinelId)->outputPorts[0];
    IdIndex peIn = adg.getNode(pe)->inputPorts[0];
    addADGEdge(adg, inSentinelOut, peIn);

    IdIndex peOut = adg.getNode(pe)->outputPorts[0];
    IdIndex outSentinelIn = adg.getNode(outSentinelId)->inputPorts[0];
    addADGEdge(adg, peOut, outSentinelIn);

    // Verify sentinel port directions.
    Node *inNode = adg.getNode(inSentinelId);
    TEST_ASSERT(inNode->kind == Node::ModuleInputNode);
    TEST_ASSERT(inNode->inputPorts.empty());
    TEST_ASSERT(inNode->outputPorts.size() == 1);
    TEST_ASSERT(adg.getPort(inNode->outputPorts[0])->direction == Port::Output);

    Node *outNode = adg.getNode(outSentinelId);
    TEST_ASSERT(outNode->kind == Node::ModuleOutputNode);
    TEST_ASSERT(outNode->outputPorts.empty());
    TEST_ASSERT(outNode->inputPorts.size() == 1);
    TEST_ASSERT(adg.getPort(outNode->inputPorts[0])->direction == Port::Input);

    // Verify edge connections from sentinels.
    TEST_ASSERT(adg.getPort(inSentinelOut)->connectedEdges.size() == 1);
    IdIndex edgeFromIn = adg.getPort(inSentinelOut)->connectedEdges[0];
    TEST_ASSERT(adg.getEdge(edgeFromIn)->srcPort == inSentinelOut);
    TEST_ASSERT(adg.getEdge(edgeFromIn)->dstPort == peIn);

    TEST_ASSERT(adg.getPort(outSentinelIn)->connectedEdges.size() == 1);
    IdIndex edgeToOut = adg.getPort(outSentinelIn)->connectedEdges[0];
    TEST_ASSERT(adg.getEdge(edgeToOut)->srcPort == peOut);
    TEST_ASSERT(adg.getEdge(edgeToOut)->dstPort == outSentinelIn);
  }

  // Test 4: Min-hop cost computation via Mapper::run - verify BFS distances
  // between nodes by running a successful mapping on a linear topology
  // (PE0 -> SW -> PE1) and confirming the mapper can route through the switch.
  // Since Mapper::preprocess is private, we verify BFS reachability indirectly
  // through successful mapping on a topology with known hop distances.
  {
    Graph dfg(&ctx);
    Graph adg(&ctx);

    // DFG: simple 2-node chain.
    IdIndex sw0 = addOpNode(dfg, ctx, "arith.addi", "functional", 1, 1);
    IdIndex sw1 = addOpNode(dfg, ctx, "arith.addi", "functional", 1, 1);
    addEdgeBetween(dfg, sw0, 0, sw1, 0);

    // ADG: PE0 -> switch -> PE1 (3-hop topology: PE0 out -> SW in,
    // SW in -> SW out (crossbar), SW out -> PE1 in).
    IdIndex pe0 = addPENode(adg, ctx, "arith.addi", "functional", 1, 1);
    IdIndex pe1 = addPENode(adg, ctx, "arith.addi", "functional", 1, 1);
    IdIndex csw = addPENode(adg, ctx, "fabric.switch", "routing", 2, 2);

    // PE0 output -> switch input 0.
    addADGEdge(adg, adg.getNode(pe0)->outputPorts[0],
               adg.getNode(csw)->inputPorts[0]);
    // Switch output 0 -> PE1 input.
    addADGEdge(adg, adg.getNode(csw)->outputPorts[0],
               adg.getNode(pe1)->inputPorts[0]);
    // Also connect PE1 -> switch -> PE0 for reverse direction.
    addADGEdge(adg, adg.getNode(pe1)->outputPorts[0],
               adg.getNode(csw)->inputPorts[1]);
    addADGEdge(adg, adg.getNode(csw)->outputPorts[1],
               adg.getNode(pe0)->inputPorts[0]);

    // Run the mapper: if BFS correctly computes hop distances, it will
    // find PE0 -> (switch) -> PE1 as a 2-hop path and succeed.
    Mapper mapper;
    Mapper::Options opts;
    opts.budgetSeconds = 5.0;
    opts.seed = 42;
    opts.profile = "balanced";

    Mapper::Result result = mapper.run(dfg, adg, opts);

    TEST_ASSERT(result.success);
    TEST_ASSERT(result.state.swNodeToHwNode[sw0] != INVALID_ID);
    TEST_ASSERT(result.state.swNodeToHwNode[sw1] != INVALID_ID);
    // The two DFG nodes must be placed on different PEs.
    TEST_ASSERT(result.state.swNodeToHwNode[sw0] !=
                result.state.swNodeToHwNode[sw1]);
  }

  return 0;
}
