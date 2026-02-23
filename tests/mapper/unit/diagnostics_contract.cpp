//===-- diagnostics_contract.cpp - Diagnostics contract test ------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Verifies the diagnostics contract: that failed mappings produce specific
// constraint-class prefixed messages with meaningful content.
//
//===----------------------------------------------------------------------===//

#include "TestUtil.h"
#include "loom/Mapper/Graph.h"
#include "loom/Mapper/Mapper.h"
#include "loom/Mapper/MappingState.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"

#include <vector>

using namespace loom;

/// Test accessor for private Mapper methods.
class MapperTestAccess {
public:
  static void preprocess(Mapper &mapper, const Graph &adg) {
    mapper.preprocess(adg);
  }
  static bool runValidation(Mapper &mapper, const MappingState &state,
                            const Graph &dfg, const Graph &adg,
                            std::string &diagnostics) {
    return mapper.runValidation(state, dfg, adg, diagnostics);
  }
};

namespace {

void setStringAttr(Node *node, mlir::MLIRContext &ctx,
                   llvm::StringRef name, llvm::StringRef value) {
  node->attributes.push_back(mlir::NamedAttribute(
      mlir::StringAttr::get(&ctx, name), mlir::StringAttr::get(&ctx, value)));
}

void setIntAttr(Node *node, mlir::MLIRContext &ctx,
                llvm::StringRef name, int64_t value) {
  node->attributes.push_back(mlir::NamedAttribute(
      mlir::StringAttr::get(&ctx, name),
      mlir::IntegerAttr::get(mlir::IntegerType::get(&ctx, 64), value)));
}

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

/// Check that a diagnostics string contains non-whitespace content.
bool hasContent(const std::string &diag) {
  for (char c : diag) {
    if (c != ' ' && c != '\t' && c != '\n' && c != '\r')
      return true;
  }
  return false;
}

/// Check that a diagnostics string contains a given substring.
bool containsSubstring(const std::string &diag, const std::string &sub) {
  return diag.find(sub) != std::string::npos;
}

} // namespace

int main() {
  mlir::MLIRContext ctx;
  ctx.loadAllAvailableDialects();

  // Test 1: Incompatible ops -> CandidateBuilder failure.
  // DFG has arith.muli but ADG only supports arith.addi.
  // Diagnostics must contain "CPL_MAPPER_NO_COMPATIBLE_HW" and the op name.
  {
    Graph dfg(&ctx);
    Graph adg(&ctx);

    addOpNode(dfg, ctx, "arith.muli", "functional", 1, 1);
    addPENode(adg, ctx, "arith.addi", "functional", 1, 1);

    Mapper mapper;
    Mapper::Options opts;
    opts.budgetSeconds = 5.0;
    opts.seed = 0;
    opts.profile = "balanced";

    Mapper::Result result = mapper.run(dfg, adg, opts);

    TEST_ASSERT(!result.success);
    TEST_ASSERT(!result.diagnostics.empty());
    TEST_ASSERT(hasContent(result.diagnostics));
    TEST_ASSERT(containsSubstring(result.diagnostics,
                                  "CPL_MAPPER_NO_COMPATIBLE_HW"));
    TEST_ASSERT(containsSubstring(result.diagnostics, "arith.muli"));
  }

  // Test 2: Routing failure -> diagnostic must start with "Routing failed".
  // DFG has 2 connected nodes but ADG has 2 disconnected PEs (no routing
  // path between them). Placement can succeed (1 node per PE) but routing
  // must fail because there is no physical path between the PEs.
  // The mapper reports "Routing failed after refinement" as the first
  // failure diagnostic.
  {
    Graph dfg(&ctx);
    Graph adg(&ctx);

    IdIndex sw0 = addOpNode(dfg, ctx, "arith.addi", "functional", 1, 1);
    IdIndex sw1 = addOpNode(dfg, ctx, "arith.addi", "functional", 1, 1);
    addEdgeBetween(dfg, sw0, 0, sw1, 0);

    // Two disconnected PEs: no edges between them.
    addPENode(adg, ctx, "arith.addi", "functional", 1, 1);
    addPENode(adg, ctx, "arith.addi", "functional", 1, 1);

    Mapper mapper;
    Mapper::Options opts;
    opts.budgetSeconds = 5.0;
    opts.seed = 0;
    opts.profile = "balanced";

    Mapper::Result result = mapper.run(dfg, adg, opts);

    TEST_ASSERT(!result.success);
    TEST_ASSERT(!result.diagnostics.empty());
    // The first failure must be routing (not placement or validation).
    // Mapper::run reports exactly "Routing failed after refinement".
    TEST_ASSERT(result.diagnostics.find("Routing failed") == 0);
  }

  // Test 3: Capacity violation -> diagnostic must start with "Placement failed".
  // DFG has 3 nodes (arith.addi) but ADG only has 2 PEs. The mapper cannot
  // place all 3 DFG nodes on 2 non-temporal PEs. Since placement runs before
  // routing/validation, the first failure is the placement stage itself.
  {
    Graph dfg(&ctx);
    Graph adg(&ctx);

    IdIndex sw0 = addOpNode(dfg, ctx, "arith.addi", "functional", 1, 1);
    IdIndex sw1 = addOpNode(dfg, ctx, "arith.addi", "functional", 1, 1);
    IdIndex sw2 = addOpNode(dfg, ctx, "arith.addi", "functional", 1, 1);
    addEdgeBetween(dfg, sw0, 0, sw1, 0);
    addEdgeBetween(dfg, sw1, 0, sw2, 0);

    // Only 2 PEs with direct connection.
    IdIndex hw0 = addPENode(adg, ctx, "arith.addi", "functional", 1, 1);
    IdIndex hw1 = addPENode(adg, ctx, "arith.addi", "functional", 1, 1);
    addADGEdge(adg, adg.getNode(hw0)->outputPorts[0],
               adg.getNode(hw1)->inputPorts[0]);

    Mapper mapper;
    Mapper::Options opts;
    opts.budgetSeconds = 5.0;
    opts.seed = 0;
    opts.profile = "balanced";

    Mapper::Result result = mapper.run(dfg, adg, opts);

    // Not enough PEs for all DFG nodes.
    TEST_ASSERT(!result.success);
    TEST_ASSERT(!result.diagnostics.empty());
    // The first failure must be at the placement stage (not routing or
    // validation). Mapper::run reports exactly "Placement failed" when
    // runPlacement returns false.
    TEST_ASSERT(result.diagnostics.find("Placement failed") == 0);
  }

  // Test 4: Deterministic C2 validation failure via manual MappingState.
  // Construct a DFG with 1 node and an ADG with 1 PE. Manually map the
  // SW node to the HW node correctly, but map a SW output port to an HW
  // input port (direction mismatch). This passes C1 but fails C2 with
  // "C2: direction mismatch sw_port=..." as the first validation error.
  {
    Graph dfg(&ctx);
    Graph adg(&ctx);

    IdIndex sw0 = addOpNode(dfg, ctx, "arith.addi", "functional", 1, 1);
    IdIndex hw0 = addPENode(adg, ctx, "arith.addi", "functional", 1, 1);

    // Manually construct MappingState.
    MappingState state;
    state.init(dfg, adg);
    state.mapNode(sw0, hw0, dfg, adg);
    // Correct port mapping for input (so C1 passes).
    state.mapPort(dfg.getNode(sw0)->inputPorts[0],
                  adg.getNode(hw0)->inputPorts[0], dfg, adg);
    // INTENTIONALLY WRONG: map SW output port to HW input port
    // (direction mismatch).
    IdIndex swOutPort = dfg.getNode(sw0)->outputPorts[0];
    IdIndex hwInPort = adg.getNode(hw0)->inputPorts[0];
    state.swPortToHwPort[swOutPort] = hwInPort;

    Mapper mapper;
    std::string validationDiag;
    bool valid = MapperTestAccess::runValidation(mapper, state, dfg, adg,
                                                 validationDiag);

    TEST_ASSERT(!valid);
    TEST_ASSERT(!validationDiag.empty());
    // The first validation failure must be C2 (port direction mismatch).
    TEST_ASSERT(validationDiag.find("C2:") == 0);
    TEST_ASSERT(containsSubstring(validationDiag, "direction mismatch"));
  }

  // Test 5: Deterministic C4 validation failure via manual MappingState.
  // Construct a DFG with 2 nodes and an ADG with 1 PE. Manually map both
  // SW nodes to the same HW node and set up all ports/edges correctly so
  // C1-C3 pass. C4 must fail with "C4: capacity exceeded on hw_node=..."
  // because 2 ops are placed on 1 non-temporal PE.
  {
    Graph dfg(&ctx);
    Graph adg(&ctx);

    // DFG: 2 nodes, 1 edge.
    IdIndex sw0 = addOpNode(dfg, ctx, "arith.addi", "functional", 1, 1);
    IdIndex sw1 = addOpNode(dfg, ctx, "arith.addi", "functional", 1, 1);
    IdIndex dfgEdge = addEdgeBetween(dfg, sw0, 0, sw1, 0);

    // ADG: 1 PE with 2 inputs and 2 outputs (enough ports for both ops).
    IdIndex hw0 = addPENode(adg, ctx, "arith.addi", "functional", 2, 2);
    // Self-loop edge for routing: output[0] -> input[1].
    addADGEdge(adg, adg.getNode(hw0)->outputPorts[0],
               adg.getNode(hw0)->inputPorts[1]);

    // Manually construct MappingState.
    MappingState state;
    state.init(dfg, adg);

    // Map both DFG nodes to the same PE.
    state.mapNode(sw0, hw0, dfg, adg);
    state.mapNode(sw1, hw0, dfg, adg);

    // Map ports correctly (different HW ports for each SW node).
    state.mapPort(dfg.getNode(sw0)->inputPorts[0],
                  adg.getNode(hw0)->inputPorts[0], dfg, adg);
    state.mapPort(dfg.getNode(sw0)->outputPorts[0],
                  adg.getNode(hw0)->outputPorts[0], dfg, adg);
    state.mapPort(dfg.getNode(sw1)->inputPorts[0],
                  adg.getNode(hw0)->inputPorts[1], dfg, adg);
    state.mapPort(dfg.getNode(sw1)->outputPorts[0],
                  adg.getNode(hw0)->outputPorts[1], dfg, adg);

    // Set up routing path for the DFG edge (sw0 out -> sw1 in).
    // Path: [hw0_outPort0, hw0_inPort1] (output[0] -> input[1] via edge).
    state.swEdgeToHwPaths.resize(dfg.countEdges());
    state.swEdgeToHwPaths[dfgEdge] = {
        adg.getNode(hw0)->outputPorts[0],
        adg.getNode(hw0)->inputPorts[1]};

    Mapper mapper;
    MapperTestAccess::preprocess(mapper, adg);
    std::string validationDiag;
    bool valid = MapperTestAccess::runValidation(mapper, state, dfg, adg,
                                                 validationDiag);

    TEST_ASSERT(!valid);
    TEST_ASSERT(!validationDiag.empty());
    // C1-C3 must pass (valid mapping with correct directions and route).
    // C4 must fail because 2 ops on 1 non-temporal PE exceeds capacity.
    TEST_ASSERT(validationDiag.find("C4:") == 0);
    TEST_ASSERT(containsSubstring(validationDiag, "capacity exceeded"));
  }

  return 0;
}
