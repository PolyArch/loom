//===-- cpsat_group_constraint.cpp - CP-SAT group atomicity test ---*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Verify that the CP-SAT solver enforces group atomicity constraints:
// when a group candidate exists (multiple SW nodes -> 1 HW PE), all group
// members must be placed on the same PE or none of them.
//
//===----------------------------------------------------------------------===//

#include "TestUtil.h"
#include "loom/Mapper/CPSATSolver.h"
#include "loom/Mapper/Graph.h"
#include "loom/Mapper/MappingState.h"
#include "loom/Mapper/TechMapper.h"
#include "loom/Mapper/ConnectivityMatrix.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"

using namespace loom;

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

void setArrayStrAttr(Node *node, mlir::MLIRContext &ctx,
                     llvm::StringRef name,
                     llvm::ArrayRef<llvm::StringRef> values) {
  llvm::SmallVector<mlir::Attribute, 4> attrs;
  for (auto v : values)
    attrs.push_back(mlir::StringAttr::get(&ctx, v));
  node->attributes.push_back(mlir::NamedAttribute(
      mlir::StringAttr::get(&ctx, name),
      mlir::ArrayAttr::get(&ctx, attrs)));
}

void setArrayIntAttr(Node *node, mlir::MLIRContext &ctx,
                     llvm::StringRef name,
                     llvm::ArrayRef<int64_t> values) {
  llvm::SmallVector<mlir::Attribute, 4> attrs;
  for (auto v : values)
    attrs.push_back(mlir::IntegerAttr::get(
        mlir::IntegerType::get(&ctx, 64), v));
  node->attributes.push_back(mlir::NamedAttribute(
      mlir::StringAttr::get(&ctx, name),
      mlir::ArrayAttr::get(&ctx, attrs)));
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

} // namespace

int main() {
  mlir::MLIRContext ctx;
  ctx.loadAllAvailableDialects();

  // Test 1: Group candidate data structure verification.
  // Even without OR-Tools, verify that group candidates are correctly
  // identified and can be used to build group constraints.
  {
    Graph dfg(&ctx);
    Graph adg(&ctx);

    // DFG: addi -> muli chain.
    IdIndex sw0 = addOpNode(dfg, ctx, "arith.addi", "functional", 2, 1);
    IdIndex sw1 = addOpNode(dfg, ctx, "arith.muli", "functional", 2, 1);
    addEdgeBetween(dfg, sw0, 0, sw1, 0);

    // ADG: multi-op PE with body_ops = [arith.addi, arith.muli].
    auto hwNode = std::make_unique<Node>();
    hwNode->kind = Node::OperationNode;
    setStringAttr(hwNode.get(), ctx, "op_name", "fabric.pe");
    setStringAttr(hwNode.get(), ctx, "resource_class", "functional");
    setArrayStrAttr(hwNode.get(), ctx, "body_ops",
                    {"arith.addi", "arith.muli"});
    setArrayIntAttr(hwNode.get(), ctx, "body_edges", {0, 1});
    IdIndex hwId = adg.addNode(std::move(hwNode));

    for (int i = 0; i < 3; ++i) {
      auto port = std::make_unique<Port>();
      port->parentNode = hwId;
      port->direction = Port::Input;
      IdIndex pid = adg.addPort(std::move(port));
      adg.getNode(hwId)->inputPorts.push_back(pid);
    }
    for (int i = 0; i < 2; ++i) {
      auto port = std::make_unique<Port>();
      port->parentNode = hwId;
      port->direction = Port::Output;
      IdIndex pid = adg.addPort(std::move(port));
      adg.getNode(hwId)->outputPorts.push_back(pid);
    }

    TechMapper mapper;
    auto candidates = mapper.map(dfg, adg);

    // Verify group candidates exist for both SW nodes.
    TEST_ASSERT(candidates.count(sw0) > 0);
    TEST_ASSERT(candidates.count(sw1) > 0);

    bool foundGroupSw0 = false;
    bool foundGroupSw1 = false;
    IdIndex groupHw = INVALID_ID;

    for (const auto &c : candidates[sw0]) {
      if (c.isGroup && c.swNodeIds.size() == 2) {
        foundGroupSw0 = true;
        groupHw = c.hwNodeId;
        // Verify group contains both SW nodes.
        bool hasSw0 = false, hasSw1 = false;
        for (IdIndex s : c.swNodeIds) {
          if (s == sw0) hasSw0 = true;
          if (s == sw1) hasSw1 = true;
        }
        TEST_ASSERT(hasSw0);
        TEST_ASSERT(hasSw1);
        break;
      }
    }
    TEST_ASSERT(foundGroupSw0);
    TEST_ASSERT(groupHw == hwId);

    for (const auto &c : candidates[sw1]) {
      if (c.isGroup && c.swNodeIds.size() == 2) {
        foundGroupSw1 = true;
        break;
      }
    }
    TEST_ASSERT(foundGroupSw1);
  }

  // Test 2: CP-SAT solver with group atomicity (requires OR-Tools).
  // Create a scenario with 2 SW nodes, 1 multi-op PE, and 2 single-op PEs.
  // The solver should place both SW nodes on the multi-op PE together.
  if (CPSATSolver::isAvailable()) {
    Graph dfg(&ctx);
    Graph adg(&ctx);

    // DFG: addi -> muli.
    IdIndex sw0 = addOpNode(dfg, ctx, "arith.addi", "functional", 2, 1);
    IdIndex sw1 = addOpNode(dfg, ctx, "arith.muli", "functional", 2, 1);
    addEdgeBetween(dfg, sw0, 0, sw1, 0);

    // ADG: multi-op PE (hwId0) with body_ops = [addi, muli].
    auto multiPE = std::make_unique<Node>();
    multiPE->kind = Node::OperationNode;
    setStringAttr(multiPE.get(), ctx, "op_name", "fabric.pe");
    setStringAttr(multiPE.get(), ctx, "resource_class", "functional");
    setArrayStrAttr(multiPE.get(), ctx, "body_ops",
                    {"arith.addi", "arith.muli"});
    setArrayIntAttr(multiPE.get(), ctx, "body_edges", {0, 1});
    IdIndex hwMulti = adg.addNode(std::move(multiPE));
    for (int i = 0; i < 3; ++i) {
      auto port = std::make_unique<Port>();
      port->parentNode = hwMulti;
      port->direction = Port::Input;
      IdIndex pid = adg.addPort(std::move(port));
      adg.getNode(hwMulti)->inputPorts.push_back(pid);
    }
    for (int i = 0; i < 2; ++i) {
      auto port = std::make_unique<Port>();
      port->parentNode = hwMulti;
      port->direction = Port::Output;
      IdIndex pid = adg.addPort(std::move(port));
      adg.getNode(hwMulti)->outputPorts.push_back(pid);
    }

    // ADG: single-op PE (hwAdd) for addi only.
    IdIndex hwAdd = addPENode(adg, ctx, "fabric.pe", "functional", 2, 1);
    setStringAttr(adg.getNode(hwAdd), ctx, "supported_ops", "arith.addi");

    // ADG: single-op PE (hwMul) for muli only.
    IdIndex hwMul = addPENode(adg, ctx, "fabric.pe", "functional", 2, 1);
    setStringAttr(adg.getNode(hwMul), ctx, "supported_ops", "arith.muli");

    // Run tech-mapping.
    TechMapper techMapper;
    auto candidates = techMapper.map(dfg, adg);

    // Build candidate set manually to include group + individual options.
    // Ensure both sw0 and sw1 can go to the multi-op PE (as a group)
    // and also individually to their respective single-op PEs.
    CandidateSet cands;

    // Group candidate for sw0 at hwMulti.
    Candidate groupCandSw0;
    groupCandSw0.hwNodeId = hwMulti;
    groupCandSw0.swNodeIds = {sw0, sw1};
    groupCandSw0.isGroup = true;
    cands[sw0].push_back(groupCandSw0);

    // Individual candidate for sw0 at hwAdd.
    Candidate indivCandSw0;
    indivCandSw0.hwNodeId = hwAdd;
    indivCandSw0.swNodeIds = {sw0};
    indivCandSw0.isGroup = false;
    cands[sw0].push_back(indivCandSw0);

    // Group candidate for sw1 at hwMulti.
    Candidate groupCandSw1;
    groupCandSw1.hwNodeId = hwMulti;
    groupCandSw1.swNodeIds = {sw0, sw1};
    groupCandSw1.isGroup = true;
    cands[sw1].push_back(groupCandSw1);

    // Individual candidate for sw1 at hwMul.
    Candidate indivCandSw1;
    indivCandSw1.hwNodeId = hwMul;
    indivCandSw1.swNodeIds = {sw1};
    indivCandSw1.isGroup = false;
    cands[sw1].push_back(indivCandSw1);

    ConnectivityMatrix cm;
    CPSATSolver solver;
    CPSATSolver::Options opts;
    opts.timeLimitSeconds = 10.0;

    auto result = solver.solveFullProblem(dfg, adg, cands, cm, nullptr, opts);

    TEST_ASSERT(result.success);

    // Verify group atomicity: both SW nodes map to same HW node.
    IdIndex hw0 = result.state.swNodeToHwNode[sw0];
    IdIndex hw1 = result.state.swNodeToHwNode[sw1];
    TEST_ASSERT(hw0 != INVALID_ID);
    TEST_ASSERT(hw1 != INVALID_ID);

    // Key assertion: if one goes to hwMulti, the other must too.
    // (The group constraint ensures they can't be split.)
    if (hw0 == hwMulti) {
      TEST_ASSERT(hw1 == hwMulti);
    }
    if (hw1 == hwMulti) {
      TEST_ASSERT(hw0 == hwMulti);
    }
  }

  // Test 3: CP-SAT sub-problem with group atomicity (requires OR-Tools).
  if (CPSATSolver::isAvailable()) {
    Graph dfg(&ctx);
    Graph adg(&ctx);

    // DFG: sw0(addi) -> sw1(muli), plus sw2(addi) independent.
    IdIndex sw0 = addOpNode(dfg, ctx, "arith.addi", "functional", 2, 1);
    IdIndex sw1 = addOpNode(dfg, ctx, "arith.muli", "functional", 2, 1);
    IdIndex sw2 = addOpNode(dfg, ctx, "arith.addi", "functional", 2, 1);
    addEdgeBetween(dfg, sw0, 0, sw1, 0);

    // ADG: multi-op PE + 2 single PEs.
    auto multiPE = std::make_unique<Node>();
    multiPE->kind = Node::OperationNode;
    setStringAttr(multiPE.get(), ctx, "op_name", "fabric.pe");
    setStringAttr(multiPE.get(), ctx, "resource_class", "functional");
    setArrayStrAttr(multiPE.get(), ctx, "body_ops",
                    {"arith.addi", "arith.muli"});
    setArrayIntAttr(multiPE.get(), ctx, "body_edges", {0, 1});
    IdIndex hwMulti = adg.addNode(std::move(multiPE));
    for (int i = 0; i < 3; ++i) {
      auto port = std::make_unique<Port>();
      port->parentNode = hwMulti;
      port->direction = Port::Input;
      IdIndex pid = adg.addPort(std::move(port));
      adg.getNode(hwMulti)->inputPorts.push_back(pid);
    }
    for (int i = 0; i < 2; ++i) {
      auto port = std::make_unique<Port>();
      port->parentNode = hwMulti;
      port->direction = Port::Output;
      IdIndex pid = adg.addPort(std::move(port));
      adg.getNode(hwMulti)->outputPorts.push_back(pid);
    }

    IdIndex hwAdd = addPENode(adg, ctx, "fabric.pe", "functional", 2, 1);
    IdIndex hwMul = addPENode(adg, ctx, "fabric.pe", "functional", 2, 1);

    // Set up initial mapping: sw2 is already mapped to hwAdd.
    MappingState currentState;
    currentState.init(dfg, adg);
    currentState.mapNode(sw2, hwAdd, dfg, adg);

    // Build candidates for the sub-problem (sw0, sw1).
    CandidateSet cands;

    Candidate gc0;
    gc0.hwNodeId = hwMulti;
    gc0.swNodeIds = {sw0, sw1};
    gc0.isGroup = true;
    cands[sw0].push_back(gc0);

    Candidate ic0;
    ic0.hwNodeId = hwAdd;
    ic0.swNodeIds = {sw0};
    ic0.isGroup = false;
    cands[sw0].push_back(ic0);

    Candidate gc1;
    gc1.hwNodeId = hwMulti;
    gc1.swNodeIds = {sw0, sw1};
    gc1.isGroup = true;
    cands[sw1].push_back(gc1);

    Candidate ic1;
    ic1.hwNodeId = hwMul;
    ic1.swNodeIds = {sw1};
    ic1.isGroup = false;
    cands[sw1].push_back(ic1);

    ConnectivityMatrix cm;
    CPSATSolver solver;
    CPSATSolver::Options opts;
    opts.timeLimitSeconds = 10.0;

    llvm::SmallVector<IdIndex, 2> subNodes = {sw0, sw1};
    auto result = solver.solveSubProblem(dfg, adg, subNodes, currentState,
                                         cands, cm, opts);

    TEST_ASSERT(result.success);

    // Verify group atomicity in sub-problem result.
    IdIndex hw0 = result.state.swNodeToHwNode[sw0];
    IdIndex hw1 = result.state.swNodeToHwNode[sw1];
    TEST_ASSERT(hw0 != INVALID_ID);
    TEST_ASSERT(hw1 != INVALID_ID);

    // Group constraint: if one goes to hwMulti, both must.
    if (hw0 == hwMulti) {
      TEST_ASSERT(hw1 == hwMulti);
    }
    if (hw1 == hwMulti) {
      TEST_ASSERT(hw0 == hwMulti);
    }

    // sw2 should remain mapped to hwAdd (unchanged by sub-problem).
    TEST_ASSERT(result.state.swNodeToHwNode[sw2] == hwAdd);
  }

  return 0;
}
