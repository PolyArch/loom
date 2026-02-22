//===-- routing_direct.cpp - Direct routing test -------------------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Verify that findPath returns a direct 2-element path when source and dest
// are directly connected.
//
//===----------------------------------------------------------------------===//

#include "TestUtil.h"
#include "loom/Mapper/Graph.h"
#include "loom/Mapper/MappingState.h"

using namespace loom;

int main() {
  Graph dfg;
  Graph adg;

  // ADG: 2 nodes with 1 output port and 1 input port, directly connected.
  auto hw0 = std::make_unique<Node>();
  hw0->kind = Node::OperationNode;
  IdIndex hw0Id = adg.addNode(std::move(hw0));

  auto hw1 = std::make_unique<Node>();
  hw1->kind = Node::OperationNode;
  IdIndex hw1Id = adg.addNode(std::move(hw1));

  auto outPort = std::make_unique<Port>();
  outPort->parentNode = hw0Id;
  outPort->direction = Port::Output;
  IdIndex outPid = adg.addPort(std::move(outPort));
  adg.getNode(hw0Id)->outputPorts.push_back(outPid);

  auto inPort = std::make_unique<Port>();
  inPort->parentNode = hw1Id;
  inPort->direction = Port::Input;
  IdIndex inPid = adg.addPort(std::move(inPort));
  adg.getNode(hw1Id)->inputPorts.push_back(inPid);

  auto edge = std::make_unique<Edge>();
  edge->srcPort = outPid;
  edge->dstPort = inPid;
  IdIndex eid = adg.addEdge(std::move(edge));
  adg.getPort(outPid)->connectedEdges.push_back(eid);
  adg.getPort(inPid)->connectedEdges.push_back(eid);

  // DFG: 1 edge to route.
  auto sw0 = std::make_unique<Node>();
  dfg.addNode(std::move(sw0));
  auto sw1 = std::make_unique<Node>();
  dfg.addNode(std::move(sw1));

  auto swOut = std::make_unique<Port>();
  swOut->direction = Port::Output;
  swOut->parentNode = 0;
  IdIndex swOutPid = dfg.addPort(std::move(swOut));
  dfg.getNode(0)->outputPorts.push_back(swOutPid);

  auto swIn = std::make_unique<Port>();
  swIn->direction = Port::Input;
  swIn->parentNode = 1;
  IdIndex swInPid = dfg.addPort(std::move(swIn));
  dfg.getNode(1)->inputPorts.push_back(swInPid);

  auto swEdge = std::make_unique<Edge>();
  swEdge->srcPort = swOutPid;
  swEdge->dstPort = swInPid;
  IdIndex swEid = dfg.addEdge(std::move(swEdge));
  dfg.getPort(swOutPid)->connectedEdges.push_back(swEid);
  dfg.getPort(swInPid)->connectedEdges.push_back(swEid);

  // Setup state and map ports.
  MappingState state;
  state.init(dfg, adg);
  state.mapNode(0, hw0Id, dfg, adg);
  state.mapNode(1, hw1Id, dfg, adg);
  state.mapPort(swOutPid, outPid, dfg, adg);
  state.mapPort(swInPid, inPid, dfg, adg);

  // Route the edge: path should be [outPid, inPid].
  llvm::SmallVector<IdIndex, 8> path = {outPid, inPid};
  auto r = state.mapEdge(swEid, path, dfg, adg);
  TEST_ASSERT(r == ActionResult::Success);
  TEST_ASSERT(state.swEdgeToHwPaths[swEid].size() == 2);
  TEST_ASSERT(state.swEdgeToHwPaths[swEid][0] == outPid);
  TEST_ASSERT(state.swEdgeToHwPaths[swEid][1] == inPid);

  return 0;
}
