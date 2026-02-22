//===-- action_log_record.cpp - Action log recording test ----------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Verify that all mapper actions (map/unmap node/port/edge) are recorded
// in the action log with correct types and arguments.
//
//===----------------------------------------------------------------------===//

#include "TestUtil.h"
#include "loom/Mapper/Graph.h"
#include "loom/Mapper/MappingState.h"

using namespace loom;

int main() {
  Graph dfg;
  Graph adg;

  // DFG: 1 node, 2 ports, 1 edge.
  auto sw0 = std::make_unique<Node>();
  sw0->kind = Node::OperationNode;
  IdIndex sw0Id = dfg.addNode(std::move(sw0));

  auto swOut = std::make_unique<Port>();
  swOut->parentNode = sw0Id;
  swOut->direction = Port::Output;
  IdIndex swOutId = dfg.addPort(std::move(swOut));
  dfg.getNode(sw0Id)->outputPorts.push_back(swOutId);

  auto swIn = std::make_unique<Port>();
  swIn->parentNode = sw0Id;
  swIn->direction = Port::Input;
  IdIndex swInId = dfg.addPort(std::move(swIn));
  dfg.getNode(sw0Id)->inputPorts.push_back(swInId);

  auto swEdge = std::make_unique<Edge>();
  swEdge->srcPort = swOutId;
  swEdge->dstPort = swInId;
  IdIndex swEid = dfg.addEdge(std::move(swEdge));
  dfg.getPort(swOutId)->connectedEdges.push_back(swEid);
  dfg.getPort(swInId)->connectedEdges.push_back(swEid);

  // ADG: 1 node, 2 ports, 1 edge.
  auto hw0 = std::make_unique<Node>();
  IdIndex hw0Id = adg.addNode(std::move(hw0));

  auto hwOut = std::make_unique<Port>();
  hwOut->parentNode = hw0Id;
  hwOut->direction = Port::Output;
  IdIndex hwOutId = adg.addPort(std::move(hwOut));
  adg.getNode(hw0Id)->outputPorts.push_back(hwOutId);

  auto hwIn = std::make_unique<Port>();
  hwIn->parentNode = hw0Id;
  hwIn->direction = Port::Input;
  IdIndex hwInId = adg.addPort(std::move(hwIn));
  adg.getNode(hw0Id)->inputPorts.push_back(hwInId);

  auto hwEdge = std::make_unique<Edge>();
  hwEdge->srcPort = hwOutId;
  hwEdge->dstPort = hwInId;
  IdIndex hwEid = adg.addEdge(std::move(hwEdge));
  adg.getPort(hwOutId)->connectedEdges.push_back(hwEid);
  adg.getPort(hwInId)->connectedEdges.push_back(hwEid);

  MappingState state;
  state.init(dfg, adg);

  // Perform actions.
  state.mapNode(sw0Id, hw0Id, dfg, adg);    // 0: MAP_NODE
  state.mapPort(swOutId, hwOutId, dfg, adg); // 1: MAP_PORT
  state.mapPort(swInId, hwInId, dfg, adg);   // 2: MAP_PORT

  llvm::SmallVector<IdIndex, 8> path = {hwOutId, hwInId};
  state.mapEdge(swEid, path, dfg, adg);       // 3: MAP_EDGE
  state.unmapEdge(swEid, dfg, adg);            // 4: UNMAP_EDGE
  state.unmapPort(swOutId, dfg, adg);          // 5: UNMAP_PORT
  state.unmapNode(sw0Id, dfg, adg);            // 6: UNMAP_NODE

  TEST_ASSERT(state.actionLog.size() == 7);

  // Verify action types.
  TEST_ASSERT(state.actionLog[0].type == ActionRecord::MAP_NODE);
  TEST_ASSERT(state.actionLog[1].type == ActionRecord::MAP_PORT);
  TEST_ASSERT(state.actionLog[2].type == ActionRecord::MAP_PORT);
  TEST_ASSERT(state.actionLog[3].type == ActionRecord::MAP_EDGE);
  TEST_ASSERT(state.actionLog[4].type == ActionRecord::UNMAP_EDGE);
  TEST_ASSERT(state.actionLog[5].type == ActionRecord::UNMAP_PORT);
  TEST_ASSERT(state.actionLog[6].type == ActionRecord::UNMAP_NODE);

  // Verify MAP_NODE arguments.
  TEST_ASSERT(state.actionLog[0].arg0 == sw0Id);
  TEST_ASSERT(state.actionLog[0].arg1 == hw0Id);

  // Verify MAP_EDGE path arguments.
  TEST_ASSERT(state.actionLog[3].pathArgs.size() == 2);
  TEST_ASSERT(state.actionLog[3].pathArgs[0] == hwOutId);
  TEST_ASSERT(state.actionLog[3].pathArgs[1] == hwInId);

  // All actions should report success.
  for (size_t i = 0; i < state.actionLog.size(); ++i) {
    TEST_ASSERT(state.actionLog[i].constraintResult == ActionResult::Success);
  }

  return 0;
}
