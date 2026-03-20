#include "fcc/Mapper/ADGFlattener.h"
#include "ADGFlattenerContext.h"
#include "fcc/Mapper/BridgeBinding.h"
#include "fcc/Mapper/TypeCompat.h"

#include "fcc/Dialect/Fabric/FabricDialect.h"
#include "fcc/Dialect/Fabric/FabricOps.h"
#include "fcc/Dialect/Fabric/FabricTypes.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"

#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <cmath>
#include <regex>

namespace fcc {

// --- Shared helper function implementations ---

std::pair<int, int> parseGridPos(llvm::StringRef name) {
  auto str = name.str();
  std::regex re("_(\\d+)_(\\d+)$");
  std::smatch m;
  if (std::regex_search(str, m, re)) {
    return {std::stoi(m[1].str()), std::stoi(m[2].str())};
  }
  return {-1, -1};
}

std::pair<mlir::ArrayAttr, mlir::ArrayAttr>
extractFUBodyDAG(fcc::fabric::FunctionUnitOp fuOp, mlir::MLIRContext *ctx) {
  llvm::SmallVector<mlir::Attribute, 4> opNames;
  llvm::SmallVector<mlir::Attribute, 4> dagEdges;

  auto &fuBody = fuOp.getBody().front();

  // Map from Value to op index for edge tracking
  llvm::DenseMap<mlir::Value, int> valueToOpIdx;
  // Map block args to index -1 (they are FU inputs, not ops)
  for (auto arg : fuBody.getArguments())
    valueToOpIdx[arg] = -1;

  int opIdx = 0;
  for (auto &bodyOp : fuBody.getOperations()) {
    if (mlir::isa<fcc::fabric::YieldOp>(bodyOp))
      continue;

    std::string opName = bodyOp.getName().getStringRef().str();
    opNames.push_back(mlir::StringAttr::get(ctx, opName));

    // Track which ops this op depends on
    for (auto operand : bodyOp.getOperands()) {
      auto it = valueToOpIdx.find(operand);
      if (it != valueToOpIdx.end() && it->second >= 0) {
        auto edge = mlir::ArrayAttr::get(
            ctx,
            {mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 32),
                                     it->second),
             mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 32), opIdx)});
        dagEdges.push_back(edge);
      }
    }

    // Map this op's results to its index
    for (auto result : bodyOp.getResults())
      valueToOpIdx[result] = opIdx;

    opIdx++;
  }

  return {mlir::ArrayAttr::get(ctx, opNames),
          mlir::ArrayAttr::get(ctx, dagEdges)};
}

mlir::DenseI64ArrayAttr
extractFUConfigFieldWidths(fcc::fabric::FunctionUnitOp fuOp,
                           mlir::MLIRContext *ctx) {
  llvm::SmallVector<int64_t, 4> widths;
  for (mlir::Operation &bodyOp : fuOp.getBody().front().getOperations()) {
    if (mlir::isa<fcc::fabric::YieldOp>(bodyOp))
      continue;
    if (auto muxOp = mlir::dyn_cast<fcc::fabric::MuxOp>(bodyOp)) {
      unsigned numInputs = muxOp.getInputs().size();
      unsigned numResults = muxOp.getResults().size();
      if (numInputs == 1 && numResults == 1)
        continue;
      unsigned branchCount = std::max(numInputs, numResults);
      unsigned selBits = branchCount > 1 ? llvm::Log2_32_Ceil(branchCount) : 0;
      widths.push_back(static_cast<int64_t>(selBits + 2));
      continue;
    }

    llvm::StringRef opName = bodyOp.getName().getStringRef();
    if (opName == "handshake.constant") {
      if (bodyOp.getNumResults() == 0)
        continue;
      if (auto width = detail::getScalarWidth(bodyOp.getResult(0).getType()))
        widths.push_back(static_cast<int64_t>(*width));
      continue;
    }
    if (opName == "arith.cmpi" || opName == "arith.cmpf") {
      widths.push_back(4);
      continue;
    }
    if (opName == "dataflow.stream") {
      widths.push_back(5);
      continue;
    }
    if (opName == "handshake.join") {
      widths.push_back(static_cast<int64_t>(bodyOp.getNumOperands()));
      continue;
    }
  }
  if (widths.empty())
    return {};
  return mlir::DenseI64ArrayAttr::get(ctx, widths);
}

void inferMissingNodeGridPositions(
    Graph &adg, llvm::DenseMap<IdIndex, std::pair<int, int>> &nodeGridPos) {
  auto collectNeighborSamples =
      [&](IdIndex nodeId) -> llvm::SmallVector<std::pair<int, int>, 8> {
    llvm::SmallVector<std::pair<int, int>, 8> samples;
    const Node *node = adg.getNode(nodeId);
    if (!node)
      return samples;

    auto addSamplesForPort = [&](IdIndex portId) {
      const Port *port = adg.getPort(portId);
      if (!port)
        return;
      for (IdIndex edgeId : port->connectedEdges) {
        const Edge *edge = adg.getEdge(edgeId);
        if (!edge)
          continue;
        IdIndex otherPortId = INVALID_ID;
        if (edge->srcPort == portId)
          otherPortId = edge->dstPort;
        else if (edge->dstPort == portId)
          otherPortId = edge->srcPort;
        if (otherPortId == INVALID_ID)
          continue;
        const Port *otherPort = adg.getPort(otherPortId);
        if (!otherPort || otherPort->parentNode == INVALID_ID ||
            otherPort->parentNode == nodeId)
          continue;
        auto it = nodeGridPos.find(otherPort->parentNode);
        if (it == nodeGridPos.end())
          continue;
        samples.push_back(it->second);
      }
    };

    for (IdIndex portId : node->inputPorts)
      addSamplesForPort(portId);
    for (IdIndex portId : node->outputPorts)
      addSamplesForPort(portId);
    return samples;
  };

  bool changed = true;
  while (changed) {
    changed = false;
    for (IdIndex nodeId = 0; nodeId < static_cast<IdIndex>(adg.nodes.size());
         ++nodeId) {
      if (!adg.getNode(nodeId) || nodeGridPos.count(nodeId))
        continue;

      auto samples = collectNeighborSamples(nodeId);
      if (samples.empty())
        continue;

      double rowSum = 0.0;
      double colSum = 0.0;
      for (const auto &[row, col] : samples) {
        rowSum += static_cast<double>(row);
        colSum += static_cast<double>(col);
      }

      int row =
          static_cast<int>(std::lround(rowSum / static_cast<double>(samples.size())));
      int col =
          static_cast<int>(std::lround(colSum / static_cast<double>(samples.size())));
      nodeGridPos[nodeId] = {row, col};
      changed = true;
    }
  }
}

// --- flattenWireEdges: Pass 2 + legacy fallback + module output sentinels ---

void ADGFlattener::flattenWireEdges(FlattenContext &fctx, mlir::Block &body) {
  mlir::MLIRContext *ctx = fctx.ctx;

  // Collect PE info: which PE has which results, and which operands.
  // This handles both direct PE ops and InstanceOp referencing PE defs.

  size_t peIdx = 0;
  for (auto &op : body.getOperations()) {
    bool isPE = false;
    if (auto peOp = mlir::dyn_cast<fcc::fabric::SpatialPEOp>(op)) {
      isPE = peOp->hasAttr("inline_instantiation");
    } else if (auto peOp = mlir::dyn_cast<fcc::fabric::TemporalPEOp>(op)) {
      isPE = peOp->hasAttr("inline_instantiation");
    } else if (auto instOp = mlir::dyn_cast<fcc::fabric::InstanceOp>(op)) {
      isPE = fctx.peDefMap.count(instOp.getModule()) > 0 ||
             fctx.temporalPeDefMap.count(instOp.getModule()) > 0;
    }
    if (isPE) {
      PEWiringInfo info;
      info.op = &op;
      if (peIdx < peContainment.size())
        info.fuNodeIds.assign(peContainment[peIdx].fuNodeIds.begin(),
                              peContainment[peIdx].fuNodeIds.end());
      fctx.peInfos.push_back(info);
      peIdx++;
    }
  }

  // Build map: Value -> vector of output port IDs (for PEs, all FU outputs).

  // Non-PE ops: already mapped in valueToOutputPort.
  for (auto &kv : fctx.valueToOutputPort) {
    fctx.valueSrcPorts[kv.first].push_back({kv.second, -1});
  }

  // PE ops: map each PE result to all FU output ports in that PE.
  for (auto &pi : fctx.peInfos) {
    for (unsigned r = 0; r < pi.op->getNumResults(); ++r) {
      mlir::Value val = pi.op->getResult(r);
      for (IdIndex fuId : pi.fuNodeIds) {
        auto *fuNode = adg.getNode(fuId);
        if (!fuNode)
          continue;
        for (IdIndex op : fuNode->outputPorts) {
          fctx.valueSrcPorts[val].push_back({op, static_cast<int>(r)});
        }
      }
    }
  }

  // Count total FU nodes.
  fctx.totalFuNodes = 0;
  for (auto &pe : peContainment)
    fctx.totalFuNodes += pe.fuNodeIds.size();

  // Wire all single-node operations and instances using the explicit
  // opToNodeId map populated during pass 1.
  for (auto &op : body.getOperations()) {
    auto it = fctx.opToNodeId.find(&op);
    if (it == fctx.opToNodeId.end())
      continue;
    auto *node = adg.getNode(it->second);
    if (!node)
      continue;

    for (unsigned j = 0; j < op.getNumOperands(); ++j) {
      if (j >= node->inputPorts.size())
        break;
      auto srcIt = fctx.valueSrcPorts.find(op.getOperand(j));
      if (srcIt == fctx.valueSrcPorts.end())
        continue;

      IdIndex dstPortId = node->inputPorts[j];
      for (const SourceBinding &binding : srcIt->second) {
        IdIndex srcPortId = binding.portId;
        auto edge = std::make_unique<Edge>();
        edge->srcPort = srcPortId;
        edge->dstPort = dstPortId;
        if (binding.peOutputIndex >= 0) {
          setEdgeAttr(edge.get(), "pe_output_index",
                      mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 32),
                                             binding.peOutputIndex),
                      ctx);
        }
        IdIndex edgeId = adg.addEdge(std::move(edge));
        adg.ports[srcPortId]->connectedEdges.push_back(edgeId);
        adg.ports[dstPortId]->connectedEdges.push_back(edgeId);
        connectivity.outToIn[srcPortId].push_back(dstPortId);
      }
    }
  }

  // Helper: create edges from a set of source ports to a set of dest ports.
  auto createEdgesBetweenPorts =
      [&](llvm::ArrayRef<SourceBinding> srcPorts, llvm::ArrayRef<IdIndex> dstPorts,
          int peInputIndex) {
        for (const SourceBinding &binding : srcPorts) {
          IdIndex srcPortId = binding.portId;
          for (IdIndex dstPortId : dstPorts) {
            auto edge = std::make_unique<Edge>();
            edge->srcPort = srcPortId;
            edge->dstPort = dstPortId;
            if (binding.peOutputIndex >= 0) {
              setEdgeAttr(edge.get(), "pe_output_index",
                          mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 32),
                                                 binding.peOutputIndex),
                          ctx);
            }
            if (peInputIndex >= 0) {
              setEdgeAttr(edge.get(), "pe_input_index",
                          mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 32),
                                                 peInputIndex),
                          ctx);
            }
            IdIndex edgeId = adg.addEdge(std::move(edge));
            adg.ports[srcPortId]->connectedEdges.push_back(edgeId);
            adg.ports[dstPortId]->connectedEdges.push_back(edgeId);
            connectivity.outToIn[srcPortId].push_back(dstPortId);
          }
        }
      };

  // Wire PE instance operands based on SSA def-use chains.
  // Each PE instance has operands from its local switch. We create edges
  // from the producer's output port(s) to ALL FU input ports in that PE
  // (since any FU inside the PE can consume data arriving at a PE port).
  for (auto &pi : fctx.peInfos) {
    mlir::Operation *peOp = pi.op;

    for (unsigned j = 0; j < peOp->getNumOperands(); ++j) {
      mlir::Value operand = peOp->getOperand(j);

      auto srcIt = fctx.valueSrcPorts.find(operand);
      if (srcIt == fctx.valueSrcPorts.end())
        continue;

      // The destination is all FU input ports in this PE.
      for (IdIndex fuId : pi.fuNodeIds) {
        auto *fuNode = adg.getNode(fuId);
        if (!fuNode)
          continue;
        createEdgesBetweenPorts(srcIt->second, fuNode->inputPorts,
                                static_cast<int>(j));
      }
    }
  }

  // Legacy fallback: wire SW output ports -> ExtMem input ports from metadata.
  // Newer ADGs spell these edges with real SSA operands on fabric.extmemory,
  // so this block only runs when the data input ports still have no producer.
  {
    // Build a map from instance name -> SW ADG node ID.
    llvm::DenseMap<llvm::StringRef, IdIndex> swNameToNodeId;
    for (IdIndex nid = 0; nid < static_cast<IdIndex>(adg.nodes.size()); ++nid) {
      const Node *n = adg.getNode(nid);
      if (!n)
        continue;
      if (getNodeAttrStr(n, "resource_class") == "routing") {
        swNameToNodeId[getNodeAttrStr(n, "op_name")] = nid;
      }
    }

    // For each ExtMem node, find connected SW nodes and create edges.
    for (IdIndex nid = 0; nid < static_cast<IdIndex>(adg.nodes.size()); ++nid) {
      const Node *n = adg.getNode(nid);
      if (!n)
        continue;
      if (getNodeAttrStr(n, "resource_class") != "memory")
        continue;

      bool hasStructuredInputs = false;
      for (unsigned inIdx = 1; inIdx < n->inputPorts.size(); ++inIdx) {
        const Port *inPort = adg.getPort(n->inputPorts[inIdx]);
        if (inPort && !inPort->connectedEdges.empty()) {
          hasStructuredInputs = true;
          break;
        }
      }
      if (hasStructuredInputs)
        continue;

      auto connectSwitchToMemory = [&](llvm::StringRef swName,
                                       unsigned swOutputBase) {
        auto swIt = swNameToNodeId.find(swName);
        if (swIt == swNameToNodeId.end())
          return;
        IdIndex swNodeId = swIt->second;
        auto *swNode = adg.getNode(swNodeId);
        if (!swNode)
          return;

        unsigned numExtMemDataInputs = n->inputPorts.size() > 1
                                           ? n->inputPorts.size() - 1
                                           : 0;
        for (unsigned p = 0; p < numExtMemDataInputs; ++p) {
          unsigned swOutIdx = swOutputBase + p;
          unsigned extMemInIdx = 1 + p; // Skip memref port at 0.
          if (swOutIdx >= swNode->outputPorts.size() ||
              extMemInIdx >= n->inputPorts.size())
            break;

          IdIndex srcPortId = swNode->outputPorts[swOutIdx];
          IdIndex dstPortId = n->inputPorts[extMemInIdx];
          connectivity.outToIn[srcPortId].push_back(dstPortId);

          auto edge = std::make_unique<Edge>();
          edge->srcPort = srcPortId;
          edge->dstPort = dstPortId;
          IdIndex edgeId = adg.addEdge(std::move(edge));
          adg.ports[srcPortId]->connectedEdges.push_back(edgeId);
          adg.ports[dstPortId]->connectedEdges.push_back(edgeId);
        }
      };

      bool usedDetailedMetadata = false;
      for (auto &attr : n->attributes) {
        if (attr.getName() != "connected_sw_detail")
          continue;
        auto arrayAttr = mlir::dyn_cast<mlir::ArrayAttr>(attr.getValue());
        if (!arrayAttr)
          continue;
        usedDetailedMetadata = true;
        for (auto elem : arrayAttr) {
          auto dictAttr = mlir::dyn_cast<mlir::DictionaryAttr>(elem);
          if (!dictAttr)
            continue;
          auto nameAttr = dictAttr.getAs<mlir::StringAttr>("name");
          if (!nameAttr)
            continue;
          unsigned swOutputBase = 0;
          if (auto outBaseAttr =
                  dictAttr.getAs<mlir::IntegerAttr>("output_port_base")) {
            swOutputBase = static_cast<unsigned>(outBaseAttr.getInt());
          }
          connectSwitchToMemory(nameAttr.getValue(), swOutputBase);
        }
      }

      if (usedDetailedMetadata)
        continue;

      // Legacy fallback for older ADGs that only recorded switch names.
      for (auto &attr : n->attributes) {
        if (attr.getName() != "connected_sw")
          continue;
        auto arrayAttr = mlir::dyn_cast<mlir::ArrayAttr>(attr.getValue());
        if (!arrayAttr)
          continue;
        for (auto elem : arrayAttr) {
          auto strAttr = mlir::dyn_cast<mlir::StringAttr>(elem);
          if (!strAttr)
            continue;
          connectSwitchToMemory(strAttr.getValue(), /*swOutputBase=*/4);
        }
      }
    }
  }

  // Create module output sentinel nodes from the fabric.yield terminator.
  for (auto &op : body.getOperations()) {
    auto yieldOp = mlir::dyn_cast<fcc::fabric::YieldOp>(op);
    if (!yieldOp)
      continue;

    for (unsigned i = 0; i < yieldOp.getNumOperands(); ++i) {
      auto outputNode = std::make_unique<Node>();
      outputNode->kind = Node::ModuleOutputNode;

      setNodeAttr(outputNode.get(), "op_name",
                  mlir::StringAttr::get(ctx, "module_output"), ctx);
      setNodeAttr(outputNode.get(), "result_index",
                  mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 32), i),
                  ctx);
      setNodeAttr(outputNode.get(), "resource_class",
                  mlir::StringAttr::get(ctx, "boundary"), ctx);

      // Create an input port for this boundary output.
      auto port = std::make_unique<Port>();
      port->direction = Port::Input;
      port->type = yieldOp.getOperand(i).getType();
      IdIndex portId = adg.addPort(std::move(port));
      adg.ports[portId]->parentNode = static_cast<IdIndex>(adg.nodes.size());
      outputNode->inputPorts.push_back(portId);

      IdIndex nodeId = adg.addNode(std::move(outputNode));

      // Wire from the yield operand source.
      mlir::Value yieldOperand = yieldOp.getOperand(i);
      auto srcIt = fctx.valueSrcPorts.find(yieldOperand);
      if (srcIt != fctx.valueSrcPorts.end()) {
        for (const SourceBinding &binding : srcIt->second) {
          IdIndex srcPortId = binding.portId;
          auto edge = std::make_unique<Edge>();
          edge->srcPort = srcPortId;
          edge->dstPort = portId;
          if (binding.peOutputIndex >= 0) {
            setEdgeAttr(edge.get(), "pe_output_index",
                        mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 32),
                                               binding.peOutputIndex),
                        ctx);
          }
          IdIndex edgeId = adg.addEdge(std::move(edge));
          adg.ports[srcPortId]->connectedEdges.push_back(edgeId);
          adg.ports[portId]->connectedEdges.push_back(edgeId);
          connectivity.outToIn[srcPortId].push_back(portId);
        }
      }
    }
  }

  // Boundary sentinel edges are already wired via SSA:
  // - Input sentinels: their output ports were registered in valueSrcPorts
  //   for block arguments, so SW/PE instances that consume those values
  //   already have edges from the sentinel to their input ports.
  // - Output sentinels: wired above from yield operands via valueSrcPorts.
}

// --- flattenAnalyzeBridges: bridge port analysis for multi-lane memory ---

void ADGFlattener::flattenAnalyzeBridges(FlattenContext &fctx) {
  mlir::MLIRContext *ctx = fctx.ctx;
  mlir::Builder builder(ctx);

  auto findFeedingPort = [&](IdIndex portId) -> IdIndex {
    const Port *p = adg.getPort(portId);
    if (!p)
      return INVALID_ID;
    for (IdIndex edgeId : p->connectedEdges) {
      const Edge *e = adg.getEdge(edgeId);
      if (e && e->dstPort == portId)
        return e->srcPort;
    }
    return INVALID_ID;
  };

  auto findConsumingPort = [&](IdIndex portId) -> IdIndex {
    const Port *p = adg.getPort(portId);
    if (!p)
      return INVALID_ID;
    for (IdIndex edgeId : p->connectedEdges) {
      const Edge *e = adg.getEdge(edgeId);
      if (e && e->srcPort == portId)
        return e->dstPort;
    }
    return INVALID_ID;
  };

  auto findConsumingPorts = [&](IdIndex portId)
      -> llvm::SmallVector<IdIndex, 4> {
    llvm::SmallVector<IdIndex, 4> consumers;
    const Port *p = adg.getPort(portId);
    if (!p)
      return consumers;
    for (IdIndex edgeId : p->connectedEdges) {
      const Edge *e = adg.getEdge(edgeId);
      if (e && e->srcPort == portId)
        consumers.push_back(e->dstPort);
    }
    return consumers;
  };

  auto getPortOwnerKind = [&](IdIndex portId) -> llvm::StringRef {
    const Port *p = adg.getPort(portId);
    if (!p)
      return "";
    const Node *n = adg.getNode(p->parentNode);
    if (!n)
      return "";
    return getNodeAttrStr(n, "op_kind");
  };

  for (IdIndex nodeId = 0; nodeId < static_cast<IdIndex>(adg.nodes.size());
       ++nodeId) {
    Node *node = adg.getNode(nodeId);
    if (!node || node->kind != Node::OperationNode)
      continue;
    if (getNodeAttrStr(node, "resource_class") != "memory")
      continue;

    unsigned ldCount =
        static_cast<unsigned>(getNodeAttrInt(node, "ldCount", 0));
    unsigned stCount =
        static_cast<unsigned>(getNodeAttrInt(node, "stCount", 0));
    bool isExtMem = getNodeAttrStr(node, "op_kind") == "extmemory";

    if (ldCount <= 1 && stCount <= 1)
      continue;

    auto traceInputBridge =
        [&](unsigned memInputPortIdx, unsigned laneCount,
            llvm::SmallVectorImpl<IdIndex> &addTagNodes,
            IdIndex &muxNodeId) -> llvm::SmallVector<IdIndex, 4> {
      muxNodeId = INVALID_ID;
      llvm::SmallVector<IdIndex, 4> boundary;
      if (memInputPortIdx >= node->inputPorts.size())
        return boundary;

      IdIndex memInPortId = node->inputPorts[memInputPortIdx];
      IdIndex srcPortId = findFeedingPort(memInPortId);
      if (srcPortId == INVALID_ID)
        return boundary;

      llvm::DenseSet<IdIndex> seenOutputs;
      llvm::DenseSet<IdIndex> seenBoundaryPorts;
      llvm::DenseSet<IdIndex> seenRouteNodes;
      llvm::SmallVector<std::pair<IdIndex, IdIndex>, 4> boundaryPairs;

      std::function<bool(IdIndex)> visitOutputPort = [&](IdIndex outPortId) {
        if (outPortId == INVALID_ID || !seenOutputs.insert(outPortId).second)
          return false;
        const Port *outPort = adg.getPort(outPortId);
        if (!outPort || outPort->direction != Port::Output)
          return false;
        const Node *owner = adg.getNode(outPort->parentNode);
        if (!owner)
          return false;

        llvm::StringRef kind = getNodeAttrStr(owner, "op_kind");
        if (kind == "add_tag") {
          if (owner->inputPorts.empty())
            return false;
          IdIndex boundaryPort = owner->inputPorts[0];
          bool inserted = seenBoundaryPorts.insert(boundaryPort).second;
          if (inserted)
            boundaryPairs.push_back({boundaryPort, outPort->parentNode});
          return inserted;
        }

        if (kind == "map_tag" || kind == "del_tag" || kind == "fifo") {
          if (owner->inputPorts.empty())
            return false;
          return visitOutputPort(findFeedingPort(owner->inputPorts[0]));
        }

        if (kind == "spatial_sw" || kind == "temporal_sw") {
          if (seenRouteNodes.insert(outPort->parentNode).second &&
              muxNodeId == INVALID_ID) {
            muxNodeId = outPort->parentNode;
          }
          bool foundBoundary = false;
          for (IdIndex inPortId : owner->inputPorts) {
            IdIndex upstreamOutPortId = findFeedingPort(inPortId);
            if (visitOutputPort(upstreamOutPortId)) {
              foundBoundary = true;
              continue;
            }
            const Port *inPort = adg.getPort(inPortId);
            if (inPort && mlir::isa<fcc::fabric::TaggedType>(inPort->type) &&
                seenBoundaryPorts.insert(inPortId).second) {
              boundaryPairs.push_back({inPortId, INVALID_ID});
              foundBoundary = true;
            }
          }
          return foundBoundary;
        }

        if (mlir::isa<fcc::fabric::TaggedType>(outPort->type)) {
          bool inserted = seenBoundaryPorts.insert(outPortId).second;
          if (inserted)
            boundaryPairs.push_back({outPortId, INVALID_ID});
          return inserted;
        }
        return false;
      };

      visitOutputPort(srcPortId);
      llvm::sort(boundaryPairs,
                 [](const auto &lhs, const auto &rhs) {
                   return lhs.first < rhs.first;
                 });
      if (boundaryPairs.size() > laneCount)
        boundaryPairs.resize(laneCount);
      for (const auto &entry : boundaryPairs) {
        boundary.push_back(entry.first);
        addTagNodes.push_back(entry.second);
      }
      return boundary;
    };

    auto traceOutputBridge =
        [&](unsigned memOutputPortIdx, unsigned laneCount,
            IdIndex &demuxNodeId) -> llvm::SmallVector<IdIndex, 4> {
      demuxNodeId = INVALID_ID;
      llvm::SmallVector<IdIndex, 4> boundary;
      if (memOutputPortIdx >= node->outputPorts.size())
        return boundary;

      IdIndex memOutPortId = node->outputPorts[memOutputPortIdx];
      llvm::DenseSet<IdIndex> seenInputs;
      llvm::DenseSet<IdIndex> seenBoundaryPorts;
      llvm::DenseSet<IdIndex> seenRouteNodes;

      std::function<bool(IdIndex)> visitInputPort = [&](IdIndex inPortId) {
        if (inPortId == INVALID_ID || !seenInputs.insert(inPortId).second)
          return false;
        const Port *inPort = adg.getPort(inPortId);
        if (!inPort || inPort->direction != Port::Input)
          return false;
        const Node *owner = adg.getNode(inPort->parentNode);
        if (!owner)
          return false;

        llvm::StringRef kind = getNodeAttrStr(owner, "op_kind");
        if (kind == "del_tag") {
          if (!owner->outputPorts.empty()) {
            IdIndex boundaryPort = owner->outputPorts[0];
            bool inserted = seenBoundaryPorts.insert(boundaryPort).second;
            if (inserted)
              boundary.push_back(boundaryPort);
            return inserted;
          }
          return false;
        }

        if (kind == "add_tag" || kind == "map_tag" || kind == "fifo") {
          bool foundBoundary = false;
          for (IdIndex outPortId : owner->outputPorts) {
            for (IdIndex nextInPortId : findConsumingPorts(outPortId))
              foundBoundary |= visitInputPort(nextInPortId);
          }
          return foundBoundary;
        }

        if (kind == "spatial_sw" || kind == "temporal_sw") {
          if (seenRouteNodes.insert(inPort->parentNode).second &&
              demuxNodeId == INVALID_ID) {
            demuxNodeId = inPort->parentNode;
          }
          bool foundBoundary = false;
          for (IdIndex outPortId : owner->outputPorts) {
            bool branchFound = false;
            for (IdIndex nextInPortId : findConsumingPorts(outPortId))
              branchFound |= visitInputPort(nextInPortId);
            if (branchFound) {
              foundBoundary = true;
              continue;
            }
            const Port *outPort = adg.getPort(outPortId);
            if (outPort && mlir::isa<fcc::fabric::TaggedType>(outPort->type) &&
                seenBoundaryPorts.insert(outPortId).second) {
              boundary.push_back(outPortId);
              foundBoundary = true;
            }
          }
          return foundBoundary;
        }

        if (mlir::isa<fcc::fabric::TaggedType>(inPort->type)) {
          bool inserted = seenBoundaryPorts.insert(inPortId).second;
          if (inserted)
            boundary.push_back(inPortId);
          return inserted;
        }
        return false;
      };

      for (IdIndex dstPortId : findConsumingPorts(memOutPortId))
        visitInputPort(dstPortId);

      llvm::sort(boundary);
      if (boundary.size() > laneCount)
        boundary.resize(laneCount);
      return boundary;
    };

    llvm::SmallVector<IdIndex, 8> bridgeInputPorts;
    llvm::SmallVector<BridgePortCategory, 8> bridgeInputCats;
    llvm::SmallVector<unsigned, 8> bridgeInputLanes;
    llvm::SmallVector<IdIndex, 8> allAddTagNodes;
    llvm::SmallVector<IdIndex, 4> muxNodes;
    unsigned memInIdx = isExtMem ? 1 : 0;

    llvm::SmallVector<IdIndex, 4> ldAddrBoundary, stAddrBoundary,
        stDataBoundary;
    llvm::SmallVector<IdIndex, 4> ldAddrNodes, stAddrNodes, stDataNodes;
    if (ldCount > 0) {
      IdIndex muxId;
      ldAddrBoundary =
          traceInputBridge(memInIdx++, ldCount, ldAddrNodes, muxId);
      if (muxId != INVALID_ID)
        muxNodes.push_back(muxId);
    }
    if (stCount > 0) {
      IdIndex muxId;
      stAddrBoundary =
          traceInputBridge(memInIdx++, stCount, stAddrNodes, muxId);
      if (muxId != INVALID_ID)
        muxNodes.push_back(muxId);
      stDataBoundary =
          traceInputBridge(memInIdx++, stCount, stDataNodes, muxId);
      if (muxId != INVALID_ID)
        muxNodes.push_back(muxId);

      for (unsigned lane = 0; lane < stCount; ++lane) {
        if (lane < stDataBoundary.size()) {
          bridgeInputPorts.push_back(stDataBoundary[lane]);
          bridgeInputCats.push_back(BridgePortCategory::StData);
          bridgeInputLanes.push_back(lane);
        }
        if (lane < stAddrBoundary.size()) {
          bridgeInputPorts.push_back(stAddrBoundary[lane]);
          bridgeInputCats.push_back(BridgePortCategory::StAddr);
          bridgeInputLanes.push_back(lane);
        }
        if (lane < stDataNodes.size())
          allAddTagNodes.push_back(stDataNodes[lane]);
        if (lane < stAddrNodes.size())
          allAddTagNodes.push_back(stAddrNodes[lane]);
      }
    }
    for (unsigned lane = 0; lane < ldCount && lane < ldAddrBoundary.size();
         ++lane) {
      bridgeInputPorts.push_back(ldAddrBoundary[lane]);
      bridgeInputCats.push_back(BridgePortCategory::LdAddr);
      bridgeInputLanes.push_back(lane);
    }
    for (unsigned lane = 0; lane < ldCount && lane < ldAddrNodes.size();
         ++lane)
      allAddTagNodes.push_back(ldAddrNodes[lane]);

    llvm::SmallVector<IdIndex, 8> bridgeOutputPorts;
    llvm::SmallVector<BridgePortCategory, 8> bridgeOutputCats;
    llvm::SmallVector<unsigned, 8> bridgeOutputLanes;
    llvm::SmallVector<IdIndex, 4> demuxNodes;
    unsigned memOutIdx = 0;

    if (ldCount > 0) {
      IdIndex demuxId;
      auto ldDataBoundary =
          traceOutputBridge(memOutIdx++, ldCount, demuxId);
      if (demuxId != INVALID_ID)
        demuxNodes.push_back(demuxId);
      for (unsigned lane = 0; lane < ldCount && lane < ldDataBoundary.size();
           ++lane) {
        bridgeOutputPorts.push_back(ldDataBoundary[lane]);
        bridgeOutputCats.push_back(BridgePortCategory::LdData);
        bridgeOutputLanes.push_back(lane);
      }

      auto ldDoneBoundary =
          traceOutputBridge(memOutIdx++, ldCount, demuxId);
      if (demuxId != INVALID_ID)
        demuxNodes.push_back(demuxId);
      for (unsigned lane = 0; lane < ldCount && lane < ldDoneBoundary.size();
           ++lane) {
        bridgeOutputPorts.push_back(ldDoneBoundary[lane]);
        bridgeOutputCats.push_back(BridgePortCategory::LdDone);
        bridgeOutputLanes.push_back(lane);
      }
    }
    if (stCount > 0) {
      IdIndex demuxId;
      auto stDoneBoundary =
          traceOutputBridge(memOutIdx++, stCount, demuxId);
      if (demuxId != INVALID_ID)
        demuxNodes.push_back(demuxId);
      for (unsigned lane = 0; lane < stCount && lane < stDoneBoundary.size();
           ++lane) {
        bridgeOutputPorts.push_back(stDoneBoundary[lane]);
        bridgeOutputCats.push_back(BridgePortCategory::StDone);
        bridgeOutputLanes.push_back(lane);
      }
    }

    if (bridgeInputPorts.empty() && bridgeOutputPorts.empty()) {
      continue;
    }

    {
      unsigned idx = 0;
      for (unsigned lane = 0; lane < stCount && idx < allAddTagNodes.size();
           ++lane) {
        if (idx < allAddTagNodes.size()) {
          Node *atNode = adg.getNode(allAddTagNodes[idx++]);
          if (atNode) {
            atNode->attributes.push_back(builder.getNamedAttr(
                "bridge_lane_index", builder.getI32IntegerAttr(lane)));
          }
        }
        if (idx < allAddTagNodes.size()) {
          Node *atNode = adg.getNode(allAddTagNodes[idx++]);
          if (atNode) {
            atNode->attributes.push_back(builder.getNamedAttr(
                "bridge_lane_index", builder.getI32IntegerAttr(lane)));
          }
        }
      }
      for (unsigned lane = 0; lane < ldCount && idx < allAddTagNodes.size();
           ++lane) {
        Node *atNode = adg.getNode(allAddTagNodes[idx++]);
        if (atNode) {
          atNode->attributes.push_back(builder.getNamedAttr(
              "bridge_lane_index", builder.getI32IntegerAttr(lane)));
        }
      }
    }

    llvm::SmallVector<int32_t> inPorts32(bridgeInputPorts.begin(),
                                         bridgeInputPorts.end());
    llvm::SmallVector<int32_t> inCats32;
    llvm::SmallVector<int32_t> inLanes32;
    for (auto cat : bridgeInputCats)
      inCats32.push_back(static_cast<int32_t>(cat));
    for (unsigned lane : bridgeInputLanes)
      inLanes32.push_back(static_cast<int32_t>(lane));
    node->attributes.push_back(builder.getNamedAttr(
        "bridge_input_ports",
        mlir::DenseI32ArrayAttr::get(builder.getContext(), inPorts32)));
    node->attributes.push_back(builder.getNamedAttr(
        "bridge_input_categories",
        mlir::DenseI32ArrayAttr::get(builder.getContext(), inCats32)));
    node->attributes.push_back(builder.getNamedAttr(
        "bridge_input_lanes",
        mlir::DenseI32ArrayAttr::get(builder.getContext(), inLanes32)));

    llvm::SmallVector<int32_t> outPorts32(bridgeOutputPorts.begin(),
                                          bridgeOutputPorts.end());
    llvm::SmallVector<int32_t> outCats32;
    llvm::SmallVector<int32_t> outLanes32;
    for (auto cat : bridgeOutputCats)
      outCats32.push_back(static_cast<int32_t>(cat));
    for (unsigned lane : bridgeOutputLanes)
      outLanes32.push_back(static_cast<int32_t>(lane));
    node->attributes.push_back(builder.getNamedAttr(
        "bridge_output_ports",
        mlir::DenseI32ArrayAttr::get(builder.getContext(), outPorts32)));
    node->attributes.push_back(builder.getNamedAttr(
        "bridge_output_categories",
        mlir::DenseI32ArrayAttr::get(builder.getContext(), outCats32)));
    node->attributes.push_back(builder.getNamedAttr(
        "bridge_output_lanes",
        mlir::DenseI32ArrayAttr::get(builder.getContext(), outLanes32)));

    if (!muxNodes.empty()) {
      llvm::SmallVector<int32_t> muxIds(muxNodes.begin(), muxNodes.end());
      node->attributes.push_back(builder.getNamedAttr(
          "bridge_mux_nodes",
          mlir::DenseI32ArrayAttr::get(builder.getContext(), muxIds)));
    }
    if (!demuxNodes.empty()) {
      llvm::SmallVector<int32_t> demuxIds(demuxNodes.begin(),
                                          demuxNodes.end());
      node->attributes.push_back(builder.getNamedAttr(
          "bridge_demux_nodes",
          mlir::DenseI32ArrayAttr::get(builder.getContext(), demuxIds)));
    }
  }
}

} // namespace fcc
