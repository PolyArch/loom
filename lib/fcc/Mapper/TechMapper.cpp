#include "TechMapperInternal.h"
#include "fcc/Mapper/TypeCompat.h"

#include "fcc/Dialect/Fabric/FabricOps.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <map>
#include <string>
#include <utility>

namespace fcc {

using techmapper_detail::findFunctionUnitNode;
using techmapper_detail::collectVariantsForFU;
using techmapper_detail::findMatchesForFamily;
using techmapper_detail::Match;
using techmapper_detail::VariantFamily;

namespace {

void addNodeAttr(Node *node, llvm::StringRef key, mlir::Attribute value,
                 mlir::MLIRContext *ctx) {
  node->attributes.push_back(
      mlir::NamedAttribute(mlir::StringAttr::get(ctx, key), value));
}

std::unique_ptr<Node> cloneNodeShell(const Node *src, mlir::MLIRContext *ctx) {
  auto node = std::make_unique<Node>();
  node->kind = src->kind;
  node->attributes = src->attributes;
  (void)ctx;
  return node;
}

std::unique_ptr<Port> clonePort(const Port *src) {
  auto port = std::make_unique<Port>();
  port->direction = src->direction;
  port->type = src->type;
  port->attributes = src->attributes;
  return port;
}

} // namespace

bool TechMapper::buildPlan(const Graph &dfg, mlir::ModuleOp adgModule,
                           const Graph &adg, Plan &plan) {
  Plan freshPlan;
  plan = std::move(freshPlan);
  plan.originalNodeToContractedNode.assign(dfg.nodes.size(), INVALID_ID);
  plan.originalPortToContractedPort.assign(dfg.ports.size(), INVALID_ID);
  plan.originalEdgeToContractedEdge.assign(dfg.edges.size(), INVALID_ID);
  plan.originalEdgeKinds.assign(dfg.edges.size(), TechMappedEdgeKind::Routed);
  plan.contractedDFG = Graph(dfg.context);
  plan.coverageScore = 1.0;

  llvm::SmallVector<VariantFamily, 16> familyList;

  llvm::DenseMap<mlir::Block *, llvm::DenseSet<llvm::StringRef>>
      referencedTargetsByBlock;
  adgModule.walk([&](fcc::fabric::InstanceOp instOp) {
    referencedTargetsByBlock[instOp->getBlock()].insert(instOp.getModule());
  });

  auto isDefinitionOp = [&](mlir::Operation *op,
                            llvm::StringRef name) -> bool {
    if (mlir::isa<fcc::fabric::FunctionUnitOp>(op))
      return true;
    if (!mlir::isa<fcc::fabric::SpatialPEOp, fcc::fabric::TemporalPEOp>(op))
      return false;
    return !op->hasAttr("inline_instantiation");
  };

  llvm::StringMap<fcc::fabric::SpatialPEOp> peDefs;
  llvm::StringMap<fcc::fabric::TemporalPEOp> temporalPeDefs;
  llvm::StringMap<fcc::fabric::FunctionUnitOp> functionUnitDefs;
  adgModule->walk([&](fcc::fabric::SpatialPEOp peOp) {
    if (auto symAttr = peOp.getSymNameAttr();
        symAttr && isDefinitionOp(peOp.getOperation(), symAttr.getValue()))
      peDefs[symAttr.getValue()] = peOp;
  });
  adgModule->walk([&](fcc::fabric::TemporalPEOp peOp) {
    if (auto symAttr = peOp.getSymNameAttr();
        symAttr && isDefinitionOp(peOp.getOperation(), symAttr.getValue()))
      temporalPeDefs[symAttr.getValue()] = peOp;
  });
  adgModule->walk([&](fcc::fabric::FunctionUnitOp fuOp) {
    auto symName = fuOp.getSymNameAttr().getValue();
    if (isDefinitionOp(fuOp.getOperation(), symName))
      functionUnitDefs[symName] = fuOp;
  });

  if (auto fabricMod = [&]() -> fcc::fabric::ModuleOp {
        fcc::fabric::ModuleOp found;
        adgModule->walk([&](fcc::fabric::ModuleOp op) {
          if (!found)
            found = op;
        });
        return found;
      }()) {
    auto visitPEFunctionUnits =
        [&](auto peOp, llvm::StringRef peName,
            auto &&visitor) {
          auto &peBody = peOp.getBody().front();
          auto referencedIt = referencedTargetsByBlock.find(&peBody);
          const llvm::DenseSet<llvm::StringRef> *referencedTargets =
              referencedIt != referencedTargetsByBlock.end()
                  ? &referencedIt->second
                  : nullptr;
          for (mlir::Operation &bodyOp : peBody.getOperations()) {
            if (auto fuOp = mlir::dyn_cast<fcc::fabric::FunctionUnitOp>(bodyOp)) {
              llvm::StringRef symName = fuOp.getSymNameAttr().getValue();
              if (!symName.empty() && referencedTargets &&
                  referencedTargets->contains(symName))
                continue;
              visitor(fuOp, symName.str());
              continue;
            }
            auto instOp = mlir::dyn_cast<fcc::fabric::InstanceOp>(bodyOp);
            if (!instOp)
              continue;
            auto fuIt = functionUnitDefs.find(instOp.getModule());
            if (fuIt == functionUnitDefs.end())
              continue;
            visitor(fuIt->second,
                    instOp.getSymName().value_or(instOp.getModule()).str());
          }
        };

    auto recordVariantsForPE = [&](llvm::StringRef peName,
                                   fcc::fabric::FunctionUnitOp fuOp,
                                   const std::string &fuName) {
      IdIndex hwNodeId = findFunctionUnitNode(adg, peName, fuName);
      if (hwNodeId == INVALID_ID)
        return;
      const Node *hwNode = adg.getNode(hwNodeId);
      llvm::SmallVector<VariantFamily, 8> variants;
      collectVariantsForFU(fuOp, hwNode, variants);
      for (auto &variant : variants) {
        bool merged = false;
        for (auto &family : familyList) {
          if (family.signature != variant.signature)
            continue;
          family.hwNodeIds.push_back(hwNodeId);
          merged = true;
          break;
        }
        if (!merged) {
          variant.hwNodeIds.clear();
          variant.hwNodeIds.push_back(hwNodeId);
          familyList.push_back(std::move(variant));
        }
      }
    };

    for (mlir::Operation &op : fabricMod.getBody().front().getOperations()) {
      if (auto instOp = mlir::dyn_cast<fcc::fabric::InstanceOp>(op)) {
        std::string peName =
            instOp.getSymName().value_or(instOp.getModule()).str();
        auto peIt = peDefs.find(instOp.getModule());
        if (peIt != peDefs.end()) {
          visitPEFunctionUnits(
              peIt->second, peName,
              [&](fcc::fabric::FunctionUnitOp fuOp, const std::string &fuName) {
                recordVariantsForPE(peName, fuOp, fuName);
              });
          continue;
        }
        auto temporalPeIt = temporalPeDefs.find(instOp.getModule());
        if (temporalPeIt != temporalPeDefs.end()) {
          visitPEFunctionUnits(
              temporalPeIt->second, peName,
              [&](fcc::fabric::FunctionUnitOp fuOp, const std::string &fuName) {
                recordVariantsForPE(peName, fuOp, fuName);
              });
        }
        continue;
      }

      if (auto peOp = mlir::dyn_cast<fcc::fabric::SpatialPEOp>(op)) {
        llvm::StringRef peName = peOp.getSymName().value_or("");
        visitPEFunctionUnits(
            peOp, peName,
            [&](fcc::fabric::FunctionUnitOp fuOp, const std::string &fuName) {
              recordVariantsForPE(peName, fuOp, fuName);
            });
        continue;
      }

      if (auto peOp = mlir::dyn_cast<fcc::fabric::TemporalPEOp>(op)) {
        llvm::StringRef peName = peOp.getSymName().value_or("");
        visitPEFunctionUnits(
            peOp, peName,
            [&](fcc::fabric::FunctionUnitOp fuOp, const std::string &fuName) {
              recordVariantsForPE(peName, fuOp, fuName);
            });
      }
    }
  }

  for (auto &family : familyList) {
    std::sort(family.hwNodeIds.begin(), family.hwNodeIds.end());
    family.hwNodeIds.erase(
        std::unique(family.hwNodeIds.begin(), family.hwNodeIds.end()),
        family.hwNodeIds.end());
  }

  std::vector<Match> allMatches;
  unsigned techNodeCount = 0;
  unsigned totalOpCount = 0;
  for (const Node *swNode : dfg.nodeRange()) {
    if (swNode && swNode->kind == Node::OperationNode)
      ++totalOpCount;
  }
  for (unsigned familyIndex = 0; familyIndex < familyList.size(); ++familyIndex) {
    if (!familyList[familyIndex].isTechFamily())
      continue;
    auto matches = findMatchesForFamily(dfg, familyList[familyIndex], familyIndex);
    allMatches.insert(allMatches.end(), matches.begin(), matches.end());
  }

  std::sort(allMatches.begin(), allMatches.end(),
            [&](const Match &lhs, const Match &rhs) {
              if (lhs.swNodesByOp.size() != rhs.swNodesByOp.size())
                return lhs.swNodesByOp.size() > rhs.swNodesByOp.size();
              size_t lhsHw = familyList[lhs.familyIndex].hwNodeIds.size();
              size_t rhsHw = familyList[rhs.familyIndex].hwNodeIds.size();
              if (lhsHw != rhsHw)
                return lhsHw > rhsHw;
              return std::lexicographical_compare(
                  lhs.swNodesByOp.begin(), lhs.swNodesByOp.end(),
                  rhs.swNodesByOp.begin(), rhs.swNodesByOp.end());
            });

  std::vector<int> nodeToUnit(dfg.nodes.size(), -1);
  for (const Match &match : allMatches) {
    bool overlaps = false;
    for (IdIndex swNodeId : match.swNodesByOp) {
      if (swNodeId >= nodeToUnit.size() || nodeToUnit[swNodeId] >= 0) {
        overlaps = true;
        break;
      }
    }
    if (overlaps)
      continue;

    TechMapper::Unit unit;
    unit.swNodes = match.swNodesByOp;
    unit.inputBindings = match.inputBindings;
    unit.outputBindings = match.outputBindings;
    unit.internalEdges = match.internalEdges;
    unit.configurable = familyList[match.familyIndex].configurable;
    for (IdIndex hwNodeId : familyList[match.familyIndex].hwNodeIds) {
      TechMapper::Candidate candidate;
      candidate.hwNodeId = hwNodeId;
      candidate.configFields.assign(match.configFields.begin(),
                                    match.configFields.end());
      unit.candidates.push_back(std::move(candidate));
    }
    int unitIndex = static_cast<int>(plan.units.size());
    for (IdIndex swNodeId : unit.swNodes) {
      nodeToUnit[swNodeId] = unitIndex;
      ++techNodeCount;
    }
    plan.units.push_back(std::move(unit));
  }

  if (totalOpCount > 0)
    plan.coverageScore = static_cast<double>(techNodeCount) /
                         static_cast<double>(totalOpCount);

  auto &contracted = plan.contractedDFG;
  contracted.reserve(dfg.countNodes(), dfg.countPorts(), dfg.countEdges());

  std::vector<bool> unitCreated(plan.units.size(), false);
  for (IdIndex swNodeId = 0; swNodeId < static_cast<IdIndex>(dfg.nodes.size());
       ++swNodeId) {
    const Node *swNode = dfg.getNode(swNodeId);
    if (!swNode)
      continue;

    int unitIndex = swNodeId < nodeToUnit.size() ? nodeToUnit[swNodeId] : -1;
    if (unitIndex >= 0) {
      if (unitCreated[unitIndex]) {
        plan.originalNodeToContractedNode[swNodeId] =
            plan.units[unitIndex].contractedNodeId;
        continue;
      }

      const auto &candidate = plan.units[unitIndex].candidates.front();
      const Node *hwNode = adg.getNode(candidate.hwNodeId);
      if (!hwNode)
        return false;

      auto node = std::make_unique<Node>();
      node->kind = Node::OperationNode;
      addNodeAttr(node.get(), "op_name",
                  mlir::StringAttr::get(dfg.context, "techmap_group"),
                  dfg.context);
      addNodeAttr(node.get(), "tech_group_size",
                  mlir::IntegerAttr::get(mlir::IntegerType::get(dfg.context, 32),
                                         plan.units[unitIndex].swNodes.size()),
                  dfg.context);
      IdIndex contractedNodeId = contracted.addNode(std::move(node));
      plan.units[unitIndex].contractedNodeId = contractedNodeId;
      for (IdIndex member : plan.units[unitIndex].swNodes)
        plan.originalNodeToContractedNode[member] = contractedNodeId;

      for (IdIndex hwPortId : hwNode->inputPorts) {
        const Port *hwPort = adg.getPort(hwPortId);
        auto port = std::make_unique<Port>();
        port->direction = Port::Input;
        port->type = hwPort ? hwPort->type : mlir::Type();
        IdIndex portId = contracted.addPort(std::move(port));
        contracted.ports[portId]->parentNode = contractedNodeId;
        contracted.nodes[contractedNodeId]->inputPorts.push_back(portId);
        plan.units[unitIndex].contractedInputPorts.push_back(portId);
      }
      for (IdIndex hwPortId : hwNode->outputPorts) {
        const Port *hwPort = adg.getPort(hwPortId);
        auto port = std::make_unique<Port>();
        port->direction = Port::Output;
        port->type = hwPort ? hwPort->type : mlir::Type();
        IdIndex portId = contracted.addPort(std::move(port));
        contracted.ports[portId]->parentNode = contractedNodeId;
        contracted.nodes[contractedNodeId]->outputPorts.push_back(portId);
        plan.units[unitIndex].contractedOutputPorts.push_back(portId);
      }
      plan.contractedCandidates[contractedNodeId] = {};
      for (const auto &unitCandidate : plan.units[unitIndex].candidates)
        plan.contractedCandidates[contractedNodeId].push_back(
            unitCandidate.hwNodeId);
      unitCreated[unitIndex] = true;
      continue;
    }

    auto node = cloneNodeShell(swNode, dfg.context);
    IdIndex contractedNodeId = contracted.addNode(std::move(node));
    plan.originalNodeToContractedNode[swNodeId] = contractedNodeId;

    for (IdIndex swPortId : swNode->inputPorts) {
      const Port *swPort = dfg.getPort(swPortId);
      auto port = clonePort(swPort);
      IdIndex contractedPortId = contracted.addPort(std::move(port));
      contracted.ports[contractedPortId]->parentNode = contractedNodeId;
      contracted.nodes[contractedNodeId]->inputPorts.push_back(contractedPortId);
      plan.originalPortToContractedPort[swPortId] = contractedPortId;
    }
    for (IdIndex swPortId : swNode->outputPorts) {
      const Port *swPort = dfg.getPort(swPortId);
      auto port = clonePort(swPort);
      IdIndex contractedPortId = contracted.addPort(std::move(port));
      contracted.ports[contractedPortId]->parentNode = contractedNodeId;
      contracted.nodes[contractedNodeId]->outputPorts.push_back(contractedPortId);
      plan.originalPortToContractedPort[swPortId] = contractedPortId;
    }
  }

  for (auto &unit : plan.units) {
    for (const auto &binding : unit.inputBindings) {
      if (binding.swPortId >= plan.originalPortToContractedPort.size() ||
          binding.hwPortIndex >= unit.contractedInputPorts.size())
        return false;
      plan.originalPortToContractedPort[binding.swPortId] =
          unit.contractedInputPorts[binding.hwPortIndex];
    }
    for (const auto &binding : unit.outputBindings) {
      if (binding.swPortId >= plan.originalPortToContractedPort.size() ||
          binding.hwPortIndex >= unit.contractedOutputPorts.size())
        return false;
      plan.originalPortToContractedPort[binding.swPortId] =
          unit.contractedOutputPorts[binding.hwPortIndex];
    }
    for (IdIndex edgeId : unit.internalEdges) {
      if (edgeId < plan.originalEdgeKinds.size())
        plan.originalEdgeKinds[edgeId] = TechMappedEdgeKind::IntraFU;
    }
  }

  std::map<std::string, IdIndex> dedupEdges;
  for (IdIndex edgeId = 0; edgeId < static_cast<IdIndex>(dfg.edges.size());
       ++edgeId) {
    if (plan.originalEdgeKinds[edgeId] == TechMappedEdgeKind::IntraFU)
      continue;
    const Edge *edge = dfg.getEdge(edgeId);
    if (!edge)
      continue;
    IdIndex srcPort = plan.originalPortToContractedPort[edge->srcPort];
    IdIndex dstPort = plan.originalPortToContractedPort[edge->dstPort];
    if (srcPort == INVALID_ID || dstPort == INVALID_ID) {
      plan.diagnostics = "tech-mapping lost an external port binding";
      return false;
    }

    std::string key = std::to_string(srcPort) + ":" + std::to_string(dstPort);
    auto found = dedupEdges.find(key);
    if (found != dedupEdges.end()) {
      plan.originalEdgeToContractedEdge[edgeId] = found->second;
      continue;
    }

    auto newEdge = std::make_unique<Edge>();
    newEdge->srcPort = srcPort;
    newEdge->dstPort = dstPort;
    newEdge->attributes = edge->attributes;
    IdIndex contractedEdgeId = contracted.addEdge(std::move(newEdge));
    contracted.ports[srcPort]->connectedEdges.push_back(contractedEdgeId);
    contracted.ports[dstPort]->connectedEdges.push_back(contractedEdgeId);
    dedupEdges[key] = contractedEdgeId;
    plan.originalEdgeToContractedEdge[edgeId] = contractedEdgeId;
  }

  return true;
}

bool TechMapper::expandPlanMapping(
    const Graph &originalDfg, const Graph &adg, const Plan &plan,
    const MappingState &contractedState, MappingState &expandedState,
    llvm::SmallVectorImpl<FUConfigSelection> &fuConfigs) {
  expandedState.init(originalDfg, adg);
  fuConfigs.clear();

  for (IdIndex swNodeId = 0;
       swNodeId < static_cast<IdIndex>(plan.originalNodeToContractedNode.size());
       ++swNodeId) {
    IdIndex contractedNodeId = plan.originalNodeToContractedNode[swNodeId];
    if (contractedNodeId == INVALID_ID ||
        contractedNodeId >= contractedState.swNodeToHwNode.size())
      continue;
    IdIndex hwNodeId = contractedState.swNodeToHwNode[contractedNodeId];
    if (hwNodeId == INVALID_ID)
      continue;
    expandedState.swNodeToHwNode[swNodeId] = hwNodeId;
    expandedState.hwNodeToSwNodes[hwNodeId].push_back(swNodeId);
  }

  for (IdIndex swPortId = 0;
       swPortId < static_cast<IdIndex>(plan.originalPortToContractedPort.size());
       ++swPortId) {
    IdIndex contractedPortId = plan.originalPortToContractedPort[swPortId];
    if (contractedPortId == INVALID_ID ||
        contractedPortId >= contractedState.swPortToHwPort.size())
      continue;
    IdIndex hwPortId = contractedState.swPortToHwPort[contractedPortId];
    if (hwPortId == INVALID_ID)
      continue;
    expandedState.swPortToHwPort[swPortId] = hwPortId;
    expandedState.hwPortToSwPorts[hwPortId].push_back(swPortId);
  }

  for (IdIndex swEdgeId = 0;
       swEdgeId < static_cast<IdIndex>(plan.originalEdgeToContractedEdge.size());
       ++swEdgeId) {
    if (plan.originalEdgeKinds[swEdgeId] == TechMappedEdgeKind::IntraFU)
      continue;
    IdIndex contractedEdgeId = plan.originalEdgeToContractedEdge[swEdgeId];
    if (contractedEdgeId == INVALID_ID ||
        contractedEdgeId >= contractedState.swEdgeToHwPaths.size())
      continue;
    expandedState.swEdgeToHwPaths[swEdgeId] =
        contractedState.swEdgeToHwPaths[contractedEdgeId];
    llvm::ArrayRef<IdIndex> path = expandedState.swEdgeToHwPaths[swEdgeId];
    for (size_t i = 0; i + 1 < path.size(); i += 2) {
      IdIndex outPort = path[i];
      IdIndex inPort = path[i + 1];
      const Port *hwOut = adg.getPort(outPort);
      if (!hwOut)
        continue;
      for (IdIndex hwEdgeId : hwOut->connectedEdges) {
        const Edge *hwEdge = adg.getEdge(hwEdgeId);
        if (!hwEdge)
          continue;
        if (hwEdge->srcPort == outPort && hwEdge->dstPort == inPort) {
          expandedState.hwEdgeToSwEdges[hwEdgeId].push_back(swEdgeId);
          break;
        }
      }
    }
  }

  for (const Unit &unit : plan.units) {
    if (unit.contractedNodeId == INVALID_ID ||
        unit.contractedNodeId >= contractedState.swNodeToHwNode.size())
      continue;
    IdIndex hwNodeId = contractedState.swNodeToHwNode[unit.contractedNodeId];
    if (hwNodeId == INVALID_ID)
      continue;
    for (const Candidate &candidate : unit.candidates) {
      if (candidate.hwNodeId != hwNodeId || candidate.configFields.empty())
        continue;
      FUConfigSelection selection;
      selection.hwNodeId = hwNodeId;
      if (const Node *hwNode = adg.getNode(hwNodeId)) {
        selection.hwName = getNodeAttrStr(hwNode, "op_name").str();
        selection.peName = getNodeAttrStr(hwNode, "pe_name").str();
      }
      selection.swNodeIds.append(unit.swNodes.begin(), unit.swNodes.end());
      selection.fields.append(candidate.configFields.begin(),
                              candidate.configFields.end());
      fuConfigs.push_back(std::move(selection));
      break;
    }
  }

  return true;
}

} // namespace fcc
