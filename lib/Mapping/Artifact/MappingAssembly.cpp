#include "Mapping/Artifact/MappingArtifact.h"
#include "Mapping/Artifact/SystemMappingArtifact.h"

#include "MappingAssemblyInternal.h"
#include "TechMappingCanonicalKeyInternal.h"

#include "Mapping/IR/MappingOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Verifier.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdint>
#include <map>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

using namespace mlir;

namespace loom::mapping {
namespace {

std::vector<std::uint8_t> bytes(DenseI8ArrayAttr attribute) {
  std::vector<std::uint8_t> result;
  result.reserve(attribute.size());
  for (std::int8_t byte : attribute.asArrayRef())
    result.push_back(static_cast<std::uint8_t>(byte));
  return result;
}

std::string recordKey(DenseI8ArrayAttr attribute) {
  std::string result;
  result.reserve(attribute.size());
  for (std::int8_t byte : attribute.asArrayRef())
    result.push_back(static_cast<char>(byte));
  return result;
}

void appendU32(std::string &result, std::uint32_t value) {
  for (unsigned byte = 0; byte < 4; ++byte)
    result.push_back(static_cast<char>(value >> (8 * (3 - byte))));
}

void appendU64(std::string &result, std::uint64_t value) {
  for (unsigned byte = 0; byte < 8; ++byte)
    result.push_back(static_cast<char>(value >> (8 * (7 - byte))));
}

void appendFramed(std::string &result, llvm::StringRef value) {
  appendU64(result, value.size());
  result.append(value.data(), value.size());
}

void canonicalizeRefinements(Operation *operation) {
  auto refinements = operation->getAttrOfType<ArrayAttr>("refinements");
  if (!refinements)
    return;
  SmallVector<Attribute> ordered(refinements.begin(), refinements.end());
  llvm::sort(ordered, [](Attribute left, Attribute right) {
    auto leftAssignment =
        cast<::mapping::PhysicalRefinementAssignmentAttr>(left);
    auto rightAssignment =
        cast<::mapping::PhysicalRefinementAssignmentAttr>(right);
    return recordKey(leftAssignment.getDomain().getRecord()) <
           recordKey(rightAssignment.getDomain().getRecord());
  });
  operation->setAttr("refinements",
                     ArrayAttr::get(operation->getContext(), ordered));
}

std::string eventKey(Attribute event) {
  std::string result;
  if (auto transition = dyn_cast<::mapping::ActorTransitionEventAttr>(event)) {
    appendU32(result, 0);
    appendFramed(result, recordKey(transition.getActor().getRecord()));
    appendU32(result, transition.getTransition());
    return result;
  }
  if (auto produced =
          dyn_cast<::mapping::GraphProducerEndpointRefAttr>(event)) {
    appendU32(result, 1);
    appendFramed(result, recordKey(produced.getRecord()));
    return result;
  }
  auto consumed = cast<::mapping::GraphConsumerEndpointRefAttr>(event);
  appendU32(result, 2);
  appendFramed(result, recordKey(consumed.getRecord()));
  return result;
}

std::string eventPointKey(::mapping::SpatialEventPointAttr point) {
  std::string result = eventKey(point.getEvent());
  auto offset = point.getGuaranteedOffset();
  appendU32(result, offset ? 1 : 0);
  if (offset)
    appendFramed(result, recordKey(offset.getRecord()));
  return result;
}

std::string ownerKey(Attribute owner) {
  std::string result;
  if (auto compute = dyn_cast<::mapping::ComputeRealizationRefAttr>(owner)) {
    appendU32(result, 0);
    appendU64(result, compute.getEntity());
    return result;
  }
  if (auto memory = dyn_cast<::mapping::MemoryRealizationRefAttr>(owner)) {
    appendU32(result, 1);
    appendU64(result, memory.getEntity());
    return result;
  }
  if (auto binding = dyn_cast<::mapping::MemoryBindingRefAttr>(owner)) {
    appendU32(result, 2);
    appendU64(result, binding.getEntity());
    return result;
  }
  auto route = cast<::mapping::RouteTreeNodeRefAttr>(owner);
  appendU32(result, 3);
  appendFramed(result, recordKey(route.getLogicalNet().getRecord()));
  appendU64(result, route.getNodeOrdinal());
  return result;
}

void appendTypedValues(std::string &result, ArrayAttr values) {
  appendU64(result, values.size());
  for (Attribute value : values)
    appendFramed(
        result,
        recordKey(cast<::mapping::OwnerTypedValueAttr>(value).getRecord()));
}

std::string resourceUseSemanticKey(::mapping::ResourceUseOp use) {
  std::string result = ownerKey(use.getOwner());
  appendFramed(result, recordKey(use.getUseSite().getRecord()));
  auto activation = use.getActivation();
  appendFramed(result, eventPointKey(activation.getTrigger()));
  auto release = activation.getRelease();
  appendU32(result, release ? 1 : 0);
  if (release)
    appendFramed(result, eventPointKey(release));
  appendTypedValues(result, use.getParameters());
  appendTypedValues(result, use.getSharingAssignments());
  return result;
}

std::string memoryIntervalKey(Attribute interval) {
  std::string result;
  if (isa<::mapping::MemoryWholeIntervalAttr>(interval)) {
    appendU32(result, 0);
    return result;
  }
  auto range = cast<::mapping::MemoryByteRangeAttr>(interval);
  appendU32(result, 1);
  appendU64(result, range.getOffsetBytes());
  appendU64(result, range.getSizeBytes());
  return result;
}

std::string memoryBindingTargetKey(Attribute target) {
  std::string result;
  if (auto local = dyn_cast<::mapping::MemoryLocalRegionAttr>(target)) {
    appendU32(result, 0);
    appendFramed(result, recordKey(local.getServiceRegion().getRecord()));
    appendU64(result, local.getPhysicalOffsetBytes());
    return result;
  }
  appendU32(result, 1);
  return result;
}

std::string memoryBindingSemanticKey(::mapping::MemoryBindingOp binding) {
  std::string result;
  appendFramed(result, recordKey(binding.getLogicalMemory().getRecord()));
  appendFramed(result, memoryIntervalKey(binding.getInterval()));
  appendFramed(result, memoryBindingTargetKey(binding.getTarget()));
  return result;
}

void canonicalizeMemoryBinding(::mapping::MemoryBindingOp binding) {
  Block &body = binding.getBody().front();
  SmallVector<::mapping::ExposureEntryOp> exposures;
  for (auto exposure : body.getOps<::mapping::ExposureEntryOp>())
    exposures.push_back(exposure);
  llvm::sort(exposures, [](auto left, auto right) {
    return recordKey(left.getExposure().getRecord()) <
           recordKey(right.getExposure().getRecord());
  });
  for (auto exposure : exposures)
    exposure->moveBefore(&body, body.end());
}

void canonicalizeMemoryEngine(
    ::mapping::MemoryEngineBindingOp binding,
    const llvm::DenseMap<std::uint64_t, std::uint64_t> &bindingRenumbering) {
  Block &body = binding.getBody().front();
  std::vector<Operation *> operations;
  for (Operation &operation : body) {
    if (auto addressed =
            dyn_cast<::mapping::AddressedMemoryOperationOp>(operation)) {
      SmallVector<::mapping::AddressedMemoryUseOp> uses;
      for (auto use : addressed.getBody()
                          .front()
                          .getOps<::mapping::AddressedMemoryUseOp>()) {
        auto found = bindingRenumbering.find(use.getBinding().getEntity());
        if (found != bindingRenumbering.end())
          use.setBindingAttr(::mapping::MemoryBindingRefAttr::get(
              binding.getContext(), found->second));
        uses.push_back(use);
      }
      llvm::sort(uses, [](auto left, auto right) {
        return recordKey(left.getLaunch().getRecord()) <
               recordKey(right.getLaunch().getRecord());
      });
      for (auto use : uses)
        use->moveBefore(&addressed.getBody().front(),
                        addressed.getBody().front().end());
    } else if (auto fence =
                   dyn_cast<::mapping::FenceMemoryOperationOp>(operation)) {
      SmallVector<::mapping::FenceMemoryUseOp> uses;
      for (auto use :
           fence.getBody().front().getOps<::mapping::FenceMemoryUseOp>())
        uses.push_back(use);
      llvm::sort(uses, [](auto left, auto right) {
        return recordKey(left.getLaunch().getRecord()) <
               recordKey(right.getLaunch().getRecord());
      });
      for (auto use : uses)
        use->moveBefore(&fence.getBody().front(),
                        fence.getBody().front().end());
    }
    operations.push_back(&operation);
  }
  llvm::sort(operations, [](Operation *left, Operation *right) {
    auto actor = [](Operation *operation) {
      if (auto addressed =
              dyn_cast<::mapping::AddressedMemoryOperationOp>(operation))
        return recordKey(addressed.getActor().getRecord());
      return recordKey(cast<::mapping::FenceMemoryOperationOp>(operation)
                           .getActor()
                           .getRecord());
    };
    return actor(left) < actor(right);
  });
  for (Operation *operation : operations)
    operation->moveBefore(&body, body.end());
}

Attribute routeNodeKey(MLIRContext *context,
                       ::mapping::GraphProducerEndpointRefAttr logicalNet,
                       std::uint64_t ordinal) {
  Builder builder(context);
  return ArrayAttr::get(context,
                        {logicalNet, builder.getI64IntegerAttr(ordinal)});
}

void canonicalizeRoute(
    ::mapping::RouteTreeOp route,
    llvm::DenseMap<Attribute, std::uint64_t> &routeNodeRenumbering) {
  Block &body = route.getBody().front();
  llvm::DenseMap<std::uint64_t, ::mapping::RouteNodeOp> nodes;
  llvm::DenseMap<std::uint64_t, SmallVector<::mapping::RouteNodeOp, 4>>
      children;
  std::uint64_t rootOrdinal = 0;
  for (auto node : body.getOps<::mapping::RouteNodeOp>()) {
    const std::uint64_t ordinal = node.getNodeOrdinal();
    nodes.try_emplace(ordinal, node);
    auto parent = node.getParentNodeOrdinal();
    if (parent)
      children[*parent].push_back(node);
    else
      rootOrdinal = ordinal;
  }
  for (auto &entry : children)
    llvm::sort(entry.second, [](auto left, auto right) {
      return recordKey(left.getIncomingTraversal()->getRecord()) <
             recordKey(right.getIncomingTraversal()->getRecord());
    });

  SmallVector<::mapping::RouteNodeOp> preorder;
  SmallVector<::mapping::RouteNodeOp> stack{nodes.lookup(rootOrdinal)};
  while (!stack.empty()) {
    auto node = stack.pop_back_val();
    preorder.push_back(node);
    auto found = children.find(node.getNodeOrdinal());
    if (found == children.end())
      continue;
    for (auto child : llvm::reverse(found->second))
      stack.push_back(child);
  }

  llvm::DenseMap<std::uint64_t, std::uint64_t> localRenumbering;
  for (auto [ordinal, node] : llvm::enumerate(preorder))
    localRenumbering.try_emplace(node.getNodeOrdinal(), ordinal);

  Builder builder(route.getContext());
  for (auto node : preorder) {
    const std::uint64_t oldOrdinal = node.getNodeOrdinal();
    routeNodeRenumbering.try_emplace(
        routeNodeKey(route.getContext(), route.getLogicalNet(), oldOrdinal),
        localRenumbering.lookup(oldOrdinal));
    auto oldParent = node.getParentNodeOrdinal();
    node.setNodeOrdinalAttr(
        builder.getI64IntegerAttr(localRenumbering.lookup(oldOrdinal)));
    if (oldParent)
      node.setParentNodeOrdinalAttr(
          builder.getI64IntegerAttr(localRenumbering.lookup(*oldParent)));
    canonicalizeRefinements(node);
    node->moveBefore(&body, body.end());
  }

  SmallVector<::mapping::RouteSinkOp> sinks(
      body.getOps<::mapping::RouteSinkOp>().begin(),
      body.getOps<::mapping::RouteSinkOp>().end());
  llvm::sort(sinks, [](auto left, auto right) {
    return recordKey(left.getSink().getRecord()) <
           recordKey(right.getSink().getRecord());
  });
  for (auto sink : sinks) {
    sink.setNodeOrdinalAttr(builder.getI64IntegerAttr(localRenumbering.lookup(
        static_cast<std::uint64_t>(sink.getNodeOrdinal()))));
    sink->moveBefore(&body, body.end());
  }
}

void canonicalizeChildren(::mapping::ComputeRealizationOp realization) {
  Block &block = realization.getBody().front();
  std::vector<Operation *> children;
  for (Operation &operation : block)
    children.push_back(&operation);
  llvm::sort(children, [](Operation *left, Operation *right) {
    return detail::canonicalTechChildKey(*left) <
           detail::canonicalTechChildKey(*right);
  });
  for (Operation *operation : children)
    operation->moveBefore(&block, block.end());
}

void canonicalizeChildren(::mapping::MemoryRealizationOp realization) {
  Block &block = realization.getBody().front();
  std::vector<Operation *> children;
  for (Operation &operation : block)
    children.push_back(&operation);
  llvm::sort(children, [](Operation *left, Operation *right) {
    return detail::canonicalTechChildKey(*left) <
           detail::canonicalTechChildKey(*right);
  });
  for (Operation *operation : children)
    operation->moveBefore(&block, block.end());
}

void canonicalizeTech(::mapping::TechOp root) {
  SmallVector<Attribute> covers(root.getCovers().begin(),
                                root.getCovers().end());
  llvm::sort(covers, [](Attribute left, Attribute right) {
    return bytes(cast<::mapping::GraphRefAttr>(left).getRecord()) <
           bytes(cast<::mapping::GraphRefAttr>(right).getRecord());
  });
  root.setCoversAttr(ArrayAttr::get(root.getContext(), covers));

  Block &body = root.getBody().front();
  std::vector<::mapping::ComputeRealizationOp> computeRealizations;
  for (auto realization : body.getOps<::mapping::ComputeRealizationOp>()) {
    canonicalizeChildren(realization);
    computeRealizations.push_back(realization);
  }
  llvm::sort(computeRealizations, [](auto left, auto right) {
    return detail::canonicalTechRealizationPayloadKey(left) <
           detail::canonicalTechRealizationPayloadKey(right);
  });

  std::vector<::mapping::MemoryRealizationOp> memoryRealizations;
  for (auto realization : body.getOps<::mapping::MemoryRealizationOp>()) {
    canonicalizeChildren(realization);
    memoryRealizations.push_back(realization);
  }
  llvm::sort(memoryRealizations, [](auto left, auto right) {
    return detail::canonicalTechRealizationPayloadKey(left) <
           detail::canonicalTechRealizationPayloadKey(right);
  });

  Builder builder(root.getContext());
  std::uint64_t entityId = 0;
  for (auto realization : computeRealizations) {
    realization.setEntityIdAttr(builder.getI64IntegerAttr(entityId++));
    realization->moveBefore(&body, body.end());
  }
  for (auto realization : memoryRealizations) {
    realization.setEntityIdAttr(builder.getI64IntegerAttr(entityId++));
    realization->moveBefore(&body, body.end());
  }
}

void canonicalizeSpatial(::mapping::SpatialOp root) {
  Block &body = root.getBody().front();

  SmallVector<::mapping::ComputeBindingOp> computeBindings;
  for (auto binding : body.getOps<::mapping::ComputeBindingOp>()) {
    canonicalizeRefinements(binding);
    computeBindings.push_back(binding);
  }
  llvm::sort(computeBindings, [](auto left, auto right) {
    return left.getRealization().getEntity() <
           right.getRealization().getEntity();
  });

  SmallVector<::mapping::MemoryBindingOp> memoryBindings;
  for (auto binding : body.getOps<::mapping::MemoryBindingOp>()) {
    canonicalizeMemoryBinding(binding);
    memoryBindings.push_back(binding);
  }
  llvm::sort(memoryBindings, [](auto left, auto right) {
    return memoryBindingSemanticKey(left) < memoryBindingSemanticKey(right);
  });
  llvm::DenseMap<std::uint64_t, std::uint64_t> memoryBindingRenumbering;
  Builder builder(root.getContext());
  for (auto [ordinal, binding] : llvm::enumerate(memoryBindings)) {
    memoryBindingRenumbering.try_emplace(
        static_cast<std::uint64_t>(binding.getEntityId()), ordinal);
    binding.setEntityIdAttr(builder.getI64IntegerAttr(ordinal));
  }

  SmallVector<::mapping::MemoryEngineBindingOp> memoryEngines;
  for (auto binding : body.getOps<::mapping::MemoryEngineBindingOp>()) {
    canonicalizeMemoryEngine(binding, memoryBindingRenumbering);
    memoryEngines.push_back(binding);
  }
  llvm::sort(memoryEngines, [](auto left, auto right) {
    return left.getRealization().getEntity() <
           right.getRealization().getEntity();
  });

  llvm::DenseMap<Attribute, std::uint64_t> routeNodeRenumbering;
  SmallVector<::mapping::RouteTreeOp> routes;
  for (auto route : body.getOps<::mapping::RouteTreeOp>()) {
    canonicalizeRoute(route, routeNodeRenumbering);
    routes.push_back(route);
  }
  llvm::sort(routes, [](auto left, auto right) {
    return recordKey(left.getLogicalNet().getRecord()) <
           recordKey(right.getLogicalNet().getRecord());
  });

  SmallVector<::mapping::ResourceUseOp> uses;
  for (auto use : body.getOps<::mapping::ResourceUseOp>()) {
    if (auto bindingOwner =
            dyn_cast<::mapping::MemoryBindingRefAttr>(use.getOwner())) {
      auto found = memoryBindingRenumbering.find(bindingOwner.getEntity());
      if (found != memoryBindingRenumbering.end())
        use->setAttr("owner", ::mapping::MemoryBindingRefAttr::get(
                                  root.getContext(), found->second));
    }
    if (auto routeOwner =
            dyn_cast<::mapping::RouteTreeNodeRefAttr>(use.getOwner())) {
      Attribute key =
          routeNodeKey(root.getContext(), routeOwner.getLogicalNet(),
                       routeOwner.getNodeOrdinal());
      use->setAttr("owner", ::mapping::RouteTreeNodeRefAttr::get(
                                root.getContext(), routeOwner.getLogicalNet(),
                                routeNodeRenumbering.lookup(key)));
    }
    uses.push_back(use);
  }
  llvm::sort(uses, [](auto left, auto right) {
    return resourceUseSemanticKey(left) < resourceUseSemanticKey(right);
  });

  for (auto binding : computeBindings)
    binding->moveBefore(&body, body.end());
  for (auto binding : memoryEngines)
    binding->moveBefore(&body, body.end());
  for (auto binding : memoryBindings)
    binding->moveBefore(&body, body.end());
  for (auto route : routes)
    route->moveBefore(&body, body.end());
  for (auto use : uses)
    use->moveBefore(&body, body.end());
}

llvm::Expected<SystemPresburgerCell>
decodeSystemCell(::mapping::SystemPresburgerCellAttr attribute) {
  SystemPresburgerCell cell;
  cell.dimensionCount = attribute.getDimensionCount();
  cell.symbolCount = attribute.getSymbolCount();
  const auto appendRows = [](ArrayAttr attributes,
                             std::vector<std::vector<std::int64_t>> &rows) {
    rows.reserve(attributes.size());
    for (Attribute attribute : attributes) {
      auto values = cast<DenseI64ArrayAttr>(attribute).asArrayRef();
      rows.emplace_back(values.begin(), values.end());
    }
  };
  appendRows(attribute.getEqualities(), cell.equalities);
  appendRows(attribute.getInequalities(), cell.inequalities);
  return canonicalizeSystemPresburgerCell(cell);
}

::mapping::SystemPresburgerCellAttr
systemCellAttr(MLIRContext *context, const SystemPresburgerCell &cell) {
  SmallVector<Attribute> equalities;
  SmallVector<Attribute> inequalities;
  for (const auto &row : cell.equalities)
    equalities.push_back(DenseI64ArrayAttr::get(context, row));
  for (const auto &row : cell.inequalities)
    inequalities.push_back(DenseI64ArrayAttr::get(context, row));
  return ::mapping::SystemPresburgerCellAttr::get(
      context, cell.dimensionCount, cell.symbolCount,
      ArrayAttr::get(context, equalities),
      ArrayAttr::get(context, inequalities));
}

std::string systemCellKey(::mapping::SystemPresburgerCellAttr cell) {
  std::string result;
  appendU32(result, cell.getDimensionCount());
  appendU32(result, cell.getSymbolCount());
  const auto appendRows = [&](ArrayAttr rows) {
    appendU64(result, rows.size());
    for (Attribute attribute : rows) {
      auto row = cast<DenseI64ArrayAttr>(attribute).asArrayRef();
      appendU64(result, row.size());
      for (std::int64_t value : row)
        appendU64(result, static_cast<std::uint64_t>(value));
    }
  };
  appendRows(cell.getEqualities());
  appendRows(cell.getInequalities());
  return result;
}

template <typename BindingOp, typename ClauseOp>
llvm::Error canonicalizeSystemBinding(BindingOp binding) {
  struct Group final {
    Attribute target;
    SmallVector<::mapping::SystemPresburgerCellAttr> cells;
  };
  std::map<std::string, Group> groups;
  const auto targetKey = [](Attribute target) {
    std::string key;
    if constexpr (std::is_same_v<ClauseOp, ::mapping::ThreadPresburgerClauseOp>)
      key = recordKey(
          cast<::mapping::FabricAccCoreOccurrenceRefAttr>(target).getRecord());
    else
      appendU64(
          key,
          cast<::mapping::SpatialMappingImportRefAttr>(target).getOrdinal());
    return key;
  };
  for (auto clause : binding.getBody().front().template getOps<ClauseOp>()) {
    Attribute target = clause.getTarget();
    Group &group =
        groups.try_emplace(targetKey(target), Group{target, {}}).first->second;
    for (Attribute rawCell : clause.getCells()) {
      auto canonical =
          decodeSystemCell(cast<::mapping::SystemPresburgerCellAttr>(rawCell));
      if (!canonical)
        return canonical.takeError();
      group.cells.push_back(systemCellAttr(binding.getContext(), *canonical));
    }
  }
  if (binding.getDefaultTarget())
    groups.erase(targetKey(*binding.getDefaultTarget()));

  struct OrderedGroup final {
    std::string key;
    Group group;
  };
  std::vector<OrderedGroup> ordered;
  ordered.reserve(groups.size());
  for (auto &[targetKey, group] : groups) {
    llvm::sort(group.cells, [](auto lhs, auto rhs) {
      return systemCellKey(lhs) < systemCellKey(rhs);
    });
    group.cells.erase(std::unique(group.cells.begin(), group.cells.end()),
                      group.cells.end());
    std::string key;
    for (auto cell : group.cells)
      appendFramed(key, systemCellKey(cell));
    appendFramed(key, targetKey);
    ordered.push_back({std::move(key), std::move(group)});
  }
  llvm::sort(ordered, [](const auto &lhs, const auto &rhs) {
    return lhs.key < rhs.key;
  });

  Block &body = binding.getBody().front();
  while (!body.empty())
    body.front().erase();
  OpBuilder builder(binding.getContext());
  builder.setInsertionPointToEnd(&body);
  for (const OrderedGroup &entry : ordered) {
    OperationState state(binding.getLoc(), ClauseOp::getOperationName());
    SmallVector<Attribute> cells(entry.group.cells.begin(),
                                 entry.group.cells.end());
    state.addAttribute("cells", ArrayAttr::get(binding.getContext(), cells));
    state.addAttribute("target", entry.group.target);
    builder.create(state);
  }
  return llvm::Error::success();
}

std::string systemRouteSemanticKey(::mapping::TransferLegRealizationOp route) {
  std::string result;
  appendFramed(result, recordKey(route.getLeg().getRecord()));
  appendFramed(result, recordKey(route.getRootEndpoint().getRecord()));
  for (auto node :
       route.getBody().front().getOps<::mapping::SystemRouteNodeOp>()) {
    appendU64(result, node.getParentNodeOrdinal());
    appendFramed(result, recordKey(node.getIncomingTraversal().getRecord()));
  }
  for (auto sink :
       route.getBody().front().getOps<::mapping::SystemRouteSinkOp>()) {
    appendFramed(result, recordKey(sink.getTerminal().getRecord()));
    appendU64(result, sink.getNodeOrdinal());
  }
  return result;
}

void canonicalizeSystemRoute(::mapping::TransferLegRealizationOp route) {
  Block &body = route.getBody().front();
  llvm::DenseMap<std::uint64_t, SmallVector<::mapping::SystemRouteNodeOp, 4>>
      children;
  for (auto node : body.getOps<::mapping::SystemRouteNodeOp>())
    children[node.getParentNodeOrdinal()].push_back(node);
  for (auto &entry : children)
    llvm::sort(entry.second, [](auto left, auto right) {
      return recordKey(left.getIncomingTraversal().getRecord()) <
             recordKey(right.getIncomingTraversal().getRecord());
    });

  SmallVector<::mapping::SystemRouteNodeOp> preorder;
  SmallVector<::mapping::SystemRouteNodeOp> stack;
  if (auto found = children.find(0); found != children.end())
    for (auto child : llvm::reverse(found->second))
      stack.push_back(child);
  while (!stack.empty()) {
    auto node = stack.pop_back_val();
    preorder.push_back(node);
    if (auto found = children.find(node.getNodeOrdinal());
        found != children.end())
      for (auto child : llvm::reverse(found->second))
        stack.push_back(child);
  }

  llvm::DenseMap<std::uint64_t, std::uint64_t> renumbering;
  renumbering.try_emplace(0, 0);
  for (auto [index, node] : llvm::enumerate(preorder))
    renumbering.try_emplace(node.getNodeOrdinal(), index + 1);

  Builder builder(route.getContext());
  for (auto node : preorder) {
    const std::uint64_t oldOrdinal = node.getNodeOrdinal();
    const std::uint64_t oldParent = node.getParentNodeOrdinal();
    node.setNodeOrdinalAttr(
        builder.getI64IntegerAttr(renumbering.lookup(oldOrdinal)));
    node.setParentNodeOrdinalAttr(
        builder.getI64IntegerAttr(renumbering.lookup(oldParent)));
    node->moveBefore(&body, body.end());
  }

  SmallVector<::mapping::SystemRouteSinkOp> sinks(
      body.getOps<::mapping::SystemRouteSinkOp>().begin(),
      body.getOps<::mapping::SystemRouteSinkOp>().end());
  llvm::sort(sinks, [](auto left, auto right) {
    return recordKey(left.getTerminal().getRecord()) <
           recordKey(right.getTerminal().getRecord());
  });
  for (auto sink : sinks) {
    sink.setNodeOrdinalAttr(
        builder.getI64IntegerAttr(renumbering.lookup(sink.getNodeOrdinal())));
    sink->moveBefore(&body, body.end());
  }
}

std::string servicePlanSemanticKey(::mapping::ServicePlanOp plan) {
  std::string result;
  for (auto route :
       plan.getBody().front().getOps<::mapping::TransferLegRealizationOp>())
    appendFramed(result, systemRouteSemanticKey(route));
  return result;
}

llvm::Error
canonicalizeServiceRealization(::mapping::ServiceRealizationOp service) {
  Block &body = service.getBody().front();
  struct Plan final {
    std::string key;
    ::mapping::ServicePlanOp operation;
  };
  std::vector<Plan> plans;
  for (auto plan : body.getOps<::mapping::ServicePlanOp>()) {
    Block &planBody = plan.getBody().front();
    SmallVector<::mapping::TransferLegRealizationOp> routes;
    for (auto route : planBody.getOps<::mapping::TransferLegRealizationOp>()) {
      canonicalizeSystemRoute(route);
      routes.push_back(route);
    }
    llvm::sort(routes, [](auto left, auto right) {
      return recordKey(left.getLeg().getRecord()) <
             recordKey(right.getLeg().getRecord());
    });
    for (auto route : routes)
      route->moveBefore(&planBody, planBody.end());
    plans.push_back({servicePlanSemanticKey(plan), plan});
  }
  llvm::sort(plans, [](const Plan &left, const Plan &right) {
    return left.key < right.key;
  });
  for (std::size_t index = 1; index < plans.size(); ++index)
    if (plans[index - 1].key == plans[index].key)
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "System service realization duplicates a complete plan semantic "
          "key");

  Builder builder(service.getContext());
  for (auto [ordinal, plan] : llvm::enumerate(plans)) {
    plan.operation.setPlanOrdinalAttr(builder.getI64IntegerAttr(ordinal));
    plan.operation->moveBefore(&body, body.end());
  }
  return llvm::Error::success();
}

llvm::Error canonicalizeSystem(::mapping::SystemOp root) {
  SmallVector<Attribute> roots(root.getRootThreadLaunches().begin(),
                               root.getRootThreadLaunches().end());
  llvm::sort(roots, [](Attribute lhs, Attribute rhs) {
    return recordKey(
               cast<::mapping::RootThreadLaunchRefAttr>(lhs).getRecord()) <
           recordKey(cast<::mapping::RootThreadLaunchRefAttr>(rhs).getRecord());
  });
  root.setRootThreadLaunchesAttr(ArrayAttr::get(root.getContext(), roots));

  struct Import final {
    std::size_t oldOrdinal = 0;
    std::string key;
    Attribute value;
  };
  std::vector<Import> imports;
  for (auto [ordinal, attribute] :
       llvm::enumerate(root.getSpatialMappingImports()))
    imports.push_back(
        {ordinal,
         recordKey(
             cast<::mapping::ArtifactRootReferenceAttr>(attribute).getRecord()),
         attribute});
  llvm::sort(imports, [](const Import &lhs, const Import &rhs) {
    return lhs.key < rhs.key;
  });
  std::vector<std::uint64_t> importRenumbering(imports.size());
  SmallVector<Attribute> orderedImports;
  orderedImports.reserve(imports.size());
  for (auto [ordinal, entry] : llvm::enumerate(imports)) {
    importRenumbering[entry.oldOrdinal] = ordinal;
    orderedImports.push_back(entry.value);
  }
  root.setSpatialMappingImportsAttr(
      ArrayAttr::get(root.getContext(), orderedImports));

  Block &body = root.getBody().front();
  SmallVector<::mapping::ThreadExecutionBindingOp> threadBindings;
  SmallVector<::mapping::GraphExecutionBindingOp> graphBindings;
  SmallVector<::mapping::ServiceRealizationOp> services;
  for (Operation &operation : body) {
    if (auto graph = dyn_cast<::mapping::GraphExecutionBindingOp>(operation)) {
      if (graph.getDefaultTarget())
        graph->setAttr(
            "default_target",
            ::mapping::SpatialMappingImportRefAttr::get(
                root.getContext(),
                importRenumbering[graph.getDefaultTarget()->getOrdinal()]));
      for (auto clause :
           graph.getBody().front().getOps<::mapping::GraphPresburgerClauseOp>())
        clause->setAttr(
            "target", ::mapping::SpatialMappingImportRefAttr::get(
                          root.getContext(),
                          importRenumbering[clause.getTarget().getOrdinal()]));
      if (llvm::Error error =
              canonicalizeSystemBinding<::mapping::GraphExecutionBindingOp,
                                        ::mapping::GraphPresburgerClauseOp>(
                  graph))
        return error;
      graphBindings.push_back(graph);
    } else if (auto thread =
                   dyn_cast<::mapping::ThreadExecutionBindingOp>(operation)) {
      if (llvm::Error error =
              canonicalizeSystemBinding<::mapping::ThreadExecutionBindingOp,
                                        ::mapping::ThreadPresburgerClauseOp>(
                  thread))
        return error;
      threadBindings.push_back(thread);
    } else if (auto service =
                   dyn_cast<::mapping::ServiceRealizationOp>(operation)) {
      if (llvm::Error error = canonicalizeServiceRealization(service))
        return error;
      services.push_back(service);
    }
  }
  llvm::sort(threadBindings, [](auto left, auto right) {
    return recordKey(left.getKey().getRecord()) <
           recordKey(right.getKey().getRecord());
  });
  llvm::sort(graphBindings, [](auto left, auto right) {
    return recordKey(left.getKey().getRecord()) <
           recordKey(right.getKey().getRecord());
  });
  llvm::sort(services, [](auto left, auto right) {
    return recordKey(left.getKey().getRecord()) <
           recordKey(right.getKey().getRecord());
  });
  for (auto binding : threadBindings)
    binding->moveBefore(&body, body.end());
  for (auto binding : graphBindings)
    binding->moveBefore(&body, body.end());
  for (auto service : services)
    service->moveBefore(&body, body.end());
  return llvm::Error::success();
}

} // namespace

llvm::Expected<detail::CanonicalTechMappingAssembly>
detail::prepareCanonicalTechMappingAssembly(::mapping::TechOp root) {
  OwningOpRef<Operation *> clone(root->clone());
  auto canonical = cast<::mapping::TechOp>(clone.get());
  canonicalizeTech(canonical);
  if (failed(verify(canonical)))
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "mapping artifact is structurally invalid");

  std::string text;
  llvm::raw_string_ostream stream(text);
  canonical.print(stream, OpPrintingFlags().enableDebugInfo(false));
  stream << '\n';
  stream.flush();
  return detail::CanonicalTechMappingAssembly{
      std::move(clone), CanonicalSemanticBytes(std::vector<std::uint8_t>(
                            text.begin(), text.end()))};
}

llvm::Expected<CanonicalSemanticBytes>
writeCanonicalMappingAssembly(::mapping::TechOp root) {
  auto prepared = detail::prepareCanonicalTechMappingAssembly(root);
  if (!prepared)
    return prepared.takeError();
  return std::move(prepared->bytes);
}

llvm::Expected<detail::CanonicalSpatialMappingAssembly>
detail::prepareCanonicalSpatialMappingAssembly(::mapping::SpatialOp root) {
  OwningOpRef<Operation *> clone(root->clone());
  auto canonical = cast<::mapping::SpatialOp>(clone.get());
  if (failed(verify(canonical)))
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "spatial mapping is structurally invalid");
  canonicalizeSpatial(canonical);
  if (failed(verify(canonical)))
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "canonical spatial mapping is structurally invalid");

  std::string text;
  llvm::raw_string_ostream stream(text);
  canonical.print(stream, OpPrintingFlags().enableDebugInfo(false));
  stream << '\n';
  stream.flush();
  return detail::CanonicalSpatialMappingAssembly{
      std::move(clone), CanonicalSemanticBytes(std::vector<std::uint8_t>(
                            text.begin(), text.end()))};
}

llvm::Expected<CanonicalSemanticBytes>
writeCanonicalSpatialMappingAssembly(::mapping::SpatialOp root) {
  auto prepared = detail::prepareCanonicalSpatialMappingAssembly(root);
  if (!prepared)
    return prepared.takeError();
  return std::move(prepared->bytes);
}

llvm::Expected<detail::CanonicalSystemMappingAssembly>
detail::prepareCanonicalSystemMappingAssembly(::mapping::SystemOp root) {
  OwningOpRef<Operation *> clone(root->clone());
  auto canonical = cast<::mapping::SystemOp>(clone.get());
  if (failed(verify(canonical)))
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "system mapping is structurally invalid");
  if (llvm::Error error = canonicalizeSystem(canonical))
    return std::move(error);
  if (failed(verify(canonical)))
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "canonical system mapping is structurally invalid");
  std::string text;
  llvm::raw_string_ostream stream(text);
  canonical.print(stream, OpPrintingFlags().enableDebugInfo(false));
  stream << '\n';
  stream.flush();
  return detail::CanonicalSystemMappingAssembly{
      std::move(clone), CanonicalSemanticBytes(std::vector<std::uint8_t>(
                            text.begin(), text.end()))};
}

llvm::Expected<CanonicalSemanticBytes>
writeCanonicalSystemMappingAssembly(::mapping::SystemOp root) {
  auto prepared = detail::prepareCanonicalSystemMappingAssembly(root);
  if (!prepared)
    return prepared.takeError();
  return std::move(prepared->bytes);
}

} // namespace loom::mapping
