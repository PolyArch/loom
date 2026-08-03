#include "Mapping/Artifact/MappingArtifact.h"

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
#include <string>
#include <tuple>
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
  for (auto route : routes)
    route->moveBefore(&body, body.end());
  for (auto use : uses)
    use->moveBefore(&body, body.end());
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

} // namespace loom::mapping
