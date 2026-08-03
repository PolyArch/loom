#include "PnR/SpatialMappingMaterializer.h"

#include "Dataflow/IR/DataflowReferenceCodec.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Mapping/IR/MappingDialect.h"
#include "Mapping/IR/MappingOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace loom::pnr {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "spatial_mapping_materialization_invalid: " +
                                     message);
}

mlir::DenseI8ArrayAttr denseBytes(mlir::MLIRContext *context,
                                  llvm::ArrayRef<std::uint8_t> bytes) {
  std::vector<std::int8_t> signedBytes;
  signedBytes.reserve(bytes.size());
  for (std::uint8_t byte : bytes)
    signedBytes.push_back(static_cast<std::int8_t>(byte));
  return mlir::DenseI8ArrayAttr::get(context, signedBytes);
}

template <typename Attr, typename Ref>
Attr fabricAttr(mlir::MLIRContext *context, const Ref &reference) {
  return Attr::get(
      context,
      denseBytes(context, ::loom::fabric::canonicalFabricBytes(reference)));
}

template <typename Attr, typename Ref>
llvm::Expected<Attr> dataflowAttr(mlir::MLIRContext *context,
                                  const ArtifactIdentity &owner,
                                  const Ref &reference) {
  auto bytes = ::dataflow::encodeDataflowReference(owner, reference);
  if (!bytes)
    return bytes.takeError();
  return Attr::get(context, denseBytes(context, *bytes));
}

::mapping::ArtifactIdentityAttr identityAttr(mlir::MLIRContext *context,
                                             const ArtifactIdentity &identity) {
  return ::mapping::ArtifactIdentityAttr::get(
      context, denseBytes(context, identity.bytes()));
}

llvm::Expected<std::vector<::loom::mapping::SpatialComputeBindingView>>
materializeComputeBindings(mlir::OpBuilder &builder, mlir::Location location,
                           mlir::Block &body,
                           const SpatialCandidateState &candidate) {
  const FrozenSpatialPnrProblem &problem = candidate.problem();
  const auto realizations = problem.realizations().computeRealizations();
  const auto placements = problem.realizations().computePlacements();
  const auto contexts = problem.realizations().computeInstructionContexts();
  std::vector<::loom::mapping::SpatialComputeBindingView> bindings;
  bindings.reserve(realizations.size());
  for (PnrIndex ordinal = 0; ordinal < realizations.size(); ++ordinal) {
    const auto &realization = realizations[ordinal];
    const auto &selection = candidate.computeBinding(ordinal);
    if (selection.placement >= placements.size() ||
        selection.instructionContext >= contexts.size())
      return invalid("compute selection exceeds its frozen reverse table");
    const auto &placement = placements[selection.placement];
    if (placement.realization != ordinal ||
        selection.placement < realization.placementOffset ||
        selection.placement >=
            realization.placementOffset + realization.placementCount ||
        selection.instructionContext < placement.contextOffset ||
        selection.instructionContext >=
            placement.contextOffset + placement.contextCount)
      return invalid("compute selection is outside its realization domain");

    builder.setInsertionPointToEnd(&body);
    ::mapping::ComputeBindingOp::create(
        builder, location,
        ::mapping::ComputeRealizationRefAttr::get(builder.getContext(),
                                                  realization.reference.entity),
        fabricAttr<::mapping::FabricFuOccurrenceRefAttr>(builder.getContext(),
                                                         placement.fu),
        fabricAttr<::mapping::InstructionContextRefAttr>(
            builder.getContext(), contexts[selection.instructionContext]),
        builder.getArrayAttr({}));
    bindings.push_back(::loom::mapping::SpatialComputeBindingView{
        realization.reference.entity,
        placement.fu,
        contexts[selection.instructionContext],
        {}});
  }
  return bindings;
}

llvm::Error materializeRouteTree(mlir::OpBuilder &builder,
                                 mlir::Location location, mlir::Block &parent,
                                 const SpatialCandidateState &candidate,
                                 PnrIndex netOrdinal,
                                 const ArtifactIdentity &dataflowIdentity) {
  const FrozenSpatialPnrProblem &problem = candidate.problem();
  const auto logicalNets = problem.transfers().logicalNets();
  const auto sinks = problem.transfers().logicalNetSinks();
  const auto endpoints = problem.routing().routingEndpoints();
  const auto arcs = problem.routing().routingArcs();
  const auto arcSources = problem.routing().arcSources();
  const auto traversals = problem.routing().traversals();
  if (netOrdinal >= logicalNets.size())
    return invalid("logical-net ordinal exceeds the frozen transfer index");
  const FrozenSpatialLogicalNet &net = logicalNets[netOrdinal];
  const RouteTreeState &tree = candidate.routeTree(netOrdinal);
  const auto source = tree.sourceEndpoint();
  if (!source || *source >= endpoints.size())
    return invalid("closed RouteTree has no valid source endpoint");
  if (*source != candidate.logicalNetSourceEndpoint(netOrdinal))
    return invalid("RouteTree source diverges from selected terminal binding");

  auto producer = dataflowAttr<::mapping::GraphProducerEndpointRefAttr>(
      builder.getContext(), dataflowIdentity, net.producer);
  if (!producer)
    return producer.takeError();
  builder.setInsertionPointToEnd(&parent);
  auto route = ::mapping::RouteTreeOp::create(
      builder, location, *producer,
      fabricAttr<::mapping::FabricTransportEndpointRefAttr>(
          builder.getContext(), endpoints[*source].reference));
  auto *body = new mlir::Block();
  route.getBody().push_back(body);

  const auto nodes = tree.nodeStorage();
  std::vector<PnrIndex> ordinals(nodes.size(), getInvalidPnrIndex());
  PnrIndex nextOrdinal = 0;
  for (auto [slot, node] : llvm::enumerate(nodes)) {
    if (!node.isActive())
      continue;
    ordinals[slot] = nextOrdinal++;
  }
  if (nextOrdinal != tree.activeNodeCount())
    return invalid("RouteTree active-node count diverges from node storage");

  builder.setInsertionPointToEnd(body);
  for (auto [slotValue, node] : llvm::enumerate(nodes)) {
    if (!node.isActive())
      continue;
    const PnrIndex slot = static_cast<PnrIndex>(slotValue);
    mlir::IntegerAttr parentOrdinal;
    ::mapping::FabricPhysicalTraversalRefAttr traversal;
    if (node.parentArc == getInvalidPnrIndex()) {
      if (node.endpoint != *source)
        return invalid("RouteTree has a non-source root node");
    } else {
      if (node.parentArc >= arcs.size() || node.parentArc >= arcSources.size())
        return invalid("RouteTree node names an absent routing arc");
      const auto &arc = arcs[node.parentArc];
      if (arc.target != node.endpoint || arc.traversal >= traversals.size())
        return invalid("RouteTree node disagrees with its incoming arc");
      auto parentSlot = tree.findNode(arcSources[node.parentArc]);
      if (!parentSlot || *parentSlot >= ordinals.size() ||
          ordinals[*parentSlot] == getInvalidPnrIndex())
        return invalid("RouteTree incoming arc has no active parent node");
      parentOrdinal = builder.getI64IntegerAttr(ordinals[*parentSlot]);
      traversal = fabricAttr<::mapping::FabricPhysicalTraversalRefAttr>(
          builder.getContext(), traversals[arc.traversal].reference);
    }
    ::mapping::RouteNodeOp::create(builder, location, ordinals[slot],
                                   parentOrdinal, traversal,
                                   builder.getArrayAttr({}));
  }

  for (PnrIndex sinkOrdinal = 0; sinkOrdinal < net.sinkCount; ++sinkOrdinal) {
    const auto endpoint = tree.sinkEndpoint(sinkOrdinal);
    if (!endpoint || *endpoint >= endpoints.size())
      return invalid("RouteTree has an unbound sink endpoint");
    if (*endpoint != candidate.logicalNetSinkEndpoint(netOrdinal, sinkOrdinal))
      return invalid("RouteTree sink diverges from selected terminal binding");
    const auto slot = tree.findNode(*endpoint);
    if (!slot || *slot >= ordinals.size() ||
        ordinals[*slot] == getInvalidPnrIndex())
      return invalid("RouteTree sink endpoint is not present in the tree");
    const PnrIndex sinkIndex = net.sinkOffset + sinkOrdinal;
    if (sinkIndex >= sinks.size())
      return invalid("RouteTree sink exceeds the frozen logical-net slice");
    auto sink = dataflowAttr<::mapping::GraphConsumerEndpointRefAttr>(
        builder.getContext(), dataflowIdentity, sinks[sinkIndex]);
    if (!sink)
      return sink.takeError();
    ::mapping::RouteSinkOp::create(builder, location, *sink, ordinals[*slot]);
  }
  return llvm::Error::success();
}

llvm::Error materializeResourceUses(
    mlir::OpBuilder &builder, mlir::Location location, mlir::Block &body,
    llvm::ArrayRef<::loom::mapping::SpatialComputeUseRequirement> uses,
    const ArtifactIdentity &dataflowIdentity) {
  builder.setInsertionPointToEnd(&body);
  for (const auto &use : uses) {
    auto actor = dataflowAttr<::mapping::ActorRefAttr>(
        builder.getContext(), dataflowIdentity, use.actor);
    if (!actor)
      return actor.takeError();
    auto transition = ::mapping::ActorTransitionEventAttr::get(
        builder.getContext(), *actor, use.transition);
    auto trigger = ::mapping::SpatialEventPointAttr::get(
        builder.getContext(), transition, ::mapping::OwnerTypedValueAttr());
    auto activation = ::mapping::SpatialRelativeActivationAttr::get(
        builder.getContext(), trigger, ::mapping::SpatialEventPointAttr());
    ::mapping::ResourceUseOp::create(
        builder, location,
        ::mapping::ComputeRealizationRefAttr::get(builder.getContext(),
                                                  use.realization),
        fabricAttr<::mapping::FabricUsePatternRefAttr>(builder.getContext(),
                                                       use.pattern),
        activation, builder.getArrayAttr({}), builder.getArrayAttr({}));
  }
  return llvm::Error::success();
}

} // namespace

llvm::Expected<::loom::mapping::FinalizedSpatialMapping>
finalizeSpatialMappingCandidate(
    const SpatialCandidateState &candidate,
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::mapping::TechMappingView &techMapping,
    const ::loom::fabric::FabricArtifactView &fabric,
    const ArtifactStore &store) {
  if (llvm::Error error = candidate.verify())
    return std::move(error);
  const FrozenSpatialPnrProblem &problem = candidate.problem();
  if (problem.dataflowIdentity() != dataflow.identity() ||
      problem.techMappingIdentity() != techMapping.identity() ||
      problem.fabricIdentity() != fabric.identity())
    return invalid("candidate and sealed upstream views have different owners");
  if (techMapping.dataflowIdentity() != dataflow.identity() ||
      techMapping.fabricIdentity() != fabric.identity())
    return invalid("TechMapping upstream closure is inconsistent");
  if (candidate.unroutedObligationCount() != 0)
    return invalid("candidate still has unrouted sink obligations");
  if (candidate.capacityOveruse() != 0)
    return invalid("candidate still exceeds an atomic resource capacity");
  if (!problem.realizations().memoryRealizations().empty())
    return invalid("Spatial memory binding materialization is unavailable");

  mlir::DialectRegistry registry;
  registry.insert<::mapping::MappingDialect>();
  mlir::MLIRContext context(registry, mlir::MLIRContext::Threading::DISABLED);
  context.loadDialect<::mapping::MappingDialect>();
  mlir::OpBuilder builder(&context);
  const mlir::Location location = builder.getUnknownLoc();
  auto module = mlir::ModuleOp::create(location);
  builder.setInsertionPointToStart(module.getBody());
  auto root = ::mapping::SpatialOp::create(
      builder, location, identityAttr(&context, techMapping.identity()),
      identityAttr(&context, dataflow.identity()),
      identityAttr(&context, fabric.identity()));
  auto *body = new mlir::Block();
  root.getBody().push_back(body);

  auto bindings =
      materializeComputeBindings(builder, location, *body, candidate);
  if (!bindings)
    return bindings.takeError();
  const auto logicalNets = problem.transfers().logicalNets();
  for (PnrIndex net = 0; net < logicalNets.size(); ++net)
    if (llvm::Error error = materializeRouteTree(
            builder, location, *body, candidate, net, dataflow.identity()))
      return std::move(error);
  auto uses = ::loom::mapping::deriveSpatialComputeUseRequirements(
      dataflow, techMapping, fabric, *bindings);
  if (!uses)
    return uses.takeError();
  if (llvm::Error error = materializeResourceUses(builder, location, *body,
                                                  *uses, dataflow.identity()))
    return std::move(error);

  return ::loom::mapping::finalizeSpatialMapping(root, dataflow, techMapping,
                                                 fabric, store);
}

} // namespace loom::pnr
