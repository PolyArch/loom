#include "Mapping/Artifact/MappingArtifact.h"
#include "Mapping/Artifact/MappingConstraintSet.h"

#include "Common/ArtifactFinalizer.h"
#include "Common/IndexWidth.h"
#include "Common/PointerLayout.h"
#include "ConfiguredHardwareProjectionInternal.h"
#include "Dataflow/IR/DataflowActorSemantics.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Dataflow/IR/DataflowReferenceCodec.h"
#include "Dataflow/IR/DataflowServiceSchema.h"
#include "Dataflow/IR/OperationSchema.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/IR/ImplementationFamily.h"
#include "Fabric/IR/OperationResourceContract.h"
#include "Fabric/IR/TemporalPeResourceContract.h"
#include "Fabric/IR/UsePatternValue.h"
#include "Fabric/Identity/FabricHandshake.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Fabric/Identity/FabricRefText.h"
#include "Mapping/Artifact/MappingProgressAnalysis.h"
#include "Mapping/Artifact/SpatialPhysicalDemandProjection.h"
#include "Mapping/IR/MappingActivationKey.h"
#include "Mapping/IR/MappingDialect.h"
#include "MappingAssemblyInternal.h"
#include "MappingResourceUseImport.h"
#include "SpatialMappingCapacityVerification.h"
#include "SpatialMappingMemoryImport.h"
#include "SpatialMappingTagAssignments.h"

#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

using namespace mlir;

namespace loom::mapping {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "mapping_artifact_invalid: " + message);
}

llvm::Error incomplete(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::operation_not_supported),
      "spatial_mapping_incomplete: " + message);
}

std::vector<std::uint8_t> unsignedBytes(DenseI8ArrayAttr record) {
  std::vector<std::uint8_t> result;
  result.reserve(record.size());
  for (std::int8_t byte : record.asArrayRef())
    result.push_back(static_cast<std::uint8_t>(byte));
  return result;
}

std::string byteKey(llvm::ArrayRef<std::uint8_t> bytes) {
  return std::string(reinterpret_cast<const char *>(bytes.data()),
                     bytes.size());
}

llvm::Expected<ArtifactIdentity>
decodeIdentity(::mapping::ArtifactIdentityAttr attribute) {
  return ArtifactIdentity::fromBytes(unsignedBytes(attribute.getRecord()));
}

template <typename Ref, typename Attr>
llvm::Expected<Ref> decodeDataflow(Attr attribute,
                                   const ArtifactIdentity &owner) {
  return ::dataflow::decodeDataflowReference<Ref>(
      unsignedBytes(attribute.getRecord()), owner);
}

template <typename Ref, typename Attr>
llvm::Expected<Ref> decodeFabric(Attr attribute) {
  return ::loom::fabric::decodeFabricRef<Ref>(
      unsignedBytes(attribute.getRecord()));
}

struct ParsedSpatialRoot final {
  std::unique_ptr<MLIRContext> context;
  OwningOpRef<ModuleOp> module;
  ::mapping::SpatialOp root;
};

llvm::Expected<ParsedSpatialRoot>
parseSpatialRoot(const CanonicalSemanticBytes &canonicalBytes) {
  std::string wrapped = "module {\n";
  wrapped.append(reinterpret_cast<const char *>(canonicalBytes.bytes().data()),
                 canonicalBytes.bytes().size());
  wrapped += "}\n";

  DialectRegistry registry;
  registry.insert<::mapping::MappingDialect>();
  auto context =
      std::make_unique<MLIRContext>(registry, MLIRContext::Threading::DISABLED);
  auto module = parseSourceString<ModuleOp>(wrapped, context.get());
  if (!module)
    return invalid("canonical SpatialMapping payload cannot be parsed");

  ::mapping::SpatialOp root;
  unsigned rootCount = 0;
  for (Operation &operation : module->getBody()->without_terminator()) {
    auto candidate = dyn_cast<::mapping::SpatialOp>(operation);
    if (!candidate)
      return invalid("mapping artifact contains a non-SpatialMapping root");
    root = candidate;
    ++rootCount;
  }
  if (rootCount != 1)
    return invalid(
        "mapping artifact must contain exactly one SpatialMapping root");
  if (failed(verify(root)))
    return invalid("SpatialMapping root is structurally invalid");
  return ParsedSpatialRoot{std::move(context), std::move(module), root};
}

llvm::Expected<std::vector<SpatialPhysicalRefinementView>>
importRefinements(ArrayAttr refinements) {
  if (!refinements.empty())
    return invalid(
        "nonempty physical refinement requires its owner value codec");
  return std::vector<SpatialPhysicalRefinementView>();
}

template <typename Endpoint>
llvm::Expected<std::uint32_t>
transportPayloadWidth(const ::dataflow::CanonicalDataflowProgramView &dataflow,
                      const Endpoint &endpoint) {
  auto type = dataflow.tokenType(endpoint);
  if (!type)
    return type.takeError();
  return dataflow.transportPayloadBitWidth(*type);
}

const ::loom::fabric::FabricPhysicalTraversalView *
findTraversal(const ::loom::fabric::FabricArtifactView &fabric,
              const ::loom::fabric::FabricPhysicalTraversalRef &reference) {
  auto found =
      llvm::find_if(fabric.physicalTraversals(), [&](const auto &candidate) {
        return candidate.reference == reference;
      });
  return found == fabric.physicalTraversals().end() ? nullptr : &*found;
}

const TechComputeRealizationView *
findComputeRealization(const TechMappingView &techMapping,
                       std::uint64_t entity) {
  auto found = llvm::find_if(
      techMapping.computeRealizations(),
      [&](const auto &candidate) { return candidate.entityId == entity; });
  return found == techMapping.computeRealizations().end() ? nullptr : &*found;
}

const SpatialComputeBindingView *
findComputeBinding(llvm::ArrayRef<SpatialComputeBindingView> bindings,
                   std::uint64_t realization) {
  auto found = llvm::find_if(bindings, [&](const auto &candidate) {
    return candidate.realization == realization;
  });
  return found == bindings.end() ? nullptr : &*found;
}

llvm::Expected<SpatialComputeBindingView>
importComputeBinding(::mapping::ComputeBindingOp record,
                     const TechMappingView &techMapping,
                     const ::loom::fabric::FabricArtifactView &fabric) {
  const std::uint64_t realizationEntity = record.getRealization().getEntity();
  const TechComputeRealizationView *realization =
      findComputeRealization(techMapping, realizationEntity);
  if (!realization)
    return invalid("ComputeBinding references an absent Tech realization");
  auto occurrence = decodeFabric<::loom::fabric::FabricFuOccurrenceRef>(
      record.getOccurrence());
  if (!occurrence)
    return occurrence.takeError();
  auto context =
      decodeFabric<::loom::fabric::InstructionContextRef>(record.getContext());
  if (!context)
    return context.takeError();
  if (llvm::Error error =
          ::loom::fabric::validateFabricRef(fabric, *occurrence))
    return std::move(error);
  if (llvm::Error error = ::loom::fabric::validateFabricRef(fabric, *context))
    return std::move(error);
  auto definition = fabric.fuTemplateOf(*occurrence);
  if (!definition || *definition != realization->capabilityTemplate.fu)
    return invalid("ComputeBinding occurrence has the wrong FU definition");
  auto parentPe = fabric.parentPeOf(*occurrence);
  if (!parentPe || context->pe != *parentPe ||
      context->ordinal >= fabric.peResidentContextCount(*parentPe))
    return invalid("ComputeBinding instruction context is incompatible");
  auto refinements = importRefinements(record.getRefinements());
  if (!refinements)
    return refinements.takeError();
  return SpatialComputeBindingView{realizationEntity, *occurrence, *context,
                                   std::move(*refinements)};
}

struct TerminalProjectionContext final {
  const ::dataflow::CanonicalDataflowProgramView &dataflow;
  const TechMappingView &techMapping;
  const ::loom::fabric::FabricArtifactView &fabric;
  llvm::ArrayRef<SpatialComputeBindingView> computeBindings;
  llvm::ArrayRef<SpatialMemoryEngineBindingView> memoryBindings;
};

using ActorTerminalAttachments =
    std::variant<std::vector<::loom::fabric::FabricFuPortAttachmentView>,
                 ::loom::fabric::FabricTransportEndpointRef>;

template <typename ActorTerminal>
llvm::Expected<ActorTerminalAttachments>
actorTerminalAttachments(const TerminalProjectionContext &context,
                         const ActorTerminal &terminal) {
  const TechComputeRealizationView *computeOwner = nullptr;
  for (const auto &realization : context.techMapping.computeRealizations()) {
    if (llvm::any_of(realization.actors, [&](const auto &actor) {
          return actor.actor == terminal.actor;
        })) {
      if (computeOwner)
        return invalid("actor belongs to multiple Compute Realizations");
      computeOwner = &realization;
    }
  }
  const TechMemoryRealizationView *memoryOwner = nullptr;
  const TechMemoryActorView *memoryActor = nullptr;
  for (const auto &realization : context.techMapping.memoryRealizations()) {
    for (const auto &actor : realization.actors) {
      if (actor.actor != terminal.actor)
        continue;
      if (memoryOwner)
        return invalid("actor belongs to multiple Memory Realizations");
      memoryOwner = &realization;
      memoryActor = &actor;
    }
  }
  if (computeOwner && memoryOwner)
    return invalid("actor belongs to compute and memory realizations");
  if (!computeOwner && !memoryOwner)
    return invalid("route terminal actor has no Tech realization");

  const auto direction =
      std::is_same_v<ActorTerminal, ::dataflow::ActorTokenOperandRef>
          ? ::loom::fabric::FabricPortDirection::Input
          : ::loom::fabric::FabricPortDirection::Output;
  if (memoryOwner) {
    auto endpoint = resolveTechMemoryActorTerminal(context.dataflow,
                                                   *memoryActor, terminal);
    if (!endpoint)
      return endpoint.takeError();
    auto binding = llvm::find_if(context.memoryBindings, [&](const auto &row) {
      return row.realization == memoryOwner->entityId;
    });
    if (binding == context.memoryBindings.end())
      return invalid("route terminal owner has no MemoryEngineBinding");
    return ActorTerminalAttachments(
        std::in_place_type<::loom::fabric::FabricTransportEndpointRef>,
        ::loom::fabric::FabricTransportEndpointRef{
            ::loom::fabric::FabricTransportEndpointOwnerRef::of(
                binding->occurrence),
            endpoint->ordinal});
  }

  const TechComputeBoundaryView *boundary = nullptr;
  for (const auto &candidate : computeOwner->boundaries) {
    if (candidate.actor == terminal.actor && candidate.direction == direction &&
        candidate.portOrdinal == terminal.ordinal) {
      if (boundary)
        return invalid("actor terminal has duplicate boundary witnesses");
      boundary = &candidate;
    }
  }
  if (!boundary)
    return invalid("route terminal has no Tech boundary witness");
  const SpatialComputeBindingView *binding =
      findComputeBinding(context.computeBindings, computeOwner->entityId);
  if (!binding)
    return invalid("route terminal owner has no ComputeBinding");
  const auto attachments = context.fabric.fuOccurrencePortAttachments(
      ::loom::fabric::FabricFuOccurrencePortRef{binding->occurrence, direction,
                                                boundary->fabricPort.ordinal});
  if (attachments.empty())
    return invalid("ComputeBinding terminal has no local attachment");
  return ActorTerminalAttachments(
      std::in_place_type<
          std::vector<::loom::fabric::FabricFuPortAttachmentView>>,
      attachments.begin(), attachments.end());
}

template <typename Endpoint>
llvm::Expected<std::vector<::loom::fabric::FabricTransportEndpointRef>>
terminalDomain(const TerminalProjectionContext &context,
               const Endpoint &endpoint) {
  auto width = transportPayloadWidth(context.dataflow, endpoint);
  if (!width)
    return width.takeError();
  std::vector<::loom::fabric::FabricTransportEndpointRef> result;
  const auto appendActorDomain =
      [&](const auto &actor,
          ::loom::fabric::FabricPortDirection direction) -> llvm::Error {
    auto attachments = actorTerminalAttachments(context, actor);
    if (!attachments)
      return attachments.takeError();
    return std::visit(
        [&](const auto &selected) -> llvm::Error {
          using Selection = std::decay_t<decltype(selected)>;
          if constexpr (std::is_same_v<
                            Selection,
                            std::vector<
                                ::loom::fabric::FabricFuPortAttachmentView>>) {
            for (const auto &attachment : selected) {
              auto path =
                  context.fabric.transportEndpointDataPath(attachment.endpoint);
              if (context.fabric.transportEndpointDirection(
                      attachment.endpoint) == direction &&
                  path && path->payloadWidthBits >= *width)
                result.push_back(attachment.endpoint);
            }
          } else {
            auto path = context.fabric.transportEndpointDataPath(selected);
            if (context.fabric.transportEndpointDirection(selected) ==
                    direction &&
                path && path->payloadWidthBits >= *width)
              result.push_back(selected);
          }
          return llvm::Error::success();
        },
        *attachments);
  };
  bool graphBoundary = true;
  if constexpr (std::is_same_v<Endpoint,
                               ::dataflow::CanonicalGraphProducerEndpointRef>) {
    if (const auto *actor =
            std::get_if<::dataflow::ActorTokenResultRef>(&endpoint)) {
      graphBoundary = false;
      if (llvm::Error error = appendActorDomain(
              *actor, ::loom::fabric::FabricPortDirection::Output))
        return std::move(error);
    }
  } else {
    if (const auto *actor =
            std::get_if<::dataflow::ActorTokenOperandRef>(&endpoint)) {
      graphBoundary = false;
      if (llvm::Error error = appendActorDomain(
              *actor, ::loom::fabric::FabricPortDirection::Input))
        return std::move(error);
    }
  }
  if (graphBoundary) {
    const auto direction =
        std::is_same_v<Endpoint, ::dataflow::CanonicalGraphProducerEndpointRef>
            ? ::loom::fabric::FabricPortDirection::Input
            : ::loom::fabric::FabricPortDirection::Output;
    for (const auto &attachment :
         context.fabric.moduleBoundaryTransportAttachments()) {
      if (attachment.boundary.direction != direction ||
          context.fabric.transportEndpointDirection(attachment.endpoint) !=
              direction)
        continue;
      auto path = context.fabric.transportEndpointDataPath(attachment.endpoint);
      if (path && path->payloadWidthBits >= *width)
        result.push_back(attachment.endpoint);
    }
  }
  llvm::sort(result, [](const auto &left, const auto &right) {
    return ::loom::fabric::canonicalFabricBytes(left) <
           ::loom::fabric::canonicalFabricBytes(right);
  });
  result.erase(std::unique(result.begin(), result.end()), result.end());
  if (result.empty())
    return invalid("route terminal has no compatible Fabric endpoint");
  return result;
}

template <typename Endpoint>
llvm::Expected<std::optional<::loom::fabric::FabricPhysicalTraversalRef>>
selectedLocalTraversal(
    const TerminalProjectionContext &context, const Endpoint &terminal,
    const ::loom::fabric::FabricTransportEndpointRef &selected) {
  const auto actorTraversal = [&](const auto &actor)
      -> llvm::Expected<
          std::optional<::loom::fabric::FabricPhysicalTraversalRef>> {
    auto attachments = actorTerminalAttachments(context, actor);
    if (!attachments)
      return attachments.takeError();
    return std::visit(
        [&](const auto &domain)
            -> llvm::Expected<
                std::optional<::loom::fabric::FabricPhysicalTraversalRef>> {
          using Domain = std::decay_t<decltype(domain)>;
          if constexpr (std::is_same_v<
                            Domain,
                            std::vector<
                                ::loom::fabric::FabricFuPortAttachmentView>>) {
            auto found = llvm::find_if(domain, [&](const auto &attachment) {
              return attachment.endpoint == selected;
            });
            if (found == domain.end())
              return invalid(
                  "selected actor terminal has no local traversal witness");
            return std::optional<::loom::fabric::FabricPhysicalTraversalRef>(
                found->localTraversal);
          } else {
            if (domain != selected)
              return invalid(
                  "selected memory terminal differs from its exact endpoint");
            return std::optional<::loom::fabric::FabricPhysicalTraversalRef>();
          }
        },
        *attachments);
  };
  if constexpr (std::is_same_v<Endpoint,
                               ::dataflow::CanonicalGraphProducerEndpointRef>) {
    if (const auto *actor =
            std::get_if<::dataflow::ActorTokenResultRef>(&terminal))
      return actorTraversal(*actor);
  } else {
    if (const auto *actor =
            std::get_if<::dataflow::ActorTokenOperandRef>(&terminal))
      return actorTraversal(*actor);
  }
  return std::optional<::loom::fabric::FabricPhysicalTraversalRef>();
}

template <typename Endpoint>
llvm::Error requireTerminalEndpoint(
    const TerminalProjectionContext &context, const Endpoint &terminal,
    const ::loom::fabric::FabricTransportEndpointRef &selected) {
  auto domain = terminalDomain(context, terminal);
  if (!domain)
    return domain.takeError();
  if (!llvm::is_contained(*domain, selected))
    return invalid("RouteTree endpoint is outside its terminal domain");
  return llvm::Error::success();
}

llvm::Expected<SpatialRouteTreeView>
importRouteTree(::mapping::RouteTreeOp record,
                const TerminalProjectionContext &context,
                const TechResidualLogicalNetView &residual) {
  auto logicalNet =
      decodeDataflow<::dataflow::CanonicalGraphProducerEndpointRef>(
          record.getLogicalNet(), context.dataflow.identity());
  if (!logicalNet)
    return logicalNet.takeError();
  if (*logicalNet != residual.producer)
    return invalid("RouteTree logical-net key disagrees with TechMapping");
  auto rootEndpoint = decodeFabric<::loom::fabric::FabricTransportEndpointRef>(
      record.getRootEndpoint());
  if (!rootEndpoint)
    return rootEndpoint.takeError();
  if (llvm::Error error =
          ::loom::fabric::validateFabricRef(context.fabric, *rootEndpoint))
    return std::move(error);
  if (llvm::Error error =
          requireTerminalEndpoint(context, *logicalNet, *rootEndpoint))
    return std::move(error);
  auto payloadWidth = transportPayloadWidth(context.dataflow, *logicalNet);
  if (!payloadWidth)
    return payloadWidth.takeError();

  auto rootLocalTraversal =
      selectedLocalTraversal(context, *logicalNet, *rootEndpoint);
  if (!rootLocalTraversal)
    return rootLocalTraversal.takeError();
  SpatialRouteTreeView result{
      *logicalNet, *rootEndpoint, std::move(*rootLocalTraversal), {}, {}};
  std::set<std::vector<std::uint8_t>> endpoints;
  result.nodes.reserve(std::distance(
      record.getBody().front().getOps<::mapping::RouteNodeOp>().begin(),
      record.getBody().front().getOps<::mapping::RouteNodeOp>().end()));
  for (auto node : record.getBody().front().getOps<::mapping::RouteNodeOp>()) {
    if (node.getNodeOrdinal() != result.nodes.size())
      return invalid("RouteTree node ordinals are not canonical");
    std::optional<std::uint64_t> parent = node.getParentNodeOrdinal();
    std::optional<::loom::fabric::FabricPhysicalTraversalRef> traversal;
    ::loom::fabric::FabricTransportEndpointRef endpoint = *rootEndpoint;
    if (parent) {
      if (*parent >= result.nodes.size())
        return invalid("RouteTree parent is not in preorder");
      auto decoded = decodeFabric<::loom::fabric::FabricPhysicalTraversalRef>(
          *node.getIncomingTraversal());
      if (!decoded)
        return decoded.takeError();
      if (llvm::Error error =
              ::loom::fabric::validateFabricRef(context.fabric, *decoded))
        return std::move(error);
      const auto *physical = findTraversal(context.fabric, *decoded);
      if (!physical || physical->destinations.size() != 1 ||
          !llvm::is_contained(physical->sources,
                              result.nodes[*parent].endpoint))
        return invalid("RouteTree traversal is not continuous");
      endpoint = physical->destinations.front();
      traversal = *decoded;
    } else if (node.getNodeOrdinal() != 0) {
      return invalid("nonzero RouteTree node has no parent");
    }
    auto path = context.fabric.transportEndpointDataPath(endpoint);
    if (!path || path->payloadWidthBits < *payloadWidth)
      return invalid("RouteTree segment narrows below its logical payload");
    if (!endpoints.insert(::loom::fabric::canonicalFabricBytes(endpoint))
             .second)
      return invalid("RouteTree contains a repeated physical endpoint");
    auto refinements = importRefinements(node.getRefinements());
    if (!refinements)
      return refinements.takeError();
    result.nodes.push_back(SpatialRouteNodeView{node.getNodeOrdinal(), endpoint,
                                                parent, traversal,
                                                std::move(*refinements)});
  }

  std::map<std::string, ::dataflow::CanonicalGraphConsumerEndpointRef>
      requiredSinks;
  for (const auto &sink : residual.sinks) {
    auto key =
        ::dataflow::encodeDataflowReference(context.dataflow.identity(), sink);
    if (!key)
      return key.takeError();
    requiredSinks.emplace(byteKey(*key), sink);
  }
  for (auto attachment :
       record.getBody().front().getOps<::mapping::RouteSinkOp>()) {
    auto sink = decodeDataflow<::dataflow::CanonicalGraphConsumerEndpointRef>(
        attachment.getSink(), context.dataflow.identity());
    if (!sink)
      return sink.takeError();
    auto key =
        ::dataflow::encodeDataflowReference(context.dataflow.identity(), *sink);
    if (!key)
      return key.takeError();
    auto required = requiredSinks.find(byteKey(*key));
    if (required == requiredSinks.end())
      return invalid("RouteTree contains a non-residual sink");
    if (attachment.getNodeOrdinal() >= result.nodes.size())
      return invalid("RouteTree sink names an absent node");
    if (llvm::Error error = requireTerminalEndpoint(
            context, *sink, result.nodes[attachment.getNodeOrdinal()].endpoint))
      return std::move(error);
    auto localTraversal = selectedLocalTraversal(
        context, *sink, result.nodes[attachment.getNodeOrdinal()].endpoint);
    if (!localTraversal)
      return localTraversal.takeError();
    result.sinks.push_back(SpatialRouteSinkView{
        *sink, attachment.getNodeOrdinal(), std::move(*localTraversal)});
    requiredSinks.erase(required);
  }
  if (!requiredSinks.empty())
    return invalid("RouteTree omits a residual sink obligation");
  return result;
}

llvm::Expected<SpatialRegisterFifoTransferView>
importRegisterFifoTransfer(::mapping::RegisterFifoTransferOp record,
                           const TerminalProjectionContext &context,
                           const TechResidualLogicalNetView &residual) {
  auto logicalNet =
      decodeDataflow<::dataflow::CanonicalGraphProducerEndpointRef>(
          record.getLogicalNet(), context.dataflow.identity());
  auto sink = decodeDataflow<::dataflow::CanonicalGraphConsumerEndpointRef>(
      record.getSink(), context.dataflow.identity());
  auto write = decodeFabric<::loom::fabric::FabricPhysicalTraversalRef>(
      record.getWriteTraversal());
  auto read = decodeFabric<::loom::fabric::FabricPhysicalTraversalRef>(
      record.getReadTraversal());
  if (!logicalNet)
    return logicalNet.takeError();
  if (!sink)
    return sink.takeError();
  if (!write)
    return write.takeError();
  if (!read)
    return read.takeError();
  if (*logicalNet != residual.producer || residual.sinks.size() != 1 ||
      *sink != residual.sinks.front())
    return invalid("register-FIFO transfer does not name one exact residual "
                   "edge");

  auto options = deriveSpatialPeLocalTransferOptions(
      context.dataflow, context.techMapping, context.fabric,
      context.computeBindings, residual);
  if (!options)
    return options.takeError();
  const SpatialPeLocalTransferOptionView *selected = nullptr;
  for (const SpatialPeLocalTransferOptionView &option : *options) {
    if (option.writeTraversal != *write || option.readTraversal != *read)
      continue;
    if (selected)
      return invalid("register-FIFO transfer matches multiple physical "
                     "alternatives");
    selected = &option;
  }
  if (!selected)
    return invalid("register-FIFO transfer is outside its derived physical "
                   "alternative domain");
  return SpatialRegisterFifoTransferView{
      selected->producer,     selected->sink,           selected->pe,
      selected->registerFifo, selected->writeTraversal, selected->readTraversal,
      selected->tag};
}

llvm::Expected<SpatialActivityEventRef>
importEvent(Attribute attribute,
            const ::dataflow::CanonicalDataflowProgramView &dataflow) {
  if (auto transition =
          dyn_cast<::mapping::ActorTransitionEventAttr>(attribute)) {
    auto actor = decodeDataflow<::dataflow::ActorRef>(transition.getActor(),
                                                      dataflow.identity());
    if (!actor)
      return actor.takeError();
    auto resolved = dataflow.resolve(*actor);
    if (!resolved)
      return resolved.takeError();
    auto projection =
        ::dataflow::projectRegisteredActorSchemaProjection(resolved->op);
    if (!projection)
      return projection.takeError();
    auto cases = ::dataflow::semantics::projectActorHandshakeCases(
        projection->schema, resolved->op->getNumOperands(),
        resolved->op->getNumResults());
    if (!cases)
      return cases.takeError();
    if (transition.getTransition() >= cases->size() ||
        (*cases)[transition.getTransition()].ordinal !=
            transition.getTransition())
      return invalid("ResourceUse actor transition is out of range");
    return SpatialActivityEventRef(
        SpatialActorTransitionEventRef{*actor, transition.getTransition()});
  }
  if (auto producer =
          dyn_cast<::mapping::GraphProducerEndpointRefAttr>(attribute)) {
    auto decoded =
        decodeDataflow<::dataflow::CanonicalGraphProducerEndpointRef>(
            producer, dataflow.identity());
    if (!decoded)
      return decoded.takeError();
    if (llvm::Error error = dataflow.validate(*decoded))
      return std::move(error);
    return SpatialActivityEventRef(*decoded);
  }
  auto decoded = decodeDataflow<::dataflow::CanonicalGraphConsumerEndpointRef>(
      cast<::mapping::GraphConsumerEndpointRefAttr>(attribute),
      dataflow.identity());
  if (!decoded)
    return decoded.takeError();
  if (llvm::Error error = dataflow.validate(*decoded))
    return std::move(error);
  return SpatialActivityEventRef(*decoded);
}

llvm::Expected<SpatialEventPointView>
importEventPoint(::mapping::SpatialEventPointAttr point,
                 const ::dataflow::CanonicalDataflowProgramView &dataflow) {
  auto event = importEvent(point.getEvent(), dataflow);
  if (!event)
    return event.takeError();
  if (point.getGuaranteedOffset())
    return invalid("guaranteed event offset requires its owner timing codec");
  return SpatialEventPointView{std::move(*event), std::nullopt};
}

using RequiredComputeUse = SpatialComputeUseRequirement;
using RequiredMemoryUse = detail::SpatialMemoryResourceUseRequirement;

llvm::Expected<std::string>
requiredUseKey(const RequiredComputeUse &use,
               const ArtifactIdentity &dataflowIdentity) {
  std::string result;
  auto appendU64 = [&](std::uint64_t value) {
    for (unsigned byte = 0; byte < 8; ++byte)
      result.push_back(static_cast<char>(value >> (8 * (7 - byte))));
  };
  auto appendFramed = [&](llvm::ArrayRef<std::uint8_t> bytes) {
    appendU64(bytes.size());
    result.append(reinterpret_cast<const char *>(bytes.data()), bytes.size());
  };
  appendU64(use.realization);
  auto encodedEvent =
      encodeSpatialActivityEventKey(dataflowIdentity, use.trigger);
  if (!encodedEvent)
    return encodedEvent.takeError();
  appendFramed(*encodedEvent);
  appendFramed(::loom::fabric::canonicalFabricBytes(use.pattern));
  appendU64(use.release.size());
  for (const auto &release : use.release) {
    auto encodedRelease =
        encodeSpatialActivityEventKey(dataflowIdentity, release);
    if (!encodedRelease)
      return encodedRelease.takeError();
    appendFramed(*encodedRelease);
  }
  return result;
}

llvm::Expected<std::map<std::string, RequiredComputeUse>>
deriveRequiredComputeUses(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const TechMappingView &techMapping,
    const ::loom::fabric::FabricArtifactView &fabric,
    llvm::ArrayRef<SpatialComputeBindingView> bindings,
    llvm::ArrayRef<SpatialRegisterFifoTransferView> registerFifoTransfers) {
  std::map<std::string, RequiredComputeUse> result;
  auto requirements = deriveSpatialComputeUseRequirements(
      dataflow, techMapping, fabric, bindings, registerFifoTransfers);
  if (!requirements)
    return requirements.takeError();
  for (const auto &use : *requirements) {
    auto key = requiredUseKey(use, dataflow.identity());
    if (!key)
      return key.takeError();
    if (!result.emplace(std::move(*key), use).second)
      return invalid("compute owner derives a duplicate ResourceUse");
  }
  return result;
}

llvm::Expected<std::string>
requiredMemoryUseKey(const RequiredMemoryUse &use,
                     const ArtifactIdentity &dataflowIdentity) {
  std::string result;
  std::visit(
      [&](const auto &owner) {
        using Owner = std::decay_t<decltype(owner)>;
        std::uint64_t value = 0;
        if constexpr (std::is_same_v<Owner,
                                     SpatialMemoryEngineResourceOwnerRef>) {
          result.push_back(0);
          value = owner.realization;
        } else {
          result.push_back(1);
          value = owner.binding;
        }
        for (unsigned byte = 0; byte < 8; ++byte)
          result.push_back(static_cast<char>(value >> (8 * (7 - byte))));
      },
      use.owner);
  auto encodedEvent =
      encodeSpatialActivityEventKey(dataflowIdentity, use.trigger);
  if (!encodedEvent)
    return encodedEvent.takeError();
  for (unsigned byte = 0; byte < 8; ++byte)
    result.push_back(
        static_cast<char>(encodedEvent->size() >> (8 * (7 - byte))));
  result.append(reinterpret_cast<const char *>(encodedEvent->data()),
                encodedEvent->size());
  return result;
}

llvm::Expected<std::map<std::string, RequiredMemoryUse>>
deriveRequiredMemoryUses(const detail::ImportedSpatialMemoryView &memory,
                         const ArtifactIdentity &dataflowIdentity) {
  std::map<std::string, RequiredMemoryUse> result;
  for (const auto &use : memory.requiredResourceUses) {
    auto key = requiredMemoryUseKey(use, dataflowIdentity);
    if (!key)
      return key.takeError();
    if (!result.emplace(std::move(*key), use).second)
      return invalid("memory owner derives a duplicate ResourceUse");
  }
  return result;
}

llvm::Expected<SpatialResourceOwnerRef> importSpatialResourceOwner(
    mlir::Attribute attribute,
    const ::dataflow::CanonicalDataflowProgramView &dataflow) {
  if (auto compute = dyn_cast<::mapping::ComputeRealizationRefAttr>(attribute))
    return SpatialResourceOwnerRef(
        SpatialComputeResourceOwnerRef{compute.getEntity()});
  if (auto engine = dyn_cast<::mapping::MemoryRealizationRefAttr>(attribute))
    return SpatialResourceOwnerRef(
        SpatialMemoryEngineResourceOwnerRef{engine.getEntity()});
  if (auto binding = dyn_cast<::mapping::MemoryBindingRefAttr>(attribute))
    return SpatialResourceOwnerRef(
        SpatialMemoryBindingResourceOwnerRef{binding.getEntity()});
  if (auto route = dyn_cast<::mapping::RouteTreeNodeRefAttr>(attribute)) {
    auto logicalNet =
        decodeDataflow<::dataflow::CanonicalGraphProducerEndpointRef>(
            route.getLogicalNet(), dataflow.identity());
    if (!logicalNet)
      return logicalNet.takeError();
    return SpatialResourceOwnerRef(
        SpatialRouteNodeResourceOwnerRef{*logicalNet, route.getNodeOrdinal()});
  }
  return invalid("ResourceUse has an unsupported Spatial owner");
}

llvm::Expected<SpatialResourceUseView> importResourceUse(
    ::mapping::ResourceUseOp record,
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::fabric::FabricArtifactView &fabric,
    std::map<std::string, RequiredComputeUse> &requiredCompute,
    std::map<std::string, RequiredMemoryUse> &requiredMemory,
    std::map<std::string, detail::RequiredPhysicalTagUse> &requiredTags,
    std::map<::loom::fabric::FabricOrdinal, std::set<std::string>>
        &assignedDomainValues,
    std::uint64_t resourceUseOrdinal,
    std::optional<SpatialPhysicalTagSegmentView> &physicalTagSegment) {
  physicalTagSegment.reset();
  auto pattern =
      decodeFabric<::loom::fabric::FabricUsePatternRef>(record.getUseSite());
  if (!pattern)
    return pattern.takeError();
  if (llvm::Error error = ::loom::fabric::validateFabricRef(fabric, *pattern))
    return std::move(error);
  auto activation = mlir::dyn_cast<::mapping::SpatialRelativeActivationAttr>(
      record.getActivation());
  if (!activation)
    return invalid("Spatial ResourceUse has a non-Spatial activation");
  auto trigger = importEventPoint(activation.getTrigger(), dataflow);
  if (!trigger)
    return trigger.takeError();
  if (trigger->guaranteedOffset)
    return invalid("ResourceUse trigger has an unsupported guaranteed offset");
  std::vector<SpatialEventPointView> release;
  release.reserve(activation.getRelease().size());
  for (mlir::Attribute attribute : activation.getRelease()) {
    auto imported = importEventPoint(
        mlir::cast<::mapping::SpatialEventPointAttr>(attribute), dataflow);
    if (!imported)
      return imported.takeError();
    if (imported->guaranteedOffset)
      return invalid(
          "ResourceUse release has an unsupported guaranteed offset");
    release.push_back(std::move(*imported));
  }
  auto values =
      detail::importResourceUsePatternValues(record, fabric, *pattern);
  if (!values)
    return values.takeError();
  auto importedOwner = importSpatialResourceOwner(record.getOwner(), dataflow);
  if (!importedOwner)
    return importedOwner.takeError();

  auto tagKey = detail::physicalTagUseKey(*importedOwner, trigger->event,
                                          *pattern, dataflow.identity());
  if (!tagKey)
    return tagKey.takeError();
  auto tagUse = requiredTags.find(*tagKey);
  if (tagUse != requiredTags.end()) {
    if (!values->parameters.empty() || values->sharingAssignments.size() != 1)
      return invalid(
          "Physical Tag ResourceUse has the wrong owner-typed value shape");
    const auto *tag = std::get_if<::fabric::PhysicalTagPatternValue>(
        &values->sharingAssignments.front());
    if (!tag ||
        tag->value.getBitWidth() != tagUse->second.assignmentPoint.tagWidthBits)
      return invalid(
          "Physical Tag ResourceUse value disagrees with its assignment point");
    auto encoded = ::fabric::encodeUsePatternValue(
        ::fabric::UsePatternValueSchema::physicalTag(
            tagUse->second.assignmentPoint.tagWidthBits),
        values->sharingAssignments.front());
    if (!encoded)
      return encoded.takeError();
    const std::string valueKey = byteKey(*encoded);
    const auto matchDomains = fabric.physicalTagMatchDomains();
    for (::loom::fabric::FabricOrdinal domain : tagUse->second.matchDomains) {
      if (domain >= matchDomains.size())
        return invalid("Physical Tag assignment names an absent match domain");
      if (matchDomains[domain].kind ==
          ::loom::fabric::FabricPhysicalTagMatchDomainKind::TemporalSwitchTable)
        continue;
      if (!assignedDomainValues[domain].insert(valueKey).second)
        return invalid(
            "Physical Tag assignments collide in one Fabric match domain");
    }
    physicalTagSegment = SpatialPhysicalTagSegmentView{
        tagUse->second.routeTreeOrdinal, tagUse->second.segmentOrdinal,
        tagUse->second.nodeOrdinals, resourceUseOrdinal};
    requiredTags.erase(tagUse);
    if (!release.empty())
      return invalid("Physical Tag ResourceUse requires intrinsic release");
    return SpatialResourceUseView{
        std::move(*importedOwner), *pattern,
        SpatialRelativeActivationView{std::move(*trigger), {}},
        std::move(values->parameters), std::move(values->sharingAssignments)};
  }

  if (auto compute =
          dyn_cast<::mapping::ComputeRealizationRefAttr>(record.getOwner())) {
    std::vector<SpatialActivityEventRef> releaseEvents;
    releaseEvents.reserve(release.size());
    for (const auto &point : release)
      releaseEvents.push_back(point.event);
    RequiredComputeUse keyValue{compute.getEntity(), trigger->event, *pattern,
                                std::move(releaseEvents)};
    auto key = requiredUseKey(keyValue, dataflow.identity());
    if (!key)
      return key.takeError();
    auto found = requiredCompute.find(*key);
    if (found == requiredCompute.end())
      return invalid("ResourceUse is not required by its compute owner");
    requiredCompute.erase(found);
    return SpatialResourceUseView{
        SpatialComputeResourceOwnerRef{compute.getEntity()}, *pattern,
        SpatialRelativeActivationView{std::move(*trigger), std::move(release)},
        std::move(values->parameters), std::move(values->sharingAssignments)};
  }

  detail::SpatialMemoryResourceOwnerRef memoryOwner =
      SpatialMemoryEngineResourceOwnerRef{};
  if (auto engine =
          dyn_cast<::mapping::MemoryRealizationRefAttr>(record.getOwner())) {
    memoryOwner = SpatialMemoryEngineResourceOwnerRef{engine.getEntity()};
  } else if (auto binding =
                 dyn_cast<::mapping::MemoryBindingRefAttr>(record.getOwner())) {
    memoryOwner = SpatialMemoryBindingResourceOwnerRef{binding.getEntity()};
  } else {
    return invalid("ResourceUse has an unsupported Spatial owner");
  }
  RequiredMemoryUse keyValue{std::move(memoryOwner), trigger->event, {}};
  auto key = requiredMemoryUseKey(keyValue, dataflow.identity());
  if (!key)
    return key.takeError();
  auto found = requiredMemory.find(*key);
  if (found == requiredMemory.end() ||
      !llvm::is_contained(found->second.admissiblePatterns, *pattern))
    return invalid("ResourceUse is not admitted by its memory owner");
  requiredMemory.erase(found);
  if (!release.empty())
    return invalid("memory ResourceUse requires intrinsic release");
  return SpatialResourceUseView{
      std::move(*importedOwner), *pattern,
      SpatialRelativeActivationView{std::move(*trigger), {}},
      std::move(values->parameters), std::move(values->sharingAssignments)};
}

llvm::Expected<std::optional<::loom::PointerLayout>>
pointerLayoutFor(const ::dataflow::CanonicalDataflowProgramView &dataflow,
                 const ::dataflow::CanonicalActorSchemaProjection &actor) {
  auto addressSpace = ::dataflow::projectActorPointerAddressSpace(actor);
  if (!addressSpace)
    return addressSpace.takeError();
  if (!*addressSpace)
    return std::optional<::loom::PointerLayout>();
  auto layout = dataflow.pointerLayout(**addressSpace);
  if (!layout)
    return layout.takeError();
  return std::optional<::loom::PointerLayout>(*layout);
}

struct SelectedHandshakeProjection final {
  ::loom::fabric::FabricHandshakeSelection selection;
  std::vector<std::vector<::loom::fabric::FabricPhysicalTraversalRef>>
      routeTraversals;
};

llvm::Expected<SelectedHandshakeProjection> deriveSelectedHandshakeSelection(
    const TerminalProjectionContext &context,
    llvm::ArrayRef<SpatialRegisterFifoTransferView> registerFifoTransfers,
    llvm::ArrayRef<SpatialRouteTreeView> routes,
    llvm::ArrayRef<SpatialResourceUseView> uses,
    llvm::ArrayRef<SpatialPhysicalTagSegmentView> physicalTagSegments) {
  SelectedHandshakeProjection result;
  auto &selection = result.selection;
  for (const auto &realization : context.techMapping.computeRealizations()) {
    const auto *binding =
        findComputeBinding(context.computeBindings, realization.entityId);
    if (!binding)
      return invalid("selected handshake compute owner has no binding");
    std::vector<::loom::fabric::FabricFuOperationHandshakeBinding>
        actorBindings;
    actorBindings.reserve(realization.actors.size());
    for (const auto &actorBinding : realization.actors) {
      auto actor = context.dataflow.resolve(actorBinding.actor);
      if (!actor)
        return actor.takeError();
      auto projection =
          ::dataflow::projectRegisteredActorSchemaProjection(actor->op);
      if (!projection)
        return projection.takeError();
      auto indexWidth = ::loom::getIndexBitWidth(actor->op);
      if (!indexWidth)
        return indexWidth.takeError();
      auto pointerLayout = pointerLayoutFor(context.dataflow, *projection);
      if (!pointerLayout)
        return pointerLayout.takeError();
      actorBindings.push_back(
          {actorBinding.fabricOperation, std::move(*projection), *indexWidth,
           std::move(*pointerLayout), actorBinding.operandPorts,
           actorBinding.resultPorts});
    }
    auto fuSelection = ::loom::fabric::makeFuHandshakeSelection(
        context.fabric, binding->occurrence, realization.capabilityTemplate,
        actorBindings);
    if (!fuSelection)
      return fuSelection.takeError();
    selection.fuCapabilities.push_back(std::move(*fuSelection));
  }

  auto memorySelections = detail::deriveSpatialMemoryHandshakeSelections(
      context.dataflow, context.techMapping, context.fabric,
      context.memoryBindings, uses);
  if (!memorySelections)
    return memorySelections.takeError();
  selection.memoryOperations = std::move(*memorySelections);

  const auto appendTraversal =
      [&](const ::loom::fabric::FabricPhysicalTraversalRef &traversal,
          std::vector<::loom::fabric::FabricPhysicalTraversalRef> &route) {
        route.push_back(traversal);
        const auto *sw =
            std::get_if<::loom::fabric::FabricSwitchTraversalPayload>(
                &traversal.payload);
        if (!sw || context.fabric.switchSchedule(sw->owner) !=
                       ::fabric::Schedule::Temporal)
          selection.traversals.push_back(traversal);
      };

  for (const SpatialRegisterFifoTransferView &transfer :
       registerFifoTransfers) {
    result.routeTraversals.emplace_back();
    appendTraversal(transfer.writeTraversal, result.routeTraversals.back());
    appendTraversal(transfer.readTraversal, result.routeTraversals.back());
  }

  for (const SpatialRouteTreeView &route : routes) {
    result.routeTraversals.emplace_back();
    auto &routeTraversal = result.routeTraversals.back();
    if (route.localTraversal)
      appendTraversal(*route.localTraversal, routeTraversal);
    for (const SpatialRouteNodeView &node : route.nodes) {
      if (node.incomingTraversal)
        appendTraversal(*node.incomingTraversal, routeTraversal);
    }
    for (const SpatialRouteSinkView &sink : route.sinks) {
      if (sink.nodeOrdinal >= route.nodes.size())
        return invalid("selected handshake sink names an absent route node");
      if (sink.localTraversal)
        appendTraversal(*sink.localTraversal, routeTraversal);
    }
    llvm::sort(routeTraversal, [](const auto &lhs, const auto &rhs) {
      return ::loom::fabric::canonicalFabricBytes(lhs) <
             ::loom::fabric::canonicalFabricBytes(rhs);
    });
    routeTraversal.erase(
        std::unique(routeTraversal.begin(), routeTraversal.end()),
        routeTraversal.end());
  }
  llvm::sort(selection.traversals, [](const auto &lhs, const auto &rhs) {
    return ::loom::fabric::canonicalFabricBytes(lhs) <
           ::loom::fabric::canonicalFabricBytes(rhs);
  });
  selection.traversals.erase(
      std::unique(selection.traversals.begin(), selection.traversals.end()),
      selection.traversals.end());

  auto packedRows = deriveSpatialTemporalSwitchPackedRows(
      context.fabric, routes, uses, physicalTagSegments);
  if (!packedRows)
    return packedRows.takeError();
  std::map<std::vector<std::uint8_t>, ::loom::fabric::FabricOrdinal>
      nextOccurrenceRow;
  for (const SpatialTemporalSwitchPackedRowView &row : *packedRows) {
    const auto occurrenceKey =
        ::loom::fabric::canonicalFabricBytes(row.occurrence);
    const ::loom::fabric::FabricOrdinal rowOrdinal =
        nextOccurrenceRow[occurrenceKey]++;
    std::map<::loom::fabric::FabricOrdinal,
             std::vector<::loom::fabric::FabricPhysicalTraversalRef>>
        byInput;
    for (const SpatialTemporalSwitchRouteSignatureView &signature :
         row.signatures)
      byInput[signature.input].insert(byInput[signature.input].end(),
                                      signature.traversals.begin(),
                                      signature.traversals.end());
    for (auto &[input, traversals] : byInput) {
      llvm::sort(traversals, [](const auto &lhs, const auto &rhs) {
        return ::loom::fabric::canonicalFabricBytes(lhs) <
               ::loom::fabric::canonicalFabricBytes(rhs);
      });
      traversals.erase(std::unique(traversals.begin(), traversals.end()),
                       traversals.end());
      selection.switchActivations.push_back(
          {{row.occurrence, rowOrdinal, input}, std::move(traversals)});
    }
  }
  return result;
}

struct ImportedSpatialView final {
  ArtifactIdentity techMappingIdentity;
  ArtifactIdentity dataflowIdentity;
  ArtifactIdentity fabricIdentity;
  std::vector<SpatialComputeBindingView> computeBindings;
  std::vector<SpatialMemoryEngineBindingView> memoryEngineBindings;
  std::vector<SpatialMemoryBindingView> memoryBindings;
  std::vector<SpatialRegisterFifoTransferView> registerFifoTransfers;
  std::vector<SpatialRouteTreeView> routeTrees;
  std::vector<SpatialResourceUseView> resourceUses;
  std::vector<SpatialPhysicalTagSegmentView> physicalTagSegments;
  ConfiguredHardwareProjectionView configuredHardware;
  ::loom::fabric::FabricHandshakeSelection handshakeSelection;
};

llvm::Expected<ImportedSpatialView>
importView(const ArtifactIdentity &mappingIdentity, ::mapping::SpatialOp root,
           const ::dataflow::CanonicalDataflowProgramView &dataflow,
           const TechMappingView &techMapping,
           const ::loom::fabric::FabricArtifactView &fabric,
           const ::loom::fabric::FabricHandshakeContext *handshakeContext) {
  (void)mappingIdentity;
  auto techIdentity = decodeIdentity(root.getTechMapping());
  auto dataflowIdentity = decodeIdentity(root.getDataflow());
  auto fabricIdentity = decodeIdentity(root.getFabric());
  if (!techIdentity)
    return techIdentity.takeError();
  if (!dataflowIdentity)
    return dataflowIdentity.takeError();
  if (!fabricIdentity)
    return fabricIdentity.takeError();
  if (*techIdentity != techMapping.identity() ||
      *dataflowIdentity != dataflow.identity() ||
      *fabricIdentity != fabric.identity())
    return invalid("SpatialMapping upstream binding does not match importer");
  if (techMapping.dataflowIdentity() != dataflow.identity() ||
      techMapping.fabricIdentity() != fabric.identity())
    return invalid(
        "TechMapping upstream closure disagrees with SpatialMapping");
  std::vector<SpatialComputeBindingView> computeBindings;
  std::set<std::uint64_t> boundComputes;
  for (auto record :
       root.getBody().front().getOps<::mapping::ComputeBindingOp>()) {
    auto binding = importComputeBinding(record, techMapping, fabric);
    if (!binding)
      return binding.takeError();
    if (!boundComputes.insert(binding->realization).second)
      return invalid("duplicate ComputeBinding realization");
    computeBindings.push_back(std::move(*binding));
  }
  if (computeBindings.size() != techMapping.computeRealizations().size())
    return invalid("SpatialMapping omits a Tech compute realization");
  auto importedMemory =
      detail::importSpatialMemoryView(root, dataflow, techMapping, fabric);
  if (!importedMemory)
    return importedMemory.takeError();

  TerminalProjectionContext terminalContext{dataflow, techMapping, fabric,
                                            computeBindings,
                                            importedMemory->engineBindings};
  std::map<std::string, const TechResidualLogicalNetView *> residual;
  for (const auto &net : techMapping.residualLogicalNets()) {
    auto encoded =
        ::dataflow::encodeDataflowReference(dataflow.identity(), net.producer);
    if (!encoded)
      return encoded.takeError();
    residual.emplace(byteKey(*encoded), &net);
  }
  std::vector<SpatialRegisterFifoTransferView> registerFifoTransfers;
  std::set<std::pair<std::vector<std::uint8_t>, ::loom::fabric::FabricOrdinal>>
      occupiedRegisterFifos;
  for (auto record :
       root.getBody().front().getOps<::mapping::RegisterFifoTransferOp>()) {
    auto producer =
        decodeDataflow<::dataflow::CanonicalGraphProducerEndpointRef>(
            record.getLogicalNet(), dataflow.identity());
    if (!producer)
      return producer.takeError();
    auto encoded =
        ::dataflow::encodeDataflowReference(dataflow.identity(), *producer);
    if (!encoded)
      return encoded.takeError();
    auto found = residual.find(byteKey(*encoded));
    if (found == residual.end())
      return invalid(
          "register-FIFO transfer does not name a residual logical net");
    auto transfer =
        importRegisterFifoTransfer(record, terminalContext, *found->second);
    if (!transfer)
      return transfer.takeError();
    if (!occupiedRegisterFifos
             .emplace(::loom::fabric::canonicalFabricBytes(transfer->pe),
                      transfer->registerFifo)
             .second)
      return invalid("one register FIFO is selected by multiple transfers");
    registerFifoTransfers.push_back(std::move(*transfer));
    residual.erase(found);
  }
  std::vector<SpatialRouteTreeView> routes;
  for (auto record : root.getBody().front().getOps<::mapping::RouteTreeOp>()) {
    auto producer =
        decodeDataflow<::dataflow::CanonicalGraphProducerEndpointRef>(
            record.getLogicalNet(), dataflow.identity());
    if (!producer)
      return producer.takeError();
    auto encoded =
        ::dataflow::encodeDataflowReference(dataflow.identity(), *producer);
    if (!encoded)
      return encoded.takeError();
    auto found = residual.find(byteKey(*encoded));
    if (found == residual.end())
      return invalid("RouteTree does not name a residual logical net");
    auto route = importRouteTree(record, terminalContext, *found->second);
    if (!route)
      return route.takeError();
    routes.push_back(std::move(*route));
    residual.erase(found);
  }
  if (!residual.empty())
    return invalid("SpatialMapping omits a residual logical net RouteTree");

  auto requiredUses = deriveRequiredComputeUses(
      dataflow, techMapping, fabric, computeBindings, registerFifoTransfers);
  if (!requiredUses)
    return requiredUses.takeError();
  auto requiredMemoryUses =
      deriveRequiredMemoryUses(*importedMemory, dataflow.identity());
  if (!requiredMemoryUses)
    return requiredMemoryUses.takeError();
  auto requiredTagUses = detail::deriveRequiredPhysicalTagUses(
      dataflow, techMapping, fabric, routes);
  if (!requiredTagUses)
    return requiredTagUses.takeError();
  std::map<::loom::fabric::FabricOrdinal, std::set<std::string>>
      assignedTagDomainValues;
  std::vector<SpatialResourceUseView> uses;
  std::vector<SpatialPhysicalTagSegmentView> physicalTagSegments;
  for (auto record :
       root.getBody().front().getOps<::mapping::ResourceUseOp>()) {
    std::optional<SpatialPhysicalTagSegmentView> physicalTagSegment;
    auto use = importResourceUse(record, dataflow, fabric, *requiredUses,
                                 *requiredMemoryUses, *requiredTagUses,
                                 assignedTagDomainValues, uses.size(),
                                 physicalTagSegment);
    if (!use)
      return use.takeError();
    if (physicalTagSegment)
      physicalTagSegments.push_back(std::move(*physicalTagSegment));
    uses.push_back(std::move(*use));
  }
  if (!requiredUses->empty())
    return invalid("SpatialMapping omits a required compute ResourceUse");
  if (!requiredMemoryUses->empty())
    return invalid("SpatialMapping omits a required memory ResourceUse");
  if (!requiredTagUses->empty())
    return invalid("SpatialMapping omits a required Physical Tag ResourceUse");
  llvm::sort(physicalTagSegments, [](const SpatialPhysicalTagSegmentView &lhs,
                                     const SpatialPhysicalTagSegmentView &rhs) {
    return std::tie(lhs.routeTreeOrdinal, lhs.segmentOrdinal) <
           std::tie(rhs.routeTreeOrdinal, rhs.segmentOrdinal);
  });
  std::vector<std::uint64_t> nextSegment(routes.size(), 0);
  std::vector<std::vector<bool>> taggedNodes;
  taggedNodes.reserve(routes.size());
  for (const SpatialRouteTreeView &route : routes)
    taggedNodes.emplace_back(route.nodes.size(), false);
  for (const SpatialPhysicalTagSegmentView &segment : physicalTagSegments) {
    if (segment.routeTreeOrdinal >= routes.size() ||
        segment.resourceUseOrdinal >= uses.size() ||
        segment.segmentOrdinal != nextSegment[segment.routeTreeOrdinal]++ ||
        segment.nodeOrdinals.empty())
      return invalid("Physical Tag segment projection is not canonical");
    for (std::uint64_t node : segment.nodeOrdinals) {
      if (node >= taggedNodes[segment.routeTreeOrdinal].size() ||
          taggedNodes[segment.routeTreeOrdinal][node])
        return invalid("Physical Tag segment repeats a RouteTree node");
      taggedNodes[segment.routeTreeOrdinal][node] = true;
    }
  }

  auto operandQueueMatchGroups = deriveSpatialPeOperandQueueMatchGroups(
      techMapping, fabric, computeBindings, routes, uses, physicalTagSegments);
  if (!operandQueueMatchGroups)
    return operandQueueMatchGroups.takeError();
  auto handshake =
      deriveSelectedHandshakeSelection(terminalContext, registerFifoTransfers,
                                       routes, uses, physicalTagSegments);
  if (!handshake)
    return handshake.takeError();
  if (handshakeContext) {
    if (llvm::Error error =
            ::loom::fabric::verifySelectedCombinationalHandshakeAcyclic(
                fabric, handshake->selection, *handshakeContext))
      return std::move(error);
  } else if (llvm::Error error =
                 ::loom::fabric::verifySelectedCombinationalHandshakeAcyclic(
                     fabric, handshake->selection)) {
    return std::move(error);
  }

  auto capacityOveruse = detail::deriveSpatialCapacityOveruse(
      fabric, dataflow.identity(), uses, handshake->routeTraversals);
  if (!capacityOveruse)
    return capacityOveruse.takeError();
  if (capacityOveruse->total != 0) {
    if (!capacityOveruse->firstWitness)
      return invalid("CapacityOveruse has no canonical witness");
    const auto &witness = *capacityOveruse->firstWitness;
    return invalid(llvm::Twine("CapacityOveruse at ") +
                   ::loom::fabric::printFabricRef(witness.owner) + " state " +
                   llvm::Twine(witness.state.ordinal()) + " dimension " +
                   llvm::Twine(witness.dimension.ordinal()) + " uses " +
                   llvm::Twine(witness.usage) + " of " +
                   llvm::Twine(witness.capacity));
  }

  auto configuredHardware = detail::deriveConfiguredHardwareProjection(
      dataflow, techMapping, fabric, computeBindings,
      importedMemory->engineBindings, importedMemory->memoryBindings,
      registerFifoTransfers, routes, uses, physicalTagSegments,
      *operandQueueMatchGroups);
  if (!configuredHardware)
    return configuredHardware.takeError();

  auto progress = deriveSpatialMappingProgressClosure(
      dataflow, techMapping, fabric, computeBindings, routes);
  if (!progress)
    return progress.takeError();
  switch (progress->kind) {
  case MappingProgressClosureKind::ProvenNoClosedWaitSet:
    break;
  case MappingProgressClosureKind::ProvenClosedWaitSet:
    return invalid("HardProgressViolation");
  case MappingProgressClosureKind::ProofNotEstablished:
    return incomplete("proof_not_established");
  }

  return ImportedSpatialView{*techIdentity,
                             *dataflowIdentity,
                             *fabricIdentity,
                             std::move(computeBindings),
                             std::move(importedMemory->engineBindings),
                             std::move(importedMemory->memoryBindings),
                             std::move(registerFifoTransfers),
                             std::move(routes),
                             std::move(uses),
                             std::move(physicalTagSegments),
                             std::move(*configuredHardware),
                             std::move(handshake->selection)};
}

struct PreparedSpatialMapping final {
  ArtifactRootReference reference;
  CanonicalSemanticBytes canonicalBytes;
  OwningOpRef<Operation *> canonicalRoot;
};

llvm::Expected<PreparedSpatialMapping>
prepareSpatialMapping(::mapping::SpatialOp source) {
  auto assembly = detail::prepareCanonicalSpatialMappingAssembly(source);
  if (!assembly)
    return assembly.takeError();
  ArtifactRootReference reference{
      mappingArtifactSchema.identity.str(), mappingArtifactSchema.version,
      finalizeArtifactIdentity(mappingArtifactSchema, assembly->bytes)};
  return PreparedSpatialMapping{std::move(reference),
                                std::move(assembly->bytes),
                                std::move(assembly->root)};
}

llvm::Error
publishPreparedSpatialMapping(const PreparedSpatialMapping &prepared,
                              const ArtifactStore &store) {
  auto stored = store.put(mappingArtifactSchema, prepared.canonicalBytes);
  if (!stored)
    return stored.takeError();
  if (*stored != prepared.reference.artifact)
    return invalid("ArtifactStore returned a different Mapping identity");
  return llvm::Error::success();
}

llvm::Error requirePublishedUpstream(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const TechMappingView &techMapping,
    const ::loom::fabric::FabricArtifactView &fabric,
    const ArtifactStore &store) {
  const ArtifactRootReference references[] = {
      {::dataflow::canonicalDataflowSchema.identity.str(),
       ::dataflow::canonicalDataflowSchema.version, dataflow.identity()},
      {mappingArtifactSchema.identity.str(), mappingArtifactSchema.version,
       techMapping.identity()},
      {::loom::fabric::fabricArtifactSchema.identity.str(),
       ::loom::fabric::fabricArtifactSchema.version, fabric.identity()}};
  for (const auto &reference : references) {
    auto bytes = store.get(reference);
    if (!bytes)
      return bytes.takeError();
  }
  return llvm::Error::success();
}

llvm::Expected<SpatialMappingView>
strictImport(const ArtifactIdentity &mappingIdentity,
             const CanonicalSemanticBytes &canonicalBytes,
             const ArtifactStore &store) {
  if (finalizeArtifactIdentity(mappingArtifactSchema, canonicalBytes) !=
      mappingIdentity)
    return invalid("mapping identity does not match canonical bytes");
  auto parsed = parseSpatialRoot(canonicalBytes);
  if (!parsed)
    return parsed.takeError();

  auto dataflowIdentity = decodeIdentity(parsed->root.getDataflow());
  auto fabricIdentity = decodeIdentity(parsed->root.getFabric());
  auto techIdentity = decodeIdentity(parsed->root.getTechMapping());
  if (!dataflowIdentity)
    return dataflowIdentity.takeError();
  if (!fabricIdentity)
    return fabricIdentity.takeError();
  if (!techIdentity)
    return techIdentity.takeError();
  auto dataflow = ::dataflow::importCanonicalDataflow(
      {::dataflow::canonicalDataflowSchema.identity.str(),
       ::dataflow::canonicalDataflowSchema.version, *dataflowIdentity},
      store);
  if (!dataflow)
    return dataflow.takeError();
  auto dataflowView = dataflow->view();
  if (!dataflowView)
    return dataflowView.takeError();
  auto fabric = ::loom::fabric::importEntireFabricRoot(
      {::loom::fabric::fabricArtifactSchema.identity.str(),
       ::loom::fabric::fabricArtifactSchema.version, *fabricIdentity},
      store);
  if (!fabric)
    return fabric.takeError();
  auto tech = importTechMapping({mappingArtifactSchema.identity.str(),
                                 mappingArtifactSchema.version, *techIdentity},
                                store);
  if (!tech)
    return tech.takeError();
  auto view =
      SpatialMappingView::import(mappingIdentity, parsed->root, *dataflowView,
                                 tech->view(), fabric->view());
  if (!view)
    return view.takeError();
  auto rewritten = writeCanonicalSpatialMappingAssembly(parsed->root);
  if (!rewritten)
    return rewritten.takeError();
  if (!rewritten->bytes().equals(canonicalBytes.bytes()))
    return invalid("stored SpatialMapping payload is not canonical");
  return view;
}

} // namespace

llvm::Expected<std::vector<std::uint8_t>>
encodeSpatialActivityEventKey(const ArtifactIdentity &dataflowIdentity,
                              const SpatialActivityEventRef &event) {
  auto encoded = std::visit(
      [&](const auto &typed)
          -> llvm::Expected<std::tuple<std::uint32_t, std::vector<std::uint8_t>,
                                       std::optional<std::uint32_t>>> {
        using Event = std::decay_t<decltype(typed)>;
        if constexpr (std::is_same_v<Event, SpatialActorTransitionEventRef>) {
          auto actor = ::dataflow::encodeDataflowReference(dataflowIdentity,
                                                           typed.actor);
          if (!actor)
            return actor.takeError();
          return std::make_tuple(0U, std::move(*actor), typed.transition);
        } else {
          auto bytes =
              ::dataflow::encodeDataflowReference(dataflowIdentity, typed);
          if (!bytes)
            return bytes.takeError();
          constexpr std::uint32_t tag =
              std::is_same_v<Event,
                             ::dataflow::CanonicalGraphProducerEndpointRef>
                  ? 1U
                  : 2U;
          return std::make_tuple(tag, std::move(*bytes), std::nullopt);
        }
      },
      event);
  if (!encoded)
    return encoded.takeError();
  return canonicalSpatialActivityEventKey(
      std::get<0>(*encoded), std::get<1>(*encoded), std::get<2>(*encoded));
}

llvm::Expected<std::vector<SpatialComputeUseRequirement>>
deriveSpatialComputeBindingUseRequirements(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const TechComputeRealizationView &realization,
    const ::loom::fabric::FabricArtifactView &fabric,
    const SpatialComputeBindingView &binding,
    llvm::ArrayRef<SpatialRegisterFifoTransferView> registerFifoTransfers) {
  std::vector<SpatialComputeUseRequirement> result;
  if (binding.realization != realization.entityId)
    return invalid("compute ResourceUse binding has the wrong realization");
  auto definition = fabric.fuTemplateOf(binding.occurrence);
  if (!definition || *definition != realization.capabilityTemplate.fu)
    return invalid("compute ResourceUse binding has the wrong FU definition");
  auto parentPe = fabric.parentPeOf(binding.occurrence);
  if (!parentPe || binding.context.pe != *parentPe ||
      binding.context.ordinal >= fabric.peResidentContextCount(*parentPe))
    return invalid("compute ResourceUse binding has an invalid context");
  const bool temporal =
      fabric.peSchedule(*parentPe) == ::fabric::Schedule::Temporal;
  const auto isRegisterFifoSink = [&](::dataflow::ActorRef actor,
                                      std::uint64_t ordinal) {
    const ::dataflow::CanonicalGraphConsumerEndpointRef sink(
        ::dataflow::ActorTokenOperandRef{actor, ordinal});
    return llvm::any_of(registerFifoTransfers, [&](const auto &transfer) {
      return transfer.sink == sink;
    });
  };

  if (temporal) {
    for (const auto &boundary : realization.boundaries) {
      if (boundary.direction != ::loom::fabric::FabricPortDirection::Input)
        continue;
      if (isRegisterFifoSink(boundary.actor, boundary.portOrdinal))
        continue;
      auto pattern = ::fabric::resolveTemporalPeOperandQueuePattern(
          fabric, binding.context, binding.occurrence,
          boundary.fabricPort.ordinal,
          ::fabric::TemporalOperandQueueUse::Enqueue);
      if (!pattern)
        return pattern.takeError();
      result.push_back(SpatialComputeUseRequirement{
          realization.entityId,
          ::dataflow::CanonicalGraphConsumerEndpointRef(
              ::dataflow::ActorTokenOperandRef{boundary.actor,
                                               boundary.portOrdinal}),
          *pattern,
          {}});
    }
  }

  for (const auto &actorBinding : realization.actors) {
    auto actor = dataflow.resolve(actorBinding.actor);
    if (!actor)
      return actor.takeError();
    auto projection =
        ::dataflow::projectRegisteredActorSchemaProjection(actor->op);
    if (!projection)
      return projection.takeError();
    auto cases = ::dataflow::semantics::projectActorHandshakeCases(
        projection->schema, actorBinding.operandPorts.size(),
        actorBinding.resultPorts.size());
    if (!cases)
      return cases.takeError();
    auto occurrenceOperation = ::loom::fabric::deriveFabricFuOccurrenceNode(
        fabric, actorBinding.fabricOperation, binding.occurrence);
    if (!occurrenceOperation)
      return occurrenceOperation.takeError();
    const auto *capability =
        fabric.resolvedFabricOpCapability(*occurrenceOperation);
    if (!capability)
      return invalid("selected compute actor has no physical capability");
    auto requiresHandoff = ::fabric::requiresActiveResultHandoff(
        capability->resourceStateAndTimingContract);
    if (!requiresHandoff)
      return requiresHandoff.takeError();
    for (const auto &transition : *cases) {
      auto pattern = ::fabric::resolveOperationUsePattern(
          capability->resourceStateAndTimingContract, transition.ordinal);
      if (!pattern)
        return pattern.takeError();
      const SpatialActorTransitionEventRef event{actorBinding.actor,
                                                 transition.ordinal};
      std::vector<SpatialActivityEventRef> release;
      std::vector<std::pair<std::vector<std::uint8_t>, SpatialActivityEventRef>>
          keyedRelease;
      if (*requiresHandoff) {
        keyedRelease.reserve(transition.activeResults.size());
        for (std::uint32_t resultOrdinal : transition.activeResults) {
          SpatialActivityEventRef produced =
              ::dataflow::CanonicalGraphProducerEndpointRef(
                  ::dataflow::ActorTokenResultRef{actorBinding.actor,
                                                  resultOrdinal});
          auto key =
              encodeSpatialActivityEventKey(dataflow.identity(), produced);
          if (!key)
            return key.takeError();
          keyedRelease.emplace_back(std::move(*key), std::move(produced));
        }
      }
      llvm::sort(keyedRelease, [](const auto &left, const auto &right) {
        return left.first < right.first;
      });
      for (auto &entry : keyedRelease) {
        if (!release.empty() &&
            entry.first == keyedRelease[release.size() - 1].first)
          return invalid("actor handshake case has duplicate active results");
        release.push_back(std::move(entry.second));
      }
      result.push_back(SpatialComputeUseRequirement{
          realization.entityId, event,
          ::loom::fabric::FabricUsePatternRef{
              ::loom::fabric::FabricUsePatternOwnerRef(
                  ::loom::fabric::FabricInventoryOwnerRef::of(
                      *occurrenceOperation)),
              pattern->ordinal()},
          std::move(release)});
      if (!temporal)
        continue;
      for (std::uint32_t operand : transition.consumedInputs) {
        if (isRegisterFifoSink(actorBinding.actor, operand))
          continue;
        const TechComputeBoundaryView *boundary = nullptr;
        for (const auto &candidate : realization.boundaries) {
          if (candidate.actor != actorBinding.actor ||
              candidate.direction !=
                  ::loom::fabric::FabricPortDirection::Input ||
              candidate.portOrdinal != operand)
            continue;
          if (boundary)
            return invalid(
                "actor input has duplicate compute boundary witnesses");
          boundary = &candidate;
        }
        if (!boundary)
          continue;
        auto queuePattern = ::fabric::resolveTemporalPeOperandQueuePattern(
            fabric, binding.context, binding.occurrence,
            boundary->fabricPort.ordinal,
            ::fabric::TemporalOperandQueueUse::Dequeue);
        if (!queuePattern)
          return queuePattern.takeError();
        result.push_back(SpatialComputeUseRequirement{
            realization.entityId, event, *queuePattern, {}});
      }
    }
  }
  return result;
}

llvm::Expected<std::vector<SpatialComputeUseRequirement>>
deriveSpatialComputeUseRequirements(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const TechMappingView &techMapping,
    const ::loom::fabric::FabricArtifactView &fabric,
    llvm::ArrayRef<SpatialComputeBindingView> bindings,
    llvm::ArrayRef<SpatialRegisterFifoTransferView> registerFifoTransfers) {
  if (bindings.size() != techMapping.computeRealizations().size())
    return invalid("compute ResourceUse projection has incomplete bindings");
  std::vector<SpatialComputeUseRequirement> result;
  for (const auto &realization : techMapping.computeRealizations()) {
    const SpatialComputeBindingView *binding = nullptr;
    for (const auto &candidate : bindings) {
      if (candidate.realization != realization.entityId)
        continue;
      if (binding)
        return invalid("compute ResourceUse projection has duplicate bindings");
      binding = &candidate;
    }
    if (!binding)
      return invalid("Compute realization has no Spatial binding");
    auto requirements = deriveSpatialComputeBindingUseRequirements(
        dataflow, realization, fabric, *binding, registerFifoTransfers);
    if (!requirements)
      return requirements.takeError();
    result.insert(result.end(), std::make_move_iterator(requirements->begin()),
                  std::make_move_iterator(requirements->end()));
  }
  return result;
}

llvm::Expected<SpatialMappingView> SpatialMappingView::import(
    const ArtifactIdentity &mappingIdentity, ::mapping::SpatialOp root,
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const TechMappingView &techMapping,
    const ::loom::fabric::FabricArtifactView &fabric,
    const ::loom::fabric::FabricHandshakeContext *handshakeContext) {
  auto imported = importView(mappingIdentity, root, dataflow, techMapping,
                             fabric, handshakeContext);
  if (!imported)
    return imported.takeError();
  return SpatialMappingView(
      mappingIdentity, std::move(imported->techMappingIdentity),
      std::move(imported->dataflowIdentity),
      std::move(imported->fabricIdentity), std::move(imported->computeBindings),
      std::move(imported->memoryEngineBindings),
      std::move(imported->memoryBindings),
      std::move(imported->registerFifoTransfers),
      std::move(imported->routeTrees), std::move(imported->resourceUses),
      std::move(imported->physicalTagSegments),
      std::move(imported->configuredHardware),
      std::move(imported->handshakeSelection));
}

llvm::Error verifySpatialMappingBase(
    ::mapping::SpatialOp source,
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const TechMappingView &techMapping,
    const ::loom::fabric::FabricArtifactView &fabric) {
  auto prepared = prepareSpatialMapping(source);
  if (!prepared)
    return prepared.takeError();
  auto view = SpatialMappingView::import(
      prepared->reference.artifact,
      cast<::mapping::SpatialOp>(prepared->canonicalRoot.get()), dataflow,
      techMapping, fabric);
  if (!view)
    return view.takeError();
  return llvm::Error::success();
}

llvm::Expected<FinalizedSpatialMapping> finalizeSpatialMapping(
    ::mapping::SpatialOp source,
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const TechMappingView &techMapping,
    const ::loom::fabric::FabricArtifactView &fabric,
    const SpatialMappingConstraintSetView &constraints,
    const ArtifactStore &store,
    const ::loom::fabric::FabricHandshakeContext *handshakeContext) {
  if (llvm::Error error =
          requirePublishedUpstream(dataflow, techMapping, fabric, store))
    return std::move(error);
  auto constraintBytes =
      store.get({mappingConstraintSetSchema.identity.str(),
                 mappingConstraintSetSchema.version, constraints.identity()});
  if (!constraintBytes)
    return constraintBytes.takeError();
  auto prepared = prepareSpatialMapping(source);
  if (!prepared)
    return prepared.takeError();
  auto view = SpatialMappingView::import(
      prepared->reference.artifact,
      cast<::mapping::SpatialOp>(prepared->canonicalRoot.get()), dataflow,
      techMapping, fabric, handshakeContext);
  if (!view)
    return view.takeError();
  if (llvm::Error error = admitSpatialMappingConstraints(
          dataflow, techMapping, fabric, constraints, *view))
    return std::move(error);
  if (llvm::Error error = publishPreparedSpatialMapping(*prepared, store))
    return std::move(error);
  return FinalizedSpatialMapping(std::move(prepared->reference),
                                 std::move(prepared->canonicalBytes),
                                 std::move(*view));
}

llvm::Expected<FinalizedSpatialMapping>
importSpatialMapping(const ArtifactRootReference &reference,
                     const ArtifactStore &store) {
  if (reference.schemaIdentity != mappingArtifactSchema.identity ||
      reference.schemaVersion != mappingArtifactSchema.version)
    return invalid("root reference has the wrong Mapping schema");
  auto canonicalBytes = store.get(reference);
  if (!canonicalBytes)
    return canonicalBytes.takeError();
  auto view = strictImport(reference.artifact, *canonicalBytes, store);
  if (!view)
    return view.takeError();
  return FinalizedSpatialMapping(reference, std::move(*canonicalBytes),
                                 std::move(*view));
}

} // namespace loom::mapping
