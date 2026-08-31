#include "Mapping/Artifact/MappingConstraintSet.h"

#include "MappingConstraintCanonicalization.h"

#include "Common/ArtifactFinalizer.h"
#include "Common/ArtifactLocalReference.h"
#include "Dataflow/IR/DataflowOps.h"
#include "Dataflow/IR/DataflowReferenceCodec.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Mapping/IR/MappingAttrs.h"
#include "Mapping/IR/MappingDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

using namespace mlir;

namespace loom::mapping {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "mapping_constraint_set_invalid: " + message);
}

std::vector<std::uint8_t> unsignedBytes(DenseI8ArrayAttr record) {
  std::vector<std::uint8_t> result;
  result.reserve(record.size());
  for (std::int8_t byte : record.asArrayRef())
    result.push_back(static_cast<std::uint8_t>(byte));
  return result;
}

llvm::Expected<ArtifactIdentity>
decodeIdentity(::mapping::ArtifactIdentityAttr attribute) {
  return ArtifactIdentity::fromBytes(unsignedBytes(attribute.getRecord()));
}

llvm::Expected<ArtifactRootReference>
decodeRootReference(::mapping::ArtifactRootReferenceAttr attribute) {
  std::vector<std::uint8_t> bytes = unsignedBytes(attribute.getRecord());
  auto decoded = decodeArtifactRootReferencePrefix(bytes);
  if (!decoded)
    return decoded.takeError();
  if (decoded->byteCount != bytes.size() ||
      encodeArtifactRootReference(decoded->reference) != bytes)
    return invalid("no-good Mapping reference has noncanonical framing");
  return std::move(decoded->reference);
}

llvm::Expected<SpatialRuntimeCounterexampleNoGoodView::Lineage>
decodeRuntimeCounterexampleLineage(
    ::mapping::RuntimeCounterexampleLineageAttr attribute) {
  auto parent = decodeRootReference(attribute.getParentMapping());
  if (!parent)
    return parent.takeError();
  auto evidence = decodeRootReference(attribute.getRuntimeEvidence());
  if (!evidence)
    return evidence.takeError();
  auto request = decodeRootReference(attribute.getEvaluationRequest());
  if (!request)
    return request.takeError();
  auto execution = decodeRootReference(attribute.getRuntimeExecution());
  if (!execution)
    return execution.takeError();
  auto digest = ComponentViewDigest::fromBytes(
      unsignedBytes(attribute.getCertificateDigest()));
  if (!digest)
    return digest.takeError();
  return SpatialRuntimeCounterexampleNoGoodView::Lineage{
      std::move(*parent), std::move(*evidence), std::move(*request),
      std::move(*execution), std::move(*digest)};
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

template <typename T>
llvm::Expected<T> contextual(llvm::Expected<T> value,
                             const llvm::Twine &context) {
  if (!value)
    return llvm::joinErrors(invalid(context), value.takeError());
  return std::move(*value);
}

llvm::Error contextual(llvm::Error error, const llvm::Twine &context) {
  if (!error)
    return llvm::Error::success();
  return llvm::joinErrors(invalid(context), std::move(error));
}

bool isIntervalProjection(::mapping::SpatialConstraintProjection projection) {
  return projection ==
         ::mapping::SpatialConstraintProjection::NetAssignedTagValues;
}

std::vector<Attribute> normalizeAddressRegions(MLIRContext *context,
                                               ArrayRef<Attribute> values) {
  struct ServiceRanges final {
    ::mapping::FabricMemoryServiceRefAttr service;
    std::vector<Attribute> intervals;
  };
  std::map<std::string, ServiceRanges> byService;
  for (Attribute value : values) {
    auto region = cast<::mapping::ConstraintAddressRegionAttr>(value);
    const std::string key = detail::constraintAttributeKey(region.getService());
    auto [position, inserted] =
        byService.try_emplace(key, ServiceRanges{region.getService(), {}});
    (void)inserted;
    position->second.intervals.insert(position->second.intervals.end(),
                                      region.getIntervals().begin(),
                                      region.getIntervals().end());
  }

  std::vector<Attribute> result;
  for (auto &[key, service] : byService) {
    (void)key;
    std::vector<Attribute> intervals =
        detail::normalizeUnsignedIntervalConstraintDomain(context,
                                                          service.intervals);
    if (intervals.empty())
      continue;
    result.push_back(::mapping::ConstraintAddressRegionAttr::get(
        context, service.service, ArrayAttr::get(context, intervals)));
  }
  return result;
}

std::vector<Attribute>
normalizeDomain(MLIRContext *context,
                ::mapping::SpatialConstraintProjection projection,
                ArrayRef<Attribute> values) {
  if (isIntervalProjection(projection))
    return detail::normalizeUnsignedIntervalConstraintDomain(context, values);
  if (projection == ::mapping::SpatialConstraintProjection::MemoryAddressRegion)
    return normalizeAddressRegions(context, values);
  return detail::normalizeExactConstraintDomain(values);
}

std::vector<Attribute>
intersectDomains(MLIRContext *context,
                 ::mapping::SpatialConstraintProjection projection,
                 ArrayRef<Attribute> lhs, ArrayRef<Attribute> rhs) {
  if (isIntervalProjection(projection)) {
    return detail::intersectUnsignedIntervalConstraintDomains(context, lhs,
                                                              rhs);
  }
  if (projection ==
      ::mapping::SpatialConstraintProjection::MemoryAddressRegion) {
    std::map<std::string, ::mapping::ConstraintAddressRegionAttr>
        rightByService;
    for (Attribute value : normalizeAddressRegions(context, rhs)) {
      auto region = cast<::mapping::ConstraintAddressRegionAttr>(value);
      rightByService.emplace(
          detail::constraintAttributeKey(region.getService()), region);
    }
    std::vector<Attribute> result;
    for (Attribute value : normalizeAddressRegions(context, lhs)) {
      auto leftRegion = cast<::mapping::ConstraintAddressRegionAttr>(value);
      auto rightRegion = rightByService.find(
          detail::constraintAttributeKey(leftRegion.getService()));
      if (rightRegion == rightByService.end())
        continue;
      std::vector<Attribute> intervals =
          detail::intersectUnsignedIntervalConstraintDomains(
              context, leftRegion.getIntervals().getValue(),
              rightRegion->second.getIntervals().getValue());
      if (!intervals.empty())
        result.push_back(::mapping::ConstraintAddressRegionAttr::get(
            context, leftRegion.getService(),
            ArrayAttr::get(context, intervals)));
    }
    return result;
  }
  return detail::intersectExactConstraintDomains(lhs, rhs);
}

::mapping::SpatialConstraintProjectionKeyAttr
spatialProjection(Operation *operation) {
  return cast<::mapping::SpatialConstraintProjectionKeyAttr>(
      operation->getAttr("projection"));
}

struct ParsedSpatialConstraintRoot final {
  std::unique_ptr<MLIRContext> context;
  OwningOpRef<ModuleOp> module;
  ::mapping::ConstraintsSpatialOp root;
};

llvm::Expected<ParsedSpatialConstraintRoot>
parseSpatialConstraintRoot(const CanonicalSemanticBytes &canonicalBytes) {
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
    return invalid("canonical Spatial constraint payload cannot be parsed");

  ::mapping::ConstraintsSpatialOp root;
  unsigned rootCount = 0;
  for (Operation &operation : module->getBody()->without_terminator()) {
    auto candidate = dyn_cast<::mapping::ConstraintsSpatialOp>(operation);
    if (!candidate)
      return invalid("constraint artifact contains a non-Spatial root");
    root = candidate;
    ++rootCount;
  }
  if (rootCount != 1)
    return invalid("constraint artifact must contain exactly one Spatial root");
  if (failed(verify(root)))
    return invalid("Spatial constraint root is structurally invalid");
  return ParsedSpatialConstraintRoot{std::move(context), std::move(module),
                                     root};
}

bool hasComputeRealization(const TechMappingView &techMapping,
                           std::uint64_t entity) {
  return llvm::any_of(techMapping.computeRealizations(),
                      [&](const TechComputeRealizationView &realization) {
                        return realization.entityId == entity;
                      });
}

bool hasMemoryRealization(const TechMappingView &techMapping,
                          std::uint64_t entity) {
  return llvm::any_of(techMapping.memoryRealizations(),
                      [&](const TechMemoryRealizationView &realization) {
                        return realization.entityId == entity;
                      });
}

llvm::Error validateResidualProducer(
    const ::dataflow::CanonicalGraphProducerEndpointRef &producer,
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const TechMappingView &techMapping) {
  if (llvm::Error error = dataflow.validate(producer))
    return contextual(std::move(error),
                      "constraint producer endpoint does not resolve");
  if (!techMapping.residualLogicalNet(producer))
    return invalid("constraint producer has no residual logical net");
  return llvm::Error::success();
}

llvm::Error validateResidualSink(
    const ::dataflow::CanonicalGraphProducerEndpointRef &producer,
    const ::dataflow::CanonicalGraphConsumerEndpointRef &consumer,
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const TechMappingView &techMapping) {
  if (llvm::Error error = dataflow.validate(consumer))
    return contextual(std::move(error),
                      "constraint consumer endpoint does not resolve");
  auto consumers = dataflow.graphConsumers(producer);
  if (!consumers)
    return contextual(consumers.takeError(),
                      "constraint producer relation does not resolve");
  if (llvm::find(*consumers, consumer) == consumers->end())
    return invalid("constraint transfer sink is not fed by its producer");
  const TechResidualLogicalNetView *net =
      techMapping.residualLogicalNet(producer);
  if (!net || llvm::find(net->sinks, consumer) == net->sinks.end())
    return invalid("constraint transfer sink is realization-internal");
  return llvm::Error::success();
}

bool isMappedAddressedMemoryActor(const TechMappingView &techMapping,
                                  ::dataflow::ActorRef actor) {
  for (const TechMemoryRealizationView &realization :
       techMapping.memoryRealizations())
    for (const TechMemoryActorView &mapped : realization.actors)
      if (mapped.actor == actor)
        return true;
  return false;
}

llvm::Expected<SpatialConstraintSubject>
decodeSubject(::mapping::SpatialConstraintProjection projection,
              Attribute attribute,
              const ::dataflow::CanonicalDataflowProgramView &dataflow,
              const TechMappingView &techMapping) {
  using Projection = ::mapping::SpatialConstraintProjection;
  switch (projection) {
  case Projection::ComputePlacement:
  case Projection::ComputeParentPe:
  case Projection::ComputeInstructionContext:
  case Projection::ComputeFuContext: {
    const std::uint64_t entity =
        cast<::mapping::ComputeRealizationRefAttr>(attribute).getEntity();
    if (!hasComputeRealization(techMapping, entity))
      return invalid(
          "constraint names a stale or wrong-kind ComputeRealization");
    return SpatialConstraintSubject(TechComputeRealizationRef{entity});
  }
  case Projection::MemoryPlacement: {
    const std::uint64_t entity =
        cast<::mapping::MemoryRealizationRefAttr>(attribute).getEntity();
    if (!hasMemoryRealization(techMapping, entity))
      return invalid(
          "constraint names a stale or wrong-kind MemoryRealization");
    return SpatialConstraintSubject(TechMemoryRealizationRef{entity});
  }
  case Projection::NetAssignedTagValues:
  case Projection::NetSelectedPhysicalTraversals:
  case Projection::NetTraversalResourceStates: {
    auto producer =
        decodeDataflow<::dataflow::CanonicalGraphProducerEndpointRef>(
            cast<::mapping::GraphProducerEndpointRefAttr>(attribute),
            dataflow.identity());
    if (!producer)
      return contextual(producer.takeError(),
                        "constraint producer reference is malformed");
    if (llvm::Error error =
            validateResidualProducer(*producer, dataflow, techMapping))
      return std::move(error);
    return SpatialConstraintSubject(std::move(*producer));
  }
  case Projection::SpatialTransferAttachment: {
    auto terminal = cast<::mapping::SpatialTransferTerminalAttr>(attribute);
    auto producer =
        decodeDataflow<::dataflow::CanonicalGraphProducerEndpointRef>(
            terminal.getProducer(), dataflow.identity());
    if (!producer)
      return contextual(producer.takeError(),
                        "constraint transfer producer is malformed");
    if (llvm::Error error =
            validateResidualProducer(*producer, dataflow, techMapping))
      return std::move(error);
    std::optional<::dataflow::CanonicalGraphConsumerEndpointRef> consumer;
    if (terminal.getConsumer()) {
      auto decoded =
          decodeDataflow<::dataflow::CanonicalGraphConsumerEndpointRef>(
              terminal.getConsumer(), dataflow.identity());
      if (!decoded)
        return contextual(decoded.takeError(),
                          "constraint transfer consumer is malformed");
      if (llvm::Error error =
              validateResidualSink(*producer, *decoded, dataflow, techMapping))
        return std::move(error);
      consumer = std::move(*decoded);
    }
    return SpatialConstraintSubject(SpatialConstraintTransferTerminal{
        std::move(*producer), std::move(consumer)});
  }
  case Projection::MemoryOperationPort: {
    auto actor = decodeDataflow<::dataflow::ActorRef>(
        cast<::mapping::ActorRefAttr>(attribute), dataflow.identity());
    if (!actor)
      return contextual(actor.takeError(),
                        "constraint memory actor reference is malformed");
    auto resolved = dataflow.resolve(*actor);
    if (!resolved)
      return contextual(resolved.takeError(),
                        "constraint memory actor does not resolve");
    if (!isa<::dataflow::LoadOp, ::dataflow::StoreOp>(resolved->op) ||
        !isMappedAddressedMemoryActor(techMapping, *actor))
      return invalid(
          "memory_operation_port subject is not a realized load or store");
    return SpatialConstraintSubject(std::move(*actor));
  }
  case Projection::MemoryBoundServices:
  case Projection::MemoryAddressRegion: {
    auto root = decodeDataflow<::dataflow::LogicalMemoryRootRef>(
        cast<::mapping::LogicalMemoryRootRefAttr>(attribute),
        dataflow.identity());
    if (!root)
      return contextual(root.takeError(),
                        "constraint logical memory root is malformed");
    auto resolved = dataflow.resolve(*root);
    if (!resolved)
      return contextual(resolved.takeError(),
                        "constraint logical memory root does not resolve");
    return SpatialConstraintSubject(std::move(*root));
  }
  }
  llvm_unreachable("unknown Spatial constraint projection");
}

template <typename Ref, typename Attr>
llvm::Expected<Ref>
decodeValidatedFabric(Attr attribute,
                      const ::loom::fabric::FabricArtifactView &fabric,
                      const llvm::Twine &description) {
  auto reference = decodeFabric<Ref>(attribute);
  if (!reference)
    return contextual(reference.takeError(), description + " is malformed");
  if (llvm::Error error = ::loom::fabric::validateFabricRef(fabric, *reference))
    return contextual(std::move(error), description + " does not resolve");
  return std::move(*reference);
}

SpatialConstraintUnsignedInterval
decodeInterval(::mapping::ConstraintUnsignedIntervalAttr interval) {
  return SpatialConstraintUnsignedInterval{interval.getLower().getValue(),
                                           interval.getUpper().getValue()};
}

llvm::Expected<SpatialConstraintDomainValue>
decodeDomainValue(::mapping::SpatialConstraintProjection projection,
                  Attribute attribute,
                  const ::loom::fabric::FabricArtifactView &fabric) {
  using Projection = ::mapping::SpatialConstraintProjection;
  switch (projection) {
  case Projection::ComputePlacement: {
    auto value = decodeValidatedFabric<::loom::fabric::FabricFuOccurrenceRef>(
        cast<::mapping::FabricFuOccurrenceRefAttr>(attribute), fabric,
        "constraint FU occurrence");
    if (!value)
      return value.takeError();
    return SpatialConstraintDomainValue(std::move(*value));
  }
  case Projection::ComputeParentPe: {
    auto value = decodeValidatedFabric<::loom::fabric::FabricPeOccurrenceRef>(
        cast<::mapping::FabricPeOccurrenceRefAttr>(attribute), fabric,
        "constraint PE occurrence");
    if (!value)
      return value.takeError();
    return SpatialConstraintDomainValue(std::move(*value));
  }
  case Projection::ComputeInstructionContext: {
    auto value = decodeValidatedFabric<::loom::fabric::InstructionContextRef>(
        cast<::mapping::InstructionContextRefAttr>(attribute), fabric,
        "constraint instruction context");
    if (!value)
      return value.takeError();
    return SpatialConstraintDomainValue(std::move(*value));
  }
  case Projection::ComputeFuContext: {
    auto tuple = cast<::mapping::ConstraintFuContextAttr>(attribute);
    auto fu = decodeValidatedFabric<::loom::fabric::FabricFuOccurrenceRef>(
        tuple.getFu(), fabric, "constraint FU/context FU");
    if (!fu)
      return fu.takeError();
    auto context = decodeValidatedFabric<::loom::fabric::InstructionContextRef>(
        tuple.getInstructionContext(), fabric,
        "constraint FU/context instruction context");
    if (!context)
      return context.takeError();
    return SpatialConstraintDomainValue(
        SpatialConstraintFuContext{std::move(*fu), std::move(*context)});
  }
  case Projection::MemoryPlacement: {
    auto value =
        decodeValidatedFabric<::loom::fabric::FabricMemoryOccurrenceRef>(
            cast<::mapping::FabricMemoryOccurrenceRefAttr>(attribute), fabric,
            "constraint memory occurrence");
    if (!value)
      return value.takeError();
    return SpatialConstraintDomainValue(std::move(*value));
  }
  case Projection::NetAssignedTagValues:
    return SpatialConstraintDomainValue(decodeInterval(
        cast<::mapping::ConstraintUnsignedIntervalAttr>(attribute)));
  case Projection::NetSelectedPhysicalTraversals: {
    auto value =
        decodeValidatedFabric<::loom::fabric::FabricPhysicalTraversalRef>(
            cast<::mapping::FabricPhysicalTraversalRefAttr>(attribute), fabric,
            "constraint physical traversal");
    if (!value)
      return value.takeError();
    return SpatialConstraintDomainValue(std::move(*value));
  }
  case Projection::NetTraversalResourceStates: {
    auto value = decodeValidatedFabric<::loom::fabric::FabricResourceStateRef>(
        cast<::mapping::FabricResourceStateRefAttr>(attribute), fabric,
        "constraint resource state");
    if (!value)
      return value.takeError();
    return SpatialConstraintDomainValue(std::move(*value));
  }
  case Projection::SpatialTransferAttachment: {
    auto value =
        decodeValidatedFabric<::loom::fabric::FabricTransportEndpointRef>(
            cast<::mapping::FabricTransportEndpointRefAttr>(attribute), fabric,
            "constraint transport endpoint");
    if (!value)
      return value.takeError();
    return SpatialConstraintDomainValue(std::move(*value));
  }
  case Projection::MemoryOperationPort: {
    auto value =
        decodeValidatedFabric<::loom::fabric::FabricMemoryOperationPortRef>(
            cast<::mapping::FabricMemoryOperationPortRefAttr>(attribute),
            fabric, "constraint memory operation port");
    if (!value)
      return value.takeError();
    return SpatialConstraintDomainValue(std::move(*value));
  }
  case Projection::MemoryBoundServices: {
    auto value = decodeValidatedFabric<::loom::fabric::FabricMemoryServiceRef>(
        cast<::mapping::FabricMemoryServiceRefAttr>(attribute), fabric,
        "constraint memory service");
    if (!value)
      return value.takeError();
    return SpatialConstraintDomainValue(std::move(*value));
  }
  case Projection::MemoryAddressRegion: {
    auto region = cast<::mapping::ConstraintAddressRegionAttr>(attribute);
    auto service =
        decodeValidatedFabric<::loom::fabric::FabricMemoryServiceRef>(
            region.getService(), fabric,
            "constraint address-region memory service");
    if (!service)
      return service.takeError();
    std::vector<SpatialConstraintUnsignedInterval> intervals;
    intervals.reserve(region.getIntervals().size());
    for (Attribute interval : region.getIntervals())
      intervals.push_back(decodeInterval(
          cast<::mapping::ConstraintUnsignedIntervalAttr>(interval)));
    return SpatialConstraintDomainValue(SpatialConstraintAddressRegion{
        std::move(*service), std::move(intervals)});
  }
  }
  llvm_unreachable("unknown Spatial constraint projection");
}

llvm::Expected<std::vector<SpatialConstraintSubject>>
decodeSubjects(::mapping::SpatialConstraintProjection projection,
               ArrayAttr attributes,
               const ::dataflow::CanonicalDataflowProgramView &dataflow,
               const TechMappingView &techMapping) {
  std::vector<SpatialConstraintSubject> result;
  result.reserve(attributes.size());
  for (Attribute attribute : attributes) {
    auto subject = decodeSubject(projection, attribute, dataflow, techMapping);
    if (!subject)
      return subject.takeError();
    result.push_back(std::move(*subject));
  }
  return result;
}

llvm::Expected<std::vector<SpatialConstraintDomainValue>>
decodeDomain(::mapping::SpatialConstraintProjection projection,
             ArrayAttr attributes,
             const ::loom::fabric::FabricArtifactView &fabric) {
  std::vector<SpatialConstraintDomainValue> result;
  result.reserve(attributes.size());
  for (Attribute attribute : attributes) {
    auto value = decodeDomainValue(projection, attribute, fabric);
    if (!value)
      return value.takeError();
    result.push_back(std::move(*value));
  }
  return result;
}

/// Decodes one closed no-good literal. Every reference is validated against the
/// same exact D/T/F owners the conjunctive clause kinds use, so a literal can
/// never name a net, sink, traversal, or endpoint outside this closure.
llvm::Expected<SpatialNoGoodLiteral>
decodeNoGoodLiteral(Attribute attribute,
                    const ::dataflow::CanonicalDataflowProgramView &dataflow,
                    const TechMappingView &techMapping,
                    const ::loom::fabric::FabricArtifactView &fabric,
                    const ArtifactStore &store,
                    llvm::ArrayRef<SpatialMappingIdentityEqualsLiteral>
                        importedMappingCache) {
  if (auto literal = dyn_cast<::mapping::NetUsesTraversalAttr>(attribute)) {
    auto producer =
        decodeDataflow<::dataflow::CanonicalGraphProducerEndpointRef>(
            literal.getProducer(), dataflow.identity());
    if (!producer)
      return contextual(producer.takeError(),
                        "no-good net producer reference is malformed");
    if (llvm::Error error =
            validateResidualProducer(*producer, dataflow, techMapping))
      return std::move(error);
    std::optional<::dataflow::CanonicalGraphConsumerEndpointRef> consumer;
    if (literal.getConsumer()) {
      auto decoded =
          decodeDataflow<::dataflow::CanonicalGraphConsumerEndpointRef>(
              literal.getConsumer(), dataflow.identity());
      if (!decoded)
        return contextual(decoded.takeError(),
                          "no-good net sink reference is malformed");
      if (llvm::Error error =
              validateResidualSink(*producer, *decoded, dataflow, techMapping))
        return std::move(error);
      consumer = std::move(*decoded);
    }
    auto traversal = decodeFabric<::loom::fabric::FabricPhysicalTraversalRef>(
        literal.getTraversal());
    if (!traversal)
      return contextual(traversal.takeError(),
                        "no-good traversal reference is malformed");
    if (llvm::Error error =
            ::loom::fabric::validateFabricRef(fabric, *traversal))
      return contextual(std::move(error),
                        "no-good traversal does not resolve");
    return SpatialNoGoodLiteral(SpatialNetUsesTraversalLiteral{
        std::move(*producer), std::move(consumer), std::move(*traversal)});
  }

  if (auto literal = dyn_cast<::mapping::NetTagEqualsAttr>(attribute)) {
    auto producer =
        decodeDataflow<::dataflow::CanonicalGraphProducerEndpointRef>(
            literal.getProducer(), dataflow.identity());
    if (!producer)
      return contextual(producer.takeError(),
                        "no-good tag producer reference is malformed");
    if (llvm::Error error =
            validateResidualProducer(*producer, dataflow, techMapping))
      return std::move(error);
    const llvm::APInt value = ::fabric::canonicalPhysicalTagValue(
        literal.getValue().getValue());
    if (value.getBitWidth() == 0)
      return invalid("no-good Physical Tag value has zero width");
    return SpatialNoGoodLiteral(SpatialNetTagEqualsLiteral{
        std::move(*producer), literal.getSegmentOrdinal(), value});
  }

  if (auto literal =
          dyn_cast<::mapping::SpatialMappingIdentityEqualsAttr>(attribute)) {
    auto reference = decodeRootReference(literal.getSpatialMapping());
    if (!reference)
      return reference.takeError();
    if (reference->schemaIdentity != mappingArtifactSchema.identity ||
        reference->schemaVersion != mappingArtifactSchema.version)
      return invalid("no-good parent has the wrong Mapping schema");
    for (const auto &cached : importedMappingCache) {
      if (cached.mapping != *reference)
        continue;
      if (!cached.importedMapping ||
          cached.importedMapping->reference() != *reference)
        return invalid("no-good parent Mapping cache has foreign identity");
      const SpatialMappingView &view = cached.importedMapping->view();
      if (view.dataflowIdentity() != dataflow.identity() ||
          view.techMappingIdentity() != techMapping.identity() ||
          view.fabricIdentity() != fabric.identity())
        return invalid("no-good parent Mapping cache has foreign D/T/F owners");
      return SpatialNoGoodLiteral(SpatialMappingIdentityEqualsLiteral{
          std::move(*reference), cached.importedMapping});
    }
    auto parent = importSpatialMapping(*reference, store);
    if (!parent)
      return contextual(parent.takeError(),
                        "no-good parent SpatialMapping cannot be imported");
    if (parent->view().dataflowIdentity() != dataflow.identity() ||
        parent->view().techMappingIdentity() != techMapping.identity() ||
        parent->view().fabricIdentity() != fabric.identity())
      return invalid("no-good parent SpatialMapping has foreign D/T/F owners");
    return SpatialNoGoodLiteral(SpatialMappingIdentityEqualsLiteral{
        std::move(*reference),
        std::make_shared<const FinalizedSpatialMapping>(std::move(*parent))});
  }

  auto literal = cast<::mapping::TransferAttachmentEqualsAttr>(attribute);
  auto terminalAttr = literal.getTerminal();
  auto producer = decodeDataflow<::dataflow::CanonicalGraphProducerEndpointRef>(
      terminalAttr.getProducer(), dataflow.identity());
  if (!producer)
    return contextual(producer.takeError(),
                      "no-good transfer producer is malformed");
  if (llvm::Error error =
          validateResidualProducer(*producer, dataflow, techMapping))
    return std::move(error);
  std::optional<::dataflow::CanonicalGraphConsumerEndpointRef> consumer;
  if (terminalAttr.getConsumer()) {
    auto decoded =
        decodeDataflow<::dataflow::CanonicalGraphConsumerEndpointRef>(
            terminalAttr.getConsumer(), dataflow.identity());
    if (!decoded)
      return contextual(decoded.takeError(),
                        "no-good transfer consumer is malformed");
    if (llvm::Error error =
            validateResidualSink(*producer, *decoded, dataflow, techMapping))
      return std::move(error);
    consumer = std::move(*decoded);
  }
  auto endpoint = decodeFabric<::loom::fabric::FabricTransportEndpointRef>(
      literal.getEndpoint());
  if (!endpoint)
    return contextual(endpoint.takeError(),
                      "no-good transport endpoint is malformed");
  if (llvm::Error error = ::loom::fabric::validateFabricRef(fabric, *endpoint))
    return contextual(std::move(error),
                      "no-good transport endpoint does not resolve");
  return SpatialNoGoodLiteral(SpatialTransferAttachmentEqualsLiteral{
      SpatialConstraintTransferTerminal{std::move(*producer),
                                        std::move(consumer)},
      std::move(*endpoint)});
}

mlir::DenseI8ArrayAttr denseBytes(MLIRContext *context,
                                  llvm::ArrayRef<std::uint8_t> bytes) {
  std::vector<std::int8_t> signedBytes;
  signedBytes.reserve(bytes.size());
  for (std::uint8_t byte : bytes)
    signedBytes.push_back(static_cast<std::int8_t>(byte));
  return mlir::DenseI8ArrayAttr::get(context, signedBytes);
}

/// Encodes one closed no-good literal back to its canonical attribute. The
/// inverse of `decodeNoGoodLiteral`; the round trip is what makes republishing
/// the same counterexample idempotent.
llvm::Expected<Attribute>
encodeNoGoodLiteral(MLIRContext *context, const SpatialNoGoodLiteral &literal,
                    const ArtifactIdentity &dataflowIdentity) {
  const auto producerAttr =
      [&](const ::dataflow::CanonicalGraphProducerEndpointRef &producer)
      -> llvm::Expected<::mapping::GraphProducerEndpointRefAttr> {
    auto encoded =
        ::dataflow::encodeDataflowReference(dataflowIdentity, producer);
    if (!encoded)
      return encoded.takeError();
    return ::mapping::GraphProducerEndpointRefAttr::get(
        context, denseBytes(context, *encoded));
  };
  const auto consumerAttr =
      [&](const ::dataflow::CanonicalGraphConsumerEndpointRef &consumer)
      -> llvm::Expected<::mapping::GraphConsumerEndpointRefAttr> {
    auto encoded =
        ::dataflow::encodeDataflowReference(dataflowIdentity, consumer);
    if (!encoded)
      return encoded.takeError();
    return ::mapping::GraphConsumerEndpointRefAttr::get(
        context, denseBytes(context, *encoded));
  };

  if (const auto *uses = std::get_if<SpatialNetUsesTraversalLiteral>(&literal)) {
    auto producer = producerAttr(uses->producer);
    if (!producer)
      return producer.takeError();
    ::mapping::GraphConsumerEndpointRefAttr consumer;
    if (uses->consumer) {
      auto encoded = consumerAttr(*uses->consumer);
      if (!encoded)
        return encoded.takeError();
      consumer = *encoded;
    }
    return Attribute(::mapping::NetUsesTraversalAttr::get(
        context, *producer, consumer,
        ::mapping::FabricPhysicalTraversalRefAttr::get(
            context, denseBytes(context, ::loom::fabric::canonicalFabricBytes(
                                             uses->traversal)))));
  }

  if (const auto *tag = std::get_if<SpatialNetTagEqualsLiteral>(&literal)) {
    auto producer = producerAttr(tag->producer);
    if (!producer)
      return producer.takeError();
    const llvm::APInt value =
        ::fabric::canonicalPhysicalTagValue(tag->value);
    return Attribute(::mapping::NetTagEqualsAttr::get(
        context, *producer, tag->segmentOrdinal,
        IntegerAttr::get(IntegerType::get(context, value.getBitWidth()),
                         value)));
  }

  if (const auto *mapping =
          std::get_if<SpatialMappingIdentityEqualsLiteral>(&literal)) {
    return Attribute(::mapping::SpatialMappingIdentityEqualsAttr::get(
        context,
        ::mapping::ArtifactRootReferenceAttr::get(
            context,
            denseBytes(context,
                       encodeArtifactRootReference(mapping->mapping)))));
  }

  const auto &attachment =
      std::get<SpatialTransferAttachmentEqualsLiteral>(literal);
  auto producer = producerAttr(attachment.terminal.producer);
  if (!producer)
    return producer.takeError();
  ::mapping::GraphConsumerEndpointRefAttr consumer;
  if (attachment.terminal.consumer) {
    auto encoded = consumerAttr(*attachment.terminal.consumer);
    if (!encoded)
      return encoded.takeError();
    consumer = *encoded;
  }
  return Attribute(::mapping::TransferAttachmentEqualsAttr::get(
      context,
      ::mapping::SpatialTransferTerminalAttr::get(context, *producer, consumer),
      ::mapping::FabricTransportEndpointRefAttr::get(
          context, denseBytes(context, ::loom::fabric::canonicalFabricBytes(
                                           attachment.endpoint)))));
}

struct PreparedSpatialConstraintSet final {
  ArtifactRootReference reference;
  CanonicalSemanticBytes canonicalBytes;
  std::unique_ptr<MLIRContext> context;
  OwningOpRef<ModuleOp> module;
  ::mapping::ConstraintsSpatialOp root;
};

llvm::Expected<PreparedSpatialConstraintSet>
prepareSpatialConstraintSet(::mapping::ConstraintsSpatialOp source) {
  auto canonicalBytes = writeCanonicalSpatialConstraintAssembly(source);
  if (!canonicalBytes)
    return canonicalBytes.takeError();
  auto parsed = parseSpatialConstraintRoot(*canonicalBytes);
  if (!parsed)
    return parsed.takeError();
  ArtifactRootReference reference{
      mappingConstraintSetSchema.identity.str(),
      mappingConstraintSetSchema.version,
      finalizeArtifactIdentity(mappingConstraintSetSchema, *canonicalBytes)};
  return PreparedSpatialConstraintSet{
      std::move(reference), std::move(*canonicalBytes),
      std::move(parsed->context), std::move(parsed->module), parsed->root};
}

llvm::Error requirePublishedUpstreams(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const TechMappingView &techMapping,
    const ::loom::fabric::FabricArtifactView &fabric,
    const ArtifactStore &store) {
  const ArtifactRootReference dataflowReference{
      ::dataflow::canonicalDataflowSchema.identity.str(),
      ::dataflow::canonicalDataflowSchema.version, dataflow.identity()};
  const ArtifactRootReference techMappingReference{
      mappingArtifactSchema.identity.str(), mappingArtifactSchema.version,
      techMapping.identity()};
  const ArtifactRootReference fabricReference{
      ::loom::fabric::fabricArtifactSchema.identity.str(),
      ::loom::fabric::fabricArtifactSchema.version, fabric.identity()};
  auto dataflowBytes = store.get(dataflowReference);
  if (!dataflowBytes)
    return dataflowBytes.takeError();
  auto techMappingBytes = store.get(techMappingReference);
  if (!techMappingBytes)
    return techMappingBytes.takeError();
  auto fabricBytes = store.get(fabricReference);
  if (!fabricBytes)
    return fabricBytes.takeError();
  return llvm::Error::success();
}

llvm::Expected<SpatialMappingConstraintSetView>
strictImport(const ArtifactIdentity &identity,
             const CanonicalSemanticBytes &canonicalBytes,
             const ArtifactStore &store,
             llvm::ArrayRef<SpatialMappingIdentityEqualsLiteral>
                 importedMappingCache = {}) {
  if (finalizeArtifactIdentity(mappingConstraintSetSchema, canonicalBytes) !=
      identity)
    return invalid("constraint identity does not match canonical bytes");
  auto parsed = parseSpatialConstraintRoot(canonicalBytes);
  if (!parsed)
    return parsed.takeError();

  auto dataflowIdentity = decodeIdentity(parsed->root.getDataflow());
  if (!dataflowIdentity)
    return dataflowIdentity.takeError();
  auto techMappingIdentity = decodeIdentity(parsed->root.getTechMapping());
  if (!techMappingIdentity)
    return techMappingIdentity.takeError();
  auto fabricIdentity = decodeIdentity(parsed->root.getFabric());
  if (!fabricIdentity)
    return fabricIdentity.takeError();

  ArtifactRootReference dataflowReference{
      ::dataflow::canonicalDataflowSchema.identity.str(),
      ::dataflow::canonicalDataflowSchema.version, *dataflowIdentity};
  auto dataflow = ::dataflow::importCanonicalDataflow(dataflowReference, store);
  if (!dataflow)
    return dataflow.takeError();
  auto dataflowView = dataflow->view();
  if (!dataflowView)
    return dataflowView.takeError();

  ArtifactRootReference techMappingReference{
      mappingArtifactSchema.identity.str(), mappingArtifactSchema.version,
      *techMappingIdentity};
  auto techMapping = importTechMapping(techMappingReference, store);
  if (!techMapping)
    return techMapping.takeError();

  ArtifactRootReference fabricReference{
      ::loom::fabric::fabricArtifactSchema.identity.str(),
      ::loom::fabric::fabricArtifactSchema.version, *fabricIdentity};
  auto fabric = ::loom::fabric::importEntireFabricRoot(fabricReference, store);
  if (!fabric)
    return fabric.takeError();

  auto view = SpatialMappingConstraintSetView::import(
      identity, parsed->root, *dataflowView, techMapping->view(),
      fabric->view(), store, importedMappingCache);
  if (!view)
    return view.takeError();
  auto rewritten = writeCanonicalSpatialConstraintAssembly(parsed->root);
  if (!rewritten)
    return rewritten.takeError();
  if (!rewritten->bytes().equals(canonicalBytes.bytes()))
    return invalid("stored Spatial constraint payload is not canonical");
  return view;
}

llvm::Error publishPreparedSpatialConstraintSet(
    const PreparedSpatialConstraintSet &prepared, const ArtifactStore &store) {
  auto stored = store.put(mappingConstraintSetSchema, prepared.canonicalBytes);
  if (!stored)
    return stored.takeError();
  if (*stored != prepared.reference.artifact)
    return invalid("ArtifactStore returned a different constraint identity");
  return llvm::Error::success();
}

} // namespace

llvm::Expected<CanonicalSemanticBytes>
writeCanonicalSpatialConstraintAssembly(::mapping::ConstraintsSpatialOp root) {
  OwningOpRef<Operation *> clone(root->clone());
  auto canonical = cast<::mapping::ConstraintsSpatialOp>(clone.get());
  if (failed(verify(canonical)))
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "Spatial MappingConstraintSet is structurally invalid");

  Block &body = canonical.getBody().front();
  detail::canonicalizeConstraintClauses(
      body, canonical.getLoc(),
      [](MLIRContext *context, Attribute projection,
         ArrayRef<Attribute> values) {
        const auto kind = static_cast<::mapping::SpatialConstraintProjection>(
            cast<::mapping::SpatialConstraintProjectionKeyAttr>(projection)
                .getValue());
        return normalizeDomain(context, kind, values);
      },
      [](MLIRContext *context, Attribute projection, ArrayRef<Attribute> lhs,
         ArrayRef<Attribute> rhs) {
        const auto kind = static_cast<::mapping::SpatialConstraintProjection>(
            cast<::mapping::SpatialConstraintProjectionKeyAttr>(projection)
                .getValue());
        return intersectDomains(context, kind, lhs, rhs);
      });
  if (failed(verify(canonical)))
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "canonical Spatial MappingConstraintSet is structurally invalid");

  std::string text;
  llvm::raw_string_ostream stream(text);
  canonical.print(stream, OpPrintingFlags().enableDebugInfo(false));
  stream << '\n';
  stream.flush();
  return CanonicalSemanticBytes(
      std::vector<std::uint8_t>(text.begin(), text.end()));
}

llvm::Expected<SpatialMappingConstraintSetView>
SpatialMappingConstraintSetView::import(
    const ArtifactIdentity &identity, ::mapping::ConstraintsSpatialOp root,
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const TechMappingView &techMapping,
    const ::loom::fabric::FabricArtifactView &fabric,
    const ArtifactStore &store,
    llvm::ArrayRef<SpatialMappingIdentityEqualsLiteral>
        importedMappingCache) {
  auto dataflowIdentity = decodeIdentity(root.getDataflow());
  if (!dataflowIdentity)
    return dataflowIdentity.takeError();
  auto techMappingIdentity = decodeIdentity(root.getTechMapping());
  if (!techMappingIdentity)
    return techMappingIdentity.takeError();
  auto fabricIdentity = decodeIdentity(root.getFabric());
  if (!fabricIdentity)
    return fabricIdentity.takeError();

  if (*dataflowIdentity != dataflow.identity())
    return invalid(
        "Spatial constraint dataflow binding does not match importer");
  if (*techMappingIdentity != techMapping.identity())
    return invalid(
        "Spatial constraint TechMapping binding does not match importer");
  if (*fabricIdentity != fabric.identity())
    return invalid("Spatial constraint Fabric binding does not match importer");
  if (techMapping.dataflowIdentity() != dataflow.identity() ||
      techMapping.fabricIdentity() != fabric.identity())
    return invalid(
        "Spatial constraint inputs do not form one exact D/T/F tuple");

  std::vector<SpatialConstraintClauseView> clauses;
  clauses.reserve(std::distance(root.getBody().front().begin(),
                                root.getBody().front().end()));
  for (Operation &operation : root.getBody().front()) {
    if (auto restriction =
            dyn_cast<::mapping::ConstraintDomainRestrictionOp>(operation)) {
      const auto projection =
          static_cast<::mapping::SpatialConstraintProjection>(
              spatialProjection(restriction).getValue());
      auto subject = decodeSubject(projection, restriction.getSubject(),
                                   dataflow, techMapping);
      if (!subject)
        return subject.takeError();
      auto domain =
          decodeDomain(projection, restriction.getAdmissibleDomain(), fabric);
      if (!domain)
        return domain.takeError();
      clauses.emplace_back(SpatialDomainRestrictionView{
          projection, std::move(*subject), std::move(*domain)});
      continue;
    }
    if (auto equal = dyn_cast<::mapping::ConstraintEqualOp>(operation)) {
      const auto projection =
          static_cast<::mapping::SpatialConstraintProjection>(
              spatialProjection(equal).getValue());
      auto subjects = decodeSubjects(projection, equal.getSubjects(), dataflow,
                                     techMapping);
      if (!subjects)
        return subjects.takeError();
      clauses.emplace_back(SpatialEqualView{projection, std::move(*subjects)});
      continue;
    }
    if (auto disjoint = dyn_cast<::mapping::ConstraintDisjointOp>(operation)) {
      const auto projection =
          static_cast<::mapping::SpatialConstraintProjection>(
              spatialProjection(disjoint).getValue());
      auto subjects = decodeSubjects(projection, disjoint.getSubjects(),
                                     dataflow, techMapping);
      if (!subjects)
        return subjects.takeError();
      clauses.emplace_back(
          SpatialDisjointView{projection, std::move(*subjects)});
      continue;
    }
    auto noGood =
        dyn_cast<::mapping::ConstraintRuntimeCounterexampleNoGoodOp>(operation);
    if (!noGood)
      return invalid("Spatial constraint body holds an unknown clause kind");
    if (noGood.getLiterals().empty())
      return invalid("runtime-counterexample no-good clause is empty");
    SpatialRuntimeCounterexampleNoGoodView clause;
    if (auto lineage = noGood.getLineage()) {
      auto decoded = decodeRuntimeCounterexampleLineage(*lineage);
      if (!decoded)
        return decoded.takeError();
      clause.lineage = std::move(*decoded);
    }
    clause.literals.reserve(noGood.getLiterals().size());
    for (Attribute attribute : noGood.getLiterals()) {
      auto literal =
          decodeNoGoodLiteral(attribute, dataflow, techMapping, fabric,
                              store, importedMappingCache);
      if (!literal)
        return literal.takeError();
      clause.literals.push_back(std::move(*literal));
    }
    clauses.emplace_back(std::move(clause));
  }

  return SpatialMappingConstraintSetView(
      identity, std::move(*dataflowIdentity), std::move(*techMappingIdentity),
      std::move(*fabricIdentity), std::move(clauses));
}

llvm::Expected<FinalizedSpatialMappingConstraintSet>
finalizeSpatialMappingConstraintSet(::mapping::ConstraintsSpatialOp source,
                                    const ArtifactStore &store) {
  auto prepared = prepareSpatialConstraintSet(source);
  if (!prepared)
    return prepared.takeError();
  auto view = strictImport(prepared->reference.artifact,
                           prepared->canonicalBytes, store);
  if (!view)
    return view.takeError();
  if (llvm::Error error = publishPreparedSpatialConstraintSet(*prepared, store))
    return std::move(error);
  return FinalizedSpatialMappingConstraintSet(
      std::move(prepared->reference), std::move(prepared->canonicalBytes),
      std::move(*view));
}

llvm::Expected<FinalizedSpatialMappingConstraintSet>
finalizeSpatialMappingConstraintSet(
    ::mapping::ConstraintsSpatialOp source,
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const TechMappingView &techMapping,
    const ::loom::fabric::FabricArtifactView &fabric,
    const ArtifactStore &store) {
  if (llvm::Error error =
          requirePublishedUpstreams(dataflow, techMapping, fabric, store))
    return std::move(error);
  auto prepared = prepareSpatialConstraintSet(source);
  if (!prepared)
    return prepared.takeError();
  auto view = SpatialMappingConstraintSetView::import(
      prepared->reference.artifact, prepared->root, dataflow, techMapping,
      fabric, store);
  if (!view)
    return view.takeError();
  if (llvm::Error error = publishPreparedSpatialConstraintSet(*prepared, store))
    return std::move(error);
  return FinalizedSpatialMappingConstraintSet(
      std::move(prepared->reference), std::move(prepared->canonicalBytes),
      std::move(*view));
}

llvm::Error
appendRuntimeCounterexampleConstraint(
    ::mapping::ConstraintsSpatialOp root,
    const ArtifactIdentity &dataflowIdentity,
    llvm::ArrayRef<SpatialNoGoodLiteral> literals,
    std::optional<SpatialRuntimeCounterexampleNoGoodView::Lineage> lineage) {
  if (literals.empty())
    return invalid("a runtime-counterexample no-good clause must be non-empty");
  MLIRContext *context = root.getContext();
  OpBuilder builder(context);
  builder.setInsertionPointToEnd(&root.getBody().front());
  std::vector<Attribute> encoded;
  encoded.reserve(literals.size());
  for (const SpatialNoGoodLiteral &literal : literals) {
    auto attribute = encodeNoGoodLiteral(context, literal, dataflowIdentity);
    if (!attribute)
      return attribute.takeError();
    encoded.push_back(*attribute);
  }
  ::mapping::RuntimeCounterexampleLineageAttr lineageAttr;
  if (lineage) {
    const auto rootAttr = [&](const ArtifactRootReference &reference) {
      return ::mapping::ArtifactRootReferenceAttr::get(
          context,
          denseBytes(context, encodeArtifactRootReference(reference)));
    };
    lineageAttr = ::mapping::RuntimeCounterexampleLineageAttr::get(
        context, rootAttr(lineage->parentMapping),
        rootAttr(lineage->runtimeEvidence),
        rootAttr(lineage->evaluationRequest),
        rootAttr(lineage->runtimeExecution),
        denseBytes(context, lineage->certificateDigest.bytes()));
  }
  ::mapping::ConstraintRuntimeCounterexampleNoGoodOp::create(
      builder, builder.getUnknownLoc(), builder.getArrayAttr(encoded),
      lineageAttr);
  return llvm::Error::success();
}

llvm::Expected<FinalizedSpatialMappingConstraintSet>
finalizeRuntimeCounterexampleConstraintSet(
    const ArtifactRootReference &parent,
    llvm::ArrayRef<SpatialNoGoodLiteral> literals, const ArtifactStore &store,
    std::optional<SpatialRuntimeCounterexampleNoGoodView::Lineage> lineage) {
  // Importing the parent proves it is a 1.3 payload over one exact D/T/F
  // closure before anything is added to it.
  auto imported = importSpatialMappingConstraintSet(parent, store);
  if (!imported)
    return imported.takeError();
  auto parsed = parseSpatialConstraintRoot(imported->canonicalBytes());
  if (!parsed)
    return parsed.takeError();
  if (llvm::Error error = appendRuntimeCounterexampleConstraint(
          parsed->root, imported->view().dataflowIdentity(), literals,
          lineage))
    return std::move(error);
  // Canonicalization owns literal ordering, deduplication, and the union with
  // any clause the parent already carried, so republishing an identical
  // counterexample reproduces the parent identity exactly.
  return finalizeSpatialMappingConstraintSet(parsed->root, store);
}

llvm::Expected<FinalizedSpatialMappingConstraintSet>
finalizeSpatialRuntimeCounterexampleConstraintSet(
    const ArtifactRootReference &parent,
    llvm::ArrayRef<SpatialNoGoodLiteral> literals, const ArtifactStore &store) {
  return finalizeRuntimeCounterexampleConstraintSet(parent, literals, store,
                                                     std::nullopt);
}

llvm::Expected<FinalizedSpatialMappingConstraintSet>
finalizePromotedSpatialRuntimeCounterexampleConstraintSet(
    const ArtifactRootReference &parent,
    llvm::ArrayRef<SpatialNoGoodLiteral> literals,
    const SpatialRuntimeCounterexampleNoGoodView::Lineage &lineage,
    const ArtifactStore &store) {
  return finalizeRuntimeCounterexampleConstraintSet(parent, literals, store,
                                                     lineage);
}

llvm::Expected<FinalizedSpatialMappingConstraintSet>
finalizePromotedSpatialRuntimeCounterexampleConstraintSet(
    const FinalizedSpatialMappingConstraintSet &parent,
    llvm::ArrayRef<SpatialNoGoodLiteral> literals,
    const SpatialRuntimeCounterexampleNoGoodView::Lineage &lineage,
    const ArtifactStore &store) {
  std::vector<SpatialMappingIdentityEqualsLiteral> importedMappingCache;
  const auto remember = [&](const SpatialMappingIdentityEqualsLiteral &literal)
      -> llvm::Error {
    if (!literal.importedMapping)
      return llvm::Error::success();
    if (literal.importedMapping->reference() != literal.mapping)
      return invalid("no-good Mapping cache differs from its literal");
    if (llvm::any_of(importedMappingCache, [&](const auto &cached) {
          return cached.mapping == literal.mapping;
        }))
      return llvm::Error::success();
    importedMappingCache.push_back(literal);
    return llvm::Error::success();
  };
  for (const SpatialConstraintClauseView &clause : parent.view().clauses()) {
    const auto *noGood =
        std::get_if<SpatialRuntimeCounterexampleNoGoodView>(&clause);
    if (!noGood)
      continue;
    for (const SpatialNoGoodLiteral &literal : noGood->literals)
      if (const auto *identity =
              std::get_if<SpatialMappingIdentityEqualsLiteral>(&literal))
        if (llvm::Error error = remember(*identity))
          return std::move(error);
  }
  for (const SpatialNoGoodLiteral &literal : literals)
    if (const auto *identity =
            std::get_if<SpatialMappingIdentityEqualsLiteral>(&literal))
      if (llvm::Error error = remember(*identity))
        return std::move(error);

  auto parsed = parseSpatialConstraintRoot(parent.canonicalBytes());
  if (!parsed)
    return parsed.takeError();
  if (llvm::Error error = appendRuntimeCounterexampleConstraint(
          parsed->root, parent.view().dataflowIdentity(), literals, lineage))
    return std::move(error);
  auto prepared = prepareSpatialConstraintSet(parsed->root);
  if (!prepared)
    return prepared.takeError();
  auto view = strictImport(prepared->reference.artifact,
                           prepared->canonicalBytes, store,
                           importedMappingCache);
  if (!view)
    return view.takeError();
  if (llvm::Error error = publishPreparedSpatialConstraintSet(*prepared, store))
    return std::move(error);
  return FinalizedSpatialMappingConstraintSet(
      std::move(prepared->reference), std::move(prepared->canonicalBytes),
      std::move(*view));
}

llvm::Expected<FinalizedSpatialMappingConstraintSet>
importSpatialMappingConstraintSet(const ArtifactRootReference &reference,
                                  const ArtifactStore &store) {
  if (reference.schemaIdentity != mappingConstraintSetSchema.identity ||
      reference.schemaVersion != mappingConstraintSetSchema.version)
    return invalid("root reference has the wrong constraint schema");
  auto canonicalBytes = store.get(reference);
  if (!canonicalBytes)
    return canonicalBytes.takeError();
  auto view = strictImport(reference.artifact, *canonicalBytes, store);
  if (!view)
    return view.takeError();
  return FinalizedSpatialMappingConstraintSet(
      reference, std::move(*canonicalBytes), std::move(*view));
}

} // namespace loom::mapping
