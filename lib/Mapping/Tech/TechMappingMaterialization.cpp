#include "TechMappingCandidate.h"

#include "Dataflow/IR/DataflowReferenceCodec.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Mapping/IR/MappingDialect.h"
#include "Mapping/IR/MappingOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace loom::mapping::detail {
namespace {

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

llvm::Expected<mlir::DenseI64ArrayAttr>
ordinalArray(mlir::MLIRContext *context,
             llvm::ArrayRef<std::uint64_t> ordinals) {
  std::vector<std::int64_t> values;
  values.reserve(ordinals.size());
  for (std::uint64_t ordinal : ordinals) {
    if (ordinal >
        static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "tech_mapping_generation_invalid: port ordinal exceeds i64");
    values.push_back(static_cast<std::int64_t>(ordinal));
  }
  return mlir::DenseI64ArrayAttr::get(context, values);
}

llvm::Expected<mlir::ArrayAttr> memoryEndpointArray(
    mlir::MLIRContext *context,
    llvm::ArrayRef<::loom::fabric::FabricMemoryEngineTemplateEndpointRef>
        endpoints) {
  std::vector<mlir::Attribute> attributes;
  attributes.reserve(endpoints.size());
  for (const auto &endpoint : endpoints)
    attributes.push_back(
        fabricAttr<::mapping::FabricMemoryEngineTemplateEndpointRefAttr>(
            context, endpoint));
  return mlir::ArrayAttr::get(context, attributes);
}

llvm::Error materializeCompute(mlir::OpBuilder &builder,
                               mlir::Location location,
                               const ArtifactIdentity &dataflowOwner,
                               const TechComputeRealizationView &row,
                               std::uint64_t entityId, mlir::Block &parent) {
  builder.setInsertionPointToEnd(&parent);
  auto realization = ::mapping::ComputeRealizationOp::create(
      builder, location, entityId,
      fabricAttr<::mapping::FabricFuCapabilityTemplateRefAttr>(
          builder.getContext(), row.capabilityTemplate));
  mlir::Block *body = new mlir::Block();
  realization.getBody().push_back(body);
  builder.setInsertionPointToEnd(body);
  for (const TechComputeActorView &actor : row.actors) {
    auto actorReference = dataflowAttr<::mapping::ActorRefAttr>(
        builder.getContext(), dataflowOwner, actor.actor);
    if (!actorReference)
      return actorReference.takeError();
    auto operands = ordinalArray(builder.getContext(), actor.operandPorts);
    if (!operands)
      return operands.takeError();
    auto results = ordinalArray(builder.getContext(), actor.resultPorts);
    if (!results)
      return results.takeError();
    ::mapping::ComputeActorOp::create(
        builder, location, *actorReference,
        fabricAttr<::mapping::FabricFuTemplateNodeRefAttr>(
            builder.getContext(), actor.fabricOperation),
        *operands, *results);
  }
  for (const TechComputeBoundaryView &boundary : row.boundaries) {
    auto actorReference = dataflowAttr<::mapping::ActorRefAttr>(
        builder.getContext(), dataflowOwner, boundary.actor);
    if (!actorReference)
      return actorReference.takeError();
    ::mapping::ComputeBoundaryOp::create(
        builder, location, *actorReference,
        boundary.direction == ::loom::fabric::FabricPortDirection::Input
            ? ::mapping::PortDirection::Input
            : ::mapping::PortDirection::Output,
        boundary.portOrdinal,
        fabricAttr<::mapping::FabricFuTemplatePortRefAttr>(
            builder.getContext(), boundary.fabricPort));
  }
  return llvm::Error::success();
}

llvm::Error materializeMemory(mlir::OpBuilder &builder, mlir::Location location,
                              const ArtifactIdentity &dataflowOwner,
                              const TechMemoryRealizationView &row,
                              std::uint64_t entityId, mlir::Block &parent) {
  builder.setInsertionPointToEnd(&parent);
  auto realization = ::mapping::MemoryRealizationOp::create(
      builder, location, entityId,
      fabricAttr<::mapping::FabricMemoryEngineTemplateRefAttr>(
          builder.getContext(), row.engine));
  mlir::Block *body = new mlir::Block();
  realization.getBody().push_back(body);
  builder.setInsertionPointToEnd(body);
  for (const TechMemoryActorView &actor : row.actors) {
    auto actorReference = dataflowAttr<::mapping::ActorRefAttr>(
        builder.getContext(), dataflowOwner, actor.actor);
    if (!actorReference)
      return actorReference.takeError();
    auto operands =
        memoryEndpointArray(builder.getContext(), actor.operandPorts);
    if (!operands)
      return operands.takeError();
    auto results = memoryEndpointArray(builder.getContext(), actor.resultPorts);
    if (!results)
      return results.takeError();
    ::mapping::MemoryActorOp::create(
        builder, location, *actorReference,
        fabricAttr<::mapping::FabricMemoryEngineTemplateOperationPortRefAttr>(
            builder.getContext(), actor.operationPort),
        fabricAttr<
            ::mapping::FabricMemoryEngineTemplateCapabilityAlternativeRefAttr>(
            builder.getContext(), actor.capability),
        *operands, *results);
  }
  for (const TechMemoryGraphBoundaryView &boundary : row.graphBoundaries) {
    mlir::Attribute terminal;
    if (const auto *producer =
            std::get_if<::dataflow::CanonicalGraphProducerEndpointRef>(
                &boundary.terminal)) {
      auto encoded = dataflowAttr<::mapping::GraphProducerEndpointRefAttr>(
          builder.getContext(), dataflowOwner, *producer);
      if (!encoded)
        return encoded.takeError();
      terminal = *encoded;
    } else {
      auto encoded = dataflowAttr<::mapping::GraphConsumerEndpointRefAttr>(
          builder.getContext(), dataflowOwner,
          std::get<::dataflow::CanonicalGraphConsumerEndpointRef>(
              boundary.terminal));
      if (!encoded)
        return encoded.takeError();
      terminal = *encoded;
    }
    ::mapping::MemoryGraphBoundaryOp::create(
        builder, location, terminal,
        fabricAttr<::mapping::FabricMemoryEngineTemplateEndpointRefAttr>(
            builder.getContext(), boundary.endpoint));
  }
  for (const TechMemoryInternalEdgeView &edge : row.internalEdges) {
    auto producer = dataflowAttr<::mapping::GraphProducerEndpointRefAttr>(
        builder.getContext(), dataflowOwner, edge.producer);
    if (!producer)
      return producer.takeError();
    auto consumer = dataflowAttr<::mapping::GraphConsumerEndpointRefAttr>(
        builder.getContext(), dataflowOwner, edge.consumer);
    if (!consumer)
      return consumer.takeError();
    ::mapping::MemoryInternalEdgeOp::create(
        builder, location, *producer, *consumer,
        fabricAttr<
            ::mapping::FabricMemoryEngineTemplateInternalConnectionRefAttr>(
            builder.getContext(), edge.connection));
  }
  return llvm::Error::success();
}

} // namespace

llvm::Expected<ArtifactRootReference>
materializeTechMappingCandidate(const TechMappingGenerationInputs &inputs,
                                llvm::ArrayRef<const TechMatchRow *> rows) {
  mlir::DialectRegistry registry;
  registry.insert<::mapping::MappingDialect>();
  mlir::MLIRContext context(registry, mlir::MLIRContext::Threading::DISABLED);
  context.loadDialect<::mapping::MappingDialect>();
  mlir::OpBuilder builder(&context);
  const mlir::Location location = builder.getUnknownLoc();
  auto module = mlir::ModuleOp::create(location);
  builder.setInsertionPointToStart(module.getBody());

  std::vector<mlir::Attribute> covers;
  covers.reserve(inputs.covers.size());
  for (const auto &graph : inputs.covers) {
    auto attribute = dataflowAttr<::mapping::GraphRefAttr>(
        &context, inputs.dataflow.identity(), graph);
    if (!attribute)
      return attribute.takeError();
    covers.push_back(*attribute);
  }
  auto root = ::mapping::TechOp::create(
      builder, location, identityAttr(&context, inputs.dataflow.identity()),
      identityAttr(&context, inputs.fabric.identity()),
      mlir::ArrayAttr::get(&context, covers));
  mlir::Block *body = new mlir::Block();
  root.getBody().push_back(body);

  for (std::size_t ordinal = 0; ordinal < rows.size(); ++ordinal) {
    const TechMatchRow *row = rows[ordinal];
    llvm::Error error = std::visit(
        [&](const auto &realization) -> llvm::Error {
          using T = std::decay_t<decltype(realization)>;
          if constexpr (std::is_same_v<T, TechComputeRealizationView>)
            return materializeCompute(builder, location,
                                      inputs.dataflow.identity(), realization,
                                      ordinal, *body);
          else
            return materializeMemory(builder, location,
                                     inputs.dataflow.identity(), realization,
                                     ordinal, *body);
        },
        row->realization);
    if (error)
      return std::move(error);
  }
  auto finalized = finalizeTechMapping(root, inputs.store);
  if (!finalized)
    return finalized.takeError();
  return finalized->reference();
}

} // namespace loom::mapping::detail
