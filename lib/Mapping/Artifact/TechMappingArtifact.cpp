#include "Mapping/Artifact/MappingArtifact.h"

#include "Common/ArtifactFinalizer.h"
#include "Common/IndexWidth.h"
#include "Common/PointerLayout.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Dataflow/IR/DataflowOps.h"
#include "Dataflow/IR/DataflowReferenceCodec.h"
#include "Dataflow/IR/DataflowServiceSchema.h"
#include "Dataflow/IR/OperationSchema.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Mapping/IR/MappingDialect.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
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
#include <utility>
#include <variant>
#include <vector>

using namespace mlir;

namespace loom::mapping {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "mapping_artifact_invalid: " + message);
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

template <typename Ref, typename Attr>
llvm::Expected<Ref> decodeDataflow(Attr attribute,
                                   const ArtifactIdentity &dataflowIdentity) {
  return ::dataflow::decodeDataflowReference<Ref>(
      unsignedBytes(attribute.getRecord()), dataflowIdentity);
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

struct ParsedTechRoot final {
  std::unique_ptr<MLIRContext> context;
  OwningOpRef<ModuleOp> module;
  ::mapping::TechOp root;
};

llvm::Expected<ParsedTechRoot>
parseTechRoot(const CanonicalSemanticBytes &canonicalBytes) {
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
    return invalid("canonical mapping payload cannot be parsed");

  ::mapping::TechOp root;
  unsigned rootCount = 0;
  for (Operation &operation : module->getBody()->without_terminator()) {
    auto candidate = dyn_cast<::mapping::TechOp>(operation);
    if (!candidate)
      return invalid("mapping artifact contains a non-TechMapping root");
    root = candidate;
    ++rootCount;
  }
  if (rootCount != 1)
    return invalid(
        "mapping artifact must contain exactly one TechMapping root");
  if (failed(verify(root)))
    return invalid("mapping artifact root is structurally invalid");
  return ParsedTechRoot{std::move(context), std::move(module), root};
}

llvm::Expected<std::optional<PointerLayout>>
pointerLayoutFor(const ::dataflow::CanonicalActorSchemaProjection &projection,
                 Operation *actor) {
  auto addressSpace = ::dataflow::projectActorPointerAddressSpace(projection);
  if (!addressSpace)
    return addressSpace.takeError();
  if (!*addressSpace)
    return std::optional<PointerLayout>{};
  auto layout = resolvePointerLayout(actor, **addressSpace);
  if (!layout)
    return layout.takeError();
  return std::optional<PointerLayout>(*layout);
}

::dataflow::GraphRef graphOf(const ::dataflow::GraphIngressTokenRef &endpoint) {
  return std::visit([](const auto &value) { return value.graph; }, endpoint);
}

::dataflow::GraphRef graphOf(const ::dataflow::GraphEgressTokenRef &endpoint) {
  return std::visit([](const auto &value) { return value.graph; }, endpoint);
}

llvm::Expected<::dataflow::GraphRef>
graphOf(const ::dataflow::CanonicalGraphProducerEndpointRef &endpoint,
        const ::dataflow::CanonicalDataflowProgramView &dataflow) {
  if (const auto *ingress =
          std::get_if<::dataflow::GraphIngressTokenRef>(&endpoint))
    return graphOf(*ingress);
  const auto &result = std::get<::dataflow::ActorTokenResultRef>(endpoint);
  auto actor = dataflow.resolve(result.actor);
  if (!actor)
    return actor.takeError();
  return actor->graph;
}

llvm::Expected<::dataflow::GraphRef>
graphOf(const ::dataflow::CanonicalGraphConsumerEndpointRef &endpoint,
        const ::dataflow::CanonicalDataflowProgramView &dataflow) {
  if (const auto *egress =
          std::get_if<::dataflow::GraphEgressTokenRef>(&endpoint))
    return graphOf(*egress);
  const auto &operand = std::get<::dataflow::ActorTokenOperandRef>(endpoint);
  auto actor = dataflow.resolve(operand.actor);
  if (!actor)
    return actor.takeError();
  return actor->graph;
}

template <typename Ref>
llvm::Expected<std::vector<std::uint8_t>>
dataflowKey(const ArtifactIdentity &owner, const Ref &reference) {
  return ::dataflow::encodeDataflowReference(owner, reference);
}

struct MemoryActorPorts final {
  ::dataflow::ActorRef actor;
  std::vector<::loom::fabric::FabricMemoryEngineTemplateEndpointRef> operands;
  std::vector<::loom::fabric::FabricMemoryEngineTemplateEndpointRef> results;
};

struct MemoryBoundaryKey final {
  bool producer = false;
  std::vector<std::uint8_t> terminal;

  friend bool operator<(const MemoryBoundaryKey &lhs,
                        const MemoryBoundaryKey &rhs) {
    return std::tie(lhs.producer, lhs.terminal) <
           std::tie(rhs.producer, rhs.terminal);
  }
};

using MemoryEdgeKey =
    std::pair<std::vector<std::uint8_t>, std::vector<std::uint8_t>>;

llvm::Expected<::loom::fabric::FabricMemoryEngineTemplateEndpointRef>
decodeMemoryEndpoint(Attribute attribute) {
  auto endpoint =
      dyn_cast<::mapping::FabricMemoryEngineTemplateEndpointRefAttr>(attribute);
  if (!endpoint)
    return invalid("memory port map contains a non-endpoint reference");
  return decodeFabric<::loom::fabric::FabricMemoryEngineTemplateEndpointRef>(
      endpoint);
}

llvm::Expected<
    std::vector<::loom::fabric::FabricMemoryEngineTemplateEndpointRef>>
decodeMemoryEndpoints(ArrayAttr attributes) {
  std::vector<::loom::fabric::FabricMemoryEngineTemplateEndpointRef> result;
  result.reserve(attributes.size());
  for (Attribute attribute : attributes) {
    auto endpoint = decodeMemoryEndpoint(attribute);
    if (!endpoint)
      return endpoint.takeError();
    result.push_back(*endpoint);
  }
  return result;
}

llvm::Expected<::loom::fabric::FabricMemoryEngineTemplateEndpointRef>
endpointForRole(
    const ::loom::fabric::FabricMemoryEngineTemplateRef &engine,
    const ::loom::fabric::MemoryCapabilityAlternativeView &capability,
    ::dataflow::semantics::ServiceValueRole role) {
  const auto binding = llvm::find_if(
      capability.roleToEndpoint,
      [&](const ::fabric::MemoryRoleEndpointBindingRecord &candidate) {
        return candidate.role == role;
      });
  if (binding == capability.roleToEndpoint.end())
    return invalid("selected memory capability omits a service role");
  return ::loom::fabric::FabricMemoryEngineTemplateEndpointRef{
      engine, binding->endpointOrdinal};
}

llvm::Error verifyMemoryEndpointDirection(
    const ::loom::fabric::FabricArtifactView &fabric,
    const ::loom::fabric::FabricMemoryEngineTemplateEndpointRef &endpoint,
    ::loom::fabric::FabricPortDirection expected) {
  const auto *descriptor = fabric.memoryEngineTemplateEndpoint(endpoint);
  if (!descriptor)
    return invalid("memory endpoint reference does not resolve");
  if (descriptor->direction != expected)
    return invalid("memory endpoint has the wrong direction");
  return llvm::Error::success();
}

llvm::Expected<TechMemoryActorView>
importMemoryActor(::mapping::MemoryActorOp record,
                  const ::dataflow::CanonicalDataflowProgramView &dataflow,
                  const ::loom::fabric::FabricArtifactView &fabric,
                  const ::loom::fabric::FabricMemoryEngineTemplateRef &engine) {
  auto actorRef = contextual(decodeDataflow<::dataflow::ActorRef>(
                                 record.getActor(), dataflow.identity()),
                             "memory actor reference is malformed");
  if (!actorRef)
    return actorRef.takeError();
  auto actor = contextual(dataflow.resolve(*actorRef),
                          "memory actor reference does not resolve");
  if (!actor)
    return actor.takeError();
  if (actor->kind != ::dataflow::CanonicalDataflowActorKind::Memory)
    return invalid("non-memory actor is owned by a Memory Realization");

  auto operationPort = contextual(
      decodeFabric<::loom::fabric::FabricMemoryEngineTemplateOperationPortRef>(
          record.getOperationPort()),
      "memory operation-port reference is malformed");
  if (!operationPort)
    return operationPort.takeError();
  auto capability = contextual(
      decodeFabric<
          ::loom::fabric::FabricMemoryEngineTemplateCapabilityAlternativeRef>(
          record.getCapability()),
      "memory capability reference is malformed");
  if (!capability)
    return capability.takeError();
  if (operationPort->engine != engine || capability->port != *operationPort)
    return invalid("memory actor correspondence crosses its engine owner");
  if (llvm::Error error =
          contextual(::loom::fabric::validateFabricRef(fabric, *operationPort),
                     "memory operation-port reference does not resolve"))
    return std::move(error);
  if (llvm::Error error =
          contextual(::loom::fabric::validateFabricRef(fabric, *capability),
                     "memory capability reference does not resolve"))
    return std::move(error);

  auto operands = decodeMemoryEndpoints(record.getOperandPorts());
  if (!operands)
    return operands.takeError();
  auto results = decodeMemoryEndpoints(record.getResultPorts());
  if (!results)
    return results.takeError();
  for (const auto &endpoint : llvm::concat<
           const ::loom::fabric::FabricMemoryEngineTemplateEndpointRef>(
           *operands, *results)) {
    if (endpoint.engine != engine)
      return invalid("memory actor port map crosses its engine owner");
    if (llvm::Error error =
            contextual(::loom::fabric::validateFabricRef(fabric, endpoint),
                       "memory actor endpoint does not resolve"))
      return std::move(error);
  }

  auto projection =
      ::dataflow::projectRegisteredActorSchemaProjection(actor->op);
  if (!projection)
    return projection.takeError();
  auto service = ::dataflow::semantics::CanonicalService::forActor(actor->op);
  if (!service)
    return service.takeError();
  std::optional<::dataflow::semantics::CanonicalMemoryAccessView> access;
  if (service->kind() != ::dataflow::semantics::ServiceKind::MemoryFence) {
    auto projected =
        ::dataflow::semantics::getCanonicalMemoryAccessView(actor->op);
    if (!projected)
      return projected.takeError();
    access.emplace(std::move(*projected));
  }

  const auto *port = fabric.memoryEngineTemplateOperationPort(*operationPort);
  const auto *selected =
      fabric.memoryEngineTemplateCapabilityAlternative(*capability);
  if (!port || !selected)
    return invalid("selected memory capability cannot be resolved");
  auto matches = port->matchingCapabilities(*projection, *service, access);
  if (!matches)
    return matches.takeError();
  if (!llvm::any_of(*matches,
                    [&](const ::fabric::MemoryCapabilityMatch &match) {
                      return match.alternativeOrdinal == capability->ordinal;
                    }))
    return invalid("selected memory capability does not admit the actor");

  if (operands->size() != service->arguments().size() ||
      results->size() != service->results().size())
    return invalid("memory actor port map does not cover every service value");
  for (auto [ordinal, value] : llvm::enumerate(service->arguments())) {
    auto expected = endpointForRole(engine, *selected, value.role);
    if (!expected)
      return expected.takeError();
    if ((*operands)[ordinal] != *expected)
      return invalid("memory actor operand map disagrees with its capability");
    if (llvm::Error error = verifyMemoryEndpointDirection(
            fabric, *expected, ::loom::fabric::FabricPortDirection::Input))
      return std::move(error);
  }
  for (auto [ordinal, value] : llvm::enumerate(service->results())) {
    auto expected = endpointForRole(engine, *selected, value.role);
    if (!expected)
      return expected.takeError();
    if ((*results)[ordinal] != *expected)
      return invalid("memory actor result map disagrees with its capability");
    if (llvm::Error error = verifyMemoryEndpointDirection(
            fabric, *expected, ::loom::fabric::FabricPortDirection::Output))
      return std::move(error);
  }

  return TechMemoryActorView{*actorRef, *operationPort, *capability,
                             std::move(*operands), std::move(*results)};
}

llvm::Expected<TechMemoryGraphBoundaryView> importMemoryBoundary(
    ::mapping::MemoryGraphBoundaryOp record,
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::fabric::FabricArtifactView &fabric,
    const ::loom::fabric::FabricMemoryEngineTemplateRef &engine) {
  std::optional<TechMemoryGraphEndpointRef> terminal;
  ::loom::fabric::FabricPortDirection expectedDirection;
  if (auto producer = dyn_cast<::mapping::GraphProducerEndpointRefAttr>(
          record.getTerminal())) {
    auto decoded =
        decodeDataflow<::dataflow::CanonicalGraphProducerEndpointRef>(
            producer, dataflow.identity());
    if (!decoded)
      return decoded.takeError();
    if (!std::holds_alternative<::dataflow::GraphIngressTokenRef>(*decoded))
      return invalid("memory graph boundary producer is not graph ingress");
    if (llvm::Error error = dataflow.validate(*decoded))
      return std::move(error);
    terminal.emplace(*decoded);
    expectedDirection = ::loom::fabric::FabricPortDirection::Input;
  } else {
    auto consumer =
        cast<::mapping::GraphConsumerEndpointRefAttr>(record.getTerminal());
    auto decoded =
        decodeDataflow<::dataflow::CanonicalGraphConsumerEndpointRef>(
            consumer, dataflow.identity());
    if (!decoded)
      return decoded.takeError();
    if (!std::holds_alternative<::dataflow::GraphEgressTokenRef>(*decoded))
      return invalid("memory graph boundary consumer is not graph egress");
    if (llvm::Error error = dataflow.validate(*decoded))
      return std::move(error);
    terminal.emplace(*decoded);
    expectedDirection = ::loom::fabric::FabricPortDirection::Output;
  }

  auto endpoint =
      decodeFabric<::loom::fabric::FabricMemoryEngineTemplateEndpointRef>(
          record.getEndpoint());
  if (!endpoint)
    return endpoint.takeError();
  if (endpoint->engine != engine)
    return invalid("memory graph boundary crosses its engine owner");
  if (llvm::Error error =
          contextual(::loom::fabric::validateFabricRef(fabric, *endpoint),
                     "memory graph-boundary endpoint does not resolve"))
    return std::move(error);
  if (llvm::Error error =
          verifyMemoryEndpointDirection(fabric, *endpoint, expectedDirection))
    return std::move(error);
  return TechMemoryGraphBoundaryView{std::move(*terminal), *endpoint};
}

llvm::Expected<TechMemoryInternalEdgeView> importMemoryInternalEdge(
    ::mapping::MemoryInternalEdgeOp record,
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::fabric::FabricArtifactView &fabric,
    const ::loom::fabric::FabricMemoryEngineTemplateRef &engine) {
  auto producer = decodeDataflow<::dataflow::CanonicalGraphProducerEndpointRef>(
      record.getProducer(), dataflow.identity());
  if (!producer)
    return producer.takeError();
  auto consumer = decodeDataflow<::dataflow::CanonicalGraphConsumerEndpointRef>(
      record.getConsumer(), dataflow.identity());
  if (!consumer)
    return consumer.takeError();
  if (llvm::Error error = dataflow.validate(*producer))
    return std::move(error);
  if (llvm::Error error = dataflow.validate(*consumer))
    return std::move(error);
  auto actualProducer = dataflow.graphProducer(*consumer);
  if (!actualProducer)
    return actualProducer.takeError();
  if (*actualProducer != *producer)
    return invalid(
        "memory internal-edge record is not a canonical software edge");

  auto connection = decodeFabric<
      ::loom::fabric::FabricMemoryEngineTemplateInternalConnectionRef>(
      record.getConnection());
  if (!connection)
    return connection.takeError();
  if (connection->engine != engine || connection->source.engine != engine ||
      connection->sink.engine != engine)
    return invalid("memory internal connection crosses its engine owner");
  if (llvm::Error error =
          contextual(::loom::fabric::validateFabricRef(fabric, *connection),
                     "memory internal connection does not resolve"))
    return std::move(error);
  return TechMemoryInternalEdgeView{*producer, *consumer, *connection};
}

llvm::Expected<MemoryBoundaryKey>
boundaryKey(const ArtifactIdentity &owner,
            const TechMemoryGraphEndpointRef &terminal) {
  if (const auto *producer =
          std::get_if<::dataflow::CanonicalGraphProducerEndpointRef>(
              &terminal)) {
    auto key = dataflowKey(owner, *producer);
    if (!key)
      return key.takeError();
    return MemoryBoundaryKey{true, std::move(*key)};
  }
  auto key = dataflowKey(
      owner, std::get<::dataflow::CanonicalGraphConsumerEndpointRef>(terminal));
  if (!key)
    return key.takeError();
  return MemoryBoundaryKey{false, std::move(*key)};
}

llvm::Expected<MemoryEdgeKey>
edgeKey(const ArtifactIdentity &owner,
        const ::dataflow::CanonicalGraphProducerEndpointRef &producer,
        const ::dataflow::CanonicalGraphConsumerEndpointRef &consumer) {
  auto producerBytes = dataflowKey(owner, producer);
  if (!producerBytes)
    return producerBytes.takeError();
  auto consumerBytes = dataflowKey(owner, consumer);
  if (!consumerBytes)
    return consumerBytes.takeError();
  return MemoryEdgeKey{std::move(*producerBytes), std::move(*consumerBytes)};
}

llvm::Error verifyMemoryCorrespondenceClosure(
    const TechMemoryRealizationView &realization,
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::fabric::FabricArtifactView &fabric) {
  const auto *engine = fabric.memoryEngineTemplate(realization.engine);
  if (!engine)
    return invalid("Memory Realization engine does not resolve");
  for (const TechMemoryActorView &actor : realization.actors)
    if (actor.operationPort.engine != realization.engine ||
        actor.capability.port != actor.operationPort)
      return invalid("memory actor crosses its Memory Realization engine");
  if (engine->schedule == ::fabric::Schedule::Temporal) {
    if (!engine->residentContextCount ||
        realization.actors.size() > *engine->residentContextCount)
      return invalid(
          "Memory Realization exceeds Temporal resident context capacity");
  } else {
    std::set<std::uint64_t> selectedPorts;
    for (const TechMemoryActorView &actor : realization.actors)
      if (!selectedPorts.insert(actor.operationPort.ordinal).second)
        return invalid("Spatial memory operation port hosts multiple software "
                       "operations");
  }

  std::map<std::uint64_t, const TechMemoryActorView *> actors;
  std::optional<::dataflow::GraphRef> graph;
  for (const TechMemoryActorView &actor : realization.actors) {
    auto resolved = dataflow.resolve(actor.actor);
    if (!resolved)
      return resolved.takeError();
    if (graph && *graph != resolved->graph)
      return invalid("Memory Realization crosses a graph boundary");
    graph = resolved->graph;
    actors.emplace(actor.actor.entity.value(), &actor);
  }
  if (!graph)
    return invalid("Memory Realization has no actor");

  std::map<MemoryBoundaryKey,
           ::loom::fabric::FabricMemoryEngineTemplateEndpointRef>
      boundaries;
  for (const TechMemoryGraphBoundaryView &boundary :
       realization.graphBoundaries) {
    auto key = boundaryKey(dataflow.identity(), boundary.terminal);
    if (!key)
      return key.takeError();
    auto boundaryGraph = std::visit(
        [&](const auto &endpoint) { return graphOf(endpoint, dataflow); },
        boundary.terminal);
    if (!boundaryGraph)
      return boundaryGraph.takeError();
    if (*boundaryGraph != *graph)
      return invalid("memory graph boundary belongs to another graph");
    if (!boundaries.emplace(std::move(*key), boundary.endpoint).second)
      return invalid("duplicate memory graph boundary");
  }

  std::map<MemoryEdgeKey, const TechMemoryInternalEdgeView *> edges;
  for (const TechMemoryInternalEdgeView &edge : realization.internalEdges) {
    auto key = edgeKey(dataflow.identity(), edge.producer, edge.consumer);
    if (!key)
      return key.takeError();
    auto producerGraph = graphOf(edge.producer, dataflow);
    if (!producerGraph)
      return producerGraph.takeError();
    auto consumerGraph = graphOf(edge.consumer, dataflow);
    if (!consumerGraph)
      return consumerGraph.takeError();
    if (*producerGraph != *graph || *consumerGraph != *graph)
      return invalid("memory internal edge belongs to another graph");
    if (!edges.emplace(std::move(*key), &edge).second)
      return invalid("duplicate memory internal edge");
  }

  std::set<MemoryBoundaryKey> requiredBoundaries;
  for (const auto &[id, actor] : actors) {
    (void)id;
    auto resolvedActor = dataflow.resolve(actor->actor);
    if (!resolvedActor)
      return resolvedActor.takeError();
    auto service =
        ::dataflow::semantics::CanonicalService::forActor(resolvedActor->op);
    if (!service)
      return service.takeError();
    for (auto [ordinal, endpoint] : llvm::enumerate(actor->operandPorts)) {
      auto operand = service->argumentValue(resolvedActor->op, ordinal);
      if (!operand)
        return operand.takeError();
      ::dataflow::CanonicalGraphConsumerEndpointRef consumer =
          ::dataflow::ActorTokenOperandRef{actor->actor,
                                           (*operand)->getOperandNumber()};
      auto producer = dataflow.graphProducer(consumer);
      if (!producer)
        return producer.takeError();
      if (std::holds_alternative<::dataflow::GraphIngressTokenRef>(*producer)) {
        auto encoded = dataflowKey(dataflow.identity(), *producer);
        if (!encoded)
          return encoded.takeError();
        MemoryBoundaryKey key{true, std::move(*encoded)};
        requiredBoundaries.insert(key);
        auto found = boundaries.find(key);
        if (found == boundaries.end() || found->second != endpoint)
          return invalid("memory graph ingress correspondence is incomplete");
      }
    }
    for (auto [ordinal, endpoint] : llvm::enumerate(actor->resultPorts)) {
      auto result = service->resultValue(resolvedActor->op, ordinal);
      if (!result)
        return result.takeError();
      ::dataflow::CanonicalGraphProducerEndpointRef producer =
          ::dataflow::ActorTokenResultRef{actor->actor,
                                          result->getResultNumber()};
      auto consumers = dataflow.graphConsumers(producer);
      if (!consumers)
        return consumers.takeError();
      for (const auto &consumer : *consumers) {
        if (!std::holds_alternative<::dataflow::GraphEgressTokenRef>(consumer))
          continue;
        auto encoded = dataflowKey(dataflow.identity(), consumer);
        if (!encoded)
          return encoded.takeError();
        MemoryBoundaryKey key{false, std::move(*encoded)};
        requiredBoundaries.insert(key);
        auto found = boundaries.find(key);
        if (found == boundaries.end() || found->second != endpoint)
          return invalid("memory graph egress correspondence is incomplete");
      }
    }
  }
  if (requiredBoundaries.size() != boundaries.size())
    return invalid("Memory Realization contains an unused graph boundary");

  for (const auto &entry : edges) {
    const TechMemoryInternalEdgeView &edge = *entry.second;
    const auto &connection = edge.connection;
    const auto *producer =
        std::get_if<::dataflow::ActorTokenResultRef>(&edge.producer);
    const auto *consumer =
        std::get_if<::dataflow::ActorTokenOperandRef>(&edge.consumer);
    if (!producer || !consumer)
      return invalid("memory internal edge must connect two memory actors");
    auto sourceActor = actors.find(producer->actor.entity.value());
    auto sinkActor = actors.find(consumer->actor.entity.value());
    if (sourceActor == actors.end() || sinkActor == actors.end())
      return invalid("memory internal edge escapes its realization");

    const auto endpointForResult =
        [&]() -> llvm::Expected<
                  ::loom::fabric::FabricMemoryEngineTemplateEndpointRef> {
      auto resolved = dataflow.resolve(sourceActor->second->actor);
      if (!resolved)
        return resolved.takeError();
      auto service =
          ::dataflow::semantics::CanonicalService::forActor(resolved->op);
      if (!service)
        return service.takeError();
      for (unsigned ordinal = 0;
           ordinal < sourceActor->second->resultPorts.size(); ++ordinal) {
        auto result = service->resultValue(resolved->op, ordinal);
        if (!result)
          return result.takeError();
        if (result->getResultNumber() == producer->ordinal)
          return sourceActor->second->resultPorts[ordinal];
      }
      return invalid("memory internal edge uses a non-service actor result");
    };
    const auto endpointForOperand =
        [&]() -> llvm::Expected<
                  ::loom::fabric::FabricMemoryEngineTemplateEndpointRef> {
      auto resolved = dataflow.resolve(sinkActor->second->actor);
      if (!resolved)
        return resolved.takeError();
      auto service =
          ::dataflow::semantics::CanonicalService::forActor(resolved->op);
      if (!service)
        return service.takeError();
      for (unsigned ordinal = 0;
           ordinal < sinkActor->second->operandPorts.size(); ++ordinal) {
        auto operand = service->argumentValue(resolved->op, ordinal);
        if (!operand)
          return operand.takeError();
        if ((*operand)->getOperandNumber() == consumer->ordinal)
          return sinkActor->second->operandPorts[ordinal];
      }
      return invalid("memory internal edge uses a non-service actor operand");
    };
    auto sourceEndpoint = endpointForResult();
    if (!sourceEndpoint)
      return sourceEndpoint.takeError();
    auto sinkEndpoint = endpointForOperand();
    if (!sinkEndpoint)
      return sinkEndpoint.takeError();
    if (connection.source != *sourceEndpoint ||
        connection.sink != *sinkEndpoint)
      return invalid(
          "memory internal connection disagrees with actor port maps");
  }
  return llvm::Error::success();
}

llvm::Expected<TechMemoryRealizationView> importMemoryRealization(
    ::mapping::MemoryRealizationOp record,
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::fabric::FabricArtifactView &fabric) {
  auto engine = decodeFabric<::loom::fabric::FabricMemoryEngineTemplateRef>(
      record.getEngine());
  if (!engine)
    return engine.takeError();
  if (llvm::Error error =
          contextual(::loom::fabric::validateFabricRef(fabric, *engine),
                     "memory engine template does not resolve"))
    return std::move(error);

  TechMemoryRealizationView result{record.getEntityId(), *engine, {}, {}, {}};
  for (Operation &child : record.getBody().front()) {
    if (auto actor = dyn_cast<::mapping::MemoryActorOp>(child)) {
      auto imported = importMemoryActor(actor, dataflow, fabric, *engine);
      if (!imported)
        return imported.takeError();
      result.actors.push_back(std::move(*imported));
      continue;
    }
    if (auto boundary = dyn_cast<::mapping::MemoryGraphBoundaryOp>(child)) {
      auto imported = importMemoryBoundary(boundary, dataflow, fabric, *engine);
      if (!imported)
        return imported.takeError();
      result.graphBoundaries.push_back(std::move(*imported));
      continue;
    }
    auto imported =
        importMemoryInternalEdge(cast<::mapping::MemoryInternalEdgeOp>(child),
                                 dataflow, fabric, *engine);
    if (!imported)
      return imported.takeError();
    result.internalEdges.push_back(std::move(*imported));
  }
  if (llvm::Error error =
          verifyTechMemoryRealizationClosure(result, dataflow, fabric))
    return std::move(error);
  return result;
}

llvm::Expected<std::vector<std::uint64_t>>
decodeComputePorts(llvm::ArrayRef<std::int64_t> ports) {
  std::vector<std::uint64_t> result;
  result.reserve(ports.size());
  for (std::int64_t port : ports) {
    if (port < 0)
      return invalid("compute port ordinal is negative");
    result.push_back(static_cast<std::uint64_t>(port));
  }
  return result;
}

llvm::Expected<TechComputeActorView> importComputeActor(
    ::mapping::ComputeActorOp record,
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::fabric::FabricArtifactView &fabric,
    const ::loom::fabric::FabricFuCapabilityTemplateRef &selectedTemplate) {
  auto actorRef = decodeDataflow<::dataflow::ActorRef>(record.getActor(),
                                                       dataflow.identity());
  if (!actorRef)
    return actorRef.takeError();
  auto actor = dataflow.resolve(*actorRef);
  if (!actor)
    return actor.takeError();
  if (actor->kind == ::dataflow::CanonicalDataflowActorKind::Memory)
    return invalid("memory actor is owned by a Compute Realization");
  auto operation = decodeFabric<::loom::fabric::FabricFuTemplateNodeRef>(
      record.getFabricOp());
  if (!operation)
    return operation.takeError();
  if (operation->node != ::loom::fabric::FabricFuNodeKind::Op ||
      operation->fu != selectedTemplate.fu)
    return invalid(
        "compute actor selects an operation outside its FU template");
  if (llvm::Error error = ::loom::fabric::validateFabricRef(fabric, *operation))
    return std::move(error);

  llvm::ArrayRef<::loom::fabric::FabricFuCapabilityTemplateRecord> inventory =
      fabric.fuCapabilityTemplates(selectedTemplate.fu);
  if (selectedTemplate.ordinal >= inventory.size() ||
      !llvm::is_contained(inventory[selectedTemplate.ordinal].activeNodes,
                          *operation))
    return invalid(
        "compute actor operation is inactive in the selected template");
  const auto *capability = fabric.resolvedFabricOpCapability(*operation);
  if (!capability)
    return invalid("compute actor operation has no resolved capability");
  auto projection =
      ::dataflow::projectRegisteredActorSchemaProjection(actor->op);
  if (!projection)
    return projection.takeError();
  auto indexBitWidth = getIndexBitWidth(actor->op);
  if (!indexBitWidth)
    return indexBitWidth.takeError();
  auto pointerLayout = pointerLayoutFor(*projection, actor->op);
  if (!pointerLayout)
    return pointerLayout.takeError();
  auto operands = decodeComputePorts(record.getOperandPorts());
  if (!operands)
    return operands.takeError();
  auto results = decodeComputePorts(record.getResultPorts());
  if (!results)
    return results.takeError();
  if (llvm::Error error = capability->admitCorrespondence(
          *projection, *indexBitWidth, *operands, *results,
          *pointerLayout ? &**pointerLayout : nullptr))
    return contextual(std::move(error),
                      "compute actor port correspondence is incompatible");
  return TechComputeActorView{*actorRef, *operation, std::move(*operands),
                              std::move(*results)};
}

llvm::Expected<TechComputeRealizationView> importComputeRealization(
    ::mapping::ComputeRealizationOp record,
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::fabric::FabricArtifactView &fabric) {
  auto selectedTemplate =
      decodeFabric<::loom::fabric::FabricFuCapabilityTemplateRef>(
          record.getCapabilityTemplate());
  if (!selectedTemplate)
    return selectedTemplate.takeError();
  if (llvm::Error error = contextual(
          ::loom::fabric::validateFabricRef(fabric, *selectedTemplate),
          "FU capability template does not resolve"))
    return std::move(error);

  TechComputeRealizationView result{
      record.getEntityId(), *selectedTemplate, {}, {}};
  std::optional<::dataflow::GraphRef> graph;
  std::set<std::uint64_t> actorIds;
  for (Operation &child : record.getBody().front()) {
    if (auto actor = dyn_cast<::mapping::ComputeActorOp>(child)) {
      auto imported =
          importComputeActor(actor, dataflow, fabric, *selectedTemplate);
      if (!imported)
        return imported.takeError();
      auto resolved = dataflow.resolve(imported->actor);
      if (!resolved)
        return resolved.takeError();
      if (graph && *graph != resolved->graph)
        return invalid("Compute Realization crosses a graph boundary");
      graph = resolved->graph;
      if (!actorIds.insert(imported->actor.entity.value()).second)
        return invalid("Compute Realization duplicates an actor");
      result.actors.push_back(std::move(*imported));
      continue;
    }

    auto boundary = cast<::mapping::ComputeBoundaryOp>(child);
    auto actorRef = decodeDataflow<::dataflow::ActorRef>(boundary.getActor(),
                                                         dataflow.identity());
    if (!actorRef)
      return actorRef.takeError();
    auto actor = dataflow.resolve(*actorRef);
    if (!actor)
      return actor.takeError();
    auto port = decodeFabric<::loom::fabric::FabricFuTemplatePortRef>(
        boundary.getFuPort());
    if (!port)
      return port.takeError();
    if (port->fu != selectedTemplate->fu)
      return invalid("compute boundary crosses its FU template owner");
    if (llvm::Error error = ::loom::fabric::validateFabricRef(fabric, *port))
      return std::move(error);
    const auto direction =
        boundary.getDirection() == ::mapping::PortDirection::Input
            ? ::loom::fabric::FabricPortDirection::Input
            : ::loom::fabric::FabricPortDirection::Output;
    if (port->direction != direction)
      return invalid("compute boundary directions disagree");
    result.boundaries.push_back(TechComputeBoundaryView{
        *actorRef, direction, boundary.getPortOrdinal(), *port});
  }
  if (llvm::Error error =
          verifyTechComputeRealizationClosure(result, dataflow, fabric))
    return std::move(error);
  return result;
}

llvm::Expected<
    std::pair<std::vector<::dataflow::GraphRef>, std::set<std::uint64_t>>>
importCovers(::mapping::TechOp root,
             const ::dataflow::CanonicalDataflowProgramView &dataflow) {
  std::vector<::dataflow::GraphRef> covers;
  std::set<std::uint64_t> coveredGraphIds;
  covers.reserve(root.getCovers().size());
  for (Attribute attribute : root.getCovers()) {
    auto graphAttribute = cast<::mapping::GraphRefAttr>(attribute);
    auto graph = decodeDataflow<::dataflow::GraphRef>(graphAttribute,
                                                      dataflow.identity());
    if (!graph)
      return graph.takeError();
    auto resolved = dataflow.resolve(*graph);
    if (!resolved)
      return llvm::joinErrors(
          invalid("TechMapping covers a graph that does not resolve"),
          resolved.takeError());
    if (!coveredGraphIds.insert(graph->entity.value()).second)
      return invalid("TechMapping covers contains a duplicate graph");
    covers.push_back(*graph);
  }
  return std::make_pair(std::move(covers), std::move(coveredGraphIds));
}

struct ImportedTechMappingView final {
  ArtifactIdentity dataflowIdentity;
  ArtifactIdentity fabricIdentity;
  std::vector<::dataflow::GraphRef> covers;
  std::vector<TechComputeRealizationView> compute;
  std::vector<TechMemoryRealizationView> memory;
};

struct PreparedTechMapping final {
  ArtifactRootReference reference;
  CanonicalSemanticBytes canonicalBytes;
};

llvm::Expected<PreparedTechMapping>
prepareTechMapping(::mapping::TechOp source) {
  auto canonicalBytes = writeCanonicalMappingAssembly(source);
  if (!canonicalBytes)
    return canonicalBytes.takeError();
  ArtifactRootReference reference{
      mappingArtifactSchema.identity.str(), mappingArtifactSchema.version,
      finalizeArtifactIdentity(mappingArtifactSchema, *canonicalBytes)};
  return PreparedTechMapping{std::move(reference), std::move(*canonicalBytes)};
}

llvm::Error publishPreparedTechMapping(const PreparedTechMapping &prepared,
                                       const ArtifactStore &store) {
  auto stored = store.put(mappingArtifactSchema, prepared.canonicalBytes);
  if (!stored)
    return stored.takeError();
  if (*stored != prepared.reference.artifact)
    return invalid("ArtifactStore returned a different Mapping identity");
  return llvm::Error::success();
}

llvm::Expected<ImportedTechMappingView>
importView(const ArtifactIdentity &mappingIdentity, ::mapping::TechOp root,
           const ::dataflow::CanonicalDataflowProgramView &dataflow,
           const ::loom::fabric::FabricArtifactView &fabric) {
  auto dataflowIdentity = decodeIdentity(root.getDataflow());
  if (!dataflowIdentity)
    return dataflowIdentity.takeError();
  auto fabricIdentity = decodeIdentity(root.getFabric());
  if (!fabricIdentity)
    return fabricIdentity.takeError();
  if (*dataflowIdentity != dataflow.identity())
    return invalid("TechMapping dataflow binding does not match its importer");
  if (*fabricIdentity != fabric.identity())
    return invalid("TechMapping Fabric binding does not match its importer");

  auto cover = importCovers(root, dataflow);
  if (!cover)
    return cover.takeError();
  std::vector<TechComputeRealizationView> compute;
  std::vector<TechMemoryRealizationView> memory;
  std::set<std::uint64_t> realizationIds;
  std::set<std::uint64_t> mappedActorIds;

  for (Operation &child : root.getBody().front()) {
    if (auto realization = dyn_cast<::mapping::ComputeRealizationOp>(child)) {
      auto imported = importComputeRealization(realization, dataflow, fabric);
      if (!imported)
        return imported.takeError();
      if (!realizationIds.insert(imported->entityId).second)
        return invalid("duplicate Mapping EntityId");
      for (const TechComputeActorView &actor : imported->actors) {
        auto resolved = dataflow.resolve(actor.actor);
        if (!resolved)
          return resolved.takeError();
        if (cover->second.find(resolved->graph.entity.value()) ==
            cover->second.end())
          return invalid("compute actor belongs to an uncovered graph");
        if (!mappedActorIds.insert(actor.actor.entity.value()).second)
          return invalid("Dataflow actor is mapped more than once");
      }
      compute.push_back(std::move(*imported));
      continue;
    }
    auto imported = importMemoryRealization(
        cast<::mapping::MemoryRealizationOp>(child), dataflow, fabric);
    if (!imported)
      return imported.takeError();
    if (!realizationIds.insert(imported->entityId).second)
      return invalid("duplicate Mapping EntityId");
    for (const TechMemoryActorView &actor : imported->actors) {
      auto resolved = dataflow.resolve(actor.actor);
      if (!resolved)
        return resolved.takeError();
      if (cover->second.find(resolved->graph.entity.value()) ==
          cover->second.end())
        return invalid("memory actor belongs to an uncovered graph");
      if (!mappedActorIds.insert(actor.actor.entity.value()).second)
        return invalid("Dataflow actor is mapped more than once");
    }
    memory.push_back(std::move(*imported));
  }

  for (const ::dataflow::CanonicalActorView &actor : dataflow.actors()) {
    if (cover->second.find(actor.graph.entity.value()) == cover->second.end())
      continue;
    if (mappedActorIds.find(actor.ref.entity.value()) == mappedActorIds.end())
      return invalid("TechMapping does not cover every actor in covers");
  }
  (void)mappingIdentity;
  return ImportedTechMappingView{*dataflowIdentity, *fabricIdentity,
                                 std::move(cover->first), std::move(compute),
                                 std::move(memory)};
}

llvm::Expected<TechMappingView>
strictImport(const ArtifactIdentity &mappingIdentity,
             const CanonicalSemanticBytes &canonicalBytes,
             const ArtifactStore &store) {
  if (finalizeArtifactIdentity(mappingArtifactSchema, canonicalBytes) !=
      mappingIdentity)
    return invalid("mapping identity does not match canonical bytes");
  auto parsed = parseTechRoot(canonicalBytes);
  if (!parsed)
    return parsed.takeError();

  auto dataflowIdentity = decodeIdentity(parsed->root.getDataflow());
  if (!dataflowIdentity)
    return dataflowIdentity.takeError();
  ArtifactRootReference dataflowReference{
      ::dataflow::canonicalDataflowSchema.identity.str(),
      ::dataflow::canonicalDataflowSchema.version, *dataflowIdentity};
  auto dataflow = ::dataflow::importCanonicalDataflow(dataflowReference, store);
  if (!dataflow)
    return dataflow.takeError();
  auto dataflowView = dataflow->view();
  if (!dataflowView)
    return dataflowView.takeError();

  auto fabricIdentity = decodeIdentity(parsed->root.getFabric());
  if (!fabricIdentity)
    return fabricIdentity.takeError();
  ArtifactRootReference fabricReference{
      ::loom::fabric::fabricArtifactSchema.identity.str(),
      ::loom::fabric::fabricArtifactSchema.version, *fabricIdentity};
  auto fabric = ::loom::fabric::importEntireFabricRoot(fabricReference, store);
  if (!fabric)
    return fabric.takeError();

  auto view = TechMappingView::import(mappingIdentity, parsed->root,
                                      *dataflowView, fabric->view());
  if (!view)
    return view.takeError();
  auto rewritten = writeCanonicalMappingAssembly(parsed->root);
  if (!rewritten)
    return rewritten.takeError();
  if (!rewritten->bytes().equals(canonicalBytes.bytes()))
    return invalid("stored mapping payload is not canonical");
  return view;
}

llvm::Expected<TechMappingView>
strictImport(const ArtifactIdentity &mappingIdentity,
             const CanonicalSemanticBytes &canonicalBytes,
             const ::dataflow::CanonicalDataflowProgramView &dataflow,
             const ::loom::fabric::FabricArtifactView &fabric) {
  if (finalizeArtifactIdentity(mappingArtifactSchema, canonicalBytes) !=
      mappingIdentity)
    return invalid("mapping identity does not match canonical bytes");
  auto parsed = parseTechRoot(canonicalBytes);
  if (!parsed)
    return parsed.takeError();
  auto view =
      TechMappingView::import(mappingIdentity, parsed->root, dataflow, fabric);
  if (!view)
    return view.takeError();
  auto rewritten = writeCanonicalMappingAssembly(parsed->root);
  if (!rewritten)
    return rewritten.takeError();
  if (!rewritten->bytes().equals(canonicalBytes.bytes()))
    return invalid("stored mapping payload is not canonical");
  return view;
}

llvm::Error requirePublishedUpstream(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::fabric::FabricArtifactView &fabric,
    const ArtifactStore &store) {
  const ArtifactRootReference dataflowReference{
      ::dataflow::canonicalDataflowSchema.identity.str(),
      ::dataflow::canonicalDataflowSchema.version, dataflow.identity()};
  auto dataflowBytes = store.get(dataflowReference);
  if (!dataflowBytes)
    return dataflowBytes.takeError();
  const ArtifactRootReference fabricReference{
      ::loom::fabric::fabricArtifactSchema.identity.str(),
      ::loom::fabric::fabricArtifactSchema.version, fabric.identity()};
  auto fabricBytes = store.get(fabricReference);
  if (!fabricBytes)
    return fabricBytes.takeError();
  return llvm::Error::success();
}

} // namespace

llvm::Error verifyTechMemoryRealizationClosure(
    const TechMemoryRealizationView &realization,
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::fabric::FabricArtifactView &fabric) {
  return verifyMemoryCorrespondenceClosure(realization, dataflow, fabric);
}

llvm::Expected<TechMappingView> TechMappingView::import(
    const ArtifactIdentity &mappingIdentity, ::mapping::TechOp root,
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::fabric::FabricArtifactView &fabric) {
  auto imported = importView(mappingIdentity, root, dataflow, fabric);
  if (!imported)
    return imported.takeError();
  return TechMappingView(
      mappingIdentity, std::move(imported->dataflowIdentity),
      std::move(imported->fabricIdentity), std::move(imported->covers),
      std::move(imported->compute), std::move(imported->memory));
}

llvm::Expected<FinalizedTechMapping>
finalizeTechMapping(::mapping::TechOp source, const ArtifactStore &store) {
  auto prepared = prepareTechMapping(source);
  if (!prepared)
    return prepared.takeError();
  auto view = strictImport(prepared->reference.artifact,
                           prepared->canonicalBytes, store);
  if (!view)
    return view.takeError();
  if (llvm::Error error = publishPreparedTechMapping(*prepared, store))
    return std::move(error);
  return FinalizedTechMapping(std::move(prepared->reference),
                              std::move(prepared->canonicalBytes),
                              std::move(*view));
}

llvm::Expected<FinalizedTechMapping>
finalizeTechMapping(::mapping::TechOp source,
                    const ::dataflow::CanonicalDataflowProgramView &dataflow,
                    const ::loom::fabric::FabricArtifactView &fabric,
                    const ArtifactStore &store) {
  if (llvm::Error error = requirePublishedUpstream(dataflow, fabric, store))
    return std::move(error);
  auto prepared = prepareTechMapping(source);
  if (!prepared)
    return prepared.takeError();
  auto view = strictImport(prepared->reference.artifact,
                           prepared->canonicalBytes, dataflow, fabric);
  if (!view)
    return view.takeError();
  if (llvm::Error error = publishPreparedTechMapping(*prepared, store))
    return std::move(error);
  return FinalizedTechMapping(std::move(prepared->reference),
                              std::move(prepared->canonicalBytes),
                              std::move(*view));
}

llvm::Expected<FinalizedTechMapping>
importTechMapping(const ArtifactRootReference &reference,
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
  return FinalizedTechMapping(reference, std::move(*canonicalBytes),
                              std::move(*view));
}

} // namespace loom::mapping
