#include "Mapping/Artifact/MappingArtifact.h"

#include "Dataflow/IR/DataflowOps.h"
#include "Dataflow/IR/DataflowReferenceCodec.h"
#include "Fabric/IR/OperationResourceContract.h"
#include "Fabric/Identity/FabricFuCapabilityTemplate.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <map>
#include <set>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

namespace loom::mapping {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "mapping_artifact_invalid: " + message);
}

struct ActorPortKey final {
  std::uint64_t actor = 0;
  ::loom::fabric::FabricPortDirection direction =
      ::loom::fabric::FabricPortDirection::Input;
  std::uint64_t ordinal = 0;

  friend bool operator==(ActorPortKey lhs, ActorPortKey rhs) {
    return std::tie(lhs.actor, lhs.direction, lhs.ordinal) ==
           std::tie(rhs.actor, rhs.direction, rhs.ordinal);
  }
  friend bool operator<(ActorPortKey lhs, ActorPortKey rhs) {
    return std::tie(lhs.actor, lhs.direction, lhs.ordinal) <
           std::tie(rhs.actor, rhs.direction, rhs.ordinal);
  }
};

struct TemplateEndpointKey final {
  bool boundary = false;
  ::loom::fabric::FabricFuNodeKind nodeKind =
      ::loom::fabric::FabricFuNodeKind::Op;
  std::uint64_t fu = 0;
  std::uint64_t nodeOrdinal = 0;
  ::loom::fabric::FabricPortDirection direction =
      ::loom::fabric::FabricPortDirection::Input;
  std::uint64_t portOrdinal = 0;

  friend bool operator==(TemplateEndpointKey lhs, TemplateEndpointKey rhs) {
    return std::tie(lhs.boundary, lhs.nodeKind, lhs.fu, lhs.nodeOrdinal,
                    lhs.direction, lhs.portOrdinal) ==
           std::tie(rhs.boundary, rhs.nodeKind, rhs.fu, rhs.nodeOrdinal,
                    rhs.direction, rhs.portOrdinal);
  }
  friend bool operator<(TemplateEndpointKey lhs, TemplateEndpointKey rhs) {
    return std::tie(lhs.boundary, lhs.nodeKind, lhs.fu, lhs.nodeOrdinal,
                    lhs.direction, lhs.portOrdinal) <
           std::tie(rhs.boundary, rhs.nodeKind, rhs.fu, rhs.nodeOrdinal,
                    rhs.direction, rhs.portOrdinal);
  }
};

using TemplateConnection = std::pair<TemplateEndpointKey, TemplateEndpointKey>;

TemplateEndpointKey endpointKey(
    const ::loom::fabric::FabricFuCapabilityTemplateEndpointRef &endpoint) {
  if (const auto *boundary =
          std::get_if<::loom::fabric::FabricFuTemplatePortRef>(
              &endpoint.payload))
    return {true,
            ::loom::fabric::FabricFuNodeKind::Op,
            boundary->fu.id(),
            0,
            boundary->direction,
            boundary->ordinal};
  const auto &port =
      std::get<::loom::fabric::FabricFuNodePortRef>(endpoint.payload);
  return {false,          port.node.node, port.node.fu.id(), port.node.ordinal,
          port.direction, port.ordinal};
}

TemplateEndpointKey
endpointKey(const ::loom::fabric::FabricFuTemplatePortRef &port) {
  return {true,           ::loom::fabric::FabricFuNodeKind::Op,
          port.fu.id(),   0,
          port.direction, port.ordinal};
}

TemplateEndpointKey actorEndpoint(const TechComputeActorView &actor,
                                  ::loom::fabric::FabricPortDirection direction,
                                  std::uint64_t softwareOrdinal) {
  const auto &ports = direction == ::loom::fabric::FabricPortDirection::Input
                          ? actor.operandPorts
                          : actor.resultPorts;
  return {false,
          actor.fabricOperation.node,
          actor.fabricOperation.fu.id(),
          actor.fabricOperation.ordinal,
          direction,
          ports[softwareOrdinal]};
}

bool actorIsInside(
    const ::dataflow::CanonicalGraphProducerEndpointRef &producer,
    const std::map<std::uint64_t, const TechComputeActorView *> &actors) {
  const auto *result = std::get_if<::dataflow::ActorTokenResultRef>(&producer);
  return result && actors.count(result->actor.entity.value());
}

bool actorIsInside(
    const ::dataflow::CanonicalGraphConsumerEndpointRef &consumer,
    const std::map<std::uint64_t, const TechComputeActorView *> &actors) {
  const auto *operand =
      std::get_if<::dataflow::ActorTokenOperandRef>(&consumer);
  return operand && actors.count(operand->actor.entity.value());
}

llvm::Expected<std::map<ActorPortKey, ::loom::fabric::FabricFuTemplatePortRef>>
verifyBoundaryDomain(
    const TechComputeRealizationView &realization,
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const std::map<std::uint64_t, const TechComputeActorView *> &actors) {
  std::map<ActorPortKey, ::loom::fabric::FabricFuTemplatePortRef> boundaries;
  for (const TechComputeBoundaryView &boundary : realization.boundaries) {
    auto selected = actors.find(boundary.actor.entity.value());
    if (selected == actors.end())
      return invalid("compute boundary names an actor outside its realization");
    auto resolved = dataflow.resolve(boundary.actor);
    if (!resolved)
      return resolved.takeError();
    const std::uint64_t arity =
        boundary.direction == ::loom::fabric::FabricPortDirection::Input
            ? resolved->op->getNumOperands()
            : resolved->op->getNumResults();
    if (boundary.portOrdinal >= arity)
      return invalid("compute boundary software port is out of range");
    ActorPortKey key{boundary.actor.entity.value(), boundary.direction,
                     boundary.portOrdinal};
    if (!boundaries.emplace(key, boundary.fabricPort).second)
      return invalid("compute boundary duplicates a software actor port");
  }

  std::set<ActorPortKey> expected;
  for (const auto &[id, actor] : actors) {
    auto resolved = dataflow.resolve(actor->actor);
    if (!resolved)
      return resolved.takeError();
    for (std::uint64_t ordinal = 0; ordinal < actor->operandPorts.size();
         ++ordinal) {
      ::dataflow::CanonicalGraphConsumerEndpointRef consumer =
          ::dataflow::ActorTokenOperandRef{actor->actor, ordinal};
      auto producer = dataflow.graphProducer(consumer);
      if (!producer)
        return producer.takeError();
      if (!actorIsInside(*producer, actors))
        expected.insert(ActorPortKey{
            id, ::loom::fabric::FabricPortDirection::Input, ordinal});
    }
    for (std::uint64_t ordinal = 0; ordinal < actor->resultPorts.size();
         ++ordinal) {
      ::dataflow::CanonicalGraphProducerEndpointRef producer =
          ::dataflow::ActorTokenResultRef{actor->actor, ordinal};
      auto consumers = dataflow.graphConsumers(producer);
      if (!consumers)
        return consumers.takeError();
      if (llvm::any_of(*consumers, [&](const auto &consumer) {
            return !actorIsInside(consumer, actors);
          }))
        expected.insert(ActorPortKey{
            id, ::loom::fabric::FabricPortDirection::Output, ordinal});
    }
  }

  std::set<ActorPortKey> actual;
  for (const auto &[key, port] : boundaries) {
    (void)port;
    actual.insert(key);
  }
  if (actual != expected)
    return invalid("compute FU-boundary correspondence is incomplete");
  return boundaries;
}

llvm::Error verifyBoundaryProducerIdentity(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const std::map<std::uint64_t, const TechComputeActorView *> &actors,
    const std::map<ActorPortKey, ::loom::fabric::FabricFuTemplatePortRef>
        &boundaries) {
  std::map<TemplateEndpointKey, std::vector<std::uint8_t>> producerByInput;
  for (const auto &[software, physical] : boundaries) {
    if (software.direction != ::loom::fabric::FabricPortDirection::Input)
      continue;
    const TechComputeActorView *actor = actors.at(software.actor);
    const ::dataflow::CanonicalGraphConsumerEndpointRef consumer =
        ::dataflow::ActorTokenOperandRef{actor->actor, software.ordinal};
    auto producer = dataflow.graphProducer(consumer);
    if (!producer)
      return producer.takeError();
    auto identity =
        ::dataflow::encodeDataflowReference(dataflow.identity(), *producer);
    if (!identity)
      return identity.takeError();
    auto [found, inserted] =
        producerByInput.emplace(endpointKey(physical), *identity);
    if (!inserted && found->second != *identity)
      return invalid("one FU boundary input represents distinct Dataflow "
                     "producers");
  }
  return llvm::Error::success();
}

llvm::Expected<std::set<TemplateConnection>> projectTemplateConnections(
    const ::loom::fabric::FabricFuCapabilityTemplateRecord &record,
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const std::map<std::uint64_t, const TechComputeActorView *> &actors,
    const std::map<ActorPortKey, ::loom::fabric::FabricFuTemplatePortRef>
        &boundaries) {
  auto projected =
      ::loom::fabric::projectFabricFuCapabilityTemplateTerminalEdges(record);
  if (!projected)
    return projected.takeError();
  std::set<TemplateEndpointKey> selectedSources;
  std::set<TemplateEndpointKey> selectedSinks;
  std::set<TemplateEndpointKey> deadResultSources;
  for (const auto &[id, actor] : actors) {
    (void)id;
    for (std::uint64_t ordinal = 0; ordinal < actor->operandPorts.size();
         ++ordinal)
      selectedSinks.insert(actorEndpoint(
          *actor, ::loom::fabric::FabricPortDirection::Input, ordinal));
    for (std::uint64_t ordinal = 0; ordinal < actor->resultPorts.size();
         ++ordinal) {
      const TemplateEndpointKey source = actorEndpoint(
          *actor, ::loom::fabric::FabricPortDirection::Output, ordinal);
      selectedSources.insert(source);
      auto consumers =
          dataflow.graphConsumers(::dataflow::CanonicalGraphProducerEndpointRef{
              ::dataflow::ActorTokenResultRef{actor->actor, ordinal}});
      if (!consumers)
        return consumers.takeError();
      if (consumers->empty())
        deadResultSources.insert(source);
    }
  }
  std::set<TemplateEndpointKey> selectedBoundarySources;
  std::set<TemplateEndpointKey> selectedBoundarySinks;
  for (const auto &[software, physical] : boundaries) {
    auto &selected =
        software.direction == ::loom::fabric::FabricPortDirection::Input
            ? selectedBoundarySources
            : selectedBoundarySinks;
    selected.insert(endpointKey(physical));
  }
  std::set<TemplateConnection> connections;
  for (const auto &edge : *projected) {
    const TemplateEndpointKey source = endpointKey(edge.source);
    const TemplateEndpointKey sink = endpointKey(edge.destination);
    const bool deadResultDisposition =
        !source.boundary && sink.boundary && deadResultSources.count(source);
    if ((source.boundary && !selectedBoundarySources.count(source)) ||
        (!source.boundary && !selectedSources.count(source)) ||
        (sink.boundary && !selectedBoundarySinks.count(sink) &&
         !deadResultDisposition) ||
        (!sink.boundary && !selectedSinks.count(sink)))
      continue;
    connections.emplace(source, sink);
  }
  return connections;
}

llvm::Expected<std::set<TemplateConnection>> expectedConnections(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const std::map<std::uint64_t, const TechComputeActorView *> &actors,
    const std::map<ActorPortKey, ::loom::fabric::FabricFuTemplatePortRef>
        &boundaries) {
  std::set<TemplateConnection> expected;
  for (const auto &[id, actor] : actors) {
    for (std::uint64_t ordinal = 0; ordinal < actor->operandPorts.size();
         ++ordinal) {
      ::dataflow::CanonicalGraphConsumerEndpointRef consumer =
          ::dataflow::ActorTokenOperandRef{actor->actor, ordinal};
      auto producer = dataflow.graphProducer(consumer);
      if (!producer)
        return producer.takeError();
      TemplateEndpointKey source;
      if (const auto *result =
              std::get_if<::dataflow::ActorTokenResultRef>(&*producer);
          result && actors.count(result->actor.entity.value())) {
        const auto *producerActor = actors.at(result->actor.entity.value());
        if (result->ordinal >= producerActor->resultPorts.size())
          return invalid(
              "internal software edge uses an unmapped actor result");
        source = actorEndpoint(*producerActor,
                               ::loom::fabric::FabricPortDirection::Output,
                               result->ordinal);
      } else {
        source = endpointKey(boundaries.at(ActorPortKey{
            id, ::loom::fabric::FabricPortDirection::Input, ordinal}));
      }
      expected.emplace(source,
                       actorEndpoint(*actor,
                                     ::loom::fabric::FabricPortDirection::Input,
                                     ordinal));
    }

    for (std::uint64_t ordinal = 0; ordinal < actor->resultPorts.size();
         ++ordinal) {
      ::dataflow::CanonicalGraphProducerEndpointRef producer =
          ::dataflow::ActorTokenResultRef{actor->actor, ordinal};
      auto consumers = dataflow.graphConsumers(producer);
      if (!consumers)
        return consumers.takeError();
      if (!llvm::any_of(*consumers, [&](const auto &consumer) {
            return !actorIsInside(consumer, actors);
          }))
        continue;
      expected.emplace(
          actorEndpoint(*actor, ::loom::fabric::FabricPortDirection::Output,
                        ordinal),
          endpointKey(boundaries.at(ActorPortKey{
              id, ::loom::fabric::FabricPortDirection::Output, ordinal})));
    }
  }
  return expected;
}

llvm::Error addDeadResultDispositions(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const std::map<std::uint64_t, const TechComputeActorView *> &actors,
    const std::set<TemplateConnection> &actual,
    std::set<TemplateConnection> &expected) {
  for (const auto &[id, actor] : actors) {
    (void)id;
    for (std::uint64_t ordinal = 0; ordinal < actor->resultPorts.size();
         ++ordinal) {
      const ::dataflow::CanonicalGraphProducerEndpointRef producer =
          ::dataflow::ActorTokenResultRef{actor->actor, ordinal};
      auto consumers = dataflow.graphConsumers(producer);
      if (!consumers)
        return consumers.takeError();
      if (!consumers->empty())
        continue;

      const TemplateEndpointKey source = actorEndpoint(
          *actor, ::loom::fabric::FabricPortDirection::Output, ordinal);
      auto disposition = actual.end();
      for (auto connection = actual.begin(); connection != actual.end();
           ++connection) {
        if (!(connection->first == source))
          continue;
        if (disposition != actual.end())
          return invalid("dead actor result has multiple physical "
                         "dispositions");
        disposition = connection;
      }
      if (disposition == actual.end() || !disposition->second.boundary ||
          disposition->second.direction !=
              ::loom::fabric::FabricPortDirection::Output)
        return invalid("dead actor result lacks one exact FU output discard "
                       "path");
      expected.insert(*disposition);
    }
  }
  return llvm::Error::success();
}

} // namespace

llvm::Expected<bool> deriveTechComputeActivityDefinednessAdmission(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    ::dataflow::ActorRef actor,
    const ::loom::fabric::ResolvedFabricOpCapabilityView &capability) {
  auto resolved = dataflow.resolve(actor);
  if (!resolved)
    return resolved.takeError();
  const auto operandIsAlwaysDefined =
      [&](unsigned ordinal) -> llvm::Expected<bool> {
    auto producer = dataflow.graphProducer(
        ::dataflow::ActorTokenOperandRef{actor, ordinal});
    if (!producer)
      return producer.takeError();
    auto fact = dataflow.activityDefinedness(*producer);
    if (!fact)
      return fact.takeError();
    return *fact == ::dataflow::ActivityDefinedness::AlwaysDefined;
  };
  const auto orderedContractAdmits =
      [&](::dataflow::OperationSchemaId schema) -> llvm::Expected<bool> {
    const auto *parameters = std::get_if<::fabric::FixedVectorAdapterParams>(
        &capability.parameterizedCapability);
    if (!parameters)
      return false;
    auto maximumLaneCount =
        ::fabric::maximumFixedVectorAdapterLaneCount(*parameters);
    if (!maximumLaneCount)
      return maximumLaneCount.takeError();
    return ::fabric::isOrderedCardinalityOperationResourceContract(
        capability.resourceStateAndTimingContract, schema, *maximumLaneCount);
  };

  switch (capability.implementationFamily) {
  case ::fabric::ImplementationFamilyId::FixedVectorParallelize: {
    auto adapter = llvm::dyn_cast<::dataflow::ParallelizeOp>(resolved->op);
    if (!adapter)
      return false;
    auto contract = orderedContractAdmits(
        ::dataflow::OperationSchemaId::DataflowParallelize);
    if (!contract || !*contract)
      return contract;
    return operandIsAlwaysDefined(
        adapter.getScalarPhaseMutable().getOperandNumber());
  }
  case ::fabric::ImplementationFamilyId::FixedVectorSerialize: {
    auto adapter = llvm::dyn_cast<::dataflow::SerializeOp>(resolved->op);
    if (!adapter)
      return false;
    auto contract =
        orderedContractAdmits(::dataflow::OperationSchemaId::DataflowSerialize);
    if (!contract || !*contract)
      return contract;
    auto mask =
        operandIsAlwaysDefined(adapter.getMaskMutable().getOperandNumber());
    if (!mask || !*mask)
      return mask;
    return operandIsAlwaysDefined(
        adapter.getGroupPhaseMutable().getOperandNumber());
  }
  default:
    return true;
  }
}

llvm::Error verifyTechComputeRealizationClosure(
    const TechComputeRealizationView &realization,
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::fabric::FabricArtifactView &fabric) {
  llvm::ArrayRef<::loom::fabric::FabricFuCapabilityTemplateRecord> inventory =
      fabric.fuCapabilityTemplates(realization.capabilityTemplate.fu);
  if (realization.capabilityTemplate.ordinal >= inventory.size())
    return invalid("compute capability template does not resolve");
  const auto &record = inventory[realization.capabilityTemplate.ordinal];

  std::set<TemplateEndpointKey> activeOperations;
  for (const auto &node : record.activeNodes)
    if (node.node == ::loom::fabric::FabricFuNodeKind::Op)
      activeOperations.insert({false, node.node, node.fu.id(), node.ordinal,
                               ::loom::fabric::FabricPortDirection::Input, 0});

  std::map<std::uint64_t, const TechComputeActorView *> actors;
  std::set<TemplateEndpointKey> mappedOperations;
  for (const TechComputeActorView &actor : realization.actors) {
    if (!actors.emplace(actor.actor.entity.value(), &actor).second)
      return invalid("compute realization duplicates an actor");
    mappedOperations.insert({false, actor.fabricOperation.node,
                             actor.fabricOperation.fu.id(),
                             actor.fabricOperation.ordinal,
                             ::loom::fabric::FabricPortDirection::Input, 0});
  }
  if (actors.empty() || mappedOperations != activeOperations)
    return invalid(
        "compute actor-to-operation correspondence is not bijective");

  auto boundaries = verifyBoundaryDomain(realization, dataflow, actors);
  if (!boundaries)
    return boundaries.takeError();
  if (llvm::Error error =
          verifyBoundaryProducerIdentity(dataflow, actors, *boundaries))
    return error;
  auto actual =
      projectTemplateConnections(record, dataflow, actors, *boundaries);
  if (!actual)
    return actual.takeError();
  auto expected = expectedConnections(dataflow, actors, *boundaries);
  if (!expected)
    return expected.takeError();
  if (llvm::Error error =
          addDeadResultDispositions(dataflow, actors, *actual, *expected))
    return error;
  if (*actual != *expected)
    return invalid("selected FU template topology does not exactly realize "
                   "software edges");
  return llvm::Error::success();
}

} // namespace loom::mapping
