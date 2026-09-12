//===- CompositeFuSynthesis.cpp - mined composite FU templates ----------===//
//
// Turns one mined common subgraph into Fabric capability: one `fabric.fu`
// whose operation resources are the shape's nodes in its induced topology, one
// capability-template record, and one coverage witness per occurrence proving
// `S subset-of Materialize(F)`. Capability derivation, FU authoring, and
// realization-closure verification all stay with their existing owners.
//
//===----------------------------------------------------------------------===//

#include "DSE/CompositeFuMining.h"

#include "Fabric/IR/OperationResourceContract.h"
#include "Mapping/Artifact/MappingArtifact.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace loom::dse {
namespace {

llvm::Error failure(FuReverseSynthesisFailure kind,
                    const llvm::Twine &message) {
  return llvm::make_error<FuReverseSynthesisError>(kind, message.str());
}

llvm::Expected<std::uint32_t> payloadWidth(::mlir::Type type) {
  std::string diagnostic;
  ::mlir::FailureOr<unsigned> width =
      ::fabric::getSemanticPayloadWidth(type, diagnostic);
  if (::mlir::failed(width))
    return failure(FuReverseSynthesisFailure::UnsupportedActorProjection,
                   diagnostic);
  return static_cast<std::uint32_t>(*width);
}

llvm::Expected<::loom::adg::PortType> bitsPort(std::uint32_t width) {
  auto type = ::loom::adg::PortType::bits(width);
  if (!type)
    return failure(FuReverseSynthesisFailure::FabricFinalizationFailed,
                   llvm::toString(type.takeError()));
  return std::move(*type);
}

/// The exact actor projections bound to one node across every occurrence.
llvm::Expected<std::vector<::dataflow::CanonicalActorSchemaProjection>>
projectNodeActors(const ::dataflow::CanonicalDataflowProgramView &dataflow,
                  const CompositeFuCandidate &candidate, std::size_t node) {
  std::vector<::dataflow::CanonicalActorSchemaProjection> projections;
  projections.reserve(candidate.occurrences.size());
  for (const CompositeFuOccurrence &occurrence : candidate.occurrences) {
    if (node >= occurrence.actors.size())
      return failure(FuReverseSynthesisFailure::UnsupportedActorInventory,
                     "mined occurrence does not bind every template node");
    auto resolved = dataflow.resolve(occurrence.actors[node]);
    if (!resolved)
      return failure(FuReverseSynthesisFailure::InvalidGraphReference,
                     llvm::toString(resolved.takeError()));
    auto projection =
        ::dataflow::projectRegisteredActorSchemaProjection(resolved->op);
    if (!projection)
      return failure(FuReverseSynthesisFailure::UnsupportedActorProjection,
                     llvm::toString(projection.takeError()));
    if (projection->schema != candidate.nodes[node].schema ||
        projection->type != candidate.nodes[node].type)
      return failure(FuReverseSynthesisFailure::UnsupportedActorProjection,
                     "mined occurrence actor does not match its node type");
    projections.push_back(std::move(*projection));
  }
  if (projections.empty())
    return failure(FuReverseSynthesisFailure::EmptyGraphSet,
                   "mined candidate has no occurrence");
  return projections;
}

/// The least registered family that owns the node schema and whose canonical
/// capability derivation admits every occurrence's actor. Family choice stays
/// with the derivation owner; this only walks the registered candidates in a
/// deterministic order.
llvm::Expected<::fabric::CanonicalImplementationCapability> deriveNodeCapability(
    ::dataflow::OperationSchemaId schema,
    llvm::ArrayRef<::dataflow::CanonicalActorSchemaProjection> actors) {
  llvm::SmallVector<::fabric::ImplementationFamilyId, 2> families =
      ::fabric::implementationFamiliesFor(schema);
  llvm::sort(families, [](::fabric::ImplementationFamilyId left,
                          ::fabric::ImplementationFamilyId right) {
    return static_cast<std::uint32_t>(left) < static_cast<std::uint32_t>(right);
  });
  std::string diagnostic = "no registered implementation family owns the "
                           "mined node schema";
  for (::fabric::ImplementationFamilyId family : families) {
    auto capability =
        ::fabric::deriveCanonicalImplementationCapability(family, actors);
    if (capability)
      return std::move(*capability);
    diagnostic = llvm::toString(capability.takeError());
  }
  return failure(FuReverseSynthesisFailure::CapabilityDerivationRejected,
                 diagnostic);
}

/// Authoring order of the template nodes. A mined shape whose internal
/// relation contains a cycle is a recurrence; authoring it needs an explicit
/// FU backedge, which is outside this synthesis profile.
llvm::Expected<std::vector<std::size_t>>
topologicalNodeOrder(const CompositeFuCandidate &candidate) {
  std::vector<std::size_t> pending(candidate.nodes.size(), 0);
  std::vector<std::vector<std::size_t>> successors(candidate.nodes.size());
  for (const CompositeFuInternalEdge &edge : candidate.internalEdges) {
    if (edge.producerNode >= candidate.nodes.size() ||
        edge.consumerNode >= candidate.nodes.size())
      return failure(FuReverseSynthesisFailure::UnsupportedGraphTopology,
                     "mined internal edge names an unknown template node");
    ++pending[edge.consumerNode];
    successors[edge.producerNode].push_back(edge.consumerNode);
  }
  std::vector<std::size_t> ready;
  for (std::size_t node = 0; node != pending.size(); ++node)
    if (pending[node] == 0)
      ready.push_back(node);
  std::vector<std::size_t> order;
  while (!ready.empty()) {
    const std::size_t node = ready.front();
    ready.erase(ready.begin());
    order.push_back(node);
    for (std::size_t successor : successors[node])
      if (--pending[successor] == 0)
        ready.push_back(successor);
  }
  if (order.size() != candidate.nodes.size())
    return failure(FuReverseSynthesisFailure::UnsupportedGraphTopology,
                   "mined shape contains a recurrence and needs an explicit "
                   "FU backedge");
  return order;
}

} // namespace

llvm::Expected<CompositeFuTemplate> deriveCompositeFuTemplate(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const CompositeFuCandidate &candidate) {
  if (candidate.nodes.empty())
    return failure(FuReverseSynthesisFailure::UnsupportedActorInventory,
                   "mined candidate has no template node");
  CompositeFuTemplate result;
  result.operations.reserve(candidate.nodes.size());
  for (std::size_t node = 0; node != candidate.nodes.size(); ++node) {
    auto actors = projectNodeActors(dataflow, candidate, node);
    if (!actors)
      return actors.takeError();
    auto capability = deriveNodeCapability(candidate.nodes[node].schema,
                                           *actors);
    if (!capability)
      return capability.takeError();
    const ::mlir::FunctionType type = candidate.nodes[node].type;
    std::vector<std::uint32_t> inputWidths;
    std::vector<std::uint32_t> resultWidths;
    std::vector<::loom::adg::PortType> outputTypes;
    for (::mlir::Type input : type.getInputs()) {
      auto width = payloadWidth(input);
      if (!width)
        return width.takeError();
      inputWidths.push_back(*width);
    }
    for (::mlir::Type output : type.getResults()) {
      auto width = payloadWidth(output);
      if (!width)
        return width.takeError();
      resultWidths.push_back(*width);
      auto port = bitsPort(*width);
      if (!port)
        return port.takeError();
      outputTypes.push_back(*port);
    }
    std::vector<std::uint64_t> operands(type.getNumInputs());
    std::vector<std::uint64_t> results(type.getNumResults());
    for (std::size_t ordinal = 0; ordinal != operands.size(); ++ordinal)
      operands[ordinal] = ordinal;
    for (std::size_t ordinal = 0; ordinal != results.size(); ++ordinal)
      results[ordinal] = ordinal;
    for (const ::dataflow::CanonicalActorSchemaProjection &actor : *actors)
      if (llvm::Error error =
              ::fabric::verifyImplementationFamilyPortCorrespondence(
                  capability->family, capability->parameters, actor, operands,
                  results, inputWidths, resultWidths))
        return failure(FuReverseSynthesisFailure::CapabilityDerivationRejected,
                       llvm::toString(std::move(error)));
    result.operations.push_back(
        CompositeFuOperation{std::move(*capability), std::move(outputTypes)});
  }

  for (const CompositeFuBoundaryPort &port : candidate.inputs) {
    if (port.node >= candidate.nodes.size() ||
        port.portOrdinal >= candidate.nodes[port.node].type.getNumInputs())
      return failure(FuReverseSynthesisFailure::UnsupportedGraphTopology,
                     "mined boundary input names an unknown node port");
    auto width =
        payloadWidth(candidate.nodes[port.node].type.getInput(port.portOrdinal));
    if (!width)
      return width.takeError();
    auto type = bitsPort(*width);
    if (!type)
      return type.takeError();
    result.inputTypes.push_back(*type);
  }
  for (const CompositeFuBoundaryPort &port : candidate.outputs) {
    if (port.node >= candidate.nodes.size() ||
        port.portOrdinal >= candidate.nodes[port.node].type.getNumResults())
      return failure(FuReverseSynthesisFailure::UnsupportedGraphTopology,
                     "mined boundary output names an unknown node port");
    auto width = payloadWidth(
        candidate.nodes[port.node].type.getResult(port.portOrdinal));
    if (!width)
      return width.takeError();
    auto type = bitsPort(*width);
    if (!type)
      return type.takeError();
    result.outputTypes.push_back(*type);
  }
  if (result.outputTypes.empty())
    return failure(FuReverseSynthesisFailure::UnsupportedGraphTopology,
                   "mined template publishes no FU result");
  return result;
}

llvm::Expected<CompositeFuAuthoring>
authorCompositeFu(::loom::adg::PeBuilder &pe,
                  llvm::ArrayRef<::loom::adg::PeValue> inputs,
                  const CompositeFuCandidate &candidate,
                  const CompositeFuTemplate &fu) {
  using namespace ::loom::adg;
  auto order = topologicalNodeOrder(candidate);
  if (!order)
    return order.takeError();
  if (fu.operations.size() != candidate.nodes.size())
    return failure(FuReverseSynthesisFailure::CapabilityDerivationRejected,
                   "mined template does not describe every node");

  auto builder = pe.addFu(inputs, FuSpec{fu.inputTypes, fu.outputTypes});
  if (!builder)
    return failure(FuReverseSynthesisFailure::FabricFinalizationFailed,
                   llvm::toString(builder.takeError()));
  std::vector<FuValue> boundary;
  boundary.reserve(fu.inputTypes.size());
  for (std::size_t ordinal = 0; ordinal != fu.inputTypes.size(); ++ordinal) {
    auto value = builder->input(ordinal);
    if (!value)
      return failure(FuReverseSynthesisFailure::FabricFinalizationFailed,
                     llvm::toString(value.takeError()));
    boundary.push_back(*value);
  }

  std::map<std::pair<std::uint32_t, std::uint64_t>, std::size_t> boundaryInput;
  for (const auto &indexed : llvm::enumerate(candidate.inputs))
    boundaryInput.emplace(std::make_pair(indexed.value().node,
                                         indexed.value().portOrdinal),
                          indexed.index());
  std::map<std::pair<std::uint32_t, std::uint64_t>,
           std::pair<std::uint32_t, std::uint64_t>>
      internalSource;
  for (const CompositeFuInternalEdge &edge : candidate.internalEdges)
    internalSource.emplace(
        std::make_pair(edge.consumerNode, edge.consumerOperand),
        std::make_pair(edge.producerNode, edge.producerResult));

  std::vector<std::optional<FuNode>> nodes(candidate.nodes.size());
  for (std::size_t node : *order) {
    const ::mlir::FunctionType type = candidate.nodes[node].type;
    std::vector<FuValue> operands;
    operands.reserve(type.getNumInputs());
    for (std::uint64_t ordinal = 0; ordinal != type.getNumInputs(); ++ordinal) {
      const auto key =
          std::make_pair(static_cast<std::uint32_t>(node), ordinal);
      const auto source = internalSource.find(key);
      if (source != internalSource.end()) {
        const std::optional<FuNode> &producer = nodes[source->second.first];
        if (!producer)
          return failure(FuReverseSynthesisFailure::UnsupportedGraphTopology,
                         "mined template authors a node before its producer");
        auto value = producer->output(source->second.second);
        if (!value)
          return failure(FuReverseSynthesisFailure::FabricFinalizationFailed,
                         llvm::toString(value.takeError()));
        operands.push_back(*value);
        continue;
      }
      const auto port = boundaryInput.find(key);
      if (port == boundaryInput.end())
        return failure(FuReverseSynthesisFailure::UnsupportedGraphTopology,
                       "mined node operand has neither an internal source nor "
                       "a boundary port");
      operands.push_back(boundary[port->second]);
    }
    const CompositeFuOperation &operation = fu.operations[node];
    auto authored = builder->addOperation(
        operands,
        OperationCapabilitySpec{operation.capability.family,
                                operation.capability.parameters,
                                operation.capability.enabledSchemas,
                                operation.outputTypes,
                                ::fabric::oneCycleElasticOperationResourceContract()});
    if (!authored)
      return failure(FuReverseSynthesisFailure::FabricFinalizationFailed,
                     llvm::toString(authored.takeError()));
    nodes[node] = *authored;
  }

  CompositeFuAuthoring authoring;
  authoring.nodes.reserve(nodes.size());
  for (const std::optional<FuNode> &node : nodes) {
    if (!node)
      return failure(FuReverseSynthesisFailure::FabricFinalizationFailed,
                     "mined template left a node unauthored");
    authoring.nodes.push_back(*node);
  }
  auto capability = builder->addCapabilityTemplateWithHandle(
      FuCapabilityTemplateSpec{authoring.nodes, {}});
  if (!capability)
    return failure(FuReverseSynthesisFailure::FabricFinalizationFailed,
                   llvm::toString(capability.takeError()));
  authoring.capability = *capability;

  std::vector<FuValue> outputs;
  outputs.reserve(candidate.outputs.size());
  for (const CompositeFuBoundaryPort &port : candidate.outputs) {
    auto value = nodes[port.node]->output(port.portOrdinal);
    if (!value)
      return failure(FuReverseSynthesisFailure::FabricFinalizationFailed,
                     llvm::toString(value.takeError()));
    outputs.push_back(*value);
  }
  if (llvm::Error error = builder->close(outputs))
    return failure(FuReverseSynthesisFailure::FabricFinalizationFailed,
                   llvm::toString(std::move(error)));
  return authoring;
}

llvm::Expected<CompositeFuTemplateArtifacts> synthesizeCompositeFuTemplate(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const CompositeFuCandidate &candidate, const ArtifactStore &store) {
  using namespace ::loom::adg;
  auto fu = deriveCompositeFuTemplate(dataflow, candidate);
  if (!fu)
    return fu.takeError();

  std::vector<PortType> outerInputs;
  outerInputs.reserve(fu->inputTypes.size());
  for (const PortType &type : fu->inputTypes) {
    auto outer = bitsPort(std::max<std::uint32_t>(1, type.width()));
    if (!outer)
      return outer.takeError();
    outerInputs.push_back(*outer);
  }
  std::vector<PortType> outerOutputs;
  outerOutputs.reserve(fu->outputTypes.size());
  for (const PortType &type : fu->outputTypes) {
    auto outer = bitsPort(std::max<std::uint32_t>(1, type.width()));
    if (!outer)
      return outer.takeError();
    outerOutputs.push_back(*outer);
  }

  DesignBuilder design(store);
  auto spatial = design.createSpatialCore("composite-fu-template", outerInputs,
                                          outerOutputs);
  if (!spatial)
    return failure(FuReverseSynthesisFailure::FabricFinalizationFailed,
                   llvm::toString(spatial.takeError()));
  std::vector<SpatialValue> spatialInputs;
  spatialInputs.reserve(outerInputs.size());
  for (std::size_t ordinal = 0; ordinal != outerInputs.size(); ++ordinal) {
    auto input = spatial->input(ordinal);
    if (!input)
      return failure(FuReverseSynthesisFailure::FabricFinalizationFailed,
                     llvm::toString(input.takeError()));
    spatialInputs.push_back(*input);
  }
  auto pe = spatial->addPe(spatialInputs,
                           PeSpec::spatial(outerInputs, outerOutputs));
  if (!pe)
    return failure(FuReverseSynthesisFailure::FabricFinalizationFailed,
                   llvm::toString(pe.takeError()));
  std::vector<PeValue> peInputs;
  peInputs.reserve(outerInputs.size());
  for (std::size_t ordinal = 0; ordinal != outerInputs.size(); ++ordinal) {
    auto input = pe->input(ordinal);
    if (!input)
      return failure(FuReverseSynthesisFailure::FabricFinalizationFailed,
                     llvm::toString(input.takeError()));
    peInputs.push_back(*input);
  }
  auto authoring = authorCompositeFu(*pe, peInputs, candidate, *fu);
  if (!authoring)
    return authoring.takeError();
  if (llvm::Error error = pe->close())
    return failure(FuReverseSynthesisFailure::FabricFinalizationFailed,
                   llvm::toString(std::move(error)));
  std::vector<SpatialValue> outputs;
  outputs.reserve(outerOutputs.size());
  for (std::size_t ordinal = 0; ordinal != outerOutputs.size(); ++ordinal) {
    auto output = pe->output(ordinal);
    if (!output)
      return failure(FuReverseSynthesisFailure::FabricFinalizationFailed,
                     llvm::toString(output.takeError()));
    outputs.push_back(*output);
  }
  if (llvm::Error error = spatial->close(outputs))
    return failure(FuReverseSynthesisFailure::FabricFinalizationFailed,
                   llvm::toString(std::move(error)));
  auto finalized = std::move(design).finalize();
  if (!finalized)
    return failure(FuReverseSynthesisFailure::FabricFinalizationFailed,
                   llvm::toString(finalized.takeError()));
  if (finalized->roots().size() != 1)
    return failure(FuReverseSynthesisFailure::FabricFinalizationFailed,
                   "composite FU synthesis did not publish exactly one Fabric "
                   "root");
  const ::loom::fabric::FinalizedFabricRoot &module = finalized->roots().front();

  // Canonical finalization relabels FU graph nodes and capability rows, so the
  // authored order is translated through the design's own correspondence
  // rather than assumed. Nothing here reconstructs that relation.
  auto capabilityTarget = finalized->resolve(authoring->capability);
  if (!capabilityTarget)
    return failure(FuReverseSynthesisFailure::CoverageNotEstablished,
                   llvm::toString(capabilityTarget.takeError()));
  if (capabilityTarget->artifact != module.view().identity())
    return failure(FuReverseSynthesisFailure::CoverageNotEstablished,
                   "composite FU capability resolved against a foreign Fabric");
  const ::loom::fabric::FabricFuCapabilityTemplateRef capabilityTemplate =
      capabilityTarget->entity;
  std::vector<::loom::fabric::FabricFuTemplateNodeRef> operationNodes;
  operationNodes.reserve(candidate.nodes.size());
  for (std::size_t node = 0; node != candidate.nodes.size(); ++node) {
    auto target = finalized->resolve(authoring->nodes[node]);
    if (!target)
      return failure(FuReverseSynthesisFailure::CoverageNotEstablished,
                     llvm::toString(target.takeError()));
    if (target->artifact != module.view().identity() ||
        target->entity.fu != capabilityTemplate.fu)
      return failure(FuReverseSynthesisFailure::CoverageNotEstablished,
                     "a composite FU node resolved outside its own FU");
    const auto *operation =
        module.view().resolvedFabricOpCapability(target->entity);
    if (!operation)
      return failure(FuReverseSynthesisFailure::CoverageNotEstablished,
                     "a composite FU node is not an operation resource");
    if (operation->implementationFamily !=
            fu->operations[node].capability.family ||
        operation->enabledOperationSchemas !=
            fu->operations[node].capability.enabledSchemas)
      return failure(FuReverseSynthesisFailure::CoverageNotEstablished,
                     "a composite FU node lost its derived capability");
    operationNodes.push_back(target->entity);
  }

  std::vector<FuSynthesisCoverageWitness> coverage;
  coverage.reserve(candidate.occurrences.size());
  for (const CompositeFuOccurrence &occurrence : candidate.occurrences) {
    std::vector<::loom::mapping::TechComputeActorView> actors;
    actors.reserve(candidate.nodes.size());
    for (std::size_t node = 0; node != candidate.nodes.size(); ++node) {
      const ::mlir::FunctionType type = candidate.nodes[node].type;
      std::vector<std::uint64_t> operandPorts(type.getNumInputs());
      std::vector<std::uint64_t> resultPorts(type.getNumResults());
      for (std::size_t ordinal = 0; ordinal != operandPorts.size(); ++ordinal)
        operandPorts[ordinal] = ordinal;
      for (std::size_t ordinal = 0; ordinal != resultPorts.size(); ++ordinal)
        resultPorts[ordinal] = ordinal;
      actors.push_back({occurrence.actors[node], operationNodes[node],
                        std::move(operandPorts), std::move(resultPorts)});
    }
    std::vector<::loom::mapping::TechComputeBoundaryView> boundaries;
    boundaries.reserve(candidate.inputs.size() + candidate.outputs.size());
    for (const auto &indexed : llvm::enumerate(candidate.inputs))
      boundaries.push_back(
          {occurrence.actors[indexed.value().node],
           ::loom::fabric::FabricPortDirection::Input,
           indexed.value().portOrdinal,
           {capabilityTemplate.fu,
            ::loom::fabric::FabricPortDirection::Input, indexed.index()}});
    for (const auto &indexed : llvm::enumerate(candidate.outputs))
      boundaries.push_back(
          {occurrence.actors[indexed.value().node],
           ::loom::fabric::FabricPortDirection::Output,
           indexed.value().portOrdinal,
           {capabilityTemplate.fu,
            ::loom::fabric::FabricPortDirection::Output, indexed.index()}});
    const ::loom::mapping::TechComputeRealizationView prospective{
        0, capabilityTemplate, actors, boundaries};
    if (llvm::Error error =
            ::loom::mapping::verifyTechComputeRealizationClosure(
                prospective, dataflow, module.view()))
      return failure(FuReverseSynthesisFailure::CoverageNotEstablished,
                     llvm::toString(std::move(error)));
    coverage.push_back({occurrence.graph, module.view().identity(),
                        capabilityTemplate, std::move(actors),
                        std::move(boundaries)});
  }

  return CompositeFuTemplateArtifacts{module, capabilityTemplate,
                                      std::move(coverage)};
}

} // namespace loom::dse
