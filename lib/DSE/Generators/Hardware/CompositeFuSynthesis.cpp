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

/// The acceptance result of one placed composite FU: the canonical capability
/// template it published and one witness per mined occurrence.
struct CompositeFuCoverage final {
  ::loom::fabric::FabricFuCapabilityTemplateRef capabilityTemplate;
  std::vector<FuSynthesisCoverageWitness> witnesses;
};

/// Proves `S subset-of Materialize(F)` for one placed composite FU. Canonical
/// finalization relabels FU graph nodes and capability rows, so every binding
/// is named through the design's own authored-to-canonical resolution; nothing
/// here reconstructs that relation or consults a Mapping search.
llvm::Expected<CompositeFuCoverage> projectCompositeFuCoverage(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const CompositeFuCandidate &candidate,
    const ::loom::adg::CompositeFuSpec &spec,
    const ::loom::adg::FinalizedFabricDesign &design,
    const ::loom::adg::CompositeFuPlacement &placement,
    const ::loom::fabric::FinalizedFabricRoot &module) {
  auto capabilityTarget = design.resolve(placement.capability);
  if (!capabilityTarget)
    return failure(FuReverseSynthesisFailure::CoverageNotEstablished,
                   llvm::toString(capabilityTarget.takeError()));
  if (capabilityTarget->artifact != module.view().identity())
    return failure(FuReverseSynthesisFailure::CoverageNotEstablished,
                   "composite FU capability resolved against a foreign Fabric");
  CompositeFuCoverage coverage;
  coverage.capabilityTemplate = capabilityTarget->entity;
  if (placement.nodes.size() != candidate.nodes.size())
    return failure(FuReverseSynthesisFailure::CoverageNotEstablished,
                   "placed composite FU does not hold one node per mined node");
  std::vector<::loom::fabric::FabricFuTemplateNodeRef> operationNodes;
  operationNodes.reserve(candidate.nodes.size());
  for (std::size_t node = 0; node != candidate.nodes.size(); ++node) {
    auto target = design.resolve(placement.nodes[node]);
    if (!target)
      return failure(FuReverseSynthesisFailure::CoverageNotEstablished,
                     llvm::toString(target.takeError()));
    if (target->artifact != module.view().identity() ||
        target->entity.fu != coverage.capabilityTemplate.fu)
      return failure(FuReverseSynthesisFailure::CoverageNotEstablished,
                     "a composite FU node resolved outside its own FU");
    const auto *operation =
        module.view().resolvedFabricOpCapability(target->entity);
    if (!operation)
      return failure(FuReverseSynthesisFailure::CoverageNotEstablished,
                     "a composite FU node is not an operation resource");
    if (operation->implementationFamily !=
            spec.nodes[node].implementationFamily ||
        operation->enabledOperationSchemas !=
            spec.nodes[node].enabledOperations)
      return failure(FuReverseSynthesisFailure::CoverageNotEstablished,
                     "a composite FU node lost its derived capability");
    operationNodes.push_back(target->entity);
  }

  coverage.witnesses.reserve(candidate.occurrences.size());
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
           {coverage.capabilityTemplate.fu,
            ::loom::fabric::FabricPortDirection::Input, indexed.index()}});
    for (const auto &indexed : llvm::enumerate(candidate.outputs))
      boundaries.push_back(
          {occurrence.actors[indexed.value().node],
           ::loom::fabric::FabricPortDirection::Output,
           indexed.value().portOrdinal,
           {coverage.capabilityTemplate.fu,
            ::loom::fabric::FabricPortDirection::Output, indexed.index()}});
    const ::loom::mapping::TechComputeRealizationView prospective{
        0, coverage.capabilityTemplate, actors, boundaries};
    if (llvm::Error error =
            ::loom::mapping::verifyTechComputeRealizationClosure(
                prospective, dataflow, module.view()))
      return failure(FuReverseSynthesisFailure::CoverageNotEstablished,
                     llvm::toString(std::move(error)));
    coverage.witnesses.push_back({occurrence.graph, module.view().identity(),
                                  coverage.capabilityTemplate,
                                  std::move(actors), std::move(boundaries)});
  }
  return coverage;
}

} // namespace

llvm::Expected<::loom::adg::CompositeFuSpec> deriveCompositeFuTemplate(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const CompositeFuCandidate &candidate) {
  if (candidate.nodes.empty())
    return failure(FuReverseSynthesisFailure::UnsupportedActorInventory,
                   "mined candidate has no template node");
  ::loom::adg::CompositeFuSpec spec;
  spec.nodes.reserve(candidate.nodes.size());
  for (std::size_t node = 0; node != candidate.nodes.size(); ++node) {
    auto actors = projectNodeActors(dataflow, candidate, node);
    if (!actors)
      return actors.takeError();
    auto capability =
        deriveNodeCapability(candidate.nodes[node].schema, *actors);
    if (!capability)
      return capability.takeError();
    const ::mlir::FunctionType type = candidate.nodes[node].type;
    ::loom::adg::CompositeFuNodeSpec declaration;
    declaration.implementationFamily = capability->family;
    declaration.hardwareParameters = capability->parameters;
    declaration.enabledOperations = capability->enabledSchemas;
    std::vector<std::uint32_t> inputWidths;
    std::vector<std::uint32_t> resultWidths;
    for (::mlir::Type input : type.getInputs()) {
      auto width = payloadWidth(input);
      if (!width)
        return width.takeError();
      inputWidths.push_back(*width);
      auto port = bitsPort(*width);
      if (!port)
        return port.takeError();
      declaration.inputTypes.push_back(*port);
    }
    for (::mlir::Type output : type.getResults()) {
      auto width = payloadWidth(output);
      if (!width)
        return width.takeError();
      resultWidths.push_back(*width);
      auto port = bitsPort(*width);
      if (!port)
        return port.takeError();
      declaration.outputTypes.push_back(*port);
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
    spec.nodes.push_back(std::move(declaration));
  }

  spec.internalEdges.reserve(candidate.internalEdges.size());
  for (const CompositeFuInternalEdge &edge : candidate.internalEdges)
    spec.internalEdges.push_back({edge.producerNode, edge.producerResult,
                                  edge.consumerNode, edge.consumerOperand});
  spec.inputs.reserve(candidate.inputs.size());
  for (const CompositeFuBoundaryPort &port : candidate.inputs) {
    if (port.node >= spec.nodes.size() ||
        port.portOrdinal >= spec.nodes[port.node].inputTypes.size())
      return failure(FuReverseSynthesisFailure::UnsupportedGraphTopology,
                     "mined boundary input names an unknown node port");
    spec.inputs.push_back({port.node, port.portOrdinal});
  }
  spec.outputs.reserve(candidate.outputs.size());
  for (const CompositeFuBoundaryPort &port : candidate.outputs) {
    if (port.node >= spec.nodes.size() ||
        port.portOrdinal >= spec.nodes[port.node].outputTypes.size())
      return failure(FuReverseSynthesisFailure::UnsupportedGraphTopology,
                     "mined boundary output names an unknown node port");
    spec.outputs.push_back({port.node, port.portOrdinal});
  }
  if (spec.outputs.empty())
    return failure(FuReverseSynthesisFailure::UnsupportedGraphTopology,
                   "mined template publishes no FU result");
  return spec;
}

llvm::Expected<CompositeFuTemplateArtifacts> synthesizeCompositeFuTemplate(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const CompositeFuCandidate &candidate, const ArtifactStore &store) {
  using namespace ::loom::adg;
  auto fu = deriveCompositeFuTemplate(dataflow, candidate);
  if (!fu)
    return fu.takeError();

  // One Spatial PE carrying exactly this FU. The PE and core ports are the
  // FU's own boundary widened to a nonzero payload, which is the narrowest
  // shell that proves the template materializes back to its actors.
  std::vector<PortType> outerInputs;
  outerInputs.reserve(fu->inputs.size());
  for (const CompositeFuPortSpec &port : fu->inputs) {
    auto outer = bitsPort(std::max<std::uint32_t>(
        1, fu->nodes[port.node].inputTypes[port.portOrdinal].width()));
    if (!outer)
      return outer.takeError();
    outerInputs.push_back(*outer);
  }
  std::vector<PortType> outerOutputs;
  outerOutputs.reserve(fu->outputs.size());
  for (const CompositeFuPortSpec &port : fu->outputs) {
    auto outer = bitsPort(std::max<std::uint32_t>(
        1, fu->nodes[port.node].outputTypes[port.portOrdinal].width()));
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
  auto placement = addCompositeFu(*pe, peInputs, *fu);
  if (!placement)
    return failure(FuReverseSynthesisFailure::FabricFinalizationFailed,
                   llvm::toString(placement.takeError()));
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

  auto coverage = projectCompositeFuCoverage(dataflow, candidate, *fu,
                                             *finalized, *placement, module);
  if (!coverage)
    return coverage.takeError();
  return CompositeFuTemplateArtifacts{module, coverage->capabilityTemplate,
                                      std::move(coverage->witnesses)};
}

} // namespace loom::dse
