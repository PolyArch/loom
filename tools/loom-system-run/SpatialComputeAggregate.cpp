#include "SpatialComputeAggregate.h"

#include "SystemRunError.h"

#include "Common/IndexWidth.h"
#include "Common/MappingDebugLog.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Dataflow/IR/OperationSchema.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Identity/FabricFuCapabilityTemplate.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Fabric/Identity/FabricRefImport.h"
#include "Fabric/Artifact/FabricSystemContracts.h"
#include "Fabric/Artifact/FabricSystemRootView.h"
#include "Hardware/Implementation/HardwareImplementation.h"
#include "Mapping/Artifact/MappingArtifact.h"
#include "Mapping/IR/MappingSchema.h"
#include "Runtime/Gem5BuiltinModels.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"

#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <tuple>
#include <variant>
#include <vector>

namespace loom::system_run {
namespace {

/// One compute class: an operation schema at one element width. Address and
/// pointer payloads are measured at the pointer width the exact program's
/// DataLayout declares; `index` at the width the program owns.
struct ComputeClassKey {
  ::dataflow::OperationSchemaId schema;
  std::uint32_t elementBits;
  bool operator<(const ComputeClassKey &other) const {
    return std::tie(schema, elementBits) <
           std::tie(other.schema, other.elementBits);
  }
};

struct ComputeClassAccumulator {
  std::uint64_t retiredElementFirings = 0;
  std::uint64_t boundRealizations = 0;
  /// The first actor seen in this class stands for it in FU admission.
  std::optional<::dataflow::CanonicalActorSchemaProjection> representative;
  unsigned indexBitWidth = 0;
  /// Lane count of the actor most recently classified into this class.
  std::uint64_t lanes = 1;
  /// Canonical bytes of every FU capability template the TechMapping realized
  /// an actor of this class with. They admit the class where no single FU
  /// operation node admits its representative, such as a fused template.
  std::set<std::vector<std::uint8_t>> realizedCapabilityTemplates;
};

struct ActorElementShape {
  std::uint32_t elementBits = 0;
  std::uint64_t lanes = 1;
};

/// Element width and lane count of one Compute actor from its first result,
/// or first operand when it produces nothing.
llvm::Expected<ActorElementShape> actorElementShape(mlir::Operation *op) {
  mlir::Type type;
  if (op->getNumResults() != 0)
    type = op->getResult(0).getType();
  else if (op->getNumOperands() != 0)
    type = op->getOperand(0).getType();
  else
    return invalid("compute actor has no typed element");
  ActorElementShape shape;
  if (auto vector = llvm::dyn_cast<mlir::VectorType>(type)) {
    for (std::int64_t extent : vector.getShape()) {
      if (extent <= 0)
        return invalid("compute actor has a non-positive vector extent");
      shape.lanes *= static_cast<std::uint64_t>(extent);
    }
    type = vector.getElementType();
  }
  if (type.isIntOrFloat()) {
    shape.elementBits = type.getIntOrFloatBitWidth();
  } else if (llvm::isa<mlir::IndexType>(type)) {
    auto width = ::loom::getIndexBitWidth(op);
    if (!width)
      return width.takeError();
    shape.elementBits = *width;
  } else {
    const llvm::TypeSize bits =
        mlir::DataLayout::closest(op).getTypeSizeInBits(type);
    if (bits.isScalable() || bits.getFixedValue() == 0)
      return invalid("compute actor element type has no fixed width");
    shape.elementBits = static_cast<std::uint32_t>(bits.getFixedValue());
  }
  if (shape.elementBits == 0)
    return invalid("compute actor element type has no width");
  return shape;
}

/// The class one Compute actor belongs to, registered in `classes` with the
/// actor as its admission representative when the class is new.
llvm::Expected<ComputeClassAccumulator *>
computeClassOf(mlir::Operation *op,
               std::map<ComputeClassKey, ComputeClassAccumulator> &classes) {
  auto shape = actorElementShape(op);
  if (!shape)
    return shape.takeError();
  auto projection = ::dataflow::projectRegisteredActorSchemaProjection(op);
  if (!projection)
    return projection.takeError();
  ComputeClassAccumulator &cls =
      classes[{projection->schema, shape->elementBits}];
  if (!cls.representative) {
    auto indexBitWidth = ::loom::getIndexBitWidth(op);
    if (!indexBitWidth)
      return indexBitWidth.takeError();
    cls.representative = std::move(*projection);
    cls.indexBitWidth = *indexBitWidth;
  }
  cls.lanes = shape->lanes;
  return &cls;
}

/// Adds the retired firings of every Compute-kind actor over the replay's
/// LaunchToTerminal window to its class. Other windows describe the same
/// firings at a different boundary and would double count; other payloads
/// describe Fabric resources, not actors.
llvm::Error accumulateRetiredComputeFirings(
    const sim::SpatialSimulationExecution &replay,
    const dataflow::CanonicalDataflowArtifact &dataflow,
    std::map<ComputeClassKey, ComputeClassAccumulator> &classes) {
  for (const sim::ActivitySummary &summary : replay.activitySummaries) {
    if (summary.window != sim::ActivityWindow::LaunchToTerminal)
      continue;
    const auto *transitions =
        std::get_if<sim::ActorTransitionsActivity>(&summary.payload);
    if (!transitions)
      continue;
    for (const sim::ActorTransitionEntry &entry : transitions->transitions) {
      auto actor = dataflow.view().resolve(entry.actor);
      if (!actor)
        return actor.takeError();
      if (actor->kind != dataflow::CanonicalDataflowActorKind::Compute)
        continue;
      auto cls = computeClassOf(actor->op, classes);
      if (!cls)
        return cls.takeError();
      const std::uint64_t firings = entry.counts.retiredFirings;
      if (firings > std::numeric_limits<std::uint64_t>::max() / (*cls)->lanes)
        return invalid("candidate retired compute firings overflow");
      const std::uint64_t elements = firings * (*cls)->lanes;
      if (elements > std::numeric_limits<std::uint64_t>::max() -
                         (*cls)->retiredElementFirings)
        return invalid("candidate retired compute firings overflow");
      (*cls)->retiredElementFirings += elements;
    }
  }
  return llvm::Error::success();
}

/// Counts the compute realizations the SpatialMapping bound, by class of each
/// actor the realization covers. The TechMapping owns the realization to
/// actor relation; the SpatialMapping selects which realizations are bound.
llvm::Error accumulateBoundRealizations(
    const mapping::FinalizedSpatialMapping &spatial,
    const dataflow::CanonicalDataflowArtifact &dataflow,
    const ArtifactStore &artifacts,
    std::map<ComputeClassKey, ComputeClassAccumulator> &classes) {
  const ArtifactRootReference techReference{
      mapping::mappingArtifactSchema.identity.str(),
      mapping::mappingArtifactSchema.version,
      spatial.view().techMappingIdentity()};
  auto tech = mapping::importTechMapping(techReference, artifacts);
  if (!tech)
    return tech.takeError();
  if (tech->view().dataflowIdentity() != dataflow.identity())
    return invalid("candidate TechMapping names a foreign Dataflow");
  std::map<std::uint64_t, const mapping::TechComputeRealizationView *>
      realizations;
  for (const mapping::TechComputeRealizationView &realization :
       tech->view().computeRealizations())
    realizations.emplace(realization.entityId, &realization);
  for (const mapping::SpatialComputeBindingView &binding :
       spatial.view().computeBindings()) {
    auto realization = realizations.find(binding.realization);
    if (realization == realizations.end())
      return invalid("candidate SpatialMapping binds an unknown realization");
    for (const mapping::TechComputeActorView &covered :
         realization->second->actors) {
      auto actor = dataflow.view().resolve(covered.actor);
      if (!actor)
        return actor.takeError();
      if (actor->kind != dataflow::CanonicalDataflowActorKind::Compute)
        continue;
      auto cls = computeClassOf(actor->op, classes);
      if (!cls)
        return cls.takeError();
      ++(*cls)->boundRealizations;
      (*cls)->realizedCapabilityTemplates.insert(
          fabric::canonicalFabricBytes(realization->second->capabilityTemplate));
    }
  }
  return llvm::Error::success();
}

struct ComputeClassCapacity {
  std::uint64_t peakIssueLanesPerCycle = 0;
  std::uint64_t placementSlots = 0;
};

/// Result lanes one FU operation node issues for a class: the payload width
/// of its first result port over the class element width, or one when the
/// node has no result port. The port is the Fabric-owned transport authority.
std::uint64_t capabilityResultLanes(
    const fabric::ResolvedFabricOpCapabilityView &capability,
    std::uint32_t elementBits) {
  for (const fabric::ResolvedFabricOpPhysicalPortView &port :
       capability.physicalPorts) {
    if (port.reference.direction != fabric::FabricPortDirection::Output)
      continue;
    if (port.payloadWidthBits < elementBits)
      return 1;
    return port.payloadWidthBits / elementBits;
  }
  return 1;
}

/// Issue lanes one FU occurrence offers a class: every operation node that
/// admits the class representative under the TechMapping's own admission
/// rule, or, when no single node admits it, one issue per capability template
/// the TechMapping realized the class with (a fused or composite template).
/// Returns the lanes and the number of issuing nodes or templates.
std::pair<std::uint64_t, std::uint64_t>
occurrenceClassIssue(const fabric::FabricArtifactView &fabric,
                     fabric::FabricFuTemplateRef definition,
                     const ComputeClassKey &key,
                     const ComputeClassAccumulator &accumulator) {
  std::uint64_t lanes = 0;
  std::uint64_t issuers = 0;
  for (const fabric::ResolvedFabricOpCapabilityView &capability :
       fabric.resolvedFabricOpCapabilities(definition)) {
    if (llvm::Error rejected = capability.admit(*accumulator.representative,
                                                accumulator.indexBitWidth)) {
      llvm::consumeError(std::move(rejected));
      continue;
    }
    lanes += capabilityResultLanes(capability, key.elementBits);
    ++issuers;
  }
  if (issuers != 0)
    return {lanes, issuers};
  for (auto [ordinal, record] :
       llvm::enumerate(fabric.fuCapabilityTemplates(definition))) {
    if (!accumulator.realizedCapabilityTemplates.count(
            fabric::canonicalFabricBytes(fabric::FabricFuCapabilityTemplateRef{
                definition, static_cast<fabric::FabricOrdinal>(ordinal)})))
      continue;
    std::uint64_t templateLanes = 1;
    for (const fabric::FabricFuTemplateNodeRef &node : record.activeNodes) {
      if (node.node != fabric::FabricFuNodeKind::Op)
        continue;
      if (const auto *capability = fabric.resolvedFabricOpCapability(node)) {
        templateLanes = capabilityResultLanes(*capability, key.elementBits);
        break;
      }
    }
    lanes += templateLanes;
    ++issuers;
  }
  return {lanes, issuers};
}

/// The speed-of-light inventory of one Fabric for every observed class: each
/// FU occurrence is counted once per issuing node or realized template. A
/// Temporal PE issues once per cycle however many resident instruction
/// contexts share its FU; those contexts are its placement slots.
llvm::Expected<std::map<ComputeClassKey, ComputeClassCapacity>>
fabricComputeCapacity(
    const fabric::FabricArtifactView &fabric,
    const std::map<ComputeClassKey, ComputeClassAccumulator> &classes) {
  std::map<ComputeClassKey, ComputeClassCapacity> capacity;
  for (const auto &[key, accumulator] : classes)
    capacity.emplace(key, ComputeClassCapacity{});
  for (fabric::FabricFuOccurrenceRef occurrence : fabric.fuOccurrences()) {
    auto definition = fabric.fuTemplateOf(occurrence);
    auto pe = fabric.parentPeOf(occurrence);
    if (!definition || !pe)
      return invalid("candidate Fabric FU occurrence has no template or PE");
    auto schedule = fabric.peSchedule(*pe);
    if (!schedule)
      return invalid("candidate Fabric PE has no schedule");
    std::uint64_t slotsPerIssuer = 1;
    if (*schedule == ::fabric::Schedule::Temporal) {
      slotsPerIssuer = fabric.peResidentContextCount(*pe);
      if (slotsPerIssuer == 0)
        return invalid("candidate Temporal PE has no resident context");
    }
    for (auto &[key, entry] : capacity) {
      const ComputeClassAccumulator &accumulator = classes.at(key);
      if (!accumulator.representative)
        continue;
      const auto [lanes, issuers] =
          occurrenceClassIssue(fabric, *definition, key, accumulator);
      if (issuers == 0)
        continue;
      if (lanes > std::numeric_limits<std::uint64_t>::max() -
                      entry.peakIssueLanesPerCycle ||
          issuers > std::numeric_limits<std::uint64_t>::max() / slotsPerIssuer ||
          issuers * slotsPerIssuer >
              std::numeric_limits<std::uint64_t>::max() - entry.placementSlots)
        return invalid("candidate Fabric compute capacity overflows");
      entry.peakIssueLanesPerCycle += lanes;
      entry.placementSlots += issuers * slotsPerIssuer;
    }
  }
  return capacity;
}

/// The SpatialCore clock period expressed in gem5 ticks. The Fabric clock
/// contract is the sole authority; no reference frequency is assumed.
llvm::Expected<std::uint64_t>
spatialCoreReferenceCycleTicks(const ArtifactRootReference &implementationReference,
                               const ArtifactStore &artifacts,
                               const BlobStore &blobs) {
  auto implementation = hardware::importHardwareImplementation(
      implementationReference, artifacts, blobs);

  if (!implementation)
    return implementation.takeError();
  auto fabric = fabric::importEntireFabricRoot(
      implementation->implementation().fabric(), artifacts);
  if (!fabric)
    return fabric.takeError();
  auto system = fabric::requireSystemRoot(fabric->view());
  if (!system)
    return system.takeError();
  auto domain = system->effectiveHardwareDomain(
      implementation->implementation().subject(),
      fabric::FabricClockResetKind::Clock);
  if (!domain)
    return domain.takeError();
  const auto *record = system->hardwareDomainContract(*domain);
  const auto *clock =
      record ? std::get_if<fabric::ClockDomainContractRecord>(&record->contract())
             : nullptr;
  if (!clock)
    return invalid("SpatialCore occurrence has no exact Clock domain contract");
  if (clock->periodFs() % runtime::gem5TickFemtoseconds != 0)
    return invalid("SpatialCore clock period is not a whole number of gem5 ticks");
  const std::uint64_t ticks = clock->periodFs() / runtime::gem5TickFemtoseconds;
  if (ticks == 0)
    return invalid("SpatialCore clock period is shorter than one gem5 tick");
  return ticks;
}

} // namespace

llvm::Expected<application::ApplicationSystemComputeInputs>
aggregateSpatialComputeInputs(
    llvm::ArrayRef<SpatialInvocationCase> invocations,
    llvm::ArrayRef<const sim::SpatialSimulationExecution *> cgraReplays,
    const std::optional<sim::SystemComputationInterval> &computation,
    const ArtifactStore &artifacts, const BlobStore &blobs) {
  if (invocations.size() != cgraReplays.size())
    return invalid("candidate compute aggregation lost a CGRA replay");
  application::ApplicationSystemComputeInputs inputs;
  std::map<ComputeClassKey, ComputeClassAccumulator> classes;
  std::set<std::string> launchedAccCores;
  std::set<ArtifactIdentity::Storage> visitedMappings;
  std::set<ArtifactIdentity::Storage> visitedImplementations;
  std::optional<ArtifactRootReference> fabricReference;
  std::map<ArtifactIdentity::Storage,
           std::unique_ptr<dataflow::CanonicalDataflowArtifact>>
      dataflows;
  const auto dataflowOf = [&](const ArtifactRootReference &reference)
      -> llvm::Expected<const dataflow::CanonicalDataflowArtifact *> {
    auto found = dataflows.find(reference.artifact.bytes());
    if (found != dataflows.end())
      return found->second.get();
    auto imported = dataflow::importCanonicalDataflow(reference, artifacts);
    if (!imported)
      return imported.takeError();
    auto owned = std::make_unique<dataflow::CanonicalDataflowArtifact>(
        std::move(*imported));
    const auto *view = owned.get();
    dataflows.emplace(reference.artifact.bytes(), std::move(owned));
    return view;
  };
  for (std::size_t ordinal = 0; ordinal != invocations.size(); ++ordinal) {
    const SpatialInvocationCase &invocation = invocations[ordinal];
    // The boundary device refuses unfinished launches at either source marker,
    // so completion membership selects whole invocations, excluding warmup.
    if (!computation ||
        invocation.systemCgraCompletionTick < computation->beginTick ||
        invocation.systemCgraCompletionTick > computation->endTick)
      continue;
    const sim::SpatialSimulationExecution *replay = cgraReplays[ordinal];
    if (!replay)
      return invalid("candidate compute aggregation lost a CGRA replay");
    auto dataflow = dataflowOf(invocation.dataflow);
    if (!dataflow)
      return dataflow.takeError();
    if (llvm::Error error =
            accumulateRetiredComputeFirings(*replay, **dataflow, classes))
      return std::move(error);
    launchedAccCores.insert(invocation.accCoreReference);
    if (!fabricReference)
      fabricReference = invocation.fabric;
    else if (fabricReference->artifact != invocation.fabric.artifact)
      return invalid("candidate SpatialCores disagree on their Fabric");
    if (visitedMappings.insert(invocation.spatialMapping.artifact.bytes()).second) {
      auto mapping =
          mapping::importSpatialMapping(invocation.spatialMapping, artifacts);
      if (!mapping)
        return mapping.takeError();
      if (llvm::Error error = accumulateBoundRealizations(*mapping, **dataflow,
                                                          artifacts, classes))
        return std::move(error);
    }
    if (visitedImplementations
            .insert(invocation.hardwareImplementation.artifact.bytes())
            .second) {
      auto ticks = spatialCoreReferenceCycleTicks(
          invocation.hardwareImplementation, artifacts, blobs);
      if (!ticks)
        return ticks.takeError();
      if (inputs.referenceCycleTicks != 0 && inputs.referenceCycleTicks != *ticks)
        return invalid("candidate SpatialCores disagree on their reference cycle");
      inputs.referenceCycleTicks = *ticks;
    }
  }
  std::map<ComputeClassKey, ComputeClassCapacity> capacity;
  if (fabricReference) {
    auto fabric = fabric::importEntireFabricRoot(*fabricReference, artifacts);
    if (!fabric)
      return fabric.takeError();
    auto measured = fabricComputeCapacity(fabric->view(), classes);
    if (!measured)
      return measured.takeError();
    capacity = std::move(*measured);
  }
  inputs.classes.reserve(classes.size());
  for (const auto &[key, accumulator] : classes) {
    const ComputeClassCapacity &bound = capacity[key];
    if (accumulator.retiredElementFirings != 0 &&
        bound.peakIssueLanesPerCycle == 0)
      loom::mapping_debug::emit(
          loom::mapping_debug::Level::Summary,
          loom::mapping_debug::Stage::Deployment,
          loom::mapping_debug::Event::MappingFailure,
          [&](llvm::json::Object &fields) {
            fields["operation"] = "compute_class_without_admitting_fu";
            fields["schema"] = ::dataflow::operationSchemaSpelling(key.schema);
            fields["element_bits"] = static_cast<int64_t>(key.elementBits);
            fields["retired_element_firings"] =
                static_cast<int64_t>(accumulator.retiredElementFirings);
            fields["realized_capability_templates"] = static_cast<int64_t>(
                accumulator.realizedCapabilityTemplates.size());
          });
    inputs.classes.push_back({key.schema, key.elementBits,
                              accumulator.retiredElementFirings,
                              bound.peakIssueLanesPerCycle,
                              bound.placementSlots,
                              accumulator.boundRealizations});
  }
  inputs.launchedAccCores = launchedAccCores.size();
  return inputs;
}
} // namespace loom::system_run
