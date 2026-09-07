#include "SpatialComputeAggregate.h"

#include "SystemRunError.h"

#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Artifact/FabricSystemContracts.h"
#include "Fabric/Artifact/FabricSystemRootView.h"
#include "Hardware/Implementation/HardwareImplementation.h"
#include "Mapping/Artifact/MappingArtifact.h"
#include "Runtime/Gem5BuiltinModels.h"

#include <limits>
#include <set>
#include <variant>

namespace loom::system_run {
namespace {

/// Retired firings of Compute-kind actors over the replay's LaunchToTerminal
/// window. Other windows describe the same firings at a different boundary and
/// would double count; other payloads describe Fabric resources, not actors.
llvm::Expected<std::uint64_t>
retiredComputeFirings(const sim::SpatialSimulationExecution &replay,
                      const ArtifactRootReference &dataflowReference,
                      const ArtifactStore &artifacts) {
  auto dataflow = dataflow::importCanonicalDataflow(dataflowReference, artifacts);
  if (!dataflow)
    return dataflow.takeError();
  std::uint64_t firings = 0;
  for (const sim::ActivitySummary &summary : replay.activitySummaries) {
    if (summary.window != sim::ActivityWindow::LaunchToTerminal)
      continue;
    const auto *transitions =
        std::get_if<sim::ActorTransitionsActivity>(&summary.payload);
    if (!transitions)
      continue;
    for (const sim::ActorTransitionEntry &entry : transitions->transitions) {
      auto actor = dataflow->view().resolve(entry.actor);
      if (!actor)
        return actor.takeError();
      if (actor->kind != dataflow::CanonicalDataflowActorKind::Compute)
        continue;
      if (entry.counts.retiredFirings >
          std::numeric_limits<std::uint64_t>::max() - firings)
        return invalid("candidate retired compute firings overflow");
      firings += entry.counts.retiredFirings;
    }
  }
  return firings;
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
    const ArtifactStore &artifacts, const BlobStore &blobs) {
  if (invocations.size() != cgraReplays.size())
    return invalid("candidate compute aggregation lost a CGRA replay");
  application::ApplicationSystemComputeInputs inputs;
  // PE entity identity is Fabric-local, so a mapped compute unit is the
  // (Fabric, PE occurrence) pair. A PE shared by several SpatialMappings offers
  // its cycles once.
  std::set<std::pair<ArtifactIdentity::Storage, fabric::FabricEntityId>>
      mappedComputePes;
  std::set<std::string> launchedAccCores;
  std::set<ArtifactIdentity::Storage> visitedMappings;
  std::set<ArtifactIdentity::Storage> visitedImplementations;
  for (std::size_t ordinal = 0; ordinal != invocations.size(); ++ordinal) {
    const SpatialInvocationCase &invocation = invocations[ordinal];
    const sim::SpatialSimulationExecution *replay = cgraReplays[ordinal];
    if (!replay)
      return invalid("candidate compute aggregation lost a CGRA replay");
    auto firings =
        retiredComputeFirings(*replay, invocation.dataflow, artifacts);
    if (!firings)
      return firings.takeError();
    if (*firings > std::numeric_limits<std::uint64_t>::max() -
                       inputs.retiredComputeFirings)
      return invalid("candidate retired compute firings overflow");
    inputs.retiredComputeFirings += *firings;
    launchedAccCores.insert(invocation.accCoreReference);

    if (visitedMappings.insert(invocation.spatialMapping.artifact.bytes()).second) {
      auto mapping =
          mapping::importSpatialMapping(invocation.spatialMapping, artifacts);
      if (!mapping)
        return mapping.takeError();
      const ArtifactIdentity::Storage fabricIdentity =
          mapping->view().fabricIdentity().bytes();
      for (const mapping::SpatialComputeBindingView &binding :
           mapping->view().computeBindings())
        mappedComputePes.emplace(fabricIdentity, binding.context.pe.id());
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
  inputs.mappedComputeUnits = mappedComputePes.size();
  inputs.launchedAccCores = launchedAccCores.size();
  return inputs;
}

} // namespace loom::system_run
