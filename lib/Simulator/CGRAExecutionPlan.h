#ifndef LOOM_LIB_SIMULATOR_CGRAEXECUTIONPLAN_H
#define LOOM_LIB_SIMULATOR_CGRAEXECUTIONPLAN_H

#include "CGRAMemoryPlan.h"
#include "CGRAPhysicalActionRuntime.h"
#include "CGRATransportPlan.h"
#include "Simulator/CGRAAdmission.h"

#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Fabric/Identity/FabricRefImport.h"
#include "Mapping/Artifact/MappingArtifact.h"

#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace loom::sim::detail {

struct CgraPhysicalUsePlan final {
  std::uint64_t patternOffset = 0;
  std::uint32_t patternCount = 0;
  std::uint64_t resourceOwnerOrdinal = 0;
};

struct CgraComputeTransitionPlan final {
  std::uint32_t caseOrdinal = 0;
  std::uint64_t physicalUseOffset = 0;
  std::uint32_t physicalUseCount = 0;
};

struct CgraTemporalDispatchDomainPlan final {
  ::loom::fabric::FabricPeOccurrenceRef pe;
  std::uint32_t allocationUnit = 0;
  std::uint32_t candidateCount = 0;
  std::uint32_t resetPosition = 0;
};

struct CgraComputeActorPlan final {
  ::dataflow::ActorRef actor;
  ::dataflow::GraphRef graph;
  ::loom::fabric::FabricFuOccurrenceRef occurrence;
  ::loom::fabric::InstructionContextRef context;
  std::uint64_t transitionOffset = 0;
  std::uint32_t transitionCount = 0;
  std::optional<std::uint64_t> temporalDispatchDomain;
  std::uint32_t temporalDispatchPosition = 0;
};

struct CgraFrozenExecutionPlan final {
  CgraExecutionPlanSummary summary;
  std::vector<::dataflow::GraphRef> mappedGraphs;
  std::vector<CgraComputeActorPlan> computeActors;
  std::vector<CgraComputeTransitionPlan> computeTransitions;
  std::vector<CgraTemporalDispatchDomainPlan> temporalDispatchDomains;
  std::vector<std::uint64_t> actorTransitionPhysicalUses;
  std::vector<CgraPhysicalUsePlan> physicalUses;
  std::vector<::loom::fabric::FabricUsePatternRef> physicalUsePatterns;
  std::vector<CgraPhysicalUseClientKind> physicalUseClients;
  std::vector<CgraPhysicalUseTiming> physicalUseTimings;
  CgraResourceRuntimePlan resources;
  CgraMemoryPlan memory;
  CgraTransportPlan transport;
};

llvm::Expected<CgraFrozenExecutionPlan> freezeCgraExecutionPlan(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::mapping::TechMappingView &tech,
    const ::loom::fabric::FabricArtifactView &fabric,
    const ::loom::mapping::SpatialMappingView &spatial);

} // namespace loom::sim::detail

#endif // LOOM_LIB_SIMULATOR_CGRAEXECUTIONPLAN_H
