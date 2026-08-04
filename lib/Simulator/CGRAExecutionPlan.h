#ifndef LOOM_LIB_SIMULATOR_CGRAEXECUTIONPLAN_H
#define LOOM_LIB_SIMULATOR_CGRAEXECUTIONPLAN_H

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
  ::loom::fabric::FabricUsePatternRef reference;
  std::uint64_t resourceOwnerOrdinal = 0;
  std::uint32_t requesterOrdinal = 0;
  std::uint32_t eligibilityOrdinal = 0;
  std::optional<std::uint32_t> transitionOrdinal;
};

struct CgraComputeTransitionPlan final {
  std::uint32_t caseOrdinal = 0;
  std::uint64_t physicalUseOffset = 0;
  std::uint32_t physicalUseCount = 0;
};

struct CgraComputeActorPlan final {
  ::dataflow::ActorRef actor;
  ::dataflow::GraphRef graph;
  ::loom::fabric::FabricFuOccurrenceRef occurrence;
  ::loom::fabric::InstructionContextRef context;
  std::uint64_t transitionOffset = 0;
  std::uint32_t transitionCount = 0;
};

struct CgraFrozenExecutionPlan final {
  CgraExecutionPlanSummary summary;
  std::vector<::dataflow::GraphRef> mappedGraphs;
  std::vector<CgraComputeActorPlan> computeActors;
  std::vector<CgraComputeTransitionPlan> computeTransitions;
  std::vector<std::uint64_t> actorTransitionPhysicalUses;
  std::vector<CgraPhysicalUsePlan> physicalUses;
  std::vector<CgraPhysicalUseTiming> physicalUseTimings;
  CgraResourceRuntimePlan resources;
  CgraTransportPlan transport;
};

llvm::Expected<CgraFrozenExecutionPlan> freezeCgraExecutionPlan(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::mapping::TechMappingView &tech,
    const ::loom::fabric::FabricArtifactView &fabric,
    const ::loom::mapping::SpatialMappingView &spatial);

} // namespace loom::sim::detail

#endif // LOOM_LIB_SIMULATOR_CGRAEXECUTIONPLAN_H
