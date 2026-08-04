#ifndef LOOM_LIB_SIMULATOR_CGRAEXECUTIONPLAN_H
#define LOOM_LIB_SIMULATOR_CGRAEXECUTIONPLAN_H

#include "CGRAResourceRuntime.h"
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
  std::uint32_t acquireRank = 0;
  std::uint32_t releaseRank = 0;
  std::optional<std::uint32_t> commitRank;
  std::optional<std::uint32_t> transitionOrdinal;
};

struct CgraFrozenExecutionPlan final {
  CgraExecutionPlanSummary summary;
  std::vector<::dataflow::GraphRef> mappedGraphs;
  std::vector<CgraPhysicalUsePlan> physicalUses;
  CgraResourceRuntimePlan resources;
};

llvm::Expected<CgraFrozenExecutionPlan> freezeCgraExecutionPlan(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::mapping::TechMappingView &tech,
    const ::loom::fabric::FabricArtifactView &fabric,
    const ::loom::mapping::SpatialMappingView &spatial);

} // namespace loom::sim::detail

#endif // LOOM_LIB_SIMULATOR_CGRAEXECUTIONPLAN_H
