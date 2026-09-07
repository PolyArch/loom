#ifndef LOOM_LIB_SIMULATOR_CGRAPREPAREDEXECUTIONINTERNAL_H
#define LOOM_LIB_SIMULATOR_CGRAPREPAREDEXECUTIONINTERNAL_H

#include "CGRAExecutionPlan.h"
#include "CgraTransportGraph.h"
#include "DFGSimulatorInternal.h"

#include "Fabric/Artifact/FabricArtifact.h"
#include "Mapping/Artifact/MappingArtifact.h"
#include "Mapping/Inspection/SpatialMappingInspection.h"
#include "Simulator/CGRAAdmission.h"

#include <utility>
#include <vector>

namespace loom::sim::detail {

struct PreparedCgraGraph final {
  ::dataflow::GraphRef graph;
  PreparedGraphExecution execution;
  std::vector<::dataflow::ActorRef> actors;
  CgraTransportGraph transport;
};

} // namespace loom::sim::detail

namespace loom::sim {

struct PreparedCgraExecution::Impl final {
  ::dataflow::CanonicalDataflowArtifact dataflow;
  ::loom::fabric::FinalizedFabricRoot fabric;
  ::loom::mapping::FinalizedTechMapping tech;
  ::loom::mapping::FinalizedSpatialMapping spatial;
  ::loom::mapping::SpatialMappingInspection inspection;
  detail::CgraFrozenExecutionPlan executionPlan;
  std::vector<detail::PreparedCgraGraph> graphs;

  Impl(::dataflow::CanonicalDataflowArtifact dataflow,
       ::loom::fabric::FinalizedFabricRoot fabric,
       ::loom::mapping::FinalizedTechMapping tech,
       ::loom::mapping::FinalizedSpatialMapping spatial,
       ::loom::mapping::SpatialMappingInspection inspection,
       detail::CgraFrozenExecutionPlan executionPlan,
       std::vector<detail::PreparedCgraGraph> graphs)
      : dataflow(std::move(dataflow)), fabric(std::move(fabric)),
        tech(std::move(tech)), spatial(std::move(spatial)),
        inspection(std::move(inspection)),
        executionPlan(std::move(executionPlan)), graphs(std::move(graphs)) {}
};

} // namespace loom::sim

#endif // LOOM_LIB_SIMULATOR_CGRAPREPAREDEXECUTIONINTERNAL_H
