#include "Simulator/CGRAAdmission.h"

#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Mapping/Artifact/MappingArtifact.h"
#include "Mapping/Inspection/SpatialMappingInspection.h"
#include "Simulator/SimulationAdmission.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

#include <cstdint>
#include <system_error>
#include <utility>
#include <vector>

namespace loom::sim {
namespace {

llvm::Error invalid(llvm::Twine message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument), message);
}

template <typename Realization>
llvm::Expected<std::uint64_t>
realizationGraph(const Realization &realization,
                 const ::dataflow::CanonicalDataflowProgramView &dataflow) {
  if (realization.actors.empty())
    return invalid("CGRA admission found an empty Tech realization");
  std::optional<std::uint64_t> graph;
  for (const auto &actor : realization.actors) {
    auto resolved = dataflow.resolve(actor.actor);
    if (!resolved)
      return resolved.takeError();
    const std::uint64_t current = resolved->graph.entity.value();
    if (graph && *graph != current)
      return invalid("CGRA admission found a cross-graph Tech realization");
    graph = current;
  }
  return *graph;
}

llvm::Expected<std::vector<::dataflow::GraphRef>>
deriveMappedGraphs(const ::dataflow::CanonicalDataflowProgramView &dataflow,
                   const ::loom::mapping::TechMappingView &tech,
                   const ::loom::mapping::SpatialMappingView &spatial) {
  llvm::DenseMap<std::uint64_t, std::uint64_t> computeGraphs;
  computeGraphs.reserve(tech.computeRealizations().size());
  for (const auto &realization : tech.computeRealizations()) {
    auto graph = realizationGraph(realization, dataflow);
    if (!graph)
      return graph.takeError();
    if (!computeGraphs.try_emplace(realization.entityId, *graph).second)
      return invalid("CGRA admission found duplicate compute realizations");
  }

  llvm::DenseMap<std::uint64_t, std::uint64_t> memoryGraphs;
  memoryGraphs.reserve(tech.memoryRealizations().size());
  for (const auto &realization : tech.memoryRealizations()) {
    auto graph = realizationGraph(realization, dataflow);
    if (!graph)
      return graph.takeError();
    if (!memoryGraphs.try_emplace(realization.entityId, *graph).second)
      return invalid("CGRA admission found duplicate memory realizations");
  }

  llvm::DenseSet<std::uint64_t> selectedGraphs;
  selectedGraphs.reserve(tech.covers().size());
  for (const auto &binding : spatial.computeBindings()) {
    auto graph = computeGraphs.find(binding.realization);
    if (graph == computeGraphs.end())
      return invalid("CGRA admission found an unknown compute realization");
    selectedGraphs.insert(graph->second);
  }
  for (const auto &binding : spatial.memoryEngineBindings()) {
    auto graph = memoryGraphs.find(binding.realization);
    if (graph == memoryGraphs.end())
      return invalid("CGRA admission found an unknown memory realization");
    selectedGraphs.insert(graph->second);
  }

  std::vector<::dataflow::GraphRef> result;
  result.reserve(tech.covers().size());
  for (::dataflow::GraphRef graph : tech.covers()) {
    if (!selectedGraphs.contains(graph.entity.value()))
      return invalid("CGRA admission found a covered graph without a selected "
                     "physical realization");
    result.push_back(graph);
  }
  if (result.empty())
    return invalid("CGRA admission requires a nonempty covered graph set");
  return result;
}

} // namespace

struct PreparedCgraExecution::Impl final {
  ::dataflow::CanonicalDataflowArtifact dataflow;
  ::dataflow::CanonicalDataflowProgramView dataflowView;
  ::loom::fabric::FinalizedFabricRoot fabric;
  ::loom::mapping::FinalizedTechMapping tech;
  ::loom::mapping::FinalizedSpatialMapping spatial;
  ::loom::mapping::SpatialMappingInspection inspection;
  std::vector<::dataflow::GraphRef> mappedGraphs;

  Impl(::dataflow::CanonicalDataflowArtifact dataflow,
       ::dataflow::CanonicalDataflowProgramView dataflowView,
       ::loom::fabric::FinalizedFabricRoot fabric,
       ::loom::mapping::FinalizedTechMapping tech,
       ::loom::mapping::FinalizedSpatialMapping spatial,
       ::loom::mapping::SpatialMappingInspection inspection,
       std::vector<::dataflow::GraphRef> mappedGraphs)
      : dataflow(std::move(dataflow)), dataflowView(std::move(dataflowView)),
        fabric(std::move(fabric)), tech(std::move(tech)),
        spatial(std::move(spatial)), inspection(std::move(inspection)),
        mappedGraphs(std::move(mappedGraphs)) {}
};

PreparedCgraExecution::PreparedCgraExecution(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
PreparedCgraExecution::PreparedCgraExecution(
    PreparedCgraExecution &&) noexcept = default;
PreparedCgraExecution &
PreparedCgraExecution::operator=(PreparedCgraExecution &&) noexcept = default;
PreparedCgraExecution::~PreparedCgraExecution() = default;

llvm::Expected<PreparedCgraExecution>
prepareCgraExecution(const ArtifactRootReference &dataflowReference,
                     const ArtifactRootReference &fabricReference,
                     const ArtifactRootReference &spatialMappingReference,
                     const ArtifactStore &store) {
  auto dataflow = ::dataflow::importCanonicalDataflow(dataflowReference, store);
  if (!dataflow)
    return dataflow.takeError();
  auto dataflowView = dataflow->view();
  if (!dataflowView)
    return dataflowView.takeError();
  auto fabric = ::loom::fabric::importEntireFabricRoot(fabricReference, store);
  if (!fabric)
    return fabric.takeError();
  auto spatial =
      ::loom::mapping::importSpatialMapping(spatialMappingReference, store);
  if (!spatial)
    return spatial.takeError();
  if (spatial->view().dataflowIdentity() != dataflowReference.artifact ||
      spatial->view().fabricIdentity() != fabricReference.artifact)
    return invalid("CGRA admission received a foreign Dataflow or Fabric");

  const ArtifactRootReference techReference{
      ::loom::mapping::mappingArtifactSchema.identity.str(),
      ::loom::mapping::mappingArtifactSchema.version,
      spatial->view().techMappingIdentity()};
  auto tech = ::loom::mapping::importTechMapping(techReference, store);
  if (!tech)
    return tech.takeError();
  if (tech->view().dataflowIdentity() != dataflowReference.artifact ||
      tech->view().fabricIdentity() != fabricReference.artifact)
    return invalid("CGRA admission found a foreign TechMapping owner");

  auto inspection = ::loom::mapping::inspectSpatialMapping(
      *dataflowView, tech->view(), fabric->view(), spatial->view());
  if (!inspection)
    return inspection.takeError();
  if (inspection->summary.selectedActorCount == 0 ||
      inspection->summary.resourceUseCount == 0)
    return invalid("CGRA admission requires selected actors and resources");
  auto mappedGraphs =
      deriveMappedGraphs(*dataflowView, tech->view(), spatial->view());
  if (!mappedGraphs)
    return mappedGraphs.takeError();

  return PreparedCgraExecution(std::make_unique<PreparedCgraExecution::Impl>(
      std::move(*dataflow), std::move(*dataflowView), std::move(*fabric),
      std::move(*tech), std::move(*spatial), std::move(*inspection),
      std::move(*mappedGraphs)));
}

llvm::Expected<::dataflow::GraphRef> admitCgraSpatialSimulation(
    const PreparedCgraExecution &prepared,
    const CanonicalSimulationWorkload &workload,
    const CanonicalSimulationRuntimeInput &runtimeInput) {
  if (!prepared.impl_)
    return invalid("CGRA admission received a moved-from preparation");
  auto graph = admitDfgSpatialSimulation(workload, runtimeInput,
                                         prepared.impl_->dataflowView);
  if (!graph)
    return graph.takeError();
  if (!llvm::is_contained(prepared.impl_->mappedGraphs, *graph))
    return invalid("CGRA workload graph has no selected physical mapping");
  return *graph;
}

} // namespace loom::sim
