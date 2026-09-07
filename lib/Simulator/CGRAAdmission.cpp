#include "Simulator/CGRAAdmission.h"

#include "CGRAExecutionPlan.h"
#include "CGRAPreparedExecutionInternal.h"

#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Evaluation/ArtifactImportCache.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Mapping/Artifact/MappingArtifact.h"
#include "Mapping/Inspection/SpatialMappingInspection.h"
#include "Simulator/SimulationAdmission.h"

#include "llvm/ADT/STLExtras.h"

#include "llvm/ADT/DenseMap.h"

#include <array>
#include <cstdint>
#include <limits>
#include <system_error>
#include <utility>

namespace loom::sim {
namespace {

llvm::Error invalid(llvm::Twine message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument), message);
}

} // namespace

PreparedCgraExecution::PreparedCgraExecution(std::shared_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
PreparedCgraExecution::PreparedCgraExecution(
    PreparedCgraExecution &&) noexcept = default;
PreparedCgraExecution &
PreparedCgraExecution::operator=(PreparedCgraExecution &&) noexcept = default;
PreparedCgraExecution::~PreparedCgraExecution() = default;

CgraExecutionPlanSummary PreparedCgraExecution::summary() const {
  if (!impl_)
    return {};
  return impl_->executionPlan.summary;
}

llvm::Expected<CgraExecutionOwnerReferences>
PreparedCgraExecution::ownerReferences() const {
  if (!impl_)
    return invalid("CGRA admission received a moved-from preparation");
  return CgraExecutionOwnerReferences{
      {::dataflow::canonicalDataflowSchema.identity.str(),
       ::dataflow::canonicalDataflowSchema.version, impl_->dataflow.identity()},
      impl_->fabric.reference(),
      impl_->tech.reference(),
      impl_->spatial.reference()};
}

llvm::Expected<PreparedCgraExecution>
prepareCgraExecution(const ArtifactRootReference &dataflowReference,
                     const ArtifactRootReference &fabricReference,
                     const ArtifactRootReference &spatialMappingReference,
                     const ArtifactStore &store) {
  // The preparation seals structural and physical inputs. Workload admission
  // and dynamic state are constructed separately for every execution.
  const std::array<ArtifactRootReference, 3> references{
      dataflowReference, fabricReference, spatialMappingReference};
  auto prepared = evaluation::importCachedArtifact<PreparedCgraExecution>(
      store, nullptr, references,
      [&]() -> llvm::Expected<PreparedCgraExecution> {
        // Reuse the strict Fabric import across this preparation and its
        // nested SpatialMapping and TechMapping imports.
        ::loom::fabric::FabricArtifactImportSession fabricImportSession;
        auto dataflow =
            ::dataflow::importCanonicalDataflow(dataflowReference, store);
        if (!dataflow)
          return dataflow.takeError();
        // This private owner is not cached until preparation returns. Keep
        // its verifier serial instead of starting a nested MLIR worker pool.
        dataflow->module().getContext()->disableMultithreading();
        const auto &dataflowView = dataflow->view();
        auto fabric =
            ::loom::fabric::importEntireFabricRoot(fabricReference, store);
        if (!fabric)
          return fabric.takeError();
        auto spatial = ::loom::mapping::importSpatialMapping(
            spatialMappingReference, store);
        if (!spatial)
          return spatial.takeError();
        if (spatial->view().dataflowIdentity() != dataflowReference.artifact ||
            spatial->view().fabricIdentity() != fabricReference.artifact)
          return invalid(
              "CGRA admission received a foreign Dataflow or Fabric");

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
            dataflowView, tech->view(), fabric->view(), spatial->view());
        if (!inspection)
          return inspection.takeError();
        if (inspection->summary.selectedActorCount == 0 ||
            inspection->summary.resourceUseCount == 0)
          return invalid(
              "CGRA admission requires selected actors and resources");
        auto executionPlan = detail::freezeCgraExecutionPlan(
            dataflowView, tech->view(), fabric->view(), spatial->view());
        if (!executionPlan)
          return executionPlan.takeError();

        std::vector<detail::PreparedCgraGraph> graphs;
        graphs.reserve(executionPlan->mappedGraphs.size());
        for (const ::dataflow::GraphRef &graphRef :
             executionPlan->mappedGraphs) {
          auto graphView = dataflowView.resolve(graphRef);
          if (!graphView)
            return graphView.takeError();
          auto prepared = detail::prepareGraphExecution(
              dataflow->module(),
              mlir::cast<::dataflow::GraphOp>(graphView->op));
          if (!prepared)
            return prepared.takeError();
          if (auto *failure =
                  std::get_if<detail::GraphPreparationFailure>(&*prepared)) {
            const std::string diagnostic =
                failure->diagnostics.empty()
                    ? "mapped graph has no CGRA semantic provider"
                    : failure->diagnostics.front();
            return llvm::createStringError(std::errc::not_supported, "%s",
                                           diagnostic.c_str());
          }
          detail::PreparedGraphExecution execution =
              std::move(std::get<detail::PreparedGraphExecution>(*prepared));
          llvm::DenseMap<mlir::Operation *, ::dataflow::ActorRef> references;
          references.reserve(dataflowView.actors().size());
          for (const ::dataflow::CanonicalActorView &actor :
               dataflowView.actors())
            references.try_emplace(actor.op, actor.ref);
          std::vector<::dataflow::ActorRef> actors;
          actors.reserve(execution.actorPlans.size());
          for (const detail::ActorExecutionPlan &actor : execution.actorPlans) {
            auto found = references.find(actor.operation);
            if (found == references.end())
              return invalid("prepared CGRA actor has no canonical reference");
            actors.push_back(found->second);
          }
          auto transport = detail::freezeCgraTransportGraph(
              *executionPlan, dataflowView, graphRef, execution);
          if (!transport)
            return transport.takeError();
          graphs.push_back({graphRef, std::move(execution), std::move(actors),
                            std::move(*transport)});
        }

        dataflow->module().getContext()->enableMultithreading();
        return PreparedCgraExecution(
            std::make_shared<PreparedCgraExecution::Impl>(
                std::move(*dataflow), std::move(*fabric), std::move(*tech),
                std::move(*spatial), std::move(*inspection),
                std::move(*executionPlan), std::move(graphs)));
      },
      [&](const PreparedCgraExecution &retained)
          -> llvm::Expected<std::uint64_t> {
        std::uint64_t byteCount = 0;
        const auto revalidate =
            [&](const ArtifactRootReference &reference) -> llvm::Error {
          auto bytes = store.get(reference);
          if (!bytes)
            return bytes.takeError();
          const std::uint64_t size = bytes->bytes().size();
          byteCount =
              size > std::numeric_limits<std::uint64_t>::max() - byteCount
                  ? std::numeric_limits<std::uint64_t>::max()
                  : byteCount + size;
          return llvm::Error::success();
        };
        if (llvm::Error error = revalidate(retained.impl_->tech.reference()))
          return std::move(error);
        for (const auto &dependency :
             retained.impl_->fabric.directDependencies())
          if (llvm::Error error = revalidate(dependency.root))
            return std::move(error);
        return byteCount;
      });
  if (!prepared)
    return prepared.takeError();
  return PreparedCgraExecution((*prepared)->impl_);
}

llvm::Expected<::dataflow::GraphRef> admitCgraSpatialSimulation(
    const PreparedCgraExecution &prepared,
    const CanonicalSimulationWorkload &workload,
    const CanonicalSimulationRuntimeInput &runtimeInput) {
  if (!prepared.impl_)
    return invalid("CGRA admission received a moved-from preparation");
  auto graph = admitDfgSpatialSimulation(workload, runtimeInput,
                                         prepared.impl_->dataflow.view());
  if (!graph)
    return graph.takeError();
  if (!llvm::is_contained(prepared.impl_->executionPlan.mappedGraphs, *graph))
    return invalid("CGRA workload graph has no selected physical mapping");
  return *graph;
}

} // namespace loom::sim
