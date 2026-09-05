#include "JointDesignExplorationFixture.h"

#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "Config/ResolvedConfig.h"
#include "DSE/JointHardwareReopen.h"
#include "DSE/ResourceTimeFrontier.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Evaluation/Evidence.h"
#include "Evaluation/ModelParameter.h"
#include "Evaluation/Models/CanonicalDataflowFabricAnalytic.h"
#include "Fabric/Artifact/FabricSystemRootView.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Frontend/IR/LoomOps.h"
#include "Mapping/Artifact/SystemMappingArtifact.h"
#include "Mapping/Artifact/SystemMappingExecutionProjection.h"
#include "PnR/System/SystemMappingMigration.h"
#include "ResourceTimeAdjacentMappingSelection.h"
#include "Simulator/SimulationArtifacts.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <optional>
#include <set>
#include <string>
#include <system_error>
#include <tuple>
#include <utility>
#include <variant>

namespace loom::dse::joint_test {
namespace {

[[noreturn]] void fail(const llvm::Twine &message) {
  llvm::errs() << "joint design exploration anchor failed: " << message << '\n';
  std::exit(EXIT_FAILURE);
}

template <typename T> T take(llvm::Expected<T> value) {
  if (!value)
    fail(llvm::toString(value.takeError()));
  return std::move(*value);
}

std::string key(llvm::ArrayRef<std::uint8_t> bytes) {
  return std::string(reinterpret_cast<const char *>(bytes.data()),
                     bytes.size());
}

} // namespace

TemporaryDirectory::TemporaryDirectory() {
  if (std::error_code error =
          llvm::sys::fs::createUniqueDirectory("loom-joint-design", path_))
    fail("cannot create test directory: " + error.message());
}

TemporaryDirectory::~TemporaryDirectory() {
  llvm::sys::fs::remove_directories(path_);
}

llvm::StringRef TemporaryDirectory::path() const { return path_; }

mlir::MLIRContext makeContext() {
  mlir::DialectRegistry registry;
  registry
      .insert<dataflow::DataflowDialect, mlir::arith::ArithDialect,
              mlir::DLTIDialect, mlir::func::FuncDialect,
              mlir::LLVM::LLVMDialect, loom::LoomDialect>();
  return mlir::MLIRContext(registry, mlir::MLIRContext::Threading::DISABLED);
}

dataflow::CanonicalDataflowArtifact buildDataflow(mlir::MLIRContext &context,
                                                  std::int32_t constant) {
  const std::string source = R"mlir(
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 64>>} {
  dataflow.graph private @sync(%start: none, %value: i32) -> i32
      attributes {input_segments = array<i32: 1, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %result:2 = dataflow.sync %start, %value
        : (none, i32) -> (none, i32)
    dataflow.graph.return values(%result#1 : i32) streams() memories()
        complete(%result#0 : none)
  }
  dataflow.thread private @worker domain(#dataflow.thread_domain<dense>)(
      %value: i32) ctrl (%ctrl: none) iv (%i: index) {
    %result, %done = dataflow.graph.launch @sync deps(%ctrl)
        values(%value) stream_inputs() memories() stream_outputs()
        : (none, i32) -> (i32, none)
    dataflow.thread.yield %done : none
  }
  llvm.func internal @host() {
    %value = arith.constant )mlir" +
                             std::to_string(constant) + R"mlir( : i32
    %extent = arith.constant 4 : index
    %thread = dataflow.thread.launch @worker(%value) grid(%extent)
        : (i32) -> !dataflow.thread_token
    llvm.return
  }
}

)mlir";
  auto module = mlir::parseSourceString<mlir::ModuleOp>(source, &context);
  if (!module)
    fail("cannot parse Dataflow fixture");
  return take(dataflow::finalizeCanonicalDataflow(*module));
}

ArtifactRootReference
publishApplicationWorkload(const dataflow::CanonicalDataflowArtifact &artifact,
                           const ArtifactStore &store) {
  auto view = take(artifact.view());
  if (view.rootThreadLaunches().size() != 1 ||
      view.staticGraphLaunches().size() != 1)
    fail("application fixture does not have one rooted graph launch");
  dataflow::RootedGraphLaunchRef launch{view.rootThreadLaunches().front().ref,
                                        view.staticGraphLaunches().front().ref};
  sim::SpatialSimulationWorkload draft{launch};
  auto logicalDomain =
      take(view.projectRootThreadLogicalDomain(launch.rootThreadLaunch));
  draft.denseCoordinates.assign(logicalDomain.coordinateRank, 0);
  auto shapes = take(sim::projectSpatialSimulationBoundaryShapes(view, launch));
  draft.valueInputPlan.assign(shapes.valueInputs.size(),
                              sim::RuntimeValueInput{});
  auto workload = take(sim::finalizeSimulationWorkload(draft, view));
  return take(sim::publishSimulationWorkload(workload, store));
}

ArtifactRootReference
publishApplicationRuntimeInput(const ArtifactRootReference &workload,
                               std::int32_t value, const ArtifactStore &store) {
  auto imported = take(sim::importSpatialSimulationWorkload(workload, store));
  auto view = take(imported.dataflow.view());
  const auto *spatial = imported.workload.spatial();
  if (!spatial)
    fail("application fixture workload is not Spatial");
  sim::SpatialSimulationRuntimeInputDraft draft{imported.workload.identity()};
  for (auto [ordinal, source] : llvm::enumerate(spatial->valueInputPlan))
    if (std::holds_alternative<sim::RuntimeValueInput>(source))
      draft.runtimeValues.push_back(
          {static_cast<std::uint64_t>(ordinal),
           {1, {sim::SemanticLane::defined(llvm::APInt(32, value))}}});
  auto runtime =
      take(sim::finalizeSimulationRuntimeInput(draft, imported.workload, view));
  return take(sim::publishSimulationRuntimeInput(runtime, store));
}

evaluation::models::FpaFeatureView
projectFpaFeatures(const ArtifactRootReference &dataflow,
                   const ArtifactRootReference &system,
                   const ResolvedConfig &config, const ArtifactStore &artifacts,
                   const BlobStore &blobs) {
  auto prepared =
      take(evaluation::models::prepareCanonicalDataflowFabricEvaluation(
          dataflow, system, config, artifacts, blobs));
  const evaluation::EvaluationModelDescriptor *descriptor =
      prepared.request.modelBinding().descriptorRef().descriptor();
  if (!descriptor)
    fail("FPA feature fixture lost its model descriptor");
  auto evaluationCase = take(evaluation::EvaluationCase::get(
      descriptor->caseSignature, prepared.request.subjectBindings(),
      prepared.request.workload(), prepared.request.runtimeInput(),
      prepared.request.baseConditions(), prepared.resolution, artifacts,
      blobs));
  auto projected = take(evaluation::projectModelFeatures(
      evaluation::models::fpaModelParameterContractRef(), evaluationCase,
      prepared.resolution, artifacts, blobs));
  const auto *features = projected.getIf<evaluation::models::FpaFeatureView>();
  if (!features)
    fail("FPA contract returned a foreign feature view");
  return *features;
}

std::vector<fabric::FabricModuleEntityCorrespondence>
identityModuleEntityCorrespondence(const fabric::FabricArtifactView &module) {
  std::vector<fabric::FabricModuleEntityCorrespondence> result;
  const auto append = [&](auto occurrences, fabric::FabricEntityKind kind) {
    for (std::uint64_t ordinal = 0; ordinal != occurrences.size(); ++ordinal) {
      const auto occurrence = occurrences[ordinal];
      result.push_back(
          {{kind, occurrence.id(), ordinal}, {kind, occurrence.id(), ordinal}});
    }
  };
  append(module.peOccurrences(), fabric::FabricEntityKind::FabricPeOccurrence);
  append(module.fuOccurrences(), fabric::FabricEntityKind::FabricFuOccurrence);
  append(module.memoryOccurrences(),
         fabric::FabricEntityKind::FabricMemoryOccurrence);
  append(module.switchOccurrences(),
         fabric::FabricEntityKind::FabricSwitchOccurrence);
  append(module.fifoOccurrences(),
         fabric::FabricEntityKind::FabricFifoOccurrence);
  append(module.boundaryOccurrences(),
         fabric::FabricEntityKind::FabricBoundaryOccurrence);
  llvm::sort(result, [](const auto &lhs, const auto &rhs) {
    return std::tie(lhs.source.kind, lhs.source.occurrenceOrdinal) <
           std::tie(rhs.source.kind, rhs.source.occurrenceOrdinal);
  });
  return result;
}

bool everyCoreIsUsed(const ArtifactRootReference &systemReference,
                     llvm::ArrayRef<ArtifactRootReference> mappings,
                     const ArtifactStore &store) {
  auto systemArtifact =
      take(fabric::importEntireFabricRoot(systemReference, store));
  auto system = take(fabric::requireSystemRoot(systemArtifact.view()));
  std::set<std::string> used;
  for (const ArtifactRootReference &reference : mappings) {
    auto mapping = take(mapping::importSystemMapping(reference, store));
    ArtifactRootReference dataflowReference{
        dataflow::canonicalDataflowSchema.identity.str(),
        dataflow::canonicalDataflowSchema.version,
        mapping.view().dataflowIdentity()};
    auto dataflowArtifact =
        take(dataflow::importCanonicalDataflow(dataflowReference, store));
    auto dataflowView = take(dataflowArtifact.view());
    auto projection = take(mapping::projectSystemExecutionContexts(
        dataflowView, mapping.view().executionBindings()));
    for (const auto &domain : projection.instructionDomains)
      used.insert(key(fabric::canonicalFabricBytes(domain.context.accCore)));
  }
  return llvm::all_of(
      system.artifact().accCoreOccurrences(),
      [&](fabric::AccCoreOccurrenceRef core) {
        return used.count(key(fabric::canonicalFabricBytes(core))) != 0;
      });
}

llvm::Expected<ResourceTimeSpectrumFunnelResult>
verifyAdjacentResourceTimeSchedule(
    const ArtifactRootReference &dataflowReference,
    const fabric::FabricSystemRootView &system,
    ::dataflow::RootThreadLaunchRef root, std::uint64_t resourceCount,
    llvm::ArrayRef<ArtifactRootReference> mappings, bool rejectResourceCount,
    const ArtifactStore &store) {
  if (resourceCount == 0)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "adjacent schedule has no resource");
  auto dataflowArtifact =
      dataflow::importCanonicalDataflow(dataflowReference, store);
  if (!dataflowArtifact)
    return dataflowArtifact.takeError();
  auto dataflow = dataflowArtifact->view();
  if (!dataflow)
    return dataflow.takeError();
  auto projection = projectResourceTimeDataflow(*dataflow, system, "host", 100);
  if (!projection)
    return projection.takeError();
  if (projection->regions.size() != 1 || projection->regionBounds.size() != 1 ||
      projection->regions.front().region != root)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "adjacent schedule lost its resource-time projection");

  const ResourceTimeRegionFeature &region = projection->regions.front();
  auto speedupPoint =
      llvm::find_if(region.speedupCurve, [&](const auto &point) {
        return point.resourceUnits == std::vector<std::uint64_t>{resourceCount};
      });
  if (speedupPoint == region.speedupCurve.end() ||
      speedupPoint->executionTimePicoseconds == 0)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "adjacent schedule has no exact requested-resource point");
  const std::uint64_t speedupPointOrdinal =
      speedupPoint - region.speedupCurve.begin();
  const std::uint64_t executionTime = speedupPoint->executionTimePicoseconds;

  ResourceTimeScheduleHint hint;
  hint.actions = {{ResourceTimeActionKind::AdmitRegion,
                   root,
                   speedupPointOrdinal,
                   0,
                   0,
                   {},
                   {},
                   {}},
                  {ResourceTimeActionKind::AdvanceEvent,
                   std::nullopt,
                   std::nullopt,
                   0,
                   executionTime,
                   {root},
                   {},
                   {}}};
  hint.states = {{0, {}, {root}, {}, executionTime},
                 {0,
                  {{root, speedupPointOrdinal, speedupPoint->resourceUnits,
                    executionTime}},
                  {},
                  {},
                  executionTime},
                 {executionTime, {}, {}, {root}, executionTime}};
  hint.estimatedMakespanPicoseconds = executionTime;
  hint.optimisticMakespanLowerBoundPicoseconds = executionTime;
  hint.peakConcurrentRegions = 1;
  hint.totalAllocatedResourceTime = executionTime * resourceCount;
  hint.support = speedupPoint->support;

  auto bounds = projection->regionBounds;
  if (rejectResourceCount) {
    bounds.front().maximumUsefulResourceUnits = resourceCount - 1;
    bounds.front().minimumFeasibleResourceUnits = 1;
    bounds.front().minimumSupport = ResourceTimeEstimateSupport::Exact;
  }
  return verifyResourceTimeMappingFinalists(
      {hint}, projection->regions, bounds, mappings, store, {},
      ResourceTimeConcurrencyBounds{1, 1, ResourceTimeEstimateSupport::Exact});
}

void exerciseAdjacentResourceTimeMappingRepair(
    llvm::StringRef temporaryPath, const JointDesignExplorationPlan &plan,
    const JointDesignExecution &parentExecution,
    const JointDesignPolicy &policy, ::dataflow::RootThreadLaunchRef mappedRoot,
    const ArtifactRootReference &systemReference,
    const fabric::FabricSystemRootView &system,
    const ArtifactRootReference &alternateSystem,
    const ArtifactRootReference &parentMapping, bool runBoundedQuality,
    const JointBoundedQualityPolicy *incompleteQualityPolicy,
    const ArtifactStore &store, const BlobStore &blobs) {
  llvm::SmallString<128> adjacentJournal(temporaryPath);
  llvm::sys::path::append(adjacentJournal, "adjacent-resource-time");
  const std::array adjacentPartitions = {
      pnr::SystemBindingPartitionIntent{mappedRoot, 2}};
  const std::array adjacentRoots = {mappedRoot};
  JointHardwareReopenRequest adjacentRequest{
      take(DseProducerSemanticBuildIdentity::get(
          "loom.test.resource_time_adjacent.v1")),
      adjacentJournal.str().str(),
      {},
      JointDesignStoppingPolicy::FirstVerified,
      std::nullopt,
      std::nullopt,
      take(SiteCapacity::get(2, 0, 0)),
      take(PlanExecutionPolicy::get(2, take(SiteResourceClaim::get(1, 0, 0))))};
  adjacentRequest.invocationSemanticInputs = {alternateSystem};
  const auto verifyAdjacentSchedule =
      [&](llvm::ArrayRef<ArtifactRootReference> candidates,
          bool rejectByResourceBound)
      -> llvm::Expected<ResourceTimeSpectrumFunnelResult> {
    return verifyAdjacentResourceTimeSchedule(
        plan.frontier.pairs.front().software.dataflow, system, mappedRoot,
        adjacentPartitions.front().partitionCount, candidates,
        rejectByResourceBound, store);
  };
  std::vector<ArtifactRootReference> acceptedAdjacentMappings;
  std::uint64_t coldAdjacentVerifications = 0;
  std::uint64_t incrementalAdjacentVerifications = 0;
  const auto verifyAdjacentMapping =
      [&](JointResourceTimeMappingRepairSide side,
          llvm::ArrayRef<ArtifactRootReference> candidates)
      -> llvm::Expected<ResourceTimeSpectrumFunnelResult> {
    if (side == JointResourceTimeMappingRepairSide::Cold)
      ++coldAdjacentVerifications;
    else
      ++incrementalAdjacentVerifications;
    acceptedAdjacentMappings.insert(acceptedAdjacentMappings.end(),
                                    candidates.begin(), candidates.end());
    return verifyAdjacentSchedule(candidates, false);
  };
  auto adjacentRepair = take(executeResourceTimeAdjacentMappingRepair(
      plan, parentExecution, policy, adjacentPartitions, adjacentRoots,
      verifyAdjacentMapping, std::move(adjacentRequest), store, blobs));
  if (!adjacentRepair.incrementalExecution)
    fail("adjacent resource-time repair omitted its System execution");
  auto &adjacentExecution = *adjacentRepair.incrementalExecution;
  std::vector<ArtifactRootReference> adjacentSemanticInputs =
      projectJointDesignSemanticInputs(adjacentRepair.plan);
  adjacentSemanticInputs.push_back(alternateSystem);
  const auto adjacentClosure = take(DseRunClosure::get(
      take(DseProducerSemanticBuildIdentity::get(
          "loom.test.resource_time_adjacent.v1")),
      adjacentSemanticInputs, adjacentRepair.plan.resolvedConfig, {}, store));
  if (!adjacentExecution.invocationRunKey() ||
      *adjacentExecution.invocationRunKey() != adjacentClosure.runKey().bytes())
    fail("adjacent repair closure omitted its invocation semantic input");
  if (!adjacentRepair.migrationSeed)
    fail("completed adjacent repair omitted its finalized migration seed");
  const auto adjacentSeed = take(pnr::importSystemMappingMigrationSeed(
      *adjacentRepair.migrationSeed, store));
  if (adjacentSeed.reopenedRoots() !=
          llvm::ArrayRef<::dataflow::RootThreadLaunchRef>(adjacentRoots) ||
      adjacentRepair.coldExecution.summary.techMappingDispatchCount == 0 ||
      adjacentRepair.coldExecution.summary.spatialPnrDispatchCount == 0 ||
      adjacentRepair.coldExecution.summary.systemPnrDispatchCount == 0 ||
      adjacentRepair.coldExecution.summary.coldReopenWallTimeNanoseconds !=
          adjacentRepair.coldExecution.summary.executionWallTimeNanoseconds ||
      adjacentRepair.coldExecution.summary
              .incrementalReopenWallTimeNanoseconds != 0 ||
      adjacentExecution.summary.techMappingDispatchCount == 0 ||
      adjacentExecution.summary.spatialPnrDispatchCount == 0 ||
      adjacentExecution.summary.systemPnrDispatchCount != 1 ||
      adjacentExecution.summary.incrementalReopenWallTimeNanoseconds !=
          adjacentExecution.summary.executionWallTimeNanoseconds ||
      adjacentExecution.summary.coldReopenWallTimeNanoseconds != 0 ||
      adjacentExecution.summary.preservedTechMappings != 0 ||
      adjacentExecution.summary.preservedSpatialMappings != 0 ||
      adjacentRepair.reuseDisposition !=
          JointMappingReuseDisposition::ColdFallback ||
      adjacentRepair.plan.resolvedConfig.dse.systemPnr.search.completionGoal !=
          ResolvedPnrCompletionGoal::ExhaustConfiguredWork)
    fail("adjacent resource-time finalist retained a reopened root Mapping");
  std::vector<ArtifactRootReference> adjacentMappings;
  for (const auto &pair : adjacentExecution.mappedPairs)
    adjacentMappings.insert(adjacentMappings.end(), pair.systemMappings.begin(),
                            pair.systemMappings.end());
  llvm::sort(adjacentMappings, artifactRootReferenceLess);
  adjacentMappings.erase(
      std::unique(adjacentMappings.begin(), adjacentMappings.end()),
      adjacentMappings.end());
  std::vector<ArtifactRootReference> coldAdjacentMappings;
  for (const auto &pair : adjacentRepair.coldExecution.mappedPairs)
    coldAdjacentMappings.insert(coldAdjacentMappings.end(),
                                pair.systemMappings.begin(),
                                pair.systemMappings.end());
  llvm::sort(coldAdjacentMappings, artifactRootReferenceLess);
  coldAdjacentMappings.erase(
      std::unique(coldAdjacentMappings.begin(), coldAdjacentMappings.end()),
      coldAdjacentMappings.end());
  if (adjacentMappings.empty() ||
      llvm::is_contained(adjacentMappings, parentMapping))
    fail("adjacent resource-time repair did not publish a distinct Mapping");
  if (coldAdjacentMappings.empty() || !adjacentRepair.coldMapping ||
      !adjacentRepair.incrementalMapping ||
      !llvm::is_contained(coldAdjacentMappings, *adjacentRepair.coldMapping) ||
      !llvm::is_contained(adjacentMappings,
                          *adjacentRepair.incrementalMapping) ||
      !llvm::is_contained(acceptedAdjacentMappings,
                          *adjacentRepair.coldMapping) ||
      !llvm::is_contained(acceptedAdjacentMappings,
                          *adjacentRepair.incrementalMapping) ||
      coldAdjacentVerifications == 0 || incrementalAdjacentVerifications == 0 ||
      !adjacentRepair.coldSelectionSpectrum ||
      !adjacentRepair.incrementalSelectionSpectrum ||
      !std::holds_alternative<VerifiedResourceTimeSpectrum>(
          adjacentRepair.coldSelectionSpectrum->verification) ||
      !std::holds_alternative<VerifiedResourceTimeSpectrum>(
          adjacentRepair.incrementalSelectionSpectrum->verification) ||
      !llvm::is_contained(adjacentRepair.coldEligibleMappings,
                          *adjacentRepair.coldMapping) ||
      !llvm::is_contained(adjacentRepair.incrementalEligibleMappings,
                          *adjacentRepair.incrementalMapping))
    fail("adjacent resource-time repair did not publish a paired cold and "
         "incremental Mapping accepted by its caller-owned oracle");
  auto adjacentMapping = take(
      mapping::importSystemMapping(*adjacentRepair.incrementalMapping, store));
  if (adjacentMapping.view().dataflowIdentity() !=
          plan.frontier.pairs.front().software.dataflow.artifact ||
      adjacentMapping.view().fabricIdentity() != systemReference.artifact)
    fail("adjacent resource-time repair changed immutable owners");
  auto adjacentDataflowArtifact = take(dataflow::importCanonicalDataflow(
      plan.frontier.pairs.front().software.dataflow, store));
  auto adjacentDataflow = take(adjacentDataflowArtifact.view());
  auto adjacentContexts = take(mapping::projectSystemExecutionContexts(
      adjacentDataflow, adjacentMapping.view().executionBindings()));
  auto adjacentResources = take(
      pnr::projectResourceTimeMappingResources(adjacentContexts, mappedRoot));
  auto coldAdjacentMapping =
      take(mapping::importSystemMapping(*adjacentRepair.coldMapping, store));
  auto coldAdjacentContexts = take(mapping::projectSystemExecutionContexts(
      adjacentDataflow, coldAdjacentMapping.view().executionBindings()));
  auto coldAdjacentResources = take(pnr::projectResourceTimeMappingResources(
      coldAdjacentContexts, mappedRoot));
  if (adjacentResources.size() != adjacentPartitions.front().partitionCount ||
      coldAdjacentResources.size() != adjacentPartitions.front().partitionCount)
    fail("adjacent resource-time repair did not remap the reopened root to "
         "its requested resource count");
  const auto rejectAdjacentMapping =
      [&](JointResourceTimeMappingRepairSide,
          llvm::ArrayRef<ArtifactRootReference> candidates) {
        return verifyAdjacentSchedule(candidates, true);
      };
  const auto rejectedSelection =
      take(joint_reopen_detail::selectResourceTimePartitionMapping(
          adjacentRepair.coldExecution,
          plan.frontier.pairs.front().software.dataflow, systemReference,
          adjacentPartitions, adjacentRoots, nullptr, {},
          PreMappingSpectrumEndpoint::Automatic,
          JointResourceTimeMappingRepairSide::Cold, rejectAdjacentMapping,
          store));
  const auto *rejectedSpectrum =
      rejectedSelection.spectrum
          ? std::get_if<IncompleteResourceTimeSpectrum>(
                &rejectedSelection.spectrum->verification)
          : nullptr;
  if (rejectedSelection.mapping ||
      adjacentRepair.coldExecution.summary.selectedMapping ||
      !rejectedSpectrum ||
      rejectedSpectrum->reason !=
          ResourceTimeSpectrumIncompleteReason::ProofNotEstablished ||
      rejectedSelection.eligibleMappings.empty() ||
      !llvm::all_of(rejectedSelection.eligibleMappings,
                    [&](const auto &mapping) {
                      return llvm::is_contained(coldAdjacentMappings, mapping);
                    }))
    fail("adjacent resource-time selection lost its typed no-match frontier");

  std::uint64_t endpointMismatchVerifications = 0;
  const auto verifyEndpointMismatch =
      [&](JointResourceTimeMappingRepairSide,
          llvm::ArrayRef<ArtifactRootReference> candidates) {
        ++endpointMismatchVerifications;
        return verifyAdjacentSchedule(candidates, false);
      };
  const auto endpointMismatch =
      take(joint_reopen_detail::selectResourceTimePartitionMapping(
          adjacentRepair.coldExecution,
          plan.frontier.pairs.front().software.dataflow, systemReference,
          adjacentPartitions, adjacentRoots, nullptr, {},
          PreMappingSpectrumEndpoint::MaxSpatial,
          JointResourceTimeMappingRepairSide::Cold, verifyEndpointMismatch,
          store));
  const auto *mismatchedSpectrum =
      endpointMismatch.spectrum ? std::get_if<VerifiedResourceTimeSpectrum>(
                                      &endpointMismatch.spectrum->verification)
                                : nullptr;
  if (endpointMismatch.mapping ||
      adjacentRepair.coldExecution.summary.selectedMapping ||
      endpointMismatchVerifications == 0 || !mismatchedSpectrum ||
      mismatchedSpectrum->scenarios.empty() ||
      !llvm::all_of(mismatchedSpectrum->scenarios,
                    [](const auto &scenario) {
                      return scenario.spectrumClass ==
                             PreMappingSpectrumClass::Intermediate;
                    }) ||
      endpointMismatch.eligibleMappings !=
          adjacentRepair.coldEligibleMappings ||
      !endpointMismatch.executionIncompleteReasons.empty())
    fail("adjacent resource-time selection accepted a mismatched endpoint");

  const auto forgeImportAccounting =
      [&](JointResourceTimeMappingRepairSide,
          llvm::ArrayRef<ArtifactRootReference> candidates)
      -> llvm::Expected<ResourceTimeSpectrumFunnelResult> {
    auto result = verifyAdjacentSchedule(candidates, false);
    if (!result)
      return result.takeError();
    result->accounting.independentlyImportedMappings = 0;
    return result;
  };
  auto malformedSelection =
      joint_reopen_detail::selectResourceTimePartitionMapping(
          adjacentRepair.coldExecution,
          plan.frontier.pairs.front().software.dataflow, systemReference,
          adjacentPartitions, adjacentRoots, nullptr, {},
          PreMappingSpectrumEndpoint::Automatic,
          JointResourceTimeMappingRepairSide::Cold, forgeImportAccounting,
          store);
  if (malformedSelection)
    fail("adjacent resource-time selection accepted forged import "
         "accounting");
  llvm::consumeError(malformedSelection.takeError());

  std::uint64_t cancellationVerifications = 0;
  const auto cancelSelection = [&](JointResourceTimeMappingRepairSide,
                                   llvm::ArrayRef<ArtifactRootReference>)
      -> llvm::Expected<ResourceTimeSpectrumFunnelResult> {
    ++cancellationVerifications;
    return ResourceTimeSpectrumFunnelResult{
        ResourceTimeSpectrumVerification{IncompleteResourceTimeSpectrum{
            ResourceTimeSpectrumIncompleteReason::CancelledOrTimeout,
            "adjacent resource-time selection cancelled", 0}},
        ResourceTimeSpectrumFunnelAccounting{}};
  };
  const auto cancelledSelection =
      take(joint_reopen_detail::selectResourceTimePartitionMapping(
          adjacentRepair.coldExecution,
          plan.frontier.pairs.front().software.dataflow, systemReference,
          adjacentPartitions, adjacentRoots, nullptr, {},
          PreMappingSpectrumEndpoint::Automatic,
          JointResourceTimeMappingRepairSide::Cold, cancelSelection, store));
  const auto *cancelledSpectrum =
      cancelledSelection.spectrum
          ? std::get_if<IncompleteResourceTimeSpectrum>(
                &cancelledSelection.spectrum->verification)
          : nullptr;
  if (cancelledSelection.mapping ||
      adjacentRepair.coldExecution.summary.selectedMapping ||
      cancellationVerifications != 1 || !cancelledSpectrum ||
      cancelledSpectrum->reason !=
          ResourceTimeSpectrumIncompleteReason::CancelledOrTimeout ||
      cancelledSelection.eligibleMappings !=
          adjacentRepair.coldEligibleMappings ||
      !cancelledSelection.executionIncompleteReasons.empty())
    fail("adjacent resource-time cancellation lost its exact frontier");

  if (runBoundedQuality) {
    if (!incompleteQualityPolicy)
      fail("quality-promotion fixture lost its incomplete repair policy");
    llvm::SmallString<128> incompleteAdjacentJournal(temporaryPath);
    llvm::sys::path::append(incompleteAdjacentJournal,
                            "adjacent-resource-time-quality-incomplete");
    JointHardwareReopenRequest incompleteAdjacentRequest{
        take(DseProducerSemanticBuildIdentity::get(
            "loom.test.resource_time_adjacent.quality_incomplete.v1")),
        incompleteAdjacentJournal.str().str(),
        {},
        JointDesignStoppingPolicy::BoundedQuality,
        *incompleteQualityPolicy,
        std::nullopt,
        take(SiteCapacity::get(2, 0, 0)),
        take(PlanExecutionPolicy::get(2,
                                      take(SiteResourceClaim::get(1, 0, 0))))};
    const auto incompleteAdjacent =
        take(executeResourceTimeAdjacentMappingRepair(
            plan, parentExecution, policy, adjacentPartitions, adjacentRoots,
            verifyAdjacentMapping, std::move(incompleteAdjacentRequest), store,
            blobs));
    if (!incompleteAdjacent.incrementalExecution)
      fail("bounded incomplete adjacent repair omitted its System execution");
    const auto retainsMappedPair = [](const auto &execution) {
      return llvm::any_of(execution.mappedPairs, [](const auto &pair) {
        return !pair.systemMappings.empty();
      });
    };
    for (const auto *execution : {&incompleteAdjacent.coldExecution,
                                  &*incompleteAdjacent.incrementalExecution})
      if (!retainsMappedPair(*execution) ||
          execution->summary.qualityDisposition !=
              JointDesignQualityDisposition::Unsupported ||
          execution->summary.selectedMapping ||
          execution->summary.selectedPlanOrdinal)
        fail("bounded incomplete adjacent repair retained a selected Mapping");
    if (incompleteAdjacent.coldMapping || incompleteAdjacent.incrementalMapping)
      fail("bounded incomplete adjacent repair published a Mapping join");
  }
}

} // namespace loom::dse::joint_test
