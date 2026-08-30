#include "DSE/JointDesignExploration.h"
#include "ADG/Builtin.h"
#include "Application/Build.h"
#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "Config/ResolvedConfig.h"
#include "DSE/HardwareDecision.h"
#include "DSE/JointHardwareReopen.h"
#include "DSE/JointMappingMigration.h"
#include "DSE/ResolvedConfigView.h"
#include "DSE/RootCompleteTechMappingCandidateGenerator.h"
#include "JointHardwareReopenExecution.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Evaluation/Evidence.h"
#include "Evaluation/ModelParameter.h"
#include "Evaluation/ModelParameterBundle.h"
#include "Evaluation/Models/CanonicalDataflowFabricAnalytic.h"
#include "Evaluation/Models/FpaParameterContract.h"
#include "Evaluation/ProductionRegistry.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Artifact/FabricSystemRootView.h"
#include "Fabric/Identity/FabricPhysicalTiming.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Frontend/IR/LoomOps.h"
#include "Mapping/Artifact/MappingConstraintSet.h"
#include "Mapping/Artifact/SystemMappingArtifact.h"
#include "Mapping/Artifact/SystemMappingExecutionProjection.h"
#include "PnR/System/SystemMappingMigration.h"
#include "Simulator/SimulationArtifacts.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <set>
#include <string>
#include <system_error>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

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

class TemporaryDirectory final {
public:
  TemporaryDirectory() {
    if (std::error_code error =
            llvm::sys::fs::createUniqueDirectory("loom-joint-design", path_))
      fail("cannot create test directory: " + error.message());
  }
  ~TemporaryDirectory() { llvm::sys::fs::remove_directories(path_); }
  llvm::StringRef path() const { return path_; }

private:
  llvm::SmallString<128> path_;
};

mlir::MLIRContext makeContext() {
  mlir::DialectRegistry registry;
  registry
      .insert<dataflow::DataflowDialect, mlir::arith::ArithDialect,
              mlir::DLTIDialect, mlir::func::FuncDialect, loom::LoomDialect>();
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
  func.func private @host() {
    %value = arith.constant )mlir" +
                             std::to_string(constant) + R"mlir( : i32
    %extent = arith.constant 4 : index
    %thread = dataflow.thread.launch @worker(%value) grid(%extent)
        : (i32) -> !dataflow.thread_token
    return
  }
}

)mlir";
  auto module = mlir::parseSourceString<mlir::ModuleOp>(source, &context);
  if (!module)
    fail("cannot parse Dataflow fixture");
  return take(dataflow::finalizeCanonicalDataflow(*module));
}

loom::ArtifactRootReference
publishApplicationWorkload(const dataflow::CanonicalDataflowArtifact &artifact,
                           const loom::ArtifactStore &store) {
  auto view = take(artifact.view());
  if (view.rootThreadLaunches().size() != 1 ||
      view.staticGraphLaunches().size() != 1)
    fail("application fixture does not have one rooted graph launch");
  dataflow::RootedGraphLaunchRef launch{view.rootThreadLaunches().front().ref,
                                        view.staticGraphLaunches().front().ref};
  loom::sim::SpatialSimulationWorkload draft{launch};
  auto logicalDomain =
      take(view.projectRootThreadLogicalDomain(launch.rootThreadLaunch));
  draft.denseCoordinates.assign(logicalDomain.coordinateRank, 0);
  auto shapes =
      take(loom::sim::projectSpatialSimulationBoundaryShapes(view, launch));
  draft.valueInputPlan.assign(shapes.valueInputs.size(),
                              loom::sim::RuntimeValueInput{});
  auto workload = take(loom::sim::finalizeSimulationWorkload(draft, view));
  return take(loom::sim::publishSimulationWorkload(workload, store));
}

loom::evaluation::models::FpaFeatureView
projectFpaFeatures(const loom::ArtifactRootReference &dataflow,
                   const loom::ArtifactRootReference &system,
                   const loom::ResolvedConfig &config,
                   const loom::ArtifactStore &artifacts,
                   const loom::BlobStore &blobs) {
  auto prepared =
      take(loom::evaluation::models::prepareCanonicalDataflowFabricEvaluation(
          dataflow, system, config, artifacts, blobs));
  const loom::evaluation::EvaluationModelDescriptor *descriptor =
      prepared.request.modelBinding().descriptorRef().descriptor();
  if (!descriptor)
    fail("FPA feature fixture lost its model descriptor");
  auto evaluationCase = take(loom::evaluation::EvaluationCase::get(
      descriptor->caseSignature, prepared.request.subjectBindings(),
      prepared.request.workload(), prepared.request.runtimeInput(),
      prepared.request.baseConditions(), prepared.resolution, artifacts,
      blobs));
  auto projected = take(loom::evaluation::projectModelFeatures(
      loom::evaluation::models::fpaModelParameterContractRef(), evaluationCase,
      prepared.resolution, artifacts, blobs));
  const auto *features =
      projected.getIf<loom::evaluation::models::FpaFeatureView>();
  if (!features)
    fail("FPA contract returned a foreign feature view");
  return *features;
}

std::string key(llvm::ArrayRef<std::uint8_t> bytes) {
  return std::string(reinterpret_cast<const char *>(bytes.data()),
                     bytes.size());
}

std::vector<loom::fabric::FabricModuleEntityCorrespondence>
identityModuleEntityCorrespondence(
    const loom::fabric::FabricArtifactView &module) {
  std::vector<loom::fabric::FabricModuleEntityCorrespondence> result;
  const auto append = [&](auto occurrences,
                          loom::fabric::FabricEntityKind kind) {
    for (std::uint64_t ordinal = 0; ordinal != occurrences.size(); ++ordinal) {
      const auto occurrence = occurrences[ordinal];
      result.push_back(
          {{kind, occurrence.id(), ordinal}, {kind, occurrence.id(), ordinal}});
    }
  };
  append(module.peOccurrences(),
         loom::fabric::FabricEntityKind::FabricPeOccurrence);
  append(module.fuOccurrences(),
         loom::fabric::FabricEntityKind::FabricFuOccurrence);
  append(module.memoryOccurrences(),
         loom::fabric::FabricEntityKind::FabricMemoryOccurrence);
  append(module.switchOccurrences(),
         loom::fabric::FabricEntityKind::FabricSwitchOccurrence);
  append(module.fifoOccurrences(),
         loom::fabric::FabricEntityKind::FabricFifoOccurrence);
  append(module.boundaryOccurrences(),
         loom::fabric::FabricEntityKind::FabricBoundaryOccurrence);
  llvm::sort(result, [](const auto &lhs, const auto &rhs) {
    return std::tie(lhs.source.kind, lhs.source.occurrenceOrdinal) <
           std::tie(rhs.source.kind, rhs.source.occurrenceOrdinal);
  });
  return result;
}

bool everyCoreIsUsed(const loom::ArtifactRootReference &systemReference,
                     llvm::ArrayRef<loom::ArtifactRootReference> mappings,
                     const loom::ArtifactStore &store) {
  auto systemArtifact =
      take(loom::fabric::importEntireFabricRoot(systemReference, store));
  auto system = take(loom::fabric::requireSystemRoot(systemArtifact.view()));
  std::set<std::string> used;
  for (const loom::ArtifactRootReference &reference : mappings) {
    auto mapping = take(loom::mapping::importSystemMapping(reference, store));
    loom::ArtifactRootReference dataflowReference{
        dataflow::canonicalDataflowSchema.identity.str(),
        dataflow::canonicalDataflowSchema.version,
        mapping.view().dataflowIdentity()};
    auto dataflowArtifact =
        take(dataflow::importCanonicalDataflow(dataflowReference, store));
    auto dataflowView = take(dataflowArtifact.view());
    auto projection = take(loom::mapping::projectSystemExecutionContexts(
        dataflowView, mapping.view().executionBindings()));
    for (const auto &domain : projection.instructionDomains)
      used.insert(
          key(loom::fabric::canonicalFabricBytes(domain.context.accCore)));
  }
  return llvm::all_of(
      system.artifact().accCoreOccurrences(),
      [&](loom::fabric::AccCoreOccurrenceRef core) {
        return used.count(key(loom::fabric::canonicalFabricBytes(core))) != 0;
      });
}

/// Selects which hardware mutation families the matrix executes. Each family
/// owns one independent cold plus preserve-first child PnR, so a selector lets
/// the families run as separate processes instead of one serial tail. An empty
/// selector disables the matrix; `allMutationFamilies` runs every family in one
/// process.
constexpr llvm::StringLiteral allMutationFamilies = "*";

/// The typed repair dispositions each mutation family must reach. Pinning them
/// per family keeps the matrix's disposition coverage provable when the
/// families run as separate processes, and is stricter than asserting that some
/// family somewhere reached each disposition.
struct MutationFamilyContract {
  llvm::StringLiteral label;
  loom::dse::JointMappingReuseDisposition rebase;
  loom::dse::JointSystemMappingReuseDisposition system;
};

constexpr MutationFamilyContract mutationFamilyContracts[] = {
    {"fu", loom::dse::JointMappingReuseDisposition::ColdFallback,
     loom::dse::JointSystemMappingReuseDisposition::ColdFallback},
    {"memory", loom::dse::JointMappingReuseDisposition::ColdFallback,
     loom::dse::JointSystemMappingReuseDisposition::ColdFallback},
    {"fifo", loom::dse::JointMappingReuseDisposition::ColdFallback,
     loom::dse::JointSystemMappingReuseDisposition::ColdFallback},
    {"operand", loom::dse::JointMappingReuseDisposition::ColdFallback,
     loom::dse::JointSystemMappingReuseDisposition::ColdFallback},
    {"switch", loom::dse::JointMappingReuseDisposition::LocalRepair,
     loom::dse::JointSystemMappingReuseDisposition::ColdFallback},
    {"spatial-core", loom::dse::JointMappingReuseDisposition::ColdFallback,
     loom::dse::JointSystemMappingReuseDisposition::ColdFallback},
    {"acc-core-add", loom::dse::JointMappingReuseDisposition::Preserved,
     loom::dse::JointSystemMappingReuseDisposition::Preserved},
    {"acc-core-remove", loom::dse::JointMappingReuseDisposition::Preserved,
     loom::dse::JointSystemMappingReuseDisposition::Reopened},
    {"transport", loom::dse::JointMappingReuseDisposition::Preserved,
     loom::dse::JointSystemMappingReuseDisposition::ColdFallback},
    {"service", loom::dse::JointMappingReuseDisposition::Preserved,
     loom::dse::JointSystemMappingReuseDisposition::ColdFallback},
    {"combined", loom::dse::JointMappingReuseDisposition::ColdFallback,
     loom::dse::JointSystemMappingReuseDisposition::ColdFallback},
};

const MutationFamilyContract &mutationFamilyContract(llvm::StringRef label) {
  for (const MutationFamilyContract &contract : mutationFamilyContracts)
    if (contract.label == label)
      return contract;
  fail("mutation family has no declared disposition contract: " + label.str());
}

/// Proves, without executing any Mapping work, that the declared family
/// contracts still span every repair disposition the matrix must witness.
void requireMutationDispositionCoverage() {
  bool lowerLayerPreservation = false;
  bool systemPreservation = false;
  bool systemReopen = false;
  bool coldFallback = false;
  for (const MutationFamilyContract &contract : mutationFamilyContracts) {
    lowerLayerPreservation |=
        contract.rebase != loom::dse::JointMappingReuseDisposition::ColdFallback;
    systemPreservation |=
        contract.system ==
        loom::dse::JointSystemMappingReuseDisposition::Preserved;
    systemReopen |= contract.system ==
                    loom::dse::JointSystemMappingReuseDisposition::Reopened;
    coldFallback |=
        contract.rebase ==
            loom::dse::JointMappingReuseDisposition::ColdFallback ||
        contract.system ==
            loom::dse::JointSystemMappingReuseDisposition::ColdFallback;
  }
  if (!lowerLayerPreservation || !systemPreservation || !systemReopen ||
      !coldFallback)
    fail("mutation family contracts do not span every repair disposition");
}

/// Selects which bounded-quality sections run. The four sections share one
/// fixture and one bounded quality policy but are otherwise independent, so a
/// selector lets them run as parallel processes. Empty disables them;
/// `allMutationFamilies` (the shared "*" selector) runs all four in order.
void exerciseJointExploration(bool runFifoHardwareRepair,
                              bool runOperandHardwareRepair,
                              bool runTransportRepair,
                              llvm::StringRef qualitySection,
                              llvm::StringRef mutationFamily) {
  const auto qualityRuns = [qualitySection](llvm::StringRef section) {
    return qualitySection == allMutationFamilies ||
           qualitySection == section;
  };
  // One import session for the whole anchor, matching what the loom-dse and
  // loom-system-run entry points install; nested production sessions defer to
  // it through ReuseEnclosing.
  loom::fabric::FabricArtifactImportSession fabricImportSession;
  TemporaryDirectory temporary;
  llvm::SmallString<128> blobPath(temporary.path());
  llvm::sys::path::append(blobPath, "blobs");
  if (std::error_code error = llvm::sys::fs::create_directories(blobPath))
    fail("cannot create BlobStore directory: " + error.message());
  loom::ArtifactStore store(temporary.path());
  loom::BlobStore blobs(blobPath);
  mlir::MLIRContext context = makeContext();

  auto first = buildDataflow(context, 7);
  auto second = buildDataflow(context, 11);
  take(dataflow::publishCanonicalDataflow(first, store));
  take(dataflow::publishCanonicalDataflow(second, store));
  const loom::ArtifactRootReference firstWorkload =
      publishApplicationWorkload(first, store);
  const loom::ArtifactRootReference secondWorkload =
      publishApplicationWorkload(second, store);
  auto small = take(loom::adg::buildBuiltinTarget(
      store, loom::adg::BuiltinTargetPreset::Small));
  auto alternate = take(loom::adg::buildBuiltinTarget(
      store, loom::adg::BuiltinTargetPreset::Coverage));
  if (small.roots().size() != 1 || alternate.roots().size() != 1)
    fail("builtin fixture did not publish one complete System");
  const loom::ArtifactRootReference system = small.roots().front().reference();
  const loom::ArtifactRootReference alternateSystem =
      alternate.roots().front().reference();
  auto systemArtifact =
      take(loom::fabric::importEntireFabricRoot(system, store));
  auto systemView =
      take(loom::fabric::requireSystemRoot(systemArtifact.view()));
  auto timingProfiles = take(
      loom::fabric::projectNormalizedSystemPhysicalTimingProfiles(systemView));
  std::vector<loom::ArtifactRootReference> timingProfileRoots;
  for (const auto &profile : timingProfiles)
    timingProfileRoots.push_back(
        take(loom::fabric::publishFabricPhysicalTimingProfile(profile, store)));

  const loom::dse::JointDesignPolicy policy =
      take(loom::dse::JointDesignPolicy::get(2, 1, 1, 2, 32));
  loom::ResolvedConfig config = loom::defaultResolvedConfig();
  config.dse.techMapping.candidatePublicationLimit = 4;
  auto plan = take(loom::dse::buildJointDesignExplorationPlan(
      {{{firstWorkload}, {secondWorkload}}, {system}}, timingProfileRoots,
      policy, config, store));
  if (plan.frontier.eligiblePairCount != 2 || !plan.frontier.truncated ||
      plan.frontier.pairs.size() != 1 || plan.pairOutputs.size() != 1)
    fail("bounded pair frontier did not declare deterministic truncation");
  if (plan.frontier.analyticEvaluatedPairCount != 2 ||
      plan.frontier.analyticDeferredPairCount != 1 ||
      plan.frontier.pairProjections.size() != 1 ||
      plan.frontier.pairProjections.front().softwareActorCount == 0 ||
      plan.frontier.pairProjections.front().systemAccCoreCount == 0)
    fail("analytic pair funnel lost bounded ranking evidence");
  if (plan.pairOutputs.front().techMappings.empty() ||
      plan.pairOutputs.front().spatialMappings.empty())
    fail("joint Mapping plan lost an intermediate result projection");
  const auto &systemNode = std::get<loom::dse::GeneratePlanNodeDefinition>(
      plan.resolvedConfig.dse.planNodes
          [plan.pairOutputs.front().systemMappings.producerNodeOrdinal]);
  const auto &join =
      std::get<loom::dse::BoundedPlanOutputJoin>(systemNode.inputBindings[1]);
  if (join.outputs.empty() || join.maximumArtifacts != 32)
    fail("joint Mapping plan lost its explicit SpatialMapping bound");
  for (const loom::dse::PlanOutputRef &spatialOutput : join.outputs) {
    const auto &spatialNode = std::get<loom::dse::GeneratePlanNodeDefinition>(
        plan.resolvedConfig.dse.planNodes[spatialOutput.producerNodeOrdinal]);
    const auto &techJoin = std::get<loom::dse::BoundedPlanOutputJoin>(
        spatialNode.inputBindings.front());
    if (techJoin.outputs.size() != 1 || techJoin.maximumArtifacts != 2)
      fail("joint Mapping plan lost its TechMapping admission bound");
    const auto &techOutput = techJoin.outputs.front();
    const auto &techNode = std::get<loom::dse::GeneratePlanNodeDefinition>(
        plan.resolvedConfig.dse.planNodes[techOutput.producerNodeOrdinal]);
    if (techNode.descriptor !=
        loom::dse::applicationGraphTechMappingCandidateGeneratorDescriptor()
            .reference())
      fail("joint Mapping plan used a whole-program TechMapping cover");
  }

  auto view =
      take(loom::dse::projectResolvedDseConfigView(plan.resolvedConfig));
  auto execution = take(loom::dse::executeDsePlan(view, store, blobs));
  const loom::dse::CompletedDsePlanExecution *completed =
      std::get_if<loom::dse::CompletedDsePlanExecution>(&execution);
  if (!completed) {
    const auto &incomplete =
        std::get<loom::dse::IncompleteDsePlanExecution>(execution);
    const auto *reason =
        std::get_if<loom::dse::CandidateGeneratorIncompleteReason>(
            &incomplete.reason());
    if (!reason ||
        *reason != loom::dse::CandidateGeneratorIncompleteReason::
                       SemanticLimitReached ||
        incomplete.executionStopped())
      fail("joint Mapping plan changed retained frontier semantics: " +
           loom::dse::toString(incomplete.reason()));
    completed = &incomplete.availableExecution();
  }
  const std::vector<loom::ArtifactRootReference> mappings =
      completed->resolve(plan.pairOutputs.front().systemMappings).vec();
  if (mappings.empty())
    fail("joint Mapping plan produced no complete SystemMapping");
  for (const loom::ArtifactRootReference &reference : mappings) {
    auto mapping = take(loom::mapping::importSystemMapping(reference, store));
    if (mapping.view().dataflowIdentity() !=
            plan.frontier.pairs.front().software.dataflow.artifact ||
        mapping.view().fabricIdentity() != system.artifact)
      fail("joint Mapping output lost its exact pair owners");
  }

  std::optional<loom::dse::JointBoundedQualityPolicy> incompleteRepairQuality;
  std::optional<loom::dse::JointBoundedQualityPolicy> qualityPolicy;
  std::shared_ptr<std::uint64_t> qualityAcquisitionCount;
  if (!qualitySection.empty()) {
    if (llvm::Error error =
            loom::evaluation::registerProductionEvaluationRegistry())
      fail(llvm::toString(std::move(error)));
    loom::dse::CandidateMeasureObjectiveCatalogs objectiveCatalogs;
    objectiveCatalogs.dimensions = {
        {0, loom::ResolvedObjectiveDirection::Minimize, 0, 100}};
    objectiveCatalogs.weightedLevels = {{{{0, 1}}}};
    objectiveCatalogs.totalOrderings = {{{0}}};
    auto objectiveProgram = take(
        loom::dse::ObjectiveProgram::getCandidateMeasures(objectiveCatalogs));
    auto sharedObjectiveProgram =
        std::make_shared<const loom::dse::ObjectiveProgram>(
            std::move(objectiveProgram));
    qualityAcquisitionCount = std::make_shared<std::uint64_t>(0);

    loom::dse::JointBoundedQualityPolicy quality;
    quality.objectiveProgram = sharedObjectiveProgram;
    quality.objectiveDimensionLabels = {"mapping_quality"};
    quality.paretoDimensions = {0};
    quality.finalTotalOrdering = 0;
    quality.acquire = [sharedObjectiveProgram,
                       count = qualityAcquisitionCount](
                          const loom::dse::JointDesignExecution &result,
                          std::uint64_t planOrdinal)
        -> llvm::Expected<loom::dse::JointDesignQualityAcquisition> {
      ++*count;
      if (!result.summary.selectedMapping)
        return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                       "quality fixture has no Mapping");
      loom::dse::ObjectiveVector objective =
          sharedObjectiveProgram->makeVector();
      const std::array<std::uint64_t, 1> measures = {
          planOrdinal == 1 ? UINT64_C(0) : UINT64_C(1)};
      if (llvm::Error error = sharedObjectiveProgram->evaluateCandidateMeasures(
              measures, objective))
        return std::move(error);
      return loom::dse::JointDesignQualityAcquisition{
          std::vector<loom::dse::JointDesignQualityCandidate>{
              {{*result.summary.selectedMapping, std::move(objective)},
               std::nullopt,
               {{loom::resolvedObjectiveInteger(measures[0])}, {}, {}}}}};
    };
    quality.hardwarePromotion = loom::dse::JointHardwarePromotionQualityPolicy{
        sharedObjectiveProgram,
        {"predicted_mapping_quality"},
        0,
        [sharedObjectiveProgram](
            const loom::dse::JointDesignExplorationPlan &candidate,
            std::uint64_t planOrdinal)
            -> llvm::Expected<loom::dse::JointDesignQualityAcquisition> {
          if (candidate.frontier.systemFrontier.size() != 1)
            return llvm::createStringError(
                llvm::inconvertibleErrorCode(),
                "promotion fixture has no exact System");
          loom::dse::ObjectiveVector objective =
              sharedObjectiveProgram->makeVector();
          const std::array<std::uint64_t, 1> measures = {
              planOrdinal == 1 ? UINT64_C(0) : UINT64_C(1)};
          if (llvm::Error error =
                  sharedObjectiveProgram->evaluateCandidateMeasures(measures,
                                                                    objective))
            return std::move(error);
          return loom::dse::JointDesignQualityAcquisition{
              std::vector<loom::dse::JointDesignQualityCandidate>{
                  {{candidate.frontier.systemFrontier.front(),
                   std::move(objective)},
                   std::nullopt,
                   {{loom::resolvedObjectiveInteger(measures[0])}, {}, {}}}}};
        }};
    quality.maximumHardwareSpectrumParents = 1;
    quality.maximumHardwareRepairProbes = 1;
    qualityPolicy = quality;
    incompleteRepairQuality = quality;
    incompleteRepairQuality->acquire =
        [](const loom::dse::JointDesignExecution &result, std::uint64_t)
        -> llvm::Expected<loom::dse::JointDesignQualityAcquisition> {
      if (!result.summary.selectedMapping)
        return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                       "incomplete quality fixture has no "
                                       "Mapping");
      return loom::dse::JointDesignQualityAcquisition{
          loom::dse::IncompleteJointDesignQuality{
              loom::dse::JointDesignQualityIncompleteReason::Unsupported,
              result.summary.selectedMapping, std::nullopt}};
    };
  }
  if (qualityRuns("promotion")) {
    const loom::dse::JointDesignPolicy promotionPolicy =
        take(loom::dse::JointDesignPolicy::get(2, 2, 1, 2, 32));
    auto firstPlan = take(loom::dse::buildJointDesignExplorationPlan(
        {{{firstWorkload}}, {system}}, timingProfileRoots, promotionPolicy,
        config, store));
    auto secondPlan = take(loom::dse::buildJointDesignExplorationPlan(
        {{{secondWorkload}}, {system}}, timingProfileRoots, promotionPolicy,
        config, store));
    const std::array promotionPlans = {&firstPlan, &secondPlan};

    auto alternateRoot =
        take(loom::fabric::importEntireFabricRoot(alternateSystem, store));
    auto alternateView =
        take(loom::fabric::requireSystemRoot(alternateRoot.view()));
    auto alternateTiming =
        take(loom::fabric::projectNormalizedSystemPhysicalTimingProfiles(
            alternateView));
    std::vector<loom::ArtifactRootReference> alternateTimingRoots;
    for (const auto &profile : alternateTiming)
      alternateTimingRoots.push_back(take(
          loom::fabric::publishFabricPhysicalTimingProfile(profile, store)));
    auto alternatePlan = take(loom::dse::buildJointDesignExplorationPlan(
        {{{secondWorkload}}, {alternateSystem}}, alternateTimingRoots,
        promotionPolicy, config, store));

    const auto firstFeatures =
        projectFpaFeatures(firstPlan.frontier.softwareFrontier.front().dataflow,
                           system, config, store, blobs);
    const auto alternateFeatures = projectFpaFeatures(
        alternatePlan.frontier.softwareFrontier.front().dataflow,
        alternateSystem, config, store, blobs);
    const loom::evaluation::models::FpaMetricPredictionView firstObservation{
        take(loom::evaluation::DecimalValue::get(1, 8)),
        take(loom::evaluation::DecimalValue::get(1, -6)),
        take(loom::evaluation::DecimalValue::get(1, -3)),
        take(loom::evaluation::DecimalValue::get(1, -4))};
    const loom::evaluation::models::FpaMetricPredictionView
        alternateObservation{take(loom::evaluation::DecimalValue::get(5, 8)),
                             take(loom::evaluation::DecimalValue::get(2, -6)),
                             take(loom::evaluation::DecimalValue::get(2, -3)),
                             take(loom::evaluation::DecimalValue::get(2, -4))};
    auto parameters = take(loom::evaluation::models::trainFpaGbdtParameters(
        {{firstFeatures, firstObservation, {0x21}, {0x31}},
         {alternateFeatures, alternateObservation, {0x21}, {0x32}}},
        loom::evaluation::models::FpaGbdtTrainingConfig{13, 4, 2, 1, 1, 1}));
    auto bundle = take(loom::evaluation::finalizeModelParameterBundle(
        loom::evaluation::models::fpaModelParameterContractRef(),
        loom::evaluation::OwnerValue::get(std::move(parameters)), store,
        blobs));
    const std::array<std::uint8_t, 1> digestOwner = {0x41};
    const std::array<std::uint8_t, 1> digestValue = {0x42};
    const loom::ComponentViewDigest candidateDigest =
        take(loom::computeComponentViewDigest(digestOwner, digestValue));
    std::vector<loom::application::PreparedApplicationMappingAlternative>
        applicationAlternatives = {
            {0,
             0,
             candidateDigest,
             candidateDigest,
             {},
             firstPlan.frontier.softwareFrontier.front().dataflow,
             {},
             {},
             firstPlan},
            {1,
             1,
             candidateDigest,
             candidateDigest,
             {},
             alternatePlan.frontier.softwareFrontier.front().dataflow,
             {},
             {},
             alternatePlan}};
    loom::application::PreparedApplicationBuild preparedApplication{
        {},
        promotionPolicy,
        {},
        {},
        {},
        {},
        {},
        {},
        0,
        false,
        {},
        {},
        {},
        {},
        {},
        std::move(applicationAlternatives),
        {},
        {},
        loom::dse::StructuredOwnershipSelectionMode::SemanticConformance,
        loom::dse::StructuredOwnershipSelectionMode::SemanticConformance,
        {},
        std::nullopt,
        std::nullopt,
        firstPlan.frontier.softwareFrontier.front().dataflow,
        system,
        firstWorkload,
        firstWorkload,
        candidateDigest,
        systemView.artifact().accCoreOccurrences().size(),
        std::nullopt,
        std::nullopt,
        bundle.reference(),
        {}};
    const auto qualityPolicyExecution =
        take(loom::dse::PlanExecutionPolicy::get(
            32, take(loom::dse::SiteResourceClaim::get(1, 0, 0))));
    auto applicationQuality =
        take(loom::application::makeApplicationBoundedQualityPolicy(
            preparedApplication, qualityPolicyExecution, store, blobs));
    if (!applicationQuality.hardwarePromotion ||
        applicationQuality.objectiveDimensionLabels.size() != 7 ||
        applicationQuality.semanticInputs.size() != 3 ||
        !llvm::is_contained(applicationQuality.semanticInputs,
                            bundle.reference()))
      fail("application quality policy lost its frozen FPA closure");
    if (llvm::count_if(applicationQuality.semanticInputs, [](const auto &root) {
          return root.schemaIdentity ==
                     loom::evaluation::EvaluationRequest::artifactSchema
                         .identity &&
                 root.schemaVersion ==
                     loom::evaluation::EvaluationRequest::artifactSchema
                         .version;
        }) != 2)
      fail("application quality closure lost an exact FPA Request");
    auto firstPhysical =
        take(applicationQuality.hardwarePromotion->acquire(firstPlan, 0));
    auto alternatePhysical =
        take(applicationQuality.hardwarePromotion->acquire(alternatePlan, 1));
    auto *firstPhysicalObjectives =
        std::get_if<std::vector<loom::dse::JointDesignQualityCandidate>>(
            &firstPhysical);
    auto *alternatePhysicalObjectives =
        std::get_if<std::vector<loom::dse::JointDesignQualityCandidate>>(
            &alternatePhysical);
    if (!firstPhysicalObjectives || firstPhysicalObjectives->size() != 1 ||
        !alternatePhysicalObjectives ||
        alternatePhysicalObjectives->size() != 1 ||
        !firstPhysicalObjectives->front().evidence ||
        !alternatePhysicalObjectives->front().evidence)
      fail("application FPA promotion did not publish completed Evidence");
    const loom::ArtifactRootReference sharedQualityEvidence =
        *firstPhysicalObjectives->front().evidence;
    for (const auto *objectives :
         {firstPhysicalObjectives, alternatePhysicalObjectives}) {
      const loom::ArtifactRootReference &evidence =
          *objectives->front().evidence;
      if (evidence.schemaIdentity !=
              loom::evaluation::EvaluationEvidence::artifactSchema.identity ||
          evidence.schemaVersion !=
              loom::evaluation::EvaluationEvidence::artifactSchema.version)
        fail("application FPA promotion returned a foreign Evidence root");
      take(store.get(evidence));
    }
    auto physicalOrder =
        take(applicationQuality.hardwarePromotion->objectiveProgram
                 ->compareTotalOrdering(
                     alternatePhysicalObjectives->front().objective.objective,
                     loom::encodeArtifactRootReference(alternateSystem),
                     firstPhysicalObjectives->front().objective.objective,
                     loom::encodeArtifactRootReference(system), 0));
    if (physicalOrder >= 0)
      fail("frozen FPA predictions did not rank the better physical plan");

    const loom::dse::JointBoundedQualityPolicy repairQuality = *qualityPolicy;
    llvm::SmallString<128> promotionJournal(temporary.path());
    llvm::sys::path::append(promotionJournal, "hardware-quality-promotion");
    auto promoted = take(loom::dse::executeJointDesignWithHardwareReopen(
        promotionPlans, promotionPolicy,
        {take(loom::dse::DseProducerSemanticBuildIdentity::get(
             "loom.test.hardware_quality_promotion.v1")),
         promotionJournal.str().str(),
         {},
         loom::dse::JointDesignStoppingPolicy::BoundedQuality,
         std::move(*qualityPolicy),
         5,
         take(loom::dse::SiteCapacity::get(2, 0, 0)),
         take(loom::dse::PlanExecutionPolicy::get(
             32, take(loom::dse::SiteResourceClaim::get(1, 0, 0))))},
        store, blobs));
    if (promoted.summary.hardwareParentPromotions != 1 ||
        promoted.summary.hardwareReopensDeferredByQuality == 0 ||
        promoted.summary.hardwarePromotionObservations.size() != 2)
      fail("bounded hardware promotion lost its exact work ledger");
    if (!promoted.invocationManifest() ||
        promoted.supportingInvocationManifests().empty())
      fail("bounded hardware promotion lost its production DSE occurrences");
    take(loom::dse::importJointDesignInvocationManifest(
        *promoted.invocationManifest(), store, blobs));
    auto contentOnly = take(loom::dse::InvocationManifestReference::get(
        promoted.invocationManifest()->resolvedConfig(),
        promoted.invocationManifest()->blob(),
        promoted.invocationManifest()->occurrence(), store, blobs));
    if (contentOnly.journalReceipt())
      fail("strict manifest import forged an ExecutionJournal receipt");
    llvm::Error foreignBinding =
        loom::dse::appendJointDesignSupportingInvocationManifest(
            promoted, std::move(contentOnly));
    if (!foreignBinding)
      fail("joint execution accepted a manifest without its journal receipt");
    const std::string foreignBindingMessage =
        llvm::toString(std::move(foreignBinding));
    if (foreignBindingMessage.find("no ExecutionJournal receipt") ==
        std::string::npos)
      fail("foreign manifest rejection lost its journal ownership reason");
    for (const auto &supporting : promoted.supportingInvocationManifests())
      take(loom::dse::importJointDesignInvocationManifest(supporting, store,
                                                          blobs));
    for (const auto &observation :
         promoted.summary.hardwarePromotionObservations) {
      const bool expectedPromotion = observation.planOrdinal == 1;
      if (observation.promotedToExactMapping != expectedPromotion ||
          observation.incompleteReason ||
          observation.objectiveCodes.size() != 1)
        fail("bounded hardware promotion ignored its objective order");
    }
    if (!promoted.summary.selectedMapping)
      fail("bounded hardware promotion produced no selected Mapping");
    const loom::ArtifactRootReference promotedMapping =
        *promoted.summary.selectedMapping;
    llvm::SmallString<128> repairJournal(temporary.path());
    llvm::sys::path::append(repairJournal, "repair-quality-selection");
    const loom::dse::JointDesignExplorationPlan *firstRepairPlan = &firstPlan;
    auto firstRepair = take(loom::dse::executeJointDesignWithHardwareReopen(
        llvm::ArrayRef<const loom::dse::JointDesignExplorationPlan *>(
            &firstRepairPlan, 1),
        promotionPolicy,
        {take(loom::dse::DseProducerSemanticBuildIdentity::get(
             "loom.test.repair_quality_selection.v1")),
         repairJournal.str().str(),
         {},
         loom::dse::JointDesignStoppingPolicy::BoundedQuality,
         repairQuality,
         5,
         take(loom::dse::SiteCapacity::get(2, 0, 0)),
         take(loom::dse::PlanExecutionPolicy::get(
             32, take(loom::dse::SiteResourceClaim::get(1, 0, 0))))},
        store, blobs));
    if (!firstRepair.summary.selectedMapping ||
        firstRepair.summary.qualityDisposition !=
            loom::dse::JointDesignQualityDisposition::Complete)
      fail("single-plan repair quality execution produced no exact Mapping");
    const loom::ArtifactRootReference firstRepairMapping =
        *firstRepair.summary.selectedMapping;
    std::vector<loom::dse::JointDesignExecution> repairExecutions;
    repairExecutions.push_back(std::move(firstRepair));
    repairExecutions.push_back(std::move(promoted));
    const std::uint64_t acquisitionsBeforeRepairSelection =
        *qualityAcquisitionCount;
    auto repairSelection = take(loom::dse::selectJointRepairMappingByQuality(
        repairExecutions, repairQuality, store));
    const auto *selectedRepair =
        std::get_if<loom::dse::JointRepairQualitySelection>(&repairSelection);
    if (!selectedRepair || selectedRepair->executionOrdinal != 1 ||
        selectedRepair->mapping != promotedMapping)
      fail("repair quality selection ignored the shared objective order");
    if (*qualityAcquisitionCount != acquisitionsBeforeRepairSelection)
      fail("repair quality selection repeated an acquired objective");
    bool foundFirstEvidenceDomain = false;
    bool foundSecondEvidenceDomain = false;
    std::vector<loom::dse::JointDesignQualityObservation *>
        sharedEvidenceObservations;
    for (loom::dse::JointDesignExecution &execution : repairExecutions)
      for (loom::dse::JointDesignQualityObservation &observation :
           execution.summary.qualityObservations) {
        if (observation.candidate == firstRepairMapping)
          foundFirstEvidenceDomain = true;
        else if (observation.candidate == promotedMapping)
          foundSecondEvidenceDomain = true;
        else
          continue;
        observation.evidence = sharedQualityEvidence;
        sharedEvidenceObservations.push_back(&observation);
      }
    if (!foundFirstEvidenceDomain || !foundSecondEvidenceDomain)
      fail("repair quality lost its selected Mapping observation");
    auto foreignEvidenceSelection =
        loom::dse::selectJointRepairMappingByQuality(
            repairExecutions, repairQuality, store);
    if (foreignEvidenceSelection)
      fail("repair quality accepted Evidence across pair domains");
    const std::string foreignEvidenceError =
        llvm::toString(foreignEvidenceSelection.takeError());
    if (!llvm::StringRef(foreignEvidenceError).contains("Evidence crossed"))
      fail("repair quality Evidence-domain rejection lost its reason");
    for (loom::dse::JointDesignQualityObservation *observation :
         sharedEvidenceObservations)
      observation->evidence.reset();
    loom::dse::CandidateMeasureObjectiveCatalogs runtimeCatalogs;
    runtimeCatalogs.dimensions = {
        {0, loom::ResolvedObjectiveDirection::Minimize, 0, 100},
        {1, loom::ResolvedObjectiveDirection::Minimize, 0, 100},
        {2, loom::ResolvedObjectiveDirection::Minimize, 0, 100}};
    runtimeCatalogs.weightedLevels = {
        {{{0, 1}}}, {{{1, 1}}}, {{{2, 1}}}};
    runtimeCatalogs.totalOrderings = {{{0, 1, 2}}};
    auto runtimeProgram =
        std::make_shared<const loom::dse::ObjectiveProgram>(
            take(loom::dse::ObjectiveProgram::getCandidateMeasures(
                runtimeCatalogs)));
    loom::dse::JointBoundedQualityPolicy runtimeRepairQuality = repairQuality;
    runtimeRepairQuality.objectiveProgram = runtimeProgram;
    runtimeRepairQuality.objectiveDimensionLabels = {
        "dfg_cycles", "cgra_cycles", "acc_core_count"};
    runtimeRepairQuality.provenanceDomain =
        loom::dse::JointDesignQualityProvenanceDomain::ApplicationRuntime;
    runtimeRepairQuality.paretoDimensions = {0, 1, 2};
    std::vector<loom::dse::JointDesignExecutionSummary> savedRuntimeSummaries;
    for (const loom::dse::JointDesignExecution &execution : repairExecutions)
      savedRuntimeSummaries.push_back(execution.summary);
    for (auto indexed : llvm::enumerate(repairExecutions)) {
      indexed.value().summary.qualityObjectiveDimensionLabels =
          runtimeRepairQuality.objectiveDimensionLabels;
      for (loom::dse::JointDesignQualityObservation &observation :
           indexed.value().summary.qualityObservations) {
        observation.provenance.rawMeasures = {
            loom::resolvedObjectiveInteger(indexed.index() + 1),
            loom::resolvedObjectiveInteger(2),
            loom::resolvedObjectiveInteger(3)};
        observation.provenance.resourceCoreCost = 3;
        loom::dse::ObjectiveVector objective = runtimeProgram->makeVector();
        if (llvm::Error error = runtimeProgram->evaluateCandidateMeasures(
                observation.provenance.rawMeasures, objective))
          fail(llvm::toString(std::move(error)));
        observation.objectiveCodes.assign(objective.codes().begin(),
                                          objective.codes().end());
      }
    }
    repairExecutions[0].summary.qualityDisposition =
        loom::dse::JointDesignQualityDisposition::Unsupported;
    repairExecutions[0].summary.qualityIncompleteCandidate =
        firstRepairMapping;
    repairExecutions[0].summary.selectedMapping.reset();
    repairExecutions[0].summary.selectedPlanOrdinal.reset();
    for (loom::dse::JointDesignQualityObservation &observation :
         repairExecutions[0].summary.qualityObservations)
      if (observation.candidate == firstRepairMapping) {
        observation.objectiveCodes.clear();
        observation.incompleteReason =
            loom::dse::JointDesignQualityIncompleteReason::Unsupported;
        observation.provenance.rawMeasures.clear();
      }
    repairExecutions[1]
        .summary.qualityObservations.front()
        .provenance.resourceCoreCost.reset();
    auto missingLaterResource = loom::dse::selectJointRepairMappingByQuality(
        repairExecutions, runtimeRepairQuality, store);
    if (missingLaterResource)
      fail("repair quality returned incomplete before validating a later "
           "runtime provenance");
    const std::string missingLaterResourceError =
        llvm::toString(missingLaterResource.takeError());
    if (!llvm::StringRef(missingLaterResourceError)
             .contains("lost its exact resource count"))
      fail("repair quality runtime provenance rejection lost its reason");
    for (auto indexed : llvm::enumerate(repairExecutions))
      indexed.value().summary =
          std::move(savedRuntimeSummaries[indexed.index()]);
    auto savedForeignMappedPairs = repairExecutions[1].mappedPairs;
    auto savedForeignSelectedMapping =
        repairExecutions[1].summary.selectedMapping;
    auto savedForeignQualityObservations =
        repairExecutions[1].summary.qualityObservations;
    repairExecutions[1].mappedPairs = repairExecutions[0].mappedPairs;
    repairExecutions[1].summary.selectedMapping =
        repairExecutions[0].summary.selectedMapping;
    repairExecutions[1].summary.qualityObservations =
        repairExecutions[0].summary.qualityObservations;
    auto &foreignPair = repairExecutions[1].mappedPairs.front().pair;
    foreignPair.system = foreignPair.system == system ? alternateSystem : system;
    auto foreignDomainSelection =
        loom::dse::selectJointRepairMappingByQuality(
            repairExecutions, repairQuality, store);
    if (foreignDomainSelection)
      fail("repair quality accepted one Mapping across pair domains");
    const std::string foreignDomainError =
        llvm::toString(foreignDomainSelection.takeError());
    if (!llvm::StringRef(foreignDomainError).contains("crossed pair"))
      fail("repair quality pair-domain rejection lost its reason");
    repairExecutions[1].mappedPairs = std::move(savedForeignMappedPairs);
    repairExecutions[1].summary.selectedMapping = savedForeignSelectedMapping;
    repairExecutions[1].summary.qualityObservations =
        std::move(savedForeignQualityObservations);
    repairExecutions[0].summary.qualityDisposition =
        loom::dse::JointDesignQualityDisposition::Unsupported;
    repairExecutions[0].summary.qualityIncompleteCandidate = firstRepairMapping;
    repairExecutions[0].summary.selectedMapping.reset();
    repairExecutions[0].summary.selectedPlanOrdinal.reset();
    for (loom::dse::JointDesignQualityObservation &observation :
         repairExecutions[0].summary.qualityObservations) {
      if (observation.candidate != firstRepairMapping)
        continue;
      observation.objectiveCodes.clear();
      observation.incompleteReason =
          loom::dse::JointDesignQualityIncompleteReason::Unsupported;
    }
    for (loom::dse::JointDesignQualityObservation &observation :
         repairExecutions[1].summary.qualityObservations) {
      if (observation.candidate != firstRepairMapping)
        continue;
      observation.objectiveCodes.clear();
      observation.incompleteReason =
          loom::dse::JointDesignQualityIncompleteReason::Unsupported;
    }
    const auto later = llvm::find_if(
        repairExecutions[1].summary.qualityObservations,
        [&](const loom::dse::JointDesignQualityObservation &observation) {
          return observation.candidate != firstRepairMapping &&
                 !observation.incompleteReason;
        });
    if (later == repairExecutions[1].summary.qualityObservations.end())
      fail("repair quality fixture has no later complete observation");
    auto &laterObservation = *later;
    const auto laterRawMeasures = laterObservation.provenance.rawMeasures;
    laterObservation.provenance.rawMeasures = {
        loom::resolvedObjectiveInteger(99)};
    auto invalidLaterObservation =
        loom::dse::selectJointRepairMappingByQuality(repairExecutions,
                                                     repairQuality, store);
    if (invalidLaterObservation)
      fail("repair quality returned incomplete before validating all "
           "executions");
    const std::string invalidLaterObservationError =
        llvm::toString(invalidLaterObservation.takeError());
    if (!llvm::StringRef(invalidLaterObservationError)
             .contains("raw measures disagree"))
      fail("repair quality later-provenance rejection lost its reason");
    laterObservation.provenance.rawMeasures = laterRawMeasures;
    auto incompleteRepair = take(loom::dse::selectJointRepairMappingByQuality(
        repairExecutions, repairQuality, store));
    const auto *incomplete =
        std::get_if<loom::dse::JointRepairQualityIncomplete>(
            &incompleteRepair);
    if (!incomplete ||
        incomplete->executionOrdinal != 0 ||
        incomplete->incomplete.reason !=
            loom::dse::JointDesignQualityIncompleteReason::Unsupported ||
        incomplete->incomplete.candidate != firstRepairMapping)
      fail("repair quality selection ranked a typed incomplete candidate");
    if (*qualityAcquisitionCount != acquisitionsBeforeRepairSelection)
      fail("repair quality incomplete selection repeated acquisition");
    repairExecutions[0].summary.qualityObservations.clear();
    auto missingIncompleteObservation =
        loom::dse::selectJointRepairMappingByQuality(repairExecutions,
                                                     repairQuality, store);
    if (missingIncompleteObservation)
      fail("repair quality accepted a missing incomplete observation");
    const std::string missingObservationError =
        llvm::toString(missingIncompleteObservation.takeError());
    if (!llvm::StringRef(missingObservationError)
             .contains("has no observation"))
      fail("repair quality missing-observation rejection lost its reason");
  }

  auto mappedDataflow = take(dataflow::importCanonicalDataflow(
      plan.frontier.pairs.front().software.dataflow, store));
  auto mappedDataflowView = take(mappedDataflow.view());
  std::vector<dataflow::RootThreadLaunchRef> mappedRoots;
  for (const auto &root : mappedDataflowView.rootThreadLaunches())
    mappedRoots.push_back(root.ref);
  if (mappedRoots.size() != 1 ||
      systemView.artifact().accCoreOccurrences().size() < 2)
    fail("adjacent resource-time repair fixture lacks one root and two cores");
  loom::dse::JointDesignExecution parentExecution{
      std::move(execution), {{plan.frontier.pairs.front(), mappings}}, {}};
  parentExecution.summary.selectedMapping = mappings.front();
  parentExecution.summary.selectedPlanOrdinal = 0;
  parentExecution.summary.verifiedAlternatives = mappings.size();
  const auto targetModules =
      take(loom::dse::projectJointDesignTargetModules(system, store));
  std::vector<loom::pnr::SystemModuleCorrespondence>
      identityModuleCorrespondence;
  for (const auto &module : targetModules)
    identityModuleCorrespondence.push_back({module, module});
  loom::dse::HardwareImpactProjection systemOnlyImpact{system, system, {},
                                                       {},     {},     {}};
  systemOnlyImpact.family = loom::dse::HardwareMutationFamily::SystemTransport;
  systemOnlyImpact.locality = loom::dse::HardwareMutationLocality::LocalCone;
  systemOnlyImpact.system.kind = loom::dse::HardwareMappingImpactKind::Reopen;
  if (!systemView.transportResources().empty())
    systemOnlyImpact.system.transportRoots.push_back(
        systemView.transportResources().front());
  const auto preservedFrontier = take(loom::dse::rebaseJointMappingFrontier(
      plan, parentExecution, system, identityModuleCorrespondence,
      systemOnlyImpact, store));
  if (preservedFrontier.disposition !=
          loom::dse::JointMappingReuseDisposition::Preserved ||
      preservedFrontier.seed.techMappings.empty() ||
      preservedFrontier.seed.spatialMappings.empty() ||
      preservedFrontier.accounting.invalidatedTechMappings != 0 ||
      preservedFrontier.accounting.invalidatedSpatialMappings != 0)
    fail("System-only impact did not preserve lower Mapping layers");
  if (preservedFrontier.accounting.parentThreadBindingCount == 0 ||
      preservedFrontier.accounting.parentGraphBindingCount == 0 ||
      preservedFrontier.accounting.reopenedThreadBindingCount !=
          preservedFrontier.accounting.parentThreadBindingCount ||
      preservedFrontier.accounting.reopenedGraphBindingCount !=
          preservedFrontier.accounting.parentGraphBindingCount ||
      preservedFrontier.accounting.preservedThreadBindingCount != 0 ||
      preservedFrontier.accounting.preservedGraphBindingCount != 0)
    fail("System transport impact did not reopen its exact binding cone");

  auto rootlessTechImpact = systemOnlyImpact;
  rootlessTechImpact.tech.kind = loom::dse::HardwareMappingImpactKind::Reopen;
  auto rootlessTech = loom::dse::rebaseJointMappingFrontier(
      plan, parentExecution, system, identityModuleCorrespondence,
      rootlessTechImpact, store);
  if (rootlessTech ||
      llvm::toString(rootlessTech.takeError())
              .find("typed Tech impact has no realization root") ==
          std::string::npos)
    fail("rootless Tech reopen was not rejected");

  auto rootlessSpatialImpact = systemOnlyImpact;
  rootlessSpatialImpact.spatial.kind =
      loom::dse::HardwareMappingImpactKind::Reopen;
  auto rootlessSpatial = loom::dse::rebaseJointMappingFrontier(
      plan, parentExecution, system, identityModuleCorrespondence,
      rootlessSpatialImpact, store);
  if (rootlessSpatial ||
      llvm::toString(rootlessSpatial.takeError())
              .find("typed Spatial impact has no placement or route root") ==
          std::string::npos)
    fail("rootless Spatial reopen was not rejected");

  auto targetModule =
      take(loom::fabric::importEntireFabricRoot(targetModules.front(), store));
  if (targetModule.view().fifoOccurrences().empty())
    fail("FIFO feedback fixture has no physical FIFO");
  auto feedbackParentMapping =
      take(loom::mapping::importSystemMapping(mappings.front(), store));
  std::optional<loom::ArtifactRootReference> feedbackSpatialMapping;
  std::optional<loom::fabric::FabricFifoOccurrenceRef> feedbackFifo;
  for (const auto &reference : feedbackParentMapping.view()
                                   .executionBindings()
                                   .spatialMappingImports()) {
    auto spatial = take(loom::mapping::importSpatialMapping(reference, store));
    if (spatial.view().fabricIdentity() != targetModule.view().identity())
      continue;
    for (const auto fifo : targetModule.view().fifoOccurrences())
      if (loom::mapping::spatialMappingUsesFifoOccurrence(spatial.view(),
                                                          fifo)) {
        feedbackSpatialMapping = reference;
        feedbackFifo = fifo;
        break;
      }
    if (feedbackSpatialMapping)
      break;
  }
  if (!feedbackSpatialMapping || !feedbackFifo)
    fail("FIFO feedback fixture has no selected physical FIFO");
  std::optional<loom::fabric::FabricPeOccurrenceRef> operandPe;
  for (const auto pe : targetModule.view().peOccurrences())
    if (targetModule.view().peSchedule(pe) == ::fabric::Schedule::Temporal) {
      operandPe = pe;
      break;
    }
  if (!operandPe)
    fail("operand-buffer feedback fixture has no Temporal PE");
  const auto operandMode = targetModule.view().peOperandBufferMode(*operandPe);
  const std::uint32_t operandEntries =
      targetModule.view().peOperandBufferSize(*operandPe);
  if (!operandMode || operandEntries == 0 ||
      operandEntries == std::numeric_limits<std::uint32_t>::max())
    fail("operand-buffer feedback fixture has no growable Temporal PE");

  if (!mutationFamily.empty()) {
    const bool runEveryFamily = mutationFamily == allMutationFamilies;
    requireMutationDispositionCoverage();
    std::uint64_t repairOrdinal = 0;
    std::uint64_t executedFamilies = 0;
    const auto executeMutation =
        [&](llvm::StringRef label, loom::dse::JointHardwareMutationChild child)
        -> std::optional<loom::dse::JointHardwareMutationRepair> {
      if (child.impacts.empty())
        fail(label + " mutation has no typed impact");
      // Child materialization stays on every path so the cross-family fixture
      // lineage is identical regardless of which family this process executes.
      if (!runEveryFamily && label != mutationFamily)
        return std::nullopt;
      ++executedFamilies;
      llvm::SmallString<128> journal(temporary.path());
      llvm::sys::path::append(journal, "mutation-" +
                                           std::to_string(repairOrdinal++) +
                                           "-" + label.str());
      const std::string producer =
          "loom.test.hardware_mutation_matrix." + label.str() + ".v1";
      loom::dse::JointHardwareReopenRequest repairRequest{
          take(loom::dse::DseProducerSemanticBuildIdentity::get(producer)),
          journal.str().str(),
          {},
          loom::dse::JointDesignStoppingPolicy::FirstVerified,
          std::nullopt,
          std::nullopt,
          take(loom::dse::SiteCapacity::get(2, 0, 0)),
          take(loom::dse::PlanExecutionPolicy::get(
              2, take(loom::dse::SiteResourceClaim::get(1, 0, 0))))};
      // The matrix is the evidence owner for cold versus preserve-first
      // repair, so it asks for the independent cold oracle that ordinary
      // production repair does not run.
      repairRequest.coldComparisonBaseline = true;
      auto repair = take(loom::dse::executeJointHardwareMutationRepair(
          plan, parentExecution, policy, mappings.front(), std::move(child),
          std::move(repairRequest), store, blobs));
      const auto verified = [](const auto &statistics, std::size_t count) {
        return statistics.importRequests == count &&
               statistics.cacheMisses == count &&
               statistics.uniqueConstructions == count &&
               statistics.deterministicWork != 0 &&
               statistics.retainedBytes != 0;
      };
      if (repair.coldMappings.empty() || repair.incrementalMappings.empty() ||
          repair.coldExecution->summary.techMappingDispatchCount == 0 ||
          repair.coldExecution->summary.spatialPnrDispatchCount == 0 ||
          repair.coldExecution->summary.systemPnrDispatchCount == 0 ||
          repair.incrementalExecution.summary.systemPnrDispatchCount == 0 ||
          repair.coldExecution->summary.coldReopenWallTimeNanoseconds == 0 ||
          (repair.rebase.disposition ==
                   loom::dse::JointMappingReuseDisposition::ColdFallback
               ? repair.incrementalExecution.summary
                         .coldReopenWallTimeNanoseconds == 0
               : repair.incrementalExecution.summary
                         .incrementalReopenWallTimeNanoseconds == 0) ||
          !verified(repair.coldVerification, repair.coldMappings.size()) ||
          !verified(repair.incrementalVerification,
                    repair.incrementalMappings.size()))
        fail(label + " mutation did not execute and independently verify both "
                     "cold and preserve-first Mapping paths");
      if (llvm::Error error = loom::dse::validateJointMappingRebaseAccounting(
              repair.rebase.accounting))
        fail(label + " mutation has an open repair cone: " +
             llvm::toString(std::move(error)));
      const MutationFamilyContract &contract = mutationFamilyContract(label);
      if (repair.rebase.disposition != contract.rebase ||
          repair.systemDisposition != contract.system)
        fail(label + " mutation reached " +
             loom::dse::jointMappingReuseDispositionSpelling(
                 repair.rebase.disposition) +
             "/" +
             loom::dse::jointSystemMappingReuseDispositionSpelling(
                 repair.systemDisposition) +
             " instead of its declared " +
             loom::dse::jointMappingReuseDispositionSpelling(contract.rebase) +
             "/" +
             loom::dse::jointSystemMappingReuseDispositionSpelling(
                 contract.system));
      if (contract.rebase != loom::dse::JointMappingReuseDisposition::ColdFallback &&
          repair.rebase.accounting.preservedTechMappings == 0 &&
          repair.rebase.accounting.preservedSpatialMappings == 0 &&
          repair.rebase.accounting.repairedTechMappings == 0 &&
          repair.rebase.accounting.repairedSpatialMappings == 0)
        fail(label + " mutation reported reuse without a preserved or "
                     "repaired lower-layer Mapping");
      for (const auto *roots :
           {&repair.coldMappings, &repair.incrementalMappings})
        for (const loom::ArtifactRootReference &reference : *roots) {
          auto mapping =
              take(loom::mapping::importSystemMapping(reference, store));
          if (mapping.view().fabricIdentity() != repair.child.system.artifact)
            fail(label + " mutation Mapping names a foreign System");
        }
      // One durable line per family so a sharded run still reports the typed
      // dispositions its family reached.
      llvm::outs() << "mutation-family " << label << " rebase="
                   << loom::dse::jointMappingReuseDispositionSpelling(
                          repair.rebase.disposition)
                   << " system="
                   << loom::dse::jointSystemMappingReuseDispositionSpelling(
                          repair.systemDisposition);
      for (const auto &failure : repair.rebase.failures)
        llvm::outs() << " failure=" << static_cast<int>(failure.reason) << ":"
                     << failure.diagnostic;
      llvm::outs() << "\n";
      return repair;
    };

    const auto materializeModule =
        [&](llvm::StringRef label,
            loom::dse::SpatialMicroarchitectureDecisionDomain decision) {
          auto child = take(loom::dse::materializeJointModuleHardwareMutation(
              config, system, targetModules.front(), std::move(decision), store,
              blobs));
          if (child.system == system || child.impacts.size() != 1)
            fail(label + " did not materialize one distinct hardware child");
          return child;
        };
    const auto materializeSystem =
        [&](llvm::StringRef label, const loom::ArtifactRootReference &parent,
            loom::dse::SystemCompositionDecisionDomain decision,
            llvm::ArrayRef<loom::ArtifactRootReference> modules) {
          auto child = take(loom::dse::materializeJointSystemHardwareMutation(
              config, parent, std::move(decision), modules, store, blobs));
          if (child.system == parent || child.impacts.size() != 1)
            fail(label + " did not materialize one distinct hardware child");
          return child;
        };

    std::optional<loom::dse::JointHardwareMutationChild> fuChild;
    for (const auto pe : targetModule.view().peOccurrences()) {
      std::vector<loom::fabric::FabricFuOccurrenceRef> inventory;
      for (const auto fu : targetModule.view().fuOccurrences())
        if (targetModule.view().parentPeOf(fu) == pe)
          inventory.push_back(fu);
      for (const auto prototype : targetModule.view().fuOccurrences()) {
        if (targetModule.view().parentPeOf(prototype) == pe)
          continue;
        auto candidateInventory = inventory;
        candidateInventory.push_back(prototype);
        auto candidate = loom::dse::materializeJointModuleHardwareMutation(
            config, system, targetModules.front(),
            loom::dse::ChangeFuInventoryDomain{pe,
                                               {std::move(candidateInventory)}},
            store, blobs);
        if (!candidate) {
          llvm::consumeError(candidate.takeError());
          continue;
        }
        fuChild = std::move(*candidate);
        break;
      }
      if (fuChild)
        break;
    }
    if (!fuChild)
      fail("FU mutation fixture has no valid inventory rewrite");
    executeMutation("fu", std::move(*fuChild));

    const auto memory = targetModule.view().memoryOccurrences().front();
    const std::uint64_t memoryCapacity =
        targetModule.view().localMemoryServiceCapacityBytes(memory);
    if (memoryCapacity == 0 ||
        memoryCapacity == std::numeric_limits<std::uint64_t>::max())
      fail("memory mutation fixture has no growable Local Memory Service");
    auto memoryChild = materializeModule(
        "memory", loom::dse::ResizeMemoryDomain{memory, {memoryCapacity + 1}});
    const loom::dse::JointHardwareMutationChild memorySpatialChild =
        memoryChild;
    executeMutation("memory", std::move(memoryChild));

    auto fifoChild = materializeModule(
        "fifo", loom::dse::ResizeFifoDomain{
                    targetModule.view().fifoOccurrences().front(), {257}});
    const loom::dse::JointHardwareMutationChild combinedFirst = fifoChild;
    executeMutation("fifo", std::move(fifoChild));

    executeMutation("operand",
                    materializeModule(
                        "operand", loom::dse::ResizeTemporalOperandBufferDomain{
                                       *operandPe, {operandEntries + 1}}));

    std::optional<loom::fabric::FabricSwitchOccurrenceRef> temporalSwitch;
    for (const auto target : targetModule.view().switchOccurrences()) {
      if (targetModule.view().switchSchedule(target) ==
          ::fabric::Schedule::Temporal) {
        temporalSwitch = target;
        break;
      }
    }
    if (!temporalSwitch ||
        targetModule.view().switchRouteTableSize(*temporalSwitch) ==
            std::numeric_limits<std::uint32_t>::max())
      fail("switch mutation fixture has no growable Temporal switch");
    executeMutation(
        "switch",
        materializeModule(
            "switch",
            loom::dse::ResizeSwitchRouteTableDomain{
                *temporalSwitch,
                {static_cast<std::uint32_t>(
                    targetModule.view().switchRouteTableSize(*temporalSwitch) +
                    1)}}));

    const auto &moduleLineage =
        memorySpatialChild.executionBindingCorrespondence->modules();
    const auto replacedModule =
        llvm::find_if(moduleLineage, [&](const auto &entry) {
          return entry.parent == targetModules.front() &&
                 entry.child != entry.parent;
        });
    if (replacedModule == moduleLineage.end())
      fail("SpatialCore mutation fixture lost replacement Module lineage");
    const std::array replacementModules = {replacedModule->child};
    executeMutation("spatial-core",
                    materializeSystem(
                        "spatial-core", system,
                        loom::dse::ReplaceSpatialAttachmentDomain{
                            systemView.artifact().accCoreOccurrences().front(),
                            {replacedModule->child}},
                        replacementModules));

    executeMutation("acc-core-add",
                    materializeSystem(
                        "acc-core-add", system,
                        loom::dse::AddAccCoreDomain{
                            systemView.artifact().accCoreOccurrences().front(),
                            {targetModules.front()}},
                        targetModules));

    executeMutation(
        "acc-core-remove",
        materializeSystem(
            "acc-core-remove", system,
            loom::dse::RemoveAccCoreDomain{
                {systemView.artifact().accCoreOccurrences().front()}},
            targetModules));

    std::optional<loom::dse::JointHardwareMutationChild> transportChild;
    for (auto indexedFirst :
         llvm::enumerate(systemView.artifact().pointConnections())) {
      for (const auto &second :
           systemView.artifact().pointConnections().drop_front(
               indexedFirst.index() + 1)) {
        const auto &first = indexedFirst.value();
        if (!llvm::equal(
                systemView.artifact().transportEndpointType(first.source),
                systemView.artifact().transportEndpointType(
                    second.destination)) ||
            !llvm::equal(
                systemView.artifact().transportEndpointType(second.source),
                systemView.artifact().transportEndpointType(first.destination)))
          continue;
        auto firstDestination = first.destination;
        auto secondDestination = second.destination;
        if (loom::fabric::canonicalFabricBytes(secondDestination) <
            loom::fabric::canonicalFabricBytes(firstDestination))
          std::swap(firstDestination, secondDestination);
        auto candidate = loom::dse::materializeJointSystemHardwareMutation(
            config, system,
            loom::dse::SwapTransportConnectionSourcesDomain{
                firstDestination, {secondDestination}},
            targetModules, store, blobs);
        if (!candidate) {
          llvm::consumeError(candidate.takeError());
          continue;
        }
        transportChild = std::move(*candidate);
        break;
      }
      if (transportChild)
        break;
    }
    if (!transportChild)
      fail("transport mutation fixture has no legal alternate connection");
    executeMutation("transport", std::move(*transportChild));

    std::optional<loom::fabric::SystemServiceEndpointRef> memoryEndpoint;
    for (const auto &attachment : systemView.spatialAttachments()) {
      if (attachment.spatialEndpoint.memory() && attachment.serviceEndpoint) {
        memoryEndpoint = *attachment.serviceEndpoint;
        break;
      }
    }
    if (!memoryEndpoint)
      fail("service mutation fixture has no memory endpoint");
    const auto *endpointOwner =
        systemView.serviceEndpointOwner(*memoryEndpoint);
    const auto *memoryOwner =
        endpointOwner ? std::get_if<loom::fabric::FabricMemoryServiceRef>(
                            &endpointOwner->owner().payload)
                      : nullptr;
    const auto *memoryService =
        memoryOwner ? std::get_if<loom::fabric::SystemMemoryServiceRef>(
                          &memoryOwner->payload)
                    : nullptr;
    const auto *memoryContract =
        memoryService ? systemView.memoryService(*memoryService) : nullptr;
    if (!memoryContract || memoryContract->regions().empty() ||
        memoryContract->regions().front().sizeBytes ==
            std::numeric_limits<std::uint64_t>::max())
      fail("service mutation fixture has no growable memory region");
    executeMutation("service",
                    materializeSystem(
                        "service", system,
                        loom::dse::ResizeSystemMemoryRegionDomain{
                            *memoryService,
                            0,
                            {memoryContract->regions().front().sizeBytes + 1}},
                        targetModules));

    auto combinedSystem =
        take(loom::fabric::importEntireFabricRoot(combinedFirst.system, store));
    auto combinedSystemView =
        take(loom::fabric::requireSystemRoot(combinedSystem.view()));
    auto combinedModules = take(loom::dse::projectJointDesignTargetModules(
        combinedFirst.system, store));
    auto combinedSecond = materializeSystem(
        "combined-tail", combinedFirst.system,
        loom::dse::AddAccCoreDomain{
            combinedSystemView.artifact().accCoreOccurrences().front(),
            {combinedModules.front()}},
        combinedModules);
    auto combined = take(loom::dse::composeJointHardwareMutationChildren(
        std::move(combinedFirst), std::move(combinedSecond), store));
    if (combined.impacts.size() != 2)
      fail("combined mutation lost ordered component impacts");
    executeMutation("combined", std::move(combined));
    const std::uint64_t expectedFamilies =
        runEveryFamily ? std::size(mutationFamilyContracts) : 1;
    if (executedFamilies != expectedFamilies)
      fail("mutation family selector matched " +
           std::to_string(executedFamilies) + " of " +
           std::to_string(expectedFamilies) + " families: " +
           mutationFamily.str());
    return;
  }

  std::optional<::fabric::OperandBufferMode> separatedMode;
  if (*operandMode == ::fabric::OperandBufferMode::AllFuShare)
    separatedMode = ::fabric::OperandBufferMode::PerInputPort;
  else if (*operandMode == ::fabric::OperandBufferMode::PerInputPort)
    separatedMode = ::fabric::OperandBufferMode::PerInstruction;
  loom::ArtifactRootReference operandTech{
      loom::mapping::mappingArtifactSchema.identity.str(),
      loom::mapping::mappingArtifactSchema.version,
      take(loom::mapping::importSpatialMapping(*feedbackSpatialMapping, store))
          .view()
          .techMappingIdentity()};
  loom::dse::SpatialOperandQueueRuntimeFeedback operandFeedback;
  operandFeedback.parentMapping = mappings.front();
  operandFeedback.owners = loom::sim::CgraExecutionOwnerReferences{
      plan.frontier.pairs.front().software.dataflow, targetModules.front(),
      operandTech, *feedbackSpatialMapping};
  operandFeedback.disposition =
      loom::dse::SpatialOperandQueueRuntimeFeedbackDisposition::Exact;
  operandFeedback.reason =
      loom::dse::SpatialOperandQueueRuntimeFeedbackReason::ExactClosedWait;
  operandFeedback.repairTarget = loom::dse::SpatialOperandBufferRepairTarget{
      *operandPe, *operandMode, operandEntries, separatedMode,
      operandEntries + 1};
  if (runOperandHardwareRepair) {
    llvm::SmallString<128> operandJournal(temporary.path());
    llvm::sys::path::append(operandJournal, "operand-buffer-repair");
    const auto operandRepair =
        take(loom::dse::executeSpatialOperandBufferHardwareFeedbackReopen(
            plan, parentExecution, policy, operandFeedback,
            {take(loom::dse::DseProducerSemanticBuildIdentity::get(
                 "loom.test.spatial_operand_buffer_feedback.v1")),
             operandJournal.str().str(),
             {},
             loom::dse::JointDesignStoppingPolicy::FirstVerified,
             std::nullopt,
             std::nullopt,
             take(loom::dse::SiteCapacity::get(2, 0, 0)),
             take(loom::dse::PlanExecutionPolicy::get(
                 2, take(loom::dse::SiteResourceClaim::get(1, 0, 0))))},
            store, blobs));
    if (operandRepair.childSystems.empty() ||
        operandRepair.executions.empty() ||
        operandRepair.reuseDispositions.empty())
      fail("exact operand-buffer feedback did not materialize a bounded child");
    if (separatedMode && operandRepair.childSystems.size() != 2)
      fail("exact operand-buffer feedback did not retain both bounded "
           "mode/depth alternatives");
    const std::uint64_t expectedOperandCandidateLimit = separatedMode ? 2 : 1;
    if (operandRepair.candidateLimit != expectedOperandCandidateLimit ||
        operandRepair.candidatesPlanned != operandRepair.candidatesReserved ||
        operandRepair.candidatesReserved !=
            operandRepair.candidatesConsumed +
                operandRepair.candidatesRejected +
                operandRepair.candidatesCancelled)
      fail("operand-buffer hardware child budget ledger is not closed: limit=" +
           llvm::Twine(operandRepair.candidateLimit) +
           " planned=" + llvm::Twine(operandRepair.candidatesPlanned) +
           " reserved=" + llvm::Twine(operandRepair.candidatesReserved) +
           " consumed=" + llvm::Twine(operandRepair.candidatesConsumed) +
           " rejected=" + llvm::Twine(operandRepair.candidatesRejected) +
           " cancelled=" + llvm::Twine(operandRepair.candidatesCancelled));
    bool operandMappingVerified = false;
    for (std::size_t ordinal = 0; ordinal != operandRepair.executions.size();
         ++ordinal) {
      for (const auto &pair : operandRepair.executions[ordinal].mappedPairs)
        for (const auto &mapping : pair.systemMappings) {
          auto imported =
              take(loom::mapping::importSystemMapping(mapping, store));
          if (imported.view().fabricIdentity() !=
              operandRepair.childSystems[ordinal].artifact)
            fail("operand-buffer child Mapping names the parent System");
          operandMappingVerified = true;
        }
    }
    if (!operandMappingVerified)
      fail("exact operand-buffer feedback produced no verified SystemMapping");
    auto incompleteOperandFeedback = operandFeedback;
    incompleteOperandFeedback.disposition = loom::dse::
        SpatialOperandQueueRuntimeFeedbackDisposition::ProofNotEstablished;
    const auto rejectedOperandRepair =
        take(loom::dse::executeSpatialOperandBufferHardwareFeedbackReopen(
            plan, parentExecution, policy, incompleteOperandFeedback,
            {take(loom::dse::DseProducerSemanticBuildIdentity::get(
                 "loom.test.spatial_operand_buffer_feedback.negative.v1")),
             operandJournal.str().str(),
             {},
             loom::dse::JointDesignStoppingPolicy::FirstVerified,
             std::nullopt,
             std::nullopt,
             take(loom::dse::SiteCapacity::get(2, 0, 0)),
             take(loom::dse::PlanExecutionPolicy::get(
                 2, take(loom::dse::SiteResourceClaim::get(1, 0, 0))))},
            store, blobs));
    if (!rejectedOperandRepair.childSystems.empty() ||
        !rejectedOperandRepair.executions.empty())
      fail("incomplete operand-buffer feedback synthesized a hardware child");
  }

  if (qualityRuns("operand") && separatedMode) {
    if (!incompleteRepairQuality)
      fail("bounded operand cap fixture lost its quality policy");
    auto cappedOperandQuality = *incompleteRepairQuality;
    cappedOperandQuality.maximumHardwareRepairProbes = 1;
    llvm::SmallString<128> cappedOperandJournal(temporary.path());
    llvm::sys::path::append(cappedOperandJournal,
                            "operand-buffer-repair-capped");
    const auto cappedOperandRepair =
        take(loom::dse::executeSpatialOperandBufferHardwareFeedbackReopen(
            plan, parentExecution, policy, operandFeedback,
            {take(loom::dse::DseProducerSemanticBuildIdentity::get(
                 "loom.test.spatial_operand_buffer_feedback.capped.v1")),
             cappedOperandJournal.str().str(),
             {},
             loom::dse::JointDesignStoppingPolicy::BoundedQuality,
             std::move(cappedOperandQuality),
             std::nullopt,
             take(loom::dse::SiteCapacity::get(2, 0, 0)),
             take(loom::dse::PlanExecutionPolicy::get(
                 2, take(loom::dse::SiteResourceClaim::get(1, 0, 0))))},
            store, blobs));
    if (cappedOperandRepair.candidateLimit != 1 ||
        cappedOperandRepair.candidatesPlanned != 1 ||
        cappedOperandRepair.candidatesReserved != 1 ||
        cappedOperandRepair.candidatesConsumed +
                cappedOperandRepair.candidatesRejected +
                cappedOperandRepair.candidatesCancelled !=
            cappedOperandRepair.candidatesReserved ||
        cappedOperandRepair.childSystems.size() > 1 ||
        cappedOperandRepair.executions.size() > 1)
      fail("bounded operand hardware repair exceeded the shared probe cap");

    auto invalidOperandQuality = *incompleteRepairQuality;
    invalidOperandQuality.maximumHardwareRepairProbes = 0;
    auto invalidOperandRepair =
        loom::dse::executeSpatialOperandBufferHardwareFeedbackReopen(
            plan, parentExecution, policy, operandFeedback,
            {take(loom::dse::DseProducerSemanticBuildIdentity::get(
                 "loom.test.spatial_operand_buffer_feedback.invalid.v1")),
             cappedOperandJournal.str().str(),
             {},
             loom::dse::JointDesignStoppingPolicy::BoundedQuality,
             std::move(invalidOperandQuality),
             std::nullopt,
             take(loom::dse::SiteCapacity::get(2, 0, 0)),
             take(loom::dse::PlanExecutionPolicy::get(
                 2, take(loom::dse::SiteResourceClaim::get(1, 0, 0))))},
            store, blobs);
    if (invalidOperandRepair)
      fail("bounded operand repair accepted a zero probe limit");
    const std::string invalidOperandMessage =
        llvm::toString(invalidOperandRepair.takeError());
    if (!llvm::StringRef(invalidOperandMessage)
             .contains("positive probe limit"))
      fail("bounded operand zero-limit rejection lost its typed reason");
  }

  auto transportSpatial =
      take(loom::mapping::importSpatialMapping(*feedbackSpatialMapping, store));
  auto transportDataflow = take(dataflow::importCanonicalDataflow(
      plan.frontier.pairs.front().software.dataflow, store));
  auto transportDataflowView = take(transportDataflow.view());
  auto transportTech =
      take(loom::mapping::importTechMapping(operandTech, store));
  std::optional<dataflow::CanonicalGraphProducerEndpointRef> transportProducer;
  std::optional<loom::fabric::FabricPhysicalTraversalRef> transportTraversal;
  std::optional<std::uint64_t> transportActorOrdinal;
  for (const auto &route : transportSpatial.view().routeTrees()) {
    const auto *producer =
        std::get_if<dataflow::ActorTokenResultRef>(&route.logicalNet);
    if (!producer)
      continue;
    const auto node = llvm::find_if(route.nodes, [](const auto &candidate) {
      return candidate.incomingTraversal.has_value();
    });
    if (node == route.nodes.end())
      continue;
    auto actor = take(transportDataflowView.resolve(producer->actor));
    std::uint64_t graphLocalOrdinal = 0;
    bool foundActor = false;
    for (const auto &candidate : transportDataflowView.actors()) {
      if (candidate.graph != actor.graph)
        continue;
      if (candidate.ref == producer->actor) {
        foundActor = true;
        break;
      }
      ++graphLocalOrdinal;
    }
    if (!foundActor)
      fail("transport feedback producer has no graph-local actor ordinal");
    transportProducer = route.logicalNet;
    transportTraversal = *node->incomingTraversal;
    transportActorOrdinal = graphLocalOrdinal;
    break;
  }
  if (!transportProducer || !transportTraversal || !transportActorOrdinal)
    fail("transport feedback fixture has no routed actor result");
  loom::sim::CgraClosedWaitSetDiagnostic exactTransportWait;
  exactTransportWait.ownerReferences = loom::sim::CgraExecutionOwnerReferences{
      plan.frontier.pairs.front().software.dataflow, targetModules.front(),
      operandTech, *feedbackSpatialMapping};
  loom::sim::CgraClosedWaitSetDiagnostic::Transfer transportTransfer;
  transportTransfer.bindingOrdinal = 0;
  transportTransfer.occurrenceOrdinal = 0;
  transportTransfer.producerActorOrdinal = *transportActorOrdinal;
  transportTransfer.blocked = true;
  transportTransfer.blockingActorOrdinal = *transportActorOrdinal;
  transportTransfer.producer = *transportProducer;
  transportTransfer.blockingTraversals.push_back(*transportTraversal);
  exactTransportWait.transfers.push_back(std::move(transportTransfer));
  exactTransportWait.actorWaitCycle.push_back(
      {*transportActorOrdinal, *transportActorOrdinal,
       loom::sim::CgraClosedWaitSetDiagnostic::ActorWaitKind::
           OutputBackpressure});
  const auto transportFeedback =
      take(loom::dse::deriveSpatialTransportRuntimeFeedback(
          mappings.front(), exactTransportWait, store));
  if (transportFeedback.disposition !=
          loom::dse::SpatialTransportRuntimeFeedbackDisposition::Exact ||
      transportFeedback.alternatives.size() != 1 ||
      transportFeedback.alternatives.front().producer != *transportProducer ||
      transportFeedback.alternatives.front().forbiddenTraversal !=
          *transportTraversal)
    fail("exact storage wait did not produce one canonical reroute");
  std::vector<loom::fabric::FabricPhysicalTraversalRef> transportDomain;
  for (const auto &traversal : targetModule.view().admittedTraversals())
    if (traversal != *transportTraversal)
      transportDomain.push_back(traversal);
  auto transportConstraints =
      take(loom::mapping::finalizeSpatialNetTraversalDomainConstraintSet(
          transportDataflowView, transportTech.view(), targetModule.view(),
          *transportProducer, transportDomain, store));
  llvm::Error parentConstraintAdmission =
      loom::mapping::admitSpatialMappingConstraints(
          transportDataflowView, transportTech.view(), targetModule.view(),
          transportConstraints.view(), transportSpatial.view());
  if (!parentConstraintAdmission)
    fail("reroute constraint admitted the blocked parent RouteTree");
  llvm::consumeError(std::move(parentConstraintAdmission));
  if (runTransportRepair) {
    llvm::SmallString<128> transportJournal(temporary.path());
    llvm::sys::path::append(transportJournal, "transport-runtime-repair");
    const auto transportRepair =
        take(loom::dse::executeSpatialTransportRuntimeRepair(
            plan, parentExecution, policy, transportFeedback,
            {take(loom::dse::DseProducerSemanticBuildIdentity::get(
                 "loom.test.spatial_transport_feedback.v1")),
             transportJournal.str().str(),
             {},
             loom::dse::JointDesignStoppingPolicy::FirstVerified,
             std::nullopt,
             std::nullopt,
             take(loom::dse::SiteCapacity::get(2, 0, 0)),
             take(loom::dse::PlanExecutionPolicy::get(
                 2, take(loom::dse::SiteResourceClaim::get(1, 0, 0))))},
            store, blobs));
    if (transportRepair.candidateLimit != 1 ||
        transportRepair.candidatesPlanned != 1 ||
        transportRepair.candidatesReserved != 1 ||
        transportRepair.candidatesConsumed +
                transportRepair.candidatesRejected +
                transportRepair.candidatesCancelled !=
            1 ||
        transportRepair.constraintSets.size() != 1 ||
        transportRepair.executions.size() != 1 ||
        transportRepair.childSystems !=
            std::vector<loom::ArtifactRootReference>{system} ||
        transportRepair.reuseDispositions !=
            std::vector<loom::dse::JointMappingReuseDisposition>{
                loom::dse::JointMappingReuseDisposition::ColdFallback} ||
        transportRepair.executions.front().summary.techMappingDispatchCount !=
            0 ||
        transportRepair.executions.front().summary.spatialPnrDispatchCount != 1)
      fail("bounded transport reroute did not use the constrained Spatial "
           "provider with a closed cold-fallback ledger");
    const auto &transportExecution = transportRepair.executions.front();
    if (!transportExecution.summary.selectedMapping ||
        transportExecution.summary.selectedPlanOrdinal !=
            std::optional<std::uint64_t>(0) ||
        !llvm::any_of(transportExecution.mappedPairs, [&](const auto &pair) {
          return llvm::is_contained(
              pair.systemMappings,
              *transportExecution.summary.selectedMapping);
        }))
      fail("first-verified transport repair did not publish its selected "
           "Mapping");
  }

  loom::sim::CgraClosedWaitSetDiagnostic exactFifoWait;
  exactFifoWait.pendingActorFirings = 1;
  exactFifoWait.pendingTransfers = 1;
  exactFifoWait.pendingPhysicalActions = 1;
  exactFifoWait.actorFirings.push_back({0, 0, 0, 1, 0, true, false});
  loom::sim::CgraClosedWaitSetDiagnostic::Transfer blockedTransfer;
  blockedTransfer.bindingOrdinal = 0;
  blockedTransfer.occurrenceOrdinal = 0;
  blockedTransfer.producerActorOrdinal = 0;
  blockedTransfer.blocked = true;
  blockedTransfer.blockingActorOrdinal = 0;
  blockedTransfer.blockingFifoOccurrence = *feedbackFifo;
  // The claimed capacity must match the physical FIFO: the feedback path grows
  // the occurrence to capacity + 1, which must not be a no-op resize.
  const std::uint32_t feedbackFifoCapacity =
      loom::adg::builtinSmallTarget.scale.interconnectFifoDepth;
  blockedTransfer.blockingStorageOccupancy = feedbackFifoCapacity;
  blockedTransfer.blockingStorageCapacity = feedbackFifoCapacity;
  exactFifoWait.transfers.push_back(std::move(blockedTransfer));
  exactFifoWait.physicalActions.push_back(
      {0, 0, 0, 0, true, true, true, true, false});
  exactFifoWait.transferWaitCycle.push_back({0, 0, 0, 0, 0});
  const auto exactFifoFeedback =
      take(loom::dse::deriveSpatialFifoRuntimeFeedback(
          mappings.front(), *feedbackSpatialMapping, exactFifoWait, store));
  if (exactFifoFeedback.disposition !=
          loom::dse::SpatialFifoRuntimeFeedbackDisposition::Exact ||
      exactFifoFeedback.minimumCandidateDepth != feedbackFifoCapacity + 1 ||
      exactFifoFeedback.occupancy != feedbackFifoCapacity ||
      exactFifoFeedback.capacity != feedbackFifoCapacity)
    fail("exact FIFO wait did not admit the minimal hardware candidate");

  auto crossTagFifoWait = exactFifoWait;
  using ClosedWait = loom::sim::CgraClosedWaitSetDiagnostic;
  ClosedWait::WaitEdge orderWait;
  orderWait.from = ClosedWait::WaitOwnerKey{
      ClosedWait::WaitActorFiringKey{0, 0}};
  orderWait.to = ClosedWait::WaitOwnerKey{
      ClosedWait::WaitStorageQueueKey{
          ClosedWait::WaitStorageDomain::TraversalStorage, 0,
          ClosedWait::WaitQueueClass::global()}};
  orderWait.kind = ClosedWait::WaitEdgeKind::StorageOrder;
  orderWait.fifoOccurrence = *feedbackFifo;
  orderWait.awaitedTagValue = llvm::APInt(4, 1);
  orderWait.headTagValue = llvm::APInt(4, 2);
  ClosedWait::WaitEdge consumerWait;
  consumerWait.from = orderWait.to;
  consumerWait.to = orderWait.from;
  consumerWait.kind = ClosedWait::WaitEdgeKind::StorageConsumer;
  crossTagFifoWait.waitCertificate = {orderWait, consumerWait};
  const auto crossTagFeedback =
      take(loom::dse::deriveSpatialFifoRuntimeFeedback(
          mappings.front(), *feedbackSpatialMapping, crossTagFifoWait, store));
  if (crossTagFeedback.disposition !=
          loom::dse::SpatialFifoRuntimeFeedbackDisposition::Exact ||
      crossTagFeedback.reason !=
          loom::dse::SpatialFifoRuntimeFeedbackReason::
              ExactCrossTagGlobalHolCycle ||
      crossTagFeedback.currentQueueDiscipline !=
          ::fabric::FifoQueueDiscipline::StrictFifo ||
      crossTagFeedback.candidateQueueDiscipline !=
          ::fabric::FifoQueueDiscipline::PerTagVirtualChannel ||
      crossTagFeedback.minimumCandidateDepth)
    fail("cross-tag global HOL did not admit the VC hardware candidate");
  if (qualityRuns("fifo")) {
    if (!incompleteRepairQuality)
      fail("quality-promotion fixture lost its incomplete repair policy");
    llvm::SmallString<128> incompleteFifoJournal(temporary.path());
    llvm::sys::path::append(incompleteFifoJournal,
                            "fifo-hardware-feedback-quality-incomplete");
    const auto incompleteQualityFifo =
        take(loom::dse::executeSpatialFifoHardwareFeedbackReopen(
            plan, parentExecution, policy, exactFifoFeedback,
            {take(loom::dse::DseProducerSemanticBuildIdentity::get(
                 "loom.test.spatial_fifo_feedback.quality_incomplete.v1")),
             incompleteFifoJournal.str().str(),
             {},
             loom::dse::JointDesignStoppingPolicy::BoundedQuality,
             *incompleteRepairQuality,
             std::nullopt,
             take(loom::dse::SiteCapacity::get(2, 0, 0)),
             take(loom::dse::PlanExecutionPolicy::get(
                 2, take(loom::dse::SiteResourceClaim::get(1, 0, 0))))},
            store, blobs));
    if (incompleteQualityFifo.executions.size() != 1)
      fail("bounded incomplete FIFO repair lost its exact child execution");
    const auto &incompleteExecution = incompleteQualityFifo.executions.front();
    const bool incompleteFifoHasMapping =
        llvm::any_of(incompleteExecution.mappedPairs, [](const auto &pair) {
          return !pair.systemMappings.empty();
        });
    if (!incompleteFifoHasMapping ||
        incompleteExecution.summary.qualityDisposition !=
            loom::dse::JointDesignQualityDisposition::Unsupported ||
        incompleteExecution.summary.selectedMapping ||
        incompleteExecution.summary.selectedPlanOrdinal)
      fail("bounded incomplete FIFO repair retained a selected Mapping");
  }
  if (runFifoHardwareRepair) {
    llvm::SmallString<128> fifoJournal(temporary.path());
    llvm::sys::path::append(fifoJournal, "fifo-hardware-feedback");
    const auto fifoHardwareRepair =
        take(loom::dse::executeSpatialFifoHardwareFeedbackReopen(
            plan, parentExecution, policy, exactFifoFeedback,
            {take(loom::dse::DseProducerSemanticBuildIdentity::get(
                 "loom.test.spatial_fifo_feedback.v1")),
             fifoJournal.str().str(),
             {},
             loom::dse::JointDesignStoppingPolicy::FirstVerified,
             std::nullopt,
             std::nullopt,
             take(loom::dse::SiteCapacity::get(2, 0, 0)),
             take(loom::dse::PlanExecutionPolicy::get(
                 2, take(loom::dse::SiteResourceClaim::get(1, 0, 0))))},
            store, blobs));
    if (fifoHardwareRepair.childSystems.size() != 1 ||
        fifoHardwareRepair.executions.size() != 1 ||
        fifoHardwareRepair.reuseDispositions.size() != 1 ||
        fifoHardwareRepair.childSystems.front() == system)
      fail("exact FIFO feedback did not materialize one typed System child");
    if (fifoHardwareRepair.candidateLimit != 1 ||
        fifoHardwareRepair.candidatesPlanned !=
            fifoHardwareRepair.candidatesReserved ||
        fifoHardwareRepair.candidatesReserved !=
            fifoHardwareRepair.candidatesConsumed +
                fifoHardwareRepair.candidatesRejected +
                fifoHardwareRepair.candidatesCancelled)
      fail("FIFO hardware child budget ledger is not closed");
    std::vector<loom::ArtifactRootReference> fifoChildMappings;
    for (const auto &pair : fifoHardwareRepair.executions.front().mappedPairs)
      fifoChildMappings.insert(fifoChildMappings.end(),
                               pair.systemMappings.begin(),
                               pair.systemMappings.end());
    if (fifoChildMappings.empty())
      fail("exact FIFO hardware child produced no verified SystemMapping");
    const auto &repairSummary = fifoHardwareRepair.executions.front().summary;
    if (repairSummary.parentSpatialDecisions == 0 ||
        repairSummary.repairedTechDecisions == 0 ||
        repairSummary.parentRouteNodeCount == 0 ||
        (fifoHardwareRepair.reuseDispositions.front() ==
                 loom::dse::JointMappingReuseDisposition::ColdFallback
             ? (repairSummary.repairedSpatialDecisions != 0 ||
                repairSummary.coldReopenWallTimeNanoseconds == 0 ||
                repairSummary.reopenedSpatialDecisions == 0)
             : repairSummary.repairedSpatialDecisions == 0))
      fail("FIFO hardware repair did not expose decision and route-cone "
           "accounting");
    for (const auto &reference : fifoChildMappings) {
      auto childMapping =
          take(loom::mapping::importSystemMapping(reference, store));
      if (childMapping.view().fabricIdentity() !=
          fifoHardwareRepair.childSystems.front().artifact)
        fail("FIFO hardware repair Mapping names the parent System");
    }
  }
  auto incompleteFifoWait = exactFifoWait;
  incompleteFifoWait.transferWaitCycle.clear();
  const auto incompleteFifoFeedback =
      take(loom::dse::deriveSpatialFifoRuntimeFeedback(
          mappings.front(), *feedbackSpatialMapping, incompleteFifoWait,
          store));
  if (incompleteFifoFeedback.disposition !=
          loom::dse::SpatialFifoRuntimeFeedbackDisposition::
              ProofNotEstablished ||
      incompleteFifoFeedback.reason !=
          loom::dse::SpatialFifoRuntimeFeedbackReason::MissingWaitCycle ||
      incompleteFifoFeedback.minimumCandidateDepth)
    fail("probe-incomplete FIFO wait synthesized a hardware child");
  if (runFifoHardwareRepair)
    return;
  const auto moduleRoot =
      take(loom::fabric::FabricModulePhysicalOwnerRef::create(*feedbackFifo));
  loom::dse::HardwareImpactProjection localSpatialImpact{
      targetModules.front(), system, {}, {}, {}, {}};
  localSpatialImpact.family = loom::dse::HardwareMutationFamily::SpatialFifo;
  localSpatialImpact.locality = loom::dse::HardwareMutationLocality::LocalCone;
  localSpatialImpact.tech.kind = loom::dse::HardwareMappingImpactKind::Rebase;
  localSpatialImpact.spatial.kind =
      loom::dse::HardwareMappingImpactKind::Reopen;
  localSpatialImpact.spatial.placementRoots.push_back(moduleRoot);
  localSpatialImpact.moduleEntities =
      identityModuleEntityCorrespondence(targetModule.view());
  const auto localRepairFrontier = take(loom::dse::rebaseJointMappingFrontier(
      plan, parentExecution, system, identityModuleCorrespondence,
      localSpatialImpact, store));
  if (localRepairFrontier.disposition !=
          loom::dse::JointMappingReuseDisposition::Preserved ||
      localRepairFrontier.seed.techMappings.empty() ||
      localRepairFrontier.seed.spatialMappings.empty() ||
      localRepairFrontier.accounting.invalidatedSpatialMappings != 0 ||
      localRepairFrontier.accounting.repairedSpatialMappings == 0)
    fail("typed local Spatial impact did not preserve and revalidate its "
         "selected cone");

  auto impactSpatial =
      take(loom::mapping::importSpatialMapping(*feedbackSpatialMapping, store));
  std::optional<loom::fabric::FabricFifoOccurrenceRef> unusedFifo;
  for (const auto fifo : targetModule.view().fifoOccurrences())
    if (!loom::mapping::spatialMappingUsesFifoOccurrence(impactSpatial.view(),
                                                         fifo)) {
      unusedFifo = fifo;
      break;
    }
  if (!unusedFifo)
    fail("mapping-reuse fixture has no unused FIFO for a zero-cone witness");
  loom::dse::HardwareImpactProjection unusedImpact{
      targetModules.front(), system, {}, {}, {}, {}};
  unusedImpact.family = loom::dse::HardwareMutationFamily::SpatialFifo;
  unusedImpact.locality = loom::dse::HardwareMutationLocality::LocalCone;
  unusedImpact.tech.kind = loom::dse::HardwareMappingImpactKind::Rebase;
  unusedImpact.spatial.kind = loom::dse::HardwareMappingImpactKind::Reopen;
  unusedImpact.spatial.placementRoots.push_back(
      take(loom::fabric::FabricModulePhysicalOwnerRef::create(*unusedFifo)));
  unusedImpact.moduleEntities =
      identityModuleEntityCorrespondence(targetModule.view());
  const auto unusedFrontier = take(loom::dse::rebaseJointMappingFrontier(
      plan, parentExecution, system, identityModuleCorrespondence, unusedImpact,
      store));
  if (unusedFrontier.disposition !=
          loom::dse::JointMappingReuseDisposition::Preserved ||
      unusedFrontier.accounting.invalidatedSpatialMappings != 0 ||
      unusedFrontier.accounting.repairedSpatialMappings != 0 ||
      unusedFrontier.seed.spatialMappings.empty())
    fail("unused FIFO impact did not produce a zero-cone Spatial preserve");

  auto globalImpact = localSpatialImpact;
  globalImpact.family = loom::dse::HardwareMutationFamily::FuCapability;
  globalImpact.locality = loom::dse::HardwareMutationLocality::GlobalReopen;
  globalImpact.tech.kind = loom::dse::HardwareMappingImpactKind::Reopen;
  globalImpact.tech.realizationRoots.push_back(moduleRoot);
  const auto coldFallbackFrontier = take(loom::dse::rebaseJointMappingFrontier(
      plan, parentExecution, system, identityModuleCorrespondence, globalImpact,
      store));
  if (coldFallbackFrontier.disposition !=
          loom::dse::JointMappingReuseDisposition::ColdFallback ||
      !coldFallbackFrontier.seed.techMappings.empty() ||
      !coldFallbackFrontier.seed.spatialMappings.empty() ||
      coldFallbackFrontier.accounting.parentTechMappings == 0 ||
      coldFallbackFrontier.accounting.parentSpatialMappings == 0 ||
      coldFallbackFrontier.accounting.invalidatedTechMappings !=
          coldFallbackFrontier.accounting.parentTechMappings ||
      coldFallbackFrontier.accounting.invalidatedSpatialMappings !=
          coldFallbackFrontier.accounting.parentSpatialMappings ||
      coldFallbackFrontier.accounting.invalidationRootCount == 0 ||
      coldFallbackFrontier.accounting.invalidationConeDecisionCount == 0)
    fail("typed global impact did not preserve a cold fallback");

  const auto requireLocalModuleRebase =
      [&](loom::dse::HardwareMutationFamily family,
          loom::fabric::FabricModulePhysicalOwnerRef owner) {
        auto impact = localSpatialImpact;
        impact.family = family;
        impact.tech.kind = loom::dse::HardwareMappingImpactKind::Rebase;
        impact.spatial.kind = loom::dse::HardwareMappingImpactKind::Rebase;
        impact.spatial.placementRoots = {owner};
        const auto result = take(loom::dse::rebaseJointMappingFrontier(
            plan, parentExecution, system, identityModuleCorrespondence, impact,
            store));
        if (result.disposition !=
                loom::dse::JointMappingReuseDisposition::Preserved ||
            result.seed.techMappings.empty() ||
            result.seed.spatialMappings.empty() ||
            result.accounting.invalidatedTechMappings != 0 ||
            result.accounting.invalidatedSpatialMappings != 0)
          fail("local Module mutation did not preserve its typed Mapping "
               "frontier");
      };
  if (!targetModule.view().memoryOccurrences().empty())
    requireLocalModuleRebase(
        loom::dse::HardwareMutationFamily::SpatialMemory,
        take(loom::fabric::FabricModulePhysicalOwnerRef::create(
            targetModule.view().memoryOccurrences().front())));
  if (!targetModule.view().peOccurrences().empty())
    requireLocalModuleRebase(
        loom::dse::HardwareMutationFamily::InstructionCapacity,
        take(loom::fabric::FabricModulePhysicalOwnerRef::create(
            targetModule.view().peOccurrences().front())));
  if (!targetModule.view().switchOccurrences().empty()) {
    auto switchImpact = localSpatialImpact;
    switchImpact.family = loom::dse::HardwareMutationFamily::SpatialSwitch;
    switchImpact.locality = loom::dse::HardwareMutationLocality::GlobalReopen;
    switchImpact.tech.kind = loom::dse::HardwareMappingImpactKind::Reopen;
    switchImpact.spatial.kind = loom::dse::HardwareMappingImpactKind::Reopen;
    switchImpact.tech.realizationRoots = {
        take(loom::fabric::FabricModulePhysicalOwnerRef::create(
            targetModule.view().switchOccurrences().front()))};
    switchImpact.spatial.placementRoots = switchImpact.tech.realizationRoots;
    const auto switchFallback = take(loom::dse::rebaseJointMappingFrontier(
        plan, parentExecution, system, identityModuleCorrespondence,
        switchImpact, store));
    if (switchFallback.disposition !=
            loom::dse::JointMappingReuseDisposition::ColdFallback ||
        !switchFallback.seed.techMappings.empty() ||
        !switchFallback.seed.spatialMappings.empty())
      fail("global switch mutation did not produce a typed cold fallback");
  }
  llvm::SmallString<128> adjacentJournal(temporary.path());
  llvm::sys::path::append(adjacentJournal, "adjacent-resource-time");
  const std::array adjacentPartitions = {
      loom::pnr::SystemBindingPartitionIntent{mappedRoots.front(), 2}};
  const std::array adjacentRoots = {mappedRoots.front()};
  loom::dse::JointHardwareReopenRequest adjacentRequest{
      take(loom::dse::DseProducerSemanticBuildIdentity::get(
          "loom.test.resource_time_adjacent.v1")),
      adjacentJournal.str().str(),
      {},
      loom::dse::JointDesignStoppingPolicy::FirstVerified,
      std::nullopt,
      std::nullopt,
      take(loom::dse::SiteCapacity::get(2, 0, 0)),
      take(loom::dse::PlanExecutionPolicy::get(
          2, take(loom::dse::SiteResourceClaim::get(1, 0, 0))))};
  adjacentRequest.invocationSemanticInputs = {alternateSystem};
  const auto adjacentRepair =
      take(loom::dse::executeResourceTimeAdjacentMappingRepair(
          plan, parentExecution, policy, adjacentPartitions, adjacentRoots,
          std::move(adjacentRequest), store, blobs));
  std::vector<loom::ArtifactRootReference> adjacentSemanticInputs =
      loom::dse::projectJointDesignSemanticInputs(adjacentRepair.plan);
  adjacentSemanticInputs.push_back(alternateSystem);
  const auto adjacentClosure = take(loom::dse::DseRunClosure::get(
      take(loom::dse::DseProducerSemanticBuildIdentity::get(
          "loom.test.resource_time_adjacent.v1")),
      adjacentSemanticInputs, adjacentRepair.plan.resolvedConfig, {}, store));
  if (!adjacentRepair.execution.invocationRunKey() ||
      *adjacentRepair.execution.invocationRunKey() !=
          adjacentClosure.runKey().bytes())
    fail("adjacent repair closure omitted its invocation semantic input");
  const auto adjacentSeed = take(loom::pnr::importSystemMappingMigrationSeed(
      adjacentRepair.migrationSeed, store));
  if (adjacentSeed.reopenedRoots() !=
          llvm::ArrayRef<dataflow::RootThreadLaunchRef>(adjacentRoots) ||
      adjacentRepair.coldExecution.summary.techMappingDispatchCount == 0 ||
      adjacentRepair.coldExecution.summary.spatialPnrDispatchCount == 0 ||
      adjacentRepair.coldExecution.summary.systemPnrDispatchCount == 0 ||
      adjacentRepair.coldExecution.summary.coldReopenWallTimeNanoseconds !=
          adjacentRepair.coldExecution.summary.executionWallTimeNanoseconds ||
      adjacentRepair.coldExecution.summary
              .incrementalReopenWallTimeNanoseconds != 0 ||
      adjacentRepair.execution.summary.techMappingDispatchCount != 0 ||
      adjacentRepair.execution.summary.spatialPnrDispatchCount != 0 ||
      adjacentRepair.execution.summary.systemPnrDispatchCount != 1 ||
      adjacentRepair.execution.summary.incrementalReopenWallTimeNanoseconds !=
          adjacentRepair.execution.summary.executionWallTimeNanoseconds ||
      adjacentRepair.execution.summary.coldReopenWallTimeNanoseconds != 0 ||
      adjacentRepair.execution.summary.preservedTechMappings == 0 ||
      adjacentRepair.execution.summary.preservedSpatialMappings == 0)
    fail("adjacent resource-time finalist did not use preserve-first repair");
  std::vector<loom::ArtifactRootReference> adjacentMappings;
  for (const auto &pair : adjacentRepair.execution.mappedPairs)
    adjacentMappings.insert(adjacentMappings.end(), pair.systemMappings.begin(),
                            pair.systemMappings.end());
  llvm::sort(adjacentMappings, loom::artifactRootReferenceLess);
  adjacentMappings.erase(
      std::unique(adjacentMappings.begin(), adjacentMappings.end()),
      adjacentMappings.end());
  std::vector<loom::ArtifactRootReference> coldAdjacentMappings;
  for (const auto &pair : adjacentRepair.coldExecution.mappedPairs)
    coldAdjacentMappings.insert(coldAdjacentMappings.end(),
                                pair.systemMappings.begin(),
                                pair.systemMappings.end());
  llvm::sort(coldAdjacentMappings, loom::artifactRootReferenceLess);
  coldAdjacentMappings.erase(
      std::unique(coldAdjacentMappings.begin(), coldAdjacentMappings.end()),
      coldAdjacentMappings.end());
  if (adjacentMappings.empty() ||
      llvm::is_contained(adjacentMappings, mappings.front()))
    fail("adjacent resource-time repair did not publish a distinct Mapping");
  if (coldAdjacentMappings.empty() || !adjacentRepair.coldMapping ||
      !adjacentRepair.incrementalMapping ||
      !llvm::is_contained(coldAdjacentMappings, *adjacentRepair.coldMapping) ||
      !llvm::is_contained(adjacentMappings, *adjacentRepair.incrementalMapping))
    fail("adjacent resource-time repair did not publish a paired cold and "
         "incremental Mapping");
  auto adjacentMapping =
      take(loom::mapping::importSystemMapping(adjacentMappings.front(), store));
  if (adjacentMapping.view().dataflowIdentity() !=
          plan.frontier.pairs.front().software.dataflow.artifact ||
      adjacentMapping.view().fabricIdentity() != system.artifact)
    fail("adjacent resource-time repair changed immutable owners");
  if (qualityRuns("adjacent")) {
    if (!incompleteRepairQuality)
      fail("quality-promotion fixture lost its incomplete repair policy");
    llvm::SmallString<128> incompleteAdjacentJournal(temporary.path());
    llvm::sys::path::append(incompleteAdjacentJournal,
                            "adjacent-resource-time-quality-incomplete");
    loom::dse::JointHardwareReopenRequest incompleteAdjacentRequest{
        take(loom::dse::DseProducerSemanticBuildIdentity::get(
            "loom.test.resource_time_adjacent.quality_incomplete.v1")),
        incompleteAdjacentJournal.str().str(),
        {},
        loom::dse::JointDesignStoppingPolicy::BoundedQuality,
        *incompleteRepairQuality,
        std::nullopt,
        take(loom::dse::SiteCapacity::get(2, 0, 0)),
        take(loom::dse::PlanExecutionPolicy::get(
            2, take(loom::dse::SiteResourceClaim::get(1, 0, 0))))};
    const auto incompleteAdjacent =
        take(loom::dse::executeResourceTimeAdjacentMappingRepair(
            plan, parentExecution, policy, adjacentPartitions, adjacentRoots,
            std::move(incompleteAdjacentRequest), store, blobs));
    const auto retainsMappedPair = [](const auto &execution) {
      return llvm::any_of(execution.mappedPairs, [](const auto &pair) {
        return !pair.systemMappings.empty();
      });
    };
    for (const auto *execution :
         {&incompleteAdjacent.coldExecution, &incompleteAdjacent.execution})
      if (!retainsMappedPair(*execution) ||
          execution->summary.qualityDisposition !=
              loom::dse::JointDesignQualityDisposition::Unsupported ||
          execution->summary.selectedMapping ||
          execution->summary.selectedPlanOrdinal)
        fail("bounded incomplete adjacent repair retained a selected "
             "Mapping");
    if (incompleteAdjacent.coldMapping || incompleteAdjacent.incrementalMapping)
      fail("bounded incomplete adjacent repair published a Mapping join");
  }

  const std::vector<loom::ArtifactRootReference> systems = {system,
                                                            alternateSystem};
  const std::vector<loom::dse::JointMemberPromotion> memberPromotions = {
      {plan.frontier.pairs.front().software.dataflow,
       loom::dse::CompletedSelection{mappings, {}}}};
  auto selected = take(loom::dse::selectJointDesignSystems(
      systems, memberPromotions, {}, loom::dse::AllPassingSelection{}, nullptr,
      store));
  const bool covered = everyCoreIsUsed(system, mappings, store);
  bool sawMissingAlternate = false;
  bool sawUnusedPrimary = false;
  std::vector<loom::dse::JointSystemGateOutcome> *outcomes = nullptr;
  if (auto *completedSelection =
          std::get_if<loom::dse::JointDesignSelection>(&selected)) {
    outcomes = &completedSelection->systemOutcomes;
    if (!covered ||
        completedSelection->selectedSystems !=
            std::vector<loom::ArtifactRootReference>{system} ||
        completedSelection->acceptedMappings != mappings)
      fail("aggregate selection bypassed member-local System gates");
  } else {
    auto &noFeasible =
        std::get<loom::dse::JointDesignNoFeasibleSystem>(selected);
    outcomes = &noFeasible.systemOutcomes;
    if (covered)
      fail("fully covered System was rejected before aggregate selection");
  }
  for (const loom::dse::JointSystemGateOutcome &outcome : *outcomes) {
    if (const auto *missing =
            std::get_if<loom::dse::JointSystemMissingMember>(&outcome))
      sawMissingAlternate |= missing->system == alternateSystem;
    if (const auto *unused =
            std::get_if<loom::dse::JointSystemUnusedAccCore>(&outcome))
      sawUnusedPrimary |= unused->system == system;
  }
  if (!sawMissingAlternate || sawUnusedPrimary == covered)
    fail("typed System dispositions lost missing-member or AccCore coverage");

  auto oversized = loom::dse::buildBoundedJointFrontier(
      {{{firstWorkload}, {secondWorkload}}, {system}},
      take(loom::dse::JointDesignPolicy::get(1, 1, 1, 1, 1)), store);
  if (oversized)
    fail("joint frontier accepted a software set beyond its resolved bound");
  const std::string oversizedMessage = llvm::toString(oversized.takeError());
  if (!llvm::StringRef(oversizedMessage).contains("exceeds"))
    fail("frontier-bound rejection lost its diagnostic");
}

} // namespace

int main(int argc, char **argv) {
  const llvm::StringRef mode = argc == 2 ? argv[1] : "";
  // `mutation-matrix` and `quality-promotion` run every section in one
  // process; the `=NAME` forms run one independent section so the sections can
  // be sharded across processes.
  llvm::StringRef mutationFamily;
  if (mode == "mutation-matrix")
    mutationFamily = allMutationFamilies;
  else if (mode.starts_with("mutation-matrix="))
    mutationFamily = mode.drop_front(llvm::StringRef("mutation-matrix=").size());
  llvm::StringRef qualitySection;
  if (mode == "quality-promotion")
    qualitySection = allMutationFamilies;
  else if (mode.starts_with("quality-promotion="))
    qualitySection =
        mode.drop_front(llvm::StringRef("quality-promotion=").size());
  if (!qualitySection.empty() && qualitySection != allMutationFamilies &&
      qualitySection != "promotion" && qualitySection != "operand" &&
      qualitySection != "fifo" && qualitySection != "adjacent")
    fail("unknown quality-promotion section: " + qualitySection);
  if (argc > 2 || (argc == 2 && mutationFamily.empty() &&
                   qualitySection.empty() && mode != "fifo-feedback" &&
                   mode != "operand-feedback" && mode != "transport-feedback"))
    fail("expected no workflow, fifo-feedback, operand-feedback, "
         "transport-feedback, quality-promotion, quality-promotion=SECTION, "
         "mutation-matrix, or mutation-matrix=FAMILY");
  exerciseJointExploration(mode == "fifo-feedback", mode == "operand-feedback",
                           mode == "transport-feedback", qualitySection,
                           mutationFamily);
  return 0;
}
