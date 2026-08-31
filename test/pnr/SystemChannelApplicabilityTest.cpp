#include "ADG/Builtin.h"
#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "Config/ResolvedConfig.h"
#include "DSE/CandidateGenerator.h"
#include "DSE/Plan.h"
#include "DSE/ResolvedConfigView.h"
#include "DSE/RootCompleteSystemPnrCandidateGenerator.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Artifact/FabricSystemRootView.h"
#include "Fabric/Identity/FabricPhysicalTiming.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Mapping/Artifact/MappingArtifact.h"
#include "Mapping/Artifact/MappingConstraintSet.h"
#include "Mapping/Artifact/SystemMappingArtifact.h"
#include "Mapping/Artifact/SystemMappingConstraintSet.h"
#include "Mapping/IR/MappingDialect.h"
#include "Mapping/Tech/TechMappingConfig.h"
#include "Mapping/Tech/TechMappingGenerator.h"
#include "PnR/MappingObjective.h"
#include "PnR/PnrConfig.h"
#include "PnR/SpatialPnrGenerator.h"
#include "PnR/System/SystemActionDomain.h"
#include "PnR/System/SystemAnnealingSearch.h"
#include "PnR/System/SystemCandidateState.h"
#include "PnR/System/SystemMappingMaterializer.h"
#include "PnR/System/SystemPnrProblem.h"
#include "PnR/System/SystemPnrSearchDomain.h"

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

#include <array>
#include <cstdlib>
#include <functional>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <system_error>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace {

[[noreturn]] void fail(const llvm::Twine &message) {
  llvm::errs() << "System channel applicability anchor failed: " << message
               << '\n';
  std::exit(EXIT_FAILURE);
}

void require(bool condition, const llvm::Twine &message) {
  if (!condition)
    fail(message);
}

struct ImmediateStopSource final {
  static bool query(const void *) { return true; }

  loom::ExecutionControlView control() const { return {this, query}; }
};

template <typename T>
void requireFailureContains(llvm::Expected<T> value, llvm::StringRef expected) {
  if (value)
    fail("adverse channel applicability input unexpectedly succeeded");
  const std::string diagnostic = llvm::toString(value.takeError());
  require(llvm::StringRef(diagnostic).contains(expected),
          "adverse diagnostic changed: " + diagnostic);
}

void requireFailureContains(
    const loom::mapping::SystemMappingBaseVerification &verification,
    llvm::StringRef expected) {
  if (std::holds_alternative<loom::mapping::VerifiedSystemMappingBase>(
          verification))
    fail("adverse channel applicability input unexpectedly succeeded; "
         "expected: " +
         expected);
  const std::string &diagnostic = std::visit(
      [](const auto &result) -> const std::string & {
        using Result = std::decay_t<decltype(result)>;
        if constexpr (std::is_same_v<Result,
                                     loom::mapping::VerifiedSystemMappingBase>)
          llvm_unreachable("verified result has no diagnostic");
        else
          return result.diagnostic;
      },
      verification);
  require(llvm::StringRef(diagnostic).contains(expected),
          "adverse diagnostic changed: " + diagnostic);
}

template <typename T> T take(llvm::Expected<T> value) {
  if (!value)
    fail(llvm::toString(value.takeError()));
  return std::move(*value);
}

std::vector<std::uint8_t> unsignedBytes(mlir::DenseI8ArrayAttr attribute) {
  std::vector<std::uint8_t> result;
  result.reserve(attribute.size());
  for (std::int8_t byte : attribute.asArrayRef())
    result.push_back(static_cast<std::uint8_t>(byte));
  return result;
}

mlir::DenseI8ArrayAttr bytesAttr(mlir::MLIRContext *context,
                                 llvm::ArrayRef<std::uint8_t> bytes) {
  std::vector<std::int8_t> signedBytes;
  signedBytes.reserve(bytes.size());
  for (std::uint8_t byte : bytes)
    signedBytes.push_back(static_cast<std::int8_t>(byte));
  return mlir::DenseI8ArrayAttr::get(context, signedBytes);
}

std::string
endpointKey(const loom::fabric::FabricTransportEndpointRef &endpoint) {
  const auto bytes = loom::fabric::canonicalFabricBytes(endpoint);
  return std::string(reinterpret_cast<const char *>(bytes.data()),
                     bytes.size());
}

std::optional<std::vector<loom::fabric::FabricPhysicalTraversalRef>>
findPhysicalTraversalCycle(const loom::fabric::FabricSystemRootView &system,
                           loom::fabric::FabricTransportEndpointRef start) {
  using Traversal = loom::fabric::FabricPhysicalTraversalView;
  std::map<std::string, std::vector<const Traversal *>> outgoing;
  for (const Traversal &traversal : system.artifact().physicalTraversals())
    for (const auto source : traversal.sources)
      outgoing[endpointKey(source)].push_back(&traversal);

  std::vector<loom::fabric::FabricPhysicalTraversalRef> path;
  std::set<std::string> visiting{endpointKey(start)};
  std::function<bool(loom::fabric::FabricTransportEndpointRef)> search =
      [&](loom::fabric::FabricTransportEndpointRef current) {
        const auto found = outgoing.find(endpointKey(current));
        if (found == outgoing.end())
          return false;
        for (const Traversal *traversal : found->second) {
          path.push_back(traversal->reference);
          for (const auto destination : traversal->destinations) {
            if (destination == start)
              return true;
            const std::string key = endpointKey(destination);
            if (visiting.insert(key).second) {
              if (search(destination))
                return true;
              visiting.erase(key);
            }
          }
          path.pop_back();
        }
        return false;
      };
  if (!search(start))
    return std::nullopt;
  return path;
}

class TemporaryDirectory final {
public:
  TemporaryDirectory() {
    std::error_code error = llvm::sys::fs::createUniqueDirectory(
        "loom-system-channel-applicability", path_);
    if (error)
      fail("cannot create ArtifactStore directory: " + error.message());
  }

  ~TemporaryDirectory() { llvm::sys::fs::remove_directories(path_); }

  llvm::StringRef path() const { return path_; }

private:
  llvm::SmallString<128> path_;
};

mlir::MLIRContext makeContext() {
  mlir::DialectRegistry registry;
  registry.insert<dataflow::DataflowDialect, mlir::arith::ArithDialect,
                  mapping::MappingDialect, mlir::DLTIDialect,
                  mlir::func::FuncDialect>();
  return mlir::MLIRContext(registry, mlir::MLIRContext::Threading::DISABLED);
}

dataflow::CanonicalDataflowArtifact buildDataflow(mlir::MLIRContext &context) {
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 64>>} {
  dataflow.graph private @consume(%start: none, %input: i32) -> ()
      attributes {input_segments = array<i32: 0, 1, 0>,
                  result_segments = array<i32: 0, 0, 0>} {
    %done = dataflow.sync %start : (none) -> (none)
    dataflow.graph.return %done : none
  }
  dataflow.thread private @producer domain(#dataflow.thread_domain<dense>)(
      %channel: !dataflow.channel<i32>) ctrl (%ctrl: none) iv (%iv: index) {
    %value = arith.constant 7 : i32
    dataflow.channel.send %channel, %value : !dataflow.channel<i32>
    dataflow.thread.yield
  }
  dataflow.thread private @consumer domain(#dataflow.thread_domain<dense>)(
      %channel: !dataflow.channel<i32>) ctrl (%ctrl: none) iv (%iv: index) {
    %done = dataflow.graph.launch @consume deps(%ctrl) values()
        stream_inputs(%channel source_map affine_map<(d0) -> (0)>) memories()
        stream_outputs() : (none, !dataflow.channel<i32>) -> none
    dataflow.thread.yield %done : none
  }
  func.func private @host(%channel: !dataflow.channel<i32>) {
    %producer_extent = arith.constant 2 : index
    %consumer_extent = arith.constant 3 : index
    %produced = dataflow.thread.launch @producer(%channel)
        grid(%producer_extent)
        : (!dataflow.channel<i32>) -> !dataflow.thread_token
    %consumed = dataflow.thread.launch @consumer(%channel)
        grid(%consumer_extent)
        : (!dataflow.channel<i32>) -> !dataflow.thread_token
    return
  }
}
)mlir",
                                                        &context);
  if (!module)
    fail("cannot parse Dataflow fixture");
  return take(dataflow::finalizeCanonicalDataflow(*module));
}

dataflow::CanonicalDataflowArtifact
buildInstructionOnlyDataflow(mlir::MLIRContext &context) {
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 64>>} {
  dataflow.thread private @worker domain(#dataflow.thread_domain<dense>)()
      ctrl (%ctrl: none) {
    dataflow.thread.yield
  }
  func.func private @host() {
    %token = dataflow.thread.launch @worker()
        : () -> !dataflow.thread_token
    return
  }
}
)mlir",
                                                        &context);
  if (!module)
    fail("cannot parse instruction-only Dataflow fixture");
  return take(dataflow::finalizeCanonicalDataflow(*module));
}

dataflow::CanonicalDataflowArtifact
buildRootFreeDataflow(mlir::MLIRContext &context) {
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 64>>} {
  func.func private @helper() {
    return
  }
}
)mlir",
                                                        &context);
  if (!module)
    fail("cannot parse root-free Dataflow fixture");
  return take(dataflow::finalizeCanonicalDataflow(*module));
}

enum class SystemSearchProbeDomain : std::uint8_t {
  Assignment,
  Routing,
};

loom::ResolvedConfig
buildResolvedConfig(SystemSearchProbeDomain probeDomain =
                        SystemSearchProbeDomain::Routing) {
  loom::ResolvedConfig resolved = loom::defaultResolvedConfig();
  constexpr std::uint64_t maximum = std::numeric_limits<std::uint64_t>::max();
  resolved.dse.objectiveCatalogs.dimensions = {
      {loom::ResolvedMappingViolationObjectiveSource{
           loom::ResolvedPnrViolationKind::UnroutedObligation},
       loom::ResolvedObjectiveDirection::Minimize,
       loom::resolvedObjectiveInteger(0), loom::resolvedObjectiveInteger(1), 0,
       maximum},
      {loom::ResolvedMappingViolationObjectiveSource{
           loom::ResolvedPnrViolationKind::CapacityOveruse},
       loom::ResolvedObjectiveDirection::Minimize,
       loom::resolvedObjectiveInteger(0), loom::resolvedObjectiveInteger(1), 0,
       maximum},
      {loom::ResolvedMappingMeasureObjectiveSource{static_cast<std::uint32_t>(
           loom::pnr::MappingMeasureKind::TotalSelectedTraversalClaim)},
       loom::ResolvedObjectiveDirection::Minimize,
       loom::resolvedObjectiveInteger(0), loom::resolvedObjectiveInteger(1), 0,
       maximum}};
  resolved.dse.objectiveCatalogs.weightedLevels = {{{{0, 1}, {1, 1}, {2, 1}}}};
  resolved.dse.objectiveCatalogs.totalOrderings = {{{0}}};
  resolved.dse.techMapping.candidatePublicationLimit = 1;
  resolved.dse.spatialPnr.temporaryViolations.admitted = {
      loom::ResolvedPnrViolationKind::UnroutedObligation,
      loom::ResolvedPnrViolationKind::CapacityOveruse};
  resolved.dse.spatialPnr.objectiveSelection = {0, 0};
  auto &search = resolved.dse.spatialPnr.search;
  search.initializer.seedAttemptCount = 1;
  search.actionProposal = {0, 1, 0};
  search.annealing.calibrationProposalCount = 1;
  search.annealing.fallbackTemperature = 1;
  search.annealing.minimumTemperature = 1;
  search.annealing.coolingRatio = {1, 2};
  search.annealing.proposalsPerLevelBase = 1;
  search.annealing.proposalsPerMovableDecision = 0;
  search.exactRepair = {loom::ResolvedPnrExactRepairKind::Disabled, 0, 0};
  resolved.dse.systemPnr.temporaryViolations.admitted = {
      loom::ResolvedPnrViolationKind::UnroutedObligation,
      loom::ResolvedPnrViolationKind::CapacityOveruse};
  resolved.dse.systemPnr.objectiveSelection = {0, 0};
  auto &systemSearch = resolved.dse.systemPnr.search;
  systemSearch.initializer.seedAttemptCount = 1;
  systemSearch.actionProposal =
      probeDomain == SystemSearchProbeDomain::Assignment
          ? loom::ResolvedPnrActionProposalPolicy{1, 0, 0}
          : loom::ResolvedPnrActionProposalPolicy{0, 1, 0};
  systemSearch.annealing.calibrationProposalCount = 1;
  systemSearch.annealing.fallbackTemperature = 1;
  systemSearch.annealing.minimumTemperature = 1;
  systemSearch.annealing.coolingRatio = {1, 2};
  systemSearch.annealing.proposalsPerLevelBase = 1;
  systemSearch.annealing.proposalsPerMovableDecision = 0;
  systemSearch.exactRepair = {loom::ResolvedPnrExactRepairKind::Disabled, 0, 0};
  return resolved;
}

loom::ArtifactRootReference
generateSpatialMapping(const dataflow::CanonicalDataflowProgramView &dataflow,
                       const loom::fabric::FinalizedFabricRoot &module,
                       const loom::ResolvedConfig &resolved,
                       loom::ArtifactStore &store) {
  const auto techConfig =
      take(loom::mapping::projectResolvedTechMappingConfigView(resolved));
  const std::array<dataflow::GraphRef, 1> covers = {
      dataflow.graphs().front().ref};
  auto techOutcome = loom::mapping::generateTechMappings(
      {dataflow, covers, module.view(), techConfig, store});
  const auto *tech =
      std::get_if<loom::mapping::GeneratedTechMappings>(&techOutcome);
  require(tech && tech->candidates.size() == 1,
          "TechMapping fixture did not produce one candidate");
  auto imported =
      take(loom::mapping::importTechMapping(tech->candidates.front(), store));
  auto constraints =
      take(loom::mapping::finalizeEmptySpatialMappingConstraintSet(
          dataflow, imported.view(), module.view(), store));
  const auto spatialConfig =
      take(loom::pnr::projectResolvedSpatialPnrConfigView(resolved));
  const auto physicalTiming =
      take(loom::fabric::projectNormalizedFabricPhysicalTimingProfile(
          module.view()));
  auto spatialOutcome = loom::pnr::generateSpatialMappings(
      {dataflow, imported.view(), module.view(), physicalTiming, spatialConfig,
       constraints.view(), store});
  const auto *spatial =
      std::get_if<loom::pnr::GeneratedSpatialMappings>(&spatialOutcome);
  if (!spatial)
    std::visit(
        [&](const auto &outcome) {
          using Outcome = std::decay_t<decltype(outcome)>;
          if constexpr (std::is_same_v<
                            Outcome,
                            loom::pnr::InterruptedSpatialPnrGeneration>)
            fail("SpatialMapping fixture was interrupted at " +
                 loom::pnr::spatialPnrInterruptionStageSpelling(
                     outcome.snapshot.stage));
          else if constexpr (!std::is_same_v<
                                 Outcome, loom::pnr::GeneratedSpatialMappings>)
            fail("SpatialMapping fixture did not produce one candidate: " +
                 outcome.diagnostic);
        },
        spatialOutcome);
  require(spatial->candidates.size() == 1,
          "SpatialMapping fixture produced the wrong candidate count");
  return spatial->candidates.front();
}

loom::mapping::SystemPresburgerCell
bounded(loom::mapping::SystemPresburgerCell cell,
        std::optional<std::int64_t> lower, std::optional<std::int64_t> upper) {
  const std::size_t width = static_cast<std::size_t>(cell.dimensionCount) +
                            cell.symbolCount + cell.localCount + 1;
  require(cell.dimensionCount == 1, "fixture binding must be rank one");
  if (lower) {
    std::vector<std::int64_t> lowerRow(width, 0);
    lowerRow.front() = 1;
    lowerRow.back() = -*lower;
    cell.inequalities.push_back(std::move(lowerRow));
  }
  if (upper) {
    std::vector<std::int64_t> upperRow(width, 0);
    upperRow.front() = -1;
    upperRow.back() = *upper;
    cell.inequalities.push_back(std::move(upperRow));
  }
  return take(loom::mapping::canonicalizeSystemPresburgerCell(cell));
}

std::vector<loom::ArtifactRootReference>
normalizedTimingProfileRoots(const loom::ArtifactRootReference &systemReference,
                             loom::ArtifactStore &store) {
  auto artifact =
      take(loom::fabric::importEntireFabricRoot(systemReference, store));
  auto system = take(loom::fabric::requireSystemRoot(artifact.view()));
  auto profiles =
      take(loom::fabric::projectNormalizedSystemPhysicalTimingProfiles(system));
  std::vector<loom::ArtifactRootReference> roots;
  roots.reserve(profiles.size());
  for (const auto &profile : profiles)
    roots.push_back(
        take(loom::fabric::publishFabricPhysicalTimingProfile(profile, store)));
  llvm::sort(roots, loom::artifactRootReferenceLess);
  return roots;
}

void verifyRootCompleteSystemAdapter(
    const loom::ArtifactRootReference &dataflowReference,
    llvm::ArrayRef<loom::ArtifactRootReference> spatialMappings,
    const loom::ArtifactRootReference &systemReference,
    const loom::ResolvedConfig &resolved, loom::ArtifactStore &store,
    llvm::StringRef directory) {
  llvm::SmallString<128> blobPath(directory);
  llvm::sys::path::append(blobPath, "blobs");
  if (std::error_code error = llvm::sys::fs::create_directories(blobPath))
    fail("cannot create BlobStore directory: " + error.message());
  const loom::BlobStore blobs(blobPath);
  if (llvm::Error error =
          loom::dse::registerRootCompleteSystemPnrCandidateGenerator())
    fail(llvm::toString(std::move(error)));
  const auto config =
      take(loom::pnr::projectResolvedSystemPnrConfigView(resolved));
  const auto &descriptor =
      loom::dse::rootCompleteSystemPnrCandidateGeneratorDescriptor();
  const auto physicalTimingProfiles =
      normalizedTimingProfileRoots(systemReference, store);
  require(
      descriptor.kind ==
              loom::dse::rootCompleteSystemPnrCandidateGeneratorKind &&
          descriptor.inputSlots.size() == 6 &&
          descriptor.outputSlots.size() == 1 &&
          descriptor.implementationSemanticIdentity ==
              "loom.mapping.root_complete_system_pnr.generator.v13" &&
          descriptor.workUnits.size() ==
              loom::dse::pnrCandidateGeneratorWorkUnits.size() &&
          descriptor.inputSlots[0].semanticRole == "dataflow" &&
          descriptor.inputSlots[1].semanticRole == "spatial_mapping" &&
          descriptor.inputSlots[2].semanticRole == "fabric" &&
          descriptor.inputSlots[3].semanticRole == "physical_timing_profile" &&
          descriptor.inputSlots[4].semanticRole == "migration_seed" &&
          descriptor.inputSlots[4].cardinality ==
              loom::dse::PlanValueCardinality::ZeroOrOne &&
          descriptor.inputSlots[5].semanticRole == "finalized_migration_seed" &&
          descriptor.inputSlots[5].cardinality ==
              loom::dse::PlanValueCardinality::ZeroOrOne,
      "root-complete System descriptor lost its exact timing-bound input "
      "shape");
  auto inputs =
      take(loom::dse::bindRootCompleteSystemPnrCandidateGeneratorInputs(
          dataflowReference, spatialMappings, systemReference,
          physicalTimingProfiles));
  auto binding = take(
      loom::dse::resolveRootCompleteSystemPnrCandidateGeneratorBinding(config));
  auto first =
      take(loom::dse::invokeCandidateGenerator(inputs, binding, store, blobs));
  const auto *completed =
      std::get_if<loom::dse::CompletedCandidateGeneratorResult>(&first.outcome);
  require(completed && completed->outputBindings.size() == 1 &&
              completed->outputBindings.front().artifacts.size() == 1 &&
              completed->lineageEdges.size() == 1,
          "root-complete System adapter did not publish one SystemMapping");
  require(first.workSummary.size() ==
                  loom::dse::pnrCandidateGeneratorWorkUnits.size() &&
              first.workSummary[0].consumed == 1 &&
              first.workSummary[3].consumed != 0 &&
              first.workSummary[4].consumed == 1 &&
              first.workSummary[5].consumed != 0,
          "root-complete System adapter lost real bounded search work");
  for (const auto &unit : first.workSummary)
    require(unit.planned == unit.consumed,
            "completed System provider left planned work unconsumed");
  const auto output = completed->outputBindings.front().artifacts.front();
  auto imported = take(loom::mapping::importSystemMapping(output, store));
  require(imported.view().dataflowIdentity() == dataflowReference.artifact &&
              imported.view().fabricIdentity() == systemReference.artifact,
          "root-complete System adapter published a foreign Mapping");

  auto replay =
      take(loom::dse::invokeCandidateGenerator(inputs, binding, store, blobs));
  const auto *replayed =
      std::get_if<loom::dse::CompletedCandidateGeneratorResult>(
          &replay.outcome);
  require(replayed &&
              replayed->outputBindings.front().artifacts ==
                  completed->outputBindings.front().artifacts &&
              replay.workSummary == first.workSummary,
          "root-complete System adapter changed output or work on replay");

  const ImmediateStopSource stop;
  auto cancelled = take(loom::dse::invokeCandidateGenerator(
      inputs, binding, store, blobs, stop.control()));
  const auto *interrupted =
      std::get_if<loom::dse::IncompleteCandidateGeneratorResult>(
          &cancelled.outcome);
  require(interrupted &&
              interrupted->reason ==
                  loom::dse::CandidateGeneratorIncompleteReason::
                      CancelledOrTimeout &&
              llvm::all_of(cancelled.workSummary,
                           [](const auto &work) { return work.consumed == 0; }),
          "System interruption did not map to a zero-frontier cancellation");

  loom::ResolvedConfig planned = resolved;
  planned.dse.planNodes = {loom::dse::GeneratePlanNodeDefinition{
      descriptor.reference(),
      {loom::dse::ExactPlanArtifacts{{dataflowReference}},
       loom::dse::ExactPlanArtifacts{spatialMappings.vec()},
       loom::dse::ExactPlanArtifacts{{systemReference}},
       loom::dse::ExactPlanArtifacts{physicalTimingProfiles},
       loom::dse::ExactPlanArtifacts{}, loom::dse::ExactPlanArtifacts{}},
      config.canonicalViewBytes().vec(),
      config.digest()}};
  auto planView = take(loom::dse::projectResolvedDseConfigView(planned));
  auto planOutcome = take(loom::dse::executeDsePlan(planView, store, blobs));
  const auto *planCompleted =
      std::get_if<loom::dse::CompletedDsePlanExecution>(&planOutcome);
  require(planCompleted &&
              planCompleted->resolve(loom::dse::PlanOutputRef{0, 0}) ==
                  llvm::ArrayRef(completed->outputBindings.front().artifacts) &&
              planCompleted->generateWorkSummaries().size() == 1 &&
              planCompleted->generateWorkSummaries().front().units ==
                  first.workSummary,
          "central Generate plan changed System output or work");

  if (!spatialMappings.empty()) {
    loom::ResolvedConfig limited = resolved;
    limited.dse.systemPnr.search.initializer.seedAttemptCount = 2;
    limited.dse.systemPnr.search.initializer.assignmentAttemptLimitPerSeed = 1;
    limited.dse.systemPnr.search.routing.endpointExpansionLimit = 1;
    const auto limitedConfig =
        take(loom::pnr::projectResolvedSystemPnrConfigView(limited));
    auto limitedBinding =
        take(loom::dse::resolveRootCompleteSystemPnrCandidateGeneratorBinding(
            limitedConfig));
    auto limitedResult = take(loom::dse::invokeCandidateGenerator(
        inputs, limitedBinding, store, blobs));
    const auto *incomplete =
        std::get_if<loom::dse::IncompleteCandidateGeneratorResult>(
            &limitedResult.outcome);
    require(incomplete &&
                incomplete->reason ==
                    loom::dse::CandidateGeneratorIncompleteReason::
                        SemanticLimitReached &&
                limitedResult.workSummary.size() ==
                    loom::dse::pnrCandidateGeneratorWorkUnits.size() &&
                limitedResult.workSummary.front().consumed == 2,
            "bounded System search stopped before its configured seeds");
  }
}

void verifyRootFreeSystemAdapter(
    const loom::ArtifactRootReference &dataflowReference,
    const loom::ArtifactRootReference &foreignSpatialMapping,
    const loom::ArtifactRootReference &systemReference,
    const loom::ResolvedConfig &resolved, loom::ArtifactStore &store,
    llvm::StringRef directory) {
  llvm::SmallString<128> blobPath(directory);
  llvm::sys::path::append(blobPath, "root-free-blobs");
  if (std::error_code error = llvm::sys::fs::create_directories(blobPath))
    fail("cannot create root-free BlobStore directory: " + error.message());
  const loom::BlobStore blobs(blobPath);
  const auto config =
      take(loom::pnr::projectResolvedSystemPnrConfigView(resolved));
  auto binding = take(
      loom::dse::resolveRootCompleteSystemPnrCandidateGeneratorBinding(config));
  const auto physicalTimingProfiles =
      normalizedTimingProfileRoots(systemReference, store);
  auto emptyInputs =
      take(loom::dse::bindRootCompleteSystemPnrCandidateGeneratorInputs(
          dataflowReference, {}, systemReference, physicalTimingProfiles));
  auto result = take(
      loom::dse::invokeCandidateGenerator(emptyInputs, binding, store, blobs));
  const auto *completed =
      std::get_if<loom::dse::CompletedCandidateGeneratorResult>(
          &result.outcome);
  require(completed && completed->outputBindings.front().artifacts.empty() &&
              completed->lineageEdges.empty() &&
              result.workSummary.size() ==
                  loom::dse::pnrCandidateGeneratorWorkUnits.size() &&
              llvm::all_of(result.workSummary,
                           [](const auto &unit) {
                             return unit.planned == 0 && unit.consumed == 0;
                           }),
          "root-free System adapter did not complete with exact zero work");

  auto foreignInputs =
      take(loom::dse::bindRootCompleteSystemPnrCandidateGeneratorInputs(
          dataflowReference, {foreignSpatialMapping}, systemReference,
          physicalTimingProfiles));
  auto foreign =
      loom::dse::invokeCandidateGenerator(foreignInputs, binding, store, blobs);
  requireFailureContains(std::move(foreign), "foreign Dataflow owner");
}

} // namespace

int main() {
  TemporaryDirectory directory;
  loom::ArtifactStore store(directory.path());
  mlir::MLIRContext context = makeContext();

  auto dataflowArtifact = buildDataflow(context);
  const auto dataflowReference =
      take(dataflow::publishCanonicalDataflow(dataflowArtifact, store));
  auto dataflow = take(dataflowArtifact.view());
  auto design = take(loom::adg::buildBuiltinTarget(
      store, loom::adg::BuiltinTargetPreset::Small));
  auto system =
      take(loom::fabric::requireSystemRoot(design.roots().front().view()));
  require(system.artifact().accCoreOccurrences().size() >= 2,
          "built-in fixture must expose at least two AccCore occurrences");
  std::size_t messageRouterCount = 0;
  for (loom::fabric::SystemTransportResourceRef resource :
       system.transportResources()) {
    const auto owner =
        loom::fabric::FabricTransportEndpointOwnerRef::of(resource);
    if (system.artifact().transportEndpointCount(owner) != 4)
      continue;
    const auto patterns = system.transferPatterns(resource);
    std::size_t unicastPatterns = 0;
    std::size_t multicastPatterns = 0;
    for (loom::fabric::FabricTransferPatternRef pattern : patterns) {
      const auto *record = system.transferPattern(pattern);
      require(record, "built-in message router has no transfer pattern");
      if (record->egresses().size() == 1)
        ++unicastPatterns;
      else if (record->egresses().size() == 2)
        ++multicastPatterns;
    }
    if (patterns.size() == 6 && unicastPatterns == 4 && multicastPatterns == 2)
      ++messageRouterCount;
  }
  require(messageRouterCount ==
              system.artifact().accCoreOccurrences().size() + 1,
          "built-in message plane is not a finite-degree execution ring");
  auto module = take(loom::fabric::importEntireFabricRoot(
      design.roots().front().directDependencies().front().root, store));
  auto resolved = buildResolvedConfig();
  const auto spatialMapping =
      generateSpatialMapping(dataflow, module, resolved, store);
  verifyRootCompleteSystemAdapter(dataflowReference, {spatialMapping},
                                  design.roots().front().reference(), resolved,
                                  store, directory.path());
  auto instructionOnlyArtifact = buildInstructionOnlyDataflow(context);
  const auto instructionOnlyReference =
      take(dataflow::publishCanonicalDataflow(instructionOnlyArtifact, store));
  verifyRootCompleteSystemAdapter(instructionOnlyReference, {},
                                  design.roots().front().reference(), resolved,
                                  store, directory.path());
  auto rootFreeArtifact = buildRootFreeDataflow(context);
  const auto rootFreeReference =
      take(dataflow::publishCanonicalDataflow(rootFreeArtifact, store));
  verifyRootFreeSystemAdapter(rootFreeReference, spatialMapping,
                              design.roots().front().reference(), resolved,
                              store, directory.path());

  std::vector<dataflow::RootThreadLaunchRef> roots;
  for (const auto &root : dataflow.rootThreadLaunches())
    roots.push_back(root.ref);
  require(roots.size() == 2, "fixture must contain two root launches");
  auto constraints =
      take(loom::mapping::finalizeEmptySystemMappingConstraintSet(
          dataflow, system, roots, store));
  auto partition = take(loom::pnr::projectWholeDomainPresburgerPartitionPlan(
      dataflow, constraints.view().rootThreadLaunches()));

  std::optional<dataflow::RootThreadLaunchRef> consumerRoot;
  std::optional<dataflow::CanonicalProducerTerminalRef> channelProducer;
  for (const auto &root : dataflow.rootThreadLaunches()) {
    if (llvm::Error error = dataflow.forEachProducerTerminal(
            root.ref,
            [&](const dataflow::CanonicalProducerTerminalView &producer)
                -> llvm::Error {
              const auto *channel =
                  std::get_if<dataflow::ChannelProducerTerminalRef>(
                      &producer.terminal);
              if (!channel)
                return llvm::Error::success();
              channelProducer = producer.terminal;
              auto consumers = dataflow.channelConsumers(channel->producer);
              if (!consumers)
                return consumers.takeError();
              require(consumers->size() == 1,
                      "fixture channel must have one static consumer");
              const auto *stream =
                  std::get_if<dataflow::GraphStreamInputConsumerRef>(
                      &consumers->front().consumer);
              require(stream,
                      "fixture channel consumer must be a graph stream");
              consumerRoot = stream->launch.rootThreadLaunch;
              return llvm::Error::success();
            }))
      fail(llvm::toString(std::move(error)));
  }
  require(consumerRoot.has_value(),
          "fixture did not identify its consumer root");
  for (auto &binding : partition.bindings)
    if (const auto *root =
            std::get_if<dataflow::RootThreadLaunchRef>(&binding.key);
        root && *root == *consumerRoot) {
      require(binding.cells.size() == 1,
              "whole-domain consumer binding is not singleton");
      const auto legal = binding.cells.front();
      binding.cells = {bounded(legal, std::nullopt, 1),
                       bounded(legal, 2, std::nullopt)};
    }

  const auto config =
      take(loom::pnr::projectResolvedSystemPnrConfigView(resolved));
  auto searchDomain = take(loom::pnr::projectSystemPnrSearchDomain(
      dataflow, system, config, constraints, partition,
      loom::pnr::SystemHierarchicalGraphSearchInput{{spatialMapping}}, store));
  const ImmediateStopSource stop;
  const auto interruptedSystem =
      loom::pnr::generateSystemMappings({dataflow,
                                         system,
                                         {},
                                         searchDomain,
                                         config,
                                         constraints,
                                         store,
                                         stop.control()});
  const auto *interrupted =
      std::get_if<loom::pnr::InterruptedSystemPnrGeneration>(
          &interruptedSystem);
  require(interrupted &&
              interrupted->snapshot.stage ==
                  loom::pnr::SystemPnrInterruptionStage::InputAdmission &&
              interrupted->snapshot.frontier.seedAttemptSlots == 0 &&
              !interrupted->snapshot.bestSelectedRank &&
              !interrupted->snapshot.closureResidual.violationValues,
          "System generator interruption lost its typed empty frontier");
  auto problem = take(loom::pnr::freezeSystemPnrProblemWithNormalizedTiming(
      dataflow, system, searchDomain, config, constraints, store));

  require(problem->threadDecisions().size() == 3,
          "consumer partition did not create two execution decisions");

  const auto channelService =
      llvm::find_if(problem->serviceDomains(), [&](const auto &service) {
        const auto *transfer =
            std::get_if<loom::mapping::TransferObligationFamilyKey>(
                &service.key);
        return transfer && channelProducer && *transfer == *channelProducer;
      });
  require(channelService != problem->serviceDomains().end(),
          "frozen problem omitted the channel obligation");
  const loom::pnr::PnrIndex channelServiceOrdinal =
      static_cast<loom::pnr::PnrIndex>(channelService -
                                       problem->serviceDomains().begin());
  std::vector<loom::pnr::PnrIndex> channelContexts;
  for (const auto &[ordinal, serviceContext] :
       llvm::enumerate(problem->serviceContexts()))
    if (serviceContext.service == channelServiceOrdinal)
      channelContexts.push_back(static_cast<loom::pnr::PnrIndex>(ordinal));
  require(channelContexts.size() == 2,
          "channel producer domain did not split into two plan contexts");
  std::vector<std::size_t> applicableCounts;
  for (loom::pnr::PnrIndex contextOrdinal : channelContexts) {
    const auto &serviceContext = problem->serviceContexts()[contextOrdinal];
    require(!serviceContext.cells.empty(),
            "channel plan context has no selection relation");
    applicableCounts.push_back(serviceContext.applicableMessageSinks.size());
  }
  llvm::sort(applicableCounts);
  require(applicableCounts == std::vector<std::size_t>({0, 2}),
          "source_map did not derive one empty and one two-owner plan");

  std::vector<loom::pnr::PnrIndex> consumerDecisions;
  for (const auto &[ordinal, decision] :
       llvm::enumerate(problem->threadDecisions()))
    if (decision.root == *consumerRoot)
      consumerDecisions.push_back(static_cast<loom::pnr::PnrIndex>(ordinal));
  require(consumerDecisions.size() == 2,
          "consumer execution partition did not remain exact");

  const auto selectedCoreOrdinal = [&](loom::pnr::PnrIndex decision,
                                       loom::pnr::PnrIndex choice) {
    const auto domain = problem->threadChoiceCatalogOrdinals(decision);
    require(choice < domain.size(), "test choice is outside its thread domain");
    return domain[choice];
  };
  const std::vector<loom::pnr::PnrIndex> graphChoices(
      problem->graphDecisions().size(), 0);
  auto findCandidate = [&](bool requireDistinctOwners) {
    loom::pnr::SystemCandidateStateHandle selected;
    std::string lastDiagnostic;
    std::vector<loom::pnr::PnrIndex> threadChoices(
        problem->threadDecisions().size(), 0);
    const auto firstDomain =
        problem->threadChoiceCatalogOrdinals(consumerDecisions[0]);
    const auto secondDomain =
        problem->threadChoiceCatalogOrdinals(consumerDecisions[1]);
    for (loom::pnr::PnrIndex first = 0; first < firstDomain.size() && !selected;
         ++first) {
      for (loom::pnr::PnrIndex second = 0;
           second < secondDomain.size() && !selected; ++second) {
        const bool distinct =
            selectedCoreOrdinal(consumerDecisions[0], first) !=
            selectedCoreOrdinal(consumerDecisions[1], second);
        if (distinct != requireDistinctOwners)
          continue;
        threadChoices[consumerDecisions[0]] = first;
        threadChoices[consumerDecisions[1]] = second;
        auto candidate = loom::pnr::initializeSystemCandidate(
            problem, threadChoices, graphChoices);
        if (candidate)
          selected = std::move(*candidate);
        else
          lastDiagnostic = llvm::toString(candidate.takeError());
      }
    }
    require(static_cast<bool>(selected),
            requireDistinctOwners
                ? "realistic Fabric has no routed distinct-owner assignment: " +
                      lastDiagnostic
                : "realistic Fabric has no routed same-owner assignment: " +
                      lastDiagnostic);
    return selected;
  };

  auto distinctOwners = findCandidate(true);
  auto sameOwner = findCandidate(false);
  require(distinctOwners->selectedAccCore(consumerDecisions[0]) !=
                  distinctOwners->selectedAccCore(consumerDecisions[1]) &&
              sameOwner->selectedAccCore(consumerDecisions[0]) ==
                  sameOwner->selectedAccCore(consumerDecisions[1]),
          "candidate fixtures do not select the requested owner relation");

  const auto exerciseSearchProbeDomain =
      [&](const loom::pnr::FrozenSystemPnrProblemHandle &scenarioProblem,
          SystemSearchProbeDomain probeDomain) {
        auto firstAnnealed = take(loom::pnr::initializeSystemCandidate(
            scenarioProblem, sameOwner->threadChoices(),
            sameOwner->graphChoices()));
        auto secondAnnealed = take(loom::pnr::initializeSystemCandidate(
            scenarioProblem, sameOwner->threadChoices(),
            sameOwner->graphChoices()));
        loom::pnr::SystemActionDomainScratch actionDomain;
        if (llvm::Error error = actionDomain.rebuild(*firstAnnealed))
          fail(llvm::toString(std::move(error)));
        require(!actionDomain.view().bindingAnchors.empty() &&
                    !actionDomain.view().routingAnchors.empty(),
                "System search did not expose independent binding and routing "
                "Action domains");
        loom::pnr::SystemAnnealingSearchScratch firstSearch;
        loom::pnr::SystemAnnealingSearchScratch secondSearch;
        auto firstRun = firstSearch.run(firstAnnealed, 0);
        if (!firstRun)
          fail(llvm::Twine(probeDomain == SystemSearchProbeDomain::Assignment
                               ? "assignment-domain"
                               : "routing-domain") +
               " search failed: " + llvm::toString(firstRun.takeError()));
        auto secondRun = secondSearch.run(secondAnnealed, 0);
        if (!secondRun)
          fail(llvm::Twine(probeDomain == SystemSearchProbeDomain::Assignment
                               ? "assignment-domain"
                               : "routing-domain") +
               " replay failed: " + llvm::toString(secondRun.takeError()));
        const auto firstStatistics = *firstRun;
        const auto secondStatistics = *secondRun;
        require(firstStatistics == secondStatistics,
                "System transactional search changed work on replay");
        if (probeDomain == SystemSearchProbeDomain::Assignment)
          require(firstStatistics.assignmentAttempts != 0,
                  "assignment-domain search performed no assignment probes");
        else
          require(firstStatistics.endpointExpansions != 0,
                  "routing-domain search performed no endpoint probes");
        require(
            firstAnnealed->threadChoices() == secondAnnealed->threadChoices() &&
                firstAnnealed->graphChoices() ==
                    secondAnnealed->graphChoices(),
            "System transactional search changed decisions on replay");
        if (llvm::Error error = firstAnnealed->verify())
          fail(llvm::toString(std::move(error)));
        if (llvm::Error error = secondAnnealed->verify())
          fail(llvm::toString(std::move(error)));
        auto firstDraft = take(loom::pnr::materializeSystemCandidateDraft(
            *firstAnnealed, context));
        auto secondDraft = take(loom::pnr::materializeSystemCandidateDraft(
            *secondAnnealed, context));
        const auto firstBytes =
            take(loom::mapping::writeCanonicalSystemMappingAssembly(
                mlir::cast<::mapping::SystemOp>(firstDraft.get())));
        const auto secondBytes =
            take(loom::mapping::writeCanonicalSystemMappingAssembly(
                mlir::cast<::mapping::SystemOp>(secondDraft.get())));
        require(firstBytes.bytes() == secondBytes.bytes(),
                "System transactional search changed canonical replay output");
      };

  exerciseSearchProbeDomain(problem, SystemSearchProbeDomain::Routing);
  const loom::ResolvedConfig assignmentResolved =
      buildResolvedConfig(SystemSearchProbeDomain::Assignment);
  const auto assignmentConfig =
      take(loom::pnr::projectResolvedSystemPnrConfigView(assignmentResolved));
  auto assignmentSearchDomain = take(loom::pnr::projectSystemPnrSearchDomain(
      dataflow, system, assignmentConfig, constraints, partition,
      loom::pnr::SystemHierarchicalGraphSearchInput{{spatialMapping}}, store));
  auto assignmentProblem =
      take(loom::pnr::freezeSystemPnrProblemWithNormalizedTiming(
          dataflow, system, assignmentSearchDomain, assignmentConfig,
          constraints,
          store));
  require(assignmentProblem->threadDecisions().size() ==
                  problem->threadDecisions().size() &&
              assignmentProblem->graphDecisions().size() ==
                  problem->graphDecisions().size(),
          "System probe scenarios changed the frozen decision domain");
  exerciseSearchProbeDomain(assignmentProblem,
                            SystemSearchProbeDomain::Assignment);

  std::optional<loom::pnr::PnrIndex> channelLeg;
  for (const auto &[ordinal, leg] : llvm::enumerate(problem->serviceLegs()))
    if (leg.key.obligation == channelService->key) {
      require(!channelLeg, "channel applicability created duplicate legs");
      channelLeg = static_cast<loom::pnr::PnrIndex>(ordinal);
    }
  require(channelLeg.has_value(),
          "active channel plan has no frozen route leg");
  require(distinctOwners->serviceRoutes()[*channelLeg].sinkCount == 2,
          "distinct owners did not materialize two route branches");
  require(sameOwner->serviceRoutes()[*channelLeg].sinkCount == 1,
          "same-owner consumer points did not collapse to one route branch");

  auto materialized = take(
      loom::pnr::materializeSystemCandidateDraft(*distinctOwners, context));
  auto mappingRoot = mlir::cast<::mapping::SystemOp>(materialized.get());
  ::mapping::ServiceRealizationOp channelRealization;
  for (auto service : mappingRoot.getBody()
                          .front()
                          .getOps<::mapping::ServiceRealizationOp>()) {
    auto key = take(loom::mapping::decodeSystemServiceObligationKey(
        unsignedBytes(service.getKey().getRecord()),
        problem->dataflowIdentity()));
    if (key == channelService->key)
      channelRealization = service;
  }
  require(channelRealization,
          "materialized Mapping omitted the channel realization");
  std::size_t emptyPlans = 0;
  std::size_t routedPlans = 0;
  for (auto plan : channelRealization.getBody()
                       .front()
                       .getOps<::mapping::ServicePlanOp>()) {
    auto routes =
        plan.getBody().front().getOps<::mapping::TransferLegRealizationOp>();
    if (routes.begin() == routes.end()) {
      ++emptyPlans;
      continue;
    }
    require(llvm::hasSingleElement(routes),
            "channel plan contains more than one message leg");
    auto route = *routes.begin();
    auto nodes = route.getBody().front().getOps<::mapping::SystemRouteNodeOp>();
    require(std::distance(nodes.begin(), nodes.end()) + 1 >= 8,
            "distinct-owner channel did not exercise a multi-hop ring route");
    auto sinks = route.getBody().front().getOps<::mapping::SystemRouteSinkOp>();
    require(std::distance(sinks.begin(), sinks.end()) == 2,
            "materialized distinct-owner route lost a branch");
    auto first = *sinks.begin();
    auto second = *std::next(sinks.begin());
    require(first.getTerminal() == second.getTerminal() &&
                first.getNodeOrdinal() != second.getNodeOrdinal(),
            "same terminal was not attached to two distinct owner nodes");
    ++routedPlans;
  }
  require(emptyPlans == 1 && routedPlans == 1,
          "non-surjective source_map did not materialize exact plans");
  const auto canonical =
      take(loom::mapping::writeCanonicalSystemMappingAssembly(mappingRoot));

  const auto findChannelPlans = [&](::mapping::SystemOp root) {
    ::mapping::ServicePlanOp emptyPlan;
    ::mapping::TransferLegRealizationOp routedLeg;
    for (auto service :
         root.getBody().front().getOps<::mapping::ServiceRealizationOp>()) {
      auto key = take(loom::mapping::decodeSystemServiceObligationKey(
          unsignedBytes(service.getKey().getRecord()),
          problem->dataflowIdentity()));
      if (key != channelService->key)
        continue;
      for (auto plan :
           service.getBody().front().getOps<::mapping::ServicePlanOp>()) {
        auto routes = plan.getBody()
                          .front()
                          .getOps<::mapping::TransferLegRealizationOp>();
        if (routes.empty()) {
          emptyPlan = plan;
          continue;
        }
        require(llvm::hasSingleElement(routes),
                "channel adverse fixture has multiple routes");
        routedLeg = *routes.begin();
      }
    }
    require(emptyPlan && routedLeg,
            "channel adverse fixture lost its empty or routed plan");
    return std::make_pair(emptyPlan, routedLeg);
  };

  mlir::OwningOpRef<mlir::Operation *> missingPairDraft(mappingRoot->clone());
  auto missingPairRoot =
      mlir::cast<::mapping::SystemOp>(missingPairDraft.get());
  auto [missingEmptyPlan, missingRoutedLeg] = findChannelPlans(missingPairRoot);
  (void)missingEmptyPlan;
  auto missingSinks =
      missingRoutedLeg.getBody().front().getOps<::mapping::SystemRouteSinkOp>();
  require(std::distance(missingSinks.begin(), missingSinks.end()) == 2,
          "missing-pair fixture did not start with two pairs");
  std::uint64_t removedNode = (*missingSinks.begin()).getNodeOrdinal();
  (*missingSinks.begin()).erase();
  while (removedNode != 0) {
    auto nodes = missingRoutedLeg.getBody()
                     .front()
                     .getOps<::mapping::SystemRouteNodeOp>();
    auto node = llvm::find_if(nodes, [&](::mapping::SystemRouteNodeOp value) {
      return value.getNodeOrdinal() == removedNode;
    });
    require(node != nodes.end(),
            "missing-pair fixture cannot resolve its removed branch");
    const bool hasChild = llvm::any_of(nodes, [&](auto value) {
      return value.getParentNodeOrdinal() == removedNode;
    });
    const bool hasSink = llvm::any_of(
        missingRoutedLeg.getBody()
            .front()
            .getOps<::mapping::SystemRouteSinkOp>(),
        [&](auto value) { return value.getNodeOrdinal() == removedNode; });
    if (hasChild || hasSink)
      break;
    const std::uint64_t parent = (*node).getParentNodeOrdinal();
    (*node).erase();
    removedNode = parent;
  }
  requireFailureContains(loom::mapping::verifySystemMappingBase(
                             missingPairRoot, dataflow, system, store),
                         "applicable sink-owner set");

  mlir::OwningOpRef<mlir::Operation *> nonSinkLeafDraft(mappingRoot->clone());
  auto nonSinkLeafRoot =
      mlir::cast<::mapping::SystemOp>(nonSinkLeafDraft.get());
  auto [nonSinkLeafEmptyPlan, nonSinkLeafLeg] =
      findChannelPlans(nonSinkLeafRoot);
  (void)nonSinkLeafEmptyPlan;
  auto nonSinkLeafSinks =
      nonSinkLeafLeg.getBody().front().getOps<::mapping::SystemRouteSinkOp>();
  (*nonSinkLeafSinks.begin()).erase();
  requireFailureContains(loom::mapping::verifySystemMappingBase(
                             nonSinkLeafRoot, dataflow, system, store),
                         "non-sink leaf");

  mlir::OwningOpRef<mlir::Operation *> handshakeCycleDraft(
      mappingRoot->clone());
  auto handshakeCycleRoot =
      mlir::cast<::mapping::SystemOp>(handshakeCycleDraft.get());
  ::mapping::TransferLegRealizationOp cycleLeg;
  ::mapping::SystemRouteNodeOp cycleChild;
  ::mapping::SystemRouteSinkOp cycleSink;
  std::uint64_t cycleParent = 0;
  std::vector<loom::fabric::FabricPhysicalTraversalRef> cycle;
  for (auto service : handshakeCycleRoot.getBody()
                          .front()
                          .getOps<::mapping::ServiceRealizationOp>()) {
    if (cycleLeg)
      break;
    for (auto plan :
         service.getBody().front().getOps<::mapping::ServicePlanOp>()) {
      if (cycleLeg)
        break;
      for (auto leg : plan.getBody()
                          .front()
                          .getOps<::mapping::TransferLegRealizationOp>()) {
        if (cycleLeg)
          break;
        auto nodes =
            leg.getBody().front().getOps<::mapping::SystemRouteNodeOp>();
        auto sinks =
            leg.getBody().front().getOps<::mapping::SystemRouteSinkOp>();
        std::vector<
            std::pair<std::uint64_t,
                      std::vector<loom::fabric::FabricTransportEndpointRef>>>
            positions;
        positions.push_back(
            {0,
             {take(loom::fabric::decodeFabricRef<
                   loom::fabric::FabricTransportEndpointRef>(
                 unsignedBytes(leg.getRootEndpoint().getRecord())))}});
        for (auto node : nodes) {
          const auto reference = take(loom::fabric::decodeFabricRef<
                                      loom::fabric::FabricPhysicalTraversalRef>(
              unsignedBytes(node.getIncomingTraversal().getRecord())));
          const auto traversal = llvm::find_if(
              system.artifact().physicalTraversals(),
              [&](const auto &value) { return value.reference == reference; });
          require(traversal != system.artifact().physicalTraversals().end(),
                  "handshake-cycle fixture names an absent traversal");
          positions.emplace_back(node.getNodeOrdinal(),
                                 traversal->destinations);
        }
        for (const auto &position : positions) {
          const std::uint64_t ordinal = position.first;
          const auto &endpoints = position.second;
          auto child = llvm::find_if(nodes, [&](auto value) {
            return value.getParentNodeOrdinal() == ordinal;
          });
          auto sink = llvm::find_if(sinks, [&](auto value) {
            return value.getNodeOrdinal() == ordinal;
          });
          if (child == nodes.end() && sink == sinks.end())
            continue;
          if (child == nodes.end() && endpoints.size() != 1)
            continue;
          for (const auto endpoint : endpoints) {
            auto found = findPhysicalTraversalCycle(system, endpoint);
            if (!found)
              continue;
            cycleLeg = leg;
            cycleParent = ordinal;
            if (child != nodes.end())
              cycleChild = *child;
            else
              cycleSink = *sink;
            cycle = std::move(*found);
            break;
          }
          if (cycleLeg)
            break;
        }
      }
    }
  }
  require(cycleLeg && !cycle.empty(),
          "finite-degree System fixture exposes no routed physical cycle");
  std::uint64_t nextNode = 1;
  for (auto node :
       cycleLeg.getBody().front().getOps<::mapping::SystemRouteNodeOp>())
    nextNode = std::max(nextNode, node.getNodeOrdinal() + 1);
  std::uint64_t parent = cycleParent;
  mlir::OpBuilder cycleBuilder(&context);
  cycleBuilder.setInsertionPoint(cycleChild ? cycleChild.getOperation()
                                            : cycleSink.getOperation());
  for (const auto traversal : cycle) {
    ::mapping::SystemRouteNodeOp::create(
        cycleBuilder, cycleBuilder.getUnknownLoc(), nextNode, parent,
        ::mapping::FabricPhysicalTraversalRefAttr::get(
            &context, bytesAttr(&context, loom::fabric::canonicalFabricBytes(
                                              traversal))));
    parent = nextNode++;
  }
  if (cycleChild)
    cycleChild.setParentNodeOrdinal(parent);
  else
    cycleSink.setNodeOrdinal(parent);
  const auto physicalCycleVerification = loom::mapping::verifySystemMappingBase(
      handshakeCycleRoot, dataflow, system, store);
  require(std::holds_alternative<loom::mapping::VerifiedSystemMappingBase>(
              physicalCycleVerification),
          "row-aware handshake projection rejected a physically cyclic route "
          "whose selected activation has a combinational break");

  mlir::OwningOpRef<mlir::Operation *> duplicatePairDraft(mappingRoot->clone());
  auto duplicatePairRoot =
      mlir::cast<::mapping::SystemOp>(duplicatePairDraft.get());
  auto [duplicateEmptyPlan, duplicateRoutedLeg] =
      findChannelPlans(duplicatePairRoot);
  (void)duplicateEmptyPlan;
  auto duplicateSinks = duplicateRoutedLeg.getBody()
                            .front()
                            .getOps<::mapping::SystemRouteSinkOp>();
  duplicateRoutedLeg.getBody().front().getOperations().push_back(
      (*duplicateSinks.begin()).getOperation()->clone());
  requireFailureContains(loom::mapping::verifySystemMappingBase(
                             duplicatePairRoot, dataflow, system, store),
                         "structurally invalid");

  mlir::OwningOpRef<mlir::Operation *> inactivePairDraft(mappingRoot->clone());
  auto inactivePairRoot =
      mlir::cast<::mapping::SystemOp>(inactivePairDraft.get());
  auto [inactiveEmptyPlan, inactiveRoutedLeg] =
      findChannelPlans(inactivePairRoot);
  inactiveEmptyPlan.getBody().front().getOperations().push_back(
      inactiveRoutedLeg->clone());
  requireFailureContains(loom::mapping::verifySystemMappingBase(
                             inactivePairRoot, dataflow, system, store),
                         "applicable sink-owner set");

  for (std::size_t replay = 1; replay != 3; ++replay) {
    auto replayed = take(
        loom::pnr::materializeSystemCandidateDraft(*distinctOwners, context));
    auto replayedBytes =
        take(loom::mapping::writeCanonicalSystemMappingAssembly(
            mlir::cast<::mapping::SystemOp>(replayed.get())));
    require(replayedBytes.bytes() == canonical.bytes(),
            "channel Mapping changed across canonical replay");
  }

  std::vector<loom::pnr::SystemServiceRouteSelection> staleRoutes;
  std::vector<loom::pnr::SystemServiceRouteNodeSelection> staleNodes;
  std::vector<loom::pnr::SystemServiceRouteSinkSelection> staleSinks;
  staleRoutes.reserve(sameOwner->serviceRoutes().size());
  for (std::size_t ordinal = 0; ordinal != sameOwner->serviceRoutes().size();
       ++ordinal) {
    const auto &candidate =
        ordinal == *channelLeg ? *distinctOwners : *sameOwner;
    auto route = candidate.serviceRoutes()[ordinal];
    const auto nodes =
        candidate.serviceRouteNodes().slice(route.nodeOffset, route.nodeCount);
    const auto sinks =
        candidate.serviceRouteSinks().slice(route.sinkOffset, route.sinkCount);
    route.nodeOffset = static_cast<loom::pnr::PnrIndex>(staleNodes.size());
    route.sinkOffset = static_cast<loom::pnr::PnrIndex>(staleSinks.size());
    staleRoutes.push_back(route);
    staleNodes.insert(staleNodes.end(), nodes.begin(), nodes.end());
    staleSinks.insert(staleSinks.end(), sinks.begin(), sinks.end());
  }
  requireFailureContains(
      loom::pnr::SystemCandidateState::create(
          problem,
          {sameOwner->threadChoices(), sameOwner->graphChoices(), staleRoutes,
           staleNodes, staleSinks, sameOwner->serviceTargets(),
           sameOwner->instructionResourceUses(),
           sameOwner->serviceResourceUses()}),
      "applicable sink-owner set");
  return EXIT_SUCCESS;
}
