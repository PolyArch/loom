#include "Mapping/Artifact/SystemMappingConstraintSet.h"
#include "ADG/Builtin.h"
#include "Common/ArtifactLocalReference.h"
#include "Common/ArtifactStore.h"
#include "Config/ResolvedConfig.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Dataflow/IR/DataflowOps.h"
#include "Dataflow/IR/DataflowReferenceCodec.h"
#include "Fabric/Artifact/FabricSystemRootView.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Mapping/IR/MappingDialect.h"
#include "Mapping/Tech/TechMappingConfig.h"
#include "Mapping/Tech/TechMappingGenerator.h"
#include "PnR/MappingObjective.h"
#include "PnR/PnrConfig.h"
#include "PnR/SpatialPnrGenerator.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <cstdlib>
#include <limits>
#include <optional>
#include <system_error>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace {

[[noreturn]] void fail(const llvm::Twine &message) {
  llvm::errs() << "System MappingConstraintSet anchor failed: " << message
               << '\n';
  std::exit(EXIT_FAILURE);
}

template <typename T> T take(llvm::Expected<T> value) {
  if (!value)
    fail(llvm::toString(value.takeError()));
  return std::move(*value);
}

void require(bool condition, const llvm::Twine &message) {
  if (!condition)
    fail(message);
}

template <typename T>
void requireFailure(llvm::Expected<T> value, const llvm::Twine &message) {
  if (value)
    fail(message);
  llvm::consumeError(value.takeError());
}

template <typename T>
void requireFailureContains(llvm::Expected<T> value,
                            llvm::StringRef expectedDiagnostic,
                            const llvm::Twine &message) {
  if (value)
    fail(message);
  const std::string diagnostic = llvm::toString(value.takeError());
  if (!llvm::StringRef(diagnostic).contains(expectedDiagnostic))
    fail(message + ": " + diagnostic);
}

class TemporaryDirectory final {
public:
  TemporaryDirectory() {
    std::error_code error = llvm::sys::fs::createUniqueDirectory(
        "loom-system-mapping-constraints", path_);
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
                  mlir::DLTIDialect, mlir::func::FuncDialect>();
  return mlir::MLIRContext(registry, mlir::MLIRContext::Threading::DISABLED);
}

loom::ResolvedObjectiveCatalogs spatialObjectiveCatalogs() {
  loom::ResolvedObjectiveCatalogs catalogs;
  constexpr std::uint64_t maximum = std::numeric_limits<std::uint64_t>::max();
  catalogs.dimensions = {
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
       maximum},
  };
  catalogs.weightedLevels = {{{{0, 1}, {1, 1}, {2, 1}}}};
  catalogs.totalOrderings = {{{0}}};
  return catalogs;
}

loom::ResolvedConfig spatialGenerationConfig() {
  loom::ResolvedConfig resolved = loom::defaultResolvedConfig();
  resolved.dse.objectiveCatalogs = spatialObjectiveCatalogs();
  resolved.dse.techMapping.candidatePublicationLimit = 1;
  resolved.dse.spatialPnr.temporaryViolations.admitted = {
      loom::ResolvedPnrViolationKind::UnroutedObligation,
      loom::ResolvedPnrViolationKind::CapacityOveruse};
  resolved.dse.spatialPnr.objectiveSelection = {0, 0};
  auto &search = resolved.dse.spatialPnr.search;
  search.initializer.seedAttemptCount = 1;
  search.actionProposal = {1, 3, 2};
  search.annealing.calibrationProposalCount = 1;
  search.annealing.fallbackTemperature = 1;
  search.annealing.minimumTemperature = 1;
  search.annealing.coolingRatio = {1, 2};
  search.annealing.proposalsPerLevelBase = 1;
  search.annealing.proposalsPerMovableDecision = 0;
  search.exactRepair = {loom::ResolvedPnrExactRepairKind::CpSat, 64, 1024};
  return resolved;
}

loom::ArtifactRootReference
generateSpatialMapping(const dataflow::CanonicalDataflowProgramView &dataflow,
                       const loom::fabric::FinalizedFabricRoot &module,
                       loom::ArtifactStore &store) {
  const loom::ResolvedConfig resolved = spatialGenerationConfig();
  const auto techConfig =
      take(loom::mapping::projectResolvedTechMappingConfigView(resolved));
  const auto mappedGraph =
      llvm::find_if(dataflow.graphs(), [&](const auto &graph) {
        for (const auto &actor : dataflow.actors()) {
          if (actor.graph == graph.ref && mlir::isa<dataflow::SyncOp>(actor.op))
            return true;
        }
        return false;
      });
  if (mappedGraph == dataflow.graphs().end())
    fail("constraint fixture has no memory graph");
  const std::array<dataflow::GraphRef, 1> covers = {mappedGraph->ref};
  auto techOutcome = loom::mapping::generateTechMappings(
      {dataflow, covers, module.view(), techConfig, store});
  const auto *techCandidates =
      std::get_if<loom::mapping::GeneratedTechMappings>(&techOutcome);
  if (!techCandidates) {
    if (std::holds_alternative<loom::mapping::ProvenInfeasibleTechMapping>(
            techOutcome))
      fail("TechMapping generator proved the fixture infeasible");
    if (std::holds_alternative<loom::mapping::IncompleteTechMappingGeneration>(
            techOutcome))
      fail("TechMapping generator did not establish a fixture");
    if (const auto *invalid =
            std::get_if<loom::mapping::InvalidTechMappingGeneration>(
                &techOutcome))
      fail("TechMapping generator rejected the fixture: " +
           invalid->diagnostic);
    fail("TechMapping generator failed internally: " +
         std::get<loom::mapping::InternalTechMappingGeneration>(techOutcome)
             .diagnostic);
  }
  if (techCandidates->candidates.size() != 1)
    fail("TechMapping generator produced " +
         llvm::Twine(techCandidates->candidates.size()) + " candidates");
  auto tech = take(loom::mapping::importTechMapping(
      techCandidates->candidates.front(), store));
  auto constraints =
      take(loom::mapping::finalizeEmptySpatialMappingConstraintSet(
          dataflow, tech.view(), module.view(), store));
  const auto pnrConfig =
      take(loom::pnr::projectResolvedSpatialPnrConfigView(resolved));
  const auto physicalTiming =
      take(loom::fabric::projectNormalizedFabricPhysicalTimingProfile(
          module.view()));
  auto spatialOutcome = loom::pnr::generateSpatialMappings(
      {dataflow, tech.view(), module.view(), physicalTiming, pnrConfig,
       constraints.view(), store});
  const auto *spatialCandidates =
      std::get_if<loom::pnr::GeneratedSpatialMappings>(&spatialOutcome);
  if (!spatialCandidates)
    std::visit(
        [&](const auto &outcome) {
          using Outcome = std::decay_t<decltype(outcome)>;
          if constexpr (std::is_same_v<
                            Outcome,
                            loom::pnr::InterruptedSpatialPnrGeneration>)
            fail("Spatial PnR interrupted at " +
                 loom::pnr::spatialPnrInterruptionStageSpelling(
                     outcome.snapshot.stage));
          else if constexpr (!std::is_same_v<
                                 Outcome, loom::pnr::GeneratedSpatialMappings>)
            fail("Spatial PnR failed: " + outcome.diagnostic);
        },
        spatialOutcome);
  if (spatialCandidates->candidates.size() != 1)
    fail("Spatial PnR produced " +
         llvm::Twine(spatialCandidates->candidates.size()) + " candidates");
  return spatialCandidates->candidates.front();
}

loom::adg::FinalizedFabricDesign
buildUnattachedSpatialModule(const loom::ArtifactStore &store) {
  loom::adg::DesignBuilder design(store);
  auto expansion = take(loom::adg::expandBuiltinSpatialCore(
      design, loom::adg::BuiltinTargetPreset::Small));
  const auto bits128 = take(loom::adg::PortType::bits(128));
  expansion.outputs.front() =
      take(expansion.spatialCore.addFifo(expansion.outputs.front(),
                                         loom::adg::FifoSpec{bits128, 2, true}))
          .value();
  if (llvm::Error error = expansion.spatialCore.close(expansion.outputs))
    fail(llvm::toString(std::move(error)));
  return take(std::move(design).finalize());
}

dataflow::CanonicalDataflowArtifact buildDataflow(mlir::MLIRContext &context,
                                                  int value) {
  const std::string source = R"mlir(
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 64>>} {
  dataflow.graph private @sync(%ctrl: none, %x: i32) -> i32
      attributes {input_segments = array<i32: 1, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %result:2 = dataflow.sync %ctrl, %x
        : (none, i32) -> (none, i32)
    dataflow.graph.return values(%result#1 : i32) streams() memories()
        complete(%result#0 : none)
  }
  dataflow.graph private @service(%ctrl: none) -> ()
      attributes {input_segments = array<i32: 0, 0, 0>,
                  result_segments = array<i32: 0, 0, 0>} {
    %fenced = dataflow.fence %ctrl
        {contract = #dataflow.fence_contract<ordering = seq_cst,
                                             sync_scope = <system>>}
    dataflow.graph.return values() streams() memories()
        complete(%fenced : none)
  }
  dataflow.thread private @worker domain(#dataflow.thread_domain<dense>)(
      %x: i32) ctrl (%ctrl: none) {
    %done = dataflow.graph.launch @service deps(%ctrl)
        values() stream_inputs() memories() stream_outputs()
        : (none) -> none
    %synced, %complete = dataflow.graph.launch @sync deps(%done)
        values(%x) stream_inputs() memories() stream_outputs()
        : (none, i32) -> (i32, none)
    dataflow.thread.yield %complete : none
  }
  func.func private @host() {
    %value = arith.constant )mlir" +
                             std::to_string(value) + R"mlir( : i32
    %first = dataflow.thread.launch @worker(%value)
        : (i32) -> !dataflow.thread_token
    %second = dataflow.thread.launch @worker(%value)
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

mlir::DenseI8ArrayAttr denseBytes(mlir::MLIRContext *context,
                                  llvm::ArrayRef<std::uint8_t> bytes) {
  std::vector<std::int8_t> signedBytes;
  signedBytes.reserve(bytes.size());
  for (std::uint8_t byte : bytes)
    signedBytes.push_back(static_cast<std::int8_t>(byte));
  return mlir::DenseI8ArrayAttr::get(context, signedBytes);
}

::mapping::ArtifactIdentityAttr
identityAttr(mlir::MLIRContext *context,
             const loom::ArtifactIdentity &identity) {
  return ::mapping::ArtifactIdentityAttr::get(
      context, denseBytes(context, identity.bytes()));
}

::mapping::RootThreadLaunchRefAttr
rootThreadLaunchAttr(mlir::MLIRContext *context,
                     const loom::ArtifactIdentity &owner,
                     dataflow::RootThreadLaunchRef reference) {
  return ::mapping::RootThreadLaunchRefAttr::get(
      context,
      denseBytes(context,
                 take(dataflow::encodeDataflowReference(owner, reference))));
}

mlir::OwningOpRef<mlir::ModuleOp> buildRawConstraintModule(
    mlir::MLIRContext &context, const loom::ArtifactIdentity &dataflowIdentity,
    const loom::ArtifactIdentity &fabricIdentity,
    llvm::ArrayRef<dataflow::RootThreadLaunchRef> rootThreadLaunches,
    llvm::ArrayRef<loom::ArtifactRootReference> spatialMappings = {}) {
  mlir::OpBuilder builder(&context);
  auto module = mlir::ModuleOp::create(builder.getUnknownLoc());
  builder.setInsertionPointToStart(module.getBody());

  std::vector<mlir::Attribute> roots;
  roots.reserve(rootThreadLaunches.size());
  for (const auto root : rootThreadLaunches)
    roots.push_back(rootThreadLaunchAttr(&context, dataflowIdentity, root));

  std::vector<mlir::Attribute> mappingReferences;
  mappingReferences.reserve(spatialMappings.size());
  for (const auto &mapping : spatialMappings)
    mappingReferences.push_back(::mapping::ArtifactRootReferenceAttr::get(
        &context,
        denseBytes(&context, loom::encodeArtifactRootReference(mapping))));

  auto constraint = ::mapping::ConstraintsSystemOp::create(
      builder, builder.getUnknownLoc(),
      identityAttr(&context, dataflowIdentity),
      identityAttr(&context, fabricIdentity), builder.getArrayAttr(roots),
      builder.getArrayAttr(mappingReferences));
  constraint.getBody().emplaceBlock();
  return module;
}

template <typename Attr, typename Ref>
Attr dataflowAttr(mlir::MLIRContext *context,
                  const loom::ArtifactIdentity &owner, const Ref &reference) {
  return Attr::get(context,
                   denseBytes(context, take(dataflow::encodeDataflowReference(
                                           owner, reference))));
}

template <typename Attr, typename Ref>
Attr fabricAttr(mlir::MLIRContext *context, const Ref &reference) {
  return Attr::get(
      context,
      denseBytes(context, loom::fabric::canonicalFabricBytes(reference)));
}

::mapping::SystemServiceObligationKeyAttr
serviceObligationAttr(mlir::MLIRContext *context,
                      const loom::ArtifactIdentity &owner,
                      const loom::mapping::SystemServiceObligationKey &key) {
  return ::mapping::SystemServiceObligationKeyAttr::get(
      context,
      denseBytes(context, take(loom::mapping::encodeSystemServiceObligationKey(
                              owner, key))));
}

::mapping::CanonicalServiceLegKeyAttr
serviceLegAttr(mlir::MLIRContext *context, const loom::ArtifactIdentity &owner,
               const loom::mapping::CanonicalServiceLegKey &key) {
  return ::mapping::CanonicalServiceLegKeyAttr::get(
      context,
      denseBytes(context, take(loom::mapping::encodeCanonicalServiceLegKey(
                              owner, key))));
}

::mapping::SystemTransferTerminalKeyAttr
transferTerminalAttr(mlir::MLIRContext *context,
                     const loom::ArtifactIdentity &owner,
                     const loom::mapping::SystemTransferTerminalKey &key) {
  return ::mapping::SystemTransferTerminalKeyAttr::get(
      context,
      denseBytes(context, take(loom::mapping::encodeSystemTransferTerminalKey(
                              owner, key))));
}

void addRestriction(mlir::OpBuilder &builder,
                    ::mapping::ConstraintsSystemOp root,
                    ::mapping::SystemConstraintProjection projection,
                    mlir::Attribute subject,
                    llvm::ArrayRef<mlir::Attribute> domain) {
  builder.setInsertionPointToEnd(&root.getBody().front());
  mlir::OperationState state(
      builder.getUnknownLoc(),
      ::mapping::ConstraintDomainRestrictionOp::getOperationName());
  state.addAttribute(
      "projection",
      ::mapping::SystemConstraintProjectionKeyAttr::get(
          builder.getContext(), static_cast<std::uint32_t>(projection)));
  state.addAttribute("subject", subject);
  state.addAttribute("admissible_domain", builder.getArrayAttr(domain));
  builder.create(state);
}

loom::CanonicalSemanticBytes rawConstraintBytes(mlir::ModuleOp module) {
  auto root =
      llvm::cast<::mapping::ConstraintsSystemOp>(module.getBody()->front());
  std::string text;
  llvm::raw_string_ostream stream(text);
  root.print(stream, mlir::OpPrintingFlags().enableDebugInfo(false));
  stream << '\n';
  stream.flush();
  return loom::CanonicalSemanticBytes(
      std::vector<std::uint8_t>(text.begin(), text.end()));
}

void addThreadRestriction(
    mlir::OpBuilder &builder, ::mapping::ConstraintsSystemOp root,
    const loom::ArtifactIdentity &dataflowIdentity,
    dataflow::RootThreadLaunchRef subject,
    llvm::ArrayRef<loom::fabric::AccCoreOccurrenceRef> domain) {
  std::vector<mlir::Attribute> values;
  values.reserve(domain.size());
  for (const auto &core : domain)
    values.push_back(::mapping::FabricAccCoreOccurrenceRefAttr::get(
        builder.getContext(),
        denseBytes(builder.getContext(),
                   loom::fabric::canonicalFabricBytes(core))));
  addRestriction(
      builder, root, ::mapping::SystemConstraintProjection::ThreadTargetAccCore,
      rootThreadLaunchAttr(builder.getContext(), dataflowIdentity, subject),
      values);
}

void addThreadEquality(mlir::OpBuilder &builder,
                       ::mapping::ConstraintsSystemOp root,
                       const loom::ArtifactIdentity &dataflowIdentity,
                       llvm::ArrayRef<dataflow::RootThreadLaunchRef> subjects) {
  builder.setInsertionPointToEnd(&root.getBody().front());
  mlir::OperationState state(builder.getUnknownLoc(),
                             ::mapping::ConstraintEqualOp::getOperationName());
  state.addAttribute(
      "projection",
      ::mapping::SystemConstraintProjectionKeyAttr::get(
          builder.getContext(),
          static_cast<std::uint32_t>(
              ::mapping::SystemConstraintProjection::ThreadTargetAccCore)));
  std::vector<mlir::Attribute> values;
  values.reserve(subjects.size());
  for (const auto subject : subjects)
    values.push_back(
        rootThreadLaunchAttr(builder.getContext(), dataflowIdentity, subject));
  state.addAttribute("subjects", builder.getArrayAttr(values));
  builder.create(state);
}

void addThreadDisjoint(mlir::OpBuilder &builder,
                       ::mapping::ConstraintsSystemOp root,
                       const loom::ArtifactIdentity &dataflowIdentity,
                       llvm::ArrayRef<dataflow::RootThreadLaunchRef> subjects) {
  builder.setInsertionPointToEnd(&root.getBody().front());
  mlir::OperationState state(
      builder.getUnknownLoc(),
      ::mapping::ConstraintDisjointOp::getOperationName());
  state.addAttribute(
      "projection",
      ::mapping::SystemConstraintProjectionKeyAttr::get(
          builder.getContext(),
          static_cast<std::uint32_t>(
              ::mapping::SystemConstraintProjection::ThreadTargetAccCore)));
  std::vector<mlir::Attribute> values;
  values.reserve(subjects.size());
  for (const auto subject : subjects)
    values.push_back(
        rootThreadLaunchAttr(builder.getContext(), dataflowIdentity, subject));
  state.addAttribute("subjects", builder.getArrayAttr(values));
  builder.create(state);
}

} // namespace

int main() {
  TemporaryDirectory directory;
  loom::ArtifactStore store(directory.path());
  mlir::MLIRContext context = makeContext();

  auto dataflow = buildDataflow(context, 7);
  auto dataflowReference =
      take(dataflow::publishCanonicalDataflow(dataflow, store));
  (void)dataflowReference;
  auto dataflowView = take(dataflow.view());
  require(dataflowView.rootThreadLaunches().size() == 2,
          "fixture must expose two root thread launches");

  auto design = take(loom::adg::buildBuiltinTarget(
      store, loom::adg::BuiltinTargetPreset::Small));
  require(design.roots().size() == 1,
          "builtin target must publish one System root");
  auto system =
      take(loom::fabric::requireSystemRoot(design.roots().front().view()));
  require(design.roots().front().directDependencies().size() == 1,
          "builtin System must import one SpatialCore Module");
  auto spatialModule = take(loom::fabric::importEntireFabricRoot(
      design.roots().front().directDependencies().front().root, store));
  const auto spatialMapping =
      generateSpatialMapping(dataflowView, spatialModule, store);
  auto alternateDesign = buildUnattachedSpatialModule(store);
  auto alternateModule = take(loom::fabric::importEntireFabricRoot(
      alternateDesign.roots().front().reference(), store));
  require(alternateModule.view().identity() != spatialModule.view().identity(),
          "alternate SpatialCore unexpectedly has the same identity");
  const auto unattachedSpatialMapping =
      generateSpatialMapping(dataflowView, alternateModule, store);

  const auto firstRoot = dataflowView.rootThreadLaunches()[0].ref;
  const auto secondRoot = dataflowView.rootThreadLaunches()[1].ref;
  std::vector<dataflow::RootThreadLaunchRef> noncanonicalAuthoring{
      secondRoot, firstRoot, secondRoot};
  auto first = take(loom::mapping::finalizeEmptySystemMappingConstraintSet(
      dataflowView, system, noncanonicalAuthoring, store));
  auto second = take(loom::mapping::finalizeEmptySystemMappingConstraintSet(
      dataflowView, system, {firstRoot, secondRoot}, store));

  require(first.reference() == second.reference(),
          "root authoring order changed constraint identity");
  require(
      first.canonicalBytes().bytes().equals(second.canonicalBytes().bytes()),
      "root authoring order changed canonical bytes");
  require(first.view().dataflowIdentity() == dataflowView.identity() &&
              first.view().fabricIdentity() == system.artifact().identity(),
          "constraint view lost its exact D/F bindings");
  require(first.view().rootThreadLaunches().size() == 2 &&
              first.view().rootThreadLaunches()[0] == firstRoot &&
              first.view().rootThreadLaunches()[1] == secondRoot,
          "constraint view did not preserve canonical root launches");
  require(first.view().spatialMappingReferences().empty() &&
              first.view().clauseCount() == 0,
          "empty constraints manufactured result-time mapping facts");

  auto imported = take(loom::mapping::importSystemMappingConstraintSet(
      first.reference(), store));
  require(imported.reference() == first.reference() &&
              imported.canonicalBytes().bytes().equals(
                  first.canonicalBytes().bytes()) &&
              imported.view().rootThreadLaunches() ==
                  first.view().rootThreadLaunches(),
          "strict roundtrip changed the System constraint set");

  mlir::MLIRContext rawContext;
  rawContext.loadDialect<::mapping::MappingDialect>();
  auto constrainedModule = buildRawConstraintModule(
      rawContext, dataflowView.identity(), system.artifact().identity(),
      {firstRoot, secondRoot});
  auto constrainedRoot = llvm::cast<::mapping::ConstraintsSystemOp>(
      constrainedModule->getBody()->front());
  mlir::OpBuilder constrainedBuilder(&rawContext);
  const auto core = system.artifact().accCoreOccurrences().front();
  addThreadRestriction(constrainedBuilder, constrainedRoot,
                       dataflowView.identity(), firstRoot, {core});
  auto constrained = take(loom::mapping::finalizeSystemMappingConstraintSet(
      constrainedRoot, dataflowView, system, store));
  require(constrained.view().clauseCount() == 1,
          "System DomainRestriction was not retained");

  auto redundantModule = buildRawConstraintModule(
      rawContext, dataflowView.identity(), system.artifact().identity(),
      {firstRoot, secondRoot});
  auto redundantRoot = llvm::cast<::mapping::ConstraintsSystemOp>(
      redundantModule->getBody()->front());
  addThreadEquality(constrainedBuilder, redundantRoot, dataflowView.identity(),
                    {secondRoot, firstRoot});
  addThreadRestriction(constrainedBuilder, redundantRoot,
                       dataflowView.identity(), secondRoot, {core, core});
  addThreadRestriction(constrainedBuilder, redundantRoot,
                       dataflowView.identity(), firstRoot, {core});
  auto redundant = take(loom::mapping::finalizeSystemMappingConstraintSet(
      redundantRoot, dataflowView, system, store));

  auto normalizedModule = buildRawConstraintModule(
      rawContext, dataflowView.identity(), system.artifact().identity(),
      {firstRoot, secondRoot});
  auto normalizedRoot = llvm::cast<::mapping::ConstraintsSystemOp>(
      normalizedModule->getBody()->front());
  addThreadRestriction(constrainedBuilder, normalizedRoot,
                       dataflowView.identity(), firstRoot, {core});
  addThreadEquality(constrainedBuilder, normalizedRoot, dataflowView.identity(),
                    {firstRoot, secondRoot});
  auto normalized = take(loom::mapping::finalizeSystemMappingConstraintSet(
      normalizedRoot, dataflowView, system, store));
  require(redundant.reference() == normalized.reference() &&
              redundant.canonicalBytes().bytes().equals(
                  normalized.canonicalBytes().bytes()),
          "System clause authoring order or duplication changed identity");
  require(redundant.view().clauseCount() == 2,
          "System equality closure did not produce one restriction");

  std::vector<dataflow::RootedGraphLaunchRef> rootedGraphs;
  dataflowView.forEachRootedGraphLaunch(
      [&](dataflow::RootedGraphLaunchRef graph) {
        rootedGraphs.push_back(graph);
      });
  require(!rootedGraphs.empty(),
          "service fixture has no rooted graph launch subject");
  auto obligations = take(loom::mapping::projectSystemServiceObligations(
      dataflowView, {firstRoot, secondRoot}));
  const auto operationObligation =
      llvm::find_if(obligations, [](const auto &obligation) {
        return std::holds_alternative<
                   loom::mapping::OperationServiceObligationFamilyKey>(
                   obligation.key) &&
               !obligation.legs.empty();
      });
  require(operationObligation != obligations.end(),
          "service fixture has no operation-service obligation");
  const auto operationKey =
      std::get<loom::mapping::OperationServiceObligationFamilyKey>(
          operationObligation->key);
  const auto leg = operationObligation->legs.front();
  const loom::mapping::SystemTransferTerminalKey terminal =
      loom::mapping::SystemTransferSourceTerminalKey{leg};

  const auto &fabric = system.artifact();
  require(!fabric.systemMemoryServices().empty(),
          "builtin System has no memory service");
  const auto memoryService = loom::fabric::FabricMemoryServiceRef::system(
      fabric.systemMemoryServices().front());
  require(fabric.inventorySize(
              loom::fabric::FabricInventoryOwnerRef::of(memoryService),
              loom::fabric::FabricInventoryKind::MemoryServiceRegion) != 0,
          "builtin System memory service has no region");
  const loom::fabric::FabricMemoryServiceRegionRef serviceRegion{memoryService,
                                                                 0};
  require(!fabric.transportEndpoints().empty(),
          "builtin System has no transport endpoint");
  const auto transportEndpoint = fabric.transportEndpoints().front();
  require(!fabric.physicalTraversals().empty(),
          "builtin System has no physical traversal");
  const auto physicalTraversal = fabric.physicalTraversals().front().reference;
  const auto stateTraversal =
      llvm::find_if(fabric.physicalTraversals(), [](const auto &traversal) {
        return !traversal.resourceStates.empty();
      });
  require(stateTraversal != fabric.physicalTraversals().end(),
          "builtin System has no traversal resource state");
  const auto resourceState = stateTraversal->resourceStates.front();

  auto catalogModule = buildRawConstraintModule(
      rawContext, dataflowView.identity(), system.artifact().identity(),
      {firstRoot, secondRoot}, {spatialMapping});
  auto catalogRoot = llvm::cast<::mapping::ConstraintsSystemOp>(
      catalogModule->getBody()->front());
  addThreadRestriction(constrainedBuilder, catalogRoot, dataflowView.identity(),
                       firstRoot, {core});
  addRestriction(
      constrainedBuilder, catalogRoot,
      ::mapping::SystemConstraintProjection::GraphSelectedSpatialMapping,
      dataflowAttr<::mapping::RootedGraphLaunchRefAttr>(
          &rawContext, dataflowView.identity(), rootedGraphs.front()),
      {::mapping::ConstraintSpatialMappingReferenceAttr::get(&rawContext, 0)});
  addRestriction(
      constrainedBuilder, catalogRoot,
      ::mapping::SystemConstraintProjection::GraphTargetSpatialCore,
      dataflowAttr<::mapping::RootedGraphLaunchRefAttr>(
          &rawContext, dataflowView.identity(), rootedGraphs.front()),
      {fabricAttr<::mapping::FabricSpatialCoreOccurrenceRefAttr>(
          &rawContext, loom::fabric::SpatialCoreOccurrenceRef{core})});
  addRestriction(constrainedBuilder, catalogRoot,
                 ::mapping::SystemConstraintProjection::ServiceTargetRegion,
                 serviceObligationAttr(
                     &rawContext, dataflowView.identity(),
                     loom::mapping::SystemServiceObligationKey{operationKey}),
                 {fabricAttr<::mapping::FabricMemoryServiceRegionRefAttr>(
                     &rawContext, serviceRegion)});
  addRestriction(
      constrainedBuilder, catalogRoot,
      ::mapping::SystemConstraintProjection::TransferTerminalAttachment,
      transferTerminalAttr(&rawContext, dataflowView.identity(), terminal),
      {fabricAttr<::mapping::FabricTransportEndpointRefAttr>(
          &rawContext, transportEndpoint)});
  addRestriction(
      constrainedBuilder, catalogRoot,
      ::mapping::SystemConstraintProjection::TransferSelectedTraversals,
      serviceLegAttr(&rawContext, dataflowView.identity(), leg),
      {fabricAttr<::mapping::FabricPhysicalTraversalRefAttr>(
          &rawContext, physicalTraversal)});
  addRestriction(constrainedBuilder, catalogRoot,
                 ::mapping::SystemConstraintProjection::TransferResourceStates,
                 serviceLegAttr(&rawContext, dataflowView.identity(), leg),
                 {fabricAttr<::mapping::FabricResourceStateRefAttr>(
                     &rawContext, resourceState)});
  const auto ui8 = mlir::IntegerType::get(
      &rawContext, 8, mlir::IntegerType::SignednessSemantics::Unsigned);
  addRestriction(
      constrainedBuilder, catalogRoot,
      ::mapping::SystemConstraintProjection::TransferAssignedTagValues,
      serviceLegAttr(&rawContext, dataflowView.identity(), leg),
      {::mapping::ConstraintUnsignedIntervalAttr::get(
          &rawContext, mlir::IntegerAttr::get(ui8, 0),
          mlir::IntegerAttr::get(ui8, 2))});
  auto catalog = take(loom::mapping::finalizeSystemMappingConstraintSet(
      catalogRoot, dataflowView, system, store));
  require(catalog.view().clauses().size() == 8,
          "System projection catalog lost a typed clause");
  for (const auto &clause : catalog.view().clauses()) {
    const auto *restriction =
        std::get_if<loom::mapping::SystemDomainRestrictionView>(&clause);
    require(restriction && restriction->admissibleDomain.size() == 1,
            "System catalog clause did not import as one typed restriction");
    switch (restriction->projection) {
    case ::mapping::SystemConstraintProjection::ThreadTargetAccCore:
      require(std::holds_alternative<loom::fabric::AccCoreOccurrenceRef>(
                  restriction->admissibleDomain.front()),
              "thread target imported with the wrong carrier");
      break;
    case ::mapping::SystemConstraintProjection::GraphTargetSpatialCore:
      require(std::holds_alternative<loom::fabric::SpatialCoreOccurrenceRef>(
                  restriction->admissibleDomain.front()),
              "graph target imported with the wrong carrier");
      break;
    case ::mapping::SystemConstraintProjection::GraphSelectedSpatialMapping:
      require(std::holds_alternative<loom::ArtifactRootReference>(
                  restriction->admissibleDomain.front()) &&
                  std::get<loom::ArtifactRootReference>(
                      restriction->admissibleDomain.front()) == spatialMapping,
              "graph mapping imported with the wrong carrier");
      break;
    case ::mapping::SystemConstraintProjection::ServiceTargetRegion:
      require(
          std::holds_alternative<loom::fabric::FabricMemoryServiceRegionRef>(
              restriction->admissibleDomain.front()),
          "service target imported with the wrong carrier");
      break;
    case ::mapping::SystemConstraintProjection::TransferTerminalAttachment:
      require(std::holds_alternative<loom::fabric::FabricTransportEndpointRef>(
                  restriction->admissibleDomain.front()),
              "terminal attachment imported with the wrong carrier");
      break;
    case ::mapping::SystemConstraintProjection::TransferSelectedTraversals:
      require(std::holds_alternative<loom::fabric::FabricPhysicalTraversalRef>(
                  restriction->admissibleDomain.front()),
              "transfer traversal imported with the wrong carrier");
      break;
    case ::mapping::SystemConstraintProjection::TransferResourceStates:
      require(std::holds_alternative<loom::fabric::FabricResourceStateRef>(
                  restriction->admissibleDomain.front()),
              "transfer state imported with the wrong carrier");
      break;
    case ::mapping::SystemConstraintProjection::TransferAssignedTagValues:
      require(std::holds_alternative<
                  loom::mapping::SpatialConstraintUnsignedInterval>(
                  restriction->admissibleDomain.front()),
              "transfer tag imported with the wrong carrier");
      break;
    }
  }

  auto duplicateTableModule = buildRawConstraintModule(
      rawContext, dataflowView.identity(), system.artifact().identity(),
      {firstRoot, secondRoot}, {spatialMapping, spatialMapping});
  auto duplicateTableRoot = llvm::cast<::mapping::ConstraintsSystemOp>(
      duplicateTableModule->getBody()->front());
  addRestriction(
      constrainedBuilder, duplicateTableRoot,
      ::mapping::SystemConstraintProjection::GraphSelectedSpatialMapping,
      dataflowAttr<::mapping::RootedGraphLaunchRefAttr>(
          &rawContext, dataflowView.identity(), rootedGraphs.front()),
      {::mapping::ConstraintSpatialMappingReferenceAttr::get(&rawContext, 1),
       ::mapping::ConstraintSpatialMappingReferenceAttr::get(&rawContext, 0)});
  auto duplicateTable = take(loom::mapping::finalizeSystemMappingConstraintSet(
      duplicateTableRoot, dataflowView, system, store));
  auto canonicalTableModule = buildRawConstraintModule(
      rawContext, dataflowView.identity(), system.artifact().identity(),
      {firstRoot, secondRoot}, {spatialMapping});
  auto canonicalTableRoot = llvm::cast<::mapping::ConstraintsSystemOp>(
      canonicalTableModule->getBody()->front());
  addRestriction(
      constrainedBuilder, canonicalTableRoot,
      ::mapping::SystemConstraintProjection::GraphSelectedSpatialMapping,
      dataflowAttr<::mapping::RootedGraphLaunchRefAttr>(
          &rawContext, dataflowView.identity(), rootedGraphs.front()),
      {::mapping::ConstraintSpatialMappingReferenceAttr::get(&rawContext, 0)});
  auto canonicalTable = take(loom::mapping::finalizeSystemMappingConstraintSet(
      canonicalTableRoot, dataflowView, system, store));
  require(duplicateTable.reference() == canonicalTable.reference() &&
              duplicateTable.view().spatialMappingReferences().size() == 1,
          "SpatialMapping table order or duplication changed identity");

  auto unattachedModule = buildRawConstraintModule(
      rawContext, dataflowView.identity(), system.artifact().identity(),
      {firstRoot, secondRoot}, {unattachedSpatialMapping});
  auto unattachedRoot = llvm::cast<::mapping::ConstraintsSystemOp>(
      unattachedModule->getBody()->front());
  addRestriction(
      constrainedBuilder, unattachedRoot,
      ::mapping::SystemConstraintProjection::GraphSelectedSpatialMapping,
      dataflowAttr<::mapping::RootedGraphLaunchRefAttr>(
          &rawContext, dataflowView.identity(), rootedGraphs.front()),
      {::mapping::ConstraintSpatialMappingReferenceAttr::get(&rawContext, 0)});
  requireFailureContains(
      loom::mapping::finalizeSystemMappingConstraintSet(
          unattachedRoot, dataflowView, system, store),
      "not an attached Module",
      "SpatialMapping from an unattached Module was accepted");

  auto contradictoryModule = buildRawConstraintModule(
      rawContext, dataflowView.identity(), system.artifact().identity(),
      {firstRoot, secondRoot});
  auto contradictoryRoot = llvm::cast<::mapping::ConstraintsSystemOp>(
      contradictoryModule->getBody()->front());
  addThreadEquality(constrainedBuilder, contradictoryRoot,
                    dataflowView.identity(), {firstRoot, secondRoot});
  addThreadDisjoint(constrainedBuilder, contradictoryRoot,
                    dataflowView.identity(), {secondRoot, firstRoot});
  auto contradictory = take(loom::mapping::finalizeSystemMappingConstraintSet(
      contradictoryRoot, dataflowView, system, store));
  require(contradictory.view().clauses().size() == 2,
          "collapsed Disjoint did not retain equality and forced empty domain");
  const auto emptyRestriction =
      llvm::find_if(contradictory.view().clauses(), [](const auto &clause) {
        const auto *restriction =
            std::get_if<loom::mapping::SystemDomainRestrictionView>(&clause);
        return restriction && restriction->admissibleDomain.empty();
      });
  require(emptyRestriction != contradictory.view().clauses().end(),
          "collapsed Disjoint did not produce a forced empty restriction");

  auto staleOrdinalModule = buildRawConstraintModule(
      rawContext, dataflowView.identity(), system.artifact().identity(),
      {firstRoot, secondRoot}, {spatialMapping});
  auto staleOrdinalRoot = llvm::cast<::mapping::ConstraintsSystemOp>(
      staleOrdinalModule->getBody()->front());
  addRestriction(
      constrainedBuilder, staleOrdinalRoot,
      ::mapping::SystemConstraintProjection::GraphSelectedSpatialMapping,
      dataflowAttr<::mapping::RootedGraphLaunchRefAttr>(
          &rawContext, dataflowView.identity(), rootedGraphs.front()),
      {::mapping::ConstraintSpatialMappingReferenceAttr::get(&rawContext, 1)});
  requireFailureContains(
      loom::mapping::finalizeSystemMappingConstraintSet(
          staleOrdinalRoot, dataflowView, system, store),
      "SpatialMapping table ordinal is out of range",
      "out-of-range SpatialMapping table ordinal was accepted");

  auto unusedTableModule = buildRawConstraintModule(
      rawContext, dataflowView.identity(), system.artifact().identity(),
      {firstRoot, secondRoot}, {spatialMapping});
  auto unusedTableBytes = rawConstraintBytes(*unusedTableModule);
  auto unusedTableIdentity = take(
      store.put(loom::mapping::mappingConstraintSetSchema, unusedTableBytes));
  requireFailureContains(
      loom::mapping::importSystemMappingConstraintSet(
          {loom::mapping::mappingConstraintSetSchema.identity.str(),
           loom::mapping::mappingConstraintSetSchema.version,
           unusedTableIdentity},
          store),
      "stored System constraint payload is not canonical",
      "strict import accepted an unused SpatialMapping table row");

  auto outsideScopeModule =
      buildRawConstraintModule(rawContext, dataflowView.identity(),
                               system.artifact().identity(), {firstRoot});
  auto outsideScopeRoot = llvm::cast<::mapping::ConstraintsSystemOp>(
      outsideScopeModule->getBody()->front());
  addThreadRestriction(constrainedBuilder, outsideScopeRoot,
                       dataflowView.identity(), secondRoot, {core});
  requireFailureContains(
      loom::mapping::finalizeSystemMappingConstraintSet(
          outsideScopeRoot, dataflowView, system, store),
      "outside the root scope",
      "constraint subject outside the root scope was accepted");

  auto foreignCoreModule =
      buildRawConstraintModule(rawContext, dataflowView.identity(),
                               system.artifact().identity(), {firstRoot});
  auto foreignCoreRoot = llvm::cast<::mapping::ConstraintsSystemOp>(
      foreignCoreModule->getBody()->front());
  addThreadRestriction(constrainedBuilder, foreignCoreRoot,
                       dataflowView.identity(), firstRoot,
                       {loom::fabric::AccCoreOccurrenceRef(
                           std::numeric_limits<std::uint64_t>::max())});
  requireFailureContains(loom::mapping::finalizeSystemMappingConstraintSet(
                             foreignCoreRoot, dataflowView, system, store),
                         "does not resolve",
                         "foreign AccCore occurrence was accepted");

  auto wrongCarrierModule =
      buildRawConstraintModule(rawContext, dataflowView.identity(),
                               system.artifact().identity(), {firstRoot});
  auto wrongCarrierRoot = llvm::cast<::mapping::ConstraintsSystemOp>(
      wrongCarrierModule->getBody()->front());
  addRestriction(
      constrainedBuilder, wrongCarrierRoot,
      ::mapping::SystemConstraintProjection::ThreadTargetAccCore,
      rootThreadLaunchAttr(&rawContext, dataflowView.identity(), firstRoot),
      {fabricAttr<::mapping::FabricTransportEndpointRefAttr>(
          &rawContext, transportEndpoint)});
  requireFailure(
      loom::mapping::finalizeSystemMappingConstraintSet(
          wrongCarrierRoot, dataflowView, system, store),
      "System projection accepted a carrier from another typed domain");

  auto wrongProfileModule =
      buildRawConstraintModule(rawContext, dataflowView.identity(),
                               system.artifact().identity(), {firstRoot});
  auto wrongProfileRoot = llvm::cast<::mapping::ConstraintsSystemOp>(
      wrongProfileModule->getBody()->front());
  constrainedBuilder.setInsertionPointToEnd(
      &wrongProfileRoot.getBody().front());
  mlir::OperationState wrongProfileState(
      constrainedBuilder.getUnknownLoc(),
      ::mapping::ConstraintDomainRestrictionOp::getOperationName());
  wrongProfileState.addAttribute(
      "projection",
      ::mapping::SpatialConstraintProjectionKeyAttr::get(
          &rawContext,
          static_cast<std::uint32_t>(
              ::mapping::SpatialConstraintProjection::ComputePlacement)));
  wrongProfileState.addAttribute(
      "subject", ::mapping::ComputeRealizationRefAttr::get(&rawContext, 0));
  wrongProfileState.addAttribute("admissible_domain",
                                 constrainedBuilder.getArrayAttr({}));
  constrainedBuilder.create(wrongProfileState);
  requireFailure(loom::mapping::finalizeSystemMappingConstraintSet(
                     wrongProfileRoot, dataflowView, system, store),
                 "System root accepted a Spatial projection key");

  auto rawModule = buildRawConstraintModule(rawContext, dataflowView.identity(),
                                            system.artifact().identity(),
                                            noncanonicalAuthoring);
  auto rawBytes = rawConstraintBytes(*rawModule);
  require(!rawBytes.bytes().equals(first.canonicalBytes().bytes()),
          "raw persisted fixture accidentally became canonical");
  auto rawIdentity =
      take(store.put(loom::mapping::mappingConstraintSetSchema, rawBytes));
  const loom::ArtifactRootReference rawReference{
      loom::mapping::mappingConstraintSetSchema.identity.str(),
      loom::mapping::mappingConstraintSetSchema.version, rawIdentity};
  requireFailureContains(
      loom::mapping::importSystemMappingConstraintSet(rawReference, store),
      "stored System constraint payload is not canonical",
      "strict import accepted persisted unsorted duplicate references");

  requireFailure(loom::mapping::finalizeEmptySystemMappingConstraintSet(
                     dataflowView, system, {}, store),
                 "empty root launch coverage was accepted");

  auto foreignDataflow = buildDataflow(context, 8);
  take(dataflow::publishCanonicalDataflow(foreignDataflow, store));
  auto foreignView = take(foreignDataflow.view());
  requireFailure(loom::mapping::finalizeEmptySystemMappingConstraintSet(
                     dataflowView, system,
                     {foreignView.rootThreadLaunches().front().ref}, store),
                 "foreign root launch reference was accepted");

  std::vector<dataflow::RootedGraphLaunchRef> foreignGraphs;
  foreignView.forEachRootedGraphLaunch(
      [&](dataflow::RootedGraphLaunchRef graph) {
        foreignGraphs.push_back(graph);
      });
  auto foreignMappingModule = buildRawConstraintModule(
      rawContext, foreignView.identity(), system.artifact().identity(),
      {foreignView.rootThreadLaunches().front().ref}, {spatialMapping});
  auto foreignMappingRoot = llvm::cast<::mapping::ConstraintsSystemOp>(
      foreignMappingModule->getBody()->front());
  addRestriction(
      constrainedBuilder, foreignMappingRoot,
      ::mapping::SystemConstraintProjection::GraphSelectedSpatialMapping,
      dataflowAttr<::mapping::RootedGraphLaunchRefAttr>(
          &rawContext, foreignView.identity(), foreignGraphs.front()),
      {::mapping::ConstraintSpatialMappingReferenceAttr::get(&rawContext, 0)});
  requireFailureContains(
      loom::mapping::finalizeSystemMappingConstraintSet(
          foreignMappingRoot, foreignView, system, store),
      "foreign Dataflow owner",
      "SpatialMapping with a foreign Dataflow owner was accepted");

  llvm::outs() << "System MappingConstraintSet anchors passed\n";
  return EXIT_SUCCESS;
}
