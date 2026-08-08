#include "PnR/System/SystemCandidateState.h"
#include "ADG/Builtin.h"
#include "Common/ArtifactLocalReference.h"
#include "Common/ArtifactStore.h"
#include "Config/ResolvedConfig.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Dataflow/IR/DataflowReferenceCodec.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Artifact/FabricSystemRootView.h"
#include "Fabric/IR/ResourceContract.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Mapping/Artifact/MappingArtifact.h"
#include "Mapping/Artifact/MappingConstraintSet.h"
#include "Mapping/Artifact/SystemMappingArtifact.h"
#include "Mapping/Artifact/SystemMappingConstraintSet.h"
#include "Mapping/IR/MappingAttrs.h"
#include "Mapping/IR/MappingDialect.h"
#include "Mapping/Tech/TechMappingConfig.h"
#include "Mapping/Tech/TechMappingGenerator.h"
#include "PnR/MappingObjective.h"
#include "PnR/PnrConfig.h"
#include "PnR/SpatialPnrGenerator.h"
#include "PnR/System/SystemMappingMaterializer.h"
#include "PnR/System/SystemPnrProblem.h"
#include "PnR/System/SystemPnrSearchDomain.h"
#include "SystemCandidateStateTestSupport.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <limits>
#include <optional>
#include <string>
#include <system_error>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace {

using loom::pnr::test::byteList;
using loom::pnr::test::bytesAttr;
using loom::pnr::test::unsignedBytes;

[[noreturn]] void fail(const llvm::Twine &message) {
  llvm::errs() << "System CandidateState anchor failed: " << message << '\n';
  std::exit(EXIT_FAILURE);
}

void require(bool condition, const llvm::Twine &message) {
  if (!condition)
    fail(message);
}

template <typename T> T take(llvm::Expected<T> value) {
  if (!value)
    fail(llvm::toString(value.takeError()));
  return std::move(*value);
}

template <typename T>
void requireFailureContains(llvm::Expected<T> value,
                            llvm::StringRef diagnostic) {
  if (value)
    fail("adverse CandidateState input unexpectedly succeeded");
  const std::string actual = llvm::toString(value.takeError());
  require(llvm::StringRef(actual).contains(diagnostic),
          "adverse diagnostic changed: " + actual);
}

template <typename T>
void requireUnsupported(
    llvm::Expected<T> value,
    loom::pnr::UnsupportedSystemPnrSearchDomainReason expectedReason,
    llvm::StringRef diagnostic) {
  if (value)
    fail("unsupported System search-domain input unexpectedly succeeded");
  bool matched = false;
  llvm::Error remaining = llvm::handleErrors(
      value.takeError(),
      [&](const loom::pnr::UnsupportedSystemPnrSearchDomain &error) {
        matched = true;
        require(error.reason() == expectedReason,
                "unsupported System search-domain reason changed");
        std::string actual;
        llvm::raw_string_ostream stream(actual);
        error.log(stream);
        stream.flush();
        require(llvm::StringRef(actual).contains(diagnostic),
                "unsupported System search-domain diagnostic changed");
      });
  if (remaining)
    fail(llvm::toString(std::move(remaining)));
  require(matched, "System search-domain failure lost its typed reason");
}

void requireVerificationFailureContains(mlir::Operation *operation,
                                        llvm::StringRef expected) {
  std::vector<std::string> diagnostics;
  mlir::ScopedDiagnosticHandler capture(
      operation->getContext(), [&](mlir::Diagnostic &diagnostic) {
        diagnostics.push_back(diagnostic.str());
        return mlir::success();
      });
  require(mlir::failed(mlir::verify(operation)),
          "adverse SystemMapping operation unexpectedly verified");
  require(llvm::any_of(diagnostics,
                       [&](const std::string &diagnostic) {
                         return llvm::StringRef(diagnostic).contains(expected);
                       }),
          "adverse SystemMapping diagnostic changed");
}

std::vector<std::uint8_t>
replaceEvery(llvm::ArrayRef<std::uint8_t> bytes,
             llvm::ArrayRef<std::uint8_t> original,
             llvm::ArrayRef<std::uint8_t> replacement) {
  require(!original.empty() && original.size() == replacement.size(),
          "wire replacement fixture has incompatible fields");
  std::vector<std::uint8_t> result(bytes.begin(), bytes.end());
  std::size_t replacements = 0;
  auto cursor = result.begin();
  while (cursor != result.end()) {
    auto found =
        std::search(cursor, result.end(), original.begin(), original.end());
    if (found == result.end())
      break;
    std::copy(replacement.begin(), replacement.end(), found);
    cursor = found + replacement.size();
    ++replacements;
  }
  require(replacements != 0, "wire replacement fixture found no field");
  return result;
}

std::string identityAttr(const loom::ArtifactIdentity &identity) {
  return "#mapping.artifact_identity<" + byteList(identity.bytes()) + ">";
}

template <typename Ref>
std::string dataflowAttr(llvm::StringRef spelling,
                         const loom::ArtifactIdentity &identity,
                         const Ref &reference) {
  return "#mapping." + spelling.str() + "<" +
         byteList(
             take(dataflow::encodeDataflowReference(identity, reference))) +
         ">";
}

::mapping::ArtifactRootReferenceAttr
rootReferenceAttr(mlir::MLIRContext *context,
                  const loom::ArtifactRootReference &reference) {
  return ::mapping::ArtifactRootReferenceAttr::get(
      context,
      bytesAttr(context, loom::encodeArtifactRootReference(reference)));
}

template <typename Attr, typename Ref>
Attr constraintDataflowAttr(mlir::MLIRContext *context,
                            const loom::ArtifactIdentity &owner,
                            const Ref &reference) {
  return Attr::get(context,
                   bytesAttr(context, take(dataflow::encodeDataflowReference(
                                          owner, reference))));
}

template <typename Attr, typename Ref>
Attr constraintFabricAttr(mlir::MLIRContext *context, const Ref &reference) {
  return Attr::get(
      context,
      bytesAttr(context, loom::fabric::canonicalFabricBytes(reference)));
}

::mapping::SystemServiceObligationKeyAttr
serviceObligationAttr(mlir::MLIRContext *context,
                      const loom::ArtifactIdentity &owner,
                      const loom::mapping::SystemServiceObligationKey &key) {
  return ::mapping::SystemServiceObligationKeyAttr::get(
      context,
      bytesAttr(context, take(loom::mapping::encodeSystemServiceObligationKey(
                             owner, key))));
}

::mapping::SystemTransferTerminalKeyAttr
transferTerminalAttr(mlir::MLIRContext *context,
                     const loom::ArtifactIdentity &owner,
                     const loom::mapping::SystemTransferTerminalKey &key) {
  return ::mapping::SystemTransferTerminalKeyAttr::get(
      context,
      bytesAttr(context, take(loom::mapping::encodeSystemTransferTerminalKey(
                             owner, key))));
}

mlir::OwningOpRef<mlir::ModuleOp> buildSystemConstraintModule(
    mlir::MLIRContext &context, const loom::ArtifactIdentity &dataflowIdentity,
    const loom::ArtifactIdentity &fabricIdentity,
    llvm::ArrayRef<dataflow::RootThreadLaunchRef> roots) {
  mlir::OpBuilder builder(&context);
  auto module = mlir::ModuleOp::create(builder.getUnknownLoc());
  builder.setInsertionPointToStart(module.getBody());
  std::vector<mlir::Attribute> rootAttributes;
  rootAttributes.reserve(roots.size());
  for (const auto root : roots)
    rootAttributes.push_back(
        constraintDataflowAttr<::mapping::RootThreadLaunchRefAttr>(
            &context, dataflowIdentity, root));
  auto constraint = ::mapping::ConstraintsSystemOp::create(
      builder, builder.getUnknownLoc(),
      ::mapping::ArtifactIdentityAttr::get(
          &context, bytesAttr(&context, dataflowIdentity.bytes())),
      ::mapping::ArtifactIdentityAttr::get(
          &context, bytesAttr(&context, fabricIdentity.bytes())),
      builder.getArrayAttr(rootAttributes), builder.getArrayAttr({}));
  constraint.getBody().emplaceBlock();
  return module;
}

void addSystemRestriction(mlir::OpBuilder &builder,
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

void addSystemEquality(mlir::OpBuilder &builder,
                       ::mapping::ConstraintsSystemOp root,
                       ::mapping::SystemConstraintProjection projection,
                       llvm::ArrayRef<mlir::Attribute> subjects) {
  builder.setInsertionPointToEnd(&root.getBody().front());
  mlir::OperationState state(builder.getUnknownLoc(),
                             ::mapping::ConstraintEqualOp::getOperationName());
  state.addAttribute(
      "projection",
      ::mapping::SystemConstraintProjectionKeyAttr::get(
          builder.getContext(), static_cast<std::uint32_t>(projection)));
  state.addAttribute("subjects", builder.getArrayAttr(subjects));
  builder.create(state);
}

class TemporaryDirectory final {
public:
  TemporaryDirectory() {
    std::error_code error = llvm::sys::fs::createUniqueDirectory(
        "loom-system-candidate-state", path_);
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
  dataflow.graph private @sync(%start: none, %value: i32) -> i32
      attributes {input_segments = array<i32: 1, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %result:2 = dataflow.sync %start, %value
        : (none, i32) -> (none, i32)
    dataflow.graph.return values(%result#1 : i32) streams() memories()
        complete(%result#0 : none)
  }
  dataflow.thread private @worker domain(#dataflow.thread_domain<dense>)(
      %value: i32) ctrl (%ctrl: none) iv (%iv: index) {
    %first_result, %first_done = dataflow.graph.launch @sync deps(%ctrl)
        values(%value) stream_inputs() memories() stream_outputs()
        : (none, i32) -> (i32, none)
    %second_result, %second_done = dataflow.graph.launch @sync deps(%first_done)
        values(%first_result) stream_inputs() memories() stream_outputs()
        : (none, i32) -> (i32, none)
    dataflow.thread.yield %second_done : none
  }
  func.func private @host() {
    %value = arith.constant 7 : i32
    %extent = arith.constant 8 : index
    %first = dataflow.thread.launch @worker(%value) grid(%extent)
        : (i32) -> !dataflow.thread_token
    %second = dataflow.thread.launch @worker(%value) grid(%extent)
        : (i32) -> !dataflow.thread_token
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
buildMemoryDataflow(mlir::MLIRContext &context) {
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 64>>} {
  dataflow.graph private @load(
      %ctrl: none, %index: index, %memory: memref<4xi32>) -> i32
      attributes {input_segments = array<i32: 1, 0, 1>,
                  result_segments = array<i32: 1, 0, 0>} {
    %value, %loaded = dataflow.load %memory[%index] %ctrl : memref<4xi32>
    %done = dataflow.store %memory[%index] %value %loaded : memref<4xi32>
    dataflow.graph.return values(%value : i32) streams() memories()
        complete(%done : none)
  }
  dataflow.thread private @worker domain(#dataflow.thread_domain<dense>)(
      %index: index, %memory: memref<4xi32>) ctrl (%ctrl: none) {
    %value, %done = dataflow.graph.launch @load deps(%ctrl)
        values(%index) stream_inputs() memories(%memory) stream_outputs()
        : (none, index, memref<4xi32>) -> (i32, none)
    dataflow.thread.yield %done : none
  }
  func.func private @host(%index: index, %memory: memref<4xi32>) {
    %completion = dataflow.thread.launch @worker(%index, %memory)
        : (index, memref<4xi32>) -> !dataflow.thread_token
    return
  }
}
)mlir",
                                                        &context);
  if (!module)
    fail("cannot parse memory Dataflow fixture");
  return take(dataflow::finalizeCanonicalDataflow(*module));
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

loom::ResolvedConfig buildResolvedConfig() {
  loom::ResolvedConfig resolved = loom::defaultResolvedConfig();
  resolved.dse.objectiveCatalogs = spatialObjectiveCatalogs();
  resolved.dse.techMapping.candidatePublicationLimit = 1;
  resolved.dse.spatialPnr.temporaryViolations.admitted = {
      loom::ResolvedPnrViolationKind::UnroutedObligation,
      loom::ResolvedPnrViolationKind::CapacityOveruse};
  resolved.dse.spatialPnr.objectiveSelection = {0, 0, {}};
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
  resolved.dse.systemPnr.objectiveSelection = {0, 0, {}};
  return resolved;
}

loom::adg::FinalizedFabricDesign buildSpatialModule(loom::ArtifactStore &store,
                                                    bool addBoundaryBuffer) {
  loom::adg::DesignBuilder design(store);
  auto expansion = take(loom::adg::expandBuiltinSpatialCore(
      design, loom::adg::BuiltinTargetPreset::Small));
  if (addBoundaryBuffer) {
    const auto bits128 = take(loom::adg::PortType::bits(128));
    expansion.outputs.front() = take(expansion.spatialCore.addFifo(
                                         expansion.outputs.front(),
                                         loom::adg::FifoSpec{bits128, 2, true}))
                                    .value();
  }
  if (llvm::Error error = expansion.spatialCore.close(expansion.outputs))
    fail(llvm::toString(std::move(error)));
  auto finalized = take(std::move(design).finalize());
  require(finalized.roots().size() == 1,
          "SpatialCore fixture did not publish one Module root");
  return finalized;
}

loom::ArtifactRootReference
generateSpatialMapping(const dataflow::CanonicalDataflowProgramView &dataflow,
                       const loom::fabric::FinalizedFabricRoot &module,
                       const loom::ResolvedConfig &resolved,
                       loom::ArtifactStore &store,
                       mlir::MLIRContext *context = nullptr) {
  const auto techConfig =
      take(loom::mapping::projectResolvedTechMappingConfigView(resolved));
  const std::array<dataflow::GraphRef, 1> covers = {
      dataflow.graphs().front().ref};
  auto techOutcome = loom::mapping::generateTechMappings(
      {dataflow, covers, module.view(), techConfig, store});
  const auto *techCandidates =
      std::get_if<loom::mapping::GeneratedTechMappings>(&techOutcome);
  if (!techCandidates) {
    if (const auto *invalid =
            std::get_if<loom::mapping::InvalidTechMappingGeneration>(
                &techOutcome))
      fail("TechMapping fixture is invalid: " + invalid->diagnostic);
    if (const auto *internal =
            std::get_if<loom::mapping::InternalTechMappingGeneration>(
                &techOutcome))
      fail("TechMapping fixture failed internally: " + internal->diagnostic);
    if (std::holds_alternative<loom::mapping::ProvenInfeasibleTechMapping>(
            techOutcome))
      fail("TechMapping fixture is proven infeasible");
    fail("TechMapping fixture ended without a proof or candidate");
  }
  require(techCandidates->candidates.size() == 1,
          "TechMapping fixture did not produce one candidate");
  auto tech = take(loom::mapping::importTechMapping(
      techCandidates->candidates.front(), store));
  auto constraints = [&]() {
    if (!context)
      return take(loom::mapping::finalizeEmptySpatialMappingConstraintSet(
          dataflow, tech.view(), module.view(), store));
    require(dataflow.logicalMemoryRoots().size() == 1,
            "boundary-only Mapping fixture requires one logical memory root");
    const std::string text =
        "module {\n  mapping.constraints.spatial dataflow(" +
        identityAttr(dataflow.identity()) + ") tech_mapping(" +
        identityAttr(tech.view().identity()) + ") fabric(" +
        identityAttr(module.view().identity()) +
        ") {\n    mapping.constraint.domain_restriction "
        "projection(memory_bound_services) subject(" +
        dataflowAttr("logical_memory_root_ref", dataflow.identity(),
                     dataflow.logicalMemoryRoots().front().ref) +
        ") admissible_domain([])\n  }\n}\n";
    auto parsed = mlir::parseSourceString<mlir::ModuleOp>(text, context);
    if (!parsed)
      fail("cannot parse boundary-only Spatial MappingConstraintSet");
    auto roots = parsed->getOps<::mapping::ConstraintsSpatialOp>();
    return take(loom::mapping::finalizeSpatialMappingConstraintSet(
        *roots.begin(), dataflow, tech.view(), module.view(), store));
  }();
  const auto spatialConfig =
      take(loom::pnr::projectResolvedSpatialPnrConfigView(resolved));
  auto spatialOutcome = loom::pnr::generateSpatialMappings(
      {dataflow, tech.view(), module.view(), spatialConfig, constraints.view(),
       store});
  const auto *spatialCandidates =
      std::get_if<loom::pnr::GeneratedSpatialMappings>(&spatialOutcome);
  if (!spatialCandidates)
    std::visit(
        [&](const auto &outcome) {
          using Outcome = std::decay_t<decltype(outcome)>;
          if constexpr (!std::is_same_v<Outcome,
                                        loom::pnr::GeneratedSpatialMappings>)
            fail("SpatialMapping fixture did not produce one candidate: " +
                 outcome.diagnostic);
        },
        spatialOutcome);
  require(spatialCandidates->candidates.size() == 1,
          "SpatialMapping fixture did not produce one candidate");
  return spatialCandidates->candidates.front();
}

} // namespace
int main() {
  using loom::pnr::test::countOccurrences;
  using loom::pnr::test::rawSystemBytes;
  using loom::pnr::test::verifyFinalizedSystemMappingWorkflow;
  using loom::pnr::test::verifySystemResourceActionWorkflow;
  using loom::pnr::test::verifySystemServiceTargetRejections;
  using loom::pnr::test::withFirstCoordinateLowerBound;
  TemporaryDirectory directory;
  loom::ArtifactStore store(directory.path());
  mlir::MLIRContext context = makeContext();

  auto dataflowArtifact = buildDataflow(context);
  take(dataflow::publishCanonicalDataflow(dataflowArtifact, store));
  auto dataflow = take(dataflowArtifact.view());
  auto baselineDesign = take(loom::adg::buildBuiltinTarget(
      store, loom::adg::BuiltinTargetPreset::Small));
  require(baselineDesign.roots().size() == 1 &&
              baselineDesign.roots().front().directDependencies().size() == 1,
          "builtin System fixture did not publish one Module dependency");
  auto primaryModule = take(loom::fabric::importEntireFabricRoot(
      baselineDesign.roots().front().directDependencies().front().root, store));
  auto alternateDesign = buildSpatialModule(store, true);
  auto design = loom::pnr::test::buildHeterogeneousSystem(
      store, baselineDesign.roots().front(), primaryModule,
      alternateDesign.roots().front(), context);
  const auto &systemRoot = design.roots().front();
  auto system = take(loom::fabric::requireSystemRoot(systemRoot.view()));
  require(systemRoot.directDependencies().size() == 2,
          "heterogeneous System did not retain both SpatialCores");

  const loom::ResolvedConfig resolved = buildResolvedConfig();
  const auto config =
      take(loom::pnr::projectResolvedSystemPnrConfigView(resolved));

  auto memoryDataflowArtifact = buildMemoryDataflow(context);
  take(dataflow::publishCanonicalDataflow(memoryDataflowArtifact, store));
  auto memoryDataflow = take(memoryDataflowArtifact.view());
  auto endpointDesign = loom::pnr::test::buildHeterogeneousSystem(
      store, baselineDesign.roots().front(), primaryModule, primaryModule,
      context, /*extraSupportsRead=*/false,
      /*routeExtraMemoryThroughTransform=*/true);
  auto endpointSystem = take(
      loom::fabric::requireSystemRoot(endpointDesign.roots().front().view()));
  loom::ResolvedConfig memoryResolved = resolved;
  memoryResolved.dse.spatialPnr.search =
      loom::defaultResolvedConfig().dse.spatialPnr.search;
  const auto memoryMapping = generateSpatialMapping(
      memoryDataflow, primaryModule, memoryResolved, store, &context);
  verifySystemResourceActionWorkflow(store, baselineDesign.roots().front(),
                                     primaryModule, memoryDataflow,
                                     memoryMapping, resolved, config, context);
  std::vector<dataflow::RootThreadLaunchRef> memoryRoots{
      memoryDataflow.rootThreadLaunches().front().ref};
  auto memoryConstraints =
      take(loom::mapping::finalizeEmptySystemMappingConstraintSet(
          memoryDataflow, endpointSystem, memoryRoots, store));
  auto memoryPartition =
      take(loom::pnr::projectWholeDomainPresburgerPartitionPlan(
          memoryDataflow, memoryConstraints.view().rootThreadLaunches()));
  auto memorySpatial =
      take(loom::mapping::importSpatialMapping(memoryMapping, store));
  loom::ArtifactRootReference memoryTechReference{
      loom::mapping::mappingArtifactSchema.identity.str(),
      loom::mapping::mappingArtifactSchema.version,
      memorySpatial.view().techMappingIdentity()};
  auto memoryTech =
      take(loom::mapping::importTechMapping(memoryTechReference, store));
  auto memorySpatialConstraints =
      take(loom::mapping::finalizeEmptySpatialMappingConstraintSet(
          memoryDataflow, memoryTech.view(), primaryModule.view(), store));
  auto memorySpatialConfig =
      take(loom::pnr::projectResolvedSpatialPnrConfigView(memoryResolved));
  requireUnsupported(
      loom::pnr::projectSystemPnrSearchDomain(
          memoryDataflow, endpointSystem, config, memoryConstraints,
          memoryPartition,
          loom::pnr::SystemFlatGraphSearchInput{
              {{memoryTechReference, memorySpatialConfig,
                memorySpatialConstraints.reference()}},
              {}},
          store),
      loom::pnr::UnsupportedSystemPnrSearchDomainReason::
          FlatOperationServiceDomainProjectionUnavailable,
      "flat operation-service compatibility projection is not implemented");
  auto memorySearchDomain = take(loom::pnr::projectSystemPnrSearchDomain(
      memoryDataflow, endpointSystem, config, memoryConstraints,
      memoryPartition,
      loom::pnr::SystemHierarchicalGraphSearchInput{{memoryMapping}}, store));
  const auto memoryService = llvm::find_if(
      memorySearchDomain.serviceObligations(), [](const auto &service) {
        const auto *operation =
            std::get_if<loom::mapping::OperationServiceObligationFamilyKey>(
                &service.key);
        return operation &&
               std::holds_alternative<dataflow::LogicalMemoryRootOrViewRef>(
                   *operation);
      });
  require(memoryService != memorySearchDomain.serviceObligations().end(),
          "endpoint-factorization fixture has no memory obligation");
  std::vector<const loom::pnr::SystemSearchServiceTargetCompatibility *>
      addressedRows;
  for (const auto &row : memoryService->targetCompatibility) {
    const auto *subject =
        std::get_if<loom::pnr::SystemServiceMemberTargetSubject>(&row.subject);
    if (subject &&
        std::holds_alternative<dataflow::AddressedMemoryActorMemberRef>(
            subject->member))
      addressedRows.push_back(&row);
  }
  require(addressedRows.size() == 4,
          "two memory subjects did not produce endpoint-factorized rows");
  const loom::fabric::SystemServiceEndpointRef *supportedEndpoint = nullptr;
  const loom::fabric::SystemServiceEndpointRef *unsupportedEndpoint = nullptr;
  const loom::fabric::SystemServiceEndpointRef *transformedEndpoint = nullptr;
  std::vector<loom::fabric::SystemServiceEndpointRef> targetEndpoints;
  for (const auto *row : addressedRows)
    if (!llvm::is_contained(targetEndpoints, row->boundEndpoint))
      targetEndpoints.push_back(row->boundEndpoint);
  require(targetEndpoints.size() == 2,
          "same Module path did not produce two exact endpoint keys");
  for (const auto &endpoint : targetEndpoints) {
    const auto targetPlans =
        take(loom::fabric::projectFabricMemoryServiceTargetPlans(endpointSystem,
                                                                 endpoint));
    if (llvm::any_of(targetPlans, [](const auto &plan) {
          return llvm::any_of(plan.branches, [](const auto &branch) {
            return !branch.transformPath.empty();
          });
        })) {
      require(!transformedEndpoint,
              "more than one endpoint unexpectedly uses a transform chain");
      transformedEndpoint = &endpoint;
    }
    std::size_t emptyRows = 0;
    std::size_t nonemptyRows = 0;
    for (const auto *row : addressedRows) {
      if (row->boundEndpoint != endpoint)
        continue;
      const auto *regions =
          std::get_if<std::vector<loom::fabric::FabricMemoryServiceRegionRef>>(
              &row->compatibleTargets);
      require(regions, "memory target row has a non-region domain");
      regions->empty() ? ++emptyRows : ++nonemptyRows;
    }
    require(emptyRows + nonemptyRows == 2,
            "one endpoint did not retain both memory subjects");
    if (emptyRows == 0)
      supportedEndpoint = &endpoint;
    else {
      require(emptyRows == 1 && nonemptyRows == 1,
              "adverse endpoint does not distinguish read from write");
      unsupportedEndpoint = &endpoint;
    }
  }
  require(supportedEndpoint && unsupportedEndpoint,
          "endpoint rows unioned or intersected distinct read capabilities");
  require(transformedEndpoint == unsupportedEndpoint,
          "adverse endpoint did not exercise the explicit transform closure");
  const auto transformedTargetPlans =
      take(loom::fabric::projectFabricMemoryServiceTargetPlans(
          endpointSystem, *transformedEndpoint));
  std::vector<loom::fabric::SystemServiceTransformRef> foreignTransformPath;
  std::optional<loom::fabric::FabricMemoryServiceRegionRef> otherEndpointRegion;
  for (const auto &plan : transformedTargetPlans)
    for (const auto &branch : plan.branches)
      if (!branch.transformPath.empty()) {
        foreignTransformPath = branch.transformPath;
        otherEndpointRegion = branch.region;
        break;
      }
  require(!foreignTransformPath.empty() && otherEndpointRegion,
          "adverse endpoint has no concrete transform path");

  const loom::fabric::FabricMemoryEndpointRef *unsupportedOccurrence = nullptr;
  for (const auto &attachment : endpointSystem.spatialAttachments()) {
    if (attachment.serviceEndpoint != *unsupportedEndpoint)
      continue;
    require(!unsupportedOccurrence && attachment.spatialEndpoint.memory(),
            "unsupported endpoint has an ambiguous memory attachment");
    unsupportedOccurrence = attachment.spatialEndpoint.memory();
  }
  require(unsupportedOccurrence,
          "unsupported endpoint has no exact occurrence attachment");
  const loom::fabric::FabricMemoryEndpointRef unsupportedSystemEndpoint{
      loom::fabric::FabricMemoryEndpointOwnerRef::of(*unsupportedEndpoint), 0};
  std::size_t unsupportedEmptyTerminalRows = 0;
  std::size_t unsupportedNonemptyTerminalRows = 0;
  for (const auto &row : memoryService->transferTerminalCompatibility) {
    const auto &bound =
        std::get<loom::pnr::SystemMemoryOrFenceTerminalEndpoint>(
            row.boundEndpoint)
            .endpoint;
    if (bound != unsupportedSystemEndpoint && bound != *unsupportedOccurrence)
      continue;
    row.compatibleTransportEndpoints.empty()
        ? ++unsupportedEmptyTerminalRows
        : ++unsupportedNonemptyTerminalRows;
  }
  require(unsupportedEmptyTerminalRows != 0 &&
              unsupportedNonemptyTerminalRows != 0,
          "memory terminal rows lost per-member endpoint compatibility");

  std::vector<loom::fabric::AccCoreOccurrenceRef> supportedCores;
  for (const auto &attachment : endpointSystem.spatialAttachments()) {
    if (attachment.serviceEndpoint != *supportedEndpoint)
      continue;
    const auto *occurrence = attachment.spatialEndpoint.memory();
    require(occurrence, "supported endpoint has an incomplete attachment");
    const auto *spatialCore =
        std::get_if<loom::fabric::SpatialCoreOccurrenceRef>(
            &occurrence->owner.payload);
    require(spatialCore,
            "supported memory attachment is not occurrence-qualified");
    if (!llvm::is_contained(supportedCores, spatialCore->core))
      supportedCores.push_back(spatialCore->core);
  }
  require(!supportedCores.empty(),
          "supported endpoint has no exact occurrence attachment");
  llvm::sort(supportedCores, [](const auto left, const auto right) {
    return loom::fabric::canonicalFabricBytes(left) <
           loom::fabric::canonicalFabricBytes(right);
  });

  auto bindingProblem = take(loom::pnr::freezeSystemPnrProblem(
      memoryDataflow, endpointSystem, memorySearchDomain, config,
      memoryConstraints, store));
  std::vector<std::uint32_t> operationLegWidths;
  for (const auto &leg : bindingProblem->serviceLegs())
    if (std::holds_alternative<
            loom::mapping::OperationServiceObligationFamilyKey>(
            leg.key.obligation))
      operationLegWidths.push_back(leg.requiredPayloadWidthBits);
  llvm::sort(operationLegWidths);
  require(operationLegWidths == std::vector<std::uint32_t>({0, 32, 64, 64}),
          "operation service legs lost their maximum-width envelopes");
  std::vector<loom::pnr::PnrIndex> memoryContextOrdinals;
  for (const auto &[ordinal, serviceContext] :
       llvm::enumerate(bindingProblem->serviceContexts()))
    if (serviceContext.service < bindingProblem->serviceDomains().size() &&
        std::holds_alternative<
            loom::mapping::OperationServiceObligationFamilyKey>(
            bindingProblem->serviceDomains()[serviceContext.service].key))
      memoryContextOrdinals.push_back(
          static_cast<loom::pnr::PnrIndex>(ordinal));
  require(memoryContextOrdinals.size() == 1,
          "one graph-backed memory obligation did not form one context");
  const auto &memoryContext =
      bindingProblem->serviceContexts()[memoryContextOrdinals.front()];
  std::vector<loom::pnr::PnrIndex> memoryThreadChoices(
      bindingProblem->threadDecisions().size(), 0);
  std::vector<loom::pnr::PnrIndex> memoryGraphChoices(
      bindingProblem->graphDecisions().size(), 0);
  const auto memoryThreadDomain =
      bindingProblem->threadChoiceCatalogOrdinals(memoryContext.threadDecision);
  loom::pnr::PnrIndex supportedChoice = loom::pnr::getInvalidPnrIndex();
  loom::pnr::PnrIndex unsupportedChoice = loom::pnr::getInvalidPnrIndex();
  for (loom::pnr::PnrIndex choice = 0; choice != memoryThreadDomain.size();
       ++choice) {
    const auto core = bindingProblem->accCores()[memoryThreadDomain[choice]];
    if (llvm::is_contained(supportedCores, core))
      supportedChoice = choice;
    else
      unsupportedChoice = choice;
  }
  require(supportedChoice != loom::pnr::getInvalidPnrIndex() &&
              unsupportedChoice != loom::pnr::getInvalidPnrIndex(),
          "memory context did not retain both occurrence choices");
  memoryThreadChoices[memoryContext.threadDecision] = supportedChoice;
  auto supportedCandidate = take(loom::pnr::initializeSystemCandidate(
      bindingProblem, memoryThreadChoices, memoryGraphChoices));
  auto selectedTargetDomain = take(
      supportedCandidate->serviceTargetDomain(memoryContextOrdinals.front()));
  const auto *selectedPlans =
      std::get_if<std::vector<loom::pnr::SystemMemoryServiceTargetPlan>>(
          &selectedTargetDomain);
  require(selectedPlans && !selectedPlans->empty(),
          "matching target rows did not retain their nonempty intersection");
  const auto *selectedPlan =
      std::get_if<loom::pnr::SystemMemoryServiceTargetPlan>(
          &supportedCandidate->serviceTarget(memoryContextOrdinals.front()));
  require(selectedPlan && *selectedPlan == selectedPlans->front() &&
              selectedPlan->branches.size() == 1,
          "canonical candidate did not select the first exact target");
  const auto selectedRegion = selectedPlan->branches.front().region;

  std::vector<loom::pnr::SystemServiceTargetSelection> foreignTargets(
      supportedCandidate->serviceTargets().begin(),
      supportedCandidate->serviceTargets().end());
  auto foreignPlan = *selectedPlan;
  auto &foreignRegion = foreignPlan.branches.front().region;
  foreignRegion.ordinal += 1000;
  foreignTargets[memoryContextOrdinals.front()] = foreignPlan;
  requireFailureContains(
      loom::pnr::SystemCandidateState::create(
          bindingProblem,
          {supportedCandidate->threadChoices(),
           supportedCandidate->graphChoices(),
           supportedCandidate->serviceRoutes(),
           supportedCandidate->serviceRouteNodes(),
           supportedCandidate->serviceRouteSinks(), foreignTargets,
           supportedCandidate->instructionResourceUses(),
           supportedCandidate->serviceResourceUses()}),
      "selected service target is outside its exact H domain");

  auto memoryDraft = take(
      loom::pnr::materializeSystemCandidateDraft(*supportedCandidate, context));
  auto memoryRoot = mlir::cast<::mapping::SystemOp>(memoryDraft.get());
  const auto selectedRegionBytes =
      loom::fabric::canonicalFabricBytes(selectedRegion);
  std::size_t memoryTargetCount = 0;
  ::mapping::ServiceRealizationOp selectedMemoryService;
  ::mapping::ServicePlanOp selectedMemoryPlan;
  for (auto service :
       memoryRoot.getBody().front().getOps<::mapping::ServiceRealizationOp>())
    for (auto plan :
         service.getBody().front().getOps<::mapping::ServicePlanOp>())
      for (auto target :
           plan.getBody().front().getOps<::mapping::MemoryRegionTargetOp>()) {
        selectedMemoryService = service;
        selectedMemoryPlan = plan;
        ++memoryTargetCount;
        require(unsignedBytes(target.getServiceRegion().getRecord()) ==
                    std::vector<std::uint8_t>(selectedRegionBytes.begin(),
                                              selectedRegionBytes.end()),
                "materialized memory target changed its selected region");
        require(target.getTransformPath().empty(),
                "direct service target gained a transform path");
      }
  require(memoryTargetCount == 1,
          "one memory service context did not materialize one target");

  verifySystemServiceTargetRejections(
      memoryRoot, memoryDataflow, endpointSystem, store, context,
      foreignTransformPath, *otherEndpointRegion);

  require(!supportedCandidate->instructionResourceUses().empty(),
          "candidate omitted InstructionCore occupancy choices");
  const std::size_t expectedServiceUseCount =
      llvm::count_if(memoryContext.subjects,
                     [](const auto &subject) {
                       return std::holds_alternative<
                           loom::pnr::SystemServiceMemberTargetSubject>(
                           subject);
                     }) *
      selectedPlan->branches.size();
  require(supportedCandidate->serviceResourceUses().size() ==
              expectedServiceUseCount,
          "addressed members and target branches did not select exact uses");
  std::vector<loom::pnr::SystemServiceResourceUseSelection> foreignUses(
      supportedCandidate->serviceResourceUses().begin(),
      supportedCandidate->serviceResourceUses().end());
  foreignUses.front().pattern.ordinal += 1000;
  requireFailureContains(
      loom::pnr::SystemCandidateState::create(
          bindingProblem,
          {supportedCandidate->threadChoices(),
           supportedCandidate->graphChoices(),
           supportedCandidate->serviceRoutes(),
           supportedCandidate->serviceRouteNodes(),
           supportedCandidate->serviceRouteSinks(),
           supportedCandidate->serviceTargets(),
           supportedCandidate->instructionResourceUses(), foreignUses}),
      "service ResourceUse is foreign or inadmissible");
  requireFailureContains(
      loom::pnr::SystemCandidateState::create(
          bindingProblem, {supportedCandidate->threadChoices(),
                           supportedCandidate->graphChoices(),
                           supportedCandidate->serviceRoutes(),
                           supportedCandidate->serviceRouteNodes(),
                           supportedCandidate->serviceRouteSinks(),
                           supportedCandidate->serviceTargets(),
                           {},
                           supportedCandidate->serviceResourceUses()}),
      "InstructionCore ResourceUse count is incomplete");
  std::size_t instructionUseCount = 0;
  std::size_t serviceUseCount = 0;
  for (auto use :
       memoryRoot.getBody().front().getOps<::mapping::ResourceUseOp>()) {
    auto activation = mlir::dyn_cast<::mapping::SystemRelativeActivationAttr>(
        use.getActivation());
    require(static_cast<bool>(activation),
            "System ResourceUse lost its typed activation");
    if (mlir::isa<::mapping::InstructionExecutionResourceOwnerRefAttr>(
            use.getOwner())) {
      require(activation.getRelease().size() == 1,
              "InstructionCore occupancy lost root completion release");
      ++instructionUseCount;
      continue;
    }
    auto owner =
        mlir::dyn_cast<::mapping::ServicePlanElementRefAttr>(use.getOwner());
    require(owner && mlir::isa<::mapping::MemoryRegionElementKeyAttr>(
                         owner.getElement()),
            "addressed service use lost its exact MemoryRegion owner");
    require(activation.getRelease().empty(),
            "addressed service use gained a causal release");
    auto event =
        take(dataflow::decodeDataflowReference<dataflow::EventFamilyKey>(
            unsignedBytes(activation.getTrigger().getEvent().getRecord()),
            memoryDataflow.identity()));
    require(std::holds_alternative<dataflow::ContextualActorTransitionEventRef>(
                event) &&
                std::get<dataflow::ContextualActorTransitionEventRef>(event)
                        .transitionCaseOrdinal == 0,
            "addressed service use did not trigger on its issue transition");
    ++serviceUseCount;
  }
  require(instructionUseCount ==
                  supportedCandidate->instructionResourceUses().size() &&
              serviceUseCount ==
                  supportedCandidate->serviceResourceUses().size(),
          "materializer did not preserve the candidate ResourceUse closure");

  const auto canonicalMemoryDraft =
      take(loom::mapping::writeCanonicalSystemMappingAssembly(memoryRoot));
  const llvm::StringRef canonicalMemoryText(
      reinterpret_cast<const char *>(canonicalMemoryDraft.bytes().data()),
      canonicalMemoryDraft.bytes().size());
  const std::size_t baselinePlanCount =
      countOccurrences(canonicalMemoryText, "mapping.service_plan ");
  mlir::OwningOpRef<mlir::Operation *> alternateTargetDraft(
      memoryDraft->clone());
  auto alternateTargetRoot =
      mlir::cast<::mapping::SystemOp>(alternateTargetDraft.get());
  ::mapping::ServiceRealizationOp alternateTargetService;
  ::mapping::ServicePlanOp alternateTargetPlan;
  for (auto service : alternateTargetRoot.getBody()
                          .front()
                          .getOps<::mapping::ServiceRealizationOp>())
    for (auto plan :
         service.getBody().front().getOps<::mapping::ServicePlanOp>())
      if (!plan.getBody()
               .front()
               .getOps<::mapping::MemoryRegionTargetOp>()
               .empty()) {
        alternateTargetService = service;
        alternateTargetPlan = plan;
      }
  require(selectedMemoryService && selectedMemoryPlan &&
              alternateTargetService && alternateTargetPlan,
          "memory target plan lookup failed");
  auto distinctPlan =
      mlir::cast<::mapping::ServicePlanOp>(alternateTargetPlan->clone());
  distinctPlan.setPlanOrdinalAttr(
      mlir::Builder(&context).getI64IntegerAttr(1000));
  auto distinctTarget = *distinctPlan.getBody()
                             .front()
                             .getOps<::mapping::MemoryRegionTargetOp>()
                             .begin();
  distinctTarget.setServiceRegionAttr(
      constraintFabricAttr<::mapping::FabricMemoryServiceRegionRefAttr>(
          &context, foreignRegion));
  alternateTargetService.getBody().front().push_back(distinctPlan);
  auto alternateSelection = *alternateTargetService.getBody()
                                 .front()
                                 .getOps<::mapping::ServicePlanSelectionOp>()
                                 .begin();
  mlir::OpBuilder alternateBuilder(&context);
  alternateBuilder.setInsertionPointToEnd(
      &alternateSelection.getBody().front());
  ::mapping::ServicePlanPresburgerClauseOp::create(
      alternateBuilder, alternateBuilder.getUnknownLoc(),
      alternateBuilder.getArrayAttr({::mapping::SystemPresburgerCellAttr::get(
          &context, 0, 0, 0, alternateBuilder.getArrayAttr({}),
          alternateBuilder.getArrayAttr({}))}),
      1000);
  const auto distinctTargetBytes = take(
      loom::mapping::writeCanonicalSystemMappingAssembly(alternateTargetRoot));
  const llvm::StringRef distinctTargetText(
      reinterpret_cast<const char *>(distinctTargetBytes.bytes().data()),
      distinctTargetBytes.bytes().size());
  require(countOccurrences(distinctTargetText, "mapping.service_plan ") ==
              baselinePlanCount + 1,
          "canonicalization merged plans with different service targets");

  memoryThreadChoices[memoryContext.threadDecision] = unsupportedChoice;
  requireFailureContains(
      loom::pnr::initializeSystemCandidate(bindingProblem, memoryThreadChoices,
                                           memoryGraphChoices),
      "matching service target rows have an empty intersection");
  requireFailureContains(
      loom::pnr::SystemCandidateState::create(
          bindingProblem, {memoryThreadChoices, memoryGraphChoices,
                           supportedCandidate->serviceRoutes(),
                           supportedCandidate->serviceRouteNodes(),
                           supportedCandidate->serviceRouteSinks(),
                           supportedCandidate->serviceTargets(),
                           supportedCandidate->instructionResourceUses(),
                           supportedCandidate->serviceResourceUses()}),
      "matching service target rows have an empty intersection");

  const auto belongsToSupportedExecution = [&](const auto &row) {
    const auto &endpoint =
        std::get<loom::pnr::SystemMemoryOrFenceTerminalEndpoint>(
            row.boundEndpoint)
            .endpoint;
    if (const auto *system =
            std::get_if<loom::fabric::SystemServiceEndpointRef>(
                &endpoint.owner.payload))
      return *system == *supportedEndpoint;
    const auto *spatialCore =
        std::get_if<loom::fabric::SpatialCoreOccurrenceRef>(
            &endpoint.owner.payload);
    return spatialCore && llvm::is_contained(supportedCores, spatialCore->core);
  };
  const auto restrictedTerminalRow = llvm::find_if(
      memoryService->transferTerminalCompatibility, [&](const auto &row) {
        return belongsToSupportedExecution(row) &&
               !row.compatibleTransportEndpoints.empty();
      });
  require(restrictedTerminalRow !=
              memoryService->transferTerminalCompatibility.end(),
          "constraint fixture has no supported transfer terminal row");
  const auto restrictedTerminal = restrictedTerminalRow->terminal;
  const loom::mapping::SystemTransferTerminalKey peerTerminal = [&] {
    if (const auto *source =
            std::get_if<loom::mapping::SystemTransferSourceTerminalKey>(
                &restrictedTerminal))
      return loom::mapping::SystemTransferTerminalKey(
          loom::mapping::SystemTransferSinkTerminalKey{source->leg, 0});
    return loom::mapping::SystemTransferTerminalKey(
        loom::mapping::SystemTransferSourceTerminalKey{
            std::get<loom::mapping::SystemTransferSinkTerminalKey>(
                restrictedTerminal)
                .leg});
  }();
  const auto peerTerminalRow = llvm::find_if(
      memoryService->transferTerminalCompatibility, [&](const auto &row) {
        return row.terminal == peerTerminal &&
               belongsToSupportedExecution(row) &&
               !row.compatibleTransportEndpoints.empty();
      });
  require(peerTerminalRow != memoryService->transferTerminalCompatibility.end(),
          "constraint fixture has no supported peer terminal");
  std::size_t expectedConstrainedTerminalRows = 0;
  bool expectedUnrestrictedTerminal = false;
  for (const auto &row : memoryService->transferTerminalCompatibility) {
    if (!belongsToSupportedExecution(row))
      continue;
    if (row.terminal == restrictedTerminal || row.terminal == peerTerminal)
      ++expectedConstrainedTerminalRows;
    else if (!row.compatibleTransportEndpoints.empty())
      expectedUnrestrictedTerminal = true;
  }
  require(expectedConstrainedTerminalRows >= 2 && expectedUnrestrictedTerminal,
          "constraint fixture lacks retained terminal coverage");
  const auto memoryOperation =
      std::get<loom::mapping::OperationServiceObligationFamilyKey>(
          memoryService->key);

  auto constrainedModule = buildSystemConstraintModule(
      context, memoryDataflow.identity(), endpointSystem.artifact().identity(),
      memoryRoots);
  auto constrainedRoot = llvm::cast<::mapping::ConstraintsSystemOp>(
      constrainedModule->getBody()->front());
  mlir::OpBuilder constraintBuilder(&context);
  std::vector<mlir::Attribute> supportedCoreAttributes;
  supportedCoreAttributes.reserve(supportedCores.size());
  for (const auto core : supportedCores)
    supportedCoreAttributes.push_back(
        constraintFabricAttr<::mapping::FabricAccCoreOccurrenceRefAttr>(
            &context, core));
  addSystemRestriction(
      constraintBuilder, constrainedRoot,
      ::mapping::SystemConstraintProjection::ThreadTargetAccCore,
      constraintDataflowAttr<::mapping::RootThreadLaunchRefAttr>(
          &context, memoryDataflow.identity(), memoryRoots.front()),
      supportedCoreAttributes);
  const auto memorySubject = serviceObligationAttr(
      &context, memoryDataflow.identity(),
      loom::mapping::SystemServiceObligationKey{memoryOperation});
  addSystemRestriction(
      constraintBuilder, constrainedRoot,
      ::mapping::SystemConstraintProjection::ServiceTargetRegion, memorySubject,
      {});
  const auto restrictedTerminalSubject = transferTerminalAttr(
      &context, memoryDataflow.identity(), restrictedTerminal);
  const auto peerTerminalSubject =
      transferTerminalAttr(&context, memoryDataflow.identity(), peerTerminal);
  addSystemEquality(
      constraintBuilder, constrainedRoot,
      ::mapping::SystemConstraintProjection::TransferTerminalAttachment,
      {restrictedTerminalSubject, peerTerminalSubject});
  addSystemRestriction(
      constraintBuilder, constrainedRoot,
      ::mapping::SystemConstraintProjection::TransferTerminalAttachment,
      peerTerminalSubject, {});
  auto constrainedSystemConstraints =
      take(loom::mapping::finalizeSystemMappingConstraintSet(
          constrainedRoot, memoryDataflow, endpointSystem, store));
  auto constrainedSearchDomain = take(loom::pnr::projectSystemPnrSearchDomain(
      memoryDataflow, endpointSystem, config, constrainedSystemConstraints,
      memoryPartition,
      loom::pnr::SystemHierarchicalGraphSearchInput{{memoryMapping}}, store));
  require(constrainedSearchDomain.digest() != memorySearchDomain.digest(),
          "exact System K did not change the H digest");
  for (const auto &binding : constrainedSearchDomain.bindings()) {
    if (!std::holds_alternative<dataflow::RootThreadLaunchRef>(binding.key))
      continue;
    const auto *thread = std::get_if<loom::pnr::SystemThreadBindingDomain>(
        &binding.atoms.front().domain);
    require(thread && thread->compatibleAccCores == supportedCores,
            "thread constraint did not restrict the H atom domain");
  }
  const auto constrainedMemoryService = llvm::find_if(
      constrainedSearchDomain.serviceObligations(),
      [&](const auto &service) { return service.key == memoryService->key; });
  require(constrainedMemoryService !=
              constrainedSearchDomain.serviceObligations().end(),
          "constrained H lost the memory obligation");
  std::size_t constrainedAddressedRows = 0;
  for (const auto &row : constrainedMemoryService->targetCompatibility) {
    const auto *subject =
        std::get_if<loom::pnr::SystemServiceMemberTargetSubject>(&row.subject);
    if (!subject ||
        !std::holds_alternative<dataflow::AddressedMemoryActorMemberRef>(
            subject->member))
      continue;
    ++constrainedAddressedRows;
    const auto *regions =
        std::get_if<std::vector<loom::fabric::FabricMemoryServiceRegionRef>>(
            &row.compatibleTargets);
    require(regions && regions->empty(),
            "service target restriction was not folded into its H row");
  }
  require(constrainedAddressedRows == addressedRows.size() / 2,
          "thread constraint did not restrict service row key coverage");
  std::size_t constrainedTerminalRows = 0;
  bool retainedUnrestrictedTerminal = false;
  for (const auto &row :
       constrainedMemoryService->transferTerminalCompatibility) {
    if (row.terminal == restrictedTerminal || row.terminal == peerTerminal) {
      ++constrainedTerminalRows;
      require(row.compatibleTransportEndpoints.empty(),
              "terminal restriction was not folded into its H row");
    } else if (!row.compatibleTransportEndpoints.empty()) {
      retainedUnrestrictedTerminal = true;
    }
  }
  require(constrainedTerminalRows == expectedConstrainedTerminalRows &&
              retainedUnrestrictedTerminal,
          "terminal restriction removed a row or leaked across subjects");
  auto adoptedConstrained = take(loom::pnr::adoptSystemPnrSearchDomain(
      loom::pnr::systemPnrSearchDomainSchemaDescriptorBytes(),
      constrainedSearchDomain.canonicalViewBytes(),
      constrainedSearchDomain.digest(), store));
  require(adoptedConstrained.canonicalViewBytes() ==
              constrainedSearchDomain.canonicalViewBytes(),
          "strict H adoption changed constraint-folded rows");

  std::vector<loom::ArtifactRootReference> spatialMappings;
  std::vector<loom::pnr::FlatSpatialReopenProblem> flatProblems;
  const auto spatialConfig =
      take(loom::pnr::projectResolvedSpatialPnrConfigView(resolved));
  for (const auto &dependency : systemRoot.directDependencies()) {
    auto module =
        take(loom::fabric::importEntireFabricRoot(dependency.root, store));
    auto spatialReference =
        generateSpatialMapping(dataflow, module, resolved, store);
    spatialMappings.push_back(spatialReference);
    auto spatial =
        take(loom::mapping::importSpatialMapping(spatialReference, store));
    loom::ArtifactRootReference techReference{
        loom::mapping::mappingArtifactSchema.identity.str(),
        loom::mapping::mappingArtifactSchema.version,
        spatial.view().techMappingIdentity()};
    auto tech = take(loom::mapping::importTechMapping(techReference, store));
    auto spatialConstraints =
        take(loom::mapping::finalizeEmptySpatialMappingConstraintSet(
            dataflow, tech.view(), module.view(), store));
    flatProblems.push_back({std::move(techReference), spatialConfig,
                            spatialConstraints.reference()});
  }
  std::vector<dataflow::RootThreadLaunchRef> roots;
  for (const dataflow::CanonicalRootThreadLaunchView &root :
       dataflow.rootThreadLaunches())
    roots.push_back(root.ref);
  auto constraints =
      take(loom::mapping::finalizeEmptySystemMappingConstraintSet(
          dataflow, system, roots, store));
  auto partition = take(loom::pnr::projectWholeDomainPresburgerPartitionPlan(
      dataflow, constraints.view().rootThreadLaunches()));
  auto reversedProblems = flatProblems;
  auto reversedSeeds = spatialMappings;
  std::reverse(reversedProblems.begin(), reversedProblems.end());
  std::reverse(reversedSeeds.begin(), reversedSeeds.end());
  auto flatDomain = take(loom::pnr::projectSystemPnrSearchDomain(
      dataflow, system, config, constraints, partition,
      loom::pnr::SystemFlatGraphSearchInput{reversedProblems, reversedSeeds},
      store));
  auto canonicalFlatDomain = take(loom::pnr::projectSystemPnrSearchDomain(
      dataflow, system, config, constraints, partition,
      loom::pnr::SystemFlatGraphSearchInput{flatProblems, spatialMappings},
      store));
  require(flatDomain.canonicalViewBytes() ==
              canonicalFlatDomain.canonicalViewBytes(),
          "flat problem or seed authoring order changed canonical H");
  auto adoptedFlat = take(loom::pnr::adoptSystemPnrSearchDomain(
      loom::pnr::systemPnrSearchDomainSchemaDescriptorBytes(),
      flatDomain.canonicalViewBytes(), flatDomain.digest(), store));
  require(adoptedFlat.canonicalViewBytes() == flatDomain.canonicalViewBytes(),
          "strict H adoption changed a valid flat graph domain");
  for (const auto &binding : flatDomain.bindings()) {
    if (!std::holds_alternative<dataflow::RootedGraphLaunchRef>(binding.key))
      continue;
    for (const auto &atom : binding.atoms) {
      const auto *flat =
          std::get_if<loom::pnr::SystemFlatGraphBindingDomain>(&atom.domain);
      require(flat && flat->exactSpatialReopenProblems.size() == 2 &&
                  flat->compatibleImmutableSeeds.size() == 2,
              "flat graph atom lost its exact covered problems or seeds");
    }
  }
  requireFailureContains(
      loom::pnr::projectSystemPnrSearchDomain(
          dataflow, system, config, constraints, partition,
          loom::pnr::SystemFlatGraphSearchInput{}, store),
      "flat graph search requires at least one reopen problem");
  auto wrongConfigProblems = flatProblems;
  wrongConfigProblems.front().spatialConfig = config;
  requireFailureContains(
      loom::pnr::projectSystemPnrSearchDomain(
          dataflow, system, config, constraints, partition,
          loom::pnr::SystemFlatGraphSearchInput{wrongConfigProblems, {}},
          store),
      "non-Spatial resolved config");
  auto mismatchedConstraintProblems = flatProblems;
  mismatchedConstraintProblems.front().spatialConstraintReference =
      flatProblems.back().spatialConstraintReference;
  requireFailureContains(loom::pnr::projectSystemPnrSearchDomain(
                             dataflow, system, config, constraints, partition,
                             loom::pnr::SystemFlatGraphSearchInput{
                                 mismatchedConstraintProblems, {}},
                             store),
                         "Spatial MappingConstraintSet has foreign T/F owners");
  requireFailureContains(
      loom::pnr::projectSystemPnrSearchDomain(
          dataflow, system, config, constraints, partition,
          loom::pnr::SystemFlatGraphSearchInput{{flatProblems.front()},
                                                {spatialMappings.back()}},
          store),
      "flat seed does not match a listed reopen problem");
  auto singleFlatDomain = take(loom::pnr::projectSystemPnrSearchDomain(
      dataflow, system, config, constraints, partition,
      loom::pnr::SystemFlatGraphSearchInput{{flatProblems.front()},
                                            {spatialMappings.front()}},
      store));
  auto mismatchedFlatBytes =
      replaceEvery(singleFlatDomain.canonicalViewBytes(),
                   loom::encodeArtifactRootReference(
                       flatProblems.front().spatialConstraintReference),
                   loom::encodeArtifactRootReference(
                       flatProblems.back().spatialConstraintReference));
  auto mismatchedFlatDigest =
      take(loom::pnr::computeSystemPnrSearchDomainDigest(
          loom::pnr::systemPnrSearchDomainSchemaDescriptorBytes(),
          mismatchedFlatBytes));
  requireFailureContains(
      loom::pnr::adoptSystemPnrSearchDomain(
          loom::pnr::systemPnrSearchDomainSchemaDescriptorBytes(),
          mismatchedFlatBytes, mismatchedFlatDigest, store),
      "Spatial MappingConstraintSet has foreign T/F owners");
  auto searchDomain = take(loom::pnr::projectSystemPnrSearchDomain(
      dataflow, system, config, constraints, partition,
      loom::pnr::SystemHierarchicalGraphSearchInput{spatialMappings}, store));
  require(!searchDomain.serviceObligations().empty(),
          "System route fixture has no service obligation");
  auto problem = take(loom::pnr::freezeSystemPnrProblem(
      dataflow, system, searchDomain, config, constraints, store));

  require(problem->threadDecisions().size() == 2 &&
              problem->graphDecisions().size() == 4,
          "frozen System problem merged execution atoms");
  require(problem->accCores().size() == 5 &&
              problem->spatialMappings().size() == 2 &&
              problem->targetClasses().size() == 2,
          "frozen System target catalogs are incomplete");
  require(!problem->serviceLegs().empty(),
          "frozen System problem lost its service legs");

  auto first = take(loom::pnr::initializeCanonicalSystemCandidate(problem));
  auto second = take(loom::pnr::initializeCanonicalSystemCandidate(problem));
  require(first.state->threadChoices() == second.state->threadChoices() &&
              first.state->graphChoices() == second.state->graphChoices() &&
              first.assignmentAttempts == second.assignmentAttempts,
          "canonical System initializer is not deterministic");
  require(first.state->serviceRoutes().size() == problem->serviceLegs().size(),
          "canonical System initializer did not route every service leg");
  for (const loom::pnr::SystemServiceRouteSelection &route :
       first.state->serviceRoutes()) {
    require(route.nodeCount != 0 && route.sinkCount != 0,
            "canonical System route is empty");
    require(route.rootEndpoint != loom::pnr::getInvalidPnrIndex(),
            "canonical System route has no root endpoint");
  }
  if (llvm::Error error = first.state->verify())
    fail(llvm::toString(std::move(error)));

  std::vector<loom::pnr::SystemServiceRouteSelection> incompleteRoutes(
      first.state->serviceRoutes().begin(), first.state->serviceRoutes().end());
  incompleteRoutes.front().sinkCount = 0;
  requireFailureContains(
      loom::pnr::SystemCandidateState::create(
          problem,
          {first.state->threadChoices(), first.state->graphChoices(),
           incompleteRoutes, first.state->serviceRouteNodes(),
           first.state->serviceRouteSinks(), first.state->serviceTargets(),
           first.state->instructionResourceUses(),
           first.state->serviceResourceUses()}),
      "service route does not cover the applicable sink-owner set");

  std::vector<loom::pnr::SystemServiceRouteSinkSelection> foreignSinks(
      first.state->serviceRouteSinks().begin(),
      first.state->serviceRouteSinks().end());
  foreignSinks.front().terminal =
      problem->serviceLegs()[first.state->serviceRoutes().front().leg]
          .sourceTerminal;
  requireFailureContains(
      loom::pnr::SystemCandidateState::create(
          problem,
          {first.state->threadChoices(), first.state->graphChoices(),
           first.state->serviceRoutes(), first.state->serviceRouteNodes(),
           foreignSinks, first.state->serviceTargets(),
           first.state->instructionResourceUses(),
           first.state->serviceResourceUses()}),
      "service route sink is outside its exact H domain");

  auto withCanonicalRoutes =
      [&](llvm::ArrayRef<loom::pnr::PnrIndex> threadChoices,
          llvm::ArrayRef<loom::pnr::PnrIndex> graphChoices) {
        return loom::pnr::SystemCandidateInitialization{
            threadChoices,
            graphChoices,
            first.state->serviceRoutes(),
            first.state->serviceRouteNodes(),
            first.state->serviceRouteSinks(),
            first.state->serviceTargets(),
            {},
            {}};
      };

  auto firstDraft =
      take(loom::pnr::materializeSystemCandidateDraft(*first.state, context));
  auto secondDraft =
      take(loom::pnr::materializeSystemCandidateDraft(*first.state, context));
  auto firstRoot = mlir::cast<::mapping::SystemOp>(firstDraft.get());
  verifyFinalizedSystemMappingWorkflow(*first.state, dataflow, system,
                                       constraints.view(), store, context,
                                       problem->serviceDomains().size());
  std::size_t materializedRouteCount = 0;
  for (auto service :
       firstRoot.getBody().front().getOps<::mapping::ServiceRealizationOp>()) {
    auto selections =
        service.getBody().front().getOps<::mapping::ServicePlanSelectionOp>();
    require(selections.begin() != selections.end(),
            "materialized service has no contextual plan selection");
    for (auto selection : selections)
      take(loom::mapping::decodeServicePlanSelectionKey(
          unsignedBytes(selection.getKey().getRecord()),
          problem->dataflowIdentity()));
    require(llvm::hasSingleElement(
                service.getBody().front().getOps<::mapping::ServicePlanOp>()),
            "materialized service has more than one selected plan");
    auto plan =
        *service.getBody().front().getOps<::mapping::ServicePlanOp>().begin();
    require(plan.getPlanOrdinal() == 0,
            "materialized selected service plan has a nonzero ordinal");
    for (auto route :
         plan.getBody().front().getOps<::mapping::TransferLegRealizationOp>()) {
      const auto leg = take(loom::mapping::decodeCanonicalServiceLegKey(
          unsignedBytes(route.getLeg().getRecord()),
          problem->dataflowIdentity()));
      loom::pnr::PnrIndex selectedOrdinal = loom::pnr::getInvalidPnrIndex();
      for (const auto &[ordinal, selected] :
           llvm::enumerate(first.state->serviceRoutes()))
        if (problem->serviceLegs()[selected.leg].key == leg) {
          selectedOrdinal = static_cast<loom::pnr::PnrIndex>(ordinal);
          break;
        }
      require(selectedOrdinal != loom::pnr::getInvalidPnrIndex(),
              "materialized route has no selected Candidate route");
      const auto &selected = first.state->serviceRoutes()[selectedOrdinal];
      const auto expectedRoot = loom::fabric::canonicalFabricBytes(
          problem->routingTopology()
              .endpoints()[selected.rootEndpoint]
              .reference);
      require(unsignedBytes(route.getRootEndpoint().getRecord()) ==
                  std::vector<std::uint8_t>(expectedRoot.begin(),
                                            expectedRoot.end()),
              "materialized route changed its selected root endpoint");

      auto selectedNodes = first.state->serviceRouteNodes().slice(
          selected.nodeOffset, selected.nodeCount);
      auto materializedNodes =
          route.getBody().front().getOps<::mapping::SystemRouteNodeOp>();
      require(
          std::distance(materializedNodes.begin(), materializedNodes.end()) +
                  1 ==
              selectedNodes.size(),
          "materialized route changed its node count");
      for (const auto &[nodeOrdinal, node] :
           llvm::enumerate(materializedNodes)) {
        const auto &expected = selectedNodes[nodeOrdinal + 1];
        const auto expectedTraversal = loom::fabric::canonicalFabricBytes(
            problem->routingTopology()
                .traversals()[expected.incomingTraversal]
                .reference);
        require(node.getNodeOrdinal() == nodeOrdinal + 1 &&
                    node.getParentNodeOrdinal() == expected.parentNode &&
                    unsignedBytes(node.getIncomingTraversal().getRecord()) ==
                        std::vector<std::uint8_t>(expectedTraversal.begin(),
                                                  expectedTraversal.end()),
                "materialized route changed a selected traversal");
      }

      auto selectedSinks = first.state->serviceRouteSinks().slice(
          selected.sinkOffset, selected.sinkCount);
      auto materializedSinks =
          route.getBody().front().getOps<::mapping::SystemRouteSinkOp>();
      require(std::distance(materializedSinks.begin(),
                            materializedSinks.end()) == selectedSinks.size(),
              "materialized route changed its sink count");
      for (const auto &[sinkOrdinal, sink] :
           llvm::enumerate(materializedSinks)) {
        const auto &expected = selectedSinks[sinkOrdinal];
        const auto expectedTerminal =
            take(loom::mapping::encodeSystemTransferTerminalKey(
                problem->dataflowIdentity(),
                problem->serviceTerminals()[expected.terminal].key));
        require(unsignedBytes(sink.getTerminal().getRecord()) ==
                        expectedTerminal &&
                    sink.getNodeOrdinal() == expected.node,
                "materialized route changed a selected sink attachment");
      }
      ++materializedRouteCount;
    }
  }
  require(materializedRouteCount == first.state->serviceRoutes().size(),
          "materializer omitted a selected service route");
  auto firstBytes =
      take(loom::mapping::writeCanonicalSystemMappingAssembly(firstRoot));
  auto secondBytes = take(loom::mapping::writeCanonicalSystemMappingAssembly(
      mlir::cast<::mapping::SystemOp>(secondDraft.get())));
  require(firstBytes.bytes() == secondBytes.bytes(),
          "System execution materialization is not deterministic");

  mlir::OwningOpRef<mlir::Operation *> reordered(firstDraft->clone());
  auto reorderedRoot = mlir::cast<::mapping::SystemOp>(reordered.get());
  llvm::SmallVector<mlir::Attribute> reversedRoots(
      reorderedRoot.getRootThreadLaunches().begin(),
      reorderedRoot.getRootThreadLaunches().end());
  std::reverse(reversedRoots.begin(), reversedRoots.end());
  reorderedRoot.setRootThreadLaunchesAttr(
      mlir::ArrayAttr::get(&context, reversedRoots));
  auto reorderedBytes =
      take(loom::mapping::writeCanonicalSystemMappingAssembly(reorderedRoot));
  require(reorderedBytes.bytes() == firstBytes.bytes(),
          "System root authoring order changed canonical bytes");
  auto rawReorderedBytes = rawSystemBytes(reorderedRoot);
  require(rawReorderedBytes.bytes() != firstBytes.bytes(),
          "noncanonical System fixture accidentally matched canonical bytes");
  requireFailureContains(loom::mapping::strictImportSystemExecutionBindings(
                             rawReorderedBytes, dataflow, system, store),
                         "payload is not canonical");

  mlir::OwningOpRef<mlir::Operation *> missingThread(firstDraft->clone());
  auto missingRoot = mlir::cast<::mapping::SystemOp>(missingThread.get());
  auto missingBinding = *missingRoot.getBody()
                             .front()
                             .getOps<::mapping::ThreadExecutionBindingOp>()
                             .begin();
  missingBinding.erase();
  requireVerificationFailureContains(missingRoot,
                                     "exactly one ThreadExecutionBinding");

  mlir::OwningOpRef<mlir::Operation *> defaultOnly(firstDraft->clone());
  auto defaultRoot = mlir::cast<::mapping::SystemOp>(defaultOnly.get());
  auto defaultBinding = *defaultRoot.getBody()
                             .front()
                             .getOps<::mapping::ThreadExecutionBindingOp>()
                             .begin();
  auto defaultClause = *defaultBinding.getBody()
                            .front()
                            .getOps<::mapping::ThreadPresburgerClauseOp>()
                            .begin();
  defaultBinding->setAttr("default_target", defaultClause.getTarget());
  defaultClause.erase();
  auto defaultBytes =
      take(loom::mapping::writeCanonicalSystemMappingAssembly(defaultRoot));
  auto defaultExecution =
      take(loom::mapping::strictImportSystemExecutionBindings(
          defaultBytes, dataflow, system, store));
  require(defaultExecution.threadBindings().front().clauses.empty() &&
              defaultExecution.threadBindings().front().defaultTarget,
          "default-only whole-domain relation did not round trip");

  mlir::OwningOpRef<mlir::Operation *> graphDefaultOnly(firstDraft->clone());
  auto graphDefaultRoot =
      mlir::cast<::mapping::SystemOp>(graphDefaultOnly.get());
  auto graphDefaultBinding = *graphDefaultRoot.getBody()
                                  .front()
                                  .getOps<::mapping::GraphExecutionBindingOp>()
                                  .begin();
  auto graphDefaultClause = *graphDefaultBinding.getBody()
                                 .front()
                                 .getOps<::mapping::GraphPresburgerClauseOp>()
                                 .begin();
  graphDefaultBinding->setAttr("default_target",
                               graphDefaultClause.getTarget());
  graphDefaultClause.erase();
  auto graphDefaultBytes = take(
      loom::mapping::writeCanonicalSystemMappingAssembly(graphDefaultRoot));
  auto graphDefaultExecution =
      take(loom::mapping::strictImportSystemExecutionBindings(
          graphDefaultBytes, dataflow, system, store));
  require(graphDefaultExecution.graphBindings().front().clauses.empty() &&
              graphDefaultExecution.graphBindings().front().defaultTarget,
          "default-only graph relation did not round trip");

  mlir::OwningOpRef<mlir::Operation *> emptyThread(firstDraft->clone());
  auto emptyThreadBinding = *mlir::cast<::mapping::SystemOp>(emptyThread.get())
                                 .getBody()
                                 .front()
                                 .getOps<::mapping::ThreadExecutionBindingOp>()
                                 .begin();
  emptyThreadBinding.getBody().front().front().erase();
  requireVerificationFailureContains(emptyThreadBinding,
                                     "requires a clause or default target");

  mlir::OwningOpRef<mlir::Operation *> emptyGraph(firstDraft->clone());
  auto emptyGraphBinding = *mlir::cast<::mapping::SystemOp>(emptyGraph.get())
                                .getBody()
                                .front()
                                .getOps<::mapping::GraphExecutionBindingOp>()
                                .begin();
  emptyGraphBinding.getBody().front().front().erase();
  requireVerificationFailureContains(emptyGraphBinding,
                                     "requires a clause or default target");

  mlir::OwningOpRef<mlir::Operation *> domainGap(firstDraft->clone());
  auto gapBinding = *mlir::cast<::mapping::SystemOp>(domainGap.get())
                         .getBody()
                         .front()
                         .getOps<::mapping::ThreadExecutionBindingOp>()
                         .begin();
  auto gapClause = *gapBinding.getBody()
                        .front()
                        .getOps<::mapping::ThreadPresburgerClauseOp>()
                        .begin();
  auto wholeCell =
      mlir::cast<::mapping::SystemPresburgerCellAttr>(gapClause.getCells()[0]);
  auto partialCell = withFirstCoordinateLowerBound(wholeCell, 1);
  gapClause->setAttr("cells", mlir::ArrayAttr::get(&context, {partialCell}));
  auto gapBytes = take(loom::mapping::writeCanonicalSystemMappingAssembly(
      mlir::cast<::mapping::SystemOp>(domainGap.get())));
  requireFailureContains(loom::mapping::strictImportSystemExecutionBindings(
                             gapBytes, dataflow, system, store),
                         "does not cover its Dataflow may-domain");

  mlir::OwningOpRef<mlir::Operation *> domainOverlap(firstDraft->clone());
  auto overlapBinding = *mlir::cast<::mapping::SystemOp>(domainOverlap.get())
                             .getBody()
                             .front()
                             .getOps<::mapping::ThreadExecutionBindingOp>()
                             .begin();
  auto overlapClause = *overlapBinding.getBody()
                            .front()
                            .getOps<::mapping::ThreadPresburgerClauseOp>()
                            .begin();
  llvm::SmallVector<mlir::Attribute> overlappingCells = {wholeCell,
                                                         partialCell};
  overlapClause->setAttr("cells",
                         mlir::ArrayAttr::get(&context, overlappingCells));
  auto overlapBytes = take(loom::mapping::writeCanonicalSystemMappingAssembly(
      mlir::cast<::mapping::SystemOp>(domainOverlap.get())));
  requireFailureContains(loom::mapping::strictImportSystemExecutionBindings(
                             overlapBytes, dataflow, system, store),
                         "overlapping Presburger cells");

  mlir::OwningOpRef<mlir::Operation *> redundantDefault(firstDraft->clone());
  auto redundantBinding =
      *mlir::cast<::mapping::SystemOp>(redundantDefault.get())
           .getBody()
           .front()
           .getOps<::mapping::ThreadExecutionBindingOp>()
           .begin();
  auto redundantClause = *redundantBinding.getBody()
                              .front()
                              .getOps<::mapping::ThreadPresburgerClauseOp>()
                              .begin();
  redundantBinding->setAttr("default_target", redundantClause.getTarget());
  requireFailureContains(loom::mapping::strictImportSystemExecutionBindings(
                             rawSystemBytes(mlir::cast<::mapping::SystemOp>(
                                 redundantDefault.get())),
                             dataflow, system, store),
                         "default is forbidden for an empty complement");

  const auto selectedMapping = first.state->selectedSpatialMapping(0);
  const auto unselectedMapping = spatialMappings.front() == selectedMapping
                                     ? spatialMappings.back()
                                     : spatialMappings.front();
  mlir::OwningOpRef<mlir::Operation *> extraImport(firstDraft->clone());
  auto extraImportRoot = mlir::cast<::mapping::SystemOp>(extraImport.get());
  llvm::SmallVector<mlir::Attribute> imports(
      extraImportRoot.getSpatialMappingImports().begin(),
      extraImportRoot.getSpatialMappingImports().end());
  imports.push_back(rootReferenceAttr(&context, unselectedMapping));
  extraImportRoot.setSpatialMappingImportsAttr(
      mlir::ArrayAttr::get(&context, imports));
  auto extraImportBytes =
      take(loom::mapping::writeCanonicalSystemMappingAssembly(extraImportRoot));
  requireFailureContains(loom::mapping::strictImportSystemExecutionBindings(
                             extraImportBytes, dataflow, system, store),
                         "not the exact selected B_graph range");

  mlir::OwningOpRef<mlir::Operation *> incompatible(firstDraft->clone());
  auto incompatibleRoot = mlir::cast<::mapping::SystemOp>(incompatible.get());
  llvm::SmallVector<mlir::Attribute> incompatibleImports(
      incompatibleRoot.getSpatialMappingImports().begin(),
      incompatibleRoot.getSpatialMappingImports().end());
  incompatibleImports.push_back(rootReferenceAttr(&context, unselectedMapping));
  incompatibleRoot.setSpatialMappingImportsAttr(
      mlir::ArrayAttr::get(&context, incompatibleImports));
  auto incompatibleBinding = *incompatibleRoot.getBody()
                                  .front()
                                  .getOps<::mapping::GraphExecutionBindingOp>()
                                  .begin();
  auto incompatibleClause = *incompatibleBinding.getBody()
                                 .front()
                                 .getOps<::mapping::GraphPresburgerClauseOp>()
                                 .begin();
  incompatibleClause->setAttr(
      "target", ::mapping::SpatialMappingImportRefAttr::get(&context, 1));
  auto incompatibleBytes = take(
      loom::mapping::writeCanonicalSystemMappingAssembly(incompatibleRoot));
  requireFailureContains(loom::mapping::strictImportSystemExecutionBindings(
                             incompatibleBytes, dataflow, system, store),
                         "graph and thread targets are incompatible");

  auto execution = take(loom::mapping::strictImportSystemExecutionBindings(
      firstBytes, dataflow, system, store));
  require(execution.rootThreadLaunches().size() == 2 &&
              execution.threadBindings().size() == 2 &&
              execution.graphBindings().size() == 4,
          "strict execution import lost factorized binding keys");
  require(execution.spatialMappingImports().size() == 1,
          "System import table is not the exact selected B_graph range");
  for (const auto &binding : execution.threadBindings())
    require(binding.clauses.size() == 1 && !binding.defaultTarget,
            "whole-domain thread binding was not canonicalized");
  for (const auto &binding : execution.graphBindings())
    require(binding.clauses.size() == 1 && !binding.defaultTarget &&
                binding.clauses.front().target ==
                    execution.spatialMappingImports().front(),
            "whole-domain graph binding did not resolve its exact import");
  for (loom::pnr::PnrIndex decision = 0;
       decision != problem->graphDecisions().size(); ++decision) {
    const auto graphDomain = problem->graphChoiceCatalogOrdinals(decision);
    const auto selectedMapping =
        graphDomain[first.state->graphChoice(decision)];
    const auto threadDomain = problem->threadChoiceCatalogOrdinals(
        problem->graphDecisions()[decision].launch.rootThreadLaunch ==
                problem->threadDecisions().front().root
            ? 0
            : 1);
    const auto selectedCore = threadDomain[first.state->threadChoice(
        problem->graphDecisions()[decision].launch.rootThreadLaunch ==
                problem->threadDecisions().front().root
            ? 0
            : 1)];
    require(problem->spatialMappingTargetClass(selectedMapping) ==
                problem->accCoreTargetClass(selectedCore),
            "canonical initializer selected incompatible execution targets");
  }

  std::vector<loom::pnr::PnrIndex> threadChoices(
      problem->threadDecisions().size(), 0);
  std::vector<loom::pnr::PnrIndex> graphChoices(
      problem->graphDecisions().size(), 0);
  require(problem->threadChoiceCatalogOrdinals(0).size() > 1,
          "fixture needs two compatible AccCore choices");
  const auto initialThreadDomain = problem->threadChoiceCatalogOrdinals(0);
  loom::pnr::PnrIndex sameClassFirst = 0;
  loom::pnr::PnrIndex sameClassSecond = 0;
  loom::pnr::PnrIndex sharedClass = 0;
  bool foundSameClassAlternative = false;
  for (loom::pnr::PnrIndex firstChoice = 0;
       firstChoice != initialThreadDomain.size() && !foundSameClassAlternative;
       ++firstChoice)
    for (loom::pnr::PnrIndex secondChoice = firstChoice + 1;
         secondChoice != initialThreadDomain.size(); ++secondChoice)
      if (problem->accCoreTargetClass(initialThreadDomain[firstChoice]) ==
          problem->accCoreTargetClass(initialThreadDomain[secondChoice])) {
        sameClassFirst = firstChoice;
        sameClassSecond = secondChoice;
        sharedClass =
            problem->accCoreTargetClass(initialThreadDomain[firstChoice]);
        foundSameClassAlternative = true;
        break;
      }
  require(foundSameClassAlternative,
          "fixture needs two AccCores in one SpatialCore target class");

  for (loom::pnr::PnrIndex decision = 0;
       decision != problem->threadDecisions().size(); ++decision) {
    const auto domain = problem->threadChoiceCatalogOrdinals(decision);
    bool found = false;
    for (loom::pnr::PnrIndex choice = 0; choice != domain.size(); ++choice)
      if (problem->accCoreTargetClass(domain[choice]) == sharedClass) {
        threadChoices[decision] = choice;
        found = true;
        break;
      }
    require(found, "thread domain lost a compatible target class");
  }
  for (loom::pnr::PnrIndex decision = 0;
       decision != problem->graphDecisions().size(); ++decision) {
    const auto domain = problem->graphChoiceCatalogOrdinals(decision);
    bool found = false;
    for (loom::pnr::PnrIndex choice = 0; choice != domain.size(); ++choice)
      if (problem->spatialMappingTargetClass(domain[choice]) == sharedClass) {
        graphChoices[decision] = choice;
        found = true;
        break;
      }
    require(found, "graph domain lost a compatible target class");
  }
  threadChoices[0] = sameClassFirst;
  auto sameClassBase = take(loom::pnr::initializeSystemCandidate(
      problem, threadChoices, graphChoices));
  threadChoices[0] = sameClassSecond;
  auto alternate = take(loom::pnr::initializeSystemCandidate(
      problem, threadChoices, graphChoices));
  requireFailureContains(
      loom::pnr::SystemCandidateState::create(
          problem,
          {threadChoices, graphChoices, sameClassBase->serviceRoutes(),
           sameClassBase->serviceRouteNodes(),
           sameClassBase->serviceRouteSinks(), sameClassBase->serviceTargets(),
           alternate->instructionResourceUses(),
           alternate->serviceResourceUses()}),
      "is not admitted by H");
  if (llvm::Error error = alternate->verify())
    fail(llvm::toString(std::move(error)));
  require(alternate->selectedAccCore(0) != sameClassBase->selectedAccCore(0),
          "explicit thread choice did not change the selected AccCore");
  verifyFinalizedSystemMappingWorkflow(*alternate, dataflow, system,
                                       constraints.view(), store, context,
                                       problem->serviceDomains().size());

  const auto firstThreadDomain = problem->threadChoiceCatalogOrdinals(0);
  const auto firstGraphDomain = problem->graphChoiceCatalogOrdinals(0);
  bool foundMismatch = false;
  for (loom::pnr::PnrIndex threadChoice = 0;
       threadChoice != firstThreadDomain.size() && !foundMismatch;
       ++threadChoice)
    for (loom::pnr::PnrIndex graphChoice = 0;
         graphChoice != firstGraphDomain.size() && !foundMismatch;
         ++graphChoice)
      if (problem->accCoreTargetClass(firstThreadDomain[threadChoice]) !=
          problem->spatialMappingTargetClass(firstGraphDomain[graphChoice])) {
        threadChoices.assign(problem->threadDecisions().size(), threadChoice);
        graphChoices.assign(problem->graphDecisions().size(), graphChoice);
        requireFailureContains(
            loom::pnr::SystemCandidateState::create(
                problem, withCanonicalRoutes(threadChoices, graphChoices)),
            "target classes are incompatible");
        foundMismatch = true;
      }
  require(foundMismatch,
          "heterogeneous fixture did not expose an incompatible target pair");

  threadChoices.assign(problem->threadDecisions().size(), 0);
  graphChoices.assign(problem->graphDecisions().size(), 0);
  threadChoices[0] = problem->threadChoiceCatalogOrdinals(0).size();
  requireFailureContains(
      loom::pnr::SystemCandidateState::create(
          problem, withCanonicalRoutes(threadChoices, graphChoices)),
      "thread choice is outside its H domain");
  threadChoices.pop_back();
  requireFailureContains(
      loom::pnr::SystemCandidateState::create(
          problem, withCanonicalRoutes(threadChoices, graphChoices)),
      "thread choice count does not match H");

  llvm::outs() << "System CandidateState anchors passed\n";
  return EXIT_SUCCESS;
}
