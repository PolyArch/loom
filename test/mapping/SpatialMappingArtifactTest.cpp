#include "ADG/Builder.h"
#include "ADG/Builtin.h"
#include "CgraAdmissionTestSupport.h"
#include "SpatialCandidateSelectionTestSupport.h"
#include "SpatialMemoryMappingArtifactTestSupport.h"
#include "SpatialRuntimeCounterexampleExactRepairTestSupport.h"
#include "SpatialRuntimeCounterexampleTestSupport.h"
#include "TechMappingCandidateTestSupport.h"
#include "TemporalMappingFabricTestSupport.h"
#include "TemporalPeTagDomainTestSupport.h"

#include "Common/ArtifactLocalReference.h"
#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "Config/ResolvedConfig.h"
#include "ConfiguredHardwareProjectionInternal.h"
#include "DSE/MappingCandidateGenerator.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Dataflow/IR/DataflowReferenceCodec.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/IR/FabricDialect.h"
#include "Fabric/IR/FabricOps.h"
#include "Fabric/IR/OperationResourceContract.h"
#include "Fabric/IR/UsePatternValue.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Fabric/Identity/FabricSemanticFieldRelation.h"
#include "Mapping/Artifact/MappingArtifact.h"
#include "Mapping/Artifact/MappingConstraintSet.h"
#include "Mapping/Artifact/SpatialPhysicalDemandProjection.h"
#include "Mapping/IR/MappingDialect.h"
#include "Mapping/Inspection/SpatialMappingInspection.h"
#include "Mapping/Tech/TechMappingConfig.h"
#include "Mapping/Tech/TechMappingGenerator.h"
#include "PnR/MappingObjective.h"
#include "PnR/PnrConfig.h"
#include "PnR/SpatialCandidateInitializer.h"
#include "PnR/SpatialCanonicalSeed.h"
#include "PnR/SpatialGlobalRoutingClosure.h"
#include "PnR/SpatialMappingMaterializer.h"
#include "PnR/SpatialPathFinderRouter.h"
#include "PnR/SpatialPnrGenerator.h"
#include "PnR/SpatialPnrProblem.h"
#include "PnR/SpatialRouteCostState.h"
#include "PnR/SpatialTagAssignment.h"
#include "PnR/SpatialTagContinuity.h"
#include "SpatialOperandPairingPressure.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <numeric>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace {

[[noreturn]] void fail(llvm::StringRef message) {
  llvm::errs() << "spatial mapping artifact test: " << message << '\n';
  std::exit(1);
}

template <typename T> T take(llvm::Expected<T> value) {
  if (!value)
    fail(llvm::toString(value.takeError()));
  return std::move(*value);
}

void requireSuccess(llvm::Error error) {
  if (error)
    fail(llvm::toString(std::move(error)));
}

template <typename T> bool rejected(llvm::Expected<T> value) {
  if (value)
    return false;
  llvm::consumeError(value.takeError());
  return true;
}

bool rejected(llvm::Error error) {
  if (!error)
    return false;
  llvm::consumeError(std::move(error));
  return true;
}

template <typename Callable>
bool rejectedWithoutDiagnostic(mlir::MLIRContext &context,
                               Callable &&callable) {
  mlir::ScopedDiagnosticHandler capture(
      &context, [](mlir::Diagnostic &) { return mlir::success(); });
  return rejected(std::forward<Callable>(callable)());
}

class TemporaryDirectory final {
public:
  TemporaryDirectory() {
    llvm::SmallString<128> path;
    if (std::error_code error = llvm::sys::fs::createUniqueDirectory(
            "loom-spatial-mapping-artifact", path))
      fail("cannot create ArtifactStore directory: " + error.message());
    path_ = path.str().str();
  }

  ~TemporaryDirectory() {
    if (std::error_code error = llvm::sys::fs::remove_directories(path_))
      llvm::errs() << "cannot remove test directory: " << error.message()
                   << '\n';
  }

  llvm::StringRef path() const { return path_; }

private:
  std::string path_;
};

std::size_t storedObjectCount(llvm::StringRef root) {
  std::size_t count = 0;
  std::error_code error;
  llvm::sys::fs::recursive_directory_iterator iterator(root, error), end;
  if (error)
    fail("cannot inspect ArtifactStore: " + error.message());
  while (iterator != end) {
    count += llvm::sys::fs::is_regular_file(iterator->path());
    iterator.increment(error);
    if (error)
      fail("cannot inspect ArtifactStore: " + error.message());
  }
  return count;
}

mlir::MLIRContext makeContext() {
  mlir::DialectRegistry registry;
  registry.insert<::dataflow::DataflowDialect, ::mapping::MappingDialect,
                  ::fabric::FabricDialect, mlir::arith::ArithDialect,
                  mlir::DLTIDialect, mlir::func::FuncDialect,
                  mlir::LLVM::LLVMDialect, mlir::memref::MemRefDialect>();
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
      %value: i32) ctrl (%ctrl: none) {
    %result, %done = dataflow.graph.launch @sync deps(%ctrl)
        values(%value) stream_inputs() memories() stream_outputs()
        : (none, i32) -> (i32, none)
    dataflow.thread.yield %done : none
  }
  func.func private @host() {
    %value = arith.constant 7 : i32
    %thread = dataflow.thread.launch @worker(%value)
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

enum class ComputeContractKind { OneCycleElastic, LoopStream, Transparent };

const ::fabric::ResourceContract &
computeContract(ComputeContractKind contractKind) {
  switch (contractKind) {
  case ComputeContractKind::OneCycleElastic:
    return ::fabric::oneCycleElasticOperationResourceContract();
  case ComputeContractKind::LoopStream:
    return ::fabric::loopStreamOperationResourceContract();
  case ComputeContractKind::Transparent:
    return ::fabric::loopCarryOperationResourceContract();
  }
  fail("unknown compute contract kind");
}

loom::fabric::FinalizedFabricRoot buildFabric(
    loom::ArtifactStore &store,
    ComputeContractKind contractKind = ComputeContractKind::OneCycleElastic) {
  using loom::adg::DesignBuilder;
  using loom::adg::PeSpec;
  using loom::adg::PortType;

  const PortType bits128 = take(PortType::bits(128));
  const std::vector<PortType> types(4, bits128);
  DesignBuilder builder(store);
  auto spatial = take(builder.createSpatialCore("sync", types, types));
  std::vector<loom::adg::SpatialValue> spatialInputs;
  for (std::size_t ordinal = 0; ordinal < types.size(); ++ordinal)
    spatialInputs.push_back(take(spatial.input(ordinal)));
  auto pe = take(spatial.addPe(spatialInputs, PeSpec::spatial(types, types)));
  std::vector<loom::adg::PeValue> peInputs;
  for (std::size_t ordinal = 0; ordinal < types.size(); ++ordinal)
    peInputs.push_back(take(pe.input(ordinal)));
  loom::test::addTokenSyncFu(pe, peInputs, bits128,
                             computeContract(contractKind));
  requireSuccess(pe.close());
  std::vector<loom::adg::SpatialValue> outputs;
  for (std::size_t ordinal = 0; ordinal < types.size(); ++ordinal)
    outputs.push_back(take(pe.output(ordinal)));
  requireSuccess(spatial.close(outputs));
  auto design = take(std::move(builder).finalize());
  if (design.roots().size() != 1)
    fail("Fabric fixture did not publish exactly one root");
  return design.roots().front();
}

loom::fabric::FinalizedFabricRoot
buildTemporalFabric(loom::ArtifactStore &store) {
  auto design = loom::test::buildTemporalCapacityFabric(store);
  if (design.roots().size() != 1)
    fail("Temporal Fabric fixture did not publish exactly one root");
  return design.roots().front();
}

loom::fabric::FinalizedFabricRoot
buildTemporalSwitchPackingFabric(loom::ArtifactStore &store,
                                 std::uint64_t residentRows = 2) {
  auto design =
      loom::test::buildTemporalSwitchPackingFabric(store, residentRows);
  if (design.roots().size() != 1)
    fail("Temporal switch fixture did not publish exactly one root");
  return design.roots().front();
}

void temporalPeTagMatchDomainsAreIngressLocal() {
  TemporaryDirectory directory;
  loom::ArtifactStore store(directory.path());
  const auto fabric = loom::test::buildBoundaryTemporalFabric(store);
  requireSuccess(loom::test::verifyTemporalPeIngressTagDomains(fabric));
}

mlir::OwningOpRef<mlir::ModuleOp>
parseSpatial(mlir::MLIRContext &context,
             const loom::CanonicalSemanticBytes &bytes) {
  std::string text = "module {\n";
  text.append(reinterpret_cast<const char *>(bytes.bytes().data()),
              bytes.bytes().size());
  text += "}\n";
  return mlir::parseSourceString<mlir::ModuleOp>(text, &context);
}

void selectLegalTemporalBinding(loom::pnr::SpatialCandidateState &candidate,
                                loom::pnr::SpatialCandidateScratch &scratch,
                                bool requireLocalTagInterference,
                                bool requireDistinctBoundaryEndpoints) {
  const auto &problem = candidate.problem();
  const auto realizations = problem.realizations().computeRealizations();
  if (realizations.size() != 1)
    fail("Temporal SpatialMapping fixture does not have one realization");
  const auto &realization = realizations.front();
  std::optional<loom::pnr::SpatialComputeBindingSelection> legal;
  for (loom::pnr::PnrIndex placement = realization.placementOffset;
       placement != realization.placementOffset + realization.placementCount;
       ++placement) {
    const auto &record = problem.realizations().computePlacements()[placement];
    for (loom::pnr::PnrIndex context = record.contextOffset;
         context != record.contextOffset + record.contextCount; ++context)
      if (problem.capacity().computeInstructionContextOveruse()[context] == 0) {
        legal = loom::pnr::SpatialComputeBindingSelection{placement, context};
        break;
      }
    if (legal)
      break;
  }
  if (!legal)
    fail("Temporal SpatialMapping fixture has no legal compute binding");
  auto move = take(candidate.beginMove(scratch));
  requireSuccess(
      move.setComputeBinding(0, legal->placement, legal->instructionContext));
  std::vector<loom::pnr::PnrIndex> selectedAttachments(
      problem.ports().portDemands().size());
  std::vector<loom::pnr::PnrIndex> selectedEndpoints;
  for (auto [demandOrdinal, demand] :
       llvm::enumerate(problem.ports().portDemands())) {
    const auto &domain =
        problem.ports()
            .placementDomains()[demand.placementDomainOffset +
                                legal->placement - realization.placementOffset];
    selectedAttachments[demandOrdinal] = domain.attachmentOptionOffset;
    if (requireLocalTagInterference)
      continue;
    const auto options = problem.ports().attachmentOptions();
    const auto available = options.slice(domain.attachmentOptionOffset,
                                         domain.attachmentOptionCount);
    const auto found = llvm::find_if(available, [&](const auto &option) {
      return !llvm::is_contained(selectedEndpoints, option.endpoint);
    });
    if (found == available.end())
      fail("Temporal route fixture has no distinct PE ingress assignment");
    selectedAttachments[demandOrdinal] =
        domain.attachmentOptionOffset +
        static_cast<loom::pnr::PnrIndex>(found - available.begin());
    selectedEndpoints.push_back(found->endpoint);
  }
  if (requireLocalTagInterference) {
    bool foundSharedIngress = false;
    const auto demands = problem.ports().portDemands();
    const auto domains = problem.ports().placementDomains();
    const auto options = problem.ports().attachmentOptions();
    for (std::size_t lhs = 0; !foundSharedIngress && lhs < demands.size();
         ++lhs) {
      const auto &lhsDomain =
          domains[demands[lhs].placementDomainOffset + legal->placement -
                  realization.placementOffset];
      for (std::size_t rhs = lhs + 1;
           !foundSharedIngress && rhs < demands.size(); ++rhs) {
        if (demands[lhs].logicalNet == demands[rhs].logicalNet)
          continue;
        const auto &rhsDomain =
            domains[demands[rhs].placementDomainOffset + legal->placement -
                    realization.placementOffset];
        for (loom::pnr::PnrIndex lhsOption = lhsDomain.attachmentOptionOffset;
             !foundSharedIngress &&
             lhsOption != lhsDomain.attachmentOptionOffset +
                              lhsDomain.attachmentOptionCount;
             ++lhsOption)
          for (loom::pnr::PnrIndex rhsOption = rhsDomain.attachmentOptionOffset;
               rhsOption != rhsDomain.attachmentOptionOffset +
                                rhsDomain.attachmentOptionCount;
               ++rhsOption)
            if (options[lhsOption].endpoint == options[rhsOption].endpoint) {
              selectedAttachments[lhs] = lhsOption;
              selectedAttachments[rhs] = rhsOption;
              foundSharedIngress = true;
              break;
            }
      }
    }
    if (!foundSharedIngress)
      fail("Temporal route fixture has no shared ingress attachment");
  }
  for (auto [demandOrdinal, option] : llvm::enumerate(selectedAttachments))
    requireSuccess(move.setPortAttachment(
        static_cast<loom::pnr::PnrIndex>(demandOrdinal), option));
  requireSuccess(loom::test::selectReachableGraphBoundaries(
      candidate, move, selectedAttachments, requireDistinctBoundaryEndpoints));
  if (!take(move.close()))
    fail("legal Temporal binding closes a selected handshake cycle");
  requireSuccess(move.commit());
}

void operandPairingPressureIsIncremental(
    loom::pnr::SpatialCandidateState &candidate,
    loom::pnr::SpatialCandidateScratch &scratch) {
  const auto &problem = candidate.problem();
  const auto groups = problem.ports().operandPairingGroups();
  if (groups.empty())
    fail("Temporal workflow has no Dataflow-owned operand pairing group");
  const auto options = problem.ports().attachmentOptions();
  std::optional<loom::pnr::PnrIndex> changedDemand;
  std::optional<loom::pnr::PnrIndex> sharedOption;
  for (loom::pnr::PnrIndex group = 0; group < groups.size() && !sharedOption;
       ++group) {
    const auto members = problem.ports().operandPairingGroupMembers(group);
    for (std::size_t lhs = 0; lhs < members.size() && !sharedOption; ++lhs) {
      const auto &lhsOption = options[candidate.portAttachment(members[lhs])];
      for (std::size_t rhs = lhs + 1; rhs < members.size() && !sharedOption;
           ++rhs) {
        const loom::pnr::PnrIndex demand = members[rhs];
        const auto &record = problem.ports().portDemands()[demand];
        const auto &binding = candidate.computeBinding(record.realization);
        for (const auto &domain : problem.ports().placementDomains().slice(
                 record.placementDomainOffset, record.placementDomainCount)) {
          if (domain.placement != binding.placement)
            continue;
          for (loom::pnr::PnrIndex option = domain.attachmentOptionOffset;
               option !=
               domain.attachmentOptionOffset + domain.attachmentOptionCount;
               ++option) {
            if (option != candidate.portAttachment(demand) &&
                options[option].endpoint == lhsOption.endpoint &&
                options[option].progressBoundary ==
                    loom::mapping::SpatialDurableProgressBoundaryKind::
                        TemporalPeOperandQueue) {
              changedDemand = demand;
              sharedOption = option;
              break;
            }
          }
        }
      }
    }
  }
  if (!changedDemand || !sharedOption)
    fail("Temporal workflow has no same-ingress analytic alternative");

  const std::uint64_t baseline = candidate.sharedOperandIngressPressure();
  {
    auto move = take(candidate.beginMove(scratch));
    requireSuccess(move.setPortAttachment(*changedDemand, *sharedOption));
    if (candidate.sharedOperandIngressPressure() <= baseline)
      fail("same-ingress ordered inputs did not increase pairing pressure");
    const auto objectivePressure = take(loom::pnr::spatialMappingMeasureValue(
        candidate,
        loom::pnr::MappingMeasureKind::SharedOperandIngressPressure));
    if (objectivePressure != candidate.sharedOperandIngressPressure())
      fail("central objective lost incremental operand pairing pressure");
    std::vector<loom::pnr::PnrIndex> registerFifoTransfers(
        problem.transfers().logicalNets().size());
    for (loom::pnr::PnrIndex logicalNet = 0;
         logicalNet < registerFifoTransfers.size(); ++logicalNet)
      registerFifoTransfers[logicalNet] =
          candidate.registerFifoTransfer(logicalNet);
    const auto coldPressure =
        take(loom::pnr::detail::measureSpatialOperandIngressPressure(
            problem, candidate.portAttachmentSelections(),
            registerFifoTransfers));
    if (coldPressure != candidate.sharedOperandIngressPressure())
      fail("incremental operand pairing pressure differs from cold replay");
    move.rollback();
  }
  if (candidate.sharedOperandIngressPressure() != baseline)
    fail("operand pairing rollback did not restore the candidate measure");
  requireSuccess(candidate.verify());
}

void verifyTagDomainIncidence(
    const loom::pnr::SpatialCandidateState &candidate,
    const loom::pnr::SpatialRouteCostState &costs) {
  using namespace loom::pnr;
  for (PnrIndex logicalNet = 0;
       logicalNet < candidate.problem().transfers().logicalNets().size();
       ++logicalNet) {
    std::vector<PnrIndex> domains;
    for (PnrIndex segment = 0;
         segment < candidate.tagSegments(logicalNet).size(); ++segment) {
      const auto selected = candidate.tagSegmentDomains(logicalNet, segment);
      domains.insert(domains.end(), selected.begin(), selected.end());
    }
    llvm::sort(domains);
    domains.erase(std::unique(domains.begin(), domains.end()), domains.end());
    const auto uses = costs.logicalNetTagDomainUses(logicalNet);
    if (domains.size() != uses.size() ||
        !llvm::equal(domains, uses, [](PnrIndex domain, const auto &use) {
          return domain == use.domain;
        }))
      fail("marginal row cost changed selected tag-domain membership");
  }
}

enum class TemporalSwitchRouteFixture : std::uint8_t {
  None,
  PackedRows,
  SameInputSeparatedRows,
  ContendingSeparatedRows,
  ExceedsResidentCapacity,
};

void completeCandidateRoundTrip(
    bool temporal, bool boundaryWrapped = false, bool forceTagConflict = false,
    ComputeContractKind contractKind = ComputeContractKind::OneCycleElastic,
    TemporalSwitchRouteFixture switchFixture =
        TemporalSwitchRouteFixture::None) {
  TemporaryDirectory directory;
  loom::ArtifactStore store(directory.path());
  llvm::SmallString<128> blobPath(directory.path());
  llvm::sys::path::append(blobPath, "blobs");
  if (std::error_code error = llvm::sys::fs::create_directories(blobPath))
    fail("cannot create BlobStore directory: " + error.message());
  const loom::BlobStore blobs(blobPath);
  mlir::MLIRContext context = makeContext();

  auto dataflowArtifact = buildDataflow(context);
  const auto dataflowReference =
      take(dataflow::publishCanonicalDataflow(dataflowArtifact, store));
  auto dataflow = take(dataflowArtifact.view());
  const bool switchPackingFabric =
      switchFixture != TemporalSwitchRouteFixture::None;
  const bool requireSeparatedSwitchRows =
      switchFixture == TemporalSwitchRouteFixture::SameInputSeparatedRows ||
      switchFixture == TemporalSwitchRouteFixture::ContendingSeparatedRows ||
      switchFixture == TemporalSwitchRouteFixture::ExceedsResidentCapacity;
  const bool requireContendingSwitchRows =
      switchFixture == TemporalSwitchRouteFixture::ContendingSeparatedRows;
  const bool exceedsSwitchResidentCapacity =
      switchFixture == TemporalSwitchRouteFixture::ExceedsResidentCapacity;
  const std::uint64_t switchResidentRows =
      exceedsSwitchResidentCapacity ? 1 : 2;
  const auto fabric =
      boundaryWrapped ? loom::test::buildBoundaryTemporalFabric(store)
      : switchPackingFabric
          ? buildTemporalSwitchPackingFabric(store, switchResidentRows)
      : temporal ? buildTemporalFabric(store)
                 : buildFabric(store, contractKind);

  loom::ResolvedConfig resolved = loom::defaultResolvedConfig();
  resolved.dse.techMapping.candidatePublicationLimit = 1;
  const auto techConfig =
      take(loom::mapping::projectResolvedTechMappingConfigView(resolved));
  const std::array<dataflow::GraphRef, 1> covers = {
      dataflow.graphs().front().ref};
  auto generated = loom::mapping::generateTechMappings(
      {dataflow, covers, fabric.view(), techConfig, store});
  auto *candidates =
      std::get_if<loom::mapping::GeneratedTechMappings>(&generated);
  if (!candidates || candidates->candidates.size() != 1)
    fail("TechMapping fixture did not produce one candidate");
  const auto tech = take(
      loom::mapping::importTechMapping(candidates->candidates.front(), store));
  const auto constraints = loom::test::buildSpatialMappingConstraints(
      context, dataflow, tech.view(), fabric.view(), store, forceTagConflict);
  if (boundaryWrapped && !forceTagConflict)
    loom::test::exerciseSpatialTagConstraintRelations(
        context, dataflow, tech.view(), fabric.view(), store);
  if (!temporal && !boundaryWrapped && !forceTagConflict)
    loom::test::exerciseSpatialAttachmentConstraintRelations(
        context, dataflow, tech.view(), fabric.view(), store);
  if (contractKind == ComputeContractKind::OneCycleElastic && !temporal &&
      !boundaryWrapped && !forceTagConflict) {
    loom::ResolvedConfig generatorResolved =
        loom::test::buildSpatialPnrTestResolvedConfig();
    auto &search = generatorResolved.dse.spatialPnr.search;
    search.initializer.seedAttemptCount = 2;
    search.actionProposal = {0, 1, 0};
    search.annealing.calibrationProposalCount = 1;
    search.annealing.fallbackTemperature = 1;
    search.annealing.minimumTemperature = 1;
    search.annealing.coolingRatio = {1, 2};
    search.annealing.proposalsPerLevelBase = 1;
    search.annealing.proposalsPerMovableDecision = 0;
    search.exactRepair = {loom::ResolvedPnrExactRepairKind::Disabled, 0, 0};
    const auto generatorConfig =
        take(loom::pnr::projectResolvedSpatialPnrConfigView(generatorResolved));
    const auto physicalTiming =
        take(loom::fabric::projectNormalizedFabricPhysicalTimingProfile(
            fabric.view()));
    const auto physicalTimingReference =
        take(loom::fabric::publishFabricPhysicalTimingProfile(physicalTiming,
                                                              store));
    const auto typedGeneratorInputs =
        take(loom::dse::bindSpatialPnrCandidateGeneratorInputs(
            dataflowReference, candidates->candidates.front(),
            fabric.reference(), physicalTimingReference,
            constraints.reference()));
    const auto generatorBinding = take(
        loom::dse::resolveSpatialPnrCandidateGeneratorBinding(generatorConfig));
    const loom::pnr::SpatialPnrGenerationInputs generatorInputs{
        dataflow,        tech.view(),        fabric.view(), physicalTiming,
        generatorConfig, constraints.view(), store};
    auto generatedSpatial = loom::dse::invokeSpatialPnrCandidateGenerator(
        typedGeneratorInputs, generatorBinding, store);
    const auto *generated =
        std::get_if<loom::pnr::GeneratedSpatialMappings>(&generatedSpatial);
    if (!generated || generated->candidates.size() != 2 ||
        generated->accounting.seedAttemptSlots != 2 ||
        generated->accounting.preparedSeeds != 2 ||
        generated->accounting.finalizedRestarts != 2 ||
        generated->accounting.publicationSlots != 2)
      fail("Spatial PnR generator refilled or lost a fixed restart slot");
    if (!std::is_sorted(generated->candidates.begin(),
                        generated->candidates.end(),
                        loom::artifactRootReferenceLess))
      fail("Spatial PnR generator did not return a canonical candidate set");
    auto genericSpatial = take(loom::dse::invokeCandidateGenerator(
        typedGeneratorInputs, generatorBinding, store, blobs));
    const auto *genericCompleted =
        std::get_if<loom::dse::CompletedCandidateGeneratorResult>(
            &genericSpatial.outcome);
    if (!genericCompleted || genericCompleted->outputBindings.size() != 1 ||
        genericCompleted->outputBindings.front().artifacts !=
            generated->candidates)
      fail("typed Generate provider diverged from its Spatial PnR owner");
    if (genericCompleted->lineageEdges.size() != generated->candidates.size())
      fail("typed Generate provider lost Spatial Mapping lineage");
    for (const loom::dse::CandidateGeneratorLineageEdge &edge :
         genericCompleted->lineageEdges)
      if (edge.kind != loom::dse::CandidateGeneratorLineageEdgeKind::
                           MechanicalDerivation ||
          edge.outputSlot != loom::dse::CandidateGeneratorOutputSlotRef(0) ||
          !std::binary_search(generated->candidates.begin(),
                              generated->candidates.end(), edge.output,
                              loom::artifactRootReferenceLess) ||
          !edge.parents.empty() || !edge.ownerPayload.empty())
        fail("Spatial Mapping lineage is not a mechanical derivation");
    auto repeatedSpatial = loom::dse::invokeSpatialPnrCandidateGenerator(
        typedGeneratorInputs, generatorBinding, store);
    const auto *repeated =
        std::get_if<loom::pnr::GeneratedSpatialMappings>(&repeatedSpatial);
    if (!repeated || repeated->candidates != generated->candidates ||
        !(repeated->accounting == generated->accounting))
      fail("Spatial PnR generator is not deterministic for exact inputs");
    auto directSpatial = loom::pnr::generateSpatialMappings(generatorInputs);
    const auto *direct =
        std::get_if<loom::pnr::GeneratedSpatialMappings>(&directSpatial);
    if (!direct || direct->candidates != generated->candidates ||
        !(direct->accounting == generated->accounting))
      fail("central Spatial generator diverged from its PnR owner API");
    auto generatedView = take(loom::mapping::importSpatialMapping(
        generated->candidates.front(), store));
    if (generatedView.view().computeBindings().empty() ||
        generatedView.view().routeTrees().empty() ||
        generatedView.view().resourceUses().empty())
      fail("Spatial PnR generator published an empty Mapping");
    const auto inspection = take(loom::mapping::inspectSpatialMapping(
        dataflow, tech.view(), fabric.view(), generatedView.view()));
    if (inspection.summary.computeRealizationCount == 0 ||
        inspection.summary.selectedActorCount == 0 ||
        inspection.summary.routeTreeCount == 0 ||
        inspection.summary.routeNodeCount == 0 ||
        inspection.summary.routeSinkCount == 0 ||
        inspection.summary.resourceUseCount == 0 ||
        inspection.computeOccupancy.empty() || inspection.routes.empty())
      fail("Spatial Mapping inspection projected an empty physical mapping");
    const auto foreignFabric = buildTemporalFabric(store);
    loom::test::exerciseCgraAdmission(
        dataflowReference, fabric.reference(), generated->candidates.front(),
        foreignFabric.reference(), store, blobs, false, true, false, true);
    auto wrongInspection = loom::mapping::inspectSpatialMapping(
        dataflow, tech.view(), foreignFabric.view(), generatedView.view());
    if (wrongInspection)
      fail("Spatial Mapping inspection accepted a foreign Fabric");
    llvm::consumeError(wrongInspection.takeError());
    requireSuccess(loom::mapping::admitSpatialMappingConstraints(
        dataflow, tech.view(), fabric.view(), constraints.view(),
        generatedView.view()));

    const auto placementRejectingConstraints =
        loom::test::buildSpatialMappingConstraints(
            context, dataflow, tech.view(), fabric.view(), store,
            /*restrictTagsToZero=*/false, /*rejectComputePlacement=*/true);
    llvm::Error rejection = loom::mapping::admitSpatialMappingConstraints(
        dataflow, tech.view(), fabric.view(),
        placementRejectingConstraints.view(), generatedView.view());
    bool rejectedByConstraintSet = false;
    llvm::handleAllErrors(
        std::move(rejection),
        [&](const loom::mapping::SpatialMappingConstraintRejection &) {
          rejectedByConstraintSet = true;
        },
        [&](const llvm::ErrorInfoBase &error) { fail(error.message()); });
    if (!rejectedByConstraintSet)
      fail("Spatial constraint admission accepted a forbidden compute "
           "placement");

    loom::test::exerciseSpatialRuntimeCounterexampleNoGood(
        dataflow, tech.view(), fabric.view(), foreignFabric.view(), constraints,
        generatedView, physicalTiming, generatorConfig, store);

    generatorResolved.dse.systemPnr.search.routing.negotiation =
        loom::ResolvedDualSubgradientPolicy{
            loom::ResolvedDualDirectionKernel::ProjectedSigned,
            std::nullopt,
            {loom::ResolvedDualStepScheduleKind::Constant, 1, 0, 0, 0}};
    const auto wrongDomainConfig =
        take(loom::pnr::projectResolvedSystemPnrConfigView(generatorResolved));
    const loom::pnr::SpatialPnrGenerationInputs wrongDomainInputs{
        dataflow,          tech.view(),        fabric.view(), physicalTiming,
        wrongDomainConfig, constraints.view(), store};
    const auto wrongDomainOutcome =
        loom::pnr::generateSpatialMappings(wrongDomainInputs);
    const auto *invalid = std::get_if<loom::pnr::InvalidSpatialPnrGeneration>(
        &wrongDomainOutcome);
    if (!invalid ||
        invalid->reason !=
            loom::pnr::InvalidSpatialPnrGenerationReason::FrozenInput ||
        invalid->accounting.seedAttemptSlots != 0 ||
        invalid->accounting.publicationSlots != 0)
      fail("Spatial PnR generator checked provider support before exact input "
           "coupling");
  }
  loom::ResolvedConfig pnrResolved =
      loom::test::buildSpatialPnrTestResolvedConfig();
  pnrResolved.dse.spatialPnr.search.exactRepair = {
      loom::ResolvedPnrExactRepairKind::CpSat, 256, 1024};
  const auto pnrConfig =
      take(loom::pnr::projectResolvedSpatialPnrConfigView(pnrResolved));
  auto problem = take(loom::pnr::freezeSpatialPnrProblem(
      dataflow, tech.view(), fabric.view(), pnrConfig, constraints.view()));
  if (problem->routing().tagContinuity().traversalPointOrdinals().size() !=
      problem->routing().traversals().size())
    fail("Spatial freeze omitted its traversal-dense tag-continuity index");
  loom::pnr::SpatialCandidateStateHandle candidate;
  if (temporal) {
    candidate = take(loom::pnr::createCanonicalSpatialCandidate(problem));
    for (loom::pnr::PnrIndex net = 0;
         net < problem->transfers().logicalNets().size(); ++net) {
      const auto unrouted = take(
          loom::pnr::deriveSpatialTagContinuity(candidate->routeTree(net)));
      if (!unrouted.segments().empty() || !unrouted.segmentDomains().empty() ||
          !unrouted.domainSegments().empty() ||
          !llvm::equal(unrouted.segmentDomainOffsets(),
                       std::array<loom::pnr::PnrIndex, 1>{0}) ||
          unrouted.domainSegmentOffsets().size() !=
              problem->routing().tagContinuity().matchDomains().size() + 1 ||
          llvm::any_of(unrouted.domainSegmentOffsets(),
                       [](loom::pnr::PnrIndex value) { return value != 0; }))
        fail("unrouted tag continuity did not produce a complete empty CSR");
    }
    loom::pnr::SpatialCandidateScratch candidateScratch;
    requireSuccess(candidateScratch.prepare(*problem));
    selectLegalTemporalBinding(
        *candidate, candidateScratch,
        boundaryWrapped || forceTagConflict || requireSeparatedSwitchRows,
        switchPackingFabric &&
            (!requireSeparatedSwitchRows || requireContendingSwitchRows));
    if (!boundaryWrapped && !forceTagConflict && !switchPackingFabric)
      operandPairingPressureIsIncremental(*candidate, candidateScratch);
    if (forceTagConflict || exceedsSwitchResidentCapacity) {
      auto costs = take(loom::pnr::SpatialRouteCostState::create(*candidate));
      loom::pnr::SpatialNetRouterScratch router;
      requireSuccess(router.prepare(*problem));
      std::vector<loom::pnr::PnrIndex> logicalNets(
          problem->transfers().logicalNets().size());
      std::iota(logicalNets.begin(), logicalNets.end(), 0);
      requireSuccess(router.beginConstraintSweep(logicalNets));
      auto move = take(candidate->beginMove(candidateScratch));
      for (loom::pnr::PnrIndex logicalNet : logicalNets) {
        requireSuccess(costs.selectLogicalNet(logicalNet));
        take(router.routeWholeNet(
            move, *candidate, costs, logicalNet,
            pnrConfig.policy().search.routing.endpointExpansionLimit));
        requireSuccess(costs.acceptSelectedLogicalNet());
        requireSuccess(router.finishConstraintNet(logicalNet));
      }
      if (!take(move.close()))
        fail("tag-pressure fixture closed a selected handshake cycle");
      requireSuccess(move.commit());
      requireSuccess(costs.resetFromCandidate());
      if (!costs.hasTagPressureViolation())
        fail("PathFinder cost state ignored committed tag pressure");
      if (exceedsSwitchResidentCapacity) {
        if (candidate->tagResidentCapacityOveruse() == 0)
          fail("Temporal switch fixture did not exceed resident capacity");
        requireSuccess(candidate->verify());
        return;
      }
      if (candidate->tagConflictCount() == 0)
        fail("conflicting tag fixture did not construct a collision");
      if (switchPackingFabric) {
        verifyTagDomainIncidence(*candidate, costs);
        std::optional<loom::pnr::PnrIndex> sharedConflictNet;
        for (loom::pnr::PnrIndex logicalNet : logicalNets)
          for (const auto &use : costs.logicalNetTagDomainUses(logicalNet))
            if (use.marginalResidentCount == 0 &&
                costs.tagDomainConflictCount(use.domain) != 0)
              sharedConflictNet = logicalNet;
        if (!sharedConflictNet ||
            !costs.logicalNetHasTagPressure(*sharedConflictNet))
          fail("shared conflicting row lost its participating logical net");
        loom::pnr::SpatialPathFinderRouterScratch regionalRouter;
        requireSuccess(regionalRouter.prepare(*problem));
        auto regionalMove = take(candidate->beginMove(candidateScratch));
        auto regional = regionalRouter.routeToClosureInMove(
            regionalMove, *candidate, costs,
            {pnrConfig.policy().search.routing.endpointExpansionLimit, 2, 1, 1},
            {&*sharedConflictNet, 1}, {},
            loom::pnr::SpatialRoutingClosureRequirement::ExactRegional, 1);
        if (regional)
          fail("one-net regional closure ignored its shared row conflict");
        bool expandedPastLimit = false;
        llvm::handleAllErrors(
            regional.takeError(),
            [&](const loom::pnr::SpatialPathFinderClosureFailure &failure) {
              expandedPastLimit = failure.kind() ==
                  loom::pnr::SpatialPathFinderClosureFailure::Kind::RegionalLimit;
            });
        if (!expandedPastLimit)
          fail("shared row did not expand the exact regional conflict closure");
        regionalMove.rollback();
        requireSuccess(costs.resetFromCandidate());
      }
      const std::vector<loom::pnr::RouteCost> baselineTagCosts(
          costs.currentArcCosts().begin(), costs.currentArcCosts().end());
      requireSuccess(costs.advancePathFinderIteration());
      if (!llvm::any_of(
              llvm::zip_equal(baselineTagCosts, costs.currentArcCosts()),
              [](const auto &entry) {
                return std::get<1>(entry) > std::get<0>(entry);
              }))
        fail("PathFinder did not raise cost around a conflicting tag domain");
      requireSuccess(costs.resetFromCandidate());
      if (!llvm::equal(baselineTagCosts, costs.currentArcCosts()))
        fail("PathFinder reset retained tag-domain history pressure");
    }
  } else {
    loom::pnr::SpatialPathFinderSeedWorkSummary firstWork;
    loom::pnr::SpatialPathFinderSeedWorkSummary secondWork;
    auto first = take(
        loom::pnr::createCanonicalPathFinderSpatialSeed(problem, firstWork));
    auto second = take(
        loom::pnr::createCanonicalPathFinderSpatialSeed(problem, secondWork));
    if (firstWork.negotiationIterations != secondWork.negotiationIterations ||
        firstWork.endpointExpansions != secondWork.endpointExpansions ||
        first.candidate->unroutedObligationCount() != 0 ||
        second.candidate->unroutedObligationCount() != 0)
      fail("canonical Spatial routing seed is not closed and deterministic");
    for (loom::pnr::PnrIndex net = 0;
         net < problem->transfers().logicalNets().size(); ++net) {
      const auto &firstTree = first.candidate->routeTree(net);
      const auto &secondTree = second.candidate->routeTree(net);
      if (!firstTree.isRouted() || !secondTree.isRouted() ||
          firstTree.sourceEndpoint() != secondTree.sourceEndpoint() ||
          !llvm::equal(firstTree.nodeStorage(), secondTree.nodeStorage()))
        fail("canonical Spatial routing seed changed its RouteTree");
      for (loom::pnr::PnrIndex sink = 0;
           sink < problem->transfers().logicalNets()[net].sinkCount; ++sink)
        if (firstTree.sinkEndpoint(sink) != secondTree.sinkEndpoint(sink))
          fail("canonical Spatial routing seed changed a sink attachment");
    }
    requireSuccess(second.candidate->verify());
    candidate = std::move(first.candidate);
  }
  loom::pnr::SpatialGlobalRoutingClosureScratch globalRoutingClosure;
  std::optional<loom::pnr::HandshakeActiveDemandStatistics>
      handshakeAfterClosure;
  std::optional<loom::pnr::SpatialProgressStatistics> progressAfterClosure;
  if (forceTagConflict) {
    const std::uint64_t selectedTraversalClaim =
        candidate->totalSelectedTraversalClaim();
    const std::uint64_t unroutedObligations =
        candidate->unroutedObligationCount();
    const std::uint64_t tagConflicts = candidate->tagConflictCount();
    llvm::Error rejectedClosure = globalRoutingClosure.run(*candidate);
    if (!rejectedClosure)
      fail("global routing closure accepted conflicting Physical Tags");
    bool observedBoundedFailure = false;
    llvm::handleAllErrors(
        std::move(rejectedClosure),
        [&](const loom::pnr::SpatialActionTransitionFailure &failure) {
          observedBoundedFailure =
              failure.kind() ==
              loom::pnr::SpatialActionTransitionFailureKind::WorkLimit;
        });
    if (!observedBoundedFailure)
      fail("global routing closure lost its bounded tag-conflict failure");
    if (candidate->totalSelectedTraversalClaim() != selectedTraversalClaim ||
        candidate->unroutedObligationCount() != unroutedObligations ||
        candidate->tagConflictCount() != tagConflicts)
      fail("rejected global routing closure changed the candidate");
  } else {
    const auto handshakeBeforeClosure =
        candidate->handshake().materializationStatistics();
    const auto progressBeforeClosure = candidate->progress().statistics();
    requireSuccess(globalRoutingClosure.run(*candidate));
    const std::size_t retainedClosureBytes =
        globalRoutingClosure.retainedStorageBytes();
    requireSuccess(globalRoutingClosure.run(*candidate));
    const std::size_t repeatedClosureBytes =
        globalRoutingClosure.retainedStorageBytes();
    if (repeatedClosureBytes > retainedClosureBytes)
      fail("warmed global routing closure grew worker-local storage from " +
           std::to_string(retainedClosureBytes) + " to " +
           std::to_string(repeatedClosureBytes));
    handshakeAfterClosure = candidate->handshake().materializationStatistics();
    progressAfterClosure = candidate->progress().statistics();
    if (handshakeAfterClosure->cachedVerificationCount <=
            handshakeBeforeClosure.cachedVerificationCount ||
        progressAfterClosure->cachedVerificationCount <=
            progressBeforeClosure.cachedVerificationCount)
      fail("global routing closure did not verify incremental candidate state");
    if (handshakeAfterClosure->coldVerificationConstructionCount !=
            handshakeBeforeClosure.coldVerificationConstructionCount ||
        progressAfterClosure->coldVerificationCount !=
            progressBeforeClosure.coldVerificationCount ||
        progressAfterClosure->coldProgressScanCount !=
            progressBeforeClosure.coldProgressScanCount)
      fail("global routing closure invoked a publication verifier");
  }
  requireSuccess(candidate->verify());
  if (handshakeAfterClosure &&
      (candidate->handshake()
               .materializationStatistics()
               .coldVerificationConstructionCount !=
           handshakeAfterClosure->coldVerificationConstructionCount + 1 ||
       candidate->progress().statistics().coldVerificationCount !=
           progressAfterClosure->coldVerificationCount + 1 ||
       candidate->progress().statistics().coldProgressScanCount !=
           progressAfterClosure->coldProgressScanCount + 1))
    fail("final candidate verification omitted an independent reconstruction");

  std::vector<const loom::pnr::RouteTreeState *> selectedRoutes;
  selectedRoutes.reserve(problem->transfers().logicalNets().size());
  for (loom::pnr::PnrIndex net = 0;
       net < problem->transfers().logicalNets().size(); ++net)
    selectedRoutes.push_back(&candidate->routeTree(net));
  const auto tagAssignments =
      take(loom::pnr::deriveCanonicalSpatialTagAssignments(*problem,
                                                           selectedRoutes));
  const auto repeatedTagAssignments =
      take(loom::pnr::deriveCanonicalSpatialTagAssignments(*problem,
                                                           selectedRoutes));
  const std::uint64_t expectedTagConflicts = tagAssignments.conflictCount();
  if (tagAssignments.segments() != repeatedTagAssignments.segments() ||
      tagAssignments.values() != repeatedTagAssignments.values() ||
      tagAssignments.segmentDomains() !=
          repeatedTagAssignments.segmentDomains() ||
      tagAssignments.unassignedCount() != 0 ||
      (forceTagConflict ? expectedTagConflicts == 0
                        : expectedTagConflicts != 0))
    fail("canonical Physical Tag assignment is incomplete or unstable");
  if (candidate->tagUnassignedCount() != 0 ||
      candidate->tagConflictCount() != expectedTagConflicts)
    fail("Spatial Candidate lost its Physical Tag assignment violations");
  if (take(loom::pnr::spatialMappingViolationValue(
          *candidate, loom::ResolvedPnrViolationKind::TagUnassigned)) != 0 ||
      take(loom::pnr::spatialMappingViolationValue(
          *candidate, loom::ResolvedPnrViolationKind::TagConflict)) !=
          expectedTagConflicts)
    fail("Spatial objective did not project Candidate-owned Tag violations");
  for (auto [segment, value] :
       llvm::zip_equal(tagAssignments.segments(), tagAssignments.values())) {
    if (!value)
      fail("routed Physical Tag segment has no canonical value");
    const auto encoded =
        take(fabric::encodePhysicalTagValue(segment.tagWidthBits, *value));
    if (take(fabric::decodePhysicalTagValue(segment.tagWidthBits, encoded)) !=
        value->zextOrTrunc(segment.tagWidthBits))
      fail("candidate Physical Tag disagrees with the Fabric owner codec");
  }

  bool observedSharedLocalDomain = false;
  bool observedPackedSwitchDomain = false;
  bool observedSeparatedSwitchDomain = false;
  std::size_t maximumLocalDomainOccupancy = 0;
  const auto matchDomains = problem->routing().tagContinuity().matchDomains();
  for (loom::pnr::PnrIndex domain = 0; domain < matchDomains.size(); ++domain) {
    const auto members = tagAssignments.domainSegments(domain);
    maximumLocalDomainOccupancy =
        std::max(maximumLocalDomainOccupancy, members.size());
    observedSharedLocalDomain |= members.size() > 1;
    std::size_t distinctValues = 0;
    for (std::size_t position = 0; position != members.size(); ++position) {
      const loom::pnr::PnrIndex member = members[position];
      if (llvm::none_of(members.take_front(position),
                        [&](loom::pnr::PnrIndex prior) {
                          return tagAssignments.values()[prior] ==
                                 tagAssignments.values()[member];
                        }))
        ++distinctValues;
    }
    const bool temporalSwitch =
        matchDomains[domain].kind ==
        loom::fabric::FabricPhysicalTagMatchDomainKind::TemporalSwitchTable;
    if (temporalSwitch) {
      if (candidate->tagDomainResidentCount(domain) != distinctValues ||
          candidate->tagDomainResidentCapacityOveruse(domain) != 0)
        fail("Temporal switch residency did not count packed tag rows");
      observedPackedSwitchDomain |= members.size() > distinctValues;
      observedSeparatedSwitchDomain |= distinctValues > 1;
      continue;
    }
    for (auto [position, lhs] : llvm::enumerate(members))
      for (loom::pnr::PnrIndex rhs : llvm::drop_begin(members, position + 1))
        if (!forceTagConflict &&
            tagAssignments.values()[lhs] == tagAssignments.values()[rhs])
          fail("one local Physical Tag match domain contains a collision");
  }
  if (switchPackingFabric && !forceTagConflict && !requireSeparatedSwitchRows &&
      !observedPackedSwitchDomain)
    fail("Temporal route fixture did not pack compatible switch segments");
  if (requireSeparatedSwitchRows && !observedSeparatedSwitchDomain)
    fail("Temporal route fixture did not select distinct resident rows");
  if (switchPackingFabric && forceTagConflict) {
    const bool observedSwitchConflict = llvm::any_of(
        llvm::seq<loom::pnr::PnrIndex>(0, matchDomains.size()),
        [&](loom::pnr::PnrIndex domain) {
          return matchDomains[domain].kind ==
                     loom::fabric::FabricPhysicalTagMatchDomainKind::
                         TemporalSwitchTable &&
                 candidate->tagDomainConflictCount(domain) != 0;
        });
    if (!observedSwitchConflict)
      fail("incompatible Temporal switch signatures did not conflict");
  }
  if (switchPackingFabric && !forceTagConflict && !requireSeparatedSwitchRows) {
    std::optional<loom::pnr::PnrIndex> changedSegment;
    std::optional<loom::pnr::PnrIndex> changedDomain;
    for (loom::pnr::PnrIndex domain = 0; domain < matchDomains.size();
         ++domain) {
      const auto members = tagAssignments.domainSegments(domain);
      if (matchDomains[domain].kind !=
              loom::fabric::FabricPhysicalTagMatchDomainKind::
                  TemporalSwitchTable ||
          members.size() < 2)
        continue;
      for (std::size_t position = 1; position < members.size(); ++position)
        if (tagAssignments.values()[members[position]] ==
            tagAssignments.values()[members.front()]) {
          changedSegment = members[position];
          changedDomain = domain;
          break;
        }
      if (changedSegment)
        break;
    }
    if (!changedSegment || !changedDomain)
      fail("switch projection fixture has no packed resident-row pair");
    const auto netOffsets = tagAssignments.netSegmentOffsets();
    const auto netEnd = llvm::upper_bound(netOffsets, *changedSegment);
    if (netEnd == netOffsets.begin())
      fail("packed switch segment has no logical-net owner");
    const loom::pnr::PnrIndex logicalNet =
        static_cast<loom::pnr::PnrIndex>(netEnd - netOffsets.begin() - 1);
    const loom::pnr::PnrIndex localSegment =
        *changedSegment - netOffsets[logicalNet];
    const auto original = candidate->tagValues(logicalNet)[localSegment];
    if (!original || original->getBitWidth() > 8)
      fail("switch projection fixture has no bounded exact tag value");
    std::optional<llvm::APInt> replacement;
    const auto domainMembers = tagAssignments.domainSegments(*changedDomain);
    for (std::uint64_t value = 0;
         value != (std::uint64_t{1} << original->getBitWidth()); ++value) {
      const llvm::APInt candidateValue(original->getBitWidth(), value);
      if (llvm::none_of(domainMembers, [&](loom::pnr::PnrIndex member) {
            const auto &assigned = tagAssignments.values()[member];
            return assigned && *assigned == candidateValue;
          })) {
        replacement = candidateValue;
        break;
      }
    }
    if (!replacement)
      fail("switch projection fixture exhausted its tag domain");

    loom::pnr::SpatialCandidateScratch switchScratch;
    requireSuccess(switchScratch.prepare(*problem));
    auto tagCosts = take(loom::pnr::SpatialRouteCostState::create(*candidate));
    verifyTagDomainIncidence(*candidate, tagCosts);
    const auto applyTag = [&](const llvm::APInt &value) {
      auto move = take(candidate->beginMove(switchScratch));
      requireSuccess(move.setPhysicalTagValue(logicalNet, localSegment, value));
      const auto projection = take(move.projectCurrentRoutes());
      if (take(move.close()) != projection.selectedHandshakeAcyclic)
        fail("incremental switch regrouping disagreed with its route "
             "projection");
      const auto delta = take(move.summarizeCurrentTagAssignmentDelta());
      requireSuccess(tagCosts.synchronizeTagProjection(delta, {}));
      verifyTagDomainIncidence(*candidate, tagCosts);
      requireSuccess(move.commit());
      requireSuccess(tagCosts.commitTagProjectionDelta());
      const auto coldCosts =
          take(loom::pnr::SpatialRouteCostState::create(*candidate));
      for (loom::pnr::PnrIndex net = 0;
           net < problem->transfers().logicalNets().size(); ++net)
        if (!llvm::equal(tagCosts.logicalNetTagDomainUses(net),
                         coldCosts.logicalNetTagDomainUses(net)))
          fail("incremental row regrouping changed domain incidence or cost");
      requireSuccess(candidate->verify());
    };
    applyTag(*replacement);
    applyTag(*original);
  }
  if (boundaryWrapped && !observedSharedLocalDomain)
    fail(
        ("Temporal route fixture did not exercise local tag interference: " +
         llvm::Twine(tagAssignments.segments().size()) + " segments across " +
         llvm::Twine(problem->routing().tagContinuity().matchDomains().size()) +
         " domains, maximum local occupancy " +
         llvm::Twine(maximumLocalDomainOccupancy))
            .str());

  bool observedDisjointDomainReuse = false;
  const auto segmentDomainOffsets = tagAssignments.segmentDomainOffsets();
  const auto segmentDomains = tagAssignments.segmentDomains();
  for (loom::pnr::PnrIndex lhs = 0; lhs < tagAssignments.segments().size();
       ++lhs)
    for (loom::pnr::PnrIndex rhs = lhs + 1;
         rhs < tagAssignments.segments().size(); ++rhs) {
      if (tagAssignments.values()[lhs] != tagAssignments.values()[rhs])
        continue;
      const auto lhsDomains = segmentDomains.slice(
          segmentDomainOffsets[lhs],
          segmentDomainOffsets[lhs + 1] - segmentDomainOffsets[lhs]);
      const auto rhsDomains = segmentDomains.slice(
          segmentDomainOffsets[rhs],
          segmentDomainOffsets[rhs + 1] - segmentDomainOffsets[rhs]);
      observedDisjointDomainReuse |=
          llvm::none_of(lhsDomains, [&](loom::pnr::PnrIndex domain) {
            return llvm::is_contained(rhsDomains, domain);
          });
    }
  if (boundaryWrapped && !observedDisjointDomainReuse)
    fail("Physical Tags were made globally unique across disjoint domains");

  if (boundaryWrapped) {
    const std::vector<loom::pnr::SpatialTagContinuitySegment> originalSegments(
        candidate->tagSegments(0).begin(), candidate->tagSegments(0).end());
    const std::vector<std::optional<llvm::APInt>> originalValues(
        candidate->tagValues(0).begin(), candidate->tagValues(0).end());
    if (originalSegments.empty() ||
        originalSegments.size() != originalValues.size())
      fail("routed Spatial Candidate has no Physical Tag decisions");

    loom::pnr::SpatialCandidateScratch tagScratch;
    requireSuccess(tagScratch.prepare(*problem));
    auto tagCosts = take(loom::pnr::SpatialRouteCostState::create(*candidate));
    const std::vector<loom::pnr::RouteCost> originalTagCosts(
        tagCosts.currentArcCosts().begin(), tagCosts.currentArcCosts().end());
    const bool originalTagPressure = tagCosts.hasTagPressureViolation();
    auto move = take(candidate->beginMove(tagScratch));
    requireSuccess(move.ripUpWholeRoute(0));
    const auto provisional = take(move.projectCurrentRoutes());
    if (candidate->unroutedObligationCount() != 0 ||
        provisional.unroutedObligationCount !=
            problem->transfers().logicalNets()[0].sinkCount)
      fail("active Spatial move did not project its provisional RouteTree");
    const bool closed = take(move.close());
    if (closed != provisional.selectedHandshakeAcyclic)
      fail("provisional tag-aware handshake projection disagrees with move "
           "closure");
    const auto projectionStatistics =
        tagScratch.handshakeProjectionStatistics();
    if (projectionStatistics.projectionCount != 1 ||
        projectionStatistics.peakActiveNodeCount == 0 ||
        projectionStatistics.peakActiveArcCount == 0)
      fail("provisional handshake projection omitted its worker-local "
           "construction statistics");
    if (!closed)
      fail("Physical Tag rollback fixture closed a handshake cycle");
    const auto tagDelta = take(move.summarizeCurrentTagAssignmentDelta());
    const loom::pnr::PnrIndex rippedNet = 0;
    requireSuccess(tagCosts.synchronizeTagProjection(
        tagDelta, llvm::ArrayRef<loom::pnr::PnrIndex>(&rippedNet, 1)));
    if (!tagCosts.hasActiveTagProjectionDelta())
      fail("Physical Tag route-cost delta did not retain its inverse");
    if (!candidate->tagSegments(0).empty() || !candidate->tagValues(0).empty())
      fail("route rip-up retained stale Physical Tag decisions");
    move.rollback();
    requireSuccess(tagCosts.rollbackTagProjectionDelta());
    if (tagCosts.hasActiveTagProjectionDelta() ||
        tagCosts.hasTagPressureViolation() != originalTagPressure ||
        !llvm::equal(tagCosts.currentArcCosts(), originalTagCosts))
      fail("Physical Tag route-cost rollback changed its exact projection");
    if (!llvm::equal(candidate->tagSegments(0), originalSegments) ||
        !llvm::equal(candidate->tagValues(0), originalValues) ||
        candidate->tagUnassignedCount() != 0 ||
        candidate->tagConflictCount() != expectedTagConflicts)
      fail("Spatial move rollback changed Physical Tag decisions");
    requireSuccess(candidate->verify());
  }

  if (requireContendingSwitchRows) {
    // Select a routed net through the frozen switch-traversal relation, then
    // compare each incremental transition with an independent cold projection.
    std::optional<loom::pnr::PnrIndex> selectedLogicalNet;
    const auto switchSelections =
        problem->handshake().switchTraversalSelections();
    for (loom::pnr::PnrIndex logicalNet = 0;
         logicalNet < problem->transfers().logicalNets().size(); ++logicalNet) {
      const auto &tree = candidate->routeTree(logicalNet);
      for (const auto &node : tree.nodeStorage()) {
        if (node.parentArc == loom::pnr::getInvalidPnrIndex())
          continue;
        const loom::pnr::PnrIndex traversal =
            problem->routing().routingArcs()[node.parentArc].traversal;
        if (llvm::any_of(switchSelections, [&](const auto &selection) {
              return selection.traversal == traversal;
            })) {
          selectedLogicalNet = logicalNet;
          break;
        }
      }
      if (selectedLogicalNet)
        break;
    }
    if (!selectedLogicalNet)
      fail("switch route fixture has no selected switch traversal");

    struct SavedRoute final {
      loom::pnr::PnrIndex source = 0;
      std::vector<loom::pnr::PnrIndex> sinkEndpoints;
      std::vector<std::vector<loom::pnr::PnrIndex>> sinkChains;
    };
    const auto &logicalNetSpec =
        problem->transfers().logicalNets()[*selectedLogicalNet];
    const auto &tree = candidate->routeTree(*selectedLogicalNet);
    const auto source = tree.sourceEndpoint();
    if (!source)
      fail("switch transaction fixture requires a routed net");
    SavedRoute saved{*source,
                     std::vector<loom::pnr::PnrIndex>(logicalNetSpec.sinkCount),
                     std::vector<std::vector<loom::pnr::PnrIndex>>(
                         logicalNetSpec.sinkCount)};
    for (loom::pnr::PnrIndex sink = 0; sink < logicalNetSpec.sinkCount;
         ++sink) {
      const auto sinkEndpoint = tree.sinkEndpoint(sink);
      if (!sinkEndpoint)
        fail("switch transaction fixture requires a routed sink");
      saved.sinkEndpoints[sink] = *sinkEndpoint;
      std::vector<loom::pnr::PnrIndex> reverseArcs;
      loom::pnr::PnrIndex endpoint = *sinkEndpoint;
      while (true) {
        const auto slot = tree.findNode(endpoint);
        if (!slot)
          fail("switch transaction fixture lost a route node");
        const auto &node = tree.nodeStorage()[*slot];
        if (node.parentArc == loom::pnr::getInvalidPnrIndex())
          break;
        reverseArcs.push_back(node.parentArc);
        endpoint = problem->routing().arcSources()[node.parentArc];
      }
      if (endpoint != saved.source)
        fail("switch transaction fixture sink does not descend from its "
             "source");
      saved.sinkChains[sink].assign(reverseArcs.rbegin(), reverseArcs.rend());
    }
    std::vector<std::uint8_t> attached(
        problem->routing().routingEndpoints().size(), 0);
    const auto restoreRoute =
        [&](loom::pnr::SpatialMoveTransaction &move) -> llvm::Error {
      if (llvm::Error error =
              move.bindRouteSource(*selectedLogicalNet, saved.source))
        return error;
      std::fill(attached.begin(), attached.end(), 0);
      attached[saved.source] = 1;
      for (loom::pnr::PnrIndex sink = 0; sink < saved.sinkEndpoints.size();
           ++sink) {
        if (llvm::Error error = move.bindRouteSink(*selectedLogicalNet, sink,
                                                   saved.sinkEndpoints[sink]))
          return error;
        const auto &chain = saved.sinkChains[sink];
        loom::pnr::PnrIndex attachPoint = saved.source;
        std::size_t attachedPrefix = 0;
        for (std::size_t arc = 0; arc < chain.size(); ++arc) {
          const loom::pnr::PnrIndex target =
              problem->routing().routingArcs()[chain[arc]].target;
          if (!attached[target])
            break;
          attachPoint = target;
          attachedPrefix = arc + 1;
        }
        if (llvm::Error error = move.attachRoutePath(
                *selectedLogicalNet, attachPoint,
                llvm::ArrayRef(chain).drop_front(attachedPrefix), sink))
          return error;
        for (std::size_t arc = attachedPrefix; arc < chain.size(); ++arc)
          attached[problem->routing().routingArcs()[chain[arc]].target] = 1;
      }
      return llvm::Error::success();
    };
    loom::pnr::SpatialCandidateScratch transitionScratch;
    requireSuccess(transitionScratch.prepare(*problem));
    auto removal = take(candidate->beginMove(transitionScratch));
    requireSuccess(removal.ripUpWholeRoute(*selectedLogicalNet));
    const auto removedProjection = take(removal.projectCurrentRoutes());
    const bool removalAcyclic = take(removal.close());
    requireSuccess(removal.commit());
    if (removalAcyclic != removedProjection.selectedHandshakeAcyclic)
      fail("incremental switch removal disagreed with its cold route "
           "projection");
    requireSuccess(candidate->verify());

    auto restoration = take(candidate->beginMove(transitionScratch));
    requireSuccess(restoreRoute(restoration));
    const auto restoredProjection = take(restoration.projectCurrentRoutes());
    const bool restorationAcyclic = take(restoration.close());
    requireSuccess(restoration.commit());
    if (restorationAcyclic != restoredProjection.selectedHandshakeAcyclic)
      fail("incremental switch restoration disagreed with its cold route "
           "projection");
    requireSuccess(candidate->verify());

    auto noopMove = take(candidate->beginMove(transitionScratch));
    requireSuccess(noopMove.ripUpWholeRoute(*selectedLogicalNet));
    requireSuccess(restoreRoute(noopMove));
    if (!take(noopMove.close()))
      fail("switch no-op replay closed a handshake cycle");
    if (noopMove.hasSemanticChange())
      fail("identical RouteTree replay reported a semantic change");
    requireSuccess(noopMove.commit());
    requireSuccess(candidate->verify());
  }

  if (boundaryWrapped) {
    bool observedBoundaryOrigin = false;
    bool observedRouteSourceOrigin = false;
    bool observedUntaggedNode = false;
    bool observedRemoverStop = false;
    bool observedTagMatchDomain = false;
    for (loom::pnr::PnrIndex net = 0;
         net < problem->transfers().logicalNets().size(); ++net) {
      const auto projection = take(
          loom::pnr::deriveSpatialTagContinuity(candidate->routeTree(net)));
      const auto repeated = take(
          loom::pnr::deriveSpatialTagContinuity(candidate->routeTree(net)));
      if (!llvm::equal(projection.segments(), repeated.segments()) ||
          !llvm::equal(projection.nodeSegments(), repeated.nodeSegments()) ||
          !llvm::equal(projection.segmentDomainOffsets(),
                       repeated.segmentDomainOffsets()) ||
          !llvm::equal(projection.segmentDomains(),
                       repeated.segmentDomains()) ||
          !llvm::equal(projection.domainSegmentOffsets(),
                       repeated.domainSegmentOffsets()) ||
          !llvm::equal(projection.domainSegments(), repeated.domainSegments()))
        fail("tag-continuity projection is not deterministic");
      const auto &tree = candidate->routeTree(net);
      if (projection.nodeSegments().size() != tree.nodeStorage().size())
        fail("tag-continuity projection is not RouteTree-slot dense");
      for (const auto &segment : projection.segments()) {
        observedBoundaryOrigin |=
            segment.originKind ==
            loom::pnr::SpatialTagContinuityOriginKind::BoundaryPoint;
        observedRouteSourceOrigin |=
            segment.originKind ==
            loom::pnr::SpatialTagContinuityOriginKind::RouteSource;
      }
      for (auto [slot, node] : llvm::enumerate(tree.nodeStorage())) {
        if (!node.isActive())
          continue;
        const auto &dataPath =
            problem->routing().routingEndpoints()[node.endpoint].dataPath;
        const bool tagged = dataPath.kind == ::fabric::DataPathKind::BitsTag;
        if (!tagged)
          observedUntaggedNode = true;
        if ((projection.nodeSegments()[slot] !=
             loom::pnr::getInvalidPnrIndex()) != tagged)
          fail("tag-continuity segment disagrees with route endpoint kind");
        if (node.parentArc == loom::pnr::getInvalidPnrIndex())
          continue;
        const auto &arc = problem->routing().routingArcs()[node.parentArc];
        const auto point = problem->routing()
                               .tagContinuity()
                               .traversalPointOrdinals()[arc.traversal];
        if (point != loom::pnr::getInvalidPnrIndex() &&
            problem->routing().tagContinuity().points()[point].kind ==
                loom::fabric::FabricBoundaryTagContinuityKind::Remover &&
            projection.nodeSegments()[slot] == loom::pnr::getInvalidPnrIndex())
          observedRemoverStop = true;
      }

      const auto segmentDomainOffsets = projection.segmentDomainOffsets();
      const auto segmentDomains = projection.segmentDomains();
      const auto domainSegmentOffsets = projection.domainSegmentOffsets();
      const auto domainSegments = projection.domainSegments();
      const auto matchDomains =
          problem->routing().tagContinuity().matchDomains();
      if (segmentDomainOffsets.size() != projection.segments().size() + 1 ||
          domainSegmentOffsets.size() != matchDomains.size() + 1 ||
          segmentDomainOffsets.back() != segmentDomains.size() ||
          domainSegmentOffsets.back() != domainSegments.size())
        fail("tag match-domain incidence has malformed CSR bounds");
      for (loom::pnr::PnrIndex segment = 0;
           segment < projection.segments().size(); ++segment) {
        for (loom::pnr::PnrIndex incidence = segmentDomainOffsets[segment];
             incidence < segmentDomainOffsets[segment + 1]; ++incidence) {
          const auto domain = segmentDomains[incidence];
          if (domain >= matchDomains.size() ||
              matchDomains[domain].tagWidthBits !=
                  projection.segments()[segment].tagWidthBits)
            fail("tag segment intersects an incompatible match domain");
          const auto reverse = domainSegments.slice(
              domainSegmentOffsets[domain],
              domainSegmentOffsets[domain + 1] - domainSegmentOffsets[domain]);
          if (!llvm::is_contained(reverse, segment))
            fail("tag segment/domain reverse incidence is incomplete");
          observedTagMatchDomain = true;
        }
      }
    }
    if (!observedBoundaryOrigin || !observedRouteSourceOrigin ||
        !observedUntaggedNode || !observedRemoverStop ||
        !observedTagMatchDomain)
      fail("boundary Temporal route did not exercise start and stop semantics");
  }

  if (forceTagConflict)
    return;

  const auto placementRejectingConstraints =
      loom::test::buildSpatialMappingConstraints(
          context, dataflow, tech.view(), fabric.view(), store,
          /*restrictTagsToZero=*/false, /*rejectComputePlacement=*/true);
  const std::size_t storedBeforeRejection = storedObjectCount(directory.path());
  auto rejectedFinalization = loom::pnr::finalizeSpatialMappingCandidate(
      *candidate, dataflow, tech.view(), fabric.view(),
      placementRejectingConstraints.view(), store);
  if (rejectedFinalization)
    fail("constraint-rejected Spatial candidate was published");
  bool rejectedByConstraintSet = false;
  llvm::handleAllErrors(
      rejectedFinalization.takeError(),
      [&](const loom::mapping::SpatialMappingConstraintRejection &) {
        rejectedByConstraintSet = true;
      },
      [&](const llvm::ErrorInfoBase &error) { fail(error.message()); });
  if (!rejectedByConstraintSet)
    fail("Spatial candidate finalization lost typed constraint rejection");
  if (storedObjectCount(directory.path()) != storedBeforeRejection)
    fail("constraint-rejected Spatial candidate reached ArtifactStore");

  auto finalized = take(loom::pnr::finalizeSpatialMappingCandidate(
      *candidate, dataflow, tech.view(), fabric.view(), constraints.view(),
      store));
  auto imported =
      take(loom::mapping::importSpatialMapping(finalized.reference(), store));
  if (imported.reference() != finalized.reference() ||
      imported.view().computeBindings().size() != 1 ||
      imported.view().routeTrees().empty() ||
      imported.view().resourceUses().empty())
    fail("strict SpatialMapping round trip lost selected closure");
  if (!temporal &&
      contractKind == ComputeContractKind::OneCycleElastic) {
    const auto physicalTiming =
        take(loom::fabric::projectNormalizedFabricPhysicalTimingProfile(
            fabric.view()));
    loom::test::exerciseSpatialRuntimeCounterexampleExactRepair(
        dataflow, tech.view(), fabric.view(), constraints, imported,
        physicalTiming, pnrConfig, store);
  }
  bool observedCompleteResultTuple = false;
  bool sawComputeTransition = false;
  bool allComputeIntrinsic = true;
  for (const auto &use : imported.view().resourceUses()) {
    if (!std::holds_alternative<loom::mapping::SpatialComputeResourceOwnerRef>(
            use.owner) ||
        !std::holds_alternative<loom::mapping::SpatialActorTransitionEventRef>(
            use.activation.trigger.event))
      continue;
    sawComputeTransition = true;
    allComputeIntrinsic &= use.activation.release.empty();
    if (use.activation.release.size() != 2)
      continue;
    const auto &transition =
        std::get<loom::mapping::SpatialActorTransitionEventRef>(
            use.activation.trigger.event);
    bool firstResult = false;
    bool secondResult = false;
    for (const auto &point : use.activation.release) {
      const auto *producer =
          std::get_if<dataflow::CanonicalGraphProducerEndpointRef>(
              &point.event);
      if (!producer)
        continue;
      const auto *result = std::get_if<dataflow::ActorTokenResultRef>(producer);
      if (!result || result->actor != transition.actor)
        continue;
      firstResult |= result->ordinal == 0;
      secondResult |= result->ordinal == 1;
    }
    observedCompleteResultTuple |= firstResult && secondResult;
  }
  const bool requiresActiveResultHandoff =
      contractKind != ComputeContractKind::Transparent;
  if (requiresActiveResultHandoff && !observedCompleteResultTuple)
    fail("Spatial compute use lost its complete active-result release tuple");
  if (!requiresActiveResultHandoff &&
      (!sawComputeTransition || !allComputeIntrinsic))
    fail("same-cycle compute use gained a causal result release");
  if (requiresActiveResultHandoff) {
    const auto requireRejectedReleaseMutation = [&](bool removeMember) {
      auto mutated = parseSpatial(context, finalized.canonicalBytes());
      if (!mutated)
        fail("cannot reparse active-result release fixture");
      auto mutatedRoot = *mutated->getOps<::mapping::SpatialOp>().begin();
      std::optional<::mapping::ResourceUseOp> computeUse;
      for (auto use :
           mutatedRoot.getBody().front().getOps<::mapping::ResourceUseOp>()) {
        auto activation =
            mlir::dyn_cast<::mapping::SpatialRelativeActivationAttr>(
                use.getActivation());
        if (activation && activation.getRelease().size() == 2) {
          computeUse = use;
          break;
        }
      }
      if (!computeUse)
        fail("SpatialMapping fixture has no complete result release tuple");
      auto activation = mlir::cast<::mapping::SpatialRelativeActivationAttr>(
          computeUse->getActivation());
      auto release = llvm::to_vector(activation.getRelease());
      if (removeMember) {
        release.pop_back();
      } else {
        release.insert(release.begin(), activation.getTrigger());
      }
      computeUse->setActivationAttr(
          ::mapping::SpatialRelativeActivationAttr::get(
              &context, activation.getTrigger(),
              mlir::ArrayAttr::get(&context, release)));
      if (!rejected(loom::mapping::verifySpatialMappingBase(
              mutatedRoot, dataflow, tech.view(), fabric.view())))
        fail(removeMember
                 ? "SpatialMapping accepted an incomplete result release tuple"
                 : "SpatialMapping accepted an extra result release member");
    };
    requireRejectedReleaseMutation(true);
    requireRejectedReleaseMutation(false);
  }
  auto coldTraversalClaims =
      take(loom::pnr::projectSpatialMappingTraversalClaims(*problem,
                                                           imported.view()));
  if (coldTraversalClaims.total != candidate->totalSelectedTraversalClaim())
    fail("cold Spatial Mapping measure disagrees with Candidate state");

  if (temporal) {
    const auto operandQueueGroups =
        take(loom::mapping::deriveSpatialPeOperandQueueMatchGroups(
            tech.view(), fabric.view(), imported.view().computeBindings(),
            imported.view().routeTrees(), imported.view().resourceUses(),
            imported.view().physicalTagSegments()));
    if (operandQueueGroups.empty() ||
        llvm::any_of(operandQueueGroups, [](const auto &group) {
          return group.matches.empty() || group.tag.getBitWidth() != 4;
        }))
      fail("Temporal SpatialMapping lost its operand queue match groups");
    const auto operandProgress =
        take(loom::mapping::deriveSpatialPeOperandProgressFeedback(
            dataflow, tech.view(), operandQueueGroups));
    if (operandProgress.pairingKeyCount == 0 ||
        operandProgress.pairingKeyCount <
            operandProgress.distinctPairingKeyCount ||
        operandProgress.distinctIngressCount == 0)
      fail("Temporal SpatialMapping lost its qualified pairing projection");
    if (operandProgress.sharedIngressPressure !=
        candidate->sharedOperandIngressPressure())
      fail("persistent SpatialMapping changed operand ingress pressure");
    bool observedEnqueue = false;
    bool observedTransition = false;
    for (const auto &use : imported.view().resourceUses()) {
      const bool enqueue =
          std::holds_alternative<dataflow::CanonicalGraphConsumerEndpointRef>(
              use.activation.trigger.event);
      observedEnqueue |= enqueue;
      if (enqueue && !use.activation.release.empty())
        fail("Temporal queue use gained a causal result release");
      observedTransition |=
          std::holds_alternative<loom::mapping::SpatialActorTransitionEventRef>(
              use.activation.trigger.event);
    }
    if (!observedEnqueue || !observedTransition)
      fail("Temporal SpatialMapping round trip lost queue or operation uses");
    const auto switchRows =
        take(loom::mapping::deriveSpatialTemporalSwitchPackedRows(
            fabric.view(), imported.view().routeTrees(),
            imported.view().resourceUses(),
            imported.view().physicalTagSegments()));
    const auto &handshakeSelection = imported.view().handshakeSelection();
    std::map<std::vector<std::uint8_t>, loom::fabric::FabricOrdinal>
        nextHandshakeRow;
    std::size_t expectedSwitchActivations = 0;
    for (const auto &row : switchRows) {
      const auto occurrenceKey =
          loom::fabric::canonicalFabricBytes(row.occurrence);
      const loom::fabric::FabricOrdinal rowOrdinal =
          nextHandshakeRow[occurrenceKey]++;
      std::map<loom::fabric::FabricOrdinal,
               std::vector<loom::fabric::FabricPhysicalTraversalRef>>
          expectedByInput;
      for (const auto &signature : row.signatures)
        llvm::append_range(expectedByInput[signature.input],
                           signature.traversals);
      for (auto &entry : expectedByInput) {
        const loom::fabric::FabricOrdinal input = entry.first;
        auto &traversals = entry.second;
        llvm::sort(traversals, [](const auto &lhs, const auto &rhs) {
          return loom::fabric::canonicalFabricBytes(lhs) <
                 loom::fabric::canonicalFabricBytes(rhs);
        });
        traversals.erase(std::unique(traversals.begin(), traversals.end()),
                         traversals.end());
        const auto activation = llvm::find_if(
            handshakeSelection.switchActivations, [&](const auto &candidate) {
              return candidate.key.occurrence == row.occurrence &&
                     candidate.key.row == rowOrdinal &&
                     candidate.key.input == input;
            });
        if (activation == handshakeSelection.switchActivations.end() ||
            activation->traversals != traversals)
          fail("SpatialMapping handshake selection diverged from packed rows");
        ++expectedSwitchActivations;
      }
    }
    if (handshakeSelection.switchActivations.size() !=
        expectedSwitchActivations)
      fail("SpatialMapping handshake selection added a switch activation");
    for (const auto &traversal : handshakeSelection.traversals) {
      const auto *sw = std::get_if<loom::fabric::FabricSwitchTraversalPayload>(
          &traversal.payload);
      if (sw && fabric.view().switchSchedule(sw->owner) ==
                    ::fabric::Schedule::Temporal)
        fail("Temporal switch traversal bypassed its resident-row activation");
    }
    std::vector<loom::fabric::FabricSwitchOccurrenceRef> configuredSwitches;
    for (const auto &row : switchRows)
      if (!llvm::is_contained(configuredSwitches, row.occurrence))
        configuredSwitches.push_back(row.occurrence);
    for (const auto occurrence : configuredSwitches) {
      std::vector<loom::fabric::FabricTemporalSwitchRouteEntry> entries;
      for (const auto &row : switchRows)
        if (row.occurrence == occurrence)
          entries.push_back({row.tag, row.traversals});
      const loom::fabric::FabricSemanticConfigFieldRef field{
          loom::fabric::FabricConfigurationOwnerRef(
              loom::fabric::FabricInventoryOwnerRef::of(occurrence)),
          0};
      const auto slot =
          take(loom::mapping::detail::resolveConfiguredHardwareSlot(
              fabric.view(), field));
      const auto expected =
          take(loom::fabric::encodeTemporalSwitchConfiguration(fabric.view(),
                                                               field, entries));
      const auto configured = llvm::find_if(
          imported.view().configuredHardware().fields(),
          [&](const auto &candidate) { return candidate.slot == slot; });
      if (configured == imported.view().configuredHardware().fields().end() ||
          !configured->value.bytes().equals(expected.bytes()))
        fail("configured hardware diverged from Fabric switch rows");
    }
    if (switchPackingFabric && !requireSeparatedSwitchRows &&
        llvm::none_of(switchRows, [](const auto &row) {
          return row.signatures.size() > 1;
        }))
      fail("strict SpatialMapping did not preserve a packed switch row");
    if (requireSeparatedSwitchRows && switchRows.size() < 2)
      fail("strict SpatialMapping merged distinct switch rows");
    if (switchPackingFabric)
      loom::test::exerciseCgraAdmission(
          dataflowReference, fabric.reference(), finalized.reference(),
          buildTemporalFabric(store).reference(), store, blobs, true, false,
          requireSeparatedSwitchRows && !requireContendingSwitchRows);
  }

  if (boundaryWrapped) {
    std::size_t expectedAssignments = 0;
    for (loom::pnr::PnrIndex net = 0;
         net < problem->transfers().logicalNets().size(); ++net)
      expectedAssignments += candidate->tagSegments(net).size();
    loom::test::exerciseCgraAdmission(
        dataflowReference, fabric.reference(), finalized.reference(),
        buildTemporalFabric(store).reference(), store, blobs, true, false);
    std::size_t observedAssignments = 0;
    for (const auto &use : imported.view().resourceUses()) {
      if (!use.sharingAssignments.empty() && !use.activation.release.empty())
        fail("Physical Tag ResourceUse gained a causal release");
      for (const auto &value : use.sharingAssignments) {
        const auto *tag = std::get_if<fabric::PhysicalTagPatternValue>(&value);
        if (!tag || tag->value.getBitWidth() != 4)
          fail("SpatialMapping did not adopt an exact Physical Tag value");
        ++observedAssignments;
      }
    }
    if (observedAssignments != expectedAssignments)
      fail("SpatialMapping did not persist every continuity origin exactly "
           "once");

    loom::test::exerciseSpatialPhysicalTagRuntimeCounterexampleNoGood(
        dataflow, tech.view(), fabric.view(), constraints, imported, pnrConfig,
        *candidate, store);
    loom::test::exerciseSpatialTaggedRuntimeCounterexampleExactRepair(
        dataflow, tech.view(), fabric.view(), constraints, imported, pnrConfig,
        store);
    auto missingTag = parseSpatial(context, finalized.canonicalBytes());
    if (!missingTag)
      fail("cannot reparse Physical Tag ResourceUse fixture");
    auto missingTagRoot = *missingTag->getOps<::mapping::SpatialOp>().begin();
    std::optional<::mapping::ResourceUseOp> tagUse;
    for (auto use :
         missingTagRoot.getBody().front().getOps<::mapping::ResourceUseOp>())
      if (!use.getSharingAssignments().empty()) {
        tagUse = use;
        break;
      }
    if (!tagUse)
      fail("SpatialMapping fixture has no Physical Tag ResourceUse");
    tagUse->erase();
    if (!rejected(loom::mapping::verifySpatialMappingBase(
            missingTagRoot, dataflow, tech.view(), fabric.view())))
      fail("SpatialMapping finalized without a required Physical Tag "
           "assignment");

    auto malformedTag = parseSpatial(context, finalized.canonicalBytes());
    if (!malformedTag)
      fail("cannot reparse malformed Physical Tag fixture");
    auto malformedTagRoot =
        *malformedTag->getOps<::mapping::SpatialOp>().begin();
    std::optional<::mapping::ResourceUseOp> malformedUse;
    for (auto use :
         malformedTagRoot.getBody().front().getOps<::mapping::ResourceUseOp>())
      if (!use.getSharingAssignments().empty()) {
        malformedUse = use;
        break;
      }
    if (!malformedUse)
      fail("SpatialMapping fixture has no Physical Tag value to corrupt");
    const std::array<std::int8_t, 1> noncanonical = {
        static_cast<std::int8_t>(0xf0)};
    auto malformedValue = ::mapping::OwnerTypedValueAttr::get(
        &context, mlir::DenseI8ArrayAttr::get(&context, noncanonical));
    malformedUse->setSharingAssignmentsAttr(
        mlir::ArrayAttr::get(&context, {malformedValue}));
    if (!rejected(loom::mapping::verifySpatialMappingBase(
            malformedTagRoot, dataflow, tech.view(), fabric.view())))
      fail("SpatialMapping accepted a noncanonical Physical Tag value");

    auto collidingTags = parseSpatial(context, finalized.canonicalBytes());
    if (!collidingTags)
      fail("cannot reparse colliding Physical Tag fixture");
    auto collidingTagRoot =
        *collidingTags->getOps<::mapping::SpatialOp>().begin();
    const std::array<std::int8_t, 1> zeroTag = {0};
    auto zeroValue = ::mapping::OwnerTypedValueAttr::get(
        &context, mlir::DenseI8ArrayAttr::get(&context, zeroTag));
    std::size_t rewrittenAssignments = 0;
    for (auto use :
         collidingTagRoot.getBody().front().getOps<::mapping::ResourceUseOp>())
      if (!use.getSharingAssignments().empty()) {
        use.setSharingAssignmentsAttr(
            mlir::ArrayAttr::get(&context, {zeroValue}));
        ++rewrittenAssignments;
      }
    if (rewrittenAssignments < 2 ||
        !rejected(loom::mapping::verifySpatialMappingBase(
            collidingTagRoot, dataflow, tech.view(), fabric.view())))
      fail("SpatialMapping accepted colliding local Physical Tags");
  }

  auto missingRoute = parseSpatial(context, finalized.canonicalBytes());
  if (!missingRoute)
    fail("cannot parse finalized SpatialMapping fixture");
  auto root = *missingRoute->getOps<::mapping::SpatialOp>().begin();
  auto routes = root.getBody().front().getOps<::mapping::RouteTreeOp>();
  (*routes.begin()).erase();
  if (!rejectedWithoutDiagnostic(context, [&] {
        return loom::mapping::verifySpatialMappingBase(
            root, dataflow, tech.view(), fabric.view());
      }))
    fail("SpatialMapping finalized without a required RouteTree");

  auto missingUse = parseSpatial(context, finalized.canonicalBytes());
  if (!missingUse)
    fail("cannot reparse finalized SpatialMapping fixture");
  auto useRoot = *missingUse->getOps<::mapping::SpatialOp>().begin();
  auto uses = useRoot.getBody().front().getOps<::mapping::ResourceUseOp>();
  (*uses.begin()).erase();
  if (!rejected(loom::mapping::verifySpatialMappingBase(
          useRoot, dataflow, tech.view(), fabric.view())))
    fail("SpatialMapping finalized without a required ResourceUse");
}

} // namespace

int main() {
  temporalPeTagMatchDomainsAreIngressLocal();
  completeCandidateRoundTrip(false);
  completeCandidateRoundTrip(false, false, false,
                             ComputeContractKind::LoopStream);
  completeCandidateRoundTrip(false, false, false,
                             ComputeContractKind::Transparent);
  completeCandidateRoundTrip(true);
  completeCandidateRoundTrip(true, false, false,
                             ComputeContractKind::OneCycleElastic,
                             TemporalSwitchRouteFixture::PackedRows);
  completeCandidateRoundTrip(
      true, false, false, ComputeContractKind::OneCycleElastic,
      TemporalSwitchRouteFixture::SameInputSeparatedRows);
  completeCandidateRoundTrip(
      true, false, false, ComputeContractKind::OneCycleElastic,
      TemporalSwitchRouteFixture::ContendingSeparatedRows);
  completeCandidateRoundTrip(
      true, false, false, ComputeContractKind::OneCycleElastic,
      TemporalSwitchRouteFixture::ExceedsResidentCapacity);
  completeCandidateRoundTrip(true, false, true,
                             ComputeContractKind::OneCycleElastic,
                             TemporalSwitchRouteFixture::PackedRows);
  completeCandidateRoundTrip(true, true);
  completeCandidateRoundTrip(true, true, true);
  loom::test::exerciseSpatialRegisterFifoRuntimeCounterexampleExactRepair();
  loom::test::completeMemorySpatialMappingRoundTrip(false);
  loom::test::completeMemorySpatialMappingRoundTrip(true);
  loom::test::completeMemorySpatialMappingRoundTrip(false, true);
  llvm::outs() << "spatial mapping artifact tests passed\n";
  return 0;
}
