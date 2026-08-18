#include "ADG/Builder.h"
#include "Common/ArtifactStore.h"
#include "Config/ResolvedConfig.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Fabric/IR/OperationResourceContract.h"
#include "Fabric/Identity/FabricRefText.h"
#include "Mapping/Artifact/MappingArtifact.h"
#include "Mapping/Artifact/MappingConstraintSet.h"
#include "Mapping/Inspection/SpatialMappingInspection.h"
#include "Mapping/Tech/TechMappingConfig.h"
#include "Mapping/Tech/TechMappingGenerator.h"
#include "PnR/PnrConfig.h"
#include "PnR/SpatialCandidateInitializer.h"
#include "PnR/SpatialCanonicalSeed.h"
#include "PnR/SpatialPnrGenerator.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Parser/Parser.h"

#include <sys/resource.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <system_error>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace {

[[noreturn]] void fail(const llvm::Twine &message) {
  llvm::errs() << "FAIL: " << message << '\n';
  std::exit(EXIT_FAILURE);
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

class TemporaryDirectory final {
public:
  TemporaryDirectory() {
    llvm::SmallString<128> model("loom-spatial-pnr-scale-%%%%%%");
    if (std::error_code error =
            llvm::sys::fs::createUniqueDirectory(model, path_))
      fail("cannot create temporary directory: " + error.message());
  }

  ~TemporaryDirectory() { llvm::sys::fs::remove_directories(path_); }

  llvm::StringRef path() const { return path_; }

private:
  llvm::SmallString<128> path_;
};

mlir::MLIRContext makeContext() {
  mlir::DialectRegistry registry;
  registry.insert<::dataflow::DataflowDialect, mlir::arith::ArithDialect,
                  mlir::DLTIDialect, mlir::func::FuncDialect>();
  return mlir::MLIRContext(registry, mlir::MLIRContext::Threading::DISABLED);
}

std::uint64_t processCpuNanoseconds() {
  rusage usage{};
  if (::getrusage(RUSAGE_SELF, &usage) != 0)
    fail("getrusage failed while reading process CPU time");
  const auto timeNanoseconds = [](const timeval &value) {
    return static_cast<std::uint64_t>(value.tv_sec) * UINT64_C(1000000000) +
           static_cast<std::uint64_t>(value.tv_usec) * UINT64_C(1000);
  };
  return timeNanoseconds(usage.ru_utime) + timeNanoseconds(usage.ru_stime);
}

std::uint64_t peakResidentBytes() {
  rusage usage{};
  if (::getrusage(RUSAGE_SELF, &usage) != 0)
    fail("getrusage failed while reading peak resident memory");
  return static_cast<std::uint64_t>(usage.ru_maxrss) * UINT64_C(1024);
}

dataflow::CanonicalDataflowArtifact buildSyncChain(mlir::MLIRContext &context,
                                                   std::size_t actorCount) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  stream << "module attributes {dlti.dl_spec = "
            "#dlti.dl_spec<#dlti.dl_entry<index, 64>>} {\n"
            "  dataflow.graph private @sync_chain(%start: none, %value: i32) "
            "-> i32\n"
            "      attributes {input_segments = array<i32: 1, 0, 0>, "
            "result_segments = array<i32: 1, 0, 0>} {\n";
  for (std::size_t actor = 0; actor != actorCount; ++actor) {
    stream << "    %node" << actor << ":2 = dataflow.sync ";
    if (actor == 0) {
      stream << "%start, %value";
    } else {
      stream << "%node" << actor - 1 << "#0, %node" << actor - 1 << "#1";
    }
    stream << " : (none, i32) -> (none, i32)\n";
  }
  stream << "    dataflow.graph.return values(%node" << actorCount - 1
         << "#1 : i32) streams() memories() complete(%node" << actorCount - 1
         << "#0 : none)\n"
            "  }\n}\n";
  stream.flush();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(text, &context);
  if (!module)
    fail("cannot parse sync-chain scale fixture");
  return take(dataflow::finalizeCanonicalDataflow(*module));
}

void addSyncFu(loom::adg::PeBuilder &pe, const loom::adg::PortType &bits32,
               const loom::adg::PortType &bits128, std::uint32_t laneCount) {
  const std::vector<loom::adg::PortType> inputTypes(laneCount, bits32);
  const std::vector<loom::adg::PortType> outputTypes(laneCount, bits128);
  std::vector<loom::adg::PeValue> peInputs;
  for (std::uint32_t lane = 0; lane != laneCount; ++lane)
    peInputs.push_back(take(pe.input(lane)));
  auto fu =
      take(pe.addFu(peInputs, loom::adg::FuSpec{inputTypes, outputTypes}));
  std::vector<loom::adg::FuValue> operationInputs;
  for (std::uint32_t lane = 0; lane != laneCount; ++lane)
    operationInputs.push_back(take(fu.input(lane)));
  auto operation = take(fu.addOperation(
      operationInputs,
      loom::adg::OperationCapabilitySpec{
          ::fabric::ImplementationFamilyId::TokenSync,
          ::fabric::RoutedTokenParams{128, laneCount},
          {::dataflow::OperationSchemaId::DataflowSync},
          std::vector<loom::adg::PortType>(laneCount, bits32),
          ::fabric::oneCycleElasticOperationResourceContract()}));
  requireSuccess(fu.addCapabilityTemplate(
      loom::adg::FuCapabilityTemplateSpec{{operation}, {}}));
  std::vector<loom::adg::FuValue> outputs;
  for (std::uint32_t lane = 0; lane != laneCount; ++lane)
    outputs.push_back(take(operation.output(lane)));
  requireSuccess(fu.close(outputs));
}

loom::fabric::FinalizedFabricRoot
buildBoundedMeshFabric(const loom::ArtifactStore &store, std::uint32_t rows,
                       std::uint32_t columns, std::uint32_t laneCount,
                       std::uint32_t unitsPerCell) {
  constexpr std::uint32_t meshLinkLaneCount = 2;
  const std::uint32_t siteCount = rows * columns;
  const loom::adg::PortType bits128 = take(loom::adg::PortType::bits(128));
  const loom::adg::PortType bits32 = take(loom::adg::PortType::bits(32));
  const std::vector<loom::adg::PortType> peTypes(laneCount, bits128);
  const std::vector<loom::adg::PortType> tileTypes(
      static_cast<std::size_t>(unitsPerCell) * laneCount, bits128);
  const std::vector<loom::adg::PortType> boundaryInputTypes(3, bits128);
  const std::vector<loom::adg::PortType> boundaryOutputTypes(2, bits128);

  loom::adg::DesignBuilder design(store);
  auto unit = take(design.createSpatialCore("sync-unit", peTypes, peTypes));
  std::vector<loom::adg::SpatialValue> unitInputs;
  unitInputs.reserve(laneCount);
  for (std::uint32_t lane = 0; lane != laneCount; ++lane)
    unitInputs.push_back(take(unit.input(lane)));
  auto pe = take(
      unit.addPe(unitInputs, loom::adg::PeSpec::spatial(peTypes, peTypes)));
  addSyncFu(pe, bits32, bits128, laneCount);
  requireSuccess(pe.close());
  std::vector<loom::adg::SpatialValue> unitOutputs;
  unitOutputs.reserve(laneCount);
  for (std::uint32_t lane = 0; lane != laneCount; ++lane)
    unitOutputs.push_back(take(pe.output(lane)));
  requireSuccess(unit.close(unitOutputs));
  const auto unitClocks =
      take(unit.domainSlots(loom::fabric::FabricClockResetKind::Clock));
  const auto unitResets =
      take(unit.domainSlots(loom::fabric::FabricClockResetKind::Reset));

  auto tile = take(design.createSpatialCore("mesh-tile", tileTypes, tileTypes));
  const auto tileClock =
      take(tile.declareDomainSlot(loom::fabric::FabricClockResetKind::Clock));
  const auto tileReset =
      take(tile.declareDomainSlot(loom::fabric::FabricClockResetKind::Reset));
  std::vector<loom::adg::SpatialValue> tileOutputs;
  tileOutputs.reserve(tileTypes.size());
  for (std::uint32_t ordinal = 0; ordinal != unitsPerCell; ++ordinal) {
    std::vector<loom::adg::SpatialValue> inputs;
    inputs.reserve(laneCount);
    for (std::uint32_t lane = 0; lane != laneCount; ++lane)
      inputs.push_back(take(tile.input(ordinal * laneCount + lane)));
    auto outputs = take(tile.instantiate(
        unit, inputs,
        {{unitClocks.front(), tileClock}, {unitResets.front(), tileReset}}));
    tileOutputs.insert(tileOutputs.end(), outputs.begin(), outputs.end());
  }
  for (std::size_t ordinal = 0; ordinal != tileTypes.size(); ++ordinal) {
    const auto input = take(tile.inputDomainMember(ordinal));
    const auto output = take(tile.outputDomainMember(ordinal));
    requireSuccess(tile.assignDomainSlot(input, tileClock));
    requireSuccess(tile.assignDomainSlot(input, tileReset));
    requireSuccess(tile.assignDomainSlot(output, tileClock));
    requireSuccess(tile.assignDomainSlot(output, tileReset));
  }
  requireSuccess(tile.close(tileOutputs));

  std::vector<loom::adg::MeshCellAttachmentSpec> attachmentSpecs;
  attachmentSpecs.reserve(static_cast<std::size_t>(siteCount) + 2);
  for (std::uint32_t y = 0; y != rows; ++y)
    for (std::uint32_t x = 0; x != columns; ++x)
      attachmentSpecs.push_back({x, y, tileTypes, tileTypes});
  const std::size_t inputAttachmentOrdinal = attachmentSpecs.size();
  attachmentSpecs.push_back({0, 0, {}, boundaryInputTypes});
  const std::size_t outputAttachmentOrdinal = attachmentSpecs.size();
  attachmentSpecs.push_back({columns - 1, rows - 1, boundaryOutputTypes, {}});

  auto mesh = take(design.createSpatialCore(
      "finite-degree-mesh", boundaryInputTypes, boundaryOutputTypes));
  const auto meshClock =
      take(mesh.declareDomainSlot(loom::fabric::FabricClockResetKind::Clock));
  const auto meshReset =
      take(mesh.declareDomainSlot(loom::fabric::FabricClockResetKind::Reset));
  auto network = take(
      mesh.addMeshSwitchNetwork(take(loom::adg::MeshSwitchNetworkSpec::spatial(
          columns, rows, meshLinkLaneCount, bits128,
          std::move(attachmentSpecs)))));

  for (std::uint32_t site = 0; site != siteCount; ++site) {
    auto attachment = take(network.attachment(site));
    auto outputs = take(
        mesh.instantiate(tile, attachment.inputs(),
                         {{tileClock, meshClock}, {tileReset, meshReset}}));
    requireSuccess(attachment.connectOutputs(outputs));
  }

  for (const auto &member : network.domainMembers()) {
    requireSuccess(mesh.assignDomainSlot(member, meshClock));
    requireSuccess(mesh.assignDomainSlot(member, meshReset));
  }
  for (std::size_t ordinal = 0; ordinal != boundaryInputTypes.size();
       ++ordinal) {
    const auto member = take(mesh.inputDomainMember(ordinal));
    requireSuccess(mesh.assignDomainSlot(member, meshClock));
    requireSuccess(mesh.assignDomainSlot(member, meshReset));
  }
  for (std::size_t ordinal = 0; ordinal != boundaryOutputTypes.size();
       ++ordinal) {
    const auto member = take(mesh.outputDomainMember(ordinal));
    requireSuccess(mesh.assignDomainSlot(member, meshClock));
    requireSuccess(mesh.assignDomainSlot(member, meshReset));
  }

  auto inputAttachment = take(network.attachment(inputAttachmentOrdinal));
  std::vector<loom::adg::SpatialValue> moduleInputs;
  moduleInputs.reserve(boundaryInputTypes.size());
  for (std::size_t ordinal = 0; ordinal != boundaryInputTypes.size(); ++ordinal)
    moduleInputs.push_back(take(mesh.input(ordinal)));
  requireSuccess(inputAttachment.connectOutputs(moduleInputs));

  auto outputAttachment = take(network.attachment(outputAttachmentOrdinal));
  requireSuccess(mesh.close(outputAttachment.inputs()));
  auto finalized = take(std::move(design).finalize());
  if (finalized.roots().size() != 3)
    fail("bounded mesh fixture did not publish its unit, tile, and top-level "
         "roots");

  const auto &view = finalized.roots().back().view();
  if (view.peOccurrences().size() !=
      static_cast<std::size_t>(siteCount) * unitsPerCell)
    fail("bounded mesh fixture changed its PE coverage");
  const std::uint64_t undirectedEdges =
      static_cast<std::uint64_t>(rows) * (columns - 1) +
      static_cast<std::uint64_t>(columns) * (rows - 1);
  if (view.fifoOccurrences().size() != undirectedEdges * 2 * meshLinkLaneCount)
    fail("bounded mesh fixture changed its registered transport boundaries");
  return finalized.roots().back();
}

loom::mapping::FinalizedTechMapping
generateTechMapping(const dataflow::CanonicalDataflowProgramView &dataflow,
                    const loom::fabric::FabricArtifactView &fabric,
                    const loom::ArtifactStore &store) {
  loom::ResolvedConfig resolved = loom::defaultResolvedConfig();
  resolved.dse.techMapping.matchRowAttemptLimit = UINT64_C(2000000);
  resolved.dse.techMapping.partialCoverExpansionLimit = UINT64_C(2000000);
  resolved.dse.techMapping.candidatePublicationLimit = 1;
  const auto config =
      take(loom::mapping::projectResolvedTechMappingConfigView(resolved));
  const std::array<dataflow::GraphRef, 1> covers = {
      dataflow.graphs().front().ref};
  const auto start = std::chrono::steady_clock::now();
  auto outcome = loom::mapping::generateTechMappings(
      {dataflow, covers, fabric, config, store});
  const auto elapsed = std::chrono::steady_clock::now() - start;
  if (elapsed >= std::chrono::seconds(10))
    fail("scale TechMapping exceeded its ten-second budget");
  const auto *generated =
      std::get_if<loom::mapping::GeneratedTechMappings>(&outcome);
  if (!generated || generated->candidates.size() != 1) {
    std::string detail = "variant=" + std::to_string(outcome.index());
    std::visit(
        [&](const auto &value) {
          detail += " match_rows=" +
                    std::to_string(value.accounting.matchRowAttempts) +
                    " cover_expansions=" +
                    std::to_string(value.accounting.partialCoverExpansions);
          using Outcome = std::decay_t<decltype(value)>;
          if constexpr (std::is_same_v<
                            Outcome,
                            loom::mapping::InvalidTechMappingGeneration> ||
                        std::is_same_v<
                            Outcome,
                            loom::mapping::InternalTechMappingGeneration>)
            detail += " diagnostic=" + value.diagnostic;
        },
        outcome);
    fail("scale fixture did not produce one TechMapping: " + detail);
  }
  llvm::outs()
      << "tech_mapping_scale actors=" << dataflow.actors().size() << " wall_ns="
      << std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count()
      << " match_rows=" << generated->accounting.matchRowAttempts
      << " cover_expansions=" << generated->accounting.partialCoverExpansions
      << '\n';
  return take(
      loom::mapping::importTechMapping(generated->candidates.front(), store));
}

loom::pnr::ResolvedPnrConfigView scalePnrConfig() {
  loom::ResolvedConfig resolved = loom::defaultResolvedConfig();
  auto &search = resolved.dse.spatialPnr.search;
  search.initializer.seedAttemptCount = 1;
  search.initializer.assignmentAttemptLimitPerSeed = UINT64_C(2000000);
  search.actionProposal = {1, 1, 0};
  search.routing.endpointExpansionLimit = UINT64_C(8000000);
  search.routing.negotiationIterationLimit = 64;
  search.routing.noProgressIterationLimit = 8;
  search.routing.noProgressTrendWindow = 4;
  search.annealing.calibrationProposalCount = 1;
  search.annealing.fallbackTemperature = 1;
  search.annealing.minimumTemperature = 1;
  search.annealing.coolingRatio = {1, 2};
  search.annealing.proposalsPerLevelBase = 1;
  search.annealing.proposalsPerMovableDecision = 2;
  search.exactRepair = {loom::ResolvedPnrExactRepairKind::CpSat, 256, 1024};
  return take(loom::pnr::projectResolvedSpatialPnrConfigView(resolved));
}

struct ScaleObservation final {
  std::uint64_t wallNanoseconds = 0;
  std::uint64_t cpuNanoseconds = 0;
  std::uint64_t peakResidentBytes = 0;
};

ScaleObservation observeSpatialPnr(
    const dataflow::CanonicalDataflowProgramView &dataflow,
    const loom::mapping::TechMappingView &tech,
    const loom::fabric::FabricArtifactView &fabric,
    const loom::fabric::FabricPhysicalTimingProfileView &physicalTiming,
    const loom::pnr::ResolvedPnrConfigView &config,
    const loom::mapping::SpatialMappingConstraintSetView &constraints,
    const loom::ArtifactStore &store,
    loom::pnr::SpatialPnrGenerationOutcome &outcome) {
  const std::uint64_t cpuStart = processCpuNanoseconds();
  const auto wallStart = std::chrono::steady_clock::now();
  outcome = loom::pnr::generateSpatialMappings(
      {dataflow, tech, fabric, physicalTiming, config, constraints, store, 2});
  const auto wallEnd = std::chrono::steady_clock::now();
  return {static_cast<std::uint64_t>(
              std::chrono::duration_cast<std::chrono::nanoseconds>(wallEnd -
                                                                   wallStart)
                  .count()),
          processCpuNanoseconds() - cpuStart, peakResidentBytes()};
}

void printObservation(
    llvm::StringRef kind, std::size_t actorCount,
    const ScaleObservation &observation, std::uint64_t totalWallNanoseconds,
    const loom::pnr::SpatialPnrGenerationAccounting &accounting) {
  llvm::outs() << "spatial_pnr_scale kind=" << kind << " actors=" << actorCount
               << " total_wall_ns=" << totalWallNanoseconds
               << " wall_ns=" << observation.wallNanoseconds
               << " cpu_ns=" << observation.cpuNanoseconds
               << " peak_rss_bytes=" << observation.peakResidentBytes
               << " seed_slots=" << accounting.seedAttemptSlots
               << " assignments=" << accounting.initializerAssignmentAttempts
               << " endpoint_expansions=" << accounting.endpointExpansionSlots
               << " negotiations=" << accounting.negotiationIterationSlots
               << " base_proposals=" << accounting.annealingBaseProposalSlots
               << " movable_proposals="
               << accounting.annealingMovableProposalSlots
               << " final_closures=" << accounting.finalClosureAttempts
               << " publications=" << accounting.publicationSlots << '\n';
}

std::string spatialOutcomeDiagnostic(
    const loom::pnr::SpatialPnrGenerationOutcome &outcome) {
  std::string detail = "variant=" + std::to_string(outcome.index()) + " ";
  detail += std::visit(
      [](const auto &value) -> std::string {
        using Outcome = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<Outcome,
                                     loom::pnr::GeneratedSpatialMappings>)
          return "generated";
        else if constexpr (std::is_same_v<
                               Outcome,
                               loom::pnr::ProvenInfeasibleSpatialMapping>)
          return "proven infeasible: " + value.diagnostic;
        else if constexpr (std::is_same_v<
                               Outcome,
                               loom::pnr::InterruptedSpatialPnrGeneration>)
          return (llvm::Twine("interrupted at ") +
                  loom::pnr::spatialPnrInterruptionStageSpelling(
                      value.snapshot.stage))
              .str();
        else {
          std::string result;
          if constexpr (
              std::is_same_v<Outcome,
                             loom::pnr::IncompleteSpatialPnrGeneration> ||
              std::is_same_v<Outcome,
                             loom::pnr::UnsupportedSpatialPnrGeneration> ||
              std::is_same_v<Outcome, loom::pnr::InvalidSpatialPnrGeneration> ||
              std::is_same_v<Outcome, loom::pnr::InternalSpatialPnrGeneration>)
            result = "reason=" +
                     std::to_string(static_cast<unsigned>(value.reason)) + " ";
          return result + value.diagnostic;
        }
      },
      outcome);
  return detail;
}

void regularMeshProducesTypedOutcome() {
  constexpr std::size_t actorCount = 1009;
  const auto caseStart = std::chrono::steady_clock::now();
  TemporaryDirectory directory;
  loom::ArtifactStore store(directory.path());
  mlir::MLIRContext context = makeContext();
  auto dataflowArtifact = buildSyncChain(context, actorCount);
  take(dataflow::publishCanonicalDataflow(dataflowArtifact, store));
  auto dataflow = take(dataflowArtifact.view());
  auto fabric = buildBoundedMeshFabric(store, 16, 16, 2, 4);
  auto tech = generateTechMapping(dataflow, fabric.view(), store);
  if (tech.view().computeRealizations().size() != actorCount)
    fail("regular scale TechMapping did not retain singleton actor rows");
  auto constraints =
      take(loom::mapping::finalizeEmptySpatialMappingConstraintSet(
          dataflow, tech.view(), fabric.view(), store));
  auto config = scalePnrConfig();
  auto physicalTiming =
      take(loom::fabric::projectNormalizedFabricPhysicalTimingProfile(
          fabric.view()));
  loom::pnr::SpatialPnrGenerationOutcome outcome;
  const ScaleObservation observation =
      observeSpatialPnr(dataflow, tech.view(), fabric.view(), physicalTiming,
                        config, constraints.view(), store, outcome);
  const auto *generated =
      std::get_if<loom::pnr::GeneratedSpatialMappings>(&outcome);
  if (!generated || generated->candidates.empty() ||
      generated->termination !=
          loom::pnr::PnrGenerationTermination::FixedAttemptsCompleted)
    fail("regular finite-degree mesh produced no completed Mapping: " +
         spatialOutcomeDiagnostic(outcome));
  for (const auto &reference : generated->candidates) {
    auto mapping = take(loom::mapping::importSpatialMapping(reference, store));
    auto inspection = take(loom::mapping::inspectSpatialMapping(
        dataflow, tech.view(), fabric.view(), mapping.view()));
    if (inspection.summary.selectedActorCount != actorCount ||
        inspection.summary.computeRealizationCount != actorCount ||
        inspection.summary.routeTreeCount == 0)
      fail("regular scale Mapping inspection lost physical work");
  }
  const std::uint64_t totalWallNanoseconds =
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now() - caseStart)
          .count();
  if (totalWallNanoseconds >= UINT64_C(90) * UINT64_C(1000000000))
    fail("regular scale Mapping exceeded its ninety-second budget");
  if (observation.peakResidentBytes >= UINT64_C(8) * 1024 * 1024 * 1024)
    fail("regular scale Spatial PnR exceeded its 8 GiB resident budget");
  printObservation("regular", actorCount, observation, totalWallNanoseconds,
                   generated->accounting);
}

void techMappingCompletesWithinBudget() {
  constexpr std::size_t actorCount = 1009;
  TemporaryDirectory directory;
  loom::ArtifactStore store(directory.path());
  mlir::MLIRContext context = makeContext();
  auto dataflowArtifact = buildSyncChain(context, actorCount);
  take(dataflow::publishCanonicalDataflow(dataflowArtifact, store));
  auto dataflow = take(dataflowArtifact.view());
  auto fabric = buildBoundedMeshFabric(store, 16, 16, 2, 4);
  auto tech = generateTechMapping(dataflow, fabric.view(), store);
  if (tech.view().computeRealizations().size() != dataflow.actors().size())
    fail("scale TechMapping did not retain singleton actor rows");
}

void irregularMeshProvesResidentContextPigeonhole() {
  constexpr std::size_t actorCount = 1009;
  const auto caseStart = std::chrono::steady_clock::now();
  TemporaryDirectory directory;
  loom::ArtifactStore store(directory.path());
  mlir::MLIRContext context = makeContext();
  auto dataflowArtifact = buildSyncChain(context, actorCount);
  take(dataflow::publishCanonicalDataflow(dataflowArtifact, store));
  auto dataflow = take(dataflowArtifact.view());
  auto fabric = buildBoundedMeshFabric(store, 9, 14, 2, 4);
  loom::ResolvedConfig resolved = loom::defaultResolvedConfig();
  resolved.dse.techMapping.matchRowAttemptLimit = UINT64_C(2000000);
  resolved.dse.techMapping.partialCoverExpansionLimit = UINT64_C(2000000);
  resolved.dse.techMapping.candidatePublicationLimit = 1;
  const auto config =
      take(loom::mapping::projectResolvedTechMappingConfigView(resolved));
  const std::array<dataflow::GraphRef, 1> covers = {
      dataflow.graphs().front().ref};
  const std::uint64_t cpuStart = processCpuNanoseconds();
  const auto wallStart = std::chrono::steady_clock::now();
  const auto outcome = loom::mapping::generateTechMappings(
      {dataflow, covers, fabric.view(), config, store});
  const auto wallEnd = std::chrono::steady_clock::now();
  const ScaleObservation observation{
      static_cast<std::uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(wallEnd -
                                                               wallStart)
              .count()),
      processCpuNanoseconds() - cpuStart, peakResidentBytes()};
  const auto *proof =
      std::get_if<loom::mapping::ProvenInfeasibleTechMapping>(&outcome);
  if (!proof)
    fail("resident context pigeonhole was not proven by Tech root supply");
  if (proof->accounting.computeContextRejectedChecks == 0 ||
      proof->accounting.candidateEvaluations != 0 ||
      proof->accounting.publicationSlots != 0)
    fail("resident context pigeonhole consumed candidate or publication work");
  const std::uint64_t totalWallNanoseconds =
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now() - caseStart)
          .count();
  if (totalWallNanoseconds >= UINT64_C(90) * UINT64_C(1000000000))
    fail("irregular scale Mapping exceeded its ninety-second budget");
  if (observation.peakResidentBytes >= UINT64_C(8) * 1024 * 1024 * 1024)
    fail("irregular scale Spatial PnR exceeded its 8 GiB resident budget");
  llvm::outs() << "tech_mapping_root_supply kind=irregular_infeasible actors="
               << actorCount << " total_wall_ns=" << totalWallNanoseconds
               << " wall_ns=" << observation.wallNanoseconds
               << " cpu_ns=" << observation.cpuNanoseconds
               << " peak_rss_bytes=" << observation.peakResidentBytes
               << " context_checks="
               << proof->accounting.computeContextMatchingChecks
               << " context_rejections="
               << proof->accounting.computeContextRejectedChecks
               << " candidate_evaluations="
               << proof->accounting.candidateEvaluations
               << " publications=" << proof->accounting.publicationSlots
               << '\n';
}

void irregularMeshProducesTypedOutcome() {
  constexpr std::size_t actorCount = 481;
  const auto caseStart = std::chrono::steady_clock::now();
  TemporaryDirectory directory;
  loom::ArtifactStore store(directory.path());
  mlir::MLIRContext context = makeContext();
  auto dataflowArtifact = buildSyncChain(context, actorCount);
  take(dataflow::publishCanonicalDataflow(dataflowArtifact, store));
  auto dataflow = take(dataflowArtifact.view());
  auto fabric = buildBoundedMeshFabric(store, 9, 14, 2, 4);
  auto tech = generateTechMapping(dataflow, fabric.view(), store);
  if (tech.view().computeRealizations().size() != actorCount)
    fail("feasible irregular TechMapping did not retain singleton actor rows");
  auto constraints =
      take(loom::mapping::finalizeEmptySpatialMappingConstraintSet(
          dataflow, tech.view(), fabric.view(), store));
  auto config = scalePnrConfig();
  auto physicalTiming =
      take(loom::fabric::projectNormalizedFabricPhysicalTimingProfile(
          fabric.view()));
  loom::pnr::SpatialPnrGenerationOutcome outcome;
  const ScaleObservation observation =
      observeSpatialPnr(dataflow, tech.view(), fabric.view(), physicalTiming,
                        config, constraints.view(), store, outcome);
  const auto *generated =
      std::get_if<loom::pnr::GeneratedSpatialMappings>(&outcome);
  if (!generated || generated->candidates.empty() ||
      generated->termination !=
          loom::pnr::PnrGenerationTermination::FixedAttemptsCompleted)
    fail("feasible irregular mesh produced no completed Mapping: " +
         spatialOutcomeDiagnostic(outcome));
  for (const auto &reference : generated->candidates) {
    auto mapping = take(loom::mapping::importSpatialMapping(reference, store));
    auto inspection = take(loom::mapping::inspectSpatialMapping(
        dataflow, tech.view(), fabric.view(), mapping.view()));
    if (inspection.summary.selectedActorCount != actorCount ||
        inspection.summary.computeRealizationCount != actorCount ||
        inspection.summary.routeTreeCount == 0)
      fail("feasible irregular Mapping inspection lost physical work");
  }
  const std::uint64_t totalWallNanoseconds =
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now() - caseStart)
          .count();
  if (totalWallNanoseconds >= UINT64_C(90) * UINT64_C(1000000000))
    fail("feasible irregular Mapping exceeded its ninety-second budget");
  if (observation.peakResidentBytes >= UINT64_C(8) * 1024 * 1024 * 1024)
    fail("feasible irregular Spatial PnR exceeded its 8 GiB resident budget");
  printObservation("irregular_feasible", actorCount, observation,
                   totalWallNanoseconds, generated->accounting);
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2)
    fail("expected exactly one test case name");
  const llvm::StringRef name(argv[1]);
  if (name == "tech")
    techMappingCompletesWithinBudget();
  else if (name == "regular")
    regularMeshProducesTypedOutcome();
  else if (name == "irregular")
    irregularMeshProvesResidentContextPigeonhole();
  else if (name == "irregular-feasible")
    irregularMeshProducesTypedOutcome();
  else
    fail("unknown test case: " + name);
  return EXIT_SUCCESS;
}
