#include "Common/ArtifactStore.h"
#include "Common/ArtifactText.h"
#include "Config/ResolvedConfig.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Artifact/FabricArtifactCodec.h"
#include "Mapping/Artifact/MappingArtifact.h"
#include "Mapping/Tech/TechMappingConfig.h"
#include "Mapping/Tech/TechMappingGenerator.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include <chrono>
#include <cstdint>
#include <ctime>
#include <memory>
#include <optional>
#include <string>
#include <system_error>
#include <utility>
#include <variant>
#include <vector>

namespace {

llvm::cl::opt<std::string>
    inputPath(llvm::cl::Positional, llvm::cl::desc("<canonical Dataflow MLIR>"),
              llvm::cl::Required);

llvm::cl::opt<std::string>
    artifactStorePath("artifact-store",
                      llvm::cl::desc("ArtifactStore directory"),
                      llvm::cl::value_desc("path"), llvm::cl::Required);

llvm::cl::list<std::string> fabricReferenceFiles(
    "fabric-reference-file",
    llvm::cl::desc("file containing one exact Module root identity, or a "
                   "System root identity with one imported Module"),
    llvm::cl::value_desc("path"), llvm::cl::OneOrMore);

llvm::cl::opt<std::string> accelerationProfile(
    "loom-accel-profile",
    llvm::cl::desc("builtin acceleration preset or configuration path"),
    llvm::cl::value_desc("preset-or-path"), llvm::cl::init(""));

llvm::cl::opt<std::string> reportPath("report",
                                      llvm::cl::desc("coverage report JSON"),
                                      llvm::cl::value_desc("path"),
                                      llvm::cl::Required);

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      llvm::Twine("tech_mapping_coverage_invalid: ") + message);
}

int reportError(llvm::Error error) {
  llvm::errs() << "loom-tech-map: " << llvm::toString(std::move(error)) << '\n';
  return 1;
}

llvm::Expected<mlir::OwningOpRef<mlir::ModuleOp>>
readModule(mlir::MLIRContext &context, llvm::StringRef path) {
  auto buffer = llvm::MemoryBuffer::getFileOrSTDIN(path);
  if (!buffer)
    return llvm::createStringError(buffer.getError(), "cannot read %s",
                                   path.str().c_str());
  llvm::SourceMgr sourceManager;
  sourceManager.AddNewSourceBuffer(std::move(*buffer), llvm::SMLoc());
  auto module = mlir::parseSourceFile<mlir::ModuleOp>(sourceManager, &context);
  if (!module)
    return invalid("cannot parse canonical Dataflow MLIR");
  return module;
}

llvm::Expected<loom::ArtifactIdentity>
readArtifactIdentity(llvm::StringRef path) {
  auto buffer = llvm::MemoryBuffer::getFile(path);
  if (!buffer)
    return llvm::createStringError(buffer.getError(), "cannot read %s",
                                   path.str().c_str());
  llvm::StringRef spelling = (*buffer)->getBuffer().trim();
  if (spelling.empty())
    return invalid("Fabric reference file is empty");
  return loom::parseArtifactIdentityHex(spelling);
}

llvm::json::Object
accounting(const loom::mapping::TechMappingGenerationAccounting &value) {
  llvm::json::Object result;
  result["match_row_attempts"] = value.matchRowAttempts;
  result["partial_cover_expansions"] = value.partialCoverExpansions;
  result["compute_context_projection_work"] =
      value.computeContextProjectionWork;
  result["compute_context_matching_checks"] =
      value.computeContextMatchingChecks;
  result["compute_context_rejected_checks"] =
      value.computeContextRejectedChecks;
  result["compute_context_matching_work"] = value.computeContextMatchingWork;
  result["memory_supply_projection_work"] = value.memorySupplyProjectionWork;
  result["memory_supply_checks"] = value.memorySupplyChecks;
  result["memory_supply_partial_checks"] = value.memorySupplyPartialChecks;
  result["memory_supply_full_checks"] = value.memorySupplyFullChecks;
  result["memory_supply_rejected_checks"] = value.memorySupplyRejectedChecks;
  result["memory_supply_empty_domain_rejections"] =
      value.memorySupplyEmptyDomainRejections;
  result["memory_supply_exclusive_resource_rejections"] =
      value.memorySupplyExclusiveResourceRejections;
  result["memory_supply_spatial_port_rejections"] =
      value.memorySupplySpatialPortRejections;
  result["memory_supply_temporal_ingress_rejections"] =
      value.memorySupplyTemporalIngressRejections;
  result["memory_supply_internal_connection_rejections"] =
      value.memorySupplyInternalConnectionRejections;
  result["memory_supply_resident_capacity_rejections"] =
      value.memorySupplyResidentCapacityRejections;
  result["memory_supply_joint_assignment_rejections"] =
      value.memorySupplyJointAssignmentRejections;
  result["memory_supply_search_work"] = value.memorySupplySearchWork;
  result["publication_slots"] = value.publicationSlots;
  return result;
}

void appendFeedback(
    llvm::json::Object &result,
    const loom::mapping::TechMappingGenerationFeedback &feedback) {
  if (!feedback.computeContextHall) {
    result["compute_context_hall"] = nullptr;
    return;
  }
  const auto &hall = *feedback.computeContextHall;
  llvm::json::Array groups;
  for (const auto &group : hall.groups())
    groups.push_back(llvm::json::Object{
        {"demand_count", group.demandCount},
        {"compatible_context_count", group.compatibleContexts.size()}});
  result["compute_context_hall"] = llvm::json::Object{
      {"cover_demand_count", hall.coverDemandCount()},
      {"cover_maximum_matching", hall.coverMaximumMatching()},
      {"hall_demand_count", hall.hallDemandCount()},
      {"hall_context_value_count", hall.hallContextValueCount()},
      {"deficit", hall.deficit()},
      {"groups", std::move(groups)}};
}

llvm::json::Object
generatedReport(const loom::mapping::GeneratedTechMappings &generated,
                const loom::ArtifactStore &store,
                std::uint64_t expectedActors) {
  llvm::json::Object result;
  result["status"] = "generated";
  result["classification"] = "pending-spatial-capacity";
  result["termination"] =
      generated.termination ==
              loom::mapping::TechMappingGenerationTermination::SearchExhausted
          ? "search-exhausted"
          : "semantic-limit-reached";
  result["accounting"] = accounting(generated.accounting);
  appendFeedback(result, generated.feedback);
  result["candidate_count"] = generated.candidates.size();
  llvm::json::Array candidates;
  for (const loom::ArtifactRootReference &candidate : generated.candidates)
    candidates.push_back(loom::formatArtifactIdentityHex(candidate.artifact));
  result["candidates"] = std::move(candidates);

  if (!generated.candidates.empty()) {
    auto first =
        loom::mapping::importTechMapping(generated.candidates.front(), store);
    if (!first) {
      result["status"] = "internal";
      result["classification"] = "internal";
      result["reason"] = "candidate-strict-import-failed";
      result["diagnostic"] = llvm::toString(first.takeError());
      return result;
    }
    std::uint64_t coveredActors = 0;
    for (const auto &realization : first->view().computeRealizations())
      coveredActors += realization.actors.size();
    for (const auto &realization : first->view().memoryRealizations())
      coveredActors += realization.actors.size();
    result["covered_actor_count"] = coveredActors;
    result["compute_realization_count"] =
        first->view().computeRealizations().size();
    result["memory_realization_count"] =
        first->view().memoryRealizations().size();
    if (coveredActors != expectedActors) {
      result["status"] = "internal";
      result["classification"] = "internal";
      result["reason"] = "candidate-cover-count-mismatch";
    }
  }
  return result;
}

llvm::json::Object
outcomeReport(const loom::mapping::TechMappingGenerationOutcome &outcome,
              const loom::ArtifactStore &store, std::uint64_t expectedActors) {
  if (const auto *generated =
          std::get_if<loom::mapping::GeneratedTechMappings>(&outcome))
    return generatedReport(*generated, store, expectedActors);
  if (const auto *infeasible =
          std::get_if<loom::mapping::ProvenInfeasibleTechMapping>(&outcome)) {
    llvm::json::Object result;
    result["status"] = "proven-infeasible";
    result["classification"] = "capability-rejected";
    result["reason"] = "no-complete-exact-cover";
    result["accounting"] = accounting(infeasible->accounting);
    appendFeedback(result, infeasible->feedback);
    return result;
  }
  if (const auto *incomplete =
          std::get_if<loom::mapping::IncompleteTechMappingGeneration>(
              &outcome)) {
    llvm::json::Object result;
    result["status"] = "incomplete";
    result["classification"] = "incomplete";
    result["reason"] = "proof-not-established";
    result["accounting"] = accounting(incomplete->accounting);
    appendFeedback(result, incomplete->feedback);
    return result;
  }
  if (const auto *interrupted =
          std::get_if<loom::mapping::InterruptedTechMappingGeneration>(
              &outcome)) {
    llvm::json::Object result;
    result["status"] = "incomplete";
    result["classification"] = "incomplete";
    result["reason"] = "cancelled-or-timeout";
    result["interruption_stage"] =
        loom::mapping::techMappingInterruptionStageSpelling(
            interrupted->snapshot.stage);
    result["accounting"] = accounting(interrupted->accounting);
    appendFeedback(result, interrupted->feedback);
    return result;
  }
  if (const auto *invalid =
          std::get_if<loom::mapping::InvalidTechMappingGeneration>(&outcome)) {
    llvm::json::Object result;
    result["status"] = "invalid";
    result["classification"] = "invalid";
    result["diagnostic"] = invalid->diagnostic;
    result["accounting"] = accounting(invalid->accounting);
    appendFeedback(result, invalid->feedback);
    return result;
  }
  const auto &internal =
      std::get<loom::mapping::InternalTechMappingGeneration>(outcome);
  llvm::json::Object result;
  result["status"] = "internal";
  result["classification"] = "internal";
  result["diagnostic"] = internal.diagnostic;
  result["accounting"] = accounting(internal.accounting);
  appendFeedback(result, internal.feedback);
  return result;
}

llvm::Error writeReport(llvm::StringRef outputPath, llvm::json::Object report) {
  llvm::SmallString<256> parent(outputPath);
  llvm::sys::path::remove_filename(parent);
  if (!parent.empty())
    if (std::error_code error = llvm::sys::fs::create_directories(parent))
      return llvm::createStringError(error, "cannot create %s", parent.c_str());
  std::error_code error;
  llvm::raw_fd_ostream output(outputPath, error, llvm::sys::fs::OF_Text);
  if (error)
    return llvm::createStringError(error, "cannot open %s",
                                   outputPath.str().c_str());
  output << llvm::formatv("{0:2}", llvm::json::Value(std::move(report)))
         << '\n';
  return llvm::Error::success();
}

} // namespace

int main(int argc, char **argv) {
  llvm::InitLLVM init(argc, argv);
  llvm::cl::ParseCommandLineOptions(
      argc, argv,
      "Generate exact TechMapping coverage for one Canonical Dataflow "
      "program.\n");

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::registerAllExtensions(registry);
  registry.insert<::dataflow::DataflowDialect>();
  mlir::MLIRContext context(registry, mlir::MLIRContext::Threading::DISABLED);
  auto source = readModule(context, inputPath);
  if (!source)
    return reportError(source.takeError());
  auto dataflow = dataflow::finalizeCanonicalDataflow(**source);
  if (!dataflow)
    return reportError(dataflow.takeError());
  auto dataflowView = dataflow->view();
  if (!dataflowView)
    return reportError(dataflowView.takeError());
  if (dataflowView->graphs().empty() || dataflowView->actors().empty())
    return reportError(invalid("Dataflow program has no nonempty graph"));

  loom::ArtifactStore store(artifactStorePath);
  auto dataflowReference = dataflow::publishCanonicalDataflow(*dataflow, store);
  if (!dataflowReference)
    return reportError(dataflowReference.takeError());

  llvm::Expected<loom::ResolvedConfig> resolved =
      loom::resolveConfigProfile(accelerationProfile);
  if (!resolved)
    return reportError(resolved.takeError());
  auto techConfig =
      loom::mapping::projectResolvedTechMappingConfigView(*resolved);
  if (!techConfig)
    return reportError(techConfig.takeError());

  std::vector<::dataflow::GraphRef> covers;
  covers.reserve(dataflowView->graphs().size());
  for (const auto &graph : dataflowView->graphs())
    covers.push_back(graph.ref);

  llvm::json::Object report;
  report["kind"] = "tech_mapping_coverage";
  report["canonical_dataflow"] =
      loom::formatArtifactIdentityHex(dataflowReference->artifact);
  report["graph_count"] = dataflowView->graphs().size();
  report["actor_count"] = dataflowView->actors().size();
  llvm::json::Array fabricReports;
  bool failed = false;

  for (const std::string &referencePath : fabricReferenceFiles) {
    auto identity = readArtifactIdentity(referencePath);
    if (!identity)
      return reportError(identity.takeError());
    const loom::ArtifactRootReference reference{
        loom::fabric::fabricArtifactSchema.identity.str(),
        loom::fabric::fabricArtifactSchema.version, *identity};
    auto fabric = loom::fabric::importEntireFabricRoot(reference, store);
    if (!fabric)
      return reportError(fabric.takeError());
    loom::fabric::FabricArtifactView fabricView = fabric->view();
    if (fabricView.rootKind() == loom::fabric::FabricRootKind::System) {
      if (fabricView.importedModules().size() != 1)
        return reportError(invalid(
            "System Fabric root must import exactly one SpatialCore Module"));
      fabricView = fabricView.importedModules().front();
    }
    if (fabricView.rootKind() != loom::fabric::FabricRootKind::Module)
      return reportError(
          invalid("TechMapping requires an exact Fabric Module root"));

    const auto wallStarted = std::chrono::steady_clock::now();
    const std::clock_t cpuStarted = std::clock();
    if (cpuStarted == static_cast<std::clock_t>(-1))
      return reportError(invalid("process CPU clock is unavailable"));
    const loom::mapping::TechMappingGenerationOutcome outcome =
        loom::mapping::generateTechMappings(
            {*dataflowView, covers, fabricView, *techConfig, store});
    const std::clock_t cpuFinished = std::clock();
    if (cpuFinished == static_cast<std::clock_t>(-1))
      return reportError(invalid("process CPU clock is unavailable"));
    const double cpuElapsed =
        static_cast<double>(cpuFinished - cpuStarted) / CLOCKS_PER_SEC;
    const double wallElapsed =
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      wallStarted)
            .count();
    llvm::json::Object fabricReport =
        outcomeReport(outcome, store, dataflowView->actors().size());
    fabricReport["fabric"] =
        loom::formatArtifactIdentityHex(fabricView.identity());
    fabricReport["input_fabric_root"] =
        loom::formatArtifactIdentityHex(*identity);
    fabricReport["generation_cpu_seconds"] = cpuElapsed;
    fabricReport["generation_wall_seconds"] = wallElapsed;
    const llvm::json::Value *classification =
        fabricReport.get("classification");
    if (!classification ||
        (classification->getAsString() != "pending-spatial-capacity" &&
         classification->getAsString() != "capability-rejected"))
      failed = true;
    fabricReports.push_back(std::move(fabricReport));
  }
  report["fabrics"] = std::move(fabricReports);
  if (llvm::Error error = writeReport(reportPath, std::move(report)))
    return reportError(std::move(error));
  return failed ? 2 : 0;
}
