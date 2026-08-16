#include "ADG/Builtin.h"
#include "Common/ArtifactStore.h"
#include "Common/ArtifactText.h"
#include "Common/BlobStore.h"
#include "Config/ResolvedConfig.h"
#include "DSE/PreMappingExploration.h"
#include "Dataflow/IR/OperationSchema.h"
#include "Frontend/Compilation/OwnershipCandidateGenerator.h"
#include "Frontend/Executable/CompilerTargetBinding.h"
#include "Frontend/IR/LoomOps.h"
#include "Frontend/IR/StructuredProgramArtifact.h"
#include "Frontend/Raising/StructuredRaising.h"
#include "Simulator/SimulationArtifacts.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace {

llvm::cl::opt<std::string>
    targetModulePath(llvm::cl::Positional,
                     llvm::cl::desc("<target LLVM IR module (.ll/.bc)>"),
                     llvm::cl::Required);

llvm::cl::opt<std::string>
    artifactStorePath("artifact-store",
                      llvm::cl::desc("ArtifactStore directory"),
                      llvm::cl::value_desc("path"), llvm::cl::Required);

llvm::cl::opt<std::string> accelerationProfile(
    "loom-accel-profile",
    llvm::cl::desc("builtin acceleration preset or configuration path"),
    llvm::cl::value_desc("preset-or-path"), llvm::cl::init(""));

llvm::cl::opt<std::string> outputPath("output",
                                      llvm::cl::desc("comparison report JSON"),
                                      llvm::cl::value_desc("path"),
                                      llvm::cl::Required);

llvm::cl::opt<std::string> canonicalOutputPath(
    "canonical-output",
    llvm::cl::desc("optional finalized Canonical Dataflow MLIR projection"),
    llvm::cl::value_desc("path"), llvm::cl::init(""));

llvm::cl::opt<unsigned>
    candidateJobs("candidate-jobs",
                  llvm::cl::desc("parallel ownership-candidate workers"),
                  llvm::cl::value_desc("count"), llvm::cl::init(1));

llvm::cl::list<std::string> operatorProtocolSymbols(
    "operator-protocol-symbol",
    llvm::cl::desc("defined LLVM callable that roots the invocation-local "
                   "operator protocol ownership domain"),
    llvm::cl::value_desc("symbol"), llvm::cl::ZeroOrMore);

llvm::cl::opt<std::int64_t> expectedEntryResult(
    "expected-entry-result",
    llvm::cl::desc("required signed i32 result from the source entry oracle"),
    llvm::cl::value_desc("value"));

llvm::cl::opt<std::uint64_t>
    maxEventSteps("max-event-steps",
                  llvm::cl::desc("maximum aggregate DFG event wavefronts"),
                  llvm::cl::init(100000));

llvm::cl::opt<std::uint64_t>
    maxEventCount("max-event-count",
                  llvm::cl::desc("maximum aggregate DFG event count"),
                  llvm::cl::init(1000000));

llvm::cl::opt<std::uint64_t>
    maxCaptureBytes("max-capture-bytes",
                    llvm::cl::desc("maximum retained capture bytes"),
                    llvm::cl::init(256ULL * 1024ULL * 1024ULL));

llvm::cl::opt<double> maxSimulationWallSeconds(
    "max-simulation-wall-seconds",
    llvm::cl::desc("maximum aggregate DFG replay wall time"),
    llvm::cl::init(15.0));

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      llvm::Twine("source_backed_dfg_invalid: ") + message);
}

llvm::Error unsupported(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::not_supported),
      llvm::Twine("source_backed_dfg_unsupported: ") + message);
}

int reportError(llvm::Error error) {
  llvm::errs() << "loom-dfg-run: " << llvm::toString(std::move(error)) << '\n';
  return 1;
}

llvm::Error requireExpectedEntryResult(
    const loom::sim::SourceBackedDfgValidationResult &replay) {
  if (expectedEntryResult.getNumOccurrences() == 0)
    return llvm::Error::success();
  if (expectedEntryResult < std::numeric_limits<std::int32_t>::min() ||
      expectedEntryResult > std::numeric_limits<std::int32_t>::max())
    return invalid("expected entry result is outside signed i32");
  if (!replay.sourceReturnValue)
    return invalid("source workload oracle did not return an entry result");
  const loom::sim::CanonicalValueSequence &value = *replay.sourceReturnValue;
  if (value.tokenCount != 1 || value.lanes.size() != 1 ||
      value.lanes.front().state != loom::sim::SemanticState::Defined ||
      value.lanes.front().bits.getBitWidth() != 32)
    return invalid("source workload oracle result is not one defined i32");
  const llvm::APInt expected(
      32, static_cast<std::uint64_t>(expectedEntryResult), true);
  if (value.lanes.front().bits != expected)
    return invalid("source workload oracle rejected the entry: expected " +
                   llvm::Twine(expectedEntryResult) + ", got " +
                   llvm::Twine(value.lanes.front().bits.getSExtValue()));
  return llvm::Error::success();
}

llvm::Expected<std::unique_ptr<llvm::Module>>
readModule(llvm::LLVMContext &context, llvm::StringRef path) {
  llvm::SMDiagnostic diagnostic;
  std::unique_ptr<llvm::Module> module =
      llvm::parseIRFile(path, diagnostic, context);
  if (module)
    return std::move(module);
  std::string message;
  llvm::raw_string_ostream stream(message);
  diagnostic.print("loom-dfg-run", stream);
  return invalid(stream.str());
}

struct NullaryProgramInputs final {
  loom::sim::CanonicalSimulationWorkload workload;
  loom::sim::CanonicalSimulationRuntimeInput runtimeInput;
};

struct SourceBackedCompilation final {
  loom::dse::SelectedPreMappingCompilation selected;
  loom::ArtifactIdentity structuredInitialIdentity;
  loom::ArtifactIdentity workloadIdentity;
  loom::ArtifactIdentity runtimeInputIdentity;
  std::vector<loom::dse::DsePlanGenerateInvocationRecords>
      planGenerateInvocations;
  loom::dse::DsePlanGenerateInvocationSummary planGenerateSummary;
};

llvm::Expected<NullaryProgramInputs> makeNullaryProgramInputs(
    const loom::frontend::StructuredProgramCandidate &candidate,
    llvm::StringRef symbol) {
  auto view = candidate.view();
  if (!view)
    return view.takeError();
  std::optional<loom::frontend::StructuredEntityRef> entry;
  mlir::LLVM::LLVMFuncOp entryOp;
  for (const loom::frontend::StructuredEntity &entity :
       view->entities(loom::frontend::StructuredEntityKind::Operation)) {
    auto function =
        llvm::dyn_cast_or_null<mlir::LLVM::LLVMFuncOp>(entity.operation);
    if (!function || function.getSymName() != symbol)
      continue;
    if (entry)
      return invalid("program entry symbol is not unique");
    entry = entity.reference;
    entryOp = function;
  }
  if (!entry)
    return invalid("program entry symbol does not resolve");
  if (entryOp.isVarArg() || entryOp.getFunctionType().getNumParams() != 0)
    return unsupported("focused DFG runner requires a nullary program entry");

  loom::sim::StructuredProgramSimulationWorkload workloadDraft{*entry};
  workloadDraft.observableContract.returnValue =
      !llvm::isa<mlir::LLVM::LLVMVoidType>(
          entryOp.getFunctionType().getReturnType());
  auto workload = loom::sim::finalizeSimulationWorkload(workloadDraft, *view);
  if (!workload)
    return workload.takeError();
  loom::sim::StructuredProgramSimulationRuntimeInputDraft runtimeDraft{
      workload->identity()};
  auto runtimeInput =
      loom::sim::finalizeSimulationRuntimeInput(runtimeDraft, *workload, *view);
  if (!runtimeInput)
    return runtimeInput.takeError();
  return NullaryProgramInputs{std::move(*workload), std::move(*runtimeInput)};
}

llvm::Expected<std::vector<loom::frontend::StructuredEntityRef>>
resolveOperatorProtocolRoots(
    const loom::frontend::StructuredProgramCandidate &candidate) {
  llvm::SmallVector<llvm::StringRef> symbols;
  symbols.reserve(operatorProtocolSymbols.size());
  for (const std::string &symbol : operatorProtocolSymbols) {
    symbols.push_back(symbol);
  }
  return loom::frontend::resolveDefinedLlvmCallables(candidate, symbols);
}

llvm::Expected<SourceBackedCompilation>
compileTarget(std::unique_ptr<llvm::Module> module,
              const loom::fabric::FinalizedFabricRoot &fabric,
              const loom::ResolvedConfig &config,
              const loom::ArtifactStore &store, const loom::BlobStore &blobs) {
  loom::frontend::PreMappingCompilationOptions compilation;
  auto source = loom::frontend::raiseLlvmModuleToStructured(
      std::move(module), fabric, compilation.raising);
  if (!source)
    return source.takeError();
  auto inputs = makeNullaryProgramInputs(source->structuredProgram, "main");
  if (!inputs)
    return inputs.takeError();
  const loom::ArtifactIdentity structuredInitialIdentity =
      source->structuredProgram.identity();
  const loom::ArtifactIdentity workloadIdentity = inputs->workload.identity();
  const loom::ArtifactIdentity runtimeInputIdentity =
      inputs->runtimeInput.identity();
  auto protocolRoots = resolveOperatorProtocolRoots(source->structuredProgram);
  if (!protocolRoots)
    return protocolRoots.takeError();
  loom::dse::PreMappingExplorationOptions exploration{
      {compilation.lowering,
       {loom::evaluation::MetricRequestOrdinal(0),
        loom::ResolvedObjectiveDirection::Minimize, 1},
       candidateJobs}};
  exploration.ownership.selectionMode =
      loom::dse::StructuredOwnershipSelectionMode::SemanticConformance;
  exploration.ownership.protocolCallableRoots = std::move(*protocolRoots);
  exploration.ownership.functionalReplayLimits.maxWavefrontSteps =
      maxEventSteps;
  exploration.ownership.functionalReplayLimits.maxEventCount = maxEventCount;
  exploration.ownership.functionalReplayLimits.maxRetainedCaptureBytes =
      maxCaptureBytes;
  exploration.ownership.functionalReplayLimits.maxSimulationWallTime =
      std::chrono::duration_cast<std::chrono::steady_clock::duration>(
          std::chrono::duration<double>(maxSimulationWallSeconds));
  auto outcome = loom::dse::exploreStructuredCompilationToPreMapping(
      std::move(*source), inputs->workload, inputs->runtimeInput, fabric,
      config, exploration, store, blobs);
  if (!outcome)
    return outcome.takeError();
  auto generateSummary = std::visit(
      [&](const auto &result) {
        return loom::dse::validateAndSummarizeDsePlanGenerateInvocations(
            result.planGenerateInvocations, store);
      },
      *outcome);
  if (!generateSummary)
    return generateSummary.takeError();
  if (const auto *incomplete =
          std::get_if<loom::dse::IncompletePreMappingExploration>(&*outcome)) {
    std::string message;
    llvm::raw_string_ostream stream(message);
    stream << "central DSE did not complete: reason="
           << loom::dse::toString(incomplete->reason);
    if (incomplete->planNodeOrdinal)
      stream << ", plan_node=" << *incomplete->planNodeOrdinal;
    stream << ", retained_evidence=" << incomplete->retainedEvidence.size()
           << ", completed_generate_invocations="
           << generateSummary->completedInvocations
           << ", incomplete_generate_invocations="
           << generateSummary->incompleteInvocations;
    return unsupported(stream.str());
  }
  if (std::holds_alternative<loom::dse::CompletedPreMappingNoFeasibleCandidate>(
          *outcome)) {
    std::string message;
    llvm::raw_string_ostream stream(message);
    stream << "central DSE found no feasible ownership candidate after "
           << generateSummary->completedInvocations
           << " completed Generate invocations";
    return unsupported(stream.str());
  }
  auto completed =
      std::get<loom::dse::CompletedPreMappingSelection>(std::move(*outcome));
  if (completed.selected.size() != 1)
    return invalid("TopK(1) did not select exactly one candidate");
  if (generateSummary->completedInvocations == 0 ||
      generateSummary->incompleteInvocations != 0)
    return invalid("completed central DSE has invalid Generate provenance");
  return SourceBackedCompilation{std::move(completed.selected.front()),
                                 structuredInitialIdentity,
                                 workloadIdentity,
                                 runtimeInputIdentity,
                                 std::move(completed.planGenerateInvocations),
                                 *generateSummary};
}

llvm::Expected<loom::ArtifactIdentity> initialCanonicalDataflowIdentity(
    const loom::dse::SelectedPreMappingCompilation &selected) {
  if (selected.dataflowRewriteDerivations.empty())
    return selected.compilation.canonicalDataflow.identity();

  std::vector<loom::ArtifactRootReference> roots;
  bool selectedIsDerived = false;
  for (const loom::dse::DataflowRewriteDerivation &candidate :
       selected.dataflowRewriteDerivations) {
    if (candidate.child.artifact ==
        selected.compilation.canonicalDataflow.identity())
      selectedIsDerived = true;
    bool parentIsDerived = false;
    for (const loom::dse::DataflowRewriteDerivation &other :
         selected.dataflowRewriteDerivations)
      parentIsDerived |= candidate.parent == other.child;
    if (!parentIsDerived && llvm::find(roots, candidate.parent) == roots.end())
      roots.push_back(candidate.parent);
  }
  if (!selectedIsDerived)
    return invalid("selected D* is absent from its rewrite lineage");
  if (roots.size() != 1 ||
      roots.front().schemaIdentity !=
          dataflow::canonicalDataflowSchema.identity ||
      roots.front().schemaVersion != dataflow::canonicalDataflowSchema.version)
    return invalid("rewrite lineage does not have one exact D0 root");
  return roots.front().artifact;
}

llvm::Expected<std::vector<std::string>>
selectedSourceFiles(const loom::dse::SelectedPreMappingCompilation &selected) {
  auto view = selected.compilation.structuredProgram.view();
  if (!view)
    return view.takeError();
  std::vector<std::string> files;
  for (const auto &provenance : selected.compilation.sourceProvenance) {
    auto entity = view->resolve(provenance.operation);
    if (!entity)
      return entity.takeError();
    mlir::Operation *operation = entity->operation;
    if (!operation || (!llvm::isa<::loom::SpatialRegionOp>(operation) &&
                       !operation->getParentOfType<::loom::SpatialRegionOp>()))
      continue;
    files.insert(files.end(), provenance.sourceFiles.begin(),
                 provenance.sourceFiles.end());
  }
  std::sort(files.begin(), files.end());
  files.erase(std::unique(files.begin(), files.end()), files.end());
  if (files.empty())
    return invalid("selected graph has no source provenance");
  return files;
}

llvm::Error
writeReport(llvm::StringRef path,
            llvm::ArrayRef<std::string> selectedSourceFiles,
            const SourceBackedCompilation &compilation,
            const dataflow::CanonicalDataflowProgramView &view,
            const loom::SystemCompilerTargetBindings &compilerTargets,
            const loom::sim::SourceBackedDfgValidationResult &replay) {
  llvm::SmallString<256> parent(path);
  llvm::sys::path::remove_filename(parent);
  if (!parent.empty())
    if (std::error_code error = llvm::sys::fs::create_directories(parent))
      return llvm::createStringError(error, "cannot create %s", parent.c_str());

  llvm::json::Object firings;
  for (const auto &[schema, count] : replay.operationFireCounts)
    firings[dataflow::operationSchemaSpelling(schema)] = count;
  llvm::json::Object root;
  root["kind"] = "source_backed_dfg_comparison";
  root["status"] = "pass";
  root["execution_terminal"] = "retired";
  auto canonicalDataflowInitial =
      initialCanonicalDataflowIdentity(compilation.selected);
  if (!canonicalDataflowInitial)
    return canonicalDataflowInitial.takeError();
  llvm::json::Object artifacts;
  artifacts["canonical_dataflow"] = loom::formatArtifactIdentityHex(
      compilation.selected.compilation.canonicalDataflow.identity());
  artifacts["canonical_dataflow_initial"] =
      loom::formatArtifactIdentityHex(*canonicalDataflowInitial);
  artifacts["simulation_workload"] =
      loom::formatArtifactIdentityHex(compilation.workloadIdentity);
  artifacts["simulation_runtime_input"] =
      loom::formatArtifactIdentityHex(compilation.runtimeInputIdentity);
  artifacts["structured_initial"] =
      loom::formatArtifactIdentityHex(compilation.structuredInitialIdentity);
  artifacts["structured_selected"] = loom::formatArtifactIdentityHex(
      compilation.selected.compilation.structuredProgram.identity());
  root["artifacts"] = std::move(artifacts);
  root["graphs"] = view.graphs().size();
  root["actors"] = view.actors().size();
  llvm::json::Array actorReferences;
  for (const dataflow::CanonicalActorView &actor : view.actors()) {
    llvm::json::Object reference;
    reference["artifact"] = loom::formatArtifactIdentityHex(actor.ref.artifact);
    reference["entity"] = std::to_string(actor.ref.entity.value());
    actorReferences.push_back(std::move(reference));
  }
  root["actor_refs"] = std::move(actorReferences);
  root["dynamic_calls"] = replay.dynamicActivations;
  root["value_lanes_compared"] = replay.valueLanesCompared;
  root["memory_bytes_compared"] = replay.memoryBytesCompared;
  root["floating_variance_bytes"] = 0;
  root["floating_variance_kind"] = "none";
  root["wavefront_steps"] = replay.wavefrontSteps;
  root["event_count"] = replay.eventCount;
  root["simulation_seconds"] = replay.simulationSeconds;
  root["wavefront_steps_per_second"] =
      replay.simulationSeconds > 0.0
          ? static_cast<double>(replay.wavefrontSteps) /
                replay.simulationSeconds
          : 0.0;
  root["operation_firings"] = std::move(firings);
  llvm::json::Object sourceOracle;
  sourceOracle["comparison"] = "equivalent";
  if (replay.sourceReturnValue && replay.sourceReturnValue->tokenCount == 1 &&
      replay.sourceReturnValue->lanes.size() == 1 &&
      replay.sourceReturnValue->lanes.front().state ==
          loom::sim::SemanticState::Defined &&
      replay.sourceReturnValue->lanes.front().bits.getBitWidth() <= 64)
    sourceOracle["entry_result"] =
        replay.sourceReturnValue->lanes.front().bits.getSExtValue();
  else
    sourceOracle["entry_result"] = nullptr;
  root["source_oracle"] = std::move(sourceOracle);

  llvm::json::Object transformLineage;
  transformLineage["ownership"] = compilation.selected.derivations.size();
  transformLineage["execution_shape"] =
      compilation.selected.executionShapeDerivations.size();
  transformLineage["special_math_accuracy"] =
      compilation.selected.specialMathAccuracyDerivations.size();
  transformLineage["schedule"] =
      compilation.selected.scheduleDerivations.size();
  llvm::json::Array memoryCommunication;
  for (const loom::dse::StructuredMemoryCommunicationDerivation &derivation :
       compilation.selected.memoryCommunicationDerivations)
    memoryCommunication.push_back(static_cast<std::int64_t>(
        loom::frontend::structuredMemoryCommunicationDecisionKind(
            derivation.decision)));
  transformLineage["memory_communication"] = std::move(memoryCommunication);
  llvm::json::Array dataflowRewrite;
  for (const loom::dse::DataflowRewriteDerivation &derivation :
       compilation.selected.dataflowRewriteDerivations)
    dataflowRewrite.push_back(static_cast<std::int64_t>(
        dataflow::dataflowRewriteKind(derivation.decision)));
  transformLineage["dataflow_rewrite"] = std::move(dataflowRewrite);
  root["transform_lineage"] = std::move(transformLineage);

  const auto &planGenerateSummary = compilation.planGenerateSummary;
  if (planGenerateSummary.completedInvocations == 0 ||
      planGenerateSummary.incompleteInvocations != 0)
    return invalid("completed DSE retained no Generate provenance");
  llvm::json::Object dseExecution;
  dseExecution["plan_executions"] = planGenerateSummary.planExecutions;
  dseExecution["generate_invocations"] =
      planGenerateSummary.completedInvocations;
  dseExecution["incomplete_generate_invocations"] =
      planGenerateSummary.incompleteInvocations;
  dseExecution["input_bindings"] = planGenerateSummary.inputBindings;
  dseExecution["input_artifacts"] = planGenerateSummary.inputArtifacts;
  dseExecution["output_bindings"] = planGenerateSummary.outputBindings;
  dseExecution["output_artifacts"] = planGenerateSummary.outputArtifacts;
  dseExecution["generate_lineage_edges"] = planGenerateSummary.lineageEdges;
  root["dse_execution"] = std::move(dseExecution);
  llvm::json::Object target;
  target["host_binding"] = loom::formatArtifactIdentityHex(
      compilerTargets.host().reference().artifact);
  llvm::json::Array instructionBindings;
  std::uint64_t instructionCoreCount = 0;
  for (const auto &group : compilerTargets.instructionGroups()) {
    instructionBindings.push_back(
        loom::formatArtifactIdentityHex(group.binding().reference().artifact));
    instructionCoreCount += group.processors().size();
  }
  target["instruction_bindings"] = std::move(instructionBindings);
  target["instruction_core_count"] = instructionCoreCount;
  const auto &instruction =
      compilerTargets.instructionGroups().front().binding().binding();
  target["target_triple"] = instruction.targetTriple();
  target["data_layout"] = instruction.dataLayout();
  root["compiler_target"] = std::move(target);
  llvm::json::Array sourceFiles;
  for (const std::string &file : selectedSourceFiles)
    sourceFiles.push_back(file);
  root["selected_source_files"] = std::move(sourceFiles);

  std::error_code error;
  llvm::raw_fd_ostream output(path, error, llvm::sys::fs::OF_Text);
  if (error)
    return llvm::createStringError(error, "cannot open %s", path.str().c_str());
  output << llvm::formatv("{0:2}", llvm::json::Value(std::move(root))) << '\n';
  return llvm::Error::success();
}

llvm::Error
writeCanonicalDataflow(llvm::StringRef path,
                       const dataflow::CanonicalDataflowArtifact &canonical) {
  if (path.empty())
    return llvm::Error::success();
  std::string message;
  std::unique_ptr<llvm::ToolOutputFile> output =
      mlir::openOutputFile(path, &message);
  if (!output)
    return invalid("cannot open canonical output: " + message);
  canonical.module()->print(output->os());
  output->os() << '\n';
  output->keep();
  return llvm::Error::success();
}

} // namespace

int main(int argc, char **argv) {
  llvm::InitLLVM init(argc, argv);
  llvm::cl::ParseCommandLineOptions(
      argc, argv,
      "Compile one exact LLVM program through central Pre-Mapping DSE and "
      "report its source-backed DFG semantic replay.\n");
  if (candidateJobs == 0)
    return reportError(invalid("candidate-jobs must be positive"));
  if (maxEventSteps == 0)
    return reportError(invalid("max-event-steps must be positive"));
  if (maxEventCount == 0)
    return reportError(invalid("max-event-count must be positive"));
  if (maxCaptureBytes == 0)
    return reportError(invalid("max-capture-bytes must be positive"));
  if (!std::isfinite(maxSimulationWallSeconds) ||
      maxSimulationWallSeconds <= 0.0)
    return reportError(
        invalid("max-simulation-wall-seconds must be finite and positive"));

  llvm::Expected<loom::ResolvedConfig> config =
      loom::resolveConfigProfile(accelerationProfile);
  if (!config)
    return reportError(config.takeError());
  loom::ArtifactStore store(artifactStorePath);
  llvm::SmallString<128> blobPath(artifactStorePath);
  llvm::sys::path::append(blobPath, "blobs");
  if (std::error_code error = llvm::sys::fs::create_directories(blobPath))
    return reportError(
        invalid("cannot create BlobStore directory: " + error.message()));
  const loom::BlobStore blobs(blobPath);
  auto design = loom::adg::buildBuiltinTarget(
      store, config->hardwareTarget.templateIdentity,
      config->hardwareTarget.schemaVersion.major,
      config->hardwareTarget.schemaVersion.minor,
      config->hardwareTarget.parameters);
  if (!design)
    return reportError(design.takeError());
  if (design->roots().size() != 1)
    return reportError(invalid("builtin target has no unique Fabric root"));

  llvm::LLVMContext targetContext;
  auto target = readModule(targetContext, targetModulePath);
  if (!target)
    return reportError(target.takeError());
  const loom::CompilerTargetPolicy targetPolicy =
      loom::portableRiscV64CompilerTargetPolicy();
  auto compilerTargets = loom::resolveSystemCompilerTargetBindings(
      design->roots().front(), targetPolicy, store);
  if (!compilerTargets)
    return reportError(compilerTargets.takeError());
  for (const auto &group : compilerTargets->instructionGroups())
    if (llvm::Error error = loom::validateModuleCompilerTarget(
            **target, group.binding().binding()))
      return reportError(std::move(error));
  auto selected = compileTarget(std::move(*target), design->roots().front(),
                                *config, store, blobs);
  if (!selected)
    return reportError(selected.takeError());
  auto view = selected->selected.compilation.canonicalDataflow.view();
  if (!view)
    return reportError(view.takeError());
  if (llvm::Error error = writeCanonicalDataflow(
          canonicalOutputPath,
          selected->selected.compilation.canonicalDataflow))
    return reportError(std::move(error));
  if (view->graphs().empty())
    return reportError(unsupported("selected program is graph-free"));
  if (!selected->selected.functionalReplay ||
      selected->selected.functionalReplay->status !=
          loom::sim::SourceBackedDfgValidationStatus::Equivalent)
    return reportError(
        invalid("selected graph has no equivalent functional replay"));
  const auto &replay = *selected->selected.functionalReplay;
  if (llvm::Error error = requireExpectedEntryResult(replay))
    return reportError(std::move(error));
  if (replay.dynamicActivations == 0 ||
      (replay.valueLanesCompared == 0 && replay.memoryBytesCompared == 0) ||
      replay.eventCount == 0 || replay.operationFireCounts.empty())
    return reportError(invalid("execution produced no substantive workload"));
  auto sourceFiles = selectedSourceFiles(selected->selected);
  if (!sourceFiles)
    return reportError(sourceFiles.takeError());
  if (llvm::Error error = writeReport(outputPath, *sourceFiles, *selected,
                                      *view, *compilerTargets, replay))
    return reportError(std::move(error));
  return 0;
}
