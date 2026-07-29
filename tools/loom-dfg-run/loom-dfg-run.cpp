#include "ADG/Builtin.h"
#include "Common/ArtifactStore.h"
#include "Common/ArtifactText.h"
#include "Common/ResolvedConfig.h"
#include "DSE/PreMappingExploration.h"
#include "Dataflow/IR/OperationSchema.h"
#include "Frontend/Raising/StructuredRaising.h"
#include "Simulator/SimulationArtifacts.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Support/FileUtilities.h"
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

#include <cstdint>
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
    builtinName("builtin", llvm::cl::desc("builtin Fabric target preset"),
                llvm::cl::value_desc("small|default|large"),
                llvm::cl::Required);

llvm::cl::opt<std::string>
    artifactStorePath("artifact-store",
                      llvm::cl::desc("ArtifactStore directory"),
                      llvm::cl::value_desc("path"), llvm::cl::Required);

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

llvm::cl::opt<std::uint64_t>
    maxEventSteps("max-event-steps",
                  llvm::cl::desc("maximum aggregate DFG event wavefronts"),
                  llvm::cl::init(100000));

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

llvm::Expected<loom::dse::SelectedPreMappingCompilation>
compileTarget(std::unique_ptr<llvm::Module> module,
              const loom::fabric::FinalizedFabricRoot &fabric,
              const loom::ArtifactStore &store) {
  loom::frontend::PreMappingCompilationOptions compilation;
  auto source = loom::frontend::raiseLlvmModuleToStructured(
      std::move(module), fabric, compilation.raising);
  if (!source)
    return source.takeError();
  auto inputs = makeNullaryProgramInputs(source->structuredProgram, "main");
  if (!inputs)
    return inputs.takeError();
  loom::dse::PreMappingExplorationOptions exploration{
      {compilation.lowering,
       {loom::evaluation::MetricRequestOrdinal(0),
        loom::dse::ObjectiveDirection::Minimize, 1},
       candidateJobs}};
  exploration.ownership.functionalReplayLimits.maxWavefrontSteps =
      maxEventSteps;
  auto outcome = loom::dse::exploreStructuredCompilationToPreMapping(
      std::move(*source), inputs->workload, inputs->runtimeInput, fabric,
      loom::defaultResolvedConfig(), exploration, store);
  if (!outcome)
    return outcome.takeError();
  if (const auto *incomplete =
          std::get_if<loom::dse::IncompleteSelection>(&*outcome)) {
    std::string message;
    llvm::raw_string_ostream stream(message);
    stream << "central DSE did not complete: reason="
           << loom::dse::toString(incomplete->reason) << ", candidate="
           << loom::formatArtifactIdentityHex(incomplete->candidate.artifact)
           << ", retained_evidence=" << incomplete->retainedEvidence.size();
    return unsupported(stream.str());
  }
  if (std::holds_alternative<loom::dse::CompletedNoFeasibleCandidate>(*outcome))
    return unsupported("central DSE found no feasible ownership candidate");
  auto completed =
      std::get<loom::dse::CompletedPreMappingSelection>(std::move(*outcome));
  if (completed.selected.size() != 1)
    return invalid("TopK(1) did not select exactly one candidate");
  return std::move(completed.selected.front());
}

llvm::Error
writeReport(llvm::StringRef path, std::uint64_t graphCount,
            std::uint64_t actorCount,
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
  root["graphs"] = graphCount;
  root["actors"] = actorCount;
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

  auto preset = loom::adg::parseBuiltinTargetPreset(builtinName);
  if (!preset)
    return reportError(preset.takeError());
  loom::ArtifactStore store(artifactStorePath);
  auto design = loom::adg::buildBuiltinTarget(store, *preset);
  if (!design)
    return reportError(design.takeError());
  if (design->roots().size() != 1)
    return reportError(invalid("builtin target has no unique Fabric root"));

  llvm::LLVMContext targetContext;
  auto target = readModule(targetContext, targetModulePath);
  if (!target)
    return reportError(target.takeError());
  auto selected =
      compileTarget(std::move(*target), design->roots().front(), store);
  if (!selected)
    return reportError(selected.takeError());
  auto view = selected->compilation.canonicalDataflow.view();
  if (!view)
    return reportError(view.takeError());
  if (llvm::Error error = writeCanonicalDataflow(
          canonicalOutputPath, selected->compilation.canonicalDataflow))
    return reportError(std::move(error));
  if (view->graphs().empty())
    return reportError(unsupported("selected program is graph-free"));
  if (!selected->functionalReplay ||
      selected->functionalReplay->status !=
          loom::sim::SourceBackedDfgValidationStatus::Equivalent)
    return reportError(
        invalid("selected graph has no equivalent functional replay"));
  const auto &replay = *selected->functionalReplay;
  if (replay.dynamicActivations == 0 ||
      (replay.valueLanesCompared == 0 && replay.memoryBytesCompared == 0) ||
      replay.eventCount == 0 || replay.operationFireCounts.empty())
    return reportError(invalid("execution produced no substantive workload"));
  if (llvm::Error error = writeReport(outputPath, view->graphs().size(),
                                      view->actors().size(), replay))
    return reportError(std::move(error));
  return 0;
}
