// loom-pre-mapping: read an LLVM IR (.ll / .bc) file via parseIRFile, resolve
// one exact builtin Fabric target through an existing ArtifactStore, and run
// the production pre-Mapping compilation libraries. Without an explicit
// ownership selection, central Generate/Evaluate/Promote explores Structured
// candidates against the exact Fabric. Focused explicit selections bypass
// ranking but use the same mechanical boundaries and candidate materializer.
//
// CLI shape mirrors loom-raise / loom-adg:
//
//     loom-pre-mapping --artifact-store=<dir>
//                      [--loom-accel-profile=<preset-or-path>]
//                      --counts=<counts.json>
//                      [--whole-callable-spatial=<symbol>]
//                      [--operation-spatial=<symbol>
//                       --operation-spatial-scope-index=<index>
//                       (--canonical-index-width=<bits> |
//                        --pointer-addressed)]
//                      [--fmuladd-shape=fused|split]
//                      [--candidate-jobs=<positive count>]
//                      [-o output.mlir] input.ll
//
// stdin is read when input is "-" or the positional arg is missing.
// The finalized Canonical Dataflow module is printed to -o (stdout by
// default). Graph and actor counts from the imported canonical view are
// written as one JSON object, so a graph-free whole-program result is
// distinguishable from a nonempty Spatial graph.

#include "ADG/Builtin.h"
#include "Common/ArtifactStore.h"
#include "Common/ArtifactText.h"
#include "Common/BlobStore.h"
#include "Config/ResolvedConfig.h"
#include "DSE/PreMappingExploration.h"
#include "Frontend/Compilation/OwnershipCandidateGenerator.h"
#include "Frontend/Compilation/PreMappingCompilation.h"
#include "Frontend/Compilation/StructuredExecutionShape.h"
#include "Simulator/SimulationArtifacts.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/AsmState.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

#include <cinttypes>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <system_error>
#include <vector>

namespace {

::llvm::cl::opt<std::string>
    inputFilename(::llvm::cl::Positional,
                  ::llvm::cl::desc("<input LLVM IR (.ll/.bc), or - for stdin>"),
                  ::llvm::cl::init("-"));

::llvm::cl::opt<std::string> outputFilename("o",
                                            ::llvm::cl::desc("Output filename"),
                                            ::llvm::cl::value_desc("filename"),
                                            ::llvm::cl::init("-"));

::llvm::cl::opt<std::string>
    artifactStorePath("artifact-store",
                      ::llvm::cl::desc("existing ArtifactStore directory"),
                      ::llvm::cl::value_desc("path"), ::llvm::cl::Required);

::llvm::cl::opt<std::string> accelerationProfile(
    "loom-accel-profile",
    ::llvm::cl::desc("builtin acceleration preset or configuration path"),
    ::llvm::cl::value_desc("preset-or-path"), ::llvm::cl::init(""));

::llvm::cl::opt<std::string>
    countsFilename("counts",
                   ::llvm::cl::desc("output path for structured graph/actor "
                                    "counts as one JSON object"),
                   ::llvm::cl::value_desc("filename"), ::llvm::cl::Required);

::llvm::cl::opt<std::string> rootReferenceFilename(
    "root-reference",
    ::llvm::cl::desc("optional canonical Dataflow root-reference JSON output"),
    ::llvm::cl::value_desc("filename"), ::llvm::cl::init(""));

::llvm::cl::opt<std::string> fabricRootReferenceFilename(
    "fabric-root-reference",
    ::llvm::cl::desc("optional exact Fabric root-reference JSON output"),
    ::llvm::cl::value_desc("filename"), ::llvm::cl::init(""));

::llvm::cl::opt<std::string> applicationEntry(
    "application-entry",
    ::llvm::cl::desc("defined LLVM ABI entry that owns the application scope"),
    ::llvm::cl::value_desc("symbol"), ::llvm::cl::init("main"));

::llvm::cl::opt<std::string> applicationWorkloadSetFilename(
    "application-workload-set",
    ::llvm::cl::desc("optional canonical Spatial workload-root set output"),
    ::llvm::cl::value_desc("filename"), ::llvm::cl::init(""));

::llvm::cl::opt<std::string> wholeCallableSpatial(
    "whole-callable-spatial",
    ::llvm::cl::desc("materialize one exact LLVM callable as an "
                     "explicit whole-callable Spatial ownership candidate"),
    ::llvm::cl::value_desc("symbol"), ::llvm::cl::init(""));

::llvm::cl::opt<std::string> operationSpatial(
    "operation-spatial",
    ::llvm::cl::desc("materialize one structured operation in an LLVM "
                     "callable as an explicit Spatial candidate"),
    ::llvm::cl::value_desc("callable-symbol"), ::llvm::cl::init(""));

::llvm::cl::opt<std::uint64_t> operationSpatialScopeIndex(
    "operation-spatial-scope-index",
    ::llvm::cl::desc("zero-based index in the callable's canonical eligible "
                     "Spatial ownership scope enumeration"),
    ::llvm::cl::value_desc("index"));

::llvm::cl::opt<unsigned> candidateJobs(
    "candidate-jobs",
    ::llvm::cl::desc("parallel ownership candidate workers; affects execution "
                     "time only"),
    ::llvm::cl::value_desc("count"), ::llvm::cl::init(1));

::llvm::cl::list<std::string> operatorProtocolSymbols(
    "operator-protocol-symbol",
    ::llvm::cl::desc("defined LLVM callable that roots the invocation-local "
                     "operator protocol ownership domain"),
    ::llvm::cl::value_desc("symbol"), ::llvm::cl::ZeroOrMore);

::llvm::cl::opt<loom::dse::StructuredOwnershipSelectionMode>
    ownershipSelectionMode(
        "ownership-selection-mode",
        ::llvm::cl::desc("pre-Mapping ownership selection intent"),
        ::llvm::cl::values(
            clEnumValN(
                loom::dse::StructuredOwnershipSelectionMode::BenefitQualified,
                "benefit-qualified",
                "rank independently materialized ownership alternatives"),
            clEnumValN(loom::dse::StructuredOwnershipSelectionMode::
                           SemanticConformance,
                       "semantic-conformance",
                       "rank a bounded chain of equivalent protocol closures")),
        ::llvm::cl::init(
            loom::dse::StructuredOwnershipSelectionMode::BenefitQualified));

::llvm::cl::opt<unsigned> canonicalIndexWidth(
    "canonical-index-width",
    ::llvm::cl::desc("explicit canonical index width materialized for a "
                     "selected Spatial candidate"),
    ::llvm::cl::value_desc("bits"), ::llvm::cl::init(0));

::llvm::cl::opt<bool> pointerAddressed(
    "pointer-addressed",
    ::llvm::cl::desc("retain exact LLVM pointer addressing in a selected "
                     "Spatial candidate"),
    ::llvm::cl::init(false));

enum class FMulAddShapeOption { Unspecified, Fused, Split };

::llvm::cl::opt<FMulAddShapeOption> fmuladdShape(
    "fmuladd-shape",
    ::llvm::cl::desc("typed execution-shape decision for selected "
                     "llvm.intr.fmuladd operations"),
    ::llvm::cl::values(clEnumValN(FMulAddShapeOption::Fused, "fused",
                                  "one fused math.fma"),
                       clEnumValN(FMulAddShapeOption::Split, "split",
                                  "separate arith.mulf and arith.addf")),
    ::llvm::cl::init(FMulAddShapeOption::Unspecified));

int reportError(::llvm::Error error) {
  ::llvm::errs() << "loom-pre-mapping: " << ::llvm::toString(std::move(error))
                 << "\n";
  return 1;
}

::llvm::Expected<loom::frontend::StructuredEntityRef>
resolveCallable(const loom::frontend::StructuredProgramCandidate &candidate,
                ::llvm::StringRef symbol) {
  auto resolved =
      loom::frontend::resolveDefinedLlvmCallables(candidate, {symbol});
  if (!resolved)
    return resolved.takeError();
  return resolved->front();
}

::llvm::Expected<std::vector<loom::frontend::StructuredEntityRef>>
resolveOperatorProtocolRoots(
    const loom::frontend::StructuredProgramCandidate &candidate) {
  ::llvm::SmallVector<::llvm::StringRef> symbols;
  symbols.reserve(operatorProtocolSymbols.size());
  for (const std::string &symbol : operatorProtocolSymbols) {
    symbols.push_back(symbol);
  }
  return loom::frontend::resolveDefinedLlvmCallables(candidate, symbols);
}

struct NullaryProgramInputs final {
  loom::sim::CanonicalSimulationWorkload workload;
  loom::sim::CanonicalSimulationRuntimeInput runtimeInput;
};

::llvm::Expected<NullaryProgramInputs> makeNullaryProgramInputs(
    const loom::frontend::StructuredProgramCandidate &candidate,
    ::llvm::StringRef symbol) {
  auto entry = resolveCallable(candidate, symbol);
  if (!entry)
    return entry.takeError();
  auto view = candidate.view();
  if (!view)
    return view.takeError();
  auto entity = view->resolve(*entry);
  if (!entity)
    return entity.takeError();
  auto function =
      ::llvm::dyn_cast_or_null<::mlir::LLVM::LLVMFuncOp>(entity->operation);
  if (!function || function.isVarArg() ||
      function.getFunctionType().getNumParams() != 0)
    return ::llvm::createStringError(
        std::make_error_code(std::errc::not_supported),
        "focused pre-Mapping DSE requires a nullary program entry");

  loom::sim::StructuredProgramSimulationWorkload workloadDraft{*entry};
  workloadDraft.observableContract.returnValue =
      !::llvm::isa<::mlir::LLVM::LLVMVoidType>(
          function.getFunctionType().getReturnType());
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

::llvm::Expected<loom::frontend::StructuredEntityRef>
resolveUniqueStructuredOperation(
    const loom::frontend::StructuredProgramCandidate &candidate,
    ::llvm::StringRef callableSymbol, std::optional<std::uint64_t> scopeIndex) {
  auto view = candidate.view();
  if (!view)
    return view.takeError();
  auto domain = loom::frontend::enumerateSpatialOwnershipScopeDomain(candidate);
  if (!domain)
    return domain.takeError();
  std::vector<loom::frontend::StructuredEntityRef> callableScopes;
  for (const loom::frontend::SpatialOwnershipScopeDomainEntry &entry :
       *domain) {
    const auto *scope =
        std::get_if<loom::frontend::SpatialOwnershipScope>(&entry);
    if (!scope)
      continue;
    auto entity = view->resolve(scope->selection);
    if (!entity)
      return entity.takeError();
    ::mlir::Operation *operation = entity->operation;
    if (!operation || ::llvm::isa<::mlir::LLVM::LLVMFuncOp>(operation))
      continue;
    auto callable = operation->getParentOfType<::mlir::LLVM::LLVMFuncOp>();
    if (!callable || callable.getSymName() != callableSymbol)
      continue;
    callableScopes.push_back(scope->selection);
  }
  if (callableScopes.empty())
    return ::llvm::createStringError(
        ::llvm::inconvertibleErrorCode(),
        "callable contains no eligible Spatial ownership scope");
  if (scopeIndex) {
    if (*scopeIndex >= callableScopes.size())
      return ::llvm::createStringError(
          ::llvm::inconvertibleErrorCode(),
          "operation Spatial scope index is out of range [0, %zu)",
          callableScopes.size());
    return callableScopes[*scopeIndex];
  }
  if (callableScopes.size() != 1)
    return ::llvm::createStringError(
        ::llvm::inconvertibleErrorCode(),
        "callable has %zu eligible Spatial ownership scopes; select one with "
        "--operation-spatial-scope-index",
        callableScopes.size());
  return callableScopes.front();
}

std::unique_ptr<::llvm::Module> readLLVMModule(::llvm::LLVMContext &llvmContext,
                                               const std::string &filename) {
  ::llvm::SMDiagnostic err;
  if (filename == "-") {
    auto buffer = ::llvm::MemoryBuffer::getSTDIN();
    if (auto bufErr = buffer.getError()) {
      ::llvm::errs() << "loom-pre-mapping: cannot read stdin: "
                     << bufErr.message() << "\n";
      return nullptr;
    }
    auto module =
        ::llvm::parseIR((*buffer)->getMemBufferRef(), err, llvmContext);
    if (!module) {
      err.print("loom-pre-mapping", ::llvm::errs());
      return nullptr;
    }
    return module;
  }
  auto module = ::llvm::parseIRFile(filename, err, llvmContext);
  if (!module) {
    err.print("loom-pre-mapping", ::llvm::errs());
    return nullptr;
  }
  return module;
}

} // namespace

int main(int argc, char **argv) {
  ::llvm::InitLLVM y(argc, argv);
  ::mlir::registerAsmPrinterCLOptions();
  ::mlir::registerMLIRContextCLOptions();
  ::mlir::registerPassManagerCLOptions();
  ::llvm::cl::ParseCommandLineOptions(
      argc, argv,
      "Loom LLVM IR -> pre-Mapping compilation driver\n"
      "Resolves one exact builtin Fabric target through an existing "
      "ArtifactStore. The default path runs central Structured "
      "Generate/Evaluate/Promote between mechanical LLVM raising and "
      "Dataflow lowering; explicit ownership flags materialize one focused "
      "candidate. Emits the finalized Canonical Dataflow module and "
      "structured graph/actor counts.\n");

  ::llvm::Expected<loom::ResolvedConfig> config =
      loom::resolveConfigProfile(accelerationProfile);
  if (!config)
    return reportError(config.takeError());

  // Parse LLVM IR.
  ::llvm::LLVMContext llvmContext;
  std::unique_ptr<::llvm::Module> llvmModule =
      readLLVMModule(llvmContext, inputFilename);
  if (!llvmModule)
    return 1;

  // Resolve the exact builtin Fabric target through its owner.
  loom::ArtifactStore store(artifactStorePath);
  ::llvm::SmallString<128> blobPath(artifactStorePath);
  ::llvm::sys::path::append(blobPath, "blobs");
  if (std::error_code error = ::llvm::sys::fs::create_directories(blobPath))
    return reportError(::llvm::createStringError(
        error, "cannot create BlobStore directory: %s", blobPath.c_str()));
  const loom::BlobStore blobs(blobPath);
  auto design = loom::adg::buildBuiltinTarget(
      store, config->hardwareTarget.templateIdentity,
      config->hardwareTarget.schemaVersion.major,
      config->hardwareTarget.schemaVersion.minor,
      config->hardwareTarget.parameters);
  if (!design)
    return reportError(design.takeError());
  if (design->roots().size() != 1)
    return reportError(::llvm::createStringError(
        ::llvm::inconvertibleErrorCode(),
        "builtin target did not produce exactly one Fabric root"));

  const bool hasExplicitSelection =
      !wholeCallableSpatial.empty() || !operationSpatial.empty();

  if (!wholeCallableSpatial.empty() && !operationSpatial.empty())
    return reportError(::llvm::createStringError(
        ::llvm::inconvertibleErrorCode(),
        "whole-callable and operation Spatial selections are exclusive"));
  if (canonicalIndexWidth != 0 && !hasExplicitSelection)
    return reportError(::llvm::createStringError(
        ::llvm::inconvertibleErrorCode(),
        "canonical index width requires a Spatial selection"));
  if (pointerAddressed && !hasExplicitSelection)
    return reportError(::llvm::createStringError(
        ::llvm::inconvertibleErrorCode(),
        "pointer-addressed projection requires a Spatial selection"));
  if (pointerAddressed && canonicalIndexWidth != 0)
    return reportError(::llvm::createStringError(
        ::llvm::inconvertibleErrorCode(),
        "pointer-addressed and canonical index width are exclusive"));
  if (operationSpatialScopeIndex.getNumOccurrences() != 0 &&
      operationSpatial.empty())
    return reportError(::llvm::createStringError(
        ::llvm::inconvertibleErrorCode(),
        "operation Spatial scope index requires an operation selection"));
  if (fmuladdShape != FMulAddShapeOption::Unspecified && !hasExplicitSelection)
    return reportError(::llvm::createStringError(
        ::llvm::inconvertibleErrorCode(),
        "fmuladd shape requires a Spatial selection"));
  if (candidateJobs == 0)
    return reportError(
        ::llvm::createStringError(::llvm::inconvertibleErrorCode(),
                                  "candidate worker count must be positive"));

  loom::frontend::PreMappingCompilationOptions compilationOptions;
  compilationOptions.raising.applyPassManagerCommandLineOptions = true;
  compilationOptions.lowering.applyPassManagerCommandLineOptions = true;

  std::optional<loom::frontend::PreMappingCompilation> compiled;
  std::vector<loom::dse::DsePlanGenerateInvocationRecords>
      planGenerateInvocations;
  loom::dse::DsePlanGenerateInvocationSummary planGenerateSummary;
  bool planSearchComplete = true;
  std::optional<loom::frontend::StructuredCompilation> explicitInput;
  std::optional<loom::frontend::MaterializedOwnershipCandidate> selected;
  if (!hasExplicitSelection) {
    auto source = loom::frontend::raiseLlvmModuleToStructured(
        std::move(llvmModule), design->roots().front(),
        compilationOptions.raising);
    if (!source)
      return reportError(source.takeError());
    auto inputs = makeNullaryProgramInputs(source->structuredProgram, "main");
    if (!inputs)
      return reportError(inputs.takeError());
    auto protocolRoots =
        resolveOperatorProtocolRoots(source->structuredProgram);
    if (!protocolRoots)
      return reportError(protocolRoots.takeError());
    loom::dse::PreMappingExplorationOptions exploration{
        {compilationOptions.lowering,
         {loom::evaluation::MetricRequestOrdinal(0),
          loom::ResolvedObjectiveDirection::Minimize, 1},
         candidateJobs}};
    exploration.ownership.selectionMode = ownershipSelectionMode;
    exploration.ownership.protocolCallableRoots = std::move(*protocolRoots);
    auto outcome = loom::dse::exploreStructuredCompilationToPreMapping(
        std::move(*source), inputs->workload, inputs->runtimeInput,
        design->roots().front(), *config, exploration, store, blobs);
    if (!outcome)
      return reportError(outcome.takeError());
    auto generateSummary = std::visit(
        [&](const auto &result) {
          return loom::dse::validateAndSummarizeDsePlanGenerateInvocations(
              result.planGenerateInvocations, store);
        },
        *outcome);
    if (!generateSummary)
      return reportError(generateSummary.takeError());
    if (const auto *incomplete =
            std::get_if<loom::dse::IncompletePreMappingExploration>(
                &*outcome)) {
      const std::string location =
          incomplete->planNodeOrdinal
              ? " at plan node " + std::to_string(*incomplete->planNodeOrdinal)
              : " before plan execution";
      return reportError(::llvm::createStringError(
          ::llvm::inconvertibleErrorCode(),
          "central DSE is incomplete%s: %s; completed Generate "
          "invocations=%" PRIu64 ", incomplete Generate invocations=%" PRIu64,
          location.c_str(),
          loom::dse::toString(incomplete->reason).str().c_str(),
          generateSummary->completedInvocations,
          generateSummary->incompleteInvocations));
    }
    if (std::holds_alternative<
            loom::dse::CompletedPreMappingNoFeasibleCandidate>(*outcome))
      return reportError(::llvm::createStringError(
          ::llvm::inconvertibleErrorCode(),
          "central DSE completed without a feasible candidate after %" PRIu64
          " Generate invocations",
          generateSummary->completedInvocations));
    auto &completion =
        std::get<loom::dse::CompletedPreMappingSelection>(*outcome);
    if (completion.selected.size() != 1)
      return reportError(::llvm::createStringError(
          ::llvm::inconvertibleErrorCode(),
          "TopK(1) returned %zu pre-Mapping candidates",
          completion.selected.size()));
    planSearchComplete = completion.searchComplete();
    if (planSearchComplete && generateSummary->incompleteInvocations != 0)
      return reportError(::llvm::createStringError(
          ::llvm::inconvertibleErrorCode(),
          "exhaustive central DSE retained incomplete Generate work"));
    compiled.emplace(std::move(completion.selected.front().compilation));
    planGenerateInvocations = std::move(completion.planGenerateInvocations);
    planGenerateSummary = *generateSummary;
  } else {
    auto mechanical = loom::frontend::raiseLlvmModuleToStructured(
        std::move(llvmModule), design->roots().front().reference(), store,
        compilationOptions.raising);
    if (!mechanical)
      return reportError(mechanical.takeError());
    explicitInput.emplace(std::move(*mechanical));
  }

  if (!wholeCallableSpatial.empty()) {
    auto callable =
        resolveCallable(explicitInput->structuredProgram, wholeCallableSpatial);
    if (!callable)
      return reportError(callable.takeError());
    loom::frontend::SpatialOwnershipOptions ownershipOptions;
    ownershipOptions.lowering = compilationOptions.lowering;
    if (canonicalIndexWidth != 0)
      ownershipOptions.addressProjection =
          loom::frontend::RootRelativeAddressProjection{canonicalIndexWidth};
    else if (pointerAddressed)
      ownershipOptions.addressProjection =
          loom::frontend::PointerAddressedAddressProjection{};
    loom::frontend::SpatialOwnershipDecisionPoint ownershipDecision{
        ownershipOptions.addressProjection,
        ownershipOptions.forallOwnershipShape,
        ownershipOptions.directCallSpecializationShape};
    auto owned = loom::frontend::materializeStructuredSpatialOwnershipDecision(
        explicitInput->structuredProgram, {*callable}, ownershipDecision);
    if (!owned)
      return reportError(owned.takeError());
    auto executionShapes =
        loom::frontend::enumerateStructuredExecutionShapeDecisions(
            owned->structuredProgram);
    if (!executionShapes)
      return reportError(executionShapes.takeError());
    if (executionShapes->empty() &&
        fmuladdShape != FMulAddShapeOption::Unspecified)
      return reportError(::llvm::createStringError(
          ::llvm::inconvertibleErrorCode(),
          "fmuladd shape was supplied for a selection with no unresolved "
          "fmuladd"));
    if (!executionShapes->empty()) {
      if (fmuladdShape == FMulAddShapeOption::Unspecified)
        return reportError(::llvm::createStringError(
            ::llvm::inconvertibleErrorCode(),
            "selected Spatial region requires --fmuladd-shape"));
      const loom::raising::FMulAddExecutionShape shape =
          fmuladdShape == FMulAddShapeOption::Fused
              ? loom::raising::FMulAddExecutionShape::Fused
              : loom::raising::FMulAddExecutionShape::Split;
      auto shaped = loom::frontend::materializeStructuredExecutionShapeDecision(
          std::move(*owned), {shape});
      if (!shaped)
        return reportError(shaped.takeError());
      owned = std::move(*shaped);
    }
    auto materialized = loom::frontend::finalizeSpatialOwnershipCandidate(
        std::move(*owned), design->roots().front(), ownershipOptions.lowering);
    if (!materialized)
      return reportError(materialized.takeError());
    selected.emplace(std::move(*materialized));
  } else if (!operationSpatial.empty()) {
    std::optional<std::uint64_t> scopeIndex;
    if (operationSpatialScopeIndex.getNumOccurrences() != 0)
      scopeIndex = operationSpatialScopeIndex;
    auto operation = resolveUniqueStructuredOperation(
        explicitInput->structuredProgram, operationSpatial, scopeIndex);
    if (!operation)
      return reportError(operation.takeError());
    loom::frontend::SpatialOwnershipOptions ownershipOptions;
    ownershipOptions.lowering = compilationOptions.lowering;
    if (canonicalIndexWidth != 0)
      ownershipOptions.addressProjection =
          loom::frontend::RootRelativeAddressProjection{canonicalIndexWidth};
    else if (pointerAddressed)
      ownershipOptions.addressProjection =
          loom::frontend::PointerAddressedAddressProjection{};
    loom::frontend::SpatialOwnershipDecisionPoint ownershipDecision{
        ownershipOptions.addressProjection,
        ownershipOptions.forallOwnershipShape,
        ownershipOptions.directCallSpecializationShape};
    auto owned = loom::frontend::materializeStructuredSpatialOwnershipDecision(
        explicitInput->structuredProgram, {*operation}, ownershipDecision);
    if (!owned)
      return reportError(owned.takeError());
    auto executionShapes =
        loom::frontend::enumerateStructuredExecutionShapeDecisions(
            owned->structuredProgram);
    if (!executionShapes)
      return reportError(executionShapes.takeError());
    if (executionShapes->empty() &&
        fmuladdShape != FMulAddShapeOption::Unspecified)
      return reportError(::llvm::createStringError(
          ::llvm::inconvertibleErrorCode(),
          "fmuladd shape was supplied for a selection with no unresolved "
          "fmuladd"));
    if (!executionShapes->empty()) {
      if (fmuladdShape == FMulAddShapeOption::Unspecified)
        return reportError(::llvm::createStringError(
            ::llvm::inconvertibleErrorCode(),
            "selected Spatial region requires --fmuladd-shape"));
      const loom::raising::FMulAddExecutionShape shape =
          fmuladdShape == FMulAddShapeOption::Fused
              ? loom::raising::FMulAddExecutionShape::Fused
              : loom::raising::FMulAddExecutionShape::Split;
      auto shaped = loom::frontend::materializeStructuredExecutionShapeDecision(
          std::move(*owned), {shape});
      if (!shaped)
        return reportError(shaped.takeError());
      owned = std::move(*shaped);
    }
    auto materialized = loom::frontend::finalizeSpatialOwnershipCandidate(
        std::move(*owned), design->roots().front(), ownershipOptions.lowering);
    if (!materialized)
      return reportError(materialized.takeError());
    selected.emplace(std::move(*materialized));
  }

  const dataflow::CanonicalDataflowArtifact &canonical =
      selected ? selected->canonicalDataflow : compiled->canonicalDataflow;

  // The canonical view is imported, not cached: entity counts come from the
  // same validated projection every consumer sees.
  auto view = canonical.view();
  if (!view)
    return reportError(view.takeError());

  // Emit the finalized Canonical Dataflow module.
  std::string errMsg;
  auto outputFile = ::mlir::openOutputFile(outputFilename, &errMsg);
  if (!outputFile) {
    ::llvm::errs() << "loom-pre-mapping: cannot open output: " << errMsg
                   << "\n";
    return 1;
  }
  canonical.module()->print(outputFile->os());
  outputFile->os() << "\n";
  outputFile->keep();

  // Emit the structured counts.
  auto countsFile = ::mlir::openOutputFile(countsFilename, &errMsg);
  if (!countsFile) {
    ::llvm::errs() << "loom-pre-mapping: cannot open counts output: " << errMsg
                   << "\n";
    return 1;
  }
  if (!hasExplicitSelection && planGenerateSummary.completedInvocations == 0 &&
      planGenerateSummary.incompleteInvocations == 0)
    return reportError(::llvm::createStringError(
        ::llvm::inconvertibleErrorCode(),
        "completed DSE retained no Generate provenance"));
  countsFile->os() << "{\"actors\": " << view->actors().size()
                   << ", \"central_generate_invocations\": "
                   << planGenerateSummary.completedInvocations
                   << ", \"central_incomplete_generate_invocations\": "
                   << planGenerateSummary.incompleteInvocations
                   << ", \"central_generate_input_artifacts\": "
                   << planGenerateSummary.inputArtifacts
                   << ", \"central_generate_lineage_edges\": "
                   << planGenerateSummary.lineageEdges
                   << ", \"central_generate_output_artifacts\": "
                   << planGenerateSummary.outputArtifacts
                   << ", \"central_plan_executions\": "
                   << planGenerateSummary.planExecutions
                   << ", \"graphs\": " << view->graphs().size()
                   << ", \"search_complete\": "
                   << (planSearchComplete ? "true" : "false") << "}\n";
  countsFile->keep();

  std::optional<loom::ArtifactRootReference> dataflowReference;
  if (!rootReferenceFilename.empty() ||
      !applicationWorkloadSetFilename.empty()) {
    auto published = dataflow::publishCanonicalDataflow(canonical, store);
    if (!published)
      return reportError(published.takeError());
    dataflowReference = std::move(*published);
  }

  if (!rootReferenceFilename.empty()) {
    if (llvm::Error error = loom::writeArtifactRootReferenceJsonFile(
            rootReferenceFilename, *dataflowReference))
      return reportError(std::move(error));
  }

  if (!applicationWorkloadSetFilename.empty()) {
    auto roots =
        view->projectRootThreadLaunchesReachableFromAbiEntry(applicationEntry);
    if (!roots)
      return reportError(roots.takeError());
    std::vector<loom::ArtifactRootReference> workloads;
    for (const dataflow::RootThreadLaunchRef &root : *roots) {
      auto domain = view->projectRootThreadLogicalDomain(root);
      if (!domain)
        return reportError(domain.takeError());
      if (domain->coordinateRank != 0)
        return reportError(::llvm::createStringError(
            ::llvm::inconvertibleErrorCode(),
            "application workload authoring requires explicit coordinates "
            "for non-rank-zero root thread launches"));
      llvm::Error workloadError = llvm::Error::success();
      view->forEachRootedGraphLaunch(
          [&](dataflow::RootedGraphLaunchRef launch) {
            if (workloadError || launch.rootThreadLaunch != root)
              return;
            loom::sim::SpatialSimulationWorkload draft{launch};
            auto shapes = loom::sim::projectSpatialSimulationBoundaryShapes(
                *view, launch);
            if (!shapes) {
              workloadError = shapes.takeError();
              return;
            }
            draft.valueInputPlan.assign(shapes->valueInputs.size(),
                                        loom::sim::RuntimeValueInput{});
            auto workload = loom::sim::finalizeSimulationWorkload(draft, *view);
            if (!workload) {
              workloadError = workload.takeError();
              return;
            }
            auto reference =
                loom::sim::publishSimulationWorkload(*workload, store);
            if (!reference) {
              workloadError = reference.takeError();
              return;
            }
            workloads.push_back(std::move(*reference));
          });
      if (workloadError)
        return reportError(std::move(workloadError));
    }
    ::llvm::sort(workloads, loom::artifactRootReferenceLess);
    workloads.erase(std::unique(workloads.begin(), workloads.end()),
                    workloads.end());
    if (workloads.empty())
      return reportError(::llvm::createStringError(
          ::llvm::inconvertibleErrorCode(),
          "application entry reaches no authorable Spatial workload"));
    if (llvm::Error error = loom::writeArtifactRootReferenceSetJsonFile(
            applicationWorkloadSetFilename, workloads))
      return reportError(std::move(error));
  }

  if (!fabricRootReferenceFilename.empty())
    if (llvm::Error error = loom::writeArtifactRootReferenceJsonFile(
            fabricRootReferenceFilename, design->roots().front().reference()))
      return reportError(std::move(error));

  return 0;
}
