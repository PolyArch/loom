// loom-pre-mapping: read an LLVM IR (.ll / .bc) file via parseIRFile, resolve
// one exact builtin Fabric target through an existing ArtifactStore, and run
// the production pre-Mapping compilation library
// (loom::frontend::compileLlvmModuleToPreMapping). The mechanical
// LLVM-to-Structured raising and Structured-to-Dataflow lowering boundaries,
// including the Structured Program and Canonical Dataflow finalizers, execute
// inside the shared library; this driver owns only CLI and presentation.
//
// CLI shape mirrors loom-raise / loom-adg:
//
//     loom-pre-mapping --builtin=small|default|large --artifact-store=<dir>
//                      --counts=<counts.json>
//                      [--whole-callable-spatial=<symbol>]
//                      [--operation-spatial=<symbol>
//                       --operation-spatial-scope-index=<index>
//                       --canonical-index-width=<bits>]
//                      [--fmuladd-shape=fused|split]
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
#include "Frontend/Compilation/OwnershipCandidateGenerator.h"
#include "Frontend/Compilation/PreMappingCompilation.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/AsmState.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

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
    builtinName("builtin", ::llvm::cl::desc("builtin Fabric target preset"),
                ::llvm::cl::value_desc("small|default|large"),
                ::llvm::cl::Required);

::llvm::cl::opt<std::string>
    artifactStorePath("artifact-store",
                      ::llvm::cl::desc("existing ArtifactStore directory"),
                      ::llvm::cl::value_desc("path"), ::llvm::cl::Required);

::llvm::cl::opt<std::string>
    countsFilename("counts",
                   ::llvm::cl::desc("output path for structured graph/actor "
                                    "counts as one JSON object"),
                   ::llvm::cl::value_desc("filename"), ::llvm::cl::Required);

::llvm::cl::opt<std::string> candidateInventoryFilename(
    "candidate-inventory",
    ::llvm::cl::desc("write every scope-local Spatial ownership attempt as "
                     "a diagnostic JSON inventory; accepted candidates are "
                     "published through their Artifact owners"),
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

::llvm::cl::opt<unsigned> canonicalIndexWidth(
    "canonical-index-width",
    ::llvm::cl::desc("explicit canonical index width materialized for a "
                     "selected Spatial candidate"),
    ::llvm::cl::value_desc("bits"), ::llvm::cl::init(0));

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
  auto view = candidate.view();
  if (!view)
    return view.takeError();
  std::optional<loom::frontend::StructuredEntityRef> resolved;
  for (const loom::frontend::StructuredEntity &entity :
       view->entities(loom::frontend::StructuredEntityKind::Operation)) {
    auto function =
        ::llvm::dyn_cast_or_null<::mlir::LLVM::LLVMFuncOp>(entity.operation);
    if (!function || function.getSymName() != symbol)
      continue;
    if (resolved)
      return ::llvm::createStringError(
          ::llvm::inconvertibleErrorCode(),
          "callable symbol is not unique in the Structured Program");
    resolved = entity.reference;
  }
  if (!resolved)
    return ::llvm::createStringError(::llvm::inconvertibleErrorCode(),
                                     "callable symbol does not resolve");
  return *resolved;
}

::llvm::Expected<loom::frontend::StructuredEntityRef>
resolveUniqueStructuredOperation(
    const loom::frontend::StructuredProgramCandidate &candidate,
    ::llvm::StringRef callableSymbol, std::optional<std::uint64_t> scopeIndex) {
  auto view = candidate.view();
  if (!view)
    return view.takeError();
  auto scopes =
      loom::frontend::enumerateOperationSpatialOwnershipScopes(candidate);
  if (!scopes)
    return scopes.takeError();
  std::vector<loom::frontend::StructuredEntityRef> callableScopes;
  for (const loom::frontend::StructuredEntityRef &scope : *scopes) {
    auto entity = view->resolve(scope);
    if (!entity)
      return entity.takeError();
    ::mlir::Operation *operation = entity->operation;
    if (!operation)
      continue;
    auto callable = operation->getParentOfType<::mlir::LLVM::LLVMFuncOp>();
    if (!callable || callable.getSymName() != callableSymbol)
      continue;
    callableScopes.push_back(scope);
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

::llvm::StringRef
scopeKindSpelling(loom::frontend::SpatialOwnershipScopeKind kind) {
  switch (kind) {
  case loom::frontend::SpatialOwnershipScopeKind::WholeCallable:
    return "whole_callable";
  case loom::frontend::SpatialOwnershipScopeKind::Operation:
    return "operation";
  }
  llvm_unreachable("unknown Spatial ownership scope kind");
}

::llvm::StringRef rejectionKindSpelling(
    loom::frontend::SpatialOwnershipCandidateRejectionKind kind) {
  switch (kind) {
  case loom::frontend::SpatialOwnershipCandidateRejectionKind::NonFinalizable:
    return "non_finalizable";
  case loom::frontend::SpatialOwnershipCandidateRejectionKind::
      ExactFabricInadmissible:
    return "exact_fabric_inadmissible";
  }
  llvm_unreachable("unknown Spatial ownership rejection kind");
}

::llvm::json::Value optionalIndexWidth(
    const loom::frontend::SpatialOwnershipDecisionPoint &decision) {
  if (!decision.canonicalIndexWidth)
    return nullptr;
  return static_cast<std::int64_t>(*decision.canonicalIndexWidth);
}

::llvm::json::Value optionalFmuladdShape(
    const loom::frontend::SpatialOwnershipDecisionPoint &decision) {
  if (!decision.fmuladdExecutionShape)
    return nullptr;
  switch (*decision.fmuladdExecutionShape) {
  case loom::raising::FMulAddExecutionShape::Fused:
    return "fused";
  case loom::raising::FMulAddExecutionShape::Split:
    return "split";
  }
  llvm_unreachable("unknown fmuladd execution shape");
}

::llvm::Error writeCandidateInventory(
    ::llvm::StringRef outputPath,
    const loom::frontend::StructuredProgramCandidate &parent,
    const ::loom::fabric::FinalizedFabricRoot &fabric,
    const loom::ArtifactStore &store,
    const loom::lowering::CanonicalDataflowLoweringOptions &lowering) {
  auto scopes = loom::frontend::enumerateSpatialOwnershipScopes(parent);
  if (!scopes)
    return scopes.takeError();

  std::uint64_t acceptedCount = 0;
  std::uint64_t rejectedCount = 0;
  ::llvm::json::Array scopeRecords;
  for (const loom::frontend::SpatialOwnershipScope &scope : *scopes) {
    auto domain = loom::frontend::enumerateSpatialOwnershipDecisionDomain(
        parent, scope.selection);
    if (!domain)
      return domain.takeError();

    ::llvm::json::Array decisions;
    for (const loom::frontend::SpatialOwnershipDecisionPoint &decision :
         *domain) {
      ::llvm::json::Object record{
          {"canonical_index_width", optionalIndexWidth(decision)},
          {"fmuladd_shape", optionalFmuladdShape(decision)},
      };
      auto candidate = loom::frontend::materializeSpatialOwnershipDecision(
          parent, scope, decision, fabric, lowering);
      if (!candidate) {
        std::optional<loom::frontend::SpatialOwnershipCandidateRejectionKind>
            rejectionKind;
        std::string diagnostic;
        ::llvm::Error unhandled = ::llvm::handleErrors(
            candidate.takeError(),
            [&](const loom::frontend::SpatialOwnershipCandidateRejection
                    &rejection) {
              rejectionKind = rejection.kind();
              diagnostic = rejection.message();
            });
        if (unhandled)
          return unhandled;
        if (!rejectionKind)
          return ::llvm::createStringError(
              ::llvm::inconvertibleErrorCode(),
              "candidate rejection handler produced no classification");
        record["diagnostic"] = std::move(diagnostic);
        record["rejection_kind"] = rejectionKindSpelling(*rejectionKind);
        record["status"] = "rejected";
        ++rejectedCount;
        decisions.push_back(std::move(record));
        continue;
      }

      auto structured = loom::frontend::publishStructuredProgram(
          candidate->structuredProgram, store);
      if (!structured)
        return structured.takeError();
      auto publishedDataflow = dataflow::publishCanonicalDataflow(
          candidate->canonicalDataflow, store);
      if (!publishedDataflow)
        return publishedDataflow.takeError();
      auto view = candidate->canonicalDataflow.view();
      if (!view)
        return view.takeError();
      record["actors"] = static_cast<std::int64_t>(view->actors().size());
      record["canonical_dataflow"] =
          loom::formatArtifactIdentityHex(publishedDataflow->artifact);
      record["graphs"] = static_cast<std::int64_t>(view->graphs().size());
      record["status"] = "accepted";
      record["structured_program"] =
          loom::formatArtifactIdentityHex(structured->artifact);
      ++acceptedCount;
      decisions.push_back(std::move(record));
    }

    scopeRecords.push_back(::llvm::json::Object{
        {"decisions", std::move(decisions)},
        {"scope_kind", scopeKindSpelling(scope.kind)},
        {"scope_ordinal", static_cast<std::int64_t>(scope.selection.ordinal)},
    });
  }

  std::string errorMessage;
  auto output = ::mlir::openOutputFile(outputPath, &errorMessage);
  if (!output)
    return ::llvm::createStringError(::llvm::inconvertibleErrorCode(),
                                     "cannot open candidate inventory: %s",
                                     errorMessage.c_str());
  ::llvm::json::Object root{
      {"accepted", static_cast<std::int64_t>(acceptedCount)},
      {"parent_structured_program",
       loom::formatArtifactIdentityHex(parent.identity())},
      {"rejected", static_cast<std::int64_t>(rejectedCount)},
      {"scopes", std::move(scopeRecords)},
  };
  output->os() << ::llvm::formatv("{0:2}", ::llvm::json::Value(std::move(root)))
               << '\n';
  output->keep();
  return ::llvm::Error::success();
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
      "ArtifactStore and runs the production pre-Mapping compilation "
      "library (mechanical LLVM-to-Structured raising, Structured Program "
      "finalization, mechanical Structured-to-Dataflow lowering, and "
      "Canonical Dataflow finalization). Emits the finalized Canonical "
      "Dataflow module and structured graph/actor counts.\n");

  auto preset = loom::adg::parseBuiltinTargetPreset(builtinName);
  if (!preset)
    return reportError(preset.takeError());

  // Parse LLVM IR.
  ::llvm::LLVMContext llvmContext;
  std::unique_ptr<::llvm::Module> llvmModule =
      readLLVMModule(llvmContext, inputFilename);
  if (!llvmModule)
    return 1;

  // Resolve the exact builtin Fabric target through its owner.
  loom::ArtifactStore store(artifactStorePath);
  auto design = loom::adg::buildBuiltinTarget(store, *preset);
  if (!design)
    return reportError(design.takeError());
  if (design->roots().size() != 1)
    return reportError(::llvm::createStringError(
        ::llvm::inconvertibleErrorCode(),
        "builtin target did not produce exactly one Fabric root"));

  loom::frontend::PreMappingCompilationOptions compilationOptions;
  compilationOptions.raising.applyPassManagerCommandLineOptions = true;
  compilationOptions.lowering.applyPassManagerCommandLineOptions = true;
  auto compiled = loom::frontend::compileLlvmModuleToPreMapping(
      std::move(llvmModule), design->roots().front().reference(), store,
      compilationOptions);
  if (!compiled)
    return reportError(compiled.takeError());

  if (!candidateInventoryFilename.empty())
    if (::llvm::Error error = writeCandidateInventory(
            candidateInventoryFilename, compiled->structuredProgram,
            design->roots().front(), store, compilationOptions.lowering))
      return reportError(std::move(error));

  if (!wholeCallableSpatial.empty() && !operationSpatial.empty())
    return reportError(::llvm::createStringError(
        ::llvm::inconvertibleErrorCode(),
        "whole-callable and operation Spatial selections are exclusive"));
  if (canonicalIndexWidth != 0 && operationSpatial.empty() &&
      wholeCallableSpatial.empty())
    return reportError(::llvm::createStringError(
        ::llvm::inconvertibleErrorCode(),
        "canonical index width requires a Spatial selection"));
  if (operationSpatialScopeIndex.getNumOccurrences() != 0 &&
      operationSpatial.empty())
    return reportError(::llvm::createStringError(
        ::llvm::inconvertibleErrorCode(),
        "operation Spatial scope index requires an operation selection"));
  if (fmuladdShape != FMulAddShapeOption::Unspecified &&
      wholeCallableSpatial.empty() && operationSpatial.empty())
    return reportError(::llvm::createStringError(
        ::llvm::inconvertibleErrorCode(),
        "fmuladd shape requires a Spatial selection"));

  std::optional<loom::frontend::MaterializedOwnershipCandidate> selected;
  if (!wholeCallableSpatial.empty()) {
    auto callable =
        resolveCallable(compiled->structuredProgram, wholeCallableSpatial);
    if (!callable)
      return reportError(callable.takeError());
    loom::frontend::WholeCallableSpatialOwnershipOptions ownershipOptions;
    ownershipOptions.lowering = compilationOptions.lowering;
    if (canonicalIndexWidth != 0)
      ownershipOptions.canonicalIndexWidth = canonicalIndexWidth;
    if (fmuladdShape == FMulAddShapeOption::Fused)
      ownershipOptions.fmuladdExecutionShape =
          loom::raising::FMulAddExecutionShape::Fused;
    else if (fmuladdShape == FMulAddShapeOption::Split)
      ownershipOptions.fmuladdExecutionShape =
          loom::raising::FMulAddExecutionShape::Split;
    auto materialized =
        loom::frontend::materializeWholeCallableSpatialOwnership(
            compiled->structuredProgram, *callable, design->roots().front(),
            ownershipOptions);
    if (!materialized)
      return reportError(materialized.takeError());
    selected.emplace(std::move(*materialized));
  } else if (!operationSpatial.empty()) {
    std::optional<std::uint64_t> scopeIndex;
    if (operationSpatialScopeIndex.getNumOccurrences() != 0)
      scopeIndex = operationSpatialScopeIndex;
    auto operation = resolveUniqueStructuredOperation(
        compiled->structuredProgram, operationSpatial, scopeIndex);
    if (!operation)
      return reportError(operation.takeError());
    loom::frontend::OperationSpatialOwnershipOptions ownershipOptions;
    ownershipOptions.lowering = compilationOptions.lowering;
    if (canonicalIndexWidth != 0)
      ownershipOptions.canonicalIndexWidth = canonicalIndexWidth;
    if (fmuladdShape == FMulAddShapeOption::Fused)
      ownershipOptions.fmuladdExecutionShape =
          loom::raising::FMulAddExecutionShape::Fused;
    else if (fmuladdShape == FMulAddShapeOption::Split)
      ownershipOptions.fmuladdExecutionShape =
          loom::raising::FMulAddExecutionShape::Split;
    auto materialized = loom::frontend::materializeOperationSpatialOwnership(
        compiled->structuredProgram, *operation, design->roots().front(),
        ownershipOptions);
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
  countsFile->os() << "{\"actors\": " << view->actors().size()
                   << ", \"graphs\": " << view->graphs().size() << "}\n";
  countsFile->keep();

  return 0;
}
