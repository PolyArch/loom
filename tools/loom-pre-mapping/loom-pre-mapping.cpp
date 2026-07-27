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
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

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

::llvm::cl::opt<std::string> wholeCallableSpatial(
    "whole-callable-spatial",
    ::llvm::cl::desc("materialize one exact LLVM callable as an "
                     "explicit whole-callable Spatial ownership candidate"),
    ::llvm::cl::value_desc("symbol"), ::llvm::cl::init(""));

::llvm::cl::opt<std::string> operationSpatial(
    "operation-spatial",
    ::llvm::cl::desc("materialize the unique structured operation in one "
                     "LLVM callable as an explicit Spatial candidate"),
    ::llvm::cl::value_desc("callable-symbol"), ::llvm::cl::init(""));

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
    ::llvm::StringRef callableSymbol) {
  auto view = candidate.view();
  if (!view)
    return view.takeError();
  std::optional<loom::frontend::StructuredEntityRef> resolved;
  for (const loom::frontend::StructuredEntity &entity :
       view->entities(loom::frontend::StructuredEntityKind::Operation)) {
    ::mlir::Operation *operation = entity.operation;
    if (!operation || operation->getNumRegions() == 0 ||
        ::llvm::isa<::mlir::LLVM::LLVMFuncOp>(operation))
      continue;
    auto callable = operation->getParentOfType<::mlir::LLVM::LLVMFuncOp>();
    if (!callable || callable.getSymName() != callableSymbol)
      continue;
    if (resolved)
      return ::llvm::createStringError(
          ::llvm::inconvertibleErrorCode(),
          "callable contains more than one structured operation");
    resolved = entity.reference;
  }
  if (!resolved)
    return ::llvm::createStringError(
        ::llvm::inconvertibleErrorCode(),
        "callable contains no structured operation");
  return *resolved;
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

  if (!wholeCallableSpatial.empty() && !operationSpatial.empty())
    return reportError(::llvm::createStringError(
        ::llvm::inconvertibleErrorCode(),
        "whole-callable and operation Spatial selections are exclusive"));
  if (canonicalIndexWidth != 0 && operationSpatial.empty() &&
      wholeCallableSpatial.empty())
    return reportError(::llvm::createStringError(
        ::llvm::inconvertibleErrorCode(),
        "canonical index width requires a Spatial selection"));
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
    auto operation = resolveUniqueStructuredOperation(
        compiled->structuredProgram, operationSpatial);
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
