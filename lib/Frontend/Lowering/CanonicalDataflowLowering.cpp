#include "Frontend/Lowering/CanonicalDataflowLowering.h"

#include "Dataflow/IR/DataflowGraphValidation.h"
#include "Frontend/Lowering/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Pass/PassManager.h"

#include "llvm/Support/Error.h"

namespace loom::lowering {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "canonical_dataflow_lowering_invalid: " +
                                     message);
}

} // namespace

llvm::Error
lowerStructuredModuleInPlace(mlir::ModuleOp module,
                             CanonicalDataflowLoweringOptions options) {
  if (!module)
    return invalid("missing Structured Program module");

  registerLoweringPasses();
  mlir::PassManager pipeline(module.getContext());
  pipeline.enableVerifier(options.verifyEach);
  if (options.applyPassManagerCommandLineOptions &&
      failed(mlir::applyPassManagerCLOptions(pipeline)))
    return invalid("cannot apply pass-manager command-line options");
  buildLoweringPipeline(pipeline);
  if (failed(pipeline.run(module)))
    return invalid("mechanical SCF-to-Dataflow lowering failed");
  if (llvm::Error error = dataflow::validateFinalizedProgram(module))
    return error;
  return llvm::Error::success();
}

llvm::Expected<dataflow::CanonicalDataflowArtifact>
lowerStructuredModuleToCanonicalDataflow(
    mlir::ModuleOp module, CanonicalDataflowLoweringOptions options) {
  if (!module)
    return invalid("missing Structured Program module");
  mlir::OwningOpRef<mlir::ModuleOp> clone(
      mlir::cast<mlir::ModuleOp>(module->clone()));
  if (llvm::Error error = lowerStructuredModuleInPlace(clone.get(), options))
    return std::move(error);
  return dataflow::finalizeCanonicalDataflow(clone.get());
}

} // namespace loom::lowering
