#include "Fabric/Tech/Synthesizer/FailureDiagnostic.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"

namespace loom::fabric::tech {

void annotateAndDiagnoseGroupFailure(
    ::mlir::ModuleOp module, ::llvm::StringRef groupName,
    ::llvm::ArrayRef<::mlir::func::FuncOp> parents,
    SynthFailureReason reason,
    ::llvm::ArrayRef<::std::string> notes, bool failAsError) {
  ::mlir::MLIRContext *ctx = module.getContext();
  ::llvm::StringRef reasonStr = failureReasonString(reason);
  auto attr = ::mlir::StringAttr::get(ctx, reasonStr);
  for (::mlir::func::FuncOp func : parents)
    func->setAttr("loom.synth_failed", attr);

  ::mlir::InFlightDiagnostic diag =
      failAsError ? module.emitError() : module.emitWarning();
  diag << "loom-generalize-subgraphs-to-fu: group \"" << groupName
       << "\": synthesis failed: " << reasonStr;
  for (const ::std::string &n : notes)
    diag.attachNote() << n;
}

} // namespace loom::fabric::tech
