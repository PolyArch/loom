#ifndef LOOM_FABRIC_TECH_SYNTHESIZER_FAILUREDIAGNOSTIC_H
#define LOOM_FABRIC_TECH_SYNTHESIZER_FAILUREDIAGNOSTIC_H

// Shared helper used by `loom-generalize-subgraphs-to-fu` (and the
// `loom-synth-verifier-test` lit-driver tool) to emit the canonical
// per-group synthesis-failure diagnostic and stamp the
// `loom.synth_failed` attribute on every offending input function.
//
// The function exists so the production pass and the test tool can
// share one source of truth for both the diagnostic text and the
// attribute string. Any drift between the helper and the pass is a
// regression.
//
// Output contract (must remain byte-identical between callers):
//   * Every entry in `parents` gets `loom.synth_failed = "<reason>"`,
//     where `<reason>` is `failureReasonString(reason)`.
//   * One diagnostic is emitted on `module`. Severity is `error` when
//     `failAsError` is true, `warning` otherwise. The text is exactly:
//         loom-generalize-subgraphs-to-fu: group "<groupName>": \
//             synthesis failed: <reason>
//     and every entry in `notes` is attached as a note in iteration
//     order.
//
// The helper does NOT call `signalPassFailure()`. The pass owns that
// decision because the test driver does not run inside an
// `mlir::Pass`.

#include "Fabric/Tech/Synthesizer/Synthesizer.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include <string>

namespace loom::fabric::tech {

// Annotate every entry in `parents` with `loom.synth_failed = "<reason>"`
// and emit the canonical synthesis-failure diagnostic on `module`. See
// the file-header comment for the exact output contract. `notes` are
// attached as notes on the diagnostic in iteration order.
void annotateAndDiagnoseGroupFailure(
    ::mlir::ModuleOp module, ::llvm::StringRef groupName,
    ::llvm::ArrayRef<::mlir::func::FuncOp> parents,
    SynthFailureReason reason,
    ::llvm::ArrayRef<::std::string> notes, bool failAsError);

} // namespace loom::fabric::tech

#endif // LOOM_FABRIC_TECH_SYNTHESIZER_FAILUREDIAGNOSTIC_H
