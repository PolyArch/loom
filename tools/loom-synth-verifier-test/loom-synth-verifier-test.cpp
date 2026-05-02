// CLI helper for the `verifier_failed` lit test.
//
// The four production strategies (`anchor`, `mcs`, `incremental`,
// `incremental_random`) all run `mlir::verify(wrapper)` on the freshly
// built wrapper before transferring ownership to the splice loop. With
// today's strategy code the verifier always passes, so the
// `verifier_failed` enum is reachable only when a future strategy
// regresses or when the worker-to-main-thread re-parse path drops a
// detail. This helper closes the test gap by deliberately constructing
// an ill-formed wrapper -- an inner `fabric.fu` whose body contains an
// op outside the allow-set enforced by `FuOp::verify` (`fabric.op`,
// `fabric.mux`, `fabric.demux`, `fabric.yield`) -- and then driving
// the canonical failure-diagnostic helper exactly the way the
// production pass would on a real verifier failure.
//
// Usage:
//   loom-synth-verifier-test
//
// Output:
//   * The `loom-generalize-subgraphs-to-fu: group "<g>": synthesis
//     failed: verifier_failed` warning, mirrored on stderr by the
//     helper's diagnostic emission path.
//   * A printed module containing the input `func.func` carrying
//     `loom.synth_failed = "verifier_failed"` so FileCheck can pin the
//     attribute string the production pass writes.
//
// The tool intentionally does NOT call `signalPassFailure` or any pass
// machinery -- the helper is the part of the pass under test, and it
// exits with status 0 on success regardless of whether the simulated
// failure was reported as a warning or escalated to error.

#include "Fabric/IR/FabricDialect.h"
#include "Fabric/IR/FabricOps.h"
#include "Fabric/IR/FabricTypes.h"
#include "Fabric/Tech/Synthesizer/FailureDiagnostic.h"
#include "Fabric/Tech/Synthesizer/Synthesizer.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Verifier.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

#include <string>
#include <vector>

namespace {

// Build a fresh module containing one input `func.func` that mimics the
// shape the pass operates on (carrying `loom.synth_group`), plus a
// deliberately-broken wrapper `func.func` whose inner `fabric.fu` body
// embeds an `arith.addi` -- not in the allow-set enforced by
// `FuOp::verify` -- so the verifier rejects it.
//
// Returns the built module by value via `OwningOpRef`. The caller owns
// it. The returned `inputFunc` and `brokenWrapper` are pointers into
// that module's body; they remain valid for the module's lifetime.
struct BuiltModule {
  ::mlir::OwningOpRef<::mlir::ModuleOp> module;
  ::mlir::func::FuncOp inputFunc;
  ::mlir::func::FuncOp brokenWrapper;
};

BuiltModule buildModule(::mlir::MLIRContext &ctx,
                        ::llvm::StringRef groupName) {
  ::mlir::OpBuilder builder(&ctx);
  ::mlir::Location loc = builder.getUnknownLoc();

  auto module = ::mlir::ModuleOp::create(loc);
  builder.setInsertionPointToEnd(module.getBody());

  // Input function shape: `func.func @pat_addi(i32, i32) -> i32` with a
  // `loom.synth_group` attribute. The body is a simple return so
  // `mlir::verify(module)` accepts it; the test does not run the pass
  // on this function -- the helper only stamps `loom.synth_failed` on
  // it the way the production pass would on the failing group.
  auto i32 = builder.getI32Type();
  auto inputFnType =
      ::mlir::FunctionType::get(&ctx, {i32, i32}, {i32});
  auto inputFunc =
      ::mlir::func::FuncOp::create(loc, "pat_addi", inputFnType);
  inputFunc->setAttr("loom.synth_group",
                     builder.getStringAttr(groupName));
  ::mlir::Block *inputEntry = inputFunc.addEntryBlock();
  ::mlir::OpBuilder inputBodyBuilder(inputEntry, inputEntry->end());
  // Trivial body: return the first argument so the function verifies.
  ::mlir::OperationState retState(
      loc, ::mlir::func::ReturnOp::getOperationName());
  retState.addOperands(::mlir::ValueRange(inputEntry->getArgument(0)));
  inputBodyBuilder.create(retState);
  module.push_back(inputFunc);

  // Broken wrapper: `func.func @fu_<group>(bits<32>, bits<32>) -> bits<32>`
  // whose body is `[fabric.fu, func.return]`, but whose inner FU body
  // contains an `arith.addi` -- an op NOT in the allow-set enforced by
  // `FuOp::verify`. The verifier therefore rejects the wrapper, which
  // is exactly the "compiler-bug" path the `verifier_failed` enum
  // value guards.
  auto bits32 = ::fabric::BitsType::get(&ctx, 32u);
  auto wrapperType =
      ::mlir::FunctionType::get(&ctx, {bits32, bits32}, {bits32});
  std::string wrapperName = "fu_";
  wrapperName += groupName.str();
  auto wrapper =
      ::mlir::func::FuncOp::create(loc, wrapperName, wrapperType);
  ::mlir::Block *wrapperEntry = wrapper.addEntryBlock();
  ::mlir::OpBuilder wrapperBodyBuilder(wrapperEntry, wrapperEntry->end());

  // Inner fabric.fu: operands are the wrapper's args, results are the
  // wrapper's result types. The FU's region entry block carries the
  // same arg types as the operands (per FuOp::verify).
  ::mlir::OperationState fuState(
      loc, ::fabric::FuOp::getOperationName());
  fuState.addOperands(::mlir::ValueRange(wrapperEntry->getArguments()));
  fuState.addTypes({bits32});
  ::mlir::Region *fuRegion = fuState.addRegion();
  ::mlir::Block *fuEntry = new ::mlir::Block();
  fuRegion->push_back(fuEntry);
  ::llvm::SmallVector<::mlir::Type, 2> fuArgTypes{bits32, bits32};
  ::llvm::SmallVector<::mlir::Location, 2> fuArgLocs(2, loc);
  fuEntry->addArguments(fuArgTypes, fuArgLocs);
  ::mlir::Operation *rawFu = wrapperBodyBuilder.create(fuState);
  auto fu = ::mlir::cast<::fabric::FuOp>(rawFu);

  // Inject a deliberately-illegal op inside the FU body. `FuOp::verify`
  // rejects every op whose identity is outside the allow-set
  // (`fabric.op` / `fabric.mux` / `fabric.demux` / `fabric.yield`); the
  // identity check fires before any operand/result type check, so the
  // malformed op only has to be of an out-of-allow-set kind. We use
  // `arith.addi` here -- it is in a registered dialect (so the printer
  // round-trips deterministically), and the FU's bits<32> arg types
  // never reach `arith.addi`'s own verifier because `FuOp::verify`
  // short-circuits first.
  ::mlir::OpBuilder fuBodyBuilder(fuEntry, fuEntry->end());
  ::mlir::OperationState addState(
      loc, ::mlir::arith::AddIOp::getOperationName());
  addState.addOperands(::mlir::ValueRange(fuEntry->getArguments()));
  addState.addTypes({bits32});
  fuBodyBuilder.create(addState);

  // FU yield. The yield value is the FU's first arg-aliased SSA value
  // (bits<32>), which keeps the FU result-type contract consistent
  // even though the FU is malformed in the allow-set sense.
  ::mlir::OperationState yieldState(
      loc, ::fabric::YieldOp::getOperationName());
  yieldState.addOperands(
      ::mlir::ValueRange(fuEntry->getArgument(0)));
  fuBodyBuilder.create(yieldState);

  // Wrapper-level return.
  ::mlir::OperationState wrapperRet(
      loc, ::mlir::func::ReturnOp::getOperationName());
  wrapperRet.addOperands(::mlir::ValueRange(fu.getResult(0)));
  wrapperBodyBuilder.create(wrapperRet);

  module.push_back(wrapper);

  ::mlir::OwningOpRef<::mlir::ModuleOp> owned(module);
  return BuiltModule{std::move(owned), inputFunc, wrapper};
}

} // namespace

int main(int argc, char **argv) {
  ::llvm::cl::ParseCommandLineOptions(
      argc, argv,
      "loom-synth-verifier-test: drive the canonical "
      "synthesis-failure diagnostic on a deliberately-malformed "
      "fabric.fu wrapper so the `verifier_failed` enum path is "
      "exercised end-to-end without instrumenting the production "
      "strategies.\n");

  ::mlir::DialectRegistry registry;
  registry.insert<::mlir::func::FuncDialect, ::fabric::FabricDialect,
                  ::mlir::arith::ArithDialect>();
  ::mlir::MLIRContext ctx(registry);
  ctx.loadAllAvailableDialects();

  // Install a stderr-mirroring handler so the canonical helper warning
  // lands on stderr in the same shape lit pipelines expect from the
  // production `loom` driver. We mirror MLIR's default formatting:
  // `<location>: <severity>: <message>`.
  auto severityLabel = [](::mlir::DiagnosticSeverity s) -> ::llvm::StringRef {
    switch (s) {
    case ::mlir::DiagnosticSeverity::Note:
      return "note";
    case ::mlir::DiagnosticSeverity::Warning:
      return "warning";
    case ::mlir::DiagnosticSeverity::Error:
      return "error";
    case ::mlir::DiagnosticSeverity::Remark:
      return "remark";
    }
    return "diag";
  };
  ctx.getDiagEngine().registerHandler(
      [&](::mlir::Diagnostic &d) {
        ::llvm::errs() << "loom-synth-verifier-test: "
                       << severityLabel(d.getSeverity()) << ": "
                       << d.str() << "\n";
        for (const ::mlir::Diagnostic &n : d.getNotes())
          ::llvm::errs() << "  note: " << n.str() << "\n";
        return ::mlir::success();
      });

  ::llvm::StringRef groupName = "alu_int_32";
  BuiltModule built = buildModule(ctx, groupName);

  // Step 1: confirm the wrapper does NOT pass MLIR's verifier. The
  // production pass invokes `mlir::verify(wrapper)` at the end of the
  // strategy run; this is the same check. We capture the verifier's
  // own diagnostics through a ScopedDiagnosticHandler so the only text
  // FileCheck inspects is the canonical helper output.
  bool verifierRejected = false;
  {
    ::mlir::ScopedDiagnosticHandler capture(
        &ctx, [&](::mlir::Diagnostic &) { return ::mlir::success(); });
    if (::mlir::failed(::mlir::verify(built.brokenWrapper)))
      verifierRejected = true;
  }
  if (!verifierRejected) {
    ::llvm::errs()
        << "loom-synth-verifier-test: error: the constructed wrapper "
           "unexpectedly passed mlir::verify; the test setup no longer "
           "exercises the verifier_failed path.\n";
    return 1;
  }

  // Step 2: build the SynthResult and drive the shared helper. The
  // helper writes `loom.synth_failed = "verifier_failed"` on the input
  // function and emits the canonical
  // `loom-generalize-subgraphs-to-fu: group "<g>": synthesis failed:
  // verifier_failed` warning on the module -- byte-identical with the
  // production pass.
  ::loom::fabric::tech::SynthResult result;
  result.failureReason =
      ::loom::fabric::tech::SynthFailureReason::VerifierFailed;
  result.notes.push_back(
      "loom-synth-verifier-test: synthesized FU failed MLIR verifier");

  ::llvm::SmallVector<::mlir::func::FuncOp, 1> parents{built.inputFunc};
  ::std::vector<::std::string> notes(result.notes.begin(),
                                     result.notes.end());
  ::loom::fabric::tech::annotateAndDiagnoseGroupFailure(
      built.module.get(), groupName,
      ::llvm::ArrayRef<::mlir::func::FuncOp>(parents),
      result.failureReason,
      ::llvm::ArrayRef<::std::string>(notes),
      /*failAsError=*/false);

  // Step 3: drop the broken wrapper before printing -- the production
  // pass never appends an unverified wrapper to the module either.
  // Keeping it would invite the printer to round-trip an op the
  // FabricDialect's allow-set rejects.
  built.brokenWrapper.erase();

  // Step 4: print the post-helper module so FileCheck can pin the
  // `loom.synth_failed = "verifier_failed"` attribute on the input
  // function.
  built.module->print(::llvm::outs());
  ::llvm::outs() << "\n";
  return 0;
}
