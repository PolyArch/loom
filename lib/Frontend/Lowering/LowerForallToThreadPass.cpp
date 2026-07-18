// Reject implicit thread ownership for host scf.forall operations.
//
// dataflow.thread.launch models only zero-based grid extents. Thread
// promotion therefore requires a recognized Loom mapping and a prior
// structured-domain transformation that preserves lower bounds and steps.
// Neither authority is currently materialized by this pass.

#include "Frontend/Lowering/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

namespace {

struct LowerForallToThreadPass
    : public ::mlir::PassWrapper<LowerForallToThreadPass,
                                 ::mlir::OperationPass<::mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerForallToThreadPass)

  ::llvm::StringRef getArgument() const final {
    return "loom-lower-forall-to-thread";
  }

  ::llvm::StringRef getDescription() const final {
    return "Reject scf.forall thread promotion without a recognized Loom "
           "mapping and faithfully represented domain.";
  }

  void getDependentDialects(::mlir::DialectRegistry &registry) const final {
    registry.insert<::mlir::func::FuncDialect, ::mlir::scf::SCFDialect>();
  }

  void runOnOperation() final {
    bool rejected = false;
    getOperation().walk([&](::mlir::scf::ForallOp forall) {
      if (!forall->getParentOfType<::mlir::func::FuncOp>())
        return;
      forall.emitError(
          "loom-lower-forall-to-thread: raw scf.forall has no recognized "
          "Loom thread mapping; preserve it until structured ownership and "
          "its complete domain are selected");
      rejected = true;
    });
    if (rejected)
      signalPassFailure();
  }
};

} // namespace

namespace loom {
namespace lowering {

std::unique_ptr<::mlir::Pass> createLowerForallToThreadPass() {
  return std::make_unique<LowerForallToThreadPass>();
}

void registerLowerForallToThreadPass() {
  static bool once = []() {
    ::mlir::PassRegistration<LowerForallToThreadPass>();
    return true;
  }();
  (void)once;
}

} // namespace lowering
} // namespace loom
