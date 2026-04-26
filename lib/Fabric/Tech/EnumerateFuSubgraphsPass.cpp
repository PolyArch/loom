#include "Fabric/Tech/Passes.h"

#include "Dataflow/IR/DataflowDialect.h"
#include "Fabric/IR/FabricOps.h"
#include "Fabric/Tech/SubgraphEnumerator.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace {

struct EnumerateFuSubgraphsPass
    : public ::mlir::PassWrapper<EnumerateFuSubgraphsPass,
                                 ::mlir::OperationPass<::mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(EnumerateFuSubgraphsPass)

  ::llvm::StringRef getArgument() const final {
    return "loom-enumerate-fu-subgraphs";
  }
  ::llvm::StringRef getDescription() const final {
    return "Enumerate dataflow.subgraphs supported by each fabric.fu and "
           "emit them as siblings of the FU.";
  }

  void getDependentDialects(::mlir::DialectRegistry &registry) const final {
    registry.insert<::dataflow::DataflowDialect,
                    ::mlir::arith::ArithDialect,
                    ::mlir::math::MathDialect>();
  }

  void runOnOperation() final {
    ::mlir::ModuleOp module = getOperation();
    ::llvm::SmallVector<::fabric::FuOp> fus;
    module.walk([&](::fabric::FuOp fu) { fus.push_back(fu); });

    auto *ctx = &getContext();
    unsigned counter = 0;
    for (::fabric::FuOp fu : fus) {
      std::string baseName =
          "fu" + std::to_string(counter++) + "_subgraph";
      ::llvm::StringRef unsupported;
      auto cands = ::fabric::enumerateFuSubgraphs(fu, module, baseName,
                                                  &unsupported);
      if (!unsupported.empty()) {
        fu.emitWarning("fabric.fu enumeration skipped: contains unsupported "
                       "op '")
            << unsupported << "'";
      }
      for (auto &cand : cands) {
        cand.subgraph->setAttr(
            "loom.from_fu_config",
            ::mlir::StringAttr::get(ctx, cand.configDescription));
      }
    }
  }
};

} // namespace

namespace fabric {

std::unique_ptr<::mlir::Pass> createEnumerateFuSubgraphsPass() {
  return std::make_unique<EnumerateFuSubgraphsPass>();
}

void registerFabricTechPasses() {
  ::mlir::PassRegistration<EnumerateFuSubgraphsPass>();
  ::mlir::registerPass(createMapSubgraphToFusPass);
  ::mlir::registerPass([] { return createPartitionGraphPass(); });
}

} // namespace fabric
