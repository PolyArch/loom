#include "fcc/ADG/ADGVerifier.h"

#include "fcc/Dialect/Fabric/FabricDialect.h"
#include "fcc/Dialect/Fabric/FabricOps.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/raw_ostream.h"

namespace fcc {

mlir::LogicalResult verifyFabricModule(mlir::ModuleOp topModule) {
  bool ok = true;

  // Find fabric.module
  fcc::fabric::ModuleOp fabricMod;
  topModule->walk([&](fcc::fabric::ModuleOp mod) { fabricMod = mod; });
  if (!fabricMod)
    return mlir::success(); // No fabric.module, nothing to check

  auto &body = fabricMod.getBody().front();

  // Collect all SSA values that are "produced" (block args + instance results)
  llvm::DenseSet<mlir::Value> produced;
  for (auto arg : body.getArguments())
    produced.insert(arg);

  // Collect all SSA values that are "consumed" (instance operands + yield operands)
  llvm::DenseSet<mlir::Value> consumed;

  for (auto &op : body.getOperations()) {
    // Track produced values
    for (auto result : op.getResults())
      produced.insert(result);

    // Track consumed values (operands of instances and yield)
    if (auto instOp = mlir::dyn_cast<fcc::fabric::InstanceOp>(op)) {
      for (auto operand : instOp.getOperands())
        consumed.insert(operand);
    }
    if (auto yieldOp = mlir::dyn_cast<fcc::fabric::YieldOp>(op)) {
      for (auto operand : yieldOp.getOperands())
        consumed.insert(operand);
    }
  }

  // Check: every instance result must be consumed somewhere
  for (auto &op : body.getOperations()) {
    auto instOp = mlir::dyn_cast<fcc::fabric::InstanceOp>(op);
    if (!instOp)
      continue;

    std::string instName = instOp.getSymName().value_or("?").str();
    for (unsigned i = 0; i < instOp.getNumResults(); ++i) {
      auto result = instOp.getResult(i);
      if (!consumed.count(result)) {
        llvm::errs() << "fabric verify: dangling output port O" << i
                     << " of instance '" << instName
                     << "' — result #" << i << " is not used\n";
        ok = false;
      }
    }
  }

  // Check: every module block arg must be consumed
  for (auto arg : body.getArguments()) {
    if (!consumed.count(arg)) {
      llvm::errs() << "fabric verify: dangling module input port I"
                   << arg.getArgNumber() << " — not connected to any instance\n";
      ok = false;
    }
  }

  return ok ? mlir::success() : mlir::failure();
}

} // namespace fcc
