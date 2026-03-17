#include "fcc/ADG/ADGVerifier.h"

#include "fcc/Dialect/Fabric/FabricDialect.h"
#include "fcc/Dialect/Fabric/FabricOps.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringMap.h"
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

  llvm::StringMap<fcc::fabric::InstanceOp> instByName;
  for (auto &op : body.getOperations()) {
    if (auto instOp = mlir::dyn_cast<fcc::fabric::InstanceOp>(op)) {
      if (auto symName = instOp.getSymName())
        instByName[*symName] = instOp;
    }
  }

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

    // Track consumed SSA operands for all graph-region operations.
    for (auto operand : op.getOperands())
      consumed.insert(operand);

    if (auto extOp = mlir::dyn_cast<fcc::fabric::ExtMemoryOp>(op)) {
      if (extOp.getNumOperands() > 0)
        continue;

      if (auto argIdxAttr =
              extOp->getAttrOfType<mlir::IntegerAttr>("memref_arg_index")) {
        auto argIdx = static_cast<unsigned>(argIdxAttr.getInt());
        if (argIdx < body.getNumArguments())
          consumed.insert(body.getArgument(argIdx));
      }

      auto markBackEdgesConsumed = [&](mlir::ArrayAttr detailAttr,
                                       bool detailed) {
        for (auto elem : detailAttr) {
          llvm::StringRef swName;
          int64_t outputBase = detailed ? 0 : 4;
          if (detailed) {
            auto dictAttr = mlir::dyn_cast<mlir::DictionaryAttr>(elem);
            if (!dictAttr)
              continue;
            auto nameAttr = dictAttr.getAs<mlir::StringAttr>("name");
            if (!nameAttr)
              continue;
            swName = nameAttr.getValue();
            if (auto outBaseAttr =
                    dictAttr.getAs<mlir::IntegerAttr>("output_port_base")) {
              outputBase = outBaseAttr.getInt();
            }
          } else {
            auto strAttr = mlir::dyn_cast<mlir::StringAttr>(elem);
            if (!strAttr)
              continue;
            swName = strAttr.getValue();
          }

          auto instIt = instByName.find(swName);
          if (instIt == instByName.end())
            continue;
          auto swInst = instIt->second;
          unsigned numDataInputs =
              extOp.getFunctionType().getNumInputs() > 0
                  ? extOp.getFunctionType().getNumInputs() - 1
                  : 0;
          for (unsigned p = 0; p < numDataInputs; ++p) {
            unsigned resultIdx = static_cast<unsigned>(outputBase) + p;
            if (resultIdx < swInst.getNumResults())
              consumed.insert(swInst.getResult(resultIdx));
          }
        }
      };

      if (auto detailAttr =
              extOp->getAttrOfType<mlir::ArrayAttr>("connected_sw_detail")) {
        markBackEdgesConsumed(detailAttr, /*detailed=*/true);
      } else if (auto connAttr =
                     extOp->getAttrOfType<mlir::ArrayAttr>("connected_sw")) {
        markBackEdgesConsumed(connAttr, /*detailed=*/false);
      }
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
