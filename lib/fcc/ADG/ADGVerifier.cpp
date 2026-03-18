#include "fcc/ADG/ADGVerifier.h"

#include "fcc/Dialect/Fabric/FabricDialect.h"
#include "fcc/Dialect/Fabric/FabricOps.h"
#include "fcc/Mapper/TypeCompat.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Verifier.h"

namespace fcc {

namespace {

using SymbolTargetMap = llvm::StringMap<mlir::Operation *>;

bool isDefinitionOnlyOp(mlir::Operation &op,
                        const llvm::DenseSet<llvm::StringRef> &referencedSyms) {
  if (mlir::isa<fcc::fabric::FunctionUnitOp, fcc::fabric::SpatialPEOp,
                fcc::fabric::TemporalPEOp>(op))
    return true;

  if (auto swOp = mlir::dyn_cast<fcc::fabric::SpatialSwOp>(op)) {
    auto symName = swOp.getSymName();
    return symName && referencedSyms.contains(*symName);
  }
  if (auto swOp = mlir::dyn_cast<fcc::fabric::TemporalSwOp>(op)) {
    auto symName = swOp.getSymName();
    return symName && referencedSyms.contains(*symName);
  }
  return false;
}

bool isGraphNodeOp(mlir::Operation &op,
                   const llvm::DenseSet<llvm::StringRef> &referencedSyms) {
  if (mlir::isa<fcc::fabric::InstanceOp, fcc::fabric::ExtMemoryOp,
                fcc::fabric::MemoryOp, fcc::fabric::FifoOp,
                fcc::fabric::AddTagOp, fcc::fabric::DelTagOp,
                fcc::fabric::MapTagOp>(op)) {
    return true;
  }
  if (mlir::isa<fcc::fabric::SpatialSwOp, fcc::fabric::TemporalSwOp>(op))
    return !isDefinitionOnlyOp(op, referencedSyms);
  return false;
}

std::optional<mlir::FunctionType> getDeclaredFunctionType(mlir::Operation &op,
                                                          const SymbolTargetMap &targets) {
  if (auto instOp = mlir::dyn_cast<fcc::fabric::InstanceOp>(op)) {
    auto targetIt = targets.find(instOp.getModule());
    if (targetIt == targets.end())
      return std::nullopt;
    auto *target = targetIt->second;
    if (auto pe = mlir::dyn_cast<fcc::fabric::SpatialPEOp>(target))
      return pe.getFunctionType();
    if (auto pe = mlir::dyn_cast<fcc::fabric::TemporalPEOp>(target))
      return pe.getFunctionType();
    if (auto sw = mlir::dyn_cast<fcc::fabric::SpatialSwOp>(target))
      return sw.getFunctionType();
    if (auto sw = mlir::dyn_cast<fcc::fabric::TemporalSwOp>(target))
      return sw.getFunctionType();
    return std::nullopt;
  }
  if (auto extOp = mlir::dyn_cast<fcc::fabric::ExtMemoryOp>(op))
    return extOp.getFunctionType();
  if (auto memOp = mlir::dyn_cast<fcc::fabric::MemoryOp>(op))
    return memOp.getFunctionType();
  if (auto fifoOp = mlir::dyn_cast<fcc::fabric::FifoOp>(op))
    return fifoOp.getFunctionType();
  if (auto swOp = mlir::dyn_cast<fcc::fabric::SpatialSwOp>(op))
    return swOp.getFunctionType();
  if (auto swOp = mlir::dyn_cast<fcc::fabric::TemporalSwOp>(op))
    return swOp.getFunctionType();
  return std::nullopt;
}

std::string describeOp(mlir::Operation &op) {
  if (auto instOp = mlir::dyn_cast<fcc::fabric::InstanceOp>(op)) {
    auto symName = instOp.getSymName();
    if (symName)
      return ("instance '" + symName->str() + "'");
    return "fabric.instance";
  }
  auto printNamed = [&](auto namedOp) -> std::string {
    auto symName = namedOp.getSymName();
    if (symName)
      return (op.getName().getStringRef().str() + " '" + symName->str() + "'");
    return op.getName().getStringRef().str();
  };
  if (auto swOp = mlir::dyn_cast<fcc::fabric::SpatialSwOp>(op))
    return printNamed(swOp);
  if (auto swOp = mlir::dyn_cast<fcc::fabric::TemporalSwOp>(op))
    return printNamed(swOp);
  if (auto extOp = mlir::dyn_cast<fcc::fabric::ExtMemoryOp>(op))
    return printNamed(extOp);
  if (auto memOp = mlir::dyn_cast<fcc::fabric::MemoryOp>(op))
    return printNamed(memOp);
  if (auto fifoOp = mlir::dyn_cast<fcc::fabric::FifoOp>(op))
    return printNamed(fifoOp);
  return op.getName().getStringRef().str();
}

} // namespace

mlir::LogicalResult verifyFabricModule(mlir::ModuleOp topModule) {
  if (failed(mlir::verify(topModule)))
    return mlir::failure();

  bool ok = true;

  // Find fabric.module
  fcc::fabric::ModuleOp fabricMod;
  topModule->walk([&](fcc::fabric::ModuleOp mod) { fabricMod = mod; });
  if (!fabricMod)
    return mlir::success(); // No fabric.module, nothing to check

  auto &body = fabricMod.getBody().front();

  llvm::DenseSet<llvm::StringRef> referencedSyms;
  for (auto &op : body.getOperations()) {
    if (auto instOp = mlir::dyn_cast<fcc::fabric::InstanceOp>(op))
      referencedSyms.insert(instOp.getModule());
  }

  SymbolTargetMap symbolTargets;
  topModule.walk([&](mlir::Operation *op) {
    if (auto peOp = mlir::dyn_cast<fcc::fabric::SpatialPEOp>(op))
      if (auto symName = peOp.getSymName())
        symbolTargets[*symName] = op;
    if (auto peOp = mlir::dyn_cast<fcc::fabric::TemporalPEOp>(op))
      if (auto symName = peOp.getSymName())
        symbolTargets[*symName] = op;
    if (auto swOp = mlir::dyn_cast<fcc::fabric::SpatialSwOp>(op))
      if (auto symName = swOp.getSymName())
        symbolTargets[*symName] = op;
    if (auto swOp = mlir::dyn_cast<fcc::fabric::TemporalSwOp>(op))
      if (auto symName = swOp.getSymName())
        symbolTargets[*symName] = op;
  });

  llvm::DenseSet<mlir::Value> consumed;

  for (auto &op : body.getOperations()) {
    if (!isGraphNodeOp(op, referencedSyms) &&
        !mlir::isa<fcc::fabric::YieldOp>(op)) {
      continue;
    }

    for (auto operand : op.getOperands())
      consumed.insert(operand);
  }

  for (auto &op : body.getOperations()) {
    if (!isGraphNodeOp(op, referencedSyms))
      continue;

    std::string opDesc = describeOp(op);

    if (auto fnType = getDeclaredFunctionType(op, symbolTargets)) {
      if (fnType->getNumInputs() != op.getNumOperands()) {
        llvm::errs() << "fabric verify: dangling input ports on " << opDesc
                     << " — expected " << fnType->getNumInputs()
                     << " connected input(s), got " << op.getNumOperands()
                     << "\n";
        ok = false;
      }
      if (fnType->getNumResults() != op.getNumResults()) {
        llvm::errs() << "fabric verify: output port count mismatch on "
                     << opDesc << " — expected " << fnType->getNumResults()
                     << " output(s), got " << op.getNumResults() << "\n";
        ok = false;
      }

      unsigned operandPairs =
          std::min<unsigned>(fnType->getNumInputs(), op.getNumOperands());
      for (unsigned i = 0; i < operandPairs; ++i) {
        if (!isHardwarePortCompatible(op.getOperand(i).getType(),
                                      fnType->getInput(i))) {
          llvm::errs() << "fabric verify: tag-kind mismatch on " << opDesc
                       << " input I" << i << " — producer type "
                       << op.getOperand(i).getType() << " is incompatible with "
                       << fnType->getInput(i) << "\n";
          ok = false;
        }
      }

      unsigned resultPairs =
          std::min<unsigned>(fnType->getNumResults(), op.getNumResults());
      for (unsigned i = 0; i < resultPairs; ++i) {
        if (!isHardwarePortCompatible(fnType->getResult(i),
                                      op.getResult(i).getType())) {
          llvm::errs() << "fabric verify: tag-kind mismatch on " << opDesc
                       << " output O" << i << " — declared type "
                       << fnType->getResult(i) << " is incompatible with "
                       << op.getResult(i).getType() << "\n";
          ok = false;
        }
      }
    } else if (auto instOp = mlir::dyn_cast<fcc::fabric::InstanceOp>(op)) {
      llvm::errs() << "fabric verify: " << opDesc << " references unknown target '"
                   << instOp.getModule() << "'\n";
      ok = false;
    }

    for (unsigned i = 0; i < op.getNumResults(); ++i) {
      mlir::Value result = op.getResult(i);
      if (!consumed.count(result)) {
        llvm::errs() << "fabric verify: dangling output port O" << i
                     << " of " << opDesc << " — result is not connected\n";
        ok = false;
      }
    }
  }

  for (auto arg : body.getArguments()) {
    if (!consumed.count(arg)) {
      llvm::errs() << "fabric verify: dangling module input port I"
                   << arg.getArgNumber() << " — not connected to any instance\n";
      ok = false;
    }
  }

  auto yieldOp = mlir::dyn_cast<fcc::fabric::YieldOp>(body.getTerminator());
  if (yieldOp) {
    auto fnType = fabricMod.getFunctionType();
    unsigned resultPairs =
        std::min<unsigned>(yieldOp.getNumOperands(), fnType.getNumResults());
    for (unsigned i = 0; i < resultPairs; ++i) {
      if (!isHardwarePortCompatible(yieldOp.getOperand(i).getType(),
                                    fnType.getResult(i))) {
        llvm::errs() << "fabric verify: tag-kind mismatch on module output O"
                     << i << " — source type " << yieldOp.getOperand(i).getType()
                     << " is incompatible with " << fnType.getResult(i)
                     << "\n";
        ok = false;
      }
    }
  }

  return ok ? mlir::success() : mlir::failure();
}

} // namespace fcc
