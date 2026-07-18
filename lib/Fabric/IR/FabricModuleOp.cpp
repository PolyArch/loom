#include "Fabric/IR/FabricOps.h"

#include "Fabric/IR/FabricTypes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace fabric;

namespace {

static bool isSubordinateProviderResult(Value value) {
  auto result = dyn_cast<OpResult>(value);
  if (!result)
    return false;

  if (auto mem = dyn_cast<MemOp>(result.getOwner()))
    return !mem.getSymNameAttr() && isa<MemRefType>(result.getType());

  return isa<InstantiateOp>(result.getOwner()) &&
         isa<MemRefType>(result.getType());
}

static LogicalResult verifyMemoryExportProviders(fabric::ModuleOp module,
                                                 Block &entry) {
  auto yield = dyn_cast<YieldOp>(entry.getTerminator());
  if (!yield)
    return success();

  ArrayRef<Type> resultTypes = module.getFunctionType().getResults();
  for (auto [index, pair] :
       llvm::enumerate(llvm::zip(yield.getValues(), resultTypes))) {
    Value value;
    Type resultType;
    std::tie(value, resultType) = pair;
    if (!isa<MemRefType>(resultType) || value.getType() != resultType)
      continue;
    if (isSubordinateProviderResult(value))
      continue;
    return yield.emitOpError("memref module result #")
           << index
           << " must originate from a subordinate provider result of "
              "fabric.mem or fabric.instantiate";
  }
  return success();
}

} // namespace

LogicalResult fabric::ModuleOp::verify() {
  // Module inputs are entry-block arguments rather than ODS operands, so
  // validate both signature directions here for consistent diagnostics.
  FunctionType ft = getFunctionType();
  for (auto [i, type] : llvm::enumerate(ft.getInputs())) {
    if (!isFabricModulePortType(type))
      return emitOpError("input #")
             << i << " type " << type
             << " is not an allowed fabric.module port type "
                "(allowed: !fabric.bits<W>, !fabric.bits_tag<W,T>, "
                "memref<...>)";
  }
  for (auto [i, type] : llvm::enumerate(ft.getResults())) {
    if (!isFabricModulePortType(type))
      return emitOpError("result #")
             << i << " type " << type
             << " is not an allowed fabric.module port type "
                "(allowed: !fabric.bits<W>, !fabric.bits_tag<W,T>, "
                "memref<...>)";
  }

  Block &entry = getBody().front();
  if (entry.getNumArguments() != ft.getNumInputs())
    return emitOpError("entry block argument count (")
           << entry.getNumArguments() << ") must match declared input count ("
           << ft.getNumInputs() << ")";
  for (auto [i, pair] :
       llvm::enumerate(llvm::zip(entry.getArguments(), ft.getInputs()))) {
    BlockArgument argument;
    Type declared;
    std::tie(argument, declared) = pair;
    if (argument.getType() != declared)
      return emitOpError("entry block argument #")
             << i << " type " << argument.getType()
             << " must equal declared input type " << declared;
  }

  for (Operation &op : entry) {
    if (isa<PeOp, SwitchOp, MemOp, FifoOp, fabric::ModuleOp, InstantiateOp,
            BoundaryOp, YieldOp>(op))
      continue;
    return op.emitOpError(
        "is not allowed inside fabric.module; only fabric.pe, "
        "fabric.switch, fabric.mem, fabric.fifo, fabric.module, "
        "fabric.instantiate, and fabric.boundary are permitted (plus the "
        "implicit terminator fabric.yield)");
  }

  if (failed(verifyMemoryExportProviders(*this, entry)))
    return failure();

  // Only direct module-body transport values are point-to-point. Memory
  // capabilities are export-provenance checked above and are not linear.
  auto countBodyConsumers = [&](Value value) {
    unsigned count = 0;
    for (OpOperand &use : value.getUses())
      if (use.getOwner()->getBlock() == &entry)
        ++count;
    return count;
  };

  for (auto [i, argument] : llvm::enumerate(entry.getArguments())) {
    if (!isa<BitsType, BitsTagType>(argument.getType()))
      continue;
    unsigned consumerCount = countBodyConsumers(argument);
    if (consumerCount > 1)
      return emitOpError(
                 "transport source is used by more than one consumer in this "
                 "fabric.module body: block argument #")
             << i << " of type " << argument.getType() << " has "
             << consumerCount << " consuming uses";
  }

  for (Operation &sourceOp : entry) {
    for (auto [i, result] : llvm::enumerate(sourceOp.getResults())) {
      if (!isa<BitsType, BitsTagType>(result.getType()))
        continue;
      unsigned consumerCount = countBodyConsumers(result);
      if (consumerCount > 1)
        return sourceOp.emitOpError(
                   "transport source is used by more than one consumer in "
                   "this fabric.module body: result #")
               << i << " of type " << result.getType() << " has "
               << consumerCount << " consuming uses";
    }
  }
  return success();
}
