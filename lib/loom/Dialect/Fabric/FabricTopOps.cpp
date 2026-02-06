//===-- FabricTopOps.cpp - Fabric top-level operation impls ------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include "loom/Dialect/Fabric/FabricOps.h"
#include "loom/Dialect/Dataflow/DataflowTypes.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"

#include "llvm/ADT/SmallPtrSet.h"

using namespace mlir;
using namespace loom::fabric;

//===----------------------------------------------------------------------===//
// ModuleOp parse/print
//
// fabric.module @name(%arg0: T0, %arg1: T1) -> (R0, R1) { body }
//===----------------------------------------------------------------------===//

ParseResult ModuleOp::parse(OpAsmParser &parser, OperationState &result) {
  // Parse @sym_name.
  StringAttr symName;
  if (parser.parseSymbolName(symName))
    return failure();
  result.addAttribute(getSymNameAttrName(result.name), symName);

  // Parse argument list: (%arg0: T0, %arg1: T1, ...).
  SmallVector<OpAsmParser::Argument> entryArgs;
  if (parser.parseArgumentList(entryArgs, OpAsmParser::Delimiter::Paren,
                               /*allowType=*/true))
    return failure();

  SmallVector<Type> argTypes;
  for (auto &arg : entryArgs)
    argTypes.push_back(arg.type);

  // Parse -> (result_types).
  SmallVector<Type> resultTypes;
  if (parser.parseArrow() || parser.parseLParen())
    return failure();
  if (failed(parser.parseOptionalRParen())) {
    if (parser.parseTypeList(resultTypes) || parser.parseRParen())
      return failure();
  }

  auto fnType = FunctionType::get(parser.getContext(), argTypes, resultTypes);
  result.addAttribute(getFunctionTypeAttrName(result.name),
                      TypeAttr::get(fnType));

  // Parse region body.
  auto *body = result.addRegion();
  if (parser.parseRegion(*body, entryArgs))
    return failure();
  if (body->empty())
    body->emplaceBlock();

  return success();
}

void ModuleOp::print(OpAsmPrinter &p) {
  p << " @" << getSymName() << "(";

  Block &entryBlock = getBody().front();
  llvm::interleaveComma(entryBlock.getArguments(), p,
                        [&](BlockArgument arg) {
                          p.printRegionArgument(arg);
                        });
  p << ")";

  p << " -> (";
  llvm::interleaveComma(getFunctionType().getResults(), p,
                        [&](Type t) { p.printType(t); });
  p << ")";

  p << " ";
  p.printRegion(getBody(), /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/true);
}

//===----------------------------------------------------------------------===//
// ModuleOp verify
//===----------------------------------------------------------------------===//

LogicalResult ModuleOp::verify() {
  auto fnType = getFunctionType();

  // Port ordering: memref* -> native* -> tagged*.
  auto checkOrdering = [&](ArrayRef<Type> types,
                           StringRef label) -> LogicalResult {
    // 0 = memref, 1 = native, 2 = tagged
    unsigned lastCat = 0;
    for (Type t : types) {
      unsigned cat;
      if (isa<MemRefType>(t))
        cat = 0;
      else if (isa<dataflow::TaggedType>(t))
        cat = 2;
      else
        cat = 1;
      if (cat < lastCat)
        return emitOpError("[COMP_MODULE_PORT_ORDER] ")
               << label
               << " must follow port ordering: memref*, native*, tagged*";
      lastCat = cat;
    }
    return success();
  };

  if (failed(checkOrdering(fnType.getInputs(), "inputs")))
    return failure();
  if (failed(checkOrdering(fnType.getResults(), "outputs")))
    return failure();

  // Body must have at least one non-terminator.
  Block &body = getBody().front();
  bool hasOp = false;
  for (auto &op : body) {
    if (!op.hasTrait<OpTrait::IsTerminator>()) {
      hasOp = true;
      break;
    }
  }
  if (!hasOp)
    return emitOpError("[COMP_MODULE_EMPTY_BODY] "
        "body must contain at least one non-terminator operation");

  // Yield operand types must match result types.
  // COMP_MODULE_MISSING_YIELD: the SingleBlockImplicitTerminator trait
  // guarantees a YieldOp exists; this check catches operand count mismatch.
  auto yield = cast<YieldOp>(body.getTerminator());
  if (yield.getOperands().size() != fnType.getNumResults())
    return emitOpError("[COMP_MODULE_MISSING_YIELD] yield operand count (")
           << yield.getOperands().size() << ") must match result count ("
           << fnType.getNumResults() << ")";

  for (auto [idx, pair] : llvm::enumerate(
           llvm::zip(yield.getOperandTypes(), fnType.getResults()))) {
    if (std::get<0>(pair) != std::get<1>(pair))
      return emitOpError("[COMP_FABRIC_TYPE_MISMATCH] yield operand #")
             << idx << " type " << std::get<0>(pair)
             << " must match result type " << std::get<1>(pair);
  }

  // COMP_MEMORY_PRIVATE_OUTPUT: each memref yield operand must trace back
  // to a MemoryOp with is_private = false.
  for (auto [idx, operand] : llvm::enumerate(yield.getOperands())) {
    if (!isa<MemRefType>(operand.getType()))
      continue;
    auto *defOp = operand.getDefiningOp();
    if (!defOp) {
      // Block argument: not produced by a memory op.
      return emitOpError("[COMP_MEMORY_PRIVATE_OUTPUT] yield memref operand #")
             << idx << " is not produced by a fabric.memory with "
             << "is_private = false";
    }
    auto memOp = dyn_cast<MemoryOp>(defOp);
    if (!memOp) {
      return emitOpError("[COMP_MEMORY_PRIVATE_OUTPUT] yield memref operand #")
             << idx << " is not produced by a fabric.memory";
    }
    if (memOp.getIsPrivate()) {
      return emitOpError("[COMP_MEMORY_PRIVATE_OUTPUT] yield memref operand #")
             << idx << " is produced by a fabric.memory with is_private = true";
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Instance helpers
//===----------------------------------------------------------------------===//

/// Get the function_type from a target operation that may be a ModuleOp, PEOp,
/// TemporalPEOp, SwitchOp, TemporalSwOp, MemoryOp, or ExtMemoryOp.
static std::optional<FunctionType> getTargetFunctionType(Operation *target) {
  if (auto mod = dyn_cast<ModuleOp>(target))
    return mod.getFunctionType();
  if (auto pe = dyn_cast<PEOp>(target)) {
    if (auto ft = pe.getFunctionType())
      return *ft;
    return std::nullopt;
  }
  if (auto tpe = dyn_cast<TemporalPEOp>(target))
    return tpe.getFunctionType();
  if (auto sw = dyn_cast<SwitchOp>(target)) {
    if (auto ft = sw.getFunctionType())
      return *ft;
    return std::nullopt;
  }
  if (auto tsw = dyn_cast<TemporalSwOp>(target)) {
    if (auto ft = tsw.getFunctionType())
      return *ft;
    return std::nullopt;
  }
  if (auto mem = dyn_cast<MemoryOp>(target)) {
    if (auto ft = mem.getFunctionType())
      return *ft;
    return std::nullopt;
  }
  if (auto ext = dyn_cast<ExtMemoryOp>(target)) {
    if (auto ft = ext.getFunctionType())
      return *ft;
    return std::nullopt;
  }
  return std::nullopt;
}

/// Check for cyclic instance references starting from an operation.
static bool hasCyclicReference(Operation *start, SymbolTableCollection &tables) {
  llvm::SmallPtrSet<Operation *, 8> visited;
  SmallVector<Operation *> worklist;
  worklist.push_back(start);

  while (!worklist.empty()) {
    Operation *current = worklist.pop_back_val();
    if (!visited.insert(current).second)
      return true;

    current->walk([&](InstanceOp inst) {
      auto *resolved = tables.lookupNearestSymbolFrom(
          inst.getOperation(), inst.getModuleAttr());
      if (resolved)
        worklist.push_back(resolved);
    });
  }
  return false;
}

//===----------------------------------------------------------------------===//
// InstanceOp verify
//===----------------------------------------------------------------------===//

LogicalResult InstanceOp::verify() {
  // COMP_INSTANCE_UNRESOLVED: symbol must exist.
  SymbolTableCollection tables;
  auto *target = tables.lookupNearestSymbolFrom(
      getOperation(), getModuleAttr());
  if (!target)
    return emitOpError("[COMP_INSTANCE_UNRESOLVED] referenced symbol '")
           << getModule() << "' does not exist";

  // COMP_INSTANCE_ILLEGAL_TARGET: cannot instantiate tag ops.
  if (isa<AddTagOp>(target) || isa<MapTagOp>(target) || isa<DelTagOp>(target))
    return emitOpError("[COMP_INSTANCE_ILLEGAL_TARGET] cannot instantiate "
                       "tag operation '")
           << getModule() << "'";

  // COMP_INSTANCE_OPERAND_MISMATCH / COMP_INSTANCE_RESULT_MISMATCH:
  // Compare types against target's function_type.
  auto targetFnType = getTargetFunctionType(target);
  if (targetFnType) {
    auto fnType = *targetFnType;
    if (getOperands().size() != fnType.getNumInputs())
      return emitOpError("[COMP_INSTANCE_OPERAND_MISMATCH] operand count (")
             << getOperands().size() << ") does not match target input count ("
             << fnType.getNumInputs() << ")";
    for (auto [idx, pair] : llvm::enumerate(
             llvm::zip(getOperandTypes(), fnType.getInputs()))) {
      if (std::get<0>(pair) != std::get<1>(pair))
        return emitOpError("[COMP_INSTANCE_OPERAND_MISMATCH] operand #")
               << idx << " type " << std::get<0>(pair)
               << " does not match target input type " << std::get<1>(pair);
    }

    if (getResults().size() != fnType.getNumResults())
      return emitOpError("[COMP_INSTANCE_RESULT_MISMATCH] result count (")
             << getResults().size() << ") does not match target result count ("
             << fnType.getNumResults() << ")";
    for (auto [idx, pair] : llvm::enumerate(
             llvm::zip(getResultTypes(), fnType.getResults()))) {
      if (std::get<0>(pair) != std::get<1>(pair))
        return emitOpError("[COMP_INSTANCE_RESULT_MISMATCH] result #")
               << idx << " type " << std::get<0>(pair)
               << " does not match target result type " << std::get<1>(pair);
    }
  }

  // COMP_INSTANCE_CYCLIC_REFERENCE: check for cycles.
  if (hasCyclicReference(target, tables))
    return emitOpError("[COMP_INSTANCE_CYCLIC_REFERENCE] instance of '")
           << getModule() << "' forms a cyclic reference";

  return success();
}
