//===-- FabricTopOps.cpp - Fabric top-level operation impls ------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include "loom/Dialect/Fabric/FabricOps.h"
#include "loom/Dialect/Dataflow/DataflowTypes.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"

#include "llvm/ADT/DenseMap.h"
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

// Forward declaration (defined in Instance helpers section below).
static Operation *lookupBySymName(Operation *from, StringRef name);

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

  // COMP_ADG_COMBINATIONAL_LOOP: detect cycles among purely combinational ops.
  {
    // Classify ops as combinational (zero-delay).
    // For InstanceOp, resolve the target symbol to determine if the
    // instantiated module is combinational.
    auto isCombinational = [](Operation *op) -> bool {
      if (isa<SwitchOp, TemporalSwOp, AddTagOp, MapTagOp, DelTagOp>(op))
        return true;
      if (auto inst = dyn_cast<InstanceOp>(op)) {
        auto *target = lookupBySymName(inst.getOperation(), inst.getModule());
        if (target)
          return isa<SwitchOp, TemporalSwOp, AddTagOp, MapTagOp, DelTagOp>(
              target);
      }
      return false;
    };

    // Build adjacency list for combinational ops only.
    // Map from op to index, then build directed edges.
    SmallVector<Operation *> combOps;
    llvm::DenseMap<Operation *, unsigned> opIndex;
    for (auto &op : body) {
      if (op.hasTrait<OpTrait::IsTerminator>())
        continue;
      if (isCombinational(&op)) {
        opIndex[&op] = combOps.size();
        combOps.push_back(&op);
      }
    }

    if (!combOps.empty()) {
      unsigned n = combOps.size();
      SmallVector<SmallVector<unsigned>> adj(n);

      for (unsigned i = 0; i < n; ++i) {
        for (Value result : combOps[i]->getResults()) {
          for (Operation *user : result.getUsers()) {
            auto it = opIndex.find(user);
            if (it != opIndex.end())
              adj[i].push_back(it->second);
          }
        }
      }

      // DFS cycle detection: 0=white, 1=gray, 2=black.
      SmallVector<int> color(n, 0);
      bool hasCombLoop = false;
      std::function<void(unsigned)> dfs = [&](unsigned u) {
        if (hasCombLoop)
          return;
        color[u] = 1;
        for (unsigned v : adj[u]) {
          if (color[v] == 1) {
            hasCombLoop = true;
            return;
          }
          if (color[v] == 0)
            dfs(v);
          if (hasCombLoop)
            return;
        }
        color[u] = 2;
      };

      for (unsigned i = 0; i < n && !hasCombLoop; ++i) {
        if (color[i] == 0)
          dfs(i);
      }

      if (hasCombLoop)
        return emitOpError("[COMP_ADG_COMBINATIONAL_LOOP] "
            "a cycle of purely combinational operations exists; "
            "insert a fabric.fifo or sequential element to break it");
    }
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
/// TemporalPEOp, SwitchOp, TemporalSwOp, MemoryOp, ExtMemoryOp, or FifoOp.
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
  if (auto fifo = dyn_cast<FifoOp>(target)) {
    if (auto ft = fifo.getFunctionType())
      return *ft;
    return std::nullopt;
  }
  return std::nullopt;
}

/// Look up an operation by its sym_name attribute, searching from the
/// outermost parent. Unlike SymbolTable::lookupNearestSymbolFrom, this finds
/// any op with a sym_name attribute, not just ops with the Symbol trait.
static Operation *lookupBySymName(Operation *from, StringRef name) {
  Operation *scope = from;
  while (scope->getParentOp())
    scope = scope->getParentOp();

  Operation *result = nullptr;
  scope->walk([&](Operation *op) -> WalkResult {
    if (auto attr = op->getAttrOfType<StringAttr>("sym_name")) {
      if (attr.getValue() == name) {
        result = op;
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });
  return result;
}

/// Check for cyclic instance references starting from an operation.
static bool hasCyclicReference(Operation *start) {
  llvm::SmallPtrSet<Operation *, 8> visited;
  SmallVector<Operation *> worklist;
  worklist.push_back(start);

  while (!worklist.empty()) {
    Operation *current = worklist.pop_back_val();
    if (!visited.insert(current).second)
      return true;

    current->walk([&](InstanceOp inst) {
      auto *resolved = lookupBySymName(inst.getOperation(), inst.getModule());
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
  auto *target = lookupBySymName(getOperation(), getModule());
  if (!target)
    return emitOpError("[COMP_INSTANCE_UNRESOLVED] referenced symbol '")
           << getModule() << "' does not exist";

  // COMP_PE_INSTANCE_ILLEGAL_TARGET: inside fabric.pe, only named fabric.pe
  // targets are legal.
  if (getOperation()->getParentOfType<PEOp>() && !isa<PEOp>(target))
    return emitOpError("[COMP_PE_INSTANCE_ILLEGAL_TARGET] inside fabric.pe, "
                       "fabric.instance may only target a named fabric.pe; '")
           << getModule() << "' is not a fabric.pe";

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
  if (hasCyclicReference(target))
    return emitOpError("[COMP_INSTANCE_CYCLIC_REFERENCE] instance of '")
           << getModule() << "' forms a cyclic reference";

  return success();
}
