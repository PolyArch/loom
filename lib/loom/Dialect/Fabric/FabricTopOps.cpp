//===-- FabricTopOps.cpp - Fabric top-level operation impls ------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include "loom/Dialect/Fabric/FabricOps.h"
#include "loom/Dialect/Dataflow/DataflowTypes.h"
#include "loom/Hardware/Common/FabricError.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace mlir;
using namespace loom;
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

/// Per-operation combinational path info.
struct CombInfo {
  llvm::SmallBitVector resultMask; // which results are combinational
  // operandDeps[i] = bitmask of which operands can influence result i
  // through combinational paths. Size equals resultMask size.
  SmallVector<llvm::SmallBitVector> operandDeps;

  /// Construct an empty (non-combinational) CombInfo for the given result count.
  static CombInfo empty(unsigned numResults) {
    CombInfo info;
    info.resultMask.resize(numResults);
    return info;
  }

  /// Construct a fully-connected CombInfo where every output depends on every
  /// input through a combinational path.
  static CombInfo fullConnectivity(unsigned numResults, unsigned numOperands) {
    CombInfo info;
    info.resultMask = llvm::SmallBitVector(numResults, true);
    for (unsigned o = 0; o < numResults; ++o)
      info.operandDeps.push_back(llvm::SmallBitVector(numOperands, true));
    return info;
  }
};

/// Helper: extract connectivity-based operandDeps from a switch-like op.
/// The connectivity_table is a flat DenseI8ArrayAttr of size numOut*numIn,
/// stored output-major: table[o * numIn + i] = 1 means output o receives
/// from input i. Returns per-output operandDeps using the table, or
/// full-connectivity deps if no table is present.
static CombInfo
getSwitchCombInfo(Operation *op, unsigned numResults, unsigned numOperands) {
  CombInfo info;
  info.resultMask = llvm::SmallBitVector(numResults, true);
  auto ctAttr = op->getAttrOfType<DenseI8ArrayAttr>("connectivity_table");
  if (ctAttr) {
    auto ct = ctAttr.asArrayRef();
    for (unsigned o = 0; o < numResults; ++o) {
      llvm::SmallBitVector deps(numOperands);
      for (unsigned i = 0; i < numOperands; ++i) {
        unsigned flatIdx = o * numOperands + i;
        if (flatIdx < ct.size() && ct[flatIdx])
          deps.set(i);
      }
      info.operandDeps.push_back(deps);
    }
  } else {
    // No connectivity table: full crossbar -- all outputs depend on all inputs.
    return CombInfo::fullConnectivity(numResults, numOperands);
  }
  return info;
}

/// Compute combinational path info for `op`. For SwitchOp/TemporalSwOp,
/// uses the connectivity_table to determine per-output input dependencies.
/// For AddTagOp/MapTagOp/DelTagOp, all inputs feed all outputs. For
/// InstanceOp, delegates to the resolved target. For ModuleOp, performs
/// per-yield-operand backward reachability analysis tracking which block
/// arguments (inputs) are reachable from each output independently.
/// Uses a path-based visited set to avoid infinite recursion.
static CombInfo getCombInfo(Operation *op,
                            llvm::SmallPtrSetImpl<Operation *> &visited) {
  // ModuleOp: per-yield-operand backward reachability. Handled first because
  // ModuleOp has 0 SSA results; logical counts come from the function type.
  if (auto mod = dyn_cast<ModuleOp>(op)) {
    unsigned numOutputs = mod.getFunctionType().getNumResults();
    unsigned numInputs = mod.getFunctionType().getNumInputs();
    if (numOutputs == 0)
      return CombInfo::empty(0);
    if (!visited.insert(op).second)
      return CombInfo::empty(numOutputs); // cycle guard
    CombInfo info;
    info.resultMask.resize(numOutputs);
    Block &body = mod.getBody().front();
    auto yield = cast<YieldOp>(body.getTerminator());
    info.operandDeps.resize(numOutputs, llvm::SmallBitVector(numInputs));
    for (unsigned i = 0; i < yield.getNumOperands(); ++i) {
      SmallVector<Value> worklist;
      llvm::SmallPtrSet<Value, 16> seen;
      worklist.push_back(yield.getOperand(i));
      llvm::SmallBitVector reachedArgs(numInputs);
      while (!worklist.empty()) {
        Value v = worklist.pop_back_val();
        if (!seen.insert(v).second)
          continue;
        if (auto ba = dyn_cast<BlockArgument>(v)) {
          reachedArgs.set(ba.getArgNumber());
          continue; // keep walking to find all reachable inputs
        }
        Operation *defOp = v.getDefiningOp();
        if (!defOp)
          continue;
        auto defInfo = getCombInfo(defOp, visited);
        unsigned resultIdx = cast<OpResult>(v).getResultNumber();
        if (resultIdx < defInfo.resultMask.size() &&
            defInfo.resultMask.test(resultIdx)) {
          // Only walk through operands that can influence this result.
          if (resultIdx < defInfo.operandDeps.size()) {
            const auto &deps = defInfo.operandDeps[resultIdx];
            for (unsigned k = 0; k < defOp->getNumOperands(); ++k) {
              if (k < deps.size() && deps.test(k))
                worklist.push_back(defOp->getOperand(k));
            }
          } else {
            for (Value operand : defOp->getOperands())
              worklist.push_back(operand);
          }
        }
      }
      if (reachedArgs.any()) {
        info.resultMask.set(i);
        info.operandDeps[i] = reachedArgs;
      }
    }
    visited.erase(op); // allow re-evaluation from other call paths
    return info;
  }

  // Determine logical result/operand counts. Named definitions (SwitchOp, etc.)
  // have 0 SSA results; use the function_type attribute instead.
  unsigned numResults = op->getNumResults();
  unsigned numOperands = op->getNumOperands();
  if (numResults == 0) {
    if (auto ftAttr = op->getAttrOfType<TypeAttr>("function_type")) {
      if (auto ft = dyn_cast<FunctionType>(ftAttr.getValue())) {
        numResults = ft.getNumResults();
        numOperands = ft.getNumInputs();
      }
    }
  }
  if (numResults == 0)
    return CombInfo::empty(0);

  // SwitchOp/TemporalSwOp: use connectivity_table for per-output deps.
  if (isa<SwitchOp, TemporalSwOp>(op))
    return getSwitchCombInfo(op, numResults, numOperands);

  // AddTagOp, MapTagOp, DelTagOp: all outputs depend on all inputs.
  if (isa<AddTagOp, MapTagOp, DelTagOp>(op))
    return CombInfo::fullConnectivity(numResults, numOperands);

  // InstanceOp: delegate to resolved target.
  if (auto inst = dyn_cast<InstanceOp>(op)) {
    auto *target = lookupBySymName(inst.getOperation(), inst.getModule());
    if (target) {
      auto targetInfo = getCombInfo(target, visited);
      if (targetInfo.resultMask.size() == numResults)
        return targetInfo;
    }
    return CombInfo::empty(numResults);
  }

  // Non-combinational ops.
  return CombInfo::empty(numResults);
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
        return emitOpError(cplErrCode(CplError::MODULE_PORT_ORDER) + " ")
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
    return emitOpError(cplErrMsg(CplError::MODULE_EMPTY_BODY,
        "body must contain at least one non-terminator operation"));

  // Yield operand types must match result types.
  // CPL_MODULE_MISSING_YIELD: the SingleBlockImplicitTerminator trait
  // guarantees a YieldOp exists; this check catches operand count mismatch.
  auto yield = cast<YieldOp>(body.getTerminator());
  if (yield.getOperands().size() != fnType.getNumResults())
    return emitOpError(cplErrMsg(CplError::MODULE_MISSING_YIELD,
                       "yield operand count ("))
           << yield.getOperands().size() << ") must match result count ("
           << fnType.getNumResults() << ")";

  for (auto [idx, pair] : llvm::enumerate(
           llvm::zip(yield.getOperandTypes(), fnType.getResults()))) {
    if (std::get<0>(pair) != std::get<1>(pair))
      return emitOpError(cplErrMsg(CplError::FABRIC_TYPE_MISMATCH,
                         "yield operand #"))
             << idx << " type " << std::get<0>(pair)
             << " must match result type " << std::get<1>(pair);
  }

  // CPL_FANOUT_MODULE_INNER: each SSA result of a non-terminator operation
  // in the module body must have at most one use (including yield/terminator).
  for (auto &op : body) {
    if (op.hasTrait<OpTrait::IsTerminator>())
      continue;
    for (Value result : op.getResults()) {
      unsigned useCount = 0;
      for (auto &use : result.getUses())
        ++useCount;
      if (useCount > 1)
        return emitOpError(cplErrMsg(CplError::FANOUT_MODULE_INNER,
                           "SSA result of '"))
               << op.getName() << "' has " << useCount
               << " consumers; strict 1-to-1 requires at most 1"
                  " (use switch broadcast for data duplication)";
    }
  }

  // CPL_FANOUT_MODULE_BOUNDARY: each module input (block argument) must have
  // at most one use (including yield/terminator).
  for (auto arg : body.getArguments()) {
    unsigned useCount = 0;
    for (auto &use : arg.getUses())
      ++useCount;
    if (useCount > 1)
      return emitOpError(cplErrMsg(CplError::FANOUT_MODULE_BOUNDARY,
                         "module input argument #"))
             << arg.getArgNumber() << " has " << useCount
             << " consumers; strict 1-to-1 requires at most 1"
                " (use switch broadcast for data duplication)";
  }

  // CPL_ADG_COMBINATIONAL_LOOP: detect cycles among purely combinational ops.
  // Uses a per-result graph: each node is (op, result_index). An edge from
  // (A, r) to (B, s) exists when result r of A is used at operand k of B,
  // and operandDeps[s][k] of B is true (operand k can influence result s).
  // This respects switch connectivity tables and per-output module deps.
  {
    llvm::SmallPtrSet<Operation *, 8> visited;

    SmallVector<Operation *> combOps;
    SmallVector<CombInfo> combInfos;
    llvm::DenseMap<Operation *, unsigned> opIndex;
    for (auto &op : body) {
      if (op.hasTrait<OpTrait::IsTerminator>())
        continue;
      auto info = getCombInfo(&op, visited);
      if (info.resultMask.any()) {
        opIndex[&op] = combOps.size();
        combOps.push_back(&op);
        combInfos.push_back(std::move(info));
      }
    }

    if (!combOps.empty()) {
      unsigned n = combOps.size();

      // Assign per-result node IDs.
      SmallVector<unsigned> nodeBase(n);
      unsigned totalNodes = 0;
      for (unsigned i = 0; i < n; ++i) {
        nodeBase[i] = totalNodes;
        totalNodes += combOps[i]->getNumResults();
      }

      SmallVector<SmallVector<unsigned>> adj(totalNodes);

      for (unsigned i = 0; i < n; ++i) {
        for (unsigned r = 0; r < combOps[i]->getNumResults(); ++r) {
          if (!combInfos[i].resultMask.test(r))
            continue;
          Value result = combOps[i]->getResult(r);
          for (OpOperand &use : result.getUses()) {
            Operation *user = use.getOwner();
            auto it = opIndex.find(user);
            if (it == opIndex.end())
              continue;
            unsigned userIdx = it->second;
            unsigned opIdx = use.getOperandNumber();
            // Add edge to each result of user that this operand influences.
            for (unsigned s = 0; s < combInfos[userIdx].operandDeps.size();
                 ++s) {
              if (!combInfos[userIdx].resultMask.test(s))
                continue;
              if (opIdx < combInfos[userIdx].operandDeps[s].size() &&
                  combInfos[userIdx].operandDeps[s].test(opIdx))
                adj[nodeBase[i] + r].push_back(nodeBase[userIdx] + s);
            }
          }
        }
      }

      // DFS cycle detection: 0=white, 1=gray, 2=black.
      SmallVector<int> color(totalNodes, 0);
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

      for (unsigned i = 0; i < totalNodes && !hasCombLoop; ++i) {
        if (color[i] == 0)
          dfs(i);
      }

      if (hasCombLoop)
        return emitOpError(cplErrMsg(CplError::ADG_COMBINATIONAL_LOOP,
            "a cycle of purely combinational operations exists; "
            "insert a fabric.fifo or sequential element to break it"));
    }
  }

  // CPL_MEMORY_PRIVATE_OUTPUT: each memref yield operand must trace back
  // to a MemoryOp with is_private = false.
  for (auto [idx, operand] : llvm::enumerate(yield.getOperands())) {
    if (!isa<MemRefType>(operand.getType()))
      continue;
    auto *defOp = operand.getDefiningOp();
    if (!defOp) {
      // Block argument: not produced by a memory op.
      return emitOpError(cplErrMsg(CplError::MEMORY_PRIVATE_OUTPUT,
                         "yield memref operand #"))
             << idx << " is not produced by a fabric.memory with "
             << "is_private = false";
    }
    auto memOp = dyn_cast<MemoryOp>(defOp);
    if (!memOp) {
      return emitOpError(cplErrMsg(CplError::MEMORY_PRIVATE_OUTPUT,
                         "yield memref operand #"))
             << idx << " is not produced by a fabric.memory";
    }
    if (memOp.getIsPrivate()) {
      return emitOpError(cplErrMsg(CplError::MEMORY_PRIVATE_OUTPUT,
                         "yield memref operand #"))
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

/// Get the bit width of a type for routing compatibility checks.
/// Returns std::nullopt for types without a well-defined bit width.
static std::optional<unsigned> getRoutingBitWidth(Type t) {
  if (auto bitsType = dyn_cast<dataflow::BitsType>(t))
    return bitsType.getWidth();
  if (auto intTy = dyn_cast<IntegerType>(t))
    return intTy.getWidth();
  if (isa<Float16Type, BFloat16Type>(t))
    return 16u;
  if (isa<Float32Type>(t))
    return 32u;
  if (isa<Float64Type>(t))
    return 64u;
  return std::nullopt;
}

/// Check whether two types are width-compatible at an instance boundary.
/// This is needed for fabric.instance calls inside fabric.pe bodies, where
/// body block args use native types (i32, f32) but the target named PE
/// interface uses bits<N> types.
/// Rules:
///   - Exact match: always compatible.
///   - Both untagged with matching bit width: compatible
///     (e.g., i32 <-> bits<32>).
///   - Both tagged with matching value bit widths and tag types: compatible
///     (e.g., tagged<i32, i4> <-> tagged<bits<32>, i4>).
///   - NoneType must match exactly.
///   - Mixed native/tagged: never compatible.
static bool isWidthCompatible(Type actual, Type expected) {
  if (actual == expected)
    return true;

  bool isTaggedA = isa<dataflow::TaggedType>(actual);
  bool isTaggedE = isa<dataflow::TaggedType>(expected);

  if (isTaggedA != isTaggedE)
    return false;

  if (isTaggedA) {
    auto tagA = cast<dataflow::TaggedType>(actual);
    auto tagE = cast<dataflow::TaggedType>(expected);
    if (tagA.getTagType() != tagE.getTagType())
      return false;
    auto wA = getRoutingBitWidth(tagA.getValueType());
    auto wE = getRoutingBitWidth(tagE.getValueType());
    if (!wA || !wE)
      return tagA.getValueType() == tagE.getValueType();
    return *wA == *wE;
  }

  if (isa<NoneType>(actual) || isa<NoneType>(expected))
    return actual == expected;

  auto wA = getRoutingBitWidth(actual);
  auto wE = getRoutingBitWidth(expected);
  if (!wA || !wE)
    return false;
  return *wA == *wE;
}

//===----------------------------------------------------------------------===//
// InstanceOp verify
//===----------------------------------------------------------------------===//

LogicalResult InstanceOp::verify() {
  // CPL_INSTANCE_UNRESOLVED: symbol must exist.
  auto *target = lookupBySymName(getOperation(), getModule());
  if (!target)
    return emitOpError(cplErrMsg(CplError::INSTANCE_UNRESOLVED,
                       "referenced symbol '"))
           << getModule() << "' does not exist";

  // CPL_PE_INSTANCE_ILLEGAL_TARGET: inside fabric.pe, only named fabric.pe
  // targets are legal.
  if (getOperation()->getParentOfType<PEOp>() && !isa<PEOp>(target))
    return emitOpError(cplErrMsg(CplError::PE_INSTANCE_ILLEGAL_TARGET,
                       "inside fabric.pe, "
                       "fabric.instance may only target a named fabric.pe; '"))
           << getModule() << "' is not a fabric.pe";

  // CPL_INSTANCE_OPERAND_MISMATCH / CPL_INSTANCE_RESULT_MISMATCH:
  // Compare types against target's function_type.
  // Inside fabric.pe bodies, block args use native types (i32, f32) while
  // the target named PE has bits-typed interface ports. Use width-compatible
  // matching for instances inside PE bodies; exact matching elsewhere.
  auto targetFnType = getTargetFunctionType(target);
  if (targetFnType) {
    auto fnType = *targetFnType;
    bool insidePE = getOperation()->getParentOfType<PEOp>() != nullptr;

    if (getOperands().size() != fnType.getNumInputs())
      return emitOpError(cplErrMsg(CplError::INSTANCE_OPERAND_MISMATCH,
                         "operand count ("))
             << getOperands().size() << ") does not match target input count ("
             << fnType.getNumInputs() << ")";
    for (auto [idx, pair] : llvm::enumerate(
             llvm::zip(getOperandTypes(), fnType.getInputs()))) {
      Type actual = std::get<0>(pair);
      Type expected = std::get<1>(pair);
      if (actual != expected) {
        if (insidePE && isWidthCompatible(actual, expected))
          continue;
        return emitOpError(cplErrMsg(CplError::INSTANCE_OPERAND_MISMATCH,
                           "operand #"))
               << idx << " type " << actual
               << " does not match target input type " << expected;
      }
    }

    if (getResults().size() != fnType.getNumResults())
      return emitOpError(cplErrMsg(CplError::INSTANCE_RESULT_MISMATCH,
                         "result count ("))
             << getResults().size() << ") does not match target result count ("
             << fnType.getNumResults() << ")";
    for (auto [idx, pair] : llvm::enumerate(
             llvm::zip(getResultTypes(), fnType.getResults()))) {
      Type actual = std::get<0>(pair);
      Type expected = std::get<1>(pair);
      if (actual != expected) {
        if (insidePE && isWidthCompatible(actual, expected))
          continue;
        return emitOpError(cplErrMsg(CplError::INSTANCE_RESULT_MISMATCH,
                           "result #"))
               << idx << " type " << actual
               << " does not match target result type " << expected;
      }
    }
  }

  // CPL_INSTANCE_CYCLIC_REFERENCE: check for cycles.
  if (hasCyclicReference(target))
    return emitOpError(cplErrMsg(CplError::INSTANCE_CYCLIC_REFERENCE,
                       "instance of '"))
           << getModule() << "' forms a cyclic reference";

  return success();
}
