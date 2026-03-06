//===-- FabricTopOps.cpp - Fabric top-level operation impls ------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include "loom/Dialect/Fabric/FabricOps.h"
#include "loom/Dialect/Fabric/FabricTypeUtils.h"
#include "loom/Dialect/Dataflow/DataflowTypes.h"
#include "loom/Hardware/Common/FabricConstants.h"
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

/// Compute a union route bitmask from a TemporalSwOp's route_table (ArrayAttr).
/// Each valid slot contributes its route bits (which connected positions are
/// enabled). Returns a SmallBitVector of size `popcount(connectivity_table)`.
/// If no route_table is present, returns an empty (all-false) vector.
static llvm::SmallBitVector
getTemporalSwRouteUnion(Operation *op, ArrayRef<int8_t> ct,
                        unsigned numOutputs, unsigned numInputs) {
  // Count connected positions (popcount of connectivity table).
  unsigned popcount = 0;
  for (int8_t v : ct)
    if (v == 1)
      ++popcount;

  llvm::SmallBitVector unionBits(popcount);
  auto rtAttr = op->getAttrOfType<ArrayAttr>("route_table");
  if (!rtAttr)
    return unionBits;

  for (Attribute slotAttr : rtAttr) {
    auto strAttr = dyn_cast<StringAttr>(slotAttr);
    if (!strAttr)
      continue;
    StringRef slot = strAttr.getValue();
    if (slot.contains("invalid"))
      continue;

    if (slot.starts_with("0x")) {
      // Hex format: extract route bits from slot encoding.
      // Slot format: slot[M+K : M+1] are the K route bits (M = tag width).
      // For union purposes, decode the hex value and extract the route-bit
      // region. Tag width = ceil(log2(numOutputs * numInputs)) but we just
      // need the K route bits at bits [M+K-1 : M]. Since the exact tag width
      // isn't stored on the op, use a simpler approach: parse the hex, mask
      // out the lower tag bits, and take the next popcount bits.
      //
      // Find tag width from interface type (tagged<V, iK>).
      // For inline ops, get tag type from the first SSA operand.
      // For named ops, get it from the function_type attribute.
      unsigned tagWidth = 0;
      if (op->getNumOperands() > 0) {
        if (auto tagged =
                dyn_cast<dataflow::TaggedType>(op->getOperand(0).getType())) {
          if (auto intTy = dyn_cast<IntegerType>(tagged.getTagType()))
            tagWidth = intTy.getWidth();
        }
      } else if (auto ftAttr = op->getAttrOfType<TypeAttr>("function_type")) {
        if (auto ft = dyn_cast<FunctionType>(ftAttr.getValue())) {
          if (!ft.getInputs().empty()) {
            if (auto tagged =
                    dyn_cast<dataflow::TaggedType>(ft.getInputs().front())) {
              if (auto intTy = dyn_cast<IntegerType>(tagged.getTagType()))
                tagWidth = intTy.getWidth();
            }
          }
        }
      }
      // Parse hex value.
      uint64_t hexVal = 0;
      slot.substr(2).getAsInteger(16, hexVal);
      // Route bits are at [tagWidth + popcount - 1 : tagWidth].
      uint64_t routeBits = (hexVal >> tagWidth) & ((1ULL << popcount) - 1);
      for (unsigned b = 0; b < popcount; ++b) {
        if (routeBits & (1ULL << b))
          unionBits.set(b);
      }
    } else {
      // Human-readable format: parse "O[out]<-I[in]" pairs.
      // Map each (out, in) to the route-bit position (row-major order of
      // connected positions in connectivity_table).
      //
      // Build a mapping from (out, in) -> bit position.
      SmallVector<std::pair<unsigned, unsigned>> connectedPos;
      for (unsigned o = 0; o < numOutputs; ++o)
        for (unsigned i = 0; i < numInputs; ++i)
          if (ct[o * numInputs + i] == 1)
            connectedPos.push_back({o, i});

      StringRef routes = slot;
      while (!routes.empty()) {
        size_t oPos = routes.find("O[");
        if (oPos == StringRef::npos)
          break;
        routes = routes.substr(oPos + 2);
        size_t oBracket = routes.find(']');
        if (oBracket == StringRef::npos)
          break;
        unsigned outIdx;
        if (routes.substr(0, oBracket).getAsInteger(10, outIdx))
          break;
        routes = routes.substr(oBracket + 1);
        size_t iPos = routes.find("I[");
        if (iPos == StringRef::npos)
          break;
        routes = routes.substr(iPos + 2);
        size_t iBracket = routes.find(']');
        if (iBracket == StringRef::npos)
          break;
        unsigned inIdx;
        if (routes.substr(0, iBracket).getAsInteger(10, inIdx))
          break;
        routes = routes.substr(iBracket + 1);
        // Find the bit position for (outIdx, inIdx).
        for (unsigned b = 0; b < connectedPos.size(); ++b) {
          if (connectedPos[b].first == outIdx &&
              connectedPos[b].second == inIdx) {
            unionBits.set(b);
            break;
          }
        }
      }
    }
  }
  return unionBits;
}

/// Helper: expand a route bitmask (one bit per connected position) against a
/// connectivity_table into per-output operandDeps. The route bits are ordered
/// row-major over connected positions in the connectivity table.
static CombInfo
expandRouteBitsToInfo(ArrayRef<int8_t> ct, const llvm::SmallBitVector &routeBits,
                      unsigned numResults, unsigned numOperands) {
  CombInfo info;
  info.resultMask.resize(numResults);
  unsigned bitIdx = 0;
  for (unsigned o = 0; o < numResults; ++o) {
    llvm::SmallBitVector deps(numOperands);
    for (unsigned i = 0; i < numOperands; ++i) {
      unsigned flatIdx = o * numOperands + i;
      if (flatIdx < ct.size() && ct[flatIdx] == 1) {
        if (bitIdx < routeBits.size() && routeBits.test(bitIdx))
          deps.set(i);
        ++bitIdx;
      }
    }
    if (deps.any())
      info.resultMask.set(o);
    info.operandDeps.push_back(deps);
  }
  return info;
}

/// Helper: extract combinational path info from a switch-like op.
/// The connectivity_table is a flat DenseI8ArrayAttr of size numOut*numIn,
/// stored output-major: table[o * numIn + i] = 1 means output o receives
/// from input i.
///
/// Configuration-aware: if route_table is absent (unconfigured switch), the
/// switch contributes no combinational edges (empty CombInfo). Only when a
/// route_table is present does the switch become combinational on the paths
/// that the routing configuration actually enables.
static CombInfo
getSwitchCombInfo(Operation *op, unsigned numResults, unsigned numOperands) {
  auto ctAttr = op->getAttrOfType<DenseI8ArrayAttr>("connectivity_table");

  // Synthesize default full-crossbar connectivity if absent (per spec,
  // missing connectivity_table means all-1s full crossbar).
  SmallVector<int8_t> defaultCt;
  auto getConnectivity = [&]() -> ArrayRef<int8_t> {
    if (ctAttr)
      return ctAttr.asArrayRef();
    defaultCt.assign(numResults * numOperands, 1);
    return ArrayRef<int8_t>(defaultCt);
  };

  // SwitchOp: check DenseI8ArrayAttr route_table.
  if (auto rtAttr = op->getAttrOfType<DenseI8ArrayAttr>("route_table")) {
    auto ct = getConnectivity();
    // Build route bitmask from DenseI8 route_table.
    llvm::SmallBitVector routeBits(rtAttr.size());
    for (unsigned b = 0; b < rtAttr.size(); ++b) {
      if (rtAttr[b])
        routeBits.set(b);
    }
    return expandRouteBitsToInfo(ct, routeBits, numResults, numOperands);
  }

  // TemporalSwOp: check ArrayAttr route_table.
  if (auto rtAttr = op->getAttrOfType<ArrayAttr>("route_table")) {
    auto ct = getConnectivity();
    auto unionBits =
        getTemporalSwRouteUnion(op, ct, numResults, numOperands);
    return expandRouteBitsToInfo(ct, unionBits, numResults, numOperands);
  }

  // No route_table: unconfigured switch contributes no combinational edges.
  return CombInfo::empty(numResults);
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

  // Port ordering: memref* -> bits* -> tagged*.
  auto checkOrdering = [&](ArrayRef<Type> types,
                           StringRef label) -> LogicalResult {
    // 0 = memref, 1 = bits/none, 2 = tagged
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
               << " must follow port ordering: memref*, bits*, tagged*";
      lastCat = cat;
    }
    return success();
  };

  if (failed(checkOrdering(fnType.getInputs(), "inputs")))
    return failure();
  if (failed(checkOrdering(fnType.getResults(), "outputs")))
    return failure();

  // Bits-only module boundary: reject native scalar ports.
  // Allowed: memref, none, bits<N>, tagged<bits<N>|none, iK>.
  auto isAllowedModuleType = [](Type t) -> bool {
    if (isa<MemRefType>(t) || isa<NoneType>(t) || isa<dataflow::BitsType>(t))
      return true;
    if (auto tagged = dyn_cast<dataflow::TaggedType>(t)) {
      Type v = tagged.getValueType();
      return isa<NoneType>(v) || isa<dataflow::BitsType>(v);
    }
    return false;
  };
  auto checkModuleTypes = [&](ArrayRef<Type> types,
                              StringRef label) -> LogicalResult {
    for (auto [idx, t] : llvm::enumerate(types)) {
      if (!isAllowedModuleType(t))
        return emitOpError(cplErrMsg(CplError::MODULE_NATIVE_PORT,
                           "module "))
               << label << " port #" << idx
               << " has native type '" << t
               << "'; must use !dataflow.bits<N>, none, memref, or "
                  "!dataflow.tagged<!dataflow.bits<N>|none, iK>";
    }
    return success();
  };
  if (failed(checkModuleTypes(fnType.getInputs(), "input")))
    return failure();
  if (failed(checkModuleTypes(fnType.getResults(), "output")))
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
        return emitOpError(cplErrMsg(CfgError::ADG_COMBINATIONAL_LOOP,
            "route_table configuration creates a combinational loop; "
            "change routing or insert a fabric.fifo to break it"));
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

/// Get the function_type from a target operation. Only PEOp, TemporalPEOp,
/// and ModuleOp are valid instance targets.
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
    auto wA = getNativeBitWidth(tagA.getValueType());
    auto wE = getNativeBitWidth(tagE.getValueType());
    if (!wA || !wE)
      return tagA.getValueType() == tagE.getValueType();
    return *wA == *wE;
  }

  if (isa<NoneType>(actual) || isa<NoneType>(expected))
    return actual == expected;

  auto wA = getNativeBitWidth(actual);
  auto wE = getNativeBitWidth(expected);
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

  // CPL_INSTANCE_ILLEGAL_TARGET: only PE, TemporalPE, and ModuleOp are
  // valid instance targets. Switch, FIFO, memory, and extmemory should be
  // used inline.
  if (!isa<PEOp, TemporalPEOp, ModuleOp>(target))
    return emitOpError(cplErrMsg(CplError::INSTANCE_ILLEGAL_TARGET,
                       "fabric.instance may only target fabric.pe, "
                       "fabric.temporal_pe, or fabric.module; '"))
           << getModule() << "' is not a valid target";

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
  // matching only when the target port is bits-typed (the native-to-bits
  // bridge case); native-to-native calls use exact matching.
  auto targetFnType = getTargetFunctionType(target);
  if (targetFnType) {
    auto fnType = *targetFnType;
    bool insidePE = getOperation()->getParentOfType<PEOp>() != nullptr;

    // Width-compatible matching only applies to native↔bits bridge.
    auto isBitsTarget = [](Type expected) -> bool {
      Type valT = expected;
      if (auto tagged = dyn_cast<dataflow::TaggedType>(expected))
        valT = tagged.getValueType();
      return isa<dataflow::BitsType>(valT);
    };

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
        if (insidePE && isBitsTarget(expected) &&
            isWidthCompatible(actual, expected))
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
        if (insidePE && isBitsTarget(expected) &&
            isWidthCompatible(actual, expected))
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
