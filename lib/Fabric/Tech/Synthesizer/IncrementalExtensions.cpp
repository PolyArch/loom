// Candidate generators for `IncrementalSynthesizer::run`. Factored out
// of `Incremental.cpp` so the main left-fold loop stays focused on
// control flow and the per-extension synthesis primitives live in
// their own translation unit.
//
// Extension generators:
//   * `widenOplistCandidates` -- for each diff site whose FU op and sg
//     op share a hardware share-group + width, generate a candidate
//     FU whose `op_list` at that position is the sorted union.
//   * `insertMuxDemuxCandidates` -- tier B baseline. Detects (a) sg has
//     one extra tail op the FU must learn to realize, or (b) the FU
//     has one extra head op sg cannot reach; in either case inserts a
//     `fabric.demux` + `fabric.mux` pair so both shapes can be
//     materialized from the same FU.
//   * `hasBackEdgeInDiff` -- predicate gating the tier-C structural
//     extension hook (defined in `IncrementalExtensionsTierC.cpp`).
//
// Spec source: `docs/spec-generalize-subgraphs-to-fu.md`, section
// "Strategy: incremental > extend_to_cover".

#include "IncrementalExtensions.h"

#include "Common/HwShareGroup.h"
#include "Common/IndexWidth.h"
#include "Fabric/IR/FabricOps.h"
#include "Fabric/IR/FabricTypes.h"
#include "Fabric/Tech/Synthesizer/Alignment.h"
#include "Fabric/Tech/Synthesizer/HwParams.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <algorithm>
#include <optional>
#include <set>
#include <string>
#include <utility>

namespace loom::fabric::tech::detail {

namespace {

//===----------------------------------------------------------------------===//
// Common helpers (mirrored from Incremental.cpp's anonymous namespace).
// Keeping them duplicated keeps the public Incremental.h surface small;
// the alternative would be to lift them into Alignment.h, which the
// follow-up tier-C task will do once the SCC alignment APIs need the
// same primitives.
//===----------------------------------------------------------------------===//

unsigned bitWidthOf(::mlir::Type t) {
  if (auto i = ::llvm::dyn_cast<::mlir::IntegerType>(t))
    return i.getWidth();
  if (auto f = ::llvm::dyn_cast<::mlir::FloatType>(t))
    return f.getWidth();
  if (::llvm::isa<::mlir::IndexType>(t))
    return ::loom::getIndexWidth();
  return 0;
}

::fabric::FuOp innerFuOf(::mlir::func::FuncOp wrapper) {
  if (!wrapper || wrapper.getBody().empty())
    return {};
  for (::mlir::Operation &op : wrapper.getBody().front().getOperations())
    if (auto fu = ::mlir::dyn_cast<::fabric::FuOp>(op))
      return fu;
  return {};
}

::llvm::StringRef firstOpListSymbol(::fabric::OpOp op) {
  ::mlir::ArrayAttr opList = op.getOpList();
  if (opList.empty())
    return {};
  auto sym = ::llvm::dyn_cast<::mlir::FlatSymbolRefAttr>(opList[0]);
  if (!sym)
    return {};
  return sym.getValue();
}

::mlir::ArrayAttr sortedOpList(const ::std::set<::std::string> &names,
                               ::mlir::MLIRContext *ctx) {
  ::llvm::SmallVector<::mlir::Attribute, 4> attrs;
  attrs.reserve(names.size());
  for (const ::std::string &n : names)
    attrs.push_back(::mlir::FlatSymbolRefAttr::get(ctx, n));
  return ::mlir::ArrayAttr::get(ctx, attrs);
}

//===----------------------------------------------------------------------===//
// Yield-anchor chain walking. Each call walks back from a single yield
// operand position, collecting the linear sequence of fabric.ops (FU
// side) or dataflow ops (sg side) along operand 0. This is enough for
// tier B baseline; richer DAG matching belongs to the MCS strategy.
//===----------------------------------------------------------------------===//

struct FuYieldChain {
  ::llvm::SmallVector<::fabric::OpOp, 4> ops;
  ::mlir::Value head;
};

FuYieldChain collectFuYieldChain(::fabric::FuOp fu, unsigned yieldIdx) {
  FuYieldChain c;
  if (!fu)
    return c;
  ::mlir::Block &body = fu.getBody().front();
  ::mlir::Operation *yield = body.getTerminator();
  if (!yield || yieldIdx >= yield->getNumOperands())
    return c;
  ::mlir::Value v = yield->getOperand(yieldIdx);
  c.head = v;
  ::mlir::Value cur = v;
  while (cur) {
    auto def = cur.getDefiningOp();
    if (!def)
      break;
    auto opOp = ::mlir::dyn_cast<::fabric::OpOp>(def);
    if (!opOp)
      break;
    c.ops.push_back(opOp);
    if (opOp.getInputs().empty())
      break;
    cur = opOp.getInputs()[0];
  }
  return c;
}

struct SgYieldChain {
  ::llvm::SmallVector<::mlir::Operation *, 4> ops;
  ::llvm::SmallVector<::llvm::StringRef, 4> names;
  unsigned terminalArg = ~0u;
};

SgYieldChain collectSgYieldChain(::dataflow::SubgraphOp sg, unsigned yieldIdx) {
  SgYieldChain c;
  if (!sg)
    return c;
  ::mlir::Block &body = sg.getBody().front();
  ::mlir::Operation *yield = body.getTerminator();
  if (!yield || yieldIdx >= yield->getNumOperands())
    return c;
  ::mlir::Value v = yield->getOperand(yieldIdx);
  while (v) {
    if (auto barg = ::llvm::dyn_cast<::mlir::BlockArgument>(v)) {
      c.terminalArg = barg.getArgNumber();
      break;
    }
    auto opRes = ::llvm::dyn_cast<::mlir::OpResult>(v);
    if (!opRes)
      break;
    ::mlir::Operation *def = opRes.getOwner();
    c.ops.push_back(def);
    c.names.push_back(def->getName().getStringRef());
    if (def->getNumOperands() == 0)
      break;
    v = def->getOperand(0);
  }
  return c;
}

//===----------------------------------------------------------------------===//
// Cloning + position-based fabric.op lookup helpers.
//===----------------------------------------------------------------------===//

::mlir::OwningOpRef<::mlir::func::FuncOp>
cloneWrapper(::mlir::func::FuncOp wrapper) {
  if (!wrapper)
    return {};
  ::mlir::Operation *clonedRaw = wrapper->clone();
  return ::mlir::OwningOpRef<::mlir::func::FuncOp>(
      ::mlir::cast<::mlir::func::FuncOp>(clonedRaw));
}

::fabric::OpOp findFabricOpByIndex(::mlir::func::FuncOp wrapper,
                                   unsigned targetIdx) {
  ::fabric::FuOp fu = innerFuOf(wrapper);
  if (!fu)
    return {};
  unsigned i = 0;
  for (::mlir::Operation &raw : fu.getBody().front().getOperations()) {
    auto op = ::mlir::dyn_cast<::fabric::OpOp>(raw);
    if (!op)
      continue;
    if (i == targetIdx)
      return op;
    ++i;
  }
  return {};
}

unsigned indexOfFabricOp(::fabric::FuOp fu, ::fabric::OpOp op) {
  if (!fu || !op)
    return ~0u;
  unsigned i = 0;
  for (::mlir::Operation &raw : fu.getBody().front().getOperations()) {
    auto cur = ::mlir::dyn_cast<::fabric::OpOp>(raw);
    if (!cur)
      continue;
    if (cur == op)
      return i;
    ++i;
  }
  return ~0u;
}

//===----------------------------------------------------------------------===//
// fabric.op / mux / demux emission helpers (mirrored from Anchor.cpp's
// equivalents; kept private to this TU).
//===----------------------------------------------------------------------===//

::fabric::OpOp emitFabricOp(::mlir::OpBuilder &builder, ::mlir::Location loc,
                            ::mlir::ArrayAttr opList,
                            ::mlir::ArrayAttr hwParams,
                            ::mlir::ValueRange operands,
                            ::mlir::Type resultType) {
  ::mlir::OperationState state(loc, ::fabric::OpOp::getOperationName());
  state.addOperands(operands);
  state.addTypes({resultType});
  state.addAttribute("op_list", opList);
  if (hwParams)
    state.addAttribute("hw_params", hwParams);
  return ::mlir::cast<::fabric::OpOp>(builder.create(state));
}

::fabric::DemuxOp emitDemux2(::mlir::OpBuilder &builder, ::mlir::Location loc,
                             ::mlir::Value input, ::mlir::Type bits) {
  ::mlir::OperationState state(loc, ::fabric::DemuxOp::getOperationName());
  state.addOperands({input});
  state.addTypes({bits, bits});
  return ::mlir::cast<::fabric::DemuxOp>(builder.create(state));
}

::fabric::MuxOp emitMux2(::mlir::OpBuilder &builder, ::mlir::Location loc,
                         ::mlir::Value a, ::mlir::Value b, ::mlir::Type bits) {
  ::mlir::OperationState state(loc, ::fabric::MuxOp::getOperationName());
  state.addOperands({a, b});
  state.addTypes({bits});
  return ::mlir::cast<::fabric::MuxOp>(builder.create(state));
}

//===----------------------------------------------------------------------===//
// Tail-extension detection (sg longer than FU by 1 along a yield chain)
// and FU-extra-head detection (FU longer than sg by 1).
//===----------------------------------------------------------------------===//

struct TailExtension {
  ::llvm::StringRef opName;
  ::llvm::SmallVector<unsigned, 4> operandArgIdx;
  unsigned resultBw = 0;
  ::mlir::Type resultType;
  // Source-side op the tail mirrors. Used by buildHwParamsUnion so the
  // synthesized fabric.op carries the right `predicate` / `step_op` /
  // etc. allowed-set rather than `[{}]`.
  ::mlir::Operation *srcOp = nullptr;
};

::std::optional<TailExtension>
detectSingleTailExtension(::fabric::FuOp fu, unsigned yieldIdx,
                          ::dataflow::SubgraphOp sg) {
  if (!fu || !sg)
    return std::nullopt;
  FuYieldChain fchain = collectFuYieldChain(fu, yieldIdx);
  SgYieldChain schain = collectSgYieldChain(sg, yieldIdx);
  if (schain.ops.empty())
    return std::nullopt;
  if (schain.ops.size() != fchain.ops.size() + 1)
    return std::nullopt;
  for (unsigned i = 0; i < fchain.ops.size(); ++i) {
    ::llvm::StringRef fname = firstOpListSymbol(fchain.ops[i]);
    ::llvm::StringRef sname = schain.names[i + 1];
    if (fname != sname)
      return std::nullopt;
  }
  ::mlir::Operation *extra = schain.ops[0];
  if (extra->getNumResults() != 1)
    return std::nullopt;
  unsigned bw = bitWidthOf(extra->getResult(0).getType());
  if (bw == 0)
    return std::nullopt;
  TailExtension t;
  t.opName = extra->getName().getStringRef();
  t.resultBw = bw;
  t.resultType = extra->getResult(0).getType();
  t.srcOp = extra;
  t.operandArgIdx.reserve(extra->getNumOperands());
  for (unsigned i = 0; i < extra->getNumOperands(); ++i) {
    ::mlir::Value v = extra->getOperand(i);
    if (i == 0) {
      t.operandArgIdx.push_back(~0u);
      continue;
    }
    auto barg = ::llvm::dyn_cast<::mlir::BlockArgument>(v);
    if (!barg)
      return std::nullopt;
    t.operandArgIdx.push_back(barg.getArgNumber());
  }
  return t;
}

::std::optional<unsigned>
detectFuExtraHead(::fabric::FuOp fu, unsigned yieldIdx,
                  ::dataflow::SubgraphOp sg) {
  if (!fu || !sg)
    return std::nullopt;
  FuYieldChain fchain = collectFuYieldChain(fu, yieldIdx);
  SgYieldChain schain = collectSgYieldChain(sg, yieldIdx);
  if (fchain.ops.empty())
    return std::nullopt;
  if (fchain.ops.size() != schain.ops.size() + 1)
    return std::nullopt;
  for (unsigned i = 0; i < schain.ops.size(); ++i) {
    ::llvm::StringRef fname = firstOpListSymbol(fchain.ops[i + 1]);
    ::llvm::StringRef sname = schain.names[i];
    if (fname != sname)
      return std::nullopt;
  }
  return 0u;
}

//===----------------------------------------------------------------------===//
// Build a candidate that grafts the tail extension onto the FU.
//===----------------------------------------------------------------------===//

::mlir::OwningOpRef<::mlir::func::FuncOp>
buildMuxDemuxCandidate(::mlir::func::FuncOp curWrapper, unsigned yieldIdx,
                       const TailExtension &tail,
                       ::dataflow::SubgraphOp sg) {
  if (!curWrapper)
    return {};
  ::mlir::MLIRContext *ctx = curWrapper.getContext();
  ::mlir::Location loc = ::mlir::UnknownLoc::get(ctx);
  ::mlir::Block &sgBlock = sg.getBody().front();

  ::llvm::SmallVector<unsigned, 4> appendedSgArg;
  for (unsigned a : tail.operandArgIdx)
    if (a != ~0u)
      appendedSgArg.push_back(a);
  if (appendedSgArg.empty())
    return {};

  ::fabric::FuOp curFu = innerFuOf(curWrapper);
  if (!curFu)
    return {};

  auto oldType = curWrapper.getFunctionType();
  ::llvm::SmallVector<::mlir::Type, 4> newInputTypes(oldType.getInputs().begin(),
                                                     oldType.getInputs().end());
  ::llvm::SmallVector<::mlir::Type, 4> newPortBitsTypes;
  newPortBitsTypes.reserve(appendedSgArg.size());
  for (unsigned a : appendedSgArg) {
    if (a >= sgBlock.getNumArguments())
      return {};
    unsigned bw = bitWidthOf(sgBlock.getArgument(a).getType());
    if (bw == 0)
      return {};
    auto bits = ::fabric::BitsType::get(ctx, bw);
    newInputTypes.push_back(bits);
    newPortBitsTypes.push_back(bits);
  }
  ::llvm::SmallVector<::mlir::Type, 4> newResultTypes(oldType.getResults().begin(),
                                                      oldType.getResults().end());

  auto newFuncType =
      ::mlir::FunctionType::get(ctx, newInputTypes, newResultTypes);
  ::std::string symName = curWrapper.getName().str();
  auto newWrapper = ::mlir::func::FuncOp::create(loc, symName, newFuncType);
  ::mlir::Block *newEntry = newWrapper.addEntryBlock();

  ::mlir::OperationState fuState(loc, ::fabric::FuOp::getOperationName());
  fuState.addOperands(::mlir::ValueRange(newEntry->getArguments()));
  fuState.addTypes(newResultTypes);
  ::mlir::Region *fuRegion = fuState.addRegion();
  ::mlir::Block *fuEntry = new ::mlir::Block();
  fuRegion->push_back(fuEntry);
  ::llvm::SmallVector<::mlir::Location, 4> fuArgLocs(newInputTypes.size(), loc);
  fuEntry->addArguments(newInputTypes, fuArgLocs);
  ::mlir::OpBuilder funcBuilder(newEntry, newEntry->end());
  ::mlir::Operation *rawNewFu = funcBuilder.create(fuState);
  auto newFu = ::mlir::cast<::fabric::FuOp>(rawNewFu);

  ::mlir::OpBuilder bodyBuilder(fuEntry, fuEntry->end());

  ::mlir::IRMapping mapping;
  ::mlir::Block &oldBlock = curFu.getBody().front();
  for (unsigned i = 0; i < oldBlock.getNumArguments(); ++i)
    mapping.map(oldBlock.getArgument(i), fuEntry->getArgument(i));

  ::mlir::Operation *oldYield = oldBlock.getTerminator();
  for (::mlir::Operation &raw : oldBlock.without_terminator())
    bodyBuilder.clone(raw, mapping);

  if (!oldYield || yieldIdx >= oldYield->getNumOperands())
    return {};
  ::mlir::Value oldHead = oldYield->getOperand(yieldIdx);
  ::mlir::Value newHead = mapping.lookupOrNull(oldHead);
  if (!newHead)
    return {};

  auto bits = ::llvm::dyn_cast<::fabric::BitsType>(newHead.getType());
  if (!bits)
    return {};
  ::fabric::DemuxOp demux = emitDemux2(bodyBuilder, loc, newHead, bits);
  ::mlir::Value armToYield = demux.getOutputs()[0];
  ::mlir::Value armToTail = demux.getOutputs()[1];

  unsigned firstAppendedIdx = oldBlock.getNumArguments();
  ::llvm::SmallVector<::mlir::Value, 4> extraOperands;
  extraOperands.reserve(tail.operandArgIdx.size());
  unsigned appendedSeen = 0;
  for (unsigned a : tail.operandArgIdx) {
    if (a == ~0u) {
      extraOperands.push_back(armToTail);
    } else {
      if (appendedSeen >= newPortBitsTypes.size())
        return {};
      extraOperands.push_back(
          fuEntry->getArgument(firstAppendedIdx + appendedSeen));
      ++appendedSeen;
    }
  }
  ::mlir::ArrayAttr opList =
      sortedOpList({tail.opName.str()}, ctx);
  ::mlir::Operation *peers[1] = {tail.srcOp};
  ::mlir::ArrayAttr hwParams = ::loom::fabric::tech::buildHwParamsUnion(
      ctx, tail.opName,
      tail.srcOp ? ::llvm::ArrayRef<::mlir::Operation *>(peers, 1)
                 : ::llvm::ArrayRef<::mlir::Operation *>());
  auto tailOp = emitFabricOp(bodyBuilder, loc, opList, hwParams, extraOperands,
                             bits);

  ::fabric::MuxOp mux = emitMux2(bodyBuilder, loc, armToYield,
                                 tailOp.getOutputs()[0], bits);

  ::llvm::SmallVector<::mlir::Value, 4> newYieldOperands;
  newYieldOperands.reserve(oldYield->getNumOperands());
  for (unsigned i = 0; i < oldYield->getNumOperands(); ++i) {
    if (i == yieldIdx) {
      newYieldOperands.push_back(mux.getOutput());
    } else {
      ::mlir::Value v = mapping.lookupOrNull(oldYield->getOperand(i));
      if (!v)
        return {};
      newYieldOperands.push_back(v);
    }
  }
  ::mlir::OperationState yieldState(loc,
                                    ::fabric::YieldOp::getOperationName());
  yieldState.addOperands(newYieldOperands);
  bodyBuilder.create(yieldState);

  ::mlir::OperationState retState(
      loc, ::mlir::func::ReturnOp::getOperationName());
  retState.addOperands(::mlir::ValueRange(newFu.getResults()));
  funcBuilder.create(retState);

  return ::mlir::OwningOpRef<::mlir::func::FuncOp>(newWrapper);
}

//===----------------------------------------------------------------------===//
// Build a candidate that inserts demux/mux around the FU's extra head
// op so the FU can also realize the shorter sg shape.
//===----------------------------------------------------------------------===//

::mlir::OwningOpRef<::mlir::func::FuncOp>
buildSkipHeadCandidate(::mlir::func::FuncOp curWrapper, unsigned yieldIdx) {
  if (!curWrapper)
    return {};
  ::mlir::MLIRContext *ctx = curWrapper.getContext();
  ::mlir::Location loc = ::mlir::UnknownLoc::get(ctx);
  ::fabric::FuOp curFu = innerFuOf(curWrapper);
  if (!curFu)
    return {};

  FuYieldChain fchain = collectFuYieldChain(curFu, yieldIdx);
  if (fchain.ops.empty())
    return {};
  ::fabric::OpOp headOp = fchain.ops[0];
  if (!headOp || headOp.getInputs().empty())
    return {};
  unsigned headIdx = indexOfFabricOp(curFu, headOp);
  if (headIdx == ~0u)
    return {};

  auto oldType = curWrapper.getFunctionType();
  ::std::string symName = curWrapper.getName().str();
  auto newWrapper = ::mlir::func::FuncOp::create(loc, symName, oldType);
  ::mlir::Block *newEntry = newWrapper.addEntryBlock();

  ::mlir::OperationState fuState(loc, ::fabric::FuOp::getOperationName());
  fuState.addOperands(::mlir::ValueRange(newEntry->getArguments()));
  fuState.addTypes(oldType.getResults());
  ::mlir::Region *fuRegion = fuState.addRegion();
  ::mlir::Block *fuEntry = new ::mlir::Block();
  fuRegion->push_back(fuEntry);
  ::llvm::SmallVector<::mlir::Type, 4> inTypes(oldType.getInputs().begin(),
                                               oldType.getInputs().end());
  ::llvm::SmallVector<::mlir::Location, 4> argLocs(inTypes.size(), loc);
  fuEntry->addArguments(inTypes, argLocs);
  ::mlir::OpBuilder funcBuilder(newEntry, newEntry->end());
  ::mlir::Operation *rawNewFu = funcBuilder.create(fuState);
  auto newFu = ::mlir::cast<::fabric::FuOp>(rawNewFu);

  ::mlir::OpBuilder bodyBuilder(fuEntry, fuEntry->end());
  ::mlir::IRMapping mapping;
  ::mlir::Block &oldBlock = curFu.getBody().front();
  for (unsigned i = 0; i < oldBlock.getNumArguments(); ++i)
    mapping.map(oldBlock.getArgument(i), fuEntry->getArgument(i));

  ::mlir::Operation *oldYield = oldBlock.getTerminator();
  ::fabric::OpOp newHeadOp;
  for (::mlir::Operation &raw : oldBlock.without_terminator()) {
    ::mlir::Operation *cloned = bodyBuilder.clone(raw, mapping);
    if (auto h = ::mlir::dyn_cast<::fabric::OpOp>(&raw))
      if (h == headOp)
        newHeadOp = ::mlir::cast<::fabric::OpOp>(cloned);
  }
  if (!newHeadOp)
    return {};
  ::mlir::Value preHead = newHeadOp.getInputs()[0];
  auto bits = ::llvm::dyn_cast<::fabric::BitsType>(preHead.getType());
  if (!bits)
    return {};

  bodyBuilder.setInsertionPointAfterValue(preHead);
  ::fabric::DemuxOp demux = emitDemux2(bodyBuilder, loc, preHead, bits);
  ::mlir::Value armToHead = demux.getOutputs()[0];
  ::mlir::Value armToMux = demux.getOutputs()[1];
  newHeadOp.setOperand(0, armToHead);

  bodyBuilder.setInsertionPointAfter(newHeadOp);
  ::mlir::Value headOut = newHeadOp.getOutputs()[0];
  ::fabric::MuxOp mux = emitMux2(bodyBuilder, loc, headOut, armToMux, bits);

  ::llvm::SmallVector<::mlir::Value, 4> newYieldOperands;
  newYieldOperands.reserve(oldYield->getNumOperands());
  for (unsigned i = 0; i < oldYield->getNumOperands(); ++i) {
    ::mlir::Value mapped = mapping.lookupOrNull(oldYield->getOperand(i));
    if (!mapped)
      return {};
    if (i == yieldIdx)
      newYieldOperands.push_back(mux.getOutput());
    else
      newYieldOperands.push_back(mapped);
  }
  bodyBuilder.setInsertionPointToEnd(fuEntry);
  ::mlir::OperationState yieldState(loc,
                                    ::fabric::YieldOp::getOperationName());
  yieldState.addOperands(newYieldOperands);
  bodyBuilder.create(yieldState);

  ::mlir::OperationState retState(
      loc, ::mlir::func::ReturnOp::getOperationName());
  retState.addOperands(::mlir::ValueRange(newFu.getResults()));
  funcBuilder.create(retState);

  return ::mlir::OwningOpRef<::mlir::func::FuncOp>(newWrapper);
}

//===----------------------------------------------------------------------===//
// Recursive-compression candidate (spec Q12). Builds the baseline
// mux/demux tail-extension candidate, then walks the existing FU body
// for a fabric.op whose `op_list[0]` shares a hardware share-group +
// result width with the new tail op. When such a match exists, the
// tail's op_list is widened to the union of (tail name, matched name)
// so the synthesized fabric.op carries the share-group anchor and
// downstream cost ranking can credit the share-aware merge. Returns
// `nullptr` when no qualifying matched op is found, signalling that
// no extra recursive-compression candidate is produced for this
// diff site.
//
// Producing a structurally identical candidate with a widened tail
// op_list keeps the IR strictly verifier-clean (no new operand
// rewiring beyond the baseline) while emitting a behavior-changing
// extra candidate that the cost model can rank against the baseline.
::mlir::OwningOpRef<::mlir::func::FuncOp>
buildShareRecurseCandidate(::mlir::func::FuncOp curWrapper, unsigned yieldIdx,
                           const TailExtension &tail,
                           ::dataflow::SubgraphOp sg) {
  if (!curWrapper)
    return {};
  ::fabric::FuOp fu = innerFuOf(curWrapper);
  if (!fu)
    return {};

  // Find an existing FU body fabric.op whose op_list[0] shares a
  // hardware share-group + result width with the new tail op. Walk in
  // body order so the first qualifying anchor wins deterministically.
  ::llvm::StringRef anchorName;
  for (::mlir::Operation &raw : fu.getBody().front().getOperations()) {
    auto fop = ::mlir::dyn_cast<::fabric::OpOp>(raw);
    if (!fop)
      continue;
    if (fop->getNumResults() != 1)
      continue;
    auto fBits = ::llvm::dyn_cast<::fabric::BitsType>(
        fop->getResult(0).getType());
    if (!fBits || fBits.getWidth() != tail.resultBw)
      continue;
    ::llvm::StringRef fName = firstOpListSymbol(fop);
    if (fName.empty() || fName == tail.opName)
      continue;
    if (!::loom::common::sameShareGroup(fName, tail.opName))
      continue;
    anchorName = fName;
    break;
  }
  if (anchorName.empty())
    return {};

  // Build the baseline mux/demux candidate first; the recursive variant
  // is structurally identical with a widened tail op_list.
  auto baseline = buildMuxDemuxCandidate(curWrapper, yieldIdx, tail, sg);
  if (!baseline)
    return {};

  // Locate the freshly emitted tail fabric.op in the cloned body. It is
  // the unique fabric.op whose op_list[0] equals `tail.opName` and whose
  // result width matches `tail.resultBw`. Walk in body order so the
  // first match wins deterministically.
  ::fabric::FuOp newFu = innerFuOf(baseline.get());
  if (!newFu)
    return {};
  ::fabric::OpOp tailFabricOp;
  for (::mlir::Operation &raw : newFu.getBody().front().getOperations()) {
    auto fop = ::mlir::dyn_cast<::fabric::OpOp>(raw);
    if (!fop)
      continue;
    if (fop->getNumResults() != 1)
      continue;
    auto fBits = ::llvm::dyn_cast<::fabric::BitsType>(
        fop->getResult(0).getType());
    if (!fBits || fBits.getWidth() != tail.resultBw)
      continue;
    ::llvm::StringRef fName = firstOpListSymbol(fop);
    if (fName != tail.opName)
      continue;
    tailFabricOp = fop;
    break;
  }
  if (!tailFabricOp)
    return {};

  // Union the tail op's existing op_list with the matched anchor name
  // so the synthesized fabric.op carries both names in its share-group
  // anchor list. `sortedOpList` keeps the union deterministic.
  ::std::set<::std::string> names;
  for (::mlir::Attribute a : tailFabricOp.getOpList())
    if (auto sym = ::llvm::dyn_cast<::mlir::FlatSymbolRefAttr>(a))
      names.insert(sym.getValue().str());
  names.insert(anchorName.str());
  ::mlir::ArrayAttr widened =
      sortedOpList(names, baseline->getContext());
  tailFabricOp->setAttr("op_list", widened);

  return baseline;
}

} // namespace

//===----------------------------------------------------------------------===//
// Public extension entry points.
//===----------------------------------------------------------------------===//

::llvm::SmallVector<::mlir::OwningOpRef<::mlir::func::FuncOp>, 4>
widenOplistCandidates(::mlir::func::FuncOp curWrapper,
                      ::dataflow::SubgraphOp sg) {
  ::llvm::SmallVector<::mlir::OwningOpRef<::mlir::func::FuncOp>, 4> out;
  if (!curWrapper || !sg)
    return out;
  ::fabric::FuOp fu = innerFuOf(curWrapper);
  if (!fu)
    return out;

  ::mlir::MLIRContext *ctx = curWrapper.getContext();
  unsigned yieldArity = 0;
  if (auto term = fu.getBody().front().getTerminator())
    yieldArity = term->getNumOperands();
  unsigned sgYieldArity = 0;
  if (auto t = sg.getBody().front().getTerminator())
    sgYieldArity = t->getNumOperands();
  if (yieldArity != sgYieldArity)
    return out;

  for (unsigned k = 0; k < yieldArity; ++k) {
    FuYieldChain fchain = collectFuYieldChain(fu, k);
    SgYieldChain schain = collectSgYieldChain(sg, k);
    unsigned m = ::std::min(fchain.ops.size(), schain.ops.size());
    for (unsigned i = 0; i < m; ++i) {
      ::fabric::OpOp fop = fchain.ops[i];
      ::mlir::Operation *sop = schain.ops[i];
      if (!fop || !sop)
        continue;
      ::llvm::StringRef sName = sop->getName().getStringRef();
      ::llvm::StringRef fName = firstOpListSymbol(fop);
      if (sName.empty() || fName.empty())
        continue;
      if (fop->getNumResults() != 1 || sop->getNumResults() != 1)
        continue;
      auto fBits = ::llvm::dyn_cast<::fabric::BitsType>(
          fop->getResult(0).getType());
      unsigned sw = bitWidthOf(sop->getResult(0).getType());
      if (!fBits || fBits.getWidth() != sw)
        continue;
      if (fop->getNumOperands() != sop->getNumOperands())
        continue;
      bool alreadyIn = false;
      for (::mlir::Attribute a : fop.getOpList()) {
        if (auto sym = ::llvm::dyn_cast<::mlir::FlatSymbolRefAttr>(a))
          if (sym.getValue() == sName) {
            alreadyIn = true;
            break;
          }
      }
      if (alreadyIn)
        continue;
      if (!::loom::common::sameShareGroup(fName, sName))
        continue;
      ::std::set<::std::string> names;
      for (::mlir::Attribute a : fop.getOpList())
        if (auto sym = ::llvm::dyn_cast<::mlir::FlatSymbolRefAttr>(a))
          names.insert(sym.getValue().str());
      names.insert(sName.str());
      ::mlir::ArrayAttr widened = sortedOpList(names, ctx);

      auto cloned = cloneWrapper(curWrapper);
      if (!cloned)
        continue;
      unsigned targetIdx = indexOfFabricOp(fu, fop);
      if (targetIdx == ~0u)
        continue;
      ::fabric::OpOp clonedOp = findFabricOpByIndex(cloned.get(), targetIdx);
      if (!clonedOp)
        continue;
      clonedOp->setAttr("op_list", widened);
      out.push_back(std::move(cloned));
    }
  }
  return out;
}

::llvm::SmallVector<::mlir::OwningOpRef<::mlir::func::FuncOp>, 4>
insertMuxDemuxCandidates(::mlir::func::FuncOp curWrapper,
                         ::dataflow::SubgraphOp sg,
                         const ::loom::SynthConfig &cfg) {
  ::llvm::SmallVector<::mlir::OwningOpRef<::mlir::func::FuncOp>, 4> out;
  if (!curWrapper || !sg)
    return out;
  ::fabric::FuOp fu = innerFuOf(curWrapper);
  if (!fu)
    return out;
  unsigned yieldArity = 0;
  if (auto term = fu.getBody().front().getTerminator())
    yieldArity = term->getNumOperands();
  unsigned sgYieldArity = 0;
  if (auto t = sg.getBody().front().getTerminator())
    sgYieldArity = t->getNumOperands();
  if (yieldArity != sgYieldArity)
    return out;
  for (unsigned k = 0; k < yieldArity; ++k) {
    auto tail = detectSingleTailExtension(fu, k, sg);
    if (tail.has_value()) {
      auto cand = buildMuxDemuxCandidate(curWrapper, k, *tail, sg);
      if (cand)
        out.push_back(std::move(cand));
      // Recursive-compression candidate (spec Q12): when enabled, emit
      // one extra candidate that widens an existing FU body fabric.op's
      // op_list to absorb the new tail op when both share a hardware
      // share-group + result width. This produces a candidate with the
      // same fabric.op count and structure as the baseline but with the
      // new tail position's op_list pre-widened so cost-rank can favor
      // share-aware merging when the workload exposes the opportunity.
      if (cfg.subgraphShareRecurse) {
        auto rec = buildShareRecurseCandidate(curWrapper, k, *tail, sg);
        if (rec)
          out.push_back(std::move(rec));
      }
      continue;
    }
    auto extra = detectFuExtraHead(fu, k, sg);
    if (extra.has_value()) {
      auto cand = buildSkipHeadCandidate(curWrapper, k);
      if (cand)
        out.push_back(std::move(cand));
      continue;
    }
  }
  return out;
}

bool hasBackEdgeInDiff(::mlir::func::FuncOp /*curWrapper*/,
                       ::dataflow::SubgraphOp sg) {
  if (!sg)
    return false;
  return !backEdges(sg).empty();
}

} // namespace loom::fabric::tech::detail
