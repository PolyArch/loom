// Tokenize residual `llvm.load` / `llvm.store` ops inside
// `dataflow.graph` bodies into the streaming-memory primitives
// `dataflow.load` / `dataflow.store`. The resulting ops bind the
// pointer to a memref via `unrealized_conversion_cast`, drive the
// address port from an `index`-typed offset, and accept the graph
// body's `thread_ctrl` block argument as the `none`-typed firing
// token.
//
// Two recognition modes:
//   * Top-level (load/store sits directly in the graph entry block):
//     require the chain to terminate at a graph block argument so
//     the rewrite preserves the brief's "block-arg base" contract.
//        - gep %arg_ptr[%idx]   -> (arg_ptr, idx)
//        - dataflow.carry init=arg_ptr  -> (carry_result, null)
//        - graph block-arg !llvm.ptr   -> (block_arg, null)
//
//   * Nested (load/store sits inside an scf.for / scf.if region of
//     the graph body): apply a permissive fallback that accepts any
//     !llvm.ptr SSA value as a per-iteration bridge target. The
//     bridge cast is emitted at the load/store site rather than
//     hoisted, so the per-iteration pointer is reinterpreted as a
//     memref<?xT> and read at offset 0 (or at the gep's idx).
//
// Loads whose pointer is none of the above (e.g. derived from
// `llvm.alloca` or `llvm.mlir.addressof` at the top level) remain untouched
// during recognition and are rejected by the residual gate. Raw LLVM memory
// operations do not provide the canonical dataflow completion contract.
// Pointer-element loads are also skipped: bridging !llvm.ptr-of-ptr to
// memref<?x!llvm.ptr> trips the streaming load verifier.

#include "Frontend/Lowering/Passes.h"

#include "GraphMemoryAddressing.h"
#include "GraphRegionLowering.h"
#include "StreamOrdinal.h"

#include "Common/IndexWidth.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Dataflow/IR/DataflowOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/DataLayout.h"

namespace {

using ::loom::lowering::LinearByteTerm;

bool isGraphPtrBlockArg(::mlir::Value v, ::dataflow::GraphOp graph) {
  auto blockArg = ::llvm::dyn_cast<::mlir::BlockArgument>(v);
  if (!blockArg || blockArg.getOwner() != &graph.getBody().front())
    return false;
  return ::llvm::isa<::mlir::LLVM::LLVMPointerType>(blockArg.getType());
}

std::optional<::llvm::DataLayout>
getModuleLLVMDataLayout(::mlir::Operation *scope) {
  auto module = scope->getParentOfType<::mlir::ModuleOp>();
  if (!module)
    return std::nullopt;
  auto layoutAttr = module->getAttrOfType<::mlir::StringAttr>(
      ::mlir::LLVM::LLVMDialect::getDataLayoutAttrName());
  if (!layoutAttr)
    return std::nullopt;
  return ::llvm::DataLayout(layoutAttr.getValue());
}

// The distinguished leading `none` block argument is the graph start firing
// token; it is separate from the payload-only FunctionType.
::mlir::Value getThreadCtrl(::dataflow::GraphOp graph) {
  ::mlir::Block &entry = graph.getBody().front();
  if (entry.getNumArguments() == 0)
    return {};
  ::mlir::Value first = entry.getArgument(0);
  return ::llvm::isa<::mlir::NoneType>(first.getType()) ? first
                                                        : ::mlir::Value{};
}

struct BridgeKey {
  ::mlir::Value ptr;
  ::mlir::Type elem;
  bool operator==(const BridgeKey &o) const {
    return ptr == o.ptr && elem == o.elem;
  }
};

struct BridgeKeyInfo {
  static BridgeKey getEmptyKey() { return {{}, {}}; }
  static BridgeKey getTombstoneKey() {
    return {::mlir::Value::getFromOpaquePointer((void *)1),
            ::mlir::Type::getFromOpaquePointer((void *)1)};
  }
  static unsigned getHashValue(const BridgeKey &k) {
    return ::llvm::hash_combine(::mlir::hash_value(k.ptr),
                                ::mlir::hash_value(k.elem));
  }
  static bool isEqual(const BridgeKey &a, const BridgeKey &b) { return a == b; }
};

using ScopedValueCache =
    ::llvm::DenseMap<::mlir::Block *,
                     ::llvm::DenseMap<::mlir::Value, ::mlir::Value>>;
using ScopedBridgeCache =
    ::llvm::DenseMap<::mlir::Block *,
                     ::llvm::DenseMap<BridgeKey, ::mlir::Value, BridgeKeyInfo>>;

// Result of resolving one llvm.load / llvm.store address. Every GEP in a
// supported chain contributes either one typed linear term or constant bias;
// intermediate pointers never become independent memory capabilities.
struct AddrResolution {
  ::mlir::Value ptr;
  ::llvm::SmallVector<::loom::lowering::LinearByteTerm, 4> linearByteTerms;
  ::dataflow::StreamOp ordinalStream;
  std::int64_t ordinalElementBias = 0;
  unsigned byteToElementShift = 0;
  ::mlir::Type linearIndexType;
  std::int64_t linearByteBias = 0;
  ::llvm::SmallVector<::mlir::Operation *, 4> gepsToErase;
};

unsigned getElementByteWidth(::mlir::Type elemTy);

unsigned getElementByteWidth(::mlir::Type elemTy) {
  unsigned bitWidth = 0;
  if (auto intTy = ::llvm::dyn_cast<::mlir::IntegerType>(elemTy))
    bitWidth = intTy.getWidth();
  else if (auto floatTy = ::llvm::dyn_cast<::mlir::FloatType>(elemTy))
    bitWidth = floatTy.getWidth();
  if (bitWidth == 0 || bitWidth % 8 != 0)
    return 0;
  return bitWidth / 8;
}

bool isPowerOfTwo(unsigned value) {
  return value != 0 && (value & (value - 1)) == 0;
}

unsigned log2PowerOfTwo(unsigned value) {
  unsigned shift = 0;
  while (value > 1) {
    value >>= 1;
    ++shift;
  }
  return shift;
}

unsigned getByteToElementShift(::mlir::LLVM::GEPOp gep, ::mlir::Type elemTy) {
  auto gepElemTy = ::llvm::dyn_cast<::mlir::IntegerType>(gep.getElemType());
  if (!gepElemTy || gepElemTy.getWidth() != 8)
    return 0;
  unsigned byteWidth = getElementByteWidth(elemTy);
  if (byteWidth <= 1 || !isPowerOfTwo(byteWidth))
    return 0;
  return log2PowerOfTwo(byteWidth);
}

std::optional<std::int64_t> getSingleIndexElementStride(::mlir::LLVM::GEPOp gep,
                                                        ::mlir::Type elemTy) {
  if (!gep.getDynamicIndices().empty())
    return std::nullopt;
  auto rawIndices = gep.getRawConstantIndices();
  if (rawIndices.size() != 1)
    return std::nullopt;
  unsigned gepByteWidth = getElementByteWidth(gep.getElemType());
  unsigned elementByteWidth = getElementByteWidth(elemTy);
  if (gepByteWidth == 0 || elementByteWidth == 0)
    return std::nullopt;
  std::int64_t byteOffset =
      static_cast<std::int64_t>(rawIndices.front()) * gepByteWidth;
  if (byteOffset % static_cast<std::int64_t>(elementByteWidth) != 0)
    return std::nullopt;
  return byteOffset / static_cast<std::int64_t>(elementByteWidth);
}

::dataflow::StreamOp getUnitStridePointerCarryStream(::dataflow::CarryOp carry,
                                                     ::dataflow::GraphOp graph,
                                                     ::mlir::Type elemTy,
                                                     unsigned indexBits) {
  if (!::llvm::isa<::mlir::LLVM::LLVMPointerType>(carry.getOutput().getType()))
    return {};
  if (!isGraphPtrBlockArg(carry.getInit(), graph))
    return {};
  auto gep = carry.getCarry().getDefiningOp<::mlir::LLVM::GEPOp>();
  if (!gep)
    return {};
  if (getModuleLLVMDataLayout(gep))
    return {};
  ::mlir::Value gepBase = gep.getBase();
  if (gepBase != carry.getOutput()) {
    auto gate = gepBase.getDefiningOp<::dataflow::GateOp>();
    if (!gate || gate.getAfterValue() != gepBase ||
        gate.getBeforeValue() != carry.getOutput() ||
        gate.getBeforeCond() != carry.getCond())
      return {};
  }
  std::optional<std::int64_t> elementStride =
      getSingleIndexElementStride(gep, elemTy);
  if (!elementStride || *elementStride != 1)
    return {};
  auto stream = carry.getCond().getDefiningOp<::dataflow::StreamOp>();
  if (!stream || stream.getPhase() != carry.getCond() ||
      !::loom::lowering::isZeroBasedUnitOrdinalStream(stream, indexBits))
    return {};
  return stream;
}

std::optional<::dataflow::CarryOp> getGatedPointerCarry(::mlir::Value value) {
  auto gate = value.getDefiningOp<::dataflow::GateOp>();
  if (!gate || gate.getAfterValue() != value)
    return std::nullopt;
  auto carry = gate.getBeforeValue().getDefiningOp<::dataflow::CarryOp>();
  if (!carry || gate.getBeforeCond() != carry.getCond() ||
      !::llvm::isa<::mlir::LLVM::LLVMPointerType>(carry.getOutput().getType()))
    return std::nullopt;
  return carry;
}

std::optional<::dataflow::CarryOp> getPointerCarry(::mlir::Value value) {
  if (auto carry = value.getDefiningOp<::dataflow::CarryOp>()) {
    if (carry.getOutput() == value &&
        ::llvm::isa<::mlir::LLVM::LLVMPointerType>(carry.getOutput().getType()))
      return carry;
  }
  return getGatedPointerCarry(value);
}

std::optional<AddrResolution> resolvePointer(::mlir::Value loadStorePtr,
                                             ::dataflow::GraphOp graph,
                                             bool topLevel, ::mlir::Type elemTy,
                                             unsigned indexBits) {
  if (auto gep = loadStorePtr.getDefiningOp<::mlir::LLVM::GEPOp>()) {
    if (auto linear = ::loom::lowering::resolveLinearGepAddress(
            gep, graph, elemTy, indexBits)) {
      AddrResolution resolution;
      resolution.ptr = linear->root;
      resolution.linearByteTerms = std::move(linear->terms);
      resolution.byteToElementShift = linear->byteToElementShift;
      resolution.linearIndexType = linear->indexType;
      resolution.linearByteBias = linear->byteBias;
      resolution.gepsToErase = std::move(linear->gepsLeafToRoot);
      return resolution;
    }
  }
  if (topLevel) {
    if (auto gep = loadStorePtr.getDefiningOp<::mlir::LLVM::GEPOp>()) {
      ::mlir::Value base = gep.getBase();
      if (auto carry = getPointerCarry(base)) {
        if (::dataflow::StreamOp stream = getUnitStridePointerCarryStream(
                *carry, graph, elemTy, indexBits)) {
          std::optional<std::int64_t> bias =
              getSingleIndexElementStride(gep, elemTy);
          if (!bias)
            return std::nullopt;
          AddrResolution resolution;
          resolution.ptr = carry->getInit();
          resolution.ordinalStream = stream;
          resolution.ordinalElementBias = *bias;
          return resolution;
        }
      }
      return std::nullopt;
    }
    if (auto carry = loadStorePtr.getDefiningOp<::dataflow::CarryOp>()) {
      if (::dataflow::StreamOp stream = getUnitStridePointerCarryStream(
              carry, graph, elemTy, indexBits)) {
        AddrResolution resolution;
        resolution.ptr = carry.getInit();
        resolution.ordinalStream = stream;
        return resolution;
      }
      return std::nullopt;
    }
    if (auto carry = getGatedPointerCarry(loadStorePtr)) {
      if (::dataflow::StreamOp stream = getUnitStridePointerCarryStream(
              *carry, graph, elemTy, indexBits)) {
        AddrResolution resolution;
        resolution.ptr = carry->getInit();
        resolution.ordinalStream = stream;
        return resolution;
      }
      return std::nullopt;
    }
    if (isGraphPtrBlockArg(loadStorePtr, graph)) {
      AddrResolution resolution;
      resolution.ptr = loadStorePtr;
      return resolution;
    }
    return std::nullopt;
  }
  // Nested permissive fallback.
  if (!::llvm::isa<::mlir::LLVM::LLVMPointerType>(loadStorePtr.getType()))
    return std::nullopt;
  if (auto gep = loadStorePtr.getDefiningOp<::mlir::LLVM::GEPOp>()) {
    if (getModuleLLVMDataLayout(gep))
      return std::nullopt;
    auto dynIdxs = gep.getDynamicIndices();
    if (dynIdxs.size() == 1) {
      ::mlir::Value idx = dynIdxs.front();
      unsigned byteToElementShift = getByteToElementShift(gep, elemTy);
      if (byteToElementShift == 0 &&
          ::llvm::isa<::mlir::IntegerType, ::mlir::IndexType>(idx.getType())) {
        AddrResolution resolution;
        resolution.ptr = gep.getBase();
        resolution.linearByteTerms.push_back({idx, 1});
        resolution.linearIndexType = idx.getType();
        resolution.gepsToErase.push_back(gep.getOperation());
        return resolution;
      }
    }
  }
  AddrResolution resolution;
  resolution.ptr = loadStorePtr;
  return resolution;
}

// Materialize (or look up) an unrealized_conversion_cast bridging
// `ptr : !llvm.ptr` to `memref<?xElem>`. Graph-root pointers are hoisted and
// cached at graph entry; genuinely dynamic pointers are bridged at the access.
bool isSupportedBridgePointer(::mlir::Value ptr, ::dataflow::GraphOp graph) {
  auto ptrTy = ::llvm::dyn_cast<::mlir::LLVM::LLVMPointerType>(ptr.getType());
  if (!ptrTy)
    return false;
  unsigned addressSpace = ptrTy.getAddressSpace();
  std::optional<::llvm::DataLayout> llvmDataLayout =
      getModuleLLVMDataLayout(graph);
  if (llvmDataLayout && llvmDataLayout->isNonIntegralAddressSpace(addressSpace))
    return false;
  // The bridge does not preserve LLVM address spaces in the memref type.
  return addressSpace == 0;
}

::mlir::Value getMemrefBridge(
    ::mlir::OpBuilder &builder, ::dataflow::GraphOp graph,
    ::llvm::DenseMap<BridgeKey, ::mlir::Value, BridgeKeyInfo> &cache,
    ::mlir::Value ptr, ::mlir::Type elem, ::mlir::Location loc, bool topLevel,
    ::mlir::Operation *insertBeforeIfNested) {
  if (!isSupportedBridgePointer(ptr, graph))
    return {};
  if (topLevel) {
    BridgeKey key{ptr, elem};
    if (auto it = cache.find(key); it != cache.end())
      return it->second;
  }
  ::mlir::OpBuilder::InsertionGuard g(builder);
  if (topLevel)
    builder.setInsertionPointToStart(&graph.getBody().front());
  else
    builder.setInsertionPoint(insertBeforeIfNested);
  auto memrefTy = ::mlir::MemRefType::get({::mlir::ShapedType::kDynamic}, elem);
  ::mlir::Value bridge =
      ::mlir::UnrealizedConversionCastOp::create(
          builder, loc, ::mlir::TypeRange{memrefTy}, ::mlir::ValueRange{ptr})
          .getResult(0);
  if (topLevel)
    cache.try_emplace(BridgeKey{ptr, elem}, bridge);
  return bridge;
}

// `indexBits` is the canonical index width the pass boundary already
// resolved; this predicate never resolves it again.
bool canMaterializeIndex(::mlir::Type type, unsigned indexBits) {
  if (::llvm::isa<::mlir::IndexType>(type))
    return true;
  auto intTy = ::llvm::dyn_cast<::mlir::IntegerType>(type);
  if (!intTy || !intTy.isSignless())
    return false;
  return intTy.getWidth() <= indexBits;
}

::mlir::Value getIndexCast(::mlir::OpBuilder &builder,
                           ::dataflow::GraphOp graph, ScopedValueCache &cache,
                           unsigned indexBits, ::mlir::Value iv,
                           ::mlir::Location loc, bool topLevel,
                           ::mlir::Operation *insertBeforeIfNested) {
  if (::llvm::isa<::mlir::IndexType>(iv.getType()))
    return iv;
  if (!canMaterializeIndex(iv.getType(), indexBits))
    return {};
  auto &blockCache =
      cache.try_emplace(insertBeforeIfNested->getBlock()).first->second;
  if (auto it = blockCache.find(iv); it != blockCache.end())
    return it->second;
  bool hoist = topLevel && ::llvm::isa<::mlir::BlockArgument>(iv);
  ::mlir::OpBuilder::InsertionGuard g(builder);
  if (hoist)
    builder.setInsertionPointToStart(&graph.getBody().front());
  else
    builder.setInsertionPoint(insertBeforeIfNested);
  ::mlir::Value out = ::mlir::arith::IndexCastOp::create(
                          builder, loc, builder.getIndexType(), iv)
                          .getResult();
  blockCache.try_emplace(iv, out);
  return out;
}

::mlir::TypedAttr getIntegerLikeAttr(::mlir::OpBuilder &builder,
                                     ::mlir::Type type, std::int64_t value) {
  if (auto intTy = ::llvm::dyn_cast<::mlir::IntegerType>(type))
    return builder.getIntegerAttr(intTy, value);
  if (::llvm::isa<::mlir::IndexType>(type))
    return builder.getIndexAttr(value);
  return {};
}

::mlir::Value getLinearByteIndex(::mlir::OpBuilder &builder,
                                 ::llvm::ArrayRef<LinearByteTerm> terms,
                                 ::mlir::Type indexType, std::int64_t byteBias,
                                 ::mlir::Location loc,
                                 ::mlir::Operation *insertBefore) {
  if (!indexType)
    return {};
  ::mlir::OpBuilder::InsertionGuard g(builder);
  builder.setInsertionPoint(insertBefore);
  ::mlir::Value result;
  for (const LinearByteTerm &term : terms) {
    ::mlir::Value contribution = term.index;
    if (contribution.getType() != indexType) {
      auto destination = ::llvm::dyn_cast<::mlir::IntegerType>(indexType);
      if (!destination || !destination.isSignless())
        return {};
      if (::llvm::isa<::mlir::IndexType>(contribution.getType())) {
        contribution = ::mlir::arith::IndexCastOp::create(
                           builder, loc, destination, contribution)
                           .getResult();
      } else {
        auto source =
            ::llvm::dyn_cast<::mlir::IntegerType>(contribution.getType());
        if (!source || !source.isSignless() ||
            source.getWidth() >= destination.getWidth())
          return {};
        contribution = ::mlir::arith::ExtSIOp::create(builder, loc, destination,
                                                      contribution)
                           .getResult();
      }
    }
    if (term.byteStride != 1) {
      ::mlir::TypedAttr strideAttr =
          getIntegerLikeAttr(builder, indexType, term.byteStride);
      if (!strideAttr)
        return {};
      ::mlir::Value stride =
          ::mlir::arith::ConstantOp::create(builder, loc, indexType, strideAttr)
              .getResult();
      contribution =
          ::mlir::arith::MulIOp::create(builder, loc, contribution, stride)
              .getResult();
    }
    result = result ? ::mlir::arith::AddIOp::create(builder, loc, result,
                                                    contribution)
                          .getResult()
                    : contribution;
  }
  if (byteBias != 0) {
    ::mlir::TypedAttr biasAttr =
        getIntegerLikeAttr(builder, indexType, byteBias);
    if (!biasAttr)
      return {};
    ::mlir::Value bias =
        ::mlir::arith::ConstantOp::create(builder, loc, indexType, biasAttr)
            .getResult();
    result = result ? ::mlir::arith::AddIOp::create(builder, loc, result, bias)
                          .getResult()
                    : bias;
  }
  return result;
}

::mlir::Value projectStreamValue(::mlir::OpBuilder &builder,
                                 ::dataflow::StreamOp stream,
                                 ::mlir::Value value, ::mlir::Location loc) {
  return ::dataflow::GateOp::create(builder, loc, builder.getI1Type(),
                                    value.getType(), stream.getPhase(), value)
      .getAfterValue();
}

::mlir::Value getBiasedStreamOrdinal(::mlir::OpBuilder &builder,
                                     ::dataflow::StreamOp stream,
                                     ::mlir::Value ordinal, std::int64_t bias,
                                     ::mlir::Value ctrl, ::mlir::Location loc,
                                     ::mlir::Operation *insertBefore) {
  if (bias == 0)
    return ordinal;

  ::mlir::OpBuilder::InsertionGuard g(builder);
  builder.setInsertionPoint(insertBefore);
  ::mlir::TypedAttr biasAttr =
      getIntegerLikeAttr(builder, ordinal.getType(), bias);
  if (!biasAttr)
    return {};
  ::mlir::Value biasValue = ::dataflow::ConstantOp::create(
                                builder, loc, ordinal.getType(), ctrl, biasAttr)
                                .getValue();
  ::mlir::Value stableBiasRaw =
      ::dataflow::InvariantOp::create(builder, loc, ordinal.getType(),
                                      stream.getPhase(), biasValue)
          .getOutput();
  ::mlir::Value stableBias =
      projectStreamValue(builder, stream, stableBiasRaw, loc);
  return ::mlir::arith::AddIOp::create(builder, loc, ordinal, stableBias)
      .getResult();
}

::mlir::Value getElementIndex(::mlir::OpBuilder &builder,
                              ::dataflow::GraphOp graph,
                              ScopedValueCache &cache, unsigned indexBits,
                              ::mlir::Value intIndex,
                              unsigned byteToElementShift, ::mlir::Value ctrl,
                              ::mlir::Location loc, bool topLevel,
                              ::mlir::Operation *insertBeforeIfNested) {
  if (byteToElementShift == 0)
    return getIndexCast(builder, graph, cache, indexBits, intIndex, loc,
                        topLevel, insertBeforeIfNested);
  if (!canMaterializeIndex(intIndex.getType(), indexBits))
    return {};
  auto intTy = ::llvm::dyn_cast<::mlir::IntegerType>(intIndex.getType());
  if (!intTy)
    return {};

  ::mlir::OpBuilder::InsertionGuard g(builder);
  bool hoist = topLevel && ::llvm::isa<::mlir::BlockArgument>(intIndex);
  if (hoist)
    builder.setInsertionPointToStart(&graph.getBody().front());
  else
    builder.setInsertionPoint(insertBeforeIfNested);
  auto shiftAttr = builder.getIntegerAttr(intTy, byteToElementShift);
  ::mlir::Value shiftAmount =
      ::dataflow::ConstantOp::create(builder, loc, intTy, ctrl, shiftAttr)
          .getValue();
  ::mlir::Value elemIndex =
      ::mlir::arith::ShRSIOp::create(builder, loc, intIndex, shiftAmount)
          .getResult();
  return getIndexCast(builder, graph, cache, indexBits, elemIndex, loc,
                      topLevel, insertBeforeIfNested);
}

::mlir::Value getZeroIndex(::mlir::OpBuilder &builder,
                           ::dataflow::GraphOp graph, ::mlir::Value &cached,
                           ::mlir::Location loc, bool topLevel,
                           ::mlir::Operation *insertBeforeIfNested) {
  if (topLevel && cached)
    return cached;
  ::mlir::OpBuilder::InsertionGuard g(builder);
  if (topLevel)
    builder.setInsertionPointToStart(&graph.getBody().front());
  else
    builder.setInsertionPoint(insertBeforeIfNested);
  ::mlir::Value c0 =
      ::mlir::arith::ConstantOp::create(builder, loc, builder.getIndexType(),
                                        builder.getIndexAttr(0))
          .getResult();
  if (topLevel)
    cached = c0;
  return c0;
}

// Per-graph rewrite state bundle, threaded through tryRewriteOne to
// keep the call sites concise.
struct RewriteCtx {
  ::dataflow::GraphOp graph;
  ::mlir::Value ctrl;
  // Resolved once at the pass boundary and read-only from here on.
  unsigned indexBits = 0;
  ::llvm::DenseMap<BridgeKey, ::mlir::Value, BridgeKeyInfo> bridgeCache;
  ScopedValueCache indexCastCache;
  ScopedBridgeCache addressCache;
  ::mlir::Value zeroIdx;
  ::llvm::SmallVector<::mlir::Operation *, 8> deadGeps;
};

// Attempt to rewrite a single load or store. Returns true if a
// rewrite happened.
bool tryRewriteOne(::mlir::Operation *op, bool topLevel,
                   ::mlir::OpBuilder &builder, RewriteCtx &ctx) {
  ::mlir::Value ptrArg;
  ::mlir::Type elemTy;
  bool isLoad = ::llvm::isa<::mlir::LLVM::LoadOp>(op);
  if (isLoad) {
    auto load = ::llvm::cast<::mlir::LLVM::LoadOp>(op);
    if (load.getVolatile_() ||
        load.getOrdering() != ::mlir::LLVM::AtomicOrdering::not_atomic)
      return false;
    ptrArg = load.getAddr();
    elemTy = load.getResult().getType();
  } else {
    auto store = ::llvm::cast<::mlir::LLVM::StoreOp>(op);
    if (store.getVolatile_() ||
        store.getOrdering() != ::mlir::LLVM::AtomicOrdering::not_atomic)
      return false;
    ptrArg = store.getAddr();
    elemTy = store.getValue().getType();
  }
  // Pointer-element loads/stores trip the streaming verifier; skip.
  if (::llvm::isa<::mlir::LLVM::LLVMPointerType>(elemTy))
    return false;
  auto resolved =
      resolvePointer(ptrArg, ctx.graph, topLevel, elemTy, ctx.indexBits);
  if (!resolved)
    return false;
  if (!isSupportedBridgePointer(resolved->ptr, ctx.graph))
    return false;
  if (resolved->ordinalStream &&
      !canMaterializeIndex(resolved->ordinalStream.getIv().getType(),
                           ctx.indexBits))
    return false;
  for (const LinearByteTerm &term : resolved->linearByteTerms) {
    if (!canMaterializeIndex(term.index.getType(), ctx.indexBits))
      return false;
  }
  ::mlir::Location loc = op->getLoc();
  ::mlir::OpBuilder::InsertionGuard g(builder);
  builder.setInsertionPoint(op);
  ::mlir::Value addr;
  auto &addressCache =
      ctx.addressCache.try_emplace(op->getBlock()).first->second;
  BridgeKey addressKey{ptrArg, elemTy};
  if (auto it = addressCache.find(addressKey); it != addressCache.end()) {
    addr = it->second;
  } else {
    if (resolved->ordinalStream) {
      ::mlir::Value ordinal = resolved->ordinalStream.getIv();
      ordinal = getBiasedStreamOrdinal(builder, resolved->ordinalStream,
                                       ordinal, resolved->ordinalElementBias,
                                       ctx.ctrl, loc, op);
      if (!ordinal)
        return false;
      addr = getElementIndex(
          builder, ctx.graph, ctx.indexCastCache, ctx.indexBits, ordinal,
          resolved->byteToElementShift, ctx.ctrl, loc, topLevel, op);
    } else if (!resolved->linearByteTerms.empty() ||
               resolved->linearByteBias != 0) {
      ::mlir::Value intIndex = getLinearByteIndex(
          builder, resolved->linearByteTerms, resolved->linearIndexType,
          resolved->linearByteBias, loc, op);
      if (!intIndex)
        return false;
      addr = getElementIndex(
          builder, ctx.graph, ctx.indexCastCache, ctx.indexBits, intIndex,
          resolved->byteToElementShift, ctx.ctrl, loc, topLevel, op);
    } else {
      addr = getZeroIndex(builder, ctx.graph, ctx.zeroIdx, loc, topLevel, op);
    }
    if (!addr)
      return false;
    addressCache.try_emplace(addressKey, addr);
  }
  bool staticBinding = topLevel || isGraphPtrBlockArg(resolved->ptr, ctx.graph);
  ::mlir::Value mem =
      getMemrefBridge(builder, ctx.graph, ctx.bridgeCache, resolved->ptr,
                      elemTy, loc, staticBinding, op);
  if (!mem)
    return false;
  if (isLoad) {
    auto load = ::llvm::cast<::mlir::LLVM::LoadOp>(op);
    auto newLoad = ::dataflow::LoadOp::create(
        builder, loc, /*data=*/elemTy, /*done=*/builder.getNoneType(),
        /*mem=*/mem, /*addr=*/addr, /*ctrl=*/ctx.ctrl);
    load.getResult().replaceAllUsesWith(newLoad.getData());
  } else {
    auto store = ::llvm::cast<::mlir::LLVM::StoreOp>(op);
    ::dataflow::StoreOp::create(
        builder, loc, /*done=*/builder.getNoneType(), /*mem=*/mem,
        /*addr=*/addr, /*data=*/store.getValue(), /*ctrl=*/ctx.ctrl);
  }
  op->erase();
  ctx.deadGeps.append(resolved->gepsToErase);
  return true;
}

// The canonical index width of one graph, read at the exact scope that owns
// it. The pass boundary validates every graph before anything mutates, so a
// graph that cannot resolve one is simply left alone here.
std::optional<unsigned> getGraphIndexBits(::dataflow::GraphOp graph) {
  ::llvm::Expected<unsigned> bits = ::loom::getIndexBitWidth(graph);
  if (bits)
    return *bits;
  ::llvm::consumeError(bits.takeError());
  return std::nullopt;
}

unsigned sinkBranchSelectedLoads(::dataflow::GraphOp graph,
                                 ::mlir::OpBuilder &builder) {
  ::llvm::SmallVector<::mlir::LLVM::LoadOp, 4> loads;
  graph.getBody().walk([&](::mlir::LLVM::LoadOp load) {
    if (!load.getVolatile_() &&
        load.getOrdering() == ::mlir::LLVM::AtomicOrdering::not_atomic)
      loads.push_back(load);
  });

  unsigned rewritten = 0;
  for (::mlir::LLVM::LoadOp load : loads) {
    auto selected = ::llvm::dyn_cast<::mlir::OpResult>(load.getAddr());
    if (!selected || !selected.hasOneUse())
      continue;
    if (!::llvm::isa<::mlir::LLVM::LLVMPointerType>(selected.getType()))
      continue;

    if (auto select =
            ::llvm::dyn_cast<::mlir::arith::SelectOp>(selected.getOwner())) {
      ::mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPoint(load);
      auto cloneLoadAndYield = [&](::mlir::OpBuilder &bodyBuilder,
                                   ::mlir::Location loc,
                                   ::mlir::Value address) {
        ::mlir::IRMapping mapping;
        mapping.map(load.getAddr(), address);
        auto cloned = ::llvm::cast<::mlir::LLVM::LoadOp>(
            bodyBuilder.clone(*load.getOperation(), mapping));
        ::mlir::scf::YieldOp::create(bodyBuilder, loc, cloned.getResult());
      };
      auto thenBuilder = [&](::mlir::OpBuilder &bodyBuilder,
                             ::mlir::Location loc) {
        cloneLoadAndYield(bodyBuilder, loc, select.getTrueValue());
      };
      auto elseBuilder = [&](::mlir::OpBuilder &bodyBuilder,
                             ::mlir::Location loc) {
        cloneLoadAndYield(bodyBuilder, loc, select.getFalseValue());
      };
      auto branch = ::mlir::scf::IfOp::create(builder, load.getLoc(),
                                              select.getCondition(),
                                              thenBuilder, elseBuilder);
      load.getResult().replaceAllUsesWith(branch.getResult(0));
      load.erase();
      if (select->use_empty())
        select.erase();
      ++rewritten;
      continue;
    }

    auto branch = ::llvm::dyn_cast<::mlir::scf::IfOp>(selected.getOwner());
    if (!branch || branch.getElseRegion().empty())
      continue;

    unsigned resultIndex = selected.getResultNumber();
    ::mlir::scf::YieldOp thenYield = ::llvm::dyn_cast<::mlir::scf::YieldOp>(
        branch.getThenRegion().front().getTerminator());
    ::mlir::scf::YieldOp elseYield = ::llvm::dyn_cast<::mlir::scf::YieldOp>(
        branch.getElseRegion().front().getTerminator());
    if (!thenYield || !elseYield || resultIndex >= thenYield.getNumOperands() ||
        resultIndex >= elseYield.getNumOperands())
      continue;
    for (::mlir::scf::YieldOp yield : {thenYield, elseYield}) {
      ::mlir::IRMapping mapping;
      mapping.map(load.getAddr(), yield.getOperand(resultIndex));
      ::mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPoint(yield);
      auto cloned = ::llvm::cast<::mlir::LLVM::LoadOp>(
          builder.clone(*load.getOperation(), mapping));
      yield.setOperand(resultIndex, cloned.getResult());
    }
    selected.setType(load.getResult().getType());
    load.getResult().replaceAllUsesWith(selected);
    load.erase();
    ++rewritten;
  }
  return rewritten;
}

::mlir::LogicalResult
checkGraphRegionLoweringPreconditionsAfterLoadSinking(::mlir::ModuleOp module) {
  ::mlir::OwningOpRef<::mlir::ModuleOp> scratch(
      ::mlir::cast<::mlir::ModuleOp>(module->clone()));
  ::mlir::OpBuilder builder(module.getContext());
  scratch->walk([&](::dataflow::GraphOp graph) {
    if (!graph.isExternal())
      (void)sinkBranchSelectedLoads(graph, builder);
  });
  return ::loom::lowering::checkGraphRegionLoweringPreconditions(*scratch);
}

unsigned rewriteOneGraph(::dataflow::GraphOp graph,
                         ::mlir::OpBuilder &builder) {
  ::mlir::Value ctrl = getThreadCtrl(graph);
  if (!ctrl)
    return 0;
  std::optional<unsigned> indexBits = getGraphIndexBits(graph);
  if (!indexBits)
    return 0;

  RewriteCtx ctx;
  ctx.graph = graph;
  ctx.ctrl = ctrl;
  ctx.indexBits = *indexBits;

  unsigned rewrites = sinkBranchSelectedLoads(graph, builder);

  // Collect rewrite targets up front so the walk is independent of
  // mutations performed by tryRewriteOne.
  struct Target {
    ::mlir::Operation *op;
    bool topLevel;
  };
  ::llvm::SmallVector<Target, 16> targets;
  ::mlir::Block &entry = graph.getBody().front();
  graph.getBody().walk([&](::mlir::Operation *op) {
    if (::llvm::isa<::mlir::LLVM::LoadOp, ::mlir::LLVM::StoreOp>(op))
      targets.push_back({op, op->getBlock() == &entry});
    return ::mlir::WalkResult::advance();
  });

  for (auto &t : targets) {
    if (tryRewriteOne(t.op, t.topLevel, builder, ctx))
      ++rewrites;
  }
  // Erase orphan geps (those whose only uses were the rewritten
  // load/store ops). Some may have other live uses -- skip those
  // silently.
  ::llvm::SmallPtrSet<::mlir::Operation *, 8> visitedDeadGeps;
  for (::mlir::Operation *gep : ctx.deadGeps) {
    if (visitedDeadGeps.contains(gep))
      continue;
    if (gep->use_empty()) {
      visitedDeadGeps.insert(gep);
      gep->erase();
    }
  }
  return rewrites;
}

::mlir::LogicalResult checkResidualMemoryEffects(::dataflow::GraphOp graph) {
  ::mlir::WalkResult result =
      graph.getBody().walk(
          [](::mlir::Operation *op) -> ::mlir::WalkResult {
            bool lacksCompletion =
                ::llvm::isa<::mlir::LLVM::LoadOp, ::mlir::LLVM::StoreOp,
                            ::mlir::LLVM::MemcpyOp, ::mlir::LLVM::MemmoveOp,
                            ::mlir::LLVM::MemsetOp>(op);
            if (!lacksCompletion)
              return ::mlir::WalkResult::advance();

            op->emitError()
                << "loom-lower-graph-memory: residual memory operation '"
                << op->getName().getStringRef()
                << "' has no explicit completion event";
            return ::mlir::WalkResult::interrupt();
          });
  return result.wasInterrupted() ? ::mlir::failure() : ::mlir::success();
}

// One unusable index-width declaration fails the pass with its owner's reason,
// before any graph is rewritten.
::mlir::LogicalResult checkGraphIndexWidths(::mlir::ModuleOp module) {
  ::mlir::WalkResult result = module.walk([&](::dataflow::GraphOp graph) {
    if (graph.isExternal())
      return ::mlir::WalkResult::advance();
    ::llvm::Expected<unsigned> bits = ::loom::getIndexBitWidth(graph);
    if (bits)
      return ::mlir::WalkResult::advance();
    module.emitError("loom-lower-graph-memory: ")
        << ::llvm::toString(bits.takeError());
    return ::mlir::WalkResult::interrupt();
  });
  return result.wasInterrupted() ? ::mlir::failure() : ::mlir::success();
}

::mlir::LogicalResult checkNormalizedMemoryEffects(::mlir::ModuleOp module) {
  ::mlir::OwningOpRef<::mlir::ModuleOp> scratch(
      ::mlir::cast<::mlir::ModuleOp>(module->clone()));
  ::mlir::OpBuilder builder(module.getContext());
  ::llvm::SmallVector<::dataflow::GraphOp, 8> graphs;
  scratch->walk([&](::dataflow::GraphOp graph) { graphs.push_back(graph); });
  for (auto graph : graphs) {
    if (!graph.isExternal())
      (void)rewriteOneGraph(graph, builder);
  }
  for (auto graph : graphs) {
    if (!graph.isExternal() &&
        ::mlir::failed(checkResidualMemoryEffects(graph)))
      return ::mlir::failure();
  }
  return ::mlir::success();
}

struct LowerGraphMemoryPass
    : public ::mlir::PassWrapper<LowerGraphMemoryPass,
                                 ::mlir::OperationPass<::mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerGraphMemoryPass)

  ::llvm::StringRef getArgument() const final {
    return "loom-lower-graph-memory";
  }
  ::llvm::StringRef getDescription() const final {
    return "Lower graph-local memory accesses and recursively flatten "
           "structured graph regions with per-alias-partition memory "
           "frontiers.";
  }

  void getDependentDialects(::mlir::DialectRegistry &registry) const final {
    registry.insert<::mlir::arith::ArithDialect, ::mlir::func::FuncDialect,
                    ::mlir::LLVM::LLVMDialect, ::mlir::memref::MemRefDialect,
                    ::mlir::scf::SCFDialect, ::dataflow::DataflowDialect>();
  }

  void runOnOperation() final {
    ::mlir::ModuleOp module = getOperation();
    ::mlir::OpBuilder builder(&getContext());

    // Every graph's canonical index width is resolved at its own scope,
    // before anything is normalized or mutated, so an unusable declaration is
    // reported with its owner's reason instead of degrading into a later
    // residual memory error.
    if (::mlir::failed(checkGraphIndexWidths(module))) {
      signalPassFailure();
      return;
    }

    if (::mlir::failed(
            checkGraphRegionLoweringPreconditionsAfterLoadSinking(module))) {
      signalPassFailure();
      return;
    }
    if (::mlir::failed(checkNormalizedMemoryEffects(module))) {
      signalPassFailure();
      return;
    }

    ::llvm::SmallVector<::dataflow::GraphOp, 8> graphs;
    module.walk([&](::dataflow::GraphOp graph) { graphs.push_back(graph); });

    for (::dataflow::GraphOp graph : graphs) {
      if (graph.isExternal())
        continue;
      (void)rewriteOneGraph(graph, builder);
    }

    for (::dataflow::GraphOp graph : graphs) {
      if (graph.isExternal())
        continue;
      std::optional<unsigned> indexBits = getGraphIndexBits(graph);
      if (!indexBits)
        continue;
      if (::mlir::failed(
              ::loom::lowering::lowerGraphRegions(graph, *indexBits))) {
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace

namespace loom {
namespace lowering {

std::unique_ptr<::mlir::Pass> createLowerGraphMemoryPass() {
  return std::make_unique<LowerGraphMemoryPass>();
}

void registerLowerGraphMemoryPass() {
  static bool once = []() {
    ::mlir::PassRegistration<LowerGraphMemoryPass>();
    return true;
  }();
  (void)once;
}

} // namespace lowering
} // namespace loom
