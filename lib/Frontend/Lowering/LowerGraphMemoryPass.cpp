// Tokenize residual `llvm.load` / `llvm.store` ops inside
// `dataflow.graph.func` bodies into the streaming-memory primitives
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
// Loads / stores whose pointer is none of the above (e.g. derived
// from `llvm.alloca` or `llvm.mlir.addressof` at the top level) are
// left in place. Pointer-element loads / stores (loading/storing a
// !llvm.ptr value) are also skipped: bridging !llvm.ptr-of-ptr to
// memref<?x!llvm.ptr> trips the streaming load verifier.

#include "Frontend/Lowering/Passes.h"

#include "Dataflow/IR/DataflowDialect.h"
#include "Dataflow/IR/DataflowOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

namespace {

bool isGraphPtrBlockArg(::mlir::Value v, ::dataflow::GraphFuncOp graph) {
  auto blockArg = ::llvm::dyn_cast<::mlir::BlockArgument>(v);
  if (!blockArg || blockArg.getOwner() != &graph.getBody().front())
    return false;
  return ::llvm::isa<::mlir::LLVM::LLVMPointerType>(blockArg.getType());
}

// The leading `none`-typed signature input is the `thread_ctrl`
// firing token; we rely on the lowering pipeline's invariant that
// graph.func signatures begin with `(none, ...)`.
::mlir::Value getThreadCtrl(::dataflow::GraphFuncOp graph) {
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

// Result of resolving the (memref-base, optional-int-index) pair for
// one llvm.load / llvm.store. `intIndex` is null when the address
// port should be `0 : index`.
struct AddrResolution {
  ::mlir::Value ptr;
  ::mlir::Value intIndex;
  ::mlir::Operation *gepToErase = nullptr;
};

std::optional<AddrResolution>
resolvePointer(::mlir::Value loadStorePtr, ::dataflow::GraphFuncOp graph,
               bool topLevel) {
  if (topLevel) {
    if (auto gep = loadStorePtr.getDefiningOp<::mlir::LLVM::GEPOp>()) {
      ::mlir::Value base = gep.getBase();
      auto dynIdxs = gep.getDynamicIndices();
      if (!isGraphPtrBlockArg(base, graph) || dynIdxs.size() != 1)
        return std::nullopt;
      ::mlir::Value idx = dynIdxs.front();
      if (!::llvm::isa<::mlir::IntegerType, ::mlir::IndexType>(idx.getType()))
        return std::nullopt;
      return AddrResolution{base, idx, gep.getOperation()};
    }
    if (auto carry = loadStorePtr.getDefiningOp<::dataflow::CarryOp>()) {
      if (!isGraphPtrBlockArg(carry.getInit(), graph))
        return std::nullopt;
      return AddrResolution{carry.getOutput(), {}, nullptr};
    }
    if (isGraphPtrBlockArg(loadStorePtr, graph))
      return AddrResolution{loadStorePtr, {}, nullptr};
    return std::nullopt;
  }
  // Nested permissive fallback.
  if (!::llvm::isa<::mlir::LLVM::LLVMPointerType>(loadStorePtr.getType()))
    return std::nullopt;
  if (auto gep = loadStorePtr.getDefiningOp<::mlir::LLVM::GEPOp>()) {
    auto dynIdxs = gep.getDynamicIndices();
    if (dynIdxs.size() == 1) {
      ::mlir::Value idx = dynIdxs.front();
      if (::llvm::isa<::mlir::IntegerType, ::mlir::IndexType>(idx.getType()))
        return AddrResolution{gep.getBase(), idx, gep.getOperation()};
    }
  }
  return AddrResolution{loadStorePtr, {}, nullptr};
}

// Materialize (or look up) an unrealized_conversion_cast bridging
// `ptr : !llvm.ptr` to `memref<?xElem>`. Top-level rewrites hoist
// and cache the cast at the start of the graph body; nested rewrites
// emit the cast at the load/store site to respect SSA dominance.
::mlir::Value
getMemrefBridge(::mlir::OpBuilder &builder, ::dataflow::GraphFuncOp graph,
                ::llvm::DenseMap<BridgeKey, ::mlir::Value, BridgeKeyInfo>
                    &cache,
                ::mlir::Value ptr, ::mlir::Type elem, ::mlir::Location loc,
                bool topLevel, ::mlir::Operation *insertBeforeIfNested) {
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

::mlir::Value
getIndexCast(::mlir::OpBuilder &builder, ::dataflow::GraphFuncOp graph,
             ::llvm::DenseMap<::mlir::Value, ::mlir::Value> &cache,
             ::mlir::Value iv, ::mlir::Location loc, bool topLevel,
             ::mlir::Operation *insertBeforeIfNested) {
  if (::llvm::isa<::mlir::IndexType>(iv.getType()))
    return iv;
  if (topLevel) {
    if (auto it = cache.find(iv); it != cache.end())
      return it->second;
  }
  ::mlir::OpBuilder::InsertionGuard g(builder);
  if (topLevel)
    builder.setInsertionPointToStart(&graph.getBody().front());
  else
    builder.setInsertionPoint(insertBeforeIfNested);
  ::mlir::Value out = ::mlir::arith::IndexCastOp::create(
                          builder, loc, builder.getIndexType(), iv)
                          .getResult();
  if (topLevel)
    cache.try_emplace(iv, out);
  return out;
}

::mlir::Value getZeroIndex(::mlir::OpBuilder &builder,
                           ::dataflow::GraphFuncOp graph,
                           ::mlir::Value &cached, ::mlir::Location loc,
                           bool topLevel,
                           ::mlir::Operation *insertBeforeIfNested) {
  if (topLevel && cached)
    return cached;
  ::mlir::OpBuilder::InsertionGuard g(builder);
  if (topLevel)
    builder.setInsertionPointToStart(&graph.getBody().front());
  else
    builder.setInsertionPoint(insertBeforeIfNested);
  ::mlir::Value c0 = ::mlir::arith::ConstantOp::create(
                         builder, loc, builder.getIndexType(),
                         builder.getIndexAttr(0))
                         .getResult();
  if (topLevel)
    cached = c0;
  return c0;
}

// Per-graph rewrite state bundle, threaded through tryRewriteOne to
// keep the call sites concise.
struct RewriteCtx {
  ::dataflow::GraphFuncOp graph;
  ::mlir::Value ctrl;
  ::llvm::DenseMap<BridgeKey, ::mlir::Value, BridgeKeyInfo> bridgeCache;
  ::llvm::DenseMap<::mlir::Value, ::mlir::Value> indexCastCache;
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
    ptrArg = load.getAddr();
    elemTy = load.getResult().getType();
  } else {
    auto store = ::llvm::cast<::mlir::LLVM::StoreOp>(op);
    ptrArg = store.getAddr();
    elemTy = store.getValue().getType();
  }
  // Pointer-element loads/stores trip the streaming verifier; skip.
  if (::llvm::isa<::mlir::LLVM::LLVMPointerType>(elemTy))
    return false;
  auto resolved = resolvePointer(ptrArg, ctx.graph, topLevel);
  if (!resolved)
    return false;
  ::mlir::Location loc = op->getLoc();
  ::mlir::OpBuilder::InsertionGuard g(builder);
  builder.setInsertionPoint(op);
  ::mlir::Value mem =
      getMemrefBridge(builder, ctx.graph, ctx.bridgeCache, resolved->ptr,
                      elemTy, loc, topLevel, op);
  ::mlir::Value addr =
      resolved->intIndex
          ? getIndexCast(builder, ctx.graph, ctx.indexCastCache,
                         resolved->intIndex, loc, topLevel, op)
          : getZeroIndex(builder, ctx.graph, ctx.zeroIdx, loc, topLevel, op);
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
  if (resolved->gepToErase)
    ctx.deadGeps.push_back(resolved->gepToErase);
  return true;
}

unsigned rewriteOneGraph(::dataflow::GraphFuncOp graph,
                         ::mlir::OpBuilder &builder) {
  ::mlir::Value ctrl = getThreadCtrl(graph);
  if (!ctrl)
    return 0;

  RewriteCtx ctx;
  ctx.graph = graph;
  ctx.ctrl = ctrl;

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

  unsigned rewrites = 0;
  for (auto &t : targets) {
    if (tryRewriteOne(t.op, t.topLevel, builder, ctx))
      ++rewrites;
  }

  // Erase orphan geps (those whose only uses were the rewritten
  // load/store ops). Some may have other live uses -- skip those
  // silently.
  for (::mlir::Operation *gep : ctx.deadGeps)
    if (gep->use_empty())
      gep->erase();
  return rewrites;
}

struct LowerGraphMemoryPass
    : public ::mlir::PassWrapper<LowerGraphMemoryPass,
                                 ::mlir::OperationPass<::mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerGraphMemoryPass)

  ::llvm::StringRef getArgument() const final {
    return "loom-lower-graph-memory";
  }
  ::llvm::StringRef getDescription() const final {
    return "Tokenize residual llvm.load / llvm.store ops inside "
           "dataflow.graph.func bodies into dataflow.load / dataflow.store "
           "with an unrealized_conversion_cast pointer-to-memref bridge.";
  }

  void getDependentDialects(::mlir::DialectRegistry &registry) const final {
    registry.insert<::mlir::arith::ArithDialect, ::mlir::func::FuncDialect,
                    ::mlir::LLVM::LLVMDialect, ::mlir::memref::MemRefDialect,
                    ::dataflow::DataflowDialect>();
  }

  void runOnOperation() final {
    ::mlir::ModuleOp module = getOperation();
    ::mlir::OpBuilder builder(&getContext());

    ::llvm::SmallVector<::dataflow::GraphFuncOp, 8> graphs;
    for (auto graph : module.getOps<::dataflow::GraphFuncOp>())
      graphs.push_back(graph);

    for (::dataflow::GraphFuncOp graph : graphs) {
      if (graph.isExternal())
        continue;
      (void)rewriteOneGraph(graph, builder);
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
