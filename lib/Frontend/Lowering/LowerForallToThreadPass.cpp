// Lower top-level scf.forall ops inside func.func bodies into
// `dataflow.thread` symbol definitions plus matching
// `dataflow.thread.launch` ops. The forall's body is moved (not
// cloned) into the thread definition; captured outside-defined SSA
// values become explicit body operands of the launch and thread
// entry-block arguments.
//
// Smoke deliverable: only scf.forall ops that are direct children of
// a func.func body are promoted. Nested foralls remain in their
// enclosing control/data region so later graph extraction cannot
// clone dataflow launchers into a graph body. Aggregation-form
// foralls (with shared_outs / op results) are left in place; the
// raise pipeline already lowers those through Part 2 normalization.
// Dynamic upper bounds are forwarded via the launch's gridUpperBounds
// operand list; static bounds are still expressed dynamically as a
// constant index for simplicity (the spec accepts kDynamic
// sentinels).
//
// Per the spec, the thread body's entry block has the layout
// `(args_*, thread_ctrl, iv_*)`: the first N entry block args mirror
// `function_type.inputs`, then a `none`-typed thread_ctrl slot, then
// one `index`-typed grid iv slot per forall induction variable. The
// thread_ctrl slot is the per-launch AccCore start signal that root
// `dataflow.graph.launch` ops in the body consume as their `ctrl_in`.

#include "Frontend/Lowering/Passes.h"

#include "Dataflow/IR/DataflowDialect.h"
#include "Dataflow/IR/DataflowOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

namespace {

// Sanitize a string into a legal MLIR symbol component (alphanumeric
// + underscore). Anything else collapses into an underscore.
std::string sanitizeSymbol(::llvm::StringRef in) {
  std::string out;
  out.reserve(in.size());
  for (char c : in) {
    if (::llvm::isAlnum(c) || c == '_')
      out.push_back(c);
    else
      out.push_back('_');
  }
  if (out.empty())
    out = "_";
  return out;
}

// Derive a unique symbol name in the module's top-level symbol table
// based on the candidate stem.
std::string uniqueSymbol(::mlir::ModuleOp module, ::llvm::StringRef stem) {
  ::mlir::SymbolTable st(module);
  std::string base = stem.str();
  if (!st.lookup(base))
    return base;
  unsigned suffix = 1;
  while (true) {
    std::string candidate = base + "_" + std::to_string(suffix);
    if (!st.lookup(candidate))
      return candidate;
    ++suffix;
  }
}

// Materialize an Index-typed SSA value from a static-int lower bound
// or upper bound or step. Used for the launch's gridUpperBounds.
::mlir::Value makeIndexValue(::mlir::OpBuilder &builder, ::mlir::Location loc,
                             ::mlir::OpFoldResult ofr) {
  if (auto attr = ::llvm::dyn_cast<::mlir::Attribute>(ofr)) {
    auto intAttr = ::llvm::dyn_cast<::mlir::IntegerAttr>(attr);
    int64_t v = intAttr.getInt();
    return ::mlir::arith::ConstantOp::create(
        builder, loc, builder.getIndexAttr(v));
  }
  return ::llvm::cast<::mlir::Value>(ofr);
}

// Collect every value used inside the forall region that is defined above it.
void collectCapturedValues(::mlir::scf::ForallOp forall,
                           ::llvm::SetVector<::mlir::Value> &captures) {
  ::mlir::getUsedValuesDefinedAbove(forall.getRegion(), captures);
}

struct LowerForallToThreadPass
    : public ::mlir::PassWrapper<LowerForallToThreadPass,
                                 ::mlir::OperationPass<::mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerForallToThreadPass)

  ::llvm::StringRef getArgument() const final {
    return "loom-lower-forall-to-thread";
  }
  ::llvm::StringRef getDescription() const final {
    return "Lower top-level scf.forall ops inside func.func bodies into "
           "dataflow.thread definitions plus dataflow.thread.launch ops.";
  }

  void getDependentDialects(::mlir::DialectRegistry &registry) const final {
    registry.insert<::mlir::arith::ArithDialect, ::mlir::func::FuncDialect,
                    ::mlir::scf::SCFDialect, ::dataflow::DataflowDialect>();
  }

  void runOnOperation() final {
    ::mlir::ModuleOp module = getOperation();
    ::mlir::MLIRContext *ctx = &getContext();
    ::mlir::OpBuilder builder(ctx);

    // Snapshot foralls per func.func before mutating, so the
    // source-order seq index is stable.
    struct Pending {
      ::mlir::func::FuncOp func;
      ::llvm::SmallVector<::mlir::scf::ForallOp, 4> foralls;
    };
    ::llvm::SmallVector<Pending, 4> pending;
    module.walk([&](::mlir::func::FuncOp func) {
      Pending p;
      p.func = func;
      // Source-order top-level foralls: walk the func body and only
      // record foralls whose immediate parent op is the func itself.
      func.walk([&](::mlir::scf::ForallOp forall) {
        if (forall->getParentOp() != func.getOperation())
          return;
        p.foralls.push_back(forall);
      });
      if (!p.foralls.empty())
        pending.push_back(std::move(p));
    });

    for (Pending &p : pending) {
      ::llvm::StringRef funcSym = p.func.getSymName();
      std::string stem = "t_" + sanitizeSymbol(funcSym);
      for (auto [seq, forall] : ::llvm::enumerate(p.foralls)) {
        if (failed(promoteOne(module, forall, stem, seq, builder)))
          return signalPassFailure();
      }
    }
  }

  ::mlir::LogicalResult promoteOne(::mlir::ModuleOp module,
                                   ::mlir::scf::ForallOp forall,
                                   ::llvm::StringRef stem, size_t seq,
                                   ::mlir::OpBuilder &builder) {
    ::mlir::Location loc = forall.getLoc();

    // Effect-form requirement: no shared_outs / op results / non-empty
    // in_parallel terminator. The raise pipeline today emits effect-
    // form foralls for our 5 kernels; we leave anything else in
    // place as a TODO.
    if (!forall.getOutputs().empty() || forall.getNumResults() != 0)
      return ::mlir::success();
    auto inParallel = forall.getTerminator();
    if (!inParallel.getRegion().front().empty())
      return ::mlir::success();

    // Collect captured outside-defined SSA values.
    ::llvm::SetVector<::mlir::Value> captures;
    collectCapturedValues(forall, captures);

    // Build the function type: inputs are the captured value types.
    ::llvm::SmallVector<::mlir::Type, 8> inputTypes;
    for (::mlir::Value v : captures)
      inputTypes.push_back(v.getType());
    ::mlir::FunctionType functionType =
        builder.getFunctionType(inputTypes, /*results=*/{});

    // Build a unique symbol like t_<func>_<seq>.
    std::string symStem =
        (stem + "_" + ::llvm::Twine(seq)).str();
    std::string symName = uniqueSymbol(module, symStem);

    // Insert the dataflow.thread definition at module scope.
    builder.setInsertionPointToEnd(module.getBody());
    auto threadOp = ::dataflow::ThreadOp::create(
        builder, loc, symName, functionType,
        ::llvm::ArrayRef<::mlir::NamedAttribute>{});
    threadOp.setSymVisibilityAttr(builder.getStringAttr("private"));

    // Build the thread body. Entry block layout: captured args first,
    // then a `none`-typed `thread_ctrl` slot (per spec section 5.4.1),
    // then one `index`-typed iv argument per forall induction
    // variable.
    ::mlir::Region &threadBody = threadOp.getBody();
    ::mlir::Block *entry = builder.createBlock(&threadBody);
    for (::mlir::Type ty : inputTypes)
      entry->addArgument(ty, loc);
    ::mlir::BlockArgument threadCtrlArg =
        entry->addArgument(builder.getType<::mlir::NoneType>(), loc);
    (void)threadCtrlArg;
    ::llvm::SmallVector<::mlir::Value, 4> forallIvs;
    for (::mlir::Value iv : forall.getInductionVars())
      forallIvs.push_back(iv);
    ::llvm::SmallVector<::mlir::BlockArgument, 4> ivBlockArgs;
    for (size_t i = 0, e = forallIvs.size(); i < e; ++i) {
      ::mlir::BlockArgument ivArg =
          entry->addArgument(builder.getIndexType(), loc);
      ivBlockArgs.push_back(ivArg);
    }

    // Move the forall body's ops into the thread body's entry block.
    ::mlir::Block &forallBody = forall.getRegion().front();
    ::mlir::IRMapping mapping;
    for (auto [i, captured] : ::llvm::enumerate(captures))
      mapping.map(captured, entry->getArgument(i));
    for (auto [i, iv] : ::llvm::enumerate(forallIvs))
      mapping.map(iv, ivBlockArgs[i]);

    builder.setInsertionPointToEnd(entry);
    for (::mlir::Operation &op : forallBody.without_terminator())
      builder.clone(op, mapping);
    // The original in_parallel terminator becomes a dataflow.thread.yield.
    ::dataflow::ThreadYieldOp::create(builder, loc);

    // Materialize the launch at the original forall site.
    builder.setInsertionPoint(forall);
    ::llvm::SmallVector<::mlir::Value, 4> upperBounds;
    for (::mlir::OpFoldResult ofr : forall.getMixedUpperBound())
      upperBounds.push_back(makeIndexValue(builder, loc, ofr));
    ::llvm::SmallVector<::mlir::Value, 8> bodyOperands;
    bodyOperands.reserve(captures.size());
    for (::mlir::Value v : captures)
      bodyOperands.push_back(v);

    auto callee = ::mlir::FlatSymbolRefAttr::get(builder.getContext(), symName);
    ::dataflow::ThreadLaunchOp::create(
        builder, loc, /*asyncToken=*/::mlir::Type{}, callee, bodyOperands,
        upperBounds, /*asyncDependencies=*/::mlir::ValueRange{});

    forall.erase();
    return ::mlir::success();
  }
};

} // namespace

namespace loom {
namespace lowering {

std::unique_ptr<::mlir::Pass> createLowerForallToThreadPass() {
  return std::make_unique<LowerForallToThreadPass>();
}

void registerLowerForallToThreadPass() {
  static bool once = []() {
    ::mlir::PassRegistration<LowerForallToThreadPass>();
    return true;
  }();
  (void)once;
}

} // namespace lowering
} // namespace loom
