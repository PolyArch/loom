// Lower effect-form scf.forall ops inside func.func bodies into
// `dataflow.thread` symbol definitions plus matching
// `dataflow.thread.launch` ops. The forall's body is moved (not
// cloned) into the thread definition; captured outside-defined SSA
// values become explicit body operands of the launch and thread
// entry-block arguments.
//
// Promoted foralls may be direct children of a func.func body or
// guarded by one or more scf.if ops directly under that func.func.
// Foralls nested under loops, dataflow threads, or graph bodies remain
// in place so later graph extraction cannot clone launchers into a
// graph body. Aggregation-form foralls (with shared_outs / op
// results) are left in place; the raise pipeline already lowers those
// through Part 2 normalization.
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
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
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
    return ::mlir::arith::ConstantOp::create(builder, loc,
                                             builder.getIndexAttr(v));
  }
  return ::llvm::cast<::mlir::Value>(ofr);
}

// Collect every value used inside the forall region that is defined above it.
void collectCapturedValues(::mlir::scf::ForallOp forall,
                           ::llvm::SetVector<::mlir::Value> &captures) {
  ::mlir::getUsedValuesDefinedAbove(forall.getRegion(), captures);
}

bool isPromotableForall(::mlir::func::FuncOp func,
                        ::mlir::scf::ForallOp forall) {
  for (::mlir::Operation *parent = forall->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (parent == func.getOperation())
      return true;
    if (!::llvm::isa<::mlir::scf::IfOp>(parent))
      return false;
  }
  return false;
}

bool isLlvmPointerType(::mlir::Type type) {
  return ::llvm::isa<::mlir::LLVM::LLVMPointerType>(type);
}

bool isSideEffectFreeSetupOp(::mlir::Operation &op) {
  if (op.getNumRegions() != 0 || op.getNumSuccessors() != 0)
    return false;
  if (op.hasTrait<::mlir::OpTrait::IsTerminator>())
    return false;
  if (op.hasTrait<::mlir::OpTrait::SymbolTable>())
    return false;
  if (::llvm::isa<::mlir::FunctionOpInterface, ::mlir::CallOpInterface>(op))
    return false;
  if (auto effects = ::llvm::dyn_cast<::mlir::MemoryEffectOpInterface>(&op))
    return effects.hasNoEffect();
  return ::mlir::isPure(&op);
}

bool isBlockedStandaloneStructuredNumericOp(::mlir::Operation *op) {
  ::llvm::StringRef name = op->getName().getStringRef();
  return name == "arith.divsi" || name == "arith.divui" ||
         name == "arith.remf" || name == "arith.remsi" ||
         name == "arith.remui" || name == "llvm.fptosi" ||
         name == "llvm.sitofp";
}

bool isFloatType(::mlir::Type type) {
  return ::llvm::isa<::mlir::FloatType>(type);
}

bool touchesFloatType(::mlir::Operation *op) {
  for (::mlir::Value operand : op->getOperands()) {
    if (isFloatType(operand.getType()))
      return true;
  }
  for (::mlir::Value result : op->getResults()) {
    if (isFloatType(result.getType()))
      return true;
  }
  return false;
}

bool isMemsetIntrinsic(::mlir::Operation *op) {
  return op->getName().getStringRef() == "llvm.intr.memset";
}

bool isZeroIntegerConstant(::mlir::Value value) {
  if (auto constant = value.getDefiningOp<::mlir::arith::ConstantOp>()) {
    auto intAttr = ::llvm::dyn_cast<::mlir::IntegerAttr>(constant.getValue());
    return intAttr && intAttr.getValue().isZero();
  }
  if (auto constant = value.getDefiningOp<::dataflow::ConstantOp>()) {
    auto intAttr =
        ::llvm::dyn_cast<::mlir::IntegerAttr>(constant.getConstValue());
    return intAttr && intAttr.getValue().isZero();
  }
  return false;
}

bool isSupportedStandaloneStructuredMemsetOp(::mlir::Operation *op) {
  if (!isMemsetIntrinsic(op))
    return false;
  if (auto volatileAttr = op->getAttrOfType<::mlir::BoolAttr>("isVolatile")) {
    if (volatileAttr.getValue())
      return false;
  }
  if (op->getNumOperands() != 3)
    return false;

  ::mlir::Value dst = op->getOperand(0);
  ::mlir::Value byteValue = op->getOperand(1);
  ::mlir::Value byteCount = op->getOperand(2);
  return isLlvmPointerType(dst.getType()) &&
         ::llvm::isa<::mlir::IntegerType>(byteValue.getType()) &&
         ::llvm::isa<::mlir::IntegerType, ::mlir::IndexType>(
             byteCount.getType()) &&
         isZeroIntegerConstant(byteValue);
}

bool isEffectFormStructuredBodyCandidate(::mlir::Block *body);

bool isEffectFormStructuredRegionCandidate(::mlir::Region &region) {
  if (region.empty())
    return true;
  if (!region.hasOneBlock())
    return false;
  return isEffectFormStructuredBodyCandidate(&region.front());
}

bool isEffectFormForallGraphCandidate(::mlir::scf::ForallOp forall) {
  if (!forall.getOutputs().empty() || forall.getNumResults() != 0)
    return false;
  auto inParallel = forall.getTerminator();
  if (inParallel.getRegion().empty() || !inParallel.getRegion().front().empty())
    return false;
  return isEffectFormStructuredBodyCandidate(forall.getBody());
}

bool isEffectFormStructuredBodyCandidate(::mlir::Block *body) {
  if (!body)
    return true;
  for (::mlir::Operation &nested : body->without_terminator()) {
    if (::llvm::isa<::dataflow::GraphLaunchOp, ::dataflow::ThreadLaunchOp,
                    ::dataflow::GraphFuncOp, ::dataflow::ThreadOp,
                    ::mlir::scf::ForOp, ::mlir::scf::WhileOp>(&nested))
      return false;
    if (::llvm::isa<::mlir::FunctionOpInterface, ::mlir::CallOpInterface>(
            &nested))
      return false;
    if (auto forall = ::llvm::dyn_cast<::mlir::scf::ForallOp>(&nested)) {
      if (!isEffectFormForallGraphCandidate(forall))
        return false;
      continue;
    }
    if (auto ifOp = ::llvm::dyn_cast<::mlir::scf::IfOp>(&nested)) {
      if (ifOp.getNumResults() != 0)
        return false;
      if (!isEffectFormStructuredRegionCandidate(ifOp.getThenRegion()))
        return false;
      if (!isEffectFormStructuredRegionCandidate(ifOp.getElseRegion()))
        return false;
      continue;
    }
    if (nested.getNumRegions() != 0)
      return false;
    if (nested.hasTrait<::mlir::OpTrait::SymbolTable>())
      return false;
  }
  return true;
}

bool isSupportedStandaloneStructuredTopLevelOp(::mlir::Operation *op) {
  if (auto forOp = ::llvm::dyn_cast<::mlir::scf::ForOp>(op))
    return forOp.getInitArgs().empty();
  if (auto whileOp = ::llvm::dyn_cast<::mlir::scf::WhileOp>(op))
    return whileOp->getNumOperands() == whileOp->getNumResults();
  if (auto ifOp = ::llvm::dyn_cast<::mlir::scf::IfOp>(op))
    return ifOp.getNumResults() == 0;
  return false;
}

bool hasUnsupportedStandaloneStructuredBody(::mlir::Operation *root) {
  bool unsupported = false;
  root->walk([&](::mlir::Operation *nested) -> ::mlir::WalkResult {
    if (nested == root)
      return ::mlir::WalkResult::advance();
    if (::llvm::isa<::dataflow::GraphLaunchOp, ::dataflow::ThreadLaunchOp,
                    ::dataflow::GraphFuncOp, ::dataflow::ThreadOp,
                    ::mlir::func::FuncOp>(nested)) {
      unsupported = true;
      return ::mlir::WalkResult::interrupt();
    }
    if (isMemsetIntrinsic(nested) &&
        !isSupportedStandaloneStructuredMemsetOp(nested)) {
      unsupported = true;
      return ::mlir::WalkResult::interrupt();
    }
    if (::llvm::isa<::mlir::CallOpInterface>(nested) &&
        !::llvm::isa<::mlir::LLVM::CallIntrinsicOp>(nested)) {
      unsupported = true;
      return ::mlir::WalkResult::interrupt();
    }
    if (auto forOp = ::llvm::dyn_cast<::mlir::scf::ForOp>(nested)) {
      if (!forOp.getInitArgs().empty()) {
        unsupported = true;
        return ::mlir::WalkResult::interrupt();
      }
    }
    if (auto forall = ::llvm::dyn_cast<::mlir::scf::ForallOp>(nested)) {
      if (isEffectFormForallGraphCandidate(forall))
        return ::mlir::WalkResult::advance();
      unsupported = true;
      return ::mlir::WalkResult::interrupt();
    }
    if (auto inParallel = ::llvm::dyn_cast<::mlir::scf::InParallelOp>(nested)) {
      if (!inParallel.getRegion().empty() &&
          inParallel.getRegion().front().empty())
        return ::mlir::WalkResult::advance();
      unsupported = true;
      return ::mlir::WalkResult::interrupt();
    }
    if (nested->getNumRegions() != 0 &&
        !::llvm::isa<::mlir::scf::ForOp, ::mlir::scf::WhileOp,
                     ::mlir::scf::IfOp, ::mlir::scf::IndexSwitchOp>(nested)) {
      unsupported = true;
      return ::mlir::WalkResult::interrupt();
    }
    return ::mlir::WalkResult::advance();
  });
  return unsupported;
}

bool isStandaloneStructuredGraphCandidate(::mlir::func::FuncOp func) {
  if (func.isExternal())
    return false;
  if (func.getSymName() == "main")
    return false;
  if (func.getSymName().starts_with("arm_"))
    return false;
  if (!func.getFunctionType().getResults().empty())
    return false;
  if (!func.getBody().hasOneBlock())
    return false;

  bool sawStructuredOp = false;
  bool sawMemset = false;
  bool sawBlockedSetupNumericOp = false;
  bool sawBlockedBodyNumericOp = false;
  bool sawSequentialStructuredControl = false;
  bool sawFloatTypedOp = false;
  ::mlir::Block &entry = func.getBody().front();
  for (::mlir::Operation &op : entry.without_terminator()) {
    if (touchesFloatType(&op))
      sawFloatTypedOp = true;
    if (isBlockedStandaloneStructuredNumericOp(&op))
      sawBlockedSetupNumericOp = true;
    if (isMemsetIntrinsic(&op)) {
      if (!isSupportedStandaloneStructuredMemsetOp(&op))
        return false;
      sawMemset = true;
      continue;
    }
    if (isSideEffectFreeSetupOp(op))
      continue;
    if (!isSupportedStandaloneStructuredTopLevelOp(&op))
      return false;
    if (hasUnsupportedStandaloneStructuredBody(&op))
      return false;
    op.walk([&](::mlir::Operation *nested) {
      if (touchesFloatType(nested))
        sawFloatTypedOp = true;
      if (isBlockedStandaloneStructuredNumericOp(nested))
        sawBlockedBodyNumericOp = true;
      if (isMemsetIntrinsic(nested))
        sawMemset = true;
      if (::llvm::isa<::mlir::scf::WhileOp, ::mlir::scf::IndexSwitchOp>(nested))
        sawSequentialStructuredControl = true;
    });
    sawStructuredOp = true;
  }

  return sawStructuredOp && sawSequentialStructuredControl &&
         !sawFloatTypedOp &&
         (!sawMemset ||
          (!sawBlockedSetupNumericOp && !sawBlockedBodyNumericOp));
}

struct LowerForallToThreadPass
    : public ::mlir::PassWrapper<LowerForallToThreadPass,
                                 ::mlir::OperationPass<::mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerForallToThreadPass)

  ::llvm::StringRef getArgument() const final {
    return "loom-lower-forall-to-thread";
  }
  ::llvm::StringRef getDescription() const final {
    return "Lower effect-form scf.forall ops inside func.func bodies into "
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
      if (isStandaloneStructuredGraphCandidate(func))
        return;
      Pending p;
      p.func = func;
      // Source-order foralls in the function body or under direct
      // scf.if guards. Loop-nested foralls stay in place because their
      // launch cardinality depends on the enclosing loop execution.
      func.walk([&](::mlir::scf::ForallOp forall) {
        if (!isPromotableForall(func, forall))
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

    // Effect-form requirement: no shared_outs, op results, or non-empty
    // in_parallel terminator. Other forms remain unchanged.
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
    std::string symStem = (stem + "_" + ::llvm::Twine(seq)).str();
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
