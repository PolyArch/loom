// Lower scf.for ops with iter_args (i.e., structured reductions /
// loop-carried recurrences) found inside dataflow.thread bodies into
// `dataflow.graph` symbol definitions plus matching
// `dataflow.graph.launch` ops.
//
// scf.for ops without iter_args are left in place. Straight-line
// dataflow.thread bodies are also extracted into graph bodies so
// element-wise kernels have an explicit SpatialCore graph surface. The
// graph function_type contains only normalized application payload ports.
// Start/done remain explicit launch protocol endpoints, and graph.return owns
// the segmented payload boundary plus retirement frontier.

#include "Frontend/Lowering/Passes.h"
#include "Frontend/Lowering/StreamLoopAttrs.h"
#include "GraphRegionLowering.h"

#include "Dataflow/IR/DataflowDialect.h"
#include "Dataflow/IR/DataflowGraphValidation.h"
#include "Dataflow/IR/DataflowOps.h"
#include "Dataflow/IR/DataflowThreadCompletion.h"
#include "Frontend/IR/LoomDialect.h"
#include "Frontend/IR/LoomOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"

#include <array>

namespace {

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

void addThreadCompletionFrontier(::dataflow::ThreadOp thread,
                                 ::mlir::Value completion) {
  auto yield = ::mlir::cast<::dataflow::ThreadYieldOp>(
      thread.getBody().front().getTerminator());
  ::llvm::SmallVector<::mlir::Value, 4> candidates(
      yield.getCompletionFrontier().begin(),
      yield.getCompletionFrontier().end());
  if (!::llvm::is_contained(candidates, completion))
    candidates.push_back(completion);
  ::llvm::SmallVector<::mlir::Value, 4> frontier =
      ::dataflow::computeMinimalThreadCompletionFrontier(candidates);
  yield.getCompletionFrontierMutable().assign(frontier);
}

::mlir::FailureOr<::mlir::Value>
propagateIfCompletion(::mlir::scf::IfOp ifOp, ::mlir::Value completion,
                      ::mlir::Value fallback, ::mlir::OpBuilder &builder) {
  ::mlir::Region *completionRegion =
      completion.getDefiningOp()->getParentRegion();
  bool inThen = ifOp.getThenRegion().isAncestor(completionRegion);
  bool inElse = ifOp.getElseRegion().isAncestor(completionRegion);
  if (inThen == inElse)
    return ::mlir::failure();

  ::llvm::SmallVector<::mlir::Type, 4> resultTypes(
      ifOp.getResultTypes().begin(), ifOp.getResultTypes().end());
  resultTypes.push_back(builder.getNoneType());
  builder.setInsertionPoint(ifOp);
  auto replacement = ::mlir::scf::IfOp::create(
      builder, ifOp.getLoc(), resultTypes, ifOp.getCondition(),
      /*withElseRegion=*/true);

  auto moveBranch = [&](::mlir::Region &source, ::mlir::Region &target,
                        ::mlir::Value branchCompletion) {
    ::mlir::Block &targetBlock = target.front();
    ::mlir::scf::YieldOp defaultYield;
    if (!targetBlock.empty())
      defaultYield = ::llvm::dyn_cast<::mlir::scf::YieldOp>(
          targetBlock.back());
    ::llvm::SmallVector<::mlir::Value, 4> yielded;
    if (!source.empty()) {
      ::mlir::Block &sourceBlock = source.front();
      auto sourceYield = ::mlir::cast<::mlir::scf::YieldOp>(
          sourceBlock.getTerminator());
      yielded.append(sourceYield.getResults().begin(),
                     sourceYield.getResults().end());
      targetBlock.getOperations().splice(
          defaultYield ? defaultYield->getIterator() : targetBlock.end(),
          sourceBlock.getOperations(),
          sourceBlock.begin(), sourceYield->getIterator());
      sourceYield.erase();
    }
    yielded.push_back(branchCompletion);
    if (defaultYield)
      builder.setInsertionPoint(defaultYield);
    else
      builder.setInsertionPointToEnd(&targetBlock);
    ::mlir::scf::YieldOp::create(builder, ifOp.getLoc(), yielded);
    if (defaultYield)
      defaultYield.erase();
  };

  moveBranch(ifOp.getThenRegion(), replacement.getThenRegion(),
             inThen ? completion : fallback);
  moveBranch(ifOp.getElseRegion(), replacement.getElseRegion(),
             inElse ? completion : fallback);

  for (auto [oldResult, newResult] :
       ::llvm::zip_equal(ifOp.getResults(),
                         replacement.getResults().take_front(
                             ifOp.getNumResults())))
    oldResult.replaceAllUsesWith(newResult);
  ::mlir::Value propagated = replacement.getResult(ifOp.getNumResults());
  ifOp.erase();
  return propagated;
}

::mlir::FailureOr<::mlir::Value>
propagateCompletionToThread(::dataflow::ThreadOp thread,
                            ::mlir::Value completion,
                            ::mlir::Value fallback,
                            ::mlir::OpBuilder &builder) {
  while (completion.getDefiningOp()->getParentOfType<::dataflow::ThreadOp>() ==
             thread &&
         completion.getDefiningOp()->getBlock() != &thread.getBody().front()) {
    ::mlir::Operation *parent = completion.getDefiningOp()->getParentOp();
    while (parent && parent != thread.getOperation() &&
           !::llvm::isa<::mlir::scf::IfOp>(parent))
      parent = parent->getParentOp();
    auto ifOp = ::llvm::dyn_cast_or_null<::mlir::scf::IfOp>(parent);
    if (!ifOp)
      return ::mlir::failure();
    auto propagated =
        propagateIfCompletion(ifOp, completion, fallback, builder);
    if (::mlir::failed(propagated))
      return ::mlir::failure();
    completion = *propagated;
  }
  return completion;
}

// Collect every value used inside the loop region that is defined above it.
void collectExternalUses(::mlir::scf::ForOp loop,
                         ::llvm::SetVector<::mlir::Value> &captures) {
  ::mlir::getUsedValuesDefinedAbove(loop.getRegion(), captures);
}

bool isNestedInReductionFor(::mlir::scf::ForOp loop) {
  for (::mlir::Operation *parent = loop->getParentOp(); parent;
       parent = parent->getParentOp()) {
    auto parentLoop = ::mlir::dyn_cast<::mlir::scf::ForOp>(parent);
    if (parentLoop && !parentLoop.getInitArgs().empty())
      return true;
  }
  return false;
}

bool isNestedInGraphParallelRegion(::mlir::scf::ForOp loop) {
  for (::mlir::Operation *parent = loop->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (::llvm::isa<::mlir::scf::ForallOp, ::mlir::scf::ParallelOp>(parent))
      return true;
  }
  return false;
}

bool isNestedInSelectedGraphParallelRegion(::mlir::scf::ForOp loop) {
  bool sawParallel = false;
  for (::mlir::Operation *parent = loop->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (!::llvm::isa<::mlir::scf::ForallOp, ::mlir::scf::ParallelOp>(parent))
      continue;
    sawParallel = true;
    if (!::loom::lowering::hasGraphOwnedParallelProvenance(parent))
      return false;
  }
  return sawParallel;
}

bool isPureScalarEpilogueOp(::mlir::Operation *op) {
  if (op->getNumResults() == 0 || op->getNumRegions() != 0 ||
      op->getNumSuccessors() != 0)
    return false;
  if (op->hasTrait<::mlir::OpTrait::IsTerminator>())
    return false;
  if (op->hasTrait<::mlir::OpTrait::SymbolTable>())
    return false;
  if (::llvm::isa<::mlir::FunctionOpInterface, ::mlir::CallOpInterface>(op))
    return false;
  return ::dataflow::isCanonicalDataflowActor(
      op, ::dataflow::CanonicalDataflowActorKind::Compute);
}

::mlir::LogicalResult
checkSelectedGraphLoweringLeaves(::mlir::Operation *root) {
  ::mlir::WalkResult result =
      root->walk(
          [&](::mlir::Operation *op) -> ::mlir::WalkResult {
            if (op->getNumRegions() != 0 || op->getNumSuccessors() != 0 ||
                op->hasTrait<::mlir::OpTrait::IsTerminator>())
              return ::mlir::WalkResult::advance();
            if (::loom::lowering::isSupportedGraphLoweringLeaf(op))
              return ::mlir::WalkResult::advance();
            op->emitError()
                << "loom-lower-for-to-graph: operation '"
                << op->getName().getStringRef()
                << "' is not a registered canonical Dataflow actor or a "
                   "supported graph-lowering operation";
            return ::mlir::WalkResult::interrupt();
          });
  return result.wasInterrupted() ? ::mlir::failure() : ::mlir::success();
}

struct ClassifiedGraphValues {
  ::llvm::SmallVector<::mlir::Value, 8> values;
  ::llvm::SmallVector<::mlir::Value, 4> memories;

  ::llvm::SmallVector<::mlir::Value, 8> ordered() const {
    ::llvm::SmallVector<::mlir::Value, 8> result(values.begin(), values.end());
    result.append(memories.begin(), memories.end());
    return result;
  }

  std::array<int32_t, 3> segments() const {
    return {static_cast<int32_t>(values.size()), 0,
            static_cast<int32_t>(memories.size())};
  }
};

::llvm::SmallVector<unsigned, 8> graphPortOrder(::mlir::ValueRange values) {
  ::llvm::SmallVector<unsigned, 8> order;
  order.reserve(values.size());
  for (bool memory : {false, true}) {
    for (auto [index, value] : ::llvm::enumerate(values)) {
      if (::dataflow::DataflowDialect::isMemoryCapabilityType(
              value.getType()) == memory)
        order.push_back(index);
    }
  }
  return order;
}

ClassifiedGraphValues classifyGraphValues(::mlir::ValueRange values) {
  ClassifiedGraphValues classified;
  for (unsigned index : graphPortOrder(values)) {
    ::mlir::Value value = values[index];
    if (::dataflow::DataflowDialect::isMemoryCapabilityType(value.getType()))
      classified.memories.push_back(value);
    else
      classified.values.push_back(value);
  }
  return classified;
}

template <typename GetAttr>
::llvm::SmallVector<::mlir::DictionaryAttr, 8>
reorderGraphInterfaceAttrs(::mlir::ValueRange values, GetAttr getAttr) {
  ::llvm::SmallVector<::mlir::DictionaryAttr, 8> attrs;
  attrs.reserve(values.size());
  for (unsigned index : graphPortOrder(values))
    attrs.push_back(getAttr(index));
  return attrs;
}

::mlir::Value findGraphPublicationMemoryRoot(::mlir::Value value,
                                             ::mlir::Block &threadEntry) {
  ::llvm::DenseSet<::mlir::Value> visited;
  while (value && visited.insert(value).second) {
    if (auto argument = ::llvm::dyn_cast<::mlir::BlockArgument>(value))
      return argument.getOwner() == &threadEntry ? value : ::mlir::Value{};
    ::mlir::Operation *def = value.getDefiningOp();
    if (!def)
      return {};
    if (::llvm::isa<::mlir::memref::AllocOp, ::mlir::memref::AllocaOp,
                    ::mlir::memref::GetGlobalOp, ::mlir::LLVM::AddressOfOp>(
            def))
      return value;
    if (auto view = ::llvm::dyn_cast<::mlir::ViewLikeOpInterface>(def)) {
      if (value != view.getViewDest())
        return {};
      value = view.getViewSource();
      continue;
    }
    if (auto cast = ::llvm::dyn_cast<::mlir::UnrealizedConversionCastOp>(def)) {
      if (cast.getInputs().size() != 1)
        return {};
      value = cast.getInputs().front();
      continue;
    }
    if (auto gep = ::llvm::dyn_cast<::mlir::LLVM::GEPOp>(def)) {
      value = gep.getBase();
      continue;
    }
    if (auto bitcast = ::llvm::dyn_cast<::mlir::LLVM::BitcastOp>(def)) {
      value = bitcast.getArg();
      continue;
    }
    return {};
  }
  return {};
}

::llvm::SmallVector<::mlir::NamedAttribute, 2>
graphSegmentAttrs(::mlir::OpBuilder &builder,
                  const ClassifiedGraphValues &inputs,
                  const ClassifiedGraphValues &results,
                  int32_t inputStreamCount = 0, int32_t resultStreamCount = 0) {
  std::array<int32_t, 3> inputSegments = inputs.segments();
  std::array<int32_t, 3> resultSegments = results.segments();
  inputSegments[1] = inputStreamCount;
  resultSegments[1] = resultStreamCount;
  return {
      builder.getNamedAttr("input_segments",
                           builder.getDenseI32ArrayAttr(inputSegments)),
      builder.getNamedAttr("result_segments",
                           builder.getDenseI32ArrayAttr(resultSegments)),
  };
}

void collectEpilogueOps(::mlir::scf::ForOp loop,
                        ::llvm::SmallVectorImpl<::mlir::Operation *> &ops) {
  ops.clear();
  ::mlir::Block *block = loop->getBlock();
  ::llvm::DenseSet<::mlir::Value> available;
  for (::mlir::Value value : loop.getResults())
    available.insert(value);

  for (auto it = std::next(::mlir::Block::iterator(loop.getOperation())),
            end = block->end();
       it != end; ++it) {
    ::mlir::Operation *op = &*it;
    if (op->hasTrait<::mlir::OpTrait::IsTerminator>())
      break;
    bool dependsOnReduction = false;
    bool dependsOnUnsupportedLocal = false;
    for (::mlir::Value operand : op->getOperands()) {
      if (available.contains(operand)) {
        dependsOnReduction = true;
        continue;
      }
      ::mlir::Operation *def = operand.getDefiningOp();
      if (def && def->getBlock() == block && def->isBeforeInBlock(op) &&
          !def->isBeforeInBlock(loop.getOperation())) {
        dependsOnUnsupportedLocal = true;
        break;
      }
    }
    if (dependsOnUnsupportedLocal)
      break;
    if (!dependsOnReduction)
      continue;
    if (!isPureScalarEpilogueOp(op))
      break;
    ops.push_back(op);
    for (::mlir::Value result : op->getResults())
      available.insert(result);
  }
}

void collectAdditionalCaptures(::mlir::scf::ForOp loop,
                               ::llvm::ArrayRef<::mlir::Operation *> ops,
                               ::llvm::SetVector<::mlir::Value> &captures) {
  ::llvm::DenseSet<::mlir::Operation *> opSet;
  for (::mlir::Operation *op : ops)
    opSet.insert(op);
  ::llvm::DenseSet<::mlir::Value> loopResults;
  for (::mlir::Value value : loop.getResults())
    loopResults.insert(value);

  for (::mlir::Operation *op : ops) {
    for (::mlir::Value operand : op->getOperands()) {
      if (loopResults.contains(operand))
        continue;
      ::mlir::Operation *def = operand.getDefiningOp();
      if (def && opSet.contains(def))
        continue;
      captures.insert(operand);
    }
  }
}

void computeGraphOutputs(::mlir::scf::ForOp loop,
                         ::llvm::ArrayRef<::mlir::Operation *> epilogueOps,
                         ::llvm::SmallVectorImpl<::mlir::Value> &outputs) {
  outputs.clear();

  ::llvm::DenseSet<::mlir::Operation *> epilogueSet;
  for (::mlir::Operation *op : epilogueOps)
    epilogueSet.insert(op);

  auto usedOutsideEpilogue = [&](::mlir::Value value) {
    for (::mlir::OpOperand &use : value.getUses())
      if (!epilogueSet.contains(use.getOwner()))
        return true;
    return false;
  };
  auto usedByEpilogue = [&](::mlir::Value value) {
    for (::mlir::OpOperand &use : value.getUses())
      if (epilogueSet.contains(use.getOwner()))
        return true;
    return false;
  };

  for (::mlir::Value value : loop.getResults()) {
    if (!::dataflow::DataflowDialect::isMemoryCapabilityType(value.getType()) ||
        usedOutsideEpilogue(value))
      outputs.push_back(value);
  }
  for (::mlir::Operation *op : epilogueOps) {
    for (::mlir::Value value : op->getResults()) {
      if (!usedByEpilogue(value))
        outputs.push_back(value);
    }
  }
}

struct LowerForToGraphPass
    : public ::mlir::PassWrapper<LowerForToGraphPass,
                                 ::mlir::OperationPass<::mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerForToGraphPass)

  ::llvm::StringRef getArgument() const final {
    return "loom-lower-for-to-graph";
  }
  ::llvm::StringRef getDescription() const final {
    return "Lower scf.for ops with iter_args inside dataflow.thread bodies "
           "into dataflow.graph definitions plus dataflow.graph.launch "
           "ops.";
  }

  void getDependentDialects(::mlir::DialectRegistry &registry) const final {
    registry.insert<::mlir::arith::ArithDialect, ::mlir::func::FuncDialect,
                    ::mlir::LLVM::LLVMDialect, ::mlir::math::MathDialect,
                    ::mlir::memref::MemRefDialect, ::mlir::scf::SCFDialect,
                    ::mlir::ub::UBDialect, ::dataflow::DataflowDialect,
                    ::loom::LoomDialect>();
  }

  void runOnOperation() final {
    ::mlir::ModuleOp module = getOperation();
    ::mlir::MLIRContext *ctx = &getContext();
    ::mlir::OwningOpRef<::mlir::ModuleOp> scratch(
        ::mlir::cast<::mlir::ModuleOp>(module->clone()));
    ::mlir::OpBuilder scratchBuilder(ctx);
    if (::mlir::failed(stageThreadCandidates(*scratch, scratchBuilder))) {
      signalPassFailure();
      return;
    }

    promoteStandaloneMemcpyFunctions(*scratch, scratchBuilder);
    promoteStandaloneScalarReturnFunctions(*scratch, scratchBuilder);
    promoteStandaloneStructuredStatusOutParamFunctions(*scratch,
                                                        scratchBuilder);
    promoteStandaloneStructuredOutParamFunctions(*scratch, scratchBuilder);
    promoteStandaloneStructuredFunctions(*scratch, scratchBuilder);

    if (::mlir::failed(publishSpatialRegions(*scratch, scratchBuilder)) ||
        ::mlir::failed(finalizePublishedModule(*scratch))) {
      signalPassFailure();
      return;
    }

    module->setAttrs((*scratch)->getAttrs());
    module.getBodyRegion().takeBody(scratch->getBodyRegion());
  }

  ::mlir::LogicalResult stageThreadCandidates(::mlir::ModuleOp module,
                                              ::mlir::OpBuilder &builder) {
    struct Pending {
      ::dataflow::ThreadOp thread;
      ::llvm::SmallVector<::mlir::scf::ForOp, 4> loops;
    };
    ::llvm::SmallVector<Pending, 4> pending;
    for (::dataflow::ThreadOp thread : module.getOps<::dataflow::ThreadOp>()) {
      Pending p;
      p.thread = thread;
      thread.walk([&](::mlir::scf::ForOp loop) {
        if (loop.getInitArgs().empty())
          return; // No iter_args: keep as scf.for in the thread body.
        if (isNestedInReductionFor(loop))
          return;
        if (isNestedInSelectedGraphParallelRegion(loop))
          return;
        // Skip loops already owned by a graph boundary.
        if (loop->getParentOfType<::dataflow::GraphOp>())
          return;
        p.loops.push_back(loop);
      });
      if (!p.loops.empty())
        pending.push_back(std::move(p));
    }

    for (Pending &p : pending) {
      ::llvm::StringRef threadSym = p.thread.getSymName();
      std::string stem = "g_" + sanitizeSymbol(threadSym);
      for (auto [seq, loop] : ::llvm::enumerate(p.loops)) {
        if (failed(stageOne(module, loop, stem, seq, builder)))
          return ::mlir::failure();
      }
    }

    return stageStraightLineThreads(module, builder);
  }

  ::mlir::LogicalResult finalizePublishedModule(::mlir::ModuleOp module) {
    // Stream endpoints temporarily retain channel block arguments in the
    // scratch module until graph-region lowering replaces them with ports.
    ::mlir::PassManager lowerer(module.getContext());
    lowerer.enableVerifier(false);
    lowerer.addPass(::mlir::createCanonicalizerPass());
    lowerer.addPass(::loom::lowering::createLowerKnownLibraryCallsPass());
    lowerer.addPass(::loom::lowering::createLowerGraphMemoryPass());
    if (::mlir::failed(lowerer.run(module)) || ::mlir::failed(verify(module)))
      return ::mlir::failure();

    ::mlir::PassManager finalizer(module.getContext());
    finalizer.enableVerifier(true);
    finalizer.addPass(::loom::lowering::createLowerGraphConstantsPass());
    finalizer.addPass(::mlir::createCanonicalizerPass());
    if (::mlir::failed(finalizer.run(module)))
      return ::mlir::failure();

    if (auto error = ::dataflow::validateFinalizedProgram(module)) {
      module.emitError("canonical Dataflow publication failed: ")
          << ::llvm::toString(std::move(error));
      return ::mlir::failure();
    }
    return ::mlir::success();
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
    if (auto kind = ::dataflow::classifyCanonicalDataflowActor(&op))
      return *kind == ::dataflow::CanonicalDataflowActorKind::Compute;
    if (!::loom::lowering::isSupportedGraphLoweringLeaf(&op))
      return false;
    if (auto effects = ::llvm::dyn_cast<::mlir::MemoryEffectOpInterface>(&op))
      return effects.hasNoEffect();
    return ::mlir::isPure(&op);
  }

  bool isStandaloneMemcpyFunctionCandidate(::mlir::func::FuncOp func) {
    if (func.isExternal())
      return false;
    if (!func.getFunctionType().getResults().empty())
      return false;
    ::mlir::Region &body = func.getBody();
    if (!body.hasOneBlock())
      return false;
    ::mlir::Block &entry = body.front();
    ::mlir::LLVM::MemcpyOp candidate;
    for (::mlir::Operation &op : entry.without_terminator()) {
      if (auto memcpy = ::llvm::dyn_cast<::mlir::LLVM::MemcpyOp>(&op)) {
        if (candidate || memcpy.getIsVolatile())
          return false;
        candidate = memcpy;
        continue;
      }
      if (!isSideEffectFreeSetupOp(op))
        return false;
    }
    return static_cast<bool>(candidate);
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

  bool isMemsetIntrinsic(::mlir::Operation *op) {
    return ::llvm::isa<::mlir::LLVM::MemsetOp>(op);
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
    auto memset = ::llvm::dyn_cast<::mlir::LLVM::MemsetOp>(op);
    if (!memset)
      return false;
    if (memset.getIsVolatile())
      return false;
    return isZeroIntegerConstant(memset.getVal());
  }

  bool hasUnsupportedStandaloneStructuredBody(::mlir::Operation *root) {
    bool unsupported = false;
    root->walk([&](::mlir::Operation *nested) -> ::mlir::WalkResult {
      if (nested == root)
        return ::mlir::WalkResult::advance();
      if (::llvm::isa<::dataflow::GraphLaunchOp, ::dataflow::ThreadLaunchOp,
                      ::dataflow::GraphOp, ::dataflow::ThreadOp,
                      ::mlir::func::FuncOp>(nested)) {
        unsupported = true;
        return ::mlir::WalkResult::interrupt();
      }
      if (isMemsetIntrinsic(nested) &&
          !isSupportedStandaloneStructuredMemsetOp(nested)) {
        unsupported = true;
        return ::mlir::WalkResult::interrupt();
      }
      if (::llvm::isa<::mlir::CallOpInterface>(nested)) {
        unsupported = true;
        return ::mlir::WalkResult::interrupt();
      }
      if (auto forOp = ::llvm::dyn_cast<::mlir::scf::ForOp>(nested)) {
        if (!forOp.getInitArgs().empty()) {
          unsupported = true;
          return ::mlir::WalkResult::interrupt();
        }
      }
      if (auto whileOp = ::llvm::dyn_cast<::mlir::scf::WhileOp>(nested)) {
        if (::llvm::any_of(
                whileOp.getInits().getTypes(),
                ::dataflow::DataflowDialect::isMemoryCapabilityType)) {
          unsupported = true;
          return ::mlir::WalkResult::interrupt();
        }
      }
      if (auto forall = ::llvm::dyn_cast<::mlir::scf::ForallOp>(nested)) {
        if (::loom::lowering::hasGraphOwnedParallelProvenance(forall) &&
            isEffectFormForallGraphCandidate(forall))
          return ::mlir::WalkResult::advance();
        unsupported = true;
        return ::mlir::WalkResult::interrupt();
      }
      if (auto inParallel =
              ::llvm::dyn_cast<::mlir::scf::InParallelOp>(nested)) {
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
      if (nested->getNumRegions() == 0 &&
          !nested->hasTrait<::mlir::OpTrait::IsTerminator>() &&
          !::loom::lowering::isSupportedGraphLoweringLeaf(nested)) {
        unsupported = true;
        return ::mlir::WalkResult::interrupt();
      }
      return ::mlir::WalkResult::advance();
    });
    return unsupported;
  }

  bool isScalarReturnGraphBodyOp(::mlir::Operation *op) {
    if (::llvm::isa<::mlir::func::ReturnOp, ::mlir::scf::YieldOp>(op))
      return true;
    if (::llvm::isa<::mlir::scf::IfOp>(op))
      return true;
    if (::llvm::isa<::mlir::FunctionOpInterface>(op))
      return false;
    if (::llvm::isa<::mlir::CallOpInterface>(op))
      return false;
    if (op->getNumRegions() != 0 || op->getNumSuccessors() != 0)
      return false;
    if (op->hasTrait<::mlir::OpTrait::SymbolTable>())
      return false;
    if (::dataflow::isCanonicalDataflowActor(op))
      return true;
    if (!::loom::lowering::isSupportedGraphLoweringLeaf(op))
      return false;
    if (auto effects = ::llvm::dyn_cast<::mlir::MemoryEffectOpInterface>(op)) {
      if (effects.template hasEffect<::mlir::MemoryEffects::Write>() ||
          effects.template hasEffect<::mlir::MemoryEffects::Allocate>() ||
          effects.template hasEffect<::mlir::MemoryEffects::Free>())
        return false;
      return true;
    }
    return ::mlir::isPure(op);
  }

  bool isStandaloneScalarReturnFunctionCandidate(::mlir::func::FuncOp func) {
    if (func.isExternal())
      return false;
    if (func.getSymName() == "main")
      return false;
    if (func.getFunctionType().getResults().empty())
      return false;
    if (!func.getBody().hasOneBlock())
      return false;
    auto returnOp = ::llvm::dyn_cast<::mlir::func::ReturnOp>(
        func.getBody().front().getTerminator());
    if (!returnOp ||
        returnOp.getNumOperands() != func.getFunctionType().getNumResults())
      return false;

    bool hasLoad = false;
    bool unsupported = false;
    func.getBody().walk([&](::mlir::Operation *nested) -> ::mlir::WalkResult {
      if (::llvm::isa<::mlir::LLVM::LoadOp>(nested))
        hasLoad = true;
      if (!isScalarReturnGraphBodyOp(nested)) {
        unsupported = true;
        return ::mlir::WalkResult::interrupt();
      }
      return ::mlir::WalkResult::advance();
    });
    return hasLoad && !unsupported;
  }

  void promoteGraphOnlyFunction(::mlir::ModuleOp module,
                                ::mlir::func::FuncOp func,
                                ::mlir::OpBuilder &builder) {
    ::mlir::Type noneType = builder.getType<::mlir::NoneType>();
    ::mlir::Location loc = func.getLoc();
    std::string stem = "g_" + sanitizeSymbol(func.getSymName()) + "_0";
    if (::mlir::SymbolTable(module).lookup(stem))
      return;
    std::string graphName = uniqueSymbol(module, stem);

    ::mlir::Block &funcEntry = func.getBody().front();
    auto returnOp =
        ::llvm::cast<::mlir::func::ReturnOp>(funcEntry.getTerminator());
    ClassifiedGraphValues graphInputs =
        classifyGraphValues(funcEntry.getArguments());
    ClassifiedGraphValues graphResults =
        classifyGraphValues(returnOp.getOperands());
    ::llvm::SmallVector<::mlir::Value, 8> orderedInputs = graphInputs.ordered();
    ::llvm::SmallVector<::mlir::Value, 8> orderedResults =
        graphResults.ordered();
    auto graphArgAttrs = reorderGraphInterfaceAttrs(
        funcEntry.getArguments(), [&](unsigned index) {
          return ::mlir::function_interface_impl::getArgAttrDict(func, index);
        });
    auto graphResultAttrs =
        reorderGraphInterfaceAttrs(returnOp.getOperands(), [&](unsigned index) {
          return ::mlir::function_interface_impl::getResultAttrDict(func,
                                                                    index);
        });
    ::llvm::SmallVector<::mlir::Type, 8> inputTypes;
    for (::mlir::Value value : orderedInputs)
      inputTypes.push_back(value.getType());
    ::llvm::SmallVector<::mlir::Type, 4> resultTypes;
    for (::mlir::Value value : orderedResults)
      resultTypes.push_back(value.getType());
    ::mlir::FunctionType graphType =
        builder.getFunctionType(inputTypes, resultTypes);
    auto segmentAttrs = graphSegmentAttrs(builder, graphInputs, graphResults);

    builder.setInsertionPointToEnd(module.getBody());
    auto graph = ::dataflow::GraphOp::create(builder, loc, graphName, graphType,
                                             segmentAttrs);
    graph.setSymVisibilityAttr(builder.getStringAttr("private"));
    ::mlir::function_interface_impl::setAllArgAttrDicts(graph, graphArgAttrs);
    ::mlir::function_interface_impl::setAllResultAttrDicts(graph,
                                                           graphResultAttrs);

    ::mlir::Block *graphEntry = builder.createBlock(&graph.getBody());
    graphEntry->addArgument(noneType, loc);
    for (::mlir::Type ty : inputTypes)
      graphEntry->addArgument(ty, loc);

    ::mlir::IRMapping mapping;
    for (auto [i, value] : ::llvm::enumerate(orderedInputs))
      mapping.map(value, graphEntry->getArgument(i + 1));

    builder.setInsertionPointToEnd(graphEntry);
    for (::mlir::Operation &op : funcEntry.without_terminator())
      builder.clone(op, mapping);
    ::llvm::SmallVector<::mlir::Value, 4> returnValues;
    for (::mlir::Value value : graphResults.values)
      returnValues.push_back(mapping.lookupOrDefault(value));
    ::llvm::SmallVector<::mlir::Value, 4> returnMemories;
    for (::mlir::Value value : graphResults.memories)
      returnMemories.push_back(mapping.lookupOrDefault(value));
    ::dataflow::GraphReturnOp::create(
        builder, loc, returnValues, ::mlir::ValueRange{}, returnMemories,
        ::mlir::ValueRange{graphEntry->getArgument(0)});
  }

  void promoteStandaloneMemcpyFunctions(::mlir::ModuleOp module,
                                        ::mlir::OpBuilder &builder) {
    ::llvm::SmallVector<::mlir::func::FuncOp, 4> funcs;
    module.walk([&](::mlir::func::FuncOp func) {
      if (isStandaloneMemcpyFunctionCandidate(func))
        funcs.push_back(func);
    });

    for (::mlir::func::FuncOp func : funcs)
      promoteGraphOnlyFunction(module, func, builder);
  }

  void promoteStandaloneScalarReturnFunctions(::mlir::ModuleOp module,
                                              ::mlir::OpBuilder &builder) {
    ::llvm::SmallVector<::mlir::func::FuncOp, 4> funcs;
    module.walk([&](::mlir::func::FuncOp func) {
      if (isStandaloneScalarReturnFunctionCandidate(func))
        funcs.push_back(func);
    });

    for (::mlir::func::FuncOp func : funcs)
      promoteGraphOnlyFunction(module, func, builder);
  }

  bool isResultBearingStandaloneStructuredTopLevelOp(::mlir::Operation *op) {
    if (auto whileOp = ::llvm::dyn_cast<::mlir::scf::WhileOp>(op))
      return whileOp->getNumResults() > 0 &&
             whileOp->getNumOperands() <= whileOp->getNumResults();
    if (auto ifOp = ::llvm::dyn_cast<::mlir::scf::IfOp>(op))
      return ifOp.getNumResults() > 0;
    if (auto switchOp = ::llvm::dyn_cast<::mlir::scf::IndexSwitchOp>(op))
      return switchOp->getNumResults() > 0;
    return false;
  }

  bool isEntryPointerArgument(::mlir::Value value, ::mlir::Block &entry) {
    auto arg = ::llvm::dyn_cast<::mlir::BlockArgument>(value);
    return arg && arg.getOwner() == &entry &&
           ::dataflow::DataflowDialect::isMemoryCapabilityType(arg.getType());
  }

  bool isScalarNonPointerType(::mlir::Type type) {
    return ::llvm::isa<::mlir::IntegerType, ::mlir::FloatType,
                       ::mlir::IndexType>(type) &&
           !::dataflow::DataflowDialect::isMemoryCapabilityType(type);
  }

  bool structuredRootStoresOnlyEntryPointers(::mlir::Operation *root,
                                             ::mlir::Block &entry,
                                             bool &sawStore) {
    bool ok = true;
    root->walk([&](::mlir::Operation *nested) -> ::mlir::WalkResult {
      auto store = ::llvm::dyn_cast<::mlir::LLVM::StoreOp>(nested);
      if (!store)
        return ::mlir::WalkResult::advance();
      sawStore = true;
      if (!isEntryPointerArgument(store.getAddr(), entry)) {
        ok = false;
        return ::mlir::WalkResult::interrupt();
      }
      return ::mlir::WalkResult::advance();
    });
    return ok;
  }

  bool isStandaloneStructuredStatusOutParamFunctionCandidate(
      ::mlir::func::FuncOp func) {
    if (func.isExternal())
      return false;
    if (func.getSymName() == "main")
      return false;
    if (func.getFunctionType().getNumResults() != 1)
      return false;
    if (!isScalarNonPointerType(func.getFunctionType().getResult(0)))
      return false;
    if (!func.getBody().hasOneBlock())
      return false;
    ::mlir::Block &entry = func.getBody().front();

    bool hasPointerInput = false;
    for (::mlir::Type ty : func.getFunctionType().getInputs())
      hasPointerInput |=
          ::dataflow::DataflowDialect::isMemoryCapabilityType(ty);
    if (!hasPointerInput)
      return false;

    auto returnOp =
        ::llvm::dyn_cast<::mlir::func::ReturnOp>(entry.getTerminator());
    if (!returnOp || returnOp.getNumOperands() != 1)
      return false;

    bool sawStructuredOp = false;
    bool sawStore = false;
    ::llvm::DenseSet<::mlir::Value> structuredResults;
    for (::mlir::Operation &op : entry.without_terminator()) {
      if (isSideEffectFreeSetupOp(op))
        continue;

      if (!isSupportedStandaloneStructuredTopLevelOp(&op) &&
          !isResultBearingStandaloneStructuredTopLevelOp(&op))
        return false;
      if (hasUnsupportedStandaloneStructuredBody(&op))
        return false;
      op.walk([&](::mlir::Operation *nested) {
        if (::llvm::isa<::mlir::LLVM::StoreOp>(nested))
          sawStore = true;
      });
      for (::mlir::Value result : op.getResults())
        structuredResults.insert(result);
      sawStructuredOp = true;
    }

    return sawStructuredOp && sawStore &&
           (structuredResults.contains(returnOp.getOperand(0)) ||
            isZeroIntegerConstant(returnOp.getOperand(0)));
  }

  void promoteStandaloneStructuredStatusOutParamFunctions(
      ::mlir::ModuleOp module, ::mlir::OpBuilder &builder) {
    ::llvm::SmallVector<::mlir::func::FuncOp, 4> funcs;
    module.walk([&](::mlir::func::FuncOp func) {
      if (isStandaloneStructuredStatusOutParamFunctionCandidate(func))
        funcs.push_back(func);
    });

    for (::mlir::func::FuncOp func : funcs)
      promoteGraphOnlyFunction(module, func, builder);
  }

  bool
  isStandaloneStructuredOutParamFunctionCandidate(::mlir::func::FuncOp func) {
    if (func.isExternal())
      return false;
    if (func.getSymName() == "main")
      return false;
    if (!func.getFunctionType().getResults().empty())
      return false;
    if (!func.getBody().hasOneBlock())
      return false;
    ::mlir::Block &entry = func.getBody().front();

    bool hasPointerInput = false;
    for (::mlir::Type ty : func.getFunctionType().getInputs())
      hasPointerInput |=
          ::dataflow::DataflowDialect::isMemoryCapabilityType(ty);
    if (!hasPointerInput)
      return false;

    bool sawStructuredOp = false;
    bool sawStore = false;
    ::llvm::DenseSet<::mlir::Value> structuredResults;
    for (::mlir::Operation &op : entry.without_terminator()) {
      if (!sawStructuredOp && isSideEffectFreeSetupOp(op))
        continue;

      if (!sawStructuredOp) {
        if (!isResultBearingStandaloneStructuredTopLevelOp(&op))
          return false;
        if (hasUnsupportedStandaloneStructuredBody(&op))
          return false;
        for (::mlir::Value result : op.getResults())
          structuredResults.insert(result);
        sawStructuredOp = true;
        continue;
      }

      auto store = ::llvm::dyn_cast<::mlir::LLVM::StoreOp>(&op);
      if (!store)
        return false;
      if (sawStore)
        return false;
      if (!structuredResults.contains(store.getValue()))
        return false;
      if (!isEntryPointerArgument(store.getAddr(), entry))
        return false;
      sawStore = true;
    }

    return sawStructuredOp && sawStore;
  }

  void
  promoteStandaloneStructuredOutParamFunctions(::mlir::ModuleOp module,
                                               ::mlir::OpBuilder &builder) {
    ::llvm::SmallVector<::mlir::func::FuncOp, 4> funcs;
    module.walk([&](::mlir::func::FuncOp func) {
      if (isStandaloneStructuredOutParamFunctionCandidate(func))
        funcs.push_back(func);
    });

    for (::mlir::func::FuncOp func : funcs)
      promoteGraphOnlyFunction(module, func, builder);
  }

  bool isStandaloneStructuredFunctionCandidate(::mlir::func::FuncOp func) {
    if (func.isExternal())
      return false;
    if (func.getSymName() == "main")
      return false;
    if (!func.getFunctionType().getResults().empty())
      return false;
    ::mlir::Region &body = func.getBody();
    if (!body.hasOneBlock())
      return false;
    ::mlir::Block &entry = body.front();

    bool sawStructuredOp = false;
    for (::mlir::Operation &op : entry.without_terminator()) {
      if (isMemsetIntrinsic(&op)) {
        if (!isSupportedStandaloneStructuredMemsetOp(&op))
          return false;
        continue;
      }
      if (isSideEffectFreeSetupOp(op))
        continue;
      if (!isSupportedStandaloneStructuredTopLevelOp(&op) &&
          !isResultBearingStandaloneStructuredTopLevelOp(&op))
        return false;
      if (hasUnsupportedStandaloneStructuredBody(&op))
        return false;
      sawStructuredOp = true;
    }

    return sawStructuredOp;
  }

  void promoteStandaloneStructuredFunctions(::mlir::ModuleOp module,
                                            ::mlir::OpBuilder &builder) {
    ::llvm::SmallVector<::mlir::func::FuncOp, 4> funcs;
    module.walk([&](::mlir::func::FuncOp func) {
      if (isStandaloneStructuredFunctionCandidate(func))
        funcs.push_back(func);
    });

    for (::mlir::func::FuncOp func : funcs)
      promoteGraphOnlyFunction(module, func, builder);
  }

  ::mlir::LogicalResult stageOne(::mlir::ModuleOp module,
                                 ::mlir::scf::ForOp loop,
                                 ::llvm::StringRef stem, size_t seq,
                                 ::mlir::OpBuilder &builder) {
    if (isNestedInGraphParallelRegion(loop))
      return loop.emitOpError(
          "cannot extract a recurrence nested in scf.forall/scf.parallel "
          "without a selected graph-owned P[] representation");
    if (::mlir::failed(checkSelectedGraphLoweringLeaves(loop)))
      return ::mlir::failure();

    ::mlir::Location loc = loop.getLoc();
    auto stepKind = ::loom::lowering::inferStreamStepKind(loop);
    if (::mlir::failed(stepKind))
      return loop.emitOpError("has invalid 'loom.stream_step_kind'");
    auto predicate = ::loom::lowering::inferStreamPredicate(loop);
    if (::mlir::failed(predicate))
      return loop.emitOpError("has invalid 'loom.stream_predicate'");

    // Inputs to the graph callable after the leading start argument:
    //   lb, ub, step,
    //   captures (external uses besides iv / iter_args / lb / ub / step),
    //   initial iter_arg values.
    ::llvm::SmallVector<::mlir::Operation *, 4> epilogueOps;
    collectEpilogueOps(loop, epilogueOps);

    ::llvm::SetVector<::mlir::Value> captures;
    collectExternalUses(loop, captures);
    collectAdditionalCaptures(loop, epilogueOps, captures);
    // Don't pass lb/ub/step twice: drop them from captures since we
    // forward them explicitly as separate body operands.
    captures.remove(loop.getLowerBound());
    captures.remove(loop.getUpperBound());
    captures.remove(loop.getStep());

    ::llvm::SmallVector<::mlir::Value, 4> initArgs(loop.getInitArgs().begin(),
                                                   loop.getInitArgs().end());
    for (::mlir::Value init : initArgs) {
      if (::dataflow::DataflowDialect::isMemoryCapabilityType(init.getType()))
        return loop.emitOpError()
               << "cannot extract loop-carried memory capability "
               << init.getType()
               << "; project the recurrence to an explicit index domain";
    }
    ::llvm::SmallVector<::mlir::Value, 4> outputValues;
    computeGraphOutputs(loop, epilogueOps, outputValues);

    ::llvm::SmallVector<::mlir::Value, 8> inputValues{
        loop.getLowerBound(), loop.getUpperBound(), loop.getStep()};
    inputValues.append(captures.begin(), captures.end());
    inputValues.append(initArgs.begin(), initArgs.end());
    ClassifiedGraphValues graphInputs = classifyGraphValues(inputValues);
    ClassifiedGraphValues graphResults = classifyGraphValues(outputValues);
    ::llvm::SmallVector<::mlir::Value, 8> orderedInputs = graphInputs.ordered();
    ::llvm::SmallVector<::mlir::Type, 8> valueResultTypes;
    for (::mlir::Value value : graphResults.values)
      valueResultTypes.push_back(value.getType());
    ::llvm::SmallVector<::mlir::Type, 4> memoryResultTypes;
    for (::mlir::Value value : graphResults.memories)
      memoryResultTypes.push_back(value.getType());

    // Unique sym.
    std::string symStem = (stem + "_" + ::llvm::Twine(seq)).str();
    std::string symName = uniqueSymbol(module, symStem);

    builder.setInsertionPoint(loop);
    auto spatial = ::loom::SpatialRegionOp::create(
        builder, loc, graphInputs.values, ::mlir::ValueRange{},
        graphInputs.memories, ::mlir::ValueRange{}, valueResultTypes,
        memoryResultTypes, builder.getArrayAttr({}),
        builder.getStringAttr(symName));

    ::mlir::Block *entry = builder.createBlock(&spatial.getBody());
    for (::mlir::Value input : orderedInputs)
      entry->addArgument(input.getType(), loc);

    // Map captured / iter_arg / lb-ub-step to entry block args.
    ::llvm::DenseMap<::mlir::Value, ::mlir::BlockArgument> boundaryArgs;
    for (auto [i, value] : ::llvm::enumerate(orderedInputs))
      boundaryArgs[value] = entry->getArgument(i);
    ::mlir::BlockArgument lbArg = boundaryArgs.lookup(loop.getLowerBound());
    ::mlir::BlockArgument ubArg = boundaryArgs.lookup(loop.getUpperBound());
    ::mlir::BlockArgument stepArg = boundaryArgs.lookup(loop.getStep());
    ::llvm::SmallVector<::mlir::BlockArgument, 4> captureArgs;
    for (::mlir::Value capture : captures)
      captureArgs.push_back(boundaryArgs.lookup(capture));
    ::llvm::SmallVector<::mlir::Value, 4> initArgVals;
    for (::mlir::Value init : initArgs)
      initArgVals.push_back(boundaryArgs.lookup(init));

    // Build the new scf.for with the graph-side iter_arg seeds.
    builder.setInsertionPointToEnd(entry);
    auto newLoop = ::mlir::scf::ForOp::create(builder, loc, lbArg, ubArg,
                                              stepArg, initArgVals,
                                              /*bodyBuilder=*/nullptr);
    ::loom::lowering::setStreamLoopConfiguration(builder, newLoop, *stepKind,
                                                 *predicate);

    // Move the original loop body into the new loop, remapping ivs +
    // captures.
    ::mlir::Block &origBody = loop.getRegion().front();
    ::mlir::Block &newBody = newLoop.getRegion().front();
    // The new body already has [iv, iter_args*] block args from
    // ForOp::create above. We replace the empty body with a clone of
    // the original ops.
    ::mlir::IRMapping mapping;
    mapping.map(loop.getLowerBound(), lbArg);
    mapping.map(loop.getUpperBound(), ubArg);
    mapping.map(loop.getStep(), stepArg);
    mapping.map(loop.getInductionVar(), newBody.getArgument(0));
    for (size_t i = 0; i < initArgs.size(); ++i) {
      mapping.map(loop.getRegionIterArgs()[i], newBody.getArgument(1 + i));
    }
    for (auto [i, result] : ::llvm::enumerate(loop.getResults()))
      mapping.map(result, newLoop.getResult(i));
    for (auto [i, captured] : ::llvm::enumerate(captures))
      mapping.map(captured, captureArgs[i]);

    builder.setInsertionPointToEnd(&newBody);
    for (::mlir::Operation &op : origBody)
      builder.clone(op, mapping);

    builder.setInsertionPointToEnd(entry);
    for (::mlir::Operation *op : epilogueOps)
      builder.clone(*op, mapping);

    builder.setInsertionPointToEnd(entry);
    ::llvm::SmallVector<::mlir::Value, 4> yieldValues;
    for (::mlir::Value value : graphResults.values)
      yieldValues.push_back(mapping.lookup(value));
    ::llvm::SmallVector<::mlir::Value, 4> yieldMemories;
    for (::mlir::Value value : graphResults.memories)
      yieldMemories.push_back(mapping.lookup(value));
    ::loom::SpatialYieldOp::create(builder, loc, yieldValues, yieldMemories);

    for (auto [i, value] : ::llvm::enumerate(graphResults.values))
      value.replaceAllUsesWith(spatial.getValueResults()[i]);
    for (auto [i, value] : ::llvm::enumerate(graphResults.memories))
      value.replaceAllUsesWith(spatial.getMemoryResults()[i]);
    for (::mlir::Operation *op : ::llvm::reverse(epilogueOps))
      op->erase();
    loop.erase();
    return ::mlir::success();
  }

  ::mlir::LogicalResult publishSpatialRegions(::mlir::ModuleOp module,
                                              ::mlir::OpBuilder &builder) {
    ::llvm::SmallVector<::loom::SpatialRegionOp, 8> regions;
    module.walk([&](::loom::SpatialRegionOp spatial) {
      regions.push_back(spatial);
    });

    for (::loom::SpatialRegionOp spatial : regions) {
      auto thread = spatial->getParentOfType<::dataflow::ThreadOp>();
      if (!thread)
        return spatial.emitOpError(
            "expected staged candidate inside dataflow.thread");
      for (::mlir::Operation *parent = spatial->getParentOp();
           parent && parent != thread.getOperation();
           parent = parent->getParentOp()) {
        if (!::llvm::isa<::mlir::scf::IfOp>(parent))
          return spatial.emitOpError(
              "completion propagation through enclosing '")
                 << parent->getName()
                 << "' is not implemented; spatial candidate cannot be "
                    "published";
      }
    }

    for (::loom::SpatialRegionOp spatial : regions) {

      auto thread = spatial->getParentOfType<::dataflow::ThreadOp>();
      if (!thread)
        return spatial.emitOpError(
            "expected staged candidate inside dataflow.thread");
      ::mlir::Block &threadEntry = thread.getBody().front();
      size_t ctrlIndex = thread.getFunctionType().getInputs().size();
      if (threadEntry.getNumArguments() <= ctrlIndex)
        return thread.emitOpError("is missing thread control block argument");
      ::mlir::Value threadCtrl = threadEntry.getArgument(ctrlIndex);

      ClassifiedGraphValues graphInputs;
      graphInputs.values.append(spatial.getValueInputs().begin(),
                                spatial.getValueInputs().end());
      graphInputs.memories.append(spatial.getMemoryInputs().begin(),
                                  spatial.getMemoryInputs().end());
      ClassifiedGraphValues graphResults;
      graphResults.values.append(spatial.getValueResults().begin(),
                                 spatial.getValueResults().end());
      graphResults.memories.append(spatial.getMemoryResults().begin(),
                                   spatial.getMemoryResults().end());

      ::llvm::SmallVector<::mlir::Type, 8> inputTypes;
      for (::mlir::Value value : graphInputs.values)
        inputTypes.push_back(value.getType());
      for (::mlir::Value channel : spatial.getStreamInputs())
        inputTypes.push_back(
            ::llvm::cast<::dataflow::ChannelType>(channel.getType())
                .getElementType());
      for (::mlir::Value memory : graphInputs.memories)
        inputTypes.push_back(memory.getType());
      ::llvm::SmallVector<::mlir::Type, 4> resultTypes;
      for (::mlir::Value value : graphResults.values)
        resultTypes.push_back(value.getType());
      for (::mlir::Value channel : spatial.getStreamOutputs())
        resultTypes.push_back(
            ::llvm::cast<::dataflow::ChannelType>(channel.getType())
                .getElementType());
      for (::mlir::Value memory : graphResults.memories)
        resultTypes.push_back(memory.getType());
      auto functionType = builder.getFunctionType(inputTypes, resultTypes);
      auto segmentAttrs = graphSegmentAttrs(
          builder, graphInputs, graphResults,
          static_cast<int32_t>(spatial.getStreamInputs().size()),
          static_cast<int32_t>(spatial.getStreamOutputs().size()));

      std::string graphName = uniqueSymbol(
          module, spatial.getGraphName().value_or("g_spatial_candidate"));
      ::mlir::Location loc = spatial.getLoc();
      builder.setInsertionPointToEnd(module.getBody());
      auto graph = ::dataflow::GraphOp::create(
          builder, loc, graphName, functionType, segmentAttrs);
      graph.setSymVisibilityAttr(builder.getStringAttr("private"));

      ::llvm::SmallVector<::mlir::Value, 8> memoryRoots(
          graphInputs.memories.size());
      ::llvm::DenseMap<::mlir::Value, unsigned> capturedRootCounts;
      bool hasUnknownMemoryRoot = false;
      for (auto [index, memory] : ::llvm::enumerate(graphInputs.memories)) {
        ::mlir::Value root =
            findGraphPublicationMemoryRoot(memory, threadEntry);
        memoryRoots[index] = root;
        if (root)
          ++capturedRootCounts[root];
        else
          hasUnknownMemoryRoot = true;
      }

      auto getThreadArgAttrs = [&](::mlir::Value value) {
        if (!value)
          return ::mlir::DictionaryAttr{};
        auto argument = ::llvm::dyn_cast<::mlir::BlockArgument>(value);
        if (!argument || argument.getOwner() != &threadEntry ||
            argument.getArgNumber() >= thread.getFunctionType().getNumInputs())
          return ::mlir::DictionaryAttr{};
        return ::mlir::function_interface_impl::getArgAttrDict(
            thread, argument.getArgNumber());
      };

      ::llvm::SmallVector<::mlir::DictionaryAttr, 8> graphArgAttrs;
      graphArgAttrs.reserve(inputTypes.size());
      for (::mlir::Value input : graphInputs.values) {
        ::mlir::NamedAttrList attrs(getThreadArgAttrs(input));
        graphArgAttrs.push_back(attrs.getDictionary(builder.getContext()));
      }
      for ([[maybe_unused]] ::mlir::Value channel : spatial.getStreamInputs())
        graphArgAttrs.push_back(builder.getDictionaryAttr({}));
      for (auto [index, memory] : ::llvm::enumerate(graphInputs.memories)) {
        ::mlir::NamedAttrList attrs(getThreadArgAttrs(memory));
        ::mlir::Value root = memoryRoots[index];
        ::mlir::DictionaryAttr rootAttrs = getThreadArgAttrs(root);
        ::mlir::Attribute noAlias =
            rootAttrs ? rootAttrs.get("llvm.noalias") : ::mlir::Attribute{};
        bool uniqueKnownRoot = !hasUnknownMemoryRoot && root &&
                               capturedRootCounts.lookup(root) == 1;
        if (uniqueKnownRoot && noAlias)
          attrs.set("llvm.noalias", noAlias);
        else
          attrs.erase("llvm.noalias");
        graphArgAttrs.push_back(attrs.getDictionary(builder.getContext()));
      }
      ::mlir::function_interface_impl::setAllArgAttrDicts(graph, graphArgAttrs);

      ::mlir::Block *graphEntry = builder.createBlock(&graph.getBody());
      graphEntry->addArgument(builder.getNoneType(), loc);
      for (::mlir::Type type : inputTypes)
        graphEntry->addArgument(type, loc);

      ::mlir::IRMapping mapping;
      ::mlir::Block &spatialEntry = spatial.getBody().front();
      size_t spatialArgument = 0;
      size_t graphArgument = 1;
      for ([[maybe_unused]] ::mlir::Value input : spatial.getValueInputs())
        mapping.map(spatialEntry.getArgument(spatialArgument++),
                    graphEntry->getArgument(graphArgument++));
      graphArgument += spatial.getStreamInputs().size();
      size_t inputChannelArgument = graphEntry->getNumArguments();
      for (::mlir::Value channel : spatial.getStreamInputs()) {
        graphEntry->addArgument(channel.getType(), loc);
        mapping.map(spatialEntry.getArgument(spatialArgument++),
                    graphEntry->getArgument(inputChannelArgument++));
      }
      for ([[maybe_unused]] ::mlir::Value memory : spatial.getMemoryInputs())
        mapping.map(spatialEntry.getArgument(spatialArgument++),
                    graphEntry->getArgument(graphArgument++));
      size_t outputChannelArgument = graphEntry->getNumArguments();
      for (::mlir::Value channel : spatial.getStreamOutputs()) {
        graphEntry->addArgument(channel.getType(), loc);
        mapping.map(spatialEntry.getArgument(spatialArgument++),
                    graphEntry->getArgument(outputChannelArgument++));
      }

      builder.setInsertionPointToEnd(graphEntry);
      for (::mlir::Operation &op : spatialEntry.without_terminator())
        builder.clone(op, mapping);
      auto spatialYield =
          ::mlir::cast<::loom::SpatialYieldOp>(spatialEntry.getTerminator());
      ::llvm::SmallVector<::mlir::Value, 4> returnValues;
      for (::mlir::Value value : spatialYield.getValues())
        returnValues.push_back(mapping.lookup(value));
      ::llvm::SmallVector<::mlir::Value, 4> returnMemories;
      for (::mlir::Value value : spatialYield.getMemories())
        returnMemories.push_back(mapping.lookup(value));
      ::dataflow::GraphReturnOp::create(
          builder, loc, returnValues, ::mlir::ValueRange{}, returnMemories,
          ::mlir::ValueRange{graphEntry->getArgument(0)});

      builder.setInsertionPoint(spatial);
      auto callee =
          ::mlir::FlatSymbolRefAttr::get(builder.getContext(), graphName);
      ::llvm::SmallVector<::mlir::Type, 4> valueResultTypes;
      for (::mlir::Value result : spatial.getValueResults())
        valueResultTypes.push_back(result.getType());
      ::llvm::SmallVector<::mlir::Type, 4> memoryResultTypes;
      for (::mlir::Value result : spatial.getMemoryResults())
        memoryResultTypes.push_back(result.getType());
      auto launch = ::dataflow::GraphLaunchOp::create(
          builder, loc, valueResultTypes, memoryResultTypes,
          builder.getNoneType(), callee, spatial.getSourceMaps(),
          ::mlir::ValueRange{threadCtrl}, spatial.getValueInputs(),
          spatial.getStreamInputs(), spatial.getMemoryInputs(),
          spatial.getStreamOutputs());
      auto propagated = propagateCompletionToThread(
          thread, launch.getDone(), threadCtrl, builder);
      if (::mlir::failed(propagated))
        return launch.emitOpError(
            "failed to propagate completion through enclosing structured "
            "control");
      addThreadCompletionFrontier(thread, *propagated);
      for (auto [index, result] :
           ::llvm::enumerate(spatial.getValueResults()))
        result.replaceAllUsesWith(launch.getValueResults()[index]);
      for (auto [index, result] :
           ::llvm::enumerate(spatial.getMemoryResults()))
        result.replaceAllUsesWith(launch.getMemoryResults()[index]);
      spatial.erase();
    }
    return ::mlir::success();
  }

  bool hasUnsupportedStructuredGraphNestedOp(::mlir::Operation *root) {
    bool unsupported = false;
    root->walk([&](::mlir::Operation *nested) -> ::mlir::WalkResult {
      if (nested == root)
        return ::mlir::WalkResult::advance();
      if (::llvm::isa<::dataflow::GraphLaunchOp, ::dataflow::ThreadLaunchOp,
                      ::dataflow::GraphOp, ::dataflow::ThreadOp>(nested)) {
        unsupported = true;
        return ::mlir::WalkResult::interrupt();
      }
      if (auto parallel = ::llvm::dyn_cast<::mlir::scf::ParallelOp>(nested)) {
        if (isEffectFormParallelGraphCandidate(parallel))
          return ::mlir::WalkResult::advance();
        unsupported = true;
        return ::mlir::WalkResult::interrupt();
      }
      if (auto forall = ::llvm::dyn_cast<::mlir::scf::ForallOp>(nested)) {
        if (isEffectFormForallGraphCandidate(forall))
          return ::mlir::WalkResult::advance();
        unsupported = true;
        return ::mlir::WalkResult::interrupt();
      }
      if (auto inParallel =
              ::llvm::dyn_cast<::mlir::scf::InParallelOp>(nested)) {
        if (!inParallel.getRegion().empty() &&
            inParallel.getRegion().front().empty())
          return ::mlir::WalkResult::advance();
        unsupported = true;
        return ::mlir::WalkResult::interrupt();
      }
      if (nested->getNumRegions() != 0 &&
          !::llvm::isa<::mlir::scf::IfOp, ::mlir::scf::ForOp,
                       ::mlir::scf::WhileOp, ::mlir::scf::IndexSwitchOp>(
              nested)) {
        unsupported = true;
        return ::mlir::WalkResult::interrupt();
      }
      return ::mlir::WalkResult::advance();
    });
    return unsupported;
  }

  bool isStraightLineGraphCandidate(::dataflow::ThreadOp thread) {
    ::mlir::Region &body = thread.getBody();
    if (!body.hasOneBlock())
      return false;
    ::mlir::Block &entry = body.front();
    bool hasBodyOp = false;
    for (::mlir::Operation &op : entry.without_terminator()) {
      hasBodyOp = true;
      if (::llvm::isa<::loom::SpatialRegionOp, ::dataflow::GraphLaunchOp,
                      ::dataflow::ThreadLaunchOp>(op))
        return false;
      if (::llvm::isa<::dataflow::ChannelSendOp, ::dataflow::ChannelReceiveOp>(
              op))
        return false;
      for (::mlir::Value result : op.getResults())
        if (::llvm::any_of(result.getUses(), [&](::mlir::OpOperand &use) {
              return use.getOwner()->getBlock() == &entry &&
                     use.getOwner() == entry.getTerminator();
            }))
          return false;
      if (::llvm::isa<::mlir::scf::ForOp>(op))
        return false;
      if (auto forall = ::llvm::dyn_cast<::mlir::scf::ForallOp>(op)) {
        if (!isEffectFormForallGraphCandidate(forall))
          return false;
        continue;
      }
      if (auto parallel = ::llvm::dyn_cast<::mlir::scf::ParallelOp>(op)) {
        if (!isEffectFormParallelGraphCandidate(parallel))
          return false;
        continue;
      }
      if (::llvm::isa<::mlir::scf::IfOp, ::mlir::scf::WhileOp>(op)) {
        if (hasUnsupportedStructuredGraphNestedOp(&op))
          return false;
        continue;
      }
      if (::llvm::isa<::mlir::scf::IndexSwitchOp>(op)) {
        if (hasUnsupportedStructuredGraphNestedOp(&op))
          return false;
        continue;
      }
      if (op.getNumRegions() != 0)
        return false;
      if (op.hasTrait<::mlir::OpTrait::SymbolTable>())
        return false;
      if (::llvm::isa<::mlir::FunctionOpInterface>(op))
        return false;
    }
    return hasBodyOp;
  }

  bool isEffectFormForallGraphCandidate(::mlir::scf::ForallOp forall) {
    if (!forall.getOutputs().empty() || forall.getNumResults() != 0)
      return false;
    auto inParallel = forall.getTerminator();
    if (inParallel.getRegion().empty() ||
        !inParallel.getRegion().front().empty())
      return false;

    return isEffectFormStructuredBodyCandidate(forall.getBody());
  }

  bool isEffectFormParallelGraphCandidate(::mlir::scf::ParallelOp parallel) {
    if (!parallel.getInitVals().empty() || parallel.getNumResults() != 0)
      return false;
    return isEffectFormStructuredBodyCandidate(&parallel.getRegion().front());
  }

  bool isEffectFormStructuredRegionCandidate(::mlir::Region &region) {
    if (region.empty())
      return true;
    if (!region.hasOneBlock())
      return false;
    return isEffectFormStructuredBodyCandidate(&region.front());
  }

  bool isEffectFormStructuredBodyCandidate(::mlir::Block *body) {
    if (!body)
      return true;
    for (::mlir::Operation &nested : body->without_terminator()) {
      if (::llvm::isa<::dataflow::GraphLaunchOp, ::dataflow::ThreadLaunchOp,
                      ::dataflow::GraphOp, ::dataflow::ThreadOp>(&nested))
        return false;
      if (::llvm::isa<::mlir::FunctionOpInterface, ::mlir::CallOpInterface>(
              &nested))
        return false;
      if (auto forOp = ::llvm::dyn_cast<::mlir::scf::ForOp>(&nested)) {
        if (!isEffectFormStructuredBodyCandidate(forOp.getBody()))
          return false;
        continue;
      }
      if (auto whileOp = ::llvm::dyn_cast<::mlir::scf::WhileOp>(&nested)) {
        if (!isEffectFormStructuredRegionCandidate(whileOp.getBefore()) ||
            !isEffectFormStructuredRegionCandidate(whileOp.getAfter()))
          return false;
        continue;
      }
      if (auto parallel = ::llvm::dyn_cast<::mlir::scf::ParallelOp>(&nested)) {
        if (!isEffectFormParallelGraphCandidate(parallel))
          return false;
        continue;
      }
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

  ::mlir::LogicalResult stageStraightLineThreads(::mlir::ModuleOp module,
                                                 ::mlir::OpBuilder &builder) {
    ::llvm::SmallVector<::dataflow::ThreadOp, 8> threads;
    for (::dataflow::ThreadOp thread : module.getOps<::dataflow::ThreadOp>()) {
      if (!isStraightLineGraphCandidate(thread))
        continue;
      if (::mlir::failed(checkSelectedGraphLoweringLeaves(thread)))
        return ::mlir::failure();
      threads.push_back(thread);
    }

    for (::dataflow::ThreadOp thread : threads) {
      if (failed(stageStraightLineThread(module, thread, builder)))
        return ::mlir::failure();
    }
    return ::mlir::success();
  }

  ::mlir::LogicalResult stageStraightLineThread(::mlir::ModuleOp module,
                                                ::dataflow::ThreadOp thread,
                                                ::mlir::OpBuilder &builder) {
    ::mlir::Location loc = thread.getLoc();
    ::mlir::FunctionType threadType = thread.getFunctionType();
    ::mlir::Region &threadBody = thread.getBody();
    ::mlir::Block &threadEntry = threadBody.front();

    size_t threadInputCount = threadType.getInputs().size();
    if (threadEntry.getNumArguments() <= threadInputCount)
      return thread.emitOpError("is missing thread control block argument");
    ::mlir::Value threadCtrl = threadEntry.getArgument(threadInputCount);
    if (!::llvm::isa<::mlir::NoneType>(threadCtrl.getType()))
      return thread.emitOpError("thread control block argument must be none");

    ::llvm::SmallVector<::mlir::Operation *, 8> bodyOps;
    for (::mlir::Operation &op : threadEntry.without_terminator())
      bodyOps.push_back(&op);
    for (::mlir::OpOperand &use : threadCtrl.getUses()) {
      if (use.getOwner() != threadEntry.getTerminator())
        return use.getOwner()->emitOpError(
            "structured candidate must not consume thread control before "
            "graph publication");
    }

    ::llvm::SmallVector<::mlir::Value, 8> inputValues;
    for (size_t i = 0; i < threadInputCount; ++i)
      inputValues.push_back(threadEntry.getArgument(i));
    for (size_t i = threadInputCount + 1, e = threadEntry.getNumArguments();
         i < e; ++i)
      inputValues.push_back(threadEntry.getArgument(i));
    ClassifiedGraphValues graphInputs = classifyGraphValues(inputValues);
    ::llvm::SmallVector<::mlir::Value, 8> orderedInputs = graphInputs.ordered();
    ::llvm::SmallVector<::mlir::Type, 8> inputTypes;
    for (::mlir::Value value : orderedInputs)
      inputTypes.push_back(value.getType());

    std::string stem = "g_" + sanitizeSymbol(thread.getSymName()) + "_0";
    std::string graphName = uniqueSymbol(module, stem);

    builder.setInsertionPoint(&threadEntry, threadEntry.begin());
    auto spatial = ::loom::SpatialRegionOp::create(
        builder, loc, graphInputs.values, ::mlir::ValueRange{},
        graphInputs.memories, ::mlir::ValueRange{}, ::mlir::TypeRange{},
        ::mlir::TypeRange{}, builder.getArrayAttr({}),
        builder.getStringAttr(graphName));

    ::mlir::Block *spatialEntry = builder.createBlock(&spatial.getBody());
    for (::mlir::Type type : inputTypes)
      spatialEntry->addArgument(type, loc);

    ::mlir::IRMapping mapping;
    for (auto [i, value] : ::llvm::enumerate(orderedInputs))
      mapping.map(value, spatialEntry->getArgument(i));

    builder.setInsertionPointToEnd(spatialEntry);
    for (::mlir::Operation *op : bodyOps)
      builder.clone(*op, mapping);
    ::loom::SpatialYieldOp::create(builder, loc, ::mlir::ValueRange{},
                                   ::mlir::ValueRange{});

    for (::mlir::Operation *op : ::llvm::reverse(bodyOps))
      op->erase();
    return ::mlir::success();
  }
};

} // namespace

namespace loom {
namespace lowering {

std::unique_ptr<::mlir::Pass> createLowerForToGraphPass() {
  return std::make_unique<LowerForToGraphPass>();
}

void registerLowerForToGraphPass() {
  static bool once = []() {
    ::mlir::PassRegistration<LowerForToGraphPass>();
    return true;
  }();
  (void)once;
}

} // namespace lowering
} // namespace loom
