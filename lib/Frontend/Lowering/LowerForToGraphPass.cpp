// Lower scf.for ops with iter_args (i.e., structured reductions /
// loop-carried recurrences) found inside dataflow.thread bodies into
// `dataflow.graph.func` symbol definitions plus matching
// `dataflow.graph.launch` ops.
//
// scf.for ops without iter_args are left in place. Straight-line
// dataflow.thread bodies are also extracted into graph.func bodies so
// element-wise kernels have an explicit SpatialCore graph surface. The
// graph function_type contains only normalized application payload ports.
// Start/done remain explicit launch protocol endpoints, and graph.return owns
// the segmented payload boundary plus retirement frontier.

#include "Frontend/Lowering/Passes.h"
#include "Frontend/Lowering/StreamLoopAttrs.h"

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
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"

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

bool causallyDependsOn(::mlir::Value event, ::mlir::Value prerequisite,
                       ::llvm::DenseSet<::mlir::Value> &visited) {
  if (event == prerequisite)
    return true;
  if (!event || !visited.insert(event).second)
    return false;
  ::mlir::Operation *definition = event.getDefiningOp();
  if (!definition)
    return false;
  return ::llvm::any_of(definition->getOperands(),
                        [&](::mlir::Value operand) {
                          ::llvm::DenseSet<::mlir::Value> branchVisited =
                              visited;
                          return causallyDependsOn(operand, prerequisite,
                                                   branchVisited);
                        });
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

  ::llvm::SmallVector<::mlir::Value, 4> frontier;
  for (unsigned i = 0; i < candidates.size(); ++i) {
    bool covered = false;
    for (unsigned j = 0; j < candidates.size(); ++j) {
      if (i == j)
        continue;
      ::llvm::DenseSet<::mlir::Value> visited;
      if (causallyDependsOn(candidates[j], candidates[i], visited)) {
        covered = true;
        break;
      }
    }
    if (!covered)
      frontier.push_back(candidates[i]);
  }
  yield.getCompletionFrontierMutable().assign(frontier);
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
  ::llvm::StringRef name = op->getName().getStringRef();
  if (name != "arith.addf" && name != "arith.subf" && name != "arith.mulf" &&
      name != "arith.addi" && name != "arith.muli" && name != "arith.andi" &&
      name != "arith.ori" && name != "arith.shli" && name != "arith.shrui" &&
      name != "arith.index_cast" && name != "llvm.zext" &&
      name != "llvm.intr.abs" && name != "llvm.intr.bswap" &&
      name != "llvm.intr.fabs" && name != "llvm.intr.fmuladd")
    return false;
  if (auto effects = ::llvm::dyn_cast<::mlir::MemoryEffectOpInterface>(op))
    return effects.hasNoEffect();
  return ::mlir::isPure(op);
}

bool isLlvmPointerType(::mlir::Type type) {
  return ::llvm::isa<::mlir::LLVM::LLVMPointerType>(type);
}

bool isGraphMemoryPortType(::mlir::Type type) {
  return ::llvm::isa<::mlir::MemRefType, ::mlir::UnrankedMemRefType,
                     ::mlir::LLVM::LLVMPointerType>(type);
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

ClassifiedGraphValues classifyGraphValues(::mlir::ValueRange values) {
  ClassifiedGraphValues classified;
  for (::mlir::Value value : values) {
    if (isGraphMemoryPortType(value.getType()))
      classified.memories.push_back(value);
    else
      classified.values.push_back(value);
  }
  return classified;
}

::llvm::SmallVector<::mlir::NamedAttribute, 2>
graphSegmentAttrs(::mlir::OpBuilder &builder,
                  const ClassifiedGraphValues &inputs,
                  const ClassifiedGraphValues &results) {
  return {
      builder.getNamedAttr("input_segments",
                           builder.getDenseI32ArrayAttr(inputs.segments())),
      builder.getNamedAttr("result_segments",
                           builder.getDenseI32ArrayAttr(results.segments())),
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
    if (!isLlvmPointerType(value.getType()) || usedOutsideEpilogue(value))
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
           "into dataflow.graph.func definitions plus dataflow.graph.launch "
           "ops.";
  }

  void getDependentDialects(::mlir::DialectRegistry &registry) const final {
    registry.insert<::mlir::arith::ArithDialect, ::mlir::func::FuncDialect,
                    ::mlir::scf::SCFDialect,
                    ::dataflow::DataflowDialect>();
  }

  void runOnOperation() final {
    ::mlir::ModuleOp module = getOperation();
    ::mlir::MLIRContext *ctx = &getContext();
    ::mlir::OpBuilder builder(ctx);

    promoteStandaloneMemcpyFunctions(module, builder);
    promoteStandaloneScalarReturnFunctions(module, builder);
    promoteStandaloneStructuredStatusOutParamFunctions(module, builder);
    promoteStandaloneStructuredOutParamFunctions(module, builder);
    promoteStandaloneStructuredFunctions(module, builder);

    // For each dataflow.thread body, snapshot the eligible
    // scf.for ops in source order before mutating.
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
        // Skip loops already owned by a graph boundary.
        if (loop->getParentOfType<::dataflow::GraphFuncOp>())
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
        if (failed(promoteOne(module, loop, stem, seq, builder)))
          return signalPassFailure();
      }
    }

    if (failed(promoteStraightLineThreads(module, builder)))
      return signalPassFailure();
  }

  bool isSideEffectFreeSetupOp(::mlir::Operation &op) {
    if (isSupportedArmInlineAsm(&op))
      return true;
    if (op.getNumRegions() != 0 || op.getNumSuccessors() != 0)
      return false;
    if (op.hasTrait<::mlir::OpTrait::IsTerminator>())
      return false;
    if (op.hasTrait<::mlir::OpTrait::SymbolTable>())
      return false;
    if (::llvm::isa<::mlir::FunctionOpInterface, ::mlir::CallOpInterface>(op))
      return false;
    if (auto effects =
            ::llvm::dyn_cast<::mlir::MemoryEffectOpInterface>(&op))
      return effects.hasNoEffect();
    return ::mlir::isPure(&op);
  }

  bool isSupportedArmInlineAsm(::mlir::Operation *op) {
    if (op->getName().getStringRef() != "llvm.inline_asm")
      return false;
    auto asmString = op->getAttrOfType<::mlir::StringAttr>("asm_string");
    if (!asmString)
      return false;
    ::llvm::StringRef text = asmString.getValue();
    return text == "pkhbt $0, $1, $2, lsl $3" ||
           text == "pkhtb $0, $1, $2, asr $3" ||
           text == "sxtab16 $0, $1, $2" || text == "sxtb16 $0, $1";
  }

  bool containsSupportedArmInlineAsm(::mlir::func::FuncOp func) {
    bool found = false;
    func.walk([&](::mlir::Operation *op) {
      found |= isSupportedArmInlineAsm(op);
    });
    return found;
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

  bool isBlockedStandaloneStructuredSetupOp(::mlir::Operation *op) {
    ::llvm::StringRef name = op->getName().getStringRef();
    return name == "arith.divsi" || name == "arith.divui" ||
           name == "arith.remf" || name == "arith.remui" ||
           name == "llvm.fptosi" || name == "llvm.sitofp";
  }

  bool isBlockedStandaloneStructuredBodyOp(::mlir::Operation *op) {
    ::llvm::StringRef name = op->getName().getStringRef();
    return name == "arith.divsi" || name == "arith.remf" ||
           name == "arith.remui" || name == "llvm.fptosi" ||
           name == "llvm.sitofp";
  }

  bool isMemsetIntrinsic(::mlir::Operation *op) {
    return op->getName().getStringRef() == "llvm.intr.memset";
  }

  bool isZeroIntegerConstant(::mlir::Value value) {
    if (auto constant = value.getDefiningOp<::mlir::arith::ConstantOp>()) {
      auto intAttr =
          ::llvm::dyn_cast<::mlir::IntegerAttr>(constant.getValue());
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
    return ::llvm::isa<::mlir::LLVM::LLVMPointerType>(dst.getType()) &&
           ::llvm::isa<::mlir::IntegerType>(byteValue.getType()) &&
           ::llvm::isa<::mlir::IntegerType, ::mlir::IndexType>(
               byteCount.getType()) &&
           isZeroIntegerConstant(byteValue);
  }

  bool hasUnsupportedStandaloneStructuredBody(::mlir::Operation *root) {
    bool unsupported = false;
    root->walk([&](::mlir::Operation *nested) -> ::mlir::WalkResult {
      if (nested == root)
        return ::mlir::WalkResult::advance();
      if (nested->getName().getStringRef() == "llvm.inline_asm" &&
          !isSupportedArmInlineAsm(nested)) {
        unsupported = true;
        return ::mlir::WalkResult::interrupt();
      }
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
                       ::mlir::scf::IfOp, ::mlir::scf::IndexSwitchOp>(
              nested)) {
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
    if (::llvm::isa<::mlir::CallOpInterface>(op) &&
        !::llvm::isa<::mlir::LLVM::CallIntrinsicOp>(op))
      return false;
    if (::llvm::isa<::mlir::LLVM::MemcpyOp, ::mlir::LLVM::StoreOp>(op))
      return false;
    if (op->getNumRegions() != 0 || op->getNumSuccessors() != 0)
      return false;
    if (op->hasTrait<::mlir::OpTrait::SymbolTable>())
      return false;
    if (auto effects =
            ::llvm::dyn_cast<::mlir::MemoryEffectOpInterface>(op)) {
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
    auto returnOp =
        ::llvm::dyn_cast<::mlir::func::ReturnOp>(
            func.getBody().front().getTerminator());
    if (!returnOp ||
        returnOp.getNumOperands() != func.getFunctionType().getNumResults())
      return false;

    bool hasLoad = false;
    bool unsupported = false;
    func.getBody().walk([&](::mlir::Operation *nested)
                            -> ::mlir::WalkResult {
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
    auto returnOp = ::llvm::cast<::mlir::func::ReturnOp>(
        funcEntry.getTerminator());
    ClassifiedGraphValues graphInputs =
        classifyGraphValues(funcEntry.getArguments());
    ClassifiedGraphValues graphResults =
        classifyGraphValues(returnOp.getOperands());
    ::llvm::SmallVector<::mlir::Value, 8> orderedInputs =
        graphInputs.ordered();
    ::llvm::SmallVector<::mlir::Value, 8> orderedResults =
        graphResults.ordered();
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
    auto graph = ::dataflow::GraphFuncOp::create(
        builder, loc, graphName, graphType, segmentAttrs);
    graph.setSymVisibilityAttr(builder.getStringAttr("private"));

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
        builder, loc, returnValues, ::mlir::ValueRange{},
        returnMemories,
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
    return arg && arg.getOwner() == &entry && isLlvmPointerType(arg.getType());
  }

  bool isScalarNonPointerType(::mlir::Type type) {
    return ::llvm::isa<::mlir::IntegerType, ::mlir::FloatType,
                       ::mlir::IndexType>(type) &&
           !isLlvmPointerType(type);
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
      hasPointerInput |= isLlvmPointerType(ty);
    if (!hasPointerInput)
      return false;

    auto returnOp =
        ::llvm::dyn_cast<::mlir::func::ReturnOp>(entry.getTerminator());
    if (!returnOp || returnOp.getNumOperands() != 1)
      return false;

    bool sawStructuredOp = false;
    bool sawStore = false;
    bool sawBlockedSetupNumericOp = false;
    bool sawBlockedBodyNumericOp = false;
    ::llvm::DenseSet<::mlir::Value> structuredResults;
    for (::mlir::Operation &op : entry.without_terminator()) {
      if (!sawStructuredOp && isBlockedStandaloneStructuredSetupOp(&op))
        sawBlockedSetupNumericOp = true;

      if (isSideEffectFreeSetupOp(op))
        continue;

      if (!isSupportedStandaloneStructuredTopLevelOp(&op) &&
          !isResultBearingStandaloneStructuredTopLevelOp(&op))
        return false;
      if (hasUnsupportedStandaloneStructuredBody(&op))
        return false;
      op.walk([&](::mlir::Operation *nested) {
        if (isBlockedStandaloneStructuredBodyOp(nested))
          sawBlockedBodyNumericOp = true;
        if (::llvm::isa<::mlir::LLVM::StoreOp>(nested))
          sawStore = true;
      });
      for (::mlir::Value result : op.getResults())
        structuredResults.insert(result);
      sawStructuredOp = true;
    }

    return sawStructuredOp && sawStore &&
           (structuredResults.contains(returnOp.getOperand(0)) ||
            isZeroIntegerConstant(returnOp.getOperand(0))) &&
           !sawBlockedSetupNumericOp && !sawBlockedBodyNumericOp;
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

  bool isStandaloneStructuredOutParamFunctionCandidate(
      ::mlir::func::FuncOp func) {
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
    ::mlir::Block &entry = func.getBody().front();

    bool hasPointerInput = false;
    for (::mlir::Type ty : func.getFunctionType().getInputs())
      hasPointerInput |= isLlvmPointerType(ty);
    if (!hasPointerInput)
      return false;

    bool sawStructuredOp = false;
    bool sawStore = false;
    bool sawBlockedSetupNumericOp = false;
    bool sawBlockedBodyNumericOp = false;
    ::llvm::DenseSet<::mlir::Value> structuredResults;
    for (::mlir::Operation &op : entry.without_terminator()) {
      if (!sawStructuredOp && isBlockedStandaloneStructuredSetupOp(&op))
        sawBlockedSetupNumericOp = true;

      if (!sawStructuredOp && isSideEffectFreeSetupOp(op))
        continue;

      if (!sawStructuredOp) {
        if (!isResultBearingStandaloneStructuredTopLevelOp(&op))
          return false;
        if (hasUnsupportedStandaloneStructuredBody(&op))
          return false;
        op.walk([&](::mlir::Operation *nested) {
          if (isBlockedStandaloneStructuredBodyOp(nested))
            sawBlockedBodyNumericOp = true;
        });
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

    return sawStructuredOp && sawStore && !sawBlockedSetupNumericOp &&
           !sawBlockedBodyNumericOp;
  }

  void promoteStandaloneStructuredOutParamFunctions(
      ::mlir::ModuleOp module, ::mlir::OpBuilder &builder) {
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
    if (func.getSymName().starts_with("arm_") &&
        !containsSupportedArmInlineAsm(func))
      return false;
    if (!func.getFunctionType().getResults().empty())
      return false;
    ::mlir::Region &body = func.getBody();
    if (!body.hasOneBlock())
      return false;
    ::mlir::Block &entry = body.front();

    bool sawStructuredOp = false;
    bool sawMemset = false;
    bool sawBlockedSetupNumericOp = false;
    bool sawBlockedBodyNumericOp = false;
    for (::mlir::Operation &op : entry.without_terminator()) {
      if (isBlockedStandaloneStructuredSetupOp(&op))
        sawBlockedSetupNumericOp = true;
      if (isMemsetIntrinsic(&op)) {
        if (!isSupportedStandaloneStructuredMemsetOp(&op))
          return false;
        sawMemset = true;
        continue;
      }
      if (isSideEffectFreeSetupOp(op))
        continue;
      if (!isSupportedStandaloneStructuredTopLevelOp(&op) &&
          !isResultBearingStandaloneStructuredTopLevelOp(&op))
        return false;
      if (hasUnsupportedStandaloneStructuredBody(&op))
        return false;
      op.walk([&](::mlir::Operation *nested) {
        if (isBlockedStandaloneStructuredBodyOp(nested))
          sawBlockedBodyNumericOp = true;
        if (isMemsetIntrinsic(nested))
          sawMemset = true;
      });
      sawStructuredOp = true;
    }

    return sawStructuredOp &&
           (!sawMemset ||
            (!sawBlockedSetupNumericOp && !sawBlockedBodyNumericOp));
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

  ::mlir::LogicalResult promoteOne(::mlir::ModuleOp module,
                                   ::mlir::scf::ForOp loop,
                                   ::llvm::StringRef stem, size_t seq,
                                   ::mlir::OpBuilder &builder) {
    if (isNestedInGraphParallelRegion(loop))
      return loop.emitOpError(
          "cannot extract a recurrence nested in scf.forall/scf.parallel "
          "without a selected graph-owned P[] representation");

    ::mlir::Location loc = loop.getLoc();
    ::mlir::Type noneType = builder.getType<::mlir::NoneType>();
    auto stepKind = ::loom::lowering::inferStreamStepKind(loop);
    if (::mlir::failed(stepKind))
      return loop.emitOpError("has invalid 'loom.stream_step_kind'");
    auto predicate = ::loom::lowering::inferStreamPredicate(loop);
    if (::mlir::failed(predicate))
      return loop.emitOpError("has invalid 'loom.stream_predicate'");

    // Inputs to the graph callable (after the leading none ctrl_in):
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
      if (isGraphMemoryPortType(init.getType()))
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
    ::llvm::SmallVector<::mlir::Value, 8> orderedInputs =
        graphInputs.ordered();
    ::llvm::SmallVector<::mlir::Value, 8> orderedResults =
        graphResults.ordered();

    ::llvm::SmallVector<::mlir::Type, 8> inputTypes;
    for (::mlir::Value value : orderedInputs)
      inputTypes.push_back(value.getType());
    ::llvm::SmallVector<::mlir::Type, 4> resultTypes;
    for (::mlir::Value value : orderedResults)
      resultTypes.push_back(value.getType());
    ::mlir::FunctionType functionType =
        builder.getFunctionType(inputTypes, resultTypes);
    auto segmentAttrs = graphSegmentAttrs(builder, graphInputs, graphResults);

    // Unique sym.
    std::string symStem = (stem + "_" + ::llvm::Twine(seq)).str();
    std::string symName = uniqueSymbol(module, symStem);

    // Insert the dataflow.graph.func definition at module scope.
    builder.setInsertionPointToEnd(module.getBody());
    auto graphOp = ::dataflow::GraphFuncOp::create(
        builder, loc, symName, functionType, segmentAttrs);
    graphOp.setSymVisibilityAttr(builder.getStringAttr("private"));

    // Build the graph body with the explicit start argument followed by the
    // payload-only FunctionType inputs.
    ::mlir::Region &graphBody = graphOp.getBody();
    ::mlir::Block *entry = builder.createBlock(&graphBody);
    entry->addArgument(noneType, loc);
    for (size_t i = 0, e = inputTypes.size(); i < e; ++i)
      entry->addArgument(inputTypes[i], loc);

    // Map captured / iter_arg / lb-ub-step to entry block args.
    ::llvm::DenseMap<::mlir::Value, ::mlir::BlockArgument> boundaryArgs;
    for (auto [i, value] : ::llvm::enumerate(orderedInputs))
      boundaryArgs[value] = entry->getArgument(i + 1);
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

    // Seed the structural completion segment. Graph-region lowering replaces
    // it with the graph's explicit execution and effect retirement frontier.
    builder.setInsertionPointToEnd(entry);
    ::llvm::SmallVector<::mlir::Value, 4> returnVals;
    for (::mlir::Value value : graphResults.values)
      returnVals.push_back(mapping.lookup(value));
    ::llvm::SmallVector<::mlir::Value, 4> returnMemories;
    for (::mlir::Value value : graphResults.memories)
      returnMemories.push_back(mapping.lookup(value));
    ::dataflow::GraphReturnOp::create(
        builder, loc, returnVals, ::mlir::ValueRange{}, returnMemories,
        ::mlir::ValueRange{entry->getArgument(0)});

    // Materialize the graph.launch at the original loop site inside
    // the thread body. The `ctrl_in : none` operand comes from the
    // enclosing thread's `thread_ctrl` block argument (per spec
    // section 5.4.1: the thread body's entry block lays out
    // `(args_*, thread_ctrl, iv_*)` and root graph launches consume
    // the thread_ctrl as their start signal).
    builder.setInsertionPoint(loop);
    auto enclosingThread = loop->getParentOfType<::dataflow::ThreadOp>();
    if (!enclosingThread)
      return loop.emitOpError(
          "expected graph extraction candidate inside dataflow.thread");
    ::mlir::Block &threadEntry = enclosingThread.getBody().front();
    size_t ctrlIdx = enclosingThread.getFunctionType().getInputs().size();
    // The verifier guarantees this slot exists and is `none`-typed.
    ::mlir::Value ctrlIn = threadEntry.getArgument(ctrlIdx);

    ::llvm::SmallVector<::mlir::Value, 8> launchOperands = orderedInputs;

    ::llvm::SmallVector<::mlir::Type, 4> launchResultTypes;
    for (::mlir::Value value : orderedResults)
      launchResultTypes.push_back(value.getType());

    auto callee = ::mlir::FlatSymbolRefAttr::get(builder.getContext(), symName);
    auto launchOp = ::dataflow::GraphLaunchOp::create(
        builder, loc, /*doneOut=*/noneType, /*results=*/launchResultTypes,
        callee, ctrlIn, launchOperands);
    addThreadCompletionFrontier(enclosingThread, launchOp.getDoneOut());

    // Replace the selected graph outputs with the graph.launch's
    // user-data results (skip leading done_out), then erase the
    // scalar epilogue cloned into the graph.
    for (auto [i, value] : ::llvm::enumerate(orderedResults))
      value.replaceAllUsesWith(launchOp.getResults()[i]);
    for (::mlir::Operation *op : ::llvm::reverse(epilogueOps))
      op->erase();
    loop.erase();
    return ::mlir::success();
  }

  bool hasUnsupportedStructuredGraphNestedOp(::mlir::Operation *root) {
    bool unsupported = false;
    root->walk([&](::mlir::Operation *nested) -> ::mlir::WalkResult {
      if (nested == root)
        return ::mlir::WalkResult::advance();
      if (::llvm::isa<::dataflow::GraphLaunchOp, ::dataflow::ThreadLaunchOp,
                      ::mlir::scf::ForOp>(nested)) {
        unsupported = true;
        return ::mlir::WalkResult::interrupt();
      }
      if (auto forall = ::llvm::dyn_cast<::mlir::scf::ForallOp>(nested)) {
        if (isEffectFormForallGraphCandidate(forall))
          return ::mlir::WalkResult::advance();
        unsupported = true;
        return ::mlir::WalkResult::interrupt();
      }
      if (auto inParallel = ::llvm::dyn_cast<::mlir::scf::InParallelOp>(
              nested)) {
        if (!inParallel.getRegion().empty() &&
            inParallel.getRegion().front().empty())
          return ::mlir::WalkResult::advance();
        unsupported = true;
        return ::mlir::WalkResult::interrupt();
      }
      if (nested->getNumRegions() != 0 &&
          !::llvm::isa<::mlir::scf::IfOp, ::mlir::scf::WhileOp,
                       ::mlir::scf::IndexSwitchOp>(nested)) {
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
      if (::llvm::isa<::dataflow::GraphLaunchOp, ::dataflow::ThreadLaunchOp>(
              op))
        return false;
      if (::llvm::isa<::mlir::scf::ForOp>(op))
        return false;
      if (auto forall = ::llvm::dyn_cast<::mlir::scf::ForallOp>(op)) {
        if (!isEffectFormForallGraphCandidate(forall))
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

  ::mlir::LogicalResult promoteStraightLineThreads(::mlir::ModuleOp module,
                                                   ::mlir::OpBuilder &builder) {
    ::llvm::SmallVector<::dataflow::ThreadOp, 8> threads;
    for (::dataflow::ThreadOp thread : module.getOps<::dataflow::ThreadOp>()) {
      if (isStraightLineGraphCandidate(thread))
        threads.push_back(thread);
    }

    for (::dataflow::ThreadOp thread : threads) {
      if (failed(promoteStraightLineThread(module, thread, builder)))
        return ::mlir::failure();
    }
    return ::mlir::success();
  }

  ::mlir::LogicalResult promoteStraightLineThread(::mlir::ModuleOp module,
                                                  ::dataflow::ThreadOp thread,
                                                  ::mlir::OpBuilder &builder) {
    ::mlir::Location loc = thread.getLoc();
    ::mlir::Type noneType = builder.getType<::mlir::NoneType>();
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

    ::llvm::SmallVector<::mlir::Value, 8> inputValues;
    for (size_t i = 0; i < threadInputCount; ++i)
      inputValues.push_back(threadEntry.getArgument(i));
    for (size_t i = threadInputCount + 1, e = threadEntry.getNumArguments();
         i < e; ++i)
      inputValues.push_back(threadEntry.getArgument(i));
    ClassifiedGraphValues graphInputs = classifyGraphValues(inputValues);
    ClassifiedGraphValues graphResults;
    ::llvm::SmallVector<::mlir::Value, 8> orderedInputs =
        graphInputs.ordered();
    ::llvm::SmallVector<::mlir::Type, 8> inputTypes;
    for (::mlir::Value value : orderedInputs)
      inputTypes.push_back(value.getType());

    ::mlir::FunctionType graphType =
        builder.getFunctionType(inputTypes, {});
    auto segmentAttrs = graphSegmentAttrs(builder, graphInputs, graphResults);
    std::string stem = "g_" + sanitizeSymbol(thread.getSymName()) + "_0";
    std::string graphName = uniqueSymbol(module, stem);

    builder.setInsertionPointToEnd(module.getBody());
    auto graph = ::dataflow::GraphFuncOp::create(
        builder, loc, graphName, graphType, segmentAttrs);
    graph.setSymVisibilityAttr(builder.getStringAttr("private"));

    ::mlir::Block *graphEntry = builder.createBlock(&graph.getBody());
    graphEntry->addArgument(noneType, loc);
    for (::mlir::Type ty : inputTypes)
      graphEntry->addArgument(ty, loc);

    ::mlir::IRMapping mapping;
    mapping.map(threadCtrl, graphEntry->getArgument(0));
    for (auto [i, value] : ::llvm::enumerate(orderedInputs))
      mapping.map(value, graphEntry->getArgument(i + 1));

    builder.setInsertionPointToEnd(graphEntry);
    for (::mlir::Operation *op : bodyOps)
      builder.clone(*op, mapping);
    ::dataflow::GraphReturnOp::create(
        builder, loc, ::mlir::ValueRange{}, ::mlir::ValueRange{},
        ::mlir::ValueRange{},
        ::mlir::ValueRange{graphEntry->getArgument(0)});

    builder.setInsertionPoint(&threadEntry, threadEntry.begin());
    ::llvm::SmallVector<::mlir::Value, 8> launchOperands = orderedInputs;

    auto callee =
        ::mlir::FlatSymbolRefAttr::get(builder.getContext(), graphName);
    auto launch = ::dataflow::GraphLaunchOp::create(
        builder, loc, /*doneOut=*/noneType, /*results=*/::mlir::TypeRange{},
        callee, threadCtrl, launchOperands);
    addThreadCompletionFrontier(thread, launch.getDoneOut());

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
