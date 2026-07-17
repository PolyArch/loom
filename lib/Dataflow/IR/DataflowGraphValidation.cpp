#include "Dataflow/IR/DataflowGraphValidation.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Errc.h"

namespace {

llvm::Error graphError(const llvm::Twine &message) {
  return llvm::createStringError(llvm::errc::invalid_argument,
                                 message.str());
}

bool isMemoryCapabilityType(mlir::Type type) {
  return llvm::isa<mlir::MemRefType, mlir::UnrankedMemRefType,
                   mlir::LLVM::LLVMPointerType>(type);
}

bool causallyDependsOn(mlir::Value event, mlir::Value prerequisite,
                       llvm::DenseSet<mlir::Value> &visited) {
  if (event == prerequisite)
    return true;
  if (event.getDefiningOp() &&
      event.getDefiningOp() == prerequisite.getDefiningOp())
    return true;
  if (!event || !visited.insert(event).second)
    return false;
  mlir::Operation *def = event.getDefiningOp();
  if (!def)
    return false;

  auto dependsOnAnyOperand = [&]() {
    return llvm::any_of(def->getOperands(), [&](mlir::Value operand) {
      llvm::DenseSet<mlir::Value> branchVisited = visited;
      return causallyDependsOn(operand, prerequisite, branchVisited);
    });
  };

  if (auto mux = llvm::dyn_cast<dataflow::MuxOp>(def)) {
    llvm::DenseSet<mlir::Value> selectorVisited = visited;
    if (causallyDependsOn(mux.getSel(), prerequisite, selectorVisited))
      return true;
    return llvm::any_of(mux.getInputs(), [&](mlir::Value input) {
      llvm::DenseSet<mlir::Value> laneVisited = visited;
      return causallyDependsOn(input, prerequisite, laneVisited);
    });
  }
  if (auto select = llvm::dyn_cast<mlir::arith::SelectOp>(def)) {
    llvm::DenseSet<mlir::Value> selectorVisited = visited;
    if (causallyDependsOn(select.getCondition(), prerequisite,
                          selectorVisited))
      return true;
    llvm::DenseSet<mlir::Value> trueVisited = visited;
    llvm::DenseSet<mlir::Value> falseVisited = visited;
    return causallyDependsOn(select.getTrueValue(), prerequisite,
                             trueVisited) ||
           causallyDependsOn(select.getFalseValue(), prerequisite,
                             falseVisited);
  }
  if (auto carry = llvm::dyn_cast<dataflow::CarryOp>(def)) {
    llvm::DenseSet<mlir::Value> initVisited = visited;
    llvm::DenseSet<mlir::Value> carryVisited = visited;
    return causallyDependsOn(carry.getInit(), prerequisite, initVisited) ||
           causallyDependsOn(carry.getCarry(), prerequisite, carryVisited);
  }
  if (auto demux = llvm::dyn_cast<dataflow::DemuxOp>(def)) {
    llvm::DenseSet<mlir::Value> selectorVisited = visited;
    if (causallyDependsOn(demux.getSel(), prerequisite, selectorVisited))
      return true;
    return causallyDependsOn(demux.getInput(), prerequisite, visited);
  }
  if (auto gate = llvm::dyn_cast<dataflow::GateOp>(def)) {
    llvm::DenseSet<mlir::Value> conditionVisited = visited;
    if (causallyDependsOn(gate.getBeforeCond(), prerequisite,
                          conditionVisited))
      return true;
    return causallyDependsOn(gate.getBeforeValue(), prerequisite, visited);
  }
  if (auto invariant = llvm::dyn_cast<dataflow::InvariantOp>(def)) {
    llvm::DenseSet<mlir::Value> phaseVisited = visited;
    if (causallyDependsOn(invariant.getCond(), prerequisite, phaseVisited))
      return true;
    return causallyDependsOn(invariant.getInit(), prerequisite, visited);
  }
  if (auto constant = llvm::dyn_cast<dataflow::ConstantOp>(def))
    return causallyDependsOn(constant.getCtrl(), prerequisite, visited);
  return dependsOnAnyOperand();
}

bool causallyDependsOn(mlir::Value event, mlir::Value prerequisite) {
  llvm::DenseSet<mlir::Value> visited;
  return causallyDependsOn(event, prerequisite, visited);
}

bool isCovered(mlir::Value prerequisite, mlir::ValueRange completion) {
  return llvm::any_of(completion, [&](mlir::Value witness) {
    return causallyDependsOn(witness, prerequisite);
  });
}

bool coversFalseClose(mlir::Value witness, mlir::Value closeSignal,
                      llvm::DenseSet<mlir::Value> &visited) {
  if (!witness || !visited.insert(witness).second)
    return false;
  mlir::Operation *def = witness.getDefiningOp();
  if (!def)
    return false;

  if (auto demux = llvm::dyn_cast<dataflow::DemuxOp>(def)) {
    auto result = llvm::dyn_cast<mlir::OpResult>(witness);
    if (result && result.getResultNumber() == 0 &&
        demux.getSel() == closeSignal)
      return true;
    return coversFalseClose(demux.getInput(), closeSignal, visited);
  }
  if (auto sync = llvm::dyn_cast<dataflow::SyncOp>(def)) {
    return llvm::any_of(sync.getInputs(), [&](mlir::Value input) {
      llvm::DenseSet<mlir::Value> branchVisited = visited;
      return coversFalseClose(input, closeSignal, branchVisited);
    });
  }
  if (auto mux = llvm::dyn_cast<dataflow::MuxOp>(def)) {
    return llvm::any_of(mux.getInputs(), [&](mlir::Value input) {
      llvm::DenseSet<mlir::Value> branchVisited = visited;
      return coversFalseClose(input, closeSignal, branchVisited);
    });
  }
  if (auto carry = llvm::dyn_cast<dataflow::CarryOp>(def)) {
    llvm::DenseSet<mlir::Value> initVisited = visited;
    llvm::DenseSet<mlir::Value> feedbackVisited = visited;
    return coversFalseClose(carry.getInit(), closeSignal, initVisited) ||
           coversFalseClose(carry.getCarry(), closeSignal, feedbackVisited);
  }
  if (auto load = llvm::dyn_cast<dataflow::LoadOp>(def)) {
    return witness == load.getDone() &&
           coversFalseClose(load.getCtrl(), closeSignal, visited);
  }
  if (auto store = llvm::dyn_cast<dataflow::StoreOp>(def)) {
    return witness == store.getDone() &&
           coversFalseClose(store.getCtrl(), closeSignal, visited);
  }
  return false;
}

bool coversFalseClose(mlir::Value closeSignal,
                      mlir::ValueRange completion) {
  return llvm::any_of(completion, [&](mlir::Value witness) {
    llvm::DenseSet<mlir::Value> visited;
    return coversFalseClose(witness, closeSignal, visited);
  });
}

mlir::Value statefulCloseSignal(mlir::Operation *op) {
  if (auto stream = llvm::dyn_cast<dataflow::StreamOp>(op))
    return stream.getPhase();
  if (auto carry = llvm::dyn_cast<dataflow::CarryOp>(op))
    return carry.getCond();
  if (auto invariant = llvm::dyn_cast<dataflow::InvariantOp>(op))
    return invariant.getCond();
  if (auto gate = llvm::dyn_cast<dataflow::GateOp>(op))
    return gate.getBeforeCond();
  if (auto parallelize = llvm::dyn_cast<dataflow::ParallelizeOp>(op))
    return parallelize.getCont();
  if (auto serialize = llvm::dyn_cast<dataflow::SerializeOp>(op))
    return serialize.getCont();
  return {};
}

void collectStreamCloseSignals(mlir::Value value,
                               llvm::DenseSet<mlir::Value> &visited,
                               llvm::SmallVectorImpl<mlir::Value> &signals) {
  if (!value || !visited.insert(value).second)
    return;
  mlir::Operation *def = value.getDefiningOp();
  if (!def)
    return;
  if (mlir::Value signal = statefulCloseSignal(def)) {
    if (!llvm::is_contained(signals, signal))
      signals.push_back(signal);
    return;
  }
  for (mlir::Value operand : def->getOperands())
    collectStreamCloseSignals(operand, visited, signals);
}

bool isProtocolEstablishedMemory(dataflow::GraphFuncOp graph,
                                 mlir::Value value) {
  llvm::DenseSet<mlir::Value> visited;
  while (value && visited.insert(value).second) {
    if (auto argument = llvm::dyn_cast<mlir::BlockArgument>(value)) {
      return argument.getOwner() == &graph.getBody().front() &&
             argument.getArgNumber() > 0 &&
             graph.getInputPortKind(argument.getArgNumber() - 1) ==
                 dataflow::GraphPortKind::Memory;
    }
    mlir::Operation *def = value.getDefiningOp();
    if (!def)
      return false;
    if (auto view = llvm::dyn_cast<mlir::ViewLikeOpInterface>(def)) {
      value = view.getViewSource();
      continue;
    }
    if (auto cast = llvm::dyn_cast<mlir::UnrealizedConversionCastOp>(def)) {
      if (cast.getInputs().size() != 1)
        return false;
      value = cast.getInputs().front();
      continue;
    }
    return llvm::isa<mlir::memref::GetGlobalOp>(def);
  }
  return false;
}

bool hasObservableEffect(mlir::Operation *op) {
  auto effects = llvm::dyn_cast<mlir::MemoryEffectOpInterface>(op);
  if (effects)
    return effects.hasEffect<mlir::MemoryEffects::Write>() ||
           effects.hasEffect<mlir::MemoryEffects::Allocate>() ||
           effects.hasEffect<mlir::MemoryEffects::Free>();
  if (mlir::isPure(op))
    return false;
  llvm::StringRef dialect = op->getName().getDialectNamespace();
  if (dialect == "arith" || dialect == "math" || dialect == "ub")
    return false;
  if (dialect == "llvm" &&
      !llvm::isa<mlir::LLVM::CallOp, mlir::LLVM::StoreOp,
                 mlir::LLVM::MemcpyOp>(op))
    return false;
  return true;
}

} // namespace

llvm::Error dataflow::validateFinalizedGraph(GraphFuncOp graph) {
  if (!graph || graph.isExternal())
    return graphError("finalized graph must have a body");
  mlir::Block &entry = graph.getBody().front();
  auto ret = llvm::dyn_cast<GraphReturnOp>(entry.getTerminator());
  if (!ret)
    return graphError("finalized graph is missing dataflow.graph.return");

  llvm::Error structuralError = llvm::Error::success();
  graph.getBody().walk<mlir::WalkOrder::PreOrder>([&](mlir::Operation *op) {
    if (structuralError || llvm::isa<GraphReturnOp>(op))
      return mlir::WalkResult::interrupt();
    if (op->getName().getDialectNamespace() == "scf" ||
        op->getName().getDialectNamespace() == "cf" ||
        op->getNumRegions() != 0 || op->getNumSuccessors() != 0) {
      structuralError = graphError(
          llvm::Twine("finalized graph contains residual structured operation '") +
          op->getName().getStringRef() + "'");
      return mlir::WalkResult::interrupt();
    }
    if (llvm::isa<CarryOp, MuxOp, DemuxOp, GateOp, InvariantOp>(op) &&
        (llvm::any_of(op->getOperandTypes(), isMemoryCapabilityType) ||
         llvm::any_of(op->getResultTypes(), isMemoryCapabilityType))) {
      structuralError = graphError(
          llvm::Twine("finalized graph routes memory capability through '") +
          op->getName().getStringRef() + "'");
      return mlir::WalkResult::interrupt();
    }
    return mlir::WalkResult::advance();
  });
  if (structuralError)
    return structuralError;

  bool hasRealWork = llvm::any_of(entry.without_terminator(),
                                  [](mlir::Operation &) { return true; });
  if (hasRealWork && llvm::is_contained(ret.getComplete(), graph.getStart()))
    return graphError(
        "nontrivial graph uses raw start as a retirement completion witness");

  for (auto [index, value] : llvm::enumerate(ret.getValues()))
    if (!isCovered(value, ret.getComplete()))
      return graphError(llvm::Twine("retirement frontier does not causally ") +
                        "cover value output #" + llvm::Twine(index));

  for (auto [index, stream] : llvm::enumerate(ret.getStreams())) {
    llvm::DenseSet<mlir::Value> visited;
    llvm::SmallVector<mlir::Value, 2> closeSignals;
    collectStreamCloseSignals(stream, visited, closeSignals);
    bool covered = closeSignals.empty()
                       ? isCovered(stream, ret.getComplete())
                       : llvm::all_of(closeSignals, [&](mlir::Value signal) {
                           return coversFalseClose(signal, ret.getComplete());
                         });
    if (!covered)
      return graphError(llvm::Twine("retirement frontier does not causally ") +
                        "cover stream output #" + llvm::Twine(index));
  }

  for (auto [index, memory] : llvm::enumerate(ret.getMemories())) {
    if (isProtocolEstablishedMemory(graph, memory)) {
      if (!isCovered(graph.getStart(), ret.getComplete()))
        return graphError(
            llvm::Twine("retirement frontier does not cover establishment of ") +
            "memory output #" + llvm::Twine(index));
      continue;
    }
    if (!isCovered(memory, ret.getComplete()))
      return graphError(
          llvm::Twine("retirement frontier does not causally cover memory ") +
          "output #" + llvm::Twine(index));
  }

  for (mlir::Operation &op : entry.without_terminator()) {
    mlir::Value closeSignal = statefulCloseSignal(&op);
    if (!closeSignal)
      continue;
    llvm::DenseSet<mlir::Value> visited;
    llvm::SmallVector<mlir::Value, 2> sourceCloses;
    collectStreamCloseSignals(closeSignal, visited, sourceCloses);
    if (sourceCloses.empty())
      sourceCloses.push_back(closeSignal);
    if (!llvm::all_of(sourceCloses, [&](mlir::Value signal) {
          return coversFalseClose(signal, ret.getComplete());
        }))
      return graphError(
          llvm::Twine("retirement frontier does not cover close/reset of '") +
          op.getName().getStringRef() + "'");
  }

  llvm::Error effectError = llvm::Error::success();
  graph.getBody().walk([&](mlir::Operation *op) {
    if (effectError)
      return mlir::WalkResult::interrupt();
    if (auto load = llvm::dyn_cast<LoadOp>(op)) {
      if (!isCovered(load.getDone(), ret.getComplete()))
        effectError = graphError(
            "retirement frontier does not causally cover dataflow.load done");
      return effectError ? mlir::WalkResult::interrupt()
                         : mlir::WalkResult::advance();
    }
    if (auto store = llvm::dyn_cast<StoreOp>(op)) {
      if (!isCovered(store.getDone(), ret.getComplete()))
        effectError = graphError(
            "retirement frontier does not causally cover dataflow.store done");
      return effectError ? mlir::WalkResult::interrupt()
                         : mlir::WalkResult::advance();
    }
    if (auto call = llvm::dyn_cast<mlir::LLVM::CallOp>(op)) {
      bool covered = !call.getResults().empty() &&
                     llvm::any_of(call.getResults(), [&](mlir::Value result) {
                       return isCovered(result, ret.getComplete());
                     });
      if (covered)
        return mlir::WalkResult::advance();
    }
    if (llvm::isa<GraphReturnOp>(op) || mlir::isPure(op))
      return mlir::WalkResult::advance();
    if (hasObservableEffect(op)) {
      effectError = graphError(
          llvm::Twine("finalized graph contains unsupported effect operation '") +
          op->getName().getStringRef() + "'");
      return mlir::WalkResult::interrupt();
    }
    return mlir::WalkResult::advance();
  });
  return effectError;
}
