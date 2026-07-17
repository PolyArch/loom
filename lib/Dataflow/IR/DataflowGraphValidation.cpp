#include "Dataflow/IR/DataflowGraphValidation.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Errc.h"

namespace {

llvm::Error graphError(const llvm::Twine &message) {
  return llvm::createStringError(llvm::errc::invalid_argument, message.str());
}

bool isMemoryCapabilityType(mlir::Type type) {
  return llvm::isa<mlir::MemRefType, mlir::UnrankedMemRefType,
                   mlir::LLVM::LLVMPointerType>(type);
}

bool isGraphMemoryInput(dataflow::GraphOp graph, mlir::Value value) {
  auto argument = llvm::dyn_cast<mlir::BlockArgument>(value);
  return argument && argument.getOwner() == &graph.getBody().front() &&
         argument.getArgNumber() > 0 &&
         graph.getInputPortKind(argument.getArgNumber() - 1) ==
             dataflow::GraphPortKind::Memory;
}

bool isLaunchAvailableValueInput(dataflow::GraphOp graph, mlir::Value value) {
  auto argument = llvm::dyn_cast<mlir::BlockArgument>(value);
  return argument && argument.getOwner() == &graph.getBody().front() &&
         argument.getArgNumber() > 0 &&
         graph.getInputPortKind(argument.getArgNumber() - 1) ==
             dataflow::GraphPortKind::Value;
}

bool isSupportedMemoryView(mlir::Operation *op) {
  return llvm::isa<mlir::memref::CastOp>(op);
}

bool isProtocolEstablishedMemory(dataflow::GraphOp graph, mlir::Value value) {
  llvm::DenseSet<mlir::Value> visited;
  while (value && visited.insert(value).second) {
    if (isGraphMemoryInput(graph, value))
      return true;
    mlir::Operation *def = value.getDefiningOp();
    if (!def)
      return false;
    if (llvm::isa<mlir::memref::AllocOp>(def))
      return true;
    if (isSupportedMemoryView(def)) {
      value = mlir::cast<mlir::ViewLikeOpInterface>(def).getViewSource();
      continue;
    }
    if (auto cast = llvm::dyn_cast<mlir::UnrealizedConversionCastOp>(def)) {
      if (cast.getInputs().size() != 1 || cast.getResults().size() != 1)
        return false;
      value = cast.getInputs().front();
      continue;
    }
    return false;
  }
  return false;
}

bool isFreshMemoryRoot(mlir::Value value) {
  llvm::DenseSet<mlir::Value> visited;
  while (value && visited.insert(value).second) {
    mlir::Operation *def = value.getDefiningOp();
    if (!def)
      return false;
    if (llvm::isa<mlir::memref::AllocOp>(def))
      return true;
    if (isSupportedMemoryView(def)) {
      value = mlir::cast<mlir::ViewLikeOpInterface>(def).getViewSource();
      continue;
    }
    if (auto cast = llvm::dyn_cast<mlir::UnrealizedConversionCastOp>(def)) {
      if (cast.getInputs().size() != 1 || cast.getResults().size() != 1)
        return false;
      value = cast.getInputs().front();
      continue;
    }
    return false;
  }
  return false;
}

bool isCanonicalMemoryBridge(dataflow::GraphOp graph, mlir::Operation *op) {
  auto cast = llvm::dyn_cast<mlir::UnrealizedConversionCastOp>(op);
  if (!cast || cast.getInputs().size() != 1 || cast.getResults().size() != 1)
    return false;
  mlir::Type inputType = cast.getInputs().front().getType();
  mlir::Type resultType = cast.getResults().front().getType();
  const bool inputPointer = mlir::isa<mlir::LLVM::LLVMPointerType>(inputType);
  const bool resultPointer = mlir::isa<mlir::LLVM::LLVMPointerType>(resultType);
  if (inputPointer == resultPointer || (!isMemoryCapabilityType(inputType) ||
                                        !isMemoryCapabilityType(resultType)))
    return false;
  return isProtocolEstablishedMemory(graph, cast.getInputs().front());
}

bool isResidualLLVMMemoryOperation(mlir::Operation *op) {
  return llvm::isa<mlir::LLVM::LoadOp, mlir::LLVM::StoreOp,
                   mlir::LLVM::MemcpyOp>(op) ||
         op->getName().getStringRef() == "llvm.intr.memset";
}

bool hasRawPointerUse(mlir::Operation *op) {
  return llvm::any_of(op->getOperandTypes(),
                      [](mlir::Type type) {
                        return mlir::isa<mlir::LLVM::LLVMPointerType>(type);
                      }) ||
         llvm::any_of(op->getResultTypes(), [](mlir::Type type) {
           return mlir::isa<mlir::LLVM::LLVMPointerType>(type);
         });
}

using SelectorLanes = llvm::DenseMap<mlir::Value, unsigned>;

bool constrainSelectorLane(mlir::Value selector, unsigned lane,
                           SelectorLanes &selectorLanes) {
  auto [it, inserted] = selectorLanes.try_emplace(selector, lane);
  return inserted || it->second == lane;
}

bool constrainDemuxLane(mlir::Value value, SelectorLanes &selectorLanes) {
  auto result = llvm::dyn_cast<mlir::OpResult>(value);
  auto demux =
      result ? llvm::dyn_cast<dataflow::DemuxOp>(result.getOwner()) : nullptr;
  return !demux || constrainSelectorLane(
                       demux.getSel(), result.getResultNumber(), selectorLanes);
}

bool causallyDependsOn(mlir::Value event, mlir::Value prerequisite,
                       llvm::DenseSet<mlir::Value> &visited,
                       SelectorLanes &selectorLanes) {
  if (!event || !constrainDemuxLane(event, selectorLanes))
    return false;
  if (event == prerequisite)
    return true;
  if (!visited.insert(event).second)
    return false;
  mlir::Operation *def = event.getDefiningOp();
  if (!def)
    return false;

  auto dependsOn = [&](mlir::Value operand, SelectorLanes branchLanes) {
    llvm::DenseSet<mlir::Value> branchVisited = visited;
    return causallyDependsOn(operand, prerequisite, branchVisited, branchLanes);
  };
  auto dependsOnAnyOperand = [&]() {
    return llvm::any_of(def->getOperands(), [&](mlir::Value operand) {
      return dependsOn(operand, selectorLanes);
    });
  };

  if (auto sync = llvm::dyn_cast<dataflow::SyncOp>(def)) {
    if (llvm::isa<mlir::NoneType>(event.getType()) &&
        prerequisite.getDefiningOp() == sync.getOperation())
      return true;
    return dependsOnAnyOperand();
  }
  if (auto load = llvm::dyn_cast<dataflow::LoadOp>(def)) {
    if (event == load.getDone() && prerequisite == load.getData())
      return true;
    return dependsOnAnyOperand();
  }
  if (auto mux = llvm::dyn_cast<dataflow::MuxOp>(def)) {
    if (dependsOn(mux.getSel(), selectorLanes))
      return true;
    for (auto [lane, input] : llvm::enumerate(mux.getInputs())) {
      SelectorLanes laneConstraints = selectorLanes;
      if (constrainSelectorLane(mux.getSel(), lane, laneConstraints) &&
          dependsOn(input, std::move(laneConstraints)))
        return true;
    }
    return false;
  }
  if (auto select = llvm::dyn_cast<mlir::arith::SelectOp>(def)) {
    if (dependsOn(select.getCondition(), selectorLanes))
      return true;
    return dependsOn(select.getTrueValue(), selectorLanes) ||
           dependsOn(select.getFalseValue(), selectorLanes);
  }
  if (auto carry = llvm::dyn_cast<dataflow::CarryOp>(def)) {
    return dependsOn(carry.getInit(), selectorLanes) ||
           dependsOn(carry.getCarry(), selectorLanes);
  }
  if (auto demux = llvm::dyn_cast<dataflow::DemuxOp>(def)) {
    if (dependsOn(demux.getSel(), selectorLanes))
      return true;
    return dependsOn(demux.getInput(), selectorLanes);
  }
  if (auto gate = llvm::dyn_cast<dataflow::GateOp>(def)) {
    if (dependsOn(gate.getBeforeCond(), selectorLanes))
      return true;
    return dependsOn(gate.getBeforeValue(), selectorLanes);
  }
  if (auto invariant = llvm::dyn_cast<dataflow::InvariantOp>(def)) {
    if (dependsOn(invariant.getCond(), selectorLanes))
      return true;
    return dependsOn(invariant.getInit(), selectorLanes);
  }
  if (auto constant = llvm::dyn_cast<dataflow::ConstantOp>(def))
    return dependsOn(constant.getCtrl(), selectorLanes);
  return dependsOnAnyOperand();
}

bool causallyDependsOn(mlir::Value event, mlir::Value prerequisite) {
  llvm::DenseSet<mlir::Value> visited;
  SelectorLanes selectorLanes;
  return causallyDependsOn(event, prerequisite, visited, selectorLanes);
}

bool isExplicitSyncCoverage(mlir::Value witness, mlir::Value prerequisite) {
  auto sync = llvm::dyn_cast_or_null<dataflow::SyncOp>(witness.getDefiningOp());
  return sync && llvm::isa<mlir::NoneType>(witness.getType()) &&
         prerequisite.getDefiningOp() == sync.getOperation();
}

bool isExplicitLoadCoverage(mlir::Value witness, mlir::Value prerequisite) {
  auto load = llvm::dyn_cast_or_null<dataflow::LoadOp>(witness.getDefiningOp());
  return load && witness == load.getDone() && prerequisite == load.getData();
}

bool isCovered(mlir::Value prerequisite, mlir::ValueRange completion) {
  return llvm::any_of(completion, [&](mlir::Value witness) {
    return causallyDependsOn(witness, prerequisite) ||
           isExplicitSyncCoverage(witness, prerequisite) ||
           isExplicitLoadCoverage(witness, prerequisite);
  });
}

bool coversFalseClose(mlir::Value witness, mlir::Value closeSignal,
                      llvm::DenseSet<mlir::Value> &visited,
                      SelectorLanes &selectorLanes) {
  if (!witness || !constrainDemuxLane(witness, selectorLanes) ||
      !visited.insert(witness).second)
    return false;
  mlir::Operation *def = witness.getDefiningOp();
  if (!def)
    return false;

  auto covers = [&](mlir::Value value, SelectorLanes branchLanes) {
    llvm::DenseSet<mlir::Value> branchVisited = visited;
    return coversFalseClose(value, closeSignal, branchVisited, branchLanes);
  };
  if (auto demux = llvm::dyn_cast<dataflow::DemuxOp>(def)) {
    auto result = llvm::dyn_cast<mlir::OpResult>(witness);
    if (result && result.getResultNumber() == 0 &&
        demux.getSel() == closeSignal)
      return true;
    return covers(demux.getInput(), selectorLanes);
  }
  if (auto sync = llvm::dyn_cast<dataflow::SyncOp>(def)) {
    return llvm::any_of(sync.getInputs(), [&](mlir::Value input) {
      return covers(input, selectorLanes);
    });
  }
  if (auto mux = llvm::dyn_cast<dataflow::MuxOp>(def)) {
    for (auto [lane, input] : llvm::enumerate(mux.getInputs())) {
      SelectorLanes laneConstraints = selectorLanes;
      if (constrainSelectorLane(mux.getSel(), lane, laneConstraints) &&
          covers(input, std::move(laneConstraints)))
        return true;
    }
    return false;
  }
  if (auto carry = llvm::dyn_cast<dataflow::CarryOp>(def)) {
    return covers(carry.getInit(), selectorLanes) ||
           covers(carry.getCarry(), selectorLanes);
  }
  if (auto load = llvm::dyn_cast<dataflow::LoadOp>(def)) {
    return witness == load.getDone() && covers(load.getCtrl(), selectorLanes);
  }
  if (auto store = llvm::dyn_cast<dataflow::StoreOp>(def)) {
    return witness == store.getDone() && covers(store.getCtrl(), selectorLanes);
  }
  return false;
}

bool coversFalseClose(mlir::Value closeSignal, mlir::ValueRange completion) {
  return llvm::any_of(completion, [&](mlir::Value witness) {
    llvm::DenseSet<mlir::Value> visited;
    SelectorLanes selectorLanes;
    return coversFalseClose(witness, closeSignal, visited, selectorLanes);
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
    return gate.getAfterCond();
  if (auto parallelize = llvm::dyn_cast<dataflow::ParallelizeOp>(op))
    return parallelize.getCont();
  if (auto serialize = llvm::dyn_cast<dataflow::SerializeOp>(op))
    return serialize.getCont();
  return {};
}

bool hasPhaseAlignedGateValue(dataflow::GateOp gate) {
  mlir::Operation *def = gate.getBeforeValue().getDefiningOp();
  if (auto carry = llvm::dyn_cast_or_null<dataflow::CarryOp>(def))
    return carry.getOutput() == gate.getBeforeValue() &&
           carry.getCond() == gate.getBeforeCond();
  if (auto invariant = llvm::dyn_cast_or_null<dataflow::InvariantOp>(def))
    return invariant.getOutput() == gate.getBeforeValue() &&
           invariant.getCond() == gate.getBeforeCond();
  return false;
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
      !llvm::isa<mlir::LLVM::CallOp, mlir::LLVM::StoreOp, mlir::LLVM::MemcpyOp>(
          op))
    return false;
  return true;
}

} // namespace

llvm::Error dataflow::validateFinalizedGraph(GraphOp graph) {
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
          llvm::Twine(
              "finalized graph contains residual structured operation '") +
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
    if (llvm::isa<mlir::memref::GetGlobalOp>(op)) {
      structuralError = graphError(
          "finalized graph contains forbidden memory root 'memref.get_global'");
      return mlir::WalkResult::interrupt();
    }
    if (llvm::isa<mlir::memref::LoadOp, mlir::memref::StoreOp>(op) ||
        isResidualLLVMMemoryOperation(op)) {
      structuralError = graphError(
          llvm::Twine("finalized graph contains residual memory operation '") +
          op->getName().getStringRef() + "'");
      return mlir::WalkResult::interrupt();
    }
    if (auto alloc = llvm::dyn_cast<mlir::memref::AllocOp>(op)) {
      if (!llvm::all_of(alloc.getDynamicSizes(), [&](mlir::Value extent) {
            return isLaunchAvailableValueInput(graph, extent);
          })) {
        structuralError = graphError(
            "memref.alloc dynamic extent must be a graph value input");
        return mlir::WalkResult::interrupt();
      }
    } else if (op->getDialect() &&
               op->getDialect()->getNamespace() == "memref" &&
               !isSupportedMemoryView(op)) {
      structuralError = graphError(
          llvm::Twine("finalized graph contains unsupported memory capability "
                      "operation '") +
          op->getName().getStringRef() + "'");
      return mlir::WalkResult::interrupt();
    }
    if (auto cast = llvm::dyn_cast<mlir::UnrealizedConversionCastOp>(op)) {
      bool hasMemoryCapability =
          llvm::any_of(cast.getInputs(),
                       [](mlir::Value value) {
                         return isMemoryCapabilityType(value.getType());
                       }) ||
          llvm::any_of(cast.getResults(), [](mlir::Value value) {
            return isMemoryCapabilityType(value.getType());
          });
      if (hasMemoryCapability && !isCanonicalMemoryBridge(graph, op)) {
        structuralError = graphError(
            "finalized graph contains unsupported memory capability bridge");
        return mlir::WalkResult::interrupt();
      }
    }
    if (hasRawPointerUse(op) && !llvm::isa<GraphReturnOp>(op) &&
        !isCanonicalMemoryBridge(graph, op)) {
      structuralError = graphError(
          llvm::Twine("finalized graph contains residual pointer operation '") +
          op->getName().getStringRef() + "'");
      return mlir::WalkResult::interrupt();
    }
    if (llvm::any_of(op->getResultTypes(), isMemoryCapabilityType) &&
        !llvm::isa<mlir::memref::AllocOp>(op) && !isSupportedMemoryView(op) &&
        !isCanonicalMemoryBridge(graph, op)) {
      structuralError = graphError(
          llvm::Twine("finalized graph contains unsupported memory capability "
                      "producer '") +
          op->getName().getStringRef() + "'");
      return mlir::WalkResult::interrupt();
    }
    if (!llvm::isa<CanonicalDataflowActorOpInterface>(op) &&
        !llvm::isa<mlir::memref::AllocOp, mlir::memref::CastOp>(op) &&
        !isCanonicalMemoryBridge(graph, op)) {
      structuralError = graphError(
          llvm::Twine("finalized graph contains unregistered actor '") +
          op->getName().getStringRef() + "'");
      return mlir::WalkResult::interrupt();
    }
    return mlir::WalkResult::advance();
  });
  if (structuralError)
    return structuralError;

  bool hasRealWork =
      llvm::any_of(entry.without_terminator(), [&](mlir::Operation &op) {
        return !llvm::isa<mlir::memref::AllocOp, mlir::memref::CastOp>(op) &&
               !isCanonicalMemoryBridge(graph, &op);
      });
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
    if (isFreshMemoryRoot(memory) &&
        !mlir::isa<mlir::MemRefType, mlir::UnrankedMemRefType>(
            memory.getType()))
      return graphError("fresh memory export must use a memref result");
    if (isProtocolEstablishedMemory(graph, memory)) {
      if (!isCovered(graph.getStart(), ret.getComplete()))
        return graphError(
            llvm::Twine(
                "retirement frontier does not cover establishment of ") +
            "memory output #" + llvm::Twine(index));
      continue;
    }
    if (!isCovered(memory, ret.getComplete()))
      return graphError(
          llvm::Twine("retirement frontier does not causally cover memory ") +
          "output #" + llvm::Twine(index));
  }

  for (mlir::Operation &op : entry.without_terminator()) {
    if (auto gate = llvm::dyn_cast<dataflow::GateOp>(op)) {
      bool covered = coversFalseClose(gate.getAfterCond(), ret.getComplete());
      if (!covered && hasPhaseAlignedGateValue(gate))
        covered = coversFalseClose(gate.getBeforeCond(), ret.getComplete());
      if (!covered)
        return graphError(
            "retirement frontier does not cover close/reset of "
            "'dataflow.gate'");
      continue;
    }
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
    if (llvm::isa<GraphReturnOp, mlir::memref::AllocOp>(op) || mlir::isPure(op))
      return mlir::WalkResult::advance();
    if (hasObservableEffect(op)) {
      effectError = graphError(
          llvm::Twine(
              "finalized graph contains unsupported effect operation '") +
          op->getName().getStringRef() + "'");
      return mlir::WalkResult::interrupt();
    }
    return mlir::WalkResult::advance();
  });
  return effectError;
}

llvm::Error dataflow::validateFinalizedProgram(mlir::ModuleOp module) {
  if (!module)
    return graphError("finalized program must be a module");
  llvm::Error error = llvm::Error::success();
  module.walk([&](GraphOp graph) {
    if (error)
      return mlir::WalkResult::interrupt();
    error = validateFinalizedGraph(graph);
    return error ? mlir::WalkResult::interrupt() : mlir::WalkResult::advance();
  });
  return error;
}
