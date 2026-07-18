#include "Dataflow/IR/DataflowGraphValidation.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Matchers.h"
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
  return dataflow::DataflowDialect::isMemoryCapabilityType(type);
}

bool containsChannelType(mlir::Type type) {
  return type
      .walk<mlir::WalkOrder::PreOrder>([](mlir::Type nested) {
        return llvm::isa<dataflow::ChannelType>(nested)
                   ? mlir::WalkResult::interrupt()
                   : mlir::WalkResult::advance();
      })
      .wasInterrupted();
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
    if (prerequisite.getDefiningOp() == sync.getOperation())
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

bool isExplicitLoadCoverage(mlir::Value witness, mlir::Value prerequisite) {
  auto load = llvm::dyn_cast_or_null<dataflow::LoadOp>(witness.getDefiningOp());
  return load && witness == load.getDone() && prerequisite == load.getData();
}

bool isCovered(mlir::Value prerequisite, mlir::ValueRange completion) {
  return llvm::any_of(completion, [&](mlir::Value witness) {
    return causallyDependsOn(witness, prerequisite) ||
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

std::optional<mlir::Value>
getFixedSequenceStreamActivation(dataflow::StreamOp stream,
                                 std::optional<unsigned> expectedWidth) {
  auto initConstant = stream.getInit().getDefiningOp<dataflow::ConstantOp>();
  auto limitConstant = stream.getLimit().getDefiningOp<dataflow::ConstantOp>();
  auto stepConstant = stream.getStep().getDefiningOp<dataflow::ConstantOp>();
  if (!initConstant || !limitConstant || !stepConstant ||
      initConstant.getCtrl() != limitConstant.getCtrl() ||
      initConstant.getCtrl() != stepConstant.getCtrl() ||
      stream.getStepKind() != dataflow::StreamStepKind::Add ||
      stream.getPredicate() != mlir::arith::CmpIPredicate::slt)
    return std::nullopt;

  auto getKnownUnsigned = [](dataflow::ConstantOp constant) {
    auto integer = llvm::dyn_cast<mlir::IntegerAttr>(constant.getConstValue());
    if (!integer || integer.getValue().isNegative())
      return std::optional<uint64_t>{};
    return std::optional<uint64_t>{integer.getValue().getZExtValue()};
  };
  auto init = getKnownUnsigned(initConstant);
  auto limit = getKnownUnsigned(limitConstant);
  auto step = getKnownUnsigned(stepConstant);
  if (!init || !limit || !step || *init != 0 || *limit < 2 || *step != 1 ||
      (expectedWidth && *limit != *expectedWidth))
    return std::nullopt;
  return initConstant.getCtrl();
}

void collectStreamCloseSignals(mlir::Value value,
                               llvm::DenseSet<mlir::Value> &visited,
                               llvm::SmallVectorImpl<mlir::Value> &signals) {
  if (!value || !visited.insert(value).second)
    return;
  mlir::Operation *def = value.getDefiningOp();
  if (!def)
    return;
  mlir::Value signal = statefulCloseSignal(def);
  if (auto stream = llvm::dyn_cast<dataflow::StreamOp>(def)) {
    llvm::SmallVector<mlir::Value, 2> activationCloses;
    if (auto activation =
            getFixedSequenceStreamActivation(stream, std::nullopt)) {
      collectStreamCloseSignals(*activation, visited, activationCloses);
      if (!activationCloses.empty()) {
        for (mlir::Value close : activationCloses)
          if (!llvm::is_contained(signals, close))
            signals.push_back(close);
        return;
      }
    }
  }
  if (auto gate = llvm::dyn_cast<dataflow::GateOp>(def))
    if (hasPhaseAlignedGateValue(gate))
      signal = gate.getBeforeCond();
  if (signal) {
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

class GraphCardinalityAnalysis {
public:
  explicit GraphCardinalityAnalysis(dataflow::GraphOp graph) : graph(graph) {}

  bool isExactOne(mlir::Value value) {
    auto known = exactOne.find(value);
    if (known != exactOne.end())
      return known->second;
    if (!exactOneActive.insert(value).second)
      return false;
    bool result = computeExactOne(value);
    exactOneActive.erase(value);
    exactOne.try_emplace(value, result);
    return result;
  }

  bool hasProvenStreamCommit(mlir::Value value) {
    if (isExactOne(value))
      return true;
    llvm::DenseSet<mlir::Value> visited;
    llvm::SmallVector<mlir::Value, 2> closeSignals;
    collectStreamCloseSignals(value, visited, closeSignals);
    return !closeSignals.empty() &&
           llvm::all_of(closeSignals, [&](mlir::Value signal) {
             return isOneClosePhase(signal);
           });
  }

private:
  dataflow::GraphOp graph;
  llvm::DenseMap<mlir::Value, bool> exactOne;
  llvm::DenseSet<mlir::Value> exactOneActive;
  llvm::DenseMap<mlir::Value, bool> oneClosePhase;
  llvm::DenseSet<mlir::Value> oneCloseActive;
  llvm::DenseSet<mlir::Value> alignedCarryAssumptions;
  llvm::DenseMap<mlir::Value, bool> alignedCarrySystems;
  llvm::DenseSet<mlir::Value> alignedCarrySystemsActive;
  llvm::DenseSet<mlir::Value> exactOneAssumptions;

  bool allOperandsExact(mlir::Operation *op) {
    return llvm::all_of(op->getOperands(),
                        [&](mlir::Value value) { return isExactOne(value); });
  }

  bool isGraphStreamInput(mlir::Value value) {
    auto argument = llvm::dyn_cast<mlir::BlockArgument>(value);
    return argument && argument.getOwner() == &graph.getBody().front() &&
           argument.getArgNumber() != 0 &&
           graph.getInputPortKind(argument.getArgNumber() - 1) ==
               dataflow::GraphPortKind::Stream;
  }

  std::optional<uint64_t> getKnownUnsigned(mlir::Value value) {
    if (auto constant = value.getDefiningOp<dataflow::ConstantOp>()) {
      auto integer = llvm::dyn_cast<mlir::IntegerAttr>(constant.getConstValue());
      if (integer && !integer.getValue().isNegative())
        return integer.getValue().getZExtValue();
    }
    mlir::APInt constant;
    if (mlir::matchPattern(value, mlir::m_ConstantInt(&constant)) &&
        !constant.isNegative())
      return constant.getZExtValue();
    return std::nullopt;
  }

  std::optional<mlir::Value>
  getFixedSequenceSelectorActivation(mlir::Value selector, unsigned arity) {
    mlir::Value ordinal;
    if (arity == 2) {
      auto trunc = selector.getDefiningOp<mlir::arith::TruncIOp>();
      if (!trunc || !selector.getType().isInteger(1))
        return std::nullopt;
      ordinal = trunc.getIn();
    } else {
      auto cast = selector.getDefiningOp<mlir::arith::IndexCastOp>();
      if (!cast || !mlir::isa<mlir::IndexType>(selector.getType()))
        return std::nullopt;
      ordinal = cast.getIn();
    }
    auto stream = ordinal.getDefiningOp<dataflow::StreamOp>();
    if (!stream || ordinal != stream.getIv())
      return std::nullopt;
    return getFixedSequenceStreamActivation(stream, arity);
  }

  bool selectorVisitsEveryLaneOnce(mlir::Value selector, unsigned arity) {
    auto activation = getFixedSequenceSelectorActivation(selector, arity);
    return activation && isExactOne(*activation);
  }

  std::optional<mlir::Value> getFixedSequenceActivation(mlir::Value value) {
    auto result = llvm::dyn_cast<mlir::OpResult>(value);
    auto close =
        result ? llvm::dyn_cast<dataflow::DemuxOp>(result.getOwner()) : nullptr;
    if (!close || result.getResultNumber() != 0)
      return std::nullopt;
    auto stream = close.getSel().getDefiningOp<dataflow::StreamOp>();
    auto activations = close.getInput().getDefiningOp<dataflow::InvariantOp>();
    if (!stream || close.getSel() != stream.getPhase() || !activations ||
        activations.getCond() != stream.getPhase())
      return std::nullopt;
    auto activation = getFixedSequenceStreamActivation(stream, std::nullopt);
    if (!activation || *activation != activations.getInit())
      return std::nullopt;
    return *activation;
  }

  bool isUnpackLaneExact(dataflow::UnpackOp unpack, unsigned lane) {
    if (!isExactOne(unpack.getPacked()) || !isExactOne(unpack.getMask()))
      return false;
    auto mask = getKnownUnsigned(unpack.getMask());
    return mask && lane < 64 && ((*mask >> lane) & 1) != 0;
  }

  bool packInputsMatchUnpack(dataflow::PackOp pack) {
    dataflow::UnpackOp unpack;
    for (auto [lane, input] : llvm::enumerate(pack.getInputs())) {
      auto result = llvm::dyn_cast<mlir::OpResult>(input);
      auto candidate =
          result ? llvm::dyn_cast<dataflow::UnpackOp>(result.getOwner())
                 : nullptr;
      if (!candidate || result.getResultNumber() != lane)
        return false;
      if (!unpack)
        unpack = candidate;
      if (candidate != unpack)
        return false;
    }
    return unpack && unpack.getMask() == pack.getMask() &&
           isExactOne(unpack.getPacked()) && isExactOne(unpack.getMask());
  }

  bool isPackExact(dataflow::PackOp pack) {
    if (!isExactOne(pack.getMask()))
      return false;
    if (llvm::all_of(pack.getInputs(),
                     [&](mlir::Value input) { return isExactOne(input); }))
      return true;
    return packInputsMatchUnpack(pack);
  }

  bool isSerializeActivationExact(dataflow::SerializeOp serialize) {
    if (!isExactOne(serialize.getMask()))
      return false;
    dataflow::UnpackOp unpack;
    for (auto [lane, input] : llvm::enumerate(serialize.getInputs())) {
      auto result = llvm::dyn_cast<mlir::OpResult>(input);
      auto candidate =
          result ? llvm::dyn_cast<dataflow::UnpackOp>(result.getOwner())
                 : nullptr;
      if (!candidate || result.getResultNumber() != lane)
        return false;
      if (!unpack)
        unpack = candidate;
      if (candidate != unpack)
        return false;
    }
    return unpack && unpack.getMask() == serialize.getMask() &&
           isExactOne(unpack.getPacked()) && isExactOne(unpack.getMask());
  }

  bool isExactOneWhenSelected(mlir::Value value, mlir::Value selector,
                              unsigned lane) {
    GraphCardinalityAnalysis branch(graph);
    for (mlir::Operation &op : graph.getBody().front().without_terminator()) {
      auto demux = llvm::dyn_cast<dataflow::DemuxOp>(op);
      if (!demux || demux.getSel() != selector ||
          lane >= demux.getOutputs().size() || !isExactOne(demux.getInput()))
        continue;
      branch.exactOneAssumptions.insert(demux.getOutputs()[lane]);
    }
    return branch.isExactOne(value);
  }

  bool availableWhenSelected(mlir::Value value, mlir::Value selector,
                             unsigned lane,
                             llvm::DenseSet<mlir::Value> &visited) {
    if (isExactOne(value))
      return true;
    if (!visited.insert(value).second)
      return false;
    auto result = llvm::dyn_cast<mlir::OpResult>(value);
    mlir::Operation *def = result ? result.getOwner() : nullptr;
    if (!def)
      return false;
    if (auto demux = llvm::dyn_cast<dataflow::DemuxOp>(def))
      if (demux.getSel() == selector)
        return result.getResultNumber() == lane &&
               (isExactOne(demux.getInput()) ||
                isGraphStreamInput(demux.getInput()));
    if (isExactOneWhenSelected(value, selector, lane))
      return true;
    if (llvm::isa<dataflow::StreamOp, dataflow::CarryOp,
                  dataflow::InvariantOp, dataflow::GateOp,
                  dataflow::ParallelizeOp, dataflow::PackOp,
                  dataflow::UnpackOp, dataflow::SerializeOp,
                  dataflow::MuxOp>(def))
      return false;
    if (!llvm::isa<dataflow::CanonicalDataflowActorOpInterface>(def) &&
        !llvm::isa<mlir::memref::CastOp,
                   mlir::UnrealizedConversionCastOp>(def))
      return false;
    return llvm::all_of(def->getOperands(), [&](mlir::Value operand) {
      llvm::DenseSet<mlir::Value> branchVisited = visited;
      return availableWhenSelected(operand, selector, lane, branchVisited);
    });
  }

  bool availableWhenSelected(mlir::Value value, mlir::Value selector,
                             unsigned lane) {
    llvm::DenseSet<mlir::Value> visited;
    return availableWhenSelected(value, selector, lane, visited);
  }

  bool availableWhenSelectedAndAligned(
      mlir::Value value, mlir::Value selector, unsigned lane,
      mlir::Value phase, mlir::Value assumption, bool truePhaseOnly,
      llvm::DenseSet<mlir::Value> &visited) {
    if (!visited.insert(value).second)
      return false;
    auto result = llvm::dyn_cast<mlir::OpResult>(value);
    mlir::Operation *def = result ? result.getOwner() : nullptr;
    if (!def)
      return false;
    if (auto demux = llvm::dyn_cast<dataflow::DemuxOp>(def)) {
      if (demux.getSel() != selector || result.getResultNumber() != lane)
        return false;
      if (isGraphStreamInput(demux.getInput())) {
        llvm::DenseSet<mlir::Value> selectorVisited = visited;
        return isAligned(selector, phase, assumption, truePhaseOnly,
                         selectorVisited);
      }
      llvm::DenseSet<mlir::Value> inputVisited = visited;
      return isAligned(demux.getInput(), phase, assumption, truePhaseOnly,
                       inputVisited);
    }
    if (llvm::isa<dataflow::StreamOp, dataflow::CarryOp,
                  dataflow::InvariantOp, dataflow::GateOp,
                  dataflow::ParallelizeOp, dataflow::PackOp,
                  dataflow::UnpackOp, dataflow::SerializeOp,
                  dataflow::MuxOp>(def))
      return false;
    if (!llvm::isa<dataflow::CanonicalDataflowActorOpInterface>(def) &&
        !llvm::isa<mlir::memref::CastOp,
                   mlir::UnrealizedConversionCastOp>(def))
      return false;
    bool hasSelectedOperand = false;
    for (mlir::Value operand : def->getOperands()) {
      if (isMemoryCapabilityType(operand.getType()))
        continue;
      llvm::DenseSet<mlir::Value> branchVisited = visited;
      if (availableWhenSelectedAndAligned(
              operand, selector, lane, phase, assumption, truePhaseOnly,
              branchVisited)) {
        hasSelectedOperand = true;
        continue;
      }
      llvm::DenseSet<mlir::Value> alignedVisited = visited;
      if (!isAligned(operand, phase, assumption, truePhaseOnly,
                     alignedVisited))
        return false;
    }
    return hasSelectedOperand;
  }

  bool isAligned(mlir::Value value, mlir::Value phase,
                 mlir::Value assumption, bool truePhaseOnly,
                 llvm::DenseSet<mlir::Value> &visited) {
    if (value == assumption || alignedCarryAssumptions.contains(value))
      return true;
    if (value == phase)
      return !truePhaseOnly;
    if (!visited.insert(value).second)
      return truePhaseOnly &&
             llvm::any_of(alignedCarryAssumptions,
                          [&](mlir::Value carryOutput) {
                            return causallyDependsOn(value, carryOutput);
                          });
    auto result = llvm::dyn_cast<mlir::OpResult>(value);
    mlir::Operation *def = result ? result.getOwner() : nullptr;
    if (!def)
      return false;
    if (truePhaseOnly)
      if (auto activation = getFixedSequenceActivation(value)) {
        llvm::DenseSet<mlir::Value> activationVisited = visited;
        return isAligned(*activation, phase, assumption,
                         /*truePhaseOnly=*/true, activationVisited);
      }
    if (truePhaseOnly)
      if (auto demux = llvm::dyn_cast<dataflow::DemuxOp>(def);
          demux && isGraphStreamInput(demux.getInput()))
        if (auto activation = getFixedSequenceSelectorActivation(
                demux.getSel(), demux.getOutputs().size())) {
          llvm::DenseSet<mlir::Value> activationVisited = visited;
          return isAligned(*activation, phase, assumption,
                           /*truePhaseOnly=*/true, activationVisited);
        }
    if (auto sync = llvm::dyn_cast<dataflow::SyncOp>(def)) {
      bool hasAlignedInput = false;
      bool hasAlignedActivation = false;
      unsigned directStreamInputs = 0;
      for (mlir::Value input : sync.getInputs()) {
        llvm::DenseSet<mlir::Value> inputVisited = visited;
        if (isAligned(input, phase, assumption, truePhaseOnly, inputVisited)) {
          hasAlignedInput = true;
          hasAlignedActivation |= llvm::isa<mlir::NoneType>(input.getType());
          continue;
        }
        if (!isGraphStreamInput(input))
          return false;
        ++directStreamInputs;
      }
      if (directStreamInputs == 0)
        return hasAlignedInput;
      return directStreamInputs == 1 && hasAlignedActivation;
    }
    if (auto stream = llvm::dyn_cast<dataflow::StreamOp>(def))
      return truePhaseOnly && value == stream.getIv() &&
             phase == stream.getPhase();
    if (auto invariant = llvm::dyn_cast<dataflow::InvariantOp>(def))
      return !truePhaseOnly && value == invariant.getOutput() &&
             invariant.getCond() == phase &&
             isExactOne(invariant.getInit());
    if (auto carry = llvm::dyn_cast<dataflow::CarryOp>(def)) {
      if (truePhaseOnly || value != carry.getOutput() ||
          carry.getCond() != phase ||
          !isExactOne(carry.getInit()))
        return false;
      bool inserted = alignedCarryAssumptions.insert(carry.getOutput()).second;
      if (!inserted)
        return true;
      llvm::DenseSet<mlir::Value> feedbackVisited = visited;
      bool aligned = isAligned(carry.getCarry(), phase, carry.getOutput(),
                               /*truePhaseOnly=*/true, feedbackVisited);
      alignedCarryAssumptions.erase(carry.getOutput());
      return aligned;
    }
    if (auto gate = llvm::dyn_cast<dataflow::GateOp>(def)) {
      if (!truePhaseOnly || value != gate.getAfterValue() ||
          gate.getBeforeCond() != phase)
        return false;
      llvm::DenseSet<mlir::Value> branchVisited = visited;
      return isAligned(gate.getBeforeValue(), phase, assumption,
                       /*truePhaseOnly=*/false, branchVisited);
    }
    if (auto demux = llvm::dyn_cast<dataflow::DemuxOp>(def)) {
      if (!truePhaseOnly || demux.getSel() != phase ||
          result.getResultNumber() != 1)
        return false;
      llvm::DenseSet<mlir::Value> branchVisited = visited;
      return isAligned(demux.getInput(), phase, assumption,
                       /*truePhaseOnly=*/false, branchVisited);
    }
    if (auto mux = llvm::dyn_cast<dataflow::MuxOp>(def)) {
      llvm::DenseSet<mlir::Value> selectorVisited = visited;
      if (!isAligned(mux.getSel(), phase, assumption, truePhaseOnly,
                     selectorVisited))
        return false;
      for (auto [lane, input] : llvm::enumerate(mux.getInputs())) {
        llvm::DenseSet<mlir::Value> branchVisited = visited;
        if (!availableWhenSelectedAndAligned(
                input, mux.getSel(), lane, phase, assumption, truePhaseOnly,
                branchVisited))
          return false;
      }
      return true;
    }
    if (llvm::isa<dataflow::StreamOp, dataflow::GateOp,
                  dataflow::ParallelizeOp, dataflow::PackOp,
                  dataflow::UnpackOp, dataflow::SerializeOp,
                  dataflow::DemuxOp>(def))
      return false;
    if (!llvm::isa<dataflow::CanonicalDataflowActorOpInterface>(def) &&
        !llvm::isa<mlir::memref::CastOp,
                   mlir::UnrealizedConversionCastOp>(def))
      return false;

    bool hasRequiredOperand = false;
    for (mlir::Value operand : def->getOperands()) {
      if (isMemoryCapabilityType(operand.getType()))
        continue;
      llvm::DenseSet<mlir::Value> requiredVisited = visited;
      if (isAligned(operand, phase, assumption, truePhaseOnly,
                    requiredVisited)) {
        hasRequiredOperand = true;
        continue;
      }
      if (!truePhaseOnly)
        return false;
      llvm::DenseSet<mlir::Value> fullVisited = visited;
      if (!isAligned(operand, phase, assumption,
                     /*truePhaseOnly=*/false, fullVisited))
        return false;
    }
    return hasRequiredOperand;
  }

  void collectAlignedCarries(mlir::Value phase,
                             llvm::SmallVectorImpl<dataflow::CarryOp> &carries) {
    for (mlir::Operation &op : graph.getBody().front().without_terminator()) {
      auto carry = llvm::dyn_cast<dataflow::CarryOp>(op);
      if (carry && carry.getCond() == phase && isExactOne(carry.getInit()))
        carries.push_back(carry);
    }
  }

  bool isCarrySystemAligned(mlir::Value phase) {
    auto known = alignedCarrySystems.find(phase);
    if (known != alignedCarrySystems.end())
      return known->second;
    if (!alignedCarrySystemsActive.insert(phase).second)
      return true;

    llvm::SmallVector<dataflow::CarryOp, 4> carries;
    collectAlignedCarries(phase, carries);
    for (dataflow::CarryOp carry : carries)
      alignedCarryAssumptions.insert(carry.getOutput());

    bool aligned = llvm::all_of(carries, [&](dataflow::CarryOp carry) {
      llvm::DenseSet<mlir::Value> visited;
      return isAligned(carry.getCarry(), phase, carry.getOutput(),
                       /*truePhaseOnly=*/true, visited);
    });

    for (dataflow::CarryOp carry : carries)
      alignedCarryAssumptions.erase(carry.getOutput());
    alignedCarrySystemsActive.erase(phase);
    alignedCarrySystems.try_emplace(phase, aligned);
    return aligned;
  }

  bool isPhaseAligned(mlir::Value value, mlir::Value phase) {
    if (!isCarrySystemAligned(phase))
      return false;
    llvm::SmallVector<dataflow::CarryOp, 4> carries;
    collectAlignedCarries(phase, carries);
    for (dataflow::CarryOp carry : carries)
      alignedCarryAssumptions.insert(carry.getOutput());
    llvm::DenseSet<mlir::Value> visited;
    bool aligned =
        isAligned(value, phase, {}, /*truePhaseOnly=*/false, visited);
    for (dataflow::CarryOp carry : carries)
      alignedCarryAssumptions.erase(carry.getOutput());
    return aligned;
  }

  bool isOneClosePhase(mlir::Value value) {
    auto known = oneClosePhase.find(value);
    if (known != oneClosePhase.end())
      return known->second;
    if (!oneCloseActive.insert(value).second)
      return false;
    bool result = false;
    if (auto stream = value.getDefiningOp<dataflow::StreamOp>()) {
      result = value == stream.getPhase() &&
               isExactOne(stream.getInit()) &&
               isExactOne(stream.getLimit()) &&
               isExactOne(stream.getStep());
    } else if (auto gate = value.getDefiningOp<dataflow::GateOp>()) {
      result = value == gate.getAfterCond() &&
               isOneClosePhase(gate.getBeforeCond()) &&
               isPhaseAligned(gate.getBeforeValue(), gate.getBeforeCond());
    } else if (auto serialize = value.getDefiningOp<dataflow::SerializeOp>()) {
      result = value == serialize.getCont() &&
               isSerializeActivationExact(serialize);
    } else if (auto sync = value.getDefiningOp<dataflow::SyncOp>()) {
      unsigned closeInputs = 0;
      result = true;
      for (mlir::Value input : sync.getInputs()) {
        if (isOneClosePhase(input)) {
          ++closeInputs;
          continue;
        }
        if (!isExactOne(input)) {
          result = false;
          break;
        }
      }
      result &= closeInputs == 1;
    }
    oneCloseActive.erase(value);
    oneClosePhase.try_emplace(value, result);
    return result;
  }

  bool computeExactOne(mlir::Value value) {
    if (exactOneAssumptions.contains(value))
      return true;
    if (auto argument = llvm::dyn_cast<mlir::BlockArgument>(value)) {
      if (argument.getOwner() != &graph.getBody().front())
        return false;
      if (argument.getArgNumber() == 0)
        return true;
      return graph.getInputPortKind(argument.getArgNumber() - 1) !=
             dataflow::GraphPortKind::Stream;
    }

    auto result = llvm::dyn_cast<mlir::OpResult>(value);
    mlir::Operation *def = result ? result.getOwner() : nullptr;
    if (!def)
      return false;
    if (auto constant = llvm::dyn_cast<dataflow::ConstantOp>(def))
      return isExactOne(constant.getCtrl());
    if (auto sync = llvm::dyn_cast<dataflow::SyncOp>(def)) {
      bool hasExactActivation = false;
      unsigned directStreamInputs = 0;
      for (mlir::Value input : sync.getInputs()) {
        if (isExactOne(input)) {
          hasExactActivation |= llvm::isa<mlir::NoneType>(input.getType());
          continue;
        }
        if (!isGraphStreamInput(input))
          return false;
        ++directStreamInputs;
      }
      return directStreamInputs == 0 ||
             (directStreamInputs == 1 && hasExactActivation);
    }
    if (auto mux = llvm::dyn_cast<dataflow::MuxOp>(def)) {
      if (!isExactOne(mux.getSel()))
        return false;
      for (auto [lane, input] : llvm::enumerate(mux.getInputs()))
        if (!availableWhenSelected(input, mux.getSel(), lane))
          return false;
      return true;
    }
    if (auto demux = llvm::dyn_cast<dataflow::DemuxOp>(def)) {
      unsigned lane = result.getResultNumber();
      if (auto selected = getKnownUnsigned(demux.getSel()))
        return *selected == lane && isExactOne(demux.getInput());
      if (isGraphStreamInput(demux.getInput()) &&
          selectorVisitsEveryLaneOnce(demux.getSel(),
                                      demux.getOutputs().size()))
        return true;
      return lane == 0 && isOneClosePhase(demux.getSel()) &&
             isPhaseAligned(demux.getInput(), demux.getSel());
    }
    if (auto unpack = llvm::dyn_cast<dataflow::UnpackOp>(def))
      return isUnpackLaneExact(unpack, result.getResultNumber());
    if (auto pack = llvm::dyn_cast<dataflow::PackOp>(def))
      return isPackExact(pack);
    if (llvm::isa<dataflow::StreamOp, dataflow::CarryOp,
                  dataflow::InvariantOp, dataflow::GateOp,
                  dataflow::ParallelizeOp, dataflow::SerializeOp>(def))
      return false;
    if (llvm::isa<mlir::memref::AllocOp>(def))
      return true;
    if (auto cast = llvm::dyn_cast<mlir::memref::CastOp>(def))
      return isExactOne(cast.getSource());
    if (auto cast = llvm::dyn_cast<mlir::UnrealizedConversionCastOp>(def))
      return llvm::all_of(cast.getInputs(),
                          [&](mlir::Value input) { return isExactOne(input); });
    if (llvm::isa<dataflow::CanonicalDataflowActorOpInterface>(def))
      return allOperandsExact(def);
    return false;
  }
};

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
    if (llvm::any_of(op->getResultTypes(), containsChannelType)) {
      structuralError = graphError(
          llvm::Twine("finalized graph contains channel-typed result produced "
                      "by '") +
          op->getName().getStringRef() + "'");
      return mlir::WalkResult::interrupt();
    }
    if (llvm::any_of(op->getOperandTypes(), containsChannelType)) {
      structuralError = graphError(
          llvm::Twine("finalized graph contains channel-typed operand consumed "
                      "by '") +
          op->getName().getStringRef() + "'");
      return mlir::WalkResult::interrupt();
    }
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
    if (!dataflow::isCanonicalDataflowActor(op) &&
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

  GraphCardinalityAnalysis cardinality(graph);
  for (auto [index, value] : llvm::enumerate(ret.getValues()))
    if (!cardinality.isExactOne(value))
      return graphError(llvm::Twine("graph @") + graph.getSymName() +
                        " value output #" + llvm::Twine(index) +
                        " is not statically exact-one");
  for (auto [index, stream] : llvm::enumerate(ret.getStreams()))
    if (!cardinality.hasProvenStreamCommit(stream))
      return graphError(llvm::Twine("graph @") + graph.getSymName() +
                        " stream output #" + llvm::Twine(index) +
                        " has no statically proven close/commit");
  for (auto [index, witness] : llvm::enumerate(ret.getComplete()))
    if (!cardinality.isExactOne(witness))
      return graphError(llvm::Twine("graph @") + graph.getSymName() +
                        " completion witness #" + llvm::Twine(index) +
                        " is not statically one-shot");

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
