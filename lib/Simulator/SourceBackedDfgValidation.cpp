#include "Simulator/SourceBackedDfgValidation.h"

#include "SimulationWireInternal.h"

#include "Dataflow/IR/DataflowOps.h"
#include "Dataflow/IR/OperationSchemaCodec.h"
#include "Simulator/DFGSimulator.h"
#include "Simulator/NativeSimulationOracle.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <limits>
#include <optional>
#include <system_error>
#include <utility>
#include <vector>

namespace loom::sim {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      llvm::Twine("source_backed_dfg_validation_invalid: ") + message);
}

llvm::Error unsupported(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::not_supported),
      llvm::Twine("source_backed_dfg_validation_unsupported: ") + message);
}

llvm::Error executionFailed(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::io_error),
      llvm::Twine("source_backed_dfg_validation_execution_failed: ") + message);
}

llvm::Expected<std::uint64_t> fixedTypeByteCount(mlir::Operation *scope,
                                                 mlir::Type type) {
  llvm::TypeSize bytes = mlir::DataLayout::closest(scope).getTypeSize(type);
  if (bytes.isScalable() || bytes.getFixedValue() == 0)
    return unsupported("selected value has no fixed nonzero storage size");
  return bytes.getFixedValue();
}

bool fitsStorageExtent(const detail::LaneShape &shape,
                       std::uint64_t storageBytes) {
  if (shape.lanesPerToken == 0 || shape.laneBitWidth == 0 ||
      shape.lanesPerToken >
          std::numeric_limits<std::uint64_t>::max() / shape.laneBitWidth)
    return false;
  const std::uint64_t semanticBits = shape.lanesPerToken * shape.laneBitWidth;
  const std::uint64_t semanticBytes =
      semanticBits / 8 + static_cast<std::uint64_t>(semanticBits % 8 != 0);
  return semanticBytes <= storageBytes;
}

CanonicalValueSequence exceptionalValue(const detail::LaneShape &shape,
                                        SemanticState state) {
  CanonicalValueSequence value;
  value.tokenCount = 1;
  value.lanes.reserve(shape.lanesPerToken);
  for (std::uint64_t lane = 0; lane < shape.lanesPerToken; ++lane)
    value.lanes.push_back(state == SemanticState::Poison
                              ? SemanticLane::poison()
                              : SemanticLane::undef());
  return value;
}

llvm::Expected<llvm::APInt> attributeBits(mlir::Attribute attribute,
                                          std::uint32_t width) {
  if (auto integer = llvm::dyn_cast<mlir::IntegerAttr>(attribute)) {
    if (integer.getValue().getBitWidth() != width)
      return invalid("constant integer width differs from graph input");
    return integer.getValue();
  }
  if (auto floating = llvm::dyn_cast<mlir::FloatAttr>(attribute)) {
    llvm::APInt bits = floating.getValue().bitcastToAPInt();
    if (bits.getBitWidth() != width)
      return invalid("constant floating width differs from graph input");
    return bits;
  }
  return unsupported("constant graph input is not integer or floating");
}

llvm::Expected<std::optional<CanonicalValueSequence>>
fixedValueOf(mlir::Value value, const detail::LaneShape &shape) {
  if (value.getDefiningOp<mlir::LLVM::UndefOp>())
    return std::optional<CanonicalValueSequence>(
        exceptionalValue(shape, SemanticState::Undef));
  if (value.getDefiningOp<mlir::LLVM::PoisonOp>())
    return std::optional<CanonicalValueSequence>(
        exceptionalValue(shape, SemanticState::Poison));

  mlir::Attribute attribute;
  if (auto constant = value.getDefiningOp<mlir::arith::ConstantOp>())
    attribute = constant.getValue();
  else if (auto constant = value.getDefiningOp<mlir::LLVM::ConstantOp>())
    attribute = constant.getValue();
  if (!attribute)
    return std::optional<CanonicalValueSequence>{};

  CanonicalValueSequence result;
  result.tokenCount = 1;
  result.lanes.reserve(shape.lanesPerToken);
  if (shape.lanesPerToken == 1 && !llvm::isa<mlir::ElementsAttr>(attribute)) {
    auto bits = attributeBits(attribute, shape.laneBitWidth);
    if (!bits)
      return bits.takeError();
    result.lanes.push_back(SemanticLane::defined(std::move(*bits)));
    return std::optional<CanonicalValueSequence>(std::move(result));
  }

  auto dense = llvm::dyn_cast<mlir::DenseElementsAttr>(attribute);
  if (!dense || dense.getNumElements() < 0 ||
      static_cast<std::uint64_t>(dense.getNumElements()) != shape.lanesPerToken)
    return unsupported("vector constant does not match graph input shape");
  if (llvm::isa<mlir::IntegerType>(dense.getElementType())) {
    for (const llvm::APInt &bits : dense.getValues<llvm::APInt>()) {
      if (bits.getBitWidth() != shape.laneBitWidth)
        return invalid("vector integer lane width differs from graph input");
      result.lanes.push_back(SemanticLane::defined(bits));
    }
  } else if (llvm::isa<mlir::FloatType>(dense.getElementType())) {
    for (const llvm::APFloat &value : dense.getValues<llvm::APFloat>()) {
      llvm::APInt bits = value.bitcastToAPInt();
      if (bits.getBitWidth() != shape.laneBitWidth)
        return invalid("vector floating lane width differs from graph input");
      result.lanes.push_back(SemanticLane::defined(std::move(bits)));
    }
  } else {
    return unsupported("vector constant element is not bit-valued");
  }
  return std::optional<CanonicalValueSequence>(std::move(result));
}

llvm::Error requireSameCanonicalType(mlir::Type graphType,
                                     mlir::Type boundaryType) {
  auto graph = dataflow::encodeCanonicalType(graphType);
  if (!graph)
    return graph.takeError();
  auto boundary = dataflow::encodeCanonicalType(boundaryType);
  if (!boundary)
    return boundary.takeError();
  if (graph->bytes() != boundary->bytes())
    return invalid("selected boundary type differs from graph ABI type");
  return llvm::Error::success();
}

llvm::Expected<mlir::Value>
boundaryValueForThreadFormal(detail::ResolvedLaunchContext &context,
                             mlir::Value graphBinding,
                             mlir::ValueRange selectedBoundary) {
  auto argument = llvm::dyn_cast<mlir::BlockArgument>(graphBinding);
  if (!argument ||
      argument.getOwner()->getParentOp() != context.thread.getOperation() ||
      argument.getArgNumber() >=
          context.thread.getFunctionType().getNumInputs())
    return unsupported("graph input is not a selected thread formal");
  if (argument.getArgNumber() >= selectedBoundary.size())
    return invalid("thread formal exceeds the selected boundary");
  return selectedBoundary[argument.getArgNumber()];
}

llvm::Expected<WorkloadBackedSimulationInputCapturePlan>
deriveCapturePlan(const dataflow::CanonicalDataflowProgramView &view,
                  dataflow::RootedGraphLaunchRef launch,
                  const frontend::PreparedSpatialOwnershipSelection &prepared) {
  auto context = detail::resolveLaunchContext(view, launch);
  if (!context)
    return context.takeError();
  if (context->numStreamInputs != 0 || context->numStreamOutputs != 0)
    return unsupported(
        "source-backed Structured capture does not yet bind graph streams");
  if (prepared.liveOuts.size() != context->numValueResults)
    return invalid("selected live-out count differs from graph results");

  WorkloadBackedSimulationInputCapturePlan plan{launch, {}, {}, {}};
  plan.valueInputs.reserve(context->numValueInputs);
  for (std::uint64_t ordinal = 0; ordinal < context->numValueInputs;
       ++ordinal) {
    auto boundary = boundaryValueForThreadFormal(
        *context, context->graphLaunchOp.getValueInputs()[ordinal],
        prepared.liveIns);
    if (!boundary)
      return boundary.takeError();
    mlir::Type graphType = context->graphOp.getFunctionType().getInput(ordinal);
    if (llvm::Error error =
            requireSameCanonicalType(graphType, boundary->getType()))
      return std::move(error);
    const detail::LaneShape &shape = context->valueInputShapes[ordinal];
    auto fixed = fixedValueOf(*boundary, shape);
    if (!fixed)
      return fixed.takeError();
    std::uint64_t byteCount = 0;
    if (!*fixed) {
      auto bytes = fixedTypeByteCount(boundary->getDefiningOp()
                                          ? boundary->getDefiningOp()
                                          : prepared.operation,
                                      boundary->getType());
      if (!bytes)
        return bytes.takeError();
      if (!fitsStorageExtent(shape, *bytes))
        return invalid("graph input does not fit selected storage extent");
      byteCount = *bytes;
    }
    plan.valueInputs.push_back(SimulationValueInputCapture{
        ordinal, std::nullopt, *boundary, shape.lanesPerToken,
        shape.laneBitWidth, byteCount, std::move(*fixed)});
  }

  plan.valueResults.reserve(context->numValueResults);
  for (std::uint64_t ordinal = 0; ordinal < context->numValueResults;
       ++ordinal) {
    mlir::Value boundary = prepared.liveOuts[ordinal];
    mlir::Type graphType =
        context->graphOp.getFunctionType().getResult(ordinal);
    if (llvm::Error error =
            requireSameCanonicalType(graphType, boundary.getType()))
      return std::move(error);
    const detail::LaneShape &shape = context->valueResultShapes[ordinal];
    auto bytes =
        fixedTypeByteCount(boundary.getDefiningOp() ? boundary.getDefiningOp()
                                                    : prepared.operation,
                           boundary.getType());
    if (!bytes)
      return bytes.takeError();
    if (!fitsStorageExtent(shape, *bytes))
      return invalid("graph result does not fit selected storage extent");
    plan.valueResults.push_back(SimulationValueResultCapture{
        ordinal, boundary, shape.lanesPerToken, shape.laneBitWidth, *bytes});
  }

  plan.memoryRoots.reserve(context->importedRoots.size());
  for (dataflow::LogicalMemoryRootRef root : context->importedRoots) {
    auto resolved = view.resolve(root);
    if (!resolved)
      return resolved.takeError();
    if (resolved->op != context->thread.getOperation() ||
        !resolved->formalArgIndex ||
        *resolved->formalArgIndex >= prepared.liveIns.size())
      return invalid("imported memory root has no selected thread formal");
    mlir::Value boundary = prepared.liveIns[*resolved->formalArgIndex];
    if (!llvm::isa<mlir::LLVM::LLVMPointerType>(boundary.getType()))
      return invalid("imported memory root is not pointer-valued");
    plan.memoryRoots.push_back({root, boundary});
  }
  return plan;
}

RuntimeMemoryObject capturedMemoryObject(llvm::ArrayRef<std::uint8_t> bytes) {
  RuntimeMemoryObject object;
  object.initialBytes.reserve(bytes.size());
  for (std::uint8_t byte : bytes)
    object.initialBytes.push_back({SemanticState::Defined, byte});
  return object;
}

bool sameLane(const SemanticLane &lhs, const SemanticLane &rhs) {
  return lhs.state == rhs.state &&
         (lhs.state != SemanticState::Defined || lhs.bits == rhs.bits);
}

bool sameValue(const CanonicalValueSequence &lhs,
               const CanonicalValueSequence &rhs) {
  return lhs.tokenCount == rhs.tokenCount &&
         lhs.lanes.size() == rhs.lanes.size() &&
         llvm::equal(lhs.lanes, rhs.lanes, sameLane);
}

std::optional<std::size_t>
findMemoryRoot(const WorkloadBackedSimulationInputCapturePlan &plan,
               dataflow::LogicalMemoryRootRef root) {
  for (auto [ordinal, entry] : llvm::enumerate(plan.memoryRoots))
    if (entry.root == root)
      return ordinal;
  return std::nullopt;
}

llvm::Expected<bool>
compareObservations(const SpatialSimulationWorkload &workload,
                    const SpatialFunctionalObservations &observations,
                    const WorkloadBackedSimulationInputCapturePlan &plan,
                    const NativeSimulationCallCapture &native) {
  if (workload.observableContract.valueResults.size() !=
          plan.valueResults.size() ||
      observations.valueResults.size() != plan.valueResults.size() ||
      native.valueResults.size() != plan.valueResults.size())
    return executionFailed("DFG value-result projection is not total");
  for (auto [position, expected] : llvm::enumerate(native.valueResults)) {
    const auto *published =
        std::get_if<PublishedValueResult>(&observations.valueResults[position]);
    if (!published)
      return false;
    if (!sameValue(published->value, expected))
      return false;
  }

  if (observations.memories.size() !=
      workload.observableContract.memories.size())
    return executionFailed("DFG memory observation projection is not total");
  for (auto [observable, payload] : llvm::zip_equal(
           workload.observableContract.memories, observations.memories)) {
    const auto *rootOrView =
        std::get_if<dataflow::LogicalMemoryRootOrViewRef>(&observable.target);
    if (!rootOrView)
      return executionFailed("source-backed replay observed a memory exposure");
    const auto *root = std::get_if<dataflow::LogicalMemoryRootRef>(rootOrView);
    const auto *full = std::get_if<FullMemoryObservation>(&payload);
    if (!root || !full)
      return executionFailed("source-backed replay did not return root state");
    std::optional<std::size_t> rootOrdinal = findMemoryRoot(plan, *root);
    if (!rootOrdinal ||
        *rootOrdinal >= native.memoryRootObjectOrdinals.size() ||
        *rootOrdinal >= native.memoryRootByteOffsets.size())
      return executionFailed("native capture has no memory-root binding");
    const std::uint64_t objectOrdinal =
        native.memoryRootObjectOrdinals[*rootOrdinal];
    if (objectOrdinal >= native.objects.size())
      return executionFailed("native memory root names an absent object");
    const std::uint64_t byteOffset = native.memoryRootByteOffsets[*rootOrdinal];
    llvm::ArrayRef<std::uint8_t> expected(
        native.objects[objectOrdinal].finalBytes);
    if (byteOffset > expected.size() ||
        full->bytes.size() > expected.size() - byteOffset)
      return executionFailed("DFG memory observation exceeds native object");
    expected = expected.slice(byteOffset, full->bytes.size());
    for (auto [actual, byte] : llvm::zip_equal(full->bytes, expected))
      if (actual.state != SemanticState::Defined || actual.value != byte)
        return false;
  }
  return true;
}

llvm::Expected<CanonicalSimulationWorkload>
finalizeReplayWorkload(const WorkloadBackedSimulationInputCapturePlan &plan,
                       const dataflow::CanonicalDataflowProgramView &view) {
  SpatialSimulationWorkload draft{plan.launch};
  for (const SimulationValueInputCapture &input : plan.valueInputs) {
    if (input.valueInputOrdinal != draft.valueInputPlan.size())
      return invalid("graph value input capture is not dense");
    if (input.fixedValue)
      draft.valueInputPlan.push_back(*input.fixedValue);
    else
      draft.valueInputPlan.push_back(RuntimeValueInput{});
  }
  for (const SimulationValueResultCapture &result : plan.valueResults) {
    if (result.valueResultOrdinal !=
        draft.observableContract.valueResults.size())
      return invalid("graph value result capture is not dense");
    draft.observableContract.valueResults.push_back(result.valueResultOrdinal);
  }
  for (const WorkloadBackedMemoryRootCapture &root : plan.memoryRoots)
    draft.observableContract.memories.push_back(
        SpatialMemoryObservable{dataflow::LogicalMemoryRootOrViewRef{root.root},
                                MemoryObservationForm::FullState});
  if (draft.observableContract.valueResults.empty() &&
      draft.observableContract.memories.empty())
    return unsupported(
        "selected graph has no externally observable value or memory effect");
  return finalizeSimulationWorkload(draft, view);
}

} // namespace

llvm::Expected<SourceBackedDfgValidationResult> validateSourceBackedDfgReplay(
    const frontend::StructuredProgramCandidate &sourceProgram,
    const frontend::SpatialOwnershipScope &scope,
    const frontend::SpatialOwnershipDecisionPoint &decision,
    const frontend::MaterializedOwnershipCandidate &candidate,
    const CanonicalSimulationWorkload &workload,
    const CanonicalSimulationRuntimeInput &runtimeInput,
    std::uint64_t maxEventSteps) {
  auto prepared = frontend::prepareSpatialOwnershipSelection(sourceProgram,
                                                             scope, decision);
  if (!prepared)
    return prepared.takeError();
  auto view = candidate.canonicalDataflow.view();
  if (!view)
    return view.takeError();
  std::vector<dataflow::RootedGraphLaunchRef> launches;
  view->forEachRootedGraphLaunch([&](dataflow::RootedGraphLaunchRef launch) {
    launches.push_back(launch);
  });
  if (launches.size() != 1)
    return launches.empty()
               ? invalid("ownership candidate has no rooted graph launch")
               : unsupported(
                     "one ownership candidate has multiple rooted launches");

  auto plan = deriveCapturePlan(*view, launches.front(), *prepared);
  if (!plan)
    return plan.takeError();
  auto replayWorkload = finalizeReplayWorkload(*plan, *view);
  if (!replayWorkload)
    return replayWorkload.takeError();
  auto capture = executeWorkloadBackedSimulationInputCapture(
      std::move(prepared->module), prepared->operation, *plan, sourceProgram,
      workload, runtimeInput);
  if (!capture)
    return capture.takeError();
  if (capture->entryResult != 0)
    return executionFailed("source workload entry returned a nonzero status");
  if (capture->calls.empty())
    return SourceBackedDfgValidationResult{
        SourceBackedDfgValidationStatus::Inapplicable};

  SourceBackedDfgValidationResult result{
      SourceBackedDfgValidationStatus::Equivalent};
  for (const NativeSimulationCallCapture &call : capture->calls) {
    if (call.memoryRootObjectOrdinals.size() != plan->memoryRoots.size() ||
        call.memoryRootByteOffsets.size() != plan->memoryRoots.size())
      return executionFailed("native memory-root capture is not total");
    SpatialSimulationRuntimeInputDraft draft{replayWorkload->identity()};
    draft.runtimeValues = call.runtimeValues;
    draft.memoryObjects.reserve(call.objects.size());
    for (const NativeCapturedMemoryObject &object : call.objects)
      draft.memoryObjects.push_back(capturedMemoryObject(object.initialBytes));
    for (auto [ordinal, root] : llvm::enumerate(plan->memoryRoots))
      draft.memoryRootBindings.push_back(RuntimeMemoryBindingDraft{
          root.root, call.memoryRootObjectOrdinals[ordinal],
          call.memoryRootByteOffsets[ordinal]});
    auto replayInput =
        finalizeSimulationRuntimeInput(draft, *replayWorkload, *view);
    if (!replayInput)
      return replayInput.takeError();
    auto execution =
        simulateRetiredDfgWorkload(candidate.canonicalDataflow, *replayWorkload,
                                   *replayInput, maxEventSteps);
    if (!execution)
      return execution.takeError();
    auto equivalent = compareObservations(*replayWorkload->spatial(),
                                          execution->observations, *plan, call);
    if (!equivalent)
      return equivalent.takeError();
    if (!*equivalent)
      result.status = SourceBackedDfgValidationStatus::Mismatch;
    if (result.dynamicActivations ==
            std::numeric_limits<std::uint64_t>::max() ||
        result.wavefrontSteps > std::numeric_limits<std::uint64_t>::max() -
                                    execution->report.wavefrontSteps ||
        result.eventCount > std::numeric_limits<std::uint64_t>::max() -
                                execution->report.eventCount)
      return executionFailed("source-backed replay accounting overflowed");
    ++result.dynamicActivations;
    result.wavefrontSteps += execution->report.wavefrontSteps;
    result.eventCount += execution->report.eventCount;
  }
  return result;
}

} // namespace loom::sim
