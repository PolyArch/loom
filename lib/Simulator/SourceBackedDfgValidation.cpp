#include "Simulator/SourceBackedDfgValidation.h"

#include "SimulationPointerCapture.h"
#include "SimulationWireInternal.h"
#include "StructuredProgramNativeExecutionInternal.h"

#include "Common/IndexWidth.h"
#include "Dataflow/IR/DataflowOps.h"
#include "Dataflow/IR/OperationSchemaCodec.h"
#include "Frontend/IR/LoomOps.h"
#include "Simulator/DFGSimulator.h"
#include "Simulator/NativeSimulationOracle.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <chrono>
#include <cstdint>
#include <limits>
#include <map>
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

llvm::Error executionLimit(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::timed_out),
      llvm::Twine("source_backed_dfg_validation_execution_limit: ") + message);
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
                                          mlir::Type semanticType,
                                          std::uint32_t width) {
  if (auto integer = llvm::dyn_cast<mlir::IntegerAttr>(attribute)) {
    if (semanticType.isIndex()) {
      const llvm::APInt &stored = integer.getValue();
      if (!stored.isIntN(width) && !stored.isSignedIntN(width))
        return invalid("constant index is not representable at graph width");
      return stored.sextOrTrunc(width);
    }
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
    auto bits = attributeBits(attribute, value.getType(), shape.laneBitWidth);
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

llvm::Expected<std::vector<mlir::Value>> deriveDenseCoordinateBoundaries(
    const frontend::PreparedSpatialOwnershipSelection &prepared) {
  if (!prepared.sourceInductions)
    return std::vector<mlir::Value>{};
  auto forall = llvm::dyn_cast<mlir::scf::ForallOp>(prepared.operation);
  if (!forall || prepared.sourceInductions->size() !=
                     static_cast<std::size_t>(forall.getRank()))
    return invalid("thread coordinate source binding is not total");

  std::vector<mlir::Value> coordinates;
  coordinates.reserve(forall.getRank());
  mlir::OpBuilder builder = mlir::OpBuilder::atBlockBegin(forall.getBody());
  for (auto [dimension, sourceInduction] :
       llvm::enumerate(forall.getInductionVars())) {
    const auto &binding = (*prepared.sourceInductions)[dimension];
    auto resolveInput = [&](std::optional<std::uint64_t> ordinal)
        -> llvm::Expected<std::optional<mlir::Value>> {
      if (!ordinal)
        return std::optional<mlir::Value>{};
      if (*ordinal >= prepared.liveIns.size())
        return invalid("source induction input exceeds the selected boundary");
      mlir::Value value = prepared.liveIns[*ordinal];
      if (!value.getType().isIndex())
        return invalid("source induction input is not index-typed");
      return std::optional<mlir::Value>(value);
    };
    auto lower = resolveInput(binding.lowerInputOrdinal);
    if (!lower)
      return lower.takeError();
    auto step = resolveInput(binding.stepInputOrdinal);
    if (!step)
      return step.takeError();

    mlir::Value coordinate = sourceInduction;
    if (*lower || *step) {
      auto sourceWidth = ::loom::getIndexBitWidth(forall);
      if (!sourceWidth)
        return sourceWidth.takeError();
      if (*sourceWidth == 0 || *sourceWidth > 64)
        return invalid("coordinate recovery requires a 32-bit or 64-bit "
                       "selected index ABI");
      mlir::IntegerType wideType = builder.getIntegerType(*sourceWidth * 2);
      auto widen = [&](mlir::Value value) {
        return mlir::arith::IndexCastOp::create(builder, forall.getLoc(),
                                                wideType, value)
            .getResult();
      };
      coordinate = widen(sourceInduction);
      if (*lower)
        coordinate = mlir::arith::SubIOp::create(builder, forall.getLoc(),
                                                 coordinate, widen(**lower));
      if (*step)
        coordinate = mlir::arith::DivSIOp::create(builder, forall.getLoc(),
                                                  coordinate, widen(**step));
      coordinate = mlir::arith::IndexCastOp::create(
          builder, forall.getLoc(), builder.getIndexType(), coordinate);
    }
    coordinates.push_back(coordinate);
  }
  return coordinates;
}

llvm::Expected<mlir::Value> boundaryValueForThreadFormal(
    detail::ResolvedLaunchContext &context, mlir::Value graphBinding,
    const frontend::PreparedSpatialOwnershipSelection &prepared,
    mlir::ValueRange denseCoordinates) {
  auto argument = llvm::dyn_cast<mlir::BlockArgument>(graphBinding);
  if (!argument ||
      argument.getOwner()->getParentOp() != context.thread.getOperation())
    return unsupported("graph input is not a selected thread formal");
  const std::uint64_t inputCount =
      context.thread.getFunctionType().getNumInputs();
  const std::uint64_t argumentOrdinal = argument.getArgNumber();
  if (argumentOrdinal < inputCount) {
    if (argumentOrdinal >= prepared.liveIns.size())
      return invalid("thread formal exceeds the selected boundary");
    return prepared.liveIns[argumentOrdinal];
  }
  if (argumentOrdinal == inputCount)
    return invalid("graph value input is bound to the thread control token");

  const std::uint64_t coordinateOrdinal = argumentOrdinal - inputCount - 1;
  if (coordinateOrdinal >= denseCoordinates.size())
    return invalid("thread coordinate has no exact source induction binding");
  return denseCoordinates[coordinateOrdinal];
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

  auto denseCoordinates = deriveDenseCoordinateBoundaries(prepared);
  if (!denseCoordinates)
    return denseCoordinates.takeError();
  if (denseCoordinates->size() != context->threadRank)
    return invalid("selected dense coordinate rank differs from graph ABI");

  WorkloadBackedSimulationInputCapturePlan plan{launch, {}, {}, {}, {}};
  plan.denseCoordinates.reserve(denseCoordinates->size());
  for (auto [dimension, coordinate] : llvm::enumerate(*denseCoordinates)) {
    auto bytes = fixedTypeByteCount(prepared.operation, coordinate.getType());
    if (!bytes)
      return bytes.takeError();
    plan.denseCoordinates.push_back(
        {static_cast<std::uint64_t>(dimension), coordinate, *bytes});
  }
  plan.valueInputs.reserve(context->numValueInputs);
  for (std::uint64_t ordinal = 0; ordinal < context->numValueInputs;
       ++ordinal) {
    auto boundary = boundaryValueForThreadFormal(
        *context, context->graphLaunchOp.getValueInputs()[ordinal], prepared,
        *denseCoordinates);
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
        shape.laneBitWidth, byteCount, std::move(*fixed), std::nullopt});
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
    auto threadSource =
        capture_detail::threadMemorySourceForRoot(*resolved, *context);
    if (!threadSource)
      return threadSource.takeError();
    mlir::Value boundary = *threadSource;
    if (auto argument = llvm::dyn_cast<mlir::BlockArgument>(boundary)) {
      if (argument.getArgNumber() >= prepared.liveIns.size())
        return invalid("imported memory root exceeds the selected boundary");
      boundary = prepared.liveIns[argument.getArgNumber()];
    }
    if (!llvm::isa<mlir::LLVM::LLVMPointerType>(boundary.getType()))
      return invalid("imported memory root is not pointer-valued");
    plan.memoryRoots.push_back({root, boundary});
  }
  for (std::uint64_t valueOrdinal = 0; valueOrdinal < context->numValueInputs;
       ++valueOrdinal) {
    auto projection = capture_detail::pointerValueTargetForInput(view, *context,
                                                                 valueOrdinal);
    if (!projection)
      return projection.takeError();
    if (!*projection)
      continue;
    if (plan.valueInputs[valueOrdinal].fixedValue)
      return unsupported(
          "fixed first-class pointer inputs have no runtime object binding");
    plan.valueInputs[valueOrdinal].pointerTarget = (*projection)->target;
  }
  return plan;
}

struct SelectedActivationCapture final {
  mlir::OwningOpRef<mlir::ModuleOp> module;
  mlir::Operation *spatial = nullptr;
  WorkloadBackedSimulationInputCapturePlan plan;
};

llvm::Expected<SelectedActivationCapture> deriveSelectedActivationCapture(
    const frontend::MaterializedOwnershipCandidate &candidate,
    dataflow::RootedGraphLaunchRef launch, std::uint64_t expectedRank) {
  mlir::OwningOpRef<mlir::ModuleOp> module(llvm::cast<mlir::ModuleOp>(
      candidate.structuredProgram.module()->clone()));
  dataflow::ThreadOp thread;
  loom::SpatialRegionOp spatial;
  bool duplicateThread = false;
  bool duplicateSpatial = false;
  module->walk([&](mlir::Operation *operation) {
    if (auto value = llvm::dyn_cast<dataflow::ThreadOp>(operation)) {
      duplicateThread |= static_cast<bool>(thread);
      thread = value;
    }
    if (auto value = llvm::dyn_cast<loom::SpatialRegionOp>(operation)) {
      duplicateSpatial |= static_cast<bool>(spatial);
      spatial = value;
    }
  });
  if (!thread || !spatial || duplicateThread || duplicateSpatial)
    return invalid("selected candidate has no unique thread/spatial carrier");
  if (thread.getBody().empty())
    return invalid("selected thread has no body");
  const std::uint64_t inputCount = thread.getFunctionType().getNumInputs();
  mlir::Block &entry = thread.getBody().front();
  if (entry.getNumArguments() != inputCount + 1 + expectedRank)
    return invalid("selected thread coordinate ABI differs from its source");

  WorkloadBackedSimulationInputCapturePlan plan{launch, {}, {}, {}, {}};
  plan.denseCoordinates.reserve(expectedRank);
  for (std::uint64_t dimension = 0; dimension < expectedRank; ++dimension) {
    mlir::Value coordinate = entry.getArgument(inputCount + 1 + dimension);
    auto byteCount = fixedTypeByteCount(spatial, coordinate.getType());
    if (!byteCount)
      return byteCount.takeError();
    plan.denseCoordinates.push_back({dimension, coordinate, *byteCount});
  }
  return SelectedActivationCapture{std::move(module), spatial, std::move(plan)};
}

RuntimeMemoryObject
capturedMemoryObject(llvm::ArrayRef<std::uint8_t> bytes,
                     llvm::ArrayRef<RuntimeMemoryPointer> pointers) {
  RuntimeMemoryObject object;
  object.initialBytes.reserve(bytes.size());
  for (std::uint8_t byte : bytes)
    object.initialBytes.push_back({SemanticState::Defined, byte});
  object.pointerValues.assign(pointers.begin(), pointers.end());
  return object;
}

const MemoryRootBindingEntry *
findMemoryBinding(const SpatialSimulationRuntimeInput &input,
                  dataflow::LogicalMemoryRootRef root) {
  for (const MemoryRootBindingEntry &binding : input.memoryRootBindings)
    if (binding.root == root)
      return &binding;
  return nullptr;
}

llvm::Expected<std::vector<RuntimeMemoryObject>> canonicalizeNativeFinalObjects(
    const NativeSimulationCallCapture &capture,
    const WorkloadBackedSimulationInputCapturePlan &plan,
    const SpatialSimulationRuntimeInput &runtimeInput,
    mlir::Operation *layoutScope) {
  if (capture.objects.size() != runtimeInput.memoryObjects.size())
    return executionFailed(
        "native and DFG runtime object tables have different sizes");
  std::vector<std::optional<std::uint64_t>> canonicalOrdinal(
      capture.objects.size());
  for (auto [rootOrdinal, root] : llvm::enumerate(plan.memoryRoots)) {
    if (rootOrdinal >= capture.memoryRootObjectOrdinals.size())
      return executionFailed("native memory-root capture is not total");
    const MemoryRootBindingEntry *binding =
        findMemoryBinding(runtimeInput, root.root);
    if (!binding)
      return executionFailed("DFG runtime input lost an imported root");
    const std::uint64_t authorObject =
        capture.memoryRootObjectOrdinals[rootOrdinal];
    if (authorObject >= canonicalOrdinal.size() ||
        binding->binding.objectOrdinal >= canonicalOrdinal.size())
      return executionFailed("memory-root object ordinal is out of range");
    if (canonicalOrdinal[authorObject] &&
        *canonicalOrdinal[authorObject] != binding->binding.objectOrdinal)
      return executionFailed(
          "aliased roots disagree on canonical object identity");
    canonicalOrdinal[authorObject] = binding->binding.objectOrdinal;
  }
  if (llvm::any_of(canonicalOrdinal,
                   [](const auto &ordinal) { return !ordinal.has_value(); }))
    return executionFailed("native capture contains an unreferenced object");

  std::vector<RuntimeMemoryObject> objects(capture.objects.size());
  std::vector<bool> assigned(capture.objects.size(), false);
  for (auto [authorOrdinal, native] : llvm::enumerate(capture.objects)) {
    const std::uint64_t canonical = *canonicalOrdinal[authorOrdinal];
    if (assigned[canonical])
      return executionFailed("native objects collapse to one canonical slot");
    assigned[canonical] = true;
    RuntimeMemoryObject &object = objects[canonical];
    object.initialBytes.reserve(native.finalBytes.size());
    for (std::uint8_t byte : native.finalBytes)
      object.initialBytes.push_back({SemanticState::Defined, byte});
    object.pointerValues = native.finalPointers;
    for (RuntimeMemoryPointer &pointer : object.pointerValues) {
      if (pointer.target.objectOrdinal >= canonicalOrdinal.size())
        return executionFailed("native final pointer target is out of range");
      pointer.target.objectOrdinal =
          *canonicalOrdinal[pointer.target.objectOrdinal];
    }
  }
  if (llvm::Error error =
          detail::canonicalizeRuntimeMemoryPointers(objects, layoutScope))
    return std::move(error);
  return objects;
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
                    const NativeSimulationCallCapture &native,
                    const SpatialSimulationRuntimeInput &runtimeInput,
                    llvm::ArrayRef<RuntimeMemoryObject> nativeFinalObjects,
                    SourceBackedDfgValidationResult &result) {
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
    result.valueLanesCompared += published->value.lanes.size();
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
    if (!rootOrdinal)
      return executionFailed("native capture has no memory-root binding");
    const MemoryRootBindingEntry *binding =
        findMemoryBinding(runtimeInput, *root);
    if (!binding || binding->binding.objectOrdinal >= nativeFinalObjects.size())
      return executionFailed("DFG runtime input has no memory-root binding");
    const RuntimeMemoryObject &expectedObject =
        nativeFinalObjects[binding->binding.objectOrdinal];
    const std::uint64_t byteOffset = binding->binding.byteOffset;
    llvm::ArrayRef<SemanticMemoryByte> expected(expectedObject.initialBytes);
    if (byteOffset > expected.size() ||
        full->bytes.size() > expected.size() - byteOffset)
      return executionFailed("DFG memory observation exceeds native object");
    expected = expected.slice(byteOffset, full->bytes.size());
    for (auto [actual, byte] : llvm::zip_equal(full->bytes, expected)) {
      if (actual.state != byte.state ||
          (actual.state == SemanticState::Defined &&
           actual.value != byte.value))
        return false;
    }
    result.memoryBytesCompared += full->bytes.size();
  }
  return true;
}

llvm::Expected<CanonicalSimulationWorkload>
finalizeReplayWorkload(const WorkloadBackedSimulationInputCapturePlan &plan,
                       llvm::ArrayRef<std::uint64_t> denseCoordinates,
                       const dataflow::CanonicalDataflowProgramView &view) {
  if (denseCoordinates.size() != plan.denseCoordinates.size())
    return executionFailed("native dense-coordinate capture is not total");
  SpatialSimulationWorkload draft{plan.launch};
  draft.denseCoordinates.assign(denseCoordinates.begin(),
                                denseCoordinates.end());
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
    llvm::ArrayRef<frontend::StructuredExecutionShapeDecision>
        executionShapeDecisions,
    const frontend::MaterializedOwnershipCandidate &candidate,
    const CanonicalSimulationWorkload &workload,
    const CanonicalSimulationRuntimeInput &runtimeInput,
    SourceBackedDfgValidationLimits limits,
    const NativeStructuredProgramObservations *sourceObservations) {
  if (limits.maxWavefrontSteps == 0 || limits.maxEventCount == 0 ||
      limits.maxRetainedCaptureBytes == 0 ||
      limits.maxSimulationWallTime <
          std::chrono::steady_clock::duration::zero())
    return invalid("execution limits must be positive");
  auto prepared = frontend::prepareSpatialOwnershipSelection(sourceProgram,
                                                             scope, decision);
  if (!prepared)
    return prepared.takeError();
  if (executionShapeDecisions.size() > 1)
    return invalid("source replay has multiple fmuladd shape decisions");
  if (!executionShapeDecisions.empty())
    raising::materializeFMulAddInOperation(
        *prepared->operation, executionShapeDecisions.front().fmuladdShape);
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
  auto preparedDfg =
      prepareDfgExecution(candidate.canonicalDataflow, launches.front());
  if (!preparedDfg)
    return preparedDfg.takeError();
  auto launchContext = detail::resolveLaunchContext(*view, launches.front());
  if (!launchContext)
    return launchContext.takeError();

  auto plan = deriveCapturePlan(*view, launches.front(), *prepared);
  if (!plan)
    return plan.takeError();
  auto selected = deriveSelectedActivationCapture(
      candidate, launches.front(), plan->denseCoordinates.size());
  if (!selected)
    return selected.takeError();

  std::map<std::vector<std::uint64_t>, std::uint64_t> selectedActivations;
  std::uint64_t selectedSummaryBytes = 0;
  auto recordSelectedActivation =
      [&](NativeSimulationCallCapture &&call) -> llvm::Error {
    if (!call.runtimeValues.empty() || !call.valueResults.empty() ||
        !call.objects.empty() || !call.memoryRootObjectOrdinals.empty() ||
        !call.memoryRootByteOffsets.empty())
      return executionFailed(
          "selected activation capture contains non-coordinate state");
    auto found = selectedActivations.find(call.denseCoordinates);
    if (found == selectedActivations.end()) {
      const std::uint64_t keyBytes =
          sizeof(std::uint64_t) * (2 + call.denseCoordinates.size());
      if (selectedSummaryBytes >
              std::numeric_limits<std::uint64_t>::max() - keyBytes ||
          selectedSummaryBytes + keyBytes > limits.maxRetainedCaptureBytes)
        return executionLimit(
            "selected activation summary exceeded the execution limit");
      selectedSummaryBytes += keyBytes;
      selectedActivations.emplace(std::move(call.denseCoordinates), 1);
      return llvm::Error::success();
    }
    if (found->second == std::numeric_limits<std::uint64_t>::max())
      return executionFailed("selected activation count overflowed");
    ++found->second;
    return llvm::Error::success();
  };
  auto selectedObservations =
      native_detail::visitProjectedWorkloadBackedSimulationInputCaptures(
          std::move(selected->module), selected->spatial, selected->plan,
          sourceProgram, workload, runtimeInput, limits.maxRetainedCaptureBytes,
          recordSelectedActivation);
  if (!selectedObservations)
    return selectedObservations.takeError();

  std::uint64_t sourceActivationCount = 0;
  bool activationMismatch = false;
  SourceBackedDfgValidationResult result;
  result.status = SourceBackedDfgValidationStatus::Equivalent;
  std::chrono::steady_clock::duration remainingSimulationWallTime =
      limits.maxSimulationWallTime;
  auto replayActivation =
      [&](NativeSimulationCallCapture &&call) -> llvm::Error {
    if (result.wavefrontSteps >= limits.maxWavefrontSteps)
      return executionLimit("aggregate wavefront budget exhausted");
    auto replayWorkload =
        finalizeReplayWorkload(*plan, call.denseCoordinates, *view);
    if (!replayWorkload)
      return replayWorkload.takeError();
    if (call.memoryRootObjectOrdinals.size() != plan->memoryRoots.size() ||
        call.memoryRootByteOffsets.size() != plan->memoryRoots.size())
      return executionFailed("native memory-root capture is not total");
    SpatialSimulationRuntimeInputDraft draft{replayWorkload->identity()};
    draft.runtimeValues = call.runtimeValues;
    draft.memoryObjects.reserve(call.objects.size());
    for (const NativeCapturedMemoryObject &object : call.objects)
      draft.memoryObjects.push_back(
          capturedMemoryObject(object.initialBytes, object.initialPointers));
    for (auto [ordinal, root] : llvm::enumerate(plan->memoryRoots))
      draft.memoryRootBindings.push_back(RuntimeMemoryBindingDraft{
          root.root, call.memoryRootObjectOrdinals[ordinal],
          call.memoryRootByteOffsets[ordinal]});
    auto replayInput =
        finalizeSimulationRuntimeInput(draft, *replayWorkload, *view);
    if (!replayInput)
      return replayInput.takeError();
    auto nativeFinalObjects = canonicalizeNativeFinalObjects(
        call, *plan, *replayInput->spatial(), launchContext->graphOp);
    if (!nativeFinalObjects)
      return nativeFinalObjects.takeError();
    if (remainingSimulationWallTime ==
        std::chrono::steady_clock::duration::zero())
      return executionLimit("aggregate simulation wall-time budget exhausted");
    const auto started = std::chrono::steady_clock::now();
    std::optional<std::chrono::steady_clock::time_point> executionDeadline;
    if (remainingSimulationWallTime !=
        std::chrono::steady_clock::duration::max())
      executionDeadline = started + remainingSimulationWallTime;
    auto execution = simulateRetiredDfgWorkload(
        *preparedDfg, *replayWorkload, *replayInput,
        limits.maxWavefrontSteps - result.wavefrontSteps, executionDeadline);
    const auto stopped = std::chrono::steady_clock::now();
    const auto elapsed = stopped - started;
    result.simulationSeconds += std::chrono::duration<double>(elapsed).count();
    if (remainingSimulationWallTime !=
        std::chrono::steady_clock::duration::max())
      remainingSimulationWallTime =
          elapsed >= remainingSimulationWallTime
              ? std::chrono::steady_clock::duration::zero()
              : remainingSimulationWallTime - elapsed;
    auto accountExecution =
        [&](const DFGSimulationReport &report) -> llvm::Error {
      if (report.eventCount > limits.maxEventCount - result.eventCount)
        return executionLimit("aggregate event budget exhausted");
      if (result.dynamicActivations ==
              std::numeric_limits<std::uint64_t>::max() ||
          result.wavefrontSteps > std::numeric_limits<std::uint64_t>::max() -
                                      report.wavefrontSteps ||
          result.eventCount >
              std::numeric_limits<std::uint64_t>::max() - report.eventCount)
        return executionFailed("source-backed replay accounting overflowed");
      ++result.dynamicActivations;
      result.wavefrontSteps += report.wavefrontSteps;
      result.eventCount += report.eventCount;
      for (const auto &[schema, count] : report.operationFireCounts) {
        std::uint64_t &aggregate = result.operationFireCounts[schema];
        if (aggregate > std::numeric_limits<std::uint64_t>::max() - count)
          return executionFailed(
              "source-backed operation firing count overflowed");
        aggregate += count;
      }
      return llvm::Error::success();
    };
    if (!execution) {
      return llvm::handleErrors(
          execution.takeError(),
          [&](const NonRetiredDFGExecutionError &failure) -> llvm::Error {
            if (llvm::Error error = accountExecution(failure.report()))
              return error;
            result.status = SourceBackedDfgValidationStatus::Mismatch;
            return llvm::Error::success();
          });
    }
    auto equivalent = compareObservations(
        *replayWorkload->spatial(), execution->observations, *plan, call,
        *replayInput->spatial(), *nativeFinalObjects, result);
    if (!equivalent)
      return equivalent.takeError();
    if (!*equivalent)
      result.status = SourceBackedDfgValidationStatus::Mismatch;
    return accountExecution(execution->report);
  };
  std::optional<llvm::Error> deferredReplayFailure;
  auto censusAndReplay =
      [&](NativeSimulationCallCapture &&call) -> llvm::Error {
    if (sourceActivationCount == std::numeric_limits<std::uint64_t>::max())
      return executionFailed("source activation count overflowed");
    ++sourceActivationCount;
    auto selectedActivation = selectedActivations.find(call.denseCoordinates);
    if (selectedActivation == selectedActivations.end() ||
        selectedActivation->second == 0) {
      activationMismatch = true;
      return llvm::Error::success();
    }
    --selectedActivation->second;
    if (deferredReplayFailure)
      return llvm::Error::success();
    if (llvm::Error error = replayActivation(std::move(call)))
      deferredReplayFailure.emplace(std::move(error));
    return llvm::Error::success();
  };
  if (llvm::Error error = visitWorkloadBackedSimulationInputCaptures(
          std::move(prepared->module), prepared->operation, *plan,
          sourceProgram, workload, runtimeInput, limits.maxRetainedCaptureBytes,
          censusAndReplay)) {
    if (deferredReplayFailure)
      return llvm::joinErrors(std::move(error),
                              std::move(*deferredReplayFailure));
    return std::move(error);
  }
  activationMismatch |= llvm::any_of(
      selectedActivations, [](const auto &entry) { return entry.second != 0; });
  const bool wholeProgramMismatch =
      sourceObservations && !haveEquivalentFunctionalObservations(
                                *sourceObservations, *selectedObservations);
  if (activationMismatch || wholeProgramMismatch) {
    if (deferredReplayFailure)
      llvm::consumeError(std::move(*deferredReplayFailure));
    SourceBackedDfgValidationResult mismatch;
    mismatch.status = SourceBackedDfgValidationStatus::Mismatch;
    mismatch.dynamicActivations = sourceActivationCount;
    return mismatch;
  }
  if (deferredReplayFailure)
    return std::move(*deferredReplayFailure);
  if (result.dynamicActivations == 0 &&
      result.status == SourceBackedDfgValidationStatus::Equivalent)
    result.status = SourceBackedDfgValidationStatus::Inapplicable;
  if (sourceObservations)
    result.sourceReturnValue = sourceObservations->returnValue;
  return result;
}

} // namespace loom::sim
