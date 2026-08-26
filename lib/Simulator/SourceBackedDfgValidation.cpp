#include "Simulator/SourceBackedDfgValidation.h"

#include "Common/ArtifactLocalReference.h"

#include "SimulationPointerCapture.h"
#include "SimulationWireInternal.h"
#include "StructuredProgramNativeExecutionInternal.h"

#include "Dataflow/IR/DataflowOps.h"
#include "Dataflow/IR/OperationSchemaCodec.h"
#include "Frontend/IR/LoomOps.h"
#include "Simulator/DFGSimulator.h"
#include "Simulator/NativeSimulationOracle.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <chrono>
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

llvm::Expected<WorkloadBackedStreamCapture>
deriveStreamCapture(std::uint64_t ordinal, mlir::BlockArgument channel,
                    const detail::LaneShape &shape, mlir::Type graphType,
                    bool input, mlir::Operation *scope) {
  auto channelType = llvm::dyn_cast<dataflow::ChannelType>(channel.getType());
  if (!channelType)
    return invalid("selected stream boundary is not channel-typed");
  if (shape.pointerLayout)
    return unsupported(
        "source-backed pointer streams have no runtime object binding");
  if (llvm::Error error =
          requireSameCanonicalType(graphType, channelType.getElementType()))
    return std::move(error);
  auto byteCount = fixedTypeByteCount(scope, channelType.getElementType());
  if (!byteCount)
    return byteCount.takeError();
  if (!fitsStorageExtent(shape, *byteCount))
    return invalid("graph stream token does not fit selected storage extent");

  WorkloadBackedStreamCapture stream{
      ordinal, shape.lanesPerToken, shape.laneBitWidth, *byteCount, {}};
  for (mlir::OpOperand &use : channel.getUses()) {
    mlir::Operation *endpoint = use.getOwner();
    if (input && llvm::isa<dataflow::ChannelReceiveOp>(endpoint) &&
        use.getOperandNumber() == 0) {
      stream.endpoints.push_back(endpoint);
      continue;
    }
    if (!input && llvm::isa<dataflow::ChannelSendOp>(endpoint) &&
        use.getOperandNumber() == 0) {
      stream.endpoints.push_back(endpoint);
      continue;
    }
    return invalid("selected stream boundary has a non-endpoint use");
  }
  if (stream.endpoints.empty())
    return invalid("selected stream boundary has no endpoint");
  return stream;
}

llvm::Expected<WorkloadBackedSimulationInputCapturePlan>
deriveSelectedCapturePlan(const dataflow::CanonicalDataflowProgramView &view,
                          dataflow::RootedGraphLaunchRef launch,
                          dataflow::ThreadOp thread,
                          loom::SpatialRegionOp spatial) {
  auto context = detail::resolveLaunchContext(view, launch);
  if (!context)
    return context.takeError();
  if (spatial.getValueInputs().size() != context->numValueInputs ||
      spatial.getValueResults().size() != context->numValueResults ||
      spatial.getStreamInputs().size() != context->numStreamInputs ||
      spatial.getStreamOutputs().size() != context->numStreamOutputs)
    return invalid("selected Spatial boundary differs from graph ABI");
  if (thread.getBody().empty() || spatial.getBody().empty())
    return invalid("selected Spatial carrier has no body");
  const std::uint64_t inputCount = thread.getFunctionType().getNumInputs();
  if (inputCount != context->thread.getFunctionType().getNumInputs())
    return invalid("selected thread value ABI differs from graph owner");
  mlir::Block &threadEntry = thread.getBody().front();
  if (threadEntry.getNumArguments() < inputCount + 1)
    return invalid("selected thread has no control boundary");
  const std::uint64_t rank = threadEntry.getNumArguments() - inputCount - 1;
  if (rank != context->threadRank)
    return invalid("selected thread coordinate ABI differs from graph ABI");

  WorkloadBackedSimulationInputCapturePlan plan{launch, {}, {}, {}, {}, {}, {}};
  plan.denseCoordinates.reserve(rank);
  for (std::uint64_t dimension = 0; dimension < rank; ++dimension) {
    mlir::Value coordinate =
        threadEntry.getArgument(inputCount + 1 + dimension);
    auto byteCount = fixedTypeByteCount(spatial, coordinate.getType());
    if (!byteCount)
      return byteCount.takeError();
    plan.denseCoordinates.push_back({dimension, coordinate, *byteCount});
  }

  plan.valueInputs.reserve(context->numValueInputs);
  for (std::uint64_t ordinal = 0; ordinal < context->numValueInputs;
       ++ordinal) {
    mlir::Value boundary = spatial.getValueInputs()[ordinal];
    mlir::Type graphType = context->graphOp.getFunctionType().getInput(ordinal);
    if (llvm::Error error =
            requireSameCanonicalType(graphType, boundary.getType()))
      return std::move(error);
    const detail::LaneShape &shape = context->valueInputShapes[ordinal];
    auto fixed = fixedValueOf(boundary, shape);
    if (!fixed)
      return fixed.takeError();
    std::uint64_t byteCount = 0;
    if (!*fixed) {
      auto bytes = fixedTypeByteCount(
          boundary.getDefiningOp() ? boundary.getDefiningOp() : spatial,
          boundary.getType());
      if (!bytes)
        return bytes.takeError();
      if (!fitsStorageExtent(shape, *bytes))
        return invalid("graph input does not fit selected storage extent");
      byteCount = *bytes;
    }
    plan.valueInputs.push_back({ordinal, std::nullopt, boundary,
                                shape.lanesPerToken, shape.laneBitWidth,
                                byteCount, std::move(*fixed), std::nullopt,
                                false,
                                std::nullopt});
  }

  plan.valueResults.reserve(context->numValueResults);
  for (std::uint64_t ordinal = 0; ordinal < context->numValueResults;
       ++ordinal) {
    mlir::Value boundary = spatial.getValueResults()[ordinal];
    mlir::Type graphType =
        context->graphOp.getFunctionType().getResult(ordinal);
    if (llvm::Error error =
            requireSameCanonicalType(graphType, boundary.getType()))
      return std::move(error);
    const detail::LaneShape &shape = context->valueResultShapes[ordinal];
    auto bytes = fixedTypeByteCount(spatial, boundary.getType());
    if (!bytes)
      return bytes.takeError();
    if (!fitsStorageExtent(shape, *bytes))
      return invalid("graph result does not fit selected storage extent");
    plan.valueResults.push_back(
        {ordinal, boundary, shape.lanesPerToken, shape.laneBitWidth, *bytes});
  }

  plan.memoryRoots.reserve(context->importedRoots.size());
  for (dataflow::LogicalMemoryRootRef root : context->importedRoots) {
    auto resolved = view.resolve(root);
    if (!resolved)
      return resolved.takeError();
    auto source =
        capture_detail::threadMemorySourceForRoot(*resolved, *context);
    if (!source)
      return source.takeError();
    mlir::Value boundary;
    if (auto argument = llvm::dyn_cast<mlir::BlockArgument>(*source)) {
      if (argument.getOwner() != &context->thread.getBody().front() ||
          argument.getArgNumber() >= inputCount)
        return invalid("memory root has a non-input thread formal");
      boundary = threadEntry.getArgument(argument.getArgNumber());
    } else {
      for (auto [ordinal, input] :
           llvm::enumerate(context->graphLaunchOp.getValueInputs()))
        if (input == *source)
          boundary = spatial.getValueInputs()[ordinal];
      for (auto [ordinal, input] :
           llvm::enumerate(context->graphLaunchOp.getMemoryInputs()))
        if (!boundary && input == *source &&
            ordinal < spatial.getMemoryInputs().size())
          boundary = spatial.getMemoryInputs()[ordinal];
      if (!boundary)
        return invalid("memory root has no selected Spatial boundary");
    }
    if (!llvm::isa<mlir::LLVM::LLVMPointerType>(boundary.getType()))
      return invalid("imported memory root is not pointer-valued");
    plan.memoryRoots.push_back({root, boundary});
  }
  for (std::uint64_t ordinal = 0; ordinal < context->numValueInputs;
       ++ordinal) {
    auto projection =
        capture_detail::pointerValueTargetForInput(view, *context, ordinal);
    if (!projection)
      return projection.takeError();
    if (!*projection)
      continue;
    if (plan.valueInputs[ordinal].fixedValue)
      return unsupported(
          "fixed first-class pointer inputs have no runtime object binding");
    plan.valueInputs[ordinal].pointerTarget = (*projection)->target;
  }

  mlir::Block &spatialEntry = spatial.getBody().front();
  std::uint64_t argumentOrdinal = spatial.getValueInputs().size();
  plan.streamInputs.reserve(context->numStreamInputs);
  for (std::uint64_t ordinal = 0; ordinal < context->numStreamInputs;
       ++ordinal) {
    auto stream = deriveStreamCapture(
        ordinal, spatialEntry.getArgument(argumentOrdinal++),
        context->streamInputShapes[ordinal],
        context->graphOp.getFunctionType().getInput(context->numValueInputs +
                                                    ordinal),
        true, spatial);
    if (!stream)
      return stream.takeError();
    plan.streamInputs.push_back(std::move(*stream));
  }
  argumentOrdinal += spatial.getMemoryInputs().size();
  plan.streamOutputs.reserve(context->numStreamOutputs);
  for (std::uint64_t ordinal = 0; ordinal < context->numStreamOutputs;
       ++ordinal) {
    auto stream = deriveStreamCapture(
        ordinal, spatialEntry.getArgument(argumentOrdinal++),
        context->streamOutputShapes[ordinal],
        context->graphOp.getFunctionType().getResult(context->numValueResults +
                                                     ordinal),
        false, spatial);
    if (!stream)
      return stream.takeError();
    plan.streamOutputs.push_back(std::move(*stream));
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
    const dataflow::CanonicalDataflowProgramView &view,
    dataflow::RootedGraphLaunchRef launch,
    frontend::StructuredEntityRef spatialRegion) {
  dataflow::ThreadOp thread;
  loom::SpatialRegionOp spatial;
  auto structuredView = candidate.structuredProgram.view();
  if (!structuredView)
    return structuredView.takeError();
  auto entity = structuredView->resolve(spatialRegion);
  if (!entity)
    return entity.takeError();
  auto sourceSpatial =
      llvm::dyn_cast_or_null<loom::SpatialRegionOp>(entity->operation);
  if (!sourceSpatial)
    return invalid("Spatial projection does not resolve to a region");
  auto sourceThread = sourceSpatial->getParentOfType<dataflow::ThreadOp>();
  if (!sourceThread)
    return invalid("Spatial region has no thread carrier");
  mlir::IRMapping mapping;
  mlir::OwningOpRef<mlir::ModuleOp> module(
      llvm::cast<mlir::ModuleOp>(
          candidate.structuredProgram.module()->clone(mapping)));
  spatial = llvm::dyn_cast_or_null<loom::SpatialRegionOp>(
      mapping.lookupOrNull(sourceSpatial.getOperation()));
  thread = llvm::dyn_cast_or_null<dataflow::ThreadOp>(
      mapping.lookupOrNull(sourceThread.getOperation()));
  if (!thread || !spatial)
    return invalid("Spatial carrier was not mapped into the clone");
  auto plan = deriveSelectedCapturePlan(view, launch, thread, spatial);
  if (!plan)
    return plan.takeError();
  return SelectedActivationCapture{std::move(module), spatial,
                                   std::move(*plan)};
}

llvm::Expected<std::optional<dataflow::RootedGraphLaunchRef>>
selectReachableRootedLaunch(
    const frontend::MaterializedOwnershipCandidate &candidate,
    const dataflow::CanonicalDataflowProgramView &view,
    frontend::StructuredEntityRef spatialRegion,
    llvm::ArrayRef<dataflow::RootThreadLaunchRef> reachableRoots) {
  std::optional<dataflow::StaticGraphLaunchRef> selectedStaticLaunch;
  for (const lowering::StructuredSpatialGraphProjection &projection :
       candidate.spatialGraphs) {
    if (projection.spatialRegion != spatialRegion)
      continue;
    if (selectedStaticLaunch)
      return invalid("Spatial region has duplicate graph projections");
    selectedStaticLaunch = projection.staticGraphLaunch;
  }
  if (!selectedStaticLaunch && !candidate.spatialGraphs.empty())
    return invalid("Spatial region has no graph projection");

  std::optional<dataflow::RootedGraphLaunchRef> selected;
  bool duplicate = false;
  std::uint64_t matchingRootedLaunchCount = 0;
  view.forEachRootedGraphLaunch([&](dataflow::RootedGraphLaunchRef launch) {
    if (selectedStaticLaunch &&
        launch.staticGraphLaunch != *selectedStaticLaunch)
      return;
    ++matchingRootedLaunchCount;
    if (!llvm::is_contained(reachableRoots, launch.rootThreadLaunch))
      return;
    duplicate |= selected.has_value();
    selected = launch;
  });
  if (matchingRootedLaunchCount == 0) {
    if (!selectedStaticLaunch)
      return invalid("unprojected Spatial carrier has no rooted graph launch");
    return invalid(
        "Spatial projection has no matching rooted graph launch: "
        "spatial_projections=" + llvm::Twine(candidate.spatialGraphs.size()) +
        ", projection_artifact_matches=" +
        llvm::Twine(selectedStaticLaunch->artifact == view.identity()) +
        ", projection_matches_artifact=" +
        llvm::Twine(selectedStaticLaunch->artifact ==
                    candidate.canonicalDataflow.identity()) +
        ", view_matches_artifact=" +
        llvm::Twine(view.identity() == candidate.canonicalDataflow.identity()) +
        ", projection_entity=" +
        llvm::Twine(selectedStaticLaunch->entity.value()));
  }
  if (!selected)
    return std::optional<dataflow::RootedGraphLaunchRef>{};
  if (duplicate)
    return unsupported(
        "Spatial graph projection reaches multiple reachable rooted graph "
        "launches");
  return selected;
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

  if (workload.observableContract.streamOutputs.size() !=
          plan.streamOutputs.size() ||
      observations.streamOutputs.size() != plan.streamOutputs.size() ||
      native.streamOutputs.size() != plan.streamOutputs.size())
    return executionFailed("DFG stream-output projection is not total");
  for (auto [actual, expected] :
       llvm::zip_equal(observations.streamOutputs, native.streamOutputs)) {
    if (actual.termination != expected.termination ||
        !sameValue(actual.values, expected.values))
      return false;
    result.valueLanesCompared += actual.values.lanes.size();
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
  for (const WorkloadBackedStreamCapture &stream : plan.streamOutputs) {
    if (stream.graphOrdinal != draft.observableContract.streamOutputs.size())
      return invalid("graph stream output capture is not dense");
    draft.observableContract.streamOutputs.push_back(stream.graphOrdinal);
  }
  for (const WorkloadBackedMemoryRootCapture &root : plan.memoryRoots)
    draft.observableContract.memories.push_back(
        SpatialMemoryObservable{dataflow::LogicalMemoryRootOrViewRef{root.root},
                                MemoryObservationForm::FullState});
  if (draft.observableContract.valueResults.empty() &&
      draft.observableContract.streamOutputs.empty() &&
      draft.observableContract.memories.empty())
    return unsupported(
        "selected graph has no externally observable value, stream, or memory "
        "effect");
  return finalizeSimulationWorkload(draft, view);
}

} // namespace

llvm::Expected<SourceBackedDfgValidationResult> validateSourceBackedDfgReplay(
    const frontend::StructuredProgramCandidate &sourceProgram,
    const frontend::MaterializedOwnershipCandidate &candidate,
    const CanonicalSimulationWorkload &workload,
    const CanonicalSimulationRuntimeInput &runtimeInput,
    SourceBackedDfgValidationLimits limits,
    const NativeStructuredProgramObservations *sourceObservations,
    SourceBackedDfgReplayCasePublisher publishReplayCase) {
  if (limits.maxWavefrontSteps == 0 || limits.maxEventCount == 0 ||
      limits.maxRetainedCaptureBytes == 0 ||
      limits.maxSimulationWallTime <
          std::chrono::steady_clock::duration::zero())
    return invalid("execution limits must be positive");
  if (limits.maxSimulationWallTime ==
      std::chrono::steady_clock::duration::zero())
    return executionLimit("simulation wall-time budget exhausted before "
                          "source capture");
  std::optional<NativeStructuredProgramObservations> ownedSourceObservations;
  if (!sourceObservations) {
    auto observed =
        executeNativeStructuredProgram(sourceProgram, workload, runtimeInput);
    if (!observed)
      return observed.takeError();
    ownedSourceObservations.emplace(std::move(*observed));
    sourceObservations = &*ownedSourceObservations;
  }
  auto view = candidate.canonicalDataflow.view();
  if (!view)
    return view.takeError();
  const StructuredProgramSimulationWorkload *structuredWorkload =
      workload.structuredProgram();
  if (!structuredWorkload)
    return invalid("source-backed replay workload is not Structured");
  auto sourceView = sourceProgram.view();
  if (!sourceView)
    return sourceView.takeError();
  auto entry = sourceView->resolve(structuredWorkload->entryRef);
  if (!entry)
    return entry.takeError();
  auto entryFunction =
      llvm::dyn_cast_or_null<mlir::LLVM::LLVMFuncOp>(entry->operation);
  if (!entryFunction)
    return invalid("source-backed replay entry is not an LLVM function");
  auto reachableRoots = view->projectRootThreadLaunchesReachableFromAbiEntry(
      entryFunction.getSymName());
  if (!reachableRoots)
    return reachableRoots.takeError();

  std::vector<frontend::StructuredEntityRef> spatialRegions;
  spatialRegions.reserve(candidate.spatialGraphs.size());
  for (const lowering::StructuredSpatialGraphProjection &projection :
       candidate.spatialGraphs)
    if (!llvm::is_contained(spatialRegions, projection.spatialRegion))
      spatialRegions.push_back(projection.spatialRegion);
  if (spatialRegions.empty()) {
    auto structuredView = candidate.structuredProgram.view();
    if (!structuredView)
      return structuredView.takeError();
    for (const frontend::StructuredEntity &entity :
         structuredView->entities(frontend::StructuredEntityKind::Operation))
      if (llvm::isa_and_nonnull<loom::SpatialRegionOp>(entity.operation))
        spatialRegions.push_back(entity.reference);
  }
  if (spatialRegions.empty())
    return invalid("ownership candidate has no Spatial graph projection");

  SourceBackedDfgValidationResult result;
  result.status = SourceBackedDfgValidationStatus::Equivalent;
  std::chrono::steady_clock::duration remainingSimulationWallTime =
      limits.maxSimulationWallTime;
  for (frontend::StructuredEntityRef spatialRegion : spatialRegions) {
    auto launch = selectReachableRootedLaunch(candidate, *view, spatialRegion,
                                               *reachableRoots);
    if (!launch)
      return launch.takeError();
    if (!*launch)
      continue;
    auto preparedDfg =
        prepareDfgExecution(candidate.canonicalDataflow, **launch);
    if (!preparedDfg)
      return preparedDfg.takeError();
    auto launchContext = detail::resolveLaunchContext(*view, **launch);
    if (!launchContext)
      return launchContext.takeError();
    auto selected = deriveSelectedActivationCapture(
        candidate, *view, **launch, spatialRegion);
    if (!selected)
      return selected.takeError();
    const WorkloadBackedSimulationInputCapturePlan &replayPlan = selected->plan;
    const std::uint64_t firstRegionActivation = result.dynamicActivations;
    std::uint64_t capturedActivationCount = 0;

    auto replayActivation =
        [&](NativeSimulationCallCapture &&call) -> llvm::Error {
      if (result.wavefrontSteps >= limits.maxWavefrontSteps)
        return executionLimit("aggregate wavefront budget exhausted");
      auto replayWorkload =
          finalizeReplayWorkload(replayPlan, call.denseCoordinates, *view);
      if (!replayWorkload)
        return replayWorkload.takeError();
      if (call.memoryRootObjectOrdinals.size() !=
              replayPlan.memoryRoots.size() ||
          call.memoryRootByteOffsets.size() != replayPlan.memoryRoots.size())
        return executionFailed("native memory-root capture is not total");
      SpatialSimulationRuntimeInputDraft draft{replayWorkload->identity()};
      draft.runtimeValues = call.runtimeValues;
      draft.runtimeStreams = call.runtimeStreams;
      draft.memoryObjects.reserve(call.objects.size());
      for (const NativeCapturedMemoryObject &object : call.objects)
        draft.memoryObjects.push_back(
            capturedMemoryObject(object.initialBytes, object.initialPointers));
      for (auto [ordinal, root] : llvm::enumerate(replayPlan.memoryRoots))
        draft.memoryRootBindings.push_back(RuntimeMemoryBindingDraft{
            root.root, call.memoryRootObjectOrdinals[ordinal],
            call.memoryRootByteOffsets[ordinal]});
      auto replayInput =
          finalizeSimulationRuntimeInput(draft, *replayWorkload, *view);
      if (!replayInput)
        return replayInput.takeError();
      if (publishReplayCase) {
        auto replayCase = publishReplayCase(*replayWorkload, *replayInput);
        if (!replayCase)
          return replayCase.takeError();
        ++result.replayCaseOccurrences;
        result.replayCases.push_back(std::move(*replayCase));
      }
      auto nativeFinalObjects = canonicalizeNativeFinalObjects(
          call, replayPlan, *replayInput->spatial(), launchContext->graphOp);
      if (!nativeFinalObjects)
        return nativeFinalObjects.takeError();
      if (remainingSimulationWallTime ==
          std::chrono::steady_clock::duration::zero())
        return executionLimit(
            "aggregate simulation wall-time budget exhausted");
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
      result.simulationSeconds +=
          std::chrono::duration<double>(elapsed).count();
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
          *replayWorkload->spatial(), execution->observations, replayPlan, call,
          *replayInput->spatial(), *nativeFinalObjects, result);
      if (!equivalent)
        return equivalent.takeError();
      if (!*equivalent)
        result.status = SourceBackedDfgValidationStatus::Mismatch;
      return accountExecution(execution->report);
    };
    std::optional<llvm::Error> deferredReplayFailure;
    auto captureAndReplay =
        [&](NativeSimulationCallCapture &&call) -> llvm::Error {
      if (capturedActivationCount ==
          std::numeric_limits<std::uint64_t>::max())
        return executionFailed("selected activation count overflowed");
      ++capturedActivationCount;
      if (deferredReplayFailure)
        return llvm::Error::success();
      if (llvm::Error error = replayActivation(std::move(call)))
        deferredReplayFailure.emplace(std::move(error));
      return llvm::Error::success();
    };
    auto selectedObservations =
        native_detail::visitProjectedWorkloadBackedSimulationInputCaptures(
            std::move(selected->module), selected->spatial, selected->plan,
            sourceProgram, workload, runtimeInput,
            limits.maxRetainedCaptureBytes, captureAndReplay);
    if (!selectedObservations) {
      if (deferredReplayFailure)
        return llvm::joinErrors(selectedObservations.takeError(),
                                std::move(*deferredReplayFailure));
      return selectedObservations.takeError();
    }
    const bool wholeProgramMismatch =
        sourceObservations && !haveEquivalentFunctionalObservations(
                                  *sourceObservations, *selectedObservations);
    if (wholeProgramMismatch) {
      if (deferredReplayFailure)
        llvm::consumeError(std::move(*deferredReplayFailure));
      result.status = SourceBackedDfgValidationStatus::Mismatch;
      if (capturedActivationCount >
          std::numeric_limits<std::uint64_t>::max() - firstRegionActivation)
        return executionFailed("source-backed activation count overflowed");
      result.dynamicActivations =
          firstRegionActivation + capturedActivationCount;
      return result;
    }
    if (deferredReplayFailure)
      return std::move(*deferredReplayFailure);
    if (result.status == SourceBackedDfgValidationStatus::Mismatch)
      return result;
  }
  if (result.dynamicActivations == 0 &&
      result.status == SourceBackedDfgValidationStatus::Equivalent)
    result.status = SourceBackedDfgValidationStatus::Inapplicable;
  if (sourceObservations)
    result.sourceReturnValue = sourceObservations->returnValue;
  llvm::sort(result.replayCases, [](const auto &lhs, const auto &rhs) {
    if (lhs.workload != rhs.workload)
      return artifactRootReferenceLess(lhs.workload, rhs.workload);
    return artifactRootReferenceLess(lhs.runtimeInput, rhs.runtimeInput);
  });
  result.replayCases.erase(
      std::unique(result.replayCases.begin(), result.replayCases.end()),
      result.replayCases.end());
  return result;
}

} // namespace loom::sim
