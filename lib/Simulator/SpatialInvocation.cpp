#include "Simulator/SpatialInvocation.h"

#include "Dataflow/IR/DataflowOps.h"
#include "Dataflow/IR/DataflowServiceSchema.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <limits>
#include <system_error>
#include <utility>
#include <variant>

namespace loom::sim {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "spatial_invocation_invalid: " + message);
}

llvm::Expected<std::uint32_t>
transportBitCount(SpatialSimulationValueShape shape) {
  if (shape.lanesPerToken == 0 || shape.laneBitWidth == 0 ||
      shape.lanesPerToken >
          std::numeric_limits<std::uint32_t>::max() / shape.laneBitWidth)
    return invalid("boundary shape exceeds the invocation wire domain");
  return static_cast<std::uint32_t>(shape.lanesPerToken) * shape.laneBitWidth;
}

llvm::APInt
unpackLittleEndianBits(const runtime::SpatialInvocationValue &value) {
  llvm::APInt bits(value.bitCount, 0);
  for (std::size_t byte = 0; byte != value.littleEndianBits.size(); ++byte) {
    const std::uint64_t base = byte * 8;
    for (unsigned bit = 0; bit != 8 && base + bit < value.bitCount; ++bit)
      if ((value.littleEndianBits[byte] & (1U << bit)) != 0)
        bits.setBit(static_cast<unsigned>(base + bit));
  }
  return bits;
}

std::vector<std::uint8_t> packLittleEndianBits(const llvm::APInt &bits) {
  const std::size_t byteCount = (bits.getBitWidth() + 7) / 8;
  std::vector<std::uint8_t> bytes;
  bytes.reserve(byteCount);
  for (std::size_t byte = 0; byte != byteCount; ++byte) {
    const unsigned offset = static_cast<unsigned>(byte * 8);
    const unsigned width = std::min<unsigned>(8, bits.getBitWidth() - offset);
    bytes.push_back(
        static_cast<std::uint8_t>(bits.extractBitsAsZExtValue(width, offset)));
  }
  return bytes;
}

llvm::Error
validateInvocationOwner(const runtime::SpatialInvocationWire &wire,
                        const ImportedSpatialSimulationWorkload &workload,
                        const SpatialSimulationWorkload &spatial) {
  std::string wireDiagnostic;
  if (!runtime::validateSpatialInvocationWire(wire, wireDiagnostic))
    return invalid(wireDiagnostic);
  if (wire.canonicalDataflowIdentity != workload.dataflow.identity().bytes())
    return invalid("invocation names a foreign Canonical Dataflow owner");
  if (wire.rootThreadLaunchEntity !=
          spatial.launchRef.rootThreadLaunch.entity.value() ||
      wire.graphLaunchEntity !=
          spatial.launchRef.staticGraphLaunch.entity.value())
    return invalid("invocation names a foreign rooted graph launch");
  if (wire.denseCoordinates != spatial.denseCoordinates)
    return invalid("invocation dense coordinates differ from the workload");
  return llvm::Error::success();
}

llvm::Error
validateResultDestinations(const runtime::SpatialInvocationWire &wire,
                           const SpatialSimulationWorkload &workload,
                           const SpatialSimulationBoundaryShapes &shapes) {
  if (wire.results.size() != workload.observableContract.valueResults.size())
    return invalid("result destinations are not total over observable values");
  struct AddressInterval final {
    std::uint64_t begin = 0;
    std::uint64_t end = 0;
  };
  std::vector<AddressInterval> intervals;
  intervals.reserve(wire.results.size());
  for (std::size_t ordinal = 0; ordinal != wire.results.size(); ++ordinal) {
    const std::uint64_t valueResult =
        workload.observableContract.valueResults[ordinal];
    if (valueResult >= shapes.valueResults.size())
      return invalid("observable value result is outside its boundary shape");
    auto bitCount = transportBitCount(shapes.valueResults[valueResult]);
    if (!bitCount)
      return bitCount.takeError();
    const runtime::SpatialInvocationResultDestination &destination =
        wire.results[ordinal];
    if (destination.bitCount != *bitCount || destination.address == 0)
      return invalid("result destination has the wrong shape or null address");
    const std::uint64_t byteCount =
        (static_cast<std::uint64_t>(destination.bitCount) + 7) / 8;
    if (byteCount >
        std::numeric_limits<std::uint64_t>::max() - destination.address)
      return invalid("result destination address range overflows");
    intervals.push_back({destination.address, destination.address + byteCount});
  }
  llvm::sort(intervals,
             [](const AddressInterval &lhs, const AddressInterval &rhs) {
               return lhs.begin < rhs.begin;
             });
  for (std::size_t ordinal = 1; ordinal < intervals.size(); ++ordinal)
    if (intervals[ordinal - 1].end > intervals[ordinal].begin)
      return invalid("result destination address ranges overlap");
  return llvm::Error::success();
}

} // namespace

llvm::Expected<std::vector<dataflow::LogicalMemoryRootRef>>
projectSpatialInvocationWritableMemoryRoots(
    const dataflow::CanonicalDataflowProgramView &dataflow,
    dataflow::RootedGraphLaunchRef launch) {
  std::vector<dataflow::LogicalMemoryRootRef> roots;
  llvm::Error error = dataflow.forEachContextualServiceActor(
      launch.rootThreadLaunch,
      [&](dataflow::ContextualActorRef actorRef) -> llvm::Error {
        if (actorRef.launch != launch)
          return llvm::Error::success();
        auto actor = dataflow.resolve(actorRef.actor);
        if (!actor)
          return actor.takeError();
        if (llvm::isa<dataflow::FenceOp>(actor->op))
          return llvm::Error::success();
        auto access =
            dataflow::semantics::getCanonicalMemoryAccessView(actor->op);
        if (!access)
          return access.takeError();
        using dataflow::semantics::MemoryAccessOperation;
        if (access->operation() == MemoryAccessOperation::Load)
          return llvm::Error::success();
        auto memory = dataflow.resolveAddressedMemory(actorRef);
        if (!memory)
          return memory.takeError();
        if (const auto *root =
                std::get_if<dataflow::LogicalMemoryRootRef>(&*memory))
          roots.push_back(*root);
        else
          roots.push_back(
              std::get<dataflow::LogicalMemoryViewRef>(*memory).root);
        return llvm::Error::success();
      });
  if (error)
    return std::move(error);
  llvm::sort(roots, [](const auto &lhs, const auto &rhs) {
    return lhs.entity.value() < rhs.entity.value();
  });
  roots.erase(std::unique(roots.begin(), roots.end()), roots.end());
  return roots;
}

llvm::Expected<CanonicalSimulationRuntimeInput>
materializeSpatialInvocationRuntimeInput(
    const ImportedSpatialSimulationWorkload &workload,
    const runtime::SpatialInvocationWire &wire) {
  const SpatialSimulationWorkload *spatial = workload.workload.spatial();
  if (!spatial)
    return invalid("workload lost its Spatial payload");
  if (llvm::Error error = validateInvocationOwner(wire, workload, *spatial))
    return std::move(error);
  auto view = workload.dataflow.view();
  if (!view)
    return view.takeError();
  auto shapes =
      projectSpatialSimulationBoundaryShapes(*view, spatial->launchRef);
  if (!shapes)
    return shapes.takeError();
  auto memoryInputs = view->graphMemoryInputs(spatial->launchRef);
  if (!memoryInputs)
    return memoryInputs.takeError();
  if (!shapes->streamInputs.empty() ||
      !spatial->observableContract.streamOutputs.empty())
    return invalid("dynamic invocation currently admits no stream boundary");
  auto writableRoots =
      projectSpatialInvocationWritableMemoryRoots(*view, spatial->launchRef);
  if (!writableRoots)
    return writableRoots.takeError();
  if (spatial->observableContract.memories.size() != writableRoots->size())
    return invalid("dynamic invocation memory observations are not exact");
  for (std::size_t ordinal = 0; ordinal != writableRoots->size(); ++ordinal) {
    const SpatialMemoryObservable &observable =
        spatial->observableContract.memories[ordinal];
    const auto *role =
        std::get_if<dataflow::LogicalMemoryRootOrViewRef>(&observable.target);
    const auto *root =
        role ? std::get_if<dataflow::LogicalMemoryRootRef>(role) : nullptr;
    if (!root || *root != (*writableRoots)[ordinal] ||
        observable.form != MemoryObservationForm::DiffFromRuntimeInput)
      return invalid("dynamic invocation memory observation differs from "
                     "Dataflow write effects");
  }
  if (wire.values.size() != spatial->valueInputPlan.size() ||
      wire.values.size() != shapes->valueInputs.size())
    return invalid("invocation values are not total over graph value inputs");

  SpatialSimulationRuntimeInputDraft draft{workload.workload.identity()};
  draft.memoryObjects.reserve(wire.memoryObjects.size());
  for (const runtime::SpatialInvocationMemoryObject &object :
       wire.memoryObjects) {
    RuntimeMemoryObject runtimeObject;
    runtimeObject.initialBytes.reserve(object.initialBytes.size());
    for (std::uint8_t byte : object.initialBytes)
      runtimeObject.initialBytes.push_back({SemanticState::Defined, byte});
    draft.memoryObjects.push_back(std::move(runtimeObject));
  }
  draft.memoryRootBindings.reserve(wire.memoryRootBindings.size());
  for (const runtime::SpatialInvocationMemoryRootBinding &binding :
       wire.memoryRootBindings)
    draft.memoryRootBindings.push_back(
        {dataflow::LogicalMemoryRootRef{
             view->identity(),
             dataflow::LogicalMemoryRootId(binding.logicalMemoryRootEntity)},
         binding.objectOrdinal, binding.byteOffset});

  auto graphRef = view->resolve(spatial->launchRef);
  if (!graphRef)
    return graphRef.takeError();
  auto graphView = view->resolve(*graphRef);
  if (!graphView)
    return graphView.takeError();
  auto graph = llvm::dyn_cast<dataflow::GraphOp>(graphView->op);
  if (!graph)
    return invalid("dynamic invocation launch does not resolve to a graph");
  mlir::TypeRange graphInputs = graph.getFunctionType().getInputs();
  draft.runtimeValues.reserve(wire.values.size());
  for (std::size_t ordinal = 0; ordinal != wire.values.size(); ++ordinal) {
    if (!std::holds_alternative<RuntimeValueInput>(
            spatial->valueInputPlan[ordinal]))
      return invalid("dynamic invocation cannot replace a fixed value input");
    auto bitCount = transportBitCount(shapes->valueInputs[ordinal]);
    if (!bitCount)
      return bitCount.takeError();
    if (wire.values[ordinal].bitCount != *bitCount)
      return invalid("invocation value width differs from the graph input");
    auto pointerType =
        llvm::dyn_cast<mlir::LLVM::LLVMPointerType>(graphInputs[ordinal]);
    if (static_cast<bool>(pointerType) !=
        wire.values[ordinal].pointerTarget.has_value())
      return invalid("invocation pointer provenance differs from graph type");
    if (pointerType) {
      const runtime::SpatialInvocationPointerTarget &target =
          *wire.values[ordinal].pointerTarget;
      if (target.objectOrdinal >= wire.memoryObjects.size())
        return invalid("invocation pointer target object is absent");
      auto pointerLayout = view->pointerLayout(pointerType.getAddressSpace());
      if (!pointerLayout)
        return pointerLayout.takeError();
      if (pointerLayout->representationBits != wire.values[ordinal].bitCount ||
          llvm::APInt(64, target.byteOffset).getActiveBits() >
              pointerLayout->addressBits)
        return invalid("invocation pointer target has the wrong layout");
      const runtime::SpatialInvocationMemoryObject &object =
          wire.memoryObjects[target.objectOrdinal];
      if (target.byteOffset >
          std::numeric_limits<std::uint64_t>::max() - object.address)
        return invalid("invocation pointer guest address overflows");
      const llvm::APInt raw = unpackLittleEndianBits(wire.values[ordinal]);
      if (raw.zextOrTrunc(64) !=
          llvm::APInt(64, object.address + target.byteOffset))
        return invalid("invocation pointer bits differ from its guest object");
    }
    auto lanes = unpackDefinedSpatialSimulationToken(
        unpackLittleEndianBits(wire.values[ordinal]),
        shapes->valueInputs[ordinal]);
    if (!lanes)
      return lanes.takeError();
    if (pointerType) {
      if (lanes->size() != 1)
        return invalid("invocation pointer input is not scalar");
      const runtime::SpatialInvocationPointerTarget &target =
          *wire.values[ordinal].pointerTarget;
      auto pointerLayout = view->pointerLayout(pointerType.getAddressSpace());
      if (!pointerLayout)
        return pointerLayout.takeError();
      lanes->front().pointerTarget = PointerTarget{
          target.objectOrdinal,
          llvm::APInt(pointerLayout->addressBits, target.byteOffset)};
    }
    draft.runtimeValues.push_back(
        {ordinal, CanonicalValueSequence{1, std::move(*lanes)}});
  }
  if (llvm::Error error = validateResultDestinations(wire, *spatial, *shapes))
    return std::move(error);
  return finalizeSimulationRuntimeInput(draft, workload.workload, *view);
}

llvm::Expected<ImportedSpatialSimulationInputs>
materializeSpatialInvocationInputs(ImportedSpatialSimulationWorkload workload,
                                   const runtime::SpatialInvocationWire &wire) {
  auto runtimeInput = materializeSpatialInvocationRuntimeInput(workload, wire);
  if (!runtimeInput)
    return runtimeInput.takeError();
  return ImportedSpatialSimulationInputs{std::move(workload.dataflow),
                                         std::move(workload.workload),
                                         std::move(*runtimeInput)};
}

namespace {

llvm::Expected<std::vector<SpatialInvocationMemoryWrite>>
projectResultWrites(const runtime::SpatialInvocationWire &wire,
                    const dataflow::CanonicalDataflowArtifact &dataflow,
                    const CanonicalSimulationWorkload &workload,
                    const SpatialFunctionalObservations &observations) {
  const SpatialSimulationWorkload *spatial = workload.spatial();
  if (!spatial)
    return invalid("workload lost its Spatial payload");
  auto view = dataflow.view();
  if (!view)
    return view.takeError();
  auto shapes =
      projectSpatialSimulationBoundaryShapes(*view, spatial->launchRef);
  if (!shapes)
    return shapes.takeError();
  if (llvm::Error error = validateResultDestinations(wire, *spatial, *shapes))
    return std::move(error);
  if (observations.valueResults.size() != wire.results.size())
    return invalid("execution did not return every invocation value result");
  if (observations.memories.size() !=
      spatial->observableContract.memories.size())
    return invalid("execution did not return every invocation memory result");

  std::vector<SpatialInvocationMemoryWrite> writes;
  writes.reserve(wire.results.size() + observations.memories.size());
  for (std::size_t ordinal = 0; ordinal != wire.results.size(); ++ordinal) {
    const auto *published =
        std::get_if<PublishedValueResult>(&observations.valueResults[ordinal]);
    if (!published)
      return invalid("an invocation value result was not published");
    const std::uint64_t resultOrdinal =
        spatial->observableContract.valueResults[ordinal];
    auto packed = packDefinedSpatialSimulationToken(
        published->value, shapes->valueResults[resultOrdinal], 0);
    if (!packed)
      return packed.takeError();
    writes.push_back(
        {wire.results[ordinal].address, packLittleEndianBits(*packed)});
  }

  auto writableRoots =
      projectSpatialInvocationWritableMemoryRoots(*view, spatial->launchRef);
  if (!writableRoots)
    return writableRoots.takeError();
  if (writableRoots->size() != observations.memories.size())
    return invalid("invocation memory results differ from Dataflow effects");
  for (std::size_t ordinal = 0; ordinal != writableRoots->size(); ++ordinal) {
    const dataflow::LogicalMemoryRootRef root = (*writableRoots)[ordinal];
    const SpatialMemoryObservable &observable =
        spatial->observableContract.memories[ordinal];
    const auto *target =
        std::get_if<dataflow::LogicalMemoryRootOrViewRef>(&observable.target);
    const auto *targetRoot =
        target ? std::get_if<dataflow::LogicalMemoryRootRef>(target) : nullptr;
    if (!targetRoot || *targetRoot != root ||
        observable.form != MemoryObservationForm::DiffFromRuntimeInput)
      return invalid("invocation memory result has the wrong logical root");
    const auto binding =
        llvm::find_if(wire.memoryRootBindings, [&](const auto &candidate) {
          return candidate.logicalMemoryRootEntity == root.entity.value();
        });
    if (binding == wire.memoryRootBindings.end() ||
        binding->objectOrdinal >= wire.memoryObjects.size())
      return invalid("invocation memory result has no guest object");
    const runtime::SpatialInvocationMemoryObject &object =
        wire.memoryObjects[binding->objectOrdinal];
    const auto *diff =
        std::get_if<DiffMemoryObservation>(&observations.memories[ordinal]);
    if (!diff || binding->byteOffset > object.initialBytes.size() ||
        diff->byteCount != object.initialBytes.size() - binding->byteOffset)
      return invalid("invocation memory diff has the wrong object extent");
    for (const MemoryDiffRun &run : diff->runs) {
      if (run.byteOffset > diff->byteCount ||
          run.changedBytes.size() > diff->byteCount - run.byteOffset)
        return invalid("invocation memory diff run exceeds its object");
      if (binding->byteOffset >
              std::numeric_limits<std::uint64_t>::max() - run.byteOffset ||
          object.address > std::numeric_limits<std::uint64_t>::max() -
                               (binding->byteOffset + run.byteOffset))
        return invalid("invocation memory write address overflows");
      SpatialInvocationMemoryWrite write{
          object.address + binding->byteOffset + run.byteOffset, {}};
      write.bytes.reserve(run.changedBytes.size());
      for (const SemanticMemoryByte &byte : run.changedBytes) {
        if (byte.state != SemanticState::Defined)
          return invalid("invocation cannot write an exceptional memory byte");
        write.bytes.push_back(byte.value);
      }
      writes.push_back(std::move(write));
    }
  }

  llvm::sort(writes, [](const auto &lhs, const auto &rhs) {
    return lhs.address < rhs.address;
  });
  std::vector<SpatialInvocationMemoryWrite> merged;
  for (SpatialInvocationMemoryWrite &write : writes) {
    if (write.bytes.empty())
      continue;
    if (merged.empty() ||
        write.address > merged.back().address + merged.back().bytes.size()) {
      merged.push_back(std::move(write));
      continue;
    }
    SpatialInvocationMemoryWrite &prior = merged.back();
    const std::size_t priorOffset =
        static_cast<std::size_t>(write.address - prior.address);
    const std::size_t overlap =
        std::min(write.bytes.size(), prior.bytes.size() - priorOffset);
    for (std::size_t index = 0; index != overlap; ++index)
      if (prior.bytes[priorOffset + index] != write.bytes[index])
        return invalid("aliased invocation memory results disagree");
    prior.bytes.insert(prior.bytes.end(), write.bytes.begin() + overlap,
                       write.bytes.end());
  }
  return merged;
}

} // namespace

llvm::Expected<std::vector<SpatialInvocationMemoryWrite>>
projectSpatialInvocationResultWrites(
    const runtime::SpatialInvocationWire &wire,
    const ImportedSpatialSimulationInputs &inputs,
    const SpatialFunctionalObservations &observations) {
  return projectResultWrites(wire, inputs.dataflow, inputs.workload,
                             observations);
}

llvm::Expected<std::vector<SpatialInvocationMemoryWrite>>
projectSpatialInvocationResultWrites(
    const runtime::SpatialInvocationWire &wire,
    const ImportedSpatialSimulationWorkload &workload,
    const SpatialFunctionalObservations &observations) {
  return projectResultWrites(wire, workload.dataflow, workload.workload,
                             observations);
}

} // namespace loom::sim
