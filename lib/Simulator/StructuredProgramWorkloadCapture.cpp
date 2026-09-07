#include "StructuredProgramWorkloadCapture.h"

#include "NativeExecutionSupport.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <cstring>
#include <limits>
#include <utility>

namespace loom::sim::native_detail {

thread_local WorkloadCaptureContext *activeWorkloadCapture = nullptr;

static void recordWorkloadCaptureError(std::error_code code,
                                       llvm::StringRef message) {
  if (!activeWorkloadCapture || activeWorkloadCapture->error)
    return;
  activeWorkloadCapture->errorCode = code;
  activeWorkloadCapture->error = message.str();
}

static void copyWorkloadCaptureBytes(std::vector<std::uint8_t> &destination,
                                     const std::uint8_t *base,
                                     std::size_t byteCount) {
  destination.resize(byteCount);
  if (byteCount != 0)
    std::memcpy(destination.data(), base, byteCount);
}

static bool reserveWorkloadCaptureBytes(WorkloadCaptureContext &context,
                                        WorkloadCaptureActiveCall &active,
                                        std::uint64_t byteCount) {
  if (active.retainedBytes >
          std::numeric_limits<std::uint64_t>::max() - byteCount ||
      context.retainedBytes >
          std::numeric_limits<std::uint64_t>::max() - byteCount ||
      context.retainedBytes + byteCount > context.maxRetainedBytes) {
    recordWorkloadCaptureError(
        std::make_error_code(std::errc::timed_out),
        "retained capture bytes exceeded the execution limit");
    return false;
  }
  active.retainedBytes += byteCount;
  context.retainedBytes += byteCount;
  return true;
}

void workloadCaptureBegin() {
  if (!activeWorkloadCapture || activeWorkloadCapture->error)
    return;
  activeWorkloadCapture->captures.emplace_back(std::in_place);
  WorkloadCaptureActiveCall active;
  active.captureIndex = activeWorkloadCapture->captures.size() - 1;
  NativeSimulationCallCapture &capture =
      *activeWorkloadCapture->captures.back();
  capture.runtimeStreams.resize(
      activeWorkloadCapture->streamInputShapes.size());
  capture.streamOutputs.resize(
      activeWorkloadCapture->streamOutputShapes.size());
  activeWorkloadCapture->activeCalls.push_back(std::move(active));
}

static bool endRuntimeObjectLifetime(std::uint64_t ordinal) {
  WorkloadCaptureContext &context = *activeWorkloadCapture;
  WorkloadCaptureRuntimeObject &object = context.runtimeObjects[ordinal];
  if (object.byteCount == 0)
    return true;
  for (const WorkloadCaptureActiveCall &active : context.activeCalls) {
    if (llvm::is_contained(active.runtimeObjectOrdinals, ordinal)) {
      recordWorkloadCaptureError(
          std::make_error_code(std::errc::not_supported),
          "imported runtime object lifetime ends during a selected activation");
      return false;
    }
  }
  for (auto position = context.pointerPayloads.begin();
       position != context.pointerPayloads.end();) {
    TrackedPointerPayload &pointer = position->second;
    if (pointer.storageObjectOrdinal == ordinal) {
      position = context.pointerPayloads.erase(position);
      continue;
    }
    if (pointer.target && pointer.target->objectOrdinal == ordinal)
      pointer.target.reset();
    ++position;
  }
  object.byteCount = 0;
  object.stackAllocation.reset();
  object.source.reset();
  return true;
}

void workloadCaptureEnterStackFrame() {
  if (activeWorkloadCapture && !activeWorkloadCapture->error)
    ++activeWorkloadCapture->stackFrameDepth;
}

void workloadCaptureLeaveStackFrame() {
  if (!activeWorkloadCapture || activeWorkloadCapture->error)
    return;
  WorkloadCaptureContext &context = *activeWorkloadCapture;
  for (auto [ordinal, object] : llvm::enumerate(context.runtimeObjects))
    if (object.stackAllocation &&
        object.stackAllocation->frameDepth == context.stackFrameDepth &&
        !endRuntimeObjectLifetime(ordinal))
      return;
  --context.stackFrameDepth;
}

void workloadCaptureEndStackObject(void *base, std::uint64_t allocation) {
  if (!activeWorkloadCapture || activeWorkloadCapture->error)
    return;
  for (auto [ordinal, object] :
       llvm::enumerate(activeWorkloadCapture->runtimeObjects)) {
    if (object.base == base && object.stackAllocation &&
        object.stackAllocation->allocationOrdinal == allocation &&
        object.stackAllocation->frameDepth ==
            activeWorkloadCapture->stackFrameDepth) {
      endRuntimeObjectLifetime(ordinal);
      return;
    }
  }
}

void workloadCaptureRegisterObject(void *base, std::uint64_t extentFactor0,
                                   std::uint64_t extentFactor1,
                                   ProgramObjectCaptureKind kind,
                                   std::uint64_t allocation) {
  if (!activeWorkloadCapture || activeWorkloadCapture->error)
    return;
  if (!base || extentFactor0 == 0 || extentFactor1 == 0)
    return;
  if (extentFactor0 >
      std::numeric_limits<std::uint64_t>::max() / extentFactor1) {
    recordWorkloadCaptureError(
        std::make_error_code(std::errc::not_supported),
        "program object extent exceeds the runtime registry domain");
    return;
  }
  if (allocation >= activeWorkloadCapture->programObjectSources.size()) {
    recordWorkloadCaptureError(
        std::make_error_code(std::errc::invalid_argument),
        "program object callback has no source allocation");
    return;
  }
  const std::uint64_t byteCount = extentFactor0 * extentFactor1;
  if (byteCount > std::numeric_limits<std::size_t>::max()) {
    recordWorkloadCaptureError(
        std::make_error_code(std::errc::not_supported),
        "program object extent exceeds the host addressable domain");
    return;
  }
  WorkloadCaptureContext &context = *activeWorkloadCapture;
  std::size_t ordinal = context.runtimeObjects.size();
  for (auto [index, object] : llvm::enumerate(context.runtimeObjects)) {
    if (object.base != base)
      continue;
    if (kind == ProgramObjectCaptureKind::RuntimeAllocation) {
      // allocsize describes an extent; another call returning this object
      // does not establish a new allocation lifetime.
      object.byteCount = static_cast<std::size_t>(byteCount);
      return;
    }
    if (kind == ProgramObjectCaptureKind::Global && object.byteCount != 0) {
      if (object.byteCount != byteCount)
        recordWorkloadCaptureError(
            std::make_error_code(std::errc::io_error),
            "existing program object has conflicting allocation extents");
      return;
    }
    if (!endRuntimeObjectLifetime(index))
      return;
    ordinal = index;
    break;
  }
  if (ordinal == context.runtimeObjects.size())
    context.runtimeObjects.emplace_back();
  WorkloadCaptureRuntimeObject &object = context.runtimeObjects[ordinal];
  object.source = context.programObjectSources[allocation];
  object.base = static_cast<std::uint8_t *>(base);
  object.byteCount = static_cast<std::size_t>(byteCount);
  if (kind == ProgramObjectCaptureKind::StackAllocation)
    object.stackAllocation =
        WorkloadCaptureStackAllocation{allocation, context.stackFrameDepth};
}

static std::optional<std::pair<std::uint64_t, std::uint64_t>>
resolveRuntimeObject(void *pointer, bool admitOnePast = false) {
  if (!activeWorkloadCapture || !pointer)
    return std::nullopt;
  const std::uintptr_t address = reinterpret_cast<std::uintptr_t>(pointer);
  std::optional<std::pair<std::uint64_t, std::uint64_t>> onePast;
  for (std::size_t ordinal = activeWorkloadCapture->runtimeObjects.size();
       ordinal != 0; --ordinal) {
    const auto &object = activeWorkloadCapture->runtimeObjects[ordinal - 1];
    if (object.byteCount == 0)
      continue;
    const std::uintptr_t base = reinterpret_cast<std::uintptr_t>(object.base);
    if (address < base)
      continue;
    const std::uintptr_t offset = address - base;
    if (offset < object.byteCount)
      return std::pair<std::uint64_t, std::uint64_t>{ordinal - 1, offset};
    if (!admitOnePast || offset != object.byteCount)
      continue;
    if (onePast)
      return std::nullopt;
    onePast = std::pair<std::uint64_t, std::uint64_t>{ordinal - 1, offset};
  }
  return onePast;
}

static bool rangesOverlap(std::uint64_t lhsOffset, std::uint64_t lhsBytes,
                          std::uint64_t rhsOffset, std::uint64_t rhsBytes) {
  if (lhsBytes == 0 || rhsBytes == 0)
    return false;
  return lhsOffset < rhsOffset + rhsBytes && rhsOffset < lhsOffset + lhsBytes;
}

static void eraseTrackedPointers(std::uint64_t objectOrdinal,
                                 std::uint64_t byteOffset,
                                 std::uint64_t byteCount) {
  WorkloadCaptureContext &context = *activeWorkloadCapture;
  for (auto position = context.pointerPayloads.begin();
       position != context.pointerPayloads.end();) {
    const TrackedPointerPayload &pointer = position->second;
    const std::uint64_t pointerBytes = pointer.representationBits / 8;
    if (pointer.storageObjectOrdinal == objectOrdinal &&
        rangesOverlap(pointer.storageByteOffset, pointerBytes, byteOffset,
                      byteCount)) {
      position = context.pointerPayloads.erase(position);
      continue;
    }
    ++position;
  }
}

void workloadCaptureMemoryWrite(void *storage, std::uint64_t byteCount) {
  if (!activeWorkloadCapture || activeWorkloadCapture->error)
    return;
  std::optional<std::pair<std::uint64_t, std::uint64_t>> resolved =
      resolveRuntimeObject(storage);
  if (!resolved && activeWorkloadCapture->activeCalls.empty())
    return;
  if (!resolved || byteCount == 0 ||
      byteCount >
          activeWorkloadCapture->runtimeObjects[resolved->first].byteCount -
              resolved->second) {
    recordWorkloadCaptureError(
        std::make_error_code(std::errc::not_supported),
        "memory write is outside the runtime object registry");
    return;
  }
  eraseTrackedPointers(resolved->first, resolved->second, byteCount);
}

static bool validatePointerCaptureLayout(std::uint64_t addressSpace,
                                         std::uint64_t representationBits,
                                         std::uint64_t addressBits,
                                         llvm::StringRef operation) {
  if (addressSpace <= std::numeric_limits<std::uint32_t>::max() &&
      representationBits != 0 && representationBits % 8 == 0 &&
      representationBits <= std::numeric_limits<std::uint32_t>::max() &&
      addressBits != 0 &&
      addressBits <= std::numeric_limits<std::uint32_t>::max())
    return true;
  recordWorkloadCaptureError(
      std::make_error_code(std::errc::io_error),
      (llvm::Twine(operation) + " has an invalid DataLayout projection").str());
  return false;
}

void workloadCapturePointerRead(void *storage, void *value,
                                std::uint64_t addressSpace,
                                std::uint64_t representationBits,
                                std::uint64_t addressBits) {
  if (!activeWorkloadCapture || activeWorkloadCapture->error ||
      activeWorkloadCapture->activeCalls.empty())
    return;
  if (!validatePointerCaptureLayout(addressSpace, representationBits,
                                    addressBits, "pointer read"))
    return;
  auto storageTarget = resolveRuntimeObject(storage);
  if (!storageTarget) {
    recordWorkloadCaptureError(
        std::make_error_code(std::errc::not_supported),
        "pointer-read storage is outside the runtime object registry");
    return;
  }
  WorkloadCaptureActiveCall &active = activeWorkloadCapture->activeCalls.back();
  if (active.nextRoot != activeWorkloadCapture->rootCount ||
      llvm::find(active.runtimeObjectOrdinals, storageTarget->first) ==
          active.runtimeObjectOrdinals.end()) {
    recordWorkloadCaptureError(
        std::make_error_code(std::errc::not_supported),
        "selected pointer read has no imported memory service");
    return;
  }
  auto valueTarget = resolveRuntimeObject(value, /*admitOnePast=*/true);
  if (!valueTarget) {
    recordWorkloadCaptureError(
        std::make_error_code(std::errc::not_supported),
        "captured pointer target is outside the runtime object registry");
    return;
  }

  const TrackedPointerKey key{storageTarget->first, storageTarget->second};
  TrackedPointerPayload pointer{
      storageTarget->first, storageTarget->second,
      static_cast<std::uint32_t>(addressSpace),
      static_cast<std::uint32_t>(representationBits),
      PointerTarget{valueTarget->first,
                    llvm::APInt(static_cast<unsigned>(addressBits),
                                valueTarget->second)}};
  active.relevantPointerPayloads.insert(key);
  if (active.writtenPointerPayloads.find(key) ==
      active.writtenPointerPayloads.end())
    active.initialPointerPayloads.insert_or_assign(key, pointer);
  activeWorkloadCapture->pointerPayloads.insert_or_assign(key,
                                                          std::move(pointer));
}

void workloadCapturePointerWrite(void *storage, void *value,
                                 std::uint64_t addressSpace,
                                 std::uint64_t representationBits,
                                 std::uint64_t addressBits,
                                 std::uint64_t rootHint) {
  if (!activeWorkloadCapture || activeWorkloadCapture->error)
    return;
  if (!validatePointerCaptureLayout(addressSpace, representationBits,
                                    addressBits, "pointer write"))
    return;
  auto storageTarget = resolveRuntimeObject(storage);
  std::optional<std::pair<std::uint64_t, std::uint64_t>> valueTarget;
  std::optional<llvm::APInt> hintedOffset;
  if (rootHint != std::numeric_limits<std::uint64_t>::max() &&
      !activeWorkloadCapture->activeCalls.empty()) {
    WorkloadCaptureActiveCall &active =
        activeWorkloadCapture->activeCalls.back();
    NativeSimulationCallCapture &capture =
        *activeWorkloadCapture->captures[active.captureIndex];
    if (rootHint >= capture.memoryRootObjectOrdinals.size()) {
      recordWorkloadCaptureError(
          std::make_error_code(std::errc::io_error),
          "pointer-write root hint is outside the captured root table");
      return;
    }
    const std::uint64_t localObject =
        capture.memoryRootObjectOrdinals[rootHint];
    if (localObject >= active.runtimeObjectOrdinals.size()) {
      recordWorkloadCaptureError(
          std::make_error_code(std::errc::io_error),
          "pointer-write root hint names an absent runtime object");
      return;
    }
    const std::uint64_t runtimeObject =
        active.runtimeObjectOrdinals[localObject];
    const std::uintptr_t address = reinterpret_cast<std::uintptr_t>(value);
    const std::uintptr_t base = reinterpret_cast<std::uintptr_t>(
        activeWorkloadCapture->runtimeObjects[runtimeObject].base);
    llvm::APInt addressValue(static_cast<unsigned>(addressBits), address);
    llvm::APInt baseValue(static_cast<unsigned>(addressBits), base);
    hintedOffset = addressValue - baseValue;
    valueTarget = {runtimeObject, 0};
  } else {
    valueTarget = resolveRuntimeObject(value, /*admitOnePast=*/true);
  }
  if (!storageTarget && activeWorkloadCapture->activeCalls.empty())
    return;
  if (!storageTarget) {
    recordWorkloadCaptureError(
        std::make_error_code(std::errc::not_supported),
        "pointer-write storage is outside the runtime object registry during "
        "a selected activation");
    return;
  }
  const std::uint64_t pointerBytes = representationBits / 8;
  if (pointerBytes >
      activeWorkloadCapture->runtimeObjects[storageTarget->first].byteCount -
          storageTarget->second) {
    recordWorkloadCaptureError(std::make_error_code(std::errc::io_error),
                               "pointer write exceeds its runtime object");
    return;
  }
  eraseTrackedPointers(storageTarget->first, storageTarget->second,
                       pointerBytes);
  TrackedPointerPayload pointer;
  pointer.storageObjectOrdinal = storageTarget->first;
  pointer.storageByteOffset = storageTarget->second;
  pointer.addressSpace = static_cast<std::uint32_t>(addressSpace);
  pointer.representationBits = static_cast<std::uint32_t>(representationBits);
  if (valueTarget)
    pointer.target = PointerTarget{
        valueTarget->first,
        hintedOffset ? *hintedOffset
                     : llvm::APInt(static_cast<unsigned>(addressBits),
                                   valueTarget->second)};
  activeWorkloadCapture->pointerPayloads[{pointer.storageObjectOrdinal,
                                          pointer.storageByteOffset}] =
      std::move(pointer);
  if (!activeWorkloadCapture->activeCalls.empty()) {
    WorkloadCaptureActiveCall &active =
        activeWorkloadCapture->activeCalls.back();
    const TrackedPointerKey key{storageTarget->first, storageTarget->second};
    active.relevantPointerPayloads.insert(key);
    active.writtenPointerPayloads.insert(key);
  }
}

static bool
projectTrackedPointer(const WorkloadCaptureActiveCall &active,
                      std::vector<NativeCapturedMemoryObject> &objects,
                      const TrackedPointerPayload &pointer, bool initial) {
  auto storage =
      llvm::find(active.runtimeObjectOrdinals, pointer.storageObjectOrdinal);
  if (storage == active.runtimeObjectOrdinals.end())
    return true;
  if (!pointer.target) {
    recordWorkloadCaptureError(
        std::make_error_code(std::errc::not_supported),
        "captured pointer target is outside the runtime object registry");
    return false;
  }
  auto target =
      llvm::find(active.runtimeObjectOrdinals, pointer.target->objectOrdinal);
  if (target == active.runtimeObjectOrdinals.end()) {
    recordWorkloadCaptureError(
        std::make_error_code(std::errc::not_supported),
        "captured pointer target has no imported memory service");
    return false;
  }
  const std::uint64_t storageOrdinal =
      std::distance(active.runtimeObjectOrdinals.begin(), storage);
  const std::uint64_t targetOrdinal =
      std::distance(active.runtimeObjectOrdinals.begin(), target);
  RuntimeMemoryPointer projected{
      pointer.storageByteOffset, pointer.addressSpace,
      PointerTarget{targetOrdinal, pointer.target->byteOffset}};
  auto &destination = initial ? objects[storageOrdinal].initialPointers
                              : objects[storageOrdinal].finalPointers;
  destination.push_back(std::move(projected));
  return true;
}

void workloadCaptureMemoryRoot(std::uint64_t rootOrdinal, void *pointer) {
  if (!activeWorkloadCapture || activeWorkloadCapture->error)
    return;
  WorkloadCaptureContext &context = *activeWorkloadCapture;
  if (context.activeCalls.empty() ||
      context.activeCalls.back().nextCoordinate !=
          context.coordinateShapes.size() ||
      rootOrdinal >= context.rootCount ||
      context.activeCalls.back().nextRoot != rootOrdinal) {
    recordWorkloadCaptureError(
        std::make_error_code(std::errc::io_error),
        "memory-root callbacks are not in canonical order");
    return;
  }
  std::optional<std::pair<std::uint64_t, std::uint64_t>> resolved =
      resolveRuntimeObject(pointer);
  if (!resolved) {
    recordWorkloadCaptureError(
        std::make_error_code(std::errc::not_supported),
        "selected memory root is not owned by the runtime object registry");
    return;
  }

  WorkloadCaptureActiveCall &active = context.activeCalls.back();
  NativeSimulationCallCapture &capture = *context.captures[active.captureIndex];
  auto found = llvm::find(active.runtimeObjectOrdinals, resolved->first);
  std::uint64_t objectOrdinal = 0;
  if (found == active.runtimeObjectOrdinals.end()) {
    objectOrdinal = capture.objects.size();
    active.runtimeObjectOrdinals.push_back(resolved->first);
    const WorkloadCaptureRuntimeObject &object =
        context.runtimeObjects[resolved->first];
    auto *base = object.base;
    const std::size_t byteCount = object.byteCount;
    if (!reserveWorkloadCaptureBytes(context, active, byteCount))
      return;
    active.objectBases.push_back(base);
    capture.objects.emplace_back();
    capture.objects.back().source = object.source;
    if (object.stackAllocation && capture.objects.back().source)
      if (auto *stack = std::get_if<NativeStackMemoryObjectSource>(
              &*capture.objects.back().source))
        stack->captureFrameDistance =
            context.stackFrameDepth - object.stackAllocation->frameDepth;
    copyWorkloadCaptureBytes(capture.objects.back().initialBytes, base,
                             byteCount);
  } else {
    objectOrdinal = static_cast<std::uint64_t>(
        std::distance(active.runtimeObjectOrdinals.begin(), found));
  }
  capture.memoryRootObjectOrdinals.push_back(objectOrdinal);
  capture.memoryRootByteOffsets.push_back(resolved->second);
  ++active.nextRoot;
}

static CanonicalValueSequence
readWorkloadCaptureValue(void *base, const WorkloadCaptureValueShape &shape,
                         bool littleEndian) {
  return detail::readDefinedNativeValue(
      llvm::ArrayRef<std::uint8_t>(static_cast<std::uint8_t *>(base),
                                   static_cast<std::size_t>(shape.byteCount)),
      shape.lanesPerToken, shape.laneBitWidth, littleEndian);
}

void workloadCaptureCoordinate(std::uint64_t ordinal, void *base,
                               std::uint64_t byteCount) {
  if (!activeWorkloadCapture || activeWorkloadCapture->error)
    return;
  WorkloadCaptureContext &context = *activeWorkloadCapture;
  if (context.activeCalls.empty() ||
      ordinal >= context.coordinateShapes.size() || !base ||
      context.coordinateShapes[ordinal].byteCount != byteCount ||
      context.activeCalls.back().nextCoordinate != ordinal) {
    recordWorkloadCaptureError(std::make_error_code(std::errc::io_error),
                               "dense coordinate callback is malformed");
    return;
  }
  const WorkloadCaptureValueShape &shape = context.coordinateShapes[ordinal];
  WorkloadCaptureActiveCall &active = context.activeCalls.back();
  if (!reserveWorkloadCaptureBytes(context, active, byteCount))
    return;
  CanonicalValueSequence value =
      readWorkloadCaptureValue(base, shape, context.littleEndian);
  if (value.tokenCount != 1 || value.lanes.size() != 1 ||
      value.lanes.front().state != SemanticState::Defined ||
      value.lanes.front().bits.isNegative() ||
      value.lanes.front().bits.getActiveBits() > 64) {
    recordWorkloadCaptureError(
        std::make_error_code(std::errc::io_error),
        "dense coordinate is not one defined unsigned 64-bit value");
    return;
  }
  context.captures[active.captureIndex]->denseCoordinates.push_back(
      value.lanes.front().bits.getZExtValue());
  ++active.nextCoordinate;
}

void workloadCaptureValue(std::uint64_t ordinal, void *base,
                          std::uint64_t byteCount) {
  if (!activeWorkloadCapture || activeWorkloadCapture->error)
    return;
  WorkloadCaptureContext &context = *activeWorkloadCapture;
  if (context.activeCalls.empty() ||
      ordinal >= context.runtimeValueShapes.size() || !base ||
      context.runtimeValueShapes[ordinal].byteCount != byteCount ||
      context.activeCalls.back().nextCoordinate !=
          context.coordinateShapes.size() ||
      context.activeCalls.back().nextRoot != context.rootCount ||
      context.activeCalls.back().nextValue != ordinal) {
    recordWorkloadCaptureError(std::make_error_code(std::errc::io_error),
                               "runtime value callback is malformed");
    return;
  }
  WorkloadCaptureActiveCall &active = context.activeCalls.back();
  const WorkloadCaptureValueShape &shape = context.runtimeValueShapes[ordinal];
  if (!reserveWorkloadCaptureBytes(context, active, byteCount))
    return;
  CanonicalValueSequence value =
      readWorkloadCaptureValue(base, shape, context.littleEndian);
  if (shape.pointerTarget) {
    NativeSimulationCallCapture &capture =
        *context.captures[active.captureIndex];
    const std::uint64_t rootOrdinal =
        shape.pointerTarget->memoryRootBindingOrdinal;
    if (value.lanes.size() != 1 ||
        rootOrdinal >= capture.memoryRootObjectOrdinals.size() ||
        rootOrdinal >= capture.memoryRootByteOffsets.size() ||
        shape.pointerTarget->addressBitWidth == 0) {
      recordWorkloadCaptureError(
          std::make_error_code(std::errc::io_error),
          "pointer value has an invalid memory-root target");
      return;
    }
    const std::uint64_t byteOffset = capture.memoryRootByteOffsets[rootOrdinal];
    llvm::APInt offsetBits(64, byteOffset);
    if (offsetBits.getActiveBits() >= shape.pointerTarget->addressBitWidth) {
      recordWorkloadCaptureError(
          std::make_error_code(std::errc::io_error),
          "pointer value offset exceeds its signed address width");
      return;
    }
    value.lanes.front().pointerTarget = PointerTarget{
        capture.memoryRootObjectOrdinals[rootOrdinal],
        llvm::APInt(shape.pointerTarget->addressBitWidth, byteOffset)};
  }
  context.captures[active.captureIndex]->runtimeValues.push_back(
      RuntimeValueEntry{shape.graphOrdinal, std::move(value)});
  ++active.nextValue;
}

static void workloadCaptureStream(std::uint64_t ordinal, void *base,
                                  std::uint64_t byteCount, bool input) {
  if (!activeWorkloadCapture || activeWorkloadCapture->error)
    return;
  WorkloadCaptureContext &context = *activeWorkloadCapture;
  const auto &shapes =
      input ? context.streamInputShapes : context.streamOutputShapes;
  if (context.activeCalls.empty() || ordinal >= shapes.size() || !base ||
      shapes[ordinal].byteCount != byteCount ||
      context.activeCalls.back().nextCoordinate !=
          context.coordinateShapes.size() ||
      context.activeCalls.back().nextRoot != context.rootCount ||
      context.activeCalls.back().nextValue !=
          context.runtimeValueShapes.size()) {
    recordWorkloadCaptureError(std::make_error_code(std::errc::io_error),
                               "stream token callback is malformed");
    return;
  }
  WorkloadCaptureActiveCall &active = context.activeCalls.back();
  if (!reserveWorkloadCaptureBytes(context, active, byteCount))
    return;
  CanonicalValueSequence token =
      readWorkloadCaptureValue(base, shapes[ordinal], context.littleEndian);
  NativeSimulationCallCapture &capture = *context.captures[active.captureIndex];
  CanonicalStreamSequence &stream =
      input ? capture.runtimeStreams[ordinal] : capture.streamOutputs[ordinal];
  if (stream.values.tokenCount == std::numeric_limits<std::uint64_t>::max()) {
    recordWorkloadCaptureError(std::make_error_code(std::errc::io_error),
                               "stream token count overflowed");
    return;
  }
  ++stream.values.tokenCount;
  stream.values.lanes.insert(stream.values.lanes.end(),
                             std::make_move_iterator(token.lanes.begin()),
                             std::make_move_iterator(token.lanes.end()));
}

void workloadCaptureStreamInput(std::uint64_t ordinal, void *base,
                                std::uint64_t byteCount) {
  workloadCaptureStream(ordinal, base, byteCount, true);
}

void workloadCaptureStreamOutput(std::uint64_t ordinal, void *base,
                                 std::uint64_t byteCount) {
  workloadCaptureStream(ordinal, base, byteCount, false);
}

void workloadCaptureResult(std::uint64_t ordinal, void *base,
                           std::uint64_t byteCount) {
  if (!activeWorkloadCapture || activeWorkloadCapture->error)
    return;
  WorkloadCaptureContext &context = *activeWorkloadCapture;
  if (context.activeCalls.empty() ||
      ordinal >= context.valueResultShapes.size() || !base ||
      context.valueResultShapes[ordinal].byteCount != byteCount ||
      context.activeCalls.back().nextCoordinate !=
          context.coordinateShapes.size() ||
      context.activeCalls.back().nextRoot != context.rootCount ||
      context.activeCalls.back().nextValue !=
          context.runtimeValueShapes.size() ||
      context.activeCalls.back().nextResult != ordinal) {
    recordWorkloadCaptureError(std::make_error_code(std::errc::io_error),
                               "value result callback is malformed");
    return;
  }
  WorkloadCaptureActiveCall &active = context.activeCalls.back();
  if (!reserveWorkloadCaptureBytes(context, active, byteCount))
    return;
  context.captures[active.captureIndex]->valueResults.push_back(
      readWorkloadCaptureValue(base, context.valueResultShapes[ordinal],
                               context.littleEndian));
  ++active.nextResult;
}

void workloadCaptureEnd() {
  if (!activeWorkloadCapture || activeWorkloadCapture->error)
    return;
  WorkloadCaptureContext &context = *activeWorkloadCapture;
  if (context.activeCalls.empty()) {
    recordWorkloadCaptureError(std::make_error_code(std::errc::io_error),
                               "capture end has no active invocation");
    return;
  }
  WorkloadCaptureActiveCall &active = context.activeCalls.back();
  if (active.nextCoordinate != context.coordinateShapes.size() ||
      active.nextRoot != context.rootCount ||
      active.nextValue != context.runtimeValueShapes.size() ||
      active.nextResult != context.valueResultShapes.size()) {
    recordWorkloadCaptureError(std::make_error_code(std::errc::io_error),
                               "capture end observed an incomplete boundary");
    return;
  }
  NativeSimulationCallCapture &capture = *context.captures[active.captureIndex];
  for (const auto &[key, pointer] : active.initialPointerPayloads) {
    (void)key;
    if (!projectTrackedPointer(active, capture.objects, pointer,
                               /*initial=*/true))
      return;
  }
  for (const TrackedPointerKey &key : active.relevantPointerPayloads) {
    auto pointer = context.pointerPayloads.find(key);
    if (pointer == context.pointerPayloads.end())
      continue;
    if (!projectTrackedPointer(active, capture.objects, pointer->second,
                               /*initial=*/false))
      return;
  }
  for (auto [ordinal, base] : llvm::enumerate(active.objectBases)) {
    const std::uint64_t runtimeOrdinal = active.runtimeObjectOrdinals[ordinal];
    if (!reserveWorkloadCaptureBytes(
            context, active, context.runtimeObjects[runtimeOrdinal].byteCount))
      return;
    copyWorkloadCaptureBytes(capture.objects[ordinal].finalBytes, base,
                             context.runtimeObjects[runtimeOrdinal].byteCount);
  }
  if (context.visitor) {
    if (llvm::Error error = (*context.visitor)(std::move(capture))) {
      std::error_code code;
      std::string message;
      llvm::raw_string_ostream stream(message);
      llvm::handleAllErrors(std::move(error),
                            [&](const llvm::ErrorInfoBase &failure) {
                              code = failure.convertToErrorCode();
                              failure.log(stream);
                            });
      stream.flush();
      recordWorkloadCaptureError(
          code ? code : std::make_error_code(std::errc::io_error), message);
    }
    context.retainedBytes -= active.retainedBytes;
  } else {
    recordWorkloadCaptureError(std::make_error_code(std::errc::io_error),
                               "workload capture has no streaming consumer");
  }
  context.captures[active.captureIndex].reset();
  context.activeCalls.pop_back();
  while (!context.captures.empty() && !context.captures.back())
    context.captures.pop_back();
}

} // namespace loom::sim::native_detail
