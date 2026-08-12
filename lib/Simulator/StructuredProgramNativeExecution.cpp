#include "Simulator/NativeSimulationOracle.h"

#include "NativeExecutionSupport.h"
#include "SimulationWireInternal.h"
#include "StructuredProgramNativeExecutionInternal.h"

#include "Common/PointerLayout.h"
#include "Dataflow/IR/DataflowOps.h"
#include "Frontend/IR/LoomOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Verifier.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <new>
#include <optional>
#include <set>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace loom::sim {
namespace {

using native_detail::AlignedByteStorage;
using native_detail::buildObservations;
using native_detail::instrumentBlockActivations;
using native_detail::instrumentWorkloadBackedCapture;
using native_detail::MemoryTargetPlan;
using native_detail::NativeChannelCallbackNames;
using native_detail::NativeExecutionContext;
using native_detail::projectSelectedWholeProgram;
using native_detail::uniqueMlirSymbolName;
using native_detail::WorkloadCaptureCallbackNames;

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      llvm::Twine("native_structured_program_invalid: ") + message);
}

llvm::Error unsupported(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::not_supported),
      llvm::Twine("native_structured_program_unsupported: ") + message);
}

llvm::Error executionFailed(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::io_error),
      llvm::Twine("native_structured_program_execution_failed: ") + message);
}

llvm::Error classifyJitError(llvm::Error error, llvm::StringRef operation) {
  bool missingSymbol = false;
  bool otherFailure = false;
  std::string detail;
  llvm::raw_string_ostream stream(detail);
  llvm::handleAllErrors(
      std::move(error),
      [&](const llvm::orc::SymbolsNotFound &missing) {
        missingSymbol = true;
        missing.log(stream);
      },
      [&](const llvm::ErrorInfoBase &other) {
        otherFailure = true;
        other.log(stream);
      });
  stream.flush();
  if (missingSymbol && !otherFailure)
    return unsupported(llvm::Twine(operation) + ": " + detail);
  return executionFailed(llvm::Twine(operation) + ": " + detail);
}

struct WorkloadCaptureValueShape final {
  std::uint64_t graphOrdinal = 0;
  std::uint64_t lanesPerToken = 0;
  std::uint32_t laneBitWidth = 0;
  std::uint64_t byteCount = 0;
  std::optional<SimulationPointerValueTargetCapture> pointerTarget;
};

struct TrackedPointerPayload final {
  std::uint64_t storageObjectOrdinal = 0;
  std::uint64_t storageByteOffset = 0;
  std::uint32_t addressSpace = 0;
  std::uint32_t representationBits = 0;
  std::optional<PointerTarget> target;
};

using TrackedPointerKey = std::pair<std::uint64_t, std::uint64_t>;

struct WorkloadCaptureActiveCall final {
  std::size_t captureIndex = 0;
  std::uint64_t retainedBytes = 0;
  std::uint64_t nextCoordinate = 0;
  std::uint64_t nextRoot = 0;
  std::uint64_t nextValue = 0;
  std::uint64_t nextResult = 0;
  std::vector<std::uint64_t> runtimeObjectOrdinals;
  std::vector<std::uint8_t *> objectBases;
  std::map<TrackedPointerKey, TrackedPointerPayload> initialPointerPayloads;
  std::set<TrackedPointerKey> relevantPointerPayloads;
  std::set<TrackedPointerKey> writtenPointerPayloads;
};

struct WorkloadCaptureContext final {
  std::vector<std::optional<NativeSimulationCallCapture>> captures;
  std::vector<WorkloadCaptureValueShape> coordinateShapes;
  std::vector<WorkloadCaptureValueShape> runtimeValueShapes;
  std::vector<WorkloadCaptureValueShape> streamInputShapes;
  std::vector<WorkloadCaptureValueShape> valueResultShapes;
  std::vector<WorkloadCaptureValueShape> streamOutputShapes;
  std::vector<std::pair<std::uint8_t *, std::size_t>> runtimeObjects;
  std::map<TrackedPointerKey, TrackedPointerPayload> pointerPayloads;
  std::vector<WorkloadCaptureActiveCall> activeCalls;
  std::uint64_t rootCount = 0;
  std::uint64_t retainedBytes = 0;
  std::uint64_t maxRetainedBytes = 0;
  WorkloadBackedSimulationInputVisitor *visitor = nullptr;
  bool littleEndian = true;
  std::optional<std::error_code> errorCode;
  std::optional<std::string> error;
};

thread_local NativeExecutionContext *activeExecution = nullptr;
thread_local WorkloadCaptureContext *activeWorkloadCapture = nullptr;

void recordExecutionError(llvm::StringRef message) {
  if (activeExecution && !activeExecution->error)
    activeExecution->error = message.str();
}

void nativeInvalidLogicalThreadExtent() {
  recordExecutionError("logical thread extent is negative");
}

std::uint64_t nativeLogicalChannelCreate(std::uint64_t receiverCount) {
  if (!activeExecution || receiverCount == 0 ||
      receiverCount > std::numeric_limits<std::size_t>::max()) {
    recordExecutionError(
        "logical channel create has an invalid receiver count");
    return 0;
  }
  NativeExecutionContext::LogicalChannel channel;
  channel.receiverCursors.resize(static_cast<std::size_t>(receiverCount));
  activeExecution->logicalChannels.push_back(std::move(channel));
  return activeExecution->logicalChannels.size();
}

void nativeLogicalChannelSend(std::uint64_t handle, const void *base,
                              std::uint64_t byteCount) {
  if (!activeExecution || handle == 0 ||
      handle > activeExecution->logicalChannels.size() || !base ||
      byteCount == 0 || byteCount > std::numeric_limits<std::size_t>::max()) {
    recordExecutionError("logical channel send has an invalid payload");
    return;
  }
  auto &channel = activeExecution->logicalChannels[handle - 1];
  std::vector<std::uint8_t> message(static_cast<std::size_t>(byteCount));
  std::memcpy(message.data(), base, message.size());
  channel.messages.push_back(std::move(message));
}

void nativeLogicalChannelReceive(std::uint64_t handle,
                                 std::uint64_t receiverOrdinal, void *base,
                                 std::uint64_t byteCount) {
  if (!activeExecution || handle == 0 ||
      handle > activeExecution->logicalChannels.size() || !base ||
      byteCount == 0 || byteCount > std::numeric_limits<std::size_t>::max()) {
    recordExecutionError("logical channel receive has an invalid payload");
    return;
  }
  auto &channel = activeExecution->logicalChannels[handle - 1];
  if (receiverOrdinal >= channel.receiverCursors.size()) {
    recordExecutionError("logical channel receive has an invalid endpoint");
    return;
  }
  std::uint64_t &cursor = channel.receiverCursors[receiverOrdinal];
  if (cursor >= channel.messages.size() ||
      channel.messages[cursor].size() != byteCount) {
    recordExecutionError("logical channel receive has no matching message");
    return;
  }
  std::memcpy(base, channel.messages[cursor].data(),
              channel.messages[cursor].size());
  ++cursor;
}

void *nativeRuntimeObject(std::uint64_t ordinal) {
  if (!activeExecution || ordinal >= activeExecution->objects.size()) {
    recordExecutionError("runtime object callback has an invalid ordinal");
    return nullptr;
  }
  return activeExecution->objects[ordinal].data();
}

void nativeReturnValue(void *base, std::uint64_t byteCount) {
  if (!activeExecution || !activeExecution->returnShape || !base ||
      byteCount != activeExecution->returnByteCount ||
      activeExecution->returnValue) {
    recordExecutionError("return callback has an invalid projection");
    return;
  }
  activeExecution->returnValue = detail::readDefinedNativeValue(
      llvm::ArrayRef<std::uint8_t>(static_cast<const std::uint8_t *>(base),
                                   static_cast<std::size_t>(byteCount)),
      activeExecution->returnShape->lanesPerToken,
      activeExecution->returnShape->laneBitWidth,
      activeExecution->littleEndian);
}

void copyGlobalBytes(std::vector<std::uint8_t> &destination, void *base,
                     std::uint64_t byteCount) {
  destination.resize(static_cast<std::size_t>(byteCount));
  if (byteCount != 0)
    std::memcpy(destination.data(), base, destination.size());
}

void nativeGlobalBefore(std::uint64_t targetOrdinal, void *base,
                        std::uint64_t byteCount) {
  if (!activeExecution ||
      targetOrdinal >= activeExecution->globalBefore.size() || !base ||
      activeExecution->sawGlobalBefore[targetOrdinal]) {
    recordExecutionError("global-before callback has an invalid projection");
    return;
  }
  copyGlobalBytes(activeExecution->globalBefore[targetOrdinal], base,
                  byteCount);
  activeExecution->sawGlobalBefore[targetOrdinal] = true;
}

void nativeGlobalAfter(std::uint64_t targetOrdinal, void *base,
                       std::uint64_t byteCount) {
  if (!activeExecution ||
      targetOrdinal >= activeExecution->globalAfter.size() || !base ||
      activeExecution->sawGlobalAfter[targetOrdinal] ||
      !activeExecution->sawGlobalBefore[targetOrdinal] ||
      activeExecution->globalBefore[targetOrdinal].size() != byteCount) {
    recordExecutionError("global-after callback has an invalid projection");
    return;
  }
  copyGlobalBytes(activeExecution->globalAfter[targetOrdinal], base, byteCount);
  activeExecution->sawGlobalAfter[targetOrdinal] = true;
}

void nativeBlockActivation(std::uint64_t ordinal) {
  if (!activeExecution ||
      ordinal >= activeExecution->blockActivationCounts.size()) {
    recordExecutionError("block activation callback has an invalid ordinal");
    return;
  }
  std::uint64_t &count = activeExecution->blockActivationCounts[ordinal];
  if (count == std::numeric_limits<std::uint64_t>::max()) {
    recordExecutionError("block activation count overflowed");
    return;
  }
  ++count;
}

void recordWorkloadCaptureError(std::error_code code, llvm::StringRef message) {
  if (!activeWorkloadCapture || activeWorkloadCapture->error)
    return;
  activeWorkloadCapture->errorCode = code;
  activeWorkloadCapture->error = message.str();
}

void copyWorkloadCaptureBytes(std::vector<std::uint8_t> &destination,
                              const std::uint8_t *base, std::size_t byteCount) {
  destination.resize(byteCount);
  if (byteCount != 0)
    std::memcpy(destination.data(), base, byteCount);
}

bool reserveWorkloadCaptureBytes(WorkloadCaptureContext &context,
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

void workloadCaptureRegisterObject(void *base, std::uint64_t extentFactor0,
                                   std::uint64_t extentFactor1) {
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
  const std::uint64_t byteCount = extentFactor0 * extentFactor1;
  if (byteCount > std::numeric_limits<std::size_t>::max()) {
    recordWorkloadCaptureError(
        std::make_error_code(std::errc::not_supported),
        "program object extent exceeds the host addressable domain");
    return;
  }
  WorkloadCaptureContext &context = *activeWorkloadCapture;
  for (auto &object : context.runtimeObjects) {
    if (object.first != base)
      continue;
    object.second = static_cast<std::size_t>(byteCount);
    return;
  }
  context.runtimeObjects.push_back(
      {static_cast<std::uint8_t *>(base), static_cast<std::size_t>(byteCount)});
}

std::optional<std::pair<std::uint64_t, std::uint64_t>>
resolveRuntimeObject(void *pointer, bool admitOnePast = false) {
  if (!activeWorkloadCapture || !pointer)
    return std::nullopt;
  const std::uintptr_t address = reinterpret_cast<std::uintptr_t>(pointer);
  std::optional<std::pair<std::uint64_t, std::uint64_t>> onePast;
  for (std::size_t ordinal = activeWorkloadCapture->runtimeObjects.size();
       ordinal != 0; --ordinal) {
    const auto &object = activeWorkloadCapture->runtimeObjects[ordinal - 1];
    const std::uintptr_t base = reinterpret_cast<std::uintptr_t>(object.first);
    if (address < base)
      continue;
    const std::uintptr_t offset = address - base;
    if (offset < object.second)
      return std::pair<std::uint64_t, std::uint64_t>{ordinal - 1, offset};
    if (!admitOnePast || offset != object.second)
      continue;
    if (onePast)
      return std::nullopt;
    onePast = std::pair<std::uint64_t, std::uint64_t>{ordinal - 1, offset};
  }
  return onePast;
}

bool rangesOverlap(std::uint64_t lhsOffset, std::uint64_t lhsBytes,
                   std::uint64_t rhsOffset, std::uint64_t rhsBytes) {
  if (lhsBytes == 0 || rhsBytes == 0)
    return false;
  return lhsOffset < rhsOffset + rhsBytes && rhsOffset < lhsOffset + lhsBytes;
}

void eraseTrackedPointers(std::uint64_t objectOrdinal, std::uint64_t byteOffset,
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
          activeWorkloadCapture->runtimeObjects[resolved->first].second -
              resolved->second) {
    recordWorkloadCaptureError(
        std::make_error_code(std::errc::not_supported),
        "memory write is outside the runtime object registry");
    return;
  }
  eraseTrackedPointers(resolved->first, resolved->second, byteCount);
}

bool validatePointerCaptureLayout(std::uint64_t addressSpace,
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
        activeWorkloadCapture->runtimeObjects[runtimeObject].first);
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
      activeWorkloadCapture->runtimeObjects[storageTarget->first].second -
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

bool projectTrackedPointer(const WorkloadCaptureActiveCall &active,
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
    const auto &[base, byteCount] = context.runtimeObjects[resolved->first];
    if (!reserveWorkloadCaptureBytes(context, active, byteCount))
      return;
    active.objectBases.push_back(base);
    capture.objects.emplace_back();
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

CanonicalValueSequence
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

void workloadCaptureStream(std::uint64_t ordinal, void *base,
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
            context, active, context.runtimeObjects[runtimeOrdinal].second))
      return;
    copyWorkloadCaptureBytes(capture.objects[ordinal].finalBytes, base,
                             context.runtimeObjects[runtimeOrdinal].second);
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

std::string uniqueName(const llvm::Module &module, llvm::StringRef prefix) {
  std::string candidate = prefix.str();
  std::uint64_t suffix = 0;
  while (module.getNamedValue(candidate))
    candidate = (prefix + "." + llvm::Twine(++suffix)).str();
  return candidate;
}

llvm::Align requiredRuntimeObjectAlignment(const llvm::Module &module) {
  llvm::Align result(alignof(std::max_align_t));
  auto include = [&](llvm::MaybeAlign alignment) {
    if (alignment && alignment->value() > result.value())
      result = *alignment;
  };

  for (const llvm::Function &function : module) {
    for (unsigned ordinal = 0; ordinal < function.arg_size(); ++ordinal)
      include(function.getParamAlign(ordinal));
    for (const llvm::Instruction &instruction : llvm::instructions(function)) {
      if (const auto *load = llvm::dyn_cast<llvm::LoadInst>(&instruction))
        include(load->getAlign());
      else if (const auto *store =
                   llvm::dyn_cast<llvm::StoreInst>(&instruction))
        include(store->getAlign());
      else if (const auto *rmw =
                   llvm::dyn_cast<llvm::AtomicRMWInst>(&instruction))
        include(rmw->getAlign());
      else if (const auto *compare =
                   llvm::dyn_cast<llvm::AtomicCmpXchgInst>(&instruction))
        include(compare->getAlign());
      else if (const auto *memory =
                   llvm::dyn_cast<llvm::MemIntrinsic>(&instruction)) {
        include(memory->getDestAlign());
        if (const auto *transfer =
                llvm::dyn_cast<llvm::MemTransferInst>(memory))
          include(transfer->getSourceAlign());
      } else if (const auto *call =
                     llvm::dyn_cast<llvm::CallBase>(&instruction)) {
        for (unsigned ordinal = 0; ordinal < call->arg_size(); ++ordinal)
          include(call->getParamAlign(ordinal));
      }
    }
  }
  return result;
}

llvm::Expected<std::vector<std::uint8_t>>
definedMemoryBytes(const RuntimeMemoryObject &object) {
  std::vector<std::uint8_t> bytes;
  bytes.reserve(object.initialBytes.size());
  for (const SemanticMemoryByte &byte : object.initialBytes) {
    if (byte.state != SemanticState::Defined)
      return unsupported(
          "native execution requires Defined runtime memory bytes");
    bytes.push_back(byte.value);
  }
  return bytes;
}

llvm::Error patchRuntimePointerObjects(
    const llvm::Module &module,
    llvm::ArrayRef<RuntimeMemoryObject> sourceObjects,
    llvm::MutableArrayRef<AlignedByteStorage> nativeObjects) {
  if (sourceObjects.size() != nativeObjects.size())
    return invalid("runtime pointer patch has mismatched object tables");
  const llvm::DataLayout &layout = module.getDataLayout();
  for (auto [storageOrdinal, object] : llvm::enumerate(sourceObjects)) {
    for (const RuntimeMemoryPointer &stored : object.pointerValues) {
      if (stored.addressSpace != 0)
        return unsupported(
            "native execution cannot materialize a nonzero-address-space "
            "stored pointer");
      if (stored.target.objectOrdinal >= nativeObjects.size())
        return invalid("stored pointer target object is out of range");
      const unsigned pointerBits =
          layout.getPointerSizeInBits(stored.addressSpace);
      if (pointerBits != sizeof(std::uintptr_t) * 8 || pointerBits % 8 != 0)
        return unsupported(
            "native execution pointer representation differs from the host");
      if (stored.target.byteOffset.getBitWidth() > 64 &&
          !stored.target.byteOffset.isSignedIntN(64))
        return unsupported(
            "native execution pointer byte offset exceeds host range");
      const std::int64_t signedOffset =
          stored.target.byteOffset.sextOrTrunc(64).getSExtValue();
      AlignedByteStorage &target = nativeObjects[stored.target.objectOrdinal];
      if (signedOffset < 0 ||
          static_cast<std::uint64_t>(signedOffset) > target.size())
        return unsupported(
            "native execution stored pointer is outside its runtime object");
      AlignedByteStorage &storage = nativeObjects[storageOrdinal];
      const std::size_t pointerBytes = pointerBits / 8;
      if (stored.storageByteOffset > storage.size() ||
          pointerBytes > storage.size() - stored.storageByteOffset)
        return invalid("stored pointer storage range is out of bounds");
      std::uint8_t *nativeTarget =
          target.data() + static_cast<std::size_t>(signedOffset);
      std::memcpy(storage.data() + stored.storageByteOffset, &nativeTarget,
                  pointerBytes);
    }
  }
  return llvm::Error::success();
}

llvm::Expected<llvm::Constant *>
definedScalarConstant(llvm::Type *type, const SemanticLane &lane) {
  if (lane.state != SemanticState::Defined)
    return unsupported(
        "native execution requires Defined scalar and vector inputs");
  if (auto *integer = llvm::dyn_cast<llvm::IntegerType>(type)) {
    if (integer->getBitWidth() != lane.bits.getBitWidth())
      return invalid("entry integer width differs from the workload ABI");
    return llvm::ConstantInt::get(integer, lane.bits);
  }
  if (type->isFloatingPointTy()) {
    if (type->getPrimitiveSizeInBits() != lane.bits.getBitWidth())
      return invalid("entry floating width differs from the workload ABI");
    return llvm::ConstantFP::get(
        type->getContext(), llvm::APFloat(type->getFltSemantics(), lane.bits));
  }
  return unsupported("entry value type has no native constant provider");
}

llvm::Expected<llvm::Constant *>
definedValueConstant(llvm::Type *type, const CanonicalValueSequence &sequence) {
  if (sequence.tokenCount != 1)
    return invalid("entry value input does not contain exactly one token");
  if (auto *vector = llvm::dyn_cast<llvm::FixedVectorType>(type)) {
    if (sequence.lanes.size() != vector->getNumElements())
      return invalid("entry vector lane count differs from the workload ABI");
    std::vector<llvm::Constant *> elements;
    elements.reserve(sequence.lanes.size());
    for (const SemanticLane &lane : sequence.lanes) {
      llvm::Expected<llvm::Constant *> element =
          definedScalarConstant(vector->getElementType(), lane);
      if (!element)
        return element.takeError();
      elements.push_back(*element);
    }
    return llvm::ConstantVector::get(elements);
  }
  if (sequence.lanes.size() != 1)
    return invalid("entry scalar input has a non-scalar lane sequence");
  return definedScalarConstant(type, sequence.lanes.front());
}

const StructuredRuntimeValueEntry *
findRuntimeValue(const StructuredProgramSimulationRuntimeInput &input,
                 std::uint64_t argumentOrdinal) {
  auto found = llvm::lower_bound(
      input.runtimeValues, argumentOrdinal,
      [](const StructuredRuntimeValueEntry &entry, std::uint64_t ordinal) {
        return entry.argumentOrdinal < ordinal;
      });
  return found != input.runtimeValues.end() &&
                 found->argumentOrdinal == argumentOrdinal
             ? &*found
             : nullptr;
}

const StructuredPointerBindingEntry *
findPointerBinding(const StructuredProgramSimulationRuntimeInput &input,
                   std::uint64_t argumentOrdinal) {
  auto found = llvm::lower_bound(
      input.pointerBindings, argumentOrdinal,
      [](const StructuredPointerBindingEntry &entry, std::uint64_t ordinal) {
        return entry.argumentOrdinal < ordinal;
      });
  return found != input.pointerBindings.end() &&
                 found->argumentOrdinal == argumentOrdinal
             ? &*found
             : nullptr;
}

llvm::Expected<const CanonicalValueSequence *>
valueForArgument(const StructuredProgramSimulationWorkload &workload,
                 const StructuredProgramSimulationRuntimeInput &input,
                 std::uint64_t argumentOrdinal) {
  const StructuredProgramArgumentSource &source =
      workload.argumentPlan[argumentOrdinal];
  if (const auto *fixed = std::get_if<CanonicalValueSequence>(&source))
    return fixed;
  if (std::holds_alternative<StructuredRuntimeValueInput>(source)) {
    const StructuredRuntimeValueEntry *runtime =
        findRuntimeValue(input, argumentOrdinal);
    if (!runtime)
      return invalid("runtime value table is not total over the workload");
    return &runtime->value;
  }
  return invalid("a memory argument was requested as a value");
}

llvm::Value *castPointerTo(llvm::IRBuilder<> &builder, llvm::Value *pointer,
                           llvm::PointerType *target) {
  auto *source = llvm::cast<llvm::PointerType>(pointer->getType());
  if (source->getAddressSpace() == target->getAddressSpace())
    return pointer;
  return builder.CreateAddrSpaceCast(pointer, target);
}

llvm::Expected<std::uint64_t> fixedStoreBytes(const llvm::DataLayout &layout,
                                              llvm::Type *type,
                                              llvm::StringRef what) {
  const llvm::TypeSize size = layout.getTypeStoreSize(type);
  if (size.isScalable() || size.getFixedValue() == 0)
    return unsupported(what + " has no fixed nonzero native storage size");
  return size.getFixedValue();
}

llvm::Expected<std::vector<MemoryTargetPlan>>
buildMemoryPlans(const StructuredProgramSimulationWorkload &workload,
                 const StructuredProgramSimulationRuntimeInput &input,
                 const frontend::StructuredProgramCandidateView &view) {
  std::vector<MemoryTargetPlan> plans;
  plans.reserve(workload.observableContract.memories.size());
  for (const StructuredProgramMemoryObservable &observable :
       workload.observableContract.memories) {
    MemoryTargetPlan plan;
    plan.form = observable.form;
    if (const auto *argument =
            std::get_if<EntryPointerArgumentTarget>(&observable.target)) {
      const StructuredPointerBindingEntry *binding =
          findPointerBinding(input, argument->argumentOrdinal);
      if (!binding ||
          binding->binding.objectOrdinal >= input.memoryObjects.size())
        return invalid("memory observable has no exact runtime object");
      plan.objectOrdinal = binding->binding.objectOrdinal;
      plan.byteCount = input.memoryObjects[binding->binding.objectOrdinal]
                           .initialBytes.size();
    } else {
      const frontend::StructuredEntityRef &reference =
          std::get<GlobalObjectTarget>(observable.target).global;
      llvm::Expected<frontend::StructuredEntity> entity =
          view.resolve(reference);
      if (!entity)
        return entity.takeError();
      auto global =
          llvm::dyn_cast_or_null<mlir::LLVM::GlobalOp>(entity->operation);
      if (!global)
        return invalid(
            "global observable does not resolve to llvm.mlir.global");
      plan.globalSymbol = global.getSymName().str();
    }
    plans.push_back(std::move(plan));
  }
  return plans;
}

struct CallbackNames {
  std::string wrapper;
  std::string runtimeObject;
  std::optional<std::string> invalidThreadExtent;
  std::optional<std::string> blockActivation;
  std::optional<std::string> returnValue;
  std::optional<std::string> globalBefore;
  std::optional<std::string> globalAfter;
  std::optional<NativeChannelCallbackNames> channels;
};

llvm::Expected<CallbackNames> instrumentExecution(
    llvm::Module &module, llvm::StringRef entrySymbol,
    const StructuredProgramSimulationWorkload &workload,
    const StructuredProgramSimulationRuntimeInput &input,
    const detail::ResolvedStructuredProgramContext &sourceContext,
    std::vector<MemoryTargetPlan> &memoryPlans,
    NativeExecutionContext &capture) {
  llvm::Function *entry = module.getFunction(entrySymbol);
  if (!entry || entry->isDeclaration())
    return invalid("exact entry is absent after Structured lowering");
  if (entry->isVarArg())
    return unsupported("variadic Structured entries lack a workload ABI");
  if (entry->arg_size() != workload.argumentPlan.size())
    return invalid("lowered entry arguments differ from the exact workload");

  const llvm::DataLayout &layout = module.getDataLayout();
  const llvm::Align objectAlignment = requiredRuntimeObjectAlignment(module);
  capture.objects.reserve(input.memoryObjects.size());
  for (const RuntimeMemoryObject &object : input.memoryObjects) {
    llvm::Expected<std::vector<std::uint8_t>> bytes =
        definedMemoryBytes(object);
    if (!bytes)
      return bytes.takeError();
    llvm::Expected<AlignedByteStorage> storage =
        AlignedByteStorage::create(*bytes, objectAlignment);
    if (!storage)
      return storage.takeError();
    capture.objects.push_back(std::move(*storage));
  }
  if (llvm::Error error = patchRuntimePointerObjects(
          module, input.memoryObjects, capture.objects))
    return std::move(error);
  capture.littleEndian = layout.isLittleEndian();
  capture.globalBefore.resize(memoryPlans.size());
  capture.globalAfter.resize(memoryPlans.size());
  capture.sawGlobalBefore.resize(memoryPlans.size());
  capture.sawGlobalAfter.resize(memoryPlans.size());

  CallbackNames names;
  names.wrapper = uniqueName(module, "__loom_structured_entry");
  names.runtimeObject = uniqueName(module, "__loom_structured_runtime_object");
  if (workload.observableContract.returnValue)
    names.returnValue = uniqueName(module, "__loom_structured_return_value");
  if (llvm::any_of(memoryPlans, [](const MemoryTargetPlan &plan) {
        return !plan.globalSymbol.empty();
      })) {
    names.globalBefore = uniqueName(module, "__loom_structured_global_before");
    names.globalAfter = uniqueName(module, "__loom_structured_global_after");
  }

  llvm::LLVMContext &context = module.getContext();
  llvm::Type *voidType = llvm::Type::getVoidTy(context);
  llvm::Type *i64Type = llvm::Type::getInt64Ty(context);
  llvm::PointerType *pointerType = llvm::PointerType::getUnqual(context);
  llvm::FunctionType *objectCallbackType =
      llvm::FunctionType::get(pointerType, {i64Type}, false);
  llvm::FunctionCallee objectCallback =
      module.getOrInsertFunction(names.runtimeObject, objectCallbackType);

  llvm::FunctionCallee returnCallback;
  if (names.returnValue) {
    llvm::FunctionType *type =
        llvm::FunctionType::get(voidType, {pointerType, i64Type}, false);
    returnCallback = module.getOrInsertFunction(*names.returnValue, type);
  }

  llvm::FunctionCallee beforeCallback;
  llvm::FunctionCallee afterCallback;
  if (names.globalBefore) {
    llvm::FunctionType *type = llvm::FunctionType::get(
        voidType, {i64Type, pointerType, i64Type}, false);
    beforeCallback = module.getOrInsertFunction(*names.globalBefore, type);
    afterCallback = module.getOrInsertFunction(*names.globalAfter, type);
  }

  llvm::Function *wrapper = llvm::Function::Create(
      llvm::FunctionType::get(voidType, false),
      llvm::GlobalValue::ExternalLinkage, names.wrapper, module);
  llvm::BasicBlock *block = llvm::BasicBlock::Create(context, "entry", wrapper);
  llvm::IRBuilder<> builder(block);

  std::vector<llvm::Value *> arguments;
  arguments.reserve(entry->arg_size());
  for (std::uint64_t ordinal = 0; ordinal < entry->arg_size(); ++ordinal) {
    llvm::Type *type = entry->getFunctionType()->getParamType(ordinal);
    if (auto *pointer = llvm::dyn_cast<llvm::PointerType>(type)) {
      const StructuredPointerBindingEntry *binding =
          findPointerBinding(input, ordinal);
      if (!binding)
        return invalid("entry pointer has no exact runtime binding");
      llvm::Value *objectOrdinal =
          llvm::ConstantInt::get(i64Type, binding->binding.objectOrdinal);
      llvm::Value *base = builder.CreateCall(objectCallback, {objectOrdinal});
      llvm::Value *view = builder.CreateConstGEP1_64(
          llvm::Type::getInt8Ty(context), base, binding->binding.byteOffset);
      arguments.push_back(castPointerTo(builder, view, pointer));
      continue;
    }
    llvm::Expected<const CanonicalValueSequence *> sequence =
        valueForArgument(workload, input, ordinal);
    if (!sequence)
      return sequence.takeError();
    llvm::Expected<llvm::Constant *> value =
        definedValueConstant(type, **sequence);
    if (!value)
      return value.takeError();
    arguments.push_back(*value);
  }

  for (std::uint64_t ordinal = 0; ordinal < memoryPlans.size(); ++ordinal) {
    MemoryTargetPlan &plan = memoryPlans[ordinal];
    if (plan.globalSymbol.empty())
      continue;
    llvm::GlobalVariable *global =
        module.getGlobalVariable(plan.globalSymbol, true);
    if (!global || global->isDeclaration())
      return unsupported("observed global has no native storage provider");
    llvm::Expected<std::uint64_t> bytes =
        fixedStoreBytes(layout, global->getValueType(), "observed global");
    if (!bytes)
      return bytes.takeError();
    plan.byteCount = *bytes;
    llvm::Value *pointer = castPointerTo(builder, global, pointerType);
    builder.CreateCall(beforeCallback,
                       {llvm::ConstantInt::get(i64Type, ordinal), pointer,
                        llvm::ConstantInt::get(i64Type, plan.byteCount)});
  }

  llvm::CallInst *call = builder.CreateCall(entry, arguments);
  call->setCallingConv(entry->getCallingConv());
  call->setAttributes(entry->getAttributes());
  if (workload.observableContract.returnValue) {
    if (entry->getReturnType()->isVoidTy() || !sourceContext.returnShape)
      return invalid("selected return observation has no concrete result");
    llvm::Expected<std::uint64_t> bytes = fixedStoreBytes(
        layout, entry->getReturnType(), "Structured entry return");
    if (!bytes)
      return bytes.takeError();
    capture.returnShape = sourceContext.returnShape;
    capture.returnByteCount = *bytes;
    llvm::AllocaInst *storage = builder.CreateAlloca(entry->getReturnType());
    storage->setAlignment(layout.getABITypeAlign(entry->getReturnType()));
    builder.CreateStore(call, storage);
    builder.CreateCall(returnCallback,
                       {storage, llvm::ConstantInt::get(i64Type, *bytes)});
  }

  for (std::uint64_t ordinal = 0; ordinal < memoryPlans.size(); ++ordinal) {
    const MemoryTargetPlan &plan = memoryPlans[ordinal];
    if (plan.globalSymbol.empty())
      continue;
    llvm::GlobalVariable *global =
        module.getGlobalVariable(plan.globalSymbol, true);
    llvm::Value *pointer = castPointerTo(builder, global, pointerType);
    builder.CreateCall(afterCallback,
                       {llvm::ConstantInt::get(i64Type, ordinal), pointer,
                        llvm::ConstantInt::get(i64Type, plan.byteCount)});
  }
  builder.CreateRetVoid();
  if (llvm::verifyModule(module, &llvm::errs()))
    return invalid("instrumented Structured execution module does not verify");
  return names;
}

llvm::Expected<WorkloadCaptureContext> prepareWorkloadCaptureContext(
    const WorkloadBackedSimulationInputCapturePlan &plan,
    NativeExecutionContext &execution,
    const StructuredProgramSimulationRuntimeInput &input,
    mlir::Operation *layoutScope, WorkloadBackedSimulationInputVisitor *visitor,
    std::uint64_t maxRetainedBytes) {
  WorkloadCaptureContext capture;
  capture.visitor = visitor;
  capture.maxRetainedBytes = maxRetainedBytes;
  capture.rootCount = plan.memoryRoots.size();
  capture.littleEndian = execution.littleEndian;
  capture.runtimeObjects.reserve(execution.objects.size());
  for (AlignedByteStorage &object : execution.objects)
    capture.runtimeObjects.push_back({object.data(), object.size()});
  if (input.memoryObjects.size() != execution.objects.size())
    return invalid("workload capture pointer table differs from runtime "
                   "objects");
  for (auto [storageOrdinal, object] : llvm::enumerate(input.memoryObjects)) {
    for (const RuntimeMemoryPointer &pointer : object.pointerValues) {
      auto layout =
          ::loom::resolvePointerLayout(layoutScope, pointer.addressSpace);
      if (!layout)
        return layout.takeError();
      if (layout->kind != ::loom::PointerLayoutKind::StableIntegral ||
          layout->representationBits % 8 != 0)
        return unsupported("runtime pointer payload has no stable-integral "
                           "native capture projection");
      capture.pointerPayloads[{storageOrdinal, pointer.storageByteOffset}] =
          TrackedPointerPayload{storageOrdinal, pointer.storageByteOffset,
                                pointer.addressSpace,
                                layout->representationBits, pointer.target};
    }
  }
  capture.coordinateShapes.reserve(plan.denseCoordinates.size());
  for (auto [dimension, coordinate] : llvm::enumerate(plan.denseCoordinates)) {
    if (coordinate.dimension != dimension || coordinate.byteCount == 0 ||
        coordinate.byteCount > std::numeric_limits<std::uint32_t>::max() / 8)
      return invalid("dense coordinate capture is not canonical");
    capture.coordinateShapes.push_back(WorkloadCaptureValueShape{
        coordinate.dimension, 1,
        static_cast<std::uint32_t>(coordinate.byteCount * 8),
        coordinate.byteCount, std::nullopt});
  }
  for (const SimulationValueInputCapture &input : plan.valueInputs) {
    if (input.valueInputOrdinal >= plan.valueInputs.size())
      return invalid("graph value input ordinal is outside its capture plan");
    if (!input.fixedValue)
      capture.runtimeValueShapes.push_back(WorkloadCaptureValueShape{
          input.valueInputOrdinal, input.lanesPerToken, input.laneBitWidth,
          input.byteCount, input.pointerTarget});
  }
  capture.streamInputShapes.reserve(plan.streamInputs.size());
  for (auto [ordinal, stream] : llvm::enumerate(plan.streamInputs)) {
    if (stream.graphOrdinal != ordinal || stream.lanesPerToken == 0 ||
        stream.laneBitWidth == 0 || stream.byteCount == 0)
      return invalid("graph stream inputs are not dense in ABI order");
    capture.streamInputShapes.push_back(WorkloadCaptureValueShape{
        stream.graphOrdinal, stream.lanesPerToken, stream.laneBitWidth,
        stream.byteCount, std::nullopt});
  }
  capture.valueResultShapes.reserve(plan.valueResults.size());
  for (auto [ordinal, result] : llvm::enumerate(plan.valueResults)) {
    if (result.valueResultOrdinal != ordinal)
      return invalid("graph value results are not dense in ABI order");
    capture.valueResultShapes.push_back(WorkloadCaptureValueShape{
        result.valueResultOrdinal, result.lanesPerToken, result.laneBitWidth,
        result.byteCount, std::nullopt});
  }
  capture.streamOutputShapes.reserve(plan.streamOutputs.size());
  for (auto [ordinal, stream] : llvm::enumerate(plan.streamOutputs)) {
    if (stream.graphOrdinal != ordinal || stream.lanesPerToken == 0 ||
        stream.laneBitWidth == 0 || stream.byteCount == 0)
      return invalid("graph stream outputs are not dense in ABI order");
    capture.streamOutputShapes.push_back(WorkloadCaptureValueShape{
        stream.graphOrdinal, stream.lanesPerToken, stream.laneBitWidth,
        stream.byteCount, std::nullopt});
  }
  return capture;
}

llvm::Error runInstrumentedExecution(
    llvm::orc::ThreadSafeModule module, const CallbackNames &names,
    NativeExecutionContext &capture, std::unique_ptr<llvm::orc::LLJIT> jit,
    const WorkloadCaptureCallbackNames *workloadCaptureNames = nullptr,
    WorkloadCaptureContext *workloadCapture = nullptr) {
  if ((workloadCaptureNames == nullptr) != (workloadCapture == nullptr))
    return invalid("workload capture callback state is partial");
  llvm::orc::JITDylib &dylib = jit->getMainJITDylib();
  if (llvm::Expected<std::unique_ptr<llvm::orc::DynamicLibrarySearchGenerator>>
          generator =
              llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(
                  jit->getDataLayout().getGlobalPrefix()))
    dylib.addGenerator(std::move(*generator));
  else
    return classifyJitError(generator.takeError(),
                            "cannot bind host process symbols");

  llvm::orc::SymbolMap callbacks;
  callbacks[jit->mangleAndIntern(names.runtimeObject)] = {
      llvm::orc::ExecutorAddr::fromPtr(&nativeRuntimeObject),
      llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};
  if (names.invalidThreadExtent)
    callbacks[jit->mangleAndIntern(*names.invalidThreadExtent)] = {
        llvm::orc::ExecutorAddr::fromPtr(&nativeInvalidLogicalThreadExtent),
        llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};
  if (names.blockActivation)
    callbacks[jit->mangleAndIntern(*names.blockActivation)] = {
        llvm::orc::ExecutorAddr::fromPtr(&nativeBlockActivation),
        llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};
  if (names.channels) {
    callbacks[jit->mangleAndIntern(names.channels->create)] = {
        llvm::orc::ExecutorAddr::fromPtr(&nativeLogicalChannelCreate),
        llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};
    callbacks[jit->mangleAndIntern(names.channels->send)] = {
        llvm::orc::ExecutorAddr::fromPtr(&nativeLogicalChannelSend),
        llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};
    callbacks[jit->mangleAndIntern(names.channels->receive)] = {
        llvm::orc::ExecutorAddr::fromPtr(&nativeLogicalChannelReceive),
        llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};
  }
  if (names.returnValue)
    callbacks[jit->mangleAndIntern(*names.returnValue)] = {
        llvm::orc::ExecutorAddr::fromPtr(&nativeReturnValue),
        llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};
  if (names.globalBefore) {
    callbacks[jit->mangleAndIntern(*names.globalBefore)] = {
        llvm::orc::ExecutorAddr::fromPtr(&nativeGlobalBefore),
        llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};
    callbacks[jit->mangleAndIntern(*names.globalAfter)] = {
        llvm::orc::ExecutorAddr::fromPtr(&nativeGlobalAfter),
        llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};
  }
  if (workloadCaptureNames) {
    callbacks[jit->mangleAndIntern(workloadCaptureNames->begin)] = {
        llvm::orc::ExecutorAddr::fromPtr(&workloadCaptureBegin),
        llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};
    callbacks[jit->mangleAndIntern(workloadCaptureNames->end)] = {
        llvm::orc::ExecutorAddr::fromPtr(&workloadCaptureEnd),
        llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};
    if (workloadCaptureNames->registerObject)
      callbacks[jit->mangleAndIntern(*workloadCaptureNames->registerObject)] = {
          llvm::orc::ExecutorAddr::fromPtr(&workloadCaptureRegisterObject),
          llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};
    if (workloadCaptureNames->coordinate)
      callbacks[jit->mangleAndIntern(*workloadCaptureNames->coordinate)] = {
          llvm::orc::ExecutorAddr::fromPtr(&workloadCaptureCoordinate),
          llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};
    if (workloadCaptureNames->memoryRoot)
      callbacks[jit->mangleAndIntern(*workloadCaptureNames->memoryRoot)] = {
          llvm::orc::ExecutorAddr::fromPtr(&workloadCaptureMemoryRoot),
          llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};
    if (workloadCaptureNames->value)
      callbacks[jit->mangleAndIntern(*workloadCaptureNames->value)] = {
          llvm::orc::ExecutorAddr::fromPtr(&workloadCaptureValue),
          llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};
    if (workloadCaptureNames->streamInput)
      callbacks[jit->mangleAndIntern(*workloadCaptureNames->streamInput)] = {
          llvm::orc::ExecutorAddr::fromPtr(&workloadCaptureStreamInput),
          llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};
    if (workloadCaptureNames->result)
      callbacks[jit->mangleAndIntern(*workloadCaptureNames->result)] = {
          llvm::orc::ExecutorAddr::fromPtr(&workloadCaptureResult),
          llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};
    if (workloadCaptureNames->streamOutput)
      callbacks[jit->mangleAndIntern(*workloadCaptureNames->streamOutput)] = {
          llvm::orc::ExecutorAddr::fromPtr(&workloadCaptureStreamOutput),
          llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};
    if (workloadCaptureNames->memoryWrite)
      callbacks[jit->mangleAndIntern(*workloadCaptureNames->memoryWrite)] = {
          llvm::orc::ExecutorAddr::fromPtr(&workloadCaptureMemoryWrite),
          llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};
    if (workloadCaptureNames->pointerRead)
      callbacks[jit->mangleAndIntern(*workloadCaptureNames->pointerRead)] = {
          llvm::orc::ExecutorAddr::fromPtr(&workloadCapturePointerRead),
          llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};
    if (workloadCaptureNames->pointerWrite)
      callbacks[jit->mangleAndIntern(*workloadCaptureNames->pointerWrite)] = {
          llvm::orc::ExecutorAddr::fromPtr(&workloadCapturePointerWrite),
          llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};
  }
  if (llvm::Error error =
          dylib.define(llvm::orc::absoluteSymbols(std::move(callbacks))))
    return classifyJitError(std::move(error),
                            "cannot bind execution callbacks");
  if (llvm::Error error = jit->addIRModule(std::move(module)))
    return classifyJitError(std::move(error),
                            "cannot add the Structured execution module");
  if (llvm::Error error = jit->initialize(dylib))
    return classifyJitError(std::move(error),
                            "cannot initialize the Structured program");
  llvm::Expected<llvm::orc::ExecutorAddr> wrapper = jit->lookup(names.wrapper);
  if (!wrapper)
    return classifyJitError(wrapper.takeError(),
                            "cannot materialize the Structured entry");
  if (activeExecution || activeWorkloadCapture)
    return executionFailed("nested native Structured execution is unsupported");
  activeExecution = &capture;
  activeWorkloadCapture = workloadCapture;
  using Wrapper = void();
  wrapper->toPtr<Wrapper>()();
  activeWorkloadCapture = nullptr;
  activeExecution = nullptr;
  if (llvm::Error error = jit->deinitialize(dylib))
    return classifyJitError(std::move(error),
                            "cannot deinitialize the Structured program");
  if (capture.error)
    return executionFailed(*capture.error);
  for (const NativeExecutionContext::LogicalChannel &channel :
       capture.logicalChannels)
    for (std::uint64_t cursor : channel.receiverCursors)
      if (cursor != channel.messages.size())
        return executionFailed(
            "logical channel endpoint did not consume its complete sequence");
  if (workloadCapture) {
    if (workloadCapture->error) {
      const std::error_code code = workloadCapture->errorCode.value_or(
          std::make_error_code(std::errc::io_error));
      return llvm::createStringError(code, "native_workload_capture: %s",
                                     workloadCapture->error->c_str());
    }
    if (!workloadCapture->activeCalls.empty())
      return executionFailed(
          "workload-backed capture left an incomplete invocation");
  }
  return llvm::Error::success();
}

struct NativeProgramExecutionResult final {
  NativeStructuredProgramObservations observations;
};

llvm::Expected<NativeProgramExecutionResult> executePreparedProgramModule(
    mlir::OwningOpRef<mlir::ModuleOp> module,
    const frontend::StructuredProgramCandidate &sourceProgram,
    const CanonicalSimulationWorkload &workload,
    const CanonicalSimulationRuntimeInput &runtimeInput,
    const frontend::StructuredProgramCandidate *profileProgram,
    bool projectOwnership, mlir::Operation *selectedOperation = nullptr,
    const WorkloadBackedSimulationInputCapturePlan *capturePlan = nullptr,
    WorkloadBackedSimulationInputVisitor *captureVisitor = nullptr,
    std::uint64_t maxRetainedCaptureBytes =
        std::numeric_limits<std::uint64_t>::max()) {
  if (!module)
    return invalid("native execution has no Structured module");
  if ((selectedOperation == nullptr) != (capturePlan == nullptr))
    return invalid("workload-backed capture plan is partial");
  if ((captureVisitor == nullptr) != (capturePlan == nullptr))
    return invalid("workload-backed capture requires one streaming visitor");
  if (capturePlan && maxRetainedCaptureBytes == 0)
    return invalid("workload-backed capture byte limit must be positive");
  const StructuredProgramSimulationWorkload *structured =
      workload.structuredProgram();
  const StructuredProgramSimulationRuntimeInput *input =
      runtimeInput.structuredProgram();
  if (!structured || !input)
    return invalid("native execution requires Structured workload roots");

  auto sourceView = sourceProgram.view();
  if (!sourceView)
    return sourceView.takeError();
  auto verifiedWorkload = importSimulationWorkload(
      workload.canonicalBytes().bytes(), *sourceView, workload.identity());
  if (!verifiedWorkload)
    return verifiedWorkload.takeError();
  auto verifiedInput = importSimulationRuntimeInput(
      runtimeInput.canonicalBytes().bytes(), workload, *sourceView,
      runtimeInput.identity());
  if (!verifiedInput)
    return verifiedInput.takeError();
  auto sourceContext = detail::resolveStructuredProgramContext(
      *sourceView, structured->entryRef);
  if (!sourceContext)
    return sourceContext.takeError();
  auto entry = llvm::dyn_cast<mlir::LLVM::LLVMFuncOp>(sourceContext->entryOp);
  if (!entry)
    return invalid("exact workload entry is not llvm.func");
  auto plans = buildMemoryPlans(*structured, *input, *sourceView);
  if (!plans)
    return plans.takeError();

  NativeExecutionContext capture;
  std::optional<WorkloadCaptureCallbackNames> workloadCaptureNames;
  if (capturePlan) {
    auto names = instrumentWorkloadBackedCapture(*module, selectedOperation,
                                                 *capturePlan);
    if (!names)
      return names.takeError();
    workloadCaptureNames.emplace(std::move(*names));
  }
  std::optional<std::string> blockActivation;
  if (profileProgram) {
    auto callback = instrumentBlockActivations(
        *module, profileProgram->identity(), capture);
    if (!callback)
      return callback.takeError();
    blockActivation = std::move(*callback);
  }
  std::optional<native_detail::SelectedWholeProgramProjection>
      ownershipProjection;
  if (projectOwnership) {
    auto projection = projectSelectedWholeProgram(*module);
    if (!projection)
      return projection.takeError();
    ownershipProjection.emplace(std::move(*projection));
  }
  auto native = detail::lowerStructuredModuleToLlvm(std::move(module));
  if (!native)
    return native.takeError();
  if (llvm::Error error = detail::initializeNativeTarget())
    return std::move(error);
  auto targetJitOrError = llvm::orc::LLJITBuilder().create();
  if (!targetJitOrError)
    return classifyJitError(targetJitOrError.takeError(),
                            "cannot create host JIT");
  std::unique_ptr<llvm::orc::LLJIT> targetJit = std::move(*targetJitOrError);

  CallbackNames callbackNames;
  std::optional<WorkloadCaptureContext> workloadCapture;
  llvm::Error preparation =
      native->withModuleDo([&](llvm::Module &module) -> llvm::Error {
        if (llvm::Error error =
                detail::retargetStructuredOracle(module, *targetJit))
          return error;
        auto names =
            instrumentExecution(module, entry.getSymName(), *structured, *input,
                                *sourceContext, *plans, capture);
        if (!names)
          return names.takeError();
        callbackNames = std::move(*names);
        if (llvm::Error error =
                detail::prepareDeterministicMathOracle(module, *targetJit))
          return error;
        if (ownershipProjection) {
          callbackNames.invalidThreadExtent =
              std::move(ownershipProjection->invalidThreadExtent);
          callbackNames.channels = std::move(ownershipProjection->channels);
        }
        callbackNames.blockActivation = std::move(blockActivation);
        if (capturePlan) {
          auto prepared = prepareWorkloadCaptureContext(
              *capturePlan, capture, *input, sourceContext->entryOp,
              captureVisitor, maxRetainedCaptureBytes);
          if (!prepared)
            return prepared.takeError();
          workloadCapture.emplace(std::move(*prepared));
        }
        return llvm::Error::success();
      });
  if (preparation)
    return std::move(preparation);
  if (llvm::Error error = runInstrumentedExecution(
          std::move(*native), callbackNames, capture, std::move(targetJit),
          workloadCaptureNames ? &*workloadCaptureNames : nullptr,
          workloadCapture ? &*workloadCapture : nullptr))
    return std::move(error);
  auto observations = buildObservations(*structured, *input, *plans, capture);
  if (!observations)
    return observations.takeError();
  return NativeProgramExecutionResult{std::move(*observations)};
}

llvm::Expected<NativeStructuredProgramObservations>
executeProgramModule(const frontend::StructuredProgramCandidate &program,
                     const frontend::StructuredProgramCandidate &sourceProgram,
                     const CanonicalSimulationWorkload &workload,
                     const CanonicalSimulationRuntimeInput &runtimeInput,
                     bool profileSourceBlocks, bool projectOwnership) {
  auto programView = program.view();
  if (!programView)
    return programView.takeError();
  mlir::OwningOpRef<mlir::ModuleOp> cloned(
      llvm::cast<mlir::ModuleOp>(program.module()->clone()));
  auto result = executePreparedProgramModule(
      std::move(cloned), sourceProgram, workload, runtimeInput,
      profileSourceBlocks ? &program : nullptr, projectOwnership);
  if (!result)
    return result.takeError();
  return std::move(result->observations);
}

} // namespace

llvm::Expected<NativeStructuredProgramObservations>
executeNativeStructuredProgram(
    const frontend::StructuredProgramCandidate &program,
    const CanonicalSimulationWorkload &workload,
    const CanonicalSimulationRuntimeInput &runtimeInput) {
  return executeProgramModule(program, program, workload, runtimeInput, true,
                              false);
}

llvm::Expected<NativeStructuredProgramObservations>
executeSelectedStructuredProgram(
    const frontend::StructuredProgramCandidate &selectedProgram,
    const frontend::StructuredProgramCandidate &sourceProgram,
    const CanonicalSimulationWorkload &workload,
    const CanonicalSimulationRuntimeInput &runtimeInput) {
  return executeProgramModule(selectedProgram, sourceProgram, workload,
                              runtimeInput, false, true);
}

llvm::Expected<NativeStructuredProgramObservations>
executeProfiledSelectedStructuredProgram(
    const frontend::StructuredProgramCandidate &selectedProgram,
    const frontend::StructuredProgramCandidate &sourceProgram,
    const CanonicalSimulationWorkload &workload,
    const CanonicalSimulationRuntimeInput &runtimeInput) {
  return executeProgramModule(selectedProgram, sourceProgram, workload,
                              runtimeInput, true, true);
}

llvm::Error visitWorkloadBackedSimulationInputCaptures(
    mlir::OwningOpRef<mlir::ModuleOp> preparedModule,
    mlir::Operation *selectedOperation,
    const WorkloadBackedSimulationInputCapturePlan &plan,
    const frontend::StructuredProgramCandidate &sourceProgram,
    const CanonicalSimulationWorkload &workload,
    const CanonicalSimulationRuntimeInput &runtimeInput,
    std::uint64_t maxRetainedCaptureBytes,
    WorkloadBackedSimulationInputVisitor visitor) {
  auto result = executePreparedProgramModule(
      std::move(preparedModule), sourceProgram, workload, runtimeInput, nullptr,
      false, selectedOperation, &plan, &visitor, maxRetainedCaptureBytes);
  if (!result)
    return result.takeError();
  return llvm::Error::success();
}

} // namespace loom::sim

llvm::Expected<loom::sim::NativeStructuredProgramObservations>
loom::sim::native_detail::visitProjectedWorkloadBackedSimulationInputCaptures(
    mlir::OwningOpRef<mlir::ModuleOp> selectedModule,
    mlir::Operation *selectedOperation,
    const WorkloadBackedSimulationInputCapturePlan &plan,
    const frontend::StructuredProgramCandidate &sourceProgram,
    const CanonicalSimulationWorkload &workload,
    const CanonicalSimulationRuntimeInput &runtimeInput,
    std::uint64_t maxRetainedCaptureBytes,
    WorkloadBackedSimulationInputVisitor visitor) {
  auto result = executePreparedProgramModule(
      std::move(selectedModule), sourceProgram, workload, runtimeInput, nullptr,
      true, selectedOperation, &plan, &visitor, maxRetainedCaptureBytes);
  if (!result)
    return result.takeError();
  return std::move(result->observations);
}
