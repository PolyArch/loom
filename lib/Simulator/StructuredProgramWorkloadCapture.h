#ifndef LOOM_SIMULATOR_STRUCTUREDPROGRAMWORKLOADCAPTURE_H
#define LOOM_SIMULATOR_STRUCTUREDPROGRAMWORKLOADCAPTURE_H

#include "StructuredProgramNativeExecutionInternal.h"

#include <map>
#include <set>
#include <system_error>

namespace loom::sim::native_detail {

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

struct WorkloadCaptureStackAllocation final {
  std::uint64_t allocationOrdinal = 0;
  std::size_t frameDepth = 0;
};

struct WorkloadCaptureRuntimeObject final {
  std::uint8_t *base = nullptr;
  std::size_t byteCount = 0;
  std::optional<WorkloadCaptureStackAllocation> stackAllocation;
  std::optional<NativeMemoryObjectSource> source;
};

struct WorkloadCaptureContext final {
  std::vector<std::optional<NativeSimulationCallCapture>> captures;
  std::vector<WorkloadCaptureValueShape> coordinateShapes;
  std::vector<WorkloadCaptureValueShape> runtimeValueShapes;
  std::vector<WorkloadCaptureValueShape> streamInputShapes;
  std::vector<WorkloadCaptureValueShape> valueResultShapes;
  std::vector<WorkloadCaptureValueShape> streamOutputShapes;
  std::vector<WorkloadCaptureRuntimeObject> runtimeObjects;
  std::vector<NativeMemoryObjectSource> programObjectSources;
  std::size_t stackFrameDepth = 0;
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

extern thread_local WorkloadCaptureContext *activeWorkloadCapture;

void workloadCaptureBegin();
void workloadCaptureEnd();
void workloadCaptureEnterStackFrame();
void workloadCaptureLeaveStackFrame();
void workloadCaptureEndStackObject(void *base, std::uint64_t allocation);
void workloadCaptureRegisterObject(void *base, std::uint64_t extentFactor0,
                                   std::uint64_t extentFactor1,
                                   ProgramObjectCaptureKind kind,
                                   std::uint64_t allocation);
void workloadCaptureMemoryRoot(std::uint64_t rootOrdinal, void *pointer);
void workloadCaptureCoordinate(std::uint64_t ordinal, void *base,
                               std::uint64_t byteCount);
void workloadCaptureValue(std::uint64_t ordinal, void *base,
                          std::uint64_t byteCount);
void workloadCaptureStreamInput(std::uint64_t ordinal, void *base,
                                std::uint64_t byteCount);
void workloadCaptureStreamOutput(std::uint64_t ordinal, void *base,
                                 std::uint64_t byteCount);
void workloadCaptureResult(std::uint64_t ordinal, void *base,
                           std::uint64_t byteCount);
void workloadCaptureMemoryWrite(void *storage, std::uint64_t byteCount);
void workloadCapturePointerRead(void *storage, void *value,
                                std::uint64_t addressSpace,
                                std::uint64_t representationBits,
                                std::uint64_t addressBits);
void workloadCapturePointerWrite(void *storage, void *value,
                                 std::uint64_t addressSpace,
                                 std::uint64_t representationBits,
                                 std::uint64_t addressBits,
                                 std::uint64_t rootHint);

} // namespace loom::sim::native_detail

#endif // LOOM_SIMULATOR_STRUCTUREDPROGRAMWORKLOADCAPTURE_H
