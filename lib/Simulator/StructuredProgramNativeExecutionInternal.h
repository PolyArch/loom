#ifndef LOOM_SIMULATOR_STRUCTUREDPROGRAMNATIVEEXECUTIONINTERNAL_H
#define LOOM_SIMULATOR_STRUCTUREDPROGRAMNATIVEEXECUTIONINTERNAL_H

#include "Simulator/NativeSimulationOracle.h"

#include "SimulationWireInternal.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/Error.h"

#include "mlir/IR/BuiltinOps.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace loom::sim::native_detail {

class AlignedByteStorage final {
public:
  static llvm::Expected<AlignedByteStorage>
  create(llvm::ArrayRef<std::uint8_t> bytes, llvm::Align alignment);

  AlignedByteStorage(const AlignedByteStorage &) = delete;
  AlignedByteStorage &operator=(const AlignedByteStorage &) = delete;
  AlignedByteStorage(AlignedByteStorage &&other) noexcept;
  AlignedByteStorage &operator=(AlignedByteStorage &&other) noexcept;
  ~AlignedByteStorage();

  std::uint8_t *data() { return data_; }
  const std::uint8_t *data() const { return data_; }
  std::size_t size() const { return size_; }

private:
  AlignedByteStorage() = default;
  void reset();

  std::uint8_t *data_ = nullptr;
  std::size_t size_ = 0;
  std::size_t alignment_ = alignof(std::max_align_t);
};

struct MemoryTargetPlan final {
  MemoryObservationForm form = MemoryObservationForm::FullState;
  std::optional<std::uint64_t> objectOrdinal;
  std::string globalSymbol;
  std::uint64_t byteCount = 0;
};

struct NativeExecutionContext final {
  std::vector<AlignedByteStorage> objects;
  std::optional<detail::LaneShape> returnShape;
  std::uint64_t returnByteCount = 0;
  bool littleEndian = true;
  std::optional<CanonicalValueSequence> returnValue;
  std::vector<std::vector<std::uint8_t>> globalBefore;
  std::vector<std::vector<std::uint8_t>> globalAfter;
  std::vector<bool> sawGlobalBefore;
  std::vector<bool> sawGlobalAfter;
  std::vector<frontend::StructuredEntityRef> profileBlocks;
  std::vector<std::uint64_t> blockActivationCounts;
  std::optional<std::string> error;
};

struct WorkloadCaptureCallbackNames final {
  std::string begin;
  std::string end;
  std::optional<std::string> registerObject;
  std::optional<std::string> coordinate;
  std::optional<std::string> memoryRoot;
  std::optional<std::string> value;
  std::optional<std::string> result;
  std::optional<std::string> memoryWrite;
  std::optional<std::string> pointerRead;
  std::optional<std::string> pointerWrite;
};

std::string uniqueMlirSymbolName(mlir::ModuleOp module, llvm::StringRef prefix);

llvm::Expected<std::string>
instrumentBlockActivations(mlir::ModuleOp module,
                           const ArtifactIdentity &identity,
                           NativeExecutionContext &capture);

llvm::Expected<NativeStructuredProgramObservations>
buildObservations(const StructuredProgramSimulationWorkload &workload,
                  const StructuredProgramSimulationRuntimeInput &input,
                  llvm::ArrayRef<MemoryTargetPlan> plans,
                  const NativeExecutionContext &capture);

llvm::Expected<std::optional<std::string>>
projectSelectedWholeProgram(mlir::ModuleOp module);

llvm::Expected<WorkloadCaptureCallbackNames> instrumentWorkloadBackedCapture(
    mlir::ModuleOp module, mlir::Operation *selectedOperation,
    const WorkloadBackedSimulationInputCapturePlan &plan);

llvm::Expected<NativeStructuredProgramObservations>
visitProjectedWorkloadBackedSimulationInputCaptures(
    mlir::OwningOpRef<mlir::ModuleOp> selectedModule,
    mlir::Operation *selectedOperation,
    const WorkloadBackedSimulationInputCapturePlan &plan,
    const frontend::StructuredProgramCandidate &sourceProgram,
    const CanonicalSimulationWorkload &workload,
    const CanonicalSimulationRuntimeInput &runtimeInput,
    std::uint64_t maxRetainedCaptureBytes,
    WorkloadBackedSimulationInputVisitor visitor);

} // namespace loom::sim::native_detail

#endif // LOOM_SIMULATOR_STRUCTUREDPROGRAMNATIVEEXECUTIONINTERNAL_H
