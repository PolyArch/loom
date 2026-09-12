#ifndef LOOM_SIMULATOR_STRUCTUREDPROGRAMNATIVEEXECUTIONINTERNAL_H
#define LOOM_SIMULATOR_STRUCTUREDPROGRAMNATIVEEXECUTIONINTERNAL_H

#include "Runtime/OrderedChannelABI.h"
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
  struct LogicalChannel final {
    std::optional<loom::runtime::OrderedChannelABI> abi;
    /// The proven flat producer count of each complete invocation of this
    /// exact channel-create lineage; it is also the bounded message capacity
    /// the ABI instance was created with.
    std::uint64_t producerMessageCount = 0;
    std::vector<std::optional<std::uint64_t>> consumerMessageCounts;
  };

  std::vector<AlignedByteStorage> objects;
  std::vector<LogicalChannel> logicalChannels;
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
  /// The same counts restricted to the source-declared computation interval.
  std::vector<std::uint64_t> measuredBlockActivationCounts;
  /// Nesting depth of the declared interval. The markers are ordinary calls,
  /// so a source may enter the interval once per measured sample.
  std::uint64_t computationBoundaryDepth = 0;
  /// Set when the interval was entered at least once. A source that declares
  /// no interval, or a workload that never reaches it, keeps the complete
  /// execution as its measured projection.
  bool computationIntervalObserved = false;
  std::optional<llvm::Error> error;
};

struct NativeChannelCallbackNames final {
  std::string create;
  std::string rate;
  std::string send;
  std::string receive;
  std::uint64_t lineageCount = 0;
};

struct SelectedWholeProgramProjection final {
  std::optional<std::string> invalidThreadExtent;
  std::optional<NativeChannelCallbackNames> channels;
};

enum class ProgramObjectCaptureKind : std::uint64_t {
  Global,
  RuntimeAllocation,
  StackAllocation
};

struct WorkloadCaptureCallbackNames final {
  std::vector<NativeMemoryObjectSource> programObjectSources;
  std::string begin;
  std::string end;
  std::optional<std::string> registerObject;
  std::optional<std::string> enterStackFrame;
  std::optional<std::string> leaveStackFrame;
  std::optional<std::string> coordinate;
  std::optional<std::string> memoryRoot;
  std::optional<std::string> value;
  std::optional<std::string> streamInput;
  std::optional<std::string> result;
  std::optional<std::string> streamOutput;
  std::optional<std::string> memoryWrite;
  std::optional<std::string> pointerRead;
  std::optional<std::string> pointerWrite;
};

std::string uniqueMlirSymbolName(mlir::ModuleOp module, llvm::StringRef prefix);

/// Callback symbols injected by the block-activation instrumentation. The
/// boundary symbol is absent when the module defines no computation marker.
struct BlockActivationCallbackNames final {
  std::string blockActivation;
  std::optional<std::string> computationBoundary;
};

llvm::Expected<BlockActivationCallbackNames>
instrumentBlockActivations(mlir::ModuleOp module,
                           const ArtifactIdentity &identity,
                           NativeExecutionContext &capture);

llvm::Expected<NativeStructuredProgramObservations>
buildObservations(const StructuredProgramSimulationWorkload &workload,
                  const StructuredProgramSimulationRuntimeInput &input,
                  llvm::ArrayRef<MemoryTargetPlan> plans,
                  const NativeExecutionContext &capture);

llvm::Error failLogicalChannelExecution(NativeExecutionContext &capture,
                                        llvm::Error failure);

llvm::Error finishLogicalChannelExecution(NativeExecutionContext &capture);

llvm::Expected<SelectedWholeProgramProjection>
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
