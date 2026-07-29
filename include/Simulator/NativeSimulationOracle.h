#ifndef LOOM_SIMULATOR_NATIVESIMULATIONORACLE_H
#define LOOM_SIMULATOR_NATIVESIMULATIONORACLE_H

#include "Simulator/SimulationInputCapture.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/Support/Error.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace loom::sim {

struct NativeCapturedMemoryObject {
  std::vector<std::uint8_t> initialBytes;
  std::vector<std::uint8_t> finalBytes;
};

struct NativeSimulationCallCapture {
  std::vector<RuntimeValueEntry> runtimeValues;
  std::vector<CanonicalValueSequence> valueResults;
  std::vector<NativeCapturedMemoryObject> objects;
  /// Per logical-memory-root capture binding, the invocation-local object in
  /// `objects`. Static developer-tool capture and workload-backed production
  /// capture both populate this projection; raw native addresses never leave
  /// the execution provider.
  std::vector<std::uint64_t> memoryRootObjectOrdinals;
  std::vector<std::uint64_t> memoryRootByteOffsets;
};

struct NativeSimulationInputCapture {
  std::int32_t entryResult = 0;
  std::vector<NativeSimulationCallCapture> calls;
};

struct NativeStructuredBlockActivation {
  frontend::StructuredEntityRef block;
  std::uint64_t activations = 0;
};

/// Transient functional observations from executing one exact Structured
/// Program workload. The workload remains the sole owner of selected targets
/// and order; this provider result is not a persistent wire or workload key.
struct NativeStructuredProgramObservations {
  std::optional<CanonicalValueSequence> returnValue;
  std::vector<MemoryObservationPayload> memories;
  /// Total canonical block-order projection over blocks owned by defined
  /// llvm.func operations. Counts, including zero, are invocation-local and
  /// all coarser dynamic coverage is derived from this one projection.
  std::vector<NativeStructuredBlockActivation> blockActivations;
};

/// One Dataflow logical root and the exact Structured pointer value presented
/// at a selected region boundary. The pointer is an ephemeral instrumentation
/// handle. Workload-backed execution resolves it through the one runtime
/// object registry and records only object ordinals, bytes, and offsets.
struct WorkloadBackedMemoryRootCapture final {
  dataflow::LogicalMemoryRootRef root;
  mlir::Value boundaryPointer;
};

/// The exact finite graph boundary instrumented during one selected
/// Structured execution. This is a removable native-execution plan, not a
/// persistent Simulation schema or a second graph ABI authority.
struct WorkloadBackedSimulationInputCapturePlan final {
  dataflow::RootedGraphLaunchRef launch;
  std::vector<SimulationValueInputCapture> valueInputs;
  std::vector<SimulationValueResultCapture> valueResults;
  std::vector<WorkloadBackedMemoryRootCapture> memoryRoots;
};

/// Execute the exact workload entry from an immutable Structured Program.
/// Runtime objects are finite byte-addressed storage, and shared object
/// ordinals preserve pointer aliasing. This native provider accepts only
/// concrete Defined inputs and first proves execution-layout compatibility
/// before retargeting an ephemeral module clone to the host JIT.
llvm::Expected<NativeStructuredProgramObservations>
executeNativeStructuredProgram(
    const frontend::StructuredProgramCandidate &program,
    const CanonicalSimulationWorkload &workload,
    const CanonicalSimulationRuntimeInput &runtimeInput);

/// Execute one finalized ownership-selected Structured Program against the
/// immutable workload owned by its exact source program. The native oracle
/// mechanically projects supported thread/spatial ownership carriers back to
/// their sequential whole-program semantics; it does not reselect ownership
/// or execute the Canonical Dataflow projection in place of the candidate.
llvm::Expected<NativeStructuredProgramObservations>
executeSelectedStructuredProgram(
    const frontend::StructuredProgramCandidate &selectedProgram,
    const frontend::StructuredProgramCandidate &sourceProgram,
    const CanonicalSimulationWorkload &workload,
    const CanonicalSimulationRuntimeInput &runtimeInput);

/// Execute one prepared selected Structured Program from the exact production
/// workload/runtime pair and capture every dynamic invocation of its selected
/// boundary. Memory pointers must resolve through the invocation's runtime
/// object registry; an unregistered allocation is typed Unsupported rather
/// than inferred from static reaching stores.
llvm::Expected<NativeSimulationInputCapture>
executeWorkloadBackedSimulationInputCapture(
    mlir::OwningOpRef<mlir::ModuleOp> preparedModule,
    mlir::Operation *selectedOperation,
    const WorkloadBackedSimulationInputCapturePlan &plan,
    const frontend::StructuredProgramCandidate &sourceProgram,
    const CanonicalSimulationWorkload &workload,
    const CanonicalSimulationRuntimeInput &runtimeInput);

/// Exact comparison of whole-program return and memory observations. Dynamic
/// source coverage is intentionally excluded because it profiles the source
/// workload rather than the selected candidate.
bool haveEquivalentFunctionalObservations(
    const NativeStructuredProgramObservations &reference,
    const NativeStructuredProgramObservations &candidate);

/// Execute one native LLVM module and capture the finite graph inputs around
/// every dynamic execution of the exact statically selected host call. This is
/// an ephemeral independent oracle; its values and bytes may initialize a
/// typed SimulationRuntimeInput, but this record is not a persistent wire
/// format.
llvm::Expected<NativeSimulationInputCapture>
executeNativeSimulationInputCapture(
    llvm::orc::ThreadSafeModule module,
    const DirectCallSimulationInputCapturePlan &plan,
    llvm::StringRef entrySymbol = "main");

/// Lower the selected callable from one prepared Structured Program, replace
/// only that callable's body in the exact host LLVM module, and capture the
/// direct call named by `plan`. Host initialization, residual work, ABI, and
/// target properties remain owned by `hostModule`; only the selected callable
/// receives the typed compiler decisions already materialized in `module`.
llvm::Expected<NativeSimulationInputCapture>
executeStructuredDirectCallSimulationInputCapture(
    llvm::orc::ThreadSafeModule hostModule,
    mlir::OwningOpRef<mlir::ModuleOp> module,
    const DirectCallSimulationInputCapturePlan &plan,
    llvm::StringRef entrySymbol = "main");

/// Execute one exact prepared Structured Program clone and capture its finite
/// graph inputs immediately before, and value results plus memory state
/// immediately after, every dynamic execution of the selected operation. The
/// clone must already contain the DSE-selected semantic decisions, but the
/// operation itself must not yet have been replaced by a Spatial ownership
/// carrier.
llvm::Expected<NativeSimulationInputCapture>
executeStructuredSimulationInputCapture(
    mlir::OwningOpRef<mlir::ModuleOp> module,
    mlir::Operation *selectedOperation,
    const OperationSimulationInputCapturePlan &plan,
    llvm::StringRef entrySymbol = "main");

} // namespace loom::sim

#endif // LOOM_SIMULATOR_NATIVESIMULATIONORACLE_H
