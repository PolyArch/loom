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
  std::vector<std::uint64_t> memoryRootByteOffsets;
};

struct NativeSimulationInputCapture {
  std::int32_t entryResult = 0;
  std::vector<NativeSimulationCallCapture> calls;
};

/// Transient functional observations from executing one exact Structured
/// Program workload. The workload remains the sole owner of selected targets
/// and order; this provider result is not a persistent wire or workload key.
struct NativeStructuredProgramObservations {
  std::optional<CanonicalValueSequence> returnValue;
  std::vector<MemoryObservationPayload> memories;
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
