#ifndef LOOM_SIMULATOR_NATIVESIMULATIONORACLE_H
#define LOOM_SIMULATOR_NATIVESIMULATIONORACLE_H

#include "Simulator/SimulationInputCapture.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/Support/Error.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"

#include <cstdint>
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
};

struct NativeSimulationInputCapture {
  std::int32_t entryResult = 0;
  std::vector<NativeSimulationCallCapture> calls;
};

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
