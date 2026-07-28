#ifndef LOOM_SIMULATOR_SIMULATIONINPUTCAPTURE_H
#define LOOM_SIMULATOR_SIMULATIONINPUTCAPTURE_H

#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Simulator/SimulationArtifacts.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Value.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace loom::sim {

/// One finite host allocation from which a SimulationRuntimeInput memory
/// object can be captured. The MLIR value is an ephemeral instrumentation
/// handle; only captured bytes and root bindings enter the persistent input.
struct SimulationMemoryCaptureObject {
  mlir::Value base;
  std::uint64_t byteCount = 0;
  std::uint64_t boundaryOperandOrdinal = 0;
  std::uint64_t operandByteOffset = 0;
};

/// The exact projection from a Dataflow-owned logical root into one ephemeral
/// capture object. Object indices are draft-local and are canonicalized by the
/// SimulationRuntimeInput finalizer.
struct SimulationMemoryRootCapture {
  dataflow::LogicalMemoryRootRef root;
  std::uint64_t objectIndex = 0;
  std::uint64_t byteOffset = 0;
};

/// One exact graph value-input source at an execution boundary. Fixed values
/// preserve Defined, Poison, or Undef semantics in the workload. Runtime
/// values retain the source SSA value only as an ephemeral instrumentation
/// handle and are captured into SimulationRuntimeInput.
struct SimulationValueInputCapture {
  std::uint64_t valueInputOrdinal = 0;
  std::optional<std::uint64_t> boundaryOperandOrdinal;
  mlir::Value boundaryValue;
  std::uint64_t lanesPerToken = 0;
  std::uint32_t laneBitWidth = 0;
  std::uint64_t byteCount = 0;
  std::optional<CanonicalValueSequence> fixedValue;
};

/// The finite value and memory input planes of one concrete execution boundary
/// that reaches one rooted graph launch. This is a derived instrumentation
/// plan, not a persistent schema.
struct SimulationInputCapturePlan {
  dataflow::RootedGraphLaunchRef launch;
  std::vector<SimulationValueInputCapture> valueInputs;
  std::vector<SimulationMemoryCaptureObject> objects;
  std::vector<SimulationMemoryRootCapture> memoryRootBindings;
};

struct DirectCallSimulationInputCapturePlan final {
  SimulationInputCapturePlan input;
  mlir::LLVM::CallOp hostCall;
  std::string hostCallerSymbol;
  std::string hostCalleeSymbol;
  std::uint64_t hostCallOrdinal = 0;
};

/// Derive the finite host-memory capture relation for one exact call site.
/// Every imported graph root must trace through the root thread launch and its
/// enclosing LLVM callable to a call operand with a statically proven finite
/// allocation. Unknown extents fail closed with not_supported.
llvm::Expected<DirectCallSimulationInputCapturePlan>
deriveSimulationInputCapturePlan(
    const dataflow::CanonicalDataflowProgramView &program,
    dataflow::RootedGraphLaunchRef launch, mlir::LLVM::CallOp hostCall);

/// Derive the same finite memory relation for an operation-owned Spatial
/// boundary. `boundaryInputs` is the exact ordered live-in projection produced
/// by the Structured ownership owner before thread materialization.
llvm::Expected<SimulationInputCapturePlan>
deriveOperationSimulationInputCapturePlan(
    const dataflow::CanonicalDataflowProgramView &program,
    dataflow::RootedGraphLaunchRef launch, mlir::ValueRange boundaryInputs);

} // namespace loom::sim

#endif // LOOM_SIMULATOR_SIMULATIONINPUTCAPTURE_H
