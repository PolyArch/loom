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
#include <variant>
#include <vector>

namespace loom::sim {

/// One finite host allocation from which a SimulationRuntimeInput memory
/// object can be captured. The MLIR value is an ephemeral instrumentation
/// handle; only captured bytes and root bindings enter the persistent input.
struct SimulationMemoryCaptureObject {
  mlir::Value base;
  std::uint64_t byteCount = 0;
  std::uint64_t operandByteOffset = 0;
};

struct DirectCallOperandMemorySource final {
  std::uint64_t operandOrdinal = 0;
};

struct DirectCallGlobalMemorySource final {
  std::string symbol;
};

/// The exact native source of one direct-call capture object. This relation is
/// ephemeral instrumentation state: the persistent runtime input contains only
/// canonical bytes and Dataflow-owned root bindings.
using DirectCallMemorySource =
    std::variant<DirectCallOperandMemorySource, DirectCallGlobalMemorySource>;

/// The exact projection from a Dataflow-owned logical root into one ephemeral
/// capture object. Object indices are draft-local and are canonicalized by the
/// SimulationRuntimeInput finalizer.
struct SimulationMemoryRootCapture {
  dataflow::LogicalMemoryRootRef root;
  std::uint64_t objectIndex = 0;
  std::uint64_t byteOffset = 0;
  // True when the canonical memory actor relation can observe bytes supplied
  // before this graph activation. Independent native replays must agree on
  // these bytes; output-only roots may begin with different concrete storage.
  bool requiresInitialState = true;
  // Derived from all canonical write actors reachable from this root. A null
  // type means the root is read-only or does not have one uniform floating
  // lane semantics, so source-backed validation compares its bytes exactly.
  mlir::FloatType floatingWriteLaneType;
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

/// One graph value result and its exact source value at the selected
/// Structured boundary. The source value is an ephemeral instrumentation
/// handle; only the resulting semantic value enters an execution observation.
struct SimulationValueResultCapture {
  std::uint64_t valueResultOrdinal = 0;
  mlir::Value boundaryValue;
  std::uint64_t lanesPerToken = 0;
  std::uint32_t laneBitWidth = 0;
  std::uint64_t byteCount = 0;
};

/// The finite value-input, value-result, and memory planes of one concrete
/// execution boundary that reaches one rooted graph launch. This is a derived
/// instrumentation plan, not a persistent schema.
struct SimulationInputCapturePlan {
  dataflow::RootedGraphLaunchRef launch;
  std::vector<SimulationValueInputCapture> valueInputs;
  std::vector<SimulationValueResultCapture> valueResults;
  std::vector<SimulationMemoryCaptureObject> objects;
  std::vector<SimulationMemoryRootCapture> memoryRootBindings;
};

struct DirectCallSimulationInputCapturePlan final {
  SimulationInputCapturePlan input;
  std::vector<DirectCallMemorySource> memoryObjectSources;
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
    dataflow::RootedGraphLaunchRef launch, mlir::ValueRange boundaryInputs,
    mlir::ValueRange boundaryResults);

} // namespace loom::sim

#endif // LOOM_SIMULATOR_SIMULATIONINPUTCAPTURE_H
