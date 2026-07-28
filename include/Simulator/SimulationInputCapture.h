#ifndef LOOM_SIMULATOR_SIMULATIONINPUTCAPTURE_H
#define LOOM_SIMULATOR_SIMULATIONINPUTCAPTURE_H

#include "Dataflow/IR/DataflowCanonicalArtifact.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Value.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <string>
#include <vector>

namespace loom::sim {

/// One finite host allocation from which a SimulationRuntimeInput memory
/// object can be captured. The MLIR value is an ephemeral instrumentation
/// handle; only captured bytes and root bindings enter the persistent input.
struct SimulationMemoryCaptureObject {
  mlir::Value base;
  std::uint64_t byteCount = 0;
  std::uint64_t callOperandOrdinal = 0;
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

/// The memory plane of one concrete host call that reaches one rooted graph
/// launch. This is a derived instrumentation plan, not a persistent schema.
struct SimulationMemoryCapturePlan {
  dataflow::RootedGraphLaunchRef launch;
  mlir::LLVM::CallOp hostCall;
  std::string hostCallerSymbol;
  std::string hostCalleeSymbol;
  std::uint64_t hostCallOrdinal = 0;
  std::vector<SimulationMemoryCaptureObject> objects;
  std::vector<SimulationMemoryRootCapture> memoryRootBindings;
};

/// Derive the finite host-memory capture relation for one exact call site.
/// Every imported graph root must trace through the root thread launch and its
/// enclosing LLVM callable to a call operand with a statically proven finite
/// allocation. Unknown extents fail closed with not_supported.
llvm::Expected<SimulationMemoryCapturePlan> deriveSimulationMemoryCapturePlan(
    const dataflow::CanonicalDataflowProgramView &program,
    dataflow::RootedGraphLaunchRef launch, mlir::LLVM::CallOp hostCall);

} // namespace loom::sim

#endif // LOOM_SIMULATOR_SIMULATIONINPUTCAPTURE_H
