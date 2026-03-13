//===-- CpuReferenceExecutor.h - CPU-side DFG reference exec -------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Evaluates a handshake.func on the CPU to produce reference output values.
// Used by the CPU oracle to verify simulator correctness: the same inputs
// are fed to both the reference executor and the hardware simulator, and
// outputs are compared.
//
// The executor interprets operations in SSA order. Unsupported ops (e.g.,
// memory, complex control flow) cause it to report "unsupported" so the
// caller can fall back to a determinism check.
//
//===----------------------------------------------------------------------===//

#ifndef LOOM_SIMULATOR_CPUREFERENCEEXECUTOR_H
#define LOOM_SIMULATOR_CPUREFERENCEEXECUTOR_H

#include "mlir/IR/BuiltinOps.h"

#include <cstdint>
#include <string>
#include <vector>

namespace loom {
namespace sim {

/// Result of CPU reference execution.
struct CpuRefResult {
  bool supported = false;
  std::string unsupportedReason;
  /// Per-port output values (one vector per output port).
  std::vector<std::vector<uint64_t>> outputs;
};

/// Evaluate a handshake.func on the CPU to produce reference output values.
///
/// \p dfgModule  The handshake MLIR module containing the handshake.func.
/// \p inputs     Per-port input values (one vector per input port).
///               The last "ctrl" port (none type) is implicit and should NOT
///               be included.
/// \returns      CpuRefResult with outputs or unsupported reason.
CpuRefResult cpuReferenceExecute(mlir::ModuleOp dfgModule,
                                 const std::vector<std::vector<uint64_t>> &inputs);

} // namespace sim
} // namespace loom

#endif // LOOM_SIMULATOR_CPUREFERENCEEXECUTOR_H
