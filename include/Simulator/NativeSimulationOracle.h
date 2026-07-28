#ifndef LOOM_SIMULATOR_NATIVESIMULATIONORACLE_H
#define LOOM_SIMULATOR_NATIVESIMULATIONORACLE_H

#include "Simulator/SimulationInputCapture.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <vector>

namespace loom::sim {

struct NativeCapturedMemoryObject {
  std::vector<std::uint8_t> initialBytes;
  std::vector<std::uint8_t> finalBytes;
};

struct NativeSimulationCallCapture {
  std::vector<NativeCapturedMemoryObject> objects;
};

struct NativeSimulationMemoryCapture {
  std::int32_t entryResult = 0;
  std::vector<NativeSimulationCallCapture> calls;
};

/// Execute one native LLVM module and capture the finite memory objects around
/// every dynamic execution of the exact statically selected host call. This is
/// an ephemeral independent oracle; its bytes may initialize a typed
/// SimulationRuntimeInput, but this record is not a persistent wire format.
llvm::Expected<NativeSimulationMemoryCapture>
executeNativeSimulationMemoryCapture(llvm::orc::ThreadSafeModule module,
                                     const SimulationMemoryCapturePlan &plan,
                                     llvm::StringRef entrySymbol = "main");

} // namespace loom::sim

#endif // LOOM_SIMULATOR_NATIVESIMULATIONORACLE_H
