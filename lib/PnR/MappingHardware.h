#ifndef LOOM_PNR_MAPPING_HARDWARE_H
#define LOOM_PNR_MAPPING_HARDWARE_H

#include "mlir/IR/Operation.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <string>

namespace loom::pnr::detail {

enum class MemAccessKind {
  Load,
  Store,
};

struct MemOccurrenceIdentity {
  std::uint64_t loadResourceBase = 0;
  std::uint64_t loadCount = 0;
  std::uint64_t storeResourceBase = 0;
  std::uint64_t storeCount = 0;
};

struct ConcreteMemOccurrence {
  mlir::Operation *op = nullptr;
  MemOccurrenceIdentity identity;
};

bool isConcreteHardwareOperation(mlir::Operation *op,
                                 mlir::Operation *hardwareRoot);

llvm::SmallVector<ConcreteMemOccurrence, 2>
collectConcreteMemOccurrences(mlir::Operation *hardwareRoot);

std::string memResourceId(llvm::StringRef hardwareName, MemAccessKind kind,
                          const MemOccurrenceIdentity &identity,
                          std::uint64_t portIndex);

} // namespace loom::pnr::detail

#endif // LOOM_PNR_MAPPING_HARDWARE_H
