#ifndef LOOM_COMMON_POINTERLAYOUT_H
#define LOOM_COMMON_POINTERLAYOUT_H

#include "mlir/IR/Operation.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/Support/Error.h"

#include <cstdint>

namespace loom {

enum class PointerLayoutKind : std::uint8_t {
  StableIntegral,
  NonIntegral,
  Unstable,
  ExternalState,
};

struct PointerLayout {
  std::uint32_t addressSpace = 0;
  unsigned representationBits = 0;
  unsigned addressBits = 0;
  PointerLayoutKind kind = PointerLayoutKind::StableIntegral;

  friend bool operator==(const PointerLayout &lhs, const PointerLayout &rhs) {
    return lhs.addressSpace == rhs.addressSpace &&
           lhs.representationBits == rhs.representationBits &&
           lhs.addressBits == rhs.addressBits && lhs.kind == rhs.kind;
  }
  friend bool operator!=(const PointerLayout &lhs, const PointerLayout &rhs) {
    return !(lhs == rhs);
  }
};

// Parses the exact nonempty llvm.data_layout owned by the closest enclosing
// module. Target-derived defaults and host layouts are never substitutes.
llvm::Expected<llvm::DataLayout> resolveLLVMDataLayout(mlir::Operation *op);

// Resolves one LLVM pointer layout from the exact nonempty llvm.data_layout
// owned by the closest enclosing module. No target-derived fallback is legal.
llvm::Expected<PointerLayout> resolvePointerLayout(mlir::Operation *op,
                                                   std::uint32_t addressSpace);

} // namespace loom

#endif // LOOM_COMMON_POINTERLAYOUT_H
