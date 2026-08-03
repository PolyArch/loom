#ifndef FABRIC_IR_USEPATTERNVALUE_H
#define FABRIC_IR_USEPATTERNVALUE_H

#include "Fabric/IR/ResourceContract.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <variant>
#include <vector>

namespace fabric {

struct PhysicalTagPatternValue final {
  llvm::APInt value;

  friend bool operator==(const PhysicalTagPatternValue &lhs,
                         const PhysicalTagPatternValue &rhs) {
    return lhs.value == rhs.value;
  }
};

using UsePatternValue = std::variant<PhysicalTagPatternValue>;

/// Strictly adopts bytes through the exact schema-owned production codec.
llvm::Expected<UsePatternValue>
decodeUsePatternValue(const UsePatternValueSchema &schema,
                      llvm::ArrayRef<std::uint8_t> bytes);

/// Re-encodes one adopted immutable value through the same production codec.
/// A value of the wrong closed kind or width is rejected.
llvm::Expected<std::vector<std::uint8_t>>
encodeUsePatternValue(const UsePatternValueSchema &schema,
                      const UsePatternValue &value);

} // namespace fabric

#endif // FABRIC_IR_USEPATTERNVALUE_H
