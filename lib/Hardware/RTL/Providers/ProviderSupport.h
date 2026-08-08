#ifndef LOOM_LIB_HARDWARE_RTL_PROVIDERS_PROVIDERSUPPORT_H
#define LOOM_LIB_HARDWARE_RTL_PROVIDERS_PROVIDERSUPPORT_H

#include "Hardware/Configuration/ConfigurationABI.h"

#include "mlir/IR/Value.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include <cstdint>
#include <optional>
#include <string>

namespace mlir {
class Location;
class OpBuilder;
} // namespace mlir

namespace loom::hardware::rtl::detail {

struct PortableFloatFormat final {
  unsigned exponentBits;
  unsigned fractionBits;

  unsigned width() const { return 1 + exponentBits + fractionBits; }
  unsigned precision() const { return fractionBits + 1; }
  int bias() const { return (1 << (exponentBits - 1)) - 1; }
  int minimumExponent() const { return 1 - bias(); }
  int maximumExponent() const { return bias(); }

  friend bool operator==(const PortableFloatFormat &lhs,
                         const PortableFloatFormat &rhs) {
    return lhs.exponentBits == rhs.exponentBits &&
           lhs.fractionBits == rhs.fractionBits;
  }
};

std::optional<PortableFloatFormat> resolvePortableFloatFormat(mlir::Type type);

std::string buildPortableFloatFmaFunction(const PortableFloatFormat &format,
                                          const std::string &functionName);

llvm::APInt decodePhysicalCode(llvm::ArrayRef<std::uint8_t> bytes,
                               std::uint64_t bitCount);

const FiniteCodebookEntry *
findFiniteCodebookEntry(const FiniteCodebookEncoding &codebook,
                        llvm::ArrayRef<std::uint8_t> semanticValue);

mlir::Value resizeUnsigned(mlir::OpBuilder &builder, mlir::Location location,
                           mlir::Value value, unsigned width);

mlir::Value addOrSubtract(mlir::OpBuilder &builder, mlir::Location location,
                          mlir::Value lhs, mlir::Value rhs,
                          mlir::Value subtract);

} // namespace loom::hardware::rtl::detail

#endif // LOOM_LIB_HARDWARE_RTL_PROVIDERS_PROVIDERSUPPORT_H
