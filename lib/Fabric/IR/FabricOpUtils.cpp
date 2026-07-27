#include "Fabric/IR/FabricOps.h"

#include "Fabric/IR/FabricTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <limits>
#include <string>
#include <system_error>
#include <vector>

using namespace mlir;

namespace fabric {

bool isFabricModulePortType(Type type) {
  return isa<BitsType, BitsTagType, MemRefType>(type);
}

bool haveSameFabricModulePortKind(Type source, Type destination) {
  if (isa<BitsType>(source))
    return isa<BitsType>(destination);
  if (isa<BitsTagType>(source))
    return isa<BitsTagType>(destination);
  if (isa<MemRefType>(source))
    return isa<MemRefType>(destination);
  return false;
}

std::optional<unsigned> getFabricBitsWidth(Type type) {
  if (auto bits = dyn_cast<BitsType>(type))
    return bits.getWidth();
  return std::nullopt;
}

std::optional<unsigned> getFabricTransportPayloadWidth(Type type) {
  if (auto bits = dyn_cast<BitsType>(type))
    return bits.getWidth();
  if (auto tagged = dyn_cast<BitsTagType>(type))
    return tagged.getWidth();
  return std::nullopt;
}

namespace {

void appendU32(std::vector<std::uint8_t> &bytes, std::uint32_t value) {
  bytes.push_back(static_cast<std::uint8_t>(value >> 24));
  bytes.push_back(static_cast<std::uint8_t>(value >> 16));
  bytes.push_back(static_cast<std::uint8_t>(value >> 8));
  bytes.push_back(static_cast<std::uint8_t>(value));
}

void appendU64(std::vector<std::uint8_t> &bytes, std::uint64_t value) {
  for (int shift = 56; shift >= 0; shift -= 8)
    bytes.push_back(static_cast<std::uint8_t>(value >> shift));
}

llvm::Error invalidTransportType(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "fabric_transport_type_invalid: " + message);
}

} // namespace

llvm::Expected<std::vector<std::uint8_t>> encodeFabricTransportType(Type type) {
  std::vector<std::uint8_t> bytes;
  if (auto bits = dyn_cast<BitsType>(type)) {
    appendU32(bytes, 0);
    appendU32(bytes, bits.getWidth());
    return bytes;
  }
  if (auto tagged = dyn_cast<BitsTagType>(type)) {
    appendU32(bytes, 1);
    appendU32(bytes, tagged.getWidth());
    appendU32(bytes, tagged.getTagWidth());
    return bytes;
  }
  std::string spelling;
  llvm::raw_string_ostream stream(spelling);
  type.print(stream);
  return invalidTransportType("unsupported type " + spelling);
}

llvm::Expected<std::vector<std::uint8_t>>
encodeFabricTransportFunctionType(FunctionType type) {
  std::vector<std::uint8_t> bytes;
  appendU64(bytes, type.getNumInputs());
  for (Type input : type.getInputs()) {
    auto encoded = encodeFabricTransportType(input);
    if (!encoded)
      return encoded.takeError();
    appendU64(bytes, encoded->size());
    bytes.insert(bytes.end(), encoded->begin(), encoded->end());
  }
  appendU64(bytes, type.getNumResults());
  for (Type result : type.getResults()) {
    auto encoded = encodeFabricTransportType(result);
    if (!encoded)
      return encoded.takeError();
    appendU64(bytes, encoded->size());
    bytes.insert(bytes.end(), encoded->begin(), encoded->end());
  }
  return bytes;
}

LogicalResult verifyInnerInputTypesProperty(Operation *op, ValueRange inputs,
                                            ArrayRef<Type> innerInputTypes) {
  if (op->getDiscardableAttr(kInnerInputTypesPropertyName))
    return op->emitOpError("discardable attribute '")
           << kInnerInputTypesPropertyName
           << "' conflicts with the inherent property of the same name";

  if (innerInputTypes.empty())
    return success();
  if (innerInputTypes.size() != inputs.size())
    return op->emitOpError("'")
           << kInnerInputTypesPropertyName
           << "' property size does not match operand count";

  for (auto [input, innerType] : llvm::zip(inputs, innerInputTypes))
    if (input.getType() != innerType)
      return success();

  return op->emitOpError("'")
         << kInnerInputTypesPropertyName
         << "' must be empty when every destination input type equals its "
            "operand type";
}

} // namespace fabric
