#include "Fabric/IR/FabricOps.h"

#include "Fabric/IR/FabricTypes.h"
#include "llvm/ADT/STLExtras.h"

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
