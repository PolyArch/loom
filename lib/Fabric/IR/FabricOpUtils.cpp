#include "Fabric/IR/FabricOps.h"

#include "Common/IndexWidth.h"
#include "Common/VectorWidth.h"
#include "Fabric/IR/FabricTypes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <iterator>
#include <string>

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

FabricOpModeClassification classifyFabricOpModes(OpOp op) {
  ArrayAttr modes = op.getHwParamsAttr();
  DictionaryAttr software = op.getSwConfigsAttr();
  bool selectsNormalizedMode = software && software.get("mode");
  if (!modes) {
    if (selectsNormalizedMode)
      return {FabricOpModeKind::Malformed,
              "'sw_configs.mode' requires normalized hw_params"};
    return {};
  }
  if (modes.empty())
    return {FabricOpModeKind::Malformed, "'hw_params' must not be empty"};

  std::optional<FabricOpModeKind> format;
  for (auto [index, attr] : llvm::enumerate(modes)) {
    auto entry = dyn_cast<DictionaryAttr>(attr);
    if (!entry)
      return {FabricOpModeKind::Malformed,
              "'hw_params' entry #" + std::to_string(index) +
                  " must be a dictionary attribute"};

    constexpr llvm::StringLiteral normalizedKeys[] = {
        "op", "function_type", "input_ports", "output_ports", "attributes"};
    unsigned present = 0;
    for (StringRef key : normalizedKeys)
      present += static_cast<unsigned>(static_cast<bool>(entry.get(key)));
    if (present != 0 && present != std::size(normalizedKeys))
      return {FabricOpModeKind::Malformed,
              "'hw_params' entry #" + std::to_string(index) +
                  " partially specifies a normalized mode"};

    FabricOpModeKind entryFormat = present == std::size(normalizedKeys)
                                       ? FabricOpModeKind::Normalized
                                       : FabricOpModeKind::Legacy;
    if (!format) {
      format = entryFormat;
      continue;
    }
    if (*format != entryFormat)
      return {FabricOpModeKind::Malformed,
              "'hw_params' must not mix normalized modes and legacy fields"};
  }

  if (*format == FabricOpModeKind::Legacy && modes.size() != 1)
    return {FabricOpModeKind::Malformed,
            "legacy 'hw_params' must be a length-1 array, got " +
                std::to_string(modes.size())};
  if (*format == FabricOpModeKind::Legacy && selectsNormalizedMode)
    return {FabricOpModeKind::Malformed,
            "'sw_configs.mode' requires normalized hw_params"};
  return {*format, {}};
}

LogicalResult
preflightPairedLaneModes(OpOp op,
                         const FabricOpModeClassification &classification,
                         std::string &error) {
  if (!op.getPairedLanesAttr())
    return success();
  if (classification.kind == FabricOpModeKind::Malformed) {
    error = classification.diagnostic;
    return failure();
  }
  if (classification.kind != FabricOpModeKind::Normalized) {
    error = "paired_lanes requires normalized hw_params modes";
    return failure();
  }
  for (Attribute attr : op.getHwParamsAttr()) {
    auto mode = dyn_cast<DictionaryAttr>(attr);
    auto selected = mode ? mode.getAs<FlatSymbolRefAttr>("op") : nullptr;
    if (!selected || selected.getValue() != "dataflow.sync") {
      error =
          "paired_lanes requires every hw_params mode to select @dataflow.sync";
      return failure();
    }
  }
  return success();
}

FailureOr<unsigned> getSemanticPayloadWidth(Type type, std::string &error) {
  if (auto integer = dyn_cast<IntegerType>(type))
    return integer.getWidth();
  if (auto floating = dyn_cast<FloatType>(type))
    return floating.getWidth();
  if (isa<IndexType, LLVM::LLVMPointerType>(type))
    return ::loom::getIndexWidth();
  if (isa<NoneType>(type))
    return 0u;
  if (auto vector = dyn_cast<VectorType>(type)) {
    auto elementWidth = getSemanticPayloadWidth(vector.getElementType(), error);
    if (failed(elementWidth))
      return failure();
    auto width = ::loom::getFixedVectorBitWidth(vector, *elementWidth);
    if (!width) {
      error = llvm::toString(width.takeError());
      return failure();
    }
    return *width;
  }

  std::string text;
  llvm::raw_string_ostream os(text);
  type.print(os);
  error = "unsupported semantic payload type " + text;
  return failure();
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
