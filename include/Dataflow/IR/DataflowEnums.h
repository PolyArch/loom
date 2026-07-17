#ifndef DATAFLOW_IR_DATAFLOWENUMS_H
#define DATAFLOW_IR_DATAFLOWENUMS_H

#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/OpImplementation.h"

#include <cstdint>
#include <optional>

#include "Dataflow/IR/DataflowEnums.h.inc"

namespace dataflow {

inline std::optional<StreamStepKind>
getStreamStepKindFromAttr(::mlir::Attribute attr) {
  if (!attr)
    return std::nullopt;
  if (auto stepKind = ::llvm::dyn_cast<StreamStepKindAttr>(attr))
    return stepKind.getValue();
  auto integer = ::llvm::dyn_cast<::mlir::IntegerAttr>(attr);
  if (!integer)
    return std::nullopt;
  auto type = ::llvm::dyn_cast<::mlir::IntegerType>(integer.getType());
  if (!type || !type.isSignless() || type.getWidth() != 32)
    return std::nullopt;
  return symbolizeStreamStepKind(
      static_cast<std::uint32_t>(integer.getValue().getZExtValue()));
}

} // namespace dataflow

#endif // DATAFLOW_IR_DATAFLOWENUMS_H
