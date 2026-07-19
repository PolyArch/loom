#ifndef LOOM_LIB_FABRIC_IR_CONFIGUREDFUNCTIONINTERNAL_H
#define LOOM_LIB_FABRIC_IR_CONFIGUREDFUNCTIONINTERNAL_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Types.h"

namespace fabric::detail {

bool sameType(::mlir::Type lhs, ::mlir::Type rhs);
bool sameAttributes(::mlir::DictionaryAttr lhs, ::mlir::DictionaryAttr rhs);

} // namespace fabric::detail

#endif // LOOM_LIB_FABRIC_IR_CONFIGUREDFUNCTIONINTERNAL_H
