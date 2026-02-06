//===-- DataflowTypes.cpp - Dataflow dialect type verifiers -----*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include "loom/Dialect/Dataflow/DataflowTypes.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace loom::dataflow;

LogicalResult
TaggedType::verify(function_ref<InFlightDiagnostic()> emitError,
                   Type valueType, IntegerType tagType) {
  // valueType must be one of the allowed native types.
  bool validValue = valueType.isInteger(1) || valueType.isInteger(8) ||
                    valueType.isInteger(16) || valueType.isInteger(32) ||
                    valueType.isInteger(64) || valueType.isBF16() ||
                    valueType.isF16() || valueType.isF32() ||
                    valueType.isF64() || valueType.isIndex() ||
                    isa<NoneType>(valueType);
  if (!validValue)
    return emitError()
           << "tagged value type must be one of i1, i8, i16, i32, i64, "
              "bf16, f16, f32, f64, index, none; got "
           << valueType;

  // tagType must be signless integer with width 1-16.
  if (!tagType.isSignless())
    return emitError() << "tag type must be signless integer; got " << tagType;

  unsigned width = tagType.getWidth();
  if (width < 1 || width > 16)
    return emitError() << "tag type width must be 1-16; got " << width;

  return success();
}
