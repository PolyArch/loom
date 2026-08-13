#ifndef LOOM_ADG_CATALOGCAPABILITIES_H
#define LOOM_ADG_CATALOGCAPABILITIES_H

#include "Fabric/IR/ImplementationFamily.h"

#include "mlir/IR/BuiltinTypes.h"

#include <vector>

namespace loom::adg::detail {

inline ::fabric::PointerFormatRelation catalogPointerFormats() {
  return ::fabric::PointerFormatRelation::get(
      {{0, 32, 32, ::loom::PointerLayoutKind::StableIntegral},
       {0, 64, 64, ::loom::PointerLayoutKind::StableIntegral}});
}

inline ::fabric::IntegerWidthSet catalogOrdinaryIntegerWidths() {
  return ::fabric::IntegerWidthSet::get(
      {::fabric::IntegerWidth::I8, ::fabric::IntegerWidth::I16,
       ::fabric::IntegerWidth::I32, ::fabric::IntegerWidth::I64});
}

inline ::fabric::IntegerWidthSet catalogLogicIntegerWidths() {
  return ::fabric::IntegerWidthSet::get(
      {::fabric::IntegerWidth::I1, ::fabric::IntegerWidth::I8,
       ::fabric::IntegerWidth::I16, ::fabric::IntegerWidth::I32,
       ::fabric::IntegerWidth::I64});
}

inline ::fabric::FloatFormatSet catalogFloatFormats() {
  return ::fabric::FloatFormatSet::get(
      {::fabric::FloatFormat::F16, ::fabric::FloatFormat::BF16,
       ::fabric::FloatFormat::F32, ::fabric::FloatFormat::F64});
}

inline std::vector<mlir::Type>
catalogFixedVectorElementTypes(mlir::MLIRContext &context) {
  return {mlir::IntegerType::get(&context, 1),
          mlir::IntegerType::get(&context, 8),
          mlir::IntegerType::get(&context, 16),
          mlir::IntegerType::get(&context, 32),
          mlir::IntegerType::get(&context, 64),
          mlir::Float16Type::get(&context),
          mlir::BFloat16Type::get(&context),
          mlir::Float32Type::get(&context),
          mlir::Float64Type::get(&context)};
}

inline std::vector<mlir::Type>
catalogScalarPayloadTypes(mlir::MLIRContext &context) {
  std::vector<mlir::Type> result = catalogFixedVectorElementTypes(context);
  result.push_back(mlir::NoneType::get(&context));
  result.push_back(mlir::IndexType::get(&context));
  return result;
}

} // namespace loom::adg::detail

#endif // LOOM_ADG_CATALOGCAPABILITIES_H
