#ifndef LOOM_FABRIC_IR_IMPLEMENTATION_FAMILY_SCALAR_FLOAT_COMPARE_BEHAVIOR_H
#define LOOM_FABRIC_IR_IMPLEMENTATION_FAMILY_SCALAR_FLOAT_COMPARE_BEHAVIOR_H

#include "Fabric/IR/ImplementationFamily.h"

namespace fabric::detail {

bool ownsScalarFloatCompareBehaviorRelation(ImplementationFamilyId family);

llvm::Expected<std::vector<FiniteImplementationFamilyBehaviorPoint>>
resolveScalarFloatCompareBehaviorDomain(
    ImplementationFamilyId family, const FamilyCapabilityParams &params,
    llvm::ArrayRef<::dataflow::OperationSchemaId> enabledSchemas,
    llvm::ArrayRef<std::uint32_t> physicalInputWidths,
    llvm::ArrayRef<std::uint32_t> physicalResultWidths,
    ::mlir::MLIRContext &context);

llvm::Expected<::loom::CanonicalSemanticBytes>
projectScalarFloatCompareBehavior(
    ImplementationFamilyId family, const FamilyCapabilityParams &params,
    llvm::ArrayRef<::dataflow::OperationSchemaId> enabledSchemas,
    const ::dataflow::CanonicalActorSchemaProjection &actor,
    llvm::ArrayRef<FiniteImplementationFamilyBehaviorPoint> domain);

} // namespace fabric::detail

#endif // LOOM_FABRIC_IR_IMPLEMENTATION_FAMILY_SCALAR_FLOAT_COMPARE_BEHAVIOR_H
