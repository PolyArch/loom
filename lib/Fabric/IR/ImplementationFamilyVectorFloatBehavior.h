#ifndef LOOM_LIB_FABRIC_IR_IMPLEMENTATION_FAMILY_VECTOR_FLOAT_BEHAVIOR_H
#define LOOM_LIB_FABRIC_IR_IMPLEMENTATION_FAMILY_VECTOR_FLOAT_BEHAVIOR_H

#include "Fabric/IR/ImplementationFamily.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <vector>

namespace fabric::detail {

bool ownsFixedVectorFloatBehaviorRelation(ImplementationFamilyId family);

llvm::Expected<std::vector<FiniteImplementationFamilyBehaviorPoint>>
resolveFixedVectorFloatBehaviorDomain(
    ImplementationFamilyId family, const FamilyCapabilityParams &params,
    llvm::ArrayRef<::dataflow::OperationSchemaId> enabledSchemas,
    llvm::ArrayRef<std::uint32_t> physicalInputWidths,
    llvm::ArrayRef<std::uint32_t> physicalResultWidths,
    ::mlir::MLIRContext &context);

llvm::Expected<::loom::CanonicalSemanticBytes> projectFixedVectorFloatBehavior(
    ImplementationFamilyId family,
    const ::dataflow::CanonicalActorSchemaProjection &actor,
    llvm::ArrayRef<FiniteImplementationFamilyBehaviorPoint> domain);

} // namespace fabric::detail

#endif // LOOM_LIB_FABRIC_IR_IMPLEMENTATION_FAMILY_VECTOR_FLOAT_BEHAVIOR_H
