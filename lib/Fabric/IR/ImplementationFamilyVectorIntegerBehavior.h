#ifndef LOOM_LIB_FABRIC_IR_IMPLEMENTATIONFAMILYVECTORINTEGERBEHAVIOR_H
#define LOOM_LIB_FABRIC_IR_IMPLEMENTATIONFAMILYVECTORINTEGERBEHAVIOR_H

#include "Fabric/IR/ImplementationFamily.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <vector>

namespace fabric::detail {

bool ownsFixedVectorIntegerBehaviorRelation(ImplementationFamilyId family);

llvm::Expected<std::vector<FiniteImplementationFamilyBehaviorPoint>>
resolveFixedVectorIntegerBehaviorDomain(
    ImplementationFamilyId family, const FamilyCapabilityParams &params,
    llvm::ArrayRef<::dataflow::OperationSchemaId> enabledSchemas,
    llvm::ArrayRef<std::uint32_t> physicalInputWidths,
    llvm::ArrayRef<std::uint32_t> physicalResultWidths,
    ::mlir::MLIRContext &context);

llvm::Expected<::loom::CanonicalSemanticBytes>
projectFixedVectorIntegerBehavior(
    ImplementationFamilyId family,
    const ::dataflow::CanonicalActorSchemaProjection &actor,
    llvm::ArrayRef<FiniteImplementationFamilyBehaviorPoint> domain);

} // namespace fabric::detail

#endif // LOOM_LIB_FABRIC_IR_IMPLEMENTATIONFAMILYVECTORINTEGERBEHAVIOR_H
