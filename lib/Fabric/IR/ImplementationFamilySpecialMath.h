#ifndef LOOM_LIB_FABRIC_IR_IMPLEMENTATIONFAMILYSPECIALMATH_H
#define LOOM_LIB_FABRIC_IR_IMPLEMENTATIONFAMILYSPECIALMATH_H

#include "Fabric/IR/ImplementationFamily.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <vector>

namespace fabric::detail {

llvm::Expected<::loom::CanonicalSemanticBytes>
encodeScalarSpecialMathSemanticConfiguration(
    ImplementationFamilyId family, const FamilyCapabilityParams &params,
    const ::dataflow::CanonicalActorSchemaProjection &actor);

llvm::Expected<std::vector<::dataflow::CanonicalActorSchemaProjection>>
enumerateScalarSpecialMathBehaviorActors(
    ImplementationFamilyId family, const FamilyCapabilityParams &params,
    llvm::ArrayRef<::dataflow::OperationSchemaId> enabledSchemas,
    ::mlir::MLIRContext &context);

} // namespace fabric::detail

#endif // LOOM_LIB_FABRIC_IR_IMPLEMENTATIONFAMILYSPECIALMATH_H
