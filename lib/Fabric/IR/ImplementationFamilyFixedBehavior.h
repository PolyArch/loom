#ifndef LOOM_LIB_FABRIC_IR_IMPLEMENTATIONFAMILYFIXEDBEHAVIOR_H
#define LOOM_LIB_FABRIC_IR_IMPLEMENTATIONFAMILYFIXEDBEHAVIOR_H

#include "Fabric/IR/ImplementationFamily.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <vector>

namespace fabric::detail {

bool ownsFixedBehaviorRelation(ImplementationFamilyId family);

llvm::Expected<std::vector<FiniteImplementationFamilyBehaviorPoint>>
resolveFixedBehaviorDomain(
    ImplementationFamilyId family, const FamilyCapabilityParams &params,
    llvm::ArrayRef<::dataflow::OperationSchemaId> enabledSchemas,
    llvm::ArrayRef<std::uint32_t> physicalInputWidths,
    llvm::ArrayRef<std::uint32_t> physicalResultWidths,
    ::mlir::MLIRContext &context);

} // namespace fabric::detail

#endif // LOOM_LIB_FABRIC_IR_IMPLEMENTATIONFAMILYFIXEDBEHAVIOR_H
