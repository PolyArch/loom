#ifndef LOOM_FABRIC_IR_IMPLEMENTATION_FAMILY_BEHAVIOR_INTERNAL_H
#define LOOM_FABRIC_IR_IMPLEMENTATION_FAMILY_BEHAVIOR_INTERNAL_H

#include "Fabric/IR/ImplementationFamily.h"

namespace fabric::detail {

/// Arity-only compatibility query used by the existing semantic codec. The
/// sealed concrete-resource relation remains the authority for finalization.
llvm::Expected<bool> semanticConfigurationRequiresField(
    ImplementationFamilyId family, const FamilyCapabilityParams &params,
    llvm::ArrayRef<::dataflow::OperationSchemaId> enabledSchemas,
    std::uint32_t physicalInputCount, std::uint32_t physicalResultCount);

} // namespace fabric::detail

#endif // LOOM_FABRIC_IR_IMPLEMENTATION_FAMILY_BEHAVIOR_INTERNAL_H
