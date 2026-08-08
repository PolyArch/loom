#ifndef LOOM_FABRIC_IR_IMPLEMENTATION_FAMILY_BEHAVIOR_INTERNAL_H
#define LOOM_FABRIC_IR_IMPLEMENTATION_FAMILY_BEHAVIOR_INTERNAL_H

#include "Fabric/IR/ImplementationFamily.h"

namespace fabric::detail {

struct ImplementationFamilyBehaviorLaneImage final {
  std::vector<std::uint64_t> ordinals;
  std::uint64_t bound = 0;
};

using ImplementationFamilyBehaviorKeyComponent =
    std::variant<std::uint32_t, ::loom::CanonicalSemanticBytes,
                 ImplementationFamilyBehaviorLaneImage>;

llvm::Expected<::loom::CanonicalSemanticBytes>
encodeImplementationFamilyBehaviorKey(
    ImplementationFamilyId family, llvm::StringRef role,
    llvm::ArrayRef<ImplementationFamilyBehaviorKeyComponent> components);

/// Arity-only compatibility query used by the existing semantic codec. The
/// sealed concrete-resource relation remains the authority for finalization.
llvm::Expected<bool> semanticConfigurationRequiresField(
    ImplementationFamilyId family, const FamilyCapabilityParams &params,
    llvm::ArrayRef<::dataflow::OperationSchemaId> enabledSchemas,
    std::uint32_t physicalInputCount, std::uint32_t physicalResultCount);

} // namespace fabric::detail

#endif // LOOM_FABRIC_IR_IMPLEMENTATION_FAMILY_BEHAVIOR_INTERNAL_H
