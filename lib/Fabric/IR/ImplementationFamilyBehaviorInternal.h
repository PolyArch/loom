#ifndef LOOM_FABRIC_IR_IMPLEMENTATION_FAMILY_BEHAVIOR_INTERNAL_H
#define LOOM_FABRIC_IR_IMPLEMENTATION_FAMILY_BEHAVIOR_INTERNAL_H

#include "Fabric/IR/ImplementationFamily.h"

namespace fabric::detail {

::mlir::arith::FastMathFlags
minimalFloatingActorPermissions(const FloatBehaviorProfile &behavior);

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

llvm::Error validateImplementationFamilyBehaviorPoint(
    ImplementationFamilyId family, const FamilyCapabilityParams &params,
    const ::dataflow::CanonicalActorSchemaProjection &actor,
    llvm::ArrayRef<std::uint64_t> operandPorts,
    llvm::ArrayRef<std::uint64_t> resultPorts,
    llvm::ArrayRef<std::uint32_t> physicalInputWidths,
    llvm::ArrayRef<std::uint32_t> physicalResultWidths,
    std::optional<ResolvedIndexWidth> resolvedIndexWidth = std::nullopt);

bool ownsControlBehaviorRelation(ImplementationFamilyId family);

llvm::Expected<std::vector<FiniteImplementationFamilyBehaviorPoint>>
resolveControlBehaviorDomain(
    ImplementationFamilyId family, const FamilyCapabilityParams &params,
    llvm::ArrayRef<::dataflow::OperationSchemaId> enabledSchemas,
    llvm::ArrayRef<std::uint32_t> physicalInputWidths,
    llvm::ArrayRef<std::uint32_t> physicalResultWidths,
    ::mlir::MLIRContext &context);

llvm::Expected<::loom::CanonicalSemanticBytes> projectControlBehaviorKey(
    ImplementationFamilyId family,
    llvm::ArrayRef<FiniteImplementationFamilyBehaviorPoint> domain,
    const ::dataflow::CanonicalActorSchemaProjection &actor,
    llvm::ArrayRef<std::uint64_t> operandPorts,
    llvm::ArrayRef<std::uint64_t> resultPorts);

} // namespace fabric::detail

#endif // LOOM_FABRIC_IR_IMPLEMENTATION_FAMILY_BEHAVIOR_INTERNAL_H
