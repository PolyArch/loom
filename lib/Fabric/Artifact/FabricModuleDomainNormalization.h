#ifndef LOOM_LIB_FABRIC_ARTIFACT_FABRICMODULEDOMAINNORMALIZATION_H
#define LOOM_LIB_FABRIC_ARTIFACT_FABRICMODULEDOMAINNORMALIZATION_H

#include "FabricCanonicalLabeling.h"

#include "Fabric/IR/ModuleDomain.h"

#include "llvm/Support/Error.h"

#include <cstddef>
#include <vector>

namespace fabric {
class ModuleOp;
}

namespace loom::fabric::detail {

struct NormalizedModuleDomainSlot final {
  FabricClockResetKind kind = FabricClockResetKind::Clock;
  FabricOrdinal provisionalOrdinal = 0;
};

struct NormalizedModuleDomainMember final {
  bool boundary = false;
  FabricPortDirection direction = FabricPortDirection::Input;
  mlir::Operation *owner = nullptr;
  ::fabric::ModuleDomainAuthoringRelation::InternalMemberRole role =
      ::fabric::ModuleDomainAuthoringRelation::InternalMemberRole::Occurrence;
  FabricOrdinal ordinal = 0;

  friend bool operator==(const NormalizedModuleDomainMember &left,
                         const NormalizedModuleDomainMember &right) {
    return left.boundary == right.boundary &&
           left.direction == right.direction && left.owner == right.owner &&
           left.role == right.role && left.ordinal == right.ordinal;
  }
};

struct NormalizedModuleDomainAssignment final {
  std::size_t member = 0;
  std::size_t slot = 0;
};

struct NormalizedModuleDomainRelation final {
  std::vector<NormalizedModuleDomainSlot> slots;
  std::vector<NormalizedModuleDomainMember> members;
  std::vector<NormalizedModuleDomainAssignment> assignments;
};

llvm::Expected<NormalizedModuleDomainRelation> normalizeFabricModuleDomain(
    ::fabric::ModuleOp root,
    const ::fabric::ModuleDomainAuthoringRelation &authoring);

llvm::Expected<NormalizedModuleDomainRelation>
buildDefaultFabricModuleDomain(::fabric::ModuleOp root);

/// Recovers the sole Builder-lifetime authoring relation from one validated
/// canonical carrier. This is used only to derive a fresh ordinary draft from
/// a finalized Module; it introduces no second persistent representation.
llvm::Expected<::fabric::ModuleDomainAuthoringRelation>
recoverFabricModuleDomainAuthoring(::fabric::ModuleOp root);

/// Reconstructs the identifier-free relation from a stored canonical carrier,
/// reruns domain-aware canonical labeling, and rejects any carrier that is not
/// the exact materialization of that relation.
llvm::Expected<FabricCanonicalLabeling>
validateStoredFabricModuleDomain(::fabric::ModuleOp root);

} // namespace loom::fabric::detail

#endif // LOOM_LIB_FABRIC_ARTIFACT_FABRICMODULEDOMAINNORMALIZATION_H
