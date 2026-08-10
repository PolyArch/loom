#ifndef LOOM_DEPLOYMENT_DEPLOYMENTREFERENCE_H
#define LOOM_DEPLOYMENT_DEPLOYMENTREFERENCE_H

#include "Common/Artifact.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace loom::deployment {

class FinalizedDeployment;
struct HostExternalInterface;
struct HostProgramEntry;

/// One exact entry in the inline HostProgramLeaf catalog of a finalized
/// Deployment. The ordinal has no meaning without the exact Deployment.
struct DeploymentProgramEntryRef final {
  ArtifactIdentity deployment;
  std::uint64_t programEntryOrdinal = 0;

  friend bool operator==(const DeploymentProgramEntryRef &lhs,
                         const DeploymentProgramEntryRef &rhs) {
    return lhs.deployment == rhs.deployment &&
           lhs.programEntryOrdinal == rhs.programEntryOrdinal;
  }
};

/// One exact external interface in the inline HostProgramLeaf catalog of a
/// finalized Deployment.
struct DeploymentExternalInterfaceRef final {
  ArtifactIdentity deployment;
  std::uint64_t externalInterfaceOrdinal = 0;

  friend bool operator==(const DeploymentExternalInterfaceRef &lhs,
                         const DeploymentExternalInterfaceRef &rhs) {
    return lhs.deployment == rhs.deployment &&
           lhs.externalInterfaceOrdinal == rhs.externalInterfaceOrdinal;
  }
};

inline constexpr std::size_t deploymentCatalogReferenceWireSize =
    ArtifactIdentity::byteSize + sizeof(std::uint64_t);
using EncodedDeploymentCatalogReference =
    std::array<std::uint8_t, deploymentCatalogReferenceWireSize>;

bool deploymentProgramEntryRefLess(const DeploymentProgramEntryRef &lhs,
                                   const DeploymentProgramEntryRef &rhs);
bool deploymentExternalInterfaceRefLess(
    const DeploymentExternalInterfaceRef &lhs,
    const DeploymentExternalInterfaceRef &rhs);

EncodedDeploymentCatalogReference
encodeDeploymentProgramEntryRef(const DeploymentProgramEntryRef &reference);
EncodedDeploymentCatalogReference encodeDeploymentExternalInterfaceRef(
    const DeploymentExternalInterfaceRef &reference);

llvm::Expected<DeploymentProgramEntryRef>
decodeDeploymentProgramEntryRef(llvm::ArrayRef<std::uint8_t> bytes);
llvm::Expected<DeploymentExternalInterfaceRef>
decodeDeploymentExternalInterfaceRef(llvm::ArrayRef<std::uint8_t> bytes);

llvm::Expected<const HostProgramEntry *>
resolveDeploymentProgramEntry(const FinalizedDeployment &deployment,
                              const DeploymentProgramEntryRef &reference);
llvm::Expected<const HostExternalInterface *>
resolveDeploymentExternalInterface(
    const FinalizedDeployment &deployment,
    const DeploymentExternalInterfaceRef &reference);

} // namespace loom::deployment

#endif // LOOM_DEPLOYMENT_DEPLOYMENTREFERENCE_H
