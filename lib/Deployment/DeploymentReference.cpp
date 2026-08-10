#include "Deployment/DeploymentReference.h"

#include "Deployment/Deployment.h"

#include <algorithm>
#include <system_error>

namespace loom::deployment {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      llvm::Twine("deployment_reference_invalid: ") + message);
}

template <typename Ref, typename Ordinal>
EncodedDeploymentCatalogReference encodeReference(const Ref &reference,
                                                  Ordinal ordinal) {
  EncodedDeploymentCatalogReference bytes{};
  std::copy(reference.deployment.bytes().begin(),
            reference.deployment.bytes().end(), bytes.begin());
  for (unsigned index = 0; index < sizeof(std::uint64_t); ++index)
    bytes[ArtifactIdentity::byteSize + index] = static_cast<std::uint8_t>(
        ordinal >> (8 * (sizeof(std::uint64_t) - 1 - index)));
  return bytes;
}

template <typename Ref>
llvm::Expected<Ref> decodeReference(llvm::ArrayRef<std::uint8_t> bytes) {
  if (bytes.size() != deploymentCatalogReferenceWireSize)
    return invalid("catalog reference has the wrong byte count");
  auto identity =
      ArtifactIdentity::fromBytes(bytes.take_front(ArtifactIdentity::byteSize));
  if (!identity)
    return identity.takeError();
  std::uint64_t ordinal = 0;
  for (std::uint8_t byte : bytes.drop_front(ArtifactIdentity::byteSize))
    ordinal = (ordinal << 8) | byte;
  return Ref{*identity, ordinal};
}

template <typename Ref, typename Ordinal>
bool referenceLess(const Ref &lhs, Ordinal lhsOrdinal, const Ref &rhs,
                   Ordinal rhsOrdinal) {
  if (lhs.deployment.bytes() != rhs.deployment.bytes())
    return lhs.deployment.bytes() < rhs.deployment.bytes();
  return lhsOrdinal < rhsOrdinal;
}

} // namespace

bool deploymentProgramEntryRefLess(const DeploymentProgramEntryRef &lhs,
                                   const DeploymentProgramEntryRef &rhs) {
  return referenceLess(lhs, lhs.programEntryOrdinal, rhs,
                       rhs.programEntryOrdinal);
}

bool deploymentExternalInterfaceRefLess(
    const DeploymentExternalInterfaceRef &lhs,
    const DeploymentExternalInterfaceRef &rhs) {
  return referenceLess(lhs, lhs.externalInterfaceOrdinal, rhs,
                       rhs.externalInterfaceOrdinal);
}

EncodedDeploymentCatalogReference
encodeDeploymentProgramEntryRef(const DeploymentProgramEntryRef &reference) {
  return encodeReference(reference, reference.programEntryOrdinal);
}

EncodedDeploymentCatalogReference encodeDeploymentExternalInterfaceRef(
    const DeploymentExternalInterfaceRef &reference) {
  return encodeReference(reference, reference.externalInterfaceOrdinal);
}

llvm::Expected<DeploymentProgramEntryRef>
decodeDeploymentProgramEntryRef(llvm::ArrayRef<std::uint8_t> bytes) {
  return decodeReference<DeploymentProgramEntryRef>(bytes);
}

llvm::Expected<DeploymentExternalInterfaceRef>
decodeDeploymentExternalInterfaceRef(llvm::ArrayRef<std::uint8_t> bytes) {
  return decodeReference<DeploymentExternalInterfaceRef>(bytes);
}

llvm::Expected<const HostProgramEntry *>
resolveDeploymentProgramEntry(const FinalizedDeployment &deployment,
                              const DeploymentProgramEntryRef &reference) {
  if (reference.deployment != deployment.reference().artifact)
    return invalid("program entry names a foreign Deployment");
  llvm::ArrayRef<HostProgramEntry> entries =
      deployment.deployment().hostProgram().programEntries();
  if (reference.programEntryOrdinal >= entries.size())
    return invalid("program entry ordinal is out of range");
  const HostProgramEntry &entry = entries[reference.programEntryOrdinal];
  if (entry.entryOrdinal != reference.programEntryOrdinal)
    return invalid("program entry catalog is not dense and canonical");
  return &entry;
}

llvm::Expected<const HostExternalInterface *>
resolveDeploymentExternalInterface(
    const FinalizedDeployment &deployment,
    const DeploymentExternalInterfaceRef &reference) {
  if (reference.deployment != deployment.reference().artifact)
    return invalid("external interface names a foreign Deployment");
  llvm::ArrayRef<HostExternalInterface> interfaces =
      deployment.deployment().hostProgram().externalInterfaces();
  if (reference.externalInterfaceOrdinal >= interfaces.size())
    return invalid("external interface ordinal is out of range");
  const HostExternalInterface &interface =
      interfaces[reference.externalInterfaceOrdinal];
  if (interface.interfaceOrdinal != reference.externalInterfaceOrdinal)
    return invalid("external interface catalog is not dense and canonical");
  return &interface;
}

} // namespace loom::deployment
