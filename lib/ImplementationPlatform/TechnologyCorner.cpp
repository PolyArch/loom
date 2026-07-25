#include "ImplementationPlatform/TechnologyCorner.h"

#include "llvm/ADT/Twine.h"

#include <cstdint>

namespace loom::platform {
namespace {

llvm::Error platformError(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), message);
}

} // namespace

std::array<std::uint8_t, 8>
encodeTechnologyCornerPayload(TechnologyCornerId corner) {
  std::array<std::uint8_t, 8> payload{};
  const std::uint64_t value = corner.value();
  for (unsigned index = 0; index < payload.size(); ++index)
    payload[index] = static_cast<std::uint8_t>(value >> (56 - index * 8));
  return payload;
}

llvm::Expected<TechnologyCornerId>
decodeTechnologyCornerPayload(llvm::ArrayRef<std::uint8_t> payload) {
  if (payload.size() != 8)
    return platformError("a technology corner reference payload is exactly "
                         "eight bytes");
  std::uint64_t value = 0;
  for (std::uint8_t byte : payload)
    value = (value << 8) | byte;
  return TechnologyCornerId(value);
}

EncodedArtifactLocalReference
encodeTechnologyCornerRef(const TechnologyCornerRef &reference) {
  std::array<std::uint8_t, 8> payload =
      encodeTechnologyCornerPayload(reference.entity);
  return EncodedArtifactLocalReference{
      ArtifactRootReference{implementationPlatformSchema.identity.str(),
                            implementationPlatformSchema.version,
                            reference.artifact},
      implementationPlatformLocalKind(
          ImplementationPlatformLocalReferenceKind::TechnologyCorner),
      std::vector<std::uint8_t>(payload.begin(), payload.end())};
}

llvm::Expected<TechnologyCornerRef>
decodeTechnologyCornerRef(const EncodedArtifactLocalReference &reference) {
  if (reference.artifact.schemaIdentity !=
          implementationPlatformSchema.identity ||
      reference.artifact.schemaVersion != implementationPlatformSchema.version)
    return platformError("technology corner references require the exact "
                         "loom.implementation_platform 1.0 schema");
  if (reference.ownerLocalKind !=
      implementationPlatformLocalKind(
          ImplementationPlatformLocalReferenceKind::TechnologyCorner))
    return platformError("local-reference kind " +
                         llvm::Twine(reference.ownerLocalKind) +
                         " is not the technology corner kind of schema '" +
                         implementationPlatformSchema.identity + "'");
  llvm::Expected<TechnologyCornerId> corner =
      decodeTechnologyCornerPayload(reference.payload);
  if (!corner)
    return corner.takeError();
  return TechnologyCornerRef{reference.artifact.artifact, *corner};
}

} // namespace loom::platform
