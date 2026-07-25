#include "ImplementationPlatform/TechnologyCorner.h"

#include "Common/ArtifactLocalReferenceRegistry.h"
#include "Common/ArtifactText.h"

#include "llvm/ADT/Twine.h"

#include <cstdint>
#include <mutex>
#include <vector>

namespace loom::platform {
namespace {

llvm::Error platformError(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), message);
}

// The family's typed resolver: the published importer views of exact imported
// ImplementationPlatform Artifacts, keyed by their own identities. The vector
// holds pointers to publisher-owned views with static storage duration, so a
// copied pointer stays valid no matter how the registry grows.
std::vector<const ImplementationPlatformView *> &platformImporterViews() {
  static std::vector<const ImplementationPlatformView *> views;
  return views;
}

std::mutex &platformImporterViewMutex() {
  static std::mutex mutex;
  return mutex;
}

const ImplementationPlatformView *
findPlatformImporterView(const ArtifactIdentity &artifact) {
  std::lock_guard<std::mutex> lock(platformImporterViewMutex());
  for (const ImplementationPlatformView *view : platformImporterViews())
    if (view->identity() == artifact)
      return view;
  return nullptr;
}

llvm::Error strictDecodeTechnologyCornerPayload(llvm::ArrayRef<std::uint8_t> payload) {
  llvm::Expected<TechnologyCornerId> corner =
      decodeTechnologyCornerPayload(payload);
  if (!corner)
    return corner.takeError();
  return llvm::Error::success();
}

llvm::Error
validateTechnologyCornerLocalReference(const EncodedArtifactLocalReference &reference) {
  if (reference.artifact.schema != implementationPlatformSchema)
    return platformError("technology corner references require the exact "
                         "loom.implementation_platform 1.0 schema");
  llvm::Expected<TechnologyCornerRef> corner =
      decodeTechnologyCornerRef(reference);
  if (!corner)
    return corner.takeError();
  const ImplementationPlatformView *view =
      findPlatformImporterView(reference.artifact.artifact);
  if (!view)
    return platformError(
        "no published implementation platform importer view for artifact '" +
        formatArtifactIdentityHex(reference.artifact.artifact) + "'");
  return validateTechnologyCornerRef(*view, *corner);
}

} // namespace

ImplementationPlatformView::~ImplementationPlatformView() = default;

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
      ArtifactRootReference{implementationPlatformSchema, reference.artifact},
      implementationPlatformLocalKind(
          ImplementationPlatformLocalReferenceKind::TechnologyCorner),
      std::vector<std::uint8_t>(payload.begin(), payload.end())};
}

llvm::Expected<TechnologyCornerRef>
decodeTechnologyCornerRef(const EncodedArtifactLocalReference &reference) {
  if (reference.artifact.schema != implementationPlatformSchema)
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

llvm::Error
validateTechnologyCornerRef(const ImplementationPlatformView &view,
                            const TechnologyCornerRef &reference) {
  if (reference.artifact != view.identity())
    return platformError("the technology corner reference names a foreign "
                         "implementation platform artifact");
  if (reference.entity.value() >= view.technologyCornerCount())
    return platformError(
        "technology corner " + llvm::Twine(reference.entity.value()) +
        " does not resolve to a catalog entry of the exact platform");
  return llvm::Error::success();
}

llvm::Error
publishImplementationPlatformView(const ImplementationPlatformView &view) {
  std::lock_guard<std::mutex> lock(platformImporterViewMutex());
  for (const ImplementationPlatformView *published : platformImporterViews()) {
    if (published->identity() != view.identity())
      continue;
    if (published == &view)
      return llvm::Error::success();
    return platformError(
        "conflicting implementation platform importer views for artifact '" +
        formatArtifactIdentityHex(view.identity()) + "'");
  }
  platformImporterViews().push_back(&view);
  return llvm::Error::success();
}

llvm::Error registerImplementationPlatformLocalReferenceKinds() {
  return registerArtifactLocalReferenceKind(
      implementationPlatformSchema,
      implementationPlatformLocalKind(
          ImplementationPlatformLocalReferenceKind::TechnologyCorner),
      ArtifactLocalReferenceCodec{&strictDecodeTechnologyCornerPayload,
                                  &validateTechnologyCornerLocalReference});
}

} // namespace loom::platform
