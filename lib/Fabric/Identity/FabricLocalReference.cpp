#include "Fabric/Identity/FabricLocalReference.h"

#include "Common/ArtifactLocalReferenceRegistry.h"
#include "Common/ArtifactText.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <mutex>
#include <vector>

namespace loom::fabric {
namespace {

llvm::Error fabricLocalReferenceError(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), message);
}

// The family's typed resolver: the published importer views of exact imported
// Fabric Artifacts, keyed by their own identities. The vector holds pointers
// to publisher-owned views with static storage duration, so a copied pointer
// stays valid no matter how the registry grows.
std::vector<const FabricArtifactView *> &fabricImporterViews() {
  static std::vector<const FabricArtifactView *> views;
  return views;
}

std::mutex &fabricImporterViewMutex() {
  static std::mutex mutex;
  return mutex;
}

const FabricArtifactView *
findFabricImporterView(const ArtifactIdentity &artifact) {
  std::lock_guard<std::mutex> lock(fabricImporterViewMutex());
  for (const FabricArtifactView *view : fabricImporterViews())
    if (view->identity() == artifact)
      return view;
  return nullptr;
}

template <FabricEntityKind Kind>
llvm::Error strictDecodeFabricEntityPayload(llvm::ArrayRef<std::uint8_t> payload) {
  llvm::Expected<FabricTypedEntityRef<Kind>> ref =
      decodeFabricRef<FabricTypedEntityRef<Kind>>(payload);
  if (!ref)
    return ref.takeError();
  const std::vector<std::uint8_t> reencoded = canonicalFabricBytes(*ref);
  if (llvm::ArrayRef<std::uint8_t>(reencoded) != payload)
    return fabricLocalReferenceError("noncanonical fabric entity reference "
                                     "payload");
  return llvm::Error::success();
}

template <FabricEntityKind Kind>
llvm::Error
validateFabricEntityLocalReference(const EncodedArtifactLocalReference &reference) {
  if (reference.artifact.schema != fabricArtifactSchema)
    return fabricLocalReferenceError(
        "fabric entity references require the exact loom.fabric 1.0 schema");
  llvm::Expected<FabricTypedEntityRef<Kind>> ref =
      decodeFabricRef<FabricTypedEntityRef<Kind>>(reference.payload);
  if (!ref)
    return ref.takeError();
  const FabricArtifactView *view =
      findFabricImporterView(reference.artifact.artifact);
  if (!view)
    return fabricLocalReferenceError(
        "no published fabric importer view for artifact '" +
        formatArtifactIdentityHex(reference.artifact.artifact) + "'");
  return validateFabricEntity(*view, Kind, ref->id());
}

template <FabricEntityKind Kind>
llvm::Error registerFabricEntityKind() {
  return registerArtifactLocalReferenceKind(
      fabricArtifactSchema, fabricEntityLocalKind<Kind>(),
      ArtifactLocalReferenceCodec{&strictDecodeFabricEntityPayload<Kind>,
                                  &validateFabricEntityLocalReference<Kind>});
}

} // namespace

llvm::Error publishFabricImporterView(const FabricArtifactView &view) {
  std::lock_guard<std::mutex> lock(fabricImporterViewMutex());
  for (const FabricArtifactView *published : fabricImporterViews()) {
    if (published->identity() != view.identity())
      continue;
    if (published == &view)
      return llvm::Error::success();
    return fabricLocalReferenceError(
        "conflicting fabric importer views for artifact '" +
        formatArtifactIdentityHex(view.identity()) + "'");
  }
  fabricImporterViews().push_back(&view);
  return llvm::Error::success();
}

llvm::Error registerFabricLocalReferenceKinds() {
#define LOOM_FABRIC_ENTITY(Name, Keyword)                                      \
  if (llvm::Error error = registerFabricEntityKind<FabricEntityKind::Name>())  \
    return error;
#include "Fabric/Identity/FabricRefs.def"
  return llvm::Error::success();
}

} // namespace loom::fabric
