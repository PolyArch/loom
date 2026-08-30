#include "Fabric/Artifact/FabricArtifactMigration.h"

#include "Common/ArtifactFinalizer.h"
#include "Fabric/Artifact/FabricArtifact.h"

#include "llvm/ADT/Twine.h"

#include <system_error>
#include <utility>
#include <vector>

namespace loom::fabric {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(std::make_error_code(std::errc::invalid_argument),
                                 "fabric_artifact_invalid: " + message);
}

} // namespace

llvm::Expected<ArtifactRootReference>
migrateFabricRootV7_0ToV7_1(const ArtifactRootReference &reference,
                            const ArtifactStore &store) {
  if (reference.schemaIdentity != fabricArtifactSchemaV7_0.identity ||
      reference.schemaVersion != fabricArtifactSchemaV7_0.version)
    return invalid("migration requires an exact loom.fabric 7.0 root "
                   "reference");
  auto canonical = store.get(reference);
  if (!canonical)
    return canonical.takeError();
  if (finalizeArtifactIdentity(fabricArtifactSchemaV7_0, *canonical) !=
      reference.artifact)
    return invalid("loom.fabric 7.0 reference identity does not match its "
                   "canonical bytes");
  auto decoded = decodeFabricArtifactEnvelope(canonical->bytes());
  if (!decoded)
    return decoded.takeError();

  std::vector<FabricDirectDependency> migrated;
  migrated.reserve(decoded->dependencies.size());
  for (const FabricDirectDependency &dependency : decoded->dependencies) {
    auto migratedRoot = migrateFabricRootV7_0ToV7_1(dependency.root, store);
    if (!migratedRoot)
      return migratedRoot.takeError();
    migrated.push_back({dependency.role, *migratedRoot});
  }

  // The canonical payload embeds dependency-table ordinals of the sorted row
  // order. The identity rewrite must preserve that permutation exactly; a
  // closure whose rows would reorder under 7.1 cannot be migrated at the
  // envelope level and must be re-finalized from its authoring source.
  auto orderBefore =
      canonicalFabricDependencyOrder(decoded->dependencies);
  if (!orderBefore)
    return orderBefore.takeError();
  auto orderAfter = canonicalFabricDependencyOrder(migrated);
  if (!orderAfter)
    return orderAfter.takeError();
  if (*orderBefore != *orderAfter)
    return invalid("loom.fabric 7.0 closure reorders its canonical dependency "
                   "rows under 7.1 identities; re-finalize from the authoring "
                   "source instead of migrating");

  auto migratedBytes = encodeFabricArtifactEnvelope(
      decoded->rootKind, migrated, decoded->canonicalMlirBytecode);
  if (!migratedBytes)
    return migratedBytes.takeError();
  ArtifactIdentity identity =
      finalizeArtifactIdentity(fabricArtifactSchema, *migratedBytes);
  ArtifactRootReference result{fabricArtifactSchema.identity.str(),
                               fabricArtifactSchema.version, identity};

  auto existing = store.get(result);
  if (!existing) {
    llvm::consumeError(existing.takeError());
    auto stored = store.put(fabricArtifactSchema, *migratedBytes);
    if (!stored)
      return stored.takeError();
    if (*stored != identity)
      return invalid("ArtifactStore returned a different migrated identity");
  }

  // Re-finalization is complete only after the migrated root passes the same
  // strict import every native 7.1 root passes.
  auto imported = importEntireFabricRoot(result, store);
  if (!imported)
    return imported.takeError();
  return result;
}

} // namespace loom::fabric
