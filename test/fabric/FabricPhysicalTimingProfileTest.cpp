#include "ADG/Builtin.h"
#include "Common/ArtifactStore.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Identity/FabricPhysicalTiming.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

namespace {

[[noreturn]] void fail(llvm::StringRef test, const std::string &message) {
  llvm::errs() << test << ": " << message << '\n';
  std::exit(EXIT_FAILURE);
}

void require(llvm::StringRef test, bool condition, const std::string &message) {
  if (!condition)
    fail(test, message);
}

template <typename T> T take(llvm::StringRef test, llvm::Expected<T> value) {
  if (!value)
    fail(test, llvm::toString(value.takeError()));
  return std::move(*value);
}

template <typename T>
void expectRejected(llvm::StringRef test, llvm::Expected<T> value,
                    llvm::StringRef expected) {
  if (value)
    fail(test, "invalid physical timing profile was accepted");
  const std::string message = llvm::toString(value.takeError());
  require(test, llvm::StringRef(message).contains(expected), message);
}

loom::fabric::FinalizedFabricRoot
importBuiltinModule(llvm::StringRef test, loom::ArtifactStore &store,
                    loom::adg::BuiltinTargetPreset preset) {
  auto target = take(test, loom::adg::buildBuiltinTarget(store, preset));
  require(test,
          target.roots().size() == 1 &&
              target.roots().front().directDependencies().size() == 1,
          "builtin target did not publish one System and one Module");
  return take(
      test,
      loom::fabric::importEntireFabricRoot(
          target.roots().front().directDependencies().front().root, store));
}

void artifactRoundTrip(loom::ArtifactStore &store) {
  const auto module = importBuiltinModule(
      __func__, store, loom::adg::BuiltinTargetPreset::Small);
  const auto otherModule = importBuiltinModule(
      __func__, store, loom::adg::BuiltinTargetPreset::Coverage);

  const auto normalized =
      take(__func__, loom::fabric::projectNormalizedFabricPhysicalTimingProfile(
                         module.view()));
  require(__func__, !normalized.traversals().empty(),
          "builtin Module has no physical traversal timing domain");
  const auto normalizedReference =
      take(__func__,
           loom::fabric::publishFabricPhysicalTimingProfile(normalized, store));
  const auto owner =
      take(__func__, loom::fabric::resolveFabricPhysicalTimingProfileOwner(
                         normalizedReference, store));
  const auto imported =
      take(__func__, loom::fabric::importFabricPhysicalTimingProfile(
                         normalizedReference, module.view(), store));
  require(
      __func__,
      owner == module.view().identity() &&
          imported.fabricIdentity() == module.view().identity() &&
          imported.kind() == normalized.kind() &&
          imported.providerIdentity() == normalized.providerIdentity() &&
          imported.technologyIdentity() == normalized.technologyIdentity() &&
          imported.characterizationIdentity() ==
              normalized.characterizationIdentity() &&
          imported.requiredCombinationalDelayQuanta() ==
              normalized.requiredCombinationalDelayQuanta() &&
          imported.canonicalViewBytes() == normalized.canonicalViewBytes() &&
          imported.digest() == normalized.digest(),
      "normalized profile Artifact did not round-trip exactly");

  std::vector<loom::fabric::FabricTraversalPhysicalTiming> characterized(
      normalized.traversals().begin(), normalized.traversals().end());
  ++characterized.front().delayQuanta;
  const auto target = take(
      __func__,
      loom::fabric::createFabricPhysicalTimingProfile(
          module.view(),
          loom::fabric::FabricPhysicalTimingProfileKind::TargetCharacterization,
          "openroad-routed.1", "saed32-edk-08-2025", "saed32rvt-tt1p05v25c", 17,
          characterized));
  const auto targetReference =
      take(__func__,
           loom::fabric::publishFabricPhysicalTimingProfile(target, store));
  const auto importedTarget =
      take(__func__, loom::fabric::importFabricPhysicalTimingProfile(
                         targetReference, module.view(), store));
  require(__func__,
          targetReference != normalizedReference &&
              importedTarget.kind() ==
                  loom::fabric::FabricPhysicalTimingProfileKind::
                      TargetCharacterization &&
              importedTarget.providerIdentity() == "openroad-routed.1" &&
              importedTarget.technologyIdentity() == "saed32-edk-08-2025" &&
              importedTarget.characterizationIdentity() ==
                  "saed32rvt-tt1p05v25c" &&
              importedTarget.requiredCombinationalDelayQuanta() == 17 &&
              importedTarget.digest() != imported.digest(),
          "target characterization did not change the profile Artifact");

  expectRejected(__func__,
                 loom::fabric::importFabricPhysicalTimingProfile(
                     normalizedReference, otherModule.view(), store),
                 "another Fabric artifact");

  const auto malformedIdentity = take(
      __func__,
      store.put(loom::fabric::fabricPhysicalTimingProfileArtifactSchema,
                loom::CanonicalSemanticBytes(std::vector<std::uint8_t>{0})));
  const loom::ArtifactRootReference malformedReference{
      loom::fabric::fabricPhysicalTimingProfileArtifactSchema.identity.str(),
      loom::fabric::fabricPhysicalTimingProfileArtifactSchema.version,
      malformedIdentity};
  expectRejected(__func__,
                 loom::fabric::resolveFabricPhysicalTimingProfileOwner(
                     malformedReference, store),
                 "truncated profile Fabric identity");
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2)
    fail("main", "expected one temporary directory argument");
  const std::filesystem::path root(argv[1]);
  std::filesystem::create_directories(root);
  loom::ArtifactStore store(root.string());
  artifactRoundTrip(store);
  return EXIT_SUCCESS;
}
