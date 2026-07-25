#include "FabricArtifactPreflightInternal.h"

#include "Common/ArtifactFinalizer.h"
#include "Common/ArtifactStore.h"
#include "Fabric/Artifact/FabricArtifactCodec.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <initializer_list>
#include <string>
#include <utility>
#include <vector>

using namespace loom;
using namespace loom::fabric;
using namespace loom::fabric::detail;

namespace {

constexpr ArtifactSchemaDescriptor foreignSchema{"loom.test.foreign",
                                                 SchemaVersion{1, 0}};

[[noreturn]] void fail(const char *test, const std::string &message) {
  llvm::errs() << test << ": " << message << "\n";
  std::exit(1);
}

void require(const char *test, bool condition, const std::string &message) {
  if (!condition)
    fail(test, message);
}

template <typename T>
T takeExpected(const char *test, llvm::Expected<T> value) {
  if (!value)
    fail(test, llvm::toString(value.takeError()));
  return std::move(*value);
}

void expectErrorContains(const char *test, llvm::Error error,
                         llvm::StringRef expected) {
  if (!error)
    fail(test, "expected an error containing '" + expected.str() + "'");
  const std::string message = llvm::toString(std::move(error));
  require(test, llvm::StringRef(message).contains(expected),
          "unexpected error: " + message);
}

template <typename T>
void expectErrorContains(const char *test, llvm::Expected<T> value,
                         llvm::StringRef expected) {
  if (value)
    fail(test, "expected an error containing '" + expected.str() + "'");
  expectErrorContains(test, value.takeError(), expected);
}

void expectGateError(const char *test, llvm::Error error,
                     FabricArtifactGateFailureKind expectedKind,
                     FabricArtifactGateReason expectedReason) {
  if (!error)
    fail(test, "expected a typed Fabric artifact gate error");

  bool matched = false;
  llvm::Error remaining = llvm::handleErrors(
      std::move(error), [&](const FabricArtifactGateError &gateError) {
        matched = true;
        require(test, gateError.kind() == expectedKind,
                "Fabric artifact gate failure kind changed");
        require(test, gateError.reason() == expectedReason,
                "Fabric artifact gate failure reason changed");
      });
  if (remaining)
    fail(test, "unexpected error: " + llvm::toString(std::move(remaining)));
  require(test, matched, "error was not a FabricArtifactGateError");
}

template <typename T>
void expectGateError(const char *test, llvm::Expected<T> value,
                     FabricArtifactGateFailureKind expectedKind,
                     FabricArtifactGateReason expectedReason) {
  if (value)
    fail(test, "expected a typed Fabric artifact gate error");
  expectGateError(test, value.takeError(), expectedKind, expectedReason);
}

void expectIncompletePreflight(
    const char *test,
    llvm::Expected<FabricArtifactDependencyPreflightIncomplete> result,
    FabricArtifactPreflightBlocker expectedBlocker) {
  const FabricArtifactDependencyPreflightIncomplete incomplete =
      takeExpected(test, std::move(result));
  require(test, incomplete.blocker == expectedBlocker,
          "Fabric artifact preflight blocker changed");
}

class TemporaryDirectory {
public:
  explicit TemporaryDirectory(const char *test) : test_(test) {
    llvm::SmallString<128> path;
    if (std::error_code error = llvm::sys::fs::createUniqueDirectory(
            "loom-fabric-artifact-gate-test", path))
      fail(test_, "unable to create temporary directory: " + error.message());
    path_ = path.str().str();
  }

  ~TemporaryDirectory() {
    if (std::error_code error = llvm::sys::fs::remove_directories(path_))
      llvm::errs() << test_ << ": unable to remove temporary directory: "
                   << error.message() << "\n";
  }

  llvm::StringRef path() const { return path_; }

private:
  const char *test_;
  std::string path_;
};

ArtifactIdentity identity(const char *test, std::uint8_t seed) {
  std::array<std::uint8_t, ArtifactIdentity::byteSize> bytes{};
  for (std::size_t index = 0; index < bytes.size(); ++index)
    bytes[index] = static_cast<std::uint8_t>(seed + index);
  return takeExpected(test, ArtifactIdentity::fromBytes(bytes));
}

ArtifactRootReference fabricReference(ArtifactIdentity artifact) {
  return ArtifactRootReference{fabricArtifactSchema.identity.str(),
                               fabricArtifactSchema.version,
                               std::move(artifact)};
}

ArtifactRootReference
fabricReference(const CanonicalSemanticBytes &canonicalBytes) {
  return fabricReference(
      finalizeArtifactIdentity(fabricArtifactSchema, canonicalBytes));
}

void requireStoredReference(const char *test, ArtifactStore &store,
                            const ArtifactRootReference &reference) {
  (void)takeExpected(test, store.get(reference));
}

void requireCandidateAbsent(const char *test, ArtifactStore &store,
                            const CanonicalSemanticBytes &canonicalBytes) {
  expectErrorContains(test, store.get(fabricReference(canonicalBytes)),
                      "artifact_store_missing");
}

CanonicalSemanticBytes
envelope(const char *test, FabricRootKind rootKind,
         llvm::ArrayRef<FabricDirectDependency> dependencies,
         std::initializer_list<std::uint8_t> payload) {
  const std::vector<std::uint8_t> payloadBytes(payload);
  return takeExpected(
      test, encodeFabricArtifactEnvelope(rootKind, dependencies, payloadBytes));
}

ArtifactRootReference
storeFabric(const char *test, ArtifactStore &store, FabricRootKind rootKind,
            llvm::ArrayRef<FabricDirectDependency> dependencies,
            std::initializer_list<std::uint8_t> payload) {
  CanonicalSemanticBytes bytes =
      envelope(test, rootKind, dependencies, payload);
  const ArtifactRootReference reference = fabricReference(bytes);
  const ArtifactIdentity stored =
      takeExpected(test, store.put(fabricArtifactSchema, bytes));
  require(test, stored == reference.artifact,
          "stored Fabric fixture identity changed");
  requireStoredReference(test, store, reference);
  return reference;
}

void roleAndRootKindLegalityPrecedeDependencyLoads() {
  TemporaryDirectory directory(__func__);
  ArtifactStore store(directory.path());
  const ArtifactRootReference missing = fabricReference(identity(__func__, 1));

  const CanonicalSemanticBytes moduleWithRefinedSystem = envelope(
      __func__, FabricRootKind::Module,
      {FabricDirectDependency{FabricDependencyRole::RefinedSystem, missing}},
      {0x01});
  expectGateError(__func__,
                  preflightCanonicalFabricArtifactDependencies(
                      store, moduleWithRefinedSystem),
                  FabricArtifactGateFailureKind::Invalid,
                  FabricArtifactGateReason::InvalidDependencyRoles);
  requireCandidateAbsent(__func__, store, moduleWithRefinedSystem);

  const CanonicalSemanticBytes implementationWithoutSystem = envelope(
      __func__, FabricRootKind::InterconnectImplementation, {}, {0x02});
  expectGateError(__func__,
                  preflightCanonicalFabricArtifactDependencies(
                      store, implementationWithoutSystem),
                  FabricArtifactGateFailureKind::Invalid,
                  FabricArtifactGateReason::InvalidDependencyRoles);
  requireCandidateAbsent(__func__, store, implementationWithoutSystem);

  const ArtifactRootReference other = fabricReference(identity(__func__, 2));
  const CanonicalSemanticBytes implementationWithTwoSystems = envelope(
      __func__, FabricRootKind::InterconnectImplementation,
      {FabricDirectDependency{FabricDependencyRole::RefinedSystem, missing},
       FabricDirectDependency{FabricDependencyRole::RefinedSystem, other}},
      {0x03});
  expectGateError(__func__,
                  preflightCanonicalFabricArtifactDependencies(
                      store, implementationWithTwoSystems),
                  FabricArtifactGateFailureKind::Invalid,
                  FabricArtifactGateReason::InvalidDependencyRoles);
  requireCandidateAbsent(__func__, store, implementationWithTwoSystems);
}

void preflightRecursivelyLoadsExactFabricDependencies() {
  TemporaryDirectory directory(__func__);
  ArtifactStore store(directory.path());

  const CanonicalSemanticBytes grandchildBytes =
      envelope(__func__, FabricRootKind::Module, {}, {0x11});
  const ArtifactRootReference grandchild = fabricReference(
      finalizeArtifactIdentity(fabricArtifactSchema, grandchildBytes));
  const ArtifactRootReference child =
      storeFabric(__func__, store, FabricRootKind::Module,
                  {FabricDirectDependency{FabricDependencyRole::ImportedModule,
                                          grandchild}},
                  {0x12});
  const CanonicalSemanticBytes root = envelope(
      __func__, FabricRootKind::System,
      {FabricDirectDependency{FabricDependencyRole::ImportedModule, child}},
      {0x13});

  expectErrorContains(__func__,
                      preflightCanonicalFabricArtifactDependencies(store, root),
                      "artifact_store_missing");
  requireCandidateAbsent(__func__, store, root);

  const ArtifactIdentity publishedGrandchild =
      takeExpected(__func__, store.put(fabricArtifactSchema, grandchildBytes));
  require(__func__, publishedGrandchild == grandchild.artifact,
          "grandchild fixture identity changed");
  requireStoredReference(__func__, store, grandchild);
  expectIncompletePreflight(
      __func__, preflightCanonicalFabricArtifactDependencies(store, root),
      FabricArtifactPreflightBlocker::DependencyUseDecoder);
  requireCandidateAbsent(__func__, store, root);
}

void foreignDependencyIsLoadedBeforeUnsupportedImporterRejection() {
  TemporaryDirectory directory(__func__);
  ArtifactStore store(directory.path());

  const ArtifactRootReference system =
      storeFabric(__func__, store, FabricRootKind::System, {}, {0x21});
  const CanonicalSemanticBytes foreignBytes(
      std::vector<std::uint8_t>{0xfa, 0xce});
  const ArtifactRootReference foreign{
      foreignSchema.identity.str(), foreignSchema.version,
      finalizeArtifactIdentity(foreignSchema, foreignBytes)};
  const CanonicalSemanticBytes root = envelope(
      __func__, FabricRootKind::InterconnectImplementation,
      {FabricDirectDependency{FabricDependencyRole::RefinedSystem, system},
       FabricDirectDependency{FabricDependencyRole::ImplementationInput,
                              foreign}},
      {0x22});

  expectErrorContains(__func__,
                      preflightCanonicalFabricArtifactDependencies(store, root),
                      "artifact_store_missing");
  requireCandidateAbsent(__func__, store, root);

  const ArtifactIdentity storedForeign =
      takeExpected(__func__, store.put(foreignSchema, foreignBytes));
  require(__func__, storedForeign == foreign.artifact,
          "foreign fixture identity changed");
  requireStoredReference(__func__, store, foreign);
  expectGateError(__func__,
                  preflightCanonicalFabricArtifactDependencies(store, root),
                  FabricArtifactGateFailureKind::Unsupported,
                  FabricArtifactGateReason::DependencyImporterUnavailable);
  requireCandidateAbsent(__func__, store, root);
}

void wrongKindAndDuplicateDependenciesAreRejected() {
  TemporaryDirectory directory(__func__);
  ArtifactStore store(directory.path());

  const ArtifactRootReference system =
      storeFabric(__func__, store, FabricRootKind::System, {}, {0x31});
  const CanonicalSemanticBytes wrongKind = envelope(
      __func__, FabricRootKind::Module,
      {FabricDirectDependency{FabricDependencyRole::ImportedModule, system}},
      {0x32});
  expectGateError(
      __func__, preflightCanonicalFabricArtifactDependencies(store, wrongKind),
      FabricArtifactGateFailureKind::Invalid,
      FabricArtifactGateReason::WrongDependencyRootKind);
  requireCandidateAbsent(__func__, store, wrongKind);

  const CanonicalSemanticBytes duplicate = envelope(
      __func__, FabricRootKind::InterconnectImplementation,
      {FabricDirectDependency{FabricDependencyRole::RefinedSystem, system},
       FabricDirectDependency{FabricDependencyRole::ImplementationInput,
                              system}},
      {0x33});
  expectGateError(
      __func__, preflightCanonicalFabricArtifactDependencies(store, duplicate),
      FabricArtifactGateFailureKind::Invalid,
      FabricArtifactGateReason::DuplicateDependency);
  requireCandidateAbsent(__func__, store, duplicate);
}

void dependencyUseVerificationWaitsForItsSemanticOwner() {
  TemporaryDirectory directory(__func__);
  ArtifactStore store(directory.path());

  const ArtifactRootReference module =
      storeFabric(__func__, store, FabricRootKind::Module, {}, {0x41});
  const CanonicalSemanticBytes root = envelope(
      __func__, FabricRootKind::System,
      {FabricDirectDependency{FabricDependencyRole::ImportedModule, module}},
      {0x42});
  expectIncompletePreflight(
      __func__, preflightCanonicalFabricArtifactDependencies(store, root),
      FabricArtifactPreflightBlocker::DependencyUseDecoder);
  requireCandidateAbsent(__func__, store, root);
}

void implementationInputKindRemainsAnIncompleteOwnerBoundary() {
  TemporaryDirectory directory(__func__);
  ArtifactStore store(directory.path());

  const ArtifactRootReference system =
      storeFabric(__func__, store, FabricRootKind::System, {}, {0x51});
  const ArtifactRootReference module =
      storeFabric(__func__, store, FabricRootKind::Module, {}, {0x52});
  const CanonicalSemanticBytes root = envelope(
      __func__, FabricRootKind::InterconnectImplementation,
      {FabricDirectDependency{FabricDependencyRole::RefinedSystem, system},
       FabricDirectDependency{FabricDependencyRole::ImplementationInput,
                              module}},
      {0x53});

  expectIncompletePreflight(
      __func__, preflightCanonicalFabricArtifactDependencies(store, root),
      FabricArtifactPreflightBlocker::ImplementationInputRootKind);
  requireCandidateAbsent(__func__, store, root);
}

void candidateRootsRemainAbsentAfterFailureOrIncompletePreflight() {
  TemporaryDirectory directory(__func__);
  ArtifactStore store(directory.path());

  const ArtifactRootReference missing =
      fabricReference(identity(__func__, 0x61));
  const CanonicalSemanticBytes missingClosure = envelope(
      __func__, FabricRootKind::Module,
      {FabricDirectDependency{FabricDependencyRole::ImportedModule, missing}},
      {0x62});
  expectErrorContains(
      __func__,
      preflightCanonicalFabricArtifactDependencies(store, missingClosure),
      "artifact_store_missing");
  requireCandidateAbsent(__func__, store, missingClosure);

  const CanonicalSemanticBytes emptyPayloadRoot =
      envelope(__func__, FabricRootKind::Module, {}, {});
  expectIncompletePreflight(
      __func__,
      preflightCanonicalFabricArtifactDependencies(store, emptyPayloadRoot),
      FabricArtifactPreflightBlocker::DependencyUseDecoder);
  requireCandidateAbsent(__func__, store, emptyPayloadRoot);

  const CanonicalSemanticBytes arbitraryPayloadRoot =
      envelope(__func__, FabricRootKind::Module, {}, {0x63, 0x64, 0x65});
  expectIncompletePreflight(
      __func__,
      preflightCanonicalFabricArtifactDependencies(store, arbitraryPayloadRoot),
      FabricArtifactPreflightBlocker::DependencyUseDecoder);
  requireCandidateAbsent(__func__, store, arbitraryPayloadRoot);
}

void contentAddressedCycleConstraintUsesProductionTraversal() {
  TemporaryDirectory directory(__func__);
  ArtifactStore store(directory.path());
  const ArtifactRootReference seed = fabricReference(identity(__func__, 0x71));
  const CanonicalSemanticBytes selfReferential = envelope(
      __func__, FabricRootKind::Module,
      {FabricDirectDependency{FabricDependencyRole::ImportedModule, seed}},
      {0x72});
  const ArtifactRootReference actual = fabricReference(
      finalizeArtifactIdentity(fabricArtifactSchema, selfReferential));
  require(__func__, actual != seed,
          "self-reference fixture unexpectedly solved the content-address "
          "equation");
  requireCandidateAbsent(__func__, store, selfReferential);

  FabricArtifactClosureTraversal traversal;
  require(__func__, takeExpected(__func__, traversal.enter(actual)),
          "first traversal entry was treated as already validated");
  expectGateError(__func__, traversal.enter(actual),
                  FabricArtifactGateFailureKind::Invalid,
                  FabricArtifactGateReason::CyclicDependency);
  traversal.complete(actual);
  require(__func__, !takeExpected(__func__, traversal.enter(actual)),
          "completed traversal entry was not memoized");
}

} // namespace

int main() {
  roleAndRootKindLegalityPrecedeDependencyLoads();
  preflightRecursivelyLoadsExactFabricDependencies();
  foreignDependencyIsLoadedBeforeUnsupportedImporterRejection();
  wrongKindAndDuplicateDependenciesAreRejected();
  dependencyUseVerificationWaitsForItsSemanticOwner();
  implementationInputKindRemainsAnIncompleteOwnerBoundary();
  candidateRootsRemainAbsentAfterFailureOrIncompletePreflight();
  contentAddressedCycleConstraintUsesProductionTraversal();
  llvm::outs() << "fabric artifact gate ok\n";
  return 0;
}
