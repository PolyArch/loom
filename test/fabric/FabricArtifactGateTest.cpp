#include "FabricArtifactDependencyClosureInternal.h"

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

void expectDependencyError(const char *test, llvm::Error error,
                           FabricArtifactDependencyFailureReason expectedReason,
                           llvm::StringRef expectedDiagnostic = {}) {
  if (!error)
    fail(test, "expected a typed Fabric artifact dependency error");

  bool matched = false;
  llvm::Error remaining = llvm::handleErrors(
      std::move(error),
      [&](const FabricArtifactDependencyError &dependencyError) {
        matched = true;
        require(test, dependencyError.reason() == expectedReason,
                "Fabric artifact dependency failure reason changed");
        if (!expectedDiagnostic.empty()) {
          std::string message;
          llvm::raw_string_ostream stream(message);
          dependencyError.log(stream);
          require(test,
                  llvm::StringRef(stream.str()).contains(expectedDiagnostic),
                  "Fabric artifact dependency error lost its codec "
                  "diagnostic: " +
                      message);
        }
      });
  if (remaining)
    fail(test, "unexpected error: " + llvm::toString(std::move(remaining)));
  require(test, matched, "error was not a FabricArtifactDependencyError");
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

void appendU32Be(std::vector<std::uint8_t> &bytes, std::uint32_t value) {
  bytes.push_back(static_cast<std::uint8_t>(value >> 24));
  bytes.push_back(static_cast<std::uint8_t>(value >> 16));
  bytes.push_back(static_cast<std::uint8_t>(value >> 8));
  bytes.push_back(static_cast<std::uint8_t>(value));
}

void appendU64Be(std::vector<std::uint8_t> &bytes, std::uint64_t value) {
  for (unsigned shift = 56; shift != 0; shift -= 8)
    bytes.push_back(static_cast<std::uint8_t>(value >> shift));
  bytes.push_back(static_cast<std::uint8_t>(value));
}

CanonicalSemanticBytes
implementationInputEnvelope(const ArtifactRootReference &refinedSystem,
                            const ArtifactRootReference &implementationInput,
                            std::initializer_list<std::uint8_t> payload) {
  static constexpr char domain[] = "loom.fabric.semantic.v6\0";
  std::vector<std::uint8_t> bytes(domain, domain + sizeof(domain) - 1);
  appendU32Be(bytes, 2);
  appendU64Be(bytes, 2);
  for (const auto &[role, reference] :
       {std::pair{FabricDependencyRole::RefinedSystem, &refinedSystem},
        std::pair{FabricDependencyRole::ImplementationInput,
                  &implementationInput}}) {
    appendU32Be(bytes, static_cast<std::uint32_t>(role));
    appendU32Be(bytes,
                static_cast<std::uint32_t>(reference->schemaIdentity.size()));
    bytes.insert(bytes.end(), reference->schemaIdentity.begin(),
                 reference->schemaIdentity.end());
    appendU32Be(bytes, reference->schemaVersion.major);
    appendU32Be(bytes, reference->schemaVersion.minor);
    bytes.insert(bytes.end(), reference->artifact.bytes().begin(),
                 reference->artifact.bytes().end());
  }
  appendU64Be(bytes, payload.size());
  bytes.insert(bytes.end(), payload.begin(), payload.end());
  return CanonicalSemanticBytes(std::move(bytes));
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

  const CanonicalSemanticBytes moduleWithImportedModule = envelope(
      __func__, FabricRootKind::Module,
      {FabricDirectDependency{FabricDependencyRole::ImportedModule, missing}},
      {0x01});
  expectDependencyError(
      __func__,
      validateFabricArtifactDependencyFramingClosure(store,
                                                     moduleWithImportedModule),
      FabricArtifactDependencyFailureReason::InvalidDependencyRoles);
  requireCandidateAbsent(__func__, store, moduleWithImportedModule);

  const CanonicalSemanticBytes implementationWithoutSystem = envelope(
      __func__, FabricRootKind::InterconnectImplementation, {}, {0x02});
  expectDependencyError(
      __func__,
      validateFabricArtifactDependencyFramingClosure(
          store, implementationWithoutSystem),
      FabricArtifactDependencyFailureReason::InvalidDependencyRoles);
  requireCandidateAbsent(__func__, store, implementationWithoutSystem);

  const ArtifactRootReference other = fabricReference(identity(__func__, 2));
  const CanonicalSemanticBytes implementationWithTwoSystems = envelope(
      __func__, FabricRootKind::InterconnectImplementation,
      {FabricDirectDependency{FabricDependencyRole::RefinedSystem, missing},
       FabricDirectDependency{FabricDependencyRole::RefinedSystem, other}},
      {0x03});
  expectDependencyError(
      __func__,
      validateFabricArtifactDependencyFramingClosure(
          store, implementationWithTwoSystems),
      FabricArtifactDependencyFailureReason::InvalidDependencyRoles);
  requireCandidateAbsent(__func__, store, implementationWithTwoSystems);
}

void framingClosureRecursivelyLoadsExactFabricDependencies() {
  TemporaryDirectory directory(__func__);
  ArtifactStore store(directory.path());

  const CanonicalSemanticBytes moduleBytes =
      envelope(__func__, FabricRootKind::Module, {}, {0x11});
  const ArtifactRootReference module = fabricReference(moduleBytes);
  const CanonicalSemanticBytes systemBytes = envelope(
      __func__, FabricRootKind::System,
      {FabricDirectDependency{FabricDependencyRole::ImportedModule, module}},
      {0x12});

  expectErrorContains(
      __func__,
      validateFabricArtifactDependencyFramingClosure(store, systemBytes),
      "artifact_store_missing");
  requireCandidateAbsent(__func__, store, systemBytes);

  const ArtifactIdentity publishedModule =
      takeExpected(__func__, store.put(fabricArtifactSchema, moduleBytes));
  require(__func__, publishedModule == module.artifact,
          "module fixture identity changed");
  requireStoredReference(__func__, store, module);
  if (llvm::Error error =
          validateFabricArtifactDependencyFramingClosure(store, systemBytes))
    fail(__func__, llvm::toString(std::move(error)));
  requireCandidateAbsent(__func__, store, systemBytes);
}

void implementationInputIsRejectedBeforeLookup() {
  TemporaryDirectory directory(__func__);
  ArtifactStore store(directory.path());

  const ArtifactRootReference system =
      storeFabric(__func__, store, FabricRootKind::System, {}, {0x21});
  const ArtifactRootReference foreign{foreignSchema.identity.str(),
                                      foreignSchema.version,
                                      identity(__func__, 0x22)};
  const CanonicalSemanticBytes root =
      implementationInputEnvelope(system, foreign, {0x23});
  expectDependencyError(
      __func__, validateFabricArtifactDependencyFramingClosure(store, root),
      FabricArtifactDependencyFailureReason::
          ImplementationInputOwnerUnavailable,
      "fabric_artifact_owner_contract_unavailable");
  requireCandidateAbsent(__func__, store, root);
}

void foreignSameFamilyRoleIsInvalid() {
  TemporaryDirectory directory(__func__);
  ArtifactStore store(directory.path());

  const CanonicalSemanticBytes foreignBytes(
      std::vector<std::uint8_t>{0x0f, 0x0e});
  const ArtifactRootReference foreign{
      foreignSchema.identity.str(), foreignSchema.version,
      finalizeArtifactIdentity(foreignSchema, foreignBytes)};
  (void)takeExpected(__func__, store.put(foreignSchema, foreignBytes));

  const CanonicalSemanticBytes root = envelope(
      __func__, FabricRootKind::System,
      {FabricDirectDependency{FabricDependencyRole::ImportedModule, foreign}},
      {0x0d});
  expectDependencyError(
      __func__, validateFabricArtifactDependencyFramingClosure(store, root),
      FabricArtifactDependencyFailureReason::ForeignDependency);
  requireCandidateAbsent(__func__, store, root);
}

void malformedCandidateEnvelopeIsTypedInvalid() {
  TemporaryDirectory directory(__func__);
  ArtifactStore store(directory.path());
  const CanonicalSemanticBytes malformed(std::vector<std::uint8_t>{0x00});

  expectDependencyError(
      __func__,
      validateFabricArtifactDependencyFramingClosure(store, malformed),
      FabricArtifactDependencyFailureReason::MalformedCandidateEnvelope,
      "truncated semantic domain");
  requireCandidateAbsent(__func__, store, malformed);
}

void invalidStoredFabricDependencyIsTypedInvalid() {
  TemporaryDirectory directory(__func__);
  ArtifactStore store(directory.path());
  const CanonicalSemanticBytes malformedDependency(
      std::vector<std::uint8_t>{0x00});
  const ArtifactRootReference dependency = fabricReference(malformedDependency);
  const ArtifactIdentity stored = takeExpected(
      __func__, store.put(fabricArtifactSchema, malformedDependency));
  require(__func__, stored == dependency.artifact,
          "stored invalid Fabric fixture identity changed");
  requireStoredReference(__func__, store, dependency);

  const CanonicalSemanticBytes root = envelope(
      __func__, FabricRootKind::System,
      {FabricDirectDependency{FabricDependencyRole::ImportedModule,
                              dependency}},
      {0x30});
  expectDependencyError(
      __func__, validateFabricArtifactDependencyFramingClosure(store, root),
      FabricArtifactDependencyFailureReason::InvalidDependencyEnvelope,
      "truncated semantic domain");
  requireCandidateAbsent(__func__, store, root);
}

void wrongKindDependencyIsRejected() {
  TemporaryDirectory directory(__func__);
  ArtifactStore store(directory.path());

  const ArtifactRootReference system =
      storeFabric(__func__, store, FabricRootKind::System, {}, {0x31});
  const CanonicalSemanticBytes wrongKind = envelope(
      __func__, FabricRootKind::System,
      {FabricDirectDependency{FabricDependencyRole::ImportedModule, system}},
      {0x32});
  expectDependencyError(
      __func__,
      validateFabricArtifactDependencyFramingClosure(store, wrongKind),
      FabricArtifactDependencyFailureReason::WrongDependencyRootKind);
  requireCandidateAbsent(__func__, store, wrongKind);
}

void sameFamilyFramingClosureDoesNotPublishTheRoot() {
  TemporaryDirectory directory(__func__);
  ArtifactStore store(directory.path());

  const ArtifactRootReference module =
      storeFabric(__func__, store, FabricRootKind::Module, {}, {0x41});
  const CanonicalSemanticBytes systemBytes = envelope(
      __func__, FabricRootKind::System,
      {FabricDirectDependency{FabricDependencyRole::ImportedModule, module}},
      {0x42});
  if (llvm::Error error =
          validateFabricArtifactDependencyFramingClosure(store, systemBytes))
    fail(__func__, llvm::toString(std::move(error)));
  requireCandidateAbsent(__func__, store, systemBytes);

  const ArtifactRootReference system = fabricReference(systemBytes);
  const ArtifactIdentity storedSystem =
      takeExpected(__func__, store.put(fabricArtifactSchema, systemBytes));
  require(__func__, storedSystem == system.artifact,
          "stored System fixture identity changed");
  requireStoredReference(__func__, store, system);
  const CanonicalSemanticBytes implementationBytes = envelope(
      __func__, FabricRootKind::InterconnectImplementation,
      {FabricDirectDependency{FabricDependencyRole::RefinedSystem, system}},
      {0x43});
  if (llvm::Error error = validateFabricArtifactDependencyFramingClosure(
          store, implementationBytes))
    fail(__func__, llvm::toString(std::move(error)));
  requireCandidateAbsent(__func__, store, implementationBytes);
}

void framingClosureNeverPublishesCandidateRoots() {
  TemporaryDirectory directory(__func__);
  ArtifactStore store(directory.path());

  const ArtifactRootReference missing =
      fabricReference(identity(__func__, 0x61));
  const CanonicalSemanticBytes missingClosure = envelope(
      __func__, FabricRootKind::System,
      {FabricDirectDependency{FabricDependencyRole::ImportedModule, missing}},
      {0x62});
  expectErrorContains(
      __func__,
      validateFabricArtifactDependencyFramingClosure(store, missingClosure),
      "artifact_store_missing");
  requireCandidateAbsent(__func__, store, missingClosure);

  const CanonicalSemanticBytes emptyPayloadRoot =
      envelope(__func__, FabricRootKind::Module, {}, {});
  if (llvm::Error error = validateFabricArtifactDependencyFramingClosure(
          store, emptyPayloadRoot))
    fail(__func__, llvm::toString(std::move(error)));
  requireCandidateAbsent(__func__, store, emptyPayloadRoot);

  const CanonicalSemanticBytes arbitraryPayloadRoot =
      envelope(__func__, FabricRootKind::Module, {}, {0x63, 0x64, 0x65});
  if (llvm::Error error = validateFabricArtifactDependencyFramingClosure(
          store, arbitraryPayloadRoot))
    fail(__func__, llvm::toString(std::move(error)));
  requireCandidateAbsent(__func__, store, arbitraryPayloadRoot);
}

void contentAddressedCycleConstraintUsesProductionTraversal() {
  const ArtifactRootReference actual =
      fabricReference(identity(__func__, 0x71));

  FabricArtifactDependencyClosureTraversal traversal;
  require(__func__, takeExpected(__func__, traversal.enter(actual)),
          "first traversal entry was treated as already validated");
  expectDependencyError(
      __func__, traversal.enter(actual).takeError(),
      FabricArtifactDependencyFailureReason::CyclicDependency);
  traversal.complete(actual);
  require(__func__, !takeExpected(__func__, traversal.enter(actual)),
          "completed traversal entry was not memoized");
}

} // namespace

int main() {
  malformedCandidateEnvelopeIsTypedInvalid();
  invalidStoredFabricDependencyIsTypedInvalid();
  roleAndRootKindLegalityPrecedeDependencyLoads();
  framingClosureRecursivelyLoadsExactFabricDependencies();
  implementationInputIsRejectedBeforeLookup();
  foreignSameFamilyRoleIsInvalid();
  wrongKindDependencyIsRejected();
  sameFamilyFramingClosureDoesNotPublishTheRoot();
  framingClosureNeverPublishesCandidateRoots();
  contentAddressedCycleConstraintUsesProductionTraversal();
  llvm::outs() << "fabric artifact gate ok\n";
  return 0;
}
