#include "FabricArtifactPreflightInternal.h"

#include "Common/ArtifactFinalizer.h"
#include "Fabric/Artifact/FabricArtifactCodec.h"

#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cassert>
#include <optional>
#include <string>
#include <system_error>
#include <utility>

namespace loom::fabric::detail {
namespace {

struct MissingOwners {
  bool dependencyRootKind = false;
};

llvm::Error gateError(FabricArtifactGateFailureKind kind,
                      FabricArtifactGateReason reason,
                      const llvm::Twine &message) {
  llvm::StringRef prefix;
  switch (kind) {
  case FabricArtifactGateFailureKind::Invalid:
    prefix = "fabric_artifact_invalid";
    break;
  case FabricArtifactGateFailureKind::Unsupported:
    prefix = "fabric_artifact_unsupported";
    break;
  }
  return llvm::make_error<FabricArtifactGateError>(
      kind, reason, (prefix + ": " + message).str());
}

llvm::Error wrapCodecError(FabricArtifactGateReason reason,
                           llvm::StringRef context, llvm::Error error) {
  std::string diagnostic = llvm::toString(std::move(error));
  llvm::StringRef detail(diagnostic);
  detail.consume_front("fabric_artifact_invalid: ");
  return gateError(FabricArtifactGateFailureKind::Invalid, reason,
                   llvm::Twine(context) + ": " + detail);
}

bool isFabricSchema(const ArtifactRootReference &reference) {
  return reference.schemaIdentity == fabricArtifactSchema.identity &&
         reference.schemaVersion == fabricArtifactSchema.version;
}

ArtifactRootReference
fabricReference(const CanonicalSemanticBytes &canonicalBytes) {
  return ArtifactRootReference{
      fabricArtifactSchema.identity.str(), fabricArtifactSchema.version,
      finalizeArtifactIdentity(fabricArtifactSchema, canonicalBytes)};
}

llvm::Error validateDependencyRoles(const DecodedFabricArtifact &artifact) {
  switch (artifact.rootKind) {
  case FabricRootKind::Module:
  case FabricRootKind::System:
    for (const FabricDirectDependency &dependency : artifact.dependencies)
      if (dependency.role != FabricDependencyRole::ImportedModule)
        return gateError(
            FabricArtifactGateFailureKind::Invalid,
            FabricArtifactGateReason::InvalidDependencyRoles,
            "Module and System roots admit only ImportedModule dependencies");
    return llvm::Error::success();

  case FabricRootKind::InterconnectImplementation: {
    unsigned refinedSystems = 0;
    for (const FabricDirectDependency &dependency : artifact.dependencies) {
      if (dependency.role == FabricDependencyRole::RefinedSystem) {
        ++refinedSystems;
        continue;
      }
      if (dependency.role != FabricDependencyRole::ImplementationInput)
        return gateError(
            FabricArtifactGateFailureKind::Invalid,
            FabricArtifactGateReason::InvalidDependencyRoles,
            "InterconnectImplementation roots admit only RefinedSystem and "
            "ImplementationInput dependencies");
    }
    if (refinedSystems != 1)
      return gateError(FabricArtifactGateFailureKind::Invalid,
                       FabricArtifactGateReason::InvalidDependencyRoles,
                       "InterconnectImplementation roots require exactly one "
                       "RefinedSystem dependency");
    return llvm::Error::success();
  }
  }
  llvm_unreachable("closed Fabric root kind");
}

std::optional<FabricRootKind> expectedRootKind(FabricDependencyRole role,
                                               MissingOwners &missingOwners) {
  switch (role) {
  case FabricDependencyRole::ImportedModule:
    return FabricRootKind::Module;
  case FabricDependencyRole::RefinedSystem:
    return FabricRootKind::System;
  case FabricDependencyRole::ImplementationInput:
    missingOwners.dependencyRootKind = true;
    return std::nullopt;
  }
  llvm_unreachable("closed Fabric dependency role");
}

llvm::Error preflightDecodedFabricArtifact(
    ArtifactStore &store, const ArtifactRootReference &reference,
    const DecodedFabricArtifact &artifact,
    FabricArtifactClosureTraversal &traversal, MissingOwners &missingOwners) {
  auto shouldDescend = traversal.enter(reference);
  if (!shouldDescend)
    return shouldDescend.takeError();
  if (!*shouldDescend)
    return llvm::Error::success();

  llvm::scope_exit abandonOnFailure([&] { traversal.abandon(reference); });
  if (llvm::Error error = validateDependencyRoles(artifact))
    return error;

  for (const FabricDirectDependency &dependency : artifact.dependencies) {
    auto canonicalDependency = store.get(dependency.root);
    if (!canonicalDependency)
      return canonicalDependency.takeError();
    if (!isFabricSchema(dependency.root))
      return gateError(
          FabricArtifactGateFailureKind::Unsupported,
          FabricArtifactGateReason::DependencyImporterUnavailable,
          "the exact dependency owner has no registered strict importer");

    auto decodedDependency =
        decodeFabricArtifactEnvelope(canonicalDependency->bytes());
    if (!decodedDependency)
      return wrapCodecError(FabricArtifactGateReason::InvalidDependencyEnvelope,
                            "same-family dependency import failed",
                            decodedDependency.takeError());

    const std::optional<FabricRootKind> requiredKind =
        expectedRootKind(dependency.role, missingOwners);
    if (requiredKind && decodedDependency->rootKind != *requiredKind)
      return gateError(
          FabricArtifactGateFailureKind::Invalid,
          FabricArtifactGateReason::WrongDependencyRootKind,
          "the imported Fabric dependency has the wrong root kind for its "
          "dependency role");

    if (llvm::Error error = preflightDecodedFabricArtifact(
            store, dependency.root, *decodedDependency, traversal,
            missingOwners))
      return error;
  }

  traversal.complete(reference);
  abandonOnFailure.release();
  return llvm::Error::success();
}

FabricArtifactDependencyPreflightIncomplete
incompletePreflight(const MissingOwners &missingOwners) {
  if (missingOwners.dependencyRootKind)
    return {FabricArtifactPreflightBlocker::ImplementationInputRootKind};
  return {FabricArtifactPreflightBlocker::DependencyUseDecoder};
}

} // namespace

char FabricArtifactGateError::ID = 0;

FabricArtifactGateError::FabricArtifactGateError(
    FabricArtifactGateFailureKind kind, FabricArtifactGateReason reason,
    std::string message)
    : kind_(kind), reason_(reason), message_(std::move(message)) {}

void FabricArtifactGateError::log(llvm::raw_ostream &stream) const {
  stream << message_;
}

std::error_code FabricArtifactGateError::convertToErrorCode() const {
  return llvm::inconvertibleErrorCode();
}

llvm::Expected<bool>
FabricArtifactClosureTraversal::enter(const ArtifactRootReference &reference) {
  assert(isFabricSchema(reference));
  auto [entry, inserted] =
      states_.emplace(reference.artifact.bytes(), VisitState::Active);
  if (inserted)
    return true;
  if (entry->second == VisitState::Active)
    return gateError(FabricArtifactGateFailureKind::Invalid,
                     FabricArtifactGateReason::CyclicDependency,
                     "the exact Fabric dependency closure is cyclic");
  return false;
}

void FabricArtifactClosureTraversal::abandon(
    const ArtifactRootReference &reference) {
  assert(isFabricSchema(reference));
  auto entry = states_.find(reference.artifact.bytes());
  assert(entry != states_.end() && entry->second == VisitState::Active);
  states_.erase(entry);
}

void FabricArtifactClosureTraversal::complete(
    const ArtifactRootReference &reference) {
  assert(isFabricSchema(reference));
  auto entry = states_.find(reference.artifact.bytes());
  assert(entry != states_.end() && entry->second == VisitState::Active);
  entry->second = VisitState::Validated;
}

llvm::Expected<FabricArtifactDependencyPreflightIncomplete>
preflightCanonicalFabricArtifactDependencies(
    ArtifactStore &store, const CanonicalSemanticBytes &canonicalBytes) {
  auto decoded = decodeFabricArtifactEnvelope(canonicalBytes.bytes());
  if (!decoded)
    return wrapCodecError(FabricArtifactGateReason::MalformedCandidateEnvelope,
                          "candidate canonical envelope decode failed",
                          decoded.takeError());

  FabricArtifactClosureTraversal traversal;
  MissingOwners missingOwners;
  if (llvm::Error error =
          preflightDecodedFabricArtifact(store, fabricReference(canonicalBytes),
                                         *decoded, traversal, missingOwners))
    return std::move(error);
  return incompletePreflight(missingOwners);
}

} // namespace loom::fabric::detail
