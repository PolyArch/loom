#include "FabricArtifactDependencyClosureInternal.h"

#include "Common/ArtifactFinalizer.h"
#include "Fabric/Artifact/FabricArtifactCodec.h"

#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cassert>
#include <string>
#include <system_error>
#include <utility>

namespace loom::fabric::detail {
namespace {

llvm::Error dependencyError(FabricArtifactDependencyFailureReason reason,
                            const llvm::Twine &message) {
  const llvm::StringRef prefix =
      reason == FabricArtifactDependencyFailureReason::
                    ImplementationInputOwnerUnavailable
          ? "fabric_artifact_owner_contract_unavailable"
          : "fabric_artifact_invalid";
  return llvm::make_error<FabricArtifactDependencyError>(
      reason, (prefix + ": " + message).str());
}

llvm::Error wrapCodecError(FabricArtifactDependencyFailureReason reason,
                           llvm::StringRef context, llvm::Error error) {
  std::string diagnostic = llvm::toString(std::move(error));
  llvm::StringRef detail(diagnostic);
  detail.consume_front("fabric_artifact_invalid: ");
  return dependencyError(reason, llvm::Twine(context) + ": " + detail);
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
  for (const FabricDirectDependency &dependency : artifact.dependencies)
    if (dependency.role == FabricDependencyRole::ImplementationInput)
      return dependencyError(
          FabricArtifactDependencyFailureReason::
              ImplementationInputOwnerUnavailable,
          "ImplementationInput has no closed artifact owner or strict "
          "import contract");

  switch (artifact.rootKind) {
  case FabricRootKind::Module:
    if (!artifact.dependencies.empty())
      return dependencyError(
          FabricArtifactDependencyFailureReason::InvalidDependencyRoles,
          "Module roots admit no direct dependencies");
    return llvm::Error::success();

  case FabricRootKind::System:
    for (const FabricDirectDependency &dependency : artifact.dependencies)
      if (dependency.role != FabricDependencyRole::ImportedModule)
        return dependencyError(
            FabricArtifactDependencyFailureReason::InvalidDependencyRoles,
            "System roots admit only ImportedModule dependencies");
    return llvm::Error::success();

  case FabricRootKind::InterconnectImplementation: {
    unsigned refinedSystems = 0;
    for (const FabricDirectDependency &dependency : artifact.dependencies) {
      if (dependency.role == FabricDependencyRole::RefinedSystem) {
        ++refinedSystems;
        continue;
      }
      return dependencyError(
          FabricArtifactDependencyFailureReason::InvalidDependencyRoles,
          "InterconnectImplementation roots admit only one RefinedSystem "
          "dependency");
    }
    if (refinedSystems != 1)
      return dependencyError(
          FabricArtifactDependencyFailureReason::InvalidDependencyRoles,
          "InterconnectImplementation roots require exactly one "
          "RefinedSystem dependency");
    return llvm::Error::success();
  }
  }
  llvm_unreachable("closed Fabric root kind");
}

FabricRootKind expectedFabricRootKind(FabricDependencyRole role) {
  switch (role) {
  case FabricDependencyRole::ImportedModule:
    return FabricRootKind::Module;
  case FabricDependencyRole::RefinedSystem:
    return FabricRootKind::System;
  case FabricDependencyRole::ImplementationInput:
    llvm_unreachable("ImplementationInput has no closed owner root kind");
  }
  llvm_unreachable("closed Fabric dependency role");
}

llvm::Error validateDecodedFabricArtifact(
    const ArtifactStore &store, const ArtifactRootReference &reference,
    const DecodedFabricArtifact &artifact,
    FabricArtifactDependencyClosureTraversal &traversal) {
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
      return dependencyError(
          FabricArtifactDependencyFailureReason::ForeignDependency,
          "ImportedModule and RefinedSystem dependencies must be exact "
          "loom.fabric 2.0 roots");

    auto decodedDependency =
        decodeFabricArtifactEnvelope(canonicalDependency->bytes());
    if (!decodedDependency)
      return wrapCodecError(
          FabricArtifactDependencyFailureReason::InvalidDependencyEnvelope,
          "same-family dependency envelope decode failed",
          decodedDependency.takeError());

    const FabricRootKind requiredKind = expectedFabricRootKind(dependency.role);
    if (decodedDependency->rootKind != requiredKind)
      return dependencyError(
          FabricArtifactDependencyFailureReason::WrongDependencyRootKind,
          "the imported Fabric dependency has the wrong root kind for its "
          "dependency role");

    if (llvm::Error error = validateDecodedFabricArtifact(
            store, dependency.root, *decodedDependency, traversal))
      return error;
  }

  traversal.complete(reference);
  abandonOnFailure.release();
  return llvm::Error::success();
}

} // namespace

char FabricArtifactDependencyError::ID = 0;

FabricArtifactDependencyError::FabricArtifactDependencyError(
    FabricArtifactDependencyFailureReason reason, std::string message)
    : reason_(reason), message_(std::move(message)) {}

void FabricArtifactDependencyError::log(llvm::raw_ostream &stream) const {
  stream << message_;
}

std::error_code FabricArtifactDependencyError::convertToErrorCode() const {
  return llvm::inconvertibleErrorCode();
}

llvm::Expected<bool> FabricArtifactDependencyClosureTraversal::enter(
    const ArtifactRootReference &reference) {
  assert(isFabricSchema(reference));
  auto [entry, inserted] =
      states_.emplace(reference.artifact.bytes(), VisitState::Active);
  if (inserted)
    return true;
  if (entry->second == VisitState::Active)
    return dependencyError(
        FabricArtifactDependencyFailureReason::CyclicDependency,
        "the exact Fabric dependency closure is cyclic");
  return false;
}

void FabricArtifactDependencyClosureTraversal::abandon(
    const ArtifactRootReference &reference) {
  assert(isFabricSchema(reference));
  auto entry = states_.find(reference.artifact.bytes());
  assert(entry != states_.end() && entry->second == VisitState::Active);
  states_.erase(entry);
}

void FabricArtifactDependencyClosureTraversal::complete(
    const ArtifactRootReference &reference) {
  assert(isFabricSchema(reference));
  auto entry = states_.find(reference.artifact.bytes());
  assert(entry != states_.end() && entry->second == VisitState::Active);
  entry->second = VisitState::Validated;
}

llvm::Error validateFabricArtifactDependencyFramingClosure(
    const ArtifactStore &store, const CanonicalSemanticBytes &canonicalBytes) {
  auto decoded = decodeFabricArtifactEnvelope(canonicalBytes.bytes());
  if (!decoded)
    return wrapCodecError(
        FabricArtifactDependencyFailureReason::MalformedCandidateEnvelope,
        "candidate canonical envelope decode failed", decoded.takeError());

  FabricArtifactDependencyClosureTraversal traversal;
  return validateDecodedFabricArtifact(store, fabricReference(canonicalBytes),
                                       *decoded, traversal);
}

} // namespace loom::fabric::detail
