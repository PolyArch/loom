#ifndef LOOM_LIB_FABRIC_ARTIFACT_FABRICARTIFACTDEPENDENCYCLOSUREINTERNAL_H
#define LOOM_LIB_FABRIC_ARTIFACT_FABRICARTIFACTDEPENDENCYCLOSUREINTERNAL_H

#include "Common/ArtifactStore.h"

#include "llvm/Support/Error.h"

#include <map>
#include <string>
#include <system_error>

namespace loom::fabric::detail {

enum class FabricArtifactDependencyFailureReason {
  MalformedCandidateEnvelope,
  InvalidDependencyEnvelope,
  InvalidDependencyRoles,
  ForeignDependency,
  ImplementationInputOwnerUnavailable,
  WrongDependencyRootKind,
  CyclicDependency,
};

class FabricArtifactDependencyError final
    : public llvm::ErrorInfo<FabricArtifactDependencyError> {
public:
  static char ID;

  FabricArtifactDependencyError(FabricArtifactDependencyFailureReason reason,
                                std::string message);

  FabricArtifactDependencyFailureReason reason() const { return reason_; }

  void log(llvm::raw_ostream &stream) const override;
  std::error_code convertToErrorCode() const override;

private:
  FabricArtifactDependencyFailureReason reason_;
  std::string message_;
};

class FabricArtifactDependencyClosureTraversal {
public:
  llvm::Expected<bool> enter(const ArtifactRootReference &reference);
  void abandon(const ArtifactRootReference &reference);
  void complete(const ArtifactRootReference &reference);

private:
  enum class VisitState {
    Active,
    Validated,
  };

  std::map<ArtifactIdentity::Storage, VisitState> states_;
};

/// Validates only the exact dependency-table framing reachable from one
/// canonical loom.fabric envelope. Same-family envelopes, role/root-kind
/// legality, store resolution, and cycles are covered. Payload semantics,
/// local targets, dependency use, and publication remain the finalizer's
/// strict-import boundary and are deliberately not implied by success here.
llvm::Error validateFabricArtifactDependencyFramingClosure(
    const ArtifactStore &store, const CanonicalSemanticBytes &canonicalBytes);

} // namespace loom::fabric::detail

#endif // LOOM_LIB_FABRIC_ARTIFACT_FABRICARTIFACTDEPENDENCYCLOSUREINTERNAL_H
