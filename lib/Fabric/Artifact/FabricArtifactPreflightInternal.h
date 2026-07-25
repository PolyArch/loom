#ifndef LOOM_LIB_FABRIC_ARTIFACT_FABRICARTIFACTPREFLIGHTINTERNAL_H
#define LOOM_LIB_FABRIC_ARTIFACT_FABRICARTIFACTPREFLIGHTINTERNAL_H

#include "Common/ArtifactStore.h"

#include "llvm/Support/Error.h"

#include <map>
#include <string>
#include <system_error>

namespace loom::fabric::detail {

enum class FabricArtifactGateFailureKind {
  Invalid,
  Unsupported,
};

enum class FabricArtifactGateReason {
  MalformedCandidateEnvelope,
  InvalidDependencyEnvelope,
  InvalidDependencyRoles,
  DependencyImporterUnavailable,
  WrongDependencyRootKind,
  CyclicDependency,
};

class FabricArtifactGateError final
    : public llvm::ErrorInfo<FabricArtifactGateError> {
public:
  static char ID;

  FabricArtifactGateError(FabricArtifactGateFailureKind kind,
                          FabricArtifactGateReason reason, std::string message);

  FabricArtifactGateFailureKind kind() const { return kind_; }
  FabricArtifactGateReason reason() const { return reason_; }

  void log(llvm::raw_ostream &stream) const override;
  std::error_code convertToErrorCode() const override;

private:
  FabricArtifactGateFailureKind kind_;
  FabricArtifactGateReason reason_;
  std::string message_;
};

enum class FabricArtifactPreflightBlocker {
  ImplementationInputRootKind,
  DependencyUseDecoder,
};

struct FabricArtifactDependencyPreflightIncomplete {
  FabricArtifactPreflightBlocker blocker;
};

class FabricArtifactClosureTraversal {
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

llvm::Expected<FabricArtifactDependencyPreflightIncomplete>
preflightCanonicalFabricArtifactDependencies(
    ArtifactStore &store, const CanonicalSemanticBytes &canonicalBytes);

} // namespace loom::fabric::detail

#endif // LOOM_LIB_FABRIC_ARTIFACT_FABRICARTIFACTPREFLIGHTINTERNAL_H
