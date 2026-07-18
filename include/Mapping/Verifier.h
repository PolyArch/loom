#ifndef LOOM_MAPPING_VERIFIER_H
#define LOOM_MAPPING_VERIFIER_H

#include "Mapping/Artifact.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <memory>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace loom::mapping {

namespace detail {
struct ValidatedFabricProjection;
struct ValidatedTechMappingProjection;
class ValidatedTechMappingAccess;
} // namespace detail

enum class MappingErrorCode {
  UnsupportedSchemaVersion,
  WrongMappingProfile,
  ArtifactIdentityMismatch,
  DuplicateEntityId,
  ForeignEntityReference,
  UnresolvedEntityId,
  WrongEntityKind,
  InvalidPortConnection,
  InvalidComputeOccurrence,
  MissingFuImplementation,
  MalformedFuParentLinkage,
  InvalidInstructionContextCapacity,
  PortSignatureMismatch,
  DuplicateEdge,
  MissingSinkDriver,
  MultipleSinkDrivers,
  ActorlessGraphPassthrough,
  EmptyActorGroup,
  CrossGraphActorGroup,
  WrongActorRealizationKind,
  InvalidMemoryRealization,
  IncompleteActorToOpCorrespondence,
  IncompleteBoundaryCorrespondence,
  IncompleteMemoryBoundaryCorrespondence,
  SelectedFuMismatch,
  MemoryOperationMismatch,
  MemoryEncodingMismatch,
  InvalidInternalEdgeWitness,
  MemoryServiceMismatch,
  MemoryAccessIncompatible,
  InvalidConfiguredFunction,
  ConfiguredFunctionMismatch,
  UnaccountedGraphEdge,
  IncompleteGraphCoverage,
  InternalError,
};

class MappingError : public llvm::ErrorInfo<MappingError> {
public:
  static char ID;

  MappingError(MappingErrorCode code, std::string message)
      : code_(code), message_(std::move(message)) {}

  MappingErrorCode code() const { return code_; }
  void log(llvm::raw_ostream &stream) const override;
  std::error_code convertToErrorCode() const override;

private:
  MappingErrorCode code_;
  std::string message_;
};

class ValidatedTechMapping {
public:
  // Rvalue construction intentionally copies so existing borrows stay valid.
  ValidatedTechMapping(const ValidatedTechMapping &) = default;
  ValidatedTechMapping &operator=(const ValidatedTechMapping &) = delete;
  ValidatedTechMapping &operator=(ValidatedTechMapping &&) = delete;

  const ArtifactIdentity &identity() const { return identity_; }
  MappingProfile profile() const { return draft_.header.profile; }
  const MappingDraftHeader &header() const { return draft_.header; }
  llvm::ArrayRef<GraphRef> coveredGraphs() const {
    return draft_.coveredGraphs;
  }
  llvm::ArrayRef<ComputeRealizationDraft> realizations() const {
    return draft_.realizations;
  }
  llvm::ArrayRef<MemoryRealizationDraft> memoryRealizations() const {
    return draft_.memoryRealizations;
  }

private:
  ValidatedTechMapping(
      ArtifactIdentity identity, TechMappingDraft draft,
      std::shared_ptr<const detail::ValidatedFabricProjection> fabricProjection,
      std::shared_ptr<const detail::ValidatedTechMappingProjection>
          mappingProjection)
      : identity_(std::move(identity)), draft_(std::move(draft)),
        fabricProjection_(std::move(fabricProjection)),
        mappingProjection_(std::move(mappingProjection)) {}

  ArtifactIdentity identity_;
  TechMappingDraft draft_;
  std::shared_ptr<const detail::ValidatedFabricProjection> fabricProjection_;
  std::shared_ptr<const detail::ValidatedTechMappingProjection>
      mappingProjection_;

  friend class detail::ValidatedTechMappingAccess;
  friend llvm::Expected<ValidatedTechMapping> validateTechMapping(
      ArtifactIdentity identity, const TechMappingDraft &mapping,
      const DataflowProgramView &dataflow, const FabricHardwareView &fabric);
};

llvm::Expected<ValidatedTechMapping>
validateTechMapping(ArtifactIdentity identity, const TechMappingDraft &mapping,
                    const DataflowProgramView &dataflow,
                    const FabricHardwareView &fabric);

} // namespace loom::mapping

#endif // LOOM_MAPPING_VERIFIER_H
