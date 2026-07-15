#ifndef LOOM_MAPPING_STRUCTUREVALIDATOR_H
#define LOOM_MAPPING_STRUCTUREVALIDATOR_H

#include "Mapping/Artifact.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <string>
#include <system_error>
#include <utility>

namespace loom::mapping {

enum class MappingErrorCode {
  UnsupportedSchemaVersion,
  InvalidArtifactIdentity,
  WrongMappingProfile,
  ArtifactIdentityMismatch,
  DuplicateEntityId,
  ForeignEntityReference,
  UnresolvedEntityId,
  WrongEntityKind,
  InvalidPortConnection,
  PortSignatureMismatch,
  DuplicateEdge,
  MissingSinkDriver,
  MultipleSinkDrivers,
  ActorlessGraphPassthrough,
  EmptyActorGroup,
  CrossGraphActorGroup,
  IncompleteActorToOpCorrespondence,
  IncompleteBoundaryCorrespondence,
  SelectedFuMismatch,
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

// This result establishes structural consistency only. Configured-function
// legality remains owned by the later Fabric projector and verifier.
class ValidatedTechMappingStructure {
public:
  MappingProfile profile() const { return draft_.header.profile; }
  const MappingDraftHeader &header() const { return draft_.header; }
  llvm::ArrayRef<GraphRef> coveredGraphs() const {
    return draft_.coveredGraphs;
  }
  llvm::ArrayRef<StructuralRealizationDraft> realizations() const {
    return draft_.realizations;
  }

private:
  explicit ValidatedTechMappingStructure(TechMappingDraft draft)
      : draft_(std::move(draft)) {}

  TechMappingDraft draft_;

  friend llvm::Expected<ValidatedTechMappingStructure>
  validateTechMappingStructure(const TechMappingDraft &mapping,
                               const DataflowProgramView &dataflow,
                               const FabricHardwareView &fabric);
};

llvm::Expected<ValidatedTechMappingStructure>
validateTechMappingStructure(const TechMappingDraft &mapping,
                             const DataflowProgramView &dataflow,
                             const FabricHardwareView &fabric);

} // namespace loom::mapping

#endif // LOOM_MAPPING_STRUCTUREVALIDATOR_H
