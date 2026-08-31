#ifndef LOOM_PNR_SPATIALMAPPINGWARMSEED_H
#define LOOM_PNR_SPATIALMAPPINGWARMSEED_H

#include "Mapping/Artifact/MappingArtifact.h"
#include "PnR/SpatialCandidateState.h"

#include "llvm/Support/Error.h"

#include <cstdint>
#include <string>
#include <system_error>

namespace loom::pnr {

enum class SpatialMappingWarmSeedFailureKind : std::uint8_t {
  OwnerMismatch,
  SelectionAbsent,
  SelectionAmbiguous,
  RelationInfeasible,
  RouteProjectionInvalid,
  TagProjectionInvalid,
  CandidateVerificationFailed,
  SelectionMismatch,
};

class SpatialMappingWarmSeedFailure final
    : public llvm::ErrorInfo<SpatialMappingWarmSeedFailure> {
public:
  static char ID;

  SpatialMappingWarmSeedFailure(SpatialMappingWarmSeedFailureKind kind,
                                std::string message)
      : kind_(kind), message_(std::move(message)) {}

  SpatialMappingWarmSeedFailureKind kind() const { return kind_; }
  void log(llvm::raw_ostream &stream) const override;
  std::error_code convertToErrorCode() const override;

private:
  SpatialMappingWarmSeedFailureKind kind_;
  std::string message_;
};

/// Exact accounting for the Mapping selections retained in a same-Fabric
/// warm seed. Private terminal attachments erased by a register-FIFO Mapping
/// disposition are completed canonically by the frozen binding relation and
/// counted separately; they are never attributed to the parent Mapping.
struct SpatialMappingWarmSeedAccounting final {
  std::uint64_t computeBindings = 0;
  std::uint64_t memoryBindings = 0;
  std::uint64_t memoryOperationPlans = 0;
  std::uint64_t logicalMemoryBindings = 0;
  std::uint64_t memoryUseDispatches = 0;
  std::uint64_t memoryExposureSelections = 0;
  std::uint64_t portAttachments = 0;
  std::uint64_t graphBoundaryAttachments = 0;
  std::uint64_t canonicalPrivatePortAttachments = 0;
  std::uint64_t canonicalPrivateGraphBoundaryAttachments = 0;
  std::uint64_t registerFifoTransfers = 0;
  std::uint64_t routeTrees = 0;
  std::uint64_t routeArcs = 0;
  std::uint64_t physicalTagSegments = 0;
  std::uint64_t relationAssignmentAttempts = 0;
};

/// Invocation-local, non-artifact warm seed derived from one strictly imported
/// SpatialMapping and one immutable same-D/T/F problem. The compact snapshot is
/// a rebuildable cache; the parent Mapping remains the semantic owner.
class VerifiedSpatialMappingWarmSeed final {
public:
  const ArtifactIdentity &parentMappingIdentity() const {
    return parentMappingIdentity_;
  }
  const SpatialFullyRoutedSnapshot &snapshot() const { return snapshot_; }
  const SpatialMappingWarmSeedAccounting &accounting() const {
    return accounting_;
  }
  llvm::Expected<SpatialCandidateStateHandle> materializeCandidate() const {
    return SpatialCandidateState::materializeFullyRouted(snapshot_);
  }

private:
  VerifiedSpatialMappingWarmSeed(ArtifactIdentity parentMappingIdentity,
                                 SpatialFullyRoutedSnapshot snapshot,
                                 SpatialMappingWarmSeedAccounting accounting)
      : parentMappingIdentity_(std::move(parentMappingIdentity)),
        snapshot_(std::move(snapshot)), accounting_(accounting) {}

  ArtifactIdentity parentMappingIdentity_;
  SpatialFullyRoutedSnapshot snapshot_;
  SpatialMappingWarmSeedAccounting accounting_;

  friend llvm::Expected<VerifiedSpatialMappingWarmSeed>
  projectFinalizedSpatialMappingWarmSeed(
      const ::loom::mapping::FinalizedSpatialMapping &parent,
      FrozenSpatialPnrProblemHandle problem);
};

/// Reconstructs every persistent candidate selection from an exact finalized
/// parent Mapping under a same-D/T/F frozen problem. All references must
/// resolve uniquely. The resulting ordinary CandidateState and snapshot are
/// cold verified and compared against the complete persistent selection
/// projection; any missing inverse projection fails closed.
llvm::Expected<VerifiedSpatialMappingWarmSeed>
projectFinalizedSpatialMappingWarmSeed(
    const ::loom::mapping::FinalizedSpatialMapping &parent,
    FrozenSpatialPnrProblemHandle problem);

} // namespace loom::pnr

#endif // LOOM_PNR_SPATIALMAPPINGWARMSEED_H
