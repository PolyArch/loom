#ifndef LOOM_FABRIC_ARTIFACT_FABRICSYSTEMROOTVIEW_H
#define LOOM_FABRIC_ARTIFACT_FABRICSYSTEMROOTVIEW_H

#include "Fabric/Artifact/FabricHardwareDomainContracts.h"
#include "Fabric/Artifact/FabricSystemContracts.h"
#include "Fabric/Identity/FabricRefImport.h"

namespace loom::fabric {

struct FabricSpatialAttachmentRecordView {
  FabricImportedModuleBoundaryEndpointRef moduleEndpoint;
  FabricSpatialAttachmentEndpointRef spatialEndpoint;

  friend bool operator==(const FabricSpatialAttachmentRecordView &lhs,
                         const FabricSpatialAttachmentRecordView &rhs) {
    return lhs.moduleEndpoint == rhs.moduleEndpoint &&
           lhs.spatialEndpoint == rhs.spatialEndpoint;
  }
};

/// Zero-copy typed refinement of one complete immutable System root view.
/// It adds no storage or relation authority of its own.
class FabricSystemRootView final {
public:
  const FabricArtifactView &artifact() const { return artifact_; }

  const InstructionCoreArchitecturalContract *
  instructionCoreArchitecture(HostCoreOccurrenceRef core) const;
  const InstructionCoreArchitecturalContract *
  instructionCoreArchitecture(InstructionCoreContextRef core) const;
  const InstructionCoreMicroarchitecturalRealization *
  instructionCoreMicroarchitecture(HostCoreOccurrenceRef core) const;
  const InstructionCoreMicroarchitecturalRealization *
  instructionCoreMicroarchitecture(InstructionCoreContextRef core) const;

  /// Exact imported Module target selected by one AccCore's SpatialCore field.
  /// This is a sealed projection of the owner field, not an attachment-derived
  /// inference. A wrong-kind or unknown occurrence has no target.
  std::optional<FabricImportedModuleTargetRef>
  spatialCoreTarget(AccCoreOccurrenceRef core) const;

  llvm::ArrayRef<FabricSpatialAttachmentRecordView> spatialAttachments() const;
  llvm::ArrayRef<HardwareDomainRef> hardwareDomains() const;
  const HardwareDomainContractRecord *
  hardwareDomainContract(HardwareDomainRef domain) const;
  llvm::ArrayRef<FabricInventoryOwnerRef>
  hardwareDomainMembers(HardwareDomainRef domain) const;
  llvm::ArrayRef<SystemTransportResourceRef> transportResources() const;
  llvm::ArrayRef<FabricTransferPatternRef>
  transferPatterns(SystemTransportResourceRef resource) const;
  const SystemTransferPatternRecord *
  transferPattern(FabricTransferPatternRef pattern) const;
  const ClockCrossingContractRecord *
  clockCrossing(SystemTransportResourceRef resource) const;

private:
  explicit FabricSystemRootView(FabricArtifactView artifact)
      : artifact_(std::move(artifact)) {}

  FabricArtifactView artifact_;

  friend llvm::Expected<FabricSystemRootView>
  requireSystemRoot(const FabricArtifactView &view);
};

/// Refines exactly one complete System root. Other root kinds fail with the
/// existing typed wrong-root-kind diagnostic.
llvm::Expected<FabricSystemRootView>
requireSystemRoot(const FabricArtifactView &view);

} // namespace loom::fabric

#endif // LOOM_FABRIC_ARTIFACT_FABRICSYSTEMROOTVIEW_H
