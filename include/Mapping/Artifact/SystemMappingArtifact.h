#ifndef LOOM_MAPPING_ARTIFACT_SYSTEMMAPPINGARTIFACT_H
#define LOOM_MAPPING_ARTIFACT_SYSTEMMAPPINGARTIFACT_H

#include "Common/Artifact.h"
#include "Common/ArtifactStore.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Fabric/Artifact/FabricSystemRootView.h"
#include "Fabric/IR/UsePatternValue.h"
#include "Mapping/Artifact/MappingArtifact.h"
#include "Mapping/Artifact/SystemMappingIdentity.h"
#include "Mapping/Artifact/SystemPresburger.h"
#include "Mapping/IR/MappingOps.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace loom::mapping {

namespace detail {
class SystemMappingImportSessionState;
}

enum class SystemMappingImportVerificationDomain : std::uint8_t {
  SourceInvocation,
  IndependentReplay,
};

struct SystemMappingImportSessionStatistics final {
  std::uint64_t importRequests = 0;
  std::uint64_t cacheHits = 0;
  std::uint64_t cacheMisses = 0;
  std::uint64_t uniqueConstructions = 0;
  std::uint64_t uncachedConstructions = 0;
  std::uint64_t bytesRead = 0;
  std::uint64_t constructionNanoseconds = 0;
  std::uint64_t deterministicWork = 0;
  std::uint64_t retainedBytes = 0;
  std::uint64_t entryCount = 0;
};

class SystemMappingImportSession final {
public:
  SystemMappingImportSession(const ArtifactStore &store,
                             std::size_t entryLimit);
  ~SystemMappingImportSession();

  SystemMappingImportSession(const SystemMappingImportSession &) = delete;
  SystemMappingImportSession &
  operator=(const SystemMappingImportSession &) = delete;
  SystemMappingImportSession(SystemMappingImportSession &&) = delete;
  SystemMappingImportSession &operator=(SystemMappingImportSession &&) = delete;

  SystemMappingImportSessionStatistics statistics() const;

private:
  std::unique_ptr<detail::SystemMappingImportSessionState> state_;
  detail::SystemMappingImportSessionState *previous_ = nullptr;
};

void emitSystemMappingImportSessionStatistics(
    SystemMappingImportVerificationDomain domain,
    const SystemMappingImportSessionStatistics &statistics);

class SystemMappingConstraintSetView;
struct SystemMappingClosureProjection;

template <typename Target> struct SystemPresburgerClauseView final {
  std::vector<SystemPresburgerCell> cells;
  Target target;
};

struct SystemThreadExecutionBindingView final {
  ::dataflow::RootThreadLaunchRef key;
  std::vector<SystemPresburgerClauseView<::loom::fabric::AccCoreOccurrenceRef>>
      clauses;
  std::optional<::loom::fabric::AccCoreOccurrenceRef> defaultTarget;
};

struct SystemGraphExecutionBindingView final {
  ::dataflow::RootedGraphLaunchRef key;
  std::vector<SystemPresburgerClauseView<ArtifactRootReference>> clauses;
  std::optional<ArtifactRootReference> defaultTarget;
};

/// Strictly reconstructed execution portion of one mapping.system root.
/// Service and ResourceUse closure are intentionally not represented here.
class SystemExecutionBindingView final {
public:
  llvm::ArrayRef<::dataflow::RootThreadLaunchRef> rootThreadLaunches() const {
    return rootThreadLaunches_;
  }
  llvm::ArrayRef<ArtifactRootReference> spatialMappingImports() const {
    return spatialMappingImports_;
  }
  llvm::ArrayRef<SystemThreadExecutionBindingView> threadBindings() const {
    return threadBindings_;
  }
  llvm::ArrayRef<SystemGraphExecutionBindingView> graphBindings() const {
    return graphBindings_;
  }

private:
  SystemExecutionBindingView(
      std::vector<::dataflow::RootThreadLaunchRef> rootThreadLaunches,
      std::vector<ArtifactRootReference> spatialMappingImports,
      std::vector<SystemThreadExecutionBindingView> threadBindings,
      std::vector<SystemGraphExecutionBindingView> graphBindings)
      : rootThreadLaunches_(std::move(rootThreadLaunches)),
        spatialMappingImports_(std::move(spatialMappingImports)),
        threadBindings_(std::move(threadBindings)),
        graphBindings_(std::move(graphBindings)) {}

  std::vector<::dataflow::RootThreadLaunchRef> rootThreadLaunches_;
  std::vector<ArtifactRootReference> spatialMappingImports_;
  std::vector<SystemThreadExecutionBindingView> threadBindings_;
  std::vector<SystemGraphExecutionBindingView> graphBindings_;

  friend llvm::Expected<SystemExecutionBindingView>
  strictImportSystemExecutionBindings(
      const CanonicalSemanticBytes &,
      const ::dataflow::CanonicalDataflowProgramView &,
      const ::loom::fabric::FabricSystemRootView &, const ArtifactStore &,
      const SpatialMappingImportContext *);
};

struct SystemMemoryRegionElementView final {
  ::dataflow::LogicalMemoryRootOrViewRef logicalMemory;
  SpatialMemoryIntervalView interval;
  ::loom::fabric::FabricMemoryServiceRegionRef serviceRegion;
  std::vector<::loom::fabric::SystemServiceTransformRef> transformPath;
};

struct SystemConsistencyElementView final {
  ::dataflow::FenceActorFamilyRef fence;
  ::loom::fabric::MemoryConsistencyDomainRef consistencyDomain;
};

using SystemServicePlanElementView =
    std::variant<CanonicalServiceLegKey, SystemMemoryRegionElementView,
                 SystemConsistencyElementView>;

struct SystemMemoryExposureView final {
  ::dataflow::MemoryExposureRef exposure;
  ::loom::fabric::SubordinateEndpointRef terminal;
};

struct SystemMemoryRegionTargetView final {
  SystemMemoryRegionElementView element;
  std::vector<SystemMemoryExposureView> exposures;
};

struct SystemTransferRouteNodeView final {
  std::uint64_t ordinal = 0;
  std::uint64_t parentOrdinal = 0;
  ::loom::fabric::FabricPhysicalTraversalRef incomingTraversal;
};

struct SystemTransferRouteSinkView final {
  SystemTransferTerminalKey terminal;
  std::uint64_t nodeOrdinal = 0;
};

struct SystemTransferLegView final {
  CanonicalServiceLegKey leg;
  ::loom::fabric::FabricTransportEndpointRef rootEndpoint;
  std::vector<SystemTransferRouteNodeView> nodes;
  std::vector<SystemTransferRouteSinkView> sinks;
};

struct SystemServicePlanView final {
  std::uint64_t ordinal = 0;
  std::vector<SystemTransferLegView> transferLegs;
  std::vector<SystemMemoryRegionTargetView> memoryTargets;
  std::vector<SystemConsistencyElementView> consistencyTargets;
};

struct SystemServicePlanSelectionView final {
  ServicePlanSelectionKey key;
  std::vector<SystemPresburgerClauseView<std::uint64_t>> clauses;
  std::optional<std::uint64_t> defaultPlanOrdinal;
};

struct SystemServiceRealizationView final {
  SystemServiceObligationKey key;
  std::vector<SystemServicePlanView> plans;
  std::vector<SystemServicePlanSelectionView> selections;
};

struct SystemInstructionResourceOwnerView final {
  ::dataflow::RootThreadLaunchRef root;
  ::loom::fabric::InstructionCoreContextRef instructionContext;
};

struct SystemServicePlanResourceOwnerView final {
  SystemServiceObligationKey service;
  std::uint64_t planOrdinal = 0;
  SystemServicePlanElementView element;
};

using SystemResourceOwnerView =
    std::variant<SystemInstructionResourceOwnerView,
                 SystemServicePlanResourceOwnerView>;

struct SystemEventPointView final {
  ::dataflow::EventFamilyKey event;
  std::optional<std::vector<std::uint8_t>> guaranteedOffset;
};

struct SystemRelativeActivationView final {
  SystemEventPointView trigger;
  std::vector<SystemEventPointView> release;
};

struct SystemResourceUseView final {
  SystemResourceOwnerView owner;
  ::loom::fabric::FabricUsePatternRef useSite;
  SystemRelativeActivationView activation;
  std::vector<::fabric::UsePatternValue> parameters;
  std::vector<::fabric::UsePatternValue> sharingAssignments;
};

class SystemMappingView final {
public:
  const ArtifactIdentity &identity() const { return identity_; }
  const ArtifactIdentity &dataflowIdentity() const { return dataflowIdentity_; }
  const ArtifactIdentity &fabricIdentity() const { return fabricIdentity_; }
  const SystemExecutionBindingView &executionBindings() const {
    return executionBindings_;
  }
  llvm::ArrayRef<SystemServiceRealizationView> serviceRealizations() const {
    return serviceRealizations_;
  }
  llvm::ArrayRef<SystemResourceUseView> resourceUses() const {
    return resourceUses_;
  }

private:
  SystemMappingView(ArtifactIdentity identity,
                    ArtifactIdentity dataflowIdentity,
                    ArtifactIdentity fabricIdentity,
                    SystemExecutionBindingView executionBindings,
                    std::vector<SystemServiceRealizationView> services,
                    std::vector<SystemResourceUseView> resourceUses)
      : identity_(std::move(identity)),
        dataflowIdentity_(std::move(dataflowIdentity)),
        fabricIdentity_(std::move(fabricIdentity)),
        executionBindings_(std::move(executionBindings)),
        serviceRealizations_(std::move(services)),
        resourceUses_(std::move(resourceUses)) {}

  ArtifactIdentity identity_;
  ArtifactIdentity dataflowIdentity_;
  ArtifactIdentity fabricIdentity_;
  SystemExecutionBindingView executionBindings_;
  std::vector<SystemServiceRealizationView> serviceRealizations_;
  std::vector<SystemResourceUseView> resourceUses_;

  friend class FinalizedSystemMapping;
  friend llvm::Expected<SystemMappingView> importSystemMappingView(
      const ArtifactIdentity &, ::mapping::SystemOp,
      const ::dataflow::CanonicalDataflowProgramView &,
      const ::loom::fabric::FabricSystemRootView &, const ArtifactStore &,
      const SpatialMappingImportContext *,
      std::shared_ptr<const SystemMappingClosureProjection> *);
};

class FinalizedSystemMapping final {
public:
  const ArtifactRootReference &reference() const { return reference_; }
  const CanonicalSemanticBytes &canonicalBytes() const {
    return canonicalBytes_;
  }
  const SystemMappingView &view() const { return *view_; }
  const SystemMappingClosureProjection &verifiedClosure() const;

private:
  FinalizedSystemMapping(
      ArtifactRootReference reference, CanonicalSemanticBytes canonicalBytes,
      SystemMappingView view,
      std::shared_ptr<const SystemMappingClosureProjection> verifiedClosure)
      : reference_(std::move(reference)),
        canonicalBytes_(std::move(canonicalBytes)),
        view_(std::make_shared<const SystemMappingView>(std::move(view))),
        verifiedClosure_(std::move(verifiedClosure)) {}

  ArtifactRootReference reference_;
  CanonicalSemanticBytes canonicalBytes_;
  std::shared_ptr<const SystemMappingView> view_;
  std::shared_ptr<const SystemMappingClosureProjection> verifiedClosure_;

  friend llvm::Expected<FinalizedSystemMapping> finalizeSystemMapping(
      ::mapping::SystemOp, const ::dataflow::CanonicalDataflowProgramView &,
      const ::loom::fabric::FabricSystemRootView &,
      const SystemMappingConstraintSetView &, const ArtifactStore &,
      const SpatialMappingImportContext *);
  friend llvm::Expected<FinalizedSystemMapping>
  importSystemMapping(const ArtifactRootReference &, const ArtifactStore &);
};

llvm::Expected<CanonicalSemanticBytes>
writeCanonicalSystemMappingAssembly(::mapping::SystemOp root);

struct VerifiedSystemMappingBase final {};

enum class SystemMappingClosureFindingKind : std::uint8_t {
  InvalidClosure,
  HardProgressViolation,
};

struct RejectedSystemMappingBase final {
  SystemMappingClosureFindingKind finding;
  std::string diagnostic;
};

enum class SystemMappingIncompleteReason : std::uint8_t {
  Unsupported,
  ProofNotEstablished,
};

class SystemMappingIncompleteError final
    : public llvm::ErrorInfo<SystemMappingIncompleteError> {
public:
  static char ID;

  SystemMappingIncompleteError(SystemMappingIncompleteReason reason,
                               std::string diagnostic);

  SystemMappingIncompleteReason reason() const { return reason_; }
  llvm::StringRef diagnostic() const { return diagnostic_; }

  void log(llvm::raw_ostream &stream) const override;
  std::error_code convertToErrorCode() const override;

private:
  SystemMappingIncompleteReason reason_;
  std::string diagnostic_;
};

class SystemMappingRejectedError final
    : public llvm::ErrorInfo<SystemMappingRejectedError> {
public:
  static char ID;

  SystemMappingRejectedError(SystemMappingClosureFindingKind finding,
                             std::string diagnostic);

  SystemMappingClosureFindingKind finding() const { return finding_; }
  llvm::StringRef diagnostic() const { return diagnostic_; }

  void log(llvm::raw_ostream &stream) const override;
  std::error_code convertToErrorCode() const override;

private:
  SystemMappingClosureFindingKind finding_;
  std::string diagnostic_;
};

struct IncompleteSystemMappingBase final {
  SystemMappingIncompleteReason reason;
  std::string diagnostic;
};

struct InternalSystemMappingBaseError final {
  std::string diagnostic;
};

using SystemMappingBaseVerification =
    std::variant<VerifiedSystemMappingBase, RejectedSystemMappingBase,
                 IncompleteSystemMappingBase, InternalSystemMappingBaseError>;

/// Strictly parses, semantically validates, canonically re-emits, and adopts
/// the execution records. It never publishes an incomplete SystemMapping.
llvm::Expected<SystemExecutionBindingView> strictImportSystemExecutionBindings(
    const CanonicalSemanticBytes &bytes,
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::fabric::FabricSystemRootView &fabric,
    const ArtifactStore &store,
    const SpatialMappingImportContext *spatialMappings = nullptr);

SystemMappingBaseVerification verifySystemMappingBase(
    ::mapping::SystemOp source,
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::fabric::FabricSystemRootView &fabric,
    const ArtifactStore &store);

llvm::Expected<FinalizedSystemMapping> finalizeSystemMapping(
    ::mapping::SystemOp source,
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::fabric::FabricSystemRootView &fabric,
    const SystemMappingConstraintSetView &constraints,
    const ArtifactStore &store,
    const SpatialMappingImportContext *spatialMappings = nullptr);

llvm::Expected<FinalizedSystemMapping>
importSystemMapping(const ArtifactRootReference &reference,
                    const ArtifactStore &store);

} // namespace loom::mapping

#endif // LOOM_MAPPING_ARTIFACT_SYSTEMMAPPINGARTIFACT_H
