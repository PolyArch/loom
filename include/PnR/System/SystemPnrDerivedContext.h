#ifndef LOOM_PNR_SYSTEM_SYSTEMPNRDERIVEDCONTEXT_H
#define LOOM_PNR_SYSTEM_SYSTEMPNRDERIVEDCONTEXT_H

#include "Common/Artifact.h"
#include "Common/ArtifactStore.h"
#include "Common/MappingDebugLog.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Fabric/Artifact/FabricSystemRootView.h"
#include "Fabric/Identity/FabricPhysicalTiming.h"
#include "Mapping/Artifact/MappingArtifact.h"
#include "Mapping/Artifact/SystemMappingConstraintSet.h"
#include "PnR/PnrDerivedContext.h"

#include "llvm/Support/Error.h"

#include <cstdint>
#include <memory>

namespace loom::pnr {

class SystemStaticContext;
class SystemActiveContext;

namespace detail {
struct SystemStaticContextStorage;
struct SystemActiveContextStorage;
const SystemStaticContextStorage &
systemStaticContextStorage(const SystemStaticContext &context);
const SystemActiveContextStorage &
systemActiveContextStorage(const SystemActiveContext &context);
} // namespace detail

struct SystemStaticContextStatistics final {
  DerivedContextConstructionStatistics context;
  std::uint64_t accCoreCount = 0;
  std::uint64_t targetClassCount = 0;
  std::uint64_t endpointCount = 0;
  std::uint64_t traversalCount = 0;
  std::uint64_t routingArcCount = 0;
  std::uint64_t instructionUsePatternCount = 0;
  std::uint64_t consistencyUsePatternCount = 0;
};

/// Bounded invocation-local owner for immutable System-only projections. It
/// has no persistent codec and never owns workload or mutable candidate state.
class SystemStaticContext final {
public:
  SystemStaticContext(SystemStaticContext &&) noexcept = default;
  SystemStaticContext &operator=(SystemStaticContext &&) noexcept = default;
  SystemStaticContext(const SystemStaticContext &) = delete;
  SystemStaticContext &operator=(const SystemStaticContext &) = delete;
  ~SystemStaticContext() = default;

  const ArtifactIdentity &systemIdentity() const;
  const SystemStaticContextStatistics &statistics() const;

private:
  explicit SystemStaticContext(
      std::shared_ptr<const detail::SystemStaticContextStorage> storage)
      : storage_(std::move(storage)) {}

  std::shared_ptr<const detail::SystemStaticContextStorage> storage_;

  friend llvm::Expected<SystemStaticContext>
  buildSystemStaticContext(const ::loom::fabric::FabricSystemRootView &);
  friend llvm::Error
  revalidateSystemStaticContext(const SystemStaticContext &,
                                const ::loom::fabric::FabricSystemRootView &);
  friend const detail::SystemStaticContextStorage &
  detail::systemStaticContextStorage(const SystemStaticContext &);
};

llvm::Expected<SystemStaticContext>
buildSystemStaticContext(const ::loom::fabric::FabricSystemRootView &system);

llvm::Error revalidateSystemStaticContext(
    const SystemStaticContext &context,
    const ::loom::fabric::FabricSystemRootView &system);

void emitSystemStaticContextStatistics(const SystemStaticContext &context,
                                       mapping_debug::Stage stage,
                                       std::uint64_t hits,
                                       std::uint64_t misses);

struct SystemActiveContextStatistics final {
  DerivedContextConstructionStatistics context;
  std::uint64_t spatialMappingCount = 0;
  std::uint64_t coveredGraphCount = 0;
  std::uint64_t routeProgressObligationCount = 0;
  std::uint64_t schedulePressureCount = 0;
  std::uint64_t recurrenceProjectionCount = 0;
  std::uint64_t timingProfileCount = 0;
  std::uint64_t techMappingImportRequests = 0;
  std::uint64_t techMappingImportHits = 0;
  std::uint64_t techMappingImportMisses = 0;
};

/// Bounded immutable owner for one exact System workload dependency tuple.
/// Candidate selections, transactions, scratch, PRNG, and budgets remain
/// outside this context.
class SystemActiveContext final {
public:
  SystemActiveContext(SystemActiveContext &&) noexcept = default;
  SystemActiveContext &operator=(SystemActiveContext &&) noexcept = default;
  SystemActiveContext(const SystemActiveContext &) = delete;
  SystemActiveContext &operator=(const SystemActiveContext &) = delete;
  ~SystemActiveContext() = default;

  const ArtifactIdentity &dataflowIdentity() const;
  const ArtifactIdentity &systemIdentity() const;
  const ArtifactIdentity &constraintIdentity() const;
  llvm::ArrayRef<ArtifactRootReference> spatialMappings() const;
  const SystemActiveContextStatistics &statistics() const;

private:
  explicit SystemActiveContext(
      std::shared_ptr<const detail::SystemActiveContextStorage> storage)
      : storage_(std::move(storage)) {}

  std::shared_ptr<const detail::SystemActiveContextStorage> storage_;

  friend llvm::Expected<SystemActiveContext> buildSystemActiveContext(
      const SystemStaticContext &,
      const ::dataflow::CanonicalDataflowProgramView &,
      const ::loom::fabric::FabricSystemRootView &,
      llvm::ArrayRef<::loom::fabric::FabricPhysicalTimingProfileView>,
      const ::loom::mapping::FinalizedSystemMappingConstraintSet &,
      llvm::ArrayRef<ArtifactRootReference>, const ArtifactStore &);
  friend llvm::Error revalidateSystemActiveContext(
      const SystemActiveContext &, const SystemStaticContext &,
      const ::dataflow::CanonicalDataflowProgramView &,
      const ::loom::fabric::FabricSystemRootView &,
      llvm::ArrayRef<::loom::fabric::FabricPhysicalTimingProfileView>,
      const ::loom::mapping::FinalizedSystemMappingConstraintSet &,
      llvm::ArrayRef<ArtifactRootReference>);
  friend const detail::SystemActiveContextStorage &
  detail::systemActiveContextStorage(const SystemActiveContext &);
};

llvm::Expected<SystemActiveContext> buildSystemActiveContext(
    const SystemStaticContext &staticContext,
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::fabric::FabricSystemRootView &system,
    llvm::ArrayRef<::loom::fabric::FabricPhysicalTimingProfileView>
        physicalTimingProfiles,
    const ::loom::mapping::FinalizedSystemMappingConstraintSet &constraints,
    llvm::ArrayRef<ArtifactRootReference> spatialMappings,
    const ArtifactStore &store);

llvm::Error revalidateSystemActiveContext(
    const SystemActiveContext &context,
    const SystemStaticContext &staticContext,
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::fabric::FabricSystemRootView &system,
    llvm::ArrayRef<::loom::fabric::FabricPhysicalTimingProfileView>
        physicalTimingProfiles,
    const ::loom::mapping::FinalizedSystemMappingConstraintSet &constraints,
    llvm::ArrayRef<ArtifactRootReference> spatialMappings);

void emitSystemActiveContextStatistics(const SystemActiveContext &context,
                                       mapping_debug::Stage stage,
                                       std::uint64_t hits,
                                       std::uint64_t misses);

} // namespace loom::pnr

#endif // LOOM_PNR_SYSTEM_SYSTEMPNRDERIVEDCONTEXT_H
