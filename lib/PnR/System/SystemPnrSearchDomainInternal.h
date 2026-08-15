#ifndef LOOM_LIB_PNR_SYSTEM_SYSTEMPNRSEARCHDOMAININTERNAL_H
#define LOOM_LIB_PNR_SYSTEM_SYSTEMPNRSEARCHDOMAININTERNAL_H

#include "Mapping/Artifact/MappingArtifact.h"
#include "Mapping/Artifact/MappingProgressProjection.h"
#include "PnR/FrozenConstraintIndex.h"
#include "PnR/SpatialRecurrenceTiming.h"
#include "PnR/System/SystemPnrProblem.h"
#include "PnR/System/SystemPnrSearchDomain.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <cstdint>
#include <variant>
#include <vector>

namespace loom::pnr::detail {

template <typename Ref>
bool systemConstraintAllows(
    const SystemFrozenConstraintIndex &constraints,
    ::mapping::SystemConstraintProjection projection,
    const ::loom::mapping::SystemConstraintSubject &subject, const Ref &value) {
  const auto restricted =
      constraints.shard(projection).restrictedDomain(subject);
  if (!restricted)
    return true;
  for (const ::loom::mapping::SystemConstraintDomainValue &candidate :
       *restricted) {
    const auto *typed = std::get_if<Ref>(&candidate);
    if (typed && *typed == value)
      return true;
  }
  return false;
}

template <typename Ref>
void applySystemConstraintRestriction(
    std::vector<Ref> &values, const SystemFrozenConstraintIndex &constraints,
    ::mapping::SystemConstraintProjection projection,
    const ::loom::mapping::SystemConstraintSubject &subject) {
  values.erase(std::remove_if(values.begin(), values.end(),
                              [&](const Ref &value) {
                                return !systemConstraintAllows(
                                    constraints, projection, subject, value);
                              }),
               values.end());
}

struct CanonicalSystemPartitionBinding final {
  SystemSearchBindingKey key;
  std::vector<::loom::mapping::SystemPresburgerCell> cells;
};

struct SpatialCatalogGraphProgress final {
  ::dataflow::GraphRef graph;
  std::vector<::loom::mapping::MappingRouteProgressObligationProjection>
      routeObligations;
};

struct SpatialCatalogEntry final {
  ArtifactRootReference reference;
  ::loom::mapping::FinalizedSpatialMapping mapping;
  std::uint64_t moduleDependencyOrdinal = 0;
  std::vector<::dataflow::GraphRef> covers;
  std::vector<SpatialCatalogGraphProgress> graphProgress;
  std::vector<std::uint64_t> graphStaticSchedulePressures;
  std::vector<SpatialRecurrenceTimingProjection> graphRecurrenceTimings;
  std::uint64_t worstRouteArrivalDelayQuanta = 0;
  std::uint64_t totalRouteNegativeSlackQuanta = 0;
  ComponentViewDigest::Storage physicalTimingProfileDigest{};
  ::loom::fabric::FabricPhysicalTimingProfileKind physicalTimingProfileKind =
      ::loom::fabric::FabricPhysicalTimingProfileKind::NormalizedHeuristic;
};

llvm::Expected<std::vector<SpatialCatalogEntry>>
importSpatialCatalog(llvm::ArrayRef<ArtifactRootReference> references,
                     const ::dataflow::CanonicalDataflowProgramView &dataflow,
                     const ::loom::fabric::FabricSystemRootView &system,
                     const ArtifactStore &store);

std::vector<::loom::fabric::AccCoreOccurrenceRef>
canonicalSystemAccCores(const ::loom::fabric::FabricSystemRootView &system);

llvm::Error validateSystemBindingDomains(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::fabric::FabricSystemRootView &fabric,
    llvm::ArrayRef<SystemSearchBindingDomain> bindings,
    const SystemFrozenConstraintIndex &constraints,
    llvm::ArrayRef<ArtifactRootReference> constraintSpatialMappings,
    const ArtifactStore &store);

llvm::Expected<std::vector<std::uint8_t>>
canonicalBindingKeyBytes(const SystemSearchBindingKey &key,
                         const ArtifactIdentity &dataflowIdentity);

llvm::Expected<std::vector<::dataflow::RootThreadLaunchRef>>
canonicalRootThreadLaunchSet(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    llvm::ArrayRef<::dataflow::RootThreadLaunchRef> roots);

llvm::Expected<std::vector<CanonicalSystemPartitionBinding>>
canonicalizeAndValidateSystemPartition(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    llvm::ArrayRef<::dataflow::RootThreadLaunchRef> roots,
    const SystemBindingPartitionPlan &plan);

llvm::Expected<bool> systemPresburgerCellsIntersect(
    const ::loom::mapping::SystemPresburgerCell &lhs,
    const ::loom::mapping::SystemPresburgerCell &rhs);

llvm::Expected<std::vector<SystemSearchServiceDomain>>
projectSystemServiceDomains(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::fabric::FabricSystemRootView &fabric,
    llvm::ArrayRef<::dataflow::RootThreadLaunchRef> roots,
    llvm::ArrayRef<SystemSearchBindingDomain> bindings,
    llvm::ArrayRef<SpatialCatalogEntry> spatialCatalog,
    const SystemFrozenConstraintIndex &constraints);

llvm::Expected<std::vector<FrozenSystemMemoryServiceBinding>>
projectSystemMemoryServiceBindings(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::fabric::FabricSystemRootView &fabric,
    llvm::ArrayRef<::dataflow::RootThreadLaunchRef> roots,
    llvm::ArrayRef<SpatialCatalogEntry> spatialCatalog,
    const SystemFrozenConstraintIndex &constraints);

llvm::Error validateSystemServiceDomains(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::fabric::FabricSystemRootView &fabric,
    llvm::ArrayRef<::dataflow::RootThreadLaunchRef> roots,
    llvm::ArrayRef<SystemSearchBindingDomain> bindings,
    llvm::ArrayRef<SystemSearchServiceDomain> services,
    const SystemFrozenConstraintIndex &constraints,
    llvm::ArrayRef<ArtifactRootReference> constraintSpatialMappings,
    const ArtifactStore &store);

} // namespace loom::pnr::detail

#endif // LOOM_LIB_PNR_SYSTEM_SYSTEMPNRSEARCHDOMAININTERNAL_H
