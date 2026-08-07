#ifndef LOOM_LIB_PNR_SYSTEM_SYSTEMPNRSEARCHDOMAININTERNAL_H
#define LOOM_LIB_PNR_SYSTEM_SYSTEMPNRSEARCHDOMAININTERNAL_H

#include "Mapping/Artifact/MappingArtifact.h"
#include "PnR/System/SystemPnrSearchDomain.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <vector>

namespace loom::pnr::detail {

struct CanonicalSystemPartitionBinding final {
  SystemSearchBindingKey key;
  std::vector<::loom::mapping::SystemPresburgerCell> cells;
};

struct SpatialCatalogEntry final {
  ArtifactRootReference reference;
  ::loom::mapping::FinalizedSpatialMapping mapping;
  std::uint64_t moduleDependencyOrdinal = 0;
  std::vector<::dataflow::GraphRef> covers;
};

struct FlatSpatialReopenCatalogEntry final {
  FlatSpatialReopenProblem problem;
  std::vector<::dataflow::GraphRef> covers;
};

struct FlatSpatialSeedCatalogEntry final {
  ArtifactRootReference reference;
  std::vector<::dataflow::GraphRef> covers;
};

struct CanonicalFlatGraphCatalog final {
  std::vector<FlatSpatialReopenCatalogEntry> problems;
  std::vector<FlatSpatialSeedCatalogEntry> seeds;
};

llvm::Expected<std::vector<SpatialCatalogEntry>>
importSpatialCatalog(llvm::ArrayRef<ArtifactRootReference> references,
                     const ::dataflow::CanonicalDataflowProgramView &dataflow,
                     const ::loom::fabric::FabricSystemRootView &system,
                     const ArtifactStore &store);

std::vector<::loom::fabric::AccCoreOccurrenceRef>
canonicalSystemAccCores(const ::loom::fabric::FabricSystemRootView &system);

bool flatSpatialReopenProblemLess(const FlatSpatialReopenProblem &left,
                                  const FlatSpatialReopenProblem &right);

llvm::Expected<CanonicalFlatGraphCatalog>
canonicalizeAndValidateFlatGraphCatalog(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::fabric::FabricSystemRootView &fabric,
    llvm::ArrayRef<::dataflow::GraphRef> requiredGraphs,
    const SystemFlatGraphSearchInput &input, const ArtifactStore &store);

llvm::Expected<SystemFlatGraphBindingDomain>
projectFlatGraphBindingDomain(const CanonicalFlatGraphCatalog &catalog,
                              ::dataflow::GraphRef graph);

llvm::Error validateSystemBindingDomains(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::fabric::FabricSystemRootView &fabric,
    llvm::ArrayRef<SystemSearchBindingDomain> bindings,
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
    llvm::ArrayRef<SpatialCatalogEntry> spatialCatalog, bool flatGraphSearch);

llvm::Error validateSystemServiceDomains(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::fabric::FabricSystemRootView &fabric,
    llvm::ArrayRef<::dataflow::RootThreadLaunchRef> roots,
    llvm::ArrayRef<SystemSearchBindingDomain> bindings,
    llvm::ArrayRef<SystemSearchServiceDomain> services,
    const ArtifactStore &store);

} // namespace loom::pnr::detail

#endif // LOOM_LIB_PNR_SYSTEM_SYSTEMPNRSEARCHDOMAININTERNAL_H
