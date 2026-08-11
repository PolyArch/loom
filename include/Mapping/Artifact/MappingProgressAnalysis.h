#ifndef LOOM_MAPPING_ARTIFACT_MAPPINGPROGRESSANALYSIS_H
#define LOOM_MAPPING_ARTIFACT_MAPPINGPROGRESSANALYSIS_H

#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Mapping/Artifact/MappingArtifact.h"

#include "llvm/Support/Error.h"

#include <cstdint>

namespace loom::mapping {

/// Result of the exact progress proof currently shared by Spatial and System
/// Mapping. A dependency cycle is not itself a deadlock, so the provider fails
/// closed when the existing typed owners cannot prove whether it is broken.
enum class MappingProgressClosureKind : std::uint8_t {
  ProvenNoClosedWaitSet,
  ProvenClosedWaitSet,
  ProofNotEstablished,
};

struct MappingProgressClosure final {
  MappingProgressClosureKind kind =
      MappingProgressClosureKind::ProofNotEstablished;
};

/// Derives the reusable Dataflow basis of Mapping progress closure for exactly
/// the supplied covered graphs. Their canonical actor dependency graph is
/// analyzed in linear CSR form. An acyclic graph provides a complete
/// topological induction basis. Mapping-level route, resource, and Fabric
/// progress closure must additionally validate the selected mapping; the basis
/// alone never proves a routed candidate. Cyclic graphs remain
/// proof-not-established until their typed token and finite-buffer progress
/// mechanisms are implemented.
llvm::Expected<MappingProgressClosure> deriveMappingProgressClosure(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    llvm::ArrayRef<::dataflow::GraphRef> coveredGraphs);

/// One route-level wait dependency within a residual multicast net. Ordinals
/// address the canonical TechMapping residual-net and sink inventories; this
/// projection has no persistent identity.
struct SpatialRouteProgressDependency final {
  std::uint64_t logicalNetOrdinal = 0;
  std::uint64_t prerequisiteSinkOrdinal = 0;
  std::uint64_t dependentSinkOrdinal = 0;
};

llvm::Expected<std::vector<SpatialRouteProgressDependency>>
deriveSpatialRouteProgressDependencies(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const TechMappingView &techMapping);

/// Completes the reusable Dataflow basis against exact selected route trees.
/// A dependent multicast branch must cross a Buffered FIFO after it diverges
/// from every prerequisite branch; a FIFO on the shared prefix cannot release
/// the atomic fork.
llvm::Expected<MappingProgressClosure> deriveSpatialMappingProgressClosure(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const TechMappingView &techMapping,
    llvm::ArrayRef<SpatialRouteTreeView> routes);

} // namespace loom::mapping

#endif // LOOM_MAPPING_ARTIFACT_MAPPINGPROGRESSANALYSIS_H
