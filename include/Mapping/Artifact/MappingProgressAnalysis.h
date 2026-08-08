#ifndef LOOM_MAPPING_ARTIFACT_MAPPINGPROGRESSANALYSIS_H
#define LOOM_MAPPING_ARTIFACT_MAPPINGPROGRESSANALYSIS_H

#include "Dataflow/IR/DataflowCanonicalArtifact.h"

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
/// topological induction once Mapping-level route, resource, and Fabric
/// progress closure are independently validated. Cyclic graphs remain
/// proof-not-established until their typed token and finite-buffer progress
/// mechanisms are implemented.
llvm::Expected<MappingProgressClosure> deriveMappingProgressClosure(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    llvm::ArrayRef<::dataflow::GraphRef> coveredGraphs);

} // namespace loom::mapping

#endif // LOOM_MAPPING_ARTIFACT_MAPPINGPROGRESSANALYSIS_H
