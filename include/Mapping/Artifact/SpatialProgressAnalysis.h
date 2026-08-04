#ifndef LOOM_MAPPING_ARTIFACT_SPATIALPROGRESSANALYSIS_H
#define LOOM_MAPPING_ARTIFACT_SPATIALPROGRESSANALYSIS_H

#include "Dataflow/IR/DataflowCanonicalArtifact.h"

#include "llvm/Support/Error.h"

#include <cstdint>

namespace loom::mapping {

/// Result of the exact progress proof currently supported by Mapping. A
/// dependency cycle is not itself a deadlock, so the provider fails closed
/// when the existing typed owners cannot prove whether that cycle is broken.
enum class SpatialProgressClosureKind : std::uint8_t {
  ProvenNoClosedWaitSet,
  ProvenClosedWaitSet,
  ProofNotEstablished,
};

struct SpatialProgressClosure final {
  SpatialProgressClosureKind kind =
      SpatialProgressClosureKind::ProofNotEstablished;
};

/// Derives the reusable Dataflow basis of Spatial progress closure. The
/// canonical actor dependency graph is analyzed in linear CSR form. An
/// acyclic graph provides a complete topological induction once selected
/// handshake closure and Fabric-owned atomic-use progress are independently
/// validated. Cyclic graphs remain proof-not-established until their typed
/// token and finite-buffer progress mechanisms are implemented.
llvm::Expected<SpatialProgressClosure> deriveSpatialProgressClosure(
    const ::dataflow::CanonicalDataflowProgramView &dataflow);

} // namespace loom::mapping

#endif // LOOM_MAPPING_ARTIFACT_SPATIALPROGRESSANALYSIS_H
