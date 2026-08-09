#ifndef LOOM_DATAFLOW_TRANSFORMS_DATAFLOW_REWRITE_H
#define LOOM_DATAFLOW_TRANSFORMS_DATAFLOW_REWRITE_H

#include "Dataflow/IR/DataflowCanonicalArtifact.h"

#include "mlir/Pass/Pass.h"

#include "llvm/Support/Error.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <variant>
#include <vector>

namespace dataflow {

enum class DataflowRewriteKind : std::uint32_t {
  SyncRendezvousRefactor = 0,
  PackUnpackRoundTripEliminate = 1,
  ParallelizeSerializeRoundTripEliminate = 2,
  ElementwiseCardinalityCommute = 3,
  PureComputeFanoutRefactor = 4,
  ActivationPreservingConstantFold = 5,
  GraphDefinitionRefactor = 6,
  ElementwiseVectorDecompose = 7,
};

enum class SyncRendezvousDirection : std::uint32_t {
  DirectToTree = 0,
  TreeToDirect = 1,
};

enum class CardinalityCommuteDirection : std::uint32_t {
  MoveInside = 0,
  MoveOutside = 1,
};

struct SyncRendezvousRewrite final {
  ActorId root;
  SyncRendezvousDirection direction;

  friend bool operator==(const SyncRendezvousRewrite &lhs,
                         const SyncRendezvousRewrite &rhs) {
    return lhs.root == rhs.root && lhs.direction == rhs.direction;
  }
};

struct PackUnpackRoundTripRewrite final {
  ActorId outerAdapter;

  friend bool operator==(const PackUnpackRoundTripRewrite &lhs,
                         const PackUnpackRoundTripRewrite &rhs) {
    return lhs.outerAdapter == rhs.outerAdapter;
  }
};

struct ParallelizeSerializeRoundTripRewrite final {
  ActorId outerSerialize;

  friend bool operator==(const ParallelizeSerializeRoundTripRewrite &lhs,
                         const ParallelizeSerializeRoundTripRewrite &rhs) {
    return lhs.outerSerialize == rhs.outerSerialize;
  }
};

struct ElementwiseCardinalityCommuteRewrite final {
  ActorId compute;
  std::vector<ActorId> adapters;
  CardinalityCommuteDirection direction;

  friend bool operator==(const ElementwiseCardinalityCommuteRewrite &lhs,
                         const ElementwiseCardinalityCommuteRewrite &rhs) {
    return lhs.compute == rhs.compute && lhs.adapters == rhs.adapters &&
           lhs.direction == rhs.direction;
  }
};

struct PureComputeFanoutReplicateRewrite final {
  ActorId compute;

  friend bool operator==(const PureComputeFanoutReplicateRewrite &lhs,
                         const PureComputeFanoutReplicateRewrite &rhs) {
    return lhs.compute == rhs.compute;
  }
};

struct PureComputeFanoutFactorRewrite final {
  std::vector<ActorId> replicas;

  friend bool operator==(const PureComputeFanoutFactorRewrite &lhs,
                         const PureComputeFanoutFactorRewrite &rhs) {
    return lhs.replicas == rhs.replicas;
  }
};

struct ActivationPreservingConstantFoldRewrite final {
  ActorId compute;

  friend bool operator==(const ActivationPreservingConstantFoldRewrite &lhs,
                         const ActivationPreservingConstantFoldRewrite &rhs) {
    return lhs.compute == rhs.compute;
  }
};

struct GraphDefinitionSplitRewrite final {
  GraphId graph;
  std::vector<StaticGraphLaunchId> launches;

  friend bool operator==(const GraphDefinitionSplitRewrite &lhs,
                         const GraphDefinitionSplitRewrite &rhs) {
    return lhs.graph == rhs.graph && lhs.launches == rhs.launches;
  }
};

struct GraphDefinitionMergeRewrite final {
  GraphId lowerGraph;
  GraphId higherGraph;

  friend bool operator==(const GraphDefinitionMergeRewrite &lhs,
                         const GraphDefinitionMergeRewrite &rhs) {
    return lhs.lowerGraph == rhs.lowerGraph &&
           lhs.higherGraph == rhs.higherGraph;
  }
};

struct ElementwiseVectorChunkRewrite final {
  ActorId compute;
  std::uint64_t leadingBlocksPerChunk;

  friend bool operator==(const ElementwiseVectorChunkRewrite &lhs,
                         const ElementwiseVectorChunkRewrite &rhs) {
    return lhs.compute == rhs.compute &&
           lhs.leadingBlocksPerChunk == rhs.leadingBlocksPerChunk;
  }
};

struct ElementwiseVectorScalarizeRewrite final {
  ActorId compute;

  friend bool operator==(const ElementwiseVectorScalarizeRewrite &lhs,
                         const ElementwiseVectorScalarizeRewrite &rhs) {
    return lhs.compute == rhs.compute;
  }
};

using DataflowRewriteDecision = std::variant<
    SyncRendezvousRewrite, PackUnpackRoundTripRewrite,
    ParallelizeSerializeRoundTripRewrite, ElementwiseCardinalityCommuteRewrite,
    PureComputeFanoutReplicateRewrite, PureComputeFanoutFactorRewrite,
    ActivationPreservingConstantFoldRewrite, GraphDefinitionSplitRewrite,
    GraphDefinitionMergeRewrite, ElementwiseVectorChunkRewrite,
    ElementwiseVectorScalarizeRewrite>;

DataflowRewriteKind
dataflowRewriteKind(const DataflowRewriteDecision &decision);

llvm::ArrayRef<std::uint8_t> dataflowRewriteDecisionSchemaBytes();
llvm::Expected<std::vector<std::uint8_t>>
encodeDataflowRewriteDecision(const DataflowRewriteDecision &decision);
llvm::Expected<DataflowRewriteDecision>
adoptDataflowRewriteDecision(llvm::ArrayRef<std::uint8_t> canonicalBytes);

bool dataflowRewriteDecisionLess(const DataflowRewriteDecision &lhs,
                                 const DataflowRewriteDecision &rhs);

/// Enumerates every legal normalized decision in kinds 0 through 6 for one
/// exact parent artifact, in canonical catalog order.
llvm::Expected<std::vector<DataflowRewriteDecision>>
enumerateFixedDataflowRewriteDecisions(const CanonicalDataflowArtifact &parent);

/// Enumerates the finite kind-7 domain for one exact actor. The input carries
/// the parent identity for resolution; emitted decisions retain only the
/// parent-local ActorId owned by their lineage edge.
llvm::Expected<std::vector<DataflowRewriteDecision>>
enumerateElementwiseVectorDecompositionDecisions(
    const CanonicalDataflowArtifact &parent, ActorRef actor);

llvm::Expected<std::uint64_t>
dataflowRewriteExpansionCost(const CanonicalDataflowArtifact &parent,
                             const DataflowRewriteDecision &decision);

/// Applies one exact decision to a private clone and finalizes the complete
/// result. A no-op returns an empty optional; a changed result has a distinct
/// immutable identity.
llvm::Expected<std::optional<CanonicalDataflowArtifact>>
materializeDataflowRewrite(const CanonicalDataflowArtifact &parent,
                           const DataflowRewriteDecision &decision);

/// Developer-only bulk driver for the three one-way legacy test surfaces. It
/// composes exact per-match decisions and is not a lineage decision API.
llvm::Expected<std::unique_ptr<::mlir::Pass>>
createDataflowRewritePass(DataflowRewriteKind kind);
void registerDataflowTransformsPasses();

} // namespace dataflow

#endif // LOOM_DATAFLOW_TRANSFORMS_DATAFLOW_REWRITE_H
