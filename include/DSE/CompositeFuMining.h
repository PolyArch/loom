#ifndef LOOM_DSE_COMPOSITEFUMINING_H
#define LOOM_DSE_COMPOSITEFUMINING_H

#include "ADG/FuLibrary.h"
#include "Common/ArtifactStore.h"
#include "DSE/FuReverseSynthesis.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Dataflow/IR/OperationSchema.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/IR/ImplementationFamily.h"

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <vector>

namespace loom::dse {

/// Bounds of one mining request. They keep enumeration finite and keep a
/// mined template's FU boundary within what a Spatial PE can present. They are
/// request properties, not Fabric legality: Fabric remains the owner of which
/// capability a synthesized template may declare.
struct CompositeFuMiningLimits final {
  /// Largest admitted node count of one candidate.
  std::uint32_t maximumActorCount = 8;
  /// Largest admitted FU boundary port count of one reported candidate. The
  /// bound is not monotone in the node count, so a candidate above it is still
  /// extended and only withheld from the ranked result.
  std::uint32_t maximumBoundaryPortCount = 12;
  /// Minimum-image support a candidate must reach. Support is the least
  /// number of distinct actors any one node position binds across the graph
  /// set, which is the standard measure that stays anti-monotone inside a
  /// single graph: a shape that repeats twice in one whole-layer graph is as
  /// interesting as one shared by two graphs, and counting embeddings instead
  /// would not be monotone at all.
  std::uint64_t minimumSupport = 2;
  /// Candidates retained at one node count before the search stops growing.
  std::uint64_t maximumCandidateCount = 4096;
  /// Embeddings enumerated at one node count before the search stops growing.
  /// Shapes and embeddings grow independently, so both are bounded.
  std::uint64_t maximumOccurrenceCount = 1u << 16;
};

/// Price of one FU boundary port in the mining rank. A boundary port is what a
/// PE must present and route, so it is the one structural cost the rank pays
/// against coverage.
inline constexpr std::int64_t compositeFuBoundaryPortCost = 1;

/// The one mining request production uses. The owner that selects a template
/// and the generator that re-derives it from the configuration must mine the
/// same relation, or a named selection would not reproduce; this is that
/// single request. Its embedding budget is sized for a whole-layer graph,
/// where the search is expected to reach its bound and report that it did.
inline constexpr CompositeFuMiningLimits productionCompositeFuMiningLimits{
    8, 12, 2, 4096, 1u << 18};

/// One node of a mined shape. Node identity is the registered operation schema
/// together with the exact ordered operand and result types; exact attribute
/// payloads stay owned by the software graphs and are supplied by legal
/// bindings.
struct CompositeFuNode final {
  ::dataflow::OperationSchemaId schema{};
  ::mlir::FunctionType type;
};

/// One token edge whose producer and consumer are both nodes of the shape. The
/// induced relation is complete: a mined shape never omits one.
struct CompositeFuInternalEdge final {
  std::uint32_t producerNode = 0;
  std::uint64_t producerResult = 0;
  std::uint32_t consumerNode = 0;
  std::uint64_t consumerOperand = 0;
};

/// One FU boundary port of a mined shape, named by the node and the actor port
/// ordinal that crosses the boundary.
struct CompositeFuBoundaryPort final {
  std::uint32_t node = 0;
  std::uint64_t portOrdinal = 0;
};

/// One embedding of a mined shape in one canonical graph. `actors` is in node
/// order, so the ordinal of an actor is the ordinal of the node it binds.
struct CompositeFuOccurrence final {
  ::dataflow::GraphRef graph;
  std::vector<::dataflow::ActorRef> actors;
};

/// One mined common subgraph of the input graph set together with every
/// embedding that produced it. `canonicalKey` is the shape's identity: the
/// least code over the connected labelings of its nodes, internal edges, and
/// ordered boundary ports, so it depends on structure alone.
struct CompositeFuCandidate final {
  std::vector<CompositeFuNode> nodes;
  std::vector<CompositeFuInternalEdge> internalEdges;
  std::vector<CompositeFuBoundaryPort> inputs;
  std::vector<CompositeFuBoundaryPort> outputs;
  std::vector<CompositeFuOccurrence> occurrences;
  std::vector<std::uint8_t> canonicalKey;
  /// Actors a greedy disjoint packing of the occurrences covers. Two
  /// embeddings that share an actor cannot both be realized, so the union of
  /// all embeddings would overstate what one template absorbs.
  std::uint64_t coveredActorCount = 0;
  /// Members of the graph set holding at least one occurrence.
  std::uint64_t graphCount = 0;
  /// Least number of distinct actors any one node position binds. This is the
  /// measure the search prunes on, so a reported candidate always reaches the
  /// request's minimum.
  std::uint64_t support = 0;
  std::int64_t score = 0;
};

/// What one bounded mining request found. A search that reaches a retained
/// shape or embedding bound stops growing instead of failing: every candidate
/// it already reported is exact, and only larger shapes are unexplored. The
/// result says so explicitly, so a caller never mistakes a bounded search for
/// a complete one.
struct CompositeFuMiningResult final {
  std::vector<CompositeFuCandidate> candidates;
  /// Largest node count whose level the search completed.
  std::uint32_t exploredActorCount = 0;
  /// Whether a retained shape or embedding bound stopped the growth before
  /// the node bound did.
  bool bounded = false;
};

/// Mines the common subgraphs of a canonical graph set and reports them in
/// rank order. The rank is `coveredActorCount * graphCount` less the boundary
/// port price, with the canonical code as the final tie-break, so the order is
/// total and depends on no actor identity.
///
/// Mining reads only the canonical token-plane relation and the registered
/// operation-schema projection of each actor. An actor carrying a memory
/// capability is outside that relation and is not an admitted node.
llvm::Expected<CompositeFuMiningResult> mineCompositeFuCandidates(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    llvm::ArrayRef<::dataflow::GraphRef> graphs,
    const CompositeFuMiningLimits &limits = {});

/// Derives the hardware request of one mined candidate from the exact actors
/// of its occurrences. Each node's family is the least registered family that
/// owns the node schema and whose canonical capability derivation admits every
/// occurrence's actor at that node. A node whose family has no inverse policy
/// is a typed `CapabilityDerivationRejected`.
///
/// The result is the ADG Builder's own composite FU vocabulary, which is the
/// single owner of FU structure: no caller re-describes the topology, and the
/// mined candidate remains the software evidence that produced it.
llvm::Expected<::loom::adg::CompositeFuSpec> deriveCompositeFuTemplate(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const CompositeFuCandidate &candidate);

/// `F = Synthesize(S)` for one mined candidate: one finalized Module holding
/// one Spatial PE whose single FU is the mined template, plus one coverage
/// witness per occurrence proving `S subset-of Materialize(F)`. The witnesses
/// use the shared reverse-synthesis witness owner and the same realization
/// closure verifier TechMapping uses; this owner writes no second cover
/// algorithm and publishes no Mapping artifact.
struct CompositeFuTemplateArtifacts final {
  ::loom::fabric::FinalizedFabricRoot module;
  ::loom::fabric::FabricFuCapabilityTemplateRef capabilityTemplate;
  std::vector<FuSynthesisCoverageWitness> coverage;
};

llvm::Expected<CompositeFuTemplateArtifacts> synthesizeCompositeFuTemplate(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const CompositeFuCandidate &candidate, const ArtifactStore &store);

} // namespace loom::dse

#endif // LOOM_DSE_COMPOSITEFUMINING_H
