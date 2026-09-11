//===- CompositeFuMining.cpp - common software subgraph mining -----------===//
//
// Mines the common connected induced subgraphs of a canonical Dataflow graph
// set and ranks them as composite FU template candidates. This file owns the
// software side only: it reads the canonical token-plane relation and the
// registered operation-schema projection of each actor, and never decides
// which implementation family realizes a node.
//
//===----------------------------------------------------------------------===//

#include "DSE/CompositeFuMining.h"

#include "Dataflow/IR/DataflowStructuralRefs.h"

#include "mlir/IR/Operation.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace loom::dse {
namespace {

/// Labeling steps one shape canonicalization may take. The search is a
/// branch-and-bound over the connected labelings of at most
/// `maximumActorCount` nodes, so this bound is reached only by a request whose
/// node bound is far beyond what an FU boundary can carry.
constexpr std::uint64_t maximumShapeLabelingSteps = 1u << 20;

llvm::Error failure(FuReverseSynthesisFailure kind,
                    const llvm::Twine &message) {
  return llvm::make_error<FuReverseSynthesisError>(kind, message.str());
}

void appendU32(std::vector<std::uint8_t> &bytes, std::uint32_t value) {
  bytes.push_back(static_cast<std::uint8_t>(value >> 24));
  bytes.push_back(static_cast<std::uint8_t>(value >> 16));
  bytes.push_back(static_cast<std::uint8_t>(value >> 8));
  bytes.push_back(static_cast<std::uint8_t>(value));
}

void appendU64(std::vector<std::uint8_t> &bytes, std::uint64_t value) {
  appendU32(bytes, static_cast<std::uint32_t>(value >> 32));
  appendU32(bytes, static_cast<std::uint32_t>(value & 0xffffffffu));
}

/// The node identity of one actor: its registered operation schema and its
/// exact ordered operand and result types. Attribute payloads are deliberately
/// absent; they stay owned by the software graph.
std::vector<std::uint8_t> nodeSignature(::dataflow::OperationSchemaId schema,
                                        ::mlir::FunctionType type) {
  std::vector<std::uint8_t> bytes;
  appendU32(bytes, static_cast<std::uint32_t>(schema));
  std::string text;
  llvm::raw_string_ostream stream(text);
  type.print(stream);
  stream.flush();
  appendU32(bytes, static_cast<std::uint32_t>(text.size()));
  bytes.insert(bytes.end(), text.begin(), text.end());
  return bytes;
}

struct EdgeTarget final {
  std::size_t actor = 0;
  std::uint64_t ordinal = 0;
};

/// One admitted actor of the mining domain together with its token-plane
/// neighbourhood. Every index refers to this same admitted inventory; a
/// producer or consumer that is not admitted is external by construction.
struct MinedActor final {
  ::dataflow::ActorRef ref;
  std::size_t graph = 0;
  ::dataflow::OperationSchemaId schema{};
  ::mlir::FunctionType type;
  std::vector<std::uint8_t> signature;
  std::vector<std::optional<EdgeTarget>> operandProducers;
  std::vector<std::vector<EdgeTarget>> resultConsumers;
  std::vector<bool> resultLeavesDomain;
  std::vector<std::size_t> neighbors;
};

std::optional<::dataflow::CanonicalGraphProducerEndpointRef>
resolveOperandProducer(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    ::dataflow::ActorRef actor, std::uint64_t ordinal) {
  ::dataflow::CanonicalGraphConsumerEndpointRef consumer =
      ::dataflow::ActorTokenOperandRef{actor, ordinal};
  auto producer = dataflow.graphProducer(consumer);
  if (!producer) {
    llvm::consumeError(producer.takeError());
    return std::nullopt;
  }
  return std::move(*producer);
}

struct PendingActor final {
  ::dataflow::CanonicalActorView view;
  ::dataflow::CanonicalActorSchemaProjection projection;
  std::size_t graph = 0;
  std::vector<::dataflow::CanonicalGraphProducerEndpointRef> producers;
};

/// Collects the admitted actors of the requested graphs in canonical order. An
/// actor is admitted when it is a token-plane actor whose registered
/// projection covers every operand and result and whose operands all resolve
/// to a token producer. Everything else, including memory actors, is outside
/// the mined relation.
std::vector<MinedActor>
indexMinedActors(const ::dataflow::CanonicalDataflowProgramView &dataflow,
                 llvm::ArrayRef<::dataflow::GraphRef> graphs) {
  std::map<std::uint64_t, std::size_t> graphPosition;
  for (const auto &indexed : llvm::enumerate(graphs))
    graphPosition.emplace(indexed.value().entity.value(), indexed.index());

  std::vector<PendingActor> pending;
  std::map<std::uint64_t, std::size_t> actorPosition;
  for (const ::dataflow::CanonicalActorView &actor : dataflow.actors()) {
    const auto graph = graphPosition.find(actor.graph.entity.value());
    if (graph == graphPosition.end())
      continue;
    if (actor.kind == ::dataflow::CanonicalDataflowActorKind::Memory)
      continue;
    auto projection =
        ::dataflow::projectRegisteredActorSchemaProjection(actor.op);
    if (!projection) {
      llvm::consumeError(projection.takeError());
      continue;
    }
    if (actor.op->getNumOperands() != projection->type.getNumInputs() ||
        actor.op->getNumResults() != projection->type.getNumResults())
      continue;
    std::vector<::dataflow::CanonicalGraphProducerEndpointRef> producers;
    producers.reserve(projection->type.getNumInputs());
    bool admitted = true;
    for (std::uint64_t ordinal = 0; ordinal != projection->type.getNumInputs();
         ++ordinal) {
      auto producer = resolveOperandProducer(dataflow, actor.ref, ordinal);
      if (!producer) {
        admitted = false;
        break;
      }
      producers.push_back(std::move(*producer));
    }
    if (!admitted)
      continue;
    actorPosition.emplace(actor.ref.entity.value(), pending.size());
    pending.push_back(
        {actor, std::move(*projection), graph->second, std::move(producers)});
  }

  std::vector<MinedActor> actors;
  actors.reserve(pending.size());
  for (const PendingActor &entry : pending) {
    MinedActor mined;
    mined.ref = entry.view.ref;
    mined.graph = entry.graph;
    mined.schema = entry.projection.schema;
    mined.type = entry.projection.type;
    mined.signature = nodeSignature(mined.schema, mined.type);
    std::set<std::size_t> neighbors;
    for (const ::dataflow::CanonicalGraphProducerEndpointRef &producer :
         entry.producers) {
      std::optional<EdgeTarget> target;
      if (const auto *result =
              std::get_if<::dataflow::ActorTokenResultRef>(&producer)) {
        const auto found = actorPosition.find(result->actor.entity.value());
        if (found != actorPosition.end()) {
          target = EdgeTarget{found->second, result->ordinal};
          neighbors.insert(found->second);
        }
      }
      mined.operandProducers.push_back(target);
    }
    for (std::uint64_t ordinal = 0; ordinal != mined.type.getNumResults();
         ++ordinal) {
      std::vector<EdgeTarget> consumers;
      bool leaves = false;
      ::dataflow::CanonicalGraphProducerEndpointRef producer =
          ::dataflow::ActorTokenResultRef{mined.ref, ordinal};
      auto resolved = dataflow.graphConsumers(producer);
      if (!resolved) {
        llvm::consumeError(resolved.takeError());
        leaves = true;
      } else {
        for (const ::dataflow::CanonicalGraphConsumerEndpointRef &consumer :
             *resolved) {
          const auto *operand =
              std::get_if<::dataflow::ActorTokenOperandRef>(&consumer);
          if (!operand) {
            leaves = true;
            continue;
          }
          const auto found = actorPosition.find(operand->actor.entity.value());
          if (found == actorPosition.end()) {
            leaves = true;
            continue;
          }
          consumers.push_back(EdgeTarget{found->second, operand->ordinal});
          neighbors.insert(found->second);
        }
      }
      mined.resultConsumers.push_back(std::move(consumers));
      mined.resultLeavesDomain.push_back(leaves);
    }
    mined.neighbors.assign(neighbors.begin(), neighbors.end());
    actors.push_back(std::move(mined));
  }
  return actors;
}

/// One internal token edge of a shape, in shape-local node positions.
struct RawEdge final {
  std::size_t producer = 0;
  std::uint64_t producerResult = 0;
  std::size_t consumer = 0;
  std::uint64_t consumerOperand = 0;
};

struct ShapeDescription final {
  std::vector<std::uint8_t> key;
  std::vector<std::size_t> order;
  std::vector<CompositeFuNode> nodes;
  std::vector<CompositeFuInternalEdge> internalEdges;
  std::vector<CompositeFuBoundaryPort> inputs;
  std::vector<CompositeFuBoundaryPort> outputs;
};

/// Lexicographic comparison of a partial code against the best complete code.
/// A partial code that is already greater cannot be completed into a smaller
/// one, so the caller prunes on `1`.
int comparePrefix(const std::vector<std::uint8_t> &partial,
                  const std::vector<std::uint8_t> &best) {
  if (best.empty())
    return -1;
  const std::size_t shared = std::min(partial.size(), best.size());
  for (std::size_t index = 0; index != shared; ++index) {
    if (partial[index] < best[index])
      return -1;
    if (partial[index] > best[index])
      return 1;
  }
  return 0;
}

/// Branch-and-bound over the connected labelings of one actor set. The result
/// is the least code, which is the shape's identity: it depends on the node
/// signatures, the internal edges, and the boundary alone.
class ShapeLabeling final {
public:
  ShapeLabeling(const std::vector<MinedActor> &actors,
                const std::vector<std::size_t> &set,
                const std::vector<RawEdge> &edges,
                const std::vector<std::vector<std::size_t>> &adjacency)
      : actors_(actors), set_(set), edges_(edges), adjacency_(adjacency),
        used_(set.size(), false), placement_(set.size(), 0),
        position_(set.size(), 0) {}

  bool run() { return place(0); }

  const std::vector<std::uint8_t> &bestCode() const { return bestCode_; }
  const std::vector<std::size_t> &bestOrder() const { return bestOrder_; }

private:
  struct Incidence final {
    std::size_t otherPosition = 0;
    std::uint32_t direction = 0;
    std::uint64_t producerResult = 0;
    std::uint64_t consumerOperand = 0;

    bool operator<(const Incidence &other) const {
      if (otherPosition != other.otherPosition)
        return otherPosition < other.otherPosition;
      if (direction != other.direction)
        return direction < other.direction;
      if (producerResult != other.producerResult)
        return producerResult < other.producerResult;
      return consumerOperand < other.consumerOperand;
    }
  };

  bool adjacentToPlaced(std::size_t node) const {
    for (std::size_t neighbor : adjacency_[node])
      if (used_[neighbor])
        return true;
    return false;
  }

  void appendNode(std::size_t node) {
    const std::vector<std::uint8_t> &signature = actors_[set_[node]].signature;
    code_.insert(code_.end(), signature.begin(), signature.end());
    std::vector<Incidence> incidences;
    for (const RawEdge &edge : edges_) {
      if (edge.consumer == node && used_[edge.producer])
        incidences.push_back({position_[edge.producer], 0,
                              edge.producerResult, edge.consumerOperand});
      if (edge.producer == node && used_[edge.consumer])
        incidences.push_back({position_[edge.consumer], 1,
                              edge.producerResult, edge.consumerOperand});
    }
    llvm::sort(incidences);
    appendU32(code_, static_cast<std::uint32_t>(incidences.size()));
    for (const Incidence &incidence : incidences) {
      appendU32(code_, static_cast<std::uint32_t>(incidence.otherPosition));
      appendU32(code_, incidence.direction);
      appendU64(code_, incidence.producerResult);
      appendU64(code_, incidence.consumerOperand);
    }
  }

  void appendBoundary() {
    std::vector<std::pair<std::size_t, std::uint64_t>> inputs;
    std::vector<std::pair<std::size_t, std::uint64_t>> outputs;
    for (std::size_t position = 0; position != placement_.size(); ++position) {
      const MinedActor &actor = actors_[set_[placement_[position]]];
      for (const auto &indexed : llvm::enumerate(actor.operandProducers)) {
        const std::optional<EdgeTarget> &producer = indexed.value();
        if (!producer || !contains(producer->actor))
          inputs.emplace_back(position, indexed.index());
      }
      for (const auto &indexed : llvm::enumerate(actor.resultConsumers)) {
        bool external = actor.resultLeavesDomain[indexed.index()];
        for (const EdgeTarget &consumer : indexed.value())
          external = external || !contains(consumer.actor);
        if (external)
          outputs.emplace_back(position, indexed.index());
      }
    }
    appendU32(code_, static_cast<std::uint32_t>(inputs.size()));
    for (const auto &port : inputs) {
      appendU32(code_, static_cast<std::uint32_t>(port.first));
      appendU64(code_, port.second);
    }
    appendU32(code_, static_cast<std::uint32_t>(outputs.size()));
    for (const auto &port : outputs) {
      appendU32(code_, static_cast<std::uint32_t>(port.first));
      appendU64(code_, port.second);
    }
  }

  bool contains(std::size_t actor) const {
    return std::binary_search(set_.begin(), set_.end(), actor);
  }

  bool place(std::size_t depth) {
    if (++steps_ > maximumShapeLabelingSteps)
      return false;
    if (depth == 0)
      appendU32(code_, static_cast<std::uint32_t>(set_.size()));
    if (depth == set_.size()) {
      const std::size_t prefix = code_.size();
      appendBoundary();
      if (comparePrefix(code_, bestCode_) < 0) {
        bestCode_ = code_;
        bestOrder_.assign(placement_.begin(), placement_.end());
        for (std::size_t &node : bestOrder_)
          node = set_[node];
      }
      code_.resize(prefix);
      return true;
    }
    for (std::size_t node = 0; node != set_.size(); ++node) {
      if (used_[node])
        continue;
      if (depth != 0 && !adjacentToPlaced(node))
        continue;
      const std::size_t prefix = code_.size();
      used_[node] = true;
      placement_[depth] = node;
      position_[node] = depth;
      appendNode(node);
      bool completed = true;
      if (comparePrefix(code_, bestCode_) <= 0)
        completed = place(depth + 1);
      used_[node] = false;
      code_.resize(prefix);
      if (!completed)
        return false;
    }
    return true;
  }

  const std::vector<MinedActor> &actors_;
  const std::vector<std::size_t> &set_;
  const std::vector<RawEdge> &edges_;
  const std::vector<std::vector<std::size_t>> &adjacency_;
  std::uint64_t steps_ = 0;
  std::vector<bool> used_;
  std::vector<std::size_t> placement_;
  std::vector<std::size_t> position_;
  std::vector<std::uint8_t> code_;
  std::vector<std::uint8_t> bestCode_;
  std::vector<std::size_t> bestOrder_;
};

/// Describes one connected induced actor set: its canonical node order, its
/// complete internal edge relation, and its ordered FU boundary.
llvm::Expected<ShapeDescription>
describeShape(const std::vector<MinedActor> &actors,
              const std::vector<std::size_t> &set) {
  std::map<std::size_t, std::size_t> local;
  for (const auto &indexed : llvm::enumerate(set))
    local.emplace(indexed.value(), indexed.index());

  std::vector<RawEdge> edges;
  std::vector<std::vector<std::size_t>> adjacency(set.size());
  for (const auto &indexed : llvm::enumerate(set)) {
    const MinedActor &actor = actors[indexed.value()];
    for (const auto &operand : llvm::enumerate(actor.operandProducers)) {
      if (!operand.value())
        continue;
      const auto producer = local.find(operand.value()->actor);
      if (producer == local.end())
        continue;
      edges.push_back({producer->second, operand.value()->ordinal,
                       indexed.index(), operand.index()});
      adjacency[indexed.index()].push_back(producer->second);
      adjacency[producer->second].push_back(indexed.index());
    }
  }
  for (std::vector<std::size_t> &neighbors : adjacency) {
    llvm::sort(neighbors);
    neighbors.erase(std::unique(neighbors.begin(), neighbors.end()),
                    neighbors.end());
  }

  ShapeLabeling labeling(actors, set, edges, adjacency);
  if (!labeling.run())
    return failure(FuReverseSynthesisFailure::MiningBoundExhausted,
                   "shape canonicalization exceeded its labeling bound");

  ShapeDescription description;
  description.key = labeling.bestCode();
  description.order = labeling.bestOrder();
  std::map<std::size_t, std::uint32_t> position;
  for (const auto &indexed : llvm::enumerate(description.order))
    position.emplace(indexed.value(),
                     static_cast<std::uint32_t>(indexed.index()));
  for (std::size_t actor : description.order)
    description.nodes.push_back({actors[actor].schema, actors[actor].type});
  for (const auto &indexed : llvm::enumerate(description.order)) {
    const MinedActor &actor = actors[indexed.value()];
    const auto node = static_cast<std::uint32_t>(indexed.index());
    for (const auto &operand : llvm::enumerate(actor.operandProducers)) {
      const std::optional<EdgeTarget> &producer = operand.value();
      const auto found =
          producer ? position.find(producer->actor) : position.end();
      if (found == position.end()) {
        description.inputs.push_back({node, operand.index()});
        continue;
      }
      description.internalEdges.push_back(
          {found->second, producer->ordinal, node, operand.index()});
    }
    for (const auto &result : llvm::enumerate(actor.resultConsumers)) {
      bool external = actor.resultLeavesDomain[result.index()];
      for (const EdgeTarget &consumer : result.value())
        external = external || position.find(consumer.actor) == position.end();
      if (external)
        description.outputs.push_back({node, result.index()});
    }
  }
  llvm::sort(description.internalEdges,
             [](const CompositeFuInternalEdge &left,
                const CompositeFuInternalEdge &right) {
               if (left.consumerNode != right.consumerNode)
                 return left.consumerNode < right.consumerNode;
               if (left.consumerOperand != right.consumerOperand)
                 return left.consumerOperand < right.consumerOperand;
               if (left.producerNode != right.producerNode)
                 return left.producerNode < right.producerNode;
               return left.producerResult < right.producerResult;
             });
  return description;
}

struct LevelCandidate final {
  ShapeDescription shape;
  std::vector<std::vector<std::size_t>> occurrences;
  std::vector<std::vector<std::size_t>> orders;
  std::set<std::size_t> graphs;
};

/// Actors a greedy disjoint packing of the occurrences covers, in canonical
/// occurrence order. Two embeddings that share an actor cannot both be
/// realized, so the union of all embeddings would overstate what one template
/// can absorb. This is a ranking statistic over mined embeddings; the Fabric
/// coverage witness remains its own owner.
std::uint64_t
packedActorCount(llvm::ArrayRef<CompositeFuOccurrence> occurrences) {
  std::set<std::uint64_t> claimed;
  std::uint64_t packed = 0;
  for (const CompositeFuOccurrence &occurrence : occurrences) {
    const bool overlaps =
        llvm::any_of(occurrence.actors, [&](::dataflow::ActorRef actor) {
          return claimed.count(actor.entity.value()) != 0;
        });
    if (overlaps)
      continue;
    for (::dataflow::ActorRef actor : occurrence.actors)
      claimed.insert(actor.entity.value());
    packed += occurrence.actors.size();
  }
  return packed;
}

std::int64_t candidateScore(std::uint64_t coveredActorCount,
                            std::uint64_t graphCount,
                            std::size_t boundaryPortCount) {
  return static_cast<std::int64_t>(coveredActorCount) *
             static_cast<std::int64_t>(graphCount) -
         compositeFuBoundaryPortCost *
             static_cast<std::int64_t>(boundaryPortCount);
}

} // namespace

llvm::Expected<std::vector<CompositeFuCandidate>> mineCompositeFuCandidates(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    llvm::ArrayRef<::dataflow::GraphRef> graphs,
    const CompositeFuMiningLimits &limits) {
  if (graphs.empty())
    return failure(FuReverseSynthesisFailure::EmptyGraphSet,
                   "composite FU mining requires a non-empty graph set");
  if (limits.maximumActorCount < 2 || limits.minimumGraphSupport == 0 ||
      limits.maximumCandidateCount == 0 || limits.maximumOccurrenceCount == 0)
    return failure(FuReverseSynthesisFailure::MiningBoundExhausted,
                   "composite FU mining limits admit no candidate");
  std::set<std::uint64_t> seen;
  for (::dataflow::GraphRef graph : graphs) {
    if (graph.artifact != dataflow.identity())
      return failure(FuReverseSynthesisFailure::InvalidGraphReference,
                     "composite FU mining received a foreign graph ref");
    if (!seen.insert(graph.entity.value()).second)
      return failure(FuReverseSynthesisFailure::DuplicateGraph,
                     "composite FU mining graph set contains a duplicate");
    auto resolved = dataflow.resolve(graph);
    if (!resolved)
      return failure(FuReverseSynthesisFailure::InvalidGraphReference,
                     llvm::toString(resolved.takeError()));
  }

  const std::vector<MinedActor> actors = indexMinedActors(dataflow, graphs);
  std::set<std::vector<std::size_t>> level;
  for (std::size_t actor = 0; actor != actors.size(); ++actor)
    level.insert({actor});

  std::vector<CompositeFuCandidate> results;
  for (std::uint32_t size = 1; size <= limits.maximumActorCount; ++size) {
    if (level.empty())
      break;
    std::map<std::vector<std::uint8_t>, std::size_t> candidateByKey;
    std::vector<LevelCandidate> candidates;
    for (const std::vector<std::size_t> &set : level) {
      auto described = describeShape(actors, set);
      if (!described)
        return described.takeError();
      auto found = candidateByKey.find(described->key);
      if (found == candidateByKey.end()) {
        if (candidates.size() >= limits.maximumCandidateCount)
          return failure(FuReverseSynthesisFailure::MiningBoundExhausted,
                         "composite FU mining exceeded its candidate bound");
        found = candidateByKey.emplace(described->key, candidates.size()).first;
        candidates.push_back(LevelCandidate{*described, {}, {}, {}});
      }
      LevelCandidate &candidate = candidates[found->second];
      candidate.occurrences.push_back(set);
      candidate.orders.push_back(described->order);
      candidate.graphs.insert(actors[set.front()].graph);
    }

    std::set<std::vector<std::size_t>> next;
    for (const LevelCandidate &candidate : candidates) {
      if (candidate.graphs.size() <
          static_cast<std::size_t>(limits.minimumGraphSupport))
        continue;
      const std::size_t ports =
          candidate.shape.inputs.size() + candidate.shape.outputs.size();
      if (size >= 2 && ports <= limits.maximumBoundaryPortCount) {
        CompositeFuCandidate reported;
        reported.nodes = candidate.shape.nodes;
        reported.internalEdges = candidate.shape.internalEdges;
        reported.inputs = candidate.shape.inputs;
        reported.outputs = candidate.shape.outputs;
        reported.canonicalKey = candidate.shape.key;
        for (const auto &indexed : llvm::enumerate(candidate.occurrences)) {
          CompositeFuOccurrence occurrence;
          occurrence.graph = graphs[actors[indexed.value().front()].graph];
          for (std::size_t actor : candidate.orders[indexed.index()])
            occurrence.actors.push_back(actors[actor].ref);
          reported.occurrences.push_back(std::move(occurrence));
        }
        llvm::sort(reported.occurrences,
                   [](const CompositeFuOccurrence &left,
                      const CompositeFuOccurrence &right) {
                     if (left.graph.entity.value() !=
                         right.graph.entity.value())
                       return left.graph.entity.value() <
                              right.graph.entity.value();
                     return left.actors.front().entity.value() <
                            right.actors.front().entity.value();
                   });
        reported.coveredActorCount = packedActorCount(reported.occurrences);
        reported.graphCount = candidate.graphs.size();
        reported.score = candidateScore(reported.coveredActorCount,
                                        reported.graphCount, ports);
        results.push_back(std::move(reported));
      }
      if (size == limits.maximumActorCount)
        continue;
      for (const std::vector<std::size_t> &set : candidate.occurrences)
        for (std::size_t actor : set)
          for (std::size_t neighbor : actors[actor].neighbors) {
            if (std::binary_search(set.begin(), set.end(), neighbor))
              continue;
            std::vector<std::size_t> grown = set;
            grown.insert(std::upper_bound(grown.begin(), grown.end(), neighbor),
                         neighbor);
            next.insert(std::move(grown));
            if (next.size() > limits.maximumOccurrenceCount)
              return failure(
                  FuReverseSynthesisFailure::MiningBoundExhausted,
                  "composite FU mining exceeded its embedding bound");
          }
    }
    level = std::move(next);
  }

  llvm::sort(results, [](const CompositeFuCandidate &left,
                         const CompositeFuCandidate &right) {
    if (left.score != right.score)
      return left.score > right.score;
    if (left.coveredActorCount != right.coveredActorCount)
      return left.coveredActorCount > right.coveredActorCount;
    if (left.nodes.size() != right.nodes.size())
      return left.nodes.size() > right.nodes.size();
    const std::size_t leftPorts = left.inputs.size() + left.outputs.size();
    const std::size_t rightPorts = right.inputs.size() + right.outputs.size();
    if (leftPorts != rightPorts)
      return leftPorts < rightPorts;
    return left.canonicalKey < right.canonicalKey;
  });
  return results;
}

} // namespace loom::dse
