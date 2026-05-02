#include "Fabric/Tech/Synthesizer/ExactMcesSolver.h"

#include "Common/HwShareGroup.h"
#include "Fabric/Tech/Synthesizer/Parallel.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace loom::fabric::tech {

namespace {

using Clock = std::chrono::steady_clock;

struct NodeTuple {
  ::llvm::SmallVector<unsigned, 4> nodeIndexByGraph;
};

using TupleBucket = ::llvm::SmallVector<NodeTuple, 4>;
using SharedResultKey = ::std::pair<unsigned, unsigned>;
using PrivateNodeKey = ::std::pair<unsigned, unsigned>;

struct EdgeSig {
  unsigned operandIndex = 0;
  unsigned resultIndex = 0;
  unsigned width = 0;
  bool isBackEdge = false;

  bool operator==(const EdgeSig &other) const {
    return operandIndex == other.operandIndex &&
           resultIndex == other.resultIndex && width == other.width &&
           isBackEdge == other.isBackEdge;
  }

  bool operator<(const EdgeSig &other) const {
    return ::std::tie(operandIndex, resultIndex, width, isBackEdge) <
           ::std::tie(other.operandIndex, other.resultIndex, other.width,
                      other.isBackEdge);
  }
};

enum class DirectKind : uint8_t { None, BlockArgument, SharedResult };

struct DirectSource {
  DirectKind kind = DirectKind::None;
  unsigned index = 0;
  unsigned resultIndex = 0;

  bool operator==(const DirectSource &other) const {
    return kind == other.kind && index == other.index &&
           resultIndex == other.resultIndex;
  }
};

struct SearchState {
  ::llvm::SmallVector<NodeTuple, 8> selected;
  ::std::vector<::std::vector<int>> sharedIdByGraphNode;
  ::std::vector<::std::vector<int>> blockArgToBaseArg;
  double selectedSavings = 0.0;
};

struct SearchContext {
  ::llvm::ArrayRef<McsGraph> graphs;
  ::llvm::ArrayRef<TupleBucket> tuplesByBase;
  const ExactMcesSearchOptions *options = nullptr;
  double privateOpBaseline = 0.0;
  ::llvm::SmallVector<double, 8> futureSavingBound;
  ::llvm::SmallVector<::mlir::Type, 4> wrapperInputTypes;
};

struct ScoredCandidate {
  McesCandidate candidate;
  double estimatedCost = 0.0;
  std::uint64_t order = 0;
  std::string key;
};

struct SearchShard {
  int tupleIndex = -1;
  std::uint64_t ordinal = 0;
};

struct SearchShardResult {
  ::llvm::SmallVector<ScoredCandidate, 4> top;
  std::size_t generatedCandidates = 0;
  std::uint64_t nextOrder = 0;
  bool hitTimeout = false;
};

bool deadlineReached(const ExactMcesSearchOptions &options) {
  return Clock::now() >= options.deadline;
}

bool betterScore(const ScoredCandidate &lhs, const ScoredCandidate &rhs) {
  constexpr double kEpsilon = 1.0e-9;
  if (::std::abs(lhs.estimatedCost - rhs.estimatedCost) > kEpsilon)
    return lhs.estimatedCost < rhs.estimatedCost;
  if (lhs.order != rhs.order)
    return lhs.order < rhs.order;
  return lhs.key < rhs.key;
}

void sortTop(::llvm::SmallVectorImpl<ScoredCandidate> &top) {
  ::std::sort(top.begin(), top.end(), betterScore);
}

void keepTopCandidate(::llvm::SmallVectorImpl<ScoredCandidate> &top,
                      std::size_t cap, ScoredCandidate candidate) {
  if (cap == 0)
    return;
  if (top.size() < cap) {
    top.push_back(std::move(candidate));
    sortTop(top);
    return;
  }
  if (!betterScore(candidate, top.back()))
    return;
  top.back() = std::move(candidate);
  sortTop(top);
}

bool opNamesCompatible(::llvm::StringRef lhs, ::llvm::StringRef rhs) {
  return lhs == rhs || ::loom::common::sameShareGroup(lhs, rhs);
}

bool sameResultShape(const McsNode &lhs, const McsNode &rhs) {
  return lhs.resultWidths == rhs.resultWidths;
}

bool compatibleOperandSources(const McsOperand &lhs, const McsOperand &rhs) {
  if (lhs.width != rhs.width)
    return false;
  if (lhs.source.kind == McsValueKind::BlockArgument &&
      rhs.source.kind == McsValueKind::BlockArgument)
    return true;
  if (lhs.source.kind == McsValueKind::NodeResult &&
      rhs.source.kind == McsValueKind::NodeResult)
    return lhs.source.resultIndex == rhs.source.resultIndex;
  return true;
}

bool commutativeOperandsCompatible(const McsNode &lhs, const McsNode &rhs) {
  ::llvm::SmallVector<unsigned, 4> lhsWidths;
  ::llvm::SmallVector<unsigned, 4> rhsWidths;
  lhsWidths.reserve(lhs.operands.size());
  rhsWidths.reserve(rhs.operands.size());
  for (const McsOperand &operand : lhs.operands)
    lhsWidths.push_back(operand.width);
  for (const McsOperand &operand : rhs.operands)
    rhsWidths.push_back(operand.width);
  ::std::sort(lhsWidths.begin(), lhsWidths.end());
  ::std::sort(rhsWidths.begin(), rhsWidths.end());
  return lhsWidths == rhsWidths;
}

bool nodesCompatible(const McsNode &lhs, const McsNode &rhs) {
  if (!opNamesCompatible(lhs.opName, rhs.opName))
    return false;
  if (lhs.operands.size() != rhs.operands.size())
    return false;
  if (!sameResultShape(lhs, rhs))
    return false;
  if (lhs.commutative && rhs.commutative)
    return commutativeOperandsCompatible(lhs, rhs);
  for (auto pair : ::llvm::zip(lhs.operands, rhs.operands))
    if (!compatibleOperandSources(std::get<0>(pair), std::get<1>(pair)))
      return false;
  return true;
}

double opCost(::llvm::StringRef opName, unsigned width,
              const AreaWeights &weights) {
  if (opName == "dataflow.carry")
    return weights.carryPenalty * static_cast<double>(width);
  return baseUnitFor(::loom::common::findShareGroup(opName)) *
         (static_cast<double>(width) / 32.0);
}

double nodeOpCost(const McsNode &node, const AreaWeights &weights) {
  unsigned width = node.resultWidths.empty() ? 0 : node.resultWidths.front();
  return opCost(node.opName, width, weights);
}

::llvm::SmallVector<::mlir::Type, 4>
collectWrapperInputTypes(::llvm::ArrayRef<McsGraph> graphs) {
  ::llvm::SmallVector<::mlir::Type, 4> inputTypes;
  if (graphs.empty())
    return inputTypes;
  inputTypes.assign(graphs.front().blockArgTypes.begin(),
                    graphs.front().blockArgTypes.end());
  for (const McsGraph &graph : graphs) {
    for (auto indexed : ::llvm::enumerate(graph.blockArgTypes)) {
      if (indexed.index() >= inputTypes.size())
        inputTypes.push_back(indexed.value());
    }
  }
  return inputTypes;
}

double sharedTupleOpCost(::llvm::ArrayRef<McsGraph> graphs,
                         const NodeTuple &tuple, const AreaWeights &weights) {
  if (graphs.empty() || tuple.nodeIndexByGraph.empty())
    return 0.0;
  unsigned baseNodeIndex = tuple.nodeIndexByGraph.front();
  if (baseNodeIndex >= graphs.front().nodes.size())
    return 0.0;

  ::std::string firstName;
  for (auto indexed : ::llvm::enumerate(tuple.nodeIndexByGraph)) {
    unsigned graphIndex = static_cast<unsigned>(indexed.index());
    if (graphIndex >= graphs.size() ||
        indexed.value() >= graphs[graphIndex].nodes.size())
      continue;
    ::std::string name = graphs[graphIndex].nodes[indexed.value()].opName.str();
    if (firstName.empty() || name < firstName)
      firstName = std::move(name);
  }

  const McsNode &base = graphs.front().nodes[baseNodeIndex];
  unsigned width = base.resultWidths.empty() ? 0 : base.resultWidths.front();
  return opCost(firstName, width, weights);
}

double tupleSaving(::llvm::ArrayRef<McsGraph> graphs, const NodeTuple &tuple,
                   const AreaWeights &weights) {
  double privateCost = 0.0;
  for (auto indexed : ::llvm::enumerate(tuple.nodeIndexByGraph)) {
    unsigned graphIndex = static_cast<unsigned>(indexed.index());
    if (graphIndex >= graphs.size() ||
        indexed.value() >= graphs[graphIndex].nodes.size())
      continue;
    privateCost +=
        nodeOpCost(graphs[graphIndex].nodes[indexed.value()], weights);
  }
  return privateCost - sharedTupleOpCost(graphs, tuple, weights);
}

std::string tupleKey(const NodeTuple &tuple) {
  std::string out;
  ::llvm::raw_string_ostream os(out);
  for (unsigned index : tuple.nodeIndexByGraph)
    os << index << ',';
  return out;
}

std::string candidateKey(::llvm::ArrayRef<NodeTuple> tuples) {
  std::string out;
  ::llvm::raw_string_ostream os(out);
  for (const NodeTuple &tuple : tuples)
    os << tupleKey(tuple) << ';';
  return out;
}

std::string candidateLabel(::llvm::ArrayRef<NodeTuple> tuples,
                           double estimatedCost) {
  std::string out;
  ::llvm::raw_string_ostream os(out);
  os << "exact-mces cost-estimate=" << estimatedCost << " shared=";
  for (const NodeTuple &tuple : tuples) {
    os << '{';
    for (auto indexed : ::llvm::enumerate(tuple.nodeIndexByGraph)) {
      if (indexed.index() != 0)
        os << ',';
      os << indexed.value();
    }
    os << '}';
  }
  return out;
}

McesCandidate makeCandidate(::llvm::ArrayRef<NodeTuple> tuples,
                            double estimatedCost) {
  McesCandidate candidate;
  candidate.sharedNodes.reserve(tuples.size());
  for (auto indexed : ::llvm::enumerate(tuples)) {
    McesSharedNode shared;
    shared.id = static_cast<unsigned>(indexed.index());
    shared.nodeIndexByGraph = indexed.value().nodeIndexByGraph;
    candidate.sharedNodes.push_back(std::move(shared));
  }
  candidate.debugLabel = candidateLabel(tuples, estimatedCost);
  return candidate;
}

::llvm::SmallVector<EdgeSig, 4> edgeSignature(const McsGraph &graph,
                                              unsigned targetNodeIndex,
                                              unsigned sourceNodeIndex) {
  ::llvm::SmallVector<EdgeSig, 4> out;
  if (targetNodeIndex >= graph.nodes.size() ||
      sourceNodeIndex >= graph.nodes.size())
    return out;

  const McsNode &target = graph.nodes[targetNodeIndex];
  for (auto indexed : ::llvm::enumerate(target.operands)) {
    const McsOperand &operand = indexed.value();
    if (operand.source.kind != McsValueKind::NodeResult ||
        operand.source.nodeIndex != sourceNodeIndex)
      continue;
    EdgeSig sig;
    sig.operandIndex = target.commutative
                           ? std::numeric_limits<unsigned>::max()
                           : static_cast<unsigned>(indexed.index());
    sig.resultIndex = operand.source.resultIndex;
    sig.width = operand.width;
    sig.isBackEdge = operand.isBackEdge;
    out.push_back(sig);
  }
  ::std::sort(out.begin(), out.end());
  return out;
}

bool selfEdgePatternCompatible(::llvm::ArrayRef<McsGraph> graphs,
                               const NodeTuple &tuple) {
  if (graphs.empty() || tuple.nodeIndexByGraph.size() != graphs.size())
    return false;

  auto baseSelf = edgeSignature(graphs.front(), tuple.nodeIndexByGraph.front(),
                                tuple.nodeIndexByGraph.front());
  for (unsigned graphIndex = 1, graphCount = graphs.size();
       graphIndex < graphCount; ++graphIndex)
    if (edgeSignature(graphs[graphIndex], tuple.nodeIndexByGraph[graphIndex],
                      tuple.nodeIndexByGraph[graphIndex]) != baseSelf)
      return false;
  return true;
}

bool addBlockArgConstraint(const SearchContext &ctx,
                           ::std::vector<int> &mapping, unsigned graphIndex,
                           unsigned graphArg, unsigned baseArg) {
  auto graphs = ctx.graphs;
  if (graphs.empty() || graphIndex >= graphs.size() ||
      graphArg >= graphs[graphIndex].blockArgTypes.size() ||
      baseArg >= ctx.wrapperInputTypes.size() || graphArg >= mapping.size())
    return false;
  if (bitWidthOfMcsType(graphs[graphIndex].blockArgTypes[graphArg]) !=
      bitWidthOfMcsType(ctx.wrapperInputTypes[baseArg]))
    return false;

  int &slot = mapping[graphArg];
  if (slot >= 0)
    return slot == static_cast<int>(baseArg);
  slot = static_cast<int>(baseArg);
  return true;
}

bool addSourceBlockArgConstraint(const SearchContext &ctx,
                                 ::std::vector<int> &mapping,
                                 unsigned graphIndex, McsValueRef baseSource,
                                 McsValueRef graphSource) {
  if (baseSource.kind == McsValueKind::BlockArgument &&
      graphSource.kind == McsValueKind::BlockArgument)
    return addBlockArgConstraint(ctx, mapping, graphIndex, graphSource.argIndex,
                                 baseSource.argIndex);
  return true;
}

bool addSharedNodeBlockArgConstraints(const SearchContext &ctx,
                                      SearchState &state,
                                      const NodeTuple &tuple,
                                      unsigned graphIndex) {
  auto graphs = ctx.graphs;
  if (graphIndex == 0)
    return true;
  if (graphs.empty() || graphIndex >= graphs.size() ||
      graphIndex >= tuple.nodeIndexByGraph.size() ||
      graphIndex >= state.blockArgToBaseArg.size() ||
      tuple.nodeIndexByGraph.front() >= graphs.front().nodes.size() ||
      tuple.nodeIndexByGraph[graphIndex] >= graphs[graphIndex].nodes.size())
    return false;

  const McsNode &base = graphs.front().nodes[tuple.nodeIndexByGraph.front()];
  const McsNode &node =
      graphs[graphIndex].nodes[tuple.nodeIndexByGraph[graphIndex]];
  if (base.operands.size() != node.operands.size())
    return false;

  ::llvm::SmallVector<unsigned, 4> identity;
  identity.reserve(base.operands.size());
  for (unsigned i = 0, e = static_cast<unsigned>(base.operands.size()); i < e;
       ++i)
    identity.push_back(i);

  auto scorePermutation = [&](::llvm::ArrayRef<unsigned> permutation)
      -> ::std::optional<::std::pair<unsigned, ::std::vector<int>>> {
    ::std::vector<int> mapping = state.blockArgToBaseArg[graphIndex];
    unsigned score = 0;
    for (unsigned operandIndex = 0,
                  operandCount = static_cast<unsigned>(base.operands.size());
         operandIndex < operandCount; ++operandIndex) {
      unsigned graphOperandIndex = operandIndex;
      if (operandIndex < permutation.size())
        graphOperandIndex = permutation[operandIndex];
      if (graphOperandIndex >= node.operands.size())
        return std::nullopt;

      McsValueRef baseSource = base.operands[operandIndex].source;
      McsValueRef graphSource = node.operands[graphOperandIndex].source;
      if (baseSource.kind == McsValueKind::BlockArgument &&
          graphSource.kind == McsValueKind::BlockArgument &&
          baseSource.argIndex != graphSource.argIndex)
        ++score;
      if (!addSourceBlockArgConstraint(ctx, mapping, graphIndex, baseSource,
                                       graphSource))
        return std::nullopt;
    }
    return ::std::make_pair(score, std::move(mapping));
  };

  ::llvm::SmallVector<unsigned, 4> bestPermutation = identity;
  ::std::vector<int> bestMapping;
  unsigned bestScore = std::numeric_limits<unsigned>::max();
  ::llvm::SmallVector<unsigned, 4> permutation = identity;
  bool haveBest = false;

  if (base.commutative && node.commutative) {
    do {
      auto scored = scorePermutation(permutation);
      if (!scored.has_value())
        continue;
      if (!haveBest || scored->first < bestScore ||
          (scored->first == bestScore &&
           ::std::lexicographical_compare(
               permutation.begin(), permutation.end(), bestPermutation.begin(),
               bestPermutation.end()))) {
        haveBest = true;
        bestScore = scored->first;
        bestPermutation = permutation;
        bestMapping = std::move(scored->second);
      }
    } while (::std::next_permutation(permutation.begin(), permutation.end()));
  } else if (auto scored = scorePermutation(identity)) {
    haveBest = true;
    bestScore = scored->first;
    bestMapping = std::move(scored->second);
  }

  (void)bestScore;
  if (!haveBest)
    return false;
  state.blockArgToBaseArg[graphIndex] = std::move(bestMapping);
  return true;
}

bool completeBlockArgMap(const SearchContext &ctx, SearchState &state,
                         unsigned graphIndex) {
  auto graphs = ctx.graphs;
  if (graphs.empty() || graphIndex >= graphs.size() ||
      graphIndex >= state.blockArgToBaseArg.size())
    return false;
  const McsGraph &graph = graphs[graphIndex];
  if (ctx.wrapperInputTypes.empty())
    return false;
  unsigned wrapperArgCount = ctx.wrapperInputTypes.size();

  ::std::set<unsigned> used;
  for (int mapped : state.blockArgToBaseArg[graphIndex]) {
    if (mapped < 0)
      continue;
    unsigned baseArg = static_cast<unsigned>(mapped);
    if (baseArg >= wrapperArgCount || !used.insert(baseArg).second)
      return false;
  }

  for (unsigned graphArg = 0,
                argCount = static_cast<unsigned>(graph.blockArgTypes.size());
       graphArg < argCount; ++graphArg) {
    if (state.blockArgToBaseArg[graphIndex][graphArg] >= 0)
      continue;

    auto canMapTo = [&](unsigned baseArg) {
      return !used.count(baseArg) &&
             bitWidthOfMcsType(graph.blockArgTypes[graphArg]) ==
                 bitWidthOfMcsType(ctx.wrapperInputTypes[baseArg]);
    };

    unsigned chosen = wrapperArgCount;
    if (graphArg < wrapperArgCount && canMapTo(graphArg)) {
      chosen = graphArg;
    } else {
      for (unsigned baseArg = 0, baseCount = wrapperArgCount;
           baseArg < baseCount; ++baseArg) {
        if (canMapTo(baseArg)) {
          chosen = baseArg;
          break;
        }
      }
    }
    if (chosen >= wrapperArgCount)
      return false;
    state.blockArgToBaseArg[graphIndex][graphArg] = static_cast<int>(chosen);
    used.insert(chosen);
  }
  return true;
}

bool finalizeBlockArgMaps(const SearchContext &ctx, SearchState &state) {
  auto graphs = ctx.graphs;
  if (graphs.empty())
    return false;

  for (unsigned graphIndex = 1, graphCount = graphs.size();
       graphIndex < graphCount; ++graphIndex) {
    if (graphs[graphIndex].yieldSources.size() !=
        graphs.front().yieldSources.size())
      return false;
    for (unsigned yieldIndex = 0, yieldCount = static_cast<unsigned>(
                                      graphs.front().yieldSources.size());
         yieldIndex < yieldCount; ++yieldIndex) {
      if (!addSourceBlockArgConstraint(
              ctx, state.blockArgToBaseArg[graphIndex], graphIndex,
              graphs.front().yieldSources[yieldIndex],
              graphs[graphIndex].yieldSources[yieldIndex]))
        return false;
    }
  }

  for (unsigned graphIndex = 0, graphCount = graphs.size();
       graphIndex < graphCount; ++graphIndex)
    if (!completeBlockArgMap(ctx, state, graphIndex))
      return false;
  return true;
}

SearchState makeInitialState(::llvm::ArrayRef<McsGraph> graphs) {
  SearchState state;
  state.sharedIdByGraphNode.reserve(graphs.size());
  state.blockArgToBaseArg.reserve(graphs.size());
  for (const McsGraph &graph : graphs) {
    state.sharedIdByGraphNode.push_back(
        ::std::vector<int>(graph.nodes.size(), -1));
    state.blockArgToBaseArg.push_back(
        ::std::vector<int>(graph.blockArgTypes.size(), -1));
  }
  if (!graphs.empty()) {
    for (unsigned argIndex = 0, argCount = static_cast<unsigned>(
                                    graphs.front().blockArgTypes.size());
         argIndex < argCount; ++argIndex)
      state.blockArgToBaseArg.front()[argIndex] = static_cast<int>(argIndex);
  }
  return state;
}

bool tupleDisjoint(const SearchState &state, const NodeTuple &tuple) {
  if (state.sharedIdByGraphNode.size() != tuple.nodeIndexByGraph.size())
    return false;
  for (auto indexed : ::llvm::enumerate(tuple.nodeIndexByGraph)) {
    unsigned graphIndex = static_cast<unsigned>(indexed.index());
    if (graphIndex >= state.sharedIdByGraphNode.size() ||
        indexed.value() >= state.sharedIdByGraphNode[graphIndex].size())
      return false;
    if (state.sharedIdByGraphNode[graphIndex][indexed.value()] >= 0)
      return false;
  }
  return true;
}

bool tryAddTuple(const SearchContext &ctx, SearchState &state,
                 const NodeTuple &tuple) {
  if (tuple.nodeIndexByGraph.size() != ctx.graphs.size())
    return false;
  if (!tupleDisjoint(state, tuple))
    return false;
  if (!selfEdgePatternCompatible(ctx.graphs, tuple))
    return false;

  for (unsigned graphIndex = 1, graphCount = ctx.graphs.size();
       graphIndex < graphCount; ++graphIndex)
    if (!addSharedNodeBlockArgConstraints(ctx, state, tuple, graphIndex))
      return false;

  unsigned sharedId = static_cast<unsigned>(state.selected.size());
  for (auto indexed : ::llvm::enumerate(tuple.nodeIndexByGraph)) {
    unsigned graphIndex = static_cast<unsigned>(indexed.index());
    state.sharedIdByGraphNode[graphIndex][indexed.value()] =
        static_cast<int>(sharedId);
  }
  state.selected.push_back(tuple);
  state.selectedSavings +=
      tupleSaving(ctx.graphs, tuple, ctx.options->costWeights);
  return true;
}

DirectSource canonicalDirectSource(const SearchState &state,
                                   unsigned graphIndex, McsValueRef source) {
  if (graphIndex >= state.blockArgToBaseArg.size() ||
      graphIndex >= state.sharedIdByGraphNode.size())
    return {};

  if (source.kind == McsValueKind::BlockArgument) {
    if (source.argIndex >= state.blockArgToBaseArg[graphIndex].size())
      return {};
    int mapped = state.blockArgToBaseArg[graphIndex][source.argIndex];
    if (mapped < 0)
      return {};
    return {DirectKind::BlockArgument, static_cast<unsigned>(mapped), 0};
  }

  if (source.nodeIndex >= state.sharedIdByGraphNode[graphIndex].size())
    return {};
  int sharedId = state.sharedIdByGraphNode[graphIndex][source.nodeIndex];
  if (sharedId < 0)
    return {};
  return {DirectKind::SharedResult, static_cast<unsigned>(sharedId),
          source.resultIndex};
}

bool directSourcesCommon(const SearchState &state,
                         ::llvm::ArrayRef<McsValueRef> sources) {
  if (sources.empty() || sources.size() != state.sharedIdByGraphNode.size())
    return false;
  DirectSource first = canonicalDirectSource(state, 0, sources.front());
  if (first.kind == DirectKind::None)
    return false;
  for (unsigned graphIndex = 1, graphCount = sources.size();
       graphIndex < graphCount; ++graphIndex)
    if (!(canonicalDirectSource(state, graphIndex, sources[graphIndex]) ==
          first))
      return false;
  return true;
}

unsigned sourceWidth(const McsGraph &graph, McsValueRef source) {
  if (source.kind == McsValueKind::BlockArgument) {
    if (source.argIndex >= graph.blockArgTypes.size())
      return 0;
    return bitWidthOfMcsType(graph.blockArgTypes[source.argIndex]);
  }
  if (source.nodeIndex >= graph.nodes.size())
    return 0;
  const McsNode &node = graph.nodes[source.nodeIndex];
  if (source.resultIndex >= node.resultWidths.size())
    return 0;
  return node.resultWidths[source.resultIndex];
}

unsigned sharedResultWidth(::llvm::ArrayRef<McsGraph> graphs,
                           const SearchState &state, unsigned sharedId,
                           unsigned resultIndex) {
  if (graphs.empty() || sharedId >= state.selected.size() ||
      state.selected[sharedId].nodeIndexByGraph.empty())
    return 0;
  unsigned nodeIndex = state.selected[sharedId].nodeIndexByGraph.front();
  if (nodeIndex >= graphs.front().nodes.size())
    return 0;
  const McsNode &node = graphs.front().nodes[nodeIndex];
  if (resultIndex >= node.resultWidths.size())
    return 0;
  return node.resultWidths[resultIndex];
}

void collectSharedRefs(::llvm::ArrayRef<McsGraph> graphs,
                       const SearchState &state, unsigned graphIndex,
                       McsValueRef source,
                       ::std::set<SharedResultKey> &sharedRefs,
                       ::std::set<PrivateNodeKey> &seenPrivate) {
  if (source.kind == McsValueKind::BlockArgument)
    return;
  if (graphIndex >= graphs.size() ||
      graphIndex >= state.sharedIdByGraphNode.size() ||
      source.nodeIndex >= graphs[graphIndex].nodes.size() ||
      source.nodeIndex >= state.sharedIdByGraphNode[graphIndex].size())
    return;

  int sharedId = state.sharedIdByGraphNode[graphIndex][source.nodeIndex];
  if (sharedId >= 0) {
    sharedRefs.insert({static_cast<unsigned>(sharedId), source.resultIndex});
    return;
  }

  PrivateNodeKey key{graphIndex, source.nodeIndex};
  if (!seenPrivate.insert(key).second)
    return;
  const McsNode &node = graphs[graphIndex].nodes[source.nodeIndex];
  for (const McsOperand &operand : node.operands)
    collectSharedRefs(graphs, state, graphIndex, operand.source, sharedRefs,
                      seenPrivate);
}

double adapterCost(::llvm::ArrayRef<McsGraph> graphs, const SearchState &state,
                   ::llvm::ArrayRef<McsValueRef> sources, unsigned width,
                   const AreaWeights &weights,
                   ::std::set<SharedResultKey> &demuxedShared) {
  if (directSourcesCommon(state, sources))
    return 0.0;

  double cost = 0.0;
  ::std::set<SharedResultKey> sharedRefs;
  for (auto indexed : ::llvm::enumerate(sources)) {
    ::std::set<PrivateNodeKey> seenPrivate;
    collectSharedRefs(graphs, state, static_cast<unsigned>(indexed.index()),
                      indexed.value(), sharedRefs, seenPrivate);
  }
  if (graphs.size() >= 2) {
    for (SharedResultKey key : sharedRefs) {
      if (!demuxedShared.insert(key).second)
        continue;
      unsigned demuxWidth =
          sharedResultWidth(graphs, state, key.first, key.second);
      cost += weights.demuxPenalty * static_cast<double>(graphs.size()) *
              static_cast<double>(demuxWidth);
    }
  }
  if (graphs.size() >= 2)
    cost += weights.muxPenalty * static_cast<double>(graphs.size()) *
            static_cast<double>(width);
  return cost;
}

bool directSourceCompatibleForPermutation(const SearchState &state,
                                          unsigned graphIndex,
                                          McsValueRef baseSource,
                                          McsValueRef graphSource) {
  DirectSource base = canonicalDirectSource(state, 0, baseSource);
  DirectSource candidate =
      canonicalDirectSource(state, graphIndex, graphSource);
  return base.kind != DirectKind::None && base == candidate;
}

::llvm::SmallVector<unsigned, 4>
operandPermutationForCost(::llvm::ArrayRef<McsGraph> graphs,
                          const SearchState &state, const NodeTuple &tuple,
                          unsigned graphIndex) {
  ::llvm::SmallVector<unsigned, 4> identity;
  if (graphs.empty() || graphIndex >= graphs.size() ||
      tuple.nodeIndexByGraph.empty() ||
      tuple.nodeIndexByGraph.front() >= graphs.front().nodes.size() ||
      graphIndex >= tuple.nodeIndexByGraph.size() ||
      tuple.nodeIndexByGraph[graphIndex] >= graphs[graphIndex].nodes.size())
    return identity;

  const McsNode &base = graphs.front().nodes[tuple.nodeIndexByGraph.front()];
  const McsNode &node =
      graphs[graphIndex].nodes[tuple.nodeIndexByGraph[graphIndex]];
  unsigned operandCount = static_cast<unsigned>(base.operands.size());
  identity.reserve(operandCount);
  for (unsigned i = 0; i < operandCount; ++i)
    identity.push_back(i);
  if (graphIndex == 0 || !base.commutative || !node.commutative ||
      node.operands.size() != operandCount)
    return identity;

  ::llvm::SmallVector<unsigned, 4> permutation = identity;
  do {
    bool ok = true;
    for (unsigned i = 0; i < operandCount; ++i) {
      if (base.operands[i].width != node.operands[permutation[i]].width ||
          !directSourceCompatibleForPermutation(
              state, graphIndex, base.operands[i].source,
              node.operands[permutation[i]].source)) {
        ok = false;
        break;
      }
    }
    if (ok)
      return permutation;
  } while (::std::next_permutation(permutation.begin(), permutation.end()));
  return identity;
}

double estimateCandidateCost(const SearchContext &ctx,
                             const SearchState &state) {
  double cost = ctx.privateOpBaseline - state.selectedSavings;
  if (cost < 0.0)
    cost = 0.0;

  ::std::set<SharedResultKey> demuxedShared;
  for (const NodeTuple &tuple : state.selected) {
    if (tuple.nodeIndexByGraph.empty() ||
        tuple.nodeIndexByGraph.front() >= ctx.graphs.front().nodes.size())
      continue;
    const McsNode &base =
        ctx.graphs.front().nodes[tuple.nodeIndexByGraph.front()];
    for (unsigned operandIndex = 0,
                  operandCount = static_cast<unsigned>(base.operands.size());
         operandIndex < operandCount; ++operandIndex) {
      ::llvm::SmallVector<McsValueRef, 4> sources;
      sources.reserve(ctx.graphs.size());
      for (unsigned graphIndex = 0, graphCount = ctx.graphs.size();
           graphIndex < graphCount; ++graphIndex) {
        if (graphIndex >= tuple.nodeIndexByGraph.size() ||
            tuple.nodeIndexByGraph[graphIndex] >=
                ctx.graphs[graphIndex].nodes.size())
          return std::numeric_limits<double>::infinity();
        const McsNode &node =
            ctx.graphs[graphIndex].nodes[tuple.nodeIndexByGraph[graphIndex]];
        auto permutation =
            operandPermutationForCost(ctx.graphs, state, tuple, graphIndex);
        unsigned sourceOperandIndex = operandIndex;
        if (operandIndex < permutation.size())
          sourceOperandIndex = permutation[operandIndex];
        if (sourceOperandIndex >= node.operands.size())
          return std::numeric_limits<double>::infinity();
        sources.push_back(node.operands[sourceOperandIndex].source);
      }
      cost += adapterCost(ctx.graphs, state, sources,
                          base.operands[operandIndex].width,
                          ctx.options->costWeights, demuxedShared);
    }
  }

  const McsGraph &base = ctx.graphs.front();
  for (unsigned yieldIndex = 0,
                yieldCount = static_cast<unsigned>(base.yieldSources.size());
       yieldIndex < yieldCount; ++yieldIndex) {
    ::llvm::SmallVector<McsValueRef, 4> sources;
    sources.reserve(ctx.graphs.size());
    for (const McsGraph &graph : ctx.graphs) {
      if (yieldIndex >= graph.yieldSources.size())
        return std::numeric_limits<double>::infinity();
      sources.push_back(graph.yieldSources[yieldIndex]);
    }
    cost += adapterCost(ctx.graphs, state, sources,
                        sourceWidth(base, base.yieldSources[yieldIndex]),
                        ctx.options->costWeights, demuxedShared);
  }

  return cost;
}

bool shouldPrune(const SearchContext &ctx, const SearchShardResult &result,
                 const SearchState &state, unsigned basePos) {
  if (!ctx.options || ctx.options->candidateCap == 0 ||
      result.top.size() < ctx.options->candidateCap)
    return false;
  double futureSaving = 0.0;
  if (basePos < ctx.futureSavingBound.size())
    futureSaving = ctx.futureSavingBound[basePos];
  double lowerBound =
      ctx.privateOpBaseline - state.selectedSavings - futureSaving;
  if (lowerBound < 0.0)
    lowerBound = 0.0;
  constexpr double kEpsilon = 1.0e-9;
  return lowerBound > result.top.back().estimatedCost + kEpsilon;
}

void finishCandidate(const SearchContext &ctx, SearchShardResult &result,
                     SearchState state) {
  if (state.selected.empty())
    return;
  if (!finalizeBlockArgMaps(ctx, state))
    return;

  double estimatedCost = estimateCandidateCost(ctx, state);
  if (!std::isfinite(estimatedCost))
    return;

  ++result.generatedCandidates;
  std::string key = candidateKey(state.selected);
  ScoredCandidate scored;
  scored.estimatedCost = estimatedCost;
  scored.order = result.nextOrder++;
  scored.key = key;
  scored.candidate = makeCandidate(state.selected, estimatedCost);
  keepTopCandidate(result.top, ctx.options->candidateCap, std::move(scored));
}

void dfs(const SearchContext &ctx, SearchShardResult &result,
         const SearchState &state, unsigned basePos) {
  if (deadlineReached(*ctx.options)) {
    result.hitTimeout = true;
    return;
  }
  if (basePos >= ctx.tuplesByBase.size()) {
    finishCandidate(ctx, result, state);
    return;
  }
  if (shouldPrune(ctx, result, state, basePos))
    return;

  for (const NodeTuple &tuple : ctx.tuplesByBase[basePos]) {
    SearchState next = state;
    if (!tryAddTuple(ctx, next, tuple))
      continue;
    dfs(ctx, result, next, basePos + 1);
    if (result.hitTimeout)
      return;
  }

  dfs(ctx, result, state, basePos + 1);
}

SearchShardResult runShard(const SearchContext &ctx, const SearchShard &shard) {
  constexpr std::uint64_t kShardOrderStride = 1ULL << 48;
  SearchShardResult result;
  result.nextOrder = shard.ordinal * kShardOrderStride;
  SearchState state = makeInitialState(ctx.graphs);
  unsigned basePos = 0;

  if (!ctx.tuplesByBase.empty()) {
    if (shard.tupleIndex >= 0) {
      unsigned tupleIndex = static_cast<unsigned>(shard.tupleIndex);
      if (tupleIndex >= ctx.tuplesByBase.front().size())
        return result;
      if (!tryAddTuple(ctx, state, ctx.tuplesByBase.front()[tupleIndex]))
        return result;
    }
    basePos = 1;
  }

  dfs(ctx, result, state, basePos);
  return result;
}

template <typename Fn>
bool enumerateTuplesForBase(::llvm::ArrayRef<McsGraph> graphs,
                            unsigned baseNodeIndex,
                            const ExactMcesSearchOptions &options,
                            Fn &&handleTuple) {
  const McsNode &baseNode = graphs.front().nodes[baseNodeIndex];
  ::llvm::SmallVector<::llvm::SmallVector<unsigned, 8>, 4> choices;
  choices.resize(graphs.size());
  choices.front().push_back(baseNodeIndex);

  for (unsigned graphIndex = 1, graphCount = graphs.size();
       graphIndex < graphCount; ++graphIndex) {
    for (const McsNode &node : graphs[graphIndex].nodes)
      if (nodesCompatible(baseNode, node))
        choices[graphIndex].push_back(node.index);
    ::std::sort(choices[graphIndex].begin(), choices[graphIndex].end());
    if (choices[graphIndex].empty())
      return true;
  }

  NodeTuple cur;
  cur.nodeIndexByGraph.resize(graphs.size());
  cur.nodeIndexByGraph.front() = baseNodeIndex;

  auto recurse = [&](auto &&self, unsigned graphIndex) -> bool {
    if (deadlineReached(options))
      return false;
    if (graphIndex == graphs.size())
      return handleTuple(cur);
    for (unsigned nodeIndex : choices[graphIndex]) {
      cur.nodeIndexByGraph[graphIndex] = nodeIndex;
      if (!self(self, graphIndex + 1))
        return false;
    }
    return true;
  };
  return recurse(recurse, 1);
}

bool buildTupleBuckets(::llvm::ArrayRef<McsGraph> graphs,
                       const ExactMcesSearchOptions &options,
                       ::llvm::SmallVectorImpl<TupleBucket> &tuplesByBase) {
  tuplesByBase.clear();
  tuplesByBase.resize(graphs.front().nodes.size());

  for (const McsNode &node : graphs.front().nodes) {
    bool completed = enumerateTuplesForBase(
        graphs, node.index, options, [&](const NodeTuple &tuple) {
          if (node.index >= tuplesByBase.size())
            return false;
          tuplesByBase[node.index].push_back(tuple);
          return true;
        });
    if (!completed)
      return false;
    if (node.index < tuplesByBase.size())
      ::std::sort(tuplesByBase[node.index].begin(),
                  tuplesByBase[node.index].end(),
                  [](const NodeTuple &lhs, const NodeTuple &rhs) {
                    return tupleKey(lhs) < tupleKey(rhs);
                  });
  }
  return true;
}

double privateBaselineCost(::llvm::ArrayRef<McsGraph> graphs,
                           const AreaWeights &weights) {
  double total = 0.0;
  for (const McsGraph &graph : graphs)
    for (const McsNode &node : graph.nodes)
      total += nodeOpCost(node, weights);
  return total;
}

::llvm::SmallVector<double, 8>
computeFutureSavingBound(::llvm::ArrayRef<McsGraph> graphs,
                         ::llvm::ArrayRef<TupleBucket> tuplesByBase,
                         const AreaWeights &weights) {
  ::llvm::SmallVector<double, 8> perBase;
  perBase.resize(tuplesByBase.size());
  for (auto indexed : ::llvm::enumerate(tuplesByBase)) {
    double bestSaving = 0.0;
    for (const NodeTuple &tuple : indexed.value())
      bestSaving = ::std::max(bestSaving, tupleSaving(graphs, tuple, weights));
    perBase[indexed.index()] = bestSaving;
  }

  ::llvm::SmallVector<double, 8> suffix;
  suffix.resize(perBase.size() + 1);
  suffix[perBase.size()] = 0.0;
  for (std::size_t i = perBase.size(); i > 0; --i)
    suffix[i - 1] = suffix[i] + perBase[i - 1];
  return suffix;
}

::llvm::SmallVector<SearchShard, 8>
makeShards(::llvm::ArrayRef<TupleBucket> tuplesByBase) {
  ::llvm::SmallVector<SearchShard, 8> shards;
  if (tuplesByBase.empty())
    return shards;
  for (auto indexed : ::llvm::enumerate(tuplesByBase.front())) {
    SearchShard shard;
    shard.tupleIndex = static_cast<int>(indexed.index());
    shard.ordinal = static_cast<std::uint64_t>(indexed.index());
    shards.push_back(shard);
  }
  SearchShard skipShard;
  skipShard.tupleIndex = -1;
  skipShard.ordinal = static_cast<std::uint64_t>(shards.size());
  shards.push_back(skipShard);
  return shards;
}

ExactMcesSearchResult
mergeShardResults(::llvm::ArrayRef<SearchShardResult> shardResults,
                  std::size_t candidateCap) {
  ExactMcesSearchResult result;
  ::std::set<std::string> seenKeys;
  ::llvm::SmallVector<ScoredCandidate, 4> mergedTop;

  for (const SearchShardResult &shardResult : shardResults) {
    result.generatedCandidates += shardResult.generatedCandidates;
    result.hitTimeout = result.hitTimeout || shardResult.hitTimeout;
    for (const ScoredCandidate &candidate : shardResult.top) {
      if (!seenKeys.insert(candidate.key).second)
        continue;
      keepTopCandidate(mergedTop, candidateCap, candidate);
    }
  }

  sortTop(mergedTop);
  result.candidates.reserve(mergedTop.size());
  for (ScoredCandidate &candidate : mergedTop)
    result.candidates.push_back(std::move(candidate.candidate));

  result.hitCap = candidateCap == 0 ||
                  result.generatedCandidates > result.candidates.size();
  result.provedOptimal = !result.hitTimeout && !result.hitCap;
  return result;
}

} // namespace

ExactMcesSearchResult
ExactMcesSolver::enumerate(::llvm::ArrayRef<McsGraph> graphs,
                           const ExactMcesSearchOptions &options) const {
  ExactMcesSearchResult result;
  if (graphs.empty() || options.candidateCap == 0) {
    result.hitCap = options.candidateCap == 0;
    return result;
  }
  if (deadlineReached(options)) {
    result.hitTimeout = true;
    return result;
  }
  if (graphs.front().nodes.empty()) {
    result.provedOptimal = true;
    return result;
  }

  ::llvm::SmallVector<TupleBucket, 8> tuplesByBase;
  if (!buildTupleBuckets(graphs, options, tuplesByBase)) {
    result.hitTimeout = true;
    return result;
  }
  if (deadlineReached(options)) {
    result.hitTimeout = true;
    return result;
  }

  SearchContext ctx;
  ctx.graphs = graphs;
  ctx.tuplesByBase = tuplesByBase;
  ctx.options = &options;
  ctx.privateOpBaseline = privateBaselineCost(graphs, options.costWeights);
  ctx.futureSavingBound =
      computeFutureSavingBound(graphs, tuplesByBase, options.costWeights);
  ctx.wrapperInputTypes = collectWrapperInputTypes(graphs);

  ::llvm::SmallVector<SearchShard, 8> shards = makeShards(tuplesByBase);
  if (shards.empty()) {
    SearchShard shard;
    shards.push_back(shard);
  }

  ::llvm::SmallVector<SearchShardResult, 8> shardResults;
  if (options.workers > 1 && shards.size() > 1) {
    WorkerPool pool(options.workers);
    shardResults = pool.parallelMap<SearchShard, SearchShardResult>(
        shards, [&](const SearchShard &shard) { return runShard(ctx, shard); });
  } else {
    shardResults.reserve(shards.size());
    for (const SearchShard &shard : shards)
      shardResults.push_back(runShard(ctx, shard));
  }

  return mergeShardResults(shardResults, options.candidateCap);
}

} // namespace loom::fabric::tech
