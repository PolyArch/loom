#include "Fabric/Tech/Synthesizer/McesSolver.h"

#include "Common/HwShareGroup.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <chrono>
#include <set>
#include <string>
#include <vector>

namespace loom::fabric::tech {

namespace {

struct NodeTuple {
  ::llvm::SmallVector<unsigned, 4> nodeIndexByGraph;
};

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
  std::sort(lhsWidths.begin(), lhsWidths.end());
  std::sort(rhsWidths.begin(), rhsWidths.end());
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

std::string candidateLabel(::llvm::ArrayRef<NodeTuple> tuples) {
  std::string out;
  ::llvm::raw_string_ostream os(out);
  os << "shared=";
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

McesCandidate makeCandidate(::llvm::ArrayRef<NodeTuple> tuples) {
  McesCandidate candidate;
  candidate.sharedNodes.reserve(tuples.size());
  for (auto indexed : ::llvm::enumerate(tuples)) {
    McesSharedNode shared;
    shared.id = static_cast<unsigned>(indexed.index());
    shared.nodeIndexByGraph = indexed.value().nodeIndexByGraph;
    candidate.sharedNodes.push_back(std::move(shared));
  }
  candidate.debugLabel = candidateLabel(tuples);
  return candidate;
}

using Clock = std::chrono::steady_clock;

bool deadlineReached(const McesSearchOptions &options) {
  return Clock::now() >= options.deadline;
}

template <typename Fn>
bool enumerateTuplesForBase(::llvm::ArrayRef<McsGraph> graphs,
                            unsigned baseNodeIndex,
                            const McesSearchOptions &options,
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

bool tupleDisjoint(const ::std::vector<::std::set<unsigned>> &used,
                   const NodeTuple &tuple) {
  if (used.size() != tuple.nodeIndexByGraph.size())
    return false;
  for (auto indexed : ::llvm::enumerate(tuple.nodeIndexByGraph))
    if (used[indexed.index()].count(indexed.value()))
      return false;
  return true;
}

void markTupleUsed(::std::vector<::std::set<unsigned>> &used,
                   const NodeTuple &tuple) {
  for (auto indexed : ::llvm::enumerate(tuple.nodeIndexByGraph))
    used[indexed.index()].insert(indexed.value());
}

void unmarkTupleUsed(::std::vector<::std::set<unsigned>> &used,
                     const NodeTuple &tuple) {
  for (auto indexed : ::llvm::enumerate(tuple.nodeIndexByGraph))
    used[indexed.index()].erase(indexed.value());
}

} // namespace

McesSearchResult
McesSolver::enumerate(::llvm::ArrayRef<McsGraph> graphs,
                      const McesSearchOptions &options) const {
  McesSearchResult result;
  if (graphs.empty() || options.candidateCap == 0) {
    result.hitCap = options.candidateCap == 0;
    return result;
  }
  if (deadlineReached(options)) {
    result.hitTimeout = true;
    return result;
  }

  ::llvm::SmallVector<::llvm::SmallVector<NodeTuple, 4>, 8> tuplesByBase;
  tuplesByBase.resize(graphs.front().nodes.size());

  bool completed = true;
  const std::size_t tupleLimitPerBase = std::max<std::size_t>(
      1, options.candidateCap == 0 ? 1 : options.candidateCap);
  for (const McsNode &node : graphs.front().nodes) {
    bool hitTupleLimit = false;
    completed = enumerateTuplesForBase(
        graphs, node.index, options, [&](const NodeTuple &tuple) {
          if (node.index >= tuplesByBase.size())
            return false;
          tuplesByBase[node.index].push_back(tuple);
          if (tuplesByBase[node.index].size() >= tupleLimitPerBase) {
            hitTupleLimit = true;
            return false;
          }
          return true;
        });
    if (!completed && hitTupleLimit)
      completed = true;
    if (!completed)
      break;
  }
  if (!completed) {
    result.hitTimeout = true;
    return result;
  }

  ::std::set<std::string> seenCandidates;
  auto addCandidate = [&](::llvm::ArrayRef<NodeTuple> candidateTuples) -> bool {
    if (candidateTuples.empty())
      return true;
    if (result.candidates.size() >= options.candidateCap) {
      result.hitCap = result.candidates.size() >= options.candidateCap;
      return false;
    }
    std::string key = candidateKey(candidateTuples);
    if (!seenCandidates.insert(key).second)
      return true;
    result.candidates.push_back(makeCandidate(candidateTuples));
    ++result.generatedCandidates;
    result.hitCap = result.candidates.size() >= options.candidateCap;
    return !result.hitCap;
  };

  ::llvm::SmallVector<NodeTuple, 8> current;
  ::std::vector<::std::set<unsigned>> used(graphs.size());
  auto dfs = [&](auto &&self, unsigned basePos) -> bool {
    if (deadlineReached(options)) {
      result.hitTimeout = true;
      return false;
    }
    if (basePos >= tuplesByBase.size())
      return addCandidate(current);

    for (const NodeTuple &tuple : tuplesByBase[basePos]) {
      if (!tupleDisjoint(used, tuple))
        continue;
      markTupleUsed(used, tuple);
      current.push_back(tuple);
      if (!self(self, basePos + 1))
        return false;
      current.pop_back();
      unmarkTupleUsed(used, tuple);
    }
    return self(self, basePos + 1);
  };

  dfs(dfs, 0);

  return result;
}

::llvm::SmallVector<McesCandidate, 4>
McesSolver::enumerate(::llvm::ArrayRef<McsGraph> graphs,
                      std::size_t cap) const {
  McesSearchOptions options;
  options.candidateCap = cap;
  options.deadline = Clock::time_point::max();
  return enumerate(graphs, options).candidates;
}

} // namespace loom::fabric::tech
