#include "Fabric/IR/ConfiguredFunction.h"

#include "ConfiguredFunctionInternal.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <string>
#include <utility>

namespace fabric {
namespace {

using ::fabric::detail::sameAttributes;
using ::fabric::detail::sameType;

struct CoverageAssignments {
  ::llvm::SmallVector<int, 8> nodeMap;
  ::llvm::SmallVector<int, 8> reverseNodeMap;
  ::llvm::DenseMap<unsigned, unsigned> inputMap;
  ::llvm::DenseMap<unsigned, unsigned> reverseInputMap;
  ::llvm::SmallVector<::llvm::SmallVector<int, 4>, 4> laneMap;
  ::llvm::SmallVector<::llvm::SmallVector<int, 4>, 4> reverseLaneMap;
};

struct OutputConstraintState {
  CoverageAssignments assignments;
  ::llvm::SmallVector<::llvm::SmallVector<unsigned, 8>, 8> domains;
  ::llvm::SmallVector<int, 8> outputMap;
  ::llvm::SmallVector<int, 8> candidateOwner;
};

struct PairedOutputGroup {
  unsigned patternNode = 0;
  ::llvm::SmallVector<unsigned, 8> outputs;
  bool normalizesLanePermutations = false;
};

struct PairedOutputGroupState {
  CoverageAssignments outsideAssignments;
  ::llvm::SmallVector<unsigned, 8> selectedOutputs;
};

using PairedOutputGroupStates =
    ::llvm::DenseMap<unsigned, ::llvm::SmallVector<PairedOutputGroupState, 2>>;

class CoverageMatcher {
public:
  CoverageMatcher(const ConfiguredFunction &pattern,
                  const ConfiguredFunction &candidate)
      : pattern(pattern), candidate(candidate) {}

  bool run(ConfiguredFunctionMatch *witness) {
    if (pattern.nodes.size() != candidate.nodes.size() ||
        pattern.inputs.size() > candidate.inputs.size() ||
        pattern.outputs.size() > candidate.outputs.size())
      return false;

    CoverageAssignments state;
    state.nodeMap.assign(pattern.nodes.size(), -1);
    state.reverseNodeMap.assign(candidate.nodes.size(), -1);
    state.laneMap.resize(pattern.nodes.size());
    state.reverseLaneMap.resize(pattern.nodes.size());
    ::llvm::SmallVector<unsigned, 8> outputMap(pattern.outputs.size(), 0);
    if (!matchOutputs(state, outputMap))
      return false;

    if (witness)
      buildWitness(state, outputMap, *witness);
    return true;
  }

private:
  const ConfiguredFunction &pattern;
  const ConfiguredFunction &candidate;

  bool isPairedSync(unsigned patternNode, unsigned candidateNode) const {
    if (patternNode >= pattern.nodes.size() ||
        candidateNode >= candidate.nodes.size())
      return false;
    const ConfiguredFunctionNode &lhs = pattern.nodes[patternNode];
    const ConfiguredFunctionNode &rhs = candidate.nodes[candidateNode];
    return lhs.operationName == "dataflow.sync" &&
           rhs.operationName == "dataflow.sync" && !rhs.pairedLanes.empty();
  }

  bool matchInput(unsigned patternPort, unsigned candidatePort,
                  CoverageAssignments &state) const {
    auto existing = state.inputMap.find(patternPort);
    if (existing != state.inputMap.end())
      return existing->second == candidatePort;
    if (state.reverseInputMap.count(candidatePort))
      return false;
    state.inputMap[patternPort] = candidatePort;
    state.reverseInputMap[candidatePort] = patternPort;
    return true;
  }

  bool matchPairedLane(unsigned patternNode, unsigned patternLane,
                       unsigned candidateNode, unsigned candidateLane,
                       CoverageAssignments &state) const {
    if (!isPairedSync(patternNode, candidateNode))
      return false;
    const ConfiguredFunctionNode &lhs = pattern.nodes[patternNode];
    const ConfiguredFunctionNode &rhs = candidate.nodes[candidateNode];
    if (patternLane >= lhs.functionType.getNumInputs() ||
        patternLane >= lhs.functionType.getNumResults() ||
        candidateLane >= rhs.pairedLanes.size() ||
        candidateLane >= rhs.functionType.getNumInputs() ||
        candidateLane >= rhs.functionType.getNumResults() ||
        !sameType(lhs.functionType.getInput(patternLane),
                  rhs.functionType.getInput(candidateLane)) ||
        !sameType(lhs.functionType.getResult(patternLane),
                  rhs.functionType.getResult(candidateLane)))
      return false;

    int &mapped = state.laneMap[patternNode][patternLane];
    if (mapped >= 0)
      return static_cast<unsigned>(mapped) == candidateLane;
    int &reverse = state.reverseLaneMap[patternNode][candidateLane];
    if (reverse >= 0)
      return false;
    mapped = static_cast<int>(candidateLane);
    reverse = static_cast<int>(patternLane);
    return matchValue(lhs.operands[patternLane], rhs.operands[candidateLane],
                      state);
  }

  bool matchPairedSync(unsigned patternNode, unsigned candidateNode,
                       CoverageAssignments &state) const {
    if (!isPairedSync(patternNode, candidateNode))
      return false;
    if (state.nodeMap[patternNode] >= 0)
      return static_cast<unsigned>(state.nodeMap[patternNode]) == candidateNode;
    if (state.reverseNodeMap[candidateNode] >= 0)
      return false;

    const ConfiguredFunctionNode &lhs = pattern.nodes[patternNode];
    const ConfiguredFunctionNode &rhs = candidate.nodes[candidateNode];
    const unsigned softwareLanes = lhs.functionType.getNumInputs();
    const unsigned physicalLanes = rhs.pairedLanes.size();
    if (!sameAttributes(lhs.attributes, rhs.attributes) || softwareLanes == 0 ||
        softwareLanes != lhs.functionType.getNumResults() ||
        softwareLanes != lhs.operands.size() ||
        physicalLanes != rhs.functionType.getNumInputs() ||
        physicalLanes != rhs.functionType.getNumResults() ||
        physicalLanes != rhs.operands.size() || softwareLanes > physicalLanes)
      return false;

    state.nodeMap[patternNode] = static_cast<int>(candidateNode);
    state.reverseNodeMap[candidateNode] = static_cast<int>(patternNode);
    state.laneMap[patternNode].assign(softwareLanes, -1);
    state.reverseLaneMap[patternNode].assign(physicalLanes, -1);
    return true;
  }

  bool matchOrdinaryNode(unsigned patternNode, unsigned candidateNode,
                         CoverageAssignments &state) const {
    if (patternNode >= pattern.nodes.size() ||
        candidateNode >= candidate.nodes.size())
      return false;
    if (state.nodeMap[patternNode] >= 0)
      return static_cast<unsigned>(state.nodeMap[patternNode]) == candidateNode;
    if (state.reverseNodeMap[candidateNode] >= 0)
      return false;

    const ConfiguredFunctionNode &lhs = pattern.nodes[patternNode];
    const ConfiguredFunctionNode &rhs = candidate.nodes[candidateNode];
    if (lhs.operationName != rhs.operationName ||
        !sameType(lhs.functionType, rhs.functionType) ||
        !sameAttributes(lhs.attributes, rhs.attributes) ||
        lhs.operands.size() != rhs.operands.size())
      return false;

    state.nodeMap[patternNode] = static_cast<int>(candidateNode);
    state.reverseNodeMap[candidateNode] = static_cast<int>(patternNode);
    for (auto [lhsOperand, rhsOperand] :
         ::llvm::zip(lhs.operands, rhs.operands)) {
      if (!matchValue(lhsOperand, rhsOperand, state))
        return false;
    }
    return true;
  }

  bool matchValue(const ConfiguredValue &lhs, const ConfiguredValue &rhs,
                  CoverageAssignments &state) const {
    if (lhs.kind != rhs.kind)
      return false;
    if (lhs.kind == ConfiguredValue::Kind::InputPort)
      return matchInput(lhs.index, rhs.index, state);
    if (lhs.index >= pattern.nodes.size() ||
        rhs.index >= candidate.nodes.size())
      return false;

    if (isPairedSync(lhs.index, rhs.index)) {
      if (!matchPairedSync(lhs.index, rhs.index, state))
        return false;
      return matchPairedLane(lhs.index, lhs.result, rhs.index, rhs.result,
                             state);
    }
    if (lhs.result != rhs.result)
      return false;
    return matchOrdinaryNode(lhs.index, rhs.index, state);
  }

  bool residualPairedLanesHaveMatching(const CoverageAssignments &state) const {
    for (unsigned patternNode = 0; patternNode < pattern.nodes.size();
         ++patternNode) {
      if (state.nodeMap[patternNode] < 0)
        continue;
      unsigned candidateNode =
          static_cast<unsigned>(state.nodeMap[patternNode]);
      if (!isPairedSync(patternNode, candidateNode))
        continue;

      ::llvm::SmallVector<unsigned, 4> patternLanes;
      for (unsigned patternLane = 0;
           patternLane < state.laneMap[patternNode].size(); ++patternLane)
        if (state.laneMap[patternNode][patternLane] < 0)
          patternLanes.push_back(patternLane);
      ::llvm::SmallVector<int, 4> laneOwner(
          candidate.nodes[candidateNode].pairedLanes.size(), -1);
      auto augment = [&](auto &&self, unsigned patternLane,
                         ::llvm::SmallVectorImpl<bool> &seen) -> bool {
        for (unsigned candidateLane = 0;
             candidateLane < candidate.nodes[candidateNode].pairedLanes.size();
             ++candidateLane) {
          if (seen[candidateLane] ||
              state.reverseLaneMap[patternNode][candidateLane] >= 0)
            continue;
          CoverageAssignments trial = state;
          if (!matchPairedLane(patternNode, patternLane, candidateNode,
                               candidateLane, trial))
            continue;
          seen[candidateLane] = true;
          int owner = laneOwner[candidateLane];
          if (owner >= 0 && !self(self, static_cast<unsigned>(owner), seen))
            continue;
          laneOwner[candidateLane] = static_cast<int>(patternLane);
          return true;
        }
        return false;
      };
      for (unsigned patternLane : patternLanes) {
        ::llvm::SmallVector<bool, 4> seen(laneOwner.size(), false);
        if (!augment(augment, patternLane, seen))
          return false;
      }
    }
    return true;
  }

  bool matchUnobservedPairedLanes(CoverageAssignments &state) const {
    for (unsigned patternNode = 0; patternNode < pattern.nodes.size();
         ++patternNode) {
      if (state.nodeMap[patternNode] < 0)
        continue;
      unsigned candidateNode =
          static_cast<unsigned>(state.nodeMap[patternNode]);
      if (!isPairedSync(patternNode, candidateNode))
        continue;

      for (unsigned patternLane = 0;
           patternLane < state.laneMap[patternNode].size(); ++patternLane) {
        if (state.laneMap[patternNode][patternLane] >= 0)
          continue;
        for (unsigned candidateLane = 0;
             candidateLane < candidate.nodes[candidateNode].pairedLanes.size();
             ++candidateLane) {
          CoverageAssignments trial = state;
          if (!matchPairedLane(patternNode, patternLane, candidateNode,
                               candidateLane, trial) ||
              !residualPairedLanesHaveMatching(trial) ||
              !matchUnobservedPairedLanes(trial))
            continue;
          state = std::move(trial);
          return true;
        }
        return false;
      }
    }

    return ::llvm::all_of(state.nodeMap,
                          [](int mapped) { return mapped >= 0; });
  }

  bool validateInputs(const CoverageAssignments &state) const {
    for (const ConfiguredBoundaryInput &input : pattern.inputs) {
      auto mapped = state.inputMap.find(input.fuPort);
      if (mapped == state.inputMap.end())
        return false;
      auto candidateInput = ::llvm::find_if(
          candidate.inputs, [&](const ConfiguredBoundaryInput &other) {
            return other.fuPort == mapped->second;
          });
      if (candidateInput == candidate.inputs.end() ||
          !sameType(input.type, candidateInput->type))
        return false;
    }

    for (const ConfiguredBoundaryInput &input : candidate.inputs) {
      if (state.reverseInputMap.count(input.fuPort))
        continue;
      bool foundUse = false;
      for (auto [candidateNode, node] : ::llvm::enumerate(candidate.nodes)) {
        for (auto [operandIndex, operand] : ::llvm::enumerate(node.operands)) {
          if (operand.kind != ConfiguredValue::Kind::InputPort ||
              operand.index != input.fuPort)
            continue;
          foundUse = true;
          int patternNode = state.reverseNodeMap[candidateNode];
          if (patternNode < 0 || node.pairedLanes.empty() ||
              operandIndex >=
                  state.reverseLaneMap[static_cast<unsigned>(patternNode)]
                      .size() ||
              state.reverseLaneMap[static_cast<unsigned>(patternNode)]
                                  [operandIndex] >= 0)
            return false;
        }
      }
      if (!foundUse)
        return false;
    }
    return true;
  }

  bool outputsMayReorder(unsigned lhsOutputIndex, unsigned rhsOutputIndex,
                         const CoverageAssignments &state) const {
    const ConfiguredValue &lhs = pattern.outputs[lhsOutputIndex].value;
    const ConfiguredValue &rhs = pattern.outputs[rhsOutputIndex].value;
    if (lhs.kind != ConfiguredValue::Kind::NodeResult ||
        rhs.kind != ConfiguredValue::Kind::NodeResult ||
        lhs.index != rhs.index || lhs.index >= pattern.nodes.size() ||
        pattern.nodes[lhs.index].operationName != "dataflow.sync")
      return false;
    int candidateNode = state.nodeMap[lhs.index];
    return candidateNode < 0 ||
           isPairedSync(lhs.index, static_cast<unsigned>(candidateNode));
  }

  bool hasInternalResultUse(const ConfiguredFunction &function,
                            unsigned sourceNode) const {
    return ::llvm::any_of(
        function.nodes, [&](const ConfiguredFunctionNode &node) {
          return ::llvm::any_of(
              node.operands, [&](const ConfiguredValue &operand) {
                return operand.kind == ConfiguredValue::Kind::NodeResult &&
                       operand.index == sourceNode;
              });
        });
  }

  bool collectPairedOutputGroup(unsigned seedOutput,
                                const OutputConstraintState &state,
                                PairedOutputGroup &group) const {
    const ConfiguredValue &seed = pattern.outputs[seedOutput].value;
    if (seed.kind != ConfiguredValue::Kind::NodeResult ||
        seed.index >= pattern.nodes.size() ||
        pattern.nodes[seed.index].operationName != "dataflow.sync")
      return false;

    group = {};
    group.patternNode = seed.index;
    unsigned unassignedOutputs = 0;
    for (unsigned patternOutputIndex = 0;
         patternOutputIndex < pattern.outputs.size(); ++patternOutputIndex) {
      const ConfiguredValue &value = pattern.outputs[patternOutputIndex].value;
      if (value.kind != ConfiguredValue::Kind::NodeResult ||
          value.index != group.patternNode)
        continue;
      group.outputs.push_back(patternOutputIndex);
      unassignedOutputs += state.outputMap[patternOutputIndex] < 0;
    }
    if (unassignedOutputs < 2)
      return false;

    const unsigned laneCount =
        pattern.nodes[group.patternNode].functionType.getNumResults();
    ::llvm::SmallVector<bool, 8> seenLanes(laneCount, false);
    bool contiguous = true;
    bool uniqueLanes = true;
    for (auto [position, patternOutputIndex] :
         ::llvm::enumerate(group.outputs)) {
      const ConfiguredValue &value = pattern.outputs[patternOutputIndex].value;
      if (value.result >= laneCount || seenLanes[value.result]) {
        uniqueLanes = false;
      } else {
        seenLanes[value.result] = true;
      }
      if (position > 0 && group.outputs[position - 1] + 1 != patternOutputIndex)
        contiguous = false;
    }
    group.normalizesLanePermutations =
        contiguous && uniqueLanes &&
        !hasInternalResultUse(pattern, group.patternNode);
    return true;
  }

  bool samePortMap(const ::llvm::DenseMap<unsigned, unsigned> &lhs,
                   const ::llvm::DenseMap<unsigned, unsigned> &rhs) const {
    if (lhs.size() != rhs.size())
      return false;
    for (auto [key, value] : lhs) {
      auto other = rhs.find(key);
      if (other == rhs.end() || other->second != value)
        return false;
    }
    return true;
  }

  bool sameAssignmentsOutsideGroup(const CoverageAssignments &lhs,
                                   const CoverageAssignments &rhs,
                                   unsigned patternNode) const {
    if (lhs.nodeMap != rhs.nodeMap ||
        lhs.reverseNodeMap != rhs.reverseNodeMap ||
        !samePortMap(lhs.inputMap, rhs.inputMap) ||
        !samePortMap(lhs.reverseInputMap, rhs.reverseInputMap) ||
        lhs.laneMap.size() != rhs.laneMap.size() ||
        lhs.reverseLaneMap.size() != rhs.reverseLaneMap.size())
      return false;
    for (unsigned node = 0; node < lhs.laneMap.size(); ++node) {
      if (node == patternNode)
        continue;
      if (lhs.laneMap[node] != rhs.laneMap[node] ||
          lhs.reverseLaneMap[node] != rhs.reverseLaneMap[node])
        return false;
    }
    return true;
  }

  bool rememberPairedOutputGroupState(const OutputConstraintState &state,
                                      const PairedOutputGroup &group,
                                      PairedOutputGroupStates &seen) const {
    if (!group.normalizesLanePermutations)
      return true;
    int candidateNode = state.assignments.nodeMap[group.patternNode];
    if (candidateNode < 0 ||
        !isPairedSync(group.patternNode,
                      static_cast<unsigned>(candidateNode)) ||
        hasInternalResultUse(candidate, static_cast<unsigned>(candidateNode)))
      return true;

    // Contiguous sink results have the same external order boundary. The
    // selected physical outputs and assignments outside this sync therefore
    // identify a complete equivalence class of lane permutations.
    ::llvm::SmallVector<unsigned, 8> selectedOutputs;
    for (unsigned patternOutputIndex : group.outputs)
      if (state.outputMap[patternOutputIndex] >= 0)
        selectedOutputs.push_back(
            static_cast<unsigned>(state.outputMap[patternOutputIndex]));
    ::llvm::sort(selectedOutputs);

    // DenseMap reserves the two largest unsigned keys.
    unsigned selectedHash =
        static_cast<unsigned>(::llvm::hash_combine_range(
            selectedOutputs.begin(), selectedOutputs.end())) &
        (~0U >> 1);
    auto &states = seen[selectedHash];
    for (const PairedOutputGroupState &other : states)
      if (selectedOutputs == other.selectedOutputs &&
          sameAssignmentsOutsideGroup(
              state.assignments, other.outsideAssignments, group.patternNode))
        return false;

    PairedOutputGroupState memoized;
    memoized.outsideAssignments = state.assignments;
    memoized.outsideAssignments.laneMap[group.patternNode].clear();
    memoized.outsideAssignments.reverseLaneMap[group.patternNode].clear();
    memoized.selectedOutputs = std::move(selectedOutputs);
    states.push_back(std::move(memoized));
    return true;
  }

  unsigned
  outputMultiplicity(::llvm::ArrayRef<ConfiguredBoundaryOutput> outputs,
                     const ConfiguredValue &value) const {
    return static_cast<unsigned>(
        ::llvm::count_if(outputs, [&](const ConfiguredBoundaryOutput &output) {
          return output.value == value;
        }));
  }

  enum class ExtraOutputLegality { Invalid, Potential, Legal };

  ExtraOutputLegality
  classifyExtraOutput(unsigned outputIndex,
                      const CoverageAssignments &state) const {
    if (outputIndex >= candidate.outputs.size())
      return ExtraOutputLegality::Invalid;
    const ConfiguredBoundaryOutput &output = candidate.outputs[outputIndex];
    if (output.value.kind != ConfiguredValue::Kind::NodeResult ||
        output.value.index >= candidate.nodes.size())
      return ExtraOutputLegality::Invalid;
    const ConfiguredFunctionNode &source = candidate.nodes[output.value.index];
    if (source.operationName != "dataflow.sync" || source.pairedLanes.empty() ||
        output.value.result >= source.pairedLanes.size() ||
        outputMultiplicity(candidate.outputs, output.value) != 1)
      return ExtraOutputLegality::Invalid;

    int patternNode = state.reverseNodeMap[output.value.index];
    if (patternNode < 0)
      return ExtraOutputLegality::Potential;
    if (!isPairedSync(static_cast<unsigned>(patternNode), output.value.index))
      return ExtraOutputLegality::Invalid;

    const auto &selectedLanes =
        state.reverseLaneMap[static_cast<unsigned>(patternNode)];
    if (selectedLanes[output.value.result] < 0)
      return ExtraOutputLegality::Legal;
    bool hasUnassignedSibling = false;
    for (auto [laneIndex, softwareLane] : ::llvm::enumerate(selectedLanes)) {
      if (laneIndex == output.value.result)
        continue;
      if (softwareLane >= 0)
        return ExtraOutputLegality::Legal;
      hasUnassignedSibling = true;
    }
    return hasUnassignedSibling ? ExtraOutputLegality::Potential
                                : ExtraOutputLegality::Invalid;
  }

  bool matchOutput(unsigned patternOutputIndex, unsigned candidateOutputIndex,
                   CoverageAssignments &state) const {
    const ConfiguredBoundaryOutput &lhs = pattern.outputs[patternOutputIndex];
    const ConfiguredBoundaryOutput &rhs =
        candidate.outputs[candidateOutputIndex];
    return sameType(lhs.type, rhs.type) &&
           matchValue(lhs.value, rhs.value, state) &&
           outputMultiplicity(pattern.outputs, lhs.value) ==
               outputMultiplicity(candidate.outputs, rhs.value);
  }

  bool filterOutputDomains(OutputConstraintState &state, bool &changed) const {
    for (unsigned patternOutputIndex = 0;
         patternOutputIndex < pattern.outputs.size(); ++patternOutputIndex) {
      ::llvm::SmallVector<unsigned, 8> filtered;
      for (unsigned candidateOutputIndex : state.domains[patternOutputIndex]) {
        int owner = state.candidateOwner[candidateOutputIndex];
        if (owner >= 0 && static_cast<unsigned>(owner) != patternOutputIndex)
          continue;
        CoverageAssignments trial = state.assignments;
        if (matchOutput(patternOutputIndex, candidateOutputIndex, trial))
          filtered.push_back(candidateOutputIndex);
      }
      if (filtered.empty())
        return false;
      if (filtered.size() != state.domains[patternOutputIndex].size()) {
        state.domains[patternOutputIndex] = std::move(filtered);
        changed = true;
      }
    }
    return true;
  }

  bool pruneOutputOrder(OutputConstraintState &state, bool &changed) const {
    for (unsigned lhsOutputIndex = 0; lhsOutputIndex < pattern.outputs.size();
         ++lhsOutputIndex) {
      for (unsigned rhsOutputIndex = lhsOutputIndex + 1;
           rhsOutputIndex < pattern.outputs.size(); ++rhsOutputIndex) {
        if (outputsMayReorder(lhsOutputIndex, rhsOutputIndex,
                              state.assignments))
          continue;

        auto &lhsDomain = state.domains[lhsOutputIndex];
        auto &rhsDomain = state.domains[rhsOutputIndex];
        unsigned rhsMaximum = rhsDomain.back();
        ::llvm::SmallVector<unsigned, 8> filteredLhs;
        for (unsigned candidateOutputIndex : lhsDomain)
          if (candidateOutputIndex < rhsMaximum)
            filteredLhs.push_back(candidateOutputIndex);
        if (filteredLhs.empty())
          return false;
        if (filteredLhs.size() != lhsDomain.size()) {
          lhsDomain = std::move(filteredLhs);
          changed = true;
        }

        unsigned lhsMinimum = lhsDomain.front();
        ::llvm::SmallVector<unsigned, 8> filteredRhs;
        for (unsigned candidateOutputIndex : rhsDomain)
          if (lhsMinimum < candidateOutputIndex)
            filteredRhs.push_back(candidateOutputIndex);
        if (filteredRhs.empty())
          return false;
        if (filteredRhs.size() != rhsDomain.size()) {
          rhsDomain = std::move(filteredRhs);
          changed = true;
        }
      }
    }
    return true;
  }

  bool hasOutputMatching(const OutputConstraintState &state) const {
    ::llvm::SmallVector<int, 8> patternOwner = state.outputMap;
    ::llvm::SmallVector<bool, 8> lockedPatterns(pattern.outputs.size(), false);
    for (unsigned patternOutputIndex = 0;
         patternOutputIndex < pattern.outputs.size(); ++patternOutputIndex)
      lockedPatterns[patternOutputIndex] =
          patternOwner[patternOutputIndex] >= 0;

    ::llvm::SmallVector<int, 8> outputOwner(candidate.outputs.size(), -1);
    for (unsigned patternOutputIndex = 0;
         patternOutputIndex < pattern.outputs.size(); ++patternOutputIndex) {
      int candidateOutputIndex = patternOwner[patternOutputIndex];
      if (candidateOutputIndex < 0)
        continue;
      int &owner = outputOwner[static_cast<unsigned>(candidateOutputIndex)];
      if (owner >= 0 && static_cast<unsigned>(owner) != patternOutputIndex)
        return false;
      owner = static_cast<int>(patternOutputIndex);
    }

    auto assignRequired = [&](auto &&self, unsigned candidateOutputIndex,
                              ::llvm::SmallVectorImpl<bool> &seen) -> bool {
      for (unsigned patternOutputIndex = 0;
           patternOutputIndex < pattern.outputs.size(); ++patternOutputIndex) {
        if (lockedPatterns[patternOutputIndex] || seen[patternOutputIndex] ||
            !::llvm::is_contained(state.domains[patternOutputIndex],
                                  candidateOutputIndex))
          continue;
        seen[patternOutputIndex] = true;
        int previousCandidate = patternOwner[patternOutputIndex];
        if (previousCandidate >= 0 &&
            !self(self, static_cast<unsigned>(previousCandidate), seen))
          continue;
        patternOwner[patternOutputIndex] =
            static_cast<int>(candidateOutputIndex);
        return true;
      }
      return false;
    };
    for (unsigned candidateOutputIndex = 0;
         candidateOutputIndex < candidate.outputs.size();
         ++candidateOutputIndex) {
      if (classifyExtraOutput(candidateOutputIndex, state.assignments) !=
              ExtraOutputLegality::Invalid ||
          ::llvm::is_contained(patternOwner,
                               static_cast<int>(candidateOutputIndex)))
        continue;
      ::llvm::SmallVector<bool, 8> seen(pattern.outputs.size(), false);
      if (!assignRequired(assignRequired, candidateOutputIndex, seen))
        return false;
    }

    outputOwner.assign(candidate.outputs.size(), -1);
    for (unsigned patternOutputIndex = 0;
         patternOutputIndex < pattern.outputs.size(); ++patternOutputIndex) {
      int candidateOutputIndex = patternOwner[patternOutputIndex];
      if (candidateOutputIndex < 0)
        continue;
      int &owner = outputOwner[static_cast<unsigned>(candidateOutputIndex)];
      if (owner >= 0 && static_cast<unsigned>(owner) != patternOutputIndex)
        return false;
      owner = static_cast<int>(patternOutputIndex);
    }

    auto augmentPattern = [&](auto &&self, unsigned patternOutputIndex,
                              ::llvm::SmallVectorImpl<bool> &seen) -> bool {
      for (unsigned candidateOutputIndex : state.domains[patternOutputIndex]) {
        if (seen[candidateOutputIndex])
          continue;
        seen[candidateOutputIndex] = true;
        int owner = outputOwner[candidateOutputIndex];
        if (owner >= 0 && (lockedPatterns[static_cast<unsigned>(owner)] ||
                           !self(self, static_cast<unsigned>(owner), seen)))
          continue;
        outputOwner[candidateOutputIndex] =
            static_cast<int>(patternOutputIndex);
        patternOwner[patternOutputIndex] =
            static_cast<int>(candidateOutputIndex);
        return true;
      }
      return false;
    };
    for (unsigned patternOutputIndex = 0;
         patternOutputIndex < pattern.outputs.size(); ++patternOutputIndex) {
      if (patternOwner[patternOutputIndex] >= 0)
        continue;
      ::llvm::SmallVector<bool, 8> seen(candidate.outputs.size(), false);
      if (!augmentPattern(augmentPattern, patternOutputIndex, seen))
        return false;
    }

    for (unsigned candidateOutputIndex = 0;
         candidateOutputIndex < candidate.outputs.size();
         ++candidateOutputIndex)
      if (classifyExtraOutput(candidateOutputIndex, state.assignments) ==
              ExtraOutputLegality::Invalid &&
          !::llvm::is_contained(patternOwner,
                                static_cast<int>(candidateOutputIndex)))
        return false;
    return true;
  }

  bool assignOutput(OutputConstraintState &state, unsigned patternOutputIndex,
                    unsigned candidateOutputIndex) const {
    if (!::llvm::is_contained(state.domains[patternOutputIndex],
                              candidateOutputIndex))
      return false;
    int mapped = state.outputMap[patternOutputIndex];
    if (mapped >= 0)
      return static_cast<unsigned>(mapped) == candidateOutputIndex;
    if (state.candidateOwner[candidateOutputIndex] >= 0)
      return false;

    CoverageAssignments trial = state.assignments;
    if (!matchOutput(patternOutputIndex, candidateOutputIndex, trial))
      return false;
    state.assignments = std::move(trial);
    state.domains[patternOutputIndex].assign(1, candidateOutputIndex);
    state.outputMap[patternOutputIndex] =
        static_cast<int>(candidateOutputIndex);
    state.candidateOwner[candidateOutputIndex] =
        static_cast<int>(patternOutputIndex);
    return true;
  }

  bool commitForcedOutputs(OutputConstraintState &state, bool &changed) const {
    for (unsigned patternOutputIndex = 0;
         patternOutputIndex < pattern.outputs.size(); ++patternOutputIndex) {
      if (state.outputMap[patternOutputIndex] >= 0 ||
          state.domains[patternOutputIndex].size() != 1)
        continue;
      unsigned candidateOutputIndex = state.domains[patternOutputIndex].front();
      if (!assignOutput(state, patternOutputIndex, candidateOutputIndex))
        return false;
      changed = true;
    }
    return true;
  }

  bool propagateOutputConstraints(OutputConstraintState &state) const {
    bool changed = true;
    while (changed) {
      changed = false;
      if (!filterOutputDomains(state, changed) ||
          !pruneOutputOrder(state, changed) || !hasOutputMatching(state) ||
          !commitForcedOutputs(state, changed))
        return false;
    }
    return true;
  }

  bool pairedOutputGroupHasMatching(const OutputConstraintState &state,
                                    const PairedOutputGroup &group) const {
    ::llvm::SmallVector<int, 8> candidateOwner(candidate.outputs.size(), -1);
    for (unsigned candidateOutputIndex = 0;
         candidateOutputIndex < candidate.outputs.size();
         ++candidateOutputIndex)
      if (state.candidateOwner[candidateOutputIndex] >= 0)
        candidateOwner[candidateOutputIndex] = -2;

    auto augment = [&](auto &&self, unsigned patternOutputIndex,
                       ::llvm::SmallVectorImpl<bool> &seen) -> bool {
      for (unsigned candidateOutputIndex : state.domains[patternOutputIndex]) {
        if (seen[candidateOutputIndex] ||
            candidateOwner[candidateOutputIndex] == -2)
          continue;
        CoverageAssignments trial = state.assignments;
        if (!matchOutput(patternOutputIndex, candidateOutputIndex, trial))
          continue;
        seen[candidateOutputIndex] = true;
        int owner = candidateOwner[candidateOutputIndex];
        if (owner >= 0 && !self(self, static_cast<unsigned>(owner), seen))
          continue;
        candidateOwner[candidateOutputIndex] =
            static_cast<int>(patternOutputIndex);
        return true;
      }
      return false;
    };

    for (unsigned patternOutputIndex : group.outputs) {
      if (state.outputMap[patternOutputIndex] >= 0)
        continue;
      ::llvm::SmallVector<bool, 8> seen(candidate.outputs.size(), false);
      if (!augment(augment, patternOutputIndex, seen))
        return false;
    }
    return true;
  }

  bool solvePairedOutputGroup(OutputConstraintState &state,
                              const PairedOutputGroup &group,
                              PairedOutputGroupStates &seen) const {
    if (!rememberPairedOutputGroupState(state, group, seen) ||
        !pairedOutputGroupHasMatching(state, group))
      return false;

    unsigned branchOutput = pattern.outputs.size();
    for (unsigned patternOutputIndex : group.outputs)
      if (state.outputMap[patternOutputIndex] < 0) {
        branchOutput = patternOutputIndex;
        break;
      }

    if (branchOutput == pattern.outputs.size())
      return solveOutputConstraints(state);

    for (unsigned candidateOutputIndex : state.domains[branchOutput]) {
      OutputConstraintState trial = state;
      if (!assignOutput(trial, branchOutput, candidateOutputIndex) ||
          !solvePairedOutputGroup(trial, group, seen))
        continue;
      state = std::move(trial);
      return true;
    }
    return false;
  }

  bool outputsRespectOrder(const OutputConstraintState &state) const {
    for (unsigned lhsOutputIndex = 0; lhsOutputIndex < pattern.outputs.size();
         ++lhsOutputIndex)
      for (unsigned rhsOutputIndex = lhsOutputIndex + 1;
           rhsOutputIndex < pattern.outputs.size(); ++rhsOutputIndex)
        if (state.outputMap[lhsOutputIndex] >=
                state.outputMap[rhsOutputIndex] &&
            !outputsMayReorder(lhsOutputIndex, rhsOutputIndex,
                               state.assignments))
          return false;
    return true;
  }

  bool validateExtraOutputs(::llvm::ArrayRef<bool> usedOutputs,
                            const CoverageAssignments &state) const {
    for (unsigned outputIndex = 0; outputIndex < candidate.outputs.size();
         ++outputIndex) {
      if (usedOutputs[outputIndex])
        continue;
      if (classifyExtraOutput(outputIndex, state) != ExtraOutputLegality::Legal)
        return false;
    }
    return true;
  }

  bool solveOutputConstraints(OutputConstraintState &state) const {
    if (!propagateOutputConstraints(state))
      return false;

    unsigned branchOutput = pattern.outputs.size();
    for (unsigned patternOutputIndex = 0;
         patternOutputIndex < pattern.outputs.size(); ++patternOutputIndex) {
      if (state.outputMap[patternOutputIndex] < 0) {
        PairedOutputGroup group;
        if (collectPairedOutputGroup(patternOutputIndex, state, group)) {
          PairedOutputGroupStates seen;
          return solvePairedOutputGroup(state, group, seen);
        }
        branchOutput = patternOutputIndex;
        break;
      }
    }

    if (branchOutput == pattern.outputs.size()) {
      if (!outputsRespectOrder(state))
        return false;
      CoverageAssignments trial = state.assignments;
      ::llvm::SmallVector<bool, 8> usedOutputs(candidate.outputs.size(), false);
      for (int candidateOutputIndex : state.outputMap)
        usedOutputs[static_cast<unsigned>(candidateOutputIndex)] = true;
      if (!matchUnobservedPairedLanes(trial) || !validateInputs(trial) ||
          !validateExtraOutputs(usedOutputs, trial))
        return false;
      state.assignments = std::move(trial);
      return true;
    }

    for (unsigned candidateOutputIndex : state.domains[branchOutput]) {
      OutputConstraintState trial = state;
      if (!assignOutput(trial, branchOutput, candidateOutputIndex) ||
          !solveOutputConstraints(trial))
        continue;
      state = std::move(trial);
      return true;
    }
    return false;
  }

  bool matchOutputs(CoverageAssignments &state,
                    ::llvm::SmallVectorImpl<unsigned> &outputMap) const {
    OutputConstraintState search;
    search.assignments = state;
    search.domains.resize(pattern.outputs.size());
    for (auto &domain : search.domains)
      for (unsigned candidateOutputIndex = 0;
           candidateOutputIndex < candidate.outputs.size();
           ++candidateOutputIndex)
        domain.push_back(candidateOutputIndex);
    search.outputMap.assign(pattern.outputs.size(), -1);
    search.candidateOwner.assign(candidate.outputs.size(), -1);
    if (!solveOutputConstraints(search))
      return false;

    state = std::move(search.assignments);
    for (unsigned patternOutputIndex = 0;
         patternOutputIndex < pattern.outputs.size(); ++patternOutputIndex)
      outputMap[patternOutputIndex] =
          static_cast<unsigned>(search.outputMap[patternOutputIndex]);
    return true;
  }

  void buildWitness(const CoverageAssignments &state,
                    ::llvm::ArrayRef<unsigned> outputMap,
                    ConfiguredFunctionMatch &witness) const {
    witness = {};
    for (int mapped : state.nodeMap)
      witness.nodeMap.push_back(static_cast<unsigned>(mapped));
    for (const ConfiguredBoundaryInput &input : pattern.inputs)
      witness.inputPorts.emplace_back(input.fuPort,
                                      state.inputMap.lookup(input.fuPort));
    for (auto [patternOutputIndex, candidateOutputIndex] :
         ::llvm::enumerate(outputMap))
      witness.outputPorts.emplace_back(
          pattern.outputs[patternOutputIndex].fuPort,
          candidate.outputs[candidateOutputIndex].fuPort);

    for (unsigned patternNode = 0; patternNode < pattern.nodes.size();
         ++patternNode) {
      if (state.laneMap[patternNode].empty())
        continue;
      unsigned candidateNode =
          static_cast<unsigned>(state.nodeMap[patternNode]);
      ConfiguredFunctionMatch::PairedLaneSelection selection;
      selection.softwareNode = patternNode;
      selection.physicalLaneCount =
          candidate.nodes[candidateNode].pairedLanes.size();
      for (int candidateLane : state.laneMap[patternNode])
        selection.lanes.push_back(
            candidate.nodes[candidateNode]
                .pairedLanes[static_cast<unsigned>(candidateLane)]);
      witness.pairedLaneSelections.push_back(std::move(selection));
    }
  }
};

} // namespace

std::string ConfiguredFunctionMatch::PairedLaneSelection::bitmask() const {
  std::string value(physicalLaneCount, '0');
  for (const PairedLane &lane : lanes) {
    if (lane.maskBit >= value.size())
      return {};
    value[lane.maskBit] = '1';
  }
  return value;
}

bool matchConfiguredFunctionsForCoverage(const ConfiguredFunction &pattern,
                                         const ConfiguredFunction &candidate,
                                         ConfiguredFunctionMatch *witness) {
  return CoverageMatcher(pattern, candidate).run(witness);
}

} // namespace fabric
