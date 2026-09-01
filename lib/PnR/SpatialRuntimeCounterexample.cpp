#include "PnR/SpatialCandidateState.h"
#include "PnR/SpatialMappingSelectionProjection.h"

#include "SpatialCandidateStateInternal.h"

#include "llvm/ADT/STLExtras.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>

using namespace loom::pnr;

namespace {

using detail::attachmentTraversal;
using detail::candidateError;

} // namespace

std::optional<PnrIndex>
SpatialCandidateState::firstRuntimeCounterexampleViolation() const {
  const auto found =
      llvm::find(runtimeCounterexampleClauseViolated_, std::uint8_t{1});
  if (found == runtimeCounterexampleClauseViolated_.end())
    return std::nullopt;
  return static_cast<PnrIndex>(found -
                               runtimeCounterexampleClauseViolated_.begin());
}

llvm::Expected<bool> SpatialCandidateState::runtimeCounterexampleLiteralHolds(
    const FrozenNoGoodResolvedLiteral &literal,
    llvm::ArrayRef<const RouteTreeState *> routes,
    llvm::ArrayRef<llvm::ArrayRef<std::optional<llvm::APInt>>>
        tagValues) const {
  if (literal.kind ==
      FrozenNoGoodResolvedLiteral::Kind::SpatialMappingIdentityEquals) {
    if (!literal.importedMapping)
      return candidateError(
          "runtime-counterexample SpatialMapping cache is absent");
    return spatialMappingSelectionEqualsCandidate(
        literal.importedMapping->view(), *this, routes, tagValues);
  }
  const auto logicalNets = problem_->transfers().logicalNets();
  if (routes.size() != logicalNets.size() ||
      tagValues.size() != logicalNets.size() ||
      literal.logicalNet >= logicalNets.size())
    return candidateError(
        "runtime-counterexample literal has an invalid logical net");
  const RouteTreeState *route = routes[literal.logicalNet];
  if (!route || &route->routingGraph() != &problem_->routing())
    return candidateError(
        "runtime-counterexample literal has a foreign RouteTree");

  if (usesRegisterFifo(literal.logicalNet))
    return false;

  if (literal.kind == FrozenNoGoodResolvedLiteral::Kind::NetTagEquals) {
    if (!literal.tagValue)
      return candidateError(
          "runtime-counterexample Physical Tag literal has no value");
    const auto values = tagValues[literal.logicalNet];
    if (literal.target >= values.size() || !values[literal.target])
      return false;
    return ::fabric::comparePhysicalTagValues(*values[literal.target],
                                              *literal.tagValue) == 0;
  }

  if (literal.kind ==
      FrozenNoGoodResolvedLiteral::Kind::TransferAttachmentEquals) {
    if (literal.sink) {
      if (*literal.sink >= logicalNets[literal.logicalNet].sinkCount)
        return candidateError(
            "runtime-counterexample sink attachment is out of range");
      return logicalNetSinkEndpoint(literal.logicalNet, *literal.sink) ==
             literal.target;
    }
    return logicalNetSourceEndpoint(literal.logicalNet) == literal.target;
  }
  if (literal.kind != FrozenNoGoodResolvedLiteral::Kind::NetUsesTraversal)
    return candidateError(
        "runtime-counterexample literal has an unknown frozen kind");
  if (route->isUnrouted())
    return false;

  const auto selectedLocalTraversal = [&](FrozenSpatialTerminalBinding binding)
      -> llvm::Expected<std::optional<PnrIndex>> {
    PnrIndex option = getInvalidPnrIndex();
    switch (binding.kind) {
    case FrozenSpatialTerminalBindingKind::PortDemand:
      if (binding.index >= portAttachments_.size())
        return candidateError(
            "runtime-counterexample port attachment is out of range");
      option = portAttachments_[binding.index];
      break;
    case FrozenSpatialTerminalBindingKind::GraphBoundary:
      if (binding.index >= graphBoundaryAttachments_.size())
        return candidateError(
            "runtime-counterexample boundary attachment is out of range");
      option = graphBoundaryAttachments_[binding.index];
      break;
    }
    return attachmentTraversal(problem_->ports(), option);
  };
  const auto sourceBinding =
      problem_->transfers().logicalNetSourceBindings()[literal.logicalNet];
  auto sourceTraversal = selectedLocalTraversal(sourceBinding);
  if (!sourceTraversal)
    return sourceTraversal.takeError();
  if (*sourceTraversal && **sourceTraversal == literal.target)
    return true;

  const FrozenSpatialLogicalNet &net = logicalNets[literal.logicalNet];
  const auto sinkBindings = problem_->transfers().logicalNetSinkBindings();
  if (net.sinkOffset > sinkBindings.size() ||
      net.sinkCount > sinkBindings.size() - net.sinkOffset)
    return candidateError(
        "runtime-counterexample sink binding range is invalid");
  const auto sinkLocalHolds = [&](PnrIndex sink) -> llvm::Expected<bool> {
    auto traversal =
        selectedLocalTraversal(sinkBindings[net.sinkOffset + sink]);
    if (!traversal)
      return traversal.takeError();
    return *traversal && **traversal == literal.target;
  };

  const auto arcs = problem_->routing().routingArcs();
  if (!literal.sink) {
    for (PnrIndex sink = 0; sink < net.sinkCount; ++sink) {
      auto holds = sinkLocalHolds(sink);
      if (!holds)
        return holds.takeError();
      if (*holds)
        return true;
    }
    for (const RouteTreeNode &node : route->nodeStorage()) {
      if (!node.isActive() || node.parentArc == getInvalidPnrIndex())
        continue;
      if (node.parentArc >= arcs.size())
        return candidateError(
            "runtime-counterexample RouteTree arc is out of range");
      if (arcs[node.parentArc].traversal == literal.target)
        return true;
    }
    return false;
  }

  if (*literal.sink >= net.sinkCount)
    return candidateError(
        "runtime-counterexample traversal sink is out of range");
  auto sinkHolds = sinkLocalHolds(*literal.sink);
  if (!sinkHolds)
    return sinkHolds.takeError();
  if (*sinkHolds)
    return true;
  const auto endpoint = route->sinkEndpoint(*literal.sink);
  if (!endpoint)
    return false;
  auto slot = route->findNode(*endpoint);
  if (!slot)
    return candidateError(
        "runtime-counterexample sink is absent from its RouteTree");
  for (std::size_t depth = 0; depth <= route->nodeStorage().size(); ++depth) {
    const RouteTreeNode &node = route->node(*slot);
    if (node.parentArc == getInvalidPnrIndex())
      return false;
    if (node.parentArc >= arcs.size())
      return candidateError(
          "runtime-counterexample branch arc is out of range");
    if (arcs[node.parentArc].traversal == literal.target)
      return true;
    slot = route->parentNodeSlot(*slot);
    if (!slot)
      return candidateError("runtime-counterexample branch parent is absent");
  }
  return candidateError("runtime-counterexample RouteTree branch is cyclic");
}

llvm::Expected<std::uint64_t>
SpatialCandidateState::countRuntimeCounterexampleViolations(
    llvm::ArrayRef<const RouteTreeState *> routes,
    llvm::ArrayRef<llvm::ArrayRef<std::optional<llvm::APInt>>>
        tagValues) const {
  const auto clauses = problem_->constraints().resolvedNoGoods();
  const auto literals = problem_->constraints().resolvedNoGoodLiterals();
  std::uint64_t violations = 0;
  for (const FrozenNoGoodResolvedClause &clause : clauses) {
    if (clause.literalOffset > literals.size() ||
        clause.literalCount > literals.size() - clause.literalOffset)
      return candidateError(
          "runtime-counterexample clause literal range is invalid");
    bool allHold = true;
    for (const FrozenNoGoodResolvedLiteral &literal :
         literals.slice(clause.literalOffset, clause.literalCount)) {
      auto holds =
          runtimeCounterexampleLiteralHolds(literal, routes, tagValues);
      if (!holds)
        return holds.takeError();
      if (!*holds) {
        allHold = false;
        break;
      }
    }
    if (allHold) {
      if (violations == std::numeric_limits<std::uint64_t>::max())
        return candidateError(
            "runtime-counterexample violation count overflows u64");
      ++violations;
    }
  }
  return violations;
}
