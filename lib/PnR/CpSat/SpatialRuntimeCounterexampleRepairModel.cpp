#include "SpatialRuntimeCounterexampleRepairModel.h"

#include "SpatialExactRepairModel.h"

#include "Fabric/IR/PhysicalTag.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Errc.h"

#include <array>
#include <functional>
#include <limits>
#include <set>
#include <tuple>

using namespace loom::pnr;
using namespace loom::pnr::detail;
using namespace operations_research::sat;

namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      llvm::errc::invalid_argument,
      "invalid Spatial runtime-counterexample repair model: %s",
      message.str().c_str());
}

int compareUnsigned(const llvm::APInt &lhs, const llvm::APInt &rhs) {
  return ::fabric::comparePhysicalTagValues(lhs, rhs);
}

bool unsignedLess(const llvm::APInt &lhs, const llvm::APInt &rhs) {
  return compareUnsigned(lhs, rhs) < 0;
}

llvm::Expected<llvm::APInt> nextUnsigned(const llvm::APInt &value) {
  if (value.getBitWidth() == std::numeric_limits<unsigned>::max())
    return invalid("Physical Tag alternative width overflows");
  llvm::APInt next =
      value.isAllOnes() ? value.zext(value.getBitWidth() + 1) : value;
  ++next;
  return ::fabric::canonicalPhysicalTagValue(next);
}

bool valueAllowed(
    const llvm::APInt &value,
    const std::optional<llvm::ArrayRef<
        ::loom::mapping::SpatialConstraintDomainValue>> &restriction) {
  if (!restriction)
    return true;
  return llvm::any_of(*restriction, [&](const auto &domainValue) {
    const auto *interval =
        std::get_if<::loom::mapping::SpatialConstraintUnsignedInterval>(
            &domainValue);
    return interval && compareUnsigned(interval->lower, value) <= 0 &&
           compareUnsigned(value, interval->upper) < 0;
  });
}

llvm::Expected<std::vector<llvm::APInt>> enumerateTagAlternatives(
    const SpatialCandidateState &candidate,
    const FrozenNoGoodResolvedLiteral &literal) {
  if (literal.kind != FrozenNoGoodResolvedLiteral::Kind::NetTagEquals ||
      !literal.tagValue)
    return invalid("tag breaker has no exact forbidden value");
  const auto nets = candidate.problem().transfers().logicalNets();
  if (literal.logicalNet >= nets.size())
    return invalid("tag breaker logical net is out of range");
  const auto segments = candidate.tagSegments(literal.logicalNet);
  if (literal.target >= segments.size())
    return invalid("tag breaker segment is out of range");
  const auto domains =
      candidate.tagSegmentDomains(literal.logicalNet, literal.target);
  const std::uint32_t width = segments[literal.target].tagWidthBits;
  const auto restriction = candidate.problem()
                               .constraints()
                               .shard(::mapping::SpatialConstraintProjection::
                                          NetAssignedTagValues)
                               .restrictedDomain(::loom::mapping::
                                                     SpatialConstraintSubject{
                                                         nets[literal.logicalNet]
                                                             .producer});
  std::vector<llvm::APInt> alternatives;
  const auto remember = [&](llvm::APInt value) {
    value = ::fabric::canonicalPhysicalTagValue(value);
    if (compareUnsigned(value, *literal.tagValue) == 0 ||
        !::fabric::isRepresentablePhysicalTagValue(width, value) ||
        !valueAllowed(value, restriction) ||
        llvm::any_of(domains, [&](PnrIndex domain) {
          return candidate.tagDomainValueConflicts(domain, value);
        }))
      return;
    alternatives.push_back(std::move(value));
  };

  std::uint64_t segmentCount = 0;
  for (PnrIndex logicalNet = 0; logicalNet < nets.size(); ++logicalNet) {
    const auto values = candidate.tagValues(logicalNet);
    if (values.size() >
        std::numeric_limits<std::uint64_t>::max() - segmentCount)
      return invalid("Physical Tag segment inventory exceeds u64");
    segmentCount += values.size();
    for (const auto &value : values)
      if (value)
        remember(*value);
  }
  const std::uint64_t probeLimit =
      segmentCount > std::numeric_limits<std::uint64_t>::max() - 2
          ? std::numeric_limits<std::uint64_t>::max()
          : segmentCount + 2;
  std::uint64_t probes = 0;
  const auto probeRange = [&](llvm::APInt value, const llvm::APInt *upper)
      -> llvm::Error {
    value = ::fabric::canonicalPhysicalTagValue(value);
    while (probes < probeLimit &&
           (!upper || compareUnsigned(value, *upper) < 0) &&
           ::fabric::isRepresentablePhysicalTagValue(width, value)) {
      remember(value);
      ++probes;
      auto next = nextUnsigned(value);
      if (!next)
        return next.takeError();
      value = std::move(*next);
    }
    return llvm::Error::success();
  };
  if (!restriction) {
    if (llvm::Error error = probeRange(llvm::APInt(1, 0), nullptr))
      return std::move(error);
  } else {
    for (const auto &domainValue : *restriction) {
      const auto *interval =
          std::get_if<::loom::mapping::SpatialConstraintUnsignedInterval>(
              &domainValue);
      if (!interval)
        return invalid("tag breaker restriction is not an interval");
      if (llvm::Error error = probeRange(interval->lower, &interval->upper))
        return std::move(error);
    }
  }
  llvm::sort(alternatives, unsignedLess);
  alternatives.erase(
      std::unique(alternatives.begin(), alternatives.end(),
                  [](const llvm::APInt &lhs, const llvm::APInt &rhs) {
                    return compareUnsigned(lhs, rhs) == 0;
                  }),
      alternatives.end());
  return alternatives;
}

llvm::Expected<PnrIndex> terminalDecision(
    const SpatialBindingRelationModel &bindings,
    FrozenSpatialTerminalBinding binding) {
  switch (binding.kind) {
  case FrozenSpatialTerminalBindingKind::PortDemand:
    return bindings.portDecisionOffset() + binding.index;
  case FrozenSpatialTerminalBindingKind::GraphBoundary:
    return bindings.graphBoundaryDecisionOffset() + binding.index;
  }
  llvm_unreachable("unknown frozen terminal binding kind");
}

llvm::ArrayRef<PnrIndex>
terminalChoices(const SpatialBindingRelationModel &bindings,
                FrozenSpatialTerminalBinding binding) {
  switch (binding.kind) {
  case FrozenSpatialTerminalBindingKind::PortDemand:
    return bindings.portAttachmentChoices(binding.index);
  case FrozenSpatialTerminalBindingKind::GraphBoundary:
    return bindings.graphBoundaryAttachmentChoices(binding.index);
  }
  llvm_unreachable("unknown frozen terminal binding kind");
}

llvm::Error addTerminalChoicePredicate(
    CpModelBuilder &model, const SpatialCandidateState &candidate,
    const SpatialBindingRelationModel &bindings,
    llvm::ArrayRef<IntVar> bindingVariables,
    llvm::ArrayRef<int> decisionVariables,
    FrozenSpatialTerminalBinding binding,
    const std::function<bool(const FrozenSpatialAttachmentOption &)> &allowed) {
  auto decision = terminalDecision(bindings, binding);
  if (!decision)
    return decision.takeError();
  if (*decision >= decisionVariables.size() ||
      decisionVariables[*decision] < 0 ||
      static_cast<std::size_t>(decisionVariables[*decision]) >=
          bindingVariables.size())
    return invalid("breaker terminal is outside its repair region");
  const auto choices = terminalChoices(bindings, binding);
  const auto options = candidate.problem().ports().attachmentOptions();
  TableConstraint accepted = model.AddAllowedAssignments(
      {bindingVariables[decisionVariables[*decision]]});
  bool hasAllowedChoice = false;
  for (auto indexed : llvm::enumerate(choices)) {
    if (indexed.value() >= options.size())
      return invalid("breaker terminal choice is out of range");
    if (!allowed(options[indexed.value()]))
      continue;
    const std::array<std::int64_t, 1> tuple{
        static_cast<std::int64_t>(indexed.index())};
    accepted.AddTuple(tuple);
    hasAllowedChoice = true;
  }
  // An empty table is the exact proof that this breaker branch has no terminal
  // choice; CP-SAT, rather than an ad-hoc precheck, owns that infeasibility.
  (void)hasAllowedChoice;
  return llvm::Error::success();
}

llvm::Error constrainExternalDisposition(
    CpModelBuilder &model,
    const SpatialLocalDispositionModel &localDispositions,
    PnrIndex logicalNet) {
  const auto local = localDispositions.localForLogicalNet(logicalNet);
  if (!local)
    return invalid("breaker logical net is outside its disposition region");
  model.AddEquality(localDispositions.variables()[*local],
                    localDispositions.externalValue(*local));
  return llvm::Error::success();
}

llvm::Error constrainRegisterFifoDisposition(
    CpModelBuilder &model,
    const SpatialLocalDispositionModel &localDispositions,
    PnrIndex logicalNet) {
  const auto local = localDispositions.localForLogicalNet(logicalNet);
  if (!local)
    return invalid("breaker logical net is outside its disposition region");
  const auto selected = localDispositions.localSelected(logicalNet);
  if (!selected)
    return invalid("breaker logical net has no local-selection predicate");
  model.AddEquality(*selected, 1);
  return llvm::Error::success();
}

llvm::Error pinRepairRegionToCurrent(
    CpModelBuilder &model, const SpatialCandidateState &candidate,
    const SpatialBindingRelationModel &bindings,
    llvm::ArrayRef<IntVar> bindingVariables,
    llvm::ArrayRef<int> decisionVariables,
    const SpatialLocalDispositionModel &localDispositions) {
  for (PnrIndex decision = 0; decision < decisionVariables.size(); ++decision) {
    const int local = decisionVariables[decision];
    if (local < 0)
      continue;
    if (static_cast<std::size_t>(local) >= bindingVariables.size())
      return invalid("tag breaker binding variable is out of range");
    auto current =
        currentExactRepairBindingChoice(candidate, bindings, decision);
    if (!current)
      return current.takeError();
    model.AddEquality(bindingVariables[local], *current);
  }
  if (localDispositions.variables().size() !=
      localDispositions.currentValues().size())
    return invalid("tag breaker disposition inventory is malformed");
  for (auto [variable, current] : llvm::zip_equal(
           localDispositions.variables(), localDispositions.currentValues()))
    model.AddEquality(variable, current);
  return llvm::Error::success();
}

llvm::Error addTraversalLocalTerminalConstraints(
    CpModelBuilder &model, const SpatialCandidateState &candidate,
    const SpatialBindingRelationModel &bindings,
    llvm::ArrayRef<IntVar> bindingVariables,
    llvm::ArrayRef<int> decisionVariables,
    const FrozenNoGoodResolvedLiteral &literal) {
  const auto &transfers = candidate.problem().transfers();
  const auto nets = transfers.logicalNets();
  if (literal.logicalNet >= nets.size())
    return invalid("traversal breaker logical net is out of range");
  const auto excludesTraversal = [&](const FrozenSpatialAttachmentOption &option) {
    return !option.localTraversal || *option.localTraversal != literal.target;
  };
  if (llvm::Error error = addTerminalChoicePredicate(
          model, candidate, bindings, bindingVariables, decisionVariables,
          transfers.logicalNetSourceBindings()[literal.logicalNet],
          excludesTraversal))
    return error;
  const FrozenSpatialLogicalNet &net = nets[literal.logicalNet];
  if (literal.sink) {
    if (*literal.sink >= net.sinkCount)
      return invalid("traversal breaker sink is out of range");
    return addTerminalChoicePredicate(
        model, candidate, bindings, bindingVariables, decisionVariables,
        transfers.logicalNetSinkBindings()[net.sinkOffset + *literal.sink],
        excludesTraversal);
  }
  for (FrozenSpatialTerminalBinding sink :
       transfers.logicalNetSinkBindings().slice(net.sinkOffset,
                                                net.sinkCount))
    if (llvm::Error error = addTerminalChoicePredicate(
            model, candidate, bindings, bindingVariables, decisionVariables,
            sink, excludesTraversal))
      return error;
  return llvm::Error::success();
}

} // namespace

llvm::Expected<std::vector<SpatialRuntimeCounterexampleBreaker>>
loom::pnr::detail::enumerateSpatialRuntimeCounterexampleBreakers(
    const SpatialCandidateState &candidate, PnrIndex clauseOrdinal) {
  const FrozenSpatialPnrProblem &problem = candidate.problem();
  const auto clauses = problem.constraints().resolvedNoGoods();
  const auto literals = problem.constraints().resolvedNoGoodLiterals();
  if (clauseOrdinal >= clauses.size())
    return invalid("clause ordinal is out of range");
  const FrozenNoGoodResolvedClause &clause = clauses[clauseOrdinal];
  if (clause.literalOffset > literals.size() || clause.literalCount == 0 ||
      clause.literalCount > literals.size() - clause.literalOffset)
    return invalid("clause literal range is malformed");

  std::vector<SpatialRuntimeCounterexampleBreaker> result;
  std::set<PnrIndex> localDispositionNets;
  std::optional<PnrIndex> mappingIdentityLiteral;
  for (PnrIndex local = 0; local < clause.literalCount; ++local) {
    const FrozenNoGoodResolvedLiteral &literal =
        literals[clause.literalOffset + local];
    if (literal.kind != FrozenNoGoodResolvedLiteral::Kind::
                            SpatialMappingIdentityEquals &&
        localDispositionNets.insert(literal.logicalNet).second)
      result.push_back({
          clauseOrdinal, local,
          SpatialRuntimeCounterexampleBreakerKind::RegisterFifoDisposition,
          std::nullopt});
    SpatialRuntimeCounterexampleBreakerKind kind;
    switch (literal.kind) {
    case FrozenNoGoodResolvedLiteral::Kind::TransferAttachmentEquals:
      kind = SpatialRuntimeCounterexampleBreakerKind::TransferAttachment;
      break;
    case FrozenNoGoodResolvedLiteral::Kind::NetUsesTraversal:
      kind = SpatialRuntimeCounterexampleBreakerKind::NetTraversal;
      break;
    case FrozenNoGoodResolvedLiteral::Kind::NetTagEquals:
      kind = SpatialRuntimeCounterexampleBreakerKind::NetTag;
      {
        auto alternatives = enumerateTagAlternatives(candidate, literal);
        if (!alternatives)
          return alternatives.takeError();
        for (llvm::APInt &value : *alternatives)
          result.push_back(
              {clauseOrdinal, local, kind, std::move(value)});
      }
      continue;
    case FrozenNoGoodResolvedLiteral::Kind::SpatialMappingIdentityEquals:
      mappingIdentityLiteral = local;
      continue;
    }
    result.push_back({clauseOrdinal, local, kind, std::nullopt});
  }
  // A learned clause carries the complete-assignment literal beside one or
  // more certificate-derived local anchors. Breaking any such anchor also
  // makes the exact parent Mapping identity false, so a separate identity
  // branch would only reopen unrelated decisions. An explicitly authored
  // identity-only clause has no local anchor and retains its own typed branch.
  if (result.empty() && mappingIdentityLiteral)
    result.push_back(
        {clauseOrdinal, *mappingIdentityLiteral,
         SpatialRuntimeCounterexampleBreakerKind::MappingIdentity,
         std::nullopt});
  return result;
}

llvm::Expected<const FrozenNoGoodResolvedLiteral *>
loom::pnr::detail::resolveSpatialRuntimeCounterexampleBreaker(
    const FrozenSpatialPnrProblem &problem,
    const SpatialRuntimeCounterexampleBreaker &breaker) {
  const auto clauses = problem.constraints().resolvedNoGoods();
  const auto literals = problem.constraints().resolvedNoGoodLiterals();
  if (breaker.clauseOrdinal >= clauses.size())
    return invalid("breaker clause ordinal is out of range");
  const FrozenNoGoodResolvedClause &clause = clauses[breaker.clauseOrdinal];
  if (clause.literalOffset > literals.size() ||
      clause.literalCount > literals.size() - clause.literalOffset ||
      breaker.clauseLocalLiteralOrdinal >= clause.literalCount)
    return invalid("breaker literal ordinal is out of range");
  return &literals[clause.literalOffset +
                   breaker.clauseLocalLiteralOrdinal];
}

llvm::Error
loom::pnr::detail::addSpatialRuntimeCounterexampleBreakerConstraint(
    CpModelBuilder &model, const SpatialCandidateState &candidate,
    const SpatialBindingRelationModel &bindings,
    llvm::ArrayRef<IntVar> bindingVariables,
    llvm::ArrayRef<int> decisionVariables,
    const SpatialLocalDispositionModel &localDispositions,
    const SpatialRuntimeCounterexampleBreaker &breaker) {
  auto resolved =
      resolveSpatialRuntimeCounterexampleBreaker(candidate.problem(), breaker);
  if (!resolved)
    return resolved.takeError();
  const FrozenNoGoodResolvedLiteral &literal = **resolved;
  switch (breaker.kind) {
  case SpatialRuntimeCounterexampleBreakerKind::RegisterFifoDisposition:
    if (literal.kind == FrozenNoGoodResolvedLiteral::Kind::
                            SpatialMappingIdentityEquals)
      return invalid("Mapping-wide literal has no local disposition owner");
    return constrainRegisterFifoDisposition(model, localDispositions,
                                             literal.logicalNet);
  case SpatialRuntimeCounterexampleBreakerKind::TransferAttachment: {
    if (literal.kind !=
        FrozenNoGoodResolvedLiteral::Kind::TransferAttachmentEquals)
      return invalid("attachment breaker names another literal kind");
    if (llvm::Error error = constrainExternalDisposition(
            model, localDispositions, literal.logicalNet))
      return error;
    const auto &transfers = candidate.problem().transfers();
    const FrozenSpatialLogicalNet &net =
        transfers.logicalNets()[literal.logicalNet];
    FrozenSpatialTerminalBinding binding =
        transfers.logicalNetSourceBindings()[literal.logicalNet];
    if (literal.sink) {
      if (*literal.sink >= net.sinkCount)
        return invalid("attachment breaker sink is out of range");
      binding = transfers.logicalNetSinkBindings()[net.sinkOffset +
                                                   *literal.sink];
    }
    return addTerminalChoicePredicate(
        model, candidate, bindings, bindingVariables, decisionVariables,
        binding, [&](const FrozenSpatialAttachmentOption &option) {
          return option.endpoint != literal.target;
        });
  }
  case SpatialRuntimeCounterexampleBreakerKind::NetTraversal:
    if (literal.kind != FrozenNoGoodResolvedLiteral::Kind::NetUsesTraversal)
      return invalid("traversal breaker names another literal kind");
    if (llvm::Error error = constrainExternalDisposition(
            model, localDispositions, literal.logicalNet))
      return error;
    return addTraversalLocalTerminalConstraints(
        model, candidate, bindings, bindingVariables, decisionVariables,
        literal);
  case SpatialRuntimeCounterexampleBreakerKind::NetTag:
    if (literal.kind != FrozenNoGoodResolvedLiteral::Kind::NetTagEquals)
      return invalid("tag breaker names another literal kind");
    if (!literal.tagValue || !breaker.physicalTagValue ||
        compareUnsigned(*literal.tagValue, *breaker.physicalTagValue) == 0)
      return invalid("tag breaker has no distinct exact replacement value");
    return pinRepairRegionToCurrent(model, candidate, bindings,
                                    bindingVariables, decisionVariables,
                                    localDispositions);
  case SpatialRuntimeCounterexampleBreakerKind::MappingIdentity:
    if (literal.kind != FrozenNoGoodResolvedLiteral::Kind::
                            SpatialMappingIdentityEquals)
      return invalid("Mapping identity breaker names another literal kind");
    return invalid(
        "exact Mapping identity breaker requires a finite decision branch");
  }
  llvm_unreachable("unknown runtime-counterexample breaker kind");
}

llvm::Expected<bool>
loom::pnr::detail::spatialRuntimeTraversalRequiresRouteCut(
    const SpatialCandidateState &candidate,
    const SpatialRuntimeCounterexampleBreaker &breaker) {
  if (breaker.kind != SpatialRuntimeCounterexampleBreakerKind::NetTraversal)
    return false;
  auto resolved =
      resolveSpatialRuntimeCounterexampleBreaker(candidate.problem(), breaker);
  if (!resolved)
    return resolved.takeError();
  const FrozenNoGoodResolvedLiteral &literal = **resolved;
  if (literal.kind != FrozenNoGoodResolvedLiteral::Kind::NetUsesTraversal)
    return invalid("traversal route-cut query names another literal kind");
  if (candidate.usesRegisterFifo(literal.logicalNet))
    return false;
  const RouteTreeState &route = candidate.routeTree(literal.logicalNet);
  if (route.isUnrouted())
    return false;
  const auto arcs = candidate.problem().routing().routingArcs();
  const auto sources = candidate.problem().routing().arcSources();
  if (!literal.sink) {
    for (const RouteTreeNode &node : route.nodeStorage()) {
      if (!node.isActive() || node.parentArc == getInvalidPnrIndex())
        continue;
      if (node.parentArc >= arcs.size())
        return invalid("traversal route-cut arc is out of range");
      if (arcs[node.parentArc].traversal == literal.target)
        return true;
    }
    return false;
  }
  const auto endpoint = route.sinkEndpoint(*literal.sink);
  if (!endpoint)
    return false;
  auto slot = route.findNode(*endpoint);
  if (!slot)
    return invalid("traversal route-cut sink is absent");
  for (std::size_t depth = 0; depth <= route.nodeStorage().size(); ++depth) {
    const RouteTreeNode &node = route.node(*slot);
    if (node.parentArc == getInvalidPnrIndex())
      return false;
    if (node.parentArc >= arcs.size() || node.parentArc >= sources.size())
      return invalid("traversal route-cut branch arc is out of range");
    if (arcs[node.parentArc].traversal == literal.target)
      return true;
    slot = route.findNode(sources[node.parentArc]);
    if (!slot)
      return invalid("traversal route-cut branch parent is absent");
  }
  return invalid("traversal route-cut branch is cyclic");
}
