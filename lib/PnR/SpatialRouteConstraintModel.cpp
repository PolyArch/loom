#include "SpatialRouteConstraintModel.h"

#include "Dataflow/IR/DataflowReferenceCodec.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "PnR/SpatialCandidateState.h"
#include "PnR/SpatialPnrProblem.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <cassert>
#include <limits>
#include <map>
#include <numeric>
#include <system_error>
#include <utility>
#include <variant>

using namespace loom;
using namespace loom::fabric;
using namespace loom::mapping;
using namespace loom::pnr;
using namespace loom::pnr::detail;

namespace {

using Projection = ::mapping::SpatialConstraintProjection;
using Key = std::vector<std::uint8_t>;

llvm::Error freezeInvalid(Projection projection, const llvm::Twine &message) {
  return llvm::make_error<SpatialPnrFreezeFailure>(
      SpatialPnrFreezeFailureKind::Invalid,
      ("invalid Spatial route constraint projection: " + message).str(),
      projection);
}

llvm::Error runtimeInvalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "spatial_route_constraint_invalid: %s", message.str().c_str());
}

llvm::Expected<Key>
dataflowKey(const ArtifactIdentity &identity,
            const dataflow::CanonicalGraphProducerEndpointRef &reference) {
  return dataflow::encodeDataflowReference(identity, reference);
}

template <typename Ref> Key fabricKey(const Ref &reference) {
  return canonicalFabricBytes(reference);
}

class DisjointSet final {
public:
  explicit DisjointSet(std::size_t size) : parent_(size) {
    std::iota(parent_.begin(), parent_.end(), std::size_t{0});
  }

  std::size_t find(std::size_t value) {
    while (parent_[value] != value) {
      parent_[value] = parent_[parent_[value]];
      value = parent_[value];
    }
    return value;
  }

  void unite(std::size_t lhs, std::size_t rhs) {
    lhs = find(lhs);
    rhs = find(rhs);
    if (lhs == rhs)
      return;
    if (lhs > rhs)
      std::swap(lhs, rhs);
    parent_[rhs] = lhs;
  }

private:
  std::vector<std::size_t> parent_;
};

llvm::Expected<PnrIndex> checkedIndex(std::size_t value, llvm::StringRef table,
                                      PnrCapacityMeasure measure) {
  return checkedPnrIndex(
      {"FrozenSpatialRouteConstraintModel", table, table, measure}, value);
}

bool bitIsSet(llvm::ArrayRef<std::uint64_t> bits, PnrIndex value) {
  return value / 64 < bits.size() &&
         (bits[value / 64] & (std::uint64_t{1} << (value % 64))) != 0;
}

void setBit(std::vector<std::uint64_t> &bits, PnrIndex value) {
  bits[value / 64] |= std::uint64_t{1} << (value % 64);
}

bool intersects(llvm::ArrayRef<std::uint64_t> lhs,
                llvm::ArrayRef<std::uint64_t> rhs) {
  assert(lhs.size() == rhs.size());
  for (std::size_t word = 0; word < lhs.size(); ++word)
    if ((lhs[word] & rhs[word]) != 0)
      return true;
  return false;
}

bool equalBits(llvm::ArrayRef<std::uint64_t> lhs,
               llvm::ArrayRef<std::uint64_t> rhs) {
  return llvm::equal(lhs, rhs);
}

template <typename Ref>
llvm::Expected<std::vector<PnrIndex>> decodeDomain(
    Projection projection,
    const std::optional<llvm::ArrayRef<SpatialConstraintDomainValue>> &domain,
    const std::map<Key, PnrIndex> &ordinals) {
  std::vector<PnrIndex> result;
  if (!domain)
    return result;
  result.reserve(domain->size());
  for (const SpatialConstraintDomainValue &value : *domain) {
    const Ref *reference = std::get_if<Ref>(&value);
    if (!reference)
      return freezeInvalid(projection,
                           "domain contains a value of the wrong type");
    const auto found = ordinals.find(fabricKey(*reference));
    if (found == ordinals.end())
      return freezeInvalid(projection,
                           "domain names a value absent from the frozen owner");
    result.push_back(found->second);
  }
  llvm::sort(result);
  result.erase(std::unique(result.begin(), result.end()), result.end());
  return result;
}

} // namespace

llvm::Expected<std::shared_ptr<const SpatialRouteConstraintModel>>
SpatialRouteConstraintModel::create(const ArtifactIdentity &dataflowIdentity,
                                    const FrozenConstraintIndex &constraints,
                                    const FrozenSpatialTransferIndex &transfers,
                                    const FrozenSpatialResourceIndex &resources,
                                    const FrozenSpatialRoutingGraph &routing) {
  auto result = std::make_shared<SpatialRouteConstraintModel>();
  const auto logicalNets = transfers.logicalNets();
  const std::size_t netCount = logicalNets.size();
  if (netCount > getPnrIndexMax())
    return freezeInvalid(Projection::NetSelectedPhysicalTraversals,
                         "logical-net count exceeds PnrIndex");

  std::map<Key, PnrIndex> netOrdinals;
  for (auto [ordinal, net] : llvm::enumerate(logicalNets)) {
    auto key = dataflowKey(dataflowIdentity, net.producer);
    if (!key)
      return freezeInvalid(
          Projection::NetSelectedPhysicalTraversals,
          "logical-net producer reference cannot be encoded: " +
              llvm::toString(key.takeError()));
    const bool inserted =
        netOrdinals.try_emplace(std::move(*key), static_cast<PnrIndex>(ordinal))
            .second;
    if (!inserted)
      return freezeInvalid(Projection::NetSelectedPhysicalTraversals,
                           "logical-net producer key is not unique");
  }

  std::map<Key, PnrIndex> traversalOrdinals;
  for (auto [ordinal, traversal] : llvm::enumerate(routing.traversals()))
    if (!traversalOrdinals
             .try_emplace(fabricKey(traversal.reference),
                          static_cast<PnrIndex>(ordinal))
             .second)
      return freezeInvalid(Projection::NetSelectedPhysicalTraversals,
                           "physical traversal reference is not unique");
  std::map<Key, PnrIndex> resourceOrdinals;
  for (auto [ordinal, state] : llvm::enumerate(resources.resourceStates()))
    if (!resourceOrdinals
             .try_emplace(fabricKey(state.reference),
                          static_cast<PnrIndex>(ordinal))
             .second)
      return freezeInvalid(Projection::NetTraversalResourceStates,
                           "resource-state reference is not unique");

  result->traversalDomains_.resize(netCount);
  result->resourceStateDomains_.resize(netCount);
  result->netConstraintFlags_.assign(netCount, 0);
  const FrozenConstraintShard &traversalShard =
      constraints.shard(Projection::NetSelectedPhysicalTraversals);
  const FrozenConstraintShard &resourceShard =
      constraints.shard(Projection::NetTraversalResourceStates);
  for (PnrIndex net = 0; net < netCount; ++net) {
    const SpatialConstraintSubject subject{logicalNets[net].producer};
    const auto traversalDomain = traversalShard.restrictedDomain(subject);
    auto traversals = decodeDomain<FabricPhysicalTraversalRef>(
        Projection::NetSelectedPhysicalTraversals, traversalDomain,
        traversalOrdinals);
    if (!traversals)
      return traversals.takeError();
    SpatialRouteConstraintDomain &traversalRecord =
        result->traversalDomains_[net];
    traversalRecord.restricted = traversalDomain.has_value();
    auto traversalOffset = checkedIndex(result->traversalDomainValues_.size(),
                                        "route_traversal_domain_values",
                                        PnrCapacityMeasure::Offset);
    if (!traversalOffset)
      return traversalOffset.takeError();
    auto traversalCount =
        checkedIndex(traversals->size(), "route_traversal_domain_values",
                     PnrCapacityMeasure::Count);
    if (!traversalCount)
      return traversalCount.takeError();
    traversalRecord.valueOffset = *traversalOffset;
    traversalRecord.valueCount = *traversalCount;
    result->traversalDomainValues_.insert(result->traversalDomainValues_.end(),
                                          traversals->begin(),
                                          traversals->end());

    const auto resourceDomain = resourceShard.restrictedDomain(subject);
    auto states = decodeDomain<FabricResourceStateRef>(
        Projection::NetTraversalResourceStates, resourceDomain,
        resourceOrdinals);
    if (!states)
      return states.takeError();
    SpatialRouteConstraintDomain &resourceRecord =
        result->resourceStateDomains_[net];
    resourceRecord.restricted = resourceDomain.has_value();
    auto resourceOffset = checkedIndex(
        result->resourceStateDomainValues_.size(),
        "route_resource_state_domain_values", PnrCapacityMeasure::Offset);
    if (!resourceOffset)
      return resourceOffset.takeError();
    auto resourceCount =
        checkedIndex(states->size(), "route_resource_state_domain_values",
                     PnrCapacityMeasure::Count);
    if (!resourceCount)
      return resourceCount.takeError();
    resourceRecord.valueOffset = *resourceOffset;
    resourceRecord.valueCount = *resourceCount;
    result->resourceStateDomainValues_.insert(
        result->resourceStateDomainValues_.end(), states->begin(),
        states->end());
    result->netConstraintFlags_[net] = static_cast<std::uint8_t>(
        traversalRecord.restricted || resourceRecord.restricted);
  }

  DisjointSet equality(netCount);
  std::vector<std::vector<PnrIndex>> netRelations(netCount);
  const auto appendRelations =
      [&](const FrozenConstraintShard &shard,
          SpatialRouteConstraintProjection projection,
          llvm::ArrayRef<FrozenConstraintRelation> relations,
          SpatialRouteConstraintRelationKind kind) -> llvm::Error {
    const Projection owner =
        projection == SpatialRouteConstraintProjection::Traversal
            ? Projection::NetSelectedPhysicalTraversals
            : Projection::NetTraversalResourceStates;
    for (const FrozenConstraintRelation &relation : relations) {
      auto relationOrdinal =
          checkedIndex(result->relations_.size(), "route_relations",
                       PnrCapacityMeasure::Count);
      if (!relationOrdinal)
        return relationOrdinal.takeError();
      auto memberOffset =
          checkedIndex(result->relationMembers_.size(),
                       "route_relation_members", PnrCapacityMeasure::Offset);
      if (!memberOffset)
        return memberOffset.takeError();
      std::vector<PnrIndex> members;
      members.reserve(relation.memberCount);
      for (PnrIndex subjectOrdinal : shard.relationMembers().slice(
               relation.memberOffset, relation.memberCount)) {
        if (subjectOrdinal >= shard.subjects().size())
          return freezeInvalid(owner,
                               "relation contains an out-of-range subject");
        const auto *producer =
            std::get_if<dataflow::CanonicalGraphProducerEndpointRef>(
                &shard.subjects()[subjectOrdinal]);
        if (!producer)
          return freezeInvalid(owner, "relation has a non-net subject");
        auto key = dataflowKey(dataflowIdentity, *producer);
        if (!key)
          return freezeInvalid(
              owner, "relation subject reference cannot be encoded: " +
                         llvm::toString(key.takeError()));
        const auto found = netOrdinals.find(*key);
        if (found == netOrdinals.end())
          return freezeInvalid(owner, "relation names a foreign logical net");
        members.push_back(found->second);
      }
      llvm::sort(members);
      if (std::adjacent_find(members.begin(), members.end()) != members.end() ||
          members.size() < 2)
        return freezeInvalid(owner,
                             "relation members are not distinct and variadic");
      result->relationMembers_.insert(result->relationMembers_.end(),
                                      members.begin(), members.end());
      auto memberCount = checkedIndex(members.size(), "route_relation_members",
                                      PnrCapacityMeasure::Count);
      if (!memberCount)
        return memberCount.takeError();
      result->relations_.push_back(
          {projection, kind, *memberOffset, *memberCount});
      for (PnrIndex net : members) {
        netRelations[net].push_back(*relationOrdinal);
        result->netConstraintFlags_[net] = 1;
      }
      if (kind == SpatialRouteConstraintRelationKind::Equal)
        for (PnrIndex member : llvm::drop_begin(members))
          equality.unite(members.front(), member);
    }
    return llvm::Error::success();
  };

  if (llvm::Error error = appendRelations(
          traversalShard, SpatialRouteConstraintProjection::Traversal,
          traversalShard.equalityClasses(),
          SpatialRouteConstraintRelationKind::Equal))
    return std::move(error);
  if (llvm::Error error = appendRelations(
          traversalShard, SpatialRouteConstraintProjection::Traversal,
          traversalShard.disjointGroups(),
          SpatialRouteConstraintRelationKind::Disjoint))
    return std::move(error);
  if (llvm::Error error = appendRelations(
          resourceShard, SpatialRouteConstraintProjection::ResourceState,
          resourceShard.equalityClasses(),
          SpatialRouteConstraintRelationKind::Equal))
    return std::move(error);
  if (llvm::Error error = appendRelations(
          resourceShard, SpatialRouteConstraintProjection::ResourceState,
          resourceShard.disjointGroups(),
          SpatialRouteConstraintRelationKind::Disjoint))
    return std::move(error);

  result->netRelationOffsets_.reserve(netCount + 1);
  result->netRelationOffsets_.push_back(0);
  for (std::vector<PnrIndex> &relations : netRelations) {
    llvm::sort(relations);
    result->netRelations_.insert(result->netRelations_.end(), relations.begin(),
                                 relations.end());
    auto offset =
        checkedIndex(result->netRelations_.size(), "route_relation_incidence",
                     PnrCapacityMeasure::Offset);
    if (!offset)
      return offset.takeError();
    result->netRelationOffsets_.push_back(*offset);
  }

  std::map<std::size_t, std::vector<PnrIndex>> components;
  for (PnrIndex net = 0; net < netCount; ++net)
    components[equality.find(net)].push_back(net);
  result->netEqualityComponents_.resize(netCount);
  result->equalityComponentOffsets_.push_back(0);
  for (const auto &[root, members] : components) {
    (void)root;
    auto component =
        checkedIndex(result->equalityComponentOffsets_.size() - 1,
                     "route_equality_components", PnrCapacityMeasure::Count);
    if (!component)
      return component.takeError();
    for (PnrIndex net : members)
      result->netEqualityComponents_[net] = *component;
    result->equalityComponentMembers_.insert(
        result->equalityComponentMembers_.end(), members.begin(),
        members.end());
    auto offset = checkedIndex(result->equalityComponentMembers_.size(),
                               "route_equality_component_members",
                               PnrCapacityMeasure::Offset);
    if (!offset)
      return offset.takeError();
    result->equalityComponentOffsets_.push_back(*offset);
  }
  return std::shared_ptr<const SpatialRouteConstraintModel>(std::move(result));
}

llvm::ArrayRef<PnrIndex>
SpatialRouteConstraintModel::equalityClosure(PnrIndex logicalNet) const {
  assert(logicalNet < netEqualityComponents_.size());
  const PnrIndex component = netEqualityComponents_[logicalNet];
  return llvm::ArrayRef(equalityComponentMembers_)
      .slice(equalityComponentOffsets_[component],
             equalityComponentOffsets_[component + 1] -
                 equalityComponentOffsets_[component]);
}

llvm::ArrayRef<PnrIndex>
SpatialRouteConstraintModel::netRelations(PnrIndex logicalNet) const {
  assert(logicalNet + 1 < netRelationOffsets_.size());
  return llvm::ArrayRef(netRelations_)
      .slice(netRelationOffsets_[logicalNet],
             netRelationOffsets_[logicalNet + 1] -
                 netRelationOffsets_[logicalNet]);
}

llvm::ArrayRef<PnrIndex> SpatialRouteConstraintModel::relationMembers(
    const SpatialRouteConstraintRelation &relation) const {
  return llvm::ArrayRef(relationMembers_)
      .slice(relation.memberOffset, relation.memberCount);
}

bool SpatialRouteConstraintModel::netHasConstraints(PnrIndex logicalNet) const {
  assert(logicalNet < netConstraintFlags_.size());
  return netConstraintFlags_[logicalNet] != 0;
}

void SpatialRouteConstraintScratch::clearBits(
    std::vector<std::uint64_t> &bits) {
  std::fill(bits.begin(), bits.end(), 0);
}

llvm::Error
SpatialRouteConstraintScratch::prepare(const FrozenSpatialPnrProblem &problem) {
  problem_ = &problem;
  model_ = &problem.routeConstraints();
  const std::size_t traversalWords =
      (problem.routing().traversals().size() + 63) / 64;
  const std::size_t resourceWords =
      (problem.resources().resourceStates().size() + 63) / 64;
  activeTraversalBits_.assign(
      problem.activeRouting().activeTraversalBits().begin(),
      problem.activeRouting().activeTraversalBits().end());
  if (activeTraversalBits_.size() != traversalWords)
    return runtimeInvalid("active traversal domain has the wrong width");
  eligibleTraversalBits_.assign(traversalWords, 0);
  selectedTraversalBits_.assign(traversalWords, 0);
  referenceBits_.assign(std::max(traversalWords, resourceWords), 0);
  seenBits_.assign(std::max(traversalWords, resourceWords), 0);
  selectedResourceStateBits_.assign(resourceWords, 0);
  pendingNets_.assign(problem.transfers().logicalNets().size(), 0);
  relationMarks_.assign(model_->relations().size(), 0);
  affectedRelations_.clear();
  affectedRelations_.reserve(model_->relations().size());
  relationEpoch_ = 0;
  sweepActive_ = false;
  return llvm::Error::success();
}

llvm::Error SpatialRouteConstraintScratch::beginSweep(
    llvm::ArrayRef<PnrIndex> logicalNets) {
  if (!problem_ || !model_)
    return runtimeInvalid("scratch is not prepared");
  std::fill(pendingNets_.begin(), pendingNets_.end(), 0);
  for (PnrIndex net : logicalNets) {
    if (net >= pendingNets_.size())
      return runtimeInvalid("constraint sweep net is out of range");
    if (pendingNets_[net])
      return runtimeInvalid("constraint sweep repeats a logical net");
    pendingNets_[net] = 1;
  }
  sweepActive_ = true;
  return llvm::Error::success();
}

llvm::Error SpatialRouteConstraintScratch::collectSelected(
    const SpatialCandidateState &candidate, PnrIndex logicalNet,
    SpatialRouteConstraintProjection projection,
    std::vector<std::uint64_t> &bits) {
  clearBits(bits);
  if (logicalNet >= candidate.problem().transfers().logicalNets().size())
    return runtimeInvalid("selected-set net is out of range");
  const auto &routing = candidate.problem().routing();
  const RouteTreeState &tree = candidate.routeTree(logicalNet);
  for (const RouteTreeNode &node : tree.nodeStorage()) {
    if (!node.isActive() || node.parentArc == getInvalidPnrIndex())
      continue;
    if (node.parentArc >= routing.routingArcs().size())
      return runtimeInvalid("RouteTree arc is out of range");
    const PnrIndex traversal = routing.routingArcs()[node.parentArc].traversal;
    if (traversal >= routing.traversals().size())
      return runtimeInvalid("RouteTree traversal is out of range");
    if (projection == SpatialRouteConstraintProjection::Traversal) {
      setBit(bits, traversal);
      continue;
    }
    const FrozenSpatialTraversal &record = routing.traversals()[traversal];
    for (PnrIndex state : routing.traversalResourceStates().slice(
             record.resourceStateOffset, record.resourceStateCount)) {
      if (state >= candidate.problem().resources().resourceStates().size())
        return runtimeInvalid("traversal resource state is out of range");
      setBit(bits, state);
    }
  }
  return llvm::Error::success();
}

bool SpatialRouteConstraintScratch::traversalAllowedByResourceBits(
    PnrIndex traversal, llvm::ArrayRef<std::uint64_t> bits,
    bool requireSubset) const {
  const auto &routing = problem_->routing();
  const FrozenSpatialTraversal &record = routing.traversals()[traversal];
  bool intersectsSelected = false;
  for (PnrIndex state : routing.traversalResourceStates().slice(
           record.resourceStateOffset, record.resourceStateCount)) {
    const bool selected = bitIsSet(bits, state);
    if (requireSubset && !selected)
      return false;
    intersectsSelected |= selected;
  }
  return requireSubset || !intersectsSelected;
}

llvm::Expected<llvm::ArrayRef<std::uint64_t>>
SpatialRouteConstraintScratch::eligibleTraversals(
    const SpatialCandidateState &candidate, PnrIndex logicalNet) {
  if (!problem_ || !model_)
    return runtimeInvalid("constraint scratch is not prepared");
  if (!sweepActive_)
    return runtimeInvalid("constraint sweep is not active");
  if (&candidate.problem() != problem_)
    return runtimeInvalid("constraint sweep belongs to another freeze");
  if (logicalNet >= pendingNets_.size() || !pendingNets_[logicalNet])
    return runtimeInvalid("logical net is not pending in the route sweep");
  eligibleTraversalBits_ = activeTraversalBits_;
  if (!model_->netHasConstraints(logicalNet))
    return llvm::ArrayRef(eligibleTraversalBits_);

  const SpatialRouteConstraintDomain &traversalDomain =
      model_->traversalDomains_[logicalNet];
  if (traversalDomain.restricted) {
    clearBits(selectedTraversalBits_);
    for (PnrIndex traversal :
         llvm::ArrayRef(model_->traversalDomainValues_)
             .slice(traversalDomain.valueOffset, traversalDomain.valueCount))
      setBit(selectedTraversalBits_, traversal);
    for (std::size_t word = 0; word < eligibleTraversalBits_.size(); ++word)
      eligibleTraversalBits_[word] &= selectedTraversalBits_[word];
  }
  const SpatialRouteConstraintDomain &resourceDomain =
      model_->resourceStateDomains_[logicalNet];
  if (resourceDomain.restricted) {
    clearBits(selectedResourceStateBits_);
    for (PnrIndex state :
         llvm::ArrayRef(model_->resourceStateDomainValues_)
             .slice(resourceDomain.valueOffset, resourceDomain.valueCount))
      setBit(selectedResourceStateBits_, state);
    for (PnrIndex traversal = 0;
         traversal < problem_->routing().traversals().size(); ++traversal)
      if (bitIsSet(eligibleTraversalBits_, traversal) &&
          !traversalAllowedByResourceBits(traversal, selectedResourceStateBits_,
                                          true))
        eligibleTraversalBits_[traversal / 64] &=
            ~(std::uint64_t{1} << (traversal % 64));
  }

  for (PnrIndex relationOrdinal : model_->netRelations(logicalNet)) {
    const SpatialRouteConstraintRelation &relation =
        model_->relations()[relationOrdinal];
    for (PnrIndex peer : model_->relationMembers(relation)) {
      if (peer == logicalNet || pendingNets_[peer])
        continue;
      std::vector<std::uint64_t> &selected =
          relation.projection == SpatialRouteConstraintProjection::Traversal
              ? selectedTraversalBits_
              : selectedResourceStateBits_;
      if (llvm::Error error =
              collectSelected(candidate, peer, relation.projection, selected))
        return std::move(error);
      if (relation.projection == SpatialRouteConstraintProjection::Traversal) {
        for (std::size_t word = 0; word < eligibleTraversalBits_.size(); ++word)
          if (relation.kind == SpatialRouteConstraintRelationKind::Equal)
            eligibleTraversalBits_[word] &= selected[word];
          else
            eligibleTraversalBits_[word] &= ~selected[word];
        continue;
      }
      const bool requireSubset =
          relation.kind == SpatialRouteConstraintRelationKind::Equal;
      for (PnrIndex traversal = 0;
           traversal < problem_->routing().traversals().size(); ++traversal)
        if (bitIsSet(eligibleTraversalBits_, traversal) &&
            !traversalAllowedByResourceBits(traversal, selected, requireSubset))
          eligibleTraversalBits_[traversal / 64] &=
              ~(std::uint64_t{1} << (traversal % 64));
    }
  }
  return llvm::ArrayRef(eligibleTraversalBits_);
}

llvm::Error SpatialRouteConstraintScratch::finishNet(PnrIndex logicalNet) {
  if (!sweepActive_ || logicalNet >= pendingNets_.size() ||
      !pendingNets_[logicalNet])
    return runtimeInvalid("finished net is not pending in the route sweep");
  pendingNets_[logicalNet] = 0;
  return llvm::Error::success();
}

llvm::Error SpatialRouteConstraintScratch::verifyNetDomains(
    const SpatialCandidateState &candidate, PnrIndex logicalNet) {
  const auto checkDomain =
      [&](SpatialRouteConstraintProjection projection,
          const SpatialRouteConstraintDomain &domain,
          llvm::ArrayRef<PnrIndex> values,
          std::vector<std::uint64_t> &selected) -> llvm::Error {
    if (!domain.restricted)
      return llvm::Error::success();
    if (llvm::Error error =
            collectSelected(candidate, logicalNet, projection, selected))
      return error;
    clearBits(referenceBits_);
    for (PnrIndex value : values.slice(domain.valueOffset, domain.valueCount))
      setBit(referenceBits_, value);
    for (std::size_t word = 0; word < selected.size(); ++word)
      if ((selected[word] & ~referenceBits_[word]) != 0)
        return runtimeInvalid("selected route set escapes its domain");
    return llvm::Error::success();
  };
  if (llvm::Error error =
          checkDomain(SpatialRouteConstraintProjection::Traversal,
                      model_->traversalDomains_[logicalNet],
                      model_->traversalDomainValues_, selectedTraversalBits_))
    return error;
  return checkDomain(SpatialRouteConstraintProjection::ResourceState,
                     model_->resourceStateDomains_[logicalNet],
                     model_->resourceStateDomainValues_,
                     selectedResourceStateBits_);
}

llvm::Error SpatialRouteConstraintScratch::verifyRelation(
    const SpatialCandidateState &candidate, PnrIndex relationOrdinal) {
  if (relationOrdinal >= model_->relations().size())
    return runtimeInvalid("route relation ordinal is out of range");
  const SpatialRouteConstraintRelation &relation =
      model_->relations()[relationOrdinal];
  const auto members = model_->relationMembers(relation);
  clearBits(referenceBits_);
  clearBits(seenBits_);
  bool first = true;
  for (PnrIndex net : members) {
    std::vector<std::uint64_t> &selected =
        relation.projection == SpatialRouteConstraintProjection::Traversal
            ? selectedTraversalBits_
            : selectedResourceStateBits_;
    if (llvm::Error error =
            collectSelected(candidate, net, relation.projection, selected))
      return error;
    const std::size_t words = selected.size();
    if (relation.kind == SpatialRouteConstraintRelationKind::Equal) {
      if (first)
        std::copy(selected.begin(), selected.end(), referenceBits_.begin());
      else if (!equalBits(selected,
                          llvm::ArrayRef(referenceBits_).take_front(words)))
        return runtimeInvalid("selected route sets violate Equal");
    } else {
      if (intersects(selected, llvm::ArrayRef(seenBits_).take_front(words)))
        return runtimeInvalid("selected route sets violate Disjoint");
      for (std::size_t word = 0; word < words; ++word)
        seenBits_[word] |= selected[word];
    }
    first = false;
  }
  return llvm::Error::success();
}

llvm::Error SpatialRouteConstraintScratch::verifyAll(
    const SpatialCandidateState &candidate) {
  if (!problem_ || &candidate.problem() != problem_)
    return runtimeInvalid("verification candidate belongs to another freeze");
  for (PnrIndex net = 0;
       net < candidate.problem().transfers().logicalNets().size(); ++net)
    if (llvm::Error error = verifyNetDomains(candidate, net))
      return error;
  for (PnrIndex relation = 0; relation < model_->relations().size(); ++relation)
    if (llvm::Error error = verifyRelation(candidate, relation))
      return error;
  return llvm::Error::success();
}

llvm::Error SpatialRouteConstraintScratch::verifyAffected(
    const SpatialCandidateState &candidate,
    llvm::ArrayRef<PnrIndex> logicalNets) {
  if (!problem_ || &candidate.problem() != problem_)
    return runtimeInvalid("verification candidate belongs to another freeze");
  if (++relationEpoch_ == 0) {
    std::fill(relationMarks_.begin(), relationMarks_.end(), 0);
    relationEpoch_ = 1;
  }
  affectedRelations_.clear();
  for (PnrIndex net : logicalNets) {
    if (net >= candidate.problem().transfers().logicalNets().size())
      return runtimeInvalid("affected route net is out of range");
    if (llvm::Error error = verifyNetDomains(candidate, net))
      return error;
    for (PnrIndex relation : model_->netRelations(net)) {
      if (relationMarks_[relation] == relationEpoch_)
        continue;
      relationMarks_[relation] = relationEpoch_;
      affectedRelations_.push_back(relation);
    }
  }
  for (PnrIndex relation : affectedRelations_)
    if (llvm::Error error = verifyRelation(candidate, relation))
      return error;
  return llvm::Error::success();
}

std::size_t SpatialRouteConstraintScratch::retainedStorageBytes() const {
  return activeTraversalBits_.capacity() * sizeof(std::uint64_t) +
         eligibleTraversalBits_.capacity() * sizeof(std::uint64_t) +
         selectedTraversalBits_.capacity() * sizeof(std::uint64_t) +
         selectedResourceStateBits_.capacity() * sizeof(std::uint64_t) +
         referenceBits_.capacity() * sizeof(std::uint64_t) +
         seenBits_.capacity() * sizeof(std::uint64_t) +
         pendingNets_.capacity() * sizeof(std::uint8_t) +
         relationMarks_.capacity() * sizeof(std::uint64_t) +
         affectedRelations_.capacity() * sizeof(PnrIndex);
}
