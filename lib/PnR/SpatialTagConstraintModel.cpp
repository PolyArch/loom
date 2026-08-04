#include "SpatialTagConstraintModel.h"

#include "Dataflow/IR/DataflowReferenceCodec.h"
#include "PnR/SpatialPnrProblem.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <numeric>
#include <variant>
#include <vector>

using namespace loom;
using namespace loom::mapping;
using namespace loom::pnr;
using namespace loom::pnr::detail;

namespace {

using Projection = ::mapping::SpatialConstraintProjection;
using Key = std::vector<std::uint8_t>;

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::make_error<SpatialPnrFreezeFailure>(
      SpatialPnrFreezeFailureKind::Invalid,
      ("invalid Spatial tag constraint projection: " + message).str(),
      Projection::NetAssignedTagValues);
}

llvm::Expected<PnrIndex> checked(std::size_t value, llvm::StringRef table,
                                 PnrCapacityMeasure measure) {
  return checkedPnrIndex({"SpatialTagConstraintModel", table, table, measure},
                         value);
}

llvm::Expected<Key>
producerKey(const ArtifactIdentity &owner,
            const ::dataflow::CanonicalGraphProducerEndpointRef &producer) {
  return ::dataflow::encodeDataflowReference(owner, producer);
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

} // namespace

llvm::Expected<std::shared_ptr<const SpatialTagConstraintModel>>
SpatialTagConstraintModel::create(const ArtifactIdentity &dataflowIdentity,
                                  const FrozenSpatialTransferIndex &transfers,
                                  const FrozenConstraintIndex &constraints) {
  auto result = std::make_shared<SpatialTagConstraintModel>();
  const auto nets = transfers.logicalNets();
  if (llvm::Error error = preflightPnrIndexCapacity(
          {"SpatialTagConstraintModel", "logical_nets", "logical_nets",
           PnrCapacityMeasure::Count},
          nets.size()))
    return std::move(error);

  std::map<Key, PnrIndex> netOrdinals;
  for (auto [ordinal, net] : llvm::enumerate(nets)) {
    auto key = producerKey(dataflowIdentity, net.producer);
    if (!key)
      return key.takeError();
    auto checkedOrdinal =
        checked(ordinal, "logical_nets", PnrCapacityMeasure::Index);
    if (!checkedOrdinal)
      return checkedOrdinal.takeError();
    if (!netOrdinals.try_emplace(std::move(*key), *checkedOrdinal).second)
      return invalid("residual logical-net producer is not unique");
  }

  const FrozenConstraintShard &shard =
      constraints.shard(Projection::NetAssignedTagValues);
  const auto subjectNet =
      [&](PnrIndex subjectOrdinal) -> llvm::Expected<PnrIndex> {
    if (subjectOrdinal >= shard.subjects().size())
      return invalid("relation contains an out-of-range subject");
    const auto *producer =
        std::get_if<::dataflow::CanonicalGraphProducerEndpointRef>(
            &shard.subjects()[subjectOrdinal]);
    if (!producer)
      return invalid("relation contains a non-net subject");
    auto key = producerKey(dataflowIdentity, *producer);
    if (!key)
      return key.takeError();
    const auto found = netOrdinals.find(*key);
    if (found == netOrdinals.end())
      return invalid("relation names no residual logical net");
    return found->second;
  };

  DisjointSet equality(nets.size());
  for (const FrozenConstraintRelation &relation : shard.equalityClasses()) {
    const auto members = shard.relationMembers().slice(relation.memberOffset,
                                                       relation.memberCount);
    if (members.empty())
      return invalid("equality class has no member");
    auto first = subjectNet(members.front());
    if (!first)
      return first.takeError();
    for (PnrIndex subject : members.drop_front()) {
      auto current = subjectNet(subject);
      if (!current)
        return current.takeError();
      equality.unite(*first, *current);
    }
  }

  result->netClasses_.resize(nets.size());
  std::map<std::size_t, PnrIndex> rootClasses;
  for (std::size_t net = 0; net < nets.size(); ++net) {
    const std::size_t root = equality.find(net);
    auto [found, inserted] = rootClasses.try_emplace(root, 0);
    if (inserted) {
      auto classOrdinal = checked(rootClasses.size() - 1, "equality_classes",
                                  PnrCapacityMeasure::Index);
      if (!classOrdinal)
        return classOrdinal.takeError();
      found->second = *classOrdinal;
    }
    result->netClasses_[net] = found->second;
  }

  const std::size_t classCount = rootClasses.size();
  std::vector<PnrIndex> classCounts(classCount, 0);
  for (PnrIndex equalityClass : result->netClasses_)
    ++classCounts[equalityClass];
  result->classMemberOffsets_.reserve(classCount + 1);
  result->classMemberOffsets_.push_back(0);
  for (PnrIndex count : classCounts) {
    auto end =
        checkedPnrIndexAdd({"SpatialTagConstraintModel", "equality_classes",
                            "class_members", PnrCapacityMeasure::Offset},
                           result->classMemberOffsets_.back(), count);
    if (!end)
      return end.takeError();
    result->classMemberOffsets_.push_back(*end);
  }
  result->classMembers_.resize(nets.size());
  std::vector<PnrIndex> classCursors(result->classMemberOffsets_.begin(),
                                     result->classMemberOffsets_.end() - 1);
  for (PnrIndex net = 0; net < result->netClasses_.size(); ++net)
    result->classMembers_[classCursors[result->netClasses_[net]]++] = net;

  result->groupMemberOffsets_.push_back(0);
  for (const FrozenConstraintRelation &relation : shard.disjointGroups()) {
    std::vector<PnrIndex> classes;
    for (PnrIndex subject : shard.relationMembers().slice(
             relation.memberOffset, relation.memberCount)) {
      auto net = subjectNet(subject);
      if (!net)
        return net.takeError();
      classes.push_back(result->netClasses_[*net]);
    }
    llvm::sort(classes);
    classes.erase(std::unique(classes.begin(), classes.end()), classes.end());
    if (classes.size() < 2)
      return invalid("disjoint group collapses below two equality classes");
    result->groupMembers_.insert(result->groupMembers_.end(), classes.begin(),
                                 classes.end());
    auto end = checked(result->groupMembers_.size(), "group_members",
                       PnrCapacityMeasure::Offset);
    if (!end)
      return end.takeError();
    result->groupMemberOffsets_.push_back(*end);
  }

  std::vector<PnrIndex> classGroupCounts(classCount, 0);
  for (PnrIndex group = 0; group + 1 < result->groupMemberOffsets_.size();
       ++group)
    for (PnrIndex equalityClass : result->disjointGroupMembers(group))
      ++classGroupCounts[equalityClass];
  result->classGroupOffsets_.push_back(0);
  for (PnrIndex count : classGroupCounts) {
    auto end =
        checkedPnrIndexAdd({"SpatialTagConstraintModel", "equality_classes",
                            "class_groups", PnrCapacityMeasure::Offset},
                           result->classGroupOffsets_.back(), count);
    if (!end)
      return end.takeError();
    result->classGroupOffsets_.push_back(*end);
  }
  result->classGroups_.resize(result->groupMembers_.size());
  std::vector<PnrIndex> groupCursors(result->classGroupOffsets_.begin(),
                                     result->classGroupOffsets_.end() - 1);
  for (PnrIndex group = 0; group + 1 < result->groupMemberOffsets_.size();
       ++group)
    for (PnrIndex equalityClass : result->disjointGroupMembers(group))
      result->classGroups_[groupCursors[equalityClass]++] = group;

  result->hasRelations_ =
      !shard.equalityClasses().empty() || !shard.disjointGroups().empty();
  return std::shared_ptr<const SpatialTagConstraintModel>(std::move(result));
}

llvm::ArrayRef<PnrIndex>
SpatialTagConstraintModel::classMembers(PnrIndex equalityClass) const {
  return llvm::ArrayRef(classMembers_)
      .slice(classMemberOffsets_[equalityClass],
             classMemberOffsets_[equalityClass + 1] -
                 classMemberOffsets_[equalityClass]);
}

llvm::ArrayRef<PnrIndex>
SpatialTagConstraintModel::classDisjointGroups(PnrIndex equalityClass) const {
  return llvm::ArrayRef(classGroups_)
      .slice(classGroupOffsets_[equalityClass],
             classGroupOffsets_[equalityClass + 1] -
                 classGroupOffsets_[equalityClass]);
}

llvm::ArrayRef<PnrIndex>
SpatialTagConstraintModel::disjointGroupMembers(PnrIndex group) const {
  return llvm::ArrayRef(groupMembers_)
      .slice(groupMemberOffsets_[group],
             groupMemberOffsets_[group + 1] - groupMemberOffsets_[group]);
}
