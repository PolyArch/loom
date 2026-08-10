#include "Mapping/Artifact/SystemMappingConstraintSet.h"

#include "Mapping/Artifact/SystemMappingExecutionProjection.h"

#include "Common/ArtifactLocalReference.h"
#include "Dataflow/IR/DataflowReferenceCodec.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Mapping/Artifact/SystemMappingArtifact.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

using namespace loom;
using namespace loom::mapping;

char SystemMappingConstraintRejection::ID;

void SystemMappingConstraintRejection::log(llvm::raw_ostream &stream) const {
  stream << "system_mapping_rejected_by_constraint_set: projection "
         << ::mapping::stringifySystemConstraintProjection(projection_)
         << " clause " << clauseOrdinal_ << ": " << message_;
}

std::error_code SystemMappingConstraintRejection::convertToErrorCode() const {
  return std::make_error_code(std::errc::operation_not_permitted);
}

namespace {

using Projection = ::mapping::SystemConstraintProjection;
using Subject = SystemConstraintSubject;
using DomainValue = SystemConstraintDomainValue;

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      llvm::inconvertibleErrorCode(),
      "system_mapping_constraint_admission_invalid: " + message);
}

std::string byteKey(llvm::ArrayRef<std::uint8_t> bytes) {
  return std::string(reinterpret_cast<const char *>(bytes.data()),
                     bytes.size());
}

template <typename Ref> std::string fabricKey(const Ref &reference) {
  return byteKey(::loom::fabric::canonicalFabricBytes(reference));
}

template <typename Ref>
llvm::Expected<std::string> dataflowKey(const ArtifactIdentity &owner,
                                        const Ref &reference) {
  auto encoded = ::dataflow::encodeDataflowReference(owner, reference);
  if (!encoded)
    return encoded.takeError();
  return byteKey(*encoded);
}

std::string mappingKey(const ArtifactRootReference &reference) {
  return byteKey(encodeArtifactRootReference(reference));
}

int compareUnsigned(const llvm::APInt &lhs, const llvm::APInt &rhs) {
  const unsigned width = std::max(lhs.getBitWidth(), rhs.getBitWidth());
  const llvm::APInt left = lhs.zextOrTrunc(width);
  const llvm::APInt right = rhs.zextOrTrunc(width);
  if (left.ult(right))
    return -1;
  if (right.ult(left))
    return 1;
  return 0;
}

struct ExactSet final {
  std::vector<std::string> values;
};

struct UnsignedSet final {
  std::vector<SpatialConstraintUnsignedInterval> intervals;
};

using ProjectedSet = std::variant<ExactSet, UnsignedSet>;

void normalize(ExactSet &set) {
  llvm::sort(set.values);
  set.values.erase(std::unique(set.values.begin(), set.values.end()),
                   set.values.end());
}

llvm::Error normalize(UnsignedSet &set) {
  for (const auto &interval : set.intervals)
    if (compareUnsigned(interval.lower, interval.upper) >= 0)
      return invalid("projected interval is empty or reversed");
  llvm::sort(set.intervals, [](const auto &lhs, const auto &rhs) {
    const int lower = compareUnsigned(lhs.lower, rhs.lower);
    return lower != 0 ? lower < 0 : compareUnsigned(lhs.upper, rhs.upper) < 0;
  });
  std::vector<SpatialConstraintUnsignedInterval> merged;
  for (auto &interval : set.intervals) {
    if (merged.empty() ||
        compareUnsigned(merged.back().upper, interval.lower) < 0) {
      merged.push_back(std::move(interval));
      continue;
    }
    if (compareUnsigned(merged.back().upper, interval.upper) < 0)
      merged.back().upper = std::move(interval.upper);
  }
  set.intervals = std::move(merged);
  return llvm::Error::success();
}

bool equal(const ProjectedSet &lhs, const ProjectedSet &rhs) {
  if (lhs.index() != rhs.index())
    return false;
  if (const auto *left = std::get_if<ExactSet>(&lhs))
    return left->values == std::get<ExactSet>(rhs).values;
  const auto &left = std::get<UnsignedSet>(lhs).intervals;
  const auto &right = std::get<UnsignedSet>(rhs).intervals;
  if (left.size() != right.size())
    return false;
  for (auto [a, b] : llvm::zip_equal(left, right))
    if (compareUnsigned(a.lower, b.lower) != 0 ||
        compareUnsigned(a.upper, b.upper) != 0)
      return false;
  return true;
}

bool subset(const ProjectedSet &lhs, const ProjectedSet &rhs) {
  if (lhs.index() != rhs.index())
    return false;
  if (const auto *left = std::get_if<ExactSet>(&lhs)) {
    const auto &right = std::get<ExactSet>(rhs);
    return std::includes(right.values.begin(), right.values.end(),
                         left->values.begin(), left->values.end());
  }
  const auto &left = std::get<UnsignedSet>(lhs).intervals;
  const auto &right = std::get<UnsignedSet>(rhs).intervals;
  std::size_t cursor = 0;
  for (const auto &interval : left) {
    while (cursor < right.size() &&
           compareUnsigned(right[cursor].upper, interval.lower) <= 0)
      ++cursor;
    if (cursor == right.size() ||
        compareUnsigned(right[cursor].lower, interval.lower) > 0 ||
        compareUnsigned(interval.upper, right[cursor].upper) > 0)
      return false;
  }
  return true;
}

llvm::Expected<std::string> exactValueKey(const DomainValue &value) {
  return std::visit(
      [](const auto &selected) -> llvm::Expected<std::string> {
        using Value = std::decay_t<decltype(selected)>;
        if constexpr (std::is_same_v<Value, ArtifactRootReference>)
          return mappingKey(selected);
        else if constexpr (std::is_same_v<Value,
                                          SpatialConstraintUnsignedInterval>)
          return invalid("tag interval used as an exact-set value");
        else
          return fabricKey(selected);
      },
      value);
}

llvm::Expected<ProjectedSet>
constraintDomain(Projection projection, llvm::ArrayRef<DomainValue> values) {
  if (projection == Projection::TransferAssignedTagValues) {
    UnsignedSet result;
    for (const auto &value : values) {
      const auto *interval =
          std::get_if<SpatialConstraintUnsignedInterval>(&value);
      if (!interval)
        return invalid("tag domain has a non-interval carrier");
      result.intervals.push_back(*interval);
    }
    if (llvm::Error error = normalize(result))
      return std::move(error);
    return ProjectedSet(std::move(result));
  }
  ExactSet result;
  for (const auto &value : values) {
    auto key = exactValueKey(value);
    if (!key)
      return key.takeError();
    result.values.push_back(std::move(*key));
  }
  normalize(result);
  return ProjectedSet(std::move(result));
}

struct LegProjection final {
  ExactSet traversals;
  ExactSet resourceStates;
  UnsignedSet tags;
};

class ProjectionIndex final {
public:
  static llvm::Expected<ProjectionIndex>
  build(const ::dataflow::CanonicalDataflowProgramView &dataflow,
        const ::loom::fabric::FabricSystemRootView &fabric,
        const SystemMappingView &mapping) {
    ProjectionIndex result(dataflow, fabric, mapping);
    if (llvm::Error error = result.buildExecution())
      return std::move(error);
    if (llvm::Error error = result.buildServices())
      return std::move(error);
    return result;
  }

  llvm::Expected<ProjectedSet> project(Projection projection,
                                       const Subject &subject) const {
    switch (projection) {
    case Projection::ThreadTargetAccCore: {
      const auto *root = std::get_if<::dataflow::RootThreadLaunchRef>(&subject);
      if (!root)
        return invalid("thread projection has a non-thread subject");
      auto key = dataflowKey(dataflow_.identity(), *root);
      if (!key)
        return key.takeError();
      return exact(threadTargets_, *key, "thread binding");
    }
    case Projection::GraphSelectedSpatialMapping:
    case Projection::GraphTargetSpatialCore: {
      const auto *graph =
          std::get_if<::dataflow::RootedGraphLaunchRef>(&subject);
      if (!graph)
        return invalid("graph projection has a non-graph subject");
      auto key = dataflowKey(dataflow_.identity(), *graph);
      if (!key)
        return key.takeError();
      return exact(projection == Projection::GraphSelectedSpatialMapping
                       ? graphMappings_
                       : graphSpatialCores_,
                   *key, "graph binding");
    }
    case Projection::ServiceTargetRegion: {
      const auto *service =
          std::get_if<OperationServiceObligationFamilyKey>(&subject);
      if (!service)
        return invalid("service projection has a non-service subject");
      auto bytes = encodeSystemServiceObligationKey(
          dataflow_.identity(), SystemServiceObligationKey{*service});
      if (!bytes)
        return bytes.takeError();
      return exact(serviceRegions_, byteKey(*bytes), "service realization");
    }
    case Projection::TransferTerminalAttachment: {
      const auto *terminal = std::get_if<SystemTransferTerminalKey>(&subject);
      if (!terminal)
        return invalid("terminal projection has a non-terminal subject");
      auto bytes =
          encodeSystemTransferTerminalKey(dataflow_.identity(), *terminal);
      if (!bytes)
        return bytes.takeError();
      return exact(terminalAttachments_, byteKey(*bytes), "transfer terminal");
    }
    case Projection::TransferSelectedTraversals:
    case Projection::TransferResourceStates:
    case Projection::TransferAssignedTagValues: {
      const auto *leg = std::get_if<CanonicalServiceLegKey>(&subject);
      if (!leg)
        return invalid("leg projection has a non-leg subject");
      auto bytes = encodeCanonicalServiceLegKey(dataflow_.identity(), *leg);
      if (!bytes)
        return bytes.takeError();
      auto found = legs_.find(byteKey(*bytes));
      if (found == legs_.end())
        return invalid("service leg has no final realization");
      if (projection == Projection::TransferSelectedTraversals)
        return ProjectedSet(found->second.traversals);
      if (projection == Projection::TransferResourceStates)
        return ProjectedSet(found->second.resourceStates);
      return ProjectedSet(found->second.tags);
    }
    }
    llvm_unreachable("unknown System constraint projection");
  }

private:
  ProjectionIndex(const ::dataflow::CanonicalDataflowProgramView &dataflow,
                  const ::loom::fabric::FabricSystemRootView &fabric,
                  const SystemMappingView &mapping)
      : dataflow_(dataflow), fabric_(fabric), mapping_(mapping) {}

  llvm::Expected<ProjectedSet>
  exact(const std::map<std::string, ExactSet> &index, llvm::StringRef key,
        llvm::StringRef description) const {
    auto found = index.find(key.str());
    if (found == index.end())
      return invalid(description + " has no final projection");
    return ProjectedSet(found->second);
  }

  llvm::Error buildExecution() {
    const auto &execution = mapping_.executionBindings();
    auto contexts =
        projectSystemExecutionContexts(dataflow_, execution);
    if (!contexts)
      return contexts.takeError();
    for (const auto &domain : contexts->instructionDomains) {
      auto key = dataflowKey(dataflow_.identity(), domain.root);
      if (!key)
        return key.takeError();
      auto &targets = threadTargets_[*key];
      targets.values.push_back(fabricKey(domain.context.accCore));
      normalize(targets);
    }
    for (const auto &domain : contexts->spatialDomains) {
      auto graphKey = dataflowKey(dataflow_.identity(), domain.graph);
      if (!graphKey)
        return graphKey.takeError();
      auto &mappings = graphMappings_[*graphKey];
      mappings.values.push_back(mappingKey(domain.spatialMapping));
      normalize(mappings);
      auto &cores = graphSpatialCores_[*graphKey];
      cores.values.push_back(fabricKey(
          ::loom::fabric::SpatialCoreOccurrenceRef{domain.context.accCore}));
      normalize(cores);
    }
    return llvm::Error::success();
  }

  llvm::Error buildServices() {
    std::map<std::string, const ::loom::fabric::FabricPhysicalTraversalView *>
        traversals;
    for (const auto &traversal : fabric_.artifact().physicalTraversals())
      traversals.emplace(fabricKey(traversal.reference), &traversal);

    for (const auto &service : mapping_.serviceRealizations()) {
      auto serviceBytes =
          encodeSystemServiceObligationKey(dataflow_.identity(), service.key);
      if (!serviceBytes)
        return serviceBytes.takeError();
      auto &regions = serviceRegions_[byteKey(*serviceBytes)];
      for (const auto &plan : service.plans) {
        for (const auto &target : plan.memoryTargets)
          regions.values.push_back(fabricKey(target.element.serviceRegion));
        for (const auto &route : plan.transferLegs) {
          auto legBytes =
              encodeCanonicalServiceLegKey(dataflow_.identity(), route.leg);
          if (!legBytes)
            return legBytes.takeError();
          auto &leg = legs_[byteKey(*legBytes)];
          auto sourceBytes = encodeSystemTransferTerminalKey(
              dataflow_.identity(), SystemTransferSourceTerminalKey{route.leg});
          if (!sourceBytes)
            return sourceBytes.takeError();
          terminalAttachments_[byteKey(*sourceBytes)].values.push_back(
              fabricKey(route.rootEndpoint));

          std::map<std::uint64_t,
                   std::vector<::loom::fabric::FabricTransportEndpointRef>>
              nodeEndpoints;
          nodeEndpoints.emplace(0, std::vector{route.rootEndpoint});
          for (const auto &node : route.nodes) {
            auto parent = nodeEndpoints.find(node.parentOrdinal);
            auto traversal = traversals.find(fabricKey(node.incomingTraversal));
            if (parent == nodeEndpoints.end() || traversal == traversals.end())
              return invalid(
                  "service route has an unresolved parent or traversal");
            if (!llvm::any_of(parent->second, [&](const auto &endpoint) {
                  return llvm::is_contained(traversal->second->sources,
                                            endpoint);
                }))
              return invalid("service route traversal is discontinuous");
            nodeEndpoints.emplace(node.ordinal,
                                  traversal->second->destinations);
            leg.traversals.values.push_back(fabricKey(node.incomingTraversal));
            for (const auto &state : traversal->second->resourceStates)
              leg.resourceStates.values.push_back(fabricKey(state));
          }
          for (const auto &sink : route.sinks) {
            auto endpoint = nodeEndpoints.find(sink.nodeOrdinal);
            if (endpoint == nodeEndpoints.end())
              return invalid("service route sink has an unresolved node");
            auto terminalBytes = encodeSystemTransferTerminalKey(
                dataflow_.identity(), sink.terminal);
            if (!terminalBytes)
              return terminalBytes.takeError();
            auto &attachments = terminalAttachments_[byteKey(*terminalBytes)];
            for (const auto &selected : endpoint->second)
              attachments.values.push_back(fabricKey(selected));
          }
        }
      }
      normalize(regions);
    }
    for (auto &[key, values] : terminalAttachments_) {
      (void)key;
      normalize(values);
    }
    for (auto &[key, leg] : legs_) {
      (void)key;
      normalize(leg.traversals);
      normalize(leg.resourceStates);
      if (llvm::Error error = normalize(leg.tags))
        return error;
    }
    return llvm::Error::success();
  }

  const ::dataflow::CanonicalDataflowProgramView &dataflow_;
  const ::loom::fabric::FabricSystemRootView &fabric_;
  const SystemMappingView &mapping_;
  std::map<std::string, ExactSet> threadTargets_;
  std::map<std::string, ExactSet> graphMappings_;
  std::map<std::string, ExactSet> graphSpatialCores_;
  std::map<std::string, ExactSet> serviceRegions_;
  std::map<std::string, ExactSet> terminalAttachments_;
  std::map<std::string, LegProjection> legs_;
};

bool insertDisjoint(std::set<std::string> &seen, const ExactSet &set) {
  for (const auto &value : set.values)
    if (!seen.insert(value).second)
      return false;
  return true;
}

struct UnsignedLess final {
  bool operator()(const llvm::APInt &lhs, const llvm::APInt &rhs) const {
    return compareUnsigned(lhs, rhs) < 0;
  }
};

bool insertDisjoint(std::map<llvm::APInt, llvm::APInt, UnsignedLess> &seen,
                    const UnsignedSet &set) {
  for (const auto &interval : set.intervals) {
    auto next = seen.lower_bound(interval.lower);
    if (next != seen.end() && compareUnsigned(next->first, interval.upper) < 0)
      return false;
    if (next != seen.begin()) {
      auto previous = std::prev(next);
      if (compareUnsigned(interval.lower, previous->second) < 0)
        return false;
    }
    seen.emplace(interval.lower, interval.upper);
  }
  return true;
}

llvm::Error reject(Projection projection, std::uint64_t clause,
                   const llvm::Twine &message) {
  return llvm::make_error<SystemMappingConstraintRejection>(projection, clause,
                                                            message.str());
}

} // namespace

llvm::Error loom::mapping::admitSystemMappingConstraints(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::fabric::FabricSystemRootView &fabric,
    const SystemMappingConstraintSetView &constraints,
    const SystemMappingView &systemMapping) {
  if (constraints.dataflowIdentity() != dataflow.identity() ||
      constraints.fabricIdentity() != fabric.artifact().identity() ||
      systemMapping.dataflowIdentity() != dataflow.identity() ||
      systemMapping.fabricIdentity() != fabric.artifact().identity() ||
      constraints.rootThreadLaunches() !=
          systemMapping.executionBindings().rootThreadLaunches())
    return invalid("D/F/root/K/M exact owner tuple does not match");

  auto index = ProjectionIndex::build(dataflow, fabric, systemMapping);
  if (!index)
    return index.takeError();
  for (auto [ordinal, clause] : llvm::enumerate(constraints.clauses())) {
    if (const auto *restriction =
            std::get_if<SystemDomainRestrictionView>(&clause)) {
      auto actual =
          index->project(restriction->projection, restriction->subject);
      auto allowed = constraintDomain(restriction->projection,
                                      restriction->admissibleDomain);
      if (!actual)
        return actual.takeError();
      if (!allowed)
        return allowed.takeError();
      if (!subset(*actual, *allowed))
        return reject(restriction->projection, ordinal,
                      "projected set is not a subset of the admissible domain");
      continue;
    }
    if (const auto *equalClause = std::get_if<SystemEqualView>(&clause)) {
      auto baseline = index->project(equalClause->projection,
                                     equalClause->subjects.front());
      if (!baseline)
        return baseline.takeError();
      for (const auto &subject : llvm::drop_begin(equalClause->subjects)) {
        auto actual = index->project(equalClause->projection, subject);
        if (!actual)
          return actual.takeError();
        if (!equal(*baseline, *actual))
          return reject(equalClause->projection, ordinal,
                        "subjects have different projected sets");
      }
      continue;
    }
    const auto &disjoint = std::get<SystemDisjointView>(clause);
    std::optional<ProjectedSet> first;
    std::set<std::string> exactSeen;
    std::map<llvm::APInt, llvm::APInt, UnsignedLess> intervalSeen;
    for (const auto &subject : disjoint.subjects) {
      auto actual = index->project(disjoint.projection, subject);
      if (!actual)
        return actual.takeError();
      if (first && first->index() != actual->index())
        return invalid("one Disjoint clause produced different carrier kinds");
      first = *actual;
      const bool inserted =
          std::holds_alternative<ExactSet>(*actual)
              ? insertDisjoint(exactSeen, std::get<ExactSet>(*actual))
              : insertDisjoint(intervalSeen, std::get<UnsignedSet>(*actual));
      if (!inserted)
        return reject(disjoint.projection, ordinal,
                      "subjects have intersecting projected sets");
    }
  }
  return llvm::Error::success();
}
