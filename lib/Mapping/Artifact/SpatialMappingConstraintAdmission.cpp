#include "Mapping/Artifact/MappingConstraintSet.h"

#include "Dataflow/IR/DataflowReferenceCodec.h"
#include "Fabric/IR/MemoryServiceContract.h"
#include "Fabric/IR/UsePatternValue.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "SpatialMappingTagAssignments.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CheckedArithmetic.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

using namespace loom;
using namespace loom::fabric;
using namespace loom::mapping;

char SpatialMappingConstraintRejection::ID;

void SpatialMappingConstraintRejection::log(llvm::raw_ostream &stream) const {
  stream << "spatial_mapping_rejected_by_constraint_set: ";
  if (const auto *projection =
          std::get_if<::mapping::SpatialConstraintProjection>(&owner_)) {
    stream << "projection "
           << ::mapping::stringifySpatialConstraintProjection(*projection);
  } else {
    stream << "no-good literal ";
    switch (std::get<SpatialNoGoodLiteralKind>(owner_)) {
    case SpatialNoGoodLiteralKind::NetUsesTraversal:
      stream << "net_uses_traversal";
      break;
    case SpatialNoGoodLiteralKind::TransferAttachmentEquals:
      stream << "transfer_attachment_equals";
      break;
    case SpatialNoGoodLiteralKind::NetTagEquals:
      stream << "net_tag_equals";
      break;
    case SpatialNoGoodLiteralKind::SpatialMappingIdentityEquals:
      stream << "spatial_mapping_identity_equals";
      break;
    }
  }
  stream << " clause " << clauseOrdinal_ << ": " << message_;
}

std::error_code SpatialMappingConstraintRejection::convertToErrorCode() const {
  return std::make_error_code(std::errc::operation_not_permitted);
}

namespace {

using Projection = ::mapping::SpatialConstraintProjection;
using Subject = SpatialConstraintSubject;
using DomainValue = SpatialConstraintDomainValue;

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      llvm::inconvertibleErrorCode(),
      "spatial_mapping_constraint_admission_invalid: " + message);
}

std::string byteKey(llvm::ArrayRef<std::uint8_t> bytes) {
  return std::string(reinterpret_cast<const char *>(bytes.data()),
                     bytes.size());
}

void appendU64(std::string &bytes, std::uint64_t value) {
  for (unsigned byte = 0; byte != 8; ++byte)
    bytes.push_back(static_cast<char>(value >> (8 * (7 - byte))));
}

void appendFramed(std::string &bytes, llvm::ArrayRef<std::uint8_t> component) {
  appendU64(bytes, component.size());
  bytes.append(reinterpret_cast<const char *>(component.data()),
               component.size());
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

llvm::Expected<std::string> exactValueKey(const DomainValue &value) {
  return std::visit(
      [&](const auto &selected) -> llvm::Expected<std::string> {
        using Value = std::decay_t<decltype(selected)>;
        if constexpr (std::is_same_v<Value, SpatialConstraintFuContext>) {
          std::string result;
          const auto fu = ::loom::fabric::canonicalFabricBytes(selected.fu);
          const auto context =
              ::loom::fabric::canonicalFabricBytes(selected.instructionContext);
          appendFramed(result, fu);
          appendFramed(result, context);
          return result;
        } else if constexpr (std::is_same_v<
                                 Value, SpatialConstraintUnsignedInterval> ||
                             std::is_same_v<Value,
                                            SpatialConstraintAddressRegion>) {
          return invalid("interval carrier used as an exact-set value");
        } else {
          return fabricKey(selected);
        }
      },
      value);
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

struct AddressServiceSet final {
  FabricMemoryServiceRef service;
  std::string key;
  std::vector<SpatialConstraintUnsignedInterval> intervals;
};

struct AddressSet final {
  std::vector<AddressServiceSet> services;
};

using ProjectedSet = std::variant<ExactSet, UnsignedSet, AddressSet>;

void normalizeExact(ExactSet &set) {
  llvm::sort(set.values);
  set.values.erase(std::unique(set.values.begin(), set.values.end()),
                   set.values.end());
}

llvm::Error
normalizeIntervals(std::vector<SpatialConstraintUnsignedInterval> &intervals) {
  for (const auto &interval : intervals)
    if (compareUnsigned(interval.lower, interval.upper) >= 0)
      return invalid("projected interval is empty or reversed");
  llvm::sort(intervals, [](const auto &lhs, const auto &rhs) {
    const int lower = compareUnsigned(lhs.lower, rhs.lower);
    return lower != 0 ? lower < 0 : compareUnsigned(lhs.upper, rhs.upper) < 0;
  });
  std::vector<SpatialConstraintUnsignedInterval> merged;
  merged.reserve(intervals.size());
  for (auto &interval : intervals) {
    if (merged.empty() ||
        compareUnsigned(merged.back().upper, interval.lower) < 0) {
      merged.push_back(std::move(interval));
      continue;
    }
    if (compareUnsigned(merged.back().upper, interval.upper) < 0)
      merged.back().upper = std::move(interval.upper);
  }
  intervals = std::move(merged);
  return llvm::Error::success();
}

llvm::Error normalizeAddress(AddressSet &set) {
  llvm::sort(set.services, [](const auto &lhs, const auto &rhs) {
    return lhs.key < rhs.key;
  });
  std::vector<AddressServiceSet> merged;
  merged.reserve(set.services.size());
  for (auto &service : set.services) {
    if (merged.empty() || merged.back().key != service.key) {
      if (llvm::Error error = normalizeIntervals(service.intervals))
        return error;
      if (!service.intervals.empty())
        merged.push_back(std::move(service));
      continue;
    }
    auto &target = merged.back().intervals;
    target.insert(target.end(),
                  std::make_move_iterator(service.intervals.begin()),
                  std::make_move_iterator(service.intervals.end()));
    if (llvm::Error error = normalizeIntervals(target))
      return error;
  }
  set.services = std::move(merged);
  return llvm::Error::success();
}

bool equalIntervals(llvm::ArrayRef<SpatialConstraintUnsignedInterval> lhs,
                    llvm::ArrayRef<SpatialConstraintUnsignedInterval> rhs) {
  if (lhs.size() != rhs.size())
    return false;
  for (auto [left, right] : llvm::zip_equal(lhs, rhs))
    if (compareUnsigned(left.lower, right.lower) != 0 ||
        compareUnsigned(left.upper, right.upper) != 0)
      return false;
  return true;
}

bool subsetIntervals(llvm::ArrayRef<SpatialConstraintUnsignedInterval> lhs,
                     llvm::ArrayRef<SpatialConstraintUnsignedInterval> rhs) {
  std::size_t right = 0;
  for (const auto &interval : lhs) {
    while (right < rhs.size() &&
           compareUnsigned(rhs[right].upper, interval.lower) <= 0)
      ++right;
    if (right == rhs.size() ||
        compareUnsigned(rhs[right].lower, interval.lower) > 0 ||
        compareUnsigned(interval.upper, rhs[right].upper) > 0)
      return false;
  }
  return true;
}

bool equalProjected(const ProjectedSet &lhs, const ProjectedSet &rhs) {
  if (lhs.index() != rhs.index())
    return false;
  return std::visit(
      [&](const auto &left) {
        using Set = std::decay_t<decltype(left)>;
        const auto &right = std::get<Set>(rhs);
        if constexpr (std::is_same_v<Set, ExactSet>) {
          return left.values == right.values;
        } else if constexpr (std::is_same_v<Set, UnsignedSet>) {
          return equalIntervals(left.intervals, right.intervals);
        } else {
          if (left.services.size() != right.services.size())
            return false;
          for (auto [leftService, rightService] :
               llvm::zip_equal(left.services, right.services))
            if (leftService.key != rightService.key ||
                !equalIntervals(leftService.intervals, rightService.intervals))
              return false;
          return true;
        }
      },
      lhs);
}

bool subsetProjected(const ProjectedSet &lhs, const ProjectedSet &rhs) {
  if (lhs.index() != rhs.index())
    return false;
  return std::visit(
      [&](const auto &left) {
        using Set = std::decay_t<decltype(left)>;
        const auto &right = std::get<Set>(rhs);
        if constexpr (std::is_same_v<Set, ExactSet>) {
          return std::includes(right.values.begin(), right.values.end(),
                               left.values.begin(), left.values.end());
        } else if constexpr (std::is_same_v<Set, UnsignedSet>) {
          return subsetIntervals(left.intervals, right.intervals);
        } else {
          std::size_t rightService = 0;
          for (const auto &leftService : left.services) {
            while (rightService < right.services.size() &&
                   right.services[rightService].key < leftService.key)
              ++rightService;
            if (rightService == right.services.size() ||
                right.services[rightService].key != leftService.key ||
                !subsetIntervals(leftService.intervals,
                                 right.services[rightService].intervals))
              return false;
          }
          return true;
        }
      },
      lhs);
}

struct UnsignedLess final {
  bool operator()(const llvm::APInt &lhs, const llvm::APInt &rhs) const {
    return compareUnsigned(lhs, rhs) < 0;
  }
};

using ExactDisjointIndex = std::set<std::string>;
using IntervalDisjointIndex = std::map<llvm::APInt, llvm::APInt, UnsignedLess>;
using AddressDisjointIndex = std::map<std::string, IntervalDisjointIndex>;
using DisjointIndex = std::variant<ExactDisjointIndex, IntervalDisjointIndex,
                                   AddressDisjointIndex>;

bool insertDisjointIntervals(
    IntervalDisjointIndex &index,
    llvm::ArrayRef<SpatialConstraintUnsignedInterval> intervals) {
  for (const auto &interval : intervals) {
    const auto next = index.lower_bound(interval.lower);
    if (next != index.end() && compareUnsigned(next->first, interval.upper) < 0)
      return false;
    if (next != index.begin()) {
      const auto previous = std::prev(next);
      if (compareUnsigned(interval.lower, previous->second) < 0)
        return false;
    }
    index.emplace(interval.lower, interval.upper);
  }
  return true;
}

DisjointIndex makeDisjointIndex(const ProjectedSet &projected) {
  return std::visit(
      [](const auto &selected) -> DisjointIndex {
        using Set = std::decay_t<decltype(selected)>;
        if constexpr (std::is_same_v<Set, ExactSet>)
          return ExactDisjointIndex{};
        if constexpr (std::is_same_v<Set, UnsignedSet>)
          return IntervalDisjointIndex{};
        return AddressDisjointIndex{};
      },
      projected);
}

llvm::Expected<bool> insertDisjoint(DisjointIndex &index,
                                    const ProjectedSet &projected) {
  if (index.index() != projected.index())
    return invalid("one Disjoint clause produced different carrier kinds");
  return std::visit(
      [&](const auto &selected) -> bool {
        using Set = std::decay_t<decltype(selected)>;
        if constexpr (std::is_same_v<Set, ExactSet>) {
          auto &seen = std::get<ExactDisjointIndex>(index);
          for (const auto &value : selected.values)
            if (!seen.insert(value).second)
              return false;
          return true;
        } else if constexpr (std::is_same_v<Set, UnsignedSet>) {
          return insertDisjointIntervals(std::get<IntervalDisjointIndex>(index),
                                         selected.intervals);
        } else {
          auto &seen = std::get<AddressDisjointIndex>(index);
          for (const auto &service : selected.services)
            if (!insertDisjointIntervals(seen[service.key], service.intervals))
              return false;
          return true;
        }
      },
      projected);
}

llvm::Expected<ProjectedSet>
domainFromConstraint(Projection projection,
                     llvm::ArrayRef<DomainValue> values) {
  if (projection == Projection::NetAssignedTagValues) {
    UnsignedSet result;
    result.intervals.reserve(values.size());
    for (const auto &value : values) {
      const auto *interval =
          std::get_if<SpatialConstraintUnsignedInterval>(&value);
      if (!interval)
        return invalid("tag domain has a non-interval carrier");
      result.intervals.push_back(*interval);
    }
    if (llvm::Error error = normalizeIntervals(result.intervals))
      return std::move(error);
    return ProjectedSet(std::move(result));
  }
  if (projection == Projection::MemoryAddressRegion) {
    AddressSet result;
    result.services.reserve(values.size());
    for (const auto &value : values) {
      const auto *region = std::get_if<SpatialConstraintAddressRegion>(&value);
      if (!region)
        return invalid("address domain has a non-address carrier");
      result.services.push_back(AddressServiceSet{
          region->service, fabricKey(region->service), region->intervals});
    }
    if (llvm::Error error = normalizeAddress(result))
      return std::move(error);
    return ProjectedSet(std::move(result));
  }
  ExactSet result;
  result.values.reserve(values.size());
  for (const auto &value : values) {
    auto key = exactValueKey(value);
    if (!key)
      return key.takeError();
    result.values.push_back(std::move(*key));
  }
  normalizeExact(result);
  return ProjectedSet(std::move(result));
}

::dataflow::LogicalMemoryRootRef
rootOf(const ::dataflow::LogicalMemoryRootOrViewRef &memory) {
  return std::visit(
      [](const auto &reference) -> ::dataflow::LogicalMemoryRootRef {
        using Ref = std::decay_t<decltype(reference)>;
        if constexpr (std::is_same_v<Ref, ::dataflow::LogicalMemoryRootRef>)
          return reference;
        else
          return reference.root;
      },
      memory);
}

struct ComputeProjection final {
  ExactSet placement;
  ExactSet parentPe;
  ExactSet instructionContext;
  ExactSet fuContext;
};

struct NetProjection final {
  UnsignedSet tags;
  ExactSet traversals;
  ExactSet resourceStates;
  FabricTransportEndpointRef source;
  std::map<std::string, FabricTransportEndpointRef> sinks;
  /// Traversals on the exact branch from the route root to each sink,
  /// rebuilt by walking `parentOrdinal`. A sink-qualified no-good literal is
  /// checked against its own branch, never against the whole tree.
  std::map<std::string, ExactSet> sinkBranchTraversals;
};

struct MemoryRootProjection final {
  ExactSet services;
  AddressSet addresses;
};

class ProjectionIndex final {
public:
  static llvm::Expected<ProjectionIndex>
  build(const ::dataflow::CanonicalDataflowProgramView &dataflow,
        const TechMappingView &techMapping, const FabricArtifactView &fabric,
        const SpatialMappingView &mapping) {
    ProjectionIndex result(dataflow, techMapping, fabric, mapping);
    if (llvm::Error error = result.buildCompute())
      return std::move(error);
    if (llvm::Error error = result.buildMemoryEngines())
      return std::move(error);
    if (llvm::Error error = result.buildRoutes())
      return std::move(error);
    if (llvm::Error error = result.buildTags())
      return std::move(error);
    if (llvm::Error error = result.buildMemoryRoots())
      return std::move(error);
    return result;
  }

  llvm::Expected<ProjectedSet> project(Projection projection,
                                       const Subject &subject) const {
    switch (projection) {
    case Projection::ComputePlacement:
    case Projection::ComputeParentPe:
    case Projection::ComputeInstructionContext:
    case Projection::ComputeFuContext: {
      const auto *reference = std::get_if<TechComputeRealizationRef>(&subject);
      if (!reference)
        return invalid("compute projection has a non-compute subject");
      auto found = compute_.find(reference->entity);
      if (found == compute_.end())
        return invalid("compute projection has no final binding");
      const ExactSet *selected = nullptr;
      switch (projection) {
      case Projection::ComputePlacement:
        selected = &found->second.placement;
        break;
      case Projection::ComputeParentPe:
        selected = &found->second.parentPe;
        break;
      case Projection::ComputeInstructionContext:
        selected = &found->second.instructionContext;
        break;
      case Projection::ComputeFuContext:
        selected = &found->second.fuContext;
        break;
      default:
        llvm_unreachable("non-compute projection in compute branch");
      }
      return ProjectedSet(*selected);
    }
    case Projection::MemoryPlacement: {
      const auto *reference = std::get_if<TechMemoryRealizationRef>(&subject);
      if (!reference)
        return invalid("memory placement has a non-memory subject");
      auto found = memoryPlacements_.find(reference->entity);
      if (found == memoryPlacements_.end())
        return invalid("memory realization has no final binding");
      return ProjectedSet(found->second);
    }
    case Projection::NetAssignedTagValues:
    case Projection::NetSelectedPhysicalTraversals:
    case Projection::NetTraversalResourceStates: {
      const auto *producer =
          std::get_if<::dataflow::CanonicalGraphProducerEndpointRef>(&subject);
      if (!producer)
        return invalid("net projection has a non-net subject");
      auto key = dataflowKey(dataflow_.identity(), *producer);
      if (!key)
        return key.takeError();
      auto found = nets_.find(*key);
      if (found == nets_.end())
        return invalid("residual logical net has no final RouteTree");
      if (projection == Projection::NetAssignedTagValues)
        return ProjectedSet(found->second.tags);
      if (projection == Projection::NetSelectedPhysicalTraversals)
        return ProjectedSet(found->second.traversals);
      return ProjectedSet(found->second.resourceStates);
    }
    case Projection::SpatialTransferAttachment: {
      const auto *terminal =
          std::get_if<SpatialConstraintTransferTerminal>(&subject);
      if (!terminal)
        return invalid("attachment projection has a non-terminal subject");
      auto key = dataflowKey(dataflow_.identity(), terminal->producer);
      if (!key)
        return key.takeError();
      auto found = nets_.find(*key);
      if (found == nets_.end())
        return invalid("transfer terminal has no final RouteTree");
      FabricTransportEndpointRef endpoint = found->second.source;
      if (terminal->consumer) {
        auto consumer = dataflowKey(dataflow_.identity(), *terminal->consumer);
        if (!consumer)
          return consumer.takeError();
        auto sink = found->second.sinks.find(*consumer);
        if (sink == found->second.sinks.end())
          return invalid("transfer sink has no final attachment");
        endpoint = sink->second;
      }
      ExactSet value{{fabricKey(endpoint)}};
      return ProjectedSet(std::move(value));
    }
    case Projection::MemoryOperationPort: {
      const auto *actor = std::get_if<::dataflow::ActorRef>(&subject);
      if (!actor)
        return invalid("memory port projection has a non-actor subject");
      auto key = dataflowKey(dataflow_.identity(), *actor);
      if (!key)
        return key.takeError();
      auto found = memoryOperationPorts_.find(*key);
      if (found == memoryOperationPorts_.end())
        return invalid("memory actor has no final operation port");
      return ProjectedSet(found->second);
    }
    case Projection::MemoryBoundServices:
    case Projection::MemoryAddressRegion: {
      const auto *root =
          std::get_if<::dataflow::LogicalMemoryRootRef>(&subject);
      if (!root)
        return invalid("memory-root projection has a non-root subject");
      auto found = memoryRoots_.find(root->entity.value());
      if (found == memoryRoots_.end()) {
        if (projection == Projection::MemoryBoundServices)
          return ProjectedSet(ExactSet{});
        return ProjectedSet(AddressSet{});
      }
      if (projection == Projection::MemoryBoundServices)
        return ProjectedSet(found->second.services);
      return ProjectedSet(found->second.addresses);
    }
    }
    llvm_unreachable("unknown Spatial constraint projection");
  }

  /// Whether one no-good literal holds of this sealed Mapping. Every answer is
  /// read from the independently rebuilt projection indexes; nothing consults
  /// solver state, search history, or the constraint set itself.
  llvm::Expected<bool> holds(const SpatialNoGoodLiteral &literal) const {
    if (const auto *uses =
            std::get_if<SpatialNetUsesTraversalLiteral>(&literal)) {
      auto key = dataflowKey(dataflow_.identity(), uses->producer);
      if (!key)
        return key.takeError();
      auto net = nets_.find(*key);
      if (net == nets_.end())
        return false;
      const std::string traversal = fabricKey(uses->traversal);
      if (!uses->consumer)
        return llvm::is_contained(net->second.traversals.values, traversal);
      auto sinkKey = dataflowKey(dataflow_.identity(), *uses->consumer);
      if (!sinkKey)
        return sinkKey.takeError();
      auto branch = net->second.sinkBranchTraversals.find(*sinkKey);
      if (branch == net->second.sinkBranchTraversals.end())
        return false;
      return llvm::is_contained(branch->second.values, traversal);
    }

    if (const auto *tag =
            std::get_if<SpatialNetTagEqualsLiteral>(&literal)) {
      std::optional<std::uint64_t> routeOrdinal;
      for (const auto indexed : llvm::enumerate(mapping_.routeTrees())) {
        if (indexed.value().logicalNet != tag->producer)
          continue;
        if (routeOrdinal)
          return invalid("sealed Mapping repeats a no-good tag producer");
        routeOrdinal = indexed.index();
      }
      if (!routeOrdinal)
        return false;
      const SpatialPhysicalTagSegmentView *selected = nullptr;
      for (const SpatialPhysicalTagSegmentView &segment :
           mapping_.physicalTagSegments()) {
        if (segment.routeTreeOrdinal != *routeOrdinal ||
            segment.segmentOrdinal != tag->segmentOrdinal)
          continue;
        if (selected)
          return invalid("sealed Mapping repeats a Physical Tag segment");
        selected = &segment;
      }
      if (!selected ||
          selected->resourceUseOrdinal >= mapping_.resourceUses().size())
        return false;
      const auto &assignments =
          mapping_.resourceUses()[selected->resourceUseOrdinal]
              .sharingAssignments;
      if (assignments.size() != 1)
        return invalid("sealed Mapping Physical Tag has the wrong shape");
      const auto *value =
          std::get_if<::fabric::PhysicalTagPatternValue>(&assignments.front());
      return value &&
             ::fabric::comparePhysicalTagValues(value->value, tag->value) == 0;
    }

    if (const auto *mapping =
            std::get_if<SpatialMappingIdentityEqualsLiteral>(&literal))
      return mapping_.identity() == mapping->mapping.artifact;

    const auto &attachment =
        std::get<SpatialTransferAttachmentEqualsLiteral>(literal);
    auto key = dataflowKey(dataflow_.identity(), attachment.terminal.producer);
    if (!key)
      return key.takeError();
    auto net = nets_.find(*key);
    if (net == nets_.end())
      return false;
    FabricTransportEndpointRef selected = net->second.source;
    if (attachment.terminal.consumer) {
      auto sinkKey =
          dataflowKey(dataflow_.identity(), *attachment.terminal.consumer);
      if (!sinkKey)
        return sinkKey.takeError();
      auto sink = net->second.sinks.find(*sinkKey);
      if (sink == net->second.sinks.end())
        return false;
      selected = sink->second;
    }
    return fabricKey(selected) == fabricKey(attachment.endpoint);
  }

private:
  ProjectionIndex(const ::dataflow::CanonicalDataflowProgramView &dataflow,
                  const TechMappingView &techMapping,
                  const FabricArtifactView &fabric,
                  const SpatialMappingView &mapping)
      : dataflow_(dataflow), techMapping_(techMapping), fabric_(fabric),
        mapping_(mapping) {}

  llvm::Error buildCompute() {
    for (const auto &binding : mapping_.computeBindings()) {
      auto parent = fabric_.parentPeOf(binding.occurrence);
      if (!parent)
        return invalid("final compute occurrence has no owning PE");
      ComputeProjection projection;
      projection.placement.values.push_back(fabricKey(binding.occurrence));
      projection.parentPe.values.push_back(fabricKey(*parent));
      projection.instructionContext.values.push_back(
          fabricKey(binding.context));
      DomainValue tuple =
          SpatialConstraintFuContext{binding.occurrence, binding.context};
      auto tupleKey = exactValueKey(tuple);
      if (!tupleKey)
        return tupleKey.takeError();
      projection.fuContext.values.push_back(std::move(*tupleKey));
      if (!compute_.try_emplace(binding.realization, std::move(projection))
               .second)
        return invalid("final Mapping repeats a compute binding");
    }
    return llvm::Error::success();
  }

  llvm::Error buildMemoryEngines() {
    for (const auto &binding : mapping_.memoryEngineBindings()) {
      ExactSet placement{{fabricKey(binding.occurrence)}};
      if (!memoryPlacements_
               .try_emplace(binding.realization, std::move(placement))
               .second)
        return invalid("final Mapping repeats a memory engine binding");
      for (const auto &operation : binding.operations) {
        const auto *addressed =
            std::get_if<SpatialAddressedMemoryOperationView>(&operation);
        if (!addressed)
          continue;
        FabricMemoryOperationPortRef port = std::visit(
            [](const auto &selected) -> FabricMemoryOperationPortRef {
              using Placement = std::decay_t<decltype(selected)>;
              if constexpr (std::is_same_v<Placement,
                                           FabricMemoryOperationPortRef>)
                return selected;
              else
                return selected.port;
            },
            addressed->placement);
        auto actor = dataflowKey(dataflow_.identity(), addressed->actor);
        if (!actor)
          return actor.takeError();
        if (!memoryOperationPorts_
                 .try_emplace(std::move(*actor), ExactSet{{fabricKey(port)}})
                 .second)
          return invalid("final Mapping repeats a memory actor placement");
      }
    }
    return llvm::Error::success();
  }

  llvm::Error buildRoutes() {
    std::map<std::string, const FabricPhysicalTraversalView *> traversals;
    for (const auto &traversal : fabric_.physicalTraversals())
      if (!traversals.try_emplace(fabricKey(traversal.reference), &traversal)
               .second)
        return invalid("Fabric repeats a physical traversal projection");

    for (const auto &route : mapping_.routeTrees()) {
      auto key = dataflowKey(dataflow_.identity(), route.logicalNet);
      if (!key)
        return key.takeError();
      NetProjection projection;
      projection.source = route.rootEndpoint;
      const auto appendTraversal =
          [&](const std::optional<FabricPhysicalTraversalRef> &selected)
          -> llvm::Error {
        if (!selected)
          return llvm::Error::success();
        const std::string traversalKey = fabricKey(*selected);
        projection.traversals.values.push_back(traversalKey);
        auto found = traversals.find(traversalKey);
        if (found == traversals.end())
          return invalid("RouteTree selects an absent physical traversal");
        for (const auto &state : found->second->resourceStates)
          projection.resourceStates.values.push_back(fabricKey(state));
        return llvm::Error::success();
      };
      if (llvm::Error error = appendTraversal(route.localTraversal))
        return error;
      for (const auto &node : route.nodes) {
        if (llvm::Error error = appendTraversal(node.incomingTraversal))
          return error;
      }
      for (const auto &sink : route.sinks)
        if (llvm::Error error = appendTraversal(sink.localTraversal))
          return error;
      normalizeExact(projection.traversals);
      normalizeExact(projection.resourceStates);
      for (const auto &sink : route.sinks) {
        if (sink.nodeOrdinal >= route.nodes.size())
          return invalid("RouteTree sink has an out-of-range node");
        auto sinkKey = dataflowKey(dataflow_.identity(), sink.sink);
        if (!sinkKey)
          return sinkKey.takeError();
        auto branchTraversals = spatialRouteBranchTraversals(route, sink);
        if (!branchTraversals)
          return branchTraversals.takeError();
        ExactSet branch;
        for (const auto &traversal : *branchTraversals)
          branch.values.push_back(fabricKey(traversal));
        normalizeExact(branch);
        if (!projection.sinkBranchTraversals
                 .try_emplace(*sinkKey, std::move(branch))
                 .second)
          return invalid("RouteTree repeats a logical sink branch");
        if (!projection.sinks
                 .try_emplace(std::move(*sinkKey),
                              route.nodes[sink.nodeOrdinal].endpoint)
                 .second)
          return invalid("RouteTree repeats a logical sink");
      }
      if (!nets_.try_emplace(std::move(*key), std::move(projection)).second)
        return invalid("final Mapping repeats a residual logical net");
    }
    return llvm::Error::success();
  }

  llvm::Error buildTags() {
    auto required = ::loom::mapping::detail::deriveRequiredPhysicalTagUses(
        dataflow_, techMapping_, fabric_, mapping_.routeTrees());
    if (!required)
      return required.takeError();
    std::set<std::string> observed;
    for (const auto &use : mapping_.resourceUses()) {
      auto key = ::loom::mapping::detail::physicalTagUseKey(
          use.owner, use.activation.trigger.event, use.useSite,
          dataflow_.identity());
      if (!key)
        return key.takeError();
      auto found = required->find(*key);
      if (found == required->end())
        continue;
      if (!observed.insert(*key).second || !use.parameters.empty() ||
          use.sharingAssignments.size() != 1)
        return invalid("Physical Tag projection has a malformed ResourceUse");
      const auto *tag = std::get_if<::fabric::PhysicalTagPatternValue>(
          &use.sharingAssignments.front());
      if (!tag)
        return invalid("Physical Tag projection has a non-tag assignment");
      const auto *producer =
          std::get_if<::dataflow::CanonicalGraphProducerEndpointRef>(
              &found->second.trigger);
      if (!producer)
        return invalid("Physical Tag projection has a non-net trigger");
      auto netKey = dataflowKey(dataflow_.identity(), *producer);
      if (!netKey)
        return netKey.takeError();
      auto net = nets_.find(*netKey);
      if (net == nets_.end())
        return invalid("Physical Tag projection has no RouteTree");
      llvm::APInt lower = tag->value.zext(tag->value.getBitWidth() + 1);
      llvm::APInt upper = lower + 1;
      net->second.tags.intervals.push_back(SpatialConstraintUnsignedInterval{
          std::move(lower), std::move(upper)});
    }
    if (observed.size() != required->size())
      return invalid("final Mapping omits a required Physical Tag assignment");
    for (auto &[key, net] : nets_) {
      (void)key;
      if (llvm::Error error = normalizeIntervals(net.tags.intervals))
        return error;
    }
    return llvm::Error::success();
  }

  llvm::Error buildMemoryRoots() {
    for (const auto &binding : mapping_.memoryBindings()) {
      const auto *local =
          std::get_if<SpatialMemoryLocalRegionView>(&binding.target);
      if (!local)
        continue;
      const auto root = rootOf(binding.logicalMemory);
      auto &projection = memoryRoots_[root.entity.value()];
      projection.services.values.push_back(
          fabricKey(local->serviceRegion.service));

      if (local->serviceRegion.service.kind() != FabricMemoryServiceKind::Local)
        return invalid("Spatial local binding names a non-local service");
      const auto memory = std::get<FabricMemoryOccurrenceRef>(
          local->serviceRegion.service.payload);
      const auto *service = fabric_.localMemoryService(memory);
      if (!service || local->serviceRegion.ordinal >= service->regions().size())
        return invalid("Spatial local binding names an absent service region");
      const auto &region = service->regions()[local->serviceRegion.ordinal];

      std::uint64_t size = 0;
      if (const auto *range =
              std::get_if<SpatialMemoryByteRangeView>(&binding.interval)) {
        size = range->sizeBytes;
      } else {
        auto extent = dataflow_.staticMemoryByteExtent(binding.logicalMemory);
        if (!extent)
          return extent.takeError();
        if (!*extent)
          return invalid("local Whole binding has no finite extent");
        size = **extent;
      }
      if (size == 0)
        continue;
      const auto base = llvm::checkedAddUnsigned(region.addressBaseBytes,
                                                 local->physicalOffsetBytes);
      if (!base)
        return invalid("physical memory address base overflows u64");
      const auto end = llvm::checkedAddUnsigned(*base, size);
      if (!end)
        return invalid("physical memory address interval overflows u64");
      projection.addresses.services.push_back(AddressServiceSet{
          local->serviceRegion.service,
          fabricKey(local->serviceRegion.service),
          {SpatialConstraintUnsignedInterval{llvm::APInt(64, *base),
                                             llvm::APInt(64, *end)}}});
    }
    for (auto &[root, projection] : memoryRoots_) {
      (void)root;
      normalizeExact(projection.services);
      if (llvm::Error error = normalizeAddress(projection.addresses))
        return error;
    }
    return llvm::Error::success();
  }

  const ::dataflow::CanonicalDataflowProgramView &dataflow_;
  const TechMappingView &techMapping_;
  const FabricArtifactView &fabric_;
  const SpatialMappingView &mapping_;
  std::map<std::uint64_t, ComputeProjection> compute_;
  std::map<std::uint64_t, ExactSet> memoryPlacements_;
  std::map<std::string, NetProjection> nets_;
  std::map<std::string, ExactSet> memoryOperationPorts_;
  std::map<std::uint64_t, MemoryRootProjection> memoryRoots_;
};

llvm::Error reject(SpatialMappingConstraintRejection::Owner owner,
                   std::uint64_t clause,
                   const llvm::Twine &message) {
  return std::visit(
      [&](auto typed) -> llvm::Error {
        return llvm::make_error<SpatialMappingConstraintRejection>(
            typed, clause, message.str());
      },
      owner);
}

SpatialNoGoodLiteralKind literalKind(const SpatialNoGoodLiteral &literal) {
  if (std::holds_alternative<SpatialNetUsesTraversalLiteral>(literal))
    return SpatialNoGoodLiteralKind::NetUsesTraversal;
  if (std::holds_alternative<SpatialTransferAttachmentEqualsLiteral>(literal))
    return SpatialNoGoodLiteralKind::TransferAttachmentEquals;
  if (std::holds_alternative<SpatialNetTagEqualsLiteral>(literal))
    return SpatialNoGoodLiteralKind::NetTagEquals;
  return SpatialNoGoodLiteralKind::SpatialMappingIdentityEquals;
}

} // namespace

llvm::Expected<bool> loom::mapping::spatialMappingHoldsNoGoodLiteral(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const TechMappingView &techMapping,
    const ::loom::fabric::FabricArtifactView &fabric,
    const SpatialMappingView &spatialMapping,
    const SpatialNoGoodLiteral &literal) {
  if (techMapping.dataflowIdentity() != dataflow.identity() ||
      techMapping.fabricIdentity() != fabric.identity() ||
      spatialMapping.dataflowIdentity() != dataflow.identity() ||
      spatialMapping.techMappingIdentity() != techMapping.identity() ||
      spatialMapping.fabricIdentity() != fabric.identity())
    return invalid("D/T/F/S exact owner tuple does not match");
  auto index =
      ProjectionIndex::build(dataflow, techMapping, fabric, spatialMapping);
  if (!index)
    return index.takeError();
  return index->holds(literal);
}

llvm::Error loom::mapping::admitSpatialMappingConstraints(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const TechMappingView &techMapping,
    const ::loom::fabric::FabricArtifactView &fabric,
    const SpatialMappingConstraintSetView &constraints,
    const SpatialMappingView &spatialMapping) {
  if (techMapping.dataflowIdentity() != dataflow.identity() ||
      techMapping.fabricIdentity() != fabric.identity() ||
      constraints.dataflowIdentity() != dataflow.identity() ||
      constraints.techMappingIdentity() != techMapping.identity() ||
      constraints.fabricIdentity() != fabric.identity() ||
      spatialMapping.dataflowIdentity() != dataflow.identity() ||
      spatialMapping.techMappingIdentity() != techMapping.identity() ||
      spatialMapping.fabricIdentity() != fabric.identity())
    return invalid("D/T/F/K/S exact owner tuple does not match");

  auto index =
      ProjectionIndex::build(dataflow, techMapping, fabric, spatialMapping);
  if (!index)
    return index.takeError();

  for (auto [ordinal, clause] : llvm::enumerate(constraints.clauses())) {
    if (const auto *restriction =
            std::get_if<SpatialDomainRestrictionView>(&clause)) {
      auto actual =
          index->project(restriction->projection, restriction->subject);
      if (!actual)
        return actual.takeError();
      auto allowed = domainFromConstraint(restriction->projection,
                                          restriction->admissibleDomain);
      if (!allowed)
        return allowed.takeError();
      if (!subsetProjected(*actual, *allowed))
        return reject(restriction->projection, ordinal,
                      "projected set is not a subset of the admissible domain");
      continue;
    }
    if (const auto *equal = std::get_if<SpatialEqualView>(&clause)) {
      auto baseline =
          index->project(equal->projection, equal->subjects.front());
      if (!baseline)
        return baseline.takeError();
      for (const auto &subject : llvm::drop_begin(equal->subjects)) {
        auto actual = index->project(equal->projection, subject);
        if (!actual)
          return actual.takeError();
        if (!equalProjected(*baseline, *actual))
          return reject(equal->projection, ordinal,
                        "subjects have different projected sets");
      }
      continue;
    }
    if (const auto *disjoint = std::get_if<SpatialDisjointView>(&clause)) {
      std::optional<DisjointIndex> seen;
      for (const auto &subject : disjoint->subjects) {
        auto actual = index->project(disjoint->projection, subject);
        if (!actual)
          return actual.takeError();
        if (!seen)
          seen = makeDisjointIndex(*actual);
        auto inserted = insertDisjoint(*seen, *actual);
        if (!inserted)
          return inserted.takeError();
        if (!*inserted)
          return reject(disjoint->projection, ordinal,
                        "subjects have intersecting projected sets");
      }
      continue;
    }

    // A no-good is violated only when every listed choice still holds. One
    // literal that changed is enough to satisfy the clause, which is exactly
    // why the witness never marks the Fabric intrinsically infeasible.
    const auto &noGood =
        std::get<SpatialRuntimeCounterexampleNoGoodView>(clause);
    if (noGood.literals.empty())
      return invalid("runtime-counterexample no-good clause is empty");
    bool allHold = true;
    for (const SpatialNoGoodLiteral &literal : noGood.literals) {
      auto held = index->holds(literal);
      if (!held)
        return held.takeError();
      if (!*held) {
        allHold = false;
        break;
      }
    }
    if (allHold)
      return reject(
          literalKind(noGood.literals.front()), ordinal,
          "every literal of a runtime-counterexample no-good still holds, so "
          "this Mapping repeats the recorded counterexample");
  }
  return llvm::Error::success();
}
