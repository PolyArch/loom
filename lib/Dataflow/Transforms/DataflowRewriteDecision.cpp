#include "Dataflow/Transforms/DataflowRewrite.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

namespace dataflow {
namespace {

constexpr llvm::StringLiteral decisionSchema =
    "loom.dataflow_rewrite.decision.2.0";

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "dataflow_rewrite_decision_invalid: " +
                                     message);
}

void appendU32(std::vector<std::uint8_t> &bytes, std::uint32_t value) {
  for (int shift = 24; shift >= 0; shift -= 8)
    bytes.push_back(static_cast<std::uint8_t>(value >> shift));
}

void appendU64(std::vector<std::uint8_t> &bytes, std::uint64_t value) {
  for (int shift = 56; shift >= 0; shift -= 8)
    bytes.push_back(static_cast<std::uint8_t>(value >> shift));
}

template <typename Id> void appendId(std::vector<std::uint8_t> &bytes, Id id) {
  appendU64(bytes, id.value());
}

template <typename Id>
bool isCanonicalSet(llvm::ArrayRef<Id> ids, std::size_t minimumSize) {
  if (ids.size() < minimumSize)
    return false;
  return llvm::is_sorted(
             ids, [](Id lhs, Id rhs) { return lhs.value() < rhs.value(); }) &&
         std::adjacent_find(ids.begin(), ids.end(), [](Id lhs, Id rhs) {
           return lhs == rhs;
         }) == ids.end();
}

template <typename Id>
llvm::Error appendIdSet(std::vector<std::uint8_t> &bytes,
                        llvm::ArrayRef<Id> ids, std::size_t minimumSize,
                        llvm::StringRef name) {
  if (!isCanonicalSet(ids, minimumSize))
    return invalid(name + " must be a sorted unique reference set");
  appendU64(bytes, ids.size());
  for (Id id : ids)
    appendId(bytes, id);
  return llvm::Error::success();
}

class Decoder final {
public:
  explicit Decoder(llvm::ArrayRef<std::uint8_t> bytes) : bytes_(bytes) {}

  llvm::Expected<std::uint32_t> u32(llvm::StringRef name) {
    if (bytes_.size() - offset_ < 4)
      return invalid(name + " is truncated");
    std::uint32_t value = 0;
    for (std::uint8_t byte : bytes_.slice(offset_, 4))
      value = (value << 8) | byte;
    offset_ += 4;
    return value;
  }

  llvm::Expected<std::uint64_t> u64(llvm::StringRef name) {
    if (bytes_.size() - offset_ < 8)
      return invalid(name + " is truncated");
    std::uint64_t value = 0;
    for (std::uint8_t byte : bytes_.slice(offset_, 8))
      value = (value << 8) | byte;
    offset_ += 8;
    return value;
  }

  template <typename Id> llvm::Expected<Id> id(llvm::StringRef name) {
    auto value = u64(name);
    if (!value)
      return value.takeError();
    return Id(*value);
  }

  template <typename Id>
  llvm::Expected<std::vector<Id>> idSet(std::size_t minimumSize,
                                        llvm::StringRef name) {
    const std::string countName = (name + " count").str();
    auto count = u64(countName);
    if (!count)
      return count.takeError();
    if (*count > (bytes_.size() - offset_) / 8)
      return invalid(name + " count exceeds the payload");
    std::vector<Id> ids;
    ids.reserve(static_cast<std::size_t>(*count));
    for (std::uint64_t ordinal = 0; ordinal != *count; ++ordinal) {
      auto value = id<Id>(name);
      if (!value)
        return value.takeError();
      ids.push_back(*value);
    }
    if (!isCanonicalSet<Id>(ids, minimumSize))
      return invalid(name + " must be a sorted unique reference set");
    return ids;
  }

  bool atEnd() const { return offset_ == bytes_.size(); }

private:
  llvm::ArrayRef<std::uint8_t> bytes_;
  std::size_t offset_ = 0;
};

template <typename Id> int compareId(Id lhs, Id rhs) {
  if (lhs.value() < rhs.value())
    return -1;
  if (rhs.value() < lhs.value())
    return 1;
  return 0;
}

template <typename Id>
int compareIds(llvm::ArrayRef<Id> lhs, llvm::ArrayRef<Id> rhs) {
  const std::size_t common = std::min(lhs.size(), rhs.size());
  for (std::size_t index = 0; index != common; ++index)
    if (int order = compareId(lhs[index], rhs[index]))
      return order;
  if (lhs.size() < rhs.size())
    return -1;
  if (rhs.size() < lhs.size())
    return 1;
  return 0;
}

template <typename Enum> bool validBinaryEnum(Enum value) {
  return static_cast<std::uint32_t>(value) <= 1;
}

} // namespace

DataflowRewriteKind
dataflowRewriteKind(const DataflowRewriteDecision &decision) {
  return std::visit(
      [](const auto &typed) -> DataflowRewriteKind {
        using T = std::decay_t<decltype(typed)>;
        if constexpr (std::is_same_v<T, SyncRendezvousRewrite>)
          return DataflowRewriteKind::SyncRendezvousRefactor;
        if constexpr (std::is_same_v<T, PackUnpackRoundTripRewrite>)
          return DataflowRewriteKind::PackUnpackRoundTripEliminate;
        if constexpr (std::is_same_v<T, ParallelizeSerializeRoundTripRewrite>)
          return DataflowRewriteKind::ParallelizeSerializeRoundTripEliminate;
        if constexpr (std::is_same_v<T, ElementwiseCardinalityCommuteRewrite>)
          return DataflowRewriteKind::ElementwiseCardinalityCommute;
        if constexpr (std::is_same_v<T, PureComputeFanoutReplicateRewrite> ||
                      std::is_same_v<T, PureComputeFanoutFactorRewrite>)
          return DataflowRewriteKind::PureComputeFanoutRefactor;
        if constexpr (std::is_same_v<T,
                                     ActivationPreservingConstantFoldRewrite>)
          return DataflowRewriteKind::ActivationPreservingConstantFold;
        if constexpr (std::is_same_v<T, GraphDefinitionSplitRewrite> ||
                      std::is_same_v<T, GraphDefinitionMergeRewrite>)
          return DataflowRewriteKind::GraphDefinitionRefactor;
        return DataflowRewriteKind::ElementwiseVectorDecompose;
      },
      decision);
}

llvm::ArrayRef<std::uint8_t> dataflowRewriteDecisionSchemaBytes() {
  return {reinterpret_cast<const std::uint8_t *>(decisionSchema.data()),
          decisionSchema.size()};
}

llvm::Expected<std::vector<std::uint8_t>>
encodeDataflowRewriteDecision(const DataflowRewriteDecision &decision) {
  std::vector<std::uint8_t> bytes;
  appendU32(bytes, static_cast<std::uint32_t>(dataflowRewriteKind(decision)));

  llvm::Error error = std::visit(
      [&](const auto &typed) -> llvm::Error {
        using T = std::decay_t<decltype(typed)>;
        if constexpr (std::is_same_v<T, SyncRendezvousRewrite>) {
          if (!validBinaryEnum(typed.direction))
            return invalid("sync direction is unknown");
          appendId(bytes, typed.root);
          appendU32(bytes, static_cast<std::uint32_t>(typed.direction));
        } else if constexpr (std::is_same_v<T, PackUnpackRoundTripRewrite>) {
          appendId(bytes, typed.outerAdapter);
        } else if constexpr (std::is_same_v<
                                 T, ParallelizeSerializeRoundTripRewrite>) {
          appendId(bytes, typed.outerSerialize);
        } else if constexpr (std::is_same_v<
                                 T, ElementwiseCardinalityCommuteRewrite>) {
          if (!validBinaryEnum(typed.direction))
            return invalid("cardinality commute direction is unknown");
          appendId(bytes, typed.compute);
          if (llvm::Error setError =
                  appendIdSet(bytes, llvm::ArrayRef<ActorId>(typed.adapters), 1,
                              "cardinality adapter set"))
            return setError;
          appendU32(bytes, static_cast<std::uint32_t>(typed.direction));
        } else if constexpr (std::is_same_v<
                                 T, PureComputeFanoutReplicateRewrite>) {
          appendU32(bytes, 0);
          appendId(bytes, typed.compute);
        } else if constexpr (std::is_same_v<T,
                                            PureComputeFanoutFactorRewrite>) {
          appendU32(bytes, 1);
          return appendIdSet(bytes, llvm::ArrayRef<ActorId>(typed.replicas), 2,
                             "fanout replica set");
        } else if constexpr (std::is_same_v<
                                 T, ActivationPreservingConstantFoldRewrite>) {
          appendId(bytes, typed.compute);
        } else if constexpr (std::is_same_v<T, GraphDefinitionSplitRewrite>) {
          appendU32(bytes, 0);
          appendId(bytes, typed.graph);
          return appendIdSet(
              bytes, llvm::ArrayRef<StaticGraphLaunchId>(typed.launches), 1,
              "graph split launch set");
        } else if constexpr (std::is_same_v<T, GraphDefinitionMergeRewrite>) {
          if (typed.lowerGraph.value() >= typed.higherGraph.value())
            return invalid("graph merge pair is not canonically ordered");
          appendU32(bytes, 1);
          appendId(bytes, typed.lowerGraph);
          appendId(bytes, typed.higherGraph);
        } else if constexpr (std::is_same_v<T, ElementwiseVectorChunkRewrite>) {
          if (typed.leadingBlocksPerChunk == 0)
            return invalid("vector chunk extent is zero");
          appendId(bytes, typed.compute);
          appendU32(bytes, 0);
          appendU64(bytes, typed.leadingBlocksPerChunk);
        } else {
          appendId(bytes, typed.compute);
          appendU32(bytes, 1);
        }
        return llvm::Error::success();
      },
      decision);
  if (error)
    return std::move(error);
  return bytes;
}

llvm::Expected<DataflowRewriteDecision>
adoptDataflowRewriteDecision(llvm::ArrayRef<std::uint8_t> canonicalBytes) {
  Decoder decoder(canonicalBytes);
  auto rawKind = decoder.u32("kind");
  if (!rawKind)
    return rawKind.takeError();
  if (*rawKind > static_cast<std::uint32_t>(
                     DataflowRewriteKind::ElementwiseVectorDecompose))
    return invalid("kind is unknown");

  std::optional<DataflowRewriteDecision> decision;
  switch (static_cast<DataflowRewriteKind>(*rawKind)) {
  case DataflowRewriteKind::SyncRendezvousRefactor: {
    auto root = decoder.id<ActorId>("sync root");
    auto direction = decoder.u32("sync direction");
    if (!root)
      return root.takeError();
    if (!direction)
      return direction.takeError();
    if (*direction > 1)
      return invalid("sync direction is unknown");
    decision = SyncRendezvousRewrite{
        *root, static_cast<SyncRendezvousDirection>(*direction)};
    break;
  }
  case DataflowRewriteKind::PackUnpackRoundTripEliminate: {
    auto outer = decoder.id<ActorId>("outer adapter");
    if (!outer)
      return outer.takeError();
    decision = PackUnpackRoundTripRewrite{*outer};
    break;
  }
  case DataflowRewriteKind::ParallelizeSerializeRoundTripEliminate: {
    auto outer = decoder.id<ActorId>("outer serialize");
    if (!outer)
      return outer.takeError();
    decision = ParallelizeSerializeRoundTripRewrite{*outer};
    break;
  }
  case DataflowRewriteKind::ElementwiseCardinalityCommute: {
    auto compute = decoder.id<ActorId>("compute");
    auto adapters = decoder.idSet<ActorId>(1, "cardinality adapter set");
    auto direction = decoder.u32("cardinality commute direction");
    if (!compute)
      return compute.takeError();
    if (!adapters)
      return adapters.takeError();
    if (!direction)
      return direction.takeError();
    if (*direction > 1)
      return invalid("cardinality commute direction is unknown");
    decision = ElementwiseCardinalityCommuteRewrite{
        *compute, std::move(*adapters),
        static_cast<CardinalityCommuteDirection>(*direction)};
    break;
  }
  case DataflowRewriteKind::PureComputeFanoutRefactor: {
    auto variant = decoder.u32("fanout variant");
    if (!variant)
      return variant.takeError();
    if (*variant == 0) {
      auto compute = decoder.id<ActorId>("fanout compute");
      if (!compute)
        return compute.takeError();
      decision = PureComputeFanoutReplicateRewrite{*compute};
    } else if (*variant == 1) {
      auto replicas = decoder.idSet<ActorId>(2, "fanout replica set");
      if (!replicas)
        return replicas.takeError();
      decision = PureComputeFanoutFactorRewrite{std::move(*replicas)};
    } else {
      return invalid("fanout variant is unknown");
    }
    break;
  }
  case DataflowRewriteKind::ActivationPreservingConstantFold: {
    auto compute = decoder.id<ActorId>("constant-fold compute");
    if (!compute)
      return compute.takeError();
    decision = ActivationPreservingConstantFoldRewrite{*compute};
    break;
  }
  case DataflowRewriteKind::GraphDefinitionRefactor: {
    auto variant = decoder.u32("graph refactor variant");
    if (!variant)
      return variant.takeError();
    if (*variant == 0) {
      auto graph = decoder.id<GraphId>("split graph");
      auto launches =
          decoder.idSet<StaticGraphLaunchId>(1, "graph split launch set");
      if (!graph)
        return graph.takeError();
      if (!launches)
        return launches.takeError();
      decision = GraphDefinitionSplitRewrite{*graph, std::move(*launches)};
    } else if (*variant == 1) {
      auto lower = decoder.id<GraphId>("lower graph");
      auto higher = decoder.id<GraphId>("higher graph");
      if (!lower)
        return lower.takeError();
      if (!higher)
        return higher.takeError();
      if (lower->value() >= higher->value())
        return invalid("graph merge pair is not canonically ordered");
      decision = GraphDefinitionMergeRewrite{*lower, *higher};
    } else {
      return invalid("graph refactor variant is unknown");
    }
    break;
  }
  case DataflowRewriteKind::ElementwiseVectorDecompose: {
    auto compute = decoder.id<ActorId>("vector compute");
    auto mode = decoder.u32("vector decomposition mode");
    if (!compute)
      return compute.takeError();
    if (!mode)
      return mode.takeError();
    if (*mode == 0) {
      auto extent = decoder.u64("vector chunk extent");
      if (!extent)
        return extent.takeError();
      if (*extent == 0)
        return invalid("vector chunk extent is zero");
      decision = ElementwiseVectorChunkRewrite{*compute, *extent};
    } else if (*mode == 1) {
      decision = ElementwiseVectorScalarizeRewrite{*compute};
    } else {
      return invalid("vector decomposition mode is unknown");
    }
    break;
  }
  }

  if (!decoder.atEnd())
    return invalid("payload has trailing bytes");
  auto reencoded = encodeDataflowRewriteDecision(*decision);
  if (!reencoded)
    return reencoded.takeError();
  if (llvm::ArrayRef<std::uint8_t>(*reencoded) != canonicalBytes)
    return invalid("payload does not re-encode exactly");
  return std::move(*decision);
}

bool dataflowRewriteDecisionLess(const DataflowRewriteDecision &lhs,
                                 const DataflowRewriteDecision &rhs) {
  const DataflowRewriteKind lhsKind = dataflowRewriteKind(lhs);
  const DataflowRewriteKind rhsKind = dataflowRewriteKind(rhs);
  if (lhsKind != rhsKind)
    return static_cast<std::uint32_t>(lhsKind) <
           static_cast<std::uint32_t>(rhsKind);

  switch (lhsKind) {
  case DataflowRewriteKind::SyncRendezvousRefactor: {
    const auto &left = std::get<SyncRendezvousRewrite>(lhs);
    const auto &right = std::get<SyncRendezvousRewrite>(rhs);
    if (int order = compareId(left.root, right.root))
      return order < 0;
    return left.direction < right.direction;
  }
  case DataflowRewriteKind::PackUnpackRoundTripEliminate:
    return compareId(std::get<PackUnpackRoundTripRewrite>(lhs).outerAdapter,
                     std::get<PackUnpackRoundTripRewrite>(rhs).outerAdapter) <
           0;
  case DataflowRewriteKind::ParallelizeSerializeRoundTripEliminate:
    return compareId(std::get<ParallelizeSerializeRoundTripRewrite>(lhs)
                         .outerSerialize,
                     std::get<ParallelizeSerializeRoundTripRewrite>(rhs)
                         .outerSerialize) < 0;
  case DataflowRewriteKind::ElementwiseCardinalityCommute: {
    const auto &left = std::get<ElementwiseCardinalityCommuteRewrite>(lhs);
    const auto &right = std::get<ElementwiseCardinalityCommuteRewrite>(rhs);
    if (int order = compareId(left.compute, right.compute))
      return order < 0;
    if (int order = compareIds<ActorId>(left.adapters, right.adapters))
      return order < 0;
    return left.direction < right.direction;
  }
  case DataflowRewriteKind::PureComputeFanoutRefactor: {
    const bool leftReplicate =
        std::holds_alternative<PureComputeFanoutReplicateRewrite>(lhs);
    const bool rightReplicate =
        std::holds_alternative<PureComputeFanoutReplicateRewrite>(rhs);
    if (leftReplicate != rightReplicate)
      return leftReplicate;
    if (leftReplicate)
      return compareId(
                 std::get<PureComputeFanoutReplicateRewrite>(lhs).compute,
                 std::get<PureComputeFanoutReplicateRewrite>(rhs).compute) < 0;
    return compareIds<ActorId>(
               std::get<PureComputeFanoutFactorRewrite>(lhs).replicas,
               std::get<PureComputeFanoutFactorRewrite>(rhs).replicas) < 0;
  }
  case DataflowRewriteKind::ActivationPreservingConstantFold:
    return compareId(
               std::get<ActivationPreservingConstantFoldRewrite>(lhs).compute,
               std::get<ActivationPreservingConstantFoldRewrite>(rhs).compute) <
           0;
  case DataflowRewriteKind::GraphDefinitionRefactor: {
    const bool leftSplit =
        std::holds_alternative<GraphDefinitionSplitRewrite>(lhs);
    const bool rightSplit =
        std::holds_alternative<GraphDefinitionSplitRewrite>(rhs);
    if (leftSplit != rightSplit)
      return leftSplit;
    if (leftSplit) {
      const auto &left = std::get<GraphDefinitionSplitRewrite>(lhs);
      const auto &right = std::get<GraphDefinitionSplitRewrite>(rhs);
      if (int order = compareId(left.graph, right.graph))
        return order < 0;
      return compareIds<StaticGraphLaunchId>(left.launches, right.launches) < 0;
    }
    const auto &left = std::get<GraphDefinitionMergeRewrite>(lhs);
    const auto &right = std::get<GraphDefinitionMergeRewrite>(rhs);
    if (int order = compareId(left.lowerGraph, right.lowerGraph))
      return order < 0;
    return compareId(left.higherGraph, right.higherGraph) < 0;
  }
  case DataflowRewriteKind::ElementwiseVectorDecompose: {
    const bool leftChunk =
        std::holds_alternative<ElementwiseVectorChunkRewrite>(lhs);
    const bool rightChunk =
        std::holds_alternative<ElementwiseVectorChunkRewrite>(rhs);
    const ActorId leftCompute =
        leftChunk ? std::get<ElementwiseVectorChunkRewrite>(lhs).compute
                  : std::get<ElementwiseVectorScalarizeRewrite>(lhs).compute;
    const ActorId rightCompute =
        rightChunk ? std::get<ElementwiseVectorChunkRewrite>(rhs).compute
                   : std::get<ElementwiseVectorScalarizeRewrite>(rhs).compute;
    if (int order = compareId(leftCompute, rightCompute))
      return order < 0;
    if (leftChunk != rightChunk)
      return leftChunk;
    if (!leftChunk)
      return false;
    return std::get<ElementwiseVectorChunkRewrite>(lhs).leadingBlocksPerChunk >
           std::get<ElementwiseVectorChunkRewrite>(rhs).leadingBlocksPerChunk;
  }
  }
  return false;
}

} // namespace dataflow
