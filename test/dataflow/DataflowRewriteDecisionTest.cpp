#include "Dataflow/Transforms/DataflowRewrite.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

[[noreturn]] void fail(llvm::StringRef message) {
  llvm::errs() << "dataflow rewrite decision: " << message << '\n';
  std::exit(EXIT_FAILURE);
}

void require(bool condition, llvm::StringRef message) {
  if (!condition)
    fail(message);
}

template <typename T> T take(llvm::Expected<T> value) {
  if (!value)
    fail(llvm::toString(value.takeError()));
  return std::move(*value);
}

template <typename T> bool rejected(llvm::Expected<T> value) {
  if (value)
    return false;
  llvm::consumeError(value.takeError());
  return true;
}

void appendU32(std::vector<std::uint8_t> &bytes, std::uint32_t value) {
  for (int shift = 24; shift >= 0; shift -= 8)
    bytes.push_back(static_cast<std::uint8_t>(value >> shift));
}

void appendU64(std::vector<std::uint8_t> &bytes, std::uint64_t value) {
  for (int shift = 56; shift >= 0; shift -= 8)
    bytes.push_back(static_cast<std::uint8_t>(value >> shift));
}

template <typename Decision>
void requireWire(const Decision &decision,
                 llvm::ArrayRef<std::uint8_t> expected) {
  dataflow::DataflowRewriteDecision erased = decision;
  std::vector<std::uint8_t> encoded =
      take(dataflow::encodeDataflowRewriteDecision(erased));
  require(llvm::ArrayRef<std::uint8_t>(encoded) == expected,
          "decision wire does not match schema 2.0");
  require(take(dataflow::adoptDataflowRewriteDecision(encoded)) == erased,
          "decision wire does not round trip");
}

void exactSchemaAndPayloads() {
  constexpr llvm::StringLiteral schema = "loom.dataflow_rewrite.decision.2.0";
  require(dataflow::dataflowRewriteDecisionSchemaBytes() ==
              llvm::ArrayRef<std::uint8_t>(
                  reinterpret_cast<const std::uint8_t *>(schema.data()),
                  schema.size()),
          "decision schema descriptor is not version 2.0");

  std::vector<std::uint8_t> bytes;
  appendU32(bytes, 0);
  appendU64(bytes, 11);
  appendU32(bytes, 1);
  requireWire(
      dataflow::SyncRendezvousRewrite{
          dataflow::ActorId(11),
          dataflow::SyncRendezvousDirection::TreeToDirect},
      bytes);

  bytes.clear();
  appendU32(bytes, 1);
  appendU64(bytes, 12);
  requireWire(dataflow::PackUnpackRoundTripRewrite{dataflow::ActorId(12)},
              bytes);

  bytes.clear();
  appendU32(bytes, 2);
  appendU64(bytes, 13);
  requireWire(
      dataflow::ParallelizeSerializeRoundTripRewrite{dataflow::ActorId(13)},
      bytes);

  bytes.clear();
  appendU32(bytes, 3);
  appendU64(bytes, 14);
  appendU64(bytes, 2);
  appendU64(bytes, 15);
  appendU64(bytes, 16);
  appendU32(bytes, 0);
  requireWire(
      dataflow::ElementwiseCardinalityCommuteRewrite{
          dataflow::ActorId(14),
          {dataflow::ActorId(15), dataflow::ActorId(16)},
          dataflow::CardinalityCommuteDirection::MoveInside},
      bytes);

  bytes.clear();
  appendU32(bytes, 4);
  appendU32(bytes, 0);
  appendU64(bytes, 17);
  requireWire(
      dataflow::PureComputeFanoutReplicateRewrite{dataflow::ActorId(17)},
      bytes);

  bytes.clear();
  appendU32(bytes, 4);
  appendU32(bytes, 1);
  appendU64(bytes, 2);
  appendU64(bytes, 18);
  appendU64(bytes, 19);
  requireWire(dataflow::PureComputeFanoutFactorRewrite{{dataflow::ActorId(18),
                                                        dataflow::ActorId(19)}},
              bytes);

  bytes.clear();
  appendU32(bytes, 5);
  appendU64(bytes, 20);
  requireWire(
      dataflow::ActivationPreservingConstantFoldRewrite{dataflow::ActorId(20)},
      bytes);

  bytes.clear();
  appendU32(bytes, 6);
  appendU32(bytes, 0);
  appendU64(bytes, 21);
  appendU64(bytes, 2);
  appendU64(bytes, 22);
  appendU64(bytes, 23);
  requireWire(
      dataflow::GraphDefinitionSplitRewrite{
          dataflow::GraphId(21),
          {dataflow::StaticGraphLaunchId(22),
           dataflow::StaticGraphLaunchId(23)}},
      bytes);

  bytes.clear();
  appendU32(bytes, 6);
  appendU32(bytes, 1);
  appendU64(bytes, 24);
  appendU64(bytes, 25);
  requireWire(dataflow::GraphDefinitionMergeRewrite{dataflow::GraphId(24),
                                                    dataflow::GraphId(25)},
              bytes);

  bytes.clear();
  appendU32(bytes, 7);
  appendU64(bytes, 26);
  appendU32(bytes, 0);
  appendU64(bytes, 4);
  requireWire(dataflow::ElementwiseVectorChunkRewrite{dataflow::ActorId(26), 4},
              bytes);

  bytes.clear();
  appendU32(bytes, 7);
  appendU64(bytes, 26);
  appendU32(bytes, 1);
  requireWire(
      dataflow::ElementwiseVectorScalarizeRewrite{dataflow::ActorId(26)},
      bytes);
}

void malformedPayloadsFailClosed() {
  require(rejected(dataflow::adoptDataflowRewriteDecision({0, 0})),
          "decision 1.0 payload was reinterpreted as 2.0");

  std::vector<std::uint8_t> unknown;
  appendU32(unknown, 8);
  require(rejected(dataflow::adoptDataflowRewriteDecision(unknown)),
          "unknown rewrite kind was accepted");

  dataflow::DataflowRewriteDecision duplicateSet =
      dataflow::PureComputeFanoutFactorRewrite{
          {dataflow::ActorId(4), dataflow::ActorId(4)}};
  require(rejected(dataflow::encodeDataflowRewriteDecision(duplicateSet)),
          "duplicate reference set was accepted");

  dataflow::DataflowRewriteDecision unsortedSet =
      dataflow::ElementwiseCardinalityCommuteRewrite{
          dataflow::ActorId(1),
          {dataflow::ActorId(3), dataflow::ActorId(2)},
          dataflow::CardinalityCommuteDirection::MoveOutside};
  require(rejected(dataflow::encodeDataflowRewriteDecision(unsortedSet)),
          "unsorted reference set was accepted");

  dataflow::DataflowRewriteDecision singletonFactor =
      dataflow::PureComputeFanoutFactorRewrite{{dataflow::ActorId(4)}};
  require(rejected(dataflow::encodeDataflowRewriteDecision(singletonFactor)),
          "singleton fanout factor set was accepted");

  dataflow::DataflowRewriteDecision reversedMerge =
      dataflow::GraphDefinitionMergeRewrite{dataflow::GraphId(8),
                                            dataflow::GraphId(7)};
  require(rejected(dataflow::encodeDataflowRewriteDecision(reversedMerge)),
          "noncanonical graph pair was accepted");

  dataflow::DataflowRewriteDecision zeroChunk =
      dataflow::ElementwiseVectorChunkRewrite{dataflow::ActorId(1), 0};
  require(rejected(dataflow::encodeDataflowRewriteDecision(zeroChunk)),
          "zero vector chunk was accepted");
}

void semanticOrderingIsStable() {
  using Decision = dataflow::DataflowRewriteDecision;
  std::vector<Decision> shuffled = {
      dataflow::ElementwiseVectorScalarizeRewrite{dataflow::ActorId(9)},
      dataflow::ElementwiseVectorChunkRewrite{dataflow::ActorId(9), 2},
      dataflow::PackUnpackRoundTripRewrite{dataflow::ActorId(1)},
      dataflow::ElementwiseVectorChunkRewrite{dataflow::ActorId(9), 4},
      dataflow::SyncRendezvousRewrite{
          dataflow::ActorId(2),
          dataflow::SyncRendezvousDirection::DirectToTree},
  };
  std::sort(shuffled.begin(), shuffled.end(),
            dataflow::dataflowRewriteDecisionLess);
  require(
      std::holds_alternative<dataflow::SyncRendezvousRewrite>(shuffled[0]) &&
          std::holds_alternative<dataflow::PackUnpackRoundTripRewrite>(
              shuffled[1]),
      "catalog kind order is unstable");
  require(
      std::get<dataflow::ElementwiseVectorChunkRewrite>(shuffled[2])
                  .leadingBlocksPerChunk == 4 &&
          std::get<dataflow::ElementwiseVectorChunkRewrite>(shuffled[3])
                  .leadingBlocksPerChunk == 2 &&
          std::holds_alternative<dataflow::ElementwiseVectorScalarizeRewrite>(
              shuffled[4]),
      "vector decomposition order is not descending chunks then scalar");
}

} // namespace

int main() {
  exactSchemaAndPayloads();
  malformedPayloadsFailClosed();
  semanticOrderingIsStable();
  return EXIT_SUCCESS;
}
