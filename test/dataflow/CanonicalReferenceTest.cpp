#include "Dataflow/IR/DataflowArtifactLocalReference.h"
#include "Dataflow/IR/DataflowReferenceCodec.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

using namespace dataflow;
using namespace loom;

namespace {

[[noreturn]] void fail(llvm::StringRef message) {
  llvm::errs() << "canonical reference: " << message << '\n';
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

ArtifactIdentity identity(std::uint8_t seed) {
  ArtifactIdentity::Storage bytes{};
  for (std::size_t index = 0; index < bytes.size(); ++index)
    bytes[index] = static_cast<std::uint8_t>(seed + index);
  return take(ArtifactIdentity::fromBytes(bytes));
}

void appendU32(std::vector<std::uint8_t> &bytes, std::uint32_t value) {
  for (int shift = 24; shift >= 0; shift -= 8)
    bytes.push_back(static_cast<std::uint8_t>(value >> shift));
}

void appendU64(std::vector<std::uint8_t> &bytes, std::uint64_t value) {
  for (int shift = 56; shift >= 0; shift -= 8)
    bytes.push_back(static_cast<std::uint8_t>(value >> shift));
}

template <typename Ref>
void requireRoundTrip(const ArtifactIdentity &owner, const Ref &reference) {
  std::vector<std::uint8_t> bytes = take(encodeDataflowReference(reference));
  Ref decoded = take(decodeDataflowReference<Ref>(bytes, owner));
  require(decoded == reference, "typed reference wire did not round trip");

  EncodedArtifactLocalReference encoded =
      take(encodeDataflowArtifactLocalReference(owner, reference));
  Ref imported = take(decodeDataflowArtifactLocalReference<Ref>(encoded));
  require(imported == reference,
          "owner-local artifact reference did not round trip");
}

void entityAndRootedReferenceRoundTrip() {
  static_assert(sizeof(StructuralOrdinal) == sizeof(std::uint64_t));
  ArtifactIdentity owner = identity(7);
  GraphRef graph{owner, GraphId(1)};
  ActorRef actor{owner, ActorId(2)};
  RootThreadLaunchRef root{owner, RootThreadLaunchId(3)};
  StaticGraphLaunchRef launch{owner, StaticGraphLaunchId(4)};
  LogicalMemoryRootRef memory{owner, LogicalMemoryRootId(5)};

  requireRoundTrip(owner, graph);
  requireRoundTrip(owner, actor);
  requireRoundTrip(owner, root);
  requireRoundTrip(owner, launch);
  requireRoundTrip(owner, memory);

  RootedGraphLaunchRef rooted{root, launch};
  requireRoundTrip(owner, rooted);
  std::vector<std::uint8_t> expected;
  appendU64(expected, 3);
  appendU64(expected, 4);
  require(take(encodeDataflowReference(rooted)) == expected,
          "rooted launch wire must contain the two typed local IDs in order");

  RootedGraphLaunchRef mixed{
      root, StaticGraphLaunchRef{identity(31), StaticGraphLaunchId(4)}};
  require(rejected(encodeDataflowReference(mixed)),
          "mixed-artifact nested references must be rejected");
}

void ownerLocalReferenceCatalogAnchors() {
  static_assert(std::is_same_v<EventFamilyKey, StaticTransferEventRef>);
  static_assert(dataflowArtifactLocalReferenceKindCount() == 26);
  static_assert(dataflowArtifactLocalReferenceKindOrdinal(
                    DataflowArtifactLocalReferenceKind::GraphRef) == 0);
  static_assert(dataflowArtifactLocalReferenceKindOrdinal(
                    DataflowArtifactLocalReferenceKind::RootedGraphLaunchRef) ==
                5);
  static_assert(
      dataflowArtifactLocalReferenceKindOrdinal(
          DataflowArtifactLocalReferenceKind::RootThreadBoundaryTransferRef) ==
      12);
  static_assert(
      dataflowArtifactLocalReferenceKindOrdinal(
          DataflowArtifactLocalReferenceKind::LogicalMemoryRootOrViewRef) ==
      21);
  static_assert(
      dataflowArtifactLocalReferenceKindOrdinal(
          DataflowArtifactLocalReferenceKind::StaticTransferEventRef) == 25);

  constexpr std::array<llvm::StringLiteral, 26> expectedTargets = {
      "GraphRef",
      "ActorRef",
      "RootThreadLaunchRef",
      "StaticGraphLaunchRef",
      "LogicalMemoryRootRef",
      "RootedGraphLaunchRef",
      "GraphIngressTokenRef",
      "GraphEgressTokenRef",
      "ActorTokenResultRef",
      "ActorTokenOperandRef",
      "CanonicalGraphProducerEndpointRef",
      "CanonicalGraphConsumerEndpointRef",
      "RootThreadBoundaryTransferRef",
      "GraphLaunchBoundaryTransferRef",
      "ThreadChannelSendSiteRef",
      "ThreadChannelReceiveSiteRef",
      "ChannelProducerRef",
      "ChannelConsumerRef",
      "CanonicalProducerTerminalRef",
      "CanonicalSinkTerminalRef",
      "LogicalMemoryViewRef",
      "LogicalMemoryRootOrViewRef",
      "ContextualActorRef",
      "MemoryExposureRef",
      "FenceActorFamilyRef",
      "StaticTransferEventRef",
  };
  llvm::ArrayRef<DataflowArtifactLocalReferenceKindDescriptor> catalog =
      dataflowArtifactLocalReferenceKindCatalog();
  require(catalog.size() == expectedTargets.size(),
          "owner-local reference catalog count changed");
  for (std::size_t index = 0; index < expectedTargets.size(); ++index) {
    require(dataflowArtifactLocalReferenceKindOrdinal(catalog[index].kind) ==
                index,
            "owner-local reference catalog ordinal changed");
    require(catalog[index].typedTarget == expectedTargets[index],
            "owner-local reference catalog order changed");
  }
}

void closedUnionWireTagAnchors() {
  ArtifactIdentity owner = identity(9);
  RootThreadLaunchRef launch{owner, RootThreadLaunchId(17)};
  RootThreadBoundaryTransferRef start = RootThreadStartTransferRef{launch};
  RootThreadBoundaryTransferRef completion =
      RootThreadCompletionTransferRef{launch};

  std::vector<std::uint8_t> expectedStart;
  appendU32(expectedStart, 0);
  appendU64(expectedStart, 17);
  std::vector<std::uint8_t> expectedCompletion;
  appendU32(expectedCompletion, 2);
  appendU64(expectedCompletion, 17);

  std::vector<std::uint8_t> startBytes = take(encodeDataflowReference(start));
  std::vector<std::uint8_t> completionBytes =
      take(encodeDataflowReference(completion));
  require(startBytes == expectedStart,
          "root-thread Start wire tag changed from zero");
  require(completionBytes == expectedCompletion,
          "root-thread Completion wire tag changed from two");
  require(startBytes != completionBytes,
          "same-shaped Start and Completion must retain distinct wire tags");
  require(take(decodeDataflowReference<RootThreadBoundaryTransferRef>(
              expectedStart, owner)) == start,
          "root-thread Start wire decoded as another alternative");
  require(take(decodeDataflowReference<RootThreadBoundaryTransferRef>(
              expectedCompletion, owner)) == completion,
          "root-thread Completion wire decoded as another alternative");
}

void terminalAndEventRoundTrip() {
  ArtifactIdentity owner = identity(11);
  constexpr std::uint64_t streamOrdinal = 0x100000003ULL;
  RootedGraphLaunchRef rooted{
      RootThreadLaunchRef{owner, RootThreadLaunchId(9)},
      StaticGraphLaunchRef{owner, StaticGraphLaunchId(12)}};
  ChannelProducerRef producer =
      GraphStreamOutputProducerRef{rooted, streamOrdinal};
  CanonicalProducerTerminalRef terminal = ChannelProducerTerminalRef{producer};
  StaticTransferEventRef event = ProducedTransferEventRef{terminal};

  requireRoundTrip(owner, producer);
  requireRoundTrip(owner, terminal);
  requireRoundTrip(owner, event);

  std::vector<std::uint8_t> expected;
  appendU32(expected, 0); // Produced
  appendU32(expected, 2); // ChannelProducer terminal
  appendU32(expected, 0); // GraphStreamOutput producer
  appendU64(expected, 9);
  appendU64(expected, 12);
  appendU64(expected, streamOrdinal);
  require(take(encodeDataflowReference(event)) == expected,
          "event-family wire must use declaration-order discriminants");

  EventFamilyKey key = event;
  require(take(encodeDataflowReference(key)) == expected,
          "EventFamilyKey must be exactly StaticTransferEventRef");

  RootThreadBoundaryTransferRef rootTransfer =
      RootThreadValueInputTransferRef{rooted.rootThreadLaunch, 6};
  CanonicalProducerTerminalRef rootSource =
      RootThreadBoundarySourceRef{rootTransfer};
  CanonicalSinkTerminalRef rootSink = RootThreadBoundarySinkRef{rootTransfer};
  requireRoundTrip(owner, rootSource);
  requireRoundTrip(owner, rootSink);

  GraphLaunchBoundaryTransferRef graphTransfer =
      GraphLaunchValueResultTransferRef{rooted, 7};
  CanonicalProducerTerminalRef graphSource =
      GraphLaunchBoundarySourceRef{graphTransfer};
  CanonicalSinkTerminalRef graphSink =
      GraphLaunchBoundarySinkRef{graphTransfer};
  requireRoundTrip(owner, graphSource);
  requireRoundTrip(owner, graphSink);

  ChannelConsumerRef consumer =
      GraphStreamInputConsumerRef{rooted, streamOrdinal};
  StaticTransferEventRef consumed = ConsumedTransferEventRef{
      CanonicalSinkTerminalRef{ChannelConsumerTerminalRef{consumer}}};
  requireRoundTrip(owner, consumed);
}

void projectionOrderingAndWireRoundTrip() {
  EventLogicalProjection empty;
  CanonicalSemanticBytes emptyBytes = take(encodeEventLogicalProjection(empty));
  const std::array<std::uint8_t, 8> emptyWire{};
  require(emptyBytes.bytes() == llvm::ArrayRef<std::uint8_t>(emptyWire),
          "empty projection must encode a zero u64 count");
  require(take(decodeEventLogicalProjection(emptyBytes.bytes())) == empty,
          "empty projection did not round trip");

  EventLogicalProjection projection{
      CoordinateSlot{0},      CoordinateSlot{1},      LaunchParameterSlot{0},
      LaunchParameterSlot{1}, LaunchParameterSlot{2},
  };
  require(projection.size() == 5,
          "projection fixture must contain every canonical slot");
  require(std::holds_alternative<CoordinateSlot>(projection[0]) &&
              std::get<CoordinateSlot>(projection[0]).ordinal == 0 &&
              std::holds_alternative<CoordinateSlot>(projection[1]) &&
              std::get<CoordinateSlot>(projection[1]).ordinal == 1 &&
              std::holds_alternative<LaunchParameterSlot>(projection[2]) &&
              std::get<LaunchParameterSlot>(projection[2]).ordinal == 0,
          "projection order must be Coordinate before LaunchParameter");

  CanonicalSemanticBytes bytes = take(encodeEventLogicalProjection(projection));
  std::vector<std::uint8_t> expected;
  appendU64(expected, 5);
  appendU32(expected, 0);
  appendU64(expected, 0);
  appendU32(expected, 0);
  appendU64(expected, 1);
  for (std::uint64_t ordinal = 0; ordinal < 3; ++ordinal) {
    appendU32(expected, 1);
    appendU64(expected, ordinal);
  }
  require(bytes.bytes() == llvm::ArrayRef<std::uint8_t>(expected),
          "projection wire must use u64be count and u32be/u64be slots");
  require(take(decodeEventLogicalProjection(bytes.bytes())) == projection,
          "non-empty projection did not round trip");

  std::vector<std::uint8_t> reordered = expected;
  const std::size_t firstSlot = 8;
  const std::size_t thirdSlot = firstSlot + 24;
  std::copy(reordered.begin() + thirdSlot, reordered.begin() + thirdSlot + 12,
            reordered.begin() + firstSlot);
  require(rejected(decodeEventLogicalProjection(reordered)),
          "noncanonical projection order must be rejected");

  std::vector<std::uint8_t> duplicate = expected;
  std::copy(duplicate.begin() + firstSlot, duplicate.begin() + firstSlot + 12,
            duplicate.begin() + firstSlot + 12);
  require(rejected(decodeEventLogicalProjection(duplicate)),
          "duplicate projection slots must be rejected");

  std::vector<std::uint8_t> unknownKind = expected;
  unknownKind[firstSlot + 3] = 2;
  require(rejected(decodeEventLogicalProjection(unknownKind)),
          "unknown projection slot kinds must be rejected");

  std::vector<std::uint8_t> truncated = expected;
  truncated.pop_back();
  require(rejected(decodeEventLogicalProjection(truncated)),
          "truncated projection slots must be rejected");

  std::vector<std::uint8_t> trailingProjection = expected;
  trailingProjection.push_back(0);
  require(rejected(decodeEventLogicalProjection(trailingProjection)),
          "trailing projection bytes must be rejected");

  EventLogicalProjection noncanonical{LaunchParameterSlot{0},
                                      CoordinateSlot{0}};
  require(rejected(encodeEventLogicalProjection(noncanonical)),
          "the projection encoder must reject noncanonical input order");
}

void strictOwnerLocalRejection() {
  ArtifactIdentity owner = identity(19);
  RootedGraphLaunchRef rooted{
      RootThreadLaunchRef{owner, RootThreadLaunchId(4)},
      StaticGraphLaunchRef{owner, StaticGraphLaunchId(8)}};
  EncodedArtifactLocalReference encoded =
      take(encodeDataflowArtifactLocalReference(owner, rooted));

  require(rejected(encodeDataflowArtifactLocalReference(identity(29), rooted)),
          "the explicit outer artifact must match every nested reference");

  EncodedArtifactLocalReference foreign = encoded;
  foreign.artifact.schemaIdentity = "loom.fabric";
  require(rejected(decodeDataflowArtifactLocalReference<RootedGraphLaunchRef>(
              foreign)),
          "foreign owner schema must be rejected");

  require(rejected(decodeDataflowArtifactLocalReference<StaticTransferEventRef>(
              encoded)),
          "wrong owner-local kind must be rejected");

  EncodedArtifactLocalReference unknownKind = encoded;
  unknownKind.ownerLocalKind = dataflowArtifactLocalReferenceKindCount();
  require(rejected(decodeDataflowArtifactLocalReference<RootedGraphLaunchRef>(
              unknownKind)),
          "unknown owner-local kind ordinals must be rejected");

  EncodedArtifactLocalReference trailing = encoded;
  trailing.payload.push_back(0);
  require(rejected(decodeDataflowArtifactLocalReference<RootedGraphLaunchRef>(
              trailing)),
          "trailing payload bytes must be rejected");

  std::vector<std::uint8_t> unknownVariant(4, 0xff);
  require(rejected(decodeDataflowReference<StaticTransferEventRef>(
              unknownVariant, owner)),
          "unknown closed-union discriminants must be rejected");
}

} // namespace

int main() {
  entityAndRootedReferenceRoundTrip();
  ownerLocalReferenceCatalogAnchors();
  closedUnionWireTagAnchors();
  terminalAndEventRoundTrip();
  projectionOrderingAndWireRoundTrip();
  strictOwnerLocalRejection();
  return EXIT_SUCCESS;
}
