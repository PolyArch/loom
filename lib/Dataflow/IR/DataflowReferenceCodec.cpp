#include "Dataflow/IR/DataflowReferenceCodec.h"
#include "Dataflow/IR/DataflowStructuralRefUnions.def"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <optional>
#include <type_traits>
#include <utility>

namespace dataflow {

char DataflowReferenceError::ID = 0;

void DataflowReferenceError::log(llvm::raw_ostream &stream) const {
  stream << message_;
}

std::error_code DataflowReferenceError::convertToErrorCode() const {
  return llvm::inconvertibleErrorCode();
}

llvm::Error makeDataflowReferenceError(DataflowReferenceErrorKind kind,
                                       const llvm::Twine &message) {
  return llvm::make_error<DataflowReferenceError>(kind, message.str());
}

namespace {

class ReferenceWriter {
public:
  explicit ReferenceWriter(const ::loom::ArtifactIdentity *expectedArtifact) {
    if (expectedArtifact)
      artifact_ = *expectedArtifact;
  }

  llvm::Error bind(const ::loom::ArtifactIdentity &artifact) {
    if (artifact_ && *artifact_ != artifact)
      return makeDataflowReferenceError(
          DataflowReferenceErrorKind::ForeignArtifact,
          "nested Canonical Dataflow references bind different artifacts");
    artifact_ = artifact;
    return llvm::Error::success();
  }

  void putU32(std::uint32_t value) {
    for (int shift = 24; shift >= 0; shift -= 8)
      bytes_.push_back(static_cast<std::uint8_t>(value >> shift));
  }

  void putU64(std::uint64_t value) {
    for (int shift = 56; shift >= 0; shift -= 8)
      bytes_.push_back(static_cast<std::uint8_t>(value >> shift));
  }

  llvm::Expected<std::vector<std::uint8_t>> take() {
    if (!artifact_)
      return makeDataflowReferenceError(
          DataflowReferenceErrorKind::MissingArtifact,
          "Canonical Dataflow reference does not bind an artifact");
    return std::move(bytes_);
  }

private:
  std::optional<::loom::ArtifactIdentity> artifact_;
  std::vector<std::uint8_t> bytes_;
};

class ReferenceReader {
public:
  ReferenceReader(llvm::ArrayRef<std::uint8_t> bytes,
                  const ::loom::ArtifactIdentity &artifact)
      : bytes_(bytes), artifact_(artifact) {}

  llvm::Expected<std::uint32_t> getU32() {
    if (bytes_.size() < 4)
      return makeDataflowReferenceError(
          DataflowReferenceErrorKind::MalformedSyntax,
          "truncated Canonical Dataflow variant discriminant");
    std::uint32_t value = 0;
    for (unsigned index = 0; index < 4; ++index)
      value = (value << 8) | bytes_[index];
    bytes_ = bytes_.drop_front(4);
    return value;
  }

  llvm::Expected<std::uint64_t> getU64() {
    if (bytes_.size() < 8)
      return makeDataflowReferenceError(
          DataflowReferenceErrorKind::MalformedSyntax,
          "truncated Canonical Dataflow identifier or ordinal");
    std::uint64_t value = 0;
    for (unsigned index = 0; index < 8; ++index)
      value = (value << 8) | bytes_[index];
    bytes_ = bytes_.drop_front(8);
    return value;
  }

  const ::loom::ArtifactIdentity &artifact() const { return artifact_; }
  bool empty() const { return bytes_.empty(); }

private:
  llvm::ArrayRef<std::uint8_t> bytes_;
  const ::loom::ArtifactIdentity &artifact_;
};

template <typename T> struct TypeTag {};

template <typename Union, typename Alternative>
struct ClosedUnionAlternativeTraits;

#define LOOM_DEFINE_CLOSED_UNION_ALTERNATIVE(Union, WireTag, Type)             \
  template <> struct ClosedUnionAlternativeTraits<Union, Type> {               \
    static constexpr std::uint32_t wireTag = WireTag;                          \
  };
#define LOOM_DEFINE_CLOSED_UNION_ALTERNATIVES(Union, Label, Alternatives)      \
  Alternatives(LOOM_DEFINE_CLOSED_UNION_ALTERNATIVE, Union)
LOOM_DATAFLOW_CLOSED_UNIONS(LOOM_DEFINE_CLOSED_UNION_ALTERNATIVES)
#undef LOOM_DEFINE_CLOSED_UNION_ALTERNATIVES
#undef LOOM_DEFINE_CLOSED_UNION_ALTERNATIVE

#define LOOM_COUNT_CLOSED_UNION_ALTERNATIVE(Union, WireTag, Type) +1
#define LOOM_CHECK_CLOSED_UNION_ALTERNATIVE(Union, WireTag, Type)              \
  static_assert(                                                               \
      std::is_same_v<std::variant_alternative_t<WireTag, Union>, Type>,        \
      #Union " alternatives must retain their declared wire-tag order");
#define LOOM_CHECK_CLOSED_UNION(Union, Label, Alternatives)                    \
  static_assert(                                                               \
      std::variant_size_v<Union> ==                                            \
          (0 Alternatives(LOOM_COUNT_CLOSED_UNION_ALTERNATIVE, Union)),        \
      #Union " declaration and wire catalog must contain the same entries");   \
  Alternatives(LOOM_CHECK_CLOSED_UNION_ALTERNATIVE, Union)
LOOM_DATAFLOW_CLOSED_UNIONS(LOOM_CHECK_CLOSED_UNION)
#undef LOOM_CHECK_CLOSED_UNION
#undef LOOM_CHECK_CLOSED_UNION_ALTERNATIVE
#undef LOOM_COUNT_CLOSED_UNION_ALTERNATIVE

template <typename Union> std::uint32_t closedUnionWireTag(const Union &value) {
  return std::visit(
      [](const auto &alternative) {
        using Alternative = std::decay_t<decltype(alternative)>;
        return ClosedUnionAlternativeTraits<Union, Alternative>::wireTag;
      },
      value);
}

#define LOOM_DECLARE_CLOSED_UNION_DECODER(Union, Label, Alternatives)          \
  llvm::Expected<Union> decodeReference(ReferenceReader &reader,               \
                                        TypeTag<Union>);
LOOM_DATAFLOW_STRUCTURAL_REFERENCE_UNIONS(LOOM_DECLARE_CLOSED_UNION_DECODER)
#undef LOOM_DECLARE_CLOSED_UNION_DECODER

template <CanonicalDataflowEntityKind Kind>
llvm::Error encodeReference(
    ReferenceWriter &writer,
    const ::loom::ArtifactReference<CanonicalDataflowEntityId<Kind>> &ref) {
  if (llvm::Error error = writer.bind(ref.artifact))
    return error;
  writer.putU64(ref.entity.value());
  return llvm::Error::success();
}

llvm::Error encodeReference(ReferenceWriter &writer,
                            const RootedGraphLaunchRef &ref) {
  if (llvm::Error error = encodeReference(writer, ref.rootThreadLaunch))
    return error;
  return encodeReference(writer, ref.staticGraphLaunch);
}

#define LOOM_ENCODE_ENTITY_ORDINAL(Type, EntityField, OrdinalField)            \
  llvm::Error encodeReference(ReferenceWriter &writer, const Type &ref) {      \
    if (llvm::Error error = encodeReference(writer, ref.EntityField))          \
      return error;                                                            \
    writer.putU64(ref.OrdinalField);                                           \
    return llvm::Error::success();                                             \
  }

llvm::Error encodeReference(ReferenceWriter &writer,
                            const GraphStartTokenRef &ref) {
  return encodeReference(writer, ref.graph);
}
LOOM_ENCODE_ENTITY_ORDINAL(GraphValueInputTokenRef, graph, ordinal)
LOOM_ENCODE_ENTITY_ORDINAL(GraphStreamInputTokenRef, graph, ordinal)
LOOM_ENCODE_ENTITY_ORDINAL(GraphValueOutputTokenRef, graph, ordinal)
LOOM_ENCODE_ENTITY_ORDINAL(GraphStreamOutputTokenRef, graph, ordinal)
LOOM_ENCODE_ENTITY_ORDINAL(GraphCompletionFrontierTokenRef, graph, ordinal)
LOOM_ENCODE_ENTITY_ORDINAL(ActorTokenResultRef, actor, ordinal)
LOOM_ENCODE_ENTITY_ORDINAL(ActorTokenOperandRef, actor, ordinal)
LOOM_ENCODE_ENTITY_ORDINAL(RootThreadValueInputTransferRef, launch, ordinal)
LOOM_ENCODE_ENTITY_ORDINAL(ThreadChannelSendSiteRef, launch, ordinal)
LOOM_ENCODE_ENTITY_ORDINAL(ThreadChannelReceiveSiteRef, launch, ordinal)
LOOM_ENCODE_ENTITY_ORDINAL(LogicalMemoryViewRef, root, viewOrdinal)

#undef LOOM_ENCODE_ENTITY_ORDINAL

llvm::Error encodeReference(ReferenceWriter &writer,
                            const RootThreadStartTransferRef &ref) {
  return encodeReference(writer, ref.launch);
}

llvm::Error encodeReference(ReferenceWriter &writer,
                            const RootThreadCompletionTransferRef &ref) {
  return encodeReference(writer, ref.launch);
}

#define LOOM_ENCODE_ROOTED_ORDINAL(Type, OrdinalField)                         \
  llvm::Error encodeReference(ReferenceWriter &writer, const Type &ref) {      \
    if (llvm::Error error = encodeReference(writer, ref.launch))               \
      return error;                                                            \
    writer.putU64(ref.OrdinalField);                                           \
    return llvm::Error::success();                                             \
  }

llvm::Error encodeReference(ReferenceWriter &writer,
                            const GraphLaunchStartTransferRef &ref) {
  return encodeReference(writer, ref.launch);
}
LOOM_ENCODE_ROOTED_ORDINAL(GraphLaunchValueInputTransferRef, ordinal)
LOOM_ENCODE_ROOTED_ORDINAL(GraphLaunchValueResultTransferRef, ordinal)
llvm::Error encodeReference(ReferenceWriter &writer,
                            const GraphLaunchDoneTransferRef &ref) {
  return encodeReference(writer, ref.launch);
}
LOOM_ENCODE_ROOTED_ORDINAL(GraphStreamOutputProducerRef, ordinal)
LOOM_ENCODE_ROOTED_ORDINAL(GraphStreamInputConsumerRef, ordinal)

#undef LOOM_ENCODE_ROOTED_ORDINAL

llvm::Error encodeReference(ReferenceWriter &writer,
                            const RootThreadBoundarySourceRef &ref);
llvm::Error encodeReference(ReferenceWriter &writer,
                            const GraphLaunchBoundarySourceRef &ref);
llvm::Error encodeReference(ReferenceWriter &writer,
                            const ChannelProducerTerminalRef &ref);
llvm::Error encodeReference(ReferenceWriter &writer,
                            const RootThreadBoundarySinkRef &ref);
llvm::Error encodeReference(ReferenceWriter &writer,
                            const GraphLaunchBoundarySinkRef &ref);
llvm::Error encodeReference(ReferenceWriter &writer,
                            const ChannelConsumerTerminalRef &ref);
llvm::Error encodeReference(ReferenceWriter &writer,
                            const MessageTransferMemberRef &ref);
llvm::Error encodeReference(ReferenceWriter &writer,
                            const AddressedMemoryActorMemberRef &ref);
llvm::Error encodeReference(ReferenceWriter &writer,
                            const FenceActorMemberRef &ref);
llvm::Error encodeReference(ReferenceWriter &writer,
                            const ProducedTransferEventRef &ref);
llvm::Error encodeReference(ReferenceWriter &writer,
                            const ConsumedTransferEventRef &ref);

template <typename... Alternatives>
llvm::Error encodeClosedVariant(ReferenceWriter &writer,
                                const std::variant<Alternatives...> &value) {
  return std::visit(
      [&](const auto &alternative) -> llvm::Error {
        using Union = std::variant<Alternatives...>;
        using Alternative = std::decay_t<decltype(alternative)>;
        writer.putU32(
            ClosedUnionAlternativeTraits<Union, Alternative>::wireTag);
        return encodeReference(writer, alternative);
      },
      value);
}

#define LOOM_DEFINE_CLOSED_UNION_ENCODER(Union, Label, Alternatives)           \
  llvm::Error encodeReference(ReferenceWriter &writer, const Union &ref) {     \
    return encodeClosedVariant(writer, ref);                                   \
  }
LOOM_DATAFLOW_STRUCTURAL_REFERENCE_UNIONS(LOOM_DEFINE_CLOSED_UNION_ENCODER)
#undef LOOM_DEFINE_CLOSED_UNION_ENCODER

#define LOOM_ENCODE_WRAPPER(Type, Field)                                       \
  llvm::Error encodeReference(ReferenceWriter &writer, const Type &ref) {      \
    return encodeReference(writer, ref.Field);                                 \
  }

LOOM_ENCODE_WRAPPER(RootThreadBoundarySourceRef, transfer)
LOOM_ENCODE_WRAPPER(GraphLaunchBoundarySourceRef, transfer)
LOOM_ENCODE_WRAPPER(ChannelProducerTerminalRef, producer)
LOOM_ENCODE_WRAPPER(RootThreadBoundarySinkRef, transfer)
LOOM_ENCODE_WRAPPER(GraphLaunchBoundarySinkRef, transfer)
LOOM_ENCODE_WRAPPER(ChannelConsumerTerminalRef, consumer)

#undef LOOM_ENCODE_WRAPPER

llvm::Error encodeReference(ReferenceWriter &writer,
                            const ContextualActorRef &ref) {
  if (llvm::Error error = encodeReference(writer, ref.launch))
    return error;
  return encodeReference(writer, ref.actor);
}

llvm::Error encodeReference(ReferenceWriter &writer,
                            const MemoryExposureRef &ref) {
  if (llvm::Error error = encodeReference(writer, ref.launch))
    return error;
  writer.putU64(ref.memoryResultOrdinal);
  return llvm::Error::success();
}

llvm::Error encodeReference(ReferenceWriter &writer,
                            const FenceActorFamilyRef &ref) {
  return encodeReference(writer, ref.actor);
}

llvm::Error encodeReference(ReferenceWriter &,
                            const MessageTransferMemberRef &) {
  return llvm::Error::success();
}
llvm::Error encodeReference(ReferenceWriter &writer,
                            const AddressedMemoryActorMemberRef &ref) {
  return encodeReference(writer, ref.actor);
}
llvm::Error encodeReference(ReferenceWriter &writer,
                            const FenceActorMemberRef &ref) {
  return encodeReference(writer, ref.actor);
}
llvm::Error encodeReference(ReferenceWriter &writer,
                            const ProducedTransferEventRef &ref) {
  return encodeReference(writer, ref.terminal);
}
llvm::Error encodeReference(ReferenceWriter &writer,
                            const ConsumedTransferEventRef &ref) {
  return encodeReference(writer, ref.terminal);
}
template <CanonicalDataflowEntityKind Kind>
llvm::Expected<::loom::ArtifactReference<CanonicalDataflowEntityId<Kind>>>
decodeEntityReference(ReferenceReader &reader) {
  llvm::Expected<std::uint64_t> id = reader.getU64();
  if (!id)
    return id.takeError();
  return ::loom::ArtifactReference<CanonicalDataflowEntityId<Kind>>{
      reader.artifact(), CanonicalDataflowEntityId<Kind>(*id)};
}

llvm::Expected<GraphRef> decodeReference(ReferenceReader &reader,
                                         TypeTag<GraphRef>) {
  return decodeEntityReference<CanonicalDataflowEntityKind::Graph>(reader);
}
llvm::Expected<ActorRef> decodeReference(ReferenceReader &reader,
                                         TypeTag<ActorRef>) {
  return decodeEntityReference<CanonicalDataflowEntityKind::Actor>(reader);
}
llvm::Expected<RootThreadLaunchRef>
decodeReference(ReferenceReader &reader, TypeTag<RootThreadLaunchRef>) {
  return decodeEntityReference<CanonicalDataflowEntityKind::RootThreadLaunch>(
      reader);
}
llvm::Expected<StaticGraphLaunchRef>
decodeReference(ReferenceReader &reader, TypeTag<StaticGraphLaunchRef>) {
  return decodeEntityReference<CanonicalDataflowEntityKind::StaticGraphLaunch>(
      reader);
}
llvm::Expected<LogicalMemoryRootRef>
decodeReference(ReferenceReader &reader, TypeTag<LogicalMemoryRootRef>) {
  return decodeEntityReference<CanonicalDataflowEntityKind::LogicalMemoryRoot>(
      reader);
}

llvm::Expected<RootedGraphLaunchRef>
decodeReference(ReferenceReader &reader, TypeTag<RootedGraphLaunchRef>) {
  llvm::Expected<RootThreadLaunchRef> root =
      decodeReference(reader, TypeTag<RootThreadLaunchRef>{});
  if (!root)
    return root.takeError();
  llvm::Expected<StaticGraphLaunchRef> launch =
      decodeReference(reader, TypeTag<StaticGraphLaunchRef>{});
  if (!launch)
    return launch.takeError();
  return RootedGraphLaunchRef{*root, *launch};
}

template <typename Ref, typename EntityRef>
llvm::Expected<Ref> decodeEntityOrdinal(ReferenceReader &reader,
                                        TypeTag<EntityRef> entityTag) {
  llvm::Expected<EntityRef> entity = decodeReference(reader, entityTag);
  if (!entity)
    return entity.takeError();
  llvm::Expected<std::uint64_t> ordinal = reader.getU64();
  if (!ordinal)
    return ordinal.takeError();
  return Ref{*entity, *ordinal};
}

llvm::Expected<GraphStartTokenRef>
decodeReference(ReferenceReader &reader, TypeTag<GraphStartTokenRef>) {
  llvm::Expected<GraphRef> graph = decodeReference(reader, TypeTag<GraphRef>{});
  if (!graph)
    return graph.takeError();
  return GraphStartTokenRef{*graph};
}

#define LOOM_DECODE_ENTITY_ORDINAL(Type, EntityType)                           \
  llvm::Expected<Type> decodeReference(ReferenceReader &reader,                \
                                       TypeTag<Type>) {                        \
    return decodeEntityOrdinal<Type>(reader, TypeTag<EntityType>{});           \
  }

LOOM_DECODE_ENTITY_ORDINAL(GraphValueInputTokenRef, GraphRef)
LOOM_DECODE_ENTITY_ORDINAL(GraphStreamInputTokenRef, GraphRef)
LOOM_DECODE_ENTITY_ORDINAL(GraphValueOutputTokenRef, GraphRef)
LOOM_DECODE_ENTITY_ORDINAL(GraphStreamOutputTokenRef, GraphRef)
LOOM_DECODE_ENTITY_ORDINAL(GraphCompletionFrontierTokenRef, GraphRef)
LOOM_DECODE_ENTITY_ORDINAL(ActorTokenResultRef, ActorRef)
LOOM_DECODE_ENTITY_ORDINAL(ActorTokenOperandRef, ActorRef)
LOOM_DECODE_ENTITY_ORDINAL(RootThreadValueInputTransferRef, RootThreadLaunchRef)
LOOM_DECODE_ENTITY_ORDINAL(ThreadChannelSendSiteRef, RootThreadLaunchRef)
LOOM_DECODE_ENTITY_ORDINAL(ThreadChannelReceiveSiteRef, RootThreadLaunchRef)
LOOM_DECODE_ENTITY_ORDINAL(LogicalMemoryViewRef, LogicalMemoryRootRef)

#undef LOOM_DECODE_ENTITY_ORDINAL

llvm::Expected<RootThreadStartTransferRef>
decodeReference(ReferenceReader &reader, TypeTag<RootThreadStartTransferRef>) {
  llvm::Expected<RootThreadLaunchRef> launch =
      decodeReference(reader, TypeTag<RootThreadLaunchRef>{});
  if (!launch)
    return launch.takeError();
  return RootThreadStartTransferRef{*launch};
}

llvm::Expected<RootThreadCompletionTransferRef>
decodeReference(ReferenceReader &reader,
                TypeTag<RootThreadCompletionTransferRef>) {
  llvm::Expected<RootThreadLaunchRef> launch =
      decodeReference(reader, TypeTag<RootThreadLaunchRef>{});
  if (!launch)
    return launch.takeError();
  return RootThreadCompletionTransferRef{*launch};
}

template <typename Ref>
llvm::Expected<Ref> decodeRootedOrdinal(ReferenceReader &reader) {
  llvm::Expected<RootedGraphLaunchRef> launch =
      decodeReference(reader, TypeTag<RootedGraphLaunchRef>{});
  if (!launch)
    return launch.takeError();
  llvm::Expected<std::uint64_t> ordinal = reader.getU64();
  if (!ordinal)
    return ordinal.takeError();
  return Ref{*launch, *ordinal};
}

llvm::Expected<GraphLaunchStartTransferRef>
decodeReference(ReferenceReader &reader, TypeTag<GraphLaunchStartTransferRef>) {
  llvm::Expected<RootedGraphLaunchRef> launch =
      decodeReference(reader, TypeTag<RootedGraphLaunchRef>{});
  if (!launch)
    return launch.takeError();
  return GraphLaunchStartTransferRef{*launch};
}

#define LOOM_DECODE_ROOTED_ORDINAL(Type)                                       \
  llvm::Expected<Type> decodeReference(ReferenceReader &reader,                \
                                       TypeTag<Type>) {                        \
    return decodeRootedOrdinal<Type>(reader);                                  \
  }

LOOM_DECODE_ROOTED_ORDINAL(GraphLaunchValueInputTransferRef)
LOOM_DECODE_ROOTED_ORDINAL(GraphLaunchValueResultTransferRef)
LOOM_DECODE_ROOTED_ORDINAL(GraphStreamOutputProducerRef)
LOOM_DECODE_ROOTED_ORDINAL(GraphStreamInputConsumerRef)

#undef LOOM_DECODE_ROOTED_ORDINAL

llvm::Expected<GraphLaunchDoneTransferRef>
decodeReference(ReferenceReader &reader, TypeTag<GraphLaunchDoneTransferRef>) {
  llvm::Expected<RootedGraphLaunchRef> launch =
      decodeReference(reader, TypeTag<RootedGraphLaunchRef>{});
  if (!launch)
    return launch.takeError();
  return GraphLaunchDoneTransferRef{*launch};
}

llvm::Expected<RootThreadBoundarySourceRef>
decodeReference(ReferenceReader &reader, TypeTag<RootThreadBoundarySourceRef>) {
  auto transfer =
      decodeReference(reader, TypeTag<RootThreadBoundaryTransferRef>{});
  if (!transfer)
    return transfer.takeError();
  return RootThreadBoundarySourceRef{*transfer};
}

llvm::Expected<GraphLaunchBoundarySourceRef>
decodeReference(ReferenceReader &reader,
                TypeTag<GraphLaunchBoundarySourceRef>) {
  auto transfer =
      decodeReference(reader, TypeTag<GraphLaunchBoundaryTransferRef>{});
  if (!transfer)
    return transfer.takeError();
  return GraphLaunchBoundarySourceRef{*transfer};
}

llvm::Expected<ChannelProducerTerminalRef>
decodeReference(ReferenceReader &reader, TypeTag<ChannelProducerTerminalRef>) {
  auto producer = decodeReference(reader, TypeTag<ChannelProducerRef>{});
  if (!producer)
    return producer.takeError();
  return ChannelProducerTerminalRef{*producer};
}

llvm::Expected<RootThreadBoundarySinkRef>
decodeReference(ReferenceReader &reader, TypeTag<RootThreadBoundarySinkRef>) {
  auto transfer =
      decodeReference(reader, TypeTag<RootThreadBoundaryTransferRef>{});
  if (!transfer)
    return transfer.takeError();
  return RootThreadBoundarySinkRef{*transfer};
}

llvm::Expected<GraphLaunchBoundarySinkRef>
decodeReference(ReferenceReader &reader, TypeTag<GraphLaunchBoundarySinkRef>) {
  auto transfer =
      decodeReference(reader, TypeTag<GraphLaunchBoundaryTransferRef>{});
  if (!transfer)
    return transfer.takeError();
  return GraphLaunchBoundarySinkRef{*transfer};
}

llvm::Expected<ChannelConsumerTerminalRef>
decodeReference(ReferenceReader &reader, TypeTag<ChannelConsumerTerminalRef>) {
  auto consumer = decodeReference(reader, TypeTag<ChannelConsumerRef>{});
  if (!consumer)
    return consumer.takeError();
  return ChannelConsumerTerminalRef{*consumer};
}

llvm::Expected<ContextualActorRef>
decodeReference(ReferenceReader &reader, TypeTag<ContextualActorRef>) {
  auto launch = decodeReference(reader, TypeTag<RootedGraphLaunchRef>{});
  if (!launch)
    return launch.takeError();
  auto actor = decodeReference(reader, TypeTag<ActorRef>{});
  if (!actor)
    return actor.takeError();
  return ContextualActorRef{*launch, *actor};
}

llvm::Expected<MemoryExposureRef> decodeReference(ReferenceReader &reader,
                                                  TypeTag<MemoryExposureRef>) {
  auto launch = decodeReference(reader, TypeTag<RootedGraphLaunchRef>{});
  if (!launch)
    return launch.takeError();
  auto ordinal = reader.getU64();
  if (!ordinal)
    return ordinal.takeError();
  return MemoryExposureRef{*launch, *ordinal};
}

llvm::Expected<FenceActorFamilyRef>
decodeReference(ReferenceReader &reader, TypeTag<FenceActorFamilyRef>) {
  auto actor = decodeReference(reader, TypeTag<ActorRef>{});
  if (!actor)
    return actor.takeError();
  return FenceActorFamilyRef{*actor};
}

llvm::Expected<MessageTransferMemberRef>
decodeReference(ReferenceReader &, TypeTag<MessageTransferMemberRef>) {
  return MessageTransferMemberRef{};
}

llvm::Expected<AddressedMemoryActorMemberRef>
decodeReference(ReferenceReader &reader,
                TypeTag<AddressedMemoryActorMemberRef>) {
  auto actor = decodeReference(reader, TypeTag<ContextualActorRef>{});
  if (!actor)
    return actor.takeError();
  return AddressedMemoryActorMemberRef{*actor};
}

llvm::Expected<FenceActorMemberRef>
decodeReference(ReferenceReader &reader, TypeTag<FenceActorMemberRef>) {
  auto actor = decodeReference(reader, TypeTag<ContextualActorRef>{});
  if (!actor)
    return actor.takeError();
  return FenceActorMemberRef{*actor};
}

llvm::Expected<ProducedTransferEventRef>
decodeReference(ReferenceReader &reader, TypeTag<ProducedTransferEventRef>) {
  auto terminal =
      decodeReference(reader, TypeTag<CanonicalProducerTerminalRef>{});
  if (!terminal)
    return terminal.takeError();
  return ProducedTransferEventRef{*terminal};
}

llvm::Expected<ConsumedTransferEventRef>
decodeReference(ReferenceReader &reader, TypeTag<ConsumedTransferEventRef>) {
  auto terminal = decodeReference(reader, TypeTag<CanonicalSinkTerminalRef>{});
  if (!terminal)
    return terminal.takeError();
  return ConsumedTransferEventRef{*terminal};
}

// clang-format off
#define LOOM_DECODE_CLOSED_UNION_ALTERNATIVE(Union, WireTag, Type)            \
  case WireTag: {                                                             \
    llvm::Expected<Type> ref = decodeReference(reader, TypeTag<Type>{});      \
    if (!ref)                                                                 \
      return ref.takeError();                                                 \
    return Union(std::in_place_type<Type>, std::move(*ref));                 \
  }
#define LOOM_DEFINE_CLOSED_UNION_DECODER(Union, Label, Alternatives)          \
  llvm::Expected<Union> decodeReference(ReferenceReader &reader,              \
                                        TypeTag<Union>) {                     \
    llvm::Expected<std::uint32_t> wireTag = reader.getU32();                  \
    if (!wireTag)                                                             \
      return wireTag.takeError();                                             \
    switch (*wireTag) {                                                       \
      Alternatives(LOOM_DECODE_CLOSED_UNION_ALTERNATIVE, Union)              \
    default:                                                                  \
      return makeDataflowReferenceError(                                     \
          DataflowReferenceErrorKind::MalformedSyntax,                       \
          llvm::Twine("unknown ") + Label + " discriminant " +              \
              llvm::Twine(*wireTag));                                        \
    }                                                                         \
  }
LOOM_DATAFLOW_STRUCTURAL_REFERENCE_UNIONS(LOOM_DEFINE_CLOSED_UNION_DECODER)
#undef LOOM_DEFINE_CLOSED_UNION_DECODER
#undef LOOM_DECODE_CLOSED_UNION_ALTERNATIVE
// clang-format on

template <typename Ref>
llvm::Expected<std::vector<std::uint8_t>>
encodeComplete(const Ref &reference,
               const ::loom::ArtifactIdentity *expectedArtifact) {
  ReferenceWriter writer(expectedArtifact);
  if (llvm::Error error = encodeReference(writer, reference))
    return std::move(error);
  return writer.take();
}

template <typename Ref>
llvm::Expected<Ref> decodeComplete(llvm::ArrayRef<std::uint8_t> bytes,
                                   const ::loom::ArtifactIdentity &artifact) {
  ReferenceReader reader(bytes, artifact);
  llvm::Expected<Ref> reference = decodeReference(reader, TypeTag<Ref>{});
  if (!reference)
    return reference.takeError();
  if (!reader.empty())
    return makeDataflowReferenceError(
        DataflowReferenceErrorKind::MalformedSyntax,
        "trailing Canonical Dataflow reference bytes");
  return std::move(*reference);
}

std::uint64_t slotOrdinal(const EventLogicalInputSlot &slot) {
  return std::visit([](const auto &typed) { return typed.ordinal; }, slot);
}

void appendU32(std::vector<std::uint8_t> &bytes, std::uint32_t value) {
  for (int shift = 24; shift >= 0; shift -= 8)
    bytes.push_back(static_cast<std::uint8_t>(value >> shift));
}

void appendU64(std::vector<std::uint8_t> &bytes, std::uint64_t value) {
  for (int shift = 56; shift >= 0; shift -= 8)
    bytes.push_back(static_cast<std::uint8_t>(value >> shift));
}

llvm::Expected<EventLogicalInputSlot>
decodeEventLogicalInputSlot(std::uint32_t wireTag, StructuralOrdinal ordinal) {
  switch (wireTag) {
    // clang-format off
#define LOOM_DECODE_EVENT_SLOT(Union, WireTag, Type)                          \
  case WireTag:                                                               \
    return EventLogicalInputSlot(std::in_place_type<Type>, Type{ordinal});
  LOOM_DATAFLOW_EVENT_LOGICAL_INPUT_SLOT_ALTERNATIVES(
      LOOM_DECODE_EVENT_SLOT, EventLogicalInputSlot)
#undef LOOM_DECODE_EVENT_SLOT
    // clang-format on
  default:
    return makeDataflowReferenceError(
        DataflowReferenceErrorKind::MalformedSyntax,
        llvm::Twine("unknown event logical slot discriminant ") +
            llvm::Twine(wireTag));
  }
}

#undef LOOM_DATAFLOW_CLOSED_UNIONS
#undef LOOM_DATAFLOW_STRUCTURAL_REFERENCE_UNIONS
#undef LOOM_DATAFLOW_EVENT_LOGICAL_INPUT_SLOT_ALTERNATIVES
#undef LOOM_DATAFLOW_STATIC_TRANSFER_EVENT_REF_ALTERNATIVES
#undef LOOM_DATAFLOW_SERVICE_MEMBER_REF_ALTERNATIVES
#undef LOOM_DATAFLOW_MEMORY_ROOT_OR_VIEW_REF_ALTERNATIVES
#undef LOOM_DATAFLOW_SINK_TERMINAL_REF_ALTERNATIVES
#undef LOOM_DATAFLOW_PRODUCER_TERMINAL_REF_ALTERNATIVES
#undef LOOM_DATAFLOW_CHANNEL_CONSUMER_REF_ALTERNATIVES
#undef LOOM_DATAFLOW_CHANNEL_PRODUCER_REF_ALTERNATIVES
#undef LOOM_DATAFLOW_GRAPH_LAUNCH_TRANSFER_REF_ALTERNATIVES
#undef LOOM_DATAFLOW_ROOT_THREAD_TRANSFER_REF_ALTERNATIVES
#undef LOOM_DATAFLOW_GRAPH_CONSUMER_ENDPOINT_REF_ALTERNATIVES
#undef LOOM_DATAFLOW_GRAPH_PRODUCER_ENDPOINT_REF_ALTERNATIVES
#undef LOOM_DATAFLOW_GRAPH_EGRESS_TOKEN_REF_ALTERNATIVES
#undef LOOM_DATAFLOW_GRAPH_INGRESS_TOKEN_REF_ALTERNATIVES

} // namespace

#define LOOM_DATAFLOW_REFERENCE_CODEC(Type)                                    \
  llvm::Expected<std::vector<std::uint8_t>>                                    \
  DataflowReferenceCodecTraits<Type>::encode(                                  \
      const Type &reference,                                                   \
      const ::loom::ArtifactIdentity *expectedArtifact) {                      \
    return encodeComplete(reference, expectedArtifact);                        \
  }                                                                            \
  llvm::Expected<Type> DataflowReferenceCodecTraits<Type>::decode(             \
      llvm::ArrayRef<std::uint8_t> bytes,                                      \
      const ::loom::ArtifactIdentity &artifact) {                              \
    return decodeComplete<Type>(bytes, artifact);                              \
  }
#include "Dataflow/IR/DataflowRefs.def"

bool eventLogicalInputSlotLess(const EventLogicalInputSlot &lhs,
                               const EventLogicalInputSlot &rhs) {
  const std::uint32_t lhsWireTag = closedUnionWireTag(lhs);
  const std::uint32_t rhsWireTag = closedUnionWireTag(rhs);
  if (lhsWireTag != rhsWireTag)
    return lhsWireTag < rhsWireTag;
  return slotOrdinal(lhs) < slotOrdinal(rhs);
}

llvm::Expected<::loom::CanonicalSemanticBytes>
encodeEventLogicalProjection(const EventLogicalProjection &projection) {
  for (std::size_t index = 1; index < projection.size(); ++index)
    if (!eventLogicalInputSlotLess(projection[index - 1], projection[index]))
      return makeDataflowReferenceError(
          DataflowReferenceErrorKind::Noncanonical,
          "event logical projection is not strictly sorted and unique");

  std::vector<std::uint8_t> bytes;
  if (projection.size() > (std::numeric_limits<std::size_t>::max() - 8) / 12)
    return makeDataflowReferenceError(
        DataflowReferenceErrorKind::MalformedSyntax,
        "event logical projection wire size overflows native memory");
  bytes.reserve(8 + projection.size() * 12);
  appendU64(bytes, static_cast<std::uint64_t>(projection.size()));
  for (const EventLogicalInputSlot &slot : projection) {
    appendU32(bytes, closedUnionWireTag(slot));
    appendU64(bytes, slotOrdinal(slot));
  }
  return ::loom::CanonicalSemanticBytes(std::move(bytes));
}

llvm::Expected<EventLogicalProjection>
decodeEventLogicalProjection(llvm::ArrayRef<std::uint8_t> bytes) {
  if (bytes.size() < 8)
    return makeDataflowReferenceError(
        DataflowReferenceErrorKind::MalformedSyntax,
        "truncated event logical projection count");
  std::uint64_t count = 0;
  for (unsigned index = 0; index < 8; ++index)
    count = (count << 8) | bytes[index];
  bytes = bytes.drop_front(8);
  if (count > std::numeric_limits<std::size_t>::max() ||
      count > bytes.size() / 12)
    return makeDataflowReferenceError(
        DataflowReferenceErrorKind::MalformedSyntax,
        "truncated event logical projection slots");
  if (bytes.size() != static_cast<std::size_t>(count) * 12)
    return makeDataflowReferenceError(
        DataflowReferenceErrorKind::MalformedSyntax,
        "trailing event logical projection bytes");

  EventLogicalProjection projection;
  projection.reserve(static_cast<std::size_t>(count));
  for (std::uint64_t index = 0; index < count; ++index) {
    std::uint32_t kind = 0;
    for (unsigned byte = 0; byte < 4; ++byte)
      kind = (kind << 8) | bytes[byte];
    bytes = bytes.drop_front(4);
    std::uint64_t ordinal = 0;
    for (unsigned byte = 0; byte < 8; ++byte)
      ordinal = (ordinal << 8) | bytes[byte];
    bytes = bytes.drop_front(8);
    llvm::Expected<EventLogicalInputSlot> slot =
        decodeEventLogicalInputSlot(kind, ordinal);
    if (!slot)
      return slot.takeError();
    projection.push_back(std::move(*slot));
    if (projection.size() > 1 &&
        !eventLogicalInputSlotLess(projection[projection.size() - 2],
                                   projection.back()))
      return makeDataflowReferenceError(
          DataflowReferenceErrorKind::Noncanonical,
          "event logical projection is not strictly sorted and unique");
  }
  return projection;
}

} // namespace dataflow
