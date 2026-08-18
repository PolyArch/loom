#include "Simulator/SpatialChannelWire.h"

#include "SimulationWireInternal.h"

#include <algorithm>
#include <array>
#include <system_error>

namespace loom::sim {
namespace {

constexpr std::array<std::uint8_t, 8> kSpatialChannelStreamMagic{
    'L', 'S', 'C', 'H', '0', '0', '0', '1'};

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "spatial_channel_wire_invalid: " + message);
}

llvm::Expected<detail::LaneShape> streamInputShape(
    const dataflow::CanonicalDataflowProgramView &dataflow,
    dataflow::RootedGraphLaunchRef launch, std::uint64_t ordinal) {
  auto context = detail::resolveLaunchContext(dataflow, launch);
  if (!context)
    return context.takeError();
  if (ordinal >= context->streamInputShapes.size())
    return invalid("stream input ordinal exceeds the graph boundary");
  return context->streamInputShapes[ordinal];
}

llvm::Expected<detail::LaneShape> streamOutputShape(
    const dataflow::CanonicalDataflowProgramView &dataflow,
    dataflow::RootedGraphLaunchRef launch, std::uint64_t ordinal) {
  auto context = detail::resolveLaunchContext(dataflow, launch);
  if (!context)
    return context.takeError();
  if (ordinal >= context->streamOutputShapes.size())
    return invalid("stream output ordinal exceeds the graph boundary");
  return context->streamOutputShapes[ordinal];
}

std::vector<std::uint8_t> encodeCanonical(
    const CanonicalStreamSequence &stream, const detail::LaneShape &shape) {
  detail::WireWriter writer;
  writer.bytes(kSpatialChannelStreamMagic);
  detail::encodeStreamSequence(writer, stream, shape);
  return writer.take();
}

} // namespace

llvm::Expected<std::vector<std::uint8_t>> encodeSpatialChannelStream(
    const CanonicalStreamSequence &stream,
    const dataflow::CanonicalDataflowProgramView &dataflow,
    dataflow::RootedGraphLaunchRef launch,
    std::uint64_t streamOutputOrdinal, std::uint64_t memoryObjectCount) {
  auto shape = streamOutputShape(dataflow, launch, streamOutputOrdinal);
  if (!shape)
    return shape.takeError();
  if (llvm::Error error = detail::validateValueSequence(
          stream.values, *shape, "Spatial channel stream", memoryObjectCount))
    return std::move(error);
  if (static_cast<std::uint32_t>(stream.termination) >
      static_cast<std::uint32_t>(StreamTermination::OpenAfterLast))
    return invalid("stream termination is outside its typed domain");
  return encodeCanonical(stream, *shape);
}

llvm::Expected<CanonicalStreamSequence> decodeSpatialChannelStream(
    llvm::ArrayRef<std::uint8_t> bytes,
    const dataflow::CanonicalDataflowProgramView &dataflow,
    dataflow::RootedGraphLaunchRef launch,
    std::uint64_t streamInputOrdinal, std::uint64_t memoryObjectCount) {
  if (bytes.size() < kSpatialChannelStreamMagic.size() ||
      !std::equal(kSpatialChannelStreamMagic.begin(),
                  kSpatialChannelStreamMagic.end(), bytes.begin()))
    return invalid("stream has the wrong or truncated identity");
  auto shape = streamInputShape(dataflow, launch, streamInputOrdinal);
  if (!shape)
    return shape.takeError();
  detail::WireReader reader(bytes.drop_front(kSpatialChannelStreamMagic.size()));
  auto stream = detail::decodeStreamSequence(reader, *shape);
  if (!stream)
    return stream.takeError();
  if (!reader.atEnd())
    return invalid("stream has trailing bytes");
  if (llvm::Error error = detail::validateValueSequence(
          stream->values, *shape, "Spatial channel stream", memoryObjectCount))
    return std::move(error);
  if (encodeCanonical(*stream, *shape) !=
      std::vector<std::uint8_t>(bytes.begin(), bytes.end()))
    return invalid("stream bytes are not canonical");
  return stream;
}

} // namespace loom::sim
