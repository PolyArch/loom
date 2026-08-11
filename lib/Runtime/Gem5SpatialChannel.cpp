#include "Runtime/Gem5SpatialChannel.h"

#include "Common/ArtifactText.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <limits>
#include <string>
#include <system_error>
#include <tuple>

namespace loom::runtime {
namespace {

constexpr std::array<std::uint8_t, 8> kProjectionMagic{'L', 'G', 'C', 'P',
                                                       '0', '0', '0', '1'};

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "gem5_spatial_channel_invalid: " + message);
}

void appendU32(std::vector<std::uint8_t> &bytes, std::uint32_t value) {
  for (int shift = 24; shift >= 0; shift -= 8)
    bytes.push_back(static_cast<std::uint8_t>(value >> shift));
}

void appendU64(std::vector<std::uint8_t> &bytes, std::uint64_t value) {
  for (int shift = 56; shift >= 0; shift -= 8)
    bytes.push_back(static_cast<std::uint8_t>(value >> shift));
}

class Reader final {
public:
  explicit Reader(llvm::ArrayRef<std::uint8_t> bytes) : bytes_(bytes) {}

  llvm::Expected<std::uint32_t> u32() {
    if (bytes_.size() < 4)
      return invalid("truncated u32");
    std::uint32_t value = 0;
    for (unsigned index = 0; index != 4; ++index)
      value = (value << 8) | bytes_[index];
    bytes_ = bytes_.drop_front(4);
    return value;
  }

  llvm::Expected<std::uint64_t> u64() {
    if (bytes_.size() < 8)
      return invalid("truncated u64");
    std::uint64_t value = 0;
    for (unsigned index = 0; index != 8; ++index)
      value = (value << 8) | bytes_[index];
    bytes_ = bytes_.drop_front(8);
    return value;
  }

  llvm::Expected<std::string> string() {
    auto size = u64();
    if (!size)
      return size.takeError();
    if (*size > bytes_.size())
      return invalid("truncated string");
    std::string value(reinterpret_cast<const char *>(bytes_.data()),
                      static_cast<std::size_t>(*size));
    bytes_ = bytes_.drop_front(static_cast<std::size_t>(*size));
    return value;
  }

  llvm::Expected<ArtifactRootReference> root() {
    auto schema = string();
    if (!schema)
      return schema.takeError();
    auto major = u32();
    if (!major)
      return major.takeError();
    auto minor = u32();
    if (!minor)
      return minor.takeError();
    if (bytes_.size() < ArtifactIdentity::byteSize)
      return invalid("truncated artifact identity");
    ArtifactIdentity::Storage storage{};
    std::copy_n(bytes_.begin(), ArtifactIdentity::byteSize, storage.begin());
    bytes_ = bytes_.drop_front(ArtifactIdentity::byteSize);
    auto identity = ArtifactIdentity::fromBytes(storage);
    if (!identity)
      return identity.takeError();
    return ArtifactRootReference{
        std::move(*schema), {*major, *minor}, std::move(*identity)};
  }

  bool empty() const { return bytes_.empty(); }

private:
  llvm::ArrayRef<std::uint8_t> bytes_;
};

void appendString(std::vector<std::uint8_t> &bytes, llvm::StringRef value) {
  appendU64(bytes, value.size());
  bytes.insert(bytes.end(), value.bytes_begin(), value.bytes_end());
}

void appendRoot(std::vector<std::uint8_t> &bytes,
                const ArtifactRootReference &root) {
  appendString(bytes, root.schemaIdentity);
  appendU32(bytes, root.schemaVersion.major);
  appendU32(bytes, root.schemaVersion.minor);
  const auto identity = root.artifact.bytes();
  bytes.insert(bytes.end(), identity.begin(), identity.end());
}

llvm::Error validateRange(std::uint64_t address, std::uint64_t capacity) {
  if (capacity <= gem5SpatialChannelBufferHeaderBytes)
    return invalid("channel buffer cannot hold a payload");
  if (address > std::numeric_limits<std::uint64_t>::max() - capacity)
    return invalid("channel buffer range overflows u64");
  return llvm::Error::success();
}

llvm::Error canonicalize(Gem5SpatialChannelProjection &projection) {
  llvm::sort(projection.inputs, [](const auto &lhs, const auto &rhs) {
    return std::tie(lhs.consumerStreamInputOrdinal,
                    lhs.producerStreamOutputOrdinal, lhs.address) <
           std::tie(rhs.consumerStreamInputOrdinal,
                    rhs.producerStreamOutputOrdinal, rhs.address);
  });
  llvm::sort(projection.outputs, [](const auto &lhs, const auto &rhs) {
    return std::tie(lhs.producerStreamOutputOrdinal, lhs.address) <
           std::tie(rhs.producerStreamOutputOrdinal, rhs.address);
  });
  for (const Gem5SpatialChannelInput &input : projection.inputs) {
    if (input.producerWorkload.schemaIdentity.empty() ||
        input.producerRuntimeInput.schemaIdentity.empty())
      return invalid("channel input has an empty producer root schema");
    if (llvm::Error error = validateRange(input.address, input.capacityBytes))
      return error;
  }
  for (const Gem5SpatialChannelOutput &output : projection.outputs)
    if (llvm::Error error = validateRange(output.address, output.capacityBytes))
      return error;
  if (std::adjacent_find(projection.inputs.begin(), projection.inputs.end(),
                         [](const auto &lhs, const auto &rhs) {
                           return lhs.consumerStreamInputOrdinal ==
                                  rhs.consumerStreamInputOrdinal;
                         }) != projection.inputs.end())
    return invalid("consumer stream input is bound more than once");
  if (std::adjacent_find(projection.outputs.begin(), projection.outputs.end(),
                         [](const auto &lhs, const auto &rhs) {
                           return lhs.producerStreamOutputOrdinal ==
                                  rhs.producerStreamOutputOrdinal;
                         }) != projection.outputs.end())
    return invalid("producer stream output is bound more than once");
  return llvm::Error::success();
}

std::vector<std::uint8_t>
encodeCanonical(const Gem5SpatialChannelProjection &projection) {
  std::vector<std::uint8_t> bytes(kProjectionMagic.begin(),
                                  kProjectionMagic.end());
  appendU64(bytes, projection.inputs.size());
  for (const Gem5SpatialChannelInput &input : projection.inputs) {
    appendRoot(bytes, input.producerWorkload);
    appendRoot(bytes, input.producerRuntimeInput);
    appendU64(bytes, input.producerStreamOutputOrdinal);
    appendU64(bytes, input.consumerStreamInputOrdinal);
    appendU64(bytes, input.address);
    appendU64(bytes, input.capacityBytes);
  }
  appendU64(bytes, projection.outputs.size());
  for (const Gem5SpatialChannelOutput &output : projection.outputs) {
    appendU64(bytes, output.producerStreamOutputOrdinal);
    appendU64(bytes, output.address);
    appendU64(bytes, output.capacityBytes);
  }
  return bytes;
}

} // namespace

llvm::Expected<std::vector<std::uint8_t>>
encodeGem5SpatialChannelProjection(Gem5SpatialChannelProjection projection) {
  if (llvm::Error error = canonicalize(projection))
    return std::move(error);
  return encodeCanonical(projection);
}

llvm::Expected<Gem5SpatialChannelProjection>
decodeGem5SpatialChannelProjection(llvm::ArrayRef<std::uint8_t> bytes) {
  if (bytes.size() < kProjectionMagic.size() ||
      !std::equal(kProjectionMagic.begin(), kProjectionMagic.end(),
                  bytes.begin()))
    return invalid("wrong or truncated projection identity");
  Reader reader(bytes.drop_front(kProjectionMagic.size()));
  auto inputCount = reader.u64();
  if (!inputCount || *inputCount > bytes.size())
    return inputCount ? invalid("input count exceeds the projection size")
                      : inputCount.takeError();
  Gem5SpatialChannelProjection projection;
  projection.inputs.reserve(static_cast<std::size_t>(*inputCount));
  for (std::uint64_t index = 0; index != *inputCount; ++index) {
    auto workload = reader.root();
    if (!workload)
      return workload.takeError();
    auto runtime = reader.root();
    if (!runtime)
      return runtime.takeError();
    auto producer = reader.u64();
    if (!producer)
      return producer.takeError();
    auto consumer = reader.u64();
    if (!consumer)
      return consumer.takeError();
    auto address = reader.u64();
    if (!address)
      return address.takeError();
    auto capacity = reader.u64();
    if (!capacity)
      return capacity.takeError();
    projection.inputs.push_back({std::move(*workload), std::move(*runtime),
                                 *producer, *consumer, *address, *capacity});
  }
  auto outputCount = reader.u64();
  if (!outputCount || *outputCount > bytes.size())
    return outputCount ? invalid("output count exceeds the projection size")
                       : outputCount.takeError();
  projection.outputs.reserve(static_cast<std::size_t>(*outputCount));
  for (std::uint64_t index = 0; index != *outputCount; ++index) {
    auto producer = reader.u64();
    if (!producer)
      return producer.takeError();
    auto address = reader.u64();
    if (!address)
      return address.takeError();
    auto capacity = reader.u64();
    if (!capacity)
      return capacity.takeError();
    projection.outputs.push_back({*producer, *address, *capacity});
  }
  if (!reader.empty())
    return invalid("projection has trailing bytes");
  const std::vector<std::uint8_t> original(bytes.begin(), bytes.end());
  if (llvm::Error error = canonicalize(projection))
    return std::move(error);
  if (encodeCanonical(projection) != original)
    return invalid("projection bytes are not canonical");
  return projection;
}

std::array<std::uint8_t, gem5SpatialChannelBufferHeaderBytes>
encodeGem5SpatialChannelBufferHeader(std::uint64_t payloadBytes) {
  return encodeGem5SpatialChannelBufferHeaderPortable(payloadBytes);
}

llvm::Expected<std::uint64_t>
decodeGem5SpatialChannelBufferHeader(llvm::ArrayRef<std::uint8_t> bytes) {
  std::uint64_t payloadBytes = 0;
  std::string diagnostic;
  if (!decodeGem5SpatialChannelBufferHeaderPortable(bytes.data(), bytes.size(),
                                                    payloadBytes, diagnostic))
    return invalid(diagnostic);
  return payloadBytes;
}

} // namespace loom::runtime
