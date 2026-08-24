#include "Runtime/Gem5SpatialChannel.h"
#include "Runtime/Gem5SpatialChannelPlan.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

using namespace loom;
using namespace loom::runtime;

namespace {

[[noreturn]] void fail(const std::string &message) {
  llvm::errs() << "Gem5SpatialChannelTest: " << message << '\n';
  std::exit(EXIT_FAILURE);
}

void require(bool condition, const std::string &message) {
  if (!condition)
    fail(message);
}

template <typename T> T take(llvm::Expected<T> value) {
  if (!value)
    fail(llvm::toString(value.takeError()));
  return std::move(*value);
}

void strictProjectionRoundTrips() {
  Gem5SpatialChannelProjection projection{{{4, 1, 5, 3}, {2, 0, 3, 2}},
                                          {{7, 9, 3}, {3, 7, 2}}};
  const std::vector<std::uint8_t> encoded =
      take(encodeGem5SpatialChannelProjection(projection));
  Gem5SpatialChannelProjection decoded =
      take(decodeGem5SpatialChannelProjection(encoded));
  require(decoded.inputs.size() == 2 && decoded.outputs.size() == 2,
          "strict projection lost an entry");
  require(decoded.inputs[0].consumerStreamInputOrdinal == 0 &&
              decoded.inputs[1].consumerStreamInputOrdinal == 1 &&
              decoded.inputs[0].channelOrdinal == 3 &&
              decoded.inputs[0].capacityMessages == 2 &&
              decoded.outputs[0].producerStreamOutputOrdinal == 3,
          "strict projection is not canonically ordered");
  require(take(encodeGem5SpatialChannelProjection(decoded)) == encoded,
          "strict projection roundtrip changed bytes");

  projection.inputs[1].consumerStreamInputOrdinal = 1;
  auto duplicate = encodeGem5SpatialChannelProjection(std::move(projection));
  require(!duplicate, "duplicate consumer binding was accepted");
  llvm::consumeError(duplicate.takeError());

  decoded.outputs[0].capacityMessages = 0;
  auto emptyCapacity = encodeGem5SpatialChannelProjection(std::move(decoded));
  require(!emptyCapacity, "zero message capacity was accepted");
  llvm::consumeError(emptyCapacity.takeError());

  std::vector<std::uint8_t> trailing = encoded;
  trailing.push_back(0);
  auto noncanonical = decodeGem5SpatialChannelProjection(trailing);
  require(!noncanonical, "strict projection accepted trailing bytes");
  llvm::consumeError(noncanonical.takeError());
}

void portablePlanRoundTrips() {
  Gem5SpatialChannelEnginePlan plan{
      {{1, 3, 64, 0x5000, 8192}, {0, 2, 32, 0x3000, 4096}},
      {{0x9000, 4096}, {0x7000, 4096}}};
  const std::string encoded = encodeGem5SpatialChannelEnginePlan(plan);
  Gem5SpatialChannelEnginePlan decoded;
  std::string diagnostic;
  require(decodeGem5SpatialChannelEnginePlan(encoded, decoded, diagnostic),
          diagnostic);
  require(decoded.inputs.size() == 2 && decoded.outputs.size() == 2 &&
              decoded.inputs[0].consumerStreamInputOrdinal == 0 &&
              decoded.outputs[0].address == 0x7000,
          "portable plan is not canonically ordered");
  require(encodeGem5SpatialChannelEnginePlan(decoded) == encoded,
          "portable plan roundtrip changed text");

  std::string reordered = encoded;
  const std::size_t first = reordered.find("input 0");
  const std::size_t second = reordered.find("input 1");
  require(first != std::string::npos && second != std::string::npos,
          "portable plan fixture is incomplete");
  std::swap(reordered[first + 6], reordered[second + 6]);
  require(!decodeGem5SpatialChannelEnginePlan(reordered, decoded, diagnostic),
          "portable plan accepted noncanonical input order");
}

void bufferHeaderRoundTrips() {
  constexpr std::uint64_t payloadBytes = 12345;
  const auto header =
      encodeGem5SpatialChannelBufferHeaderPortable(payloadBytes);
  std::uint64_t decoded = 0;
  std::string diagnostic;
  require(decodeGem5SpatialChannelBufferHeaderPortable(
              header.data(), header.size(), decoded, diagnostic) &&
              decoded == payloadBytes,
          diagnostic);
  require(take(decodeGem5SpatialChannelBufferHeader(header)) == payloadBytes,
          "LLVM wrapper disagrees with the portable header codec");
  auto corrupt = header;
  corrupt[4] = 1;
  require(!decodeGem5SpatialChannelBufferHeaderPortable(
              corrupt.data(), corrupt.size(), decoded, diagnostic),
          "buffer header accepted reserved bits");
}

} // namespace

int main() {
  strictProjectionRoundTrips();
  portablePlanRoundTrips();
  bufferHeaderRoundTrips();
  return EXIT_SUCCESS;
}
