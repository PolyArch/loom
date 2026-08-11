#ifndef LOOM_RUNTIME_GEM5SPATIALCHANNELPLAN_H
#define LOOM_RUNTIME_GEM5SPATIALCHANNELPLAN_H

#include <algorithm>
#include <array>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace loom::runtime {

inline constexpr std::array<std::uint8_t, 4> gem5SpatialChannelBufferMagic{
    'L', 'G', 'C', '1'};
inline constexpr std::size_t gem5SpatialChannelBufferHeaderBytes = 16;

struct Gem5SpatialChannelEngineInput final {
  std::uint64_t consumerStreamInputOrdinal = 0;
  std::uint64_t producerObservationOrdinal = 0;
  std::uint32_t tokenBitWidth = 0;
  std::uint64_t address = 0;
  std::uint64_t capacityBytes = 0;
};

struct Gem5SpatialChannelEngineOutput final {
  std::uint64_t address = 0;
  std::uint64_t capacityBytes = 0;
};

struct Gem5SpatialChannelEnginePlan final {
  std::vector<Gem5SpatialChannelEngineInput> inputs;
  std::vector<Gem5SpatialChannelEngineOutput> outputs;
};

inline std::array<std::uint8_t, gem5SpatialChannelBufferHeaderBytes>
encodeGem5SpatialChannelBufferHeaderPortable(std::uint64_t payloadBytes) {
  std::array<std::uint8_t, gem5SpatialChannelBufferHeaderBytes> bytes{};
  std::copy(gem5SpatialChannelBufferMagic.begin(),
            gem5SpatialChannelBufferMagic.end(), bytes.begin());
  bytes[7] = 1;
  for (unsigned index = 0; index != 8; ++index)
    bytes[8 + index] =
        static_cast<std::uint8_t>(payloadBytes >> ((7 - index) * 8));
  return bytes;
}

inline bool decodeGem5SpatialChannelBufferHeaderPortable(
    const std::uint8_t *bytes, std::size_t size, std::uint64_t &payloadBytes,
    std::string &error) {
  if (size != gem5SpatialChannelBufferHeaderBytes) {
    error = "channel buffer header has the wrong size";
    return false;
  }
  if (!std::equal(gem5SpatialChannelBufferMagic.begin(),
                  gem5SpatialChannelBufferMagic.end(), bytes) ||
      bytes[4] != 0 || bytes[5] != 0 || bytes[6] != 0 || bytes[7] != 1) {
    error = "channel buffer header has the wrong identity";
    return false;
  }
  payloadBytes = 0;
  for (unsigned index = 0; index != 8; ++index)
    payloadBytes = (payloadBytes << 8) | bytes[8 + index];
  if (payloadBytes == 0) {
    error = "channel buffer payload is empty";
    return false;
  }
  return true;
}

inline std::string
encodeGem5SpatialChannelEnginePlan(Gem5SpatialChannelEnginePlan plan) {
  std::sort(plan.inputs.begin(), plan.inputs.end(),
            [](const auto &lhs, const auto &rhs) {
              return lhs.consumerStreamInputOrdinal <
                     rhs.consumerStreamInputOrdinal;
            });
  std::sort(plan.outputs.begin(), plan.outputs.end(),
            [](const auto &lhs, const auto &rhs) {
              if (lhs.address != rhs.address)
                return lhs.address < rhs.address;
              return lhs.capacityBytes < rhs.capacityBytes;
            });
  std::ostringstream output;
  output << "loom.gem5_spatial_channel_plan 1.0\ninputs " << plan.inputs.size()
         << '\n';
  for (const Gem5SpatialChannelEngineInput &input : plan.inputs)
    output << "input " << input.consumerStreamInputOrdinal << ' '
           << input.producerObservationOrdinal << ' ' << input.tokenBitWidth
           << ' ' << input.address << ' ' << input.capacityBytes << '\n';
  output << "outputs " << plan.outputs.size() << '\n';
  for (const Gem5SpatialChannelEngineOutput &entry : plan.outputs)
    output << "output " << entry.address << ' ' << entry.capacityBytes << '\n';
  output << "end\n";
  return output.str();
}

namespace detail {

inline bool parseChannelPlanUnsigned(std::string_view text,
                                     std::uint64_t &value) {
  if (text.empty())
    return false;
  const char *begin = text.data();
  const char *end = begin + text.size();
  const auto result = std::from_chars(begin, end, value);
  return result.ec == std::errc() && result.ptr == end;
}

inline std::vector<std::string_view>
splitChannelPlanLine(std::string_view line) {
  std::vector<std::string_view> fields;
  while (!line.empty()) {
    const std::size_t separator = line.find(' ');
    fields.push_back(line.substr(0, separator));
    if (separator == std::string_view::npos)
      break;
    line.remove_prefix(separator + 1);
    if (line.empty() || line.front() == ' ')
      return {};
  }
  return fields;
}

inline bool nextChannelPlanLine(std::string_view &text,
                                std::string_view &line) {
  if (text.empty())
    return false;
  const std::size_t newline = text.find('\n');
  if (newline == std::string_view::npos)
    return false;
  line = text.substr(0, newline);
  text.remove_prefix(newline + 1);
  return !line.empty();
}

} // namespace detail

inline bool
decodeGem5SpatialChannelEnginePlan(std::string_view text,
                                   Gem5SpatialChannelEnginePlan &plan,
                                   std::string &error) {
  const std::string original(text);
  std::string_view line;
  if (!detail::nextChannelPlanLine(text, line) ||
      line != "loom.gem5_spatial_channel_plan 1.0") {
    error = "channel plan has the wrong identity";
    return false;
  }
  if (!detail::nextChannelPlanLine(text, line)) {
    error = "channel plan omits its input count";
    return false;
  }
  auto fields = detail::splitChannelPlanLine(line);
  std::uint64_t inputCount = 0;
  if (fields.size() != 2 || fields[0] != "inputs" ||
      !detail::parseChannelPlanUnsigned(fields[1], inputCount) ||
      inputCount > original.size()) {
    error = "channel plan input count is invalid";
    return false;
  }
  Gem5SpatialChannelEnginePlan candidate;
  candidate.inputs.reserve(static_cast<std::size_t>(inputCount));
  for (std::uint64_t ordinal = 0; ordinal != inputCount; ++ordinal) {
    if (!detail::nextChannelPlanLine(text, line)) {
      error = "channel plan input table is truncated";
      return false;
    }
    fields = detail::splitChannelPlanLine(line);
    std::uint64_t consumer = 0;
    std::uint64_t observation = 0;
    std::uint64_t width = 0;
    std::uint64_t address = 0;
    std::uint64_t capacity = 0;
    if (fields.size() != 6 || fields[0] != "input" ||
        !detail::parseChannelPlanUnsigned(fields[1], consumer) ||
        !detail::parseChannelPlanUnsigned(fields[2], observation) ||
        !detail::parseChannelPlanUnsigned(fields[3], width) || width == 0 ||
        width > UINT32_MAX ||
        !detail::parseChannelPlanUnsigned(fields[4], address) ||
        !detail::parseChannelPlanUnsigned(fields[5], capacity) ||
        capacity <= gem5SpatialChannelBufferHeaderBytes ||
        address > UINT64_MAX - capacity) {
      error = "channel plan input is invalid";
      return false;
    }
    candidate.inputs.push_back({consumer, observation,
                                static_cast<std::uint32_t>(width), address,
                                capacity});
  }
  if (!detail::nextChannelPlanLine(text, line)) {
    error = "channel plan omits its output count";
    return false;
  }
  fields = detail::splitChannelPlanLine(line);
  std::uint64_t outputCount = 0;
  if (fields.size() != 2 || fields[0] != "outputs" ||
      !detail::parseChannelPlanUnsigned(fields[1], outputCount) ||
      outputCount > original.size()) {
    error = "channel plan output count is invalid";
    return false;
  }
  candidate.outputs.reserve(static_cast<std::size_t>(outputCount));
  for (std::uint64_t ordinal = 0; ordinal != outputCount; ++ordinal) {
    if (!detail::nextChannelPlanLine(text, line)) {
      error = "channel plan output table is truncated";
      return false;
    }
    fields = detail::splitChannelPlanLine(line);
    std::uint64_t address = 0;
    std::uint64_t capacity = 0;
    if (fields.size() != 3 || fields[0] != "output" ||
        !detail::parseChannelPlanUnsigned(fields[1], address) ||
        !detail::parseChannelPlanUnsigned(fields[2], capacity) ||
        capacity <= gem5SpatialChannelBufferHeaderBytes ||
        address > UINT64_MAX - capacity) {
      error = "channel plan output is invalid";
      return false;
    }
    candidate.outputs.push_back({address, capacity});
  }
  if (!detail::nextChannelPlanLine(text, line) || line != "end" ||
      !text.empty() ||
      encodeGem5SpatialChannelEnginePlan(candidate) != original) {
    error = "channel plan is not canonical";
    return false;
  }
  plan = std::move(candidate);
  return true;
}

} // namespace loom::runtime

#endif // LOOM_RUNTIME_GEM5SPATIALCHANNELPLAN_H
