#include "EDA/Adapters/OpenSource/MappedRtlSimulation.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <limits>
#include <sstream>
#include <string>
#include <system_error>

namespace loom::eda::open_source {
namespace {

llvm::Error invalid(const llvm::Twine &detail) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "mapped_rtl_result_invalid: " + detail);
}

llvm::StringRef terminalStatusSpellingImpl(MappedRtlTerminalStatus status) {
  switch (status) {
  case MappedRtlTerminalStatus::Retired:
    return "retired";
  case MappedRtlTerminalStatus::StoppedByLimit:
    return "stopped";
  }
  llvm_unreachable("closed mapped RTL terminal status");
}

llvm::Expected<MappedRtlTerminalStatus>
parseTerminalStatus(llvm::StringRef spelling) {
  for (MappedRtlTerminalStatus status : {MappedRtlTerminalStatus::Retired,
                                         MappedRtlTerminalStatus::StoppedByLimit})
    if (spelling == mappedRtlTerminalStatusSpelling(status))
      return status;
  return invalid("terminal status is unknown");
}

llvm::StringRef
streamTerminationSpellingImpl(sim::StreamTermination termination) {
  switch (termination) {
  case sim::StreamTermination::ClosedAfterLast:
    return "closed";
  case sim::StreamTermination::OpenAfterLast:
    return "open";
  }
  llvm_unreachable("closed stream termination domain");
}

llvm::Expected<sim::StreamTermination>
parseStreamTermination(llvm::StringRef spelling) {
  for (sim::StreamTermination termination : {
           sim::StreamTermination::ClosedAfterLast,
           sim::StreamTermination::OpenAfterLast})
    if (spelling == mappedRtlStreamTerminationSpelling(termination))
      return termination;
  return invalid("stream termination is unknown");
}

llvm::Expected<std::uint64_t> parseUnsigned(llvm::StringRef token,
                                            llvm::StringRef field) {
  std::uint64_t value = 0;
  if (token.empty() || (token.size() > 1 && token.front() == '0') ||
      token.getAsInteger(10, value))
    return invalid(field + " is not a canonical unsigned integer");
  return value;
}

llvm::Expected<std::uint32_t> parseWidth(llvm::StringRef token,
                                         llvm::StringRef field) {
  auto value = parseUnsigned(token, field);
  if (!value)
    return value.takeError();
  if (*value == 0 || *value > std::numeric_limits<std::uint32_t>::max())
    return invalid(field + " is outside the supported bit-width domain");
  return static_cast<std::uint32_t>(*value);
}

llvm::Expected<llvm::APInt> parseBits(llvm::StringRef token,
                                      std::uint32_t width) {
  if (!token.consume_front("b") || token.size() != width)
    return invalid("bit token does not have its declared width");
  for (char character : token)
    if (character != '0' && character != '1')
      return invalid("bit token contains an unknown HDL state");
  return llvm::APInt(width, token, 2);
}

std::string renderBits(const llvm::APInt &value) {
  llvm::SmallString<128> digits;
  value.toString(digits, 2, false);
  std::string result("b");
  result.append(value.getBitWidth() - digits.size(), '0');
  result.append(digits.begin(), digits.end());
  return result;
}

llvm::Error validateResult(const MappedRtlSimulationResult &result) {
  if (result.terminal == MappedRtlTerminalStatus::Retired) {
    if (!result.retirementCycle)
      return invalid("Retired result omits retirement_cycle");
    if (result.launchCycle > *result.retirementCycle ||
        *result.retirementCycle > result.terminalCycle)
      return invalid("Retired progress coordinates are not ordered");
  } else if (result.retirementCycle) {
    return invalid("StoppedByLimit result carries retirement_cycle");
  } else if (result.launchCycle > result.terminalCycle) {
    return invalid("terminal_cycle precedes launch_cycle");
  }
  for (const MappedRtlValueObservation &value : result.valueResults)
    if (value.token && value.token->getBitWidth() == 0)
      return invalid("value result token has zero width");
  for (const MappedRtlStreamObservation &stream : result.streamOutputs) {
    if (stream.tokenBitWidth == 0)
      return invalid("stream output has zero token width");
    for (const llvm::APInt &token : stream.tokens)
      if (token.getBitWidth() != stream.tokenBitWidth)
        return invalid("stream output token width is inconsistent");
  }
  for (const MappedRtlMemoryObservation &memory : result.memories)
    if (memory.bytes.empty())
      return invalid("memory observation is empty");
  return llvm::Error::success();
}

class TokenReader final {
public:
  explicit TokenReader(llvm::StringRef contents) : stream_(contents.str()) {}

  llvm::Expected<std::string> take(llvm::StringRef field) {
    std::string token;
    if (!(stream_ >> token))
      return invalid("result ended before " + field);
    return token;
  }

  llvm::Error expect(llvm::StringRef expected) {
    auto token = take(expected);
    if (!token)
      return token.takeError();
    if (*token != expected)
      return invalid("expected '" + expected + "' but found '" + *token + "'");
    return llvm::Error::success();
  }

  bool finished() {
    std::string trailing;
    return !(stream_ >> trailing);
  }

private:
  std::istringstream stream_;
};

llvm::Expected<std::uint64_t> takeUnsigned(TokenReader &reader,
                                           llvm::StringRef field) {
  auto token = reader.take(field);
  if (!token)
    return token.takeError();
  return parseUnsigned(*token, field);
}

llvm::Expected<std::uint32_t> takeWidth(TokenReader &reader,
                                        llvm::StringRef field) {
  auto token = reader.take(field);
  if (!token)
    return token.takeError();
  return parseWidth(*token, field);
}

llvm::Expected<std::uint8_t> parseHexByte(llvm::StringRef token) {
  unsigned value = 0;
  if (token.size() != 2 || token.getAsInteger(16, value) || value > 0xff)
    return invalid("defined memory byte is not two hexadecimal digits");
  return static_cast<std::uint8_t>(value);
}

} // namespace

llvm::StringRef
mappedRtlTerminalStatusSpelling(MappedRtlTerminalStatus status) {
  return terminalStatusSpellingImpl(status);
}

llvm::StringRef
mappedRtlStreamTerminationSpelling(sim::StreamTermination termination) {
  return streamTerminationSpellingImpl(termination);
}

llvm::Expected<std::string>
renderMappedRtlSimulationResult(const MappedRtlSimulationResult &result) {
  if (llvm::Error error = validateResult(result))
    return std::move(error);
  std::string text;
  llvm::raw_string_ostream output(text);
  output << mappedRtlResultSchema << " " << mappedRtlResultVersion
         << "\nterminal " << mappedRtlTerminalStatusSpelling(result.terminal)
         << "\nlaunch_cycle " << result.launchCycle << "\n";
  if (result.retirementCycle)
    output << "retirement_cycle " << *result.retirementCycle << "\n";
  else
    output << "retirement_cycle absent\n";
  output << "terminal_cycle " << result.terminalCycle << "\nvalue_results "
         << result.valueResults.size() << "\n";
  for (const auto &[ordinal, value] : llvm::enumerate(result.valueResults)) {
    output << "value " << ordinal << " ";
    if (!value.token)
      output << "absent\n";
    else
      output << value.token->getBitWidth() << " " << renderBits(*value.token)
             << "\n";
  }
  output << "stream_outputs " << result.streamOutputs.size() << "\n";
  for (const auto &[ordinal, stream] : llvm::enumerate(result.streamOutputs)) {
    output << "stream " << ordinal << " "
           << mappedRtlStreamTerminationSpelling(stream.termination) << " "
           << stream.tokenBitWidth << " " << stream.tokens.size();
    for (const llvm::APInt &token : stream.tokens)
      output << " " << renderBits(token);
    output << "\n";
  }
  output << "memories " << result.memories.size() << "\n";
  for (const auto &[ordinal, memory] : llvm::enumerate(result.memories)) {
    output << "memory " << ordinal << " " << memory.bytes.size();
    for (const sim::SemanticMemoryByte &byte : memory.bytes) {
      output << " ";
      switch (byte.state) {
      case sim::SemanticState::Defined:
        // Lowercase, as SystemVerilog's %h renders it: the format letters of
        // $fwrite are case-insensitive, so the harness cannot emit uppercase.
        output << "d" << llvm::format_hex_no_prefix(byte.value, 2);
        break;
      case sim::SemanticState::Undef:
        output << "u";
        break;
      case sim::SemanticState::Poison:
        output << "p";
        break;
      }
    }
    output << "\n";
  }
  output << "end\n";
  return text;
}

llvm::Expected<MappedRtlSimulationResult>
parseMappedRtlSimulationResult(llvm::StringRef contents) {
  TokenReader reader(contents);
  if (llvm::Error error = reader.expect(mappedRtlResultSchema))
    return std::move(error);
  if (llvm::Error error = reader.expect(mappedRtlResultVersion))
    return std::move(error);
  if (llvm::Error error = reader.expect("terminal"))
    return std::move(error);
  auto terminal = reader.take("terminal status");
  if (!terminal)
    return terminal.takeError();
  MappedRtlSimulationResult result;
  auto terminalStatus = parseTerminalStatus(*terminal);
  if (!terminalStatus)
    return terminalStatus.takeError();
  result.terminal = *terminalStatus;

  if (llvm::Error error = reader.expect("launch_cycle"))
    return std::move(error);
  auto launch = takeUnsigned(reader, "launch_cycle");
  if (!launch)
    return launch.takeError();
  result.launchCycle = *launch;
  if (llvm::Error error = reader.expect("retirement_cycle"))
    return std::move(error);
  auto retirement = reader.take("retirement_cycle");
  if (!retirement)
    return retirement.takeError();
  if (*retirement != "absent") {
    auto parsed = parseUnsigned(*retirement, "retirement_cycle");
    if (!parsed)
      return parsed.takeError();
    result.retirementCycle = *parsed;
  }
  if (llvm::Error error = reader.expect("terminal_cycle"))
    return std::move(error);
  auto terminalCycle = takeUnsigned(reader, "terminal_cycle");
  if (!terminalCycle)
    return terminalCycle.takeError();
  result.terminalCycle = *terminalCycle;

  if (llvm::Error error = reader.expect("value_results"))
    return std::move(error);
  auto valueCount = takeUnsigned(reader, "value_results count");
  if (!valueCount || *valueCount > std::numeric_limits<std::size_t>::max())
    return valueCount ? invalid("value_results count is too large")
                      : valueCount.takeError();
  result.valueResults.reserve(static_cast<std::size_t>(*valueCount));
  for (std::uint64_t ordinal = 0; ordinal != *valueCount; ++ordinal) {
    if (llvm::Error error = reader.expect("value"))
      return std::move(error);
    auto foundOrdinal = takeUnsigned(reader, "value ordinal");
    if (!foundOrdinal || *foundOrdinal != ordinal)
      return foundOrdinal ? invalid("value ordinal is not dense")
                          : foundOrdinal.takeError();
    auto widthOrAbsent = reader.take("value payload");
    if (!widthOrAbsent)
      return widthOrAbsent.takeError();
    if (*widthOrAbsent == "absent") {
      result.valueResults.push_back({std::nullopt});
      continue;
    }
    auto width = parseWidth(*widthOrAbsent, "value width");
    if (!width)
      return width.takeError();
    auto token = reader.take("value bits");
    if (!token)
      return token.takeError();
    auto bits = parseBits(*token, *width);
    if (!bits)
      return bits.takeError();
    result.valueResults.push_back({std::move(*bits)});
  }

  if (llvm::Error error = reader.expect("stream_outputs"))
    return std::move(error);
  auto streamCount = takeUnsigned(reader, "stream_outputs count");
  if (!streamCount || *streamCount > std::numeric_limits<std::size_t>::max())
    return streamCount ? invalid("stream_outputs count is too large")
                       : streamCount.takeError();
  result.streamOutputs.reserve(static_cast<std::size_t>(*streamCount));
  for (std::uint64_t ordinal = 0; ordinal != *streamCount; ++ordinal) {
    if (llvm::Error error = reader.expect("stream"))
      return std::move(error);
    auto foundOrdinal = takeUnsigned(reader, "stream ordinal");
    if (!foundOrdinal || *foundOrdinal != ordinal)
      return foundOrdinal ? invalid("stream ordinal is not dense")
                          : foundOrdinal.takeError();
    auto termination = reader.take("stream termination");
    if (!termination)
      return termination.takeError();
    auto state = parseStreamTermination(*termination);
    if (!state)
      return state.takeError();
    auto width = takeWidth(reader, "stream width");
    if (!width)
      return width.takeError();
    auto tokenCount = takeUnsigned(reader, "stream token count");
    if (!tokenCount || *tokenCount > std::numeric_limits<std::size_t>::max())
      return tokenCount ? invalid("stream token count is too large")
                        : tokenCount.takeError();
    MappedRtlStreamObservation stream{*width, {}, *state};
    stream.tokens.reserve(static_cast<std::size_t>(*tokenCount));
    for (std::uint64_t tokenOrdinal = 0; tokenOrdinal != *tokenCount;
         ++tokenOrdinal) {
      auto token = reader.take("stream token");
      if (!token)
        return token.takeError();
      auto bits = parseBits(*token, *width);
      if (!bits)
        return bits.takeError();
      stream.tokens.push_back(std::move(*bits));
    }
    result.streamOutputs.push_back(std::move(stream));
  }

  if (llvm::Error error = reader.expect("memories"))
    return std::move(error);
  auto memoryCount = takeUnsigned(reader, "memories count");
  if (!memoryCount || *memoryCount > std::numeric_limits<std::size_t>::max())
    return memoryCount ? invalid("memories count is too large")
                       : memoryCount.takeError();
  result.memories.reserve(static_cast<std::size_t>(*memoryCount));
  for (std::uint64_t ordinal = 0; ordinal != *memoryCount; ++ordinal) {
    if (llvm::Error error = reader.expect("memory"))
      return std::move(error);
    auto foundOrdinal = takeUnsigned(reader, "memory ordinal");
    if (!foundOrdinal || *foundOrdinal != ordinal)
      return foundOrdinal ? invalid("memory ordinal is not dense")
                          : foundOrdinal.takeError();
    auto byteCount = takeUnsigned(reader, "memory byte count");
    if (!byteCount || *byteCount == 0 ||
        *byteCount > std::numeric_limits<std::size_t>::max())
      return byteCount ? invalid("memory byte count is invalid")
                       : byteCount.takeError();
    MappedRtlMemoryObservation memory;
    memory.bytes.reserve(static_cast<std::size_t>(*byteCount));
    for (std::uint64_t byteOrdinal = 0; byteOrdinal != *byteCount;
         ++byteOrdinal) {
      auto token = reader.take("memory byte");
      if (!token)
        return token.takeError();
      if (*token == "u")
        memory.bytes.push_back({sim::SemanticState::Undef, 0});
      else if (*token == "p")
        memory.bytes.push_back({sim::SemanticState::Poison, 0});
      else if (llvm::StringRef(*token).consume_front("d")) {
        llvm::StringRef digits(*token);
        digits = digits.drop_front();
        auto value = parseHexByte(digits);
        if (!value)
          return value.takeError();
        memory.bytes.push_back({sim::SemanticState::Defined, *value});
      } else {
        return invalid("memory byte state is unknown");
      }
    }
    result.memories.push_back(std::move(memory));
  }
  if (llvm::Error error = reader.expect("end"))
    return std::move(error);
  if (!reader.finished())
    return invalid("result has trailing tokens");
  if (llvm::Error error = validateResult(result))
    return std::move(error);
  auto canonical = renderMappedRtlSimulationResult(result);
  if (!canonical)
    return canonical.takeError();
  if (*canonical != contents)
    return invalid("result bytes are not canonical");
  return result;
}

} // namespace loom::eda::open_source
