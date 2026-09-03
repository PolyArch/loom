#include "EDA/Adapters/OpenSource/MappedRtlSimulation.h"

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
      "mapped_rtl_configuration_transport_receipt_invalid: " + detail);
}

llvm::Expected<std::uint64_t> parseUnsigned(llvm::StringRef token,
                                            llvm::StringRef field) {
  std::uint64_t value = 0;
  if (token.empty() || (token.size() > 1 && token.front() == '0') ||
      token.getAsInteger(10, value))
    return invalid(field + " is not a canonical unsigned integer");
  return value;
}

class TokenReader final {
public:
  explicit TokenReader(llvm::StringRef contents) : stream_(contents.str()) {}

  llvm::Expected<std::string> take(llvm::StringRef field) {
    std::string token;
    if (!(stream_ >> token))
      return invalid("receipt ended before " + field);
    return token;
  }

  llvm::Error expect(llvm::StringRef expected) {
    auto token = take(expected);
    if (!token)
      return token.takeError();
    if (*token != expected)
      return invalid("expected '" + expected + "' but found '" + *token +
                     "'");
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

} // namespace

llvm::Expected<std::string> renderMappedRtlConfigurationTransportReceipt(
    const MappedRtlConfigurationTransportReceipt &receipt) {
  std::string text;
  llvm::raw_string_ostream output(text);
  output << mappedRtlConfigurationTransportReceiptSchema << " "
         << mappedRtlConfigurationTransportReceiptVersion << "\nprograms "
         << receipt.programs.size() << "\n";
  for (const auto &[ordinal, program] : llvm::enumerate(receipt.programs))
    output << "program " << ordinal << " payload_writes "
           << program.payloadWrites << " atomic_commits "
           << program.atomicCommits << " active_word_comparisons "
           << program.activeWordComparisons << " passing_status_reads "
           << program.passingStatusReads << "\n";
  output << "end\n";
  return text;
}

llvm::Expected<MappedRtlConfigurationTransportReceipt>
parseMappedRtlConfigurationTransportReceipt(llvm::StringRef contents) {
  TokenReader reader(contents);
  if (llvm::Error error =
          reader.expect(mappedRtlConfigurationTransportReceiptSchema))
    return std::move(error);
  if (llvm::Error error =
          reader.expect(mappedRtlConfigurationTransportReceiptVersion))
    return std::move(error);
  if (llvm::Error error = reader.expect("programs"))
    return std::move(error);
  auto count = takeUnsigned(reader, "program count");
  if (!count || *count > std::numeric_limits<std::size_t>::max())
    return count ? invalid("program count is too large") : count.takeError();

  MappedRtlConfigurationTransportReceipt receipt;
  receipt.programs.reserve(static_cast<std::size_t>(*count));
  for (std::uint64_t ordinal = 0; ordinal != *count; ++ordinal) {
    if (llvm::Error error = reader.expect("program"))
      return std::move(error);
    auto foundOrdinal = takeUnsigned(reader, "program ordinal");
    if (!foundOrdinal || *foundOrdinal != ordinal)
      return foundOrdinal ? invalid("program ordinal is not dense")
                          : foundOrdinal.takeError();
    if (llvm::Error error = reader.expect("payload_writes"))
      return std::move(error);
    auto payloadWrites = takeUnsigned(reader, "payload write count");
    if (!payloadWrites)
      return payloadWrites.takeError();
    if (llvm::Error error = reader.expect("atomic_commits"))
      return std::move(error);
    auto atomicCommits = takeUnsigned(reader, "atomic commit count");
    if (!atomicCommits)
      return atomicCommits.takeError();
    if (llvm::Error error = reader.expect("active_word_comparisons"))
      return std::move(error);
    auto activeWordComparisons =
        takeUnsigned(reader, "active-word comparison count");
    if (!activeWordComparisons)
      return activeWordComparisons.takeError();
    if (llvm::Error error = reader.expect("passing_status_reads"))
      return std::move(error);
    auto passingStatusReads = takeUnsigned(reader, "passing status read count");
    if (!passingStatusReads)
      return passingStatusReads.takeError();
    receipt.programs.push_back({*payloadWrites, *atomicCommits,
                                *activeWordComparisons, *passingStatusReads});
  }
  if (llvm::Error error = reader.expect("end"))
    return std::move(error);
  if (!reader.finished())
    return invalid("receipt has trailing tokens");
  auto canonical = renderMappedRtlConfigurationTransportReceipt(receipt);
  if (!canonical)
    return canonical.takeError();
  if (*canonical != contents)
    return invalid("receipt bytes are not canonical");
  return receipt;
}

} // namespace loom::eda::open_source
