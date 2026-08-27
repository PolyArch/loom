#include "InvocationBundleInternal.h"

#include "Common/BlobDigest.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <optional>
#include <string>
#include <sys/stat.h>
#include <system_error>
#include <unistd.h>
#include <vector>

namespace loom::external_tool {
namespace {

constexpr std::uint64_t observationHeaderBytes = 512;
constexpr std::uint64_t maximumObservationRowBytes = 96;
constexpr std::uint64_t nanosecondsPerSecond = 1'000'000'000;

llvm::Error observationError(const llvm::Twine &detail) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "external_tool_command_observation_invalid: " + detail);
}

llvm::Expected<std::string> readObservationFile(int bundleRoot,
                                                std::uint64_t maximumBytes) {
  const int descriptor =
      ::openat(bundleRoot, kCommandObservationsPath.str().c_str(),
               O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK);
  if (descriptor < 0)
    return observationError(llvm::Twine("cannot open command observations: ") +
                            std::strerror(errno));
  struct Close final {
    int descriptor;
    ~Close() { ::close(descriptor); }
  } close{descriptor};
  struct stat before{};
  if (::fstat(descriptor, &before) != 0 || !S_ISREG(before.st_mode) ||
      before.st_size < 0 ||
      static_cast<std::uint64_t>(before.st_size) > maximumBytes)
    return observationError("command observations are not an ordinary "
                            "bounded file");
  std::string contents;
  contents.reserve(static_cast<std::size_t>(before.st_size));
  char buffer[4096];
  while (true) {
    const ssize_t count = ::read(descriptor, buffer, sizeof(buffer));
    if (count == 0)
      break;
    if (count < 0) {
      if (errno == EINTR)
        continue;
      return observationError(
          llvm::Twine("cannot read command observations: ") +
          std::strerror(errno));
    }
    if (static_cast<std::uint64_t>(count) >
        maximumBytes - static_cast<std::uint64_t>(contents.size()))
      return observationError("command observations exceed their byte bound");
    contents.append(buffer, static_cast<std::size_t>(count));
  }
  struct stat after{};
  if (::fstat(descriptor, &after) != 0 || before.st_dev != after.st_dev ||
      before.st_ino != after.st_ino || before.st_mode != after.st_mode ||
      before.st_size != after.st_size ||
      before.st_mtim.tv_sec != after.st_mtim.tv_sec ||
      before.st_mtim.tv_nsec != after.st_mtim.tv_nsec ||
      before.st_ctim.tv_sec != after.st_ctim.tv_sec ||
      before.st_ctim.tv_nsec != after.st_ctim.tv_nsec ||
      contents.size() != static_cast<std::uint64_t>(after.st_size))
    return observationError("command observations changed while reading");
  return contents;
}

llvm::Expected<std::uint64_t> observationByteLimit(std::uint64_t commandCount) {
  if (commandCount >
      (std::numeric_limits<std::uint64_t>::max() - observationHeaderBytes) /
          maximumObservationRowBytes)
    return observationError("command count overflows the observation bound");
  return observationHeaderBytes + commandCount * maximumObservationRowBytes;
}

llvm::Expected<std::uint64_t> parseWallNanoseconds(llvm::StringRef spelling) {
  const auto [secondsText, fractionText] = spelling.split('.');
  if (secondsText.empty() || fractionText.empty() || fractionText.size() > 9 ||
      !llvm::all_of(secondsText,
                    [](char value) { return value >= '0' && value <= '9'; }) ||
      !llvm::all_of(fractionText,
                    [](char value) { return value >= '0' && value <= '9'; }))
    return observationError("command wall time is not canonical decimal "
                            "seconds");
  std::uint64_t seconds = 0;
  std::uint64_t fraction = 0;
  if (secondsText.getAsInteger(10, seconds) ||
      fractionText.getAsInteger(10, fraction) ||
      seconds >
          std::numeric_limits<std::uint64_t>::max() / nanosecondsPerSecond)
    return observationError("command wall time overflows nanoseconds");
  for (std::size_t digits = fractionText.size(); digits != 9; ++digits)
    fraction *= 10;
  const std::uint64_t whole = seconds * nanosecondsPerSecond;
  if (fraction > std::numeric_limits<std::uint64_t>::max() - whole)
    return observationError("command wall time overflows nanoseconds");
  return whole + fraction;
}

} // namespace

llvm::Expected<std::vector<ExternalToolCommandExecutionObservation>>
loadCommandExecutionObservations(const PreparedExternalToolInvocation &prepared,
                                 const BlobDigest &attemptToken,
                                 std::uint64_t commandCount, int bundleRoot) {
  auto maximumBytes = observationByteLimit(commandCount);
  if (!maximumBytes)
    return maximumBytes.takeError();
  auto contents = readObservationFile(bundleRoot, *maximumBytes);
  if (!contents)
    return contents.takeError();
  llvm::SmallVector<llvm::StringRef, 16> lines;
  llvm::StringRef(*contents).split(lines, '\n');
  if (lines.size() < 5 || !lines.back().empty())
    return observationError("command observations are truncated");
  lines.pop_back();
  if (lines.front() != "loom.external_tool_command_observations 1.0" ||
      lines[1] != "manifest " + formatBlobDigestHex(prepared.manifestDigest) ||
      lines[2] != "attempt " + formatBlobDigestHex(attemptToken) ||
      lines.back() != "end")
    return observationError(
        "command observations do not bind the current attempt");
  std::vector<ExternalToolCommandExecutionObservation> observations;
  observations.reserve(lines.size() - 4);
  for (std::size_t index = 3; index + 1 < lines.size(); ++index) {
    const llvm::StringRef line = lines[index];
    llvm::SmallVector<llvm::StringRef, 5> fields;
    line.split(fields, ' ', -1, false);
    std::uint64_t ordinal = 0;
    std::uint64_t exitCode = 0;
    if (fields.size() != 4 || fields[0] != "command" ||
        fields[1].getAsInteger(10, ordinal) || ordinal >= commandCount ||
        fields[3].getAsInteger(10, exitCode) || exitCode > 255 ||
        (!observations.empty() &&
         observations.back().commandOrdinal >= ordinal))
      return observationError("command observation row is invalid");
    auto wall = parseWallNanoseconds(fields[2]);
    if (!wall)
      return wall.takeError();
    observations.push_back({ordinal, *wall, static_cast<int>(exitCode)});
  }
  return observations;
}

} // namespace loom::external_tool
