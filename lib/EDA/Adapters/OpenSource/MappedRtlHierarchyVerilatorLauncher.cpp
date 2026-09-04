// The hierarchy Verilator launcher: the typed auxiliary tool named by the
// mapped-RTL adapters as Verilator's VM_HIER_VERILATOR. See
// MappedRtlHierarchyLauncher.h for the contract.

#include "EDA/Adapters/OpenSource/MappedRtlHierarchyLauncher.h"

#include "Common/BlobDigest.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/raw_ostream.h"

#include <unistd.h>

#include <cctype>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <string>
#include <vector>

namespace {

using loom::eda::open_source::MappedRtlHierarchyLauncherExit;

constexpr llvm::StringLiteral kLauncherName =
    "loom-mapped-rtl-hierarchy-verilator";
constexpr llvm::StringLiteral kArgumentFileOption = "-f";

/// Emits one launcher line as a single write so concurrent launchers never
/// interleave their records on the shared error stream.
void emitLine(const llvm::Twine &detail) {
  const std::string line = (kLauncherName + ": " + detail + "\n").str();
  llvm::errs() << line;
}

int failWith(MappedRtlHierarchyLauncherExit code, const llvm::Twine &detail) {
  emitLine(detail);
  return static_cast<int>(code);
}

std::string digestHex(llvm::StringRef bytes) {
  return loom::formatBlobDigestHex(
      loom::computeBlobDigest(llvm::ArrayRef<std::uint8_t>(
          reinterpret_cast<const std::uint8_t *>(bytes.data()), bytes.size())));
}

bool isHierarchicalChildLine(llvm::StringRef line) {
  llvm::StringRef trimmed = line.trim();
  if (!trimmed.consume_front(
          loom::eda::open_source::verilatorHierarchicalChildOption))
    return false;
  return trimmed.empty() ||
         std::isspace(static_cast<unsigned char>(trimmed.front()));
}

/// Splits the argument file into lines that keep their own terminators so the
/// filtered sibling reproduces every retained byte exactly.
llvm::SmallVector<llvm::StringRef, 64> terminatedLines(llvm::StringRef bytes) {
  llvm::SmallVector<llvm::StringRef, 64> lines;
  while (!bytes.empty()) {
    const std::size_t newline = bytes.find('\n');
    if (newline == llvm::StringRef::npos) {
      lines.push_back(bytes);
      break;
    }
    lines.push_back(bytes.take_front(newline + 1));
    bytes = bytes.drop_front(newline + 1);
  }
  return lines;
}

struct FilteredArguments final {
  std::string contents;
  std::size_t removedTestbenchTokens = 0;
  std::size_t preambleTokens = 0;
};

FilteredArguments projectArguments(llvm::StringRef bytes,
                                   llvm::StringRef testbench,
                                   llvm::StringRef preamble, bool child) {
  FilteredArguments result;
  result.contents.reserve(bytes.size());
  llvm::StringRef preambleLine;
  for (llvm::StringRef line : terminatedLines(bytes)) {
    if (line.trim() == preamble) {
      ++result.preambleTokens;
      preambleLine = line;
      continue;
    }
    if (child && line.trim() == testbench) {
      ++result.removedTestbenchTokens;
      continue;
    }
    result.contents.append(line.data(), line.size());
  }
  if (!preambleLine.empty()) {
    std::string prefix = preambleLine.str();
    if (prefix.back() != '\n')
      prefix.push_back('\n');
    result.contents.insert(0, prefix);
  }
  return result;
}

llvm::Error publishImmutable(llvm::StringRef path, llvm::StringRef contents) {
  auto existing = llvm::MemoryBuffer::getFile(path, false, false);
  if (existing && (*existing)->getBuffer() == contents)
    return llvm::Error::success();
  const std::string partial =
      (path + "." + llvm::Twine(llvm::sys::Process::getProcessId()) + ".tmp")
          .str();
  std::error_code error;
  {
    llvm::raw_fd_ostream output(partial, error, llvm::sys::fs::OF_None);
    if (error)
      return llvm::createStringError(error, "cannot create " + partial);
    output << contents;
    output.close();
    if (output.has_error())
      return llvm::createStringError(output.error(), "cannot write " + partial);
  }
  if ((error = llvm::sys::fs::rename(partial, path))) {
    (void)llvm::sys::fs::remove(partial);
    return llvm::createStringError(error, "cannot publish " + path);
  }
  return llvm::Error::success();
}

[[noreturn]] void execVerilator(const std::string &verilator,
                                const std::vector<std::string> &arguments) {
  std::vector<char *> argv;
  argv.reserve(arguments.size() + 2);
  std::string program = verilator;
  argv.push_back(program.data());
  std::vector<std::string> owned = arguments;
  for (std::string &argument : owned)
    argv.push_back(argument.data());
  argv.push_back(nullptr);
  execv(program.c_str(), argv.data());
  const int savedErrno = errno;
  std::exit(failWith(MappedRtlHierarchyLauncherExit::InputOutput,
                     "cannot execute " + verilator + ": " +
                         std::strerror(savedErrno)));
}

} // namespace

int main(int argc, char **argv) {
  using namespace loom::eda::open_source;
  const char *verilator =
      std::getenv(mappedRtlHierarchyVerilatorVariable.data());
  const char *testbench =
      std::getenv(mappedRtlHierarchyTestbenchVariable.data());
  const char *preamble = std::getenv(mappedRtlHierarchyPreambleVariable.data());
  if (!verilator || !*verilator || !testbench || !*testbench || !preamble ||
      !*preamble)
    return failWith(MappedRtlHierarchyLauncherExit::Configuration,
                    llvm::Twine(mappedRtlHierarchyVerilatorVariable) + ", " +
                        mappedRtlHierarchyTestbenchVariable + " and " +
                        mappedRtlHierarchyPreambleVariable +
                        " must name the Verilator executable, harness and "
                        "preamble paths");
  std::vector<std::string> arguments(argv + 1, argv + argc);
  std::optional<std::size_t> argumentFile;
  for (std::size_t index = 0; index + 1 < arguments.size(); ++index) {
    if (arguments[index] == kArgumentFileOption) {
      argumentFile = index + 1;
      break;
    }
  }
  if (!argumentFile)
    execVerilator(verilator, arguments);

  const std::string &argumentPath = arguments[*argumentFile];
  auto buffer = llvm::MemoryBuffer::getFile(argumentPath, false, false);
  if (!buffer)
    return failWith(MappedRtlHierarchyLauncherExit::InputOutput,
                    "cannot read " + argumentPath + ": " +
                        buffer.getError().message());
  const llvm::StringRef contents = (*buffer)->getBuffer();
  const bool child =
      llvm::any_of(terminatedLines(contents), isHierarchicalChildLine);
  const FilteredArguments filtered =
      projectArguments(contents, testbench, preamble, child);
  if (filtered.preambleTokens != 1)
    return failWith(MappedRtlHierarchyLauncherExit::PreambleTokenCount,
                    "argument file " + argumentPath + " names " +
                        llvm::Twine(filtered.preambleTokens) +
                        " preamble tokens; exactly one is required");
  if (child && filtered.removedTestbenchTokens != 1)
    return failWith(MappedRtlHierarchyLauncherExit::TestbenchTokenCount,
                    "child argument file " + argumentPath + " names " +
                        llvm::Twine(filtered.removedTestbenchTokens) +
                        " harness tokens; exactly one is required");
  const std::string filteredPath =
      argumentPath + mappedRtlHierarchyArgumentsSuffix.str();
  if (llvm::Error error = publishImmutable(filteredPath, filtered.contents))
    return failWith(MappedRtlHierarchyLauncherExit::InputOutput,
                    llvm::toString(std::move(error)));
  emitLine("arguments=" + argumentPath + " sha256=" + digestHex(contents) +
           " filtered=" + filteredPath +
           " sha256=" + digestHex(filtered.contents));
  arguments[*argumentFile] = filteredPath;
  execVerilator(verilator, arguments);
}
