// Exercises the hierarchy Verilator launcher against a recording Verilator
// stand-in: concurrent invocations on one child argument file publish one
// immutable filtered sibling, the harness token count is enforced, and root
// argument files pass through untouched.

#include "EDA/Adapters/OpenSource/MappedRtlHierarchyLauncher.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <optional>
#include <string>
#include <vector>

namespace {

using namespace loom::eda::open_source;

constexpr std::size_t kConcurrentInvocations = 8;
constexpr llvm::StringLiteral kRecordDirectoryVariable =
    "LOOM_TEST_LAUNCHER_RECORD_DIRECTORY";

[[noreturn]] void fail(llvm::StringRef message) {
  llvm::errs() << "mapped RTL hierarchy launcher test: " << message << '\n';
  std::exit(EXIT_FAILURE);
}

void require(bool condition, llvm::StringRef message) {
  if (!condition)
    fail(message);
}

void writeFile(const std::filesystem::path &path, llvm::StringRef contents) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  output.write(contents.data(), static_cast<std::streamsize>(contents.size()));
  if (!output)
    fail("cannot write " + path.string());
}

std::string readFile(const std::filesystem::path &path) {
  std::ifstream input(path, std::ios::binary);
  if (!input)
    fail("cannot read " + path.string());
  return std::string(std::istreambuf_iterator<char>(input),
                     std::istreambuf_iterator<char>());
}

struct LauncherRun final {
  int exitCode = 0;
  /// The argument vector and argument-file bytes the Verilator stand-in saw,
  /// absent when the launcher failed before executing it.
  std::optional<std::string> verilatorArguments;
  std::optional<std::string> verilatorArgumentFile;
};

class LauncherHarness final {
public:
  explicit LauncherHarness(const std::filesystem::path &root)
      : root_(root), recordDirectory_(root / "records"),
        verilator_(root / "fake-verilator") {
    std::filesystem::create_directories(recordDirectory_);
    // The stand-in records its argument vector and copies the argument file
    // it was handed, keyed by its own process id.
    writeFile(verilator_,
              "#!/usr/bin/env bash\nset -euo pipefail\n"
              "printf '%s\\n' \"$@\" > \"$" +
                  kRecordDirectoryVariable.str() +
                  "/$$.argv\"\n"
                  "if [[ \"$#\" -ge 2 && \"$1\" == '-f' ]]; then cp -- \"$2\" "
                  "\"$" +
                  kRecordDirectoryVariable.str() + "/$$.args\"; fi\n");
    std::filesystem::permissions(verilator_, std::filesystem::perms::owner_all);
  }

  std::vector<LauncherRun> run(std::size_t count, llvm::StringRef argumentFile,
                               bool configured = true) const {
    std::vector<std::string> environment{
        ("PATH=" + std::string(std::getenv("PATH") ? std::getenv("PATH")
                                                   : "/usr/bin:/bin")),
        (kRecordDirectoryVariable + "=" + recordDirectory_.string()).str()};
    if (configured) {
      environment.push_back(
          (mappedRtlHierarchyVerilatorVariable + "=" + verilator_.string())
              .str());
      environment.push_back(
          (mappedRtlHierarchyTestbenchVariable + "=drivers/testbench.sv")
              .str());
    }
    std::vector<llvm::StringRef> environmentRefs(environment.begin(),
                                                 environment.end());
    const std::string launcher = LOOM_MAPPED_RTL_HIERARCHY_LAUNCHER_PATH;
    const std::vector<llvm::StringRef> arguments{launcher, "-f", argumentFile};
    std::vector<llvm::sys::ProcessInfo> processes;
    for (std::size_t ordinal = 0; ordinal != count; ++ordinal) {
      std::string message;
      bool executionFailed = false;
      processes.push_back(llvm::sys::ExecuteNoWait(
          launcher, arguments, llvm::ArrayRef<llvm::StringRef>(environmentRefs),
          {}, 0, &message, &executionFailed));
      require(!executionFailed && processes.back().Pid != 0,
              "could not start the hierarchy launcher: " + message);
    }
    std::vector<LauncherRun> runs;
    for (const llvm::sys::ProcessInfo &process : processes) {
      std::string message;
      const llvm::sys::ProcessInfo waited =
          llvm::sys::Wait(process, std::nullopt, &message);
      require(waited.Pid == process.Pid, "launcher wait failed: " + message);
      LauncherRun run;
      run.exitCode = waited.ReturnCode;
      const std::filesystem::path record =
          recordDirectory_ / (std::to_string(process.Pid) + ".argv");
      if (std::filesystem::exists(record)) {
        run.verilatorArguments = readFile(record);
        const std::filesystem::path copied =
            recordDirectory_ / (std::to_string(process.Pid) + ".args");
        if (std::filesystem::exists(copied))
          run.verilatorArgumentFile = readFile(copied);
      }
      runs.push_back(std::move(run));
    }
    return runs;
  }

  const std::filesystem::path &root() const { return root_; }

private:
  std::filesystem::path root_;
  std::filesystem::path recordDirectory_;
  std::filesystem::path verilator_;
};

std::string childArguments(unsigned testbenchTokens) {
  std::string text = "--cc\n-Mdir work/verilator/Vblock \n"
                     "/absolute/drivers/verilator-library/block.sv\n"
                     " --prefix Vblock\n"
                     " --top-module-encoded block\n"
                     " --hierarchical-child 1\n"
                     "--hierarchical-block block,block\n";
  for (unsigned ordinal = 0; ordinal != testbenchTokens; ++ordinal)
    text += "drivers/testbench.sv\n";
  text += "\"--main\" \"-j\" \"8\" \"-y\" \"drivers/verilator-library\" "
          "\"+libext+.sv\"\n";
  return text;
}

std::string rootArguments() {
  return "Vblock/block.sv\n--hierarchical-block block,block\n--threads 1\n"
         "--cc\ndrivers/testbench.sv\n"
         "\"--main\" \"-j\" \"8\" \"-y\" \"drivers/verilator-library\" "
         "\"+libext+.sv\"\n";
}

bool hasPartialFiles(const std::filesystem::path &directory) {
  for (const auto &entry : std::filesystem::directory_iterator(directory))
    if (llvm::StringRef(entry.path().filename().string()).ends_with(".tmp"))
      return true;
  return false;
}

void concurrentChildInvocationsShareOneFilteredFile(
    const LauncherHarness &harness) {
  const std::filesystem::path directory = harness.root() / "work/verilator";
  const std::filesystem::path arguments = directory / "Vblock__hierMkArgs.f";
  writeFile(arguments, childArguments(1));
  const std::string expected = childArguments(0);
  const std::filesystem::path filtered =
      directory /
      ("Vblock__hierMkArgs.f" + mappedRtlHierarchyChildArgumentsSuffix.str());
  const std::vector<LauncherRun> runs =
      harness.run(kConcurrentInvocations, arguments.string());
  for (const LauncherRun &run : runs) {
    require(run.exitCode == 0, "concurrent child invocation failed");
    require(run.verilatorArguments &&
                *run.verilatorArguments == "-f\n" + filtered.string() + "\n",
            "Verilator did not receive the filtered sibling");
    require(run.verilatorArgumentFile && *run.verilatorArgumentFile == expected,
            "Verilator read a filtered file that differs from the expected "
            "bytes");
  }
  require(readFile(filtered) == expected,
          "the published filtered sibling has unexpected bytes");
  require(readFile(arguments) == childArguments(1),
          "the Verilator-generated argument file was edited in place");
  require(!hasPartialFiles(directory),
          "a concurrent invocation left a partial file behind");
}

void tokenCountIsEnforced(const LauncherHarness &harness, unsigned tokens) {
  const std::filesystem::path directory = harness.root() / "work/verilator";
  const std::string name =
      "Vtokens" + std::to_string(tokens) + "__hierMkArgs.f";
  writeFile(directory / name, childArguments(tokens));
  const std::vector<LauncherRun> runs =
      harness.run(1, (directory / name).string());
  require(runs.front().exitCode ==
                  static_cast<int>(
                      MappedRtlHierarchyLauncherExit::TestbenchTokenCount) &&
              !runs.front().verilatorArguments,
          "a child argument file with the wrong harness token count did not "
          "fail closed");
  require(
      !std::filesystem::exists(
          directory / (name + mappedRtlHierarchyChildArgumentsSuffix.str())),
      "a rejected child argument file published a filtered sibling");
}

void rootArgumentsPassThrough(const LauncherHarness &harness) {
  const std::filesystem::path directory = harness.root() / "work/verilator";
  const std::filesystem::path arguments = directory / "Vtop__hierMkArgs.f";
  writeFile(arguments, rootArguments());
  const std::vector<LauncherRun> runs = harness.run(1, arguments.string());
  require(runs.front().exitCode == 0 && runs.front().verilatorArguments &&
              *runs.front().verilatorArguments ==
                  "-f\n" + arguments.string() + "\n" &&
              runs.front().verilatorArgumentFile == rootArguments(),
          "the root argument file did not pass through unchanged");
  require(!std::filesystem::exists(
              directory / ("Vtop__hierMkArgs.f" +
                           mappedRtlHierarchyChildArgumentsSuffix.str())),
          "the root argument file gained a filtered sibling");
}

void missingConfigurationFailsClosed(const LauncherHarness &harness) {
  const std::vector<LauncherRun> runs = harness.run(
      1, (harness.root() / "work/verilator/Vblock__hierMkArgs.f").string(),
      false);
  require(
      runs.front().exitCode ==
              static_cast<int>(MappedRtlHierarchyLauncherExit::Configuration) &&
          !runs.front().verilatorArguments,
      "an unconfigured launcher did not fail closed");
}

} // namespace

int main() {
  llvm::SmallString<256> rootStorage;
  if (const std::error_code error = llvm::sys::fs::createUniqueDirectory(
          "loom-mapped-rtl-hierarchy-launcher", rootStorage))
    fail("cannot create a scratch directory: " + error.message());
  const std::filesystem::path root(rootStorage.str().str());
  {
    const LauncherHarness harness(root);
    concurrentChildInvocationsShareOneFilteredFile(harness);
    tokenCountIsEnforced(harness, 0);
    tokenCountIsEnforced(harness, 2);
    rootArgumentsPassThrough(harness);
    missingConfigurationFailsClosed(harness);
  }
  std::filesystem::remove_all(root);
  return EXIT_SUCCESS;
}
