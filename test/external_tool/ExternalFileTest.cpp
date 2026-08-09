#include "ExternalTool/ExternalFile.h"

#include "ExternalTool/LocalConfig.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

using namespace loom::external_tool;

namespace {

[[noreturn]] void fail(llvm::StringRef test, const std::string &message) {
  std::cerr << test.str() << ": " << message << '\n';
  std::exit(1);
}

void require(llvm::StringRef test, bool condition,
             const std::string &message) {
  if (!condition)
    fail(test, message);
}

template <typename T>
T take(llvm::StringRef test, llvm::Expected<T> value) {
  if (!value)
    fail(test, llvm::toString(value.takeError()));
  return std::move(*value);
}

template <typename T>
void expectErrorContains(llvm::StringRef test, llvm::Expected<T> value,
                         llvm::StringRef expected) {
  if (value)
    fail(test, "expected an error");
  const std::string message = llvm::toString(value.takeError());
  require(test, llvm::StringRef(message).contains(expected), message);
}

void writeFile(llvm::StringRef test, const std::filesystem::path &path,
               llvm::StringRef contents) {
  std::filesystem::create_directories(path.parent_path());
  std::error_code error;
  llvm::raw_fd_ostream output(path.string(), error, llvm::sys::fs::OF_None);
  if (error)
    fail(test, "could not create fixture: " + error.message());
  output << contents;
  output.close();
  if (output.has_error())
    fail(test, "could not write fixture");
}

ExternalFileFingerprint alphaFingerprint(llvm::StringRef test) {
  return take(test, parseExternalFileFingerprint(
                        "8ed3f6ad685b959ead7022518e1af76cd816f8e8ec7ccdda1ed4"
                        "018e8f2223f8"));
}

ExternalFileFingerprint betaFingerprint(llvm::StringRef test) {
  return take(test, parseExternalFileFingerprint(
                        "f44e64e75f3948e9f73f8dfa94721c4ce8cbb4f265c4790c702"
                        "b2d41cfbf2753"));
}

void exactBytesResolveByFingerprint(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  const std::filesystem::path later = root / "z" / "alpha.lib";
  const std::filesystem::path earlier = root / "a" / "alpha.lib";
  const std::filesystem::path other = root / "b" / "beta.lib";
  writeFile(test, later, "alpha");
  writeFile(test, earlier, "alpha");
  writeFile(test, other, "beta");

  LocalToolConfig config;
  config.externalFiles = {{"later", later.string()},
                          {"other", other.string()},
                          {"earlier", earlier.string()}};
  std::vector<ExternalFileRequirement> requirements{
      {"liberty_tt", alphaFingerprint(test)}};
  std::vector<ResolvedExternalFile> resolved =
      take(test, resolveExternalFiles(requirements, config));
  require(test, resolved.size() == 1, "wrong resolved file count");
  require(test, resolved[0].providerInputSlot == "liberty_tt" &&
                    resolved[0].localFileKey == "earlier" &&
                    resolved[0].absolutePath == earlier.string() &&
                    resolved[0].fingerprint == alphaFingerprint(test),
          "resolver did not freeze the canonical matching local file");
}

void missingAndMalformedRequirementsFail(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  const std::filesystem::path beta = root / "missing" / "beta.lib";
  writeFile(test, beta, "beta");
  LocalToolConfig config;
  config.externalFiles = {{"beta", beta.string()}};
  expectErrorContains(
      test,
      resolveExternalFiles({{"liberty_tt", alphaFingerprint(test)}}, config),
      "no configured external file matches");

  expectErrorContains(test,
                      parseExternalFileFingerprint(
                          "8ED3F6AD685B959EAD7022518E1AF76CD816F8E8EC7CCDDA1ED4"
                          "018E8F2223F8"),
                      "lowercase");
  expectErrorContains(test, parseExternalFileFingerprint("00"),
                      "64 lowercase");

  std::vector<ExternalFileRequirement> duplicateSlots{
      {"liberty_tt", alphaFingerprint(test)},
      {"liberty_tt", alphaFingerprint(test)}};
  expectErrorContains(test, resolveExternalFiles(duplicateSlots, config),
                      "duplicate provider input slot");
}

void unsafeLocalEntriesFail(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  const std::filesystem::path target = root / "unsafe" / "target.lib";
  writeFile(test, target, "alpha");

  const std::filesystem::path link = root / "unsafe" / "link.lib";
  std::filesystem::create_symlink(target, link);
  LocalToolConfig symlinkConfig;
  symlinkConfig.externalFiles = {{"link", link.string()}};
  expectErrorContains(
      test,
      resolveExternalFiles({{"liberty_tt", alphaFingerprint(test)}},
                           symlinkConfig),
      "symlink");

  LocalToolConfig directoryConfig;
  directoryConfig.externalFiles = {{"directory", target.parent_path().string()}};
  expectErrorContains(
      test,
      resolveExternalFiles({{"liberty_tt", alphaFingerprint(test)}},
                           directoryConfig),
      "ordinary file");

  LocalToolConfig duplicatePathConfig;
  duplicatePathConfig.externalFiles = {{"first", target.string()},
                                       {"second", target.string()}};
  expectErrorContains(
      test,
      resolveExternalFiles({{"liberty_tt", alphaFingerprint(test)}},
                           duplicatePathConfig),
      "duplicate canonical path");
}

void exactFileTreesResolveByMemberLayout(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  const std::filesystem::path later = root / "tree-z" / "reference.ndm";
  const std::filesystem::path earlier = root / "tree-a" / "reference.ndm";
  for (const std::filesystem::path &tree : {later, earlier}) {
    writeFile(test, tree / "pcat", "alpha");
    writeFile(test, tree / "parts" / "p0", "beta");
  }

  LocalToolConfig config;
  config.externalFileTrees = {{"later", later.string()},
                              {"earlier", earlier.string()}};
  const ExternalFileTreeRequirement requirement{
      "reference_library",
      {{"parts/p0", betaFingerprint(test)}, {"pcat", alphaFingerprint(test)}}};
  std::vector<ResolvedExternalFileTree> resolved =
      take(test, resolveExternalFileTrees({requirement}, config));
  require(test, resolved.size() == 1, "wrong resolved tree count");
  require(test,
          resolved.front().providerInputSlot == "reference_library" &&
              resolved.front().localFileTreeKey == "earlier" &&
              resolved.front().absolutePath == earlier.string() &&
              resolved.front().members == requirement.members,
          "resolver did not freeze the canonical matching tree");

  writeFile(test, earlier / "pcat", "beta");
  writeFile(test, later / "pcat", "beta");
  expectErrorContains(test, resolveExternalFileTrees({requirement}, config),
                      "no configured external file tree matches");
  writeFile(test, earlier / "pcat", "alpha");
  writeFile(test, later / "pcat", "alpha");

  std::filesystem::remove(earlier / "parts" / "p0");
  std::filesystem::remove(later / "parts" / "p0");
  expectErrorContains(test, resolveExternalFileTrees({requirement}, config),
                      "no configured external file tree matches");
  writeFile(test, earlier / "parts" / "p0", "beta");
  writeFile(test, later / "parts" / "p0", "beta");

  writeFile(test, earlier / "extra", "alpha");
  writeFile(test, later / "extra", "alpha");
  expectErrorContains(test, resolveExternalFileTrees({requirement}, config),
                      "no configured external file tree matches");
}

void malformedAndUnsafeFileTreesFail(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  const std::filesystem::path tree = root / "unsafe-tree";
  writeFile(test, tree / "member", "alpha");
  LocalToolConfig config;
  config.externalFileTrees = {{"tree", tree.string()}};

  expectErrorContains(
      test,
      resolveExternalFileTrees(
          {{"reference_library", {{"../member", alphaFingerprint(test)}}}},
          config),
      "canonical relative path");
  expectErrorContains(
      test,
      resolveExternalFileTrees({{"reference_library",
                                 {{"member", alphaFingerprint(test)},
                                  {"member", alphaFingerprint(test)}}}},
                               config),
      "duplicate member path");

  std::filesystem::remove(tree / "member");
  std::filesystem::create_symlink(root / "tree-a" / "reference.ndm" / "pcat",
                                  tree / "member");
  expectErrorContains(
      test,
      resolveExternalFileTrees(
          {{"reference_library", {{"member", alphaFingerprint(test)}}}},
          config),
      "symlink");
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2)
    fail("main", "expected one test root");
  const std::filesystem::path root =
      std::filesystem::absolute(argv[1]).lexically_normal();
  std::filesystem::create_directories(root);
  exactBytesResolveByFingerprint(root);
  missingAndMalformedRequirementsFail(root);
  unsafeLocalEntriesFail(root);
  exactFileTreesResolveByMemberLayout(root);
  malformedAndUnsafeFileTreesFail(root);
  return 0;
}
