#include "PortableProviderTestSupport.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace {

[[noreturn]] void fail(llvm::StringRef test, const std::string &message) {
  llvm::errs() << test << ": " << message << '\n';
  std::exit(EXIT_FAILURE);
}

void require(llvm::StringRef test, bool condition, llvm::StringRef message) {
  if (!condition)
    fail(test, message.str());
}

void take(llvm::StringRef test, llvm::Error error) {
  if (error)
    fail(test, llvm::toString(std::move(error)));
}

void expectError(llvm::StringRef test, llvm::Error error,
                 llvm::StringRef expected) {
  if (!error)
    fail(test, "accepted an invalid artifact path");
  const std::string message = llvm::toString(std::move(error));
  require(test, llvm::StringRef(message).contains(expected), message);
}

std::string readFile(const std::filesystem::path &path) {
  std::ifstream input(path, std::ios::binary);
  return std::string(std::istreambuf_iterator<char>(input),
                     std::istreambuf_iterator<char>());
}

void artifactWriterIsExactAndConfined(const std::filesystem::path &root) {
  using loom::hardware::test::PortableProviderArtifact;
  using loom::hardware::test::writePortableProviderArtifacts;

  const std::vector<PortableProviderArtifact> artifacts{
      {"rtl/design.sv", "module design; endmodule\n"},
      {"testbench.sv", "module testbench; endmodule\n"},
      {"synth.ys", "read_verilog rtl/design.sv\n"}};
  take(__func__, writePortableProviderArtifacts(root, artifacts));
  for (const PortableProviderArtifact &artifact : artifacts)
    require(__func__,
            readFile(root / artifact.relativePath) == artifact.contents,
            "artifact writer changed package-owned bytes");

  expectError(
      __func__,
      writePortableProviderArtifacts(root, {{"../outside.sv", "forbidden"}}),
      "relative");
  expectError(__func__,
              writePortableProviderArtifacts(
                  root, {{std::filesystem::absolute(root / "absolute.sv"),
                          "forbidden"}}),
              "relative");
  expectError(__func__,
              writePortableProviderArtifacts(
                  root, {{"same.sv", "first"}, {"same.sv", "second"}}),
              "duplicate");

  const std::filesystem::path atomic = root / "atomic";
  expectError(
      __func__,
      writePortableProviderArtifacts(
          atomic, {{"first.sv", "partial"}, {"../second.sv", "forbidden"}}),
      "relative");
  require(__func__, !std::filesystem::exists(atomic),
          "invalid artifact list left a partial root");

  const std::filesystem::path hardlinkRoot = root.string() + "-hardlink";
  const std::filesystem::path hardlinkOutside =
      root.string() + "-hardlink-outside.sv";
  std::filesystem::remove_all(hardlinkRoot);
  std::filesystem::remove(hardlinkOutside);
  std::filesystem::create_directories(hardlinkRoot);
  {
    std::ofstream output(hardlinkOutside, std::ios::binary);
    output << "outside sentinel\n";
  }
  std::filesystem::create_hard_link(hardlinkOutside,
                                    hardlinkRoot / "design.sv");
  expectError(__func__,
              writePortableProviderArtifacts(hardlinkRoot,
                                             {{"design.sv", "replacement\n"}}),
              "already exists");
  require(__func__, readFile(hardlinkOutside) == "outside sentinel\n",
          "artifact publication changed an inode outside its root");

  const std::filesystem::path transactional = root.string() + "-transactional";
  std::filesystem::remove_all(transactional);
  expectError(
      __func__,
      writePortableProviderArtifacts(
          transactional, {{"prefix", "partial"}, {"prefix/nested", "blocked"}}),
      "could not create artifact directory");
  require(__func__, !std::filesystem::exists(transactional),
          "failed artifact publication left a partial root");
}

} // namespace

int main(int argc, char **argv) {
  require("main", argc == 2, "expected one artifact root");
  artifactWriterIsExactAndConfined(argv[1]);
  return EXIT_SUCCESS;
}
