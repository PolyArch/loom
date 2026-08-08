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

  const std::filesystem::path outside = root.string() + "-outside";
  std::filesystem::create_directories(outside);
  std::filesystem::create_directory_symlink(std::filesystem::absolute(outside),
                                            root / "escape");
  expectError(
      __func__,
      writePortableProviderArtifacts(root, {{"escape/result.sv", "forbidden"}}),
      "escapes");

  const std::filesystem::path atomic = root / "atomic";
  expectError(
      __func__,
      writePortableProviderArtifacts(
          atomic, {{"first.sv", "partial"}, {"../second.sv", "forbidden"}}),
      "relative");
  require(__func__, !std::filesystem::exists(atomic / "first.sv"),
          "invalid artifact list left a partial file");
}

} // namespace

int main(int argc, char **argv) {
  require("main", argc == 2, "expected one artifact root");
  artifactWriterIsExactAndConfined(argv[1]);
  return EXIT_SUCCESS;
}
