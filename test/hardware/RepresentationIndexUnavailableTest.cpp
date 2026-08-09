#include "Common/BlobStore.h"
#include "Hardware/Implementation/RepresentationIndex.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <filesystem>

using namespace loom;
using namespace loom::hardware;

namespace {

[[noreturn]] void fail(llvm::StringRef message) {
  llvm::errs() << message << '\n';
  std::exit(EXIT_FAILURE);
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2)
    fail("expected one test-root argument");
  const std::filesystem::path root(argv[1]);
  std::filesystem::remove_all(root);
  std::filesystem::create_directories(root);
  const BlobStore blobs(root.string());

  auto format = RepresentationFormatDescriptorRef::get(
      RepresentationFormatKind::SystemVerilogRtl);
  if (!format)
    fail(llvm::toString(format.takeError()));
  auto index = indexRepresentation(
      *format, {RepresentationObjectKind::Module, "top"}, {}, blobs);
  if (index)
    fail("HDL indexing succeeded without CIRCT");

  bool matched = false;
  llvm::Error remainder = llvm::handleErrors(
      index.takeError(), [&](const RepresentationIndexFailure &failure) {
        matched =
            failure.kind() == RepresentationIndexFailureKind::Unsupported &&
            failure.reason() == "HDL representation indexing requires CIRCT";
      });
  if (remainder)
    fail(llvm::toString(std::move(remainder)));
  if (!matched)
    fail("HDL indexing did not return the exact typed Unsupported failure");

  std::filesystem::remove_all(root);
  return EXIT_SUCCESS;
}
