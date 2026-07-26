#include "ADG/Builder.h"

#include "Common/ArtifactStore.h"
#include "Fabric/Identity/FabricRefs.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <string>
#include <utility>

namespace {

[[noreturn]] void fail(llvm::StringRef test, const std::string &message) {
  llvm::errs() << test << ": " << message << '\n';
  std::exit(EXIT_FAILURE);
}

void require(llvm::StringRef test, bool condition, llvm::StringRef message) {
  if (!condition)
    fail(test, message.str());
}

template <typename T> T take(llvm::StringRef test, llvm::Expected<T> value) {
  if (!value)
    fail(test, llvm::toString(value.takeError()));
  return std::move(*value);
}

void expectError(llvm::StringRef test, llvm::Error error,
                 llvm::StringRef diagnostic) {
  if (!error)
    fail(test, "accepted invalid ADG authoring");
  const std::string message = llvm::toString(std::move(error));
  require(test, llvm::StringRef(message).contains(diagnostic), message);
}

template <typename T>
void expectError(llvm::StringRef test, llvm::Expected<T> value,
                 llvm::StringRef diagnostic) {
  if (value)
    fail(test, "accepted invalid ADG authoring");
  expectError(test, value.takeError(), diagnostic);
}

class TemporaryDirectory {
public:
  explicit TemporaryDirectory(llvm::StringRef test) : test_(test.str()) {
    llvm::SmallString<128> path;
    if (std::error_code error =
            llvm::sys::fs::createUniqueDirectory("loom-adg-builder", path))
      fail(test, error.message());
    path_ = path.str().str();
  }

  ~TemporaryDirectory() {
    if (std::error_code error = llvm::sys::fs::remove_directories(path_))
      llvm::errs() << test_ << ": unable to remove temporary directory: "
                   << error.message() << '\n';
  }

  llvm::StringRef path() const { return path_; }

private:
  std::string test_;
  std::string path_;
};

using loom::adg::BoundarySpec;
using loom::adg::DesignBuilder;
using loom::adg::FifoSpec;
using loom::adg::PortType;
using loom::adg::SpatialValue;

void regularAndIrregularSpatialCoresFinalize() {
  const llvm::StringRef test = __func__;
  TemporaryDirectory directory(test);
  loom::ArtifactStore store(directory.path());
  DesignBuilder design(store);

  const PortType bits4 = take(test, PortType::bits(4));
  const PortType bits32 = take(test, PortType::bits(32));
  const PortType bits64 = take(test, PortType::bits(64));
  const PortType tagged32x4 = take(test, PortType::taggedBits(32, 4));

  auto regular = take(
      test, design.createSpatialCore("regular", {bits32, bits4}, {tagged32x4}));
  SpatialValue regularData = take(test, regular.input(0));
  SpatialValue regularTag = take(test, regular.input(1));
  auto regularBoundary = take(
      test, regular.addBoundary({regularData, regularTag},
                                BoundarySpec::s2t(bits32, bits4, tagged32x4)));
  SpatialValue regularQueued =
      take(test, regular.addFifo(regularBoundary.front(),
                                 FifoSpec{tagged32x4, 2, true}));
  if (llvm::Error error = regular.close({regularQueued}))
    fail(test, llvm::toString(std::move(error)));

  auto irregular = take(test, design.createSpatialCore(
                                  "irregular", {bits64, bits4}, {tagged32x4}));
  SpatialValue irregularData = take(test, irregular.input(0));
  SpatialValue irregularTag = take(test, irregular.input(1));
  SpatialValue narrowed =
      take(test, irregular.addFifo(irregularData, FifoSpec{bits32, 3, false}));
  auto irregularBoundary =
      take(test,
           irregular.addBoundary({narrowed, irregularTag},
                                 BoundarySpec::s2t(bits32, bits4, tagged32x4)));
  if (llvm::Error error = irregular.close({irregularBoundary.front()}))
    fail(test, llvm::toString(std::move(error)));

  loom::adg::FinalizedFabricDesign finalized =
      take(test, std::move(design).finalize());
  require(test, finalized.roots().size() == 2,
          "finalized design did not contain both SpatialCore roots");
  for (const loom::fabric::FinalizedFabricRoot &root : finalized.roots()) {
    require(test,
            root.view().rootKind() == loom::fabric::FabricRootKind::Module,
            "SpatialCore finalized with a non-Module root kind");
    require(test, !root.view().admittedTraversals().empty(),
            "SpatialCore lost its physical traversal inventory");
  }
}

void foreignHandlesAndIncompleteRootsFailClosed() {
  const llvm::StringRef test = __func__;
  TemporaryDirectory directory(test);
  loom::ArtifactStore store(directory.path());
  DesignBuilder design(store);
  const PortType bits32 = take(test, PortType::bits(32));

  auto first =
      take(test, design.createSpatialCore("first", {bits32}, {bits32}));
  auto second =
      take(test, design.createSpatialCore("second", {bits32}, {bits32}));
  SpatialValue foreign = take(test, first.input(0));
  expectError(test, second.addFifo(foreign, FifoSpec{bits32, 1, false}),
              "foreign SpatialValue");

  if (llvm::Error error = first.close({foreign}))
    fail(test, llvm::toString(std::move(error)));
  expectError(test, std::move(design).finalize(),
              "SpatialCore 'second' is not closed");
}

} // namespace

int main() {
  regularAndIrregularSpatialCoresFinalize();
  foreignHandlesAndIncompleteRootsFailClosed();
  return EXIT_SUCCESS;
}
