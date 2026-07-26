#include "Fabric/Artifact/FabricArtifact.h"

#include "Common/ArtifactFinalizer.h"
#include "Fabric/IR/FabricDialect.h"
#include "Fabric/IR/FabricOps.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

namespace {

using loom::ArtifactIdentity;
using loom::ArtifactStore;
using loom::CanonicalSemanticBytes;
using loom::fabric::FinalizedFabricRoot;

[[noreturn]] void fail(llvm::StringRef test, const std::string &message) {
  llvm::errs() << test << ": " << message << '\n';
  std::exit(EXIT_FAILURE);
}

void require(llvm::StringRef test, bool condition, const std::string &message) {
  if (!condition)
    fail(test, message);
}

void requireFabricError(llvm::StringRef test, llvm::Error error,
                        loom::fabric::FabricRefErrorKind expected) {
  if (!error)
    fail(test, "accepted an invalid finalized Fabric reference");
  const loom::fabric::FabricRefErrorKind actual =
      loom::fabric::takeFabricRefErrorKind(std::move(error));
  require(test, actual == expected,
          "finalized Fabric reference failure kind changed");
}

template <typename T> T take(llvm::StringRef test, llvm::Expected<T> value) {
  if (!value)
    fail(test, llvm::toString(value.takeError()));
  return std::move(*value);
}

template <typename T>
void expectRejected(llvm::StringRef test, llvm::Expected<T> value,
                    llvm::StringRef diagnostic) {
  if (value)
    fail(test, "accepted invalid input");
  const std::string message = llvm::toString(value.takeError());
  require(test, llvm::StringRef(message).contains(diagnostic), message);
}

class TemporaryDirectory {
public:
  explicit TemporaryDirectory(llvm::StringRef test) : test_(test.str()) {
    llvm::SmallString<128> path;
    if (std::error_code error = llvm::sys::fs::createUniqueDirectory(
            "loom-fabric-finalizer-test", path))
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

mlir::MLIRContext &context() {
  static mlir::MLIRContext *ctx = [] {
    mlir::DialectRegistry registry;
    registry.insert<::fabric::FabricDialect>();
    auto *result = new mlir::MLIRContext(registry);
    result->loadAllAvailableDialects();
    return result;
  }();
  return *ctx;
}

mlir::OwningOpRef<mlir::ModuleOp> parse(llvm::StringRef test,
                                        llvm::StringRef source) {
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(source, &context());
  if (!module)
    fail(test, "unable to parse Fabric source");
  return module;
}

::fabric::ModuleOp root(llvm::StringRef test, mlir::ModuleOp module) {
  ::fabric::ModuleOp selected;
  for (::fabric::ModuleOp candidate : module.getOps<::fabric::ModuleOp>()) {
    if (selected)
      fail(test, "fixture has more than one root");
    selected = candidate;
  }
  if (!selected)
    fail(test, "fixture has no Fabric root");
  return selected;
}

std::string moduleSource(llvm::StringRef name, bool reverse) {
  const llvm::StringLiteral first = R"mlir(
    %x = fabric.fifo %a [max_depth = 2, bypassable = true]
         : !fabric.bits<32>
    %y = fabric.boundary [s2t] %x, %ta
         : (!fabric.bits<32>, !fabric.bits<4>)
        -> !fabric.bits_tag<32, 4>
  )mlir";
  const llvm::StringLiteral second = R"mlir(
    %u = fabric.fifo %b [max_depth = 3, bypassable = false]
         : !fabric.bits<32>
    %v = fabric.boundary [s2t] %u, %tb
         : (!fabric.bits<32>, !fabric.bits<4>)
        -> !fabric.bits_tag<32, 4>
  )mlir";

  std::string source;
  llvm::raw_string_ostream stream(source);
  stream << "module { fabric.module @" << name
         << "(%a: !fabric.bits<32>, %ta: !fabric.bits<4>, "
            "%b: !fabric.bits<32>, %tb: !fabric.bits<4>) "
            "-> (!fabric.bits_tag<32, 4>, !fabric.bits_tag<32, 4>) {\n";
  if (reverse)
    stream << second << first;
  else
    stream << first << second;
  stream << "fabric.yield %y, %v : !fabric.bits_tag<32, 4>, "
            "!fabric.bits_tag<32, 4>\n} }\n";
  return stream.str();
}

void canonicalPublicationAndStrictImport() {
  const llvm::StringRef test = __func__;
  TemporaryDirectory directory(test);
  ArtifactStore store(directory.path());
  mlir::OwningOpRef<mlir::ModuleOp> first =
      parse(test, moduleSource("first", false));
  mlir::OwningOpRef<mlir::ModuleOp> second =
      parse(test, moduleSource("second", true));

  FinalizedFabricRoot firstResult =
      take(test, loom::fabric::finalizeFabricRoot(root(test, *first), store));
  FinalizedFabricRoot secondResult =
      take(test, loom::fabric::finalizeFabricRoot(root(test, *second), store));
  require(test,
          firstResult.reference().artifact == secondResult.reference().artifact,
          "source names or graph-region order changed Fabric identity");
  require(test, firstResult.directDependencies().empty(),
          "Module root retained a direct dependency");
  require(test, firstResult.view().pointConnections().size() == 2,
          "fixed FIFO-to-boundary connections were not imported");

  std::optional<loom::fabric::FabricFifoOccurrenceRef> nonBypassableFifo;
  for (const loom::fabric::FabricPhysicalTraversalRef &traversal :
       firstResult.view().admittedTraversals()) {
    if (llvm::Error error =
            loom::fabric::validateFabricRef(firstResult.view(), traversal))
      fail(test, llvm::toString(std::move(error)));
    if (traversal.kind() !=
        loom::fabric::FabricPhysicalTraversalKind::FifoTraversal)
      continue;
    const auto &fifo =
        std::get<loom::fabric::FabricFifoTraversalPayload>(traversal.payload);
    if (fifo.mode == loom::fabric::FabricFifoTraversalMode::Buffered &&
        !firstResult.view().admitsTraversal(
            loom::fabric::FabricPhysicalTraversalRef::fifoTraversal(
                fifo.owner, loom::fabric::FabricFifoTraversalMode::Bypass)))
      nonBypassableFifo = fifo.owner;
  }
  require(test, nonBypassableFifo.has_value(),
          "non-bypassable FIFO capability was not preserved");

  const loom::fabric::FabricInventoryOwnerRef fifoOwner =
      loom::fabric::FabricInventoryOwnerRef::of(*nonBypassableFifo);
  const ::fabric::ResourceContract *fifoContract =
      firstResult.view().resourceContract(fifoOwner);
  require(test,
          fifoContract && fifoContract->stateCount() == 1 &&
              fifoContract->usePatternCount() == 3,
          "finalized view did not expose the FIFO owner contract");
  requireFabricError(
      test,
      loom::fabric::validateFabricRef(
          firstResult.view(),
          loom::fabric::FabricPhysicalTraversalRef::fifoTraversal(
              *nonBypassableFifo,
              loom::fabric::FabricFifoTraversalMode::Bypass)),
      loom::fabric::FabricRefErrorKind::TraversalNotAdmitted);

  const loom::fabric::FabricPointConnectionPayload connection =
      firstResult.view().pointConnections().front();
  loom::fabric::FabricTransportEndpointRef stale = connection.destination;
  stale.ordinal = firstResult.view().transportEndpointCount(stale.owner);
  requireFabricError(test,
                     loom::fabric::validateFabricRef(firstResult.view(), stale),
                     loom::fabric::FabricRefErrorKind::OrdinalOutOfRange);

  std::vector<std::uint8_t> foreignBytes(ArtifactIdentity::byteSize, 0x5a);
  const ArtifactIdentity foreign =
      take(test, ArtifactIdentity::fromBytes(foreignBytes));
  requireFabricError(test,
                     loom::fabric::checkFabricBinding(
                         firstResult.view(),
                         loom::fabric::FabricImportBinding{
                             foreign, loom::fabric::FabricRootKind::Module}),
                     loom::fabric::FabricRefErrorKind::ForeignArtifact);

  FinalizedFabricRoot imported =
      take(test, loom::fabric::importEntireFabricRoot(firstResult.reference(),
                                                      store));
  require(test,
          imported.reference().artifact == firstResult.reference().artifact,
          "strict import changed Fabric identity");
  require(test,
          imported.canonicalBytes().bytes().equals(
              firstResult.canonicalBytes().bytes()),
          "strict import changed canonical bytes");
}

void malformedStoredPayloadIsRejected() {
  const llvm::StringRef test = __func__;
  TemporaryDirectory directory(test);
  ArtifactStore store(directory.path());
  std::vector<std::uint8_t> bytes = {'n', 'o', 't', '-', 'f',
                                     'a', 'b', 'r', 'i', 'c'};
  CanonicalSemanticBytes malformed(bytes);
  ArtifactIdentity identity =
      take(test, store.put(loom::fabric::fabricArtifactSchema, malformed));
  expectRejected(test,
                 loom::fabric::importEntireFabricRoot(
                     {loom::fabric::fabricArtifactSchema.identity.str(),
                      loom::fabric::fabricArtifactSchema.version, identity},
                     store),
                 "fabric_artifact_invalid");
}

} // namespace

int main() {
  canonicalPublicationAndStrictImport();
  malformedStoredPayloadIsRejected();
  return EXIT_SUCCESS;
}
