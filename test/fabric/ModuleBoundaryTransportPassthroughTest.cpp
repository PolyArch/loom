#include "Fabric/Artifact/FabricArtifact.h"
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
#include <system_error>
#include <utility>

namespace {

[[noreturn]] void fail(llvm::StringRef test, const llvm::Twine &message) {
  llvm::errs() << test << ": " << message << '\n';
  std::exit(EXIT_FAILURE);
}

void require(llvm::StringRef test, bool condition, const llvm::Twine &message) {
  if (!condition)
    fail(test, message);
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
    fail(test, "accepted an invalid direct Module boundary passthrough");
  const std::string message = llvm::toString(value.takeError());
  require(test, llvm::StringRef(message).contains(diagnostic), message);
}

class TemporaryDirectory {
public:
  explicit TemporaryDirectory(llvm::StringRef test) : test_(test.str()) {
    llvm::SmallString<128> path;
    if (std::error_code error = llvm::sys::fs::createUniqueDirectory(
            "loom-fabric-module-passthrough-test", path))
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
    auto *result =
        new mlir::MLIRContext(registry, mlir::MLIRContext::Threading::DISABLED);
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

mlir::OwningOpRef<mlir::ModuleOp> parseUnchecked(llvm::StringRef test,
                                                 llvm::StringRef source) {
  mlir::ParserConfig config(&context(), false);
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(source, config);
  if (!module)
    fail(test, "unable to parse unchecked Fabric source");
  return module;
}

::fabric::ModuleOp root(llvm::StringRef test, mlir::ModuleOp module) {
  ::fabric::ModuleOp selected;
  for (::fabric::ModuleOp candidate : module.getOps<::fabric::ModuleOp>()) {
    if (selected)
      fail(test, "fixture has more than one Fabric Module root");
    selected = candidate;
  }
  if (!selected)
    fail(test, "fixture has no Fabric Module root");
  return selected;
}

void directTokenPassthroughsAreSealedAndReimported() {
  constexpr llvm::StringLiteral test = __func__;
  TemporaryDirectory directory(test);
  loom::ArtifactStore store(directory.path());
  auto source = parse(test, R"mlir(
    module {
      fabric.module @passthrough(
          %data: !fabric.bits<32>,
          %unused: memref<8xi32>,
          %tagged: !fabric.bits_tag<4, 5>,
          %spare: !fabric.bits<8>)
          -> (!fabric.bits<16>, !fabric.bits_tag<0, 3>) {
        fabric.yield %data : !fabric.bits<32> to !fabric.bits<16>,
                     %tagged : !fabric.bits_tag<4, 5> to !fabric.bits_tag<0, 3>
      }
    }
  )mlir");

  loom::fabric::FinalizedFabricRoot finalized =
      take(test, loom::fabric::finalizeFabricRoot(root(test, *source), store));
  const auto passthroughs =
      finalized.view().moduleBoundaryTransportPassthroughs();
  require(test, passthroughs.size() == 2,
          "sealed view omitted a direct Module boundary passthrough");

  const loom::fabric::FabricModuleTemplateRef module =
      passthroughs.front().input.module;
  require(test,
          passthroughs[0].input ==
                  loom::fabric::FabricModuleBoundaryEndpointRef{
                      module, loom::fabric::FabricPortDirection::Input, 0} &&
              passthroughs[0].output ==
                  loom::fabric::FabricModuleBoundaryEndpointRef{
                      module, loom::fabric::FabricPortDirection::Output, 0},
          "bits passthrough changed its signature ordinals");
  require(test,
          passthroughs[1].input ==
                  loom::fabric::FabricModuleBoundaryEndpointRef{
                      module, loom::fabric::FabricPortDirection::Input, 2} &&
              passthroughs[1].output ==
                  loom::fabric::FabricModuleBoundaryEndpointRef{
                      module, loom::fabric::FabricPortDirection::Output, 1},
          "bits_tag passthrough collapsed a memory signature hole");
  require(test, finalized.view().moduleBoundaryTransportAttachments().empty(),
          "direct passthrough became a resource attachment");
  require(test, finalized.view().pointConnections().empty(),
          "direct passthrough became a routable point connection");
  require(test, finalized.view().physicalTraversals().empty(),
          "direct passthrough became a physical traversal");

  loom::fabric::FinalizedFabricRoot reimported = take(
      test, loom::fabric::importEntireFabricRoot(finalized.reference(), store));
  require(test,
          reimported.view().moduleBoundaryTransportPassthroughs() ==
              passthroughs,
          "strict import changed Module boundary passthroughs");
}

void invalidDirectPassthroughsFailBeforePublication() {
  constexpr llvm::StringLiteral test = __func__;
  TemporaryDirectory directory(test);
  loom::ArtifactStore store(directory.path());

  auto fanout = parseUnchecked(test, R"mlir(
    module {
      fabric.module @fanout(%value: !fabric.bits<8>)
          -> (!fabric.bits<8>, !fabric.bits<8>) {
        fabric.yield %value, %value : !fabric.bits<8>, !fabric.bits<8>
      }
    }
  )mlir");
  expectRejected(test,
                 loom::fabric::finalizeFabricRoot(root(test, *fanout), store),
                 "the authoring module does not verify");

  auto crossKind = parseUnchecked(test, R"mlir(
    module {
      fabric.module @cross_kind(%value: !fabric.bits_tag<8, 2>)
          -> !fabric.bits<8> {
        fabric.yield %value : !fabric.bits_tag<8, 2> to !fabric.bits<8>
      }
    }
  )mlir");
  expectRejected(
      test, loom::fabric::finalizeFabricRoot(root(test, *crossKind), store),
      "the authoring module does not verify");

  auto memory = parseUnchecked(test, R"mlir(
    module {
      fabric.module @memory(%value: memref<8xi32>) -> memref<8xi32> {
        fabric.yield %value : memref<8xi32>
      }
    }
  )mlir");
  expectRejected(test,
                 loom::fabric::finalizeFabricRoot(root(test, *memory), store),
                 "the authoring module does not verify");
}

} // namespace

int main() {
  directTokenPassthroughsAreSealedAndReimported();
  invalidDirectPassthroughsFailBeforePublication();
  return EXIT_SUCCESS;
}
