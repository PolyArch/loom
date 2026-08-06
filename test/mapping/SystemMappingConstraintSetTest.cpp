#include "Mapping/Artifact/SystemMappingConstraintSet.h"
#include "ADG/Builtin.h"
#include "Common/ArtifactStore.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Fabric/Artifact/FabricSystemRootView.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <system_error>
#include <utility>
#include <vector>

namespace {

[[noreturn]] void fail(const llvm::Twine &message) {
  llvm::errs() << "System MappingConstraintSet anchor failed: " << message
               << '\n';
  std::exit(EXIT_FAILURE);
}

template <typename T> T take(llvm::Expected<T> value) {
  if (!value)
    fail(llvm::toString(value.takeError()));
  return std::move(*value);
}

void require(bool condition, const llvm::Twine &message) {
  if (!condition)
    fail(message);
}

template <typename T>
void requireFailure(llvm::Expected<T> value, const llvm::Twine &message) {
  if (value)
    fail(message);
  llvm::consumeError(value.takeError());
}

class TemporaryDirectory final {
public:
  TemporaryDirectory() {
    std::error_code error = llvm::sys::fs::createUniqueDirectory(
        "loom-system-mapping-constraints", path_);
    if (error)
      fail("cannot create ArtifactStore directory: " + error.message());
  }

  ~TemporaryDirectory() { llvm::sys::fs::remove_directories(path_); }

  llvm::StringRef path() const { return path_; }

private:
  llvm::SmallString<128> path_;
};

mlir::MLIRContext makeContext() {
  mlir::DialectRegistry registry;
  registry.insert<dataflow::DataflowDialect, mlir::arith::ArithDialect,
                  mlir::DLTIDialect, mlir::func::FuncDialect>();
  return mlir::MLIRContext(registry, mlir::MLIRContext::Threading::DISABLED);
}

dataflow::CanonicalDataflowArtifact buildDataflow(mlir::MLIRContext &context,
                                                  int value) {
  const std::string source = R"mlir(
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 64>>} {
  dataflow.graph private @sync(%start: none, %value: i32) -> i32
      attributes {input_segments = array<i32: 1, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %result:2 = dataflow.sync %start, %value
        : (none, i32) -> (none, i32)
    dataflow.graph.return values(%result#1 : i32) streams() memories()
        complete(%result#0 : none)
  }
  dataflow.thread private @worker domain(#dataflow.thread_domain<dense>)(
      %value: i32) ctrl (%ctrl: none) {
    %result, %done = dataflow.graph.launch @sync deps(%ctrl)
        values(%value) stream_inputs() memories() stream_outputs()
        : (none, i32) -> (i32, none)
    dataflow.thread.yield %done : none
  }
  func.func private @host() {
    %value = arith.constant )mlir" +
                             std::to_string(value) + R"mlir( : i32
    %first = dataflow.thread.launch @worker(%value)
        : (i32) -> !dataflow.thread_token
    %second = dataflow.thread.launch @worker(%value)
        : (i32) -> !dataflow.thread_token
    return
  }
}
)mlir";
  auto module = mlir::parseSourceString<mlir::ModuleOp>(source, &context);
  if (!module)
    fail("cannot parse Dataflow fixture");
  return take(dataflow::finalizeCanonicalDataflow(*module));
}

} // namespace

int main() {
  TemporaryDirectory directory;
  loom::ArtifactStore store(directory.path());
  mlir::MLIRContext context = makeContext();

  auto dataflow = buildDataflow(context, 7);
  auto dataflowReference =
      take(dataflow::publishCanonicalDataflow(dataflow, store));
  (void)dataflowReference;
  auto dataflowView = take(dataflow.view());
  require(dataflowView.rootThreadLaunches().size() == 2,
          "fixture must expose two root thread launches");

  auto design = take(loom::adg::buildBuiltinTarget(
      store, loom::adg::BuiltinTargetPreset::Small));
  require(design.roots().size() == 1,
          "builtin target must publish one System root");
  auto system =
      take(loom::fabric::requireSystemRoot(design.roots().front().view()));

  const auto firstRoot = dataflowView.rootThreadLaunches()[0].ref;
  const auto secondRoot = dataflowView.rootThreadLaunches()[1].ref;
  std::vector<dataflow::RootThreadLaunchRef> reversed{secondRoot, firstRoot};
  auto first = take(loom::mapping::finalizeEmptySystemMappingConstraintSet(
      dataflowView, system, reversed, store));
  auto second = take(loom::mapping::finalizeEmptySystemMappingConstraintSet(
      dataflowView, system, {firstRoot, secondRoot}, store));

  require(first.reference() == second.reference(),
          "root authoring order changed constraint identity");
  require(
      first.canonicalBytes().bytes().equals(second.canonicalBytes().bytes()),
      "root authoring order changed canonical bytes");
  require(first.view().dataflowIdentity() == dataflowView.identity() &&
              first.view().fabricIdentity() == system.artifact().identity(),
          "constraint view lost its exact D/F bindings");
  require(first.view().rootThreadLaunches().size() == 2 &&
              first.view().rootThreadLaunches()[0] == firstRoot &&
              first.view().rootThreadLaunches()[1] == secondRoot,
          "constraint view did not preserve canonical root launches");
  require(first.view().spatialMappingReferences().empty() &&
              first.view().clauseCount() == 0,
          "empty constraints manufactured result-time mapping facts");

  auto imported = take(loom::mapping::importSystemMappingConstraintSet(
      first.reference(), store));
  require(imported.reference() == first.reference() &&
              imported.canonicalBytes().bytes().equals(
                  first.canonicalBytes().bytes()) &&
              imported.view().rootThreadLaunches() ==
                  first.view().rootThreadLaunches(),
          "strict roundtrip changed the System constraint set");

  requireFailure(loom::mapping::finalizeEmptySystemMappingConstraintSet(
                     dataflowView, system, {}, store),
                 "empty root launch coverage was accepted");

  auto foreignDataflow = buildDataflow(context, 8);
  auto foreignView = take(foreignDataflow.view());
  requireFailure(loom::mapping::finalizeEmptySystemMappingConstraintSet(
                     dataflowView, system,
                     {foreignView.rootThreadLaunches().front().ref}, store),
                 "foreign root launch reference was accepted");

  llvm::outs() << "System MappingConstraintSet anchors passed\n";
  return EXIT_SUCCESS;
}
