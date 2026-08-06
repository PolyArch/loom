#include "Mapping/Artifact/SystemMappingConstraintSet.h"
#include "ADG/Builtin.h"
#include "Common/ArtifactStore.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Dataflow/IR/DataflowReferenceCodec.h"
#include "Fabric/Artifact/FabricSystemRootView.h"
#include "Mapping/IR/MappingDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
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

template <typename T>
void requireFailureContains(llvm::Expected<T> value,
                            llvm::StringRef expectedDiagnostic,
                            const llvm::Twine &message) {
  if (value)
    fail(message);
  const std::string diagnostic = llvm::toString(value.takeError());
  if (!llvm::StringRef(diagnostic).contains(expectedDiagnostic))
    fail(message + ": " + diagnostic);
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

mlir::DenseI8ArrayAttr denseBytes(mlir::MLIRContext *context,
                                  llvm::ArrayRef<std::uint8_t> bytes) {
  std::vector<std::int8_t> signedBytes;
  signedBytes.reserve(bytes.size());
  for (std::uint8_t byte : bytes)
    signedBytes.push_back(static_cast<std::int8_t>(byte));
  return mlir::DenseI8ArrayAttr::get(context, signedBytes);
}

::mapping::ArtifactIdentityAttr
identityAttr(mlir::MLIRContext *context,
             const loom::ArtifactIdentity &identity) {
  return ::mapping::ArtifactIdentityAttr::get(
      context, denseBytes(context, identity.bytes()));
}

::mapping::RootThreadLaunchRefAttr
rootThreadLaunchAttr(mlir::MLIRContext *context,
                     const loom::ArtifactIdentity &owner,
                     dataflow::RootThreadLaunchRef reference) {
  return ::mapping::RootThreadLaunchRefAttr::get(
      context,
      denseBytes(context,
                 take(dataflow::encodeDataflowReference(owner, reference))));
}

mlir::OwningOpRef<mlir::ModuleOp> buildRawConstraintModule(
    mlir::MLIRContext &context, const loom::ArtifactIdentity &dataflowIdentity,
    const loom::ArtifactIdentity &fabricIdentity,
    llvm::ArrayRef<dataflow::RootThreadLaunchRef> rootThreadLaunches) {
  mlir::OpBuilder builder(&context);
  auto module = mlir::ModuleOp::create(builder.getUnknownLoc());
  builder.setInsertionPointToStart(module.getBody());

  std::vector<mlir::Attribute> roots;
  roots.reserve(rootThreadLaunches.size());
  for (const auto root : rootThreadLaunches)
    roots.push_back(rootThreadLaunchAttr(&context, dataflowIdentity, root));

  auto constraint = ::mapping::ConstraintsSystemOp::create(
      builder, builder.getUnknownLoc(),
      identityAttr(&context, dataflowIdentity),
      identityAttr(&context, fabricIdentity), builder.getArrayAttr(roots),
      builder.getArrayAttr({}));
  constraint.getBody().emplaceBlock();
  return module;
}

loom::CanonicalSemanticBytes rawConstraintBytes(mlir::ModuleOp module) {
  auto root =
      llvm::cast<::mapping::ConstraintsSystemOp>(module.getBody()->front());
  std::string text;
  llvm::raw_string_ostream stream(text);
  root.print(stream, mlir::OpPrintingFlags().enableDebugInfo(false));
  stream << '\n';
  stream.flush();
  return loom::CanonicalSemanticBytes(
      std::vector<std::uint8_t>(text.begin(), text.end()));
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
  std::vector<dataflow::RootThreadLaunchRef> noncanonicalAuthoring{
      secondRoot, firstRoot, secondRoot};
  auto first = take(loom::mapping::finalizeEmptySystemMappingConstraintSet(
      dataflowView, system, noncanonicalAuthoring, store));
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

  mlir::MLIRContext rawContext;
  rawContext.loadDialect<::mapping::MappingDialect>();
  auto rawModule = buildRawConstraintModule(rawContext, dataflowView.identity(),
                                            system.artifact().identity(),
                                            noncanonicalAuthoring);
  auto rawBytes = rawConstraintBytes(*rawModule);
  require(!rawBytes.bytes().equals(first.canonicalBytes().bytes()),
          "raw persisted fixture accidentally became canonical");
  auto rawIdentity =
      take(store.put(loom::mapping::mappingConstraintSetSchema, rawBytes));
  const loom::ArtifactRootReference rawReference{
      loom::mapping::mappingConstraintSetSchema.identity.str(),
      loom::mapping::mappingConstraintSetSchema.version, rawIdentity};
  requireFailureContains(
      loom::mapping::importSystemMappingConstraintSet(rawReference, store),
      "stored System constraint payload is not canonical",
      "strict import accepted persisted unsorted duplicate references");

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
