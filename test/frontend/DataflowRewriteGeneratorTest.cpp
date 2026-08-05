#include "ADG/Builtin.h"
#include "Common/ArtifactStore.h"
#include "DSE/CandidateGenerator.h"
#include "DSE/DataflowRewriteCandidateGenerator.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Dataflow/IR/DataflowOps.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <system_error>
#include <utility>

namespace {

[[noreturn]] void fail(const std::string &message) {
  llvm::errs() << "dataflowRewriteGenerator: " << message << '\n';
  std::exit(EXIT_FAILURE);
}

template <typename T> T take(llvm::Expected<T> value) {
  if (!value)
    fail(llvm::toString(value.takeError()));
  return std::move(*value);
}

mlir::MLIRContext &context() {
  static mlir::MLIRContext *result = [] {
    mlir::DialectRegistry registry;
    registry.insert<dataflow::DataflowDialect>();
    auto *created =
        new mlir::MLIRContext(registry, mlir::MLIRContext::Threading::DISABLED);
    created->loadAllAvailableDialects();
    return created;
  }();
  return *result;
}

dataflow::CanonicalDataflowArtifact roundTripProgram() {
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  dataflow.graph private @roundtrip(%ctrl: none) -> i24
      attributes {input_segments = array<i32: 0, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %bits = dataflow.constant %ctrl {const_value = 66051 : i24} : i24
    %lanes = dataflow.unpack %bits : i24 -> vector<3xi8>
    %restored = dataflow.pack %lanes : vector<3xi8> -> i24
    %retired:2 = dataflow.sync %ctrl, %restored
        : (none, i24) -> (none, i24)
    dataflow.graph.return values(%retired#1 : i24) streams() memories()
        complete(%retired#0 : none)
  }
}
)mlir",
                                                        &context());
  if (!module)
    fail("cannot parse the canonical rewrite fixture");
  return take(dataflow::finalizeCanonicalDataflow(module.get()));
}

std::pair<unsigned, unsigned>
adapterCounts(const dataflow::CanonicalDataflowArtifact &artifact) {
  unsigned packs = 0;
  unsigned unpacks = 0;
  artifact.module().walk([&](dataflow::PackOp) { ++packs; });
  artifact.module().walk([&](dataflow::UnpackOp) { ++unpacks; });
  return {packs, unpacks};
}

void exactParentAndOneAtomicChildArePublished() {
  llvm::SmallString<128> directory;
  std::error_code error = llvm::sys::fs::createUniqueDirectory(
      "loom-dataflow-rewrite-generator", directory);
  if (error)
    fail("cannot create ArtifactStore directory: " + error.message());
  loom::ArtifactStore store(directory);
  auto design = take(loom::adg::buildBuiltinTarget(
      store, loom::adg::BuiltinTargetPreset::Small));
  auto parent = roundTripProgram();
  const std::vector<std::uint8_t> parentBytes(
      parent.canonicalBytes().bytes().begin(),
      parent.canonicalBytes().bytes().end());
  auto parentReference =
      take(dataflow::publishCanonicalDataflow(parent, store));

  auto config =
      take(loom::dse::projectResolvedDataflowRewriteGeneratorConfigView());
  auto inputs = take(loom::dse::bindDataflowRewriteCandidateGeneratorInputs(
      {parentReference}, design.roots().front().reference()));
  auto binding =
      take(loom::dse::resolveDataflowRewriteCandidateGeneratorBinding(config));
  auto outcome =
      take(loom::dse::invokeCandidateGenerator(inputs, binding, store));
  auto *completed =
      std::get_if<loom::dse::CompletedCandidateGeneratorInvocation>(&outcome);
  if (!completed || completed->outputBindings.size() != 1 ||
      completed->outputBindings.front().artifacts.size() != 2)
    fail("generator did not publish the parent and one atomic child");

  bool sawParent = false;
  bool sawChild = false;
  for (const loom::ArtifactRootReference &reference :
       completed->outputBindings.front().artifacts) {
    auto candidate = take(dataflow::importCanonicalDataflow(reference, store));
    auto [packs, unpacks] = adapterCounts(candidate);
    if (reference == parentReference) {
      sawParent = packs == 1 && unpacks == 1;
      if (!candidate.canonicalBytes().bytes().equals(parentBytes))
        fail("generator mutated the immutable parent");
    } else {
      sawChild = packs == 0 && unpacks == 0;
    }
  }
  if (!sawParent || !sawChild)
    fail("generator omitted or malformed one rewrite identity");

  std::error_code cleanup = llvm::sys::fs::remove_directories(directory);
  if (cleanup)
    fail("cannot remove ArtifactStore directory: " + cleanup.message());
}

} // namespace

int main() {
  exactParentAndOneAtomicChildArePublished();
  return EXIT_SUCCESS;
}
