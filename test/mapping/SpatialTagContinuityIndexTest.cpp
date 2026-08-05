#include "Common/ArtifactStore.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/IR/FabricDialect.h"
#include "Fabric/IR/FabricOps.h"
#include "PnR/SpatialTagContinuity.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <cstdlib>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

namespace {

[[noreturn]] void fail(llvm::StringRef message) {
  llvm::errs() << "spatial tag continuity index test: " << message << '\n';
  std::exit(1);
}

template <typename T> T take(llvm::Expected<T> value) {
  if (!value)
    fail(llvm::toString(value.takeError()));
  return std::move(*value);
}

class TemporaryDirectory final {
public:
  TemporaryDirectory() {
    llvm::SmallString<128> path;
    if (std::error_code error = llvm::sys::fs::createUniqueDirectory(
            "loom-spatial-tag-continuity-index", path))
      fail("cannot create ArtifactStore directory: " + error.message());
    path_ = path.str().str();
  }

  ~TemporaryDirectory() {
    if (std::error_code error = llvm::sys::fs::remove_directories(path_))
      llvm::errs() << "cannot remove test directory: " << error.message()
                   << '\n';
  }

  llvm::StringRef path() const { return path_; }

private:
  std::string path_;
};

mlir::MLIRContext makeContext() {
  mlir::DialectRegistry registry;
  registry.insert<::fabric::FabricDialect>();
  return mlir::MLIRContext(registry, mlir::MLIRContext::Threading::DISABLED);
}

loom::fabric::FinalizedFabricRoot
buildTagBoundaryFabric(mlir::MLIRContext &context,
                       const loom::ArtifactStore &store) {
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  fabric.module @tag_boundaries(
      %dynamic_data: !fabric.bits<32>,
      %dynamic_tag: !fabric.bits<4>,
      %configured_data: !fabric.bits<16>,
      %rewrite_input: !fabric.bits_tag<8, 3>,
      %remove_input: !fabric.bits_tag<64, 5>)
      -> (!fabric.bits_tag<32, 4>, !fabric.bits_tag<16, 6>,
          !fabric.bits_tag<8, 7>, !fabric.bits<64>, !fabric.bits<5>) {
    %dynamic = fabric.boundary [s2t] %dynamic_data, %dynamic_tag
        : (!fabric.bits<32>, !fabric.bits<4>)
       -> !fabric.bits_tag<32, 4>
    %queued = fabric.fifo %dynamic [max_depth = 2, bypassable = false]
        : !fabric.bits_tag<32, 4>
    %configured = fabric.boundary [s2t] %configured_data
        : !fabric.bits<16> -> !fabric.bits_tag<16, 6>
    %rewritten = fabric.boundary [t2t] %rewrite_input
        {hw_params = [{lut_size = 5 : i32}]}
        : !fabric.bits_tag<8, 3> -> !fabric.bits_tag<8, 7>
    %removed:2 = fabric.boundary [t2s] %remove_input
        : !fabric.bits_tag<64, 5> -> (!fabric.bits<64>, !fabric.bits<5>)
    fabric.yield %queued, %configured, %rewritten, %removed#0, %removed#1
        : !fabric.bits_tag<32, 4>, !fabric.bits_tag<16, 6>,
          !fabric.bits_tag<8, 7>, !fabric.bits<64>, !fabric.bits<5>
  }
}
)mlir",
                                                        &context);
  if (!module)
    fail("cannot parse tag-continuity Fabric fixture");
  auto roots = module->getOps<::fabric::ModuleOp>();
  if (std::distance(roots.begin(), roots.end()) != 1)
    fail("tag-continuity Fabric fixture does not have one root");
  return take(loom::fabric::finalizeFabricRoot(*roots.begin(), store));
}

void frozenTagContinuityIndexIsOwnerNormalized() {
  TemporaryDirectory directory;
  loom::ArtifactStore store(directory.path());
  mlir::MLIRContext context = makeContext();
  const auto fabric = buildTagBoundaryFabric(context, store);
  const auto index =
      take(loom::pnr::freezeSpatialTagContinuityIndex(fabric.view()));

  const auto points = index.points();
  const auto boundaries = fabric.view().boundaryOccurrences();
  if (points.size() != 4 || points.size() != boundaries.size())
    fail("frozen tag-continuity index lost a boundary point");
  for (auto [ordinal, point] : llvm::enumerate(points))
    if (point.reference != boundaries[ordinal])
      fail("frozen tag-continuity index changed canonical boundary order");

  const auto traversals = fabric.view().physicalTraversals();
  const auto traversalPoints = index.traversalPointOrdinals();
  if (traversalPoints.size() != traversals.size())
    fail("frozen tag-continuity index is not traversal-dense");
  std::vector<std::uint32_t> pointUseCount(points.size(), 0);
  bool observedNonBoundary = false;
  for (auto [ordinal, traversal] : llvm::enumerate(traversals)) {
    const auto point = traversalPoints[ordinal];
    if (traversal.reference.kind() !=
        loom::fabric::FabricPhysicalTraversalKind::BoundaryTraversal) {
      observedNonBoundary = true;
      if (point != loom::pnr::getInvalidPnrIndex())
        fail("non-boundary traversal acquired a tag-continuity point");
      continue;
    }
    if (point >= points.size())
      fail("boundary traversal has no tag-continuity point");
    const auto &owner = std::get<loom::fabric::FabricBoundaryTraversalPayload>(
                            traversal.reference.payload)
                            .owner;
    if (points[point].reference != owner)
      fail("boundary traversal was assigned to a foreign continuity point");
    ++pointUseCount[point];
  }
  if (!observedNonBoundary)
    fail("tag-continuity fixture has no ordinary physical traversal");

  bool observedSplitRemover = false;
  for (auto [ordinal, point] : llvm::enumerate(points))
    if (point.kind == loom::fabric::FabricBoundaryTagContinuityKind::Remover) {
      observedSplitRemover = pointUseCount[ordinal] == 2;
      if (point.inputTagWidthBits != 5 || point.outputTagWidthBits != 0)
        fail("frozen remover changed its exact tag widths");
    }
  if (!observedSplitRemover)
    fail("split remover did not share one tag-continuity point");

  const auto domains = fabric.view().physicalTagMatchDomains();
  if (domains.size() != 1 ||
      domains.front().kind !=
          loom::fabric::FabricPhysicalTagMatchDomainKind::BoundaryLookup ||
      domains.front().ingress || domains.front().tagWidthBits != 3)
    fail("t2t boundary did not expose its exact owner-wide tag match domain");
  for (auto boundary : boundaries) {
    const auto point = fabric.view().boundaryTagContinuityPoint(boundary);
    if (!point)
      fail("boundary match-domain fixture lost a continuity point");
    const auto owner =
        loom::fabric::FabricTransportEndpointOwnerRef::of(boundary);
    for (std::uint64_t ordinal = 0;
         ordinal < fabric.view().transportEndpointCount(owner); ++ordinal) {
      const loom::fabric::FabricTransportEndpointRef endpoint{owner, ordinal};
      if (fabric.view().transportEndpointDirection(endpoint) !=
          loom::fabric::FabricPortDirection::Input)
        continue;
      const auto domain =
          fabric.view().transportEndpointTagMatchDomain(endpoint);
      const bool expected =
          point->kind ==
          loom::fabric::FabricBoundaryTagContinuityKind::Rewriter;
      if (domain.has_value() != expected || (domain && *domain != 0))
        fail("boundary input acquired the wrong tag match domain");
    }
  }

  const auto frozenDomains = index.matchDomains();
  if (!llvm::equal(frozenDomains, domains))
    fail("frozen tag match domains changed canonical Fabric order");
  const auto endpointDomains = index.endpointMatchDomainOrdinals();
  const auto endpoints = fabric.view().transportEndpoints();
  if (endpointDomains.size() != endpoints.size())
    fail("frozen tag match-domain index is not endpoint dense");
  for (auto [ordinal, endpoint] : llvm::enumerate(endpoints)) {
    const auto expected =
        fabric.view().transportEndpointTagMatchDomain(endpoint);
    const auto actual = endpointDomains[ordinal];
    if (expected) {
      if (actual != *expected || actual >= frozenDomains.size())
        fail("frozen endpoint names the wrong tag match domain");
    } else if (actual != loom::pnr::getInvalidPnrIndex()) {
      fail("transport-only endpoint acquired a frozen tag match domain");
    }
  }
}

} // namespace

int main() {
  frozenTagContinuityIndexIsOwnerNormalized();
  return 0;
}
