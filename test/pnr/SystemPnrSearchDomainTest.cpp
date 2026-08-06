#include "PnR/System/SystemPnrSearchDomain.h"
#include "ADG/Builtin.h"
#include "Common/ArtifactStore.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Fabric/Artifact/FabricSystemRootView.h"
#include "Mapping/Artifact/SystemMappingConstraintSet.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdlib>
#include <system_error>
#include <utility>
#include <variant>
#include <vector>

namespace {

[[noreturn]] void fail(const llvm::Twine &message) {
  llvm::errs() << "System PnR search-domain anchor failed: " << message << '\n';
  std::exit(EXIT_FAILURE);
}

template <typename T> T take(llvm::Expected<T> value) {
  if (!value)
    fail(llvm::toString(value.takeError()));
  return std::move(*value);
}

template <typename T>
void requireFailure(llvm::Expected<T> value, const llvm::Twine &message) {
  if (value)
    fail(message);
  llvm::consumeError(value.takeError());
}

void require(bool condition, const llvm::Twine &message) {
  if (!condition)
    fail(message);
}

class TemporaryDirectory final {
public:
  TemporaryDirectory() {
    std::error_code error = llvm::sys::fs::createUniqueDirectory(
        "loom-system-pnr-search-domain", path_);
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
                  mlir::DLTIDialect, mlir::func::FuncDialect,
                  mlir::scf::SCFDialect>();
  return mlir::MLIRContext(registry, mlir::MLIRContext::Threading::DISABLED);
}

dataflow::CanonicalDataflowArtifact buildDataflow(mlir::MLIRContext &context) {
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
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
      %value: i32) ctrl (%ctrl: none) iv (%iv: index) {
    %result, %done = dataflow.graph.launch @sync deps(%ctrl)
        values(%value) stream_inputs() memories() stream_outputs()
        : (none, i32) -> (i32, none)
    dataflow.thread.yield %done : none
  }
  func.func private @host() {
    %value = arith.constant 7 : i32
    %extent = arith.constant 8 : index
    %first = dataflow.thread.launch @worker(%value) grid(%extent)
        : (i32) -> !dataflow.thread_token
    %second = dataflow.thread.launch @worker(%value) grid(%extent)
        : (i32) -> !dataflow.thread_token
    return
  }
}
)mlir",
                                                        &context);
  if (!module)
    fail("cannot parse Dataflow fixture");
  return take(dataflow::finalizeCanonicalDataflow(*module));
}

dataflow::CanonicalDataflowArtifact
buildConditionalDataflow(mlir::MLIRContext &context) {
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
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
      %condition: i1, %value: i32) ctrl (%ctrl: none) iv (%iv: index) {
    %frontier = scf.if %condition -> (none) {
      %result, %done = dataflow.graph.launch @sync deps(%ctrl)
          values(%value) stream_inputs() memories() stream_outputs()
          : (none, i32) -> (i32, none)
      scf.yield %done : none
    } else {
      scf.yield %ctrl : none
    }
    dataflow.thread.yield %frontier : none
  }
  func.func private @host() {
    %condition = arith.constant true
    %value = arith.constant 7 : i32
    %extent = arith.constant 8 : index
    %completion = dataflow.thread.launch @worker(%condition, %value)
        grid(%extent) : (i1, i32) -> !dataflow.thread_token
    return
  }
}
)mlir",
                                                        &context);
  if (!module)
    fail("cannot parse conditional Dataflow fixture");
  return take(dataflow::finalizeCanonicalDataflow(*module));
}

template <typename T>
void requireUnsupported(
    llvm::Expected<T> value,
    loom::pnr::UnsupportedSystemPnrSearchDomainReason expectedReason,
    const llvm::Twine &message) {
  if (value)
    fail(message);
  bool matched = false;
  llvm::Error remaining = llvm::handleErrors(
      value.takeError(),
      [&](const loom::pnr::UnsupportedSystemPnrSearchDomain &error) {
        matched = true;
        require(error.reason() == expectedReason,
                "unsupported search-domain reason changed");
      });
  if (remaining)
    fail(llvm::toString(std::move(remaining)));
  require(matched, message);
}

} // namespace

int main() {
  TemporaryDirectory directory;
  loom::ArtifactStore store(directory.path());
  mlir::MLIRContext context = makeContext();

  auto dataflow = buildDataflow(context);
  take(dataflow::publishCanonicalDataflow(dataflow, store));
  auto dataflowView = take(dataflow.view());
  require(dataflowView.rootThreadLaunches().size() == 2,
          "fixture must contain two root launches");
  auto logicalDomain = take(dataflowView.projectRootThreadLogicalDomain(
      dataflowView.rootThreadLaunches().front().ref));
  require(logicalDomain.coordinateRank == 1 &&
              logicalDomain.launchParameters.size() == 2,
          "Dataflow owner did not expose extent-first logical parameters");

  auto design = take(loom::adg::buildBuiltinTarget(
      store, loom::adg::BuiltinTargetPreset::Small));
  auto system =
      take(loom::fabric::requireSystemRoot(design.roots().front().view()));
  std::vector<dataflow::RootThreadLaunchRef> roots{
      dataflowView.rootThreadLaunches()[1].ref,
      dataflowView.rootThreadLaunches()[0].ref};
  auto constraints =
      take(loom::mapping::finalizeEmptySystemMappingConstraintSet(
          dataflowView, system, roots, store));
  auto plan = take(loom::pnr::projectWholeDomainPresburgerPartitionPlan(
      dataflowView, roots));
  auto domain = take(loom::pnr::projectSystemPnrSearchDomain(
      dataflowView, system, constraints, plan, {}, store));

  require(domain.rootThreadLaunches().size() == 2 &&
              domain.bindings().size() == 4,
          "search domain lost a root or rooted graph binding");
  std::size_t threadBindings = 0;
  std::size_t graphBindings = 0;
  for (const loom::pnr::SystemSearchBindingDomain &binding :
       domain.bindings()) {
    require(binding.atoms.size() == 1,
            "whole-domain plan did not produce one atom per binding");
    const auto &targets = binding.atoms.front().domains;
    if (std::holds_alternative<dataflow::RootThreadLaunchRef>(binding.key)) {
      ++threadBindings;
      require(targets.compatibleAccCores &&
                  targets.compatibleAccCores->size() ==
                      system.artifact().accCoreOccurrences().size() &&
                  !targets.compatibleSpatialMappings,
              "thread atom target domain is incomplete or ill-typed");
    } else {
      ++graphBindings;
      require(targets.compatibleSpatialMappings &&
                  targets.compatibleSpatialMappings->empty() &&
                  !targets.compatibleAccCores,
              "graph atom target domain is incomplete or ill-typed");
    }
  }
  require(threadBindings == 2 && graphBindings == 2,
          "binding-key variants changed their complete coverage");

  std::reverse(plan.bindings.begin(), plan.bindings.end());
  auto reordered = take(loom::pnr::projectSystemPnrSearchDomain(
      dataflowView, system, constraints, plan, {}, store));
  require(reordered.canonicalViewBytes() == domain.canonicalViewBytes() &&
              reordered.digest() == domain.digest(),
          "partition authoring order changed canonical H");

  auto redundant = take(loom::pnr::projectWholeDomainPresburgerPartitionPlan(
      dataflowView, roots));
  auto &redundantRows = redundant.bindings.front().cells.front().inequalities;
  std::reverse(redundantRows.begin(), redundantRows.end());
  redundantRows.push_back(redundantRows.front());
  auto normalized = take(loom::pnr::projectSystemPnrSearchDomain(
      dataflowView, system, constraints, redundant, {}, store));
  require(normalized.canonicalViewBytes() == domain.canonicalViewBytes() &&
              normalized.digest() == domain.digest(),
          "redundant or reordered Presburger constraints changed canonical H");

  auto adopted = take(loom::pnr::adoptSystemPnrSearchDomain(
      loom::pnr::systemPnrSearchDomainSchemaDescriptorBytes(),
      domain.canonicalViewBytes(), domain.digest(), store));
  require(adopted.canonicalViewBytes() == domain.canonicalViewBytes() &&
              adopted.bindings().size() == domain.bindings().size(),
          "strict H adoption changed the owner view");

  auto badDigestBytes = domain.digest().bytes();
  badDigestBytes[0] ^= 1;
  auto badDigest =
      take(loom::pnr::SystemPnrSearchDomainDigest::fromBytes(badDigestBytes));
  requireFailure(loom::pnr::adoptSystemPnrSearchDomain(
                     loom::pnr::systemPnrSearchDomainSchemaDescriptorBytes(),
                     domain.canonicalViewBytes(), badDigest, store),
                 "modified H digest was accepted");

  auto overlap = take(loom::pnr::projectWholeDomainPresburgerPartitionPlan(
      dataflowView, roots));
  overlap.bindings.front().cells.push_back(overlap.bindings.front().cells[0]);
  requireFailure(loom::pnr::projectSystemPnrSearchDomain(
                     dataflowView, system, constraints, overlap, {}, store),
                 "overlapping Presburger cells were accepted");

  auto gap = take(loom::pnr::projectWholeDomainPresburgerPartitionPlan(
      dataflowView, roots));
  gap.bindings.front().cells.front().inequalities.push_back({1, 0, 0, -1});
  requireFailure(loom::pnr::projectSystemPnrSearchDomain(
                     dataflowView, system, constraints, gap, {}, store),
                 "gapped Presburger partition was accepted");

  auto conditional = buildConditionalDataflow(context);
  take(dataflow::publishCanonicalDataflow(conditional, store));
  auto conditionalView = take(conditional.view());
  auto conditionalRoot = conditionalView.rootThreadLaunches().front().ref;
  std::vector<dataflow::RootThreadLaunchRef> conditionalRoots{conditionalRoot};
  requireUnsupported(
      loom::pnr::projectWholeDomainPresburgerPartitionPlan(conditionalView,
                                                           conditionalRoots),
      loom::pnr::UnsupportedSystemPnrSearchDomainReason::
          RootedGraphMayDomainProjectionUnavailable,
      "conditional graph launch reused the complete parent thread domain");

  llvm::outs() << "System PnR search-domain anchors passed\n";
  return EXIT_SUCCESS;
}
