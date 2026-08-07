#include "PnR/System/SystemPnrSearchDomain.h"
#include "ADG/Builtin.h"
#include "Common/ArtifactStore.h"
#include "Config/ResolvedConfig.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Dataflow/IR/DataflowReferenceCodec.h"
#include "Fabric/Artifact/FabricSystemRootView.h"
#include "Mapping/Artifact/SystemMappingConstraintSet.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AffineMap.h"
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
void requireFailureContains(llvm::Expected<T> value,
                            llvm::StringRef expectedDiagnostic,
                            const llvm::Twine &message) {
  if (value)
    fail(message);
  const std::string diagnostic = llvm::toString(value.takeError());
  if (!llvm::StringRef(diagnostic).contains(expectedDiagnostic))
    fail(message + ": " + diagnostic);
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
    %first_result, %first_done = dataflow.graph.launch @sync deps(%ctrl)
        values(%value) stream_inputs() memories() stream_outputs()
        : (none, i32) -> (i32, none)
    %second_result, %second_done = dataflow.graph.launch @sync deps(%first_done)
        values(%value) stream_inputs() memories() stream_outputs()
        : (none, i32) -> (i32, none)
    dataflow.thread.yield %second_done : none
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
    llvm::StringRef expectedDiagnostic, const llvm::Twine &message) {
  if (value)
    fail(message);
  bool matched = false;
  llvm::Error remaining = llvm::handleErrors(
      value.takeError(),
      [&](const loom::pnr::UnsupportedSystemPnrSearchDomain &error) {
        matched = true;
        require(error.reason() == expectedReason,
                "unsupported search-domain reason changed");
        std::string diagnostic;
        llvm::raw_string_ostream stream(diagnostic);
        error.log(stream);
        stream.flush();
        require(llvm::StringRef(diagnostic).contains(expectedDiagnostic),
                "unsupported search-domain diagnostic changed");
      });
  if (remaining)
    fail(llvm::toString(std::move(remaining)));
  require(matched, message);
}

void appendU32(std::vector<std::uint8_t> &bytes, std::uint32_t value) {
  for (int shift = 24; shift >= 0; shift -= 8)
    bytes.push_back(static_cast<std::uint8_t>(value >> shift));
}

void appendU64(std::vector<std::uint8_t> &bytes, std::uint64_t value) {
  for (int shift = 56; shift >= 0; shift -= 8)
    bytes.push_back(static_cast<std::uint8_t>(value >> shift));
}

std::vector<std::uint8_t>
encodedThreadBindingKey(const loom::ArtifactIdentity &owner,
                        dataflow::RootThreadLaunchRef reference) {
  auto local = take(dataflow::encodeDataflowReference(owner, reference));
  std::vector<std::uint8_t> bytes;
  appendU32(bytes, 0);
  appendU64(bytes, local.size());
  bytes.insert(bytes.end(), local.begin(), local.end());
  return bytes;
}

std::size_t uniqueSubsequenceOffset(llvm::ArrayRef<std::uint8_t> bytes,
                                    llvm::ArrayRef<std::uint8_t> needle,
                                    const llvm::Twine &description) {
  const auto first =
      std::search(bytes.begin(), bytes.end(), needle.begin(), needle.end());
  require(first != bytes.end(), description + " is absent");
  const auto second =
      std::search(std::next(first), bytes.end(), needle.begin(), needle.end());
  require(second == bytes.end(), description + " is not unique");
  return static_cast<std::size_t>(std::distance(bytes.begin(), first));
}

} // namespace

int main() {
  TemporaryDirectory directory;
  loom::ArtifactStore store(directory.path());
  mlir::MLIRContext context = makeContext();

  loom::mapping::SystemPresburgerCell consumerCell;
  consumerCell.dimensionCount = 1;
  consumerCell.inequalities = {{1, 0}, {-1, 3}};
  const mlir::AffineExpr consumerPoint = mlir::getAffineDimExpr(0, &context);
  const mlir::AffineMap evenSourceMap =
      mlir::AffineMap::get(1, 0, consumerPoint * 2);
  const auto evenProducerPoints = take(
      loom::mapping::imageSystemPresburgerCell(consumerCell, evenSourceMap));
  loom::mapping::SystemPresburgerCell producerTwo;
  producerTwo.dimensionCount = 1;
  producerTwo.equalities = {{1, -2}};
  loom::mapping::SystemPresburgerCell producerThree;
  producerThree.dimensionCount = 1;
  producerThree.equalities = {{1, -3}};
  require(take(loom::mapping::intersectSystemPresburgerCells(evenProducerPoints,
                                                             producerTwo))
                  .has_value() &&
              !take(loom::mapping::intersectSystemPresburgerCells(
                        evenProducerPoints, producerThree))
                   .has_value(),
          "affine image convexified an exact non-unit-stride source_map");

  auto dataflow = buildDataflow(context);
  take(dataflow::publishCanonicalDataflow(dataflow, store));
  auto dataflowView = take(dataflow.view());
  require(dataflowView.rootThreadLaunches().size() == 2,
          "fixture must contain two root launches");
  require(dataflowView.staticGraphLaunches().size() == 2,
          "fixture must contain two static launches of one graph definition");
  const auto firstStaticLaunch = dataflowView.staticGraphLaunches()[0].ref;
  const auto secondStaticLaunch = dataflowView.staticGraphLaunches()[1].ref;
  const auto firstRoot = dataflowView.rootThreadLaunches()[0].ref;
  const auto secondRoot = dataflowView.rootThreadLaunches()[1].ref;
  require(firstStaticLaunch != secondStaticLaunch &&
              take(dataflowView.resolve(dataflow::RootedGraphLaunchRef{
                  firstRoot, firstStaticLaunch})) ==
                  take(dataflowView.resolve(dataflow::RootedGraphLaunchRef{
                      firstRoot, secondStaticLaunch})),
          "repeated launch sites did not retain distinct keys for one graph");
  auto logicalDomain = take(dataflowView.projectRootThreadLogicalDomain(
      dataflowView.rootThreadLaunches().front().ref));
  require(logicalDomain.coordinateRank == 1 &&
              logicalDomain.launchParameters.size() == 2,
          "Dataflow owner did not expose extent-first logical parameters");

  auto design = take(loom::adg::buildBuiltinTarget(
      store, loom::adg::BuiltinTargetPreset::Small));
  auto system =
      take(loom::fabric::requireSystemRoot(design.roots().front().view()));
  std::vector<dataflow::RootThreadLaunchRef> roots{secondRoot, firstRoot};
  auto constraints =
      take(loom::mapping::finalizeEmptySystemMappingConstraintSet(
          dataflowView, system, roots, store));
  auto duplicateRootConstraints =
      take(loom::mapping::finalizeEmptySystemMappingConstraintSet(
          dataflowView, system, {secondRoot, firstRoot, secondRoot}, store));
  require(duplicateRootConstraints.reference() == constraints.reference() &&
              duplicateRootConstraints.view().rootThreadLaunches() ==
                  constraints.view().rootThreadLaunches(),
          "duplicate root authoring changed the canonical constraint input");
  auto plan = take(loom::pnr::projectWholeDomainPresburgerPartitionPlan(
      dataflowView, constraints.view().rootThreadLaunches()));
  const auto config = take(loom::pnr::projectResolvedSystemPnrConfigView(
      loom::defaultResolvedConfig()));
  auto domain = take(loom::pnr::projectSystemPnrSearchDomain(
      dataflowView, system, config, constraints, plan, {}, store));

  require(domain.rootThreadLaunches().size() == 2 &&
              domain.bindings().size() == 6,
          "search domain lost a root or rooted graph binding");
  std::vector<dataflow::RootedGraphLaunchRef> expectedGraphKeys;
  dataflowView.forEachRootedGraphLaunch(
      [&](dataflow::RootedGraphLaunchRef reference) {
        expectedGraphKeys.push_back(reference);
      });
  require(expectedGraphKeys.size() == 4,
          "fixture did not expose four exact rooted launch keys");
  std::size_t threadBindings = 0;
  std::size_t graphBindings = 0;
  std::vector<dataflow::RootedGraphLaunchRef> actualGraphKeys;
  for (const loom::pnr::SystemSearchBindingDomain &binding :
       domain.bindings()) {
    require(binding.atoms.size() == 1,
            "whole-domain plan did not produce one atom per binding");
    const auto &targets = binding.atoms.front().domain;
    if (std::holds_alternative<dataflow::RootThreadLaunchRef>(binding.key)) {
      ++threadBindings;
      const auto *thread =
          std::get_if<loom::pnr::SystemThreadBindingDomain>(&targets);
      require(thread && thread->compatibleAccCores.size() ==
                            system.artifact().accCoreOccurrences().size(),
              "thread atom target domain is incomplete or ill-typed");
    } else {
      ++graphBindings;
      actualGraphKeys.push_back(
          std::get<dataflow::RootedGraphLaunchRef>(binding.key));
      const auto *graph =
          std::get_if<loom::pnr::SystemHierarchicalGraphBindingDomain>(
              &targets);
      require(graph && graph->compatibleSpatialMappings.empty(),
              "graph atom target domain is incomplete or ill-typed");
    }
  }
  require(threadBindings == 2 && graphBindings == 4,
          "binding-key variants changed their complete coverage");
  for (const auto &expected : expectedGraphKeys)
    require(llvm::count(actualGraphKeys, expected) == 1,
            "H lost or merged an exact RootedGraphLaunchRef binding key");

  std::reverse(plan.bindings.begin(), plan.bindings.end());
  auto reordered = take(loom::pnr::projectSystemPnrSearchDomain(
      dataflowView, system, config, constraints, plan, {}, store));
  require(reordered.canonicalViewBytes() == domain.canonicalViewBytes() &&
              reordered.digest() == domain.digest(),
          "partition authoring order changed canonical H");

  auto redundant = take(loom::pnr::projectWholeDomainPresburgerPartitionPlan(
      dataflowView, roots));
  auto &redundantRows = redundant.bindings.front().cells.front().inequalities;
  std::reverse(redundantRows.begin(), redundantRows.end());
  redundantRows.push_back(redundantRows.front());
  auto normalized = take(loom::pnr::projectSystemPnrSearchDomain(
      dataflowView, system, config, constraints, redundant, {}, store));
  require(normalized.canonicalViewBytes() == domain.canonicalViewBytes() &&
              normalized.digest() == domain.digest(),
          "redundant or reordered Presburger constraints changed canonical H");

  auto parity = take(loom::pnr::projectWholeDomainPresburgerPartitionPlan(
      dataflowView, roots));
  auto even = parity.bindings.front().cells.front();
  even.localCount = 1;
  const std::size_t localColumn =
      static_cast<std::size_t>(even.dimensionCount) + even.symbolCount;
  for (auto &row : even.equalities)
    row.insert(row.begin() + localColumn, 0);
  for (auto &row : even.inequalities)
    row.insert(row.begin() + localColumn, 0);
  std::vector<std::int64_t> congruence(localColumn + even.localCount + 1, 0);
  congruence.front() = 1;
  congruence[localColumn] = -2;
  even.equalities.push_back(congruence);
  auto odd = even;
  odd.equalities.back().back() = -1;
  parity.bindings.front().cells = {odd, even};
  auto parityDomain = take(loom::pnr::projectSystemPnrSearchDomain(
      dataflowView, system, config, constraints, parity, {}, store));
  const auto &parityAtoms = parityDomain.bindings().front().atoms;
  require(
      parityAtoms.size() == 2 && llvm::all_of(parityAtoms,
                                              [](const auto &atom) {
                                                return atom.cell.localCount ==
                                                       1;
                                              }),
      "Presburger local variables were lost from an exact parity partition");
  auto adoptedParity = take(loom::pnr::adoptSystemPnrSearchDomain(
      loom::pnr::systemPnrSearchDomainSchemaDescriptorBytes(),
      parityDomain.canonicalViewBytes(), parityDomain.digest(), store));
  require(
      adoptedParity.canonicalViewBytes() == parityDomain.canonicalViewBytes() &&
          adoptedParity.bindings().front().atoms.front().cell.localCount == 1,
      "strict H adoption changed a Presburger local-variable partition");

  auto adopted = take(loom::pnr::adoptSystemPnrSearchDomain(
      loom::pnr::systemPnrSearchDomainSchemaDescriptorBytes(),
      domain.canonicalViewBytes(), domain.digest(), store));
  require(adopted.canonicalViewBytes() == domain.canonicalViewBytes() &&
              adopted.bindings().size() == domain.bindings().size(),
          "strict H adoption changed the owner view");

  auto duplicatedBindingBytes = std::vector<std::uint8_t>(
      domain.canonicalViewBytes().begin(), domain.canonicalViewBytes().end());
  const auto firstThreadKey =
      encodedThreadBindingKey(dataflowView.identity(), firstRoot);
  const auto secondThreadKey =
      encodedThreadBindingKey(dataflowView.identity(), secondRoot);
  require(firstThreadKey.size() == secondThreadKey.size(),
          "thread binding keys do not have equal replacement width");
  const std::size_t secondThreadOffset = uniqueSubsequenceOffset(
      duplicatedBindingBytes, secondThreadKey, "second thread binding key");
  std::copy(firstThreadKey.begin(), firstThreadKey.end(),
            duplicatedBindingBytes.begin() + secondThreadOffset);
  auto duplicatedBindingDigest =
      take(loom::pnr::computeSystemPnrSearchDomainDigest(
          loom::pnr::systemPnrSearchDomainSchemaDescriptorBytes(),
          duplicatedBindingBytes));
  requireFailureContains(
      loom::pnr::adoptSystemPnrSearchDomain(
          loom::pnr::systemPnrSearchDomainSchemaDescriptorBytes(),
          duplicatedBindingBytes, duplicatedBindingDigest, store),
      "partition plan contains a duplicate binding",
      "strict H adoption accepted a duplicate persisted binding key");

  auto badDigestBytes = domain.digest().bytes();
  badDigestBytes[0] ^= 1;
  auto badDigest =
      take(loom::pnr::SystemPnrSearchDomainDigest::fromBytes(badDigestBytes));
  requireFailureContains(
      loom::pnr::adoptSystemPnrSearchDomain(
          loom::pnr::systemPnrSearchDomainSchemaDescriptorBytes(),
          domain.canonicalViewBytes(), badDigest, store),
      "digest does not match canonical view bytes",
      "modified H digest was accepted");

  auto integerEmpty = take(loom::pnr::projectWholeDomainPresburgerPartitionPlan(
      dataflowView, constraints.view().rootThreadLaunches()));
  auto &emptyCell = integerEmpty.bindings.front().cells.front();
  std::vector<std::int64_t> noIntegerPoint(
      static_cast<std::size_t>(emptyCell.dimensionCount) +
          emptyCell.symbolCount + emptyCell.localCount + 1,
      0);
  noIntegerPoint.front() = 2;
  noIntegerPoint.back() = -1;
  emptyCell.equalities.push_back(std::move(noIntegerPoint));
  requireFailureContains(
      loom::pnr::projectSystemPnrSearchDomain(
          dataflowView, system, config, constraints, integerEmpty, {}, store),
      "Presburger cell is integer-empty",
      "integer-empty Presburger cell was accepted");

  auto overlap = take(loom::pnr::projectWholeDomainPresburgerPartitionPlan(
      dataflowView, constraints.view().rootThreadLaunches()));
  overlap.bindings.front().cells.push_back(overlap.bindings.front().cells[0]);
  requireFailureContains(
      loom::pnr::projectSystemPnrSearchDomain(dataflowView, system, config,
                                              constraints, overlap, {}, store),
      "Presburger partition cells overlap",
      "overlapping Presburger cells were accepted");

  auto gap = take(loom::pnr::projectWholeDomainPresburgerPartitionPlan(
      dataflowView, constraints.view().rootThreadLaunches()));
  gap.bindings.front().cells.front().inequalities.push_back({1, 0, 0, -1});
  requireFailureContains(
      loom::pnr::projectSystemPnrSearchDomain(dataflowView, system, config,
                                              constraints, gap, {}, store),
      "Presburger partition does not cover the Dataflow may-domain",
      "gapped Presburger partition was accepted");

  auto conditional = buildConditionalDataflow(context);
  take(dataflow::publishCanonicalDataflow(conditional, store));
  auto conditionalView = take(conditional.view());
  auto conditionalRoot = conditionalView.rootThreadLaunches().front().ref;
  std::vector<dataflow::RootedGraphLaunchRef> nestedGraphKeys;
  conditionalView.forEachRootedGraphLaunch(
      [&](dataflow::RootedGraphLaunchRef reference) {
        nestedGraphKeys.push_back(reference);
      });
  require(nestedGraphKeys.size() == 1,
          "conditional fixture lost its nested rooted graph launch");
  const loom::pnr::SystemSearchBindingKey nestedKey(nestedGraphKeys.front());
  require(std::holds_alternative<dataflow::RootedGraphLaunchRef>(nestedKey) &&
              std::get<dataflow::RootedGraphLaunchRef>(nestedKey) ==
                  nestedGraphKeys.front(),
          "nested rooted launch did not preserve its exact binding-key kind");
  std::vector<dataflow::RootThreadLaunchRef> conditionalRoots{conditionalRoot};
  requireUnsupported(
      loom::pnr::projectWholeDomainPresburgerPartitionPlan(conditionalView,
                                                           conditionalRoots),
      loom::pnr::UnsupportedSystemPnrSearchDomainReason::
          RootedGraphMayDomainProjectionUnavailable,
      "does not publish the exact may-domain of a nested or repeated rooted "
      "graph launch",
      "conditional graph launch reused the complete parent thread domain");

  llvm::outs() << "System PnR search-domain anchors passed\n";
  return EXIT_SUCCESS;
}
