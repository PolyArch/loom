#include "ADG/Builtin.h"
#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "Config/ResolvedConfig.h"
#include "DSE/RootCompleteTechMappingCandidateGenerator.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Mapping/Artifact/MappingArtifact.h"
#include "Mapping/Tech/TechMappingConfig.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <system_error>
#include <utility>
#include <variant>

namespace {

[[noreturn]] void fail(const llvm::Twine &message) {
  llvm::errs() << "root-complete TechMapping generator anchor failed: "
               << message << '\n';
  std::exit(EXIT_FAILURE);
}

template <typename T> T take(llvm::Expected<T> value) {
  if (!value)
    fail(llvm::toString(value.takeError()));
  return std::move(*value);
}

struct ImmediateStopSource final {
  static bool query(const void *) { return true; }

  loom::ExecutionControlView control() const { return {this, query}; }
};

class TemporaryDirectory final {
public:
  TemporaryDirectory() {
    std::error_code error = llvm::sys::fs::createUniqueDirectory(
        "loom-root-complete-tech-mapping", path_);
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

dataflow::CanonicalDataflowArtifact
buildRootedDataflow(mlir::MLIRContext &context) {
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
  dataflow.graph private @sync_second(%start: none, %value: i32) -> i32
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
    %second, %second_done = dataflow.graph.launch @sync_second deps(%done)
        values(%result) stream_inputs() memories() stream_outputs()
        : (none, i32) -> (i32, none)
    dataflow.thread.yield %second_done : none
  }
  func.func private @host() {
    %value = arith.constant 7 : i32
    %thread = dataflow.thread.launch @worker(%value)
        : (i32) -> !dataflow.thread_token
    return
  }
}
)mlir",
                                                        &context);
  if (!module)
    fail("cannot parse rooted Dataflow fixture");
  return take(dataflow::finalizeCanonicalDataflow(*module));
}

dataflow::CanonicalDataflowArtifact
buildSingleGraphDataflow(mlir::MLIRContext &context) {
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 64>>} {
  dataflow.graph private @sync(%start: none, %value: i64) -> i64
      attributes {input_segments = array<i32: 1, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %result:2 = dataflow.sync %start, %value
        : (none, i64) -> (none, i64)
    dataflow.graph.return values(%result#1 : i64) streams() memories()
        complete(%result#0 : none)
  }
  dataflow.thread private @worker domain(#dataflow.thread_domain<dense>)(
      %value: i64) ctrl (%ctrl: none) {
    %result, %done = dataflow.graph.launch @sync deps(%ctrl)
        values(%value) stream_inputs() memories() stream_outputs()
        : (none, i64) -> (i64, none)
    dataflow.thread.yield %done : none
  }
  func.func private @host() {
    %value = arith.constant 7 : i64
    %thread = dataflow.thread.launch @worker(%value)
        : (i64) -> !dataflow.thread_token
    return
  }
}
)mlir",
                                                        &context);
  if (!module)
    fail("cannot parse single-graph Dataflow fixture");
  return take(dataflow::finalizeCanonicalDataflow(*module));
}

dataflow::CanonicalDataflowArtifact
buildInfeasibleDataflow(mlir::MLIRContext &context) {
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 64>>} {
  dataflow.graph private @unsupported(
      %start: none, %lhs: i128, %rhs: i128) -> i128
      attributes {input_segments = array<i32: 2, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %sum = arith.addi %lhs, %rhs : i128
    %result:2 = dataflow.sync %start, %sum
        : (none, i128) -> (none, i128)
    dataflow.graph.return values(%result#1 : i128) streams() memories()
        complete(%result#0 : none)
  }
}
)mlir",
                                                        &context);
  if (!module)
    fail("cannot parse infeasible Dataflow fixture");
  return take(dataflow::finalizeCanonicalDataflow(*module));
}

dataflow::CanonicalDataflowArtifact
buildGraphFreeDataflow(mlir::MLIRContext &context) {
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 64>>} {
  func.func private @host() {
    return
  }
}
)mlir",
                                                        &context);
  if (!module)
    fail("cannot parse graph-free Dataflow fixture");
  return take(dataflow::finalizeCanonicalDataflow(*module));
}

loom::fabric::FinalizedFabricRoot
buildSmallSpatialCore(loom::ArtifactStore &store) {
  loom::adg::DesignBuilder builder(store);
  auto expansion = take(loom::adg::expandBuiltinSpatialCore(
      builder, loom::adg::BuiltinTargetPreset::Small));
  if (llvm::Error error = expansion.spatialCore.close(expansion.outputs))
    fail(llvm::toString(std::move(error)));
  auto design = take(std::move(builder).finalize());
  if (design.roots().size() != 1)
    fail("builtin SpatialCore did not publish exactly one Fabric root");
  return design.roots().front();
}

void rootCompleteAdapterPublishesExactTechMapping() {
  TemporaryDirectory directory;
  loom::ArtifactStore store(directory.path());
  llvm::SmallString<128> blobPath(directory.path());
  llvm::sys::path::append(blobPath, "blobs");
  if (std::error_code error = llvm::sys::fs::create_directories(blobPath))
    fail("cannot create BlobStore directory: " + error.message());
  const loom::BlobStore blobs(blobPath);
  mlir::MLIRContext context = makeContext();
  auto dataflowArtifact = buildRootedDataflow(context);
  auto dataflowReference =
      take(dataflow::publishCanonicalDataflow(dataflowArtifact, store));
  auto dataflowView = take(dataflowArtifact.view());
  auto fabricRoot = buildSmallSpatialCore(store);
  const auto &fabricReference = fabricRoot.reference();

  loom::ResolvedConfig resolved = loom::defaultResolvedConfig();
  resolved.dse.techMapping.candidatePublicationLimit = 1;
  auto config =
      take(loom::mapping::projectResolvedTechMappingConfigView(resolved));
  auto inputs =
      take(loom::dse::bindRootCompleteTechMappingCandidateGeneratorInputs(
          {dataflowReference}, fabricReference));
  auto binding =
      take(loom::dse::resolveRootCompleteTechMappingCandidateGeneratorBinding(
          config));
  auto outcome =
      take(loom::dse::invokeCandidateGenerator(inputs, binding, store, blobs));
  const auto *completed =
      std::get_if<loom::dse::CompletedCandidateGeneratorResult>(
          &outcome.outcome);
  if (!completed || completed->outputBindings.size() != 1 ||
      completed->outputBindings.front().artifacts.size() != 1)
    fail("root-complete adapter lost its canonical TechMapping singleton");
  if (completed->lineageEdges.size() != 1 ||
      completed->lineageEdges.front().kind !=
          loom::dse::CandidateGeneratorLineageEdgeKind::MechanicalDerivation ||
      !completed->lineageEdges.front().parents.empty() ||
      !completed->lineageEdges.front().ownerPayload.empty())
    fail("root-complete adapter published non-mechanical lineage");

  auto tech = take(loom::mapping::importTechMapping(
      completed->outputBindings.front().artifacts.front(), store));
  if (tech.view().dataflowIdentity() != dataflowView.identity() ||
      tech.view().fabricIdentity() != fabricRoot.view().identity() ||
      tech.view().covers().size() != dataflowView.graphs().size())
    fail("TechMapping did not bind the exact root-complete D/F closure");
  for (std::size_t ordinal = 0; ordinal != dataflowView.graphs().size();
       ++ordinal)
    if (tech.view().covers()[ordinal] != dataflowView.graphs()[ordinal].ref)
      fail("TechMapping graph cover differs from canonical graph order");
}

void emptyCandidateSetIsACompletedEmptySet() {
  TemporaryDirectory directory;
  loom::ArtifactStore store(directory.path());
  llvm::SmallString<128> blobPath(directory.path());
  llvm::sys::path::append(blobPath, "blobs");
  if (std::error_code error = llvm::sys::fs::create_directories(blobPath))
    fail("cannot create BlobStore directory: " + error.message());
  const loom::BlobStore blobs(blobPath);
  auto fabric = buildSmallSpatialCore(store);
  loom::ResolvedConfig resolved = loom::defaultResolvedConfig();
  auto config =
      take(loom::mapping::projectResolvedTechMappingConfigView(resolved));
  auto inputs =
      take(loom::dse::bindRootCompleteTechMappingCandidateGeneratorInputs(
          {}, fabric.reference()));
  auto binding =
      take(loom::dse::resolveRootCompleteTechMappingCandidateGeneratorBinding(
          config));
  auto outcome =
      take(loom::dse::invokeCandidateGenerator(inputs, binding, store, blobs));
  const auto *completed =
      std::get_if<loom::dse::CompletedCandidateGeneratorResult>(
          &outcome.outcome);
  if (!completed || completed->outputBindings.size() != 1 ||
      !completed->outputBindings.front().artifacts.empty() ||
      !completed->lineageEdges.empty())
    fail("empty Dataflow candidate set did not propagate as completed empty");
}

void graphFreeDataflowContributesNoCandidate() {
  TemporaryDirectory directory;
  loom::ArtifactStore store(directory.path());
  llvm::SmallString<128> blobPath(directory.path());
  llvm::sys::path::append(blobPath, "blobs");
  if (std::error_code error = llvm::sys::fs::create_directories(blobPath))
    fail("cannot create BlobStore directory: " + error.message());
  const loom::BlobStore blobs(blobPath);
  mlir::MLIRContext context = makeContext();
  auto dataflow = buildGraphFreeDataflow(context);
  auto dataflowReference =
      take(dataflow::publishCanonicalDataflow(dataflow, store));
  auto fabric = buildSmallSpatialCore(store);
  loom::ResolvedConfig resolved = loom::defaultResolvedConfig();
  auto config =
      take(loom::mapping::projectResolvedTechMappingConfigView(resolved));
  auto inputs =
      take(loom::dse::bindRootCompleteTechMappingCandidateGeneratorInputs(
          {dataflowReference}, fabric.reference()));
  auto binding =
      take(loom::dse::resolveRootCompleteTechMappingCandidateGeneratorBinding(
          config));
  auto outcome =
      take(loom::dse::invokeCandidateGenerator(inputs, binding, store, blobs));
  const auto *completed =
      std::get_if<loom::dse::CompletedCandidateGeneratorResult>(
          &outcome.outcome);
  if (!completed || completed->outputBindings.size() != 1 ||
      !completed->outputBindings.front().artifacts.empty() ||
      !completed->lineageEdges.empty())
    fail("graph-free Dataflow did not contribute a completed empty set");
}

void descriptorReusesTheExactTechMappingOwnerContract() {
  loom::ResolvedConfig resolved = loom::defaultResolvedConfig();
  auto config =
      take(loom::mapping::projectResolvedTechMappingConfigView(resolved));
  if (llvm::Error error =
          loom::dse::registerRootCompleteTechMappingCandidateGenerator())
    fail(llvm::toString(std::move(error)));
  const auto &descriptor =
      loom::dse::rootCompleteTechMappingCandidateGeneratorDescriptor();
  if (descriptor.kind !=
          loom::dse::rootCompleteTechMappingCandidateGeneratorKind ||
      descriptor.determinism !=
          loom::dse::CandidateGeneratorDeterminism::Deterministic ||
      descriptor.inputSlots.size() != 2 || descriptor.outputSlots.size() != 1 ||
      descriptor.workUnits.size() != 4 ||
      descriptor.resolvedConfigView.schemaDescriptorBytes !=
          config.schemaDescriptorBytes() ||
      descriptor.workUnits[0].spelling != "match_row_attempt" ||
      descriptor.workUnits[1].spelling != "partial_cover_expansion" ||
      descriptor.workUnits[2].spelling != "candidate_evaluation" ||
      descriptor.workUnits[3].spelling != "publication_slot")
    fail("descriptor diverges from the exact TechMapping owner contract");
}

void finiteDataflowSetComposesIndependentOwnerInvocations() {
  TemporaryDirectory directory;
  loom::ArtifactStore store(directory.path());
  llvm::SmallString<128> blobPath(directory.path());
  llvm::sys::path::append(blobPath, "blobs");
  if (std::error_code error = llvm::sys::fs::create_directories(blobPath))
    fail("cannot create BlobStore directory: " + error.message());
  const loom::BlobStore blobs(blobPath);
  mlir::MLIRContext context = makeContext();
  auto multiGraph = buildRootedDataflow(context);
  auto singleGraph = buildSingleGraphDataflow(context);
  auto multiReference =
      take(dataflow::publishCanonicalDataflow(multiGraph, store));
  auto singleReference =
      take(dataflow::publishCanonicalDataflow(singleGraph, store));
  std::vector<loom::ArtifactRootReference> candidates = {multiReference,
                                                         singleReference};
  llvm::sort(candidates, loom::artifactRootReferenceLess);
  auto fabric = buildSmallSpatialCore(store);
  loom::ResolvedConfig resolved = loom::defaultResolvedConfig();
  resolved.dse.techMapping.candidatePublicationLimit = 1;
  auto config =
      take(loom::mapping::projectResolvedTechMappingConfigView(resolved));
  auto inputs =
      take(loom::dse::bindRootCompleteTechMappingCandidateGeneratorInputs(
          candidates, fabric.reference()));
  auto binding =
      take(loom::dse::resolveRootCompleteTechMappingCandidateGeneratorBinding(
          config));
  auto outcome =
      take(loom::dse::invokeCandidateGenerator(inputs, binding, store, blobs));
  const auto *completed =
      std::get_if<loom::dse::CompletedCandidateGeneratorResult>(
          &outcome.outcome);
  if (!completed || completed->outputBindings.front().artifacts.size() != 2 ||
      completed->lineageEdges.size() != 2)
    fail("finite Dataflow set did not complete two owner invocations");

  bool sawMultiGraph = false;
  bool sawSingleGraph = false;
  for (const auto &candidate :
       completed->outputBindings.front().artifacts) {
    const auto owner = take(loom::mapping::importTechMapping(candidate, store))
                           .view()
                           .dataflowIdentity();
    sawMultiGraph = sawMultiGraph || owner == multiGraph.identity();
    sawSingleGraph = sawSingleGraph || owner == singleGraph.identity();
  }
  if (!sawMultiGraph || !sawSingleGraph)
    fail("TechMapping outputs lost an exact Dataflow owner");
}

void infeasibleAndIncompleteOutcomesRemainDistinct() {
  TemporaryDirectory directory;
  loom::ArtifactStore store(directory.path());
  llvm::SmallString<128> blobPath(directory.path());
  llvm::sys::path::append(blobPath, "blobs");
  if (std::error_code error = llvm::sys::fs::create_directories(blobPath))
    fail("cannot create BlobStore directory: " + error.message());
  const loom::BlobStore blobs(blobPath);
  mlir::MLIRContext context = makeContext();
  auto infeasible = buildInfeasibleDataflow(context);
  auto infeasibleReference =
      take(dataflow::publishCanonicalDataflow(infeasible, store));
  auto multiGraph = buildRootedDataflow(context);
  auto multiReference =
      take(dataflow::publishCanonicalDataflow(multiGraph, store));
  auto fabric = buildSmallSpatialCore(store);

  loom::ResolvedConfig resolved = loom::defaultResolvedConfig();
  auto completeConfig =
      take(loom::mapping::projectResolvedTechMappingConfigView(resolved));
  auto infeasibleInputs =
      take(loom::dse::bindRootCompleteTechMappingCandidateGeneratorInputs(
          {infeasibleReference}, fabric.reference()));
  auto completeBinding =
      take(loom::dse::resolveRootCompleteTechMappingCandidateGeneratorBinding(
          completeConfig));
  auto infeasibleOutcome = take(loom::dse::invokeCandidateGenerator(
      infeasibleInputs, completeBinding, store, blobs));
  const auto *completed =
      std::get_if<loom::dse::CompletedCandidateGeneratorResult>(
          &infeasibleOutcome.outcome);
  if (!completed || !completed->outputBindings.front().artifacts.empty())
    fail("proven-infeasible Dataflow did not produce a completed empty set");

  resolved.dse.techMapping.matchRowAttemptLimit = 1;
  auto limitedConfig =
      take(loom::mapping::projectResolvedTechMappingConfigView(resolved));
  auto limitedInputs =
      take(loom::dse::bindRootCompleteTechMappingCandidateGeneratorInputs(
          {multiReference}, fabric.reference()));
  auto limitedBinding =
      take(loom::dse::resolveRootCompleteTechMappingCandidateGeneratorBinding(
          limitedConfig));
  auto limitedOutcome = take(loom::dse::invokeCandidateGenerator(
      limitedInputs, limitedBinding, store, blobs));
  const auto *incomplete =
      std::get_if<loom::dse::IncompleteCandidateGeneratorResult>(
          &limitedOutcome.outcome);
  if (!incomplete ||
      incomplete->reason !=
          loom::dse::CandidateGeneratorIncompleteReason::ProofNotEstablished)
    fail("limit-before-proof did not remain a typed incomplete outcome");
}

void incompleteTraversalContinuesIndependentInputs() {
  TemporaryDirectory directory;
  loom::ArtifactStore store(directory.path());
  llvm::SmallString<128> blobPath(directory.path());
  llvm::sys::path::append(blobPath, "blobs");
  if (std::error_code error = llvm::sys::fs::create_directories(blobPath))
    fail("cannot create BlobStore directory: " + error.message());
  const loom::BlobStore blobs(blobPath);
  mlir::MLIRContext context = makeContext();
  auto multiGraph = buildRootedDataflow(context);
  auto singleGraph = buildSingleGraphDataflow(context);
  auto multiReference =
      take(dataflow::publishCanonicalDataflow(multiGraph, store));
  auto singleReference =
      take(dataflow::publishCanonicalDataflow(singleGraph, store));
  std::vector<loom::ArtifactRootReference> candidates = {multiReference,
                                                         singleReference};
  llvm::sort(candidates, loom::artifactRootReferenceLess);
  auto fabric = buildSmallSpatialCore(store);
  loom::ResolvedConfig resolved = loom::defaultResolvedConfig();
  resolved.dse.techMapping.matchRowAttemptLimit = 1;
  resolved.dse.techMapping.candidatePublicationLimit = 1;
  auto config =
      take(loom::mapping::projectResolvedTechMappingConfigView(resolved));
  auto inputs =
      take(loom::dse::bindRootCompleteTechMappingCandidateGeneratorInputs(
          candidates, fabric.reference()));
  auto binding =
      take(loom::dse::resolveRootCompleteTechMappingCandidateGeneratorBinding(
          config));
  auto outcome =
      take(loom::dse::invokeCandidateGenerator(inputs, binding, store, blobs));
  const auto *incomplete =
      std::get_if<loom::dse::IncompleteCandidateGeneratorResult>(
          &outcome.outcome);
  if (!incomplete ||
      incomplete->reason !=
          loom::dse::CandidateGeneratorIncompleteReason::ProofNotEstablished ||
      incomplete->retainedOutputBindings.front().artifacts.size() != 1 ||
      incomplete->lineageEdges.size() != 1)
    fail("incomplete traversal suppressed an independent completed input");
  auto retained = take(loom::mapping::importTechMapping(
      incomplete->retainedOutputBindings.front().artifacts.front(), store));
  if (retained.view().dataflowIdentity() != singleReference.artifact)
    fail("incomplete traversal retained the wrong independent input");
}

void interruptionMapsToCancelledProviderOutcome() {
  TemporaryDirectory directory;
  loom::ArtifactStore store(directory.path());
  llvm::SmallString<128> blobPath(directory.path());
  llvm::sys::path::append(blobPath, "blobs");
  if (std::error_code error = llvm::sys::fs::create_directories(blobPath))
    fail("cannot create BlobStore directory: " + error.message());
  const loom::BlobStore blobs(blobPath);
  mlir::MLIRContext context = makeContext();
  auto dataflow = buildSingleGraphDataflow(context);
  const auto dataflowReference =
      take(dataflow::publishCanonicalDataflow(dataflow, store));
  const auto fabric = buildSmallSpatialCore(store);
  const auto config = take(loom::mapping::projectResolvedTechMappingConfigView(
      loom::defaultResolvedConfig()));
  const auto inputs =
      take(loom::dse::bindRootCompleteTechMappingCandidateGeneratorInputs(
          {dataflowReference}, fabric.reference()));
  const auto binding =
      take(loom::dse::resolveRootCompleteTechMappingCandidateGeneratorBinding(
          config));
  const ImmediateStopSource stop;
  const auto outcome = take(loom::dse::invokeCandidateGenerator(
      inputs, binding, store, blobs, stop.control()));
  const auto *incomplete =
      std::get_if<loom::dse::IncompleteCandidateGeneratorResult>(
          &outcome.outcome);
  if (!incomplete ||
      incomplete->reason !=
          loom::dse::CandidateGeneratorIncompleteReason::CancelledOrTimeout ||
      !incomplete->retainedOutputBindings.front().artifacts.empty() ||
      llvm::any_of(outcome.workSummary,
                   [](const auto &work) { return work.consumed != 0; }))
    fail("Tech interruption did not map to a cancelled provider outcome");
}

} // namespace

int main() {
  rootCompleteAdapterPublishesExactTechMapping();
  emptyCandidateSetIsACompletedEmptySet();
  graphFreeDataflowContributesNoCandidate();
  descriptorReusesTheExactTechMappingOwnerContract();
  finiteDataflowSetComposesIndependentOwnerInvocations();
  infeasibleAndIncompleteOutcomesRemainDistinct();
  incompleteTraversalContinuesIndependentInputs();
  interruptionMapsToCancelledProviderOutcome();
  llvm::outs() << "root-complete TechMapping generator anchor passed\n";
  return EXIT_SUCCESS;
}
