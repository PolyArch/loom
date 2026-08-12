#include "ADG/Builtin.h"
#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "Config/ResolvedConfig.h"
#include "DSE/CandidateGenerator.h"
#include "DSE/StructuredMemoryCommunicationCandidateGenerator.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Frontend/Compilation/StructuredMemoryCommunication.h"
#include "Frontend/IR/LoomDialect.h"
#include "Frontend/IR/StructuredProgramArtifact.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/ThreadPool.h"
#include "llvm/Support/Threading.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <cstdlib>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace {

[[noreturn]] void fail(const std::string &message) {
  llvm::errs() << "structuredMemoryCommunicationLineage: " << message << '\n';
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
    registry.insert<dataflow::DataflowDialect, loom::LoomDialect,
                    mlir::arith::ArithDialect, mlir::DLTIDialect,
                    mlir::func::FuncDialect, mlir::memref::MemRefDialect,
                    mlir::scf::SCFDialect>();
    auto *created =
        new mlir::MLIRContext(registry, mlir::MLIRContext::Threading::DISABLED);
    created->loadAllAvailableDialects();
    return created;
  }();
  return *result;
}

loom::frontend::StructuredProgramCandidate parseProgram(llvm::StringRef text) {
  auto module = mlir::parseSourceString<mlir::ModuleOp>(text, &context());
  if (!module)
    fail("cannot parse a catalog fixture");
  return take(loom::frontend::finalizeStructuredProgram(module.get()));
}

loom::frontend::StructuredProgramCandidate stageProgram() {
  return parseProgram(R"mlir(
#layout = #dlti.dl_spec<#dlti.dl_entry<index, 32>>
module attributes {dlti.dl_spec = #layout} {
  memref.global constant @stage_source : memref<1xi32> = dense<[7]>
  memref.global @stage_target : memref<1xi32> = dense<[0]>

  dataflow.thread private @stage domain(#dataflow.thread_domain<dense>)(
      %source: memref<1xi32>, %target: memref<1xi32>) ctrl (%start: none) {
    "loom.spatial_region"(%source, %target)
        <{operandSegmentSizes = array<i32: 0, 0, 2, 0>,
          resultSegmentSizes = array<i32: 0, 0>}> ({
      ^bb0(%input: memref<1xi32>, %output: memref<1xi32>):
        %c0 = arith.constant 0 : index
        %value = memref.load %input[%c0] : memref<1xi32>
        memref.store %value, %output[%c0] : memref<1xi32>
        "loom.spatial_yield"()
            <{operandSegmentSizes = array<i32: 0, 0>}> : () -> ()
    }) {graph_name = "stage_catalog", source_maps = []} :
        (memref<1xi32>, memref<1xi32>) -> ()
    dataflow.thread.yield
  }

  func.func @entry() {
    %source = memref.get_global @stage_source : memref<1xi32>
    %target = memref.get_global @stage_target : memref<1xi32>
    %token = dataflow.thread.launch @stage(%source, %target) :
        (memref<1xi32>, memref<1xi32>) -> !dataflow.thread_token
    dataflow.thread.wait %token : !dataflow.thread_token
    return
  }
}
)mlir");
}

loom::frontend::StructuredProgramCandidate layoutProgram() {
  return parseProgram(R"mlir(
#layout = #dlti.dl_spec<#dlti.dl_entry<index, 32>>
module attributes {dlti.dl_spec = #layout} {
  memref.global constant @layout_source : memref<2x3xi32> = dense<0>
  memref.global @layout_target : memref<2x3xi32> = dense<0>

  dataflow.thread private @layout domain(#dataflow.thread_domain<dense>)(
      %source: memref<2x3xi32>, %target: memref<2x3xi32>)
      ctrl (%start: none) {
    "loom.spatial_region"(%source, %target)
        <{operandSegmentSizes = array<i32: 0, 0, 2, 0>,
          resultSegmentSizes = array<i32: 0, 0>}> ({
      ^bb0(%input: memref<2x3xi32>, %output: memref<2x3xi32>):
        %local = memref.alloc() : memref<2x3xi32>
        memref.copy %input, %local :
            memref<2x3xi32> to memref<2x3xi32>
        memref.copy %local, %output :
            memref<2x3xi32> to memref<2x3xi32>
        memref.dealloc %local : memref<2x3xi32>
        "loom.spatial_yield"()
            <{operandSegmentSizes = array<i32: 0, 0>}> : () -> ()
    }) {graph_name = "layout_catalog", source_maps = []} :
        (memref<2x3xi32>, memref<2x3xi32>) -> ()
    dataflow.thread.yield
  }

  func.func @entry() {
    %source = memref.get_global @layout_source : memref<2x3xi32>
    %target = memref.get_global @layout_target : memref<2x3xi32>
    %token = dataflow.thread.launch @layout(%source, %target) :
        (memref<2x3xi32>, memref<2x3xi32>) -> !dataflow.thread_token
    dataflow.thread.wait %token : !dataflow.thread_token
    return
  }
}
)mlir");
}

loom::frontend::StructuredProgramCandidate pipelineProgram() {
  return parseProgram(R"mlir(
#layout = #dlti.dl_spec<#dlti.dl_entry<index, 32>>
module attributes {dlti.dl_spec = #layout} {
  memref.global constant @pipeline_source : memref<2xi32> = dense<[3, 5]>
  memref.global @pipeline_target : memref<2xi32> = dense<[0, 0]>

  dataflow.thread private @pipeline domain(#dataflow.thread_domain<dense>)(
      %source: memref<2xi32>, %target: memref<2xi32>) ctrl (%start: none) {
    "loom.spatial_region"(%source, %target)
        <{operandSegmentSizes = array<i32: 0, 0, 2, 0>,
          resultSegmentSizes = array<i32: 0, 0>}> ({
      ^bb0(%input: memref<2xi32>, %output: memref<2xi32>):
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        scf.for %i = %c0 to %c2 step %c1 {
          %buffer = memref.alloc() : memref<2xi32>
          memref.copy %input, %buffer : memref<2xi32> to memref<2xi32>
          %value = memref.load %buffer[%i] : memref<2xi32>
          memref.store %value, %output[%i] : memref<2xi32>
          memref.dealloc %buffer : memref<2xi32>
        }
        "loom.spatial_yield"()
            <{operandSegmentSizes = array<i32: 0, 0>}> : () -> ()
    }) {graph_name = "pipeline_catalog", source_maps = []} :
        (memref<2xi32>, memref<2xi32>) -> ()
    dataflow.thread.yield
  }

  func.func @entry() {
    %source = memref.get_global @pipeline_source : memref<2xi32>
    %target = memref.get_global @pipeline_target : memref<2xi32>
    %token = dataflow.thread.launch @pipeline(%source, %target) :
        (memref<2xi32>, memref<2xi32>) -> !dataflow.thread_token
    dataflow.thread.wait %token : !dataflow.thread_token
    return
  }
}
)mlir");
}

loom::frontend::StructuredProgramCandidate channelProgram() {
  return parseProgram(R"mlir(
#layout = #dlti.dl_spec<#dlti.dl_entry<index, 32>>
module attributes {dlti.dl_spec = #layout} {
  memref.global @channel_source : memref<2xi32> = dense<[11, 13]>
  memref.global @channel_target : memref<2xi32> = dense<[0, 0]>

  dataflow.thread private @producer domain(#dataflow.thread_domain<dense>)(
      %source: memref<2xi32>, %temporary: memref<2xi32>) ctrl (%start: none) {
    "loom.spatial_region"(%source, %temporary)
        <{operandSegmentSizes = array<i32: 0, 0, 2, 0>,
          resultSegmentSizes = array<i32: 0, 0>}> ({
      ^bb0(%input: memref<2xi32>, %buffer: memref<2xi32>):
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        scf.for %i = %c0 to %c2 step %c1 {
          %value = memref.load %input[%i] : memref<2xi32>
          memref.store %value, %buffer[%i] : memref<2xi32>
        }
        "loom.spatial_yield"()
            <{operandSegmentSizes = array<i32: 0, 0>}> : () -> ()
    }) {graph_name = "channel_catalog_producer", source_maps = []} :
        (memref<2xi32>, memref<2xi32>) -> ()
    dataflow.thread.yield
  }

  dataflow.thread private @consumer domain(#dataflow.thread_domain<dense>)(
      %temporary: memref<2xi32>, %target: memref<2xi32>) ctrl (%start: none) {
    "loom.spatial_region"(%temporary, %target)
        <{operandSegmentSizes = array<i32: 0, 0, 2, 0>,
          resultSegmentSizes = array<i32: 0, 0>}> ({
      ^bb0(%buffer: memref<2xi32>, %output: memref<2xi32>):
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        scf.for %i = %c0 to %c2 step %c1 {
          %value = memref.load %buffer[%i] : memref<2xi32>
          memref.store %value, %output[%i] : memref<2xi32>
        }
        "loom.spatial_yield"()
            <{operandSegmentSizes = array<i32: 0, 0>}> : () -> ()
    }) {graph_name = "channel_catalog_consumer", source_maps = []} :
        (memref<2xi32>, memref<2xi32>) -> ()
    dataflow.thread.yield
  }

  func.func @entry() {
    %source = memref.get_global @channel_source : memref<2xi32>
    %target = memref.get_global @channel_target : memref<2xi32>
    %temporary = memref.alloc() : memref<2xi32>
    %producer = dataflow.thread.launch @producer(%source, %temporary) :
        (memref<2xi32>, memref<2xi32>) -> !dataflow.thread_token
    dataflow.thread.wait %producer : !dataflow.thread_token
    %consumer = dataflow.thread.launch @consumer(%temporary, %target) :
        (memref<2xi32>, memref<2xi32>) -> !dataflow.thread_token
    dataflow.thread.wait %consumer : !dataflow.thread_token
    memref.dealloc %temporary : memref<2xi32>
    return
  }
}
)mlir");
}

void requireKinds(
    const loom::frontend::StructuredProgramCandidate &candidate,
    llvm::ArrayRef<loom::frontend::StructuredMemoryCommunicationDecisionKind>
        expected) {
  auto domain =
      take(loom::frontend::enumerateStructuredMemoryCommunicationDecisions(
          candidate, 64));
  std::vector<loom::frontend::StructuredMemoryCommunicationDecisionKind> actual;
  for (const auto &decision : domain.decisions)
    actual.push_back(
        loom::frontend::structuredMemoryCommunicationDecisionKind(decision));
  if (llvm::ArrayRef(actual) != expected)
    fail("catalog fixture produced the wrong canonical decision kinds");
}

bool sameOutputBindings(
    llvm::ArrayRef<loom::dse::CandidateGeneratorOutputBinding> lhs,
    llvm::ArrayRef<loom::dse::CandidateGeneratorOutputBinding> rhs) {
  if (lhs.size() != rhs.size())
    return false;
  for (auto [left, right] : llvm::zip_equal(lhs, rhs))
    if (left.slot != right.slot || left.artifacts != right.artifacts)
      return false;
  return true;
}

bool sameFormalResult(const loom::dse::CandidateGeneratorProviderResult &lhs,
                      const loom::dse::CandidateGeneratorProviderResult &rhs) {
  if (lhs.workSummary != rhs.workSummary ||
      lhs.outcome.index() != rhs.outcome.index())
    return false;
  if (const auto *left =
          std::get_if<loom::dse::CompletedCandidateGeneratorResult>(
              &lhs.outcome)) {
    const auto &right =
        std::get<loom::dse::CompletedCandidateGeneratorResult>(rhs.outcome);
    return sameOutputBindings(left->outputBindings, right.outputBindings) &&
           left->lineageEdges == right.lineageEdges;
  }
  const auto &left =
      std::get<loom::dse::IncompleteCandidateGeneratorResult>(lhs.outcome);
  const auto &right =
      std::get<loom::dse::IncompleteCandidateGeneratorResult>(rhs.outcome);
  return left.reason == right.reason &&
         sameOutputBindings(left.retainedOutputBindings,
                            right.retainedOutputBindings) &&
         left.lineageEdges == right.lineageEdges;
}

void requireReplay(
    const loom::dse::CompletedCandidateGeneratorResult &completed,
    const loom::ArtifactStore &store) {
  for (const loom::dse::CandidateGeneratorLineageEdge &edge :
       completed.lineageEdges) {
    if (edge.parents.size() != 1)
      fail("memory lineage edge does not have one exact parent");
    auto parent = take(
        loom::frontend::importStructuredProgram(edge.parents.front(), store));
    auto decision =
        take(loom::frontend::adoptStructuredMemoryCommunicationDecision(
            edge.ownerPayload));
    auto replayed =
        take(loom::frontend::materializeStructuredMemoryCommunicationDecision(
            parent, decision));
    auto reference = take(loom::frontend::publishStructuredProgram(
        replayed.structuredProgram, store));
    if (reference != edge.output)
      fail("parent and owner decision did not replay the exact child");
  }
}

void catalogLineageIsClosed(llvm::StringRef directory) {
  loom::ArtifactStore store(directory);
  llvm::SmallString<128> blobPath(directory);
  llvm::sys::path::append(blobPath, "blobs");
  if (std::error_code error = llvm::sys::fs::create_directories(blobPath))
    fail("cannot create BlobStore directory: " + error.message());
  const loom::BlobStore blobs(blobPath);
  auto design = take(loom::adg::buildBuiltinTarget(
      store, loom::adg::BuiltinTargetPreset::Small));

  auto stage = stageProgram();
  auto layout = layoutProgram();
  auto pipeline = pipelineProgram();
  auto channel = channelProgram();
  using Kind = loom::frontend::StructuredMemoryCommunicationDecisionKind;
  requireKinds(stage, {Kind::StageConstantGlobal});
  requireKinds(layout, {Kind::PermuteLocalBufferLayout});
  requireKinds(pipeline, {Kind::PipelineStagedLoop});
  requireKinds(channel, {Kind::PromoteOrderedBufferToChannel});
  auto layoutDomain =
      take(loom::frontend::enumerateStructuredMemoryCommunicationDecisions(
          layout, 64));
  for (auto [position, decision] : llvm::enumerate(layoutDomain.decisions)) {
    const auto *permutation =
        std::get_if<loom::frontend::PermuteLocalBufferLayoutDecision>(
            &decision);
    if (!permutation || permutation->adjacentStoragePosition != position)
      fail("layout parameter domain is not in canonical position order");
  }

  std::vector<loom::ArtifactRootReference> parents = {
      take(loom::frontend::publishStructuredProgram(stage, store)),
      take(loom::frontend::publishStructuredProgram(layout, store)),
      take(loom::frontend::publishStructuredProgram(pipeline, store)),
      take(loom::frontend::publishStructuredProgram(channel, store))};
  llvm::sort(parents, loom::artifactRootReferenceLess);

  loom::ResolvedConfig resolved = loom::defaultResolvedConfig();
  resolved.dse.memoryCommunication.scopeExpansionLimit = 64;
  auto config =
      take(loom::dse::
               projectResolvedStructuredMemoryCommunicationGeneratorConfigView(
                   resolved));
  auto inputs =
      take(loom::dse::bindStructuredMemoryCommunicationCandidateGeneratorInputs(
          parents));
  auto binding = take(
      loom::dse::resolveStructuredMemoryCommunicationCandidateGeneratorBinding(
          config));
  auto emptyInputs = take(
      loom::dse::bindStructuredMemoryCommunicationCandidateGeneratorInputs({}));
  auto empty = take(
      loom::dse::invokeCandidateGenerator(emptyInputs, binding, store, blobs));
  const auto *emptyCompleted =
      std::get_if<loom::dse::CompletedCandidateGeneratorResult>(&empty.outcome);
  if (!emptyCompleted || emptyCompleted->outputBindings.size() != 1 ||
      !emptyCompleted->outputBindings.front().artifacts.empty() ||
      !emptyCompleted->lineageEdges.empty() ||
      empty.workSummary !=
          std::vector<loom::dse::CandidateGeneratorWorkUnitSummary>{
              {loom::dse::CandidateGeneratorWorkUnitRef(0), 0, 0},
              {loom::dse::CandidateGeneratorWorkUnitRef(1), 0, 0}})
    fail("empty finite input did not complete with exact zero work");
  auto expected =
      take(loom::dse::invokeCandidateGenerator(inputs, binding, store, blobs));
  const auto *completed =
      std::get_if<loom::dse::CompletedCandidateGeneratorResult>(
          &expected.outcome);
  const std::size_t outputCount =
      completed && completed->outputBindings.size() == 1
          ? completed->outputBindings.front().artifacts.size()
          : 0;
  const std::size_t lineageCount =
      completed ? completed->lineageEdges.size() : 0;
  if (!completed || completed->outputBindings.size() != 1 || outputCount != 8 ||
      lineageCount != 4) {
    std::string kinds;
    if (completed)
      for (const auto &edge : completed->lineageEdges) {
        auto decision =
            take(loom::frontend::adoptStructuredMemoryCommunicationDecision(
                edge.ownerPayload));
        if (!kinds.empty())
          kinds.push_back(',');
        kinds += std::to_string(static_cast<unsigned>(
            loom::frontend::structuredMemoryCommunicationDecisionKind(
                decision)));
      }
    fail("closed catalog did not publish four parents and four children: " +
         std::to_string(outputCount) + " outputs, " +
         std::to_string(lineageCount) + " lineage edges, kinds " + kinds);
  }
  const std::vector<loom::dse::CandidateGeneratorWorkUnitSummary> expectedWork =
      {{loom::dse::CandidateGeneratorWorkUnitRef(0), 20, 20},
       {loom::dse::CandidateGeneratorWorkUnitRef(1), 4, 4}};
  if (expected.workSummary != expectedWork)
    fail("closed catalog changed exact scope or decision work");
  if (llvm::Error error =
          loom::dse::validateCanonicalCandidateGeneratorInvocation(
              inputs, binding, completed->outputBindings,
              completed->lineageEdges, true, store))
    fail(llvm::toString(std::move(error)));
  requireReplay(*completed, store);

  for (unsigned workerCount : {2U, 4U}) {
    std::vector<std::optional<loom::dse::CandidateGeneratorProviderResult>>
        results(workerCount);
    std::vector<std::string> errors(workerCount);
    llvm::DefaultThreadPool pool(
        llvm::heavyweight_hardware_concurrency(workerCount));
    for (unsigned worker = 0; worker != workerCount; ++worker)
      pool.async([&, worker] {
        auto result =
            loom::dse::invokeCandidateGenerator(inputs, binding, store, blobs);
        if (!result) {
          errors[worker] = llvm::toString(result.takeError());
          return;
        }
        results[worker].emplace(std::move(*result));
      });
    pool.wait();
    for (unsigned worker = 0; worker != workerCount; ++worker) {
      if (!errors[worker].empty())
        fail("parallel catalog invocation failed: " + errors[worker]);
      if (!results[worker] || !sameFormalResult(expected, *results[worker]))
        fail("worker count changed catalog output, lineage, order, or work");
    }
  }

  resolved.dse.memoryCommunication.scopeExpansionLimit = 3;
  auto limitedConfig =
      take(loom::dse::
               projectResolvedStructuredMemoryCommunicationGeneratorConfigView(
                   resolved));
  std::vector<loom::ArtifactRootReference> layoutOnly = {parents.front()};
  for (const loom::ArtifactRootReference &parent : parents) {
    auto candidate =
        take(loom::frontend::importStructuredProgram(parent, store));
    auto domain =
        take(loom::frontend::enumerateStructuredMemoryCommunicationDecisions(
            candidate, 64));
    if (domain.decisions.size() == 1 &&
        loom::frontend::structuredMemoryCommunicationDecisionKind(
            domain.decisions.front()) == Kind::PermuteLocalBufferLayout) {
      layoutOnly.front() = parent;
      break;
    }
  }
  auto limitedInputs =
      take(loom::dse::bindStructuredMemoryCommunicationCandidateGeneratorInputs(
          layoutOnly));
  auto limitedBinding = take(
      loom::dse::resolveStructuredMemoryCommunicationCandidateGeneratorBinding(
          limitedConfig));
  auto limited = take(loom::dse::invokeCandidateGenerator(
      limitedInputs, limitedBinding, store, blobs));
  const auto *limitedCompleted =
      std::get_if<loom::dse::CompletedCandidateGeneratorResult>(
          &limited.outcome);
  if (!limitedCompleted || limitedCompleted->lineageEdges.size() != 1 ||
      limitedCompleted->outputBindings.front().artifacts.size() != 2 ||
      limited.workSummary !=
          std::vector<loom::dse::CandidateGeneratorWorkUnitSummary>{
              {loom::dse::CandidateGeneratorWorkUnitRef(0), 3, 3},
              {loom::dse::CandidateGeneratorWorkUnitRef(1), 1, 1}})
    fail("scope limit cut one admitted allocation parameter domain");

  if (loom::dse::structuredMemoryCommunicationCandidateGeneratorDescriptor()
          .implementationSemanticIdentity !=
      "loom.compiler.structured_memory_communication.generator.v4")
    fail("memory generator does not expose the v4 semantic identity");
  static constexpr std::array<llvm::StringLiteral, 3> legacyConfigSchemas = {
      "loom.structured_memory_communication_generator.config.1.0",
      "loom.structured_memory_communication_generator.config.2.0",
      "loom.structured_memory_communication_generator.config.3.0"};
  for (llvm::StringLiteral legacyConfigSchema : legacyConfigSchemas) {
    auto legacyDigest = take(loom::computeComponentViewDigest(
        {reinterpret_cast<const std::uint8_t *>(legacyConfigSchema.data()),
         legacyConfigSchema.size()},
        config.canonicalViewBytes()));
    auto legacy = loom::dse::ResolvedCandidateGeneratorBinding::get(
        loom::dse::structuredMemoryCommunicationCandidateGeneratorDescriptor()
            .reference(),
        config.canonicalViewBytes(), legacyDigest);
    if (legacy)
      fail("registry v4 reinterpreted a legacy generator config binding");
    llvm::consumeError(legacy.takeError());
  }
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2)
    fail("expected one test-owned working directory");
  catalogLineageIsClosed(argv[1]);
  return EXIT_SUCCESS;
}
