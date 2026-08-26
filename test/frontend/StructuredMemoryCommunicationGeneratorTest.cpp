#include "ADG/Builder.h"
#include "ADG/Builtin.h"
#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "Config/ResolvedConfig.h"
#include "DSE/CandidateGenerator.h"
#include "DSE/StructuredMemoryCommunicationCandidateGenerator.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Dataflow/IR/DataflowOps.h"
#include "Frontend/Compilation/StructuredMemoryCommunication.h"
#include "Frontend/IR/LoomDialect.h"
#include "Frontend/IR/LoomOps.h"
#include "Frontend/IR/StructuredProgramArtifact.h"
#include "Frontend/Lowering/CanonicalDataflowLowering.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <cstdlib>
#include <string>
#include <system_error>
#include <utility>
#include <variant>
#include <vector>

namespace {

[[noreturn]] void fail(const std::string &message) {
  llvm::errs() << "structuredMemoryCommunicationGenerator: " << message << '\n';
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
                    mlir::func::FuncDialect, mlir::memref::MemRefDialect>();
    auto *created =
        new mlir::MLIRContext(registry, mlir::MLIRContext::Threading::DISABLED);
    created->loadAllAvailableDialects();
    return created;
  }();
  return *result;
}

struct ProgramOptions final {
  int firstValue = 1;
  bool deriveConstantAlias = false;
  bool siblingConstantAlias = false;
  bool uninitializedConstant = false;
  bool mutableTableLaunch = false;
  bool alternateConstantLaunch = false;
  std::uint64_t loadAlignment = 0;
  bool misalignedSecondLoad = false;
};

loom::frontend::StructuredProgramCandidate
parseProgram(const ProgramOptions &options = {}) {
  std::string source = R"mlir(
#layout = #dlti.dl_spec<#dlti.dl_entry<index, 32>>
module attributes {dlti.dl_spec = #layout} {
)mlir";
  if (options.uninitializedConstant) {
    source +=
        "  memref.global constant @table : memref<4xi32> = uninitialized\n";
  } else {
    source += "  memref.global constant @table : memref<4xi32> = dense<[";
    source += std::to_string(options.firstValue);
    source += ", 2, 3, 4]> alignment = 64\n";
  }
  source += "  memref.global constant @other_table : memref<4xi32> = "
            "dense<[10, 11, 12, 13]> alignment = 64\n";
  source += R"mlir(
  memref.global @mutable : memref<1xi32> = dense<[9]>
  memref.global @mutable_table : memref<4xi32> = dense<[5, 6, 7, 8]>

  dataflow.thread private @selected domain(#dataflow.thread_domain<dense>)(
)mlir";
  source += "      %table: memref<4xi32>, ";
  if (options.siblingConstantAlias)
    source += "%sibling: memref<4xi32>, ";
  source += R"mlir(%mutable: memref<1xi32>,
      %out: memref<1xi32>) ctrl (%start: none) {
    "loom.spatial_region"(%table, %mutable, %out)
        <{operandSegmentSizes = array<i32: 0, 0, 3, 0>,
          resultSegmentSizes = array<i32: 0, 0>}> ({
      ^bb0(%table_input: memref<4xi32>, %mutable_input: memref<1xi32>,
           %target: memref<1xi32>):
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
)mlir";
  source += "        %a = memref.load %table_input[%c0]";
  if (options.loadAlignment)
    source += " alignment (" + std::to_string(options.loadAlignment) + ")";
  source += " : memref<4xi32>\n";
  source += "        %b = memref.load %table_input[%c1]";
  if (options.loadAlignment && options.misalignedSecondLoad)
    source += " alignment (" + std::to_string(options.loadAlignment) + ")";
  source += R"mlir( : memref<4xi32>
        %ignored = memref.load %mutable_input[%c0] : memref<1xi32>
)mlir";
  if (options.deriveConstantAlias) {
    source +=
        R"mlir(        %table_alias = memref.cast %table_input : memref<4xi32> to memref<?xi32>
        %alias_value = memref.load %table_alias[%c0] : memref<?xi32>
        %partial = arith.addi %a, %b : i32
        %sum = arith.addi %partial, %alias_value : i32
)mlir";
  } else {
    source += "        %sum = arith.addi %a, %b : i32\n";
  }
  source += R"mlir(
        memref.store %sum, %target[%c0] : memref<1xi32>
        "loom.spatial_yield"()
            <{operandSegmentSizes = array<i32: 0, 0>}> : () -> ()
    }) {graph_name = "selected_graph", source_maps = []} :
        (memref<4xi32>, memref<1xi32>, memref<1xi32>) -> ()
    dataflow.thread.yield
  }

  func.func @entry(%out: memref<1xi32>) {
    %table = memref.get_global @table : memref<4xi32>
    %mutable = memref.get_global @mutable : memref<1xi32>
)mlir";
  source += "    %token = dataflow.thread.launch @selected(%table, ";
  if (options.siblingConstantAlias)
    source += "%table, ";
  source += R"mlir(%mutable, %out)
        : (memref<4xi32>, )mlir";
  if (options.siblingConstantAlias)
    source += "memref<4xi32>, ";
  source += R"mlir(memref<1xi32>, memref<1xi32>) ->
          !dataflow.thread_token
    dataflow.thread.wait %token : !dataflow.thread_token
)mlir";
  if (options.mutableTableLaunch) {
    source +=
        R"mlir(    %mutable_table = memref.get_global @mutable_table : memref<4xi32>
    %other = dataflow.thread.launch @selected(%mutable_table, %mutable, %out)
        : (memref<4xi32>, memref<1xi32>, memref<1xi32>) ->
          !dataflow.thread_token
    dataflow.thread.wait %other : !dataflow.thread_token
)mlir";
  }
  if (options.alternateConstantLaunch) {
    source +=
        R"mlir(    %other_table = memref.get_global @other_table : memref<4xi32>
    %constant_other = dataflow.thread.launch @selected(%other_table, %mutable, %out)
        : (memref<4xi32>, memref<1xi32>, memref<1xi32>) ->
          !dataflow.thread_token
    dataflow.thread.wait %constant_other : !dataflow.thread_token
)mlir";
  }
  source += R"mlir(
    return
  }
}
)mlir";
  auto module = mlir::parseSourceString<mlir::ModuleOp>(source, &context());
  if (!module)
    fail("cannot parse Structured memory fixture");
  return take(loom::frontend::finalizeStructuredProgram(module.get()));
}

std::size_t
countAllocations(const loom::frontend::StructuredProgramCandidate &candidate) {
  std::size_t count = 0;
  candidate.module().walk([&](mlir::memref::AllocOp) { ++count; });
  return count;
}

void configRoundTripsAndRejectsMalformedBytes() {
  loom::ResolvedConfig resolved = loom::defaultResolvedConfig();
  resolved.dse.memoryCommunication.scopeExpansionLimit = 7;
  auto projected =
      take(loom::dse::
               projectResolvedStructuredMemoryCommunicationGeneratorConfigView(
                   resolved));
  auto adopted = take(
      loom::dse::adoptResolvedStructuredMemoryCommunicationGeneratorConfigView(
          loom::dse::
              resolvedStructuredMemoryCommunicationGeneratorConfigSchemaBytes(),
          projected.canonicalViewBytes(), projected.digest()));
  if (adopted.scopeExpansionLimit() != 7 ||
      adopted.canonicalViewBytes() != projected.canonicalViewBytes())
    fail("memory communication config did not round-trip exactly");

  std::vector<std::uint8_t> malformed(projected.canonicalViewBytes().begin(),
                                      projected.canonicalViewBytes().end());
  malformed.push_back(0);
  auto digest = take(loom::computeComponentViewDigest(
      loom::dse::
          resolvedStructuredMemoryCommunicationGeneratorConfigSchemaBytes(),
      malformed));
  auto rejected =
      loom::dse::adoptResolvedStructuredMemoryCommunicationGeneratorConfigView(
          loom::dse::
              resolvedStructuredMemoryCommunicationGeneratorConfigSchemaBytes(),
          malformed, digest);
  if (rejected)
    fail("memory communication config accepted trailing bytes");
  if (!llvm::StringRef(llvm::toString(rejected.takeError()))
           .contains("trailing"))
    fail("malformed config reported the wrong failure");

  std::vector<std::uint8_t> outOfRange = {0, 0, 0, 1, 0, 0, 0, 0};
  auto outOfRangeDigest = take(loom::computeComponentViewDigest(
      loom::dse::
          resolvedStructuredMemoryCommunicationGeneratorConfigSchemaBytes(),
      outOfRange));
  auto rangeRejected =
      loom::dse::adoptResolvedStructuredMemoryCommunicationGeneratorConfigView(
          loom::dse::
              resolvedStructuredMemoryCommunicationGeneratorConfigSchemaBytes(),
          outOfRange, outOfRangeDigest);
  if (rangeRejected)
    fail("memory communication config accepted a value outside uint32");
  if (!llvm::StringRef(llvm::toString(rangeRejected.takeError()))
           .contains("uint32"))
    fail("out-of-range config reported the wrong failure");
}

void constantGlobalStagingIsTypedAndMechanical() {
  auto parent = parseProgram();
  auto decisions =
      take(loom::frontend::enumerateStructuredMemoryCommunicationDecisions(
          parent, 64));
  if (decisions.decisions.size() != 1 ||
      loom::frontend::structuredMemoryCommunicationDecisionKind(
          decisions.decisions.front()) !=
          loom::frontend::StructuredMemoryCommunicationDecisionKind::
              StageConstantGlobal)
    fail("constant and mutable globals did not produce one exact decision");

  auto child =
      take(loom::frontend::materializeStructuredMemoryCommunicationDecision(
          parent, decisions.decisions.front()));
  if (countAllocations(parent) != 0 ||
      countAllocations(child.structuredProgram) != 1)
    fail("staging did not preserve the parent and add one logical buffer");

  std::size_t copies = 0;
  std::size_t stagedLoads = 0;
  std::size_t mutableLoads = 0;
  child.structuredProgram.module().walk([&](mlir::memref::CopyOp copy) {
    ++copies;
    if (!llvm::isa<mlir::BlockArgument>(copy.getSource()) ||
        !copy.getTarget().getDefiningOp<mlir::memref::AllocOp>())
      fail("staging copy does not connect the constant root to the buffer");
  });
  child.structuredProgram.module().walk([&](mlir::memref::LoadOp load) {
    if (load.getMemref().getDefiningOp<mlir::memref::AllocOp>())
      ++stagedLoads;
    if (auto argument = llvm::dyn_cast<mlir::BlockArgument>(load.getMemref());
        argument && argument.getArgNumber() == 1)
      ++mutableLoads;
  });
  if (copies != 1 || stagedLoads != 2 || mutableLoads != 1)
    fail("staging changed the wrong memory uses");

  auto d0 = take(loom::lowering::lowerStructuredProgramToCanonicalDataflow(
      child.structuredProgram));
  std::size_t residualCopies = 0;
  std::size_t freshRoots = 0;
  d0.module().walk([&](mlir::memref::CopyOp) { ++residualCopies; });
  d0.module().walk([&](mlir::memref::AllocOp) { ++freshRoots; });
  if (residualCopies != 0 || freshRoots != 1)
    fail("mechanical D0 lowering did not expand the staging copy");

  auto foreign = parseProgram({.firstValue = 5});
  auto rejected =
      loom::frontend::materializeStructuredMemoryCommunicationDecision(
          foreign, decisions.decisions.front());
  if (rejected)
    fail("cross-parent memory decision was accepted");
  if (!llvm::StringRef(llvm::toString(rejected.takeError()))
           .contains("different candidate"))
    fail("cross-parent memory decision reported the wrong failure");

  auto aliasParent = parseProgram({.deriveConstantAlias = true});
  auto aliasDecisions =
      take(loom::frontend::enumerateStructuredMemoryCommunicationDecisions(
          aliasParent, 64));
  if (!aliasDecisions.decisions.empty())
    fail("derived constant-memory alias produced a staging decision");

  auto siblingAlias = parseProgram({.siblingConstantAlias = true});
  auto siblingAliasDecisions =
      take(loom::frontend::enumerateStructuredMemoryCommunicationDecisions(
          siblingAlias, 64));
  if (siblingAliasDecisions.decisions.size() != 1)
    fail("a sibling formal alias invalidated constant staging");

  auto uninitialized = parseProgram({.uninitializedConstant = true});
  auto uninitializedDecisions =
      take(loom::frontend::enumerateStructuredMemoryCommunicationDecisions(
          uninitialized, 64));
  if (!uninitializedDecisions.decisions.empty())
    fail("uninitialized constant global produced a staging decision");

  auto multipleLaunches = parseProgram({.mutableTableLaunch = true});
  auto multipleLaunchDecisions =
      take(loom::frontend::enumerateStructuredMemoryCommunicationDecisions(
          multipleLaunches, 64));
  if (!multipleLaunchDecisions.decisions.empty())
    fail("mixed constant and mutable root launches produced a decision");

  auto differentConstants = parseProgram({.alternateConstantLaunch = true});
  auto differentConstantDecisions =
      take(loom::frontend::enumerateStructuredMemoryCommunicationDecisions(
          differentConstants, 64));
  if (differentConstantDecisions.decisions.size() != 1)
    fail("different constant roots across launches invalidated staging");
}

void stagingPreservesAlignmentAndExpandsMemoryWork() {
  auto parent = parseProgram({.loadAlignment = 64});
  auto decisions =
      take(loom::frontend::enumerateStructuredMemoryCommunicationDecisions(
          parent, 64));
  if (decisions.decisions.size() != 1)
    fail("aligned constant input did not produce one decision");
  auto child =
      take(loom::frontend::materializeStructuredMemoryCommunicationDecision(
          parent, decisions.decisions.front()));

  std::optional<std::uint64_t> allocationAlignment;
  child.structuredProgram.module().walk([&](mlir::memref::AllocOp allocation) {
    allocationAlignment = allocation.getAlignment();
  });
  if (allocationAlignment != 64)
    fail("staging allocation did not preserve the strongest load alignment");

  std::size_t alignedStagedLoads = 0;
  std::size_t unalignedStagedLoads = 0;
  child.structuredProgram.module().walk([&](mlir::memref::LoadOp load) {
    if (!load.getMemref().getDefiningOp<mlir::memref::AllocOp>())
      return;
    if (load.getAlignment() == 64)
      ++alignedStagedLoads;
    if (!load.getAlignment())
      ++unalignedStagedLoads;
  });
  if (alignedStagedLoads != 1 || unalignedStagedLoads != 1)
    fail("staging changed the explicit load alignment contract");

  auto invalidAlignment =
      parseProgram({.loadAlignment = 64, .misalignedSecondLoad = true});
  auto invalidAlignmentDecisions =
      take(loom::frontend::enumerateStructuredMemoryCommunicationDecisions(
          invalidAlignment, 64));
  if (!invalidAlignmentDecisions.decisions.empty())
    fail("staging accepted an alignment not established by the new buffer");

  auto parentD0 =
      take(loom::lowering::lowerStructuredProgramToCanonicalDataflow(parent));
  auto childD0 = take(loom::lowering::lowerStructuredProgramToCanonicalDataflow(
      child.structuredProgram));
  auto countMemoryActors = [](const dataflow::CanonicalDataflowArtifact &d0) {
    std::pair<std::size_t, std::size_t> counts;
    d0.module().walk([&](dataflow::LoadOp) { ++counts.first; });
    d0.module().walk([&](dataflow::StoreOp) { ++counts.second; });
    return counts;
  };
  const auto parentCounts = countMemoryActors(parentD0);
  const auto childCounts = countMemoryActors(childD0);
  if (childCounts.first <= parentCounts.first ||
      childCounts.second <= parentCounts.second)
    fail("staging copy was not expanded into additional load/store actors");

  childD0.module().walk([&](dataflow::GraphOp graph) {
    graph.getBody().walk([&](mlir::memref::GetGlobalOp) {
      fail("canonical graph retained a global symbol lookup");
    });
  });
}

void scopeExpansionLimitIsInvocationWide() {
  llvm::SmallString<128> directory;
  std::error_code error = llvm::sys::fs::createUniqueDirectory(
      "loom-memory-communication-limit", directory);
  if (error)
    fail("cannot create ArtifactStore directory: " + error.message());
  loom::ArtifactStore store(directory);
  llvm::SmallString<128> blobPath(directory);
  llvm::sys::path::append(blobPath, "blobs");
  if (std::error_code error = llvm::sys::fs::create_directories(blobPath))
    fail("cannot create BlobStore directory: " + error.message());
  const loom::BlobStore blobs(blobPath);
  auto design = take(loom::adg::buildBuiltinTarget(
      store, loom::adg::BuiltinTargetPreset::Small));
  auto first = parseProgram({.firstValue = 1});
  auto second = parseProgram({.firstValue = 8});
  auto firstReference =
      take(loom::frontend::publishStructuredProgram(first, store));
  auto secondReference =
      take(loom::frontend::publishStructuredProgram(second, store));
  std::vector<loom::ArtifactRootReference> parents = {firstReference,
                                                      secondReference};
  llvm::sort(parents, loom::artifactRootReferenceLess);

  loom::ResolvedConfig resolved = loom::defaultResolvedConfig();
  resolved.dse.memoryCommunication.scopeExpansionLimit = 1;
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
  auto outcome =
      take(loom::dse::invokeCandidateGenerator(inputs, binding, store, blobs));
  auto *completed = std::get_if<loom::dse::CompletedCandidateGeneratorResult>(
      &outcome.outcome);
  if (!completed || completed->outputBindings.size() != 1 ||
      completed->outputBindings.front().artifacts.size() != 3 ||
      completed->lineageEdges.size() != 1)
    fail("memory scope limit was restarted for each input parent");

  std::error_code cleanup = llvm::sys::fs::remove_directories(directory);
  if (cleanup)
    fail("cannot remove ArtifactStore directory: " + cleanup.message());
}

void providerPublishesParentAndAdmittedChild() {
  llvm::SmallString<128> directory;
  std::error_code error = llvm::sys::fs::createUniqueDirectory(
      "loom-memory-communication-generator", directory);
  if (error)
    fail("cannot create ArtifactStore directory: " + error.message());
  loom::ArtifactStore store(directory);
  llvm::SmallString<128> blobPath(directory);
  llvm::sys::path::append(blobPath, "blobs");
  if (std::error_code error = llvm::sys::fs::create_directories(blobPath))
    fail("cannot create BlobStore directory: " + error.message());
  const loom::BlobStore blobs(blobPath);
  auto design = take(loom::adg::buildBuiltinTarget(
      store, loom::adg::BuiltinTargetPreset::Small));
  auto parent = parseProgram();
  auto parentReference =
      take(loom::frontend::publishStructuredProgram(parent, store));
  auto config =
      take(loom::dse::
               projectResolvedStructuredMemoryCommunicationGeneratorConfigView(
                   loom::defaultResolvedConfig()));
  auto inputs =
      take(loom::dse::bindStructuredMemoryCommunicationCandidateGeneratorInputs(
          {parentReference}));
  auto binding = take(
      loom::dse::resolveStructuredMemoryCommunicationCandidateGeneratorBinding(
          config));
  auto outcome =
      take(loom::dse::invokeCandidateGenerator(inputs, binding, store, blobs));
  auto *completed = std::get_if<loom::dse::CompletedCandidateGeneratorResult>(
      &outcome.outcome);
  if (!completed || completed->outputBindings.size() != 1 ||
      completed->outputBindings.front().artifacts.size() != 2)
    fail("provider did not publish the parent and one admitted child");

  bool foundChild = false;
  for (const loom::ArtifactRootReference &reference :
       completed->outputBindings.front().artifacts) {
    auto candidate =
        take(loom::frontend::importStructuredProgram(reference, store));
    foundChild |= countAllocations(candidate) == 1;
  }
  if (!foundChild)
    fail("provider output omitted the staged child");
  if (completed->lineageEdges.size() != 1)
    fail("provider did not return one exact child lineage edge");
  const loom::dse::CandidateGeneratorLineageEdge &edge =
      completed->lineageEdges.front();
  if (edge.kind !=
          loom::dse::CandidateGeneratorLineageEdgeKind::CandidateDecision ||
      edge.outputSlot != loom::dse::CandidateGeneratorOutputSlotRef(0) ||
      edge.parents != std::vector<loom::ArtifactRootReference>{parentReference})
    fail("provider returned the wrong memory decision lineage");
  auto decodedDecision =
      take(loom::frontend::adoptStructuredMemoryCommunicationDecision(
          edge.ownerPayload));
  if (!(decodedDecision ==
        take(loom::frontend::enumerateStructuredMemoryCommunicationDecisions(
                 parent, 64))
            .decisions.front()))
    fail("memory decision lineage payload did not round-trip");

  std::error_code cleanup = llvm::sys::fs::remove_directories(directory);
  if (cleanup)
    fail("cannot remove ArtifactStore directory: " + cleanup.message());
}

void invalidInMemoryDecisionFailsClosed() {
  auto parent = parseProgram();
  auto decisions =
      take(loom::frontend::enumerateStructuredMemoryCommunicationDecisions(
          parent, 64));
  if (decisions.decisions.empty())
    fail("invalid-decision fixture has no legal memory decision");
  const auto &anchor =
      loom::frontend::structuredMemoryCommunicationDecisionAnchor(
          decisions.decisions.front());
  loom::frontend::StructuredMemoryCommunicationDecision invalid =
      loom::frontend::PipelineStagedLoopDecision{anchor};
  auto encoded =
      loom::frontend::encodeStructuredMemoryCommunicationDecision(invalid);
  if (encoded)
    fail("memory encoder accepted a typed decision with the wrong anchor");
  llvm::consumeError(encoded.takeError());
  auto materialized =
      loom::frontend::materializeStructuredMemoryCommunicationDecision(parent,
                                                                       invalid);
  if (materialized)
    fail("memory materializer accepted a typed decision with the wrong anchor");
  llvm::consumeError(materialized.takeError());
}

void lineageCodecRejectsAnOutOfRangeMemoryInput() {
  llvm::SmallString<128> directory;
  std::error_code error = llvm::sys::fs::createUniqueDirectory(
      "loom-memory-lineage-context", directory);
  if (error)
    fail("cannot create ArtifactStore directory: " + error.message());
  loom::ArtifactStore store(directory);
  auto parent = parseProgram();
  auto parentReference =
      take(loom::frontend::publishStructuredProgram(parent, store));
  const loom::frontend::StructuredMemoryCommunicationDecision decision =
      loom::frontend::StageConstantGlobalDecision{
          {parent.identity(), loom::frontend::StructuredEntityKind::Value,
           999999}};
  auto encoded = take(
      loom::frontend::encodeStructuredMemoryCommunicationDecision(decision));
  const auto *contract =
      loom::dse::structuredMemoryCommunicationCandidateGeneratorDescriptor()
          .ownerLineagePayload;
  if (!contract)
    fail("memory generator has no owner lineage contract");
  llvm::Error validation = contract->validateCanonical(
      encoded, parentReference, {parentReference}, store);
  if (!validation)
    fail("memory lineage accepted an out-of-range parent-local value");
  llvm::consumeError(std::move(validation));
  error = llvm::sys::fs::remove_directories(directory);
  if (error)
    fail("cannot remove ArtifactStore directory: " + error.message());
}

} // namespace

int main() {
  configRoundTripsAndRejectsMalformedBytes();
  constantGlobalStagingIsTypedAndMechanical();
  stagingPreservesAlignmentAndExpandsMemoryWork();
  scopeExpansionLimitIsInvocationWide();
  providerPublishesParentAndAdmittedChild();
  invalidInMemoryDecisionFailsClosed();
  lineageCodecRejectsAnOutOfRangeMemoryInput();
  return EXIT_SUCCESS;
}
