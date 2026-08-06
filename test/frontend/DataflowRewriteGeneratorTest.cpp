#include "ADG/Builder.h"
#include "ADG/Builtin.h"
#include "ADG/FuLibrary.h"
#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "Config/ResolvedConfig.h"
#include "DSE/CandidateGenerator.h"
#include "DSE/DataflowRewriteCandidateGenerator.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Dataflow/IR/DataflowOps.h"
#include "Dataflow/Transforms/DataflowRewrite.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <cstdlib>
#include <system_error>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

static_assert(std::is_same_v<decltype(dataflow::ElementwiseVectorChunkRewrite::
                                          leadingBlocksPerChunk),
                             std::int64_t>);

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
    registry.insert<dataflow::DataflowDialect, mlir::LLVM::LLVMDialect>();
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

dataflow::CanonicalDataflowArtifact wideVectorAddProgram() {
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  dataflow.graph private @wide_add(
      %start: none, %lhs: vector<4xi32>, %rhs: vector<4xi32>)
      -> vector<4xi32>
      attributes {input_segments = array<i32: 2, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %sum = arith.addi %lhs, %rhs : vector<4xi32>
    %retired:2 = dataflow.sync %start, %sum
        : (none, vector<4xi32>) -> (none, vector<4xi32>)
    dataflow.graph.return values(%retired#1 : vector<4xi32>)
        streams() memories() complete(%retired#0 : none)
  }
}
)mlir",
                                                        &context());
  if (!module)
    fail("cannot parse the wide-vector rewrite fixture");
  return take(dataflow::finalizeCanonicalDataflow(module.get()));
}

dataflow::CanonicalDataflowArtifact twoWideVectorActorsProgram() {
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  dataflow.graph private @two_wide_actors(
      %start: none, %lhs: vector<4xi32>, %rhs: vector<4xi32>)
      -> vector<4xi32>
      attributes {input_segments = array<i32: 2, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %sum = arith.addi %lhs, %rhs : vector<4xi32>
    %product = arith.muli %sum, %rhs : vector<4xi32>
    %retired:2 = dataflow.sync %start, %product
        : (none, vector<4xi32>) -> (none, vector<4xi32>)
    dataflow.graph.return values(%retired#1 : vector<4xi32>)
        streams() memories() complete(%retired#0 : none)
  }
}
)mlir",
                                                        &context());
  if (!module)
    fail("cannot parse the recursive rewrite fixture");
  return take(dataflow::finalizeCanonicalDataflow(module.get()));
}

dataflow::CanonicalDataflowArtifact unitLeadingVectorAddProgram() {
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  dataflow.graph private @unit_leading_add(
      %start: none, %lhs: vector<1x4xi32>, %rhs: vector<1x4xi32>)
      -> vector<1x4xi32>
      attributes {input_segments = array<i32: 2, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %sum = arith.addi %lhs, %rhs : vector<1x4xi32>
    %retired:2 = dataflow.sync %start, %sum
        : (none, vector<1x4xi32>) -> (none, vector<1x4xi32>)
    dataflow.graph.return values(%retired#1 : vector<1x4xi32>)
        streams() memories() complete(%retired#0 : none)
  }
}
)mlir",
                                                        &context());
  if (!module)
    fail("cannot parse the unit-leading-vector rewrite fixture");
  return take(dataflow::finalizeCanonicalDataflow(module.get()));
}

dataflow::CanonicalDataflowArtifact vectorDivisionProgram() {
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  dataflow.graph private @vector_division(
      %start: none, %lhs: vector<4xi32>, %rhs: vector<4xi32>)
      -> vector<4xi32>
      attributes {input_segments = array<i32: 2, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %quotient = arith.divsi %lhs, %rhs : vector<4xi32>
    %retired:2 = dataflow.sync %start, %quotient
        : (none, vector<4xi32>) -> (none, vector<4xi32>)
    dataflow.graph.return values(%retired#1 : vector<4xi32>)
        streams() memories() complete(%retired#0 : none)
  }
}
)mlir",
                                                        &context());
  if (!module)
    fail("cannot parse the vector-division rewrite fixture");
  return take(dataflow::finalizeCanonicalDataflow(module.get()));
}

dataflow::CanonicalDataflowArtifact saturatingVectorAddProgram() {
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  dataflow.graph private @saturating_add(
      %start: none, %lhs: vector<4xi8>, %rhs: vector<4xi8>)
      -> vector<4xi8>
      attributes {input_segments = array<i32: 2, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %sum = llvm.intr.sadd.sat(%lhs, %rhs)
        : (vector<4xi8>, vector<4xi8>) -> vector<4xi8>
    %retired:2 = dataflow.sync %start, %sum
        : (none, vector<4xi8>) -> (none, vector<4xi8>)
    dataflow.graph.return values(%retired#1 : vector<4xi8>)
        streams() memories() complete(%retired#0 : none)
  }
}
)mlir",
                                                        &context());
  if (!module)
    fail("cannot parse the saturating-vector rewrite fixture");
  return take(dataflow::finalizeCanonicalDataflow(module.get()));
}

dataflow::CanonicalDataflowArtifact vectorCompareProgram() {
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  dataflow.graph private @compare(
      %start: none, %lhs: vector<4xi32>, %rhs: vector<4xi32>)
      -> vector<4xi1>
      attributes {input_segments = array<i32: 2, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %comparison = arith.cmpi slt, %lhs, %rhs
        : vector<4xi32>
    %retired:2 = dataflow.sync %start, %comparison
        : (none, vector<4xi1>) -> (none, vector<4xi1>)
    dataflow.graph.return values(%retired#1 : vector<4xi1>)
        streams() memories() complete(%retired#0 : none)
  }
}
)mlir",
                                                        &context());
  if (!module)
    fail("cannot parse the vector-compare rewrite fixture");
  return take(dataflow::finalizeCanonicalDataflow(module.get()));
}

dataflow::CanonicalDataflowArtifact saturatingFloatConversionProgram() {
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  dataflow.graph private @saturating_convert(
      %start: none, %input: vector<4xf32>) -> vector<4xi16>
      attributes {input_segments = array<i32: 1, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %converted = llvm.call_intrinsic "llvm.fptosi.sat.v4i16.v4f32"(%input)
        : (vector<4xf32>) -> vector<4xi16>
    %retired:2 = dataflow.sync %start, %converted
        : (none, vector<4xi16>) -> (none, vector<4xi16>)
    dataflow.graph.return values(%retired#1 : vector<4xi16>)
        streams() memories() complete(%retired#0 : none)
  }
}
)mlir",
                                                        &context());
  if (!module)
    fail("cannot parse the saturating-conversion rewrite fixture");
  return take(dataflow::finalizeCanonicalDataflow(module.get()));
}

dataflow::ActorRef
actorForSchema(const dataflow::CanonicalDataflowArtifact &artifact,
               dataflow::OperationSchemaId schema) {
  auto view = take(artifact.view());
  for (const dataflow::CanonicalActorView &actor : view.actors())
    if (dataflow::operationSchemaOf(actor.op) == schema)
      return actor.ref;
  fail("canonical fixture does not contain the requested actor schema");
}

loom::adg::FinalizedFabricDesign
narrowVectorComputeFabric(const loom::ArtifactStore &store) {
  using namespace loom::adg;
  DesignBuilder design(store);
  const PortType outer = take(PortType::bits(192));
  std::vector<PortType> boundary(5, outer);
  auto spatial =
      take(design.createSpatialCore("narrow-vector", boundary, {outer}));
  std::vector<SpatialValue> spatialInputs;
  for (std::size_t ordinal = 0; ordinal != boundary.size(); ++ordinal)
    spatialInputs.push_back(take(spatial.input(ordinal)));
  auto pe = take(spatial.addPe(
      spatialInputs, PeSpec::spatial(boundary, {outer, outer, outer, outer})));
  std::vector<PeValue> peInputs;
  for (std::size_t ordinal = 0; ordinal != boundary.size(); ++ordinal)
    peInputs.push_back(take(pe.input(ordinal)));

  if (llvm::Error error = addVectorComputeFu(
          pe, {peInputs[0], peInputs[1], peInputs[2], peInputs[3]},
          VectorComputeFuParameters{192, 64}))
    fail(llvm::toString(std::move(error)));
  const fabric::IntegerWidthSet integers = fabric::IntegerWidthSet::get(
      {fabric::IntegerWidth::I8, fabric::IntegerWidth::I16,
       fabric::IntegerWidth::I32, fabric::IntegerWidth::I64});
  const fabric::FloatFormatSet floats = fabric::FloatFormatSet::get(
      {fabric::FloatFormat::F16, fabric::FloatFormat::BF16,
       fabric::FloatFormat::F32, fabric::FloatFormat::F64});
  const VectorStructuralFuParameters structural{
      192, 192, 64,
      fabric::FixedVectorSliceAlignMergeParams{
          integers, floats, 128, 32, 0, fabric::ResolvedIndexWidthSet::get({})},
      fabric::FixedVectorShuffleParams{integers, floats, 128, 128, 32, 8, 4}};
  if (llvm::Error error =
          addVectorStructuralFu(pe, {peInputs[0], peInputs[1]}, structural))
    fail(llvm::toString(std::move(error)));
  if (llvm::Error error =
          addTokenControlFu(pe, peInputs, TokenControlFuParameters{192, 64}))
    fail(llvm::toString(std::move(error)));
  if (llvm::Error error = pe.close())
    fail(llvm::toString(std::move(error)));
  if (llvm::Error error = spatial.close({take(pe.output(0))}))
    fail(llvm::toString(std::move(error)));
  return take(std::move(design).finalize());
}

std::pair<unsigned, unsigned>
adapterCounts(const dataflow::CanonicalDataflowArtifact &artifact) {
  unsigned packs = 0;
  unsigned unpacks = 0;
  artifact.module().walk([&](dataflow::PackOp) { ++packs; });
  artifact.module().walk([&](dataflow::UnpackOp) { ++unpacks; });
  return {packs, unpacks};
}

bool hasMask(mlir::vector::ShuffleOp shuffle,
             llvm::ArrayRef<std::int64_t> expected) {
  return shuffle.getMask() == expected;
}

void verifyExactTwoChunkWiring(
    const dataflow::CanonicalDataflowArtifact &artifact) {
  dataflow::GraphOp graph;
  artifact.module().walk([&](dataflow::GraphOp candidate) {
    if (!graph)
      graph = candidate;
  });
  if (!graph)
    fail("chunk candidate has no graph");
  mlir::Block &body = graph.getBody().front();
  if (body.getNumArguments() != 3)
    fail("chunk candidate changed the graph boundary");

  dataflow::SyncOp operandSync;
  dataflow::SyncOp retirementSync;
  for (dataflow::SyncOp sync : body.getOps<dataflow::SyncOp>()) {
    if (sync.getInputs().size() == 2 &&
        llvm::all_of(sync.getInputs(), [](mlir::Value input) {
          return llvm::isa<mlir::VectorType>(input.getType());
        })) {
      if (operandSync)
        fail("chunk candidate has more than one operand rendezvous");
      operandSync = sync;
    } else if (sync.getInputs().size() == 2 &&
               llvm::isa<mlir::NoneType>(sync.getInputs().front().getType())) {
      retirementSync = sync;
    }
  }
  if (!operandSync || !retirementSync ||
      operandSync.getInputs()[0] != body.getArgument(1) ||
      operandSync.getInputs()[1] != body.getArgument(2))
    fail("chunk candidate does not rendezvous the exact graph operands");

  std::array<std::array<mlir::vector::ShuffleOp, 2>, 2> extracts;
  mlir::vector::ShuffleOp concatenation;
  for (mlir::vector::ShuffleOp shuffle :
       body.getOps<mlir::vector::ShuffleOp>()) {
    bool classified = false;
    for (unsigned operand = 0; operand != 2; ++operand) {
      if (shuffle.getV1() != operandSync.getOutputs()[operand] ||
          shuffle.getV2() != operandSync.getOutputs()[operand])
        continue;
      if (hasMask(shuffle, {0, 1})) {
        extracts[operand][0] = shuffle;
        classified = true;
      } else if (hasMask(shuffle, {2, 3})) {
        extracts[operand][1] = shuffle;
        classified = true;
      }
    }
    if (!classified && hasMask(shuffle, {0, 1, 2, 3})) {
      if (concatenation)
        fail("chunk candidate has multiple result concatenations");
      concatenation = shuffle;
    }
  }
  if (!concatenation || !extracts[0][0] || !extracts[0][1] || !extracts[1][0] ||
      !extracts[1][1])
    fail("chunk candidate has incorrect leading-block shuffle masks");

  std::array<mlir::arith::AddIOp, 2> chunkAdds;
  for (mlir::arith::AddIOp add : body.getOps<mlir::arith::AddIOp>()) {
    for (unsigned chunk = 0; chunk != 2; ++chunk)
      if (add.getLhs() == extracts[0][chunk].getVector() &&
          add.getRhs() == extracts[1][chunk].getVector())
        chunkAdds[chunk] = add;
  }
  if (!chunkAdds[0] || !chunkAdds[1] ||
      concatenation.getV1() != chunkAdds[0].getResult() ||
      concatenation.getV2() != chunkAdds[1].getResult())
    fail("chunk candidate crossed or reversed operand/result blocks");

  if (retirementSync.getInputs()[0] != body.getArgument(0) ||
      retirementSync.getInputs()[1] != concatenation.getVector())
    fail("chunk candidate detached its result from retirement");
  auto returned = mlir::cast<dataflow::GraphReturnOp>(body.getTerminator());
  if (returned.getValues().size() != 1 || returned.getComplete().size() != 1 ||
      returned.getValues().front() != retirementSync.getOutputs()[1] ||
      returned.getComplete().front() != retirementSync.getOutputs()[0])
    fail("chunk candidate does not publish the wired result at graph return");
}

void exactParentAndOneAtomicChildArePublished() {
  llvm::SmallString<128> directory;
  std::error_code error = llvm::sys::fs::createUniqueDirectory(
      "loom-dataflow-rewrite-generator", directory);
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
  auto parent = roundTripProgram();
  const std::vector<std::uint8_t> parentBytes(
      parent.canonicalBytes().bytes().begin(),
      parent.canonicalBytes().bytes().end());
  auto parentReference =
      take(dataflow::publishCanonicalDataflow(parent, store));

  auto config =
      take(loom::dse::projectResolvedDataflowRewriteGeneratorConfigView(
          loom::defaultResolvedConfig()));
  auto inputs = take(loom::dse::bindDataflowRewriteCandidateGeneratorInputs(
      {parentReference}, design.roots().front().reference()));
  auto binding =
      take(loom::dse::resolveDataflowRewriteCandidateGeneratorBinding(config));
  auto outcome =
      take(loom::dse::invokeCandidateGenerator(inputs, binding, store, blobs));
  auto *completed = std::get_if<loom::dse::CompletedCandidateGeneratorResult>(
      &outcome.outcome);
  if (!completed || completed->outputBindings.size() != 1 ||
      completed->outputBindings.front().artifacts.size() != 2)
    fail("generator did not publish the parent and one atomic child");
  if (completed->lineageEdges.size() != 1 ||
      completed->lineageEdges.front().parents !=
          std::vector<loom::ArtifactRootReference>{parentReference})
    fail("generator lost its exact atomic rewrite lineage");
  take(dataflow::adoptDataflowRewriteDecision(
      completed->lineageEdges.front().ownerPayload));

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

void configRoundTripsAndRejectsZeroLimit() {
  loom::ResolvedConfig resolved = loom::defaultResolvedConfig();
  resolved.dse.dataflowRewrite.scopeExpansionLimit = 7;
  auto projected = take(
      loom::dse::projectResolvedDataflowRewriteGeneratorConfigView(resolved));
  auto adopted =
      take(loom::dse::adoptResolvedDataflowRewriteGeneratorConfigView(
          loom::dse::resolvedDataflowRewriteGeneratorConfigSchemaBytes(),
          projected.canonicalViewBytes(), projected.digest()));
  if (adopted.scopeExpansionLimit() != 7 ||
      adopted.canonicalViewBytes() != projected.canonicalViewBytes())
    fail("dataflow rewrite config did not round-trip its semantic limit");

  loom::ResolvedConfig distinct = resolved;
  distinct.dse.dataflowRewrite.scopeExpansionLimit = 8;
  auto distinctView = take(
      loom::dse::projectResolvedDataflowRewriteGeneratorConfigView(distinct));
  if (distinctView.digest() == projected.digest() ||
      distinctView.canonicalViewBytes() == projected.canonicalViewBytes())
    fail("scope expansion limit is absent from config identity");

  std::vector<std::uint8_t> changed(projected.canonicalViewBytes().begin(),
                                    projected.canonicalViewBytes().end());
  changed.back() ^= 1;
  auto staleDigest = loom::dse::adoptResolvedDataflowRewriteGeneratorConfigView(
      loom::dse::resolvedDataflowRewriteGeneratorConfigSchemaBytes(), changed,
      projected.digest());
  if (staleDigest)
    fail("dataflow rewrite config accepted a stale component digest");
  llvm::consumeError(staleDigest.takeError());

  std::vector<std::uint8_t> trailing(projected.canonicalViewBytes().begin(),
                                     projected.canonicalViewBytes().end());
  trailing.push_back(0);
  auto trailingDigest = take(loom::computeComponentViewDigest(
      loom::dse::resolvedDataflowRewriteGeneratorConfigSchemaBytes(),
      trailing));
  auto trailingView =
      loom::dse::adoptResolvedDataflowRewriteGeneratorConfigView(
          loom::dse::resolvedDataflowRewriteGeneratorConfigSchemaBytes(),
          trailing, trailingDigest);
  if (trailingView)
    fail("dataflow rewrite config accepted trailing canonical bytes");
  llvm::consumeError(trailingView.takeError());

  resolved.dse.dataflowRewrite.scopeExpansionLimit = 0;
  auto invalid =
      loom::dse::projectResolvedDataflowRewriteGeneratorConfigView(resolved);
  if (invalid)
    fail("dataflow rewrite config accepted a zero semantic limit");
  llvm::consumeError(invalid.takeError());
}

void invalidInMemoryDecisionFailsClosed() {
  auto parent = roundTripProgram();
  const dataflow::DataflowRewriteDecision decision =
      static_cast<dataflow::DataflowRewriteKind>(99);
  auto encoded = dataflow::encodeDataflowRewriteDecision(decision);
  if (encoded)
    fail("Dataflow encoder accepted an unknown in-memory rewrite kind");
  llvm::consumeError(encoded.takeError());
  auto materialized = dataflow::materializeDataflowRewrite(parent, decision);
  if (materialized)
    fail("Dataflow materializer accepted an unknown in-memory rewrite kind");
  llvm::consumeError(materialized.takeError());

  auto rewritePass = dataflow::createDataflowRewritePass(
      static_cast<dataflow::DataflowRewriteKind>(99));
  if (rewritePass)
    fail("Dataflow pass factory accepted an unknown in-memory rewrite kind");
  llvm::consumeError(rewritePass.takeError());
}

void lineageCodecRejectsAnOutOfRangeActor() {
  llvm::SmallString<128> directory;
  std::error_code error = llvm::sys::fs::createUniqueDirectory(
      "loom-dataflow-lineage-context", directory);
  if (error)
    fail("cannot create ArtifactStore directory: " + error.message());
  loom::ArtifactStore store(directory);
  auto parent = roundTripProgram();
  auto parentReference =
      take(dataflow::publishCanonicalDataflow(parent, store));
  const dataflow::DataflowRewriteDecision decision =
      dataflow::ElementwiseVectorScalarizeRewrite{
          dataflow::ActorRef{parent.identity(), dataflow::ActorId(999999)}};
  auto encoded = take(dataflow::encodeDataflowRewriteDecision(decision));
  const auto *contract =
      loom::dse::dataflowRewriteCandidateGeneratorDescriptor()
          .ownerLineagePayload;
  if (!contract)
    fail("Dataflow generator has no owner lineage contract");
  llvm::Error validation =
      contract->validateCanonical(encoded, {parentReference}, store);
  if (!validation)
    fail("Dataflow lineage accepted an out-of-range parent-local actor");
  llvm::consumeError(std::move(validation));
  error = llvm::sys::fs::remove_directories(directory);
  if (error)
    fail("cannot remove ArtifactStore directory: " + error.message());
}

void wideVectorActorIsChunkedForExactNarrowComputeFabric() {
  llvm::SmallString<128> directory;
  std::error_code error = llvm::sys::fs::createUniqueDirectory(
      "loom-dataflow-vector-chunk-generator", directory);
  if (error)
    fail("cannot create ArtifactStore directory: " + error.message());
  loom::ArtifactStore store(directory);
  llvm::SmallString<128> blobPath(directory);
  llvm::sys::path::append(blobPath, "blobs");
  if (std::error_code error = llvm::sys::fs::create_directories(blobPath))
    fail("cannot create BlobStore directory: " + error.message());
  const loom::BlobStore blobs(blobPath);
  auto design = narrowVectorComputeFabric(store);
  auto parent = wideVectorAddProgram();
  auto parentReference =
      take(dataflow::publishCanonicalDataflow(parent, store));

  loom::ResolvedConfig resolved = loom::defaultResolvedConfig();
  resolved.dse.dataflowRewrite.scopeExpansionLimit = 16;
  auto config = take(
      loom::dse::projectResolvedDataflowRewriteGeneratorConfigView(resolved));
  auto inputs = take(loom::dse::bindDataflowRewriteCandidateGeneratorInputs(
      {parentReference}, design.roots().front().reference()));
  auto binding =
      take(loom::dse::resolveDataflowRewriteCandidateGeneratorBinding(config));
  auto outcome =
      take(loom::dse::invokeCandidateGenerator(inputs, binding, store, blobs));
  auto *completed = std::get_if<loom::dse::CompletedCandidateGeneratorResult>(
      &outcome.outcome);
  if (!completed || completed->outputBindings.size() != 1 ||
      completed->outputBindings.front().artifacts.empty())
    fail("narrow vector Fabric produced no admitted chunk candidate");

  bool sawTwoChunks = false;
  for (const loom::ArtifactRootReference &reference :
       completed->outputBindings.front().artifacts) {
    if (reference == parentReference)
      fail("generator admitted the unsupported wide parent");
    auto child = take(dataflow::importCanonicalDataflow(reference, store));
    unsigned narrowAdds = 0;
    unsigned wideAdds = 0;
    unsigned shuffles = 0;
    unsigned jointOperandSyncs = 0;
    child.module().walk([&](mlir::arith::AddIOp add) {
      auto vector = llvm::dyn_cast<mlir::VectorType>(add.getType());
      if (vector && vector.getShape() == llvm::ArrayRef<std::int64_t>{2})
        ++narrowAdds;
      if (vector && vector.getShape() == llvm::ArrayRef<std::int64_t>{4})
        ++wideAdds;
    });
    child.module().walk([&](mlir::vector::ShuffleOp) { ++shuffles; });
    child.module().walk([&](dataflow::SyncOp sync) {
      jointOperandSyncs +=
          llvm::all_of(sync.getInputs(), [](mlir::Value input) {
            return llvm::isa<mlir::VectorType>(input.getType());
          });
    });
    if (narrowAdds == 2 && wideAdds == 0 && shuffles == 5 &&
        jointOperandSyncs == 1) {
      verifyExactTwoChunkWiring(child);
      sawTwoChunks = true;
    }
  }
  if (!sawTwoChunks)
    fail("generator did not materialize the exact two-chunk vector graph");

  std::error_code cleanup = llvm::sys::fs::remove_directories(directory);
  if (cleanup)
    fail("cannot remove ArtifactStore directory: " + cleanup.message());
}

void recursiveRewriteRetainsItsRootedInternalLineage() {
  llvm::SmallString<128> directory;
  std::error_code error = llvm::sys::fs::createUniqueDirectory(
      "loom-dataflow-recursive-rewrite-lineage", directory);
  if (error)
    fail("cannot create ArtifactStore directory: " + error.message());
  loom::ArtifactStore store(directory);
  llvm::SmallString<128> blobPath(directory);
  llvm::sys::path::append(blobPath, "blobs");
  if (std::error_code error = llvm::sys::fs::create_directories(blobPath))
    fail("cannot create BlobStore directory: " + error.message());
  const loom::BlobStore blobs(blobPath);
  auto design = narrowVectorComputeFabric(store);
  auto parentReference = take(
      dataflow::publishCanonicalDataflow(twoWideVectorActorsProgram(), store));

  loom::ResolvedConfig resolved = loom::defaultResolvedConfig();
  resolved.dse.dataflowRewrite.scopeExpansionLimit = 64;
  auto config = take(
      loom::dse::projectResolvedDataflowRewriteGeneratorConfigView(resolved));
  auto inputs = take(loom::dse::bindDataflowRewriteCandidateGeneratorInputs(
      {parentReference}, design.roots().front().reference()));
  auto binding =
      take(loom::dse::resolveDataflowRewriteCandidateGeneratorBinding(config));
  auto outcome =
      take(loom::dse::invokeCandidateGenerator(inputs, binding, store, blobs));
  auto *completed = std::get_if<loom::dse::CompletedCandidateGeneratorResult>(
      &outcome.outcome);
  if (!completed || completed->outputBindings.size() != 1 ||
      completed->outputBindings.front().artifacts.empty())
    fail("recursive rewrite produced no admitted final candidate");

  const auto &outputs = completed->outputBindings.front().artifacts;
  bool sawInternal = false;
  bool sawFinalDependingOnInternal = false;
  for (const loom::dse::CandidateGeneratorLineageEdge &edge :
       completed->lineageEdges) {
    const bool returned = llvm::is_contained(outputs, edge.output);
    sawInternal |= !returned;
    if (!returned)
      continue;
    sawFinalDependingOnInternal |= llvm::any_of(
        edge.parents, [&](const loom::ArtifactRootReference &parent) {
          return llvm::any_of(
              completed->lineageEdges,
              [&](const loom::dse::CandidateGeneratorLineageEdge &producer) {
                return producer.output == parent &&
                       !llvm::is_contained(outputs, producer.output);
              });
        });
  }
  if (!sawInternal || !sawFinalDependingOnInternal)
    fail("recursive rewrite discarded its internal production path");

  std::error_code cleanup = llvm::sys::fs::remove_directories(directory);
  if (cleanup)
    fail("cannot remove ArtifactStore directory: " + cleanup.message());
}

void semanticLimitNeverPromotesAnExploredPrefix() {
  llvm::SmallString<128> directory;
  std::error_code error = llvm::sys::fs::createUniqueDirectory(
      "loom-dataflow-vector-chunk-limit", directory);
  if (error)
    fail("cannot create ArtifactStore directory: " + error.message());
  loom::ArtifactStore store(directory);
  llvm::SmallString<128> blobPath(directory);
  llvm::sys::path::append(blobPath, "blobs");
  if (std::error_code error = llvm::sys::fs::create_directories(blobPath))
    fail("cannot create BlobStore directory: " + error.message());
  const loom::BlobStore blobs(blobPath);
  auto design = narrowVectorComputeFabric(store);
  auto parent = wideVectorAddProgram();
  auto parentReference =
      take(dataflow::publishCanonicalDataflow(parent, store));

  loom::ResolvedConfig resolved = loom::defaultResolvedConfig();
  // Three fixed rule probes plus the two narrow actors of the first chunk
  // candidate consume the complete limit. That child is admissible, but the
  // remaining typed decisions have not been explored.
  resolved.dse.dataflowRewrite.scopeExpansionLimit = 5;
  auto config = take(
      loom::dse::projectResolvedDataflowRewriteGeneratorConfigView(resolved));
  auto inputs = take(loom::dse::bindDataflowRewriteCandidateGeneratorInputs(
      {parentReference}, design.roots().front().reference()));
  auto binding =
      take(loom::dse::resolveDataflowRewriteCandidateGeneratorBinding(config));
  auto outcome =
      take(loom::dse::invokeCandidateGenerator(inputs, binding, store, blobs));
  auto *incomplete = std::get_if<loom::dse::IncompleteCandidateGeneratorResult>(
      &outcome.outcome);
  if (!incomplete ||
      incomplete->reason !=
          loom::dse::CandidateGeneratorIncompleteReason::SemanticLimitReached ||
      incomplete->retainedOutputBindings.size() != 1 ||
      incomplete->retainedOutputBindings.front().artifacts.size() != 1)
    fail("semantic limit promoted or discarded the explored candidate prefix");

  auto retained = take(dataflow::importCanonicalDataflow(
      incomplete->retainedOutputBindings.front().artifacts.front(), store));
  unsigned retainedNarrowAdds = 0;
  retained.module().walk([&](mlir::arith::AddIOp add) {
    auto type = llvm::dyn_cast<mlir::VectorType>(add.getType());
    retainedNarrowAdds +=
        type && type.getShape() == llvm::ArrayRef<std::int64_t>{2};
  });
  if (retainedNarrowAdds != 2)
    fail("semantic limit retained a different candidate identity");

  resolved.dse.dataflowRewrite.scopeExpansionLimit = 4;
  config = take(
      loom::dse::projectResolvedDataflowRewriteGeneratorConfigView(resolved));
  binding =
      take(loom::dse::resolveDataflowRewriteCandidateGeneratorBinding(config));
  outcome =
      take(loom::dse::invokeCandidateGenerator(inputs, binding, store, blobs));
  incomplete = std::get_if<loom::dse::IncompleteCandidateGeneratorResult>(
      &outcome.outcome);
  if (!incomplete ||
      incomplete->reason !=
          loom::dse::CandidateGeneratorIncompleteReason::SemanticLimitReached ||
      incomplete->retainedOutputBindings.size() != 1 ||
      !incomplete->retainedOutputBindings.front().artifacts.empty())
    fail("semantic limit admitted a candidate before its exact charge fit");

  std::error_code cleanup = llvm::sys::fs::remove_directories(directory);
  if (cleanup)
    fail("cannot remove ArtifactStore directory: " + cleanup.message());
}

void decompositionUsesRegisteredSoftwareSemanticsAndOneValidator() {
  auto division = vectorDivisionProgram();
  const dataflow::ActorRef divisionActor =
      actorForSchema(division, dataflow::OperationSchemaId::ArithDivSI);
  if (!take(dataflow::enumerateElementwiseVectorDecompositionDecisions(
                division, divisionActor))
           .empty())
    fail("non-total integer division entered the decomposition domain");

  auto unitLeading = unitLeadingVectorAddProgram();
  const dataflow::ActorRef unitLeadingActor =
      actorForSchema(unitLeading, dataflow::OperationSchemaId::ArithAddI);
  auto unitLeadingDecisions =
      take(dataflow::enumerateElementwiseVectorDecompositionDecisions(
          unitLeading, unitLeadingActor));
  if (unitLeadingDecisions.size() != 1 ||
      !std::holds_alternative<dataflow::ElementwiseVectorScalarizeRewrite>(
          unitLeadingDecisions.front()))
    fail("unit leading extent did not retain exact scalarization");
  auto unitLeadingScalarized = take(dataflow::materializeDataflowRewrite(
      unitLeading, unitLeadingDecisions.front()));
  if (!unitLeadingScalarized)
    fail("unit-leading scalarization produced no candidate");
  unsigned unitLeadingAdds = 0;
  unitLeadingScalarized->module().walk([&](mlir::arith::AddIOp add) {
    unitLeadingAdds += add.getType().isInteger(32);
  });
  if (unitLeadingAdds != 4)
    fail("unit-leading scalarization did not cover every row-major element");

  auto saturating = saturatingVectorAddProgram();
  const dataflow::ActorRef saturatingActor =
      actorForSchema(saturating, dataflow::OperationSchemaId::LLVMSAddSat);
  auto decisions =
      take(dataflow::enumerateElementwiseVectorDecompositionDecisions(
          saturating, saturatingActor));
  bool sawTwoBlockChunk = false;
  for (const dataflow::DataflowRewriteDecision &decision : decisions)
    if (const auto *chunk =
            std::get_if<dataflow::ElementwiseVectorChunkRewrite>(&decision))
      sawTwoBlockChunk |= chunk->leadingBlocksPerChunk == 2;
  if (!sawTwoBlockChunk)
    fail("registered saturating vector arithmetic was excluded from the "
         "elementwise decomposition domain");

  auto scalarized = take(dataflow::materializeDataflowRewrite(
      saturating,
      dataflow::ElementwiseVectorScalarizeRewrite{saturatingActor}));
  if (!scalarized)
    fail("legal saturating vector scalarization produced no candidate");
  unsigned scalarSaturatingAdds = 0;
  unsigned extracts = 0;
  unsigned inserts = 0;
  scalarized->module().walk([&](mlir::LLVM::SAddSat op) {
    if (op.getType().isInteger(8))
      ++scalarSaturatingAdds;
  });
  scalarized->module().walk([&](mlir::vector::ExtractOp) { ++extracts; });
  scalarized->module().walk([&](mlir::vector::InsertOp) { ++inserts; });
  if (scalarSaturatingAdds != 4 || extracts != 8 || inserts != 4)
    fail("scalarization did not materialize the exact row-major actor graph");

  auto invalidChunkCost = dataflow::dataflowRewriteExpansionCost(
      saturating, dataflow::ElementwiseVectorChunkRewrite{saturatingActor, 4});
  if (invalidChunkCost)
    fail("rewrite cost accepted a non-proper chunk decision");
  llvm::consumeError(invalidChunkCost.takeError());

  auto comparison = vectorCompareProgram();
  const dataflow::ActorRef comparisonActor =
      actorForSchema(comparison, dataflow::OperationSchemaId::ArithCmpI);
  auto invalidScalarCost = dataflow::dataflowRewriteExpansionCost(
      comparison, dataflow::ElementwiseVectorScalarizeRewrite{comparisonActor});
  if (invalidScalarCost)
    fail("rewrite cost accepted scalarization without a result-typed base");
  llvm::consumeError(invalidScalarCost.takeError());

  auto comparisonChunk = take(dataflow::materializeDataflowRewrite(
      comparison, dataflow::ElementwiseVectorChunkRewrite{comparisonActor, 2}));
  if (!comparisonChunk)
    fail("result-changing elementwise actor produced no chunk candidate");
  unsigned narrowComparisons = 0;
  comparisonChunk->module().walk([&](mlir::arith::CmpIOp comparison) {
    auto type = llvm::dyn_cast<mlir::VectorType>(comparison.getType());
    if (type && type.getShape() == llvm::ArrayRef<std::int64_t>{2} &&
        type.getElementType().isInteger(1))
      ++narrowComparisons;
  });
  if (narrowComparisons != 2)
    fail("chunk rewrite lost the comparison result element type");

  auto conversion = saturatingFloatConversionProgram();
  const dataflow::ActorRef conversionActor =
      actorForSchema(conversion, dataflow::OperationSchemaId::LLVMFPToSISat);
  auto conversionChunk = take(dataflow::materializeDataflowRewrite(
      conversion, dataflow::ElementwiseVectorChunkRewrite{conversionActor, 2}));
  if (!conversionChunk)
    fail("registered intrinsic carrier produced no chunk candidate");
  unsigned canonicalIntrinsics = 0;
  conversionChunk->module().walk([&](mlir::LLVM::CallIntrinsicOp intrinsic) {
    if (intrinsic.getIntrin() == "llvm.fptosi.sat.v2i16.v2f32" &&
        dataflow::operationSchemaOf(intrinsic) ==
            dataflow::OperationSchemaId::LLVMFPToSISat)
      ++canonicalIntrinsics;
  });
  if (canonicalIntrinsics != 2)
    fail("chunk rewrite did not regenerate the intrinsic overload spelling");
}

} // namespace

int main() {
  configRoundTripsAndRejectsZeroLimit();
  invalidInMemoryDecisionFailsClosed();
  lineageCodecRejectsAnOutOfRangeActor();
  exactParentAndOneAtomicChildArePublished();
  wideVectorActorIsChunkedForExactNarrowComputeFabric();
  recursiveRewriteRetainsItsRootedInternalLineage();
  semanticLimitNeverPromotesAnExploredPrefix();
  decompositionUsesRegisteredSoftwareSemanticsAndOneValidator();
  return EXIT_SUCCESS;
}
