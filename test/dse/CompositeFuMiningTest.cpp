#include "DSE/CompositeFuMining.h"

#include "ADG/Builtin.h"
#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "DSE/FabricTemplateCandidateGenerator.h"
#include "DSE/TechMappingComposedSupply.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Dataflow/IR/DataflowDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"

#include <cstdlib>
#include <iostream>
#include <set>
#include <string>
#include <variant>
#include <vector>

namespace {

constexpr llvm::StringLiteral testName = "composite-fu-mining";

[[noreturn]] void fail(const std::string &message) {
  std::cerr << testName.str() << ": " << message << '\n';
  std::exit(EXIT_FAILURE);
}

void require(bool condition, llvm::StringRef message) {
  if (!condition)
    fail(message.str());
}

template <typename T> T take(llvm::Expected<T> value) {
  if (!value)
    fail(llvm::toString(value.takeError()));
  return std::move(*value);
}

class TemporaryDirectory final {
public:
  TemporaryDirectory() {
    std::error_code error =
        llvm::sys::fs::createUniqueDirectory("loom-composite-fu-mining", path_);
    if (error)
      fail("cannot create test directory: " + error.message());
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

dataflow::CanonicalDataflowArtifact loadDataflow(mlir::MLIRContext &context,
                                                 llvm::StringRef path) {
  auto source = llvm::MemoryBuffer::getFile(path);
  if (!source)
    fail("cannot read Dataflow graph set: " + source.getError().message());
  auto module =
      mlir::parseSourceString<mlir::ModuleOp>((*source)->getBuffer(), &context);
  if (!module)
    fail("cannot parse Dataflow fixture");
  return take(dataflow::finalizeCanonicalDataflow(*module));
}

/// Canonical finalization renames private graph symbols and orders the graphs
/// by its own walk, so each scene selects its pair by the width of the graphs'
/// first data operand: the widening pair takes i8 operands and the core pair
/// takes i64 operands.
std::vector<dataflow::GraphRef>
graphsWithOperandWidth(const dataflow::CanonicalDataflowProgramView &dataflow,
                       unsigned width) {
  std::vector<dataflow::GraphRef> refs;
  for (const dataflow::CanonicalGraphView &graph : dataflow.graphs()) {
    mlir::Block &body = graph.op->getRegion(0).front();
    auto operand =
        llvm::dyn_cast<mlir::IntegerType>(body.getArgument(1).getType());
    if (operand && operand.getWidth() == width)
      refs.push_back(graph.ref);
  }
  if (refs.size() != 2)
    fail("fixture does not hold exactly two graphs with i" +
         std::to_string(width) + " operands");
  return refs;
}

/// The registered operation schemas of one candidate's nodes, as a multiset
/// projection. Node order is canonical but automorphism-relative, so the test
/// pins the shape's inventory and its boundary rather than a labeling.
std::vector<std::uint32_t>
nodeSchemas(const loom::dse::CompositeFuCandidate &candidate) {
  std::vector<std::uint32_t> schemas;
  for (const loom::dse::CompositeFuNode &node : candidate.nodes)
    schemas.push_back(static_cast<std::uint32_t>(node.schema));
  llvm::sort(schemas);
  return schemas;
}

std::vector<std::uint64_t>
occurrenceActors(const loom::dse::CompositeFuOccurrence &occurrence) {
  std::vector<std::uint64_t> actors;
  for (dataflow::ActorRef actor : occurrence.actors)
    actors.push_back(actor.entity.value());
  llvm::sort(actors);
  return actors;
}

/// The int8 dot product with zero points lowers to two widening casts, two
/// zero-point subtractions, one multiply, and one accumulate. Two graphs that
/// share exactly that shape and differ in the tail operation must rank it
/// first: it covers the most actors of the pair for the fewest boundary ports.
void minesTheSharedMultiplyAccumulateShape(llvm::StringRef fixture) {
  mlir::MLIRContext context = makeContext();
  dataflow::CanonicalDataflowArtifact program = loadDataflow(context, fixture);
  const std::vector<dataflow::GraphRef> graphs = graphsWithOperandWidth(program.view(), 8);

  auto mined = take(loom::dse::mineCompositeFuCandidates(program.view(), graphs));
  const std::vector<loom::dse::CompositeFuCandidate> &candidates =
      mined.candidates;
  require(!candidates.empty(), "mining reported no common subgraph");
  const loom::dse::CompositeFuCandidate &best = candidates.front();
  require(!mined.bounded && mined.exploredActorCount >= best.nodes.size(),
          "a small fixture exhausted a mining bound");

  const std::vector<std::uint32_t> expected = {
      static_cast<std::uint32_t>(dataflow::OperationSchemaId::ArithAddI),
      static_cast<std::uint32_t>(dataflow::OperationSchemaId::ArithSubI),
      static_cast<std::uint32_t>(dataflow::OperationSchemaId::ArithSubI),
      static_cast<std::uint32_t>(dataflow::OperationSchemaId::ArithMulI),
      static_cast<std::uint32_t>(dataflow::OperationSchemaId::ArithExtSI),
      static_cast<std::uint32_t>(dataflow::OperationSchemaId::ArithExtSI)};
  std::vector<std::uint32_t> sortedExpected = expected;
  llvm::sort(sortedExpected);
  require(nodeSchemas(best) == sortedExpected,
          "the ranked candidate is not the widening multiply-accumulate shape");
  require(best.inputs.size() == 5 && best.outputs.size() == 1,
          "the ranked candidate does not present the two operands, the two "
          "zero points, the accumulator, and one result");
  require(best.internalEdges.size() == 5,
          "the ranked candidate lost an internal edge of its induced shape");
  require(best.graphCount == 2 && best.occurrences.size() == 2 &&
              best.coveredActorCount == 12,
          "the ranked candidate is not shared by both graphs");
  for (const loom::dse::CompositeFuOccurrence &occurrence : best.occurrences)
    require(occurrence.actors.size() == best.nodes.size(),
            "a mined occurrence does not bind every template node");
  require(occurrenceActors(best.occurrences.front()) !=
              occurrenceActors(best.occurrences.back()),
          "the two mined occurrences bind the same actors");
  for (const loom::dse::CompositeFuCandidate &candidate : candidates)
    require(candidate.score <= best.score,
            "mining did not report its candidates in rank order");

  // The widening cast and the ordinary integer datapath both have an inverse
  // capability policy, so this shape derives a complete hardware request
  // without any Fabric finalization.
  auto request = take(loom::dse::deriveCompositeFuTemplate(program.view(), best));
  require(request.nodes.size() == best.nodes.size() &&
              request.internalEdges.size() == best.internalEdges.size() &&
              request.inputs.size() == best.inputs.size() &&
              request.outputs.size() == best.outputs.size(),
          "the derived composite FU request lost a node or a boundary port");
  for (const loom::adg::CompositeFuNodeSpec &node : request.nodes)
    require(!node.enabledOperations.empty() && !node.outputTypes.empty(),
            "a derived composite FU node enables no operation");

  // The tail operations differ, so no shape that contains one is common: the
  // graph-support prune is what keeps mining from reporting a whole graph.
  for (const loom::dse::CompositeFuCandidate &candidate : candidates) {
    for (const loom::dse::CompositeFuNode &node : candidate.nodes)
      require(node.schema != dataflow::OperationSchemaId::ArithShRSI &&
                  node.schema != dataflow::OperationSchemaId::ArithMaxSI,
              "mining reported a candidate that only one graph contains");
    require(candidate.support >= 2,
            "mining reported a candidate below its minimum-image support");
  }
}

/// `S subset-of Materialize(F)`: synthesizing the ranked candidate publishes
/// one FU whose unique capability template binds every actor of every mined
/// occurrence through a complete ordered correspondence.
void synthesizesTheMinedTemplateBackToItsActors(llvm::StringRef fixture) {
  TemporaryDirectory directory;
  loom::ArtifactStore store(directory.path());
  mlir::MLIRContext context = makeContext();
  dataflow::CanonicalDataflowArtifact program = loadDataflow(context, fixture);
  const std::vector<dataflow::GraphRef> graphs = graphsWithOperandWidth(program.view(), 64);

  auto mined = take(loom::dse::mineCompositeFuCandidates(program.view(), graphs));
  const std::vector<loom::dse::CompositeFuCandidate> &candidates =
      mined.candidates;
  require(!candidates.empty(), "mining reported no common subgraph");
  require(!mined.bounded, "a small fixture exhausted a mining bound");
  const loom::dse::CompositeFuCandidate &best = candidates.front();
  require(best.nodes.size() == 4 && best.inputs.size() == 5 &&
              best.outputs.size() == 1,
          "the ranked candidate is not the four-actor multiply-accumulate "
          "shape");

  auto synthesized = take(
      loom::dse::synthesizeCompositeFuTemplate(program.view(), best, store));
  require(synthesized.module.view().fuTemplates().size() == 1,
          "composite FU synthesis published more than one FU definition");
  require(synthesized.module.view()
                  .resolvedFabricOpCapabilities(
                      synthesized.capabilityTemplate.fu)
                  .size() == best.nodes.size(),
          "the synthesized FU does not expose one operation per mined node");
  require(synthesized.coverage.size() == best.occurrences.size(),
          "composite FU synthesis lost a coverage witness");

  std::set<std::uint64_t> covered;
  for (const auto &indexed : llvm::enumerate(synthesized.coverage)) {
    const loom::dse::FuSynthesisCoverageWitness &witness = indexed.value();
    const loom::dse::CompositeFuOccurrence &occurrence =
        best.occurrences[indexed.index()];
    require(witness.graph == occurrence.graph,
            "a coverage witness names a graph outside its occurrence");
    require(witness.fabric == synthesized.module.view().identity() &&
                witness.capabilityTemplate == synthesized.capabilityTemplate,
            "a coverage witness names a foreign Fabric owner");
    require(witness.actors.size() == best.nodes.size(),
            "a coverage witness does not bind every mined actor");
    require(witness.boundaries.size() ==
                best.inputs.size() + best.outputs.size(),
            "a coverage witness does not bind every FU boundary port");
    for (const auto &actor : witness.actors)
      covered.insert(actor.actor.entity.value());
  }
  std::set<std::uint64_t> minedActors;
  for (const loom::dse::CompositeFuOccurrence &occurrence : best.occurrences)
    for (dataflow::ActorRef actor : occurrence.actors)
      minedActors.insert(actor.entity.value());
  require(covered == minedActors,
          "the synthesized FU does not materialize back to the exact mined "
          "actors");
}

/// The mined template reaches a Module by being placed while it is built. The
/// generator config carries only the selection, so this also exercises that the
/// generator re-mines the exact named Dataflow and refuses a shape it does not
/// mine.
void placesTheMinedTemplateInABuiltinModule(llvm::StringRef fixture) {
  TemporaryDirectory directory;
  loom::ArtifactStore store(directory.path());
  loom::BlobStore blobs((llvm::Twine(directory.path()) + "/blobs").str());
  mlir::MLIRContext context = makeContext();
  dataflow::CanonicalDataflowArtifact program = loadDataflow(context, fixture);
  const std::vector<dataflow::GraphRef> graphs =
      graphsWithOperandWidth(program.view(), 64);
  auto mined = take(loom::dse::mineCompositeFuCandidates(program.view(), graphs));
  const std::vector<loom::dse::CompositeFuCandidate> &candidates =
      mined.candidates;
  require(!candidates.empty(), "mining reported no common subgraph");
  auto published = take(dataflow::publishCanonicalDataflow(program, store));

  const loom::adg::BuiltinTargetDescriptor &descriptor =
      loom::adg::builtinSmallTarget;
  loom::dse::MinedCompositeFuSelection selection{program.identity(), {}};
  selection.templates.push_back({candidates.front().canonicalKey, 1});
  auto config = take(loom::dse::resolveFabricTemplateConfig(
      descriptor.templateIdentity, descriptor.schemaMajor,
      descriptor.schemaMinor, descriptor.scale, selection));
  require(config.minedCompositeFus().has_value(),
          "the resolved template config lost its mined selection");
  auto inputs =
      take(loom::dse::bindFabricTemplateCandidateGeneratorInputs(published));
  auto binding =
      take(loom::dse::resolveFabricTemplateCandidateGeneratorBinding(config));
  auto result = take(
      loom::dse::invokeCandidateGenerator(inputs, binding, store, blobs));
  const auto *completed =
      std::get_if<loom::dse::CompletedCandidateGeneratorResult>(
          &result.outcome);
  require(completed && completed->outputBindings.front().artifacts.size() == 1,
          "a template offering a mined composite FU published no System");

  // A shape the named Dataflow does not mine is a typed rejection, not a
  // silently catalog-only Module.
  loom::dse::MinedCompositeFuSelection foreign{program.identity(), {}};
  foreign.templates.push_back({{0, 1, 2, 3}, 1});
  auto foreignConfig = take(loom::dse::resolveFabricTemplateConfig(
      descriptor.templateIdentity, descriptor.schemaMajor,
      descriptor.schemaMinor, descriptor.scale, foreign));
  auto foreignBinding = take(
      loom::dse::resolveFabricTemplateCandidateGeneratorBinding(foreignConfig));
  auto rejected = loom::dse::invokeCandidateGenerator(inputs, foreignBinding,
                                                      store, blobs);
  if (rejected)
    fail("a foreign mined shape key produced a Module");
  llvm::consumeError(rejected.takeError());
}

/// One whole-layer-sized graph: a chain of multiply-accumulate motifs, which
/// is the shape and the scale a lowered layer actually presents. The fixture
/// is generated rather than written out because the case that matters is the
/// actor count, not the exact arithmetic.
std::string layerSizedProgram(unsigned motifs) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  stream << "module attributes {dlti.dl_spec = #dlti.dl_spec<"
            "#dlti.dl_entry<index, 64>>} {\n"
         << "  dataflow.graph private @layer(%start: none, %a: i64, "
            "%b: i64, %zp: i64, %acc: i64) -> i64\n"
         << "      attributes {input_segments = array<i32: 4, 0, 0>,\n"
         << "                  result_segments = array<i32: 1, 0, 0>} {\n";
  for (unsigned motif = 0; motif != motifs; ++motif) {
    stream << "    %ca" << motif << " = arith.subi %a, %zp : i64\n"
           << "    %cb" << motif << " = arith.subi %b, %zp : i64\n"
           << "    %p" << motif << " = arith.muli %ca" << motif << ", %cb"
           << motif << " : i64\n"
           << "    %s" << motif << " = arith.addi %p" << motif << ", ";
    if (motif == 0)
      stream << "%acc : i64\n";
    else
      stream << "%s" << (motif - 1) << " : i64\n";
  }
  stream << "    %result:2 = dataflow.sync %start, %s" << (motifs - 1)
         << " : (none, i64) -> (none, i64)\n"
         << "    dataflow.graph.return values(%result#1 : i64) streams() "
            "memories() complete(%result#0 : none)\n"
         << "  }\n"
         << "  dataflow.thread private @layer_worker "
            "domain(#dataflow.thread_domain<dense>)(\n"
         << "      %a: i64, %b: i64, %zp: i64, %acc: i64) ctrl (%ctrl: none) "
            "{\n"
         << "    %value, %done = dataflow.graph.launch @layer deps(%ctrl)\n"
         << "        values(%a, %b, %zp, %acc) stream_inputs() memories() "
            "stream_outputs()\n"
         << "        : (none, i64, i64, i64, i64) -> (i64, none)\n"
         << "    dataflow.thread.yield %done : none\n"
         << "  }\n"
         << "  func.func @application() {\n"
         << "    %a = arith.constant 3 : i64\n"
         << "    %b = arith.constant 5 : i64\n"
         << "    %zp = arith.constant 1 : i64\n"
         << "    %acc = arith.constant 7 : i64\n"
         << "    %thread = dataflow.thread.launch @layer_worker(%a, %b, %zp, "
            "%acc)\n"
         << "        : (i64, i64, i64, i64) -> !dataflow.thread_token\n"
         << "    return\n"
         << "  }\n"
         << "}\n";
  stream.flush();
  return text;
}

/// Mining one whole-layer graph with the request production uses. The search
/// is expected to reach its embedding bound at this size, and what matters is
/// that it reports the bound and still returns what it completely enumerated,
/// rather than failing or returning nothing.
void minesOneLayerSizedGraphUnderItsBound() {
  mlir::MLIRContext context = makeContext();
  const std::string source = layerSizedProgram(150);
  auto module = mlir::parseSourceString<mlir::ModuleOp>(source, &context);
  if (!module)
    fail("cannot parse the generated layer-sized program");
  dataflow::CanonicalDataflowArtifact program =
      take(dataflow::finalizeCanonicalDataflow(*module));
  std::vector<dataflow::GraphRef> graphs;
  for (const dataflow::CanonicalGraphView &graph : program.view().graphs())
    graphs.push_back(graph.ref);
  require(graphs.size() == 1, "the generated program is not one graph");

  auto mined = take(loom::dse::mineCompositeFuCandidates(
      program.view(), graphs, loom::dse::productionCompositeFuMiningLimits));
  require(!mined.candidates.empty(),
          "a layer-sized graph mined no candidate at all");
  require(mined.exploredActorCount >= 2,
          "a layer-sized graph completed no level");
  // Support is the single-graph measure: one shape repeated through the layer
  // is what makes a composite worth building, and no reported candidate may
  // fall below the request's minimum.
  for (const loom::dse::CompositeFuCandidate &candidate : mined.candidates)
    require(candidate.support >= 2 &&
                candidate.nodes.size() <= mined.exploredActorCount,
            "a reported candidate is below the support bound or above the "
            "completed level");
  const loom::dse::CompositeFuCandidate &best = mined.candidates.front();
  require(best.nodes.size() >= 2 && best.coveredActorCount >= best.nodes.size(),
          "the ranked candidate covers less than one realization");
}

/// The composed supply an exact Hall deficit gets. The relation is the one the
/// mlperf-tiny anomaly whole-layer candidates observe: a demand no existing
/// capability can close, whose only other answer is growing Temporal
/// residency. The proposal must name a template the miner reproduces and size
/// its occurrences from the deficit it answers.
void proposesAComposedSupplyForAHallDeficit(llvm::StringRef fixture) {
  TemporaryDirectory directory;
  loom::ArtifactStore store(directory.path());
  mlir::MLIRContext context = makeContext();
  dataflow::CanonicalDataflowArtifact program = loadDataflow(context, fixture);
  auto published = take(dataflow::publishCanonicalDataflow(program, store));

  loom::mapping::TechMappingComputeContextHallDemandGroup group;
  group.capabilities.push_back(
      {loom::fabric::FabricFuTemplateRef(1), 0});
  group.demandCount = 290;
  for (std::uint64_t context = 0; context != 80; ++context)
    group.compatibleContexts.push_back(
        {loom::fabric::FabricPeOccurrenceRef(context), 0});
  auto feedback =
      take(loom::mapping::TechMappingComputeContextHallDeficit::get(290, 80,
                                                                   {group}));
  require(feedback.deficit() == 210,
          "the fixture relation is not the observed deficit");

  const loom::adg::BuiltinTargetScale &scale =
      loom::adg::builtinCoverageTarget.scale;
  auto proposal = take(loom::dse::proposeMinedCompositeFuSupply(
      published, feedback, scale, store));
  require(proposal.has_value(),
          "a deficit over mineable software got no composed supply");
  require(proposal->selection.dataflow == program.identity() &&
              proposal->selection.templates.size() == 1,
          "the composed supply does not name exactly its own Dataflow and "
          "one template");
  require(proposal->actorsPerRealization >= 2 && proposal->support >= 2,
          "the composed supply named a template that is not composite or not "
          "repeated");
  const std::uint64_t needed =
      (feedback.deficit() + proposal->actorsPerRealization - 1) /
      proposal->actorsPerRealization;
  require(proposal->occurrences ==
              std::min<std::uint64_t>(needed, scale.spatialPeCount),
          "the composed supply is not sized from the deficit it answers");
  require(proposal->selection.templates.front().occurrences ==
              proposal->occurrences,
          "the selection and the proposal disagree on the site count");

  // The configuration carries only this key, so the generator must be able to
  // find it by mining the same Dataflow with the same request.
  std::vector<dataflow::GraphRef> graphs;
  for (const dataflow::CanonicalGraphView &graph : program.view().graphs())
    graphs.push_back(graph.ref);
  auto mined = take(loom::dse::mineCompositeFuCandidates(
      program.view(), graphs, loom::dse::productionCompositeFuMiningLimits));
  const auto found = llvm::find_if(
      mined.candidates, [&](const loom::dse::CompositeFuCandidate &candidate) {
        return candidate.canonicalKey ==
               proposal->selection.templates.front().shapeKey;
      });
  require(found != mined.candidates.end() &&
              found->nodes.size() == proposal->actorsPerRealization,
          "the named shape key is not one the same request reproduces");
}

/// Every port of a mined node takes part in the token relation, because the FU
/// physical model has no place for a result that reaches nothing: a
/// `fabric.op` result with no consumer derives no capability row and the
/// Fabric authored from it is invalid. This fixture makes the refused shape
/// the one the rank would otherwise take, since a result no actor consumes
/// costs no boundary port.
void minesOnlyShapesTheFuModelCanMaterialize(llvm::StringRef fixture) {
  TemporaryDirectory directory;
  loom::ArtifactStore store(directory.path());
  mlir::MLIRContext context = makeContext();
  dataflow::CanonicalDataflowArtifact program = loadDataflow(context, fixture);
  std::vector<dataflow::GraphRef> graphs;
  for (const dataflow::CanonicalGraphView &graph : program.view().graphs())
    graphs.push_back(graph.ref);
  require(graphs.size() == 1, "the fixture does not hold exactly one graph");

  auto mined = take(loom::dse::mineCompositeFuCandidates(
      program.view(), graphs, loom::dse::productionCompositeFuMiningLimits));
  require(!mined.candidates.empty(), "mining reported no common subgraph");

  for (const loom::dse::CompositeFuCandidate &candidate : mined.candidates)
    for (const auto &node : llvm::enumerate(candidate.nodes))
      for (std::uint64_t result = 0;
           result != node.value().type.getNumResults(); ++result) {
        const bool internal = llvm::any_of(
            candidate.internalEdges,
            [&](const loom::dse::CompositeFuInternalEdge &edge) {
              return edge.producerNode == node.index() &&
                     edge.producerResult == result;
            });
        const bool boundary = llvm::any_of(
            candidate.outputs,
            [&](const loom::dse::CompositeFuBoundaryPort &port) {
              return port.node == node.index() && port.portOrdinal == result;
            });
        require(internal || boundary,
                "a mined candidate holds a node result that reaches neither "
                "an internal edge nor an FU boundary port");
      }

  // The end-to-end oracle: the ranked candidate authors a Fabric whose FU
  // capability rows all derive. A dead physical result fails exactly here.
  const loom::dse::CompositeFuCandidate &best = mined.candidates.front();
  auto synthesized =
      take(loom::dse::synthesizeCompositeFuTemplate(program.view(), best, store));
  require(synthesized.coverage.size() == best.occurrences.size(),
          "composite FU synthesis lost a coverage witness");
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 3) {
    std::cerr << testName.str() << ": usage: " << argv[0]
              << " <dataflow.mlir> <mining|synthesis>\n";
    return EXIT_FAILURE;
  }
  const llvm::StringRef scene(argv[2]);
  if (scene == "mining")
    minesTheSharedMultiplyAccumulateShape(argv[1]);
  else if (scene == "synthesis")
    synthesizesTheMinedTemplateBackToItsActors(argv[1]);
  else if (scene == "placement")
    placesTheMinedTemplateInABuiltinModule(argv[1]);
  else if (scene == "scale")
    minesOneLayerSizedGraphUnderItsBound();
  else if (scene == "proposal")
    proposesAComposedSupplyForAHallDeficit(argv[1]);
  else if (scene == "materialization")
    minesOnlyShapesTheFuModelCanMaterialize(argv[1]);
  else
    fail("unknown scene " + scene.str());
  return EXIT_SUCCESS;
}
