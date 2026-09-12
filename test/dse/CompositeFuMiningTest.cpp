#include "DSE/CompositeFuMining.h"

#include "ADG/Builtin.h"
#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "DSE/FabricTemplateCandidateGenerator.h"
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

  auto candidates =
      take(loom::dse::mineCompositeFuCandidates(program.view(), graphs));
  require(!candidates.empty(), "mining reported no common subgraph");
  const loom::dse::CompositeFuCandidate &best = candidates.front();

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

  // The tail operations differ, so no shape that contains one is common: the
  // graph-support prune is what keeps mining from reporting a whole graph.
  for (const loom::dse::CompositeFuCandidate &candidate : candidates) {
    for (const loom::dse::CompositeFuNode &node : candidate.nodes)
      require(node.schema != dataflow::OperationSchemaId::ArithShRSI &&
                  node.schema != dataflow::OperationSchemaId::ArithMaxSI,
              "mining reported a candidate that only one graph contains");
    require(candidate.graphCount >= 2,
            "mining reported a candidate below its graph-support bound");
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

  auto candidates =
      take(loom::dse::mineCompositeFuCandidates(program.view(), graphs));
  require(!candidates.empty(), "mining reported no common subgraph");
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
  std::set<std::uint64_t> mined;
  for (const loom::dse::CompositeFuOccurrence &occurrence : best.occurrences)
    for (dataflow::ActorRef actor : occurrence.actors)
      mined.insert(actor.entity.value());
  require(covered == mined,
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
  auto candidates =
      take(loom::dse::mineCompositeFuCandidates(program.view(), graphs));
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
  else
    fail("unknown scene " + scene.str());
  return EXIT_SUCCESS;
}
