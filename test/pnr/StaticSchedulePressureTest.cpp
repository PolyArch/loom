#include "StaticSchedulePressure.h"

#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Dataflow/IR/OperationSchema.h"

#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <utility>

namespace {

[[noreturn]] void fail(const llvm::Twine &message) {
  llvm::errs() << "static schedule pressure test failed: " << message << '\n';
  std::exit(EXIT_FAILURE);
}

template <typename T> T take(llvm::Expected<T> value) {
  if (!value)
    fail(llvm::toString(value.takeError()));
  return std::move(*value);
}

void require(bool condition, const llvm::Twine &message) {
  if (!condition)
    fail(message);
}

} // namespace

int main() {
  mlir::DialectRegistry registry;
  registry.insert<dataflow::DataflowDialect>();
  mlir::MLIRContext context(registry, mlir::MLIRContext::Threading::DISABLED);
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  dataflow.graph private @feedback(%start: none, %phase: i1) -> ()
      attributes {input_segments = array<i32: 1, 0, 0>,
                  result_segments = array<i32: 0, 0, 0>} {
    %stable = dataflow.invariant %phase, %start : none
    %carried = dataflow.carry %phase, %start, %lanes#1 : none
    %lanes:2 = dataflow.demux %phase, %carried
        : (i1, none) -> (none, none)
    dataflow.graph.return values() streams() memories()
        complete(%lanes#0 : none)
  }
}
)mlir",
                                                        &context);
  if (!module)
    fail("cannot parse recurrence fixture");
  auto artifact = take(dataflow::finalizeCanonicalDataflow(*module));
  const auto view = take(artifact.view());
  const std::array<dataflow::GraphRef, 1> covers = {view.graphs().front().ref};
  const auto analysis =
      take(loom::pnr::detail::deriveStaticScheduleAnalysis(view, covers));
  require(analysis.graphCriticalLength(covers.front()) == 2,
          "Dataflow graph critical path projection changed");

  std::map<dataflow::OperationSchemaId,
           const loom::pnr::detail::StaticActorCriticality *>
      actors;
  for (const auto &actor : analysis.actors()) {
    const auto resolved = take(view.resolve(actor.actor));
    const auto schema = dataflow::requireOperationSchema(resolved.op);
    actors.emplace(schema, &actor);
  }
  require(actors.size() == 3, "fixture actor inventory changed");

  const auto *invariant =
      actors.at(dataflow::OperationSchemaId::DataflowInvariant);
  const auto *carry = actors.at(dataflow::OperationSchemaId::DataflowCarry);
  const auto *demux = actors.at(dataflow::OperationSchemaId::DataflowDemux);
  require(invariant->temporalStateCarrier,
          "invariant must be recognized as a Temporal state carrier");
  require(carry->temporalStateCarrier,
          "carry must be recognized as a Temporal state carrier");
  require(!demux->temporalStateCarrier,
          "ordinary data movement must not become a state carrier");
  require(invariant->graphCriticalLength == 0 &&
              invariant->recurrenceCriticalLength == 0,
          "off-path actor acquired criticality");
  require(carry->graphCriticalLength == 2 &&
              carry->recurrenceCriticalLength == 2,
          "carry criticality does not match the recurrence path");
  require(demux->graphCriticalLength == 2 &&
              demux->recurrenceCriticalLength == 2,
          "demux criticality does not match the recurrence path");

  std::uint64_t forwardWeight = 0;
  std::uint64_t feedbackWeight = 0;
  for (const auto &edge : analysis.edges()) {
    if (edge.producer.actor == carry->actor &&
        edge.consumer.actor == demux->actor)
      forwardWeight = edge.weight;
    if (edge.producer.actor == demux->actor &&
        edge.consumer.actor == carry->actor)
      feedbackWeight = edge.weight;
  }
  require(forwardWeight == 4,
          "forward recurrence edge must carry graph and recurrence pressure");
  require(feedbackWeight == 2,
          "feedback edge must carry recurrence pressure exactly once");

  require(analysis.feedbacks().size() == 1,
          "carry Next did not produce one canonical feedback edge");
  const auto &feedback = analysis.feedbacks().front();
  require(feedback.producer.actor == demux->actor &&
              feedback.consumer.actor == carry->actor &&
              feedback.consumer.ordinal ==
                  static_cast<std::uint64_t>(
                      dataflow::semantics::CarryInput::Next) &&
              feedback.dependenceDistance == 1,
          "canonical recurrence feedback witness changed");
  require(analysis.recurrenceTopologies().size() == 1 &&
              analysis.recurrenceTopologies().front().postInitializationAcyclic,
          "carry feedback removal did not expose the canonical DAG");

  llvm::outs() << "static schedule pressure test passed\n";
  return EXIT_SUCCESS;
}
