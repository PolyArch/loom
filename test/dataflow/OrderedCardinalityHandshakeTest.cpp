#include "Dataflow/IR/DataflowActorSemantics.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <variant>

namespace {

[[noreturn]] void fail(llvm::StringRef message) {
  llvm::errs() << "ordered cardinality handshake test: " << message << '\n';
  std::exit(EXIT_FAILURE);
}

void require(bool condition, llvm::StringRef message) {
  if (!condition)
    fail(message);
}

template <typename T> T take(llvm::Expected<T> value) {
  if (!value)
    fail(llvm::toString(value.takeError()));
  return std::move(*value);
}

void requireOnce(const dataflow::semantics::ActorResultProductionGroup &group,
                 llvm::ArrayRef<std::uint32_t> activeResults,
                 llvm::StringRef message) {
  require(group.activeResults == activeResults &&
              std::holds_alternative<
                  dataflow::semantics::ActorResultProductionOnce>(group.repeat),
          message);
}

void parallelizeOwnsOrderedCloseGroups() {
  using namespace dataflow::semantics;
  const auto cases = take(projectActorHandshakeCases(
      dataflow::OperationSchemaId::DataflowParallelize, 2, 3));
  require(cases.size() == 4, "parallelize case domain changed");

  require(cases[0].productionGroups.empty() && cases[0].activeResults.empty(),
          "parallelize accumulate manufactured a result group");
  require(cases[1].productionGroups.size() == 1,
          "parallelize full case lost its result group");
  requireOnce(cases[1].productionGroups[0], {0, 1, 2},
              "parallelize full group changed");
  require(cases[2].productionGroups.size() == 1,
          "parallelize empty close lost its terminal group");
  requireOnce(cases[2].productionGroups[0], {2},
              "parallelize empty close group changed");
  require(cases[3].productionGroups.size() == 2,
          "parallelize partial close is not a two-group production");
  requireOnce(cases[3].productionGroups[0], {0, 1, 2},
              "parallelize partial payload group changed");
  requireOnce(cases[3].productionGroups[1], {2},
              "parallelize partial terminal group changed");
  require(cases[3].activeResults == llvm::ArrayRef<std::uint32_t>({0, 1, 2}),
          "parallelize legacy active-result union is not sorted unique");
}

void serializeOwnsMaskOrderedRepetition() {
  using namespace dataflow::semantics;
  const auto cases = take(projectActorHandshakeCases(
      dataflow::OperationSchemaId::DataflowSerialize, 3, 2));
  require(cases.size() == 2, "serialize case domain changed");
  require(cases[0].productionGroups.size() == 1,
          "serialize active case lost its repeated group");
  const ActorResultProductionGroup &active = cases[0].productionGroups[0];
  const auto *repeat =
      std::get_if<ActorResultProductionForEachDefinedOneLane>(&active.repeat);
  require(active.activeResults == llvm::ArrayRef<std::uint32_t>({0, 1}) &&
              repeat && repeat->maskInputOrdinal == 2,
          "serialize active group lost its mask-ordered repetition");
  require(cases[0].activeResults == llvm::ArrayRef<std::uint32_t>({0, 1}),
          "serialize legacy active-result union changed");
  require(cases[1].productionGroups.size() == 1,
          "serialize close lost its terminal group");
  requireOnce(cases[1].productionGroups[0], {1},
              "serialize close group changed");
}

void ordinaryActorRetainsOneAtomicTuple() {
  using namespace dataflow::semantics;
  const auto cases = take(
      projectActorHandshakeCases(dataflow::OperationSchemaId::ArithAddI, 2, 1));
  require(cases.size() == 1 && cases[0].productionGroups.size() == 1,
          "ordinary actor lost its single atomic production");
  requireOnce(cases[0].productionGroups[0], {0},
              "ordinary actor production tuple changed");
}

} // namespace

int main() {
  parallelizeOwnsOrderedCloseGroups();
  serializeOwnsMaskOrderedRepetition();
  ordinaryActorRetainsOneAtomicTuple();
  return EXIT_SUCCESS;
}
