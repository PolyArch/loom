#include "InitializerRelationSolver.h"

#include "llvm/Support/Error.h"

#include <cstdlib>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace {

using loom::pnr::PnrIndex;
using loom::pnr::detail::InitializerRelationInput;
using loom::pnr::detail::InitializerRelationKind;
using loom::pnr::detail::InitializerRelationMemberInput;
using loom::pnr::detail::InitializerRelationModel;
using loom::pnr::detail::InitializerRelationSolveFailure;
using loom::pnr::detail::InitializerRelationSolveFailureKind;
using loom::pnr::detail::InitializerRelationSolver;

[[noreturn]] void fail(const std::string &message) {
  std::cerr << "initializer relation solver test failed: " << message << '\n';
  std::exit(1);
}

template <typename T> T take(llvm::Expected<T> value) {
  if (!value)
    fail(llvm::toString(value.takeError()));
  return std::move(*value);
}

InitializerRelationModel
makeModel(std::vector<PnrIndex> choiceCounts,
          std::vector<InitializerRelationInput> relations) {
  auto model = InitializerRelationModel::create(std::move(choiceCounts),
                                                std::move(relations));
  if (!model)
    fail(llvm::toString(model.takeError()));
  return std::move(*model);
}

void equalityAndDisjointnessReachFixedPoint() {
  InitializerRelationModel model = makeModel(
      {2, 2, 2},
      {{InitializerRelationKind::Equal, {{0, {0, 1}}, {1, {1, 2}}}},
       {InitializerRelationKind::Disjoint, {{1, {1, 2}}, {2, {1, 2}}}}});
  InitializerRelationSolver solver(model);
  const auto result = take(solver.solveCanonical(/*assignmentLimit=*/16));
  const std::vector<PnrIndex> expected{1, 0, 1};
  if (result.choices != expected || result.assignmentAttempts != 0)
    fail("singleton propagation did not reach the canonical fixed point");
  if (llvm::Error error = model.verifyChoices(result.choices))
    fail("solved assignment failed exact relation verification: " +
         llvm::toString(std::move(error)));
  if (llvm::Error error = model.verifyChoices({0, 0, 1}))
    llvm::consumeError(std::move(error));
  else
    fail("exact relation verification accepted an invalid assignment");
}

void canonicalSearchBacktracksWithoutCopyingState() {
  InitializerRelationModel model =
      makeModel({2, 2, 2}, {{InitializerRelationKind::Disjoint,
                             {{0, {0, 1}}, {1, {0, 2}}, {2, {0, 2}}}}});
  InitializerRelationSolver solver(model);
  const std::size_t retainedBytes = solver.retainedStorageBytes();
  const auto result = take(solver.solveCanonical(/*assignmentLimit=*/16));
  const std::vector<PnrIndex> expected{1, 0, 1};
  if (result.choices != expected || result.assignmentAttempts != 3)
    fail("canonical DFS did not backtrack to the first complete assignment");
  if (solver.retainedStorageBytes() != retainedBytes)
    fail("warm canonical DFS expanded retained solver storage");
}

void workLimitDoesNotBecomeInfeasibility() {
  InitializerRelationModel model =
      makeModel({2, 2, 2}, {{InitializerRelationKind::Disjoint,
                             {{0, {0, 1}}, {1, {0, 2}}, {2, {0, 2}}}}});
  InitializerRelationSolver solver(model);
  auto result = solver.solveCanonical(/*assignmentLimit=*/1);
  if (result)
    fail("bounded initializer ignored its assignment limit");
  bool observedWorkLimit = false;
  llvm::handleAllErrors(
      result.takeError(), [&](const InitializerRelationSolveFailure &error) {
        observedWorkLimit =
            error.kind() == InitializerRelationSolveFailureKind::WorkLimit;
      });
  if (!observedWorkLimit)
    fail("bounded initializer classified work exhaustion as infeasibility");
}

void sparseDomainsReusePreparedStorage() {
  constexpr PnrIndex decisionCount = 4096;
  InitializerRelationModel model =
      makeModel(std::vector<PnrIndex>(decisionCount, 2), {});
  InitializerRelationSolver solver(model);
  const std::size_t retainedBytes = solver.retainedStorageBytes();
  const auto result =
      take(solver.solveCanonical(/*assignmentLimit=*/decisionCount));
  if (result.assignmentAttempts != decisionCount ||
      result.choices.size() != decisionCount)
    fail("sparse canonical solve did not consume linear assignment work");
  for (PnrIndex choice : result.choices)
    if (choice != 0)
      fail("sparse canonical solve changed canonical choice order");
  if (solver.retainedStorageBytes() != retainedBytes)
    fail("sparse warm solve expanded retained solver storage");
}

} // namespace

int main() {
  equalityAndDisjointnessReachFixedPoint();
  canonicalSearchBacktracksWithoutCopyingState();
  workLimitDoesNotBecomeInfeasibility();
  sparseDomainsReusePreparedStorage();
  return 0;
}
