#include "InitializerRelationSolver.h"

#include "PnR/DeterministicSearchProtocol.h"

#include "llvm/Support/Error.h"

#include <cstdlib>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace {

using loom::pnr::DeterministicPnrRandomStream;
using loom::pnr::PnrIndex;
using loom::pnr::PnrRandomStreamPurpose;
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

void completeAssignmentValidationBacktracks() {
  InitializerRelationModel model = makeModel({2, 2}, {});
  InitializerRelationSolver solver(model);
  std::vector<std::vector<PnrIndex>> observed;
  const auto result = take(solver.solveCanonical(
      /*assignmentLimit=*/8,
      [&](llvm::ArrayRef<PnrIndex> choices) -> llvm::Expected<bool> {
        observed.emplace_back(choices.begin(), choices.end());
        return choices == llvm::ArrayRef<PnrIndex>({0, 1});
      }));
  if (result.choices != std::vector<PnrIndex>({0, 1}) ||
      observed != std::vector<std::vector<PnrIndex>>({{0, 0}, {0, 1}}) ||
      result.assignmentAttempts != 3)
    fail("complete-assignment rejection did not continue canonical DFS");

  InitializerRelationSolver limited(model);
  auto exhausted = limited.solveCanonical(
      /*assignmentLimit=*/2,
      [](llvm::ArrayRef<PnrIndex>) -> llvm::Expected<bool> { return false; });
  if (exhausted)
    fail("complete-assignment rejection exceeded its work limit");
  bool observedWorkLimit = false;
  llvm::handleAllErrors(
      exhausted.takeError(), [&](const InitializerRelationSolveFailure &error) {
        observedWorkLimit =
            error.kind() == InitializerRelationSolveFailureKind::WorkLimit;
      });
  if (!observedWorkLimit)
    fail("complete-assignment work exhaustion became infeasibility");
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

void allDifferentCardinalityProvesInfeasibilityBeforeSearch() {
  InitializerRelationModel model =
      makeModel({2, 2, 2}, {{InitializerRelationKind::Disjoint,
                             {{0, {0, 1}}, {1, {0, 1}}, {2, {0, 1}}}}});
  InitializerRelationSolver solver(model);

  auto canonical = solver.solveCanonical(/*assignmentLimit=*/1);
  if (canonical)
    fail("all-different cardinality contradiction produced an assignment");
  bool observedCanonicalProof = false;
  llvm::handleAllErrors(
      canonical.takeError(), [&](const InitializerRelationSolveFailure &error) {
        observedCanonicalProof =
            error.kind() ==
            InitializerRelationSolveFailureKind::ProvenInfeasible;
      });
  if (!observedCanonicalProof || solver.assignmentAttempts() != 0)
    fail("all-different cardinality proof entered canonical DFS");

  auto fixed = solver.solveCanonicalWithFixedChoices(
      /*assignmentLimit=*/1,
      {loom::pnr::getInvalidPnrIndex(), loom::pnr::getInvalidPnrIndex(),
       loom::pnr::getInvalidPnrIndex()});
  if (fixed)
    fail("fixed-root solve ignored global all-different infeasibility");
  bool observedFixedProof = false;
  llvm::handleAllErrors(
      fixed.takeError(), [&](const InitializerRelationSolveFailure &error) {
        observedFixedProof =
            error.kind() ==
            InitializerRelationSolveFailureKind::ProvenInfeasible;
      });
  if (!observedFixedProof || solver.assignmentAttempts() != 0)
    fail("all-different cardinality proof entered fixed-root DFS");
}

void repeatedChoicesDoNotCreateResidentContexts() {
  InitializerRelationModel model = makeModel(
      {3, 2},
      {{InitializerRelationKind::Disjoint, {{0, {0, 0, 1}}, {1, {0, 1}}}}});
  InitializerRelationSolver solver(model);
  const auto result = take(solver.solveCanonical(/*assignmentLimit=*/8));
  if (llvm::Error error = model.verifyChoices(result.choices))
    fail("repeated context choices produced an invalid assignment: " +
         llvm::toString(std::move(error)));
  const auto members = model.members(model.relations().front());
  if (model.projectedValue(members[0], result.choices[0]) ==
      model.projectedValue(members[1], result.choices[1]))
    fail("multiple FU choices manufactured another resident context");

  auto aliased = solver.solveCanonicalWithFixedChoices(
      /*assignmentLimit=*/8, {0, 0});
  if (aliased)
    fail("fixed choices placed two realizations in one resident context");
  bool observedFixedRootFailure = false;
  llvm::handleAllErrors(
      aliased.takeError(), [&](const InitializerRelationSolveFailure &error) {
        observedFixedRootFailure =
            error.kind() ==
            InitializerRelationSolveFailureKind::FixedRootInfeasible;
      });
  if (!observedFixedRootFailure)
    fail("resident context alias changed failure classification");
}

void fixedRootFailureIsNotGlobalInfeasibility() {
  llvm::Error failure = llvm::make_error<InitializerRelationSolveFailure>(
      InitializerRelationSolveFailureKind::FixedRootInfeasible,
      "fixed root has no dependent assignment");
  bool observedFixedRootFailure = false;
  llvm::handleAllErrors(
      std::move(failure), [&](const InitializerRelationSolveFailure &error) {
        observedFixedRootFailure =
            error.kind() ==
            InitializerRelationSolveFailureKind::FixedRootInfeasible;
      });
  if (!observedFixedRootFailure)
    fail("fixed-root rejection became a whole-domain infeasibility proof");
}

void fixedChoicesConstrainTheSharedRelationModel() {
  InitializerRelationModel model =
      makeModel({2, 2, 2},
                {{InitializerRelationKind::Equal, {{0, {0, 1}}, {1, {0, 1}}}},
                 {InitializerRelationKind::Equal, {{1, {0, 1}}, {2, {0, 1}}}}});
  InitializerRelationSolver solver(model);
  const std::size_t retainedBytes = solver.retainedStorageBytes();
  const std::vector<PnrIndex> fixed{1, loom::pnr::getInvalidPnrIndex(),
                                    loom::pnr::getInvalidPnrIndex()};
  const auto result = take(
      solver.solveCanonicalWithFixedChoices(/*assignmentLimit=*/16, fixed));
  if (result.choices != std::vector<PnrIndex>({1, 1, 1}) ||
      result.assignmentAttempts != 0)
    fail("fixed root did not propagate through the relation model");
  if (solver.retainedStorageBytes() != retainedBytes)
    fail("warm fixed-root solve expanded retained solver storage");

  const std::vector<PnrIndex> contradictory{1, 0,
                                            loom::pnr::getInvalidPnrIndex()};
  auto rejected = solver.solveCanonicalWithFixedChoices(
      /*assignmentLimit=*/16, contradictory);
  if (rejected)
    fail("contradictory fixed choices produced an assignment");
  bool observedFixedRootFailure = false;
  llvm::handleAllErrors(
      rejected.takeError(), [&](const InitializerRelationSolveFailure &error) {
        observedFixedRootFailure =
            error.kind() ==
            InitializerRelationSolveFailureKind::FixedRootInfeasible;
      });
  if (!observedFixedRootFailure)
    fail("contradictory fixed choices became global infeasibility");
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

void binaryEqualityIncidenceScalesWithAllDifferentDomains() {
  constexpr PnrIndex decisionCount = 256;
  std::vector<PnrIndex> choiceCounts(decisionCount * 2, decisionCount);
  std::vector<PnrIndex> projectedValues(decisionCount);
  for (PnrIndex value = 0; value < decisionCount; ++value)
    projectedValues[value] = value;

  std::vector<InitializerRelationInput> relations;
  InitializerRelationInput residentContexts;
  residentContexts.kind = InitializerRelationKind::Disjoint;
  for (PnrIndex decision = 0; decision < decisionCount; ++decision)
    residentContexts.members.push_back({decision, projectedValues});
  relations.push_back(std::move(residentContexts));
  for (PnrIndex decision = 0; decision < decisionCount; ++decision)
    relations.push_back({InitializerRelationKind::Equal,
                         {{decision, projectedValues},
                          {decisionCount + decision, projectedValues}}});

  InitializerRelationModel model =
      makeModel(std::move(choiceCounts), std::move(relations));
  InitializerRelationSolver solver(model);
  const std::size_t retainedBytes = solver.retainedStorageBytes();
  const auto result =
      take(solver.solveCanonical(/*assignmentLimit=*/decisionCount));
  if (result.assignmentAttempts != decisionCount - 1)
    fail("all-different equality solve consumed noncanonical search work");
  for (PnrIndex decision = 0; decision < decisionCount; ++decision)
    if (result.choices[decision] != decision ||
        result.choices[decisionCount + decision] != decision)
      fail("binary equality incidence lost its resident-context value");

  const auto replay =
      take(solver.solveCanonical(/*assignmentLimit=*/decisionCount));
  if (replay.choices != result.choices)
    fail("warm binary equality incidence solve changed its assignment");
  if (replay.assignmentAttempts != result.assignmentAttempts)
    fail("warm binary equality incidence solve changed its accounting");
  if (solver.retainedStorageBytes() != retainedBytes)
    fail("warm binary equality incidence solve expanded retained storage");
}

void diversifiedSearchUsesExactWithoutReplacementOrder() {
  constexpr PnrIndex decisionCount = 4;
  constexpr PnrIndex choiceCount = 5;
  InitializerRelationModel model =
      makeModel(std::vector<PnrIndex>(decisionCount, choiceCount), {});
  InitializerRelationSolver solver(model);

  auto stream = DeterministicPnrRandomStream::create(
      /*masterSeed=*/17, /*seedIndex=*/3,
      PnrRandomStreamPurpose::InitializerDiversification);
  auto expectedStream = DeterministicPnrRandomStream::create(
      /*masterSeed=*/17, /*seedIndex=*/3,
      PnrRandomStreamPurpose::InitializerDiversification);
  std::vector<PnrIndex> expected;
  expected.reserve(decisionCount);
  for (PnrIndex decision = 0; decision < decisionCount; ++decision) {
    std::vector<PnrIndex> remaining{0, 1, 2, 3, 4};
    for (PnrIndex position = 0; position < choiceCount; ++position) {
      const std::uint64_t selected = take(expectedStream.nextBounded(
          static_cast<std::uint64_t>(choiceCount - position)));
      const PnrIndex choice = remaining[selected];
      if (position == 0)
        expected.push_back(choice);
      remaining.erase(remaining.begin() + selected);
    }
  }

  const std::size_t retainedBytes = solver.retainedStorageBytes();
  const auto result = take(solver.solveDiversified(
      /*assignmentLimit=*/decisionCount, stream));
  if (result.choices != expected || result.assignmentAttempts != decisionCount)
    fail("diversified DFS did not use the exact without-replacement order");
  if (solver.retainedStorageBytes() != retainedBytes)
    fail("warm diversified DFS expanded retained solver storage");

  auto replayStream = DeterministicPnrRandomStream::create(
      /*masterSeed=*/17, /*seedIndex=*/3,
      PnrRandomStreamPurpose::InitializerDiversification);
  const auto replay = take(solver.solveDiversified(
      /*assignmentLimit=*/decisionCount, replayStream));
  if (replay.choices != result.choices ||
      replay.assignmentAttempts != result.assignmentAttempts)
    fail("diversified initializer replay changed its assignment");
}

void independentRootDecisionsParticipateInMrvOrdering() {
  InitializerRelationModel bindingModel = makeModel({3}, {});
  const std::vector<PnrIndex> independentChoiceCounts = {2};
  InitializerRelationSolver solver(bindingModel, independentChoiceCounts);
  auto stream = DeterministicPnrRandomStream::create(
      /*masterSeed=*/91, /*seedIndex=*/2,
      PnrRandomStreamPurpose::InitializerDiversification);
  auto expectedStream = DeterministicPnrRandomStream::create(
      /*masterSeed=*/91, /*seedIndex=*/2,
      PnrRandomStreamPurpose::InitializerDiversification);

  const PnrIndex independentFirst =
      static_cast<PnrIndex>(take(expectedStream.nextBounded(2)));
  take(expectedStream.nextBounded(1));
  const PnrIndex bindingFirst =
      static_cast<PnrIndex>(take(expectedStream.nextBounded(3)));
  take(expectedStream.nextBounded(2));
  take(expectedStream.nextBounded(1));

  const auto result = take(solver.solveDiversified(
      /*assignmentLimit=*/2, stream));
  if (result.choices !=
          std::vector<PnrIndex>({bindingFirst, independentFirst}) ||
      result.assignmentAttempts != 2)
    fail("independent root decision did not participate in exact MRV order");
}

} // namespace

int main() {
  equalityAndDisjointnessReachFixedPoint();
  canonicalSearchBacktracksWithoutCopyingState();
  completeAssignmentValidationBacktracks();
  workLimitDoesNotBecomeInfeasibility();
  allDifferentCardinalityProvesInfeasibilityBeforeSearch();
  repeatedChoicesDoNotCreateResidentContexts();
  fixedRootFailureIsNotGlobalInfeasibility();
  fixedChoicesConstrainTheSharedRelationModel();
  sparseDomainsReusePreparedStorage();
  binaryEqualityIncidenceScalesWithAllDifferentDomains();
  diversifiedSearchUsesExactWithoutReplacementOrder();
  independentRootDecisionsParticipateInMrvOrdering();
  return 0;
}
