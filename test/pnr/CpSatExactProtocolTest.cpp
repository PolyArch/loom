#include "CpSatExactProtocol.h"

#include "ortools/sat/cp_model.h"
#include "ortools/sat/cp_model.pb.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <numeric>
#include <utility>
#include <vector>

namespace {

[[noreturn]] void fail(llvm::StringRef message) {
  llvm::errs() << "CP-SAT exact protocol test: " << message << '\n';
  std::exit(1);
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

void equivalentOptimaUseCanonicalAssignment() {
  using namespace loom::pnr::detail;
  using namespace operations_research;
  using namespace operations_research::sat;

  CpModelBuilder model;
  const IntVar x = model.NewIntVar(Domain(0, 1));
  const IntVar y = model.NewIntVar(Domain(0, 1));
  const IntVar objective = model.NewIntVar(Domain(0, 2));
  model.AddGreaterOrEqual(LinearExpr::Sum({x, y}), 1);
  model.AddEquality(objective, LinearExpr::Sum({x, y}));
  model.Minimize(objective);

  const std::array<std::int64_t, 2> binaryValues{0, 1};
  const std::array<CpSatCanonicalVariable, 2> variables{{
      {x.index(), binaryValues},
      {y.index(), binaryValues},
  }};
  const CpSatCanonicalResult result = take(solveCanonicalCpSat(
      model.Build(), variables, objective.index(), /*maxSolverCalls=*/8,
      /*randomSeed=*/17));
  require(result.kind == CpSatCanonicalResultKind::Assignment,
          "equivalent optimum did not produce an assignment");
  require(llvm::ArrayRef(result.assignment) ==
              llvm::ArrayRef<std::int64_t>({0, 1}),
          "equivalent optimum did not use lexicographically first values");
  require(result.objectiveValue && *result.objectiveValue == 1,
          "exact optimum value was not preserved");
  require(result.solverCalls == 3,
          "canonical extraction consumed the wrong solver-call count");
}

void wideDomainsConsumeOneCallPerVariable() {
  using namespace loom::pnr::detail;
  using namespace operations_research;
  using namespace operations_research::sat;

  CpModelBuilder model;
  const IntVar x = model.NewIntVar(Domain(0, 4095));
  const IntVar y = model.NewIntVar(Domain(0, 4095));
  model.AddGreaterOrEqual(x, 3072);
  model.AddGreaterOrEqual(y, x);

  std::vector<std::int64_t> values(4096);
  std::iota(values.begin(), values.end(), 0);
  const std::array<CpSatCanonicalVariable, 2> variables{{
      {x.index(), values},
      {y.index(), values},
  }};
  const CpSatCanonicalResult result = take(solveCanonicalCpSat(
      model.Build(), variables, std::nullopt, /*maxSolverCalls=*/3,
      /*randomSeed=*/29));
  require(result.kind == CpSatCanonicalResultKind::Assignment,
          "wide canonical domains did not produce an assignment");
  require(llvm::ArrayRef(result.assignment) ==
              llvm::ArrayRef<std::int64_t>({3072, 3072}),
          "wide canonical domains did not select the lexicographic minimum");
  require(result.solverCalls == 3,
          "wide domains changed the canonical solver-call count");
}

void solverCallBudgetLeavesNoPartialAssignment() {
  using namespace loom::pnr::detail;
  using namespace operations_research;
  using namespace operations_research::sat;

  CpModelBuilder model;
  const IntVar x = model.NewIntVar(Domain(0, 1));
  const IntVar y = model.NewIntVar(Domain(0, 1));
  model.AddNotEqual(x, y);

  const std::array<std::int64_t, 2> binaryValues{0, 1};
  const std::array<CpSatCanonicalVariable, 2> variables{{
      {x.index(), binaryValues},
      {y.index(), binaryValues},
  }};
  const CpSatCanonicalResult result = take(solveCanonicalCpSat(
      model.Build(), variables, std::nullopt, /*maxSolverCalls=*/2,
      /*randomSeed=*/19));
  require(result.kind == CpSatCanonicalResultKind::UnknownBudgetExhausted,
          "solver-call exhaustion was treated as a proof");
  require(result.assignment.empty(),
          "solver-call exhaustion exposed a partial assignment");
  require(result.solverCalls == 2,
          "solver-call budget was not consumed exactly");
}

void localInfeasibilityIsProofBearingButNotGlobal() {
  using namespace loom::pnr::detail;
  using namespace operations_research;
  using namespace operations_research::sat;

  CpModelBuilder model;
  const IntVar x = model.NewIntVar(Domain(0));
  model.AddEquality(x, 1);
  const std::array<std::int64_t, 1> values{0};
  const std::array<CpSatCanonicalVariable, 1> variables{{
      {x.index(), values},
  }};
  const CpSatCanonicalResult result = take(solveCanonicalCpSat(
      model.Build(), variables, std::nullopt, /*maxSolverCalls=*/4,
      /*randomSeed=*/23));
  require(result.kind == CpSatCanonicalResultKind::Infeasible,
          "infeasible model did not preserve its local proof status");
  require(result.assignment.empty() && result.solverCalls == 1,
          "infeasible model exposed an assignment or wrong call count");
}

void nonProofStatusesFailClosed() {
  using namespace loom::pnr::detail;
  using operations_research::sat::CpSolverStatus;

  require(classifyCpSatProofStatus(CpSolverStatus::OPTIMAL) ==
              CpSatProofStatus::Optimal,
          "OPTIMAL lost proof-bearing status");
  require(classifyCpSatProofStatus(CpSolverStatus::INFEASIBLE) ==
              CpSatProofStatus::Infeasible,
          "INFEASIBLE lost proof-bearing status");
  require(classifyCpSatProofStatus(CpSolverStatus::FEASIBLE) ==
              CpSatProofStatus::Unknown,
          "FEASIBLE was treated as proof-bearing");
  require(classifyCpSatProofStatus(CpSolverStatus::UNKNOWN) ==
              CpSatProofStatus::Unknown,
          "UNKNOWN was treated as proof-bearing");
  require(classifyCpSatProofStatus(CpSolverStatus::MODEL_INVALID) ==
              CpSatProofStatus::InternalError,
          "MODEL_INVALID was not classified as an adapter failure");
}

void seedProjectionIsUnsignedAndStable() {
  using loom::pnr::detail::projectCpSatRandomSeed;
  require(projectCpSatRandomSeed(UINT64_C(0x0123456789abcdef)) ==
              INT32_C(0x09abcdef),
          "CP-SAT seed did not use the low 31 bits");
  require(projectCpSatRandomSeed(UINT64_C(0xffffffffffffffff)) ==
              INT32_C(0x7fffffff),
          "CP-SAT seed projection produced a negative value");
  require(projectCpSatRandomSeed(UINT64_C(0x0000000080000000)) == 0,
          "CP-SAT seed projection retained the sign bit");
}

} // namespace

int main() {
  equivalentOptimaUseCanonicalAssignment();
  wideDomainsConsumeOneCallPerVariable();
  solverCallBudgetLeavesNoPartialAssignment();
  localInfeasibilityIsProofBearingButNotGlobal();
  nonProofStatusesFailClosed();
  seedProjectionIsUnsignedAndStable();
  llvm::outs() << "CP-SAT exact protocol tests passed\n";
  return 0;
}
