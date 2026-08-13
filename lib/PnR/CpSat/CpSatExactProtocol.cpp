#include "CpSatExactProtocol.h"

#include "ortools/sat/cp_model_solver.h"
#include "ortools/sat/sat_parameters.pb.h"

#include "llvm/ADT/BitVector.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <system_error>
#include <utility>

using namespace loom::pnr::detail;
using namespace operations_research::sat;

namespace {

llvm::Error protocolError(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "invalid CpSat_2_0 request: %s", message.str().c_str());
}

bool contains(const IntegerVariableProto &variable, std::int64_t value) {
  for (int index = 0; index < variable.domain_size(); index += 2)
    if (value >= variable.domain(index) && value <= variable.domain(index + 1))
      return true;
  return false;
}

llvm::Error validateVariable(const CpModelProto &model,
                             const CpSatCanonicalVariable &variable) {
  if (variable.protoIndex < 0 || variable.protoIndex >= model.variables_size())
    return protocolError("canonical variable index is out of range");
  const IntegerVariableProto &proto = model.variables(variable.protoIndex);
  if (proto.domain_size() == 0 || proto.domain_size() % 2 != 0)
    return protocolError("canonical variable has a malformed domain");
  if (variable.legalValues.empty())
    return protocolError("canonical variable has an empty legal-value set");

  std::uint64_t domainCardinality = 0;
  for (int index = 0; index < proto.domain_size(); index += 2) {
    const std::int64_t lower = proto.domain(index);
    const std::int64_t upper = proto.domain(index + 1);
    if (lower > upper)
      return protocolError("canonical variable domain is not ordered");
    const __int128 wideWidth =
        static_cast<__int128>(upper) - static_cast<__int128>(lower) + 1;
    if (wideWidth > std::numeric_limits<std::uint64_t>::max())
      return protocolError("canonical variable domain cardinality overflows");
    const auto width = static_cast<std::uint64_t>(wideWidth);
    if (width > std::numeric_limits<std::uint64_t>::max() - domainCardinality)
      return protocolError("canonical variable domain cardinality overflows");
    domainCardinality += width;
  }
  if (domainCardinality != variable.legalValues.size())
    return protocolError(
        "canonical legal values do not cover the complete variable domain");
  for (std::size_t index = 0; index < variable.legalValues.size(); ++index) {
    if (index != 0 &&
        variable.legalValues[index - 1] >= variable.legalValues[index])
      return protocolError("canonical legal values are not strictly ordered");
    if (!contains(proto, variable.legalValues[index]))
      return protocolError("canonical legal value is outside the domain");
  }
  return llvm::Error::success();
}

void fixVariable(CpModelProto &model, int variable, std::int64_t value) {
  LinearConstraintProto *constraint = model.add_constraints()->mutable_linear();
  constraint->add_vars(variable);
  constraint->add_coeffs(1);
  constraint->add_domain(value);
  constraint->add_domain(value);
}

void minimizeVariable(CpModelProto &model, int variable) {
  CpObjectiveProto *objective = model.mutable_objective();
  objective->Clear();
  objective->add_vars(variable);
  objective->add_coeffs(1);
}

SatParameters parameters(std::int32_t randomSeed) {
  SatParameters result;
  result.set_num_workers(1);
  result.set_random_seed(randomSeed);
  result.set_search_branching(SatParameters::FIXED_SEARCH);
  result.set_randomize_search(false);
  result.set_cp_model_presolve(true);
  result.set_enumerate_all_solutions(false);
  result.set_use_lns(false);
  result.set_use_lns_only(false);
  result.set_log_search_progress(false);
  result.set_log_to_stdout(false);
  return result;
}

struct SolveState final {
  std::uint64_t maxCalls;
  std::uint64_t calls = 0;
  SatParameters parameters;
};

std::optional<CpSolverResponse> solve(const CpModelProto &model,
                                      SolveState &state) {
  if (state.calls == state.maxCalls)
    return std::nullopt;
  ++state.calls;
  return SolveWithParameters(model, state.parameters);
}

CpSatCanonicalResult unknown(std::uint64_t calls) {
  return {CpSatCanonicalResultKind::UnknownBudgetExhausted,
          {},
          std::nullopt,
          calls};
}

} // namespace

CpSatProofStatus
loom::pnr::detail::classifyCpSatProofStatus(CpSolverStatus status) {
  switch (status) {
  case CpSolverStatus::OPTIMAL:
    return CpSatProofStatus::Optimal;
  case CpSolverStatus::INFEASIBLE:
    return CpSatProofStatus::Infeasible;
  case CpSolverStatus::FEASIBLE:
  case CpSolverStatus::UNKNOWN:
    return CpSatProofStatus::Unknown;
  case CpSolverStatus::MODEL_INVALID:
  case CpSolverStatus::CpSolverStatus_INT_MIN_SENTINEL_DO_NOT_USE_:
  case CpSolverStatus::CpSolverStatus_INT_MAX_SENTINEL_DO_NOT_USE_:
    return CpSatProofStatus::InternalError;
  }
  return CpSatProofStatus::InternalError;
}

std::int32_t
loom::pnr::detail::projectCpSatRandomSeed(std::uint64_t streamWord) {
  return static_cast<std::int32_t>(streamWord & UINT64_C(0x7fffffff));
}

llvm::Expected<CpSatCanonicalResult> loom::pnr::detail::solveCanonicalCpSat(
    const CpModelProto &model, llvm::ArrayRef<CpSatCanonicalVariable> variables,
    std::optional<int> objectiveVariable, std::uint64_t maxSolverCalls,
    std::int32_t randomSeed) {
  if (maxSolverCalls == 0)
    return protocolError("solver-call budget must be positive");
  llvm::BitVector observedVariables(model.variables_size());
  for (const CpSatCanonicalVariable &variable : variables) {
    if (llvm::Error error = validateVariable(model, variable))
      return std::move(error);
    if (observedVariables.test(variable.protoIndex))
      return protocolError("canonical variable is duplicated");
    observedVariables.set(variable.protoIndex);
  }
  if (model.has_objective() != objectiveVariable.has_value())
    return protocolError("objective-variable presence disagrees with model");
  if (objectiveVariable) {
    if (*objectiveVariable < 0 || *objectiveVariable >= model.variables_size())
      return protocolError("objective variable is out of range");
    const CpObjectiveProto &objective = model.objective();
    if (objective.vars_size() != 1 || objective.coeffs_size() != 1 ||
        objective.vars(0) != *objectiveVariable || objective.coeffs(0) != 1)
      return protocolError(
          "objective must minimize one exact integer objective variable");
  }

  SolveState state{maxSolverCalls, 0, parameters(randomSeed)};
  CpModelProto working = model;
  std::optional<CpSolverResponse> initial = solve(working, state);
  if (!initial)
    return unknown(state.calls);
  switch (classifyCpSatProofStatus(initial->status())) {
  case CpSatProofStatus::Infeasible:
    return CpSatCanonicalResult{
        CpSatCanonicalResultKind::Infeasible, {}, std::nullopt, state.calls};
  case CpSatProofStatus::Unknown:
    return unknown(state.calls);
  case CpSatProofStatus::InternalError:
    return protocolError("OR-Tools rejected the exact repair model");
  case CpSatProofStatus::Optimal:
    break;
  }

  std::optional<std::int64_t> objectiveValue;
  if (objectiveVariable) {
    if (*objectiveVariable >= initial->solution_size())
      return protocolError("optimal response omitted the objective variable");
    objectiveValue = initial->solution(*objectiveVariable);
    fixVariable(working, *objectiveVariable, *objectiveValue);
  }

  std::vector<std::int64_t> assignment;
  assignment.reserve(variables.size());
  for (const CpSatCanonicalVariable &variable : variables) {
    CpModelProto trial = working;
    minimizeVariable(trial, variable.protoIndex);
    std::optional<CpSolverResponse> response = solve(trial, state);
    if (!response)
      return unknown(state.calls);
    switch (classifyCpSatProofStatus(response->status())) {
    case CpSatProofStatus::Optimal:
      break;
    case CpSatProofStatus::Infeasible:
      return protocolError(
          "proven model became infeasible during canonical extraction");
    case CpSatProofStatus::Unknown:
      return unknown(state.calls);
    case CpSatProofStatus::InternalError:
      return protocolError("OR-Tools rejected a canonical minimization model");
    }
    if (variable.protoIndex >= response->solution_size())
      return protocolError("optimal response omitted a canonical variable");
    const std::int64_t value = response->solution(variable.protoIndex);
    if (!std::binary_search(variable.legalValues.begin(),
                            variable.legalValues.end(), value))
      return protocolError("optimal response selected an illegal value");
    working = std::move(trial);
    fixVariable(working, variable.protoIndex, value);
    assignment.push_back(value);
  }
  return CpSatCanonicalResult{CpSatCanonicalResultKind::Assignment,
                              std::move(assignment), objectiveValue,
                              state.calls};
}
