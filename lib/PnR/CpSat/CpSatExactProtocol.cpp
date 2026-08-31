#include "CpSatExactProtocol.h"

#include "ortools/sat/cp_model_checker.h"
#include "ortools/sat/cp_model_solver.h"
#include "ortools/sat/sat_parameters.pb.h"

#include "llvm/Support/SHA256.h"

#include <array>

#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/STLExtras.h"
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
      "invalid CpSat_3_0 request: %s", message.str().c_str());
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

llvm::Error
validateVariables(const CpModelProto &model,
                  llvm::ArrayRef<CpSatCanonicalVariable> variables) {
  llvm::BitVector observedVariables(model.variables_size());
  for (const CpSatCanonicalVariable &variable : variables) {
    if (llvm::Error error = validateVariable(model, variable))
      return error;
    if (observedVariables.test(variable.protoIndex))
      return protocolError("canonical variable is duplicated");
    observedVariables.set(variable.protoIndex);
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

std::optional<std::vector<std::int64_t>>
canonicalBlockCoefficients(const CpModelProto &model,
                           llvm::ArrayRef<CpSatCanonicalVariable> variables) {
  assert(!variables.empty());
  std::vector<std::int64_t> coefficients(variables.size(), 1);
  for (std::size_t index = variables.size() - 1; index != 0; --index) {
    const auto values = variables[index].legalValues;
    const __int128 radix = static_cast<__int128>(values.back()) -
                           static_cast<__int128>(values.front()) + 1;
    const __int128 coefficient =
        static_cast<__int128>(coefficients[index]) * radix;
    if (coefficient > std::numeric_limits<std::int64_t>::max())
      return std::nullopt;
    coefficients[index - 1] = static_cast<std::int64_t>(coefficient);
  }

  std::vector<int> protoIndices;
  protoIndices.reserve(variables.size());
  for (const CpSatCanonicalVariable &variable : variables)
    protoIndices.push_back(variable.protoIndex);
  if (PossibleIntegerOverflow(model, protoIndices, coefficients))
    return std::nullopt;
  return coefficients;
}

void minimizeCanonicalBlock(CpModelProto &model,
                            llvm::ArrayRef<CpSatCanonicalVariable> variables,
                            llvm::ArrayRef<std::int64_t> coefficients) {
  assert(variables.size() == coefficients.size());
  CpObjectiveProto *objective = model.mutable_objective();
  objective->Clear();
  for (auto [variable, coefficient] :
       llvm::zip_equal(variables, coefficients)) {
    objective->add_vars(variable.protoIndex);
    objective->add_coeffs(coefficient);
  }
}

void installCanonicalDecisionStrategy(
    CpModelProto &model, llvm::ArrayRef<CpSatCanonicalVariable> variables) {
  model.clear_search_strategy();
  if (variables.empty())
    return;
  DecisionStrategyProto *strategy = model.add_search_strategy();
  for (const CpSatCanonicalVariable &variable : variables) {
    auto *expression = strategy->add_exprs();
    expression->add_vars(variable.protoIndex);
    expression->add_coeffs(1);
  }
  strategy->set_variable_selection_strategy(
      DecisionStrategyProto::CHOOSE_FIRST);
  strategy->set_domain_reduction_strategy(
      DecisionStrategyProto::SELECT_MIN_VALUE);
}

SatParameters parameters(std::int32_t randomSeed) {
  SatParameters result;
  result.set_num_workers(1);
  result.set_random_seed(randomSeed);
  result.set_search_branching(SatParameters::FIXED_SEARCH);
  result.set_randomize_search(false);
  result.set_cp_model_presolve(true);
  // FIXED_SEARCH over the canonical decision strategy does not consume the
  // failed-literal information presolve probing computes, and probing
  // dominated repair solve time on temporal fabrics. Level zero keeps the
  // rest of presolve; the exact-protocol descriptors version this choice.
  result.set_cp_model_probing_level(0);
  // A convergence budget per solve. Deterministic time is an instruction-count
  // clock, so the same model and seed exhaust it identically on every host; an
  // exhausted solve returns Unknown and flows into the existing typed
  // incomplete outcome instead of consuming the whole invocation deadline.
  result.set_max_deterministic_time(2.0);
  result.set_enumerate_all_solutions(false);
  result.set_use_lns(false);
  result.set_use_lns_only(false);
  result.set_log_search_progress(false);
  result.set_log_to_stdout(false);
  return result;
}

/// Invocation-lifetime memo of completed canonical solves. The result is a
/// pure function of the serialized model, the canonical variable layout and
/// the random seed under one protocol version, so replaying a hit is exact
/// memoization, not an approximation. Budget-exhausted outcomes depend on the
/// caller's call budget and are never cached. Worker threads keep independent
/// memos; identical keys produce identical results on every thread.
struct CanonicalSolveMemo final {
  static constexpr std::size_t entryLimit = 128;
  struct Entry final {
    std::array<std::uint8_t, 32> key;
    CpSatCanonicalResult result;
  };
  std::vector<Entry> entries;

  const CpSatCanonicalResult *find(const std::array<std::uint8_t, 32> &key) {
    for (const Entry &entry : entries)
      if (entry.key == key)
        return &entry.result;
    return nullptr;
  }
  void retain(const std::array<std::uint8_t, 32> &key,
              const CpSatCanonicalResult &result) {
    if (entries.size() == entryLimit)
      entries.erase(entries.begin());
    entries.push_back({key, result});
  }
};

thread_local CanonicalSolveMemo canonicalSolveMemo;

std::array<std::uint8_t, 32>
canonicalSolveKey(const CpModelProto &model,
                  llvm::ArrayRef<CpSatCanonicalVariable> variables,
                  std::optional<int> objectiveVariable,
                  std::int32_t randomSeed) {
  llvm::SHA256 hash;
  const std::string modelBytes = model.SerializeAsString();
  hash.update(llvm::ArrayRef<std::uint8_t>(
      reinterpret_cast<const std::uint8_t *>(modelBytes.data()),
      modelBytes.size()));
  const auto updateWord = [&](std::uint64_t word) {
    std::array<std::uint8_t, 8> bytes{};
    for (std::size_t index = 0; index != bytes.size(); ++index)
      bytes[index] = static_cast<std::uint8_t>(word >> (index * 8));
    hash.update(bytes);
  };
  updateWord(static_cast<std::uint64_t>(
      static_cast<std::uint32_t>(randomSeed)));
  updateWord(objectiveVariable
                 ? static_cast<std::uint64_t>(*objectiveVariable) + 1
                 : 0);
  updateWord(variables.size());
  for (const CpSatCanonicalVariable &variable : variables) {
    updateWord(static_cast<std::uint64_t>(variable.protoIndex));
    updateWord(variable.legalValues.size());
    for (std::int64_t value : variable.legalValues)
      updateWord(static_cast<std::uint64_t>(value));
  }
  return hash.final();
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
          calls,
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
    std::int32_t randomSeed, SpatialPnrWorkLedgerView workLedger) {
  if (maxSolverCalls == 0)
    return protocolError("solver-call budget must be positive");
  if (const std::string validation = ValidateCpModel(model);
      !validation.empty())
    return protocolError("exact repair model is invalid: " + validation);
  if (llvm::Error error = validateVariables(model, variables))
    return std::move(error);
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

  const std::array<std::uint8_t, 32> memoKey =
      canonicalSolveKey(model, variables, objectiveVariable, randomSeed);
  if (const CpSatCanonicalResult *memo = canonicalSolveMemo.find(memoKey);
      memo && memo->logicalSolverCalls <= maxSolverCalls) {
    // The recorded completion fits the caller's call budget, so this budget
    // provably reaches the same result; a smaller budget must still run and
    // observe its own typed exhaustion.
    CpSatCanonicalResult replay = *memo;
    replay.solverCalls = 0;
    return replay;
  }
  CpModelProto working = model;
  installCanonicalDecisionStrategy(working, variables);
  SolveState state{maxSolverCalls, 0, parameters(randomSeed)};
  if (state.calls == state.maxCalls)
    return unknown(state.calls);
  if (llvm::Error error =
          workLedger.plan(SpatialPnrWorkKind::ExactRepairSolverCall))
    return std::move(error);
  std::optional<CpSolverResponse> initial = solve(working, state);
  if (initial)
    if (llvm::Error error =
            workLedger.consume(SpatialPnrWorkKind::ExactRepairSolverCall))
      return std::move(error);
  if (!initial)
    return unknown(state.calls);
  switch (classifyCpSatProofStatus(initial->status())) {
  case CpSatProofStatus::Infeasible: {
    const CpSatCanonicalResult result{
        CpSatCanonicalResultKind::Infeasible, {}, std::nullopt, state.calls,
        state.calls};
    canonicalSolveMemo.retain(memoKey, result);
    return result;
  }
  case CpSatProofStatus::Unknown:
    return unknown(state.calls);
  case CpSatProofStatus::InternalError:
    return protocolError("OR-Tools rejected the exact repair model: " +
                         initial->solution_info());
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
  for (std::size_t begin = 0; begin != variables.size();) {
    std::size_t end = begin + 1;
    auto coefficients =
        canonicalBlockCoefficients(working, variables.slice(begin, 1));
    assert(coefficients && "one canonical variable must be int64 encodable");
    while (end != variables.size()) {
      auto extended = canonicalBlockCoefficients(
          working, variables.slice(begin, end - begin + 1));
      if (!extended)
        break;
      coefficients = std::move(extended);
      ++end;
    }

    CpModelProto trial = working;
    minimizeCanonicalBlock(trial, variables.slice(begin, end - begin),
                           *coefficients);
    if (state.calls == state.maxCalls)
      return unknown(state.calls);
    if (llvm::Error error =
            workLedger.plan(SpatialPnrWorkKind::ExactRepairSolverCall))
      return std::move(error);
    std::optional<CpSolverResponse> response = solve(trial, state);
    if (response)
      if (llvm::Error error =
              workLedger.consume(SpatialPnrWorkKind::ExactRepairSolverCall))
        return std::move(error);
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
      return protocolError(
          "OR-Tools rejected a canonical minimization model: " +
          response->solution_info());
    }
    working = std::move(trial);
    for (const CpSatCanonicalVariable &variable :
         variables.slice(begin, end - begin)) {
      if (variable.protoIndex >= response->solution_size())
        return protocolError("optimal response omitted a canonical variable");
      const std::int64_t value = response->solution(variable.protoIndex);
      if (!std::binary_search(variable.legalValues.begin(),
                              variable.legalValues.end(), value))
        return protocolError("optimal response selected an illegal value");
      fixVariable(working, variable.protoIndex, value);
      assignment.push_back(value);
    }
    begin = end;
  }
  {
    const CpSatCanonicalResult result{
        CpSatCanonicalResultKind::Assignment, std::move(assignment),
        objectiveValue, state.calls, state.calls};
    canonicalSolveMemo.retain(memoKey, result);
    return result;
  }
}

llvm::Expected<CpSatCanonicalResult>
loom::pnr::detail::solveFixedCpSatAssignment(
    const CpModelProto &model, llvm::ArrayRef<CpSatCanonicalVariable> variables,
    llvm::ArrayRef<std::int64_t> assignment,
    std::optional<int> objectiveVariable, std::uint64_t maxSolverCalls,
    std::int32_t randomSeed, SpatialPnrWorkLedgerView workLedger) {
  if (variables.size() != assignment.size())
    return protocolError("fixed assignment variable and value counts disagree");
  if (llvm::Error error = validateVariables(model, variables))
    return std::move(error);

  CpModelProto fixed = model;
  for (std::size_t index = 0; index < variables.size(); ++index) {
    const CpSatCanonicalVariable &variable = variables[index];
    const std::int64_t value = assignment[index];
    if (!std::binary_search(variable.legalValues.begin(),
                            variable.legalValues.end(), value))
      return protocolError("fixed assignment selected an illegal value");
    fixVariable(fixed, variable.protoIndex, value);
  }
  auto solved = solveCanonicalCpSat(fixed, {}, objectiveVariable,
                                    maxSolverCalls, randomSeed, workLedger);
  if (!solved)
    return solved.takeError();
  if (solved->kind == CpSatCanonicalResultKind::Assignment)
    solved->assignment.assign(assignment.begin(), assignment.end());
  return solved;
}
