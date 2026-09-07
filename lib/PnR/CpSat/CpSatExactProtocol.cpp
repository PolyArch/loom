#include "CpSatExactProtocol.h"

#include "Common/MappingDebugLog.h"

#include "ortools/sat/cp_model_checker.h"
#include "ortools/sat/cp_model_solver.h"
#include "ortools/sat/sat_parameters.pb.h"

#include "llvm/Support/SHA256.h"

#include <array>

#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
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

llvm::Error retainFeasibleSolutionHint(CpModelProto &model,
                                      const CpSolverResponse &response) {
  if (response.solution_size() != model.variables_size())
    return protocolError("feasible response omitted model variables");
  auto *hint = model.mutable_solution_hint();
  hint->Clear();
  for (int variable = 0; variable < model.variables_size(); ++variable) {
    hint->add_vars(variable);
    hint->add_values(response.solution(variable));
  }
  return llvm::Error::success();
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

void installProofDecisionStrategy(
    CpModelProto &model, llvm::ArrayRef<CpSatCanonicalVariable> variables,
    std::optional<int> objectiveVariable,
    llvm::ArrayRef<int> proofPriorityVariables) {
  model.clear_search_strategy();
  if (variables.empty())
    return;
  DecisionStrategyProto *strategy = model.add_search_strategy();
  if (objectiveVariable) {
    auto *expression = strategy->add_exprs();
    expression->add_vars(*objectiveVariable);
    expression->add_coeffs(1);
  }
  for (int variable : proofPriorityVariables) {
    if (objectiveVariable && variable == *objectiveVariable)
      continue;
    auto *expression = strategy->add_exprs();
    expression->add_vars(variable);
    expression->add_coeffs(1);
  }
  for (const CpSatCanonicalVariable &variable : variables) {
    if (objectiveVariable && variable.protoIndex == *objectiveVariable)
      continue;
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
  // exhausted solve returns UNKNOWN or FEASIBLE without the required proof.
  // Both remain incomplete; neither means the solver-call limit was reached.
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
/// memoization, not an approximation. Incomplete outcomes are never cached.
/// Worker threads keep independent memos; identical keys produce identical
/// results on every thread.
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
                  std::int32_t randomSeed,
                  llvm::ArrayRef<int> proofPriorityVariables) {
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
  updateWord(proofPriorityVariables.size());
  for (int variable : proofPriorityVariables)
    updateWord(static_cast<std::uint64_t>(variable));
  return hash.final();
}

struct SolveState final {
  std::uint64_t maxCalls;
  std::uint64_t calls = 0;
  SatParameters parameters;
  CpSolverStatus lastStatus = CpSolverStatus::UNKNOWN;
  double lastDeterministicTime = 0;
  double lastWallSeconds = 0;
  double lastUserSeconds = 0;
  std::int64_t lastConflicts = 0;
  std::int64_t lastBranches = 0;
  std::int64_t lastBinaryPropagations = 0;
  std::int64_t lastIntegerPropagations = 0;
  std::string lastSolutionInfo{};
  double deterministicTime = 0;
  double wallSeconds = 0;
  double userSeconds = 0;
};

std::optional<CpSolverResponse> solve(const CpModelProto &model,
                                      SolveState &state) {
  if (state.calls == state.maxCalls)
    return std::nullopt;
  ++state.calls;
  CpSolverResponse response = SolveWithParameters(model, state.parameters);
  state.lastStatus = response.status();
  state.lastDeterministicTime = response.deterministic_time();
  state.lastWallSeconds = response.wall_time();
  state.lastUserSeconds = response.user_time();
  state.lastConflicts = response.num_conflicts();
  state.lastBranches = response.num_branches();
  state.lastBinaryPropagations = response.num_binary_propagations();
  state.lastIntegerPropagations = response.num_integer_propagations();
  state.lastSolutionInfo = response.solution_info();
  state.deterministicTime += state.lastDeterministicTime;
  state.wallSeconds += state.lastWallSeconds;
  state.userSeconds += state.lastUserSeconds;
  return response;
}

enum class ProofPhase : std::uint8_t { Initial, CanonicalBlock };

llvm::StringRef spelling(ProofPhase phase) {
  switch (phase) {
  case ProofPhase::Initial:
    return "initial_proof";
  case ProofPhase::CanonicalBlock:
    return "canonical_block";
  }
  llvm_unreachable("unknown CP-SAT proof phase");
}

CpSatCanonicalResult incomplete(CpSatCanonicalResultKind kind,
                                const SolveState &state, ProofPhase phase,
                                std::size_t blockBegin = 0,
                                std::size_t blockEnd = 0) {
  loom::mapping_debug::emit(
      loom::mapping_debug::Level::Summary,
      loom::mapping_debug::Stage::SpatialPnr,
      loom::mapping_debug::Event::MappingFailure,
      [&](llvm::json::Object &fields) {
        fields["failure_scope"] = "cp_sat_canonical";
        fields["termination"] = cpSatCanonicalResultKindSpelling(kind);
        fields["proof_phase"] = spelling(phase);
        fields["solver_seed"] = state.parameters.random_seed();
        fields["solver_calls"] = state.calls;
        // Incomplete protocol calls never hit the completed-solve memo.
        fields["logical_solver_calls"] = state.calls;
        fields["max_solver_calls"] = state.maxCalls;
        fields["max_deterministic_time_per_call"] =
            state.parameters.max_deterministic_time();
        fields["total_deterministic_time"] = state.deterministicTime;
        fields["total_solver_wall_seconds"] = state.wallSeconds;
        fields["total_solver_user_seconds"] = state.userSeconds;
        if (state.calls != 0) {
          fields["last_cp_sat_status"] = CpSolverStatus_Name(state.lastStatus);
          fields["last_deterministic_time"] = state.lastDeterministicTime;
          fields["last_solver_wall_seconds"] = state.lastWallSeconds;
          fields["last_solver_user_seconds"] = state.lastUserSeconds;
          fields["last_conflicts"] = state.lastConflicts;
          fields["last_branches"] = state.lastBranches;
          fields["last_binary_propagations"] = state.lastBinaryPropagations;
          fields["last_integer_propagations"] = state.lastIntegerPropagations;
          fields["last_solution_info"] = state.lastSolutionInfo;
        }
        if (phase == ProofPhase::CanonicalBlock) {
          fields["canonical_block_begin"] = blockBegin;
          fields["canonical_block_end"] = blockEnd;
        }
      });
  return {kind, {}, std::nullopt, state.calls, state.calls};
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
    return CpSatProofStatus::Feasible;
  case CpSolverStatus::UNKNOWN:
    return CpSatProofStatus::Unknown;
  case CpSolverStatus::MODEL_INVALID:
  case CpSolverStatus::CpSolverStatus_INT_MIN_SENTINEL_DO_NOT_USE_:
  case CpSolverStatus::CpSolverStatus_INT_MAX_SENTINEL_DO_NOT_USE_:
    return CpSatProofStatus::InternalError;
  }
  return CpSatProofStatus::InternalError;
}

llvm::StringRef loom::pnr::detail::cpSatCanonicalResultKindSpelling(
    CpSatCanonicalResultKind kind) {
  switch (kind) {
  case CpSatCanonicalResultKind::Assignment:
    return "assignment";
  case CpSatCanonicalResultKind::Infeasible:
    return "infeasible";
  case CpSatCanonicalResultKind::SolverCallLimitReached:
    return "solver_call_limit_reached";
  case CpSatCanonicalResultKind::SolverUnknown:
    return "solver_unknown";
  case CpSatCanonicalResultKind::FeasibleWithoutOptimalityProof:
    return "feasible_without_optimality_proof";
  case CpSatCanonicalResultKind::Interrupted:
    return "interrupted";
  }
  llvm_unreachable("unknown CP-SAT canonical result kind");
}

std::int32_t
loom::pnr::detail::projectCpSatRandomSeed(std::uint64_t streamWord) {
  return static_cast<std::int32_t>(streamWord & UINT64_C(0x7fffffff));
}

llvm::Expected<CpSatCanonicalResult> loom::pnr::detail::solveCanonicalCpSat(
    const CpModelProto &model, llvm::ArrayRef<CpSatCanonicalVariable> variables,
    std::optional<int> objectiveVariable, std::uint64_t maxSolverCalls,
    std::int32_t randomSeed, SpatialPnrWorkLedgerView workLedger,
    llvm::ArrayRef<int> proofPriorityVariables,
    ExecutionControlView executionControl) {
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
  llvm::BitVector observedPriorityVariables(model.variables_size());
  for (int variable : proofPriorityVariables) {
    if (variable < 0 || variable >= model.variables_size())
      return protocolError("proof-priority variable is out of range");
    if (observedPriorityVariables.test(variable))
      return protocolError("proof-priority variable is duplicated");
    observedPriorityVariables.set(variable);
  }

  const std::array<std::uint8_t, 32> memoKey =
      canonicalSolveKey(model, variables, objectiveVariable, randomSeed,
                        proofPriorityVariables);
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
  installProofDecisionStrategy(working, variables, objectiveVariable,
                               proofPriorityVariables);
  SolveState state{maxSolverCalls, 0, parameters(randomSeed)};
  if (state.calls == state.maxCalls)
    return incomplete(CpSatCanonicalResultKind::SolverCallLimitReached, state,
                      ProofPhase::Initial);
  if (executionControl.stopRequested())
    return incomplete(CpSatCanonicalResultKind::Interrupted, state,
                      ProofPhase::Initial);
  if (llvm::Error error =
          workLedger.plan(SpatialPnrWorkKind::ExactRepairSolverCall))
    return std::move(error);
  std::optional<CpSolverResponse> initial = solve(working, state);
  if (initial)
    if (llvm::Error error =
            workLedger.consume(SpatialPnrWorkKind::ExactRepairSolverCall))
      return std::move(error);
  if (!initial)
    return incomplete(CpSatCanonicalResultKind::SolverCallLimitReached, state,
                      ProofPhase::Initial);
  switch (classifyCpSatProofStatus(initial->status())) {
  case CpSatProofStatus::Infeasible: {
    const CpSatCanonicalResult result{
        CpSatCanonicalResultKind::Infeasible, {}, std::nullopt, state.calls,
        state.calls};
    canonicalSolveMemo.retain(memoKey, result);
    return result;
  }
  case CpSatProofStatus::Feasible:
    return incomplete(CpSatCanonicalResultKind::FeasibleWithoutOptimalityProof,
                      state, ProofPhase::Initial);
  case CpSatProofStatus::Unknown:
    return incomplete(CpSatCanonicalResultKind::SolverUnknown, state,
                      ProofPhase::Initial);
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

  // The proven objective leaves a feasible complete assignment for every
  // canonical subproblem. Preserve it across solver restarts so extraction
  // starts with an incumbent; each block still needs its own optimum proof.
  if (llvm::Error error = retainFeasibleSolutionHint(working, *initial))
    return std::move(error);

  installCanonicalDecisionStrategy(working, variables);

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

    CpModelProto trial;
    std::optional<CpSolverResponse> response;
    for (;;) {
      trial = working;
      minimizeCanonicalBlock(trial, variables.slice(begin, end - begin),
                             *coefficients);
      if (state.calls == state.maxCalls)
        return incomplete(
            CpSatCanonicalResultKind::SolverCallLimitReached, state,
            ProofPhase::CanonicalBlock, begin, end);
      if (executionControl.stopRequested())
        return incomplete(CpSatCanonicalResultKind::Interrupted, state,
                          ProofPhase::CanonicalBlock, begin, end);
      if (llvm::Error error =
              workLedger.plan(SpatialPnrWorkKind::ExactRepairSolverCall))
        return std::move(error);
      response = solve(trial, state);
      if (response)
        if (llvm::Error error = workLedger.consume(
                SpatialPnrWorkKind::ExactRepairSolverCall))
          return std::move(error);
      if (!response)
        return incomplete(
            CpSatCanonicalResultKind::SolverCallLimitReached, state,
            ProofPhase::CanonicalBlock, begin, end);
      const CpSatProofStatus status =
          classifyCpSatProofStatus(response->status());
      if (status == CpSatProofStatus::Optimal)
        break;
      if ((status == CpSatProofStatus::Feasible ||
           status == CpSatProofStatus::Unknown) &&
          end - begin > 1) {
        if (status == CpSatProofStatus::Feasible)
          if (llvm::Error error = retainFeasibleSolutionHint(working, *response))
            return std::move(error);
        const std::size_t splitEnd = begin + (end - begin) / 2;
        loom::mapping_debug::emit(
            loom::mapping_debug::Level::Summary,
            loom::mapping_debug::Stage::SpatialPnr,
            loom::mapping_debug::Event::Statistics,
            [&](llvm::json::Object &fields) {
              fields["operation"] = "cp_sat_canonical_block_split";
              fields["canonical_block_begin"] = begin;
              fields["canonical_block_end"] = end;
              fields["replacement_block_end"] = splitEnd;
              fields["cp_sat_status"] =
                  CpSolverStatus_Name(response->status());
              fields["solver_calls"] = state.calls;
            });
        end = splitEnd;
        coefficients = canonicalBlockCoefficients(
            working, variables.slice(begin, end - begin));
        assert(coefficients &&
               "a subset of an encodable canonical block must be encodable");
        continue;
      }
      switch (status) {
      case CpSatProofStatus::Infeasible:
        return protocolError(
            "proven model became infeasible during canonical extraction");
      case CpSatProofStatus::Feasible:
        return incomplete(
            CpSatCanonicalResultKind::FeasibleWithoutOptimalityProof, state,
            ProofPhase::CanonicalBlock, begin, end);
      case CpSatProofStatus::Unknown:
        return incomplete(CpSatCanonicalResultKind::SolverUnknown, state,
                          ProofPhase::CanonicalBlock, begin, end);
      case CpSatProofStatus::InternalError:
        return protocolError(
            "OR-Tools rejected a canonical minimization model: " +
            response->solution_info());
      case CpSatProofStatus::Optimal:
        llvm_unreachable("optimal canonical proof did not exit the retry loop");
      }
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
    if (llvm::Error error = retainFeasibleSolutionHint(working, *response))
      return std::move(error);
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
    std::int32_t randomSeed, SpatialPnrWorkLedgerView workLedger,
    llvm::ArrayRef<int> proofPriorityVariables,
    ExecutionControlView executionControl) {
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
                                    maxSolverCalls, randomSeed, workLedger,
                                    proofPriorityVariables, executionControl);
  if (!solved)
    return solved.takeError();
  if (solved->kind == CpSatCanonicalResultKind::Assignment)
    solved->assignment.assign(assignment.begin(), assignment.end());
  return solved;
}
