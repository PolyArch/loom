#ifndef LOOM_LIB_PNR_CPSATEXACTPROTOCOL_H
#define LOOM_LIB_PNR_CPSATEXACTPROTOCOL_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include "Common/ExecutionControl.h"
#include "PnR/SpatialPnrWorkLedger.h"

#include "ortools/sat/cp_model.pb.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace loom::pnr::detail {

enum class CpSatProofStatus : std::uint8_t {
  Optimal,
  Infeasible,
  Feasible,
  Unknown,
  InternalError,
};

CpSatProofStatus
classifyCpSatProofStatus(operations_research::sat::CpSolverStatus status);

/// Projects the one ExactRepair stream word owned by a repair invocation to
/// OR-Tools' nonnegative signed seed domain.
std::int32_t projectCpSatRandomSeed(std::uint64_t streamWord);

struct CpSatCanonicalVariable final {
  int protoIndex;
  llvm::ArrayRef<std::int64_t> legalValues;
};

enum class CpSatCanonicalResultKind : std::uint8_t {
  Assignment,
  Infeasible,
  SolverCallLimitReached,
  SolverUnknown,
  FeasibleWithoutOptimalityProof,
  /// The invocation's execution control requested a stop between two solver
  /// calls; the work consumed so far is reported and nothing was decided.
  Interrupted,
};

llvm::StringRef cpSatCanonicalResultKindSpelling(CpSatCanonicalResultKind kind);

struct CpSatCanonicalResult final {
  CpSatCanonicalResultKind kind;
  std::vector<std::int64_t> assignment;
  std::optional<std::int64_t> objectiveValue;
  /// Actual solver invocations paid by this call. Exact memo hits report zero.
  std::uint64_t solverCalls;
  /// Deterministic solver work of the canonical result, including work a memo
  /// hit avoided. Search control and branch budgets consume this value so
  /// cache history cannot change the explored assignment prefix.
  std::uint64_t logicalSolverCalls;
};

/// Solves one exact CP-SAT model and extracts the lexicographically first
/// assignment in the supplied typed variable/value order. The objective, when
/// present, must be the single integer variable named by objectiveVariable.
/// Canonical extraction packs consecutive variables into exact int64
/// mixed-radix objectives and consumes one solve per encodable block.
/// `executionControl` is observed before every solver call, so a stop request
/// interrupts a long canonical extraction between two bounded solves.
llvm::Expected<CpSatCanonicalResult>
solveCanonicalCpSat(const operations_research::sat::CpModelProto &model,
                    llvm::ArrayRef<CpSatCanonicalVariable> variables,
                    std::optional<int> objectiveVariable,
                    std::uint64_t maxSolverCalls, std::int32_t randomSeed,
                    SpatialPnrWorkLedgerView workLedger = {},
                    llvm::ArrayRef<int> proofPriorityVariables = {},
                    ExecutionControlView executionControl = {});

/// Proves one complete supplied assignment with a single solver call. The
/// assignment uses the same typed variable/value order as canonical
/// extraction and is returned only when the fixed model is proof-bearing.
llvm::Expected<CpSatCanonicalResult>
solveFixedCpSatAssignment(const operations_research::sat::CpModelProto &model,
                          llvm::ArrayRef<CpSatCanonicalVariable> variables,
                          llvm::ArrayRef<std::int64_t> assignment,
                          std::optional<int> objectiveVariable,
                          std::uint64_t maxSolverCalls, std::int32_t randomSeed,
                          SpatialPnrWorkLedgerView workLedger = {},
                          llvm::ArrayRef<int> proofPriorityVariables = {},
                          ExecutionControlView executionControl = {});

} // namespace loom::pnr::detail

#endif // LOOM_LIB_PNR_CPSATEXACTPROTOCOL_H
