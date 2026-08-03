#ifndef LOOM_LIB_PNR_CPSATEXACTPROTOCOL_H
#define LOOM_LIB_PNR_CPSATEXACTPROTOCOL_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include "ortools/sat/cp_model.pb.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace loom::pnr::detail {

enum class CpSatProofStatus : std::uint8_t {
  Optimal,
  Infeasible,
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
  UnknownBudgetExhausted,
};

struct CpSatCanonicalResult final {
  CpSatCanonicalResultKind kind;
  std::vector<std::int64_t> assignment;
  std::optional<std::int64_t> objectiveValue;
  std::uint64_t solverCalls;
};

/// Solves one exact CP-SAT model and extracts the lexicographically first
/// assignment in the supplied typed variable/value order. The objective, when
/// present, must be the single integer variable named by objectiveVariable.
llvm::Expected<CpSatCanonicalResult>
solveCanonicalCpSat(const operations_research::sat::CpModelProto &model,
                    llvm::ArrayRef<CpSatCanonicalVariable> variables,
                    std::optional<int> objectiveVariable,
                    std::uint64_t maxSolverCalls, std::int32_t randomSeed);

} // namespace loom::pnr::detail

#endif // LOOM_LIB_PNR_CPSATEXACTPROTOCOL_H
