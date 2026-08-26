#ifndef LOOM_TEST_SYSTEM_EXECUTIONMATRIXINVOCATION_H
#define LOOM_TEST_SYSTEM_EXECUTIONMATRIXINVOCATION_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <string>

namespace loom::system_test {

enum class ExecutionMatrixCell : std::uint8_t {
  SpatialDfg,
  SpatialCgra,
  SpatialRtl,
  SystemDfg,
  SystemCgra,
  SystemRtl,
  PairedSpatialCgra,
  PairedSystemCgra,
};

enum class ExecutionMatrixAttemptKind : std::uint8_t {
  Ordinary,
  Diagnostic,
};

struct ExecutionMatrixInvocation final {
  ExecutionMatrixCell cell;
  ExecutionMatrixAttemptKind attempt;
};

const char *executionMatrixCellName(ExecutionMatrixCell cell);
const char *executionMatrixAttemptName(ExecutionMatrixAttemptKind attempt);
std::string
executionMatrixInvocationName(const ExecutionMatrixInvocation &invocation);

llvm::Expected<ExecutionMatrixInvocation>
parseExecutionMatrixInvocation(llvm::StringRef spelling);

} // namespace loom::system_test

#endif // LOOM_TEST_SYSTEM_EXECUTIONMATRIXINVOCATION_H
