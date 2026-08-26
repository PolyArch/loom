#include "ExecutionMatrixInvocation.h"

#include "llvm/Support/ErrorHandling.h"

namespace loom::system_test {

const char *executionMatrixCellName(ExecutionMatrixCell cell) {
  switch (cell) {
  case ExecutionMatrixCell::SpatialDfg:
    return "spatial-dfg";
  case ExecutionMatrixCell::SpatialCgra:
    return "spatial-cgra";
  case ExecutionMatrixCell::SpatialRtl:
    return "spatial-rtl";
  case ExecutionMatrixCell::SystemDfg:
    return "system-dfg";
  case ExecutionMatrixCell::SystemCgra:
    return "system-cgra";
  case ExecutionMatrixCell::SystemRtl:
    return "system-rtl";
  case ExecutionMatrixCell::PairedSpatialCgra:
    return "paired-spatial-cgra";
  case ExecutionMatrixCell::PairedSystemCgra:
    return "paired-system-cgra";
  }
  llvm_unreachable("closed execution matrix cell");
}

const char *executionMatrixAttemptName(ExecutionMatrixAttemptKind attempt) {
  switch (attempt) {
  case ExecutionMatrixAttemptKind::Ordinary:
    return "ordinary";
  case ExecutionMatrixAttemptKind::Diagnostic:
    return "diagnostic";
  }
  llvm_unreachable("closed execution matrix attempt kind");
}

std::string
executionMatrixInvocationName(const ExecutionMatrixInvocation &invocation) {
  if (invocation.attempt == ExecutionMatrixAttemptKind::Diagnostic &&
      invocation.cell != ExecutionMatrixCell::PairedSystemCgra)
    return "diagnostic-" +
           std::string(executionMatrixCellName(invocation.cell));
  return executionMatrixCellName(invocation.cell);
}

llvm::Expected<ExecutionMatrixInvocation>
parseExecutionMatrixInvocation(llvm::StringRef spelling) {
  if (spelling == "spatial-dfg")
    return ExecutionMatrixInvocation{ExecutionMatrixCell::SpatialDfg,
                                     ExecutionMatrixAttemptKind::Ordinary};
  if (spelling == "spatial-cgra")
    return ExecutionMatrixInvocation{ExecutionMatrixCell::SpatialCgra,
                                     ExecutionMatrixAttemptKind::Ordinary};
  if (spelling == "spatial-rtl")
    return ExecutionMatrixInvocation{ExecutionMatrixCell::SpatialRtl,
                                     ExecutionMatrixAttemptKind::Ordinary};
  if (spelling == "system-dfg")
    return ExecutionMatrixInvocation{ExecutionMatrixCell::SystemDfg,
                                     ExecutionMatrixAttemptKind::Ordinary};
  if (spelling == "system-cgra")
    return ExecutionMatrixInvocation{ExecutionMatrixCell::SystemCgra,
                                     ExecutionMatrixAttemptKind::Ordinary};
  if (spelling == "system-rtl")
    return ExecutionMatrixInvocation{ExecutionMatrixCell::SystemRtl,
                                     ExecutionMatrixAttemptKind::Ordinary};
  if (spelling == "paired-spatial-cgra")
    return ExecutionMatrixInvocation{ExecutionMatrixCell::PairedSpatialCgra,
                                     ExecutionMatrixAttemptKind::Ordinary};
  if (spelling == "paired-system-cgra")
    return ExecutionMatrixInvocation{ExecutionMatrixCell::PairedSystemCgra,
                                     ExecutionMatrixAttemptKind::Diagnostic};
  if (spelling == "diagnostic-system-dfg")
    return ExecutionMatrixInvocation{ExecutionMatrixCell::SystemDfg,
                                     ExecutionMatrixAttemptKind::Diagnostic};
  if (spelling == "diagnostic-system-cgra")
    return ExecutionMatrixInvocation{ExecutionMatrixCell::SystemCgra,
                                     ExecutionMatrixAttemptKind::Diagnostic};
  if (spelling == "diagnostic-system-rtl")
    return ExecutionMatrixInvocation{ExecutionMatrixCell::SystemRtl,
                                     ExecutionMatrixAttemptKind::Diagnostic};
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "unknown execution matrix invocation '%s'",
                                 spelling.str().c_str());
}

} // namespace loom::system_test
