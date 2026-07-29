#ifndef LOOM_EVALUATION_STANDARDFINDINGS_H
#define LOOM_EVALUATION_STANDARDFINDINGS_H

#include "Evaluation/Finding.h"

namespace loom::evaluation::standard_findings {

inline constexpr FindingKind FunctionalMismatch{0};

struct FunctionalMismatchOccurrence final {};

/// Registers the schema-1.0 findings owned by the shared Evaluation registry.
/// Repeated registration in one process is a no-op.
llvm::Error registerStandardFindings();

} // namespace loom::evaluation::standard_findings

#endif // LOOM_EVALUATION_STANDARDFINDINGS_H
