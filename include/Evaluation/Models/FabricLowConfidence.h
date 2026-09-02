#ifndef LOOM_EVALUATION_MODELS_FABRICLOWCONFIDENCE_H
#define LOOM_EVALUATION_MODELS_FABRICLOWCONFIDENCE_H

#include "Evaluation/ModelDescriptor.h"

#include "llvm/Support/Error.h"

#include <cstdint>

namespace loom::fabric {
class FinalizedFabricRoot;
} // namespace loom::fabric

namespace loom::evaluation::models {

llvm::Error registerFabricLowConfidenceProvider();
EvaluationModelDescriptorRef fabricLowConfidenceModelDescriptorRef();

/// The picosecond clock basis of every low-confidence estimate over one
/// Fabric: the structure-derived critical delay whose reciprocal the analytic
/// models report as LimitingClockFrequency. A consumer that expresses measured
/// cycles in the analytic picosecond domain multiplies by exactly this period.
llvm::Expected<std::uint64_t> fabricLowConfidenceClockPeriodPicoseconds(
    const fabric::FinalizedFabricRoot &fabricRoot);

} // namespace loom::evaluation::models

#endif // LOOM_EVALUATION_MODELS_FABRICLOWCONFIDENCE_H
