#ifndef LOOM_MODELING_ANALYTICMODELSUPPORT_H
#define LOOM_MODELING_ANALYTICMODELSUPPORT_H

#include "Evaluation/Case.h"
#include "Evaluation/Evidence.h"
#include "Evaluation/ModelDescriptor.h"

#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>

namespace loom {
class ArtifactStore;
}

namespace dataflow {
class CanonicalDataflowProgramView;
}

namespace loom::fabric {
class FinalizedFabricRoot;
}

namespace loom::evaluation::models::detail {

const ResolvedModelConfigViewContract &emptyStaticPressureConfigView();

llvm::Expected<CaseArtifactResolution>
resolveSingleSubjectFabricCase(const ArtifactRootReference &subject,
                               const ArtifactRootReference &fabric,
                               const ArtifactStore &artifactStore);

llvm::Expected<MetricResult>
staticPressureRuntimeMetric(std::uint64_t instructionLeaves,
                            std::uint64_t spatialPressure);

llvm::Expected<std::optional<std::uint64_t>> canonicalDataflowStaticPressure(
    const ::dataflow::CanonicalDataflowProgramView &program,
    const fabric::FinalizedFabricRoot &fabricRoot);

} // namespace loom::evaluation::models::detail

#endif // LOOM_MODELING_ANALYTICMODELSUPPORT_H
