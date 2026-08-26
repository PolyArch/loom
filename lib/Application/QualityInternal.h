#ifndef LOOM_LIB_APPLICATION_QUALITYINTERNAL_H
#define LOOM_LIB_APPLICATION_QUALITYINTERNAL_H

#include "Application/Build.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <variant>
#include <vector>

namespace loom::application::detail {

struct ApplicationRuntimeValidation;

std::optional<dse::JointBoundedQualityPolicy>
rebaseApplicationBoundedQualityPolicy(
    const std::optional<dse::JointBoundedQualityPolicy> &policy,
    std::uint64_t planOrdinalBase);

llvm::Error recordApplicationQualityInvocation(
    dse::JointDesignExecution &execution, std::uint64_t planOrdinalBase,
    std::vector<ApplicationPairQualityInvocationRecord> &invocations);

using ApplicationRepairQualityChoice =
    std::variant<std::monostate, dse::JointRepairQualitySelection,
                 dse::JointRepairQualityIncomplete>;

llvm::Expected<ApplicationRepairQualityChoice> chooseApplicationRepairByQuality(
    llvm::ArrayRef<dse::JointDesignExecution> executions,
    const std::optional<dse::JointBoundedQualityPolicy> &quality,
    const ArtifactStore &artifacts);

llvm::Expected<ApplicationRuntimeValidation>
projectApplicationQualityRuntime(
    const dse::JointDesignExecution &execution,
    const ArtifactRootReference &mapping,
    const dse::JointBoundedQualityPolicy &quality,
    const ArtifactStore &artifacts);

} // namespace loom::application::detail

#endif // LOOM_LIB_APPLICATION_QUALITYINTERNAL_H
