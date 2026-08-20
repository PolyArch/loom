#ifndef LOOM_MAPPING_ARTIFACT_SYSTEMMAPPINGHARDWAREDEMAND_H
#define LOOM_MAPPING_ARTIFACT_SYSTEMMAPPINGHARDWAREDEMAND_H

#include "Common/ArtifactLocalReference.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <utility>
#include <vector>

namespace loom {
class ArtifactStore;
}

namespace loom::mapping {

/// Exact imported-Spatial capacity pressure after the bounded execution-
/// binding relation has no capacity-closed assignment. It requests one
/// monotonic candidate extension and does not prove larger hardware sufficient.
class SystemAccCoreCapacityPressure final {
public:
  static llvm::Expected<SystemAccCoreCapacityPressure>
  get(ArtifactRootReference system, ArtifactRootReference targetModule,
      std::vector<ArtifactRootReference> spatialMappings,
      std::uint64_t compatibleAccCoreCount, std::uint64_t assignmentAttempts,
      std::uint64_t witnessUsage, std::uint64_t witnessCapacity);

  const ArtifactRootReference &system() const { return system_; }
  const ArtifactRootReference &targetModule() const { return targetModule_; }
  llvm::ArrayRef<ArtifactRootReference> spatialMappings() const {
    return spatialMappings_;
  }
  std::uint64_t compatibleAccCoreCount() const {
    return compatibleAccCoreCount_;
  }
  std::uint64_t assignmentAttempts() const { return assignmentAttempts_; }
  std::uint64_t witnessUsage() const { return witnessUsage_; }
  std::uint64_t witnessCapacity() const { return witnessCapacity_; }
  std::uint64_t additionalAccCoreCount() const { return 1; }

private:
  SystemAccCoreCapacityPressure(
      ArtifactRootReference system, ArtifactRootReference targetModule,
      std::vector<ArtifactRootReference> spatialMappings,
      std::uint64_t compatibleAccCoreCount, std::uint64_t assignmentAttempts,
      std::uint64_t witnessUsage, std::uint64_t witnessCapacity)
      : system_(std::move(system)), targetModule_(std::move(targetModule)),
        spatialMappings_(std::move(spatialMappings)),
        compatibleAccCoreCount_(compatibleAccCoreCount),
        assignmentAttempts_(assignmentAttempts), witnessUsage_(witnessUsage),
        witnessCapacity_(witnessCapacity) {}

  ArtifactRootReference system_;
  ArtifactRootReference targetModule_;
  std::vector<ArtifactRootReference> spatialMappings_;
  std::uint64_t compatibleAccCoreCount_ = 0;
  std::uint64_t assignmentAttempts_ = 0;
  std::uint64_t witnessUsage_ = 0;
  std::uint64_t witnessCapacity_ = 0;
};

llvm::ArrayRef<std::uint8_t> systemAccCoreCapacityPressureSchemaBytes();

std::vector<std::uint8_t> encodeSystemAccCoreCapacityPressure(
    const SystemAccCoreCapacityPressure &feedback);

llvm::Expected<SystemAccCoreCapacityPressure>
adoptSystemAccCoreCapacityPressure(
    llvm::ArrayRef<std::uint8_t> bytes, const ArtifactRootReference &system,
    llvm::ArrayRef<ArtifactRootReference> spatialMappings,
    const ArtifactStore &store);

} // namespace loom::mapping

#endif // LOOM_MAPPING_ARTIFACT_SYSTEMMAPPINGHARDWAREDEMAND_H
