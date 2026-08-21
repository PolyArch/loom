#ifndef LOOM_MAPPING_ARTIFACT_SYSTEMMAPPINGHARDWAREDEMAND_H
#define LOOM_MAPPING_ARTIFACT_SYSTEMMAPPINGHARDWAREDEMAND_H

#include "Common/Artifact.h"
#include "Common/ArtifactLocalReference.h"
#include "Dataflow/IR/DataflowStructuralRefs.h"
#include "Fabric/Identity/FabricRefs.h"
#include "Mapping/Artifact/SystemPresburger.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <utility>
#include <vector>

namespace loom {
class ArtifactStore;
}

namespace loom::mapping {

inline constexpr ArtifactSchemaDescriptor
    systemExecutionBindingCheckpointArtifactSchema{
        "loom.mapping.system_execution_binding_checkpoint",
        SchemaVersion{1, 0}};

struct SystemThreadExecutionCheckpoint final {
  ::dataflow::RootThreadLaunchRef root;
  SystemPresburgerCell cell;
  ::loom::fabric::AccCoreOccurrenceRef target;
};

struct SystemGraphExecutionCheckpoint final {
  ::dataflow::RootedGraphLaunchRef launch;
  SystemPresburgerCell cell;
  ArtifactRootReference target;
};

/// Persisted semantic projection of one incomplete System PnR execution-
/// binding assignment. It contains no service, route, resource-use, or
/// legality claim; a child PnR invocation may use it only as an initializer
/// preference and must rebuild and verify the complete Mapping.
class FinalizedSystemExecutionBindingCheckpoint final {
public:
  const ArtifactRootReference &reference() const { return reference_; }
  const ArtifactRootReference &dataflow() const { return dataflow_; }
  const ArtifactRootReference &system() const { return system_; }
  llvm::ArrayRef<SystemThreadExecutionCheckpoint> threadBindings() const {
    return threadBindings_;
  }
  llvm::ArrayRef<SystemGraphExecutionCheckpoint> graphBindings() const {
    return graphBindings_;
  }

private:
  FinalizedSystemExecutionBindingCheckpoint(
      ArtifactRootReference reference, ArtifactRootReference dataflow,
      ArtifactRootReference system,
      std::vector<SystemThreadExecutionCheckpoint> threadBindings,
      std::vector<SystemGraphExecutionCheckpoint> graphBindings)
      : reference_(std::move(reference)), dataflow_(std::move(dataflow)),
        system_(std::move(system)), threadBindings_(std::move(threadBindings)),
        graphBindings_(std::move(graphBindings)) {}

  ArtifactRootReference reference_;
  ArtifactRootReference dataflow_;
  ArtifactRootReference system_;
  std::vector<SystemThreadExecutionCheckpoint> threadBindings_;
  std::vector<SystemGraphExecutionCheckpoint> graphBindings_;

  friend llvm::Expected<FinalizedSystemExecutionBindingCheckpoint>
  finalizeSystemExecutionBindingCheckpoint(
      ArtifactRootReference, ArtifactRootReference,
      std::vector<SystemThreadExecutionCheckpoint>,
      std::vector<SystemGraphExecutionCheckpoint>, const ArtifactStore &);
  friend llvm::Expected<FinalizedSystemExecutionBindingCheckpoint>
  importSystemExecutionBindingCheckpoint(const ArtifactRootReference &,
                                         const ArtifactStore &);
};

llvm::Expected<FinalizedSystemExecutionBindingCheckpoint>
finalizeSystemExecutionBindingCheckpoint(
    ArtifactRootReference dataflow, ArtifactRootReference system,
    std::vector<SystemThreadExecutionCheckpoint> threadBindings,
    std::vector<SystemGraphExecutionCheckpoint> graphBindings,
    const ArtifactStore &store);

llvm::Expected<FinalizedSystemExecutionBindingCheckpoint>
importSystemExecutionBindingCheckpoint(const ArtifactRootReference &reference,
                                       const ArtifactStore &store);

/// Exact imported-Spatial capacity pressure after the bounded execution-
/// binding relation has no capacity-closed assignment. It requests one
/// monotonic candidate extension and does not prove larger hardware sufficient.
class SystemAccCoreCapacityPressure final {
public:
  static llvm::Expected<SystemAccCoreCapacityPressure>
  get(ArtifactRootReference system, ArtifactRootReference targetModule,
      ::loom::fabric::AccCoreOccurrenceRef witnessAccCore,
      std::vector<ArtifactRootReference> spatialMappings,
      std::uint64_t compatibleAccCoreCount, std::uint64_t assignmentAttempts,
      std::uint64_t witnessUsage, std::uint64_t witnessCapacity,
      ArtifactRootReference executionBindingCheckpoint);

  const ArtifactRootReference &system() const { return system_; }
  const ArtifactRootReference &targetModule() const { return targetModule_; }
  ::loom::fabric::AccCoreOccurrenceRef witnessAccCore() const {
    return witnessAccCore_;
  }
  llvm::ArrayRef<ArtifactRootReference> spatialMappings() const {
    return spatialMappings_;
  }
  std::uint64_t compatibleAccCoreCount() const {
    return compatibleAccCoreCount_;
  }
  std::uint64_t assignmentAttempts() const { return assignmentAttempts_; }
  std::uint64_t witnessUsage() const { return witnessUsage_; }
  std::uint64_t witnessCapacity() const { return witnessCapacity_; }
  const ArtifactRootReference &executionBindingCheckpoint() const {
    return executionBindingCheckpoint_;
  }
  std::uint64_t additionalAccCoreCount() const { return 1; }

private:
  SystemAccCoreCapacityPressure(
      ArtifactRootReference system, ArtifactRootReference targetModule,
      ::loom::fabric::AccCoreOccurrenceRef witnessAccCore,
      std::vector<ArtifactRootReference> spatialMappings,
      std::uint64_t compatibleAccCoreCount, std::uint64_t assignmentAttempts,
      std::uint64_t witnessUsage, std::uint64_t witnessCapacity,
      ArtifactRootReference executionBindingCheckpoint)
      : system_(std::move(system)), targetModule_(std::move(targetModule)),
        witnessAccCore_(witnessAccCore),
        spatialMappings_(std::move(spatialMappings)),
        compatibleAccCoreCount_(compatibleAccCoreCount),
        assignmentAttempts_(assignmentAttempts), witnessUsage_(witnessUsage),
        witnessCapacity_(witnessCapacity),
        executionBindingCheckpoint_(std::move(executionBindingCheckpoint)) {}

  ArtifactRootReference system_;
  ArtifactRootReference targetModule_;
  ::loom::fabric::AccCoreOccurrenceRef witnessAccCore_;
  std::vector<ArtifactRootReference> spatialMappings_;
  std::uint64_t compatibleAccCoreCount_ = 0;
  std::uint64_t assignmentAttempts_ = 0;
  std::uint64_t witnessUsage_ = 0;
  std::uint64_t witnessCapacity_ = 0;
  ArtifactRootReference executionBindingCheckpoint_;
};

llvm::ArrayRef<std::uint8_t> systemAccCoreCapacityPressureSchemaBytes();

std::vector<std::uint8_t> encodeSystemAccCoreCapacityPressure(
    const SystemAccCoreCapacityPressure &feedback);

llvm::Expected<SystemAccCoreCapacityPressure>
adoptSystemAccCoreCapacityPressure(
    llvm::ArrayRef<std::uint8_t> bytes, const ArtifactRootReference &system,
    const ArtifactRootReference &dataflow,
    llvm::ArrayRef<ArtifactRootReference> spatialMappings,
    const ArtifactStore &store);

} // namespace loom::mapping

#endif // LOOM_MAPPING_ARTIFACT_SYSTEMMAPPINGHARDWAREDEMAND_H
