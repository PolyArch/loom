#ifndef LOOM_PNR_PNRCONFIG_H
#define LOOM_PNR_PNRCONFIG_H

#include "Common/ComponentViewDigest.h"
#include "Common/ResolvedPnrPolicy.h"
#include "Dataflow/IR/DataflowCanonicalEntity.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <utility>
#include <vector>

namespace loom {
struct ResolvedConfig;
}

namespace loom::pnr {

struct ResolvedPnrConfigViewAccess;

enum class PnrConfigDomain : std::uint32_t { Spatial, System };

enum class PnrWorkUnit : std::uint32_t {
  SeedAttempt,
  AssignmentAttemptPerSeed,
  EndpointExpansion,
  NegotiationIteration,
  ConsecutiveNoProgressIteration,
  NoProgressTrendTransition,
  CalibrationProposal,
  TemperatureLevel,
  ProposalPerLevelBase,
  ProposalPerMovableDecision,
  ExactRepairRegionDecision,
  ExactRepairSolverCall,
};

llvm::ArrayRef<std::uint8_t> resolvedSpatialPnrConfigSchemaDescriptorBytes();
llvm::ArrayRef<std::uint8_t> resolvedSystemPnrConfigSchemaDescriptorBytes();

struct DeterministicWorkBudgetEntry final {
  PnrWorkUnit unit;
  std::uint64_t limit;
};

/// Invocation-specific System search granularity for one Dataflow root. This
/// is a schedule preference carried by the resolved provider binding; it does
/// not constrain Mapping legality or prescribe physical AccCore identities.
struct SystemBindingPartitionIntent final {
  ::dataflow::RootThreadLaunchRef root;
  std::uint64_t partitionCount = 1;

  friend bool operator==(const SystemBindingPartitionIntent &lhs,
                         const SystemBindingPartitionIntent &rhs) {
    return lhs.root == rhs.root && lhs.partitionCount == rhs.partitionCount;
  }
};

class ResolvedPnrConfigView final {
public:
  PnrConfigDomain domain() const { return domain_; }
  const ResolvedPnrPolicyConfig &policy() const { return policy_; }
  const ResolvedObjectiveCatalogs &selectedObjectiveCatalogs() const {
    return selectedObjectiveCatalogs_;
  }
  llvm::ArrayRef<SystemBindingPartitionIntent> systemBindingPartitions() const {
    return systemBindingPartitions_;
  }

  llvm::ArrayRef<std::uint8_t> schemaDescriptorBytes() const;
  llvm::ArrayRef<std::uint8_t> canonicalViewBytes() const {
    return canonicalBytes_;
  }
  const ComponentViewDigest &digest() const { return digest_; }

private:
  ResolvedPnrConfigView(
      PnrConfigDomain domain, ResolvedPnrPolicyConfig policy,
      ResolvedObjectiveCatalogs selectedObjectiveCatalogs,
      std::vector<SystemBindingPartitionIntent> systemBindingPartitions,
      std::vector<std::uint8_t> canonicalBytes, ComponentViewDigest digest)
      : domain_(domain), policy_(std::move(policy)),
        selectedObjectiveCatalogs_(std::move(selectedObjectiveCatalogs)),
        systemBindingPartitions_(std::move(systemBindingPartitions)),
        canonicalBytes_(std::move(canonicalBytes)), digest_(digest) {}

  PnrConfigDomain domain_;
  ResolvedPnrPolicyConfig policy_;
  ResolvedObjectiveCatalogs selectedObjectiveCatalogs_;
  std::vector<SystemBindingPartitionIntent> systemBindingPartitions_;
  std::vector<std::uint8_t> canonicalBytes_;
  ComponentViewDigest digest_;

  friend llvm::Expected<ResolvedPnrConfigView>
  projectResolvedSpatialPnrConfigView(const ResolvedConfig &config);
  friend llvm::Expected<ResolvedPnrConfigView>
  projectResolvedSystemPnrConfigView(const ResolvedConfig &config);
  friend llvm::Expected<ResolvedPnrConfigView>
  adoptResolvedSpatialPnrConfigView(
      llvm::ArrayRef<std::uint8_t> schemaDescriptorBytes,
      llvm::ArrayRef<std::uint8_t> canonicalViewBytes,
      const ComponentViewDigest &digest);
  friend llvm::Expected<ResolvedPnrConfigView> adoptResolvedSystemPnrConfigView(
      llvm::ArrayRef<std::uint8_t> schemaDescriptorBytes,
      llvm::ArrayRef<std::uint8_t> canonicalViewBytes,
      const ComponentViewDigest &digest);
  friend llvm::Expected<ResolvedPnrConfigView>
  specializeResolvedSystemPnrConfigView(
      const ResolvedPnrConfigView &,
      llvm::ArrayRef<SystemBindingPartitionIntent>);
  friend struct ResolvedPnrConfigViewAccess;
};

llvm::Expected<ResolvedPnrConfigView>
projectResolvedSpatialPnrConfigView(const ResolvedConfig &config);
llvm::Expected<ResolvedPnrConfigView>
projectResolvedSystemPnrConfigView(const ResolvedConfig &config);

llvm::Expected<ResolvedPnrConfigView> adoptResolvedSpatialPnrConfigView(
    llvm::ArrayRef<std::uint8_t> schemaDescriptorBytes,
    llvm::ArrayRef<std::uint8_t> canonicalViewBytes,
    const ComponentViewDigest &digest);
llvm::Expected<ResolvedPnrConfigView> adoptResolvedSystemPnrConfigView(
    llvm::ArrayRef<std::uint8_t> schemaDescriptorBytes,
    llvm::ArrayRef<std::uint8_t> canonicalViewBytes,
    const ComponentViewDigest &digest);

/// Returns the same System provider policy with one canonical, Dataflow-rooted
/// partition intent. The resulting component digest changes mechanically.
llvm::Expected<ResolvedPnrConfigView> specializeResolvedSystemPnrConfigView(
    const ResolvedPnrConfigView &base,
    llvm::ArrayRef<SystemBindingPartitionIntent> partitions);

std::vector<DeterministicWorkBudgetEntry>
deriveDeterministicWorkBudgetView(const ResolvedPnrConfigView &view);

} // namespace loom::pnr

#endif // LOOM_PNR_PNRCONFIG_H
