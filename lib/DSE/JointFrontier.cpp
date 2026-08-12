#include "DSE/JointDesignPolicy.h"

#include "Common/ArtifactLocalReference.h"
#include "Common/ArtifactStore.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Artifact/FabricSystemRootView.h"
#include "Simulator/SimulationArtifacts.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <limits>
#include <system_error>
#include <utility>
#include <vector>

namespace loom::dse {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "joint_frontier_invalid: " + message);
}

llvm::Error canonicalizeUnique(std::vector<ArtifactRootReference> &roots,
                               llvm::StringRef subject) {
  if (roots.empty())
    return invalid(subject + " frontier is empty");
  llvm::sort(roots, artifactRootReferenceLess);
  if (std::adjacent_find(roots.begin(), roots.end()) != roots.end())
    return invalid(subject + " frontier contains a duplicate root");
  return llvm::Error::success();
}

bool scopeLess(const JointSoftwareScope &lhs, const JointSoftwareScope &rhs) {
  return std::lexicographical_compare(
      lhs.workloads.begin(), lhs.workloads.end(), rhs.workloads.begin(),
      rhs.workloads.end(), artifactRootReferenceLess);
}

} // namespace

llvm::Expected<BoundedJointFrontier>
buildBoundedJointFrontier(JointDesignInputs inputs,
                          const JointDesignPolicy &policy,
                          const ArtifactStore &artifactStore) {
  if (inputs.applicationScopes.empty())
    return invalid("application scope frontier is empty");
  if (llvm::Error error = canonicalizeUnique(inputs.systemFrontier, "System"))
    return std::move(error);
  if (inputs.applicationScopes.size() > policy.maximumSoftwareFrontier())
    return invalid("software frontier exceeds its resolved bound");
  if (inputs.systemFrontier.size() > policy.maximumSystemFrontier())
    return invalid("System frontier exceeds its resolved bound");

  std::vector<JointSoftwareScope> softwareFrontier;
  softwareFrontier.reserve(inputs.applicationScopes.size());
  for (std::vector<ArtifactRootReference> &workloads :
       inputs.applicationScopes) {
    if (llvm::Error error = canonicalizeUnique(workloads, "workload scope"))
      return std::move(error);
    std::optional<ArtifactRootReference> dataflowReference;
    for (const ArtifactRootReference &workload : workloads) {
      auto imported =
          sim::importSpatialSimulationWorkload(workload, artifactStore);
      if (!imported)
        return imported.takeError();
      ArtifactRootReference owner{
          dataflow::canonicalDataflowSchema.identity.str(),
          dataflow::canonicalDataflowSchema.version,
          imported->dataflow.identity()};
      if (dataflowReference && *dataflowReference != owner)
        return invalid("one application scope has multiple Dataflow owners");
      dataflowReference = std::move(owner);
    }
    softwareFrontier.push_back(
        JointSoftwareScope{std::move(*dataflowReference),
                           std::move(workloads)});
  }
  llvm::sort(softwareFrontier, scopeLess);
  if (std::adjacent_find(softwareFrontier.begin(), softwareFrontier.end()) !=
      softwareFrontier.end())
    return invalid("application frontier contains a duplicate scope");
  for (const ArtifactRootReference &root : inputs.systemFrontier) {
    auto artifact = fabric::importEntireFabricRoot(root, artifactStore);
    if (!artifact)
      return artifact.takeError();
    auto system = fabric::requireSystemRoot(artifact->view());
    if (!system)
      return system.takeError();
    if (system->artifact().accCoreOccurrences().empty())
      return invalid("System frontier contains no AccCore occurrence");
  }

  if (softwareFrontier.size() >
      std::numeric_limits<std::uint64_t>::max() /
          inputs.systemFrontier.size())
    return invalid("eligible pair count overflows u64");
  const std::uint64_t eligible =
      static_cast<std::uint64_t>(softwareFrontier.size()) *
      static_cast<std::uint64_t>(inputs.systemFrontier.size());
  const std::uint64_t retained =
      std::min(eligible, policy.maximumPairEvaluations());

  std::vector<JointDesignPair> pairs;
  pairs.reserve(static_cast<std::size_t>(retained));
  for (const JointSoftwareScope &software : softwareFrontier) {
    for (const ArtifactRootReference &system : inputs.systemFrontier) {
      if (pairs.size() == retained)
        break;
      pairs.push_back({software, system});
    }
    if (pairs.size() == retained)
      break;
  }
  return BoundedJointFrontier{std::move(softwareFrontier),
                              std::move(inputs.systemFrontier),
                              std::move(pairs), eligible,
                              retained != eligible};
}

} // namespace loom::dse
