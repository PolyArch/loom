#include "DSE/JointDesignPolicy.h"

#include "Common/ArtifactLocalReference.h"
#include "Common/ArtifactStore.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Artifact/FabricSystemRootView.h"

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

} // namespace

llvm::Expected<BoundedJointFrontier>
buildBoundedJointFrontier(JointDesignInputs inputs,
                          const JointDesignPolicy &policy,
                          const ArtifactStore &artifactStore) {
  if (llvm::Error error =
          canonicalizeUnique(inputs.softwareFrontier, "software"))
    return std::move(error);
  if (llvm::Error error = canonicalizeUnique(inputs.systemFrontier, "System"))
    return std::move(error);
  if (inputs.softwareFrontier.size() > policy.maximumSoftwareFrontier())
    return invalid("software frontier exceeds its resolved bound");
  if (inputs.systemFrontier.size() > policy.maximumSystemFrontier())
    return invalid("System frontier exceeds its resolved bound");

  for (const ArtifactRootReference &software : inputs.softwareFrontier) {
    auto artifact = dataflow::importCanonicalDataflow(software, artifactStore);
    if (!artifact)
      return artifact.takeError();
    auto view = artifact->view();
    if (!view)
      return view.takeError();
    if (view->rootThreadLaunches().empty())
      return invalid("software frontier contains no root thread launch");
  }
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

  if (inputs.softwareFrontier.size() >
      std::numeric_limits<std::uint64_t>::max() /
          inputs.systemFrontier.size())
    return invalid("eligible pair count overflows u64");
  const std::uint64_t eligible =
      static_cast<std::uint64_t>(inputs.softwareFrontier.size()) *
      static_cast<std::uint64_t>(inputs.systemFrontier.size());
  const std::uint64_t retained =
      std::min(eligible, policy.maximumPairEvaluations());

  std::vector<JointDesignPair> pairs;
  pairs.reserve(static_cast<std::size_t>(retained));
  for (const ArtifactRootReference &software : inputs.softwareFrontier) {
    for (const ArtifactRootReference &system : inputs.systemFrontier) {
      if (pairs.size() == retained)
        break;
      pairs.push_back({software, system});
    }
    if (pairs.size() == retained)
      break;
  }
  return BoundedJointFrontier{std::move(inputs.softwareFrontier),
                              std::move(inputs.systemFrontier),
                              std::move(pairs), eligible,
                              retained != eligible};
}

} // namespace loom::dse
