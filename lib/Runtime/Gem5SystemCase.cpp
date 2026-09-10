#include "Runtime/Gem5SystemExecution.h"

#include "Deployment/Package.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Runtime/Gem5SimulationBinding.h"

#include <functional>
#include <map>
#include <set>
#include <utility>

namespace loom::runtime {

llvm::Expected<evaluation::CaseArtifactResolution>
buildGem5SystemCaseResolution(
    const deployment::FinalizedDeployment &deployment,
    const runtime::FinalizedGem5SimulationBinding &binding,
    const ArtifactRootReference &workload,
    const ArtifactRootReference &runtimeInput, const ArtifactStore &artifacts,
    const BlobStore &blobs) {
  auto package =
      deployment::deriveDeploymentPackageClosure(deployment, artifacts, blobs);
  if (!package)
    return package.takeError();
  std::map<ArtifactRootReference, std::vector<ArtifactRootReference>,
           decltype(&artifactRootReferenceLess)>
      entries(&artifactRootReferenceLess);
  for (const ArtifactRootReference &root : package->artifacts())
    entries.emplace(root, std::vector<ArtifactRootReference>{});
  std::vector<ArtifactRootReference> deploymentClosure;
  for (const ArtifactRootReference &root : package->artifacts())
    if (root != deployment.reference())
      deploymentClosure.push_back(root);
  entries[deployment.reference()] = deploymentClosure;
  entries[workload] = package->artifacts().vec();
  std::vector<ArtifactRootReference> runtimeClosure =
      package->artifacts().vec();
  runtimeClosure.push_back(workload);
  entries[runtimeInput] = std::move(runtimeClosure);

  std::set<ArtifactRootReference, decltype(&artifactRootReferenceLess)>
      fabricClosure(&artifactRootReferenceLess);
  std::function<llvm::Error(const ArtifactRootReference &)> addFabric =
      [&](const ArtifactRootReference &root) -> llvm::Error {
    if (!fabricClosure.insert(root).second)
      return llvm::Error::success();
    auto imported = fabric::importEntireFabricRoot(root, artifacts);
    if (!imported)
      return imported.takeError();
    entries.emplace(root, std::vector<ArtifactRootReference>{});
    for (const fabric::FabricDirectDependency &dependency :
         imported->directDependencies())
      if (llvm::Error error = addFabric(dependency.root))
        return error;
    return llvm::Error::success();
  };
  if (llvm::Error error = addFabric(binding.binding().fabric()))
    return std::move(error);
  if (llvm::Error error =
          addFabric(binding.binding().interconnectImplementation()))
    return std::move(error);
  entries[binding.reference()] = {fabricClosure.begin(), fabricClosure.end()};

  std::vector<evaluation::CaseArtifactResolution::Entry> result;
  result.reserve(entries.size());
  for (auto &[root, closure] : entries)
    result.push_back({root, std::move(closure)});
  return evaluation::CaseArtifactResolution::get(std::move(result));
}

} // namespace loom::runtime
