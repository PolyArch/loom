#include "Evaluation/Models/PhysicalRailAnalysis.h"

#include "Common/ArtifactStore.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Hardware/Implementation/HardwareImplementation.h"
#include "ImplementationPlatform/ImplementationPlatform.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <array>
#include <functional>
#include <map>
#include <set>
#include <system_error>
#include <utility>
#include <vector>

namespace loom::evaluation::models {

llvm::Expected<CaseArtifactResolution>
resolveHardwareImplementationPhysicalCase(
    const ArtifactRootReference &hardwareImplementation,
    const hardware::ExternalImplementationContractCatalog &externalContracts,
    const ArtifactStore &artifactStore, const BlobStore &blobStore) {
  auto implementation = hardware::importHardwareImplementation(
      hardwareImplementation, externalContracts, artifactStore, blobStore);
  if (!implementation)
    return implementation.takeError();

  std::map<ArtifactRootReference, std::vector<ArtifactRootReference>,
           decltype(&artifactRootReferenceLess)>
      entries(&artifactRootReferenceLess);
  std::set<ArtifactRootReference, decltype(&artifactRootReferenceLess)>
      completed(&artifactRootReferenceLess);
  std::set<ArtifactRootReference, decltype(&artifactRootReferenceLess)>
      visiting(&artifactRootReferenceLess);
  const auto merge = [&](const ArtifactRootReference &owner,
                         llvm::ArrayRef<ArtifactRootReference> dependencies) {
    std::vector<ArtifactRootReference> &closure = entries[owner];
    closure.insert(closure.end(), dependencies.begin(), dependencies.end());
    llvm::sort(closure, artifactRootReferenceLess);
    closure.erase(std::unique(closure.begin(), closure.end()), closure.end());
  };

  std::function<llvm::Expected<std::vector<ArtifactRootReference>>(
      const ArtifactRootReference &)>
      fabricClosure = [&](const ArtifactRootReference &reference)
      -> llvm::Expected<std::vector<ArtifactRootReference>> {
    if (completed.count(reference) != 0)
      return entries[reference];
    if (!visiting.insert(reference).second)
      return llvm::createStringError(
          std::make_error_code(std::errc::invalid_argument),
          "physical_case_resolution_invalid: Fabric dependency closure is "
          "cyclic");
    auto root = fabric::importEntireFabricRoot(reference, artifactStore);
    if (!root) {
      visiting.erase(reference);
      return root.takeError();
    }
    std::vector<ArtifactRootReference> closure;
    for (const fabric::FabricDirectDependency &dependency :
         root->directDependencies()) {
      closure.push_back(dependency.root);
      auto nested = fabricClosure(dependency.root);
      if (!nested) {
        visiting.erase(reference);
        return nested.takeError();
      }
      closure.insert(closure.end(), nested->begin(), nested->end());
    }
    llvm::sort(closure, artifactRootReferenceLess);
    closure.erase(std::unique(closure.begin(), closure.end()), closure.end());
    entries[reference] = closure;
    visiting.erase(reference);
    completed.insert(reference);
    return closure;
  };

  const hardware::HardwareImplementation &view =
      implementation->implementation();
  std::vector<ArtifactRootReference> hardwareClosure = {
      view.fabric(), view.configurationAbi()};
  auto fabricDependencies = fabricClosure(view.fabric());
  if (!fabricDependencies)
    return fabricDependencies.takeError();
  hardwareClosure.insert(hardwareClosure.end(), fabricDependencies->begin(),
                         fabricDependencies->end());
  const std::array<ArtifactRootReference, 1> configurationDependencies = {
      view.fabric()};
  merge(view.configurationAbi(), configurationDependencies);
  if (view.implementationPlatform()) {
    auto platform = platform::importImplementationPlatform(
        *view.implementationPlatform(), artifactStore);
    if (!platform)
      return platform.takeError();
    hardwareClosure.push_back(*view.implementationPlatform());
    entries.try_emplace(*view.implementationPlatform());
  }
  merge(hardwareImplementation, hardwareClosure);

  std::vector<CaseArtifactResolution::Entry> resolved;
  resolved.reserve(entries.size());
  for (auto &[artifact, closure] : entries)
    resolved.push_back({artifact, std::move(closure)});
  return CaseArtifactResolution::get(std::move(resolved));
}

} // namespace loom::evaluation::models
