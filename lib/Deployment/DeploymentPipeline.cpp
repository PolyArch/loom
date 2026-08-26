#include "Deployment/DeploymentPipeline.h"
#include "Deployment/DeploymentDiagnostics.h"

#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Dataflow/IR/DataflowReferenceCodec.h"
#include "Frontend/Compilation/StaticGlobalMemory.h"
#include "Frontend/Compilation/StaticMemoryBinding.h"
#include "Frontend/Executable/CompilerTargetBinding.h"
#include "Mapping/Artifact/SystemMappingArtifact.h"

#include "llvm/IR/Module.h"
#include "llvm/Support/Error.h"

#include <chrono>
#include <cstdint>
#include <map>
#include <optional>
#include <system_error>
#include <utility>
#include <vector>

namespace loom::deployment {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      llvm::Twine("deployment_pipeline_invalid: ") + message);
}

struct LogicalMemorySelection final {
  dataflow::RootedGraphLaunchRef launch;
  dataflow::LogicalMemoryRootRef root;
  std::optional<std::uint64_t> globalOrdinal;
};

llvm::Expected<std::vector<std::uint8_t>>
selectionKey(const ArtifactIdentity &owner,
             dataflow::RootedGraphLaunchRef launch,
             dataflow::LogicalMemoryRootRef root) {
  auto launchBytes = dataflow::encodeDataflowReference(owner, launch);
  if (!launchBytes)
    return launchBytes.takeError();
  auto rootBytes = dataflow::encodeDataflowReference(owner, root);
  if (!rootBytes)
    return rootBytes.takeError();
  launchBytes->insert(launchBytes->end(), rootBytes->begin(), rootBytes->end());
  return launchBytes;
}

llvm::Expected<std::vector<StaticMemoryImageLeaf>> deriveStaticMemoryImages(
    const ArtifactRootReference &systemMapping,
    const HostProgramLeaf &hostProgram, const llvm::Module &finalLinkedModule,
    const ArtifactStore &artifacts, const BlobStore &blobs) {
  auto systemMappingArtifact =
      mapping::importSystemMapping(systemMapping, artifacts);
  if (!systemMappingArtifact)
    return invalid("cannot import SystemMapping: " +
                   llvm::toString(systemMappingArtifact.takeError()));

  const ArtifactRootReference dataflowReference{
      dataflow::canonicalDataflowSchema.identity.str(),
      dataflow::canonicalDataflowSchema.version,
      systemMappingArtifact->view().dataflowIdentity()};
  auto dataflow =
      dataflow::importCanonicalDataflow(dataflowReference, artifacts);
  if (!dataflow)
    return invalid("cannot import Canonical Dataflow: " +
                   llvm::toString(dataflow.takeError()));
  auto dataflowView = dataflow->view();
  if (!dataflowView)
    return invalid("cannot reconstruct Canonical Dataflow view: " +
                   llvm::toString(dataflowView.takeError()));

  auto layoutBinding = importCompilerTargetBinding(
      hostProgram.compilerTargetBinding(), artifacts);
  if (!layoutBinding)
    return invalid("cannot import host CompilerTargetBinding: " +
                   llvm::toString(layoutBinding.takeError()));
  if (llvm::Error error = validateModuleCompilerTarget(
          finalLinkedModule, layoutBinding->binding()))
    return invalid(
        "final linked module is incompatible with the host target: " +
        llvm::toString(std::move(error)));

  auto catalog = frontend::projectStaticGlobalMemory(finalLinkedModule);
  if (!catalog)
    return invalid("cannot project final linked static memory: " +
                   llvm::toString(catalog.takeError()));

  std::map<std::vector<std::uint8_t>, LogicalMemorySelection> selections;
  for (const mapping::SystemGraphExecutionBindingView &binding :
       systemMappingArtifact->view().executionBindings().graphBindings()) {
    auto sources = frontend::deriveRootedLogicalMemorySources(
        *catalog, *dataflowView, binding.key);
    if (!sources)
      return invalid("cannot derive selected logical-memory sources: " +
                     llvm::toString(sources.takeError()));

    for (const frontend::RootedLogicalMemorySource &source : *sources) {
      std::optional<std::uint64_t> imageOrdinal = source.globalOrdinal;
      if (imageOrdinal && catalog->globals[*imageOrdinal].provision !=
                              frontend::StaticGlobalProvision::Image)
        imageOrdinal.reset();

      auto key =
          selectionKey(dataflowView->identity(), binding.key, source.root);
      if (!key)
        return invalid("cannot encode selected logical-memory source: " +
                       llvm::toString(key.takeError()));
      auto [entry, inserted] = selections.try_emplace(
          std::move(*key),
          LogicalMemorySelection{binding.key, source.root, imageOrdinal});
      if (inserted)
        continue;
      if (entry->second.globalOrdinal != imageOrdinal)
        return invalid(
            "one rooted logical memory source has incompatible final-link "
            "bindings");
    }
  }

  std::vector<StaticMemoryImageLeaf> images;
  images.reserve(selections.size());
  for (const auto &[key, selection] : selections) {
    (void)key;
    if (!selection.globalOrdinal)
      continue;
    auto image = buildStaticMemoryImageLeaf(
        dataflowReference, selection.launch, selection.root,
        hostProgram.compilerTargetBinding(), *catalog, *selection.globalOrdinal,
        artifacts, blobs);
    if (!image)
      return invalid("cannot build static logical-memory image: " +
                     llvm::toString(image.takeError()));
    images.push_back(std::move(*image));
  }
  return images;
}

} // namespace

llvm::Expected<FinalizedDeployment> buildDeploymentFromLinkedProgram(
    DeploymentPipelineInputs inputs, const llvm::Module &finalLinkedModule,
    const ArtifactStore &artifacts, const BlobStore &blobs) {
  const auto staticMemoryBegin = std::chrono::steady_clock::now();
  auto staticMemory =
      deriveStaticMemoryImages(inputs.systemMapping, inputs.hostProgram,
                               finalLinkedModule, artifacts, blobs);
  emitDeploymentConstructionOperationStatistics(
      {DeploymentConstructionMode::Build,
       DeploymentConstructionOperation::StaticMemoryDerivation,
       static_cast<std::uint64_t>(
           std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now() - staticMemoryBegin)
               .count()),
       staticMemory ? staticMemory->size() : 0, std::nullopt, std::nullopt});
  if (!staticMemory)
    return staticMemory.takeError();

  return buildDeployment(
      ExactDeploymentInputs{
          std::move(inputs.systemMapping), std::move(inputs.hostProgram),
          std::move(inputs.instructionCoreBinaries),
          std::move(inputs.hardwareBindings), std::move(*staticMemory)},
      artifacts, blobs);
}

} // namespace loom::deployment
