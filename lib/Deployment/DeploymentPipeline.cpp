#include "Deployment/DeploymentPipeline.h"

#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Frontend/Compilation/StaticGlobalMemory.h"
#include "Frontend/Compilation/StaticMemoryBinding.h"
#include "Frontend/Executable/CompilerTargetBinding.h"
#include "Mapping/Artifact/SystemMappingArtifact.h"

#include "llvm/IR/Module.h"
#include "llvm/Support/Error.h"

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
  dataflow::LogicalMemoryRootRef root;
  std::optional<std::uint64_t> globalOrdinal;
};

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

  std::map<std::uint64_t, LogicalMemorySelection> selections;
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

      const std::uint64_t key = source.root.entity.value();
      auto [entry, inserted] = selections.try_emplace(
          key, LogicalMemorySelection{source.root, imageOrdinal});
      if (inserted)
        continue;
      if (entry->second.globalOrdinal != imageOrdinal)
        return invalid(
            "one logical memory root has incompatible final-link sources");
    }
  }

  std::vector<StaticMemoryImageLeaf> images;
  images.reserve(selections.size());
  for (const auto &[key, selection] : selections) {
    (void)key;
    if (!selection.globalOrdinal)
      continue;
    auto image = buildStaticMemoryImageLeaf(
        dataflowReference, selection.root, hostProgram.compilerTargetBinding(),
        *catalog, *selection.globalOrdinal, artifacts, blobs);
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
  auto staticMemory =
      deriveStaticMemoryImages(inputs.systemMapping, inputs.hostProgram,
                               finalLinkedModule, artifacts, blobs);
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
