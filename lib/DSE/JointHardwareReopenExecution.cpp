#include "JointHardwareReopenExecution.h"

#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "Config/ResolvedConfig.h"
#include "DSE/ExecutionJournal.h"
#include "DSE/ResolvedConfigView.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Identity/FabricPhysicalTiming.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

#include <system_error>

namespace loom::dse {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "joint_hardware_reopen_execution_invalid: " + message);
}

} // namespace

llvm::Expected<JointDesignExecution>
executeJointPlan(const JointDesignExplorationPlan &plan,
                 llvm::ArrayRef<ArtifactRootReference> evidence,
                 const JointHardwareReopenRequest &request,
                 SiteScheduler &scheduler, const ArtifactStore &artifacts,
                 const BlobStore &blobs,
                 const PlanExecutionPolicy *executionPolicy) {
  const ResolvedConfig &config = plan.resolvedConfig;
  auto publishedConfig = artifacts.put(ResolvedConfig::artifactSchema,
                                       canonicalResolvedConfigBytes(config));
  if (!publishedConfig)
    return publishedConfig.takeError();
  if (*publishedConfig != resolvedConfigIdentity(config))
    return invalid("ResolvedConfig publication changed its identity");
  std::vector<ArtifactRootReference> semanticInputs =
      projectJointDesignSemanticInputs(plan);
  auto closure = DseRunClosure::get(request.producer, semanticInputs, config,
                                    evidence, artifacts);
  if (!closure)
    return closure.takeError();
  auto configView = projectResolvedDseConfigView(config);
  if (!configView)
    return configView.takeError();
  llvm::SmallString<256> journalPath(request.journalRoot);
  llvm::sys::path::append(journalPath,
                          llvm::toHex(closure->runKey().bytes(), true));
  if (std::error_code error = llvm::sys::fs::create_directories(journalPath))
    return invalid("cannot create Mapping alternative journal: " +
                   error.message());
  auto journal = openExecutionJournal(journalPath, *closure, *configView);
  if (!journal)
    return journal.takeError();
  return executeJointDesignExploration(
      plan, *closure, *journal, scheduler,
      executionPolicy ? *executionPolicy : request.executionPolicy, artifacts,
      blobs);
}

llvm::Expected<std::vector<ArtifactRootReference>>
normalizedTimingProfiles(const ArtifactRootReference &system,
                         const ArtifactStore &artifacts) {
  auto modules = projectJointDesignTargetModules(system, artifacts);
  if (!modules)
    return modules.takeError();
  std::vector<ArtifactRootReference> profiles;
  profiles.reserve(modules->size());
  for (const ArtifactRootReference &moduleReference : *modules) {
    auto module = fabric::importEntireFabricRoot(moduleReference, artifacts);
    if (!module)
      return module.takeError();
    auto timing =
        fabric::projectNormalizedFabricPhysicalTimingProfile(module->view());
    if (!timing)
      return timing.takeError();
    auto reference =
        fabric::publishFabricPhysicalTimingProfile(*timing, artifacts);
    if (!reference)
      return reference.takeError();
    profiles.push_back(std::move(*reference));
  }
  return profiles;
}

} // namespace loom::dse
