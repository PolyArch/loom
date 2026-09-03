#include "ActivationRepairLineage.h"

#include "Application/ActivationDecision.h"
#include "Application/Build.h"
#include "Common/ArtifactStore.h"
#include "DSE/HardwareMutationRepairRecord.h"
#include "Mapping/Artifact/SystemMappingArtifact.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"

#include <utility>
#include <vector>

namespace loom::application::activation_detail {
namespace {

llvm::Error reject(const llvm::Twine &message) {
  return llvm::make_error<ApplicationActivationDecisionError>(
      ApplicationActivationDecisionErrorReason::HardwareMutationRepairMismatch,
      message.str());
}

} // namespace

llvm::Error
validateHardwareMutationRepairs(const ApplicationActivationDecisionDraft &draft,
                                const ArtifactStore &artifacts) {
  std::vector<
      std::pair<ArtifactRootReference, dse::HardwareMutationRepairRecord>>
      repairs;
  repairs.reserve(draft.hardwareMutationRepairRecords.size());
  for (const ArtifactRootReference &reference :
       draft.hardwareMutationRepairRecords) {
    auto imported =
        dse::importHardwareMutationRepairRecord(reference, artifacts);
    if (!imported)
      return reject("hardware mutation repair record failed strict import: " +
                    llvm::toString(imported.takeError()));
    auto parent = mapping::importSystemMapping(imported->record().parentMapping,
                                               artifacts);
    if (!parent)
      return reject(
          "hardware mutation repair parent Mapping failed strict import: " +
          llvm::toString(parent.takeError()));
    if (parent->view().dataflowIdentity() !=
        draft.planning.canonicalDataflow.artifact)
      return reject(
          "hardware mutation repair record names a foreign CanonicalDataflow");
    repairs.push_back({reference, imported->record()});
  }

  std::vector<ArtifactRootReference> reachableSystems = {draft.fabric};
  std::vector<bool> reachableRepairs(repairs.size(), false);
  bool grewReachability = true;
  while (grewReachability) {
    grewReachability = false;
    for (const auto indexed : llvm::enumerate(repairs)) {
      if (reachableRepairs[indexed.index()] ||
          !llvm::is_contained(reachableSystems,
                              indexed.value().second.parentSystem))
        continue;
      reachableRepairs[indexed.index()] = true;
      grewReachability = true;
      const auto rememberSystem = [&](const ArtifactRootReference &system) {
        if (!llvm::is_contained(reachableSystems, system))
          reachableSystems.push_back(system);
      };
      rememberSystem(indexed.value().second.childSystem);
      for (const dse::HardwareMutationImpactRecord &impact :
           indexed.value().second.impacts)
        if (impact.child)
          rememberSystem(*impact.child);
    }
  }
  if (llvm::is_contained(reachableRepairs, false))
    return reject(
        "hardware mutation repair provenance is not rooted in the source "
        "Fabric");

  std::vector<ArtifactRootReference> selectedRepairs;
  for (const auto &entry : repairs)
    if (entry.second.childSystem == draft.selectedSystem &&
        llvm::is_contained(entry.second.incremental.mappings,
                           draft.selectedMapping))
      selectedRepairs.push_back(entry.first);
  if (selectedRepairs.size() > 1)
    return reject(
        "more than one hardware mutation repair record selects the exact "
        "SystemMapping");

  const bool hardwareAlternative =
      draft.disposition ==
      ApplicationPairDecisionDisposition::HardwareDseAlternative;
  if (selectedRepairs.empty()) {
    if (draft.selectedHardwareMutationRepairRecord)
      return reject(
          "selected hardware mutation repair record does not select the "
          "activation SystemMapping");
    return llvm::Error::success();
  }
  if (!hardwareAlternative || !draft.selectedHardwareMutationRepairRecord ||
      *draft.selectedHardwareMutationRepairRecord != selectedRepairs.front())
    return reject(
        "the unique hardware mutation repair for the activation SystemMapping "
        "is not selected");
  return llvm::Error::success();
}

} // namespace loom::application::activation_detail
