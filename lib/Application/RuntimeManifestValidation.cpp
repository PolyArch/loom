#include "Application/RuntimeManifest.h"

#include "RuntimeManifestValidation.h"
#include "RuntimeProductContract.h"

#include "Application/ActivationDecision.h"
#include "Application/Build.h"
#include "Common/ArtifactLocalReference.h"
#include "Common/ArtifactStore.h"
#include "Common/ArtifactText.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Deployment/Deployment.h"
#include "Deployment/Package.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Frontend/IR/StructuredProgramArtifact.h"
#include "Mapping/Artifact/SystemMappingArtifact.h"
#include "Simulator/SimulationArtifacts.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <utility>
#include <vector>

namespace loom::application::detail {
namespace {
llvm::Error reject(ApplicationRuntimeManifestErrorReason reason,
                   const llvm::Twine &message) {
  return llvm::make_error<ApplicationRuntimeManifestError>(reason,
                                                           message.str());
}
} // namespace

llvm::Error canonicalizeDigestSet(std::vector<ComponentViewDigest> &digests,
                                  llvm::StringRef name) {
  llvm::sort(digests, [](const auto &lhs, const auto &rhs) {
    return lhs.bytes() < rhs.bytes();
  });
  for (std::size_t index = 1; index < digests.size(); ++index)
    if (digests[index - 1] == digests[index])
      return reject(
          ApplicationRuntimeManifestErrorReason::PairDecisionIncomplete,
          name + " repeats one digest");
  return llvm::Error::success();
}

llvm::Error verifyManifestDraft(ApplicationRuntimeManifestDraft &draft,
                                const ArtifactStore &artifacts,
                                const BlobStore &blobs) {
  auto activation = importApplicationActivationDecision(
      draft.activationDecision, artifacts, blobs);
  if (!activation)
    return reject(
        ApplicationRuntimeManifestErrorReason::ActivationDecisionMismatch,
        "activation decision failed strict import: " +
            llvm::toString(activation.takeError()));
  const ApplicationActivationDecision &decision = activation->decision();
  if (decision.sourceProgram() != draft.sourceProgram ||
      decision.fabric() != draft.fabric ||
      decision.workload() != draft.workload ||
      decision.runtimeInput() != draft.runtimeInput ||
      decision.sourceBackedReplayCases() !=
          llvm::ArrayRef<sim::SourceBackedDfgReplayCaseReference>(
              draft.sourceBackedReplayCases))
    return reject(
        ApplicationRuntimeManifestErrorReason::ActivationDecisionMismatch,
        "activation decision differs from the runtime source lineage");
  if (decision.dseInvocation().occurrence().runKey.bytes() !=
          draft.invocationRunKey ||
      decision.disposition() != draft.pairDisposition ||
      decision.selectedCandidateIdentity() != draft.selectedCandidateIdentity ||
      decision.selectedPlanOrdinal() != draft.selectedPlanOrdinal ||
      decision.selectedSystem() != draft.selectedSystem ||
      decision.selectedMapping() != draft.selectedMapping ||
      decision.runtimeEvidence() !=
          llvm::ArrayRef<ArtifactRootReference>(draft.runtimeEvidence) ||
      decision.oracleEvidence() !=
          llvm::ArrayRef<ArtifactRootReference>(draft.oracleEvidence) ||
      decision.selectedHardwareMutationRepairRecord() !=
          draft.selectedHardwareMutationRepairRecord ||
      decision.hardwareMutationRepairRecords() !=
          llvm::ArrayRef<ArtifactRootReference>(
              draft.hardwareMutationRepairRecords))
    return reject(
        ApplicationRuntimeManifestErrorReason::ActivationDecisionMismatch,
        "activation decision differs from the selected runtime execution");
  std::vector<ComponentViewDigest> decisionScheduleHints;
  decisionScheduleHints.reserve(decision.selectedScheduleHints().size());
  for (const dse::ResourceTimeScheduleHint &hint :
       decision.selectedScheduleHints()) {
    auto digest = dse::deriveResourceTimeScheduleHintDigest(hint);
    if (!digest)
      return reject(
          ApplicationRuntimeManifestErrorReason::ActivationDecisionMismatch,
          "activation decision schedule hint cannot be derived: " +
              llvm::toString(digest.takeError()));
    decisionScheduleHints.push_back(*digest);
  }
  if (llvm::Error error = canonicalizeDigestSet(
          decisionScheduleHints, "activation decision schedule hints"))
    return error;
  if (decisionScheduleHints != draft.selectedScheduleHintDigests)
    return reject(
        ApplicationRuntimeManifestErrorReason::ActivationDecisionMismatch,
        "activation decision differs from the selected schedule hints");

  auto expectedPair = deriveApplicationPairIdentity(
      draft.sourceProgram, draft.fabric, draft.workload, draft.runtimeInput);
  if (!expectedPair)
    return expectedPair.takeError();
  if (*expectedPair != draft.pairIdentity)
    return reject(ApplicationRuntimeManifestErrorReason::PairIdentityMismatch,
                  "runtime manifest pair identity does not match its exact "
                  "source, Fabric, workload, and runtime input roots");
  if (draft.pairDisposition !=
          ApplicationPairDecisionDisposition::VerifiedFeasible &&
      draft.pairDisposition !=
          ApplicationPairDecisionDisposition::HardwareDseAlternative)
    return reject(ApplicationRuntimeManifestErrorReason::PairDecisionIncomplete,
                  "runtime manifest does not carry a completed pair decision");
  const bool selectedDifferentSystem = draft.selectedSystem != draft.fabric;
  if (selectedDifferentSystem !=
      (draft.pairDisposition ==
       ApplicationPairDecisionDisposition::HardwareDseAlternative))
    return reject(
        ApplicationRuntimeManifestErrorReason::PairDecisionIncomplete,
        "hardware-alternative disposition differs from the selected System");
  if (draft.selectedScheduleHintDigests.empty())
    return reject(ApplicationRuntimeManifestErrorReason::PairDecisionIncomplete,
                  "runtime manifest has no selected schedule hint");

  for (const ArtifactRootReference *root :
       {&draft.sourceProgram, &draft.fabric, &draft.workload,
        &draft.runtimeInput, &draft.selectedSystem}) {
    auto stored = artifacts.get(*root);
    if (!stored)
      return reject(ApplicationRuntimeManifestErrorReason::PairIdentityMismatch,
                    "runtime manifest pair root is unavailable: " +
                        llvm::toString(stored.takeError()));
  }
  if (draft.sourceProgram.schemaIdentity !=
          frontend::structuredProgramArtifactSchema.identity ||
      draft.sourceProgram.schemaVersion !=
          frontend::structuredProgramArtifactSchema.version ||
      draft.fabric.schemaIdentity != fabric::fabricArtifactSchema.identity ||
      draft.fabric.schemaVersion != fabric::fabricArtifactSchema.version ||
      draft.selectedSystem.schemaIdentity !=
          fabric::fabricArtifactSchema.identity ||
      draft.selectedSystem.schemaVersion !=
          fabric::fabricArtifactSchema.version ||
      draft.workload.schemaIdentity != sim::simulationWorkloadSchema.identity ||
      draft.workload.schemaVersion != sim::simulationWorkloadSchema.version ||
      draft.runtimeInput.schemaIdentity !=
          sim::simulationRuntimeInputSchema.identity ||
      draft.runtimeInput.schemaVersion !=
          sim::simulationRuntimeInputSchema.version)
    return reject(ApplicationRuntimeManifestErrorReason::PairIdentityMismatch,
                  "runtime manifest pair roots use foreign schemas");
  if (draft.activationWorkload.schemaIdentity !=
          sim::simulationWorkloadSchema.identity ||
      draft.activationWorkload.schemaVersion !=
          sim::simulationWorkloadSchema.version ||
      draft.activationRuntimeInput.schemaIdentity !=
          sim::simulationRuntimeInputSchema.identity ||
      draft.activationRuntimeInput.schemaVersion !=
          sim::simulationRuntimeInputSchema.version)
    return reject(ApplicationRuntimeManifestErrorReason::DeploymentMismatch,
                  "runtime manifest activation roots use foreign schemas");

  auto sourceInputs = sim::importStructuredProgramSimulationInputs(
      draft.workload, draft.runtimeInput, artifacts);
  if (!sourceInputs)
    return reject(ApplicationRuntimeManifestErrorReason::PairIdentityMismatch,
                  "runtime manifest source activation failed strict import: " +
                      llvm::toString(sourceInputs.takeError()));
  if (sourceInputs->structuredProgram.identity() !=
      draft.sourceProgram.artifact)
    return reject(ApplicationRuntimeManifestErrorReason::PairIdentityMismatch,
                  "runtime manifest source activation names a foreign "
                  "StructuredProgram");

  auto importedMapping =
      mapping::importSystemMapping(draft.selectedMapping, artifacts);
  if (!importedMapping)
    return reject(ApplicationRuntimeManifestErrorReason::MappingMismatch,
                  "cannot import selected SystemMapping: " +
                      llvm::toString(importedMapping.takeError()));
  if (importedMapping->view().fabricIdentity() != draft.selectedSystem.artifact)
    return reject(ApplicationRuntimeManifestErrorReason::MappingMismatch,
                  "selected SystemMapping names a foreign selected System");
  const ArtifactRootReference dataflow{
      dataflow::canonicalDataflowSchema.identity.str(),
      dataflow::canonicalDataflowSchema.version,
      importedMapping->view().dataflowIdentity()};

  auto importedDeployment =
      deployment::importDeployment(draft.deployment, artifacts, blobs);
  if (!importedDeployment)
    return reject(ApplicationRuntimeManifestErrorReason::DeploymentMismatch,
                  "cannot import selected Deployment: " +
                      llvm::toString(importedDeployment.takeError()));
  if (!importedDeployment->deployment().systemMapping() ||
      *importedDeployment->deployment().systemMapping() !=
          draft.selectedMapping)
    return reject(ApplicationRuntimeManifestErrorReason::DeploymentMismatch,
                  "Deployment does not bind the selected SystemMapping");
  auto activationInputs = sim::importSystemSimulationInputs(
      draft.activationWorkload, draft.activationRuntimeInput, artifacts, blobs);
  if (!activationInputs)
    return reject(
        ApplicationRuntimeManifestErrorReason::DeploymentMismatch,
        "runtime manifest Deployment activation failed strict import: " +
            llvm::toString(activationInputs.takeError()));
  if (activationInputs->deployment.reference() != draft.deployment)
    return reject(ApplicationRuntimeManifestErrorReason::DeploymentMismatch,
                  "runtime manifest activation inputs name a foreign "
                  "Deployment");
  auto baselineInputs = sim::importSystemSimulationInputs(
      draft.hostOnlyBaseline.inputs.workload,
      draft.hostOnlyBaseline.inputs.runtimeInput, artifacts, blobs);
  if (!baselineInputs)
    return reject(ApplicationRuntimeManifestErrorReason::DeploymentMismatch,
                  "host-only baseline failed strict import: " +
                      llvm::toString(baselineInputs.takeError()));
  const auto &baseline = baselineInputs->deployment;
  if (baseline.reference() != draft.hostOnlyBaseline.deployment ||
      !baseline.deployment().hostOnly() ||
      baseline.deployment().hostOnly()->fabric != draft.selectedSystem ||
      baseline.deployment().hostProgram().compilerTargetBinding() !=
          importedDeployment->deployment()
              .hostProgram()
              .compilerTargetBinding())
    return reject(
        ApplicationRuntimeManifestErrorReason::DeploymentMismatch,
        "host-only baseline changes the selected System or HostCore target");
  auto expectedBaselineInputs = materializeApplicationActivationInputs(
      draft.sourceProgram, draft.workload, draft.runtimeInput, baseline,
      artifacts,
      activationInputs->runtimeInput.system()->maximumSimulatedTicks);
  if (!expectedBaselineInputs)
    return expectedBaselineInputs.takeError();
  if (expectedBaselineInputs->workload !=
          draft.hostOnlyBaseline.inputs.workload ||
      expectedBaselineInputs->runtimeInput !=
          draft.hostOnlyBaseline.inputs.runtimeInput)
    return reject(
        ApplicationRuntimeManifestErrorReason::DeploymentMismatch,
        "host-only baseline does not use the exact source invocation");
  if (draft.productOracle)
    if (llvm::Error error = detail::verifyRuntimeProductContract(
            *draft.productOracle, *sourceInputs, *baselineInputs, baseline,
            artifacts, blobs))
      return error;

  auto deploymentClosure = deployment::deriveDeploymentPackageClosure(
      *importedDeployment, artifacts, blobs);
  if (!deploymentClosure)
    return reject(ApplicationRuntimeManifestErrorReason::DeploymentMismatch,
                  "cannot derive selected Deployment closure: " +
                      llvm::toString(deploymentClosure.takeError()));

  if (draft.productOracle)
    if (llvm::Error error = detail::verifyRuntimeProductContract(
            *draft.productOracle, *sourceInputs, *activationInputs,
            *importedDeployment, artifacts, blobs))
      return error;
  // The strict activation decision owns the exact replay and Evidence proof.
  // The equality checks above bind this manifest to that same immutable domain.
  const ApplicationRuntimeEvidenceJoin &runtime =
      decision.runtimeEvidenceJoin();
  if (!runtime.allCgraExecutionsRetired)
    return reject(
        ApplicationRuntimeManifestErrorReason::RuntimeEvidenceMismatch,
        "completed Application runtime contains a closed-wait CGRA execution");

  std::vector<ArtifactRootReference> deploymentAndExecutionClosure =
      deploymentClosure->artifacts().vec();
  deploymentAndExecutionClosure.insert(deploymentAndExecutionClosure.end(),
                                       runtime.executionOutputs.begin(),
                                       runtime.executionOutputs.end());
  llvm::sort(deploymentAndExecutionClosure, artifactRootReferenceLess);
  deploymentAndExecutionClosure.erase(
      std::unique(deploymentAndExecutionClosure.begin(),
                  deploymentAndExecutionClosure.end()),
      deploymentAndExecutionClosure.end());

  draft.runtimeRequestDependencies.clear();
  std::set_difference(runtime.requestDependencies.begin(),
                      runtime.requestDependencies.end(),
                      deploymentAndExecutionClosure.begin(),
                      deploymentAndExecutionClosure.end(),
                      std::back_inserter(draft.runtimeRequestDependencies),
                      artifactRootReferenceLess);
  for (const ArtifactRootReference &reference :
       draft.runtimeRequestDependencies) {
    auto stored = artifacts.get(reference);
    if (!stored)
      return reject(
          ApplicationRuntimeManifestErrorReason::RuntimeEvidenceMismatch,
          "runtime Evidence Request dependency is unavailable: " +
              formatArtifactRootReferenceJson(reference) + ": " +
              llvm::toString(stored.takeError()));
  }

  if (draft.transitionGraph) {
    if (llvm::Error error = pnr::verifyResourceTimeTransitionGraph(
            *draft.transitionGraph, artifacts, blobs))
      return reject(
          ApplicationRuntimeManifestErrorReason::TransitionGraphMismatch,
          "resource-time transition graph failed independent replay: " +
              llvm::toString(std::move(error)));
    for (const pnr::ResourceTimeTransition &transition :
         draft.transitionGraph->transitions)
      if (!transition.safePoint ||
          transition.safePoint->kind !=
              pnr::ResourceTimeSafePointKind::Completion ||
          transition.safePoint->artifact != dataflow)
        return reject(
            ApplicationRuntimeManifestErrorReason::TransitionGraphMismatch,
            "runtime manifest supports only canonical Dataflow root "
            "completion safe points");
    if (draft.transitionGraph->entry.mapping != draft.selectedMapping ||
        draft.transitionGraph->entry.deployment != draft.deployment)
      return reject(
          ApplicationRuntimeManifestErrorReason::TransitionGraphMismatch,
          "resource-time transition graph entry differs from the selected "
          "Mapping and Deployment");
  }
  return llvm::Error::success();
}

} // namespace loom::application::detail
