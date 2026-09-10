#include "QualityInternal.h"

#include "ApplicationMappingImage.h"
#include "ApplicationRuntimeValidationInternal.h"
#include "ApplicationSystemRuntimeEvidence.h"
#include "BuildInternal.h"
#include "Common/ArtifactText.h"
#include "Evaluation/ModelProvider.h"
#include "Evaluation/ProductionRegistry.h"
#include "Runtime/Gem5SystemExecution.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

#include <system_error>

namespace loom::application::detail {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "application_system_quality_invalid: " + message);
}

llvm::Expected<evaluation::EvaluationEvidence>
executeSystem(const deployment::FinalizedDeployment &deployment,
              const runtime::FinalizedGem5SimulationBinding &binding,
              const ApplicationActivationInputs &inputs,
              const evaluation::CaseArtifactResolution &resolution,
              const ApplicationSystemQualityContext &context,
              const ResolvedConfig &config, const ArtifactStore &artifacts,
              const BlobStore &blobs,
              std::vector<evaluation::EvaluationEvidence> &completedEvidence) {
  using namespace evaluation;
  auto subjects = EvaluationSubjectBindings::get(
      {{CaseSubjectRoleRef(0), {deployment.reference()}},
       {CaseSubjectRoleRef(1), {binding.reference()}}});
  if (!subjects)
    return subjects.takeError();
  auto evaluationCase = EvaluationCase::get(
      systemSimulationCaseSignatureRef(), std::move(*subjects), inputs.workload,
      inputs.runtimeInput, {}, resolution, artifacts, blobs);
  if (!evaluationCase)
    return evaluationCase.takeError();
  auto descriptor = builtinEvaluationModelDescriptorRef(
      BuiltinEvaluationModel::Gem5SystemCgra);
  if (!descriptor)
    return descriptor.takeError();
  auto model = ResolvedModelBinding::project(*descriptor, {}, config);
  if (!model)
    return model.takeError();
  auto metric = MetricRequest::get(
      {MetricKind::Runtime, EvaluationScope{ScopeFormRef(0), {}}}, {},
      *evaluationCase, resolution, artifacts);
  if (!metric)
    return metric.takeError();
  auto request = EvaluationRequest::get(*evaluationCase, {std::move(*metric)},
                                        {}, std::move(*model), 0, resolution,
                                        artifacts, blobs);
  if (!request)
    return request.takeError();
  auto requestRoot = publishEvaluationRequest(*request, artifacts);
  if (!requestRoot)
    return requestRoot.takeError();
  if (context.deploymentRequest.executionControl.stopRequested())
    return llvm::make_error<
        external_tool::ExternalToolExecutionAdmissionStoppedError>();
  for (const EvaluationEvidence &evidence : completedEvidence)
    if (evidence.requestRef() == *requestRoot)
      return evidence;
  auto preparationContext = context.preparationContext;
  if (auto error = llvm::sys::fs::create_directories(
          preparationContext.bundleDestination))
    return invalid("cannot create System evaluation workspace: " +
                   error.message());
  llvm::SmallString<256> pattern(preparationContext.bundleDestination);
  llvm::sys::path::append(pattern,
                          formatArtifactIdentityHex(requestRoot->artifact));
  llvm::SmallString<256> attempt;
  if (auto error = llvm::sys::fs::createUniqueDirectory(pattern, attempt))
    return invalid("cannot create System evaluation attempt: " +
                   error.message());
  llvm::sys::path::append(attempt, "evaluation");
  preparationContext.bundleDestination = attempt.str().str();
  auto begin = build_detail::MonotonicClock::now();
  auto preparation = prepareEvaluationModelInvocation(
      *request, resolution, artifacts, blobs, preparationContext);
  build_detail::emitElapsed(
      ApplicationBuildOperation::SystemEvaluationPreparation, begin);
  if (!preparation)
    return preparation.takeError();
  if (auto *terminal = std::get_if<EvaluationEvidence>(&*preparation))
    return std::move(*terminal);
  const auto &prepared =
      std::get<EvaluationModelPreparedInvocation>(*preparation);
  begin = build_detail::MonotonicClock::now();
  auto execution = external_tool::executeExternalToolInvocationBundleObserved(
      prepared.externalInvocation(), context.deploymentRequest.executionControl,
      external_tool::ExternalToolResultReusePolicy::AllowExactReuse);
  build_detail::emitElapsed(
      ApplicationBuildOperation::SystemEvaluationExecution, begin);
  if (!execution)
    return execution.takeError();
  build_detail::ApplicationBuildOperationTimer timer(
      ApplicationBuildOperation::SystemEvaluationImport);
  auto evidence = importEvaluationModelInvocation(
      *request, resolution, prepared, *execution, artifacts, blobs);
  if (!evidence)
    return evidence.takeError();
  if (evidence->outcomeKind() == EvidenceOutcomeKind::Completed)
    completedEvidence.push_back(*evidence);
  return std::move(*evidence);
}

std::optional<dse::JointDesignQualityIncompleteReason>
incompleteReason(evaluation::EvidenceOutcomeKind outcome) {
  using Reason = dse::JointDesignQualityIncompleteReason;
  switch (outcome) {
  case evaluation::EvidenceOutcomeKind::Completed:
    return std::nullopt;
  case evaluation::EvidenceOutcomeKind::Unsupported:
    return Reason::Unsupported;
  case evaluation::EvidenceOutcomeKind::ExecutionFailed:
    return Reason::ExecutionFailed;
  case evaluation::EvidenceOutcomeKind::CancelledOrTimeout:
    return Reason::CancelledOrTimeout;
  }
  llvm_unreachable("closed Evaluation outcome domain");
}

} // namespace

llvm::Expected<ApplicationSystemQualityObservation>
acquireApplicationSystemQuality(
    const PreparedApplicationBuild &prepared,
    const ImportedApplicationMapping &mapping,
    const ApplicationSystemQualityContext &context,
    const ResolvedConfig &config, const ArtifactStore &artifacts,
    const BlobStore &blobs,
    std::vector<evaluation::EvaluationEvidence> &completedSystemEvidence) {
  build_detail::ApplicationBuildOperationTimer timer(
      ApplicationBuildOperation::SystemQualityEvaluation);
  ApplicationSystemQualityObservation observation;
  if (context.deploymentRequest.executionControl.stopRequested()) {
    observation.outcome =
        dse::JointDesignQualityIncompleteReason::CancelledOrTimeout;
    return observation;
  }
  auto image =
      buildApplicationMappingImage(prepared, mapping, context.finalLinkedModule,
                                   context.deploymentRequest, artifacts, blobs);
  if (!image)
    return image.takeError();
  auto baseline = buildApplicationHostOnlyDeployment(
      prepared, context.finalLinkedModule, mapping.system.reference(),
      context.deploymentRequest, artifacts, blobs);
  if (!baseline)
    return baseline.takeError();
  auto binding = runtime::finalizeBuiltinGem5SimulationBinding(
      mapping.system.reference(), context.gem5Build, {}, artifacts);
  if (!binding)
    return binding.takeError();
  const auto maximumTicks =
      prepared.portfolioInput
          ? prepared.portfolioInput->input.profile.maximumSimulatedTicks
          : std::nullopt;
  std::vector<ImportedApplicationSystemRun> runs;
  for (const auto *deployment : {&*baseline, &image->deployment}) {
    if (context.deploymentRequest.executionControl.stopRequested()) {
      observation.outcome =
          dse::JointDesignQualityIncompleteReason::CancelledOrTimeout;
      return observation;
    }
    auto inputs = materializeApplicationActivationInputs(
        prepared.preMappingSourceProgram, prepared.preMappingWorkload,
        prepared.preMappingRuntimeInput, *deployment, artifacts, maximumTicks);
    if (!inputs)
      return inputs.takeError();
    auto resolution = runtime::buildGem5SystemCaseResolution(
        *deployment, *binding, inputs->workload, inputs->runtimeInput,
        artifacts, blobs);
    if (!resolution)
      return resolution.takeError();
    auto evidence =
        executeSystem(*deployment, *binding, *inputs, *resolution, context,
                      config, artifacts, blobs, completedSystemEvidence);
    if (!evidence) {
      auto error = evidence.takeError();
      if (!error.isA<
              external_tool::ExternalToolExecutionAdmissionStoppedError>())
        return std::move(error);
      llvm::consumeError(std::move(error));
      observation.outcome =
          dse::JointDesignQualityIncompleteReason::CancelledOrTimeout;
      return observation;
    }
    auto root = evaluation::publishEvaluationEvidence(*evidence, artifacts);
    if (!root)
      return root.takeError();
    observation.evidence.push_back(*root);
    if (auto reason = incompleteReason(evidence->outcomeKind())) {
      observation.outcome = *reason;
      return observation;
    }
    if (evidence->outputBindings().size() != 1 ||
        evidence->outputBindings().front().artifacts.size() != 1)
      return invalid(
          "completed native System Evidence has no unique execution");
    const auto validationBegin = build_detail::MonotonicClock::now();
    auto imported = importApplicationSystemRun(
        {evidence->outputBindings().front().artifacts.front(), *root,
         std::nullopt},
        deployment->reference(), inputs->workload, inputs->runtimeInput,
        *resolution, artifacts, blobs);
    build_detail::emitElapsed(
        ApplicationBuildOperation::SystemEvaluationValidation, validationBegin);
    if (!imported)
      return imported.takeError();
    runs.push_back(std::move(*imported));
  }
  auto compared = compareApplicationSystemRuns(runs[0], runs[1]);
  if (!compared)
    return compared.takeError();
  if (!*compared || !runs[1].measurement.computationInterval) {
    observation.outcome =
        !*compared
            ? dse::JointDesignQualityIncompleteReason::ExecutionFailed
            : dse::JointDesignQualityIncompleteReason::ProofNotEstablished;
    return observation;
  }
  observation.outcome = runs[1].measurement.computationInterval->elapsedTicks();
  return observation;
}

} // namespace loom::application::detail
