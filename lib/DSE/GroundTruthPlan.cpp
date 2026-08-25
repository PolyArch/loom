#include "DSE/GroundTruthPlan.h"

#include "DSE/EvidenceObligationSetConfig.h"
#include "DSE/ModelParameterCalibrationAcquisition.h"
#include "Evaluation/Models/FpaParameterContract.h"
#include "Evaluation/Models/SystemRuntimeParameterContract.h"
#include "Evaluation/ProductionRegistry.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

namespace loom::dse {
namespace {

struct ObligationRecord final {
  ModelParameterCalibrationTarget target;
  CalibrationPartitionRole partition;
  EvidenceObligationTemplate obligation;
};

struct GateRecord final {
  ModelParameterCalibrationTarget target;
  CalibrationPartitionRole partition;
  QualityGatePolicy gate;
};

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "ground_truth_plan_invalid: " + message);
}

void canonicalizeRoots(std::vector<ArtifactRootReference> &roots) {
  llvm::sort(roots, artifactRootReferenceLess);
  roots.erase(std::unique(roots.begin(), roots.end()), roots.end());
}

llvm::Error validateEvidenceRoots(llvm::ArrayRef<ArtifactRootReference> roots,
                                  llvm::StringRef partition) {
  if (roots.empty())
    return invalid(partition + " Evidence partition is empty");
  for (const ArtifactRootReference &root : roots)
    if (root.schemaIdentity !=
            evaluation::EvaluationEvidence::artifactSchema.identity ||
        root.schemaVersion !=
            evaluation::EvaluationEvidence::artifactSchema.version)
      return invalid(partition + " partition contains a non-Evidence root");
  return llvm::Error::success();
}

llvm::Error canonicalizeTrack(GroundTruthModelTrack &track) {
  canonicalizeRoots(track.evidence.training);
  canonicalizeRoots(track.evidence.validation);
  canonicalizeRoots(track.evidence.heldOut);
  if (llvm::Error error =
          validateEvidenceRoots(track.evidence.training, "Training"))
    return error;
  if (llvm::Error error =
          validateEvidenceRoots(track.evidence.validation, "Validation"))
    return error;
  if (llvm::Error error =
          validateEvidenceRoots(track.evidence.heldOut, "HeldOut"))
    return error;
  if (track.training.minimumTrainingRowsPerLeaf >
      track.evidence.training.size())
    return invalid("Training partition cannot populate one requested leaf");
  return llvm::Error::success();
}

std::uint32_t findObligation(llvm::ArrayRef<ObligationRecord> obligations,
                             ModelParameterCalibrationTarget target,
                             CalibrationPartitionRole partition) {
  for (std::size_t index = 0; index != obligations.size(); ++index)
    if (obligations[index].target == target &&
        obligations[index].partition == partition)
      return static_cast<std::uint32_t>(index);
  llvm_unreachable("ground-truth obligation was not constructed");
}

std::uint32_t findGate(llvm::ArrayRef<GateRecord> gates,
                       ModelParameterCalibrationTarget target,
                       CalibrationPartitionRole partition) {
  for (std::size_t index = 0; index != gates.size(); ++index)
    if (gates[index].target == target && gates[index].partition == partition)
      return static_cast<std::uint32_t>(index);
  llvm_unreachable("ground-truth quality gate was not constructed");
}

llvm::Expected<QualityGatePolicy>
makeGate(std::uint32_t obligation, ModelParameterCalibrationTarget target,
         evaluation::DecimalValue threshold) {
  constexpr std::size_t calibrationQuantileCount = 2;
  const std::size_t metricCount =
      (target == ModelParameterCalibrationTarget::Fpa ? 4 : 1) *
      calibrationQuantileCount;
  std::vector<QualityGateClause> clauses;
  clauses.reserve(metricCount);
  for (std::size_t metric = 0; metric != metricCount; ++metric)
    clauses.push_back({{MetricGate{
        obligation, evaluation::MetricRequestOrdinal(metric),
        MetricGateComparator::LE, evaluation::MetricValue{threshold}}}});
  return QualityGatePolicy::get(std::move(clauses));
}

llvm::Expected<GeneratePlanNodeDefinition>
makeTrainer(ModelParameterCalibrationTarget target,
            const GroundTruthModelTrack &track) {
  auto config = resolveDeterministicGbdtTrainingConfig(track.training);
  if (!config)
    return config.takeError();
  const CandidateGeneratorDescriptor &descriptor =
      target == ModelParameterCalibrationTarget::Fpa
          ? fpaGbdtTrainingCandidateGeneratorDescriptor()
          : systemRuntimeGbdtTrainingCandidateGeneratorDescriptor();
  std::vector<PlanInputBinding> inputs = {
      ExactPlanArtifacts{track.evidence.training},
      ExactPlanArtifacts{track.evidence.validation},
      ExactPlanArtifacts{track.evidence.heldOut},
      ExactPlanArtifacts{
          track.evidence.priorParameterBundle
              ? std::vector<ArtifactRootReference>{*track.evidence
                                                        .priorParameterBundle}
              : std::vector<ArtifactRootReference>{}}};
  return GeneratePlanNodeDefinition{descriptor.reference(), std::move(inputs),
                                    config->canonicalViewBytes().vec(),
                                    config->digest()};
}

llvm::Expected<PromotePlanNodeDefinition>
makePromote(ModelParameterCalibrationTarget target,
            CalibrationPartitionRole partition, PlanInputBinding candidates,
            llvm::ArrayRef<ArtifactRootReference> evidence,
            std::uint32_t obligationOrdinal, std::uint32_t gateOrdinal) {
  auto config = projectResolvedEvidenceObligationSetConfigView(
      {EvidenceObligationTemplateRef(obligationOrdinal)});
  if (!config)
    return config.takeError();
  const PromotionAcquisitionDescriptor &descriptor =
      modelParameterCalibrationPromotionAcquisitionDescriptor(target,
                                                              partition);
  return PromotePlanNodeDefinition{
      descriptor.reference(),
      {std::move(candidates), ExactPlanArtifacts{evidence.vec()}},
      config->canonicalViewBytes().vec(),
      config->digest(),
      QualityGatePolicyRef(gateOrdinal),
      AllPassingSelection{},
      partition == CalibrationPartitionRole::HeldOut
          ? PromotePurpose::ModelRelease
          : PromotePurpose::CandidateSelection};
}

} // namespace

llvm::Expected<ResolvedGroundTruthPlan>
buildGroundTruthPlan(ResolvedConfig baseConfig, GroundTruthPlanInputs inputs) {
  if (!inputs.fpa && !inputs.systemRuntime)
    return invalid("at least one calibration track is required");
  if (!baseConfig.dse.modelAuthorizations.empty() ||
      !baseConfig.dse.evidenceObligationTemplates.empty() ||
      !baseConfig.dse.qualityGatePolicies.empty() ||
      !baseConfig.dse.planNodes.empty())
    return invalid("base ResolvedConfig already owns a DSE invocation plan");
  if (inputs.fpa)
    if (llvm::Error error = canonicalizeTrack(*inputs.fpa))
      return std::move(error);
  if (inputs.systemRuntime)
    if (llvm::Error error = canonicalizeTrack(*inputs.systemRuntime))
      return std::move(error);

  if (llvm::Error error = registerFpaGbdtTrainingCandidateGenerator())
    return std::move(error);
  if (llvm::Error error = registerSystemRuntimeGbdtTrainingCandidateGenerator())
    return std::move(error);
  if (llvm::Error error =
          registerModelParameterCalibrationPromotionAcquisitions())
    return std::move(error);

  std::vector<ObligationRecord> obligations;
  const auto appendObligations =
      [&](ModelParameterCalibrationTarget target,
          const GroundTruthModelTrack &track) -> llvm::Error {
    for (CalibrationPartitionRole partition :
         {CalibrationPartitionRole::Validation,
          CalibrationPartitionRole::HeldOut}) {
      auto obligation =
          prepareModelParameterCalibrationEvidenceObligationTemplate(
              target, partition, baseConfig);
      if (!obligation)
        return obligation.takeError();
      obligations.push_back({target, partition, std::move(*obligation)});
    }
    return llvm::Error::success();
  };
  if (inputs.fpa)
    if (llvm::Error error = appendObligations(
            ModelParameterCalibrationTarget::Fpa, *inputs.fpa))
      return std::move(error);
  if (inputs.systemRuntime)
    if (llvm::Error error =
            appendObligations(ModelParameterCalibrationTarget::SystemRuntime,
                              *inputs.systemRuntime))
      return std::move(error);
  llvm::sort(obligations, [](const ObligationRecord &lhs,
                             const ObligationRecord &rhs) {
    return std::lexicographical_compare(lhs.obligation.canonicalBytes().begin(),
                                        lhs.obligation.canonicalBytes().end(),
                                        rhs.obligation.canonicalBytes().begin(),
                                        rhs.obligation.canonicalBytes().end());
  });

  std::vector<GateRecord> gates;
  const auto appendGates =
      [&](ModelParameterCalibrationTarget target,
          const GroundTruthModelTrack &track) -> llvm::Error {
    for (auto [partition, threshold] :
         {std::pair{CalibrationPartitionRole::Validation,
                    track.maximumValidationError},
          std::pair{CalibrationPartitionRole::HeldOut,
                    track.maximumHeldOutError}}) {
      auto gate = makeGate(findObligation(obligations, target, partition),
                           target, threshold);
      if (!gate)
        return gate.takeError();
      gates.push_back({target, partition, std::move(*gate)});
    }
    return llvm::Error::success();
  };
  if (inputs.fpa)
    if (llvm::Error error =
            appendGates(ModelParameterCalibrationTarget::Fpa, *inputs.fpa))
      return std::move(error);
  if (inputs.systemRuntime)
    if (llvm::Error error =
            appendGates(ModelParameterCalibrationTarget::SystemRuntime,
                        *inputs.systemRuntime))
      return std::move(error);
  llvm::sort(gates, [](const GateRecord &lhs, const GateRecord &rhs) {
    return canonicalQualityGatePolicyBytes(lhs.gate) <
           canonicalQualityGatePolicyBytes(rhs.gate);
  });

  for (const ObligationRecord &record : obligations)
    baseConfig.dse.evidenceObligationTemplates.push_back(record.obligation);
  for (const GateRecord &record : gates)
    baseConfig.dse.qualityGatePolicies.push_back(record.gate);
  if (inputs.fpa)
    baseConfig.dse.modelAuthorizations.push_back(
        {llvm::cantFail(evaluation::builtinEvaluationModelDescriptorRef(
            evaluation::BuiltinEvaluationModel::
                FpaModelParameterCalibration))});
  if (inputs.systemRuntime)
    baseConfig.dse.modelAuthorizations.push_back(
        {llvm::cantFail(evaluation::builtinEvaluationModelDescriptorRef(
            evaluation::BuiltinEvaluationModel::
                SystemRuntimeModelParameterCalibration))});

  std::optional<GroundTruthTrackOutputs> fpaOutputs;
  std::optional<GroundTruthTrackOutputs> runtimeOutputs;
  const auto appendTrack =
      [&](ModelParameterCalibrationTarget target,
          const GroundTruthModelTrack &track,
          std::optional<GroundTruthTrackOutputs> &outputs) -> llvm::Error {
    const std::uint64_t trainerNode = baseConfig.dse.planNodes.size();
    auto trainer = makeTrainer(target, track);
    if (!trainer)
      return trainer.takeError();
    baseConfig.dse.planNodes.push_back(std::move(*trainer));

    const std::uint32_t validationObligation = findObligation(
        obligations, target, CalibrationPartitionRole::Validation);
    const std::uint32_t validationGate =
        findGate(gates, target, CalibrationPartitionRole::Validation);
    const std::uint64_t validationNode = baseConfig.dse.planNodes.size();
    auto validation =
        makePromote(target, CalibrationPartitionRole::Validation,
                    PlanOutputRef{trainerNode, 0}, track.evidence.validation,
                    validationObligation, validationGate);
    if (!validation)
      return validation.takeError();
    baseConfig.dse.planNodes.push_back(std::move(*validation));

    const std::uint32_t heldOutObligation =
        findObligation(obligations, target, CalibrationPartitionRole::HeldOut);
    const std::uint32_t heldOutGate =
        findGate(gates, target, CalibrationPartitionRole::HeldOut);
    const std::uint64_t heldOutNode = baseConfig.dse.planNodes.size();
    auto heldOut =
        makePromote(target, CalibrationPartitionRole::HeldOut,
                    PlanOutputRef{validationNode, 0}, track.evidence.heldOut,
                    heldOutObligation, heldOutGate);
    if (!heldOut)
      return heldOut.takeError();
    baseConfig.dse.planNodes.push_back(std::move(*heldOut));
    outputs = GroundTruthTrackOutputs{{trainerNode, 0},
                                      {validationNode, 1},
                                      {heldOutNode, 0},
                                      {heldOutNode, 1}};
    return llvm::Error::success();
  };
  if (inputs.fpa)
    if (llvm::Error error = appendTrack(ModelParameterCalibrationTarget::Fpa,
                                        *inputs.fpa, fpaOutputs))
      return std::move(error);
  if (inputs.systemRuntime)
    if (llvm::Error error =
            appendTrack(ModelParameterCalibrationTarget::SystemRuntime,
                        *inputs.systemRuntime, runtimeOutputs))
      return std::move(error);

  auto view = projectResolvedDseConfigView(baseConfig);
  if (!view)
    return view.takeError();
  std::vector<ArtifactRootReference> semanticInputs;
  std::vector<ArtifactRootReference> preexistingEvidence;
  const auto collectInputs = [&](const GroundTruthModelTrack &track) {
    if (track.evidence.priorParameterBundle)
      semanticInputs.push_back(*track.evidence.priorParameterBundle);
    preexistingEvidence.insert(preexistingEvidence.end(),
                               track.evidence.training.begin(),
                               track.evidence.training.end());
    preexistingEvidence.insert(preexistingEvidence.end(),
                               track.evidence.validation.begin(),
                               track.evidence.validation.end());
    preexistingEvidence.insert(preexistingEvidence.end(),
                               track.evidence.heldOut.begin(),
                               track.evidence.heldOut.end());
  };
  if (inputs.fpa)
    collectInputs(*inputs.fpa);
  if (inputs.systemRuntime)
    collectInputs(*inputs.systemRuntime);
  canonicalizeRoots(semanticInputs);
  canonicalizeRoots(preexistingEvidence);
  return ResolvedGroundTruthPlan(
      std::move(baseConfig), std::move(*view), std::move(semanticInputs),
      std::move(preexistingEvidence), std::move(fpaOutputs),
      std::move(runtimeOutputs));
}

} // namespace loom::dse
