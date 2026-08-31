#include "DSE/SpatialTransportCegar.h"

#include "DSE/SpatialRuntimeFeedback.h"

#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "Config/ResolvedConfig.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Evaluation/Evidence.h"
#include "Evaluation/Models/CgraSimulation.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Mapping/Artifact/MappingArtifact.h"
#include "Mapping/Artifact/MappingConstraintSet.h"
#include "PnR/DeterministicSearchProtocol.h"
#include "PnR/PnrConfig.h"
#include "PnR/SpatialMappingMaterializer.h"
#include "PnR/SpatialPnrProblem.h"

#include "llvm/ADT/STLExtras.h"

#include <set>
#include <memory>
#include <system_error>

namespace loom::dse {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "spatial_transport_cegar_invalid: " + message);
}

bool deadlineReached(const SpatialTransportCegarPolicy &policy) {
  return policy.deadline && std::chrono::steady_clock::now() >= *policy.deadline;
}

bool cegarStopRequested(const void *context) {
  return deadlineReached(
      *static_cast<const SpatialTransportCegarPolicy *>(context));
}

std::optional<std::chrono::steady_clock::duration>
cegarRemainingTime(const void *context) {
  const auto &policy =
      *static_cast<const SpatialTransportCegarPolicy *>(context);
  if (!policy.deadline)
    return std::nullopt;
  const auto now = std::chrono::steady_clock::now();
  if (now >= *policy.deadline)
    return std::chrono::steady_clock::duration::zero();
  return *policy.deadline - now;
}

llvm::Expected<pnr::PnrIndex> findPromotedClause(
    const mapping::SpatialMappingConstraintSetView &constraints,
    const SpatialTransportRuntimeFeedback &feedback) {
  if (!feedback.parentSpatialMapping || !feedback.runtimeEvidence ||
      !feedback.evaluationRequest || !feedback.runtimeExecution ||
      !feedback.certificateDigest)
    return invalid("exact feedback has incomplete durable lineage");
  std::optional<pnr::PnrIndex> result;
  for (auto indexed : llvm::enumerate(constraints.clauses())) {
    const auto *noGood =
        std::get_if<mapping::SpatialRuntimeCounterexampleNoGoodView>(
            &indexed.value());
    if (!noGood || !noGood->lineage)
      continue;
    const auto &lineage = *noGood->lineage;
    if (lineage.parentMapping != *feedback.parentSpatialMapping ||
        lineage.runtimeEvidence != *feedback.runtimeEvidence ||
        lineage.evaluationRequest != *feedback.evaluationRequest ||
        lineage.runtimeExecution != *feedback.runtimeExecution ||
        lineage.certificateDigest != feedback.certificateDigest->value())
      continue;
    if (result)
      return invalid("promoted runtime lineage resolves to multiple clauses");
    result = static_cast<pnr::PnrIndex>(indexed.index());
  }
  if (!result)
    return invalid("promoted runtime lineage resolves to no exact clause");
  return *result;
}

std::uint64_t noGoodCount(
    const mapping::SpatialMappingConstraintSetView &constraints) {
  return llvm::count_if(constraints.clauses(), [](const auto &clause) {
    return std::holds_alternative<
        mapping::SpatialRuntimeCounterexampleNoGoodView>(clause);
  });
}

} // namespace

llvm::StringRef spatialTransportCegarTerminationSpelling(
    SpatialTransportCegarTermination termination) {
  switch (termination) {
  case SpatialTransportCegarTermination::Retired:
    return "retired";
  case SpatialTransportCegarTermination::ProofNotEstablished:
    return "proof_not_established";
  case SpatialTransportCegarTermination::NoProgress:
    return "no_progress";
  case SpatialTransportCegarTermination::RepeatedCertificate:
    return "repeated_certificate";
  case SpatialTransportCegarTermination::RepairTerminal:
    return "repair_terminal";
  case SpatialTransportCegarTermination::RuntimeIncomplete:
    return "runtime_incomplete";
  case SpatialTransportCegarTermination::IterationBudgetExhausted:
    return "iteration_budget_exhausted";
  case SpatialTransportCegarTermination::ClauseBudgetExhausted:
    return "clause_budget_exhausted";
  case SpatialTransportCegarTermination::TimedOut:
    return "timed_out";
  }
  llvm_unreachable("unknown Spatial transport CEGAR termination");
}

llvm::Expected<SpatialTransportCegarResult>
executeSpatialTransportCegar(
    const ArtifactRootReference &parentMapping,
    const ArtifactRootReference &parentConstraints,
    const evaluation::models::VerifiedCgraClosedWaitEvidence &parentEvidence,
    const ResolvedConfig &config,
    const fabric::FabricPhysicalTimingProfileView &physicalTiming,
    const SpatialTransportCegarPolicy &policy,
    const ArtifactStore &artifacts, const BlobStore &blobs) {
  if (policy.maximumIterations == 0 ||
      policy.maximumAccumulatedClauses == 0 ||
      policy.maximumSolverCallsPerIteration == 0 ||
      policy.maximumRuntimeEventFramesPerIteration == 0)
    return invalid("every CEGAR budget must be positive");
  if (parentEvidence.certificate().owners.spatialMapping != parentMapping)
    return invalid("initial Evidence does not own the parent Mapping");

  const sim::CgraExecutionOwnerReferences &owners =
      parentEvidence.certificate().owners;
  auto dataflow = dataflow::importCanonicalDataflow(owners.dataflow, artifacts);
  if (!dataflow)
    return dataflow.takeError();
  auto dataflowView = dataflow->view();
  if (!dataflowView)
    return dataflowView.takeError();
  auto tech = mapping::importTechMapping(owners.techMapping, artifacts);
  if (!tech)
    return tech.takeError();
  auto fabricArtifact =
      fabric::importEntireFabricRoot(owners.fabric, artifacts);
  if (!fabricArtifact)
    return fabricArtifact.takeError();
  if (llvm::Error error = fabric::validateFabricPhysicalTimingProfile(
          fabricArtifact->view(), physicalTiming))
    return error;
  auto spatialConfig = pnr::projectResolvedSpatialPnrConfigView(config);
  if (!spatialConfig)
    return spatialConfig.takeError();

  SpatialTransportCegarResult result;
  ArtifactRootReference currentMapping = parentMapping;
  ArtifactRootReference currentConstraints = parentConstraints;
  auto importedInitialConstraints =
      mapping::importSpatialMappingConstraintSet(parentConstraints, artifacts);
  if (!importedInitialConstraints)
    return importedInitialConstraints.takeError();
  auto currentConstraintSet =
      std::make_shared<const mapping::FinalizedSpatialMappingConstraintSet>(
          std::move(*importedInitialConstraints));
  evaluation::models::VerifiedCgraClosedWaitEvidence currentEvidence =
      parentEvidence;
  std::set<ComponentViewDigest::Storage> observedCertificates;
  observedCertificates.insert(
      currentEvidence.certificateDigest().value().bytes());

  for (std::uint64_t iteration = 0; iteration < policy.maximumIterations;
       ++iteration) {
    if (deadlineReached(policy)) {
      result.termination = SpatialTransportCegarTermination::TimedOut;
      return result;
    }
    ExecutionResourceTracker promotionTracker;
    auto feedback = deriveSpatialTransportRuntimeFeedback(
        currentMapping, *currentConstraintSet, currentEvidence, artifacts);
    if (!feedback)
      return feedback.takeError();
    if (feedback->disposition !=
            SpatialTransportRuntimeFeedbackDisposition::Exact ||
        !feedback->constraintSet || !feedback->importedConstraintSet ||
        !feedback->runtimeExecution ||
        !feedback->evaluationRequest || !feedback->certificateDigest) {
      result.termination =
          SpatialTransportCegarTermination::ProofNotEstablished;
      return result;
    }
    if (*feedback->constraintSet == currentConstraints) {
      result.termination =
          SpatialTransportCegarTermination::RepeatedCertificate;
      result.finalMapping = currentMapping;
      result.finalConstraints = currentConstraints;
      result.finalEvidence = currentEvidence.evidence();
      return result;
    }
    const auto &constraints = *feedback->importedConstraintSet;
    if (constraints.reference() != *feedback->constraintSet)
      return invalid("incremental promoted constraint cache has foreign identity");
    if (noGoodCount(constraints.view()) >
        policy.maximumAccumulatedClauses) {
      result.termination =
          SpatialTransportCegarTermination::ClauseBudgetExhausted;
      result.finalConstraints = feedback->constraintSet;
      return result;
    }
    auto clauseOrdinal = findPromotedClause(constraints.view(), *feedback);
    if (!clauseOrdinal)
      return clauseOrdinal.takeError();
    const ExecutionResourceStatistics promotionWork =
        promotionTracker.observe();

    ExecutionResourceTracker freezeTracker;
    auto problem = pnr::freezeSpatialPnrProblem(
        *dataflowView, tech->view(), fabricArtifact->view(), physicalTiming,
        *spatialConfig, constraints.view());
    if (!problem)
      return problem.takeError();
    const ExecutionResourceStatistics freezeWork = freezeTracker.observe();

    ExecutionResourceTracker warmSeedTracker;
    auto parent = mapping::importSpatialMapping(currentMapping, artifacts);
    if (!parent)
      return parent.takeError();
    auto warm =
        pnr::projectFinalizedSpatialMappingWarmSeed(*parent, *problem);
    if (!warm)
      return warm.takeError();
    auto candidate = warm->materializeCandidate();
    if (!candidate)
      return candidate.takeError();
    if (!(*candidate)->runtimeCounterexampleClauseViolated(*clauseOrdinal))
      return invalid("warm parent does not violate its promoted clause");
    const ExecutionResourceStatistics warmSeedWork =
        warmSeedTracker.observe();

    ExecutionResourceTracker repairTracker;
    pnr::SpatialExactRepairScratch repair;
    pnr::DeterministicPnrRandomStream repairStream =
        pnr::DeterministicPnrRandomStream::create(
            spatialConfig->policy().determinism.masterSeed, iteration,
            pnr::PnrRandomStreamPurpose::ExactRepair);
    auto repaired = repair.repair(
        **candidate, iteration, policy.maximumSolverCallsPerIteration,
        repairStream, {}, *clauseOrdinal,
        ExecutionControlView{&policy, cegarStopRequested,
                             cegarRemainingTime});
    if (!repaired)
      return repaired.takeError();
    const ExecutionResourceStatistics repairWork = repairTracker.observe();

    auto structure =
        sim::digestCgraClosedWaitStructure(currentEvidence.certificate());
    if (!structure)
      return structure.takeError();
    SpatialTransportCegarIteration record{
        currentMapping,
        currentEvidence.evidence(),
        currentEvidence.execution(),
        currentEvidence.request(),
        currentEvidence.certificateDigest().value(),
        structure->value(),
        *feedback->constraintSet,
        warm->accounting(),
        *repaired,
        std::nullopt,
        std::nullopt,
        false,
        {promotionWork, freezeWork, warmSeedWork, repairWork, {}, {}, {}}};
    if (repaired->kind != pnr::SpatialExactRepairResultKind::Repaired) {
      result.iterations.push_back(std::move(record));
      if (repaired->kind == pnr::SpatialExactRepairResultKind::TimedOut)
        result.termination = SpatialTransportCegarTermination::TimedOut;
      else if (repaired->kind ==
               pnr::SpatialExactRepairResultKind::ProofNotEstablished)
        result.termination =
            SpatialTransportCegarTermination::ProofNotEstablished;
      else
        result.termination = SpatialTransportCegarTermination::RepairTerminal;
      result.finalConstraints = feedback->constraintSet;
      return result;
    }
    if ((*candidate)->runtimeCounterexampleViolation() != 0)
      return invalid("repaired candidate retains a no-good violation");
    if (llvm::Error error = (*candidate)->verify())
      return error;
    ExecutionResourceTracker finalizationTracker;
    auto child = pnr::finalizeSpatialMappingCandidate(
        **candidate, *dataflowView, tech->view(), fabricArtifact->view(),
        constraints.view(), artifacts);
    if (!child)
      return child.takeError();
    if (child->reference() == currentMapping) {
      result.iterations.push_back(std::move(record));
      result.termination = SpatialTransportCegarTermination::NoProgress;
      result.finalConstraints = feedback->constraintSet;
      return result;
    }
    if (llvm::Error error = mapping::admitSpatialMappingConstraints(
            *dataflowView, tech->view(), fabricArtifact->view(),
            constraints.view(), child->view()))
      return error;
    record.work.childFinalization = finalizationTracker.observe();
    record.childMapping = child->reference();

    ExecutionResourceTracker runtimeTracker;
    auto prepared = evaluation::models::prepareCgraSimulationEvaluation(
        owners.dataflow, owners.fabric, child->reference(),
        currentEvidence.workload(), currentEvidence.runtimeInput(), config,
        artifacts, blobs);
    if (!prepared)
      return prepared.takeError();
    auto evaluated = evaluation::models::evaluateCgraSimulationWithDiagnostics(
        *prepared,
        {policy.maximumRuntimeEventFramesPerIteration, policy.deadline},
        artifacts, blobs);
    if (!evaluated)
      return evaluated.takeError();
    auto evidenceReference =
        evaluation::publishEvaluationEvidence(evaluated->evidence, artifacts);
    if (!evidenceReference)
      return evidenceReference.takeError();
    record.work.runtimeEvaluation = runtimeTracker.observe();
    record.childEvidence = *evidenceReference;
    result.finalMapping = child->reference();
    result.finalConstraints = feedback->constraintSet;
    result.finalEvidence = *evidenceReference;

    if (evaluated->evidence.outcomeKind() !=
        evaluation::EvidenceOutcomeKind::Completed) {
      result.iterations.push_back(std::move(record));
      result.termination = deadlineReached(policy)
                               ? SpatialTransportCegarTermination::TimedOut
                               : SpatialTransportCegarTermination::
                                     RuntimeIncomplete;
      return result;
    }
    ExecutionResourceTracker verificationTracker;
    if (!evaluated->closedWait) {
      auto terminal = evaluation::models::classifyCompletedCgraSimulationEvidence(
          evaluated->evidence, prepared->resolution, artifacts, blobs);
      if (!terminal)
        return terminal.takeError();
      if (*terminal !=
          evaluation::models::CgraSimulationEvidenceTerminal::Retired)
        return invalid("Completed child has no retained wait or retirement");
      record.work.evidenceVerification = verificationTracker.observe();
      record.retired = true;
      result.iterations.push_back(std::move(record));
      result.termination = SpatialTransportCegarTermination::Retired;
      return result;
    }

    auto verified =
        evaluation::models::importVerifiedCgraClosedWaitEvidence(
            *evidenceReference, artifacts, blobs);
    if (!verified)
      return verified.takeError();
    record.work.evidenceVerification = verificationTracker.observe();
    const auto [position, inserted] =
        observedCertificates.insert(
            verified->certificateDigest().value().bytes());
    (void)position;
    result.iterations.push_back(std::move(record));
    if (!inserted) {
      result.termination =
          SpatialTransportCegarTermination::RepeatedCertificate;
      return result;
    }
    currentMapping = child->reference();
    currentConstraints = *feedback->constraintSet;
    currentConstraintSet = feedback->importedConstraintSet;
    currentEvidence = std::move(*verified);
  }

  result.termination =
      SpatialTransportCegarTermination::IterationBudgetExhausted;
  return result;
}

} // namespace loom::dse
