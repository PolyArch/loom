#include "Application/SystemQor.h"

#include "Application/ProductOracleEvaluation.h"
#include "ApplicationSystemRuntimeEvidence.h"
#include "Common/ArtifactText.h"
#include "Simulator/SystemActivity.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/JSON.h"

#include <algorithm>
#include <tuple>
#include <limits>
#include <system_error>

namespace loom::application {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(std::make_error_code(std::errc::invalid_argument),
                                 "application System QoR: " + message);
}

/// Sign of `numerator/denominator - targetNumerator/targetDenominator`, taken
/// without leaving the exact integer domain.
int compareToTarget(std::uint64_t numerator, std::uint64_t denominator,
                    std::uint64_t targetNumerator,
                    std::uint64_t targetDenominator) {
  constexpr unsigned width = 128;
  const auto measured = llvm::APInt(width, numerator) * targetDenominator;
  const auto target = llvm::APInt(width, denominator) * targetNumerator;
  return measured.ugt(target) ? 1 : (measured.ult(target) ? -1 : 0);
}

int compareRatios(evaluation::ExactRatio lhs, evaluation::ExactRatio rhs) {
  return compareToTarget(lhs.numerator(), lhs.denominator(), rhs.numerator(),
                         rhs.denominator());
}
evaluation::ExactRatio zeroRatio() {
  return llvm::cantFail(evaluation::ExactRatio::get(0, 1));
}
/// A resource is saturated only strictly above the shared exclusive target.
bool isSaturated(evaluation::ExactRatio value) {
  return compareToTarget(value.numerator(), value.denominator(),
                         applicationMinimumResourceUtilizationNumerator,
                         applicationMinimumResourceUtilizationDenominator) > 0;
}

/// The accelerated window is explained by its invocation phase only while
/// configuration residency stays strictly below the exclusive launch budget.
bool isLaunchBound(evaluation::ExactRatio launchOverhead) {
  return compareToTarget(launchOverhead.numerator(),
                         launchOverhead.denominator(),
                         applicationMaximumLaunchOverheadNumerator,
                         applicationMaximumLaunchOverheadDenominator) >= 0;
}

llvm::Expected<detail::ImportedApplicationSystemRun>
importRun(const FinalizedApplicationRuntimeManifest &manifest,
          const ApplicationSystemRunEvidence &roots,
          const ArtifactRootReference &expectedDeployment,
          const ArtifactRootReference &expectedWorkload,
          const ArtifactRootReference &expectedInput,
          const evaluation::CaseArtifactResolution &resolution,
          const ResolvedConfig &config, const ArtifactStore &artifacts,
          const BlobStore &blobs) {
  auto run = detail::importApplicationSystemRun(roots, expectedDeployment,
                                                expectedWorkload, expectedInput,
                                                resolution, artifacts, blobs);
  if (!run)
    return run.takeError();
  std::optional<ArtifactRootReference> oracleRequest;
  if (manifest.manifest().productOracle()) {
    if (!roots.productOracleEvidence)
      return invalid("product pair member has no independent source oracle Evidence");
    auto prepared = prepareProductOracleEvaluation(manifest, roots.execution, resolution,
                                                    config, artifacts, blobs);
    if (!prepared)
      return prepared.takeError();
    auto oracle = evaluation::importEvaluationEvidence(*roots.productOracleEvidence,
                                                        prepared->resolution, artifacts, blobs);
    if (!oracle)
      return oracle.takeError();
    const auto expected = evaluation::evaluationRequestReference(prepared->request);
    const auto *result = std::get_if<evaluation::CompletedEvidence>(&oracle->outcome());
    if (oracle->requestRef() != expected || !result || result->findingResults.size() != 1 ||
        !std::holds_alternative<evaluation::AbsentFinding>(result->findingResults.front().result))
      return invalid("product pair member lacks the exact passing source oracle");
    oracleRequest = expected;
  } else if (roots.productOracleEvidence) {
    return invalid("non-product pair member carries an unowned product oracle");
  }
  run->measurement.productOracleRequest = std::move(oracleRequest);
  return run;
}

/// All useful host and accelerator work occupies the source-declared interval,
/// while saturation is measured over the invocation phase of the accelerated
/// window inside it.
llvm::Expected<std::optional<ApplicationSystemWindowMeasurement>>
measureCandidateWindow(const sim::CanonicalSimulationExecution &execution,
                       const ApplicationSystemComputeInputs &compute) {
  const auto &interval = execution.system()->computationInterval;
  if (!interval)
    return std::optional<ApplicationSystemWindowMeasurement>{};
  const auto &phases = execution.system()->acceleratedPhases;
  if ((compute.launchedAccCores != 0) != phases.has_value())
    return invalid("measured candidate launches disagree with its accelerated "
                   "window");
  // An unaccelerated computation saturates nothing and carries no launch
  // overhead: both ratios stay zero rather than dividing by an absent phase.
  const std::uint64_t invocationTicks =
      phases ? phases->invocation.elapsedTicks() : 0;
  auto memoryUtilization = evaluation::ExactRatio::get(
      phases ? phases->invocation.occupiedTicks() : 0,
      invocationTicks == 0 ? 1 : invocationTicks);
  if (!memoryUtilization)
    return memoryUtilization.takeError();
  const std::uint64_t acceleratedTicks = phases ? phases->elapsedTicks() : 0;
  auto launchOverhead = evaluation::ExactRatio::get(
      phases ? phases->configurationResidency.elapsedTicks() : 0,
      acceleratedTicks == 0 ? 1 : acceleratedTicks);
  if (!launchOverhead)
    return launchOverhead.takeError();
  if (compute.launchedAccCores != 0 && compute.referenceCycleTicks == 0)
    return invalid("measured candidate has no exact reference cycle");
  const std::uint64_t elapsed = invocationTicks;
  constexpr unsigned width = 128;
  ApplicationSystemComputeMeasurement measurement{compute, {}, zeroRatio(),
                                                  std::nullopt, zeroRatio()};
  measurement.classes.reserve(compute.classes.size());
  for (auto indexed : llvm::enumerate(compute.classes)) {
    const ApplicationSystemComputeClassInputs &cls = indexed.value();
    if (indexed.index() != 0) {
      const auto &prior = compute.classes[indexed.index() - 1];
      if (std::tie(prior.schema, prior.elementBits) >=
          std::tie(cls.schema, cls.elementBits))
        return invalid("candidate compute classes are not sorted and distinct");
    }
    if (compute.launchedAccCores == 0 && cls.retiredElementFirings != 0)
      return invalid("unlaunched computation carries compute firings");
    if (cls.retiredElementFirings != 0 && cls.peakIssueLanesPerCycle == 0)
      return invalid("retired compute class has no admitting FU on the Fabric");
    const llvm::APInt capacity = llvm::APInt(width, elapsed) *
                                 cls.peakIssueLanesPerCycle *
                                 compute.launchedAccCores;
    const llvm::APInt firings =
        llvm::APInt(width, cls.retiredElementFirings) *
        compute.referenceCycleTicks;
    const llvm::APInt slots =
        llvm::APInt(width, cls.placementSlots) * compute.launchedAccCores;
    if (capacity.getActiveBits() > 64 || firings.getActiveBits() > 64 ||
        slots.getActiveBits() > 64)
      return invalid("candidate compute occupancy exceeds the exact ratio domain");
    auto occupancy = evaluation::ExactRatio::get(
        firings.getZExtValue(), capacity.isZero() ? 1 : capacity.getZExtValue());
    if (!occupancy)
      return occupancy.takeError();
    auto placement = evaluation::ExactRatio::get(
        cls.boundRealizations, slots.isZero() ? 1 : slots.getZExtValue());
    if (!placement)
      return placement.takeError();
    if (compareRatios(*occupancy, measurement.occupancy) > 0) {
      measurement.occupancy = *occupancy;
      measurement.bindingClass = indexed.index();
    }
    if (compareRatios(*placement, measurement.placementUtilization) > 0)
      measurement.placementUtilization = *placement;
    measurement.classes.push_back({cls, *occupancy, *placement});
  }
  return std::optional<ApplicationSystemWindowMeasurement>{
      {*interval, phases, *launchOverhead, *memoryUtilization,
       std::move(measurement)}};
}

void writeRoot(llvm::json::OStream &json, llvm::StringRef name,
               const ArtifactRootReference &reference) {
  json.attributeObject(name, [&] { writeArtifactRootReferenceJsonFields(json, reference); });
}

void writeRatio(llvm::json::OStream &json, llvm::StringRef name, evaluation::ExactRatio value) {
  json.attributeObject(name, [&] {
    json.attribute("numerator", value.numerator());
    json.attribute("denominator", value.denominator());
  });
}

/// An absent phase is the empty span at the start of the computation: a
/// computation with no accelerator invocation has no accelerated window.
void writePhase(llvm::json::OStream &json, llvm::StringRef name,
                const sim::SystemAcceleratedPhase *phase) {
  json.attributeObject(name, [&] {
    json.attribute("begin_tick", phase ? phase->beginTick : 0);
    json.attribute("end_tick", phase ? phase->endTick : 0);
    json.attribute("elapsed_ticks", phase ? phase->elapsedTicks() : 0);
  });
}

void writeRun(llvm::json::OStream &json,
              const ApplicationSystemRunMeasurement &run,
              const ApplicationSystemWindowMeasurement *candidate = nullptr) {
  writeRoot(json, "request", run.request);
  writeRoot(json, "evidence", run.roots.evidence);
  writeRoot(json, "execution", run.roots.execution);
  if (run.productOracleRequest) {
    writeRoot(json, "product_oracle_request", *run.productOracleRequest);
    writeRoot(json, "product_oracle_evidence", *run.roots.productOracleEvidence);
  }
  json.attribute("elapsed_ticks", run.elapsedTicks);
  json.attributeObject("shared_memory", [&] {
    json.attribute("occupied_ticks", run.memoryActivity.occupiedTicks);
    writeRatio(json, "utilization", run.memoryUtilization);
  });
  if (!run.computationInterval) {
    json.attribute("computation_interval", nullptr);
    return;
  }
  const auto &interval = *run.computationInterval;
  json.attributeObject("computation_interval", [&] {
    json.attribute("begin_tick", interval.beginTick);
    json.attribute("end_tick", interval.endTick);
    json.attribute("elapsed_ticks", interval.elapsedTicks());
    json.attributeObject("shared_memory", [&] {
      json.attribute("occupied_ticks", interval.occupiedTicks());
      writeRatio(json, "utilization",
                 llvm::cantFail(evaluation::ExactRatio::get(
                     interval.occupiedTicks(), interval.elapsedTicks())));
    });
    if (!candidate)
      return;
    json.attributeObject("accelerated_window", [&] {
      writePhase(json, "configuration_residency",
                 candidate->phases ? &candidate->phases->configurationResidency
                                   : nullptr);
      writePhase(json, "invocation",
                 candidate->phases ? &candidate->phases->invocation : nullptr);
      json.attribute("elapsed_ticks", candidate->acceleratedTicks());
      writeRatio(json, "launch_overhead", candidate->launchOverhead);
      json.attributeObject("shared_memory", [&] {
        json.attribute("occupied_ticks",
                       candidate->phases
                           ? candidate->phases->invocation.occupiedTicks()
                           : 0);
        writeRatio(json, "utilization", candidate->memoryUtilization);
      });
    });
    json.attributeObject("compute", [&] {
      const auto &compute = candidate->compute;
      json.attribute("launched_acc_cores", compute.inputs.launchedAccCores);
      json.attribute("reference_cycle_ticks", compute.inputs.referenceCycleTicks);
      json.attributeArray("classes", [&] {
        for (const ApplicationSystemComputeClassMeasurement &cls : compute.classes)
          json.object([&] {
            json.attribute("schema",
                           ::dataflow::operationSchemaSpelling(cls.inputs.schema));
            json.attribute("element_bits", cls.inputs.elementBits);
            json.attribute("retired_element_firings",
                           cls.inputs.retiredElementFirings);
            json.attribute("peak_issue_lanes_per_cycle",
                           cls.inputs.peakIssueLanesPerCycle);
            json.attribute("placement_slots", cls.inputs.placementSlots);
            json.attribute("bound_realizations", cls.inputs.boundRealizations);
            writeRatio(json, "occupancy", cls.occupancy);
            writeRatio(json, "placement_utilization", cls.placementUtilization);
          });
      });
      writeRatio(json, "occupancy", compute.occupancy);
      if (compute.bindingClass)
        json.attribute("binding_class",
                       ::dataflow::operationSchemaSpelling(
                           compute.classes[*compute.bindingClass].inputs.schema));
      else
        json.attribute("binding_class", nullptr);
      writeRatio(json, "placement_utilization", compute.placementUtilization);
    });
  });
}

} // namespace

llvm::StringRef
applicationSystemBottleneckSpelling(ApplicationSystemBottleneck bottleneck) {
  switch (bottleneck) {
  case ApplicationSystemBottleneck::LaunchBound:
    return "launch_bound";
  case ApplicationSystemBottleneck::MemoryBandwidthBound:
    return "memory_bandwidth_bound";
  case ApplicationSystemBottleneck::ComputeBound:
    return "compute_bound";
  case ApplicationSystemBottleneck::HostBound:
    return "host_bound";
  case ApplicationSystemBottleneck::LatencyBound:
    return "latency_bound";
  case ApplicationSystemBottleneck::Unmeasured:
    return "unmeasured";
  }
  llvm_unreachable("closed System bottleneck classification");
}

ApplicationSystemBottleneck ApplicationSystemQor::bottleneck() const {
  if (!window_)
    return ApplicationSystemBottleneck::Unmeasured;
  if (isLaunchBound(window_->launchOverhead))
    return ApplicationSystemBottleneck::LaunchBound;
  if (isSaturated(window_->memoryUtilization))
    return ApplicationSystemBottleneck::MemoryBandwidthBound;
  if (isSaturated(window_->compute.occupancy))
    return ApplicationSystemBottleneck::ComputeBound;
  if (compareToTarget(window_->acceleratedTicks(),
                      window_->window.elapsedTicks(),
                      applicationHostBoundWindowNumerator,
                      applicationHostBoundWindowDenominator) < 0)
    return ApplicationSystemBottleneck::HostBound;
  return ApplicationSystemBottleneck::LatencyBound;
}

ApplicationSystemQorStatus ApplicationSystemQor::status() const {
  if (!speedup_ || !window_)
    return ApplicationSystemQorStatus::Unmeasured;
  if (tier_ == EvaluationTier::Functional)
    return ApplicationSystemQorStatus::Functional;
  const bool saturated = isSaturated(window_->memoryUtilization) ||
                         isSaturated(window_->compute.occupancy);
  return speedup_->numerator() > speedup_->denominator() && saturated &&
                 window_->compute.inputs.launchedAccCores != 0 &&
                 !isLaunchBound(window_->launchOverhead)
             ? ApplicationSystemQorStatus::Qualified
             : ApplicationSystemQorStatus::NotQualified;
}

llvm::Expected<ApplicationSystemQor> qualifyApplicationSystemQor(
    const FinalizedApplicationRuntimeManifest &manifest,
    const ApplicationSystemRunEvidence &hostOnly,
    const evaluation::CaseArtifactResolution &hostResolution,
    const ApplicationSystemRunEvidence &candidate,
    const evaluation::CaseArtifactResolution &candidateResolution,
    const ApplicationSystemComputeInputs &candidateCompute,
    const ResolvedConfig &config, const ArtifactStore &artifacts, const BlobStore &blobs) {
  const auto &runtime = manifest.manifest();
  const auto &baseline = runtime.hostOnlyBaseline();
  auto host = importRun(manifest, hostOnly, baseline.deployment, baseline.inputs.workload,
                        baseline.inputs.runtimeInput, hostResolution, config, artifacts, blobs);
  if (!host)
    return host.takeError();
  auto accelerated = importRun(manifest, candidate, runtime.deployment(), runtime.activationWorkload(),
                               runtime.activationRuntimeInput(), candidateResolution,
                               config, artifacts, blobs);
  if (!accelerated)
    return accelerated.takeError();
  auto compared = detail::compareApplicationSystemRuns(*host, *accelerated);
  if (!compared)
    return compared.takeError();
  if (!*compared)
    return invalid("complete System pair functional observations differ");
  std::optional<evaluation::ExactRatio> speedup;
  if (host->measurement.computationInterval) {
    auto ratio = evaluation::ExactRatio::get(
        host->measurement.computationInterval->elapsedTicks(),
        accelerated->measurement.computationInterval->elapsedTicks());
    if (!ratio)
      return ratio.takeError();
    speedup = *ratio;
  }
  auto window =
      measureCandidateWindow(accelerated->execution, candidateCompute);
  if (!window)
    return window.takeError();
  return ApplicationSystemQor(
      manifest.reference(), runtime.evaluationTier(), host->gem5Binding,
      std::move(host->measurement), std::move(accelerated->measurement),
      std::move(*window), speedup);
}

void writeApplicationSystemQorJsonFields(llvm::json::OStream &json,
                                       const ApplicationSystemQor &qor) {
  json.attribute("schema", applicationSystemQorProjectionSchema);
  json.attribute("version", applicationSystemQorProjectionVersion);
  json.attribute("evaluation_tier", toString(qor.evaluationTier()));
  writeRoot(json, "application_runtime_manifest", qor.runtimeManifest());
  writeRoot(json, "gem5_binding", qor.gem5Binding());
  json.attributeObject("host_only", [&] { writeRun(json, qor.hostOnly()); });
  json.attributeObject("candidate", [&] {
    writeRun(json, qor.candidate(),
             qor.candidateWindow() ? &*qor.candidateWindow() : nullptr);
  });
  if (qor.speedup())
    writeRatio(json, "speedup", *qor.speedup());
  else
    json.attribute("speedup", nullptr);
  switch (qor.status()) {
  case ApplicationSystemQorStatus::Qualified:
    json.attribute("status", "qualified");
    break;
  case ApplicationSystemQorStatus::NotQualified:
    json.attribute("status", "not_qualified");
    break;
  case ApplicationSystemQorStatus::Functional:
    json.attribute("status", "functional");
    break;
  case ApplicationSystemQorStatus::Unmeasured:
    json.attribute("status", "unmeasured");
    break;
  }
  json.attribute("bottleneck", applicationSystemBottleneckSpelling(qor.bottleneck()));
  json.attributeObject("target", [&] {
    json.attribute("strict_speedup", true);
    json.attributeArray("window_branches", [&] {
      json.value("memory_service_utilization");
      json.value("compute_occupancy");
    });
    json.attributeObject("minimum_window_utilization_exclusive", [&] {
      json.attribute("numerator", applicationMinimumResourceUtilizationNumerator);
      json.attribute("denominator", applicationMinimumResourceUtilizationDenominator);
    });
    json.attributeObject("host_bound_window_fraction_exclusive", [&] {
      json.attribute("numerator", applicationHostBoundWindowNumerator);
      json.attribute("denominator", applicationHostBoundWindowDenominator);
    });
    json.attributeObject("maximum_launch_overhead_exclusive", [&] {
      json.attribute("numerator", applicationMaximumLaunchOverheadNumerator);
      json.attribute("denominator", applicationMaximumLaunchOverheadDenominator);
    });
  });
}

} // namespace loom::application
