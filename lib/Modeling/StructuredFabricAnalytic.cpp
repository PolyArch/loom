#include "Evaluation/Models/StructuredFabricAnalytic.h"

#include "AnalyticModelSupport.h"

#include "Common/ArtifactLocalReference.h"
#include "Common/ArtifactStore.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Evaluation/ModelProvider.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Artifact/FabricArtifactCodec.h"
#include "Frontend/IR/LoomOps.h"
#include "Frontend/IR/StructuredProgramArtifact.h"
#include "Frontend/Lowering/CanonicalDataflowLowering.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <map>
#include <mutex>
#include <optional>
#include <utility>
#include <vector>

namespace loom::evaluation::models {
namespace {

constexpr EvaluationCaseKind kCaseKind(0);
constexpr EvaluationModelKind kModelKind(2);
constexpr CaseSubjectRoleRef kStructuredProgramRole(0);
constexpr CaseSubjectRoleRef kFabricRole(1);

EvaluationCaseSignatureRef caseSignatureRef() {
  return llvm::cantFail(
      EvaluationCaseSignatureRef::get(evaluationSchemaVersion(), kCaseKind));
}

const ArtifactSchemaDescriptor *const kStructuredSchemas[] = {
    &frontend::structuredProgramArtifactSchema};
const ArtifactSchemaDescriptor *const kFabricSchemas[] = {
    &fabric::fabricArtifactSchema};

const CaseSubjectRoleDescriptor kSubjectRoles[] = {
    {kStructuredProgramRole, "structured_program",
     SubjectRoleCardinality::ExactlyOne, kStructuredSchemas, nullptr},
    {kFabricRole, "fabric", SubjectRoleCardinality::ExactlyOne, kFabricSchemas,
     nullptr},
};

const EvaluationCaseSignatureDescriptor kCaseSignature{
    kCaseKind,
    "structured_program_with_fabric",
    "One exact Structured Program evaluated against one exact Fabric.",
    kSubjectRoles,
    ArtifactRequirement::Forbidden,
    {},
    ArtifactRequirement::Forbidden,
    {},
    nullptr,
    AbsentReferenceCycle{},
    {}};

const ScopeFormRef kWholeCaseScopeForms[] = {ScopeFormRef(0)};
const MetricCapability kMetricCapabilities[] = {
    {MetricKind::Runtime, kWholeCaseScopeForms,
     observationFormMask(ObservationForm::Point)},
    {MetricKind::LimitingClockFrequency, kWholeCaseScopeForms,
     observationFormMask(ObservationForm::Point)},
    {MetricKind::TotalArea, kWholeCaseScopeForms,
     observationFormMask(ObservationForm::Point)},
    {MetricKind::DynamicPower, kWholeCaseScopeForms,
     observationFormMask(ObservationForm::Point)},
    {MetricKind::LeakagePower, kWholeCaseScopeForms,
     observationFormMask(ObservationForm::Point)}};
const ModeledPhenomenon kModeledPhenomena[] = {
    ModeledPhenomenon::StructuredProgram, ModeledPhenomenon::SpatialResources};
const EvaluationModelDescriptor kModelDescriptor{
    kModelKind,
    "structured_fabric_low_confidence",
    "loom.structured_fabric.low_confidence.v1",
    caseSignatureRef(),
    {},
    kMetricCapabilities,
    {},
    {},
    {},
    detail::emptyLowConfidenceConfigView(),
    kModeledPhenomena,
    EvaluationExecutionMethod::Analytic,
    {},
    DeterminismContract::Deterministic,
    {}};

bool isInsideSpatialRegion(mlir::Operation *operation) {
  return static_cast<bool>(operation->getParentOfType<loom::SpatialRegionOp>());
}

bool isInsideGlobal(mlir::Operation *operation) {
  return static_cast<bool>(operation->getParentOfType<mlir::LLVM::GlobalOp>());
}

bool isExecutableLeaf(mlir::Operation *operation) {
  if (operation->getNumRegions() != 0 ||
      operation->hasTrait<mlir::OpTrait::IsTerminator>() ||
      mlir::isa<mlir::SymbolOpInterface>(operation) ||
      isInsideGlobal(operation))
    return false;
  return true;
}

struct StaticWorkload final {
  std::uint64_t instructionLeaves = 0;
  bool hasSpatialRegion = false;
};

using CachedMetrics = std::optional<detail::LowConfidenceMetricSet>;

std::map<std::vector<std::uint8_t>, CachedMetrics> &metricCache() {
  static std::map<std::vector<std::uint8_t>, CachedMetrics> cache;
  return cache;
}

std::mutex &metricCacheMutex() {
  static std::mutex mutex;
  return mutex;
}

std::vector<std::uint8_t>
metricCacheKey(const ArtifactRootReference &structuredProgram,
               const ArtifactRootReference &fabricReference,
               const ComponentViewDigest &configDigest) {
  std::vector<std::uint8_t> key =
      encodeArtifactRootReference(structuredProgram);
  std::vector<std::uint8_t> fabricBytes =
      encodeArtifactRootReference(fabricReference);
  key.insert(key.end(), fabricBytes.begin(), fabricBytes.end());
  key.insert(key.end(), configDigest.bytes().begin(),
             configDigest.bytes().end());
  return key;
}

StaticWorkload projectStaticWorkload(mlir::ModuleOp module) {
  StaticWorkload workload;
  module.walk([&](mlir::Operation *operation) {
    if (mlir::isa<loom::SpatialRegionOp>(operation)) {
      workload.hasSpatialRegion = true;
      return;
    }
    if (!isExecutableLeaf(operation))
      return;
    if (!isInsideSpatialRegion(operation))
      ++workload.instructionLeaves;
  });
  return workload;
}

llvm::Expected<std::optional<detail::LowConfidenceMetricSet>> estimateMetrics(
    const frontend::StructuredProgramCandidate &program,
    const fabric::FinalizedFabricRoot &fabricRoot,
    const dataflow::CanonicalDataflowProgramView *projectedDataflow = nullptr) {
  const StaticWorkload workload = projectStaticWorkload(program.module());

  detail::AnalyticWorkloadEstimate pressure;
  if (workload.hasSpatialRegion) {
    std::optional<dataflow::CanonicalDataflowArtifact> lowered;
    std::optional<dataflow::CanonicalDataflowProgramView> loweredView;
    if (!projectedDataflow) {
      auto dataflowProgram =
          lowering::lowerStructuredProgramToCanonicalDataflow(program);
      if (!dataflowProgram)
        return dataflowProgram.takeError();
      lowered.emplace(std::move(*dataflowProgram));
      auto view = lowered->view();
      if (!view)
        return view.takeError();
      loweredView.emplace(std::move(*view));
      projectedDataflow = &*loweredView;
    }

    auto projected = detail::projectCanonicalDataflowWorkload(
        *projectedDataflow, fabricRoot);
    if (!projected)
      return projected.takeError();
    if (!*projected)
      return std::optional<detail::LowConfidenceMetricSet>{};
    pressure = **projected;
  }

  auto metrics = detail::estimateLowConfidenceMetrics(
      workload.instructionLeaves, pressure, fabricRoot);
  if (!metrics)
    return metrics.takeError();
  return std::optional<detail::LowConfidenceMetricSet>(std::move(*metrics));
}

llvm::Expected<EvaluationModelResult>
evaluate(const EvaluationRequest &request, const CaseArtifactResolution &,
         const ArtifactStore &artifactStore) {
  llvm::ArrayRef<ArtifactRootReference> structured =
      request.subjectBindings().subjects(kStructuredProgramRole);
  llvm::ArrayRef<ArtifactRootReference> fabric =
      request.subjectBindings().subjects(kFabricRole);
  if (structured.size() != 1 || fabric.size() != 1)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "structured_fabric_model_invalid: exact subjects are not total");

  const std::vector<std::uint8_t> cacheKey =
      metricCacheKey(structured.front(), fabric.front(),
                     request.modelBinding().resolvedModelConfig().digest());
  bool cacheHit = false;
  CachedMetrics cachedMetrics;
  {
    std::lock_guard<std::mutex> lock(metricCacheMutex());
    auto found = metricCache().find(cacheKey);
    if (found != metricCache().end()) {
      cacheHit = true;
      cachedMetrics = found->second;
    }
  }

  CachedMetrics metrics;
  if (cacheHit) {
    if (auto stored = artifactStore.get(structured.front()); !stored)
      return stored.takeError();
    if (auto stored = artifactStore.get(fabric.front()); !stored)
      return stored.takeError();
    metrics = std::move(cachedMetrics);
  } else {
    auto program =
        frontend::importStructuredProgram(structured.front(), artifactStore);
    if (!program)
      return program.takeError();
    auto fabricRoot =
        fabric::importEntireFabricRoot(fabric.front(), artifactStore);
    if (!fabricRoot)
      return fabricRoot.takeError();
    auto computed = estimateMetrics(*program, *fabricRoot);
    if (!computed)
      return computed.takeError();
    metrics = std::move(*computed);
    std::lock_guard<std::mutex> lock(metricCacheMutex());
    metricCache().try_emplace(cacheKey, metrics);
  }
  if (!metrics)
    return EvaluationModelResult{
        {}, UnsupportedEvidence{OutcomeReason::RuntimeCapabilityUnavailable}};

  std::vector<MetricResult> metricResults;
  metricResults.reserve(request.metricRequests().size());
  for (const MetricRequest &metric : request.metricRequests()) {
    auto result = metrics->result(metric.query().metric);
    if (!result)
      return result.takeError();
    metricResults.push_back(std::move(*result));
  }
  return EvaluationModelResult{{},
                               CompletedEvidence{std::move(metricResults), {}}};
}

const EvaluationModelProvider kProvider{kModelDescriptor.reference(),
                                        &evaluate};

} // namespace

llvm::Error registerStructuredFabricAnalyticModel() {
  if (llvm::Error error = registerEvaluationCaseSignature(kCaseSignature))
    return error;
  if (llvm::Error error = registerEvaluationModelDescriptor(kModelDescriptor))
    return error;
  return registerEvaluationModelProvider(kProvider);
}

llvm::Expected<PreparedStructuredFabricEvaluation>
prepareStructuredFabricEvaluation(
    const ArtifactRootReference &structuredProgram,
    const ArtifactRootReference &fabricReference, const ResolvedConfig &config,
    const ArtifactStore &artifactStore) {
  if (llvm::Error error = registerStructuredFabricAnalyticModel())
    return std::move(error);

  auto modelBinding =
      ResolvedModelBinding::project(kModelDescriptor.reference(), {}, config);
  if (!modelBinding)
    return modelBinding.takeError();
  const std::vector<std::uint8_t> cacheKey =
      metricCacheKey(structuredProgram, fabricReference,
                     modelBinding->resolvedModelConfig().digest());
  bool cached = false;
  {
    std::lock_guard<std::mutex> lock(metricCacheMutex());
    cached = metricCache().count(cacheKey) != 0;
  }
  if (cached) {
    if (auto stored = artifactStore.get(structuredProgram); !stored)
      return stored.takeError();
  } else {
    auto program =
        frontend::importStructuredProgram(structuredProgram, artifactStore);
    if (!program)
      return program.takeError();
  }

  auto resolution = detail::resolveSingleSubjectFabricCase(
      structuredProgram, fabricReference, artifactStore);
  if (!resolution)
    return resolution.takeError();

  auto bindings = EvaluationSubjectBindings::get(
      {{kStructuredProgramRole, {structuredProgram}},
       {kFabricRole, {fabricReference}}});
  if (!bindings)
    return bindings.takeError();
  auto evaluationCase = EvaluationCase::get(
      caseSignatureRef(), std::move(*bindings), std::nullopt, std::nullopt, {},
      *resolution, artifactStore);
  if (!evaluationCase)
    return evaluationCase.takeError();
  std::vector<MetricRequest> metrics;
  metrics.reserve(std::size(kMetricCapabilities));
  for (const MetricCapability &capability : kMetricCapabilities) {
    auto metric = MetricRequest::get(
        MetricQuery{capability.kind, EvaluationScope{ScopeFormRef(0), {}}}, {},
        *evaluationCase, *resolution, artifactStore);
    if (!metric)
      return metric.takeError();
    metrics.push_back(std::move(*metric));
  }
  auto request = EvaluationRequest::get(*evaluationCase, metrics, {},
                                        std::move(*modelBinding), 0,
                                        *resolution, artifactStore);
  if (!request)
    return request.takeError();
  auto published = publishEvaluationRequest(*request, artifactStore);
  if (!published)
    return published.takeError();
  return PreparedStructuredFabricEvaluation{
      std::move(*request), std::move(*resolution), kStructuredProgramRole};
}

llvm::Error primeStructuredFabricAnalyticResult(
    const ArtifactRootReference &structuredProgramReference,
    const frontend::StructuredProgramCandidate &structuredProgram,
    const dataflow::CanonicalDataflowArtifact &canonicalDataflow,
    const fabric::FinalizedFabricRoot &fabricRoot, const ResolvedConfig &config,
    const ArtifactStore &artifactStore) {
  if (llvm::Error error = registerStructuredFabricAnalyticModel())
    return error;
  if (structuredProgramReference.schemaIdentity !=
          frontend::structuredProgramArtifactSchema.identity ||
      structuredProgramReference.schemaVersion !=
          frontend::structuredProgramArtifactSchema.version ||
      structuredProgramReference.artifact != structuredProgram.identity())
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "structured_fabric_model_invalid: cache subject mismatch");
  if (auto stored = artifactStore.get(structuredProgramReference); !stored)
    return stored.takeError();
  if (auto stored = artifactStore.get(fabricRoot.reference()); !stored)
    return stored.takeError();

  auto configView =
      ResolvedModelConfigView::project(kModelDescriptor.reference(), config);
  if (!configView)
    return configView.takeError();
  auto dataflowView = canonicalDataflow.view();
  if (!dataflowView)
    return dataflowView.takeError();
  auto metrics = estimateMetrics(structuredProgram, fabricRoot, &*dataflowView);
  if (!metrics)
    return metrics.takeError();

  const std::vector<std::uint8_t> key = metricCacheKey(
      structuredProgramReference, fabricRoot.reference(), configView->digest());
  std::lock_guard<std::mutex> lock(metricCacheMutex());
  auto [found, inserted] = metricCache().try_emplace(key, *metrics);
  if (!inserted && found->second != *metrics)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "structured_fabric_model_invalid: nondeterministic cached result");
  return llvm::Error::success();
}

} // namespace loom::evaluation::models
