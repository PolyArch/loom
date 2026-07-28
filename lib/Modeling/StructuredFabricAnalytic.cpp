#include "Evaluation/Models/StructuredFabricAnalytic.h"

#include "AnalyticModelSupport.h"

#include "Common/ArtifactStore.h"
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
#include <optional>
#include <utility>
#include <vector>

namespace loom::evaluation::models {
namespace {

constexpr EvaluationCaseKind kCaseKind(0);
constexpr EvaluationModelKind kModelKind(0);
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

const ScopeFormRef kRuntimeScopeForms[] = {ScopeFormRef(0)};
const MetricCapability kMetricCapabilities[] = {
    {MetricKind::Runtime, kRuntimeScopeForms,
     observationFormMask(ObservationForm::Point)}};
const ModeledPhenomenon kModeledPhenomena[] = {
    ModeledPhenomenon::StructuredProgram, ModeledPhenomenon::SpatialResources};
const EvaluationModelDescriptor kModelDescriptor{
    kModelKind,
    "structured_fabric_static_pressure",
    "loom.structured_fabric.static_pressure.v1",
    caseSignatureRef(),
    {},
    kMetricCapabilities,
    {},
    {},
    {},
    detail::emptyStaticPressureConfigView(),
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

llvm::Expected<std::optional<MetricResult>>
estimateRuntimePicoseconds(const frontend::StructuredProgramCandidate &program,
                           const fabric::FinalizedFabricRoot &fabricRoot) {
  const StaticWorkload workload = projectStaticWorkload(program.module());

  std::uint64_t spatialPressure = 0;
  if (workload.hasSpatialRegion) {
    auto dataflowProgram =
        lowering::lowerStructuredProgramToCanonicalDataflow(program);
    if (!dataflowProgram)
      return dataflowProgram.takeError();
    auto view = dataflowProgram->view();
    if (!view)
      return view.takeError();

    auto pressure = detail::canonicalDataflowStaticPressure(*view, fabricRoot);
    if (!pressure)
      return pressure.takeError();
    if (!*pressure)
      return std::optional<MetricResult>{};
    spatialPressure = **pressure;
  }

  auto metric = detail::staticPressureRuntimeMetric(workload.instructionLeaves,
                                                    spatialPressure);
  if (!metric)
    return metric.takeError();
  return std::optional<MetricResult>(std::move(*metric));
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

  auto program =
      frontend::importStructuredProgram(structured.front(), artifactStore);
  if (!program)
    return program.takeError();
  auto fabricRoot =
      fabric::importEntireFabricRoot(fabric.front(), artifactStore);
  if (!fabricRoot)
    return fabricRoot.takeError();
  auto runtime = estimateRuntimePicoseconds(*program, *fabricRoot);
  if (!runtime)
    return runtime.takeError();
  if (!*runtime)
    return EvaluationModelResult{
        {}, UnsupportedEvidence{OutcomeReason::RuntimeCapabilityUnavailable}};

  std::vector<MetricResult> metricResults;
  metricResults.reserve(request.metricRequests().size());
  for (const MetricRequest &metric : request.metricRequests()) {
    if (metric.query().metric != MetricKind::Runtime)
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "structured_fabric_model_invalid: unsupported metric reached "
          "provider");
    metricResults.push_back(**runtime);
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
prepareStructuredFabricRuntimeEvaluation(
    const ArtifactRootReference &structuredProgram,
    const ArtifactRootReference &fabricReference, const ResolvedConfig &config,
    const ArtifactStore &artifactStore) {
  if (llvm::Error error = registerStructuredFabricAnalyticModel())
    return std::move(error);

  auto program =
      frontend::importStructuredProgram(structuredProgram, artifactStore);
  if (!program)
    return program.takeError();
  auto fabricRoot =
      fabric::importEntireFabricRoot(fabricReference, artifactStore);
  if (!fabricRoot)
    return fabricRoot.takeError();

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
  auto metric = MetricRequest::get(
      MetricQuery{MetricKind::Runtime, EvaluationScope{ScopeFormRef(0), {}}},
      {}, *evaluationCase, *resolution, artifactStore);
  if (!metric)
    return metric.takeError();
  auto modelBinding =
      ResolvedModelBinding::project(kModelDescriptor.reference(), {}, config);
  if (!modelBinding)
    return modelBinding.takeError();
  auto request = EvaluationRequest::get(*evaluationCase, {*metric}, {},
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

} // namespace loom::evaluation::models
