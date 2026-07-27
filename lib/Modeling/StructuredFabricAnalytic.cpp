#include "Evaluation/Models/StructuredFabricAnalytic.h"

#include "Common/ArtifactStore.h"
#include "Common/ResolvedConfig.h"
#include "Dataflow/IR/OperationSchema.h"
#include "Dataflow/IR/OperationSchemaCodec.h"
#include "Evaluation/ModelProvider.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Artifact/FabricArtifactCodec.h"
#include "Frontend/Compilation/FabricCapabilityIndex.h"
#include "Frontend/IR/LoomOps.h"
#include "Frontend/IR/StructuredProgramArtifact.h"
#include "Frontend/Lowering/CanonicalDataflowLowering.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/Support/CheckedArithmetic.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <optional>
#include <utility>
#include <vector>

namespace loom::evaluation::models {
namespace {

constexpr EvaluationCaseKind kCaseKind(0);
constexpr EvaluationModelKind kModelKind(0);
constexpr CaseSubjectRoleRef kStructuredProgramRole(0);
constexpr CaseSubjectRoleRef kFabricRole(1);

// The exact descriptor identity pins these low-fidelity assumptions. They are
// model parameters, not physical Fabric facts or measured timing.
constexpr std::uint64_t kInstructionLeafPicoseconds = 1000;
constexpr std::uint64_t kSpatialPressurePicoseconds = 250;

struct EmptyConfigView {};

EvaluationCaseSignatureRef caseSignatureRef() {
  return llvm::cantFail(
      EvaluationCaseSignatureRef::get(evaluationSchemaVersion(), kCaseKind));
}

llvm::ArrayRef<std::uint8_t> configSchemaBytes() {
  static constexpr llvm::StringLiteral descriptor =
      "loom.structured_fabric.static_pressure.config.1.0";
  return {reinterpret_cast<const std::uint8_t *>(descriptor.data()),
          descriptor.size()};
}

llvm::Expected<OwnerValue> projectConfig(const ResolvedConfig &) {
  return OwnerValue::get(EmptyConfigView{});
}

llvm::Expected<std::vector<std::uint8_t>>
encodeConfig(const OwnerValue &value) {
  if (!value.getIf<EmptyConfigView>())
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "structured Fabric model config has the "
                                   "wrong owner type");
  return std::vector<std::uint8_t>{};
}

llvm::Expected<OwnerValue> adoptConfig(llvm::ArrayRef<std::uint8_t> bytes,
                                       const ComponentViewDigest &) {
  if (!bytes.empty())
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "structured Fabric model config must be empty");
  return OwnerValue::get(EmptyConfigView{});
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
const ResolvedModelConfigViewContract kConfigView{
    configSchemaBytes(), &projectConfig, &encodeConfig, &adoptConfig};

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
    kConfigView,
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

struct ActorDemand final {
  mlir::Operation *representative = nullptr;
  std::uint64_t count = 0;
};

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

llvm::Expected<std::optional<std::uint64_t>>
estimateRuntimePicoseconds(const frontend::StructuredProgramCandidate &program,
                           const fabric::FinalizedFabricRoot &fabricRoot) {
  const StaticWorkload workload = projectStaticWorkload(program.module());

  frontend::FabricCapabilityIndex capabilities(fabricRoot.view());
  std::uint64_t spatialPressure = 0;
  if (workload.hasSpatialRegion) {
    auto dataflowProgram =
        lowering::lowerStructuredProgramToCanonicalDataflow(program);
    if (!dataflowProgram)
      return dataflowProgram.takeError();
    auto view = dataflowProgram->view();
    if (!view)
      return view.takeError();

    std::map<std::vector<std::uint8_t>, ActorDemand> actorDemands;
    for (const dataflow::CanonicalActorView &actor : view->actors()) {
      auto projection =
          dataflow::projectRegisteredActorSchemaProjectionBytes(actor.op);
      if (!projection)
        return projection.takeError();
      std::vector<std::uint8_t> key(projection->bytes().begin(),
                                    projection->bytes().end());
      ActorDemand &demand = actorDemands[key];
      if (!demand.representative)
        demand.representative = actor.op;
      ++demand.count;
    }

    for (const auto &[key, demand] : actorDemands) {
      (void)key;
      llvm::Expected<std::uint64_t> capacity =
          dataflow::isCanonicalDataflowActor(
              demand.representative,
              dataflow::CanonicalDataflowActorKind::Memory)
              ? capabilities.admittingMemoryResourceCount(demand.representative)
              : capabilities.admittingOperationResourceCount(
                    demand.representative);
      if (!capacity)
        return capacity.takeError();
      if (*capacity == 0)
        return std::optional<std::uint64_t>{};
      const std::uint64_t pressure =
          demand.count / *capacity + (demand.count % *capacity != 0 ? 1 : 0);
      spatialPressure = std::max(spatialPressure, pressure);
    }
  }

  const std::optional<std::uint64_t> instructionTime = llvm::checkedMulUnsigned(
      workload.instructionLeaves, kInstructionLeafPicoseconds);
  const std::optional<std::uint64_t> spatialTime =
      llvm::checkedMulUnsigned(spatialPressure, kSpatialPressurePicoseconds);
  if (!instructionTime || !spatialTime)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "structured_fabric_model_overflow: static pressure product");
  const std::optional<std::uint64_t> total =
      llvm::checkedAddUnsigned(*instructionTime, *spatialTime);
  if (!total || *total > static_cast<std::uint64_t>(
                             std::numeric_limits<std::int64_t>::max()))
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "structured_fabric_model_overflow: Runtime estimate");
  return std::optional<std::uint64_t>(*total);
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

  auto decimal = DecimalValue::get(static_cast<std::int64_t>(**runtime), -12);
  if (!decimal)
    return decimal.takeError();
  std::vector<MetricResult> metricResults;
  metricResults.reserve(request.metricRequests().size());
  for (const MetricRequest &metric : request.metricRequests()) {
    if (metric.query().metric != MetricKind::Runtime)
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "structured_fabric_model_invalid: unsupported metric reached "
          "provider");
    metricResults.push_back({UncertaintyKind::Unknown,
                             PointObservation{MetricValue{*decimal}},
                             {}});
  }
  return EvaluationModelResult{{},
                               CompletedEvidence{std::move(metricResults), {}}};
}

const EvaluationModelProvider kProvider{kModelDescriptor.reference(),
                                        &evaluate};

llvm::Expected<std::vector<CaseArtifactResolution::Entry>>
resolveFabricClosure(const ArtifactRootReference &reference,
                     const ArtifactStore &artifactStore) {
  std::vector<CaseArtifactResolution::Entry> entries;
  std::map<ArtifactRootReference, std::vector<ArtifactRootReference>,
           decltype(&artifactRootReferenceLess)>
      resolved(&artifactRootReferenceLess);

  std::function<llvm::Error(const ArtifactRootReference &)> visit =
      [&](const ArtifactRootReference &current) -> llvm::Error {
    if (resolved.count(current) != 0)
      return llvm::Error::success();
    auto root = fabric::importEntireFabricRoot(current, artifactStore);
    if (!root)
      return root.takeError();
    std::vector<ArtifactRootReference> closure;
    for (const fabric::FabricDirectDependency &dependency :
         root->directDependencies()) {
      if (llvm::Error error = visit(dependency.root))
        return error;
      closure.push_back(dependency.root);
      const auto &nested = resolved.find(dependency.root)->second;
      closure.insert(closure.end(), nested.begin(), nested.end());
    }
    std::sort(closure.begin(), closure.end(), artifactRootReferenceLess);
    closure.erase(std::unique(closure.begin(), closure.end()), closure.end());
    resolved.emplace(current, std::move(closure));
    return llvm::Error::success();
  };

  if (llvm::Error error = visit(reference))
    return std::move(error);
  entries.reserve(resolved.size());
  for (auto &[artifact, closure] : resolved)
    entries.push_back({artifact, std::move(closure)});
  return entries;
}

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

  auto fabricEntries = resolveFabricClosure(fabricReference, artifactStore);
  if (!fabricEntries)
    return fabricEntries.takeError();
  fabricEntries->push_back({structuredProgram, {}});
  auto resolution = CaseArtifactResolution::get(std::move(*fabricEntries));
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
  return PreparedStructuredFabricEvaluation{std::move(*request),
                                            std::move(*resolution)};
}

} // namespace loom::evaluation::models
